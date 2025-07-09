"""
Production-Ready MATLAB Solver Integration for Benchmark System.

This module provides a complete Python interface for MATLAB optimization solvers (SeDuMi, SDPT3)
that integrates seamlessly with the benchmark system's SolverInterface architecture.

Features:
- Full SolverInterface compliance with standardized SolverResult format
- Problem registry integration for DIMACS/SDPLIB problem resolution
- Dynamic version detection and robust error handling
- Concurrent execution safety with enhanced temporary file management
- Production-ready performance optimization and monitoring

Integration Points:
- Works with existing ProblemData and problem registry system
- Compatible with BenchmarkRunner for automated benchmarking
- Supports all cone problem types (LP, QP, SOCP, SDP)
- Provides detailed diagnostics and performance metrics
"""

import os
import sys
import json
import subprocess
import time
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.solver_interface import SolverInterface, SolverResult
from scripts.data_loaders.problem_loader import ProblemData
from scripts.data_loaders.python.problem_interface import ProblemInterface
from scripts.utils.temp_file_manager import TempFileManager, temp_file_context
from scripts.utils.logger import get_logger

logger = get_logger("matlab_runner")


class MatlabSolver(SolverInterface):
    """
    Python interface for MATLAB optimization solvers with enhanced temp file management.
    
    This implementation demonstrates the temporary file management system for Task 12.
    """
    
    SUPPORTED_SOLVERS = {
        'sedumi': 'SeDuMi',
        'sdpt3': 'SDPT3'
    }
    
    def __init__(self, matlab_solver: str, matlab_executable: str = 'matlab',
                 timeout: Optional[float] = 300, use_octave: bool = False, **kwargs):
        """
        Initialize production-ready MATLAB solver interface.
        
        Args:
            matlab_solver: MATLAB solver name ('sedumi' or 'sdpt3')
            matlab_executable: Path to MATLAB executable
            timeout: Solver timeout in seconds
            use_octave: Use Octave instead of MATLAB
            **kwargs: Additional configuration parameters
        """
        if matlab_solver not in self.SUPPORTED_SOLVERS:
            raise ValueError(f"Unsupported MATLAB solver: {matlab_solver}. "
                           f"Supported: {list(self.SUPPORTED_SOLVERS.keys())}")
        
        # Generate solver name for registration
        solver_name = f"matlab_{matlab_solver}"
        
        super().__init__(solver_name, matlab_solver=matlab_solver, 
                        matlab_executable=matlab_executable, timeout=timeout, **kwargs)
        
        self.matlab_solver = matlab_solver
        self.matlab_executable = matlab_executable
        self.timeout = timeout
        self.use_octave = use_octave
        
        # Initialize temp file manager with MATLAB-specific configuration
        self.temp_manager = TempFileManager(
            base_prefix=f"matlab_{matlab_solver}_result",
            cleanup_age_hours=1  # Clean up files older than 1 hour
        )
        
        # Initialize problem interface for problem resolution
        try:
            self.problem_interface = ProblemInterface()
            problem_stats = self.problem_interface.get_problem_statistics()
            logger.debug(f"Loaded problem interface with {problem_stats['total_problems']} problems")
        except Exception as e:
            logger.warning(f"Failed to initialize problem interface: {e}")
            self.problem_interface = None
        
        # Cache for version information
        self._version_cache = None
        self._matlab_version_cache = None
        
        # Verify MATLAB/Octave availability and cache version info
        self._verify_matlab_availability()
        self._detect_solver_versions()
        
        logger.info(f"Initialized production MATLAB solver '{self.solver_name}' "
                   f"using {matlab_solver} via {matlab_executable}")
    
    def _verify_matlab_availability(self) -> None:
        """Verify that MATLAB/Octave is available and can execute."""
        logger.info(f"Verifying MATLAB availability: {self.matlab_executable}")
        
        try:
            cmd = [self.matlab_executable, '-batch', 'disp("MATLAB_OK")']
            if self.use_octave:
                cmd = [self.matlab_executable, '--eval', 'disp("Octave_OK")']
            
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=45,  # Increased timeout for MATLAB startup
                cwd=project_root
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                error_msg = self._parse_matlab_error(result.stderr, result.stdout)
                raise RuntimeError(f"MATLAB/Octave execution failed: {error_msg}")
            
            # Check for expected output
            expected_output = "MATLAB_OK" if not self.use_octave else "Octave_OK"
            if expected_output not in result.stdout:
                raise RuntimeError(f"MATLAB/Octave verification failed: unexpected output")
            
            logger.info(f"✓ MATLAB verified successfully in {execution_time:.2f}s")
            
            # Warn if startup is very slow
            if execution_time > 10:
                logger.warning(f"MATLAB startup is slow ({execution_time:.2f}s) - consider using pre-warmed sessions")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"MATLAB/Octave verification timed out after 45s - check MATLAB installation")
        except FileNotFoundError:
            raise RuntimeError(f"MATLAB/Octave executable not found: {self.matlab_executable}")
    
    def _detect_solver_versions(self) -> None:
        """Detect and cache MATLAB and solver version information."""
        logger.debug("Detecting MATLAB and solver versions...")
        
        try:
            # Construct command to get version information
            version_cmd = ("addpath(genpath('.')); "
                          "try "
                          "result = matlab_version_detection(); "
                          "disp(['VERSION_INFO:', jsonencode(result)]); "
                          "catch ME "
                          "disp(['VERSION_ERROR:', ME.message]); "
                          "end")
            
            if self.use_octave:
                cmd = [self.matlab_executable, '--eval', version_cmd]
            else:
                cmd = [self.matlab_executable, '-batch', version_cmd]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # Shorter timeout for version detection
                cwd=project_root
            )
            
            if result.returncode == 0:
                # Parse version information from output
                self._parse_version_output(result.stdout)
            else:
                logger.warning(f"Version detection failed: {result.stderr}")
                self._set_default_versions()
                
        except subprocess.TimeoutExpired:
            logger.warning("Version detection timed out, using default versions")
            self._set_default_versions()
        except Exception as e:
            logger.warning(f"Version detection error: {e}, using default versions")
            self._set_default_versions()
    
    def _parse_version_output(self, output: str) -> None:
        """Parse version information from MATLAB output."""
        try:
            for line in output.split('\n'):
                if line.startswith('VERSION_INFO:'):
                    version_json = line[len('VERSION_INFO:'):].strip()
                    version_data = json.loads(version_json)
                    
                    self._matlab_version_cache = version_data.get('matlab_version', 'unknown')
                    
                    # Extract solver-specific version
                    solver_versions = version_data.get('solver_versions', {})
                    if self.matlab_solver in solver_versions:
                        self._version_cache = solver_versions[self.matlab_solver]
                    else:
                        self._version_cache = f"{self.SUPPORTED_SOLVERS[self.matlab_solver]} (version unknown)"
                    
                    logger.debug(f"Detected versions - MATLAB: {self._matlab_version_cache}, "
                               f"{self.matlab_solver}: {self._version_cache}")
                    return
                elif line.startswith('VERSION_ERROR:'):
                    error_msg = line[len('VERSION_ERROR:'):].strip()
                    logger.warning(f"MATLAB version detection error: {error_msg}")
                    break
            
            # If no version info found, use defaults
            self._set_default_versions()
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse version JSON: {e}")
            self._set_default_versions()
    
    def _set_default_versions(self) -> None:
        """Set default version information when detection fails."""
        self._matlab_version_cache = 'unknown'
        self._version_cache = f"{self.SUPPORTED_SOLVERS[self.matlab_solver]} (version unknown)"
    
    def _parse_matlab_error(self, stderr: str, stdout: str) -> str:
        """Parse MATLAB error messages to extract meaningful information."""
        # Combine stderr and stdout for analysis
        full_output = f"{stderr}\n{stdout}".strip()
        
        # Common MATLAB error patterns
        error_patterns = [
            "Error:",
            "error:",
            "Undefined function",
            "undefined function",
            "File not found",
            "file not found",
            "License error",
            "license error",
            "Syntax error",
            "syntax error"
        ]
        
        # Extract relevant error lines
        error_lines = []
        for line in full_output.split('\n'):
            line = line.strip()
            if any(pattern in line for pattern in error_patterns):
                error_lines.append(line)
        
        if error_lines:
            return "; ".join(error_lines[:3])  # Return up to 3 most relevant error lines
        
        # If no specific error patterns found, return first few lines of output
        output_lines = [line.strip() for line in full_output.split('\n') if line.strip()]
        if output_lines:
            return "; ".join(output_lines[:2])
        
        return "Unknown MATLAB error"
    
    def _construct_matlab_command(self, problem_name: str, result_file: str) -> List[str]:
        """Construct safe MATLAB command with proper argument handling."""
        # Validate inputs to prevent issues
        if not problem_name or not isinstance(problem_name, str):
            raise ValueError("Invalid problem name")
        
        if not result_file or not isinstance(result_file, str):
            raise ValueError("Invalid result file path")
        
        # Escape single quotes in arguments by doubling them (MATLAB convention)
        safe_problem_name = problem_name.replace("'", "''")
        safe_solver_name = self.matlab_solver.replace("'", "''")
        safe_result_file = result_file.replace("'", "''")
        
        # Construct MATLAB command string
        # Get the path to MATLAB scripts
        matlab_script_dir = Path(__file__).parent.absolute()
        
        # Create MATLAB command that adds path and then runs the function
        matlab_command = f"addpath('{matlab_script_dir}'); matlab_runner('{safe_problem_name}', '{safe_solver_name}', '{safe_result_file}')"
        
        # Build command array
        if self.use_octave:
            cmd = [self.matlab_executable, '--eval', matlab_command]
        else:
            cmd = [self.matlab_executable, '-batch', matlab_command]
        
        return cmd
    
    def solve(self, problem_data: ProblemData, timeout: Optional[float] = None) -> SolverResult:
        """
        Solve optimization problem using MATLAB solver with full registry integration.
        
        Args:
            problem_data: Problem data in unified format
            timeout: Optional timeout override
            
        Returns:
            SolverResult with standardized fields
        """
        solve_timeout = timeout or self.timeout
        start_time = time.time()
        
        # Clean up old orphaned files before starting
        cleaned_count = self.temp_manager.cleanup_orphaned_files()
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} orphaned temporary files")
        
        # Use context manager for automatic temp file cleanup
        try:
            with temp_file_context(".json") as result_file:
                logger.debug(f"Using temporary result file: {result_file}")
                
                # Enhanced problem data validation
                if not self._validate_problem_data(problem_data):
                    return SolverResult.create_error_result(
                        "Problem data validation failed",
                        solve_time=time.time() - start_time,
                        solver_name=self.solver_name,
                        solver_version=self.get_version()
                    )
                
                # Resolve problem path from registry
                problem_name, problem_path = self._resolve_problem_info(problem_data)
                
                # Check solver compatibility with problem type
                if not self._check_solver_compatibility(problem_data):
                    return SolverResult.create_error_result(
                        f"Problem type not supported by {self.matlab_solver}",
                        solve_time=time.time() - start_time,
                        solver_name=self.solver_name,
                        solver_version=self.get_version()
                    )
                
                # Construct safe MATLAB command
                try:
                    cmd = self._construct_matlab_command(problem_name, result_file)
                except ValueError as e:
                    return SolverResult.create_error_result(
                        f"Invalid command parameters: {e}",
                        solve_time=time.time() - start_time,
                        solver_name=self.solver_name,
                        solver_version=self.get_version()
                    )
                
                logger.debug(f"Executing MATLAB command: {' '.join(cmd)}")
                
                # Execute with timeout and enhanced error handling
                try:
                    # Add startup buffer to timeout for MATLAB initialization
                    adjusted_timeout = solve_timeout + 15  # Extra 15s for MATLAB startup
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=adjusted_timeout,
                        cwd=project_root
                    )
                    
                    solve_time = time.time() - start_time
                    
                    # Check execution success with enhanced error parsing
                    if result.returncode != 0:
                        error_msg = self._parse_matlab_error(result.stderr, result.stdout)
                        full_error = f"MATLAB execution failed (code {result.returncode}): {error_msg}"
                        logger.error(full_error)
                        
                        # Log additional debug info
                        if result.stdout.strip():
                            logger.debug(f"MATLAB stdout: {result.stdout.strip()}")
                        if result.stderr.strip():
                            logger.debug(f"MATLAB stderr: {result.stderr.strip()}")
                        
                        return SolverResult.create_error_result(
                            full_error,
                            solve_time=solve_time,
                            solver_name=self.solver_name,
                            solver_version=self.get_version()
                        )
                    
                    # Read JSON result file with enhanced error handling
                    if not os.path.exists(result_file):
                        return SolverResult.create_error_result(
                            "MATLAB solver did not produce result file",
                            solve_time=solve_time,
                            solver_name=self.solver_name,
                            solver_version=self.get_version()
                        )
                    
                    # Check if result file has content
                    try:
                        file_stat = os.stat(result_file)
                        if file_stat.st_size == 0:
                            return SolverResult.create_error_result(
                                "MATLAB solver produced empty result file",
                                solve_time=solve_time,
                                solver_name=self.solver_name,
                                solver_version=self.get_version()
                            )
                    except OSError as e:
                        return SolverResult.create_error_result(
                            f"Error accessing result file: {e}",
                            solve_time=solve_time,
                            solver_name=self.solver_name,
                            solver_version=self.get_version()
                        )
                    
                    # Read and parse JSON result
                    try:
                        with open(result_file, 'r') as f:
                            matlab_result = json.load(f)
                    except json.JSONDecodeError as e:
                        return SolverResult.create_error_result(
                            f"Invalid JSON in result file: {e}",
                            solve_time=solve_time,
                            solver_name=self.solver_name,
                            solver_version=self.get_version()
                        )
                    except IOError as e:
                        return SolverResult.create_error_result(
                            f"Error reading result file: {e}",
                            solve_time=solve_time,
                            solver_name=self.solver_name,
                            solver_version=self.get_version()
                        )
                    
                    # Convert MATLAB result to SolverResult
                    return self._convert_matlab_result(matlab_result, solve_time)
                    
                except subprocess.TimeoutExpired:
                    return SolverResult.create_timeout_result(
                        solve_timeout,
                        solver_name=self.solver_name,
                        solver_version=self.get_version()
                    )
                    
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"MATLAB solver execution failed: {e}")
            return SolverResult.create_error_result(
                str(e),
                solve_time=solve_time,
                solver_name=self.solver_name,
                solver_version=self.get_version()
            )
    
    def _convert_matlab_result(self, matlab_result: Dict[str, Any], solve_time: float) -> SolverResult:
        """Convert MATLAB JSON result to SolverResult format with enhanced metadata."""
        
        # Extract solver version information (use cached versions if available)
        solver_version = matlab_result.get('solver_version', self._version_cache or 'unknown')
        matlab_version = matlab_result.get('matlab_version', self._matlab_version_cache or 'unknown')
        combined_version = f"{solver_version} (MATLAB {matlab_version})"
        
        # Handle None/null values from JSON
        def safe_float(value):
            return None if value is None or value == [] else float(value)
        
        def safe_int(value):
            return None if value is None or value == [] else int(value)
        
        try:
            return SolverResult(
                solve_time=solve_time,
                status=matlab_result.get('status', 'unknown').upper(),
                primal_objective_value=safe_float(matlab_result.get('primal_objective_value')),
                dual_objective_value=safe_float(matlab_result.get('dual_objective_value')),
                duality_gap=safe_float(matlab_result.get('duality_gap')),
                primal_infeasibility=safe_float(matlab_result.get('primal_infeasibility')),
                dual_infeasibility=safe_float(matlab_result.get('dual_infeasibility')),
                iterations=safe_int(matlab_result.get('iterations')),
                solver_name=self.solver_name,
                solver_version=combined_version,
                additional_info={
                    'matlab_output': matlab_result,
                    'matlab_version': matlab_version,
                    'solver_backend': self.matlab_solver,
                    'execution_environment': 'octave' if self.use_octave else 'matlab',
                    'temp_file_stats': self.temp_manager.get_temp_file_stats(),
                    'problem_interface_available': self.problem_interface is not None,
                    'version_cache_available': self._version_cache is not None
                }
            )
        except Exception as e:
            # If conversion fails, return error result
            return SolverResult.create_error_result(
                f"Failed to convert MATLAB result: {e}",
                solve_time=solve_time,
                solver_name=self.solver_name,
                solver_version=combined_version
            )
    
    def _validate_problem_data(self, problem_data: ProblemData) -> bool:
        """Validate that problem data has required fields for MATLAB integration."""
        try:
            # Check for required name field
            if not hasattr(problem_data, 'name') or not problem_data.name:
                logger.error("Problem data missing required 'name' field")
                return False
            
            # Check that problem exists in registry using problem interface
            if self.problem_interface:
                try:
                    self.problem_interface.get_problem_config(problem_data.name)
                except ValueError:
                    logger.error(f"Problem '{problem_data.name}' not found in problem registry")
                    return False
            else:
                logger.warning("Problem interface not available for validation")
            
            # Basic structure validation
            if not hasattr(problem_data, 'problem_class'):
                logger.warning("Problem data missing 'problem_class' field")
            
            return True
            
        except Exception as e:
            logger.error(f"Problem data validation error: {e}")
            return False
    
    def _resolve_problem_info(self, problem_data: ProblemData) -> tuple[str, str]:
        """Resolve problem name and path using problem interface."""
        problem_name = problem_data.name
        
        # Get problem info from problem interface
        if not self.problem_interface:
            raise ValueError("Problem interface not available")
        
        try:
            problem_info = self.problem_interface.get_problem_config(problem_name)
        except ValueError as e:
            raise ValueError(f"Problem '{problem_name}' not found in registry: {e}")
        
        # Resolve absolute path
        problem_path = str(project_root / problem_info['file_path'])
        
        logger.debug(f"Resolved problem '{problem_name}' to path: {problem_path}")
        return problem_name, problem_path
    
    def _check_solver_compatibility(self, problem_data: ProblemData) -> bool:
        """Check if solver is compatible with problem type using problem interface."""
        try:
            # Get problem info from problem interface
            if not self.problem_interface:
                logger.warning("Problem interface not available for compatibility check")
                return False
            
            problem_name = problem_data.name
            try:
                problem_info = self.problem_interface.get_problem_config(problem_name)
            except ValueError:
                logger.error(f"Problem '{problem_name}' not found for compatibility check")
                return False
            
            # Check file type compatibility
            file_type = problem_info.get('file_type', '')
            
            # MATLAB solvers support both MAT (DIMACS) and DAT-S (SDPLIB) formats
            supported_types = ['mat', 'dat-s']
            
            if file_type not in supported_types:
                logger.warning(f"Unsupported file type '{file_type}' for MATLAB solver")
                return False
            
            # Additional compatibility checks based on problem class
            problem_class = getattr(problem_data, 'problem_class', '')
            
            # Both SeDuMi and SDPT3 support LP, QP, SOCP, and SDP
            # No specific exclusions needed for these solvers
            
            return True
            
        except Exception as e:
            logger.error(f"Solver compatibility check failed: {e}")
            return False
    
    def get_version(self) -> str:
        """Get MATLAB solver version information with dynamic detection."""
        try:
            # Use cached version if available
            if self._version_cache:
                return self._version_cache
            
            # Fallback to static version info
            return f"{self.SUPPORTED_SOLVERS[self.matlab_solver]} (MATLAB)"
                
        except Exception:
            return f"{self.SUPPORTED_SOLVERS.get(self.matlab_solver, 'Unknown')} (version detection failed)"
    
    def validate_problem_compatibility(self, problem_data: ProblemData) -> bool:
        """Check if problem is compatible with MATLAB solver."""
        # Use the enhanced compatibility check
        return (self._validate_problem_data(problem_data) and 
                self._check_solver_compatibility(problem_data))
    
    def get_temp_file_stats(self) -> Dict[str, Any]:
        """Get current temporary file statistics."""
        return self.temp_manager.get_temp_file_stats()
    
    def cleanup_orphaned_files(self) -> int:
        """Clean up orphaned temporary files and return count cleaned."""
        return self.temp_manager.cleanup_orphaned_files()


class SeDuMiSolver(MatlabSolver):
    """Convenience class for SeDuMi solver."""
    
    def __init__(self, **kwargs):
        super().__init__(matlab_solver='sedumi', **kwargs)


class SDPT3Solver(MatlabSolver):
    """Convenience class for SDPT3 solver."""
    
    def __init__(self, **kwargs):
        super().__init__(matlab_solver='sdpt3', **kwargs)