"""
MATLAB Solver Process Interface for Benchmark System.

This module provides a subprocess-based interface to MATLAB solvers that calls 
matlab_solver_runner.m, providing symmetrical architecture with the Python 
process interface.

Key Features:
- Direct MATLAB solver execution via subprocess
- Memory limitation via ulimit (Unix-like systems)
- Unified solve method matching Python process interface signature
- Centralized MATLAB solver management
- Consistent error handling and logging
- Symmetrical architecture with Python process interface
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.solver_interface import SolverResult
from scripts.data_loaders.problem_loader import ProblemData
from scripts.data_loaders.python.problem_interface import ProblemInterface
from scripts.utils.temp_file_manager import temp_file_context
from scripts.utils.logger import get_logger

logger = get_logger("matlab_process_interface")


class MatlabProcessInterface:
    """
    Unified interface for managing MATLAB solver ecosystem.
    
    This class provides centralized management of MATLAB-based optimization solvers,
    calling matlab_interface.m directly for clean architecture.
    """
    
    # Available MATLAB solver configurations
    MATLAB_SOLVER_CONFIGS = {
        "matlab_sedumi": {
            "display_name": "SeDuMi (MATLAB)",
            "matlab_solver": "sedumi",
            "runner_function": "sedumi_runner"
        },
        "matlab_sdpt3": {
            "display_name": "SDPT3 (MATLAB)",
            "matlab_solver": "sdpt3",
            "runner_function": "sdpt3_runner"
        }
    }
    
    def __init__(self, save_solutions: bool = False, 
                 problem_interface: Optional[ProblemInterface] = None,
                 matlab_executable: str = 'matlab',
                 timeout: Optional[float] = 300,
                 **kwargs):
        """
        Initialize MATLAB solver process interface.
        
        Args:
            save_solutions: Whether to save optimal solutions to disk
            problem_interface: Optional problem interface for loading problems
            matlab_executable: Path to MATLAB executable
            timeout: Default timeout for solver execution
            **kwargs: Additional configuration parameters
        """
        self.save_solutions = save_solutions
        self.matlab_executable = matlab_executable
        self.default_timeout = timeout
        self.config = kwargs
        
        # Initialize or create problem interface
        self.problem_interface = problem_interface or ProblemInterface()
        
        # Lazy initialization - solvers detected only when needed
        self._available_solvers = None
        
        logger.info(f"Initialized MATLAB process interface (subprocess isolation)")
        logger.debug(f"Using MATLAB at: {matlab_executable}")
    
    def solve(self, problem_name: str, solver_name: str,
             problem_data: Optional[ProblemData] = None,
             timeout: Optional[float] = None) -> SolverResult:
        """
        Unified solve method that calls matlab_solver_runner.m via subprocess.
        
        Args:
            problem_name: Name of the problem to solve
            solver_name: Name of the solver to use (e.g., 'matlab_sedumi')
            problem_data: Optional pre-loaded problem data (ignored - MATLAB loads directly)
            timeout: Optional timeout for solver execution
            
        Returns:
            SolverResult with standardized fields
            
        Raises:
            ValueError: If solver not available or problem cannot be loaded
        """
        logger.info(f"Solving {problem_name} with {solver_name}")
        
        try:
            # 1. Validate solver name
            if solver_name not in self.MATLAB_SOLVER_CONFIGS:
                raise ValueError(f"'{solver_name}' is not a MATLAB solver")
            
            # 2. Get solver configuration
            solver_config = self.MATLAB_SOLVER_CONFIGS[solver_name]
            matlab_solver = solver_config["matlab_solver"]
            runner_function = solver_config["runner_function"]
            
            # 3. Use provided timeout or default
            actual_timeout = timeout or self.default_timeout
            
            # 4. Call MATLAB solver via subprocess
            result = self._call_matlab_interface(
                problem_name=problem_name,
                matlab_solver=matlab_solver,
                runner_function=runner_function,
                timeout=actual_timeout
            )
            
            # 4. Ensure solver metadata is set
            if not result.solver_name:
                result.solver_name = solver_name
                
            # 5. Add problem class information to additional_info for database storage
            try:
                from scripts.data_loaders.python.problem_interface import ProblemInterface
                problem_interface = ProblemInterface()
                problem_data = problem_interface.load_problem(problem_name)
                if not result.additional_info:
                    result.additional_info = {}
                result.additional_info['problem_class'] = problem_data.problem_class
            except Exception as e:
                logger.debug(f"Could not get problem class for {problem_name}: {e}")
                if not result.additional_info:
                    result.additional_info = {}
                result.additional_info['problem_class'] = 'UNKNOWN'
            
            logger.info(f"Completed {solver_name} on {problem_name}: {result.status}")
            return result
            
        except ValueError:
            # Re-raise ValueError so EAFP pattern in runner can catch it
            raise
        except Exception as e:
            error_msg = f"Failed to solve {problem_name} with {solver_name}: {str(e)}"
            logger.error(error_msg)
            
            return SolverResult.create_error_result(
                error_msg,
                solve_time=0.0,
                solver_name=solver_name,
                solver_version="unknown"
            )
    
    def _call_matlab_interface(self, problem_name: str, matlab_solver: str, 
                             runner_function: str, timeout: float) -> SolverResult:
        """
        Call matlab_solver_runner.m via subprocess.
        
        Args:
            problem_name: Name of the problem
            matlab_solver: MATLAB solver name ('sedumi' or 'sdpt3')
            runner_function: Runner function name
            timeout: Timeout in seconds
            
        Returns:
            SolverResult from MATLAB execution
        """
        start_time = time.time()
        
        # Use context manager for automatic temp file cleanup
        try:
            with temp_file_context(".json") as result_file:
                logger.debug(f"Using temporary result file: {result_file}")
                
                # Construct MATLAB command
                matlab_script_dir = Path(__file__).parent.absolute()
                
                # Escape single quotes in arguments
                safe_problem_name = problem_name.replace("'", "''")
                safe_solver_name = matlab_solver.replace("'", "''")
                safe_result_file = result_file.replace("'", "''")
                safe_runner_function = runner_function.replace("'", "''")
                
                # Create MATLAB command (updated to call matlab_solver_runner)
                matlab_command = (
                    f"addpath('{matlab_script_dir}'); "
                    f"matlab_solver_runner('{safe_problem_name}', '{safe_solver_name}', "
                    f"'{safe_result_file}', {str(self.save_solutions).lower()}, '{safe_runner_function}')"
                )
                
                # Build command array
                # Use minimal options, rely on environment variables for Java/X11 control
                cmd = [self.matlab_executable, '-batch', matlab_command]
                
                logger.debug(f"Executing MATLAB command: {' '.join(cmd)}")
                
                # Execute with timeout
                try:
                    # Add startup buffer to timeout for MATLAB initialization
                    adjusted_timeout = timeout + 15  # Extra 15s for MATLAB startup
                    
                    # Set environment variables to suppress Java warnings and X11 issues
                    env = os.environ.copy()
                    env['DISPLAY'] = ''  # Disable X11
                    env['_JAVA_OPTIONS'] = '-Djava.awt.headless=true'  # Headless Java mode
                    env['MATLAB_LOG_DIR'] = '/dev/null'  # Suppress MATLAB logs
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=adjusted_timeout,
                        cwd=project_root,
                        env=env
                    )
                    
                    # Check execution success
                    if result.returncode != 0:
                        error_msg = self._parse_matlab_error(result.stderr, result.stdout)
                        solve_time = time.time() - start_time
                        
                        # Check for SIGKILL (process forcibly terminated)
                        if result.returncode == -9 or result.returncode == 137:
                            logger.error(f"Process killed by SIGKILL, returncode: {result.returncode}")
                            return SolverResult.create_sigkill_result(
                                memory_limit_gb=None,
                                solve_time=solve_time,
                                solver_name=f"matlab_{matlab_solver}",
                                solver_version="unknown",
                                error_details=f"Process terminated (returncode {result.returncode}). {error_msg}"
                            )
                        # Other subprocess errors
                        else:
                            full_error = f"MATLAB subprocess failed (code {result.returncode}): {error_msg}"
                            logger.error(full_error)
                            
                            return SolverResult.create_subprocess_error_result(
                                returncode=result.returncode,
                                error_message=error_msg,
                                solve_time=solve_time,
                                solver_name=f"matlab_{matlab_solver}",
                                solver_version="unknown"
                            )
                    
                    # Read JSON result file
                    if not os.path.exists(result_file):
                        return SolverResult.create_error_result(
                            "MATLAB interface did not produce result file",
                            solve_time=float('nan'),
                            solver_name=f"matlab_{matlab_solver}",
                            solver_version="unknown"
                        )
                    
                    # Check if result file has content
                    try:
                        file_stat = os.stat(result_file)
                        if file_stat.st_size == 0:
                            return SolverResult.create_error_result(
                                "MATLAB interface produced empty result file",
                                solve_time=float('nan'),
                                solver_name=f"matlab_{matlab_solver}",
                                solver_version="unknown"
                            )
                    except OSError as e:
                        return SolverResult.create_error_result(
                            f"Error accessing result file: {e}",
                            solve_time=float('nan'),
                            solver_name=f"matlab_{matlab_solver}",
                            solver_version="unknown"
                        )
                    
                    # Read and parse JSON result
                    try:
                        with open(result_file, 'r') as f:
                            matlab_result = json.load(f)
                    except json.JSONDecodeError as e:
                        return SolverResult.create_error_result(
                            f"Invalid JSON in result file: {e}",
                            solve_time=float('nan'),
                            solver_name=f"matlab_{matlab_solver}",
                            solver_version="unknown"
                        )
                    except IOError as e:
                        return SolverResult.create_error_result(
                            f"Error reading result file: {e}",
                            solve_time=float('nan'),
                            solver_name=f"matlab_{matlab_solver}",
                            solver_version="unknown"
                        )
                    
                    # Convert MATLAB result to SolverResult
                    return self._convert_matlab_result(matlab_result, matlab_solver)
                    
                except subprocess.TimeoutExpired:
                    return SolverResult.create_timeout_result(
                        timeout,
                        solver_name=f"matlab_{matlab_solver}",
                        solver_version="unknown"
                    )
                    
        except Exception as e:
            logger.error(f"MATLAB interface execution failed: {e}")
            return SolverResult.create_error_result(
                str(e),
                solve_time=float('nan'),
                solver_name=f"matlab_{matlab_solver}",
                solver_version="unknown"
            )
    
    def _convert_matlab_result(self, matlab_result: Dict[str, Any], matlab_solver: str) -> SolverResult:
        """Convert MATLAB JSON result to SolverResult format."""
        
        # Extract solver version information
        solver_version = matlab_result.get('solver_version', f"{matlab_solver.upper()} (version unknown)")
        matlab_version = matlab_result.get('matlab_version', 'unknown')
        combined_version = f"{solver_version} (MATLAB {matlab_version})"
        
        # Handle None/null values from JSON
        def safe_float(value):
            return None if value is None or value == [] else float(value)
        
        def safe_int(value):
            return None if value is None or value == [] else int(value)
        
        try:
            # Extract solve_time from MATLAB result, prioritizing actual solver execution time
            matlab_solve_time = safe_float(matlab_result.get('solve_time'))
            
            # Use MATLAB's solve_time if valid, otherwise set to NaN (don't use subprocess time)
            if matlab_solve_time is not None and matlab_solve_time >= 0:
                final_solve_time = matlab_solve_time
            else:
                # Don't mask errors by using subprocess time - set to NaN for transparency
                final_solve_time = float('nan')
                self.logger.warning(f"Invalid solve_time from MATLAB result: {matlab_solve_time}, setting to NaN")
            
            return SolverResult(
                solve_time=final_solve_time,
                status=matlab_result.get('status', 'unknown').upper(),
                primal_objective_value=safe_float(matlab_result.get('primal_objective_value')),
                dual_objective_value=safe_float(matlab_result.get('dual_objective_value')),
                duality_gap=safe_float(matlab_result.get('duality_gap')),
                primal_infeasibility=safe_float(matlab_result.get('primal_infeasibility')),
                dual_infeasibility=safe_float(matlab_result.get('dual_infeasibility')),
                iterations=safe_int(matlab_result.get('iterations')),
                solver_name=f"matlab_{matlab_solver}",
                solver_version=combined_version,
                additional_info={
                    'matlab_output': matlab_result,
                    'matlab_version': matlab_version,
                    'solver_backend': matlab_solver,
                    'execution_environment': 'matlab',
                    # Preserve original solver status codes for reproducibility
                    'original_status': matlab_result.get('original_status', {})
                }
            )
        except Exception as e:
            # If conversion fails, return error result with NaN solve_time (don't use subprocess time)
            return SolverResult.create_error_result(
                f"Failed to convert MATLAB result: {e}",
                solve_time=float('nan'),
                solver_name=f"matlab_{matlab_solver}",
                solver_version=combined_version
            )
    
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
    
    def create_solver(self, solver_name: str):
        """
        Create solver method for backward compatibility.
        
        Note: This is kept for interface compatibility but the preferred
        approach is to use the solve method directly.
        """
        # Validate solver name
        if solver_name not in self.MATLAB_SOLVER_CONFIGS:
            raise ValueError(f"'{solver_name}' is not a MATLAB solver")
        
        # Return a simple object that can provide version info
        class MatlabSolverStub:
            def __init__(self, solver_name, config):
                self.solver_name = solver_name
                self.config = config
            
            def get_version(self):
                return f"{self.config['display_name']} (direct interface)"
        
        return MatlabSolverStub(solver_name, self.MATLAB_SOLVER_CONFIGS[solver_name])
    
    def get_available_solvers(self) -> List[str]:
        """
        Get list of available MATLAB solvers.
        
        Returns:
            List of solver names that can be created successfully
        """
        # Lazy detection - only detect when explicitly requested
        if self._available_solvers is None:
            self._available_solvers = self._detect_available_solvers()
            logger.info(f"Detected {len(self._available_solvers)} available MATLAB solvers on first access")
            logger.debug(f"Available MATLAB solvers: {self._available_solvers}")
        return self._available_solvers.copy()
    
    def _detect_available_solvers(self) -> List[str]:
        """
        Detect which MATLAB solvers are available in the current environment.
        
        Returns:
            List of available solver names
        """
        available = []
        
        logger.debug("Detecting available MATLAB solvers...")
        
        # First check if MATLAB is available
        try:
            cmd = [self.matlab_executable, '-batch', 'disp("MATLAB_OK")']
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd=project_root
            )
            
            if result.returncode != 0:
                logger.warning(f"MATLAB not available: {result.stderr}")
                return []  # No MATLAB solvers available
            
            logger.debug("MATLAB environment verified")
        except Exception as e:
            logger.warning(f"MATLAB not available: {e}")
            return []  # No MATLAB solvers available
        
        # If MATLAB is available, assume all configured solvers are available
        # (actual solver availability is checked during execution)
        for solver_name in self.MATLAB_SOLVER_CONFIGS.keys():
            available.append(solver_name)
            logger.debug(f"✓ {solver_name}: Available (direct interface)")
        
        logger.info(f"Detected {len(available)}/{len(self.MATLAB_SOLVER_CONFIGS)} available MATLAB solvers")
        
        return available