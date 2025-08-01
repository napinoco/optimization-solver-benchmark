"""
Python Solver Process Interface for Benchmark System.

This module provides a subprocess-based interface for Python optimization solvers,
creating symmetry with the MATLAB process interface. It executes Python solvers
in isolated subprocesses with resource limitations for improved stability.

Key Features:
- Subprocess isolation for each solver execution
- Memory limitation via ulimit (Unix-like systems)
- Timeout control for solver execution
- Unified interface matching MATLAB solver management
- Protection against solver crashes affecting main process
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
from scripts.utils.resource_limits import build_resource_limited_command, format_memory_limit_display
from scripts.utils.environment_info import get_memory_info
from scripts.utils.logger import get_logger

logger = get_logger("python_process_interface")


class PythonProcessInterface:
    """
    Process-based interface for managing Python solver execution.
    
    This class provides subprocess isolation for Python-based optimization solvers,
    matching the architecture of MatlabProcessInterface for consistency.
    Each solver execution runs in a separate process with configurable resource limits.
    """
    
    # Python solver configurations (delegated to python_solver_runner.py)
    PYTHON_SOLVER_CONFIGS = {
        'cvxpy_clarabel': {'display_name': 'CLARABEL (CVXPY)'},
        'cvxpy_scs': {'display_name': 'SCS (CVXPY)'},
        'cvxpy_ecos': {'display_name': 'ECOS (CVXPY)'},
        'cvxpy_osqp': {'display_name': 'OSQP (CVXPY)'},
        'cvxpy_cvxopt': {'display_name': 'CVXOPT (CVXPY)'},
        'cvxpy_sdpa': {'display_name': 'SDPA (CVXPY)'},
        'cvxpy_scip': {'display_name': 'SCIP (CVXPY)'},
        'cvxpy_highs': {'display_name': 'HIGHS (CVXPY)'},
        'scipy_linprog': {'display_name': 'LINPROG (SciPy)'},
    }
    
    def __init__(self, save_solutions: bool = False,
                 problem_interface: Optional[ProblemInterface] = None,
                 python_executable: str = sys.executable,
                 memory_limit_gb: Optional[float] = None,
                 timeout: Optional[float] = 300,
                 **kwargs):
        """
        Initialize Python process interface.
        
        Args:
            save_solutions: Whether to save optimal solutions to disk
            problem_interface: Optional problem interface for loading problems
            python_executable: Path to Python executable
            memory_limit_gb: Memory limit for solver processes in GB (None for auto-detection based on system memory)
            timeout: Default timeout for solver execution
            **kwargs: Additional configuration parameters
        """
        self.save_solutions = save_solutions
        self.python_executable = python_executable
        
        # Set memory limit: use provided value or auto-detect based on system memory
        if memory_limit_gb is None:
            try:
                memory_info = get_memory_info()
                system_memory_gb = memory_info['total_gb']
                self.default_memory_limit = system_memory_gb * 0.95  # Use 95% of system memory
                logger.info(f"Auto-detected Python memory limit: {format_memory_limit_display(self.default_memory_limit)} (95% of {system_memory_gb:.1f}GB system memory)")
            except Exception as e:
                self.default_memory_limit = 8.0  # Fallback
                logger.warning(f"Failed to detect system memory, using fallback: {format_memory_limit_display(self.default_memory_limit)} - {e}")
        else:
            self.default_memory_limit = memory_limit_gb
            logger.info(f"Using provided Python memory limit: {format_memory_limit_display(self.default_memory_limit)}")
        
        self.default_timeout = timeout
        self.config = kwargs
        
        # Initialize or create problem interface
        self.problem_interface = problem_interface or ProblemInterface()
        
        # Lazy initialization - solvers detected only when needed
        self._available_solvers = None
        
        logger.info(f"Initialized Python process interface (subprocess isolation)")
        logger.debug(f"Python executable: {python_executable}")
        logger.debug(f"Default memory limit: {format_memory_limit_display(memory_limit_gb)}")
    
    def solve(self, problem_name: str, solver_name: str,
              problem_data: Optional[ProblemData] = None,
              timeout: Optional[float] = None,
              memory_limit_gb: Optional[float] = None) -> SolverResult:
        """
        Solve optimization problem using subprocess isolation.
        
        Args:
            problem_name: Name of the problem to solve
            solver_name: Name of the solver to use (e.g., 'cvxpy_clarabel')
            problem_data: Optional pre-loaded problem data (ignored - subprocess loads directly)
            timeout: Optional timeout for solver execution
            memory_limit_gb: Optional memory limit override
            
        Returns:
            SolverResult with standardized fields
            
        Raises:
            ValueError: If solver not available
        """
        logger.info(f"Solving {problem_name} with {solver_name} (subprocess)")
        
        try:
            # 1. Validate solver name
            if solver_name not in self.PYTHON_SOLVER_CONFIGS:
                raise ValueError(f"'{solver_name}' is not a Python solver")
            
            # 2. Use provided limits or defaults
            actual_timeout = timeout or self.default_timeout
            actual_memory_limit = memory_limit_gb or self.default_memory_limit
            
            # 3. Execute solver in subprocess
            result = self._call_python_solver(
                problem_name=problem_name,
                solver_name=solver_name,
                timeout=actual_timeout,
                memory_limit_gb=actual_memory_limit
            )
            
            # 4. Ensure solver metadata is set
            if not result.solver_name:
                result.solver_name = solver_name
            
            # 5. Add problem class information for database storage
            try:
                problem_data = self.problem_interface.load_problem(problem_name)
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
    
    def _call_python_solver(self, problem_name: str, solver_name: str,
                           timeout: float, memory_limit_gb: float) -> SolverResult:
        """
        Execute Python solver in subprocess with resource limits.
        
        Args:
            problem_name: Name of the problem
            solver_name: Python solver name
            timeout: Timeout in seconds
            memory_limit_gb: Memory limit in GB
            
        Returns:
            SolverResult from solver execution
        """
        start_time = time.time()
        
        # Use context manager for automatic temp file cleanup
        try:
            with temp_file_context(".json") as result_file:
                logger.debug(f"Using temporary result file: {result_file}")
                
                # Construct Python solver runner command
                solver_runner_path = Path(__file__).parent / "python_solver_runner.py"
                
                cmd = [
                    self.python_executable,
                    str(solver_runner_path),
                    '--problem', problem_name,
                    '--solver', solver_name,
                    '--result-file', result_file
                ]
                
                if self.save_solutions:
                    cmd.append('--save-solutions')
                
                # Apply resource limits using utility
                cmd = build_resource_limited_command(cmd, memory_limit_gb)
                
                logger.debug(f"Executing Python solver command: {' '.join(cmd)}")
                logger.debug(f"Memory limit: {format_memory_limit_display(memory_limit_gb)}")
                
                # Execute with timeout
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=project_root
                    )
                    
                    # Check execution success
                    if result.returncode != 0:
                        error_msg = self._parse_python_error(result.stderr, result.stdout)
                        solve_time = time.time() - start_time
                        
                        # Check for SIGKILL (process forcibly terminated)
                        if result.returncode == -9 or result.returncode == 137:
                            logger.error(f"Process killed by SIGKILL, returncode: {result.returncode}")
                            # Include memory limit info since we set ulimit
                            return SolverResult.create_sigkill_result(
                                memory_limit_gb=memory_limit_gb,
                                solve_time=solve_time,
                                solver_name=solver_name,
                                solver_version="unknown",
                                error_details=f"Process terminated (returncode {result.returncode}). {error_msg}"
                            )
                        # Check for Python MemoryError (caught by Python, not SIGKILL)
                        elif "MemoryError" in error_msg or "cannot allocate memory" in error_msg:
                            logger.error(f"Python MemoryError detected")
                            # Still a subprocess error but with memory-related details
                            return SolverResult.create_subprocess_error_result(
                                returncode=result.returncode,
                                error_message=f"Python memory allocation failed: {error_msg}",
                                solve_time=solve_time,
                                solver_name=solver_name,
                                solver_version="unknown"
                            )
                        # Other subprocess errors
                        else:
                            full_error = f"Python subprocess failed (code {result.returncode}): {error_msg}"
                            logger.error(full_error)
                            
                            # Use subprocess error result
                            return SolverResult.create_subprocess_error_result(
                                returncode=result.returncode,
                                error_message=error_msg,
                                solve_time=solve_time,
                                solver_name=solver_name,
                                solver_version="unknown"
                            )
                    
                    # Read JSON result file
                    if not os.path.exists(result_file):
                        return SolverResult.create_error_result(
                            "Python solver did not produce result file",
                            solve_time=time.time() - start_time,
                            solver_name=solver_name,
                            solver_version="unknown"
                        )
                    
                    # Read and parse JSON result
                    try:
                        with open(result_file, 'r') as f:
                            solver_result_dict = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        return SolverResult.create_error_result(
                            f"Error reading result file: {e}",
                            solve_time=time.time() - start_time,
                            solver_name=solver_name,
                            solver_version="unknown"
                        )
                    
                    # Convert dict to SolverResult
                    return self._dict_to_solver_result(solver_result_dict, solver_name)
                    
                except subprocess.TimeoutExpired:
                    return SolverResult.create_timeout_result(
                        timeout,
                        solver_name=solver_name,
                        solver_version="unknown"
                    )
                    
        except Exception as e:
            error_msg = f"Unexpected error in Python solver subprocess: {str(e)}"
            logger.error(error_msg)
            return SolverResult.create_error_result(
                error_msg,
                solve_time=time.time() - start_time,
                solver_name=solver_name,
                solver_version="unknown"
            )
    
    def _parse_python_error(self, stderr: str, stdout: str) -> str:
        """Parse error messages from Python subprocess output."""
        # Look for specific error patterns
        error_lines = []
        
        # Check stderr first
        if stderr:
            lines = stderr.strip().split('\n')
            # Look for Python exception messages
            for i, line in enumerate(lines):
                if 'Traceback' in line:
                    # Include traceback and error message
                    error_lines.extend(lines[i:])
                    break
                elif 'Error:' in line or 'error:' in line:
                    error_lines.append(line)
        
        # If no clear error in stderr, check stdout
        if not error_lines and stdout:
            lines = stdout.strip().split('\n')
            for line in lines:
                if 'Error:' in line or 'error:' in line:
                    error_lines.append(line)
        
        # Return parsed error or raw stderr
        if error_lines:
            return '\n'.join(error_lines[-10:])  # Last 10 lines to avoid huge errors
        else:
            return stderr.strip() or stdout.strip() or "Unknown error"
    
    def _dict_to_solver_result(self, result_dict: Dict[str, Any], solver_name: str) -> SolverResult:
        """Convert dictionary from subprocess to SolverResult object."""
        # Handle both direct fields and nested 'additional_info'
        return SolverResult(
            solve_time=result_dict.get('solve_time', 0.0),
            status=result_dict.get('status', 'error'),
            primal_objective_value=result_dict.get('primal_objective_value'),
            dual_objective_value=result_dict.get('dual_objective_value'),
            duality_gap=result_dict.get('duality_gap'),
            primal_infeasibility=result_dict.get('primal_infeasibility'),
            dual_infeasibility=result_dict.get('dual_infeasibility'),
            iterations=result_dict.get('iterations'),
            solver_name=result_dict.get('solver_name', solver_name),
            solver_version=result_dict.get('solver_version', 'unknown'),
            additional_info=result_dict.get('additional_info', {})
        )
    
    def get_available_solvers(self) -> List[str]:
        """
        Get list of available Python solvers.
        
        This returns all configured solvers - actual availability is checked
        during execution by the subprocess.
        
        Returns:
            List of solver names
        """
        return list(self.PYTHON_SOLVER_CONFIGS.keys())
    
    def create_solver(self, solver_name: str) -> Any:
        """
        Create solver instance (not used in subprocess mode).
        
        This method exists for interface compatibility but is not used
        since solvers are created in the subprocess.
        
        Raises:
            NotImplementedError: Always raised as subprocess mode doesn't create local instances
        """
        raise NotImplementedError("Process interface does not create local solver instances")
    
    def is_solver_available(self, solver_name: str) -> bool:
        """Check if solver is configured (actual availability checked in subprocess)."""
        return solver_name in self.PYTHON_SOLVER_CONFIGS