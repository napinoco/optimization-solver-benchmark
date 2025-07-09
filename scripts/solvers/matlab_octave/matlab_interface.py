"""
Unified MATLAB Solver Interface for Benchmark System.

This module provides a unified interface for MATLAB solvers that matches the
architecture of the Python interface, enabling consistent solver management
across different solver ecosystems.

Key Features:
- Unified solve method matching Python interface signature
- Centralized MATLAB solver management
- Consistent error handling and logging
- Symmetrical architecture with Python interface
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.solver_interface import SolverResult
from scripts.data_loaders.problem_loader import ProblemData
from scripts.data_loaders.python.problem_interface import ProblemInterface
from scripts.solvers.matlab_octave.matlab_runner import MatlabSolver, SeDuMiSolver, SDPT3Solver
from scripts.utils.logger import get_logger

logger = get_logger("matlab_interface")


class MatlabInterface:
    """
    Unified interface for managing MATLAB solver ecosystem.
    
    This class provides centralized management of MATLAB-based optimization solvers,
    matching the architecture of PythonInterface for consistency across solver ecosystems.
    """
    
    # Available MATLAB solver configurations
    MATLAB_SOLVER_CONFIGS = {
        "matlab_sedumi": {
            "class": SeDuMiSolver,
            "display_name": "SeDuMi (MATLAB)",
            "matlab_solver": "sedumi"
        },
        "matlab_sdpt3": {
            "class": SDPT3Solver,
            "display_name": "SDPT3 (MATLAB)",
            "matlab_solver": "sdpt3"
        }
    }
    
    def __init__(self, save_solutions: bool = False, 
                 problem_interface: Optional[ProblemInterface] = None,
                 matlab_executable: str = 'matlab',
                 use_octave: bool = False,
                 timeout: Optional[float] = 300,
                 **kwargs):
        """
        Initialize MATLAB solver interface.
        
        Args:
            save_solutions: Whether to save optimal solutions to disk
            problem_interface: Optional problem interface for loading problems
            matlab_executable: Path to MATLAB/Octave executable
            use_octave: Use Octave instead of MATLAB
            timeout: Default timeout for solver execution
            **kwargs: Additional configuration parameters
        """
        self.save_solutions = save_solutions
        self.matlab_executable = matlab_executable
        self.use_octave = use_octave
        self.default_timeout = timeout
        self.config = kwargs
        
        # Initialize or create problem interface
        self.problem_interface = problem_interface or ProblemInterface()
        
        # Lazy initialization - solvers detected only when needed
        self._available_solvers = None
        
        logger.info(f"Initialized MATLAB interface (lazy solver detection)")
        logger.debug(f"Using {'Octave' if use_octave else 'MATLAB'} at: {matlab_executable}")
    
    def solve(self, problem_name: str, solver_name: str,
             problem_data: Optional[ProblemData] = None,
             timeout: Optional[float] = None) -> SolverResult:
        """
        Unified solve method that handles problem loading and solver execution.
        
        This method provides a consistent interface matching the Python implementation,
        enabling symmetrical architecture across all solver ecosystems.
        
        Args:
            problem_name: Name of the problem to solve
            solver_name: Name of the solver to use (e.g., 'matlab_sedumi')
            problem_data: Optional pre-loaded problem data (if None, will load)
            timeout: Optional timeout for solver execution
            
        Returns:
            SolverResult with standardized fields
            
        Raises:
            ValueError: If solver not available or problem cannot be loaded
        """
        logger.info(f"Solving {problem_name} with {solver_name}")
        
        try:
            # 1. Create solver instance (will raise ValueError if not a MATLAB solver)
            solver = self.create_solver(solver_name)
            
            # 2. Load problem data if not provided
            if problem_data is None:
                logger.debug(f"Loading problem data for {problem_name}")
                problem_data = self.problem_interface.load_problem(problem_name)
            
            # 3. Ensure problem data has the name attribute for MATLAB resolution
            if not hasattr(problem_data, 'name') or problem_data.name != problem_name:
                problem_data.name = problem_name
            
            # 4. Execute solver (MatlabSolver.solve handles all the complexity)
            result = solver.solve(problem_data, timeout=timeout or self.default_timeout)
            
            # 5. Ensure solver metadata is set consistently
            if not result.solver_name:
                result.solver_name = solver_name
            
            logger.info(f"Completed {solver_name} on {problem_name}: {result.status}")
            return result
            
        except ValueError:
            # Re-raise ValueError so EAFP pattern in runner can catch it
            raise
        except Exception as e:
            error_msg = f"Failed to solve {problem_name} with {solver_name}: {str(e)}"
            logger.error(error_msg)
            
            # Try to get solver version for error result
            solver_version = "unknown"
            try:
                if solver_name in self.MATLAB_SOLVER_CONFIGS:
                    temp_solver = self.create_solver(solver_name)
                    solver_version = temp_solver.get_version()
            except:
                pass
            
            return SolverResult.create_error_result(
                error_msg,
                solve_time=0.0,
                solver_name=solver_name,
                solver_version=solver_version
            )
    
    def create_solver(self, solver_name: str) -> MatlabSolver:
        """
        Create MATLAB solver instance based on solver name.
        
        Args:
            solver_name: Name of solver to create (e.g., 'matlab_sedumi')
            
        Returns:
            MatlabSolver instance
            
        Raises:
            ValueError: If solver name is unknown
        """
        logger.debug(f"Creating MATLAB solver: {solver_name}")
        
        # Check if solver is known
        if solver_name not in self.MATLAB_SOLVER_CONFIGS:
            raise ValueError(f"'{solver_name}' is not a MATLAB solver")
        
        # Get solver configuration
        solver_config = self.MATLAB_SOLVER_CONFIGS[solver_name]
        solver_class = solver_config["class"]
        
        # Create solver with unified parameters
        solver_kwargs = {
            "matlab_executable": self.matlab_executable,
            "use_octave": self.use_octave,
            "save_solutions": self.save_solutions,
            "timeout": self.default_timeout
        }
        solver_kwargs.update(self.config)
        
        # Try to create solver instance (EAFP approach)
        try:
            solver = solver_class(**solver_kwargs)
            logger.debug(f"Successfully created {solver_name}")
            return solver
        except Exception as e:
            raise ValueError(f"Failed to create solver '{solver_name}': {e}")
    
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

    def get_solver_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about MATLAB solver availability.
        
        Returns:
            Dictionary with solver statistics
        """
        total_solvers = len(self.MATLAB_SOLVER_CONFIGS)
        
        # For statistics, we need to actually detect solvers
        available_solvers_list = self.get_available_solvers()
        available_solvers = len(available_solvers_list)
        
        return {
            "total_configured": total_solvers,
            "total_available": available_solvers,
            "availability_rate": available_solvers / total_solvers if total_solvers > 0 else 0,
            "execution_environment": "octave" if self.use_octave else "matlab",
            "matlab_executable": self.matlab_executable,
            "unavailable_solvers": list(set(self.MATLAB_SOLVER_CONFIGS.keys()) - set(available_solvers_list))
        }
    
    def _detect_available_solvers(self) -> List[str]:
        """
        Detect which MATLAB solvers are available in the current environment.
        
        Returns:
            List of available solver names
        """
        available = []
        
        logger.debug("Detecting available MATLAB solvers...")
        
        # First check if MATLAB/Octave is available at all
        try:
            # Try to create a simple test solver
            test_solver = MatlabSolver(
                matlab_solver="sedumi",
                matlab_executable=self.matlab_executable,
                use_octave=self.use_octave,
                timeout=30
            )
            logger.debug("MATLAB/Octave environment verified")
        except Exception as e:
            logger.warning(f"MATLAB/Octave not available: {e}")
            return []  # No MATLAB solvers available
        
        # Now check each configured solver
        for solver_name, config in self.MATLAB_SOLVER_CONFIGS.items():
            try:
                # Try to create solver instance
                solver = self.create_solver(solver_name)
                
                # Test basic functionality (version detection)
                version = solver.get_version()
                
                available.append(solver_name)
                logger.debug(f"✓ {solver_name}: {version}")
                
            except Exception as e:
                logger.debug(f"✗ {solver_name}: {e}")
                continue
        
        logger.info(f"Detected {len(available)}/{len(self.MATLAB_SOLVER_CONFIGS)} available MATLAB solvers")
        
        return available
    
