"""
Python Solver Interface Module for Benchmark System.

This module provides a unified interface for managing Python optimization solvers,
creating symmetry with the MATLAB interface module. It handles Python-specific
solver creation, backend detection, and compatibility validation.

Key Features:
- Unified Python solver management (CVXPY backends + SciPy)
- Dynamic backend availability detection
- Solver compatibility validation
- Version detection and tracking
- Consistent interface with MATLAB solver management

Architecture:
This module extracts Python solver management logic from the benchmark runner
to create a clean separation between orchestration and solver-specific logic.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.solver_interface import SolverInterface, SolverResult
from scripts.solvers.python.scipy_runner import ScipySolver
from scripts.solvers.python.cvxpy_runner import CvxpySolver
from scripts.data_loaders.problem_loader import ProblemData
from scripts.data_loaders.python.problem_interface import ProblemInterface
from scripts.utils.logger import get_logger

logger = get_logger("python_interface")


class PythonInterface:
    """
    Interface for managing Python solver ecosystem.
    
    This class provides centralized management of Python-based optimization solvers,
    handling backend detection, solver creation, and compatibility validation.
    Creates architectural symmetry with the MATLAB interface module.
    """
    
    # Available Python solver configurations
    PYTHON_SOLVER_CONFIGS = {
        "scipy_linprog": {
            "class": ScipySolver,
            "display_name": "SciPy linprog",
            "kwargs": {}
        },
        "cvxpy_clarabel": {
            "class": CvxpySolver,
            "display_name": "CLARABEL (via CVXPY)",
            "kwargs": {"backend": "CLARABEL"}
        },
        "cvxpy_scs": {
            "class": CvxpySolver,
            "display_name": "SCS (via CVXPY)",
            "kwargs": {"backend": "SCS"}
        },
        "cvxpy_ecos": {
            "class": CvxpySolver,
            "display_name": "ECOS (via CVXPY)",
            "kwargs": {"backend": "ECOS"}
        },
        "cvxpy_osqp": {
            "class": CvxpySolver,
            "display_name": "OSQP (via CVXPY)",
            "kwargs": {"backend": "OSQP"}
        },
        "cvxpy_cvxopt": {
            "class": CvxpySolver,
            "display_name": "CVXOPT (via CVXPY)",
            "kwargs": {"backend": "CVXOPT"}
        },
        "cvxpy_sdpa": {
            "class": CvxpySolver,
            "display_name": "SDPA (via CVXPY)",
            "kwargs": {"backend": "SDPA"}
        },
        "cvxpy_scip": {
            "class": CvxpySolver,
            "display_name": "SCIP (via CVXPY)",
            "kwargs": {"backend": "SCIP"}
        },
        "cvxpy_highs": {
            "class": CvxpySolver,
            "display_name": "HiGHS (via CVXPY)",
            "kwargs": {"backend": "HIGHS"}
        }
    }
    
    def __init__(self, save_solutions: bool = False, problem_interface: Optional[ProblemInterface] = None, **kwargs):
        """
        Initialize Python solver interface.
        
        Args:
            save_solutions: Whether to save optimal solutions to disk
            problem_interface: Optional problem interface for loading problems
            **kwargs: Additional configuration parameters
        """
        self.save_solutions = save_solutions
        self.config = kwargs
        
        # Initialize or create problem interface
        self.problem_interface = problem_interface or ProblemInterface()
        
        # Lazy initialization - solvers detected only when needed
        self._available_solvers = None
        
        logger.info("Initialized Python interface (lazy solver detection)")
    
    def create_solver(self, solver_name: str) -> SolverInterface:
        """
        Create Python solver instance based on solver name.
        
        Args:
            solver_name: Name of solver to create (e.g., 'cvxpy_clarabel', 'scipy_linprog')
            
        Returns:
            Solver instance implementing SolverInterface
            
        Raises:
            ValueError: If solver name is unknown or cannot be created
        """
        logger.debug(f"Creating Python solver: {solver_name}")
        
        # Check if solver is known
        if solver_name not in self.PYTHON_SOLVER_CONFIGS:
            raise ValueError(f"'{solver_name}' is not a Python solver")
        
        # Get solver configuration
        solver_config = self.PYTHON_SOLVER_CONFIGS[solver_name]
        solver_class = solver_config["class"]
        solver_kwargs = solver_config["kwargs"].copy()
        
        # Add common parameters
        solver_kwargs["save_solutions"] = self.save_solutions
        solver_kwargs.update(self.config)
        
        # Try to create solver instance (EAFP approach)
        try:
            solver = solver_class(**solver_kwargs)
            logger.debug(f"Successfully created {solver_name}: {solver.get_version()}")
            return solver
        except Exception as e:
            # Include backend info in error message if available
            backend = solver_config.get("kwargs", {}).get("backend", "")
            backend_msg = f" (backend: {backend})" if backend else ""
            raise ValueError(f"Failed to create solver '{solver_name}'{backend_msg}: {e}")
    
    def get_available_solvers(self) -> List[str]:
        """
        Get list of available Python solvers.
        
        Returns:
            List of solver names that can be created successfully
        """
        # Lazy detection - only detect when explicitly requested
        if self._available_solvers is None:
            self._available_solvers = self._detect_available_solvers()
            logger.info(f"Detected {len(self._available_solvers)} available Python solvers on first access")
            logger.debug(f"Available Python solvers: {self._available_solvers}")
        return self._available_solvers.copy()
    
    def solve(self, problem_name: str, solver_name: str,
             problem_data: Optional[ProblemData] = None,
             timeout: Optional[float] = None) -> SolverResult:
        """
        Unified solve method that handles problem loading and solver execution.
        
        This method provides a consistent interface matching the MATLAB implementation,
        enabling symmetrical architecture across all solver ecosystems.
        
        Args:
            problem_name: Name of the problem to solve
            solver_name: Name of the solver to use
            problem_data: Optional pre-loaded problem data (if None, will load)
            timeout: Optional timeout for solver execution
            
        Returns:
            SolverResult with standardized fields
            
        Raises:
            ValueError: If solver not available or problem cannot be loaded
        """
        logger.info(f"Solving {problem_name} with {solver_name}")
        
        try:
            # 1. Create solver instance (will raise ValueError if not a Python solver)
            solver = self.create_solver(solver_name)
            
            # 2. Load problem data if not provided
            if problem_data is None:
                logger.debug(f"Loading problem data for {problem_name}")
                problem_data = self.problem_interface.load_problem(problem_name)
            
            # 3. Validate compatibility
            if not solver.validate_problem_compatibility(problem_data):
                problem_type = problem_data.problem_class
                return SolverResult.create_error_result(
                    f"Solver {solver_name} cannot handle {problem_type} problems",
                    solve_time=0.0,
                    solver_name=solver_name,
                    solver_version=solver.get_version()
                )
            
            # 4. Execute solver
            result = solver.solve(problem_data, timeout=timeout)
            
            # 5. Ensure solver metadata is set
            if not result.solver_name:
                result.solver_name = solver_name
            if not result.solver_version:
                result.solver_version = solver.get_version()
            
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
    
    def get_solver_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Python solver availability.
        
        Returns:
            Dictionary with solver statistics
        """
        total_solvers = len(self.PYTHON_SOLVER_CONFIGS)
        
        # For statistics, we need to actually detect solvers
        available_solvers_list = self.get_available_solvers()
        available_solvers = len(available_solvers_list)
        
        # Group by type
        scipy_solvers = [name for name in available_solvers_list if name.startswith("scipy_")]
        cvxpy_solvers = [name for name in available_solvers_list if name.startswith("cvxpy_")]
        
        return {
            "total_configured": total_solvers,
            "total_available": available_solvers,
            "availability_rate": available_solvers / total_solvers if total_solvers > 0 else 0,
            "scipy_solvers": len(scipy_solvers),
            "cvxpy_solvers": len(cvxpy_solvers),
            "cvxpy_backends": [name.split("_", 1)[1] for name in cvxpy_solvers],
            "unavailable_solvers": list(set(self.PYTHON_SOLVER_CONFIGS.keys()) - set(available_solvers_list))
        }
    
    def _detect_available_solvers(self) -> List[str]:
        """
        Detect which Python solvers are available in the current environment.
        
        Returns:
            List of available solver names
        """
        available = []
        
        logger.debug("Detecting available Python solvers...")
        
        for solver_name, config in self.PYTHON_SOLVER_CONFIGS.items():
            try:
                # Attempt to create solver instance
                solver_class = config["class"]
                solver_kwargs = config["kwargs"].copy()
                
                # Add minimal parameters for testing
                solver_kwargs["save_solutions"] = False
                
                # Try to create solver
                solver = solver_class(**solver_kwargs)
                
                # Test basic functionality (version detection)
                version = solver.get_version()
                
                available.append(solver_name)
                logger.debug(f"✓ {solver_name}: {version}")
                
            except Exception as e:
                logger.debug(f"✗ {solver_name}: {e}")
                continue
        
        logger.info(f"Detected {len(available)}/{len(self.PYTHON_SOLVER_CONFIGS)} available Python solvers")
        
        return available
    
