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

from scripts.solvers.solver_interface import SolverInterface
from scripts.solvers.python.scipy_runner import ScipySolver
from scripts.solvers.python.cvxpy_runner import CvxpySolver
from scripts.data_loaders.problem_loader import ProblemData
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
    
    def __init__(self, save_solutions: bool = False, **kwargs):
        """
        Initialize Python solver interface.
        
        Args:
            save_solutions: Whether to save optimal solutions to disk
            **kwargs: Additional configuration parameters
        """
        self.save_solutions = save_solutions
        self.config = kwargs
        
        # Detect available solvers on initialization
        self.available_solvers = self._detect_available_solvers()
        
        logger.info(f"Initialized Python interface with {len(self.available_solvers)} available solvers")
        logger.debug(f"Available Python solvers: {self.available_solvers}")
    
    def create_solver(self, solver_name: str) -> SolverInterface:
        """
        Create Python solver instance based on solver name.
        
        Args:
            solver_name: Name of solver to create (e.g., 'cvxpy_clarabel', 'scipy_linprog')
            
        Returns:
            Solver instance implementing SolverInterface
            
        Raises:
            ValueError: If solver name is unknown or solver not available
        """
        logger.debug(f"Creating Python solver: {solver_name}")
        
        # Check if solver is known
        if solver_name not in self.PYTHON_SOLVER_CONFIGS:
            available_names = list(self.PYTHON_SOLVER_CONFIGS.keys())
            raise ValueError(f"Unknown Python solver: {solver_name}. "
                           f"Available: {available_names}")
        
        # Check if solver is available (backend installed)
        if solver_name not in self.available_solvers:
            solver_config = self.PYTHON_SOLVER_CONFIGS[solver_name]
            backend = solver_config.get("kwargs", {}).get("backend", "N/A")
            raise ValueError(f"Python solver {solver_name} not available. "
                           f"Backend {backend} may not be installed.")
        
        # Get solver configuration
        solver_config = self.PYTHON_SOLVER_CONFIGS[solver_name]
        solver_class = solver_config["class"]
        solver_kwargs = solver_config["kwargs"].copy()
        
        # Add common parameters
        solver_kwargs["save_solutions"] = self.save_solutions
        solver_kwargs.update(self.config)
        
        # Create and return solver instance
        try:
            solver = solver_class(**solver_kwargs)
            logger.debug(f"Successfully created {solver_name}: {solver.get_version()}")
            return solver
        except Exception as e:
            raise ValueError(f"Failed to create Python solver {solver_name}: {e}")
    
    def get_available_solvers(self) -> List[str]:
        """
        Get list of available Python solvers.
        
        Returns:
            List of solver names that can be created successfully
        """
        return self.available_solvers.copy()
    
    def get_solver_display_name(self, solver_name: str) -> str:
        """
        Get display name for solver.
        
        Args:
            solver_name: Internal solver name
            
        Returns:
            Human-readable display name
        """
        return self.PYTHON_SOLVER_CONFIGS.get(solver_name, {}).get("display_name", solver_name)
    
    def validate_solver_compatibility(self, solver_name: str, problem_data: ProblemData) -> bool:
        """
        Check if Python solver can handle the given problem type.
        
        Args:
            solver_name: Name of the solver
            problem_data: Problem data to validate
            
        Returns:
            True if solver can handle the problem, False otherwise
        """
        try:
            # Create solver temporarily to check compatibility
            solver = self.create_solver(solver_name)
            return solver.validate_problem_compatibility(problem_data)
        except Exception as e:
            logger.debug(f"Compatibility check failed for {solver_name}: {e}")
            return False
    
    def get_solver_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Python solver availability.
        
        Returns:
            Dictionary with solver statistics
        """
        total_solvers = len(self.PYTHON_SOLVER_CONFIGS)
        available_solvers = len(self.available_solvers)
        
        # Group by type
        scipy_solvers = [name for name in self.available_solvers if name.startswith("scipy_")]
        cvxpy_solvers = [name for name in self.available_solvers if name.startswith("cvxpy_")]
        
        return {
            "total_configured": total_solvers,
            "total_available": available_solvers,
            "availability_rate": available_solvers / total_solvers if total_solvers > 0 else 0,
            "scipy_solvers": len(scipy_solvers),
            "cvxpy_solvers": len(cvxpy_solvers),
            "cvxpy_backends": [name.split("_", 1)[1] for name in cvxpy_solvers],
            "unavailable_solvers": list(set(self.PYTHON_SOLVER_CONFIGS.keys()) - set(self.available_solvers))
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
    
    def refresh_available_solvers(self) -> List[str]:
        """
        Refresh the list of available solvers (re-detect).
        
        Returns:
            Updated list of available solver names
        """
        logger.info("Refreshing available Python solvers...")
        self.available_solvers = self._detect_available_solvers()
        return self.available_solvers
    
    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate Python solver environment and return detailed report.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating Python solver environment...")
        
        validation_report = {
            "status": "success",
            "solvers": {},
            "summary": {
                "total_solvers": len(self.PYTHON_SOLVER_CONFIGS),
                "working_solvers": 0,
                "failed_solvers": 0
            },
            "statistics": self.get_solver_statistics()
        }
        
        # Test each configured solver
        for solver_name in self.PYTHON_SOLVER_CONFIGS:
            try:
                solver = self.create_solver(solver_name)
                version = solver.get_version()
                
                validation_report["solvers"][solver_name] = {
                    "status": "working",
                    "version": version,
                    "display_name": self.get_solver_display_name(solver_name)
                }
                validation_report["summary"]["working_solvers"] += 1
                
            except Exception as e:
                validation_report["solvers"][solver_name] = {
                    "status": "error",
                    "error": str(e),
                    "display_name": self.get_solver_display_name(solver_name)
                }
                validation_report["summary"]["failed_solvers"] += 1
                validation_report["status"] = "partial" if validation_report["summary"]["working_solvers"] > 0 else "failed"
        
        # Log summary
        working = validation_report["summary"]["working_solvers"]
        total = validation_report["summary"]["total_solvers"]
        logger.info(f"Python solver validation complete: {working}/{total} working")
        
        return validation_report


def get_python_interface(save_solutions: bool = False, **kwargs) -> PythonInterface:
    """
    Factory function to get Python interface instance.
    
    Args:
        save_solutions: Whether to save optimal solutions to disk
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PythonInterface instance
    """
    return PythonInterface(save_solutions=save_solutions, **kwargs)


# Module-level convenience functions for backward compatibility
def create_python_solver(solver_name: str, save_solutions: bool = False, **kwargs) -> SolverInterface:
    """
    Convenience function to create a Python solver.
    
    Args:
        solver_name: Name of solver to create
        save_solutions: Whether to save solutions
        **kwargs: Additional configuration
        
    Returns:
        Solver instance
    """
    interface = get_python_interface(save_solutions=save_solutions, **kwargs)
    return interface.create_solver(solver_name)


def get_available_python_solvers() -> List[str]:
    """
    Convenience function to get available Python solvers.
    
    Returns:
        List of available solver names
    """
    interface = get_python_interface()
    return interface.get_available_solvers()


if __name__ == "__main__":
    # Test script for Python interface
    print("Testing Python Solver Interface...")
    print("=" * 50)
    
    # Initialize interface
    print("\n1. Initializing Python interface...")
    interface = get_python_interface()
    print(f"   ✓ Interface initialized")
    
    # Show available solvers
    available = interface.get_available_solvers()
    print(f"\n2. Available solvers: {len(available)}")
    for solver_name in available:
        display_name = interface.get_solver_display_name(solver_name)
        print(f"   • {solver_name}: {display_name}")
    
    # Show statistics
    stats = interface.get_solver_statistics()
    print(f"\n3. Statistics:")
    print(f"   Total configured: {stats['total_configured']}")
    print(f"   Available: {stats['total_available']}")
    print(f"   Availability rate: {stats['availability_rate']:.1%}")
    print(f"   SciPy solvers: {stats['scipy_solvers']}")
    print(f"   CVXPY solvers: {stats['cvxpy_solvers']}")
    
    # Test solver creation
    if available:
        print(f"\n4. Testing solver creation...")
        test_solver = available[0]
        try:
            solver = interface.create_solver(test_solver)
            version = solver.get_version()
            print(f"   ✓ Created {test_solver}: {version}")
        except Exception as e:
            print(f"   ✗ Failed to create {test_solver}: {e}")
    
    # Validation report
    print(f"\n5. Environment validation...")
    validation = interface.validate_environment()
    print(f"   Status: {validation['status']}")
    print(f"   Working: {validation['summary']['working_solvers']}")
    print(f"   Failed: {validation['summary']['failed_solvers']}")
    
    print("\n" + "=" * 50)
    print("✓ Python interface testing complete!")