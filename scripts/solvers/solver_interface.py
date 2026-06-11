"""
Solver Interface and Result Classes for Benchmark System.

This module defines the abstract base class for all solvers and the standardized
result format that all solvers must return. This ensures consistent behavior
and data format across all solver implementations.

The standardized result format includes 8 required fields:
- solve_time: Time taken to solve the problem (float, seconds)
- status: Solution status (string)
- primal_objective_value: Primal objective value (float or None)
- dual_objective_value: Dual objective value (float or None)
- duality_gap: Gap between primal and dual objectives (float or None)
- primal_infeasibility: Primal constraint violation (float or None)
- dual_infeasibility: Dual constraint violation (float or None)
- iterations: Number of solver iterations (int or None)
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loaders.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("solver_interface")


@dataclass
class SolverResult:
    """
    Standardized result format returned by all solvers.
    
    This class ensures consistent data format across all solver implementations.
    All 8 fields are required for database insertion and result analysis.
    """

    # Required timing information (may be None if not applicable)
    solve_time: Optional[float]  # Time taken to solve the problem (seconds)

    # Required solution status
    status: str  # Solution status (e.g., "OPTIMAL", "INFEASIBLE", "UNBOUNDED", "ERROR")

    # Required objective values (may be None if not available)
    primal_objective_value: Optional[float]  # Primal objective value
    dual_objective_value: Optional[float]    # Dual objective value

    # Required optimality measures (may be None if not available)
    duality_gap: Optional[float]         # Gap between primal and dual objectives
    primal_infeasibility: Optional[float]  # Primal constraint violation measure
    dual_infeasibility: Optional[float]    # Dual constraint violation measure

    # Required iteration count (may be None if not available)
    iterations: Optional[int]  # Number of solver iterations

    # Optional additional information
    solver_name: Optional[str] = None      # Name of the solver used
    solver_version: Optional[str] = None   # Version of the solver
    additional_info: Optional[Dict[str, Any]] = None  # Any additional solver-specific information

    def __post_init__(self):
        """Validate the result data after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate that the result contains all required fields with proper types.
        
        Raises:
            ValueError: If any required field is missing or has invalid type
        """
        # Validate solve_time (can be None)
        if self.solve_time is not None:
            if not isinstance(self.solve_time, (int, float)):
                raise ValueError(f"solve_time must be numeric or None, got {type(self.solve_time)}")
            if self.solve_time < 0:
                raise ValueError(f"solve_time must be non-negative, got {self.solve_time}")

        # Validate status
        if not isinstance(self.status, str):
            raise ValueError(f"status must be string, got {type(self.status)}")
        if not self.status.strip():
            raise ValueError("status cannot be empty")

        # Validate numeric fields (can be None)
        numeric_fields = [
            'primal_objective_value', 'dual_objective_value',
            'duality_gap', 'primal_infeasibility', 'dual_infeasibility'
        ]

        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric or None, got {type(value)}")

        # Validate iterations
        if self.iterations is not None and not isinstance(self.iterations, int):
            raise ValueError(f"iterations must be int or None, got {type(self.iterations)}")
        if self.iterations is not None and self.iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {self.iterations}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format for database storage.
        
        Returns:
            Dictionary with all result fields
        """
        return {
            'solve_time': self.solve_time,
            'status': self.status,
            'primal_objective_value': self.primal_objective_value,
            'dual_objective_value': self.dual_objective_value,
            'duality_gap': self.duality_gap,
            'primal_infeasibility': self.primal_infeasibility,
            'dual_infeasibility': self.dual_infeasibility,
            'iterations': self.iterations,
            'solver_name': self.solver_name,
            'solver_version': self.solver_version,
            'additional_info': self.additional_info
        }

    @classmethod
    def create_error_result(cls, error_message: str, solve_time: float = 0.0,
                          solver_name: str = "unknown", solver_version: str = "unknown") -> 'SolverResult':
        """
        Create a standardized error result.
        
        Args:
            error_message: Description of the error
            solve_time: Time spent before error occurred
            solver_name: Name of the solver that failed
            solver_version: Version of the solver that failed
            
        Returns:
            SolverResult indicating error status
        """
        return cls(
            solve_time=solve_time,
            status="ERROR",
            primal_objective_value=None,
            dual_objective_value=None,
            duality_gap=None,
            primal_infeasibility=None,
            dual_infeasibility=None,
            iterations=None,
            solver_name=solver_name,
            solver_version=solver_version,
            additional_info={"error_message": error_message}
        )

    @classmethod
    def create_timeout_result(cls, timeout_duration: float, solver_name: str = "unknown",
                            solver_version: str = "unknown") -> 'SolverResult':
        """
        Create a standardized timeout result.
        
        Args:
            timeout_duration: Time limit that was exceeded
            solver_name: Name of the solver that timed out
            solver_version: Version of the solver that timed out
            
        Returns:
            SolverResult indicating timeout status
        """
        return cls(
            solve_time=timeout_duration,
            status="TIMEOUT",
            primal_objective_value=None,
            dual_objective_value=None,
            duality_gap=None,
            primal_infeasibility=None,
            dual_infeasibility=None,
            iterations=None,
            solver_name=solver_name,
            solver_version=solver_version,
            additional_info={"timeout_duration": timeout_duration}
        )

    @classmethod
    def create_subprocess_error_result(cls, returncode: int, error_message: str, solve_time: float = 0.0,
                                     solver_name: str = "unknown", solver_version: str = "unknown") -> 'SolverResult':
        """
        Create a standardized result for subprocess execution errors.
        
        Args:
            returncode: Process return code
            error_message: Error message from subprocess
            solve_time: Time spent before error occurred
            solver_name: Name of the solver that failed
            solver_version: Version of the solver that failed
            
        Returns:
            SolverResult indicating subprocess error
        """
        return cls(
            solve_time=solve_time,
            status="SUBPROCESS_ERROR",
            primal_objective_value=None,
            dual_objective_value=None,
            duality_gap=None,
            primal_infeasibility=None,
            dual_infeasibility=None,
            iterations=None,
            solver_name=solver_name,
            solver_version=solver_version,
            additional_info={
                'returncode': returncode,
                'error_type': 'SUBPROCESS_ERROR',
                'error_message': error_message
            }
        )

    @classmethod
    def create_sigkill_result(cls, memory_limit_gb: Optional[float] = None, solve_time: float = 0.0,
                            solver_name: str = "unknown", solver_version: str = "unknown",
                            error_details: str = "") -> 'SolverResult':
        """
        Create a standardized SIGKILL result (process was forcibly terminated).
        
        Args:
            memory_limit_gb: Memory limit if known (may be None if killed for other reasons)
            solve_time: Time spent before SIGKILL occurred
            solver_name: Name of the solver that was killed
            solver_version: Version of the solver that was killed
            error_details: Additional error details
            
        Returns:
            SolverResult indicating SIGKILL status
        """
        additional_info = {
            'error_type': 'SIGKILL',
            'error_details': error_details
        }
        if memory_limit_gb is not None:
            additional_info['memory_limit_gb'] = memory_limit_gb

        return cls(
            solve_time=solve_time,
            status="SIGKILL",
            primal_objective_value=None,
            dual_objective_value=None,
            duality_gap=None,
            primal_infeasibility=None,
            dual_infeasibility=None,
            iterations=None,
            solver_name=solver_name,
            solver_version=solver_version,
            additional_info=additional_info
        )

    @classmethod
    def create_unsupported_result(cls, problem_type: str, solver_name: str = "unknown",
                                solver_version: str = "unknown") -> 'SolverResult':
        """
        Create a standardized result for unsupported problem types.
        
        Args:
            problem_type: Type of problem that is not supported (LP, QP, SOCP, SDP)
            solver_name: Name of the solver
            solver_version: Version of the solver
            
        Returns:
            SolverResult indicating unsupported problem type
        """
        return cls(
            solve_time=None,
            status="UNSUPPORTED",
            primal_objective_value=None,
            dual_objective_value=None,
            duality_gap=None,
            primal_infeasibility=None,
            dual_infeasibility=None,
            iterations=None,
            solver_name=solver_name,
            solver_version=solver_version,
            additional_info={
                "reason": f"Solver does not support {problem_type} problems",
                "problem_type": problem_type
            }
        )


class SolverInterface(ABC):
    """
    Abstract base class for all optimization solvers.
    
    All solver implementations must inherit from this class and implement
    required methods. This ensures consistent behavior across all solvers.
    """

    def __init__(self, solver_name: str, **kwargs):
        """
        Initialize the solver interface.
        
        Args:
            solver_name: Name of the solver
            **kwargs: Solver-specific configuration parameters
        """
        self.solver_name = solver_name
        self.config = kwargs
        self.logger = get_logger(f"solver_{solver_name}")

    @abstractmethod
    def solve(self, problem_data: ProblemData, timeout: Optional[float] = None) -> SolverResult:
        """
        Solve the optimization problem and return standardized result.
        
        Args:
            problem_data: Problem data in unified format
            timeout: Optional timeout in seconds. If specified, solver execution will be 
                    terminated after this duration and a TIMEOUT result will be returned.
                    Different solver backends may handle timeout differently - some support 
                    native timeout parameters while others use manual detection. If None,
                    solver will run without time limit (not recommended for automated systems).
            
        Returns:
            SolverResult with standardized fields
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement solve() method")

    @abstractmethod
    def get_version(self) -> str:
        """
        Get the version of the solver.
        
        Returns:
            Version string of the solver
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_version() method")

    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get information about the solver configuration.
        
        Returns:
            Dictionary with solver information
        """
        return {
            'solver_name': self.solver_name,
            'version': self.get_version(),
            'config': self.config
        }


    def validate_problem_compatibility(self, problem_data: ProblemData) -> bool:
        """
        Check if the solver can handle the given problem type.
        
        Args:
            problem_data: Problem data to validate
            
        Returns:
            True if solver can handle the problem, False otherwise
        """
        # Default implementation - subclasses should override for specific checks
        return True


