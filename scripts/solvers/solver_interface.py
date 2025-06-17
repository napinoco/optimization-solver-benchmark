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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import time
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("solver_interface")


@dataclass
class SolverResult:
    """
    Standardized result format returned by all solvers.
    
    This class ensures consistent data format across all solver implementations.
    All 8 fields are required for database insertion and result analysis.
    """
    
    # Required timing information
    solve_time: float  # Time taken to solve the problem (seconds)
    
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
        # Validate solve_time
        if not isinstance(self.solve_time, (int, float)):
            raise ValueError(f"solve_time must be numeric, got {type(self.solve_time)}")
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
    def create_error_result(cls, error_message: str, solve_time: float = 0.0) -> 'SolverResult':
        """
        Create a standardized error result.
        
        Args:
            error_message: Description of the error
            solve_time: Time spent before error occurred
            
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
            additional_info={"error_message": error_message}
        )
    
    @classmethod
    def create_timeout_result(cls, timeout_duration: float) -> 'SolverResult':
        """
        Create a standardized timeout result.
        
        Args:
            timeout_duration: Time limit that was exceeded
            
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
            additional_info={"timeout_duration": timeout_duration}
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
            timeout: Optional timeout in seconds
            
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
    
    def solve_with_timing(self, problem_data: ProblemData, timeout: Optional[float] = None) -> SolverResult:
        """
        Solve problem with automatic timing measurement.
        
        This is a convenience method that wraps the solve() method with timing.
        Subclasses can override this if they need custom timing logic.
        
        Args:
            problem_data: Problem data in unified format
            timeout: Optional timeout in seconds
            
        Returns:
            SolverResult with accurate timing information
        """
        start_time = time.time()
        
        try:
            result = self.solve(problem_data, timeout)
            
            # Ensure the result has the actual solve time
            actual_solve_time = time.time() - start_time
            result.solve_time = actual_solve_time
            result.solver_name = self.solver_name
            result.solver_version = self.get_version()
            
            return result
            
        except Exception as e:
            solve_time = time.time() - start_time
            self.logger.error(f"Solver {self.solver_name} failed: {e}")
            return SolverResult.create_error_result(str(e), solve_time)
    
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
    
    def preprocess_problem(self, problem_data: ProblemData) -> ProblemData:
        """
        Preprocess problem data before solving (optional).
        
        Args:
            problem_data: Original problem data
            
        Returns:
            Preprocessed problem data (default: returns original)
        """
        # Default implementation - no preprocessing
        return problem_data


# Utility functions for result analysis
def calculate_duality_gap(primal_value: Optional[float], dual_value: Optional[float]) -> Optional[float]:
    """
    Calculate the duality gap between primal and dual objective values.
    
    Args:
        primal_value: Primal objective value
        dual_value: Dual objective value
        
    Returns:
        Absolute duality gap or None if either value is missing
    """
    if primal_value is None or dual_value is None:
        return None
    return abs(primal_value - dual_value)


def is_optimal_solution(result: SolverResult, tolerance: float = 1e-6) -> bool:
    """
    Check if the result represents an optimal solution within tolerance.
    
    Args:
        result: Solver result to check
        tolerance: Tolerance for optimality checks
        
    Returns:
        True if result indicates optimal solution
    """
    # Check status
    if result.status not in ["OPTIMAL", "OPTIMAL_INACCURATE"]:
        return False
    
    # Check duality gap if available
    if result.duality_gap is not None and result.duality_gap > tolerance:
        return False
    
    # Check infeasibilities if available
    if result.primal_infeasibility is not None and result.primal_infeasibility > tolerance:
        return False
    
    if result.dual_infeasibility is not None and result.dual_infeasibility > tolerance:
        return False
    
    return True


if __name__ == "__main__":
    # Test the SolverResult class
    print("Testing SolverResult class...")
    
    # Test valid result
    try:
        result = SolverResult(
            solve_time=1.234,
            status="OPTIMAL",
            primal_objective_value=5.67,
            dual_objective_value=5.67,
            duality_gap=0.0,
            primal_infeasibility=1e-8,
            dual_infeasibility=1e-8,
            iterations=42
        )
        print("✓ Valid result created successfully")
        print(f"  Result: {result}")
        print(f"  Is optimal: {is_optimal_solution(result)}")
        
    except Exception as e:
        print(f"✗ Failed to create valid result: {e}")
    
    # Test error result
    try:
        error_result = SolverResult.create_error_result("Test error message", 0.5)
        print("✓ Error result created successfully")
        print(f"  Error result: {error_result}")
        
    except Exception as e:
        print(f"✗ Failed to create error result: {e}")
    
    # Test validation
    try:
        invalid_result = SolverResult(
            solve_time=-1.0,  # Invalid negative time
            status="OPTIMAL",
            primal_objective_value=None,
            dual_objective_value=None,
            duality_gap=None,
            primal_infeasibility=None,
            dual_infeasibility=None,
            iterations=None
        )
        print("✗ Should have failed validation")
        
    except ValueError as e:
        print(f"✓ Validation correctly caught error: {e}")
    
    print("\n✓ SolverResult test completed!")