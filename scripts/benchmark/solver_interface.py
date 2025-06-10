import time
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger
from scripts.benchmark.problem_loader import ProblemData

logger = get_logger("solver_interface")

@dataclass
class SolverResult:
    """Standardized result from solver execution."""
    solver_name: str
    problem_name: str
    solve_time: float  # seconds
    status: str  # 'optimal', 'infeasible', 'unbounded', 'timeout', 'error'
    objective_value: Optional[float] = None
    duality_gap: Optional[float] = None
    iterations: Optional[int] = None
    error_message: Optional[str] = None
    solver_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate result data."""
        if self.solve_time < 0:
            raise ValueError("Solve time cannot be negative")
        
        valid_statuses = ['optimal', 'infeasible', 'unbounded', 'timeout', 'error', 'unknown']
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of {valid_statuses}")
    
    def is_successful(self) -> bool:
        """Check if the solver found an optimal solution."""
        return self.status == 'optimal'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for database storage."""
        return {
            'solver_name': self.solver_name,
            'problem_name': self.problem_name,
            'solve_time': self.solve_time,
            'status': self.status,
            'objective_value': self.objective_value,
            'duality_gap': self.duality_gap,
            'iterations': self.iterations,
            'error_message': self.error_message,
            'solver_info': self.solver_info
        }

class TimeoutError(Exception):
    """Exception raised when solver execution times out."""
    pass

class SolverInterface(ABC):
    """Abstract base class for optimization solvers."""
    
    def __init__(self, name: str, timeout: float = 300.0):
        """
        Initialize solver interface.
        
        Args:
            name: Name of the solver
            timeout: Maximum time allowed for solving (seconds)
        """
        self.name = name
        self.timeout = timeout
        self.logger = get_logger(f"solver.{name}")
    
    @abstractmethod
    def solve(self, problem: ProblemData) -> SolverResult:
        """
        Solve the optimization problem.
        
        Args:
            problem: Problem data to solve
            
        Returns:
            SolverResult containing solve status and results
        """
        pass
    
    def solve_with_timeout(self, problem: ProblemData) -> SolverResult:
        """
        Solve problem with timeout handling.
        
        Args:
            problem: Problem data to solve
            
        Returns:
            SolverResult with timeout handling
        """
        self.logger.info(f"Starting solve for problem '{problem.name}' with timeout {self.timeout}s")
        
        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Solver timed out after {self.timeout} seconds")
        
        # Store original handler
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        
        try:
            # Set alarm for timeout
            signal.alarm(int(self.timeout))
            
            # Record start time
            start_time = time.time()
            
            # Execute solve
            result = self.solve(problem)
            
            # Cancel alarm
            signal.alarm(0)
            
            # Update actual solve time
            actual_time = time.time() - start_time
            result.solve_time = actual_time
            
            self.logger.info(f"Solve completed in {actual_time:.3f}s with status '{result.status}'")
            return result
            
        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            elapsed_time = time.time() - start_time
            self.logger.warning(f"Solver timed out after {elapsed_time:.3f}s")
            
            return SolverResult(
                solver_name=self.name,
                problem_name=problem.name,
                solve_time=elapsed_time,
                status='timeout',
                error_message=f"Solver timed out after {self.timeout} seconds"
            )
            
        except Exception as e:
            signal.alarm(0)  # Cancel alarm
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Solver error after {elapsed_time:.3f}s: {error_msg}")
            
            return SolverResult(
                solver_name=self.name,
                problem_name=problem.name,
                solve_time=elapsed_time,
                status='error',
                error_message=error_msg
            )
            
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGALRM, original_handler)
    
    def get_version(self) -> str:
        """Get solver version information."""
        return "unknown"
    
    def get_info(self) -> Dict[str, Any]:
        """Get solver information."""
        return {
            "name": self.name,
            "version": self.get_version(),
            "timeout": self.timeout
        }

class MockSolver(SolverInterface):
    """Mock solver for testing the interface."""
    
    def __init__(self, name: str = "MockSolver", timeout: float = 300.0, 
                 simulate_time: float = 0.1, simulate_status: str = 'optimal'):
        """
        Initialize mock solver.
        
        Args:
            name: Solver name
            timeout: Timeout in seconds
            simulate_time: Time to simulate solving (seconds)
            simulate_status: Status to return
        """
        super().__init__(name, timeout)
        self.simulate_time = simulate_time
        self.simulate_status = simulate_status
    
    def solve(self, problem: ProblemData) -> SolverResult:
        """Mock solve implementation."""
        self.logger.debug(f"Mock solving problem '{problem.name}'")
        
        # Simulate solving time
        time.sleep(self.simulate_time)
        
        # Create mock result based on problem type
        objective_value = None
        if self.simulate_status == 'optimal':
            if problem.problem_class == 'LP':
                # Mock LP optimal value
                objective_value = 10.5
            elif problem.problem_class == 'QP':
                # Mock QP optimal value
                objective_value = 2.3
        
        return SolverResult(
            solver_name=self.name,
            problem_name=problem.name,
            solve_time=self.simulate_time,
            status=self.simulate_status,
            objective_value=objective_value,
            iterations=10,
            solver_info={"mock": True}
        )
    
    def get_version(self) -> str:
        """Get mock solver version."""
        return "1.0.0-mock"

if __name__ == "__main__":
    # Test script to verify solver interface
    try:
        print("Testing Solver Interface...\n")
        
        # Import problem loader for testing
        from scripts.benchmark.problem_loader import load_problem
        
        # Test SolverResult validation
        print("Testing SolverResult validation:")
        
        # Valid result
        try:
            result = SolverResult(
                solver_name="TestSolver",
                problem_name="test_problem",
                solve_time=1.5,
                status="optimal",
                objective_value=10.0
            )
            print(f"✓ Valid result created: {result}")
        except Exception as e:
            print(f"✗ Failed to create valid result: {e}")
        
        # Invalid status
        try:
            invalid_result = SolverResult(
                solver_name="TestSolver",
                problem_name="test_problem",
                solve_time=1.5,
                status="invalid_status"
            )
            print("✗ Should have failed with invalid status")
        except ValueError as e:
            print(f"✓ Correctly caught invalid status: {e}")
        
        # Test MockSolver
        print("\nTesting MockSolver:")
        mock_solver = MockSolver(simulate_time=0.01)
        
        # Load a test problem
        problem = load_problem("simple_lp")
        print(f"Loaded problem: {problem.name}")
        
        # Test normal solve
        result = mock_solver.solve_with_timeout(problem)
        print(f"Mock solve result: {result}")
        print(f"Result dict: {result.to_dict()}")
        print(f"Is successful: {result.is_successful()}")
        
        # Test timeout simulation
        print("\nTesting timeout:")
        timeout_solver = MockSolver(simulate_time=0.01, timeout=0.005)  # Very short timeout
        timeout_result = timeout_solver.solve_with_timeout(problem)
        print(f"Timeout result: {timeout_result}")
        
        # Test error simulation
        print("\nTesting error handling:")
        class ErrorSolver(SolverInterface):
            def solve(self, problem):
                raise RuntimeError("Simulated solver error")
        
        error_solver = ErrorSolver("ErrorSolver", timeout=1.0)
        error_result = error_solver.solve_with_timeout(problem)
        print(f"Error result: {error_result}")
        
        print("\n✓ All interface tests passed!")
        
    except Exception as e:
        logger.error(f"Interface test failed: {e}")
        raise