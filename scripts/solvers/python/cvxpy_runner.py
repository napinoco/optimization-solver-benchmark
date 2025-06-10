import sys
import time
import numpy as np
import cvxpy as cp
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.solver_interface import SolverInterface, SolverResult
from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("cvxpy_solver")

class CvxpySolver(SolverInterface):
    """CVXPY-based solver for LP and QP problems."""
    
    def __init__(self, name: str = "CVXPY", timeout: float = 300.0, 
                 solver: str = "CLARABEL", verbose: bool = False,
                 solver_options: Optional[Dict] = None):
        """
        Initialize CVXPY solver.
        
        Args:
            name: Solver name
            timeout: Timeout in seconds
            solver: CVXPY backend solver (CLARABEL, OSQP, SCS, etc.)
            verbose: Whether to enable verbose solver output
            solver_options: Additional options for the backend solver
        """
        super().__init__(name, timeout)
        self.backend_solver = solver
        self.verbose = verbose
        self.solver_options = solver_options or {}
        
        # Verify solver availability
        available_solvers = cp.installed_solvers()
        if solver not in available_solvers:
            self.logger.warning(f"Requested solver {solver} not available. "
                              f"Available solvers: {available_solvers}")
            # Fall back to first available solver
            if available_solvers:
                self.backend_solver = available_solvers[0]
                self.logger.info(f"Using fallback solver: {self.backend_solver}")
            else:
                raise RuntimeError("No CVXPY solvers available")
        
        self.logger.info(f"Initialized CVXPY solver with backend '{self.backend_solver}'")
    
    def solve(self, problem: ProblemData) -> SolverResult:
        """
        Solve optimization problem using CVXPY.
        
        Args:
            problem: Problem data to solve
            
        Returns:
            SolverResult containing solve status and results
        """
        self.logger.debug(f"Solving {problem.problem_class} problem '{problem.name}'")
        
        start_time = time.time()
        
        try:
            if problem.problem_class == "LP":
                return self._solve_lp(problem, start_time)
            elif problem.problem_class == "QP":
                return self._solve_qp(problem, start_time)
            else:
                raise ValueError(f"Unsupported problem class: {problem.problem_class}")
                
        except Exception as e:
            solve_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Solver failed: {error_msg}")
            
            return SolverResult(
                solver_name=self.name,
                problem_name=problem.name,
                solve_time=solve_time,
                status='error',
                error_message=error_msg
            )
    
    def _solve_lp(self, problem: ProblemData, start_time: float) -> SolverResult:
        """Solve linear programming problem using CVXPY."""
        
        # Extract problem data
        c = problem.c
        A_ub = problem.A_ub
        b_ub = problem.b_ub
        A_eq = problem.A_eq
        b_eq = problem.b_eq
        bounds = problem.bounds
        
        n_vars = len(c)
        
        # Create CVXPY variables
        x = cp.Variable(n_vars, name="x")
        
        # Create objective: minimize c^T * x
        objective = cp.Minimize(cp.sum(cp.multiply(c, x)))
        
        # Create constraints
        constraints = []
        
        # Inequality constraints: A_ub * x <= b_ub
        if A_ub is not None and b_ub is not None:
            constraints.append(A_ub @ x <= b_ub)
        
        # Equality constraints: A_eq * x == b_eq
        if A_eq is not None and b_eq is not None:
            constraints.append(A_eq @ x == b_eq)
        
        # Variable bounds
        if bounds:
            for i, bound in enumerate(bounds):
                if bound is not None:
                    if isinstance(bound, tuple):
                        lower, upper = bound
                        if lower is not None:
                            constraints.append(x[i] >= lower)
                        if upper is not None:
                            constraints.append(x[i] <= upper)
                    else:
                        # Assume non-negative if bound is not a tuple
                        constraints.append(x[i] >= 0)
                else:
                    # Default to non-negative
                    constraints.append(x[i] >= 0)
        else:
            # Default bounds: non-negative
            constraints.append(x >= 0)
        
        # Create problem
        cvx_problem = cp.Problem(objective, constraints)
        
        self.logger.debug(f"LP problem: {n_vars} variables, {len(constraints)} constraints")
        
        # Solve
        solver_options = self.solver_options.copy()
        if 'verbose' not in solver_options:
            solver_options['verbose'] = self.verbose
        
        try:
            cvx_problem.solve(
                solver=getattr(cp, self.backend_solver),
                **solver_options
            )
        except Exception as e:
            # Handle solver-specific errors
            solve_time = time.time() - start_time
            error_msg = f"Backend solver {self.backend_solver} failed: {str(e)}"
            self.logger.error(error_msg)
            
            return SolverResult(
                solver_name=self.name,
                problem_name=problem.name,
                solve_time=solve_time,
                status='error',
                error_message=error_msg
            )
        
        solve_time = time.time() - start_time
        
        # Map CVXPY status to our standard status
        status_mapping = {
            cp.OPTIMAL: 'optimal',
            cp.INFEASIBLE: 'infeasible',
            cp.UNBOUNDED: 'unbounded',
            cp.INFEASIBLE_INACCURATE: 'infeasible',
            cp.UNBOUNDED_INACCURATE: 'unbounded',
            cp.OPTIMAL_INACCURATE: 'optimal'
        }
        
        status = status_mapping.get(cvx_problem.status, 'unknown')
        objective_value = cvx_problem.value if cvx_problem.value is not None else None
        
        # Extract solver-specific information
        solver_info = {
            "cvxpy_status": cvx_problem.status,
            "backend_solver": self.backend_solver,
            "solver_stats": cvx_problem.solver_stats.__dict__ if cvx_problem.solver_stats else None,
            "cvxpy_version": cp.__version__
        }
        
        # Add solution norm if available
        if x.value is not None:
            solver_info["solution_norm"] = float(np.linalg.norm(x.value))
        
        # Get iterations if available
        iterations = None
        if cvx_problem.solver_stats and hasattr(cvx_problem.solver_stats, 'num_iters'):
            iterations = cvx_problem.solver_stats.num_iters
        
        self.logger.debug(f"LP solve completed: status={status}, "
                         f"objective={objective_value}, time={solve_time:.3f}s")
        
        return SolverResult(
            solver_name=self.name,
            problem_name=problem.name,
            solve_time=solve_time,
            status=status,
            objective_value=objective_value,
            iterations=iterations,
            solver_info=solver_info
        )
    
    def _solve_qp(self, problem: ProblemData, start_time: float) -> SolverResult:
        """Solve quadratic programming problem using CVXPY."""
        
        # For QP: minimize 0.5 * x^T * P * x + c^T * x
        # Subject to: A_ub * x <= b_ub, A_eq * x = b_eq, bounds
        
        P = problem.P
        c = problem.c if problem.c is not None else np.zeros(P.shape[0])
        A_ub = problem.A_ub
        b_ub = problem.b_ub
        A_eq = problem.A_eq
        b_eq = problem.b_eq
        bounds = problem.bounds
        
        n_vars = P.shape[0]
        
        # Create CVXPY variables
        x = cp.Variable(n_vars, name="x")
        
        # Create objective: minimize 0.5 * x^T * P * x + c^T * x
        quadratic_term = cp.quad_form(x, P)
        linear_term = cp.sum(cp.multiply(c, x))
        objective = cp.Minimize(0.5 * quadratic_term + linear_term)
        
        # Create constraints
        constraints = []
        
        # Inequality constraints: A_ub * x <= b_ub
        if A_ub is not None and b_ub is not None:
            constraints.append(A_ub @ x <= b_ub)
        
        # Equality constraints: A_eq * x == b_eq
        if A_eq is not None and b_eq is not None:
            constraints.append(A_eq @ x == b_eq)
        
        # Variable bounds
        if bounds:
            for i, bound in enumerate(bounds):
                if bound is not None:
                    if isinstance(bound, tuple):
                        lower, upper = bound
                        if lower is not None:
                            constraints.append(x[i] >= lower)
                        if upper is not None:
                            constraints.append(x[i] <= upper)
                    else:
                        # Assume non-negative if bound is not a tuple
                        constraints.append(x[i] >= 0)
                else:
                    # Default to non-negative
                    constraints.append(x[i] >= 0)
        else:
            # Default bounds: non-negative
            constraints.append(x >= 0)
        
        # Create problem
        cvx_problem = cp.Problem(objective, constraints)
        
        self.logger.debug(f"QP problem: {n_vars} variables, P shape: {P.shape}, "
                         f"{len(constraints)} constraints")
        
        # Solve
        solver_options = self.solver_options.copy()
        if 'verbose' not in solver_options:
            solver_options['verbose'] = self.verbose
        
        try:
            cvx_problem.solve(
                solver=getattr(cp, self.backend_solver),
                **solver_options
            )
        except Exception as e:
            # Handle solver-specific errors
            solve_time = time.time() - start_time
            error_msg = f"Backend solver {self.backend_solver} failed: {str(e)}"
            self.logger.error(error_msg)
            
            return SolverResult(
                solver_name=self.name,
                problem_name=problem.name,
                solve_time=solve_time,
                status='error',
                error_message=error_msg
            )
        
        solve_time = time.time() - start_time
        
        # Map CVXPY status to our standard status
        status_mapping = {
            cp.OPTIMAL: 'optimal',
            cp.INFEASIBLE: 'infeasible',
            cp.UNBOUNDED: 'unbounded',
            cp.INFEASIBLE_INACCURATE: 'infeasible',
            cp.UNBOUNDED_INACCURATE: 'unbounded',
            cp.OPTIMAL_INACCURATE: 'optimal'
        }
        
        status = status_mapping.get(cvx_problem.status, 'unknown')
        objective_value = cvx_problem.value if cvx_problem.value is not None else None
        
        # Extract solver-specific information
        solver_info = {
            "cvxpy_status": cvx_problem.status,
            "backend_solver": self.backend_solver,
            "solver_stats": cvx_problem.solver_stats.__dict__ if cvx_problem.solver_stats else None,
            "cvxpy_version": cp.__version__
        }
        
        # Add solution norm if available
        if x.value is not None:
            solver_info["solution_norm"] = float(np.linalg.norm(x.value))
        
        # Get iterations if available
        iterations = None
        if cvx_problem.solver_stats and hasattr(cvx_problem.solver_stats, 'num_iters'):
            iterations = cvx_problem.solver_stats.num_iters
        
        self.logger.debug(f"QP solve completed: status={status}, "
                         f"objective={objective_value}, time={solve_time:.3f}s")
        
        return SolverResult(
            solver_name=self.name,
            problem_name=problem.name,
            solve_time=solve_time,
            status=status,
            objective_value=objective_value,
            iterations=iterations,
            solver_info=solver_info
        )
    
    def get_version(self) -> str:
        """Get CVXPY version information."""
        return f"cvxpy-{cp.__version__}"
    
    def get_info(self) -> Dict[str, Any]:
        """Get solver information."""
        base_info = super().get_info()
        base_info.update({
            "backend_solver": self.backend_solver,
            "available_solvers": cp.installed_solvers(),
            "cvxpy_version": cp.__version__,
            "supported_problem_types": ["LP", "QP"],
            "solver_options": self.solver_options
        })
        return base_info

# Utility function to create solvers with different backends
def create_cvxpy_solvers() -> List[CvxpySolver]:
    """Create CVXPY solver instances for different backends."""
    available_solvers = cp.installed_solvers()
    solvers = []
    
    # Preferred solver order for different problem types
    preferred_solvers = ["CLARABEL", "OSQP", "SCS", "ECOS", "CVXOPT"]
    
    for backend in preferred_solvers:
        if backend in available_solvers:
            solver_name = f"CVXPY-{backend}"
            solver = CvxpySolver(
                name=solver_name,
                solver=backend,
                timeout=300.0
            )
            solvers.append(solver)
    
    return solvers

if __name__ == "__main__":
    # Test script to verify CVXPY solver
    try:
        print("Testing CVXPY Solver...")
        
        # Import problem loader for testing
        from scripts.benchmark.problem_loader import load_problem
        
        # Test available solvers
        print(f"\nAvailable CVXPY solvers: {cp.installed_solvers()}")
        
        # Test solver initialization with different backends
        print("\nTesting solver initialization:")
        for backend in ["CLARABEL", "OSQP", "SCS"]:
            if backend in cp.installed_solvers():
                try:
                    solver = CvxpySolver(name=f"CVXPY-{backend}", solver=backend)
                    print(f"✓ {solver.name} initialized")
                    print(f"  Version: {solver.get_version()}")
                    
                    # Test info
                    info = solver.get_info()
                    print(f"  Backend: {info['backend_solver']}")
                    print(f"  CVXPY version: {info['cvxpy_version']}")
                    
                except Exception as e:
                    print(f"✗ Failed to initialize {backend}: {e}")
        
        # Test LP solving with default solver
        print("\nTesting LP solving:")
        try:
            solver = CvxpySolver()
            lp_problem = load_problem("simple_lp")
            print(f"Loaded LP problem: {lp_problem.name}")
            
            lp_result = solver.solve(lp_problem)
            print(f"LP result: {lp_result}")
            print(f"  Status: {lp_result.status}")
            print(f"  Objective: {lp_result.objective_value}")
            print(f"  Time: {lp_result.solve_time:.3f}s")
            print(f"  Backend: {lp_result.solver_info['backend_solver']}")
            print(f"  CVXPY status: {lp_result.solver_info['cvxpy_status']}")
            
        except Exception as e:
            print(f"✗ LP test failed: {e}")
        
        # Test QP solving
        print("\nTesting QP solving:")
        try:
            qp_problem = load_problem("simple_qp")
            print(f"Loaded QP problem: {qp_problem.name}")
            
            qp_result = solver.solve(qp_problem)
            print(f"QP result: {qp_result}")
            print(f"  Status: {qp_result.status}")
            print(f"  Objective: {qp_result.objective_value}")
            print(f"  Time: {qp_result.solve_time:.3f}s")
            print(f"  Backend: {qp_result.solver_info['backend_solver']}")
            print(f"  CVXPY status: {qp_result.solver_info['cvxpy_status']}")
            
        except Exception as e:
            print(f"✗ QP test failed: {e}")
        
        # Test different backends on same problem
        print("\nTesting different backends:")
        for backend in ["CLARABEL", "OSQP", "SCS"]:
            if backend in cp.installed_solvers():
                try:
                    backend_solver = CvxpySolver(name=f"CVXPY-{backend}", solver=backend)
                    result = backend_solver.solve(lp_problem)
                    print(f"  {backend}: {result.status} in {result.solve_time:.3f}s "
                          f"(obj={result.objective_value})")
                except Exception as e:
                    print(f"  {backend}: Failed - {e}")
        
        # Test error handling
        print("\nTesting error handling:")
        try:
            # Create a problem with invalid data
            from scripts.benchmark.problem_loader import ProblemData
            invalid_problem = ProblemData(
                name="invalid",
                problem_class="INVALID",
                c=np.array([1.0])
            )
            
            error_result = solver.solve(invalid_problem)
            print(f"Error handling result: {error_result.status}")
            print(f"  Error message: {error_result.error_message}")
            
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
        
        # Test multiple solver creation
        print("\nTesting multiple solver creation:")
        try:
            solvers = create_cvxpy_solvers()
            print(f"Created {len(solvers)} CVXPY solver variants:")
            for s in solvers:
                info = s.get_info()
                print(f"  {s.name}: {info['backend_solver']}")
        except Exception as e:
            print(f"✗ Multiple solver creation failed: {e}")
        
        print("\n✓ All CVXPY solver tests completed!")
        
    except Exception as e:
        logger.error(f"CVXPY solver test failed: {e}")
        raise