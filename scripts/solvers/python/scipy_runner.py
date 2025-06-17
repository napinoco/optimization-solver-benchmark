import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from scipy.optimize import linprog, minimize
import scipy

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.solver_interface import SolverInterface, SolverResult
from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("scipy_solver")

class ScipySolver(SolverInterface):
    """SciPy-based solver for LP and QP problems."""
    
    def __init__(self, method: str = "highs", options: Optional[Dict] = None, **kwargs):
        """
        Initialize SciPy solver.
        
        Args:
            method: SciPy optimization method to use
            options: Additional options for the solver
            **kwargs: Additional configuration parameters
        """
        super().__init__("scipy_linprog", method=method, options=options, **kwargs)
        self.method = method
        self.options = options or {}
        
        self.logger.info(f"Initialized SciPy solver with method '{method}'")
    
    def _create_standardized_result(self, solve_time: float, status: str, 
                                  primal_objective_value: Optional[float] = None,
                                  dual_objective_value: Optional[float] = None,
                                  duality_gap: Optional[float] = None,
                                  primal_infeasibility: Optional[float] = None,
                                  dual_infeasibility: Optional[float] = None,
                                  iterations: Optional[int] = None,
                                  additional_info: Optional[Dict] = None) -> SolverResult:
        """Create standardized SolverResult."""
        return SolverResult(
            solve_time=solve_time,
            status=status,
            primal_objective_value=primal_objective_value,
            dual_objective_value=dual_objective_value,
            duality_gap=duality_gap,
            primal_infeasibility=primal_infeasibility,
            dual_infeasibility=dual_infeasibility,
            iterations=iterations,
            solver_name=self.solver_name,
            solver_version=self.get_version(),
            additional_info=additional_info
        )
    
    def solve(self, problem_data: ProblemData, timeout: Optional[float] = None) -> SolverResult:
        """
        Solve optimization problem using SciPy.
        
        Args:
            problem_data: Problem data to solve
            timeout: Optional timeout in seconds (ignored for SciPy)
            
        Returns:
            SolverResult containing solve status and results
        """
        self.logger.debug(f"Solving {problem_data.problem_class} problem '{problem_data.name}'")
        
        start_time = time.time()
        
        try:
            if problem_data.problem_class == "LP":
                return self._solve_lp(problem_data, start_time)
            elif problem_data.problem_class == "QP":
                return self._solve_qp(problem_data, start_time)
            else:
                error_msg = f"Unsupported problem class: {problem_data.problem_class}"
                self.logger.error(error_msg)
                solve_time = time.time() - start_time
                return SolverResult.create_error_result(error_msg, solve_time)
                
        except Exception as e:
            solve_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Solver failed: {error_msg}")
            return SolverResult.create_error_result(error_msg, solve_time)
    
    def _solve_lp(self, problem_data: ProblemData, start_time: float) -> SolverResult:
        """Solve linear programming problem using scipy.optimize.linprog."""
        
        # Prepare problem data for linprog
        c = problem_data.c
        A_ub = problem_data.A_ub
        b_ub = problem_data.b_ub
        A_eq = problem_data.A_eq
        b_eq = problem_data.b_eq
        bounds = problem_data.bounds
        
        # Convert bounds to scipy format
        if bounds:
            bounds_list = []
            for bound in bounds:
                if bound is None:
                    bounds_list.append((0, None))  # Default non-negative
                elif isinstance(bound, tuple):
                    bounds_list.append(bound)
                else:
                    bounds_list.append((0, None))  # Default non-negative
            bounds = bounds_list
        else:
            # If no bounds specified, assume non-negative variables
            n_vars = len(c) if c is not None else 0
            bounds = [(0, None)] * n_vars
        
        self.logger.debug(f"LP problem dimensions: c={c.shape if c is not None else None}, "
                         f"A_ub={A_ub.shape if A_ub is not None else None}, "
                         f"A_eq={A_eq.shape if A_eq is not None else None}")
        
        # Solve using linprog
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=self.method,
            options=self.options
        )
        
        solve_time = time.time() - start_time
        
        # Map SciPy status to standard status
        status_mapping = {
            0: 'OPTIMAL',      # Optimization terminated successfully
            1: 'MAX_ITERATIONS',   # Iteration limit reached
            2: 'INFEASIBLE',   # Problem appears to be infeasible
            3: 'UNBOUNDED',    # Problem appears to be unbounded
            4: 'ERROR'         # Numerical difficulties encountered
        }
        
        status = status_mapping.get(result.status, 'UNKNOWN')
        primal_objective_value = float(result.fun) if result.success else None
        
        # Calculate infeasibility measures if available
        primal_infeasibility = None
        if hasattr(result, 'con') and result.con is not None and len(result.con) > 0:
            primal_infeasibility = float(np.max(np.abs(result.con)))
        
        # Extract solver-specific information
        additional_info = {
            "scipy_status": result.status,
            "scipy_message": result.message,
            "scipy_success": result.success,
            "scipy_nit": getattr(result, 'nit', None),
            "method": self.method
        }
        
        if hasattr(result, 'x') and result.x is not None:
            additional_info["solution_norm"] = float(np.linalg.norm(result.x))
        
        self.logger.debug(f"LP solve completed: status={status}, "
                         f"objective={primal_objective_value}, time={solve_time:.3f}s")
        
        return self._create_standardized_result(
            solve_time=solve_time,
            status=status,
            primal_objective_value=primal_objective_value,
            dual_objective_value=None,  # SciPy doesn't provide dual values easily
            duality_gap=None,
            primal_infeasibility=primal_infeasibility,
            dual_infeasibility=None,
            iterations=getattr(result, 'nit', None),
            additional_info=additional_info
        )
    
    def _solve_qp(self, problem_data: ProblemData, start_time: float) -> SolverResult:
        """Solve quadratic programming problem using scipy.optimize.minimize."""
        
        # For QP: minimize 0.5 * x^T * Q * x + c^T * x
        # Subject to: A_ub * x <= b_ub, A_eq * x = b_eq, bounds
        
        Q = getattr(problem_data, 'Q', None)
        if Q is None:
            error_msg = "QP problem missing quadratic matrix Q"
            solve_time = time.time() - start_time
            return SolverResult.create_error_result(error_msg, solve_time)
            
        c = problem_data.c if problem_data.c is not None else np.zeros(Q.shape[0])
        A_ub = problem_data.A_ub
        b_ub = problem_data.b_ub
        A_eq = problem_data.A_eq
        b_eq = problem_data.b_eq
        bounds = problem_data.bounds
        
        n_vars = Q.shape[0]
        
        # Define objective function and its gradient
        def objective(x):
            return 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(c, x)
        
        def gradient(x):
            return np.dot(Q, x) + c
        
        def hessian(x):
            return Q
        
        # Prepare constraints
        constraints = []
        
        # Equality constraints
        if A_eq is not None and b_eq is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(A_eq, x) - b_eq,
                'jac': lambda x: A_eq
            })
        
        # Inequality constraints (A_ub * x <= b_ub)
        if A_ub is not None and b_ub is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: b_ub - np.dot(A_ub, x),
                'jac': lambda x: -A_ub
            })
        
        # Convert bounds to scipy format
        if bounds:
            scipy_bounds = []
            for bound in bounds:
                if bound is None:
                    scipy_bounds.append((None, None))
                elif isinstance(bound, tuple):
                    scipy_bounds.append(bound)
                else:
                    scipy_bounds.append((0, None))  # Default non-negative
            bounds = scipy_bounds
        else:
            bounds = [(0, None)] * n_vars  # Default non-negative
        
        # Initial guess (zero vector)
        x0 = np.zeros(n_vars)
        
        self.logger.debug(f"QP problem dimensions: Q={Q.shape}, c={c.shape}, "
                         f"constraints={len(constraints)}")
        
        # Solve using minimize
        result = minimize(
            fun=objective,
            x0=x0,
            method='trust-constr',  # Good for constrained QP
            jac=gradient,
            hess=hessian,
            bounds=bounds,
            constraints=constraints,
            options=self.options
        )
        
        solve_time = time.time() - start_time
        
        # Map SciPy status to standard status
        if result.success:
            status = 'OPTIMAL'
        else:
            # Try to infer status from message
            message = result.message.lower()
            if 'infeasible' in message:
                status = 'INFEASIBLE'
            elif 'unbounded' in message:
                status = 'UNBOUNDED'
            else:
                status = 'ERROR'
        
        primal_objective_value = float(result.fun) if result.success else None
        
        # Calculate constraint violations if available
        primal_infeasibility = None
        if hasattr(result, 'constr_violation') and result.constr_violation is not None:
            primal_infeasibility = float(result.constr_violation)
        
        # Extract solver-specific information
        additional_info = {
            "scipy_success": result.success,
            "scipy_message": result.message,
            "scipy_nfev": getattr(result, 'nfev', None),
            "scipy_njev": getattr(result, 'njev', None),
            "scipy_nhev": getattr(result, 'nhev', None),
            "method": "trust-constr"
        }
        
        if hasattr(result, 'x') and result.x is not None:
            additional_info["solution_norm"] = float(np.linalg.norm(result.x))
        
        self.logger.debug(f"QP solve completed: status={status}, "
                         f"objective={primal_objective_value}, time={solve_time:.3f}s")
        
        return self._create_standardized_result(
            solve_time=solve_time,
            status=status,
            primal_objective_value=primal_objective_value,
            dual_objective_value=None,  # SciPy doesn't provide dual values easily
            duality_gap=None,
            primal_infeasibility=primal_infeasibility,
            dual_infeasibility=None,
            iterations=getattr(result, 'nit', None),
            additional_info=additional_info
        )
    
    def get_version(self) -> str:
        """Get SciPy version information."""
        return f"scipy-{scipy.__version__}"
    
    def validate_problem_compatibility(self, problem_data: ProblemData) -> bool:
        """Check if the solver can handle the given problem type."""
        return problem_data.problem_class in ["LP", "QP"]

if __name__ == "__main__":
    # Test script to verify SciPy solver
    print("Testing SciPy Solver...")
    
    # Test solver initialization
    print("\nTesting solver initialization:")
    solver = ScipySolver(method="highs")
    print(f"✓ Solver created: {solver.solver_name}")
    print(f"  Version: {solver.get_version()}")
    print(f"  Config: {solver.config}")
    
    # Test with simple LP problem
    print("\nTesting LP solving:")
    try:
        # Create a simple LP problem: minimize c^T * x subject to A_ub * x <= b_ub
        # minimize x1 + 2*x2 subject to x1 + x2 <= 1, x1, x2 >= 0
        simple_lp = ProblemData(
            name="test_lp",
            problem_class="LP",
            c=np.array([1.0, 2.0]),
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([1.0]),
            bounds=[(0, None), (0, None)]
        )
        
        result = solver.solve(simple_lp)
        print(f"✓ LP result: {result.status}")
        print(f"  Objective: {result.primal_objective_value}")
        print(f"  Time: {result.solve_time:.3f}s")
        print(f"  Iterations: {result.iterations}")
        
    except Exception as e:
        print(f"✗ LP test failed: {e}")
    
    # Test error handling
    print("\nTesting error handling:")
    try:
        invalid_problem = ProblemData(
            name="invalid",
            problem_class="INVALID",
            c=np.array([1.0])
        )
        
        error_result = solver.solve(invalid_problem)
        print(f"✓ Error handling result: {error_result.status}")
        print(f"  Additional info: {error_result.additional_info}")
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
    
    print("\n✓ SciPy solver test completed!")