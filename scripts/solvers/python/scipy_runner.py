import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from scipy.optimize import linprog, minimize

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.solver_interface import SolverInterface, SolverResult
from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("scipy_solver")

class ScipySolver(SolverInterface):
    """SciPy-based solver for LP and QP problems."""
    
    def __init__(self, name: str = "SciPy", timeout: float = 300.0, 
                 method: str = "highs", options: Optional[Dict] = None):
        """
        Initialize SciPy solver.
        
        Args:
            name: Solver name
            timeout: Timeout in seconds
            method: SciPy optimization method to use
            options: Additional options for the solver
        """
        super().__init__(name, timeout)
        self.method = method
        self.options = options or {}
        self.logger.info(f"Initialized SciPy solver with method '{method}'")
    
    def solve(self, problem: ProblemData) -> SolverResult:
        """
        Solve optimization problem using SciPy.
        
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
        """Solve linear programming problem using scipy.optimize.linprog."""
        
        # Prepare problem data for linprog
        c = problem.c
        A_ub = problem.A_ub
        b_ub = problem.b_ub
        A_eq = problem.A_eq
        b_eq = problem.b_eq
        bounds = problem.bounds
        
        # Convert bounds to scipy format
        if bounds:
            bounds_list = []
            for bound in bounds:
                if bound is None:
                    bounds_list.append((None, None))
                elif isinstance(bound, tuple):
                    bounds_list.append(bound)
                else:
                    bounds_list.append((0, None))  # Default non-negative
            bounds = bounds_list
        
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
        
        # Map SciPy status to our standard status
        status_mapping = {
            0: 'optimal',      # Optimization terminated successfully
            1: 'infeasible',   # Iteration limit reached
            2: 'infeasible',   # Problem appears to be infeasible
            3: 'unbounded',    # Problem appears to be unbounded
            4: 'error'         # Numerical difficulties encountered
        }
        
        status = status_mapping.get(result.status, 'unknown')
        objective_value = float(result.fun) if result.success else None
        
        # Extract solver-specific information
        solver_info = {
            "scipy_status": result.status,
            "scipy_message": result.message,
            "scipy_success": result.success,
            "scipy_nit": getattr(result, 'nit', None),
            "method": self.method
        }
        
        if hasattr(result, 'x') and result.x is not None:
            solver_info["solution_norm"] = float(np.linalg.norm(result.x))
        
        self.logger.debug(f"LP solve completed: status={status}, "
                         f"objective={objective_value}, time={solve_time:.3f}s")
        
        return SolverResult(
            solver_name=self.name,
            problem_name=problem.name,
            solve_time=solve_time,
            status=status,
            objective_value=objective_value,
            iterations=getattr(result, 'nit', None),
            solver_info=solver_info
        )
    
    def _solve_qp(self, problem: ProblemData, start_time: float) -> SolverResult:
        """Solve quadratic programming problem using scipy.optimize.minimize."""
        
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
        
        # Define objective function and its gradient
        def objective(x):
            return 0.5 * np.dot(x, np.dot(P, x)) + np.dot(c, x)
        
        def gradient(x):
            return np.dot(P, x) + c
        
        def hessian(x):
            return P
        
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
        
        self.logger.debug(f"QP problem dimensions: P={P.shape}, c={c.shape}, "
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
        
        # Map SciPy status to our standard status
        if result.success:
            status = 'optimal'
        else:
            # Try to infer status from message
            message = result.message.lower()
            if 'infeasible' in message:
                status = 'infeasible'
            elif 'unbounded' in message:
                status = 'unbounded'
            else:
                status = 'error'
        
        objective_value = float(result.fun) if result.success else None
        
        # Extract solver-specific information
        solver_info = {
            "scipy_success": result.success,
            "scipy_message": result.message,
            "scipy_nfev": getattr(result, 'nfev', None),
            "scipy_njev": getattr(result, 'njev', None),
            "scipy_nhev": getattr(result, 'nhev', None),
            "method": "trust-constr"
        }
        
        if hasattr(result, 'x') and result.x is not None:
            solver_info["solution_norm"] = float(np.linalg.norm(result.x))
        
        self.logger.debug(f"QP solve completed: status={status}, "
                         f"objective={objective_value}, time={solve_time:.3f}s")
        
        return SolverResult(
            solver_name=self.name,
            problem_name=problem.name,
            solve_time=solve_time,
            status=status,
            objective_value=objective_value,
            iterations=getattr(result, 'nit', None),
            solver_info=solver_info
        )
    
    def get_version(self) -> str:
        """Get SciPy version information."""
        try:
            import scipy
            return f"scipy-{scipy.__version__}"
        except ImportError:
            return "scipy-unknown"
    
    def get_info(self) -> Dict[str, Any]:
        """Get solver information."""
        base_info = super().get_info()
        base_info.update({
            "method": self.method,
            "options": self.options,
            "supported_problem_types": ["LP", "QP"]
        })
        return base_info

if __name__ == "__main__":
    # Test script to verify SciPy solver
    try:
        print("Testing SciPy Solver...")
        
        # Import problem loader for testing
        from scripts.benchmark.problem_loader import load_problem
        
        # Test solver initialization
        print("\nTesting solver initialization:")
        solver = ScipySolver(method="highs")
        print(f"✓ Solver created: {solver.name}")
        print(f"  Version: {solver.get_version()}")
        print(f"  Info: {solver.get_info()}")
        
        # Test LP solving
        print("\nTesting LP solving:")
        try:
            lp_problem = load_problem("simple_lp")
            print(f"Loaded LP problem: {lp_problem.name}")
            
            lp_result = solver.solve_with_timeout(lp_problem)
            print(f"LP result: {lp_result}")
            print(f"  Status: {lp_result.status}")
            print(f"  Objective: {lp_result.objective_value}")
            print(f"  Time: {lp_result.solve_time:.3f}s")
            print(f"  Solver info: {lp_result.solver_info}")
            
        except Exception as e:
            print(f"✗ LP test failed: {e}")
        
        # Test QP solving
        print("\nTesting QP solving:")
        try:
            qp_problem = load_problem("simple_qp")
            print(f"Loaded QP problem: {qp_problem.name}")
            
            qp_result = solver.solve_with_timeout(qp_problem)
            print(f"QP result: {qp_result}")
            print(f"  Status: {qp_result.status}")
            print(f"  Objective: {qp_result.objective_value}")
            print(f"  Time: {qp_result.solve_time:.3f}s")
            print(f"  Solver info: {qp_result.solver_info}")
            
        except Exception as e:
            print(f"✗ QP test failed: {e}")
        
        # Test different LP methods
        print("\nTesting different LP methods:")
        methods = ["highs", "highs-ds", "highs-ipm"]
        
        for method in methods:
            try:
                method_solver = ScipySolver(name=f"SciPy-{method}", method=method)
                lp_problem = load_problem("simple_lp")
                result = method_solver.solve_with_timeout(lp_problem)
                print(f"  {method}: {result.status} in {result.solve_time:.3f}s")
            except Exception as e:
                print(f"  {method}: Failed - {e}")
        
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
            
            error_result = solver.solve_with_timeout(invalid_problem)
            print(f"Error handling result: {error_result.status}")
            print(f"  Error message: {error_result.error_message}")
            
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
        
        print("\n✓ All SciPy solver tests completed!")
        
    except Exception as e:
        logger.error(f"SciPy solver test failed: {e}")
        raise