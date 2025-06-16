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
from scripts.utils.solver_diagnostics import SolverDiagnostics
from scripts.utils.version_utils import detect_cvxpy_backend_version

logger = get_logger("cvxpy_solver")

class CvxpySolver(SolverInterface):
    """CVXPY-based solver for LP, QP, SOCP, and SDP problems."""
    
    def __init__(self, name: str = None, timeout: float = 300.0, 
                 backend: str = "CLARABEL", verbose: bool = False,
                 solver_options: Optional[Dict] = None,
                 problem_optimizations: Optional[Dict] = None,
                 enable_diagnostics: bool = True):
        """
        Initialize CVXPY solver with specific backend and enhanced configuration.
        
        Args:
            name: Solver name (auto-generated if None)
            timeout: Timeout in seconds
            backend: CVXPY backend solver (CLARABEL, OSQP, SCS, etc.)
            verbose: Whether to enable verbose solver output
            solver_options: Backend-specific solver options
            problem_optimizations: Problem-type-specific parameter overrides
            enable_diagnostics: Enable parameter validation and diagnostics
        """
        # Auto-generate name with proper "(via CVXPY)" format
        if name is None:
            name = f"{backend} (via CVXPY)"
        
        super().__init__(name, timeout)
        self.backend = backend
        self.verbose = verbose
        self.solver_options = solver_options or {}
        self.problem_optimizations = problem_optimizations or {}
        self.enable_diagnostics = enable_diagnostics
        
        # Detect solver version and backend
        self.solver_version = detect_cvxpy_backend_version(backend)
        self.solver_backend = backend
        
        # Initialize diagnostics if enabled
        self.diagnostics = None
        if self.enable_diagnostics:
            try:
                self.diagnostics = SolverDiagnostics()
                # Validate parameters
                if self.solver_options:
                    validation_results = self.diagnostics.validate_backend_parameters(
                        self.backend, self.solver_options
                    )
                    failed_validations = [r for r in validation_results if not r.passed]
                    if failed_validations:
                        self.logger.warning(f"Parameter validation issues for {self.backend}:")
                        for result in failed_validations:
                            self.logger.warning(f"  {result.message}")
            except Exception as e:
                self.logger.warning(f"Could not initialize diagnostics: {e}")
        
        # Verify solver availability
        available_solvers = cp.installed_solvers()
        if backend not in available_solvers:
            self.logger.warning(f"Requested backend {backend} not available. "
                              f"Available backends: {available_solvers}")
            # Fall back to first available solver
            if available_solvers:
                self.backend = available_solvers[0]
                self.name = f"{self.backend} (via CVXPY)"
                self.logger.info(f"Using fallback backend: {self.backend}")
            else:
                raise RuntimeError("No CVXPY backends available")
        
        # Get backend capabilities
        self.backend_capabilities = self._get_backend_capabilities()
        
        self.logger.info(f"Initialized CVXPY solver '{self.name}' with backend '{self.backend}'")
    
    def _get_backend_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of the current backend solver."""
        # Define backend capabilities for different problem types
        backend_capabilities = {
            "CLARABEL": ["LP", "QP", "SOCP", "SDP"],
            "SCS": ["LP", "QP", "SOCP", "SDP"],
            "ECOS": ["LP", "QP", "SOCP"],
            "OSQP": ["QP", "SOCP"],
            "CBC": ["LP"],
            "GLOP": ["LP"],
            "GLOP_MI": ["LP"],
            "SCIP": ["LP", "QP"],
            "GUROBI": ["LP", "QP", "SOCP"],
            "MOSEK": ["LP", "QP", "SOCP", "SDP"],
            "CPLEX": ["LP", "QP", "SOCP"],
            "CVXOPT": ["LP", "QP", "SOCP"],
            "XPRESS": ["LP", "QP"]
        }
        
        return {
            "supported_problem_types": backend_capabilities.get(self.backend, ["LP", "QP"]),
            "backend_name": self.backend
        }
    
    def _get_optimized_options(self, problem_class: str) -> Dict[str, Any]:
        """Get solver options optimized for specific problem type."""
        # Start with base solver options
        options = self.solver_options.copy()
        
        # Apply problem-specific optimizations if available
        if problem_class in self.problem_optimizations:
            optimizations = self.problem_optimizations[problem_class]
            options.update(optimizations)
            self.logger.debug(f"Applied {len(optimizations)} problem-specific optimizations for {problem_class}")
        
        # Add verbosity setting
        if 'verbose' not in options:
            options['verbose'] = self.verbose
        
        return options
    
    def _map_cvxpy_status(self, cvxpy_status: str) -> str:
        """Map CVXPY status to our standardized status format."""
        status_mapping = {
            "optimal": "optimal",
            "optimal_inaccurate": "optimal",
            "infeasible": "infeasible",
            "infeasible_inaccurate": "infeasible",
            "unbounded": "unbounded",
            "unbounded_inaccurate": "unbounded",
            "solver_error": "error",
            "unknown": "unknown"
        }
        return status_mapping.get(cvxpy_status, "unknown")
    
    def _create_result(self, solver_name: str, problem_name: str, solve_time: float,
                      status: str, objective_value: Optional[float] = None,
                      duality_gap: Optional[float] = None, iterations: Optional[int] = None,
                      error_message: Optional[str] = None, solver_info: Optional[Dict] = None,
                      problem_library: str = 'light_set', run_id: Optional[str] = None) -> SolverResult:
        """Create SolverResult with version information."""
        return SolverResult(
            solver_name=solver_name,
            problem_name=problem_name,
            solve_time=solve_time,
            status=status,
            objective_value=objective_value,
            duality_gap=duality_gap,
            iterations=iterations,
            error_message=error_message,
            solver_info=solver_info,
            solver_version=self.solver_version,
            solver_backend=self.solver_backend,
            problem_library=problem_library,
            run_id=run_id
        )
    
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
            elif problem.problem_class == "SOCP":
                return self._solve_socp(problem, start_time)
            elif problem.problem_class == "SDP":
                return self._solve_sdp(problem, start_time)
            else:
                raise ValueError(f"Unsupported problem class: {problem.problem_class}")
                
        except Exception as e:
            solve_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Solver failed: {error_msg}")
            
            return self._create_result(
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
        
        # Get optimized solver options for this problem type
        solver_options = self._get_optimized_options(problem.problem_class)
        
        try:
            cvx_problem.solve(
                solver=getattr(cp, self.backend),
                **solver_options
            )
        except Exception as e:
            # Handle solver-specific errors
            solve_time = time.time() - start_time
            error_msg = f"Backend solver {self.backend} failed: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_result(
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
            "backend_solver": self.backend,
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
        
        return self._create_result(
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
        
        # Get optimized solver options for this problem type
        solver_options = self._get_optimized_options(problem.problem_class)
        
        try:
            cvx_problem.solve(
                solver=getattr(cp, self.backend),
                **solver_options
            )
        except Exception as e:
            # Handle solver-specific errors
            solve_time = time.time() - start_time
            error_msg = f"Backend solver {self.backend} failed: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_result(
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
            "backend_solver": self.backend,
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
        
        return self._create_result(
            solver_name=self.name,
            problem_name=problem.name,
            solve_time=solve_time,
            status=status,
            objective_value=objective_value,
            iterations=iterations,
            solver_info=solver_info
        )
    
    def _solve_socp(self, problem: ProblemData, start_time: float) -> SolverResult:
        """Solve Second-Order Cone Programming problem using CVXPY."""
        
        # For SOCP problems, we use the pre-built CVXPY problem from the problem data
        if problem.cvxpy_problem is None:
            raise ValueError("SOCP problem must include a pre-built CVXPY problem")
        
        cvx_problem = problem.cvxpy_problem
        
        # Get problem dimensions for logging
        try:
            n_vars = cvx_problem.size_metrics.num_scalar_variables
        except AttributeError:
            n_vars = len([v for v in cvx_problem.variables() if v.is_scalar()]) if hasattr(cvx_problem, 'variables') else 0
        
        try:
            n_constraints = cvx_problem.size_metrics.num_scalar_eq_constraints + cvx_problem.size_metrics.num_scalar_leq_constraints
        except AttributeError:
            n_constraints = len(cvx_problem.constraints) if hasattr(cvx_problem, 'constraints') else 0
        
        self.logger.debug(f"SOCP problem: {n_vars} variables, {n_constraints} constraints")
        
        # Check if backend supports SOCP
        if "SOCP" not in self.backend_capabilities["supported_problem_types"]:
            raise ValueError(f"Backend {self.backend} does not support SOCP problems")
        
        # Get optimized solver options for SOCP
        solver_options = self._get_optimized_options(problem.problem_class)
        
        try:
            cvx_problem.solve(
                solver=getattr(cp, self.backend),
                **solver_options
            )
        except Exception as e:
            solve_time = time.time() - start_time
            error_msg = f"Backend solver failed: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_result(
                solver_name=self.name,
                problem_name=problem.name,
                solve_time=solve_time,
                status='error',
                error_message=error_msg
            )
        
        solve_time = time.time() - start_time
        
        # Map CVXPY status to our status format
        status = self._map_cvxpy_status(cvx_problem.status)
        
        # Extract solution info
        objective_value = cvx_problem.value if cvx_problem.value is not None else None
        
        # Get solver statistics
        solver_info = {
            "backend_solver": self.backend,
            "cvxpy_status": cvx_problem.status,
            "solver_stats": cvx_problem.solver_stats.__dict__ if cvx_problem.solver_stats else {},
            "problem_size": {
                "variables": n_vars,
                "constraints": n_constraints,
                "socp_constraints": len([c for c in cvx_problem.constraints if hasattr(c, 'cone_type') and c.cone_type == 'SOC'])
            }
        }
        
        # Get iteration count if available
        iterations = None
        if cvx_problem.solver_stats and hasattr(cvx_problem.solver_stats, 'num_iters'):
            iterations = cvx_problem.solver_stats.num_iters
        
        self.logger.debug(f"SOCP solve completed: status={status}, "
                         f"objective={objective_value}, time={solve_time:.3f}s")
        
        return self._create_result(
            solver_name=self.name,
            problem_name=problem.name,
            solve_time=solve_time,
            status=status,
            objective_value=objective_value,
            iterations=iterations,
            solver_info=solver_info
        )
    
    def _solve_sdp(self, problem: ProblemData, start_time: float) -> SolverResult:
        """Solve Semidefinite Programming problem using CVXPY."""
        
        # For SDP problems, we use the pre-built CVXPY problem from the problem data
        if problem.cvxpy_problem is None:
            raise ValueError("SDP problem must include a pre-built CVXPY problem")
        
        cvx_problem = problem.cvxpy_problem
        
        # Get problem dimensions for logging
        try:
            n_vars = cvx_problem.size_metrics.num_scalar_variables
        except AttributeError:
            n_vars = len([v for v in cvx_problem.variables() if v.is_scalar()]) if hasattr(cvx_problem, 'variables') else 0
        
        try:
            n_constraints = cvx_problem.size_metrics.num_scalar_eq_constraints + cvx_problem.size_metrics.num_scalar_leq_constraints
        except AttributeError:
            n_constraints = len(cvx_problem.constraints) if hasattr(cvx_problem, 'constraints') else 0
        
        self.logger.debug(f"SDP problem: {n_vars} variables, {n_constraints} constraints")
        
        # Check if backend supports SDP
        if "SDP" not in self.backend_capabilities["supported_problem_types"]:
            raise ValueError(f"Backend {self.backend} does not support SDP problems")
        
        # Get optimized solver options for SDP
        solver_options = self._get_optimized_options(problem.problem_class)
        
        try:
            cvx_problem.solve(
                solver=getattr(cp, self.backend),
                **solver_options
            )
        except Exception as e:
            solve_time = time.time() - start_time
            error_msg = f"Backend solver failed: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_result(
                solver_name=self.name,
                problem_name=problem.name,
                solve_time=solve_time,
                status='error',
                error_message=error_msg
            )
        
        solve_time = time.time() - start_time
        
        # Map CVXPY status to our status format
        status = self._map_cvxpy_status(cvx_problem.status)
        
        # Extract solution info
        objective_value = cvx_problem.value if cvx_problem.value is not None else None
        
        # Get solver statistics
        solver_info = {
            "backend_solver": self.backend,
            "cvxpy_status": cvx_problem.status,
            "solver_stats": cvx_problem.solver_stats.__dict__ if cvx_problem.solver_stats else {},
            "problem_size": {
                "variables": n_vars,
                "constraints": n_constraints,
                "sdp_constraints": len([c for c in cvx_problem.constraints if hasattr(c, 'cone_type') and c.cone_type == 'PSD'])
            }
        }
        
        # Get iteration count if available
        iterations = None
        if cvx_problem.solver_stats and hasattr(cvx_problem.solver_stats, 'num_iters'):
            iterations = cvx_problem.solver_stats.num_iters
        
        self.logger.debug(f"SDP solve completed: status={status}, "
                         f"objective={objective_value}, time={solve_time:.3f}s")
        
        return self._create_result(
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
            "backend_solver": self.backend,
            "available_backends": cp.installed_solvers(),
            "cvxpy_version": cp.__version__,
            "supported_problem_types": self.backend_capabilities["supported_problem_types"],
            "solver_options": self.solver_options,
            "backend_capabilities": self.backend_capabilities
        })
        return base_info

# Utility function to create solvers with different backends
def create_cvxpy_solvers(timeout: float = 300.0, verbose: bool = False) -> List[CvxpySolver]:
    """Create CVXPY solver instances for different available backends."""
    available_backends = cp.installed_solvers()
    solver_instances = []
    
    # Define open-source backends in order of preference
    open_source_backends = {
        # General purpose solvers
        "CLARABEL": {},
        "SCS": {},
        "ECOS": {},
        
        # QP specialists  
        "OSQP": {},
        
        # LP specialists
        "CBC": {},
        "GLOP": {},
        "SCIP": {},
        
        # Other open-source
        "CVXOPT": {}
    }
    
    for backend_name, options in open_source_backends.items():
        if backend_name in available_backends:
            try:
                solver_instance = CvxpySolver(
                    backend=backend_name,
                    timeout=timeout,
                    verbose=verbose,
                    solver_options=options
                )
                solver_instances.append(solver_instance)
            except Exception as e:
                logger.warning(f"Failed to create solver for backend {backend_name}: {e}")
    
    return solver_instances

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
                    solver = CvxpySolver(backend=backend)
                    print(f"✓ {solver.name} initialized")
                    print(f"  Version: {solver.get_version()}")
                    
                    # Test info
                    info = solver.get_info()
                    print(f"  Backend: {info['backend_solver']}")
                    print(f"  CVXPY version: {info['cvxpy_version']}")
                    print(f"  Supported types: {info['supported_problem_types']}")
                    
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
                    backend_solver = CvxpySolver(backend=backend)
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