"""
CVXPY Solver for LP, QP, SOCP, and SDP Problems.

This module provides a CVXPY-based solver that uses different backends
for solving optimization problems. It implements the standardized solver
interface for consistent result format across all solvers.
"""

import sys
import time
import numpy as np
import cvxpy as cp
from pathlib import Path
from typing import Optional, Dict, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.solver_interface import SolverInterface, SolverResult
from scripts.data_loaders.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("cvxpy_solver")


class CvxpySolver(SolverInterface):
    """CVXPY-based solver for LP, QP, SOCP, and SDP problems."""
    
    def __init__(self, backend: str = "CLARABEL", verbose: bool = False,
                 solver_options: Optional[Dict] = None, **kwargs):
        """
        Initialize CVXPY solver with specific backend.
        
        Args:
            backend: CVXPY backend solver (CLARABEL, OSQP, SCS, etc.)
            verbose: Whether to enable verbose solver output
            solver_options: Backend-specific solver options
            **kwargs: Additional configuration parameters
        """
        # Auto-generate name with proper format
        solver_name = f"cvxpy_{backend.lower()}"
        
        super().__init__(solver_name, backend=backend, verbose=verbose, 
                        solver_options=solver_options, **kwargs)
        self.backend = backend
        self.verbose = verbose
        self.solver_options = solver_options or {}
        
        # Verify solver availability - no fallbacks for pure benchmarking
        available_solvers = cp.installed_solvers()
        if backend not in available_solvers:
            raise RuntimeError(f"Requested backend {backend} not available. "
                             f"Available backends: {available_solvers}")
        
        # Get backend capabilities
        self.backend_capabilities = self._get_backend_capabilities()
        
        self.logger.info(f"Initialized CVXPY solver '{self.solver_name}' with backend '{self.backend}'")
    
    def _get_backend_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of the current backend solver dynamically."""
        
        try:
            solver_obj = getattr(cp, self.backend, None)
            if solver_obj is None:
                return {
                    "supported_problem_types": [],
                    "backend_name": self.backend
                }
            
            supported_types = []
            
            # Test problem types by creating simple test problems
            # This is more reliable than hard-coding capabilities
            
            # Test LP support (all solvers should support this)
            try:
                x = cp.Variable(1)
                prob = cp.Problem(cp.Minimize(x), [x >= 0])
                prob.solve(solver=solver_obj, verbose=False)
                if prob.status not in [cp.SOLVER_ERROR]:
                    supported_types.append("LP")
            except:
                pass
            
            # Test QP support
            try:
                x = cp.Variable(1)
                prob = cp.Problem(cp.Minimize(cp.square(x)), [x >= 0])
                prob.solve(solver=solver_obj, verbose=False)
                if prob.status not in [cp.SOLVER_ERROR]:
                    supported_types.append("QP")
            except:
                pass
            
            # Test SOCP support
            try:
                x = cp.Variable(2)
                prob = cp.Problem(cp.Minimize(cp.sum(x)), [cp.norm(x) <= 1])
                prob.solve(solver=solver_obj, verbose=False)
                if prob.status not in [cp.SOLVER_ERROR]:
                    supported_types.append("SOCP")
            except:
                pass
            
            # Test SDP support
            try:
                X = cp.Variable((2, 2), symmetric=True)
                prob = cp.Problem(cp.Minimize(cp.trace(X)), [X >> 0])
                prob.solve(solver=solver_obj, verbose=False)
                if prob.status not in [cp.SOLVER_ERROR]:
                    supported_types.append("SDP")
            except:
                pass
            
            # If no tests passed, default to LP
            if not supported_types:
                supported_types = ["LP"]
                
        except Exception as e:
            logger.debug(f"Error detecting capabilities for {self.backend}: {e}")
            supported_types = ["LP"]
        
        logger.debug(f"Detected capabilities for {self.backend}: {supported_types}")
        
        return {
            "supported_problem_types": supported_types,
            "backend_name": self.backend
        }
    
    def _get_solver_options(self, timeout: Optional[float] = None) -> Dict:
        """Get solver options with verbosity and timeout."""
        options = self.solver_options.copy()
        
        # Handle backend-specific option formats
        if self.backend == "SCIPY":
            # SCIPY solver requires options in scipy_options dict
            scipy_options = options.get('scipy_options', {})
            
            # Set method to HiGHS for better performance
            if 'method' not in scipy_options:
                scipy_options['method'] = 'highs'
            
            # Add timeout if specified
            if timeout is not None:
                scipy_options['maxiter'] = int(timeout * 1000)  # rough conversion
            
            options = {
                'verbose': self.verbose,
                'scipy_options': scipy_options
            }
        elif self.backend == "SCIP":
            # SCIP solver has different parameter names
            if 'verbose' not in options:
                options['verbose'] = self.verbose
            
            # SCIP timeout parameter is not well supported in CVXPY, skip for now
            # TODO: Research correct SCIP timeout parameter format
        elif self.backend == "HIGHS":
            # HiGHS solver options
            if 'verbose' not in options:
                options['verbose'] = self.verbose
            
            # HiGHS supports time_limit parameter
            if timeout is not None:
                options['time_limit'] = timeout
        else:
            # Standard format for other solvers
            if 'verbose' not in options:
                options['verbose'] = self.verbose
            
            # Add timeout if specified
            if timeout is not None:
                options['max_time'] = timeout
        
        return options
    
    def solve(self, problem_data: ProblemData, timeout: Optional[float] = None) -> SolverResult:
        """
        Solve optimization problem using CVXPY.
        
        Args:
            problem_data: Problem data to solve
            timeout: Optional timeout in seconds
            
        Returns:
            SolverResult containing solve status and results
        """
        self.logger.debug(f"Solving {problem_data.problem_class} problem '{problem_data.name}'")
        
        start_time = time.time()
        
        try:
            # Check if backend supports this problem type
            if problem_data.problem_class not in self.backend_capabilities["supported_problem_types"]:
                error_msg = f"Backend {self.backend} does not support {problem_data.problem_class} problems"
                self.logger.error(error_msg)
                solve_time = time.time() - start_time
                return SolverResult.create_error_result(error_msg, solve_time)
            
            # Convert to CVXPY format (unified approach)
            cvx_problem = self._convert_to_cvxpy(problem_data)
            
            # Get solver options
            solver_options = self._get_solver_options(timeout)
            
            # Solve the problem
            cvx_problem.solve(
                solver=getattr(cp, self.backend),
                **solver_options
            )
            
            return self._create_result_from_cvxpy(cvx_problem, start_time)
                
        except Exception as e:
            solve_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"Solver failed: {error_msg}")
            return SolverResult.create_error_result(error_msg, solve_time)
    
    def _convert_to_cvxpy(self, problem_data: ProblemData) -> cp.Problem:
        """
        Convert ProblemData to CVXPY Problem format (unified for all problem types).
        
        This method now works entirely from SeDuMi format (A_eq/b_eq + cone_structure)
        without requiring legacy CVXPY fields.
        
        Args:
            problem_data: Problem data from any loader (SeDuMi format)
            
        Returns:
            CVXPY Problem object ready for solving
        """
        # Get cone structure from first-class field (Phase 2 enhancement)
        cone_structure = None
        if hasattr(problem_data, 'cone_structure') and problem_data.cone_structure:
            cone_structure = problem_data.cone_structure
        elif hasattr(problem_data, 'metadata') and 'cone_structure' in problem_data.metadata:
            # Backward compatibility fallback
            cone_structure = problem_data.metadata['cone_structure']
            self.logger.debug("Using cone_structure from metadata (legacy)")
        
        # Validate SeDuMi format inputs
        if problem_data.A_eq is None or problem_data.b_eq is None or problem_data.c is None:
            raise ValueError("SeDuMi format requires A_eq, b_eq, and c to be non-None")
        
        self.logger.debug(f"Converting SeDuMi problem with cone_structure: {cone_structure}")
        A_eq = problem_data.A_eq
        b_eq = problem_data.b_eq
        c = problem_data.c
        (m, n) = A_eq.shape
        y = cp.Variable(m, name="y")

        P = problem_data.P if hasattr(problem_data, 'P') else None
        obj_func = b_eq @ y - (cp.quad_form(y, P) / 2 if P is not None else 0)

        nvar_cnt = 0
        constraints = []
        if 'free_vars' in cone_structure:
            free_vars = cone_structure['free_vars']
            if free_vars:
                begin = nvar_cnt
                end = nvar_cnt + free_vars
                z = c[begin:end] - sum(A_eq[i, begin:end] * y[i] for i in range(m))
                constraints.append(z == 0)
                nvar_cnt = end
        if 'nonneg_vars' in cone_structure:
            nonneg_vars = cone_structure['nonneg_vars']
            if nonneg_vars:
                begin = nvar_cnt
                end = nvar_cnt + nonneg_vars
                z = c[begin:end] - sum(A_eq[i, begin:end] * y[i] for i in range(m))
                constraints.append(z >= 0)
                nvar_cnt = end
        if 'soc_cones' in cone_structure:
            soc_cones = cone_structure['soc_cones']
            for ndim in soc_cones:
                if ndim <= 0:
                    continue
                begin = nvar_cnt
                end = nvar_cnt + ndim
                z = c[begin:end] - (y.T @ A_eq[:, begin:end]).T
                constraints.append(cp.SOC(z[0], z[1:]))
                nvar_cnt = end
        if 'sdp_cones' in cone_structure:
            sdp_cones = cone_structure['sdp_cones']
            for ndim in sdp_cones:
                if ndim <= 0:
                    continue
                begin = nvar_cnt
                end = nvar_cnt + ndim * ndim
                z = c[begin:end].reshape(ndim, ndim) - cp.reshape(y.T @ A_eq[:, begin:end], (ndim, ndim), order='C')
                constraints.append(z >> 0)
                nvar_cnt = end

        cvx_problem = cp.Problem(cp.Maximize(obj_func), constraints)

        n_vars = nvar_cnt
        self.logger.debug(f"Converted {problem_data.problem_class} problem: {n_vars} vars, {len(constraints)} constraints")
        
        return cvx_problem

    def _create_result_from_cvxpy(self, cvx_problem: cp.Problem, start_time: float) -> SolverResult:
        """Create standardized result from CVXPY problem."""
        
        solve_time = time.time() - start_time
        
        # Map CVXPY status to standard status
        status_mapping = {
            cp.OPTIMAL: 'OPTIMAL',
            cp.INFEASIBLE: 'INFEASIBLE',
            cp.UNBOUNDED: 'UNBOUNDED',
            cp.INFEASIBLE_INACCURATE: 'INFEASIBLE',
            cp.UNBOUNDED_INACCURATE: 'UNBOUNDED',
            cp.OPTIMAL_INACCURATE: 'OPTIMAL'
        }
        
        status = status_mapping.get(cvx_problem.status, 'UNKNOWN')
        primal_objective_value = cvx_problem.value if cvx_problem.value is not None else None
        
        # Calculate dual objective if available
        dual_objective_value = None
        if hasattr(cvx_problem, 'dual_value') and cvx_problem.dual_value is not None:
            dual_objective_value = float(cvx_problem.dual_value)
        
        # Calculate duality gap if both values available
        duality_gap = None
        if primal_objective_value is not None and dual_objective_value is not None:
            duality_gap = abs(primal_objective_value - dual_objective_value)
        
        # Get iterations if available
        iterations = None
        if cvx_problem.solver_stats and hasattr(cvx_problem.solver_stats, 'num_iters'):
            iterations = cvx_problem.solver_stats.num_iters
        
        # Extract solver-specific information
        additional_info = {
            "cvxpy_status": cvx_problem.status,
            "backend_solver": self.backend,
            "solver_stats": cvx_problem.solver_stats.__dict__ if cvx_problem.solver_stats else None,
            "cvxpy_version": cp.__version__
        }
        
        # Add solution information if available
        try:
            if cvx_problem.variables:
                variables_list = list(cvx_problem.variables)
                if variables_list and variables_list[0].value is not None:
                    additional_info["solution_norm"] = float(np.linalg.norm(variables_list[0].value))
        except Exception:
            pass
        
        self.logger.debug(f"Solve completed: status={status}, "
                         f"objective={primal_objective_value}, time={solve_time:.3f}s")
        
        return SolverResult(
            solve_time=solve_time,
            status=status,
            primal_objective_value=primal_objective_value,
            dual_objective_value=dual_objective_value,
            duality_gap=duality_gap,
            primal_infeasibility=None,  # CVXPY doesn't provide this easily
            dual_infeasibility=None,
            iterations=iterations,
            solver_name=self.solver_name,
            solver_version=self.get_version(),
            additional_info=additional_info
        )
    
    def get_version(self) -> str:
        """Get CVXPY version information with backend package version."""
        backend_version = self._get_backend_version()
        return f"cvxpy-{cp.__version__}-{self.backend}-{backend_version}"
    
    def _get_backend_version(self) -> str:
        """Get version of the actual backend solver package."""
        try:
            # Map backend names to their package names and version attributes
            backend_version_map = {
                "CLARABEL": self._get_package_version("clarabel"),
                "SCS": self._get_package_version("scs"),
                "ECOS": self._get_package_version("ecos"),
                "OSQP": self._get_package_version("osqp"),
                "CVXOPT": self._get_package_version("cvxopt"),
                "SDPA": self._get_package_version("sdpa"),
                "SCIP": self._get_package_version("pyscipopt"),
                "HIGHS": self._get_package_version("highspy")
            }
            
            return backend_version_map.get(self.backend, "unknown")
            
        except Exception as e:
            self.logger.debug(f"Failed to get backend version for {self.backend}: {e}")
            return "unknown"
    
    def _get_package_version(self, package_name: str) -> str:
        """Get version of a specific Python package."""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except ImportError:
            # Fallback for Python < 3.8
            try:
                import pkg_resources
                return pkg_resources.get_distribution(package_name).version
            except:
                pass
        except:
            pass
        
        # Try direct import with __version__ attribute
        try:
            module = __import__(package_name)
            if hasattr(module, '__version__'):
                return module.__version__
        except:
            pass
        
        return "unknown"
    
    def validate_problem_compatibility(self, problem_data: ProblemData) -> bool:
        """Check if the solver can handle the given problem type."""
        return problem_data.problem_class in self.backend_capabilities["supported_problem_types"]


# Convenience function to create solvers with different backends
def create_cvxpy_solvers(verbose: bool = False) -> List[CvxpySolver]:
    """Create CVXPY solver instances for different available backends."""
    available_backends = cp.installed_solvers()
    solver_instances = []
    
    # Define open-source backends in order of preference
    open_source_backends = ["CLARABEL", "SCS", "ECOS", "OSQP"]
    
    for backend_name in open_source_backends:
        if backend_name in available_backends:
            try:
                solver_instance = CvxpySolver(backend=backend_name, verbose=verbose)
                solver_instances.append(solver_instance)
            except Exception as e:
                logger.warning(f"Failed to create solver for backend {backend_name}: {e}")
    
    return solver_instances


if __name__ == "__main__":
    # Test script to verify CVXPY solver
    print("Testing CVXPY Solver...")
    
    # Test available solvers
    print(f"\nAvailable CVXPY solvers: {cp.installed_solvers()}")
    
    # Test solver initialization
    print("\nTesting solver initialization:")
    try:
        solver = CvxpySolver(backend="CLARABEL")
        print(f"✓ {solver.solver_name} initialized")
        print(f"  Version: {solver.get_version()}")
        print(f"  Backend capabilities: {solver.backend_capabilities}")
        
    except Exception as e:
        print(f"✗ Failed to initialize CLARABEL: {e}")
        # Try fallback
        try:
            solver = CvxpySolver(backend="SCS")
            print(f"✓ Fallback {solver.solver_name} initialized")
        except Exception as e2:
            print(f"✗ Failed to initialize any solver: {e2}")
            exit(1)
    
    # Test LP solving
    print("\nTesting LP solving:")
    try:
        # Create a simple LP problem: minimize x1 + 2*x2 subject to x1 + x2 <= 1, x1, x2 >= 0
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
        if result.additional_info and 'backend_solver' in result.additional_info:
            print(f"  Backend: {result.additional_info['backend_solver']}")
        else:
            print(f"  Backend: {solver.backend}")
        
    except Exception as e:
        print(f"✗ LP test failed: {e}")
    
    # Test multiple solver creation
    print("\nTesting multiple solver creation:")
    try:
        solvers = create_cvxpy_solvers()
        print(f"Created {len(solvers)} CVXPY solver variants:")
        for s in solvers:
            print(f"  {s.solver_name}: {s.backend}")
    except Exception as e:
        print(f"✗ Multiple solver creation failed: {e}")
    
    print("\n✓ CVXPY solver test completed!")