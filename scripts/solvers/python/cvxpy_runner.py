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
from datetime import datetime

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
                 solver_options: Optional[Dict] = None, save_solutions: bool = False, **kwargs):
        """
        Initialize CVXPY solver with specific backend.
        
        Args:
            backend: CVXPY backend solver (CLARABEL, OSQP, SCS, etc.)
            verbose: Whether to enable verbose solver output
            solver_options: Backend-specific solver options
            save_solutions: Whether to save optimal solutions to disk
            **kwargs: Additional configuration parameters
        """
        # Auto-generate name with proper format
        solver_name = f"cvxpy_{backend.lower()}"
        
        super().__init__(solver_name, backend=backend, verbose=verbose, 
                        solver_options=solver_options, **kwargs)
        self.backend = backend
        self.verbose = verbose
        self.solver_options = solver_options or {}
        self.save_solutions = save_solutions
        
        # Setup solution storage directory
        if self.save_solutions:
            self.solutions_dir = Path(__file__).parent.parent.parent.parent / "problems" / "solutions"
            self.solutions_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify solver availability - no fallbacks for pure benchmarking
        available_solvers = cp.installed_solvers()
        if backend not in available_solvers:
            raise RuntimeError(f"Requested backend {backend} not available. "
                             f"Available backends: {available_solvers}")
        
        # Backend capabilities determined dynamically
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
            
            return self._create_result_from_cvxpy(cvx_problem, start_time, problem_data)
                
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
        y = cp.Variable((m, 1) , name="y")

        P = problem_data.P if hasattr(problem_data, 'P') else None
        obj_func = b_eq.T @ y - (cp.quad_form(y, P) / 2 if P is not None else 0)

        nvar_cnt = 0
        constraints = []
        cmAty = c - (y.T @ A_eq).T
        if 'free_vars' in cone_structure:
            free_vars = cone_structure['free_vars']
            if free_vars:
                begin = nvar_cnt
                end = nvar_cnt + free_vars
                z = cmAty[begin:end]
                constraints.append(z[:, 0] == 0)
                nvar_cnt = end
        if 'nonneg_vars' in cone_structure:
            nonneg_vars = cone_structure['nonneg_vars']
            if nonneg_vars:
                begin = nvar_cnt
                end = nvar_cnt + nonneg_vars
                z = cmAty[begin:end]
                constraints.append(z[:, 0] >= 0)
                nvar_cnt = end
        if 'soc_cones' in cone_structure:
            soc_cones = cone_structure['soc_cones']
            for ndim in soc_cones:
                if ndim <= 0:
                    continue
                begin = nvar_cnt
                end = nvar_cnt + ndim
                z = cmAty[begin:end]
                constraints.append(cp.SOC(z[0, 0], z[1:, 0]))
                nvar_cnt = end
        if 'sdp_cones' in cone_structure:
            sdp_cones = cone_structure['sdp_cones']
            for ndim in sdp_cones:
                if ndim <= 0:
                    continue
                begin = nvar_cnt
                end = nvar_cnt + ndim * ndim
                z = cp.reshape(cmAty[begin:end], (ndim, ndim), order='C')
                constraints.append(z >> 0)
                nvar_cnt = end

        cvx_problem = cp.Problem(cp.Maximize(obj_func), constraints)

        n_vars = nvar_cnt
        self.logger.debug(f"Converted {problem_data.problem_class} problem: {n_vars} vars, {len(constraints)} constraints")
        
        return cvx_problem

    def _create_result_from_cvxpy(self, cvx_problem: cp.Problem, start_time: float, problem_data: ProblemData) -> SolverResult:
        """Create standardized result from CVXPY problem with manual duality calculations."""
        
        solve_time = time.time() - start_time
        
        # Map CVXPY status to standard status
        status_mapping = {
            cp.OPTIMAL: 'OPTIMAL',
            cp.INFEASIBLE: 'INFEASIBLE',
            cp.UNBOUNDED: 'UNBOUNDED',
            cp.INFEASIBLE_INACCURATE: 'INFEASIBLE (INACCURATE)',
            cp.UNBOUNDED_INACCURATE: 'UNBOUNDED (INACCURATE)',
            cp.OPTIMAL_INACCURATE: 'OPTIMAL (INACCURATE)',
        }
        
        status = status_mapping.get(cvx_problem.status, 'UNKNOWN')

        # Manual duality calculations for unified comparison
        primal_objective_value = None
        primal_infeasibility = None
        dual_objective_value = None
        dual_infeasibility = None
        duality_gap = None

        try:
            A_eq = problem_data.A_eq
            b_eq = problem_data.b_eq
            c = problem_data.c
            cone_structure = problem_data.cone_structure

            # Get primal solution of SeDuMi format (=dual solution of CVXPY)
            x_list = []
            for constraint in cvx_problem.constraints:
                if not hasattr(constraint, 'dual_value') or constraint.dual_value is None:
                    x_list = []
                    break
                if isinstance(constraint, cp.SOC):
                    x_list.append(constraint.dual_value[0].reshape(-1, 1))
                    x_list.append(constraint.dual_value[1].reshape(-1, 1))
                else:
                    x_list.append(constraint.dual_value.reshape(-1, 1))

            if x_list:
                x = np.vstack(x_list)
                primal_objective_value = float(c.T @ x)
                # Primal infeasibility: ||A_eq @ x - b_eq|| / (1 + ||b_eq||)
                primal_residual = A_eq @ x - b_eq
                primal_infeasibility = float(np.linalg.norm(primal_residual) / (1 + np.linalg.norm(b_eq)))
            else:
                x = None

            # Get dual solution of SeDuMi format (=primal solution of CVXPY)
            y = cvx_problem.variables()[0].value.reshape(-1, 1)
            dual_objective_value = float(b_eq.T @ y)  # equals to cvx_problem.value

            # Dual infeasibility: ||z - (c - A_eq.T @ y)|| / (1 + ||c||)
            nvar_cnt = 0
            cmAty = c - A_eq.T @ y
            dinf2 = 0  # similar to np.sum(constraint.violation() ** 2 for constraint in cvx_problem.constraints)
            if 'free_vars' in cone_structure:
                free_vars = cone_structure['free_vars']
                if free_vars:
                    begin = nvar_cnt
                    end = nvar_cnt + free_vars
                    dinf2 += np.linalg.norm(cmAty[begin:end], ord=2) ** 2
                    nvar_cnt = end
            if 'nonneg_vars' in cone_structure:
                nonneg_vars = cone_structure['nonneg_vars']
                if nonneg_vars:
                    begin = nvar_cnt
                    end = nvar_cnt + nonneg_vars
                    dinf2 += np.linalg.norm(np.minimum(cmAty[begin:end], 0), ord=2) ** 2
                    nvar_cnt = end
            if 'soc_cones' in cone_structure:
                def proj_onto_soc(z):
                    z0 = z[0]
                    znorm = np.linalg.norm(z[1:], ord=2)
                    if znorm <= z0:
                        return z
                    elif znorm <= -z0:
                        return np.zeros_like(z)
                    else:
                        scale = (z0 + znorm) / 2
                        return np.concatenate(([1], z[1:] / znorm)) * scale
                soc_cones = cone_structure['soc_cones']
                for ndim in soc_cones:
                    if ndim <= 0:
                        continue
                    begin = nvar_cnt
                    end = nvar_cnt + ndim
                    dinf2 += np.linalg.norm(proj_onto_soc(-cmAty[begin:end]), ord=2) ** 2  # Pi_{K*}(-z) = z - Pi_K(z)
                    nvar_cnt = end
            if 'sdp_cones' in cone_structure:
                sdp_cones = cone_structure['sdp_cones']
                for ndim in sdp_cones:
                    if ndim <= 0:
                        continue
                    begin = nvar_cnt
                    end = nvar_cnt + ndim * ndim
                    eigvals = np.linalg.eigvalsh(cmAty[begin:end].reshape(ndim, ndim))
                    neg_eigvals = np.minimum(eigvals, 0)
                    dinf2 += np.sum(neg_eigvals ** 2)
                    nvar_cnt = end
            dual_infeasibility = float(np.sqrt(dinf2) / (1 + np.sum(c ** 2)))

            if primal_objective_value is not None and dual_objective_value is not None:
                duality_gap = primal_objective_value - dual_objective_value

            # Save solution if requested
            if self.save_solutions:
                self._save_solution(problem_data, x, y)

        except Exception as e:
            self.logger.debug(f"Manual duality calculation failed: {e}")
        
        # Get iterations and solver time if available
        iterations = None
        solver_time = None
        if cvx_problem.solver_stats:
            if hasattr(cvx_problem.solver_stats, 'num_iters'):
                iterations = cvx_problem.solver_stats.num_iters
            
            # Get solver time if available
            if hasattr(cvx_problem.solver_stats, 'solve_time'):
                solver_time = cvx_problem.solver_stats.solve_time
                self.logger.debug(f"Extracted solver time: {solver_time:.6f}s vs wall clock: {solve_time:.6f}s")
        
        # Use solver time if available, otherwise fall back to wall clock time
        reported_solve_time = solver_time if solver_time is not None else solve_time
        
        # Extract solver-specific information
        additional_info = {
            "cvxpy_status": cvx_problem.status,
            "backend_solver": self.backend,
            "solver_stats": cvx_problem.solver_stats.__dict__ if cvx_problem.solver_stats else None,
            "cvxpy_version": cp.__version__,
            "manual_duality_used": True
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
                         f"objective={primal_objective_value}, time={solve_time:.3f}s, "
                         f"dual_gap={duality_gap}")
        
        return SolverResult(
            solve_time=reported_solve_time,
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
                "SDPA": self._get_package_version("sdpa-python"),
                "SCIP": self._get_package_version("pyscipopt"),
                "HIGHS": self._get_package_version("highspy")
            }
            
            return backend_version_map.get(self.backend, "unknown")
            
        except Exception as e:
            self.logger.debug(f"Failed to get backend version for {self.backend}: {e}")
            return "unknown"
    
    def _save_solution(self, problem_data: ProblemData, primal_solution: np.ndarray, dual_solution: np.ndarray) -> None:
        """Save optimal solution to disk for verification and analysis."""
        try:
            # Create library-specific directory
            out_dir = self.solutions_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename: problemname_solvername.npz
            problem_name = problem_data.name.replace('/', '_').replace('\\', '_')
            filename = f"{problem_name}_{self.solver_name}.npz"
            filepath = out_dir / filename
            
            # Save solution data
            np.savez_compressed(
                filepath,
                primal_solution=primal_solution,
                dual_solution=dual_solution
            )
            
            self.logger.debug(f"Saved solution to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save solution for {problem_data.name}: {e}")
    
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

