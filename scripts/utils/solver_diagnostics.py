import sys
import time
import numpy as np
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger
from scripts.utils.solver_validation import SolverValidator, ProblemType
from scripts.benchmark.backend_selector import BackendSelector

logger = get_logger("solver_diagnostics")


@dataclass
class ParameterSpec:
    """Specification for a solver parameter."""
    name: str
    param_type: type
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_values: Optional[List[Any]] = None
    description: str = ""
    required: bool = False


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result for a solver."""
    solver_name: str
    problem_type: str
    problem_size: str
    solve_time: float
    iterations: Optional[int] = None
    objective_value: Optional[float] = None
    status: str = "unknown"


class SolverDiagnostics:
    """Comprehensive diagnostics system for CVXPY solver backends."""
    
    def __init__(self):
        self.logger = get_logger("solver_diagnostics")
        self.validator = SolverValidator()
        self.backend_selector = BackendSelector(self.validator)
        
        # Parameter specifications for each backend
        self.parameter_specs = self._load_parameter_specifications()
        
        # Performance test problems
        self.test_problems = self._create_test_problems()
    
    def _load_parameter_specifications(self) -> Dict[str, Dict[str, ParameterSpec]]:
        """Load parameter specifications for each backend."""
        specs = {
            "CLARABEL": {
                "feastol": ParameterSpec(
                    name="feastol", param_type=float, default_value=1.0e-8,
                    min_value=1.0e-12, max_value=1.0e-3,
                    description="Feasibility tolerance"
                ),
                "abstol": ParameterSpec(
                    name="abstol", param_type=float, default_value=1.0e-8,
                    min_value=1.0e-12, max_value=1.0e-3,
                    description="Absolute tolerance"
                ),
                "reltol": ParameterSpec(
                    name="reltol", param_type=float, default_value=1.0e-8,
                    min_value=1.0e-12, max_value=1.0e-3,
                    description="Relative tolerance"
                ),
                "max_iter": ParameterSpec(
                    name="max_iter", param_type=int, default_value=50000,
                    min_value=1, max_value=1000000,
                    description="Maximum iterations"
                ),
                "verbose": ParameterSpec(
                    name="verbose", param_type=bool, default_value=False,
                    description="Enable verbose output"
                )
            },
            "OSQP": {
                "eps_abs": ParameterSpec(
                    name="eps_abs", param_type=float, default_value=1.0e-6,
                    min_value=1.0e-12, max_value=1.0e-2,
                    description="Absolute tolerance"
                ),
                "eps_rel": ParameterSpec(
                    name="eps_rel", param_type=float, default_value=1.0e-6,
                    min_value=1.0e-12, max_value=1.0e-2,
                    description="Relative tolerance"
                ),
                "max_iter": ParameterSpec(
                    name="max_iter", param_type=int, default_value=10000,
                    min_value=1, max_value=1000000,
                    description="Maximum iterations"
                ),
                "rho": ParameterSpec(
                    name="rho", param_type=float, default_value=0.1,
                    min_value=1.0e-6, max_value=1000.0,
                    description="ADMM step size"
                ),
                "alpha": ParameterSpec(
                    name="alpha", param_type=float, default_value=1.6,
                    min_value=0.1, max_value=2.0,
                    description="Relaxation parameter"
                ),
                "polish": ParameterSpec(
                    name="polish", param_type=bool, default_value=True,
                    description="Polish solution at the end"
                ),
                "verbose": ParameterSpec(
                    name="verbose", param_type=bool, default_value=False,
                    description="Enable verbose output"
                )
            },
            "SCS": {
                "eps": ParameterSpec(
                    name="eps", param_type=float, default_value=1.0e-5,
                    min_value=1.0e-12, max_value=1.0e-2,
                    description="Convergence tolerance"
                ),
                "max_iters": ParameterSpec(
                    name="max_iters", param_type=int, default_value=100000,
                    min_value=1, max_value=1000000,
                    description="Maximum iterations"
                ),
                "alpha": ParameterSpec(
                    name="alpha", param_type=float, default_value=1.5,
                    min_value=0.1, max_value=2.0,
                    description="Relaxation parameter"
                ),
                "normalize": ParameterSpec(
                    name="normalize", param_type=bool, default_value=True,
                    description="Normalize problem data"
                ),
                "verbose": ParameterSpec(
                    name="verbose", param_type=bool, default_value=False,
                    description="Enable verbose output"
                )
            },
            "ECOS": {
                "feastol": ParameterSpec(
                    name="feastol", param_type=float, default_value=1.0e-8,
                    min_value=1.0e-12, max_value=1.0e-3,
                    description="Feasibility tolerance"
                ),
                "abstol": ParameterSpec(
                    name="abstol", param_type=float, default_value=1.0e-8,
                    min_value=1.0e-12, max_value=1.0e-3,
                    description="Absolute tolerance"
                ),
                "reltol": ParameterSpec(
                    name="reltol", param_type=float, default_value=1.0e-8,
                    min_value=1.0e-12, max_value=1.0e-3,
                    description="Relative tolerance"
                ),
                "max_iters": ParameterSpec(
                    name="max_iters", param_type=int, default_value=100,
                    min_value=1, max_value=10000,
                    description="Maximum iterations"
                ),
                "verbose": ParameterSpec(
                    name="verbose", param_type=bool, default_value=False,
                    description="Enable verbose output"
                )
            }
        }
        
        return specs
    
    def _create_test_problems(self) -> Dict[str, Dict[str, Any]]:
        """Create test problems for performance benchmarking."""
        
        problems = {}
        
        # Small LP problem
        problems["small_lp"] = {
            "type": "LP",
            "size": "small",
            "variables": 10,
            "constraints": 5,
            "generator": self._generate_small_lp
        }
        
        # Medium LP problem
        problems["medium_lp"] = {
            "type": "LP", 
            "size": "medium",
            "variables": 100,
            "constraints": 50,
            "generator": self._generate_medium_lp
        }
        
        # Small QP problem
        problems["small_qp"] = {
            "type": "QP",
            "size": "small", 
            "variables": 10,
            "constraints": 5,
            "generator": self._generate_small_qp
        }
        
        # Medium QP problem
        problems["medium_qp"] = {
            "type": "QP",
            "size": "medium",
            "variables": 50,
            "constraints": 25,
            "generator": self._generate_medium_qp
        }
        
        return problems
    
    def _generate_small_lp(self):
        """Generate small LP test problem."""
        try:
            import cvxpy as cp
            
            # Variables
            x = cp.Variable(10)
            
            # Objective: minimize sum of variables
            objective = cp.Minimize(cp.sum(x))
            
            # Constraints
            constraints = [
                cp.sum(x) >= 1,
                x >= 0,
                x <= 10
            ]
            
            return cp.Problem(objective, constraints), x
        except ImportError:
            return None, None
    
    def _generate_medium_lp(self):
        """Generate medium LP test problem."""
        try:
            import cvxpy as cp
            
            # Variables
            x = cp.Variable(100)
            
            # Random cost vector
            np.random.seed(42)
            c = np.random.randn(100)
            
            # Objective
            objective = cp.Minimize(c.T @ x)
            
            # Random constraints
            A = np.random.randn(50, 100)
            b = np.random.randn(50)
            
            constraints = [
                A @ x <= b,
                x >= 0
            ]
            
            return cp.Problem(objective, constraints), x
        except ImportError:
            return None, None
    
    def _generate_small_qp(self):
        """Generate small QP test problem."""
        try:
            import cvxpy as cp
            
            # Variables
            x = cp.Variable(10)
            
            # Quadratic objective
            P = np.eye(10)
            q = np.ones(10)
            
            objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
            
            # Constraints
            constraints = [
                cp.sum(x) == 1,
                x >= 0
            ]
            
            return cp.Problem(objective, constraints), x
        except ImportError:
            return None, None
    
    def _generate_medium_qp(self):
        """Generate medium QP test problem."""
        try:
            import cvxpy as cp
            
            # Variables
            x = cp.Variable(50)
            
            # Random quadratic objective
            np.random.seed(42)
            P = np.random.randn(50, 50)
            P = P.T @ P  # Make positive semidefinite
            q = np.random.randn(50)
            
            objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
            
            # Random constraints
            A = np.random.randn(25, 50)
            b = np.random.randn(25)
            
            constraints = [
                A @ x <= b,
                x >= 0
            ]
            
            return cp.Problem(objective, constraints), x
        except ImportError:
            return None, None
    
    def validate_backend_parameters(self, backend: str, parameters: Dict[str, Any]) -> List[DiagnosticResult]:
        """Validate parameters for a specific backend."""
        results = []
        start_time = time.time()
        
        if backend not in self.parameter_specs:
            results.append(DiagnosticResult(
                check_name=f"parameter_validation_{backend}",
                passed=False,
                message=f"No parameter specifications available for backend {backend}",
                execution_time=time.time() - start_time
            ))
            return results
        
        specs = self.parameter_specs[backend]
        
        for param_name, param_value in parameters.items():
            param_start = time.time()
            
            if param_name not in specs:
                results.append(DiagnosticResult(
                    check_name=f"unknown_parameter_{backend}_{param_name}",
                    passed=False,
                    message=f"Unknown parameter '{param_name}' for backend {backend}",
                    execution_time=time.time() - param_start
                ))
                continue
            
            spec = specs[param_name]
            
            # Type validation
            if not isinstance(param_value, spec.param_type):
                results.append(DiagnosticResult(
                    check_name=f"type_validation_{backend}_{param_name}",
                    passed=False,
                    message=f"Parameter '{param_name}' must be of type {spec.param_type.__name__}, got {type(param_value).__name__}",
                    execution_time=time.time() - param_start
                ))
                continue
            
            # Range validation for numeric types
            if spec.param_type in [int, float]:
                if spec.min_value is not None and param_value < spec.min_value:
                    results.append(DiagnosticResult(
                        check_name=f"range_validation_{backend}_{param_name}",
                        passed=False,
                        message=f"Parameter '{param_name}' value {param_value} is below minimum {spec.min_value}",
                        execution_time=time.time() - param_start
                    ))
                    continue
                
                if spec.max_value is not None and param_value > spec.max_value:
                    results.append(DiagnosticResult(
                        check_name=f"range_validation_{backend}_{param_name}",
                        passed=False,
                        message=f"Parameter '{param_name}' value {param_value} is above maximum {spec.max_value}",
                        execution_time=time.time() - param_start
                    ))
                    continue
            
            # Valid values validation
            if spec.valid_values is not None and param_value not in spec.valid_values:
                results.append(DiagnosticResult(
                    check_name=f"valid_values_{backend}_{param_name}",
                    passed=False,
                    message=f"Parameter '{param_name}' value {param_value} not in valid values {spec.valid_values}",
                    execution_time=time.time() - param_start
                ))
                continue
            
            # Parameter validation passed
            results.append(DiagnosticResult(
                check_name=f"parameter_validation_{backend}_{param_name}",
                passed=True,
                message=f"Parameter '{param_name}' validation passed",
                details={"value": param_value, "spec": spec.__dict__},
                execution_time=time.time() - param_start
            ))
        
        return results
    
    def run_installation_diagnostics(self, backend: str) -> List[DiagnosticResult]:
        """Run installation diagnostics for a backend."""
        results = []
        start_time = time.time()
        
        # Basic availability check
        validation_result = self.validator.validate_backend(backend)
        
        if validation_result.available:
            results.append(DiagnosticResult(
                check_name=f"installation_{backend}",
                passed=True,
                message=f"Backend {backend} is installed and available",
                details={
                    "capabilities": validation_result.capabilities.__dict__ if validation_result.capabilities else None,
                    "validation_time": validation_result.validation_time
                },
                execution_time=time.time() - start_time
            ))
            
            # Version check if available
            if validation_result.capabilities and validation_result.capabilities.version:
                results.append(DiagnosticResult(
                    check_name=f"version_{backend}",
                    passed=True,
                    message=f"Backend {backend} version: {validation_result.capabilities.version}",
                    details={"version": validation_result.capabilities.version},
                    execution_time=time.time() - start_time
                ))
        else:
            results.append(DiagnosticResult(
                check_name=f"installation_{backend}",
                passed=False,
                message=f"Backend {backend} is not available: {validation_result.error_message}",
                details={"error": validation_result.error_message},
                execution_time=time.time() - start_time
            ))
        
        return results
    
    def run_performance_benchmarks(self, backend: str) -> List[PerformanceBenchmark]:
        """Run performance benchmarks for a backend."""
        benchmarks = []
        
        # Check if backend is available
        validation_result = self.validator.validate_backend(backend)
        if not validation_result.available:
            self.logger.warning(f"Backend {backend} not available for benchmarking")
            return benchmarks
        
        # Get supported problem types for this backend
        if validation_result.capabilities:
            supported_types = [pt.value for pt in validation_result.capabilities.supported_types]
        else:
            supported_types = ["LP", "QP"]  # Default assumption
        
        try:
            import cvxpy as cp
            from scripts.solvers.python.cvxpy_runner import CvxpySolver
            
            # Create solver instance
            solver = CvxpySolver(backend=backend, verbose=False)
            
            # Run benchmarks on test problems
            for problem_name, problem_info in self.test_problems.items():
                if problem_info["type"] not in supported_types:
                    continue
                
                self.logger.info(f"Running {backend} benchmark on {problem_name}")
                
                # Generate test problem
                start_time = time.time()
                problem, variables = problem_info["generator"]()
                
                if problem is None:
                    continue
                
                # Solve with the backend
                try:
                    problem.solve(solver=getattr(cp, backend), verbose=False)
                    solve_time = time.time() - start_time
                    
                    benchmark = PerformanceBenchmark(
                        solver_name=f"{backend} (via CVXPY)",
                        problem_type=problem_info["type"],
                        problem_size=problem_info["size"],
                        solve_time=solve_time,
                        objective_value=problem.value,
                        status=problem.status
                    )
                    
                    # Try to get iterations if available
                    if hasattr(problem, 'solver_stats') and problem.solver_stats:
                        if hasattr(problem.solver_stats, 'num_iters'):
                            benchmark.iterations = problem.solver_stats.num_iters
                    
                    benchmarks.append(benchmark)
                    
                except Exception as e:
                    self.logger.warning(f"Benchmark failed for {backend} on {problem_name}: {e}")
                    
                    benchmark = PerformanceBenchmark(
                        solver_name=f"{backend} (via CVXPY)",
                        problem_type=problem_info["type"],
                        problem_size=problem_info["size"],
                        solve_time=time.time() - start_time,
                        status="error"
                    )
                    benchmarks.append(benchmark)
        
        except ImportError:
            self.logger.error("CVXPY not available for benchmarking")
        
        return benchmarks
    
    def run_comprehensive_diagnostics(self, backend: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive diagnostics for a backend."""
        self.logger.info(f"Running comprehensive diagnostics for {backend}")
        
        diagnostics = {
            "backend": backend,
            "timestamp": time.time(),
            "installation_checks": [],
            "parameter_validation": [],
            "performance_benchmarks": [],
            "summary": {}
        }
        
        # Installation diagnostics
        diagnostics["installation_checks"] = self.run_installation_diagnostics(backend)
        
        # Parameter validation if parameters provided
        if parameters:
            diagnostics["parameter_validation"] = self.validate_backend_parameters(backend, parameters)
        
        # Performance benchmarks
        diagnostics["performance_benchmarks"] = self.run_performance_benchmarks(backend)
        
        # Generate summary
        installation_passed = all(check.passed for check in diagnostics["installation_checks"])
        parameter_passed = all(check.passed for check in diagnostics["parameter_validation"]) if parameters else True
        benchmark_count = len(diagnostics["performance_benchmarks"])
        successful_benchmarks = sum(1 for b in diagnostics["performance_benchmarks"] if b.status != "error")
        
        diagnostics["summary"] = {
            "installation_status": "passed" if installation_passed else "failed",
            "parameter_validation_status": "passed" if parameter_passed else "failed",
            "benchmarks_run": benchmark_count,
            "successful_benchmarks": successful_benchmarks,
            "benchmark_success_rate": successful_benchmarks / benchmark_count if benchmark_count > 0 else 0.0,
            "overall_status": "healthy" if installation_passed and parameter_passed and (successful_benchmarks / benchmark_count if benchmark_count > 0 else 1.0) > 0.5 else "issues_detected"
        }
        
        return diagnostics
    
    def generate_configuration_template(self, backend: str, problem_type: Optional[str] = None) -> Dict[str, Any]:
        """Generate configuration template for a backend."""
        
        if backend not in self.parameter_specs:
            return {}
        
        template = {
            "name": f"{backend} (via CVXPY)",
            "environment": "python",
            "backend": backend,
            "solver_options": {},
            "timeout": 300,
            "enabled": True
        }
        
        # Add default parameters
        specs = self.parameter_specs[backend]
        for param_name, spec in specs.items():
            template["solver_options"][param_name] = spec.default_value
        
        # Add problem-specific optimizations if specified
        if problem_type:
            template["solver_options"].update(self._get_problem_specific_optimizations(backend, problem_type))
        
        return template
    
    def _get_problem_specific_optimizations(self, backend: str, problem_type: str) -> Dict[str, Any]:
        """Get problem-specific parameter optimizations."""
        optimizations = {
            "CLARABEL": {
                "LP": {"presolve_enable": True, "equilibrate_enable": True},
                "QP": {"sigma": 1.0, "alpha": 1.4},
                "SOCP": {"reduced_tol_enable": True, "max_iter": 25000},
                "SDP": {"chordal_decomposition_enable": True, "feastol": 1.0e-6}
            },
            "OSQP": {
                "QP": {"rho": 0.1, "alpha": 1.6, "polish": True},
                "SOCP": {"rho": 0.05, "eps_abs": 1.0e-7, "max_iter": 25000}
            },
            "SCS": {
                "LP": {"eps": 1.0e-6, "normalize": True, "use_indirect": False},
                "QP": {"alpha": 1.6, "adaptive_scale": True},
                "SOCP": {"eps": 1.0e-6, "max_iters": 50000},
                "SDP": {"eps": 1.0e-4, "max_iters": 25000, "use_indirect": True}
            },
            "ECOS": {
                "LP": {"feastol": 1.0e-9, "abstol": 1.0e-9, "max_iters": 200},
                "QP": {"feastol": 1.0e-8, "reltol": 1.0e-8, "nitref": 5},
                "SOCP": {"feastol": 1.0e-7, "abstol": 1.0e-7, "max_iters": 100}
            }
        }
        
        return optimizations.get(backend, {}).get(problem_type, {})


if __name__ == "__main__":
    # Test script to verify solver diagnostics
    try:
        print("Testing Solver Diagnostics System...")
        
        diagnostics = SolverDiagnostics()
        
        # Test parameter validation
        print("\nTesting parameter validation:")
        test_params = {
            "feastol": 1.0e-8,
            "max_iter": 50000,
            "verbose": False,
            "invalid_param": "should_fail"
        }
        
        validation_results = diagnostics.validate_backend_parameters("CLARABEL", test_params)
        for result in validation_results:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.check_name}: {result.message}")
        
        # Test installation diagnostics
        print("\nTesting installation diagnostics:")
        for backend in ["CLARABEL", "OSQP", "SCS", "ECOS"]:
            install_results = diagnostics.run_installation_diagnostics(backend)
            for result in install_results:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {backend}: {result.message}")
        
        # Test performance benchmarks
        print("\nTesting performance benchmarks:")
        benchmarks = diagnostics.run_performance_benchmarks("CLARABEL")
        for benchmark in benchmarks:
            print(f"  {benchmark.solver_name} on {benchmark.problem_type} ({benchmark.problem_size}): "
                  f"{benchmark.solve_time:.4f}s, status: {benchmark.status}")
        
        # Test comprehensive diagnostics
        print("\nTesting comprehensive diagnostics:")
        comprehensive = diagnostics.run_comprehensive_diagnostics("CLARABEL", test_params)
        summary = comprehensive["summary"]
        print(f"  Overall status: {summary['overall_status']}")
        print(f"  Installation: {summary['installation_status']}")
        print(f"  Parameters: {summary['parameter_validation_status']}")
        print(f"  Benchmarks: {summary['successful_benchmarks']}/{summary['benchmarks_run']} successful")
        
        # Test configuration template generation
        print("\nTesting configuration template generation:")
        template = diagnostics.generate_configuration_template("CLARABEL", "QP")
        print(f"  Generated template for CLARABEL/QP with {len(template['solver_options'])} parameters")
        
        print("\n✓ Solver diagnostics system test completed!")
        
    except Exception as e:
        logger.error(f"Solver diagnostics test failed: {e}")
        raise