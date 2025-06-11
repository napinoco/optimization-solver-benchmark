import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.problem_loader import load_problem, load_problem_registry
from scripts.benchmark.solver_interface import SolverInterface, SolverResult
from scripts.benchmark.result_collector import ResultCollector
from scripts.benchmark.environment_info import collect_environment_info
from scripts.utils.config_loader import load_config
from scripts.solvers.python.scipy_runner import ScipySolver
from scripts.solvers.python.cvxpy_runner import CvxpySolver, create_cvxpy_solvers
from scripts.utils.config_loader import load_benchmark_config, load_solvers_config
from scripts.utils.logger import get_logger
from scripts.utils.solver_validation import SolverValidator, ProblemType
from scripts.benchmark.backend_selector import BackendSelector, SelectionStrategy
from scripts.database.models import create_database

logger = get_logger("benchmark_runner")

class BenchmarkRunner:
    """Orchestrates the complete benchmark execution process."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize benchmark runner.
        
        Args:
            config_path: Path to benchmark configuration file
        """
        self.logger = get_logger("benchmark_runner")
        
        # Load configurations
        try:
            self.benchmark_config = load_benchmark_config()
            self.solver_config = load_solvers_config()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
        
        # Initialize components
        self.result_collector = ResultCollector()
        self.solvers = {}
        self.problems = {}
        
        # Backend validation and selection
        self.validator = SolverValidator()
        self.backend_selector = BackendSelector(self.validator)
        self.validation_results = None
        
        # Benchmark session tracking
        self.current_benchmark_id = None
        self.execution_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_time': 0.0
        }
        
        self.logger.info("Benchmark runner initialized")
    
    def validate_backends(self) -> Dict[str, Any]:
        """Validate all available CVXPY backends and return validation report."""
        self.logger.info("Validating CVXPY backends...")
        
        # Perform validation and cache results
        self.validation_results = self.validator.validate_all_backends()
        
        # Generate comprehensive validation report
        validation_report = self.validator.generate_validation_report(self.validation_results)
        
        # Log validation summary
        available_count = validation_report['summary']['available_backends']
        total_count = validation_report['summary']['total_backends']
        availability_pct = validation_report['summary']['availability_percentage']
        
        self.logger.info(f"Backend validation complete: {available_count}/{total_count} available ({availability_pct:.1f}%)")
        
        # Log problem type coverage
        for problem_type, coverage in validation_report['problem_type_coverage'].items():
            available_backends = coverage['available_backends']
            coverage_pct = coverage['coverage_percentage']
            recommended = coverage['recommended']
            backend_count = len(available_backends) if isinstance(available_backends, list) else available_backends
            self.logger.info(f"  {problem_type}: {backend_count} backends ({coverage_pct:.1f}%), recommended: {recommended}")
        
        return validation_report
    
    def select_optimal_backends(self, strategy: SelectionStrategy = SelectionStrategy.BALANCED) -> Dict[str, Any]:
        """Select optimal backends for comprehensive benchmarking."""
        self.logger.info(f"Selecting optimal backends using {strategy.value} strategy...")
        
        # Ensure validation is complete
        if self.validation_results is None:
            self.validate_backends()
        
        # Select backends for all problem types
        problem_types = [ProblemType.LP, ProblemType.QP, ProblemType.SOCP, ProblemType.SDP]
        selections = self.backend_selector.select_backends_for_benchmark(
            problem_types, strategy=strategy, max_backends_per_type=3
        )
        
        # Generate selection report
        selection_report = self.backend_selector.generate_selection_report(selections)
        
        # Log selection summary
        total_selections = selection_report['summary']['total_selections']
        unique_backends = selection_report['summary']['unique_backends_selected']
        selected_backends = selection_report['summary']['selected_backends']
        
        self.logger.info(f"Backend selection complete: {total_selections} selections, {unique_backends} unique backends")
        self.logger.info(f"Selected backends: {selected_backends}")
        
        return {
            'selections': selections,
            'report': selection_report,
            'strategy': strategy.value
        }
    
    def setup_solvers(self, use_validation: bool = True) -> None:
        """Initialize configured solvers from configuration file."""
        self.logger.info("Setting up solvers...")
        
        # Perform backend validation if requested
        if use_validation:
            validation_report = self.validate_backends()
            self.logger.info("Using backend validation for solver setup")
        
        # Load solver configuration
        try:
            solver_config = load_config("solvers.yaml")
            solver_definitions = solver_config.get("solvers", {})
        except Exception as e:
            self.logger.error(f"Failed to load solver configuration: {e}")
            raise
        
        timeout = self.benchmark_config.get('solver_timeout', 300.0)
        
        # Initialize each configured solver
        for solver_id, config in solver_definitions.items():
            if not config.get('enabled', True):
                self.logger.info(f"Skipping disabled solver: {solver_id}")
                continue
                
            try:
                solver_instance = self._create_solver_instance(solver_id, config, timeout)
                if solver_instance:
                    self.solvers[solver_id] = solver_instance
                    self.logger.info(f"Initialized solver: {solver_instance.name}")
                    
                    # Store solver info in database
                    solver_info = solver_instance.get_info()
                    self.result_collector.store_solver_info(
                        name=solver_instance.name,
                        version=solver_instance.get_version(),
                        environment=config.get('environment', 'unknown'),
                        metadata=solver_info
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize solver {solver_id}: {e}")
        
        if not self.solvers:
            raise RuntimeError("No solvers were successfully initialized")
        
        self.logger.info(f"Successfully initialized {len(self.solvers)} solvers")
    
    def _create_solver_instance(self, solver_id: str, config: dict, timeout: float):
        """Create a solver instance based on configuration."""
        environment = config.get('environment', 'python')
        
        if environment == 'python':
            if 'backend' in config:
                # CVXPY backend solver
                backend = config['backend']
                solver_name = config.get('name', f"{backend} (via CVXPY)")
                
                # Check if backend is available using validation results
                if self.validation_results and backend in self.validation_results:
                    validation_result = self.validation_results[backend]
                    if not validation_result.available:
                        self.logger.warning(f"Backend {backend} not available (validation: {validation_result.error_message}). Skipping {solver_id}")
                        return None
                    else:
                        self.logger.debug(f"Backend {backend} validated successfully")
                else:
                    # Fallback to direct CVXPY check if validation not available
                    import cvxpy as cp
                    available_backends = cp.installed_solvers()
                    if backend not in available_backends:
                        self.logger.warning(f"Backend {backend} not available. Skipping {solver_id}")
                        return None
                
                solver_instance = CvxpySolver(
                    name=solver_name,
                    backend=backend,
                    timeout=config.get('timeout', timeout),
                    verbose=False,
                    solver_options=config.get('solver_options', {}),
                    problem_optimizations=config.get('problem_optimizations', {}),
                    enable_diagnostics=config.get('diagnostics', {}).get('enabled', True)
                )
                return solver_instance
                
            elif solver_id == 'scipy':
                # SciPy solver
                solver_instance = ScipySolver(
                    name=config.get('name', 'SciPy'),
                    timeout=config.get('timeout', timeout)
                )
                return solver_instance
                
        self.logger.warning(f"Unknown solver configuration for {solver_id}")
        return None
    
    def load_problems(self, problem_set: str = "light_set") -> None:
        """Load problems from the registry."""
        self.logger.info(f"Loading problems from set: {problem_set}")
        
        try:
            registry = load_problem_registry()
            
            if problem_set not in registry["problems"]:
                raise ValueError(f"Problem set '{problem_set}' not found in registry")
            
            problem_count = 0
            for problem_class in registry["problems"][problem_set]:
                for problem_info in registry["problems"][problem_set][problem_class]:
                    problem_name = problem_info["name"]
                    
                    try:
                        problem = load_problem(problem_name, problem_set)
                        self.problems[problem_name] = problem
                        
                        # Store problem info in database
                        self.result_collector.store_problem_info(
                            name=problem.name,
                            problem_class=problem.problem_class,
                            file_path=problem_info["file_path"],
                            metadata={
                                "description": problem_info.get("description", ""),
                                "problem_set": problem_set,
                                "source": problem_info.get("source", "")
                            }
                        )
                        
                        problem_count += 1
                        self.logger.debug(f"Loaded problem: {problem_name} ({problem.problem_class})")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load problem {problem_name}: {e}")
            
            self.logger.info(f"Successfully loaded {problem_count} problems")
            
        except Exception as e:
            self.logger.error(f"Failed to load problems: {e}")
            raise
    
    def run_single_benchmark(self, solver_name: str, problem_name: str) -> SolverResult:
        """
        Run a single solver on a single problem.
        
        Args:
            solver_name: Name of the solver to use
            problem_name: Name of the problem to solve
            
        Returns:
            SolverResult from the benchmark run
        """
        if solver_name not in self.solvers:
            raise ValueError(f"Solver '{solver_name}' not available")
        
        if problem_name not in self.problems:
            raise ValueError(f"Problem '{problem_name}' not loaded")
        
        solver = self.solvers[solver_name]
        problem = self.problems[problem_name]
        
        self.logger.info(f"Running {solver_name} on {problem_name}")
        
        start_time = time.time()
        try:
            result = solver.solve_with_timeout(problem)
            self.execution_stats['successful_runs'] += 1
            
        except Exception as e:
            # Create error result if solver completely fails
            elapsed_time = time.time() - start_time
            error_msg = f"Benchmark execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            result = SolverResult(
                solver_name=solver_name,
                problem_name=problem_name,
                solve_time=elapsed_time,
                status='error',
                error_message=error_msg
            )
            self.execution_stats['failed_runs'] += 1
        
        # Update execution statistics
        self.execution_stats['total_runs'] += 1
        self.execution_stats['total_time'] += result.solve_time
        
        # Store result in database
        if self.current_benchmark_id:
            self.result_collector.store_result(self.current_benchmark_id, result)
        
        self.logger.info(f"Completed {solver_name} on {problem_name}: "
                        f"{result.status} in {result.solve_time:.3f}s")
        
        return result
    
    def run_sequential_benchmark(self, solver_names: Optional[List[str]] = None,
                                problem_names: Optional[List[str]] = None) -> List[SolverResult]:
        """
        Run benchmark sequentially (one at a time).
        
        Args:
            solver_names: List of solver names to run. If None, runs all available solvers.
            problem_names: List of problem names to run. If None, runs all loaded problems.
            
        Returns:
            List of SolverResults from all benchmark runs
        """
        # Use all available solvers/problems if not specified
        if solver_names is None:
            solver_names = list(self.solvers.keys())
        if problem_names is None:
            problem_names = list(self.problems.keys())
        
        total_runs = len(solver_names) * len(problem_names)
        self.logger.info(f"Starting sequential benchmark: {len(solver_names)} solvers × "
                        f"{len(problem_names)} problems = {total_runs} total runs")
        
        results = []
        run_count = 0
        
        for solver_name in solver_names:
            for problem_name in problem_names:
                run_count += 1
                self.logger.info(f"Progress: {run_count}/{total_runs} - "
                               f"{solver_name} on {problem_name}")
                
                try:
                    result = self.run_single_benchmark(solver_name, problem_name)
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to run {solver_name} on {problem_name}: {e}")
                    # Continue with next benchmark
        
        self.logger.info(f"Sequential benchmark completed: {len(results)} results")
        return results
    
    def run_parallel_benchmark(self, solver_names: Optional[List[str]] = None,
                              problem_names: Optional[List[str]] = None,
                              max_workers: Optional[int] = None) -> List[SolverResult]:
        """
        Run benchmark in parallel using ThreadPoolExecutor.
        
        Args:
            solver_names: List of solver names to run. If None, runs all available solvers.
            problem_names: List of problem names to run. If None, runs all loaded problems.
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of SolverResults from all benchmark runs
        """
        # Use all available solvers/problems if not specified
        if solver_names is None:
            solver_names = list(self.solvers.keys())
        if problem_names is None:
            problem_names = list(self.problems.keys())
        
        if max_workers is None:
            max_workers = self.benchmark_config.get('parallel_jobs', 2)
        
        # Create list of all (solver, problem) combinations
        benchmark_tasks = []
        for solver_name in solver_names:
            for problem_name in problem_names:
                benchmark_tasks.append((solver_name, problem_name))
        
        total_runs = len(benchmark_tasks)
        self.logger.info(f"Starting parallel benchmark: {total_runs} total runs "
                        f"with {max_workers} workers")
        
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_single_benchmark, solver_name, problem_name): 
                (solver_name, problem_name)
                for solver_name, problem_name in benchmark_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                solver_name, problem_name = future_to_task[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Progress: {completed_count}/{total_runs} - "
                                   f"Completed {solver_name} on {problem_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed {solver_name} on {problem_name}: {e}")
                    self.logger.debug(f"Exception details: {traceback.format_exc()}")
        
        self.logger.info(f"Parallel benchmark completed: {len(results)} results")
        return results
    
    def run_full_benchmark(self, problem_set: str = "light_set",
                          parallel: bool = True,
                          max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete benchmark including setup, execution, and summary.
        
        Args:
            problem_set: Problem set to run
            parallel: Whether to run in parallel or sequential
            max_workers: Maximum parallel workers (if parallel=True)
            
        Returns:
            Dictionary with benchmark summary and results
        """
        benchmark_start_time = time.time()
        
        try:
            # Initialize database
            self.logger.info("Initializing database...")
            create_database()
            
            # Create benchmark session
            self.logger.info("Creating benchmark session...")
            env_info = collect_environment_info()
            self.current_benchmark_id = self.result_collector.create_benchmark_session(env_info)
            
            # Setup phase
            self.logger.info("Setting up benchmark...")
            self.setup_solvers()
            self.load_problems(problem_set)
            
            if not self.solvers:
                raise RuntimeError("No solvers available")
            if not self.problems:
                raise RuntimeError("No problems loaded")
            
            # Execution phase
            self.logger.info("Starting benchmark execution...")
            if parallel:
                results = self.run_parallel_benchmark(max_workers=max_workers)
            else:
                results = self.run_sequential_benchmark()
            
            # Generate summary
            benchmark_time = time.time() - benchmark_start_time
            summary = self._generate_summary(results, benchmark_time)
            
            self.logger.info("Benchmark completed successfully!")
            self.logger.info(f"Total time: {benchmark_time:.1f}s, "
                           f"Results: {len(results)}, "
                           f"Success rate: {summary['success_rate']:.1%}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            self.logger.debug(f"Exception details: {traceback.format_exc()}")
            raise
    
    def _generate_summary(self, results: List[SolverResult], 
                         benchmark_time: float) -> Dict[str, Any]:
        """Generate benchmark summary statistics."""
        
        if not results:
            return {
                'total_runs': 0,
                'successful_runs': 0,
                'failed_runs': 0,
                'success_rate': 0.0,
                'total_benchmark_time': benchmark_time,
                'solver_statistics': {},
                'problem_statistics': {}
            }
        
        # Overall statistics
        successful_runs = sum(1 for r in results if r.is_successful())
        failed_runs = len(results) - successful_runs
        success_rate = successful_runs / len(results) if results else 0.0
        
        # Per-solver statistics
        solver_stats = {}
        for result in results:
            solver_name = result.solver_name
            if solver_name not in solver_stats:
                solver_stats[solver_name] = {
                    'total_runs': 0,
                    'successful_runs': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'problems_solved': set()
                }
            
            stats = solver_stats[solver_name]
            stats['total_runs'] += 1
            stats['total_time'] += result.solve_time
            stats['min_time'] = min(stats['min_time'], result.solve_time)
            stats['max_time'] = max(stats['max_time'], result.solve_time)
            
            if result.is_successful():
                stats['successful_runs'] += 1
                stats['problems_solved'].add(result.problem_name)
        
        # Calculate averages and clean up
        for solver_name, stats in solver_stats.items():
            if stats['total_runs'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['total_runs']
                stats['success_rate'] = stats['successful_runs'] / stats['total_runs']
            else:
                stats['avg_time'] = 0.0
                stats['success_rate'] = 0.0
            
            if stats['min_time'] == float('inf'):
                stats['min_time'] = 0.0
            
            # Convert set to count
            stats['unique_problems_solved'] = len(stats['problems_solved'])
            del stats['problems_solved']  # Remove set from final output
        
        # Per-problem statistics
        problem_stats = {}
        for result in results:
            problem_name = result.problem_name
            if problem_name not in problem_stats:
                problem_stats[problem_name] = {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'solver_results': {}
                }
            
            stats = problem_stats[problem_name]
            stats['total_attempts'] += 1
            
            if result.is_successful():
                stats['successful_attempts'] += 1
            
            stats['solver_results'][result.solver_name] = {
                'status': result.status,
                'solve_time': result.solve_time,
                'objective_value': result.objective_value
            }
        
        # Calculate problem success rates
        for problem_name, stats in problem_stats.items():
            if stats['total_attempts'] > 0:
                stats['success_rate'] = stats['successful_attempts'] / stats['total_attempts']
            else:
                stats['success_rate'] = 0.0
        
        return {
            'benchmark_id': self.current_benchmark_id,
            'total_runs': len(results),
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'success_rate': success_rate,
            'total_benchmark_time': benchmark_time,
            'avg_run_time': sum(r.solve_time for r in results) / len(results),
            'solver_statistics': solver_stats,
            'problem_statistics': problem_stats,
            'execution_stats': self.execution_stats.copy()
        }
    
    def get_benchmark_status(self) -> Dict[str, Any]:
        """Get current benchmark status and statistics."""
        return {
            'benchmark_id': self.current_benchmark_id,
            'solvers_loaded': len(self.solvers),
            'problems_loaded': len(self.problems),
            'execution_stats': self.execution_stats.copy(),
            'solver_names': list(self.solvers.keys()),
            'problem_names': list(self.problems.keys())
        }

if __name__ == "__main__":
    # Test script to verify benchmark runner
    try:
        print("Testing Benchmark Runner...")
        
        # Test runner initialization
        print("\nTesting runner initialization:")
        runner = BenchmarkRunner()
        print(f"✓ Runner initialized")
        
        # Test solver setup
        print("\nTesting solver setup:")
        runner.setup_solvers()
        status = runner.get_benchmark_status()
        print(f"✓ Solvers loaded: {status['solvers_loaded']}")
        print(f"  Available solvers: {status['solver_names']}")
        
        # Test problem loading
        print("\nTesting problem loading:")
        runner.load_problems("light_set")
        status = runner.get_benchmark_status()
        print(f"✓ Problems loaded: {status['problems_loaded']}")
        print(f"  Available problems: {status['problem_names']}")
        
        # Test single benchmark
        print("\nTesting single benchmark:")
        if runner.solvers and runner.problems:
            # Create a benchmark session for testing
            runner.current_benchmark_id = runner.result_collector.create_benchmark_session()
            
            solver_name = list(runner.solvers.keys())[0]
            problem_name = list(runner.problems.keys())[0]
            
            result = runner.run_single_benchmark(solver_name, problem_name)
            print(f"✓ Single benchmark completed: {result.solver_name} on {result.problem_name}")
            print(f"  Status: {result.status}, Time: {result.solve_time:.3f}s")
        
        # Test sequential benchmark (small subset)
        print("\nTesting sequential benchmark:")
        try:
            # Create new session for sequential test
            runner.current_benchmark_id = runner.result_collector.create_benchmark_session()
            
            # Run just one solver on available problems for testing
            solver_names = [list(runner.solvers.keys())[0]]
            results = runner.run_sequential_benchmark(solver_names=solver_names)
            print(f"✓ Sequential benchmark completed: {len(results)} results")
            
            for result in results[:3]:  # Show first 3 results
                print(f"  {result.solver_name} on {result.problem_name}: "
                      f"{result.status} ({result.solve_time:.3f}s)")
        
        except Exception as e:
            print(f"✗ Sequential benchmark test failed: {e}")
        
        # Test parallel benchmark (small subset)
        print("\nTesting parallel benchmark:")
        try:
            # Create new session for parallel test
            runner.current_benchmark_id = runner.result_collector.create_benchmark_session()
            
            # Run with 2 workers for testing
            solver_names = [list(runner.solvers.keys())[0]]
            results = runner.run_parallel_benchmark(solver_names=solver_names, max_workers=2)
            print(f"✓ Parallel benchmark completed: {len(results)} results")
            
        except Exception as e:
            print(f"✗ Parallel benchmark test failed: {e}")
        
        # Test full benchmark
        print("\nTesting full benchmark:")
        try:
            summary = runner.run_full_benchmark(parallel=False)  # Use sequential for testing
            print(f"✓ Full benchmark completed!")
            print(f"  Total runs: {summary['total_runs']}")
            print(f"  Success rate: {summary['success_rate']:.1%}")
            print(f"  Benchmark time: {summary['total_benchmark_time']:.1f}s")
            print(f"  Solvers: {list(summary['solver_statistics'].keys())}")
            
        except Exception as e:
            print(f"✗ Full benchmark test failed: {e}")
        
        print("\n✓ All benchmark runner tests completed!")
        
    except Exception as e:
        logger.error(f"Benchmark runner test failed: {e}")
        print(f"✗ Test failed: {e}")
        raise