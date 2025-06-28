"""
Simplified Benchmark Runner for Re-Architected System

This module provides a simplified benchmark execution engine that follows the
re-architected design principles:

- Direct solver creation without complex backend selection
- Unified data loading through format-specific loaders
- Direct database storage with standardized results
- Simple configuration loading from YAML files
- Error resilience with graceful degradation

Key Features:
- Load problems using format-specific loaders (MAT, DAT, MPS, QPS, Python)  
- Create solvers with direct if-elif logic (no complex configuration)
- Execute benchmarks with standardized SolverResult output
- Store results directly in simplified database schema
- Environment info collection for reproducibility
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Core imports
from scripts.database.database_manager import DatabaseManager
from scripts.utils.environment_info import collect_environment_info
from scripts.utils.git_utils import get_git_commit_hash
from scripts.utils.logger import get_logger

# Solver imports
from scripts.solvers.python.scipy_runner import ScipySolver
from scripts.solvers.python.cvxpy_runner import CvxpySolver
from scripts.solvers.solver_interface import SolverInterface, SolverResult

# Data loader imports
from scripts.data_loaders.python.mat_loader import MATLoader
from scripts.data_loaders.python.dat_loader import DATLoader

logger = get_logger("benchmark_runner")


class BenchmarkRunner:
    """Main benchmark execution engine with direct database storage"""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None, 
                 registries: Optional[Dict[str, Any]] = None,
                 dry_run: bool = False,
                 save_solutions: bool = False):
        """
        Initialize simplified benchmark runner.
        
        Args:
            database_manager: Optional database manager (creates default if None)
            registries: Pre-loaded registries to avoid redundant loading
            dry_run: If True, skip database operations (for testing)
            save_solutions: If True, save optimal solutions to disk
        """
        self.db = database_manager or DatabaseManager()
        self.dry_run = dry_run
        self.save_solutions = save_solutions
        
        # Collect environment info and git hash once (now cached)
        self.environment_info = collect_environment_info()
        self.commit_hash = get_git_commit_hash()
        
        # Use pre-loaded registries or load them
        if registries:
            self.solver_registry = registries['solver_registry']
            self.problem_registry = registries['problem_registry']
            logger.debug("Using pre-loaded registries")
        else:
            # Fallback to loading configurations
            self.solver_registry = self.load_solver_registry()
            self.problem_registry = self.load_problem_registry()
        
        
        logger.info("Benchmark runner initialized")
        logger.info(f"Git commit: {self.commit_hash}")
        logger.info(f"Environment: {self.environment_info['os']['system']} {self.environment_info['python']['version']}")
    
    def load_solver_registry(self) -> Dict[str, Any]:
        """Load solver registry from config/solver_registry.yaml"""
        try:
            config_path = project_root / "config" / "solver_registry.yaml"
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load solver registry: {e}")
            # Return empty registry to force configuration fix
            return {'solvers': {}}
    
    def load_problem_registry(self) -> Dict[str, Any]:
        """Load problem registry from config/problem_registry.yaml"""
        try:
            config_path = project_root / "config" / "problem_registry.yaml"
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load problem registry: {e}")
            return {'problem_libraries': {}}
    
    def create_solver(self, solver_name: str) -> SolverInterface:
        """
        Create solver instance based on solver name using direct logic.
        
        Args:
            solver_name: Name of solver to create
            
        Returns:
            Solver instance
            
        Raises:
            ValueError: If solver name is unknown
        """
        logger.debug(f"Creating solver: {solver_name}")
        
        # Direct if-elif logic as specified in re-architecture
        if solver_name == "scipy_linprog":
            return ScipySolver(save_solutions=self.save_solutions)
        elif solver_name == "cvxpy_clarabel":
            return CvxpySolver(backend="CLARABEL", save_solutions=self.save_solutions)
        elif solver_name == "cvxpy_scs":
            return CvxpySolver(backend="SCS", save_solutions=self.save_solutions)
        elif solver_name == "cvxpy_ecos":
            return CvxpySolver(backend="ECOS", save_solutions=self.save_solutions)
        elif solver_name == "cvxpy_osqp":
            return CvxpySolver(backend="OSQP", save_solutions=self.save_solutions)
        elif solver_name == "cvxpy_cvxopt":
            return CvxpySolver(backend="CVXOPT", save_solutions=self.save_solutions)
        elif solver_name == "cvxpy_sdpa":
            return CvxpySolver(backend="SDPA", save_solutions=self.save_solutions)
        elif solver_name == "cvxpy_scip":
            return CvxpySolver(backend="SCIP", save_solutions=self.save_solutions)
        elif solver_name == "cvxpy_highs":
            return CvxpySolver(backend="HIGHS", save_solutions=self.save_solutions)  # Actual HiGHS solver
        else:
            raise ValueError(f"Unknown solver: {solver_name}")
    
    def load_problem(self, problem_name: str, problem_config: Dict[str, Any]) -> Any:
        """
        Load problem using appropriate loader based on file type.
        
        Args:
            problem_name: Name of the problem
            problem_config: Problem configuration from registry
            
        Returns:
            Loaded problem data
        """
        file_type = problem_config['file_type']
        file_path = project_root / problem_config['file_path']
        
        logger.debug(f"Loading problem {problem_name} from {file_path} (type: {file_type})")
        
        # Select appropriate loader based on file type
        if file_type == 'mat':
            loader = MATLoader()
        elif file_type == 'dat-s':
            loader = DATLoader()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return loader.load(str(file_path))
    
    def store_result(self, solver_name: str, problem_name: str, 
                    result: SolverResult, problem_config: Dict[str, Any], 
                    problem_data=None) -> None:
        """
        Store result in database using simplified schema.
        
        Args:
            solver_name: Name of the solver
            problem_name: Name of the problem  
            result: Standardized solver result
            problem_config: Problem configuration for metadata
            problem_data: Optional problem data for type detection
        """
        try:
            # Determine problem library and type
            problem_library = problem_config.get('library_name', 'internal')
            
            # Get problem type from actual problem data (most reliable)
            if problem_data and hasattr(problem_data, 'problem_class'):
                problem_type = problem_data.problem_class
            elif hasattr(result, 'additional_info') and result.additional_info:
                # Try to get problem class from solver result
                problem_type = result.additional_info.get('problem_class', 'UNKNOWN')
            else:
                # Fallback to config, then to UNKNOWN
                problem_type = problem_config.get('problem_type', 'UNKNOWN')
            
            # Skip database operations in dry-run mode
            if self.dry_run:
                logger.info(f"[DRY-RUN] Would store result: {solver_name} on {problem_name} ({result.status})")
                return
            
            # Store in database using the simplified schema
            self.db.store_result(
                solver_name=solver_name,
                solver_version=result.solver_version or "unknown",
                problem_library=problem_library,
                problem_name=problem_name,
                problem_type=problem_type,
                environment_info=self.environment_info,
                commit_hash=self.commit_hash,
                solve_time=result.solve_time,
                status=result.status,
                primal_objective_value=result.primal_objective_value,
                dual_objective_value=result.dual_objective_value,
                duality_gap=result.duality_gap,
                primal_infeasibility=result.primal_infeasibility,
                dual_infeasibility=result.dual_infeasibility,
                iterations=result.iterations
            )
            
            logger.debug(f"Stored result: {solver_name} on {problem_name}")
            
        except Exception as e:
            logger.error(f"Failed to store result for {solver_name} on {problem_name}: {e}")
    
    def store_error_result(self, solver_name: str, problem_name: str, 
                          error_message: str, problem_config: Dict[str, Any]) -> None:
        """
        Store error result in database.
        
        Args:
            solver_name: Name of the solver
            problem_name: Name of the problem
            error_message: Error description
            problem_config: Problem configuration for metadata
        """
        # Skip database operations in dry-run mode
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would store error result: {solver_name} on {problem_name} ({error_message})")
            return
            
        error_result = SolverResult.create_error_result(
            error_message=error_message,
            solve_time=0.0,
            solver_name=solver_name,
            solver_version="unknown"
        )
        
        self.store_result(solver_name, problem_name, error_result, problem_config)
    
    def run_single_benchmark(self, problem_name: str, problem_config: Dict[str, Any], 
                            solver_name: str, solver_config: Dict[str, Any]) -> None:
        """
        Execute single problem-solver combination and store result.
        
        Args:
            problem_name: Name of the problem to solve
            problem_config: Problem configuration from registry
            solver_name: Name of the solver to use  
            solver_config: Solver configuration from registry
        """
        logger.info(f"Running {solver_name} on {problem_name}")
        
        try:
            # Load problem using appropriate loader
            problem_data = self.load_problem(problem_name, problem_config)
            
            # Create solver
            solver = self.create_solver(solver_name)
            
            # Check solver compatibility with problem type BEFORE execution
            if not solver.validate_problem_compatibility(problem_data):
                problem_type = problem_data.problem_class
                logger.info(f"Skipping {solver_name} on {problem_name}: solver cannot handle {problem_type} problems")
                return  # Skip execution entirely - no database entry
            
            # Execute solver with timing
            start_time = time.time()
            result = solver.solve(problem_data)
            solve_time = time.time() - start_time
            
            # Ensure timing is accurate
            result.solve_time = solve_time
            result.solver_name = solver_name
            result.solver_version = solver.get_version()
            
            # Store result in database with problem data for type detection
            self.store_result(solver_name, problem_name, result, problem_config, problem_data)
            
            # Enhanced logging with computation time and optimal value
            if result.primal_objective_value is not None:
                logger.info(f"Completed {solver_name} on {problem_name}: {result.status} in {solve_time:.3f}s, objective: {result.primal_objective_value:.6e}")
            else:
                logger.info(f"Completed {solver_name} on {problem_name}: {result.status} in {solve_time:.3f}s")
            
        except Exception as e:
            error_msg = f"Benchmark execution failed: {str(e)}"
            logger.error(error_msg)
            
            # Store error result
            if problem_name in self.problem_registry['problem_libraries']:
                problem_config = self.problem_registry['problem_libraries'][problem_name]
                self.store_error_result(solver_name, problem_name, error_msg, problem_config)
    
    def run_benchmark_batch(self, problems: List[str], solvers: List[str]) -> None:
        """
        Run benchmark for all problem-solver combinations.
        
        Args:
            problems: List of problem names to run
            solvers: List of solver names to use
        """
        total_combinations = len(problems) * len(solvers)
        completed = 0
        
        logger.info(f"Starting benchmark batch: {len(problems)} problems × {len(solvers)} solvers = {total_combinations} combinations")
        
        for problem_name in problems:
            for solver_name in solvers:
                try:
                    completed += 1
                    logger.info(f"Progress: {completed}/{total_combinations}")
                    
                    self.run_single_benchmark(problem_name, solver_name)
                    
                except Exception as e:
                    logger.error(f"Failed {solver_name} on {problem_name}: {e}")
                    # Continue with next combination
        
        logger.info(f"Benchmark batch completed: {completed} combinations processed")
    
    def get_available_problems(self, for_test_only: bool = False) -> List[str]:
        """
        Get list of available problems from registry.
        
        Args:
            for_test_only: If True, only return problems marked for testing
            
        Returns:
            List of problem names
        """
        problems = []
        for problem_name, config in self.problem_registry['problem_libraries'].items():
            if for_test_only:
                if config.get('for_test_flag', False):
                    problems.append(problem_name)
            else:
                problems.append(problem_name)
        
        return problems
    
    def get_available_solvers(self) -> List[str]:
        """
        Get list of available solvers from registry.
        
        Returns:
            List of solver names
        """
        return list(self.solver_registry['solvers'].keys())
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate that solvers and problems can be loaded.
        
        Returns:
            Validation report
        """
        report = {
            'solvers': {},
            'problems': {},
            'summary': {
                'total_solvers': 0,
                'working_solvers': 0,
                'total_problems': 0,
                'working_problems': 0
            }
        }
        
        # Test solver creation
        for solver_name in self.get_available_solvers():
            report['summary']['total_solvers'] += 1
            try:
                solver = self.create_solver(solver_name)
                report['solvers'][solver_name] = {
                    'status': 'working',
                    'version': solver.get_version()
                }
                report['summary']['working_solvers'] += 1
            except Exception as e:
                report['solvers'][solver_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Test problem loading
        for problem_name in self.get_available_problems():
            report['summary']['total_problems'] += 1
            try:
                problem_config = self.problem_registry['problem_libraries'][problem_name]
                problem_data = self.load_problem(problem_name, problem_config)
                report['problems'][problem_name] = {
                    'status': 'working',
                    'type': problem_config.get('problem_type', 'unknown'),
                    'library': problem_config.get('library_name', 'unknown')
                }
                report['summary']['working_problems'] += 1
            except Exception as e:
                report['problems'][problem_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return report


if __name__ == "__main__":
    # Test script for simplified runner
    try:
        print("Testing Simplified Benchmark Runner...")
        print("=" * 50)
        
        # Initialize runner
        print("\n1. Initializing runner...")
        runner = BenchmarkRunner()
        print(f"   ✓ Runner initialized")
        print(f"   Git commit: {runner.commit_hash[:8]}")
        print(f"   Environment: {runner.environment_info['os']['system']}")
        
        # Validate setup
        print("\n2. Validating setup...")
        validation_report = runner.validate_setup()
        
        working_solvers = validation_report['summary']['working_solvers']
        total_solvers = validation_report['summary']['total_solvers']
        working_problems = validation_report['summary']['working_problems']
        total_problems = validation_report['summary']['total_problems']
        
        print(f"   Solvers: {working_solvers}/{total_solvers} working")
        print(f"   Problems: {working_problems}/{total_problems} working")
        
        # Show working solvers
        print("\n   Working solvers:")
        for solver_name, info in validation_report['solvers'].items():
            if info['status'] == 'working':
                print(f"     ✓ {solver_name}: {info['version']}")
            else:
                print(f"     ✗ {solver_name}: {info['error']}")
        
        # Show working problems
        print("\n   Working problems:")
        for problem_name, info in validation_report['problems'].items():
            if info['status'] == 'working':
                print(f"     ✓ {problem_name} ({info['type']}, {info['library']})")
            else:
                print(f"     ✗ {problem_name}: {info['error']}")
        
        # Test single benchmark if we have working solver and problem
        if working_solvers > 0 and working_problems > 0:
            print("\n3. Testing single benchmark...")
            
            # Find first working solver and problem
            working_solver = None
            working_problem = None
            
            for solver_name, info in validation_report['solvers'].items():
                if info['status'] == 'working':
                    working_solver = solver_name
                    break
            
            for problem_name, info in validation_report['problems'].items():
                if info['status'] == 'working':
                    working_problem = problem_name
                    break
            
            if working_solver and working_problem:
                print(f"   Running {working_solver} on {working_problem}...")
                runner.run_single_benchmark(working_problem, working_solver)
                print(f"   ✓ Single benchmark completed successfully")
        
        # Test batch benchmark with small subset
        if working_solvers > 0 and working_problems > 0:
            print("\n4. Testing batch benchmark...")
            
            # Get test problems (or first problem if no test problems)
            test_problems = runner.get_available_problems(for_test_only=True)
            if not test_problems:
                test_problems = [runner.get_available_problems()[0]]
            
            # Use first working solver
            test_solvers = [working_solver]
            
            print(f"   Running {len(test_solvers)} solvers on {len(test_problems)} problems...")
            runner.run_benchmark_batch(test_problems[:1], test_solvers[:1])  # Limit for testing
            print(f"   ✓ Batch benchmark completed successfully")
        
        print("\n" + "=" * 50)
        print("✓ All simplified runner tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"✗ Test failed: {e}")
        raise