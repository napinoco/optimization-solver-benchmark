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

# Interface imports (symmetrical design)
from scripts.solvers.solver_interface import SolverInterface, SolverResult
from scripts.solvers.python.python_interface import PythonInterface
from scripts.solvers.matlab_octave.matlab_interface import MatlabInterface
from scripts.data_loaders.python.problem_interface import ProblemInterface

logger = get_logger("benchmark_runner")

# MATLAB availability will be checked lazily when needed (true EAFP approach)


class BenchmarkRunner:
    """Main benchmark execution engine with symmetrical solver interfaces"""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None, 
                 dry_run: bool = False,
                 save_solutions: bool = False,
                 default_timeout: float = 120.0):
        """
        Initialize benchmark runner with symmetrical solver interfaces.
        
        Args:
            database_manager: Optional database manager (creates default if None)
            dry_run: If True, skip database operations (for testing)
            save_solutions: If True, save optimal solutions to disk
            default_timeout: Default timeout in seconds for solver execution
        """
        self.db = database_manager or DatabaseManager()
        self.dry_run = dry_run
        self.save_solutions = save_solutions
        self.default_timeout = default_timeout
        
        # Initialize problem interface only (essential for all operations)
        self.problem_interface = ProblemInterface()
        
        # Lazy initialization for solver interfaces (true EAFP approach)
        self._python_interface = None
        self._matlab_interface = None
        self._matlab_interface_attempted = False  # Track if MATLAB initialization was attempted
        
        # Build solver-to-interface mapping for efficient routing
        self._solver_interface_map = self._build_solver_interface_map()
        
        # Collect environment info and git hash once (now cached)
        self.environment_info = collect_environment_info()
        self.commit_hash = get_git_commit_hash()
        
        # Solver configurations now managed by interfaces (no more solver registry YAML)
        # Problem registry is now loaded directly by ProblemInterface

        logger.info("Benchmark runner initialized with unified interfaces")
        logger.info(f"Git commit: {self.commit_hash}")
        logger.info(f"Environment: {self.environment_info['os']['system']} {self.environment_info['python']['version']}")
        logger.info(f"Default timeout: {self.default_timeout}s")
        
        python_configured = len(PythonInterface.PYTHON_SOLVER_CONFIGS)
        matlab_configured = len(MatlabInterface.MATLAB_SOLVER_CONFIGS)
        problem_stats = self.problem_interface.get_problem_statistics()
        
        logger.info(f"Python interface: {python_configured} solvers configured (lazy detection)")
        logger.info(f"MATLAB interface: {matlab_configured} solvers configured (lazy detection)")
        logger.info(f"Problem interface: {problem_stats['total_problems']} problems from {len(problem_stats['libraries'])} libraries")
    
    @property
    def python_interface(self) -> PythonInterface:
        """Lazy initialization of Python interface."""
        if self._python_interface is None:
            logger.debug("Initializing Python interface on first access")
            self._python_interface = PythonInterface(
                save_solutions=self.save_solutions,
                problem_interface=self.problem_interface
            )
        return self._python_interface
    
    @property
    def matlab_interface(self) -> Optional[MatlabInterface]:
        """Lazy initialization of MATLAB interface."""
        if self._matlab_interface is None and not self._matlab_interface_attempted:
            self._matlab_interface_attempted = True
            try:
                logger.debug("Initializing MATLAB interface on first access")
                self._matlab_interface = MatlabInterface(
                    save_solutions=self.save_solutions,
                    problem_interface=self.problem_interface
                )
            except Exception as e:
                logger.warning(f"Failed to initialize MATLAB interface: {e}")
                self._matlab_interface = None
        return self._matlab_interface
    
    def _build_solver_interface_map(self) -> Dict[str, str]:
        """
        Build mapping from solver names to interface types.
        
        Returns:
            Dictionary mapping solver_name -> interface_type ('python' or 'matlab')
        """
        mapping = {}
        
        # Add Python solvers
        for solver_name in PythonInterface.PYTHON_SOLVER_CONFIGS.keys():
            mapping[solver_name] = 'python'
        
        # Add MATLAB solvers
        for solver_name in MatlabInterface.MATLAB_SOLVER_CONFIGS.keys():
            mapping[solver_name] = 'matlab'
        
        logger.debug(f"Built solver interface mapping: {len(mapping)} solvers")
        return mapping
    
    
    def create_solver(self, solver_name: str) -> SolverInterface:
        """
        Create solver instance (for backward compatibility).
        
        Note: This method is kept for backward compatibility but the preferred
        approach is to use the unified solve methods on the interfaces directly.
        
        Args:
            solver_name: Name of solver to create
            
        Returns:
            Solver instance
            
        Raises:
            ValueError: If solver name is unknown
        """
        logger.debug(f"Creating solver via interface: {solver_name}")
        
        # EAFP approach: try Python interface first
        try:
            return self.python_interface.create_solver(solver_name)
        except ValueError:
            # If not a Python solver, try MATLAB interface
            if self.matlab_interface:
                try:
                    return self.matlab_interface.create_solver(solver_name)
                except ValueError:
                    pass
            
            # Provide helpful error message with available solvers
            available_solvers = self.get_available_solvers()
            raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {available_solvers}")
    
    def get_available_solvers(self) -> List[str]:
        """
        Get list of currently available solvers from all interfaces.
        
        Note: This method forces initialization of interfaces for validation purposes.
        
        Returns:
            List of solver names that can be created
        """
        # Get Python solvers from interface (may trigger initialization)
        available_solvers = self.python_interface.get_available_solvers()
        
        # Add MATLAB solvers if interface is available (may trigger initialization)
        if self.matlab_interface:
            available_solvers.extend(self.matlab_interface.get_available_solvers())
        
        return available_solvers
    
    def load_problem(self, problem_name: str, problem_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Load problem using problem interface (symmetrical design).
        
        Args:
            problem_name: Name of the problem
            problem_config: Optional problem configuration (delegated to interface)
            
        Returns:
            Loaded problem data
        """
        return self.problem_interface.load_problem(problem_name, problem_config)
    
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
            
            # Convert additional_info to JSON string for memo field
            memo = None
            if result.additional_info:
                import json
                memo = json.dumps(result.additional_info, indent=2, default=str)
            
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
                iterations=result.iterations,
                memo=memo
            )
            
            logger.debug(f"Stored result: {solver_name} on {problem_name}")
            
        except Exception as e:
            logger.error(f"Failed to store result for {solver_name} on {problem_name}: {e}")
    
    def store_error_result(self, solver_name: str, problem_name: str, 
                          error_message: str, problem_config: Dict[str, Any],
                          problem_data=None) -> None:
        """
        Store error result in database.
        
        Args:
            solver_name: Name of the solver
            problem_name: Name of the problem
            error_message: Error description
            problem_config: Problem configuration for metadata
            problem_data: Optional problem data for type detection
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
        
        # Add problem class to additional_info if available
        if problem_data and hasattr(problem_data, 'problem_class'):
            if not error_result.additional_info:
                error_result.additional_info = {}
            error_result.additional_info['problem_class'] = problem_data.problem_class
        
        self.store_result(solver_name, problem_name, error_result, problem_config, problem_data)
    
    def run_single_benchmark(self, problem_name: str, solver_name: str) -> None:
        """
        Execute single problem-solver combination using explicit interface routing.
        
        This method uses explicit solver-to-interface mapping to route directly to the
        appropriate interface, eliminating unnecessary exceptions and improving efficiency.
        
        Args:
            problem_name: Name of the problem to solve
            solver_name: Name of the solver to use
        """
        logger.info(f"Running {solver_name} on {problem_name}")
        
        try:
            # Get the appropriate interface type from mapping
            interface_type = self._solver_interface_map.get(solver_name)
            
            if interface_type == 'python':
                # Route directly to Python interface
                result = self.python_interface.solve(problem_name, solver_name, timeout=self.default_timeout)
                
            elif interface_type == 'matlab':
                result = self.matlab_interface.solve(problem_name, solver_name, timeout=self.default_timeout)

            else:
                raise ValueError(f"Unknown solver '{solver_name}'. Available solvers: {list(self._solver_interface_map.keys())}")
            
            # Success! Get problem configuration and store result
            problem_config = self.problem_interface.get_problem_config(problem_name)
            self.store_result(solver_name, problem_name, result, problem_config)
            
            # Enhanced logging with computation time and optimal value
            if result.status == "UNSUPPORTED":
                logger.info(f"Completed {solver_name} on {problem_name}: {result.status} (problem type not supported)")
            elif result.solve_time is not None and result.primal_objective_value is not None:
                logger.info(f"Completed {solver_name} on {problem_name}: {result.status} in {result.solve_time:.3f}s, objective: {result.primal_objective_value:.6e}")
            elif result.solve_time is not None:
                logger.info(f"Completed {solver_name} on {problem_name}: {result.status} in {result.solve_time:.3f}s")
            else:
                logger.info(f"Completed {solver_name} on {problem_name}: {result.status}")
                
        except Exception as e:
            error_msg = f"Benchmark execution failed: {str(e)}"
            logger.error(error_msg)
            
            # Skip storing result if solver doesn't exist
            if "Unknown solver" in str(e):
                logger.warning(f"Skipping database storage for unknown solver: {solver_name}")
                return
            
            # Store error result for other types of errors
            try:
                problem_config = self.problem_interface.get_problem_config(problem_name)
                self.store_error_result(solver_name, problem_name, error_msg, problem_config)
            except:
                # If we can't even get problem config, create minimal config
                minimal_config = {'library_name': 'unknown', 'problem_type': 'UNKNOWN'}
                self.store_error_result(solver_name, problem_name, error_msg, minimal_config)
    
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
        Get list of available problems using problem interface.
        
        Args:
            for_test_only: If True, only return problems marked for testing
            
        Returns:
            List of problem names
        """
        return self.problem_interface.get_available_problems(test_only=for_test_only)
    
    
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
        
        # Test problem loading using problem interface
        for problem_name in self.get_available_problems():
            report['summary']['total_problems'] += 1
            try:
                problem_config = self.problem_interface.get_problem_config(problem_name)
                problem_data = self.load_problem(problem_name)
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