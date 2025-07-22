#!/usr/bin/env python3
import sys
import os
import argparse
import time
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from scripts.benchmark.runner import BenchmarkRunner
    from scripts.database.database_manager import DatabaseManager
    from scripts.reporting.html_generator import HTMLGenerator
    from scripts.reporting.data_exporter import DataExporter
    from scripts.utils.logger import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the repository root directory.")
    sys.exit(1)

# Global cache for solver availability to avoid expensive testing
_solver_availability_cache = {}


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Setup logging configuration."""
    import logging
    
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_problem_registry() -> Optional[Dict[str, Any]]:
    """Load problem registry configuration."""
    logger = get_logger("problem_registry_loader")
    
    try:
        # Load problem registry
        problem_registry_path = Path("config/problem_registry.yaml")
        if not problem_registry_path.exists():
            logger.error("Problem registry not found")
            return None
        
        with open(problem_registry_path, 'r') as f:
            problem_registry = yaml.safe_load(f)
        
        logger.info("Problem registry loaded successfully")
        return problem_registry
        
    except Exception as e:
        logger.error(f"Failed to load problem registry: {e}")
        return None


def validate_environment() -> bool:
    """Validate that the environment is properly set up."""
    
    logger = get_logger("validation")
    issues = []
    
    # Check for required directories
    required_dirs = ['config', 'scripts', 'problems']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            issues.append(f"Missing directory: {dir_name}")
    
    # Check for configuration files
    config_files = [
        'config/site_config.yaml',
        'config/problem_registry.yaml'
    ]
    for config_file in config_files:
        if not Path(config_file).exists():
            issues.append(f"Configuration file not found: {config_file}")
    
    # Create database directory if needed
    db_dir = Path("database")
    if not db_dir.exists():
        logger.info("Creating database directory...")
        db_dir.mkdir(parents=True, exist_ok=True)
    
    # Create docs/pages directory if needed
    docs_dir = Path("docs/pages")
    if not docs_dir.exists():
        logger.info("Creating docs/pages directory...")
        docs_dir.mkdir(parents=True, exist_ok=True)
    
    if issues:
        for issue in issues:
            logger.error(issue)
        return False
    
    logger.info("Environment validation passed")
    return True


def validate_solver_setup(verbose: bool = False) -> bool:
    """Validate solver setup and availability."""
    
    logger = get_logger("solver_validation")
    
    try:
        # Initialize benchmark runner to test solver availability
        runner = BenchmarkRunner()
        
        # Run comprehensive validation
        print("Running comprehensive solver validation...")
        validation_report = runner.validate_setup()
        
        working_solvers = validation_report['summary']['working_solvers']
        total_solvers = validation_report['summary']['total_solvers']
        working_problems = validation_report['summary']['working_problems']
        total_problems = validation_report['summary']['total_problems']
        
        print(f"\nSolver Validation Results:")
        print(f"  Working Solvers: {working_solvers}/{total_solvers}")
        print(f"  Working Problems: {working_problems}/{total_problems}")
        
        # Show solver details
        if verbose:
            print(f"\nDetailed Solver Status:")
            for solver_name, info in validation_report['solvers'].items():
                status = "✓" if info['status'] == 'working' else "✗"
                if info['status'] == 'working':
                    print(f"  {status} {solver_name}: {info['version']}")
                else:
                    print(f"  {status} {solver_name}: {info['error']}")
        
        # Check MATLAB solver availability
        matlab_solvers = {name: info for name, info in validation_report['solvers'].items() 
                         if name.startswith('matlab_')}
        
        if matlab_solvers:
            matlab_working = sum(1 for info in matlab_solvers.values() if info['status'] == 'working')
            print(f"\nMATLAB Solver Status: {matlab_working}/{len(matlab_solvers)} working")
            
            if verbose:
                for solver_name, info in matlab_solvers.items():
                    status = "✓" if info['status'] == 'working' else "✗"
                    if info['status'] == 'working':
                        print(f"  {status} {solver_name}: {info['version']}")
                    else:
                        print(f"  {status} {solver_name}: {info['error']}")
        
        # Validation passes if we have at least some working solvers
        if working_solvers == 0:
            logger.error("No working solvers found")
            return False
        
        # Show warnings for missing solvers
        missing_solvers = total_solvers - working_solvers
        if missing_solvers > 0:
            logger.warning(f"{missing_solvers} solvers are not available")
        
        logger.info(f"Solver validation completed: {working_solvers}/{total_solvers} solvers working")
        return True
        
    except Exception as e:
        logger.error(f"Solver validation failed: {e}")
        return False


def run_benchmark(library_names: Optional[List[str]] = None,
                 problems: Optional[List[str]] = None,
                 solvers: Optional[List[str]] = None,
                 dry_run: bool = False,
                 save_solutions: bool = False) -> bool:
    """Run the benchmark suite with simplified registry-based approach.
    
    Args:
        library_names: List of library names to filter by
        problems: List of specific problem names to run
        solvers: List of solver names to run
        dry_run: If True, skip database operations (for testing)
        save_solutions: If True, save optimal solutions to disk
    """
    
    logger = get_logger("benchmark")
    
    try:
        logger.info("Starting benchmark execution...")
        
        # Create benchmark runner (db_manager created internally by default)
        runner = BenchmarkRunner(dry_run=dry_run, save_solutions=save_solutions)

        # Load problem registry only (solver configs now in interfaces)
        problem_registry = load_problem_registry()
        if not problem_registry:
            logger.error("Failed to load problem registry")
            return False

        # Filter problems based on library_names and problems arguments
        selected_problems = {}
        for problem_name, problem_config in problem_registry["problem_libraries"].items():
            # Filter by library_names if specified
            if library_names and problem_config.get("library_name") not in library_names:
                continue
            
            # Filter by specific problem names if specified
            if problems and problem_name not in problems:
                continue
            
            selected_problems[problem_name] = problem_config
        
        if not selected_problems:
            logger.error("No problems selected. Check your --library_names and --problems filters.")
            return False
        
        logger.info(f"Selected {len(selected_problems)} problems")
        
        # EAFP approach: No pre-filtering of solvers
        if solvers:
            # Use user-specified solvers directly
            selected_solver_names = solvers
            logger.info(f"Will attempt to run specified solvers: {selected_solver_names}")
        else:
            # Get all available solvers for default case
            selected_solver_names = runner.get_available_solvers()
            logger.info(f"Will attempt to run all available solvers: {len(selected_solver_names)} found")
        
        if not selected_solver_names:
            logger.error("No solvers specified. Use --solvers to specify solvers or remove filter to run all.")
            return False
        
        # Run benchmarks using simple nested loop
        start_time = time.time()
        total_combinations = len(selected_problems) * len(selected_solver_names)
        logger.info(f"Running {total_combinations} problem-solver combinations...")
        
        success_count = 0
        combination_num = 0
        
        for problem_name in selected_problems.keys():
            for solver_name in selected_solver_names:
                combination_num += 1
                logger.info(f"[{combination_num}/{total_combinations}] Running {solver_name} on {problem_name}")
                
                try:
                    # EAFP: Just try it! Let runner handle unknown solvers
                    runner.run_single_benchmark(problem_name, solver_name)
                    success_count += 1
                except Exception as benchmark_error:
                    logger.warning(f"Failed to run {solver_name} on {problem_name}: {benchmark_error}")
        
        duration = time.time() - start_time
        success_rate = (success_count / total_combinations) * 100 if total_combinations > 0 else 0
        
        logger.info(f"Benchmark completed in {duration:.2f}s")
        logger.info(f"Success rate: {success_rate:.1f}% ({success_count}/{total_combinations})")
        
        return success_count > 0
        
    except Exception as main_error:
        logger.error(f"Benchmark execution failed: {main_error}")
        return False


def generate_reports() -> bool:
    """Generate simplified HTML reports and data exports from benchmark results."""
    
    logger = get_logger("reporting")
    
    try:
        logger.info("Starting simplified report generation...")
        
        # Check if database exists
        database_path = "database/results.db"
        if not Path(database_path).exists():
            logger.error(f"Database not found: {database_path}")
            logger.error("Please run benchmarks first to generate data.")
            return False
        
        # Generate simplified HTML reports (3 reports only)
        logger.info("Generating simplified HTML reports...")
        
        html_generator = HTMLGenerator(output_dir="docs/pages")
        html_success = html_generator.generate_all_reports()
        
        if not html_success:
            logger.error("Failed to generate HTML reports")
            return False
        
        logger.info("HTML reports generated successfully")
        
        # Export data in JSON/CSV formats
        logger.info("Exporting data files...")
        
        data_exporter = DataExporter(output_dir="docs/pages/data")
        data_success = data_exporter.export_both_versions()
        
        if not data_success:
            logger.error("Failed to export data files")
            return False
        
        logger.info("Data export completed successfully")
        
        # Summary
        logger.info("Simplified reporting completed successfully")
        logger.info("Generated files:")
        logger.info("  • docs/pages/index.html (Overview Dashboard)")
        logger.info("  • docs/pages/results_matrix.html (Results Matrix)")
        logger.info("  • docs/pages/raw_data.html (Raw Data Table)")
        logger.info("  • docs/pages/data/benchmark_results_latest.json (Latest Results JSON)")
        logger.info("  • docs/pages/data/benchmark_results_latest.csv (Latest Results CSV)")
        logger.info("  • docs/pages/data/benchmark_results_all.json (All Results JSON)")
        logger.info("  • docs/pages/data/benchmark_results_all.csv (All Results CSV)")
        
        return True
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return False


def main():
    """Main entry point."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Optimization Solver Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                                           # Run benchmarks and generate reports
  python main.py --benchmark                                     # Run benchmarks only
  python main.py --report                                        # Generate reports only
  python main.py --validate                                      # Validate environment setup
  python main.py --benchmark --library_names DIMACS             # Run all DIMACS problems
  python main.py --benchmark --library_names DIMACS,SDPLIB      # Run DIMACS and SDPLIB problems
  python main.py --benchmark --problems nb,arch0,simple_lp_test  # Run specific problems by name
  python main.py --benchmark --library_names internal           # Run internal library problems
  python main.py --benchmark --solvers cvxpy_clarabel,scipy_linprog  # Run specific solvers
  python main.py --all --problems nb --solvers cvxpy_clarabel    # Run one problem with one solver
  python main.py --benchmark --library_names DIMACS --solvers cvxpy_scip  # Run DIMACS with SCIP
  python main.py --benchmark --problems nb --dry-run  # Test nb problem without DB storage
        """
    )
    
    # Main operation modes
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run benchmark suite only'
    )
    operation_group.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate HTML reports only'
    )
    operation_group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run benchmarks and generate reports'
    )
    operation_group.add_argument(
        '--validate',
        action='store_true',
        help='Validate environment and solver setup'
    )
    operation_group.add_argument(
        '--validate-verbose',
        action='store_true',
        help='Validate environment and solver setup with detailed output'
    )
    
    # Options
    parser.add_argument(
        '--library_names', '-l',
        type=str,
        help='Comma-separated list of library names to run (DIMACS, SDPLIB, internal)'
    )
    
    parser.add_argument(
        '--problems', '-p',
        type=str,
        help='Comma-separated list of specific problem names to run'
    )
    
    parser.add_argument(
        '--solvers', '-s',
        type=str,
        help='Comma-separated list of solvers to run'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run benchmarks without storing results in database (for testing)'
    )
    
    parser.add_argument(
        '--save-solutions',
        action='store_true',
        help='Save optimal solutions to disk for verification and analysis'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-error output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    # Parse solver list
    solvers = None
    if args.solvers:
        solvers = [s.strip() for s in args.solvers.split(',')]
    
    # Parse library names list
    library_names = None
    if args.library_names:
        library_names = [l.strip() for l in args.library_names.split(',')]
    
    # Parse problem list
    problems = None
    if args.problems:
        problems = [p.strip() for p in args.problems.split(',')]
    
    
    try:
        # Validate environment first
        if not validate_environment():
            print("Environment validation failed")
            sys.exit(1)
        
        # Execute requested operation
        success = False
        
        if args.validate or args.validate_verbose:
            # Run enhanced validation including solver testing
            solver_validation_success = validate_solver_setup(verbose=args.validate_verbose)
            if solver_validation_success:
                print("Validation completed successfully")
            else:
                print("Validation completed with issues")
            success = solver_validation_success
            
        elif args.benchmark:
            success = run_benchmark(
                library_names=library_names,
                problems=problems,
                solvers=solvers,
                dry_run=args.dry_run,
                save_solutions=args.save_solutions
            )
            
        elif args.report:
            success = generate_reports()
            
        elif args.all:
            # Run benchmarks first
            benchmark_success = run_benchmark(
                library_names=library_names,
                problems=problems,
                solvers=solvers,
                dry_run=args.dry_run,
                save_solutions=args.save_solutions
            )
            
            if benchmark_success:
                # Then generate reports
                report_success = generate_reports()
                success = report_success
            else:
                print("Benchmark failed, skipping report generation")
                success = False
        
        # Exit with appropriate code
        if success:
            print("Operation completed successfully")
            sys.exit(0)
        else:
            print("Operation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()