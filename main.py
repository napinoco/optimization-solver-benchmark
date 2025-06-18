#!/usr/bin/env python3
"""
Optimization Solver Benchmark - Main Entry Point

This script provides a unified interface to run benchmarks and generate reports
using the re-architected simplified system.

Usage Examples:
    python main.py --help                    # Show help
    python main.py --benchmark               # Run benchmarks on all problems with all solvers
    python main.py --report                  # Generate reports only
    python main.py --all                     # Run benchmarks and generate reports
    python main.py --validate               # Validate environment and solver setup
    python main.py --benchmark --problems DIMACS --solvers cvxpy_clarabel
    python main.py --benchmark --problems nb,arch0,simple_lp_test
    python main.py --benchmark --problems DIMACS,simple_lp_test
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from scripts.benchmark.runner import BenchmarkRunner
    from scripts.database.database_manager import DatabaseManager
    from scripts.reporting.simple_html_generator import SimpleHTMLGenerator
    from scripts.benchmark.problem_loader import list_available_problems
    from scripts.utils.logger import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the repository root directory.")
    sys.exit(1)


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
        'config/solver_registry.yaml', 
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


def run_benchmark(problems: Optional[List[str]] = None,
                 solvers: Optional[List[str]] = None) -> bool:
    """Run the benchmark suite."""
    
    logger = get_logger("benchmark")
    
    try:
        logger.info("Starting benchmark execution...")
        
        # Create database manager
        db_manager = DatabaseManager()
        
        # Create benchmark runner
        runner = BenchmarkRunner(db_manager)
        
        # Determine which problems to run
        problems_to_run = []
        
        if problems:
            available_problems = list_available_problems()
            all_available = []
            for lib_problems in available_problems.values():
                all_available.extend(lib_problems)
            
            # Check if any of the specified problems are library names
            library_names = list(available_problems.keys())
            
            for problem in problems:
                if problem in library_names:
                    # This is a library name, add all problems from that library
                    problems_to_run.extend(available_problems[problem])
                    logger.info(f"Added all problems from library '{problem}': {len(available_problems[problem])} problems")
                elif problem in all_available:
                    # This is an individual problem name
                    problems_to_run.append(problem)
                else:
                    # Invalid problem name
                    logger.error(f"Invalid problem or library name: {problem}")
                    logger.error(f"Available libraries: {library_names}")
                    logger.error(f"Available individual problems: {all_available}")
                    return False
            
            # Remove duplicates while preserving order
            problems_to_run = list(dict.fromkeys(problems_to_run))
            logger.info(f"Using specified problems: {problems_to_run} ({len(problems_to_run)} total)")
            
        else:
            # Run all available problems
            available_problems = list_available_problems()
            for lib_problems in available_problems.values():
                problems_to_run.extend(lib_problems)
            logger.info(f"Using all available problems: {len(problems_to_run)} problems")
        
        # Determine which solvers to run
        available_solvers = ["scipy_linprog", "cvxpy_clarabel", "cvxpy_scs", "cvxpy_ecos", "cvxpy_osqp"]
        if solvers:
            # Validate requested solvers
            invalid_solvers = [s for s in solvers if s not in available_solvers]
            if invalid_solvers:
                logger.error(f"Invalid solvers: {invalid_solvers}. Available: {available_solvers}")
                return False
            solvers_to_run = solvers
        else:
            solvers_to_run = available_solvers
        
        logger.info(f"Using solvers: {solvers_to_run}")
        
        # Run the benchmark
        start_time = time.time()
        total_combinations = len(problems_to_run) * len(solvers_to_run)
        logger.info(f"Running {total_combinations} problem-solver combinations...")
        
        success_count = 0
        for i, problem_name in enumerate(problems_to_run):
            for j, solver_name in enumerate(solvers_to_run):
                combination_num = i * len(solvers_to_run) + j + 1
                logger.info(f"[{combination_num}/{total_combinations}] Running {solver_name} on {problem_name}")
                
                try:
                    runner.run_single_benchmark(problem_name, solver_name)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to run {solver_name} on {problem_name}: {e}")
        
        duration = time.time() - start_time
        success_rate = (success_count / total_combinations) * 100 if total_combinations > 0 else 0
        
        logger.info(f"Benchmark completed in {duration:.2f}s")
        logger.info(f"Success rate: {success_rate:.1f}% ({success_count}/{total_combinations})")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        return False


def generate_reports() -> bool:
    """Generate HTML reports from benchmark results."""
    
    logger = get_logger("reporting")
    
    try:
        logger.info("Starting report generation...")
        
        # Check if database exists
        database_path = "database/results.db"
        if not Path(database_path).exists():
            logger.error(f"Database not found: {database_path}")
            logger.error("Please run benchmarks first to generate data.")
            return False
        
        # First, publish data files (JSON/CSV exports)
        from scripts.reporting.data_publisher import DataPublisher
        
        data_dir = "docs/data"
        logger.info("Publishing data files...")
        
        data_publisher = DataPublisher(
            db_path=database_path,
            output_dir=data_dir
        )
        
        if not data_publisher.publish_all_data():
            logger.error("Failed to publish data files")
            return False
        
        logger.info("Data files published successfully")
        
        # Then, generate HTML reports
        output_dir = "docs"
        generator = SimpleHTMLGenerator(
            data_dir=data_dir,
            output_dir=output_dir
        )
        
        # Generate all reports
        start_time = time.time()
        success = generator.generate_all_html()
        duration = time.time() - start_time
        
        if success:
            logger.info(f"Reports generated successfully in {duration:.2f}s")
            logger.info(f"Output directory: {output_dir}/")
        else:
            logger.error("Failed to generate HTML reports")
            return False
        
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
  python main.py --all                          # Run benchmarks and generate reports
  python main.py --benchmark                    # Run benchmarks only
  python main.py --report                       # Generate reports only
  python main.py --validate                     # Validate environment setup
  python main.py --benchmark --problems DIMACS  # Run all DIMACS problems
  python main.py --benchmark --problems nb,arch0,simple_lp_test  # Run specific problems
  python main.py --benchmark --problems DIMACS,simple_lp_test  # Mix library and individual problems
  python main.py --benchmark --solvers cvxpy_clarabel,scipy_linprog  # Run specific solvers
  python main.py --all --problems nb --solvers cvxpy_clarabel  # Run one problem with one solver
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
        help='Validate environment setup only'
    )
    
    # Options
    parser.add_argument(
        '--problems', '-p',
        type=str,
        help='Comma-separated list of problem names or library names (DIMACS, SDPLIB, internal) to run'
    )
    
    parser.add_argument(
        '--solvers', '-s',
        type=str,
        help='Comma-separated list of solvers to run'
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
        
        if args.validate:
            print("Environment validation completed successfully")
            success = True
            
        elif args.benchmark:
            success = run_benchmark(
                problems=problems,
                solvers=solvers
            )
            
        elif args.report:
            success = generate_reports()
            
        elif args.all:
            # Run benchmarks first
            benchmark_success = run_benchmark(
                problems=problems,
                solvers=solvers
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