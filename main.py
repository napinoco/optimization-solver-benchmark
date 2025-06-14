#!/usr/bin/env python3
"""
Optimization Solver Benchmark - Main Entry Point

This script provides a unified interface to run benchmarks, generate reports,
and manage the complete optimization solver benchmark system.

Usage Examples:
    python main.py --help                    # Show help
    python main.py --benchmark               # Run benchmarks only
    python main.py --report                  # Generate reports only
    python main.py --all                     # Run benchmarks and generate reports
    python main.py --config custom.yaml     # Use custom configuration
    python main.py --problem-set large_set  # Use specific problem set
    python main.py --solvers scipy,cvxpy    # Run specific solvers
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent / "scripts"))

try:
    from benchmark.runner import BenchmarkRunner
    from reporting.simple_html_generator import SimpleHTMLGenerator
    from utils.config_loader import load_config
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the repository root directory.")
    sys.exit(1)


class BenchmarkOrchestrator:
    """Main orchestrator for the benchmark system"""
    
    def __init__(self, config_path: str = "config/benchmark_config.yaml"):
        """
        Initialize the benchmark orchestrator
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        
        # Support environment variable overrides for CI/CD
        self.ci_mode = os.getenv('GITHUB_ACTIONS', '').lower() == 'true'
        
        if self.ci_mode:
            print("Running in GitHub Actions mode")
        self.config = None
        self.logger = None
        
    def setup(self):
        """Setup logging and load configuration"""
        # Setup logging
        self.logger = setup_logger()
        self.logger.info("Benchmark system starting up...")
        
        # Load configuration
        try:
            # The load_config function expects just the filename, not the full path
            config_filename = Path(self.config_path).name
            self.config = load_config(config_filename)
            self.logger.info(f"Configuration loaded from: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def run_benchmark(self, problem_set: Optional[str] = None, 
                     solvers: Optional[List[str]] = None) -> bool:
        """
        Run the benchmark suite
        
        Args:
            problem_set: Problem set to use (overrides config)
            solvers: List of solver names to use (overrides config)
            
        Returns:
            True if benchmarks completed successfully
        """
        try:
            self.logger.info("Starting benchmark execution...")
            
            # Create benchmark runner
            runner = BenchmarkRunner()
            
            # Setup solvers
            runner.setup_solvers()
            
            # Load problems
            problem_set_to_use = problem_set or "light_set"
            if problem_set:
                self.logger.info(f"Using problem set: {problem_set}")
            runner.load_problems(problem_set_to_use)
            
            # Log solvers if specified
            if solvers:
                self.logger.info(f"Using solvers: {', '.join(solvers)}")
            
            # Run the benchmark
            start_time = time.time()
            results = runner.run_sequential_benchmark(solver_names=solvers)
            duration = time.time() - start_time
            
            # Log results
            if results:
                self.logger.info(f"Benchmark completed successfully in {duration:.2f}s")
                self.logger.info(f"Total results: {len(results)}")
                
                # Log success rate
                successful = sum(1 for r in results if r.status == 'optimal')
                success_rate = successful / len(results) * 100 if results else 0
                self.logger.info(f"Success rate: {success_rate:.1f}% ({successful}/{len(results)})")
                
                return True
            else:
                self.logger.warning("Benchmark completed but no results were generated")
                return False
                
        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}")
            return False
    
    def generate_reports(self) -> bool:
        """
        Generate HTML reports from benchmark results
        
        Returns:
            True if reports generated successfully
        """
        try:
            self.logger.info("Starting report generation...")
            
            # Get paths from config
            database_path = self.config.get('database', {}).get('path', 'database/results.db')
            output_dir = self.config.get('output', {}).get('reports_dir', 'docs')
            
            # Verify database exists
            if not Path(database_path).exists():
                self.logger.error(f"Database not found: {database_path}")
                self.logger.error("Please run benchmarks first to generate data.")
                return False
            
            # First publish data files (JSON exports) that HTML generator needs
            from reporting.data_publisher import DataPublisher
            
            self.logger.info("📊 Publishing data files...")
            data_publisher = DataPublisher(
                db_path=database_path,
                output_dir=str(Path(output_dir) / "data")
            )
            
            if not data_publisher.publish_all_data():
                self.logger.error("❌ Failed to publish data files - skipping HTML generation")
                return False
            
            self.logger.info("✅ Data files published successfully")
            
            # Create HTML generator
            data_dir = Path(output_dir) / "data"
            generator = SimpleHTMLGenerator(
                data_dir=str(data_dir),
                output_dir=str(output_dir)
            )
            
            # Generate all reports
            start_time = time.time()
            success = generator.generate_all_html()
            duration = time.time() - start_time
            
            # Log results
            if success:
                self.logger.info(f"Reports generated successfully in {duration:.2f}s")
                self.logger.info(f"  dashboard: {output_dir}/index.html")
                self.logger.info(f"  solver_comparison: {output_dir}/solver_comparison.html")
                self.logger.info(f"  problem_analysis: {output_dir}/problem_analysis.html")
                self.logger.info(f"  environment_info: {output_dir}/environment_info.html")
            else:
                self.logger.error("Failed to generate reports")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return False
    
    def run_all(self, problem_set: Optional[str] = None, 
               solvers: Optional[List[str]] = None) -> bool:
        """
        Run complete benchmark suite and generate reports
        
        Args:
            problem_set: Problem set to use
            solvers: List of solver names to use
            
        Returns:
            True if both benchmarks and reports completed successfully
        """
        self.logger.info("Running complete benchmark and reporting pipeline...")
        
        # Run benchmarks
        benchmark_success = self.run_benchmark(problem_set, solvers)
        if not benchmark_success:
            self.logger.error("Benchmark failed, skipping report generation")
            return False
        
        # Generate reports
        report_success = self.generate_reports()
        if not report_success:
            self.logger.error("Report generation failed")
            return False
        
        self.logger.info("Complete benchmark pipeline completed successfully!")
        return True
    
    def validate_environment(self) -> bool:
        """
        Validate that the environment is properly set up
        
        Returns:
            True if environment is valid
        """
        issues = []
        
        # Check for required directories
        required_dirs = ['config', 'scripts', 'problems']
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                issues.append(f"Missing directory: {dir_name}")
        
        # Check for configuration file
        if not Path(self.config_path).exists():
            issues.append(f"Configuration file not found: {self.config_path}")
        
        # Check for database directory
        db_dir = Path("database")
        if not db_dir.exists():
            self.logger.info("Creating database directory...")
            db_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for output directory
        output_dir = Path(self.config.get('output', {}).get('reports_dir', 'docs') if self.config else 'docs')
        if not output_dir.exists():
            self.logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        if issues:
            for issue in issues:
                self.logger.error(issue)
            return False
        
        self.logger.info("Environment validation passed")
        return True


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Optimization Solver Benchmark System",
        epilog="""
Examples:
  python main.py --all                          # Run benchmarks and generate reports
  python main.py --benchmark                    # Run benchmarks only
  python main.py --report                       # Generate reports only
  python main.py --benchmark --solvers scipy,cvxpy  # Run specific solvers
  python main.py --all --problem-set large_set   # Use different problem set
  python main.py --config custom.yaml --all     # Use custom configuration
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/benchmark_config.yaml',
        help='Path to configuration file (default: config/benchmark_config.yaml)'
    )
    
    # Benchmark options
    parser.add_argument(
        '--problem-set', '-p',
        type=str,
        help='Problem set to use (overrides config)'
    )
    
    parser.add_argument(
        '--solvers', '-s',
        type=str,
        help='Comma-separated list of solvers to run (overrides config)'
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
    
    return parser


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle solver list parsing
    solvers = None
    if args.solvers:
        solvers = [s.strip() for s in args.solvers.split(',')]
    
    try:
        # Create orchestrator
        orchestrator = BenchmarkOrchestrator(config_path=args.config)
        
        # Setup logging and configuration
        orchestrator.setup()
        
        # Adjust logging level based on arguments
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            orchestrator.logger.debug("Verbose logging enabled")
        elif args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Validate environment
        if not orchestrator.validate_environment():
            orchestrator.logger.error("Environment validation failed")
            sys.exit(1)
        
        # Execute requested operation
        success = False
        
        if args.validate:
            orchestrator.logger.info("Environment validation completed successfully")
            success = True
            
        elif args.benchmark:
            success = orchestrator.run_benchmark(
                problem_set=args.problem_set,
                solvers=solvers
            )
            
        elif args.report:
            success = orchestrator.generate_reports()
            
        elif args.all:
            success = orchestrator.run_all(
                problem_set=args.problem_set,
                solvers=solvers
            )
        
        # Exit with appropriate code
        if success:
            orchestrator.logger.info("Operation completed successfully")
            sys.exit(0)
        else:
            orchestrator.logger.error("Operation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()