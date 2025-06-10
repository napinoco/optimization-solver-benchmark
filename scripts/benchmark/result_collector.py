import sqlite3
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.solver_interface import SolverResult
from scripts.benchmark.environment_info import collect_environment_info
from scripts.database.models import get_database_path
from scripts.utils.logger import get_logger
from scripts.utils.validation import ResultValidator, create_default_validator

logger = get_logger("result_collector")

class ResultCollector:
    """Collects and stores benchmark results in the database."""
    
    def __init__(self, db_path: Optional[str] = None, enable_validation: bool = True):
        """
        Initialize result collector.
        
        Args:
            db_path: Path to database file. If None, uses default path.
            enable_validation: Whether to enable result validation
        """
        self.db_path = db_path or get_database_path()
        self.logger = get_logger("result_collector")
        self.enable_validation = enable_validation
        
        if self.enable_validation:
            self.validator = create_default_validator()
            self.logger.info(f"Initialized result collector with database: {self.db_path} (validation enabled)")
        else:
            self.validator = None
            self.logger.info(f"Initialized result collector with database: {self.db_path} (validation disabled)")
    
    def create_benchmark_session(self, environment_info: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a new benchmark session and return its ID.
        
        Args:
            environment_info: Environment information dict. If None, collects automatically.
            
        Returns:
            benchmark_id: ID of the created benchmark session
        """
        if environment_info is None:
            environment_info = collect_environment_info()
        
        timestamp = datetime.now().isoformat()
        env_info_json = json.dumps(environment_info, default=str)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO benchmarks (timestamp, environment_info)
                   VALUES (?, ?)""",
                (timestamp, env_info_json)
            )
            benchmark_id = cursor.lastrowid
            conn.commit()
        
        self.logger.info(f"Created benchmark session {benchmark_id} at {timestamp}")
        return benchmark_id
    
    def store_result(self, benchmark_id: int, result: SolverResult, timeout: Optional[float] = None) -> int:
        """
        Store a single solver result in the database with optional validation.
        
        Args:
            benchmark_id: ID of the benchmark session
            result: SolverResult to store
            timeout: Configured timeout for validation (optional)
            
        Returns:
            result_id: ID of the stored result
        """
        # Validate result if validation is enabled
        if self.enable_validation and self.validator:
            result_dict = {
                'solver_name': result.solver_name,
                'problem_name': result.problem_name,
                'solve_time': result.solve_time,
                'status': result.status,
                'objective_value': result.objective_value
            }
            
            validation_result = self.validator.validate_single_result(result_dict, timeout)
            
            # Log validation warnings/errors
            if validation_result.errors:
                self.logger.error(f"Validation errors for {result.solver_name} on {result.problem_name}: "
                                f"{validation_result.errors}")
            if validation_result.warnings:
                self.logger.warning(f"Validation warnings for {result.solver_name} on {result.problem_name}: "
                                   f"{validation_result.warnings}")
            
            # Optionally stop storage if validation fails critically
            if not validation_result.is_valid and validation_result.severity == 'error':
                self.logger.error(f"Refusing to store invalid result: {validation_result.errors}")
                raise ValueError(f"Result validation failed: {validation_result.errors}")
        
        # Convert solver_info to JSON if it exists
        solver_info_json = None
        if result.solver_info:
            solver_info_json = json.dumps(result.solver_info, default=str)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO results 
                   (benchmark_id, solver_name, problem_name, solve_time, status, 
                    objective_value, duality_gap, iterations, error_message, solver_info)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    benchmark_id,
                    result.solver_name,
                    result.problem_name,
                    result.solve_time,
                    result.status,
                    result.objective_value,
                    result.duality_gap,
                    result.iterations,
                    result.error_message,
                    solver_info_json
                )
            )
            result_id = cursor.lastrowid
            conn.commit()
        
        self.logger.debug(f"Stored result {result_id}: {result.solver_name} on {result.problem_name} "
                         f"({result.status}, {result.solve_time:.3f}s)")
        return result_id
    
    def store_problem_info(self, name: str, problem_class: str, file_path: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store problem information in the database.
        
        Args:
            name: Problem name
            problem_class: Problem class (LP, QP, SDP, SOCP)
            file_path: Path to problem file
            metadata: Additional metadata about the problem
            
        Returns:
            problem_id: ID of the stored problem
        """
        # Check if problem already exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM problems WHERE name = ? AND problem_class = ?",
                (name, problem_class)
            )
            existing = cursor.fetchone()
            
            if existing:
                self.logger.debug(f"Problem {name} ({problem_class}) already exists with ID {existing[0]}")
                return existing[0]
        
        # Store new problem
        metadata_json = None
        if metadata:
            metadata_json = json.dumps(metadata, default=str)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO problems (name, problem_class, file_path, metadata)
                   VALUES (?, ?, ?, ?)""",
                (name, problem_class, file_path, metadata_json)
            )
            problem_id = cursor.lastrowid
            conn.commit()
        
        self.logger.info(f"Stored problem {name} ({problem_class}) with ID {problem_id}")
        return problem_id
    
    def store_solver_info(self, name: str, version: str, environment: str,
                         metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store solver information in the database.
        
        Args:
            name: Solver name
            version: Solver version
            environment: Environment (e.g., 'python', 'octave')
            metadata: Additional metadata about the solver
            
        Returns:
            solver_id: ID of the stored solver
        """
        # Check if solver already exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM solvers WHERE name = ? AND version = ? AND environment = ?",
                (name, version, environment)
            )
            existing = cursor.fetchone()
            
            if existing:
                self.logger.debug(f"Solver {name} v{version} ({environment}) already exists with ID {existing[0]}")
                return existing[0]
        
        # Store new solver
        metadata_json = None
        if metadata:
            metadata_json = json.dumps(metadata, default=str)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO solvers (name, version, environment, metadata)
                   VALUES (?, ?, ?, ?)""",
                (name, version, environment, metadata_json)
            )
            solver_id = cursor.lastrowid
            conn.commit()
        
        self.logger.info(f"Stored solver {name} v{version} ({environment}) with ID {solver_id}")
        return solver_id
    
    def get_benchmark_results(self, benchmark_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all results for a specific benchmark session.
        
        Args:
            benchmark_id: ID of the benchmark session
            
        Returns:
            List of result dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            cursor.execute(
                """SELECT r.*, b.timestamp, b.environment_info
                   FROM results r
                   JOIN benchmarks b ON r.benchmark_id = b.id
                   WHERE r.benchmark_id = ?
                   ORDER BY r.id""",
                (benchmark_id,)
            )
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result_dict = dict(row)
            # Parse JSON fields
            if result_dict['solver_info']:
                result_dict['solver_info'] = json.loads(result_dict['solver_info'])
            if result_dict['environment_info']:
                result_dict['environment_info'] = json.loads(result_dict['environment_info'])
            results.append(result_dict)
        
        self.logger.debug(f"Retrieved {len(results)} results for benchmark {benchmark_id}")
        return results
    
    def get_latest_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent benchmark results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of result dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """SELECT r.*, b.timestamp, b.environment_info
                   FROM results r
                   JOIN benchmarks b ON r.benchmark_id = b.id
                   ORDER BY b.timestamp DESC, r.id DESC
                   LIMIT ?""",
                (limit,)
            )
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result_dict = dict(row)
            # Parse JSON fields
            if result_dict['solver_info']:
                result_dict['solver_info'] = json.loads(result_dict['solver_info'])
            if result_dict['environment_info']:
                result_dict['environment_info'] = json.loads(result_dict['environment_info'])
            results.append(result_dict)
        
        self.logger.debug(f"Retrieved {len(results)} latest results")
        return results
    
    def get_solver_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all solvers.
        
        Returns:
            Dictionary with solver statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get solver performance summary
            cursor.execute(
                """SELECT 
                     solver_name,
                     COUNT(*) as total_runs,
                     AVG(solve_time) as avg_time,
                     MIN(solve_time) as min_time,
                     MAX(solve_time) as max_time,
                     SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) as optimal_count,
                     SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                     SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) as timeout_count
                   FROM results
                   GROUP BY solver_name
                   ORDER BY avg_time"""
            )
            solver_stats = [dict(zip([col[0] for col in cursor.description], row)) 
                           for row in cursor.fetchall()]
            
            # Add success rate calculation
            for stats in solver_stats:
                if stats['total_runs'] > 0:
                    stats['success_rate'] = stats['optimal_count'] / stats['total_runs']
                else:
                    stats['success_rate'] = 0.0
        
        summary = {
            'solver_statistics': solver_stats,
            'total_solvers': len(solver_stats),
            'total_results': sum(stats['total_runs'] for stats in solver_stats)
        }
        
        self.logger.debug(f"Generated solver summary with {len(solver_stats)} solvers")
        return summary
    
    def cleanup_old_results(self, keep_days: int = 30) -> int:
        """
        Clean up old benchmark results to manage database size.
        
        Args:
            keep_days: Number of days of results to keep
            
        Returns:
            Number of deleted results
        """
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - keep_days)
        cutoff_iso = cutoff_date.isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # First, delete old results
            cursor.execute(
                """DELETE FROM results 
                   WHERE benchmark_id IN (
                       SELECT id FROM benchmarks WHERE timestamp < ?
                   )""",
                (cutoff_iso,)
            )
            deleted_results = cursor.rowcount
            
            # Then delete old benchmark sessions
            cursor.execute("DELETE FROM benchmarks WHERE timestamp < ?", (cutoff_iso,))
            deleted_benchmarks = cursor.rowcount
            
            conn.commit()
        
        self.logger.info(f"Cleaned up {deleted_results} results and {deleted_benchmarks} "
                        f"benchmark sessions older than {keep_days} days")
        return deleted_results
    
    def validate_stored_results(self, benchmark_id: Optional[int] = None, 
                               timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate results already stored in the database.
        
        Args:
            benchmark_id: Specific benchmark to validate (None for all recent results)
            timeout: Timeout value for validation
            
        Returns:
            Dictionary with validation summary
        """
        if not self.enable_validation or not self.validator:
            self.logger.warning("Validation is disabled, cannot validate stored results")
            return {'validation_enabled': False}
        
        # Get results to validate
        if benchmark_id is not None:
            results = self.get_benchmark_results(benchmark_id)
        else:
            results = self.get_latest_results(limit=100)
        
        if not results:
            self.logger.info("No results found to validate")
            return {'validation_enabled': True, 'results_count': 0}
        
        # Convert to validation format
        validation_results = []
        for result in results:
            result_dict = {
                'solver_name': result['solver_name'],
                'problem_name': result['problem_name'],
                'solve_time': result['solve_time'],
                'status': result['status'],
                'objective_value': result['objective_value']
            }
            validation_results.append(result_dict)
        
        # Perform validation
        individual_validations, consistency_validation = self.validator.validate_batch_results(
            validation_results, timeout
        )
        
        # Summarize results
        total_results = len(individual_validations)
        valid_results = sum(1 for v in individual_validations if v.is_valid)
        error_results = sum(1 for v in individual_validations if v.severity == 'error')
        warning_results = sum(1 for v in individual_validations if v.severity == 'warning')
        
        # Collect error and warning details
        all_errors = []
        all_warnings = []
        for i, validation in enumerate(individual_validations):
            if validation.errors:
                result_info = f"{results[i]['solver_name']} on {results[i]['problem_name']}"
                all_errors.extend([f"{result_info}: {error}" for error in validation.errors])
            if validation.warnings:
                result_info = f"{results[i]['solver_name']} on {results[i]['problem_name']}"
                all_warnings.extend([f"{result_info}: {warning}" for warning in validation.warnings])
        
        # Add consistency validation results
        if consistency_validation.errors:
            all_errors.extend(consistency_validation.errors)
        if consistency_validation.warnings:
            all_warnings.extend(consistency_validation.warnings)
        
        summary = {
            'validation_enabled': True,
            'results_count': total_results,
            'valid_results': valid_results,
            'error_results': error_results,
            'warning_results': warning_results,
            'success_rate': valid_results / total_results if total_results > 0 else 0,
            'all_errors': all_errors,
            'all_warnings': all_warnings,
            'consistency_validation': {
                'is_valid': consistency_validation.is_valid,
                'errors': consistency_validation.errors,
                'warnings': consistency_validation.warnings
            }
        }
        
        self.logger.info(f"Validation completed: {valid_results}/{total_results} valid results, "
                        f"{error_results} errors, {warning_results} warnings")
        
        if all_errors:
            self.logger.error(f"Found {len(all_errors)} validation errors in stored results")
        if all_warnings:
            self.logger.warning(f"Found {len(all_warnings)} validation warnings in stored results")
        
        return summary

if __name__ == "__main__":
    # Test script to verify result collector
    try:
        print("Testing Result Collector...")
        
        # Import required components for testing
        from scripts.benchmark.problem_loader import load_problem
        from scripts.solvers.python.scipy_runner import ScipySolver
        from scripts.database.models import create_database
        
        # Ensure database exists
        print("\nCreating database...")
        db_path = create_database()
        print(f"Database created at: {db_path}")
        
        # Test result collector initialization
        print("\nTesting collector initialization:")
        collector = ResultCollector()
        print(f"✓ Collector initialized with database: {collector.db_path}")
        
        # Test benchmark session creation
        print("\nTesting benchmark session creation:")
        benchmark_id = collector.create_benchmark_session()
        print(f"✓ Created benchmark session: {benchmark_id}")
        
        # Test problem storage
        print("\nTesting problem storage:")
        problem_id = collector.store_problem_info(
            name="test_problem",
            problem_class="LP",
            file_path="problems/light_set/lp/simple_lp.mps",
            metadata={"description": "Simple test LP problem"}
        )
        print(f"✓ Stored problem with ID: {problem_id}")
        
        # Test solver storage
        print("\nTesting solver storage:")
        solver_id = collector.store_solver_info(
            name="SciPy",
            version="1.15.3",
            environment="python",
            metadata={"method": "highs"}
        )
        print(f"✓ Stored solver with ID: {solver_id}")
        
        # Test actual benchmark result storage
        print("\nTesting result storage:")
        try:
            # Load a problem and solve it
            problem = load_problem("simple_lp")
            solver = ScipySolver()
            result = solver.solve_with_timeout(problem)
            
            # Store the result
            result_id = collector.store_result(benchmark_id, result)
            print(f"✓ Stored result with ID: {result_id}")
            print(f"  Solver: {result.solver_name}")
            print(f"  Problem: {result.problem_name}")
            print(f"  Status: {result.status}")
            print(f"  Time: {result.solve_time:.3f}s")
            
        except Exception as e:
            print(f"✗ Result storage test failed: {e}")
        
        # Test result retrieval
        print("\nTesting result retrieval:")
        try:
            results = collector.get_benchmark_results(benchmark_id)
            print(f"✓ Retrieved {len(results)} results for benchmark {benchmark_id}")
            
            if results:
                latest_result = results[0]
                print(f"  Latest result: {latest_result['solver_name']} on {latest_result['problem_name']}")
                print(f"  Status: {latest_result['status']}, Time: {latest_result['solve_time']:.3f}s")
            
        except Exception as e:
            print(f"✗ Result retrieval test failed: {e}")
        
        # Test summary statistics
        print("\nTesting summary statistics:")
        try:
            summary = collector.get_solver_summary()
            print(f"✓ Generated summary with {summary['total_solvers']} solvers")
            print(f"  Total results: {summary['total_results']}")
            
            if summary['solver_statistics']:
                for stats in summary['solver_statistics']:
                    print(f"  {stats['solver_name']}: {stats['total_runs']} runs, "
                          f"{stats['success_rate']:.1%} success rate, "
                          f"{stats['avg_time']:.3f}s avg time")
            
        except Exception as e:
            print(f"✗ Summary statistics test failed: {e}")
        
        # Test multiple results
        print("\nTesting multiple result storage:")
        try:
            # Store a few more results for testing
            for i in range(3):
                result2 = solver.solve_with_timeout(problem)
                collector.store_result(benchmark_id, result2)
            
            # Test latest results
            latest_results = collector.get_latest_results(limit=5)
            print(f"✓ Retrieved {len(latest_results)} latest results")
            
        except Exception as e:
            print(f"✗ Multiple results test failed: {e}")
        
        print("\n✓ All result collector tests completed!")
        
    except Exception as e:
        logger.error(f"Result collector test failed: {e}")
        raise