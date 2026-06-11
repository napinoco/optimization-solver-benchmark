"""
Result Processor for Simplified Reporting
=========================================

Processes benchmark results from database for simplified HTML report generation.
Extracts latest results using commit_hash and environment_info with timestamp tiebreaker.
"""

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("result_processor")


@dataclass
class BenchmarkResult:
    """Single denormalized benchmark result model"""

    # Primary key
    id: Optional[int] = None

    # Solver information
    solver_name: str = ""
    solver_version: str = ""

    # Problem information
    problem_library: str = ""  # 'internal', 'DIMACS', 'SDPLIB'
    problem_name: str = ""
    problem_type: str = ""     # 'LP', 'QP', 'SOCP', 'SDP'

    # Environment and execution context
    environment_info: Dict[str, Any] = None
    commit_hash: str = ""
    timestamp: Optional[datetime] = None

    # Standardized solver results
    solve_time: Optional[float] = None
    status: Optional[str] = None
    primal_objective_value: Optional[float] = None
    dual_objective_value: Optional[float] = None
    duality_gap: Optional[float] = None
    primal_infeasibility: Optional[float] = None
    dual_infeasibility: Optional[float] = None
    iterations: Optional[int] = None
    memo: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'solver_name': self.solver_name,
            'solver_version': self.solver_version,
            'problem_library': self.problem_library,
            'problem_name': self.problem_name,
            'problem_type': self.problem_type,
            'environment_info': self.get_sanitized_environment_info(),
            'commit_hash': self.commit_hash,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'solve_time': self.solve_time,
            'status': self.status,
            'primal_objective_value': self.primal_objective_value,
            'dual_objective_value': self.dual_objective_value,
            'duality_gap': self.duality_gap,
            'primal_infeasibility': self.primal_infeasibility,
            'dual_infeasibility': self.dual_infeasibility,
            'iterations': self.iterations,
            'memo': self.memo
        }

    def get_sanitized_environment_info(self, env_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get sanitized environment info to remove sensitive information - public method"""
        if env_info is None:
            env_info = self.environment_info
        return self._sanitize_environment_info(env_info)

    def _sanitize_environment_info(self, env_info: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize environment info to remove sensitive information - matches database sanitization"""
        if not env_info:
            return {}

        # Create sanitized copy with minimal information for privacy protection
        sanitized = {}

        # CPU info - keep essential performance info only
        if 'cpu' in env_info:
            cpu = env_info['cpu']
            sanitized['cpu'] = {
                'cpu_count': cpu.get('cpu_count'),
                'cpu_count_physical': cpu.get('cpu_count_physical'),
                'processor': cpu.get('processor'),
                'architecture': cpu.get('architecture') or env_info.get('os', {}).get('architecture')
            }

        # Memory info - keep total only (performance relevant)
        if 'memory' in env_info:
            memory = env_info['memory']
            sanitized['memory'] = {
                'total_gb': memory.get('total_gb')
            }

        # OS info - keep basic system info only (no version details that could identify specific systems)
        if 'os' in env_info:
            os_info = env_info['os']
            sanitized['os'] = {
                'system': os_info.get('system'),      # Darwin, Linux, Windows
                'machine': os_info.get('machine'),    # arm64, x86_64
                'release': os_info.get('release')     # Keep for compatibility testing
            }
            # Remove: architecture (duplicated), platform (too detailed), version (too specific)

        # Python info - keep version only (remove all paths)
        if 'python' in env_info:
            python = env_info['python']
            sanitized['python'] = {
                'implementation': python.get('implementation'),  # CPython, PyPy
                'version': python.get('version'),                # 3.12.2
                'version_info': python.get('version_info')       # 3.12.2
            }
            # Remove: executable (contains user paths)

        # Git info - keep commit hash (essential for result tracking and table restoration)
        if 'git' in env_info:
            git = env_info['git']
            if git.get('commit_hash'):
                sanitized['git'] = {
                    'commit_hash': git.get('commit_hash')
                }
            # Remove: available, branch, is_dirty (privacy/security sensitive)
            # Keep: commit_hash (not sensitive and essential for data integrity)

        # Timezone - UTC ONLY (remove all location-specific timezone info)
        # Replace all timezone info with UTC standard to prevent location identification
        sanitized['timezone'] = {
            'timezone_name': 'UTC',
            'utc_offset_hours': 0.0
        }
        # Remove all original timezone data to prevent location identification

        # Timestamp - keep original timestamp (should be in UTC for consistency)
        if 'timestamp' in env_info:
            sanitized['timestamp'] = env_info['timestamp']

        return sanitized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary (e.g., from database row)"""
        result = cls()
        for key, value in data.items():
            if hasattr(result, key):
                if key == 'timestamp' and value:
                    result.timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
                elif key == 'environment_info' and isinstance(value, str):
                    try:
                        result.environment_info = json.loads(value) if value and value.strip() else {}
                    except json.JSONDecodeError:
                        result.environment_info = {}  # Default to empty dict if not valid JSON
                elif key == 'memo' and isinstance(value, str):
                    try:
                        result.memo = json.loads(value) if value and value.strip() else None
                    except json.JSONDecodeError:
                        result.memo = value  # Keep as string if not valid JSON
                else:
                    setattr(result, key, value)
        return result


class ResultProcessor:
    """Process latest results from database for simplified reporting"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize result processor with database path"""
        if db_path is None:
            db_path = project_root / "database" / "results.db"

        self.db_path = str(db_path)
        self.logger = get_logger("result_processor")

    def _load_problem_registry(self) -> Dict[str, Any]:
        """Load problem registry from config/problem_registry.yaml"""
        try:
            config_path = project_root / "config" / "problem_registry.yaml"
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load problem registry: {e}")
            return {'problem_libraries': {}}

    def _load_site_config(self) -> Dict[str, Any]:
        """Load site configuration from config/site_config.yaml"""
        try:
            config_path = project_root / "config" / "site_config.yaml"
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load site config: {e}")
            return {}

    def _get_active_problems(self) -> set:
        """
        Get set of active problem names from problem_registry.yaml.
        Excludes commented out problems.
        
        Returns:
            Set of active problem names
        """
        try:
            registry_path = project_root / "config" / "problem_registry.yaml"
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = yaml.safe_load(f)

            # Get problems that are not commented out
            active_problems = set()
            if 'problem_libraries' in registry:
                for problem_name, config in registry['problem_libraries'].items():
                    # Only include if it's not None (meaning it's not commented out)
                    if config is not None:
                        active_problems.add(problem_name)

            self.logger.debug(f"Found {len(active_problems)} active problems in registry")
            return active_problems

        except Exception as e:
            self.logger.warning(f"Failed to load problem registry: {e}")
            # Return empty set so no filtering occurs
            return set()

    def get_latest_results_for_reporting(self) -> List[BenchmarkResult]:
        """
        Get latest results using commit_hash and environment_info with timestamp tiebreaker.
        Only returns results for problems that are currently active in problem_registry.yaml.
        
        Returns:
            List of BenchmarkResult objects representing the latest benchmark run
        """

        self.logger.info("Extracting latest results for reporting...")

        # Get active problems from registry (exclude commented out problems)
        active_problems = self._get_active_problems()

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query for latest results using latest-per-group approach
                # This ensures we get the latest result for EVERY (solver, problem) combination
                query = """
                    SELECT r1.* FROM results r1
                    INNER JOIN (
                        SELECT solver_name, problem_name, MAX(timestamp) as max_timestamp
                        FROM results
                        GROUP BY solver_name, problem_name
                    ) r2 ON r1.solver_name = r2.solver_name 
                        AND r1.problem_name = r2.problem_name 
                        AND r1.timestamp = r2.max_timestamp
                    ORDER BY problem_library, problem_name, solver_name, id DESC
                """

                cursor = conn.cursor()
                cursor.execute(query)

                # Get column names
                columns = [description[0] for description in cursor.description]

                # Convert rows to BenchmarkResult objects and filter for active problems
                results = []
                excluded_count = 0
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row, strict=False))
                    result = BenchmarkResult.from_dict(row_dict)

                    # Only include results for active problems
                    if not active_problems or result.problem_name in active_problems:
                        results.append(result)
                    else:
                        excluded_count += 1

                self.logger.info(f"Retrieved {len(results)} latest results for reporting")
                if excluded_count > 0:
                    self.logger.info(f"Excluded {excluded_count} results for commented-out problems")
                return results

        except Exception as e:
            self.logger.error(f"Failed to get latest results: {e}")
            return []

    def get_all_results_for_export(self) -> List[BenchmarkResult]:
        """
        Get all results from database for complete export/backup.
        Only returns results for problems that are currently active in problem_registry.yaml.
        
        Returns:
            List of all BenchmarkResult objects for active problems in the database
        """

        self.logger.info("Extracting all results for export...")

        # Get active problems from registry (exclude commented out problems)
        active_problems = self._get_active_problems()

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query for all results ordered by id for database restoration
                # Note: This method returns ALL results, not just latest per group
                query = """
                    SELECT * FROM results 
                    ORDER BY id ASC
                """

                cursor = conn.cursor()
                cursor.execute(query)

                # Get column names
                columns = [description[0] for description in cursor.description]

                # Convert rows to BenchmarkResult objects and filter for active problems
                results = []
                excluded_count = 0
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row, strict=False))
                    result = BenchmarkResult.from_dict(row_dict)

                    # Only include results for active problems
                    if not active_problems or result.problem_name in active_problems:
                        results.append(result)
                    else:
                        excluded_count += 1

                self.logger.info(f"Retrieved {len(results)} total results for export")
                if excluded_count > 0:
                    self.logger.info(f"Excluded {excluded_count} results for commented-out problems")
                return results

        except Exception as e:
            self.logger.error(f"Failed to get all results: {e}")
            return []

    def get_summary_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from results"""

        if not results:
            return {
                'total_results': 0,
                'total_solvers': 0,
                'total_problems': 0,
                'success_rate': 0.0,
                'avg_solve_time': 0.0
            }

        # Calculate basic statistics
        total_results = len(results)
        solvers = set(r.solver_name for r in results)
        problems = set(r.problem_name for r in results)

        # Success rate calculation (exclude UNSUPPORTED from failure count)
        successful_results = [r for r in results if r.status and r.status.upper() == 'OPTIMAL']
        unsupported_results = [r for r in results if r.status and r.status.upper() == 'UNSUPPORTED']
        applicable_results = total_results - len(unsupported_results)  # Results where solver could attempt
        success_rate = len(successful_results) / applicable_results if applicable_results > 0 else 0.0

        # Average solve time calculation
        valid_times = [r.solve_time for r in results if r.solve_time is not None and r.solve_time > 0]
        avg_solve_time = sum(valid_times) / len(valid_times) if valid_times else 0.0

        # Problem type distribution
        problem_types = {}
        for result in results:
            ptype = result.problem_type
            if ptype not in problem_types:
                problem_types[ptype] = 0
            problem_types[ptype] += 1

        # Library distribution
        libraries = {}
        for result in results:
            lib = result.problem_library
            if lib not in libraries:
                libraries[lib] = 0
            libraries[lib] += 1

        # Status distribution
        status_distribution = {}
        for result in results:
            status = result.status or 'UNKNOWN'
            if status not in status_distribution:
                status_distribution[status] = 0
            status_distribution[status] += 1

        return {
            'total_results': total_results,
            'total_solvers': len(solvers),
            'total_problems': len(problems),
            'success_rate': success_rate,
            'avg_solve_time': avg_solve_time,
            'successful_results': len(successful_results),
            'unsupported_results': len(unsupported_results),
            'applicable_results': applicable_results,
            'problem_type_distribution': problem_types,
            'library_distribution': libraries,
            'status_distribution': status_distribution,
            'solver_names': sorted(list(solvers)),
            'problem_names': sorted(list(problems))
        }

    def get_solver_comparison(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """Generate solver comparison data"""

        solver_stats = {}

        for result in results:
            solver = result.solver_name
            if solver not in solver_stats:
                solver_stats[solver] = {
                    'solver_name': solver,
                    'problems_attempted': 0,
                    'problems_applicable': 0,  # Excludes UNSUPPORTED
                    'problems_solved': 0,
                    'problems_unsupported': 0,
                    'solve_times': []
                }

            solver_stats[solver]['problems_attempted'] += 1

            if result.status and result.status.upper() == 'UNSUPPORTED':
                solver_stats[solver]['problems_unsupported'] += 1
            else:
                solver_stats[solver]['problems_applicable'] += 1
                if result.status and result.status.upper() == 'OPTIMAL':
                    solver_stats[solver]['problems_solved'] += 1

            if result.solve_time is not None and result.solve_time > 0:
                solver_stats[solver]['solve_times'].append(result.solve_time)

        # Calculate derived statistics
        comparison_data = []
        for solver_name, stats in solver_stats.items():
            times = stats['solve_times']

            comparison_data.append({
                'solver_name': solver_name,
                'problems_attempted': stats['problems_attempted'],
                'problems_applicable': stats['problems_applicable'],
                'problems_solved': stats['problems_solved'],
                'problems_unsupported': stats['problems_unsupported'],
                'success_rate': stats['problems_solved'] / stats['problems_applicable'] if stats['problems_applicable'] > 0 else 0.0,
                'avg_solve_time': sum(times) / len(times) if times else 0.0,
                'min_solve_time': min(times) if times else 0.0,
                'max_solve_time': max(times) if times else 0.0
            })

        # Sort by success rate, then by average solve time
        comparison_data.sort(key=lambda x: (-x['success_rate'], x['avg_solve_time']))

        return comparison_data

    def get_solver_comparison_by_problem_type(self, results: List[BenchmarkResult]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate solver comparison data grouped by problem type"""

        # Group results by problem type
        by_problem_type = {}
        for result in results:
            problem_type = result.problem_type or 'UNKNOWN'
            if problem_type not in by_problem_type:
                by_problem_type[problem_type] = []
            by_problem_type[problem_type].append(result)

        # Generate comparison for each problem type
        comparison_by_type = {}
        for problem_type, type_results in by_problem_type.items():
            solver_stats = {}

            for result in type_results:
                solver = result.solver_name
                if solver not in solver_stats:
                    solver_stats[solver] = {
                        'solver_name': solver,
                        'problem_type': problem_type,
                        'problems_attempted': 0,
                        'problems_applicable': 0,
                        'problems_solved': 0,
                        'problems_unsupported': 0,
                        'solve_times': []
                    }

                solver_stats[solver]['problems_attempted'] += 1

                if result.status and result.status.upper() == 'UNSUPPORTED':
                    solver_stats[solver]['problems_unsupported'] += 1
                else:
                    solver_stats[solver]['problems_applicable'] += 1
                    if result.status and result.status.upper() == 'OPTIMAL':
                        solver_stats[solver]['problems_solved'] += 1

                if result.solve_time is not None and result.solve_time > 0:
                    solver_stats[solver]['solve_times'].append(result.solve_time)

            # Calculate derived statistics for this problem type
            type_comparison = []
            for solver_name, stats in solver_stats.items():
                times = stats['solve_times']

                type_comparison.append({
                    'solver_name': solver_name,
                    'problem_type': problem_type,
                    'problems_attempted': stats['problems_attempted'],
                    'problems_applicable': stats['problems_applicable'],
                    'problems_solved': stats['problems_solved'],
                    'problems_unsupported': stats['problems_unsupported'],
                    'success_rate': stats['problems_solved'] / stats['problems_applicable'] if stats['problems_applicable'] > 0 else 0.0,
                    'avg_solve_time': sum(times) / len(times) if times else 0.0,
                    'min_solve_time': min(times) if times else 0.0,
                    'max_solve_time': max(times) if times else 0.0
                })

            # Sort by success rate, then by average solve time
            type_comparison.sort(key=lambda x: (-x['success_rate'], x['avg_solve_time']))
            comparison_by_type[problem_type] = type_comparison

        return comparison_by_type

    def get_problem_count_by_library_and_type(self, results: List[BenchmarkResult]) -> Dict[str, Dict[str, int]]:
        """Get problem count grouped by library and problem type"""

        # Use unique problem names to avoid double counting
        unique_problems = {}
        for result in results:
            key = (result.problem_name, result.problem_library, result.problem_type)
            if key not in unique_problems:
                unique_problems[key] = {
                    'problem_name': result.problem_name,
                    'library': result.problem_library or 'unknown',
                    'type': result.problem_type or 'UNKNOWN'
                }

        # Count by library and type
        library_type_counts = {}
        for problem_data in unique_problems.values():
            library = problem_data['library']
            problem_type = problem_data['type']

            if library not in library_type_counts:
                library_type_counts[library] = {}
            if problem_type not in library_type_counts[library]:
                library_type_counts[library][problem_type] = 0

            library_type_counts[library][problem_type] += 1

        return library_type_counts

    def get_best_performers_by_library_and_type(self, results: List[BenchmarkResult]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get best performing solver by library and problem type based on solve time"""

        # Group results by library and problem type
        by_library_type = {}
        for result in results:
            library = result.problem_library or 'unknown'
            problem_type = result.problem_type or 'UNKNOWN'

            key = (library, problem_type)
            if key not in by_library_type:
                by_library_type[key] = []
            by_library_type[key].append(result)

        # Find best performer for each library-type combination
        best_performers = {}
        for (library, problem_type), type_results in by_library_type.items():
            # Calculate average solve time for each solver in this library-type
            solver_times = {}
            for result in type_results:
                if result.solve_time is not None and result.solve_time > 0:
                    solver = result.solver_name
                    if solver not in solver_times:
                        solver_times[solver] = []
                    solver_times[solver].append(result.solve_time)

            # Calculate averages and find best
            solver_averages = {}
            for solver, times in solver_times.items():
                if times:
                    solver_averages[solver] = {
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'count': len(times)
                    }

            # Find best performer (lowest average time with reasonable sample size)
            best_solver = None
            best_time = float('inf')
            for solver, stats in solver_averages.items():
                # Only consider solvers that solved at least 1 problem
                if stats['count'] >= 1 and stats['avg_time'] < best_time:
                    best_time = stats['avg_time']
                    best_solver = solver

            if library not in best_performers:
                best_performers[library] = {}

            best_performers[library][problem_type] = {
                'best_solver': best_solver,
                'avg_time': best_time if best_solver else None,
                'solver_stats': solver_averages
            }

        return best_performers

    def get_results_matrix(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate problems × solvers matrix data with enhanced metadata and sorting"""

        # Get unique problems and solvers
        unique_problems = set(r.problem_name for r in results)
        available_solvers = set(r.solver_name for r in results)

        # Load site config to get solver display order
        site_config = self._load_site_config()
        display_order = site_config.get('solvers', {}).get('display_order', [])

        # Sort solvers according to display_order, with undefined solvers at the end
        solvers = []
        for solver in display_order:
            if solver in available_solvers:
                solvers.append(solver)
                available_solvers.remove(solver)

        # Add any remaining solvers in alphabetical order
        solvers.extend(sorted(available_solvers))

        # Load problem registry to get known objective values
        problem_registry = self._load_problem_registry()

        # Build problem metadata dictionary
        problem_metadata = {}
        for result in results:
            if result.problem_name not in problem_metadata:
                # Get known objective value from problem registry
                known_objective = None
                if result.problem_name in problem_registry.get('problem_libraries', {}):
                    known_objective = problem_registry['problem_libraries'][result.problem_name].get('known_objective_value')

                problem_metadata[result.problem_name] = {
                    'problem_name': result.problem_name,
                    'problem_type': result.problem_type or 'UNKNOWN',
                    'library_name': result.problem_library or 'unknown',
                    'known_objective_value': known_objective
                }

        # Sort problems by library_name, problem_type, problem_name
        problems_sorted = sorted(
            unique_problems,
            key=lambda p: (
                problem_metadata[p]['library_name'],
                problem_metadata[p]['problem_type'],
                problem_metadata[p]['problem_name']
            )
        )

        # Create matrix with problem metadata
        matrix = {}
        for problem in problems_sorted:
            matrix[problem] = {}
            for solver in solvers:
                matrix[problem][solver] = None

        # Fill matrix with results - ensure we only keep the latest result for each (problem, solver) pair
        # Group results by (problem, solver) and keep only the latest timestamp
        latest_results = {}
        for result in results:
            key = (result.problem_name, result.solver_name)
            if key not in latest_results or result.timestamp > latest_results[key].timestamp:
                latest_results[key] = result

        # Fill matrix with latest results only
        for result in latest_results.values():
            problem = result.problem_name
            solver = result.solver_name

            matrix[problem][solver] = {
                'status': result.status,
                'solve_time': result.solve_time,
                'objective_value': result.primal_objective_value,
                'iterations': result.iterations
            }

        return {
            'problems': problems_sorted,
            'solvers': solvers,
            'matrix': matrix,
            'problem_metadata': problem_metadata
        }


def main():
    """Test the result processor"""
    processor = ResultProcessor()

    print("Testing Result Processor...")
    results = processor.get_latest_results_for_reporting()
    print(f"Retrieved {len(results)} results")

    if results:
        print("\nSummary Statistics:")
        summary = processor.get_summary_statistics(results)
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\nSolver Comparison:")
        comparison = processor.get_solver_comparison(results)
        for solver_data in comparison:
            print(f"  {solver_data['solver_name']}: {solver_data['success_rate']:.2%} success, {solver_data['avg_solve_time']:.4f}s avg")


if __name__ == "__main__":
    main()
