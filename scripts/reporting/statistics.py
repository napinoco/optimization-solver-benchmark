"""
Statistics Calculation Module

Provides comprehensive statistical analysis functions for benchmark results.
Calculates performance metrics, trends, and comparative statistics.
Enhanced with backend comparison and validation reporting.
"""

import sqlite3
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
# Use manual statistics calculations to avoid import conflicts
from datetime import datetime, timedelta
import math

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logger = logging.getLogger(__name__)


def calculate_mean(values: List[float]) -> float:
    """Calculate mean of a list of values"""
    return sum(values) / len(values) if values else 0.0


def calculate_median(values: List[float]) -> float:
    """Calculate median of a list of values"""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]


def calculate_stdev(values: List[float]) -> float:
    """Calculate standard deviation of a list of values"""
    if len(values) < 2:
        return 0.0
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


@dataclass
class SolverStatistics:
    """Comprehensive statistics for a single solver"""
    name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    avg_solve_time: float
    median_solve_time: float
    min_solve_time: float
    max_solve_time: float
    std_dev_solve_time: float
    problems_solved: int
    total_solve_time: float
    avg_objective_value: Optional[float]
    reliability_score: float  # Composite score


@dataclass
class ProblemStatistics:
    """Comprehensive statistics for a single problem"""
    name: str
    problem_type: str
    total_attempts: int
    successful_attempts: int
    success_rate: float
    avg_solve_time: float
    min_solve_time: float
    max_solve_time: float
    fastest_solver: Optional[str]
    slowest_solver: Optional[str]
    difficulty_score: float  # Based on success rate and solve times


@dataclass
class BenchmarkSummary:
    """Overall benchmark summary statistics"""
    total_runs: int
    total_solvers: int
    total_problems: int
    overall_success_rate: float
    total_solve_time: float
    avg_solve_time: float
    fastest_solver: Optional[str]
    most_reliable_solver: Optional[str]
    hardest_problem: Optional[str]
    easiest_problem: Optional[str]


class StatisticsCalculator:
    """Calculates comprehensive statistics from benchmark database"""
    
    def __init__(self, database_path: str):
        """
        Initialize the statistics calculator
        
        Args:
            database_path: Path to the SQLite database
        """
        self.database_path = database_path
        logger.info(f"Statistics calculator initialized with database: {database_path}")
    
    def _get_database_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def calculate_solver_statistics(self) -> Dict[str, SolverStatistics]:
        """Calculate comprehensive statistics for each solver"""
        logger.info("Calculating solver statistics...")
        
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Get all results grouped by solver
            cursor.execute("""
                SELECT 
                    solver_name,
                    status,
                    solve_time,
                    objective_value
                FROM results
                ORDER BY solver_name, solve_time
            """)
            
            # Group results by solver
            solver_results = {}
            for row in cursor.fetchall():
                solver = row['solver_name']
                if solver not in solver_results:
                    solver_results[solver] = []
                solver_results[solver].append({
                    'status': row['status'],
                    'solve_time': row['solve_time'],
                    'objective_value': row['objective_value']
                })
            
            # Calculate statistics for each solver
            solver_stats = {}
            for solver_name, results in solver_results.items():
                stats = self._calculate_single_solver_stats(solver_name, results)
                solver_stats[solver_name] = stats
            
            logger.info(f"Calculated statistics for {len(solver_stats)} solvers")
            return solver_stats
    
    def _calculate_single_solver_stats(self, solver_name: str, results: List[Dict]) -> SolverStatistics:
        """Calculate statistics for a single solver"""
        total_runs = len(results)
        successful_results = [r for r in results if r['status'] == 'optimal']
        failed_results = [r for r in results if r['status'] != 'optimal']
        
        successful_runs = len(successful_results)
        failed_runs = len(failed_results)
        success_rate = successful_runs / total_runs if total_runs > 0 else 0.0
        
        # Solve time statistics (only for successful runs)
        if successful_results:
            solve_times = [r['solve_time'] for r in successful_results]
            avg_solve_time = calculate_mean(solve_times)
            median_solve_time = calculate_median(solve_times)
            min_solve_time = min(solve_times)
            max_solve_time = max(solve_times)
            std_dev_solve_time = calculate_stdev(solve_times)
            total_solve_time = sum(solve_times)
            
            # Objective value statistics
            objective_values = [r['objective_value'] for r in successful_results 
                              if r['objective_value'] is not None]
            avg_objective_value = calculate_mean(objective_values) if objective_values else None
        else:
            avg_solve_time = median_solve_time = min_solve_time = max_solve_time = 0.0
            std_dev_solve_time = total_solve_time = 0.0
            avg_objective_value = None
        
        # Get unique problems solved
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(DISTINCT problem_name) 
                FROM results 
                WHERE solver_name = ? AND status = 'optimal'
            """, (solver_name,))
            problems_solved = cursor.fetchone()[0]
        
        # Calculate reliability score (success rate weighted by consistency)
        consistency_factor = 1.0 - (std_dev_solve_time / avg_solve_time) if avg_solve_time > 0 else 1.0
        consistency_factor = max(0.0, min(1.0, consistency_factor))  # Clamp between 0 and 1
        reliability_score = success_rate * 0.7 + consistency_factor * 0.3
        
        return SolverStatistics(
            name=solver_name,
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            success_rate=success_rate,
            avg_solve_time=avg_solve_time,
            median_solve_time=median_solve_time,
            min_solve_time=min_solve_time,
            max_solve_time=max_solve_time,
            std_dev_solve_time=std_dev_solve_time,
            problems_solved=problems_solved,
            total_solve_time=total_solve_time,
            avg_objective_value=avg_objective_value,
            reliability_score=reliability_score
        )
    
    def calculate_problem_statistics(self) -> Dict[str, ProblemStatistics]:
        """Calculate comprehensive statistics for each problem"""
        logger.info("Calculating problem statistics...")
        
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Get all results grouped by problem
            cursor.execute("""
                SELECT 
                    r.problem_name,
                    p.problem_class as problem_type,
                    r.solver_name,
                    r.status,
                    r.solve_time
                FROM results r
                LEFT JOIN problems p ON p.name = r.problem_name
                ORDER BY r.problem_name, r.solve_time
            """)
            
            # Group results by problem
            problem_results = {}
            for row in cursor.fetchall():
                problem = row['problem_name']
                if problem not in problem_results:
                    problem_results[problem] = {
                        'type': row['problem_type'] or 'Unknown',
                        'attempts': []
                    }
                problem_results[problem]['attempts'].append({
                    'solver': row['solver_name'],
                    'status': row['status'],
                    'solve_time': row['solve_time']
                })
            
            # Calculate statistics for each problem
            problem_stats = {}
            for problem_name, data in problem_results.items():
                stats = self._calculate_single_problem_stats(problem_name, data)
                problem_stats[problem_name] = stats
            
            logger.info(f"Calculated statistics for {len(problem_stats)} problems")
            return problem_stats
    
    def _calculate_single_problem_stats(self, problem_name: str, data: Dict) -> ProblemStatistics:
        """Calculate statistics for a single problem"""
        attempts = data['attempts']
        problem_type = data['type']
        
        total_attempts = len(attempts)
        successful_attempts = [a for a in attempts if a['status'] == 'optimal']
        successful_count = len(successful_attempts)
        success_rate = successful_count / total_attempts if total_attempts > 0 else 0.0
        
        if successful_attempts:
            solve_times = [a['solve_time'] for a in successful_attempts]
            avg_solve_time = calculate_mean(solve_times)
            min_solve_time = min(solve_times)
            max_solve_time = max(solve_times)
            
            # Find fastest and slowest solvers
            fastest_attempt = min(successful_attempts, key=lambda x: x['solve_time'])
            slowest_attempt = max(successful_attempts, key=lambda x: x['solve_time'])
            fastest_solver = fastest_attempt['solver']
            slowest_solver = slowest_attempt['solver']
        else:
            avg_solve_time = min_solve_time = max_solve_time = 0.0
            fastest_solver = slowest_solver = None
        
        # Calculate difficulty score (inverse of success rate, adjusted by solve time)
        time_factor = min(avg_solve_time / 10.0, 1.0) if avg_solve_time > 0 else 0.0  # Normalize to 0-1
        difficulty_score = (1.0 - success_rate) * 0.8 + time_factor * 0.2
        
        return ProblemStatistics(
            name=problem_name,
            problem_type=problem_type,
            total_attempts=total_attempts,
            successful_attempts=successful_count,
            success_rate=success_rate,
            avg_solve_time=avg_solve_time,
            min_solve_time=min_solve_time,
            max_solve_time=max_solve_time,
            fastest_solver=fastest_solver,
            slowest_solver=slowest_solver,
            difficulty_score=difficulty_score
        )
    
    def calculate_benchmark_summary(self, solver_stats: Dict[str, SolverStatistics], 
                                  problem_stats: Dict[str, ProblemStatistics]) -> BenchmarkSummary:
        """Calculate overall benchmark summary statistics"""
        logger.info("Calculating benchmark summary...")
        
        if not solver_stats:
            return BenchmarkSummary(0, 0, 0, 0.0, 0.0, 0.0, None, None, None, None)
        
        # Overall statistics
        total_runs = sum(stats.total_runs for stats in solver_stats.values())
        total_solvers = len(solver_stats)
        total_problems = len(problem_stats)
        
        successful_runs = sum(stats.successful_runs for stats in solver_stats.values())
        overall_success_rate = successful_runs / total_runs if total_runs > 0 else 0.0
        
        total_solve_time = sum(stats.total_solve_time for stats in solver_stats.values())
        avg_solve_time = total_solve_time / successful_runs if successful_runs > 0 else 0.0
        
        # Find best solvers and problems
        fastest_solver = min(solver_stats.keys(), 
                           key=lambda s: solver_stats[s].avg_solve_time 
                           if solver_stats[s].successful_runs > 0 else float('inf'))
        
        most_reliable_solver = max(solver_stats.keys(), 
                                 key=lambda s: solver_stats[s].reliability_score)
        
        if problem_stats:
            hardest_problem = max(problem_stats.keys(), 
                                key=lambda p: problem_stats[p].difficulty_score)
            easiest_problem = min(problem_stats.keys(), 
                                key=lambda p: problem_stats[p].difficulty_score)
        else:
            hardest_problem = easiest_problem = None
        
        return BenchmarkSummary(
            total_runs=total_runs,
            total_solvers=total_solvers,
            total_problems=total_problems,
            overall_success_rate=overall_success_rate,
            total_solve_time=total_solve_time,
            avg_solve_time=avg_solve_time,
            fastest_solver=fastest_solver,
            most_reliable_solver=most_reliable_solver,
            hardest_problem=hardest_problem,
            easiest_problem=easiest_problem
        )
    
    def get_solver_rankings(self, solver_stats: Dict[str, SolverStatistics]) -> Dict[str, List[Tuple[str, float]]]:
        """Get solver rankings by different metrics"""
        if not solver_stats:
            return {}
        
        # Filter solvers with successful runs
        active_solvers = {name: stats for name, stats in solver_stats.items() 
                         if stats.successful_runs > 0}
        
        if not active_solvers:
            return {}
        
        rankings = {}
        
        # Fastest solvers (by average solve time)
        rankings['fastest'] = sorted(
            [(name, stats.avg_solve_time) for name, stats in active_solvers.items()],
            key=lambda x: x[1]
        )
        
        # Most reliable solvers (by success rate)
        rankings['most_reliable'] = sorted(
            [(name, stats.success_rate) for name, stats in solver_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Most consistent solvers (by reliability score)
        rankings['most_consistent'] = sorted(
            [(name, stats.reliability_score) for name, stats in active_solvers.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Most productive solvers (by problems solved)
        rankings['most_productive'] = sorted(
            [(name, stats.problems_solved) for name, stats in active_solvers.items()],
            key=lambda x: x[1], reverse=True
        )
        
        return rankings
    
    def get_problem_rankings(self, problem_stats: Dict[str, ProblemStatistics]) -> Dict[str, List[Tuple[str, float]]]:
        """Get problem rankings by different metrics"""
        if not problem_stats:
            return {}
        
        rankings = {}
        
        # Hardest problems (by difficulty score)
        rankings['hardest'] = sorted(
            [(name, stats.difficulty_score) for name, stats in problem_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Easiest problems (by success rate)
        rankings['easiest'] = sorted(
            [(name, stats.success_rate) for name, stats in problem_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Fastest to solve (by average solve time, only successful ones)
        fastest_problems = [(name, stats.avg_solve_time) for name, stats in problem_stats.items() 
                          if stats.successful_attempts > 0]
        rankings['fastest_to_solve'] = sorted(fastest_problems, key=lambda x: x[1])
        
        return rankings
    
    def calculate_all_statistics(self) -> Dict[str, Any]:
        """Calculate all statistics and return as a structured dictionary"""
        logger.info("Calculating all statistics...")
        
        solver_stats = self.calculate_solver_statistics()
        problem_stats = self.calculate_problem_statistics()
        summary = self.calculate_benchmark_summary(solver_stats, problem_stats)
        solver_rankings = self.get_solver_rankings(solver_stats)
        problem_rankings = self.get_problem_rankings(problem_stats)
        
        return {
            'solver_statistics': solver_stats,
            'problem_statistics': problem_stats,
            'benchmark_summary': summary,
            'solver_rankings': solver_rankings,
            'problem_rankings': problem_rankings,
            'backend_comparison': self.calculate_backend_comparison(),
            'generated_at': datetime.now().isoformat()
        }
    
    def calculate_backend_comparison(self) -> Dict[str, Any]:
        """Calculate comprehensive backend comparison statistics."""
        logger.info("Calculating backend comparison statistics...")
        
        try:
            from scripts.utils.solver_validation import SolverValidator, ProblemType
            from scripts.benchmark.backend_selector import BackendSelector
        except ImportError:
            logger.warning("Backend validation modules not available, skipping backend comparison")
            return {}
        
        # Initialize backend analysis components
        validator = SolverValidator()
        selector = BackendSelector(validator)
        
        # Get validation results
        validation_results = validator.validate_all_backends()
        available_backends = [k for k, v in validation_results.items() if v.available]
        
        # Calculate backend performance from database
        backend_performance = self._calculate_backend_performance_from_db()
        
        # Generate comprehensive backend analysis
        backend_analysis = {
            "validation_summary": {
                "total_backends": len(validation_results),
                "available_backends": len(available_backends),
                "unavailable_backends": len(validation_results) - len(available_backends),
                "availability_percentage": len(available_backends) / len(validation_results) * 100 if validation_results else 0
            },
            "available_backends": available_backends,
            "backend_capabilities": {},
            "performance_comparison": backend_performance,
            "problem_type_coverage": {},
            "recommendations": {}
        }
        
        # Add capability information for available backends
        for backend_name, result in validation_results.items():
            if result.available and result.capabilities:
                backend_analysis["backend_capabilities"][backend_name] = {
                    "supported_types": [pt.value for pt in result.capabilities.supported_types],
                    "performance_tier": result.capabilities.performance_tier,
                    "memory_efficiency": result.capabilities.memory_efficiency,
                    "stability": result.capabilities.stability,
                    "installation_notes": result.capabilities.installation_notes
                }
        
        # Analyze problem type coverage
        for problem_type in ProblemType:
            compatible_backends = validator.get_backends_for_problem_type(problem_type)
            available_compatible = [b for b in compatible_backends if b in available_backends]
            selection = selector.select_backend_for_problem(problem_type)
            recommended = selection.selected_backend
            
            backend_analysis["problem_type_coverage"][problem_type.value] = {
                "total_compatible": len(compatible_backends),
                "available_compatible": len(available_compatible),
                "coverage_percentage": len(available_compatible) / len(compatible_backends) * 100 if compatible_backends else 0,
                "available_backends": available_compatible,
                "recommended_backend": recommended
            }
        
        # Add recommendations for each problem type
        for problem_type in ProblemType:
            selection = selector.select_backend_for_problem(problem_type)
            recommended = selection.selected_backend
            backend_analysis["recommendations"][problem_type.value] = recommended
        
        return backend_analysis
    
    def _calculate_backend_performance_from_db(self) -> Dict[str, Any]:
        """Calculate backend performance statistics from database results."""
        
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Get results grouped by solver and problem type  
            cursor.execute("""
                SELECT 
                    r.solver_name,
                    p.problem_class,
                    r.status,
                    r.solve_time,
                    r.objective_value,
                    r.iterations
                FROM results r
                JOIN problems p ON r.problem_name = p.name
                WHERE r.solver_name LIKE '% (via CVXPY)'
                ORDER BY r.solver_name, p.problem_class
            """)
            
            # Group results by backend and problem type
            backend_results = {}
            for row in cursor.fetchall():
                solver_name = row['solver_name']
                problem_type = row['problem_class'] or 'Unknown'
                
                if solver_name not in backend_results:
                    backend_results[solver_name] = {}
                if problem_type not in backend_results[solver_name]:
                    backend_results[solver_name][problem_type] = []
                
                backend_results[solver_name][problem_type].append({
                    'status': row['status'],
                    'solve_time': row['solve_time'],
                    'objective_value': row['objective_value'],
                    'iterations': row['iterations']
                })
        
        # Calculate performance metrics for each backend
        performance_analysis = {}
        
        for backend_name, problem_types in backend_results.items():
            backend_metrics = {
                "total_problems_attempted": 0,
                "total_successful": 0,
                "overall_success_rate": 0.0,
                "average_solve_time": 0.0,
                "problem_type_performance": {}
            }
            
            all_times = []
            total_attempts = 0
            total_successful = 0
            
            # Analyze performance by problem type
            for problem_type, results in problem_types.items():
                successful_results = [r for r in results if r['status'] == 'optimal']
                solve_times = [r['solve_time'] for r in successful_results]
                
                type_metrics = {
                    "attempts": len(results),
                    "successful": len(successful_results),
                    "success_rate": len(successful_results) / len(results) * 100 if results else 0,
                    "avg_solve_time": calculate_mean(solve_times),
                    "min_solve_time": min(solve_times) if solve_times else None,
                    "max_solve_time": max(solve_times) if solve_times else None,
                    "std_dev_solve_time": calculate_stdev(solve_times)
                }
                
                backend_metrics["problem_type_performance"][problem_type] = type_metrics
                
                # Accumulate overall statistics
                total_attempts += len(results)
                total_successful += len(successful_results)
                all_times.extend(solve_times)
            
            # Calculate overall metrics
            backend_metrics["total_problems_attempted"] = total_attempts
            backend_metrics["total_successful"] = total_successful
            backend_metrics["overall_success_rate"] = total_successful / total_attempts * 100 if total_attempts else 0
            backend_metrics["average_solve_time"] = calculate_mean(all_times)
            
            performance_analysis[backend_name] = backend_metrics
        
        return performance_analysis


def main():
    """Main function for testing the statistics calculator"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python statistics.py <database_path>")
        sys.exit(1)
    
    database_path = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        calculator = StatisticsCalculator(database_path)
        all_stats = calculator.calculate_all_statistics()
        
        # Print summary
        summary = all_stats['benchmark_summary']
        print("\n=== Benchmark Summary ===")
        print(f"Total Runs: {summary.total_runs}")
        print(f"Total Solvers: {summary.total_solvers}")
        print(f"Total Problems: {summary.total_problems}")
        print(f"Overall Success Rate: {summary.overall_success_rate:.1%}")
        print(f"Average Solve Time: {summary.avg_solve_time:.3f}s")
        print(f"Fastest Solver: {summary.fastest_solver}")
        print(f"Most Reliable Solver: {summary.most_reliable_solver}")
        print(f"Hardest Problem: {summary.hardest_problem}")
        print(f"Easiest Problem: {summary.easiest_problem}")
        
        # Print solver rankings
        rankings = all_stats['solver_rankings']
        if 'fastest' in rankings:
            print(f"\n=== Fastest Solvers ===")
            for i, (solver, time) in enumerate(rankings['fastest'][:3], 1):
                print(f"{i}. {solver}: {time:.3f}s")
        
        print(f"\nStatistics calculated successfully!")
        
    except Exception as e:
        logger.error(f"Failed to calculate statistics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()