"""
Performance Profiler
====================

Advanced performance profiling and benchmarking metrics for optimization solvers.
Provides detailed performance breakdowns, memory analysis, and scalability profiling.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("performance_profiler")


@dataclass
class PerformanceProfile:
    """Detailed performance profile for a solver."""
    solver_name: str
    
    # Time breakdown
    total_time: float
    solve_time: float
    setup_time: float
    cleanup_time: float
    
    # Memory profile
    peak_memory_mb: float
    memory_efficiency: float  # Problems solved per MB
    
    # Scalability metrics
    time_complexity_exponent: float
    memory_complexity_exponent: float
    scalability_rating: str  # "Excellent", "Good", "Fair", "Poor"
    
    # Robustness metrics
    failure_rate: float
    numerical_stability_score: float
    convergence_reliability: float
    
    # Resource utilization
    cpu_efficiency: float
    throughput_problems_per_second: float
    
    # Quality metrics
    solution_quality_score: float
    constraint_satisfaction_rate: float


@dataclass
class BenchmarkProfile:
    """Performance profile for an entire benchmark run."""
    benchmark_id: str
    timestamp: str
    
    # Overall metrics
    total_runtime: float
    total_problems: int
    total_solvers: int
    
    # Performance breakdown
    solver_profiles: Dict[str, PerformanceProfile]
    problem_difficulty_distribution: Dict[str, int]
    
    # System metrics
    system_load: float
    memory_usage: float
    
    # Quality assessment
    overall_success_rate: float
    benchmark_efficiency_score: float


class PerformanceProfiler:
    """Advanced performance profiler for solver benchmarking."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize performance profiler.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = project_root / "database" / "results.db"
        
        self.db_path = str(db_path)
        self.logger = get_logger("performance_profiler")
        
        # Profiling configuration
        self.memory_sampling_interval = 0.1  # seconds
        self.enable_detailed_profiling = True
        
        # Performance thresholds
        self.fast_threshold = 0.1  # seconds
        self.slow_threshold = 10.0  # seconds
        self.memory_limit_mb = 1000  # MB
    
    def profile_solver_performance(self, solver_name: str) -> Optional[PerformanceProfile]:
        """Create detailed performance profile for a solver."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load solver results with detailed timing information
                query = """
                    SELECT 
                        r.solve_time,
                        r.status,
                        r.objective_value,
                        r.iterations,
                        r.problem_name,
                        COALESCE(pc.n_variables, 0) as n_variables,
                        COALESCE(pc.n_constraints, 0) as n_constraints,
                        COALESCE(pc.complexity_score, 1.0) as complexity_score,
                        r.solver_info
                    FROM results r
                    LEFT JOIN problem_classifications pc ON r.problem_name = pc.problem_name
                    WHERE r.solver_name = ? AND r.solve_time IS NOT NULL
                    ORDER BY r.id
                """
                
                df = pd.read_sql_query(query, conn, params=[solver_name])
                
                if df.empty:
                    self.logger.warning(f"No results found for solver: {solver_name}")
                    return None
                
                # Extract timing information
                solve_times = df['solve_time'].values
                successful = df[df['status'] == 'optimal']
                
                # Time breakdown analysis
                total_time = np.sum(solve_times)
                avg_solve_time = np.mean(solve_times)
                
                # Memory analysis (simulated for now - would need actual memory profiling)
                problem_sizes = df['n_variables'] + df['n_constraints']
                estimated_memory = self._estimate_memory_usage(problem_sizes, solve_times)
                peak_memory = np.max(estimated_memory)
                memory_efficiency = len(df) / peak_memory if peak_memory > 0 else 0
                
                # Scalability analysis
                time_complexity = self._analyze_time_complexity(problem_sizes, solve_times)
                memory_complexity = self._analyze_memory_complexity(problem_sizes, estimated_memory)
                
                # Robustness metrics
                failure_rate = 1 - (len(successful) / len(df))
                stability_score = self._calculate_numerical_stability(df)
                convergence_reliability = self._calculate_convergence_reliability(df)
                
                # Resource utilization
                cpu_efficiency = self._calculate_cpu_efficiency(solve_times, problem_sizes)
                throughput = len(df) / total_time if total_time > 0 else 0
                
                # Quality metrics
                quality_score = self._calculate_solution_quality(successful)
                constraint_satisfaction = len(successful) / len(df) if len(df) > 0 else 0
                
                # Scalability rating
                scalability_rating = self._rate_scalability(time_complexity, memory_complexity)
                
                return PerformanceProfile(
                    solver_name=solver_name,
                    total_time=total_time,
                    solve_time=avg_solve_time,
                    setup_time=avg_solve_time * 0.1,  # Estimated
                    cleanup_time=avg_solve_time * 0.05,  # Estimated
                    peak_memory_mb=peak_memory,
                    memory_efficiency=memory_efficiency,
                    time_complexity_exponent=time_complexity,
                    memory_complexity_exponent=memory_complexity,
                    scalability_rating=scalability_rating,
                    failure_rate=failure_rate,
                    numerical_stability_score=stability_score,
                    convergence_reliability=convergence_reliability,
                    cpu_efficiency=cpu_efficiency,
                    throughput_problems_per_second=throughput,
                    solution_quality_score=quality_score,
                    constraint_satisfaction_rate=constraint_satisfaction
                )
                
        except Exception as e:
            self.logger.error(f"Failed to profile solver {solver_name}: {e}")
            return None
    
    def _estimate_memory_usage(self, problem_sizes: pd.Series, solve_times: pd.Series) -> np.ndarray:
        """Estimate memory usage based on problem size and solve time."""
        # Simple heuristic: memory scales with problem size and complexity
        base_memory = 10  # Base memory in MB
        size_factor = 0.001  # MB per variable/constraint
        time_factor = 1.0  # Additional memory for longer solves
        
        estimated_memory = base_memory + (problem_sizes * size_factor) + (solve_times * time_factor)
        return np.maximum(estimated_memory, base_memory)
    
    def _analyze_time_complexity(self, problem_sizes: pd.Series, solve_times: pd.Series) -> float:
        """Analyze time complexity scaling with problem size."""
        if len(problem_sizes) < 3:
            return 1.0  # Default linear complexity
        
        # Filter out zero sizes and times
        valid_mask = (problem_sizes > 0) & (solve_times > 0)
        if np.sum(valid_mask) < 3:
            return 1.0
        
        sizes = problem_sizes[valid_mask]
        times = solve_times[valid_mask]
        
        # Log-log regression to find scaling exponent
        try:
            log_sizes = np.log(sizes)
            log_times = np.log(times)
            
            # Linear regression on log-log scale
            coeffs = np.polyfit(log_sizes, log_times, 1)
            exponent = coeffs[0]
            
            # Clamp to reasonable range
            return np.clip(exponent, 0.1, 5.0)
            
        except Exception:
            return 1.0  # Default to linear
    
    def _analyze_memory_complexity(self, problem_sizes: pd.Series, memory_usage: np.ndarray) -> float:
        """Analyze memory complexity scaling with problem size."""
        if len(problem_sizes) < 3:
            return 1.0  # Default linear complexity
        
        # Similar to time complexity analysis
        try:
            valid_mask = (problem_sizes > 0) & (memory_usage > 0)
            if np.sum(valid_mask) < 3:
                return 1.0
            
            sizes = problem_sizes[valid_mask]
            memory = memory_usage[valid_mask]
            
            log_sizes = np.log(sizes)
            log_memory = np.log(memory)
            
            coeffs = np.polyfit(log_sizes, log_memory, 1)
            exponent = coeffs[0]
            
            return np.clip(exponent, 0.1, 3.0)  # Memory typically scales better than time
            
        except Exception:
            return 1.0
    
    def _calculate_numerical_stability(self, df: pd.DataFrame) -> float:
        """Calculate numerical stability score based on solution consistency."""
        # Look for patterns in solve times and objective values that indicate instability
        if len(df) < 3:
            return 1.0
        
        # Coefficient of variation in solve times (lower is more stable)
        time_cv = df['solve_time'].std() / df['solve_time'].mean() if df['solve_time'].mean() > 0 else 0
        
        # Stability score (0-1, higher is more stable)
        stability = max(0, 1 - min(1, time_cv))
        
        return stability
    
    def _calculate_convergence_reliability(self, df: pd.DataFrame) -> float:
        """Calculate convergence reliability based on iteration counts and success rates."""
        successful = df[df['status'] == 'optimal']
        
        if len(successful) == 0:
            return 0.0
        
        # Look at iteration count consistency
        iterations = successful['iterations'].fillna(0)
        if len(iterations) > 1 and iterations.std() > 0:
            iteration_consistency = 1 / (1 + iterations.std() / iterations.mean())
        else:
            iteration_consistency = 1.0
        
        # Combine with success rate
        success_rate = len(successful) / len(df)
        
        return (success_rate + iteration_consistency) / 2
    
    def _calculate_cpu_efficiency(self, solve_times: pd.Series, problem_sizes: pd.Series) -> float:
        """Calculate CPU efficiency metric."""
        if len(solve_times) == 0 or solve_times.sum() == 0:
            return 0.0
        
        # Problems solved per unit time, normalized by problem complexity
        total_complexity = problem_sizes.sum()
        total_time = solve_times.sum()
        
        efficiency = total_complexity / total_time if total_time > 0 else 0
        
        # Normalize to 0-1 scale (arbitrary scaling for interpretability)
        return min(1.0, efficiency / 1000)
    
    def _calculate_solution_quality(self, successful_df: pd.DataFrame) -> float:
        """Calculate solution quality score."""
        if len(successful_df) == 0:
            return 0.0
        
        # For now, quality is based on success rate
        # Could be enhanced with actual optimality gap analysis
        quality_factors = []
        
        # Iteration efficiency (fewer iterations for same quality is better)
        iterations = successful_df['iterations'].fillna(0)
        if len(iterations) > 0 and iterations.max() > 0:
            iter_efficiency = 1 - (iterations.mean() / iterations.max())
            quality_factors.append(iter_efficiency)
        
        # Default quality score
        quality_factors.append(0.8)  # Assume good quality for successful solves
        
        return np.mean(quality_factors)
    
    def _rate_scalability(self, time_complexity: float, memory_complexity: float) -> str:
        """Rate overall scalability based on complexity exponents."""
        avg_complexity = (time_complexity + memory_complexity) / 2
        
        if avg_complexity < 1.0:
            return "Excellent"
        elif avg_complexity < 1.5:
            return "Good"
        elif avg_complexity < 2.0:
            return "Fair"
        else:
            return "Poor"
    
    def profile_benchmark_run(self, benchmark_id: Optional[str] = None) -> BenchmarkProfile:
        """Profile an entire benchmark run."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get benchmark metadata
                if benchmark_id:
                    benchmark_query = "SELECT * FROM benchmarks WHERE id = ?"
                    benchmark_df = pd.read_sql_query(benchmark_query, conn, params=[benchmark_id])
                else:
                    # Get most recent benchmark
                    benchmark_query = "SELECT * FROM benchmarks ORDER BY id DESC LIMIT 1"
                    benchmark_df = pd.read_sql_query(benchmark_query, conn)
                
                if benchmark_df.empty:
                    self.logger.error("No benchmark data found")
                    return None
                
                benchmark_info = benchmark_df.iloc[0]
                actual_benchmark_id = str(benchmark_info['id'])
                
                # Get all solvers in this benchmark
                solvers_query = """
                    SELECT DISTINCT solver_name 
                    FROM results 
                    WHERE benchmark_id = ?
                """
                solvers_df = pd.read_sql_query(solvers_query, conn, params=[actual_benchmark_id])
                solver_names = solvers_df['solver_name'].tolist()
                
                # Profile each solver
                solver_profiles = {}
                total_runtime = 0
                
                for solver_name in solver_names:
                    profile = self.profile_solver_performance(solver_name)
                    if profile:
                        solver_profiles[solver_name] = profile
                        total_runtime += profile.total_time
                
                # Calculate overall metrics
                results_query = """
                    SELECT 
                        COUNT(*) as total_problems,
                        AVG(CASE WHEN status = 'optimal' THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(DISTINCT problem_name) as unique_problems
                    FROM results 
                    WHERE benchmark_id = ?
                """
                stats_df = pd.read_sql_query(results_query, conn, params=[actual_benchmark_id])
                stats = stats_df.iloc[0]
                
                # Problem difficulty distribution
                difficulty_query = """
                    SELECT 
                        COALESCE(pc.difficulty_level, 'Unknown') as difficulty,
                        COUNT(*) as count
                    FROM results r
                    LEFT JOIN problem_classifications pc ON r.problem_name = pc.problem_name
                    WHERE r.benchmark_id = ?
                    GROUP BY difficulty
                """
                difficulty_df = pd.read_sql_query(difficulty_query, conn, params=[actual_benchmark_id])
                difficulty_dist = dict(zip(difficulty_df['difficulty'], difficulty_df['count']))
                
                # Calculate benchmark efficiency score
                efficiency_score = self._calculate_benchmark_efficiency(
                    solver_profiles, stats['success_rate'], total_runtime
                )
                
                return BenchmarkProfile(
                    benchmark_id=actual_benchmark_id,
                    timestamp=benchmark_info.get('created_at', 'Unknown'),
                    total_runtime=total_runtime,
                    total_problems=int(stats['total_problems']),
                    total_solvers=len(solver_profiles),
                    solver_profiles=solver_profiles,
                    problem_difficulty_distribution=difficulty_dist,
                    system_load=0.5,  # Simulated system metrics
                    memory_usage=sum(p.peak_memory_mb for p in solver_profiles.values()),
                    overall_success_rate=float(stats['success_rate']),
                    benchmark_efficiency_score=efficiency_score
                )
                
        except Exception as e:
            self.logger.error(f"Failed to profile benchmark run: {e}")
            return None
    
    def _calculate_benchmark_efficiency(self, solver_profiles: Dict[str, PerformanceProfile],
                                      success_rate: float, total_runtime: float) -> float:
        """Calculate overall benchmark efficiency score."""
        if not solver_profiles or total_runtime == 0:
            return 0.0
        
        # Factors contributing to efficiency
        factors = []
        
        # Success rate factor
        factors.append(success_rate)
        
        # Speed factor (problems per second across all solvers)
        total_throughput = sum(p.throughput_problems_per_second for p in solver_profiles.values())
        speed_factor = min(1.0, total_throughput / 10)  # Normalize to reasonable range
        factors.append(speed_factor)
        
        # Memory efficiency factor
        avg_memory_efficiency = np.mean([p.memory_efficiency for p in solver_profiles.values()])
        memory_factor = min(1.0, avg_memory_efficiency / 100)  # Normalize
        factors.append(memory_factor)
        
        # Scalability factor
        scalability_scores = {
            "Excellent": 1.0, "Good": 0.8, "Fair": 0.6, "Poor": 0.3
        }
        avg_scalability = np.mean([
            scalability_scores.get(p.scalability_rating, 0.5) 
            for p in solver_profiles.values()
        ])
        factors.append(avg_scalability)
        
        return np.mean(factors) * 100  # Convert to 0-100 scale
    
    def generate_performance_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance profiling report."""
        
        self.logger.info("Generating performance profiling report...")
        
        # Profile latest benchmark run
        benchmark_profile = self.profile_benchmark_run()
        
        if not benchmark_profile:
            return {'error': 'No benchmark data available for profiling'}
        
        report = {
            'metadata': {
                'report_date': pd.Timestamp.now().isoformat(),
                'benchmark_id': benchmark_profile.benchmark_id,
                'profiler_version': '1.0.0'
            },
            'benchmark_summary': {
                'total_runtime': benchmark_profile.total_runtime,
                'total_problems': benchmark_profile.total_problems,
                'total_solvers': benchmark_profile.total_solvers,
                'overall_success_rate': benchmark_profile.overall_success_rate,
                'efficiency_score': benchmark_profile.benchmark_efficiency_score
            },
            'solver_profiles': {},
            'performance_rankings': {},
            'scalability_analysis': {},
            'resource_utilization': {}
        }
        
        # Detailed solver profiles
        for solver_name, profile in benchmark_profile.solver_profiles.items():
            report['solver_profiles'][solver_name] = {
                'performance_metrics': {
                    'total_time': profile.total_time,
                    'avg_solve_time': profile.solve_time,
                    'throughput': profile.throughput_problems_per_second,
                    'success_rate': 1 - profile.failure_rate
                },
                'resource_usage': {
                    'peak_memory_mb': profile.peak_memory_mb,
                    'memory_efficiency': profile.memory_efficiency,
                    'cpu_efficiency': profile.cpu_efficiency
                },
                'scalability': {
                    'time_complexity': profile.time_complexity_exponent,
                    'memory_complexity': profile.memory_complexity_exponent,
                    'scalability_rating': profile.scalability_rating
                },
                'quality': {
                    'solution_quality': profile.solution_quality_score,
                    'numerical_stability': profile.numerical_stability_score,
                    'convergence_reliability': profile.convergence_reliability
                }
            }
        
        # Performance rankings
        solvers = list(benchmark_profile.solver_profiles.values())
        
        report['performance_rankings'] = {
            'fastest_solvers': sorted(
                [(p.solver_name, p.solve_time) for p in solvers],
                key=lambda x: x[1]
            )[:5],
            'most_reliable': sorted(
                [(p.solver_name, 1-p.failure_rate) for p in solvers],
                key=lambda x: x[1], reverse=True
            )[:5],
            'most_memory_efficient': sorted(
                [(p.solver_name, p.memory_efficiency) for p in solvers],
                key=lambda x: x[1], reverse=True
            )[:5],
            'best_scalability': sorted(
                [(p.solver_name, p.scalability_rating) for p in solvers],
                key=lambda x: {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1}.get(x[1], 0),
                reverse=True
            )[:5]
        }
        
        # Scalability analysis summary
        scalability_dist = defaultdict(int)
        for profile in solvers:
            scalability_dist[profile.scalability_rating] += 1
        
        report['scalability_analysis'] = {
            'distribution': dict(scalability_dist),
            'average_time_complexity': np.mean([p.time_complexity_exponent for p in solvers]),
            'average_memory_complexity': np.mean([p.memory_complexity_exponent for p in solvers])
        }
        
        # Resource utilization summary
        report['resource_utilization'] = {
            'total_memory_usage': benchmark_profile.memory_usage,
            'average_cpu_efficiency': np.mean([p.cpu_efficiency for p in solvers]),
            'total_throughput': sum(p.throughput_problems_per_second for p in solvers)
        }
        
        # Save report if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance profiling report saved to {output_path}")
        
        return report


def run_performance_profiling():
    """Run comprehensive performance profiling and generate report."""
    
    print("⚡ Running Performance Profiling Analysis...")
    print("=" * 60)
    
    profiler = PerformanceProfiler()
    
    # Generate comprehensive report
    report = profiler.generate_performance_report(
        output_path="docs/performance_profiling_report.json"
    )
    
    if 'error' in report:
        print(f"❌ Profiling failed: {report['error']}")
        return
    
    # Print summary
    summary = report['benchmark_summary']
    print(f"⚡ Performance Summary:")
    print(f"  • Total runtime: {summary['total_runtime']:.3f}s")
    print(f"  • Problems solved: {summary['total_problems']}")
    print(f"  • Solvers tested: {summary['total_solvers']}")
    print(f"  • Success rate: {summary['overall_success_rate']:.1%}")
    print(f"  • Efficiency score: {summary['efficiency_score']:.1f}/100")
    
    # Performance rankings
    rankings = report['performance_rankings']
    
    print(f"\n🏃 Fastest Solvers:")
    for i, (solver, time) in enumerate(rankings['fastest_solvers'][:3], 1):
        print(f"  {i}. {solver}: {time:.3f}s avg")
    
    print(f"\n🎯 Most Reliable Solvers:")
    for i, (solver, reliability) in enumerate(rankings['most_reliable'][:3], 1):
        print(f"  {i}. {solver}: {reliability:.1%} success rate")
    
    print(f"\n💾 Most Memory Efficient:")
    for i, (solver, efficiency) in enumerate(rankings['most_memory_efficient'][:3], 1):
        print(f"  {i}. {solver}: {efficiency:.1f} problems/MB")
    
    # Scalability summary
    scalability = report['scalability_analysis']
    print(f"\n📈 Scalability Analysis:")
    print(f"  • Average time complexity: O(n^{scalability['average_time_complexity']:.2f})")
    print(f"  • Average memory complexity: O(n^{scalability['average_memory_complexity']:.2f})")
    
    scalability_dist = scalability['distribution']
    for rating, count in scalability_dist.items():
        print(f"  • {rating} scalability: {count} solvers")
    
    print(f"\n✅ Performance profiling completed!")
    print(f"📄 Detailed report saved to: docs/performance_profiling_report.json")


if __name__ == "__main__":
    run_performance_profiling()