"""
Advanced Statistical Analysis and Benchmarking Metrics
======================================================

Comprehensive statistical analysis module for optimization solver benchmarking.
Provides advanced metrics, performance profiling, and solver characterization.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, kruskal
import warnings
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("statistical_analysis")


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance metrics."""
    solver_name: str
    problem_type: str
    n_problems: int
    
    # Basic metrics
    success_rate: float
    mean_solve_time: float
    median_solve_time: float
    std_solve_time: float
    
    # Robustness metrics
    p95_solve_time: float
    p99_solve_time: float
    geometric_mean_time: float
    coefficient_of_variation: float
    
    # Accuracy metrics
    mean_objective_error: Optional[float]
    median_objective_error: Optional[float]
    constraint_violation_rate: float
    
    # Efficiency metrics
    problems_per_second: float
    relative_performance: float  # Relative to best solver
    
    # Statistical significance
    confidence_interval_95: Tuple[float, float]
    sample_variance: float


@dataclass
class SolverCharacterization:
    """Detailed solver performance characterization."""
    solver_name: str
    
    # Problem type specialization
    lp_performance: Optional[PerformanceMetrics]
    qp_performance: Optional[PerformanceMetrics]
    socp_performance: Optional[PerformanceMetrics]
    sdp_performance: Optional[PerformanceMetrics]
    
    # Scaling characteristics
    scaling_coefficient: float  # How performance scales with problem size
    size_sensitivity: str  # "Low", "Medium", "High"
    
    # Robustness profile
    failure_modes: List[str]
    stability_score: float  # 0-1, higher is more stable
    
    # Recommendation score
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for solver benchmarking."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize statistical analyzer.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = project_root / "database" / "results.db"
        
        self.db_path = str(db_path)
        self.logger = get_logger("statistical_analysis")
        
        # Statistical significance threshold
        self.alpha = 0.05
        
        # Performance thresholds
        self.fast_threshold = 1.0  # seconds
        self.slow_threshold = 60.0  # seconds
        
    def load_benchmark_data(self) -> pd.DataFrame:
        """Load benchmark results from database."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        r.solver_name,
                        r.problem_name,
                        COALESCE(pc.problem_type, p.problem_class, 'Unknown') as problem_type,
                        COALESCE(pc.n_variables, 0) as n_variables,
                        COALESCE(pc.n_constraints, 0) as n_constraints,
                        COALESCE(pc.complexity_score, 1.0) as complexity_score,
                        COALESCE(pc.difficulty_level, 'Medium') as difficulty_level,
                        r.solve_time,
                        r.objective_value,
                        r.status,
                        COALESCE(r.iterations, 0) as iterations,
                        0 as memory_mb,
                        0 as constraint_violation,
                        COALESCE(r.duality_gap, 0) as optimality_gap,
                        datetime('now') as timestamp
                    FROM results r
                    LEFT JOIN problems p ON r.problem_name = p.name
                    LEFT JOIN problem_classifications pc ON r.problem_name = pc.problem_name
                    WHERE r.solve_time IS NOT NULL
                    ORDER BY r.id DESC
                """
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    self.logger.warning("No benchmark data found in database")
                    return pd.DataFrame()
                
                # Clean and preprocess data
                df['success'] = df['status'].isin(['optimal', 'feasible'])
                df['log_solve_time'] = np.log1p(df['solve_time'])
                df['problem_size'] = df['n_variables'] + df['n_constraints']
                
                # Handle missing values
                df['constraint_violation'] = df['constraint_violation'].fillna(0)
                df['optimality_gap'] = df['optimality_gap'].fillna(np.inf)
                df['complexity_score'] = df['complexity_score'].fillna(1.0)
                
                self.logger.info(f"Loaded {len(df)} benchmark results")
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to load benchmark data: {e}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, df: pd.DataFrame, solver_name: str, 
                                    problem_type: str = None) -> Optional[PerformanceMetrics]:
        """Calculate comprehensive performance metrics for a solver."""
        
        # Filter data
        solver_data = df[df['solver_name'] == solver_name].copy()
        if problem_type:
            solver_data = solver_data[solver_data['problem_type'] == problem_type]
        
        if solver_data.empty:
            return None
        
        n_problems = len(solver_data)
        successful = solver_data[solver_data['success']]
        
        # Basic metrics
        success_rate = len(successful) / n_problems
        
        if successful.empty:
            # Return minimal metrics for completely failed solvers
            return PerformanceMetrics(
                solver_name=solver_name,
                problem_type=problem_type or "All",
                n_problems=n_problems,
                success_rate=0.0,
                mean_solve_time=np.inf,
                median_solve_time=np.inf,
                std_solve_time=np.inf,
                p95_solve_time=np.inf,
                p99_solve_time=np.inf,
                geometric_mean_time=np.inf,
                coefficient_of_variation=np.inf,
                mean_objective_error=None,
                median_objective_error=None,
                constraint_violation_rate=1.0,
                problems_per_second=0.0,
                relative_performance=np.inf,
                confidence_interval_95=(np.inf, np.inf),
                sample_variance=np.inf
            )
        
        solve_times = successful['solve_time'].values
        
        # Time statistics
        mean_time = np.mean(solve_times)
        median_time = np.median(solve_times)
        std_time = np.std(solve_times)
        
        # Robustness metrics
        p95_time = np.percentile(solve_times, 95)
        p99_time = np.percentile(solve_times, 99)
        
        # Geometric mean (more robust to outliers)
        geometric_mean_time = stats.gmean(solve_times + 1e-10)  # Avoid log(0)
        
        # Coefficient of variation
        cv = std_time / mean_time if mean_time > 0 else np.inf
        
        # Accuracy metrics
        constraint_violations = solver_data['constraint_violation'].values
        violation_rate = np.mean(constraint_violations > 1e-6)
        
        # Objective error (if reference values available)
        mean_obj_error = None
        median_obj_error = None
        if 'reference_objective' in solver_data.columns:
            obj_errors = np.abs(successful['objective_value'] - successful['reference_objective'])
            mean_obj_error = np.mean(obj_errors)
            median_obj_error = np.median(obj_errors)
        
        # Efficiency metrics
        total_time = np.sum(solve_times)
        problems_per_second = len(solve_times) / total_time if total_time > 0 else 0
        
        # Confidence interval for mean solve time
        confidence_interval = stats.t.interval(
            1 - self.alpha, len(solve_times) - 1,
            loc=mean_time, scale=stats.sem(solve_times)
        ) if len(solve_times) > 1 else (mean_time, mean_time)
        
        return PerformanceMetrics(
            solver_name=solver_name,
            problem_type=problem_type or "All",
            n_problems=n_problems,
            success_rate=success_rate,
            mean_solve_time=mean_time,
            median_solve_time=median_time,
            std_solve_time=std_time,
            p95_solve_time=p95_time,
            p99_solve_time=p99_time,
            geometric_mean_time=geometric_mean_time,
            coefficient_of_variation=cv,
            mean_objective_error=mean_obj_error,
            median_objective_error=median_obj_error,
            constraint_violation_rate=violation_rate,
            problems_per_second=problems_per_second,
            relative_performance=1.0,  # Will be calculated later
            confidence_interval_95=confidence_interval,
            sample_variance=np.var(solve_times)
        )
    
    def perform_pairwise_comparison(self, df: pd.DataFrame, solver1: str, solver2: str,
                                  problem_type: str = None) -> Dict[str, Any]:
        """Perform statistical comparison between two solvers."""
        
        # Filter data for both solvers
        data1 = df[df['solver_name'] == solver1].copy()
        data2 = df[df['solver_name'] == solver2].copy()
        
        if problem_type:
            data1 = data1[data1['problem_type'] == problem_type]
            data2 = data2[data2['problem_type'] == problem_type]
        
        # Find common problems
        common_problems = set(data1['problem_name']) & set(data2['problem_name'])
        
        if not common_problems:
            return {
                'comparison_valid': False,
                'reason': 'No common problems solved by both solvers'
            }
        
        # Get solve times for common problems
        times1 = []
        times2 = []
        
        for problem in common_problems:
            t1 = data1[data1['problem_name'] == problem]['solve_time'].iloc[0]
            t2 = data2[data2['problem_name'] == problem]['solve_time'].iloc[0]
            
            # Only include if both solved successfully
            s1 = data1[data1['problem_name'] == problem]['success'].iloc[0]
            s2 = data2[data2['problem_name'] == problem]['success'].iloc[0]
            
            if s1 and s2:
                times1.append(t1)
                times2.append(t2)
        
        if len(times1) < 3:
            return {
                'comparison_valid': False,
                'reason': 'Insufficient common successful solves for statistical comparison'
            }
        
        times1 = np.array(times1)
        times2 = np.array(times2)
        
        # Perform statistical tests
        results = {
            'comparison_valid': True,
            'n_common_problems': len(times1),
            'solver1': solver1,
            'solver2': solver2
        }
        
        # Paired t-test (parametric)
        try:
            t_stat, t_pvalue = stats.ttest_rel(times1, times2)
            results['paired_ttest'] = {
                'statistic': t_stat,
                'pvalue': t_pvalue,
                'significant': t_pvalue < self.alpha
            }
        except Exception as e:
            self.logger.warning(f"Paired t-test failed: {e}")
            results['paired_ttest'] = None
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pvalue = wilcoxon(times1, times2, alternative='two-sided')
            results['wilcoxon_test'] = {
                'statistic': w_stat,
                'pvalue': w_pvalue,
                'significant': w_pvalue < self.alpha
            }
        except Exception as e:
            self.logger.warning(f"Wilcoxon test failed: {e}")
            results['wilcoxon_test'] = None
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(times1) + np.var(times2)) / 2)
        cohens_d = (np.mean(times1) - np.mean(times2)) / pooled_std if pooled_std > 0 else 0
        
        results['effect_size'] = {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_effect_size(abs(cohens_d))
        }
        
        # Performance ratios
        results['performance_ratio'] = {
            'mean_ratio': np.mean(times1) / np.mean(times2),
            'median_ratio': np.median(times1) / np.median(times2),
            'geometric_mean_ratio': stats.gmean(times1 + 1e-10) / stats.gmean(times2 + 1e-10)
        }
        
        # Win-loss-tie analysis
        wins1 = np.sum(times1 < times2)
        wins2 = np.sum(times2 < times1)
        ties = np.sum(np.abs(times1 - times2) < 1e-6)
        
        results['win_loss_tie'] = {
            f'{solver1}_wins': wins1,
            f'{solver2}_wins': wins2,
            'ties': ties,
            f'{solver1}_win_rate': wins1 / len(times1)
        }
        
        return results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def analyze_solver_scaling(self, df: pd.DataFrame, solver_name: str) -> Dict[str, Any]:
        """Analyze how solver performance scales with problem size."""
        
        solver_data = df[df['solver_name'] == solver_name].copy()
        successful = solver_data[solver_data['success']]
        
        if len(successful) < 5:
            return {'scaling_analysis_valid': False, 'reason': 'Insufficient data'}
        
        # Problem size metrics
        sizes = successful['problem_size'].values
        times = successful['solve_time'].values
        log_times = np.log1p(times)
        log_sizes = np.log1p(sizes)
        
        results = {'scaling_analysis_valid': True}
        
        # Linear correlation with problem size
        size_corr, size_p = stats.pearsonr(sizes, times)
        results['size_correlation'] = {
            'correlation': size_corr,
            'pvalue': size_p,
            'significant': size_p < self.alpha
        }
        
        # Log-log regression (power law)
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_times)
            results['power_law_fit'] = {
                'exponent': slope,
                'r_squared': r_value**2,
                'pvalue': p_value,
                'scaling_interpretation': self._interpret_scaling(slope)
            }
        except Exception as e:
            self.logger.warning(f"Power law regression failed: {e}")
            results['power_law_fit'] = None
        
        # Complexity analysis by problem type
        complexity_analysis = {}
        for ptype in successful['problem_type'].unique():
            if pd.isna(ptype):
                continue
                
            type_data = successful[successful['problem_type'] == ptype]
            if len(type_data) >= 3:
                complexity_scores = type_data['complexity_score'].values
                type_times = type_data['solve_time'].values
                
                if len(np.unique(complexity_scores)) > 1:
                    complexity_corr, complexity_p = stats.pearsonr(complexity_scores, type_times)
                    complexity_analysis[ptype] = {
                        'correlation': complexity_corr,
                        'pvalue': complexity_p,
                        'n_problems': len(type_data)
                    }
        
        results['complexity_analysis'] = complexity_analysis
        
        return results
    
    def _interpret_scaling(self, exponent: float) -> str:
        """Interpret power law scaling exponent."""
        if exponent < 0.5:
            return "sub-linear (excellent scaling)"
        elif exponent < 1.0:
            return "sub-linear (good scaling)"
        elif exponent < 1.5:
            return "linear (moderate scaling)"
        elif exponent < 2.0:
            return "super-linear (poor scaling)"
        else:
            return "quadratic or worse (very poor scaling)"
    
    def characterize_solver(self, df: pd.DataFrame, solver_name: str) -> SolverCharacterization:
        """Create comprehensive solver characterization."""
        
        solver_data = df[df['solver_name'] == solver_name].copy()
        
        if solver_data.empty:
            return SolverCharacterization(
                solver_name=solver_name,
                lp_performance=None,
                qp_performance=None,
                socp_performance=None,
                sdp_performance=None,
                scaling_coefficient=np.inf,
                size_sensitivity="Unknown",
                failure_modes=["No data available"],
                stability_score=0.0,
                overall_score=0.0,
                strengths=[],
                weaknesses=["No benchmark data"]
            )
        
        # Performance by problem type
        problem_types = ['LP', 'QP', 'SOCP', 'SDP']
        type_performances = {}
        
        for ptype in problem_types:
            metrics = self.calculate_performance_metrics(df, solver_name, ptype)
            type_performances[ptype.lower() + '_performance'] = metrics
        
        # Scaling analysis
        scaling_results = self.analyze_solver_scaling(df, solver_name)
        scaling_coeff = scaling_results.get('power_law_fit', {}).get('exponent', np.inf)
        
        # Size sensitivity
        if scaling_coeff < 1.0:
            size_sensitivity = "Low"
        elif scaling_coeff < 1.5:
            size_sensitivity = "Medium"
        else:
            size_sensitivity = "High"
        
        # Failure mode analysis
        failures = solver_data[~solver_data['success']]
        failure_modes = []
        
        if len(failures) > 0:
            # Analyze failure patterns
            status_counts = failures['status'].value_counts()
            for status, count in status_counts.items():
                if count > 0:
                    failure_modes.append(f"{status} ({count} cases)")
        
        # Stability score (based on coefficient of variation)
        overall_metrics = self.calculate_performance_metrics(df, solver_name)
        if overall_metrics and overall_metrics.coefficient_of_variation < np.inf:
            # Lower CV means higher stability
            stability_score = max(0, 1 - min(1, overall_metrics.coefficient_of_variation / 2))
        else:
            stability_score = 0.0
        
        # Overall score calculation
        overall_score = self._calculate_overall_score(overall_metrics, scaling_coeff, stability_score)
        
        # Strengths and weaknesses analysis
        strengths, weaknesses = self._analyze_strengths_weaknesses(
            overall_metrics, type_performances, scaling_coeff, stability_score
        )
        
        return SolverCharacterization(
            solver_name=solver_name,
            lp_performance=type_performances.get('lp_performance'),
            qp_performance=type_performances.get('qp_performance'),
            socp_performance=type_performances.get('socp_performance'),
            sdp_performance=type_performances.get('sdp_performance'),
            scaling_coefficient=scaling_coeff,
            size_sensitivity=size_sensitivity,
            failure_modes=failure_modes if failure_modes else ["None observed"],
            stability_score=stability_score,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _calculate_overall_score(self, metrics: Optional[PerformanceMetrics], 
                               scaling_coeff: float, stability_score: float) -> float:
        """Calculate overall solver score (0-100)."""
        
        if not metrics:
            return 0.0
        
        # Success rate component (40% weight)
        success_component = metrics.success_rate * 40
        
        # Speed component (30% weight) - based on relative performance
        if metrics.relative_performance < np.inf:
            speed_component = max(0, 30 * (2 - metrics.relative_performance))
        else:
            speed_component = 0
        
        # Scaling component (20% weight)
        if scaling_coeff < np.inf:
            scaling_component = max(0, 20 * (2 - scaling_coeff))
        else:
            scaling_component = 0
        
        # Stability component (10% weight)
        stability_component = stability_score * 10
        
        return min(100, success_component + speed_component + scaling_component + stability_component)
    
    def _analyze_strengths_weaknesses(self, metrics: Optional[PerformanceMetrics],
                                    type_performances: Dict, scaling_coeff: float,
                                    stability_score: float) -> Tuple[List[str], List[str]]:
        """Analyze solver strengths and weaknesses."""
        
        strengths = []
        weaknesses = []
        
        if not metrics:
            return [], ["No performance data available"]
        
        # Success rate analysis
        if metrics.success_rate >= 0.95:
            strengths.append("Excellent reliability")
        elif metrics.success_rate < 0.8:
            weaknesses.append("Low success rate")
        
        # Speed analysis
        if metrics.geometric_mean_time < self.fast_threshold:
            strengths.append("Very fast solver")
        elif metrics.geometric_mean_time > self.slow_threshold:
            weaknesses.append("Slow performance")
        
        # Stability analysis
        if stability_score > 0.8:
            strengths.append("Highly stable performance")
        elif stability_score < 0.5:
            weaknesses.append("Inconsistent performance")
        
        # Scaling analysis
        if scaling_coeff < 1.0:
            strengths.append("Excellent scaling with problem size")
        elif scaling_coeff > 2.0:
            weaknesses.append("Poor scaling with problem size")
        
        # Problem type specialization
        best_type = None
        best_performance = np.inf
        
        for ptype, perf in type_performances.items():
            if perf and perf.geometric_mean_time < best_performance:
                best_performance = perf.geometric_mean_time
                best_type = ptype.replace('_performance', '').upper()
        
        if best_type:
            strengths.append(f"Specialized for {best_type} problems")
        
        return strengths, weaknesses
    
    def generate_comprehensive_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis report."""
        
        df = self.load_benchmark_data()
        if df.empty:
            return {'error': 'No benchmark data available'}
        
        report = {
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_results': len(df),
                'unique_solvers': df['solver_name'].nunique(),
                'unique_problems': df['problem_name'].nunique(),
                'problem_types': df['problem_type'].value_counts().to_dict()
            }
        }
        
        # Overall performance metrics
        solver_metrics = {}
        solver_names = df['solver_name'].unique()
        
        for solver in solver_names:
            metrics = self.calculate_performance_metrics(df, solver)
            if metrics:
                solver_metrics[solver] = metrics
        
        # Calculate relative performance
        if solver_metrics:
            best_time = min(m.geometric_mean_time for m in solver_metrics.values() 
                          if m.geometric_mean_time < np.inf)
            
            for metrics in solver_metrics.values():
                if metrics.geometric_mean_time < np.inf:
                    metrics.relative_performance = metrics.geometric_mean_time / best_time
        
        report['solver_metrics'] = {
            name: {
                'success_rate': metrics.success_rate,
                'geometric_mean_time': metrics.geometric_mean_time,
                'relative_performance': metrics.relative_performance,
                'coefficient_of_variation': metrics.coefficient_of_variation,
                'problems_per_second': metrics.problems_per_second
            }
            for name, metrics in solver_metrics.items()
        }
        
        # Pairwise comparisons
        comparisons = {}
        for i, solver1 in enumerate(solver_names):
            for solver2 in solver_names[i+1:]:
                comparison = self.perform_pairwise_comparison(df, solver1, solver2)
                if comparison.get('comparison_valid'):
                    key = f"{solver1}_vs_{solver2}"
                    comparisons[key] = {
                        'n_common_problems': comparison['n_common_problems'],
                        'wilcoxon_significant': comparison.get('wilcoxon_test', {}).get('significant', False),
                        'effect_size': comparison.get('effect_size', {}).get('cohens_d', 0),
                        'performance_ratio': comparison.get('performance_ratio', {}).get('geometric_mean_ratio', 1),
                        'win_rate': comparison.get('win_loss_tie', {}).get(f'{solver1}_win_rate', 0.5)
                    }
        
        report['pairwise_comparisons'] = comparisons
        
        # Solver characterizations
        characterizations = {}
        for solver in solver_names:
            char = self.characterize_solver(df, solver)
            characterizations[solver] = {
                'overall_score': char.overall_score,
                'scaling_coefficient': char.scaling_coefficient,
                'size_sensitivity': char.size_sensitivity,
                'stability_score': char.stability_score,
                'strengths': char.strengths,
                'weaknesses': char.weaknesses,
                'failure_modes': char.failure_modes
            }
        
        report['solver_characterizations'] = characterizations
        
        # Save report if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Comprehensive analysis report saved to {output_path}")
        
        return report


def run_statistical_analysis():
    """Run comprehensive statistical analysis and generate report."""
    
    print("🔬 Running Advanced Statistical Analysis...")
    print("=" * 60)
    
    analyzer = AdvancedStatisticalAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(
        output_path="docs/statistical_analysis_report.json"
    )
    
    if 'error' in report:
        print(f"❌ Analysis failed: {report['error']}")
        return
    
    # Print summary
    metadata = report['metadata']
    print(f"📊 Analysis Summary:")
    print(f"  • Total benchmark results: {metadata['total_results']}")
    print(f"  • Unique solvers: {metadata['unique_solvers']}")
    print(f"  • Unique problems: {metadata['unique_problems']}")
    
    print(f"\n📈 Problem Type Distribution:")
    for ptype, count in metadata['problem_types'].items():
        print(f"  • {ptype}: {count} results")
    
    # Top performing solvers
    solver_metrics = report['solver_metrics']
    if solver_metrics:
        print(f"\n🏆 Top Performing Solvers (by geometric mean time):")
        sorted_solvers = sorted(
            solver_metrics.items(),
            key=lambda x: x[1]['geometric_mean_time']
        )
        
        for i, (solver, metrics) in enumerate(sorted_solvers[:5], 1):
            print(f"  {i}. {solver}")
            print(f"     Success Rate: {metrics['success_rate']:.1%}")
            print(f"     Geometric Mean Time: {metrics['geometric_mean_time']:.3f}s")
            print(f"     Relative Performance: {metrics['relative_performance']:.2f}x")
    
    # Statistical significance summary
    comparisons = report['pairwise_comparisons']
    significant_pairs = sum(1 for comp in comparisons.values() if comp['wilcoxon_significant'])
    total_pairs = len(comparisons)
    
    print(f"\n📊 Statistical Comparisons:")
    print(f"  • Total pairwise comparisons: {total_pairs}")
    print(f"  • Statistically significant differences: {significant_pairs}")
    print(f"  • Significance rate: {significant_pairs/total_pairs:.1%}" if total_pairs > 0 else "")
    
    # Solver characterization summary
    characterizations = report['solver_characterizations']
    if characterizations:
        print(f"\n🎯 Solver Characterization Summary:")
        for solver, char in characterizations.items():
            print(f"  • {solver}:")
            print(f"    Overall Score: {char['overall_score']:.1f}/100")
            print(f"    Size Sensitivity: {char['size_sensitivity']}")
            print(f"    Stability Score: {char['stability_score']:.2f}")
            if char['strengths']:
                print(f"    Strengths: {', '.join(char['strengths'][:2])}")
    
    print(f"\n✅ Comprehensive statistical analysis completed!")
    print(f"📄 Detailed report saved to: docs/statistical_analysis_report.json")


if __name__ == "__main__":
    run_statistical_analysis()