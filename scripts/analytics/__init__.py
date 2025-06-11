"""
Analytics Module
================

Advanced analytics and statistical analysis for optimization solver benchmarking.
Provides comprehensive performance analysis, statistical testing, and visualization.
"""

from .statistical_analysis import (
    AdvancedStatisticalAnalyzer,
    PerformanceMetrics,
    SolverCharacterization,
    run_statistical_analysis
)

from .performance_profiler import (
    PerformanceProfiler,
    PerformanceProfile,
    BenchmarkProfile,
    run_performance_profiling
)

from .visualization import (
    AnalyticsVisualizer,
    create_analytics_dashboard
)

__all__ = [
    'AdvancedStatisticalAnalyzer',
    'PerformanceMetrics', 
    'SolverCharacterization',
    'run_statistical_analysis',
    'PerformanceProfiler',
    'PerformanceProfile',
    'BenchmarkProfile',
    'run_performance_profiling',
    'AnalyticsVisualizer',
    'create_analytics_dashboard'
]