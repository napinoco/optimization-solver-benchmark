"""
Analytics Visualization Module
==============================

Creates comprehensive visualizations for statistical analysis and performance profiling.
Generates charts, plots, and reports for benchmark results.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("visualization")


class AnalyticsVisualizer:
    """Creates visualizations for advanced analytics and performance profiling."""
    
    def __init__(self):
        """Initialize analytics visualizer."""
        self.logger = get_logger("visualization")
        
        # Color schemes for different solver types
        self.color_schemes = {
            'cvxpy': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'commercial': ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'academic': ['#393b79', '#637939', '#8c6d31', '#843c39', '#ad494a']
        }
        
        # Chart templates
        self.chart_templates = {
            'performance_comparison': self._create_performance_chart_template(),
            'scalability_analysis': self._create_scalability_chart_template(),
            'statistical_summary': self._create_statistical_summary_template()
        }
    
    def _create_performance_chart_template(self) -> str:
        """Create HTML template for performance comparison charts."""
        return """
        <div class="chart-container">
            <h3>{title}</h3>
            <div class="chart-content">
                <canvas id="{chart_id}" width="800" height="400"></canvas>
            </div>
            <div class="chart-description">
                <p>{description}</p>
            </div>
        </div>
        """
    
    def _create_scalability_chart_template(self) -> str:
        """Create HTML template for scalability analysis charts."""
        return """
        <div class="scalability-chart">
            <h3>Scalability Analysis: {solver_name}</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">Time Complexity:</span>
                    <span class="metric-value">O(n^{time_complexity:.2f})</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Complexity:</span>
                    <span class="metric-value">O(n^{memory_complexity:.2f})</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Scalability Rating:</span>
                    <span class="metric-value {rating_class}">{scalability_rating}</span>
                </div>
            </div>
            <div class="scalability-plot">
                <canvas id="scalability_{solver_id}" width="600" height="300"></canvas>
            </div>
        </div>
        """
    
    def _create_statistical_summary_template(self) -> str:
        """Create HTML template for statistical summary."""
        return """
        <div class="statistical-summary">
            <h2>Statistical Analysis Summary</h2>
            
            <div class="summary-stats">
                <div class="stat-box">
                    <h4>Total Results</h4>
                    <span class="stat-value">{total_results}</span>
                </div>
                <div class="stat-box">
                    <h4>Unique Solvers</h4>
                    <span class="stat-value">{unique_solvers}</span>
                </div>
                <div class="stat-box">
                    <h4>Unique Problems</h4>
                    <span class="stat-value">{unique_problems}</span>
                </div>
                <div class="stat-box">
                    <h4>Significance Rate</h4>
                    <span class="stat-value">{significance_rate:.1%}</span>
                </div>
            </div>
            
            <div class="problem-type-distribution">
                <h3>Problem Type Distribution</h3>
                <div class="distribution-chart">
                    {problem_type_chart}
                </div>
            </div>
        </div>
        """
    
    def create_performance_dashboard(self, 
                                   statistical_report: Dict[str, Any],
                                   profiling_report: Dict[str, Any],
                                   output_path: str = None) -> str:
        """Create comprehensive performance dashboard."""
        
        self.logger.info("Creating performance dashboard...")
        
        # Generate dashboard HTML
        dashboard_html = self._generate_dashboard_html(statistical_report, profiling_report)
        
        # Save dashboard if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(dashboard_html)
            
            self.logger.info(f"Performance dashboard saved to {output_path}")
        
        return dashboard_html
    
    def _generate_dashboard_html(self, 
                               statistical_report: Dict[str, Any],
                               profiling_report: Dict[str, Any]) -> str:
        """Generate complete dashboard HTML."""
        
        # Extract data from reports
        stats_metadata = statistical_report.get('metadata', {})
        solver_metrics = statistical_report.get('solver_metrics', {})
        comparisons = statistical_report.get('pairwise_comparisons', {})
        
        profiling_summary = profiling_report.get('benchmark_summary', {})
        solver_profiles = profiling_report.get('solver_profiles', {})
        rankings = profiling_report.get('performance_rankings', {})
        
        # Generate individual sections
        header_section = self._generate_header_section(stats_metadata, profiling_summary)
        overview_section = self._generate_overview_section(solver_metrics, solver_profiles)
        performance_section = self._generate_performance_section(rankings, solver_metrics)
        statistical_section = self._generate_statistical_section(comparisons, solver_metrics)
        scalability_section = self._generate_scalability_section(solver_profiles)
        
        # Combine into complete dashboard
        dashboard_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Optimization Solver Performance Dashboard</title>
            <style>
                {self._get_dashboard_css()}
            </style>
        </head>
        <body>
            {header_section}
            {overview_section}
            {performance_section}
            {statistical_section}
            {scalability_section}
            <script>
                {self._get_dashboard_js()}
            </script>
        </body>
        </html>
        """
        
        return dashboard_html
    
    def _generate_header_section(self, stats_metadata: Dict, profiling_summary: Dict) -> str:
        """Generate dashboard header section."""
        
        total_results = stats_metadata.get('total_results', 0)
        unique_solvers = stats_metadata.get('unique_solvers', 0)
        success_rate = profiling_summary.get('overall_success_rate', 0)
        efficiency_score = profiling_summary.get('efficiency_score', 0)
        
        return f"""
        <header class="dashboard-header">
            <div class="container">
                <h1>🔬 Optimization Solver Performance Dashboard</h1>
                <div class="header-stats">
                    <div class="header-stat">
                        <span class="stat-label">Total Results</span>
                        <span class="stat-value">{total_results}</span>
                    </div>
                    <div class="header-stat">
                        <span class="stat-label">Solvers Tested</span>
                        <span class="stat-value">{unique_solvers}</span>
                    </div>
                    <div class="header-stat">
                        <span class="stat-label">Success Rate</span>
                        <span class="stat-value">{success_rate:.1%}</span>
                    </div>
                    <div class="header-stat">
                        <span class="stat-label">Efficiency Score</span>
                        <span class="stat-value">{efficiency_score:.1f}/100</span>
                    </div>
                </div>
            </div>
        </header>
        """
    
    def _generate_overview_section(self, solver_metrics: Dict, solver_profiles: Dict) -> str:
        """Generate overview section with key metrics."""
        
        # Top performers
        if solver_metrics:
            sorted_solvers = sorted(
                solver_metrics.items(),
                key=lambda x: x[1].get('geometric_mean_time', float('inf'))
            )
            top_performers = sorted_solvers[:5]
        else:
            top_performers = []
        
        performers_html = ""
        for i, (solver, metrics) in enumerate(top_performers, 1):
            success_rate = metrics.get('success_rate', 0)
            avg_time = metrics.get('geometric_mean_time', 0)
            relative_perf = metrics.get('relative_performance', 1)
            
            performers_html += f"""
            <div class="performer-card">
                <div class="performer-rank">#{i}</div>
                <div class="performer-info">
                    <h4>{solver}</h4>
                    <div class="performer-metrics">
                        <span>Success: {success_rate:.1%}</span>
                        <span>Time: {avg_time:.3f}s</span>
                        <span>Relative: {relative_perf:.2f}x</span>
                    </div>
                </div>
            </div>
            """
        
        return f"""
        <section class="overview-section">
            <div class="container">
                <h2>📊 Performance Overview</h2>
                <div class="top-performers">
                    <h3>🏆 Top Performing Solvers</h3>
                    <div class="performers-grid">
                        {performers_html}
                    </div>
                </div>
            </div>
        </section>
        """
    
    def _generate_performance_section(self, rankings: Dict, solver_metrics: Dict) -> str:
        """Generate performance comparison section."""
        
        # Fastest solvers table
        fastest_solvers = rankings.get('fastest_solvers', [])
        fastest_html = ""
        for i, (solver, time) in enumerate(fastest_solvers[:5], 1):
            fastest_html += f"""
            <tr>
                <td>{i}</td>
                <td>{solver}</td>
                <td>{time:.3f}s</td>
            </tr>
            """
        
        # Most reliable solvers table
        reliable_solvers = rankings.get('most_reliable', [])
        reliable_html = ""
        for i, (solver, reliability) in enumerate(reliable_solvers[:5], 1):
            reliable_html += f"""
            <tr>
                <td>{i}</td>
                <td>{solver}</td>
                <td>{reliability:.1%}</td>
            </tr>
            """
        
        return f"""
        <section class="performance-section">
            <div class="container">
                <h2>⚡ Performance Analysis</h2>
                <div class="performance-tables">
                    <div class="performance-table">
                        <h3>🏃 Fastest Solvers</h3>
                        <table>
                            <thead>
                                <tr><th>Rank</th><th>Solver</th><th>Avg Time</th></tr>
                            </thead>
                            <tbody>
                                {fastest_html}
                            </tbody>
                        </table>
                    </div>
                    <div class="performance-table">
                        <h3>🎯 Most Reliable Solvers</h3>
                        <table>
                            <thead>
                                <tr><th>Rank</th><th>Solver</th><th>Success Rate</th></tr>
                            </thead>
                            <tbody>
                                {reliable_html}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>
        """
    
    def _generate_statistical_section(self, comparisons: Dict, solver_metrics: Dict) -> str:
        """Generate statistical analysis section."""
        
        # Statistical significance summary
        significant_comparisons = sum(1 for comp in comparisons.values() 
                                    if comp.get('wilcoxon_significant', False))
        total_comparisons = len(comparisons)
        significance_rate = significant_comparisons / total_comparisons if total_comparisons > 0 else 0
        
        # Effect sizes summary
        large_effects = sum(1 for comp in comparisons.values() 
                          if abs(comp.get('effect_size', 0)) > 0.8)
        medium_effects = sum(1 for comp in comparisons.values() 
                           if 0.5 <= abs(comp.get('effect_size', 0)) <= 0.8)
        
        return f"""
        <section class="statistical-section">
            <div class="container">
                <h2>📈 Statistical Analysis</h2>
                <div class="statistical-summary">
                    <div class="stat-card">
                        <h4>Statistical Significance</h4>
                        <div class="stat-value">{significance_rate:.1%}</div>
                        <div class="stat-detail">{significant_comparisons}/{total_comparisons} comparisons</div>
                    </div>
                    <div class="stat-card">
                        <h4>Large Effect Sizes</h4>
                        <div class="stat-value">{large_effects}</div>
                        <div class="stat-detail">Cohen's d > 0.8</div>
                    </div>
                    <div class="stat-card">
                        <h4>Medium Effect Sizes</h4>
                        <div class="stat-value">{medium_effects}</div>
                        <div class="stat-detail">0.5 ≤ Cohen's d ≤ 0.8</div>
                    </div>
                </div>
            </div>
        </section>
        """
    
    def _generate_scalability_section(self, solver_profiles: Dict) -> str:
        """Generate scalability analysis section."""
        
        scalability_cards = ""
        for solver_name, profile in solver_profiles.items():
            scalability = profile.get('scalability', {})
            time_complexity = scalability.get('time_complexity', 1.0)
            memory_complexity = scalability.get('memory_complexity', 1.0)
            rating = scalability.get('scalability_rating', 'Unknown')
            
            rating_class = rating.lower().replace(' ', '-')
            
            scalability_cards += f"""
            <div class="scalability-card">
                <h4>{solver_name}</h4>
                <div class="complexity-metrics">
                    <div class="complexity-metric">
                        <span class="metric-label">Time:</span>
                        <span class="metric-value">O(n^{time_complexity:.2f})</span>
                    </div>
                    <div class="complexity-metric">
                        <span class="metric-label">Memory:</span>
                        <span class="metric-value">O(n^{memory_complexity:.2f})</span>
                    </div>
                </div>
                <div class="scalability-rating {rating_class}">
                    {rating}
                </div>
            </div>
            """
        
        return f"""
        <section class="scalability-section">
            <div class="container">
                <h2>📏 Scalability Analysis</h2>
                <div class="scalability-grid">
                    {scalability_cards}
                </div>
            </div>
        </section>
        """
    
    def _get_dashboard_css(self) -> str:
        """Get CSS styles for the dashboard."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        
        .dashboard-header h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .header-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }
        
        .header-stat {
            text-align: center;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        
        .stat-label {
            display: block;
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .stat-value {
            display: block;
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 0.5rem;
        }
        
        section {
            background: white;
            margin-bottom: 2rem;
            padding: 2rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h2 {
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
            color: #2c3e50;
        }
        
        h3 {
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: #34495e;
        }
        
        .performers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }
        
        .performer-card {
            display: flex;
            align-items: center;
            padding: 1rem;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .performer-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .performer-rank {
            font-size: 1.5rem;
            font-weight: bold;
            color: #007bff;
            margin-right: 1rem;
            min-width: 40px;
        }
        
        .performer-info h4 {
            margin-bottom: 0.5rem;
        }
        
        .performer-metrics {
            display: flex;
            gap: 1rem;
            font-size: 0.9rem;
            color: #666;
        }
        
        .performance-tables {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        th, td {
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid #dee2e6;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        .statistical-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        .stat-card {
            text-align: center;
            padding: 1.5rem;
            border: 1px solid #e9ecef;
            border-radius: 8px;
        }
        
        .stat-card h4 {
            color: #495057;
            margin-bottom: 1rem;
        }
        
        .stat-card .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #007bff;
        }
        
        .stat-detail {
            font-size: 0.9rem;
            color: #6c757d;
            margin-top: 0.5rem;
        }
        
        .scalability-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }
        
        .scalability-card {
            padding: 1.5rem;
            border: 1px solid #e9ecef;
            border-radius: 8px;
        }
        
        .complexity-metrics {
            margin: 1rem 0;
        }
        
        .complexity-metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }
        
        .scalability-rating {
            text-align: center;
            padding: 0.5rem;
            border-radius: 4px;
            font-weight: bold;
            margin-top: 1rem;
        }
        
        .scalability-rating.excellent {
            background-color: #d4edda;
            color: #155724;
        }
        
        .scalability-rating.good {
            background-color: #d1ecf1;
            color: #0c5460;
        }
        
        .scalability-rating.fair {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .scalability-rating.poor {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        @media (max-width: 768px) {
            .performance-tables {
                grid-template-columns: 1fr;
            }
            
            .header-stats {
                grid-template-columns: 1fr 1fr;
            }
            
            .dashboard-header h1 {
                font-size: 2rem;
            }
        }
        """
    
    def _get_dashboard_js(self) -> str:
        """Get JavaScript for dashboard interactivity."""
        return """
        // Add any interactive functionality here
        console.log('Performance Dashboard loaded');
        
        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
        
        // Add tooltips for metrics
        document.querySelectorAll('.metric-value').forEach(element => {
            element.addEventListener('mouseenter', function() {
                // Could add detailed tooltips here
            });
        });
        """


def create_analytics_dashboard():
    """Create comprehensive analytics dashboard from reports."""
    
    print("🎨 Creating Analytics Dashboard...")
    print("=" * 50)
    
    # Load reports
    try:
        with open("docs/statistical_analysis_report.json", 'r') as f:
            statistical_report = json.load(f)
        
        with open("docs/performance_profiling_report.json", 'r') as f:
            profiling_report = json.load(f)
        
        # Create visualizer and generate dashboard
        visualizer = AnalyticsVisualizer()
        dashboard_html = visualizer.create_performance_dashboard(
            statistical_report,
            profiling_report,
            output_path="docs/performance_dashboard.html"
        )
        
        print("✅ Dashboard created successfully!")
        print("📄 Saved to: docs/performance_dashboard.html")
        
        # Print summary
        stats_metadata = statistical_report.get('metadata', {})
        profiling_summary = profiling_report.get('benchmark_summary', {})
        
        print(f"\n📊 Dashboard Summary:")
        print(f"  • Total benchmark results: {stats_metadata.get('total_results', 0)}")
        print(f"  • Unique solvers analyzed: {stats_metadata.get('unique_solvers', 0)}")
        print(f"  • Overall success rate: {profiling_summary.get('overall_success_rate', 0):.1%}")
        print(f"  • Efficiency score: {profiling_summary.get('efficiency_score', 0):.1f}/100")
        
    except FileNotFoundError as e:
        print(f"❌ Report file not found: {e}")
        print("💡 Run statistical analysis and performance profiling first")
    except Exception as e:
        print(f"❌ Failed to create dashboard: {e}")


if __name__ == "__main__":
    create_analytics_dashboard()