"""
Simple HTML Generator
====================

Generates simple, tabular HTML displays for benchmark results.
Focuses on clean data presentation rather than complex dashboards.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("simple_html_generator")


class SimpleHTMLGenerator:
    """Generates simple HTML pages for benchmark data."""
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize HTML generator.
        
        Args:
            data_dir: Directory containing JSON data files
            output_dir: Directory to save HTML files
        """
        if data_dir is None:
            data_dir = project_root / "docs" / "data"
        if output_dir is None:
            output_dir = project_root / "docs"
            
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.logger = get_logger("simple_html_generator")
    
    def generate_all_html(self) -> bool:
        """
        Generate all HTML pages.
        
        Returns:
            True if successful, False otherwise
        """
        
        self.logger.info("Generating HTML pages...")
        
        try:
            # Load data
            results_data = self._load_json_data("results.json")
            summary_data = self._load_json_data("summary.json")
            metadata = self._load_json_data("metadata.json")
            
            if not all([results_data, summary_data, metadata]):
                self.logger.error("Failed to load required data files")
                return False
            
            # Generate pages
            self._generate_index_page(summary_data, metadata)
            self._generate_solver_comparison_page(summary_data, metadata)
            self._generate_problem_analysis_page(summary_data, results_data)
            self._generate_wide_comparison_table(results_data)
            self._generate_environment_info_page(metadata)
            self._generate_statistical_analysis_page()
            self._generate_performance_profiling_page()
            
            self.logger.info("✅ All HTML pages generated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate HTML: {e}")
            return False
    
    def _load_json_data(self, filename: str) -> Optional[Dict]:
        """Load JSON data file."""
        
        try:
            file_path = self.data_dir / filename
            if not file_path.exists():
                self.logger.warning(f"Data file not found: {filename}")
                return None
                
            with open(file_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Failed to load {filename}: {e}")
            return None
    
    def _generate_index_page(self, summary_data: Dict, metadata: Dict) -> None:
        """Generate main index page with overview."""
        
        summary = summary_data.get('summary', {}).get('overall', {})
        solver_comparison = summary_data.get('solver_comparison', {}).get('solvers', [])
        
        # Sort solvers by success rate and speed
        top_solvers = sorted(
            solver_comparison,
            key=lambda x: (x.get('success_rate', 0), -x.get('avg_solve_time', float('inf')))
        )[:5]
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Solver Benchmark Results</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <header>
        <h1>🔬 Optimization Solver Benchmark</h1>
        <p>Comprehensive benchmarking of optimization solvers on standard problems</p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="solver_comparison.html">Solver Comparison</a>
        <a href="problem_analysis.html">Problem Analysis</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="statistical_analysis.html">Statistical Analysis</a>
        <a href="performance_profiling.html">Performance Profiling</a>
        <a href="environment_info.html">Environment Info</a>
        <a href="data/">Raw Data</a>
    </nav>
    
    <main>
        <section class="overview">
            <h2>📊 Benchmark Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Results</h3>
                    <span class="stat-value">{summary.get('total_results', 0)}</span>
                </div>
                <div class="stat-card">
                    <h3>Solvers Tested</h3>
                    <span class="stat-value">{summary.get('total_solvers', 0)}</span>
                </div>
                <div class="stat-card">
                    <h3>Problems Solved</h3>
                    <span class="stat-value">{summary.get('total_problems', 0)}</span>
                </div>
                <div class="stat-card">
                    <h3>Success Rate</h3>
                    <span class="stat-value">{summary.get('overall_success_rate', 0):.1%}</span>
                </div>
            </div>
        </section>
        
        <section class="top-performers">
            <h2>🏆 Top Performing Solvers</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Solver</th>
                        <th>Success Rate</th>
                        <th>Avg Time (s)</th>
                        <th>Problems Solved</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for i, solver in enumerate(top_solvers, 1):
            html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{solver.get('solver_name', 'Unknown')}</td>
                        <td>{solver.get('success_rate', 0):.1%}</td>
                        <td>{solver.get('avg_solve_time', 0):.4f}</td>
                        <td>{solver.get('problems_solved', 0)}</td>
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </section>
        
        <section class="recent-updates">
            <h2>📅 Latest Information</h2>
            <div class="update-info">
                <p><strong>Last Updated:</strong> {last_updated}</p>
                <p><strong>Data Format Version:</strong> {version}</p>
                <p><strong>Environment:</strong> {environment}</p>
            </div>
        </section>
    </main>
    
    <footer>
        <p>Generated automatically from benchmark results • <a href="data/results.json">Download Raw Data</a></p>
    </footer>
</body>
</html>""".format(
            last_updated=summary_data.get('summary', {}).get('last_updated', 'Unknown'),
            version=summary_data.get('metadata', {}).get('format_version', '1.0'),
            environment=metadata.get('environments', {}).get('platform', 'Unknown')
        )
        
        output_file = self.output_dir / "index.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info("📄 Generated index.html")
    
    def _generate_solver_comparison_page(self, summary_data: Dict, metadata: Dict) -> None:
        """Generate solver comparison page."""
        
        solvers = summary_data.get('solver_comparison', {}).get('solvers', [])
        solver_metadata = {s['name']: s for s in metadata.get('solvers', [])}
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Comparison - Optimization Benchmark</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <header>
        <h1>⚡ Solver Comparison</h1>
        <p>Detailed performance comparison of optimization solvers</p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="solver_comparison.html">Solver Comparison</a>
        <a href="problem_analysis.html">Problem Analysis</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="statistical_analysis.html">Statistical Analysis</a>
        <a href="performance_profiling.html">Performance Profiling</a>
        <a href="environment_info.html">Environment Info</a>
        <a href="data/">Raw Data</a>
    </nav>
    
    <main>
        <section class="solver-comparison">
            <h2>📊 Performance Comparison</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Solver</th>
                        <th>Type</th>
                        <th>Backend</th>
                        <th>Problems Attempted</th>
                        <th>Problems Solved</th>
                        <th>Success Rate</th>
                        <th>Avg Time (s)</th>
                        <th>Best Time (s)</th>
                        <th>Worst Time (s)</th>
                    </tr>
                </thead>
                <tbody>"""
        
        for solver in sorted(solvers, key=lambda x: x.get('success_rate', 0), reverse=True):
            solver_name = solver.get('solver_name', 'Unknown')
            metadata_info = solver_metadata.get(solver_name, {})
            
            html_content += f"""
                    <tr>
                        <td>{solver_name}</td>
                        <td>{metadata_info.get('type', 'Unknown')}</td>
                        <td>{metadata_info.get('backend', 'N/A')}</td>
                        <td>{solver.get('problems_attempted', 0)}</td>
                        <td>{solver.get('problems_solved', 0)}</td>
                        <td>{solver.get('success_rate', 0):.1%}</td>
                        <td>{solver.get('avg_solve_time', 0):.4f}</td>
                        <td>{solver.get('best_time', 0):.4f}</td>
                        <td>{solver.get('worst_time', 0):.4f}</td>
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </section>
        
        <section class="solver-details">
            <h2>🔧 Solver Information</h2>
            <div class="solver-grid">"""
        
        for solver_info in metadata.get('solvers', []):
            html_content += f"""
                <div class="solver-card">
                    <h3>{solver_info.get('name', 'Unknown')}</h3>
                    <p><strong>Type:</strong> {solver_info.get('type', 'Unknown')}</p>
                    <p><strong>Backend:</strong> {solver_info.get('backend', 'N/A')}</p>
                    <p><strong>Environment:</strong> {solver_info.get('environment', 'Unknown')}</p>
                </div>"""
        
        html_content += """
            </div>
        </section>
    </main>
    
    <footer>
        <p>Generated automatically from benchmark results • <a href="data/summary.json">Download Summary Data</a></p>
    </footer>
</body>
</html>"""
        
        output_file = self.output_dir / "solver_comparison.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info("📄 Generated solver_comparison.html")
    
    def _generate_problem_analysis_page(self, summary_data: Dict, results_data: Dict) -> None:
        """Generate problem analysis page."""
        
        problems = summary_data.get('problem_statistics', {}).get('problems', [])
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Problem Analysis - Optimization Benchmark</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <header>
        <h1>📋 Problem Analysis</h1>
        <p>Analysis of optimization problems and their characteristics</p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="solver_comparison.html">Solver Comparison</a>
        <a href="problem_analysis.html">Problem Analysis</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="statistical_analysis.html">Statistical Analysis</a>
        <a href="performance_profiling.html">Performance Profiling</a>
        <a href="environment_info.html">Environment Info</a>
        <a href="data/">Raw Data</a>
    </nav>
    
    <main>
        <section class="problem-overview">
            <h2>📊 Problem Statistics</h2>
            <table class="problem-table">
                <thead>
                    <tr>
                        <th>Problem</th>
                        <th>Type</th>
                        <th>Variables</th>
                        <th>Constraints</th>
                        <th>Difficulty</th>
                        <th>Solver Attempts</th>
                        <th>Successful Solves</th>
                        <th>Success Rate</th>
                        <th>Avg Time (s)</th>
                    </tr>
                </thead>
                <tbody>"""
        
        for problem in sorted(problems, key=lambda x: x.get('success_rate', 0), reverse=True):
            html_content += f"""
                    <tr>
                        <td>{problem.get('problem_name', 'Unknown')}</td>
                        <td>{problem.get('problem_type', 'Unknown')}</td>
                        <td>{problem.get('n_variables', 0)}</td>
                        <td>{problem.get('n_constraints', 0)}</td>
                        <td>{problem.get('difficulty_level', 'Unknown')}</td>
                        <td>{problem.get('solver_attempts', 0)}</td>
                        <td>{problem.get('successful_solves', 0)}</td>
                        <td>{problem.get('success_rate', 0):.1%}</td>
                        <td>{problem.get('avg_solve_time', 0):.4f}</td>
                    </tr>"""
        
        # Problem type distribution
        problem_type_dist = summary_data.get('summary', {}).get('problem_type_distribution', {})
        
        html_content += f"""
                </tbody>
            </table>
        </section>
        
        <section class="problem-distribution">
            <h2>📈 Problem Type Distribution</h2>
            <table class="distribution-table">
                <thead>
                    <tr>
                        <th>Problem Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>"""
        
        total_problems = sum(problem_type_dist.values())
        for ptype, count in sorted(problem_type_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_problems * 100) if total_problems > 0 else 0
            html_content += f"""
                    <tr>
                        <td>{ptype}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </section>
    </main>
    
    <footer>
        <p>Generated automatically from benchmark results • <a href="data/results.json">Download Complete Results</a></p>
    </footer>
</body>
</html>"""
        
        output_file = self.output_dir / "problem_analysis.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info("📄 Generated problem_analysis.html")
    
    def _generate_wide_comparison_table(self, results_data: Dict) -> None:
        """Generate wide comparison table with problems vs solvers."""
        
        results = results_data.get('results', [])
        
        # Organize data into matrix format
        problem_solver_matrix = {}
        all_problems = set()
        all_solvers = set()
        
        for result in results:
            problem_name = result.get('problem_name', 'Unknown')
            solver_name = result.get('solver_name', 'Unknown')
            
            all_problems.add(problem_name)
            all_solvers.add(solver_name)
            
            if problem_name not in problem_solver_matrix:
                problem_solver_matrix[problem_name] = {}
            
            problem_solver_matrix[problem_name][solver_name] = {
                'solve_time': result.get('solve_time'),
                'status': result.get('status', 'unknown'),
                'objective_value': result.get('objective_value'),
                'duality_gap': result.get('duality_gap'),
                'iterations': result.get('iterations'),
                'problem_type': result.get('problem_type', 'Unknown')
            }
        
        # Sort for consistent display
        sorted_problems = sorted(all_problems)
        sorted_solvers = sorted(all_solvers)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results Matrix - Optimization Benchmark</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .matrix-table {{
            font-size: 0.85em;
            width: 100%;
            overflow-x: auto;
            display: block;
            white-space: nowrap;
        }}
        .matrix-table table {{
            width: auto;
            min-width: 100%;
        }}
        .matrix-table th, .matrix-table td {{
            padding: 6px 8px;
            text-align: center;
            border: 1px solid #ddd;
            min-width: 120px;
        }}
        .matrix-table th.problem-name {{
            position: sticky;
            left: 0;
            background: #f8f9fa;
            font-weight: bold;
            min-width: 150px;
            text-align: left;
        }}
        .cell-optimal {{ background-color: #d4edda; }}
        .cell-infeasible {{ background-color: #f8d7da; }}
        .cell-error {{ background-color: #fff3cd; }}
        .cell-timeout {{ background-color: #e2e3e5; }}
        .cell-unknown {{ background-color: #f8f9fa; }}
        .result-time {{ font-weight: bold; color: #007bff; }}
        .result-obj {{ font-size: 0.9em; color: #6c757d; }}
        .result-gap {{ font-size: 0.8em; color: #28a745; }}
    </style>
</head>
<body>
    <header>
        <h1>📊 Results Matrix</h1>
        <p>Comprehensive comparison of solver performance across all problems</p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="solver_comparison.html">Solver Comparison</a>
        <a href="problem_analysis.html">Problem Analysis</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="statistical_analysis.html">Statistical Analysis</a>
        <a href="performance_profiling.html">Performance Profiling</a>
        <a href="environment_info.html">Environment Info</a>
        <a href="data/">Raw Data</a>
    </nav>
    
    <main>
        <section class="matrix-summary">
            <h2>📋 Matrix Overview</h2>
            <p><strong>Problems:</strong> {len(sorted_problems)} | <strong>Solvers:</strong> {len(sorted_solvers)} | <strong>Total Results:</strong> {len(results)}</p>
            <div class="legend">
                <span class="cell-optimal">■</span> Optimal 
                <span class="cell-infeasible">■</span> Infeasible 
                <span class="cell-error">■</span> Error 
                <span class="cell-timeout">■</span> Timeout 
                <span class="cell-unknown">■</span> Unknown
            </div>
        </section>
        
        <section class="matrix-table">
            <h2>🔍 Detailed Results</h2>
            <table>
                <thead>
                    <tr>
                        <th class="problem-name">Problem</th>"""
        
        # Add solver column headers
        for solver in sorted_solvers:
            html_content += f"""
                        <th>{solver}</th>"""
        
        html_content += """
                    </tr>
                </thead>
                <tbody>"""
        
        # Add data rows
        for problem in sorted_problems:
            problem_data = problem_solver_matrix.get(problem, {})
            problem_type = next((data.get('problem_type', 'Unknown') for data in problem_data.values() if data), 'Unknown')
            
            html_content += f"""
                    <tr>
                        <td class="problem-name">{problem}<br><small>({problem_type})</small></td>"""
            
            for solver in sorted_solvers:
                result = problem_data.get(solver)
                
                if result:
                    status = result['status']
                    solve_time = result['solve_time']
                    objective_value = result['objective_value']
                    duality_gap = result['duality_gap']
                    
                    # Determine cell class based on status
                    cell_class = f"cell-{status}" if status in ['optimal', 'infeasible', 'error', 'timeout'] else "cell-unknown"
                    
                    # Format cell content
                    cell_content = ""
                    if solve_time is not None:
                        if solve_time < 0.001:
                            time_str = f"{solve_time*1000:.2f}ms"
                        else:
                            time_str = f"{solve_time:.3f}s"
                        cell_content += f'<div class="result-time">{time_str}</div>'
                    
                    if status == 'optimal':
                        if objective_value is not None:
                            if abs(objective_value) < 1e-10:
                                obj_str = "0.0"
                            elif abs(objective_value) < 1e-3:
                                obj_str = f"{objective_value:.2e}"
                            else:
                                obj_str = f"{objective_value:.3f}"
                            cell_content += f'<div class="result-obj">obj: {obj_str}</div>'
                        
                        if duality_gap is not None and abs(duality_gap) > 1e-10:
                            if abs(duality_gap) < 1e-3:
                                gap_str = f"{duality_gap:.2e}"
                            else:
                                gap_str = f"{duality_gap:.3f}"
                            cell_content += f'<div class="result-gap">gap: {gap_str}</div>'
                    else:
                        cell_content += f'<div class="result-obj">{status}</div>'
                    
                    html_content += f"""
                        <td class="{cell_class}">{cell_content}</td>"""
                else:
                    html_content += f"""
                        <td class="cell-unknown">-</td>"""
            
            html_content += """
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </section>
    </main>
    
    <footer>
        <p>Generated automatically from benchmark results • <a href="data/results.json">Download Complete Results</a></p>
    </footer>
</body>
</html>"""
        
        output_file = self.output_dir / "results_matrix.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info("📊 Generated results_matrix.html")
    
    def _generate_environment_info_page(self, metadata: Dict) -> None:
        """Generate environment information page."""
        
        env_info = metadata.get('environments', {})
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Environment Info - Optimization Benchmark</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <header>
        <h1>🖥️ Environment Information</h1>
        <p>Technical details about the benchmark execution environment</p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="solver_comparison.html">Solver Comparison</a>
        <a href="problem_analysis.html">Problem Analysis</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="statistical_analysis.html">Statistical Analysis</a>
        <a href="performance_profiling.html">Performance Profiling</a>
        <a href="environment_info.html">Environment Info</a>
        <a href="data/">Raw Data</a>
    </nav>
    
    <main>
        <section class="environment-details">
            <h2>⚙️ System Environment</h2>
            <table class="env-table">
                <thead>
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Platform</td>
                        <td>{env_info.get('platform', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Operating System</td>
                        <td>{env_info.get('operating_system', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>OS Architecture</td>
                        <td>{env_info.get('operating_system_details', {}).get('architecture', 'Unknown')} ({env_info.get('operating_system_details', {}).get('machine', 'Unknown')})</td>
                    </tr>
                    <tr>
                        <td>Kernel Version</td>
                        <td>{env_info.get('operating_system_details', {}).get('release', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Python Version</td>
                        <td>{env_info.get('python_version', 'Unknown')} ({env_info.get('python_implementation', 'CPython')})</td>
                    </tr>
                    <tr>
                        <td>CPU Information</td>
                        <td>{env_info.get('cpu_info', {}).get('cpu_count', 'Unknown')} cores ({env_info.get('cpu_info', {}).get('cpu_count_physical', 'Unknown')} physical)</td>
                    </tr>
                    <tr>
                        <td>Memory</td>
                        <td>{env_info.get('memory_info', {}).get('total_gb', 'Unknown')} GB total, {env_info.get('memory_info', {}).get('available_gb', 'Unknown')} GB available</td>
                    </tr>
                    <tr>
                        <td>Timezone</td>
                        <td>{(env_info.get('timezone_info', {}).get('system_timezone') or env_info.get('timezone_info', {}).get('timedatectl_timezone') or env_info.get('timezone_info', {}).get('timezone_name') or 'Unknown')} {('(UTC{:+.1f})'.format(env_info.get('timezone_info', {}).get('utc_offset_hours')) if env_info.get('timezone_info', {}).get('utc_offset_hours') is not None else '')}</td>
                    </tr>
                    <tr>
                        <td>Local Time at Collection</td>
                        <td>{env_info.get('timezone_info', {}).get('current_time_local') or 'Not available'}</td>
                    </tr>
                    <tr>
                        <td>Primary Framework</td>
                        <td>{env_info.get('primary_solver_framework', 'Unknown')}</td>
                    </tr>
                    <tr>
                        <td>Last Updated</td>
                        <td>{env_info.get('last_updated', 'Unknown')}</td>
                    </tr>
                </tbody>
            </table>
        </section>
        
        <section class="supported-types">
            <h2>📋 Supported Problem Types</h2>
            <ul class="problem-types-list">"""
        
        for ptype in env_info.get('supported_problem_types', []):
            html_content += f"""
                <li>{ptype}</li>"""
        
        html_content += f"""
            </ul>
        </section>
        
        <section class="solver-overview">
            <h2>🔧 Available Solvers</h2>
            <table class="solver-table">
                <thead>
                    <tr>
                        <th>Solver</th>
                        <th>Type</th>
                        <th>Backend</th>
                        <th>Environment</th>
                    </tr>
                </thead>
                <tbody>"""
        
        for solver in metadata.get('solvers', []):
            html_content += f"""
                    <tr>
                        <td>{solver.get('name', 'Unknown')}</td>
                        <td>{solver.get('type', 'Unknown')}</td>
                        <td>{solver.get('backend', 'N/A')}</td>
                        <td>{solver.get('environment', 'Unknown')}</td>
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </section>
    </main>
    
    <footer>
        <p>Generated automatically from benchmark results • <a href="data/metadata.json">Download Metadata</a></p>
    </footer>
</body>
</html>"""
        
        output_file = self.output_dir / "environment_info.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info("📄 Generated environment_info.html")
    
    def _generate_statistical_analysis_page(self) -> None:
        """Generate statistical analysis page from existing JSON report."""
        
        # Try to load from both locations (docs_archive and docs)
        report_paths = [
            self.output_dir.parent / "docs_archive" / "statistical_analysis_report.json",
            self.output_dir / "statistical_analysis_report.json",
            self.data_dir / "statistical_analysis_report.json"
        ]
        
        statistical_data = None
        for report_path in report_paths:
            if report_path.exists():
                try:
                    with open(report_path, 'r') as f:
                        statistical_data = json.load(f)
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load {report_path}: {e}")
        
        if not statistical_data:
            self.logger.warning("No statistical analysis report found, skipping statistical analysis page")
            return
        
        metadata = statistical_data.get('metadata', {})
        solver_metrics = statistical_data.get('solver_metrics', {})
        characterizations = statistical_data.get('solver_characterizations', {})
        comparisons = statistical_data.get('pairwise_comparisons', {})
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistical Analysis - Optimization Benchmark</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .metrics-table {{ width: 100%; margin-bottom: 2rem; }}
        .metrics-table th, .metrics-table td {{ padding: 0.75rem; text-align: center; }}
        .metrics-table .solver-name {{ text-align: left; font-weight: bold; }}
        .characterization-card {{ 
            background: #f8f9fa; 
            padding: 1.5rem; 
            margin-bottom: 1rem; 
            border-radius: 8px; 
            border-left: 4px solid #007bff;
        }}
        .score-high {{ color: #28a745; font-weight: bold; }}
        .score-medium {{ color: #ffc107; font-weight: bold; }}
        .score-low {{ color: #dc3545; font-weight: bold; }}
        .strengths {{ color: #28a745; }}
        .weaknesses {{ color: #dc3545; }}
        .failure-modes {{ color: #6c757d; font-style: italic; }}
        .comparison-table {{ font-size: 0.9em; }}
        .significant {{ background-color: #fff3cd; font-weight: bold; }}
    </style>
</head>
<body>
    <header>
        <h1>📊 Statistical Analysis</h1>
        <p>Advanced statistical analysis and solver characterization</p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="solver_comparison.html">Solver Comparison</a>
        <a href="problem_analysis.html">Problem Analysis</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="statistical_analysis.html">Statistical Analysis</a>
        <a href="performance_profiling.html">Performance Profiling</a>
        <a href="environment_info.html">Environment Info</a>
        <a href="data/">Raw Data</a>
    </nav>
    
    <main>
        <section class="analysis-overview">
            <h2>📈 Analysis Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Analysis Date</h3>
                    <span class="stat-value">{metadata.get('analysis_date', 'Unknown')[:10]}</span>
                </div>
                <div class="stat-card">
                    <h3>Total Results</h3>
                    <span class="stat-value">{metadata.get('total_results', 0)}</span>
                </div>
                <div class="stat-card">
                    <h3>Unique Solvers</h3>
                    <span class="stat-value">{metadata.get('unique_solvers', 0)}</span>
                </div>
                <div class="stat-card">
                    <h3>Unique Problems</h3>
                    <span class="stat-value">{metadata.get('unique_problems', 0)}</span>
                </div>
            </div>
        </section>
        
        <section class="problem-types">
            <h2>📋 Problem Type Distribution</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Problem Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>"""
        
        problem_types = metadata.get('problem_types', {})
        total_problems = sum(problem_types.values()) if problem_types else 1
        
        for ptype, count in sorted(problem_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_problems * 100) if total_problems > 0 else 0
            html_content += f"""
                    <tr>
                        <td class="solver-name">{ptype}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </section>
        
        <section class="performance-metrics">
            <h2>⚡ Performance Metrics</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Solver</th>
                        <th>Success Rate</th>
                        <th>Geometric Mean Time (s)</th>
                        <th>Relative Performance</th>
                        <th>Coefficient of Variation</th>
                        <th>Problems/Second</th>
                    </tr>
                </thead>
                <tbody>"""
        
        # Sort solvers by relative performance
        sorted_metrics = sorted(
            solver_metrics.items(),
            key=lambda x: x[1].get('relative_performance', float('inf'))
        )
        
        for solver_name, metrics in sorted_metrics:
            success_rate = metrics.get('success_rate', 0)
            geom_mean_time = metrics.get('geometric_mean_time', float('inf'))
            rel_performance = metrics.get('relative_performance', float('inf'))
            cv = metrics.get('coefficient_of_variation', float('inf'))
            problems_per_sec = metrics.get('problems_per_second', 0)
            
            # Format values for display
            geom_str = f"{geom_mean_time:.4f}" if geom_mean_time != float('inf') else "∞"
            rel_str = f"{rel_performance:.2f}x" if rel_performance != float('inf') else "∞"
            cv_str = f"{cv:.3f}" if cv != float('inf') else "∞"
            
            html_content += f"""
                    <tr>
                        <td class="solver-name">{solver_name}</td>
                        <td>{success_rate:.1%}</td>
                        <td>{geom_str}</td>
                        <td>{rel_str}</td>
                        <td>{cv_str}</td>
                        <td>{problems_per_sec:.1f}</td>
                    </tr>"""
        
        html_content += """
                </tbody>
            </table>
        </section>
        
        <section class="solver-characterizations">
            <h2>🎯 Solver Characterizations</h2>"""
        
        # Sort by overall score
        sorted_chars = sorted(
            characterizations.items(),
            key=lambda x: x[1].get('overall_score', 0),
            reverse=True
        )
        
        for solver_name, char in sorted_chars:
            overall_score = char.get('overall_score', 0)
            scaling_coeff = char.get('scaling_coefficient', float('inf'))
            size_sensitivity = char.get('size_sensitivity', 'Unknown')
            stability_score = char.get('stability_score', 0)
            strengths = char.get('strengths', [])
            weaknesses = char.get('weaknesses', [])
            failure_modes = char.get('failure_modes', [])
            
            # Determine score class
            if overall_score >= 80:
                score_class = "score-high"
            elif overall_score >= 50:
                score_class = "score-medium"
            else:
                score_class = "score-low"
            
            # Format scaling coefficient
            scaling_str = f"{scaling_coeff:.3f}" if scaling_coeff != float('inf') else "∞"
            
            html_content += f"""
            <div class="characterization-card">
                <h3>{solver_name} <span class="{score_class}">({overall_score:.1f}/100)</span></h3>
                <div class="char-details">
                    <p><strong>Size Sensitivity:</strong> {size_sensitivity} 
                       <small>(scaling coefficient: {scaling_str})</small></p>
                    <p><strong>Stability Score:</strong> {stability_score:.3f}/1.000</p>
                    
                    {f'<p class="strengths"><strong>Strengths:</strong> {", ".join(strengths)}</p>' if strengths else ''}
                    {f'<p class="weaknesses"><strong>Weaknesses:</strong> {", ".join(weaknesses)}</p>' if weaknesses else ''}
                    {f'<p class="failure-modes"><strong>Failure Modes:</strong> {", ".join(failure_modes)}</p>' if failure_modes else ''}
                </div>
            </div>"""
        
        html_content += """
        </section>"""
        
        # Add pairwise comparisons if available
        if comparisons:
            html_content += """
        <section class="pairwise-comparisons">
            <h2>📊 Pairwise Comparisons</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Comparison</th>
                        <th>Common Problems</th>
                        <th>Statistically Significant</th>
                        <th>Effect Size</th>
                        <th>Performance Ratio</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody>"""
            
            for comp_name, comp_data in comparisons.items():
                n_common = comp_data.get('n_common_problems', 0)
                significant = comp_data.get('wilcoxon_significant', False)
                effect_size = comp_data.get('effect_size', 0)
                perf_ratio = comp_data.get('performance_ratio', 1)
                win_rate = comp_data.get('win_rate', 0.5)
                
                # Format comparison name
                comp_display = comp_name.replace('_vs_', ' vs ').replace('_', ' ')
                
                # Determine if significant
                sig_class = "significant" if significant else ""
                sig_text = "Yes" if significant else "No"
                
                html_content += f"""
                    <tr class="{sig_class}">
                        <td class="solver-name">{comp_display}</td>
                        <td>{n_common}</td>
                        <td>{sig_text}</td>
                        <td>{effect_size:.3f}</td>
                        <td>{perf_ratio:.3f}</td>
                        <td>{win_rate:.1%}</td>
                    </tr>"""
            
            html_content += """
                </tbody>
            </table>
            <p><small><em>Note: Highlighted rows indicate statistically significant differences (p &lt; 0.05).</em></small></p>
        </section>"""
        
        html_content += """
    </main>
    
    <footer>
        <p>Generated automatically from statistical analysis • <a href="data/statistical_analysis_report.json">Download Report Data</a></p>
    </footer>
</body>
</html>"""
        
        output_file = self.output_dir / "statistical_analysis.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info("📊 Generated statistical_analysis.html")
    
    def _generate_performance_profiling_page(self) -> None:
        """Generate performance profiling page from existing JSON report."""
        
        # Try to load from both locations (docs_archive and docs)
        report_paths = [
            self.output_dir.parent / "docs_archive" / "performance_profiling_report.json",
            self.output_dir / "performance_profiling_report.json",
            self.data_dir / "performance_profiling_report.json"
        ]
        
        profiling_data = None
        for report_path in report_paths:
            if report_path.exists():
                try:
                    with open(report_path, 'r') as f:
                        profiling_data = json.load(f)
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load {report_path}: {e}")
        
        if not profiling_data:
            self.logger.warning("No performance profiling report found, skipping performance profiling page")
            return
        
        metadata = profiling_data.get('metadata', {})
        benchmark_summary = profiling_data.get('benchmark_summary', {})
        solver_profiles = profiling_data.get('solver_profiles', {})
        rankings = profiling_data.get('performance_rankings', {})
        scalability = profiling_data.get('scalability_analysis', {})
        resource_util = profiling_data.get('resource_utilization', {})
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Profiling - Optimization Benchmark</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <style>
        .profile-card {{ 
            background: #f8f9fa; 
            padding: 1.5rem; 
            margin-bottom: 1.5rem; 
            border-radius: 8px; 
            border-left: 4px solid #28a745;
        }}
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 1rem; 
            margin-bottom: 1rem;
        }}
        .metric-box {{ 
            background: white; 
            padding: 1rem; 
            border-radius: 4px; 
            border: 1px solid #dee2e6;
            text-align: center;
        }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 0.9rem; color: #6c757d; }}
        .ranking-table {{ width: 100%; margin-bottom: 1.5rem; }}
        .ranking-table th, .ranking-table td {{ padding: 0.75rem; text-align: center; }}
        .ranking-table .solver-name {{ text-align: left; font-weight: bold; }}
        .rank-1 {{ background-color: #d4edda; }}
        .rank-2 {{ background-color: #f8f9fa; }}
        .rank-3 {{ background-color: #fff3cd; }}
        .scalability-excellent {{ color: #28a745; font-weight: bold; }}
        .scalability-good {{ color: #17a2b8; font-weight: bold; }}
        .scalability-poor {{ color: #dc3545; font-weight: bold; }}
        .quality-high {{ color: #28a745; }}
        .quality-medium {{ color: #ffc107; }}
        .quality-low {{ color: #dc3545; }}
    </style>
</head>
<body>
    <header>
        <h1>⚡ Performance Profiling</h1>
        <p>Detailed performance analysis and resource utilization profiling</p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="solver_comparison.html">Solver Comparison</a>
        <a href="problem_analysis.html">Problem Analysis</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="statistical_analysis.html">Statistical Analysis</a>
        <a href="performance_profiling.html">Performance Profiling</a>
        <a href="environment_info.html">Environment Info</a>
        <a href="data/">Raw Data</a>
    </nav>
    
    <main>
        <section class="profiling-overview">
            <h2>📊 Benchmark Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Runtime</h3>
                    <span class="stat-value">{benchmark_summary.get('total_runtime', 0):.4f}s</span>
                </div>
                <div class="stat-card">
                    <h3>Total Problems</h3>
                    <span class="stat-value">{benchmark_summary.get('total_problems', 0)}</span>
                </div>
                <div class="stat-card">
                    <h3>Total Solvers</h3>
                    <span class="stat-value">{benchmark_summary.get('total_solvers', 0)}</span>
                </div>
                <div class="stat-card">
                    <h3>Overall Success Rate</h3>
                    <span class="stat-value">{benchmark_summary.get('overall_success_rate', 0):.1%}</span>
                </div>
                <div class="stat-card">
                    <h3>Efficiency Score</h3>
                    <span class="stat-value">{benchmark_summary.get('efficiency_score', 0):.1f}/100</span>
                </div>
            </div>
        </section>
        
        <section class="performance-rankings">
            <h2>🏆 Performance Rankings</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;">
                <div>
                    <h3>⚡ Fastest Solvers</h3>
                    <table class="ranking-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Solver</th>
                                <th>Avg Time (s)</th>
                            </tr>
                        </thead>
                        <tbody>"""
        
        fastest_solvers = rankings.get('fastest_solvers', [])
        for i, (solver, time) in enumerate(fastest_solvers[:5], 1):
            rank_class = f"rank-{i}" if i <= 3 else ""
            html_content += f"""
                            <tr class="{rank_class}">
                                <td>{i}</td>
                                <td class="solver-name">{solver}</td>
                                <td>{time:.4f}</td>
                            </tr>"""
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div>
                    <h3>🎯 Most Reliable</h3>
                    <table class="ranking-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Solver</th>
                                <th>Success Rate</th>
                            </tr>
                        </thead>
                        <tbody>"""
        
        most_reliable = rankings.get('most_reliable', [])
        for i, (solver, rate) in enumerate(most_reliable[:5], 1):
            rank_class = f"rank-{i}" if i <= 3 else ""
            html_content += f"""
                            <tr class="{rank_class}">
                                <td>{i}</td>
                                <td class="solver-name">{solver}</td>
                                <td>{rate:.1%}</td>
                            </tr>"""
        
        html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
        
        <section class="solver-profiles">
            <h2>🔬 Detailed Solver Profiles</h2>"""
        
        # Sort solver profiles by average solve time
        sorted_profiles = sorted(
            solver_profiles.items(),
            key=lambda x: x[1].get('performance_metrics', {}).get('avg_solve_time', float('inf'))
        )
        
        for solver_name, profile in sorted_profiles:
            perf_metrics = profile.get('performance_metrics', {})
            resource_usage = profile.get('resource_usage', {})
            scalability_info = profile.get('scalability', {})
            quality = profile.get('quality', {})
            
            # Determine scalability class
            scalability_rating = scalability_info.get('scalability_rating', 'Unknown')
            if scalability_rating == 'Excellent':
                scalability_class = 'scalability-excellent'
            elif scalability_rating == 'Good':
                scalability_class = 'scalability-good'
            else:
                scalability_class = 'scalability-poor'
            
            html_content += f"""
            <div class="profile-card">
                <h3>{solver_name}</h3>
                
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">{perf_metrics.get('avg_solve_time', 0):.4f}s</div>
                        <div class="metric-label">Avg Solve Time</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{perf_metrics.get('throughput', 0):.1f}</div>
                        <div class="metric-label">Problems/Second</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{perf_metrics.get('success_rate', 0):.1%}</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{resource_usage.get('peak_memory_mb', 0):.1f} MB</div>
                        <div class="metric-label">Peak Memory</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{resource_usage.get('cpu_efficiency', 0):.3f}</div>
                        <div class="metric-label">CPU Efficiency</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value {scalability_class}">{scalability_rating}</div>
                        <div class="metric-label">Scalability</div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div>
                        <h4>Quality Metrics</h4>
                        <p><strong>Solution Quality:</strong> {quality.get('solution_quality', 0):.3f}</p>
                        <p><strong>Numerical Stability:</strong> {quality.get('numerical_stability', 0):.3f}</p>
                        <p><strong>Convergence Reliability:</strong> {quality.get('convergence_reliability', 0):.3f}</p>
                    </div>
                    <div>
                        <h4>Scalability Analysis</h4>
                        <p><strong>Time Complexity:</strong> {scalability_info.get('time_complexity', 0):.3f}</p>
                        <p><strong>Memory Complexity:</strong> {scalability_info.get('memory_complexity', 0):.3f}</p>
                        <p><strong>Memory Efficiency:</strong> {resource_usage.get('memory_efficiency', 0):.3f}</p>
                    </div>
                </div>
            </div>"""
        
        html_content += """
        </section>
        
        <section class="scalability-analysis">
            <h2>📈 Scalability Analysis</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Excellent Scalability</h3>
                    <span class="stat-value">{excellent_count}</span>
                </div>
                <div class="stat-card">
                    <h3>Good Scalability</h3>
                    <span class="stat-value">{good_count}</span>
                </div>
                <div class="stat-card">
                    <h3>Avg Time Complexity</h3>
                    <span class="stat-value">{avg_time_complexity:.3f}</span>
                </div>
                <div class="stat-card">
                    <h3>Avg Memory Complexity</h3>
                    <span class="stat-value">{avg_memory_complexity:.3f}</span>
                </div>
            </div>
        </section>
        
        <section class="resource-utilization">
            <h2>💾 Resource Utilization</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Memory Usage</h3>
                    <span class="stat-value">{total_memory:.1f} MB</span>
                </div>
                <div class="stat-card">
                    <h3>Avg CPU Efficiency</h3>
                    <span class="stat-value">{avg_cpu_efficiency:.1%}</span>
                </div>
                <div class="stat-card">
                    <h3>Total Throughput</h3>
                    <span class="stat-value">{total_throughput:.1f} prob/s</span>
                </div>
            </div>
        </section>
    </main>
    
    <footer>
        <p>Generated automatically from performance profiling • <a href="data/performance_profiling_report.json">Download Report Data</a></p>
    </footer>
</body>
</html>""".format(
            excellent_count=scalability.get('distribution', {}).get('Excellent', 0),
            good_count=scalability.get('distribution', {}).get('Good', 0),
            avg_time_complexity=scalability.get('average_time_complexity', 0),
            avg_memory_complexity=scalability.get('average_memory_complexity', 0),
            total_memory=resource_util.get('total_memory_usage', 0),
            avg_cpu_efficiency=resource_util.get('average_cpu_efficiency', 0),
            total_throughput=resource_util.get('total_throughput', 0)
        )
        
        output_file = self.output_dir / "performance_profiling.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info("⚡ Generated performance_profiling.html")
    
    def generate_css(self) -> None:
        """Generate CSS stylesheet."""
        
        css_content = """/* Simple Benchmark Results Stylesheet */

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

header {
    background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    color: white;
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 2rem;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

header p {
    font-size: 1.1rem;
    opacity: 0.9;
}

nav {
    background: white;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

nav a {
    color: #2c3e50;
    text-decoration: none;
    margin: 0 1rem;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    transition: background-color 0.2s;
}

nav a:hover {
    background-color: #ecf0f1;
}

main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
}

section {
    background: white;
    margin-bottom: 2rem;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h2 {
    color: #2c3e50;
    margin-bottom: 1.5rem;
    font-size: 1.8rem;
}

h3 {
    color: #34495e;
    margin-bottom: 1rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: #ecf0f1;
    padding: 1.5rem;
    border-radius: 8px;
    text-align: center;
}

.stat-card h3 {
    color: #7f8c8d;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stat-value {
    font-size: 2rem;
    font-weight: bold;
    color: #2c3e50;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

th, td {
    text-align: left;
    padding: 0.75rem;
    border-bottom: 1px solid #ecf0f1;
}

th {
    background-color: #f8f9fa;
    font-weight: 600;
    color: #2c3e50;
}

tr:hover {
    background-color: #f8f9fa;
}

.solver-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
}

.solver-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #3498db;
}

.solver-card h3 {
    color: #2c3e50;
    margin-bottom: 1rem;
}

.solver-card p {
    margin-bottom: 0.5rem;
    color: #7f8c8d;
}

.problem-types-list {
    list-style: none;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 0.5rem;
}

.problem-types-list li {
    background: #e8f5e8;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    text-align: center;
    font-weight: 500;
    color: #27ae60;
}

.update-info {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 4px;
    border-left: 4px solid #ffc107;
}

.update-info p {
    margin-bottom: 0.5rem;
}

footer {
    text-align: center;
    padding: 2rem;
    color: #7f8c8d;
    border-top: 1px solid #ecf0f1;
    margin-top: 2rem;
}

footer a {
    color: #3498db;
    text-decoration: none;
}

footer a:hover {
    text-decoration: underline;
}

@media (max-width: 768px) {
    header h1 {
        font-size: 2rem;
    }
    
    nav a {
        display: block;
        margin: 0.5rem 0;
    }
    
    .stats-grid {
        grid-template-columns: 1fr 1fr;
    }
    
    table {
        font-size: 0.9rem;
    }
    
    th, td {
        padding: 0.5rem;
    }
}"""

        # Ensure assets directory exists
        assets_dir = self.output_dir / "assets" / "css"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        css_file = assets_dir / "style.css"
        with open(css_file, 'w') as f:
            f.write(css_content)
        
        self.logger.info("🎨 Generated style.css")


def generate_simple_html():
    """Main function to generate simple HTML pages."""
    
    print("🎨 Generating Simple HTML Pages...")
    print("=" * 50)
    
    generator = SimpleHTMLGenerator()
    
    # Generate CSS first
    generator.generate_css()
    
    # Generate HTML pages
    success = generator.generate_all_html()
    
    if success:
        print("✅ HTML generation completed successfully!")
        print("\n📄 Generated Pages:")
        print("  • index.html - Main overview page")
        print("  • solver_comparison.html - Solver performance comparison")
        print("  • problem_analysis.html - Problem statistics and analysis")
        print("  • environment_info.html - System environment details")
        print("  • assets/css/style.css - Stylesheet")
        
        print("\n🌐 Open docs/index.html in your browser to view results")
        
    else:
        print("❌ HTML generation failed. Check logs for details.")


if __name__ == "__main__":
    generate_simple_html()