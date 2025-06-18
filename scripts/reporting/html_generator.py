"""
Simplified HTML Generator for Three Report Types
===============================================

Generates three focused HTML reports as specified in the re-architected design:
1. Overview Dashboard - Summary statistics and solver/problem counts
2. Results Matrix - Problems × solvers matrix with solve times and status
3. Raw Data - Detailed table with all result fields

Simple HTML structure without complex Bootstrap dashboards.
"""

from pathlib import Path
from typing import List, Dict, Any
import sys
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.result_processor import ResultProcessor, BenchmarkResult
from scripts.utils.logger import get_logger

logger = get_logger("html_generator")


class HTMLGenerator:
    """Generate simplified HTML reports for benchmark results"""
    
    def __init__(self, output_dir: str = None):
        """Initialize HTML generator with output directory"""
        if output_dir is None:
            output_dir = project_root / "docs" / "pages"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.result_processor = ResultProcessor()
        self.logger = get_logger("html_generator")
    
    def _get_common_css(self) -> str:
        """Get common CSS styles for all reports"""
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
        
        nav a.active {
            background-color: #3498db;
            color: white;
        }
        
        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .section {
            background: white;
            margin: 2rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 1rem 1.5rem;
            margin: 0;
            border-radius: 8px 8px 0 0;
            font-size: 1.5rem;
        }
        
        .section-content {
            padding: 1.5rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            position: sticky;
            top: 0;
        }
        
        tr:hover {
            background-color: #f8f9fa;
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            margin-top: 3rem;
        }
        
        footer a {
            color: #3498db;
            text-decoration: none;
            margin: 0 1rem;
        }
        
        footer a:hover {
            text-decoration: underline;
        }
        """
    
    def generate_all_reports(self) -> bool:
        """Generate all three HTML reports"""
        
        self.logger.info("Generating simplified HTML reports...")
        
        try:
            # Get latest results
            results = self.result_processor.get_latest_results_for_reporting()
            
            if not results:
                self.logger.warning("No results found for report generation")
                return False
            
            # Generate all three reports
            self.generate_overview(results)
            self.generate_results_matrix(results)
            self.generate_raw_data(results)
            
            self.logger.info("All simplified HTML reports generated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML reports: {e}")
            return False
    
    def generate_overview(self, results: List[BenchmarkResult]) -> str:
        """Generate overview dashboard showing summary statistics"""
        
        self.logger.info("Generating overview dashboard...")
        
        # Get summary statistics
        summary = self.result_processor.get_summary_statistics(results)
        solver_comparison = self.result_processor.get_solver_comparison(results)
        
        # Generate environment info from latest result
        env_info = results[0].environment_info if results else {}
        commit_hash = results[0].commit_hash if results else "unknown"
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Solver Benchmark - Overview</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            text-align: center;
            padding: 2rem 1rem;
            margin-bottom: 2rem;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        
        header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        nav {{
            background: white;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        
        nav a {{
            color: #2c3e50;
            text-decoration: none;
            margin: 0 1rem;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background-color 0.2s;
        }}
        
        nav a:hover {{
            background-color: #ecf0f1;
        }}
        
        nav a.active {{
            background-color: #3498db;
            color: white;
        }}
        
        main {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        
        .stat-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        
        .stat-card h3 {{
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #3498db;
        }}
        
        .section {{
            background: white;
            margin: 2rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 1rem 1.5rem;
            margin: 0;
            border-radius: 8px 8px 0 0;
            font-size: 1.5rem;
        }}
        
        .section-content {{
            padding: 1.5rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .success-rate {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        .solve-time {{
            color: #7f8c8d;
            font-family: 'Courier New', monospace;
        }}
        
        .metadata {{
            background: #ecf0f1;
            padding: 1.5rem;
            border-radius: 6px;
            margin: 2rem 0;
        }}
        
        .metadata h3 {{
            color: #2c3e50;
            margin-bottom: 1rem;
        }}
        
        .metadata p {{
            margin: 0.5rem 0;
            color: #34495e;
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            margin-top: 3rem;
        }}
        
        footer a {{
            color: #3498db;
            text-decoration: none;
            margin: 0 1rem;
        }}
        
        footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🔬 Optimization Solver Benchmark</h1>
        <p>Overview Dashboard - Latest Results</p>
        <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </header>
    
    <nav>
        <a href="index.html" class="active">Overview</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="raw_data.html">Raw Data</a>
        <a href="data/">Data Exports</a>
    </nav>
    
    <main>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Results</h3>
                <span class="stat-value">{summary['total_results']}</span>
            </div>
            <div class="stat-card">
                <h3>Solvers Tested</h3>
                <span class="stat-value">{summary['total_solvers']}</span>
            </div>
            <div class="stat-card">
                <h3>Problems Tested</h3>
                <span class="stat-value">{summary['total_problems']}</span>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <span class="stat-value">{summary['success_rate']:.1%}</span>
            </div>
            <div class="stat-card">
                <h3>Avg Solve Time</h3>
                <span class="stat-value">{summary['avg_solve_time']:.3f}s</span>
            </div>
        </div>

        <div class="section">
            <h2>🏆 Solver Comparison</h2>
            <div class="section-content">
                <table>
                    <thead>
                        <tr>
                            <th>Solver</th>
                            <th>Problems Attempted</th>
                            <th>Problems Solved</th>
                            <th>Success Rate</th>
                            <th>Avg Solve Time</th>
                            <th>Min Time</th>
                            <th>Max Time</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        for solver in solver_comparison:
            html_content += f"""
            <tr>
                <td><strong>{solver['solver_name']}</strong></td>
                <td>{solver['problems_attempted']}</td>
                <td>{solver['problems_solved']}</td>
                <td class="success-rate">{solver['success_rate']:.1%}</td>
                <td class="solve-time">{solver['avg_solve_time']:.4f}s</td>
                <td class="solve-time">{solver['min_solve_time']:.4f}s</td>
                <td class="solve-time">{solver['max_solve_time']:.4f}s</td>
            </tr>"""
        
        html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>📊 Problem Type Distribution</h2>
            <div class="section-content">
                <table>
                    <thead>
                        <tr>
                            <th>Problem Type</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        total_problems = summary['total_results']
        for ptype, count in summary['problem_type_distribution'].items():
            percentage = (count / total_problems) * 100 if total_problems > 0 else 0
            html_content += f"""
            <tr>
                <td><strong>{ptype}</strong></td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>"""
        
        html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>📚 Library Distribution</h2>
            <div class="section-content">
                <table>
                    <thead>
                        <tr>
                            <th>Library</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        for library, count in summary['library_distribution'].items():
            percentage = (count / total_problems) * 100 if total_problems > 0 else 0
            html_content += f"""
            <tr>
                <td><strong>{library}</strong></td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>"""
        
        html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="metadata">
            <h3>🔧 Environment Information</h3>
            <p><strong>Commit Hash:</strong> {commit_hash[:12]}...</p>
            <p><strong>Platform:</strong> {env_info.get('platform', 'Unknown')}</p>
            <p><strong>Python Version:</strong> {env_info.get('python_version', 'Unknown')}</p>
        </div>
    </main>

    <footer>
        <p>
            <a href="results_matrix.html">Results Matrix</a> |
            <a href="raw_data.html">Raw Data</a> |
            <a href="data/">Data Exports</a>
        </p>
        <p><small>Generated by Optimization Solver Benchmark System</small></p>
    </footer>
</body>
</html>"""
        
        # Save to file
        output_file = self.output_dir / "index.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Overview report saved to {output_file}")
        return html_content
    
    def generate_results_matrix(self, results: List[BenchmarkResult]) -> str:
        """Generate problems × solvers results matrix"""
        
        self.logger.info("Generating results matrix...")
        
        # Get matrix data
        matrix_data = self.result_processor.get_results_matrix(results)
        problems = matrix_data['problems']
        solvers = matrix_data['solvers']
        matrix = matrix_data['matrix']
        
        # Copy the same CSS from overview
        css_styles = """
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
        
        nav a.active {
            background-color: #3498db;
            color: white;
        }
        
        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .section {
            background: white;
            margin: 2rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 1rem 1.5rem;
            margin: 0;
            border-radius: 8px 8px 0 0;
            font-size: 1.5rem;
        }
        
        .section-content {
            padding: 1.5rem;
        }
        
        .matrix-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9em;
        }
        
        .matrix-table th, .matrix-table td {
            padding: 10px 8px;
            text-align: center;
            border: 1px solid #ecf0f1;
        }
        
        .matrix-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .matrix-table .problem-name {
            text-align: left;
            font-weight: bold;
            background-color: #f8f9fa;
            color: #2c3e50;
        }
        
        .status-optimal {
            background-color: #d4edda;
            color: #155724;
            font-weight: bold;
        }
        
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            font-weight: bold;
        }
        
        .status-infeasible {
            background-color: #fff3cd;
            color: #856404;
            font-weight: bold;
        }
        
        .status-unknown {
            background-color: #e2e3e5;
            color: #6c757d;
        }
        
        .solve-time {
            font-size: 0.8em;
            color: #7f8c8d;
            font-family: 'Courier New', monospace;
        }
        
        .legend {
            margin: 2rem 0;
            padding: 1.5rem;
            background: #ecf0f1;
            border-radius: 6px;
        }
        
        .legend h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .legend p {
            margin: 0.5rem 0;
            color: #34495e;
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            margin-top: 3rem;
        }
        
        footer a {
            color: #3498db;
            text-decoration: none;
            margin: 0 1rem;
        }
        
        footer a:hover {
            text-decoration: underline;
        }
        """
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Solver Benchmark - Results Matrix</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <header>
        <h1>🔬 Optimization Solver Benchmark</h1>
        <p>Results Matrix - Problems × Solvers</p>
        <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="results_matrix.html" class="active">Results Matrix</a>
        <a href="raw_data.html">Raw Data</a>
        <a href="data/">Data Exports</a>
    </nav>
    
    <main>

        <div class="section">
            <h2>📊 Results Matrix</h2>
            <div class="section-content">
                <table class="matrix-table">
                    <thead>
                        <tr>
                            <th>Problem</th>"""
        
        for solver in solvers:
            html_content += f"<th>{solver}</th>"
        
        html_content += """
                        </tr>
                    </thead>
                    <tbody>"""
        
        for problem in problems:
            html_content += f"""
            <tr>
                <td class="problem-name">{problem}</td>"""
            
            for solver in solvers:
                result = matrix[problem][solver]
                if result is None:
                    html_content += '<td class="status-unknown">—</td>'
                else:
                    status = result['status'] or 'unknown'
                    solve_time = result['solve_time']
                    
                    # Determine CSS class based on status
                    if status == 'optimal':
                        css_class = 'status-optimal'
                    elif status == 'error':
                        css_class = 'status-error'
                    elif status in ['infeasible', 'unbounded']:
                        css_class = 'status-infeasible'
                    else:
                        css_class = 'status-unknown'
                    
                    # Format cell content
                    cell_content = status.upper()
                    if solve_time is not None and solve_time > 0:
                        cell_content += f'<br><span class="solve-time">{solve_time:.3f}s</span>'
                    
                    html_content += f'<td class="{css_class}">{cell_content}</td>'
            
            html_content += "</tr>"
        
        html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="legend">
            <h3>📋 Status Legend</h3>
            <p><span class="status-optimal" style="padding: 5px 10px; border-radius: 3px;">OPTIMAL</span> - Successfully solved to optimality</p>
            <p><span class="status-error" style="padding: 5px 10px; border-radius: 3px;">ERROR</span> - Solver encountered an error</p>
            <p><span class="status-infeasible" style="padding: 5px 10px; border-radius: 3px;">INFEASIBLE/UNBOUNDED</span> - Problem has no feasible solution</p>
            <p><span class="status-unknown" style="padding: 5px 10px; border-radius: 3px;">—</span> - No result available</p>
        </div>
    </main>

    <footer>
        <p>
            <a href="index.html">Overview</a> |
            <a href="raw_data.html">Raw Data</a> |
            <a href="data/">Data Exports</a>
        </p>
        <p><small>Generated by Optimization Solver Benchmark System</small></p>
    </footer>
</body>
</html>"""
        
        # Save to file
        output_file = self.output_dir / "results_matrix.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Results matrix saved to {output_file}")
        return html_content
    
    def generate_raw_data(self, results: List[BenchmarkResult]) -> str:
        """Generate raw data table for detailed inspection"""
        
        self.logger.info("Generating raw data report...")
        
        # Use the same professional CSS as other reports
        css_styles = """
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
        
        nav a.active {
            background-color: #3498db;
            color: white;
        }
        
        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .section {
            background: white;
            margin: 2rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            color: white;
            padding: 1rem 1.5rem;
            margin: 0;
            border-radius: 8px 8px 0 0;
            font-size: 1.5rem;
        }
        
        .section-content {
            padding: 1.5rem;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9em;
        }
        
        .data-table th, .data-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .data-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            position: sticky;
            top: 0;
            border-bottom: 2px solid #dee2e6;
        }
        
        .data-table tr:hover {
            background-color: #f8f9fa;
        }
        
        .status-optimal {
            background-color: #d4edda;
            color: #155724;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        .status-infeasible {
            background-color: #fff3cd;
            color: #856404;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        .solver-name {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .problem-type {
            font-weight: 500;
            color: #3498db;
        }
        
        .library-name {
            font-style: italic;
            color: #7f8c8d;
        }
        
        .number {
            text-align: right;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .timestamp {
            font-size: 0.85em;
            color: #7f8c8d;
            font-family: 'Courier New', monospace;
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            margin-top: 3rem;
        }
        
        footer a {
            color: #3498db;
            text-decoration: none;
            margin: 0 1rem;
        }
        
        footer a:hover {
            text-decoration: underline;
        }
        """
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Solver Benchmark - Raw Data</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <header>
        <h1>🔬 Optimization Solver Benchmark</h1>
        <p>Raw Data - Detailed Results Table</p>
        <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Results: {len(results)}</small></p>
    </header>
    
    <nav>
        <a href="index.html">Overview</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="raw_data.html" class="active">Raw Data</a>
        <a href="data/">Data Exports</a>
    </nav>
    
    <main>
        <div class="section">
            <h2>📋 Detailed Results</h2>
            <div class="section-content">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Solver</th>
                            <th>Problem</th>
                            <th>Type</th>
                            <th>Library</th>
                            <th>Status</th>
                            <th>Solve Time (s)</th>
                            <th>Objective Value</th>
                            <th>Iterations</th>
                            <th>Duality Gap</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        for result in results:
            # Format values
            solve_time = f"{result.solve_time:.4f}" if result.solve_time is not None else "—"
            objective = f"{result.primal_objective_value:.6e}" if result.primal_objective_value is not None else "—"
            iterations = str(result.iterations) if result.iterations is not None else "—"
            duality_gap = f"{result.duality_gap:.6e}" if result.duality_gap is not None else "—"
            timestamp = result.timestamp.strftime('%Y-%m-%d %H:%M:%S') if result.timestamp else "—"
            
            # Status styling
            status = result.status or "unknown"
            if status == 'optimal':
                status_class = 'status-optimal'
            elif status == 'error':
                status_class = 'status-error'
            elif status in ['infeasible', 'unbounded']:
                status_class = 'status-infeasible'
            else:
                status_class = ''
            
            html_content += f"""
            <tr>
                <td><span class="solver-name">{result.solver_name}</span></td>
                <td>{result.problem_name}</td>
                <td><span class="problem-type">{result.problem_type}</span></td>
                <td><span class="library-name">{result.problem_library}</span></td>
                <td><span class="{status_class}">{status.upper()}</span></td>
                <td class="number">{solve_time}</td>
                <td class="number">{objective}</td>
                <td class="number">{iterations}</td>
                <td class="number">{duality_gap}</td>
                <td class="timestamp">{timestamp}</td>
            </tr>"""
        
        html_content += f"""
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <footer>
        <p>
            <a href="index.html">Overview</a> |
            <a href="results_matrix.html">Results Matrix</a> |
            <a href="data/">Data Exports</a>
        </p>
        <p><small>Generated by Optimization Solver Benchmark System</small></p>
    </footer>
</body>
</html>"""
        
        # Save to file
        output_file = self.output_dir / "raw_data.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Raw data report saved to {output_file}")
        return html_content


def main():
    """Test HTML generator"""
    generator = HTMLGenerator()
    
    print("Testing HTML Generator...")
    success = generator.generate_all_reports()
    
    if success:
        print("✅ All HTML reports generated successfully!")
        print("Generated files:")
        print("  - docs/pages/index.html (Overview)")
        print("  - docs/pages/results_matrix.html (Results Matrix)")
        print("  - docs/pages/raw_data.html (Raw Data)")
    else:
        print("❌ Failed to generate HTML reports")


if __name__ == "__main__":
    main()