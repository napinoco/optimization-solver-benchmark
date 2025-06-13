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
            self._generate_environment_info_page(metadata)
            
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