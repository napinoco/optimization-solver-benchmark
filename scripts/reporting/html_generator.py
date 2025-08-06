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
import yaml
from datetime import datetime, timezone

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
        
        # Load site configuration
        self.site_config = self._load_site_config()
    
    def _load_site_config(self) -> Dict[str, Any]:
        """Load site configuration from config/site_config.yaml"""
        try:
            config_path = project_root / "config" / "site_config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load site config: {e}")
            return {}
    
    def _get_overview_section(self) -> str:
        """Generate overview section HTML from site config"""
        if not self.site_config or 'site' not in self.site_config:
            return ""
        
        site_info = self.site_config.get('site', {})
        overview = site_info.get('overview', '').strip()
        author = site_info.get('author', '').strip()
        
        if not overview:
            return ""
        
        # Add author information after source code section for more natural flow
        author_html = f'<p style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e9ecef; color: #6c757d;"><strong>👤 Author:</strong> {author}</p>' if author else ""
        
        # Add author info at the end for a more natural flow
        overview_with_author = overview + author_html
        
        return f"""
        <div class="overview-section">
            <h2>📋 Project Overview</h2>
            <div class="overview-content">{overview_with_author}</div>
        </div>
        """
    
    def _get_results_matrix_note(self) -> str:
        """Generate results matrix note HTML from site config"""
        if not self.site_config or 'site' not in self.site_config:
            return ""
        
        site_info = self.site_config.get('site', {})
        note = site_info.get('results_matrix_note', '').strip()
        
        if not note:
            return ""
        
        return f"""
        <div class="matrix-note">
            <div class="matrix-note-content">{note}</div>
        </div>
        """
    
    def _analyze_multiple_environments(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze commit hashes and environments from all results"""
        commit_hashes = set()
        environments = set()
        
        for result in results:
            # Collect commit hashes
            if hasattr(result, 'commit_hash') and result.commit_hash:
                commit_hashes.add(result.commit_hash)
            
            # Collect environment platforms
            env_info = getattr(result, 'environment_info', {})
            platform = self._get_platform_info(env_info)
            if platform != 'Unknown':
                environments.add(platform)
        
        return {
            'commit_hashes': sorted(list(commit_hashes)),
            'environments': sorted(list(environments)) if environments else ['Unknown']
        }
    
    def _generate_environment_section(self, env_analysis: Dict[str, Any], env_info: Dict[str, Any]) -> str:
        """Generate environment information section HTML"""
        commit_hashes = env_analysis['commit_hashes']
        environments = env_analysis['environments']
        
        # Generate commit hash display
        if len(commit_hashes) == 1:
            commit_display = f"<p><strong>Git Commit Hash:</strong> <code>{commit_hashes[0][:8]}</code></p>"
        elif len(commit_hashes) > 1:
            commit_list = ', '.join([f"<code>{ch[:8]}</code>" for ch in commit_hashes])
            commit_display = f"<p><strong>Git Commit Hashes:</strong> {commit_list}</p>"
            commit_display += f"<p><em>⚠️ Multiple environments detected: Results from {len(commit_hashes)} different Git commits</em></p>"
        else:
            commit_display = "<p><strong>Git Commit Hash:</strong> Unknown</p>"
        
        # Generate environment display
        if len(environments) == 1:
            env_display = f"<p><strong>Platform:</strong> {environments[0]}</p>"
        elif len(environments) > 1:
            env_list = ', '.join(environments)
            env_display = f"<p><strong>Platforms:</strong> {env_list}</p>"
            env_display += f"<p><em>⚠️ Multiple platforms detected: Results from {len(environments)} different environments</em></p>"
        else:
            env_display = "<p><strong>Platform:</strong> Unknown</p>"
        
        # Python version (from latest result)
        python_info = env_info.get('python', {})
        python_version = python_info.get('version', 'Unknown')
        python_implementation = python_info.get('implementation', 'Unknown')
        if python_implementation != 'Unknown' and python_implementation != python_version:
            python_display = f"<p><strong>Python Version:</strong> {python_implementation} {python_version}</p>"
        else:
            python_display = f"<p><strong>Python Version:</strong> {python_version}</p>"
        
        # Operating System details
        os_info = env_info.get('os', {})
        os_system = os_info.get('system', 'Unknown')
        os_release = os_info.get('release', 'Unknown')
        if os_release != 'Unknown':
            os_display = f"<p><strong>Operating System:</strong> {os_system} {os_release}</p>"
        else:
            os_display = f"<p><strong>Operating System:</strong> {os_system}</p>"
        
        # CPU information
        cpu_info = env_info.get('cpu', {})
        cpu_count = cpu_info.get('cpu_count', 'Unknown')
        processor = cpu_info.get('processor', 'Unknown')
        if processor != 'Unknown' and cpu_count != 'Unknown':
            cpu_display = f"<p><strong>CPU:</strong> {processor} ({cpu_count} cores)</p>"
        elif cpu_count != 'Unknown':
            cpu_display = f"<p><strong>CPU Cores:</strong> {cpu_count}</p>"
        else:
            cpu_display = f"<p><strong>CPU:</strong> {processor}</p>"
        
        # Memory information
        memory_info = env_info.get('memory', {})
        memory_gb = memory_info.get('total_gb', 'Unknown')
        if memory_gb != 'Unknown':
            memory_display = f"<p><strong>Memory:</strong> {memory_gb:.1f} GB</p>"
        else:
            memory_display = "<p><strong>Memory:</strong> Unknown</p>"
        
        # Note about MATLAB (since we can't easily detect versions from environment)
        matlab_note = "<p><strong>MATLAB:</strong> Available (version detection via solver results)</p>"
        
        return commit_display + env_display + python_display + os_display + cpu_display + memory_display + matlab_note
    
    def _get_platform_info(self, environment_info: Dict[str, Any]) -> str:
        """Extract platform information including CPU and memory details"""
        if not isinstance(environment_info, dict):
            return 'Unknown'
            
        os_info = environment_info.get('os', {})
        cpu_info = environment_info.get('cpu', {})
        memory_info = environment_info.get('memory', {})
        
        platform_base = os_info.get('system', 'Unknown')
        cpu_count = cpu_info.get('cpu_count', 'Unknown')
        memory_gb = memory_info.get('total_gb', 'Unknown')
        
        if platform_base != 'Unknown' and cpu_count != 'Unknown' and memory_gb != 'Unknown':
            return f"{platform_base} ({cpu_count}CPU, {memory_gb:.0f}GB)"
        else:
            return platform_base
    
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
            
            # Generate all four reports
            self.generate_overview(results)
            self.generate_results_matrix(results)
            self.generate_raw_data(results)
            self.generate_data_index()
            
            self.logger.info("All simplified HTML reports generated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML reports: {e}")
            return False
    
    def generate_overview(self, results: List[BenchmarkResult]) -> str:
        """Generate overview dashboard showing performance-focused statistics"""
        
        self.logger.info("Generating overview dashboard...")
        
        # Get summary statistics
        summary = self.result_processor.get_summary_statistics(results)
        problem_counts = self.result_processor.get_problem_count_by_library_and_type(results)
        
        # Analyze multiple environments and commit hashes
        env_analysis = self._analyze_multiple_environments(results)
        
        # Generate environment info from latest result for fallback
        if results:
            # Handle both dict and object result formats
            if isinstance(results[0], dict):
                env_info = results[0].get('environment_info', {})
                latest_commit_hash = results[0].get('commit_hash', 'unknown')
            else:
                env_info = getattr(results[0], 'environment_info', {})
                latest_commit_hash = getattr(results[0], 'commit_hash', 'unknown')
        else:
            env_info = {}
            latest_commit_hash = "unknown"
        
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
        
        .overview-section {{
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        
        .overview-section h2 {{
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }}
        
        .overview-content {{
            color: #34495e;
            line-height: 1.8;
            white-space: pre-line;
        }}
        
        .overview-content strong {{
            color: #2c3e50;
        }}
        
        .status-distribution {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            padding: 1rem;
        }}
        
        .status-card {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        
        .status-card span {{
            display: block;
            margin-bottom: 0.5rem;
        }}
        
        .status-stats {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .status-count {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .status-percentage {{
            font-size: 0.9rem;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🔬 Optimization Solver Benchmark</h1>
        <p>Overview Dashboard - Latest Results</p>
        <p><small>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</small></p>
    </header>
    
    <nav>
        <a href="index.html" class="active">Overview</a>
        <a href="results_matrix.html">Results Matrix</a>
        <a href="raw_data.html">Raw Data</a>
        <a href="data/">Data Exports</a>
    </nav>
    
    <main>
        
        <!-- Project Overview Section -->
        {self._get_overview_section()}

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Solvers Tested</h3>
                <span class="stat-value">{summary['total_solvers']}</span>
            </div>
            <div class="stat-card">
                <h3>Problems Tested</h3>
                <span class="stat-value">{summary['total_problems']}</span>
            </div>
            <div class="stat-card">
                <h3>Libraries</h3>
                <span class="stat-value">{len(summary['library_distribution'])}</span>
            </div>
            <div class="stat-card">
                <h3>Problem Types</h3>
                <span class="stat-value">{len(summary['problem_type_distribution'])}</span>
            </div>
        </div>
        
        <!-- Solvers Tested Section -->
        <div class="section">
            <h2>🔧 Solvers Tested</h2>
            <div class="section-content">
                <p><strong>Total Solvers:</strong> {summary['total_solvers']}</p>
                <div style="margin-top: 1rem;">
                    <strong>Solver Names:</strong>
                    <ul style="column-count: 2; column-gap: 2rem; margin: 1rem 0; padding-left: 1.5rem;">"""
        
        for solver_name in sorted(summary['solver_names']):
            html_content += f"<li>{solver_name}</li>"
        
        html_content += """
                    </ul>
                </div>
            </div>
        </div>

        <!-- Problems Tested Section -->
        <div class="section">
            <h2>📚 Problems Tested by Library and Type</h2>
            <div class="section-content">
                <table>
                    <thead>
                        <tr>
                            <th>Library</th>
                            <th>Problem Type</th>
                            <th>Count</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        # Generate problem count table
        for library in sorted(problem_counts.keys()):
            for problem_type in sorted(problem_counts[library].keys()):
                count = problem_counts[library][problem_type]
                html_content += f"""
                        <tr>
                            <td><strong>{library}</strong></td>
                            <td>{problem_type}</td>
                            <td>{count}</td>
                        </tr>"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Performance Analysis Section -->
        <div class="section">
            <h2>📊 Performance Analysis</h2>
            <div class="section-content">
                <div style="text-align: center; padding: 2rem; color: #7f8c8d;">
                    <h3 style="color: #95a5a6; margin-bottom: 1rem;">🚧 To Be Determined</h3>
                    <p style="font-size: 1.1rem; line-height: 1.6;">
                        Performance ranking methodology is currently under development.<br>
                        Evaluation criteria for determining "best performers" are being established<br>
                        to ensure fair and meaningful comparisons across different solver types.
                    </p>
                    <p style="margin-top: 1rem; font-style: italic;">
                        Please refer to the <a href="results_matrix.html" style="color: #3498db;">Results Matrix</a> 
                        for detailed performance data in the meantime.
                    </p>
                </div>
            </div>
        </div>

        <div class="metadata">
            <h3>🔧 Environment Information</h3>
            """ + self._generate_environment_section(env_analysis, env_info) + """
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
        problem_metadata = matrix_data['problem_metadata']
        
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
        
        .table-container {
            overflow-x: auto;
            margin: 1rem 0;
            border: 1px solid #ecf0f1;
            border-radius: 6px;
            max-height: 90vh;
            overflow-y: auto;
        }
        
        .matrix-table {
            width: 100%;
            border-collapse: collapse;
            margin: 0;
            font-size: 0.85em;
        }
        
        .matrix-table th, .matrix-table td {
            padding: 4px 6px;
            text-align: center;
            border: 1px solid #ecf0f1;
            line-height: 1.2;
            vertical-align: top;
        }
        
        .matrix-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            position: sticky;
            top: 0;
            z-index: 10;
            box-shadow: 0 2px 2px -1px rgba(0, 0, 0, 0.4);
        }
        
        .matrix-table .problem-name {
            text-align: left;
            font-weight: bold;
            background-color: #f8f9fa;
            color: #2c3e50;
        }
        
        /* Status classes - no background colors, keep for structure */
        .status-optimal,
        .status-optimal-inaccurate,
        .status-error,
        .status-infeasible,
        .status-unsupported,
        .status-unknown {
            /* No background colors - focus on accuracy/speed instead */
        }
        
        /* Library boundary styles */
        .library-boundary {
            border-top: 3px solid #34495e !important;
        }
        
        /* Gray styling for non-solution statuses */
        .status-timeout,
        .status-sigkill,
        .status-subprocess-error,
        .status-unsupported {
            color: #999 !important;
        }
        
        .status-timeout .cell-status,
        .status-timeout .cell-solve-time,
        .status-timeout .cell-objective,
        .status-sigkill .cell-status,
        .status-sigkill .cell-solve-time,
        .status-sigkill .cell-objective,
        .status-subprocess-error .cell-status,
        .status-subprocess-error .cell-solve-time,
        .status-subprocess-error .cell-objective,
        .status-unsupported .cell-status,
        .status-unsupported .cell-solve-time,
        .status-unsupported .cell-objective {
            color: #999 !important;
        }
        
        /* Cell layout for fixed 3-row structure */
        .cell-content {
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-height: 42px;
        }
        
        .cell-status {
            font-size: 0.75em;
            font-weight: bold;
            line-height: 1.1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
        }
        
        .cell-solve-time {
            font-size: 0.8em;
            color: #7f8c8d;
            font-family: 'Courier New', monospace;
            line-height: 1.1;
        }
        
        .cell-objective {
            font-size: 0.8em;
            font-family: 'Courier New', monospace;
            font-weight: 500;
            line-height: 1.1;
        }
        
        /* Accuracy-based coloring for objective values */
        .accuracy-excellent {
            color: #28a745;  /* Green - excellent accuracy */
        }
        
        .accuracy-good {
            color: #ffc107;  /* Amber - good accuracy */
        }
        
        .accuracy-poor {
            color: #dc3545;  /* Red - poor accuracy */
        }
        
        .accuracy-unknown {
            color: #6c757d;  /* Gray - unknown accuracy */
        }
        
        /* Fastest time among excellent accuracy */
        .fastest-excellent {
            font-weight: bold;
            color: #2563eb;  /* Blue - emphasize fastest with good accuracy */
        }
        
        .matrix-note {
            margin: 2rem 0;
            padding: 1.5rem;
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            border-left: 4px solid #f39c12;
        }
        
        .matrix-note-content {
            color: #856404;
            line-height: 1.6;
            white-space: pre-line;
        }
        
        .matrix-note-content strong {
            color: #6c5119;
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
        <p><small>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</small></p>
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
                
                <!-- Legend for accuracy and performance indicators -->
                <div class="legend">
                    <h4>📖 Legend</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                        <div>
                            <strong>Objective Value Accuracy:</strong>
                            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                                <li><span class="accuracy-excellent">Excellent</span> - Error &lt; 1e-4 (0.01%) (Green)</li>
                                <li><span class="accuracy-good">Good</span> - Error 1e-4 to 1e-2 (0.01% to 1%) (Amber)</li>
                                <li><span class="accuracy-poor">Poor</span> - Error &gt; 1e-2 (1%) (Red)</li>
                            </ul>
                        </div>
                        <div>
                            <strong>Performance Indicators:</strong>
                            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                                <li><span class="fastest-excellent">Fastest Time</span> - Best solve time among results with excellent accuracy; fallback to good accuracy if no excellent results exist (Blue)</li>
                                <li><strong>Bold</strong> - Emphasizes best performing results</li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="table-container">
                    <table class="matrix-table">
                    <thead>
                        <tr>
                            <th>Library</th>
                            <th>Problem</th>
                            <th>Type</th>
                            <th>Known Objective</th>"""
        
        for solver in solvers:
            html_content += f"<th>{solver}</th>"
        
        html_content += """
                        </tr>
                    </thead>
                    <tbody>"""
        
        prev_library = None
        for problem in problems:
            metadata = problem_metadata[problem]
            known_obj = metadata['known_objective_value']
            current_library = metadata['library_name']
            
            # Check if this is the same library as the previous row for grouping
            is_same_library = prev_library == current_library
            
            # Handle known objective value formatting - use scientific notation with 5 decimal places
            if known_obj is None:
                known_obj_str = "—"
            else:
                try:
                    # Convert to float and format as scientific notation with 5 decimal places
                    obj_float = float(known_obj)
                    known_obj_str = f"{obj_float:.5e}"
                except (ValueError, TypeError):
                    # If conversion fails, use the string as-is
                    known_obj_str = str(known_obj)
            
            # Check if this is a library boundary
            row_class = "library-boundary" if not is_same_library and prev_library is not None else ""
            
            # Find fastest time among excellent accuracy solvers for this problem
            # If no excellent accuracy solvers exist, fallback to good accuracy solvers
            fastest_excellent_time = float('inf')
            fastest_excellent_solver = None
            fastest_good_time = float('inf')
            fastest_good_solver = None
            
            if known_obj is not None:
                for solver in solvers:
                    result = matrix[problem][solver]
                    if result and result['status']:
                        obj_val = result['objective_value']
                        solve_time = result['solve_time']
                        
                        # Only consider results with valid objective value and solve time
                        if obj_val is not None and solve_time is not None and solve_time > 0:
                            try:
                                obj_float = float(obj_val)
                                known_float = float(known_obj)
                                
                                # Calculate relative error
                                if known_float != 0:
                                    rel_error = abs(obj_float - known_float) / abs(known_float)
                                else:
                                    rel_error = abs(obj_float)
                                
                                # Check for excellent accuracy
                                if rel_error < 1e-4 and solve_time < fastest_excellent_time:
                                    fastest_excellent_time = solve_time
                                    fastest_excellent_solver = solver
                                # Check for good accuracy (fallback)
                                elif rel_error < 1e-2 and solve_time < fastest_good_time:
                                    fastest_good_time = solve_time
                                    fastest_good_solver = solver
                            except (ValueError, TypeError):
                                pass
                
                # Use excellent if available, otherwise fallback to good
                if fastest_excellent_solver is None and fastest_good_solver is not None:
                    fastest_excellent_solver = fastest_good_solver
            
            html_content += f"""
            <tr class="{row_class}">
                <td>{metadata['library_name']}</td>
                <td class="problem-name">{problem}</td>
                <td>{metadata['problem_type']}</td>
                <td>{known_obj_str}</td>"""
            
            for solver in solvers:
                result = matrix[problem][solver]
                if result is None:
                    # Empty result - use 3-row fixed layout with empty content (no record exists)
                    html_content += '''<td class="status-unknown">
                        <div class="cell-content">
                            <div class="cell-status">—</div>
                            <div class="cell-solve-time"></div>
                            <div class="cell-objective"></div>
                        </div>
                    </td>'''
                else:
                    status = result['status'] or 'unknown'
                    solve_time = result['solve_time']
                    obj_val = result['objective_value']
                    
                    # Determine CSS class based on status
                    status_lower = status.lower()
                    if status_lower == 'optimal':
                        css_class = 'status-optimal'
                    elif status_lower == 'optimal (inaccurate)':
                        css_class = 'status-optimal-inaccurate'
                    elif status_lower == 'unsupported':
                        css_class = 'status-unsupported'
                    elif status_lower == 'timeout':
                        css_class = 'status-timeout'
                    elif status_lower == 'sigkill':
                        css_class = 'status-sigkill'
                    elif status_lower == 'subprocess_error':
                        css_class = 'status-subprocess-error'
                    elif status_lower == 'error':
                        css_class = 'status-error'
                    elif status_lower in ['infeasible', 'unbounded']:
                        css_class = 'status-infeasible'
                    else:
                        css_class = 'status-unknown'
                    
                    # Format solve time (2nd row)
                    solve_time_str = "—"  # Default placeholder
                    solve_time_class = "cell-solve-time"
                    if solve_time is not None and solve_time > 0 and not (isinstance(solve_time, float) and str(solve_time).lower() == 'nan'):
                        solve_time_str = f"{solve_time:.3f}s"
                        # Check if this is the fastest among excellent accuracy
                        if solver == fastest_excellent_solver:
                            solve_time_class += " fastest-excellent"
                    
                    # Format objective value (3rd row) with accuracy-based coloring
                    obj_val_str = "—"  # Default placeholder
                    accuracy_class = "accuracy-unknown"
                    if obj_val is not None and not (isinstance(obj_val, float) and str(obj_val).lower() == 'nan'):
                        try:
                            obj_float = float(obj_val)
                            obj_val_str = f"{obj_float:.5e}"
                            
                            # Calculate accuracy compared to known objective
                            if known_obj is not None:
                                try:
                                    known_float = float(known_obj)
                                    if known_float != 0:
                                        rel_error = abs(obj_float - known_float) / abs(known_float)
                                        if rel_error < 1e-4:
                                            accuracy_class = "accuracy-excellent"
                                        elif rel_error < 1e-2:
                                            accuracy_class = "accuracy-good"
                                        else:
                                            accuracy_class = "accuracy-poor"
                                    else:
                                        # known_obj is 0, use absolute error
                                        abs_error = abs(obj_float)
                                        if abs_error < 1e-4:
                                            accuracy_class = "accuracy-excellent"
                                        elif abs_error < 1e-2:
                                            accuracy_class = "accuracy-good"
                                        else:
                                            accuracy_class = "accuracy-poor"
                                except (ValueError, TypeError):
                                    accuracy_class = "accuracy-unknown"
                        except (ValueError, TypeError):
                            # Keep the default placeholder "—" for invalid values
                            pass
                    
                    # Generate 3-row fixed layout cell
                    html_content += f'''<td class="{css_class}">
                        <div class="cell-content">
                            <div class="cell-status" title="{status.upper()}">{status.upper()}</div>
                            <div class="{solve_time_class}">{solve_time_str}</div>
                            <div class="cell-objective {accuracy_class}">{obj_val_str}</div>
                        </div>
                    </td>'''
            
            html_content += "</tr>"
            prev_library = current_library  # Update previous library for next iteration
        
        html_content += f"""
                    </tbody>
                </table>
                </div>
            </div>
        </div>

        {self._get_results_matrix_note()}

        <div class="legend">
            <h3>📋 Status Legend</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 0.5rem;">
                <div>
                    <p><span class="status-optimal" style="padding: 5px 10px; border-radius: 3px;">OPTIMAL</span> - Successfully solved to optimality</p>
                    <p><span class="status-error" style="padding: 5px 10px; border-radius: 3px;">ERROR</span> - Solver encountered an error</p>
                    <p><span class="status-infeasible" style="padding: 5px 10px; border-radius: 3px;">INFEASIBLE/UNBOUNDED</span> - Problem has no feasible solution</p>
                </div>
                <div>
                    <p><span class="status-unsupported" style="padding: 5px 10px; border-radius: 3px; color: #999;">UNSUPPORTED</span> - Solver does not support this problem type</p>
                    <p><span class="status-timeout" style="padding: 5px 10px; border-radius: 3px; color: #999;">TIMEOUT</span> - Execution time limit exceeded</p>
                    <p><span class="status-sigkill" style="padding: 5px 10px; border-radius: 3px; color: #999;">SIGKILL</span> - Process forcibly terminated (memory limits)</p>
                    <p><span class="status-subprocess-error" style="padding: 5px 10px; border-radius: 3px; color: #999;">SUBPROCESS_ERROR</span> - Subprocess execution error</p>
                    <p><span class="status-unknown" style="padding: 5px 10px; border-radius: 3px;">—</span> - No result available</p>
                </div>
            </div>
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
            padding: 1rem 1rem;
            margin-bottom: 1rem;
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
        
        .table-container {
            overflow-x: auto;
            margin: 1rem 0;
            border: 1px solid #ecf0f1;
            border-radius: 6px;
            max-height: 90vh;
            overflow-y: auto;
        }
        
        .data-table {
            width: 100%;
            min-width: 1400px;
            border-collapse: collapse;
            margin: 0;
            font-size: 0.8em;
        }
        
        .data-table th, .data-table td {
            padding: 3px 4px;
            text-align: left;
            border: 1px solid #ecf0f1;
            line-height: 1.1;
            vertical-align: middle;
            white-space: nowrap;
        }
        
        .data-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            position: sticky;
            top: 0;
            z-index: 10;
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
        
        .status-optimal-inaccurate {
            background-color: #fff3cd;
            color: #856404;
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
        
        .status-unsupported {
            background-color: #e7f3ff;
            color: #0c5460;
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
        
        .commit-hash {
            font-size: 0.7em;
            color: #6c757d;
            font-family: 'Courier New', monospace;
            max-width: 80px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .platform {
            font-size: 0.7em;
            color: #495057;
            font-weight: 500;
            max-width: 120px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .environment-info {
            font-size: 0.7em;
            color: #6c757d;
            font-family: 'Courier New', monospace;
            max-width: 150px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .solver-version {
            font-size: 0.7em;
            color: #495057;
            font-family: 'Courier New', monospace;
            max-width: 120px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .problem-name-cell {
            font-size: 0.8em;
            max-width: 120px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .memo-cell {
            font-size: 0.7em;
            color: #6c757d;
            max-width: 100px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
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
        <p><small>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | Total Results: {len(results)}</small></p>
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
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Solver</th>
                                <th>Version</th>
                                <th>Problem</th>
                                <th>Type</th>
                                <th>Library</th>
                                <th>Status</th>
                                <th>Solve Time (s)</th>
                                <th>Objective Value</th>
                                <th>Iterations</th>
                                <th>Duality Gap</th>
                                <th>Commit Hash</th>
                                <th>Platform</th>
                                <th>Timestamp</th>
                            </tr>
                        </thead>
                        <tbody>"""
        
        # Sort results by library_name, problem_type, problem_name
        sorted_results = sorted(
            results,
            key=lambda r: (
                r.problem_library or 'zzz',  # Put None/empty at end
                r.problem_type or 'zzz',      # Put None/empty at end
                r.problem_name or 'zzz'       # Put None/empty at end
            )
        )
        
        for result in sorted_results:
            # Format values
            solve_time = f"{result.solve_time:.4f}" if result.solve_time is not None else "—"
            
            # Handle objective value formatting
            if result.primal_objective_value is not None:
                try:
                    obj_float = float(result.primal_objective_value)
                    objective = f"{obj_float:.6e}"
                except (ValueError, TypeError):
                    objective = str(result.primal_objective_value)
            else:
                objective = "—"
            iterations = str(result.iterations) if result.iterations is not None else "—"
            
            # Handle duality gap formatting
            if result.duality_gap is not None:
                try:
                    gap_float = float(result.duality_gap)
                    duality_gap = f"{gap_float:.6e}"
                except (ValueError, TypeError):
                    duality_gap = str(result.duality_gap)
            else:
                duality_gap = "—"
            timestamp = result.timestamp.strftime('%Y-%m-%d %H:%M:%S') if result.timestamp else "—"
            
            # Format commit hash and environment
            commit_hash = getattr(result, 'commit_hash', None) or "—"
            if commit_hash != "—" and len(commit_hash) > 8:
                commit_hash_short = commit_hash[:8]
            else:
                commit_hash_short = commit_hash
                
            environment_info = getattr(result, 'environment_info', {})
            platform = self._get_platform_info(environment_info)
            
            # Status styling
            status = result.status or "unknown"
            status_lower = status.lower()
            if status_lower == 'optimal':
                status_class = 'status-optimal'
            elif status_lower == 'optimal (inaccurate)':
                status_class = 'status-optimal-inaccurate'
            elif status_lower == 'unsupported':
                status_class = 'status-unsupported'
            elif status_lower == 'error':
                status_class = 'status-error'
            elif status_lower in ['infeasible', 'unbounded']:
                status_class = 'status-infeasible'
            else:
                status_class = ''
            
            html_content += f"""
            <tr>
                <td><span class="solver-name">{result.solver_name}</span></td>
                <td><span class="solver-version">{result.solver_version or '—'}</span></td>
                <td><span class="problem-name-cell">{result.problem_name}</span></td>
                <td><span class="problem-type">{result.problem_type}</span></td>
                <td><span class="library-name">{result.problem_library}</span></td>
                <td><span class="{status_class}">{status.upper()}</span></td>
                <td class="number">{solve_time}</td>
                <td class="number">{objective}</td>
                <td class="number">{iterations}</td>
                <td class="number">{duality_gap}</td>
                <td><span class="commit-hash" title="{commit_hash}">{commit_hash_short}</span></td>
                <td><span class="platform">{platform}</span></td>
                <td class="timestamp">{timestamp}</td>
            </tr>"""
        
        html_content += f"""
                    </tbody>
                </table>
                </div>
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
    
    def generate_data_index(self) -> str:
        """Generate index.html for data directory with links to data files"""
        
        self.logger.info("Generating data index page...")
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Solver Benchmark - Data Exports</title>
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
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 1rem;
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
        
        .data-file {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: box-shadow 0.2s;
        }}
        
        .data-file:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .data-file h3 {{
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }}
        
        .data-file p {{
            color: #7f8c8d;
            margin-bottom: 1rem;
        }}
        
        .download-btn {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 0.75rem 1.5rem;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 600;
            transition: background-color 0.2s;
        }}
        
        .download-btn:hover {{
            background: #2980b9;
        }}
        
        .file-size {{
            font-size: 0.9rem;
            color: #95a5a6;
            margin-left: 1rem;
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
        <p>Data Exports - Download Benchmark Results</p>
        <p><small>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</small></p>
    </header>
    
    <nav>
        <a href="../index.html">Overview</a>
        <a href="../results_matrix.html">Results Matrix</a>
        <a href="../raw_data.html">Raw Data</a>
        <a href="index.html" class="active">Data Exports</a>
    </nav>
    
    <main>
        <div class="section">
            <h2>📁 Available Data Files</h2>
            <div class="section-content">
                <p style="margin-bottom: 2rem; color: #7f8c8d;">
                    Download complete benchmark results in various formats for further analysis and research.
                </p>
                
                <div class="data-file">
                    <h3>📊 Latest Results (JSON)</h3>
                    <p>Most recent benchmark results with solver information, problem details, and performance metrics. Best for analyzing current solver performance.</p>
                    <a href="benchmark_results_latest.json" class="download-btn">📥 Download Latest JSON</a>
                    <span class="file-size">Latest run only</span>
                </div>
                
                <div class="data-file">
                    <h3>📈 Latest Results (CSV)</h3>
                    <p>Most recent benchmark results in CSV format for spreadsheet analysis. Contains the latest solver-problem combinations.</p>
                    <a href="benchmark_results_latest.csv" class="download-btn">📥 Download Latest CSV</a>
                    <span class="file-size">Latest run only</span>
                </div>
                
                <div class="data-file">
                    <h3>🗄️ All Results (JSON)</h3>
                    <p>Complete historical benchmark results including all runs. Sorted by ID for database restoration. Ideal for comprehensive analysis and backup.</p>
                    <a href="benchmark_results_all.json" class="download-btn">📥 Download All JSON</a>
                    <span class="file-size">Full database export</span>
                </div>
                
                <div class="data-file">
                    <h3>📑 All Results (CSV)</h3>
                    <p>Complete historical benchmark results in CSV format. Sorted by ID for database restoration. Perfect for time-series analysis and research.</p>
                    <a href="benchmark_results_all.csv" class="download-btn">📥 Download All CSV</a>
                    <span class="file-size">Full database export</span>
                </div>
                
            </div>
        </div>
        
        <div class="section">
            <h2>📖 Data Format Documentation</h2>
            <div class="section-content">
                <h3>JSON Structure</h3>
                <ul style="margin: 1rem 0; color: #34495e; padding-left: 1.5rem;">
                    <li><strong>benchmark_results_latest.json</strong>: Latest benchmark run with metadata, summary statistics, and results array</li>
                    <li><strong>benchmark_results_all.json</strong>: Complete database export sorted by ID for restoration purposes</li>
                    <li>Each contains: metadata, summary statistics, solver comparison, and detailed results</li>
                </ul>
                
                <h3>CSV Format</h3>
                <ul style="margin: 1rem 0; color: #34495e; padding-left: 1.5rem;">
                    <li><strong>benchmark_results_latest.csv</strong>: Latest run in tabular format, sorted by problem and solver</li>
                    <li><strong>benchmark_results_all.csv</strong>: All historical data sorted by ID (ascending) for database restoration</li>
                    <li>Headers include: id, solver_name, solver_version, problem_library, problem_name, problem_type, solve_time, status, etc.</li>
                </ul>
            </div>
        </div>
    </main>

    <footer>
        <p>
            <a href="../index.html">Overview</a> |
            <a href="../results_matrix.html">Results Matrix</a> |
            <a href="../raw_data.html">Raw Data</a>
        </p>
        <p><small>Generated by Optimization Solver Benchmark System</small></p>
    </footer>
</body>
</html>"""
        
        # Save to data directory
        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_file = data_dir / "index.html"
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Data index page saved to {output_file}")
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