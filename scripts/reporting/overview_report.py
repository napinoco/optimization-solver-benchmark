"""
Overview dashboard report generator (extracted from html_generator.py).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.report_base import ReportGeneratorBase
from scripts.reporting.result_processor import BenchmarkResult


class OverviewReportGenerator(ReportGeneratorBase):
    """Generates the overview dashboard (index.html)"""

    def _get_overview_section(self) -> str:
        """Generate overview section HTML from site config"""
        if not self.site_config or "site" not in self.site_config:
            return ""

        site_info = self.site_config.get("site", {})
        overview = site_info.get("overview", "").strip()
        author = site_info.get("author", "").strip()

        if not overview:
            return ""

        # Add author information after source code section for more natural flow
        author_html = (
            f'<p style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e9ecef; color: #6c757d;"><strong>👤 Author:</strong> {author}</p>'
            if author
            else ""
        )

        # Add author info at the end for a more natural flow
        overview_with_author = overview + author_html

        return f"""
        <div class="overview-section">
            <h2>📋 Project Overview</h2>
            <div class="overview-content">{overview_with_author}</div>
        </div>
        """

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
                env_info = results[0].get("environment_info", {})
            else:
                env_info = getattr(results[0], "environment_info", {})
        else:
            env_info = {}

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
        <p><small>Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}</small></p>
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
                <span class="stat-value">{summary["total_solvers"]}</span>
            </div>
            <div class="stat-card">
                <h3>Problems Tested</h3>
                <span class="stat-value">{summary["total_problems"]}</span>
            </div>
            <div class="stat-card">
                <h3>Libraries</h3>
                <span class="stat-value">{len(summary["library_distribution"])}</span>
            </div>
            <div class="stat-card">
                <h3>Problem Types</h3>
                <span class="stat-value">{len(summary["problem_type_distribution"])}</span>
            </div>
        </div>

        <!-- Solvers Tested Section -->
        <div class="section">
            <h2>🔧 Solvers Tested</h2>
            <div class="section-content">
                <p><strong>Total Solvers:</strong> {summary["total_solvers"]}</p>
                <div style="margin-top: 1rem;">
                    <strong>Solver Names:</strong>
                    <ul style="column-count: 2; column-gap: 2rem; margin: 1rem 0; padding-left: 1.5rem;">"""

        for solver_name in sorted(summary["solver_names"]):
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

        html_content += (
            """
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
            """
            + self._generate_environment_section(env_analysis, env_info)
            + """
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
        )

        # Save to file
        output_file = self.output_dir / "index.html"
        with open(output_file, "w") as f:
            f.write(html_content)

        self.logger.info(f"Overview report saved to {output_file}")
        return html_content
