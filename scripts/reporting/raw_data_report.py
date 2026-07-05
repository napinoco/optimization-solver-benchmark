"""
Raw data table report generator (extracted from html_generator.py).
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


class RawDataReportGenerator(ReportGeneratorBase):
    """Generates the raw data table (raw_data.html)"""

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
        <p><small>Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")} | Total Results: {len(results)}</small></p>
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
                r.problem_library or "zzz",  # Put None/empty at end
                r.problem_type or "zzz",  # Put None/empty at end
                r.problem_name or "zzz",  # Put None/empty at end
            ),
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
            timestamp = result.timestamp.strftime("%Y-%m-%d %H:%M:%S") if result.timestamp else "—"

            # Format commit hash and environment
            commit_hash = getattr(result, "commit_hash", None) or "—"
            if commit_hash != "—" and len(commit_hash) > 8:
                commit_hash_short = commit_hash[:8]
            else:
                commit_hash_short = commit_hash

            environment_info = getattr(result, "environment_info", {})
            platform = self._get_platform_info(environment_info)

            # Status styling
            status = result.status or "unknown"
            status_lower = status.lower()
            if status_lower == "optimal":
                status_class = "status-optimal"
            elif status_lower == "optimal (inaccurate)":
                status_class = "status-optimal-inaccurate"
            elif status_lower == "unsupported":
                status_class = "status-unsupported"
            elif status_lower == "error":
                status_class = "status-error"
            elif status_lower in ["infeasible", "unbounded"]:
                status_class = "status-infeasible"
            else:
                status_class = ""

            html_content += f"""
            <tr>
                <td><span class="solver-name">{result.solver_name}</span></td>
                <td><span class="solver-version">{result.solver_version or "—"}</span></td>
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

        html_content += """
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
        with open(output_file, "w") as f:
            f.write(html_content)

        self.logger.info(f"Raw data report saved to {output_file}")
        return html_content
