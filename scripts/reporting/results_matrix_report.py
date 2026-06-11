"""
Results matrix report generator (extracted from html_generator.py).
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


class ResultsMatrixReportGenerator(ReportGeneratorBase):
    """Generates the problems x solvers results matrix (results_matrix.html)"""

    def _get_results_matrix_note(self) -> str:
        """Generate results matrix note HTML from site config"""
        if not self.site_config or "site" not in self.site_config:
            return ""

        site_info = self.site_config.get("site", {})
        note = site_info.get("results_matrix_note", "").strip()

        if not note:
            return ""

        return f"""
        <div class="matrix-note">
            <div class="matrix-note-content">{note}</div>
        </div>
        """

    def generate_results_matrix(self, results: List[BenchmarkResult]) -> str:
        """Generate problems × solvers results matrix"""

        self.logger.info("Generating results matrix...")

        # Get matrix data
        matrix_data = self.result_processor.get_results_matrix(results)
        problems = matrix_data["problems"]
        solvers = matrix_data["solvers"]
        matrix = matrix_data["matrix"]
        problem_metadata = matrix_data["problem_metadata"]

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
        <p><small>Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}</small></p>
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
            known_obj = metadata["known_objective_value"]
            current_library = metadata["library_name"]

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
            fastest_excellent_time = float("inf")
            fastest_excellent_solver = None
            fastest_good_time = float("inf")
            fastest_good_solver = None

            if known_obj is not None:
                for solver in solvers:
                    result = matrix[problem][solver]
                    if result and result["status"]:
                        obj_val = result["objective_value"]
                        solve_time = result["solve_time"]

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
                <td>{metadata["library_name"]}</td>
                <td class="problem-name">{problem}</td>
                <td>{metadata["problem_type"]}</td>
                <td>{known_obj_str}</td>"""

            for solver in solvers:
                result = matrix[problem][solver]
                if result is None:
                    # Empty result - use 3-row fixed layout with empty content (no record exists)
                    html_content += """<td class="status-unknown">
                        <div class="cell-content">
                            <div class="cell-status">—</div>
                            <div class="cell-solve-time"></div>
                            <div class="cell-objective"></div>
                        </div>
                    </td>"""
                else:
                    status = result["status"] or "unknown"
                    solve_time = result["solve_time"]
                    obj_val = result["objective_value"]

                    # Determine CSS class based on status
                    status_lower = status.lower()
                    if status_lower == "optimal":
                        css_class = "status-optimal"
                    elif status_lower == "optimal (inaccurate)":
                        css_class = "status-optimal-inaccurate"
                    elif status_lower == "unsupported":
                        css_class = "status-unsupported"
                    elif status_lower == "timeout":
                        css_class = "status-timeout"
                    elif status_lower == "sigkill":
                        css_class = "status-sigkill"
                    elif status_lower == "subprocess_error":
                        css_class = "status-subprocess-error"
                    elif status_lower == "error":
                        css_class = "status-error"
                    elif status_lower in ["infeasible", "unbounded"]:
                        css_class = "status-infeasible"
                    else:
                        css_class = "status-unknown"

                    # Format solve time (2nd row)
                    solve_time_str = "—"  # Default placeholder
                    solve_time_class = "cell-solve-time"
                    if (
                        solve_time is not None
                        and solve_time > 0
                        and not (isinstance(solve_time, float) and str(solve_time).lower() == "nan")
                    ):
                        solve_time_str = f"{solve_time:.3f}s"
                        # Check if this is the fastest among excellent accuracy
                        if solver == fastest_excellent_solver:
                            solve_time_class += " fastest-excellent"

                    # Format objective value (3rd row) with accuracy-based coloring
                    obj_val_str = "—"  # Default placeholder
                    accuracy_class = "accuracy-unknown"
                    if obj_val is not None and not (isinstance(obj_val, float) and str(obj_val).lower() == "nan"):
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
        with open(output_file, "w") as f:
            f.write(html_content)

        self.logger.info(f"Results matrix saved to {output_file}")
        return html_content
