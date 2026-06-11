"""
Shared base class for HTML report generators.

Holds the output directory, result processor, site configuration, and the
HTML/CSS helpers common to all report types. The concrete generators live in
overview_report.py, results_matrix_report.py, raw_data_report.py, and
data_index_report.py; HTMLGenerator (html_generator.py) remains the facade.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.result_processor import BenchmarkResult, ResultProcessor
from scripts.utils.logger import get_logger


class ReportGeneratorBase:
    """Base class providing shared state and helpers for report generators"""

    def __init__(self, output_dir: Path, result_processor: ResultProcessor, site_config: Dict[str, Any]):
        self.output_dir = output_dir
        self.result_processor = result_processor
        self.site_config = site_config
        self.logger = get_logger("html_generator")

    def _analyze_multiple_environments(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze commit hashes and environments from all results"""
        commit_hashes = set()
        environments = set()

        for result in results:
            # Collect commit hashes
            if hasattr(result, "commit_hash") and result.commit_hash:
                commit_hashes.add(result.commit_hash)

            # Collect environment platforms
            env_info = getattr(result, "environment_info", {})
            platform = self._get_platform_info(env_info)
            if platform != "Unknown":
                environments.add(platform)

        return {
            "commit_hashes": sorted(list(commit_hashes)),
            "environments": sorted(list(environments)) if environments else ["Unknown"],
        }

    def _generate_environment_section(self, env_analysis: Dict[str, Any], env_info: Dict[str, Any]) -> str:
        """Generate environment information section HTML"""
        commit_hashes = env_analysis["commit_hashes"]
        environments = env_analysis["environments"]

        # Generate commit hash display
        if len(commit_hashes) == 1:
            commit_display = f"<p><strong>Git Commit Hash:</strong> <code>{commit_hashes[0][:8]}</code></p>"
        elif len(commit_hashes) > 1:
            commit_list = ", ".join([f"<code>{ch[:8]}</code>" for ch in commit_hashes])
            commit_display = f"<p><strong>Git Commit Hashes:</strong> {commit_list}</p>"
            commit_display += f"<p><em>⚠️ Multiple environments detected: Results from {len(commit_hashes)} different Git commits</em></p>"
        else:
            commit_display = "<p><strong>Git Commit Hash:</strong> Unknown</p>"

        # Generate environment display
        if len(environments) == 1:
            env_display = f"<p><strong>Platform:</strong> {environments[0]}</p>"
        elif len(environments) > 1:
            env_list = ", ".join(environments)
            env_display = f"<p><strong>Platforms:</strong> {env_list}</p>"
            env_display += f"<p><em>⚠️ Multiple platforms detected: Results from {len(environments)} different environments</em></p>"
        else:
            env_display = "<p><strong>Platform:</strong> Unknown</p>"

        # Python version (from latest result)
        python_info = env_info.get("python", {})
        python_version = python_info.get("version", "Unknown")
        python_implementation = python_info.get("implementation", "Unknown")
        if python_implementation != "Unknown" and python_implementation != python_version:
            python_display = f"<p><strong>Python Version:</strong> {python_implementation} {python_version}</p>"
        else:
            python_display = f"<p><strong>Python Version:</strong> {python_version}</p>"

        # Operating System details
        os_info = env_info.get("os", {})
        os_system = os_info.get("system", "Unknown")
        os_release = os_info.get("release", "Unknown")
        if os_release != "Unknown":
            os_display = f"<p><strong>Operating System:</strong> {os_system} {os_release}</p>"
        else:
            os_display = f"<p><strong>Operating System:</strong> {os_system}</p>"

        # CPU information
        cpu_info = env_info.get("cpu", {})
        cpu_count = cpu_info.get("cpu_count", "Unknown")
        processor = cpu_info.get("processor", "Unknown")
        if processor != "Unknown" and cpu_count != "Unknown":
            cpu_display = f"<p><strong>CPU:</strong> {processor} ({cpu_count} cores)</p>"
        elif cpu_count != "Unknown":
            cpu_display = f"<p><strong>CPU Cores:</strong> {cpu_count}</p>"
        else:
            cpu_display = f"<p><strong>CPU:</strong> {processor}</p>"

        # Memory information
        memory_info = env_info.get("memory", {})
        memory_gb = memory_info.get("total_gb", "Unknown")
        if memory_gb != "Unknown":
            memory_display = f"<p><strong>Memory:</strong> {memory_gb:.1f} GB</p>"
        else:
            memory_display = "<p><strong>Memory:</strong> Unknown</p>"

        # Note about MATLAB (since we can't easily detect versions from environment)
        matlab_note = "<p><strong>MATLAB:</strong> Available (version detection via solver results)</p>"

        return commit_display + env_display + python_display + os_display + cpu_display + memory_display + matlab_note

    def _get_platform_info(self, environment_info: Dict[str, Any]) -> str:
        """Extract platform information including CPU and memory details"""
        if not isinstance(environment_info, dict):
            return "Unknown"

        os_info = environment_info.get("os", {})
        cpu_info = environment_info.get("cpu", {})
        memory_info = environment_info.get("memory", {})

        platform_base = os_info.get("system", "Unknown")
        cpu_count = cpu_info.get("cpu_count", "Unknown")
        memory_gb = memory_info.get("total_gb", "Unknown")

        if platform_base != "Unknown" and cpu_count != "Unknown" and memory_gb != "Unknown":
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
