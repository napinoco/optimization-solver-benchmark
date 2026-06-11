"""
Simplified HTML Generator for Three Report Types
===============================================

Generates three focused HTML reports as specified in the re-architected design:
1. Overview Dashboard - Summary statistics and solver/problem counts
2. Results Matrix - Problems × solvers matrix with solve times and status
3. Raw Data - Detailed table with all result fields

Simple HTML structure without complex Bootstrap dashboards.

HTMLGenerator is a facade: the actual report generation lives in
overview_report.py, results_matrix_report.py, raw_data_report.py, and
data_index_report.py, which share helpers via report_base.py.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.data_index_report import DataIndexReportGenerator
from scripts.reporting.overview_report import OverviewReportGenerator
from scripts.reporting.raw_data_report import RawDataReportGenerator
from scripts.reporting.result_processor import BenchmarkResult, ResultProcessor
from scripts.reporting.results_matrix_report import ResultsMatrixReportGenerator
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

        # Report generators (one per report type, sharing the same state)
        generator_args = (self.output_dir, self.result_processor, self.site_config)
        self._overview_generator = OverviewReportGenerator(*generator_args)
        self._results_matrix_generator = ResultsMatrixReportGenerator(*generator_args)
        self._raw_data_generator = RawDataReportGenerator(*generator_args)
        self._data_index_generator = DataIndexReportGenerator(*generator_args)

    def _load_site_config(self) -> Dict[str, Any]:
        """Load site configuration from config/site_config.yaml"""
        try:
            config_path = project_root / "config" / "site_config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load site config: {e}")
            return {}

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
        return self._overview_generator.generate_overview(results)

    def generate_results_matrix(self, results: List[BenchmarkResult]) -> str:
        """Generate results matrix showing problems × solvers"""
        return self._results_matrix_generator.generate_results_matrix(results)

    def generate_raw_data(self, results: List[BenchmarkResult]) -> str:
        """Generate raw data table with all result fields"""
        return self._raw_data_generator.generate_raw_data(results)

    def generate_data_index(self) -> str:
        """Generate the data export index page"""
        return self._data_index_generator.generate_data_index()


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
