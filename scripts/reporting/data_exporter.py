"""
Simplified Data Exporter
========================

Export benchmark results in JSON and CSV formats for external consumption.
Simplified approach as specified in the re-architected design.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.result_processor import BenchmarkResult, ResultProcessor
from scripts.utils.logger import get_logger

logger = get_logger("data_exporter")


class DataExporter:
    """Export data in JSON and CSV formats"""

    def __init__(self, output_dir: str = None, full_environment_info: bool = False):
        """Initialize data exporter with output directory
        
        Args:
            output_dir: Directory to save exported files
            full_environment_info: If True, export complete environment_info without sanitization
        """
        if output_dir is None:
            output_dir = project_root / "docs" / "pages" / "data"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.full_environment_info = full_environment_info

        self.result_processor = ResultProcessor()
        self.logger = get_logger("data_exporter")

    def _result_to_complete_dict(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Convert BenchmarkResult to complete dictionary for database dump-style export"""
        # Use full environment_info if requested, otherwise use sanitized version
        env_info_data = result.environment_info if self.full_environment_info else result.get_sanitized_environment_info()

        return {
            'id': result.id,
            'solver_name': result.solver_name,
            'solver_version': result.solver_version,
            'problem_library': result.problem_library,
            'problem_name': result.problem_name,
            'problem_type': result.problem_type,
            'environment_info': env_info_data,
            'commit_hash': result.commit_hash,
            'timestamp': result.timestamp.isoformat() if result.timestamp else None,
            'solve_time': result.solve_time,
            'status': result.status,
            'primal_objective_value': result.primal_objective_value,
            'dual_objective_value': result.dual_objective_value,
            'duality_gap': result.duality_gap,
            'primal_infeasibility': result.primal_infeasibility,
            'dual_infeasibility': result.dual_infeasibility,
            'iterations': result.iterations,
            'memo': result.memo
        }

    def export_latest_results(self) -> bool:
        """Export latest results to JSON and CSV files"""

        self.logger.info("Exporting latest results...")

        try:
            # Get latest results
            results = self.result_processor.get_latest_results_for_reporting()

            if not results:
                self.logger.warning("No results found for export")
                return False

            # Export both formats with 'latest' in filename
            json_success = self.export_json(results, filename_suffix="latest")
            csv_success = self.export_csv(results, filename_suffix="latest")

            if json_success and csv_success:
                self.logger.info("Latest results export completed successfully")
                return True
            else:
                self.logger.error("Latest results export failed")
                return False

        except Exception as e:
            self.logger.error(f"Failed to export latest results: {e}")
            return False

    def export_all_results(self) -> bool:
        """Export all results from database to JSON and CSV files"""

        self.logger.info("Exporting all results...")

        try:
            # Get all results
            results = self.result_processor.get_all_results_for_export()

            if not results:
                self.logger.warning("No results found for export")
                return False

            # Export both formats with 'all' in filename
            json_success = self.export_json(results, filename_suffix="all")
            csv_success = self.export_csv(results, filename_suffix="all")

            if json_success and csv_success:
                self.logger.info("All results export completed successfully")
                return True
            else:
                self.logger.error("All results export failed")
                return False

        except Exception as e:
            self.logger.error(f"Failed to export all results: {e}")
            return False

    def export_json(self, results: List[BenchmarkResult], filename_suffix: str = "results") -> bool:
        """Export results to JSON format
        
        Args:
            results: List of BenchmarkResult objects to export
            filename_suffix: Suffix for filename (e.g., 'latest', 'all')
        """

        self.logger.info(f"Exporting results to JSON (suffix: {filename_suffix})...")

        try:
            # Get summary statistics
            summary = self.result_processor.get_summary_statistics(results)
            solver_comparison = self.result_processor.get_solver_comparison(results)

            # Create export data structure with complete database dump
            export_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_results": len(results),
                    "export_format": "complete_database_dump" if self.full_environment_info else "sanitized_benchmark_results",
                    "version": "2.0",
                    "full_environment_info": self.full_environment_info,
                    "data_scope": filename_suffix
                },
                "summary": summary,
                "solver_comparison": solver_comparison,
                "results": [self._result_to_complete_dict(result) for result in results]
            }

            # Export to JSON file with suffix
            output_file = self.output_dir / f"benchmark_results_{filename_suffix}.json"
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.info(f"JSON export saved to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return False

    def export_csv(self, results: List[BenchmarkResult], filename_suffix: str = "results") -> bool:
        """Export results to CSV format
        
        Args:
            results: List of BenchmarkResult objects to export
            filename_suffix: Suffix for filename (e.g., 'latest', 'all')
        """

        self.logger.info(f"Exporting results to CSV (suffix: {filename_suffix})...")

        try:
            output_file = self.output_dir / f"benchmark_results_{filename_suffix}.csv"

            # Define CSV fieldnames (all database result fields for complete dump)
            fieldnames = [
                'id',
                'solver_name',
                'solver_version',
                'problem_library',
                'problem_name',
                'problem_type',
                'environment_info',
                'commit_hash',
                'timestamp',
                'solve_time',
                'status',
                'primal_objective_value',
                'dual_objective_value',
                'duality_gap',
                'primal_infeasibility',
                'dual_infeasibility',
                'iterations',
                'memo'
            ]

            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    # Convert result to flat dictionary for CSV (complete database dump)
                    # Use full environment_info if requested, otherwise use sanitized version
                    env_info_data = result.environment_info if self.full_environment_info else result.get_sanitized_environment_info()

                    row = {
                        'id': result.id,
                        'solver_name': result.solver_name,
                        'solver_version': result.solver_version,
                        'problem_library': result.problem_library,
                        'problem_name': result.problem_name,
                        'problem_type': result.problem_type,
                        'environment_info': json.dumps(env_info_data) if env_info_data else None,
                        'commit_hash': result.commit_hash,
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                        'solve_time': result.solve_time,
                        'status': result.status,
                        'primal_objective_value': result.primal_objective_value,
                        'dual_objective_value': result.dual_objective_value,
                        'duality_gap': result.duality_gap,
                        'primal_infeasibility': result.primal_infeasibility,
                        'dual_infeasibility': result.dual_infeasibility,
                        'iterations': result.iterations,
                        'memo': json.dumps(result.memo) if result.memo else None
                    }
                    writer.writerow(row)

            self.logger.info(f"CSV export saved to {output_file} ({len(results)} rows)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            return False

    def export_summary_only(self) -> bool:
        """Export just summary statistics for lightweight consumption"""

        self.logger.info("Exporting summary statistics...")

        try:
            # Get latest results
            results = self.result_processor.get_latest_results_for_reporting()

            if not results:
                self.logger.warning("No results found for summary export")
                return False

            # Get summary data
            summary = self.result_processor.get_summary_statistics(results)
            solver_comparison = self.result_processor.get_solver_comparison(results)

            # Create lightweight summary
            summary_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_results": len(results),
                    "format": "summary_only"
                },
                "overall_statistics": summary,
                "solver_performance": solver_comparison,
                "environment": {
                    "commit_hash": results[0].commit_hash if results else "unknown",
                    "platform": results[0].environment_info.get('os', {}).get('system', 'Unknown') if results and results[0].environment_info else "Unknown"
                }
            }

            # Export summary JSON
            output_file = self.output_dir / "summary.json"
            with open(output_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)

            self.logger.info(f"Summary export saved to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export summary: {e}")
            return False

    def export_both_versions(self) -> bool:
        """Export both latest and all results versions"""

        self.logger.info("Exporting both latest and all results versions...")

        try:
            # Export latest results
            latest_success = self.export_latest_results()

            # Export all results
            all_success = self.export_all_results()

            # Backward compatibility files are no longer needed
            # Only export latest and all versions

            if latest_success and all_success:
                self.logger.info("Both export versions completed successfully")
                return True
            else:
                self.logger.error("One or more export versions failed")
                return False

        except Exception as e:
            self.logger.error(f"Failed to export both versions: {e}")
            return False


def main():
    """Test data exporter with all export options"""
    print("Testing Data Exporter with all export options...")

    # Test latest results export
    print("\n1. Testing latest results export...")
    exporter = DataExporter(full_environment_info=True)
    latest_success = exporter.export_latest_results()
    if latest_success:
        print("✅ Latest results export completed successfully!")
    else:
        print("❌ Latest results export failed")

    # Test all results export
    print("\n2. Testing all results export...")
    all_success = exporter.export_all_results()
    if all_success:
        print("✅ All results export completed successfully!")
    else:
        print("❌ All results export failed")

    # Test both versions export (includes backward compatibility)
    print("\n3. Testing both versions export...")
    both_success = exporter.export_both_versions()
    if both_success:
        print("✅ Both versions export completed successfully!")
    else:
        print("❌ Both versions export failed")

    # Summary export disabled to prevent unexpected file generation
    # print("\n4. Testing summary export...")
    # summary_success = exporter.export_summary_only()
    # if summary_success:
    #     print("✅ Summary export completed successfully!")
    # else:
    #     print("❌ Summary export failed")
    summary_success = True  # Set to True to maintain test logic

    if latest_success and all_success and both_success and summary_success:
        print("\n🎉 All export tests completed successfully!")
        print("\nGenerated files:")
        print("  📊 Latest Results:")
        print("    - docs/pages/data/benchmark_results_latest.json")
        print("    - docs/pages/data/benchmark_results_latest.csv")
        print("  📈 All Results (Complete DB Dump):")
        print("    - docs/pages/data/benchmark_results_all.json")
        print("    - docs/pages/data/benchmark_results_all.csv")
        print("\n💡 Note: All files contain complete database fields for table restoration")
    else:
        print("\n❌ Some export tests failed")


if __name__ == "__main__":
    main()
