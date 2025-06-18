"""
Simplified Data Exporter
========================

Export benchmark results in JSON and CSV formats for external consumption.
Simplified approach as specified in the re-architected design.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import sys
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.result_processor import ResultProcessor, BenchmarkResult
from scripts.utils.logger import get_logger

logger = get_logger("data_exporter")


class DataExporter:
    """Export data in JSON and CSV formats"""
    
    def __init__(self, output_dir: str = None):
        """Initialize data exporter with output directory"""
        if output_dir is None:
            output_dir = project_root / "docs" / "pages" / "data"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.result_processor = ResultProcessor()
        self.logger = get_logger("data_exporter")
    
    def export_latest_results(self) -> bool:
        """Export latest results to JSON and CSV files"""
        
        self.logger.info("Exporting latest results...")
        
        try:
            # Get latest results
            results = self.result_processor.get_latest_results_for_reporting()
            
            if not results:
                self.logger.warning("No results found for export")
                return False
            
            # Export both formats
            json_success = self.export_json(results)
            csv_success = self.export_csv(results)
            
            if json_success and csv_success:
                self.logger.info("Data export completed successfully")
                return True
            else:
                self.logger.error("Data export failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            return False
    
    def export_json(self, results: List[BenchmarkResult]) -> bool:
        """Export results to JSON format"""
        
        self.logger.info("Exporting results to JSON...")
        
        try:
            # Get summary statistics
            summary = self.result_processor.get_summary_statistics(results)
            solver_comparison = self.result_processor.get_solver_comparison(results)
            
            # Create export data structure
            export_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_results": len(results),
                    "export_format": "simplified_benchmark_results",
                    "version": "1.0"
                },
                "summary": summary,
                "solver_comparison": solver_comparison,
                "results": [result.to_dict() for result in results]
            }
            
            # Export to JSON file
            output_file = self.output_dir / "benchmark_results.json"
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"JSON export saved to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return False
    
    def export_csv(self, results: List[BenchmarkResult]) -> bool:
        """Export results to CSV format"""
        
        self.logger.info("Exporting results to CSV...")
        
        try:
            output_file = self.output_dir / "benchmark_results.csv"
            
            # Define CSV fieldnames (all standardized result fields)
            fieldnames = [
                'solver_name',
                'solver_version', 
                'problem_name',
                'problem_type',
                'problem_library',
                'status',
                'solve_time',
                'primal_objective_value',
                'dual_objective_value',
                'duality_gap',
                'primal_infeasibility',
                'dual_infeasibility',
                'iterations',
                'commit_hash',
                'timestamp'
            ]
            
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Convert result to flat dictionary for CSV
                    row = {
                        'solver_name': result.solver_name,
                        'solver_version': result.solver_version,
                        'problem_name': result.problem_name,
                        'problem_type': result.problem_type,
                        'problem_library': result.problem_library,
                        'status': result.status,
                        'solve_time': result.solve_time,
                        'primal_objective_value': result.primal_objective_value,
                        'dual_objective_value': result.dual_objective_value,
                        'duality_gap': result.duality_gap,
                        'primal_infeasibility': result.primal_infeasibility,
                        'dual_infeasibility': result.dual_infeasibility,
                        'iterations': result.iterations,
                        'commit_hash': result.commit_hash,
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None
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
                    "platform": results[0].environment_info.get('platform', 'Unknown') if results and results[0].environment_info else "Unknown"
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


def main():
    """Test data exporter"""
    exporter = DataExporter()
    
    print("Testing Data Exporter...")
    
    # Export full results
    success = exporter.export_latest_results()
    if success:
        print("✅ Full data export completed successfully!")
    else:
        print("❌ Full data export failed")
    
    # Export summary only
    summary_success = exporter.export_summary_only()
    if summary_success:
        print("✅ Summary export completed successfully!")
    else:
        print("❌ Summary export failed")
    
    if success and summary_success:
        print("\nGenerated files:")
        print("  - docs/pages/data/benchmark_results.json (Full results)")
        print("  - docs/pages/data/benchmark_results.csv (Full results CSV)")
        print("  - docs/pages/data/summary.json (Summary statistics)")


if __name__ == "__main__":
    main()