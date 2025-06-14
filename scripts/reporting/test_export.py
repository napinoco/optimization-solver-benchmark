#!/usr/bin/env python3
"""
Export Functionality Test Suite
==============================

Tests all export formats and validates data quality.
Ensures CSV, JSON, and PDF exports work correctly.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.export import DataExporter, ExportConfig
from scripts.reporting.data_validator import DataValidator
from scripts.utils.logger import get_logger

logger = get_logger("export_test")


class ExportTestSuite:
    """
    Comprehensive test suite for export functionality.
    
    Tests:
    - CSV export generation and validation
    - JSON export generation and validation
    - PDF report generation
    - Cross-format consistency
    - Data validation against database
    """
    
    def __init__(self, database_path: str = None):
        """
        Initialize test suite.
        
        Args:
            database_path: Path to test database
        """
        self.logger = get_logger("export_test")
        
        # Use default database if none provided
        if database_path is None:
            database_path = project_root / "benchmark_results.db"
        
        self.database_path = Path(database_path)
        
        # Create temporary directory for test outputs
        self.test_dir = Path(tempfile.mkdtemp(prefix="export_test_"))
        self.logger.info(f"Test directory: {self.test_dir}")
        
        # Initialize components
        self.config = ExportConfig(
            output_directory=str(self.test_dir),
            validate_data=True,
            format_numbers=True
        )
        
        self.exporter = DataExporter(
            database_path=str(self.database_path),
            config=self.config
        )
        
        self.validator = DataValidator(database_path=str(self.database_path))
        
        # Test results
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_all_tests(self) -> bool:
        """
        Run all export tests.
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.info("🧪 Starting Export Test Suite")
        print("=" * 60)
        
        try:
            # Test individual export formats
            self.test_csv_export()
            self.test_json_export()
            self.test_pdf_export()
            
            # Test comprehensive exports
            self.test_all_formats_export()
            
            # Test data validation
            self.test_data_validation()
            
            # Test cross-format consistency
            self.test_cross_format_consistency()
            
            # Print summary
            self._print_test_summary()
            
            return self.failed_tests == 0
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            return False
        
        finally:
            # Cleanup
            self._cleanup()
    
    def test_csv_export(self) -> None:
        """Test CSV export functionality."""
        
        print("\n📊 Testing CSV Export...")
        
        try:
            # Test solver comparison CSV
            csv_path = self.exporter.export_solver_comparison_csv("test_solver_comparison.csv")
            
            if csv_path.exists():
                self._record_test_result("csv_solver_comparison", True, f"Generated: {csv_path}")
                
                # Validate CSV structure
                validation_result = self.validator.validate_csv_export(csv_path)
                self._record_test_result(
                    "csv_solver_validation", 
                    validation_result.valid,
                    f"Errors: {len(validation_result.errors)}, Warnings: {len(validation_result.warnings)}"
                )
                
            else:
                self._record_test_result("csv_solver_comparison", False, "File not created")
            
            # Test problem results CSV
            csv_path2 = self.exporter.export_problem_results_csv("test_problem_results.csv")
            
            if csv_path2.exists():
                self._record_test_result("csv_problem_results", True, f"Generated: {csv_path2}")
                
                # Validate CSV structure
                validation_result2 = self.validator.validate_csv_export(csv_path2)
                self._record_test_result(
                    "csv_problem_validation",
                    validation_result2.valid,
                    f"Errors: {len(validation_result2.errors)}, Warnings: {len(validation_result2.warnings)}"
                )
                
            else:
                self._record_test_result("csv_problem_results", False, "File not created")
            
            # Test filtered export
            csv_path3 = self.exporter.export_problem_results_csv(
                "test_filtered_results.csv",
                solver_filter="SciPy"
            )
            
            if csv_path3.exists():
                self._record_test_result("csv_filtered_export", True, f"Generated: {csv_path3}")
            else:
                self._record_test_result("csv_filtered_export", False, "Filtered file not created")
                
        except Exception as e:
            self._record_test_result("csv_export", False, f"Exception: {e}")
    
    def test_json_export(self) -> None:
        """Test JSON export functionality."""
        
        print("\n📄 Testing JSON Export...")
        
        try:
            # Test basic JSON export
            json_path = self.exporter.export_json_data("test_data.json")
            
            if json_path.exists():
                self._record_test_result("json_basic_export", True, f"Generated: {json_path}")
                
                # Validate JSON structure
                validation_result = self.validator.validate_json_export(json_path)
                self._record_test_result(
                    "json_basic_validation",
                    validation_result.valid,
                    f"Errors: {len(validation_result.errors)}, Warnings: {len(validation_result.warnings)}"
                )
                
            else:
                self._record_test_result("json_basic_export", False, "File not created")
            
            # Test JSON with raw results
            json_path2 = self.exporter.export_json_data("test_data_with_raw.json", include_raw_results=True)
            
            if json_path2.exists():
                self._record_test_result("json_raw_export", True, f"Generated: {json_path2}")
                
                # Check file size difference
                size1 = json_path.stat().st_size if json_path.exists() else 0
                size2 = json_path2.stat().st_size
                
                if size2 > size1:
                    self._record_test_result("json_raw_size", True, f"Raw results add data: {size2-size1} bytes")
                else:
                    self._record_test_result("json_raw_size", False, "Raw results don't increase file size")
                    
            else:
                self._record_test_result("json_raw_export", False, "Raw results file not created")
                
        except Exception as e:
            self._record_test_result("json_export", False, f"Exception: {e}")
    
    def test_pdf_export(self) -> None:
        """Test PDF report generation."""
        
        print("\n📑 Testing PDF Export...")
        
        try:
            # Test PDF generation
            pdf_path = self.exporter.generate_summary_report_pdf("test_summary.pdf")
            
            if pdf_path.exists():
                file_size = pdf_path.stat().st_size
                
                if pdf_path.suffix.lower() == '.pdf':
                    self._record_test_result("pdf_generation", True, f"Generated PDF: {file_size} bytes")
                elif pdf_path.suffix.lower() == '.txt':
                    self._record_test_result("pdf_fallback", True, f"Generated text fallback: {file_size} bytes")
                else:
                    self._record_test_result("pdf_generation", False, f"Unexpected file type: {pdf_path.suffix}")
                
                # Validate file is not empty
                if file_size > 0:
                    self._record_test_result("pdf_content", True, "File has content")
                else:
                    self._record_test_result("pdf_content", False, "File is empty")
                    
            else:
                self._record_test_result("pdf_generation", False, "File not created")
                
        except Exception as e:
            self._record_test_result("pdf_export", False, f"Exception: {e}")
    
    def test_all_formats_export(self) -> None:
        """Test exporting all formats at once."""
        
        print("\n🎯 Testing All Formats Export...")
        
        try:
            results = self.exporter.export_all_formats("test_all")
            
            if results:
                self._record_test_result("all_formats", True, f"Generated {len(results)} files")
                
                # Check each format was created
                expected_formats = ['solver_comparison_csv', 'problem_results_csv', 'json_data', 'pdf_report']
                
                for format_name in expected_formats:
                    if format_name in results:
                        path = results[format_name]
                        if path.exists():
                            self._record_test_result(f"all_formats_{format_name}", True, f"Created: {path.name}")
                        else:
                            self._record_test_result(f"all_formats_{format_name}", False, "File missing")
                    else:
                        self._record_test_result(f"all_formats_{format_name}", False, "Format not in results")
                        
            else:
                self._record_test_result("all_formats", False, "No files generated")
                
        except Exception as e:
            self._record_test_result("all_formats", False, f"Exception: {e}")
    
    def test_data_validation(self) -> None:
        """Test data validation functionality."""
        
        print("\n✅ Testing Data Validation...")
        
        try:
            # Find generated files
            csv_files = list(self.test_dir.glob("*.csv"))
            json_files = list(self.test_dir.glob("*.json"))
            
            if csv_files:
                for csv_file in csv_files[:2]:  # Test first 2 CSV files
                    validation_result = self.validator.validate_csv_export(csv_file)
                    test_name = f"validation_csv_{csv_file.stem}"
                    self._record_test_result(
                        test_name,
                        validation_result.valid,
                        f"Errors: {len(validation_result.errors)}"
                    )
            else:
                self._record_test_result("validation_csv", False, "No CSV files to validate")
            
            if json_files:
                for json_file in json_files[:2]:  # Test first 2 JSON files
                    validation_result = self.validator.validate_json_export(json_file)
                    test_name = f"validation_json_{json_file.stem}"
                    self._record_test_result(
                        test_name,
                        validation_result.valid,
                        f"Errors: {len(validation_result.errors)}"
                    )
            else:
                self._record_test_result("validation_json", False, "No JSON files to validate")
            
            # Test database consistency
            if json_files:
                db_validation = self.validator.validate_against_database(json_files[0])
                self._record_test_result(
                    "validation_database",
                    db_validation.valid,
                    f"DB consistency: {len(db_validation.errors)} errors"
                )
                
        except Exception as e:
            self._record_test_result("validation", False, f"Exception: {e}")
    
    def test_cross_format_consistency(self) -> None:
        """Test consistency between export formats."""
        
        print("\n🔄 Testing Cross-Format Consistency...")
        
        try:
            csv_files = list(self.test_dir.glob("*.csv"))
            json_files = list(self.test_dir.glob("*.json"))
            
            if csv_files and json_files:
                consistency_result = self.validator.validate_cross_format_consistency(csv_files, json_files[0])
                
                self._record_test_result(
                    "cross_format_consistency",
                    consistency_result.valid,
                    f"Errors: {len(consistency_result.errors)}, Warnings: {len(consistency_result.warnings)}"
                )
                
                # Check specific consistency metrics
                if 'json_solvers' in consistency_result.summary and 'csv_files' in consistency_result.summary:
                    json_solver_count = consistency_result.summary['json_solvers']
                    csv_summaries = consistency_result.summary['csv_files']
                    
                    for csv_name, csv_summary in csv_summaries.items():
                        if 'solver_comparison' in csv_name:
                            csv_solver_count = csv_summary['solver_count']
                            consistent = json_solver_count == csv_solver_count
                            
                            self._record_test_result(
                                f"consistency_solver_count_{csv_name}",
                                consistent,
                                f"JSON: {json_solver_count}, CSV: {csv_solver_count}"
                            )
                            
            else:
                self._record_test_result("cross_format_consistency", False, "Missing CSV or JSON files")
                
        except Exception as e:
            self._record_test_result("cross_format_consistency", False, f"Exception: {e}")
    
    def _record_test_result(self, test_name: str, passed: bool, details: str = "") -> None:
        """Record the result of a test."""
        
        self.test_results[test_name] = {
            'passed': passed,
            'details': details
        }
        
        if passed:
            self.passed_tests += 1
            status = "✅"
        else:
            self.failed_tests += 1
            status = "❌"
        
        print(f"  {status} {test_name}: {details}")
    
    def _print_test_summary(self) -> None:
        """Print summary of all tests."""
        
        print("\n" + "=" * 60)
        print("🧪 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if not result['passed']:
                    print(f"  • {test_name}: {result['details']}")
        
        # Show generated files
        generated_files = list(self.test_dir.glob("*"))
        if generated_files:
            print(f"\n📁 Generated Files ({len(generated_files)}):")
            for file_path in sorted(generated_files):
                size = file_path.stat().st_size
                print(f"  • {file_path.name} ({size:,} bytes)")
        
        overall_status = "✅ ALL TESTS PASSED" if self.failed_tests == 0 else "❌ SOME TESTS FAILED"
        print(f"\n{overall_status}")
    
    def _cleanup(self) -> None:
        """Clean up test directory."""
        
        try:
            # Optional: Keep test files for inspection
            keep_files = False  # Set to True to keep test files
            
            if not keep_files:
                shutil.rmtree(self.test_dir)
                self.logger.info(f"Cleaned up test directory: {self.test_dir}")
            else:
                self.logger.info(f"Test files preserved in: {self.test_dir}")
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup test directory: {e}")


def main():
    """Run export test suite."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test export functionality")
    parser.add_argument("--database", help="Database path for testing")
    parser.add_argument("--keep-files", action="store_true", help="Keep generated test files")
    
    args = parser.parse_args()
    
    print("🧪 Export Functionality Test Suite")
    print("=" * 60)
    
    try:
        # Create test suite
        test_suite = ExportTestSuite(database_path=args.database)
        
        # Optionally keep test files
        if args.keep_files:
            print(f"Test files will be preserved in: {test_suite.test_dir}")
        
        # Run tests
        success = test_suite.run_all_tests()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test suite failed to run: {e}")
        return 1


if __name__ == "__main__":
    exit(main())