"""
Database Table Restorer
======================

Restore the results table from exported JSON/CSV files to verify data integrity
and enable database recovery from exported data.
"""

import json
import csv
import sqlite3
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.reporting.result_processor import BenchmarkResult
from scripts.database.database_manager import DatabaseManager
from scripts.utils.logger import get_logger

logger = get_logger("table_restorer")


class TableRestorer:
    """Restore database table from exported data"""
    
    def __init__(self, target_db_path: Optional[str] = None):
        """
        Initialize table restorer
        
        Args:
            target_db_path: Path to target database (defaults to test database)
        """
        if target_db_path is None:
            # Use test database by default to avoid overwriting production data
            target_db_path = project_root / "database" / "results_restored.db"
        
        self.target_db_path = Path(target_db_path)
        self.logger = get_logger("table_restorer")
        
        # Create database manager for target database
        self.db_manager = DatabaseManager(str(self.target_db_path))
    
    def restore_from_json(self, json_file_path: str) -> bool:
        """
        Restore database table from JSON export file
        
        Args:
            json_file_path: Path to JSON export file
            
        Returns:
            True if restoration successful, False otherwise
        """
        
        self.logger.info(f"Restoring database from JSON file: {json_file_path}")
        
        try:
            # Load JSON data
            with open(json_file_path, 'r') as f:
                export_data = json.load(f)
            
            # Extract results from export data
            if 'results' not in export_data:
                self.logger.error("JSON file does not contain 'results' key")
                return False
            
            results_data = export_data['results']
            
            # Convert to BenchmarkResult objects
            results = []
            for result_dict in results_data:
                result = self._dict_to_benchmark_result(result_dict)
                if result:
                    results.append(result)
            
            # Insert into database
            success = self._insert_results_to_database(results)
            
            if success:
                self.logger.info(f"Successfully restored {len(results)} results from JSON")
                return True
            else:
                self.logger.error("Failed to insert results to database")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore from JSON: {e}")
            return False
    
    def restore_from_csv(self, csv_file_path: str) -> bool:
        """
        Restore database table from CSV export file
        
        Args:
            csv_file_path: Path to CSV export file
            
        Returns:
            True if restoration successful, False otherwise
        """
        
        self.logger.info(f"Restoring database from CSV file: {csv_file_path}")
        
        try:
            # Load CSV data
            results = []
            with open(csv_file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    result = self._dict_to_benchmark_result(row)
                    if result:
                        results.append(result)
            
            # Insert into database
            success = self._insert_results_to_database(results)
            
            if success:
                self.logger.info(f"Successfully restored {len(results)} results from CSV")
                return True
            else:
                self.logger.error("Failed to insert results to database")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore from CSV: {e}")
            return False
    
    def _dict_to_benchmark_result(self, data_dict: Dict[str, Any]) -> Optional[BenchmarkResult]:
        """
        Convert dictionary to BenchmarkResult object
        
        Args:
            data_dict: Dictionary containing result data
            
        Returns:
            BenchmarkResult object or None if conversion fails
        """
        
        try:
            # Handle JSON strings and empty values in CSV data
            processed_dict = {}
            for key, value in data_dict.items():
                # Convert empty strings to None (CSV limitation)
                if value == '' or value is None:
                    processed_dict[key] = None
                elif key in ['environment_info', 'memo'] and isinstance(value, str) and value:
                    try:
                        processed_dict[key] = json.loads(value)
                    except json.JSONDecodeError:
                        processed_dict[key] = value
                else:
                    processed_dict[key] = value
            
            # Convert to BenchmarkResult using the existing from_dict method
            result = BenchmarkResult.from_dict(processed_dict)
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to convert dict to BenchmarkResult: {e}")
            return None
    
    def _insert_results_to_database(self, results: List[BenchmarkResult]) -> bool:
        """
        Insert BenchmarkResult objects into database
        
        Args:
            results: List of BenchmarkResult objects
            
        Returns:
            True if insertion successful, False otherwise
        """
        
        try:
            # Create database tables if they don't exist
            self.db_manager.ensure_schema()
            
            # Insert each result directly
            for result in results:
                self._store_result_direct(result)
            
            self.logger.info(f"Successfully inserted {len(results)} results into database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert results to database: {e}")
            return False
    
    def _store_result_direct(self, result: BenchmarkResult) -> None:
        """
        Store BenchmarkResult directly to database with original metadata
        
        Args:
            result: BenchmarkResult to store
        """
        
        # Insert directly to preserve original timestamps and metadata
        query = """
        INSERT INTO results 
        (solver_name, solver_version, problem_library, problem_name, problem_type,
         environment_info, commit_hash, timestamp, solve_time, status, 
         primal_objective_value, dual_objective_value, duality_gap,
         primal_infeasibility, dual_infeasibility, iterations, memo)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Convert timestamp back to database format
        timestamp_str = result.timestamp.isoformat() if result.timestamp else None
        
        # Use direct sqlite3 connection like DatabaseManager does
        with sqlite3.connect(self.target_db_path) as conn:
            conn.execute(query, (
                result.solver_name,
                result.solver_version, 
                result.problem_library,
                result.problem_name,
                result.problem_type,
                json.dumps(result.environment_info) if result.environment_info else None,
                result.commit_hash,
                timestamp_str,
                result.solve_time,
                result.status,
                result.primal_objective_value,
                result.dual_objective_value,
                result.duality_gap,
                result.primal_infeasibility,
                result.dual_infeasibility,
                result.iterations,
                result.memo if isinstance(result.memo, str) else json.dumps(result.memo) if result.memo else None
            ))
            conn.commit()
    
    def _normalize_row_for_comparison(self, row: tuple, columns: List[str]) -> tuple:
        """
        Normalize row data for meaningful comparison (handle format differences)
        
        Args:
            row: Database row tuple
            columns: Column names
            
        Returns:
            Normalized row tuple
        """
        normalized = list(row)
        
        for i, (value, column) in enumerate(zip(row, columns)):
            # Normalize None and empty string to None for comparison
            if value is None or value == '':
                normalized[i] = None
                continue
                
            # Normalize timestamp format
            if column == 'timestamp' and isinstance(value, str):
                # Convert both formats to same format for comparison
                if 'T' in value:
                    # ISO format -> SQLite format
                    normalized[i] = value.replace('T', ' ')
                # SQLite format stays as is
            
            # Normalize JSON format
            elif column == 'environment_info' and isinstance(value, str):
                try:
                    # Parse and re-serialize to normalize format
                    parsed = json.loads(value)
                    normalized[i] = json.dumps(parsed, sort_keys=True, separators=(',', ':'))
                except json.JSONDecodeError:
                    # Keep original if not valid JSON
                    pass
            elif column == 'memo' and isinstance(value, str):
                # For memo field, handle both plain strings and JSON strings
                try:
                    # Try to parse as JSON first
                    parsed = json.loads(value)
                    if isinstance(parsed, str):
                        # JSON-encoded string, use the decoded string
                        normalized[i] = parsed
                    else:
                        # JSON object, normalize to compact format
                        normalized[i] = json.dumps(parsed, sort_keys=True, separators=(',', ':'))
                except json.JSONDecodeError:
                    # Not valid JSON, keep as plain string
                    pass
        
        return tuple(normalized)
    
    def compare_databases(self, original_db_path: str, restored_db_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare original database with restored database
        
        Args:
            original_db_path: Path to original database
            restored_db_path: Path to restored database (defaults to self.target_db_path)
            
        Returns:
            Dictionary containing comparison results
        """
        
        if restored_db_path is None:
            restored_db_path = self.target_db_path
        
        self.logger.info(f"Comparing databases: {original_db_path} vs {restored_db_path}")
        
        try:
            # Connect to both databases
            original_conn = sqlite3.connect(original_db_path)
            restored_conn = sqlite3.connect(restored_db_path)
            
            # Get row counts
            original_count = original_conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            restored_count = restored_conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
            
            # Get sample data for comparison (ordered by meaningful fields, not id)
            # Order by solver_name, problem_name, timestamp for consistent comparison
            original_sample = original_conn.execute("""
                SELECT * FROM results 
                ORDER BY solver_name, problem_name, timestamp 
                LIMIT 10
            """).fetchall()
            
            restored_sample = restored_conn.execute("""
                SELECT * FROM results 
                ORDER BY solver_name, problem_name, timestamp 
                LIMIT 10
            """).fetchall()
            
            # Get column info (extract column names from table_info)
            original_columns = [info[1] for info in original_conn.execute("PRAGMA table_info(results)").fetchall()]
            restored_columns = [info[1] for info in restored_conn.execute("PRAGMA table_info(results)").fetchall()]
            
            # Close connections
            original_conn.close()
            restored_conn.close()
            
            # Calculate differences
            row_count_match = original_count == restored_count
            schema_match = original_columns == restored_columns
            
            # Compare sample data (excluding id and timestamp which might differ)
            sample_match = True
            sample_differences = []
            
            if len(original_sample) == len(restored_sample):
                for row_idx, (orig_row, rest_row) in enumerate(zip(original_sample, restored_sample)):
                    # Compare all fields except id (index 0) which might be auto-generated differently
                    # Apply normalization for meaningful comparison
                    normalized_orig = self._normalize_row_for_comparison(orig_row, original_columns)
                    normalized_rest = self._normalize_row_for_comparison(rest_row, restored_columns)
                    
                    if normalized_orig[1:] != normalized_rest[1:]:  # Skip first field (id)
                        sample_match = False
                        # Find specific field differences (only show if truly different after normalization)
                        for field_idx in range(1, len(normalized_orig)):
                            if field_idx < len(normalized_rest) and normalized_orig[field_idx] != normalized_rest[field_idx]:
                                field_name = original_columns[field_idx] if field_idx < len(original_columns) else f"field_{field_idx}"
                                sample_differences.append({
                                    'row': row_idx,
                                    'field': field_name,
                                    'original': orig_row[field_idx],
                                    'restored': rest_row[field_idx],
                                    'normalized_original': normalized_orig[field_idx],
                                    'normalized_restored': normalized_rest[field_idx]
                                })
                        if len(sample_differences) >= 5:  # Limit to first 5 differences
                            break
                    else:
                        # This row matches after normalization
                        pass
            else:
                sample_match = False
                sample_differences.append({'error': 'Different number of sample rows'})
            
            comparison_result = {
                'original_count': original_count,
                'restored_count': restored_count,
                'row_count_match': row_count_match,
                'schema_match': schema_match,
                'sample_match': sample_match,
                'sample_differences': sample_differences,
                'original_columns': original_columns,
                'restored_columns': restored_columns,
                'success': row_count_match and schema_match and sample_match
            }
            
            self.logger.info(f"Database comparison completed: {'SUCCESS' if comparison_result['success'] else 'FAILED'}")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Failed to compare databases: {e}")
            return {'success': False, 'error': str(e)}


def restore_database_cli():
    """Command-line interface for database restoration"""
    parser = argparse.ArgumentParser(
        description="Restore SQLite database from exported JSON/CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Restore database from default JSON file
  python table_restorer.py

  # Restore from specific JSON file to specific database
  python table_restorer.py --input-json /path/to/data.json --output-db /path/to/output.db

  # Restore from CSV file
  python table_restorer.py --input-json /path/to/data.csv --output-db /path/to/output.db

  # Run tests
  python table_restorer.py --test
        """
    )
    
    parser.add_argument(
        '--input-json', '--input',
        default=str(project_root / "docs" / "pages" / "data" / "benchmark_results_all.json"),
        help='Path to input JSON or CSV file (default: docs/pages/data/benchmark_results_all.json)'
    )
    
    parser.add_argument(
        '--output-db', '--output',
        default=str(project_root / "database" / "results.db"),
        help='Path to output database file (default: database/results.db)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run comprehensive test suite instead of restoration'
    )
    
    parser.add_argument(
        '--compare-with',
        help='Compare restored database with specified original database file'
    )
    
    args = parser.parse_args()
    
    if args.test:
        run_test_suite()
        return
    
    # Validate input file
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine file type
    file_extension = input_path.suffix.lower()
    if file_extension not in ['.json', '.csv']:
        print(f"❌ Error: Unsupported file type: {file_extension}. Supported: .json, .csv")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output_db)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Restoring database from {file_extension.upper()} file...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    
    # Perform restoration
    restorer = TableRestorer(target_db_path=str(output_path))
    
    try:
        if file_extension == '.json':
            success = restorer.restore_from_json(str(input_path))
        else:  # .csv
            success = restorer.restore_from_csv(str(input_path))
        
        if success:
            print("✅ Database restoration completed successfully!")
            
            # Optional comparison
            if args.compare_with:
                comparison_path = Path(args.compare_with)
                if comparison_path.exists():
                    print(f"\n🔍 Comparing with original database: {comparison_path}")
                    comparison = restorer.compare_databases(str(comparison_path))
                    
                    print(f"📊 Comparison Results:")
                    print(f"  Original rows: {comparison.get('original_count', 'N/A')}")
                    print(f"  Restored rows: {comparison.get('restored_count', 'N/A')}")
                    print(f"  Row count match: {'✅' if comparison.get('row_count_match') else '❌'}")
                    print(f"  Schema match: {'✅' if comparison.get('schema_match') else '❌'}")
                    print(f"  Sample data match: {'✅' if comparison.get('sample_match') else '❌'}")
                    print(f"  Overall success: {'✅' if comparison.get('success') else '❌'}")
                else:
                    print(f"⚠️  Warning: Comparison database not found: {comparison_path}")
            
        else:
            print("❌ Database restoration failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during restoration: {e}")
        sys.exit(1)


def run_test_suite():
    """Run comprehensive test suite for table restoration functionality"""
    print("🔄 Testing Database Table Restoration...")
    
    # Test JSON restoration
    print("\n1. Testing JSON restoration...")
    restorer_json = TableRestorer(target_db_path="database/results_restored_json.db")
    json_file = project_root / "docs" / "pages" / "data" / "benchmark_results_all.json"
    
    if json_file.exists():
        json_success = restorer_json.restore_from_json(str(json_file))
        if json_success:
            print("✅ JSON restoration completed successfully!")
        else:
            print("❌ JSON restoration failed")
    else:
        print("❌ JSON file not found")
        json_success = False
    
    # Test JSON database comparison
    print("\n2. Testing JSON restoration comparison...")
    if json_success:
        original_db = project_root / "database" / "results.db"
        comparison = restorer_json.compare_databases(str(original_db))
        
        print(f"📊 JSON Restoration Results:")
        print(f"  Original rows: {comparison.get('original_count', 'N/A')}")
        print(f"  Restored rows: {comparison.get('restored_count', 'N/A')}")
        print(f"  Row count match: {'✅' if comparison.get('row_count_match') else '❌'}")
        print(f"  Schema match: {'✅' if comparison.get('schema_match') else '❌'}")
        print(f"  Sample data match: {'✅' if comparison.get('sample_match') else '❌'}")
        print(f"  Overall success: {'✅' if comparison.get('success') else '❌'}")
        
        if comparison.get('success'):
            print("  🎉 JSON restoration test PASSED!")
        else:
            print("  ❌ JSON restoration test FAILED!")
    
    # Test CSV restoration
    print("\n3. Testing CSV restoration...")
    restorer_csv = TableRestorer(target_db_path="database/results_restored_csv.db")
    csv_file = project_root / "docs" / "pages" / "data" / "benchmark_results_all.csv"
    
    if csv_file.exists():
        csv_success = restorer_csv.restore_from_csv(str(csv_file))
        if csv_success:
            print("✅ CSV restoration completed successfully!")
        else:
            print("❌ CSV restoration failed")
    else:
        print("❌ CSV file not found")
        csv_success = False
    
    # Test CSV database comparison
    print("\n4. Testing CSV restoration comparison...")
    if csv_success:
        original_db = project_root / "database" / "results.db"
        comparison = restorer_csv.compare_databases(str(original_db))
        
        print(f"📊 CSV Restoration Results:")
        print(f"  Original rows: {comparison.get('original_count', 'N/A')}")
        print(f"  Restored rows: {comparison.get('restored_count', 'N/A')}")
        print(f"  Row count match: {'✅' if comparison.get('row_count_match') else '❌'}")
        print(f"  Schema match: {'✅' if comparison.get('schema_match') else '❌'}")
        print(f"  Sample data match: {'✅' if comparison.get('sample_match') else '❌'}")
        print(f"  Overall success: {'✅' if comparison.get('success') else '❌'}")
        
        if comparison.get('success'):
            print("  🎉 CSV restoration test PASSED!")
        else:
            print("  ❌ CSV restoration test FAILED!")
    
    # Summary  
    print("\n5. Test Summary:")
    json_comparison = restorer_json.compare_databases(str(original_db)) if json_success else {'success': False}
    csv_comparison = restorer_csv.compare_databases(str(original_db)) if csv_success else {'success': False}
    
    json_result = "✅ PASSED" if (json_success and json_comparison.get('success', False)) else "❌ FAILED"
    csv_result = "✅ PASSED" if (csv_success and csv_comparison.get('success', False)) else "❌ FAILED" 
    
    print(f"  JSON restoration: {json_result}")
    print(f"  CSV restoration: {csv_result}")
    
    if json_comparison.get('success', False) and csv_comparison.get('success', False):
        print("\n🎉 COMPLETE SUCCESS! Both JSON and CSV exports can fully restore the original database table.")
        print("📋 The exported data contains all necessary information for table restoration.")
    else:
        print("\n⚠️ Some tests failed. Check the detailed results above.")
    
    # Keep test databases for manual inspection
    print("\n6. Test databases available for inspection:")
    if json_success:
        print(f"  JSON restored DB: {restorer_json.target_db_path}")
    if csv_success:
        print(f"  CSV restored DB: {restorer_csv.target_db_path}")
    
    print("\n📋 Table restoration test completed!")


def main():
    """Main entry point - delegates to CLI interface"""
    restore_database_cli()


if __name__ == "__main__":
    main()