"""
Data Validation Module
=====================

Validates exported data for consistency, completeness, and quality.
Ensures all export formats contain accurate and consistent information.
"""

import sys
import json
import csv
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("data_validator")


@dataclass
class ValidationResult:
    """Result of data validation."""
    
    valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]


class DataValidator:
    """
    Validates exported benchmark data for consistency and quality.
    
    Checks:
    - Data completeness across formats
    - Numerical consistency
    - Schema compliance
    - Cross-format consistency
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize data validator.
        
        Args:
            database_path: Path to source database
        """
        self.logger = get_logger("data_validator")
        
        if database_path is None:
            database_path = project_root / "benchmark_results.db"
        
        self.database_path = Path(database_path)
        self.logger.info(f"Data validator initialized with database: {self.database_path}")
    
    def validate_csv_export(self, csv_path: Path) -> ValidationResult:
        """
        Validate CSV export file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            ValidationResult with validation details
        """
        self.logger.info(f"Validating CSV export: {csv_path}")
        
        errors = []
        warnings = []
        summary = {}
        
        try:
            if not csv_path.exists():
                errors.append(f"CSV file does not exist: {csv_path}")
                return ValidationResult(False, errors, warnings, summary)
            
            # Read and validate CSV structure
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                
                if not fieldnames:
                    errors.append("CSV file has no headers")
                    return ValidationResult(False, errors, warnings, summary)
                
                rows = list(reader)
                summary['total_rows'] = len(rows)
                summary['columns'] = list(fieldnames)
                summary['column_count'] = len(fieldnames)
                
                # Validate required columns based on file type
                if 'solver_comparison' in csv_path.name:
                    required_cols = ['solver_name', 'success_rate', 'avg_solve_time']
                elif 'problem_results' in csv_path.name:
                    required_cols = ['solver_name', 'problem_name', 'status', 'solve_time']
                else:
                    required_cols = []
                
                missing_cols = [col for col in required_cols if col not in fieldnames]
                if missing_cols:
                    errors.append(f"Missing required columns: {missing_cols}")
                
                # Validate data consistency
                if rows:
                    # Check for empty rows
                    empty_rows = sum(1 for row in rows if all(not v.strip() for v in row.values()))
                    if empty_rows > 0:
                        warnings.append(f"Found {empty_rows} empty rows")
                    
                    # Validate numerical fields
                    numerical_errors = self._validate_numerical_fields(rows, fieldnames)
                    errors.extend(numerical_errors)
                    
                    # Check for duplicate entries
                    if 'solver_name' in fieldnames:
                        solver_names = [row['solver_name'] for row in rows if row.get('solver_name')]
                        duplicates = len(solver_names) - len(set(solver_names))
                        if duplicates > 0:
                            warnings.append(f"Found {duplicates} duplicate solver entries")
                
                summary['empty_rows'] = empty_rows if 'empty_rows' in locals() else 0
                summary['data_rows'] = len(rows) - summary['empty_rows']
                
        except Exception as e:
            errors.append(f"Error reading CSV file: {e}")
        
        is_valid = len(errors) == 0
        self.logger.info(f"CSV validation complete: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        return ValidationResult(is_valid, errors, warnings, summary)
    
    def validate_json_export(self, json_path: Path) -> ValidationResult:
        """
        Validate JSON export file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            ValidationResult with validation details
        """
        self.logger.info(f"Validating JSON export: {json_path}")
        
        errors = []
        warnings = []
        summary = {}
        
        try:
            if not json_path.exists():
                errors.append(f"JSON file does not exist: {json_path}")
                return ValidationResult(False, errors, warnings, summary)
            
            # Read and parse JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summary['file_size_mb'] = json_path.stat().st_size / (1024 * 1024)
            
            # Validate JSON structure
            if isinstance(data, dict):
                summary['top_level_keys'] = list(data.keys())
                
                # Check for expected sections
                expected_sections = ['metadata', 'solver_comparison']
                missing_sections = [sec for sec in expected_sections if sec not in data]
                if missing_sections:
                    warnings.append(f"Missing recommended sections: {missing_sections}")
                
                # Validate metadata section
                if 'metadata' in data:
                    metadata = data['metadata']
                    if not isinstance(metadata, dict):
                        errors.append("Metadata section is not a dictionary")
                    else:
                        if 'generated_at' not in metadata:
                            warnings.append("Metadata missing generation timestamp")
                        
                        summary['metadata_keys'] = list(metadata.keys())
                
                # Validate solver comparison section
                if 'solver_comparison' in data:
                    solvers = data['solver_comparison']
                    if isinstance(solvers, list):
                        summary['solver_count'] = len(solvers)
                        
                        # Validate solver entries
                        for i, solver in enumerate(solvers):
                            if not isinstance(solver, dict):
                                errors.append(f"Solver entry {i} is not a dictionary")
                                continue
                            
                            required_fields = ['solver_name']
                            missing_fields = [f for f in required_fields if f not in solver]
                            if missing_fields:
                                errors.append(f"Solver {i} missing fields: {missing_fields}")
                    else:
                        errors.append("Solver comparison is not a list")
                
                # Count total data points
                total_records = 0
                if 'raw_results' in data and isinstance(data['raw_results'], list):
                    total_records = len(data['raw_results'])
                elif 'solver_comparison' in data and isinstance(data['solver_comparison'], list):
                    total_records = sum(s.get('total_problems', 0) for s in data['solver_comparison'] if isinstance(s, dict))
                
                summary['total_records'] = total_records
                
            else:
                errors.append("JSON root is not a dictionary")
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format: {e}")
        except Exception as e:
            errors.append(f"Error reading JSON file: {e}")
        
        is_valid = len(errors) == 0
        self.logger.info(f"JSON validation complete: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        return ValidationResult(is_valid, errors, warnings, summary)
    
    def validate_cross_format_consistency(self, csv_paths: List[Path], 
                                        json_path: Path) -> ValidationResult:
        """
        Validate consistency across different export formats.
        
        Args:
            csv_paths: List of CSV file paths
            json_path: Path to JSON file
            
        Returns:
            ValidationResult with consistency check details
        """
        self.logger.info("Validating cross-format consistency")
        
        errors = []
        warnings = []
        summary = {}
        
        try:
            # Load JSON data
            if not json_path.exists():
                errors.append(f"JSON reference file missing: {json_path}")
                return ValidationResult(False, errors, warnings, summary)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Extract solver names from JSON
            json_solvers = set()
            if 'solver_comparison' in json_data:
                for solver in json_data['solver_comparison']:
                    if isinstance(solver, dict) and 'solver_name' in solver:
                        json_solvers.add(solver['solver_name'])
            
            summary['json_solvers'] = len(json_solvers)
            
            # Check each CSV file
            csv_summaries = {}
            for csv_path in csv_paths:
                if not csv_path.exists():
                    warnings.append(f"CSV file missing: {csv_path}")
                    continue
                
                # Load CSV data
                csv_solvers = set()
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'solver_name' in row and row['solver_name']:
                            csv_solvers.add(row['solver_name'])
                
                csv_name = csv_path.name
                csv_summaries[csv_name] = {
                    'solver_count': len(csv_solvers),
                    'solvers': sorted(csv_solvers)
                }
                
                # Compare solver sets
                if 'solver_comparison' in csv_name:
                    missing_in_csv = json_solvers - csv_solvers
                    extra_in_csv = csv_solvers - json_solvers
                    
                    if missing_in_csv:
                        errors.append(f"Solvers in JSON but missing in {csv_name}: {missing_in_csv}")
                    
                    if extra_in_csv:
                        warnings.append(f"Extra solvers in {csv_name}: {extra_in_csv}")
            
            summary['csv_files'] = csv_summaries
            
            # Check data counts consistency
            if 'metadata' in json_data and 'total_results' in json_data['metadata']:
                json_total = json_data['metadata']['total_results']
                summary['json_total_results'] = json_total
                
                # Find problem results CSV
                for csv_path in csv_paths:
                    if 'problem_results' in csv_path.name:
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            csv_total = sum(1 for _ in csv.DictReader(f))
                        
                        summary['csv_total_results'] = csv_total
                        
                        if abs(json_total - csv_total) > json_total * 0.05:  # 5% tolerance
                            warnings.append(f"Significant difference in result counts: JSON={json_total}, CSV={csv_total}")
                        break
            
        except Exception as e:
            errors.append(f"Error during cross-format validation: {e}")
        
        is_valid = len(errors) == 0
        self.logger.info(f"Cross-format validation complete: {'✅ Consistent' if is_valid else '❌ Inconsistent'}")
        
        return ValidationResult(is_valid, errors, warnings, summary)
    
    def validate_against_database(self, export_path: Path, 
                                 export_type: str = "auto") -> ValidationResult:
        """
        Validate exported data against source database.
        
        Args:
            export_path: Path to export file
            export_type: Type of export ("csv", "json", "auto")
            
        Returns:
            ValidationResult with database consistency check
        """
        if export_type == "auto":
            export_type = export_path.suffix.lower().lstrip('.')
        
        self.logger.info(f"Validating {export_type} export against database")
        
        errors = []
        warnings = []
        summary = {}
        
        try:
            if not self.database_path.exists():
                errors.append(f"Source database not found: {self.database_path}")
                return ValidationResult(False, errors, warnings, summary)
            
            # Get database statistics
            db_stats = self._get_database_statistics()
            summary['database_stats'] = db_stats
            
            if export_type == "json":
                export_stats = self._get_json_statistics(export_path)
            elif export_type == "csv":
                export_stats = self._get_csv_statistics(export_path)
            else:
                errors.append(f"Unsupported export type: {export_type}")
                return ValidationResult(False, errors, warnings, summary)
            
            summary['export_stats'] = export_stats
            
            # Compare statistics
            for key in ['total_results', 'total_solvers', 'total_problems']:
                if key in db_stats and key in export_stats:
                    db_val = db_stats[key]
                    exp_val = export_stats[key]
                    
                    if db_val != exp_val:
                        diff_pct = abs(db_val - exp_val) / max(db_val, 1) * 100
                        if diff_pct > 5:  # 5% tolerance
                            errors.append(f"{key} mismatch: DB={db_val}, Export={exp_val} ({diff_pct:.1f}% diff)")
                        else:
                            warnings.append(f"Minor {key} difference: DB={db_val}, Export={exp_val}")
            
        except Exception as e:
            errors.append(f"Error during database validation: {e}")
        
        is_valid = len(errors) == 0
        self.logger.info(f"Database validation complete: {'✅ Consistent' if is_valid else '❌ Inconsistent'}")
        
        return ValidationResult(is_valid, errors, warnings, summary)
    
    def _validate_numerical_fields(self, rows: List[Dict], fieldnames: List[str]) -> List[str]:
        """Validate numerical fields in CSV data."""
        
        errors = []
        numerical_fields = [
            'solve_time', 'objective_value', 'success_rate', 'avg_solve_time',
            'iterations', 'n_variables', 'n_constraints'
        ]
        
        for field in numerical_fields:
            if field not in fieldnames:
                continue
            
            invalid_count = 0
            for i, row in enumerate(rows):
                value = row.get(field, '').strip()
                if not value:
                    continue
                
                try:
                    float_val = float(value)
                    
                    # Check for reasonable ranges
                    if field == 'solve_time' and float_val < 0:
                        errors.append(f"Negative solve time in row {i}: {value}")
                    elif field == 'success_rate' and not (0 <= float_val <= 1):
                        errors.append(f"Success rate out of range [0,1] in row {i}: {value}")
                    elif field in ['iterations', 'n_variables', 'n_constraints'] and float_val < 0:
                        errors.append(f"Negative {field} in row {i}: {value}")
                        
                except ValueError:
                    invalid_count += 1
            
            if invalid_count > 0:
                errors.append(f"Invalid numerical values in {field}: {invalid_count} rows")
        
        return errors
    
    def _get_database_statistics(self) -> Dict[str, Any]:
        """Get statistics from source database."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM benchmark_results')
                total_results = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT solver_name) FROM benchmark_results')
                total_solvers = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT problem_name) FROM benchmark_results')
                total_problems = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM benchmark_results WHERE status = 'optimal'")
                successful_results = cursor.fetchone()[0]
                
                return {
                    'total_results': total_results,
                    'total_solvers': total_solvers,
                    'total_problems': total_problems,
                    'successful_results': successful_results,
                    'success_rate': successful_results / total_results if total_results > 0 else 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get database statistics: {e}")
            return {}
    
    def _get_json_statistics(self, json_path: Path) -> Dict[str, Any]:
        """Get statistics from JSON export."""
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats = {}
            
            if 'metadata' in data:
                metadata = data['metadata']
                stats.update({
                    'total_results': metadata.get('total_results', 0),
                    'total_solvers': metadata.get('total_solvers', 0),
                    'total_problems': metadata.get('total_problems', 0)
                })
            
            if 'solver_comparison' in data:
                solvers = data['solver_comparison']
                if isinstance(solvers, list):
                    stats['solver_count'] = len(solvers)
                    total_attempts = sum(s.get('problems_attempted', 0) for s in solvers if isinstance(s, dict))
                    if total_attempts > 0:
                        stats['total_results'] = total_attempts
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get JSON statistics: {e}")
            return {}
    
    def _get_csv_statistics(self, csv_path: Path) -> Dict[str, Any]:
        """Get statistics from CSV export."""
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            stats = {'total_results': len(rows)}
            
            if 'solver_name' in reader.fieldnames:
                solvers = set(row['solver_name'] for row in rows if row.get('solver_name'))
                stats['total_solvers'] = len(solvers)
            
            if 'problem_name' in reader.fieldnames:
                problems = set(row['problem_name'] for row in rows if row.get('problem_name'))
                stats['total_problems'] = len(problems)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get CSV statistics: {e}")
            return {}


def main():
    """Command-line interface for data validation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate exported benchmark data")
    parser.add_argument("files", nargs="+", help="Export files to validate")
    parser.add_argument("--database", help="Source database path")
    parser.add_argument("--check-consistency", action="store_true", 
                       help="Check consistency across formats")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = DataValidator(database_path=args.database)
    
    all_valid = True
    results = {}
    
    # Validate individual files
    for file_path in args.files:
        path = Path(file_path)
        
        if path.suffix.lower() == '.csv':
            result = validator.validate_csv_export(path)
        elif path.suffix.lower() == '.json':
            result = validator.validate_json_export(path)
        else:
            print(f"⚠️  Unsupported file type: {path}")
            continue
        
        results[str(path)] = result
        
        print(f"\n📄 {path.name}")
        print(f"Status: {'✅ Valid' if result.valid else '❌ Invalid'}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  ❌ {error}")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
        
        if args.verbose and result.summary:
            print("Summary:")
            for key, value in result.summary.items():
                print(f"  {key}: {value}")
        
        if not result.valid:
            all_valid = False
    
    # Check cross-format consistency if requested
    if args.check_consistency and len(args.files) > 1:
        csv_files = [Path(f) for f in args.files if f.endswith('.csv')]
        json_files = [Path(f) for f in args.files if f.endswith('.json')]
        
        if csv_files and json_files:
            consistency_result = validator.validate_cross_format_consistency(csv_files, json_files[0])
            
            print(f"\n🔄 Cross-format Consistency")
            print(f"Status: {'✅ Consistent' if consistency_result.valid else '❌ Inconsistent'}")
            
            if consistency_result.errors:
                for error in consistency_result.errors:
                    print(f"  ❌ {error}")
            
            if consistency_result.warnings:
                for warning in consistency_result.warnings:
                    print(f"  ⚠️  {warning}")
            
            if not consistency_result.valid:
                all_valid = False
    
    print(f"\n📋 Overall Result: {'✅ All Valid' if all_valid else '❌ Issues Found'}")
    return 0 if all_valid else 1


if __name__ == "__main__":
    exit(main())