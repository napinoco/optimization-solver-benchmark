"""
Essential Export Functionality
=============================

Provides multiple export formats for benchmark data including CSV, JSON, and basic PDF reports.
Designed for research use with configurable options and data validation.
"""

import os
import sys
import csv
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger
from scripts.database.models import create_database

logger = get_logger("data_exporter")


@dataclass
class ExportConfig:
    """Configuration for data export operations."""
    
    output_directory: str = "exports"
    include_metadata: bool = True
    include_timestamps: bool = True
    validate_data: bool = True
    format_numbers: bool = True
    decimal_places: int = 6


class DataExporter:
    """
    Comprehensive data export system for benchmark results.
    
    Supports multiple output formats:
    - CSV for spreadsheet analysis
    - JSON for programmatic access
    - Basic PDF reports for documentation
    """
    
    def __init__(self, database_path: Optional[str] = None, config: Optional[ExportConfig] = None):
        """
        Initialize data exporter.
        
        Args:
            database_path: Path to SQLite database
            config: Export configuration
        """
        self.logger = get_logger("data_exporter")
        
        # Set up database connection
        if database_path is None:
            database_path = project_root / "benchmark_results.db"
        
        self.database_path = Path(database_path)
        self.config = config or ExportConfig()
        
        # Create output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Data exporter initialized with database: {self.database_path}")
    
    def export_solver_comparison_csv(self, filename: Optional[str] = None) -> Path:
        """
        Export solver performance comparison as CSV.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to generated CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solver_comparison_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        self.logger.info(f"Exporting solver comparison to {output_path}")
        
        try:
            # Query solver performance data
            data = self._get_solver_comparison_data()
            
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                if not data:
                    self.logger.warning("No solver comparison data available")
                    return output_path
                
                # Define CSV columns
                fieldnames = [
                    'solver_name', 'total_problems', 'problems_solved', 
                    'success_rate', 'avg_solve_time', 'min_solve_time', 
                    'max_solve_time', 'total_time', 'avg_iterations',
                    'problem_types_supported', 'last_run'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in data:
                    # Format numerical values
                    if self.config.format_numbers:
                        row = self._format_numerical_values(row)
                    
                    writer.writerow(row)
            
            self.logger.info(f"✅ Solver comparison CSV exported: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export solver comparison CSV: {e}")
            raise
    
    def export_problem_results_csv(self, filename: Optional[str] = None, 
                                  solver_filter: Optional[str] = None,
                                  problem_filter: Optional[str] = None) -> Path:
        """
        Export detailed problem results as CSV.
        
        Args:
            filename: Output filename (auto-generated if None)
            solver_filter: Filter by solver name (SQL LIKE pattern)
            problem_filter: Filter by problem name (SQL LIKE pattern)
            
        Returns:
            Path to generated CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"problem_results_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        self.logger.info(f"Exporting problem results to {output_path}")
        
        try:
            # Query detailed results
            data = self._get_problem_results_data(solver_filter, problem_filter)
            
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                if not data:
                    self.logger.warning("No problem results data available")
                    return output_path
                
                # Define CSV columns
                fieldnames = [
                    'benchmark_id', 'solver_name', 'problem_name', 'problem_type',
                    'status', 'solve_time', 'objective_value', 'iterations',
                    'duality_gap', 'n_variables', 'n_constraints', 'timestamp',
                    'error_message'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in data:
                    # Format numerical values
                    if self.config.format_numbers:
                        row = self._format_numerical_values(row)
                    
                    writer.writerow(row)
            
            self.logger.info(f"✅ Problem results CSV exported: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export problem results CSV: {e}")
            raise
    
    def export_json_data(self, filename: Optional[str] = None,
                        include_raw_results: bool = False) -> Path:
        """
        Export comprehensive benchmark data as JSON.
        
        Args:
            filename: Output filename (auto-generated if None)
            include_raw_results: Include all raw result records
            
        Returns:
            Path to generated JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_data_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        self.logger.info(f"Exporting JSON data to {output_path}")
        
        try:
            # Collect all data
            export_data = {
                'metadata': self._get_export_metadata(),
                'solver_comparison': self._get_solver_comparison_data(),
                'problem_statistics': self._get_problem_statistics_data(),
                'environment_info': self._get_environment_info(),
            }
            
            if include_raw_results:
                export_data['raw_results'] = self._get_all_results_data()
            
            # Write JSON with proper formatting
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"✅ JSON data exported: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export JSON data: {e}")
            raise
    
    def generate_summary_report_pdf(self, filename: Optional[str] = None) -> Path:
        """
        Generate basic PDF summary report.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_summary_{timestamp}.pdf"
        
        output_path = self.output_dir / filename
        
        self.logger.info(f"Generating PDF summary report: {output_path}")
        
        try:
            # Try to import PDF generation libraries
            try:
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib import colors
            except ImportError:
                raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")
            
            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("Optimization Solver Benchmark Report", title_style))
            story.append(Spacer(1, 20))
            
            # Metadata section
            metadata = self._get_export_metadata()
            story.append(Paragraph("Report Information", styles['Heading2']))
            
            meta_data = [
                ['Generated', metadata.get('generated_at', 'Unknown')],
                ['Database', str(self.database_path)],
                ['Total Results', str(metadata.get('total_results', 0))],
                ['Total Solvers', str(metadata.get('total_solvers', 0))],
                ['Total Problems', str(metadata.get('total_problems', 0))]
            ]
            
            meta_table = Table(meta_data, colWidths=[2*inch, 3*inch])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(meta_table)
            story.append(Spacer(1, 20))
            
            # Solver comparison section
            story.append(Paragraph("Solver Performance Summary", styles['Heading2']))
            
            solver_data = self._get_solver_comparison_data()
            if solver_data:
                # Prepare table data
                table_data = [['Solver', 'Success Rate', 'Avg Time (s)', 'Problems Solved']]
                
                for solver in solver_data[:10]:  # Limit to top 10 solvers
                    table_data.append([
                        solver.get('solver_name', 'Unknown'),
                        f"{solver.get('success_rate', 0):.1%}",
                        f"{solver.get('avg_solve_time', 0):.4f}",
                        str(solver.get('problems_solved', 0))
                    ])
                
                solver_table = Table(table_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch])
                solver_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(solver_table)
            else:
                story.append(Paragraph("No solver data available", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Footer
            story.append(Paragraph(
                "Generated automatically from benchmark results",
                styles['Normal']
            ))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"✅ PDF summary report generated: {output_path}")
            return output_path
            
        except ImportError as e:
            self.logger.warning(f"⚠️  PDF generation not available: {e}")
            # Create a text-based report as fallback
            return self._generate_text_report(output_path.with_suffix('.txt'))
        except Exception as e:
            self.logger.error(f"❌ Failed to generate PDF report: {e}")
            raise
    
    def export_all_formats(self, prefix: Optional[str] = None) -> Dict[str, Path]:
        """
        Export data in all available formats.
        
        Args:
            prefix: Filename prefix for all exports
            
        Returns:
            Dictionary mapping format names to output paths
        """
        if prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"benchmark_export_{timestamp}"
        
        self.logger.info(f"Exporting all formats with prefix: {prefix}")
        
        results = {}
        
        try:
            # CSV exports
            results['solver_comparison_csv'] = self.export_solver_comparison_csv(f"{prefix}_solver_comparison.csv")
            results['problem_results_csv'] = self.export_problem_results_csv(f"{prefix}_problem_results.csv")
            
            # JSON export
            results['json_data'] = self.export_json_data(f"{prefix}_data.json", include_raw_results=True)
            
            # PDF report
            results['pdf_report'] = self.generate_summary_report_pdf(f"{prefix}_summary.pdf")
            
            self.logger.info(f"✅ All formats exported successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export all formats: {e}")
            raise
    
    def _get_solver_comparison_data(self) -> List[Dict[str, Any]]:
        """Get solver comparison statistics from database."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT 
                        solver_name,
                        COUNT(*) as total_problems,
                        SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) as problems_solved,
                        CAST(SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as success_rate,
                        AVG(solve_time) as avg_solve_time,
                        MIN(solve_time) as min_solve_time,
                        MAX(solve_time) as max_solve_time,
                        SUM(solve_time) as total_time,
                        AVG(CASE WHEN iterations IS NOT NULL THEN iterations END) as avg_iterations,
                        GROUP_CONCAT(DISTINCT problem_type) as problem_types_supported,
                        MAX(timestamp) as last_run
                    FROM benchmark_results 
                    GROUP BY solver_name
                    ORDER BY success_rate DESC, avg_solve_time ASC
                '''
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get solver comparison data: {e}")
            return []
    
    def _get_problem_results_data(self, solver_filter: Optional[str] = None,
                                 problem_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detailed problem results from database."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT 
                        benchmark_id, solver_name, problem_name, problem_type,
                        status, solve_time, objective_value, iterations,
                        duality_gap, n_variables, n_constraints, timestamp,
                        error_message
                    FROM benchmark_results 
                    WHERE 1=1
                '''
                
                params = []
                
                if solver_filter:
                    query += ' AND solver_name LIKE ?'
                    params.append(f'%{solver_filter}%')
                
                if problem_filter:
                    query += ' AND problem_name LIKE ?'
                    params.append(f'%{problem_filter}%')
                
                query += ' ORDER BY timestamp DESC'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get problem results data: {e}")
            return []
    
    def _get_problem_statistics_data(self) -> List[Dict[str, Any]]:
        """Get problem statistics from database."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT 
                        problem_name,
                        problem_type,
                        COUNT(*) as solver_attempts,
                        SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) as successful_solves,
                        CAST(SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as success_rate,
                        AVG(CASE WHEN status = 'optimal' THEN solve_time END) as avg_solve_time,
                        AVG(n_variables) as avg_variables,
                        AVG(n_constraints) as avg_constraints
                    FROM benchmark_results 
                    GROUP BY problem_name, problem_type
                    ORDER BY success_rate DESC
                '''
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get problem statistics data: {e}")
            return []
    
    def _get_all_results_data(self) -> List[Dict[str, Any]]:
        """Get all raw results from database."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM benchmark_results ORDER BY timestamp DESC')
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get all results data: {e}")
            return []
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        
        return {
            'platform': 'GitHub Actions',
            'python_version': '3.9+',
            'database_path': str(self.database_path),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def _get_export_metadata(self) -> Dict[str, Any]:
        """Get metadata about the export."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get counts
                cursor.execute('SELECT COUNT(*) FROM benchmark_results')
                total_results = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT solver_name) FROM benchmark_results')
                total_solvers = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT problem_name) FROM benchmark_results')
                total_problems = cursor.fetchone()[0]
                
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM benchmark_results')
                date_range = cursor.fetchone()
                
                return {
                    'generated_at': datetime.now().isoformat(),
                    'database_path': str(self.database_path),
                    'total_results': total_results,
                    'total_solvers': total_solvers,
                    'total_problems': total_problems,
                    'earliest_result': date_range[0],
                    'latest_result': date_range[1],
                    'export_config': asdict(self.config)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get export metadata: {e}")
            return {
                'generated_at': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _format_numerical_values(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Format numerical values in a data row."""
        
        formatted_row = row.copy()
        
        for key, value in row.items():
            if isinstance(value, float):
                if 'time' in key.lower() or 'rate' in key.lower():
                    formatted_row[key] = round(value, self.config.decimal_places)
                elif 'objective' in key.lower():
                    formatted_row[key] = f"{value:.{self.config.decimal_places}e}"
                else:
                    formatted_row[key] = round(value, self.config.decimal_places)
        
        return formatted_row
    
    def _generate_text_report(self, output_path: Path) -> Path:
        """Generate text-based report as PDF fallback."""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("OPTIMIZATION SOLVER BENCHMARK REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Metadata
                metadata = self._get_export_metadata()
                f.write("Report Information:\n")
                f.write(f"Generated: {metadata.get('generated_at', 'Unknown')}\n")
                f.write(f"Total Results: {metadata.get('total_results', 0)}\n")
                f.write(f"Total Solvers: {metadata.get('total_solvers', 0)}\n")
                f.write(f"Total Problems: {metadata.get('total_problems', 0)}\n\n")
                
                # Solver comparison
                f.write("Solver Performance Summary:\n")
                f.write("-" * 30 + "\n")
                
                solver_data = self._get_solver_comparison_data()
                for solver in solver_data[:10]:
                    f.write(f"Solver: {solver.get('solver_name', 'Unknown')}\n")
                    f.write(f"  Success Rate: {solver.get('success_rate', 0):.1%}\n")
                    f.write(f"  Avg Time: {solver.get('avg_solve_time', 0):.4f}s\n")
                    f.write(f"  Problems Solved: {solver.get('problems_solved', 0)}\n\n")
                
                f.write("\nGenerated automatically from benchmark results\n")
            
            self.logger.info(f"✅ Text report generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate text report: {e}")
            raise


def main():
    """Command-line interface for data export."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Export benchmark data in various formats")
    parser.add_argument("--format", choices=["csv", "json", "pdf", "all"], default="all",
                       help="Export format")
    parser.add_argument("--output-dir", default="exports", help="Output directory")
    parser.add_argument("--prefix", help="Filename prefix")
    parser.add_argument("--solver-filter", help="Filter by solver name")
    parser.add_argument("--problem-filter", help="Filter by problem name")
    parser.add_argument("--database", help="Database path")
    
    args = parser.parse_args()
    
    # Create exporter
    config = ExportConfig(output_directory=args.output_dir)
    exporter = DataExporter(database_path=args.database, config=config)
    
    try:
        if args.format == "csv":
            exporter.export_solver_comparison_csv()
            exporter.export_problem_results_csv(
                solver_filter=args.solver_filter,
                problem_filter=args.problem_filter
            )
        elif args.format == "json":
            exporter.export_json_data(include_raw_results=True)
        elif args.format == "pdf":
            exporter.generate_summary_report_pdf()
        elif args.format == "all":
            results = exporter.export_all_formats(prefix=args.prefix)
            print("📊 Export Summary:")
            for format_name, path in results.items():
                print(f"  {format_name}: {path}")
        
        print("✅ Export completed successfully!")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()