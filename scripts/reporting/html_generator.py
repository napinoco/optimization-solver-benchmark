"""
HTML Report Generator

Generates static HTML reports from benchmark results stored in the database.
Uses Jinja2 templates to create comprehensive benchmark reports.
"""

import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

try:
    from jinja2 import Environment, FileSystemLoader, Template
except ImportError:
    raise ImportError("jinja2 is required for HTML report generation. Install with: pip install jinja2")

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SolverStats:
    """Statistics for a single solver"""
    name: str
    total_runs: int
    successful_runs: int
    success_rate: float
    avg_time: float
    min_time: float
    max_time: float
    unique_problems_solved: int
    total_time: float


@dataclass
class ProblemStats:
    """Statistics for a single problem"""
    name: str
    problem_type: str
    total_attempts: int
    successful_attempts: int
    success_rate: float
    solver_results: Dict[str, Any]  # solver_name -> result info


@dataclass
class SummaryStats:
    """Overall benchmark summary statistics"""
    total_runs: int
    total_solvers: int
    total_problems: int
    overall_success_rate: float
    avg_run_time: float
    total_time: float


class HTMLReportGenerator:
    """Generates HTML reports from benchmark database results"""
    
    def __init__(self, database_path: str, templates_dir: str, output_dir: str = "docs"):
        """
        Initialize the HTML report generator
        
        Args:
            database_path: Path to the SQLite database
            templates_dir: Directory containing Jinja2 templates
            output_dir: Directory to save generated HTML files
        """
        self.database_path = database_path
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        logger.info(f"HTML generator initialized: DB={database_path}, Templates={templates_dir}, Output={output_dir}")
    
    def _get_database_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _query_solver_stats(self) -> Dict[str, SolverStats]:
        """Query and calculate solver statistics"""
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Get solver statistics - using solver_name directly from results table
            cursor.execute("""
                SELECT 
                    r.solver_name,
                    COUNT(r.id) as total_runs,
                    SUM(CASE WHEN r.status = 'optimal' THEN 1 ELSE 0 END) as successful_runs,
                    AVG(CASE WHEN r.status = 'optimal' THEN r.solve_time ELSE NULL END) as avg_time,
                    MIN(CASE WHEN r.status = 'optimal' THEN r.solve_time ELSE NULL END) as min_time,
                    MAX(CASE WHEN r.status = 'optimal' THEN r.solve_time ELSE NULL END) as max_time,
                    COUNT(DISTINCT CASE WHEN r.status = 'optimal' THEN r.problem_name ELSE NULL END) as unique_problems,
                    SUM(r.solve_time) as total_time
                FROM results r
                GROUP BY r.solver_name
                ORDER BY successful_runs DESC, avg_time ASC
            """)
            
            solver_stats = {}
            for row in cursor.fetchall():
                success_rate = row['successful_runs'] / row['total_runs'] if row['total_runs'] > 0 else 0.0
                
                stats = SolverStats(
                    name=row['solver_name'],
                    total_runs=row['total_runs'] or 0,
                    successful_runs=row['successful_runs'] or 0,
                    success_rate=success_rate,
                    avg_time=row['avg_time'] or 0.0,
                    min_time=row['min_time'] or 0.0,
                    max_time=row['max_time'] or 0.0,
                    unique_problems_solved=row['unique_problems'] or 0,
                    total_time=row['total_time'] or 0.0
                )
                solver_stats[row['solver_name']] = stats
            
            return solver_stats
    
    def _query_problem_stats(self) -> Dict[str, ProblemStats]:
        """Query and calculate problem statistics"""
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            
            # Get problem statistics - using problem_name directly from results table
            cursor.execute("""
                SELECT 
                    r.problem_name,
                    p.problem_class as problem_type,
                    COUNT(r.id) as total_attempts,
                    SUM(CASE WHEN r.status = 'optimal' THEN 1 ELSE 0 END) as successful_attempts
                FROM results r
                LEFT JOIN problems p ON p.name = r.problem_name
                GROUP BY r.problem_name, p.problem_class
                ORDER BY successful_attempts DESC, r.problem_name ASC
            """)
            
            problem_stats = {}
            for row in cursor.fetchall():
                success_rate = row['successful_attempts'] / row['total_attempts'] if row['total_attempts'] > 0 else 0.0
                
                # Get solver-specific results for this problem
                cursor.execute("""
                    SELECT 
                        r.solver_name,
                        r.status,
                        r.solve_time,
                        r.objective_value
                    FROM results r
                    WHERE r.problem_name = ?
                    ORDER BY r.solve_time ASC
                """, (row['problem_name'],))
                
                solver_results = {}
                for result_row in cursor.fetchall():
                    solver_results[result_row['solver_name']] = {
                        'status': result_row['status'],
                        'solve_time': result_row['solve_time'],
                        'objective_value': result_row['objective_value']
                    }
                
                stats = ProblemStats(
                    name=row['problem_name'],
                    problem_type=row['problem_type'] or 'Unknown',
                    total_attempts=row['total_attempts'] or 0,
                    successful_attempts=row['successful_attempts'] or 0,
                    success_rate=success_rate,
                    solver_results=solver_results
                )
                problem_stats[row['problem_name']] = stats
            
            return problem_stats
    
    def _query_recent_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Query recent benchmark results"""
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    b.timestamp,
                    r.solver_name,
                    r.problem_name,
                    r.status,
                    r.solve_time,
                    r.objective_value
                FROM results r
                JOIN benchmarks b ON r.benchmark_id = b.id
                ORDER BY b.timestamp DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'timestamp': row['timestamp'],
                    'solver_name': row['solver_name'],
                    'problem_name': row['problem_name'],
                    'status': row['status'],
                    'solve_time': row['solve_time'],
                    'objective_value': row['objective_value']
                })
            
            return results
    
    def _query_environment_info(self) -> Optional[Dict[str, Any]]:
        """Query latest environment information"""
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT environment_info
                FROM benchmarks
                WHERE environment_info IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row and row['environment_info']:
                try:
                    return json.loads(row['environment_info'])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse environment info JSON")
                    return None
            
            return None
    
    def _calculate_summary_stats(self, solver_stats: Dict[str, SolverStats]) -> SummaryStats:
        """Calculate overall summary statistics"""
        if not solver_stats:
            return SummaryStats(0, 0, 0, 0.0, 0.0, 0.0)
        
        total_runs = sum(stats.total_runs for stats in solver_stats.values())
        successful_runs = sum(stats.successful_runs for stats in solver_stats.values())
        total_time = sum(stats.total_time for stats in solver_stats.values())
        
        # Get unique problem count
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT id) FROM problems")
            total_problems = cursor.fetchone()[0]
        
        success_rate = successful_runs / total_runs if total_runs > 0 else 0.0
        avg_run_time = total_time / total_runs if total_runs > 0 else 0.0
        
        return SummaryStats(
            total_runs=total_runs,
            total_solvers=len(solver_stats),
            total_problems=total_problems,
            overall_success_rate=success_rate,
            avg_run_time=avg_run_time,
            total_time=total_time
        )
    
    def generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        logger.info("Generating dashboard HTML...")
        
        # Collect all data
        solver_stats = self._query_solver_stats()
        problem_stats = self._query_problem_stats()
        recent_results = self._query_recent_results()
        environment_info = self._query_environment_info()
        summary = self._calculate_summary_stats(solver_stats)
        
        # Convert dataclasses to dictionaries for template rendering
        solver_stats_dict = {name: asdict(stats) for name, stats in solver_stats.items()}
        problem_stats_dict = {name: asdict(stats) for name, stats in problem_stats.items()}
        
        # Load and render template
        template = self.jinja_env.get_template('dashboard.html')
        html_content = template.render(
            solver_stats=solver_stats_dict,
            problem_stats=problem_stats_dict,
            recent_results=recent_results,
            environment_info=environment_info,
            summary=asdict(summary),
            last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content
    
    def generate_solver_comparison_html(self) -> str:
        """Generate solver comparison HTML"""
        logger.info("Generating solver comparison HTML...")
        
        solver_stats = self._query_solver_stats()
        problem_stats = self._query_problem_stats()
        
        # Convert dataclasses to dictionaries for template rendering
        solver_stats_dict = {name: asdict(stats) for name, stats in solver_stats.items()}
        problem_stats_dict = {name: asdict(stats) for name, stats in problem_stats.items()}
        
        template = self.jinja_env.get_template('solver_comparison.html')
        html_content = template.render(
            solver_stats=solver_stats_dict,
            problem_stats=problem_stats_dict,
            last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content
    
    def generate_problem_analysis_html(self) -> str:
        """Generate problem analysis HTML"""
        logger.info("Generating problem analysis HTML...")
        
        problem_stats = self._query_problem_stats()
        
        # Convert dataclasses to dictionaries for template rendering
        problem_stats_dict = {name: asdict(stats) for name, stats in problem_stats.items()}
        
        # Create problem info for characteristics section
        problem_info = {}
        for name, stats in problem_stats.items():
            problem_info[name] = {
                'type': stats.problem_type,
                'attempts': stats.total_attempts,
                'success_rate': stats.success_rate
            }
        
        template = self.jinja_env.get_template('problem_analysis.html')
        html_content = template.render(
            problem_stats=problem_stats_dict,
            problem_info=problem_info,
            last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content
    
    def generate_environment_info_html(self) -> str:
        """Generate environment information HTML"""
        logger.info("Generating environment info HTML...")
        
        environment_info = self._query_environment_info()
        
        # Query benchmark sessions for history
        with self._get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    b.id,
                    b.timestamp,
                    COUNT(r.id) as total_results
                FROM benchmarks b
                LEFT JOIN results r ON b.id = r.benchmark_id
                GROUP BY b.id, b.timestamp
                ORDER BY b.timestamp DESC
                LIMIT 10
            """)
            
            benchmark_sessions = []
            for row in cursor.fetchall():
                benchmark_sessions.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'total_results': row['total_results'] or 0,
                    'environment_summary': 'Python benchmark session'
                })
        
        template = self.jinja_env.get_template('environment_info.html')
        html_content = template.render(
            environment_info=environment_info,
            benchmark_sessions=benchmark_sessions,
            last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return html_content
    
    def save_html_file(self, filename: str, content: str) -> str:
        """Save HTML content to a file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"HTML file saved: {output_path}")
        return str(output_path)
    
    def generate_all_reports(self) -> Dict[str, str]:
        """Generate all HTML reports and save to files"""
        logger.info("Generating all HTML reports...")
        
        reports = {}
        
        # Generate each report
        reports['dashboard'] = self.generate_dashboard_html()
        reports['solver_comparison'] = self.generate_solver_comparison_html()
        reports['problem_analysis'] = self.generate_problem_analysis_html()
        reports['environment_info'] = self.generate_environment_info_html()
        
        # Save all reports
        saved_files = {}
        saved_files['dashboard'] = self.save_html_file('index.html', reports['dashboard'])
        saved_files['solver_comparison'] = self.save_html_file('solver_comparison.html', reports['solver_comparison'])
        saved_files['problem_analysis'] = self.save_html_file('problem_analysis.html', reports['problem_analysis'])
        saved_files['environment_info'] = self.save_html_file('environment_info.html', reports['environment_info'])
        
        logger.info(f"All reports generated successfully. Files saved to: {self.output_dir}")
        return saved_files


def main():
    """Main function for testing the HTML generator"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python html_generator.py <database_path> [templates_dir] [output_dir]")
        sys.exit(1)
    
    database_path = sys.argv[1]
    templates_dir = sys.argv[2] if len(sys.argv) > 2 else "templates"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "docs"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        generator = HTMLReportGenerator(database_path, templates_dir, output_dir)
        saved_files = generator.generate_all_reports()
        
        print("HTML reports generated successfully:")
        for report_type, file_path in saved_files.items():
            print(f"  {report_type}: {file_path}")
            
    except Exception as e:
        logger.error(f"Failed to generate HTML reports: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()