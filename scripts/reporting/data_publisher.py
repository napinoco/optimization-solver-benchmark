"""
Data Publishing Infrastructure
==============================

Core data publishing system for benchmark results.
Creates clean JSON/CSV exports for external consumption.
"""

import json
import csv
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("data_publisher")


class DataPublisher:
    """Publishes benchmark results as clean, accessible data files."""
    
    def __init__(self, db_path: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Initialize data publisher.
        
        Args:
            db_path: Path to SQLite database file
            output_dir: Directory to save published data files
        """
        if db_path is None:
            db_path = project_root / "database" / "results.db"
        if output_dir is None:
            output_dir = project_root / "docs" / "data"
        
        self.db_path = str(db_path)
        self.output_dir = Path(output_dir)
        self.logger = get_logger("data_publisher")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def publish_all_data(self) -> bool:
        """
        Publish all benchmark data in standard formats.
        
        Returns:
            True if successful, False otherwise
        """
        
        self.logger.info("Publishing benchmark data...")
        
        try:
            # Generate all data files
            self._publish_results_json()
            self._publish_summary_json()
            self._publish_metadata_json()
            self._publish_results_csv()
            
            self.logger.info("✅ All data published successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to publish data: {e}")
            return False
    
    def _publish_results_json(self) -> None:
        """Publish complete results as JSON."""
        
        results_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "format_version": "1.0",
                "description": "Complete benchmark results from optimization solver comparison"
            },
            "results": self._get_all_results()
        }
        
        output_file = self.output_dir / "results.json"
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 Published results.json ({len(results_data['results'])} results)")
    
    def _publish_summary_json(self) -> None:
        """Publish summary statistics as JSON."""
        
        summary_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "format_version": "1.0",
                "description": "Summary statistics and solver comparison"
            },
            "summary": self._get_summary_statistics(),
            "solver_comparison": self._get_solver_comparison(),
            "problem_statistics": self._get_problem_statistics()
        }
        
        output_file = self.output_dir / "summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        self.logger.info("📊 Published summary.json")
    
    def _publish_metadata_json(self) -> None:
        """Publish problem and solver metadata as JSON."""
        
        metadata = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "format_version": "1.0",
                "description": "Problem and solver metadata"
            },
            "solvers": self._get_solver_metadata(),
            "problems": self._get_problem_metadata(),
            "environments": self._get_environment_info()
        }
        
        output_file = self.output_dir / "metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info("📋 Published metadata.json")
    
    def _publish_results_csv(self) -> None:
        """Publish results as CSV for spreadsheet analysis."""
        
        results = self._get_all_results()
        
        if not results:
            self.logger.warning("No results to export to CSV")
            return
        
        # Flatten results for CSV
        csv_data = []
        for result in results:
            row = {
                'benchmark_id': result.get('benchmark_id'),
                'solver_name': result.get('solver_name'),
                'problem_name': result.get('problem_name'),
                'problem_type': result.get('problem_type'),
                'solve_time': result.get('solve_time'),
                'status': result.get('status'),
                'objective_value': result.get('objective_value'),
                'iterations': result.get('iterations'),
                'duality_gap': result.get('duality_gap'),
                'timestamp': result.get('timestamp')
            }
            csv_data.append(row)
        
        output_file = self.output_dir / "results.csv"
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        self.logger.info(f"📊 Published results.csv ({len(csv_data)} rows)")
    
    def _get_all_results(self) -> List[Dict[str, Any]]:
        """Get all benchmark results from database."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get comprehensive results with problem information
                query = """
                    SELECT 
                        r.id,
                        r.benchmark_id,
                        r.solver_name,
                        r.problem_name,
                        COALESCE(p.problem_class, 'Unknown') as problem_type,
                        r.solve_time,
                        r.status,
                        r.objective_value,
                        r.duality_gap,
                        r.iterations,
                        r.error_message,
                        b.timestamp as benchmark_timestamp,
                        p.metadata
                    FROM results r
                    LEFT JOIN problems p ON r.problem_name = p.name
                    LEFT JOIN benchmarks b ON r.benchmark_id = b.id
                    ORDER BY r.benchmark_id DESC, r.solver_name, r.problem_name
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                
                columns = [description[0] for description in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    
                    # Extract structure info from metadata JSON
                    metadata_json = result.get('metadata')
                    if metadata_json:
                        try:
                            import json
                            metadata = json.loads(metadata_json)
                            
                            # Get structure analysis if available
                            structure = metadata.get('structure', {})
                            if structure:
                                result['n_variables'] = structure.get('variables', 0)
                                result['n_constraints'] = structure.get('constraints', 0)
                            else:
                                result['n_variables'] = 0
                                result['n_constraints'] = 0
                            
                            # Get library source
                            result['library_source'] = metadata.get('source', None)
                            result['difficulty_level'] = 'Unknown'  # Could calculate from structure
                            
                        except (json.JSONDecodeError, TypeError):
                            result['n_variables'] = 0
                            result['n_constraints'] = 0
                            result['library_source'] = None
                            result['difficulty_level'] = 'Unknown'
                    else:
                        result['n_variables'] = 0
                        result['n_constraints'] = 0
                        result['library_source'] = None
                        result['difficulty_level'] = 'Unknown'
                    
                    # Remove metadata field from final result to keep it clean
                    result.pop('metadata', None)
                    
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get results: {e}")
            return []
    
    def _get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_results,
                        COUNT(DISTINCT solver_name) as total_solvers,
                        COUNT(DISTINCT problem_name) as total_problems,
                        COUNT(DISTINCT benchmark_id) as total_benchmarks,
                        AVG(CASE WHEN status = 'optimal' THEN 1.0 ELSE 0.0 END) as overall_success_rate,
                        AVG(solve_time) as avg_solve_time,
                        MIN(solve_time) as min_solve_time,
                        MAX(solve_time) as max_solve_time
                    FROM results 
                    WHERE solve_time IS NOT NULL
                """)
                
                overall_stats = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))
                
                # Problem type distribution
                cursor.execute("""
                    SELECT 
                        COALESCE(p.problem_class, 'Unknown') as problem_type,
                        COUNT(*) as count
                    FROM results r
                    LEFT JOIN problems p ON r.problem_name = p.name
                    GROUP BY problem_type
                    ORDER BY count DESC
                """)
                
                problem_type_dist = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "overall": overall_stats,
                    "problem_type_distribution": problem_type_dist,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get summary statistics: {e}")
            return {}
    
    def _get_solver_comparison(self) -> Dict[str, Any]:
        """Get solver comparison data."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        solver_name,
                        COUNT(*) as problems_attempted,
                        SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) as problems_solved,
                        AVG(CASE WHEN status = 'optimal' THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(solve_time) as avg_solve_time,
                        MIN(solve_time) as best_time,
                        MAX(solve_time) as worst_time,
                        NULL as median_time
                    FROM results 
                    WHERE solve_time IS NOT NULL
                    GROUP BY solver_name
                    ORDER BY success_rate DESC, avg_solve_time ASC
                """)
                
                solver_stats = []
                for row in cursor.fetchall():
                    solver_data = {
                        "solver_name": row[0],
                        "problems_attempted": row[1],
                        "problems_solved": row[2],
                        "success_rate": row[3],
                        "avg_solve_time": row[4],
                        "best_time": row[5],
                        "worst_time": row[6],
                        "median_time": row[7]
                    }
                    solver_stats.append(solver_data)
                
                return {"solvers": solver_stats}
                
        except Exception as e:
            self.logger.error(f"Failed to get solver comparison: {e}")
            return {}
    
    def _get_problem_statistics(self) -> Dict[str, Any]:
        """Get problem-wise statistics."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        r.problem_name,
                        COALESCE(p.problem_class, 'Unknown') as problem_type,
                        COUNT(*) as solver_attempts,
                        SUM(CASE WHEN r.status = 'optimal' THEN 1 ELSE 0 END) as successful_solves,
                        AVG(CASE WHEN r.status = 'optimal' THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(r.solve_time) as avg_solve_time,
                        p.metadata
                    FROM results r
                    LEFT JOIN problems p ON r.problem_name = p.name
                    WHERE r.solve_time IS NOT NULL
                    GROUP BY r.problem_name
                    ORDER BY success_rate DESC, avg_solve_time ASC
                """)
                
                problem_stats = []
                for row in cursor.fetchall():
                    # Extract structure info from metadata JSON
                    metadata_json = row[6]
                    n_variables = 0
                    n_constraints = 0
                    difficulty_level = 'Unknown'
                    library_source = None
                    
                    if metadata_json:
                        try:
                            import json
                            metadata = json.loads(metadata_json)
                            
                            # Get structure analysis if available
                            structure = metadata.get('structure', {})
                            if structure:
                                n_variables = structure.get('variables', 0)
                                n_constraints = structure.get('constraints', 0)
                            
                            # Get library source
                            library_source = metadata.get('source', None)
                            
                        except (json.JSONDecodeError, TypeError):
                            pass
                    
                    problem_data = {
                        "problem_name": row[0],
                        "problem_type": row[1],
                        "solver_attempts": row[2],
                        "successful_solves": row[3],
                        "success_rate": row[4],
                        "avg_solve_time": row[5],
                        "n_variables": n_variables,
                        "n_constraints": n_constraints,
                        "difficulty_level": difficulty_level,
                        "library_source": library_source
                    }
                    problem_stats.append(problem_data)
                
                return {"problems": problem_stats}
                
        except Exception as e:
            self.logger.error(f"Failed to get problem statistics: {e}")
            return {}
    
    def _get_solver_metadata(self) -> List[Dict[str, Any]]:
        """Get solver metadata."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT solver_name
                    FROM results
                    ORDER BY solver_name
                """)
                
                solvers = []
                for row in cursor.fetchall():
                    solver_name = row[0]
                    
                    # Determine solver type and backend
                    if "CVXPY" in solver_name:
                        backend = solver_name.replace(" (via CVXPY)", "")
                        solver_type = "CVXPY Backend"
                    elif solver_name in ["SciPy", "scipy"]:
                        backend = solver_name
                        solver_type = "SciPy Solver"
                    else:
                        backend = solver_name
                        solver_type = "Unknown"
                    
                    solver_info = {
                        "name": solver_name,
                        "type": solver_type,
                        "backend": backend,
                        "environment": "Python"
                    }
                    solvers.append(solver_info)
                
                return solvers
                
        except Exception as e:
            self.logger.error(f"Failed to get solver metadata: {e}")
            return []
    
    def _get_problem_metadata(self) -> List[Dict[str, Any]]:
        """Get problem metadata."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        p.name,
                        p.problem_class,
                        p.metadata
                    FROM problems p
                    ORDER BY p.name
                """)
                
                problems = []
                for row in cursor.fetchall():
                    # Extract structure info from metadata JSON
                    metadata_json = row[2]
                    n_variables = 0
                    n_constraints = 0
                    difficulty_level = 'Unknown'
                    complexity_score = 1.0
                    library_source = None
                    
                    if metadata_json:
                        try:
                            import json
                            metadata = json.loads(metadata_json)
                            
                            # Get structure analysis if available
                            structure = metadata.get('structure', {})
                            if structure:
                                n_variables = structure.get('variables', 0)
                                n_constraints = structure.get('constraints', 0)
                            
                            # Get library source
                            library_source = metadata.get('source', None)
                            
                        except (json.JSONDecodeError, TypeError):
                            pass
                    
                    problem_info = {
                        "name": row[0],
                        "class": row[1],
                        "type": row[1],  # Use problem_class as type
                        "n_variables": n_variables,
                        "n_constraints": n_constraints,
                        "difficulty_level": difficulty_level,
                        "complexity_score": complexity_score,
                        "library_source": library_source
                    }
                    problems.append(problem_info)
                
                return problems
                
        except Exception as e:
            self.logger.error(f"Failed to get problem metadata: {e}")
            return []
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information from latest benchmark session."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get environment info from the most recent benchmark
                cursor.execute("""
                    SELECT environment_info, timestamp
                    FROM benchmarks
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                if result and result[0]:
                    # Parse the JSON environment info
                    env_data = json.loads(result[0])
                    
                    # Extract key information for metadata
                    os_info = env_data.get('os', {})
                    python_info = env_data.get('python', {})
                    
                    # Create enhanced OS description with Ubuntu version
                    os_description = os_info.get('ubuntu_version', 
                                               os_info.get('lsb_description',
                                                         f"{os_info.get('system', 'Linux')} {os_info.get('release', 'Unknown')}"))
                    
                    return {
                        "platform": os_info.get('platform', 'Unknown'),
                        "operating_system": os_description,
                        "operating_system_details": {
                            "system": os_info.get('system'),
                            "release": os_info.get('release'),
                            "version": os_info.get('version'),
                            "machine": os_info.get('machine'),
                            "architecture": os_info.get('architecture'),
                            "ubuntu_version": os_info.get('ubuntu_version'),
                            "version_id": os_info.get('version_id')
                        },
                        "python_version": python_info.get('version', 'Unknown'),
                        "python_implementation": python_info.get('implementation', 'Unknown'),
                        "cpu_info": {
                            "processor": env_data.get('cpu', {}).get('processor'),
                            "cpu_count": env_data.get('cpu', {}).get('cpu_count'),
                            "cpu_count_physical": env_data.get('cpu', {}).get('cpu_count_physical')
                        },
                        "memory_info": {
                            "total_gb": env_data.get('memory', {}).get('total_gb'),
                            "available_gb": env_data.get('memory', {}).get('available_gb')
                        },
                        "timezone_info": {
                            "timezone_name": env_data.get('timezone', {}).get('timezone_name'),
                            "system_timezone": env_data.get('timezone', {}).get('system_timezone'),
                            "timedatectl_timezone": env_data.get('timezone', {}).get('timedatectl_timezone'),
                            "utc_offset_hours": env_data.get('timezone', {}).get('utc_offset_hours'),
                            "current_time_utc": env_data.get('timezone', {}).get('current_time_utc'),
                            "current_time_local": env_data.get('timezone', {}).get('current_time_local'),
                            "daylight_saving": env_data.get('timezone', {}).get('daylight_saving')
                        },
                        "primary_solver_framework": "CVXPY",
                        "supported_problem_types": ["LP", "QP", "SOCP", "SDP"],
                        "last_updated": result[1],
                        "collection_timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            self.logger.warning(f"Could not load environment info from database: {e}")
        
        # Fallback to basic info if database lookup fails
        return {
            "platform": "GitHub Actions",
            "operating_system": "Ubuntu (version not available)",
            "python_version": "3.12",
            "primary_solver_framework": "CVXPY", 
            "supported_problem_types": ["LP", "QP", "SOCP", "SDP"],
            "last_updated": datetime.now().isoformat(),
            "note": "Environment details not available - using fallback values"
        }


def publish_benchmark_data():
    """Main function to publish all benchmark data."""
    
    print("📊 Publishing Benchmark Data...")
    print("=" * 50)
    
    publisher = DataPublisher()
    success = publisher.publish_all_data()
    
    if success:
        print("✅ Data publishing completed successfully!")
        print("\n📄 Published Files:")
        print("  • docs/data/results.json - Complete benchmark results")
        print("  • docs/data/summary.json - Summary statistics and comparisons")
        print("  • docs/data/metadata.json - Solver and problem metadata")
        print("  • docs/data/results.csv - Results in CSV format")
        
        print("\n🌐 Access via GitHub Pages:")
        print("  • https://[username].github.io/[repo]/data/results.json")
        print("  • https://[username].github.io/[repo]/data/summary.json")
        print("  • https://[username].github.io/[repo]/data/metadata.json")
        
    else:
        print("❌ Data publishing failed. Check logs for details.")


if __name__ == "__main__":
    publish_benchmark_data()