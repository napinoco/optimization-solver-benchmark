"""
Simple Benchmark API
===================

Provides RESTful endpoints for accessing benchmark data.
Lightweight Flask-based implementation for development and testing.
"""

import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None

from scripts.utils.logger import get_logger

logger = get_logger("benchmark_api")


class BenchmarkAPI:
    """
    Simple RESTful API for benchmark data access.
    
    Endpoints:
    - GET /api/solvers - List all solvers
    - GET /api/problems - List all problems  
    - GET /api/results - Get benchmark results (with filtering)
    - GET /api/summary - Get summary statistics
    - GET /api/health - Health check
    """
    
    def __init__(self, database_path: Optional[str] = None, debug: bool = False):
        """
        Initialize API server.
        
        Args:
            database_path: Path to SQLite database
            debug: Enable debug mode
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for API server. Install with: pip install flask flask-cors")
        
        self.logger = get_logger("benchmark_api")
        
        # Set up database connection
        if database_path is None:
            database_path = project_root / "benchmark_results.db"
        
        self.database_path = Path(database_path)
        self.debug = debug
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for development
        
        # Register routes
        self._register_routes()
        
        self.logger.info(f"Benchmark API initialized with database: {self.database_path}")
    
    def _register_routes(self):
        """Register API endpoints."""
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'database': str(self.database_path),
                'database_exists': self.database_path.exists()
            })
        
        @self.app.route('/api/solvers', methods=['GET'])
        def get_solvers():
            """Get list of all solvers."""
            try:
                solvers = self._get_solvers_data()
                return jsonify({
                    'solvers': solvers,
                    'count': len(solvers)
                })
            except Exception as e:
                self.logger.error(f"Error getting solvers: {e}")
                abort(500)
        
        @self.app.route('/api/problems', methods=['GET'])
        def get_problems():
            """Get list of all problems."""
            try:
                problems = self._get_problems_data()
                return jsonify({
                    'problems': problems,
                    'count': len(problems)
                })
            except Exception as e:
                self.logger.error(f"Error getting problems: {e}")
                abort(500)
        
        @self.app.route('/api/results', methods=['GET'])
        def get_results():
            """Get benchmark results with optional filtering."""
            try:
                # Parse query parameters
                solver_name = request.args.get('solver')
                problem_name = request.args.get('problem')
                problem_type = request.args.get('type')
                status = request.args.get('status')
                limit = request.args.get('limit', type=int, default=100)
                offset = request.args.get('offset', type=int, default=0)
                
                # Validate limit
                if limit > 1000:
                    limit = 1000
                if limit < 1:
                    limit = 100
                
                results = self._get_results_data(
                    solver_name=solver_name,
                    problem_name=problem_name,
                    problem_type=problem_type,
                    status=status,
                    limit=limit,
                    offset=offset
                )
                
                return jsonify({
                    'results': results,
                    'count': len(results),
                    'limit': limit,
                    'offset': offset,
                    'filters': {
                        'solver': solver_name,
                        'problem': problem_name,
                        'type': problem_type,
                        'status': status
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Error getting results: {e}")
                abort(500)
        
        @self.app.route('/api/summary', methods=['GET'])
        def get_summary():
            """Get summary statistics."""
            try:
                summary = self._get_summary_data()
                return jsonify(summary)
            except Exception as e:
                self.logger.error(f"Error getting summary: {e}")
                abort(500)
        
        @self.app.route('/api/solver/<solver_name>', methods=['GET'])
        def get_solver_details(solver_name: str):
            """Get detailed information about a specific solver."""
            try:
                details = self._get_solver_details(solver_name)
                if not details:
                    abort(404)
                return jsonify(details)
            except Exception as e:
                self.logger.error(f"Error getting solver details: {e}")
                abort(500)
        
        @self.app.route('/api/problem/<problem_name>', methods=['GET'])
        def get_problem_details(problem_name: str):
            """Get detailed information about a specific problem."""
            try:
                details = self._get_problem_details(problem_name)
                if not details:
                    abort(404)
                return jsonify(details)
            except Exception as e:
                self.logger.error(f"Error getting problem details: {e}")
                abort(500)
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def _get_solvers_data(self) -> List[Dict[str, Any]]:
        """Get list of all solvers with basic statistics."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT 
                        solver_name,
                        COUNT(*) as total_runs,
                        SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) as successful_runs,
                        CAST(SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as success_rate,
                        AVG(solve_time) as avg_solve_time,
                        GROUP_CONCAT(DISTINCT problem_type) as supported_types,
                        MAX(timestamp) as last_run
                    FROM benchmark_results 
                    GROUP BY solver_name
                    ORDER BY solver_name
                '''
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get solvers data: {e}")
            return []
    
    def _get_problems_data(self) -> List[Dict[str, Any]]:
        """Get list of all problems with basic statistics."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT 
                        problem_name,
                        problem_type,
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) as successful_solves,
                        CAST(SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as success_rate,
                        AVG(CASE WHEN status = 'optimal' THEN solve_time END) as avg_solve_time,
                        AVG(n_variables) as avg_variables,
                        AVG(n_constraints) as avg_constraints,
                        COUNT(DISTINCT solver_name) as solvers_tested
                    FROM benchmark_results 
                    GROUP BY problem_name, problem_type
                    ORDER BY problem_name
                '''
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get problems data: {e}")
            return []
    
    def _get_results_data(self, solver_name: Optional[str] = None,
                         problem_name: Optional[str] = None,
                         problem_type: Optional[str] = None,
                         status: Optional[str] = None,
                         limit: int = 100,
                         offset: int = 0) -> List[Dict[str, Any]]:
        """Get benchmark results with filtering and pagination."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = '''
                    SELECT 
                        id, benchmark_id, solver_name, problem_name, problem_type,
                        status, solve_time, objective_value, iterations,
                        duality_gap, n_variables, n_constraints, timestamp,
                        error_message
                    FROM benchmark_results 
                    WHERE 1=1
                '''
                
                params = []
                
                if solver_name:
                    query += ' AND solver_name = ?'
                    params.append(solver_name)
                
                if problem_name:
                    query += ' AND problem_name = ?'
                    params.append(problem_name)
                
                if problem_type:
                    query += ' AND problem_type = ?'
                    params.append(problem_type)
                
                if status:
                    query += ' AND status = ?'
                    params.append(status)
                
                query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get results data: {e}")
            return []
    
    def _get_summary_data(self) -> Dict[str, Any]:
        """Get overall summary statistics."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Overall counts
                cursor.execute('SELECT COUNT(*) FROM benchmark_results')
                total_results = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT solver_name) FROM benchmark_results')
                total_solvers = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT problem_name) FROM benchmark_results')
                total_problems = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT benchmark_id) FROM benchmark_results')
                total_benchmarks = cursor.fetchone()[0]
                
                # Success statistics
                cursor.execute("SELECT COUNT(*) FROM benchmark_results WHERE status = 'optimal'")
                successful_results = cursor.fetchone()[0]
                
                # Time statistics
                cursor.execute('SELECT AVG(solve_time), MIN(solve_time), MAX(solve_time) FROM benchmark_results')
                time_stats = cursor.fetchone()
                
                # Problem type distribution
                cursor.execute('''
                    SELECT problem_type, COUNT(*) as count 
                    FROM benchmark_results 
                    GROUP BY problem_type 
                    ORDER BY count DESC
                ''')
                
                problem_types = {}
                for row in cursor.fetchall():
                    problem_types[row[0]] = row[1]
                
                # Date range
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM benchmark_results')
                date_range = cursor.fetchone()
                
                return {
                    'overview': {
                        'total_results': total_results,
                        'total_solvers': total_solvers,
                        'total_problems': total_problems,
                        'total_benchmarks': total_benchmarks,
                        'successful_results': successful_results,
                        'success_rate': successful_results / total_results if total_results > 0 else 0
                    },
                    'performance': {
                        'avg_solve_time': time_stats[0],
                        'min_solve_time': time_stats[1],
                        'max_solve_time': time_stats[2]
                    },
                    'problem_type_distribution': problem_types,
                    'date_range': {
                        'earliest': date_range[0],
                        'latest': date_range[1]
                    },
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get summary data: {e}")
            return {'error': str(e)}
    
    def _get_solver_details(self, solver_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific solver."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Check if solver exists
                cursor.execute('SELECT COUNT(*) FROM benchmark_results WHERE solver_name = ?', (solver_name,))
                if cursor.fetchone()[0] == 0:
                    return None
                
                # Get solver statistics
                query = '''
                    SELECT 
                        solver_name,
                        COUNT(*) as total_runs,
                        SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) as successful_runs,
                        CAST(SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as success_rate,
                        AVG(solve_time) as avg_solve_time,
                        MIN(solve_time) as min_solve_time,
                        MAX(solve_time) as max_solve_time,
                        AVG(CASE WHEN iterations IS NOT NULL THEN iterations END) as avg_iterations,
                        GROUP_CONCAT(DISTINCT problem_type) as supported_types,
                        MIN(timestamp) as first_run,
                        MAX(timestamp) as last_run
                    FROM benchmark_results 
                    WHERE solver_name = ?
                    GROUP BY solver_name
                '''
                
                cursor.execute(query, (solver_name,))
                solver_stats = dict(cursor.fetchone())
                
                # Get recent results
                cursor.execute('''
                    SELECT problem_name, problem_type, status, solve_time, timestamp
                    FROM benchmark_results 
                    WHERE solver_name = ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (solver_name,))
                
                recent_results = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'solver_info': solver_stats,
                    'recent_results': recent_results
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get solver details: {e}")
            return None
    
    def _get_problem_details(self, problem_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific problem."""
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Check if problem exists
                cursor.execute('SELECT COUNT(*) FROM benchmark_results WHERE problem_name = ?', (problem_name,))
                if cursor.fetchone()[0] == 0:
                    return None
                
                # Get problem statistics
                query = '''
                    SELECT 
                        problem_name,
                        problem_type,
                        COUNT(*) as total_attempts,
                        SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) as successful_solves,
                        CAST(SUM(CASE WHEN status = 'optimal' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as success_rate,
                        AVG(CASE WHEN status = 'optimal' THEN solve_time END) as avg_solve_time,
                        MIN(CASE WHEN status = 'optimal' THEN solve_time END) as min_solve_time,
                        MAX(CASE WHEN status = 'optimal' THEN solve_time END) as max_solve_time,
                        AVG(n_variables) as avg_variables,
                        AVG(n_constraints) as avg_constraints,
                        COUNT(DISTINCT solver_name) as solvers_tested,
                        MIN(timestamp) as first_attempt,
                        MAX(timestamp) as last_attempt
                    FROM benchmark_results 
                    WHERE problem_name = ?
                    GROUP BY problem_name, problem_type
                '''
                
                cursor.execute(query, (problem_name,))
                problem_stats = dict(cursor.fetchone())
                
                # Get solver performance on this problem
                cursor.execute('''
                    SELECT solver_name, status, solve_time, iterations, timestamp
                    FROM benchmark_results 
                    WHERE problem_name = ?
                    ORDER BY timestamp DESC
                ''', (problem_name,))
                
                solver_results = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'problem_info': problem_stats,
                    'solver_results': solver_results
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get problem details: {e}")
            return None
    
    def run(self, host: str = '127.0.0.1', port: int = 5000):
        """
        Run the API server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.logger.info(f"Starting benchmark API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=self.debug)


def main():
    """Command-line interface for API server."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark data API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--database", help="Database path")
    
    args = parser.parse_args()
    
    try:
        api = BenchmarkAPI(database_path=args.database, debug=args.debug)
        api.run(host=args.host, port=args.port)
    except ImportError as e:
        print(f"❌ Cannot start API server: {e}")
        print("Install required dependencies: pip install flask flask-cors")
        sys.exit(1)
    except Exception as e:
        print(f"❌ API server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()