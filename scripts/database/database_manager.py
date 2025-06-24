"""
Database Manager for Re-Architected Benchmark System

Handles all database operations for the simplified single-table design.
Provides simple database operations with error handling for the new schema.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class DatabaseManager:
    """Manages database operations for benchmark results"""
    
    def __init__(self, db_path: str = "database/results.db"):
        """
        Initialize database manager with path to SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self.ensure_schema()
    
    def ensure_schema(self) -> None:
        """Create database schema if it doesn't exist"""
        try:
            schema_path = Path("scripts/database/schema.sql")
            
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Replace CREATE TABLE with CREATE TABLE IF NOT EXISTS for graceful handling
            schema_sql = schema_sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
            schema_sql = schema_sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS")
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_sql)
                conn.commit()
                
            self.logger.info(f"Database schema ensured at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to ensure database schema: {e}")
            raise
    
    def store_result(self, 
                    solver_name: str, 
                    solver_version: str,
                    problem_library: str, 
                    problem_name: str, 
                    problem_type: str,
                    environment_info: Dict[str, Any], 
                    commit_hash: str,
                    solve_time: Optional[float] = None,
                    status: Optional[str] = None,
                    primal_objective_value: Optional[float] = None,
                    dual_objective_value: Optional[float] = None,
                    duality_gap: Optional[float] = None,
                    primal_infeasibility: Optional[float] = None,
                    dual_infeasibility: Optional[float] = None,
                    iterations: Optional[int] = None,
                    memo: Optional[str] = None) -> None:
        """
        Store single benchmark result (append-only).
        
        Args:
            solver_name: Name of the solver
            solver_version: Version string of the solver
            problem_library: Library source (internal, DIMACS, SDPLIB)
            problem_name: Name of the problem
            problem_type: Type of problem (LP, QP, SOCP, SDP)
            environment_info: Dictionary with system information
            commit_hash: Git commit hash
            solve_time: Execution time in seconds
            status: Solver status (optimal, infeasible, error, etc.)
            primal_objective_value: Primal objective value
            dual_objective_value: Dual objective value
            duality_gap: Duality gap
            primal_infeasibility: Primal infeasibility measure
            dual_infeasibility: Dual infeasibility measure
            iterations: Number of solver iterations
            memo: Optional user memo/notes for this result
        """
        try:
            # Convert environment_info to JSON string
            environment_json = json.dumps(environment_info, sort_keys=True)
            
            insert_sql = """
            INSERT INTO results (
                solver_name, solver_version, problem_library, problem_name, problem_type,
                environment_info, commit_hash, solve_time, status,
                primal_objective_value, dual_objective_value, duality_gap,
                primal_infeasibility, dual_infeasibility, iterations, memo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(insert_sql, (
                    solver_name, solver_version, problem_library, problem_name, problem_type,
                    environment_json, commit_hash, solve_time, status,
                    primal_objective_value, dual_objective_value, duality_gap,
                    primal_infeasibility, dual_infeasibility, iterations, memo
                ))
                conn.commit()
                
            self.logger.debug(f"Stored result: {solver_name} on {problem_name}")
            
        except sqlite3.IntegrityError as e:
            # Handle duplicate entries gracefully
            self.logger.warning(f"Duplicate result ignored: {solver_name} on {problem_name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to store result: {e}")
            raise
    
    def get_latest_results(self, commit_hash: Optional[str] = None, environment_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get latest results for reporting.
        
        Args:
            commit_hash: Specific git commit hash (if None, uses latest)
            environment_info: Specific environment info (if None, uses latest)
            
        Returns:
            List of latest results as dictionaries
        """
        try:
            # Convert environment_info to JSON string if provided
            environment_json = None
            if environment_info:
                environment_json = json.dumps(environment_info, sort_keys=True)
            
            # Build query for latest results
            if commit_hash and environment_json:
                # Query for specific commit and environment
                query = """
                SELECT * FROM results 
                WHERE commit_hash = ? AND environment_info = ?
                ORDER BY timestamp DESC
                """
                params = (commit_hash, environment_json)
            else:
                # Query for latest results (most recent commit_hash and environment)
                query = """
                SELECT r1.* FROM results r1
                INNER JOIN (
                    SELECT solver_name, problem_name, MAX(timestamp) as max_timestamp
                    FROM results 
                    GROUP BY solver_name, problem_name
                ) r2 ON r1.solver_name = r2.solver_name 
                     AND r1.problem_name = r2.problem_name 
                     AND r1.timestamp = r2.max_timestamp
                ORDER BY r1.problem_library, r1.problem_name, r1.solver_name
                """
                params = ()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable dictionary-like access
                cursor = conn.execute(query, params)
                results = []
                
                for row in cursor.fetchall():
                    result_dict = dict(row)
                    # Parse environment_info JSON back to dict
                    if result_dict['environment_info']:
                        result_dict['environment_info'] = json.loads(result_dict['environment_info'])
                    results.append(result_dict)
                
                self.logger.info(f"Retrieved {len(results)} latest results")
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get latest results: {e}")
            raise
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection for advanced queries"""
        return sqlite3.connect(self.db_path)