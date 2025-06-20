"""
Result Processor for Simplified Reporting
=========================================

Processes benchmark results from database for simplified HTML report generation.
Extracts latest results using commit_hash and environment_info with timestamp tiebreaker.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("result_processor")


@dataclass
class BenchmarkResult:
    """Single denormalized benchmark result model"""
    
    # Primary key
    id: Optional[int] = None
    
    # Solver information
    solver_name: str = ""
    solver_version: str = ""
    
    # Problem information
    problem_library: str = ""  # 'internal', 'DIMACS', 'SDPLIB'
    problem_name: str = ""
    problem_type: str = ""     # 'LP', 'QP', 'SOCP', 'SDP'
    
    # Environment and execution context
    environment_info: Dict[str, Any] = None
    commit_hash: str = ""
    timestamp: Optional[datetime] = None
    
    # Standardized solver results
    solve_time: Optional[float] = None
    status: Optional[str] = None
    primal_objective_value: Optional[float] = None
    dual_objective_value: Optional[float] = None
    duality_gap: Optional[float] = None
    primal_infeasibility: Optional[float] = None
    dual_infeasibility: Optional[float] = None
    iterations: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'solver_name': self.solver_name,
            'solver_version': self.solver_version,
            'problem_library': self.problem_library,
            'problem_name': self.problem_name,
            'problem_type': self.problem_type,
            'environment_info': self.environment_info,
            'commit_hash': self.commit_hash,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'solve_time': self.solve_time,
            'status': self.status,
            'primal_objective_value': self.primal_objective_value,
            'dual_objective_value': self.dual_objective_value,
            'duality_gap': self.duality_gap,
            'primal_infeasibility': self.primal_infeasibility,
            'dual_infeasibility': self.dual_infeasibility,
            'iterations': self.iterations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary (e.g., from database row)"""
        result = cls()
        for key, value in data.items():
            if hasattr(result, key):
                if key == 'timestamp' and value:
                    result.timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
                elif key == 'environment_info' and isinstance(value, str):
                    result.environment_info = json.loads(value) if value else {}
                else:
                    setattr(result, key, value)
        return result


class ResultProcessor:
    """Process latest results from database for simplified reporting"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize result processor with database path"""
        if db_path is None:
            db_path = project_root / "database" / "results.db"
        
        self.db_path = str(db_path)
        self.logger = get_logger("result_processor")
    
    def get_latest_results_for_reporting(self) -> List[BenchmarkResult]:
        """
        Get latest results using commit_hash and environment_info with timestamp tiebreaker.
        
        Returns:
            List of BenchmarkResult objects representing the latest benchmark run
        """
        
        self.logger.info("Extracting latest results for reporting...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query for latest results using commit_hash and environment_info
                # Use timestamp as tiebreaker for truly latest results
                query = """
                    SELECT * FROM results 
                    WHERE (commit_hash, environment_info, timestamp) IN (
                        SELECT commit_hash, environment_info, MAX(timestamp)
                        FROM results 
                        GROUP BY solver_name, problem_name
                        ORDER BY timestamp DESC
                        LIMIT 1000  -- Reasonable limit for latest batch
                    )
                    ORDER BY problem_library, problem_name, solver_name
                """
                
                cursor = conn.cursor()
                cursor.execute(query)
                
                # Get column names
                columns = [description[0] for description in cursor.description]
                
                # Convert rows to BenchmarkResult objects
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row))
                    result = BenchmarkResult.from_dict(row_dict)
                    results.append(result)
                
                self.logger.info(f"Retrieved {len(results)} latest results for reporting")
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get latest results: {e}")
            return []
    
    def get_summary_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        
        if not results:
            return {
                'total_results': 0,
                'total_solvers': 0,
                'total_problems': 0,
                'success_rate': 0.0,
                'avg_solve_time': 0.0
            }
        
        # Calculate basic statistics
        total_results = len(results)
        solvers = set(r.solver_name for r in results)
        problems = set(r.problem_name for r in results)
        
        # Success rate calculation
        successful_results = [r for r in results if r.status and r.status.upper() == 'OPTIMAL']
        success_rate = len(successful_results) / total_results if total_results > 0 else 0.0
        
        # Average solve time calculation
        valid_times = [r.solve_time for r in results if r.solve_time is not None and r.solve_time > 0]
        avg_solve_time = sum(valid_times) / len(valid_times) if valid_times else 0.0
        
        # Problem type distribution
        problem_types = {}
        for result in results:
            ptype = result.problem_type
            if ptype not in problem_types:
                problem_types[ptype] = 0
            problem_types[ptype] += 1
        
        # Library distribution
        libraries = {}
        for result in results:
            lib = result.problem_library
            if lib not in libraries:
                libraries[lib] = 0
            libraries[lib] += 1
        
        return {
            'total_results': total_results,
            'total_solvers': len(solvers),
            'total_problems': len(problems),
            'success_rate': success_rate,
            'avg_solve_time': avg_solve_time,
            'problem_type_distribution': problem_types,
            'library_distribution': libraries,
            'solver_names': sorted(list(solvers)),
            'problem_names': sorted(list(problems))
        }
    
    def get_solver_comparison(self, results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """Generate solver comparison data"""
        
        solver_stats = {}
        
        for result in results:
            solver = result.solver_name
            if solver not in solver_stats:
                solver_stats[solver] = {
                    'solver_name': solver,
                    'problems_attempted': 0,
                    'problems_solved': 0,
                    'solve_times': []
                }
            
            solver_stats[solver]['problems_attempted'] += 1
            
            if result.status and result.status.upper() == 'OPTIMAL':
                solver_stats[solver]['problems_solved'] += 1
            
            if result.solve_time is not None and result.solve_time > 0:
                solver_stats[solver]['solve_times'].append(result.solve_time)
        
        # Calculate derived statistics
        comparison_data = []
        for solver_name, stats in solver_stats.items():
            times = stats['solve_times']
            
            comparison_data.append({
                'solver_name': solver_name,
                'problems_attempted': stats['problems_attempted'],
                'problems_solved': stats['problems_solved'],
                'success_rate': stats['problems_solved'] / stats['problems_attempted'] if stats['problems_attempted'] > 0 else 0.0,
                'avg_solve_time': sum(times) / len(times) if times else 0.0,
                'min_solve_time': min(times) if times else 0.0,
                'max_solve_time': max(times) if times else 0.0
            })
        
        # Sort by success rate, then by average solve time
        comparison_data.sort(key=lambda x: (-x['success_rate'], x['avg_solve_time']))
        
        return comparison_data
    
    def get_results_matrix(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate problems × solvers matrix data"""
        
        # Get unique problems and solvers
        problems = sorted(set(r.problem_name for r in results))
        solvers = sorted(set(r.solver_name for r in results))
        
        # Create matrix
        matrix = {}
        for problem in problems:
            matrix[problem] = {}
            for solver in solvers:
                matrix[problem][solver] = None
        
        # Fill matrix with results
        for result in results:
            problem = result.problem_name
            solver = result.solver_name
            
            matrix[problem][solver] = {
                'status': result.status,
                'solve_time': result.solve_time,
                'objective_value': result.primal_objective_value,
                'iterations': result.iterations
            }
        
        return {
            'problems': problems,
            'solvers': solvers,
            'matrix': matrix
        }


def main():
    """Test the result processor"""
    processor = ResultProcessor()
    
    print("Testing Result Processor...")
    results = processor.get_latest_results_for_reporting()
    print(f"Retrieved {len(results)} results")
    
    if results:
        print("\nSummary Statistics:")
        summary = processor.get_summary_statistics(results)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\nSolver Comparison:")
        comparison = processor.get_solver_comparison(results)
        for solver_data in comparison:
            print(f"  {solver_data['solver_name']}: {solver_data['success_rate']:.2%} success, {solver_data['avg_solve_time']:.4f}s avg")


if __name__ == "__main__":
    main()