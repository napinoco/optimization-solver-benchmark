"""
Database Models for Re-Architected Benchmark System

Contains the single denormalized table model and data structures
for the simplified database design.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class BenchmarkResult:
    """
    Single denormalized benchmark result model.
    
    Represents one row in the results table with all necessary information
    for a single solver-problem combination execution.
    """
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
    
    def __post_init__(self):
        """Initialize default values"""
        if self.environment_info is None:
            self.environment_info = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_name == 'timestamp' and field_value:
                result[field_name] = field_value.isoformat()
            elif field_name == 'environment_info':
                result[field_name] = field_value or {}
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary (e.g., from database row)"""
        # Handle timestamp conversion
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Handle environment_info JSON
        if 'environment_info' in data and isinstance(data['environment_info'], str):
            try:
                data['environment_info'] = json.loads(data['environment_info'])
            except (json.JSONDecodeError, TypeError):
                data['environment_info'] = {}
        
        # Filter only known fields
        known_fields = set(cls.__annotations__.keys())
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        
        return cls(**filtered_data)


# Database table schema constants
TABLE_NAME = "results"

REQUIRED_FIELDS = [
    "solver_name",
    "solver_version", 
    "problem_library",
    "problem_name",
    "problem_type",
    "environment_info",
    "commit_hash"
]

OPTIONAL_RESULT_FIELDS = [
    "solve_time",
    "status",
    "primal_objective_value",
    "dual_objective_value", 
    "duality_gap",
    "primal_infeasibility",
    "dual_infeasibility",
    "iterations"
]

ALL_FIELDS = ["id", "timestamp"] + REQUIRED_FIELDS + OPTIONAL_RESULT_FIELDS