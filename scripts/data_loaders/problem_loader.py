import os
import yaml
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("problem_loader")

# Import for lazy loading to avoid circular imports

class ProblemData:
    """
    Container for optimization problem data.
       SeDuMi-like format
           Primal Problem:
               minimize.   c.T x + (1/2) w^T P w
               subject to  A_eq x + P w = b_eq
                          x \\in K              (Primal cone constraints)

           Dual Problem:
               maximize    b^T y - (1/2) y^T P y
               subject to  z = c - A_eq.T y
                           z \\in K*             (Dual cone constraints)
    """
    def __init__(self, name: str, problem_class: str, c: np.ndarray = None, 
                 A_eq: np.ndarray = None, b_eq: np.ndarray = None,
                 A_ub: np.ndarray = None, b_ub: np.ndarray = None,
                 bounds: List[Tuple] = None, P: np.ndarray = None,
                 cone_structure: Dict[str, Any] = None,  # First-class cone structure field
                 metadata=None):
        self.name = name
        self.problem_class = problem_class
        self.c = c  # objective coefficients
        self.A_eq = A_eq  # equality constraint matrix
        self.b_eq = b_eq  # equality constraint RHS
        self.A_ub = A_ub  # inequality constraint matrix  
        self.b_ub = b_ub  # inequality constraint RHS
        self.bounds = bounds  # variable bounds
        self.P = P  # quadratic objective matrix for QP
        
        # NEW: First-class cone structure support with backward compatibility
        if cone_structure is not None:
            self.cone_structure = cone_structure
        elif metadata and 'cone_structure' in metadata:
            # Backward compatibility: extract from metadata if not provided directly
            self.cone_structure = metadata['cone_structure']
        else:
            # Default empty cone structure
            self.cone_structure = {
                'free_vars': 0,
                'nonneg_vars': 0,
                'soc_cones': [],
                'sdp_cones': []
            }
        
        # Additional problem metadata
        self.metadata = metadata or {}  # Additional problem metadata
        
        # Ensure cone_structure is also in metadata for backward compatibility
        if 'cone_structure' not in self.metadata and self.cone_structure:
            self.metadata['cone_structure'] = self.cone_structure
        
        # Basic problem dimensions for display
        self._num_variables = self._compute_num_variables()
        self._num_constraints = self._compute_num_constraints()
        
    def _compute_num_variables(self):
        """Compute number of variables from problem data."""
        if self.c is not None:
            return self.c.shape[0] if hasattr(self.c, 'shape') else len(self.c)
        elif self.A_eq is not None:
            return self.A_eq.shape[1]
        elif self.A_ub is not None:
            return self.A_ub.shape[1]
        return 0
    
    def _compute_num_constraints(self):
        """Compute number of constraints from problem data."""
        total = 0
        if self.A_eq is not None:
            total += self.A_eq.shape[0]
        if self.A_ub is not None:
            total += self.A_ub.shape[0]
        return total
        
    def __repr__(self):
        structure_info = f", {self._num_variables} vars, {self._num_constraints} constraints"
        return f"ProblemData(name='{self.name}', class='{self.problem_class}'{structure_info})"

# MPS and QPS file format parsers removed - no problems in registry use these formats

def load_problem_registry() -> Dict:
    """Load the problem registry YAML file."""
    project_root = Path(__file__).parent.parent.parent
    registry_path = project_root / "config" / "problem_registry.yaml"
    
    logger.info(f"Loading problem registry: {registry_path}")
    
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


# def load_problem(problem_name: str, problem_set: str = None) -> ProblemData:
#     """Load a specific problem by name from the registry."""
#
#     # Load from registry (new flat structure)
#     registry = load_problem_registry()
#
#     # Find the problem in the flat registry
#     problem_info = None
#     if problem_name in registry["problem_libraries"]:
#         problem_info = registry["problem_libraries"][problem_name]
#
#     if not problem_info:
#         raise ValueError(f"Problem '{problem_name}' not found in registry")
#
#     # Filter by problem_set/library if specified
#     if problem_set and problem_info.get("library_name") != problem_set:
#         raise ValueError(f"Problem '{problem_name}' is not from library '{problem_set}'")
#
#     # Get absolute path to problem file
#     project_root = Path(__file__).parent.parent.parent
#     file_path = project_root / problem_info["file_path"]
#
#     # Load based on file type
#     file_type = problem_info["file_type"]
#
#     if file_type == "mat":
#         # Use MAT loader for .mat.gz files (DIMACS)
#         from scripts.data_loaders.python.mat_loader import MATLoader
#         loader = MATLoader()
#         return loader.load(str(file_path), problem_name)
#     elif file_type == "dat-s":
#         # Use DAT loader for .dat-s files (SDPLIB)
#         from scripts.data_loaders.python.dat_loader import DATLoader
#         loader = DATLoader()
#         return loader.load(str(file_path), problem_name)
#     # MPS, QPS, and Python loaders removed - no problems in registry use these formats
#     else:
#         raise ValueError(f"Unsupported file type: {file_type}")
#
#
# if __name__ == "__main__":
#     # Test script to load and validate problems
#     try:
#         # Test loading registry
#         registry = load_problem_registry()
#         print("Registry loaded successfully:")
#         print(f"Problem sets: {list(registry['problems'].keys())}")
#
#         # Test loading each problem
#         for problem_set in registry["problems"]:
#             for problem_class in registry["problems"][problem_set]:
#                 for problem_info in registry["problems"][problem_set][problem_class]:
#                     problem_name = problem_info["name"]
#                     print(f"\nTesting problem: {problem_name}")
#
#                     problem = load_problem(problem_name, problem_set)
#                     print(f"  Loaded: {problem}")
#                     print(f"  Variables: {len(problem.c) if problem.c is not None else 0}")
#                     if problem.A_ub is not None:
#                         print(f"  Constraints: {problem.A_ub.shape[0]}")
#                     if problem.P is not None:
#                         print(f"  Quadratic matrix shape: {problem.P.shape}")
#
#     except Exception as e:
#         logger.error(f"Test failed: {e}")
#         raise