"""
DIMACS Problem Loader for SeDuMi .mat.gz format.

This module provides functionality to load and parse DIMACS optimization problems
stored in SeDuMi MATLAB format (compressed .mat.gz files).

SeDuMi Format Structure:
- A: constraint matrix (m x n)
- b: RHS vector (m x 1) 
- c: objective vector (n x 1)
- K: cone structure describing variable partitions
  - K.f: number of free variables
  - K.l: number of nonnegative variables  
  - K.q: array of second-order cone dimensions
  - K.s: array of semidefinite cone block sizes

The problem format is: min c'x subject to Ax = b, x in K
"""

import gzip
import numpy as np
import scipy.io
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("dimacs_loader")


class DimacsLoader:
    """Loader for DIMACS problems in SeDuMi .mat.gz format."""
    
    def __init__(self):
        """Initialize the DIMACS loader."""
        pass
    
    def load_sedumi_mat_gz(self, file_path: str) -> Dict[str, Any]:
        """
        Load SeDuMi problem from compressed .mat.gz file.
        
        Args:
            file_path: Path to the .mat.gz file
            
        Returns:
            Dictionary containing SeDuMi problem data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading SeDuMi .mat.gz file: {file_path}")
        
        try:
            # Read compressed MATLAB file
            with gzip.open(file_path, 'rb') as f:
                mat_data = scipy.io.loadmat(f)
            
            # SeDuMi format validation
            required_fields = ['A', 'b', 'c']
            for field in required_fields:
                if field not in mat_data:
                    raise ValueError(f"Missing required SeDuMi field: {field}")
            
            logger.debug(f"Loaded SeDuMi data with fields: {list(mat_data.keys())}")
            return mat_data
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def parse_cone_structure(self, K: np.ndarray) -> Dict[str, Any]:
        """
        Parse SeDuMi cone structure K.
        
        Args:
            K: SeDuMi cone structure (structured numpy array)
            
        Returns:
            Parsed cone information
        """
        cone_info = {
            'free_vars': 0,
            'nonneg_vars': 0, 
            'soc_cones': [],
            'sdp_cones': []
        }
        
        # K is a structured array, extract fields from first element
        try:
            # Access the fields using array indexing
            if K.dtype.names and 's' in K.dtype.names:
                s_data = K['s'][0][0]
                if s_data.size > 0:
                    cone_info['sdp_cones'] = [int(x) for x in s_data.flatten()]
            
            if K.dtype.names and 'q' in K.dtype.names:
                q_data = K['q'][0][0]
                if q_data.size > 0:
                    cone_info['soc_cones'] = [int(x) for x in q_data.flatten()]
            
            if K.dtype.names and 'l' in K.dtype.names:
                l_data = K['l'][0][0]
                if l_data.size > 0:
                    cone_info['nonneg_vars'] = int(l_data.flatten()[0])
            
            if K.dtype.names and 'f' in K.dtype.names:
                f_data = K['f'][0][0]
                if f_data.size > 0:
                    cone_info['free_vars'] = int(f_data.flatten()[0])
                    
        except (IndexError, KeyError, AttributeError) as e:
            logger.warning(f"Could not parse cone structure: {e}")
            # Fallback: treat as linear problem if parsing fails
        
        logger.debug(f"Parsed cone structure: {cone_info}")
        return cone_info
    
    def determine_problem_class(self, cone_info: Dict[str, Any]) -> str:
        """
        Determine problem class based on cone structure.
        SDP takes priority over mixed problems.
        
        Args:
            cone_info: Parsed cone information
            
        Returns:
            Problem class string ('LP', 'SOCP', 'SDP')
        """
        has_soc = len(cone_info['soc_cones']) > 0
        has_sdp = len(cone_info['sdp_cones']) > 0
        
        # SDP takes priority - if it has SDP cones, classify as SDP
        if has_sdp:
            return 'SDP'
        elif has_soc:
            return 'SOCP'
        else:
            return 'LP'
    
    def convert_to_problem_data(self, mat_data: Dict[str, Any], 
                              problem_name: str) -> ProblemData:
        """
        Convert SeDuMi data to unified ProblemData format.
        
        Args:
            mat_data: Loaded SeDuMi MATLAB data
            problem_name: Name for the problem
            
        Returns:
            ProblemData object
        """
        # Extract basic data
        A = mat_data['A']
        
        # Handle sparse matrices for b and c
        b = mat_data['b']
        if hasattr(b, 'toarray'):
            b = b.toarray().flatten()
        else:
            b = b.flatten()
            
        c = mat_data['c']
        if hasattr(c, 'toarray'):
            c = c.toarray().flatten()
        else:
            c = c.flatten()
        
        # Handle cone structure
        cone_info = {}
        if 'K' in mat_data:
            cone_info = self.parse_cone_structure(mat_data['K'])
        
        # Determine problem class
        problem_class = self.determine_problem_class(cone_info)
        
        # Convert sparse matrix to dense if needed
        if hasattr(A, 'toarray'):
            A = A.toarray()
        
        # Create metadata
        metadata = {
            'source': 'DIMACS',
            'format': 'SeDuMi .mat.gz',
            'cone_structure': cone_info,
            'original_dimensions': {
                'variables': A.shape[1],
                'constraints': A.shape[0]
            }
        }
        
        logger.info(f"Converted {problem_name}: {problem_class} problem "
                   f"({A.shape[1]} vars, {A.shape[0]} constraints)")
        
        # Create ProblemData object
        # SeDuMi format uses equality constraints: Ax = b
        return ProblemData(
            name=problem_name,
            problem_class=problem_class,
            c=c,
            A_eq=A,
            b_eq=b,
            A_ub=None,  # SeDuMi uses equality constraints
            b_ub=None,
            bounds=None,  # Bounds handled by cone structure
            metadata=metadata
        )
    
    def load_dimacs_problem(self, file_path: str, 
                           problem_name: Optional[str] = None) -> ProblemData:
        """
        Load a DIMACS problem from .mat.gz file.
        
        Args:
            file_path: Path to the .mat.gz file
            problem_name: Optional name for the problem (defaults to filename)
            
        Returns:
            ProblemData object
        """
        if problem_name is None:
            problem_name = Path(file_path).stem.replace('.mat', '')
        
        # Load and parse the file
        mat_data = self.load_sedumi_mat_gz(file_path)
        problem_data = self.convert_to_problem_data(mat_data, problem_name)
        
        logger.info(f"Successfully loaded DIMACS problem: {problem_data}")
        return problem_data


def load_dimacs_problem(file_path: str, 
                       problem_name: Optional[str] = None) -> ProblemData:
    """
    Convenience function to load a DIMACS problem.
    
    Args:
        file_path: Path to the .mat.gz file
        problem_name: Optional name for the problem
        
    Returns:
        ProblemData object
    """
    loader = DimacsLoader()
    return loader.load_dimacs_problem(file_path, problem_name)


if __name__ == "__main__":
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dimacs_loader.py <path_to_mat_gz_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        problem = load_dimacs_problem(file_path)
        print(f"Loaded problem: {problem}")
        
        # Display structure summary if available
        structure = problem.get_structure_summary()
        if structure:
            print(f"Structure analysis: {structure}")
            
    except Exception as e:
        print(f"Error loading problem: {e}")
        sys.exit(1)