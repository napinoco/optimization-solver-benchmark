"""
DAT File Loader for SDPA .dat-s format.

This module provides functionality to load and parse optimization problems
stored in SDPA sparse format (.dat-s files), commonly used by SDPLIB
and other semidefinite programming problem libraries.

SDPA Format Structure:
The file consists of six sections:
1. Comments: Lines beginning with " or *
2. m: Number of constraint matrices
3. nblocks: Number of blocks in block diagonal structure
4. Block sizes: Vector giving sizes of individual blocks (negative = diagonal)
5. Objective vector c
6. Matrix entries: One per line as <matno> <blkno> <i> <j> <entry>

The problem format is: 
(P) min c1*x1+c2*x2+...+cm*xm
    s.t. F1*x1+F2*x2+...+Fm*xm - F0 = X
         X >= 0

Where matno=0 is F0, matno=1,2,... are F1,F2,...
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("dat_loader")


class DATLoader:
    """Loader for optimization problems in SDPA .dat-s format."""
    
    def __init__(self):
        """Initialize the DAT loader."""
        pass
    
    def load(self, file_path: str) -> ProblemData:
        """
        Load problem from .dat-s file.
        
        Args:
            file_path: Path to the .dat-s file
            
        Returns:
            ProblemData object
        """
        # Parse and convert the file
        parsed_data = self.parse_sdpa_file(file_path)
        problem_data = self.convert_to_problem_data(parsed_data)
        
        logger.info(f"Successfully loaded DAT problem: {problem_data}")
        return problem_data
    
    def parse_sdpa_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse SDPA sparse format file.
        
        Args:
            file_path: Path to the .dat-s file
            
        Returns:
            Dictionary containing parsed SDPA problem data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading SDPA .dat-s file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Remove comments and empty lines
            data_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('"') and not line.startswith('*'):
                    data_lines.append(line)
            
            if len(data_lines) < 4:
                raise ValueError("Invalid SDPA format: insufficient data lines")
            
            # Parse header
            m = int(data_lines[0].split()[0])  # Number of constraints
            nblocks = int(data_lines[1].split()[0])  # Number of blocks
            
            # Parse block sizes
            block_sizes_line = data_lines[2]
            # Remove punctuation
            for char in ',(){}':
                block_sizes_line = block_sizes_line.replace(char, ' ')
            block_sizes = [int(x) for x in block_sizes_line.split() if x.strip()]
            
            if len(block_sizes) != nblocks:
                raise ValueError(f"Block sizes count ({len(block_sizes)}) doesn't match nblocks ({nblocks})")
            
            # Parse objective vector
            c_line = data_lines[3]
            c = np.array([float(x) for x in c_line.split()])
            
            if len(c) != m:
                raise ValueError(f"Objective vector length ({len(c)}) doesn't match m ({m})")
            
            # Parse matrix entries
            matrices = {}
            for line in data_lines[4:]:
                parts = line.split()
                if len(parts) >= 5:
                    matno = int(parts[0])
                    blkno = int(parts[1])
                    i = int(parts[2])
                    j = int(parts[3])
                    value = float(parts[4])
                    
                    if matno not in matrices:
                        matrices[matno] = {}
                    if blkno not in matrices[matno]:
                        matrices[matno][blkno] = {}
                    if i not in matrices[matno][blkno]:
                        matrices[matno][blkno][i] = {}
                    
                    matrices[matno][blkno][i][j] = value
            
            parsed_data = {
                'm': m,
                'nblocks': nblocks,
                'block_sizes': block_sizes,
                'c': c,
                'matrices': matrices
            }
            
            logger.debug(f"Parsed SDPA file: m={m}, nblocks={nblocks}, blocks={block_sizes}")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise
    
    def build_constraint_matrices(self, parsed_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build constraint matrices A and b from parsed SDPA data.
        
        Args:
            parsed_data: Parsed SDPA problem data
            
        Returns:
            Tuple of (A, b) matrices for the constraint Ax = b
        """
        m = parsed_data['m']
        block_sizes = parsed_data['block_sizes']
        matrices = parsed_data['matrices']
        
        # Calculate total variable dimension
        n_vars = 0
        block_starts = [0]
        for block_size in block_sizes:
            if block_size > 0:
                # Symmetric matrix: n*(n+1)/2 variables (upper triangle)
                n_vars += block_size * (block_size + 1) // 2
            else:
                # Diagonal matrix: |block_size| variables
                n_vars += abs(block_size)
            block_starts.append(n_vars)
        
        # Initialize constraint matrices
        A = np.zeros((m, n_vars))
        F0 = np.zeros(n_vars)  # F0 matrix vectorized
        
        # Build F0 vector from F0 (matno=0)
        if 0 in matrices:
            self._fill_constraint_vector(matrices[0], block_sizes, block_starts, F0)
        
        # Build A matrix from F1, F2, ..., Fm (matno=1,2,...,m)
        for matno in range(1, m + 1):
            if matno in matrices:
                constraint_vector = np.zeros(n_vars)
                self._fill_constraint_vector(matrices[matno], block_sizes, block_starts, constraint_vector)
                A[matno - 1, :] = constraint_vector
        
        # In SDPA format: F1*x1 + F2*x2 + ... + Fm*xm - F0 = X
        # Convert to standard form: Ax = b where b = vec(F0)
        b = F0
        return A, b
    
    def _fill_constraint_vector(self, matrix_data: Dict[int, Any], 
                              block_sizes: List[int], 
                              block_starts: List[int], 
                              vector: np.ndarray):
        """
        Fill constraint vector from matrix data.
        
        Args:
            matrix_data: Matrix data for one constraint
            block_sizes: Block sizes
            block_starts: Starting indices for each block
            vector: Vector to fill (modified in place)
        """
        for blkno, block_data in matrix_data.items():
            if blkno < 1 or blkno > len(block_sizes):
                continue
                
            block_size = block_sizes[blkno - 1]  # SDPA uses 1-based indexing
            block_start = block_starts[blkno - 1]
            
            for i, row_data in block_data.items():
                for j, value in row_data.items():
                    if block_size > 0:
                        # Symmetric matrix case
                        if i <= j:  # Upper triangle
                            idx = self._symmetric_index(i - 1, j - 1, block_size)
                            vector[block_start + idx] = value
                    else:
                        # Diagonal matrix case
                        if i == j and i <= abs(block_size):
                            vector[block_start + i - 1] = value
    
    def _symmetric_index(self, i: int, j: int, n: int) -> int:
        """
        Convert (i,j) position in symmetric matrix to vector index.
        
        Args:
            i, j: Matrix indices (0-based)
            n: Matrix size
            
        Returns:
            Vector index for upper triangular storage
        """
        if i > j:
            i, j = j, i  # Ensure i <= j for upper triangle
        return i * n - i * (i + 1) // 2 + j
    
    def analyze_block_structure(self, block_sizes: List[int]) -> Dict[str, Any]:
        """
        Analyze block structure to determine cone types.
        
        Args:
            block_sizes: List of block sizes
            
        Returns:
            Dictionary with cone structure information
        """
        cone_info = {
            'linear_vars': 0,
            'sdp_cones': []
        }
        
        for block_size in block_sizes:
            if block_size > 0:
                # Positive size = SDP block
                cone_info['sdp_cones'].append(block_size)
            else:
                # Negative size = diagonal (linear) block
                cone_info['linear_vars'] += abs(block_size)
        
        return cone_info
    
    def convert_to_problem_data(self, parsed_data: Dict[str, Any]) -> ProblemData:
        """
        Convert parsed SDPA data to unified ProblemData format.
        
        Args:
            parsed_data: Parsed SDPA problem data
            
        Returns:
            ProblemData object
        """
        # Build constraint matrices
        A, b = self.build_constraint_matrices(parsed_data)
        c = parsed_data['c']
        
        # Analyze cone structure
        cone_info = self.analyze_block_structure(parsed_data['block_sizes'])
        
        # Determine problem class
        if len(cone_info['sdp_cones']) > 0:
            problem_class = 'SDP'
        else:
            problem_class = 'LP'
        
        # Create metadata
        metadata = {
            'source': 'DAT file',
            'format': 'SDPA .dat-s',
            'cone_structure': cone_info,
            'block_structure': {
                'num_blocks': parsed_data['nblocks'],
                'block_sizes': parsed_data['block_sizes']
            },
            'original_dimensions': {
                'variables': A.shape[1],
                'constraints': A.shape[0]
            }
        }
        
        logger.info(f"Converted {problem_class} problem "
                   f"({A.shape[1]} vars, {A.shape[0]} constraints)")
        
        # Create ProblemData object
        # SDPA format uses equality constraints: Ax = b
        return ProblemData(
            problem_class=problem_class,
            c=c,
            A_eq=A,
            b_eq=b,
            A_ub=None,  # SDPA uses equality constraints
            b_ub=None,
            bounds=None,  # Bounds handled by cone structure
            metadata=metadata
        )


# Convenience function for backward compatibility
def load_dat_problem(file_path: str) -> ProblemData:
    """
    Convenience function to load a DAT problem.
    
    Args:
        file_path: Path to the .dat-s file
        
    Returns:
        ProblemData object
    """
    loader = DATLoader()
    return loader.load(file_path)


if __name__ == "__main__":
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dat_loader.py <path_to_dat_s_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        problem = load_dat_problem(file_path)
        print(f"Loaded problem: {problem}")
        
        # Display structure summary if available
        structure = problem.get_structure_summary()
        if structure:
            print(f"Structure analysis: {structure}")
            
    except Exception as e:
        print(f"Error loading problem: {e}")
        sys.exit(1)