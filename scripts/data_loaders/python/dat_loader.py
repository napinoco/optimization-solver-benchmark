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
from typing import Dict, List, Optional, Any
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loaders.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("dat_loader")


class DATLoader:
    """Loader for optimization problems in SDPA .dat-s format."""
    
    def __init__(self):
        """Initialize the DAT loader."""
        pass
    
    def load(self, file_path: str, problem_name: str = None) -> ProblemData:
        """
        Load problem from .dat-s file.
        
        Args:
            file_path: Path to the .dat-s file
            problem_name: Optional problem name (if not provided, extracted from file path)
            
        Returns:
            ProblemData object
        """
        # Use provided name or extract from file path as fallback
        if problem_name is None:
            problem_name = Path(file_path).stem.replace('.dat-s', '')
        
        # Parse and convert the file
        parsed_data = self.parse_sdpa_file(file_path)
        problem_data = self.convert_to_problem_data(parsed_data, problem_name)
        
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
            matrices = [[{'i': [], 'j': [], 'val': []} for _ in range(nblocks)] for _ in range(m + 1)]
            for line in data_lines[4:]:
                parts = line.split()
                if len(parts) >= 5:
                    matno = int(parts[0])
                    blkno = int(parts[1])
                    i = int(parts[2])
                    j = int(parts[3])
                    value = float(parts[4])

                    matrices[matno][blkno - 1]['i'].append(i - 1)
                    matrices[matno][blkno - 1]['j'].append(j - 1)
                    matrices[matno][blkno - 1]['val'].append(value)
                    if i != j:
                        # Ensure symmetric entry is also stored
                        matrices[matno][blkno - 1]['i'].append(j - 1)
                        matrices[matno][blkno - 1]['j'].append(i - 1)
                        matrices[matno][blkno - 1]['val'].append(value)

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
    
    def analyze_block_structure(self, block_sizes: List[int]) -> Dict[str, Any]:
        """
        Analyze block structure to determine cone types.
        Uses SeDuMi standard field names for consistency.
        
        Args:
            block_sizes: List of block sizes
            
        Returns:
            Dictionary with cone structure information in SeDuMi format
        """
        cone_info = {
            'free_vars': 0,
            'nonneg_vars': 0,
            'soc_cones': [],
            'sdp_cones': block_sizes
        }

        return cone_info
    
    def convert_to_problem_data(self, parsed_data: Dict[str, Any], 
                              problem_name: str) -> ProblemData:
        """
        Convert parsed SDPA data to unified ProblemData format.
        
        Args:
            parsed_data: Parsed SDPA problem data
            problem_name: Name for the problem
            
        Returns:
            ProblemData object
        """

        b = -parsed_data['c']

        import scipy.sparse as sp
        matrices = parsed_data['matrices']
        block_sizes = parsed_data['block_sizes']
        nblocks = parsed_data['nblocks']
        m = parsed_data['m']

        # Build constraint matrices
        for matno, blk_data in enumerate(matrices):
            for blkno_, entries in enumerate(blk_data):
                data = matrices[matno][blkno_]
                mat = sp.csc_array((data['val'], (data['i'], data['j'])),
                                   shape=(block_sizes[blkno_], block_sizes[blkno_]))
                matrices[matno][blkno_] = mat

        c = -sp.hstack([matrices[0][blkno_].reshape(1, -1) for blkno_ in range(nblocks)], format='csc')
        A = -sp.vstack([sp.hstack(
            [matrices[matno][blkno_].reshape(1, -1) for blkno_ in range(nblocks)]
        ) for matno in range(1, m + 1)], format='csc')

        # Analyze cone structure
        cone_info = self.analyze_block_structure(parsed_data['block_sizes'])
        
        # Determine problem class
        problem_class = 'SDP'

        # Create metadata with corrected cone structure field names
        metadata = {
            'source': 'DAT file',
            'format': 'SDPA .dat-s',
            'cone_structure': cone_info,
            'block_structure': {
                'num_blocks': parsed_data['nblocks'],
                'block_sizes': parsed_data['block_sizes']
            },
            'original_dimensions': {
                'variables': A.shape[1],  # n_vars (correct)
                'constraints': A.shape[0]  # m (correct)
            }
        }
        
        logger.info(f"Converted {problem_name}: {problem_class} problem "
                   f"({A.shape[1]} vars, {A.shape[0]} constraints)")
        
        # Create ProblemData object
        # SDPA format uses equality constraints: Ax = b
        return ProblemData(
            name=problem_name,
            problem_class=problem_class,
            c=c,
            A_eq=A,
            b_eq=b,
            A_ub=None,
            b_ub=None,
            bounds=None,
            cone_structure=cone_info,
            metadata=metadata
        )


# Convenience function for backward compatibility
def load_dat_problem(file_path: str, 
                    problem_name: Optional[str] = None) -> ProblemData:
    """
    Convenience function to load a DAT problem.
    
    Args:
        file_path: Path to the .dat-s file
        problem_name: Optional name for the problem
        
    Returns:
        ProblemData object
    """
    loader = DATLoader()
    return loader.load(file_path, problem_name)


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