"""
CVXPY Problem Converter for External Libraries.

This module provides functionality to convert external library problems
(DIMACS, SDPLIB) to CVXPY problem format so they can be solved by CVXPY backends.
"""

import numpy as np
import cvxpy as cp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("cvxpy_converter")


class CvxpyConverter:
    """Converts external library problems to CVXPY format."""
    
    def __init__(self):
        """Initialize the CVXPY converter."""
        pass
    
    def convert_sedumi_to_cvxpy(self, problem_data: ProblemData) -> ProblemData:
        """
        Convert SeDuMi format problem (DIMACS) to CVXPY problem.
        
        Args:
            problem_data: Problem data from DIMACS loader
            
        Returns:
            ProblemData with CVXPY problem object
        """
        logger.info(f"Converting SeDuMi problem {problem_data.name} to CVXPY format")
        
        # Extract problem data
        A = problem_data.A_eq
        b = problem_data.b_eq
        c = problem_data.c
        
        if A is None or b is None or c is None:
            raise ValueError("Invalid SeDuMi problem: missing A, b, or c")
        
        # Extract cone structure from metadata
        cone_info = problem_data.metadata.get('cone_structure', {})
        
        # Create CVXPY variables
        n_vars = len(c)
        variables = {}
        
        # Build variable vector for different cone types
        var_idx = 0
        constraints = []
        
        # Free variables
        free_vars = cone_info.get('free_vars', 0)
        if free_vars > 0:
            x_free = cp.Variable(free_vars, name='x_free')
            variables['free'] = x_free
            var_idx += free_vars
        
        # Non-negative variables
        nonneg_vars = cone_info.get('nonneg_vars', 0)
        if nonneg_vars > 0:
            x_nonneg = cp.Variable(nonneg_vars, nonneg=True, name='x_nonneg')
            variables['nonneg'] = x_nonneg
            var_idx += nonneg_vars
        
        # Second-order cone variables
        soc_cones = cone_info.get('soc_cones', [])
        soc_vars = []
        for i, cone_dim in enumerate(soc_cones):
            x_soc = cp.Variable(cone_dim, name=f'x_soc_{i}')
            soc_vars.append(x_soc)
            variables[f'soc_{i}'] = x_soc
            # Add SOC constraint: ||x[1:]||_2 <= x[0]
            if cone_dim > 1:
                constraints.append(cp.SOC(x_soc[0], x_soc[1:]))
            var_idx += cone_dim
        
        # Semidefinite cone variables 
        sdp_cones = cone_info.get('sdp_cones', [])
        sdp_vars = []
        for i, cone_size in enumerate(sdp_cones):
            # Create symmetric matrix variable
            X_sdp = cp.Variable((cone_size, cone_size), symmetric=True, name=f'X_sdp_{i}')
            sdp_vars.append(X_sdp)
            variables[f'sdp_{i}'] = X_sdp
            # Add PSD constraint
            constraints.append(X_sdp >> 0)
            var_idx += cone_size * cone_size
        
        # Build complete variable vector
        x_parts = []
        if free_vars > 0:
            x_parts.append(cp.vec(variables['free']))
        if nonneg_vars > 0:
            x_parts.append(cp.vec(variables['nonneg']))
        for soc_var in soc_vars:
            x_parts.append(cp.vec(soc_var))
        for sdp_var in sdp_vars:
            x_parts.append(cp.vec(sdp_var))
        
        if x_parts:
            x = cp.hstack(x_parts)
        else:
            x = cp.Variable(n_vars, name='x')
            variables['main'] = x
        
        # Create objective: minimize c^T x
        objective = cp.Minimize(c.T @ x)
        
        # Add equality constraints: Ax = b
        constraints.append(A @ x == b)
        
        # Create CVXPY problem
        cvxpy_problem = cp.Problem(objective, constraints)
        
        logger.info(f"Converted {problem_data.name}: {len(constraints)} constraints, "
                   f"{len(variables)} variable groups")
        
        # Return new ProblemData with CVXPY problem
        return ProblemData(
            name=problem_data.name,
            problem_class=problem_data.problem_class,
            c=c,
            A_eq=A,
            b_eq=b,
            cvxpy_problem=cvxpy_problem,
            variables=variables,
            objective=objective,
            constraints=constraints,
            metadata=problem_data.metadata,
            analyze_structure=False  # Already have structure from original
        )
    
    def convert_sdpa_to_cvxpy(self, problem_data: ProblemData) -> ProblemData:
        """
        Convert SDPA format problem (SDPLIB) to CVXPY problem.
        
        Args:
            problem_data: Problem data from SDPLIB loader
            
        Returns:
            ProblemData with CVXPY problem object
        """
        logger.info(f"Converting SDPA problem {problem_data.name} to CVXPY format")
        
        # Extract problem data
        A = problem_data.A_eq
        b = problem_data.b_eq
        c = problem_data.c
        
        if A is None or b is None or c is None:
            raise ValueError("Invalid SDPA problem: missing A, b, or c")
        
        # Fix dimension issue in SDPLIB: b and c are swapped in the loader
        # A should be (m x n), b should be (m,), c should be (n,)
        # If dimensions don't match, swap b and c
        m, n = A.shape
        if len(b) == n and len(c) == m:
            logger.debug(f"Swapping b and c vectors: b({len(b)}) <-> c({len(c)})")
            b, c = c, b
        
        # Extract block structure from metadata
        block_structure = problem_data.metadata.get('block_structure', {})
        block_sizes = block_structure.get('block_sizes', [])
        
        # Create CVXPY variables for each block
        variables = {}
        x_parts = []
        constraints = []
        var_idx = 0
        
        for i, block_size in enumerate(block_sizes):
            if block_size > 0:
                # Positive block size = SDP block (symmetric matrix)
                X = cp.Variable((block_size, block_size), symmetric=True, name=f'X_{i}')
                variables[f'sdp_{i}'] = X
                constraints.append(X >> 0)  # PSD constraint
                
                # Vectorize using upper triangle to match SDPLIB vectorization
                # SDPLIB uses symmetric matrix representation: n*(n+1)/2 variables
                upper_tri_indices = np.triu_indices(block_size)
                x_block_list = []
                for row, col in zip(upper_tri_indices[0], upper_tri_indices[1]):
                    x_block_list.append(X[row, col])
                
                if x_block_list:
                    x_block = cp.hstack(x_block_list)
                    x_parts.append(x_block)
                var_idx += block_size * (block_size + 1) // 2
                
            else:
                # Negative block size = diagonal block (linear variables)
                diag_size = abs(block_size)
                x_diag = cp.Variable(diag_size, nonneg=True, name=f'x_diag_{i}')
                variables[f'diag_{i}'] = x_diag
                x_parts.append(x_diag)
                var_idx += diag_size
        
        # Build complete variable vector
        if x_parts:
            x = cp.hstack(x_parts)
        else:
            # Fallback for problems without clear block structure
            n_vars = len(c)
            x = cp.Variable(n_vars, name='x')
            variables['main'] = x
        
        # Create objective: minimize c^T x
        objective = cp.Minimize(c.T @ x)
        
        # Add equality constraints: Ax = b
        constraints.append(A @ x == b)
        
        # Create CVXPY problem
        cvxpy_problem = cp.Problem(objective, constraints)
        
        logger.info(f"Converted {problem_data.name}: {len(constraints)} constraints, "
                   f"{len(variables)} variable groups, {len(block_sizes)} blocks")
        
        # Return new ProblemData with CVXPY problem
        return ProblemData(
            name=problem_data.name,
            problem_class=problem_data.problem_class,
            c=c,
            A_eq=A,
            b_eq=b,
            cvxpy_problem=cvxpy_problem,
            variables=variables,
            objective=objective,
            constraints=constraints,
            metadata=problem_data.metadata,
            analyze_structure=False  # Already have structure from original
        )
    
    def convert_problem(self, problem_data: ProblemData) -> ProblemData:
        """
        Convert external library problem to CVXPY format.
        
        Args:
            problem_data: Problem data from external loader
            
        Returns:
            ProblemData with CVXPY problem object
        """
        # Check if already has CVXPY problem
        if problem_data.cvxpy_problem is not None:
            logger.debug(f"Problem {problem_data.name} already has CVXPY format")
            return problem_data
        
        # Determine source format and convert accordingly
        source = problem_data.metadata.get('source', '')
        
        if source == 'DIMACS':
            return self.convert_sedumi_to_cvxpy(problem_data)
        elif source == 'SDPLIB':
            return self.convert_sdpa_to_cvxpy(problem_data)
        else:
            logger.warning(f"Unknown source format: {source}")
            return problem_data


def convert_to_cvxpy(problem_data: ProblemData) -> ProblemData:
    """
    Convenience function to convert external library problem to CVXPY format.
    
    Args:
        problem_data: Problem data from external loader
        
    Returns:
        ProblemData with CVXPY problem object
    """
    converter = CvxpyConverter()
    return converter.convert_problem(problem_data)


if __name__ == "__main__":
    # Test script
    print("Testing CVXPY converter...")
    
    try:
        from scripts.external.dimacs_loader import load_dimacs_problem
        from scripts.external.sdplib_loader import load_sdplib_problem
        
        # Test DIMACS conversion
        print("\nTesting DIMACS conversion:")
        try:
            dimacs_file = Path(__file__).parent.parent.parent / "problems/DIMACS/data/TRUSS/truss5.mat.gz"
            if dimacs_file.exists():
                problem = load_dimacs_problem(str(dimacs_file), "truss5")
                print(f"Original problem: {problem}")
                
                converted = convert_to_cvxpy(problem)
                print(f"Converted problem: {converted}")
                print(f"Has CVXPY problem: {converted.cvxpy_problem is not None}")
                print(f"Variables: {list(converted.variables.keys())}")
                print(f"Constraints: {len(converted.constraints)}")
        except Exception as e:
            print(f"DIMACS test failed: {e}")
        
        # Test SDPLIB conversion
        print("\nTesting SDPLIB conversion:")
        try:
            sdplib_file = Path(__file__).parent.parent.parent / "problems/SDPLIB/data/control1.dat-s"
            if sdplib_file.exists():
                problem = load_sdplib_problem(str(sdplib_file), "control1")
                print(f"Original problem: {problem}")
                
                converted = convert_to_cvxpy(problem)
                print(f"Converted problem: {converted}")
                print(f"Has CVXPY problem: {converted.cvxpy_problem is not None}")
                print(f"Variables: {list(converted.variables.keys())}")
                print(f"Constraints: {len(converted.constraints)}")
        except Exception as e:
            print(f"SDPLIB test failed: {e}")
        
        print("\n✓ CVXPY converter tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise