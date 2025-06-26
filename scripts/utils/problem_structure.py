"""
Problem Structure Analysis Module

This module provides utilities to analyze optimization problem structure,
including constraint counting, cone dimension analysis, and automatic 
problem classification into LP, QP, SOCP, and SDP categories.
"""

import sys
import numpy as np
import cvxpy as cp
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger
from scripts.data_loaders.problem_loader import ProblemData

logger = get_logger("problem_structure")


@dataclass
class ConeInfo:
    """Information about a specific cone in the problem."""
    cone_type: str  # 'semi_definite', 'second_order', 'non_negative', 'unrestricted'
    dimension: int
    size: Optional[Tuple[int, ...]] = None  # For matrix cones, e.g., (3, 3) for 3x3 PSD
    count: int = 1  # Number of cones of this type with same dimension


@dataclass
class ProblemStructure:
    """Comprehensive problem structure information."""
    # Basic dimensions
    num_variables: int
    num_constraints: int
    
    # Problem classification
    problem_class: str  # 'LP', 'QP', 'SOCP', 'SDP', 'MIXED'
    
    # Cone analysis
    cone_info: List[ConeInfo]
    
    # Detailed cone summaries
    semi_definite_cones: List[Tuple[int, int]]  # [(size, count), ...]
    second_order_cones: List[Tuple[int, int]]   # [(dim, count), ...]
    non_negative_dim: int
    unrestricted_dim: int
    
    # Matrix dimensions (for QP/SDP)
    has_quadratic_objective: bool = False
    quadratic_matrix_size: Optional[Tuple[int, int]] = None
    
    # Sparsity information
    constraint_matrix_nnz: Optional[int] = None  # Non-zero elements in constraint matrix
    sparsity_ratio: Optional[float] = None  # nnz / (m * n)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format as requested by user."""
        result = {
            'variables': self.num_variables,
            'constraints': self.num_constraints,
            'problem_class': self.problem_class,
            'non_negative_cone_dim': self.non_negative_dim,
            'unrestricted_cone_dim': self.unrestricted_dim
        }
        
        # Format semi-definite cones as requested: [10x33, 5x1, ...]
        # Format: [matrix_dimension x number_of_blocks] 
        if self.semi_definite_cones:
            sdp_dims = []
            for size, count in self.semi_definite_cones:
                sdp_dims.append(f"{size}x{count}")
            result['semi_definite_cone_dim'] = sdp_dims
        
        # Format second-order cones as requested: [5x2, 10x3, ...]
        if self.second_order_cones:
            soc_dims = []
            for dim, count in self.second_order_cones:
                if count == 1:
                    soc_dims.append(f"{dim}")
                else:
                    soc_dims.append(f"{dim}x{count}")
            result['second_order_cone_dim'] = soc_dims
        
        # Add sparsity information if available
        if self.sparsity_ratio is not None:
            result['sparsity_ratio'] = self.sparsity_ratio
            result['constraint_matrix_nnz'] = self.constraint_matrix_nnz
        
        return result


class ProblemStructureAnalyzer:
    """Analyzes optimization problem structure and classifies problem types."""
    
    def __init__(self):
        self.logger = get_logger("problem_structure_analyzer")
    
    def analyze_problem_data(self, problem_data: ProblemData) -> ProblemStructure:
        """
        Analyze a ProblemData object to determine its structure.
        
        Args:
            problem_data: Problem data to analyze
            
        Returns:
            ProblemStructure with comprehensive analysis
        """
        self.logger.debug(f"Analyzing structure of problem: {problem_data.name}")
        
        try:
            cone_structure = problem_data.metadata['cone_structure']

            # Extract basic dimensions
            num_vars = problem_data.metadata.get('original_dimensions', {}).get('variables', 0)
            num_constraints = problem_data.metadata.get('original_dimensions', {}).get('constraints', 0)

            # Calculate sparsity if available
            constraint_matrix_nnz = 0
            if problem_data.A_eq is not None:
                constraint_matrix_nnz = np.count_nonzero(problem_data.A_eq)
            elif problem_data.A_ub is not None:
                constraint_matrix_nnz = np.count_nonzero(problem_data.A_ub)

            total_elements = num_constraints * num_vars if num_constraints > 0 and num_vars > 0 else 1
            sparsity_ratio = constraint_matrix_nnz / total_elements if total_elements > 0 else 0

            # Build cone information
            cone_info = []
            sdp_cones = []
            soc_cones = []

            # Handle DIMACS/SDPLIB cone structures
            # DIMACS SeDuMi format
            if 'sdp_cones' in cone_structure and cone_structure['sdp_cones']:
                # Group SDP cones by size
                sdp_size_counts = {}
                for size in cone_structure['sdp_cones']:
                    sdp_size_counts[size] = sdp_size_counts.get(size, 0) + 1

                for size, count in sdp_size_counts.items():
                    sdp_cones.append((size, count))
                    cone_info.append(ConeInfo("semi_definite", size * size * count, (size, size)))

            if 'soc_cones' in cone_structure and cone_structure['soc_cones']:
                # Group SOC cones by dimension
                soc_dim_counts = {}
                for dim in cone_structure['soc_cones']:
                    soc_dim_counts[dim] = soc_dim_counts.get(dim, 0) + 1

                for dim, count in soc_dim_counts.items():
                    soc_cones.append((dim, count))
                    cone_info.append(ConeInfo("second_order", dim * count))

            nonneg_vars = cone_structure.get('nonneg_vars', 0)
            free_vars = cone_structure.get('free_vars', 0)

            # Add linear cones
            if nonneg_vars > 0:
                cone_info.append(ConeInfo("non_negative", nonneg_vars))
            if free_vars > 0:
                cone_info.append(ConeInfo("unrestricted", free_vars))

            # Classify problem based on cone structure (prioritize SDP over mixed)
            cone_types = set(cone.cone_type for cone in cone_info)

            if 'semi_definite' in cone_types:
                problem_class = 'SDP'
            elif 'second_order' in cone_types:
                problem_class = 'SOCP'
            else:
                problem_class = 'LP'

            return ProblemStructure(
                num_variables=num_vars,
                num_constraints=num_constraints,
                problem_class=problem_class,
                cone_info=cone_info,
                semi_definite_cones=sdp_cones,
                second_order_cones=soc_cones,
                non_negative_dim=nonneg_vars,
                unrestricted_dim=free_vars,
                has_quadratic_objective=False,
                quadratic_matrix_size=None,
                constraint_matrix_nnz=constraint_matrix_nnz,
                sparsity_ratio=sparsity_ratio
            )

        except Exception as e:
            self.logger.error(f"Error analyzing problem structure: {e}")
            return ProblemStructure(
                num_variables=0,
                num_constraints=0,
                problem_class=problem_data.problem_class or 'UNKNOWN',
                cone_info=[],
                semi_definite_cones=[],
                second_order_cones=[],
                non_negative_dim=0,
                unrestricted_dim=0
            )
    

def analyze_problem_structure(problem_data: ProblemData) -> ProblemStructure:
    """
    Convenience function to analyze problem structure.
    
    Args:
        problem_data: Problem data to analyze
        
    Returns:
        ProblemStructure with comprehensive analysis
    """
    analyzer = ProblemStructureAnalyzer()
    return analyzer.analyze_problem_data(problem_data)


def get_problem_structure_summary(problem_data: ProblemData) -> Dict[str, Any]:
    """
    Get problem structure summary in the format requested by user.
    
    Args:
        problem_data: Problem data to analyze
        
    Returns:
        Dictionary with structure information
    """
    structure = analyze_problem_structure(problem_data)
    return structure.to_dict()


if __name__ == "__main__":
    # Test problem structure analysis
    print("=== Problem Structure Analysis Test ===")
    
    try:
        from scripts.data_loaders.problem_loader import load_problem
        
        # Test with different problem types
        test_problems = ["simple_lp", "simple_qp"]
        
        for problem_name in test_problems:
            try:
                print(f"\nAnalyzing {problem_name}:")
                problem = load_problem(problem_name, "light_set")
                structure = analyze_problem_structure(problem)
                
                print(f"  Problem Class: {structure.problem_class}")
                print(f"  Variables: {structure.num_variables}")
                print(f"  Constraints: {structure.num_constraints}")
                print(f"  Non-negative dim: {structure.non_negative_dim}")
                print(f"  Unrestricted dim: {structure.unrestricted_dim}")
                
                if structure.has_quadratic_objective:
                    print(f"  Quadratic objective: Yes")
                
                # Test user-requested format
                summary = get_problem_structure_summary(problem)
                print(f"  Summary format: {summary}")
                
            except Exception as e:
                print(f"  Error analyzing {problem_name}: {e}")
        
        print("\n✓ Problem structure analysis test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()