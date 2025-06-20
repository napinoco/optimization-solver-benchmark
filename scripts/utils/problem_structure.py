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
from scripts.benchmark.problem_loader import ProblemData

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
        self.logger.debug(f"Analyzing structure of problem")
        
        try:
            # Check if this is an external library problem with cone structure
            if (problem_data.metadata and 
                'cone_structure' in problem_data.metadata and 
                problem_data.metadata.get('source') in ['DIMACS', 'SDPLIB']):
                return self._analyze_external_library_problem(problem_data)
            elif problem_data.cvxpy_problem is not None:
                # Analyze CVXPY problem directly
                return self._analyze_cvxpy_problem(problem_data)
            else:
                # Analyze from matrix data
                return self._analyze_matrix_problem(problem_data)
                
        except Exception as e:
            self.logger.error(f"Error analyzing problem structure: {e}")
            return self._create_fallback_structure(problem_data)
    
    def _analyze_external_library_problem(self, problem_data: ProblemData) -> ProblemStructure:
        """Analyze external library problem using metadata cone structure."""
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
        if problem_data.metadata.get('source') == 'DIMACS':
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
            
        elif problem_data.metadata.get('source') == 'SDPLIB':
            # SDPLIB SDPA format
            if 'sdp_cones' in cone_structure and cone_structure['sdp_cones']:
                # Group SDP cones by size
                sdp_size_counts = {}
                for size in cone_structure['sdp_cones']:
                    sdp_size_counts[size] = sdp_size_counts.get(size, 0) + 1
                
                for size, count in sdp_size_counts.items():
                    sdp_cones.append((size, count))
                    cone_info.append(ConeInfo("semi_definite", size * size * count, (size, size)))
            
            nonneg_vars = cone_structure.get('linear_vars', 0)
            free_vars = 0
        else:
            nonneg_vars = 0
            free_vars = 0
        
        # Add linear cones
        if nonneg_vars > 0:
            cone_info.append(ConeInfo("non_negative", nonneg_vars))
        if free_vars > 0:
            cone_info.append(ConeInfo("unrestricted", free_vars))
        
        # Classify problem based on cone structure (prioritize SDP over mixed)
        problem_class = self._classify_problem_type_from_cones(cone_info)
        
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
    
    def _classify_problem_type_from_cones(self, cone_info: List[ConeInfo]) -> str:
        """Classify problem type based on cone structure (SDP takes priority)."""
        cone_types = set(cone.cone_type for cone in cone_info)
        
        # SDP takes priority over mixed problems
        if 'semi_definite' in cone_types:
            return 'SDP'
        
        # Check for SOCP
        if 'second_order' in cone_types:
            return 'SOCP'
        
        # Default to LP
        return 'LP'
    
    def _analyze_cvxpy_problem(self, problem_data: ProblemData) -> ProblemStructure:
        """Analyze a CVXPY problem to determine structure."""
        cvx_problem = problem_data.cvxpy_problem
        
        # Get basic dimensions
        try:
            num_vars = cvx_problem.size_metrics.num_scalar_variables
        except AttributeError:
            # Fallback for older CVXPY versions
            num_vars = sum(var.size for var in cvx_problem.variables())
        
        try:
            num_constraints = (cvx_problem.size_metrics.num_scalar_eq_constraints + 
                             cvx_problem.size_metrics.num_scalar_leq_constraints)
        except AttributeError:
            num_constraints = len(cvx_problem.constraints)
        
        # Analyze cone structure
        cone_info = []
        semi_definite_cones = []
        second_order_cones = []
        non_negative_dim = 0
        unrestricted_dim = 0
        
        # Analyze constraints to identify cone types
        for constraint in cvx_problem.constraints:
            cone_analysis = self._analyze_constraint_cone(constraint)
            if cone_analysis:
                cone_info.append(cone_analysis)
                
                if cone_analysis.cone_type == 'semi_definite':
                    size = int(np.sqrt(cone_analysis.dimension))
                    semi_definite_cones.append((size, cone_analysis.count))
                elif cone_analysis.cone_type == 'second_order':
                    second_order_cones.append((cone_analysis.dimension, cone_analysis.count))
                elif cone_analysis.cone_type == 'non_negative':
                    non_negative_dim += cone_analysis.dimension
                elif cone_analysis.cone_type == 'unrestricted':
                    unrestricted_dim += cone_analysis.dimension
        
        # Classify problem type
        problem_class = self._classify_problem_type(
            cone_info, 
            has_quadratic_objective=self._has_quadratic_objective(cvx_problem)
        )
        
        return ProblemStructure(
            num_variables=num_vars,
            num_constraints=num_constraints,
            problem_class=problem_class,
            cone_info=cone_info,
            semi_definite_cones=semi_definite_cones,
            second_order_cones=second_order_cones,
            non_negative_dim=non_negative_dim,
            unrestricted_dim=unrestricted_dim,
            has_quadratic_objective=self._has_quadratic_objective(cvx_problem)
        )
    
    def _analyze_matrix_problem(self, problem_data: ProblemData) -> ProblemStructure:
        """Analyze problem from matrix data (LP/QP format)."""
        
        # Determine dimensions
        if problem_data.c is not None:
            num_vars = len(problem_data.c)
        elif problem_data.P is not None:
            num_vars = problem_data.P.shape[0]
        else:
            num_vars = 0
        
        # Count constraints
        num_constraints = 0
        constraint_matrix_nnz = 0
        
        if problem_data.A_ub is not None:
            num_constraints += problem_data.A_ub.shape[0]
            constraint_matrix_nnz += np.count_nonzero(problem_data.A_ub)
        
        if problem_data.A_eq is not None:
            num_constraints += problem_data.A_eq.shape[0]
            constraint_matrix_nnz += np.count_nonzero(problem_data.A_eq)
        
        # Calculate sparsity ratio
        total_elements = num_constraints * num_vars if num_constraints > 0 and num_vars > 0 else 1
        sparsity_ratio = constraint_matrix_nnz / total_elements if total_elements > 0 else 0
        
        # Analyze bounds to determine variable cone types
        non_negative_dim = 0
        unrestricted_dim = 0
        
        if problem_data.bounds:
            for bound in problem_data.bounds:
                if bound is None or bound == (None, None):
                    unrestricted_dim += 1
                elif isinstance(bound, tuple):
                    lower, upper = bound
                    if lower is not None and lower >= 0 and upper is None:
                        non_negative_dim += 1
                    else:
                        unrestricted_dim += 1
                else:
                    non_negative_dim += 1  # Default assume non-negative
        else:
            # Default: all variables non-negative
            non_negative_dim = num_vars
        
        # Determine problem class
        has_quadratic = problem_data.P is not None
        problem_class = "QP" if has_quadratic else "LP"
        
        # Create cone info
        cone_info = []
        if non_negative_dim > 0:
            cone_info.append(ConeInfo("non_negative", non_negative_dim))
        if unrestricted_dim > 0:
            cone_info.append(ConeInfo("unrestricted", unrestricted_dim))
        
        quadratic_matrix_size = None
        if has_quadratic and problem_data.P is not None:
            quadratic_matrix_size = problem_data.P.shape
        
        return ProblemStructure(
            num_variables=num_vars,
            num_constraints=num_constraints,
            problem_class=problem_class,
            cone_info=cone_info,
            semi_definite_cones=[],
            second_order_cones=[],
            non_negative_dim=non_negative_dim,
            unrestricted_dim=unrestricted_dim,
            has_quadratic_objective=has_quadratic,
            quadratic_matrix_size=quadratic_matrix_size,
            constraint_matrix_nnz=constraint_matrix_nnz,
            sparsity_ratio=sparsity_ratio
        )
    
    def _analyze_constraint_cone(self, constraint) -> Optional[ConeInfo]:
        """Analyze a CVXPY constraint to determine its cone type."""
        try:
            # Check for PSD constraints
            if hasattr(constraint, 'expr') and hasattr(constraint.expr, 'expr'):
                expr = constraint.expr.expr if hasattr(constraint.expr, 'expr') else constraint.expr
                
                # Check for PSD (semi-definite) constraints
                if hasattr(expr, 'args') and len(expr.args) > 0:
                    for arg in expr.args:
                        if hasattr(arg, 'shape') and len(arg.shape) == 2 and arg.shape[0] == arg.shape[1]:
                            # Square matrix - likely PSD constraint
                            matrix_size = arg.shape[0]
                            return ConeInfo("semi_definite", matrix_size * matrix_size, (matrix_size, matrix_size))
                
                # Check for SOC (second-order cone) constraints
                if 'norm' in str(constraint).lower():
                    # Estimate dimension from constraint string representation
                    constraint_str = str(constraint)
                    # This is a heuristic - in practice we'd need more sophisticated parsing
                    if hasattr(constraint.expr, 'shape'):
                        dim = constraint.expr.shape[0] if len(constraint.expr.shape) > 0 else 1
                        return ConeInfo("second_order", dim)
            
            # Check for simple inequality/equality constraints (non-negative cone)
            if '>=' in str(constraint) or '<=' in str(constraint):
                if hasattr(constraint, 'expr') and hasattr(constraint.expr, 'shape'):
                    dim = constraint.expr.shape[0] if len(constraint.expr.shape) > 0 else 1
                    return ConeInfo("non_negative", dim)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Could not analyze constraint cone: {e}")
            return None
    
    def _has_quadratic_objective(self, cvx_problem) -> bool:
        """Check if CVXPY problem has quadratic objective."""
        try:
            obj_str = str(cvx_problem.objective)
            return 'quad_form' in obj_str.lower() or 'quadratic' in obj_str.lower()
        except Exception:
            return False
    
    def _classify_problem_type(self, cone_info: List[ConeInfo], has_quadratic_objective: bool = False) -> str:
        """Classify problem type based on cone structure."""
        
        cone_types = set(cone.cone_type for cone in cone_info)
        
        # Check for SDP
        if 'semi_definite' in cone_types:
            return 'SDP'
        
        # Check for SOCP
        if 'second_order' in cone_types:
            return 'SOCP'
        
        # Check for QP
        if has_quadratic_objective:
            return 'QP'
        
        # Default to LP
        return 'LP'
    
    def _create_fallback_structure(self, problem_data: ProblemData) -> ProblemStructure:
        """Create minimal structure info when analysis fails."""
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
        from scripts.benchmark.problem_loader import load_problem
        
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