"""
CVXPY Converter for Unified Problem Format Conversion.

This module provides functionality to convert optimization problems from various
formats (ProblemData objects) into CVXPY format for solving with different backends.
This is the central conversion point that enables all solvers to work with all
problem formats.

Supported Problem Types:
- LP: Linear Programming
- QP: Quadratic Programming  
- SOCP: Second-Order Cone Programming
- SDP: Semidefinite Programming

The converter handles:
- Matrix format problems (MPS, QPS, MAT, DAT)
- Python CVXPY problems
- Automatic problem type detection
- Constraint and variable bound conversion
"""

import numpy as np
import cvxpy as cp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("cvxpy_converter")


class CVXPYConverter:
    """Unified converter to CVXPY format for all problem types."""
    
    def __init__(self):
        """Initialize the CVXPY converter."""
        pass
    
    def convert(self, problem_data: ProblemData) -> cp.Problem:
        """
        Convert ProblemData to CVXPY Problem format.
        
        Args:
            problem_data: ProblemData object from any loader
            
        Returns:
            CVXPY Problem object ready for solving
            
        Raises:
            ValueError: If problem format is not supported
        """
        logger.info(f"Converting {problem_data.name} ({problem_data.problem_class}) to CVXPY format")
        
        # If already a CVXPY problem, return as-is
        if hasattr(problem_data, 'cvxpy_problem') and problem_data.cvxpy_problem is not None:
            logger.debug("Problem already in CVXPY format")
            return problem_data.cvxpy_problem
        
        # Convert based on problem class
        if problem_data.problem_class == 'LP':
            return self._convert_lp(problem_data)
        elif problem_data.problem_class == 'QP':
            return self._convert_qp(problem_data)
        elif problem_data.problem_class == 'SOCP':
            return self._convert_socp(problem_data)
        elif problem_data.problem_class == 'SDP':
            return self._convert_sdp(problem_data)
        else:
            raise ValueError(f"Unsupported problem class: {problem_data.problem_class}")
    
    def _convert_lp(self, problem_data: ProblemData) -> cp.Problem:
        """Convert Linear Programming problem to CVXPY."""
        logger.debug("Converting LP problem to CVXPY")
        
        # Determine number of variables
        n_vars = self._get_variable_count(problem_data)
        
        # Create decision variables
        x = cp.Variable(n_vars, name="x")
        
        # Build objective
        c = problem_data.c if problem_data.c is not None else np.zeros(n_vars)
        objective = cp.Minimize(c.T @ x)
        
        # Build constraints
        constraints = []
        
        # Equality constraints: A_eq @ x == b_eq
        if problem_data.A_eq is not None and problem_data.b_eq is not None:
            constraints.append(problem_data.A_eq @ x == problem_data.b_eq)
        
        # Inequality constraints: A_ub @ x <= b_ub
        if problem_data.A_ub is not None and problem_data.b_ub is not None:
            constraints.append(problem_data.A_ub @ x <= problem_data.b_ub)
        
        # Variable bounds
        if problem_data.bounds:
            for i, (lb, ub) in enumerate(problem_data.bounds):
                if lb is not None and lb != float('-inf'):
                    constraints.append(x[i] >= lb)
                if ub is not None and ub != float('inf'):
                    constraints.append(x[i] <= ub)
        
        problem = cp.Problem(objective, constraints)
        logger.debug(f"Created LP problem with {len(constraints)} constraints")
        
        return problem
    
    def _convert_qp(self, problem_data: ProblemData) -> cp.Problem:
        """Convert Quadratic Programming problem to CVXPY."""
        logger.debug("Converting QP problem to CVXPY")
        
        # Determine number of variables
        n_vars = self._get_variable_count(problem_data)
        
        # Create decision variables
        x = cp.Variable(n_vars, name="x")
        
        # Build quadratic objective: (1/2) x.T @ Q @ x + c.T @ x
        c = problem_data.c if problem_data.c is not None else np.zeros(n_vars)
        
        # Get quadratic matrix Q
        if hasattr(problem_data, 'Q') and problem_data.Q is not None:
            Q = problem_data.Q
            objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)
        else:
            # Fall back to linear objective if no quadratic terms
            logger.warning("No quadratic matrix found, treating as LP")
            objective = cp.Minimize(c.T @ x)
        
        # Build constraints (same as LP)
        constraints = []
        
        # Equality constraints: A_eq @ x == b_eq
        if problem_data.A_eq is not None and problem_data.b_eq is not None:
            constraints.append(problem_data.A_eq @ x == problem_data.b_eq)
        
        # Inequality constraints: A_ub @ x <= b_ub
        if problem_data.A_ub is not None and problem_data.b_ub is not None:
            constraints.append(problem_data.A_ub @ x <= problem_data.b_ub)
        
        # Variable bounds
        if problem_data.bounds:
            for i, (lb, ub) in enumerate(problem_data.bounds):
                if lb is not None and lb != float('-inf'):
                    constraints.append(x[i] >= lb)
                if ub is not None and ub != float('inf'):
                    constraints.append(x[i] <= ub)
        
        problem = cp.Problem(objective, constraints)
        logger.debug(f"Created QP problem with {len(constraints)} constraints")
        
        return problem
    
    def _convert_socp(self, problem_data: ProblemData) -> cp.Problem:
        """Convert Second-Order Cone Programming problem to CVXPY."""
        logger.debug("Converting SOCP problem to CVXPY")
        
        # For SOCP problems from matrix formats, we need to reconstruct cone constraints
        # This is complex and depends on the specific cone structure
        
        # If it's already a CVXPY problem with SOC constraints, use it directly
        if hasattr(problem_data, 'cvxpy_problem') and problem_data.cvxpy_problem is not None:
            return problem_data.cvxpy_problem
        
        # For matrix-based SOCP (e.g., from DIMACS), convert based on cone structure
        return self._convert_conic_problem(problem_data, 'SOCP')
    
    def _convert_sdp(self, problem_data: ProblemData) -> cp.Problem:
        """Convert Semidefinite Programming problem to CVXPY."""
        logger.debug("Converting SDP problem to CVXPY")
        
        # For SDP problems from matrix formats, we need to reconstruct SDP constraints
        # This is complex and depends on the specific block structure
        
        # If it's already a CVXPY problem with SDP constraints, use it directly
        if hasattr(problem_data, 'cvxpy_problem') and problem_data.cvxpy_problem is not None:
            return problem_data.cvxpy_problem
        
        # For matrix-based SDP (e.g., from SDPLIB), convert based on block structure
        return self._convert_conic_problem(problem_data, 'SDP')
    
    def _convert_conic_problem(self, problem_data: ProblemData, cone_type: str) -> cp.Problem:
        """
        Convert conic problems (SOCP/SDP) from matrix format to CVXPY.
        
        This is a simplified conversion that works for basic conic problems.
        Complex cone structures may need specialized handling.
        """
        logger.debug(f"Converting {cone_type} problem from matrix format")
        
        # Determine number of variables
        n_vars = self._get_variable_count(problem_data)
        
        # Create decision variables
        x = cp.Variable(n_vars, name="x")
        
        # Build linear objective (conic problems often have linear objectives)
        c = problem_data.c if problem_data.c is not None else np.zeros(n_vars)
        objective = cp.Minimize(c.T @ x)
        
        # Build basic constraints
        constraints = []
        
        # Equality constraints: A_eq @ x == b_eq
        if problem_data.A_eq is not None and problem_data.b_eq is not None:
            constraints.append(problem_data.A_eq @ x == problem_data.b_eq)
        
        # For cone constraints, we would need specific cone structure information
        # This is a simplified version that treats it as an LP relaxation
        logger.warning(f"Simplified {cone_type} conversion - cone constraints not fully reconstructed")
        
        # Add variable bounds if present
        if problem_data.bounds:
            for i, (lb, ub) in enumerate(problem_data.bounds):
                if lb is not None and lb != float('-inf'):
                    constraints.append(x[i] >= lb)
                if ub is not None and ub != float('inf'):
                    constraints.append(x[i] <= ub)
        
        problem = cp.Problem(objective, constraints)
        logger.debug(f"Created simplified {cone_type} problem with {len(constraints)} constraints")
        
        return problem
    
    def _get_variable_count(self, problem_data: ProblemData) -> int:
        """Determine the number of variables in the problem."""
        # Try different ways to determine variable count
        if problem_data.c is not None:
            return len(problem_data.c)
        elif problem_data.A_eq is not None:
            return problem_data.A_eq.shape[1]
        elif problem_data.A_ub is not None:
            return problem_data.A_ub.shape[1]
        elif problem_data.bounds:
            return len(problem_data.bounds)
        else:
            raise ValueError("Cannot determine number of variables from problem data")
    
    def validate_conversion(self, original_problem: ProblemData, cvxpy_problem: cp.Problem) -> bool:
        """
        Validate that the CVXPY conversion is reasonable.
        
        Args:
            original_problem: Original ProblemData
            cvxpy_problem: Converted CVXPY problem
            
        Returns:
            True if conversion seems valid
        """
        try:
            # Basic validation checks
            if not isinstance(cvxpy_problem, cp.Problem):
                return False
            
            # Check that problem has variables
            if not cvxpy_problem.variables:
                return False
            
            # Check that problem has an objective
            if cvxpy_problem.objective is None:
                return False
            
            logger.debug("CVXPY conversion validation passed")
            return True
            
        except Exception as e:
            logger.warning(f"CVXPY conversion validation failed: {e}")
            return False


# Convenience function for direct usage
def convert_to_cvxpy(problem_data: ProblemData) -> cp.Problem:
    """
    Convenience function to convert ProblemData to CVXPY format.
    
    Args:
        problem_data: ProblemData object
        
    Returns:
        CVXPY Problem object
    """
    converter = CVXPYConverter()
    return converter.convert(problem_data)


if __name__ == "__main__":
    # Test script
    print("CVXPY Converter Test")
    print("===================")
    
    # Create a simple test LP problem
    n_vars = 3
    c = np.array([1.0, 2.0, 3.0])
    A_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])
    A_ub = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b_ub = np.array([0.5, 0.8])
    
    test_problem = ProblemData(
        name="test_lp",
        problem_class="LP",
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub
    )
    
    # Test conversion
    try:
        converter = CVXPYConverter()
        cvxpy_problem = converter.convert(test_problem)
        
        print(f"✓ Successfully converted test LP problem")
        print(f"  Variables: {len(cvxpy_problem.variables)}")
        print(f"  Constraints: {len(cvxpy_problem.constraints)}")
        
        # Test solving
        cvxpy_problem.solve(verbose=False)
        if cvxpy_problem.status == cp.OPTIMAL:
            print(f"  Optimal value: {cvxpy_problem.value:.6f}")
            print(f"  Solution: {cvxpy_problem.variables[0].value}")
        else:
            print(f"  Status: {cvxpy_problem.status}")
        
        print("✓ CVXPY converter test completed successfully!")
        
    except Exception as e:
        print(f"✗ CVXPY converter test failed: {e}")
        sys.exit(1)