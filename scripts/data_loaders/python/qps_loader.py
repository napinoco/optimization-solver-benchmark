"""
QPS File Loader for Quadratic Programming.

This module provides functionality to load and parse quadratic programming problems
stored in QPS (Quadratic Programming System) format, which is an extension of the
MPS format that includes quadratic objective terms.

QPS Format Structure:
The QPS format extends MPS with additional sections:
- NAME: Problem name
- ROWS: Constraint and objective definitions
- COLUMNS: Variable definitions and linear constraint matrix entries
- RHS: Right-hand side values
- BOUNDS: Variable bounds (optional)
- QUADOBJ: Quadratic objective terms (QPS extension)
- ENDATA: End of data marker

The problem format is typically:
minimize/maximize (1/2)x'Qx + c'x
subject to Ax {<=, =, >=} b
         l <= x <= u
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loaders.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("qps_loader")


class QPSLoader:
    """Loader for quadratic programming problems in QPS format."""
    
    def __init__(self):
        """Initialize the QPS loader."""
        pass
    
    def load(self, file_path: str) -> ProblemData:
        """
        Load problem from .qps file.
        
        Args:
            file_path: Path to the .qps file
            
        Returns:
            ProblemData object
        """
        problem_name = Path(file_path).stem
        
        # Parse and convert the file
        parsed_data = self.parse_qps_file(file_path)
        problem_data = self.convert_to_problem_data(parsed_data, problem_name)
        
        logger.info(f"Successfully loaded QPS problem: {problem_data}")
        return problem_data
    
    def parse_qps_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse QPS format file.
        
        Args:
            file_path: Path to the .qps file
            
        Returns:
            Dictionary containing parsed QPS problem data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading QPS file: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                lines = [line.rstrip() for line in f.readlines()]
            
            # Parse sections
            parsed_data = {
                'name': '',
                'rows': {},
                'columns': {},
                'rhs': {},
                'bounds': {},
                'quadobj': {},  # QPS extension for quadratic terms
                'objective_name': '',
                'objective_sense': 'MIN'  # Default to minimization
            }
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('*'):  # Skip empty lines and comments
                    continue
                
                # Check for section headers
                if line.startswith('NAME'):
                    parsed_data['name'] = line.split()[1] if len(line.split()) > 1 else ''
                    continue
                elif line.startswith('ROWS'):
                    current_section = 'ROWS'
                    continue
                elif line.startswith('COLUMNS'):
                    current_section = 'COLUMNS'
                    continue
                elif line.startswith('RHS'):
                    current_section = 'RHS'
                    continue
                elif line.startswith('BOUNDS'):
                    current_section = 'BOUNDS'
                    continue
                elif line.startswith('QUADOBJ'):
                    current_section = 'QUADOBJ'
                    continue
                elif line.startswith('ENDATA'):
                    break
                
                # Parse section content
                if current_section == 'ROWS':
                    self._parse_rows_line(line, parsed_data)
                elif current_section == 'COLUMNS':
                    self._parse_columns_line(line, parsed_data)
                elif current_section == 'RHS':
                    self._parse_rhs_line(line, parsed_data)
                elif current_section == 'BOUNDS':
                    self._parse_bounds_line(line, parsed_data)
                elif current_section == 'QUADOBJ':
                    self._parse_quadobj_line(line, parsed_data)
            
            logger.debug(f"Parsed QPS file: {len(parsed_data['columns'])} variables, "
                        f"{len(parsed_data['rows'])} constraints, "
                        f"{len(parsed_data['quadobj'])} quadratic terms")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise
    
    def _parse_rows_line(self, line: str, parsed_data: Dict[str, Any]):
        """Parse a line from the ROWS section."""
        parts = line.split()
        if len(parts) >= 2:
            row_type = parts[0]
            row_name = parts[1]
            
            if row_type == 'N':  # Objective row
                if not parsed_data['objective_name']:  # Take first objective
                    parsed_data['objective_name'] = row_name
            
            parsed_data['rows'][row_name] = {
                'type': row_type,
                'coefficients': {}
            }
    
    def _parse_columns_line(self, line: str, parsed_data: Dict[str, Any]):
        """Parse a line from the COLUMNS section."""
        parts = line.split()
        if len(parts) >= 3:
            col_name = parts[0]
            
            if col_name not in parsed_data['columns']:
                parsed_data['columns'][col_name] = {}
            
            # Handle multiple coefficient pairs in one line
            for i in range(1, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    row_name = parts[i]
                    coefficient = float(parts[i + 1])
                    
                    if row_name not in parsed_data['columns'][col_name]:
                        parsed_data['columns'][col_name][row_name] = 0.0
                    parsed_data['columns'][col_name][row_name] += coefficient
    
    def _parse_rhs_line(self, line: str, parsed_data: Dict[str, Any]):
        """Parse a line from the RHS section."""
        parts = line.split()
        if len(parts) >= 3:
            # Handle multiple RHS pairs in one line
            for i in range(1, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    row_name = parts[i]
                    rhs_value = float(parts[i + 1])
                    parsed_data['rhs'][row_name] = rhs_value
    
    def _parse_bounds_line(self, line: str, parsed_data: Dict[str, Any]):
        """Parse a line from the BOUNDS section."""
        parts = line.split()
        if len(parts) >= 3:
            bound_type = parts[0]
            bound_name = parts[1]  # Bound set name (often ignored)
            var_name = parts[2]
            bound_value = float(parts[3]) if len(parts) > 3 else 0.0
            
            if var_name not in parsed_data['bounds']:
                parsed_data['bounds'][var_name] = {'lower': 0.0, 'upper': float('inf')}
            
            if bound_type == 'LO':  # Lower bound
                parsed_data['bounds'][var_name]['lower'] = bound_value
            elif bound_type == 'UP':  # Upper bound
                parsed_data['bounds'][var_name]['upper'] = bound_value
            elif bound_type == 'FR':  # Free variable
                parsed_data['bounds'][var_name]['lower'] = float('-inf')
                parsed_data['bounds'][var_name]['upper'] = float('inf')
            elif bound_type == 'MI':  # Minus infinity
                parsed_data['bounds'][var_name]['lower'] = float('-inf')
            elif bound_type == 'PL':  # Plus infinity
                parsed_data['bounds'][var_name]['upper'] = float('inf')
    
    def _parse_quadobj_line(self, line: str, parsed_data: Dict[str, Any]):
        """Parse a line from the QUADOBJ section (quadratic objective terms)."""
        parts = line.split()
        if len(parts) >= 3:
            var1 = parts[0]
            var2 = parts[1]
            coeff = float(parts[2])
            
            # Store quadratic coefficient (symmetric matrix)
            if var1 not in parsed_data['quadobj']:
                parsed_data['quadobj'][var1] = {}
            parsed_data['quadobj'][var1][var2] = coeff
            
            # Ensure symmetry for off-diagonal terms
            if var1 != var2:
                if var2 not in parsed_data['quadobj']:
                    parsed_data['quadobj'][var2] = {}
                parsed_data['quadobj'][var2][var1] = coeff
    
    def convert_to_problem_data(self, parsed_data: Dict[str, Any], 
                              problem_name: str) -> ProblemData:
        """
        Convert parsed QPS data to unified ProblemData format.
        
        Args:
            parsed_data: Parsed QPS problem data
            problem_name: Name for the problem
            
        Returns:
            ProblemData object
        """
        # Get variable names and constraint names
        var_names = list(parsed_data['columns'].keys())
        constraint_rows = {name: info for name, info in parsed_data['rows'].items() 
                          if info['type'] != 'N'}
        
        n_vars = len(var_names)
        n_constraints = len(constraint_rows)
        
        # Build linear objective vector
        objective_name = parsed_data['objective_name']
        c = np.zeros(n_vars)
        
        for i, var_name in enumerate(var_names):
            if var_name in parsed_data['columns']:
                var_data = parsed_data['columns'][var_name]
                if objective_name in var_data:
                    c[i] = var_data[objective_name]
        
        # Build quadratic objective matrix Q
        Q = np.zeros((n_vars, n_vars))
        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                if var1 in parsed_data['quadobj'] and var2 in parsed_data['quadobj'][var1]:
                    Q[i, j] = parsed_data['quadobj'][var1][var2]
        
        # Separate equality and inequality constraints
        eq_constraints = []
        ineq_constraints = []
        eq_rhs = []
        ineq_rhs = []
        
        for row_name, row_info in constraint_rows.items():
            row_type = row_info['type']
            
            # Build constraint vector
            constraint_vec = np.zeros(n_vars)
            for i, var_name in enumerate(var_names):
                if var_name in parsed_data['columns']:
                    var_data = parsed_data['columns'][var_name]
                    if row_name in var_data:
                        constraint_vec[i] = var_data[row_name]
            
            # Get RHS value
            rhs_value = parsed_data['rhs'].get(row_name, 0.0)
            
            if row_type == 'E':  # Equality
                eq_constraints.append(constraint_vec)
                eq_rhs.append(rhs_value)
            elif row_type == 'L':  # Less than or equal
                ineq_constraints.append(constraint_vec)
                ineq_rhs.append(rhs_value)
            elif row_type == 'G':  # Greater than or equal (convert to <=)
                ineq_constraints.append(-constraint_vec)
                ineq_rhs.append(-rhs_value)
        
        # Convert to matrices
        A_eq = np.array(eq_constraints) if eq_constraints else None
        b_eq = np.array(eq_rhs) if eq_rhs else None
        A_ub = np.array(ineq_constraints) if ineq_constraints else None
        b_ub = np.array(ineq_rhs) if ineq_rhs else None
        
        # Build bounds
        bounds = []
        for var_name in var_names:
            if var_name in parsed_data['bounds']:
                bound_info = parsed_data['bounds'][var_name]
                bounds.append((bound_info['lower'], bound_info['upper']))
            else:
                bounds.append((0.0, None))  # Default: non-negative
        
        # Create metadata
        metadata = {
            'source': 'QPS file',
            'format': 'QPS',
            'objective_sense': parsed_data['objective_sense'],
            'has_quadratic_objective': np.any(Q != 0),
            'quadratic_matrix_density': np.count_nonzero(Q) / (n_vars * n_vars) if n_vars > 0 else 0,
            'original_dimensions': {
                'variables': n_vars,
                'constraints': n_constraints,
                'equality_constraints': len(eq_constraints),
                'inequality_constraints': len(ineq_constraints),
                'quadratic_terms': len(parsed_data['quadobj'])
            },
            'variable_names': var_names,
            'constraint_names': list(constraint_rows.keys())
        }
        
        logger.info(f"Converted {problem_name}: QP problem "
                   f"({n_vars} vars, {n_constraints} constraints, "
                   f"{np.count_nonzero(Q)} quadratic terms)")
        
        # Create ProblemData object
        problem_data = ProblemData(
            name=problem_name,
            problem_class='QP',
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            metadata=metadata
        )
        
        # Add quadratic matrix to metadata for QP solvers
        problem_data.Q = Q
        
        return problem_data


# Convenience function for backward compatibility
def load_qps_problem(file_path: str, 
                    problem_name: Optional[str] = None) -> ProblemData:
    """
    Convenience function to load a QPS problem.
    
    Args:
        file_path: Path to the .qps file
        problem_name: Optional name for the problem
        
    Returns:
        ProblemData object
    """
    loader = QPSLoader()
    return loader.load(file_path)


if __name__ == "__main__":
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python qps_loader.py <path_to_qps_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        problem = load_qps_problem(file_path)
        print(f"Loaded problem: {problem}")
        
        # Display structure summary if available
        structure = problem.get_structure_summary()
        if structure:
            print(f"Structure analysis: {structure}")
            
    except Exception as e:
        print(f"Error loading problem: {e}")
        sys.exit(1)