"""
MPS File Loader for Linear Programming.

This module provides functionality to load and parse linear programming problems
stored in MPS (Mathematical Programming System) format, which is a standard
file format for linear and mixed-integer programming problems.

MPS Format Structure:
The MPS format is a column-oriented format with several sections:
- NAME: Problem name
- ROWS: Constraint and objective definitions
- COLUMNS: Variable definitions and constraint matrix entries
- RHS: Right-hand side values
- BOUNDS: Variable bounds (optional)
- ENDATA: End of data marker

The problem format is typically:
minimize/maximize c'x
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

logger = get_logger("mps_loader")


class MPSLoader:
    """Loader for linear programming problems in MPS format."""
    
    def __init__(self):
        """Initialize the MPS loader."""
        pass
    
    def load(self, file_path: str) -> ProblemData:
        """
        Load problem from .mps file.
        
        Args:
            file_path: Path to the .mps file
            
        Returns:
            ProblemData object
        """
        problem_name = Path(file_path).stem
        
        # Parse and convert the file
        parsed_data = self.parse_mps_file(file_path)
        problem_data = self.convert_to_problem_data(parsed_data, problem_name)
        
        logger.info(f"Successfully loaded MPS problem: {problem_data}")
        return problem_data
    
    def parse_mps_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse MPS format file.
        
        Args:
            file_path: Path to the .mps file
            
        Returns:
            Dictionary containing parsed MPS problem data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading MPS file: {file_path}")
        
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
            
            logger.debug(f"Parsed MPS file: {len(parsed_data['columns'])} variables, {len(parsed_data['rows'])} constraints")
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
    
    def convert_to_problem_data(self, parsed_data: Dict[str, Any], 
                              problem_name: str) -> ProblemData:
        """
        Convert parsed MPS data to unified ProblemData format.
        
        Args:
            parsed_data: Parsed MPS problem data
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
        
        # Build objective vector
        objective_name = parsed_data['objective_name']
        c = np.zeros(n_vars)
        
        for i, var_name in enumerate(var_names):
            if var_name in parsed_data['columns']:
                var_data = parsed_data['columns'][var_name]
                if objective_name in var_data:
                    c[i] = var_data[objective_name]
        
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
            'source': 'MPS file',
            'format': 'MPS',
            'objective_sense': parsed_data['objective_sense'],
            'original_dimensions': {
                'variables': n_vars,
                'constraints': n_constraints,
                'equality_constraints': len(eq_constraints),
                'inequality_constraints': len(ineq_constraints)
            },
            'variable_names': var_names,
            'constraint_names': list(constraint_rows.keys())
        }
        
        logger.info(f"Converted {problem_name}: LP problem "
                   f"({n_vars} vars, {n_constraints} constraints)")
        
        # Create ProblemData object
        return ProblemData(
            name=problem_name,
            problem_class='LP',
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            metadata=metadata
        )


# Convenience function for backward compatibility
def load_mps_problem(file_path: str, 
                    problem_name: Optional[str] = None) -> ProblemData:
    """
    Convenience function to load an MPS problem.
    
    Args:
        file_path: Path to the .mps file
        problem_name: Optional name for the problem
        
    Returns:
        ProblemData object
    """
    loader = MPSLoader()
    return loader.load(file_path)


if __name__ == "__main__":
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mps_loader.py <path_to_mps_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        problem = load_mps_problem(file_path)
        print(f"Loaded problem: {problem}")
        
        # Display structure summary if available
        structure = problem.get_structure_summary()
        if structure:
            print(f"Structure analysis: {structure}")
            
    except Exception as e:
        print(f"Error loading problem: {e}")
        sys.exit(1)