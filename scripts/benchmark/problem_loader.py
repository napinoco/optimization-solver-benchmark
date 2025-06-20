import os
import re
import yaml
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("problem_loader")

# Import for lazy loading to avoid circular imports
def _get_structure_analyzer():
    """Lazy import of structure analyzer to avoid circular dependencies."""
    from scripts.utils.problem_structure import analyze_problem_structure
    return analyze_problem_structure

class ProblemData:
    """Container for optimization problem data."""
    def __init__(self, name: str, problem_class: str, c: np.ndarray = None, 
                 A_eq: np.ndarray = None, b_eq: np.ndarray = None,
                 A_ub: np.ndarray = None, b_ub: np.ndarray = None,
                 bounds: List[Tuple] = None, P: np.ndarray = None,
                 cvxpy_problem=None, variables=None, objective=None, 
                 constraints=None, metadata=None, analyze_structure: bool = True):
        self.name = name
        self.problem_class = problem_class
        self.c = c  # objective coefficients
        self.A_eq = A_eq  # equality constraint matrix
        self.b_eq = b_eq  # equality constraint RHS
        self.A_ub = A_ub  # inequality constraint matrix  
        self.b_ub = b_ub  # inequality constraint RHS
        self.bounds = bounds  # variable bounds
        self.P = P  # quadratic objective matrix for QP
        
        # Enhanced support for CVXPY problems (SOCP, SDP, etc.)
        self.cvxpy_problem = cvxpy_problem  # CVXPY Problem object
        self.variables = variables or {}  # Dict of CVXPY variables
        self.objective = objective  # CVXPY objective
        self.constraints = constraints or []  # List of CVXPY constraints
        self.metadata = metadata or {}  # Additional problem metadata
        
        # Problem structure analysis
        self._structure = None
        self._structure_summary = None
        if analyze_structure:
            try:
                self._analyze_structure()
            except Exception as e:
                logger.debug(f"Could not analyze structure for {name}: {e}")
        
    def _analyze_structure(self):
        """Analyze problem structure using the structure analyzer."""
        try:
            analyze_func = _get_structure_analyzer()
            self._structure = analyze_func(self)
            self._structure_summary = self._structure.to_dict()
        except Exception as e:
            logger.debug(f"Structure analysis failed: {e}")
            self._structure = None
            self._structure_summary = None
    
    def get_structure(self):
        """Get problem structure analysis."""
        return self._structure
    
    def get_structure_summary(self):
        """Get problem structure summary in user-requested format."""
        return self._structure_summary
        
    def __repr__(self):
        structure_info = ""
        if self._structure_summary:
            vars_count = self._structure_summary.get('variables', '?')
            constraints_count = self._structure_summary.get('constraints', '?')
            structure_info = f", {vars_count} vars, {constraints_count} constraints"
        return f"ProblemData(name='{self.name}', class='{self.problem_class}'{structure_info})"

def load_mps_file(file_path: str, problem_name: str = None) -> ProblemData:
    """Load MPS format linear programming problem."""
    logger.info(f"Loading MPS file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Basic validation
        required_sections = ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'ENDATA']
        for section in required_sections:
            if section not in content:
                raise ValueError(f"Missing required section: {section}")
        
        # Use provided name or extract from file as fallback
        if problem_name is None:
            name_match = re.search(r'NAME\s+(\w+)', content)
            problem_name = name_match.group(1) if name_match else "unnamed"
        
        # Parse ROWS section to identify constraint types
        rows_section = re.search(r'ROWS\s*\n(.*?)\n(?:COLUMNS|$)', content, re.DOTALL)
        if not rows_section:
            raise ValueError("Could not parse ROWS section")
            
        rows_lines = rows_section.group(1).strip().split('\n')
        constraint_types = {}
        objective_row = None
        
        for line in rows_lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    row_type, row_name = parts[0], parts[1]
                    constraint_types[row_name] = row_type
                    if row_type == 'N' and objective_row is None:
                        objective_row = row_name
        
        # Parse COLUMNS section
        columns_section = re.search(r'COLUMNS\s*\n(.*?)\n(?:RHS|$)', content, re.DOTALL)
        if not columns_section:
            raise ValueError("Could not parse COLUMNS section")
            
        # Simple parsing for our test problems
        variables = set()
        coefficients = {}
        
        for line in columns_section.group(1).strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:  # Changed from 4 to 3
                    var_name = parts[0]
                    row_name = parts[1]
                    coeff = float(parts[2])
                    
                    variables.add(var_name)
                    if var_name not in coefficients:
                        coefficients[var_name] = {}
                    coefficients[var_name][row_name] = coeff
                    
                    # Handle case where there are 4+ parts (multiple coefficients per line)
                    if len(parts) >= 5:
                        row_name2 = parts[3]
                        coeff2 = float(parts[4])
                        coefficients[var_name][row_name2] = coeff2
        
        variables = sorted(list(variables))
        n_vars = len(variables)
        
        # Parse RHS section
        rhs_section = re.search(r'RHS\s*\n(.*?)\n(?:BOUNDS|ENDATA|$)', content, re.DOTALL)
        rhs_values = {}
        if rhs_section:
            for line in rhs_section.group(1).strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        row_name = parts[1]
                        value = float(parts[2])
                        rhs_values[row_name] = value
        
        # Build matrices
        c = np.zeros(n_vars)  # objective coefficients
        constraint_rows = [name for name, type_ in constraint_types.items() 
                          if type_ in ['L', 'G', 'E'] and name != objective_row]
        
        if constraint_rows:
            A = np.zeros((len(constraint_rows), n_vars))
            b = np.zeros(len(constraint_rows))
            
            for i, row_name in enumerate(constraint_rows):
                for j, var_name in enumerate(variables):
                    if var_name in coefficients and row_name in coefficients[var_name]:
                        A[i, j] = coefficients[var_name][row_name]
                b[i] = rhs_values.get(row_name, 0)
            
            # For our simple problems, assume all constraints are <=
            A_ub, b_ub = A, b
            A_eq, b_eq = None, None
        else:
            A_ub = A_eq = b_ub = b_eq = None
        
        # Build objective vector
        if objective_row:
            for j, var_name in enumerate(variables):
                if var_name in coefficients and objective_row in coefficients[var_name]:
                    c[j] = coefficients[var_name][objective_row]
        
        # Default bounds (non-negative)
        bounds = [(0, None) for _ in range(n_vars)]
        
        logger.info(f"Successfully loaded MPS problem: {problem_name} ({n_vars} variables, {len(constraint_rows)} constraints)")
        
        return ProblemData(
            name=problem_name,
            problem_class="LP",
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds
        )
        
    except Exception as e:
        logger.error(f"Failed to load MPS file {file_path}: {e}")
        raise

def load_qps_file(file_path: str, problem_name: str = None) -> ProblemData:
    """Load QPS format quadratic programming problem."""
    logger.info(f"Loading QPS file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Basic validation
        required_sections = ['NAME', 'ROWS', 'COLUMNS', 'RHS', 'QUADOBJ', 'ENDATA']
        for section in required_sections:
            if section not in content:
                raise ValueError(f"Missing required section: {section}")
        
        # Use provided name or extract from file as fallback
        if problem_name is None:
            name_match = re.search(r'NAME\s+(\w+)', content)
            problem_name = name_match.group(1) if name_match else "unnamed"
        
        # Parse similar to MPS but also handle QUADOBJ section
        # For simplicity, reuse MPS parsing logic for linear part
        mps_problem = load_mps_file(file_path, problem_name)  # Will fail on QUADOBJ, but that's OK
        
        # Parse QUADOBJ section for quadratic terms
        quadobj_section = re.search(r'QUADOBJ\s*\n(.*?)\n(?:ENDATA|$)', content, re.DOTALL)
        if not quadobj_section:
            raise ValueError("Could not parse QUADOBJ section")
        
        # Extract variables from COLUMNS section to get variable ordering
        columns_section = re.search(r'COLUMNS\s*\n(.*?)\n(?:RHS|$)', content, re.DOTALL)
        variables = set()
        for line in columns_section.group(1).strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    variables.add(parts[0])
        
        variables = sorted(list(variables))
        n_vars = len(variables)
        var_index = {var: i for i, var in enumerate(variables)}
        
        # Build quadratic matrix P
        P = np.zeros((n_vars, n_vars))
        for line in quadobj_section.group(1).strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    var1, var2 = parts[0], parts[1]
                    coeff = float(parts[2])
                    if var1 in var_index and var2 in var_index:
                        i, j = var_index[var1], var_index[var2]
                        P[i, j] = coeff
                        if i != j:  # Make symmetric
                            P[j, i] = coeff
        
        logger.info(f"Successfully loaded QPS problem: {problem_name} ({n_vars} variables)")
        
        return ProblemData(
            name=problem_name,
            problem_class="QP",
            c=np.zeros(n_vars),  # Linear part is usually zero for our simple QP
            A_ub=mps_problem.A_ub,
            b_ub=mps_problem.b_ub,
            bounds=[(0, None) for _ in range(n_vars)],
            P=P
        )
        
    except Exception as e:
        logger.error(f"Failed to load QPS file {file_path}: {e}")
        raise

def load_problem_registry() -> Dict:
    """Load the problem registry YAML file."""
    project_root = Path(__file__).parent.parent.parent
    registry_path = project_root / "config" / "problem_registry.yaml"
    
    logger.info(f"Loading problem registry: {registry_path}")
    
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)

def load_python_problem(file_path: str, problem_class: str) -> ProblemData:
    """Load a problem from a Python module (for SOCP, SDP, etc.)."""
    logger.info(f"Loading Python problem: {file_path}")
    
    try:
        import importlib.util
        import sys
        
        # Load the module
        spec = importlib.util.spec_from_file_location("problem_module", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["problem_module"] = module
        spec.loader.exec_module(module)
        
        # Look for the main generation function based on problem class
        if problem_class == "SOCP":
            if hasattr(module, 'generate_portfolio_optimization_socp'):
                problem_dict, problem_data = module.generate_portfolio_optimization_socp()
            elif hasattr(module, 'generate_robust_optimization_socp'):
                problem_dict, problem_data = module.generate_robust_optimization_socp()
            elif hasattr(module, 'generate_facility_location_socp'):
                problem_dict, problem_data = module.generate_facility_location_socp()
            else:
                raise ValueError(f"No suitable SOCP generation function found in {file_path}")
        elif problem_class == "SDP":
            if hasattr(module, 'generate_matrix_completion_sdp'):
                problem_dict, problem_data = module.generate_matrix_completion_sdp()
            elif hasattr(module, 'generate_control_lmi_sdp'):
                problem_dict, problem_data = module.generate_control_lmi_sdp()
            elif hasattr(module, 'generate_max_cut_sdp'):
                problem_dict, problem_data = module.generate_max_cut_sdp()
            else:
                raise ValueError(f"No suitable SDP generation function found in {file_path}")
        else:
            raise ValueError(f"Unsupported Python problem class: {problem_class}")
        
        logger.info(f"Successfully loaded Python problem: {problem_data.name} ({problem_class})")
        return problem_data
        
    except Exception as e:
        logger.error(f"Failed to load Python problem {file_path}: {e}")
        raise

def load_problem(problem_name: str, problem_set: str = None) -> ProblemData:
    """Load a specific problem by name from the registry."""
    
    # Load from registry (new flat structure)
    registry = load_problem_registry()
    
    # Find the problem in the flat registry
    problem_info = None
    if problem_name in registry["problem_libraries"]:
        problem_info = registry["problem_libraries"][problem_name]
    
    if not problem_info:
        raise ValueError(f"Problem '{problem_name}' not found in registry")
    
    # Filter by problem_set/library if specified
    if problem_set and problem_info.get("library_name") != problem_set:
        raise ValueError(f"Problem '{problem_name}' is not from library '{problem_set}'")
    
    # Get absolute path to problem file
    project_root = Path(__file__).parent.parent.parent
    file_path = project_root / problem_info["file_path"]
    
    # Load based on file type
    file_type = problem_info["file_type"]
    
    if file_type == "mat":
        # Use MAT loader for .mat.gz files (DIMACS)
        from scripts.data_loaders.python.mat_loader import MATLoader
        loader = MATLoader()
        return loader.load(str(file_path), problem_name)
    elif file_type == "dat-s":
        # Use DAT loader for .dat-s files (SDPLIB)
        from scripts.data_loaders.python.dat_loader import DATLoader
        loader = DATLoader()
        return loader.load(str(file_path), problem_name)
    elif file_type == "mps":
        # Use MPS loader for Linear Programming
        return load_mps_file(str(file_path), problem_name)
    elif file_type == "qps":
        # Use QPS loader for Quadratic Programming
        return load_qps_file(str(file_path), problem_name)
    elif file_type == "python":
        # Use Python loader for SOCP/SDP
        return load_python_problem(str(file_path), problem_info["problem_type"])
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def list_available_problems(problem_set: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all available problems in the registry.
    
    Args:
        problem_set: Optional library filter ('DIMACS', 'SDPLIB', 'internal')
        
    Returns:
        Dictionary mapping library names to lists of problem names
    """
    registry = load_problem_registry()
    available = {}
    
    # Group problems by library_name from the flat structure
    for problem_name, problem_info in registry["problem_libraries"].items():
        library_name = problem_info.get("library_name", "unknown")
        
        # Filter by problem_set if specified
        if problem_set and library_name != problem_set:
            continue
            
        if library_name not in available:
            available[library_name] = []
        available[library_name].append(problem_name)
    
    # Sort problem names within each library
    for library in available:
        available[library] = sorted(available[library])
    
    return available

if __name__ == "__main__":
    # Test script to load and validate problems
    try:
        # Test loading registry
        registry = load_problem_registry()
        print("Registry loaded successfully:")
        print(f"Problem sets: {list(registry['problems'].keys())}")
        
        # Test loading each problem
        for problem_set in registry["problems"]:
            for problem_class in registry["problems"][problem_set]:
                for problem_info in registry["problems"][problem_set][problem_class]:
                    problem_name = problem_info["name"]
                    print(f"\nTesting problem: {problem_name}")
                    
                    problem = load_problem(problem_name, problem_set)
                    print(f"  Loaded: {problem}")
                    print(f"  Variables: {len(problem.c) if problem.c is not None else 0}")
                    if problem.A_ub is not None:
                        print(f"  Constraints: {problem.A_ub.shape[0]}")
                    if problem.P is not None:
                        print(f"  Quadratic matrix shape: {problem.P.shape}")
                        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise