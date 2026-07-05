"""
MPS File Loader for Linear Programming problems.

This module provides functionality to load and parse optimization problems
stored in MPS (Mathematical Programming System) format, a standard format
for linear programming problems commonly used by Netlib and other LP libraries.

MPS Format Structure:
- NAME: Problem name
- ROWS: Row (constraint) definitions with types:
  - N: Objective function (free row)
  - E: Equality constraint (=)
  - L: Less-than-or-equal constraint (<=)
  - G: Greater-than-or-equal constraint (>=)
- COLUMNS: Variable coefficients in constraints
- RHS: Right-hand side values
- BOUNDS: Variable bounds (optional)
- RANGES: Range constraints (optional)
- ENDATA: End of data

The problem format is: min c'x subject to Ax {<=,=,>=} b, l <= x <= u
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.sparse import lil_matrix

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loaders.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("mps_loader")


class MPSLoader:
    """Loader for optimization problems in MPS format."""

    def __init__(self):
        """Initialize the MPS loader."""
        pass

    def load(self, file_path: str, problem_name: str = None) -> ProblemData:
        """
        Load problem from .mps file.

        Args:
            file_path: Path to the .mps file
            problem_name: Optional problem name (if not provided, extracted from file)

        Returns:
            ProblemData object
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading MPS file: {file_path}")

        # Parse MPS file
        mps_data = self._parse_mps_file(file_path)

        # Use provided name, MPS NAME field, or filename as fallback
        if problem_name is None:
            problem_name = mps_data.get("name", Path(file_path).stem)

        # Convert to ProblemData
        problem_data = self._convert_to_problem_data(mps_data, problem_name)

        logger.info(f"Successfully loaded MPS problem: {problem_data}")
        return problem_data

    def _parse_mps_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse MPS format file.

        Args:
            file_path: Path to MPS file

        Returns:
            Dictionary containing parsed MPS data
        """
        mps_data = {
            "name": "",
            "rows": {},  # row_name -> type (N, E, L, G)
            "row_order": [],  # ordered list of row names
            "columns": {},  # col_name -> {row_name: coef}
            "col_order": [],  # ordered list of column names
            "rhs": {},  # row_name -> value
            "bounds": {},  # col_name -> {'lo': val, 'up': val, 'type': type}
            "ranges": {},  # row_name -> value
            "obj_name": None,  # name of objective row
        }

        with open(file_path, "r") as f:
            lines = f.readlines()

        current_section = None

        for line in lines:
            # Skip empty lines and comments
            line = line.rstrip("\n\r")
            if not line or line.startswith("*"):
                continue

            # Check for section headers
            upper_line = line.upper().strip()
            if upper_line.startswith("NAME"):
                mps_data["name"] = line.split()[1] if len(line.split()) > 1 else ""
                current_section = "NAME"
                continue
            elif upper_line == "ROWS":
                current_section = "ROWS"
                continue
            elif upper_line == "COLUMNS":
                current_section = "COLUMNS"
                continue
            elif upper_line == "RHS":
                current_section = "RHS"
                continue
            elif upper_line == "BOUNDS":
                current_section = "BOUNDS"
                continue
            elif upper_line == "RANGES":
                current_section = "RANGES"
                continue
            elif upper_line == "ENDATA":
                break

            # Parse section content
            if current_section == "ROWS":
                self._parse_rows_line(line, mps_data)
            elif current_section == "COLUMNS":
                self._parse_columns_line(line, mps_data)
            elif current_section == "RHS":
                self._parse_rhs_line(line, mps_data)
            elif current_section == "BOUNDS":
                self._parse_bounds_line(line, mps_data)
            elif current_section == "RANGES":
                self._parse_ranges_line(line, mps_data)

        return mps_data

    def _parse_rows_line(self, line: str, mps_data: Dict[str, Any]):
        """Parse a line from ROWS section."""
        parts = line.split()
        if len(parts) >= 2:
            row_type = parts[0].upper()
            row_name = parts[1]
            mps_data["rows"][row_name] = row_type
            mps_data["row_order"].append(row_name)
            if row_type == "N" and mps_data["obj_name"] is None:
                mps_data["obj_name"] = row_name

    def _parse_columns_line(self, line: str, mps_data: Dict[str, Any]):
        """Parse a line from COLUMNS section."""
        parts = line.split()
        if len(parts) < 3:
            return

        col_name = parts[0]

        # Initialize column if not seen before
        if col_name not in mps_data["columns"]:
            mps_data["columns"][col_name] = {}
            mps_data["col_order"].append(col_name)

        # Parse pairs of (row_name, coefficient)
        i = 1
        while i < len(parts) - 1:
            row_name = parts[i]
            try:
                coef = float(parts[i + 1])
                mps_data["columns"][col_name][row_name] = coef
                i += 2
            except (ValueError, IndexError):
                break

    def _parse_rhs_line(self, line: str, mps_data: Dict[str, Any]):
        """Parse a line from RHS section."""
        parts = line.split()
        if len(parts) < 3:
            return

        # First field is RHS name (usually ignored), then pairs of (row, value)
        i = 1
        while i < len(parts) - 1:
            row_name = parts[i]
            try:
                value = float(parts[i + 1])
                mps_data["rhs"][row_name] = value
                i += 2
            except (ValueError, IndexError):
                break

    def _parse_bounds_line(self, line: str, mps_data: Dict[str, Any]):
        """Parse a line from BOUNDS section."""
        parts = line.split()
        if len(parts) < 3:
            return

        bound_type = parts[0].upper()
        # parts[1] is bound name (usually ignored)
        col_name = parts[2]

        if col_name not in mps_data["bounds"]:
            mps_data["bounds"][col_name] = {"lo": 0.0, "up": np.inf, "type": "default"}

        if bound_type == "LO":  # Lower bound
            mps_data["bounds"][col_name]["lo"] = float(parts[3])
        elif bound_type == "UP":  # Upper bound
            mps_data["bounds"][col_name]["up"] = float(parts[3])
        elif bound_type == "FX":  # Fixed value
            val = float(parts[3])
            mps_data["bounds"][col_name]["lo"] = val
            mps_data["bounds"][col_name]["up"] = val
            mps_data["bounds"][col_name]["type"] = "fixed"
        elif bound_type == "FR":  # Free variable
            mps_data["bounds"][col_name]["lo"] = -np.inf
            mps_data["bounds"][col_name]["up"] = np.inf
            mps_data["bounds"][col_name]["type"] = "free"
        elif bound_type == "MI":  # Minus infinity (lower bound)
            mps_data["bounds"][col_name]["lo"] = -np.inf
        elif bound_type == "PL":  # Plus infinity (upper bound)
            mps_data["bounds"][col_name]["up"] = np.inf
        elif bound_type == "BV":  # Binary variable
            mps_data["bounds"][col_name]["lo"] = 0.0
            mps_data["bounds"][col_name]["up"] = 1.0
            mps_data["bounds"][col_name]["type"] = "binary"

    def _parse_ranges_line(self, line: str, mps_data: Dict[str, Any]):
        """Parse a line from RANGES section."""
        parts = line.split()
        if len(parts) < 3:
            return

        # First field is range name (usually ignored), then pairs of (row, value)
        i = 1
        while i < len(parts) - 1:
            row_name = parts[i]
            try:
                value = float(parts[i + 1])
                mps_data["ranges"][row_name] = value
                i += 2
            except (ValueError, IndexError):
                break

    def _convert_to_problem_data(self, mps_data: Dict[str, Any], problem_name: str) -> ProblemData:
        """
        Convert parsed MPS data to unified SeDuMi-style ProblemData format.

        This converts LP problems to SeDuMi standard form:
            min  c'x
            s.t. A_eq @ x = b_eq
                 x in K (cone constraints)

        All inequality constraints are converted to equality constraints with
        slack variables, and variable bounds are handled through the cone structure.

        Args:
            mps_data: Parsed MPS data
            problem_name: Name for the problem

        Returns:
            ProblemData object in SeDuMi format
        """
        n_orig_cols = len(mps_data["col_order"])

        # Build column index mapping
        col_idx = {name: i for i, name in enumerate(mps_data["col_order"])}

        # Separate rows by type
        obj_rows = []
        eq_rows = []
        le_rows = []
        ge_rows = []

        for row_name in mps_data["row_order"]:
            row_type = mps_data["rows"][row_name]
            if row_type == "N":
                obj_rows.append(row_name)
            elif row_type == "E":
                eq_rows.append(row_name)
            elif row_type == "L":
                le_rows.append(row_name)
            elif row_type == "G":
                ge_rows.append(row_name)

        # Get variable bounds
        var_bounds = []
        for col_name in mps_data["col_order"]:
            if col_name in mps_data["bounds"]:
                bound_info = mps_data["bounds"][col_name]
                var_bounds.append((bound_info["lo"], bound_info["up"]))
            else:
                # Default: x >= 0 (standard LP convention)
                var_bounds.append((0.0, np.inf))

        # Count additional variables and constraints needed
        n_le = len(le_rows)
        n_ge = len(ge_rows)
        n_ineq = n_le + n_ge  # Number of slack variables needed

        # Count upper bound constraints (finite upper bounds)
        n_upper_bounds = sum(1 for lo, up in var_bounds if up < np.inf)

        # Total variables: original + slack for inequalities + slack for upper bounds
        n_slack_ineq = n_ineq
        n_slack_upper = n_upper_bounds
        n_total_vars = n_orig_cols + n_slack_ineq + n_slack_upper

        # Total constraints: equality + inequality (with slack) + upper bound constraints
        n_orig_eq = len(eq_rows)
        n_total_constraints = n_orig_eq + n_ineq + n_upper_bounds

        # Build objective vector c (extended with zeros for slack variables)
        c = np.zeros(n_total_vars)
        obj_name = mps_data["obj_name"]
        if obj_name:
            for col_name, coeffs in mps_data["columns"].items():
                if obj_name in coeffs:
                    c[col_idx[col_name]] = coeffs[obj_name]

        # Build unified constraint matrix A_eq and RHS b_eq
        A_eq = lil_matrix((n_total_constraints, n_total_vars))
        b_eq = np.zeros(n_total_constraints)

        row_idx = 0
        slack_idx = n_orig_cols  # Index for next slack variable

        # 1. Original equality constraints (E rows)
        for row_name in eq_rows:
            for col_name, coeffs in mps_data["columns"].items():
                if row_name in coeffs:
                    A_eq[row_idx, col_idx[col_name]] = coeffs[row_name]
            b_eq[row_idx] = mps_data["rhs"].get(row_name, 0.0)
            row_idx += 1

        # 2. Less-than-or-equal constraints (L rows): Ax + s = b, s >= 0
        for row_name in le_rows:
            for col_name, coeffs in mps_data["columns"].items():
                if row_name in coeffs:
                    A_eq[row_idx, col_idx[col_name]] = coeffs[row_name]
            # Add slack variable
            A_eq[row_idx, slack_idx] = 1.0
            b_eq[row_idx] = mps_data["rhs"].get(row_name, 0.0)
            row_idx += 1
            slack_idx += 1

        # 3. Greater-than-or-equal constraints (G rows): Ax - s = b, s >= 0
        for row_name in ge_rows:
            for col_name, coeffs in mps_data["columns"].items():
                if row_name in coeffs:
                    A_eq[row_idx, col_idx[col_name]] = coeffs[row_name]
            # Subtract slack variable
            A_eq[row_idx, slack_idx] = -1.0
            b_eq[row_idx] = mps_data["rhs"].get(row_name, 0.0)
            row_idx += 1
            slack_idx += 1

        # 4. Upper bound constraints: x + s_up = u, s_up >= 0
        for var_idx, (_lo, up) in enumerate(var_bounds):
            if up < np.inf:
                A_eq[row_idx, var_idx] = 1.0
                A_eq[row_idx, slack_idx] = 1.0
                b_eq[row_idx] = up
                row_idx += 1
                slack_idx += 1

        A_eq = A_eq.tocsr()

        # Handle lower bounds by shifting variables
        # For variables with lo != 0: x_new = x - lo, so A_eq stays same, b_eq adjusts
        c_shift = 0.0
        for var_idx, (lo, _up) in enumerate(var_bounds):
            if lo != 0 and lo > -np.inf:
                # x = x_new + lo, so c'x = c'x_new + c*lo
                c_shift += c[var_idx] * lo
                # Ax = A(x_new + lo) = Ax_new + A*lo, so b = b - A*lo
                b_eq -= A_eq[:, var_idx].toarray().flatten() * lo

        A_eq = A_eq.toarray()

        # Count free variables (variables with lo = -inf)
        n_free_vars = 0
        n_nonneg_vars = 0
        for lo, _up in var_bounds:
            if lo == -np.inf:
                n_free_vars += 1
            else:
                n_nonneg_vars += 1

        # Add slack variables to nonneg count
        n_nonneg_vars += n_slack_ineq + n_slack_upper

        # Create cone structure for SeDuMi format
        cone_structure = {"free_vars": n_free_vars, "nonneg_vars": n_nonneg_vars, "soc_cones": [], "sdp_cones": []}

        # Create metadata
        metadata = {
            "source": "MPS file",
            "format": "MPS (converted to SeDuMi)",
            "original_name": mps_data["name"],
            "objective_row": obj_name,
            "objective_shift": c_shift,
            "constraint_counts": {
                "original_equality": n_orig_eq,
                "original_less_equal": n_le,
                "original_greater_equal": n_ge,
                "upper_bound_constraints": n_upper_bounds,
                "total": n_total_constraints,
            },
            "original_dimensions": {"variables": n_orig_cols, "constraints": n_orig_eq + n_le + n_ge},
            "sedumi_dimensions": {"variables": n_total_vars, "constraints": n_total_constraints},
        }

        logger.info(
            f"Converted {problem_name}: LP problem "
            f"({n_orig_cols} original vars -> {n_total_vars} total vars, "
            f"{n_total_constraints} constraints)"
        )

        # Reshape for consistency with other loaders
        c = c.reshape(-1, 1)
        b_eq = b_eq.reshape(-1, 1)

        return ProblemData(
            name=problem_name,
            problem_class="LP",
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=None,
            b_ub=None,
            bounds=None,  # Bounds handled through SeDuMi cone structure
            cone_structure=cone_structure,
            metadata=metadata,
        )


if __name__ == "__main__":
    # Test script
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mps_loader.py <path_to_mps_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        loader = MPSLoader()
        problem = loader.load(file_path)
        print(f"Loaded problem: {problem}")
        print(f"  Variables: {problem._num_variables}")
        print(f"  Constraints: {problem._num_constraints}")
        print(f"  Bounds: {len(problem.bounds) if problem.bounds else 0} variables with bounds")

    except Exception as e:
        print(f"Error loading problem: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
