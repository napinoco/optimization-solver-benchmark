"""
Python Solver Configurations (single source of truth).

This module holds the declarative configuration for all Python solvers,
shared by PythonSolverManager (python_solver_runner.py, subprocess side)
and PythonProcessInterface (python_process_interface.py, main process side).

IMPORTANT: This module must stay free of heavy imports (cvxpy, scipy, ...).
The main process imports it for solver name lookups, and pulling solver
libraries into the main process would defeat the purpose of subprocess
isolation. Solver classes are resolved from the "runner" key inside
python_solver_runner.py, which only runs in the subprocess.
"""

PYTHON_SOLVER_CONFIGS = {
    "scipy_linprog": {
        "display_name": "SciPy linprog",
        "runner": "scipy",
        "kwargs": {},
    },
    "cvxpy_clarabel": {
        "display_name": "CLARABEL (via CVXPY)",
        "runner": "cvxpy",
        "kwargs": {"backend": "CLARABEL"},
    },
    "cvxpy_scs": {
        "display_name": "SCS (via CVXPY)",
        "runner": "cvxpy",
        "kwargs": {"backend": "SCS"},
    },
    "cvxpy_ecos": {
        "display_name": "ECOS (via CVXPY)",
        "runner": "cvxpy",
        "kwargs": {"backend": "ECOS"},
    },
    "cvxpy_osqp": {
        "display_name": "OSQP (via CVXPY)",
        "runner": "cvxpy",
        "kwargs": {"backend": "OSQP"},
    },
    "cvxpy_cvxopt": {
        "display_name": "CVXOPT (via CVXPY)",
        "runner": "cvxpy",
        "kwargs": {"backend": "CVXOPT"},
    },
    "cvxpy_sdpa": {
        "display_name": "SDPA (via CVXPY)",
        "runner": "cvxpy",
        "kwargs": {"backend": "SDPA"},
    },
    "cvxpy_scip": {
        "display_name": "SCIP (via CVXPY)",
        "runner": "cvxpy",
        "kwargs": {"backend": "SCIP"},
    },
    "cvxpy_highs": {
        "display_name": "HiGHS (via CVXPY)",
        "runner": "cvxpy",
        "kwargs": {"backend": "HIGHS"},
    },
}
