"""Regression tests for solver configuration consistency.

The solver name list must stay consistent between the manager
(python_solver_runner.py, used inside the subprocess) and the process
interface (python_process_interface.py, used by the main process).
"""

import pytest

# python_solver_runner imports cvxpy at module level
pytest.importorskip("cvxpy")

from scripts.solvers.python.python_process_interface import PythonProcessInterface
from scripts.solvers.python.python_solver_runner import PythonSolverManager


def test_solver_names_match_between_manager_and_process_interface():
    manager_solvers = set(PythonSolverManager.PYTHON_SOLVER_CONFIGS.keys())
    process_solvers = set(PythonProcessInterface.PYTHON_SOLVER_CONFIGS.keys())
    assert manager_solvers == process_solvers


def test_manager_configs_have_required_fields():
    for name, config in PythonSolverManager.PYTHON_SOLVER_CONFIGS.items():
        assert "class" in config, f"{name} missing 'class'"
        assert "kwargs" in config, f"{name} missing 'kwargs'"
        assert isinstance(config["kwargs"], dict), f"{name} 'kwargs' must be a dict"


def test_cvxpy_solvers_specify_backend():
    for name, config in PythonSolverManager.PYTHON_SOLVER_CONFIGS.items():
        if name.startswith("cvxpy_"):
            assert "backend" in config["kwargs"], f"{name} missing backend kwarg"
