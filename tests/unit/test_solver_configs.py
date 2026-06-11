"""Regression tests for the centralized Python solver configuration.

scripts/solvers/python/solver_configs.py is the single source of truth,
shared by PythonSolverManager (subprocess side) and PythonProcessInterface
(main process side).
"""

import pytest

from scripts.solvers.python.solver_configs import PYTHON_SOLVER_CONFIGS


def test_configs_have_required_fields():
    for name, config in PYTHON_SOLVER_CONFIGS.items():
        assert "display_name" in config, f"{name} missing 'display_name'"
        assert "runner" in config, f"{name} missing 'runner'"
        assert isinstance(config.get("kwargs"), dict), f"{name} 'kwargs' must be a dict"


def test_cvxpy_solvers_specify_backend():
    for name, config in PYTHON_SOLVER_CONFIGS.items():
        if name.startswith("cvxpy_"):
            assert config["runner"] == "cvxpy", f"{name} must use the cvxpy runner"
            assert "backend" in config["kwargs"], f"{name} missing backend kwarg"


def test_process_interface_shares_the_config():
    from scripts.solvers.python.python_process_interface import PythonProcessInterface

    assert PythonProcessInterface.PYTHON_SOLVER_CONFIGS is PYTHON_SOLVER_CONFIGS


def test_manager_shares_the_config_and_resolves_all_runners():
    # python_solver_runner imports cvxpy at module level
    pytest.importorskip("cvxpy")
    from scripts.solvers.python.python_solver_runner import RUNNER_CLASSES, PythonSolverManager

    assert PythonSolverManager.PYTHON_SOLVER_CONFIGS is PYTHON_SOLVER_CONFIGS
    for name, config in PYTHON_SOLVER_CONFIGS.items():
        assert config["runner"] in RUNNER_CLASSES, f"{name} has unresolvable runner '{config['runner']}'"
