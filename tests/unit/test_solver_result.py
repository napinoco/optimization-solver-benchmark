"""Unit tests for the standardized SolverResult format."""

import pytest

from scripts.solvers.solver_interface import SolverResult


def make_result(**overrides):
    fields = dict(
        solve_time=1.5,
        status="OPTIMAL",
        primal_objective_value=42.0,
        dual_objective_value=41.9,
        duality_gap=0.1,
        primal_infeasibility=1e-9,
        dual_infeasibility=1e-9,
        iterations=10,
    )
    fields.update(overrides)
    return SolverResult(**fields)


class TestValidation:

    def test_valid_result(self):
        result = make_result()
        assert result.status == "OPTIMAL"

    def test_none_fields_allowed(self):
        result = make_result(
            solve_time=None, primal_objective_value=None, dual_objective_value=None,
            duality_gap=None, primal_infeasibility=None, dual_infeasibility=None,
            iterations=None,
        )
        assert result.solve_time is None

    def test_negative_solve_time_rejected(self):
        with pytest.raises(ValueError, match="solve_time must be non-negative"):
            make_result(solve_time=-1.0)

    def test_non_numeric_solve_time_rejected(self):
        with pytest.raises(ValueError, match="solve_time must be numeric"):
            make_result(solve_time="fast")

    def test_empty_status_rejected(self):
        with pytest.raises(ValueError, match="status cannot be empty"):
            make_result(status="   ")

    def test_non_string_status_rejected(self):
        with pytest.raises(ValueError, match="status must be string"):
            make_result(status=1)

    def test_non_numeric_objective_rejected(self):
        with pytest.raises(ValueError, match="primal_objective_value must be numeric"):
            make_result(primal_objective_value="big")

    def test_negative_iterations_rejected(self):
        with pytest.raises(ValueError, match="iterations must be non-negative"):
            make_result(iterations=-5)

    def test_non_int_iterations_rejected(self):
        with pytest.raises(ValueError, match="iterations must be int"):
            make_result(iterations=2.5)


class TestToDict:

    def test_contains_all_fields(self):
        result = make_result(solver_name="test", solver_version="1.0")
        d = result.to_dict()
        expected_keys = {
            'solve_time', 'status', 'primal_objective_value', 'dual_objective_value',
            'duality_gap', 'primal_infeasibility', 'dual_infeasibility', 'iterations',
            'solver_name', 'solver_version', 'additional_info',
        }
        assert set(d.keys()) == expected_keys
        assert d['solve_time'] == 1.5
        assert d['solver_name'] == "test"


class TestFactoryMethods:

    def test_error_result(self):
        result = SolverResult.create_error_result("boom", solve_time=0.5, solver_name="s")
        assert result.status == "ERROR"
        assert result.solve_time == 0.5
        assert result.additional_info["error_message"] == "boom"
        assert result.primal_objective_value is None

    def test_timeout_result(self):
        result = SolverResult.create_timeout_result(300.0, solver_name="s")
        assert result.status == "TIMEOUT"
        assert result.solve_time == 300.0
        assert result.additional_info["timeout_duration"] == 300.0

    def test_subprocess_error_result(self):
        result = SolverResult.create_subprocess_error_result(returncode=1, error_message="crash")
        assert result.status == "SUBPROCESS_ERROR"
        assert result.additional_info["returncode"] == 1
        assert result.additional_info["error_message"] == "crash"

    def test_sigkill_result_with_memory_limit(self):
        result = SolverResult.create_sigkill_result(memory_limit_gb=8.0, solve_time=2.0)
        assert result.status == "SIGKILL"
        assert result.additional_info["memory_limit_gb"] == 8.0
        assert result.additional_info["error_type"] == "SIGKILL"

    def test_sigkill_result_without_memory_limit(self):
        result = SolverResult.create_sigkill_result()
        assert result.status == "SIGKILL"
        assert "memory_limit_gb" not in result.additional_info

    def test_unsupported_result(self):
        result = SolverResult.create_unsupported_result("SDP", solver_name="scipy_linprog")
        assert result.status == "UNSUPPORTED"
        assert result.solve_time is None
        assert result.additional_info["problem_type"] == "SDP"
