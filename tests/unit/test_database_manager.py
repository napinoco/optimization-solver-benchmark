"""Unit tests for DatabaseManager using temporary SQLite databases."""

import pytest

from scripts.database.database_manager import DatabaseManager

ENV_INFO = {"os": "linux", "python": "3.12", "cpu": "test-cpu"}


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=str(tmp_path / "test.db"))


def store_sample(db, **overrides):
    fields = dict(
        solver_name="cvxpy_clarabel",
        solver_version="0.11.0",
        problem_library="DIMACS",
        problem_name="nb",
        problem_type="SDP",
        environment_info=ENV_INFO,
        commit_hash="abc123",
        solve_time=1.23,
        status="OPTIMAL",
        primal_objective_value=-0.05,
        iterations=15,
    )
    fields.update(overrides)
    db.store_result(**fields)


class TestSchema:

    def test_schema_created_independent_of_cwd(self, tmp_path, monkeypatch):
        # ensure_schema must locate schema.sql relative to the module, not the CWD
        monkeypatch.chdir(tmp_path)
        db = DatabaseManager(db_path=str(tmp_path / "cwd_test.db"))
        assert db.get_latest_results() == []

    def test_reopening_existing_database_is_safe(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        DatabaseManager(db_path=db_path)
        DatabaseManager(db_path=db_path)  # schema uses IF NOT EXISTS


class TestStoreAndRetrieve:

    def test_round_trip(self, db):
        store_sample(db)
        results = db.get_latest_results()
        assert len(results) == 1
        row = results[0]
        assert row['solver_name'] == "cvxpy_clarabel"
        assert row['problem_name'] == "nb"
        assert row['solve_time'] == 1.23
        assert row['status'] == "OPTIMAL"
        assert row['iterations'] == 15

    def test_environment_info_json_round_trip(self, db):
        store_sample(db)
        results = db.get_latest_results()
        assert results[0]['environment_info'] == ENV_INFO

    def test_optional_fields_default_to_none(self, db):
        store_sample(db, solve_time=None, status=None, primal_objective_value=None,
                     iterations=None)
        row = db.get_latest_results()[0]
        assert row['solve_time'] is None
        assert row['duality_gap'] is None

    def test_duplicate_insert_is_ignored_without_raising(self, db):
        # Same UNIQUE key (solver, version, library, problem, commit, timestamp)
        # within the same second: second insert must be swallowed with a warning
        store_sample(db)
        store_sample(db)
        with db.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        assert count == 1


class TestLatestResults:

    def test_returns_only_latest_per_solver_problem(self, db):
        # Insert two rows with explicit distinct timestamps to test the
        # latest-per-(solver, problem) selection query deterministically
        insert_sql = """
        INSERT INTO results (
            solver_name, solver_version, problem_library, problem_name, problem_type,
            environment_info, commit_hash, timestamp, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with db.get_connection() as conn:
            conn.execute(insert_sql, ("s1", "1.0", "DIMACS", "p1", "SDP", "{}",
                                      "old", "2026-01-01 00:00:00", "ERROR"))
            conn.execute(insert_sql, ("s1", "1.0", "DIMACS", "p1", "SDP", "{}",
                                      "new", "2026-01-02 00:00:00", "OPTIMAL"))
            conn.commit()

        results = db.get_latest_results()
        assert len(results) == 1
        assert results[0]['commit_hash'] == "new"
        assert results[0]['status'] == "OPTIMAL"

    def test_filter_by_commit_and_environment(self, db):
        store_sample(db, commit_hash="aaa")
        store_sample(db, commit_hash="bbb", problem_name="other")
        results = db.get_latest_results(commit_hash="aaa", environment_info=ENV_INFO)
        assert len(results) == 1
        assert results[0]['commit_hash'] == "aaa"

    def test_empty_database_returns_empty_list(self, db):
        assert db.get_latest_results() == []
