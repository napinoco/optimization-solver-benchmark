"""Integrity checks for the real problem registry (config/problem_registry.yaml)."""

from pathlib import Path

import pytest
import yaml

from scripts.data_loaders.python.problem_interface import ProblemInterface

PROJECT_ROOT = Path(__file__).parent.parent.parent
REGISTRY_PATH = PROJECT_ROOT / "config" / "problem_registry.yaml"

REQUIRED_KEYS = {"display_name", "file_path", "file_type", "library_name"}
KNOWN_LIBRARIES = {"DIMACS", "SDPLIB", "NETLIB"}


@pytest.fixture(scope="module")
def registry():
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def problems(registry):
    return registry["problem_libraries"]


def test_registry_loads_and_is_non_empty(problems):
    assert len(problems) > 0


def test_all_entries_have_required_keys(problems):
    for name, config in problems.items():
        missing = REQUIRED_KEYS - set(config.keys())
        assert not missing, f"Problem '{name}' missing keys: {missing}"


def test_all_file_types_are_supported(problems):
    supported = set(ProblemInterface.FORMAT_LOADERS.keys())
    for name, config in problems.items():
        assert config["file_type"] in supported, f"Problem '{name}' has unsupported file_type '{config['file_type']}'"


def test_all_library_names_are_known(problems):
    for name, config in problems.items():
        assert config["library_name"] in KNOWN_LIBRARIES, (
            f"Problem '{name}' has unknown library_name '{config['library_name']}'"
        )


def test_all_file_paths_are_under_problems_dir(problems):
    for name, config in problems.items():
        assert config["file_path"].startswith("problems/"), (
            f"Problem '{name}' file_path is outside problems/: {config['file_path']}"
        )


def test_known_objective_values_are_numeric(problems):
    for name, config in problems.items():
        if "known_objective_value" in config:
            value = config["known_objective_value"]
            assert isinstance(value, (int, float)), f"Problem '{name}' known_objective_value is not numeric: {value!r}"


def test_for_test_flags_are_boolean(problems):
    for name, config in problems.items():
        if "for_test_flag" in config:
            assert isinstance(config["for_test_flag"], bool), f"Problem '{name}' for_test_flag is not boolean"


def test_has_fast_test_problems(problems):
    test_problems = [n for n, c in problems.items() if c.get("for_test_flag")]
    assert len(test_problems) > 0, "Registry must define at least one for_test_flag problem"


def test_problem_files_exist(problems):
    # Skip when submodules are not checked out (all must be present to run this test)
    submodule_dirs = {
        "DIMACS": PROJECT_ROOT / "problems" / "DIMACS" / "data",
        "SDPLIB": PROJECT_ROOT / "problems" / "SDPLIB" / "data",
        "NETLIB": PROJECT_ROOT / "problems" / "NETLIB" / "mps_files",
    }
    if not all(d.exists() and any(d.iterdir()) for d in submodule_dirs.values()):
        pytest.skip("Problem library submodules not checked out")

    missing = [name for name, config in problems.items() if not (PROJECT_ROOT / config["file_path"]).exists()]
    assert not missing, f"Registered problem files not found on disk: {missing}"
