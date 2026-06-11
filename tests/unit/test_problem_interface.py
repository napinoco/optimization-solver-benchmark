"""Unit tests for ProblemInterface using a synthetic problem registry."""

import pytest
import yaml

from scripts.data_loaders.python.problem_interface import ProblemInterface

REGISTRY = {
    'problem_libraries': {
        'prob_sdp': {
            'display_name': "Prob SDP",
            'file_path': "problems/SDPLIB/data/prob_sdp.dat-s",
            'file_type': "dat-s",
            'library_name': "SDPLIB",
            'problem_type': "SDP",
            'for_test_flag': True,
            'known_objective_value': 1.0,
        },
        'prob_socp': {
            'display_name': "Prob SOCP",
            'file_path': "problems/DIMACS/data/prob_socp.mat.gz",
            'file_type': "mat",
            'library_name': "DIMACS",
            'problem_type': "SOCP",
            'for_test_flag': False,
        },
        'prob_unsupported': {
            'display_name': "Prob Unsupported",
            'file_path': "problems/OTHER/data/prob.qps",
            'file_type': "qps",
            'library_name': "OTHER",
            'problem_type': "QP",
        },
    }
}


@pytest.fixture
def interface(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(REGISTRY))
    return ProblemInterface(registry_path=str(registry_path))


class TestRegistryLoading:

    def test_loads_all_problems(self, interface):
        assert set(interface.get_available_problems()) == {
            'prob_sdp', 'prob_socp', 'prob_unsupported'
        }

    def test_missing_registry_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProblemInterface(registry_path=str(tmp_path / "missing.yaml"))

    def test_registry_without_libraries_key_uses_empty(self, tmp_path):
        registry_path = tmp_path / "empty.yaml"
        registry_path.write_text(yaml.safe_dump({'other_key': 1}))
        interface = ProblemInterface(registry_path=str(registry_path))
        assert interface.get_available_problems() == []


class TestProblemConfig:

    def test_get_problem_config(self, interface):
        config = interface.get_problem_config('prob_sdp')
        assert config['file_type'] == "dat-s"
        assert config['library_name'] == "SDPLIB"

    def test_unknown_problem_raises_value_error(self, interface):
        with pytest.raises(ValueError, match="not found in registry"):
            interface.get_problem_config('nonexistent')


class TestFiltering:

    def test_filter_by_library(self, interface):
        assert interface.get_available_problems(library_filter=["DIMACS"]) == ['prob_socp']

    def test_filter_test_only(self, interface):
        assert interface.get_available_problems(test_only=True) == ['prob_sdp']

    def test_get_problems_by_type(self, interface):
        assert interface.get_problems_by_type(["SDP"]) == ['prob_sdp']
        assert set(interface.get_problems_by_type(["SDP", "SOCP"])) == {'prob_sdp', 'prob_socp'}

    def test_get_problems_by_library(self, interface):
        assert interface.get_problems_by_library(["SDPLIB"]) == ['prob_sdp']


class TestCompatibility:

    def test_compatible_file_type(self, interface):
        assert interface.validate_problem_compatibility('prob_sdp', ["dat-s", "mat"])

    def test_incompatible_file_type(self, interface):
        assert not interface.validate_problem_compatibility('prob_sdp', ["mat"])

    def test_unknown_problem_is_incompatible(self, interface):
        assert not interface.validate_problem_compatibility('nonexistent', ["mat"])

    def test_load_unsupported_file_type_raises(self, interface):
        with pytest.raises(ValueError, match="Unsupported file type"):
            interface.load_problem('prob_unsupported')


class TestStatistics:

    def test_statistics_counts(self, interface):
        stats = interface.get_problem_statistics()
        assert stats['total_problems'] == 3
        assert stats['libraries'] == {'SDPLIB': 1, 'DIMACS': 1, 'OTHER': 1}
        assert stats['test_problems'] == 1
        assert stats['file_formats'] == {'dat-s': 1, 'mat': 1, 'qps': 1}
        assert set(stats['supported_formats']) == {'mat', 'dat-s'}
