"""Unit tests for the SDPA .dat-s loader (no external problem data required)."""

import numpy as np
import pytest

from scripts.data_loaders.python.dat_loader import DATLoader

# Small SDP: m=2 constraints, 2 blocks (one 2x2 SDP block, one diagonal block of size 2)
SDPA_STANDARD = """\
"This is a comment line
* This is another comment line
2
2
2 -2
10.0 20.0
0 1 1 1 1.0
0 1 1 2 0.5
1 1 1 1 1.0
1 2 1 1 1.0
2 1 2 2 1.0
2 2 2 2 1.0
"""

# Same problem but with curly-brace objective format (as used by e.g. gpp100.dat-s)
SDPA_BRACES = """\
2
2
(2, -2)
{+10.0,+20.0}
0 1 1 1 1.0
1 1 1 1 1.0
2 2 2 2 1.0
"""


@pytest.fixture
def loader():
    return DATLoader()


def write_dat(tmp_path, content, name="test.dat-s"):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


class TestParseSdpaFile:

    def test_parses_header(self, loader, tmp_path):
        parsed = loader.parse_sdpa_file(write_dat(tmp_path, SDPA_STANDARD))
        assert parsed['m'] == 2
        assert parsed['nblocks'] == 2
        assert parsed['block_sizes'] == [2, -2]

    def test_skips_comment_lines(self, loader, tmp_path):
        # SDPA_STANDARD starts with " and * comment lines; header must still parse
        parsed = loader.parse_sdpa_file(write_dat(tmp_path, SDPA_STANDARD))
        assert parsed['m'] == 2

    def test_objective_vector_space_separated(self, loader, tmp_path):
        parsed = loader.parse_sdpa_file(write_dat(tmp_path, SDPA_STANDARD))
        np.testing.assert_array_equal(parsed['c'], np.array([10.0, 20.0]))

    def test_objective_vector_brace_format(self, loader, tmp_path):
        parsed = loader.parse_sdpa_file(write_dat(tmp_path, SDPA_BRACES))
        np.testing.assert_array_equal(parsed['c'], np.array([10.0, 20.0]))

    def test_off_diagonal_entries_are_symmetrized(self, loader, tmp_path):
        parsed = loader.parse_sdpa_file(write_dat(tmp_path, SDPA_STANDARD))
        # Entry "0 1 1 2 0.5" must be mirrored to (2, 1) in block 1 of F0
        f0_block1 = parsed['matrices'][0][0]
        entries = set(zip(f0_block1['i'], f0_block1['j']))
        assert (0, 1) in entries
        assert (1, 0) in entries

    def test_missing_file_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.parse_sdpa_file(str(tmp_path / "does_not_exist.dat-s"))

    def test_insufficient_data_raises(self, loader, tmp_path):
        with pytest.raises(ValueError, match="insufficient data lines"):
            loader.parse_sdpa_file(write_dat(tmp_path, "2\n2\n"))

    def test_block_size_mismatch_raises(self, loader, tmp_path):
        content = "2\n2\n2\n10.0 20.0\n"  # one block size given but nblocks=2
        with pytest.raises(ValueError, match="Block sizes count"):
            loader.parse_sdpa_file(write_dat(tmp_path, content))

    def test_objective_length_mismatch_raises(self, loader, tmp_path):
        content = "2\n2\n2 -2\n10.0\n"  # one coefficient given but m=2
        with pytest.raises(ValueError, match="Objective vector length"):
            loader.parse_sdpa_file(write_dat(tmp_path, content))


class TestLoad:

    def test_load_returns_problem_data(self, loader, tmp_path):
        problem = loader.load(write_dat(tmp_path, SDPA_STANDARD), problem_name="tiny_sdp")
        assert problem.name == "tiny_sdp"
        assert problem.problem_class == "SDP"
        # Total variables: 2x2 block (4 entries) + diagonal block of 2 (4 entries as 2x2)
        assert problem.A_eq.shape[0] == 2  # m constraints
        assert problem.b_eq.shape == (2, 1)
        assert problem.A_ub is None
        assert problem.b_ub is None

    def test_load_extracts_name_from_path(self, loader, tmp_path):
        problem = loader.load(write_dat(tmp_path, SDPA_STANDARD, name="myprob.dat-s"))
        assert problem.name == "myprob"

    def test_cone_structure(self, loader, tmp_path):
        problem = loader.load(write_dat(tmp_path, SDPA_STANDARD), problem_name="tiny_sdp")
        cone = problem.cone_structure
        # Negative block size -2 expands to two 1x1 SDP cones
        assert cone['sdp_cones'] == [2, 1, 1]
        assert cone['free_vars'] == 0
        assert cone['nonneg_vars'] == 0
        assert cone['soc_cones'] == []

    def test_b_is_negated_objective(self, loader, tmp_path):
        problem = loader.load(write_dat(tmp_path, SDPA_STANDARD), problem_name="tiny_sdp")
        np.testing.assert_array_equal(problem.b_eq, np.array([[-10.0], [-20.0]]))
