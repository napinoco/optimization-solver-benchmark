"""Unit tests for the SeDuMi .mat/.mat.gz loader using synthetic fixtures."""

import gzip
import shutil

import numpy as np
import pytest
import scipy.io

from scripts.data_loaders.python.mat_loader import MATLoader


@pytest.fixture
def loader():
    return MATLoader()


def make_sedumi_mat(tmp_path, K, name="test.mat", n_vars=3, use_At=False):
    """Write a minimal SeDuMi problem (A: 2 x n_vars) to a .mat file."""
    A = np.arange(2 * n_vars, dtype=float).reshape(2, n_vars) + 1.0
    b = np.array([1.0, 2.0])
    c = np.ones(n_vars)
    data = {"b": b, "c": c, "K": K}
    if use_At:
        data["At"] = A.T
    else:
        data["A"] = A
    path = tmp_path / name
    scipy.io.savemat(str(path), data)
    return str(path)


def gzip_file(path):
    gz_path = path + ".gz"
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return gz_path


class TestLoadSedumiMat:
    def test_load_uncompressed(self, loader, tmp_path):
        path = make_sedumi_mat(tmp_path, {"l": 3.0})
        mat_data = loader.load_sedumi_mat(path)
        assert "A" in mat_data
        assert "b" in mat_data
        assert "c" in mat_data
        assert "K" in mat_data

    def test_load_gzipped(self, loader, tmp_path):
        path = make_sedumi_mat(tmp_path, {"l": 3.0})
        gz_path = gzip_file(path)
        mat_data = loader.load_sedumi_mat(gz_path)
        assert "A" in mat_data

    def test_missing_file_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load_sedumi_mat(str(tmp_path / "nope.mat"))

    def test_missing_constraint_matrix_raises(self, loader, tmp_path):
        path = tmp_path / "bad.mat"
        scipy.io.savemat(str(path), {"b": np.ones(2), "c": np.ones(3), "K": {"l": 3.0}})
        with pytest.raises(ValueError, match="A or At"):
            loader.load_sedumi_mat(str(path))

    def test_missing_required_field_raises(self, loader, tmp_path):
        path = tmp_path / "bad.mat"
        scipy.io.savemat(str(path), {"A": np.ones((2, 3)), "b": np.ones(2), "c": np.ones(3)})
        with pytest.raises(ValueError, match="Missing required SeDuMi field: K"):
            loader.load_sedumi_mat(str(path))


class TestProblemClassDetection:
    def test_lp_from_linear_cone(self, loader, tmp_path):
        problem = loader.load(make_sedumi_mat(tmp_path, {"l": 3.0}), problem_name="lp")
        assert problem.problem_class == "LP"
        assert problem.cone_structure["nonneg_vars"] == 3

    def test_socp_from_q_cone(self, loader, tmp_path):
        problem = loader.load(make_sedumi_mat(tmp_path, {"q": np.array([3.0])}), problem_name="socp")
        assert problem.problem_class == "SOCP"
        assert problem.cone_structure["soc_cones"] == [3]

    def test_sdp_takes_priority(self, loader, tmp_path):
        K = {"l": 1.0, "q": np.array([2.0]), "s": np.array([2.0])}
        problem = loader.load(make_sedumi_mat(tmp_path, K, n_vars=7), problem_name="sdp")
        assert problem.problem_class == "SDP"
        assert problem.cone_structure["sdp_cones"] == [2]


class TestConvertToProblemData:
    def test_shapes_and_equality_constraints(self, loader, tmp_path):
        problem = loader.load(make_sedumi_mat(tmp_path, {"l": 3.0}), problem_name="lp")
        assert problem.A_eq.shape == (2, 3)
        assert problem.b_eq.shape == (2, 1)
        assert problem.c.shape == (3, 1)
        assert problem.A_ub is None
        assert problem.b_ub is None

    def test_at_variant_is_transposed(self, loader, tmp_path):
        path = make_sedumi_mat(tmp_path, {"l": 3.0}, use_At=True)
        problem = loader.load(path, problem_name="lp_at")
        assert problem.A_eq.shape == (2, 3)
        assert problem.metadata["matrix_variant"] == "At"

    def test_name_extracted_from_path(self, loader, tmp_path):
        path = make_sedumi_mat(tmp_path, {"l": 3.0}, name="someprob.mat")
        problem = loader.load(path)
        assert problem.name == "someprob"
