from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from oceangla.config import config
from oceangla.gpu import (
    get_xp,
    is_cupy_installed,
    is_gpu_available,
    should_use_gpu,
    to_cpu,
    to_gpu,
)
from oceangla.model import (
    FirstLevelContrast,
    ModelSpec,
    OLSModel,
    RMAnovaModel,
    TwoWayAnovaModel,
)
from oceangla.parser import parse_args


# ---------------------------------------------------------------------------
# Mock CuPy implementation to simulate GPU acceleration on non-CUDA machines
# ---------------------------------------------------------------------------

class MockGPUArray(np.ndarray):
    """Subclass of np.ndarray that mimics a CuPy device array."""

    def get(self) -> np.ndarray:
        return np.asarray(self)


class MockCuPy:
    """Mock CuPy module matching CuPy API used across oceangla models."""

    ndarray = MockGPUArray

    @staticmethod
    def asarray(a, *args, **kwargs):
        return np.asarray(a, *args, **kwargs).view(MockGPUArray)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)

    newaxis = np.newaxis
    sum = staticmethod(np.sum)
    mean = staticmethod(np.mean)
    repeat = staticmethod(np.repeat)
    diag = staticmethod(np.diag)
    diagonal = staticmethod(np.diagonal)
    sqrt = staticmethod(np.sqrt)
    clip = staticmethod(np.clip)
    maximum = staticmethod(np.maximum)
    where = staticmethod(np.where)
    nan_to_num = staticmethod(np.nan_to_num)
    arange = staticmethod(np.arange)
    einsum = staticmethod(np.einsum)

    class linalg:
        @staticmethod
        def lstsq(a, b, rcond=None):
            res = np.linalg.lstsq(a, b, rcond=rcond)
            return (res[0].view(MockGPUArray), res[1], res[2], res[3])

        @staticmethod
        def inv(a):
            return np.linalg.inv(a).view(MockGPUArray)

    class cuda:
        @staticmethod
        def is_available():
            return True

        class runtime:
            @staticmethod
            def getDeviceCount():
                return 1


@pytest.fixture(autouse=True)
def reset_config_state():
    """Reset GPU-related config options before and after every test."""
    old_gpu = config.use_gpu
    yield
    config.use_gpu = old_gpu


# ---------------------------------------------------------------------------
# Unit tests for GPU detection and dispatching
# ---------------------------------------------------------------------------

def test_gpu_defaults_on_non_cuda():
    """Verify default CPU fallbacks when running without CUDA hardware."""
    # Under test environment (CPU only):
    assert not is_gpu_available()
    assert not should_use_gpu(False)
    assert not should_use_gpu(None)

    # Calling should_use_gpu(True) must raise RuntimeError if no GPU
    with pytest.raises(RuntimeError, match="GPU execution was requested"):
        should_use_gpu(True)

    xp = get_xp()
    assert xp is np

    arr = np.array([1.0, 2.0, 3.0])
    gpu_arr = to_gpu(arr)
    assert np.array_equal(gpu_arr, arr)

    cpu_arr = to_cpu(arr)
    assert isinstance(cpu_arr, np.ndarray)
    assert np.array_equal(cpu_arr, arr)
    assert to_cpu(None) is None
    assert to_gpu(None) is None


def test_should_use_gpu_config_override():
    """Verify should_use_gpu respects config.use_gpu."""
    config.use_gpu = False
    assert not should_use_gpu()

    config.use_gpu = True
    with pytest.raises(RuntimeError, match="GPU execution was requested"):
        should_use_gpu()


def test_mock_cupy_detection_and_conversions(monkeypatch):
    """Verify GPU dispatch functions when CuPy and GPU are available (mocked)."""
    import oceangla.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "cp", MockCuPy)
    monkeypatch.setattr(gpu_mod, "_CUPY_INSTALLED", True)

    assert is_cupy_installed()
    assert is_gpu_available()
    assert should_use_gpu(True)
    assert should_use_gpu(None)
    assert not should_use_gpu(False)

    xp = get_xp(True)
    assert xp is MockCuPy

    arr = np.array([1.0, 2.0, 3.0])
    gpu_arr = to_gpu(arr)
    assert isinstance(gpu_arr, MockGPUArray)

    cpu_arr = to_cpu(gpu_arr)
    assert type(cpu_arr) is np.ndarray
    assert np.array_equal(cpu_arr, arr)


def test_cli_gpu_flags(tmp_path):
    """Test that --gpu and --no-gpu CLI flags correctly update config.use_gpu."""
    fladir = tmp_path / "fla"
    fladir.mkdir()
    outdir = tmp_path / "out"

    # --gpu flag
    parse_args(args=["-f", str(fladir), "-o", str(outdir), "--gpu"])
    assert config.use_gpu is True

    # --no-gpu flag
    parse_args(args=["-f", str(fladir), "-o", str(outdir), "--no-gpu"])
    assert config.use_gpu is False

    # Mutually exclusive error when both specified
    with pytest.raises(SystemExit):
        parse_args(args=["-f", str(fladir), "-o", str(outdir), "--gpu", "--no-gpu"])


# ---------------------------------------------------------------------------
# Numerical tests comparing CPU (NumPy) vs GPU (MockCuPy) calculations
# ---------------------------------------------------------------------------

def test_ols_cpu_vs_gpu_identical(monkeypatch, tmp_path):
    """Verify OLSModel produces identical numerical results on CPU and GPU."""
    shape = (5, 5, 5)
    n_subs = 6
    paths = []
    for sub_id in range(1, n_subs + 1):
        data = np.random.RandomState(sub_id).randn(*shape).astype(np.float32)
        data[2, 2, 2] += 2.0
        img = nib.Nifti1Image(data, np.eye(4))
        p = tmp_path / f"sub-{sub_id:02d}_contrast.nii.gz"
        nib.save(img, p)
        paths.append(str(p))

    firstlevel_df = pd.DataFrame({
        "subject": [f"sub-{i:02d}" for i in range(1, n_subs + 1)],
        "condition": ["condA"] * n_subs,
        "frame_no": [-1] * n_subs,
        "task": ["rest"] * n_subs,
        "space": ["MNI152NLin2009cAsym"] * n_subs,
        "session": ["ses-1"] * n_subs,
        "path": paths,
    })

    subject_vars_df = pd.DataFrame({
        "subject": [f"sub-{i:02d}" for i in range(1, n_subs + 1)],
        "age": [20, 24, 28, 32, 36, 40],
    })

    spec = ModelSpec(
        model_type="OLS",
        model_name="test_ols_gpu",
        firstlevel_contrasts=[FirstLevelContrast(name="condA", scalar=1)],
        subject_variables="age",
        session="ses-1",
        space="MNI152NLin2009cAsym",
        task="rest",
    )

    config.outdir_path = tmp_path

    # 1. CPU Run
    config.use_gpu = False
    model_cpu = OLSModel(
        model_spec=spec,
        firstlevel_contrasts_df=firstlevel_df,
        subject_variables_df=subject_vars_df,
    )
    res_cpu = model_cpu.fit()
    perm_dm = model_cpu.get_permuted_design_matrix()
    perm_p_cpu = model_cpu.fit_permuted(perm_dm)
    perm_stat_cpu = model_cpu.fit_permuted_stat(perm_dm)

    # 2. GPU Run (mocked CuPy)
    import oceangla.gpu as gpu_mod
    monkeypatch.setattr(gpu_mod, "cp", MockCuPy)
    monkeypatch.setattr(gpu_mod, "_CUPY_INSTALLED", True)
    config.use_gpu = True

    model_gpu = OLSModel(
        model_spec=spec,
        firstlevel_contrasts_df=firstlevel_df,
        subject_variables_df=subject_vars_df,
    )
    res_gpu = model_gpu.fit()
    perm_p_gpu = model_gpu.fit_permuted(perm_dm)
    perm_stat_gpu = model_gpu.fit_permuted_stat(perm_dm)

    # Check that GPU results are returned as host numpy arrays
    assert isinstance(res_gpu.tstats, np.ndarray)
    assert isinstance(res_gpu.betas, np.ndarray)
    assert isinstance(res_gpu.ses, np.ndarray)
    assert isinstance(res_gpu.uncorr_pvals, np.ndarray)
    assert isinstance(perm_p_gpu, np.ndarray)
    assert isinstance(perm_stat_gpu, np.ndarray)

    # Check identical values within precision
    np.testing.assert_allclose(res_gpu.betas, res_cpu.betas, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.ses, res_cpu.ses, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.tstats, res_cpu.tstats, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.uncorr_pvals, res_cpu.uncorr_pvals, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(perm_p_gpu, perm_p_cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(perm_stat_gpu, perm_stat_cpu, rtol=1e-5, atol=1e-5)


def test_twoway_anova_cpu_vs_gpu_identical(monkeypatch, tmp_path):
    """Verify TwoWayAnovaModel produces identical numerical results on CPU and GPU."""
    shape = (4, 4, 4, 3)
    n_subs = 4
    paths = []
    for sub_id in range(1, n_subs + 1):
        data = np.random.RandomState(sub_id + 10).randn(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        p = tmp_path / f"sub-{sub_id:02d}_fir_twoway.nii.gz"
        nib.save(img, p)
        paths.append(str(p))

    firstlevel_df = pd.DataFrame({
        "subject": [f"sub-{i:02d}" for i in range(1, n_subs + 1)],
        "condition": ["fir_cond"] * n_subs,
        "frame_no": [-1] * n_subs,
        "task": ["rest"] * n_subs,
        "space": ["MNI152NLin2009cAsym"] * n_subs,
        "session": ["ses-1"] * n_subs,
        "path": paths,
    })

    subject_vars_df = pd.DataFrame({
        "subject": [f"sub-{i:02d}" for i in range(1, n_subs + 1)],
        "group": ["ctrl", "ctrl", "patient", "patient"],
    })

    spec = ModelSpec(
        model_type="fir_twoway_rm_anova",
        model_name="test_twoway_gpu",
        function_args=["fir_cond", "C(group)"],
        session="ses-1",
        space="MNI152NLin2009cAsym",
        task="rest",
    )

    config.outdir_path = tmp_path

    # 1. CPU Run
    config.use_gpu = False
    model_cpu = TwoWayAnovaModel(
        model_spec=spec,
        firstlevel_contrasts_df=firstlevel_df,
        subject_variables_df=subject_vars_df,
    )
    res_cpu = model_cpu.fit()
    perm_dm = model_cpu.get_permuted_design_matrix()
    perm_p_cpu = model_cpu.fit_permuted(perm_dm)
    perm_stat_cpu = model_cpu.fit_permuted_stat(perm_dm)

    # 2. GPU Run (mocked CuPy)
    import oceangla.gpu as gpu_mod
    monkeypatch.setattr(gpu_mod, "cp", MockCuPy)
    monkeypatch.setattr(gpu_mod, "_CUPY_INSTALLED", True)
    config.use_gpu = True

    model_gpu = TwoWayAnovaModel(
        model_spec=spec,
        firstlevel_contrasts_df=firstlevel_df,
        subject_variables_df=subject_vars_df,
    )
    res_gpu = model_gpu.fit()
    perm_p_gpu = model_gpu.fit_permuted(perm_dm)
    perm_stat_gpu = model_gpu.fit_permuted_stat(perm_dm)

    assert isinstance(res_gpu.fstats, np.ndarray)
    assert isinstance(res_gpu.epsilons, np.ndarray)
    assert isinstance(res_gpu.uncorr_pvals, np.ndarray)
    assert isinstance(res_gpu.pval_sphericity_corr, np.ndarray)
    assert isinstance(perm_p_gpu, np.ndarray)
    assert isinstance(perm_stat_gpu, np.ndarray)

    np.testing.assert_allclose(res_gpu.fstats, res_cpu.fstats, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.epsilons, res_cpu.epsilons, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.uncorr_pvals, res_cpu.uncorr_pvals, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.pval_sphericity_corr, res_cpu.pval_sphericity_corr, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(perm_p_gpu, perm_p_cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(perm_stat_gpu, perm_stat_cpu, rtol=1e-5, atol=1e-5)


def test_rm_anova_cpu_vs_gpu_identical(monkeypatch, tmp_path):
    """Verify RMAnovaModel produces identical numerical results on CPU and GPU."""
    shape = (4, 4, 4, 3)
    n_subs = 3
    paths = []
    for sub_id in range(1, n_subs + 1):
        data = np.random.RandomState(sub_id + 20).randn(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        p = tmp_path / f"sub-{sub_id:02d}_fir_rm.nii.gz"
        nib.save(img, p)
        paths.append(str(p))

    firstlevel_df = pd.DataFrame({
        "subject": [f"sub-{i:02d}" for i in range(1, n_subs + 1)],
        "condition": ["fir_cond"] * n_subs,
        "frame_no": [-1] * n_subs,
        "task": ["rest"] * n_subs,
        "space": ["MNI152NLin2009cAsym"] * n_subs,
        "session": ["ses-1"] * n_subs,
        "path": paths,
    })

    subject_vars_df = pd.DataFrame({
        "subject": [f"sub-{i:02d}" for i in range(1, n_subs + 1)],
    })

    spec = ModelSpec(
        model_type="fir_rm_anova",
        model_name="test_rmanova_gpu",
        function_args=["fir_cond"],
        session="ses-1",
        space="MNI152NLin2009cAsym",
        task="rest",
    )

    config.outdir_path = tmp_path

    # 1. CPU Run
    config.use_gpu = False
    model_cpu = RMAnovaModel(
        model_spec=spec,
        firstlevel_contrasts_df=firstlevel_df,
        subject_variables_df=subject_vars_df,
    )
    res_cpu = model_cpu.fit()
    perm_dm = model_cpu.get_permuted_design_matrix()
    perm_p_cpu = model_cpu.fit_permuted(perm_dm)
    perm_stat_cpu = model_cpu.fit_permuted_stat(perm_dm)

    # 2. GPU Run (mocked CuPy)
    import oceangla.gpu as gpu_mod
    monkeypatch.setattr(gpu_mod, "cp", MockCuPy)
    monkeypatch.setattr(gpu_mod, "_CUPY_INSTALLED", True)
    config.use_gpu = True

    model_gpu = RMAnovaModel(
        model_spec=spec,
        firstlevel_contrasts_df=firstlevel_df,
        subject_variables_df=subject_vars_df,
    )
    res_gpu = model_gpu.fit()
    perm_p_gpu = model_gpu.fit_permuted(perm_dm)
    perm_stat_gpu = model_gpu.fit_permuted_stat(perm_dm)

    assert isinstance(res_gpu.fstats, np.ndarray)
    assert isinstance(res_gpu.epsilons, np.ndarray)
    assert isinstance(res_gpu.uncorr_pvals, np.ndarray)
    assert isinstance(res_gpu.pval_sphericity_corr, np.ndarray)
    assert isinstance(perm_p_gpu, np.ndarray)
    assert isinstance(perm_stat_gpu, np.ndarray)

    np.testing.assert_allclose(res_gpu.fstats, res_cpu.fstats, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.epsilons, res_cpu.epsilons, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.uncorr_pvals, res_cpu.uncorr_pvals, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_gpu.pval_sphericity_corr, res_cpu.pval_sphericity_corr, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(perm_p_gpu, perm_p_cpu, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(perm_stat_gpu, perm_stat_cpu, rtol=1e-5, atol=1e-5)
