import inspect
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import pytest

from oceangla.config import config
from oceangla.model import (
    Model,
    ModelResults,
    ModelSpec,
    FirstLevelContrast,
    OLSModel,
    TwoWayAnovaModel,
    RMAnovaModel,
    OLSModelResults,
    TwoWayAnovaModelResults,
    RMAnovaModelResults,
    AnovaModelResults,
)


def test_model_abc_inheritance():
    """Verify that OLSModel, TwoWayAnovaModel, and RMAnovaModel inherit from Model ABC."""
    assert issubclass(OLSModel, Model)
    assert issubclass(TwoWayAnovaModel, Model)
    assert issubclass(RMAnovaModel, Model)


def test_model_results_inheritance():
    """Verify that results classes inherit from ModelResults ABC."""
    assert issubclass(OLSModelResults, ModelResults)
    assert issubclass(TwoWayAnovaModelResults, ModelResults)
    assert issubclass(RMAnovaModelResults, ModelResults)
    assert issubclass(AnovaModelResults, ModelResults)


def test_model_constructor_signature():
    """Verify constructor signature takes (model_spec, firstlevel_contrasts_df, subject_variables_df)."""
    for cls in (OLSModel, TwoWayAnovaModel, RMAnovaModel):
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        # First parameter is self
        assert params[1] == "model_spec", f"{cls.__name__} missing model_spec parameter"
        assert params[2] == "firstlevel_contrasts_df", f"{cls.__name__} missing firstlevel_contrasts_df parameter"
        assert params[3] == "subject_variables_df", f"{cls.__name__} missing subject_variables_df parameter"


def test_model_results_has_required_methods():
    """Verify that ModelResults defines fdr_correct, cluster_correct, and tfce_correct."""
    for method_name in ("fdr_correct", "cluster_correct", "tfce_correct", "save"):
        assert hasattr(ModelResults, method_name), f"ModelResults missing {method_name}"
        assert callable(getattr(ModelResults, method_name))


def test_ols_model_results_and_corrections_nifti():
    """Integration test: fit OLSModel on synthetic NIfTI data and test FDR, cluster, and TFCE corrections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config.outdir_path = tmp_path

        # Create synthetic NIfTI files for 4 subjects
        shape = (6, 6, 6)
        n_subs = 4
        paths = []
        for sub_id in range(1, n_subs + 1):
            data = np.random.randn(*shape).astype(np.float32)
            # Add signal in a small cluster
            data[2:4, 2:4, 2:4] += 3.0
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
            "age": [20, 25, 30, 35],
        })

        spec = ModelSpec(
            model_type="OLS",
            model_name="test_ols",
            firstlevel_contrasts=[FirstLevelContrast(name="condA", scalar=1)],
            subject_variables="age",
            session="ses-1",
            space="MNI152NLin2009cAsym",
            task="rest",
        )

        # 1. Instantiate model
        model = OLSModel(
            model_spec=spec,
            firstlevel_contrasts_df=firstlevel_df,
            subject_variables_df=subject_vars_df,
        )

        assert isinstance(model, Model)
        assert model.fdata is not None
        assert model.design_df is not None

        # 2. Fit model
        results = model.fit()

        assert isinstance(results, OLSModelResults)
        assert isinstance(results, ModelResults)
        # Check reference to parent model
        assert results.model is model
        # Check specific statistics
        assert results.tstats is not None
        assert results.betas is not None
        assert results.ses is not None
        assert results.uncorr_pvals is not None
        assert results.tstats.shape == results.uncorr_pvals.shape

        # 3. Test save
        results.save()
        assert (model.model_outdir / "design_matrix.tsv").exists()

        # 4. Test FDR correction
        fdr_pvals = results.fdr_correct(alpha=0.05)
        assert fdr_pvals is not None
        assert fdr_pvals.shape == results.uncorr_pvals.shape

        # 5. Test thresholded cluster correction with volumetric method kwarg ("nn1", "nn2", "nn3")
        for method in ("nn1", "nn2", "nn3"):
            clus_pvals = results.cluster_correct(n_perms=5, alpha=0.05, method=method)
            assert clus_pvals is not None
            assert clus_pvals.shape == results.uncorr_pvals.shape

        # Verify invalid method raises ValueError
        with pytest.raises(ValueError, match="method must be one of"):
            results.cluster_correct(n_perms=2, method="invalid_method")

        # 6. Test TFCE correction
        tfce_pvals = results.tfce_correct(n_perms=5, alpha=0.05, method="nn1")
        assert tfce_pvals is not None
        assert tfce_pvals.shape == results.uncorr_pvals.shape


def test_rm_anova_results():
    """Integration test: fit RMAnovaModel on synthetic NIfTI data and test ModelResults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config.outdir_path = tmp_path

        # Create 4D NIfTI files (4 frames per subject)
        shape = (4, 4, 4, 3)
        n_subs = 3
        paths = []
        for sub_id in range(1, n_subs + 1):
            data = np.random.randn(*shape).astype(np.float32)
            img = nib.Nifti1Image(data, np.eye(4))
            p = tmp_path / f"sub-{sub_id:02d}_fir.nii.gz"
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
            model_name="test_rmanova",
            function_args=["fir_cond"],
            session="ses-1",
            space="MNI152NLin2009cAsym",
            task="rest",
        )

        model = RMAnovaModel(
            model_spec=spec,
            firstlevel_contrasts_df=firstlevel_df,
            subject_variables_df=subject_vars_df,
        )

        assert isinstance(model, Model)
        results = model.fit()

        assert isinstance(results, RMAnovaModelResults)
        assert isinstance(results, ModelResults)
        assert results.model is model
        # RMAnova returns F-statistics
        assert results.fstats is not None
        assert results.epsilons is not None
        assert results.pval_sphericity_corr is not None
        assert results.uncorr_pvals is not None

        # Test corrections
        results.save()
        fdr_p = results.fdr_correct()
        assert fdr_p is not None

        clus_p = results.cluster_correct(n_perms=3, alpha=0.05, method="nn2")
        assert clus_p is not None

        tfce_p = results.tfce_correct(n_perms=3, alpha=0.05, method="nn2")
        assert tfce_p is not None


def test_twoway_anova_results():
    """Integration test: fit TwoWayAnovaModel on synthetic NIfTI data and test ModelResults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config.outdir_path = tmp_path

        # Create 4D NIfTI files (3 frames per subject, 4 subjects: 2 per group)
        shape = (4, 4, 4, 3)
        n_subs = 4
        paths = []
        for sub_id in range(1, n_subs + 1):
            data = np.random.randn(*shape).astype(np.float32)
            img = nib.Nifti1Image(data, np.eye(4))
            p = tmp_path / f"sub-{sub_id:02d}_fir.nii.gz"
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
            model_name="test_twoway_anova",
            function_args=["fir_cond", "C(group)"],
            session="ses-1",
            space="MNI152NLin2009cAsym",
            task="rest",
        )

        model = TwoWayAnovaModel(
            model_spec=spec,
            firstlevel_contrasts_df=firstlevel_df,
            subject_variables_df=subject_vars_df,
        )

        assert isinstance(model, Model)
        results = model.fit()

        assert isinstance(results, TwoWayAnovaModelResults)
        assert isinstance(results, ModelResults)
        assert results.model is model
        # TwoWayAnova returns F-statistics
        assert results.fstats is not None
        assert results.epsilons is not None
        assert results.pval_sphericity_corr is not None
        assert results.uncorr_pvals is not None

        # Test corrections
        results.save()
        fdr_p = results.fdr_correct()
        assert fdr_p is not None

        clus_p = results.cluster_correct(n_perms=3, alpha=0.05, method="nn3")
        assert clus_p is not None

        tfce_p = results.tfce_correct(n_perms=3, alpha=0.05, method="nn3")
        assert tfce_p is not None


def test_cifti_fdr_and_save():
    """Test OLSModel and ModelResults with synthetic CIFTI image."""
    from nibabel.cifti2 import Cifti2Image
    from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config.outdir_path = tmp_path

        n_subs = 3
        n_verts = 10
        paths = []
        for sub_id in range(1, n_subs + 1):
            bm = BrainModelAxis(
                name=["CIFTI_STRUCTURE_CORTEX_LEFT"] * n_verts,
                vertex=np.arange(n_verts, dtype=np.int64),
                nvertices={"CIFTI_STRUCTURE_CORTEX_LEFT": n_verts},
            )
            sc = ScalarAxis([f"sub-{sub_id:02d}"])
            data = np.random.randn(1, n_verts).astype(np.float32)
            img = Cifti2Image(data, header=(sc, bm))
            p = tmp_path / f"sub-{sub_id:02d}_contrast.dscalar.nii"
            nib.save(img, p)
            paths.append(str(p))

        firstlevel_df = pd.DataFrame({
            "subject": [f"sub-{i:02d}" for i in range(1, n_subs + 1)],
            "condition": ["condA"] * n_subs,
            "frame_no": [-1] * n_subs,
            "task": ["rest"] * n_subs,
            "space": ["fsLR"] * n_subs,
            "session": ["ses-1"] * n_subs,
            "path": paths,
        })

        subject_vars_df = pd.DataFrame({
            "subject": [f"sub-{i:02d}" for i in range(1, n_subs + 1)],
            "age": [20, 25, 30],
        })

        spec = ModelSpec(
            model_type="OLS",
            model_name="test_ols_cifti",
            firstlevel_contrasts=[FirstLevelContrast(name="condA", scalar=1)],
            subject_variables="age",
            session="ses-1",
            space="fsLR",
            task="rest",
        )

        model = OLSModel(
            model_spec=spec,
            firstlevel_contrasts_df=firstlevel_df,
            subject_variables_df=subject_vars_df,
        )

        results = model.fit()
        assert isinstance(results, OLSModelResults)
        assert results.tstats.shape[1] == n_verts

        results.save()
        fdr_p = results.fdr_correct(alpha=0.05)
        assert fdr_p is not None
        assert fdr_p.shape[1] == n_verts
