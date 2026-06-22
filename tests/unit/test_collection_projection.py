"""Unit tests for the pure collection chemical-space projection module (no Streamlit).

Covers:
  - PCA projection shape + determinism (random_state=42).
  - 2D GMM cluster labels via DIRECT sklearn (Pitfall 1 — NOT imp_gmm.fit_gmm),
    seed-stable across runs, ValueError on bad k.
  - The ``project()`` dispatcher: PCA path, umap-degrade-to-PCA path (umap absent),
    and the ``n_members >= 3`` gate.
  - UMAP determinism / n_neighbors clamp — guarded by ``pytest.importorskip("umap")``
    so they SKIP (not fail) until plan 24-11 installs umap-learn.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frontend.ui.components import collection_projection as proj

FIXTURE = Path(__file__).parent / "fixtures" / "collection_toy_combined.csv"


def _combined_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


class TestBuildDescriptorMatrix:
    def test_one_row_per_member_scaled(self):
        df = _combined_df()
        X = proj.build_descriptor_matrix(df)
        # 3 distinct members in the toy fixture (ETHANOL appears twice).
        assert X.shape[0] == df["compound_name"].nunique()
        assert X.shape[1] == len(proj.DESCRIPTOR_COLS)
        # StandardScaler output: per-feature mean ~0.
        assert np.allclose(X.mean(axis=0), 0.0, atol=1e-9)

    def test_nan_rows_do_not_crash(self):
        df = _combined_df()
        df.loc[0, "Molecular_Weight"] = np.nan  # T-24-03-02 mitigation
        X = proj.build_descriptor_matrix(df)
        assert not np.isnan(X).any()


class TestProjectPca:
    def test_pca_shape(self):
        df = _combined_df()
        X = proj.build_descriptor_matrix(df)
        coords = proj.project_pca(X)
        assert coords.shape == (X.shape[0], 2)

    def test_pca_deterministic(self):
        df = _combined_df()
        X = proj.build_descriptor_matrix(df)
        a = proj.project_pca(X)
        b = proj.project_pca(X)
        assert np.array_equal(a, b)


class TestCluster2d:
    def test_gmm2d_seed(self):
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(12, 2))
        a = proj.cluster_2d(coords, k=2)
        b = proj.cluster_2d(coords, k=2)
        assert a.shape == (12,)
        assert np.array_equal(a, b)

    def test_gmm2d_uses_direct_sklearn(self):
        # The module must NEVER route 2D coords through imp_gmm.fit_gmm (Pitfall 1).
        src = Path(proj.__file__).read_text(encoding="utf-8")
        assert "fit_gmm" not in src

    def test_bad_k_raises(self):
        coords = np.zeros((5, 2))
        with pytest.raises(ValueError):
            proj.cluster_2d(coords, k=0)
        with pytest.raises(ValueError):
            proj.cluster_2d(coords, k=6)  # k > N


class TestProjectDispatcher:
    def test_dispatch_default_projects_2d(self):
        df = _combined_df()
        X = proj.build_descriptor_matrix(df)
        coords = proj.project(X, method="pca", n_members=X.shape[0])
        assert coords.shape == (X.shape[0], 2)

    def test_too_few_members_raises(self):
        X = np.zeros((2, 5))
        with pytest.raises(ValueError):
            proj.project(X, method="pca", n_members=2)

    def test_dispatch_umap_degrades_when_unavailable(self, monkeypatch):
        # Force the lazy ``import umap`` to fail so the degrade branch runs NOW
        # and keeps running after 24-11 installs umap-learn.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "umap" or name.startswith("umap."):
                raise ImportError("simulated: umap not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        df = _combined_df()
        X = proj.build_descriptor_matrix(df)
        coords = proj.project(X, method="umap", n_members=X.shape[0])
        # Degraded to PCA → identical to the PCA path.
        assert np.array_equal(coords, proj.project_pca(X))


class TestUmapInstalled:
    """Only run once umap-learn is installed (plan 24-11)."""

    def test_umap_seed(self):
        pytest.importorskip("umap")
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 5))
        a = proj.project(X, method="umap", n_members=20)
        b = proj.project(X, method="umap", n_members=20)
        assert a.shape == (20, 2)
        assert np.allclose(a, b)

    def test_umap_small(self):
        pytest.importorskip("umap")
        rng = np.random.default_rng(2)
        X = rng.normal(size=(4, 5))  # n_neighbors must clamp, not crash
        coords = proj.project(X, method="umap", n_members=4)
        assert coords.shape == (4, 2)
