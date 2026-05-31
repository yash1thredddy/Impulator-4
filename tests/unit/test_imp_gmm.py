"""Tests for backend.modules.imp_gmm — pure-Python GMM math (sklearn-backed)."""

import json
import math
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.mixture import GaussianMixture
from structlog.testing import capture_logs

from backend.modules.imp_gmm import (
    DEFAULT_COMPONENTS,
    DEFAULT_RANDOM_STATE,
    DENSITY_GRID,
    MAX_COMPONENTS,
    MIN_COMPONENTS,
    REFERENCE_CORPUS_KEY,
    best_fit_k,
    cluster_membership,
    component_curves,
    density_curve,
    fit_gmm,
    gmm_sentinel_message,
    load_reference_corpus,
)


def _three_cluster_corpus(rng_seed: int = 0) -> np.ndarray:
    """Synthesize a deterministic 3-cluster integer-space corpus."""
    rng = np.random.default_rng(seed=rng_seed)
    return np.concatenate(
        [
            rng.normal(30, 5, 40),
            rng.normal(55, 5, 40),
            rng.normal(80, 5, 40),
        ]
    )


def _two_cluster_corpus(rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed=rng_seed)
    return np.concatenate(
        [
            rng.normal(30, 5, 30),
            rng.normal(70, 5, 30),
        ]
    )


# =============================================================================
# Constants
# =============================================================================


class TestConstants:
    """Verify exposed constants match the locked Phase 22 values."""

    def test_min_components_is_2(self):
        assert MIN_COMPONENTS == 2

    def test_max_components_is_6(self):
        assert MAX_COMPONENTS == 6

    def test_default_components_is_3(self):
        assert DEFAULT_COMPONENTS == 3

    def test_default_random_state_is_42(self):
        assert DEFAULT_RANDOM_STATE == 42

    def test_density_grid_is_200_points(self):
        assert DENSITY_GRID.shape == (200,)
        assert math.isclose(DENSITY_GRID[0], 0.0, abs_tol=1e-9)
        assert math.isclose(DENSITY_GRID[-1], 100.0, abs_tol=1e-9)

    def test_reference_corpus_key_is_v1(self):
        assert REFERENCE_CORPUS_KEY == "reference_corpus_v1"


# =============================================================================
# fit_gmm
# =============================================================================


class TestFitGmm:
    """Tests for fit_gmm (GMM-05, GMM-07, Pitfall 4)."""

    def test_fit_returns_gaussian_mixture_instance(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        assert isinstance(model, GaussianMixture)

    def test_fit_deterministic_with_fixed_seed(self):
        scores = _three_cluster_corpus()
        m1 = fit_gmm(scores, n_components=3, random_state=42)
        m2 = fit_gmm(scores, n_components=3, random_state=42)
        npt.assert_allclose(m1.means_, m2.means_)

    def test_fit_differs_with_different_seed(self):
        # Borderline corpus where different seeds can land on different local
        # optima. We assert "can differ"; not "must differ" — so we sample a
        # few seeds and expect at least one mismatch.
        rng = np.random.default_rng(seed=123)
        scores = rng.normal(50, 20, 60)
        baseline = fit_gmm(scores, n_components=3, random_state=42)
        any_differ = False
        for seed in (1, 7, 13, 99, 2024):
            other = fit_gmm(scores, n_components=3, random_state=seed)
            if not np.allclose(np.sort(baseline.means_.flatten()), np.sort(other.means_.flatten())):
                any_differ = True
                break
        assert any_differ, "Different seeds should be able to yield different fits"

    def test_fit_rescales_0_1_input_to_0_100(self):
        scores = np.linspace(0.1, 0.9, 40)
        model = fit_gmm(scores, n_components=3)
        # After Pitfall 4 rescaling the means should be in [0, 100], not [0, 1]
        assert model.means_.max() > 1.0
        assert model.means_.max() <= 100.0

    def test_fit_handles_already_0_100_input(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        assert model.means_.max() > 1.0
        assert model.means_.max() <= 100.0

    def test_fit_uses_covariance_type_full(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        assert model.covariance_type == "full"

    def test_fit_warns_on_non_convergence(self):
        # Use the same caught-by capture_logs pattern as the dedicated
        # TestNonConvergenceWarning class for parity.
        scores = np.array([50.0, 50.0, 50.1, 50.0, 50.0])
        with capture_logs() as logs:
            model = fit_gmm(scores, n_components=2)
        # Either model converged (no warning expected) or it didn't (warning
        # required). We tolerate both — the *contract* is: if not converged,
        # there must be a warning.
        if not model.converged_:
            assert any(
                e.get("event") == "gmm_did_not_converge" and e.get("log_level") == "warning"
                for e in logs
            )

    def test_fit_rejects_out_of_range_n_components(self):
        scores = _three_cluster_corpus()
        with pytest.raises(ValueError):
            fit_gmm(scores, n_components=1)
        with pytest.raises(ValueError):
            fit_gmm(scores, n_components=7)


# =============================================================================
# best_fit_k
# =============================================================================


class TestBestFitK:
    """Tests for best_fit_k (BIC-based automatic K selection)."""

    def test_returns_value_in_search_range(self):
        scores = _three_cluster_corpus()
        k = best_fit_k(scores)
        assert MIN_COMPONENTS <= k <= MAX_COMPONENTS

    def test_returns_integer(self):
        scores = _three_cluster_corpus()
        k = best_fit_k(scores)
        assert isinstance(k, int)

    def test_deterministic_with_fixed_seed(self):
        scores = _three_cluster_corpus()
        k1 = best_fit_k(scores, random_state=42)
        k2 = best_fit_k(scores, random_state=42)
        assert k1 == k2

    def test_respects_custom_k_range(self):
        scores = _three_cluster_corpus()
        k = best_fit_k(scores, k_min=2, k_max=3)
        assert 2 <= k <= 3

    def test_rescales_raw_zero_to_one_inputs(self):
        # Same shape, different scale — best K should remain the same after
        # the Pitfall-4 auto-rescale, because GMM math is shift/scale-invariant
        # when both the corpus and the candidate Ks see the same scale.
        scores_int = _three_cluster_corpus()
        scores_raw = scores_int / 100.0
        assert best_fit_k(scores_int) == best_fit_k(scores_raw)

    def test_falls_back_to_default_components_on_tiny_corpus(self):
        # Single point cannot fit any K in [2, 6]; expect the safe fallback.
        scores = np.array([50.0])
        assert best_fit_k(scores) == DEFAULT_COMPONENTS

    def test_falls_back_to_default_components_on_empty_corpus(self):
        assert best_fit_k(np.array([])) == DEFAULT_COMPONENTS

    def test_prefers_simpler_model_on_unimodal_data(self):
        # A tight unimodal cloud should not be best-fit with K=6.
        rng = np.random.default_rng(seed=0)
        scores = rng.normal(50, 5, 100)
        assert best_fit_k(scores) <= 3


# =============================================================================
# cluster_membership
# =============================================================================


class TestClusterMembership:
    """Tests for cluster_membership (GMM-04, Pitfall 2)."""

    def test_membership_sums_to_one_exactly(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        p = cluster_membership(model, 50.0)
        assert math.isclose(sum(p), 1.0, abs_tol=1e-9)

    def test_membership_sorted_by_ascending_mean(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        p = cluster_membership(model, 30.0)
        # When the query point sits at the lowest cluster center, p[0]
        # should dominate (the lowest-mean cluster comes first).
        assert p[0] > p[1]
        assert p[0] > p[2]

    def test_membership_returns_list_of_floats(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        p = cluster_membership(model, 50.0)
        assert isinstance(p, list)
        for x in p:
            assert isinstance(x, float)

    def test_membership_length_equals_n_components(self):
        scores = _three_cluster_corpus()
        for K in (MIN_COMPONENTS, DEFAULT_COMPONENTS, MAX_COMPONENTS):
            model = fit_gmm(scores, n_components=K)
            p = cluster_membership(model, 50.0)
            assert len(p) == K


# =============================================================================
# cluster_membership — score-space contract (R5 fix)
# =============================================================================


class TestClusterMembershipScoreSpace:
    """R5 fix: cluster_membership treats score as integer-space [0, 100]."""

    def test_membership_score_1_is_treated_as_integer_one(self):
        # Fit on a corpus that spans the full [0, 100] range
        scores = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        model = fit_gmm(scores, n_components=3)
        p = cluster_membership(model, 1.0)
        # Under R5: score=1.0 means "1 on the integer scale", so probability
        # mass must concentrate on the LOWEST cluster.
        assert p[0] > p[1]
        assert p[0] > p[2]

    def test_membership_does_not_rescale_inputs(self):
        scores = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        model = fit_gmm(scores, n_components=3)
        p_low = cluster_membership(model, 0.5)
        p_mid = cluster_membership(model, 50.0)
        # If the function still rescaled 0.5 → 50 these would be equal.
        assert not np.allclose(p_low, p_mid)


# =============================================================================
# density_curve
# =============================================================================


class TestDensityCurve:
    """Tests for density_curve (RESEARCH.md Q3)."""

    def test_density_shape_matches_grid(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        d = density_curve(model, DENSITY_GRID)
        assert d.shape == (200,)

    def test_density_integrates_to_one_on_default_grid(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        d = density_curve(model, DENSITY_GRID)
        area = float(np.trapezoid(d, DENSITY_GRID))
        # 200-point trapezoidal integration of a PDF that lives almost
        # entirely inside [0, 100] — should be near 1.0.
        assert math.isclose(area, 1.0, abs_tol=0.05)

    def test_density_non_negative(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        d = density_curve(model, DENSITY_GRID)
        assert np.all(d >= 0)


# =============================================================================
# component_curves
# =============================================================================


class TestComponentCurves:
    """Tests for component_curves."""

    def test_returns_four_arrays(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        out = component_curves(model, DENSITY_GRID)
        assert len(out) == 4
        means, weights, sigmas, pdfs = out
        assert isinstance(means, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert isinstance(sigmas, np.ndarray)
        assert isinstance(pdfs, np.ndarray)

    def test_means_sorted_ascending(self):
        scores = _three_cluster_corpus()
        model = fit_gmm(scores, n_components=3)
        means, _, _, _ = component_curves(model, DENSITY_GRID)
        assert np.all(np.diff(means) >= 0)

    def test_pdfs_shape_is_k_by_grid_len(self):
        scores = _three_cluster_corpus()
        K = 3
        model = fit_gmm(scores, n_components=K)
        _, _, _, pdfs = component_curves(model, DENSITY_GRID)
        assert pdfs.shape == (K, 200)


# =============================================================================
# gmm_sentinel_message (GMM-08)
# =============================================================================


class TestGmmSentinelMessage:
    """Tests for gmm_sentinel_message — locked strings per UI-SPEC."""

    def test_small_corpus_variant_locked_string(self):
        msg = gmm_sentinel_message(3, 2, "query", variant="small_corpus")
        assert msg == (
            "Insufficient data — GMM with 3 components needs at least 4 "
            "compounds. This query has only 2. Try fewer components, or "
            "switch to a different corpus above."
        )

    def test_zero_variance_variant_locked_string(self):
        msg = gmm_sentinel_message(3, 7, "query", variant="zero_variance")
        assert msg == (
            "Insufficient variation — all 7 compounds in this corpus have "
            "the same IMP score. GMM clustering requires variation."
        )

    def test_few_unique_variant_locked_string(self):
        msg = gmm_sentinel_message(
            3, 12, "query", variant="few_unique", n_unique=2
        )
        assert msg == (
            "Insufficient unique scores — corpus has 2 distinct values but "
            "3 components were requested. Try fewer components."
        )

    def test_string_includes_actual_counts(self):
        msg = gmm_sentinel_message(5, 4, "reference corpus", variant="small_corpus")
        assert "5 components" in msg
        assert "6 compounds" in msg
        assert "4" in msg
        assert "reference corpus" in msg

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            gmm_sentinel_message(3, 4, "query", variant="bogus")


# =============================================================================
# load_reference_corpus
# =============================================================================


class TestLoadReferenceCorpus:
    """Tests for load_reference_corpus."""

    def _patch_path(self, monkeypatch, tmp_path: Path, json_payload: object | None):
        target = tmp_path / "test_corpus.json"
        if json_payload is not None:
            target.write_text(
                json.dumps(json_payload) if not isinstance(json_payload, str) else json_payload,
                encoding="utf-8",
            )
        monkeypatch.setattr(
            "backend.modules.imp_gmm._REFERENCE_CORPUS_PATH", target
        )
        load_reference_corpus.cache_clear()

    def test_returns_list_on_valid_json(self, monkeypatch, tmp_path):
        self._patch_path(
            monkeypatch,
            tmp_path,
            {
                "corpus_key": "reference_corpus_v1",
                "compounds": [
                    {"compound_id": "A", "name": "alpha", "imp_final_score": 0.4},
                ],
            },
        )
        result = load_reference_corpus()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["compound_id"] == "A"

    def test_returns_empty_on_missing_file(self, monkeypatch, tmp_path):
        self._patch_path(monkeypatch, tmp_path, None)  # do not write file
        assert load_reference_corpus() == []

    def test_returns_empty_on_invalid_schema(self, monkeypatch, tmp_path):
        # File exists but is not JSON
        self._patch_path(monkeypatch, tmp_path, "{not valid json")
        assert load_reference_corpus() == []

    def test_corpus_key_matches_json_field(self, monkeypatch, tmp_path):
        # Wrong corpus_key → returns []
        self._patch_path(
            monkeypatch,
            tmp_path,
            {
                "corpus_key": "reference_corpus_v9",
                "compounds": [{"compound_id": "A"}],
            },
        )
        assert load_reference_corpus() == []


# =============================================================================
# Insufficient-data sentinel routing (sentinel-only contract; fit not exercised)
# =============================================================================


class TestInsufficientDataCases:
    """Sentinel-string coverage for degenerate inputs."""

    def test_n_samples_le_n_components(self):
        # Sentinel side: tests only the message, not the fit
        msg = gmm_sentinel_message(4, 3, "query", variant="small_corpus")
        assert "4 components" in msg
        assert "5 compounds" in msg
        assert "3" in msg

    def test_zero_variance_corpus(self):
        msg = gmm_sentinel_message(3, 5, "query", variant="zero_variance")
        assert "all 5 compounds" in msg
        assert "same IMP score" in msg

    def test_too_few_unique_scores(self):
        msg = gmm_sentinel_message(
            5, 30, "query", variant="few_unique", n_unique=3
        )
        assert "3 distinct values" in msg
        assert "5 components" in msg


# =============================================================================
# Non-convergence warning (R6 fix)
# =============================================================================


class TestNonConvergenceWarning:
    """R6 fix: fit_gmm emits a structlog warning when convergence fails."""

    def test_fit_emits_warning_when_model_does_not_converge(self, monkeypatch):
        # Force non-convergence by wrapping GaussianMixture with a partial
        # that pins max_iter=1 and tol very tight, so EM never settles on
        # a near-degenerate corpus.
        import backend.modules.imp_gmm as imp_gmm_mod

        original_cls = imp_gmm_mod.GaussianMixture

        def make_non_converging(*args, **kwargs):
            kwargs["max_iter"] = 1
            kwargs["tol"] = 1e-30
            return original_cls(*args, **kwargs)

        monkeypatch.setattr(imp_gmm_mod, "GaussianMixture", make_non_converging)

        scores = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5])
        with capture_logs() as logs:
            model = fit_gmm(scores, n_components=2)

        assert isinstance(model, GaussianMixture)
        assert model.converged_ is False
        assert any(
            e.get("event") == "gmm_did_not_converge" and e.get("log_level") == "warning"
            for e in logs
        )


# =============================================================================
# Golden corpus smoke test
# =============================================================================


class TestGoldenCorpusSmoke:
    """Smoke test against the 10-compound golden fixture."""

    def test_fits_on_golden_corpus(self):
        path = Path("tests/fixtures/golden_compounds.json")
        if not path.exists():
            pytest.skip("Golden fixture not present in this checkout")
        with path.open() as f:
            data = json.load(f)
        scores = [
            c["expected"]["IMP_Final_Score"]
            for c in data["compounds"]
            if c.get("expected", {}).get("IMP_Final_Score") is not None
        ]
        assert len(scores) >= 4
        model = fit_gmm(np.asarray(scores, dtype=float), n_components=3)
        assert model.converged_ is True
