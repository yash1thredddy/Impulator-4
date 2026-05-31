"""
IMP GMM Module — pure-Python Gaussian Mixture Model math for IMP score clustering.

Mirrors the ``imp_presentation`` separation pattern: zero Streamlit / pandas /
plotly imports, fully unit-testable in isolation. Sklearn-backed
``GaussianMixture`` is the canonical citable algorithm; ``scipy.stats.norm`` is
used for per-component weighted PDFs in :func:`component_curves`.

**Public API**:

Constants
    ``MIN_COMPONENTS`` (int = 2), ``MAX_COMPONENTS`` (int = 6),
    ``DEFAULT_COMPONENTS`` (int = 3), ``DEFAULT_RANDOM_STATE`` (int = 42),
    ``DENSITY_GRID`` (np.ndarray of shape (200,) covering [0, 100]),
    ``REFERENCE_CORPUS_KEY`` (str = ``"reference_corpus_v1"``).

Functions
    :func:`fit_gmm`
        Fit a ``GaussianMixture`` on integer-space [0, 100] IMP scores. Auto
        rescales raw [0, 1] inputs (Pitfall 4 mitigation). Uses
        ``covariance_type='full'`` per Phase 22 D-10. Emits one structlog
        warning (``gmm_did_not_converge``) when ``model.converged_`` is False
        and returns the model regardless (R6 fix from REVIEWS.md).
    :func:`cluster_membership`
        ``P(cluster_k | score)`` for a single integer-space [0, 100] score
        point, sorted by ascending cluster mean. Defensively normalized
        (Pitfall 2). R5 fix from REVIEWS.md: the score is treated as integer
        space; callers rescale raw [0, 1] inputs before calling.
    :func:`density_curve`
        Mixture PDF on a 1-D grid via ``np.exp(model.score_samples(x))``.
    :func:`component_curves`
        Per-component weighted Gaussian PDFs on a 1-D grid, in ascending-mean
        order. Used by chart helpers to overlay individual components on the
        density histogram.
    :func:`gmm_sentinel_message`
        Locked sentinel strings per Phase 22 UI-SPEC §Copywriting Contract.
        Variants: ``small_corpus``, ``zero_variance``, ``few_unique``.
    :func:`load_reference_corpus`
        Loads ``backend/data/imp_reference_corpus.json``; returns the
        ``compounds`` list (cached). Returns ``[]`` on any I/O or schema
        failure so the sentinel layer in Plan 05 triggers gracefully.

**Cluster ordering — anti-banding contract (UI-SPEC anti-requirement #12)**:
The argsort-by-mean call appears EXACTLY ONCE in this module, inside the
private :func:`_argsort_by_mean` helper. All public functions that need
ascending-mean ordering consume that helper; chart helpers in ``charts.py``
MUST NOT re-sort.

**Logging**: This module emits exactly one structlog event
(``gmm_did_not_converge`` at warning level, from :func:`fit_gmm`) when the
underlying GaussianMixture fails to converge. All other operations stay
log-quiet to keep the executor logs readable.

**Manuscript citation block**: Cluster structure of IMP scores was estimated
using Gaussian Mixture Models (Pedregosa et al. 2011, scikit-learn) with
expectation-maximisation fit, ``covariance_type='full'``, and a fixed
``random_state=42`` for reproducibility. The number of components K (2-6,
default 3) is user-selected via the Streamlit slider. Cluster-membership
probabilities ``P(cluster_k | score)`` are reported sorted by ascending
component mean.
"""

import functools
import json
from pathlib import Path

import numpy as np
import structlog
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

logger = structlog.get_logger(__name__)

# =============================================================================
# Slider / fit constants (locked per Phase 22 CONTEXT.md D-08 + D-09 + D-10)
# =============================================================================

MIN_COMPONENTS: int = 2
MAX_COMPONENTS: int = 6
DEFAULT_COMPONENTS: int = 3
DEFAULT_RANDOM_STATE: int = 42

# 200-point linspace over the integer IMP-score domain [0, 100]. Used for both
# the mixture density curve (Plotly overlay) and the per-component PDFs.
# 200 points keeps trapezoidal-integration accuracy within ~5% per RESEARCH.md
# Q3, which is plenty for a manuscript figure.
DENSITY_GRID: np.ndarray = np.linspace(0, 100, 200)

# Reference-corpus schema version key. Bump the suffix (``_v1`` → ``_v2``) when
# the JSON schema or compound list changes; the cache key in Plan 05 derives
# from this constant.
REFERENCE_CORPUS_KEY: str = "reference_corpus_v1"

# Reference-corpus JSON path: ``backend/data/imp_reference_corpus.json``.
# Resolved at module load time relative to this file (NOT cwd) so the path
# is stable regardless of where the test runner is invoked from.
_REFERENCE_CORPUS_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "imp_reference_corpus.json"
)


# =============================================================================
# Anti-banding contract: the single argsort-by-mean site in this module
# =============================================================================


def _argsort_by_mean(model: GaussianMixture) -> np.ndarray:
    """Return component indices that sort by ascending mean.

    Single source of truth for cluster ordering — chart helpers MUST consume
    pre-sorted output, never re-sort. See UI-SPEC anti-requirement #12.
    """
    return np.argsort(model.means_.flatten())


# =============================================================================
# Public API
# =============================================================================


def fit_gmm(
    scores,
    *,
    n_components: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> GaussianMixture:
    """Fit a ``GaussianMixture`` on integer-space IMP scores [0, 100].

    Parameters
    ----------
    scores : array-like
        IMP scores in either integer space [0, 100] or raw [0, 1]. Raw inputs
        are auto-rescaled (Pitfall 4 mitigation): if ``max(scores) <= 1.0``
        the array is multiplied by 100 before fitting.
    n_components : int
        Number of mixture components. Must satisfy
        ``MIN_COMPONENTS <= n_components <= MAX_COMPONENTS``.
    random_state : int, optional
        Random seed for reproducibility. Default 42 (D-09).

    Returns
    -------
    GaussianMixture
        The fitted model. Always returned, even when
        ``model.converged_ is False``. On non-convergence, a single structlog
        warning (``gmm_did_not_converge``) is emitted; the caller decides
        whether to render the sentinel.

    Raises
    ------
    ValueError
        If ``n_components`` is outside ``[MIN_COMPONENTS, MAX_COMPONENTS]``.
    """
    if not (MIN_COMPONENTS <= int(n_components) <= MAX_COMPONENTS):
        raise ValueError(
            f"n_components must be in [{MIN_COMPONENTS}, {MAX_COMPONENTS}]; "
            f"got {n_components}"
        )

    scores_arr = np.asarray(scores, dtype=float).ravel()
    # Pitfall 4: rescale raw [0, 1] inputs to integer space [0, 100] so that
    # fit-time covariance and predict-time score points share the same scale.
    if scores_arr.size > 0 and scores_arr.max() <= 1.0:
        scores_arr = scores_arr * 100.0

    X = scores_arr.reshape(-1, 1)
    model = GaussianMixture(
        n_components=int(n_components),
        covariance_type="full",
        random_state=int(random_state),
    ).fit(X)

    if not model.converged_:
        # R6 fix: surface non-convergence so the call site (Plan 05 widget)
        # can branch to the sentinel. We still return the model — the caller
        # chooses whether to render it or the sentinel.
        logger.warning(
            "gmm_did_not_converge",
            n_components=int(n_components),
            n_samples=int(scores_arr.size),
            random_state=int(random_state),
        )

    return model


def best_fit_k(
    scores,
    *,
    k_min: int = MIN_COMPONENTS,
    k_max: int = MAX_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> int:
    """Return the K in ``[k_min, k_max]`` that minimizes BIC on the given scores.

    Uses the **Bayesian Information Criterion** (lower is better) to balance
    likelihood gains against model complexity. This is the canonical
    model-selection approach for Gaussian Mixture Models per sklearn's User
    Guide §2.1.1. Each candidate K is fit with the same ``random_state`` and
    ``covariance_type='full'`` as :func:`fit_gmm`, so the BIC values are
    directly comparable. Non-converged fits are skipped (their BIC is
    untrustworthy). If every K fails or the corpus is smaller than ``k_min``,
    the function returns :data:`DEFAULT_COMPONENTS` as a safe fallback (the
    caller's sentinel layer will typically have caught small-corpus cases
    upstream).

    Parameters
    ----------
    scores : array-like
        IMP scores in integer space [0, 100] (or raw [0, 1] — auto-rescaled
        by :func:`fit_gmm`).
    k_min, k_max : int, optional
        Inclusive search range for K. Defaults: ``[MIN_COMPONENTS,
        MAX_COMPONENTS]`` = ``[2, 6]``.
    random_state : int, optional
        Same seed passed to every candidate fit so the BIC values are
        comparable. Default :data:`DEFAULT_RANDOM_STATE`.

    Returns
    -------
    int
        The K with the lowest BIC. Falls back to :data:`DEFAULT_COMPONENTS`
        when no K could be fit (tiny corpus or all fits non-convergent).

    Notes
    -----
    Manuscript citation: Schwarz (1978) for BIC; sklearn's ``GaussianMixture.bic``
    method is used internally.
    """
    scores_arr = np.asarray(scores, dtype=float).ravel()
    if scores_arr.size > 0 and scores_arr.max() <= 1.0:
        scores_arr = scores_arr * 100.0
    X = scores_arr.reshape(-1, 1)

    best_k: int | None = None
    best_bic = float("inf")
    for k in range(int(k_min), int(k_max) + 1):
        if scores_arr.size < k:
            break
        try:
            candidate = GaussianMixture(
                n_components=int(k),
                covariance_type="full",
                random_state=int(random_state),
            ).fit(X)
            if not candidate.converged_:
                continue
            bic = float(candidate.bic(X))
            if bic < best_bic:
                best_bic = bic
                best_k = int(k)
        except Exception:
            continue

    if best_k is None:
        return DEFAULT_COMPONENTS
    return best_k


def cluster_membership(model: GaussianMixture, score: float) -> list[float]:
    """Return ``P(cluster_k | score)`` sorted by ascending cluster mean.

    Parameters
    ----------
    model : GaussianMixture
        A fitted model from :func:`fit_gmm`.
    score : float
        The query point in INTEGER space [0, 100]. R5 fix from REVIEWS.md:
        this function NO LONGER rescales raw [0, 1] inputs; callers (the
        Plan 05 widget and the corpus path inside :func:`fit_gmm`) are
        responsible for rescaling. A bare value of ``1.0`` is therefore
        treated as integer-space 1 (low end), not as 100.

    Returns
    -------
    list[float]
        Probabilities of length ``model.n_components``, summing to exactly
        1.0 (defensively normalized — Pitfall 2), in ascending-mean order.
    """
    proba = model.predict_proba(np.array([[float(score)]]))[0]
    # Pitfall 2: predict_proba can return sums slightly off from 1.0 due to
    # underflow when one component dominates. Normalize defensively.
    total = proba.sum()
    if total > 0:
        proba = proba / total
    order = _argsort_by_mean(model)
    return [float(p) for p in proba[order]]


def density_curve(model: GaussianMixture, x_grid: np.ndarray) -> np.ndarray:
    """Return the mixture PDF on a 1-D grid.

    Parameters
    ----------
    model : GaussianMixture
        A fitted model.
    x_grid : array-like
        1-D grid of query points (integer-space [0, 100]).

    Returns
    -------
    np.ndarray
        Density values of shape ``(len(x_grid),)``. Always non-negative.
    """
    x = np.asarray(x_grid, dtype=float).reshape(-1, 1)
    return np.exp(model.score_samples(x))


def component_curves(
    model: GaussianMixture,
    x_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-component weighted PDFs in ascending-mean order.

    Parameters
    ----------
    model : GaussianMixture
        A fitted model.
    x_grid : array-like
        1-D grid of query points.

    Returns
    -------
    means_sorted : np.ndarray of shape (K,)
    weights_sorted : np.ndarray of shape (K,)
    sigmas_sorted : np.ndarray of shape (K,)
    pdfs_sorted : np.ndarray of shape (K, len(x_grid))
        Each row is ``weights_sorted[k] * norm.pdf(x_grid, means[k], sigmas[k])``.
    """
    order = _argsort_by_mean(model)
    means_sorted = model.means_.flatten()[order]
    weights_sorted = model.weights_[order]
    sigmas_sorted = np.sqrt(model.covariances_.flatten())[order]

    x = np.asarray(x_grid, dtype=float)
    pdfs_sorted = np.vstack(
        [
            weights_sorted[k] * norm.pdf(x, means_sorted[k], sigmas_sorted[k])
            for k in range(len(means_sorted))
        ]
    )
    return means_sorted, weights_sorted, sigmas_sorted, pdfs_sorted


def gmm_sentinel_message(
    n_components: int,
    n_samples: int,
    corpus_choice: str,
    *,
    variant: str = "small_corpus",
    n_unique: int | None = None,
) -> str:
    """Return the locked sentinel string for an insufficient-data condition.

    Strings are byte-exact per Phase 22 UI-SPEC §Copywriting Contract.

    Parameters
    ----------
    n_components : int
        The K value the user requested via the slider.
    n_samples : int
        The corpus size after grain filtering.
    corpus_choice : str
        A user-facing phrase like ``"query"`` or ``"reference corpus"`` —
        interpolated into the message.
    variant : str
        One of ``"small_corpus"``, ``"zero_variance"``, ``"few_unique"``.
    n_unique : int, optional
        Required only for the ``"few_unique"`` variant.

    Raises
    ------
    ValueError
        If ``variant`` is not one of the three locked names.
    """
    if variant == "small_corpus":
        return (
            f"Insufficient data — GMM with {n_components} components needs "
            f"at least {n_components + 1} compounds. This {corpus_choice} "
            f"has only {n_samples}. Try fewer components, or switch to a "
            f"different corpus above."
        )
    if variant == "zero_variance":
        return (
            f"Insufficient variation — all {n_samples} compounds in this "
            f"corpus have the same IMP score. GMM clustering requires "
            f"variation."
        )
    if variant == "few_unique":
        return (
            f"Insufficient unique scores — corpus has {n_unique} distinct "
            f"values but {n_components} components were requested. Try "
            f"fewer components."
        )
    raise ValueError(
        f"Unknown sentinel variant: {variant!r}. Expected one of "
        "'small_corpus', 'zero_variance', 'few_unique'."
    )


@functools.lru_cache(maxsize=1)
def load_reference_corpus() -> list[dict]:
    """Read the static reference-corpus JSON; return the ``compounds`` list.

    On any failure (missing file, invalid JSON, missing/incorrect
    ``corpus_key``, non-list ``compounds``) returns ``[]``. Downstream callers
    treat an empty corpus as a sentinel trigger (Plan 05 widget renders the
    insufficient-data message).

    Cached via :func:`functools.lru_cache` so repeated calls within the same
    Python process are O(1). Tests that need a fresh read should call
    ``load_reference_corpus.cache_clear()`` after monkeypatching the path.
    """
    try:
        with _REFERENCE_CORPUS_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

    if not isinstance(payload, dict):
        return []
    if payload.get("corpus_key") != REFERENCE_CORPUS_KEY:
        return []
    compounds = payload.get("compounds")
    if not isinstance(compounds, list):
        return []
    return compounds


__all__ = [
    "MIN_COMPONENTS",
    "MAX_COMPONENTS",
    "DEFAULT_COMPONENTS",
    "DEFAULT_RANDOM_STATE",
    "DENSITY_GRID",
    "REFERENCE_CORPUS_KEY",
    "fit_gmm",
    "cluster_membership",
    "density_curve",
    "component_curves",
    "gmm_sentinel_message",
    "load_reference_corpus",
]
