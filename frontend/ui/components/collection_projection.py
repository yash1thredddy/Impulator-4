"""Pure chemical-space projection math for the Collection Combined view (no Streamlit, no IO).

Mirrors the ``imp_gmm`` / ``collection_preflight`` separation pattern: zero Streamlit /
plotly imports at module load, fully unit-testable in isolation. Sklearn-backed
``PCA`` and ``GaussianMixture`` are the canonical, citable, seed-reproducible algorithms.

**Pitfall 1 (RESEARCH §"2D chemical-space GMM"):** the Phase 22 1D IMP-score GMM helper
is **1D only** — it ``ravel().reshape(-1, 1)`` over a single IMP score. It MUST NOT be used
to cluster 2D projection coordinates. This module therefore calls
:class:`sklearn.mixture.GaussianMixture` **directly** on the 2D coordinates; the 1D helper
serves the IMP-score population only.

**Public API**:

Constants
    ``DEFAULT_RANDOM_STATE`` (int = 42) — the reproducibility invariant (D-S3-DEP). Every
    sklearn fit in this module uses this seed so two runs on identical input agree.
    ``DESCRIPTOR_COLS`` — the per-member descriptor columns aggregated from ``combined.csv``.

Functions
    :func:`build_descriptor_matrix`
        Per-member mean of the descriptor columns, NaN-imputed (T-24-03-02), StandardScaler-
        scaled → ``(N_members, n_features)``.
    :func:`project_pca`
        Deterministic 2D PCA of a scaled descriptor matrix → ``(N, 2)``.
    :func:`cluster_2d`
        2D GMM cluster labels via **direct** sklearn (Pitfall 1). ``ValueError`` on
        ``k < 1`` or ``k > N``; one structlog warning on non-convergence.
    :func:`project`
        ``method=("pca"|"umap")`` dispatcher gating on ``n_members >= 3``. The ``"umap"``
        branch **lazy-imports** umap inside the function and **degrades to PCA** (with a
        structlog warning) when umap-learn is not installed — so the module imports cleanly
        before plan 24-11 installs the dependency.

**Logging**: emits ``gmm_did_not_converge`` (cluster_2d non-convergence) and
``umap_unavailable_degrading_to_pca`` (project umap-degrade). Otherwise log-quiet.

**Manuscript citation block**: Chemical-space structure was reduced to two dimensions via
Principal Component Analysis (Pedregosa et al. 2011, scikit-learn) and, optionally, UMAP
(McInnes et al. 2018). Cluster structure on the 2D map was estimated with a Gaussian Mixture
Model (``covariance_type='full'``), all fits using a fixed ``random_state=42`` for
reproducibility.
"""

from __future__ import annotations

import warnings

import numpy as np
import structlog
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

#: Reproducibility invariant (D-S3-DEP). Every sklearn fit in this module uses
#: this seed so a re-run on identical input yields identical output.
DEFAULT_RANDOM_STATE: int = 42

#: Per-member descriptor columns on ``combined_activities.csv`` (verified live —
#: RESEARCH §"~80% already in combined.csv"). Aggregated by member, never recomputed.
DESCRIPTOR_COLS: tuple[str, ...] = (
    "Molecular_Weight",
    "LogP",
    "HBD",
    "HBA",
    "TPSA",
)

#: Member grain key on combined.csv (mirrors PATTERNS groupby("compound_name")).
MEMBER_KEY: str = "compound_name"

#: Minimum members for any 2D projection (UI-SPEC copy: "Need at least 3 members").
MIN_MEMBERS_FOR_PROJECTION: int = 3


# =============================================================================
# Stage 1 — descriptor matrix
# =============================================================================


def build_descriptor_matrix(combined_df):
    """Per-member, NaN-safe, StandardScaler-scaled descriptor matrix.

    One row per distinct member (``compound_name``), each feature the **mean** of
    that member's :data:`DESCRIPTOR_COLS` rows on ``combined.csv``. Multi-target
    members (several activity rows) collapse to one point.

    NaN mitigation (threat T-24-03-02): non-numeric values are coerced to NaN, then
    column means are imputed with the per-column mean across members; any column that
    is entirely NaN imputes to 0.0. This guarantees the returned matrix is finite so
    the downstream PCA / GMM fit cannot crash on missing descriptors.

    Parameters
    ----------
    combined_df : pandas.DataFrame
        Combined activities table with at least :data:`MEMBER_KEY` and the
        :data:`DESCRIPTOR_COLS` columns.

    Returns
    -------
    numpy.ndarray
        Shape ``(N_members, len(DESCRIPTOR_COLS))``, scaled to zero mean / unit
        variance per feature. Finite (no NaN/inf).
    """
    cols = [c for c in DESCRIPTOR_COLS if c in combined_df.columns]
    if not cols:
        raise ValueError(
            f"combined_df has none of the descriptor columns {DESCRIPTOR_COLS!r}"
        )

    # Per-member mean; non-numeric → NaN so a stray string can't poison the fit.
    numeric = combined_df[[MEMBER_KEY, *cols]].copy()
    for c in cols:
        numeric[c] = numeric[c].apply(_to_float)
    per_member = numeric.groupby(MEMBER_KEY, sort=True)[cols].mean()

    matrix = per_member.to_numpy(dtype=float)
    # Impute remaining NaN with per-column member mean; all-NaN columns → 0.0.
    with warnings.catch_warnings():
        # An all-NaN column makes np.nanmean emit "Mean of empty slice"; the NaN
        # result is handled on the next line, so silence the noise.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_means = np.nanmean(matrix, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    if matrix.shape[0] < 2:
        # Need ≥2 members for StandardScaler variance to be meaningful; the
        # ≥3-member projection gate in project() is the user-facing guard.
        raise ValueError(
            f"build_descriptor_matrix needs >=2 members; got {matrix.shape[0]}"
        )

    scaled = StandardScaler().fit_transform(matrix)
    # StandardScaler leaves a constant column at 0 variance → all zeros, still finite.
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def _to_float(value) -> float:
    """Best-effort float coercion; non-numeric → NaN (kept out of the fit)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# =============================================================================
# Stage 1 — PCA projection (deterministic)
# =============================================================================


def project_pca(X):
    """Deterministic 2D PCA projection of a scaled descriptor matrix.

    Parameters
    ----------
    X : array-like
        ``(N, n_features)`` scaled descriptor matrix (from
        :func:`build_descriptor_matrix`).

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` coordinates. Identical across repeated calls — sklearn fixes
        component sign via ``svd_flip`` and the solver is seeded with
        :data:`DEFAULT_RANDOM_STATE`.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(f"project_pca expects a 2D matrix; got shape {X_arr.shape!r}")
    n_components = min(2, X_arr.shape[1])
    coords = PCA(
        n_components=n_components,
        random_state=DEFAULT_RANDOM_STATE,
    ).fit_transform(X_arr)
    if coords.shape[1] == 1:
        # Degenerate single-feature input: pad to 2 columns so callers always get (N, 2).
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return coords


# =============================================================================
# Stage 2 — 2D GMM cluster labels (DIRECT sklearn — Pitfall 1)
# =============================================================================


def cluster_2d(coords_2d, k: int):
    """Cluster 2D projection coordinates with a **direct** sklearn GaussianMixture.

    Pitfall 1: this never routes through the 1D Phase 22 IMP-score GMM helper. The GMM is
    fit on the 2D coordinates with ``covariance_type='full'`` and
    :data:`DEFAULT_RANDOM_STATE`, so labels are stable across runs.

    Parameters
    ----------
    coords_2d : array-like
        ``(N, 2)`` projection coordinates.
    k : int
        Number of clusters. Must satisfy ``1 <= k <= N``.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` integer cluster labels.

    Raises
    ------
    ValueError
        If ``k < 1`` or ``k > N``.
    """
    X = np.asarray(coords_2d, dtype=float)
    n = X.shape[0]
    if int(k) < 1 or int(k) > n:
        raise ValueError(f"k must be in [1, {n}]; got {k}")

    model = GaussianMixture(
        n_components=int(k),
        covariance_type="full",
        random_state=DEFAULT_RANDOM_STATE,
    ).fit(X)

    if not model.converged_:
        logger.warning(
            "gmm_did_not_converge",
            n_components=int(k),
            n_samples=int(n),
            random_state=DEFAULT_RANDOM_STATE,
        )

    return model.predict(X)


# =============================================================================
# Stage 3 — projection dispatcher (PCA default; UMAP via lazy import + degrade)
# =============================================================================


def project(X, method: str = "pca", *, n_members: int):
    """Project a scaled descriptor matrix to 2D via ``method``.

    Parameters
    ----------
    X : array-like
        ``(N, n_features)`` scaled descriptor matrix.
    method : {"pca", "umap"}
        ``"pca"`` (default, no extra dependency) or ``"umap"``. The ``"umap"``
        branch **lazy-imports** umap *inside this function* so the module imports
        cleanly before plan 24-11 installs ``umap-learn``. If the import fails
        (package absent), the projection **degrades to PCA** and a single structlog
        warning (``umap_unavailable_degrading_to_pca``) is emitted.
    n_members : int
        Number of distinct members being projected. Projection is gated on
        ``n_members >= MIN_MEMBERS_FOR_PROJECTION`` (UI-SPEC copy: "Need at least
        3 members"); a smaller value raises ``ValueError`` so the caller can render
        the locked info message.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` coordinates.

    Raises
    ------
    ValueError
        If ``n_members < MIN_MEMBERS_FOR_PROJECTION`` or ``method`` is unknown.
    """
    if int(n_members) < MIN_MEMBERS_FOR_PROJECTION:
        raise ValueError(
            f"projection needs >= {MIN_MEMBERS_FOR_PROJECTION} members; "
            f"got {n_members}"
        )

    method = str(method).lower()
    if method == "pca":
        return project_pca(X)
    if method == "umap":
        return _project_umap(X, n_members=int(n_members))
    raise ValueError(f"unknown projection method {method!r}; expected 'pca' or 'umap'")


def _project_umap(X, *, n_members: int):
    """UMAP 2D embedding (deterministic, n_neighbors-clamped); degrade to PCA if absent.

    Pitfall 2: UMAP's default ``n_neighbors=15`` errors on small member sets, so
    ``n_neighbors`` is clamped to ``min(15, max(2, n_members - 1))``.
    Pitfall 3: ``random_state=42`` forces single-thread for reproducibility (accepted
    cost; the cache layer in 24-08 absorbs reruns).

    The ``import umap`` is **inside** this function (lazy) so the module imports cleanly
    before ``umap-learn`` is installed (plan 24-11). On ``ImportError`` the call degrades
    to :func:`project_pca` with a structlog warning.
    """
    try:
        import umap  # lazy: not a module-level import (24-11 gates the install)
    except ImportError:
        logger.warning(
            "umap_unavailable_degrading_to_pca",
            n_members=int(n_members),
        )
        return project_pca(X)

    X_arr = np.asarray(X, dtype=float)
    # Clamp to the ACTUAL row count, not the n_members hint — UMAP requires
    # n_neighbors < n_samples, and the two can disagree.
    n_neighbors = min(15, max(2, X_arr.shape[0] - 1))  # Pitfall 2 clamp
    reducer = umap.UMAP(
        n_components=2,
        random_state=DEFAULT_RANDOM_STATE,
        n_neighbors=n_neighbors,
        min_dist=0.1,
    )
    return reducer.fit_transform(X_arr)


__all__ = [
    "DEFAULT_RANDOM_STATE",
    "DESCRIPTOR_COLS",
    "MEMBER_KEY",
    "MIN_MEMBERS_FOR_PROJECTION",
    "build_descriptor_matrix",
    "project_pca",
    "cluster_2d",
    "project",
]
