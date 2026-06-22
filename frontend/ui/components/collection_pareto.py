"""Pure logic for the collection Pareto / trade-off front (no Streamlit, no IO).

Non-dominated sorting over a collection of members projected onto two or more
trade-off axes (e.g. potency vs efficiency vs IMP). Used by the Stage-3
Properties & Efficiency view (plan 24-12) to surface the Pareto front of "clean"
vs "panacea" compounds. See app_research/DESIGN_collection_combined_view.md §6.

CONVENTION: every axis passed to :func:`pareto_front` is HIGHER-is-better. The
caller must flip the sign of any lower-is-better axis (e.g. ``-toxicity``) before
calling. Hand-rolled per the research recommendation — no PyPI Pareto dependency.
"""

from __future__ import annotations

import numpy as np


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated (Pareto-optimal) rows.

    Args:
        points: ``(N, k)`` array where HIGHER is better on every axis. Flip the
            sign of any lower-is-better axis upstream of this call.

    Returns:
        Boolean array of length ``N``; ``True`` where the row is on the Pareto
        front (not strictly dominated by any other row). Point ``a`` dominates
        ``b`` when ``a >= b`` on every axis AND ``a > b`` on at least one. Equal
        points do not dominate one another, so duplicates are all kept.

    O(N^2). N is bounded by the member multiselect, so this stays cheap.
    """
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        # Point i is dominated if some other point is >= i on every axis and
        # strictly > i on at least one axis.
        dominators = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        if np.any(dominators):
            on_front[i] = False
    return on_front


__all__ = ["pareto_front"]
