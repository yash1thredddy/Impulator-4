"""Pure logic for collection SAR-lite / activity cliffs (no Streamlit, no IO).

Pairwise Tanimoto similarity over the SELECTED member subset plus activity-cliff
detection (structurally-similar pairs with divergent IMP). The codebase has no
in-process Tanimoto today (similarity is delegated to ChEMBL REST), so fresh
RDKit Morgan fingerprints are computed here. Used by the Stage-3 SAR view
(plan 24-12). See app_research/DESIGN_collection_combined_view.md §6b.

RDKit is imported lazily inside the function so import-time stays Streamlit/RDKit
-free and the module is unit-testable in isolation. The O(N^2) cost is bounded by
the member multiselect (compute over the selected subset only).
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# RDKit Morgan fingerprint parameters (research-recommended defaults).
_FP_RADIUS: int = 2
_FP_SIZE: int = 2048

# Activity-cliff defaults (UI-tunable in the view).
DEFAULT_SIM_THRESHOLD: float = 0.85
DEFAULT_DELTA_THRESHOLD: float = 20.0


def tanimoto_matrix(smiles_list: list[str]) -> np.ndarray:
    """Symmetric pairwise Tanimoto matrix over ``smiles_list``.

    Uses ``rdFingerprintGenerator.GetMorganGenerator`` (the current generator
    API, not the deprecated bit-vect helper) and
    ``DataStructs.BulkTanimotoSimilarity`` for the C-level bulk op. Malformed
    SMILES yield ``None`` mols and are
    guarded: their row/column keeps the identity default (1.0 on the diagonal,
    0.0 off-diagonal) and never raises.

    Returns:
        ``(N, N)`` float array, symmetric with a unit diagonal. Empty input
        returns a ``(0, 0)`` array.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    n = len(smiles_list)
    matrix = np.eye(n)
    if n == 0:
        return matrix

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=_FP_RADIUS, fpSize=_FP_SIZE)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [gen.GetFingerprint(m) if m is not None else None for m in mols]

    n_invalid = sum(1 for fp in fps if fp is None)
    if n_invalid:
        logger.warning("sar_invalid_smiles_skipped", n_invalid=n_invalid, n_total=n)

    for i in range(n):
        if fps[i] is None:
            continue
        valid = [j for j in range(i + 1, n) if fps[j] is not None]
        if not valid:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in valid])
        for j, sim in zip(valid, sims):
            matrix[i, j] = matrix[j, i] = sim
    return matrix


def activity_cliffs(
    matrix: np.ndarray,
    imp_scores: list[float],
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    delta_threshold: float = DEFAULT_DELTA_THRESHOLD,
) -> list[tuple[int, int]]:
    """Member index pairs flagged as activity cliffs.

    A cliff is a pair ``(i, j)`` (``i < j``) that is structurally similar
    (``matrix[i, j] > sim_threshold``) yet has a large IMP gap
    (``abs(imp_i - imp_j) > delta_threshold``).

    Returns:
        Upper-triangle ``(i, j)`` pairs in ascending order. No self-pairs, no
        ``(j, i)`` duplicates.
    """
    mat = np.asarray(matrix, dtype=float)
    imp = np.asarray(imp_scores, dtype=float)
    n = mat.shape[0]
    cliffs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] > sim_threshold and abs(imp[i] - imp[j]) > delta_threshold:
                cliffs.append((i, j))
    return cliffs


__all__ = [
    "DEFAULT_SIM_THRESHOLD",
    "DEFAULT_DELTA_THRESHOLD",
    "tanimoto_matrix",
    "activity_cliffs",
]
