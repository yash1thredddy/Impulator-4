"""Structural descriptors for the ORIGINAL (query) compound — pure, no Streamlit.

Used to place the original compound as a distinct marker on the structural
relationship plots (e.g. 10×PSA/MW vs NPOL/NHA) when it is *not* itself present
in the ChEMBL analog cloud. When the original IS in the cloud (it is a ChEMBL
compound, so it appears at ~100% similarity) the caller should reuse that row's
already-computed columns instead — this helper is the fallback for novel queries.

CANONICAL SOURCE: ``backend/services/compound_service.py`` ->
``calculate_descriptors_for_smiles``. The three formula choices below MUST stay
in lock-step with that function or the marker will drift off the cloud:

  * ``ExactMolWt`` (monoisotopic) — NOT ``MolWt`` (average)   [D-18]
  * ``TPSA`` taken from the ``CalcMolDescriptors`` batch call
  * ``NPOL`` = explicit count of N + O atoms (not in CalcMolDescriptors)

The no-drift guard lives in ``tests/unit/test_query_descriptors.py``: it
recomputes the expected values with the same RDKit primitives the backend uses
and asserts equality, so any change to the backend recipe surfaces as a failure.
"""

from __future__ import annotations

from typing import Optional


# Plot-axis columns this helper provides, named to match the results dataframe.
STRUCTURAL_DESCRIPTOR_COLS = ("PSAoMW", "10xPSA_MW", "NPOLoNHA")


def compute_query_structural_descriptors(smiles: str) -> Optional[dict[str, float]]:
    """Structural ratios for one SMILES, or ``None`` if it cannot be parsed.

    Returns a dict with ``PSAoMW``, ``10xPSA_MW`` and ``NPOLoNHA`` — the exact
    column names used by the relationship plots. A key is omitted only when its
    inputs are undefined (e.g. zero molecular weight or zero heavy-atom count),
    mirroring the backend's guarded assignment.
    """
    if not smiles or not str(smiles).strip():
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import CalcMolDescriptors

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        all_descs = CalcMolDescriptors(mol)
        tpsa = all_descs["TPSA"]
        heavy_atoms = all_descs["HeavyAtomCount"]
        mw = all_descs["ExactMolWt"]  # D-18: monoisotopic, matches the cloud

        # NPOL = nitrogen + oxygen atom count (not part of CalcMolDescriptors).
        npol = sum(
            1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (7, 8)
        )

        out: dict[str, float] = {}
        if mw and mw > 0 and tpsa is not None:
            out["PSAoMW"] = tpsa / mw
            out["10xPSA_MW"] = 10 * out["PSAoMW"]
        if heavy_atoms and heavy_atoms > 0:
            out["NPOLoNHA"] = npol / heavy_atoms

        return out or None
    except Exception:
        return None
