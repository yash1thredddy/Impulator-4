"""Tests for the original-compound structural descriptor helper.

Covers ``frontend.ui.components.query_descriptors.compute_query_structural_descriptors``.

The load-bearing test is :func:`test_matches_backend_recipe_no_drift`: it
recomputes the expected ratios using the SAME RDKit primitives the backend's
``calculate_descriptors_for_smiles`` uses (ExactMolWt, TPSA from
CalcMolDescriptors, explicit N+O count). If the helper ever diverges from that
recipe, the original-compound marker would drift off the analog cloud — this
test catches that.
"""

import pytest

from frontend.ui.components.query_descriptors import (
    compute_query_structural_descriptors,
)

# Skip the whole module cleanly if RDKit is unavailable in the test env.
pytest.importorskip("rdkit")


def _expected_from_backend_recipe(smiles: str) -> dict[str, float]:
    """Independent reimplementation of the backend recipe for cross-checking."""
    from rdkit import Chem
    from rdkit.Chem.Descriptors import CalcMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    d = CalcMolDescriptors(mol)
    tpsa = d["TPSA"]
    mw = d["ExactMolWt"]
    heavy = d["HeavyAtomCount"]
    npol = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8))
    psaomw = tpsa / mw
    return {
        "PSAoMW": psaomw,
        "10xPSA_MW": 10 * psaomw,
        "NPOLoNHA": npol / heavy,
    }


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",  # quercetin
        "CCO",  # ethanol
    ],
)
def test_matches_backend_recipe_no_drift(smiles):
    """Helper output equals the backend recipe to ~9 dp for real molecules."""
    got = compute_query_structural_descriptors(smiles)
    expected = _expected_from_backend_recipe(smiles)
    assert got is not None
    for key, exp in expected.items():
        assert got[key] == pytest.approx(exp, rel=1e-9, abs=1e-9)


def test_psa_mw_is_ten_times_psaomw():
    """10xPSA_MW is exactly 10 × PSAoMW (the equivalence the user relies on)."""
    got = compute_query_structural_descriptors("CC(=O)Oc1ccccc1C(=O)O")
    assert got is not None
    assert got["10xPSA_MW"] == pytest.approx(10 * got["PSAoMW"])


def test_returns_none_for_unparseable_smiles():
    """A junk string yields None, not a crash or partial dict."""
    assert compute_query_structural_descriptors("!!!not-a-smiles!!!") is None


def test_returns_none_for_empty_input():
    assert compute_query_structural_descriptors("") is None
    assert compute_query_structural_descriptors("   ") is None
