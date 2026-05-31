"""Tests for CSV column detection logic."""
import pytest
import pandas as pd

pytest.importorskip("streamlit")

from frontend.ui.components.job_form import _detect_column_mappings


def test_inchikey_column_not_mapped_as_inchi():
    """InChIKey columns must NOT be detected as InChI."""
    df = pd.DataFrame(columns=['name', 'inchikey', 'smiles'])
    result = _detect_column_mappings(df)
    assert result['inchi'] is None, f"'inchikey' was incorrectly mapped as InChI: {result}"
    assert result.get('inchikey') == 'inchikey'


def test_inchi_key_column_not_mapped_as_inchi():
    """Underscore variant 'inchi_key' must NOT be detected as InChI."""
    df = pd.DataFrame(columns=['compound', 'inchi_key'])
    result = _detect_column_mappings(df)
    assert result['inchi'] is None
    assert result.get('inchikey') == 'inchi_key'


def test_actual_inchi_still_detected():
    """Real InChI columns must still be detected correctly."""
    df = pd.DataFrame(columns=['name', 'inchi', 'smiles'])
    result = _detect_column_mappings(df)
    assert result['inchi'] == 'inchi'


def test_exact_smiles_column_preferred_over_structure_alias():
    """When both 'Structure' and 'SMILES' columns exist, prefer the exact SMILES.

    'structure' is a SMILES alias, but a column literally named 'SMILES' is the
    more specific match and must win even when it appears later in the file.
    """
    df = pd.DataFrame(columns=['Structure', 'SMILES'])
    result = _detect_column_mappings(df)
    assert result['smiles'] == 'SMILES'


def test_structure_alias_used_when_no_smiles_column():
    """The 'structure' alias still maps to SMILES when no exact SMILES column exists."""
    df = pd.DataFrame(columns=['name', 'Structure'])
    result = _detect_column_mappings(df)
    assert result['smiles'] == 'Structure'
