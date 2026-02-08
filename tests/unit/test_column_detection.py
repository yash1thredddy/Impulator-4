"""Tests for CSV column detection logic."""
import pandas as pd
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
