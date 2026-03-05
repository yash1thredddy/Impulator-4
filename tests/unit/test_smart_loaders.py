"""
Tests for smart_load_dataframe and smart_load_summary log levels.

Verifies that missing optional files (drug_indications.csv, pdb_summary.csv,
all_similar_molecules.csv) log at DEBUG level, not WARNING, to avoid
log spam on every Streamlit rerun.
"""
import io
import json
import logging
import zipfile

import pandas as pd
import pytest
from unittest.mock import patch


def _create_zip_bytes(files: dict) -> bytes:
    """Create a ZIP file in memory with the given filename→content mapping."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, content in files.items():
            if isinstance(content, str):
                zf.writestr(name, content)
            else:
                zf.writestr(name, json.dumps(content))
    return buf.getvalue()


class TestSmartLoadDataframeLogLevel:
    """Verify smart_load_dataframe logs at DEBUG for missing files."""

    @patch('frontend.services.azure_storage.smart_download_result')
    def test_missing_csv_logs_debug_not_warning(self, mock_download, caplog):
        """Missing CSV in ZIP should log at DEBUG, not WARNING."""
        from frontend.services.azure_storage import smart_load_dataframe

        # Create ZIP with only similar_compounds.csv, not drug_indications.csv
        zip_data = _create_zip_bytes({
            'similar_compounds.csv': 'col1,col2\na,b\n',
        })
        mock_download.return_value = zip_data

        with caplog.at_level(logging.DEBUG, logger='frontend.services.azure_storage'):
            result = smart_load_dataframe(
                'drug_indications.csv',
                entry_id='test-uuid-1234',
                storage_path=None
            )

        assert result is None

        # Should have a DEBUG log, not WARNING
        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]

        assert any('drug_indications.csv' in r.message for r in debug_msgs), \
            "Expected DEBUG log for missing drug_indications.csv"
        assert not any('drug_indications.csv' in r.message for r in warning_msgs), \
            "Should NOT have WARNING for missing optional CSV"

    @patch('frontend.services.azure_storage.smart_download_result')
    def test_existing_csv_loads_successfully(self, mock_download):
        """CSV that exists in ZIP should load as DataFrame."""
        from frontend.services.azure_storage import smart_load_dataframe

        zip_data = _create_zip_bytes({
            'similar_compounds.csv': 'ChEMBL_ID,SMILES\nCHEMBL25,CCO\n',
        })
        mock_download.return_value = zip_data

        result = smart_load_dataframe(
            'similar_compounds.csv',
            entry_id='test-uuid-1234',
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'ChEMBL_ID' in result.columns

    @patch('frontend.services.azure_storage.smart_download_result')
    def test_none_zip_returns_none_silently(self, mock_download):
        """When no ZIP data is available, should return None without logging."""
        from frontend.services.azure_storage import smart_load_dataframe

        mock_download.return_value = None

        result = smart_load_dataframe(
            'anything.csv',
            entry_id='test-uuid',
        )

        assert result is None


class TestSmartLoadSummaryLogLevel:
    """Verify smart_load_summary logs at DEBUG for missing summary.json."""

    @patch('frontend.services.azure_storage.smart_download_result')
    def test_missing_summary_logs_debug(self, mock_download, caplog):
        """Missing summary.json should log at DEBUG, not WARNING."""
        from frontend.services.azure_storage import smart_load_summary

        # ZIP without summary.json
        zip_data = _create_zip_bytes({
            'some_other_file.csv': 'data\n',
        })
        mock_download.return_value = zip_data

        with caplog.at_level(logging.DEBUG, logger='frontend.services.azure_storage'):
            result = smart_load_summary(entry_id='test-uuid-1234')

        assert result is None

        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]

        assert any('summary.json' in r.message for r in debug_msgs), \
            "Expected DEBUG log for missing summary.json"
        assert not any('summary.json' in r.message for r in warning_msgs), \
            "Should NOT have WARNING for missing summary.json"

    @patch('frontend.services.azure_storage.smart_download_result')
    def test_existing_summary_loads_successfully(self, mock_download):
        """summary.json that exists should load as dict."""
        from frontend.services.azure_storage import smart_load_summary

        summary = {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
        zip_data = _create_zip_bytes({
            'summary.json': summary,
        })
        mock_download.return_value = zip_data

        result = smart_load_summary(entry_id='test-uuid-1234')

        assert result is not None
        assert result['compound_name'] == 'Aspirin'
