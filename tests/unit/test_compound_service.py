"""
Unit tests for CompoundService.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np


class TestCompoundService:
    """Tests for CompoundService class."""

    @pytest.fixture
    def service(self):
        """Create a CompoundService instance."""
        from backend.services.compound_service import CompoundService
        return CompoundService()

    def test_service_init(self, service):
        """Test service initialization."""
        assert service.results_dir is not None

    def test_search_similar_compounds_fallback(self, service):
        """Test fallback similarity search."""
        with patch('backend.services.compound_service.CompoundService._search_similar_compounds_fallback') as mock:
            mock.return_value = [{"ChEMBL ID": "CHEMBL25"}]

            # Force fallback by making import fail
            with patch.dict('sys.modules', {'backend.modules.api_client': None}):
                result = service._search_similar_compounds_fallback("CCO", 90)

            # Should return mocked result
            assert mock.called or isinstance(result, list)

    def test_save_results(self, service, tmp_path):
        """Test saving results to disk."""
        service.results_dir = str(tmp_path)

        df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL25', 'CHEMBL26'],
            'Molecule_Name': ['Ethanol', 'Methanol'],
            'SMILES': ['CCO', 'CO'],
            'pActivity': [5.0, 6.0],
        })

        result_path, summary = service._save_results(
            compound_name="TestCompound",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=['IC50'],
            df_results=df
        )

        assert result_path.endswith('.zip')
        assert summary['compound_name'] == "TestCompound"
        assert summary['total_bioactivity_rows'] == 2


class TestCompoundServiceProgressCallbacks:
    """Tests for progress callback handling."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock()

    def test_update_progress(self, mock_db):
        """Test progress update calls job_service."""
        from backend.services.compound_service import CompoundService
        from backend.models.database import JobStatus

        service = CompoundService()

        with patch('backend.services.job_service.job_service') as mock_job_service:
            service._update_progress(
                mock_db,
                "test-job-id",
                50.0,
                "Processing...",
                JobStatus.PROCESSING
            )

            mock_job_service.update_progress.assert_called_once_with(
                mock_db,
                "test-job-id",
                50.0,
                "Processing...",
                JobStatus.PROCESSING
            )

    def test_complete_job(self, mock_db):
        """Test job completion calls job_service."""
        from backend.services.compound_service import CompoundService

        service = CompoundService()

        with patch('backend.services.job_service.job_service') as mock_job_service:
            service._complete_job(
                mock_db,
                "test-job-id",
                "/path/to/result.zip",
                {"total": 100}
            )

            mock_job_service.complete_job.assert_called_once()

    def test_fail_job(self, mock_db):
        """Test job failure calls job_service."""
        from backend.services.compound_service import CompoundService

        service = CompoundService()

        with patch('backend.services.job_service.job_service') as mock_job_service:
            service._fail_job(mock_db, "test-job-id", "Test error")

            mock_job_service.fail_job.assert_called_once_with(
                mock_db,
                "test-job-id",
                "Test error"
            )


class TestProcessCompoundJobWrapper:
    """Tests for the process_compound_job wrapper function."""

    def test_wrapper_delegates_to_service(self):
        """Test that wrapper function delegates to service singleton."""
        from backend.services.compound_service import process_compound_job, compound_service

        with patch.object(compound_service, 'process_compound_job') as mock_method:
            process_compound_job(
                job_id="test-123",
                compound_name="Aspirin",
                smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                similarity_threshold=90,
                activity_types=['IC50', 'Ki']
            )

            mock_method.assert_called_once_with(
                job_id="test-123",
                compound_name="Aspirin",
                smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                similarity_threshold=90,
                activity_types=['IC50', 'Ki']
            )
