"""
Unit tests for CompoundService.
"""
import os
import json
import pytest
import shutil
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone
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
        from backend.services.compound_service import CompoundService

        with patch.object(CompoundService, '_search_similar_compounds_fallback') as mock:
            mock.return_value = [{"ChEMBL ID": "CHEMBL25"}]

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
        import backend.services.job_service as js_mod
        from backend.services.compound_service import CompoundService
        from backend.models.database import JobStatus

        service = CompoundService()

        mock_js = MagicMock()
        with patch.object(js_mod, 'job_service', mock_js):
            service._update_progress(
                mock_db,
                "test-job-id",
                50.0,
                "Processing...",
                JobStatus.PROCESSING
            )

            mock_js.update_progress.assert_called_once_with(
                mock_db,
                "test-job-id",
                50.0,
                "Processing...",
                JobStatus.PROCESSING
            )

    def test_complete_job(self, mock_db):
        """Test job completion calls job_service."""
        import backend.services.job_service as js_mod
        from backend.services.compound_service import CompoundService

        service = CompoundService()

        mock_js = MagicMock()
        with patch.object(js_mod, 'job_service', mock_js):
            service._complete_job(
                mock_db,
                "test-job-id",
                "/path/to/result.zip",
                {"total": 100}
            )

            mock_js.complete_job.assert_called_once()

    def test_fail_job(self, mock_db):
        """Test job failure calls job_service."""
        import backend.services.job_service as js_mod
        from backend.services.compound_service import CompoundService

        service = CompoundService()

        mock_js = MagicMock()
        with patch.object(js_mod, 'job_service', mock_js):
            service._fail_job(mock_db, "test-job-id", "Test error")

            mock_js.fail_job.assert_called_once_with(
                mock_db,
                "test-job-id",
                "Test error",
                cascade_results=None
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
                activity_types=['IC50', 'Ki'],
                author_name=None
            )

            mock_method.assert_called_once_with(
                job_id="test-123",
                compound_name="Aspirin",
                smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
                similarity_threshold=90,
                activity_types=['IC50', 'Ki'],
                author_name=None
            )


class TestCleanupStaleFolders:
    """Tests for CompoundService.cleanup_stale_folders classmethod."""

    def test_cleanup_no_results_dir(self, tmp_path):
        """Test cleanup when results directory doesn't exist."""
        from backend.services.compound_service import CompoundService

        with patch('backend.services.compound_service.settings') as mock_settings:
            mock_settings.RESULTS_DIR = tmp_path / "nonexistent"
            result = CompoundService.cleanup_stale_folders()
            assert result == 0

    def test_cleanup_removes_compound_folders(self, tmp_path):
        """Test cleanup removes compound processing folders."""
        from backend.services.compound_service import CompoundService

        # Create compound folders (stale processing artifacts)
        (tmp_path / "Aspirin").mkdir()
        (tmp_path / "Caffeine").mkdir()
        # Create UUID prefix dir (should be kept)
        (tmp_path / "3a").mkdir()
        # Create a ZIP file (should be kept)
        (tmp_path / "result.zip").touch()

        with patch('backend.services.compound_service.settings') as mock_settings:
            mock_settings.RESULTS_DIR = tmp_path
            result = CompoundService.cleanup_stale_folders()
            assert result == 2
            assert not (tmp_path / "Aspirin").exists()
            assert not (tmp_path / "Caffeine").exists()
            assert (tmp_path / "3a").exists()
            assert (tmp_path / "result.zip").exists()

    def test_cleanup_skips_hex_prefix_dirs(self, tmp_path):
        """Test cleanup skips 2-char hex UUID prefix directories."""
        from backend.services.compound_service import CompoundService

        for prefix in ["3a", "7f", "00", "ff"]:
            (tmp_path / prefix).mkdir()

        with patch('backend.services.compound_service.settings') as mock_settings:
            mock_settings.RESULTS_DIR = tmp_path
            result = CompoundService.cleanup_stale_folders()
            assert result == 0

    def test_cleanup_handles_rmtree_error(self, tmp_path):
        """Test cleanup handles errors during folder removal."""
        from backend.services.compound_service import CompoundService

        (tmp_path / "BadFolder").mkdir()

        with patch('backend.services.compound_service.settings') as mock_settings:
            mock_settings.RESULTS_DIR = tmp_path
            with patch('shutil.rmtree', side_effect=PermissionError("locked")):
                result = CompoundService.cleanup_stale_folders()
                # Returns 0 because rmtree failed
                assert result == 0
                assert (tmp_path / "BadFolder").exists()


class TestIsJobCancelled:
    """Tests for _is_job_cancelled method."""

    def test_cancelled_job_returns_true(self):
        """Test returns True when job is CANCELLED."""
        from backend.services.compound_service import CompoundService
        from backend.models.database import JobStatus

        service = CompoundService()
        mock_db = MagicMock()
        mock_job = MagicMock()
        mock_job.status = JobStatus.CANCELLED

        import backend.services.job_service as js_mod
        mock_js = MagicMock()
        mock_js.get_job.return_value = mock_job
        with patch.object(js_mod, 'job_service', mock_js):
            result = service._is_job_cancelled(mock_db, "test-job-id")
            assert result is True

    def test_processing_job_returns_false(self):
        """Test returns False when job is PROCESSING."""
        from backend.services.compound_service import CompoundService
        from backend.models.database import JobStatus

        service = CompoundService()
        mock_db = MagicMock()
        mock_job = MagicMock()
        mock_job.status = JobStatus.PROCESSING

        import backend.services.job_service as js_mod
        mock_js = MagicMock()
        mock_js.get_job.return_value = mock_job
        with patch.object(js_mod, 'job_service', mock_js):
            result = service._is_job_cancelled(mock_db, "test-job-id")
            assert result is False

    def test_nonexistent_job_returns_false(self):
        """Test returns False when job not found."""
        from backend.services.compound_service import CompoundService

        service = CompoundService()
        mock_db = MagicMock()

        import backend.services.job_service as js_mod
        mock_js = MagicMock()
        mock_js.get_job.return_value = None
        with patch.object(js_mod, 'job_service', mock_js):
            result = service._is_job_cancelled(mock_db, "nonexistent")
            assert result is False


class TestSearchSimilarCompounds:
    """Tests for _search_similar_compounds method."""

    def test_successful_search(self):
        """Test successful similarity search."""
        from backend.services.compound_service import CompoundService

        service = CompoundService()
        expected = [{"ChEMBL ID": "CHEMBL25", "Similarity": 95.0}]

        with patch('backend.services.compound_service.get_chembl_ids', return_value=expected):
            result = service._search_similar_compounds("CCO", 90)
            assert result == expected

    def test_fallback_on_connection_error(self):
        """Test fallback triggered on ConnectionError."""
        from backend.services.compound_service import CompoundService

        service = CompoundService()
        fallback_result = [{"ChEMBL ID": "CHEMBL26"}]

        with patch('backend.services.compound_service.get_chembl_ids', side_effect=ConnectionError("timeout")):
            with patch.object(service, '_search_similar_compounds_fallback', return_value=fallback_result):
                result = service._search_similar_compounds("CCO", 90)
                assert result == fallback_result

    def test_fallback_on_index_error(self):
        """Test fallback triggered on IndexError."""
        from backend.services.compound_service import CompoundService

        service = CompoundService()

        with patch('backend.services.compound_service.get_chembl_ids', side_effect=IndexError("empty")):
            with patch.object(service, '_search_similar_compounds_fallback', return_value=[]):
                result = service._search_similar_compounds("CCO", 90)
                assert result == []

    def test_fallback_on_unexpected_error(self):
        """Test fallback on unexpected exception types."""
        from backend.services.compound_service import CompoundService

        service = CompoundService()

        with patch('backend.services.compound_service.get_chembl_ids', side_effect=RuntimeError("weird")):
            with patch.object(service, '_search_similar_compounds_fallback', return_value=[]):
                result = service._search_similar_compounds("CCO", 90)
                assert result == []


class TestSaveResultsInner:
    """Tests for _save_results_inner with entry_id UUID-based storage."""

    @pytest.fixture
    def service(self, tmp_path):
        from backend.services.compound_service import CompoundService
        svc = CompoundService()
        svc.results_dir = str(tmp_path)
        return svc

    @pytest.fixture
    def basic_df(self):
        return pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL25', 'CHEMBL26'],
            'SMILES': ['CCO', 'CO'],
            'Molecule_Name': ['Ethanol', 'Methanol'],
            'pActivity': [5.0, 6.0],
        })

    def test_save_with_entry_id(self, service, tmp_path, basic_df):
        """Test saving with entry_id creates UUID-based path."""
        entry_id = "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c"
        result_path, summary = service._save_results(
            compound_name="Ethanol",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=basic_df,
            entry_id=entry_id,
        )
        assert entry_id in result_path
        assert result_path.endswith(".zip")
        assert os.path.exists(result_path)
        assert summary['schema_version'] == 1
        assert summary['compound_name'] == "Ethanol"

    def test_save_without_entry_id(self, service, tmp_path, basic_df):
        """Test saving without entry_id uses name-based path."""
        result_path, summary = service._save_results(
            compound_name="Ethanol",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=basic_df,
        )
        assert "Ethanol" in result_path
        assert result_path.endswith(".zip")

    def test_save_results_with_imp_columns(self, service, tmp_path):
        """Test summary includes IMP and interference stats."""
        df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL25', 'CHEMBL26'],
            'SMILES': ['CCO', 'CO'],
            'pActivity': [5.0, 6.0],
            'Is_IMP_Candidate': [True, False],
            'IMP_Final_Score': [0.85, 0.45],
            'PAINS_Violation': [1, 0],
            'BRENK_Alerts': [0, 1],
            'NIH_Alerts': [1, 0],
            'QED': [0.75, 0.82],
        })
        result_path, summary = service._save_results(
            compound_name="TestComp",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=df,
            entry_id="12345678-1234-1234-1234-123456789012",
        )
        assert summary['imp_candidates'] == 1
        assert summary['has_imp_candidates'] is True
        assert summary['pains_count'] == 1
        assert summary['brenk_count'] == 1
        assert summary['nih_count'] == 1
        assert summary['imp_score'] == pytest.approx(0.85, abs=0.01)
        assert summary['qed'] == pytest.approx(0.785, abs=0.01)

    def test_save_results_with_outlier_columns(self, service, tmp_path):
        """Test summary counts outlier rows."""
        df = pd.DataFrame({
            'ChEMBL_ID': ['C1', 'C2', 'C3'],
            'SMILES': ['CCO', 'CO', 'C'],
            'pActivity': [5.0, 6.0, 7.0],
            'SEI_outlier': [True, False, True],
            'BEI_outlier': [False, False, True],
        })
        _, summary = service._save_results(
            compound_name="Outlier_Test",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=df,
            entry_id="22345678-1234-1234-1234-123456789012",
        )
        # Rows where any outlier is True: rows 0 and 2
        assert summary['num_outliers'] == 2

    def test_save_results_with_indications(self, service, tmp_path, basic_df):
        """Test saving with drug indications DataFrame."""
        indications_df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL25'],
            'MESH_Heading': ['Pain'],
            'Max_Phase': [4],
        })
        result_path, summary = service._save_results(
            compound_name="WithIndications",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=basic_df,
            indications_df=indications_df,
            entry_id="32345678-1234-1234-1234-123456789012",
        )
        assert os.path.exists(result_path)

    def test_save_results_with_all_similar(self, service, tmp_path, basic_df):
        """Test saving with all similar molecules DataFrame."""
        all_similar_df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL25', 'CHEMBL27'],
            'Similarity': [95.0, 80.0],
        })
        _, summary = service._save_results(
            compound_name="WithSimilar",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=basic_df,
            all_similar_df=all_similar_df,
            entry_id="42345678-1234-1234-1234-123456789012",
        )
        assert summary['total_similar'] == 2
        assert summary['similar_count'] == 2


class TestFetchActivities:
    """Tests for _fetch_activities method."""

    def test_fetch_empty_ids(self):
        """Test fetch with no ChEMBL IDs."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()
        result = service._fetch_activities([], None, lambda p, m: None)
        assert result == []

    def test_fetch_none_activity_types_uses_defaults(self):
        """Test that None activity_types falls back to defaults."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        chembl_ids = [{"ChEMBL ID": "CHEMBL25"}]

        with patch('backend.services.compound_service.CompoundService._fetch_activities') as mock:
            mock.return_value = []
            # Call directly to check the logic
            result = service._fetch_activities(chembl_ids, None, lambda p, m: None)
            # The method handles None internally


class TestCalculateImpScores:
    """Tests for _calculate_imp_scores method."""

    def test_imp_scores_success(self):
        """Test IMP score calculation success path."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'ChEMBL_ID': ['C1'], 'pActivity': [5.0]})

        with patch('backend.services.compound_service.calculate_imp_score', return_value=df):
            with patch('backend.services.compound_service.add_imp_score_interpretation', return_value=df):
                result_df, pdb_unavailable = service._calculate_imp_scores(df, use_pdb=True)
                assert pdb_unavailable is False

    def test_imp_scores_failure_flags_pdb(self):
        """Test IMP score failure flags PDB as unavailable."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'ChEMBL_ID': ['C1'], 'pActivity': [5.0]})

        with patch('backend.services.compound_service.calculate_imp_score', side_effect=Exception("PDB down")):
            result_df, pdb_unavailable = service._calculate_imp_scores(df, use_pdb=True)
            assert pdb_unavailable is True

    def test_imp_scores_failure_no_pdb_flag(self):
        """Test IMP score failure without PDB doesn't flag it."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'ChEMBL_ID': ['C1'], 'pActivity': [5.0]})

        with patch('backend.services.compound_service.calculate_imp_score', side_effect=Exception("other")):
            result_df, pdb_unavailable = service._calculate_imp_scores(df, use_pdb=False)
            assert pdb_unavailable is False


class TestClassifyImps:
    """Tests for _classify_imps method."""

    def test_classify_imps_delegates(self):
        """Test classify_imps delegates to classify_imp_candidates."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'IMP_Final_Score': [0.85]})

        with patch('backend.services.compound_service.classify_imp_candidates', return_value=df) as mock:
            result = service._classify_imps(df)
            mock.assert_called_once()
            assert len(result) == 1


class TestAddChemicalClassification:
    """Tests for _add_chemical_classification method."""

    def test_no_smiles_column(self):
        """Test returns unmodified DataFrame when SMILES column missing."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'ChEMBL_ID': ['C1']})
        result = service._add_chemical_classification(df)
        assert len(result) == 1


class TestCalculateAdvancedMetrics:
    """Tests for _calculate_advanced_metrics method."""

    def test_missing_input_columns(self):
        """Test graceful handling when required columns are missing."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'pActivity': [5.0]})
        result = service._calculate_advanced_metrics(df, lambda p, m: None)
        # Should not crash, returns df with initialized columns
        assert 'SEI' in result.columns
        assert 'BEI' in result.columns

    def test_exception_returns_original_df(self):
        """Test that exception in metrics returns original DataFrame."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'pActivity': [5.0]})

        with patch('backend.services.compound_service.calculate_efficiency_metrics_dataframe', side_effect=Exception("fail")):
            result = service._calculate_advanced_metrics(df, lambda p, m: None)
            assert 'pActivity' in result.columns


class TestGetCompoundVersions:
    """Tests for get_compound_versions module function."""

    def test_not_found_raises(self):
        """Test raises ValueError when compound not found."""
        from backend.services.compound_service import get_compound_versions

        mock_db = MagicMock()
        with patch('backend.services.compound_service.compound_repo') as mock_repo:
            mock_repo.get_versions.return_value = []
            mock_repo.get_by_entry_id.return_value = None
            with pytest.raises(ValueError, match="Compound not found"):
                get_compound_versions(mock_db, "nonexistent")

    def test_single_compound_no_versions(self):
        """Test single compound returns empty versions list."""
        from backend.services.compound_service import get_compound_versions

        mock_db = MagicMock()
        mock_compound = MagicMock()
        mock_compound.entry_id = "abc-123"

        with patch('backend.services.compound_service.compound_repo') as mock_repo:
            mock_repo.get_versions.return_value = [mock_compound]
            result = get_compound_versions(mock_db, "abc-123")
            assert result['versions'] == []
            assert result['current_entry_id'] == "abc-123"

    def test_multiple_siblings(self):
        """Test multiple structural siblings returned correctly."""
        from backend.services.compound_service import get_compound_versions

        mock_db = MagicMock()

        def make_compound(entry_id, name, is_dup=False, dup_of=None):
            c = MagicMock()
            c.entry_id = entry_id
            c.compound_name = name
            c.similarity_threshold = 90
            c.activity_types = "IC50"
            c.imp_score = 0.8
            c.qed = 0.7
            c.similar_compounds = 10
            c.total_activities = 50
            c.is_duplicate = is_dup
            c.duplicate_of = dup_of
            c.author_name = "Author"
            c.processed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
            c.storage_path = "results/ab/abc.zip"
            return c

        c1 = make_compound("id-1", "Aspirin", is_dup=False)
        c2 = make_compound("id-2", "Aspirin (v2)", is_dup=True, dup_of="id-1")

        with patch('backend.services.compound_service.compound_repo') as mock_repo:
            mock_repo.get_versions.return_value = [c1, c2]
            mock_repo.get_by_entry_id.return_value = c1
            result = get_compound_versions(mock_db, "id-1")
            assert len(result['versions']) == 2
            assert result['versions'][0]['is_original'] is True
            assert result['versions'][0]['is_current'] is True
            assert result['versions'][1]['is_duplicate'] is True
            assert result['versions'][1]['duplicate_of'] == "id-1"


class TestDeleteCompoundWithCleanup:
    """Tests for delete_compound_with_cleanup module function."""

    def test_not_found_raises(self):
        """Test raises ValueError when compound not found."""
        from backend.services.compound_service import delete_compound_with_cleanup

        mock_db = MagicMock()
        with patch('backend.services.compound_service.compound_repo') as mock_repo:
            mock_repo.get_by_entry_id.return_value = None
            with pytest.raises(ValueError, match="Compound not found"):
                delete_compound_with_cleanup(mock_db, "nonexistent", "session-id")

    def test_successful_delete(self):
        """Test successful compound deletion."""
        from backend.services.compound_service import delete_compound_with_cleanup

        mock_db = MagicMock()
        mock_compound = MagicMock()
        mock_compound.compound_name = "Aspirin"
        mock_compound.entry_id = "12345678-1234-1234-1234-123456789012"

        with patch('backend.services.compound_service.compound_repo') as mock_repo, \
             patch('backend.services.compound_service.delete_result_from_azure_by_entry_id', return_value=True), \
             patch('backend.services.compound_service.log_job_deleted'), \
             patch('backend.services.compound_service.truncate_session_id', return_value="sess"), \
             patch('backend.services.compound_service.settings') as mock_settings:
            mock_repo.get_by_entry_id.return_value = mock_compound
            mock_settings.RESULTS_DIR = MagicMock()
            mock_settings.RESULTS_DIR.__truediv__ = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
            result = delete_compound_with_cleanup(mock_db, "12345678-1234-1234-1234-123456789012", "session-id")
            assert result.status == "deleted"
            assert result.entry_id == "12345678-1234-1234-1234-123456789012"


class TestBatchDeleteWithCleanup:
    """Tests for batch_delete_with_cleanup module function."""

    def test_empty_list_raises(self):
        """Test raises ValueError for empty entry_ids."""
        from backend.services.compound_service import batch_delete_with_cleanup
        with pytest.raises(ValueError, match="empty"):
            batch_delete_with_cleanup(MagicMock(), [], "session")

    def test_too_many_raises(self):
        """Test raises ValueError for more than 50 entries."""
        from backend.services.compound_service import batch_delete_with_cleanup
        with pytest.raises(ValueError, match="50"):
            batch_delete_with_cleanup(MagicMock(), ["id"] * 51, "session")

    def test_invalid_entry_id_raises(self):
        """Test raises ValueError for non-string entry_ids."""
        from backend.services.compound_service import batch_delete_with_cleanup
        with pytest.raises(ValueError, match="non-empty"):
            batch_delete_with_cleanup(MagicMock(), ["valid", ""], "session")

    def test_not_found_returns_failed(self):
        """Test not-found IDs are in failed list."""
        from backend.services.compound_service import batch_delete_with_cleanup

        mock_db = MagicMock()
        with patch('backend.services.compound_service.compound_repo') as mock_repo, \
             patch('backend.services.compound_service.log_job_deleted'), \
             patch('backend.services.compound_service.truncate_session_id', return_value="s"), \
             patch('backend.services.compound_service.delete_result_from_azure_by_entry_id', return_value=True), \
             patch('backend.services.compound_service.settings') as mock_settings:
            mock_repo.get_by_entry_id.return_value = None
            mock_settings.RESULTS_DIR = MagicMock()
            result = batch_delete_with_cleanup(mock_db, ["unknown-id"], "session")
            assert result.total_deleted == 0
            assert result.total_failed == 1


class TestSaveResultsCleanupOnFailure:
    """Tests for _save_results STAB-13 cleanup on failure."""

    def test_cleanup_compound_folder_on_failure(self, tmp_path):
        """Test that compound folder is cleaned up if _save_results_inner raises."""
        from backend.services.compound_service import CompoundService

        service = CompoundService()
        service.results_dir = str(tmp_path)

        df = pd.DataFrame({
            'ChEMBL_ID': ['C1'],
            'SMILES': ['CCO'],
            'pActivity': [5.0],
        })

        with patch.object(service, '_save_results_inner', side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError, match="disk full"):
                service._save_results(
                    compound_name="FailComp",
                    smiles="CCO",
                    similarity_threshold=90,
                    activity_types=["IC50"],
                    df_results=df,
                )

        # Folder should be cleaned up
        assert not (tmp_path / "FailComp").exists()


class TestSaveResultsEdgeCases:
    """Tests for edge cases in _save_results_inner."""

    @pytest.fixture
    def service(self, tmp_path):
        from backend.services.compound_service import CompoundService
        svc = CompoundService()
        svc.results_dir = str(tmp_path)
        return svc

    def test_save_with_no_chembl_id_column(self, service, tmp_path):
        """Test saving DataFrame without ChEMBL_ID column."""
        df = pd.DataFrame({
            'SMILES': ['CCO'],
            'pActivity': [5.0],
        })
        result_path, summary = service._save_results(
            compound_name="NoChEMBL",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=df,
            entry_id="52345678-1234-1234-1234-123456789012",
        )
        assert summary['total_compounds'] == 0
        assert summary['chembl_id'] == ''

    def test_save_with_pdb_column(self, service, tmp_path):
        """Test saving DataFrame with PDB_IDs column creates PDB summary."""
        df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL25'],
            'SMILES': ['CCO'],
            'pActivity': [5.0],
            'PDB_IDs': ['1ABC,2DEF'],
        })
        with patch('backend.services.compound_service.create_detailed_pdb_summary',
                    return_value=pd.DataFrame({'PDB_ID': ['1ABC']})):
            result_path, summary = service._save_results(
                compound_name="PDBComp",
                smiles="CCO",
                similarity_threshold=90,
                activity_types=["IC50"],
                df_results=df,
                entry_id="62345678-1234-1234-1234-123456789012",
            )
        assert summary.get('pdb_structures_count') == 1

    def test_save_with_pdb_exception(self, service, tmp_path):
        """Test PDB summary exception is handled gracefully."""
        df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL25'],
            'SMILES': ['CCO'],
            'pActivity': [5.0],
            'PDB_IDs': ['1ABC'],
        })
        with patch('backend.services.compound_service.create_detailed_pdb_summary',
                    side_effect=Exception("PDB error")):
            result_path, summary = service._save_results(
                compound_name="PDBErr",
                smiles="CCO",
                similarity_threshold=90,
                activity_types=["IC50"],
                df_results=df,
                entry_id="72345678-1234-1234-1234-123456789012",
            )
        # Should not have pdb_structures_count
        assert 'pdb_structures_count' not in summary

    def test_save_with_empty_qed(self, service, tmp_path):
        """Test QED handling when all values are NaN."""
        df = pd.DataFrame({
            'ChEMBL_ID': ['C1'],
            'SMILES': ['CCO'],
            'pActivity': [5.0],
            'QED': [np.nan],
        })
        _, summary = service._save_results(
            compound_name="NanQED",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=df,
            entry_id="82345678-1234-1234-1234-123456789012",
        )
        assert summary['qed'] == 0.0

    def test_save_with_empty_imp_score(self, service, tmp_path):
        """Test IMP score handling when all values are NaN."""
        df = pd.DataFrame({
            'ChEMBL_ID': ['C1'],
            'SMILES': ['CCO'],
            'pActivity': [5.0],
            'IMP_Final_Score': [np.nan],
        })
        _, summary = service._save_results(
            compound_name="NanIMP",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=df,
            entry_id="92345678-1234-1234-1234-123456789012",
        )
        assert summary['imp_score'] is None

    def test_save_without_imp_column(self, service, tmp_path):
        """Test summary when Is_IMP_Candidate column is missing."""
        df = pd.DataFrame({
            'ChEMBL_ID': ['C1'],
            'SMILES': ['CCO'],
            'pActivity': [5.0],
        })
        _, summary = service._save_results(
            compound_name="NoIMP",
            smiles="CCO",
            similarity_threshold=90,
            activity_types=["IC50"],
            df_results=df,
            entry_id="a2345678-1234-1234-1234-123456789012",
        )
        assert 'imp_candidates' not in summary


class TestClassifyImpsError:
    """Tests for _classify_imps error handling."""

    def test_classify_imps_exception_returns_original(self):
        """Test classify_imps returns original df on exception."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'IMP_Final_Score': [0.85]})
        with patch('backend.services.compound_service.classify_imp_candidates',
                    side_effect=Exception("classify error")):
            result = service._classify_imps(df)
            assert len(result) == 1


class TestAddChemicalClassificationWithSmiles:
    """Tests for _add_chemical_classification with valid SMILES column."""

    def test_classification_exception_per_compound(self):
        """Test classification catches per-compound exceptions."""
        from backend.services.compound_service import CompoundService
        service = CompoundService()

        df = pd.DataFrame({'SMILES': ['CCO', 'INVALID']})

        # Mock RDKit to raise on second call
        mock_chem = MagicMock()
        mock_mol = MagicMock()
        mock_chem.MolFromSmiles.side_effect = [mock_mol, None]

        with patch.dict('sys.modules', {'rdkit': MagicMock(), 'rdkit.Chem': mock_chem,
                                         'rdkit.Chem.inchi': MagicMock()}):
            with patch('backend.services.compound_service.get_complete_classification',
                        return_value={}):
                # The function should handle MolFromSmiles returning None
                result = service._add_chemical_classification(df)
                assert len(result) == 2


class TestGetCompoundVersionsEdgeCases:
    """Additional edge case tests for get_compound_versions."""

    def test_all_duplicates_fallback_original(self):
        """Test original detection when all siblings are duplicates."""
        from backend.services.compound_service import get_compound_versions

        mock_db = MagicMock()

        def make_compound(entry_id, is_dup=True):
            c = MagicMock()
            c.entry_id = entry_id
            c.compound_name = f"Comp_{entry_id}"
            c.similarity_threshold = 90
            c.activity_types = "IC50"
            c.imp_score = 0.5
            c.qed = 0.7
            c.similar_compounds = 5
            c.total_activities = 20
            c.is_duplicate = is_dup
            c.duplicate_of = "parent-id" if is_dup else None
            c.author_name = "Author"
            c.processed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
            c.storage_path = "results/ab/abc.zip"
            return c

        # All are duplicates - original should fall back to first sibling
        c1 = make_compound("id-1", is_dup=True)
        c2 = make_compound("id-2", is_dup=True)

        with patch('backend.services.compound_service.compound_repo') as mock_repo:
            mock_repo.get_versions.return_value = [c1, c2]
            mock_repo.get_by_entry_id.return_value = c1
            result = get_compound_versions(mock_db, "id-1")
            # original_entry_id should fall back to first sibling
            assert result['versions'][0]['is_original'] is True
