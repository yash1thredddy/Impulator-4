"""
Unit tests for compound_service module functions (async).
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd


# ---------------------------------------------------------------------------
# Common patch targets
# ---------------------------------------------------------------------------
_MOD = "backend.services.compound_service"


def _mock_clients():
    """Return patch context managers for all 3 httpx client factories."""
    cc = patch(f"{_MOD}.create_chembl_client", return_value=AsyncMock())
    cp = patch(f"{_MOD}.create_pdb_client", return_value=AsyncMock())
    cl = patch(f"{_MOD}.create_classifier_client", return_value=AsyncMock())
    return cc, cp, cl


def _mock_db_session():
    """Return a mock context-manager db session."""
    mock_db = MagicMock()
    mock_db.__enter__ = MagicMock(return_value=mock_db)
    mock_db.__exit__ = MagicMock(return_value=False)
    return mock_db


class TestModuleStructure:
    """Verify module-level structure: no CompoundService class, no JobCancelledException."""

    def test_no_compound_service_class(self):
        """CompoundService class must be deleted (D-13)."""
        import backend.services.compound_service as mod
        assert not hasattr(mod, 'CompoundService')

    def test_no_job_cancelled_exception(self):
        """JobCancelledException must be deleted (D-53)."""
        import backend.services.compound_service as mod
        assert not hasattr(mod, 'JobCancelledException')

    def test_no_is_job_cancelled(self):
        """_is_job_cancelled methods must be deleted."""
        import backend.services.compound_service as mod
        assert not hasattr(mod, '_is_job_cancelled')
        assert not hasattr(mod, '_is_job_cancelled_fresh')

    def test_process_compound_job_is_async(self):
        """process_compound_job must be async def (D-14)."""
        import inspect
        from backend.services.compound_service import process_compound_job
        assert inspect.iscoroutinefunction(process_compound_job)

    def test_cleanup_stale_folders_is_sync(self):
        import inspect
        from backend.services.compound_service import cleanup_stale_folders
        assert not inspect.iscoroutinefunction(cleanup_stale_folders)

    def test_scan_recovery_markers_is_sync(self):
        import inspect
        from backend.services.compound_service import scan_recovery_markers
        assert not inspect.iscoroutinefunction(scan_recovery_markers)

    def test_module_exports(self):
        from backend.services.compound_service import (
            process_compound_job,
            cleanup_stale_folders,
            scan_recovery_markers,
            get_compound_versions,
            delete_compound_with_cleanup,
            batch_delete_with_cleanup,
        )
        for fn in [process_compound_job, cleanup_stale_folders, scan_recovery_markers,
                    get_compound_versions, delete_compound_with_cleanup, batch_delete_with_cleanup]:
            assert callable(fn)


class TestCleanupStaleFolders:

    def test_cleanup_no_results_dir(self, tmp_path):
        from backend.services.compound_service import cleanup_stale_folders
        with patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = tmp_path / "nonexistent"
            assert cleanup_stale_folders() == 0

    def test_cleanup_removes_compound_folders(self, tmp_path):
        from backend.services.compound_service import cleanup_stale_folders
        (tmp_path / "Aspirin").mkdir()
        (tmp_path / "Caffeine").mkdir()
        (tmp_path / "3a").mkdir()
        (tmp_path / "result.zip").touch()

        with patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = tmp_path
            assert cleanup_stale_folders() == 2
            assert not (tmp_path / "Aspirin").exists()
            assert (tmp_path / "3a").exists()

    def test_cleanup_skips_hex_prefix_dirs(self, tmp_path):
        from backend.services.compound_service import cleanup_stale_folders
        for prefix in ["3a", "7f", "00", "ff"]:
            (tmp_path / prefix).mkdir()
        with patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = tmp_path
            assert cleanup_stale_folders() == 0

    def test_cleanup_handles_rmtree_error(self, tmp_path):
        from backend.services.compound_service import cleanup_stale_folders
        (tmp_path / "BadFolder").mkdir()
        with patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = tmp_path
            with patch('shutil.rmtree', side_effect=PermissionError("locked")):
                assert cleanup_stale_folders() == 0


class TestRecoveryMarkers:

    def test_scan_no_markers(self, tmp_path):
        from backend.services.compound_service import scan_recovery_markers
        with patch(f"{_MOD}.settings") as ms:
            ms.DATA_DIR = tmp_path
            assert scan_recovery_markers() == []

    def test_scan_finds_markers(self, tmp_path):
        import json
        from backend.services.compound_service import scan_recovery_markers
        data = {"job_id": "t-123", "entry_id": "abc", "compound_name": "T",
                "status": "COMPLETED", "result_summary": {}, "completed_at": "2026-01-01T00:00:00Z"}
        (tmp_path / ".recovery-abc.json").write_text(json.dumps(data))
        with patch(f"{_MOD}.settings") as ms:
            ms.DATA_DIR = tmp_path
            markers = scan_recovery_markers()
            assert len(markers) == 1
            assert markers[0]["job_id"] == "t-123"


class TestSaveResultsSync:

    def test_save_results_creates_zip(self, tmp_path):
        from backend.services.compound_service import _save_results_sync
        df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL25', 'CHEMBL26'],
            'Molecule_Name': ['Ethanol', 'Methanol'],
            'SMILES': ['CCO', 'CO'],
            'pActivity': [5.0, 6.0],
        })
        with patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = tmp_path
            result_path, summary = _save_results_sync(
                compound_name="TestCompound", smiles="CCO",
                similarity_threshold=90, activity_types=['IC50'], df_results=df,
            )
            assert str(result_path).endswith('.zip')
            assert summary['compound_name'] == "TestCompound"
            assert summary['total_bioactivity_rows'] == 2


class TestUpdateProgress:

    @pytest.mark.asyncio
    async def test_update_progress_calls_job_service(self):
        from backend.services.compound_service import _update_progress
        from backend.models.enums import JobStatus

        mock_js = MagicMock()
        mock_db = _mock_db_session()

        with patch(f"{_MOD}.get_db_session", return_value=mock_db):
            with patch('backend.services.job_service.job_service', mock_js):
                await _update_progress("j1", 50, "Processing...", JobStatus.PROCESSING)
                mock_js.update_progress.assert_called_once_with(
                    mock_db, "j1", 50, "Processing...", JobStatus.PROCESSING
                )


class TestMarkPendingUpload:

    @pytest.mark.asyncio
    async def test_mark_pending_upload_calls_job_service(self):
        from backend.services.compound_service import _mark_pending_upload_with_retry

        mock_js = MagicMock()
        mock_db = _mock_db_session()

        with patch(f"{_MOD}.get_db_session", return_value=mock_db):
            with patch('backend.services.job_service.job_service', mock_js):
                await _mark_pending_upload_with_retry("j1", {"entry_id": "abc"})
                mock_js.mark_pending_upload.assert_called_once_with(
                    mock_db, "j1", {"entry_id": "abc"}
                )


class TestFailJobWithRetry:

    @pytest.mark.asyncio
    async def test_fail_job_calls_job_service(self):
        from backend.services.compound_service import _fail_job_with_retry

        mock_js = MagicMock()
        mock_db = _mock_db_session()

        with patch(f"{_MOD}.get_db_session", return_value=mock_db):
            with patch('backend.services.job_service.job_service', mock_js):
                await _fail_job_with_retry("j1", "Test error")
                mock_js.fail_job.assert_called_once_with(
                    mock_db, "j1", "Test error", cascade_results=None
                )


class TestProcessCompoundJobCancellation:

    @pytest.mark.asyncio
    async def test_cancelled_error_does_not_touch_db_status(self):
        """CancelledError does not modify DB job status (D-52).

        process_compound_job catches CancelledError, logs it, and does NOT
        call _fail_job_with_retry -- the caller already set CANCELLED status.
        """
        from backend.services.compound_service import process_compound_job

        fail_mock = AsyncMock()
        # _update_progress raises CancelledError (simulates task.cancel())
        update_mock = AsyncMock(side_effect=asyncio.CancelledError())

        patches = {
            f"{_MOD}._update_progress": update_mock,
            f"{_MOD}._fail_job_with_retry": fail_mock,
            f"{_MOD}.create_chembl_client": MagicMock(return_value=AsyncMock()),
            f"{_MOD}.create_pdb_client": MagicMock(return_value=AsyncMock()),
            f"{_MOD}.create_classifier_client": MagicMock(return_value=AsyncMock()),
        }

        with _apply_patches(patches):
            # CancelledError is caught inside process_compound_job -- no re-raise
            await process_compound_job(job_id="j1", compound_name="T", smiles="CCO")
            # _fail_job_with_retry must NOT be called for cancellation
            fail_mock.assert_not_called()


class TestProcessCompoundJobFailure:

    @pytest.mark.asyncio
    async def test_unexpected_error_calls_fail_job(self):
        from backend.services.compound_service import process_compound_job

        fail_mock = AsyncMock()
        update_mock = AsyncMock()
        chembl_mock = AsyncMock(side_effect=RuntimeError("Unexpected"))

        patches = {
            f"{_MOD}._update_progress": update_mock,
            f"{_MOD}._fail_job_with_retry": fail_mock,
            f"{_MOD}.get_chembl_ids": chembl_mock,
            f"{_MOD}.create_chembl_client": MagicMock(return_value=AsyncMock()),
            f"{_MOD}.create_pdb_client": MagicMock(return_value=AsyncMock()),
            f"{_MOD}.create_classifier_client": MagicMock(return_value=AsyncMock()),
        }

        with _apply_patches(patches):
            await process_compound_job(job_id="j1", compound_name="T", smiles="CCO")
            fail_mock.assert_called_once()
            assert "RuntimeError" in fail_mock.call_args[0][1]


class TestProcessCompoundJobNoResults:

    @pytest.mark.asyncio
    async def test_no_similar_compounds_fails_job(self):
        from backend.services.compound_service import process_compound_job

        fail_mock = AsyncMock()

        patches = {
            f"{_MOD}._update_progress": AsyncMock(),
            f"{_MOD}._fail_job_with_retry": fail_mock,
            f"{_MOD}.get_chembl_ids": AsyncMock(return_value=[]),
            f"{_MOD}.cascade_similarity_counts": AsyncMock(return_value=[]),
            f"{_MOD}.create_chembl_client": MagicMock(return_value=AsyncMock()),
            f"{_MOD}.create_pdb_client": MagicMock(return_value=AsyncMock()),
            f"{_MOD}.create_classifier_client": MagicMock(return_value=AsyncMock()),
        }

        with _apply_patches(patches):
            await process_compound_job(job_id="j1", compound_name="T", smiles="CCO")
            fail_mock.assert_called_once()
            assert "No similar compounds" in fail_mock.call_args[0][1]


class TestPendingUploadFlow:

    @pytest.mark.asyncio
    async def test_no_azure_marks_completed_immediately(self):
        """Non-Azure: PENDING_UPLOAD -> COMPLETED immediately (D-35)."""
        from backend.services.compound_service import process_compound_job

        pending_mock = AsyncMock()
        completed_mock = MagicMock()

        patches = {
            f"{_MOD}._update_progress": AsyncMock(),
            f"{_MOD}._mark_pending_upload_with_retry": pending_mock,
            f"{_MOD}._mark_completed_sync": completed_mock,
            f"{_MOD}.get_chembl_ids": AsyncMock(return_value=[
                {"ChEMBL ID": "CHEMBL25", "Similarity": 95.0}
            ]),
            f"{_MOD}.fetch_all_activities_single_batch": AsyncMock(return_value=[{
                'molecule_chembl_id': 'CHEMBL25', 'standard_value': '100',
                'standard_units': 'nM', 'standard_type': 'IC50', 'target_chembl_id': 'C1',
            }]),
            f"{_MOD}.fetch_batch_molecule_data": AsyncMock(return_value={
                'CHEMBL25': {'pref_name': 'Test', 'molecule_properties': {'full_mwt': '200'},
                             'molecule_structures': {'canonical_smiles': 'CCO'}},
            }),
            f"{_MOD}.fetch_batch_target_names": AsyncMock(return_value={'C1': 'Target1'}),
            f"{_MOD}._calculate_molecular_descriptors_sync": lambda df: df,
            f"{_MOD}._add_assay_interference_flags_sync": lambda df: df,
            f"{_MOD}._calculate_advanced_metrics_sync": lambda df: df,
            f"{_MOD}.calculate_imp_score": AsyncMock(side_effect=lambda c, df, **kw: df),
            f"{_MOD}.add_imp_score_interpretation": lambda df: df,
            f"{_MOD}.classify_imp_candidates": lambda df, *a, **kw: df,
            f"{_MOD}._add_chemical_classification_async": AsyncMock(side_effect=lambda c, df: df),
            f"{_MOD}._build_all_similar_df_async": AsyncMock(return_value=pd.DataFrame()),
            f"{_MOD}._fetch_drug_indications_async": AsyncMock(return_value=pd.DataFrame()),
            f"{_MOD}._save_results_sync": lambda *a, **kw: ("/path/to/zip", {"entry_id": "abc"}),
            f"{_MOD}.is_azure_configured": lambda: False,
            f"{_MOD}.create_chembl_client": MagicMock(return_value=AsyncMock()),
            f"{_MOD}.create_pdb_client": MagicMock(return_value=AsyncMock()),
            f"{_MOD}.create_classifier_client": MagicMock(return_value=AsyncMock()),
        }

        with _apply_patches(patches):
            await process_compound_job(job_id="j1", compound_name="T", smiles="CCO")
            pending_mock.assert_called_once()
            completed_mock.assert_called_once()


class TestFetchActivitiesAsync:

    @pytest.mark.asyncio
    async def test_empty_chembl_ids_returns_empty(self):
        from backend.services.compound_service import _fetch_activities_async
        result = await _fetch_activities_async(MagicMock(), [], None, None)
        assert result == []


class TestFetchDrugIndicationsAsync:

    @pytest.mark.asyncio
    async def test_no_chembl_id_column(self):
        from backend.services.compound_service import _fetch_drug_indications_async
        df = pd.DataFrame({'SMILES': ['CCO']})
        result = await _fetch_drug_indications_async(MagicMock(), df)
        assert result.empty


class TestServicesInit:

    def test_no_compound_service_class_in_init(self):
        import backend.services as services
        assert not hasattr(services, 'CompoundService')
        assert not hasattr(services, 'compound_service_instance')

    def test_function_exports_in_init(self):
        from backend.services import process_compound_job, cleanup_stale_folders, scan_recovery_markers
        assert callable(process_compound_job)
        assert callable(cleanup_stale_folders)
        assert callable(scan_recovery_markers)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _apply_patches:
    """Context manager to apply multiple patches from a dict."""

    def __init__(self, patches_dict):
        self._patches = []
        self._mocks = {}
        for target, value in patches_dict.items():
            if callable(value) and not isinstance(value, (MagicMock, AsyncMock)):
                # Use side_effect for plain callables
                p = patch(target, side_effect=value)
            else:
                p = patch(target, value)
            self._patches.append(p)

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *args):
        for p in self._patches:
            p.stop()
