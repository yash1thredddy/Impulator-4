"""
Unit tests for upload_worker module.
"""
import asyncio
import inspect
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

_MOD = "backend.core.upload_worker"


class TestModuleAPI:
    """Verify upload_worker has the expected public API."""

    def test_start_is_callable(self):
        from backend.core.upload_worker import start
        assert callable(start)

    def test_stop_is_async(self):
        from backend.core.upload_worker import stop
        assert inspect.iscoroutinefunction(stop)

    def test_is_active_is_callable(self):
        from backend.core.upload_worker import is_active
        assert callable(is_active)

    def test_constants(self):
        from backend.core import upload_worker
        assert upload_worker.MAX_UPLOAD_ATTEMPTS == 50
        assert upload_worker.MAX_REQUEUE_CYCLES == 3
        assert upload_worker.INITIAL_OUTER_SLEEP == 180
        assert upload_worker.MAX_OUTER_SLEEP == 1800


class TestIsActive:

    def test_is_active_false_when_no_task(self):
        from backend.core import upload_worker
        upload_worker._reset()
        assert upload_worker.is_active() is False

    def test_is_active_true_when_task_running(self):
        from backend.core import upload_worker
        upload_worker._reset()

        mock_task = MagicMock()
        mock_task.done.return_value = False
        upload_worker._worker_task = mock_task

        assert upload_worker.is_active() is True
        upload_worker._reset()

    def test_is_active_false_when_task_done(self):
        from backend.core import upload_worker
        upload_worker._reset()

        mock_task = MagicMock()
        mock_task.done.return_value = True
        upload_worker._worker_task = mock_task

        assert upload_worker.is_active() is False
        upload_worker._reset()


class TestStart:

    def test_start_skips_when_no_azure(self):
        from backend.core import upload_worker
        upload_worker._reset()

        with patch(f"{_MOD}.is_azure_configured", return_value=False):
            upload_worker.start()
            assert upload_worker._worker_task is None

    @pytest.mark.asyncio
    async def test_start_creates_task_when_azure_configured(self):
        from backend.core import upload_worker
        upload_worker._reset()

        with patch(f"{_MOD}.is_azure_configured", return_value=True):
            with patch(f"{_MOD}._upload_loop", new_callable=AsyncMock):
                upload_worker.start()
                assert upload_worker._worker_task is not None
                # Cancel the task to clean up
                upload_worker._worker_task.cancel()
                try:
                    await upload_worker._worker_task
                except asyncio.CancelledError:
                    pass

        upload_worker._reset()


class TestStop:

    @pytest.mark.asyncio
    async def test_stop_sets_shutdown_flag(self):
        from backend.core import upload_worker
        upload_worker._reset()

        await upload_worker.stop()
        assert upload_worker._shutdown is True
        upload_worker._reset()

    @pytest.mark.asyncio
    async def test_stop_cancels_running_task(self):
        from backend.core import upload_worker
        upload_worker._reset()

        # Create a real asyncio.Task that we can cancel
        async def _dummy_loop():
            while True:
                await asyncio.sleep(100)

        task = asyncio.create_task(_dummy_loop())
        upload_worker._worker_task = task

        await upload_worker.stop()

        assert task.cancelled() or task.done()
        upload_worker._reset()


class TestProcessPendingUploads:

    @pytest.mark.asyncio
    async def test_no_pending_jobs_returns_false(self):
        from backend.core.upload_worker import _process_pending_uploads

        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db.query.return_value.filter.return_value.all.return_value = []

        with patch(f"{_MOD}.get_db_session", return_value=mock_db):
            result = await _process_pending_uploads()
            assert result is False

    @pytest.mark.asyncio
    async def test_successful_upload_marks_completed(self, tmp_path):
        from backend.core.upload_worker import _process_pending_uploads
        from backend.models.enums import JobStatus

        entry_id = "ab123456-test-uuid"
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.status = JobStatus.PENDING_UPLOAD  # Must match the filter check
        mock_job.result_summary = {"entry_id": entry_id}
        mock_job.upload_attempts = 0

        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_job]

        mock_js = MagicMock()

        # Create real ZIP file on disk
        zip_dir = tmp_path / entry_id[:2]
        zip_dir.mkdir()
        zip_file = zip_dir / f"{entry_id}.zip"
        zip_file.write_bytes(b"fake zip content")

        with patch(f"{_MOD}.get_db_session", return_value=mock_db), \
             patch(f"{_MOD}.settings") as ms, \
             patch("backend.core.azure_sync._upload_with_retry") as upload_mock, \
             patch("backend.services.job_service.job_service", mock_js):

            ms.RESULTS_DIR = tmp_path

            result = await _process_pending_uploads()

            assert result is True
            mock_js.mark_completed.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_zip_triggers_permanent_failure(self, tmp_path):
        from backend.core.upload_worker import _process_pending_uploads
        from backend.models.enums import JobStatus

        entry_id = "ab123456-test-uuid"
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.status = JobStatus.PENDING_UPLOAD
        mock_job.result_summary = {"entry_id": entry_id}
        mock_job.upload_attempts = 0
        mock_job.requeue_count = 0

        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_job]

        # No ZIP file on disk -- will trigger permanent failure

        with patch(f"{_MOD}.get_db_session", return_value=mock_db), \
             patch(f"{_MOD}.settings") as ms, \
             patch(f"{_MOD}._handle_permanent_failure") as pf_mock:

            ms.RESULTS_DIR = tmp_path

            result = await _process_pending_uploads()

            assert result is False
            pf_mock.assert_called_once_with(mock_db, mock_job, "ZIP file missing or empty")

    @pytest.mark.asyncio
    async def test_upload_failure_increments_attempts(self, tmp_path):
        from backend.core.upload_worker import _process_pending_uploads
        from backend.models.enums import JobStatus

        entry_id = "ab123456-test-uuid"
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.status = JobStatus.PENDING_UPLOAD
        mock_job.result_summary = {"entry_id": entry_id}
        mock_job.upload_attempts = 5

        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_job]

        # Create real ZIP file
        zip_dir = tmp_path / entry_id[:2]
        zip_dir.mkdir()
        (zip_dir / f"{entry_id}.zip").write_bytes(b"fake")

        with patch(f"{_MOD}.get_db_session", return_value=mock_db), \
             patch(f"{_MOD}.settings") as ms, \
             patch("backend.core.azure_sync._upload_with_retry", side_effect=ConnectionError("fail")):

            ms.RESULTS_DIR = tmp_path

            result = await _process_pending_uploads()

            assert result is False
            assert mock_job.upload_attempts == 6
            mock_db.commit.assert_called()


class TestHandlePermanentFailure:

    def test_requeues_when_under_max_cycles(self):
        from backend.core.upload_worker import _handle_permanent_failure
        from backend.models.enums import JobStatus

        mock_db = MagicMock()
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.requeue_count = 0
        mock_job.result_summary = {"entry_id": "abc-123"}

        with patch(f"{_MOD}.compound_repo") as cr:
            _handle_permanent_failure(mock_db, mock_job, "ZIP missing")

        assert mock_job.status == JobStatus.PENDING
        assert mock_job.requeue_count == 1
        assert mock_job.upload_attempts == 0
        mock_db.commit.assert_called()

    def test_fails_permanently_when_max_cycles_reached(self):
        from backend.core.upload_worker import _handle_permanent_failure
        from backend.models.enums import JobStatus

        mock_db = MagicMock()
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.requeue_count = 3  # At max
        mock_job.result_summary = {"entry_id": "abc-123"}

        with patch(f"{_MOD}.compound_repo") as cr:
            _handle_permanent_failure(mock_db, mock_job, "ZIP missing")

        assert mock_job.status == JobStatus.FAILED
        assert "permanently" in mock_job.error_message
        mock_db.commit.assert_called()


class TestHandleExhaustion:

    def test_requeues_when_under_max_cycles(self):
        from backend.core.upload_worker import _handle_exhaustion
        from backend.models.enums import JobStatus

        mock_db = MagicMock()
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.requeue_count = 1
        mock_job.result_summary = {"entry_id": "abc-123"}

        with patch(f"{_MOD}.compound_repo") as cr:
            _handle_exhaustion(mock_db, mock_job)

        assert mock_job.status == JobStatus.PENDING
        assert mock_job.requeue_count == 2
        assert mock_job.upload_attempts == 0
        mock_db.commit.assert_called()

    def test_fails_permanently_at_max_cycles(self):
        from backend.core.upload_worker import _handle_exhaustion
        from backend.models.enums import JobStatus

        mock_db = MagicMock()
        mock_job = MagicMock()
        mock_job.id = uuid.uuid4()
        mock_job.requeue_count = 3  # At max
        mock_job.result_summary = {"entry_id": "abc-123"}

        with patch(f"{_MOD}.compound_repo") as cr:
            _handle_exhaustion(mock_db, mock_job)

        assert mock_job.status == JobStatus.FAILED
        assert "3 full processing cycles" in mock_job.error_message
        mock_db.commit.assert_called()


class TestUploadLoop:

    @pytest.mark.asyncio
    async def test_loop_exits_on_shutdown(self):
        """Test that _upload_loop exits when _shutdown is set."""
        from backend.core import upload_worker

        upload_worker._reset()
        upload_worker._shutdown = True  # Already shutdown

        # Should exit immediately without processing
        with patch(f"{_MOD}._process_pending_uploads", new_callable=AsyncMock) as mock_proc:
            await upload_worker._upload_loop()
            mock_proc.assert_not_called()

        upload_worker._reset()

    @pytest.mark.asyncio
    async def test_loop_resets_sleep_on_success(self):
        """Test outer sleep resets to INITIAL_OUTER_SLEEP on success."""
        from backend.core import upload_worker

        upload_worker._reset()

        call_count = 0

        async def _mock_process():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                upload_worker._shutdown = True
            return True  # Success

        with patch(f"{_MOD}._process_pending_uploads", side_effect=_mock_process):
            with patch(f"{_MOD}.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
                await upload_worker._upload_loop()

                # On success, outer_sleep stays at INITIAL_OUTER_SLEEP
                if sleep_mock.call_args_list:
                    sleep_val = sleep_mock.call_args_list[0][0][0]
                    assert sleep_val == upload_worker.INITIAL_OUTER_SLEEP

        upload_worker._reset()
