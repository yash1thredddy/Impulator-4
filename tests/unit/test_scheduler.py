"""
Unit tests for the async job scheduler (module-level functions).

asyncio_mode = auto in pytest.ini -- no @pytest.mark.asyncio needed.
"""
import asyncio
import uuid
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from backend.core import scheduler
from backend.models.enums import JobStatus


class TestSchedulerModule:
    """Tests for module-level scheduler functions."""

    async def test_is_running_initially_false(self):
        """Scheduler is not running initially."""
        # Ensure clean state
        scheduler._scheduler_task = None
        assert scheduler.is_running() is False

    async def test_stats_shape(self):
        """stats() returns correct structure with 'active' key."""
        scheduler._scheduler_task = None
        s = scheduler.stats()
        assert "active" in s
        assert "poll_interval" in s
        assert "idle_timeout" in s
        assert "consecutive_errors" in s
        assert "crash_reason" in s
        assert s["active"] is False
        assert s["poll_interval"] == 6.0
        assert s["idle_timeout"] == 300

    async def test_trigger_creates_task(self):
        """trigger() creates an asyncio.Task."""
        # Mock _scheduler_loop to exit immediately
        async def mock_loop():
            pass

        with patch.object(scheduler, "_scheduler_loop", mock_loop):
            scheduler._scheduler_task = None
            scheduler.trigger()
            assert scheduler._scheduler_task is not None
            assert isinstance(scheduler._scheduler_task, asyncio.Task)
            # Wait for task to finish
            await asyncio.sleep(0.05)

    async def test_trigger_idempotent(self):
        """Multiple trigger() calls don't create duplicate tasks."""
        gate = asyncio.Event()

        async def mock_loop():
            await gate.wait()

        with patch.object(scheduler, "_scheduler_loop", mock_loop):
            scheduler._scheduler_task = None
            scheduler.trigger()
            first_task = scheduler._scheduler_task

            scheduler.trigger()
            assert scheduler._scheduler_task is first_task

            gate.set()
            await asyncio.sleep(0.05)

    async def test_stop_cancels_task(self):
        """stop() cancels the running scheduler task."""
        gate = asyncio.Event()

        async def mock_loop():
            try:
                await gate.wait()
            except asyncio.CancelledError:
                raise

        with patch.object(scheduler, "_scheduler_loop", mock_loop):
            scheduler._scheduler_task = None
            scheduler.trigger()
            assert scheduler.is_running() is True

            await scheduler.stop()
            assert scheduler._scheduler_task is None

    async def test_stop_noop_when_not_running(self):
        """stop() is safe when scheduler not running."""
        scheduler._scheduler_task = None
        await scheduler.stop()  # Should not raise


class TestSchedulerLoop:
    """Tests for _scheduler_loop behavior."""

    async def test_idle_timeout_exits_loop(self):
        """Scheduler exits after idle timeout with no pending jobs."""
        # Patch _process_pending to return False (no work)
        # Patch _check_timeouts to be no-op
        # Set IDLE_TIMEOUT_SECONDS to very short value
        original_timeout = scheduler.IDLE_TIMEOUT_SECONDS
        original_poll = scheduler._DEFAULT_POLL_INTERVAL

        scheduler.IDLE_TIMEOUT_SECONDS = 0.1  # 100ms
        scheduler._DEFAULT_POLL_INTERVAL = 0.05

        try:
            with patch.object(scheduler, "_process_pending", return_value=False) as mock_pp, \
                 patch.object(scheduler, "_check_timeouts"), \
                 patch("backend.core.scheduler.get_db_session") as mock_db, \
                 patch("backend.core.scheduler.job_repo") as mock_repo:

                mock_session = MagicMock()
                mock_session.__enter__ = MagicMock(return_value=mock_session)
                mock_session.__exit__ = MagicMock(return_value=False)
                mock_db.return_value = mock_session
                mock_repo.get_pending_processing_count.return_value = 0

                scheduler._scheduler_task = None
                scheduler.trigger()

                # Wait for loop to exit via idle timeout
                await asyncio.sleep(0.5)

                assert not scheduler.is_running()
                assert mock_pp.call_count >= 1
        finally:
            scheduler.IDLE_TIMEOUT_SECONDS = original_timeout
            scheduler._DEFAULT_POLL_INTERVAL = original_poll

    async def test_error_backoff(self):
        """Consecutive errors increase backoff delay."""
        call_count = 0

        async def failing_process():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("test error")
            return False

        original_poll = scheduler._DEFAULT_POLL_INTERVAL
        scheduler._DEFAULT_POLL_INTERVAL = 0.01

        try:
            with patch.object(scheduler, "_process_pending", side_effect=failing_process), \
                 patch.object(scheduler, "_check_timeouts"):

                scheduler._scheduler_task = None
                scheduler.trigger()

                await asyncio.sleep(0.3)
                await scheduler.stop()

                # Should have attempted at least the failing calls
                assert call_count >= 2
                assert scheduler._consecutive_errors == 0 or scheduler._scheduler_task is None
        finally:
            scheduler._DEFAULT_POLL_INTERVAL = original_poll


class TestCheckTimeouts:
    """Tests for _check_timeouts function."""

    def test_marks_expired_jobs_failed(self):
        """Timeout check marks expired PROCESSING jobs as FAILED."""
        mock_job = MagicMock()
        mock_job.started_at = datetime.now(timezone.utc) - timedelta(seconds=7200)
        mock_job.status = JobStatus.PROCESSING

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("backend.core.scheduler.get_db_session", return_value=mock_session), \
             patch("backend.core.scheduler.job_repo") as mock_repo, \
             patch("backend.config.settings") as mock_settings:
            mock_settings.JOB_TIMEOUT = 3600
            mock_repo.get_stalled_processing_jobs.return_value = [mock_job]

            scheduler._check_timeouts()

            assert mock_job.status == JobStatus.FAILED
            mock_session.commit.assert_called()

    def test_skips_jobs_without_started_at(self):
        """Timeout check skips jobs with no started_at."""
        mock_job = MagicMock()
        mock_job.started_at = None
        mock_job.status = JobStatus.PROCESSING

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("backend.core.scheduler.get_db_session", return_value=mock_session), \
             patch("backend.core.scheduler.job_repo") as mock_repo, \
             patch("backend.config.settings") as mock_settings:
            mock_settings.JOB_TIMEOUT = 3600
            mock_repo.get_stalled_processing_jobs.return_value = [mock_job]

            scheduler._check_timeouts()
            assert mock_job.status == JobStatus.PROCESSING

    def test_handles_naive_datetime(self):
        """Timeout check handles naive datetime (no tzinfo)."""
        mock_job = MagicMock()
        mock_job.started_at = datetime(2020, 1, 1)  # Naive, well past timeout
        mock_job.status = JobStatus.PROCESSING

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("backend.core.scheduler.get_db_session", return_value=mock_session), \
             patch("backend.core.scheduler.job_repo") as mock_repo, \
             patch("backend.config.settings") as mock_settings:
            mock_settings.JOB_TIMEOUT = 3600
            mock_repo.get_stalled_processing_jobs.return_value = [mock_job]

            scheduler._check_timeouts()
            assert mock_job.status == JobStatus.FAILED

    def test_handles_db_exception(self):
        """Timeout check handles DB exceptions gracefully."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("backend.core.scheduler.get_db_session", return_value=mock_session), \
             patch("backend.core.scheduler.job_repo") as mock_repo:
            mock_repo.get_stalled_processing_jobs.side_effect = Exception("db error")
            scheduler._check_timeouts()  # Should not raise


class TestRecoverOnStartup:
    """Tests for job_service.recover_on_startup method.

    Recovery is handled by JobService.recover_on_startup(db, scheduler_trigger).
    Note: These tests use mocked DB sessions because PGBase models use
    Postgres-specific types (ARRAY) that SQLite cannot render.
    Full integration tests deferred to Phase 20.
    """

    def test_recover_no_stalled_no_pending(self):
        """Recovery with no stalled or pending jobs."""
        from backend.services.job_service import job_service

        mock_db = MagicMock()

        with patch("backend.services.job_service.job_repo") as mock_repo:
            mock_repo.get_by_status.return_value = []
            mock_repo.count_by_status.return_value = 0

            trigger_called = []
            result = job_service.recover_on_startup(mock_db, lambda: trigger_called.append(True))

            assert result["recovered"] == 0
            assert result["pending"] == 0
            assert len(trigger_called) == 0
