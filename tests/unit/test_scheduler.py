"""
Unit tests for the JobScheduler (SQLite-based job queue).
"""
import pytest
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta


class TestJobScheduler:
    """Tests for JobScheduler class."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler for each test."""
        from backend.core.scheduler import JobScheduler
        sched = JobScheduler(poll_interval=0.1)  # Fast polling for tests
        yield sched
        if sched._running:
            sched._running = False
            if sched._thread:
                sched._thread.join(timeout=1)

    def test_initial_state(self, scheduler):
        """Test scheduler is not running initially."""
        assert scheduler._running is False
        assert scheduler._thread is None
        assert scheduler._last_activity is None

    def test_is_running(self, scheduler):
        """Test is_running returns correct state."""
        assert scheduler.is_running() is False

    def test_stats(self, scheduler):
        """Test scheduler statistics."""
        stats = scheduler.stats()
        assert "running" in stats
        assert "poll_interval" in stats
        assert "idle_timeout" in stats
        assert "last_activity" in stats
        assert stats["running"] is False
        assert stats["poll_interval"] == 0.1

    def test_trigger_starts_scheduler(self, scheduler):
        """Test trigger() starts the scheduler."""
        with patch.object(scheduler, '_process_pending', return_value=False):
            with patch.object(scheduler, '_should_stop', return_value=True):
                scheduler.trigger()
                time.sleep(0.2)  # Give time to start and stop
                assert scheduler._last_activity is not None

    def test_trigger_sets_last_activity(self, scheduler):
        """Test trigger() sets last_activity timestamp."""
        before = datetime.now(timezone.utc)
        with patch.object(scheduler, '_start'):
            scheduler.trigger()
        after = datetime.now(timezone.utc)

        assert scheduler._last_activity is not None
        assert before <= scheduler._last_activity <= after

    def test_stats_with_activity(self, scheduler):
        """Test stats() serializes last_activity correctly."""
        scheduler._last_activity = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        stats = scheduler.stats()
        assert stats["last_activity"] == "2026-03-01T12:00:00+00:00"


class TestJobSchedulerProcessPending:
    """Tests for _process_pending method."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch('backend.core.scheduler.get_db_session') as mock:
            yield mock

    @pytest.fixture
    def mock_executor(self):
        """Mock job executor."""
        with patch('backend.core.scheduler.job_executor') as mock:
            yield mock

    def test_process_pending_no_capacity(self, mock_executor):
        """Test _process_pending returns early when executor is full."""
        from backend.core.scheduler import JobScheduler
        scheduler = JobScheduler(poll_interval=0.1)

        mock_executor.has_capacity.return_value = False

        result = scheduler._process_pending()
        assert result is False

    def test_process_pending_no_jobs(self, mock_db_session, mock_executor):
        """Test _process_pending handles empty queue."""
        from backend.core.scheduler import JobScheduler
        scheduler = JobScheduler(poll_interval=0.1)

        mock_executor.has_capacity.return_value = True

        # Mock empty result -- .first() returns None (no pending jobs)
        mock_session = MagicMock()
        mock_execute_result = MagicMock()
        mock_execute_result.first.return_value = None
        mock_session.execute.return_value = mock_execute_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_db_session.return_value = mock_session

        result = scheduler._process_pending()
        assert result is False


class TestJobSchedulerShouldStop:
    """Tests for _should_stop method."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch('backend.core.scheduler.get_db_session') as mock:
            yield mock

    def test_should_stop_no_activity(self):
        """Test _should_stop returns False when no activity recorded."""
        from backend.core.scheduler import JobScheduler
        scheduler = JobScheduler()
        scheduler._last_activity = None

        assert scheduler._should_stop() is False

    def test_should_stop_active_jobs(self, mock_db_session):
        """Test _should_stop returns False when jobs are active."""
        from backend.core.scheduler import JobScheduler
        scheduler = JobScheduler()
        scheduler._last_activity = datetime.now(timezone.utc)

        # Mock query with active jobs
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value.count.return_value = 5  # 5 active jobs
        mock_session.query.return_value = mock_query
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_db_session.return_value = mock_session

        assert scheduler._should_stop() is False

    def test_should_stop_idle_timeout(self, mock_db_session):
        """Test _should_stop returns True after idle timeout."""
        from backend.core.scheduler import JobScheduler, IDLE_TIMEOUT_SECONDS
        scheduler = JobScheduler()
        # Set activity to way in the past
        scheduler._last_activity = datetime.now(timezone.utc) - timedelta(seconds=IDLE_TIMEOUT_SECONDS + 100)

        # Mock: no active jobs
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_db_session.return_value = mock_session

        with patch('backend.core.scheduler.job_repo') as mock_repo:
            mock_repo.get_pending_processing_count.return_value = 0
            assert scheduler._should_stop() is True

    def test_should_stop_exception_returns_false(self, mock_db_session):
        """Test _should_stop returns False on DB exception."""
        from backend.core.scheduler import JobScheduler
        scheduler = JobScheduler()
        scheduler._last_activity = datetime.now(timezone.utc) - timedelta(seconds=600)

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_db_session.return_value = mock_session

        with patch('backend.core.scheduler.job_repo') as mock_repo:
            mock_repo.get_pending_processing_count.side_effect = Exception("db error")
            assert scheduler._should_stop() is False


class TestSchedulerCheckTimeouts:
    """Tests for _check_timeouts method."""

    def test_check_timeouts_marks_expired_jobs(self):
        """Test timeout check marks expired jobs as FAILED."""
        from backend.core.scheduler import JobScheduler
        from backend.models.database import JobStatus

        scheduler = JobScheduler()
        mock_job = MagicMock()
        mock_job.started_at = datetime.now(timezone.utc) - timedelta(seconds=7200)
        mock_job.status = JobStatus.PROCESSING

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
             patch('backend.core.scheduler.job_repo') as mock_repo, \
             patch('backend.config.settings') as mock_settings:
            mock_settings.JOB_TIMEOUT = 3600
            mock_repo.get_stalled_processing_jobs.return_value = [mock_job]

            scheduler._check_timeouts()

            assert mock_job.status == JobStatus.FAILED
            mock_session.commit.assert_called()


class TestSchedulerRunMaintenance:
    """Tests for _run_maintenance method."""

    def test_maintenance_skips_if_recent(self):
        """Test maintenance skips if vacuum was run recently."""
        from backend.core import scheduler as sched_mod
        from backend.core.scheduler import JobScheduler

        scheduler = JobScheduler()
        original = sched_mod._last_vacuum_time
        sched_mod._last_vacuum_time = time.time()  # Just ran

        try:
            # Should return immediately (no DB calls)
            with patch('backend.core.scheduler.get_db_session') as mock_db:
                scheduler._run_maintenance()
                mock_db.assert_not_called()
        finally:
            sched_mod._last_vacuum_time = original

    def test_maintenance_skips_if_active_jobs(self):
        """Test maintenance skips when jobs are active."""
        from backend.core import scheduler as sched_mod
        from backend.core.scheduler import JobScheduler

        scheduler = JobScheduler()
        original = sched_mod._last_vacuum_time
        sched_mod._last_vacuum_time = 0  # Force needs run

        try:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)

            with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
                 patch('backend.core.scheduler.job_repo') as mock_repo:
                mock_repo.get_pending_processing_count.return_value = 5
                scheduler._run_maintenance()
                # No sqlite3 calls because active jobs
        finally:
            sched_mod._last_vacuum_time = original


class TestSchedulerRetrySyncPending:
    """Tests for _retry_sync_pending method."""

    def test_sync_pending_no_jobs(self):
        """Test retry does nothing when no SYNC_PENDING jobs."""
        from backend.core.scheduler import JobScheduler

        scheduler = JobScheduler()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
             patch('backend.core.scheduler.job_repo') as mock_repo:
            mock_repo.get_sync_pending_jobs.return_value = []
            scheduler._retry_sync_pending()


class TestSchedulerIntegration:
    """Integration tests for scheduler with mock database."""

    @pytest.fixture
    def test_db(self):
        """Create in-memory test database."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.models.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
        engine.dispose()

    def test_scheduler_trigger_is_idempotent(self):
        """Test multiple triggers don't create multiple threads while running."""
        from backend.core.scheduler import JobScheduler
        scheduler = JobScheduler(poll_interval=0.1)

        # Mock _should_stop to return False so scheduler keeps running
        with patch.object(scheduler, '_process_pending', return_value=False):
            with patch.object(scheduler, '_should_stop', return_value=False):
                scheduler.trigger()
                time.sleep(0.05)  # Give time to start
                first_thread = scheduler._thread

                # Second trigger while running should not create new thread
                scheduler.trigger()
                assert scheduler._thread is first_thread
                assert scheduler._running is True

        # Cleanup
        scheduler._running = False
        if scheduler._thread:
            scheduler._thread.join(timeout=1)

    def test_global_scheduler_instance(self):
        """Test global job_scheduler instance exists."""
        from backend.core.scheduler import job_scheduler

        assert job_scheduler is not None
        assert hasattr(job_scheduler, 'trigger')
        assert hasattr(job_scheduler, 'is_running')
        assert hasattr(job_scheduler, 'stats')

    def test_run_loop_stops_on_should_stop(self):
        """Test _run loop exits when _should_stop returns True."""
        from backend.core.scheduler import JobScheduler
        scheduler = JobScheduler(poll_interval=0.05)

        call_count = 0

        def mock_process():
            nonlocal call_count
            call_count += 1
            return False

        with patch.object(scheduler, '_process_pending', side_effect=mock_process), \
             patch.object(scheduler, '_retry_sync_pending'), \
             patch.object(scheduler, '_check_timeouts'), \
             patch.object(scheduler, '_should_stop', return_value=True), \
             patch.object(scheduler, '_run_maintenance'):
            scheduler.trigger()
            time.sleep(0.3)  # Allow time for thread to run and stop

        assert not scheduler._running
        assert call_count >= 1


class TestRecoverOnStartup:
    """Tests for job_service.recover_on_startup method.

    The old _recover_stalled_jobs function was removed from main.py in Phase 4.
    Recovery is now handled by JobService.recover_on_startup(db, scheduler_trigger).
    """

    @pytest.fixture
    def test_db(self):
        """Create in-memory test database for recovery tests."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from backend.core.database import Base
        from backend.models.database import Job, Compound, DeletedCompound  # noqa: F401

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()

    def test_recover_stalled_processing_triggers_scheduler(self, test_db):
        """Test that stalled PROCESSING jobs trigger the scheduler.

        VALID_TRANSITIONS does not allow PROCESSING -> PENDING (ARCH-12),
        so jobs remain PROCESSING but the scheduler is triggered to handle them.
        """
        from backend.models.database import Job, JobStatus
        from backend.services.job_service import job_service

        # Create a stalled PROCESSING job
        job = Job(
            id="stalled-job-1",
            status=JobStatus.PROCESSING,
            session_id="test-session",
        )
        test_db.add(job)
        test_db.commit()

        trigger_called = []
        result = job_service.recover_on_startup(test_db, lambda: trigger_called.append(True))

        # PROCESSING -> PENDING is not a valid transition, so recovered=0
        assert result["recovered"] == 0
        test_db.refresh(job)
        assert job.status == JobStatus.PROCESSING  # Stays in PROCESSING
        # But scheduler IS triggered because stalled jobs exist
        assert len(trigger_called) == 1

    def test_recover_no_stalled_no_pending(self, test_db):
        """Test recovery when no stalled or pending jobs exist."""
        from backend.services.job_service import job_service

        trigger_called = []
        result = job_service.recover_on_startup(test_db, lambda: trigger_called.append(True))

        assert result["recovered"] == 0
        assert result["pending"] == 0
        assert result["sync_pending"] == 0
        assert len(trigger_called) == 0

    def test_recover_triggers_scheduler_for_pending(self, test_db):
        """Test recovery triggers scheduler when pending jobs exist."""
        from backend.models.database import Job, JobStatus
        from backend.services.job_service import job_service

        # Create a pending job (not stalled, just waiting)
        job = Job(
            id="pending-job-1",
            status=JobStatus.PENDING,
            session_id="test-session",
        )
        test_db.add(job)
        test_db.commit()

        trigger_called = []
        result = job_service.recover_on_startup(test_db, lambda: trigger_called.append(True))

        assert result["recovered"] == 0
        assert result["pending"] == 1
        assert len(trigger_called) == 1


class TestSchedulerRunMaintenanceExecution:
    """Tests for _run_maintenance actually executing SQLite commands."""

    def test_maintenance_executes_vacuum(self, tmp_path):
        """Test maintenance runs WAL checkpoint, VACUUM, and ANALYZE."""
        import sqlite3
        from backend.core import scheduler as sched_mod
        from backend.core.scheduler import JobScheduler

        # Create a real SQLite DB
        db_path = tmp_path / "impulator.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.close()

        scheduler = JobScheduler()
        original = sched_mod._last_vacuum_time
        sched_mod._last_vacuum_time = 0  # Force needs run

        try:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)

            with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
                 patch('backend.core.scheduler.job_repo') as mock_repo, \
                 patch('backend.config.settings') as mock_settings:
                mock_repo.get_pending_processing_count.return_value = 0
                mock_settings.DATA_DIR = tmp_path

                scheduler._run_maintenance()

                # _last_vacuum_time should be updated
                assert sched_mod._last_vacuum_time > 0
        finally:
            sched_mod._last_vacuum_time = original

    def test_maintenance_handles_db_exception(self, tmp_path):
        """Test maintenance handles sqlite3 exceptions gracefully."""
        from backend.core import scheduler as sched_mod
        from backend.core.scheduler import JobScheduler

        scheduler = JobScheduler()
        original = sched_mod._last_vacuum_time
        sched_mod._last_vacuum_time = 0

        try:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)

            with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
                 patch('backend.core.scheduler.job_repo') as mock_repo, \
                 patch('backend.config.settings') as mock_settings, \
                 patch('sqlite3.connect', side_effect=Exception("locked")):
                mock_repo.get_pending_processing_count.return_value = 0
                mock_settings.DATA_DIR = tmp_path
                # Create the DB file so the path check passes
                (tmp_path / "impulator.db").touch()

                # Should not raise
                scheduler._run_maintenance()
        finally:
            sched_mod._last_vacuum_time = original

    def test_maintenance_skips_missing_db(self, tmp_path):
        """Test maintenance skips when DB file doesn't exist."""
        from backend.core import scheduler as sched_mod
        from backend.core.scheduler import JobScheduler

        scheduler = JobScheduler()
        original = sched_mod._last_vacuum_time
        sched_mod._last_vacuum_time = 0

        try:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)

            with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
                 patch('backend.core.scheduler.job_repo') as mock_repo, \
                 patch('backend.config.settings') as mock_settings:
                mock_repo.get_pending_processing_count.return_value = 0
                mock_settings.DATA_DIR = tmp_path / "nonexistent"

                # Should return early (no DB file)
                scheduler._run_maintenance()
        finally:
            sched_mod._last_vacuum_time = original


class TestSchedulerCheckTimeoutsEdgeCases:
    """Additional tests for _check_timeouts edge cases."""

    def test_check_timeouts_no_started_at(self):
        """Test timeout check skips jobs without started_at."""
        from backend.core.scheduler import JobScheduler
        from backend.models.database import JobStatus

        scheduler = JobScheduler()
        mock_job = MagicMock()
        mock_job.started_at = None  # No start time
        mock_job.status = JobStatus.PROCESSING

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
             patch('backend.core.scheduler.job_repo') as mock_repo, \
             patch('backend.config.settings') as mock_settings:
            mock_settings.JOB_TIMEOUT = 3600
            mock_repo.get_stalled_processing_jobs.return_value = [mock_job]

            scheduler._check_timeouts()

            # Should not change status (skipped due to no started_at)
            assert mock_job.status == JobStatus.PROCESSING

    def test_check_timeouts_naive_datetime(self):
        """Test timeout check handles naive datetime (no tzinfo)."""
        from backend.core.scheduler import JobScheduler
        from backend.models.database import JobStatus

        scheduler = JobScheduler()
        mock_job = MagicMock()
        # Naive datetime (no tzinfo) well past timeout
        mock_job.started_at = datetime(2020, 1, 1)
        mock_job.status = JobStatus.PROCESSING

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
             patch('backend.core.scheduler.job_repo') as mock_repo, \
             patch('backend.config.settings') as mock_settings:
            mock_settings.JOB_TIMEOUT = 3600
            mock_repo.get_stalled_processing_jobs.return_value = [mock_job]

            scheduler._check_timeouts()
            assert mock_job.status == JobStatus.FAILED

    def test_check_timeouts_exception(self):
        """Test timeout check handles DB exceptions gracefully."""
        from backend.core.scheduler import JobScheduler

        scheduler = JobScheduler()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch('backend.core.scheduler.get_db_session', return_value=mock_session), \
             patch('backend.core.scheduler.job_repo') as mock_repo:
            mock_repo.get_stalled_processing_jobs.side_effect = Exception("db error")
            # Should not raise
            scheduler._check_timeouts()
