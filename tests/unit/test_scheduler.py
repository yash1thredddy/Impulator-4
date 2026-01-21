"""
Unit tests for the JobScheduler (SQLite-based job queue).
"""
import pytest
import time
import json
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone


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

        # Mock empty result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result
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
        return Session()

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


class TestRecoverStalledJobs:
    """Tests for _recover_stalled_jobs function in main.py."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch('backend.main.get_db_session') as mock:
            yield mock

    @pytest.fixture
    def mock_scheduler(self):
        """Mock job scheduler."""
        with patch('backend.main.job_scheduler') as mock:
            yield mock

    def test_recover_stalled_jobs_with_stalled(self, mock_db_session, mock_scheduler):
        """Test recovery of stalled PROCESSING jobs."""
        from backend.models.database import Job, JobStatus
        from unittest.mock import MagicMock

        # Create mock stalled job
        mock_job = MagicMock()
        mock_job.status = JobStatus.PROCESSING

        # Setup mock session
        mock_session = MagicMock()
        mock_query_stalled = MagicMock()
        mock_query_stalled.filter.return_value.all.return_value = [mock_job]
        mock_query_pending = MagicMock()
        mock_query_pending.filter.return_value.count.return_value = 0

        # Configure query to return different results based on filter
        def query_side_effect(model):
            query = MagicMock()
            call_count = [0]

            def filter_side_effect(*args):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call - stalled query
                    result = MagicMock()
                    result.all.return_value = [mock_job]
                    return result
                else:
                    # Second call - pending count
                    result = MagicMock()
                    result.count.return_value = 0
                    return result
            query.filter = filter_side_effect
            return query

        mock_session.query = query_side_effect
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_db_session.return_value = mock_session

        # Import and call
        from backend.main import _recover_stalled_jobs
        _recover_stalled_jobs()

        # Verify job status was updated
        assert mock_job.status == JobStatus.PENDING
        assert mock_job.current_step == "Queued (recovered)"

        # Verify scheduler was triggered
        mock_scheduler.trigger.assert_called_once()

    def test_recover_stalled_jobs_no_stalled(self, mock_db_session, mock_scheduler):
        """Test recovery when no stalled jobs."""
        mock_session = MagicMock()

        # Create separate mock for each query result
        mock_stalled_filter = MagicMock()
        mock_stalled_filter.all.return_value = []  # No stalled jobs

        mock_pending_filter = MagicMock()
        mock_pending_filter.count.return_value = 0  # No pending jobs

        call_count = [0]

        def query_side_effect(model):
            query = MagicMock()

            def filter_side_effect(*args):
                nonlocal call_count
                call_count[0] += 1
                if call_count[0] == 1:
                    return mock_stalled_filter
                else:
                    return mock_pending_filter
            query.filter = filter_side_effect
            return query

        mock_session.query = query_side_effect
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_db_session.return_value = mock_session

        from backend.main import _recover_stalled_jobs
        _recover_stalled_jobs()

        # Scheduler should NOT be triggered (no stalled, no pending)
        mock_scheduler.trigger.assert_not_called()

    def test_recover_stalled_jobs_pending_only(self, mock_db_session, mock_scheduler):
        """Test recovery triggers scheduler when pending jobs exist."""
        mock_session = MagicMock()

        # Create separate mock for each query result
        mock_stalled_filter = MagicMock()
        mock_stalled_filter.all.return_value = []  # No stalled jobs

        mock_pending_filter = MagicMock()
        mock_pending_filter.count.return_value = 5  # 5 pending jobs

        call_count = [0]

        def query_side_effect(model):
            query = MagicMock()

            def filter_side_effect(*args):
                nonlocal call_count
                call_count[0] += 1
                if call_count[0] == 1:
                    return mock_stalled_filter
                else:
                    return mock_pending_filter
            query.filter = filter_side_effect
            return query

        mock_session.query = query_side_effect
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_db_session.return_value = mock_session

        from backend.main import _recover_stalled_jobs
        _recover_stalled_jobs()

        # Scheduler SHOULD be triggered (pending jobs exist)
        mock_scheduler.trigger.assert_called_once()
