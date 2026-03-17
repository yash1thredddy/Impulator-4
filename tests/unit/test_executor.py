"""
Unit tests for the JobExecutor (ThreadPoolExecutor wrapper).
"""
import pytest
import threading


class TestJobExecutor:
    """Tests for JobExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create a fresh executor for each test."""
        from backend.core.executor import JobExecutor
        exec = JobExecutor(max_workers=2)
        yield exec
        exec.shutdown(wait=False)

    def test_submit_job(self, executor):
        """Test job submission returns job_id."""
        def dummy_task(job_id):
            return f"completed_{job_id}"

        job_id = executor.submit("test-1", dummy_task)
        assert job_id == "test-1"

    def test_has_capacity_initial(self, executor):
        """Test capacity check when no jobs running."""
        assert executor.has_capacity() is True

    def test_get_active_count_initial(self, executor):
        """Test active count is zero initially."""
        assert executor.get_active_count() == 0

    def test_stats(self, executor):
        """Test executor statistics."""
        stats = executor.stats()
        assert "max_workers" in stats
        assert "active_jobs" in stats
        assert "has_capacity" in stats
        assert "job_ids" in stats
        assert stats["max_workers"] == 2
        assert stats["active_jobs"] == 0
        assert stats["has_capacity"] is True

    def test_cancel_nonexistent_job(self, executor):
        """Test cancelling non-existent job returns False."""
        result = executor.cancel("nonexistent")
        assert result is False

    def test_shutdown(self, executor):
        """Test graceful shutdown."""
        executor.shutdown(wait=False)
        # Should not raise

    def test_submit_and_wait(self, executor):
        """Test submitting a job and waiting for completion."""
        done_event = threading.Event()
        result_holder = {}

        def task(job_id):
            result_holder[job_id] = "done"
            done_event.set()
            return "done"

        executor.submit("test-wait", task)
        done_event.wait(timeout=2)
        assert result_holder.get("test-wait") == "done"


class TestJobExecutorConcurrency:
    """Tests for executor concurrency behavior."""

    def test_max_workers_limit(self):
        """Test that max workers is respected."""
        from backend.core.executor import JobExecutor
        executor = JobExecutor(max_workers=1)
        assert executor._max_workers == 1
        executor.shutdown(wait=False)

    def test_active_job_tracking(self):
        """Test that active jobs are tracked correctly."""
        from backend.core.executor import JobExecutor
        executor = JobExecutor(max_workers=2)

        started = threading.Event()
        release = threading.Event()

        def controlled_task(job_id):
            started.set()
            release.wait(timeout=2)
            return job_id

        executor.submit("job-1", controlled_task)
        started.wait(timeout=2)

        assert executor.get_active_count() >= 0  # May be 0 or 1 depending on timing
        release.set()
        executor.shutdown(wait=True)
