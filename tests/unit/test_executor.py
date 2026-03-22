"""
Unit tests for async executor (asyncio.Task tracker).

asyncio_mode = auto in pytest.ini -- no @pytest.mark.asyncio needed.
"""
import asyncio
from unittest.mock import patch

from backend.core import executor


class TestAsyncExecutor:
    """Tests for module-level async executor functions."""

    def setup_method(self):
        """Reset executor state before each test."""
        executor._reset()

    def teardown_method(self):
        """Clean up after each test."""
        executor._reset()

    async def test_submit_returns_job_id(self):
        """Test submit() returns the job_id."""
        async def dummy(job_id, **kwargs):
            pass

        result = await executor.submit("test-1", dummy)
        assert result == "test-1"
        # Wait for task to complete
        await asyncio.sleep(0.05)

    async def test_submit_duplicate_returns_without_new_task(self):
        """Test submitting same job_id twice doesn't create duplicate task."""
        gate = asyncio.Event()

        async def blocking(job_id, **kwargs):
            await gate.wait()

        await executor.submit("dup-1", blocking)
        assert executor.get_active_count() == 1

        result = await executor.submit("dup-1", blocking)
        assert result == "dup-1"
        assert executor.get_active_count() == 1

        gate.set()
        await asyncio.sleep(0.05)

    async def test_has_capacity_initial(self):
        """Test capacity check when no jobs running."""
        assert executor.has_capacity() is True

    async def test_get_active_count_initial(self):
        """Test active count is zero initially."""
        assert executor.get_active_count() == 0

    async def test_get_active_job_ids(self):
        """Test get_active_job_ids returns correct list."""
        gate = asyncio.Event()

        async def blocking(job_id, **kwargs):
            await gate.wait()

        await executor.submit("job-a", blocking)
        await executor.submit("job-b", blocking)

        ids = executor.get_active_job_ids()
        assert "job-a" in ids
        assert "job-b" in ids

        gate.set()
        await asyncio.sleep(0.05)

    async def test_is_active(self):
        """Test is_active for running and non-existent jobs."""
        gate = asyncio.Event()

        async def blocking(job_id, **kwargs):
            await gate.wait()

        await executor.submit("active-1", blocking)
        assert executor.is_active("active-1") is True
        assert executor.is_active("nonexistent") is False

        gate.set()
        await asyncio.sleep(0.05)

    async def test_cancel_running_task(self):
        """Test cancel sends CancelledError to running task."""
        cancelled = asyncio.Event()

        async def cancellable(job_id, **kwargs):
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        await executor.submit("cancel-me", cancellable)
        await asyncio.sleep(0.01)  # Let task start

        result = executor.cancel("cancel-me")
        assert result is True
        await asyncio.sleep(0.1)
        assert cancelled.is_set()

    async def test_cancel_nonexistent_returns_false(self):
        """Test cancelling non-existent job returns False."""
        assert executor.cancel("no-such-job") is False

    async def test_stats_shape(self):
        """Test stats() returns correct structure."""
        with patch.object(executor, "settings") as mock_settings:
            mock_settings.MAX_CONCURRENT_JOBS = 3
            s = executor.stats()

        assert "max_concurrent_jobs" in s
        assert "active_jobs" in s
        assert "slots_available" in s
        assert "has_capacity" in s
        assert "jobs" in s
        assert s["active_jobs"] == 0
        assert s["has_capacity"] is True

    async def test_shutdown_cancels_all_tasks(self):
        """Test shutdown cancels all active tasks."""
        gate = asyncio.Event()

        async def blocking(job_id, **kwargs):
            await gate.wait()

        await executor.submit("shut-1", blocking)
        await executor.submit("shut-2", blocking)
        assert executor.get_active_count() == 2

        await executor.shutdown(timeout=2.0)
        assert executor.get_active_count() == 0

    async def test_shutdown_noop_when_empty(self):
        """Test shutdown does nothing when no tasks exist."""
        await executor.shutdown(timeout=1.0)
        assert executor.get_active_count() == 0

    async def test_task_removed_after_completion(self):
        """Test completed task is removed from tracking dict."""
        async def fast(job_id, **kwargs):
            pass

        await executor.submit("fast-1", fast)
        await asyncio.sleep(0.1)
        assert executor.get_active_count() == 0
        assert "fast-1" not in executor._tasks

    async def test_task_error_logged_and_removed(self):
        """Test task that raises exception is cleaned up."""
        async def failing(job_id, **kwargs):
            raise ValueError("test error")

        await executor.submit("fail-1", failing)
        await asyncio.sleep(0.1)
        assert executor.get_active_count() == 0

    async def test_semaphore_limits_concurrency(self):
        """Test semaphore limits concurrent execution."""
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()
        gate = asyncio.Event()

        async def tracked(job_id, **kwargs):
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await gate.wait()
            async with lock:
                current -= 1

        with patch.object(executor, "settings") as mock_settings:
            mock_settings.MAX_CONCURRENT_JOBS = 2
            executor._reset()  # Reset semaphore with new limit

            # Submit 4 tasks, only 2 should run concurrently
            for i in range(4):
                await executor.submit(f"sem-{i}", tracked)

            await asyncio.sleep(0.1)
            assert max_concurrent <= 2

            gate.set()
            await asyncio.sleep(0.2)


class TestExecutorCapacity:
    """Tests for capacity tracking."""

    def setup_method(self):
        executor._reset()

    def teardown_method(self):
        executor._reset()

    async def test_has_capacity_respects_limit(self):
        """Test has_capacity returns False when at limit."""
        gate = asyncio.Event()

        async def blocking(job_id, **kwargs):
            await gate.wait()

        with patch.object(executor, "settings") as mock_settings:
            mock_settings.MAX_CONCURRENT_JOBS = 2

            await executor.submit("cap-1", blocking)
            await executor.submit("cap-2", blocking)

            assert executor.has_capacity() is False

            gate.set()
            await asyncio.sleep(0.1)
