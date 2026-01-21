"""
ThreadPoolExecutor wrapper for background job execution.
Manages concurrent job processing with configurable worker limit.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Dict, Optional, Any

from backend.config import settings

logger = logging.getLogger(__name__)


class JobExecutor:
    """
    Manages background job execution using ThreadPoolExecutor.

    Features:
    - Configurable max workers (default: 2)
    - Job tracking by ID (thread-safe)
    - Graceful cancellation support
    - Active job count monitoring

    Thread Safety:
    - All access to _futures is protected by _lock
    - Done callbacks run in worker threads but use lock
    """

    def __init__(self, max_workers: int = None):
        """
        Initialize executor with worker pool.

        Args:
            max_workers: Maximum concurrent jobs. Defaults to settings.MAX_WORKERS
        """
        self._max_workers = max_workers or settings.MAX_WORKERS
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="job_worker"
        )
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()  # Protects _futures access
        logger.info(f"JobExecutor initialized with {self._max_workers} workers")

    def submit(
        self,
        job_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Submit a job for background execution.

        Args:
            job_id: Unique job identifier
            func: Function to execute (should accept job_id as first arg)
            *args: Additional positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            The job_id for tracking
        """
        with self._lock:
            if job_id in self._futures:
                logger.warning(f"Job {job_id} already exists, skipping")
                return job_id

            future = self._executor.submit(func, job_id, *args, **kwargs)
            self._futures[job_id] = future

        # Auto-cleanup on completion (callback runs in worker thread)
        def cleanup(f: Future):
            with self._lock:
                self._futures.pop(job_id, None)
            if f.exception():
                logger.error(f"Job {job_id} raised exception: {f.exception()}")
            else:
                # Note: Job may have failed internally but handled its own exception
                # The actual job status is in the database, not the Future state
                logger.info(f"Job {job_id} finished execution")

        future.add_done_callback(cleanup)
        logger.info(f"Job {job_id} submitted to executor")
        return job_id

    def get_active_count(self) -> int:
        """Get number of currently running/pending jobs (thread-safe)."""
        with self._lock:
            return len(self._futures)

    def get_active_job_ids(self) -> list:
        """Get list of active job IDs (thread-safe snapshot)."""
        with self._lock:
            return list(self._futures.keys())

    def is_active(self, job_id: str) -> bool:
        """Check if a specific job is still running (thread-safe)."""
        with self._lock:
            future = self._futures.get(job_id)
            return future is not None and not future.done()

    def cancel(self, job_id: str) -> bool:
        """
        Attempt to cancel a job (thread-safe).

        Args:
            job_id: Job to cancel

        Returns:
            True if cancelled, False if already running/completed
        """
        with self._lock:
            future = self._futures.get(job_id)
            if future is None:
                logger.warning(f"Job {job_id} not found")
                return False

            if future.done():
                logger.info(f"Job {job_id} already completed")
                return False

            cancelled = future.cancel()
            if cancelled:
                self._futures.pop(job_id, None)
                logger.info(f"Job {job_id} cancelled")
            else:
                logger.warning(f"Job {job_id} could not be cancelled (already running)")

            return cancelled

    def has_capacity(self) -> bool:
        """Check if executor can accept more jobs (thread-safe).

        Only allows max_workers jobs at a time.
        SQLite is our queue now - don't queue in executor's internal queue.
        """
        with self._lock:
            return len(self._futures) < self._max_workers

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """
        Get queue position for a job (0 = running, >0 = waiting).
        Returns None if job not found (thread-safe snapshot).
        """
        with self._lock:
            if job_id not in self._futures:
                return None
            job_ids = list(self._futures.keys())
            return job_ids.index(job_id)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """
        Shutdown the executor gracefully.

        Args:
            wait: Wait for pending jobs to complete
            cancel_futures: Cancel pending (not running) jobs
        """
        logger.info(f"Shutting down executor (wait={wait}, cancel={cancel_futures})")
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        with self._lock:
            self._futures.clear()

    def stats(self) -> dict:
        """Get executor statistics (thread-safe snapshot)."""
        with self._lock:
            return {
                "max_workers": self._max_workers,
                "active_jobs": len(self._futures),
                "has_capacity": len(self._futures) < self._max_workers,
                "job_ids": list(self._futures.keys()),
            }


# Global executor instance
job_executor = JobExecutor()
