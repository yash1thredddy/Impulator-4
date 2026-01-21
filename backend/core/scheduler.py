"""
Event-driven job scheduler - only runs when jobs exist.

Uses SQLite as a persistent job queue. Scheduler polls for PENDING jobs
and submits them to the ThreadPoolExecutor.

Features:
- Event-driven: starts on job submit, stops after idle timeout
- Atomic job claiming: prevents race conditions with row-level locking
- 2 workers max: only 2 jobs in executor, rest stay in SQLite
- Recovery: handles stalled PROCESSING jobs on startup
"""
import json
import threading
import time
import logging
from datetime import datetime, timezone

from sqlalchemy.exc import OperationalError

from backend.core.database import get_db_session
from backend.core.executor import job_executor
from backend.models.database import Job, JobStatus

logger = logging.getLogger(__name__)

# Idle timeout: stop polling 5 min after last job completes
IDLE_TIMEOUT_SECONDS = 300


class JobScheduler:
    """Event-driven scheduler that polls only when jobs exist."""

    def __init__(self, poll_interval: float = 6.0):
        """Initialize scheduler.

        Args:
            poll_interval: Seconds between polls (default 6 sec)
        """
        self._poll_interval = poll_interval
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_activity = None  # Track last job activity

    def trigger(self):
        """Called when jobs are submitted - starts scheduler if not running."""
        with self._lock:
            self._last_activity = datetime.now(timezone.utc)
            if not self._running:
                self._start()

    def _start(self):
        """Start scheduler in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Job scheduler started")

    def _stop(self):
        """Stop scheduler."""
        self._running = False
        logger.info("Job scheduler stopped (idle timeout)")

    def _run(self):
        """Main scheduler loop - stops after idle timeout."""
        # Process immediately on start (no initial delay)
        first_run = True

        while self._running:
            try:
                if not first_run:
                    time.sleep(self._poll_interval)
                first_run = False

                logger.info(f"Scheduler poll: executor capacity={job_executor.has_capacity()}, active={job_executor.get_active_count()}")

                had_work = self._process_pending()

                if had_work:
                    self._last_activity = datetime.now(timezone.utc)

                # Check idle timeout
                if self._should_stop():
                    self._stop()
                    break

            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)

    def _should_stop(self) -> bool:
        """Check if scheduler should stop due to idle timeout."""
        if not self._last_activity:
            return False

        # Check if any jobs are still pending/processing
        try:
            with get_db_session() as db:
                active_count = (
                    db.query(Job)
                    .filter(Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]))
                    .count()
                )
                if active_count > 0:
                    return False
        except Exception as e:
            logger.error(f"Error checking active jobs: {e}")
            return False

        # No active jobs - check timeout
        elapsed = (datetime.now(timezone.utc) - self._last_activity).total_seconds()
        return elapsed >= IDLE_TIMEOUT_SECONDS

    def _process_pending(self) -> bool:
        """Check for pending jobs and submit to executor.

        Submits jobs until executor is full (2 workers).
        Returns True if any work was done.

        Uses ORM with row-level locking (with_for_update) to prevent
        race conditions where multiple threads claim the same job.
        """
        work_done = False

        # Keep submitting until executor is full or no more pending jobs
        while job_executor.has_capacity():
            try:
                with get_db_session() as db:
                    # Find oldest pending job with row-level lock
                    # skip_locked=True ensures we don't wait for locked rows
                    # (another thread may be claiming a job)
                    job = (
                        db.query(Job)
                        .filter(Job.status == JobStatus.PENDING)
                        .order_by(Job.created_at)
                        .with_for_update(skip_locked=True)
                        .first()
                    )

                    logger.info(f"Query for pending job: found={job is not None}")

                    if not job:
                        break  # No more pending jobs

                    # Claim the job by updating status (atomic with the lock)
                    job.status = JobStatus.PROCESSING
                    job.started_at = datetime.now(timezone.utc)
                    job.current_step = "Starting..."
                    db.commit()

                    # Parse input params with validation
                    job_id = job.id
                    try:
                        params = json.loads(job.input_params) if job.input_params else {}
                    except json.JSONDecodeError as e:
                        logger.error(f"Job {job_id} has malformed input_params: {e}")
                        job.status = JobStatus.FAILED
                        job.error_message = "Invalid input parameters (malformed JSON)"
                        job.completed_at = datetime.now(timezone.utc)
                        db.commit()
                        continue

                    # Validate required parameters
                    compound_name = params.get('compound_name')
                    smiles = params.get('smiles')
                    if not compound_name or not smiles:
                        logger.error(f"Job {job_id} missing required params: compound_name={compound_name}, smiles={bool(smiles)}")
                        job.status = JobStatus.FAILED
                        job.error_message = "Missing required parameters (compound_name or smiles)"
                        job.completed_at = datetime.now(timezone.utc)
                        db.commit()
                        continue

                # Import here to avoid circular imports
                from backend.services.compound_service import process_compound_job

                job_executor.submit(
                    job_id,
                    process_compound_job,
                    compound_name=compound_name,
                    smiles=smiles,
                    similarity_threshold=params.get('similarity_threshold', 90),
                    activity_types=params.get('activity_types', []),
                )
                logger.info(f"Scheduler claimed and submitted job {job_id}")
                work_done = True

            except OperationalError as e:
                # Database locked or transient error - continue polling
                logger.warning(f"Database busy, will retry: {e}")
                time.sleep(0.5)  # Brief pause before retry
                continue
            except Exception as e:
                logger.error(f"Error processing pending job: {e}", exc_info=True)
                break

        return work_done

    def is_running(self) -> bool:
        """Check if scheduler is currently running."""
        return self._running

    def stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "running": self._running,
            "poll_interval": self._poll_interval,
            "idle_timeout": IDLE_TIMEOUT_SECONDS,
            "last_activity": self._last_activity.isoformat() if self._last_activity else None,
        }


# Global scheduler instance
job_scheduler = JobScheduler()
