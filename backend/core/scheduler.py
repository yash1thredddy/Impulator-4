"""
Event-driven job scheduler - only runs when jobs exist.

Uses SQLite as a persistent job queue. Scheduler polls for PENDING jobs
and submits them to the ThreadPoolExecutor.

Features:
- Event-driven: starts on job submit, stops after idle timeout
- Atomic job claiming: single-threaded scheduler prevents double-claiming
- 2 workers max: only 2 jobs in executor, rest stay in SQLite
- Recovery: handles stalled PROCESSING jobs on startup
"""
import json
import threading
import time
import logging
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import text as _text
from sqlalchemy.exc import OperationalError

from backend.core.database import get_db_session
from backend.core.executor import job_executor
from backend.core.logging import request_id_var, session_id_var
from backend.models.database import Job, JobStatus
from backend.services.job_service import _db_write_lock

logger = logging.getLogger(__name__)

# Idle timeout: stop polling 5 min after last job completes
IDLE_TIMEOUT_SECONDS = 300

# SQLite maintenance tracking (STAB-09, STAB-17)
_last_vacuum_time: float = 0.0  # epoch timestamp of last VACUUM
_VACUUM_INTERVAL = 86400  # At most once per day (seconds)


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

                logger.debug(f"Scheduler poll: executor capacity={job_executor.has_capacity()}, active={job_executor.get_active_count()}")

                had_work = self._process_pending()
                self._retry_sync_pending()  # STAB-07: retry failed Azure uploads
                self._check_timeouts()  # STAB-06: timeout watchdog

                if had_work:
                    self._last_activity = datetime.now(timezone.utc)

                # Check idle timeout
                if self._should_stop():
                    self._run_maintenance()  # STAB-09, STAB-17: WAL + VACUUM on idle
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

    def _run_maintenance(self):
        """Run SQLite maintenance during idle periods (STAB-09, STAB-17).

        Runs WAL checkpoint + VACUUM + ANALYZE at most once per day,
        only when no jobs are active. Uses raw sqlite3 connection
        (bypasses SQLAlchemy) since VACUUM needs exclusive access.
        """
        global _last_vacuum_time
        import os
        import sqlite3

        from backend.config import settings

        now = time.time()
        if now - _last_vacuum_time < _VACUUM_INTERVAL:
            return  # Already ran today

        # Double-check no active jobs
        try:
            with get_db_session() as db:
                active = (
                    db.query(Job)
                    .filter(Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]))
                    .count()
                )
                if active > 0:
                    return  # Jobs active, skip maintenance
        except Exception:
            return

        db_path = str(settings.DATA_DIR / "impulator.db")
        if not os.path.exists(db_path):
            return

        try:
            conn = sqlite3.connect(db_path, timeout=10)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.info("WAL checkpoint completed")
                conn.execute("VACUUM")
                logger.info("VACUUM completed")
                conn.execute("ANALYZE")
                logger.info("ANALYZE completed")
                _last_vacuum_time = time.time()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"SQLite maintenance failed (non-fatal): {e}")

    def _check_timeouts(self):
        """Mark timed-out PROCESSING jobs as FAILED (STAB-06).

        Jobs exceeding settings.JOB_TIMEOUT are marked FAILED. The thread
        cannot be killed but complete_job() will no-op because status is FAILED.
        """
        from backend.config import settings
        from backend.core.exceptions import ErrorCode

        try:
            with get_db_session() as db:
                now = datetime.now(timezone.utc)
                timeout_seconds = settings.JOB_TIMEOUT

                processing_jobs = (
                    db.query(Job)
                    .filter(Job.status == JobStatus.PROCESSING)
                    .filter(Job.started_at.isnot(None))
                    .all()
                )

                for job in processing_jobs:
                    started = job.started_at
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    elapsed = (now - started).total_seconds()
                    if elapsed > timeout_seconds:
                        with _db_write_lock:
                            db.refresh(job)
                            if job.status == JobStatus.PROCESSING:
                                job.status = JobStatus.FAILED
                                job.error_message = f"Job timed out after {int(elapsed)}s (limit: {timeout_seconds}s)"
                                job.error_code = str(ErrorCode.JOB_TIMEOUT)
                                job.completed_at = now
                                db.commit()
                                logger.warning("job_timeout", extra={"job_id": job.id, "elapsed": elapsed})
        except Exception as e:
            logger.error(f"Timeout check error: {e}")

    def _retry_sync_pending(self):
        """Retry Azure upload for SYNC_PENDING jobs (STAB-07).

        Jobs enter SYNC_PENDING when Azure upload fails after all retries.
        This method picks them up and retries the upload using the stored
        result_path. On success → COMPLETED. On failure → stays SYNC_PENDING
        (retried on next poll cycle, up to 10 total attempts tracked by
        retry_count in result_summary).
        """
        from backend.core.azure_sync import upload_result_to_azure_by_entry_id

        try:
            with get_db_session() as db:
                sync_jobs = (
                    db.query(Job)
                    .filter(Job.status == JobStatus.SYNC_PENDING)
                    .all()
                )

                for job in sync_jobs:
                    # Parse retry count from result_summary
                    retry_count = 0
                    entry_id = None
                    try:
                        summary = json.loads(job.result_summary) if job.result_summary else {}
                        retry_count = summary.get('sync_retry_count', 0)
                        entry_id = summary.get('entry_id')
                    except (json.JSONDecodeError, TypeError):
                        pass

                    if retry_count >= 10:
                        # Max retries exceeded — mark as FAILED
                        with _db_write_lock:
                            db.refresh(job)
                            if job.status == JobStatus.SYNC_PENDING:
                                job.status = JobStatus.FAILED
                                job.error_message = f"Azure upload failed permanently after {retry_count} retries"
                                job.completed_at = datetime.now(timezone.utc)
                                db.commit()
                                logger.error("sync_pending_exhausted", extra={"job_id": job.id, "retries": retry_count})
                        continue

                    if not entry_id or not job.result_path:
                        continue  # Missing data, can't retry

                    # Attempt Azure upload
                    success = upload_result_to_azure_by_entry_id(entry_id, job.result_path)

                    with _db_write_lock:
                        db.refresh(job)
                        if job.status != JobStatus.SYNC_PENDING:
                            continue  # Status changed while we were uploading

                        if success:
                            job.status = JobStatus.COMPLETED
                            job.completed_at = datetime.now(timezone.utc)
                            db.commit()
                            logger.info("sync_pending_resolved", extra={"job_id": job.id, "retries": retry_count + 1})
                        else:
                            # Increment retry count
                            try:
                                summary = json.loads(job.result_summary) if job.result_summary else {}
                                summary['sync_retry_count'] = retry_count + 1
                                job.result_summary = json.dumps(summary)
                                db.commit()
                            except Exception:
                                pass
                            logger.warning("sync_pending_retry_failed", extra={"job_id": job.id, "retry": retry_count + 1})

        except Exception as e:
            logger.error(f"SYNC_PENDING retry error: {e}")

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
                    # STAB-21: Round-robin fair scheduling by session_id
                    # Interleaves jobs across sessions: one per session, oldest first.
                    # User A (999 jobs) + User B (1 job) = A, B, A, A, A...
                    fair_query = _text("""
                        SELECT id FROM (
                            SELECT id, ROW_NUMBER() OVER (
                                PARTITION BY session_id
                                ORDER BY created_at
                            ) as rn
                            FROM jobs
                            WHERE UPPER(status) = 'PENDING'
                        )
                        ORDER BY rn, id
                        LIMIT 1
                    """)
                    result = db.execute(fair_query).first()
                    if not result:
                        break  # No more pending jobs

                    job = db.query(Job).filter(Job.id == result[0]).first()
                    logger.debug(f"Fair scheduling: selected job {result[0]} (round-robin)")

                    if not job:
                        break  # Job disappeared between queries (race)

                    # Claim the job under write lock to prevent cancel-vs-claim race
                    with _db_write_lock:
                        # Re-fetch job inside lock to check if it was cancelled
                        db.refresh(job)
                        if job.status != JobStatus.PENDING:
                            logger.info(f"Job {job.id} no longer PENDING (status={job.status}), skipping")
                            continue

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

                try:
                    # Generate synthetic correlation ID for scheduler-initiated jobs
                    # (no originating HTTP request, so we create one for log traceability)
                    request_id_var.set(f"scheduler-{uuid.uuid4()}")
                    session_id_var.set("scheduler")
                    structlog.contextvars.clear_contextvars()
                    structlog.contextvars.bind_contextvars(
                        request_id=request_id_var.get(),
                        session_id="scheduler",
                    )

                    job_executor.submit(
                        job_id,
                        process_compound_job,
                        compound_name=compound_name,
                        smiles=smiles,
                        similarity_threshold=params.get('similarity_threshold', 90),
                        activity_types=params.get('activity_types', []),
                        author_name=params.get('author_name'),
                    )
                    logger.info(f"Scheduler claimed and submitted job {job_id}")
                    work_done = True
                except Exception as submit_error:
                    # Executor submission failed - revert job status to PENDING for retry
                    logger.error(f"Failed to submit job {job_id} to executor: {submit_error}")
                    try:
                        with get_db_session() as db_retry:
                            job_to_revert = db_retry.query(Job).filter(Job.id == job_id).first()
                            if job_to_revert:
                                job_to_revert.status = JobStatus.PENDING
                                job_to_revert.started_at = None
                                job_to_revert.current_step = "Queued (executor submission failed, will retry)"
                                db_retry.commit()
                                logger.info(f"Reverted job {job_id} to PENDING after submit failure")
                    except Exception as revert_error:
                        logger.error(f"Failed to revert job {job_id} status: {revert_error}")

                    # IMPORTANT: Break out of claiming loop to prevent tight retry
                    # The job is back to PENDING and will be retried on the next poll cycle
                    work_done = True  # Signal that we attempted work (prevents immediate re-poll)
                    break

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
