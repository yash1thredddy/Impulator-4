"""Background worker for PENDING_UPLOAD job Azure retry.

Two-level retry (per D-39):
- Inner: 10 retries per batch with exponential backoff (0.5s -> cap 60s)
- Outer: sleep between batches doubles (3min -> 6min -> 12min -> 24min -> cap 30min)
- Success resets outer sleep to 3min

Error classification (per D-40):
- Transient (timeout, 503, connection): increment upload_attempts, retry
- Permanent (ZIP missing, ZIP empty, 4xx): delete compound, requeue or fail

Exhaustion (per D-41, D-42):
- upload_attempts >= 50: delete compound, increment requeue_count, requeue -> PENDING
- requeue_count >= 3: mark FAILED permanently
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone

import structlog

from backend.config import settings
from backend.core.azure_sync import is_azure_configured
from backend.core.database import get_db_session
from backend.models.enums import JobStatus
from backend.models.job import Job
from backend.repositories import compound_repo

logger = structlog.get_logger(__name__)

_worker_task: asyncio.Task | None = None
_shutdown = False

MAX_UPLOAD_ATTEMPTS = 50
MAX_REQUEUE_CYCLES = 3
INITIAL_OUTER_SLEEP = 180  # 3 minutes
MAX_OUTER_SLEEP = 1800  # 30 minutes
MAX_INNER_RETRIES = 10


def start():
    """Start the upload worker as an asyncio task."""
    global _worker_task, _shutdown
    _shutdown = False
    if not is_azure_configured():
        logger.info("upload_worker_skip", reason="Azure not configured")
        return
    _worker_task = asyncio.create_task(_upload_loop(), name="upload-worker")
    logger.info("upload_worker_started")


async def stop():
    """Stop the upload worker gracefully."""
    global _shutdown
    _shutdown = True
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    logger.info("upload_worker_stopped")


def is_active() -> bool:
    """Check if upload worker task is running."""
    return _worker_task is not None and not _worker_task.done()


async def _upload_loop():
    """Main upload retry loop (per D-39).

    Two-level backoff:
    - Inner: individual job uploads with retry
    - Outer: sleep between cycles doubles on failure, resets on success
    """
    outer_sleep = INITIAL_OUTER_SLEEP
    while not _shutdown:
        try:
            uploaded_any = await _process_pending_uploads()
            if uploaded_any:
                outer_sleep = INITIAL_OUTER_SLEEP  # Reset on success
            else:
                outer_sleep = min(outer_sleep * 2, MAX_OUTER_SLEEP)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("upload_worker_error")
        await asyncio.sleep(outer_sleep)


async def _process_pending_uploads() -> bool:
    """Process all PENDING_UPLOAD jobs in one cycle (per D-43).

    Returns True if any upload succeeded this cycle.
    Uses per-job database sessions for proper error isolation (D-69).
    """
    from backend.core.azure_sync import _upload_with_retry

    uploaded_any = False
    loop = asyncio.get_running_loop()

    # Fetch job IDs in a short-lived session
    with get_db_session() as db:
        from sqlalchemy import select
        pending_job_ids = [
            job.id for job in db.scalars(
                select(Job).where(Job.status == JobStatus.PENDING_UPLOAD)
            ).all()
        ]

    # Process each job with its own session (D-69)
    for job_id in pending_job_ids:
        with get_db_session() as db:
            try:
                from sqlalchemy import select as sa_select
                job = db.scalars(
                    sa_select(Job).where(Job.id == job_id)
                ).first()
                if not job or job.status != JobStatus.PENDING_UPLOAD:
                    continue

                entry_id = (job.result_summary or {}).get("entry_id")
                if not entry_id:
                    continue

                zip_path = settings.RESULTS_DIR / entry_id[:2] / f"{entry_id}.zip"

                # D-40: Permanent error -- ZIP missing or empty
                if not zip_path.exists() or zip_path.stat().st_size == 0:
                    _handle_permanent_failure(db, job, "ZIP file missing or empty")
                    continue

                # Inner retry: try upload
                try:
                    await loop.run_in_executor(
                        None, _upload_with_retry, str(zip_path), entry_id
                    )
                    # Upload succeeded -- mark completed
                    from backend.services.job_service import job_service
                    job_service.mark_completed(db, str(job.id))
                    uploaded_any = True
                    logger.info("upload_worker_success", job_id=str(job.id))
                except Exception as e:
                    job.upload_attempts += 1
                    db.commit()
                    logger.warning(
                        "upload_worker_retry",
                        job_id=str(job.id),
                        attempts=job.upload_attempts,
                        error=str(e),
                    )
                    # D-41: Exhaustion check
                    if job.upload_attempts >= MAX_UPLOAD_ATTEMPTS:
                        _handle_exhaustion(db, job)
            except Exception as e:
                logger.error("upload_worker_job_error", job_id=str(job_id), error=str(e))

    return uploaded_any


def _handle_permanent_failure(db, job: Job, reason: str):
    """Handle permanent upload failure (D-40): delete compound, requeue or fail."""
    entry_id = (job.result_summary or {}).get("entry_id")
    if entry_id:
        try:
            compound_repo.delete_by_entry_id(db, uuid.UUID(entry_id))
        except Exception:
            pass

    if job.requeue_count < MAX_REQUEUE_CYCLES:
        job.status = JobStatus.PENDING
        job.requeue_count += 1
        job.upload_attempts = 0
        job.started_at = None
        job.current_step = f"Requeued ({reason}, cycle {job.requeue_count}/{MAX_REQUEUE_CYCLES})"
        db.commit()
        logger.warning(
            "upload_worker_requeue_permanent",
            job_id=str(job.id),
            reason=reason,
            cycle=job.requeue_count,
        )
    else:
        from backend.core.metrics import metrics
        job.status = JobStatus.FAILED
        job.error_message = f"Job failed permanently: {reason} after {MAX_REQUEUE_CYCLES} cycles"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        metrics.increment("azure_upload_failed_permanently")
        logger.error(
            "upload_worker_permanent_fail",
            job_id=str(job.id),
            reason=reason,
        )


def _handle_exhaustion(db, job: Job):
    """Handle upload attempt exhaustion (D-41, D-42)."""
    entry_id = (job.result_summary or {}).get("entry_id")
    if entry_id:
        try:
            compound_repo.delete_by_entry_id(db, uuid.UUID(entry_id))
        except Exception:
            pass

    if job.requeue_count < MAX_REQUEUE_CYCLES:
        job.status = JobStatus.PENDING
        job.requeue_count += 1
        job.upload_attempts = 0
        job.started_at = None
        job.current_step = f"Requeued (upload exhausted, cycle {job.requeue_count}/{MAX_REQUEUE_CYCLES})"
        db.commit()
        logger.warning("upload_worker_requeue", job_id=str(job.id), cycle=job.requeue_count)
    else:
        from backend.core.metrics import metrics
        job.status = JobStatus.FAILED
        job.error_message = "Job failed permanently: upload failed after 3 full processing cycles"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        metrics.increment("azure_upload_failed_permanently")
        logger.error("upload_worker_permanent_fail", job_id=str(job.id))


def _reset():
    """Reset module state for testing."""
    global _worker_task, _shutdown
    _worker_task = None
    _shutdown = False
