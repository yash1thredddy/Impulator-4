"""Event-driven async job scheduler.

Module-level functions replacing JobScheduler class.
Polls Postgres for PENDING jobs, submits to async executor.
Starts on trigger(), exits after 5min idle.
"""
import asyncio
import uuid
from datetime import datetime, timezone

import structlog

from backend.core.database import get_db_session
from backend.core.logging import request_id_var, session_id_var
from backend.models.enums import JobStatus, JobType
from backend.repositories import job_repo

logger = structlog.get_logger(__name__)

IDLE_TIMEOUT_SECONDS = 300  # 5 min idle -> stop polling
_DEFAULT_POLL_INTERVAL = 6.0

_scheduler_task: asyncio.Task | None = None
_consecutive_errors = 0
_last_activity: float | None = None  # event loop time
_crash_reason: str | None = None


def trigger():
    """Start scheduler if not running (per D-11: sync-callable)."""
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_scheduler_loop(), name="scheduler")


async def stop():
    """Stop scheduler task."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
    _scheduler_task = None


async def _scheduler_loop():
    """Main scheduler loop (per D-09, D-10)."""
    global _consecutive_errors, _last_activity, _crash_reason
    _consecutive_errors = 0
    loop = asyncio.get_running_loop()
    _last_activity = loop.time()

    try:
        while True:
            try:
                had_work = await _process_pending()
                await _check_timeouts()
                _consecutive_errors = 0
                if had_work:
                    _last_activity = loop.time()
                # Idle timeout check
                if _last_activity and (loop.time() - _last_activity) > IDLE_TIMEOUT_SECONDS:
                    try:
                        with get_db_session() as db:
                            if job_repo.get_pending_processing_count(db) == 0:
                                logger.info("scheduler_idle_stop")
                                break
                    except Exception:
                        pass  # Keep running on DB error
            except asyncio.CancelledError:
                raise  # Propagate cancellation
            except Exception as e:
                _consecutive_errors += 1
                _crash_reason = str(e)
                logger.error("scheduler_error", error=str(e), consecutive=_consecutive_errors)
            # Backoff on errors (per D-10)
            backoff = (
                min(60, _DEFAULT_POLL_INTERVAL * (2 ** _consecutive_errors))
                if _consecutive_errors
                else _DEFAULT_POLL_INTERVAL
            )
            await asyncio.sleep(backoff)
    except asyncio.CancelledError:
        logger.info("scheduler_cancelled")


async def _process_pending() -> bool:
    """Claim PENDING jobs in batch and submit to executor (D-59).

    Single DB round-trip claims up to N jobs (where N = available executor slots),
    replacing the previous per-job claiming loop.
    """
    from backend.core import executor
    from backend.config import settings

    slots = max(0, settings.MAX_CONCURRENT_JOBS - executor.get_active_count())
    if slots <= 0:
        return False

    # Batch claim: one DB round-trip for N jobs
    with get_db_session() as db:
        jobs = job_repo.claim_pending_jobs(db, limit=slots)
        db.commit()

    if not jobs:
        return False

    # Extract job data outside DB session (expire_on_commit=False keeps attrs)
    job_params = []
    for job in jobs:
        # BLOCKER-1 (HC-2): COLLECTION jobs carry no job-level smiles. Bypass the
        # single-smiles guard BEFORE it can instant-fail them. Members are loaded
        # from collections.members_config by job_id inside the coroutine (D-02);
        # nothing about the member set is read here.
        if job.job_type == JobType.COLLECTION:
            job_params.append({
                "job_id": job.id,
                "job_type": JobType.COLLECTION,
            })
            continue
        if not job.compound_name or not job.smiles:
            # Mark invalid jobs as failed
            with get_db_session() as db:
                from backend.repositories import job_repo as jr
                jr.update_status(
                    db, job.id, JobStatus.FAILED,
                    error_message="Missing required parameters",
                    completed_at=datetime.now(timezone.utc),
                )
                db.commit()
            continue
        workflow_meta = job.result_summary or {}
        job_params.append({
            "job_id": job.id,
            "compound_name": job.compound_name,
            "smiles": job.smiles,
            "similarity_threshold": job.similarity_threshold,
            "activity_types": list(job.activity_types) if job.activity_types else None,
            "author_name": workflow_meta.get("author_name"),
        })

    # Submit outside DB session
    from backend.services.compound_service import process_compound_job
    from backend.services.collection_service import process_collection_job

    for params in job_params:
        try:
            # Set contextvars for log correlation (per D-12)
            structlog.contextvars.clear_contextvars()
            request_id_var.set(f"scheduler-{uuid.uuid4()}")
            session_id_var.set("scheduler")
            structlog.contextvars.bind_contextvars(
                request_id=request_id_var.get(),
                session_id="scheduler",
            )

            # BLOCKER-2 (HC-2): branch on job_type. COLLECTION dispatches the
            # async fan-out coroutine ONCE (1 global executor slot, D-04) with NO
            # smiles/compound_name kwargs -- the coroutine loads members from
            # collections.members_config by job_id (D-02). Everything else keeps
            # the existing single-compound submit unchanged.
            if params.get("job_type") == JobType.COLLECTION:
                await executor.submit(
                    params["job_id"],
                    process_collection_job,
                )
            else:
                await executor.submit(
                    params["job_id"],
                    process_compound_job,
                    compound_name=params["compound_name"],
                    smiles=params["smiles"],
                    similarity_threshold=params["similarity_threshold"],
                    activity_types=params["activity_types"],
                    author_name=params["author_name"],
                )
        except Exception as e:
            logger.error("scheduler_submit_error", error=str(e), job_id=str(params["job_id"]))

    return bool(job_params)


async def _check_timeouts():
    """Mark timed-out PROCESSING jobs as FAILED (async wrapper, D-58)."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _check_timeouts_sync)


def _check_timeouts_sync():
    """Sync timeout check — SQL already filters by timeout threshold (D-57)."""
    from backend.config import settings

    try:
        with get_db_session() as db:
            now = datetime.now(timezone.utc)
            for job in job_repo.get_stalled_processing_jobs(db, settings.JOB_TIMEOUT):
                db.refresh(job)
                if job.status == JobStatus.PROCESSING:
                    started = job.started_at
                    elapsed = int((now - started).total_seconds()) if started else 0
                    job.status = JobStatus.FAILED
                    job.error_message = f"Job timed out after {elapsed}s"
                    job.completed_at = now
                    db.commit()
    except Exception as e:
        logger.error("timeout_check_error", error=str(e))


def is_running() -> bool:
    """Check if scheduler task is active."""
    return _scheduler_task is not None and not _scheduler_task.done()


def stats() -> dict:
    """Get scheduler statistics (per D-62)."""
    return {
        "active": is_running(),
        "poll_interval": _DEFAULT_POLL_INTERVAL,
        "idle_timeout": IDLE_TIMEOUT_SECONDS,
        "consecutive_errors": _consecutive_errors,
        "crash_reason": _crash_reason,
    }
