"""
FastAPI application entry point.
Single-container deployment optimized for local, HF Spaces, Streamlit Cloud.
"""
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

from backend.config import settings

# Ensure logs directory exists BEFORE configure_logging (file handlers need it)
LOG_DIR = Path("./data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure structured logging BEFORE any other imports that trigger logging
# This must be the FIRST thing after basic imports
from backend.core.logging import configure_logging, CorrelationIdMiddleware  # noqa: E402 -- logging must init first
configure_logging()

from backend.core.database import get_db_session  # noqa: E402 -- after logging init
from backend.core import executor, upload_worker  # noqa: E402 -- after logging init
from backend.core.azure_sync import (  # noqa: E402 -- after logging init
    sync_logs_to_azure,
    is_azure_configured,
    close_azure_client,
)
from backend.core.exceptions import (  # noqa: E402 -- after logging init
    AppException,
    ErrorCode,
    http_exception_handler,
    validation_exception_handler,
    app_exception_handler,
)
from backend.api.v1.router import api_router  # noqa: E402 -- after logging init
from backend.models.job import Job  # noqa: E402 -- after logging init
from backend.models.enums import JobStatus  # noqa: E402 -- after logging init

logger = logging.getLogger(__name__)


async def _wait_for_database(max_retries: int = 10, base_delay: float = 1.0, max_delay: float = 30.0):
    """Wait for database to become available with exponential backoff + jitter.

    Args:
        max_retries: Maximum connection attempts (default 10).
        base_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay cap in seconds (default 30.0).
    """
    import random
    from sqlalchemy import text as _text
    from backend.core.database import engine

    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(_text("SELECT 1"))
                conn.commit()
            logger.info(f"Database connected (attempt {attempt})")
            return
        except Exception as e:
            if attempt == max_retries:
                logger.critical(f"Database unreachable after {max_retries} attempts: {e}")
                raise SystemExit(1)
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            logger.warning(
                f"Database connection failed (attempt {attempt}/{max_retries}), "
                f"retrying in {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay + jitter)


# Global exception handler for uncaught thread exceptions
def _handle_thread_exception(args):
    """Handle uncaught exceptions in threads - logs to file for debugging.

    Still needed for run_in_executor threads (per D-74).
    """
    logger.critical(
        f"UNCAUGHT EXCEPTION in thread '{args.thread.name}': {args.exc_type.__name__}: {args.exc_value}",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )

# Install the global thread exception handler
threading.excepthook = _handle_thread_exception


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover -- startup/shutdown lifecycle
    """
    Application lifespan handler.
    - Startup: Wait for Postgres, run Alembic migrations, recover jobs, start upload worker
    - Shutdown: Stop scheduler, cancel tasks, requeue PROCESSING, upload logs, close Azure
    """
    from backend.core import scheduler  # Import here to avoid circular at module level

    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Ensure data directory exists
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Wait for database connectivity (retries with backoff for Postgres) (D-73)
    await _wait_for_database()

    # Run Alembic migrations (auto-upgrade to head on every startup)
    # No-op if already at head (one quick SELECT on alembic_version).
    # If migration fails, startup aborts -- container orchestrator restarts.
    try:
        from pathlib import Path as _Path
        from alembic.config import Config as AlembicConfig
        from alembic import command as alembic_command

        _alembic_ini = str(_Path(__file__).parent / "alembic.ini")
        _alembic_cfg = AlembicConfig(_alembic_ini)
        # Override script_location to absolute path (ini uses relative "alembic"
        # which resolves against cwd, not ini file location)
        _alembic_cfg.set_main_option(
            "script_location",
            str(_Path(__file__).parent / "alembic"),
        )
        alembic_command.upgrade(_alembic_cfg, "head")
        logger.info("Alembic migrations applied (upgrade head)")
    except Exception as exc:
        logger.error("Alembic migration failed: %s", exc, exc_info=True)
        raise

    # Note: Legacy compound table sync removed - database is the source of truth
    # for all compound metadata. UUID-based storage paths are the only supported format.

    # Replay recovery markers from previous DB-down crashes (D-10)
    try:
        from backend.services.compound_service import scan_recovery_markers
        from sqlalchemy.exc import IntegrityError
        recovery_markers = scan_recovery_markers()
        if recovery_markers:
            logger.info(f"Found {len(recovery_markers)} recovery markers to replay")
            from backend.services.job_service import job_service
            for marker in recovery_markers:
                marker_entry_id = marker.get('entry_id')
                marker_job_id = marker.get('job_id')
                marker_path = settings.DATA_DIR / f".recovery-{marker_entry_id}.json"
                try:
                    with get_db_session() as db:
                        job = job_service.get_job(db, marker_job_id)
                        if not job:
                            # D-22: Job not found -- orphaned marker
                            logger.warning(f"Recovery marker for {marker_entry_id}: job {marker_job_id} not found, skipping")
                        elif job.status in (JobStatus.CANCELLED, JobStatus.FAILED):
                            # D-22: Terminal state -- skip replay, marker will be cleaned up
                            logger.info(f"Recovery marker for {marker_entry_id}: job {marker_job_id} is {job.status.value}, skipping")
                        elif job.status == JobStatus.COMPLETED:
                            # D-23: Already completed -- guard against IntegrityError
                            # (compound may already have been written via InChIKey uniqueness)
                            try:
                                job_service.complete_job(db, marker_job_id, marker.get('result_summary', {}))
                                logger.info(f"Recovery marker replayed for already-completed entry_id={marker_entry_id}")
                            except IntegrityError:
                                db.rollback()
                                logger.info(f"Recovery marker for {marker_entry_id}: already completed with compound entry, skipping")
                        else:
                            # Normal replay: complete the job
                            job_service.complete_job(db, marker_job_id, marker.get('result_summary', {}))
                            logger.info(f"Recovery marker replayed for entry_id={marker_entry_id}")
                except Exception as e:
                    logger.error(f"Recovery marker replay failed for {marker_entry_id}: {e}")
                finally:
                    # D-22: ALWAYS delete marker regardless of outcome
                    # Prevents orphan accumulation and broken markers blocking startup
                    if marker_path.exists():
                        marker_path.unlink()
    except Exception as e:
        logger.warning(f"Recovery marker scan failed (non-fatal): {e}")

    # PENDING_UPLOAD recovery on startup (per D-45)
    try:
        with get_db_session() as db:
            from backend.services.job_service import job_service
            pu_stats = job_service.recover_pending_uploads(db)
            if any(pu_stats.values()):
                logger.info(f"PENDING_UPLOAD recovery: {pu_stats}")
    except Exception as e:
        logger.warning(f"PENDING_UPLOAD recovery failed (non-fatal): {e}")

    # Recover stalled jobs using state machine (ARCH-12)
    with get_db_session() as db:
        from backend.services.job_service import job_service
        recovery = job_service.recover_on_startup(db, scheduler.trigger)
        logger.info(f"Startup recovery: {recovery}")

    # Remove orphaned processing folders from crashed runs
    from backend.services.compound_service import cleanup_stale_folders
    cleanup_stale_folders()

    # Reconcile orphaned Azure uploads (time-based cleanup)
    if is_azure_configured():
        try:
            from backend.core.azure_sync import reconcile_orphaned_uploads
            cleaned = reconcile_orphaned_uploads()
            if cleaned:
                logger.info(f"Reconciled {cleaned} orphaned Azure uploads")
        except Exception as e:
            logger.warning(f"Orphan reconciliation failed (non-fatal): {e}")

    # Start upload worker for PENDING_UPLOAD retry (D-75)
    upload_worker.start()

    logger.info(f"Executor ready (max_concurrent_jobs={settings.MAX_CONCURRENT_JOBS})")

    yield

    # Shutdown (per D-54: 7-step sequence)
    logger.info("Shutting down...")

    # Step 1: Stop scheduler (set flag, cancel task if running)
    await scheduler.stop()

    # Steps 2-3: Cancel all active job tasks and gather with timeout
    await executor.shutdown()

    # Step 4: Requeue any still-PROCESSING jobs -> PENDING (safety net, per D-55)
    try:
        with get_db_session() as db:
            from sqlalchemy import select
            processing_jobs = db.scalars(select(Job).where(Job.status == JobStatus.PROCESSING)).all()
            if processing_jobs:
                for job in processing_jobs:
                    db.refresh(job)
                    if job.status == JobStatus.PROCESSING:
                        job.status = JobStatus.PENDING
                        job.started_at = None
                        job.current_step = "Queued (requeued on shutdown)"
                db.commit()
                logger.info(f"Requeued {len(processing_jobs)} PROCESSING jobs to PENDING on shutdown")
    except Exception as e:
        logger.error(f"Failed to requeue jobs on shutdown: {e}")

    # Step 5: No shared httpx clients to close -- per-job clients are created and
    # closed within process_compound_job (Plan 06 decision).

    # Step 6: Upload logs to Azure (sync fire-and-forget)
    if is_azure_configured():
        logger.info("Uploading current logs to Azure...")
        sync_logs_to_azure()

    # Step 7: Stop upload worker and close Azure blob client
    await upload_worker.stop()
    close_azure_client()

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Impurities Modulator - Compound Analysis Backend",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Register standard error response handlers
app.exception_handler(StarletteHTTPException)(http_exception_handler)
app.exception_handler(RequestValidationError)(validation_exception_handler)
app.exception_handler(AppException)(app_exception_handler)

# 10MB payload limit middleware (QUAL-02)
MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject requests with Content-Length exceeding 10MB (QUAL-02)."""
    content_length = request.headers.get("content-length")
    try:
        content_length_int = int(content_length) if content_length else 0
    except ValueError:
        content_length_int = 0
    if content_length_int > MAX_CONTENT_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "detail": "Request body too large. Maximum size is 10MB.",
                "error_code": "PAYLOAD_TOO_LARGE",
            },
        )
    return await call_next(request)


# Middleware registration order matters: Starlette executes middleware LIFO
# (last-registered wraps outermost). Register CorrelationIdMiddleware first so it
# runs INSIDE CORSMiddleware -- CORS handles preflight before request_id is needed.
app.add_middleware(CorrelationIdMiddleware)

# Add CORS middleware with explicit allowed methods and headers
# Note: For HF Spaces, we use allow_origin_regex to support *.hf.space pattern
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=r"https://.*\.hf\.space",  # HF Spaces wildcard support
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # Only needed methods
    allow_headers=[  # D-46: include all custom headers used by the frontend
        "Content-Type",
        "X-Session-ID",
        "Accept",
        "Idempotency-Key",
        "X-Admin-API-Key",
    ],
    expose_headers=["X-Request-ID"],
)


# Global exception handler to prevent internal path exposure
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle uncaught exceptions and return sanitized error messages.

    Prevents internal server paths and sensitive information from being
    exposed in API responses. The full error is still logged for debugging.
    """
    import structlog
    from backend.core.logging import request_id_var

    log = structlog.get_logger(__name__)
    log.error(
        "unhandled_exception",
        method=request.method,
        path=request.url.path,
        exc_type=type(exc).__name__,
        exc_msg=str(exc),
        exc_info=True,
    )

    # Return sanitized error message
    # Don't expose internal paths, stack traces, or implementation details
    error_message = "An internal error occurred. Please try again later."

    # Provide slightly more detail for common error types
    if isinstance(exc, ValueError):
        error_message = "Invalid input provided."
    elif isinstance(exc, FileNotFoundError):
        error_message = "The requested resource was not found."
    elif isinstance(exc, PermissionError):
        error_message = "Access denied."
    elif isinstance(exc, TimeoutError):
        error_message = "The operation timed out. Please try again."

    return JSONResponse(
        status_code=500,
        content={
            "detail": error_message,
            "error_code": str(ErrorCode.INTERNAL_ERROR),
            "request_id": request_id_var.get(""),
        },
    )


# Include API router
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
