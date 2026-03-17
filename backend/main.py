"""
FastAPI application entry point.
Single-container deployment optimized for local, HF Spaces, Streamlit Cloud.
"""
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
from backend.core.logging import configure_logging, CorrelationIdMiddleware
configure_logging()

from backend.core.database import init_db, get_db_session
from backend.core.executor import job_executor
from backend.core.scheduler import job_scheduler
from backend.core.azure_sync import (
    download_db_from_azure,
    sync_db_to_azure,
    sync_logs_to_azure,
    is_azure_configured,
)
from backend.core.exceptions import (
    AppException,
    ErrorCode,
    http_exception_handler,
    validation_exception_handler,
    app_exception_handler,
)
from backend.api.v1.router import api_router
from backend.models.database import Job, JobStatus

logger = logging.getLogger(__name__)


# Global exception handler for uncaught thread exceptions
def _handle_thread_exception(args):
    """Handle uncaught exceptions in threads - logs to file for debugging."""
    logger.critical(
        f"UNCAUGHT EXCEPTION in thread '{args.thread.name}': {args.exc_type.__name__}: {args.exc_value}",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )

# Install the global thread exception handler
threading.excepthook = _handle_thread_exception


def _recover_stalled_jobs():
    """Reset PROCESSING jobs to PENDING on startup.

    Jobs that were PROCESSING when server crashed will be requeued.
    Also triggers scheduler if there are any pending jobs.
    """
    with get_db_session() as db:
        stalled = db.query(Job).filter(Job.status == JobStatus.PROCESSING).all()
        pending_count = db.query(Job).filter(Job.status == JobStatus.PENDING).count()

        for job in stalled:
            job.status = JobStatus.PENDING
            job.current_step = "Queued (recovered)"

        if stalled:
            db.commit()
            logger.info(f"Recovered {len(stalled)} stalled jobs")

        # Handle SYNC_PENDING jobs: retry Azure upload on startup
        sync_pending = db.query(Job).filter(Job.status == JobStatus.SYNC_PENDING).all()
        if sync_pending:
            logger.info(f"Found {len(sync_pending)} SYNC_PENDING jobs for Azure retry")

        # Trigger scheduler if there are pending jobs or SYNC_PENDING jobs
        if stalled or pending_count > 0 or sync_pending:
            job_scheduler.trigger()
            logger.info(f"Scheduler triggered ({len(stalled)} recovered + {pending_count} pending + {len(sync_pending)} sync_pending)")


def _cleanup_stale_folders():
    """Remove stale compound processing folders from data/results/.

    During processing, compound_service creates folders like data/results/Aspirin/
    which are converted to ZIPs and deleted. If the process crashes mid-processing,
    these folders remain as orphans. Since _recover_stalled_jobs() has already reset
    PROCESSING jobs to PENDING, any remaining folders are stale.

    Only removes directories (not ZIP files or UUID-prefix subdirs).
    """
    import shutil

    results_dir = settings.RESULTS_DIR
    if not results_dir.exists():
        return

    cleaned = 0
    for item in results_dir.iterdir():
        if not item.is_dir():
            continue
        # Skip UUID-prefix subdirs (2-char hex: "3a", "7f", etc.) -- these contain ZIPs
        if len(item.name) == 2 and all(c in '0123456789abcdef' for c in item.name.lower()):
            continue
        # This is a compound processing folder (e.g., "Aspirin", "Caffeine")
        # It's stale because _recover_stalled_jobs already reset all PROCESSING -> PENDING
        try:
            shutil.rmtree(item)
            cleaned += 1
            logger.info(f"Cleaned up stale compound folder: {item.name}")
        except Exception as e:
            logger.warning(f"Failed to clean up stale folder {item.name}: {e}")

    if cleaned:
        logger.info(f"Cleaned up {cleaned} stale compound folder(s)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    - Startup: Download DB from Azure, initialize tables
    - Shutdown: Sync DB to Azure, shutdown executor
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Ensure data directory exists
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download database from Azure (single source of truth)
    if is_azure_configured():
        logger.info("Azure Blob configured, downloading database...")
        download_db_from_azure()
    else:
        logger.info("Azure Blob not configured, using local database only")

    # Initialize database tables
    init_db()
    logger.info("Database initialized")

    # Note: Legacy compound table sync removed - database is the source of truth
    # for all compound metadata. UUID-based storage paths are the only supported format.

    # Recover stalled jobs and start scheduler if needed
    _recover_stalled_jobs()

    # Remove orphaned processing folders from crashed runs
    _cleanup_stale_folders()

    # Reconcile orphaned Azure uploads (STAB-18)
    if is_azure_configured():
        try:
            from backend.core.azure_sync import reconcile_orphaned_uploads
            with get_db_session() as db:
                from backend.models.database import Compound
                db_entry_ids = {row[0] for row in db.query(Compound.entry_id).all() if row[0]}
                cleaned = reconcile_orphaned_uploads(db_entry_ids)
                if cleaned:
                    logger.info(f"Reconciled {cleaned} orphaned Azure uploads")
        except Exception as e:
            logger.warning(f"Orphan reconciliation failed (non-fatal): {e}")

    logger.info(f"Job executor ready (max_workers={settings.MAX_WORKERS})")

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Step 1: Requeue PROCESSING jobs to PENDING (STAB-03)
    # Do this BEFORE shutting down executor so scheduler doesn't claim them again
    try:
        from backend.services.job_service import _db_write_lock
        with get_db_session() as db:
            processing_jobs = db.query(Job).filter(Job.status == JobStatus.PROCESSING).all()
            if processing_jobs:
                with _db_write_lock:
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

    # Step 2: Shutdown API client timeout executor (STAB-14)
    try:
        from backend.modules.api_client import shutdown_api_client
        shutdown_api_client()
    except Exception as e:
        logger.warning(f"Error shutting down API client: {e}")

    # Step 3: Shutdown job executor (wait for threads to finish)
    logger.info("Waiting for running jobs to complete...")
    job_executor.shutdown(wait=True, cancel_futures=False)

    # Step 4: Reset module-level sessions (STAB-16)
    try:
        from backend.modules.pdb_client import shutdown_pdb_client
        shutdown_pdb_client()
    except Exception as e:
        logger.warning(f"Error shutting down PDB client: {e}")
    try:
        from backend.modules.chemical_classifier import shutdown_classifier
        shutdown_classifier()
    except Exception as e:
        logger.warning(f"Error shutting down classifier: {e}")

    # Step 5: Final sync to Azure
    if is_azure_configured():
        logger.info("Final sync to Azure...")
        sync_db_to_azure()
        logger.info("Uploading current logs to Azure...")
        sync_logs_to_azure()

    # Step 6: Close Azure Blob client
    from backend.core.azure_sync import close_azure_client
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

# Add CorrelationIdMiddleware BEFORE CORSMiddleware
# Middleware executes in reverse registration order, so CorrelationIdMiddleware
# runs first (registered before CORS) ensuring request_id is available during processing
app.add_middleware(CorrelationIdMiddleware)

# Add CORS middleware with explicit allowed methods and headers
# Note: For HF Spaces, we use allow_origin_regex to support *.hf.space pattern
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=r"https://.*\.hf\.space",  # HF Spaces wildcard support
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # Only needed methods
    allow_headers=["Content-Type", "X-Session-ID", "Accept"],  # Explicit headers
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
