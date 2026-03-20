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
from backend.core.logging import configure_logging, CorrelationIdMiddleware  # noqa: E402 -- logging must init first
configure_logging()

from backend.core.database import init_db, get_db_session  # noqa: E402 -- after logging init
from backend.core.executor import job_executor  # noqa: E402 -- after logging init
from backend.core.scheduler import job_scheduler  # noqa: E402 -- after logging init
from backend.core.azure_sync import (  # noqa: E402 -- after logging init
    download_db_from_azure,
    sync_db_to_azure,
    sync_logs_to_azure,
    is_azure_configured,
)
from backend.core.exceptions import (  # noqa: E402 -- after logging init
    AppException,
    ErrorCode,
    http_exception_handler,
    validation_exception_handler,
    app_exception_handler,
)
from backend.api.v1.router import api_router  # noqa: E402 -- after logging init
from backend.models.database import Job, JobStatus  # noqa: E402 -- after logging init

logger = logging.getLogger(__name__)


def _wait_for_database(max_retries: int = 10, base_delay: float = 1.0, max_delay: float = 30.0):
    """Wait for database to become available with exponential backoff + jitter.

    Args:
        max_retries: Maximum connection attempts (default 10).
        base_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay cap in seconds (default 30.0).
    """
    import time
    import random
    from sqlalchemy import text as _text
    from backend.core.database import engine, _is_postgres

    # Skip retry logic for SQLite (always local, always available)
    if not _is_postgres:
        return

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
            time.sleep(delay + jitter)


# Global exception handler for uncaught thread exceptions
def _handle_thread_exception(args):
    """Handle uncaught exceptions in threads - logs to file for debugging."""
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

    # Wait for database connectivity (retries with backoff for Postgres)
    _wait_for_database()

    # Initialize database tables
    init_db()
    logger.info("Database initialized")

    # Run Alembic migrations (auto-upgrade to head on every startup)
    # No-op if already at head (one quick SELECT on alembic_version).
    # If migration fails, startup aborts -- container orchestrator restarts.
    try:
        from pathlib import Path as _Path
        from alembic.config import Config as AlembicConfig
        from alembic import command as alembic_command

        _alembic_ini = str(_Path(__file__).parent / "alembic.ini")
        _alembic_cfg = AlembicConfig(_alembic_ini)
        alembic_command.upgrade(_alembic_cfg, "head")
        logger.info("alembic_migrations_applied", status="upgrade_head")
    except Exception as exc:
        logger.error("alembic_migration_failed", error=str(exc), exc_info=True)
        raise

    # Note: Legacy compound table sync removed - database is the source of truth
    # for all compound metadata. UUID-based storage paths are the only supported format.

    # Recover stalled jobs using state machine (ARCH-12)
    with get_db_session() as db:
        from backend.services.job_service import job_service
        recovery = job_service.recover_on_startup(db, job_scheduler.trigger)
        logger.info(f"Startup recovery: {recovery}")

    # Remove orphaned processing folders from crashed runs
    from backend.services.compound_service import CompoundService
    CompoundService.cleanup_stale_folders()

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

    # Remove orphaned processing folders from crashed runs
    from backend.services.compound_service import CompoundService
    CompoundService.cleanup_stale_folders()

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
