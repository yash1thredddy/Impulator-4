"""
FastAPI application entry point.
Single-container deployment optimized for local, HF Spaces, Streamlit Cloud.
"""
import sys
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.core.database import init_db, get_db_session
from backend.core.executor import job_executor
from backend.core.scheduler import job_scheduler
from backend.core.azure_sync import (
    AzureSyncRotatingFileHandler,
    download_db_from_azure,
    sync_db_to_azure,
    sync_compound_table_from_azure,
    sync_logs_to_azure,
    is_azure_configured,
)
from backend.api.v1.router import api_router
from backend.models.database import Job, JobStatus

# Ensure logs directory exists
LOG_DIR = Path("./data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging with both console and file handlers
log_level = logging.DEBUG if settings.DEBUG else logging.INFO
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Root logger configuration with console + Azure-syncing file handler
# When log file is full, it's automatically compressed and uploaded to Azure
logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        AzureSyncRotatingFileHandler(
            LOG_DIR / "backend.log",
            maxBytes=10_000_000,  # 10MB - uploads to Azure when full
            backupCount=2,  # Keep fewer local backups since they're in Azure
            encoding="utf-8"
        )
    ]
)
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

        # Trigger scheduler if there are pending jobs
        if stalled or pending_count > 0:
            job_scheduler.trigger()
            logger.info(f"Scheduler triggered ({len(stalled)} recovered + {pending_count} pending)")


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

    # Sync Compound table with Azure (ensure consistency)
    if is_azure_configured():
        sync_compound_table_from_azure()

    # Recover stalled jobs and start scheduler if needed
    _recover_stalled_jobs()

    logger.info(f"Job executor ready (max_workers={settings.MAX_WORKERS})")

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Shutdown executor gracefully (wait for running jobs)
    logger.info("Waiting for running jobs to complete...")
    job_executor.shutdown(wait=True, cancel_futures=False)

    # Final sync to Azure
    if is_azure_configured():
        logger.info("Final sync to Azure...")
        sync_db_to_azure()
        # Upload current log file (the one that hasn't rotated yet)
        logger.info("Uploading current logs to Azure...")
        sync_logs_to_azure()

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

# Add CORS middleware with explicit allowed methods and headers
# Note: For HF Spaces, we use allow_origin_regex to support *.hf.space pattern
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=r"https://.*\.hf\.space",  # HF Spaces wildcard support
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],  # Only needed methods
    allow_headers=["Content-Type", "X-Session-ID", "Accept"],  # Explicit headers
)


# Global exception handler to prevent internal path exposure
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle uncaught exceptions and return sanitized error messages.

    Prevents internal server paths and sensitive information from being
    exposed in API responses. The full error is still logged for debugging.
    """
    # Log the full error for debugging
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {str(exc)}",
        exc_info=True
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
        content={"detail": error_message, "error_code": "INTERNAL_ERROR"}
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
