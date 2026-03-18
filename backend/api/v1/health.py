"""
Health check endpoints.

Provides multiple levels of health checks:
- /health: Quick overview of system health
- /health/detailed: Comprehensive health check with metrics
- /health/ready: Kubernetes readiness probe
- /health/live: Kubernetes liveness probe
- /health/executor: Executor statistics
- /health/metrics: Application metrics
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.config import settings
from backend.core.database import get_db
from backend.core.executor import job_executor
from backend.core.azure_sync import is_azure_configured
from backend.core.metrics import metrics
from backend.core.auth import verify_admin_api_key
from backend.models.schemas import HealthResponse, ExecutorStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Check health of all services.

    Returns:
        HealthResponse with status of database and executor
    """
    # Check database
    db_healthy = False
    try:
        db.execute(text("SELECT 1"))
        db_healthy = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if db_healthy else "degraded",
        version=settings.APP_VERSION,
        database=db_healthy,
        azure_configured=is_azure_configured(),
        executor_active_jobs=job_executor.get_active_count(),
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Readiness probe with real subsystem checks (STAB-08, STAB-12).

    Returns 200 for healthy/degraded, 503 for unhealthy.
    Unhealthy = database down OR scheduler dead with PENDING jobs.
    """
    import shutil
    from fastapi.responses import JSONResponse
    from backend.core.scheduler import job_scheduler
    from backend.models.database import Job, JobStatus

    status = "healthy"
    checks = {}

    # Database check (critical -- 503 if down)
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception:
        checks["database"] = "unhealthy"
        return JSONResponse(status_code=503, content={"status": "unhealthy", "checks": checks})

    # Scheduler liveness (critical if PENDING jobs exist -- STAB-12)
    pending_count = db.query(Job).filter(Job.status == JobStatus.PENDING).count()
    scheduler_alive = job_scheduler.is_running()
    if pending_count > 0 and not scheduler_alive:
        checks["scheduler"] = "unhealthy"
        return JSONResponse(status_code=503, content={
            "status": "unhealthy",
            "checks": checks,
            "detail": f"Scheduler dead with {pending_count} PENDING jobs",
        })
    checks["scheduler"] = "healthy" if scheduler_alive or pending_count == 0 else "degraded"

    # Azure connectivity (degraded, not unhealthy -- STAB-08)
    if is_azure_configured():
        try:
            from backend.core.azure_sync import _get_blob_client
            blob = _get_blob_client("impulator.db")
            if blob is not None:
                blob.exists()  # Actual network call
                checks["azure"] = "healthy"
            else:
                checks["azure"] = "degraded"
                status = "degraded"
        except Exception:
            checks["azure"] = "degraded"
            status = "degraded"
    else:
        checks["azure"] = "not_configured"

    # Disk space check (check the data directory's filesystem, not just root)
    try:
        data_dir = str(settings.DATA_DIR) if settings.DATA_DIR.exists() else "/"
        usage = shutil.disk_usage(data_dir)
        free_gb = usage.free / (1024 ** 3)
        checks["disk"] = {"status": "healthy" if free_gb > 1.0 else "degraded", "free_gb": round(free_gb, 2)}
        if free_gb <= 1.0:
            status = "degraded"
    except Exception:
        checks["disk"] = {"status": "unknown"}

    return JSONResponse(status_code=200, content={"status": status, "checks": checks})


@router.get("/live")
async def liveness_check() -> dict:
    """
    Kubernetes/container liveness probe.
    Returns 200 if service is alive.
    """
    return {"status": "alive"}


@router.get("/executor", response_model=ExecutorStats)
async def executor_stats() -> ExecutorStats:
    """
    Get executor statistics.

    Returns:
        ExecutorStats with current job queue status
    """
    stats = job_executor.stats()
    return ExecutorStats(**stats)


@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Comprehensive health check for monitoring.

    Returns detailed status of all system components including:
    - Database connectivity and configuration
    - Executor status
    - Azure configuration
    - Rate limiter status
    - Application metrics

    Use this endpoint for monitoring dashboards and alerting.
    """
    checks: Dict[str, Any] = {}

    # Database connectivity
    try:
        db.execute(text("SELECT 1"))
        # Get table counts for debugging
        jobs_count = db.execute(text("SELECT COUNT(*) FROM jobs")).scalar()
        compounds_count = db.execute(text("SELECT COUNT(*) FROM compounds")).scalar()
        checks["database"] = {
            "status": "healthy",
            "jobs_count": jobs_count,
            "compounds_count": compounds_count,
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["database"] = {
            "status": "unhealthy",
            "error": "Connection failed",
        }

    # Executor status
    executor_stats = job_executor.stats()
    checks["executor"] = {
        "status": "healthy" if executor_stats.get("max_workers", 0) > 0 else "degraded",
        "active_jobs": executor_stats.get("active_jobs", 0),
        "max_workers": executor_stats.get("max_workers", 0),
        "has_capacity": executor_stats.get("has_capacity", False),
    }

    # Scheduler status (STAB-12)
    from backend.core.scheduler import job_scheduler
    sched_stats = job_scheduler.stats()
    checks["scheduler"] = {
        "status": "healthy" if sched_stats["running"] else "idle",
        "running": sched_stats["running"],
        "last_activity": sched_stats["last_activity"],
        "poll_interval": sched_stats["poll_interval"],
    }

    # Azure connectivity (STAB-08)
    if is_azure_configured():
        try:
            from backend.core.azure_sync import _get_blob_client
            blob = _get_blob_client("impulator.db")
            if blob is not None and blob.exists():
                checks["azure"] = {"status": "healthy", "configured": True}
            else:
                checks["azure"] = {"status": "degraded", "configured": True}
        except Exception as e:
            # Log the full exception for debugging but keep the public response
            # sanitized — Azure SDK errors can contain connection strings or
            # account names that must not appear in API responses.
            logger.warning("Azure connectivity check failed", exc_info=True)
            checks["azure"] = {"status": "degraded", "configured": True, "error": type(e).__name__}
    else:
        checks["azure"] = {"status": "not_configured", "configured": False}

    # Add upload failure count from metrics
    checks["azure"]["upload_failures"] = metrics.to_dict().get("azure_upload_failed_permanently", 0)

    # Rate limiter (import here to avoid circular imports)
    try:
        from backend.api.v1.jobs import rate_limiter
        checks["rate_limiter"] = {
            "status": "healthy",
            "active_sessions": rate_limiter.active_session_count,
            "max_sessions": rate_limiter.MAX_SESSIONS,
        }
    except ImportError:
        checks["rate_limiter"] = {"status": "unknown"}

    # Application metrics
    checks["metrics"] = metrics.to_dict()

    # Overall health determination
    unhealthy_components = [
        name for name, check in checks.items()
        if isinstance(check, dict) and check.get("status") == "unhealthy"
    ]

    overall_status = "healthy"
    if unhealthy_components:
        overall_status = "unhealthy"
    elif any(
        isinstance(check, dict) and check.get("status") == "degraded"
        for check in checks.values()
    ):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "version": settings.APP_VERSION,
        "environment": "production" if settings.is_production else "development",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get application metrics.

    Returns current values of all tracked metrics including:
    - Job counters (created, completed, failed)
    - API call statistics
    - Cache hit rates
    - Latency statistics

    Use this endpoint for metrics collection and dashboards.
    """
    return {
        "metrics": metrics.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/migrate")
async def run_migrations(
    db: Session = Depends(get_db),
    _admin: bool = Depends(verify_admin_api_key)
) -> Dict[str, Any]:
    """
    Run database migrations and sync to Azure.

    **Requires admin authentication via X-Admin-API-Key header.**

    This endpoint:
    1. Runs pending database migrations
    2. Syncs the updated database to Azure (if configured)

    Returns migration status and any errors encountered.
    """
    from backend.core.database import _apply_migrations_with_lock
    from backend.core.azure_sync import sync_db_to_azure, is_azure_configured

    results = {
        "status": "success",
        "migrations_applied": True,
        "azure_synced": False,
        "errors": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Run migrations
        logger.info("Running database migrations via API...")
        _apply_migrations_with_lock()
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        results["status"] = "error"
        results["migrations_applied"] = False
        results["errors"].append(f"Migration failed: {str(e)}")
        return results

    # Sync to Azure if configured
    if is_azure_configured():
        try:
            logger.info("Syncing database to Azure after migration...")
            sync_db_to_azure()
            results["azure_synced"] = True
            logger.info("Azure sync completed successfully")
        except Exception as e:
            logger.error(f"Azure sync failed: {e}", exc_info=True)
            results["status"] = "partial"
            results["errors"].append(f"Azure sync failed: {str(e)}")
    else:
        logger.info("Azure not configured, skipping sync")

    return results
