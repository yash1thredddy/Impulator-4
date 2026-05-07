"""
Health check endpoints (Postgres-only).

Provides multiple levels of health checks:
- /health: Quick overview of system health
- /health/detailed: Comprehensive health check with metrics
- /health/ready: Kubernetes readiness probe
- /health/live: Kubernetes liveness probe
- /health/executor: Executor statistics
- /health/metrics: Application metrics
"""
import logging
import time as _time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text, select, func

from backend.api.deps import DbDep
from backend.config import settings
from backend.core import executor, upload_worker
from backend.core.azure_sync import is_azure_configured
from backend.core.metrics import metrics
from backend.models.schemas import HealthResponse
from backend.models.enums import JobStatus
from backend.models.job import Job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


class ExecutorStatsResponse(BaseModel):
    """Async executor statistics response (per D-61)."""

    max_concurrent_jobs: int
    active_jobs: int
    slots_available: int
    has_capacity: bool
    jobs: list[dict[str, Any]]
    pending_uploads: int = 0
    upload_worker_active: bool = False


@router.get("", response_model=HealthResponse)
async def health_check(db: DbDep) -> HealthResponse:
    """
    Check health of all services.

    Returns:
        HealthResponse with status of database and executor
    """
    # Check database with latency measurement
    db_healthy = False
    db_latency_ms = None
    try:
        start = _time.monotonic()
        db.execute(text("SELECT 1"))
        db_latency_ms = round((_time.monotonic() - start) * 1000, 1)
        db_healthy = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if db_healthy else "degraded",
        version=settings.APP_VERSION,
        database=db_healthy,
        db_latency_ms=db_latency_ms,
        azure_configured=is_azure_configured(),
        active_jobs=executor.get_active_count(),
        max_concurrent_jobs=settings.MAX_CONCURRENT_JOBS,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ready")
async def readiness_check(db: DbDep):
    """Readiness probe with real subsystem checks (STAB-08, STAB-12).

    Returns 200 for healthy/degraded, 503 for unhealthy.
    Unhealthy = database down OR scheduler dead with PENDING jobs.
    """
    import shutil
    from fastapi.responses import ORJSONResponse
    from backend.core import scheduler

    status = "healthy"
    checks = {}

    # Database check (critical -- 503 if down)
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception:
        checks["database"] = "unhealthy"
        return ORJSONResponse(status_code=503, content={"status": "unhealthy", "checks": checks})

    # Scheduler liveness (critical if PENDING jobs exist -- STAB-12)
    pending_count = db.scalar(select(func.count()).select_from(Job).where(Job.status == JobStatus.PENDING))
    scheduler_alive = scheduler.is_running()
    if pending_count > 0 and not scheduler_alive:
        checks["scheduler"] = "unhealthy"
        return ORJSONResponse(status_code=503, content={
            "status": "unhealthy",
            "checks": checks,
            "detail": f"Scheduler dead with {pending_count} PENDING jobs",
        })
    checks["scheduler"] = "healthy" if scheduler_alive or pending_count == 0 else "degraded"

    # Azure connectivity (informational only -- Postgres is the database)
    if is_azure_configured():
        checks["azure"] = "configured"
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

    return ORJSONResponse(status_code=200, content={"status": status, "checks": checks})


@router.get("/live")
async def liveness_check() -> dict:
    """
    Kubernetes/container liveness probe.
    Returns 200 if service is alive.
    """
    return {"status": "alive"}


@router.get("/executor", response_model=ExecutorStatsResponse)
async def executor_stats_endpoint(db: DbDep) -> ExecutorStatsResponse:
    """
    Get executor statistics (per D-61).

    Returns:
        ExecutorStatsResponse with async executor stats and upload worker info
    """
    stats_data = executor.stats()

    # Add upload worker info
    pending_upload_count = db.scalar(select(func.count()).select_from(Job).where(Job.status == JobStatus.PENDING_UPLOAD))

    return ExecutorStatsResponse(
        max_concurrent_jobs=stats_data["max_concurrent_jobs"],
        active_jobs=stats_data["active_jobs"],
        slots_available=stats_data["slots_available"],
        has_capacity=stats_data["has_capacity"],
        jobs=stats_data.get("jobs", []),
        pending_uploads=pending_upload_count,
        upload_worker_active=upload_worker.is_active(),
    )


@router.get("/detailed")
async def detailed_health_check(db: DbDep) -> dict[str, Any]:
    """
    Comprehensive health check for monitoring.

    Returns detailed status of all system components including:
    - Database connectivity and configuration
    - Executor status
    - Scheduler status
    - Azure configuration
    - Upload worker status
    - Rate limiter status
    - Application metrics

    Use this endpoint for monitoring dashboards and alerting.
    """
    from backend.core import scheduler

    checks: dict[str, Any] = {}

    # Database connectivity
    try:
        start = _time.monotonic()
        db.execute(text("SELECT 1"))
        db_latency_ms = round((_time.monotonic() - start) * 1000, 1)
        # Get table counts for debugging
        jobs_count = db.execute(text("SELECT COUNT(*) FROM jobs")).scalar()
        compounds_count = db.execute(text("SELECT COUNT(*) FROM compounds")).scalar()

        pg_version = db.execute(text("SELECT version()")).scalar()
        db_check: dict[str, Any] = {
            "status": "healthy",
            "backend": "postgres",
            "latency_ms": db_latency_ms,
            "jobs_count": jobs_count,
            "compounds_count": compounds_count,
            "version": pg_version,
        }

        checks["database"] = db_check
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        checks["database"] = {
            "status": "unhealthy",
            "backend": "postgres",
            "error": "Connection failed",
        }

    # Executor status (per D-61)
    exec_stats = executor.stats()
    pending_upload_count = 0
    try:
        pending_upload_count = db.scalar(select(func.count()).select_from(Job).where(Job.status == JobStatus.PENDING_UPLOAD))
    except Exception:
        pass
    checks["executor"] = {
        "status": "healthy" if exec_stats.get("max_concurrent_jobs", 0) > 0 else "degraded",
        "max_concurrent_jobs": exec_stats.get("max_concurrent_jobs", 0),
        "active_jobs": exec_stats.get("active_jobs", 0),
        "slots_available": exec_stats.get("slots_available", 0),
        "has_capacity": exec_stats.get("has_capacity", False),
        "pending_uploads": pending_upload_count,
        "upload_worker_active": upload_worker.is_active(),
    }

    # Scheduler status (per D-62)
    sched_stats = scheduler.stats()
    checks["scheduler"] = {
        "status": "healthy" if sched_stats["active"] else "idle",
        "active": sched_stats["active"],
        "poll_interval": sched_stats["poll_interval"],
        "consecutive_errors": sched_stats.get("consecutive_errors", 0),
        "crash_reason": sched_stats.get("crash_reason", None),
    }

    # Azure connectivity (informational -- Postgres is the database, Azure is for ZIPs/logs)
    if is_azure_configured():
        checks["azure"] = {"status": "configured", "configured": True}
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
async def get_metrics() -> dict[str, Any]:
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
