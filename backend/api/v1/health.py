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
async def readiness_check(db: Session = Depends(get_db)) -> dict:
    """
    Kubernetes/container readiness probe.
    Returns 200 if service is ready to accept traffic.
    """
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        # Don't expose internal error details
        return {"status": "not ready", "error": "Database connection failed"}


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
        "pending_jobs": executor_stats.get("pending_jobs", 0),
    }

    # Azure configuration
    checks["azure"] = {
        "status": "healthy" if is_azure_configured() else "not_configured",
        "configured": is_azure_configured(),
    }

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
