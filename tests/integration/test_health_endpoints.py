"""
Integration tests for health/monitoring endpoints.

Tests:
- GET /api/v1/health (basic health check)
- GET /api/v1/health/ready (readiness probe)
- GET /api/v1/health/live (liveness probe)
- GET /api/v1/health/executor (executor stats)
- GET /api/v1/health/detailed (comprehensive health check)
- GET /api/v1/health/metrics (application metrics)
"""
from unittest.mock import MagicMock, patch


def _broken_db():
    """Yield a mock session that raises on execute — simulates DB failure."""
    session = MagicMock()
    session.execute.side_effect = Exception("connection refused")
    try:
        yield session
    finally:
        pass


class TestBasicHealth:
    """Tests for basic health endpoints."""

    def test_health_check(self, client):
        """Basic health check returns status, version, and component info."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "database" in data
        assert "azure_configured" in data
        assert "active_jobs" in data
        assert "max_concurrent_jobs" in data
        assert "timestamp" in data

    def test_readiness_probe(self, client):
        """Readiness probe returns ready status."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] in ["healthy", "degraded"]

    def test_liveness_probe(self, client):
        """Liveness probe returns alive status."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_executor_stats(self, client):
        """Executor stats returns worker info."""
        response = client.get("/api/v1/health/executor")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["max_concurrent_jobs"], int)
        assert isinstance(data["active_jobs"], int)
        assert isinstance(data["has_capacity"], bool)
        assert isinstance(data["jobs"], list)

    def test_health_check_has_latency(self, client):
        """Health check response includes db_latency_ms field."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "db_latency_ms" in data
        # Should be a number (float or None if DB is down)
        if data["db_latency_ms"] is not None:
            assert isinstance(data["db_latency_ms"], (int, float))
            assert data["db_latency_ms"] >= 0

    def test_detailed_health_has_backend_type(self, client):
        """Detailed health shows database backend type."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        db_check = data["checks"]["database"]
        assert "backend" in db_check
        assert db_check["backend"] in ("postgres", "sqlite")
        assert "latency_ms" in db_check

    def test_health_check_has_latency(self, client):
        """Health check response includes db_latency_ms field."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "db_latency_ms" in data
        # Should be a number (float or None if DB is down)
        if data["db_latency_ms"] is not None:
            assert isinstance(data["db_latency_ms"], (int, float))
            assert data["db_latency_ms"] >= 0

    def test_detailed_health_has_backend_type(self, client):
        """Detailed health shows database backend type."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        db_check = data["checks"]["database"]
        assert "backend" in db_check
        assert db_check["backend"] in ("postgres", "sqlite")
        assert "latency_ms" in db_check


class TestDetailedHealth:
    """Tests for GET /api/v1/health/detailed."""

    def test_returns_200_with_checks(self, client):
        """Detailed health check returns all component statuses."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "checks" in data
        assert "timestamp" in data

    def test_includes_database_check(self, client):
        """Response includes database health with counts."""
        data = client.get("/api/v1/health/detailed").json()
        db_check = data["checks"]["database"]

        assert db_check["status"] == "healthy"
        assert "jobs_count" in db_check
        assert "compounds_count" in db_check

    def test_includes_executor_check(self, client):
        """Response includes executor status."""
        data = client.get("/api/v1/health/detailed").json()
        executor = data["checks"]["executor"]

        assert "status" in executor
        assert "active_jobs" in executor
        assert "max_concurrent_jobs" in executor

    def test_includes_azure_check(self, client):
        """Response includes Azure configuration status."""
        data = client.get("/api/v1/health/detailed").json()
        azure = data["checks"]["azure"]

        assert "status" in azure
        assert "configured" in azure

    def test_includes_rate_limiter_check(self, client):
        """Response includes rate limiter status."""
        data = client.get("/api/v1/health/detailed").json()
        rl = data["checks"]["rate_limiter"]

        assert rl["status"] == "healthy"
        assert "active_sessions" in rl

    def test_overall_status_healthy(self, client):
        """Overall status should be healthy when all components work."""
        data = client.get("/api/v1/health/detailed").json()
        assert data["status"] in ("healthy", "degraded")  # degraded OK if azure not configured


class TestMetrics:
    """Tests for GET /api/v1/health/metrics."""

    def test_returns_200_with_metrics(self, client):
        """Metrics endpoint returns metrics dict."""
        response = client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        data = response.json()

        assert "metrics" in data
        assert "timestamp" in data
        assert isinstance(data["metrics"], dict)


# ─────────────────────────────────────────────────
# Degraded / error paths
# ─────────────────────────────────────────────────

class TestHealthDegradedPaths:
    """Tests for error and degraded scenarios in health endpoints."""

    def test_detailed_health_database_error(self, client):
        """Database failure in detailed health marks database as unhealthy."""
        from backend.core.database import get_db as real_get_db
        from backend.main import app

        app.dependency_overrides[real_get_db] = _broken_db
        try:
            response = client.get("/api/v1/health/detailed")
            assert response.status_code == 200
            data = response.json()
            assert data["checks"]["database"]["status"] == "unhealthy"
            assert data["status"] == "unhealthy"
        finally:
            pass

    def test_readiness_probe_database_error(self, client):
        """Readiness probe returns 503 when database is down."""
        from backend.core.database import get_db as real_get_db
        from backend.main import app

        app.dependency_overrides[real_get_db] = _broken_db
        try:
            response = client.get("/api/v1/health/ready")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["checks"]["database"] == "unhealthy"
        finally:
            pass

    def test_basic_health_degraded_on_db_error(self, client):
        """Basic health endpoint returns 'degraded' when DB is unreachable."""
        from backend.core.database import get_db as real_get_db
        from backend.main import app

        app.dependency_overrides[real_get_db] = _broken_db
        try:
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["database"] is False
        finally:
            pass

