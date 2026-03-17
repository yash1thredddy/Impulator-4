"""
Integration tests for health/monitoring endpoints.

Tests:
- GET /api/v1/health (basic health check)
- GET /api/v1/health/ready (readiness probe)
- GET /api/v1/health/live (liveness probe)
- GET /api/v1/health/executor (executor stats)
- GET /api/v1/health/detailed (comprehensive health check)
- GET /api/v1/health/metrics (application metrics)
- POST /api/v1/health/migrate (admin-only migration endpoint)
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
        assert "executor_active_jobs" in data
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
        assert isinstance(data["max_workers"], int)
        assert isinstance(data["active_jobs"], int)
        assert isinstance(data["has_capacity"], bool)
        assert isinstance(data["job_ids"], list)


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
        assert "max_workers" in executor

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


class TestMigrate:
    """Tests for POST /api/v1/health/migrate (admin-protected)."""

    def test_requires_admin_key(self, client):
        """Migration without admin key returns 401."""
        response = client.post("/api/v1/health/migrate")
        assert response.status_code in (401, 503)  # 401 if no key, 503 if not configured

    @patch('backend.core.auth.settings')
    def test_wrong_admin_key_returns_403(self, mock_settings, client):
        """Wrong admin key returns 403."""
        mock_settings.ADMIN_API_KEY = "correct-key"

        response = client.post(
            "/api/v1/health/migrate",
            headers={"X-Admin-API-Key": "wrong-key"},
        )
        assert response.status_code == 403

    @patch('backend.core.auth.settings')
    @patch('backend.core.database._apply_migrations_with_lock')
    def test_valid_admin_key_runs_migration(self, mock_migrate, mock_settings, client):
        """Correct admin key triggers migration."""
        mock_settings.ADMIN_API_KEY = "test-admin-key"

        response = client.post(
            "/api/v1/health/migrate",
            headers={"X-Admin-API-Key": "test-admin-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["migrations_applied"] is True
        mock_migrate.assert_called_once()


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

    @patch('backend.core.auth.settings')
    @patch('backend.core.database._apply_migrations_with_lock',
           side_effect=Exception("migration lock timeout"))
    def test_migration_failure_returns_error(self, mock_migrate, mock_settings, client):
        """Migration failure returns error status with details."""
        mock_settings.ADMIN_API_KEY = "test-admin-key"

        response = client.post(
            "/api/v1/health/migrate",
            headers={"X-Admin-API-Key": "test-admin-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["migrations_applied"] is False
        assert any("migration lock timeout" in e for e in data["errors"])

    @patch('backend.core.auth.settings')
    @patch('backend.core.database._apply_migrations_with_lock')
    @patch('backend.core.azure_sync.is_azure_configured', return_value=True)
    @patch('backend.core.azure_sync.sync_db_to_azure',
           side_effect=Exception("Azure blob timeout"))
    def test_migration_azure_sync_failure_returns_partial(
        self, mock_sync, mock_azure_cfg, mock_migrate, mock_settings, client,
    ):
        """Migration succeeds but Azure sync fails returns partial status."""
        mock_settings.ADMIN_API_KEY = "test-admin-key"

        response = client.post(
            "/api/v1/health/migrate",
            headers={"X-Admin-API-Key": "test-admin-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial"
        assert data["migrations_applied"] is True
        assert data["azure_synced"] is False
        assert any("Azure blob timeout" in e for e in data["errors"])
