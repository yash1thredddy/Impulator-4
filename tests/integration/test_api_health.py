"""
Integration tests for health API endpoints.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from backend.main import app
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check endpoint."""
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
        """Test readiness probe endpoint."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_liveness_probe(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_executor_stats(self, client):
        """Test executor statistics endpoint."""
        response = client.get("/api/v1/health/executor")
        assert response.status_code == 200
        data = response.json()
        assert "max_workers" in data
        assert "active_jobs" in data
        assert "has_capacity" in data
        assert "job_ids" in data
        assert isinstance(data["max_workers"], int)
        assert isinstance(data["active_jobs"], int)
        assert isinstance(data["has_capacity"], bool)
        assert isinstance(data["job_ids"], list)
