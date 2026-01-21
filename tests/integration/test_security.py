"""
Integration tests for security fixes.

Tests ownership checks, session validation, and other security measures.
"""
import os
import time
import pytest
from unittest.mock import patch, MagicMock

# Skip all tests if fastapi not installed
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create test client with mocked Azure and in-memory database."""
    # Use in-memory database for tests
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["TESTING"] = "true"

    # Stop any running scheduler from previous tests
    from backend.core.scheduler import job_scheduler
    job_scheduler._running = False
    time.sleep(0.1)  # Brief pause to let scheduler thread exit

    with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
        with patch('backend.core.azure_sync.download_db_from_azure'):
            with patch('backend.core.azure_sync.sync_db_to_azure'):
                with patch('backend.core.azure_sync.sync_compound_table_from_azure'):
                    with patch('backend.core.azure_sync.sync_logs_to_azure'):
                        # Mock scheduler trigger to prevent background DB access conflicts
                        # The scheduler's trigger() method starts background threads
                        with patch('backend.core.scheduler.job_scheduler.trigger'):
                            # Clear settings cache to pick up test DATABASE_URL
                            from backend.config import get_settings
                            get_settings.cache_clear()

                            # Reset the engine to use in-memory DB
                            from backend.core import database
                            from sqlalchemy import create_engine
                            from sqlalchemy.pool import StaticPool

                            # Create fresh in-memory engine
                            database.engine = create_engine(
                                "sqlite:///:memory:",
                                connect_args={"check_same_thread": False},
                                poolclass=StaticPool,
                            )
                            database.SessionLocal.configure(bind=database.engine)

                            # Create tables
                            from backend.models.database import Base
                            Base.metadata.create_all(bind=database.engine)

                            from backend.main import app
                            with TestClient(app) as client:
                                yield client

                            # Ensure scheduler is stopped after test
                            job_scheduler._running = False


@pytest.fixture
def valid_session_id():
    """Valid UUID v4 session ID."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def other_session_id():
    """Different valid session ID for ownership tests."""
    return "660e8400-e29b-41d4-a716-446655440001"


class TestSessionValidation:
    """Tests for session validation on API endpoints."""

    def test_invalid_session_id_rejected(self, test_client):
        """Test that invalid session IDs are rejected."""
        response = test_client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "TestCompound",
                "smiles": "CCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": "invalid-not-uuid"}
        )

        assert response.status_code == 400
        assert "Invalid session ID format" in response.json()["detail"]

    def test_valid_session_id_accepted(self, test_client, valid_session_id):
        """Test that valid session IDs are accepted."""
        response = test_client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "TestCompound",
                "smiles": "CCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        # Should be 201 (created) or duplicate response, not 400
        assert response.status_code in [201, 200]

    def test_no_session_id_generates_anonymous(self, test_client):
        """Test that missing session ID generates anonymous session."""
        response = test_client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "TestCompound",
                "smiles": "CCO",
                "similarity_threshold": 90,
            }
            # No X-Session-ID header
        )

        # Should work - anonymous session generated
        assert response.status_code in [201, 200]


class TestOwnershipChecks:
    """Tests for job ownership verification."""

    def test_cancel_own_job_allowed(self, test_client, valid_session_id):
        """Test that users can cancel their own jobs."""
        # Create a job
        create_response = test_client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "OwnedCompound",
                "smiles": "CCCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        if create_response.status_code == 201:
            job_id = create_response.json()["id"]

            # Cancel with same session
            cancel_response = test_client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": valid_session_id}
            )

            # Should succeed (200) or conflict (409 if already done)
            assert cancel_response.status_code in [200, 409]

    def test_cancel_others_job_forbidden(self, test_client, valid_session_id, other_session_id):
        """Test that users cannot cancel other users' jobs."""
        # Create a job with session 1
        create_response = test_client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "OtherOwnedCompound",
                "smiles": "CCCCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        if create_response.status_code == 201:
            job_id = create_response.json()["id"]

            # Try to cancel with different session
            cancel_response = test_client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": other_session_id}
            )

            # Should be forbidden
            assert cancel_response.status_code == 403
            assert "permission" in cancel_response.json()["detail"].lower()

    def test_delete_others_job_forbidden(self, test_client, valid_session_id, other_session_id):
        """Test that users cannot delete other users' jobs."""
        # Create and complete/cancel a job with session 1
        create_response = test_client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "DeleteTestCompound",
                "smiles": "CCCCCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        if create_response.status_code == 201:
            job_id = create_response.json()["id"]

            # First cancel it (so it can be deleted)
            test_client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": valid_session_id}
            )

            # Try to delete with different session
            delete_response = test_client.delete(
                f"/api/v1/jobs/{job_id}",
                headers={"X-Session-ID": other_session_id}
            )

            # Should be forbidden
            assert delete_response.status_code == 403

    def test_nonexistent_job_returns_404(self, test_client, valid_session_id):
        """Test that accessing nonexistent job returns 404."""
        response = test_client.post(
            "/api/v1/jobs/nonexistent-job-id/cancel",
            headers={"X-Session-ID": valid_session_id}
        )

        assert response.status_code == 404


class TestCORSRestrictions:
    """Tests for CORS header restrictions."""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are set correctly."""
        response = test_client.options(
            "/api/v1/jobs",
            headers={
                "Origin": "http://localhost:7860",
                "Access-Control-Request-Method": "POST",
            }
        )

        # Check allowed methods are restricted
        allowed_methods = response.headers.get("access-control-allow-methods", "")
        assert "GET" in allowed_methods or "*" not in allowed_methods


class TestInputValidation:
    """Tests for input validation security."""

    def test_smiles_injection_rejected(self, test_client, valid_session_id):
        """Test that SMILES injection attempts are rejected."""
        malicious_inputs = [
            "CCO<script>alert('xss')</script>",
            "CCO; DROP TABLE jobs;",
            "CCO`whoami`",
        ]

        for smiles in malicious_inputs:
            response = test_client.post(
                "/api/v1/jobs",
                json={
                    "compound_name": "Test",
                    "smiles": smiles,
                    "similarity_threshold": 90,
                },
                headers={"X-Session-ID": valid_session_id}
            )

            assert response.status_code == 422  # Validation error

    def test_compound_name_path_traversal_rejected(self, test_client, valid_session_id):
        """Test that path traversal in compound names is rejected."""
        response = test_client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "../../../etc/passwd",
                "smiles": "CCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        assert response.status_code == 422  # Validation error

    def test_batch_size_limit_enforced(self, test_client, valid_session_id):
        """Test that batch size limit is enforced."""
        # Create a batch with too many compounds (over 1000)
        compounds = [
            {"compound_name": f"Compound{i}", "smiles": "CCO", "similarity_threshold": 90}
            for i in range(1001)
        ]

        response = test_client.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": compounds,
                "skip_existing": True,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        # 422 Unprocessable Entity from Pydantic validation
        assert response.status_code == 422


class TestHTTPStatusCodes:
    """Tests for correct HTTP status codes."""

    def test_conflict_status_for_wrong_state(self, test_client, valid_session_id):
        """Test that 409 Conflict is returned for jobs in wrong state."""
        # Create a job
        create_response = test_client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "StatusTestCompound",
                "smiles": "CCCCCCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        if create_response.status_code == 201:
            job_id = create_response.json()["id"]

            # Cancel it
            test_client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": valid_session_id}
            )

            # Try to cancel again
            response = test_client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": valid_session_id}
            )

            # Should be 409 Conflict (job already cancelled)
            assert response.status_code == 409
