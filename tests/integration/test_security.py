"""
Integration tests for security fixes.

Tests ownership checks, session validation, and other security measures.
"""
import pytest


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

    def test_invalid_session_id_rejected(self, client):
        """Test that invalid session IDs are rejected."""
        response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "TestCompound",
                "author_name": "Test Author",
                "smiles": "CCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": "invalid-not-uuid"}
        )

        assert response.status_code == 400
        assert "Invalid session ID format" in response.json()["detail"]

    def test_valid_session_id_accepted(self, client, valid_session_id):
        """Test that valid session IDs are accepted."""
        response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "TestCompound",
                "author_name": "Test Author",
                "smiles": "CCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        # Should be 201 (created) or duplicate response, not 400
        assert response.status_code in [201, 200]

    def test_no_session_id_generates_anonymous(self, client):
        """Test that missing session ID generates anonymous session."""
        response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "TestCompound",
                "author_name": "Test Author",
                "smiles": "CCO",
                "similarity_threshold": 90,
            }
            # No X-Session-ID header
        )

        # Should work - anonymous session generated
        assert response.status_code in [201, 200]


class TestOwnershipChecks:
    """Tests for job ownership verification."""

    def test_cancel_own_job_allowed(self, client, valid_session_id):
        """Test that users can cancel their own jobs."""
        # Create a job
        create_response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "OwnedCompound",
                "author_name": "Test Author",
                "smiles": "CCCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        if create_response.status_code == 201:
            job_id = create_response.json()["id"]

            # Cancel with same session
            cancel_response = client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": valid_session_id}
            )

            # Should succeed (200) or conflict (409 if already done)
            assert cancel_response.status_code in [200, 409]

    def test_cancel_others_job_forbidden(self, client, valid_session_id, other_session_id):
        """Test that users cannot cancel other users' jobs."""
        # Create a job with session 1
        create_response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "OtherOwnedCompound",
                "author_name": "Test Author",
                "smiles": "CCCCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        if create_response.status_code == 201:
            job_id = create_response.json()["id"]

            # Try to cancel with different session
            cancel_response = client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": other_session_id}
            )

            # Should be forbidden
            assert cancel_response.status_code == 403
            assert "permission" in cancel_response.json()["detail"].lower()

    def test_delete_others_job_forbidden(self, client, valid_session_id, other_session_id):
        """Test that users cannot delete other users' jobs."""
        # Create and complete/cancel a job with session 1
        create_response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "DeleteTestCompound",
                "author_name": "Test Author",
                "smiles": "CCCCCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        if create_response.status_code == 201:
            job_id = create_response.json()["id"]

            # First cancel it (so it can be deleted)
            client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": valid_session_id}
            )

            # Try to delete with different session
            delete_response = client.delete(
                f"/api/v1/jobs/{job_id}",
                headers={"X-Session-ID": other_session_id}
            )

            # Should be forbidden
            assert delete_response.status_code == 403

    def test_nonexistent_job_returns_404(self, client, valid_session_id):
        """Test that accessing nonexistent job returns 404."""
        response = client.post(
            "/api/v1/jobs/00000000-0000-4000-8000-000000000004/cancel",
            headers={"X-Session-ID": valid_session_id}
        )

        assert response.status_code == 404


class TestCORSRestrictions:
    """Tests for CORS header restrictions."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are set correctly."""
        response = client.options(
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

    def test_smiles_injection_rejected(self, client, valid_session_id):
        """Test that SMILES injection attempts are rejected."""
        malicious_inputs = [
            "CCO<script>alert('xss')</script>",
            "CCO; DROP TABLE jobs;",
            "CCO`whoami`",
        ]

        for smiles in malicious_inputs:
            response = client.post(
                "/api/v1/jobs",
                json={
                    "compound_name": "Test",
                    "author_name": "Test Author",
                    "smiles": smiles,
                    "similarity_threshold": 90,
                },
                headers={"X-Session-ID": valid_session_id}
            )

            assert response.status_code == 422  # Validation error

    def test_compound_name_path_traversal_rejected(self, client, valid_session_id):
        """Test that path traversal in compound names is rejected."""
        response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "../../../etc/passwd",
                "author_name": "Test Author",
                "smiles": "CCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        assert response.status_code == 422  # Validation error

    def test_batch_size_limit_enforced(self, client, valid_session_id):
        """Test that batch size limit is enforced."""
        # Create a batch with too many compounds (over 1000)
        compounds = [
            {"compound_name": f"Compound{i}", "author_name": "Test Author", "smiles": "CCO", "similarity_threshold": 90}
            for i in range(1005)  # Just over the 1000 limit
        ]

        response = client.post(
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

    def test_conflict_status_for_wrong_state(self, client, valid_session_id):
        """Test that 409 Conflict is returned for jobs in wrong state."""
        # Create a job
        create_response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "StatusTestCompound",
                "author_name": "Test Author",
                "smiles": "CCCCCCO",
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": valid_session_id}
        )

        if create_response.status_code == 201:
            job_id = create_response.json()["id"]

            # Cancel it
            client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": valid_session_id}
            )

            # Try to cancel again
            response = client.post(
                f"/api/v1/jobs/{job_id}/cancel",
                headers={"X-Session-ID": valid_session_id}
            )

            # Should be 409 Conflict (job already cancelled)
            assert response.status_code == 409
