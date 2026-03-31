"""
Integration tests for job API endpoints.

Tests job submission, retrieval, cancellation, and deletion
with Pydantic response model validation (TEST-04).
"""
import uuid as _uuid

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from backend.models.schemas import (
    JobResponse,
    JobListResponse,
    ActiveJobResponse,
    DeleteResponse,
)


def _sid():
    """Generate a valid UUID v4 session ID (avoids rate-limit collisions)."""
    return str(_uuid.uuid4())


def _headers(session_id=None):
    """Build X-Session-ID header dict."""
    return {"X-Session-ID": session_id or _sid()}


def _submit_job(client, name="TestCompound", smiles="CCO", session_id=None):
    """Helper: submit a single job and return (response, data, session_id)."""
    if session_id is None:
        session_id = _sid()
    hdrs = {"X-Session-ID": session_id}
    resp = client.post(
        "/api/v1/jobs",
        json={
            "compound_name": name,
            "author_name": "Test Author",
            "smiles": smiles,
            "similarity_threshold": 90,
        },
        headers=hdrs,
    )
    return resp, resp.json(), session_id


# ============================================================================
# Job Submission
# ============================================================================


class TestJobSubmission:
    """Tests for POST /api/v1/jobs with Pydantic response model validation."""

    def test_submit_single_job_response_model(self, client):
        """POST /api/v1/jobs returns 201, response parses through JobResponse."""
        resp, data, _ = _submit_job(client)
        assert resp.status_code == 201

        # Validate through Pydantic model -- catches missing fields and type drift
        job = JobResponse(**data)
        assert job.id is not None
        assert job.status.value == "pending"
        assert job.compound_name == "TestCompound"
        assert job.smiles == "CCO"

    def test_submit_job_missing_smiles(self, client):
        """POST without smiles field returns 422."""
        resp = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "NoSmiles",
                "author_name": "Test Author",
            },
            headers=_headers(),
        )
        assert resp.status_code == 422

    def test_submit_job_missing_compound_name(self, client):
        """POST without compound_name returns 422."""
        resp = client.post(
            "/api/v1/jobs",
            json={
                "author_name": "Test Author",
                "smiles": "CCO",
            },
            headers=_headers(),
        )
        assert resp.status_code == 422

    def test_submit_job_invalid_smiles(self, client):
        """POST with invalid SMILES returns 422."""
        resp = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "BadSmiles",
                "author_name": "Test Author",
                "smiles": "NOT_A_VALID_SMILES!!!",
            },
            headers=_headers(),
        )
        assert resp.status_code == 422


# ============================================================================
# Job Retrieval
# ============================================================================


class TestJobRetrieval:
    """Tests for GET /api/v1/jobs, /jobs/{id}, /jobs/active."""

    def test_get_job_by_id(self, client):
        """GET /api/v1/jobs/{id} returns job parsed through JobResponse."""
        _, submit_data, sid = _submit_job(client)
        job_id = submit_data["id"]

        resp = client.get(f"/api/v1/jobs/{job_id}", headers={"X-Session-ID": sid})
        assert resp.status_code == 200

        job = JobResponse(**resp.json())
        assert str(job.id) == job_id
        assert job.compound_name == "TestCompound"
        assert job.status.value == "pending"

    def test_get_job_not_found(self, client):
        """GET /api/v1/jobs/00000000-0000-4000-8000-000000000001 returns 404."""
        resp = client.get(
            "/api/v1/jobs/00000000-0000-4000-8000-000000000002", headers=_headers()
        )
        assert resp.status_code == 404

    def test_list_jobs_paginated(self, client):
        """GET /api/v1/jobs with pagination validates through JobListResponse."""
        sid = _sid()
        # Submit 3 jobs with same session — use distinct SMILES to avoid duplicate detection
        for name, smiles in [("Job1", "CCO"), ("Job2", "CCCO"), ("Job3", "CCCCO")]:
            _submit_job(client, name=name, smiles=smiles, session_id=sid)

        resp = client.get(
            "/api/v1/jobs?page=1&page_size=2", headers={"X-Session-ID": sid}
        )
        assert resp.status_code == 200

        result = JobListResponse(**resp.json())
        assert len(result.items) == 2
        assert result.total == 3
        assert result.pages == 2
        assert result.page == 1
        assert result.page_size == 2

    def test_get_active_jobs(self, client):
        """GET /api/v1/jobs/active returns list of ActiveJobResponse."""
        _, _, sid = _submit_job(client, name="ActiveJob")

        resp = client.get("/api/v1/jobs/active", headers={"X-Session-ID": sid})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

        # Validate each item through ActiveJobResponse
        for item in data:
            active_job = ActiveJobResponse(**item)
            assert active_job.id is not None
            assert active_job.status.value in ("pending", "processing")


# ============================================================================
# Job Cancel / Delete
# ============================================================================


class TestJobCancelDelete:
    """Tests for POST /api/v1/jobs/{id}/cancel and DELETE /api/v1/jobs/{id}."""

    def test_cancel_pending_job(self, client):
        """POST /api/v1/jobs/{id}/cancel cancels a pending job."""
        _, submit_data, sid = _submit_job(client)
        job_id = submit_data["id"]
        hdrs = {"X-Session-ID": sid}

        resp = client.post(f"/api/v1/jobs/{job_id}/cancel", headers=hdrs)
        assert resp.status_code == 200

        job = JobResponse(**resp.json())
        assert job.status.value == "cancelled"

    def test_cancel_nonexistent_job(self, client):
        """POST /api/v1/jobs/{nonexistent-uuid}/cancel returns 404."""
        resp = client.post(
            "/api/v1/jobs/00000000-0000-4000-8000-000000000003/cancel", headers=_headers()
        )
        assert resp.status_code == 404

    def test_cancel_already_cancelled_returns_409(self, client):
        """Cancelling an already cancelled job returns 409."""
        _, submit_data, sid = _submit_job(client)
        job_id = submit_data["id"]
        hdrs = {"X-Session-ID": sid}

        # Cancel once
        client.post(f"/api/v1/jobs/{job_id}/cancel", headers=hdrs)
        # Cancel again
        resp = client.post(f"/api/v1/jobs/{job_id}/cancel", headers=hdrs)
        assert resp.status_code == 409

    def test_delete_cancelled_job(self, client):
        """DELETE /api/v1/jobs/{id} on cancelled job returns DeleteResponse."""
        _, submit_data, sid = _submit_job(client)
        job_id = submit_data["id"]
        hdrs = {"X-Session-ID": sid}

        # Cancel first (required before deletion)
        client.post(f"/api/v1/jobs/{job_id}/cancel", headers=hdrs)

        resp = client.delete(f"/api/v1/jobs/{job_id}", headers=hdrs)
        assert resp.status_code == 200

        result = DeleteResponse(**resp.json())
        assert str(result.job_id) == job_id
        assert "message" in result.model_dump()

    def test_delete_nonexistent_job(self, client):
        """DELETE /api/v1/jobs/{nonexistent-uuid} returns 404."""
        resp = client.delete(
            "/api/v1/jobs/00000000-0000-4000-8000-000000000003", headers=_headers()
        )
        assert resp.status_code == 404

    def test_delete_pending_job_returns_409(self, client):
        """DELETE on a pending (active) job returns 409."""
        _, submit_data, sid = _submit_job(client)
        job_id = submit_data["id"]

        resp = client.delete(f"/api/v1/jobs/{job_id}", headers={"X-Session-ID": sid})
        assert resp.status_code == 409


# ============================================================================
# Job Detail and Ownership
# ============================================================================


class TestJobDetailAndOwnership:
    """Tests for GET /api/v1/jobs/{id}/detail and ownership checks."""

    def test_get_job_detail(self, client):
        """GET /api/v1/jobs/{id}/detail returns detailed job info."""
        _, submit_data, sid = _submit_job(client)
        job_id = submit_data["id"]

        resp = client.get(f"/api/v1/jobs/{job_id}/detail", headers={"X-Session-ID": sid})
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == job_id
        assert "compound_name" in data
        assert "smiles" in data
        assert "similarity_threshold" in data
        assert "activity_types" in data

    def test_get_job_wrong_session_returns_403(self, client):
        """Accessing a job with a different session ID returns 403."""
        _, submit_data, sid = _submit_job(client)
        job_id = submit_data["id"]

        other_sid = _sid()
        resp = client.get(f"/api/v1/jobs/{job_id}", headers={"X-Session-ID": other_sid})
        assert resp.status_code == 403

    def test_check_duplicates(self, client):
        """POST /api/v1/jobs/check-duplicates returns CheckDuplicatesResponse."""
        resp = client.post(
            "/api/v1/jobs/check-duplicates",
            json={"compound_names": ["Aspirin", "Caffeine"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "existing" in data
        assert "new" in data


# ============================================================================
# Legacy tests preserved for backward compatibility
# ============================================================================


class TestJobEndpoints:
    """Tests for job listing and retrieval endpoints."""

    def test_get_active_jobs_empty(self, client):
        """Test getting active jobs when none exist."""
        response = client.get("/api/v1/jobs/active")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_jobs(self, client):
        """Test listing jobs with pagination."""
        response = client.get("/api/v1/jobs?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "pages" in data
        assert isinstance(data["items"], list)


class TestJobSubmissionWithScheduler:
    """Tests for job submission that verify scheduler behavior."""

    @pytest.fixture
    def client_with_mock_scheduler(self, pg_engine, mock_azure):
        """Create test client with mocked scheduler and proper test database."""
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db

        # Save original values
        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal

        # Create new SessionLocal bound to test engine
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)

        # Patch the module-level engine and SessionLocal
        db_module.engine = pg_engine
        db_module.SessionLocal = TestSessionLocal

        def override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        # Mock the scheduler trigger to prevent background job processing
        with patch('backend.core.scheduler.trigger') as mock_trigger:
            with TestClient(app) as c:
                yield c, mock_trigger

        # Restore original values
        app.dependency_overrides.clear()
        db_module.engine = original_engine
        db_module.SessionLocal = original_session_local

    def test_submit_single_job_triggers_scheduler(self, client_with_mock_scheduler):
        """Test that submitting a single job triggers the scheduler."""
        client, mock_scheduler = client_with_mock_scheduler

        response = client.post("/api/v1/jobs", json={
            "compound_name": "TestCompound",
            "author_name": "Test Author",
            "smiles": "CCO",
            "similarity_threshold": 90
        })

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["compound_name"] == "TestCompound"
        assert data["status"] == "pending"

        # Verify scheduler was triggered
        mock_scheduler.assert_called()

    def test_submit_batch_job_triggers_scheduler_once(self, client_with_mock_scheduler):
        """Test that batch submission triggers scheduler only once."""
        client, mock_scheduler = client_with_mock_scheduler

        response = client.post("/api/v1/jobs/batch", json={
            "compounds": [
                {"compound_name": "Compound1", "author_name": "Test Author", "smiles": "CCO"},
                {"compound_name": "Compound2", "author_name": "Test Author", "smiles": "CCCO"},
                {"compound_name": "Compound3", "author_name": "Test Author", "smiles": "CCCCO"}
            ],
            "similarity_threshold": 90,
            "skip_existing": False
        })

        assert response.status_code == 201
        data = response.json()
        assert "batch_id" in data
        assert len(data["jobs"]) == 3
        assert data["total_submitted"] == 3

        # Scheduler should be triggered once (not per job)
        assert mock_scheduler.call_count == 1

    def test_job_status_is_pending_after_submission(self, client_with_mock_scheduler):
        """Test that job status is PENDING after submission (scheduler handles processing)."""
        client, mock_scheduler = client_with_mock_scheduler
        session_id = "33345678-1234-4123-8123-123456789012"

        # Submit job
        submit_response = client.post(
            "/api/v1/jobs",
            json={
                "compound_name": "StatusTest",
                "author_name": "Test Author",
                "smiles": "CCO",
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": session_id}
        )
        assert submit_response.status_code == 201
        job_id = submit_response.json()["id"]

        # Check job status (must use same session ID)
        status_response = client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-Session-ID": session_id}
        )
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "pending"


class TestBatchJobOperations:
    """Tests for batch job operations."""

    @pytest.fixture
    def client(self, pg_engine, mock_azure):
        """Override client to patch scheduler trigger at the module level."""
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db

        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)
        db_module.engine = pg_engine
        db_module.SessionLocal = TestSessionLocal

        def override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        with patch('backend.core.scheduler.trigger'):
            with TestClient(app) as c:
                yield c

        app.dependency_overrides.clear()
        db_module.engine = original_engine
        db_module.SessionLocal = original_session_local

    def test_batch_returns_batch_id(self, client):
        """Test that batch submission returns a batch_id."""
        response = client.post("/api/v1/jobs/batch", json={
            "compounds": [
                {"compound_name": "BatchTest1", "author_name": "Test Author", "smiles": "CCO"},
                {"compound_name": "BatchTest2", "author_name": "Test Author", "smiles": "CCCO"}
            ],
            "similarity_threshold": 90
        })

        assert response.status_code == 201
        data = response.json()
        assert "batch_id" in data
        assert data["batch_id"] is not None
        assert len(data["batch_id"]) > 0

    def test_get_batch_summary(self, client):
        """Test getting batch summary."""
        session_id = "44345678-1234-4123-8123-123456789012"

        # First create a batch
        batch_response = client.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": [
                    {"compound_name": "SummaryTest1", "author_name": "Test Author", "smiles": "CCO"},
                    {"compound_name": "SummaryTest2", "author_name": "Test Author", "smiles": "CCCO"}
                ],
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": session_id}
        )
        batch_id = batch_response.json()["batch_id"]

        # Get summary (must use same session ID)
        summary_response = client.get(
            f"/api/v1/jobs/batch/{batch_id}",
            headers={"X-Session-ID": session_id}
        )
        assert summary_response.status_code == 200
        data = summary_response.json()
        assert "batch_id" in data
        assert "total_jobs" in data  # API returns total_jobs, not total
        assert "pending" in data

    def test_batch_nonexistent_returns_404(self, client):
        """Test getting non-existent batch returns 404."""
        response = client.get("/api/v1/jobs/batch/00000000-0000-4000-8000-000000000005")
        assert response.status_code == 404

    def test_batch_skips_internal_structure_duplicates(self, client):
        """Batch submission should skip duplicate rows within the same uploaded payload."""
        response = client.post("/api/v1/jobs/batch", json={
            "compounds": [
                {"compound_name": "Quercetin", "author_name": "Test Author", "smiles": "CCO"},
                {"compound_name": "QuercetinAlias", "author_name": "Test Author", "smiles": "CCO"},
                {"compound_name": "Caffeine", "author_name": "Test Author", "smiles": "CCN"},
            ],
            "similarity_threshold": 90
        })

        assert response.status_code == 201
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["total_submitted"] == 2
        assert data["total_skipped"] == 1
        assert data["skipped_internal_duplicates"] == ["QuercetinAlias"]

    def test_batch_skips_internal_name_duplicates(self, client):
        """Batch submission should skip repeated names in the same uploaded payload."""
        response = client.post("/api/v1/jobs/batch", json={
            "compounds": [
                {"compound_name": "CAFFEIC ACID", "author_name": "Test Author", "smiles": "CCO"},
                {"compound_name": "CAFFEIC ACID", "author_name": "Test Author", "smiles": "CCN"},
                {"compound_name": "Caffeine", "author_name": "Test Author", "smiles": "CCN"},
            ],
            "similarity_threshold": 90
        })

        assert response.status_code == 201
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["total_submitted"] == 2
        assert data["total_skipped"] == 1
        assert data["skipped_internal_duplicates"] == ["CAFFEIC ACID"]
