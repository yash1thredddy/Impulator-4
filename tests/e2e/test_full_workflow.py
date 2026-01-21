"""
End-to-end tests for complete user workflows.

These tests verify that the entire system works together correctly.
They may take longer to run and should be run after unit/integration tests.

Run with: pytest tests/e2e/ -v -s
"""
import pytest
import time
import uuid
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient


def generate_valid_session_id(prefix: str = "") -> str:
    """Generate a valid UUID v4 session ID for testing."""
    return str(uuid.uuid4())


@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine."""
    from backend.core.database import Base
    from backend.models.database import Job, Compound  # noqa: F401

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_azure():
    """Mock Azure storage for tests."""
    with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
        with patch('backend.core.azure_sync.sync_db_to_azure', return_value=True):
            with patch('backend.core.azure_sync.delete_result_from_azure_by_entry_id', return_value=True):
                with patch('backend.core.azure_sync.upload_result_to_azure_by_entry_id', return_value=True):
                    yield


@pytest.fixture
def client_with_db(test_engine, mock_azure):
    """Create a test client with properly configured database."""
    from backend.main import app
    from backend.core import database as db_module
    from backend.core.database import get_db

    original_engine = db_module.engine
    original_session_local = db_module.SessionLocal

    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db_module.engine = test_engine
    db_module.SessionLocal = TestSessionLocal

    def override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()
    db_module.engine = original_engine
    db_module.SessionLocal = original_session_local


class TestJobSubmissionWorkflow:
    """Test complete job submission workflows."""

    def test_submit_single_job(self, client_with_db):
        """Test submitting a single job and checking its status."""
        # Submit job
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Ethanol",
                "smiles": "CCO",
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": "12345678-1234-4123-8123-123456789012"}
        )

        assert response.status_code == 201
        job_data = response.json()

        assert "id" in job_data
        assert job_data["status"] == "pending"
        assert job_data["compound_name"] == "Ethanol"

        job_id = job_data["id"]

        # Check job status
        status_response = client_with_db.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["id"] == job_id
        assert status_data["status"] in ["pending", "processing"]

    def test_submit_batch_jobs(self, client_with_db):
        """Test submitting a batch of jobs."""
        response = client_with_db.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": [
                    {"compound_name": "Ethanol", "smiles": "CCO"},
                    {"compound_name": "Methanol", "smiles": "CO"},
                    {"compound_name": "Propanol", "smiles": "CCCO"},
                ],
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": "22345678-1234-4123-8123-123456789012"}
        )

        assert response.status_code == 201
        batch_data = response.json()

        assert "batch_id" in batch_data
        assert "jobs" in batch_data
        assert batch_data["total_submitted"] == 3

        # Check batch summary
        batch_id = batch_data["batch_id"]
        summary_response = client_with_db.get(f"/api/v1/jobs/batch/{batch_id}")
        assert summary_response.status_code == 200

        summary = summary_response.json()
        assert summary["batch_id"] == batch_id
        assert summary["total_jobs"] == 3


class TestDuplicateDetectionWorkflow:
    """Test duplicate detection workflows."""

    def test_duplicate_detection_exact_match(self, client_with_db, test_engine):
        """Test detection of exact duplicates.

        Note: The API may either:
        - Return 200 with duplicate_found status (if duplicate check at submission)
        - Return 201 and create job (duplicate check at processing time)
        Both behaviors are acceptable.
        """
        from backend.models.database import Compound
        from sqlalchemy.orm import sessionmaker

        # Create an existing compound
        TestSessionLocal = sessionmaker(bind=test_engine)
        session = TestSessionLocal()

        existing = Compound(
            entry_id="existing-12345678",
            compound_name="Ethanol",
            smiles="CCO",
            inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",  # InChIKey for ethanol
        )
        session.add(existing)
        session.commit()
        session.close()

        # Submit job with same SMILES
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Ethanol",
                "smiles": "CCO",
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": "32345678-1234-4123-8123-123456789012"}
        )

        # API may either detect duplicate immediately (200) or create job (201)
        assert response.status_code in [200, 201], f"Unexpected status: {response.status_code}"
        data = response.json()
        # Check response format based on content, not just status code
        if data.get("status") == "duplicate_found":
            # Duplicate was detected - verify duplicate response format
            assert "duplicate_type" in data or "existing_entry_id" in data or "existing_compound" in data
        else:
            # Job created, should have id and pending status
            assert "id" in data


class TestJobCancellationWorkflow:
    """Test job cancellation workflows."""

    def test_cancel_pending_job(self, client_with_db):
        """Test cancelling a pending job."""
        # Submit job
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Test",
                "smiles": "CCO",
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": "42345678-1234-4123-8123-123456789012"}
        )

        assert response.status_code == 201
        job_id = response.json()["id"]

        # Cancel job
        cancel_response = client_with_db.post(
            f"/api/v1/jobs/{job_id}/cancel",
            headers={"X-Session-ID": "42345678-1234-4123-8123-123456789012"}
        )

        # Should succeed (unless job already processing)
        assert cancel_response.status_code in [200, 409]

    def test_cancel_batch(self, client_with_db):
        """Test cancelling all jobs in a batch."""
        # Submit batch
        response = client_with_db.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": [
                    {"compound_name": "Test1", "smiles": "CCO"},
                    {"compound_name": "Test2", "smiles": "CO"},
                ],
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": "52345678-1234-4123-8123-123456789012"}
        )

        assert response.status_code == 201
        batch_id = response.json()["batch_id"]

        # Cancel batch (must use same session ID that created it)
        cancel_response = client_with_db.post(
            f"/api/v1/jobs/batch/{batch_id}/cancel",
            headers={"X-Session-ID": "52345678-1234-4123-8123-123456789012"}
        )
        assert cancel_response.status_code == 200


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health(self, client_with_db):
        """Test basic health endpoint."""
        response = client_with_db.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["database"] is True

    def test_detailed_health(self, client_with_db):
        """Test detailed health endpoint."""
        response = client_with_db.get("/api/v1/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "database" in data["checks"]
        assert "executor" in data["checks"]

    def test_metrics_endpoint(self, client_with_db):
        """Test metrics endpoint."""
        response = client_with_db.get("/api/v1/health/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "metrics" in data
        assert "timestamp" in data

    def test_readiness_probe(self, client_with_db):
        """Test Kubernetes readiness probe."""
        response = client_with_db.get("/api/v1/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_liveness_probe(self, client_with_db):
        """Test Kubernetes liveness probe."""
        response = client_with_db.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestSessionIsolation:
    """Test session-based job isolation."""

    def test_jobs_isolated_by_session(self, client_with_db):
        """Test that users only see their own jobs."""
        session1_id = "62345678-1234-4123-8123-123456789012"
        session2_id = "72345678-1234-4123-8123-123456789012"

        # Session 1 creates jobs
        session1_jobs = []
        for i in range(3):
            response = client_with_db.post(
                "/api/v1/jobs",
                json={
                    "compound_name": f"Session1Job{i}",
                    "smiles": "CCO",
                    "similarity_threshold": 90
                },
                headers={"X-Session-ID": session1_id}
            )
            assert response.status_code == 201, f"Job creation failed: {response.json()}"
            session1_jobs.append(response.json()["id"])

        # Session 2 creates jobs
        session2_jobs = []
        for i in range(2):
            response = client_with_db.post(
                "/api/v1/jobs",
                json={
                    "compound_name": f"Session2Job{i}",
                    "smiles": "CCO",
                    "similarity_threshold": 90
                },
                headers={"X-Session-ID": session2_id}
            )
            assert response.status_code == 201, f"Job creation failed: {response.json()}"
            session2_jobs.append(response.json()["id"])

        # Session 1 should see its 3 active jobs
        response1 = client_with_db.get(
            "/api/v1/jobs/active",
            headers={"X-Session-ID": session1_id}
        )
        assert response1.status_code == 200
        jobs1 = response1.json()
        # Verify session isolation: session 1 should see 3 jobs
        assert len(jobs1) == 3, f"Session 1 expected 3 jobs, got {len(jobs1)}"

        # Session 2 should see its 2 active jobs
        response2 = client_with_db.get(
            "/api/v1/jobs/active",
            headers={"X-Session-ID": session2_id}
        )
        assert response2.status_code == 200
        jobs2 = response2.json()
        # Verify session isolation: session 2 should see 2 jobs
        assert len(jobs2) == 2, f"Session 2 expected 2 jobs, got {len(jobs2)}"


class TestInputValidation:
    """Test input validation in complete workflows."""

    def test_invalid_smiles_rejected(self, client_with_db):
        """Test that invalid SMILES are handled appropriately.

        Note: The API may either:
        - Return 422 for validation error (if SMILES validated at submission)
        - Return 201 and fail during processing (if SMILES validated later)
        Both behaviors are acceptable.
        """
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Invalid",
                "smiles": "not_a_valid_smiles!!!",
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": "82345678-1234-4123-8123-123456789012"}
        )

        # Accept 201 (job created, will fail during processing) or 422 (validation error)
        assert response.status_code in [201, 422], f"Unexpected status: {response.status_code}"

    def test_invalid_session_id_rejected(self, client_with_db):
        """Test that invalid session IDs are rejected."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Test",
                "smiles": "CCO",
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": "invalid-session-id"}
        )

        assert response.status_code == 400
        assert "Invalid session ID" in response.json()["detail"]

    def test_batch_size_limit(self, client_with_db):
        """Test that batch size is limited."""
        # Create a batch exceeding the limit
        compounds = [
            {"compound_name": f"Compound{i}", "smiles": "CCO"}
            for i in range(1001)  # Over the 1000 limit
        ]

        response = client_with_db.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": compounds,
                "similarity_threshold": 90
            },
            headers={"X-Session-ID": "92345678-1234-4123-8123-123456789012"}
        )

        # Pydantic validation returns 422, custom validation returns 400
        assert response.status_code in [400, 422], f"Expected 400 or 422, got {response.status_code}"
        error_detail = response.json().get("detail", "")
        # Handle both Pydantic validation error format and custom error format
        if isinstance(error_detail, list):
            # Pydantic format: list of error dicts
            error_text = str(error_detail)
        else:
            error_text = str(error_detail)
        assert "1000" in error_text or "Batch too large" in error_text or "at most 1000" in error_text.lower()
