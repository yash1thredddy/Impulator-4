"""
Integration tests for job API endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine with tables using shared in-memory DB."""
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
                yield


@pytest.fixture
def client(test_engine, mock_azure):
    """Create a test client for the FastAPI app with proper test database."""
    from backend.main import app
    from backend.core import database as db_module
    from backend.core.database import get_db

    # Save original values
    original_engine = db_module.engine
    original_session_local = db_module.SessionLocal

    # Create new SessionLocal bound to test engine
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Patch the module-level engine and SessionLocal
    db_module.engine = test_engine
    db_module.SessionLocal = TestSessionLocal

    def override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    # Restore original values
    app.dependency_overrides.clear()
    db_module.engine = original_engine
    db_module.SessionLocal = original_session_local


class TestJobEndpoints:
    """Tests for job management endpoints."""

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

    def test_get_nonexistent_job(self, client):
        """Test getting a job that doesn't exist."""
        response = client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_cancel_nonexistent_job(self, client):
        """Test cancelling a job that doesn't exist."""
        response = client.post("/api/v1/jobs/nonexistent-id/cancel")
        assert response.status_code == 404

    def test_delete_nonexistent_job(self, client):
        """Test deleting a job that doesn't exist."""
        response = client.delete("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404


class TestJobSubmission:
    """Tests for job submission (may require mocking processing)."""

    @pytest.fixture
    def client(self, test_engine, mock_azure):
        """Create a test client for the FastAPI app with proper test database."""
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db

        # Save original values
        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal

        # Create new SessionLocal bound to test engine
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

        # Patch the module-level engine and SessionLocal
        db_module.engine = test_engine
        db_module.SessionLocal = TestSessionLocal

        def override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        with TestClient(app) as c:
            yield c

        # Restore original values
        app.dependency_overrides.clear()
        db_module.engine = original_engine
        db_module.SessionLocal = original_session_local

    def test_submit_job_invalid_empty_name(self, client):
        """Test job submission with empty compound name."""
        response = client.post("/api/v1/jobs", json={
            "compound_name": "",
            "smiles": "CCO",
            "similarity_threshold": 90
        })
        assert response.status_code == 422  # Validation error

    def test_submit_job_invalid_empty_smiles(self, client):
        """Test job submission with empty SMILES."""
        response = client.post("/api/v1/jobs", json={
            "compound_name": "Test",
            "smiles": "",
            "similarity_threshold": 90
        })
        assert response.status_code == 422  # Validation error

    def test_submit_job_invalid_threshold(self, client):
        """Test job submission with invalid similarity threshold."""
        response = client.post("/api/v1/jobs", json={
            "compound_name": "Test",
            "smiles": "CCO",
            "similarity_threshold": 150  # > 100
        })
        assert response.status_code == 422  # Validation error


class TestJobSubmissionWithScheduler:
    """Tests for job submission that verify scheduler behavior."""

    @pytest.fixture
    def client_with_mock_scheduler(self, test_engine, mock_azure):
        """Create test client with mocked scheduler and proper test database."""
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db

        # Save original values
        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal

        # Create new SessionLocal bound to test engine
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

        # Patch the module-level engine and SessionLocal
        db_module.engine = test_engine
        db_module.SessionLocal = TestSessionLocal

        def override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        # Mock the scheduler
        with patch('backend.api.v1.jobs.job_scheduler') as mock_scheduler:
            with TestClient(app) as c:
                yield c, mock_scheduler

        # Restore original values
        app.dependency_overrides.clear()
        db_module.engine = original_engine
        db_module.SessionLocal = original_session_local

    def test_submit_single_job_triggers_scheduler(self, client_with_mock_scheduler):
        """Test that submitting a single job triggers the scheduler."""
        client, mock_scheduler = client_with_mock_scheduler

        response = client.post("/api/v1/jobs", json={
            "compound_name": "TestCompound",
            "smiles": "CCO",
            "similarity_threshold": 90
        })

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["compound_name"] == "TestCompound"
        assert data["status"] == "pending"

        # Verify scheduler was triggered
        mock_scheduler.trigger.assert_called()

    def test_submit_batch_job_triggers_scheduler_once(self, client_with_mock_scheduler):
        """Test that batch submission triggers scheduler only once."""
        client, mock_scheduler = client_with_mock_scheduler

        response = client.post("/api/v1/jobs/batch", json={
            "compounds": [
                {"compound_name": "Compound1", "smiles": "CCO"},
                {"compound_name": "Compound2", "smiles": "CCCO"},
                {"compound_name": "Compound3", "smiles": "CCCCO"}
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
        assert mock_scheduler.trigger.call_count == 1

    def test_job_status_is_pending_after_submission(self, client_with_mock_scheduler):
        """Test that job status is PENDING after submission (scheduler handles processing)."""
        client, mock_scheduler = client_with_mock_scheduler

        # Submit job
        submit_response = client.post("/api/v1/jobs", json={
            "compound_name": "StatusTest",
            "smiles": "CCO",
            "similarity_threshold": 90
        })
        assert submit_response.status_code == 201
        job_id = submit_response.json()["id"]

        # Check job status
        status_response = client.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "pending"


class TestBatchJobOperations:
    """Tests for batch job operations."""

    @pytest.fixture
    def client(self, test_engine, mock_azure):
        """Create test client with proper test database."""
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db

        # Save original values
        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal

        # Create new SessionLocal bound to test engine
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

        # Patch the module-level engine and SessionLocal
        db_module.engine = test_engine
        db_module.SessionLocal = TestSessionLocal

        def override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        with patch('backend.api.v1.jobs.job_scheduler'):
            with TestClient(app) as c:
                yield c

        # Restore original values
        app.dependency_overrides.clear()
        db_module.engine = original_engine
        db_module.SessionLocal = original_session_local

    def test_batch_returns_batch_id(self, client):
        """Test that batch submission returns a batch_id."""
        response = client.post("/api/v1/jobs/batch", json={
            "compounds": [
                {"compound_name": "BatchTest1", "smiles": "CCO"},
                {"compound_name": "BatchTest2", "smiles": "CCCO"}
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
        # First create a batch
        batch_response = client.post("/api/v1/jobs/batch", json={
            "compounds": [
                {"compound_name": "SummaryTest1", "smiles": "CCO"},
                {"compound_name": "SummaryTest2", "smiles": "CCCO"}
            ],
            "similarity_threshold": 90
        })
        batch_id = batch_response.json()["batch_id"]

        # Get summary
        summary_response = client.get(f"/api/v1/jobs/batch/{batch_id}")
        assert summary_response.status_code == 200
        data = summary_response.json()
        assert "batch_id" in data
        assert "total_jobs" in data  # API returns total_jobs, not total
        assert "pending" in data

    def test_batch_nonexistent_returns_404(self, client):
        """Test getting non-existent batch returns 404."""
        response = client.get("/api/v1/jobs/batch/nonexistent-batch-id")
        assert response.status_code == 404
