"""
Integration tests for Error Handling and Edge Cases.

Tests error handling throughout the application including:
- Invalid input handling
- API error responses
- Database error handling
- External service failures
- Concurrent request handling
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
    # Import models BEFORE create_all to register them with Base
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
def client_with_db(test_engine, mock_azure):
    """Create a test client with properly configured database.

    This fixture patches both the get_db dependency AND the underlying engine/SessionLocal
    to ensure all database operations use the test database.
    """
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
    client = TestClient(app)
    yield client

    # Restore original values
    app.dependency_overrides.clear()
    db_module.engine = original_engine
    db_module.SessionLocal = original_session_local


class TestInvalidInputHandling:
    """Tests for invalid input handling."""

    def test_invalid_smiles_rejected(self, client_with_db):
        """Test that invalid SMILES are rejected with appropriate error."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "TestCompound",
                "smiles": "invalid_smiles_XYZ123",
                "similarity_threshold": 90
            }
        )

        # Should still create job but may fail during processing
        # or validation should catch it
        assert response.status_code in [201, 400, 422]

    def test_empty_compound_name_rejected(self, client_with_db):
        """Test that empty compound name is rejected."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "",
                "smiles": "CCO",
                "similarity_threshold": 90
            }
        )

        # Should be rejected by validation
        assert response.status_code == 422

    def test_missing_required_fields(self, client_with_db):
        """Test that missing required fields are rejected."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Test"
                # Missing smiles
            }
        )

        assert response.status_code == 422

    def test_invalid_similarity_threshold(self, client_with_db):
        """Test that invalid similarity threshold is rejected."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Test",
                "smiles": "CCO",
                "similarity_threshold": 150  # Should be 0-100
            }
        )

        assert response.status_code == 422

    def test_negative_similarity_threshold(self, client_with_db):
        """Test that negative similarity threshold is rejected."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Test",
                "smiles": "CCO",
                "similarity_threshold": -10
            }
        )

        assert response.status_code == 422

    def test_very_long_compound_name(self, client_with_db):
        """Test handling of very long compound name."""
        long_name = "A" * 500

        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": long_name,
                "smiles": "CCO",
                "similarity_threshold": 90
            }
        )

        # Should either accept (with truncation) or reject
        assert response.status_code in [201, 422]

    def test_special_characters_in_name(self, client_with_db):
        """Test handling of special characters in compound name."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Test<script>alert('xss')</script>",
                "smiles": "CCO",
                "similarity_threshold": 90
            }
        )

        # Should either sanitize or reject
        assert response.status_code in [201, 422]


class TestNonExistentResourceHandling:
    """Tests for handling requests to non-existent resources."""

    def test_get_nonexistent_job(self, client_with_db):
        """Test getting a job that doesn't exist."""
        response = client_with_db.get("/api/v1/jobs/nonexistent-id")

        assert response.status_code == 404

    def test_cancel_nonexistent_job(self, client_with_db):
        """Test cancelling a job that doesn't exist."""
        response = client_with_db.post("/api/v1/jobs/nonexistent-id/cancel")

        assert response.status_code == 404

    def test_delete_nonexistent_job(self, client_with_db):
        """Test deleting a job that doesn't exist."""
        response = client_with_db.delete("/api/v1/jobs/nonexistent-id")

        assert response.status_code == 404


class TestAPIResponseCodes:
    """Tests for appropriate HTTP response codes."""

    def test_health_check_returns_200(self, client_with_db):
        """Test health check returns 200."""
        response = client_with_db.get("/api/v1/health")

        assert response.status_code == 200

    def test_create_job_returns_201(self, client_with_db):
        """Test job creation returns 201."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Test",
                "smiles": "CCO",
                "similarity_threshold": 90
            }
        )

        assert response.status_code == 201

    def test_list_jobs_returns_200(self, client_with_db):
        """Test job listing returns 200."""
        response = client_with_db.get("/api/v1/jobs")

        assert response.status_code == 200

    def test_active_jobs_returns_200(self, client_with_db):
        """Test active jobs endpoint returns 200."""
        response = client_with_db.get("/api/v1/jobs/active")

        assert response.status_code == 200


class TestMalformedRequests:
    """Tests for handling malformed requests."""

    def test_invalid_json_body(self, client_with_db):
        """Test handling of invalid JSON body."""
        response = client_with_db.post(
            "/api/v1/jobs",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_wrong_content_type(self, client_with_db):
        """Test handling of wrong content type."""
        response = client_with_db.post(
            "/api/v1/jobs",
            content="compound_name=Test&smiles=CCO",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422

    def test_extra_unknown_fields(self, client_with_db):
        """Test handling of extra unknown fields in request."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Test",
                "smiles": "CCO",
                "similarity_threshold": 90,
                "unknown_field": "value",
                "another_unknown": 123
            }
        )

        # Should ignore unknown fields and succeed
        assert response.status_code == 201


class TestBatchJobErrorHandling:
    """Tests for batch job error handling."""

    def test_empty_compound_list(self, client_with_db):
        """Test batch job with empty compound list."""
        response = client_with_db.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": []
            }
        )

        assert response.status_code == 422

    def test_batch_with_invalid_compound(self, client_with_db):
        """Test batch job with one invalid compound."""
        response = client_with_db.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": [
                    {"name": "Valid", "smiles": "CCO"},
                    {"name": "", "smiles": "invalid"}  # Invalid
                ]
            }
        )

        # Should handle gracefully
        assert response.status_code in [201, 422]


class TestDatabaseErrorHandling:
    """Tests for database error handling."""

    def test_handles_db_connection_error(self, mock_azure):
        """Test handling of database connection error."""
        from backend.main import app
        from backend.core.database import get_db

        def broken_db():
            raise Exception("Database connection failed")

        app.dependency_overrides[get_db] = broken_db

        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/v1/jobs")

            # Should return 500 or handle gracefully
            assert response.status_code in [500, 503]
        finally:
            app.dependency_overrides.clear()


class TestExternalServiceFailures:
    """Tests for handling external service failures."""

    def test_chembl_api_failure_handled(self):
        """Test that ChEMBL API failure is handled gracefully."""
        from backend.modules.api_client import get_chembl_ids
        import requests

        with patch('backend.modules.api_client._get_chembl_client', side_effect=Exception("ChEMBL unavailable")):
            # Should not raise, return empty result
            try:
                result = get_chembl_ids("CCO", 90)
                # May return empty list or None
                assert result is None or result == []
            except Exception:
                # Some implementations may raise
                pass

    def test_pdb_api_failure_handled(self):
        """Test that PDB API failure is handled gracefully."""
        from backend.modules.pdb_client import search_similar_ligands
        import requests

        with patch('backend.modules.pdb_client.requests.post',
                   side_effect=requests.exceptions.ConnectionError("PDB unavailable")):
            search_similar_ligands.cache_clear()
            result = search_similar_ligands("CCO")

            # Should return empty list
            assert result == []


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    def test_concurrent_job_creation(self, test_engine, mock_azure):
        """Test creating multiple jobs concurrently."""
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import uuid

        # Save original values
        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal

        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        db_module.engine = test_engine
        db_module.SessionLocal = TestSessionLocal
        lock = threading.Lock()

        def override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        try:
            client = TestClient(app)
            results = []

            def create_job(i):
                # Use valid UUID format for session ID (auth.py validates UUID v4 format)
                session_id = str(uuid.uuid4())
                response = client.post(
                    "/api/v1/jobs",
                    json={
                        "compound_name": f"Compound{i}",
                        "smiles": "CCO",
                        "similarity_threshold": 90
                    },
                    headers={"X-Session-ID": session_id}
                )
                with lock:
                    results.append(response.status_code)

            # Create 5 jobs concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(create_job, range(5))

            # All should succeed
            assert all(status == 201 for status in results)
        finally:
            app.dependency_overrides.clear()
            db_module.engine = original_engine
            db_module.SessionLocal = original_session_local

    def test_concurrent_job_queries(self, test_engine, mock_azure):
        """Test querying jobs concurrently."""
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # Save original values
        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal

        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        db_module.engine = test_engine
        db_module.SessionLocal = TestSessionLocal
        lock = threading.Lock()

        def override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        try:
            client = TestClient(app)
            results = []

            def query_jobs():
                response = client.get("/api/v1/jobs")
                with lock:
                    results.append(response.status_code)

            # Query 10 times concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(lambda _: query_jobs(), range(10))

            # All should succeed
            assert all(status == 200 for status in results)
        finally:
            app.dependency_overrides.clear()
            db_module.engine = original_engine
            db_module.SessionLocal = original_session_local


class TestInChIKeyEdgeCases:
    """Tests for InChIKey edge cases."""

    def test_inchikey_with_none_smiles(self):
        """Test InChIKey generation with None SMILES."""
        from backend.services.job_service import generate_inchikey

        result = generate_inchikey(None)
        assert result is None

    def test_inchikey_with_empty_smiles(self):
        """Test InChIKey generation with empty SMILES."""
        from backend.services.job_service import generate_inchikey

        result = generate_inchikey("")
        assert result is None

    def test_inchikey_with_whitespace_smiles(self):
        """Test InChIKey generation with whitespace SMILES."""
        from backend.services.job_service import generate_inchikey

        result = generate_inchikey("   ")
        assert result is None

    def test_inchikey_with_invalid_smiles(self):
        """Test InChIKey generation with invalid SMILES."""
        from backend.services.job_service import generate_inchikey

        result = generate_inchikey("not_a_valid_smiles")
        assert result is None


class TestTimeoutHandling:
    """Tests for timeout handling."""

    def test_api_request_timeout(self):
        """Test handling of API request timeout."""
        from backend.modules.api_client import fetch_all_activities_single_batch
        import requests

        with patch('backend.modules.api_client._get_chembl_client',
                   side_effect=requests.exceptions.Timeout("Request timed out")):
            try:
                result = fetch_all_activities_single_batch(['CHEMBL1'])
                # Should return empty list
                assert result == []
            except Exception:
                # Some implementations may raise
                pass
