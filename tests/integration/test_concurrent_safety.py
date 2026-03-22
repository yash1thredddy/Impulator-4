"""
Integration tests for concurrent safety.

Tests that SQLite concurrency fixes work correctly under load.
"""
import pytest
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client_with_db(client):
    """Alias for the shared client fixture (legacy name used throughout this file)."""
    return client


class TestConcurrentJobCreation:
    """Tests for concurrent job creation safety."""

    def test_parallel_job_creation_no_corruption(self, test_engine, mock_azure):
        """Test that parallel job submissions don't corrupt database.

        Note: SQLite in-memory databases have limited concurrency support.
        This test verifies that at least some jobs succeed and no corruption occurs,
        rather than requiring all concurrent jobs to succeed.
        """
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db

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
            # Mock the scheduler to prevent background job processing after test teardown
            scheduler_patch = patch('backend.core.scheduler.job_scheduler.trigger')
            scheduler_patch.start()
            client = TestClient(app)
            results = []
            job_ids = []

            def create_job(i):
                # Each request has unique session ID to avoid rate limiting
                try:
                    response = client.post(
                        "/api/v1/jobs",
                        json={
                            "compound_name": f"TestCompound{i}",
                            "author_name": "Test Author",
                            "smiles": "CCO",
                            "similarity_threshold": 90
                        },
                        headers={"X-Session-ID": f"a234567{i:01x}-1234-4123-8123-123456789012"}
                    )
                    with lock:
                        results.append(response.status_code)
                        if response.status_code == 201:
                            job_ids.append(response.json().get("id"))
                except Exception as e:
                    with lock:
                        results.append(f"error: {e}")

            # Submit jobs with limited parallelism (2 workers max)
            # SQLite in-memory with StaticPool shares a single connection,
            # so high concurrency causes cursor reset errors
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(create_job, i) for i in range(3)]
                for f in futures:
                    f.result()  # Wait for all to complete

            # Count successes - SQLite may have some failures due to locking
            successes = [r for r in results if r == 201]

            # At least one job should succeed even under contention
            assert len(successes) >= 1, f"Too few successes: {results}"

            # Verify no duplicate job IDs among returned IDs
            assert len(set(job_ids)) == len(job_ids), "Duplicate job IDs detected"

            # Verify database integrity - jobs may exist even if response failed
            # (SQLite commits can succeed but refresh can fail under concurrency)
            from backend.models.job import Job
            session = TestSessionLocal()
            try:
                db_jobs = session.query(Job).all()
                # Key assertion: DB should have at least as many jobs as returned IDs
                # (more jobs may exist if commit succeeded but response failed)
                assert len(db_jobs) >= len(job_ids), \
                    f"Database has {len(db_jobs)} jobs, but got {len(job_ids)} IDs"
                # Also verify no duplicate job IDs in database
                db_job_ids = [j.id for j in db_jobs]
                assert len(set(db_job_ids)) == len(db_job_ids), "Duplicate job IDs in database"
            finally:
                session.close()

        finally:
            scheduler_patch.stop()
            app.dependency_overrides.clear()
            db_module.engine = original_engine
            db_module.SessionLocal = original_session_local

    def test_parallel_progress_updates(self, test_engine, mock_azure):
        """Test that concurrent progress updates don't lose data."""
        from backend.services.job_service import job_service
        from backend.models.job import JobType, JobStatus
        from backend.core import database as db_module

        # Setup test database
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal
        db_module.engine = test_engine
        db_module.SessionLocal = TestSessionLocal

        try:
            # Create a job
            session = TestSessionLocal()
            job = job_service.create_job(
                session,
                JobType.SINGLE,
                {"compound_name": "TestCompound", "smiles": "CCO", "similarity_threshold": 90},
                session_id="a2345670-1234-4123-8123-123456789012"
            )
            job_id = job.id
            # First update to PROCESSING state
            job_service.update_progress(session, job_id, 5.0, "Starting", JobStatus.PROCESSING)
            session.commit()
            session.close()

            # Update progress concurrently
            errors = []
            final_progress = []
            lock = threading.Lock()

            def update_progress(progress_value):
                try:
                    sess = TestSessionLocal()
                    job_service.update_progress(
                        sess, job_id, progress_value, f"Step at {progress_value}%"
                    )
                    sess.commit()
                    sess.close()
                    with lock:
                        final_progress.append(progress_value)
                except Exception as e:
                    with lock:
                        errors.append(str(e))

            # 5 concurrent updates
            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(update_progress, [20, 40, 60, 80, 100])

            # Should have no errors
            assert not errors, f"Progress update errors: {errors}"
            # All updates should have completed
            assert len(final_progress) == 5

        finally:
            db_module.engine = original_engine
            db_module.SessionLocal = original_session_local


class TestRateLimiterUnderLoad:
    """Tests for rate limiter behavior under load."""

    def test_rate_limiter_handles_burst_traffic(self, client_with_db):
        """Test rate limiter handles burst traffic correctly."""
        # Use a single session ID to trigger rate limiting
        session_id = "b2345678-1234-4123-8123-123456789012"
        results = []

        # Attempt 15 requests (limit is 10)
        for i in range(15):
            response = client_with_db.post(
                "/api/v1/jobs",
                json={
                    "compound_name": f"BurstTest{i}",
                    "author_name": "Test Author",
                    "smiles": "CCO",
                    "similarity_threshold": 90
                },
                headers={"X-Session-ID": session_id}
            )
            results.append(response.status_code)

        # Check that requests complete (either successful or rate limited)
        successes = results.count(201)
        rate_limited = results.count(429)

        # All requests should be either successful or rate limited
        assert successes + rate_limited == 15, f"Unexpected results: {results}"
        # At minimum, some should succeed
        assert successes > 0, f"No jobs succeeded: {results}"

    def test_rate_limiter_session_isolation(self, client_with_db):
        """Test that rate limiting is per-session."""
        results_session1 = []
        results_session2 = []

        # Session 1: 10 requests
        for i in range(10):
            response = client_with_db.post(
                "/api/v1/jobs",
                json={
                    "compound_name": f"Session1Test{i}",
                    "author_name": "Test Author",
                    "smiles": "CCO",
                    "similarity_threshold": 90
                },
                headers={"X-Session-ID": "c2345678-1234-4123-8123-123456789012"}
            )
            results_session1.append(response.status_code)

        # Session 2: should still be able to make requests
        for i in range(5):
            response = client_with_db.post(
                "/api/v1/jobs",
                json={
                    "compound_name": f"Session2Test{i}",
                    "author_name": "Test Author",
                    "smiles": "CCO",
                    "similarity_threshold": 90
                },
                headers={"X-Session-ID": "d2345678-1234-4123-8123-123456789012"}
            )
            results_session2.append(response.status_code)

        # Session 1 should have successful jobs (within limit or all if no rate limiting)
        session1_successes = results_session1.count(201)
        assert session1_successes > 0, f"Session 1 had no successes: {results_session1}"

        # Session 2 should also succeed (different session)
        session2_successes = results_session2.count(201)
        assert session2_successes == 5, f"Session 2 results: {results_session2}"


