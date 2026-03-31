"""
Unit tests for JobService.
"""
import pytest


class TestJobServiceWithRealDB:
    """Tests with real Postgres database."""

    @pytest.fixture
    def service(self):
        """Create a JobService instance."""
        from backend.services.job_service import JobService
        return JobService()

    def test_create_and_get_job(self, service, db_session):
        """Test creating and retrieving a job."""
        from backend.models.enums import JobType

        job = service.create_job(
            db_session,
            JobType.SINGLE,
            compound_name="TestCompound", smiles="CCO",
        )

        retrieved = service.get_job(db_session, job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

    def test_update_progress(self, service, db_session):
        """Test updating job progress."""
        from backend.models.enums import JobType, JobStatus

        job = service.create_job(
            db_session,
            JobType.SINGLE,
            compound_name="TestCompound", smiles="CCO",
        )

        service.update_progress(
            db_session,
            job.id,
            progress=50.0,
            current_step="Processing...",
            status=JobStatus.PROCESSING
        )

        retrieved = service.get_job(db_session, job.id)
        assert retrieved.progress == 50.0
        assert retrieved.current_step == "Processing..."
        assert retrieved.status == JobStatus.PROCESSING

    def test_fail_job(self, service, db_session):
        """Test marking a job as failed."""
        from backend.models.enums import JobType, JobStatus

        job = service.create_job(
            db_session,
            JobType.SINGLE,
            compound_name="TestCompound", smiles="CCO",
        )

        service.fail_job(db_session, job.id, "Test error message")

        retrieved = service.get_job(db_session, job.id)
        assert retrieved.status == JobStatus.FAILED
        assert retrieved.error_message == "Test error message"
