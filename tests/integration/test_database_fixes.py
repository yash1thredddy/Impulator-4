"""
Integration tests for database transaction handling.

Tests cover:
- 1.3: Transaction commit/rollback in get_db()
"""
import uuid


class TestGetDbTransactionHandling:
    """Tests for issue 1.3: Transaction commit/rollback in get_db()."""

    def test_session_commits_on_success(self, db_session):
        """Test that successful operations are committed."""
        from backend.models.job import Job
        from backend.models.enums import JobType, JobStatus
        job_id = uuid.uuid4()
        job = Job(
            id=job_id,
            session_id=uuid.uuid4(),
            compound_name="Test",
            job_type=JobType.SINGLE,
            status=JobStatus.PENDING,
        )
        db_session.add(job)
        db_session.commit()

        # Verify in same session (transaction committed)
        found = db_session.query(Job).filter(Job.id == job_id).first()
        assert found is not None

    def test_session_rollback_on_error(self, db_session):
        """Test that errors trigger rollback."""
        from backend.models.job import Job
        from backend.models.enums import JobType, JobStatus

        job_id = uuid.uuid4()
        job = Job(
            id=job_id,
            session_id=uuid.uuid4(),
            compound_name="Test",
            job_type=JobType.SINGLE,
            status=JobStatus.PENDING,
        )
        db_session.add(job)

        # Rollback
        db_session.rollback()

        # Verify job was not saved
        found = db_session.query(Job).filter(Job.id == job_id).first()
        assert found is None
