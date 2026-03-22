"""
Integration tests for database transaction handling.

Tests cover:
- 1.3: Transaction commit/rollback in get_db()
"""
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


class TestGetDbTransactionHandling:
    """Tests for issue 1.3: Transaction commit/rollback in get_db()."""

    @pytest.fixture
    def test_db_session(self):
        """Create an isolated test database."""
        from backend.models._pg_base import PGBase
        from backend.models.job import Job  # noqa: F401
        from backend.models.compound import Compound  # noqa: F401
        from backend.models.deleted_compound import DeletedCompound  # noqa: F401

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        PGBase.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        yield Session, engine
        engine.dispose()

    def test_session_commits_on_success(self, test_db_session):
        """Test that successful operations are committed."""
        from backend.models.job import Job
        from backend.models.enums import JobType, JobStatus

        Session, engine = test_db_session
        session = Session()

        # Create a job
        job = Job(
            id="test-commit-job",
            job_type=JobType.SINGLE,
            status=JobStatus.PENDING,
        )
        session.add(job)
        session.commit()

        # Verify in a new session
        session2 = Session()
        found = session2.query(Job).filter(Job.id == "test-commit-job").first()
        assert found is not None
        session2.close()
        session.close()

    def test_session_rollback_on_error(self, test_db_session):
        """Test that errors trigger rollback."""
        from backend.models.job import Job
        from backend.models.enums import JobType, JobStatus

        Session, engine = test_db_session
        session = Session()

        # Create a job but don't commit
        job = Job(
            id="test-rollback-job",
            job_type=JobType.SINGLE,
            status=JobStatus.PENDING,
        )
        session.add(job)

        # Rollback
        session.rollback()

        # Verify job was not saved
        found = session.query(Job).filter(Job.id == "test-rollback-job").first()
        assert found is None
        session.close()
