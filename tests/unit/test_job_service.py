"""
Unit tests for JobService.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestJobService:
    """Tests for JobService class."""

    @pytest.fixture
    def service(self):
        """Create a JobService instance."""
        from backend.services.job_service import JobService
        return JobService()

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = MagicMock()
        db.add = MagicMock()
        db.commit = MagicMock()
        db.refresh = MagicMock()
        db.query = MagicMock()
        return db

    def test_create_job(self, service, mock_db):
        """Test job creation."""
        from backend.models.database import JobType

        job = service.create_job(
            mock_db,
            JobType.SINGLE,
            {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        assert job is not None

    def test_get_job_not_found(self, service, mock_db):
        """Test getting a job that doesn't exist."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = service.get_job(mock_db, "nonexistent")
        assert result is None

    def test_get_active_jobs(self, service, mock_db):
        """Test getting active jobs."""
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        result = service.get_active_jobs(mock_db)
        assert isinstance(result, list)


class TestJobServiceWithRealDB:
    """Tests with real in-memory database."""

    @pytest.fixture
    def db_session(self):
        """Create an in-memory test database session."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.core.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    @pytest.fixture
    def service(self):
        """Create a JobService instance."""
        from backend.services.job_service import JobService
        return JobService()

    def test_create_and_get_job(self, service, db_session):
        """Test creating and retrieving a job."""
        from backend.models.database import JobType

        job = service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "TestCompound", "smiles": "CCO"}
        )

        retrieved = service.get_job(db_session, job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

    def test_update_progress(self, service, db_session):
        """Test updating job progress."""
        from backend.models.database import JobType, JobStatus

        job = service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "TestCompound", "smiles": "CCO"}
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
        from backend.models.database import JobType, JobStatus

        job = service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "TestCompound", "smiles": "CCO"}
        )

        service.fail_job(db_session, job.id, "Test error message")

        retrieved = service.get_job(db_session, job.id)
        assert retrieved.status == JobStatus.FAILED
        assert retrieved.error_message == "Test error message"
