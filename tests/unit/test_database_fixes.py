"""
Unit tests for database fixes.

Tests verify actual functionality:
- 1.6: check_pending_compounds optimization
- 1.7: Batch summary aggregation
- 1.9: Status transition validation
- 1.10: Pagination stability
- 1.11: Compound lookup logic
"""
import pytest
import sys
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# Mock rdkit-dependent modules before importing job_service
@pytest.fixture(scope="module", autouse=True)
def mock_rdkit_modules():
    """Mock modules that depend on rdkit to allow importing job_service."""
    mock_modules = {
        'rdkit': MagicMock(),
        'rdkit.Chem': MagicMock(),
        'rdkit.Chem.Descriptors': MagicMock(),
        'rdkit.Chem.AllChem': MagicMock(),
        'rdkit.Chem.inchi': MagicMock(),
        'rdkit.Chem.FilterCatalog': MagicMock(),
        'rdkit.Chem.Draw': MagicMock(),
        'rdkit.DataStructs': MagicMock(),
    }
    with patch.dict(sys.modules, mock_modules):
        yield


@pytest.fixture
def db_engine():
    """Create an in-memory test database engine."""
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
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def job_service(mock_rdkit_modules):
    """Get JobService with rdkit mocked."""
    from backend.services.job_service import JobService
    return JobService()


class TestStatusTransitionValidation:
    """Tests for issue 1.9: Status transition validation in update_progress."""

    def test_valid_pending_to_processing(self, job_service, db_session):
        """Test valid transition: PENDING -> PROCESSING."""
        from backend.models.enums import JobType, JobStatus

        # Create a pending job
        job = job_service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"}
        )
        assert job.status == JobStatus.PENDING

        # Transition to PROCESSING
        result = job_service.update_progress(
            db_session, job.id, 10.0, "Starting...", JobStatus.PROCESSING
        )

        assert result is not None
        assert result.status == JobStatus.PROCESSING

    def test_invalid_pending_to_completed(self, job_service, db_session):
        """Test invalid transition: PENDING -> COMPLETED should be rejected."""
        from backend.models.enums import JobType, JobStatus

        job = job_service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"}
        )

        # Try invalid transition PENDING -> COMPLETED
        result = job_service.update_progress(
            db_session, job.id, 100.0, "Done", JobStatus.COMPLETED
        )

        # Should return None (rejected)
        assert result is None

        # Job should still be PENDING
        db_session.refresh(job)
        assert job.status == JobStatus.PENDING

    def test_invalid_pending_to_failed(self, job_service, db_session):
        """Test invalid transition: PENDING -> FAILED should be rejected."""
        from backend.models.enums import JobType, JobStatus

        job = job_service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"}
        )

        # Try invalid transition PENDING -> FAILED
        result = job_service.update_progress(
            db_session, job.id, 0.0, "Error", JobStatus.FAILED
        )

        assert result is None
        db_session.refresh(job)
        assert job.status == JobStatus.PENDING

    def test_valid_processing_to_completed(self, job_service, db_session):
        """Test valid transition: PROCESSING -> COMPLETED via complete_job."""
        from backend.models.enums import JobType, JobStatus

        job = job_service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"}
        )

        # First go to PROCESSING
        job_service.update_progress(
            db_session, job.id, 10.0, "Starting", JobStatus.PROCESSING
        )

        # Then complete
        result = job_service.complete_job(
            db_session, job.id, "/path/result.zip",
            {"compound_name": "Test", "total_activities": 5}
        )

        assert result is not None
        assert result.status == JobStatus.COMPLETED

    def test_invalid_completed_to_processing(self, job_service, db_session):
        """Test invalid transition: COMPLETED -> PROCESSING should be rejected."""
        from backend.models.enums import JobType, JobStatus

        job = job_service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"}
        )

        # Go to PROCESSING then COMPLETED
        job_service.update_progress(
            db_session, job.id, 10.0, "Starting", JobStatus.PROCESSING
        )
        job_service.complete_job(
            db_session, job.id, "/path/result.zip",
            {"compound_name": "Test"}
        )

        # Try to go back to PROCESSING
        result = job_service.update_progress(
            db_session, job.id, 50.0, "Retry", JobStatus.PROCESSING
        )

        assert result is None  # Should be rejected

    def test_valid_pending_to_cancelled(self, job_service, db_session):
        """Test valid transition: PENDING -> CANCELLED."""
        from backend.models.enums import JobType, JobStatus

        job = job_service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"}
        )

        result = job_service.cancel_job(db_session, job.id)

        assert result is not None
        assert result.status == JobStatus.CANCELLED

    def test_invalid_cancel_completed_job(self, job_service, db_session):
        """Test invalid: cannot cancel a completed job - status remains unchanged."""
        from backend.models.enums import JobType, JobStatus

        job = job_service.create_job(
            db_session,
            JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"}
        )

        # Complete the job
        job_service.update_progress(
            db_session, job.id, 10.0, "Starting", JobStatus.PROCESSING
        )
        job_service.complete_job(
            db_session, job.id, "/path/result.zip", {}
        )

        # Try to cancel - returns the job but doesn't modify it
        result = job_service.cancel_job(db_session, job.id)

        # Job is returned but status remains COMPLETED (not changed to CANCELLED)
        assert result is not None
        assert result.status == JobStatus.COMPLETED  # Status unchanged


class TestBatchSummaryAggregation:
    """Tests for issue 1.7: Efficient batch summary with aggregated query."""

    def test_batch_summary_returns_correct_counts(self, job_service, db_session):
        """Test that get_batch_summary returns correct status counts."""
        from backend.models.enums import JobType, JobStatus

        batch_id = job_service.generate_batch_id()

        # Create jobs with known statuses
        job1 = job_service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Test1", "smiles": "CCO"},
            batch_id=batch_id
        )
        # job2 stays PENDING (created but not stored)
        job_service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Test2", "smiles": "CCCO"},
            batch_id=batch_id
        )
        job3 = job_service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Test3", "smiles": "CCCCO"},
            batch_id=batch_id
        )

        # Update statuses
        job_service.update_progress(
            db_session, job1.id, 50.0, "Processing", JobStatus.PROCESSING
        )
        job_service.complete_job(
            db_session, job3.id, "/path/result.zip",
            {"compound_name": "Test3"}
        )

        summary = job_service.get_batch_summary(db_session, batch_id)

        assert summary["total_jobs"] == 3
        assert summary["processing"] == 1
        assert summary["pending"] == 1
        assert summary["completed"] == 1
        assert summary["failed"] == 0
        assert summary["cancelled"] == 0

    def test_batch_summary_empty_batch(self, job_service, db_session):
        """Test batch summary returns empty dict for non-existent batch."""
        summary = job_service.get_batch_summary(db_session, "nonexistent-batch")
        assert summary == {}

    def test_batch_summary_includes_compound_names(self, job_service, db_session):
        """Test batch summary includes compound names."""
        from backend.models.enums import JobType

        batch_id = job_service.generate_batch_id()

        for i in range(7):
            job_service.create_job(
                db_session, JobType.SINGLE,
                {"compound_name": f"Compound{i}", "smiles": "CCO"},
                batch_id=batch_id
            )

        summary = job_service.get_batch_summary(db_session, batch_id)

        assert "compound_names" in summary
        assert len(summary["compound_names"]) <= 5  # Limited to first 5


class TestCheckPendingCompounds:
    """Tests for issue 1.6: Optimized check_pending_compounds."""

    def test_finds_pending_compounds(self, job_service, db_session):
        """Test that check_pending_compounds finds pending jobs."""
        from backend.models.enums import JobType

        job = job_service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Aspirin", "smiles": "CCO"}
        )

        result = job_service.check_pending_compounds(
            db_session, ["Aspirin", "NotPending"]
        )

        assert "Aspirin" in result
        assert result["Aspirin"] == job.id
        assert "NotPending" not in result

    def test_ignores_completed_compounds(self, job_service, db_session):
        """Test that check_pending_compounds ignores completed jobs."""
        from backend.models.enums import JobType, JobStatus

        job = job_service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "CompletedCompound", "smiles": "CCO"}
        )

        # Complete the job
        job_service.update_progress(
            db_session, job.id, 10.0, "Starting", JobStatus.PROCESSING
        )
        job_service.complete_job(
            db_session, job.id, "/path/result.zip", {}
        )

        result = job_service.check_pending_compounds(
            db_session, ["CompletedCompound"]
        )

        # Should not find completed job
        assert "CompletedCompound" not in result

    def test_empty_list_returns_empty_dict(self, job_service, db_session):
        """Test check_pending_compounds with empty list."""
        result = job_service.check_pending_compounds(db_session, [])
        assert result == {}

    def test_handles_special_characters(self, job_service, db_session):
        """Test check_pending_compounds handles special SQL characters."""
        from backend.models.enums import JobType

        special_name = "Test%Compound_Name"
        job = job_service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": special_name, "smiles": "CCO"}
        )

        result = job_service.check_pending_compounds(
            db_session, [special_name]
        )

        assert special_name in result
        assert result[special_name] == job.id

    def test_pending_lookup_is_case_insensitive(self, job_service, db_session):
        """Pending lookup should match even if submitted casing differs."""
        from backend.models.enums import JobType

        job = job_service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Quercetin", "smiles": "CCO"}
        )

        result = job_service.check_pending_compounds(
            db_session, ["QUERCETIN"]
        )

        assert "QUERCETIN" in result
        assert result["QUERCETIN"] == job.id


class TestCheckExistingCompounds:
    """Tests for case-insensitive existing-compound checks."""

    def test_existing_lookup_is_case_insensitive(self, job_service, db_session):
        """Existing lookup should match even if submitted casing differs."""
        from backend.models.compound import Compound

        db_session.add(Compound(entry_id="entry-quercetin", compound_name="Quercetin"))
        db_session.commit()

        result = job_service.check_existing_compounds(
            db_session, ["QUERCETIN", "NewCompound"]
        )

        assert result["QUERCETIN"] is True
        assert result["NewCompound"] is False


class TestPaginationStability:
    """Tests for issue 1.10: Pagination with stable sort order."""

    def test_pagination_no_duplicates(self, job_service, db_session):
        """Test that paginated results have no duplicates across pages."""
        from backend.models.enums import JobType

        # Create 15 jobs
        created_ids = []
        for i in range(15):
            job = job_service.create_job(
                db_session, JobType.SINGLE,
                {"compound_name": f"Compound{i}", "smiles": "CCO"}
            )
            created_ids.append(job.id)

        # Get all pages
        page1 = job_service.list_jobs(db_session, page=1, page_size=5)
        page2 = job_service.list_jobs(db_session, page=2, page_size=5)
        page3 = job_service.list_jobs(db_session, page=3, page_size=5)

        all_ids = (
            [j.id for j in page1["items"]] +
            [j.id for j in page2["items"]] +
            [j.id for j in page3["items"]]
        )

        # No duplicates
        assert len(all_ids) == len(set(all_ids))
        # Got all jobs
        assert set(all_ids) == set(created_ids)

    def test_pagination_deterministic_order(self, job_service, db_session):
        """Test that pagination returns same order on repeated calls."""
        from backend.models.enums import JobType

        for i in range(10):
            job_service.create_job(
                db_session, JobType.SINGLE,
                {"compound_name": f"Compound{i}", "smiles": "CCO"}
            )

        # Get page 1 twice
        page1_first = job_service.list_jobs(db_session, page=1, page_size=5)
        page1_second = job_service.list_jobs(db_session, page=1, page_size=5)

        ids_first = [j.id for j in page1_first["items"]]
        ids_second = [j.id for j in page1_second["items"]]

        assert ids_first == ids_second



class TestValidTransitionsConstant:
    """Tests for VALID_TRANSITIONS constant structure."""

    def test_all_statuses_have_transitions(self, mock_rdkit_modules):
        """Test that every JobStatus is in VALID_TRANSITIONS."""
        from backend.models.enums import JobStatus
        from backend.services.job_service import VALID_TRANSITIONS

        for status in JobStatus:
            assert status in VALID_TRANSITIONS, f"Missing: {status}"

    def test_terminal_states_have_no_transitions(self, mock_rdkit_modules):
        """Test that terminal states cannot transition."""
        from backend.models.enums import JobStatus
        from backend.services.job_service import VALID_TRANSITIONS

        assert VALID_TRANSITIONS[JobStatus.COMPLETED] == set()
        assert VALID_TRANSITIONS[JobStatus.FAILED] == set()
        assert VALID_TRANSITIONS[JobStatus.CANCELLED] == set()

    def test_pending_can_only_process_or_cancel(self, mock_rdkit_modules):
        """Test PENDING state valid transitions."""
        from backend.models.enums import JobStatus
        from backend.services.job_service import VALID_TRANSITIONS

        allowed = VALID_TRANSITIONS[JobStatus.PENDING]
        assert allowed == {JobStatus.PROCESSING, JobStatus.CANCELLED}

    def test_processing_transitions(self, mock_rdkit_modules):
        """Test PROCESSING state valid transitions."""
        from backend.models.enums import JobStatus
        from backend.services.job_service import VALID_TRANSITIONS

        allowed = VALID_TRANSITIONS[JobStatus.PROCESSING]
        assert JobStatus.COMPLETED in allowed
        assert JobStatus.FAILED in allowed
        assert JobStatus.CANCELLED in allowed
        assert JobStatus.PENDING not in allowed


