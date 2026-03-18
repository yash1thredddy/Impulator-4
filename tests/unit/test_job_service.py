"""
Unit tests for JobService.
"""
import pytest


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


# ---------------------------------------------------------------------------
# New test classes for uncovered paths
# ---------------------------------------------------------------------------

class TestSafeJsonLoads:
    """Tests for _safe_json_loads() utility function."""

    def test_valid_json(self):
        from backend.services.job_service import _safe_json_loads
        assert _safe_json_loads('{"key": "value"}') == {"key": "value"}

    def test_valid_json_list(self):
        from backend.services.job_service import _safe_json_loads
        assert _safe_json_loads('[1, 2, 3]') == [1, 2, 3]

    def test_json_decode_error(self):
        from backend.services.job_service import _safe_json_loads
        assert _safe_json_loads('not valid json') is None

    def test_json_decode_error_with_default(self):
        from backend.services.job_service import _safe_json_loads
        assert _safe_json_loads('bad json', default={}) == {}

    def test_none_input(self):
        from backend.services.job_service import _safe_json_loads
        assert _safe_json_loads(None) is None

    def test_none_input_with_default(self):
        from backend.services.job_service import _safe_json_loads
        assert _safe_json_loads(None, default=[]) == []

    def test_empty_string(self):
        from backend.services.job_service import _safe_json_loads
        assert _safe_json_loads('') is None

    def test_non_string_type(self):
        """Non-string types like int are caught by the TypeError handler."""
        from backend.services.job_service import _safe_json_loads
        # int is truthy but json.loads(123) raises TypeError
        assert _safe_json_loads(123, default="fallback") == "fallback"


class TestGenerateInchikey:
    """Tests for generate_inchikey() function."""

    def test_valid_smiles_returns_inchikey(self):
        from backend.services.job_service import generate_inchikey
        result = generate_inchikey("CCO")  # Ethanol
        assert result is not None
        # InChIKey is 27 characters with two hyphens
        assert len(result) == 27
        assert result.count("-") == 2

    def test_same_smiles_is_deterministic(self):
        from backend.services.job_service import generate_inchikey
        result1 = generate_inchikey("CCO")
        result2 = generate_inchikey("CCO")
        assert result1 == result2

    def test_invalid_smiles_returns_none(self):
        from backend.services.job_service import generate_inchikey
        result = generate_inchikey("INVALID_SMILES_XYZ999")
        assert result is None

    def test_empty_string_returns_none(self):
        from backend.services.job_service import generate_inchikey
        assert generate_inchikey("") is None

    def test_none_returns_none(self):
        from backend.services.job_service import generate_inchikey
        assert generate_inchikey(None) is None

    def test_whitespace_only_returns_none(self):
        from backend.services.job_service import generate_inchikey
        assert generate_inchikey("   ") is None


class TestGenerateCanonicalSmiles:
    """Tests for generate_canonical_smiles() function."""

    def test_valid_smiles_returns_canonical(self):
        from backend.services.job_service import generate_canonical_smiles
        result = generate_canonical_smiles("OCC")  # Non-canonical ethanol
        assert result is not None
        assert result == "CCO"  # Canonical form

    def test_invalid_smiles_returns_none(self):
        from backend.services.job_service import generate_canonical_smiles
        assert generate_canonical_smiles("INVALID_XYZ") is None

    def test_empty_string_returns_none(self):
        from backend.services.job_service import generate_canonical_smiles
        assert generate_canonical_smiles("") is None

    def test_none_returns_none(self):
        from backend.services.job_service import generate_canonical_smiles
        assert generate_canonical_smiles(None) is None

    def test_whitespace_only_returns_none(self):
        from backend.services.job_service import generate_canonical_smiles
        assert generate_canonical_smiles("   ") is None


class TestGetActiveJobsDetailed:
    """Detailed tests for get_active_jobs() with real DB."""

    @pytest.fixture
    def db_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.core.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
        engine.dispose()

    @pytest.fixture
    def service(self):
        from backend.services.job_service import JobService
        return JobService()

    def test_returns_recently_completed_jobs(self, service, db_session):
        """Completed jobs within the time window should appear."""
        from backend.models.database import JobType, JobStatus
        from datetime import datetime, timezone

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            session_id="test-session",
        )
        # Move to completed with recent completed_at
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        job.result_summary = '{"compound_name": "Aspirin"}'
        db_session.commit()

        result = service.get_active_jobs(db_session, session_id="test-session")
        assert len(result) == 1
        assert result[0]["status"].upper() == "COMPLETED"
        assert result[0]["compound_name"] == "Aspirin"

    def test_returns_recently_failed_jobs(self, service, db_session):
        """Failed jobs within the 20-min window should appear."""
        from backend.models.database import JobType, JobStatus
        from datetime import datetime, timezone

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "BadCompound", "smiles": "XYZ"},
            session_id="test-session",
        )
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error_message = "Processing failed"
        db_session.commit()

        result = service.get_active_jobs(db_session, session_id="test-session")
        assert len(result) == 1
        assert result[0]["status"].upper() == "FAILED"
        assert result[0]["error_message"] == "Processing failed"

    def test_completed_sorted_before_active(self, service, db_session):
        """Completed/failed jobs should sort before pending/processing."""
        from backend.models.database import JobType, JobStatus
        from datetime import datetime, timezone

        # Create a pending job
        _ = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Pending", "smiles": "CCO"},
            session_id="test-session",
        )

        # Create a completed job
        completed_job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Done", "smiles": "CC"},
            session_id="test-session",
        )
        completed_job.status = JobStatus.COMPLETED
        completed_job.completed_at = datetime.now(timezone.utc)
        completed_job.result_summary = '{"compound_name": "Done"}'
        db_session.commit()

        result = service.get_active_jobs(db_session, session_id="test-session")
        assert len(result) == 2
        # Completed should come first (sort key 0), pending second (sort key 1)
        assert result[0]["status"].upper() == "COMPLETED"
        assert result[1]["status"].upper() == "PENDING"

    def test_failed_job_with_cascade_results(self, service, db_session):
        """Failed jobs with cascade_results should include them in the response."""
        from backend.models.database import JobType, JobStatus
        from datetime import datetime, timezone
        import json

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "CascadeTest", "smiles": "CCO"},
            session_id="test-session",
        )
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error_message = "No similar compounds"
        job.result_summary = json.dumps({
            "cascade_results": [{"threshold": 70, "count": 5}]
        })
        db_session.commit()

        result = service.get_active_jobs(db_session, session_id="test-session")
        assert len(result) == 1
        assert "cascade_results" in result[0]
        assert result[0]["cascade_results"] == [{"threshold": 70, "count": 5}]
        assert "input_params" in result[0]


class TestCancelJobTerminalStates:
    """Tests for cancel_job() on terminal-state jobs."""

    @pytest.fixture
    def db_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.core.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
        engine.dispose()

    @pytest.fixture
    def service(self):
        from backend.services.job_service import JobService
        return JobService()

    def test_cannot_cancel_completed_job(self, service, db_session):
        """Cancelling a COMPLETED job should return the job unchanged."""
        from backend.models.database import JobType, JobStatus
        from datetime import datetime, timezone

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Done", "smiles": "CCO"},
        )
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        db_session.commit()

        result = service.cancel_job(db_session, job.id)
        assert result is not None
        assert result.status == JobStatus.COMPLETED  # Unchanged

    def test_cannot_cancel_failed_job(self, service, db_session):
        """Cancelling a FAILED job should return the job unchanged."""
        from backend.models.database import JobType, JobStatus
        from datetime import datetime, timezone

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Oops", "smiles": "CCO"},
        )
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        db_session.commit()

        result = service.cancel_job(db_session, job.id)
        assert result is not None
        assert result.status == JobStatus.FAILED  # Unchanged

    def test_cannot_cancel_already_cancelled_job(self, service, db_session):
        """Cancelling an already CANCELLED job should return it unchanged."""
        from backend.models.database import JobType, JobStatus
        from datetime import datetime, timezone

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Nope", "smiles": "CCO"},
        )
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        db_session.commit()

        result = service.cancel_job(db_session, job.id)
        assert result is not None
        assert result.status == JobStatus.CANCELLED

    def test_cancel_nonexistent_job_returns_none(self, service, db_session):
        """Cancelling a job that doesn't exist should return None."""
        result = service.cancel_job(db_session, "no-such-id")
        assert result is None


class TestFailJobEdgeCases:
    """Additional tests for fail_job() edge cases."""

    @pytest.fixture
    def db_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.core.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
        engine.dispose()

    @pytest.fixture
    def service(self):
        from backend.services.job_service import JobService
        return JobService()

    def test_fail_job_not_found_returns_none(self, service, db_session):
        """Failing a nonexistent job should return None."""
        result = service.fail_job(db_session, "nonexistent-id", "error msg")
        assert result is None

    def test_fail_job_with_cascade_results(self, service, db_session):
        """fail_job() should store cascade_results in result_summary JSON."""
        from backend.models.database import JobType, JobStatus
        import json

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"},
        )
        cascade = [{"threshold": 70, "count": 3}, {"threshold": 60, "count": 8}]
        service.fail_job(db_session, job.id, "No results", cascade_results=cascade)

        retrieved = service.get_job(db_session, job.id)
        assert retrieved.status == JobStatus.FAILED
        summary = json.loads(retrieved.result_summary)
        assert summary["cascade_results"] == cascade

    def test_fail_job_without_cascade_results(self, service, db_session):
        """fail_job() without cascade_results should leave result_summary as None."""
        from backend.models.database import JobType

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Test", "smiles": "CCO"},
        )
        service.fail_job(db_session, job.id, "Something broke")

        retrieved = service.get_job(db_session, job.id)
        assert retrieved.result_summary is None


class TestCheckPendingCompounds:
    """Tests for check_pending_compounds()."""

    @pytest.fixture
    def db_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.core.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
        engine.dispose()

    @pytest.fixture
    def service(self):
        from backend.services.job_service import JobService
        return JobService()

    def test_empty_list_returns_empty(self, service, db_session):
        """Empty compound_names list should return empty dict."""
        result = service.check_pending_compounds(db_session, [])
        assert result == {}

    def test_normalizes_names_case_insensitive(self, service, db_session):
        """Name matching should be case-insensitive."""
        from backend.models.database import JobType

        # Create a pending job with compound name "Aspirin"
        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        )

        # Search with different casing
        result = service.check_pending_compounds(db_session, ["aspirin"])
        assert "aspirin" in result
        assert result["aspirin"] == job.id

    def test_normalizes_names_with_whitespace(self, service, db_session):
        """Name matching should strip whitespace."""
        from backend.models.database import JobType

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        )

        result = service.check_pending_compounds(db_session, ["  Caffeine  "])
        assert "  Caffeine  " in result
        assert result["  Caffeine  "] == job.id

    def test_no_match_returns_empty(self, service, db_session):
        """When no pending jobs match, return empty dict."""
        from backend.models.database import JobType

        # Create a pending job for a different compound
        service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        )

        result = service.check_pending_compounds(db_session, ["Caffeine"])
        assert result == {}

    def test_completed_jobs_not_matched(self, service, db_session):
        """Completed jobs should NOT be returned as pending."""
        from backend.models.database import JobType, JobStatus
        from datetime import datetime, timezone

        job = service.create_job(
            db_session, JobType.SINGLE,
            {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        )
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        db_session.commit()

        result = service.check_pending_compounds(db_session, ["Aspirin"])
        assert result == {}


# ---------------------------------------------------------------------------
# Duplicate detection pure logic tests (TEST-02)
# ---------------------------------------------------------------------------


class TestComputeConfigMatch:
    """Tests for _compute_config_match() -- pure config comparison logic."""

    def _make_compound(self, threshold=90, activity_types="EC50,IC50,Kd,Ki"):
        """Build a minimal Compound-like object for testing."""
        from backend.models.database import Compound
        return Compound(
            entry_id="test-entry",
            compound_name="Test",
            similarity_threshold=threshold,
            activity_types=activity_types,
        )

    def test_identical_config(self):
        from backend.services.job_service import _compute_config_match, _normalize_activity_types
        comp = self._make_compound(threshold=90, activity_types="EC50,IC50,Kd,Ki")
        submitted_at = _normalize_activity_types(["EC50", "IC50", "Kd", "Ki"])
        assert _compute_config_match(comp, 90, submitted_at) == "identical"

    def test_different_threshold(self):
        from backend.services.job_service import _compute_config_match, _normalize_activity_types
        comp = self._make_compound(threshold=90, activity_types="EC50,IC50,Kd,Ki")
        submitted_at = _normalize_activity_types(["EC50", "IC50", "Kd", "Ki"])
        assert _compute_config_match(comp, 70, submitted_at) == "different_threshold"

    def test_different_activities(self):
        from backend.services.job_service import _compute_config_match, _normalize_activity_types
        comp = self._make_compound(threshold=90, activity_types="EC50,IC50,Kd,Ki")
        submitted_at = _normalize_activity_types(["IC50", "Ki"])
        assert _compute_config_match(comp, 90, submitted_at) == "different_activities"

    def test_different_both(self):
        from backend.services.job_service import _compute_config_match, _normalize_activity_types
        comp = self._make_compound(threshold=90, activity_types="EC50,IC50,Kd,Ki")
        submitted_at = _normalize_activity_types(["IC50"])
        assert _compute_config_match(comp, 70, submitted_at) == "different_both"

    def test_none_activity_types(self):
        """None stored activity_types should use default and still compare."""
        from backend.services.job_service import _compute_config_match, _DEFAULT_ACTIVITY_TYPES
        comp = self._make_compound(threshold=90, activity_types=None)
        assert _compute_config_match(comp, 90, _DEFAULT_ACTIVITY_TYPES) == "identical"

    def test_empty_activity_types(self):
        """Empty string activity_types should use default."""
        from backend.services.job_service import _compute_config_match, _DEFAULT_ACTIVITY_TYPES
        comp = self._make_compound(threshold=90, activity_types="")
        assert _compute_config_match(comp, 90, _DEFAULT_ACTIVITY_TYPES) == "identical"


class TestNormalizeActivityTypes:
    """Tests for _normalize_activity_types() -- list to sorted string."""

    def test_normal_list(self):
        from backend.services.job_service import _normalize_activity_types
        assert _normalize_activity_types(["EC50", "IC50"]) == "EC50,IC50"

    def test_sorts_alphabetically(self):
        from backend.services.job_service import _normalize_activity_types
        assert _normalize_activity_types(["Ki", "EC50", "IC50"]) == "EC50,IC50,Ki"

    def test_none_returns_default(self):
        from backend.services.job_service import _normalize_activity_types, _DEFAULT_ACTIVITY_TYPES
        assert _normalize_activity_types(None) == _DEFAULT_ACTIVITY_TYPES

    def test_empty_list_returns_default(self):
        from backend.services.job_service import _normalize_activity_types, _DEFAULT_ACTIVITY_TYPES
        assert _normalize_activity_types([]) == _DEFAULT_ACTIVITY_TYPES

    def test_duplicates_preserved(self):
        """List dedup is caller's responsibility; normalize just sorts and joins."""
        from backend.services.job_service import _normalize_activity_types
        result = _normalize_activity_types(["EC50", "EC50"])
        assert result == "EC50,EC50"

    def test_case_preservation(self):
        from backend.services.job_service import _normalize_activity_types
        result = _normalize_activity_types(["ec50", "IC50"])
        assert "ec50" in result
        assert "IC50" in result


class TestNormalizeActivityTypesStr:
    """Tests for _normalize_activity_types_str() -- stored string normalization."""

    def test_normal_string(self):
        from backend.services.job_service import _normalize_activity_types_str
        assert _normalize_activity_types_str("IC50,EC50") == "EC50,IC50"

    def test_none_returns_default(self):
        from backend.services.job_service import _normalize_activity_types_str, _DEFAULT_ACTIVITY_TYPES
        assert _normalize_activity_types_str(None) == _DEFAULT_ACTIVITY_TYPES

    def test_empty_string_returns_default(self):
        from backend.services.job_service import _normalize_activity_types_str, _DEFAULT_ACTIVITY_TYPES
        assert _normalize_activity_types_str("") == _DEFAULT_ACTIVITY_TYPES

    def test_extra_spaces(self):
        from backend.services.job_service import _normalize_activity_types_str
        assert _normalize_activity_types_str(" EC50 , IC50 ") == "EC50,IC50"


class TestInchikeyStructureKey:
    """Tests for _inchikey_structure_key() -- protonation-insensitive key."""

    def test_full_inchikey(self):
        from backend.services.job_service import _inchikey_structure_key
        result = _inchikey_structure_key("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        assert result == "BSYNRYMUTXBXSQ-UHFFFAOYSA"

    def test_none_returns_none(self):
        from backend.services.job_service import _inchikey_structure_key
        assert _inchikey_structure_key(None) is None

    def test_empty_string_returns_none(self):
        from backend.services.job_service import _inchikey_structure_key
        assert _inchikey_structure_key("") is None

    def test_short_inchikey_single_block(self):
        """Malformed InChIKey with no hyphens returns the original string."""
        from backend.services.job_service import _inchikey_structure_key
        result = _inchikey_structure_key("NOHYPHENS")
        assert result == "NOHYPHENS"

    def test_case_insensitive(self):
        from backend.services.job_service import _inchikey_structure_key
        result = _inchikey_structure_key("bsynrymutxbxsq-uhfffaoysa-n")
        assert result == "BSYNRYMUTXBXSQ-UHFFFAOYSA"


class TestGetNextVersionName:
    """Tests for get_next_version_name() -- needs real DB for compound queries."""

    @pytest.fixture
    def db_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool
        from backend.core.database import Base
        from backend.models.database import Compound  # noqa: F401

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
        engine.dispose()

    def _seed(self, db, name):
        from backend.models.database import Compound
        import uuid
        comp = Compound(entry_id=str(uuid.uuid4()), compound_name=name)
        db.add(comp)
        db.commit()

    def test_first_version(self, db_session):
        from backend.services.job_service import get_next_version_name
        result = get_next_version_name(db_session, "Aspirin")
        assert result == "Aspirin_v2"

    def test_second_version(self, db_session):
        from backend.services.job_service import get_next_version_name
        self._seed(db_session, "Aspirin")
        result = get_next_version_name(db_session, "Aspirin")
        assert result == "Aspirin_v2"

    def test_third_version(self, db_session):
        from backend.services.job_service import get_next_version_name
        self._seed(db_session, "Aspirin")
        self._seed(db_session, "Aspirin_v2")
        result = get_next_version_name(db_session, "Aspirin")
        assert result == "Aspirin_v3"

    def test_skip_version(self, db_session):
        from backend.services.job_service import get_next_version_name
        self._seed(db_session, "Aspirin")
        self._seed(db_session, "Aspirin_v3")
        result = get_next_version_name(db_session, "Aspirin")
        assert result == "Aspirin_v4"


class TestBuildConfigDiff:
    """Tests for _build_config_diff() -- produces diff dict or None."""

    def _make_compound(self, threshold=90, activity_types="EC50,IC50,Kd,Ki"):
        from backend.models.database import Compound
        return Compound(
            entry_id="test",
            compound_name="Test",
            similarity_threshold=threshold,
            activity_types=activity_types,
        )

    def test_identical_returns_none(self):
        from backend.services.job_service import _build_config_diff, _normalize_activity_types
        comp = self._make_compound()
        submitted_at = _normalize_activity_types(["EC50", "IC50", "Kd", "Ki"])
        assert _build_config_diff(comp, 90, submitted_at) is None

    def test_different_threshold(self):
        from backend.services.job_service import _build_config_diff, _normalize_activity_types
        comp = self._make_compound(threshold=90)
        submitted_at = _normalize_activity_types(["EC50", "IC50", "Kd", "Ki"])
        diff = _build_config_diff(comp, 70, submitted_at)
        assert diff is not None
        assert diff["similarity_threshold"]["existing"] == 90
        assert diff["similarity_threshold"]["submitted"] == 70

    def test_different_activities(self):
        from backend.services.job_service import _build_config_diff, _normalize_activity_types
        comp = self._make_compound(activity_types="EC50,IC50,Kd,Ki")
        submitted_at = _normalize_activity_types(["IC50", "Ki"])
        diff = _build_config_diff(comp, 90, submitted_at)
        assert diff is not None
        assert "IC50" in diff["activity_types"]["submitted"]

    def test_different_both(self):
        from backend.services.job_service import _build_config_diff, _normalize_activity_types
        comp = self._make_compound(threshold=90, activity_types="EC50,IC50,Kd,Ki")
        submitted_at = _normalize_activity_types(["IC50"])
        diff = _build_config_diff(comp, 70, submitted_at)
        assert diff is not None
        assert diff["similarity_threshold"]["submitted"] == 70


class TestBuildExistingAtThreshold:
    """Tests for _build_existing_at_threshold() -- builds ExistingCompoundAtThreshold from ORM object."""

    def _make_compound(self, **overrides):
        from backend.models.database import Compound
        defaults = {
            "entry_id": "test-entry-001",
            "compound_name": "TestCompound",
            "similarity_threshold": 90,
            "activity_types": "EC50,IC50,Kd,Ki",
            "imp_score": None,
            "processed_at": None,
            "author_name": None,
        }
        defaults.update(overrides)
        return Compound(**defaults)

    def test_identical_config_returns_identical_match(self):
        from backend.services.job_service import _build_existing_at_threshold
        from backend.models.schemas import ExistingCompoundAtThreshold

        comp = self._make_compound(
            entry_id="entry-abc",
            compound_name="Aspirin",
            similarity_threshold=90,
            activity_types="EC50,IC50,Kd,Ki",
        )
        result = _build_existing_at_threshold(comp, 90, "EC50,IC50,Kd,Ki")

        assert isinstance(result, ExistingCompoundAtThreshold)
        assert result.config_match == "identical"
        assert result.config_diff is None
        assert result.entry_id == "entry-abc"
        assert result.compound_name == "Aspirin"
        assert result.similarity_threshold == 90
        # Activity types are normalized (sorted CSV)
        assert result.activity_types == "EC50,IC50,Kd,Ki"

    def test_different_threshold_returns_diff(self):
        from backend.services.job_service import _build_existing_at_threshold

        comp = self._make_compound(similarity_threshold=90)
        result = _build_existing_at_threshold(comp, 70, "EC50,IC50,Kd,Ki")

        assert result.config_match == "different_threshold"
        assert result.config_diff is not None
        assert result.config_diff["similarity_threshold"]["existing"] == 90
        assert result.config_diff["similarity_threshold"]["submitted"] == 70

    def test_different_activities_returns_diff(self):
        from backend.services.job_service import _build_existing_at_threshold

        comp = self._make_compound(activity_types="EC50,IC50,Kd,Ki")
        result = _build_existing_at_threshold(comp, 90, "IC50,Ki")

        assert result.config_match == "different_activities"
        assert result.config_diff is not None
        assert "activity_types" in result.config_diff

    def test_processed_at_iso_format(self):
        from datetime import datetime, timezone
        from backend.services.job_service import _build_existing_at_threshold

        comp = self._make_compound(
            processed_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        result = _build_existing_at_threshold(comp, 90, "EC50,IC50,Kd,Ki")

        assert isinstance(result.processed_at, str)
        assert "2026-01-15" in result.processed_at

    def test_none_processed_at(self):
        from backend.services.job_service import _build_existing_at_threshold

        comp = self._make_compound(processed_at=None)
        result = _build_existing_at_threshold(comp, 90, "EC50,IC50,Kd,Ki")

        assert result.processed_at is None

    def test_imp_score_and_author_passed_through(self):
        from backend.services.job_service import _build_existing_at_threshold

        comp = self._make_compound(imp_score=0.85, author_name="Dr. Test")
        result = _build_existing_at_threshold(comp, 90, "EC50,IC50,Kd,Ki")

        assert result.imp_score == 0.85
        assert result.author_name == "Dr. Test"
