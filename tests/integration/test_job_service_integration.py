"""
Integration tests for duplicate detection state transitions in job_service.

Tests use real SQLite to verify submit/resolve/check-duplicates flows
with actual database constraints and compound matching.
"""
import json
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.core.database import Base
from backend.models.database import Compound, Job
from backend.services.job_service import (
    JobService,
    _inchikey_structure_key,
    generate_inchikey,
    get_next_version_names_bulk,
)


@pytest.fixture(scope="module")
def engine():
    """Module-scoped in-memory SQLite engine."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)
    eng.dispose()


@pytest.fixture(autouse=True)
def _clean(engine):
    """Truncate all tables after each test."""
    yield
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


@pytest.fixture
def db(engine):
    """Per-test database session."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def service():
    return JobService()


@pytest.fixture
def seed(db):
    """Factory fixture for seeding Compound rows."""
    def _create(name="TestCompound", smiles="CCO", inchikey=None, **overrides):
        if inchikey is None:
            inchikey = generate_inchikey(smiles)
        defaults = {
            "entry_id": str(uuid.uuid4()),
            "compound_name": name,
            "smiles": smiles,
            "inchikey": inchikey,
            "inchikey_structure_key": _inchikey_structure_key(inchikey),
            "similarity_threshold": 90,
            "activity_types": "EC50,IC50,Kd,Ki",
        }
        defaults.update(overrides)
        comp = Compound(**defaults)
        db.add(comp)
        db.commit()
        db.refresh(comp)
        return comp
    return _create


# ============================================================================
# Submit Job Duplicates
# ============================================================================


class TestSubmitJobDuplicates:
    """Tests for duplicate detection during job submission."""

    def test_submit_no_duplicate(self, service, db):
        """New SMILES creates a job with status=pending."""
        from backend.models.schemas import JobCreate
        from unittest.mock import patch

        request = JobCreate(
            compound_name="NewCompound",
            author_name="Test Author",
            smiles="CCO",
            similarity_threshold=90,
        )
        with patch('backend.core.scheduler.job_scheduler'):
            result = service.submit_job(db, request, session_id="test-session")

        # Should be a JobResponse (not DuplicateFoundResponse)
        assert result.status.value == "pending"
        assert result.compound_name == "NewCompound"

    def test_submit_exact_duplicate(self, service, db, seed):
        """Same SMILES + config returns duplicate_found with config_match=identical."""
        from backend.models.schemas import JobCreate, DuplicateFoundResponse
        from backend.services.job_service import _DEFAULT_ACTIVITY_TYPES

        # Seed with the default activity types so config matches when JobCreate
        # sends activity_types=None (which normalizes to _DEFAULT_ACTIVITY_TYPES)
        seed(
            name="Aspirin",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            activity_types=_DEFAULT_ACTIVITY_TYPES,
        )

        request = JobCreate(
            compound_name="Aspirin",
            author_name="Test Author",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            similarity_threshold=90,
        )
        result = service.submit_job(db, request, session_id="test-session")
        assert isinstance(result, DuplicateFoundResponse)
        assert result.status == "duplicate_found"
        assert result.config_match == "identical"

    def test_submit_structure_duplicate_different_config(self, service, db, seed):
        """Same structure, different threshold returns duplicate_found with config_match != identical."""
        from backend.models.schemas import JobCreate, DuplicateFoundResponse

        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O", similarity_threshold=90)

        request = JobCreate(
            compound_name="Aspirin",
            author_name="Test Author",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            similarity_threshold=70,
        )
        result = service.submit_job(db, request, session_id="test-session")
        assert isinstance(result, DuplicateFoundResponse)
        assert result.status == "duplicate_found"
        assert result.config_match != "identical"

    def test_submit_different_structure(self, service, db, seed):
        """Different SMILES creates job normally."""
        from backend.models.schemas import JobCreate
        from unittest.mock import patch

        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

        request = JobCreate(
            compound_name="Caffeine",
            author_name="Test Author",
            smiles="CCO",  # Different structure
            similarity_threshold=90,
        )
        with patch('backend.core.scheduler.job_scheduler'):
            result = service.submit_job(db, request, session_id="test-session")
        assert result.status.value == "pending"


# ============================================================================
# Resolve Duplicate Action
# ============================================================================


class TestResolveDuplicateAction:
    """Tests for resolve_duplicate_action method."""

    def test_replace_creates_job_with_replace_entry_id(self, service, db, seed):
        """Resolve with action=replace creates a job with replace_entry_id in input_params."""
        from backend.models.schemas import ResolveDuplicateRequest
        from unittest.mock import patch

        comp = seed(name="Ethanol", smiles="CCO")

        request = ResolveDuplicateRequest(
            action="replace",
            smiles="CCO",
            compound_name="Ethanol",
            author_name="Test Author",
            existing_entry_id=comp.entry_id,
        )
        with patch('backend.core.scheduler.job_scheduler'):
            result = service.resolve_duplicate_action(db, request, session_id="test-session")

        assert result.status.value == "pending"

        # Check job has replace_entry_id stored
        job = db.query(Job).filter(Job.id == result.id).first()
        params = json.loads(job.input_params)
        assert params.get("replace_entry_id") == comp.entry_id

        # Original compound should still exist (deferred deletion)
        original = db.query(Compound).filter(Compound.entry_id == comp.entry_id).first()
        assert original is not None

    def test_skip_returns_existing(self, service, db, seed):
        """Resolve with action=skip returns skip response, no job created."""
        from backend.models.schemas import ResolveDuplicateRequest, SkipResponse

        seed(name="Ethanol", smiles="CCO")  # side effect: creates DB row for duplicate detection

        request = ResolveDuplicateRequest(
            action="skip",
            smiles="CCO",
            compound_name="Ethanol",
            author_name="Test Author",
        )
        result = service.resolve_duplicate_action(db, request, session_id="test-session")
        assert isinstance(result, SkipResponse)
        assert result.status == "skipped"

    def test_duplicate_creates_job_with_is_duplicate(self, service, db, seed):
        """Resolve with action=duplicate creates job with is_duplicate in input_params."""
        from backend.models.schemas import ResolveDuplicateRequest
        from unittest.mock import patch

        comp = seed(name="Ethanol", smiles="CCO", activity_types="IC50,Ki")

        request = ResolveDuplicateRequest(
            action="duplicate",
            smiles="CCO",
            compound_name="Ethanol",
            author_name="Test Author",
            existing_entry_id=comp.entry_id,
            activity_types=["EC50", "IC50", "Kd", "Ki"],
        )
        with patch('backend.core.scheduler.job_scheduler'):
            result = service.resolve_duplicate_action(db, request, session_id="test-session")

        assert result.status.value == "pending"
        job = db.query(Job).filter(Job.id == result.id).first()
        params = json.loads(job.input_params)
        assert params.get("is_duplicate") is True

    def test_duplicate_identical_config_blocked(self, service, db, seed):
        """action=duplicate with identical config raises ValueError."""
        from backend.models.schemas import ResolveDuplicateRequest

        comp = seed(name="Ethanol", smiles="CCO")

        request = ResolveDuplicateRequest(
            action="duplicate",
            smiles="CCO",
            compound_name="Ethanol",
            author_name="Test Author",
            existing_entry_id=comp.entry_id,
            similarity_threshold=90,
            activity_types=["EC50", "IC50", "Kd", "Ki"],
        )
        with pytest.raises(ValueError, match="identical configuration"):
            service.resolve_duplicate_action(db, request, session_id="test-session")


# ============================================================================
# Check Duplicates Batch
# ============================================================================


class TestCheckDuplicatesBatch:
    """Tests for check_duplicates_batch via the service layer."""

    def test_batch_no_duplicates(self, service, db):
        """List of new compound names returns empty existing."""
        from backend.models.schemas import CheckDuplicatesRequest

        request = CheckDuplicatesRequest(compound_names=["NewA", "NewB", "NewC"])
        result = service.check_duplicates_batch(db, request)
        assert result.existing == []
        assert set(result.new) == {"NewA", "NewB", "NewC"}

    def test_batch_mixed(self, service, db, seed):
        """Some existing, some new."""
        from backend.models.schemas import CheckDuplicatesRequest

        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

        request = CheckDuplicatesRequest(compound_names=["Aspirin", "NewCompound"])
        result = service.check_duplicates_batch(db, request)
        assert "Aspirin" in result.existing
        assert "NewCompound" in result.new

    def test_batch_config_aware(self, service, db, seed):
        """Structure-based check includes config_match info."""
        from backend.models.schemas import CheckDuplicatesRequest, CompoundStructure

        smiles = "CCO"
        seed(name="Ethanol", smiles=smiles, similarity_threshold=90)

        request = CheckDuplicatesRequest(
            compounds=[
                CompoundStructure(compound_name="EthanolAlias", smiles=smiles),
            ],
            similarity_threshold=70,
            activity_types=["EC50", "IC50", "Kd", "Ki"],
        )
        result = service.check_duplicates_batch(db, request)
        assert len(result.structure_matches) == 1
        match = result.structure_matches[0]
        assert match.config_match == "different_threshold"


# ============================================================================
# Bulk Version Names
# ============================================================================


class TestGetNextVersionNamesBulk:
    """Tests for get_next_version_names_bulk()."""

    def test_bulk_no_conflicts(self, db, seed):
        """Three unique names with no existing compounds return themselves + _v2."""
        result = get_next_version_names_bulk(db, ["Alpha", "Beta", "Gamma"])
        assert result["Alpha"] == "Alpha_v2"
        assert result["Beta"] == "Beta_v2"
        assert result["Gamma"] == "Gamma_v2"

    def test_bulk_with_conflicts(self, db, seed):
        """Existing compounds cause v2/v3 suffixes."""
        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        seed(name="Aspirin_v2", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

        result = get_next_version_names_bulk(db, ["Aspirin", "NewCompound"])
        assert result["Aspirin"] == "Aspirin_v3"
        assert result["NewCompound"] == "NewCompound_v2"

    def test_empty_list(self, db):
        result = get_next_version_names_bulk(db, [])
        assert result == {}

    def test_bulk_name_with_existing_v_suffix(self, db, seed):
        """Input name with _vN suffix is stripped to find true base, then next version computed."""
        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        seed(name="Aspirin_v2", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

        result = get_next_version_names_bulk(db, ["Aspirin_v3"])
        # base="Aspirin", max existing version=v2, so next=v3
        assert result["Aspirin_v3"] == "Aspirin_v3"

    def test_bulk_name_with_v_suffix_higher_exists(self, db, seed):
        """Input with _vN suffix where higher versions already exist."""
        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        seed(name="Aspirin_v2", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        seed(name="Aspirin_v3", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        seed(name="Aspirin_v4", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

        result = get_next_version_names_bulk(db, ["Aspirin_v2"])
        # base="Aspirin", max existing=v4, next=v5
        assert result["Aspirin_v2"] == "Aspirin_v5"

    def test_bulk_all_whitespace_names(self, db):
        """All empty/whitespace/None names return empty dict."""
        result = get_next_version_names_bulk(db, ["", "  ", None])
        assert result == {}

    def test_bulk_mixed_valid_and_empty(self, db, seed):
        """Mixed valid and empty names only return entries for valid names."""
        result = get_next_version_names_bulk(db, ["Aspirin", "", "Caffeine"])
        assert "Aspirin" in result
        assert "Caffeine" in result
        assert "" not in result
