"""
Integration tests for duplicate detection state transitions in job_service.

Tests use real Postgres to verify submit/resolve/check-duplicates flows
with actual database constraints and compound matching.
"""
import uuid

import pytest
from unittest.mock import patch

from backend.models.compound import Compound
from backend.models.job import Job
from backend.services.job_service import (
    JobService,
    _inchikey_structure_key,
    generate_inchikey,
    get_next_version_names_bulk,
)


@pytest.fixture
def service():
    return JobService()


@pytest.fixture
def seed(db_session):
    """Factory fixture for seeding Compound rows."""
    from datetime import datetime, timezone

    def _create(name="TestCompound", smiles="CCO", inchikey=None, **overrides):
        if inchikey is None:
            inchikey = generate_inchikey(smiles)
        defaults = {
            "entry_id": uuid.uuid4(),
            "compound_name": name,
            "smiles": smiles,
            "inchikey": inchikey,
            "inchikey_structure_key": _inchikey_structure_key(inchikey),
            "similarity_threshold": 90,
            "activity_types": ["EC50", "IC50", "Kd", "Ki"],
            "processed_at": datetime.now(timezone.utc),
        }
        defaults.update(overrides)
        comp = Compound(**defaults)
        db_session.add(comp)
        db_session.commit()
        db_session.refresh(comp)
        return comp
    return _create


# ============================================================================
# Submit Job Duplicates
# ============================================================================


class TestSubmitJobDuplicates:
    """Tests for duplicate detection during job submission."""

    def test_submit_no_duplicate(self, service, db_session):
        """New SMILES creates a job with status=pending."""
        from backend.models.schemas import JobCreate

        request = JobCreate(
            compound_name="NewCompound",
            author_name="Test Author",
            smiles="CCO",
            similarity_threshold=90,
        )
        with patch('backend.core.scheduler.trigger'):
            result = service.submit_job(db_session, request, session_id="a1234567-1234-4123-8123-123456789012")

        # Should be a JobResponse (not DuplicateFoundResponse)
        assert result.status.value == "pending"
        assert result.compound_name == "NewCompound"

    def test_submit_exact_duplicate(self, service, db_session, seed):
        """Same SMILES + config returns duplicate_found with config_match=identical."""
        from backend.models.schemas import JobCreate, DuplicateFoundResponse
        from backend.services.job_service import _DEFAULT_ACTIVITY_TYPES

        # Seed with the default activity types so config matches when JobCreate
        # sends activity_types=None (which normalizes to _DEFAULT_ACTIVITY_TYPES)
        seed(
            name="Aspirin",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            activity_types=sorted(_DEFAULT_ACTIVITY_TYPES.split(",")),
        )

        request = JobCreate(
            compound_name="Aspirin",
            author_name="Test Author",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            similarity_threshold=90,
        )
        result = service.submit_job(db_session, request, session_id="a1234567-1234-4123-8123-123456789012")
        assert isinstance(result, DuplicateFoundResponse)
        assert result.status == "duplicate_found"
        assert result.config_match == "identical"

    def test_submit_structure_duplicate_different_config(self, service, db_session, seed):
        """Same structure, different threshold returns duplicate_found with config_match != identical."""
        from backend.models.schemas import JobCreate, DuplicateFoundResponse

        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O", similarity_threshold=90)

        request = JobCreate(
            compound_name="Aspirin",
            author_name="Test Author",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            similarity_threshold=70,
        )
        result = service.submit_job(db_session, request, session_id="a1234567-1234-4123-8123-123456789012")
        assert isinstance(result, DuplicateFoundResponse)
        assert result.status == "duplicate_found"
        assert result.config_match != "identical"

    def test_submit_different_structure(self, service, db_session, seed):
        """Different SMILES creates job normally."""
        from backend.models.schemas import JobCreate

        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

        request = JobCreate(
            compound_name="Caffeine",
            author_name="Test Author",
            smiles="CCO",  # Different structure
            similarity_threshold=90,
        )
        with patch('backend.core.scheduler.trigger'):
            result = service.submit_job(db_session, request, session_id="a1234567-1234-4123-8123-123456789012")
        assert result.status.value == "pending"


# ============================================================================
# Resolve Duplicate Action
# ============================================================================


class TestResolveDuplicateAction:
    """Tests for resolve_duplicate_action method."""

    def test_replace_creates_job_with_replace_entry_id(self, service, db_session, seed):
        """Resolve with action=replace creates a job with replace_entry_id in result_summary."""
        from backend.models.schemas import ResolveDuplicateRequest

        comp = seed(name="Ethanol", smiles="CCO")

        request = ResolveDuplicateRequest(
            action="replace",
            smiles="CCO",
            compound_name="Ethanol",
            author_name="Test Author",
            existing_entry_id=comp.entry_id,
        )
        with patch('backend.core.scheduler.trigger'):
            result = service.resolve_duplicate_action(db_session, request, session_id="a1234567-1234-4123-8123-123456789012")

        assert result.status.value == "pending"

        # Check job has replace_entry_id stored in result_summary
        job = db_session.query(Job).filter(Job.id == result.id).first()
        summary = job.result_summary or {}
        assert summary.get("replace_entry_id") == str(comp.entry_id)

        # Original compound should still exist (deferred deletion)
        original = db_session.query(Compound).filter(Compound.entry_id == comp.entry_id).first()
        assert original is not None

    def test_skip_returns_existing(self, service, db_session, seed):
        """Resolve with action=skip returns skip response, no job created."""
        from backend.models.schemas import ResolveDuplicateRequest, SkipResponse

        seed(name="Ethanol", smiles="CCO")  # side effect: creates DB row for duplicate detection

        request = ResolveDuplicateRequest(
            action="skip",
            smiles="CCO",
            compound_name="Ethanol",
            author_name="Test Author",
        )
        result = service.resolve_duplicate_action(db_session, request, session_id="a1234567-1234-4123-8123-123456789012")
        assert isinstance(result, SkipResponse)
        assert result.status == "skipped"

    def test_duplicate_creates_job_with_parent_id(self, service, db_session, seed):
        """Resolve with action=duplicate creates job with parent_id in result_summary."""
        from backend.models.schemas import ResolveDuplicateRequest

        comp = seed(name="Ethanol", smiles="CCO", activity_types=["IC50", "Ki"])

        request = ResolveDuplicateRequest(
            action="duplicate",
            smiles="CCO",
            compound_name="Ethanol",
            author_name="Test Author",
            existing_entry_id=comp.entry_id,
            activity_types=["EC50", "IC50", "Kd", "Ki"],
        )
        with patch('backend.core.scheduler.trigger'):
            result = service.resolve_duplicate_action(db_session, request, session_id="a1234567-1234-4123-8123-123456789012")

        assert result.status.value == "pending"
        job = db_session.query(Job).filter(Job.id == result.id).first()
        summary = job.result_summary or {}
        # New API uses parent_id_for_new instead of is_duplicate
        assert summary.get("parent_id_for_new") is not None

    def test_duplicate_identical_config_blocked(self, service, db_session, seed):
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
            service.resolve_duplicate_action(db_session, request, session_id="a1234567-1234-4123-8123-123456789012")


# ============================================================================
# Check Duplicates Batch
# ============================================================================


class TestCheckDuplicatesBatch:
    """Tests for check_duplicates_batch via the service layer."""

    def test_batch_no_duplicates(self, service, db_session):
        """List of new compound names returns empty existing."""
        from backend.models.schemas import CheckDuplicatesRequest

        request = CheckDuplicatesRequest(compound_names=["NewA", "NewB", "NewC"])
        result = service.check_duplicates_batch(db_session, request)
        assert result.existing == []
        assert set(result.new) == {"NewA", "NewB", "NewC"}

    def test_batch_mixed(self, service, db_session, seed):
        """Some existing, some new."""
        from backend.models.schemas import CheckDuplicatesRequest

        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

        request = CheckDuplicatesRequest(compound_names=["Aspirin", "NewCompound"])
        result = service.check_duplicates_batch(db_session, request)
        assert "Aspirin" in result.existing
        assert "NewCompound" in result.new

    def test_batch_config_aware(self, service, db_session, seed):
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
        result = service.check_duplicates_batch(db_session, request)
        assert len(result.structure_matches) == 1
        match = result.structure_matches[0]
        assert match.config_match == "different_threshold"


# ============================================================================
# Bulk Version Names
# ============================================================================


class TestGetNextVersionNamesBulk:
    """Tests for get_next_version_names_bulk()."""

    def test_bulk_no_conflicts(self, db_session, seed):
        """Three unique names with no existing compounds return themselves + _v2."""
        result = get_next_version_names_bulk(db_session, ["Alpha", "Beta", "Gamma"])
        assert result["Alpha"] == "Alpha_v2"
        assert result["Beta"] == "Beta_v2"
        assert result["Gamma"] == "Gamma_v2"

    def test_bulk_with_conflicts(self, db_session, seed):
        """Existing compounds cause v2/v3 suffixes."""
        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        seed(name="Aspirin_v2", smiles="CC(=O)OC1=CC=CC=C1C(=O)O", entry_id=uuid.uuid4())

        result = get_next_version_names_bulk(db_session, ["Aspirin", "NewCompound"])
        assert result["Aspirin"] == "Aspirin_v3"
        assert result["NewCompound"] == "NewCompound_v2"

    def test_empty_list(self, db_session):
        result = get_next_version_names_bulk(db_session, [])
        assert result == {}

    def test_bulk_name_with_existing_v_suffix(self, db_session, seed):
        """Input name with _vN suffix is stripped to find true base, then next version computed."""
        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        seed(name="Aspirin_v2", smiles="CC(=O)OC1=CC=CC=C1C(=O)O", entry_id=uuid.uuid4())

        result = get_next_version_names_bulk(db_session, ["Aspirin_v3"])
        # base="Aspirin", max existing version=v2, so next=v3
        assert result["Aspirin_v3"] == "Aspirin_v3"

    def test_bulk_name_with_v_suffix_higher_exists(self, db_session, seed):
        """Input with _vN suffix where higher versions already exist."""
        seed(name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        seed(name="Aspirin_v2", smiles="CC(=O)OC1=CC=CC=C1C(=O)O", entry_id=uuid.uuid4())
        seed(name="Aspirin_v3", smiles="CC(=O)OC1=CC=CC=C1C(=O)O", entry_id=uuid.uuid4())
        seed(name="Aspirin_v4", smiles="CC(=O)OC1=CC=CC=C1C(=O)O", entry_id=uuid.uuid4())

        result = get_next_version_names_bulk(db_session, ["Aspirin_v2"])
        # base="Aspirin", max existing=v4, next=v5
        assert result["Aspirin_v2"] == "Aspirin_v5"

    def test_bulk_all_whitespace_names(self, db_session):
        """All empty/whitespace/None names return empty dict."""
        result = get_next_version_names_bulk(db_session, ["", "  ", None])
        assert result == {}

    def test_bulk_mixed_valid_and_empty(self, db_session, seed):
        """Mixed valid and empty names only return entries for valid names."""
        result = get_next_version_names_bulk(db_session, ["Aspirin", "", "Caffeine"])
        assert "Aspirin" in result
        assert "Caffeine" in result
        assert "" not in result
