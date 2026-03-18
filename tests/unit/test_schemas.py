"""
Unit tests for Pydantic schemas.
"""
import pytest
from pydantic import ValidationError


class TestJobCreateSchema:
    """Tests for JobCreate schema."""

    def test_valid_job_create(self):
        """Test valid job creation schema."""
        from backend.models.schemas import JobCreate
        job = JobCreate(
            compound_name="Aspirin",
            author_name="Test Author",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            similarity_threshold=90
        )
        assert job.compound_name == "Aspirin"
        assert job.similarity_threshold == 90

    def test_default_similarity_threshold(self):
        """Test default similarity threshold."""
        from backend.models.schemas import JobCreate
        job = JobCreate(
            compound_name="Test",
            author_name="Test Author",
            smiles="CCO"
        )
        assert job.similarity_threshold == 90  # Default

    def test_invalid_similarity_threshold_too_high(self):
        """Test similarity threshold > 100 is invalid."""
        from backend.models.schemas import JobCreate
        with pytest.raises(ValidationError):
            JobCreate(
                compound_name="Test",
                author_name="Test Author",
                smiles="CCO",
                similarity_threshold=150
            )

    def test_invalid_similarity_threshold_too_low(self):
        """Test similarity threshold < 0 is invalid."""
        from backend.models.schemas import JobCreate
        with pytest.raises(ValidationError):
            JobCreate(
                compound_name="Test",
                author_name="Test Author",
                smiles="CCO",
                similarity_threshold=-10
            )

    def test_empty_compound_name(self):
        """Test empty compound name is invalid."""
        from backend.models.schemas import JobCreate
        with pytest.raises(ValidationError):
            JobCreate(
                compound_name="",
                author_name="Test Author",
                smiles="CCO"
            )

    def test_empty_smiles(self):
        """Test empty SMILES is invalid."""
        from backend.models.schemas import JobCreate
        with pytest.raises(ValidationError):
            JobCreate(
                compound_name="Test",
                author_name="Test Author",
                smiles=""
            )

    def test_default_activity_types(self):
        """Test default activity types is None."""
        from backend.models.schemas import JobCreate
        job = JobCreate(
            compound_name="Test",
            author_name="Test Author",
            smiles="CCO"
        )
        assert job.activity_types is None


class TestActiveJobResponseSchema:
    """Tests for ActiveJobResponse schema."""

    def test_valid_active_job_response(self):
        """Test valid active job response."""
        from backend.models.schemas import ActiveJobResponse
        from backend.models.database import JobStatus

        response = ActiveJobResponse(
            id="test-123",
            status=JobStatus.PROCESSING,
            progress=45.5,
            current_step="Fetching activities",
            compound_name="Aspirin"
        )
        assert response.id == "test-123"
        assert response.progress == 45.5


class TestExecutorStatsSchema:
    """Tests for ExecutorStats schema."""

    def test_valid_executor_stats(self):
        """Test valid executor stats."""
        from backend.models.schemas import ExecutorStats

        stats = ExecutorStats(
            max_workers=2,
            active_jobs=1,
            has_capacity=True,
            job_ids=["job-1"]
        )
        assert stats.max_workers == 2
        assert stats.active_jobs == 1
        assert stats.has_capacity is True
        assert "job-1" in stats.job_ids


class TestSMILESValidation:
    """Tests for SMILES validation security fixes."""

    def test_valid_smiles_accepted(self):
        """Test valid SMILES strings are accepted."""
        from backend.models.schemas import JobCreate

        valid_smiles = [
            "CCO",  # Ethanol
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "c1ccccc1",  # Benzene
            "C[C@H](N)C(=O)O",  # Alanine with stereochemistry
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        ]

        for smiles in valid_smiles:
            job = JobCreate(compound_name="Test", author_name="Test Author", smiles=smiles)
            assert job.smiles == smiles

    def test_smiles_too_long_rejected(self):
        """Test SMILES longer than 5000 characters is rejected."""
        from backend.models.schemas import JobCreate

        long_smiles = "C" * 5001
        with pytest.raises(ValidationError):
            JobCreate(compound_name="Test", author_name="Test Author", smiles=long_smiles)

    def test_smiles_injection_characters_rejected(self):
        """Test SMILES with injection characters are rejected."""
        from backend.models.schemas import JobCreate

        malicious_smiles = [
            "CCO<script>",
            "CCO; DROP TABLE",
            "CCO|cat /etc/passwd",
            "CCO`whoami`",
            "CCO{malicious}",
        ]

        for smiles in malicious_smiles:
            with pytest.raises(ValidationError):
                JobCreate(compound_name="Test", author_name="Test Author", smiles=smiles)

    def test_smiles_whitespace_stripped(self):
        """Test SMILES whitespace is stripped."""
        from backend.models.schemas import JobCreate

        job = JobCreate(compound_name="Test", author_name="Test Author", smiles="  CCO  ")
        assert job.smiles == "CCO"

    def test_smiles_special_valid_characters(self):
        """Test SMILES with valid special characters."""
        from backend.models.schemas import JobCreate

        # Valid SMILES special characters
        special_smiles = [
            "C#N",  # Triple bond (hydrogen cyanide)
            "C=C",  # Double bond (ethene)
            "[Na+]",  # Ion
            "C/C=C/C",  # Cis/trans (but-2-ene)
            "[C@H](O)(F)Cl",  # Stereochemistry (proper context)
            "c1ccccc1",  # Aromatic (benzene)
        ]

        for smiles in special_smiles:
            job = JobCreate(compound_name="Test", author_name="Test Author", smiles=smiles)
            assert job.smiles == smiles


class TestCompoundNameValidation:
    """Tests for compound name validation security fixes."""

    def test_valid_compound_names_accepted(self):
        """Test valid compound names are accepted."""
        from backend.models.schemas import JobCreate

        valid_names = [
            "Aspirin",
            "Ibuprofen-200",
            "Test_Compound",
            "Compound (1)",
            "L-Alanine",
            "5'-AMP",
            "Vitamin B12",
        ]

        for name in valid_names:
            job = JobCreate(compound_name=name, author_name="Test Author", smiles="CCO")
            assert job.compound_name == name

    def test_compound_name_too_long_rejected(self):
        """Test compound names longer than 255 characters are rejected."""
        from backend.models.schemas import JobCreate

        long_name = "A" * 256
        with pytest.raises(ValidationError) as exc_info:
            JobCreate(compound_name=long_name, author_name="Test Author", smiles="CCO")
        assert "255 characters" in str(exc_info.value).lower()

    def test_compound_name_path_traversal_rejected(self):
        """Test path traversal attempts are rejected."""
        from backend.models.schemas import JobCreate

        path_traversal_names = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "compound/../secret",
            "test/../../admin",
        ]

        for name in path_traversal_names:
            with pytest.raises(ValidationError):
                JobCreate(compound_name=name, author_name="Test Author", smiles="CCO")

    def test_compound_name_null_byte_rejected(self):
        """Test null byte injection is rejected."""
        from backend.models.schemas import JobCreate

        with pytest.raises(ValidationError):
            JobCreate(compound_name="test\x00malicious", author_name="Test Author", smiles="CCO")

    def test_compound_name_html_injection_rejected(self):
        """Test HTML/script injection is rejected."""
        from backend.models.schemas import JobCreate

        html_names = [
            "<script>alert('xss')</script>",
            "compound<img src=x onerror=alert(1)>",
            "test{javascript:alert(1)}",
        ]

        for name in html_names:
            with pytest.raises(ValidationError):
                JobCreate(compound_name=name, author_name="Test Author", smiles="CCO")

    def test_compound_name_whitespace_stripped(self):
        """Test compound name whitespace is stripped."""
        from backend.models.schemas import JobCreate

        job = JobCreate(compound_name="  Aspirin  ", author_name="Test Author", smiles="CCO")
        assert job.compound_name == "Aspirin"


class TestConsistentResponseModels:
    """Tests for consistent API response models."""

    def test_message_response(self):
        """Test MessageResponse model."""
        from backend.models.schemas import MessageResponse

        response = MessageResponse(status="success", message="Operation completed")
        assert response.status == "success"
        assert response.message == "Operation completed"

    def test_skip_response(self):
        """Test SkipResponse model."""
        from backend.models.schemas import SkipResponse

        response = SkipResponse(
            status="skipped",
            message="Compound skipped",
            compound_name="Aspirin"
        )
        assert response.status == "skipped"
        assert response.compound_name == "Aspirin"

    def test_delete_response(self):
        """Test DeleteResponse model."""
        from backend.models.schemas import DeleteResponse

        response = DeleteResponse(
            message="Job deleted",
            job_id="test-123",
            compound_name="Aspirin"
        )
        assert response.message == "Job deleted"
        assert response.job_id == "test-123"

    def test_cancel_response(self):
        """Test CancelResponse model."""
        from backend.models.schemas import CancelResponse

        response = CancelResponse(
            batch_id="batch-123",
            cancelled_count=5,
            message="Cancelled 5 jobs"
        )
        assert response.batch_id == "batch-123"
        assert response.cancelled_count == 5


class TestBatchJobCreateSchema:
    """Tests for BatchJobCreate schema validators."""

    def _make_compound(self, name="Aspirin", smiles="CCO"):
        """Helper to create a valid compound dict for BatchJobCreate."""
        return {
            "compound_name": name,
            "author_name": "Test Author",
            "smiles": smiles,
        }

    def test_valid_batch_with_compounds(self):
        """Test valid batch with a list of compounds."""
        from backend.models.schemas import BatchJobCreate

        batch = BatchJobCreate(
            compounds=[self._make_compound("Aspirin"), self._make_compound("Ibuprofen")],
            session_id="test-session",
        )
        assert len(batch.compounds) == 2
        assert batch.compounds[0].compound_name == "Aspirin"

    def test_empty_compounds_list_rejected(self):
        """Test that an empty compounds list is rejected."""
        from backend.models.schemas import BatchJobCreate

        with pytest.raises(ValidationError) as exc_info:
            BatchJobCreate(compounds=[])
        assert "too_short" in str(exc_info.value).lower() or "min_length" in str(exc_info.value).lower()

    def test_over_1000_compounds_rejected(self):
        """Test that more than 1000 compounds are rejected."""
        from backend.models.schemas import BatchJobCreate

        compounds = [self._make_compound(f"Compound-{i}") for i in range(1001)]
        with pytest.raises(ValidationError) as exc_info:
            BatchJobCreate(compounds=compounds)
        assert "too_long" in str(exc_info.value).lower() or "max_length" in str(exc_info.value).lower()

    def test_valid_duplicate_decisions(self):
        """Test valid duplicate_decisions dict with allowed action values."""
        from backend.models.schemas import BatchJobCreate

        batch = BatchJobCreate(
            compounds=[self._make_compound()],
            duplicate_decisions={
                "Quercetin": "skip",
                "Resveratrol": "replace",
                "Curcumin": "duplicate",
            },
        )
        assert batch.duplicate_decisions["Quercetin"] == "skip"
        assert batch.duplicate_decisions["Resveratrol"] == "replace"
        assert batch.duplicate_decisions["Curcumin"] == "duplicate"

    def test_invalid_duplicate_decisions_action_rejected(self):
        """Test that invalid action values in duplicate_decisions are rejected."""
        from backend.models.schemas import BatchJobCreate

        with pytest.raises(ValidationError) as exc_info:
            BatchJobCreate(
                compounds=[self._make_compound()],
                duplicate_decisions={"Aspirin": "delete"},
            )
        assert "must be one of" in str(exc_info.value).lower()

    def test_duplicate_decisions_empty_key_rejected(self):
        """Test that empty string keys in duplicate_decisions are rejected."""
        from backend.models.schemas import BatchJobCreate

        with pytest.raises(ValidationError) as exc_info:
            BatchJobCreate(
                compounds=[self._make_compound()],
                duplicate_decisions={"": "skip"},
            )
        assert "cannot be empty" in str(exc_info.value).lower()

    def test_duplicate_decisions_invalid_compound_name_rejected(self):
        """Test that compound names with invalid characters in duplicate_decisions are rejected."""
        from backend.models.schemas import BatchJobCreate

        with pytest.raises(ValidationError) as exc_info:
            BatchJobCreate(
                compounds=[self._make_compound()],
                duplicate_decisions={"<script>alert(1)</script>": "skip"},
            )
        assert "invalid compound name" in str(exc_info.value).lower()

    def test_duplicate_decisions_none_accepted(self):
        """Test that None duplicate_decisions is accepted (optional field)."""
        from backend.models.schemas import BatchJobCreate

        batch = BatchJobCreate(compounds=[self._make_compound()])
        assert batch.duplicate_decisions is None


class TestResolveDuplicateRequestSchema:
    """Tests for ResolveDuplicateRequest schema validators."""

    def _make_resolve(self, **overrides):
        """Helper to build a valid ResolveDuplicateRequest dict."""
        base = {
            "action": "skip",
            "smiles": "CCO",
            "compound_name": "Aspirin",
            "author_name": "Test Author",
        }
        base.update(overrides)
        return base

    def test_valid_resolve_skip(self):
        """Test valid resolve request with action 'skip'."""
        from backend.models.schemas import ResolveDuplicateRequest

        req = ResolveDuplicateRequest(**self._make_resolve(action="skip"))
        assert req.action.value == "skip"

    def test_valid_resolve_replace(self):
        """Test valid resolve request with action 'replace'."""
        from backend.models.schemas import ResolveDuplicateRequest

        req = ResolveDuplicateRequest(**self._make_resolve(action="replace"))
        assert req.action.value == "replace"

    def test_valid_resolve_duplicate(self):
        """Test valid resolve request with action 'duplicate'."""
        from backend.models.schemas import ResolveDuplicateRequest

        req = ResolveDuplicateRequest(**self._make_resolve(action="duplicate"))
        assert req.action.value == "duplicate"

    def test_invalid_action_rejected(self):
        """Test that an invalid action value is rejected."""
        from backend.models.schemas import ResolveDuplicateRequest

        with pytest.raises(ValidationError):
            ResolveDuplicateRequest(**self._make_resolve(action="delete"))

    def test_new_compound_name_path_traversal_rejected(self):
        """Test that path traversal in new_compound_name is rejected."""
        from backend.models.schemas import ResolveDuplicateRequest

        with pytest.raises(ValidationError) as exc_info:
            ResolveDuplicateRequest(**self._make_resolve(new_compound_name="../etc/passwd"))
        errors_str = str(exc_info.value).lower()
        assert "invalid" in errors_str

    def test_new_compound_name_too_long_rejected(self):
        """Test that new_compound_name exceeding 255 chars is rejected."""
        from backend.models.schemas import ResolveDuplicateRequest

        with pytest.raises(ValidationError) as exc_info:
            ResolveDuplicateRequest(**self._make_resolve(new_compound_name="A" * 256))
        assert "255 characters" in str(exc_info.value).lower()

    def test_new_compound_name_null_byte_rejected(self):
        """Test that null byte in new_compound_name is rejected."""
        from backend.models.schemas import ResolveDuplicateRequest

        with pytest.raises(ValidationError):
            ResolveDuplicateRequest(**self._make_resolve(new_compound_name="Test\x00Evil"))

    def test_new_compound_name_valid_accepted(self):
        """Test that a valid new_compound_name is accepted."""
        from backend.models.schemas import ResolveDuplicateRequest

        req = ResolveDuplicateRequest(**self._make_resolve(new_compound_name="Aspirin_v2"))
        assert req.new_compound_name == "Aspirin_v2"

    def test_new_compound_name_empty_string_becomes_none(self):
        """Test that an empty/whitespace new_compound_name normalises to None."""
        from backend.models.schemas import ResolveDuplicateRequest

        req = ResolveDuplicateRequest(**self._make_resolve(new_compound_name="   "))
        assert req.new_compound_name is None

    def test_existing_entry_id_optional(self):
        """Test that existing_entry_id is optional and defaults to None."""
        from backend.models.schemas import ResolveDuplicateRequest

        req = ResolveDuplicateRequest(**self._make_resolve())
        assert req.existing_entry_id is None

        req2 = ResolveDuplicateRequest(
            **self._make_resolve(existing_entry_id="3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c")
        )
        assert req2.existing_entry_id == "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c"


class TestCheckDuplicatesRequestSchema:
    """Tests for CheckDuplicatesRequest schema."""

    def test_valid_with_compound_names(self):
        """Test valid request with legacy compound_names list."""
        from backend.models.schemas import CheckDuplicatesRequest

        req = CheckDuplicatesRequest(compound_names=["Aspirin", "Ibuprofen"])
        assert len(req.compound_names) == 2

    def test_valid_with_compounds_structures(self):
        """Test valid request with compounds (structure-based)."""
        from backend.models.schemas import CheckDuplicatesRequest

        req = CheckDuplicatesRequest(
            compounds=[
                {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                {"compound_name": "Unknown", "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
            ]
        )
        assert len(req.compounds) == 2
        assert req.compounds[0].compound_name == "Aspirin"

    def test_over_1000_compound_names_rejected(self):
        """Test that more than 1000 compound_names are rejected."""
        from backend.models.schemas import CheckDuplicatesRequest

        with pytest.raises(ValidationError):
            CheckDuplicatesRequest(compound_names=["C" + str(i) for i in range(1001)])

    def test_over_1000_compounds_rejected(self):
        """Test that more than 1000 compounds are rejected."""
        from backend.models.schemas import CheckDuplicatesRequest

        compounds = [{"compound_name": f"Compound-{i}"} for i in range(1001)]
        with pytest.raises(ValidationError):
            CheckDuplicatesRequest(compounds=compounds)

    def test_similarity_threshold_validation(self):
        """Test similarity_threshold range validation on CheckDuplicatesRequest."""
        from backend.models.schemas import CheckDuplicatesRequest

        req = CheckDuplicatesRequest(compound_names=["Aspirin"], similarity_threshold=40)
        assert req.similarity_threshold == 40

        with pytest.raises(ValidationError):
            CheckDuplicatesRequest(compound_names=["Aspirin"], similarity_threshold=39)

        with pytest.raises(ValidationError):
            CheckDuplicatesRequest(compound_names=["Aspirin"], similarity_threshold=101)
