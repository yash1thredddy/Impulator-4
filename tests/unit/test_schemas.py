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
            smiles="CCO"
        )
        assert job.similarity_threshold == 90  # Default

    def test_invalid_similarity_threshold_too_high(self):
        """Test similarity threshold > 100 is invalid."""
        from backend.models.schemas import JobCreate
        with pytest.raises(ValidationError):
            JobCreate(
                compound_name="Test",
                smiles="CCO",
                similarity_threshold=150
            )

    def test_invalid_similarity_threshold_too_low(self):
        """Test similarity threshold < 0 is invalid."""
        from backend.models.schemas import JobCreate
        with pytest.raises(ValidationError):
            JobCreate(
                compound_name="Test",
                smiles="CCO",
                similarity_threshold=-10
            )

    def test_empty_compound_name(self):
        """Test empty compound name is invalid."""
        from backend.models.schemas import JobCreate
        with pytest.raises(ValidationError):
            JobCreate(
                compound_name="",
                smiles="CCO"
            )

    def test_empty_smiles(self):
        """Test empty SMILES is invalid."""
        from backend.models.schemas import JobCreate
        with pytest.raises(ValidationError):
            JobCreate(
                compound_name="Test",
                smiles=""
            )

    def test_default_activity_types(self):
        """Test default activity types is None."""
        from backend.models.schemas import JobCreate
        job = JobCreate(
            compound_name="Test",
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
            job = JobCreate(compound_name="Test", smiles=smiles)
            assert job.smiles == smiles

    def test_smiles_too_long_rejected(self):
        """Test SMILES longer than 2000 characters is rejected."""
        from backend.models.schemas import JobCreate

        long_smiles = "C" * 2001
        with pytest.raises(ValidationError) as exc_info:
            JobCreate(compound_name="Test", smiles=long_smiles)
        assert "too long" in str(exc_info.value).lower()

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
                JobCreate(compound_name="Test", smiles=smiles)

    def test_smiles_whitespace_stripped(self):
        """Test SMILES whitespace is stripped."""
        from backend.models.schemas import JobCreate

        job = JobCreate(compound_name="Test", smiles="  CCO  ")
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
            job = JobCreate(compound_name="Test", smiles=smiles)
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
            job = JobCreate(compound_name=name, smiles="CCO")
            assert job.compound_name == name

    def test_compound_name_too_long_rejected(self):
        """Test compound names longer than 100 characters are rejected."""
        from backend.models.schemas import JobCreate

        long_name = "A" * 101
        with pytest.raises(ValidationError) as exc_info:
            JobCreate(compound_name=long_name, smiles="CCO")
        # Pydantic error message says "should have at most 100 characters"
        assert "100 characters" in str(exc_info.value).lower()

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
                JobCreate(compound_name=name, smiles="CCO")

    def test_compound_name_null_byte_rejected(self):
        """Test null byte injection is rejected."""
        from backend.models.schemas import JobCreate

        with pytest.raises(ValidationError):
            JobCreate(compound_name="test\x00malicious", smiles="CCO")

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
                JobCreate(compound_name=name, smiles="CCO")

    def test_compound_name_whitespace_stripped(self):
        """Test compound name whitespace is stripped."""
        from backend.models.schemas import JobCreate

        job = JobCreate(compound_name="  Aspirin  ", smiles="CCO")
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
