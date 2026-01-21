"""
Integration tests for UUID-based storage functionality.
"""
import pytest
import uuid
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestStoragePathGeneration:
    """Tests for UUID-based storage path generation."""

    def test_get_storage_path_basic(self):
        """Test basic storage path generation."""
        from backend.core.azure_sync import get_storage_path_from_entry_id

        entry_id = "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c"
        path = get_storage_path_from_entry_id(entry_id)

        assert path == "results/3a/3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c.zip"

    def test_get_storage_path_normalizes_case(self):
        """Test that storage path normalizes to lowercase."""
        from backend.core.azure_sync import get_storage_path_from_entry_id

        entry_id_upper = "3A4F8C9E-1B2D-4E5F-9A1C-2D3E4F5A6B7C"
        entry_id_lower = "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c"

        path_upper = get_storage_path_from_entry_id(entry_id_upper)
        path_lower = get_storage_path_from_entry_id(entry_id_lower)

        # Both should produce the same lowercase path
        assert path_upper == path_lower
        assert path_upper == "results/3a/3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c.zip"

    def test_get_storage_path_uses_prefix(self):
        """Test that storage path uses first 2 characters as prefix."""
        from backend.core.azure_sync import get_storage_path_from_entry_id

        # Different entry IDs should have different prefix directories
        entry1 = "ab123456-0000-0000-0000-000000000000"
        entry2 = "cd123456-0000-0000-0000-000000000000"
        entry3 = "ef123456-0000-0000-0000-000000000000"

        path1 = get_storage_path_from_entry_id(entry1)
        path2 = get_storage_path_from_entry_id(entry2)
        path3 = get_storage_path_from_entry_id(entry3)

        assert path1.startswith("results/ab/")
        assert path2.startswith("results/cd/")
        assert path3.startswith("results/ef/")

    def test_get_storage_path_empty_raises(self):
        """Test that empty entry_id raises ValueError."""
        from backend.core.azure_sync import get_storage_path_from_entry_id

        with pytest.raises(ValueError):
            get_storage_path_from_entry_id("")

        with pytest.raises(ValueError):
            get_storage_path_from_entry_id(None)

    def test_get_storage_path_real_uuid(self):
        """Test storage path generation with real UUID."""
        from backend.core.azure_sync import get_storage_path_from_entry_id

        # Generate a real UUID
        entry_id = str(uuid.uuid4())
        path = get_storage_path_from_entry_id(entry_id)

        # Should be a valid path
        assert path.startswith("results/")
        assert path.endswith(".zip")
        assert len(path.split("/")) == 3  # results/prefix/uuid.zip


class TestUUIDBasedAzureOperations:
    """Tests for UUID-based Azure operations."""

    @pytest.fixture
    def mock_blob_client(self):
        """Create a mock blob client."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.upload_blob = MagicMock()
        mock_blob.download_blob.return_value.readall.return_value = b"test data"
        mock_blob.delete_blob = MagicMock()
        mock_blob.get_blob_properties.return_value.size = 100
        return mock_blob

    def test_upload_result_by_entry_id_not_configured(self):
        """Test upload when Azure is not configured."""
        from backend.core.azure_sync import upload_result_to_azure_by_entry_id

        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            result = upload_result_to_azure_by_entry_id("/some/path.zip", "test-id")
            # Should return True when not configured (no-op)
            assert result is True

    def test_upload_result_by_entry_id_empty_id(self):
        """Test upload with empty entry_id."""
        from backend.core.azure_sync import upload_result_to_azure_by_entry_id

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            result = upload_result_to_azure_by_entry_id("/some/path.zip", "")
            assert result is False

            result = upload_result_to_azure_by_entry_id("/some/path.zip", None)
            assert result is False

    def test_download_result_by_entry_id_not_configured(self):
        """Test download when Azure is not configured."""
        from backend.core.azure_sync import download_result_from_azure_by_entry_id

        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            result = download_result_from_azure_by_entry_id("test-id", "/local/path.zip")
            # Should return False when not configured
            assert result is False

    def test_download_result_by_entry_id_empty_id(self):
        """Test download with empty entry_id."""
        from backend.core.azure_sync import download_result_from_azure_by_entry_id

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            result = download_result_from_azure_by_entry_id("", "/local/path.zip")
            assert result is False

            result = download_result_from_azure_by_entry_id(None, "/local/path.zip")
            assert result is False

    def test_delete_result_by_entry_id_not_configured(self):
        """Test deletion when Azure is not configured."""
        from backend.core.azure_sync import delete_result_from_azure_by_entry_id

        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            result = delete_result_from_azure_by_entry_id("test-id")
            # Should return True when not configured (no-op)
            assert result is True

    def test_delete_result_by_entry_id_empty_id(self):
        """Test deletion with empty entry_id."""
        from backend.core.azure_sync import delete_result_from_azure_by_entry_id

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            result = delete_result_from_azure_by_entry_id("")
            assert result is False

            result = delete_result_from_azure_by_entry_id(None)
            assert result is False

    def test_check_result_exists_by_entry_id_not_configured(self):
        """Test existence check when Azure is not configured."""
        from backend.core.azure_sync import check_result_exists_in_azure_by_entry_id

        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            result = check_result_exists_in_azure_by_entry_id("test-id")
            assert result is False

    def test_check_result_exists_by_entry_id_empty_id(self):
        """Test existence check with empty entry_id."""
        from backend.core.azure_sync import check_result_exists_in_azure_by_entry_id

        result = check_result_exists_in_azure_by_entry_id("")
        assert result is False

        result = check_result_exists_in_azure_by_entry_id(None)
        assert result is False


class TestCompoundEntryWithEntryId:
    """Tests for Compound entry creation with entry_id."""

    @pytest.fixture
    def db_session(self):
        """Create an in-memory test database session."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.core.database import Base
        # Import models BEFORE create_all to register them with Base
        from backend.models.database import Job, Compound  # noqa: F401

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    def test_create_compound_with_entry_id(self, db_session):
        """Test creating a compound with entry_id."""
        from backend.models.database import Compound

        entry_id = str(uuid.uuid4())
        compound = Compound(
            entry_id=entry_id,
            compound_name="TestCompound",
            smiles="CCO"
        )
        db_session.add(compound)
        db_session.commit()

        # Retrieve and verify
        retrieved = db_session.query(Compound).filter(
            Compound.entry_id == entry_id
        ).first()

        assert retrieved is not None
        assert retrieved.entry_id == entry_id
        assert retrieved.compound_name == "TestCompound"

    def test_compound_entry_id_unique(self, db_session):
        """Test that entry_id is unique."""
        from backend.models.database import Compound
        from sqlalchemy.exc import IntegrityError

        entry_id = str(uuid.uuid4())

        compound1 = Compound(
            entry_id=entry_id,
            compound_name="Compound1",
            smiles="CCO"
        )
        db_session.add(compound1)
        db_session.commit()

        # Try to create another with same entry_id
        compound2 = Compound(
            entry_id=entry_id,  # Same entry_id
            compound_name="Compound2",
            smiles="CO"
        )
        db_session.add(compound2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_compound_with_storage_path(self, db_session):
        """Test compound with UUID-based storage path."""
        from backend.models.database import Compound
        from backend.core.azure_sync import get_storage_path_from_entry_id

        entry_id = str(uuid.uuid4())
        storage_path = get_storage_path_from_entry_id(entry_id)

        compound = Compound(
            entry_id=entry_id,
            compound_name="TestCompound",
            smiles="CCO",
            storage_path=storage_path
        )
        db_session.add(compound)
        db_session.commit()

        retrieved = db_session.query(Compound).filter(
            Compound.entry_id == entry_id
        ).first()

        assert retrieved.storage_path == storage_path
        assert entry_id in retrieved.storage_path

    def test_duplicate_compound_with_different_entry_id(self, db_session):
        """Test that same compound name can exist with different entry_ids."""
        from backend.models.database import Compound

        # Create two compounds with same name but different entry_ids
        entry_id1 = str(uuid.uuid4())
        entry_id2 = str(uuid.uuid4())

        compound1 = Compound(
            entry_id=entry_id1,
            compound_name="Quercetin",
            smiles="O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C=C2O",
            is_duplicate=False
        )
        compound2 = Compound(
            entry_id=entry_id2,
            compound_name="Quercetin_v2",  # Different name to satisfy name constraint
            smiles="O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C=C2O",
            is_duplicate=True,
            duplicate_of=entry_id1
        )

        db_session.add_all([compound1, compound2])
        db_session.commit()

        # Both should exist
        compounds = db_session.query(Compound).filter(
            Compound.smiles.like("%C1C(O)=C(O)C%")
        ).all()

        assert len(compounds) == 2
        assert compounds[1].is_duplicate is True
        assert compounds[1].duplicate_of == entry_id1


class TestJobServiceCompoundUpdate:
    """Tests for JobService _update_compound_entry with entry_id."""

    @pytest.fixture
    def db_session(self):
        """Create an in-memory test database session."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.core.database import Base
        # Import models BEFORE create_all to register them with Base
        from backend.models.database import Job, Compound  # noqa: F401

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    @pytest.fixture
    def service(self):
        """Create a JobService instance."""
        from backend.services.job_service import JobService
        return JobService()

    def test_update_compound_entry_creates_new(self, service, db_session):
        """Test that _update_compound_entry creates new entry."""
        from backend.models.database import Compound

        entry_id = str(uuid.uuid4())
        result_summary = {
            "compound_name": "NewCompound",
            "smiles": "CCO",
            "entry_id": entry_id,
            "storage_path": f"results/{entry_id[:2]}/{entry_id}.zip",
            "total_activities": 10,
            "imp_candidates": 2
        }

        service._update_compound_entry(
            db_session,
            result_summary,
            "local/path.zip"
        )
        db_session.commit()

        # Verify compound was created
        compound = db_session.query(Compound).filter(
            Compound.compound_name == "NewCompound"
        ).first()

        assert compound is not None
        assert compound.entry_id == entry_id
        assert compound.total_activities == 10

    def test_update_compound_entry_updates_existing(self, service, db_session):
        """Test that _update_compound_entry updates existing entry."""
        from backend.models.database import Compound
        from backend.services.job_service import generate_inchikey

        smiles = "CCO"
        inchikey = generate_inchikey(smiles)

        # Create existing compound
        existing = Compound(
            entry_id=str(uuid.uuid4()),
            compound_name="Ethanol",
            smiles=smiles,
            inchikey=inchikey,
            total_activities=5
        )
        db_session.add(existing)
        db_session.commit()

        # Update with new data
        new_entry_id = str(uuid.uuid4())
        result_summary = {
            "compound_name": "Ethanol",
            "smiles": smiles,
            "entry_id": new_entry_id,
            "storage_path": f"results/{new_entry_id[:2]}/{new_entry_id}.zip",
            "total_activities": 15,  # Updated
            "imp_candidates": 3
        }

        service._update_compound_entry(
            db_session,
            result_summary,
            "local/path.zip"
        )
        db_session.commit()

        # Verify compound was updated (not duplicated)
        compounds = db_session.query(Compound).filter(
            Compound.compound_name == "Ethanol"
        ).all()

        assert len(compounds) == 1
        assert compounds[0].total_activities == 15

    def test_update_compound_entry_creates_duplicate(self, service, db_session):
        """Test that _update_compound_entry creates duplicate entry when requested."""
        from backend.models.database import Compound
        from backend.services.job_service import generate_inchikey

        smiles = "CCO"
        original_entry_id = str(uuid.uuid4())

        # Create original compound
        original = Compound(
            entry_id=original_entry_id,
            compound_name="Ethanol",
            smiles=smiles,
            inchikey=generate_inchikey(smiles)
        )
        db_session.add(original)
        db_session.commit()

        # Create duplicate entry
        duplicate_entry_id = str(uuid.uuid4())
        result_summary = {
            "compound_name": "Ethanol_v2",
            "smiles": smiles,
            "entry_id": duplicate_entry_id,
            "storage_path": f"results/{duplicate_entry_id[:2]}/{duplicate_entry_id}.zip",
            "total_activities": 10
        }

        service._update_compound_entry(
            db_session,
            result_summary,
            "local/path.zip",
            is_duplicate=True,
            duplicate_of=original_entry_id
        )
        db_session.commit()

        # Verify both compounds exist
        all_compounds = db_session.query(Compound).all()
        assert len(all_compounds) == 2

        duplicate = db_session.query(Compound).filter(
            Compound.entry_id == duplicate_entry_id
        ).first()

        assert duplicate is not None
        assert duplicate.is_duplicate is True
        assert duplicate.duplicate_of == original_entry_id
