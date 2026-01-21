"""
Integration tests for duplicate detection API endpoints.
"""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine with tables using shared in-memory DB."""
    from backend.core.database import Base
    # Import models BEFORE create_all to register them with Base
    from backend.models.database import Job, Compound  # noqa: F401

    # Use StaticPool to ensure all connections share the same in-memory database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_azure():
    """Mock Azure storage for tests."""
    with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
        with patch('backend.core.azure_sync.sync_db_to_azure', return_value=True):
            with patch('backend.core.azure_sync.delete_result_from_azure_by_entry_id', return_value=True):
                yield


@pytest.fixture
def client_with_db(test_engine, mock_azure):
    """Create a test client with properly configured database.

    This fixture patches both the get_db dependency AND the underlying engine/SessionLocal
    to ensure all database operations use the test database.
    """
    from backend.main import app
    from backend.core import database as db_module
    from backend.core.database import get_db

    # Save original values
    original_engine = db_module.engine
    original_session_local = db_module.SessionLocal

    # Create new SessionLocal bound to test engine
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Patch the module-level engine and SessionLocal
    db_module.engine = test_engine
    db_module.SessionLocal = TestSessionLocal

    def override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client

    # Restore original values
    app.dependency_overrides.clear()
    db_module.engine = original_engine
    db_module.SessionLocal = original_session_local


class TestDuplicateDetection:
    """Tests for duplicate detection during job submission."""

    def test_submit_job_no_duplicate(self, client_with_db):
        """Test submitting a job with no duplicate."""
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "TestCompound",
                "smiles": "CCO",
                "similarity_threshold": 90
            }
        )

        # Should create job successfully (status 201)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data.get("status") != "duplicate_found"

    def test_submit_job_duplicate_exact_match(self, test_engine, client_with_db):
        """Test duplicate detection when exact match exists."""
        from backend.models.database import Compound
        from backend.services.job_service import generate_inchikey

        Session = sessionmaker(bind=test_engine)

        # First, create an existing compound in the database
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        inchikey = generate_inchikey(smiles)

        session = Session()
        existing_compound = Compound(
            entry_id="test-entry-id-12345",
            compound_name="Aspirin",
            smiles=smiles,
            inchikey=inchikey
        )
        session.add(existing_compound)
        session.commit()
        session.close()

        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Aspirin",
                "smiles": smiles,
                "similarity_threshold": 90
            }
        )

        # Should return duplicate_found
        assert response.status_code == 201
        data = response.json()
        assert data.get("status") == "duplicate_found"
        assert data.get("duplicate_type") == "exact"
        assert "existing_compound" in data
        assert data["existing_compound"]["compound_name"] == "Aspirin"

    def test_submit_job_duplicate_structure_only(self, test_engine, client_with_db):
        """Test duplicate detection when only structure matches (different name)."""
        from backend.models.database import Compound
        from backend.services.job_service import generate_inchikey

        Session = sessionmaker(bind=test_engine)

        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        inchikey = generate_inchikey(smiles)

        session = Session()
        existing_compound = Compound(
            entry_id="test-entry-id-12345",
            compound_name="Aspirin",
            smiles=smiles,
            inchikey=inchikey
        )
        session.add(existing_compound)
        session.commit()
        session.close()

        # Submit with same structure but different name
        response = client_with_db.post(
            "/api/v1/jobs",
            json={
                "compound_name": "Acetylsalicylic Acid",  # Different name
                "smiles": smiles,  # Same SMILES
                "similarity_threshold": 90
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data.get("status") == "duplicate_found"
        assert data.get("duplicate_type") == "structure_only"


class TestResolveDuplicate:
    """Tests for resolve-duplicate endpoint."""

    def test_resolve_duplicate_skip(self, client_with_db):
        """Test skipping a duplicate compound."""
        response = client_with_db.post(
            "/api/v1/jobs/resolve-duplicate",
            json={
                "action": "skip",
                "smiles": "CCO",
                "compound_name": "Ethanol"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data.get("status") == "skipped"
        assert "skipped" in data.get("message", "").lower()

    def test_resolve_duplicate_replace(self, test_engine, client_with_db):
        """Test replacing an existing compound."""
        from backend.models.database import Compound
        from backend.services.job_service import generate_inchikey

        Session = sessionmaker(bind=test_engine)

        smiles = "CCO"
        inchikey = generate_inchikey(smiles)

        session = Session()
        existing_compound = Compound(
            entry_id="existing-entry-id-12345",
            compound_name="Ethanol",
            smiles=smiles,
            inchikey=inchikey
        )
        session.add(existing_compound)
        session.commit()
        session.close()

        response = client_with_db.post(
            "/api/v1/jobs/resolve-duplicate",
            json={
                "action": "replace",
                "smiles": smiles,
                "compound_name": "Ethanol",
                "existing_entry_id": "existing-entry-id-12345"
            }
        )

        assert response.status_code == 201
        data = response.json()
        # Should create a new job
        assert "id" in data

        # Original compound should be deleted
        check_session = Session()
        deleted = check_session.query(Compound).filter(
            Compound.entry_id == "existing-entry-id-12345"
        ).first()
        assert deleted is None
        check_session.close()

    def test_resolve_duplicate_as_duplicate(self, test_engine, client_with_db):
        """Test saving as a duplicate (keeping both)."""
        from backend.models.database import Compound
        from backend.services.job_service import generate_inchikey

        Session = sessionmaker(bind=test_engine)

        smiles = "CCO"
        inchikey = generate_inchikey(smiles)

        session = Session()
        existing_compound = Compound(
            entry_id="original-entry-id-12345",
            compound_name="Ethanol",
            smiles=smiles,
            inchikey=inchikey
        )
        session.add(existing_compound)
        session.commit()
        session.close()

        response = client_with_db.post(
            "/api/v1/jobs/resolve-duplicate",
            json={
                "action": "duplicate",
                "smiles": smiles,
                "compound_name": "Ethanol_v2",
                "existing_entry_id": "original-entry-id-12345"
            }
        )

        assert response.status_code == 201
        data = response.json()
        # Should create a new job
        assert "id" in data

        # Original compound should still exist
        check_session = Session()
        original = check_session.query(Compound).filter(
            Compound.entry_id == "original-entry-id-12345"
        ).first()
        assert original is not None
        check_session.close()


class TestCheckDuplicates:
    """Tests for check-duplicates endpoint."""

    def test_check_duplicates_none_exist(self, test_engine, client_with_db):
        """Test checking duplicates when none exist."""
        response = client_with_db.post(
            "/api/v1/jobs/check-duplicates",
            json={
                "compound_names": ["Aspirin", "Ibuprofen", "Caffeine"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["existing"] == []
        assert data["processing"] == []
        assert set(data["new"]) == {"Aspirin", "Ibuprofen", "Caffeine"}

    def test_check_duplicates_some_exist(self, test_engine, client_with_db):
        """Test checking duplicates when some exist."""
        from backend.models.database import Compound

        Session = sessionmaker(bind=test_engine)

        # Create existing compounds
        session = Session()
        compound1 = Compound(
            entry_id="entry-1",
            compound_name="Aspirin"
        )
        compound2 = Compound(
            entry_id="entry-2",
            compound_name="Ibuprofen"
        )
        session.add_all([compound1, compound2])
        session.commit()
        session.close()

        response = client_with_db.post(
            "/api/v1/jobs/check-duplicates",
            json={
                "compound_names": ["Aspirin", "Ibuprofen", "Caffeine", "Quercetin"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert set(data["existing"]) == {"Aspirin", "Ibuprofen"}
        assert set(data["new"]) == {"Caffeine", "Quercetin"}


class TestDuplicateActionValidation:
    """Tests for validation of duplicate actions."""

    def test_invalid_action_rejected(self, client_with_db):
        """Test that invalid actions are rejected."""
        response = client_with_db.post(
            "/api/v1/jobs/resolve-duplicate",
            json={
                "action": "invalid_action",
                "smiles": "CCO",
                "compound_name": "Ethanol"
            }
        )

        # Should be rejected with validation error
        assert response.status_code == 422
