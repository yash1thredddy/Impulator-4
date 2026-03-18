"""
Shared fixtures for integration tests.

Provides:
- test_engine: In-memory SQLite with all tables created (module-scoped for speed)
- mock_azure: Patches Azure sync functions to no-ops
- client: FastAPI TestClient wired to the test database
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="module")
def test_engine():
    """Create a test database engine once per module (faster than per-function)."""
    from backend.core.database import Base
    from backend.models.database import Job, Compound, DeletedCompound  # noqa: F401

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(autouse=True)
def _clean_tables(test_engine):
    """Truncate all tables after each test to isolate state."""
    yield
    from backend.core.database import Base
    with test_engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()


@pytest.fixture
def db_session(test_engine):
    """Create a database session bound to the test engine.

    Eliminates the repeated `Session = sessionmaker(bind=test_engine)` boilerplate
    in every test method that needs to seed data.
    """
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_azure():
    """Mock Azure storage for tests.

    Patches all three Azure sync functions so no real Azure calls are made.
    """
    with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
        with patch('backend.core.azure_sync.sync_db_to_azure', return_value=True):
            with patch('backend.core.azure_sync.delete_result_from_azure_by_entry_id', return_value=True):
                yield


@pytest.fixture
def client(test_engine, mock_azure):
    """Create a test client for the FastAPI app with proper test database.

    Wires the app's get_db dependency to the in-memory test engine,
    patches the scheduler trigger to prevent background processing,
    and restores all originals on teardown.
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

    # Mock the scheduler to prevent background job processing after test teardown
    with patch('backend.core.scheduler.job_scheduler.trigger'):
        with TestClient(app) as c:
            yield c

    # Restore original values
    app.dependency_overrides.clear()
    db_module.engine = original_engine
    db_module.SessionLocal = original_session_local


@pytest.fixture
def seed_compound(db_session):
    """Factory fixture for creating test compounds with sensible defaults."""
    from backend.models.database import Compound
    from backend.services.job_service import _inchikey_structure_key
    import uuid
    from datetime import datetime, timezone

    def _create(name="TestCompound", smiles="CCO", inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N", **overrides):
        defaults = {
            "entry_id": str(uuid.uuid4()),
            "compound_name": name,
            "smiles": smiles,
            "inchikey": inchikey,
            "inchikey_structure_key": _inchikey_structure_key(inchikey),
            "similarity_threshold": 90,
            "activity_types": "EC50,IC50,Kd,Ki",
            "processed_at": datetime.now(timezone.utc),
        }
        defaults.update(overrides)
        comp = Compound(**defaults)
        db_session.add(comp)
        db_session.commit()
        db_session.refresh(comp)
        return comp
    return _create
