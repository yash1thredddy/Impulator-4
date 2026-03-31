"""
Shared fixtures for integration tests.

Provides:
- pg_engine: Postgres engine from root conftest (session-scoped, Alembic-provisioned)
- mock_azure: Patches Azure sync functions to no-ops
- client: FastAPI TestClient wired to the test Postgres database
- db_session: Per-test database session
- seed_compound: Factory fixture for creating test compounds
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text


@pytest.fixture
def test_engine(pg_engine):
    """Alias for pg_engine — used by integration tests that predate the rename."""
    return pg_engine


@pytest.fixture(autouse=True)
def _clean_tables(pg_engine):
    """Truncate all tables after each test for isolation."""
    yield
    with pg_engine.connect() as conn:
        conn.execute(text(
            "TRUNCATE TABLE audit_events, deleted_compounds, compounds, jobs CASCADE"
        ))
        conn.commit()


@pytest.fixture
def db_session(pg_engine):
    """Per-test database session."""
    Session = sessionmaker(bind=pg_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="module")
def mock_azure():
    """Mock Azure storage for tests.

    Patches Azure sync functions so no real Azure calls are made.
    """
    with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
        with patch('backend.core.azure_sync.delete_result_from_azure_by_entry_id', return_value=True):
            yield


@pytest.fixture(scope="module")
def client(pg_engine, mock_azure):
    """FastAPI TestClient wired to test Postgres."""
    from backend.main import app
    from backend.core import database as db_module
    from backend.core.database import get_db

    original_engine = db_module.engine
    original_session_local = db_module.SessionLocal
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)
    db_module.engine = pg_engine
    db_module.SessionLocal = TestSessionLocal

    def override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with patch('backend.core.scheduler.trigger'):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()
    db_module.engine = original_engine
    db_module.SessionLocal = original_session_local


@pytest.fixture
def seed_compound(db_session):
    """Factory fixture for creating test compounds with sensible defaults."""
    from backend.models.compound import Compound
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
