"""
Shared test fixtures for IMPULATOR tests.
"""
import json
import os
import pytest
from unittest.mock import patch
from pathlib import Path

# Load .env for non-DB secrets (AZURE_CONNECTION_STRING, etc.)
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

# SAFETY: Override DATABASE_URL BEFORE any backend imports.
# .env may contain the real Supabase URL — we must never let backend modules
# bind to it at collection time. This placeholder is replaced by the
# postgres_url fixture with a disposable testcontainer URL before any DB access.
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:1/test_placeholder"
os.environ["DIRECT_DATABASE_URL"] = ""  # Prevent Alembic from connecting to real Supabase

# Now import and clear settings cache to pick up overridden values
from backend.config import get_settings  # noqa: E402
get_settings.cache_clear()


@pytest.fixture(scope="session")
def postgres_url():
    """Provide Postgres URL for tests — always a disposable database.

    Two paths, both safe:
    - CI: Uses the GitHub Actions `services: postgres` container via
      TEST_DATABASE_URL env var (set in CI workflow, NOT your Supabase).
    - Local: Starts a disposable testcontainer via Docker.

    NEVER reads DATABASE_URL from .env — that points to your real Supabase.
    CI uses a separate env var (TEST_DATABASE_URL) to make the distinction explicit.
    """
    ci_test_url = os.environ.get("TEST_DATABASE_URL", "")
    if ci_test_url.startswith("postgresql://"):
        yield ci_test_url
    else:
        from testcontainers.postgres import PostgresContainer
        with PostgresContainer("postgres:15", driver="psycopg2") as pg:
            yield pg.get_connection_url()


@pytest.fixture(scope="session")
def pg_engine(postgres_url):
    """Session-scoped engine with Alembic-provisioned schema."""
    from sqlalchemy import create_engine

    os.environ["DATABASE_URL"] = postgres_url
    os.environ["DIRECT_DATABASE_URL"] = ""  # Force Alembic to use test DB, not real Supabase
    get_settings.cache_clear()

    engine = create_engine(postgres_url)

    # Provision schema via Alembic (bypasses settings.DATABASE_URL)
    from alembic.config import Config
    from alembic import command
    alembic_cfg = Config("backend/alembic.ini")
    # script_location in alembic.ini is relative to the ini file, but Alembic
    # resolves it relative to CWD. Set the absolute path explicitly.
    alembic_cfg.set_main_option(
        "script_location",
        str(Path(__file__).parent.parent / "backend" / "alembic"),
    )
    with engine.connect() as conn:
        alembic_cfg.attributes["connection"] = conn
        command.upgrade(alembic_cfg, "head")
        conn.commit()

    yield engine
    engine.dispose()
    os.environ.pop("DATABASE_URL", None)
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def test_compounds():
    """Sample compounds for testing."""
    return [
        {"name": "Ethanol", "smiles": "CCO"},
        {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
    ]


@pytest.fixture
def mock_azure():
    """Mock Azure for tests."""
    with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
        yield


@pytest.fixture
def sample_job_params():
    """Sample job parameters for testing."""
    return {
        "compound_name": "TestCompound",
        "smiles": "CCO",
        "similarity_threshold": 90,
        "activity_types": None,
    }


@pytest.fixture(scope="session")
def golden_compounds():
    """Load golden compound fixtures for IMP scoring regression tests."""
    fixture_path = Path(__file__).parent / "fixtures" / "golden_compounds.json"
    with open(fixture_path) as f:
        data = json.load(f)
    return data["compounds"]
