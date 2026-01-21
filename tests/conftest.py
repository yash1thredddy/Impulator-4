"""
Shared test fixtures for IMPULATOR tests.
"""
import os
import pytest
from unittest.mock import patch
from pathlib import Path

# Load .env file FIRST before any backend imports
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

# Set test-specific environment (but don't override secrets from .env)
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

# Now import and clear settings cache to pick up .env values
from backend.config import get_settings
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
