"""Tests for PubChem configuration constants."""
from frontend.config.settings import config


def test_pubchem_config_exists():
    assert hasattr(config, 'PUBCHEM_BASE_URL')
    assert config.PUBCHEM_BASE_URL == 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'


def test_pubchem_timeout():
    assert hasattr(config, 'PUBCHEM_TIMEOUT_SECONDS')
    assert config.PUBCHEM_TIMEOUT_SECONDS == 10


def test_pubchem_batch_size():
    assert hasattr(config, 'PUBCHEM_BATCH_SIZE')
    assert config.PUBCHEM_BATCH_SIZE == 100
