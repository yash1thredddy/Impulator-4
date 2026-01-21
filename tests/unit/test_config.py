"""
Unit tests for Backend Configuration Module.

Tests settings loading and validation including:
- Default values
- Environment variable overrides
- CORS origins parsing
- Path handling
"""
import pytest
from unittest.mock import patch
import os


class TestSettingsDefaults:
    """Tests for default settings values."""

    def test_default_app_name(self):
        """Test default application name."""
        from backend.config import Settings

        # Create fresh settings without env file
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.APP_NAME == "Impulator"

    def test_default_app_version(self):
        """Test default application version."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.APP_VERSION == "2.1.0"

    def test_default_debug_false(self):
        """Test debug is False by default."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.DEBUG == False

    def test_default_api_port(self):
        """Test default API port."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.API_PORT == 8000

    def test_default_frontend_port(self):
        """Test default frontend port."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.FRONTEND_PORT == 7860

    def test_default_max_workers(self):
        """Test default max workers."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.MAX_WORKERS == 2

    def test_default_job_timeout(self):
        """Test default job timeout."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.JOB_TIMEOUT == 3600  # 1 hour

    def test_default_cache_size(self):
        """Test default cache size."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.CACHE_SIZE == 2000


class TestSettingsOverrides:
    """Tests for environment variable overrides."""

    def test_override_debug(self):
        """Test DEBUG can be overridden."""
        from backend.config import Settings

        with patch.dict(os.environ, {'DEBUG': 'true'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.DEBUG == True

    def test_override_api_port(self):
        """Test API_PORT can be overridden."""
        from backend.config import Settings

        with patch.dict(os.environ, {'API_PORT': '9000'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.API_PORT == 9000

    def test_override_max_workers(self):
        """Test MAX_WORKERS can be overridden."""
        from backend.config import Settings

        with patch.dict(os.environ, {'MAX_WORKERS': '4'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.MAX_WORKERS == 4

    def test_override_database_url(self):
        """Test DATABASE_URL can be overridden."""
        from backend.config import Settings

        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///./test.db'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.DATABASE_URL == 'sqlite:///./test.db'

    def test_override_azure_connection_string(self):
        """Test AZURE_CONNECTION_STRING can be overridden."""
        from backend.config import Settings

        with patch.dict(os.environ, {'AZURE_CONNECTION_STRING': 'DefaultEndpoint=...'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.AZURE_CONNECTION_STRING == 'DefaultEndpoint=...'


class TestCORSOriginsParsing:
    """Tests for CORS origins parsing."""

    def test_cors_origins_parsing(self):
        """Test CORS origins string is parsed to list."""
        from backend.config import Settings

        with patch.dict(os.environ, {'CORS_ORIGINS': 'http://localhost:3000,http://localhost:8000'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.cors_origins_list == ['http://localhost:3000', 'http://localhost:8000']

    def test_cors_origins_with_spaces(self):
        """Test CORS origins handles spaces correctly."""
        from backend.config import Settings

        with patch.dict(os.environ, {'CORS_ORIGINS': 'http://localhost:3000, http://localhost:8000'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.cors_origins_list == ['http://localhost:3000', 'http://localhost:8000']

    def test_cors_origins_empty(self):
        """Test CORS origins handles empty string."""
        from backend.config import Settings

        with patch.dict(os.environ, {'CORS_ORIGINS': ''}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.cors_origins_list == []

    def test_cors_origins_single(self):
        """Test CORS origins handles single origin."""
        from backend.config import Settings

        with patch.dict(os.environ, {'CORS_ORIGINS': 'http://localhost:3000'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.cors_origins_list == ['http://localhost:3000']


class TestPathSettings:
    """Tests for path settings."""

    def test_data_dir_is_path(self):
        """Test DATA_DIR is a Path object."""
        from backend.config import Settings
        from pathlib import Path

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert isinstance(settings.DATA_DIR, Path)

    def test_results_dir_is_path(self):
        """Test RESULTS_DIR is a Path object."""
        from backend.config import Settings
        from pathlib import Path

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert isinstance(settings.RESULTS_DIR, Path)

    def test_default_data_dir(self):
        """Test default data directory."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert str(settings.DATA_DIR) == 'data'

    def test_default_results_dir(self):
        """Test default results directory."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert 'results' in str(settings.RESULTS_DIR)


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test get_settings returns Settings instance."""
        from backend.config import get_settings, Settings

        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_get_settings_cached(self):
        """Test get_settings returns cached instance."""
        from backend.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same object (cached)
        assert settings1 is settings2


class TestExternalAPISettings:
    """Tests for external API settings."""

    def test_chembl_api_url(self):
        """Test ChEMBL API URL default."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert 'chembl' in settings.CHEMBL_API_URL.lower()
        assert 'https://' in settings.CHEMBL_API_URL

    def test_pdb_api_url(self):
        """Test PDB API URL default."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert 'rcsb' in settings.PDB_API_URL.lower()
        assert 'https://' in settings.PDB_API_URL


class TestRateLimitSettings:
    """Tests for rate limiting settings."""

    def test_default_rate_limit_max_jobs(self):
        """Test default rate limit max jobs."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.RATE_LIMIT_MAX_JOBS == 10

    def test_override_rate_limit_max_jobs(self):
        """Test rate limit max jobs can be overridden."""
        from backend.config import Settings

        with patch.dict(os.environ, {'RATE_LIMIT_MAX_JOBS': '20'}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.RATE_LIMIT_MAX_JOBS == 20

    def test_default_rate_limit_window(self):
        """Test default rate limit window."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.RATE_LIMIT_WINDOW == 60

    def test_rate_limit_enabled_default(self):
        """Test rate limit is enabled by default."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.RATE_LIMIT_ENABLED == True


class TestAzureSettings:
    """Tests for Azure settings."""

    def test_default_azure_container(self):
        """Test default Azure container name."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.AZURE_CONTAINER == "impulator"

    def test_azure_connection_string_empty_default(self):
        """Test Azure connection string is empty by default."""
        from backend.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

        assert settings.AZURE_CONNECTION_STRING == ""
