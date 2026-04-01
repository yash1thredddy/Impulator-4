"""Import-isolation regression tests."""

import importlib
import os
import sys
from unittest.mock import patch


def test_import_backend_core_no_db():
    """Importing backend.core should not require database initialization."""
    old_url = os.environ.pop("DATABASE_URL", None)
    try:
        sys.modules.pop("backend.core", None)
        module = importlib.import_module("backend.core")
        assert hasattr(module, "sanitize_compound_name")
    finally:
        if old_url is not None:
            os.environ["DATABASE_URL"] = old_url


def test_import_backend_client_without_streamlit():
    """Low-level API client import should not require Streamlit."""
    sys.modules.pop("frontend.services", None)
    sys.modules.pop("frontend.services.backend_client", None)
    module = importlib.import_module("frontend.services.backend_client")
    assert hasattr(module, "ImpulatorAPIClient")


def test_get_engine_uses_fresh_settings_after_env_change():
    """Lazy engine init should honor env/cache changes made after config import."""
    original_url = os.environ.get("DATABASE_URL")
    test_url = "postgresql://fresh_user:fresh_pass@localhost:5432/fresh_db"

    sys.modules.pop("backend.core.database", None)
    config_module = importlib.import_module("backend.config")
    config_module.get_settings.cache_clear()

    try:
        os.environ["DATABASE_URL"] = test_url
        config_module.get_settings.cache_clear()

        database_module = importlib.import_module("backend.core.database")

        with patch.object(database_module, "create_engine") as mock_create_engine:
            mock_create_engine.return_value = object()
            database_module.engine = None
            database_module.SessionLocal = None

            database_module.get_engine()

        called_url = mock_create_engine.call_args.args[0]
        assert called_url == test_url
    finally:
        database_module = sys.modules.get("backend.core.database")
        if database_module is not None:
            database_module.engine = None
            database_module.SessionLocal = None
        if original_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original_url
        config_module.get_settings.cache_clear()
