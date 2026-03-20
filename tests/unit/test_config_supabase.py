"""Unit tests for Supabase configuration validation (CON-03).

Tests DATABASE_URL validation:
- Rejects empty URLs in production mode
- Rejects non-Postgres URLs in production mode
- Normalizes postgres:// to postgresql://
- Allows SQLite in TESTING mode for CI compatibility
"""
import os

import pytest
from unittest.mock import patch


class TestDatabaseUrlValidation:
    """Tests for DATABASE_URL field_validator in Settings."""

    def test_empty_database_url_fails_without_testing(self):
        """Empty DATABASE_URL raises ValueError when TESTING is not set."""
        with patch.dict(os.environ, {"DATABASE_URL": "", "TESTING": "false"}, clear=False):
            from pydantic import ValidationError
            from backend.config import Settings

            with pytest.raises(ValidationError, match="DATABASE_URL must be set"):
                Settings(DATABASE_URL="", TESTING=False)

    def test_sqlite_url_rejected_without_testing(self):
        """SQLite URL raises ValueError when TESTING is not set."""
        with patch.dict(os.environ, {"TESTING": "false"}, clear=False):
            from pydantic import ValidationError
            from backend.config import Settings

            with pytest.raises(ValidationError, match="PostgreSQL connection string"):
                Settings(DATABASE_URL="sqlite:///test.db", TESTING=False)

    def test_postgres_url_accepted(self):
        """Valid postgresql:// URL is accepted."""
        with patch.dict(os.environ, {"TESTING": "false"}, clear=False):
            from backend.config import Settings

            s = Settings(DATABASE_URL="postgresql://user:pass@host:5432/db", TESTING=False)
            assert s.DATABASE_URL == "postgresql://user:pass@host:5432/db"

    def test_postgres_scheme_normalized(self):
        """postgres:// URL is normalized to postgresql://."""
        with patch.dict(os.environ, {"TESTING": "false"}, clear=False):
            from backend.config import Settings

            s = Settings(DATABASE_URL="postgres://user:pass@host:5432/db", TESTING=False)
            assert s.DATABASE_URL.startswith("postgresql://")
            assert not s.DATABASE_URL.startswith("postgres://user")

    def test_sqlite_allowed_in_testing_mode(self):
        """SQLite URL is allowed when TESTING=true (CI compatibility)."""
        with patch.dict(os.environ, {"TESTING": "true"}, clear=False):
            from backend.config import Settings

            s = Settings(DATABASE_URL="sqlite:///:memory:", TESTING=True)
            assert s.DATABASE_URL == "sqlite:///:memory:"

    def test_empty_url_gets_default_in_testing(self):
        """Empty DATABASE_URL gets sqlite:///:memory: default in TESTING mode."""
        with patch.dict(os.environ, {"TESTING": "true"}, clear=False):
            from backend.config import Settings

            s = Settings(DATABASE_URL="", TESTING=True)
            assert s.DATABASE_URL == "sqlite:///:memory:"
