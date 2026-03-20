"""Unit tests for database engine configuration (CON-01, CON-02).

Tests engine setup:
- _is_postgres flag set correctly for SQLite in test environment
- SQLite engine uses correct dialect
- SQLite engine does not use QueuePool
"""
from sqlalchemy.pool import QueuePool


class TestIsPostgresFlag:
    """Tests for _is_postgres module flag."""

    def test_is_postgres_false_for_sqlite(self):
        """In TESTING mode with SQLite, _is_postgres is False."""
        # The current test environment uses SQLite (TESTING=true)
        from backend.core.database import _is_postgres

        assert _is_postgres is False

    def test_sqlite_engine_dialect(self):
        """SQLite engine reports sqlite dialect."""
        from backend.core.database import engine

        assert engine.dialect.name == "sqlite"


class TestEngineConfiguration:
    """Tests for engine pool configuration."""

    def test_sqlite_engine_is_not_queuepool(self):
        """SQLite engine does not use QueuePool."""
        from backend.core.database import engine

        assert not isinstance(engine.pool, QueuePool)
