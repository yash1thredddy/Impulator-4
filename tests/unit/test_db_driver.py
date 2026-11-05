"""Guard: the driver SQLAlchemy picks for a plain postgresql:// URL must be installed.

Prod outage 2026-11-05 06:24 UTC: SQLAlchemy 2.1 switched the default PG
driver to psycopg (v3) while only the v2 driver was installed, so engine
creation raised ModuleNotFoundError. Prod URLs are plain ``postgresql://``
(see the ``backend/config.py`` validator), so we test the default driver,
not an explicit one.
"""
from sqlalchemy import create_engine

EXPECTED_DRIVER = "psycopg"


def test_default_postgres_driver_is_installed():
    # No connection is made; create_engine only resolves and imports the DBAPI.
    engine = create_engine("postgresql://u:p@localhost:5432/x")
    try:
        assert engine.dialect.driver == EXPECTED_DRIVER
    finally:
        engine.dispose()
