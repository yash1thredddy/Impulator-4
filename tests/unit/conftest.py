"""
Shared fixtures for unit tests that need database access.

Provides:
- db_session: Per-test database session with rollback isolation
"""
import pytest
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session(pg_engine):
    """Per-test database session with automatic rollback for isolation.

    Uses transaction rollback instead of TRUNCATE — much faster for
    tests that seed a few rows then check behavior.
    """
    conn = pg_engine.connect()
    trans = conn.begin()
    Session = sessionmaker(bind=conn)
    session = Session()
    yield session
    session.close()
    trans.rollback()
    conn.close()
