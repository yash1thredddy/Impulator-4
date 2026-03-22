"""Database setup with SQLAlchemy (Postgres via Supabase)."""
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool

from backend.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.DB_POOL_SIZE,          # D-20: configurable (default 5)
    max_overflow=settings.DB_MAX_OVERFLOW,     # D-20: configurable (default 5)
    pool_timeout=settings.DB_POOL_TIMEOUT,     # D-20: configurable (default 10)
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=settings.DB_ECHO,
    connect_args={
        "application_name": "impulator",
        "options": "-c statement_timeout=60000 -c idle_in_transaction_session_timeout=30000",
    },
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # D-19: prevent lazy-load bugs in async/background context
)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for non-FastAPI code (workers, scripts)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
