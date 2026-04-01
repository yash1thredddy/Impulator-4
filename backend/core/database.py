"""Database setup with SQLAlchemy (Postgres via Supabase)."""
import logging
from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

from backend.config import get_settings

logger = logging.getLogger(__name__)

engine: Engine | None = None
SessionLocal: sessionmaker | None = None
_engine_lock = Lock()
Base = declarative_base()


def get_engine() -> Engine:
    """Get the lazily initialized SQLAlchemy engine."""
    global engine, SessionLocal
    if engine is None:
        with _engine_lock:
            if engine is None:
                settings = get_settings()
                logger.debug("Initializing database engine")
                _new_engine = create_engine(
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
                    bind=_new_engine,
                    expire_on_commit=False,  # D-19: prevent lazy-load bugs in async/background context
                )
                # Set engine last — it's the sentinel for the lock-free fast path.
                # SessionLocal must be ready before other threads see engine != None.
                engine = _new_engine
    return engine


def get_session_local() -> sessionmaker:
    """Get the lazily initialized sessionmaker."""
    global SessionLocal
    if SessionLocal is None:
        get_engine()
    return SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI endpoints."""
    db = get_session_local()()
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
    db = get_session_local()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
