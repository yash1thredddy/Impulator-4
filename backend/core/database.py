"""
SQLite database setup with SQLAlchemy.
"""
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import NullPool

from backend.config import settings

logger = logging.getLogger(__name__)

# Create engine with SQLite-specific settings
# NullPool creates a new connection per request and auto-closes it
# This prevents WAL corruption with ThreadPoolExecutor (2 workers)
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Allow multi-threaded access
        "timeout": 30,  # 30 second timeout for locks
    },
    poolclass=NullPool,  # New connection per request, auto-closed (safe for multi-threaded SQLite)
    echo=settings.DEBUG,
)


# Enable WAL mode and busy timeout for better concurrency
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Configure SQLite for better concurrency and data safety."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=60000")  # 60s busy timeout (STAB-05)
    # FULL sync ensures data integrity - important for scientific data
    cursor.execute("PRAGMA synchronous=FULL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def execute_with_retry(db: Session, operation, max_retries: int = 1):
    """Execute a DB operation with single retry on SQLITE_BUSY.

    Used inside _db_write_lock for additional resilience (STAB-05).
    The busy_timeout PRAGMA handles most contention; this catches edge cases
    where busy_timeout expires but a retry would succeed.

    Args:
        db: SQLAlchemy session
        operation: Callable that performs the DB operation
        max_retries: Number of retries (default 1 = single retry)
    """
    from sqlalchemy.exc import OperationalError as _OperationalError

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except _OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries:
                logger.warning(f"SQLite BUSY, retrying ({attempt + 1}/{max_retries})")
                time.sleep(1.0)
                continue
            raise


# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints.
    Yields a database session with proper transaction handling.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()  # Commit successful transactions
    except Exception:
        db.rollback()  # Rollback on error
        raise
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Use in non-FastAPI code (workers, scripts).
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database tables with migration locking.
    Call this on application startup.
    """
    from backend.models.database import Job, Compound, DeletedCompound  # noqa: F401

    Base.metadata.create_all(bind=engine)

    # Apply schema migrations with lock to prevent race conditions
    _apply_migrations_with_lock()


def _apply_migrations_with_lock() -> None:  # pragma: no cover -- migration lock, tested manually
    """Apply migrations with cross-platform file locking to prevent race conditions."""
    # Get data directory from database URL
    db_path = settings.DATABASE_URL.replace("sqlite:///", "").replace("sqlite:///./", "")
    data_dir = Path(db_path).parent if db_path else Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    lock_file = data_dir / ".migration.lock"
    lock_acquired = False

    try:
        # Try to acquire lock using atomic file creation
        # This is cross-platform (works on Windows and Linux)
        try:
            # O_CREAT | O_EXCL ensures atomic creation - fails if file exists
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            lock_acquired = True
        except FileExistsError:
            # Lock file exists - check if it's stale (older than 60 seconds)
            try:
                lock_age = time.time() - lock_file.stat().st_mtime
                if lock_age > 60:
                    # Stale lock, remove and retry
                    logger.warning(f"Removing stale migration lock (age: {lock_age:.0f}s)")
                    lock_file.unlink()
                    fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    lock_acquired = True
                else:
                    logger.info("Another instance is running migrations, skipping")
                    return
            except (OSError, FileNotFoundError):
                # Race condition - another process handled it
                logger.info("Migration lock contested, skipping")
                return

        if lock_acquired:
            _apply_migrations()

    finally:
        # Release lock
        if lock_acquired:
            try:
                lock_file.unlink()
            except (OSError, FileNotFoundError):
                pass


def _apply_migrations() -> None:  # pragma: no cover -- one-time migration, tested via run_migrations.py
    """Apply schema migrations within a single transaction.

    SQLite doesn't support ALTER TABLE ADD COLUMN IF NOT EXISTS,
    so we check if columns exist first.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Use engine.begin() for atomic transaction - auto-commits on success, rollbacks on error
    with engine.begin() as conn:
        migrations_applied = []

        # ========== JOBS TABLE MIGRATIONS ==========
        # Check if jobs table exists
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        ))
        if result.fetchone() is not None:
            # Get existing columns in jobs table
            result = conn.execute(text("PRAGMA table_info(jobs)"))
            jobs_columns = {row[1] for row in result.fetchall()}

            # Add session_id column if missing
            if 'session_id' not in jobs_columns:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN session_id VARCHAR(36)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_session_id ON jobs(session_id)"))
                migrations_applied.append('jobs.session_id')

            # Add batch_id column if missing
            if 'batch_id' not in jobs_columns:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN batch_id VARCHAR(36)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_batch_id ON jobs(batch_id)"))
                migrations_applied.append('jobs.batch_id')

            # Add idempotency_key column if missing (for safe retries)
            if 'idempotency_key' not in jobs_columns:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN idempotency_key VARCHAR(64)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_idempotency_key ON jobs(idempotency_key)"))
                migrations_applied.append('jobs.idempotency_key')

            # Create unique constraint for idempotency: same key per session = same job
            # SQLite doesn't support adding constraints to existing tables, so we use a unique index
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uix_job_session_idempotency
                ON jobs(session_id, idempotency_key)
                WHERE idempotency_key IS NOT NULL
            """))

            # Add request_id column if missing (correlation ID from originating HTTP request)
            if 'request_id' not in jobs_columns:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN request_id VARCHAR(36)"))
                migrations_applied.append('jobs.request_id')

            # Add error_code column if missing (machine-readable failure classification)
            if 'error_code' not in jobs_columns:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN error_code VARCHAR(50)"))
                migrations_applied.append('jobs.error_code')

            # Performance indexes for jobs table
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_created_at ON jobs(created_at)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_completed_at ON jobs(completed_at)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_status_created ON jobs(status, created_at)"))

            # Partial index for pending jobs (speeds up scheduler queries)
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_jobs_pending_queue
                ON jobs(status, created_at)
                WHERE status = 'pending'
            """))

        # ========== COMPOUNDS TABLE MIGRATIONS ==========
        # Check if compounds table exists
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='compounds'"
        ))
        if result.fetchone() is not None:
            # Get existing columns in compounds table
            result = conn.execute(text("PRAGMA table_info(compounds)"))
            compounds_columns = {row[1] for row in result.fetchall()}

            # Add entry_id column if missing (UUID for unique identification)
            if 'entry_id' not in compounds_columns:
                conn.execute(text("ALTER TABLE compounds ADD COLUMN entry_id VARCHAR(36)"))
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_compounds_entry_id ON compounds(entry_id)"))
                migrations_applied.append('compounds.entry_id')

            # Add inchikey column if missing (for duplicate detection)
            if 'inchikey' not in compounds_columns:
                conn.execute(text("ALTER TABLE compounds ADD COLUMN inchikey VARCHAR(27)"))
                migrations_applied.append('compounds.inchikey')

            # Add canonical_smiles column if missing
            if 'canonical_smiles' not in compounds_columns:
                conn.execute(text("ALTER TABLE compounds ADD COLUMN canonical_smiles TEXT"))
                migrations_applied.append('compounds.canonical_smiles')

            # Add is_duplicate column if missing
            if 'is_duplicate' not in compounds_columns:
                conn.execute(text("ALTER TABLE compounds ADD COLUMN is_duplicate BOOLEAN DEFAULT 0"))
                migrations_applied.append('compounds.is_duplicate')

            # Add duplicate_of column if missing
            if 'duplicate_of' not in compounds_columns:
                conn.execute(text("ALTER TABLE compounds ADD COLUMN duplicate_of VARCHAR(36)"))
                migrations_applied.append('compounds.duplicate_of')

            # Add author_name column if missing
            if 'author_name' not in compounds_columns:
                conn.execute(text("ALTER TABLE compounds ADD COLUMN author_name VARCHAR(100)"))
                migrations_applied.append('compounds.author_name')

            # Add activity_types column if missing (for config-aware duplicate detection)
            if 'activity_types' not in compounds_columns:
                conn.execute(text("ALTER TABLE compounds ADD COLUMN activity_types TEXT"))
                migrations_applied.append('compounds.activity_types')

            # Add similar_compounds column if missing
            if 'similar_compounds' not in compounds_columns:
                conn.execute(text("ALTER TABLE compounds ADD COLUMN similar_compounds INTEGER DEFAULT 0"))
                # Backfill from jobs.result_summary JSON (similar_count or total_compounds)
                # Note: status enum stored as uppercase 'COMPLETED' in SQLite
                conn.execute(text("""
                    UPDATE compounds
                    SET similar_compounds = COALESCE(
                        (SELECT COALESCE(
                            json_extract(j.result_summary, '$.similar_count'),
                            json_extract(j.result_summary, '$.total_compounds')
                        )
                        FROM jobs j
                        WHERE j.result_summary IS NOT NULL
                          AND UPPER(j.status) = 'COMPLETED'
                          AND json_extract(j.result_summary, '$.entry_id') = compounds.entry_id
                        ORDER BY j.completed_at DESC
                        LIMIT 1),
                    0)
                """))
                migrations_applied.append('compounds.similar_compounds (backfilled)')

            # Reorder columns: move similar_compounds after total_activities
            # SQLite requires table recreation to reorder columns
            result = conn.execute(text("PRAGMA table_info(compounds)"))
            col_names = [row[1] for row in result.fetchall()]
            # Only run if similar_compounds exists but is not in the correct position
            if 'similar_compounds' in col_names:
                sa_idx = col_names.index('similar_compounds')
                ta_idx = col_names.index('total_activities')
                if sa_idx != ta_idx + 1:
                    conn.execute(text("DROP TABLE IF EXISTS compounds_new"))
                    conn.execute(text("""
                        CREATE TABLE compounds_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            entry_id VARCHAR(36) UNIQUE,
                            compound_name VARCHAR(255) NOT NULL,
                            chembl_id VARCHAR(50),
                            smiles TEXT,
                            inchikey VARCHAR(27),
                            canonical_smiles TEXT,
                            is_duplicate BOOLEAN DEFAULT 0,
                            duplicate_of VARCHAR(36),
                            total_activities INTEGER DEFAULT 0,
                            similar_compounds INTEGER DEFAULT 0,
                            imp_candidates INTEGER DEFAULT 0,
                            imp_score FLOAT,
                            similarity_threshold INTEGER DEFAULT 90,
                            activity_types TEXT,
                            qed FLOAT,
                            num_outliers INTEGER DEFAULT 0,
                            author_name VARCHAR(100),
                            storage_path VARCHAR(500),
                            processed_at DATETIME
                        )
                    """))
                    conn.execute(text("""
                        INSERT INTO compounds_new (
                            id, entry_id, compound_name, chembl_id, smiles,
                            inchikey, canonical_smiles, is_duplicate, duplicate_of,
                            total_activities, similar_compounds, imp_candidates, imp_score,
                            similarity_threshold, activity_types, qed, num_outliers,
                            author_name, storage_path, processed_at
                        )
                        SELECT
                            id, entry_id, compound_name, chembl_id, smiles,
                            inchikey, canonical_smiles, is_duplicate, duplicate_of,
                            total_activities, similar_compounds, imp_candidates, imp_score,
                            similarity_threshold, activity_types, qed, num_outliers,
                            author_name, storage_path, processed_at
                        FROM compounds
                    """))
                    conn.execute(text("DROP TABLE compounds"))
                    conn.execute(text("ALTER TABLE compounds_new RENAME TO compounds"))
                    migrations_applied.append('compounds column reorder (similar_compounds)')

            # Create indexes AFTER columns exist (and after potential table recreation)
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_entry_id ON compounds(entry_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_compound_name ON compounds(compound_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_chembl_id ON compounds(chembl_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_inchikey ON compounds(inchikey)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_is_duplicate ON compounds(is_duplicate)"))

        # ========== DELETED_COMPOUNDS TABLE MIGRATIONS ==========
        # Check if deleted_compounds table exists, create if not
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='deleted_compounds'"
        ))
        if result.fetchone() is None:
            # Create deleted_compounds audit table
            conn.execute(text("""
                CREATE TABLE deleted_compounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id INTEGER NOT NULL,
                    entry_id VARCHAR(36),
                    compound_name VARCHAR(255) NOT NULL,
                    chembl_id VARCHAR(50),
                    smiles TEXT,
                    inchikey VARCHAR(27),
                    author_name VARCHAR(100),
                    is_duplicate BOOLEAN DEFAULT 0,
                    duplicate_of VARCHAR(36),
                    activity_types TEXT,
                    storage_path VARCHAR(500),
                    deleted_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    deleted_by_session VARCHAR(36),
                    deleted_by_job_id VARCHAR(36),
                    deletion_reason VARCHAR(255),
                    original_processed_at DATETIME
                )
            """))
            # Create indexes for deleted_compounds
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_deleted_compounds_entry_id ON deleted_compounds(entry_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_deleted_compounds_compound_name ON deleted_compounds(compound_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_deleted_compounds_deleted_at ON deleted_compounds(deleted_at)"))
            migrations_applied.append('deleted_compounds table created')
        else:
            # Migrate existing deleted_compounds table
            result = conn.execute(text("PRAGMA table_info(deleted_compounds)"))
            del_columns = {row[1] for row in result.fetchall()}
            if 'author_name' not in del_columns:
                conn.execute(text("ALTER TABLE deleted_compounds ADD COLUMN author_name VARCHAR(100)"))
                migrations_applied.append('deleted_compounds.author_name')
            if 'activity_types' not in del_columns:
                conn.execute(text("ALTER TABLE deleted_compounds ADD COLUMN activity_types TEXT"))
                migrations_applied.append('deleted_compounds.activity_types')

        # ========== PHASE 3: COMPOUNDS TABLE INTEGRITY CONSTRAINTS ==========
        # Add FK on duplicate_of, conditional unique index on inchikey_structure_key
        result = conn.execute(text("PRAGMA table_info(compounds)"))
        compounds_cols_p3 = {row[1] for row in result.fetchall()}

        if 'inchikey_structure_key' not in compounds_cols_p3:
            # Clean orphaned duplicate_of references before FK is added
            conn.execute(text("""
                UPDATE compounds SET duplicate_of = NULL
                WHERE duplicate_of IS NOT NULL
                  AND duplicate_of NOT IN (SELECT entry_id FROM compounds WHERE entry_id IS NOT NULL)
            """))

            # Recreate compounds table with FK constraint and new column
            conn.execute(text("DROP TABLE IF EXISTS compounds_new"))
            conn.execute(text("""
                CREATE TABLE compounds_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id VARCHAR(36) UNIQUE,
                    compound_name VARCHAR(255) NOT NULL,
                    chembl_id VARCHAR(50),
                    smiles TEXT,
                    inchikey VARCHAR(27),
                    inchikey_structure_key VARCHAR(27),
                    canonical_smiles TEXT,
                    is_duplicate BOOLEAN DEFAULT 0,
                    duplicate_of VARCHAR(36) REFERENCES compounds_new(entry_id) ON DELETE SET NULL,
                    total_activities INTEGER DEFAULT 0,
                    similar_compounds INTEGER DEFAULT 0,
                    imp_candidates INTEGER DEFAULT 0,
                    imp_score FLOAT,
                    similarity_threshold INTEGER DEFAULT 90,
                    activity_types TEXT,
                    qed FLOAT,
                    num_outliers INTEGER DEFAULT 0,
                    author_name VARCHAR(100),
                    storage_path VARCHAR(500),
                    processed_at DATETIME
                )
            """))

            # Copy data with computed inchikey_structure_key (first two blocks of InChIKey)
            conn.execute(text("""
                INSERT INTO compounds_new (
                    id, entry_id, compound_name, chembl_id, smiles,
                    inchikey, inchikey_structure_key, canonical_smiles,
                    is_duplicate, duplicate_of,
                    total_activities, similar_compounds, imp_candidates, imp_score,
                    similarity_threshold, activity_types, qed, num_outliers,
                    author_name, storage_path, processed_at
                )
                SELECT
                    id, entry_id, compound_name, chembl_id, smiles,
                    inchikey,
                    CASE
                        WHEN inchikey IS NOT NULL AND inchikey LIKE '%-%-%'
                        THEN SUBSTR(inchikey, 1, INSTR(inchikey, '-') - 1) || '-' ||
                             SUBSTR(
                                SUBSTR(inchikey, INSTR(inchikey, '-') + 1),
                                1,
                                CASE WHEN INSTR(SUBSTR(inchikey, INSTR(inchikey, '-') + 1), '-') > 0
                                     THEN INSTR(SUBSTR(inchikey, INSTR(inchikey, '-') + 1), '-') - 1
                                     ELSE LENGTH(SUBSTR(inchikey, INSTR(inchikey, '-') + 1))
                                END
                             )
                        ELSE NULL
                    END,
                    canonical_smiles,
                    is_duplicate, duplicate_of,
                    total_activities, similar_compounds, imp_candidates, imp_score,
                    similarity_threshold, activity_types, qed, num_outliers,
                    author_name, storage_path, processed_at
                FROM compounds
            """))

            # Drop old table and rename
            conn.execute(text("DROP TABLE compounds"))
            conn.execute(text("ALTER TABLE compounds_new RENAME TO compounds"))

            # Recreate ALL indexes (not carried over in table recreation)
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_entry_id ON compounds(entry_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_compound_name ON compounds(compound_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_chembl_id ON compounds(chembl_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_inchikey ON compounds(inchikey)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_is_duplicate ON compounds(is_duplicate)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_compounds_inchikey_structure_key ON compounds(inchikey_structure_key)"))

            # Conditional unique index: prevent concurrent non-duplicate compounds with same structure key
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uix_compounds_structure_key_non_dup
                ON compounds(inchikey_structure_key)
                WHERE is_duplicate = 0 AND inchikey_structure_key IS NOT NULL
            """))

            migrations_applied.append('Phase 3: compounds table recreation (FK, unique constraint, inchikey_structure_key)')

        if migrations_applied:
            logger.info(f"Applied migrations: {migrations_applied}")
