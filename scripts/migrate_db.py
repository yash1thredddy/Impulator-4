#!/usr/bin/env python3
"""
Database migration script for IMPULATOR.

Run this before starting the server after updating to a new version.
Creates a backup before applying migrations.

Usage:
    python scripts/migrate_db.py

Options:
    --no-backup    Skip backup creation
    --verify-only  Only verify current schema, don't apply changes
"""
import sys
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.core.database import engine, _apply_migrations, Base
from backend.models.database import Job, Compound  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """Get the database file path from DATABASE_URL."""
    db_url = settings.DATABASE_URL
    # Handle different SQLite URL formats
    db_url = db_url.replace("sqlite:///", "").replace("sqlite:///./", "")
    return Path(db_url)


def create_backup(db_path: Path) -> Path:
    """Create a timestamped backup of the database.

    Args:
        db_path: Path to the database file

    Returns:
        Path to the backup file
    """
    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}, skipping backup")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(f".db.backup_{timestamp}")

    logger.info(f"Creating backup at: {backup_path}")
    shutil.copy2(db_path, backup_path)
    logger.info(f"Backup created successfully ({backup_path.stat().st_size:,} bytes)")

    return backup_path


def verify_schema():
    """Verify the current database schema."""
    from sqlalchemy import text, inspect

    logger.info("Verifying database schema...")

    with engine.connect() as conn:
        # Get jobs table columns
        result = conn.execute(text("PRAGMA table_info(jobs)"))
        jobs_columns = {row[1]: row[2] for row in result.fetchall()}
        logger.info(f"Jobs table columns: {list(jobs_columns.keys())}")

        # Get compounds table columns
        result = conn.execute(text("PRAGMA table_info(compounds)"))
        compounds_columns = {row[1]: row[2] for row in result.fetchall()}
        logger.info(f"Compounds table columns: {list(compounds_columns.keys())}")

        # Get indexes
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"))
        indexes = [row[0] for row in result.fetchall()]
        logger.info(f"Custom indexes: {indexes}")

        # Check for required columns
        required_jobs_columns = ['id', 'status', 'session_id', 'batch_id', 'created_at']
        required_compounds_columns = ['id', 'entry_id', 'compound_name', 'inchikey']

        missing_jobs = [c for c in required_jobs_columns if c not in jobs_columns]
        missing_compounds = [c for c in required_compounds_columns if c not in compounds_columns]

        if missing_jobs:
            logger.warning(f"Missing jobs columns (will be added): {missing_jobs}")
        if missing_compounds:
            logger.warning(f"Missing compounds columns (will be added): {missing_compounds}")

        # Check for required indexes
        required_indexes = [
            'ix_jobs_session_id',
            'ix_jobs_batch_id',
            'ix_jobs_status_created',
            'ix_compounds_inchikey',
        ]
        missing_indexes = [i for i in required_indexes if i not in indexes]
        if missing_indexes:
            logger.warning(f"Missing indexes (will be created): {missing_indexes}")

        return {
            'jobs_columns': jobs_columns,
            'compounds_columns': compounds_columns,
            'indexes': indexes,
            'missing_jobs': missing_jobs,
            'missing_compounds': missing_compounds,
            'missing_indexes': missing_indexes,
        }


def run_migrations():
    """Run database migrations."""
    logger.info("Applying migrations...")

    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created/verified")

    # Apply column and index migrations
    _apply_migrations()
    logger.info("Migrations applied successfully!")


def main():
    parser = argparse.ArgumentParser(description='IMPULATOR Database Migration')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--verify-only', action='store_true', help='Only verify schema')
    args = parser.parse_args()

    logger.info(f"Database: {settings.DATABASE_URL}")
    logger.info(f"Version: {settings.APP_VERSION}")

    db_path = get_db_path()

    # Verify current schema
    schema_info = verify_schema()

    if args.verify_only:
        logger.info("Verification complete (--verify-only mode)")
        return

    # Create backup unless skipped
    if not args.no_backup:
        backup_path = create_backup(db_path)
        if backup_path:
            logger.info(f"Backup saved to: {backup_path}")
    else:
        logger.info("Skipping backup (--no-backup)")

    # Run migrations
    run_migrations()

    # Verify after migrations
    logger.info("\nVerifying schema after migrations...")
    verify_schema()

    logger.info("\nMigration complete!")


if __name__ == "__main__":
    main()
