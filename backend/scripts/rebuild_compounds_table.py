"""
Rebuild compounds table with proper column ordering.

SQLite ALTER TABLE ADD COLUMN always appends columns at the end.
This script rebuilds the table with correct column order matching the ORM model.

Steps:
1. Create backup of database
2. Create new table with correct column order
3. Copy all data from old table
4. Drop old table and rename new one
5. Recreate indexes

Run: python -m backend.scripts.rebuild_compounds_table
"""
import sqlite3
import shutil
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """Extract database path from URL."""
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        # Handle relative paths
        if db_path.startswith("./"):
            return PROJECT_ROOT / db_path[2:]
        else:
            return Path(db_path)
    else:
        raise ValueError(f"Unsupported database URL format: {db_url}")


def rebuild_compounds_table():
    """Rebuild compounds table with proper column ordering."""
    db_path = get_db_path()

    logger.info(f"Database path: {db_path}")

    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)

    # Create backup
    backup_path = db_path.parent / f"impulator_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    logger.info(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check if compounds table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compounds'")
        if not cursor.fetchone():
            logger.error("compounds table does not exist")
            sys.exit(1)

        # Get current column info
        cursor.execute("PRAGMA table_info(compounds)")
        current_cols = {row[1]: row for row in cursor.fetchall()}
        logger.info(f"Current columns: {list(current_cols.keys())}")

        # Define the correct column order (matching ORM model)
        # This is the PROPER order we want
        new_table_schema = """
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
            imp_candidates INTEGER DEFAULT 0,
            avg_oqpla_score REAL,
            similarity_threshold INTEGER DEFAULT 90,
            qed REAL,
            num_outliers INTEGER DEFAULT 0,
            storage_path VARCHAR(500),
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """

        logger.info("Creating new table with correct column order...")
        cursor.execute(new_table_schema)

        # Get columns that exist in both old and new tables
        # Build the column list for copying
        columns_to_copy = [
            'id', 'entry_id', 'compound_name', 'chembl_id', 'smiles',
            'inchikey', 'canonical_smiles', 'is_duplicate', 'duplicate_of',
            'total_activities', 'imp_candidates', 'avg_oqpla_score',
            'similarity_threshold', 'qed', 'num_outliers',
            'storage_path', 'processed_at'
        ]

        # Filter to only columns that exist in the old table
        existing_cols = [col for col in columns_to_copy if col in current_cols]

        # Build INSERT statement with defaults for missing columns
        if existing_cols:
            cols_str = ', '.join(existing_cols)
            logger.info(f"Copying columns: {cols_str}")

            # Copy data
            cursor.execute(f"""
                INSERT INTO compounds_new ({cols_str})
                SELECT {cols_str} FROM compounds
            """)

            rows_copied = cursor.rowcount
            logger.info(f"Copied {rows_copied} rows to new table")

        # Drop old table
        logger.info("Dropping old compounds table...")
        cursor.execute("DROP TABLE compounds")

        # Rename new table
        logger.info("Renaming new table to compounds...")
        cursor.execute("ALTER TABLE compounds_new RENAME TO compounds")

        # Recreate indexes
        logger.info("Creating indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS ix_compounds_entry_id ON compounds(entry_id)",
            "CREATE INDEX IF NOT EXISTS ix_compounds_compound_name ON compounds(compound_name)",
            "CREATE INDEX IF NOT EXISTS ix_compounds_chembl_id ON compounds(chembl_id)",
            "CREATE INDEX IF NOT EXISTS ix_compounds_inchikey ON compounds(inchikey)",
            "CREATE INDEX IF NOT EXISTS ix_compounds_is_duplicate ON compounds(is_duplicate)",
        ]

        for idx_sql in indexes:
            cursor.execute(idx_sql)
            logger.info(f"Created index: {idx_sql.split('ix_compounds_')[1].split(' ')[0]}")

        conn.commit()

        # Verify new column order
        cursor.execute("PRAGMA table_info(compounds)")
        new_cols = [row[1] for row in cursor.fetchall()]
        logger.info(f"New column order: {new_cols}")

        # Count rows
        cursor.execute("SELECT COUNT(*) FROM compounds")
        count = cursor.fetchone()[0]
        logger.info(f"Table rebuilt successfully with {count} rows")

        logger.info(f"Backup saved at: {backup_path}")
        logger.info("Done! Table rebuilt with proper column ordering.")

    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        conn.rollback()

        # Restore from backup
        logger.info("Restoring from backup...")
        conn.close()
        shutil.copy2(backup_path, db_path)
        logger.info("Backup restored. Original database preserved.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    rebuild_compounds_table()
