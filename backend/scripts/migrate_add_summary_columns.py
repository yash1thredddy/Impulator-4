"""
Add missing summary columns to compounds table.

This migration adds columns for home page display without ZIP downloads:
- similarity_threshold: Similarity threshold % used for search
- qed: Average QED score
- num_outliers: Number of outliers detected

Run: python -m backend.scripts.migrate_add_summary_columns
"""
import sqlite3
import sys
import logging
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


def migrate():
    """Add missing columns to compounds table."""
    # Extract database path from URL
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        # Handle relative paths
        if db_path.startswith("./"):
            db_path = PROJECT_ROOT / db_path[2:]
        else:
            db_path = Path(db_path)
    else:
        logger.error(f"Unsupported database URL format: {db_url}")
        sys.exit(1)

    logger.info(f"Database path: {db_path}")

    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check existing columns
        cursor.execute("PRAGMA table_info(compounds)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        logger.info(f"Existing columns: {sorted(existing_cols)}")

        # Define migrations
        migrations = [
            ("similarity_threshold", "INTEGER DEFAULT 90"),
            ("qed", "REAL"),
            ("num_outliers", "INTEGER DEFAULT 0"),
        ]

        # Add missing columns
        for col_name, col_type in migrations:
            if col_name not in existing_cols:
                logger.info(f"Adding column: {col_name} ({col_type})")
                cursor.execute(f"ALTER TABLE compounds ADD COLUMN {col_name} {col_type}")
            else:
                logger.info(f"Column already exists: {col_name}")

        conn.commit()
        logger.info("Migration complete!")

        # Verify columns were added
        cursor.execute("PRAGMA table_info(compounds)")
        new_cols = {row[1] for row in cursor.fetchall()}
        logger.info(f"New columns: {sorted(new_cols)}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
