"""
Migration script to add idempotency_key column to jobs table.

Run this before starting the backend:
    python scripts/add_idempotency_key.py
"""
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings


def migrate():
    """Add idempotency_key column to jobs table."""
    db_path = Path(settings.DATA_DIR) / "impulator.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'idempotency_key' in columns:
            print("✓ Column 'idempotency_key' already exists")
            return

        # Add the column
        print("Adding 'idempotency_key' column to jobs table...")
        cursor.execute("""
            ALTER TABLE jobs
            ADD COLUMN idempotency_key VARCHAR(64)
        """)

        # Create index
        print("Creating index on (session_id, idempotency_key)...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS ix_jobs_idempotency
            ON jobs(session_id, idempotency_key)
        """)

        conn.commit()
        print("✓ Migration completed successfully")

    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
