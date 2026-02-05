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
    # Parse database path from settings.DATABASE_URL (not hardcoded)
    db_url = settings.DATABASE_URL
    # Handle both "sqlite:///./path" and "sqlite:///path" formats
    db_path_str = db_url.replace("sqlite:///", "").replace("./", "")
    db_path = Path(db_path_str)

    # If relative path, resolve from project root
    if not db_path.is_absolute():
        db_path = Path(settings.DATA_DIR) / db_path.name

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
        else:
            # Add the column
            print("Adding 'idempotency_key' column to jobs table...")
            cursor.execute("""
                ALTER TABLE jobs
                ADD COLUMN idempotency_key VARCHAR(64)
            """)

        # === INDEX RECONCILIATION ===
        # Always check and fix indexes regardless of whether column existed
        print("Checking existing indexes on jobs table...")

        # Get all indexes on jobs table
        cursor.execute("PRAGMA index_list('jobs')")
        indexes = cursor.fetchall()

        # Find any existing index on (session_id, idempotency_key)
        old_index_names = []
        for idx in indexes:
            idx_name = idx[1]
            idx_unique = idx[2]

            # Check columns in this index
            cursor.execute(f"PRAGMA index_info('{idx_name}')")
            idx_columns = [row[2] for row in cursor.fetchall()]

            # If this index covers session_id and idempotency_key but is NOT unique, mark for removal
            if 'session_id' in idx_columns and 'idempotency_key' in idx_columns:
                if not idx_unique and idx_name != 'uix_job_session_idempotency':
                    old_index_names.append(idx_name)
                    print(f"  Found non-unique index to remove: {idx_name}")

        # Drop any old non-unique indexes
        for old_idx in old_index_names:
            print(f"Dropping non-unique index '{old_idx}'...")
            cursor.execute(f"DROP INDEX IF EXISTS {old_idx}")

        # Create UNIQUE index to enforce idempotency constraint
        # Uses the same name as defined in the ORM model: uix_job_session_idempotency
        print("Creating UNIQUE index on (session_id, idempotency_key)...")
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uix_job_session_idempotency
            ON jobs(session_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL
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
