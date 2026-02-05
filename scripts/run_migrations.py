#!/usr/bin/env python3
"""
Database migration script with Azure sync.

Run this script to apply all pending database migrations and sync to Azure.

Usage:
    python scripts/run_migrations.py
    python scripts/run_migrations.py --sync-only  # Only sync to Azure, no migrations
    python scripts/run_migrations.py --no-sync    # Run migrations without Azure sync
"""
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings
from backend.core.database import init_db
from backend.core.azure_sync import (
    sync_db_to_azure,
    download_db_from_azure,
    is_azure_configured,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_migrations(skip_sync: bool = False, sync_only: bool = False) -> bool:
    """
    Run database migrations and optionally sync to Azure.

    Args:
        skip_sync: If True, skip Azure sync after migrations
        sync_only: If True, only sync to Azure without running migrations

    Returns:
        True if all operations succeeded, False otherwise
    """
    success = True

    # Ensure data directory exists
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Download latest database from Azure first (if configured)
    # CRITICAL: If download fails, we must not proceed to avoid data loss
    if is_azure_configured() and not sync_only:
        print("📥 Downloading database from Azure...")
        try:
            download_success = download_db_from_azure()
            if download_success:
                print("✓ Database downloaded from Azure")
            else:
                # download_db_from_azure returns False on actual errors (not "blob not found")
                # Proceeding would risk overwriting production data with a stale/empty database
                print("✗ Failed to download database from Azure")
                print("  Aborting to prevent potential data loss.")
                print("  If this is a fresh deployment, use --no-sync to skip Azure download.")
                return False
        except Exception as e:
            print(f"✗ Azure download error: {e}")
            print("  Aborting to prevent potential data loss.")
            print("  If this is a fresh deployment, use --no-sync to skip Azure download.")
            return False

    # Run migrations (unless sync_only)
    if not sync_only:
        print("\n🔧 Running database migrations...")
        try:
            # Initialize database and run migrations
            init_db()
            print("✓ Database initialized and migrations applied")
        except Exception as e:
            print(f"✗ Migration failed: {e}")
            logger.error(f"Migration failed: {e}", exc_info=True)
            success = False

    # Sync to Azure (unless skip_sync or migrations failed)
    if not skip_sync and success:
        if is_azure_configured():
            print("\n☁️ Syncing database to Azure...")
            try:
                sync_db_to_azure()
                print("✓ Database synced to Azure")
            except Exception as e:
                print(f"✗ Azure sync failed: {e}")
                logger.error(f"Azure sync failed: {e}", exc_info=True)
                success = False
        else:
            print("\n⚠️ Azure not configured, skipping sync")

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run database migrations with Azure sync"
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip Azure sync after migrations"
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync to Azure without running migrations"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  IMPULATOR Database Migration")
    print("=" * 60)
    print(f"\nDatabase: {settings.DATABASE_URL}")
    print(f"Azure configured: {is_azure_configured()}")
    print()

    success = run_migrations(
        skip_sync=args.no_sync,
        sync_only=args.sync_only
    )

    print()
    if success:
        print("=" * 60)
        print("  ✓ Migration completed successfully")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("  ✗ Migration completed with errors")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
