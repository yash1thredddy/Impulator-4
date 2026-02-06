"""
Reset the database and Azure storage for a fresh start.

Deletes the local SQLite database, recreates empty tables,
optionally clears all result ZIPs from Azure, and syncs
the fresh database to Azure.

Usage:
    # Reset DB only (keep Azure result ZIPs)
    python -m backend.scripts.reset_database

    # Reset DB + delete all Azure result ZIPs
    python -m backend.scripts.reset_database --purge-azure

    # Skip confirmation prompt
    python -m backend.scripts.reset_database --yes
    python -m backend.scripts.reset_database --purge-azure --yes
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings  # noqa: E402
from backend.core.azure_sync import (  # noqa: E402
    is_azure_configured,
    _get_container_client,
    sync_db_to_azure,
)
from backend.core.database import init_db  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def delete_local_db():
    """Delete the local SQLite database and WAL/SHM files."""
    db_path = Path(settings.DATABASE_URL.replace("sqlite:///./", ""))
    deleted = []

    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
            deleted.append(str(p))
            logger.info(f"Deleted: {p}")

    if not deleted:
        logger.info("No local database files found")
    return deleted


def delete_local_results():
    """Delete the local results directory."""
    results_dir = settings.RESULTS_DIR
    if results_dir.exists():
        count = sum(1 for _ in results_dir.rglob("*.zip"))
        shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Deleted {count} local result ZIP files from {results_dir}")
        return count
    return 0


def purge_azure_results():
    """Delete all result ZIP files from Azure Blob storage."""
    if not is_azure_configured():
        logger.warning("Azure not configured - skipping Azure purge")
        return 0

    container = _get_container_client()
    if container is None:
        logger.error("Failed to connect to Azure container")
        return 0

    # List and delete all result blobs
    deleted = 0
    blobs = list(container.list_blobs(name_starts_with="results/"))
    logger.info(f"Found {len(blobs)} result blobs in Azure")

    for blob in blobs:
        try:
            container.delete_blob(blob.name)
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to delete {blob.name}: {e}")

    logger.info(f"Deleted {deleted}/{len(blobs)} result blobs from Azure")
    return deleted


def main():
    parser = argparse.ArgumentParser(
        description="Reset database and storage for a fresh start",
    )
    parser.add_argument(
        "--purge-azure",
        action="store_true",
        help="Also delete all result ZIP files from Azure Blob storage",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    # Confirmation
    if not args.yes:
        msg = "This will DELETE the local database and all local results"
        if args.purge_azure:
            msg += " AND all Azure result ZIPs"
        msg += ".\nThis action is IRREVERSIBLE. Continue? [y/N] "

        response = input(msg).strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    logger.info("=" * 60)
    logger.info("RESETTING DATABASE")
    logger.info("=" * 60)

    # 1. Delete local database
    logger.info("Step 1: Deleting local database...")
    delete_local_db()

    # 2. Delete local result ZIPs
    logger.info("Step 2: Deleting local results...")
    delete_local_results()

    # 3. Optionally purge Azure results
    if args.purge_azure:
        logger.info("Step 3: Purging Azure result ZIPs...")
        purge_azure_results()
    else:
        logger.info("Step 3: Skipping Azure purge (use --purge-azure to enable)")

    # 4. Recreate fresh database
    logger.info("Step 4: Creating fresh database with empty tables...")
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    logger.info("Fresh database created")

    # 5. Sync to Azure
    logger.info("Step 5: Syncing fresh database to Azure...")
    if is_azure_configured():
        if sync_db_to_azure():
            logger.info("Empty database synced to Azure successfully")
        else:
            logger.error("Failed to sync database to Azure")
    else:
        logger.warning("Azure not configured - skipped sync")

    logger.info("=" * 60)
    logger.info("RESET COMPLETE - Fresh start ready!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
