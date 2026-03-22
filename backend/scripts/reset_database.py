"""
Reset the database for a fresh start using Alembic.

Runs `alembic downgrade base` then `alembic upgrade head` to recreate
all tables from scratch. Optionally clears all result ZIPs from Azure.

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
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings  # noqa: E402
from backend.core.azure_sync import (  # noqa: E402
    is_azure_configured,
    _get_container_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Directory containing alembic.ini
BACKEND_DIR = Path(__file__).parent.parent


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
        msg = "This will DROP all database tables and recreate them, and delete all local results"
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

    # 1. Delete local result ZIPs
    logger.info("Step 1: Deleting local results...")
    delete_local_results()

    # 2. Optionally purge Azure results
    if args.purge_azure:
        logger.info("Step 2: Purging Azure result ZIPs...")
        purge_azure_results()
    else:
        logger.info("Step 2: Skipping Azure purge (use --purge-azure to enable)")

    # 3. Downgrade database to base (drop all tables)
    logger.info("Step 3: Running alembic downgrade base...")
    subprocess.run(
        ["alembic", "downgrade", "base"],
        check=True,
        cwd=str(BACKEND_DIR),
    )
    logger.info("All tables dropped")

    # 4. Upgrade database to head (recreate all tables)
    logger.info("Step 4: Running alembic upgrade head...")
    subprocess.run(
        ["alembic", "upgrade", "head"],
        check=True,
        cwd=str(BACKEND_DIR),
    )
    logger.info("Fresh database created")

    logger.info("=" * 60)
    logger.info("RESET COMPLETE - Fresh start ready!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
