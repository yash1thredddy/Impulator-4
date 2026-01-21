"""
Rebuild the Compound table from Azure ZIP files.

Use this script when the database is out of sync with Azure storage.
It scans all ZIP files in Azure and extracts compound metadata from summary.json.

Usage:
    python -m backend.scripts.rebuild_db_from_azure
"""
import io
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.core.database import SessionLocal, init_db
from backend.core.azure_sync import (
    is_azure_configured,
    _get_container_client,
    _is_uuid_path,
    _extract_entry_id_from_blob,
    sync_db_to_azure,
)
from backend.models.database import Compound
from backend.services.job_service import generate_inchikey, generate_canonical_smiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_zip_to_memory(container, blob_name: str) -> bytes:
    """Download a ZIP file from Azure into memory."""
    blob_client = container.get_blob_client(blob_name)
    download_stream = blob_client.download_blob()
    return download_stream.readall()


def extract_summary_from_zip(zip_data: bytes) -> dict:
    """Extract summary.json from a ZIP file in memory."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            if "summary.json" in zf.namelist():
                with zf.open("summary.json") as f:
                    return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to extract summary.json: {e}")
    return {}


def rebuild_compound_from_summary(db, entry_id: str, storage_path: str, summary: dict) -> bool:
    """Create or update a Compound entry from summary data."""
    compound_name = summary.get('compound_name')
    if not compound_name:
        logger.warning(f"No compound_name in summary for {entry_id}")
        return False

    smiles = summary.get('smiles') or summary.get('query_smiles')

    # Generate InChIKey and canonical SMILES
    inchikey = generate_inchikey(smiles) if smiles else None
    canonical_smiles = generate_canonical_smiles(smiles) if smiles else None

    # Extract additional summary fields for home page display
    similarity_threshold = summary.get('similarity_threshold', 90)
    qed = summary.get('qed', 0.0)
    num_outliers = summary.get('num_outliers', 0)

    # Check if compound already exists
    existing = db.query(Compound).filter(Compound.entry_id == entry_id).first()

    if existing:
        # Update existing
        existing.compound_name = compound_name
        existing.smiles = smiles
        existing.canonical_smiles = canonical_smiles
        existing.inchikey = inchikey
        existing.chembl_id = summary.get('chembl_id', '')
        existing.total_activities = summary.get('total_activities', 0)
        existing.imp_candidates = summary.get('imp_candidates', 0)
        existing.avg_oqpla_score = summary.get('avg_oqpla_score')
        existing.similarity_threshold = similarity_threshold
        existing.qed = qed
        existing.num_outliers = num_outliers
        existing.storage_path = storage_path
        existing.processed_at = datetime.now(timezone.utc)
        logger.info(f"Updated: {compound_name} ({entry_id}) [threshold={similarity_threshold}%, qed={qed:.2f}, outliers={num_outliers}]")
    else:
        # Create new
        compound = Compound(
            entry_id=entry_id,
            compound_name=compound_name,
            smiles=smiles,
            canonical_smiles=canonical_smiles,
            inchikey=inchikey,
            chembl_id=summary.get('chembl_id', ''),
            total_activities=summary.get('total_activities', 0),
            imp_candidates=summary.get('imp_candidates', 0),
            avg_oqpla_score=summary.get('avg_oqpla_score'),
            similarity_threshold=similarity_threshold,
            qed=qed,
            num_outliers=num_outliers,
            storage_path=storage_path,
            is_duplicate=False,
            duplicate_of=None,
            processed_at=datetime.now(timezone.utc),
        )
        db.add(compound)
        logger.info(f"Created: {compound_name} ({entry_id}) [threshold={similarity_threshold}%, qed={qed:.2f}, outliers={num_outliers}]")

    return True


def main():
    if not is_azure_configured():
        logger.error("Azure is not configured. Set AZURE_CONNECTION_STRING.")
        sys.exit(1)

    container = _get_container_client()
    if container is None:
        logger.error("Failed to connect to Azure container.")
        sys.exit(1)

    # Initialize database
    init_db()
    db = SessionLocal()

    try:
        # List all result blobs in Azure
        logger.info("Scanning Azure for result ZIP files...")
        blobs = list(container.list_blobs(name_starts_with="results/"))

        uuid_blobs = [b for b in blobs if _is_uuid_path(b.name)]
        logger.info(f"Found {len(uuid_blobs)} UUID-based result files in Azure")

        success_count = 0
        error_count = 0

        for i, blob in enumerate(uuid_blobs, 1):
            entry_id = _extract_entry_id_from_blob(blob.name)
            if not entry_id:
                logger.warning(f"Could not extract entry_id from {blob.name}")
                error_count += 1
                continue

            logger.info(f"[{i}/{len(uuid_blobs)}] Processing {blob.name}...")

            try:
                # Download ZIP to memory
                zip_data = download_zip_to_memory(container, blob.name)

                # Extract summary
                summary = extract_summary_from_zip(zip_data)
                if not summary:
                    logger.warning(f"No summary found in {blob.name}")
                    error_count += 1
                    continue

                # Rebuild compound entry
                if rebuild_compound_from_summary(db, entry_id, blob.name, summary):
                    success_count += 1
                else:
                    error_count += 1

            except Exception as e:
                logger.error(f"Error processing {blob.name}: {e}")
                error_count += 1

        # Commit all changes
        db.commit()
        logger.info(f"Database updated: {success_count} compounds, {error_count} errors")

        # Sync database back to Azure
        logger.info("Syncing updated database to Azure...")
        if sync_db_to_azure():
            logger.info("Database synced to Azure successfully!")
        else:
            logger.error("Failed to sync database to Azure")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
