"""
Backfill script to populate Compound table from ZIP files.

This script reads the summary.json from each compound's ZIP file
and updates the Compound table with the missing data.

NOTE: This script uses UUID-based storage paths (entry_id).
Legacy name-based paths are no longer supported.

Usage:
    python -m backend.scripts.backfill_compounds
"""
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import settings
from backend.core.database import SessionLocal
from backend.core.azure_sync import download_result_from_azure_by_entry_id, is_azure_configured
from backend.models.database import Compound

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_summary_from_zip(zip_path: str) -> dict:
    """Extract summary.json from a ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Try different possible summary file names
            for name in ['summary.json', 'metadata.json']:
                # Check both root and with compound name prefix
                for member in zf.namelist():
                    if member.endswith(name):
                        with zf.open(member) as f:
                            return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error extracting summary from {zip_path}: {e}")
        return {}


def backfill_compound(db, compound: Compound) -> bool:
    """
    Backfill a single compound's data from its ZIP file.

    Uses entry_id (UUID) to locate the ZIP file.

    Returns True if updated, False otherwise.
    """
    compound_name = compound.compound_name
    entry_id = compound.entry_id

    if not entry_id:
        logger.warning(f"No entry_id for {compound_name}, skipping")
        return False

    logger.info(f"Processing: {compound_name} (entry_id: {entry_id})")

    # UUID-based local path: results/{prefix}/{entry_id}.zip
    prefix = entry_id[:2].lower()
    local_path = Path(settings.RESULTS_DIR) / prefix / f"{entry_id}.zip"

    # If not local, try to download from Azure
    if not local_path.exists() and is_azure_configured():
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        if download_result_from_azure_by_entry_id(entry_id, tmp_path):
            local_path = Path(tmp_path)
        else:
            logger.warning(f"Could not download {entry_id} from Azure")
            return False

    if not local_path.exists():
        logger.warning(f"ZIP file not found for {compound_name} (entry_id: {entry_id})")
        return False

    # Extract summary
    summary = extract_summary_from_zip(str(local_path))

    if not summary:
        logger.warning(f"No summary found in ZIP for {compound_name}")
        return False

    # Update compound record
    updated = False

    # SMILES
    smiles = summary.get('smiles') or summary.get('query_smiles')
    if smiles and not compound.smiles:
        compound.smiles = smiles
        updated = True

    # ChEMBL ID
    chembl_id = summary.get('chembl_id')
    if chembl_id and not compound.chembl_id:
        compound.chembl_id = chembl_id
        updated = True

    # Total activities
    total_activities = summary.get('total_activities') or summary.get('total_bioactivity_rows', 0)
    if total_activities and compound.total_activities == 0:
        compound.total_activities = total_activities
        updated = True

    # IMP candidates
    imp_candidates = summary.get('imp_candidates', 0)
    if imp_candidates and compound.imp_candidates == 0:
        compound.imp_candidates = imp_candidates
        updated = True

    # Average OQPLA score
    avg_oqpla = summary.get('avg_oqpla_score')
    if avg_oqpla is not None and compound.avg_oqpla_score is None:
        compound.avg_oqpla_score = avg_oqpla
        updated = True

    # Clean up temp file if we downloaded from Azure
    if 'tmp' in str(local_path):
        try:
            os.unlink(local_path)
        except:
            pass

    if updated:
        logger.info(f"Updated {compound_name}: smiles={bool(smiles)}, chembl_id={chembl_id}, "
                   f"activities={total_activities}, imps={imp_candidates}, oqpla={avg_oqpla}")
    else:
        logger.info(f"No updates needed for {compound_name}")

    return updated


def main():
    """Main backfill function."""
    logger.info("Starting compound data backfill...")

    db = SessionLocal()

    try:
        # Get all compounds with entry_id (UUID-based storage)
        compounds = db.query(Compound).filter(Compound.entry_id.isnot(None)).all()
        logger.info(f"Found {len(compounds)} compounds with entry_id to process")

        updated_count = 0
        error_count = 0

        for compound in compounds:
            try:
                if backfill_compound(db, compound):
                    updated_count += 1
            except Exception as e:
                logger.error(f"Error processing {compound.compound_name}: {e}")
                error_count += 1

        # Commit all updates
        db.commit()

        logger.info(f"Backfill complete: {updated_count} updated, {error_count} errors")

        # Sync to Azure
        if is_azure_configured():
            from backend.core.azure_sync import sync_db_to_azure
            sync_db_to_azure()
            logger.info("Database synced to Azure")

    finally:
        db.close()


if __name__ == "__main__":
    main()
