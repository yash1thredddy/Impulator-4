"""
Compounds API endpoints.

Provides access to processed compound data from the database.
Includes CRUD operations for compound management.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, Header
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.core.database import get_db
from backend.core.auth import validate_session_id, truncate_session_id
from backend.core.azure_sync import delete_result_from_azure, delete_result_from_azure_by_entry_id
from backend.core.audit import log_job_deleted
from backend.core import sanitize_compound_name
from backend.models.database import Compound, DeletedCompound
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compounds", tags=["Compounds"])


@router.get("")
async def list_compounds(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by compound name"),
    include_duplicates: bool = Query(False, description="Include duplicate entries"),
) -> dict:
    """
    List all processed compounds from the database.

    Returns compound metadata including names, entry_ids, and summary stats.
    This is the authoritative source for compound information (not blob names).

    Args:
        page: Page number (1-indexed)
        per_page: Number of items per page (max 100)
        search: Optional search term for compound name
        include_duplicates: Whether to include duplicate entries

    Returns:
        Paginated list of compounds with metadata
    """
    query = db.query(Compound)

    # Filter out duplicates unless requested
    if not include_duplicates:
        query = query.filter(Compound.is_duplicate == False)  # noqa: E712

    # Apply search filter
    if search:
        query = query.filter(Compound.compound_name.ilike(f"%{search}%"))

    # Get total count
    total = query.count()

    # Apply pagination and ordering
    offset = (page - 1) * per_page
    compounds = (
        query
        .order_by(desc(Compound.processed_at))
        .offset(offset)
        .limit(per_page)
        .all()
    )

    # Convert to response format
    items = []
    for compound in compounds:
        items.append({
            "entry_id": compound.entry_id,
            "compound_name": compound.compound_name,
            "chembl_id": compound.chembl_id,
            "smiles": compound.smiles,
            "inchikey": compound.inchikey,
            "total_activities": compound.total_activities,
            "imp_candidates": compound.imp_candidates,
            "avg_oqpla_score": compound.avg_oqpla_score,
            "storage_path": compound.storage_path,
            "processed_at": compound.processed_at.isoformat() if compound.processed_at else None,
            "is_duplicate": compound.is_duplicate,
            "duplicate_of": compound.duplicate_of,
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }


@router.get("/{entry_id}")
async def get_compound(
    entry_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """
    Get a specific compound by entry_id.

    Args:
        entry_id: UUID of the compound entry

    Returns:
        Compound metadata
    """
    compound = db.query(Compound).filter(Compound.entry_id == entry_id).first()

    if not compound:
        raise HTTPException(status_code=404, detail="Compound not found")

    return compound.to_dict()


@router.delete("/{entry_id}")
async def delete_compound(
    entry_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
) -> dict:
    """
    Delete a compound and all associated data.

    Deletes from:
    - Database (compounds table)
    - Azure Blob Storage
    - Local cache

    Creates audit trail in deleted_compounds table.

    Args:
        entry_id: UUID of the compound entry to delete

    Returns:
        Deletion confirmation with details
    """
    # Find the compound
    compound = db.query(Compound).filter(Compound.entry_id == entry_id).first()

    if not compound:
        raise HTTPException(status_code=404, detail="Compound not found")

    compound_name = compound.compound_name
    safe_name = sanitize_compound_name(compound_name)

    # Delete from Azure - try both UUID-based and name-based paths
    azure_deleted_uuid = False
    azure_deleted_name = False

    if entry_id:
        azure_deleted_uuid = delete_result_from_azure_by_entry_id(entry_id)
        if azure_deleted_uuid:
            logger.info(f"Deleted UUID-based result from Azure: {entry_id}")

    azure_deleted_name = delete_result_from_azure(compound_name)
    if azure_deleted_name:
        logger.info(f"Deleted name-based result from Azure: {compound_name}")

    # Delete local ZIP files - try both UUID-based and name-based filenames
    local_deleted = []
    local_files_to_delete = []

    # UUID-based path: results/{prefix}/{entry_id}.zip
    if entry_id:
        prefix = entry_id[:2].lower()
        local_files_to_delete.append(settings.RESULTS_DIR / prefix / f"{entry_id}.zip")
        # Also check root for older files
        local_files_to_delete.append(settings.RESULTS_DIR / f"{entry_id}.zip")

    # Legacy name-based path
    local_files_to_delete.append(settings.RESULTS_DIR / f"{safe_name}.zip")

    for local_zip in local_files_to_delete:
        if local_zip.exists():
            try:
                local_zip.unlink()
                local_deleted.append(str(local_zip))
                logger.info(f"Deleted local result: {local_zip}")
            except Exception as e:
                logger.warning(f"Failed to delete local result {local_zip}: {e}")

    # Archive to deleted_compounds table before deletion
    deleted_record = DeletedCompound(
        original_id=compound.id,
        entry_id=compound.entry_id,
        compound_name=compound.compound_name,
        chembl_id=compound.chembl_id,
        smiles=compound.smiles,
        inchikey=compound.inchikey,
        is_duplicate=compound.is_duplicate,
        duplicate_of=compound.duplicate_of,
        storage_path=compound.storage_path,
        deleted_by_session=session_id,
        deleted_by_job_id=None,  # Direct deletion, not via job
        deletion_reason="user_request",
        original_processed_at=compound.processed_at,
    )
    db.add(deleted_record)

    # Delete from compounds table
    db.delete(compound)
    db.commit()

    # Audit log
    log_job_deleted(truncate_session_id(session_id), entry_id, compound_name)

    logger.info(f"Deleted compound: {compound_name} (entry_id={entry_id})")

    return {
        "message": "Compound deleted successfully",
        "entry_id": entry_id,
        "compound_name": compound_name,
        "azure_deleted": azure_deleted_uuid or azure_deleted_name,
        "local_deleted": local_deleted,
    }
