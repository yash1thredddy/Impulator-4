"""
Compounds API endpoints.

Provides access to processed compound data from the database.
Includes CRUD operations for compound management.
"""
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Body, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.core.database import get_db
from backend.core.auth import validate_session_id, truncate_session_id
from backend.core.azure_sync import delete_result_from_azure_by_entry_id
from backend.core.audit import log_job_deleted
from backend.models.database import Compound, DeletedCompound
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compounds", tags=["Compounds"])


def _handle_children_before_delete(db: Session, compound: Compound) -> None:
    """Handle child compounds before deleting a main compound.

    If the compound being deleted is a main compound (not a duplicate),
    promotes the oldest child to main and re-points remaining children.
    This prevents orphaned duplicate_of references.

    Args:
        db: Database session (caller must commit)
        compound: The compound about to be deleted
    """
    if compound.is_duplicate:
        return  # Duplicates don't have children

    children = db.query(Compound).filter(
        Compound.duplicate_of == compound.entry_id
    ).all()

    if not children:
        return

    # Sort by processed_at ascending (oldest first)
    children_sorted = sorted(
        children,
        key=lambda c: c.processed_at or datetime.min,
    )

    # Promote oldest child to main
    promoted = children_sorted[0]
    promoted.is_duplicate = False
    promoted.duplicate_of = None
    logger.info(
        f"Promoted '{promoted.compound_name}' ({promoted.entry_id}) to main "
        f"(was duplicate of '{compound.compound_name}')"
    )

    # Re-point remaining children to the promoted compound
    for child in children_sorted[1:]:
        child.duplicate_of = promoted.entry_id
        logger.info(
            f"Re-pointed '{child.compound_name}' ({child.entry_id}) "
            f"duplicate_of -> {promoted.entry_id}"
        )


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

    # Apply search filter with escaped wildcards to prevent SQL injection
    if search:
        # Escape SQL ILIKE special characters to prevent pattern injection attacks
        # IMPORTANT: Escape backslash FIRST, then wildcards (order matters!)
        # Without this, input like '\' would escape the trailing '%' wildcard
        search_escaped = (
            search
            .replace('\\', '\\\\')  # Escape backslash first (escape char itself)
            .replace('%', '\\%')    # Then escape % wildcard
            .replace('_', '\\_')    # Then escape _ wildcard
        )
        query = query.filter(Compound.compound_name.ilike(f"%{search_escaped}%", escape='\\'))

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
            "imp_score": compound.imp_score,
            "similarity_threshold": compound.similarity_threshold,
            "qed": compound.qed,
            "num_outliers": compound.num_outliers,
            "author_name": compound.author_name,
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

    # Delete from Azure (UUID-based storage only)
    azure_deleted = delete_result_from_azure_by_entry_id(entry_id)
    if azure_deleted:
        logger.info(f"Deleted result from Azure: {entry_id}")

    # Delete local ZIP file (UUID-based path only)
    local_deleted = []
    prefix = entry_id[:2].lower()
    local_zip = settings.RESULTS_DIR / prefix / f"{entry_id}.zip"
    if local_zip.exists():
        try:
            local_zip.unlink()
            local_deleted.append(str(local_zip))
            logger.info(f"Deleted local result: {local_zip}")
        except Exception as e:
            logger.warning(f"Failed to delete local result {local_zip}: {e}")

    # Promote children before deleting a main compound
    _handle_children_before_delete(db, compound)

    # Archive to deleted_compounds table before deletion
    deleted_record = DeletedCompound(
        original_id=compound.id,
        entry_id=compound.entry_id,
        compound_name=compound.compound_name,
        chembl_id=compound.chembl_id,
        smiles=compound.smiles,
        inchikey=compound.inchikey,
        author_name=compound.author_name,
        is_duplicate=compound.is_duplicate,
        duplicate_of=compound.duplicate_of,
        activity_types=compound.activity_types,
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
        "azure_deleted": azure_deleted,
        "local_deleted": local_deleted,
    }


@router.post("/batch-delete")
async def batch_delete_compounds(
    entry_ids: List[str] = Body(..., embed=True),
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
) -> dict:
    """Delete multiple compounds in a single batch operation.

    Deletes from database, Azure Blob Storage, and local cache.
    Creates audit trail in deleted_compounds table for each compound.

    Args:
        entry_ids: List of compound entry UUIDs to delete (max 50)

    Returns:
        Summary of deleted and not-found compounds
    """
    if not entry_ids:
        raise HTTPException(status_code=400, detail="entry_ids list cannot be empty")

    if len(entry_ids) > 50:
        raise HTTPException(status_code=400, detail="Cannot delete more than 50 compounds at once")

    # Validate all entry_ids are non-empty strings
    for eid in entry_ids:
        if not isinstance(eid, str) or not eid.strip():
            raise HTTPException(status_code=400, detail="All entry_ids must be non-empty strings")

    deleted = []
    not_found = []

    # Deduplicate to prevent same ID appearing in both deleted and not_found
    seen = set()
    unique_entry_ids = []
    for eid in entry_ids:
        if eid not in seen:
            seen.add(eid)
            unique_entry_ids.append(eid)

    for eid in unique_entry_ids:
        compound = db.query(Compound).filter(Compound.entry_id == eid).first()

        if not compound:
            not_found.append(eid)
            continue

        compound_name = compound.compound_name

        # Promote children before deleting a main compound
        _handle_children_before_delete(db, compound)

        # Archive to deleted_compounds
        deleted_record = DeletedCompound(
            original_id=compound.id,
            entry_id=compound.entry_id,
            compound_name=compound.compound_name,
            chembl_id=compound.chembl_id,
            smiles=compound.smiles,
            inchikey=compound.inchikey,
            author_name=compound.author_name,
            is_duplicate=compound.is_duplicate,
            duplicate_of=compound.duplicate_of,
            activity_types=compound.activity_types,
            storage_path=compound.storage_path,
            deleted_by_session=session_id,
            deleted_by_job_id=None,
            deletion_reason="batch_delete",
            original_processed_at=compound.processed_at,
        )
        db.add(deleted_record)
        db.delete(compound)

        # Audit log
        log_job_deleted(truncate_session_id(session_id), eid, compound_name)
        deleted.append({"entry_id": eid, "compound_name": compound_name})

        logger.info(f"Batch delete - archived: {compound_name} ({eid})")

    # Commit DB first — only delete storage after successful commit
    db.commit()

    # Now safe to delete storage (DB is committed, no data loss risk)
    for item in deleted:
        eid = item["entry_id"]
        try:
            azure_ok = delete_result_from_azure_by_entry_id(eid)
            if azure_ok:
                logger.info(f"Batch delete - Azure deleted: {eid}")
        except Exception as e:
            logger.warning(f"Batch delete - Azure cleanup failed for {eid}: {e}")

        prefix = eid[:2].lower()
        local_zip = settings.RESULTS_DIR / prefix / f"{eid}.zip"
        if local_zip.exists():
            try:
                local_zip.unlink()
                logger.info(f"Batch delete - local deleted: {local_zip}")
            except Exception as e:
                logger.warning(f"Batch delete - failed to delete local {local_zip}: {e}")

    return {
        "message": f"Deleted {len(deleted)} compound(s)",
        "deleted": deleted,
        "not_found": not_found,
        "total_deleted": len(deleted),
    }
