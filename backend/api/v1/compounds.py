"""
Compounds API endpoints.

Provides access to processed compound data from the database.
Includes CRUD operations for compound management.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Body, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.auth import validate_session_id, truncate_session_id
from backend.core.azure_sync import delete_result_from_azure_by_entry_id
from backend.core.audit import log_job_deleted
from backend.models.database import Compound
from backend.repositories import compound_repo, _db_write_lock
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
    offset = (page - 1) * per_page
    compounds, total = compound_repo.get_compounds_paginated(
        db,
        search=search,
        is_duplicate_filter=True if include_duplicates else False,
        offset=offset,
        limit=per_page,
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


@router.get("/{entry_id}/versions")
async def get_compound_versions(
    entry_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Get all structural siblings (versions) of a compound.

    Finds compounds sharing the same InChIKey structure key (first two blocks),
    which represents re-analyses with different configs.

    Args:
        entry_id: UUID of the compound entry

    Returns:
        List of version items with is_original and is_current flags
    """
    siblings = compound_repo.get_versions(db, entry_id)

    if len(siblings) <= 1:
        # get_versions returns [] if compound not found or no siblings;
        # check if compound exists for proper 404
        if not siblings:
            compound = compound_repo.get_by_entry_id(db, entry_id)
            if not compound:
                raise HTTPException(status_code=404, detail="Compound not found")
        return {"versions": [], "current_entry_id": entry_id}

    # Identify the original: oldest non-duplicate, fallback to oldest overall
    original_entry_id = None
    for s in siblings:
        if not s.is_duplicate:
            original_entry_id = s.entry_id
            break
    if original_entry_id is None:
        original_entry_id = siblings[0].entry_id

    # Batch-resolve parent names for duplicates
    parent_entry_ids = {s.duplicate_of for s in siblings if s.duplicate_of}
    parent_names = {}
    if parent_entry_ids:
        for pid in parent_entry_ids:
            parent = compound_repo.get_by_entry_id(db, pid)
            if parent:
                parent_names[pid] = parent.compound_name

    versions = []
    for s in siblings:
        versions.append({
            "entry_id": s.entry_id,
            "compound_name": s.compound_name,
            "similarity_threshold": s.similarity_threshold,
            "activity_types": s.activity_types,
            "imp_score": s.imp_score,
            "qed": s.qed,
            "similar_compounds": s.similar_compounds or 0,
            "total_activities": s.total_activities,
            "is_duplicate": s.is_duplicate or False,
            "duplicate_of": s.duplicate_of,
            "duplicate_of_name": parent_names.get(s.duplicate_of),
            "author_name": s.author_name,
            "processed_at": s.processed_at.isoformat() if s.processed_at else None,
            "storage_path": s.storage_path,
            "is_original": s.entry_id == original_entry_id,
            "is_current": s.entry_id == entry_id,
        })

    return {"versions": versions, "current_entry_id": entry_id}


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
    compound = compound_repo.get_by_entry_id(db, entry_id)

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
    # Fast-fail: check compound exists before doing any I/O
    compound = compound_repo.get_by_entry_id(db, entry_id)
    if not compound:
        raise HTTPException(status_code=404, detail="Compound not found")

    # Delete from Azure FIRST (UUID-based storage only) -- outside lock
    azure_deleted = delete_result_from_azure_by_entry_id(entry_id)
    if azure_deleted:
        logger.info(f"Deleted result from Azure: {entry_id}")

    # Delete local ZIP file -- outside lock
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

    # DB mutations under write lock
    with _db_write_lock:
        # Re-query compound inside lock (it may have been deleted by another thread)
        compound = compound_repo.get_by_entry_id(db, entry_id)
        if not compound:
            raise HTTPException(status_code=404, detail="Compound not found")

        compound_name = compound.compound_name

        # Promote children before deleting a main compound
        compound_repo.handle_children_before_delete(db, entry_id)

        # Archive to deleted_compounds table before deletion
        compound_repo.archive_compound(
            db,
            compound,
            session_id=session_id,
            deletion_reason="user_request",
        )

        # Delete from compounds table
        db.delete(compound)
        db.commit()

    # Audit log (outside lock -- no DB mutation)
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

    # DB mutations under write lock
    with _db_write_lock:
        for eid in unique_entry_ids:
            compound = compound_repo.get_by_entry_id(db, eid)

            if not compound:
                not_found.append(eid)
                continue

            compound_name = compound.compound_name

            # Promote children before deleting a main compound
            compound_repo.handle_children_before_delete(db, eid)

            # Archive to deleted_compounds
            compound_repo.archive_compound(
                db,
                compound,
                session_id=session_id,
                deletion_reason="batch_delete",
            )
            db.delete(compound)

            # Audit log
            log_job_deleted(truncate_session_id(session_id), eid, compound_name)
            deleted.append({"entry_id": eid, "compound_name": compound_name})

            logger.info(f"Batch delete - archived: {compound_name} ({eid})")

        # Commit DB first -- only delete storage after successful commit
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
