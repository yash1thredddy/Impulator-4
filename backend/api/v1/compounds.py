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
from backend.core.auth import validate_session_id
from backend.models.schemas import (
    CompoundListResponse,
    CompoundListItem,
    CompoundDetailResponse,
    CompoundDeleteResponse,
    BatchDeleteResponse,
    CompoundVersionsResponse,
)
from backend.repositories import compound_repo
from backend.services.compound_service import (
    get_compound_versions as _get_versions_service,
    delete_compound_with_cleanup,
    batch_delete_with_cleanup,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compounds", tags=["Compounds"])


@router.get("", response_model=CompoundListResponse)
async def list_compounds(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by compound name"),
    include_duplicates: bool = Query(False, description="Include duplicate entries"),
):
    """List all processed compounds from the database."""
    offset = (page - 1) * page_size
    compounds, total = compound_repo.get_compounds_paginated(
        db,
        search=search,
        is_duplicate_filter=True if include_duplicates else False,
        offset=offset,
        limit=page_size,
    )

    items = [
        CompoundListItem(
            entry_id=c.entry_id,
            compound_name=c.compound_name,
            smiles=c.smiles,
            inchikey=c.inchikey,
            threshold=c.similarity_threshold,
            similarity_threshold=c.similarity_threshold,
            activity_types=c.activity_types,
            similar_compounds=c.similar_compounds,
            total_activities=c.total_activities,
            is_duplicate=c.is_duplicate or False,
            duplicate_of=c.duplicate_of,
            created_at=c.processed_at,
            processed_at=c.processed_at,
            storage_path=c.storage_path,
            chembl_id=c.chembl_id,
            imp_candidates=c.imp_candidates,
            imp_score=c.imp_score,
            num_outliers=c.num_outliers,
            qed=c.qed,
        )
        for c in compounds
    ]

    return CompoundListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{entry_id}/versions", response_model=CompoundVersionsResponse)
async def get_compound_versions(
    entry_id: str,
    db: Session = Depends(get_db),
):
    """Get all structural siblings (versions) of a compound."""
    try:
        return _get_versions_service(db, entry_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{entry_id}", response_model=CompoundDetailResponse)
async def get_compound(
    entry_id: str,
    db: Session = Depends(get_db),
):
    """Get a specific compound by entry_id."""
    compound = compound_repo.get_by_entry_id(db, entry_id)
    if not compound:
        raise HTTPException(status_code=404, detail="Compound not found")

    return CompoundDetailResponse(
        entry_id=compound.entry_id,
        compound_name=compound.compound_name,
        smiles=compound.smiles,
        inchikey=compound.inchikey,
        inchikey_structure_key=compound.inchikey_structure_key,
        threshold=compound.similarity_threshold,
        activity_types=compound.activity_types,
        similar_compounds=compound.similar_compounds,
        total_activities=compound.total_activities,
        is_duplicate=compound.is_duplicate or False,
        duplicate_of=compound.duplicate_of,
        created_at=compound.processed_at,
    )


@router.delete("/{entry_id}", response_model=CompoundDeleteResponse)
async def delete_compound(
    entry_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Delete a compound and all associated data."""
    try:
        return delete_compound_with_cleanup(db, entry_id, session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/batch-delete", response_model=BatchDeleteResponse)
async def batch_delete_compounds(
    entry_ids: List[str] = Body(..., embed=True),
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Delete multiple compounds in a single batch operation."""
    try:
        return batch_delete_with_cleanup(db, entry_ids, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
