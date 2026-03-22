"""
Compounds API endpoints.

Provides access to processed compound data from the database.
Includes CRUD operations for compound management.
"""
import logging
import uuid

from fastapi import APIRouter, Body, Query, HTTPException

from backend.api.deps import DbDep, SessionDep
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
    db: DbDep,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    search: str | None = Query(None, description="Search by compound name"),
    include_duplicates: bool = Query(False, description="Include duplicate entries"),
):
    """List all processed compounds from the database."""
    offset = (page - 1) * page_size
    compound_rows, total = compound_repo.get_compounds_paginated(
        db,
        search=search,
        originals_only=not include_duplicates,
        offset=offset,
        limit=page_size,
    )

    items = []
    for compound, parent_name in compound_rows:
        item = CompoundListItem.model_validate(compound)
        item.is_duplicate = compound.parent_id is not None
        item.parent_name = parent_name
        items.append(item)

    return CompoundListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.get("/{entry_id}/versions", response_model=CompoundVersionsResponse)
async def get_compound_versions(
    entry_id: uuid.UUID,
    db: DbDep,
):
    """Get all structural siblings (versions) of a compound."""
    try:
        return _get_versions_service(db, entry_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{entry_id}", response_model=CompoundDetailResponse)
async def get_compound(
    entry_id: uuid.UUID,
    db: DbDep,
):
    """Get a specific compound by entry_id."""
    compound = compound_repo.get_by_entry_id(db, entry_id)
    if not compound:
        raise HTTPException(status_code=404, detail="Compound not found")

    return CompoundDetailResponse.model_validate(compound)


@router.delete("/{entry_id}", response_model=CompoundDeleteResponse)
async def delete_compound(
    entry_id: uuid.UUID,
    db: DbDep,
    session_id: SessionDep,
):
    """Delete a compound and all associated data."""
    try:
        return delete_compound_with_cleanup(db, entry_id, session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/batch-delete", response_model=BatchDeleteResponse)
async def batch_delete_compounds(
    db: DbDep,
    session_id: SessionDep,
    entry_ids: list[str] = Body(..., embed=True),
):
    """Delete multiple compounds in a single batch operation."""
    try:
        return batch_delete_with_cleanup(db, entry_ids, session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
