"""Collections API endpoints (Phase 23).

A *collection* is ONE ``JobType.COLLECTION`` job plus a row in the ``collections``
table whose members live in ``members_config`` JSONB (D-02). This router is the
thin HTTP boundary over ``collection_service`` / ``collection_repo``:

- Handlers parse the request, call the service/repo, translate domain
  ``ValueError`` -> ``HTTPException`` (400) and not-found -> 404, and return a
  typed response. No raw SQL and no business logic here -- all data access goes
  through the service / repository (Router -> Service -> Repository, ARCH-04).
- Collections are a GLOBAL resource (D-05): the list endpoint takes ``db: DbDep``
  only -- no session dependency -- so every collection is visible across
  sessions, like the entries page.
- Path-traversal on ``name`` / ``author_name`` (D-03) is rejected at the schema
  layer (``CollectionJobCreate`` validators, plan 03); a bad name never reaches
  the service.
- DELETE routes through the collection service's soft-delete path ONLY, never
  the single-compound job-cleanup path.
"""
import logging
import os
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from backend.api.deps import DbDep, SessionDep
from backend.config import settings
from backend.core.azure_sync import is_azure_configured
from backend.core.storage_paths import get_collection_storage_path
from backend.models.schemas import (
    CollectionJobCreate,
    CollectionResponse,
    CollectionDetailResponse,
    CollectionListResponse,
    CollectionSummary,
)
from backend.repositories import collection_repo
from backend.repositories.job_repository import job_repo
from backend.services.collection_service import (
    create_collection,
    delete_collection,
)
from backend.core import scheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["Collections"])


def _apply_job_status(response, job) -> None:
    """Fold the linked job's status/progress/message onto a collection response.

    Collections are GLOBAL (D-05) but their job is session-scoped, so a session
    viewing a collection it did not create cannot read ``/jobs/{id}`` (403). We
    surface the job's status here — on the global collection payload — so the
    frontend never makes the session-owned job call. ``message`` carries the
    failure reason for a failed job, else the live progress step string.
    """
    status = str(job.status.value) if job.status is not None else None
    failed = (status or "").lower() in ("failed", "error")
    response.status = status
    response.progress = job.progress
    response.message = job.error_message if failed else job.current_step


@router.post("", status_code=201, response_model=CollectionResponse)
async def create_collection_endpoint(
    request: CollectionJobCreate,
    db: DbDep,
    session_id: SessionDep,
):
    """Create a collection (+ its COLLECTION job) and submit it for processing."""
    # Stamp the caller's session onto the collection's job (mirrors jobs.create_job)
    # so it appears in the session-scoped sidebar / active-jobs view — otherwise the
    # job is created with session=None and never shows a progress bar.
    request.session_id = session_id
    try:
        collection_id, _job_id = create_collection(db, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Mirror jobs.create_job: kick the scheduler so the COLLECTION job starts.
    scheduler.trigger()

    collection = collection_repo.get_by_id(db, uuid.UUID(collection_id))
    if collection is None:
        raise HTTPException(status_code=500, detail="Collection was not persisted")
    return CollectionResponse.model_validate(collection)


@router.get("", response_model=CollectionListResponse)
async def list_collections(db: DbDep):
    """List every collection GLOBALLY (D-05) -- no per-session filtering."""
    collections = collection_repo.list_all(db)
    items = []
    for c in collections:
        summary = CollectionSummary.model_validate(c)
        job = job_repo.get_by_job_id(db, c.job_id)
        if job is not None:
            _apply_job_status(summary, job)
        items.append(summary)
    return CollectionListResponse(
        items=items,
        total=len(items),
        page=1,
        page_size=len(items) or 1,
        pages=1,
    )


@router.get("/{collection_id}", response_model=CollectionDetailResponse)
async def get_collection(collection_id: uuid.UUID, db: DbDep):
    """Get a single collection's detail, including its member input set (D-02)."""
    collection = collection_repo.get_by_id(db, collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    response = CollectionDetailResponse.model_validate(collection)
    # D-PF-6: per-member failure rows + lower-tier cascade hints live on the
    # LINKED JOB's result_summary (so they survive the D-11 auto-delete of the
    # collection row), not on the collection itself -- surface them here.
    job = job_repo.get_by_job_id(db, collection.job_id)
    if job is not None:
        response.failed_members = (job.result_summary or {}).get("failed_members")
        _apply_job_status(response, job)
    return response


@router.delete("/{collection_id}")
async def delete_collection_endpoint(collection_id: uuid.UUID, db: DbDep):
    """Soft-delete a collection and remove its ZIP (D-11) via collection_service."""
    try:
        deleted = delete_collection(db, collection_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"deleted": True, "collection_id": str(collection_id)}


@router.get("/{collection_id}/download")
async def download_collection(collection_id: uuid.UUID, db: DbDep):
    """Stream the collection's nested ZIP (D-12), never buffering the whole file.

    Returns 404 if the collection is missing and 409 if the ZIP is not ready yet
    (the job has not finished assembling/uploading it).
    """
    collection = collection_repo.get_by_id(db, collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    storage_path = collection.storage_path or get_collection_storage_path(
        str(collection_id)
    )
    if not collection.storage_path:
        raise HTTPException(
            status_code=409, detail="Collection ZIP is not ready yet"
        )

    filename = f"{collection.name}.zip"

    # Local results tree first (FileResponse streams from disk, no buffering).
    local_path = os.path.join(str(settings.RESULTS_DIR), storage_path)
    if os.path.exists(local_path):
        return FileResponse(
            local_path, media_type="application/zip", filename=filename
        )

    # Otherwise stream straight from Azure Blob (chunked, no full-file buffer).
    if is_azure_configured():
        from backend.core.azure_sync import _get_blob_client

        blob = _get_blob_client(storage_path)
        if blob is not None and blob.exists():
            downloader = blob.download_blob()
            return StreamingResponse(
                downloader.chunks(),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                },
            )

    raise HTTPException(status_code=404, detail="Collection ZIP not found")
