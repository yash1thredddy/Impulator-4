"""
Job API endpoints.
Handles job submission, status tracking, and cancellation.

Session Isolation:
- Each browser session has a unique session_id (passed in X-Session-ID header)
- Users only see their own jobs in the sidebar
- Jobs can be grouped into batches for bulk operations

Rate Limiting:
- Per-session rate limiting to prevent abuse
- Configurable limits for single jobs and batch submissions

All orchestration logic lives in job_service.py (ARCH-04).
Route handlers are thin (10-30 lines): parse input, call service, return typed response.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.executor import job_executor
from backend.core.scheduler import job_scheduler
from backend.core.auth import validate_session_id, truncate_session_id
from backend.core.audit import log_job_cancelled, log_job_deleted
from backend.models.database import JobStatus
from backend.models.schemas import (
    JobCreate,
    BatchJobCreate,
    JobResponse,
    JobListResponse,
    ActiveJobResponse,
    BatchSummary,
    ErrorResponse,
    CheckDuplicatesRequest,
    CheckDuplicatesResponse,
    DuplicateFoundResponse,
    ResolveDuplicateRequest,
    SkipResponse,
    BatchResponse,
    DeleteResponse,
    CancelResponse,
    JobDetailResponse,
    CheckAvailabilityRequest,
    CheckAvailabilityBatchRequest,
    CheckAvailabilityResponse,
    CheckAvailabilityBatchResponse,
    InputParams,
)
from backend.services.job_service import (
    job_service,
    _job_to_response,
    _check_single_availability,
    MAX_BATCH_SIZE,
)
from backend.core.rate_limiter import (
    rate_limit,
    get_rate_limiter,
    RATE_LIMIT_MAX_JOBS,
    RATE_LIMIT_MAX_BATCH,
)
from backend.repositories import job_repo

# Backward-compatible singleton (used by health.py and tests)
rate_limiter = get_rate_limiter()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def _verify_job_ownership(db: Session, job_id: str, session_id: str):
    """Verify the session owns this job.

    Returns:
        Job: The job if ownership is verified

    Raises:
        HTTPException: 404 if job not found, 403 if unauthorized
    """
    job = job_repo.get_by_job_id(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.session_id and job.session_id != session_id:
        logger.warning(
            f"Unauthorized access attempt: session {truncate_session_id(session_id)} "
            f"tried to access job {job_id}"
        )
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to access this job"
        )

    return job


# ============================================================================
# Route handlers (thin: parse input, call service, return typed response)
# ============================================================================


@router.post(
    "/check-availability",
    response_model=CheckAvailabilityResponse,
    summary="Check ChEMBL data availability before submission",
)
def check_availability(
    request: CheckAvailabilityRequest,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Check whether ChEMBL has similarity data for a compound."""
    try:
        result = _check_single_availability(
            smiles=request.smiles,
            compound_name="query",
            similarity_threshold=request.similarity_threshold,
            activity_types=request.activity_types,
            db=db,
        )
        return CheckAvailabilityResponse(result=result)
    except Exception:
        logger.exception("Availability check failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/check-availability/batch",
    response_model=CheckAvailabilityBatchResponse,
    summary="Batch check ChEMBL data availability",
)
def check_availability_batch(
    request: CheckAvailabilityBatchRequest,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Batch availability check for multiple compounds."""
    try:
        return job_service.check_availability_batch_service(db, request)
    except Exception:
        logger.exception("Batch availability check failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "",
    status_code=201,
    response_model=JobResponse,
    responses={
        200: {"model": DuplicateFoundResponse, "description": "Duplicate compound detected"},
        429: {"model": ErrorResponse},
    },
    summary="Submit a new job",
)
async def create_job(
    request: JobCreate,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    _: None = rate_limit(RATE_LIMIT_MAX_JOBS),
):
    """Submit a new compound processing job.

    Returns JobResponse (201) or DuplicateFoundResponse (200) if duplicate detected.
    """
    try:
        result = job_service.submit_job(db, request, session_id, idempotency_key)
    except Exception:
        logger.exception(f"Failed to create job for {request.compound_name}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create job. Please try again."
        )

    if isinstance(result, DuplicateFoundResponse):
        return JSONResponse(content=result.model_dump(mode="json"), status_code=200)

    # Trigger scheduler to start processing
    job_scheduler.trigger()
    logger.info(f"Job queued for {request.compound_name} (session={truncate_session_id(session_id)})")
    return result


@router.post(
    "/resolve-duplicate",
    status_code=201,
    response_model=JobResponse,
    responses={
        200: {"model": SkipResponse, "description": "Compound skipped"},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    summary="Resolve a duplicate compound",
)
async def resolve_duplicate(
    request: ResolveDuplicateRequest,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Resolve a duplicate compound situation based on user's choice."""
    try:
        result = job_service.resolve_duplicate_action(db, request, session_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if isinstance(result, SkipResponse):
        return JSONResponse(content=result.model_dump(mode="json"), status_code=200)
    return result


@router.post(
    "/check-duplicates",
    response_model=CheckDuplicatesResponse,
    summary="Check for duplicate compounds",
)
async def check_duplicates(
    request: CheckDuplicatesRequest,
    db: Session = Depends(get_db),
):
    """Check which compounds already exist or are being processed."""
    try:
        return job_service.check_duplicates_batch(db, request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post(
    "/batch",
    status_code=201,
    response_model=BatchResponse,
    responses={429: {"model": ErrorResponse}},
    summary="Submit a batch of jobs",
)
async def create_batch_job(
    request: BatchJobCreate,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
    _: None = rate_limit(RATE_LIMIT_MAX_BATCH, key_suffix="_batch"),
):
    """Submit multiple compound processing jobs."""
    if len(request.compounds) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large. Maximum {MAX_BATCH_SIZE} compounds allowed."
        )
    return job_service.submit_batch(db, request, session_id)


@router.get(
    "",
    response_model=JobListResponse,
    summary="List jobs for current session",
)
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """List jobs for the current session with optional status filter and pagination."""
    statuses = [status] if status else None
    result = job_service.list_jobs(
        db, statuses=statuses, page=page, page_size=page_size, session_id=session_id
    )
    return JobListResponse(
        items=[_job_to_response(j) for j in result["items"]],
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        pages=result["pages"],
    )


@router.get(
    "/active",
    response_model=List[ActiveJobResponse],
    summary="Get active jobs for sidebar",
)
async def get_active_jobs(
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Get active (pending/processing) jobs for the current session."""
    return job_service.get_active_jobs(db, session_id=session_id)


@router.get(
    "/batch/{batch_id}",
    response_model=BatchSummary,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Get batch summary",
)
async def get_batch_summary(
    batch_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Get summary statistics for a batch of jobs."""
    batch_jobs = job_repo.get_batch_jobs(db, batch_id)
    first_job = batch_jobs[0] if batch_jobs else None

    if not first_job:
        raise HTTPException(status_code=404, detail="Batch not found")

    if first_job.session_id and first_job.session_id != session_id:
        logger.warning(
            f"Unauthorized batch access attempt: session {truncate_session_id(session_id)} "
            f"tried to access batch {batch_id}"
        )
        raise HTTPException(status_code=403, detail="You don't have permission to access this batch")

    summary = job_service.get_batch_summary(db, batch_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Batch not found")
    return summary


@router.post(
    "/batch/{batch_id}/cancel",
    response_model=CancelResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Cancel all jobs in a batch",
)
async def cancel_batch(
    batch_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Cancel all pending/processing jobs in a batch."""
    batch_jobs = job_repo.get_batch_jobs(db, batch_id)

    if not batch_jobs:
        raise HTTPException(status_code=404, detail="Batch not found")

    first_job = batch_jobs[0]
    if first_job.session_id and first_job.session_id != session_id:
        logger.warning(
            f"Unauthorized batch cancel attempt: session {truncate_session_id(session_id)} "
            f"tried to cancel batch {batch_id}"
        )
        raise HTTPException(status_code=403, detail="You don't have permission to cancel this batch")

    cancelled_count = job_service.cancel_batch(db, batch_id)
    log_job_cancelled(truncate_session_id(session_id), f"batch:{batch_id}", f"{cancelled_count} jobs")

    return CancelResponse(
        batch_id=batch_id,
        cancelled_count=cancelled_count,
        message=f"Cancelled {cancelled_count} jobs in batch",
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Get job status",
)
async def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Get the current status of a job."""
    job = _verify_job_ownership(db, job_id, session_id)
    return _job_to_response(job)


@router.get(
    "/{job_id}/detail",
    response_model=JobDetailResponse,
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Get detailed job info",
)
async def get_job_detail(
    job_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Get detailed job information including parsed input parameters."""
    _verify_job_ownership(db, job_id, session_id)
    job = job_service.get_job_with_params(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post(
    "/{job_id}/cancel",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Cancel a job",
)
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Cancel a pending or processing job."""
    job = _verify_job_ownership(db, job_id, session_id)

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=409,
            detail=f"Job cannot be cancelled (status: {job.status.value})",
        )

    # Extract compound name for audit log
    compound_name = None
    if job.input_params:
        try:
            params = InputParams.model_validate_json(job.input_params)
            compound_name = params.compound_name
        except Exception:
            pass

    job_executor.cancel(job_id)
    job = job_service.cancel_job(db, job_id)
    log_job_cancelled(truncate_session_id(session_id), job_id, compound_name)

    return _job_to_response(job)


@router.delete(
    "/{job_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Delete a job record",
)
async def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """Delete a job record and associated result files."""
    job = _verify_job_ownership(db, job_id, session_id)

    if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete active jobs. Cancel first.",
        )

    try:
        result = job_service.delete_job_with_cleanup(db, job_id, session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    log_job_deleted(truncate_session_id(session_id), job_id, result.compound_name)
    return result
