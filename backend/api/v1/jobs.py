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
"""
import logging
import time
import threading
from collections import defaultdict
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Header
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from backend.core.database import get_db
from backend.core.executor import job_executor
from backend.core.scheduler import job_scheduler
from backend.core.auth import validate_session_id, truncate_session_id
from backend.models.database import JobStatus, JobType
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
    ExistingCompoundInfo,
    ResolveDuplicateRequest,
    DuplicateAction,
    SkipResponse,
    BatchResponse,
    DeleteResponse,
    CancelResponse,
    DuplicateMatch,
)
from backend.services.job_service import job_service, generate_inchikey, generate_canonical_smiles
from backend.models.database import Compound, DeletedCompound
from backend.core.azure_sync import delete_result_from_azure_by_entry_id
from backend.core.audit import (
    log_rate_limit_exceeded,
    log_job_cancelled,
    log_job_deleted,
)
from backend.config import settings

logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_WINDOW_SECONDS = 60  # 1 minute window
RATE_LIMIT_MAX_JOBS = 10  # Max 10 single jobs per minute per session
RATE_LIMIT_MAX_BATCH = 3  # Max 3 batch submissions per minute per session

# Batch size limits
MAX_BATCH_SIZE = 1000  # Maximum compounds per batch submission


class RateLimiter:
    """Simple in-memory rate limiter per session.

    Thread-safe implementation using defaultdict and locks.
    Automatically cleans up old entries to prevent memory leaks.
    Limited to MAX_SESSIONS to prevent unbounded growth.
    """
    MAX_SESSIONS = 10000  # Prevent unbounded memory growth

    def __init__(self, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self._requests: dict = defaultdict(list)  # session_id -> [timestamps]
        self._lock = threading.Lock()
        self._window_seconds = window_seconds

    def _cleanup_session(self, session_id: str, now: float) -> None:
        """Clean up old timestamps for a specific session."""
        cutoff = now - self._window_seconds
        if session_id in self._requests:
            self._requests[session_id] = [
                t for t in self._requests[session_id] if t > cutoff
            ]
            if not self._requests[session_id]:
                del self._requests[session_id]

    def _evict_oldest_session(self) -> None:
        """Evict the session with oldest activity when at capacity."""
        if not self._requests:
            return
        oldest = min(
            self._requests.keys(),
            key=lambda k: min(self._requests[k]) if self._requests[k] else float('inf')
        )
        del self._requests[oldest]

    @property
    def active_session_count(self) -> int:
        """Return the number of active sessions being tracked."""
        return len(self._requests)

    def check_rate_limit(self, session_id: str, limit: int) -> tuple[bool, int]:
        """Check if request is within rate limit.

        Args:
            session_id: Session identifier (or IP if no session)
            limit: Maximum requests allowed in window

        Returns:
            Tuple of (allowed: bool, remaining: int)
        """
        if not session_id:
            session_id = "anonymous"

        with self._lock:
            now = time.time()

            # Clean this session's old entries
            self._cleanup_session(session_id, now)

            # Check session limit to prevent memory leak
            if len(self._requests) >= self.MAX_SESSIONS and session_id not in self._requests:
                # Evict oldest session to make room
                self._evict_oldest_session()

            timestamps = self._requests.get(session_id, [])

            if len(timestamps) >= limit:
                return False, 0

            # Add new timestamp
            if session_id not in self._requests:
                self._requests[session_id] = []
            self._requests[session_id].append(now)

            return True, limit - len(timestamps) - 1


# Global rate limiter instance
rate_limiter = RateLimiter()

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def _job_to_response(job) -> JobResponse:
    """Convert Job model to JobResponse, extracting compound info from input_params."""
    import json

    data = {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "result_path": job.result_path,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "session_id": job.session_id,
        "batch_id": job.batch_id,
    }

    # Extract compound_name and smiles from input_params
    if job.input_params:
        try:
            params = json.loads(job.input_params)
            data["compound_name"] = params.get("compound_name", "Unknown")
            data["smiles"] = params.get("smiles", "")
        except (json.JSONDecodeError, TypeError):
            data["compound_name"] = "Unknown"
            data["smiles"] = ""

    return JobResponse(**data)


def _verify_job_ownership(db: Session, job_id: str, session_id: str) -> "Job":
    """Verify the session owns this job.

    Args:
        db: Database session
        job_id: Job ID to verify
        session_id: Session ID making the request

    Returns:
        Job: The job if ownership is verified

    Raises:
        HTTPException: 404 if job not found, 403 if unauthorized
    """
    from backend.models.database import Job

    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Allow if session matches OR if job has no session (legacy)
    if job.session_id and job.session_id != session_id:
        # Don't reveal existence of job to unauthorized users
        logger.warning(
            f"Unauthorized access attempt: session {truncate_session_id(session_id)} "
            f"tried to access job {job_id}"
        )
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to access this job"
        )

    return job


@router.post(
    "",
    status_code=201,
    responses={429: {"model": ErrorResponse}},
    summary="Submit a new job",
)
async def create_job(
    request: JobCreate,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """
    Submit a new compound processing job.

    The job runs in the background. Use GET /jobs/{id} to check status.
    Jobs are queued in SQLite and picked up by the scheduler.

    **Duplicate Detection:**
    If a compound with the same InChIKey already exists, returns a
    `duplicate_found` response instead of creating the job. Use
    `/jobs/resolve-duplicate` to handle the duplicate.

    Headers:
        X-Session-ID: Session ID for user isolation (validated UUID)

    Rate Limit:
        Max 10 jobs per minute per session

    Returns:
        - JobResponse if job created successfully
        - DuplicateFoundResponse if duplicate detected
    """
    # Use session_id from validated header, fall back to request body if anonymous
    if session_id.startswith("anon-") and request.session_id:
        session_id = request.session_id

    # Check rate limit
    allowed, remaining = rate_limiter.check_rate_limit(session_id, RATE_LIMIT_MAX_JOBS)
    if not allowed:
        logger.warning(f"Rate limit exceeded for session {truncate_session_id(session_id)}")
        log_rate_limit_exceeded(truncate_session_id(session_id), "single_job", RATE_LIMIT_MAX_JOBS)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX_JOBS} jobs per minute.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )

    # Generate InChIKey for duplicate detection
    inchikey = generate_inchikey(request.smiles) if request.smiles else None

    # Helper function to build duplicate response
    def _build_duplicate_response(existing_compound: Compound) -> DuplicateFoundResponse:
        name_matches = existing_compound.compound_name.lower().strip() == request.compound_name.lower().strip()
        duplicate_type = "exact" if name_matches else "structure_only"

        logger.info(f"Duplicate found: {request.compound_name} matches {existing_compound.compound_name} (InChIKey: {inchikey[:14]}...)")

        return DuplicateFoundResponse(
            status="duplicate_found",
            duplicate_type=duplicate_type,
            existing_compound=ExistingCompoundInfo(
                entry_id=existing_compound.entry_id,
                compound_name=existing_compound.compound_name,
                inchikey=existing_compound.inchikey,
                processed_at=existing_compound.processed_at.isoformat() if existing_compound.processed_at else None,
            ),
            submitted={
                "compound_name": request.compound_name,
                "inchikey": inchikey,
                "smiles": request.smiles,
            }
        )

    # Atomic check-and-create with retry for race condition handling
    # SQLite doesn't support FOR UPDATE, so we use retry logic instead
    max_retries = 3
    for attempt in range(max_retries):
        # Check for duplicate by InChIKey
        if inchikey:
            existing = db.query(Compound).filter(Compound.inchikey == inchikey).first()
            if existing:
                return _build_duplicate_response(existing)

        try:
            # No duplicate found - create job
            job = job_service.create_job(
                db,
                JobType.SINGLE,
                request.model_dump(exclude={"session_id"}),
                session_id=session_id,
            )

            # Trigger scheduler to start processing (if not already running)
            job_scheduler.trigger()

            logger.info(f"Job {job.id} queued for {request.compound_name} (session={truncate_session_id(session_id)}, remaining={remaining})")
            return _job_to_response(job)

        except IntegrityError:
            # Race condition: another request created the compound between check and insert
            db.rollback()
            if attempt < max_retries - 1:
                logger.info(f"Retry {attempt + 1}/{max_retries} for {request.compound_name} due to race condition")
                continue

            # On final attempt, check if duplicate was created
            if inchikey:
                existing = db.query(Compound).filter(Compound.inchikey == inchikey).first()
                if existing:
                    return _build_duplicate_response(existing)

            # Re-raise if we still can't handle it
            logger.error(f"Failed to create job for {request.compound_name} after {max_retries} retries")
            raise HTTPException(
                status_code=500,
                detail="Failed to create job due to concurrent access. Please try again."
            )


@router.post(
    "/resolve-duplicate",
    status_code=201,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Resolve a duplicate compound",
)
async def resolve_duplicate(
    request: ResolveDuplicateRequest,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """
    Resolve a duplicate compound situation based on user's choice.

    Called after `/jobs` returns a `duplicate_found` response.
    User chooses one of three actions:

    **Actions:**
    - `replace`: Delete existing compound and its results, create new job
    - `duplicate`: Create job with duplicate tag (keeps both)
    - `skip`: Don't process, return skipped status

    Returns:
        - JobResponse if job created (replace/duplicate actions)
        - Skipped message if skip action
    """
    # Use session_id from validated header, fall back to request body if anonymous
    if session_id.startswith("anon-") and request.session_id:
        session_id = request.session_id

    # Handle SKIP action
    if request.action == DuplicateAction.SKIP:
        logger.info(f"User skipped duplicate: {request.compound_name}")
        return SkipResponse(
            status="skipped",
            message=f"Compound '{request.compound_name}' processing skipped by user",
            compound_name=request.compound_name,
        )

    # Handle REPLACE action - delete existing and create new
    if request.action == DuplicateAction.REPLACE:
        if request.existing_entry_id:
            existing = db.query(Compound).filter(Compound.entry_id == request.existing_entry_id).first()
            if existing:
                old_name = existing.compound_name
                old_entry_id = existing.entry_id

                # Delete from Azure (UUID-based storage only)
                if old_entry_id:
                    delete_result_from_azure_by_entry_id(old_entry_id)

                # Delete local ZIP if exists (UUID-based path)
                if old_entry_id:
                    prefix = old_entry_id[:2].lower()
                    local_zip = settings.RESULTS_DIR / prefix / f"{old_entry_id}.zip"
                    if local_zip.exists():
                        try:
                            local_zip.unlink()
                            logger.info(f"Deleted local result for replacement: {local_zip}")
                        except Exception as e:
                            logger.warning(f"Failed to delete local result: {e}")

                # Delete from database
                db.delete(existing)
                db.commit()
                logger.info(f"Deleted existing compound '{old_name}' (entry_id={old_entry_id}) for replacement with '{request.compound_name}'")

        # Use new_compound_name if provided (for exact duplicates where user wants to change name)
        compound_name = request.new_compound_name or request.compound_name

        # Create new job
        job = job_service.create_job(
            db,
            JobType.SINGLE,
            {
                "compound_name": compound_name,
                "smiles": request.smiles,
                "similarity_threshold": request.similarity_threshold,
                "activity_types": request.activity_types,
            },
            session_id=session_id,
        )

        job_scheduler.trigger()
        logger.info(f"Job {job.id} created as replacement for {compound_name}")
        return _job_to_response(job)

    # Handle DUPLICATE action - create with duplicate tag
    if request.action == DuplicateAction.DUPLICATE:
        # Use new_compound_name if provided (for exact duplicates where user changes name)
        compound_name = request.new_compound_name or request.compound_name

        # Create job with duplicate metadata
        job = job_service.create_job(
            db,
            JobType.SINGLE,
            {
                "compound_name": compound_name,
                "smiles": request.smiles,
                "similarity_threshold": request.similarity_threshold,
                "activity_types": request.activity_types,
                # Store duplicate metadata in input_params
                "is_duplicate": True,
                "duplicate_of": request.existing_entry_id,
            },
            session_id=session_id,
        )

        job_scheduler.trigger()
        logger.info(f"Job {job.id} created as duplicate (tagged) for {compound_name}")
        return _job_to_response(job)

    # Should not reach here, but handle gracefully
    raise HTTPException(status_code=422, detail=f"Invalid action: {request.action}")


def _inchi_to_smiles(inchi: str) -> Optional[str]:
    """Convert InChI to SMILES using RDKit."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromInchi(inchi)
        if mol:
            return Chem.MolToSmiles(mol)
        return None
    except Exception as e:
        logger.debug(f"InChI to SMILES conversion failed: {e}")
        return None


@router.post(
    "/check-duplicates",
    response_model=CheckDuplicatesResponse,
    summary="Check for duplicate compounds",
)
async def check_duplicates(
    request: CheckDuplicatesRequest,
    db: Session = Depends(get_db),
):
    """
    Check which compounds already exist or are being processed.

    Supports two modes:
    1. **Name-only check (legacy)**: Provide `compound_names` list - checks by name only
    2. **Structure-based check (recommended)**: Provide `compounds` list with SMILES/InChI
       - Generates InChIKey for each compound
       - Checks for existing compounds with same InChIKey (100% accurate structure match)
       - Returns `structure_matches` with details about which compounds match by structure

    Returns:
        - existing: Compounds that already have results (by name)
        - processing: Compounds currently being processed
        - new: Compounds that are new
        - structure_matches: Compounds that match existing compounds by InChIKey (structure)
    """
    structure_matches: List[DuplicateMatch] = []

    # Determine which mode we're in
    if request.compounds:
        # New mode: structure-based checking with InChIKey
        compound_names = [c.compound_name for c in request.compounds]

        # Generate InChIKeys and check for structure matches
        for compound in request.compounds:
            smiles = compound.smiles
            # Convert InChI to SMILES if needed
            if not smiles and compound.inchi:
                smiles = _inchi_to_smiles(compound.inchi)

            if smiles:
                inchikey = generate_inchikey(smiles)
                if inchikey:
                    # Check if any existing compound has this InChIKey
                    existing_compound = db.query(Compound).filter(
                        Compound.inchikey == inchikey
                    ).first()

                    if existing_compound:
                        # Determine match type
                        name_matches = existing_compound.compound_name.lower().strip() == compound.compound_name.lower().strip()
                        match_type = "exact" if name_matches else "structure_only"

                        structure_matches.append(DuplicateMatch(
                            compound_name=compound.compound_name,
                            inchikey=inchikey,
                            existing_compound_name=existing_compound.compound_name,
                            existing_entry_id=existing_compound.entry_id,
                            match_type=match_type,
                        ))
                        logger.debug(
                            f"InChIKey match: {compound.compound_name} matches "
                            f"{existing_compound.compound_name} ({match_type})"
                        )
    elif request.compound_names:
        # Legacy mode: name-only checking
        compound_names = request.compound_names
    else:
        raise HTTPException(
            status_code=422,
            detail="Must provide either 'compound_names' or 'compounds' list"
        )

    # Check for already processed compounds (by name)
    existing_map = job_service.check_existing_compounds(db, compound_names)
    existing = [name for name, exists in existing_map.items() if exists]

    # Check for currently processing compounds
    pending_map = job_service.check_pending_compounds(db, compound_names)
    processing = list(pending_map.keys())

    # Calculate new compounds (by name)
    skip_set = set(existing) | set(processing)
    new = [name for name in compound_names if name not in skip_set]

    # Also mark compounds with structure matches as not truly new
    structure_match_names = {m.compound_name for m in structure_matches}
    new = [name for name in new if name not in structure_match_names]

    logger.info(
        f"Duplicate check: {len(existing)} existing (name), {len(processing)} processing, "
        f"{len(structure_matches)} structure matches, {len(new)} new"
    )

    return CheckDuplicatesResponse(
        existing=existing,
        processing=processing,
        new=new,
        structure_matches=structure_matches,
    )


@router.post(
    "/batch",
    status_code=201,
    responses={429: {"model": ErrorResponse}},
    summary="Submit a batch of jobs",
)
async def create_batch_job(
    request: BatchJobCreate,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """
    Submit multiple compound processing jobs.

    Each compound is processed as a separate job, linked by batch_id.

    Features:
    - Per-compound duplicate handling: each compound can have a `duplicate_action`
      field ('skip', 'replace', 'duplicate') to control how to handle existing compounds
    - Skips compounds currently being processed
    - Groups all jobs under a single batch_id for batch operations

    Headers:
        X-Session-ID: Session ID for user isolation (validated UUID)

    Returns:
        Dict with created jobs, skipped/replaced compounds, and batch summary
    """
    # Use session_id from validated header, fall back to request body if anonymous
    if session_id.startswith("anon-") and request.session_id:
        session_id = request.session_id

    # Validate batch size
    if len(request.compounds) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large. Maximum {MAX_BATCH_SIZE} compounds allowed."
        )

    # Check rate limit for batch submissions
    allowed, remaining = rate_limiter.check_rate_limit(
        f"{session_id}_batch", RATE_LIMIT_MAX_BATCH
    )
    if not allowed:
        logger.warning(f"Batch rate limit exceeded for session {truncate_session_id(session_id)}")
        log_rate_limit_exceeded(truncate_session_id(session_id), "batch_job", RATE_LIMIT_MAX_BATCH)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX_BATCH} batch submissions per minute.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )

    # Generate a batch_id to link all jobs
    batch_id = job_service.generate_batch_id()

    # Extract duplicate_decisions from request (if provided)
    # This maps compound_name -> action ('skip', 'replace', 'duplicate')
    duplicate_decisions = request.duplicate_decisions or {}

    # Check for currently processing compounds (always skipped)
    compound_names = [c.compound_name for c in request.compounds]
    pending = job_service.check_pending_compounds(db, compound_names)
    skipped_processing = list(pending.keys())

    # Track results
    skipped_existing = []  # Compounds skipped by user choice
    replaced = []  # Compounds replaced (existing deleted)
    marked_duplicate = []  # Compounds marked as duplicates

    # Create all jobs in SQLite (status: PENDING)
    # Scheduler will pick them up and process 2 at a time
    jobs = []
    for compound in request.compounds:
        compound_name = compound.compound_name

        # Skip if currently processing
        if compound_name in skipped_processing:
            continue

        # Get per-compound duplicate action (from compound data or duplicate_decisions dict)
        compound_action = getattr(compound, 'duplicate_action', None) or duplicate_decisions.get(compound_name)

        # If action is 'skip', don't process
        if compound_action == 'skip':
            skipped_existing.append(compound_name)
            continue

        # If action is 'replace', delete existing compound first
        if compound_action == 'replace':
            # Find and delete existing compound
            existing = db.query(Compound).filter(Compound.compound_name == compound_name).first()
            if existing:
                old_entry_id = existing.entry_id

                # Delete from Azure (UUID-based storage only)
                if old_entry_id:
                    delete_result_from_azure_by_entry_id(old_entry_id)

                    # Delete local ZIP if exists (UUID-based path)
                    prefix = old_entry_id[:2].lower()
                    local_zip = settings.RESULTS_DIR / prefix / f"{old_entry_id}.zip"
                    if local_zip.exists():
                        try:
                            local_zip.unlink()
                            logger.debug(f"Deleted local result for batch replace: {local_zip}")
                        except Exception as e:
                            logger.warning(f"Failed to delete local result: {e}")

                # Delete from database
                db.delete(existing)
                db.commit()
                logger.info(f"Batch replace: deleted existing compound '{compound_name}' (entry_id={old_entry_id})")
                replaced.append(compound_name)

        # Build job params - include duplicate metadata if marking as duplicate
        job_params = compound.model_dump(exclude={"session_id", "duplicate_action", "original_compound_name"})
        if compound_action == 'duplicate':
            # Find existing compound to reference - use original_compound_name if provided
            # (frontend sends new name in compound_name, original in original_compound_name)
            original_name = getattr(compound, 'original_compound_name', None) or compound_name
            existing = db.query(Compound).filter(Compound.compound_name == original_name).first()
            job_params["is_duplicate"] = True
            job_params["duplicate_of"] = existing.entry_id if existing else None
            marked_duplicate.append(compound_name)

        job = job_service.create_job(
            db,
            JobType.BATCH,
            job_params,
            session_id=session_id,
            batch_id=batch_id,
        )
        jobs.append(_job_to_response(job))

    # Trigger scheduler to start processing (if not already running)
    if jobs:
        job_scheduler.trigger()

    total_skipped = len(skipped_existing) + len(skipped_processing)
    logger.info(
        f"Batch {batch_id}: {len(jobs)} jobs queued, "
        f"{len(replaced)} replaced, {len(marked_duplicate)} as duplicates, "
        f"{total_skipped} skipped (session={truncate_session_id(session_id)})"
    )

    return BatchResponse(
        batch_id=batch_id,
        jobs=jobs,
        skipped_existing=skipped_existing,
        skipped_processing=skipped_processing,
        replaced=replaced,
        total_submitted=len(jobs),
        total_skipped=total_skipped,
    )


@router.get(
    "",
    response_model=JobListResponse,
    summary="List all jobs",
)
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    List all jobs with optional status filter and pagination.
    """
    statuses = [status] if status else None
    result = job_service.list_jobs(db, statuses=statuses, page=page, page_size=page_size)

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
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """
    Get active (pending/processing) jobs for the current session.

    Used by the frontend sidebar to display job progress.
    Users only see their own jobs when X-Session-ID is provided.

    Headers:
        X-Session-ID: Session ID for filtering jobs (required for isolation)
    """
    return job_service.get_active_jobs(db, session_id=x_session_id)


@router.get(
    "/batch/{batch_id}",
    response_model=BatchSummary,
    summary="Get batch summary",
)
async def get_batch_summary(
    batch_id: str,
    db: Session = Depends(get_db),
):
    """
    Get summary statistics for a batch of jobs.

    Returns overall progress and status counts.
    """
    summary = job_service.get_batch_summary(db, batch_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Batch not found")
    return summary


@router.post(
    "/batch/{batch_id}/cancel",
    summary="Cancel all jobs in a batch",
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def cancel_batch(
    batch_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """
    Cancel all pending/processing jobs in a batch.

    Requires ownership of the batch (same session that created it).
    Already completed or failed jobs are not affected.
    """
    from backend.models.database import Job

    # Verify the batch exists and belongs to this session
    batch_jobs = db.query(Job).filter(Job.batch_id == batch_id).all()

    if not batch_jobs:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Check ownership - verify session owns all jobs in batch
    # (All jobs in a batch should have the same session_id)
    first_job = batch_jobs[0]
    if first_job.session_id and first_job.session_id != session_id:
        logger.warning(
            f"Unauthorized batch cancel attempt: session {truncate_session_id(session_id)} "
            f"tried to cancel batch {batch_id}"
        )
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to cancel this batch"
        )

    cancelled_count = job_service.cancel_batch(db, batch_id)

    # Audit log
    log_job_cancelled(truncate_session_id(session_id), f"batch:{batch_id}", f"{cancelled_count} jobs")

    # Also cancel in executor
    # Note: Jobs already running may not stop immediately
    return CancelResponse(
        batch_id=batch_id,
        cancelled_count=cancelled_count,
        message=f"Cancelled {cancelled_count} jobs in batch",
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get job status",
)
async def get_job(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Get the current status of a job.

    Poll this endpoint (1s interval) to track progress.
    """
    job = job_service.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return _job_to_response(job)


@router.get(
    "/{job_id}/detail",
    responses={404: {"model": ErrorResponse}},
    summary="Get detailed job info",
)
async def get_job_detail(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Get detailed job information including parsed input parameters.
    """
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
    """
    Cancel a pending or processing job.

    Requires ownership of the job (same session that created it).
    Note: Jobs already running may not be cancelled immediately.
    """
    import json

    # Verify ownership
    job = _verify_job_ownership(db, job_id, session_id)

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=409,  # Conflict - job in wrong state
            detail=f"Job cannot be cancelled (status: {job.status.value})",
        )

    # Extract compound name for audit log
    compound_name = None
    if job.input_params:
        try:
            params = json.loads(job.input_params)
            compound_name = params.get("compound_name")
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to cancel in executor
    executor_cancelled = job_executor.cancel(job_id)

    # Always mark as cancelled in DB
    job = job_service.cancel_job(db, job_id)

    # Audit log
    log_job_cancelled(truncate_session_id(session_id), job_id, compound_name)

    if not executor_cancelled:
        logger.warning(f"Job {job_id} marked cancelled but was already running")

    return _job_to_response(job)


@router.delete(
    "/{job_id}",
    responses={404: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Delete a job record",
)
async def delete_job(
    job_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """
    Delete a job record and associated result files.

    - Deletes job from database
    - Deletes result ZIP from Azure
    - Deletes local result files

    Requires ownership of the job (same session that created it).
    Only completed, failed, or cancelled jobs can be deleted.
    """
    import json
    from pathlib import Path

    # Verify ownership
    job = _verify_job_ownership(db, job_id, session_id)

    if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
        raise HTTPException(
            status_code=409,  # Conflict - job in wrong state
            detail="Cannot delete active jobs. Cancel first.",
        )

    # Extract compound name from job params for file cleanup
    compound_name = None
    if job.input_params:
        try:
            params = json.loads(job.input_params)
            compound_name = params.get("compound_name")
        except (json.JSONDecodeError, TypeError):
            pass

    # Clean up result files
    if compound_name:
        # Try to find the compound entry - check job's result_summary for entry_id first
        entry_id = None
        if job.result_summary:
            try:
                result_data = json.loads(job.result_summary)
                entry_id = result_data.get("entry_id")
            except (json.JSONDecodeError, TypeError):
                pass

        # Find compound by entry_id (precise) or compound_name (fallback)
        compound_entry = None
        if entry_id:
            compound_entry = db.query(Compound).filter(Compound.entry_id == entry_id).first()
        if not compound_entry:
            # Fallback to compound_name - but this may match wrong compound for duplicates
            compound_entry = db.query(Compound).filter(Compound.compound_name == compound_name).first()
            if compound_entry:
                entry_id = compound_entry.entry_id

        # Delete from Azure (UUID-based storage only)
        if entry_id:
            azure_deleted = delete_result_from_azure_by_entry_id(entry_id)
            if azure_deleted:
                logger.info(f"Deleted result from Azure: {entry_id}")
            else:
                logger.warning(f"Failed to delete result from Azure: {entry_id}")

        # Delete local ZIP if exists (UUID-based path only)
        if entry_id:
            prefix = entry_id[:2].lower()
            local_zip = settings.RESULTS_DIR / prefix / f"{entry_id}.zip"
            if local_zip.exists():
                try:
                    local_zip.unlink()
                    logger.info(f"Deleted local result: {local_zip}")
                except Exception as e:
                    logger.warning(f"Failed to delete local result {local_zip}: {e}")

        # Archive to deleted_compounds table and delete from compounds
        if compound_entry:
            # Create audit record before deletion
            deleted_record = DeletedCompound(
                original_id=compound_entry.id,
                entry_id=compound_entry.entry_id,
                compound_name=compound_entry.compound_name,
                chembl_id=compound_entry.chembl_id,
                smiles=compound_entry.smiles,
                inchikey=compound_entry.inchikey,
                is_duplicate=compound_entry.is_duplicate,
                duplicate_of=compound_entry.duplicate_of,
                storage_path=compound_entry.storage_path,
                deleted_by_session=session_id,
                deleted_by_job_id=job_id,
                deletion_reason="user_request",
                original_processed_at=compound_entry.processed_at,
            )
            db.add(deleted_record)

            # Delete from compounds table
            db.delete(compound_entry)
            db.commit()
            logger.info(f"Archived and deleted compound: {compound_name} (entry_id={entry_id})")

    # Audit log before deletion
    log_job_deleted(truncate_session_id(session_id), job_id, compound_name)

    # Delete job record from database
    job_service.delete_job(db, job_id)

    return DeleteResponse(
        message="Job and results deleted",
        job_id=job_id,
        compound_name=compound_name,
    )
