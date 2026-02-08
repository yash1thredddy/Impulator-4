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
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Header
from sqlalchemy import func
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
    InternalDuplicateMatch,
)
from backend.services.job_service import job_service, generate_inchikey
from backend.models.database import Compound, DeletedCompound, Job
from backend.core.azure_sync import delete_result_from_azure_by_entry_id
from backend.core.audit import (
    log_rate_limit_exceeded,
    log_job_cancelled,
    log_job_deleted,
)
from backend.config import settings

logger = logging.getLogger(__name__)

# Default activity types (matches api_client.py and frontend config)
_DEFAULT_ACTIVITY_TYPES = "AC50,EC50,GI50,IC50,Kd,Ki,MIC"


def _normalize_activity_types(activity_types: Optional[List[str]]) -> str:
    """Normalize a list of activity types to sorted comma-separated string for comparison."""
    if not activity_types:
        return _DEFAULT_ACTIVITY_TYPES
    return ",".join(sorted(at.strip() for at in activity_types))


def _normalize_activity_types_str(stored: Optional[str]) -> str:
    """Normalize a stored comma-separated activity_types string for comparison."""
    if not stored:
        return _DEFAULT_ACTIVITY_TYPES
    return ",".join(sorted(at.strip() for at in stored.split(",")))


def _normalize_inchikey_input(inchikey: Optional[str]) -> Optional[str]:
    """Normalize optional InChIKey input from request payload."""
    if not inchikey:
        return None
    normalized = inchikey.strip().upper()
    if not normalized or normalized in {"NAN", "NONE"}:
        return None
    return normalized


def _inchikey_structure_key(inchikey: Optional[str]) -> Optional[str]:
    """Return a protonation-insensitive structure key from an InChIKey.

    InChIKey format is typically AAAA...-BBBB...-C where the last block encodes
    protonation/version information. For in-file duplicate detection we ignore the
    last block so acid/base forms can be treated as the same underlying structure.
    """
    if not inchikey:
        return None
    normalized = inchikey.strip().upper()
    parts = normalized.split("-")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return f"{parts[0]}-{parts[1]}"
    return normalized or None


def _compute_config_match(
    existing: Compound,
    submitted_threshold: int,
    submitted_activity_types: str,
) -> str:
    """Compare existing compound's config with submitted config.

    Returns one of:
    - 'identical': Same threshold AND same activity types
    - 'different_threshold': Different threshold, same activity types
    - 'different_activities': Same threshold, different activity types
    - 'different_both': Different threshold AND different activity types
    """
    threshold_same = (existing.similarity_threshold or 90) == submitted_threshold
    at_same = _normalize_activity_types_str(existing.activity_types) == submitted_activity_types

    if threshold_same and at_same:
        return "identical"
    elif not threshold_same and at_same:
        return "different_threshold"
    elif threshold_same and not at_same:
        return "different_activities"
    return "different_both"


def get_next_version_name(db: Session, base_name: str) -> str:
    """
    Calculate the next available version name for a compound.

    Given a base name like 'Quercetin', checks existing compounds for:
    - 'Quercetin' (original)
    - 'Quercetin_v2', 'Quercetin_v3', etc. (versions)

    Returns the next available version name (e.g., 'Quercetin_v3' if v2 exists).

    Args:
        db: Database session
        base_name: The base compound name (without version suffix)

    Returns:
        Next available version name (e.g., 'Quercetin_v2' or 'Quercetin_v4')
    """
    import re

    # Strip any existing version suffix from base_name to get true base
    version_pattern = re.compile(r'^(.+?)(_v(\d+))?$')
    match = version_pattern.match(base_name)
    if match:
        true_base = match.group(1).strip()
    else:
        true_base = base_name.strip()

    # Query all compound names that start with the base name
    existing_names = db.query(Compound.compound_name).filter(
        func.lower(func.trim(Compound.compound_name)).like(f"{true_base.lower()}%")
    ).all()

    existing_names = [name[0] for name in existing_names]

    # Find highest existing version number
    max_version = 1  # Start at 1 (original has no suffix, v2 is first duplicate)

    version_suffix_pattern = re.compile(
        rf'^{re.escape(true_base)}(_v(\d+))?$',
        re.IGNORECASE,
    )

    for name in existing_names:
        match = version_suffix_pattern.match(name)
        if match:
            if match.group(2):
                # Has version suffix like _v2, _v3
                version = int(match.group(2))
                max_version = max(max_version, version)
            else:
                # Original name (no suffix) counts as version 1
                max_version = max(max_version, 1)

    # Return next version
    next_version = max_version + 1
    return f"{true_base}_v{next_version}"


def get_next_version_names_bulk(db: Session, compound_names: List[str]) -> Dict[str, str]:
    """
    Calculate next available version names for multiple compounds in a single query.

    This is the bulk-optimized version of get_next_version_name() that avoids
    N+1 query issues. For 1000 compounds, this performs 1 query instead of 1000.

    Args:
        db: Database session
        compound_names: List of compound names to get versions for

    Returns:
        Dict mapping original compound name to its next version name
    """
    import re
    from sqlalchemy import or_

    if not compound_names:
        return {}

    # Extract true base names (strip any existing _vN suffix)
    version_pattern = re.compile(r'^(.+?)(_v(\d+))?$')
    name_to_base: Dict[str, str] = {}  # original input name -> true_base
    normalized_base_to_sample: Dict[str, str] = {}  # lower(base) -> representative base

    for original_name in compound_names:
        clean_name = (original_name or "").strip()
        if not clean_name:
            continue

        match = version_pattern.match(clean_name)
        true_base = (match.group(1) if match else clean_name).strip()
        normalized_base = true_base.lower()

        name_to_base[original_name] = true_base
        if normalized_base not in normalized_base_to_sample:
            normalized_base_to_sample[normalized_base] = true_base

    if not name_to_base:
        return {}

    # Build OR conditions for all bases (single query), case-insensitive
    like_conditions = [
        func.lower(func.trim(Compound.compound_name)).like(f"{base_norm}%")
        for base_norm in normalized_base_to_sample.keys()
    ]

    all_existing = db.query(Compound.compound_name).filter(or_(*like_conditions)).all()
    existing_names = {row[0] for row in all_existing if row[0]}

    # Compute max version for each normalized base name
    base_max_versions = {base_norm: 1 for base_norm in normalized_base_to_sample.keys()}

    for base_norm, sample_base in normalized_base_to_sample.items():
        suffix_pattern = re.compile(
            rf'^{re.escape(sample_base)}(_v(\d+))?$',
            re.IGNORECASE,
        )

        for existing_name in existing_names:
            match = suffix_pattern.match(existing_name)
            if match:
                if match.group(2):
                    version = int(match.group(2))
                    base_max_versions[base_norm] = max(base_max_versions[base_norm], version)
                else:
                    base_max_versions[base_norm] = max(base_max_versions[base_norm], 1)

    # Build result dict using original input names as keys
    result: Dict[str, str] = {}
    for original_name in compound_names:
        true_base = name_to_base.get(original_name)
        if not true_base:
            continue
        next_version = base_max_versions.get(true_base.lower(), 1) + 1
        result[original_name] = f"{true_base}_v{next_version}"

    logger.debug(f"Bulk version names: computed {len(result)} versions in 1 query")
    return result


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


def _verify_job_ownership(db: Session, job_id: str, session_id: str) -> Job:
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
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Submit a new compound processing job.

    The job runs in the background. Use GET /jobs/{id} to check status.
    Jobs are queued in SQLite and picked up by the scheduler.

    **Duplicate Detection:**
    If a compound with the same InChIKey already exists, returns a
    `duplicate_found` response instead of creating the job. Use
    `/jobs/resolve-duplicate` to handle the duplicate.

    **Idempotency:**
    Include an `Idempotency-Key` header (max 64 chars) to safely retry
    failed requests. If a job with the same key exists for your session,
    it will be returned instead of creating a new job.

    Headers:
        X-Session-ID: Session ID for user isolation (validated UUID)
        Idempotency-Key: Optional key for safe retries (max 64 chars)

    Rate Limit:
        Max 10 jobs per minute per session

    Returns:
        - JobResponse if job created successfully
        - DuplicateFoundResponse if duplicate detected
    """
    # Always use the validated header session_id (body session_id is ignored for security)

    # Check idempotency key - return existing job if already created
    if idempotency_key:
        # Truncate to max 64 chars for safety
        idempotency_key = idempotency_key[:64]
        existing_job = db.query(Job).filter(
            Job.session_id == session_id,
            Job.idempotency_key == idempotency_key
        ).first()
        if existing_job:
            logger.info(f"Idempotent request - returning existing job {existing_job.id}")
            return _job_to_response(existing_job)

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

    # Pre-compute submitted config for comparison
    submitted_threshold = request.similarity_threshold or 90
    submitted_at = _normalize_activity_types(request.activity_types)

    # Helper function to build duplicate response with config comparison
    def _build_duplicate_response(
        existing_compound: Compound,
        config_match: str,
    ) -> DuplicateFoundResponse:
        name_matches = existing_compound.compound_name.lower().strip() == request.compound_name.lower().strip()
        duplicate_type = "exact" if name_matches else "structure_only"

        # Calculate the next available version name for duplicates
        suggested_name = get_next_version_name(db, existing_compound.compound_name)

        # Build config diff for non-identical configs
        config_diff = None
        if config_match != "identical":
            config_diff = {
                "similarity_threshold": {
                    "existing": existing_compound.similarity_threshold or 90,
                    "submitted": submitted_threshold,
                },
                "activity_types": {
                    "existing": _normalize_activity_types_str(existing_compound.activity_types),
                    "submitted": submitted_at,
                },
            }

        logger.info(
            f"Duplicate found: {request.compound_name} matches "
            f"{existing_compound.compound_name} (InChIKey: {inchikey[:14]}..., "
            f"config: {config_match}), suggested: {suggested_name}"
        )

        return DuplicateFoundResponse(
            status="duplicate_found",
            duplicate_type=duplicate_type,
            config_match=config_match,
            existing_compound=ExistingCompoundInfo(
                entry_id=existing_compound.entry_id,
                compound_name=existing_compound.compound_name,
                inchikey=existing_compound.inchikey,
                processed_at=existing_compound.processed_at.isoformat() if existing_compound.processed_at else None,
                similarity_threshold=existing_compound.similarity_threshold,
                activity_types=_normalize_activity_types_str(existing_compound.activity_types),
                author_name=existing_compound.author_name,
            ),
            submitted={
                "compound_name": request.compound_name,
                "inchikey": inchikey,
                "smiles": request.smiles,
                "similarity_threshold": submitted_threshold,
                "activity_types": submitted_at,
            },
            suggested_name=suggested_name,
            config_diff=config_diff,
        )

    def _find_best_duplicate_match() -> Optional[DuplicateFoundResponse]:
        """Find the best matching existing compound for duplicate detection.

        Prioritizes: exact config match > first non-duplicate > any match.
        """
        if not inchikey:
            return None

        existing_compounds = db.query(Compound).filter(
            Compound.inchikey == inchikey
        ).all()

        if not existing_compounds:
            return None

        # Find best match: exact config > first non-duplicate > first overall
        exact_config_match = None
        first_non_duplicate = None

        for comp in existing_compounds:
            config = _compute_config_match(comp, submitted_threshold, submitted_at)
            if config == "identical":
                exact_config_match = comp
                break
            if not comp.is_duplicate and first_non_duplicate is None:
                first_non_duplicate = comp

        match_compound = exact_config_match or first_non_duplicate or existing_compounds[0]
        config_match = _compute_config_match(match_compound, submitted_threshold, submitted_at)
        return _build_duplicate_response(match_compound, config_match)

    # Atomic check-and-create with retry for race condition handling
    # SQLite doesn't support FOR UPDATE, so we use retry logic instead
    max_retries = 3
    for attempt in range(max_retries):
        # Check for duplicate by InChIKey with config comparison
        dup_response = _find_best_duplicate_match()
        if dup_response:
            return dup_response

        try:
            # No duplicate found - create job
            job = job_service.create_job(
                db,
                JobType.SINGLE,
                request.model_dump(exclude={"session_id"}),
                session_id=session_id,
                idempotency_key=idempotency_key,
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
            dup_response = _find_best_duplicate_match()
            if dup_response:
                return dup_response

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
    # Always use the validated header session_id (body session_id is ignored for security)

    # Handle SKIP action
    if request.action == DuplicateAction.SKIP:
        logger.info(f"User skipped duplicate: {request.compound_name}")
        return SkipResponse(
            status="skipped",
            message=f"Compound '{request.compound_name}' processing skipped by user",
            compound_name=request.compound_name,
        )

    # Handle REPLACE action - defer deletion until job COMPLETES (prevents data loss if job fails)
    if request.action == DuplicateAction.REPLACE:
        inherit_children_from = None
        replace_entry_id = None
        if request.existing_entry_id:
            existing = db.query(Compound).filter(Compound.entry_id == request.existing_entry_id).first()
            if existing:
                old_name = existing.compound_name
                old_entry_id = existing.entry_id
                replace_entry_id = old_entry_id

                # Only reparent children if the replaced compound is canonical (not a duplicate itself)
                if not existing.is_duplicate:
                    children = db.query(Compound).filter(
                        Compound.duplicate_of == old_entry_id
                    ).all()
                    if children:
                        inherit_children_from = old_entry_id
                        logger.info(
                            f"Compound '{old_name}' has {len(children)} children - "
                            f"will inherit after replacement completes"
                        )

                logger.info(
                    f"Replacement requested for '{old_name}' (entry_id={old_entry_id}) "
                    f"with '{request.compound_name}' - deletion deferred until job completes"
                )

        # Inherit the replaced compound's name (not the user-entered name)
        # so "Replace Quercetin_v2" creates a new "Quercetin_v2", not "Quercetin"
        if replace_entry_id and old_name:
            compound_name = request.new_compound_name or old_name
        else:
            compound_name = request.new_compound_name or request.compound_name

        # Create new job - store replace_entry_id so deletion happens on job completion
        # If the job fails, the old compound remains intact
        job_params = {
            "compound_name": compound_name,
            "author_name": request.author_name,
            "smiles": request.smiles,
            "similarity_threshold": request.similarity_threshold,
            "activity_types": request.activity_types,
        }
        if inherit_children_from:
            job_params["inherit_children_from"] = inherit_children_from
        if replace_entry_id:
            job_params["replace_entry_id"] = replace_entry_id
            # Inherit duplicate metadata from the old compound so the replacement
            # retains its duplicate status and parent link
            if existing and existing.is_duplicate:
                job_params["is_duplicate"] = True
                job_params["duplicate_of"] = existing.duplicate_of

        job = job_service.create_job(
            db,
            JobType.SINGLE,
            job_params,
            session_id=session_id,
        )

        job_scheduler.trigger()
        logger.info(f"Job {job.id} created as replacement for {compound_name}")
        return _job_to_response(job)

    # Handle DUPLICATE action - create with duplicate tag
    if request.action == DuplicateAction.DUPLICATE:
        # Duplicate action must always reference an existing parent entry.
        if not request.existing_entry_id:
            raise HTTPException(
                status_code=422,
                detail="existing_entry_id is required for duplicate action.",
            )

        existing = db.query(Compound).filter(
            Compound.entry_id == request.existing_entry_id
        ).first()
        if not existing:
            raise HTTPException(
                status_code=422,
                detail="Invalid existing_entry_id for duplicate action.",
            )

        # Block duplicate action for identical configs (server-side validation)
        submitted_at = _normalize_activity_types(request.activity_types)
        config = _compute_config_match(
            existing, request.similarity_threshold or 90, submitted_at
        )
        if config == "identical":
            raise HTTPException(
                status_code=422,
                detail="Cannot create duplicate with identical configuration. "
                       "Use 'replace' to reprocess or 'skip' to keep existing.",
            )

        # Use new_compound_name if provided (for exact duplicates where user changes name)
        compound_name = request.new_compound_name or request.compound_name

        # Create job with duplicate metadata
        job = job_service.create_job(
            db,
            JobType.SINGLE,
            {
                "compound_name": compound_name,
                "author_name": request.author_name,
                "smiles": request.smiles,
                "similarity_threshold": request.similarity_threshold,
                "activity_types": request.activity_types,
                # Store duplicate metadata in input_params
                "is_duplicate": True,
                "duplicate_of": existing.entry_id,
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
    2. **Structure-based check (recommended)**: Provide `compounds` list with SMILES/InChI/InChIKey
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
    internal_duplicates: List[InternalDuplicateMatch] = []
    submitted_threshold = request.similarity_threshold or 90
    submitted_at = _normalize_activity_types(request.activity_types)

    def _normalize_name(name: Optional[str]) -> str:
        return (name or "").strip().lower()

    # Determine which mode we're in
    if request.compounds:
        # New mode: structure-based checking with InChIKey
        compound_names: List[str] = []

        # Track first-seen compounds within the submitted payload to detect in-file duplicates.
        seen_structure_key_to_name: Dict[str, str] = {}
        seen_name_to_name: Dict[str, str] = {}

        # Generate InChIKeys and check for structure matches
        for compound in request.compounds:
            submitted_name = compound.compound_name
            normalized_submitted_name = _normalize_name(submitted_name)

            smiles = compound.smiles
            # Convert InChI to SMILES if needed
            if not smiles and compound.inchi:
                smiles = _inchi_to_smiles(compound.inchi)

            # Prefer generated InChIKey from structure input when available;
            # fall back to provided InChIKey for InChIKey-only uploads.
            provided_inchikey = _normalize_inchikey_input(getattr(compound, "inchikey", None))
            generated_inchikey = generate_inchikey(smiles) if smiles else None
            inchikey = generated_inchikey or provided_inchikey
            structure_key = _inchikey_structure_key(inchikey)

            # First detect duplicates within the uploaded payload itself.
            internal_parent_name = None
            internal_match_type = "exact"

            # Always treat repeated submitted names as in-file duplicates.
            if normalized_submitted_name:
                internal_parent_name = seen_name_to_name.get(normalized_submitted_name)
                if internal_parent_name:
                    internal_match_type = "exact"
                else:
                    seen_name_to_name[normalized_submitted_name] = submitted_name

            # Also treat repeated structure keys as in-file duplicates. This catches
            # rows that differ only by protonation/charge layer in InChI.
            if not internal_parent_name and structure_key:
                internal_parent_name = seen_structure_key_to_name.get(structure_key)
                if internal_parent_name:
                    internal_match_type = (
                        "exact"
                        if _normalize_name(internal_parent_name) == normalized_submitted_name
                        else "structure_only"
                    )
                else:
                    seen_structure_key_to_name[structure_key] = submitted_name

            if internal_parent_name:
                internal_duplicates.append(
                    InternalDuplicateMatch(
                        compound_name=submitted_name,
                        duplicate_of=internal_parent_name,
                        match_type=internal_match_type,
                        inchikey=inchikey,
                    )
                )
                logger.debug(
                    f"Internal duplicate in upload: {submitted_name} duplicates "
                    f"{internal_parent_name} ({internal_match_type})"
                )
                continue

            # Keep only non-duplicate upload entries for DB duplicate checks.
            compound_names.append(submitted_name)

            if inchikey:
                # Check if any existing compound has this InChIKey
                existing_candidates = db.query(Compound).filter(
                    Compound.inchikey == inchikey
                ).all()

                if existing_candidates:
                    # Prefer same-name candidates when available for better UX parity with single mode.
                    same_name_candidates = [
                        c for c in existing_candidates
                        if _normalize_name(c.compound_name) == normalized_submitted_name
                    ]
                    candidates_for_selection = same_name_candidates or existing_candidates

                    # Prioritize: exact config > first non-duplicate > first overall
                    exact_config_match = None
                    first_non_duplicate = None
                    for candidate in candidates_for_selection:
                        config = _compute_config_match(candidate, submitted_threshold, submitted_at)
                        if config == "identical":
                            exact_config_match = candidate
                            break
                        if not candidate.is_duplicate and first_non_duplicate is None:
                            first_non_duplicate = candidate

                    existing_compound = exact_config_match or first_non_duplicate or candidates_for_selection[0]
                    config_match = _compute_config_match(existing_compound, submitted_threshold, submitted_at)
                    config_diff = None
                    if config_match != "identical":
                        config_diff = {
                            "similarity_threshold": {
                                "existing": existing_compound.similarity_threshold or 90,
                                "submitted": submitted_threshold,
                            },
                            "activity_types": {
                                "existing": _normalize_activity_types_str(existing_compound.activity_types),
                                "submitted": submitted_at,
                            },
                        }

                    match_type = (
                        "exact"
                        if _normalize_name(existing_compound.compound_name) == normalized_submitted_name
                        else "structure_only"
                    )

                    structure_matches.append(DuplicateMatch(
                        compound_name=submitted_name,
                        inchikey=inchikey,
                        existing_compound_name=existing_compound.compound_name,
                        existing_entry_id=existing_compound.entry_id,
                        match_type=match_type,
                        config_match=config_match,
                        config_diff=config_diff,
                        existing_similarity_threshold=existing_compound.similarity_threshold or 90,
                        existing_activity_types=_normalize_activity_types_str(existing_compound.activity_types),
                        existing_author_name=existing_compound.author_name,
                        existing_processed_at=(
                            existing_compound.processed_at.isoformat()
                            if existing_compound.processed_at else None
                        ),
                    ))
                    logger.debug(
                        f"InChIKey match: {submitted_name} matches "
                        f"{existing_compound.compound_name} ({match_type}, config={config_match})"
                    )
    elif request.compound_names:
        # Legacy mode: name-only checking
        compound_names = []
        seen_names = {}
        for name in request.compound_names:
            normalized = _normalize_name(name)
            if normalized and normalized in seen_names:
                internal_duplicates.append(
                    InternalDuplicateMatch(
                        compound_name=name,
                        duplicate_of=seen_names[normalized],
                        match_type="exact",
                        inchikey=None,
                    )
                )
                continue
            if normalized:
                seen_names[normalized] = name
            compound_names.append(name)
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
        f"{len(structure_matches)} structure matches, {len(internal_duplicates)} internal duplicates, "
        f"{len(new)} new"
    )

    # Compute suggested version names for both existing-name and structure-match compounds.
    # Priority: preserve submitted structure-match names as keys, since frontend lookups
    # are keyed by submitted compound names.
    version_targets: List[str] = []
    seen_targets = set()
    for m in structure_matches:
        name = m.compound_name
        normalized = _normalize_name(name)
        if normalized and normalized not in seen_targets:
            seen_targets.add(normalized)
            version_targets.append(name)
    for name in existing:
        normalized = _normalize_name(name)
        if normalized and normalized not in seen_targets:
            seen_targets.add(normalized)
            version_targets.append(name)
    suggested_versions = get_next_version_names_bulk(db, version_targets) if version_targets else {}

    return CheckDuplicatesResponse(
        existing=existing,
        processing=processing,
        new=new,
        structure_matches=structure_matches,
        internal_duplicates=internal_duplicates,
        suggested_versions=suggested_versions,
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
    # Always use the validated header session_id (body session_id is ignored for security)

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
    skipped_internal_duplicates = []  # Compounds skipped because they duplicate another row in this batch
    failed_compounds = []  # Compounds that failed during job creation
    seen_structure_key_to_name: Dict[str, str] = {}  # In-file structure dedupe guard
    seen_name_to_name: Dict[str, str] = {}  # In-file name dedupe guard

    # Create all jobs in SQLite (status: PENDING)
    # Scheduler will pick them up and process 2 at a time
    jobs = []
    for compound in request.compounds:
        compound_name = compound.compound_name

        # Skip if currently processing
        if compound_name in skipped_processing:
            continue

        try:
            # Get per-compound duplicate action (from compound data or duplicate_decisions dict)
            compound_action = getattr(compound, 'duplicate_action', None) or duplicate_decisions.get(compound_name)

            # If action is 'skip', don't process
            if compound_action == 'skip':
                skipped_existing.append(compound_name)
                continue

            # Skip duplicates inside the submitted batch itself.
            # We keep the first occurrence and skip subsequent rows with the same name
            # or same structure.
            row_inchikey = generate_inchikey(compound.smiles) if compound.smiles else None
            row_structure_key = _inchikey_structure_key(row_inchikey)
            normalized_compound_name = (compound_name or "").strip().lower()

            if normalized_compound_name:
                first_seen_name = seen_name_to_name.get(normalized_compound_name)
                if first_seen_name:
                    skipped_internal_duplicates.append(compound_name)
                    logger.info(
                        f"Batch internal duplicate skipped by name: "
                        f"'{compound_name}' duplicates '{first_seen_name}'"
                    )
                    continue
                seen_name_to_name[normalized_compound_name] = compound_name

            if row_structure_key:
                first_seen_name = seen_structure_key_to_name.get(row_structure_key)
                if first_seen_name:
                    skipped_internal_duplicates.append(compound_name)
                    logger.info(
                        f"Batch internal duplicate skipped by structure: "
                        f"'{compound_name}' duplicates '{first_seen_name}'"
                    )
                    continue
                seen_structure_key_to_name[row_structure_key] = compound_name

            # Defer deletion until job COMPLETES (not just creation) to prevent data loss
            replace_entry_id = None
            inherit_children_from = None

            # If action is 'replace', store reference for deferred deletion on job completion
            if compound_action == 'replace':
                # Find existing compound to replace by InChIKey (precise) - avoids wrong-duplicate risk
                existing = None
                compound_inchikey = generate_inchikey(compound.smiles) if compound.smiles else None
                if compound_inchikey:
                    # Prefer canonical (non-duplicate) compound over arbitrary match
                    candidates = db.query(Compound).filter(Compound.inchikey == compound_inchikey).all()
                    existing = next(
                        (c for c in candidates if not c.is_duplicate),
                        candidates[0] if candidates else None
                    )
                if existing:
                    replace_entry_id = existing.entry_id

                    # Check if this main compound has children that need to be inherited
                    if not existing.is_duplicate:
                        children = db.query(Compound).filter(
                            Compound.duplicate_of == existing.entry_id
                        ).all()
                        if children:
                            inherit_children_from = existing.entry_id

                    logger.info(
                        f"Batch replace: deletion of '{compound_name}' (entry_id={replace_entry_id}) "
                        f"deferred until replacement job completes"
                    )

            # Build job params - include duplicate/replace metadata
            job_params = compound.model_dump(exclude={"session_id", "duplicate_action", "original_compound_name"})
            if inherit_children_from:
                job_params["inherit_children_from"] = inherit_children_from
            if replace_entry_id:
                job_params["replace_entry_id"] = replace_entry_id
                # Inherit duplicate metadata so replacement retains duplicate status
                if existing and existing.is_duplicate:
                    job_params["is_duplicate"] = True
                    job_params["duplicate_of"] = existing.duplicate_of
            if compound_action == 'duplicate':
                # Find existing compound to reference by InChIKey (precise)
                dup_inchikey = generate_inchikey(compound.smiles) if compound.smiles else None
                dup_existing = None
                if dup_inchikey:
                    dup_existing = db.query(Compound).filter(Compound.inchikey == dup_inchikey).first()

                # Fallback when InChIKey lookup fails/unavailable: try exact name match
                if not dup_existing:
                    parent_lookup_name = (
                        getattr(compound, "original_compound_name", None)
                        or compound_name
                    )
                    parent_lookup_name = (parent_lookup_name or "").strip()
                    name_candidates = (
                        db.query(Compound)
                        .filter(
                            func.lower(func.trim(Compound.compound_name))
                            == parent_lookup_name.lower()
                        )
                        .order_by(Compound.processed_at.desc())
                        .all()
                    )
                    dup_existing = next(
                        (c for c in name_candidates if not c.is_duplicate),
                        name_candidates[0] if name_candidates else None
                    )

                # Only mark as duplicate when a valid parent entry_id exists.
                # Otherwise, process as a new compound to avoid orphan duplicate records.
                if dup_existing and dup_existing.entry_id:
                    submitted_threshold = compound.similarity_threshold or 90
                    submitted_at = _normalize_activity_types(compound.activity_types)
                    config_match = _compute_config_match(dup_existing, submitted_threshold, submitted_at)

                    # Keep batch behavior aligned with single-compound duplicate resolution:
                    # identical config duplicates are not meaningful and should be skipped.
                    if config_match == "identical":
                        skipped_existing.append(compound_name)
                        logger.info(
                            f"Batch duplicate requested for '{compound_name}' with identical config; "
                            f"skipping instead of creating duplicate"
                        )
                        continue

                    job_params["is_duplicate"] = True
                    job_params["duplicate_of"] = dup_existing.entry_id
                    marked_duplicate.append(compound_name)
                else:
                    logger.warning(
                        f"Batch duplicate requested for '{compound_name}' but no parent found; "
                        f"processing as new compound"
                    )

            # Create job FIRST - if this fails, no data is lost
            job = job_service.create_job(
                db,
                JobType.BATCH,
                job_params,
                session_id=session_id,
                batch_id=batch_id,
            )
            jobs.append(_job_to_response(job))

            # Deletion is deferred to job completion via replace_entry_id in job_params
            # If the job fails, the old compound remains intact (no data loss)
            if replace_entry_id:
                replaced.append(compound_name)

        except Exception as e:
            # Track per-compound failures instead of failing entire batch
            # Log full exception for debugging, but expose only generic message to client
            logger.error(f"Failed to create job for compound '{compound_name}': {e}", exc_info=True)
            db.rollback()  # Rollback any partial transaction
            # Security: Don't expose raw exception details to clients (could leak DB/path info)
            failed_compounds.append({
                "compound_name": compound_name,
                "error": "Failed to create job. Please check compound data and try again."
            })

    # Trigger scheduler to start processing (if not already running)
    if jobs:
        job_scheduler.trigger()

    total_skipped = len(skipped_existing) + len(skipped_processing) + len(skipped_internal_duplicates)
    total_failed = len(failed_compounds)
    logger.info(
        f"Batch {batch_id}: {len(jobs)} jobs queued, "
        f"{len(replaced)} replaced, {len(marked_duplicate)} as duplicates, "
        f"{total_skipped} skipped "
        f"({len(skipped_existing)} existing, {len(skipped_processing)} processing, "
        f"{len(skipped_internal_duplicates)} internal duplicates), "
        f"{total_failed} failed (session={truncate_session_id(session_id)})"
    )
    if failed_compounds:
        logger.warning(f"Batch {batch_id} had {total_failed} compound failures: {[f['compound_name'] for f in failed_compounds]}")

    return BatchResponse(
        batch_id=batch_id,
        jobs=jobs,
        skipped_existing=skipped_existing,
        skipped_processing=skipped_processing,
        skipped_internal_duplicates=skipped_internal_duplicates,
        replaced=replaced,
        total_submitted=len(jobs),
        total_skipped=total_skipped,
        failed_compounds=failed_compounds,  # Include failed compounds in response
    )


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
    """
    List jobs for the current session with optional status filter and pagination.

    Users only see their own jobs (filtered by X-Session-ID header).
    """
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
    """
    Get active (pending/processing) jobs for the current session.

    Used by the frontend sidebar to display job progress.
    Users only see their own jobs (filtered by X-Session-ID header).

    Headers:
        X-Session-ID: Session ID for filtering jobs (required, validated UUID)
    """
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
    """
    Get summary statistics for a batch of jobs.

    Requires ownership of the batch (same session that created it).
    Returns overall progress and status counts.
    """
    # Verify ownership by checking first job in batch
    first_job = db.query(Job).filter(Job.batch_id == batch_id).first()

    if not first_job:
        raise HTTPException(status_code=404, detail="Batch not found")

    if first_job.session_id and first_job.session_id != session_id:
        logger.warning(
            f"Unauthorized batch access attempt: session {truncate_session_id(session_id)} "
            f"tried to access batch {batch_id}"
        )
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to access this batch"
        )

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
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Get job status",
)
async def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """
    Get the current status of a job.

    Poll this endpoint (1s interval) to track progress.
    Requires ownership of the job (same session that created it).
    """
    # Verify ownership (returns job or raises 403/404)
    job = _verify_job_ownership(db, job_id, session_id)

    return _job_to_response(job)


@router.get(
    "/{job_id}/detail",
    responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    summary="Get detailed job info",
)
async def get_job_detail(
    job_id: str,
    db: Session = Depends(get_db),
    session_id: str = Depends(validate_session_id),
):
    """
    Get detailed job information including parsed input parameters.

    Requires ownership of the job (same session that created it).
    """
    # Verify ownership first
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
            logger.warning(
                f"No compound found by entry_id for job {job_id} "
                f"(compound_name={compound_name}, entry_id={entry_id}). "
                f"Skipping compound deletion to avoid deleting wrong duplicate."
            )

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
