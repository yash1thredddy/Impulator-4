"""
Job service for CRUD operations and state management.
Handles job creation, progress updates, and completion tracking.
"""
import json
import re
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Callable, Union

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, InvalidRequestError

from backend.models.database import Compound, Job, JobStatus, JobType
from backend.models.schemas import (
    InputParams,
    ResultSummary,
    JobResponse,
    DuplicateFoundResponse,
    ExistingCompoundInfo,
    ExistingCompoundAtThreshold,
    ThresholdAvailability,
    CompoundAvailability,
    SkipResponse,
    CheckDuplicatesResponse,
    DuplicateMatch,
    InternalDuplicateMatch,
    BatchResponse,
    FailedCompound,
    CheckAvailabilityBatchResponse,
    DuplicateAction,
    ResolveDuplicateRequest,
    JobCreate,
    BatchJobCreate,
    CheckDuplicatesRequest,
    CheckAvailabilityBatchRequest,
    DeleteResponse,
)
from backend.core.azure_sync import sync_db_to_azure, delete_result_from_azure_by_entry_id
from backend.config import settings
from backend.core.metrics import metrics
from backend.repositories import job_repo, compound_repo, _db_write_lock

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
    # Strip any existing version suffix from base_name to get true base
    version_pattern = re.compile(r'^(.+?)(_v(\d+))?$')
    match = version_pattern.match(base_name)
    if match:
        true_base = match.group(1).strip()
    else:
        true_base = base_name.strip()

    # Query all compound names that start with the base name
    existing_names = compound_repo.find_names_by_prefix(db, true_base)

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

    # Fetch all matching names using repo calls per prefix
    existing_names: set = set()
    for base_norm in normalized_base_to_sample.keys():
        existing_names.update(compound_repo.find_names_by_prefix(db, base_norm))

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


# Valid status transitions for job state machine
VALID_TRANSITIONS = {
    JobStatus.PENDING: {JobStatus.PROCESSING, JobStatus.CANCELLED},
    JobStatus.PROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.SYNC_PENDING},
    JobStatus.SYNC_PENDING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),  # Terminal state
    JobStatus.FAILED: set(),     # Terminal state
    JobStatus.CANCELLED: set(),  # Terminal state
}


def _safe_json_loads(json_str: Optional[str], default: Any = None) -> Any:
    """Safely parse JSON string, returning default on failure."""
    if not json_str:
        return default
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default


def generate_inchikey(smiles: str) -> Optional[str]:
    """
    Generate InChIKey from SMILES (100% deterministic).

    InChIKey is a 27-character hash that uniquely identifies a chemical structure.
    Same structure always produces the same InChIKey regardless of SMILES notation.

    Args:
        smiles: SMILES string representing the molecule

    Returns:
        27-character InChIKey string, or None if generation fails
    """
    if not smiles or not smiles.strip():
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            inchikey = MolToInchiKey(mol)
            logger.debug(f"Generated InChIKey: {inchikey} for SMILES: {smiles[:50]}...")
            return inchikey
        else:
            logger.warning(f"Could not parse SMILES: {smiles[:50]}...")
            return None
    except ImportError:
        logger.error("RDKit not available - cannot generate InChIKey")
        return None
    except Exception as e:
        logger.warning(f"InChIKey generation failed: {e}")
        return None


def generate_canonical_smiles(smiles: str) -> Optional[str]:
    """
    Generate canonical SMILES from input SMILES.

    Canonical SMILES is a standardized representation that is the same
    regardless of how the original SMILES was written.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES string, or None if conversion fails
    """
    if not smiles or not smiles.strip():
        return None

    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except ImportError:
        logger.error("RDKit not available - cannot generate canonical SMILES")
        return None
    except Exception as e:
        logger.warning(f"Canonical SMILES generation failed: {e}")
        return None


# ============================================================================
# Helper functions moved from jobs.py (business logic, not HTTP concerns)
# ============================================================================


def _normalize_inchikey_input(inchikey: Optional[str]) -> Optional[str]:
    """Normalize optional InChIKey input from request payload."""
    if not inchikey:
        return None
    normalized = inchikey.strip().upper()
    if not normalized or normalized in {"NAN", "NONE"}:
        return None
    return normalized


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


def _job_to_response(job) -> JobResponse:
    """Convert Job model to JobResponse, extracting compound info from input_params."""
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

    # Extract compound_name and smiles from input_params using typed model
    if job.input_params:
        try:
            params = InputParams.model_validate_json(job.input_params)
            data["compound_name"] = params.compound_name or "Unknown"
            data["smiles"] = params.smiles or ""
        except Exception:
            data["compound_name"] = "Unknown"
            data["smiles"] = ""

    return JobResponse(**data)


def _build_config_diff(
    existing_compound: Compound,
    submitted_threshold: int,
    submitted_activity_types: str,
) -> Optional[Dict]:
    """Build config diff dict if configs differ, else None."""
    config_match = _compute_config_match(existing_compound, submitted_threshold, submitted_activity_types)
    if config_match == "identical":
        return None
    return {
        "similarity_threshold": {
            "existing": existing_compound.similarity_threshold or 90,
            "submitted": submitted_threshold,
        },
        "activity_types": {
            "existing": _normalize_activity_types_str(existing_compound.activity_types),
            "submitted": submitted_activity_types,
        },
    }


def _build_existing_at_threshold(
    existing_compound: Compound,
    submitted_threshold: int,
    submitted_activity_types: str,
) -> ExistingCompoundAtThreshold:
    """Build ExistingCompoundAtThreshold from a Compound ORM object."""
    config_match = _compute_config_match(existing_compound, submitted_threshold, submitted_activity_types)
    config_diff = _build_config_diff(existing_compound, submitted_threshold, submitted_activity_types)

    return ExistingCompoundAtThreshold(
        entry_id=existing_compound.entry_id,
        compound_name=existing_compound.compound_name,
        similarity_threshold=existing_compound.similarity_threshold,
        activity_types=_normalize_activity_types_str(existing_compound.activity_types),
        config_match=config_match,
        config_diff=config_diff,
        imp_score=existing_compound.imp_score,
        processed_at=existing_compound.processed_at.isoformat() if existing_compound.processed_at else None,
        author_name=existing_compound.author_name,
    )


def _check_single_availability(
    smiles: str,
    compound_name: str,
    similarity_threshold: int,
    activity_types: Optional[List[str]],
    db: Session,
) -> CompoundAvailability:
    """Check ChEMBL data availability for a single compound.

    Probes all thresholds and finds existing compounds by InChIKey.
    """
    from backend.modules.api_client import probe_all_thresholds

    submitted_at = _normalize_activity_types(activity_types)

    # Probe ALL thresholds (includes requested threshold)
    thresholds = probe_all_thresholds(smiles, similarity_threshold)

    threshold_items = [
        ThresholdAvailability(threshold=t["threshold"], count=t["count"])
        for t in thresholds
    ]

    # Find count at requested threshold
    count_at_threshold = 0
    for t in thresholds:
        if t["threshold"] == similarity_threshold:
            count_at_threshold = t["count"]
            break

    available = count_at_threshold > 0
    has_any_data = any(t["count"] > 0 for t in thresholds)

    # Find existing compounds by InChIKey
    existing_compounds: List[ExistingCompoundAtThreshold] = []
    inchikey = generate_inchikey(smiles)
    structure_key = _inchikey_structure_key(inchikey)
    if structure_key:
        existing = compound_repo.find_by_inchikey_like(db, structure_key)
        for comp in existing:
            existing_compounds.append(
                _build_existing_at_threshold(comp, similarity_threshold, submitted_at)
            )

    return CompoundAvailability(
        compound_name=compound_name,
        smiles=smiles,
        available=available,
        count_at_threshold=count_at_threshold,
        thresholds=threshold_items,
        existing_compounds=existing_compounds,
        has_any_data=has_any_data,
    )


# Batch size limits
MAX_BATCH_SIZE = 1000  # Maximum compounds per batch submission


class JobService:
    """Service for job management operations."""

    def create_job(
        self,
        db: Session,
        job_type: JobType,
        input_params: Dict[str, Any],
        session_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Job:
        """
        Create a new job record.

        Args:
            db: Database session
            job_type: Type of job (single/batch)
            input_params: Input parameters for the job
            session_id: Session ID for user isolation
            batch_id: Batch ID for grouping related jobs
            idempotency_key: Optional key for safe retries (unique per session)

        Returns:
            Created Job object
        """
        job = Job(
            id=str(uuid.uuid4()),
            job_type=job_type,
            status=JobStatus.PENDING,
            session_id=session_id,
            batch_id=batch_id,
            idempotency_key=idempotency_key,
            input_params=json.dumps(input_params),
            progress=0.0,
            current_step="Queued",
            created_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.commit()
        try:
            db.refresh(job)
        except InvalidRequestError:
            # Under concurrent SQLite access, refresh can fail even after
            # a successful commit. The job is already persisted and all
            # fields are set from Python-side defaults, so this is safe.
            pass
        metrics.increment('jobs_created')
        logger.info(f"Created job {job.id} ({job_type.value}) session={session_id} batch={batch_id}")
        return job

    def generate_batch_id(self) -> str:
        """Generate a new batch ID for grouping jobs."""
        return str(uuid.uuid4())

    def get_job(self, db: Session, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return job_repo.get_by_job_id(db, job_id)

    def _get_job_for_update(self, db: Session, job_id: str) -> Optional[Job]:
        """Get a job by ID with thread-safe locking.

        Note: SQLite doesn't support FOR UPDATE, so we use application-level
        locking via _db_write_lock. All callers must acquire the lock before
        calling this method and hold it through commit.
        """
        return job_repo.get_by_job_id(db, job_id)

    def _execute_with_lock(self, db: Session, operation: Callable) -> Any:
        """Execute a database write operation with thread-safe locking.

        Ensures serialized writes to SQLite to prevent corruption.

        Args:
            db: Database session
            operation: Callable that performs the database operation

        Returns:
            Result of the operation
        """
        with _db_write_lock:
            result = operation()
            db.commit()
            return result

    def get_job_with_params(self, db: Session, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job with parsed input parameters."""
        job = self.get_job(db, job_id)
        if not job:
            return None

        result = job.to_dict()
        if job.input_params:
            try:
                params = InputParams.model_validate_json(job.input_params)
                result["input_params"] = params.model_dump()
            except Exception:
                raw = job.input_params
                result["input_params"] = _safe_json_loads(raw, {})
        if job.result_summary:
            try:
                summary = ResultSummary.model_validate_json(job.result_summary)
                result["result_summary"] = summary.model_dump()
            except Exception:
                raw = job.result_summary
                result["result_summary"] = _safe_json_loads(raw, {})
        return result

    def list_jobs(
        self,
        db: Session,
        statuses: Optional[List[JobStatus]] = None,
        page: int = 1,
        page_size: int = 20,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        List jobs with optional status filter, session filter, and pagination.

        Args:
            db: Database session
            statuses: Optional list of statuses to filter by
            page: Page number (1-indexed)
            page_size: Number of items per page
            session_id: Session ID to filter by (required for user isolation)

        Returns:
            Dict with items, total, page info
        """
        offset = (page - 1) * page_size
        jobs, total = job_repo.get_jobs_paginated(
            db,
            session_id=session_id,
            status_filter=statuses,
            offset=offset,
            limit=page_size,
        )
        pages = (total + page_size - 1) // page_size

        return {
            "items": jobs,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }

    def get_active_jobs(
        self,
        db: Session,
        session_id: Optional[str] = None,
        include_recent_minutes: int = 2,
    ) -> List[Dict]:
        """
        Get active (pending/processing) jobs and recently completed jobs for sidebar.

        Includes recently completed/failed jobs so users can see "View" button
        before they disappear from the list.

        Args:
            db: Database session
            session_id: Session ID to filter by (None returns all - for admin)
            include_recent_minutes: Include completed jobs from last N minutes

        Returns:
            List of job dicts with progress info
        """
        from datetime import timedelta

        # Get pending/processing jobs
        active_jobs = job_repo.get_active_jobs(db, session_id)

        # Recently completed jobs (auto-dismiss after N minutes)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=include_recent_minutes)
        completed_jobs = job_repo.get_completed_jobs_since(db, recent_cutoff, session_id)

        # Failed jobs for session (auto-dismiss after 20 minutes)
        failed_cutoff = datetime.now(timezone.utc) - timedelta(minutes=20)
        failed_jobs = job_repo.get_failed_jobs_since(db, failed_cutoff, session_id)

        # Combine and sort: completed jobs first, then by created_at (newest first within each group)
        all_jobs = active_jobs + completed_jobs + failed_jobs
        # Sort key: (0 if completed/failed, 1 otherwise), then by created_at descending
        def _sort_ts(dt):
            """Return a UTC timestamp suitable for descending sort."""
            if dt is None:
                return datetime.min.replace(tzinfo=timezone.utc).timestamp()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()

        all_jobs.sort(
            key=lambda j: (
                0 if j.status in [JobStatus.COMPLETED, JobStatus.FAILED] else 1,
                -_sort_ts(j.created_at),
            )
        )

        result = []
        for job in all_jobs:
            item = {
                "id": job.id,
                "status": job.status.value,
                "progress": job.progress,
                "current_step": job.current_step,
                "batch_id": job.batch_id,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }
            # Extract compound name from input params using typed model
            compound_name = "Unknown"
            params_dict = {}
            if job.input_params:
                try:
                    params = InputParams.model_validate_json(job.input_params)
                    compound_name = params.compound_name or "Unknown"
                    params_dict = params.model_dump()
                except Exception:
                    raw = job.input_params
                    params_dict = _safe_json_loads(raw, {})
                    compound_name = params_dict.get("compound_name", "Unknown")
            item["compound_name"] = compound_name

            # For completed jobs, include entry_id and storage_path for UUID-based storage lookup
            entry_id = None
            storage_path = None
            if job.status == JobStatus.COMPLETED:
                # First try result_summary
                if job.result_summary:
                    try:
                        summary = ResultSummary.model_validate_json(job.result_summary)
                        entry_id = summary.entry_id
                        storage_path = summary.storage_path
                    except Exception:
                        raw = job.result_summary
                        result_data = _safe_json_loads(raw, {})
                        entry_id = result_data.get("entry_id")
                        storage_path = result_data.get("storage_path")

                # Fallback: derive from the job result path (UUID-based filename)
                if not entry_id and job.result_path:
                    from pathlib import Path
                    candidate = Path(job.result_path).stem
                    try:
                        entry_id = str(uuid.UUID(candidate))
                    except (ValueError, TypeError, AttributeError):
                        entry_id = None

                # Build storage_path deterministically from entry_id
                if entry_id and not storage_path:
                    try:
                        from backend.core.storage_paths import get_storage_path_from_entry_id
                        storage_path = get_storage_path_from_entry_id(entry_id)
                    except Exception as e:
                        logger.debug(f"Could not derive storage_path from entry_id {entry_id}: {e}")

            item["entry_id"] = entry_id
            item["storage_path"] = storage_path
            item["error_message"] = job.error_message

            # For failed jobs, include cascade similarity results and input params for resubmission
            if job.status == JobStatus.FAILED and job.result_summary:
                try:
                    failed_summary = ResultSummary.model_validate_json(job.result_summary)
                    cascade = getattr(failed_summary, 'cascade_results', None)
                    if cascade:
                        item["cascade_results"] = cascade
                        item["input_params"] = params_dict
                except Exception:
                    raw = job.result_summary
                    failed_data = _safe_json_loads(raw, {})
                    cascade = failed_data.get("cascade_results")
                    if cascade:
                        item["cascade_results"] = cascade
                        item["input_params"] = params_dict

            result.append(item)

        return result

    def get_batch_summary(self, db: Session, batch_id: str) -> Dict:
        """
        Get summary statistics for a batch of jobs using single aggregated query.

        Args:
            db: Database session
            batch_id: Batch ID to summarize

        Returns:
            Dict with batch statistics
        """
        return job_repo.get_batch_summary(db, batch_id)

    def cancel_batch(self, db: Session, batch_id: str) -> int:
        """
        Cancel all pending/processing jobs in a batch.

        Args:
            db: Database session
            batch_id: Batch ID to cancel

        Returns:
            Number of jobs cancelled
        """
        with _db_write_lock:
            cancelled_count = job_repo.cancel_batch_jobs(db, batch_id)
            if cancelled_count > 0:
                db.commit()
                logger.info(f"Cancelled {cancelled_count} jobs in batch {batch_id}")
            return cancelled_count

    def check_existing_compounds(
        self,
        db: Session,
        compound_names: List[str],
    ) -> Dict[str, bool]:
        """
        Check which compounds already have completed results.

        Checks the SQLite Compound table (database is the source of truth).
        UUID-based storage paths are used, so Azure lookup by name is not supported.
        Use InChIKey for accurate duplicate detection instead of compound names.

        Args:
            db: Database session
            compound_names: List of compound names to check

        Returns:
            Dict mapping compound_name -> exists (True if already processed)
        """
        def _normalize_name(name: str) -> str:
            return (name or "").strip().lower()

        normalized_input = {_normalize_name(name) for name in compound_names if _normalize_name(name)}
        if not normalized_input:
            return {name: False for name in compound_names}

        # Case-insensitive batch query so CSV casing differences still match
        local_existing = compound_repo.find_existing_names(db, list(normalized_input))

        result = {}
        for name in compound_names:
            result[name] = _normalize_name(name) in local_existing

        return result

    def check_pending_compounds(
        self,
        db: Session,
        compound_names: List[str],
    ) -> Dict[str, str]:
        """
        Check which compounds are currently being processed.

        Fetches all pending/processing jobs once and filters in Python
        for accurate JSON matching (avoids SQL LIKE injection issues).

        Args:
            db: Database session
            compound_names: List of compound names to check

        Returns:
            Dict mapping compound_name -> job_id (if pending/processing)
        """
        if not compound_names:
            return {}

        def _normalize_name(name: str) -> str:
            return (name or "").strip().lower()

        # Track normalized input names while preserving the original key users submitted.
        normalized_to_original: Dict[str, str] = {}
        for name in compound_names:
            normalized = _normalize_name(name)
            if normalized and normalized not in normalized_to_original:
                normalized_to_original[normalized] = name

        names_to_check = set(normalized_to_original.keys())
        result = {}

        # Fetch all pending/processing jobs in one query
        pending_jobs = job_repo.get_active_jobs(db)

        # Parse JSON and match compound names
        for job in pending_jobs:
            if not job.input_params:
                continue
            try:
                params = InputParams.model_validate_json(job.input_params)
                job_compound_name = params.compound_name
                normalized_job_name = _normalize_name(job_compound_name) if job_compound_name else ""
                if normalized_job_name and normalized_job_name in names_to_check:
                    original_name = normalized_to_original[normalized_job_name]
                    result[original_name] = job.id
                    # Remove from set to avoid duplicate matches
                    names_to_check.discard(normalized_job_name)
                    # Early exit if all found
                    if not names_to_check:
                        break
            except Exception:
                continue

        return result

    def update_progress(
        self,
        db: Session,
        job_id: str,
        progress: float,
        current_step: str,
        status: Optional[JobStatus] = None,
    ) -> Optional[Job]:
        """
        Update job progress with thread-safe locking and status validation.

        Args:
            db: Database session
            job_id: Job ID
            progress: Progress percentage (0-100)
            current_step: Description of current step
            status: Optional status update
        """
        with _db_write_lock:
            job = self._get_job_for_update(db, job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for progress update")
                return None

            # Validate status transition if status is being changed
            if status and status != job.status:
                valid_next = VALID_TRANSITIONS.get(job.status, set())
                if status not in valid_next:
                    logger.warning(
                        f"Invalid status transition {job.status.value} -> {status.value} "
                        f"for job {job_id}"
                    )
                    return None

            job.progress = progress
            job.current_step = current_step

            if status:
                job.status = status
                if status == JobStatus.PROCESSING and not job.started_at:
                    job.started_at = datetime.now(timezone.utc)

            db.commit()
            db.refresh(job)
            return job

    def complete_job(
        self,
        db: Session,
        job_id: str,
        result_path: str,
        result_summary: Dict[str, Any],
    ) -> Optional[Job]:
        """
        Mark job as completed, update Compound table, and trigger Azure sync.

        Uses thread-safe locking to prevent race conditions.

        Args:
            db: Database session
            job_id: Job ID
            result_path: Path to result file
            result_summary: Summary statistics
        """
        with _db_write_lock:
            job = self._get_job_for_update(db, job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for completion")
                return None

            # Guard: don't resurrect cancelled/failed jobs to COMPLETED
            if job.status in (JobStatus.CANCELLED, JobStatus.FAILED):
                logger.warning(f"Job {job_id} is {job.status.value}, not completing")
                return None

            # Preserve result metadata on both success and failure paths
            # (ZIP exists regardless -- allows re-processing on failure)
            job.result_path = result_path
            job.result_summary = json.dumps(result_summary)

            # Extract duplicate metadata from job input_params using typed model
            job_params = InputParams()
            if job.input_params:
                try:
                    job_params = InputParams.model_validate_json(job.input_params)
                except Exception:
                    pass

            # Try to update compound entry FIRST -- if this fails, job must be FAILED
            try:
                self._update_compound_entry(
                    db,
                    result_summary,
                    result_path,
                    is_duplicate=job_params.is_duplicate,
                    duplicate_of=job_params.duplicate_of,
                    inherit_children_from=job_params.inherit_children_from,
                    replace_entry_id=job_params.replace_entry_id,
                    session_id=job.session_id,
                    job_id=job_id,
                )
            except Exception as e:
                # Compound entry failed -- mark job as FAILED, not COMPLETED
                # ZIP result still exists -- user can re-submit
                logger.error(f"Job {job_id}: compound entry update failed, marking FAILED: {e}")
                job.status = JobStatus.FAILED
                job.error_message = f"Compound entry could not be saved: {e}"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                db.refresh(job)
                metrics.increment('jobs_failed')
                return job

            # Re-apply result metadata in case _update_compound_entry rolled back
            # the session (e.g. IntegrityError on duplicate structure key).
            # db.rollback() reverts ALL dirty ORM state, including result_path
            # and result_summary set above.
            job.result_path = result_path
            job.result_summary = json.dumps(result_summary)

            # Compound entry succeeded (or was silently skipped) -- mark COMPLETED
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.current_step = "Completed"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(job)
            metrics.increment('jobs_completed')

        # All network I/O happens OUTSIDE the lock to avoid blocking other threads

        # Clean up replaced compound's files (Azure + local)
        replace_entry_id = job_params.replace_entry_id
        if replace_entry_id:
            try:
                from backend.core.azure_sync import delete_result_from_azure_by_entry_id
                delete_result_from_azure_by_entry_id(replace_entry_id)
            except Exception as e:
                logger.warning(f"Failed to delete Azure result for replaced compound: {e}")

            try:
                from backend.config import settings
                prefix = replace_entry_id[:2].lower()
                local_zip = settings.RESULTS_DIR / prefix / f"{replace_entry_id}.zip"
                if local_zip.exists():
                    local_zip.unlink()
                    logger.debug(f"Deleted local result for replaced compound: {local_zip}")
            except Exception as e:
                logger.warning(f"Failed to delete local result for replaced compound: {e}")

        # Immediate sync to Azure (outside lock - this is network I/O)
        sync_db_to_azure()

        logger.info(f"Job {job_id} completed successfully")
        return job

    def _update_compound_entry(
        self,
        db: Session,
        result_summary: Dict[str, Any],
        result_path: str,
        is_duplicate: bool = False,
        duplicate_of: Optional[str] = None,
        inherit_children_from: Optional[str] = None,
        replace_entry_id: Optional[str] = None,
        session_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> None:
        """
        Create or update Compound entry in local database.

        This ensures the Compound table stays in sync with completed jobs,
        without waiting for the next startup sync from Azure.

        Args:
            db: Database session
            result_summary: Summary from job processing
            result_path: Path to result file
            is_duplicate: Whether this compound is a tagged duplicate
            duplicate_of: Entry ID of the original compound (if duplicate)
            inherit_children_from: Entry ID of old compound whose children should be re-pointed
            replace_entry_id: Entry ID of old compound to delete after successful replacement
            session_id: Session ID for audit trail
            job_id: Job ID for audit trail
        """
        from backend.models.database import Compound, DeletedCompound
        import uuid

        # Validate result_summary before accessing
        if not result_summary or not isinstance(result_summary, dict):
            logger.warning("Invalid result_summary (None or not dict), skipping Compound update")
            return

        compound_name = result_summary.get('compound_name')
        if not compound_name:
            logger.warning("No compound_name in result_summary, skipping Compound update")
            return

        smiles = result_summary.get('smiles') or result_summary.get('query_smiles')

        # Generate InChIKey and canonical SMILES for new/updated compounds
        inchikey = generate_inchikey(smiles) if smiles else None
        canonical_smiles = generate_canonical_smiles(smiles) if smiles else None

        # Compute inchikey_structure_key
        inchikey_structure_key = None
        if inchikey and '-' in inchikey:
            parts = inchikey.split('-')
            if len(parts) >= 2:
                inchikey_structure_key = f"{parts[0]}-{parts[1]}"

        # Use entry_id from result_summary if available (generated by compound_service)
        # Otherwise generate a new one
        entry_id_from_summary = result_summary.get('entry_id')

        # Use storage_path from result_summary if available (UUID-based path)
        # Otherwise fall back to result_path (legacy name-based path)
        storage_path = result_summary.get('storage_path') or result_path

        try:
            # Extract additional summary fields for home page display
            similarity_threshold = result_summary.get('similarity_threshold', 90)
            activity_types = result_summary.get('activity_types')
            qed = result_summary.get('qed', 0.0)
            num_outliers = result_summary.get('num_outliers', 0)
            similar_compounds = result_summary.get('total_similar', result_summary.get('similar_count', result_summary.get('total_compounds', 0)))

            # For duplicates, always create new entry (don't update existing)
            if is_duplicate:
                # Use entry_id from summary or generate new one
                entry_id = entry_id_from_summary or str(uuid.uuid4())
                compound = Compound(
                    entry_id=entry_id,
                    compound_name=compound_name,
                    smiles=smiles,
                    canonical_smiles=canonical_smiles,
                    inchikey=inchikey,
                    inchikey_structure_key=inchikey_structure_key,
                    chembl_id=result_summary.get('chembl_id', ''),
                    total_activities=result_summary.get('total_activities', 0),
                    imp_candidates=result_summary.get('imp_candidates', 0),
                    imp_score=result_summary.get('imp_score'),
                    similarity_threshold=similarity_threshold,
                    activity_types=activity_types,
                    qed=qed,
                    num_outliers=num_outliers,
                    similar_compounds=similar_compounds,
                    author_name=result_summary.get('author_name'),
                    storage_path=storage_path,
                    is_duplicate=True,
                    duplicate_of=duplicate_of,
                    processed_at=datetime.now(timezone.utc),
                )
                db.add(compound)
                logger.info(f"Created Compound entry (duplicate): {compound_name} -> {entry_id}")
                return

            # Check if compound exists, preferring InChIKey match over name
            # When replacing, ALWAYS create a fresh entry -- never update another
            # compound that happens to share the InChIKey or name. The old compound
            # (replace_entry_id) will be deleted after this new one is saved.
            existing = None
            if not replace_entry_id:
                if inchikey:
                    existing = compound_repo.find_by_inchikey(db, inchikey)
                    existing = existing[0] if existing else None

                # Only fall back to exact name match if:
                # 1. No InChIKey provided for the new compound, AND
                # 2. The existing record has no InChIKey (to avoid false matches)
                if not existing and not inchikey:
                    existing = compound_repo.find_by_name_no_inchikey(db, compound_name)

            if existing:
                # Update existing entry
                existing.smiles = smiles
                existing.canonical_smiles = canonical_smiles
                existing.inchikey = inchikey
                existing.inchikey_structure_key = inchikey_structure_key
                existing.chembl_id = result_summary.get('chembl_id', '')
                existing.total_activities = result_summary.get('total_activities', 0)
                existing.imp_candidates = result_summary.get('imp_candidates', 0)
                existing.imp_score = result_summary.get('imp_score')
                existing.similarity_threshold = similarity_threshold
                existing.activity_types = activity_types
                existing.qed = qed
                existing.num_outliers = num_outliers
                existing.similar_compounds = similar_compounds
                existing.author_name = result_summary.get('author_name')
                existing.storage_path = storage_path
                existing.processed_at = datetime.now(timezone.utc)  # Update timestamp
                # Update entry_id if missing (for older records) or use the new one
                if not existing.entry_id:
                    existing.entry_id = entry_id_from_summary or str(uuid.uuid4())
                logger.info(f"Updated Compound entry: {compound_name}")
            else:
                # Create new entry with UUID from summary or generate new
                entry_id = entry_id_from_summary or str(uuid.uuid4())
                compound = Compound(
                    entry_id=entry_id,
                    compound_name=compound_name,
                    smiles=smiles,
                    canonical_smiles=canonical_smiles,
                    inchikey=inchikey,
                    inchikey_structure_key=inchikey_structure_key,
                    chembl_id=result_summary.get('chembl_id', ''),
                    total_activities=result_summary.get('total_activities', 0),
                    imp_candidates=result_summary.get('imp_candidates', 0),
                    imp_score=result_summary.get('imp_score'),
                    similarity_threshold=similarity_threshold,
                    activity_types=activity_types,
                    qed=qed,
                    num_outliers=num_outliers,
                    similar_compounds=similar_compounds,
                    author_name=result_summary.get('author_name'),
                    storage_path=storage_path,
                    is_duplicate=is_duplicate,
                    duplicate_of=duplicate_of,
                    processed_at=datetime.now(timezone.utc),
                )
                db.add(compound)
                logger.info(f"Created Compound entry: {compound_name} -> {entry_id}")

            # Re-point orphaned children from a replaced compound to this new one
            if inherit_children_from and not is_duplicate:
                # Determine the entry_id of the newly created/updated compound
                if existing:
                    new_entry_id = existing.entry_id
                else:
                    new_entry_id = entry_id  # Set in the "else" branch above
                children = compound_repo.find_children(db, inherit_children_from)
                for child in children:
                    child.duplicate_of = new_entry_id
                    logger.info(
                        f"Re-pointed child '{child.compound_name}' ({child.entry_id}) "
                        f"duplicate_of: {inherit_children_from} -> {new_entry_id}"
                    )
                if children:
                    logger.info(f"Inherited {len(children)} children from {inherit_children_from}")

            # Delete old compound AFTER replacement succeeds (deferred from job creation)
            # This prevents data loss: if the job had failed, the old compound would remain
            # NOTE: Azure/local file cleanup is deferred to AFTER the lock is released
            # (see complete_job) to avoid blocking other threads on network I/O.
            if replace_entry_id:
                old_compound = compound_repo.get_by_entry_id(db, replace_entry_id)
                if old_compound:
                    old_name = old_compound.compound_name

                    # Promote children before deleting (prevents orphans)
                    if not old_compound.is_duplicate:
                        # Re-point any remaining children to the new compound
                        remaining_children = compound_repo.find_children(db, replace_entry_id)
                        new_id = existing.entry_id if existing else entry_id
                        for child in remaining_children:
                            child.duplicate_of = new_id
                            logger.info(
                                f"Re-pointed child '{child.compound_name}' from replaced "
                                f"compound to new entry {new_id}"
                            )

                    # Archive to deleted_compounds table before deletion (audit trail)
                    deleted_record = DeletedCompound(
                        original_id=old_compound.id,
                        entry_id=old_compound.entry_id,
                        compound_name=old_compound.compound_name,
                        chembl_id=old_compound.chembl_id,
                        smiles=old_compound.smiles,
                        inchikey=old_compound.inchikey,
                        author_name=old_compound.author_name,
                        is_duplicate=old_compound.is_duplicate,
                        duplicate_of=old_compound.duplicate_of,
                        activity_types=old_compound.activity_types,
                        storage_path=old_compound.storage_path,
                        deleted_by_session=session_id,
                        deleted_by_job_id=job_id,
                        deletion_reason="replaced",
                        original_processed_at=old_compound.processed_at,
                    )
                    db.add(deleted_record)

                    # Delete from database (DB operation - safe inside lock)
                    db.delete(old_compound)
                    logger.info(
                        f"Replacement complete: archived and deleted old compound '{old_name}' "
                        f"(entry_id={replace_entry_id}) after successful replacement"
                    )

        except IntegrityError as e:
            # Unique constraint on inchikey_structure_key -- concurrent duplicate submission
            # lost the race. Roll back and log; the other submission's compound will be kept.
            # NOTE: db.rollback() reverts ALL pending ORM changes in this session,
            # including job.result_path and job.result_summary set by complete_job.
            # We must signal the caller to re-apply them after rollback.
            db.rollback()
            logger.warning(
                f"Duplicate structure key race for {compound_name} "
                f"(structure_key={inchikey_structure_key}): {e}"
            )
            return
        except Exception as e:
            logger.error(f"Failed to update Compound entry for {compound_name}: {e}")
            raise  # Propagate to complete_job -- job must not be marked COMPLETED

    def fail_job(
        self,
        db: Session,
        job_id: str,
        error_message: str,
        cascade_results: list = None,
    ) -> Optional[Job]:
        """
        Mark job as failed with thread-safe locking.

        Args:
            db: Database session
            job_id: Job ID
            error_message: Error description
            cascade_results: Optional list of {threshold, count} dicts from
                cascade similarity probing (stored in result_summary JSON).
        """
        with _db_write_lock:
            job = self._get_job_for_update(db, job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for failure")
                return None

            job.status = JobStatus.FAILED
            job.current_step = "Failed"
            job.error_message = error_message
            job.completed_at = datetime.now(timezone.utc)

            if cascade_results is not None:
                job.result_summary = json.dumps({"cascade_results": cascade_results})

            db.commit()
            db.refresh(job)
            metrics.increment('jobs_failed')

        # Sync to Azure (outside lock - this is network I/O)
        sync_db_to_azure()

        logger.error(f"Job {job_id} failed: {error_message}")
        return job

    def cancel_job(
        self,
        db: Session,
        job_id: str,
    ) -> Optional[Job]:
        """
        Mark job as cancelled with thread-safe locking.

        Args:
            db: Database session
            job_id: Job ID
        """
        with _db_write_lock:
            job = self._get_job_for_update(db, job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for cancellation")
                return None

            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                logger.warning(f"Job {job_id} cannot be cancelled (status: {job.status})")
                return job

            job.status = JobStatus.CANCELLED
            job.current_step = "Cancelled"
            job.completed_at = datetime.now(timezone.utc)

            db.commit()
            db.refresh(job)
            metrics.increment('jobs_cancelled')

        logger.info(f"Job {job_id} cancelled")
        return job

    def delete_job(self, db: Session, job_id: str) -> bool:
        """
        Delete a job record.

        Args:
            db: Database session
            job_id: Job ID

        Returns:
            True if deleted, False if not found
        """
        result = job_repo.delete_job(db, job_id)
        if result:
            db.commit()
            logger.info(f"Job {job_id} deleted")
        return result

    def delete_job_with_cleanup(
        self, db: Session, job_id: str, session_id: str
    ) -> DeleteResponse:
        """Delete a job and clean up all associated resources.

        Orchestrates: JSON parsing, Azure deletion, local file cleanup,
        compound archiving, compound deletion, and job deletion.

        Args:
            db: Database session
            job_id: Job ID to delete
            session_id: Session ID for audit/ownership

        Returns:
            DeleteResponse with job_id and compound_name

        Raises:
            ValueError: If job not found
        """
        job = self.get_job(db, job_id)
        if not job:
            raise ValueError("Job not found")

        # Extract compound name from input_params
        compound_name = None
        if job.input_params:
            try:
                params = InputParams.model_validate_json(job.input_params)
                compound_name = params.compound_name
            except Exception:
                pass

        # Extract entry_id from result_summary
        entry_id = None
        if job.result_summary:
            try:
                summary = ResultSummary.model_validate_json(job.result_summary)
                entry_id = summary.entry_id
            except Exception:
                pass

        # Clean up result files and compound record
        if entry_id:
            delete_result_from_azure_by_entry_id(entry_id)
            prefix = entry_id[:2].lower()
            local_zip = settings.RESULTS_DIR / prefix / f"{entry_id}.zip"
            if local_zip.exists():
                try:
                    local_zip.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete local result {local_zip}: {e}")

            with _db_write_lock:
                compound_entry = compound_repo.get_by_entry_id(db, entry_id)
                if compound_entry:
                    compound_repo.archive_compound(
                        db, compound_entry,
                        session_id=session_id,
                        deletion_reason="user_request",
                        job_id=job_id,
                    )
                    compound_repo.delete_compound(db, compound_entry)
                # Delete job in same transaction to maintain atomicity
                job_repo.delete_job(db, job_id)
                db.commit()
        else:
            # No compound to clean up — just delete the job
            with _db_write_lock:
                job_repo.delete_job(db, job_id)
                db.commit()

        return DeleteResponse(
            message="Job and results deleted",
            job_id=job_id,
            compound_name=compound_name,
        )

    # ================================================================
    # Extracted orchestration methods (moved from jobs.py route handlers)
    # ================================================================

    def submit_job(
        self,
        db: Session,
        request: JobCreate,
        session_id: str,
        idempotency_key: Optional[str] = None,
    ) -> Union[JobResponse, DuplicateFoundResponse]:
        """Submit a new compound processing job with duplicate detection.

        Handles idempotency, InChIKey-based duplicate detection, and atomic
        check-and-create with retry for race conditions.

        Returns:
            JobResponse on successful creation, DuplicateFoundResponse when duplicate found.
        """
        # Check idempotency key - return existing job if already created
        if idempotency_key:
            idempotency_key = idempotency_key[:64]
            existing_job = job_repo.find_by_idempotency_key(db, session_id, idempotency_key)
            if existing_job:
                logger.info(f"Idempotent request - returning existing job {existing_job.id}")
                return _job_to_response(existing_job)

        # Generate InChIKey for duplicate detection
        inchikey = generate_inchikey(request.smiles) if request.smiles else None

        # Pre-compute submitted config for comparison
        submitted_threshold = request.similarity_threshold or 90
        submitted_at = _normalize_activity_types(request.activity_types)

        def _build_duplicate_response(
            existing_compound: Compound,
            config_match: str,
        ) -> DuplicateFoundResponse:
            name_matches = existing_compound.compound_name.lower().strip() == request.compound_name.lower().strip()
            duplicate_type = "exact" if name_matches else "structure_only"
            suggested_name = get_next_version_name(db, existing_compound.compound_name)
            config_diff = _build_config_diff(existing_compound, submitted_threshold, submitted_at)

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
            if not inchikey:
                return None
            existing_compounds = compound_repo.find_by_inchikey(db, inchikey)
            if not existing_compounds:
                return None

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
        max_retries = 3
        for attempt in range(max_retries):
            dup_response = _find_best_duplicate_match()
            if dup_response:
                return dup_response

            try:
                job = self.create_job(
                    db,
                    JobType.SINGLE,
                    request.model_dump(exclude={"session_id"}),
                    session_id=session_id,
                    idempotency_key=idempotency_key,
                )
                return _job_to_response(job)

            except IntegrityError:
                db.rollback()
                if attempt < max_retries - 1:
                    logger.info(f"Retry {attempt + 1}/{max_retries} for {request.compound_name} due to race condition")
                    continue

                dup_response = _find_best_duplicate_match()
                if dup_response:
                    return dup_response

                logger.error(f"Failed to create job for {request.compound_name} after {max_retries} retries")
                raise

    def resolve_duplicate_action(
        self,
        db: Session,
        request: ResolveDuplicateRequest,
        session_id: str,
    ) -> Union[JobResponse, SkipResponse]:
        """Resolve a duplicate compound situation based on user's choice.

        Returns:
            JobResponse if job created (replace/duplicate), SkipResponse if skipped.

        Raises:
            ValueError: For invalid requests (missing entry_id, identical config duplicate).
        """
        from backend.core.scheduler import job_scheduler

        # Handle SKIP action
        if request.action == DuplicateAction.SKIP:
            logger.info(f"User skipped duplicate: {request.compound_name}")
            return SkipResponse(
                status="skipped",
                message=f"Compound '{request.compound_name}' processing skipped by user",
                compound_name=request.compound_name,
            )

        # Handle REPLACE action
        if request.action == DuplicateAction.REPLACE:
            inherit_children_from = None
            replace_entry_id = None
            old_name = None
            if request.existing_entry_id:
                existing = compound_repo.get_by_entry_id(db, request.existing_entry_id)
                if existing:
                    old_name = existing.compound_name
                    old_entry_id = existing.entry_id
                    replace_entry_id = old_entry_id

                    if not existing.is_duplicate:
                        children = compound_repo.find_children(db, old_entry_id)
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

            if replace_entry_id and old_name:
                compound_name = request.new_compound_name or old_name
            else:
                compound_name = request.new_compound_name or request.compound_name

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
                if existing and existing.is_duplicate:
                    job_params["is_duplicate"] = True
                    job_params["duplicate_of"] = existing.duplicate_of

            job = self.create_job(
                db,
                JobType.SINGLE,
                job_params,
                session_id=session_id,
            )

            job_scheduler.trigger()
            logger.info(f"Job {job.id} created as replacement for {compound_name}")
            return _job_to_response(job)

        # Handle DUPLICATE action
        if request.action == DuplicateAction.DUPLICATE:
            if not request.existing_entry_id:
                raise ValueError("existing_entry_id is required for duplicate action.")

            existing = compound_repo.get_by_entry_id(db, request.existing_entry_id)
            if not existing:
                raise ValueError("Invalid existing_entry_id for duplicate action.")

            submitted_at = _normalize_activity_types(request.activity_types)
            config = _compute_config_match(
                existing, request.similarity_threshold or 90, submitted_at
            )
            if config == "identical":
                raise ValueError(
                    "Cannot create duplicate with identical configuration. "
                    "Use 'replace' to reprocess or 'skip' to keep existing."
                )

            compound_name = request.new_compound_name or request.compound_name

            job = self.create_job(
                db,
                JobType.SINGLE,
                {
                    "compound_name": compound_name,
                    "author_name": request.author_name,
                    "smiles": request.smiles,
                    "similarity_threshold": request.similarity_threshold,
                    "activity_types": request.activity_types,
                    "is_duplicate": True,
                    "duplicate_of": existing.entry_id,
                },
                session_id=session_id,
            )

            job_scheduler.trigger()
            logger.info(f"Job {job.id} created as duplicate (tagged) for {compound_name}")
            return _job_to_response(job)

        raise ValueError(f"Invalid action: {request.action}")

    def check_duplicates_batch(
        self,
        db: Session,
        request: CheckDuplicatesRequest,
    ) -> CheckDuplicatesResponse:
        """Check which compounds already exist or are being processed.

        Supports name-only (legacy) and structure-based (InChIKey) modes.

        Raises:
            ValueError: If neither compound_names nor compounds provided.
        """
        structure_matches: List[DuplicateMatch] = []
        internal_duplicates: List[InternalDuplicateMatch] = []
        submitted_threshold = request.similarity_threshold or 90
        submitted_at = _normalize_activity_types(request.activity_types)

        def _normalize_name(name: Optional[str]) -> str:
            return (name or "").strip().lower()

        # Determine which mode we're in
        if request.compounds:
            compound_names: List[str] = []
            seen_structure_key_to_name: Dict[str, str] = {}
            seen_name_to_name: Dict[str, str] = {}

            for compound in request.compounds:
                submitted_name = compound.compound_name
                normalized_submitted_name = _normalize_name(submitted_name)

                smiles = compound.smiles
                if not smiles and compound.inchi:
                    smiles = _inchi_to_smiles(compound.inchi)

                provided_inchikey = _normalize_inchikey_input(getattr(compound, "inchikey", None))
                generated_inchikey = generate_inchikey(smiles) if smiles else None
                inchikey = generated_inchikey or provided_inchikey
                structure_key = _inchikey_structure_key(inchikey)

                internal_parent_name = None
                internal_match_type = "exact"

                if normalized_submitted_name:
                    internal_parent_name = seen_name_to_name.get(normalized_submitted_name)
                    if internal_parent_name:
                        internal_match_type = "exact"
                    else:
                        seen_name_to_name[normalized_submitted_name] = submitted_name

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

                compound_names.append(submitted_name)

                if inchikey:
                    existing_candidates = compound_repo.find_by_inchikey(db, inchikey)

                    if existing_candidates:
                        same_name_candidates = [
                            c for c in existing_candidates
                            if _normalize_name(c.compound_name) == normalized_submitted_name
                        ]
                        candidates_for_selection = same_name_candidates or existing_candidates

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
                        config_diff = _build_config_diff(existing_compound, submitted_threshold, submitted_at)

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
            raise ValueError("Must provide either 'compound_names' or 'compounds' list")

        # Check for already processed compounds (by name)
        existing_map = self.check_existing_compounds(db, compound_names)
        existing = [name for name, exists in existing_map.items() if exists]

        # Check for currently processing compounds
        pending_map = self.check_pending_compounds(db, compound_names)
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

        # Compute suggested version names
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

    def submit_batch(
        self,
        db: Session,
        request: BatchJobCreate,
        session_id: str,
    ) -> BatchResponse:
        """Submit a batch of compound processing jobs.

        Handles per-compound duplicate decisions, internal dedup, and batch creation.
        """
        from backend.core.scheduler import job_scheduler

        # Generate a batch_id to link all jobs
        batch_id = self.generate_batch_id()

        # Extract duplicate_decisions from request
        duplicate_decisions = request.duplicate_decisions or {}

        # Check for currently processing compounds (always skipped)
        compound_names = [c.compound_name for c in request.compounds]
        pending = self.check_pending_compounds(db, compound_names)
        skipped_processing = list(pending.keys())

        # Track results
        skipped_existing = []
        replaced = []
        marked_duplicate = []
        skipped_internal_duplicates = []
        failed_compounds = []
        seen_structure_key_to_name: Dict[str, str] = {}
        seen_name_to_name: Dict[str, str] = {}

        jobs = []
        for compound in request.compounds:
            compound_name = compound.compound_name

            if compound_name in skipped_processing:
                continue

            try:
                compound_action = getattr(compound, 'duplicate_action', None) or duplicate_decisions.get(compound_name)

                if compound_action == 'skip':
                    skipped_existing.append(compound_name)
                    continue

                # Internal dedup guards
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

                replace_entry_id = None
                inherit_children_from = None

                if compound_action == 'replace':
                    existing = None
                    compound_inchikey = generate_inchikey(compound.smiles) if compound.smiles else None
                    if compound_inchikey:
                        candidates = compound_repo.find_by_inchikey(db, compound_inchikey)
                        existing = next(
                            (c for c in candidates if not c.is_duplicate),
                            candidates[0] if candidates else None
                        )
                    if existing:
                        replace_entry_id = existing.entry_id
                        if not existing.is_duplicate:
                            children = compound_repo.find_children(db, existing.entry_id)
                            if children:
                                inherit_children_from = existing.entry_id

                        logger.info(
                            f"Batch replace: deletion of '{compound_name}' (entry_id={replace_entry_id}) "
                            f"deferred until replacement job completes"
                        )

                job_params = compound.model_dump(exclude={"session_id", "duplicate_action", "original_compound_name"})
                if inherit_children_from:
                    job_params["inherit_children_from"] = inherit_children_from
                if replace_entry_id:
                    job_params["replace_entry_id"] = replace_entry_id
                    if existing and existing.is_duplicate:
                        job_params["is_duplicate"] = True
                        job_params["duplicate_of"] = existing.duplicate_of
                if compound_action == 'duplicate':
                    dup_inchikey = generate_inchikey(compound.smiles) if compound.smiles else None
                    dup_existing = None
                    if dup_inchikey:
                        dup_matches = compound_repo.find_by_inchikey(db, dup_inchikey)
                        dup_existing = dup_matches[0] if dup_matches else None

                    if not dup_existing:
                        parent_lookup_name = (
                            getattr(compound, "original_compound_name", None)
                            or compound_name
                        )
                        parent_lookup_name = (parent_lookup_name or "").strip()
                        name_candidates = compound_repo.find_by_name_case_insensitive(
                            db, parent_lookup_name
                        )
                        dup_existing = next(
                            (c for c in name_candidates if not c.is_duplicate),
                            name_candidates[0] if name_candidates else None
                        )

                    if dup_existing and dup_existing.entry_id:
                        dup_submitted_threshold = compound.similarity_threshold or 90
                        dup_submitted_at = _normalize_activity_types(compound.activity_types)
                        dup_config_match = _compute_config_match(dup_existing, dup_submitted_threshold, dup_submitted_at)

                        if dup_config_match == "identical":
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

                job = self.create_job(
                    db,
                    JobType.BATCH,
                    job_params,
                    session_id=session_id,
                    batch_id=batch_id,
                )
                jobs.append(_job_to_response(job))

                if replace_entry_id:
                    replaced.append(compound_name)

            except Exception as e:
                logger.error(f"Failed to create job for compound '{compound_name}': {e}", exc_info=True)
                db.rollback()
                failed_compounds.append(FailedCompound(
                    compound_name=compound_name,
                    error="Failed to create job. Please check compound data and try again."
                ))

        if jobs:
            job_scheduler.trigger()

        total_skipped = len(skipped_existing) + len(skipped_processing) + len(skipped_internal_duplicates)
        from backend.core.auth import truncate_session_id
        logger.info(
            f"Batch {batch_id}: {len(jobs)} jobs queued, "
            f"{len(replaced)} replaced, {len(marked_duplicate)} as duplicates, "
            f"{total_skipped} skipped "
            f"({len(skipped_existing)} existing, {len(skipped_processing)} processing, "
            f"{len(skipped_internal_duplicates)} internal duplicates), "
            f"{len(failed_compounds)} failed (session={truncate_session_id(session_id)})"
        )
        if failed_compounds:
            logger.warning(f"Batch {batch_id} had {len(failed_compounds)} compound failures: {[f.compound_name for f in failed_compounds]}")

        return BatchResponse(
            batch_id=batch_id,
            jobs=jobs,
            skipped_existing=skipped_existing,
            skipped_processing=skipped_processing,
            skipped_internal_duplicates=skipped_internal_duplicates,
            replaced=replaced,
            total_submitted=len(jobs),
            total_skipped=total_skipped,
            failed_compounds=failed_compounds,
        )

    def check_availability_batch_service(
        self,
        db: Session,
        request: CheckAvailabilityBatchRequest,
    ) -> CheckAvailabilityBatchResponse:
        """Batch availability check for multiple compounds.

        Probes ChEMBL for each compound at the requested threshold and all lower
        thresholds. Returns per-compound availability with existing compound matches.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from backend.modules.api_client import probe_all_thresholds

        results: List[CompoundAvailability] = []

        # Pre-fetch all existing compounds by InChIKey in a single DB query
        inchikey_map: Dict[tuple, str] = {}
        structure_keys: set = set()
        for compound in request.compounds:
            smiles = compound.smiles
            name = compound.compound_name
            if smiles:
                ik = generate_inchikey(smiles)
                if ik:
                    inchikey_map[(name, smiles)] = ik
                    sk = _inchikey_structure_key(ik)
                    if sk:
                        structure_keys.add(sk)

        existing_by_structure: Dict[str, List[Compound]] = {}
        if structure_keys:
            all_existing = []
            for sk in structure_keys:
                all_existing.extend(compound_repo.find_by_inchikey_like(db, sk))
            for comp in all_existing:
                sk = _inchikey_structure_key(comp.inchikey)
                if sk:
                    existing_by_structure.setdefault(sk, []).append(comp)

        submitted_at = _normalize_activity_types(request.activity_types)

        def _probe_compound(compound_input) -> CompoundAvailability:
            smiles = compound_input.smiles
            name = compound_input.compound_name

            thresholds = probe_all_thresholds(smiles, request.similarity_threshold)
            threshold_items = [
                ThresholdAvailability(threshold=t["threshold"], count=t["count"])
                for t in thresholds
            ]

            count_at_threshold = 0
            for t in thresholds:
                if t["threshold"] == request.similarity_threshold:
                    count_at_threshold = t["count"]
                    break

            available = count_at_threshold > 0
            has_any_data = any(t["count"] > 0 for t in thresholds)

            existing_compounds: List[ExistingCompoundAtThreshold] = []
            ik = inchikey_map.get((name, smiles))
            if ik:
                sk = _inchikey_structure_key(ik)
                if sk:
                    for comp in existing_by_structure.get(sk, []):
                        existing_compounds.append(
                            _build_existing_at_threshold(comp, request.similarity_threshold, submitted_at)
                        )

            return CompoundAvailability(
                compound_name=name,
                smiles=smiles,
                available=available,
                count_at_threshold=count_at_threshold,
                thresholds=threshold_items,
                existing_compounds=existing_compounds,
                has_any_data=has_any_data,
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(_probe_compound, c): c
                for c in request.compounds
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    compound = futures[future]
                    logger.warning(f"Availability probe failed for {compound.compound_name}: {e}")
                    results.append(CompoundAvailability(
                        compound_name=compound.compound_name,
                        smiles=compound.smiles,
                        available=False,
                        count_at_threshold=0,
                        has_any_data=False,
                    ))

        available_count = sum(1 for r in results if r.available)
        no_data_count = sum(1 for r in results if not r.has_any_data)
        unavailable_count = len(results) - available_count - no_data_count

        return CheckAvailabilityBatchResponse(
            results=results,
            available_count=available_count,
            unavailable_count=unavailable_count,
            no_data_count=no_data_count,
        )

    def recover_on_startup(self, db: Session, scheduler_trigger) -> dict:
        """Recover stalled jobs on startup using state machine validators.

        Resets PROCESSING -> PENDING (if valid transition), identifies SYNC_PENDING
        jobs for retry, and triggers the scheduler if there's work to do.

        Uses VALID_TRANSITIONS to validate status changes instead of direct
        ORM attribute assignment (ARCH-12).

        Args:
            db: Database session
            scheduler_trigger: Callable to trigger the job scheduler (e.g., job_scheduler.trigger)

        Returns:
            Dict with recovery stats: {recovered, pending, sync_pending}
        """
        stalled = job_repo.get_by_status(db, JobStatus.PROCESSING)
        pending_count = job_repo.count_by_status(db, [JobStatus.PENDING])
        sync_pending = job_repo.get_by_status(db, JobStatus.SYNC_PENDING)

        recovered = 0
        for job in stalled:
            original_status = job.status
            valid_next = VALID_TRANSITIONS.get(original_status, set())
            if JobStatus.PENDING in valid_next:
                job.status = JobStatus.PENDING
                job.current_step = "Queued (recovered)"
                recovered += 1
                logger.info(f"Recovered stalled job {job.id} from {original_status}")
            else:
                logger.warning(
                    f"Cannot recover job {job.id}: status {original_status} "
                    f"has no valid transition to PENDING (valid: {valid_next})"
                )

        if recovered:
            db.commit()
            logger.info(f"Startup recovery complete: {recovered} jobs recovered")

        if sync_pending:
            logger.info(f"Found {len(sync_pending)} SYNC_PENDING jobs for Azure retry")

        # Trigger scheduler if there's any work
        if stalled or pending_count > 0 or sync_pending:
            scheduler_trigger()
            logger.info(
                f"Scheduler triggered on startup "
                f"(recovered={recovered}, pending={pending_count}, sync_pending={len(sync_pending)})"
            )

        return {
            "recovered": recovered,
            "pending": pending_count,
            "sync_pending": len(sync_pending),
        }


# Global service instance
job_service = JobService()
