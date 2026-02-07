"""
Job service for CRUD operations and state management.
Handles job creation, progress updates, and completion tracking.
"""
import json
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Callable

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, case

from backend.models.database import Job, JobStatus, JobType
from backend.core.azure_sync import sync_db_to_azure

logger = logging.getLogger(__name__)

# Module-level lock for SQLite write serialization
# SQLite doesn't support SELECT ... FOR UPDATE, so we use application-level locking
_db_write_lock = threading.Lock()

# Valid status transitions for job state machine
VALID_TRANSITIONS = {
    JobStatus.PENDING: {JobStatus.PROCESSING, JobStatus.CANCELLED},
    JobStatus.PROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
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
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        logger.info(f"Created job {job.id} ({job_type.value}) session={session_id} batch={batch_id}")
        return job

    def generate_batch_id(self) -> str:
        """Generate a new batch ID for grouping jobs."""
        return str(uuid.uuid4())

    def get_job(self, db: Session, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return db.query(Job).filter(Job.id == job_id).first()

    def _get_job_for_update(self, db: Session, job_id: str) -> Optional[Job]:
        """Get a job by ID with thread-safe locking.

        Note: SQLite doesn't support FOR UPDATE, so we use application-level
        locking via _db_write_lock. All callers must acquire the lock before
        calling this method and hold it through commit.
        """
        return db.query(Job).filter(Job.id == job_id).first()

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

    def get_job_with_params(self, db: Session, job_id: str) -> Optional[Dict]:
        """Get job with parsed input parameters."""
        job = self.get_job(db, job_id)
        if not job:
            return None

        result = job.to_dict()
        if job.input_params:
            result["input_params"] = _safe_json_loads(job.input_params, {})
        if job.result_summary:
            result["result_summary"] = _safe_json_loads(job.result_summary, {})
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
        query = db.query(Job)

        # Filter by session_id for user isolation
        if session_id:
            query = query.filter(Job.session_id == session_id)

        if statuses:
            query = query.filter(Job.status.in_(statuses))

        total = query.count()
        pages = (total + page_size - 1) // page_size

        # Add Job.id as tie-breaker to ensure stable pagination order
        jobs = (
            query.order_by(desc(Job.created_at), desc(Job.id))
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

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
        active_query = db.query(Job).filter(
            Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
        )
        if session_id:
            active_query = active_query.filter(Job.session_id == session_id)
        active_jobs = active_query.all()

        # Get recently completed/failed jobs (within last N minutes)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=include_recent_minutes)
        recent_query = db.query(Job).filter(
            Job.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
            Job.completed_at >= recent_cutoff
        )
        if session_id:
            recent_query = recent_query.filter(Job.session_id == session_id)
        recent_jobs = recent_query.all()

        # Combine and sort: completed jobs first, then by created_at (newest first within each group)
        all_jobs = active_jobs + recent_jobs
        # Sort key: (0 if completed/failed, 1 otherwise), then by created_at descending
        all_jobs.sort(
            key=lambda j: (
                0 if j.status in [JobStatus.COMPLETED, JobStatus.FAILED] else 1,
                -(j.created_at or datetime.min.replace(tzinfo=timezone.utc)).timestamp()
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
            # Extract compound name from input params
            params = _safe_json_loads(job.input_params, {})
            compound_name = params.get("compound_name", "Unknown")
            item["compound_name"] = compound_name

            # For completed jobs, include entry_id and storage_path for UUID-based storage lookup
            entry_id = None
            storage_path = None
            if job.status == JobStatus.COMPLETED:
                # First try result_summary
                result_data = _safe_json_loads(job.result_summary, {})
                entry_id = result_data.get("entry_id")
                storage_path = result_data.get("storage_path")

                # Fallback: look up from Compound table
                if (not entry_id or not storage_path) and compound_name != "Unknown":
                    from backend.models.database import Compound
                    compound = db.query(Compound).filter(
                        Compound.compound_name == compound_name
                    ).order_by(Compound.processed_at.desc()).first()
                    if compound:
                        entry_id = entry_id or compound.entry_id
                        storage_path = storage_path or compound.storage_path

            item["entry_id"] = entry_id
            item["storage_path"] = storage_path
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
        # Single aggregated query for all status counts
        summary = db.query(
            func.count(Job.id).label('total'),
            func.sum(case((Job.status == JobStatus.COMPLETED, 1), else_=0)).label('completed'),
            func.sum(case((Job.status == JobStatus.PROCESSING, 1), else_=0)).label('processing'),
            func.sum(case((Job.status == JobStatus.PENDING, 1), else_=0)).label('pending'),
            func.sum(case((Job.status == JobStatus.FAILED, 1), else_=0)).label('failed'),
            func.sum(case((Job.status == JobStatus.CANCELLED, 1), else_=0)).label('cancelled'),
            func.avg(Job.progress).label('avg_progress'),
            func.min(Job.created_at).label('first_created'),
        ).filter(Job.batch_id == batch_id).first()

        if not summary or summary.total == 0:
            return {}

        # Get compound names for first 5 jobs (separate query, limited)
        first_jobs = (
            db.query(Job.input_params)
            .filter(Job.batch_id == batch_id)
            .order_by(Job.created_at)
            .limit(5)
            .all()
        )
        compound_names = []
        for (input_params,) in first_jobs:
            params = _safe_json_loads(input_params, {})
            compound_names.append(params.get("compound_name", "Unknown"))

        return {
            "batch_id": batch_id,
            "total_jobs": summary.total,
            "completed": summary.completed or 0,
            "processing": summary.processing or 0,
            "pending": summary.pending or 0,
            "failed": summary.failed or 0,
            "cancelled": summary.cancelled or 0,
            "overall_progress": summary.avg_progress or 0,
            "created_at": summary.first_created.isoformat() if summary.first_created else None,
            "compound_names": compound_names,
        }

    def cancel_batch(self, db: Session, batch_id: str) -> int:
        """
        Cancel all pending/processing jobs in a batch.

        Args:
            db: Database session
            batch_id: Batch ID to cancel

        Returns:
            Number of jobs cancelled
        """
        jobs = (
            db.query(Job)
            .filter(
                Job.batch_id == batch_id,
                Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
            )
            .all()
        )

        cancelled_count = 0
        for job in jobs:
            job.status = JobStatus.CANCELLED
            job.current_step = "Cancelled"
            job.completed_at = datetime.now(timezone.utc)
            cancelled_count += 1

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
        from backend.models.database import Compound

        # Batch query: get all matching compounds in a single query
        existing_compounds = (
            db.query(Compound.compound_name)
            .filter(Compound.compound_name.in_(compound_names))
            .all()
        )
        local_existing = {row[0] for row in existing_compounds}

        result = {}
        for name in compound_names:
            result[name] = name in local_existing

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

        # Convert to set for O(1) lookups
        names_to_check = set(compound_names)
        result = {}

        # Fetch all pending/processing jobs in one query
        pending_jobs = (
            db.query(Job)
            .filter(Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]))
            .all()
        )

        # Parse JSON and match compound names
        for job in pending_jobs:
            if not job.input_params:
                continue
            try:
                params = json.loads(job.input_params)
                job_compound_name = params.get('compound_name')
                if job_compound_name and job_compound_name in names_to_check:
                    result[job_compound_name] = job.id
                    # Remove from set to avoid duplicate matches
                    names_to_check.discard(job_compound_name)
                    # Early exit if all found
                    if not names_to_check:
                        break
            except (json.JSONDecodeError, TypeError):
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

            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.current_step = "Completed"
            job.result_path = result_path
            job.result_summary = json.dumps(result_summary)
            job.completed_at = datetime.now(timezone.utc)

            # Extract duplicate metadata from job input_params
            job_params = {}
            if job.input_params:
                try:
                    job_params = json.loads(job.input_params)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Update Compound table for local DB consistency
            self._update_compound_entry(
                db,
                result_summary,
                result_path,
                is_duplicate=job_params.get('is_duplicate', False),
                duplicate_of=job_params.get('duplicate_of'),
                inherit_children_from=job_params.get('inherit_children_from'),
                replace_entry_id=job_params.get('replace_entry_id'),
                session_id=job.session_id,
                job_id=job_id,
            )

            db.commit()
            db.refresh(job)

        # All network I/O happens OUTSIDE the lock to avoid blocking other threads

        # Clean up replaced compound's files (Azure + local)
        replace_entry_id = job_params.get('replace_entry_id')
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
                    chembl_id=result_summary.get('chembl_id', ''),
                    total_activities=result_summary.get('total_activities', 0),
                    imp_candidates=result_summary.get('imp_candidates', 0),
                    imp_score=result_summary.get('imp_score'),
                    similarity_threshold=similarity_threshold,
                    activity_types=activity_types,
                    qed=qed,
                    num_outliers=num_outliers,
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
            # When replacing, ALWAYS create a fresh entry — never update another
            # compound that happens to share the InChIKey or name. The old compound
            # (replace_entry_id) will be deleted after this new one is saved.
            existing = None
            if not replace_entry_id:
                if inchikey:
                    existing = db.query(Compound).filter(Compound.inchikey == inchikey).first()

                # Only fall back to exact name match if:
                # 1. No InChIKey provided for the new compound, AND
                # 2. The existing record has no InChIKey (to avoid false matches)
                if not existing and not inchikey:
                    existing = db.query(Compound).filter(
                        Compound.compound_name == compound_name,
                        Compound.inchikey.is_(None)
                    ).first()

            if existing:
                # Update existing entry
                existing.smiles = smiles
                existing.canonical_smiles = canonical_smiles
                existing.inchikey = inchikey
                existing.chembl_id = result_summary.get('chembl_id', '')
                existing.total_activities = result_summary.get('total_activities', 0)
                existing.imp_candidates = result_summary.get('imp_candidates', 0)
                existing.imp_score = result_summary.get('imp_score')
                existing.similarity_threshold = similarity_threshold
                existing.activity_types = activity_types
                existing.qed = qed
                existing.num_outliers = num_outliers
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
                    chembl_id=result_summary.get('chembl_id', ''),
                    total_activities=result_summary.get('total_activities', 0),
                    imp_candidates=result_summary.get('imp_candidates', 0),
                    imp_score=result_summary.get('imp_score'),
                    similarity_threshold=similarity_threshold,
                    activity_types=activity_types,
                    qed=qed,
                    num_outliers=num_outliers,
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
                children = db.query(Compound).filter(
                    Compound.duplicate_of == inherit_children_from
                ).all()
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
                old_compound = db.query(Compound).filter(
                    Compound.entry_id == replace_entry_id
                ).first()
                if old_compound:
                    old_name = old_compound.compound_name

                    # Promote children before deleting (prevents orphans)
                    if not old_compound.is_duplicate:
                        # Re-point any remaining children to the new compound
                        remaining_children = db.query(Compound).filter(
                            Compound.duplicate_of == replace_entry_id
                        ).all()
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

        except Exception as e:
            logger.error(f"Failed to update Compound entry for {compound_name}: {e}")

    def fail_job(
        self,
        db: Session,
        job_id: str,
        error_message: str,
    ) -> Optional[Job]:
        """
        Mark job as failed with thread-safe locking.

        Args:
            db: Database session
            job_id: Job ID
            error_message: Error description
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

            db.commit()
            db.refresh(job)

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
        job = self.get_job(db, job_id)
        if not job:
            return False

        db.delete(job)
        db.commit()
        logger.info(f"Job {job_id} deleted")
        return True


# Global service instance
job_service = JobService()
