"""JobRepository - centralized Job query patterns (ARCH-01).

Replaces all db.query(Job) calls scattered across jobs.py,
job_service.py, and scheduler.py.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import case, desc, func, text
from sqlalchemy.orm import Session

from backend.models.database import Job, JobStatus, JobType
from backend.repositories.base import BaseRepository, _db_write_lock

logger = structlog.get_logger(__name__)


class JobRepository(BaseRepository[Job]):
    """Repository for Job CRUD and domain queries."""

    # ---- Read methods (no lock) ----

    def get_by_job_id(self, db: Session, job_id: str) -> Optional[Job]:
        """Get a single job by its UUID."""
        return db.query(Job).filter(Job.id == job_id).first()

    def get_active_jobs(self, db: Session, session_id: Optional[str] = None) -> List[Job]:
        """Get pending/processing jobs filtered by session_id."""
        query = db.query(Job).filter(
            Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
        )
        if session_id:
            query = query.filter(Job.session_id == session_id)
        return query.all()

    def get_jobs_paginated(
        self,
        db: Session,
        session_id: Optional[str] = None,
        status_filter: Optional[List[JobStatus]] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> Tuple[List[Job], int]:
        """Get paginated jobs with optional filters. Returns (jobs, total_count)."""
        query = db.query(Job)
        if session_id:
            query = query.filter(Job.session_id == session_id)
        if status_filter:
            query = query.filter(Job.status.in_(status_filter))
        total = query.count()
        jobs = (
            query
            .order_by(desc(Job.created_at), desc(Job.id))
            .offset(offset)
            .limit(limit)
            .all()
        )
        return jobs, total

    def find_by_idempotency_key(
        self, db: Session, session_id: str, idempotency_key: str
    ) -> Optional[Job]:
        """Find a job by its idempotency key within a session."""
        return (
            db.query(Job)
            .filter(Job.session_id == session_id, Job.idempotency_key == idempotency_key)
            .first()
        )

    def get_batch_jobs(self, db: Session, batch_id: str) -> List[Job]:
        """Get all jobs in a batch."""
        return db.query(Job).filter(Job.batch_id == batch_id).all()

    def get_batch_summary(self, db: Session, batch_id: str) -> Dict[str, Any]:
        """Aggregate batch status counts in a single query."""
        summary = db.query(
            func.count(Job.id).label("total"),
            func.sum(case((Job.status == JobStatus.COMPLETED, 1), else_=0)).label("completed"),
            func.sum(case((Job.status == JobStatus.PROCESSING, 1), else_=0)).label("processing"),
            func.sum(case((Job.status == JobStatus.PENDING, 1), else_=0)).label("pending"),
            func.sum(case((Job.status == JobStatus.FAILED, 1), else_=0)).label("failed"),
            func.sum(case((Job.status == JobStatus.CANCELLED, 1), else_=0)).label("cancelled"),
            func.avg(Job.progress).label("avg_progress"),
            func.min(Job.created_at).label("first_created"),
        ).filter(Job.batch_id == batch_id).first()

        if not summary or summary.total == 0:
            return {}

        # Get compound names for first 5 jobs
        first_jobs = (
            db.query(Job.input_params)
            .filter(Job.batch_id == batch_id)
            .order_by(Job.created_at)
            .limit(5)
            .all()
        )
        compound_names: List[str] = []
        for (input_params_str,) in first_jobs:
            if input_params_str:
                try:
                    from backend.models.schemas import InputParams
                    params = InputParams.model_validate_json(input_params_str)
                    compound_names.append(params.compound_name or "Unknown")
                except Exception:
                    compound_names.append("Unknown")

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

    def get_completed_jobs_since(
        self, db: Session, cutoff: datetime, session_id: Optional[str] = None
    ) -> List[Job]:
        """Get jobs completed after cutoff, optionally by session."""
        query = db.query(Job).filter(
            Job.status == JobStatus.COMPLETED,
            Job.completed_at >= cutoff,
        )
        if session_id:
            query = query.filter(Job.session_id == session_id)
        return query.all()

    def get_failed_jobs_since(
        self, db: Session, cutoff: datetime, session_id: Optional[str] = None
    ) -> List[Job]:
        """Get jobs failed after cutoff, optionally by session."""
        query = db.query(Job).filter(
            Job.status == JobStatus.FAILED,
            Job.completed_at >= cutoff,
        )
        if session_id:
            query = query.filter(Job.session_id == session_id)
        return query.all()

    def get_by_status(self, db: Session, status: JobStatus) -> List[Job]:
        """Get all jobs with a given status."""
        return db.query(Job).filter(Job.status == status).all()

    def get_sync_pending_jobs(self, db: Session) -> List[Job]:
        """Get jobs in SYNC_PENDING state (for Azure retry)."""
        return db.query(Job).filter(Job.status == JobStatus.SYNC_PENDING).all()

    def get_stalled_processing_jobs(self, db: Session, timeout_seconds: int) -> List[Job]:
        """Get PROCESSING jobs that started but have no activity beyond timeout."""
        return (
            db.query(Job)
            .filter(
                Job.status == JobStatus.PROCESSING,
                Job.started_at.isnot(None),
            )
            .all()
        )

    def count_by_status(self, db: Session, statuses: List[JobStatus]) -> int:
        """Count jobs matching any of the given statuses."""
        return db.query(Job).filter(Job.status.in_(statuses)).count()

    def get_pending_processing_count(self, db: Session) -> int:
        """Count active (pending+processing) jobs. Used for idle-stop check."""
        return db.query(Job).filter(
            Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
        ).count()

    def claim_next_pending_job(self, db: Session) -> Optional[Job]:
        """Claim the next PENDING job using round-robin fair scheduling.

        Uses raw SQL with ROW_NUMBER window function partitioned by session_id
        for fair interleaving across sessions (STAB-21).
        """
        fair_query = text("""
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY session_id
                    ORDER BY created_at
                ) as rn
                FROM jobs
                WHERE UPPER(status) = 'PENDING'
            )
            ORDER BY rn, id
            LIMIT 1
        """)
        result = db.execute(fair_query).first()
        if not result:
            return None
        return db.query(Job).filter(Job.id == result[0]).first()

    # ---- Write methods (with _db_write_lock) ----

    def create_job(
        self,
        db: Session,
        *,
        id: str,
        job_type: JobType,
        status: JobStatus = JobStatus.PENDING,
        session_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        input_params: Optional[str] = None,
        progress: float = 0.0,
        current_step: str = "Queued",
        request_id: Optional[str] = None,
    ) -> Job:
        """Create a new Job record with write lock."""
        job = Job(
            id=id,
            job_type=job_type,
            status=status,
            session_id=session_id,
            batch_id=batch_id,
            idempotency_key=idempotency_key,
            input_params=input_params,
            progress=progress,
            current_step=current_step,
            request_id=request_id,
        )
        with _db_write_lock:
            db.add(job)
            db.flush()
        return job

    def update_status(
        self,
        db: Session,
        job_id: str,
        new_status: JobStatus,
        **extra_fields: Any,
    ) -> Optional[Job]:
        """Write-locked status update with SD-13 resurrection guard.

        Refuses to move CANCELLED or FAILED jobs to any non-terminal state.
        Extra fields (error_message, result_path, etc.) are set on the job.
        """
        with _db_write_lock:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                return None

            # SD-13: never resurrect terminated jobs
            if job.status in (JobStatus.CANCELLED, JobStatus.FAILED) and new_status not in (
                JobStatus.CANCELLED,
                JobStatus.FAILED,
            ):
                logger.warning(
                    "resurrection_guard_blocked",
                    job_id=job_id,
                    current=job.status.value,
                    attempted=new_status.value,
                )
                return None

            job.status = new_status
            for field, value in extra_fields.items():
                if hasattr(job, field):
                    setattr(job, field, value)
            db.flush()
        return job

    def update_progress(
        self, db: Session, job_id: str, progress: float, current_step: str
    ) -> Optional[Job]:
        """Write-locked progress update."""
        with _db_write_lock:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                return None
            job.progress = progress
            job.current_step = current_step
            db.flush()
        return job

    def cancel_batch_jobs(self, db: Session, batch_id: str, session_id: Optional[str] = None) -> int:
        """Cancel all pending/processing jobs in a batch. Returns count cancelled.

        Query and mutation both happen inside the write lock to prevent
        TOCTOU: a job could complete between query and lock acquisition.
        """
        cancelled = 0
        with _db_write_lock:
            jobs = (
                db.query(Job)
                .filter(
                    Job.batch_id == batch_id,
                    Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
                )
                .all()
            )
            for job in jobs:
                job.status = JobStatus.CANCELLED
                job.current_step = "Cancelled"
                job.completed_at = datetime.now(timezone.utc)
                cancelled += 1
            if cancelled:
                db.flush()
        return cancelled

    def delete_job(self, db: Session, job_id: str) -> bool:
        """Write-locked job deletion. Returns True if deleted."""
        with _db_write_lock:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                return False
            db.delete(job)
            db.flush()
        return True


# Singleton instance
job_repo = JobRepository(Job)
