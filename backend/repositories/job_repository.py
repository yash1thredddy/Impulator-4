"""JobRepository - centralized Job query patterns for Postgres.

Standalone repository using SA 2.0 select() style with Postgres-native
row-level locking (FOR UPDATE) for write-path concurrency.
Direct column access -- no JSON blob parsing, no input_params.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import case, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models.enums import JobStatus, JobType
from backend.models.job import Job

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific exceptions for Postgres constraint violations
# ---------------------------------------------------------------------------


class DuplicateEntryError(Exception):
    """Raised when a unique constraint is violated."""


class ReferenceError(Exception):  # noqa: A001 — intentional shadow
    """Raised when a foreign key constraint is violated."""


class ValidationError(Exception):  # noqa: A001
    """Raised when a CHECK constraint is violated."""


def _handle_integrity_error(e: IntegrityError) -> None:
    """Convert SQLAlchemy IntegrityError to domain-specific exception.

    Inspects the underlying ``psycopg2`` error to classify the violation.
    """
    from psycopg2.errors import (  # type: ignore[import-untyped]
        CheckViolation,
        ForeignKeyViolation,
        UniqueViolation,
    )

    orig = e.orig
    if isinstance(orig, UniqueViolation):
        raise DuplicateEntryError(str(orig)) from e
    if isinstance(orig, ForeignKeyViolation):
        raise ReferenceError(str(orig)) from e
    if isinstance(orig, CheckViolation):
        raise ValidationError(str(orig)) from e
    raise  # Re-raise unknown integrity errors


# ---------------------------------------------------------------------------
# JobRepository
# ---------------------------------------------------------------------------


class JobRepository:
    """Repository for Job CRUD and domain queries.

    All queries use SA 2.0 ``select()`` style.  Write-path concurrency is
    handled via Postgres ``FOR UPDATE`` row locks, not application-level
    threading locks.
    """

    # ---- Read methods ----

    def get_by_job_id(self, db: Session, job_id: uuid.UUID) -> Job | None:
        """Get a single job by its UUID."""
        return db.scalars(select(Job).where(Job.id == job_id)).first()

    def get_active_jobs(
        self, db: Session, session_id: uuid.UUID | None = None
    ) -> list[Job]:
        """Get pending/processing jobs, optionally filtered by session_id."""
        stmt = select(Job).where(
            Job.status.in_([
                JobStatus.PENDING,
                JobStatus.PROCESSING,
                JobStatus.PENDING_UPLOAD,
            ])
        )
        if session_id is not None:
            stmt = stmt.where(Job.session_id == session_id)
        return list(db.scalars(stmt).all())

    def get_jobs_paginated(
        self,
        db: Session,
        session_id: uuid.UUID | None = None,
        status_filter: list[JobStatus] | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Job], int]:
        """Get paginated jobs with optional filters.

        Returns ``(jobs, total_count)``.
        """
        base = select(Job)
        if session_id is not None:
            base = base.where(Job.session_id == session_id)
        if status_filter:
            base = base.where(Job.status.in_(status_filter))

        # Single query: count(*) OVER() returns total alongside rows,
        # avoiding a separate COUNT subquery round trip.
        count_window = func.count().over().label("_total")
        windowed = (
            base.add_columns(count_window)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .offset(offset)
            .limit(limit)
        )
        result_rows = db.execute(windowed).all()

        if not result_rows:
            return [], 0

        total = result_rows[0]._total
        rows = [row[0] for row in result_rows]
        return rows, total

    def find_by_idempotency_key(
        self,
        db: Session,
        session_id: uuid.UUID,
        idempotency_key: str,
    ) -> Job | None:
        """Find a job by its idempotency key within a session."""
        return db.scalars(
            select(Job).where(
                Job.session_id == session_id,
                Job.idempotency_key == idempotency_key,
            )
        ).first()

    def get_batch_jobs(self, db: Session, batch_id: uuid.UUID) -> list[Job]:
        """Get all jobs in a batch."""
        return list(
            db.scalars(select(Job).where(Job.batch_id == batch_id)).all()
        )

    def get_batch_summary(
        self, db: Session, batch_id: uuid.UUID
    ) -> dict[str, Any]:
        """Aggregate batch status counts in a single query.

        Uses direct ``Job.compound_name`` column access -- no JSON parsing.
        """
        summary = db.execute(
            select(
                func.count(Job.id).label("total"),
                func.sum(case(
                    (Job.status.in_([JobStatus.COMPLETED, JobStatus.PENDING_UPLOAD]), 1),
                    else_=0,
                )).label("completed"),
                func.sum(
                    case((Job.status == JobStatus.PROCESSING, 1), else_=0)
                ).label("processing"),
                func.sum(case((Job.status == JobStatus.PENDING, 1), else_=0)).label(
                    "pending"
                ),
                func.sum(case((Job.status == JobStatus.FAILED, 1), else_=0)).label(
                    "failed"
                ),
                func.sum(
                    case((Job.status == JobStatus.CANCELLED, 1), else_=0)
                ).label("cancelled"),
                func.avg(Job.progress).label("avg_progress"),
                func.min(Job.created_at).label("first_created"),
            ).where(Job.batch_id == batch_id)
        ).first()

        if not summary or summary.total == 0:
            return {}

        # Direct column access -- compound_name is a first-class column
        compound_names = list(
            db.scalars(
                select(Job.compound_name)
                .where(Job.batch_id == batch_id)
                .order_by(Job.created_at)
                .limit(5)
            ).all()
        )

        return {
            "batch_id": str(batch_id),
            "total_jobs": summary.total,
            "completed": summary.completed or 0,
            "processing": summary.processing or 0,
            "pending": summary.pending or 0,
            "failed": summary.failed or 0,
            "cancelled": summary.cancelled or 0,
            "overall_progress": summary.avg_progress or 0,
            "created_at": (
                summary.first_created.isoformat() if summary.first_created else None
            ),
            "compound_names": compound_names,
        }

    def get_completed_jobs_since(
        self,
        db: Session,
        cutoff: datetime,
        session_id: uuid.UUID | None = None,
    ) -> list[Job]:
        """Get jobs completed after *cutoff*, optionally by session."""
        stmt = select(Job).where(
            Job.status == JobStatus.COMPLETED,
            Job.completed_at >= cutoff,
        )
        if session_id is not None:
            stmt = stmt.where(Job.session_id == session_id)
        return list(db.scalars(stmt).all())

    def get_failed_jobs_since(
        self,
        db: Session,
        cutoff: datetime,
        session_id: uuid.UUID | None = None,
    ) -> list[Job]:
        """Get jobs failed after *cutoff*, optionally by session."""
        stmt = select(Job).where(
            Job.status == JobStatus.FAILED,
            Job.completed_at >= cutoff,
        )
        if session_id is not None:
            stmt = stmt.where(Job.session_id == session_id)
        return list(db.scalars(stmt).all())

    def get_by_status(self, db: Session, status: JobStatus) -> list[Job]:
        """Get all jobs with a given status."""
        return list(
            db.scalars(select(Job).where(Job.status == status)).all()
        )

    def get_stalled_processing_jobs(
        self, db: Session, timeout_seconds: int
    ) -> list[Job]:
        """Get PROCESSING jobs that have exceeded the timeout.

        Pushes the time comparison to Postgres via ``now() - INTERVAL`` --
        avoids fetching all processing jobs and filtering in Python.
        Uses the ``idx_jobs_processing`` partial index.
        """
        return list(
            db.scalars(
                select(Job).where(
                    Job.status == JobStatus.PROCESSING,
                    Job.started_at.isnot(None),
                    Job.started_at < func.now() - text(f"interval '{timeout_seconds} seconds'"),
                )
            ).all()
        )

    def count_by_status(self, db: Session, statuses: list[JobStatus]) -> int:
        """Count jobs matching any of the given statuses."""
        return (
            db.scalar(
                select(func.count(Job.id)).where(Job.status.in_(statuses))
            )
            or 0
        )

    def get_pending_processing_count(self, db: Session) -> int:
        """Count active (pending + processing) jobs."""
        return self.count_by_status(
            db, [JobStatus.PENDING, JobStatus.PROCESSING]
        )

    def claim_next_pending_job(self, db: Session) -> Job | None:
        """Claim the next PENDING job using round-robin fair scheduling.

        Two-step approach required because Postgres does not allow
        ``FOR UPDATE`` with window functions:

        1. Subquery with ``ROW_NUMBER()`` finds the fair-scheduled job ID
        2. Outer query locks and fetches that specific row with ``FOR UPDATE SKIP LOCKED``
        """
        # Step 1: Find the best candidate ID via window function (no lock)
        rn = func.row_number().over(
            partition_by=Job.session_id,
            order_by=Job.created_at,
        ).label("rn")

        sub = (
            select(Job.id, rn)
            .where(Job.status == JobStatus.PENDING)
            .subquery()
        )

        candidate_id = db.scalar(
            select(sub.c.id)
            .order_by(sub.c.rn, sub.c.id)
            .limit(1)
        )

        if candidate_id is None:
            return None

        # Step 2: Lock and fetch the candidate row
        stmt = (
            select(Job)
            .where(Job.id == candidate_id)
            .with_for_update(skip_locked=True)
        )
        return db.scalars(stmt).first()

    def claim_pending_jobs(self, db: Session, limit: int) -> list[Job]:
        """Claim up to *limit* PENDING jobs atomically (D-59).

        Uses ``FOR UPDATE SKIP LOCKED`` to avoid contention with other
        scheduler instances. Sets status to PROCESSING and started_at
        in a single round-trip.
        """
        stmt = (
            select(Job)
            .where(Job.status == JobStatus.PENDING)
            .order_by(Job.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        jobs = list(db.scalars(stmt).all())
        now = datetime.now(timezone.utc)
        for job in jobs:
            job.status = JobStatus.PROCESSING
            job.started_at = now
            job.current_step = "Starting..."
        db.flush()
        return jobs

    # ---- Write methods (Postgres FOR UPDATE for concurrency) ----

    def create_job(
        self,
        db: Session,
        *,
        id: uuid.UUID,
        job_type: JobType,
        compound_name: str,
        smiles: str | None = None,
        similarity_threshold: int = 90,
        activity_types: list[str] | None = None,
        session_id: uuid.UUID,
        batch_id: uuid.UUID | None = None,
        batch_index: int | None = None,
        idempotency_key: str | None = None,
        current_step: str = "Queued",
        progress: float = 0.0,
    ) -> Job:
        """Create a new Job record.

        Direct column assignment -- no ``input_params`` JSON blob.
        Constraint violations are translated to domain exceptions.
        """
        job = Job(
            id=id,
            job_type=job_type,
            compound_name=compound_name,
            smiles=smiles,
            similarity_threshold=similarity_threshold,
            activity_types=activity_types,
            session_id=session_id,
            batch_id=batch_id,
            batch_index=batch_index,
            idempotency_key=idempotency_key,
            current_step=current_step,
            progress=progress,
        )
        try:
            db.add(job)
            db.flush()
        except IntegrityError as e:
            db.rollback()
            _handle_integrity_error(e)
        return job

    def update_status(
        self,
        db: Session,
        job_id: uuid.UUID,
        new_status: JobStatus,
        **extra_fields: Any,
    ) -> Job | None:
        """Row-locked status update with SD-13 resurrection guard.

        Uses ``SELECT ... FOR UPDATE`` for safe concurrent transitions.
        Refuses to move CANCELLED or FAILED jobs to any non-terminal state.
        """
        stmt = select(Job).where(Job.id == job_id).with_for_update()
        job = db.scalars(stmt).first()
        if not job:
            return None

        # SD-13: never resurrect terminated jobs
        if job.status in (JobStatus.CANCELLED, JobStatus.FAILED) and new_status not in (
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        ):
            logger.warning(
                "resurrection_guard_blocked",
                job_id=str(job_id),
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
        self,
        db: Session,
        job_id: uuid.UUID,
        progress: float,
        current_step: str,
    ) -> Job | None:
        """Update progress and current_step for a job.

        Uses FOR UPDATE to prevent TOCTOU race with scheduler's
        _check_timeouts() (D-29). Lock hold time is ~1-5ms.
        """
        job = db.scalars(
            select(Job).where(Job.id == job_id).with_for_update()
        ).first()
        if not job:
            return None
        job.progress = progress
        job.current_step = current_step
        db.flush()
        return job

    def cancel_batch_jobs(
        self,
        db: Session,
        batch_id: uuid.UUID,
        session_id: uuid.UUID | None = None,
    ) -> int:
        """Cancel all pending/processing jobs in a batch.

        Uses ``FOR UPDATE`` to prevent TOCTOU between the select and the
        status mutation (a job could complete between the two otherwise).
        Returns the count of jobs cancelled.
        """
        stmt = (
            select(Job)
            .where(
                Job.batch_id == batch_id,
                Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
            )
            .with_for_update()
        )
        if session_id is not None:
            stmt = stmt.where(Job.session_id == session_id)

        jobs = list(db.scalars(stmt).all())
        cancelled = 0
        for job in jobs:
            now = datetime.now(timezone.utc)
            job.status = JobStatus.CANCELLED
            job.current_step = "Cancelled"
            job.cancelled_at = now
            job.completed_at = now
            cancelled += 1
        if cancelled:
            db.flush()
        return cancelled

    def delete_job(self, db: Session, job_id: uuid.UUID) -> bool:
        """Delete a job record. Returns ``True`` if deleted."""
        job = db.scalars(select(Job).where(Job.id == job_id)).first()
        if not job:
            return False
        db.delete(job)
        db.flush()
        return True


# Singleton instance (no model parameter -- standalone class)
job_repo = JobRepository()
