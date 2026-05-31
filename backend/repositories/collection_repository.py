"""CollectionRepository - centralized Collection query patterns for Postgres.

Standalone repository using SA 2.0 ``select()`` style, sync-first (HC-1):
every method is ``def name(self, db: Session, ...)`` -- synchronous only, no
async sessions (the DB layer is sync SQLAlchemy per 23-CONTEXT.md).
Mirrors :class:`~backend.repositories.job_repository.JobRepository` and
:class:`~backend.repositories.compound_repository.CompoundRepository` exactly.

A collection is ONE ``JobType.COLLECTION`` job plus a row in the ``collections``
table. Members are the input-definitions persisted in ``members_config`` JSONB
(D-02), loaded by the 1:1 ``job_id`` -- never from ``job.result_summary``.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models.collection import Collection
from backend.repositories.job_repository import _handle_integrity_error

logger = structlog.get_logger(__name__)


class CollectionRepository:
    """Repository for Collection CRUD and domain queries.

    All queries use SA 2.0 ``select()`` style. Soft-delete (D-11) is honoured
    on the global list path (D-05). Constraint violations are translated to
    domain exceptions via the shared ``_handle_integrity_error`` helper.
    """

    # ---- Read methods ----

    def get_by_id(
        self, db: Session, collection_id: uuid.UUID
    ) -> Collection | None:
        """Get a single collection by its UUID primary key."""
        return db.execute(
            select(Collection).where(Collection.id == collection_id)
        ).scalar_one_or_none()

    def get_by_job_id(
        self, db: Session, job_id: uuid.UUID
    ) -> Collection | None:
        """Load the collection row keyed on its 1:1 ``job_id`` (D-02 read path).

        This is the canonical way the scheduler / collection processing loads
        ``members_config`` -- by ``job_id``, never from ``job.result_summary``.
        """
        return db.execute(
            select(Collection).where(Collection.job_id == job_id)
        ).scalar_one_or_none()

    def list_all(self, db: Session) -> list[Collection]:
        """List all collections GLOBALLY (D-05), newest first.

        No ``session_id`` filter -- collections are a shared, global resource.
        Soft-deleted rows (``deleted_at IS NOT NULL``, D-11) are excluded.
        """
        return list(
            db.scalars(
                select(Collection)
                .where(Collection.deleted_at.is_(None))
                .order_by(desc(Collection.created_at))
            ).all()
        )

    # ---- Write methods ----

    def create(
        self,
        db: Session,
        *,
        name: str,
        author_name: str,
        job_id: uuid.UUID,
        members_config: dict[str, Any] | None = None,
        description: str | None = None,
        id: uuid.UUID | None = None,
    ) -> Collection:
        """Create a new Collection record (1:1 with a COLLECTION job).

        ``members_config`` holds the member input-definitions (D-02).
        Constraint violations (e.g. duplicate ``job_id``) are translated to
        domain exceptions.
        """
        collection = Collection(
            name=name,
            author_name=author_name,
            job_id=job_id,
            members_config=members_config,
            description=description,
        )
        if id is not None:
            collection.id = id
        try:
            db.add(collection)
            db.flush()
            db.refresh(collection)
        except IntegrityError as e:
            db.rollback()
            _handle_integrity_error(e)
        return collection

    def update_stats(
        self, db: Session, collection_id: uuid.UUID, **stats: Any
    ) -> Collection | None:
        """Update summary statistics on a collection. Returns ``None`` if absent.

        Accepts any of: ``compound_count``, ``member_failed_count`` (D-09),
        ``avg_imp_score``, ``imp_candidate_count``, ``unique_targets``.
        """
        collection = db.execute(
            select(Collection).where(Collection.id == collection_id)
        ).scalar_one_or_none()
        if collection is None:
            return None
        for field, value in stats.items():
            if hasattr(collection, field):
                setattr(collection, field, value)
        db.flush()
        return collection

    def update_storage_path(
        self, db: Session, collection_id: uuid.UUID, storage_path: str
    ) -> Collection | None:
        """Set the ZIP ``storage_path`` for a collection (D-12)."""
        collection = db.execute(
            select(Collection).where(Collection.id == collection_id)
        ).scalar_one_or_none()
        if collection is None:
            return None
        collection.storage_path = storage_path
        db.flush()
        return collection

    def soft_delete(
        self,
        db: Session,
        collection_id: uuid.UUID,
        deleted_by: uuid.UUID | None = None,
    ) -> bool:
        """Soft-delete a collection (set ``deleted_at`` / ``deleted_by``, D-11).

        Returns ``True`` if a live collection was found and marked deleted.
        Already-deleted or missing rows return ``False``.
        """
        collection = db.execute(
            select(Collection).where(
                Collection.id == collection_id,
                Collection.deleted_at.is_(None),
            )
        ).scalar_one_or_none()
        if collection is None:
            return False
        collection.deleted_at = datetime.now(timezone.utc)
        collection.deleted_by = deleted_by
        db.flush()
        return True


# Singleton instance (no model parameter -- standalone class)
collection_repo = CollectionRepository()
