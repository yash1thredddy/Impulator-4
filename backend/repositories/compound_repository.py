"""CompoundRepository - centralized Compound query patterns for Postgres.

Standalone repository using SA 2.0 select() style with parent_id/version
versioning queries. No is_duplicate/duplicate_of,
no handle_children_before_delete (Postgres trigger handles reparenting).
"""

import uuid
from typing import Any

import structlog
from sqlalchemy import delete, desc, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from backend.models.compound import Compound
from backend.models.deleted_compound import DeletedCompound
from backend.models.enums import JobType
from backend.models.job import Job
from backend.repositories.job_repository import (
    _handle_integrity_error,
)

logger = structlog.get_logger(__name__)


class CompoundRepository:
    """Repository for Compound CRUD and domain queries.

    All queries use SA 2.0 ``select()`` style.  Versioning uses
    ``parent_id`` / ``version`` columns (not ``is_duplicate`` / ``duplicate_of``).
    Child reparenting on delete is handled by a Postgres trigger --
    ``handle_children_before_delete`` is intentionally absent.
    """

    # ---- Read methods ----

    def get_by_entry_id(
        self, db: Session, entry_id: uuid.UUID
    ) -> Compound | None:
        """Get a compound by its UUID primary key."""
        return db.scalars(
            select(Compound).where(Compound.entry_id == entry_id)
        ).first()

    def get_compounds_paginated(
        self,
        db: Session,
        *,
        search: str | None = None,
        originals_only: bool | None = None,
        sort_by: str | None = None,
        sort_order: str = "desc",
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[tuple[Compound, str | None]], int]:
        """Paginated compound listing with search, filtering, and parent_name.

        Args:
            originals_only: When ``True``, filter to root compounds
                (``parent_id IS NULL``).  Replaces the old
                ``is_duplicate_filter`` parameter.

        Returns:
            ``([(compound, parent_name), ...], total_count)``
            Each element is a tuple of (Compound, parent_name) where
            parent_name is resolved via a self-join (None for root compounds).
        """
        # Self-join to resolve parent compound name
        parent = aliased(Compound, name="parent")
        # Correlated subquery: count siblings sharing the same structure key
        sibling = aliased(Compound, name="sibling")
        version_count_sub = (
            select(func.count(sibling.entry_id))
            .where(
                sibling.inchikey_structure_key == Compound.inchikey_structure_key,
                sibling.inchikey_structure_key.isnot(None),
            )
            .correlate(Compound)
            .scalar_subquery()
            .label("version_count")
        )
        base = (
            select(Compound, parent.compound_name.label("parent_name"), version_count_sub)
            .outerjoin(parent, Compound.parent_id == parent.entry_id)
        )

        # Exclude COLLECTION-member compounds from the global entries list
        # (Phase 23). Members belong to a COLLECTION job; they surface inside
        # the collection detail view, not the compound catalog. NULL-safe:
        # the ``job_id IS NULL`` arm preserves legacy/backfilled compounds that
        # have no parent job, so they are NOT silently dropped.
        base = base.outerjoin(Job, Compound.job_id == Job.id).where(
            or_(
                Compound.job_id.is_(None),
                Job.job_type != JobType.COLLECTION,
            )
        )

        # Originals-only filter (parent_id IS NULL = root compound)
        if originals_only:
            base = base.where(Compound.parent_id.is_(None))

        # Search by name, chembl_id, or smiles.
        # NOTE: %pattern% ILIKE cannot use btree indexes. For large datasets,
        # enable pg_trgm and add GIN trigram indexes:
        #   CREATE EXTENSION IF NOT EXISTS pg_trgm;
        #   CREATE INDEX idx_compounds_name_trgm ON compounds
        #       USING gin (compound_name gin_trgm_ops);
        if search:
            safe = (
                search
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            base = base.where(
                or_(
                    Compound.compound_name.ilike(f"%{safe}%", escape="\\"),
                    Compound.chembl_id.ilike(f"%{safe}%", escape="\\"),
                    Compound.smiles.ilike(f"%{safe}%", escape="\\"),
                )
            )

        # Ordering
        order_col = (
            getattr(Compound, sort_by, None) if sort_by else Compound.processed_at
        )
        if order_col is None:
            order_col = Compound.processed_at
        if sort_order == "asc":
            base = base.order_by(order_col.asc())
        else:
            base = base.order_by(desc(order_col))

        # Single query: count(*) OVER() returns total alongside rows,
        # avoiding a separate COUNT subquery round trip.
        count_window = func.count().over().label("_total")
        windowed = base.add_columns(count_window).offset(offset).limit(limit)
        rows = db.execute(windowed).all()

        if not rows:
            return [], 0

        total = rows[0]._total
        compounds = [(row[0], row[1], row[2]) for row in rows]  # (Compound, parent_name, version_count)
        return compounds, total

    def get_versions(
        self, db: Session, entry_id: uuid.UUID
    ) -> list[Compound]:
        """Get all structural siblings sharing the same InChIKey structure key.

        Single query using a correlated subquery instead of two round trips.
        Uses the ``idx_compounds_structure_key`` index for the join.
        Returns empty list if the compound has no structure key.
        """
        sk_sub = (
            select(Compound.inchikey_structure_key)
            .where(Compound.entry_id == entry_id)
            .correlate(None)
            .scalar_subquery()
        )
        return list(
            db.scalars(
                select(Compound)
                .where(
                    Compound.inchikey_structure_key == sk_sub,
                    Compound.inchikey_structure_key.isnot(None),
                )
                .order_by(Compound.processed_at.asc())
            ).all()
        )

    def find_by_structure_key(
        self, db: Session, structure_key: str
    ) -> Compound | None:
        """Find the canonical (root) compound for a structure key.

        Root compounds have ``parent_id IS NULL``.
        """
        return db.scalars(
            select(Compound).where(
                Compound.inchikey_structure_key == structure_key,
                Compound.parent_id.is_(None),
            )
        ).first()

    def find_duplicates_by_structure_key(
        self, db: Session, structure_key: str
    ) -> list[Compound]:
        """Find all compounds (root and children) matching a structure key."""
        return list(
            db.scalars(
                select(Compound).where(
                    Compound.inchikey_structure_key == structure_key
                )
            ).all()
        )

    def find_by_inchikey(
        self, db: Session, inchikey: str, for_update: bool = False
    ) -> list[Compound]:
        """Find all compounds with the given full InChIKey.

        Args:
            db: Database session
            inchikey: Full InChIKey string
            for_update: If True, acquires row-level lock (FOR UPDATE) to prevent
                concurrent compound creation race (D-27).
        """
        stmt = select(Compound).where(Compound.inchikey == inchikey)
        if for_update:
            stmt = stmt.with_for_update()
        return list(db.scalars(stmt).all())

    def find_by_inchikey_like(
        self, db: Session, prefix: str
    ) -> list[Compound]:
        """Find compounds whose InChIKey starts with *prefix*.

        Standard btree index handles ``LIKE 'prefix%'`` efficiently
        for ASCII InChIKey values.
        """
        return list(
            db.scalars(
                select(Compound).where(Compound.inchikey.like(f"{prefix}%"))
            ).all()
        )

    def find_non_duplicate_by_inchikey(
        self, db: Session, inchikey: str
    ) -> Compound | None:
        """Find a root (non-child) compound by full InChIKey.

        Filters ``parent_id IS NULL`` (replaces old ``is_duplicate == False``).
        """
        return db.scalars(
            select(Compound).where(
                Compound.inchikey == inchikey,
                Compound.parent_id.is_(None),
            )
        ).first()

    def find_by_name_no_inchikey(
        self, db: Session, compound_name: str
    ) -> Compound | None:
        """Find a compound by exact name that has no InChIKey (legacy records)."""
        return db.scalars(
            select(Compound).where(
                Compound.compound_name == compound_name,
                Compound.inchikey.is_(None),
            )
        ).first()

    def find_names_by_prefix(
        self, db: Session, prefix: str
    ) -> list[str]:
        """Find all compound names starting with *prefix* (case-insensitive)."""
        safe_prefix = (
            prefix.strip().lower()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        return list(
            db.scalars(
                select(Compound.compound_name).where(
                    func.lower(func.trim(Compound.compound_name)).like(
                        f"{safe_prefix}%", escape="\\"
                    )
                )
            ).all()
        )

    def find_existing_names(
        self, db: Session, normalized_names: list[str]
    ) -> set[str]:
        """Return set of normalized names that already exist in compounds table.

        Uses Postgres ``= ANY(ARRAY[...])`` instead of ``IN()`` for
        better query plan caching with large lists.
        Uses the expression index ``idx_compounds_name_lower``.
        """
        from sqlalchemy.dialects.postgresql import array as pg_array

        if not normalized_names:
            return set()
        rows = list(
            db.scalars(
                select(
                    func.lower(func.trim(Compound.compound_name))
                ).where(
                    func.lower(func.trim(Compound.compound_name)) == func.any_(
                        pg_array(normalized_names)
                    )
                )
            ).all()
        )
        return {r for r in rows if r}

    def find_by_name_case_insensitive(
        self, db: Session, compound_name: str
    ) -> list[Compound]:
        """Find compounds by exact case-insensitive name match, newest first."""
        return list(
            db.scalars(
                select(Compound)
                .where(
                    func.lower(func.trim(Compound.compound_name))
                    == compound_name.strip().lower()
                )
                .order_by(desc(Compound.processed_at))
            ).all()
        )

    def find_children(
        self, db: Session, parent_entry_id: uuid.UUID
    ) -> list[Compound]:
        """Find all child compounds of the given parent.

        Uses ``parent_id`` (replaces old ``duplicate_of``).
        """
        return list(
            db.scalars(
                select(Compound).where(Compound.parent_id == parent_entry_id)
            ).all()
        )

    def count_compounds(
        self,
        db: Session,
        search: str | None = None,
        originals_only: bool | None = None,
    ) -> int:
        """Count compounds with optional filtering.

        When *originals_only* is ``True``, counts only root compounds
        (``parent_id IS NULL``).
        """
        base = select(func.count(Compound.entry_id))
        if originals_only:
            base = base.where(Compound.parent_id.is_(None))
        if search:
            safe = (
                search.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            base = base.where(
                Compound.compound_name.ilike(f"%{safe}%", escape="\\")
            )
        return db.scalar(base) or 0

    # ---- Write methods (Postgres MVCC for concurrency) ----

    def create_compound(
        self, db: Session, *, parent_id: uuid.UUID | None = None, **kwargs: Any
    ) -> Compound:
        """Create a new Compound record.

        If *parent_id* is provided, auto-calculates the next ``version``
        by finding the current max version among siblings.
        Auto-derives ``inchikey_structure_key`` from ``inchikey`` when available.
        """
        if parent_id is not None:
            max_ver = db.scalar(
                select(func.max(Compound.version)).where(
                    Compound.parent_id == parent_id
                )
            ) or 1  # Parent itself is version 1
            kwargs["version"] = max_ver + 1
            kwargs["parent_id"] = parent_id

        # Auto-derive inchikey_structure_key
        inchikey = kwargs.get("inchikey")
        if inchikey and "-" in inchikey and "inchikey_structure_key" not in kwargs:
            parts = inchikey.split("-")
            if len(parts) >= 2:
                kwargs["inchikey_structure_key"] = f"{parts[0]}-{parts[1]}"

        compound = Compound(**kwargs)
        try:
            db.add(compound)
            db.flush()
        except IntegrityError as e:
            db.rollback()
            _handle_integrity_error(e)
        return compound

    def update_compound(
        self, db: Session, entry_id: uuid.UUID, **fields: Any
    ) -> Compound | None:
        """Update fields on a compound. Returns ``None`` if not found.

        Automatically recalculates ``inchikey_structure_key`` when
        ``inchikey`` is among the updated fields.
        """
        compound = db.scalars(
            select(Compound).where(Compound.entry_id == entry_id)
        ).first()
        if not compound:
            return None

        for field, value in fields.items():
            if hasattr(compound, field):
                setattr(compound, field, value)

        # Recalculate structure key if inchikey changed
        if "inchikey" in fields and fields["inchikey"] and "-" in fields["inchikey"]:
            parts = fields["inchikey"].split("-")
            if len(parts) >= 2:
                compound.inchikey_structure_key = f"{parts[0]}-{parts[1]}"

        db.flush()
        return compound

    def delete_by_entry_id(self, db: Session, entry_id: uuid.UUID) -> bool:
        """Delete a compound by entry_id. Returns True if a row was deleted."""
        result = db.execute(
            delete(Compound).where(Compound.entry_id == entry_id)
        )
        return result.rowcount > 0

    def delete_compound(self, db: Session, compound: Compound) -> None:
        """Delete a compound record.

        Child reparenting is handled by the Postgres trigger
        ``trg_reparent_on_delete`` -- no manual reparenting needed.
        """
        db.delete(compound)
        db.flush()

    def archive_compound(
        self,
        db: Session,
        compound: Compound,
        deleted_by: uuid.UUID | None = None,
        deletion_reason: str = "user_request",
    ) -> DeletedCompound:
        """Create a DeletedCompound audit record from a live compound.

        Maps all fields to the new DeletedCompound schema:
        - ``entry_id`` (not ``original_id``)
        - ``parent_id`` / ``version`` / ``config_diff`` (not ``is_duplicate`` / ``duplicate_of``)
        - ``deleted_by`` as UUID (not ``deleted_by_session`` string)
        - No ``deleted_by_job_id`` (column removed)
        """
        deleted_record = DeletedCompound(
            entry_id=compound.entry_id,
            job_id=compound.job_id,
            compound_name=compound.compound_name,
            chembl_id=compound.chembl_id,
            smiles=compound.smiles,
            canonical_smiles=compound.canonical_smiles,
            inchikey=compound.inchikey,
            inchikey_structure_key=compound.inchikey_structure_key,
            parent_id=compound.parent_id,
            version=compound.version,
            config_diff=compound.config_diff,
            imp_score=compound.imp_score,
            similar_compounds=compound.similar_compounds,
            total_activities=compound.total_activities,
            imp_candidates=compound.imp_candidates,
            qed=compound.qed,
            num_outliers=compound.num_outliers,
            similarity_threshold=compound.similarity_threshold,
            activity_types=compound.activity_types,
            author_name=compound.author_name,
            storage_path=compound.storage_path,
            original_processed_at=compound.processed_at,
            deleted_by=deleted_by,
            deletion_reason=deletion_reason,
        )
        db.add(deleted_record)
        db.flush()
        return deleted_record


# Singleton instance (no model parameter -- standalone class)
compound_repo = CompoundRepository()
