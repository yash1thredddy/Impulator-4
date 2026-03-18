"""CompoundRepository - centralized Compound query patterns (ARCH-01).

Replaces all db.query(Compound) calls scattered across compounds.py,
jobs.py, and job_service.py.
"""
from datetime import datetime
from typing import Any, List, Optional, Tuple

import structlog
from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

from backend.models.database import Compound, DeletedCompound
from backend.repositories.base import BaseRepository, _db_write_lock

logger = structlog.get_logger(__name__)


class CompoundRepository(BaseRepository[Compound]):
    """Repository for Compound CRUD and domain queries."""

    # ---- Read methods (no lock) ----

    def get_by_entry_id(self, db: Session, entry_id: str) -> Optional[Compound]:
        """Get a compound by its unique entry UUID."""
        return db.query(Compound).filter(Compound.entry_id == entry_id).first()

    def get_compounds_paginated(
        self,
        db: Session,
        *,
        search: Optional[str] = None,
        is_duplicate_filter: Optional[bool] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Compound], int]:
        """Paginated compound listing with search and filtering.

        Returns (compounds, total_count).
        """
        query = db.query(Compound)

        # Duplicate filter
        if is_duplicate_filter is not None:
            if is_duplicate_filter:
                pass  # include all
            else:
                query = query.filter(Compound.is_duplicate == False)  # noqa: E712

        # Search by name, chembl_id, or smiles
        if search:
            search_escaped = (
                search
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            query = query.filter(
                or_(
                    Compound.compound_name.ilike(f"%{search_escaped}%", escape="\\"),
                    Compound.chembl_id.ilike(f"%{search_escaped}%", escape="\\"),
                    Compound.smiles.ilike(f"%{search_escaped}%", escape="\\"),
                )
            )

        total = query.count()

        # Ordering
        order_col = getattr(Compound, sort_by, None) if sort_by else Compound.processed_at
        if order_col is None:
            order_col = Compound.processed_at
        if sort_order == "asc":
            query = query.order_by(order_col.asc())
        else:
            query = query.order_by(desc(order_col))

        compounds = query.offset(offset).limit(limit).all()
        return compounds, total

    def get_versions(self, db: Session, entry_id: str) -> List[Compound]:
        """Get all structural siblings sharing the same InChIKey structure key.

        Returns empty list if the compound has no InChIKey or no siblings.
        """
        compound = db.query(Compound).filter(Compound.entry_id == entry_id).first()
        if not compound:
            return []

        inchikey = compound.inchikey
        if not inchikey or "-" not in inchikey:
            return []

        parts = inchikey.split("-")
        if len(parts) < 2:
            return []
        structure_key = f"{parts[0]}-{parts[1]}"

        siblings = (
            db.query(Compound)
            .filter(Compound.inchikey.like(f"{structure_key}%"))
            .order_by(Compound.processed_at.asc())
            .all()
        )
        return siblings

    def find_by_structure_key(self, db: Session, structure_key: str) -> Optional[Compound]:
        """Find the canonical (non-duplicate) compound for a structure key."""
        return (
            db.query(Compound)
            .filter(
                Compound.inchikey_structure_key == structure_key,
                Compound.is_duplicate == False,  # noqa: E712
            )
            .first()
        )

    def find_duplicates_by_structure_key(self, db: Session, structure_key: str) -> List[Compound]:
        """Find all compounds (including duplicates) matching a structure key."""
        return (
            db.query(Compound)
            .filter(Compound.inchikey_structure_key == structure_key)
            .all()
        )

    def find_by_inchikey(self, db: Session, inchikey: str) -> List[Compound]:
        """Find all compounds with the given full InChIKey."""
        return db.query(Compound).filter(Compound.inchikey == inchikey).all()

    def find_by_inchikey_like(self, db: Session, prefix: str) -> List[Compound]:
        """Find compounds whose InChIKey starts with prefix (structure key match)."""
        return db.query(Compound).filter(Compound.inchikey.like(f"{prefix}%")).all()

    def find_non_duplicate_by_inchikey(self, db: Session, inchikey: str) -> Optional[Compound]:
        """Find a non-duplicate compound by full InChIKey."""
        return (
            db.query(Compound)
            .filter(Compound.inchikey == inchikey, Compound.is_duplicate == False)  # noqa: E712
            .first()
        )

    def find_by_name_no_inchikey(self, db: Session, compound_name: str) -> Optional[Compound]:
        """Find a compound by exact name that has no InChIKey (legacy records)."""
        return (
            db.query(Compound)
            .filter(
                Compound.compound_name == compound_name,
                Compound.inchikey.is_(None),
            )
            .first()
        )

    def find_names_by_prefix(self, db: Session, prefix: str) -> List[str]:
        """Find all compound names starting with the given prefix (case-insensitive)."""
        from sqlalchemy import func as sqla_func
        # Escape LIKE metacharacters so prefix values containing '%' or '_'
        # are treated as literals rather than wildcards.
        safe_prefix = (
            prefix.strip().lower()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        rows = (
            db.query(Compound.compound_name)
            .filter(
                sqla_func.lower(sqla_func.trim(Compound.compound_name))
                .like(f"{safe_prefix}%", escape="\\")
            )
            .all()
        )
        return [row[0] for row in rows if row[0]]

    def find_existing_names(self, db: Session, normalized_names: list) -> set:
        """Return set of normalized names that already exist in compounds table."""
        from sqlalchemy import func as sqla_func
        rows = (
            db.query(sqla_func.lower(sqla_func.trim(Compound.compound_name)))
            .filter(sqla_func.lower(sqla_func.trim(Compound.compound_name)).in_(normalized_names))
            .all()
        )
        return {row[0] for row in rows if row[0]}

    def find_by_name_case_insensitive(self, db: Session, compound_name: str) -> List[Compound]:
        """Find compounds by exact case-insensitive name match, newest first."""
        from sqlalchemy import func as sqla_func
        return (
            db.query(Compound)
            .filter(
                sqla_func.lower(sqla_func.trim(Compound.compound_name))
                == compound_name.strip().lower()
            )
            .order_by(desc(Compound.processed_at))
            .all()
        )

    def find_children(self, db: Session, parent_entry_id: str) -> List[Compound]:
        """Find all compounds that are duplicates of the given entry_id."""
        return (
            db.query(Compound)
            .filter(Compound.duplicate_of == parent_entry_id)
            .all()
        )

    def count_compounds(
        self,
        db: Session,
        search: Optional[str] = None,
        is_duplicate_filter: Optional[bool] = None,
    ) -> int:
        """Count compounds with optional filtering."""
        query = db.query(Compound)
        if is_duplicate_filter is not None and not is_duplicate_filter:
            query = query.filter(Compound.is_duplicate == False)  # noqa: E712
        if search:
            search_escaped = (
                search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            query = query.filter(
                Compound.compound_name.ilike(f"%{search_escaped}%", escape="\\")
            )
        return query.count()

    # ---- Write methods (with _db_write_lock) ----

    def create_compound(self, db: Session, **kwargs: Any) -> Compound:
        """Write-locked compound creation.

        Automatically sets inchikey_structure_key from inchikey if provided.
        """
        inchikey = kwargs.get("inchikey")
        if inchikey and "-" in inchikey and "inchikey_structure_key" not in kwargs:
            parts = inchikey.split("-")
            if len(parts) >= 2:
                kwargs["inchikey_structure_key"] = f"{parts[0]}-{parts[1]}"

        compound = Compound(**kwargs)
        with _db_write_lock:
            db.add(compound)
            db.flush()
        return compound

    def update_compound(self, db: Session, entry_id: str, **fields: Any) -> Optional[Compound]:
        """Write-locked field update on a compound."""
        with _db_write_lock:
            compound = db.query(Compound).filter(Compound.entry_id == entry_id).first()
            if not compound:
                return None
            for field, value in fields.items():
                if hasattr(compound, field):
                    setattr(compound, field, value)

            # Update inchikey_structure_key if inchikey changed
            if "inchikey" in fields and fields["inchikey"] and "-" in fields["inchikey"]:
                parts = fields["inchikey"].split("-")
                if len(parts) >= 2:
                    compound.inchikey_structure_key = f"{parts[0]}-{parts[1]}"

            db.flush()
        return compound

    def delete_compound(self, db: Session, compound: Compound) -> None:
        """Write-locked compound deletion."""
        with _db_write_lock:
            db.delete(compound)
            db.flush()

    def archive_compound(
        self,
        db: Session,
        compound: Compound,
        session_id: Optional[str] = None,
        deletion_reason: str = "user_request",
        job_id: Optional[str] = None,
    ) -> DeletedCompound:
        """Create a DeletedCompound audit record from a live compound.

        NOTE: Does NOT acquire _db_write_lock — callers must hold the lock
        when calling this method (e.g., delete_compound wraps the entire
        archive+delete sequence in a single lock acquisition).
        """
        deleted_record = DeletedCompound(
            original_id=compound.id,
            entry_id=compound.entry_id,
            compound_name=compound.compound_name,
            chembl_id=compound.chembl_id,
            smiles=compound.smiles,
            inchikey=compound.inchikey,
            author_name=compound.author_name,
            is_duplicate=compound.is_duplicate,
            duplicate_of=compound.duplicate_of,
            activity_types=compound.activity_types,
            storage_path=compound.storage_path,
            deleted_by_session=session_id,
            deleted_by_job_id=job_id,
            deletion_reason=deletion_reason,
            original_processed_at=compound.processed_at,
        )
        db.add(deleted_record)
        db.flush()
        return deleted_record

    def handle_children_before_delete(self, db: Session, entry_id: str) -> int:
        """Promote oldest child and reparent remaining children.

        If the compound is a main compound (not a duplicate), promotes
        the oldest child to main and re-points remaining children.
        Returns the count of reparented children.

        Args:
            db: Database session (caller must commit after)
            entry_id: entry_id of the compound about to be deleted
        """
        compound = db.query(Compound).filter(Compound.entry_id == entry_id).first()
        if not compound or compound.is_duplicate:
            return 0

        children = (
            db.query(Compound)
            .filter(Compound.duplicate_of == entry_id)
            .all()
        )
        if not children:
            return 0

        # Sort by processed_at ascending (oldest first)
        children_sorted = sorted(
            children,
            key=lambda c: c.processed_at or datetime.min,
        )

        # Promote oldest child to main
        promoted = children_sorted[0]
        promoted.is_duplicate = False
        promoted.duplicate_of = None
        logger.info(
            "promoted_child",
            promoted_name=promoted.compound_name,
            promoted_entry_id=promoted.entry_id,
            was_duplicate_of=entry_id,
        )

        reparented = 0
        # Re-point remaining children to the promoted compound
        for child in children_sorted[1:]:
            child.duplicate_of = promoted.entry_id
            reparented += 1
            logger.info(
                "reparented_child",
                child_name=child.compound_name,
                child_entry_id=child.entry_id,
                new_parent=promoted.entry_id,
            )

        return reparented + 1  # promoted + reparented


# Singleton instance
compound_repo = CompoundRepository(Compound)
