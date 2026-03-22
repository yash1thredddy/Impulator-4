"""DeletedCompound ORM model for Postgres.

Maps to the ``deleted_compounds`` archive table created in Phase 11.
26 columns, BIGINT IDENTITY PK. Decoupled archive -- no foreign keys.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Identity
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.models._pg_base import PGBase

__all__ = ["DeletedCompound"]


class DeletedCompound(PGBase):
    """Postgres ORM model for the ``deleted_compounds`` table.

    Archive of deleted compounds for audit trail and recovery.
    No foreign keys -- this is a decoupled archive.
    All mirrored columns are nullable (archive -- no constraints).
    """

    __tablename__ = "deleted_compounds"

    # Archive identity (BIGINT IDENTITY)
    id: Mapped[int] = mapped_column(
        sa.BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    # Original compound data (mirrored from compounds table)
    entry_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    compound_name: Mapped[str] = mapped_column(
        sa.String(255),
        nullable=False,
    )
    chembl_id: Mapped[str | None] = mapped_column(
        sa.String(50),
        nullable=True,
    )
    smiles: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
    )
    canonical_smiles: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
    )
    inchikey: Mapped[str | None] = mapped_column(
        sa.String(27),
        nullable=True,
    )
    inchikey_structure_key: Mapped[str | None] = mapped_column(
        sa.String(25),
        nullable=True,
    )

    # Versioning state at time of deletion
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    version: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
    )
    config_diff: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Analysis results
    imp_score: Mapped[float | None] = mapped_column(
        sa.Float,
        nullable=True,
    )
    similar_compounds: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
    )
    total_activities: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
    )
    imp_candidates: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
    )
    qed: Mapped[float | None] = mapped_column(
        sa.Float,
        nullable=True,
    )
    num_outliers: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
    )

    # Analysis config
    similarity_threshold: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
    )
    activity_types: Mapped[list[str] | None] = mapped_column(
        ARRAY(sa.Text),
        nullable=True,
    )

    # Metadata
    author_name: Mapped[str | None] = mapped_column(
        sa.String(100),
        nullable=True,
    )
    storage_path: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
    )

    # Original timestamp
    original_processed_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )

    # Deletion metadata
    deleted_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    deleted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    deletion_reason: Mapped[str | None] = mapped_column(
        sa.String(255),
        nullable=True,
    )
