"""Compound ORM model for Postgres.

Maps to the ``compounds`` table created in Phase 11 (baseline schema).
22 columns, entry_id UUID as primary key (no integer id).
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.models._pg_base import PGBase

__all__ = ["Compound"]


class Compound(PGBase):
    """Postgres ORM model for the ``compounds`` table.

    Stores processed compound metadata. Uses entry_id (UUID) as primary key
    instead of an integer id. Supports versioning via parent_id self-FK.
    """

    __tablename__ = "compounds"

    # Identity (entry_id is PK, NOT an integer id)
    entry_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    )

    # Job link (1:1)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=True,
    )

    # Compound data
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

    # InChIKey
    inchikey: Mapped[str | None] = mapped_column(
        sa.String(27),
        nullable=True,
    )
    inchikey_structure_key: Mapped[str | None] = mapped_column(
        sa.String(25),
        nullable=True,
    )

    # Versioning (replaces is_duplicate/duplicate_of)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("compounds.entry_id", ondelete="NO ACTION"),
        nullable=True,
    )
    version: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("1"),
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
    similar_compounds: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )
    total_activities: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )
    imp_candidates: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )
    qed: Mapped[float | None] = mapped_column(
        sa.Float,
        nullable=True,
    )
    num_outliers: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )

    # Analysis config
    similarity_threshold: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("90"),
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

    # Timestamps
    processed_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "entry_id != parent_id",
            name="chk_no_self_parent",
        ),
        CheckConstraint(
            "similarity_threshold BETWEEN 40 AND 100",
            name="chk_compound_threshold_range",
        ),
        CheckConstraint(
            "version >= 1",
            name="chk_version_positive",
        ),
        CheckConstraint(
            "(version = 1 AND parent_id IS NULL) OR (version > 1 AND parent_id IS NOT NULL)",
            name="chk_root_has_no_parent",
        ),
    )
