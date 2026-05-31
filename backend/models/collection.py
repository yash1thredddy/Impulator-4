"""Collection ORM model for Postgres.

Maps to the ``collections`` table created in migration 0007 (Phase 23).
SQLAlchemy 2.0 ``Mapped[]`` style, Postgres-native types, sync-first (HC-1 —
no async sessions referenced in this model).

A collection is ONE :class:`~backend.models.job.Job` of
:class:`~backend.models.enums.JobType.COLLECTION` plus this dedicated row.

Key design decisions:
- **D-02**: member input-definitions live in the ``members_config`` JSONB
  column (same shape as ``job.result_summary``), NOT in ``job.result_summary``.
- **D-07**: ``name`` is indexed but NOT unique.
- **D-09**: ``member_failed_count`` tracks members that failed compute without
  failing the parent job.
- **D-11**: soft-delete via ``deleted_at`` / ``deleted_by``.
- No ``status`` column — status is derived from the linked job.
- No member junction table — members are ``compounds WHERE job_id = collections.job_id``.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.models._pg_base import PGBase

__all__ = ["Collection"]


class Collection(PGBase):
    """Postgres ORM model for the ``collections`` table.

    1:1 with a ``JobType.COLLECTION`` job via the unique ``job_id`` FK.
    """

    __tablename__ = "collections"

    # Identity
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    )

    # Descriptive metadata
    name: Mapped[str] = mapped_column(
        sa.String(255),
        nullable=False,
        index=True,  # D-07: indexed, NOT unique
    )
    description: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
    )
    author_name: Mapped[str] = mapped_column(
        sa.String(100),
        nullable=False,
    )

    # Linked job (1:1). Members are compounds sharing this job_id.
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # D-02: member input-definitions (same shape as job.result_summary)
    members_config: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Storage
    storage_path: Mapped[str | None] = mapped_column(
        sa.String(500),
        nullable=True,
    )

    # Summary statistics
    compound_count: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )
    member_failed_count: Mapped[int] = mapped_column(  # D-09
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )
    avg_imp_score: Mapped[float | None] = mapped_column(
        sa.Float,
        nullable=True,
    )
    imp_candidate_count: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )
    unique_targets: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )

    # Soft delete (D-11)
    deleted_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    deleted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # Timestamps (timezone-aware)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
