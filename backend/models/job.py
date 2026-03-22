"""Job ORM model for Postgres.

Maps to the ``jobs`` table created in Phase 10 (baseline schema).
20 columns, SQLAlchemy 2.0 Mapped[] style, Postgres-native types.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.models._pg_base import PGBase
from backend.models.enums import JobStatus, JobType

__all__ = ["Job"]


class Job(PGBase):
    """Postgres ORM model for the ``jobs`` table.

    Tracks compound analysis jobs from submission through completion.
    """

    __tablename__ = "jobs"

    # Identity
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=sa.text("gen_random_uuid()"),
    )

    # Session isolation
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )

    # Input parameters
    compound_name: Mapped[str] = mapped_column(
        sa.String(255),
        nullable=False,
    )
    smiles: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
    )
    similarity_threshold: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("90"),
    )
    activity_types: Mapped[list[str] | None] = mapped_column(
        ARRAY(sa.Text),
        nullable=True,
    )

    # Status
    status: Mapped[JobStatus] = mapped_column(
        sa.Enum(
            JobStatus,
            name="job_status",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        server_default=sa.text("'pending'"),
    )
    job_type: Mapped[JobType] = mapped_column(
        sa.Enum(
            JobType,
            name="job_type",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        server_default=sa.text("'single'"),
    )

    # Batch grouping
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    batch_index: Mapped[int | None] = mapped_column(
        sa.Integer,
        nullable=True,
    )

    # Idempotency
    idempotency_key: Mapped[str | None] = mapped_column(
        sa.String(255),
        nullable=True,
    )

    # Progress tracking
    current_step: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
    )
    progress: Mapped[float | None] = mapped_column(
        sa.Float,
        nullable=True,
        server_default=sa.text("0.0"),
    )

    # Error handling
    error_message: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
    )

    # Results
    result_summary: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Timestamps (all timezone-aware)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )

    # Upload tracking (Phase 19.2 -- PENDING_UPLOAD retry)
    upload_attempts: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )
    requeue_count: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        server_default=sa.text("0"),
    )

    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "idempotency_key",
            name="uix_job_session_idempotency",
        ),
        CheckConstraint(
            "similarity_threshold BETWEEN 40 AND 100",
            name="chk_threshold_range",
        ),
    )
