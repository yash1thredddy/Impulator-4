"""AuditEvent ORM model for Postgres.

Maps to the ``audit_events`` table created in Phase 12 (baseline schema).
6 columns, BIGINT IDENTITY PK. Decoupled -- no FKs to jobs or compounds.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Identity
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.models._pg_base import PGBase
from backend.models.enums import AuditEventType

__all__ = ["AuditEvent"]


class AuditEvent(PGBase):
    """Postgres ORM model for the ``audit_events`` table.

    Structured audit trail for security and lifecycle events.
    """

    __tablename__ = "audit_events"

    # Identity (BIGINT IDENTITY)
    id: Mapped[int] = mapped_column(
        sa.BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    # Event classification
    event_type: Mapped[AuditEventType] = mapped_column(
        sa.Enum(
            AuditEventType,
            name="audit_event_type",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )

    # Actor
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # Severity
    severity: Mapped[str] = mapped_column(
        sa.String(20),
        nullable=False,
        server_default=sa.text("'warning'"),
    )

    # Payload
    details: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
