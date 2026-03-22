"""Unified enum definitions for Postgres ORM models.

All enums inherit from (str, enum.Enum) so they serialize as strings.
Each enum maps 1:1 to a Postgres ENUM type created in the baseline schema.
"""

import enum

__all__ = ["JobStatus", "JobType", "AuditEventType"]


class JobStatus(str, enum.Enum):
    """Matches Postgres ``job_status`` ENUM type.

    Six states: pending -> processing -> pending_upload -> completed | failed | cancelled.
    The ``pending_upload`` state means the compound ZIP is written locally and
    the compound entry exists in the database, but Azure upload has not yet
    succeeded.  From the user's perspective results are already viewable.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    PENDING_UPLOAD = "pending_upload"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    """Matches Postgres ``job_type`` ENUM type."""

    SINGLE = "single"
    BATCH = "batch"


class AuditEventType(str, enum.Enum):
    """Matches Postgres ``audit_event_type`` ENUM type.

    13 values: 10 security events + 3 lifecycle events.
    """

    # Security events (10)
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    JOB_CANCELLED = "job_cancelled"
    JOB_DELETED = "job_deleted"
    BATCH_CANCELLED = "batch_cancelled"
    VALIDATION_FAILED = "validation_failed"
    PATH_TRAVERSAL_BLOCKED = "path_traversal_blocked"
    INVALID_SMILES = "invalid_smiles"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    SUSPICIOUS_INPUT = "suspicious_input"

    # Lifecycle events (3)
    COMPOUND_CREATED = "compound_created"
    COMPOUND_DELETED = "compound_deleted"
    JOB_CREATED = "job_created"
