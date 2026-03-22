"""Dual audit logging helper -- writes to both audit_events table and file-based audit log.

Inline helper rather than a dedicated AuditEventRepository because audit writes
are simple INSERT-only operations. A full repo class for a single db.add() would
be over-engineering.
"""

import uuid
import logging

from sqlalchemy.orm import Session

from backend.models.audit import AuditEvent
from backend.models.enums import AuditEventType

logger = logging.getLogger(__name__)

__all__ = ["log_audit_event"]


def log_audit_event(
    db: Session,
    event_type: AuditEventType,
    *,
    session_id: uuid.UUID | None = None,
    severity: str = "info",
    details: dict | None = None,
) -> None:
    """Write audit event to DB audit_events table AND existing file-based logger.

    The DB write is added to the current transaction (no flush/commit here --
    the enclosing service method commits). If the DB write fails, the file
    audit still runs so we never lose audit trail completely.

    Args:
        db: Active SQLAlchemy session (will be flushed/committed by caller).
        event_type: One of the 13 AuditEventType enum values.
        session_id: Actor's session UUID (None for system events).
        severity: One of 'info', 'warning', 'error', 'critical'.
        details: Arbitrary JSONB payload (job_id, compound_name, reason, etc.).
    """
    # DB audit write
    try:
        event = AuditEvent(
            event_type=event_type,
            session_id=session_id,
            severity=severity,
            details=details or {},
        )
        db.add(event)
        # Don't flush here -- let the enclosing transaction handle it
    except Exception as e:
        logger.warning(f"Failed to write DB audit event ({event_type.value}): {e}")

    # File-based audit write (existing infrastructure)
    try:
        from backend.core.audit import AuditEvent as FileAuditEvent, log_security_event
        # Map DB enum name to file-based enum (they share the same names)
        file_event = getattr(FileAuditEvent, event_type.name, None)
        if file_event:
            log_security_event(
                file_event,
                session_id=str(session_id) if session_id else None,
                details=details,
                severity=severity,
            )
    except Exception as e:
        logger.warning(f"Failed to write file audit event ({event_type.value}): {e}")
