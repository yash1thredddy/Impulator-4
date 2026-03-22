"""
Audit logging for security-relevant events.

Provides a dedicated audit log for security events that may need
to be reviewed separately from application logs. Events are logged
with structured data for easy analysis.

The audit logger's handlers (console + file + audit_file) are configured
centrally in logging.py via dictConfig. This module only needs to obtain
the structlog logger and emit events with `audit=True` flag.

Security Events Logged:
- Rate limit exceeded
- Job cancellations
- Job deletions
- Validation failures
- Path traversal attempts
- Authentication/authorization failures
"""
from enum import Enum
from typing import Any

import structlog


class AuditEvent(str, Enum):
    """Types of security-relevant events."""
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


# Audit logger -- handlers configured in logging.py dictConfig (console + file + audit_file)
audit_logger = structlog.get_logger("audit")


def log_security_event(
    event: AuditEvent,
    session_id: str | None = None,
    details: dict[str, Any] | None = None,
    severity: str = "warning",
) -> None:
    """
    Log a security-relevant event to the audit log.

    Args:
        event: Type of security event
        session_id: Session ID of the user (if available)
        details: Additional details about the event
        severity: Log level (info, warning, error, critical)
    """
    log_method = getattr(audit_logger, severity, audit_logger.warning)
    log_method(
        event.value,
        audit=True,
        session_id=session_id or "anonymous",
        **(details or {}),
    )


def log_rate_limit_exceeded(
    session_id: str,
    limit_type: str,
    limit_value: int
) -> None:
    """Log when a rate limit is exceeded."""
    log_security_event(
        AuditEvent.RATE_LIMIT_EXCEEDED,
        session_id=session_id,
        details={
            "limit_type": limit_type,
            "limit_value": limit_value,
        },
        severity="warning"
    )


def log_job_cancelled(
    session_id: str,
    job_id: str,
    compound_name: str | None = None
) -> None:
    """Log when a job is cancelled."""
    log_security_event(
        AuditEvent.JOB_CANCELLED,
        session_id=session_id,
        details={
            "job_id": job_id,
            "compound_name": compound_name,
        },
        severity="info"
    )


def log_job_deleted(
    session_id: str,
    job_id: str,
    compound_name: str | None = None
) -> None:
    """Log when a job and its results are deleted."""
    log_security_event(
        AuditEvent.JOB_DELETED,
        session_id=session_id,
        details={
            "job_id": job_id,
            "compound_name": compound_name,
        },
        severity="info"
    )


def log_validation_failed(
    session_id: str,
    field: str,
    value: str,
    reason: str
) -> None:
    """Log when input validation fails."""
    # Truncate value to avoid logging very long inputs
    truncated_value = value[:100] + "..." if len(value) > 100 else value
    log_security_event(
        AuditEvent.VALIDATION_FAILED,
        session_id=session_id,
        details={
            "field": field,
            "value": truncated_value,
            "reason": reason,
        },
        severity="warning"
    )


def log_path_traversal_blocked(
    attempted_path: str
) -> None:
    """Log when a path traversal attempt is blocked."""
    log_security_event(
        AuditEvent.PATH_TRAVERSAL_BLOCKED,
        details={
            "attempted_path": attempted_path,
        },
        severity="error"
    )


def log_suspicious_input(
    session_id: str,
    field: str,
    pattern_matched: str
) -> None:
    """Log when potentially malicious input is detected."""
    log_security_event(
        AuditEvent.SUSPICIOUS_INPUT,
        session_id=session_id,
        details={
            "field": field,
            "pattern_matched": pattern_matched,
        },
        severity="warning"
    )
