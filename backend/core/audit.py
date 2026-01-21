"""
Audit logging for security-relevant events.

Provides a dedicated audit log for security events that may need
to be reviewed separately from application logs. Events are logged
with structured data for easy analysis.

Security Events Logged:
- Rate limit exceeded
- Job cancellations
- Job deletions
- Validation failures
- Path traversal attempts
- Authentication/authorization failures
"""
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import settings


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


# Configure audit logger
AUDIT_LOG_DIR = Path(settings.DATA_DIR) / "logs"
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Separate audit logger with its own file
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # Don't propagate to root logger

# Create file handler for audit log
audit_handler = logging.FileHandler(
    AUDIT_LOG_DIR / "audit.log",
    encoding="utf-8"
)
audit_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(message)s"
))
audit_logger.addHandler(audit_handler)


def log_security_event(
    event: AuditEvent,
    session_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    severity: str = "warning"
) -> None:
    """
    Log a security-relevant event to the audit log.

    Args:
        event: Type of security event
        session_id: Session ID of the user (if available)
        details: Additional details about the event
        severity: Log level (info, warning, error, critical)
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event.value,
        "session_id": session_id or "anonymous",
        "details": details or {},
    }

    message = json.dumps(log_entry)

    if severity == "critical":
        audit_logger.critical(message)
    elif severity == "error":
        audit_logger.error(message)
    elif severity == "warning":
        audit_logger.warning(message)
    else:
        audit_logger.info(message)


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
    compound_name: Optional[str] = None
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
    compound_name: Optional[str] = None
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
