"""Session validation for API endpoints.

Provides session ID validation to ensure only valid UUID-formatted
session IDs are accepted, preventing injection attacks.
"""
import re
import uuid
import logging
from fastapi import Header, HTTPException
from typing import Optional

logger = logging.getLogger(__name__)

# Valid session ID format: UUID v4
SESSION_ID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    re.IGNORECASE
)


def validate_session_id(x_session_id: Optional[str] = Header(None, alias="X-Session-ID")) -> str:
    """Validate and return session ID.

    Ensures session ID is a valid UUID format to prevent injection.
    If no session ID is provided, generates an anonymous session.

    Args:
        x_session_id: Session ID from X-Session-ID header

    Returns:
        Validated session ID string

    Raises:
        HTTPException: If session ID format is invalid
    """
    if not x_session_id:
        # Generate anonymous session for unauthenticated requests
        return f"anon-{uuid.uuid4()}"

    # Strip whitespace
    x_session_id = x_session_id.strip()

    # Validate format
    if not SESSION_ID_PATTERN.match(x_session_id):
        # Log with truncated session ID for security
        logger.warning(f"Invalid session ID format: {x_session_id[:8]}...")
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID format. Must be a valid UUID."
        )

    return x_session_id


def truncate_session_id(session_id: Optional[str]) -> str:
    """Truncate session ID for safe logging.

    Args:
        session_id: Full session ID

    Returns:
        Truncated session ID (first 8 chars + "...")
    """
    if not session_id:
        return "unknown"
    return f"{session_id[:8]}..."
