"""Session and admin authentication for API endpoints.

Provides:
- Session ID validation to ensure only valid UUID-formatted session IDs are accepted
- Admin API key authentication for sensitive endpoints (migrations, etc.)
"""
import re
import uuid
import hmac
import logging
from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# Admin API key header
admin_api_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)

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


def verify_admin_api_key(
    api_key: Optional[str] = Security(admin_api_key_header)
) -> bool:
    """Verify admin API key for protected endpoints.

    This dependency should be used on sensitive admin endpoints like
    database migrations, cache clearing, etc.

    Args:
        api_key: API key from X-Admin-API-Key header

    Returns:
        True if valid

    Raises:
        HTTPException: If API key is missing, not configured, or invalid
    """
    configured_key = settings.ADMIN_API_KEY

    # Check if admin key is configured
    if not configured_key:
        logger.error("Admin API key not configured - admin endpoints disabled")
        raise HTTPException(
            status_code=503,
            detail="Admin endpoints are disabled. Configure ADMIN_API_KEY to enable."
        )

    # Check if key was provided
    if not api_key:
        logger.warning("Admin endpoint accessed without API key")
        raise HTTPException(
            status_code=401,
            detail="Admin API key required. Provide X-Admin-API-Key header."
        )

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(api_key, configured_key):
        logger.warning("Invalid admin API key attempted")
        raise HTTPException(
            status_code=403,
            detail="Invalid admin API key."
        )

    logger.info("Admin API key verified successfully")
    return True
