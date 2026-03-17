"""
Rate limiting for API endpoints.

Provides a RateLimiter class and FastAPI injectable dependencies
for per-session, per-route rate limiting.
"""
import time
import threading
from collections import defaultdict
from typing import Tuple

from fastapi import Depends, HTTPException

from backend.core.auth import validate_session_id


# Rate limiting configuration
RATE_LIMIT_WINDOW_SECONDS = 60  # 1 minute window
RATE_LIMIT_MAX_JOBS = 10  # Max 10 single jobs per minute per session
RATE_LIMIT_MAX_BATCH = 3  # Max 3 batch submissions per minute per session


class RateLimiter:
    """Simple in-memory rate limiter per session.

    Thread-safe implementation using defaultdict and locks.
    Automatically cleans up old entries to prevent memory leaks.
    Limited to MAX_SESSIONS to prevent unbounded growth.
    """
    MAX_SESSIONS = 10000  # Prevent unbounded memory growth

    def __init__(self, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self._requests: dict = defaultdict(list)  # session_id -> [timestamps]
        self._lock = threading.Lock()
        self._window_seconds = window_seconds

    def _cleanup_session(self, session_id: str, now: float) -> None:
        """Clean up old timestamps for a specific session."""
        cutoff = now - self._window_seconds
        if session_id in self._requests:
            self._requests[session_id] = [
                t for t in self._requests[session_id] if t > cutoff
            ]
            if not self._requests[session_id]:
                del self._requests[session_id]

    def _evict_oldest_session(self) -> None:
        """Evict the session with oldest activity when at capacity."""
        if not self._requests:
            return
        oldest = min(
            self._requests.keys(),
            key=lambda k: min(self._requests[k]) if self._requests[k] else float('inf')
        )
        del self._requests[oldest]

    @property
    def active_session_count(self) -> int:
        """Return the number of active sessions being tracked."""
        return len(self._requests)

    def check_rate_limit(self, session_id: str, limit: int) -> tuple[bool, int]:
        """Check if request is within rate limit.

        Args:
            session_id: Session identifier (or IP if no session)
            limit: Maximum requests allowed in window

        Returns:
            Tuple of (allowed: bool, remaining: int)
        """
        if not session_id:
            session_id = "anonymous"

        with self._lock:
            now = time.time()

            # Clean this session's old entries
            self._cleanup_session(session_id, now)

            # Check session limit to prevent memory leak
            if len(self._requests) >= self.MAX_SESSIONS and session_id not in self._requests:
                # Evict oldest session to make room
                self._evict_oldest_session()

            timestamps = self._requests.get(session_id, [])

            if len(timestamps) >= limit:
                return False, 0

            # Add new timestamp
            if session_id not in self._requests:
                self._requests[session_id] = []
            self._requests[session_id].append(now)

            return True, limit - len(timestamps) - 1


# Module-level singleton
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """FastAPI dependency that returns the global RateLimiter instance."""
    return _rate_limiter


def rate_limit(limit: int = RATE_LIMIT_MAX_JOBS, key_suffix: str = ""):
    """Factory for per-route rate limit dependency.

    Args:
        limit: Max requests per window per session
        key_suffix: Optional suffix for session key (e.g., "_batch")
    """
    def dependency(
        session_id: str = Depends(validate_session_id),
        limiter: RateLimiter = Depends(get_rate_limiter),
    ):
        effective_key = f"{session_id}{key_suffix}" if key_suffix else session_id
        allowed, remaining = limiter.check_rate_limit(effective_key, limit)
        if not allowed:
            from backend.core.audit import log_rate_limit_exceeded
            from backend.core.auth import truncate_session_id
            log_rate_limit_exceeded(truncate_session_id(session_id), f"rate_limit_{limit}", limit)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {limit} requests per minute.",
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )
    return Depends(dependency)
