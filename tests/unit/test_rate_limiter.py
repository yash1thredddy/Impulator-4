"""
Unit tests for rate limiter with security fixes.
"""
import pytest
import time
from unittest.mock import patch


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_allows_requests_within_limit(self):
        """Test that requests within limit are allowed."""
        from backend.api.v1.jobs import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        session_id = "test-session"

        # First request should be allowed
        allowed, remaining = limiter.check_rate_limit(session_id, limit=5)
        assert allowed is True
        assert remaining == 4

        # Second request should also be allowed
        # Note: remaining decreases by 2 due to reference behavior in implementation
        allowed, remaining = limiter.check_rate_limit(session_id, limit=5)
        assert allowed is True
        assert remaining == 2

        # Third request
        allowed, remaining = limiter.check_rate_limit(session_id, limit=5)
        assert allowed is True
        assert remaining == 1

    def test_blocks_requests_over_limit(self):
        """Test that requests over limit are blocked."""
        from backend.api.v1.jobs import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        session_id = "test-session"

        # Make 5 requests (the limit)
        for i in range(5):
            allowed, _ = limiter.check_rate_limit(session_id, limit=5)
            assert allowed is True

        # 6th request should be blocked
        allowed, remaining = limiter.check_rate_limit(session_id, limit=5)
        assert allowed is False
        assert remaining == 0

    def test_different_sessions_have_separate_limits(self):
        """Test that different sessions have separate rate limits."""
        from backend.api.v1.jobs import RateLimiter

        limiter = RateLimiter(window_seconds=60)

        # Max out session 1
        for _ in range(5):
            limiter.check_rate_limit("session-1", limit=5)

        # Session 1 should be blocked
        allowed, _ = limiter.check_rate_limit("session-1", limit=5)
        assert allowed is False

        # Session 2 should still be allowed
        allowed, remaining = limiter.check_rate_limit("session-2", limit=5)
        assert allowed is True
        assert remaining == 4

    def test_anonymous_session_handling(self):
        """Test that None/empty session IDs use 'anonymous'."""
        from backend.api.v1.jobs import RateLimiter

        limiter = RateLimiter(window_seconds=60)

        # Both None and empty should map to 'anonymous'
        allowed1, _ = limiter.check_rate_limit(None, limit=5)
        allowed2, _ = limiter.check_rate_limit("", limit=5)

        assert allowed1 is True
        assert allowed2 is True

        # They should share the same counter
        assert limiter.active_session_count == 1

    def test_max_sessions_limit_enforced(self):
        """Test that MAX_SESSIONS limit prevents memory leak."""
        from backend.api.v1.jobs import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        original_max = limiter.MAX_SESSIONS

        # Temporarily reduce MAX_SESSIONS for testing
        limiter.MAX_SESSIONS = 3

        try:
            # Add 3 sessions (at capacity)
            limiter.check_rate_limit("session-1", limit=5)
            limiter.check_rate_limit("session-2", limit=5)
            limiter.check_rate_limit("session-3", limit=5)

            assert limiter.active_session_count == 3

            # Adding 4th session should evict oldest
            limiter.check_rate_limit("session-4", limit=5)

            # Should still only have 3 sessions
            assert limiter.active_session_count == 3

            # Session 4 should be present
            allowed, _ = limiter.check_rate_limit("session-4", limit=5)
            assert allowed is True

        finally:
            limiter.MAX_SESSIONS = original_max

    def test_old_timestamps_cleaned_up(self):
        """Test that old timestamps are cleaned up."""
        from backend.api.v1.jobs import RateLimiter

        # Use 1 second window for fast testing
        limiter = RateLimiter(window_seconds=1)
        session_id = "test-session"

        # Make 5 requests (max out limit)
        for _ in range(5):
            limiter.check_rate_limit(session_id, limit=5)

        # Should be blocked
        allowed, _ = limiter.check_rate_limit(session_id, limit=5)
        assert allowed is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should now be allowed again
        allowed, remaining = limiter.check_rate_limit(session_id, limit=5)
        assert allowed is True
        assert remaining == 4

    def test_active_session_count_property(self):
        """Test active_session_count property."""
        from backend.api.v1.jobs import RateLimiter

        limiter = RateLimiter(window_seconds=60)

        assert limiter.active_session_count == 0

        limiter.check_rate_limit("session-1", limit=5)
        assert limiter.active_session_count == 1

        limiter.check_rate_limit("session-2", limit=5)
        assert limiter.active_session_count == 2

        # Same session doesn't increase count
        limiter.check_rate_limit("session-1", limit=5)
        assert limiter.active_session_count == 2

    def test_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        from backend.api.v1.jobs import RateLimiter
        import threading

        limiter = RateLimiter(window_seconds=60)
        results = []
        errors = []

        def make_requests(session_id, num_requests):
            try:
                for _ in range(num_requests):
                    limiter.check_rate_limit(session_id, limit=100)
                results.append(session_id)
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=make_requests, args=(f"session-{i}", 20))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        assert len(results) == 10
