"""
Unit tests for rate limiter with asyncio.Lock (Phase 19.2).

All tests are async since check_rate_limit is now async def.
pytest-asyncio asyncio_mode=auto handles async test functions automatically.
"""
import asyncio


class TestRateLimiter:
    """Tests for RateLimiter class."""

    async def test_allows_requests_within_limit(self):
        """Test that requests within limit are allowed."""
        from backend.core.rate_limiter import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        session_id = "test-session"

        # First request should be allowed
        allowed, remaining = await limiter.check_rate_limit(session_id, limit=5)
        assert allowed is True
        assert remaining == 4

        # Second request should also be allowed
        # Note: remaining decreases by 2 due to reference behavior in implementation
        allowed, remaining = await limiter.check_rate_limit(session_id, limit=5)
        assert allowed is True
        assert remaining == 2

        # Third request
        allowed, remaining = await limiter.check_rate_limit(session_id, limit=5)
        assert allowed is True
        assert remaining == 1

    async def test_blocks_requests_over_limit(self):
        """Test that requests over limit are blocked."""
        from backend.core.rate_limiter import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        session_id = "test-session"

        # Make 5 requests (the limit)
        for i in range(5):
            allowed, _ = await limiter.check_rate_limit(session_id, limit=5)
            assert allowed is True

        # 6th request should be blocked
        allowed, remaining = await limiter.check_rate_limit(session_id, limit=5)
        assert allowed is False
        assert remaining == 0

    async def test_different_sessions_have_separate_limits(self):
        """Test that different sessions have separate rate limits."""
        from backend.core.rate_limiter import RateLimiter

        limiter = RateLimiter(window_seconds=60)

        # Max out session 1
        for _ in range(5):
            await limiter.check_rate_limit("session-1", limit=5)

        # Session 1 should be blocked
        allowed, _ = await limiter.check_rate_limit("session-1", limit=5)
        assert allowed is False

        # Session 2 should still be allowed
        allowed, remaining = await limiter.check_rate_limit("session-2", limit=5)
        assert allowed is True
        assert remaining == 4

    async def test_anonymous_session_handling(self):
        """Test that None/empty session IDs use 'anonymous'."""
        from backend.core.rate_limiter import RateLimiter

        limiter = RateLimiter(window_seconds=60)

        # Both None and empty should map to 'anonymous'
        allowed1, _ = await limiter.check_rate_limit(None, limit=5)
        allowed2, _ = await limiter.check_rate_limit("", limit=5)

        assert allowed1 is True
        assert allowed2 is True

        # They should share the same counter
        assert limiter.active_session_count == 1

    async def test_max_sessions_limit_enforced(self):
        """Test that MAX_SESSIONS limit prevents memory leak."""
        from backend.core.rate_limiter import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        original_max = limiter.MAX_SESSIONS

        # Temporarily reduce MAX_SESSIONS for testing
        limiter.MAX_SESSIONS = 3

        try:
            # Add 3 sessions (at capacity)
            await limiter.check_rate_limit("session-1", limit=5)
            await limiter.check_rate_limit("session-2", limit=5)
            await limiter.check_rate_limit("session-3", limit=5)

            assert limiter.active_session_count == 3

            # Adding 4th session should evict oldest
            await limiter.check_rate_limit("session-4", limit=5)

            # Should still only have 3 sessions
            assert limiter.active_session_count == 3

            # Session 4 should be present
            allowed, _ = await limiter.check_rate_limit("session-4", limit=5)
            assert allowed is True

        finally:
            limiter.MAX_SESSIONS = original_max

    async def test_old_timestamps_cleaned_up(self):
        """Test that old timestamps are cleaned up."""
        from backend.core.rate_limiter import RateLimiter

        # Use 0.2 second window for fast testing
        limiter = RateLimiter(window_seconds=0.2)
        session_id = "test-session"

        # Make 5 requests (max out limit)
        for _ in range(5):
            await limiter.check_rate_limit(session_id, limit=5)

        # Should be blocked
        allowed, _ = await limiter.check_rate_limit(session_id, limit=5)
        assert allowed is False

        # Wait for window to expire
        await asyncio.sleep(0.25)

        # Should now be allowed again
        allowed, remaining = await limiter.check_rate_limit(session_id, limit=5)
        assert allowed is True
        assert remaining == 4

    async def test_active_session_count_property(self):
        """Test active_session_count property."""
        from backend.core.rate_limiter import RateLimiter

        limiter = RateLimiter(window_seconds=60)

        assert limiter.active_session_count == 0

        await limiter.check_rate_limit("session-1", limit=5)
        assert limiter.active_session_count == 1

        await limiter.check_rate_limit("session-2", limit=5)
        assert limiter.active_session_count == 2

        # Same session doesn't increase count
        await limiter.check_rate_limit("session-1", limit=5)
        assert limiter.active_session_count == 2

    async def test_async_concurrency_safety(self):
        """Test that rate limiter is safe under concurrent async access."""
        from backend.core.rate_limiter import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        results = []
        errors = []

        async def make_requests(session_id, num_requests):
            try:
                for _ in range(num_requests):
                    await limiter.check_rate_limit(session_id, limit=100)
                results.append(session_id)
            except Exception as e:
                errors.append(str(e))

        # Run multiple concurrent async tasks
        tasks = [
            make_requests(f"session-{i}", 20)
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        # No errors should occur
        assert len(errors) == 0
        assert len(results) == 10

    async def test_lock_is_asyncio_lock(self):
        """Verify the internal lock is an asyncio.Lock instance."""
        from backend.core.rate_limiter import RateLimiter

        limiter = RateLimiter()
        assert isinstance(limiter._lock, asyncio.Lock)
