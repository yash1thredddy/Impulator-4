"""
Tests for api_client caching and rate limiting improvements.

Tests the following fixes:
- 3.7: cache_non_none decorator (doesn't cache None results)
- 3.9: TTL support for caches
- 3.12: RateLimiter class
"""

import time
import pytest
import threading
from unittest.mock import MagicMock, patch

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.modules.api_client import cache_non_none, RateLimiter


class TestCacheNonNone:
    """Tests for the cache_non_none decorator."""

    def test_caches_non_none_results(self):
        """Test that non-None results are cached."""
        call_count = [0]

        @cache_non_none(maxsize=10)
        def fetch_data(key):
            call_count[0] += 1
            return f"data_{key}"

        # First call - should execute function
        result1 = fetch_data("test")
        assert result1 == "data_test"
        assert call_count[0] == 1

        # Second call - should return cached result
        result2 = fetch_data("test")
        assert result2 == "data_test"
        assert call_count[0] == 1  # No additional call

    def test_does_not_cache_none_results(self):
        """Test that None results are NOT cached (fix for 3.7)."""
        call_count = [0]
        return_none = [True]

        @cache_non_none(maxsize=10)
        def fetch_data(key):
            call_count[0] += 1
            if return_none[0]:
                return None
            return f"data_{key}"

        # First call returns None - should NOT be cached
        result1 = fetch_data("test")
        assert result1 is None
        assert call_count[0] == 1

        # Second call - should execute function again (None wasn't cached)
        result2 = fetch_data("test")
        assert result2 is None
        assert call_count[0] == 2  # Function called again

        # Now return non-None - should be cached
        return_none[0] = False
        result3 = fetch_data("test")
        assert result3 == "data_test"
        assert call_count[0] == 3

        # Should be cached now
        result4 = fetch_data("test")
        assert result4 == "data_test"
        assert call_count[0] == 3  # No additional call

    def test_ttl_expiration(self):
        """Test that cached entries expire after TTL (fix for 3.9)."""
        call_count = [0]

        @cache_non_none(maxsize=10, ttl_seconds=0.5)  # 0.5 second TTL
        def fetch_data(key):
            call_count[0] += 1
            return f"data_{key}"

        # First call
        result1 = fetch_data("test")
        assert result1 == "data_test"
        assert call_count[0] == 1

        # Immediate second call - should use cache
        result2 = fetch_data("test")
        assert result2 == "data_test"
        assert call_count[0] == 1

        # Wait for TTL to expire
        time.sleep(0.6)

        # Call after TTL - should execute function again
        result3 = fetch_data("test")
        assert result3 == "data_test"
        assert call_count[0] == 2  # Function called again

    def test_cache_clear(self):
        """Test that cache_clear() works."""
        call_count = [0]

        @cache_non_none(maxsize=10)
        def fetch_data(key):
            call_count[0] += 1
            return f"data_{key}"

        fetch_data("test")
        assert call_count[0] == 1

        fetch_data("test")
        assert call_count[0] == 1  # Cached

        fetch_data.cache_clear()

        fetch_data("test")
        assert call_count[0] == 2  # Cache cleared, function called again

    def test_cache_info(self):
        """Test that cache_info() returns correct statistics."""
        @cache_non_none(maxsize=10)
        def fetch_data(key):
            return f"data_{key}"

        fetch_data("a")
        fetch_data("b")
        fetch_data("a")  # Hit

        info = fetch_data.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.currsize == 2
        assert info.maxsize == 10

    def test_maxsize_eviction(self):
        """Test that oldest entries are evicted when maxsize is reached."""
        @cache_non_none(maxsize=3)
        def fetch_data(key):
            return f"data_{key}"

        # Fill cache
        fetch_data("a")
        fetch_data("b")
        fetch_data("c")

        info = fetch_data.cache_info()
        assert info.currsize == 3

        # Add one more - should evict oldest
        fetch_data("d")

        info = fetch_data.cache_info()
        assert info.currsize == 3  # Still 3


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_rate_limiting(self):
        """Test that rate limiter enforces minimum interval."""
        limiter = RateLimiter(calls_per_second=10)  # 0.1 second interval

        start = time.time()

        # Make 5 rapid calls
        for _ in range(5):
            limiter.wait()

        elapsed = time.time() - start

        # Should take at least 0.4 seconds (4 intervals between 5 calls)
        assert elapsed >= 0.35  # Allow small margin

    def test_no_wait_when_interval_passed(self):
        """Test that no wait occurs when sufficient time has passed."""
        limiter = RateLimiter(calls_per_second=10)

        limiter.wait()
        time.sleep(0.15)  # Wait longer than interval

        start = time.time()
        limiter.wait()
        elapsed = time.time() - start

        # Should be nearly instant (no additional wait needed)
        assert elapsed < 0.05

    def test_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        limiter = RateLimiter(calls_per_second=20)  # 0.05 second interval
        call_times = []
        lock = threading.Lock()

        def worker():
            for _ in range(5):
                limiter.wait()
                with lock:
                    call_times.append(time.time())

        threads = [threading.Thread(target=worker) for _ in range(3)]
        start = time.time()

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check that calls are properly spaced
        call_times.sort()
        for i in range(1, len(call_times)):
            interval = call_times[i] - call_times[i-1]
            # Each interval should be at least ~0.045 seconds (with small margin)
            assert interval >= 0.04, f"Interval {i} was only {interval}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
