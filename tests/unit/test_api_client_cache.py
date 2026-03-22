"""
Tests for api_client caching (Phase 19.2 -- asyncio.Lock).

Tests the following:
- cache_non_none decorator (doesn't cache None results)
- TTL support for caches
- Async wrapper function

All tests are async since cache_non_none wrapper is now async def.
pytest-asyncio asyncio_mode=auto handles async test functions automatically.
"""

import asyncio
import time
import pytest

from backend.modules.api_client import cache_non_none


class TestCacheNonNone:
    """Tests for the cache_non_none decorator."""

    async def test_caches_non_none_results(self):
        """Test that non-None results are cached."""
        call_count = [0]

        @cache_non_none(maxsize=10)
        async def fetch_data(key):
            call_count[0] += 1
            return f"data_{key}"

        # First call - should execute function
        result1 = await fetch_data("test")
        assert result1 == "data_test"
        assert call_count[0] == 1

        # Second call - should return cached result
        result2 = await fetch_data("test")
        assert result2 == "data_test"
        assert call_count[0] == 1  # No additional call

    async def test_does_not_cache_none_results(self):
        """Test that None results are NOT cached."""
        call_count = [0]
        return_none = [True]

        @cache_non_none(maxsize=10)
        async def fetch_data(key):
            call_count[0] += 1
            if return_none[0]:
                return None
            return f"data_{key}"

        # First call returns None - should NOT be cached
        result1 = await fetch_data("test")
        assert result1 is None
        assert call_count[0] == 1

        # Second call - should execute function again (None wasn't cached)
        result2 = await fetch_data("test")
        assert result2 is None
        assert call_count[0] == 2  # Function called again

        # Now return non-None - should be cached
        return_none[0] = False
        result3 = await fetch_data("test")
        assert result3 == "data_test"
        assert call_count[0] == 3

        # Should be cached now
        result4 = await fetch_data("test")
        assert result4 == "data_test"
        assert call_count[0] == 3  # No additional call

    async def test_ttl_expiration(self):
        """Test that cached entries expire after TTL."""
        call_count = [0]

        @cache_non_none(maxsize=10, ttl_seconds=0.1)  # 0.1 second TTL
        async def fetch_data(key):
            call_count[0] += 1
            return f"data_{key}"

        # First call
        result1 = await fetch_data("test")
        assert result1 == "data_test"
        assert call_count[0] == 1

        # Immediate second call - should use cache
        result2 = await fetch_data("test")
        assert result2 == "data_test"
        assert call_count[0] == 1

        # Wait for TTL to expire
        await asyncio.sleep(0.15)

        # Call after TTL - should execute function again
        result3 = await fetch_data("test")
        assert result3 == "data_test"
        assert call_count[0] == 2  # Function called again

    async def test_cache_clear(self):
        """Test that cache_clear() works (now async)."""
        call_count = [0]

        @cache_non_none(maxsize=10)
        async def fetch_data(key):
            call_count[0] += 1
            return f"data_{key}"

        await fetch_data("test")
        assert call_count[0] == 1

        await fetch_data("test")
        assert call_count[0] == 1  # Cached

        await fetch_data.cache_clear()

        await fetch_data("test")
        assert call_count[0] == 2  # Cache cleared, function called again

    async def test_cache_info(self):
        """Test that cache_info() returns correct statistics."""
        @cache_non_none(maxsize=10)
        async def fetch_data(key):
            return f"data_{key}"

        await fetch_data("a")
        await fetch_data("b")
        await fetch_data("a")  # Hit

        info = fetch_data.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.currsize == 2
        assert info.maxsize == 10

    async def test_maxsize_eviction(self):
        """Test that oldest entries are evicted when maxsize is reached."""
        @cache_non_none(maxsize=3)
        async def fetch_data(key):
            return f"data_{key}"

        # Fill cache
        await fetch_data("a")
        await fetch_data("b")
        await fetch_data("c")

        info = fetch_data.cache_info()
        assert info.currsize == 3

        # Add one more - should evict oldest
        await fetch_data("d")

        info = fetch_data.cache_info()
        assert info.currsize == 3  # Still 3

    async def test_wrapper_is_async(self):
        """Verify the wrapper function is a coroutine function."""
        import inspect

        @cache_non_none(maxsize=10)
        async def fetch_data(key):
            return f"data_{key}"

        assert inspect.iscoroutinefunction(fetch_data)

    async def test_lock_is_asyncio_lock(self):
        """Verify cache_non_none uses asyncio.Lock internally."""
        # We verify indirectly: concurrent async access should not deadlock
        @cache_non_none(maxsize=100)
        async def fetch_data(key):
            await asyncio.sleep(0.01)
            return f"data_{key}"

        # Run 20 concurrent cache operations
        tasks = [fetch_data(f"key-{i}") for i in range(20)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 20
        assert all(r.startswith("data_") for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
