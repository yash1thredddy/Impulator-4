# ==============================================================================
# Caching Utilities
# ==============================================================================
"""
Caching utilities for molecular calculations and API responses.
Provides LRU caching with optional TTL (time-to-live) support.
"""

import time
import hashlib
import logging
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, Optional, TypeVar
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Sentinel for distinguishing cached None from cache miss
_CACHE_MISS = object()


# ==============================================================================
# TTL Cache Implementation
# ==============================================================================

class TTLCache:
    """
    Cache with time-to-live (TTL) expiration.

    Items are automatically expired after the specified TTL.

    Attributes:
        maxsize: Maximum number of items in cache
        ttl: Time-to-live in seconds for each item

    Example:
        cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        cache.set("key", "value")
        value = cache.get("key")
    """

    def __init__(self, maxsize: int = 1000, ttl: float = 3600.0):
        """
        Initialize TTL cache.

        Args:
            maxsize: Maximum cache size
            ttl: Time-to-live in seconds
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired."""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, ts in self._timestamps.items()
            if current_time - ts > self.ttl
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)

    def _evict_oldest(self) -> None:
        """Evict the oldest entry if cache is full."""
        # Use > instead of >= to avoid off-by-one: evict when we need room
        while len(self._cache) >= self.maxsize:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            self._timestamps.pop(oldest_key, None)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        with self._lock:
            if key in self._cache and not self._is_expired(key):
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]

            self._misses += 1
            return default

    def get_with_sentinel(self, key: str) -> Any:
        """
        Get a value from cache, returning sentinel for cache miss.

        This allows distinguishing between a cached None value and
        a cache miss.

        Args:
            key: Cache key

        Returns:
            Cached value (including None) or _CACHE_MISS sentinel
        """
        with self._lock:
            if key in self._cache and not self._is_expired(key):
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]

            self._misses += 1
            return _CACHE_MISS

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            self._cleanup_expired()
            self._evict_oldest()

            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def has_key(self, key: str) -> bool:
        """
        Check if key exists and is not expired (without side effects).

        Unlike __contains__ or get(), this doesn't modify hit/miss counters
        and correctly handles cached None values.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        with self._lock:
            return key in self._cache and not self._is_expired(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired (no side effects)."""
        return self.has_key(key)

    def __len__(self) -> int:
        """Return number of non-expired items."""
        with self._lock:
            self._cleanup_expired()
            return len(self._cache)


# ==============================================================================
# Cache Decorator
# ==============================================================================

def cached(
    cache: Optional[TTLCache] = None,
    maxsize: int = 1000,
    ttl: float = 3600.0,
    key_func: Optional[Callable] = None,
) -> Callable:
    """
    Decorator to cache function results.

    Args:
        cache: Optional TTLCache instance to use
        maxsize: Cache size if creating new cache
        ttl: TTL if creating new cache
        key_func: Optional function to generate cache key from args

    Example:
        @cached(maxsize=500, ttl=1800)
        def expensive_calculation(smiles):
            ...
    """
    _cache = cache or TTLCache(maxsize=maxsize, ttl=ttl)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _make_cache_key(func.__name__, args, kwargs)

            # Try to get from cache (using sentinel to handle None values)
            result = _cache.get_with_sentinel(key)
            if result is not _CACHE_MISS:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Calculate and cache (including None results)
            result = func(*args, **kwargs)
            _cache.set(key, result)

            return result

        # Attach cache to function for external access
        wrapper.cache = _cache
        wrapper.cache_clear = _cache.clear
        wrapper.cache_stats = _cache.stats

        return wrapper
    return decorator


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a cache key from function arguments."""
    key_parts = [func_name]

    for arg in args:
        key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


# ==============================================================================
# Pre-configured Caches
# ==============================================================================

# Cache for molecular property calculations (1 hour TTL)
molecule_cache = TTLCache(maxsize=5000, ttl=3600.0)

# Cache for SMILES conversions (24 hour TTL)
conversion_cache = TTLCache(maxsize=10000, ttl=86400.0)

# Cache for API responses (15 minute TTL)
api_cache = TTLCache(maxsize=1000, ttl=900.0)


# ==============================================================================
# Specialized Decorators
# ==============================================================================

def cache_molecule(func: Callable) -> Callable:
    """
    Decorator specifically for caching molecular calculations.

    Uses SMILES and all additional arguments as the cache key.

    Example:
        @cache_molecule
        def calculate_properties(smiles):
            ...
    """
    @wraps(func)
    def wrapper(smiles: str, *args, **kwargs):
        # Use SMILES + args + kwargs as cache key to avoid incorrect results
        key = _make_cache_key(func.__name__, (smiles,) + args, kwargs)

        result = molecule_cache.get_with_sentinel(key)
        if result is not _CACHE_MISS:
            return result

        result = func(smiles, *args, **kwargs)
        molecule_cache.set(key, result)

        return result

    wrapper.cache = molecule_cache
    wrapper.cache_clear = molecule_cache.clear
    wrapper.cache_stats = molecule_cache.stats

    return wrapper


def cache_conversion(func: Callable) -> Callable:
    """
    Decorator for caching molecular identifier conversions.

    Uses identifier and all additional arguments as the cache key.

    Example:
        @cache_conversion
        def inchi_key_to_smiles(inchi_key):
            ...
    """
    @wraps(func)
    def wrapper(identifier: str, *args, **kwargs):
        # Use identifier + args + kwargs as cache key to avoid incorrect results
        key = _make_cache_key(func.__name__, (identifier,) + args, kwargs)

        result = conversion_cache.get_with_sentinel(key)
        if result is not _CACHE_MISS:
            return result

        result = func(identifier, *args, **kwargs)
        conversion_cache.set(key, result)

        return result

    wrapper.cache = conversion_cache
    wrapper.cache_clear = conversion_cache.clear
    wrapper.cache_stats = conversion_cache.stats

    return wrapper


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all pre-configured caches.

    Returns:
        Dictionary with stats for each cache
    """
    return {
        "molecule_cache": molecule_cache.stats(),
        "conversion_cache": conversion_cache.stats(),
        "api_cache": api_cache.stats(),
    }


def clear_all_caches() -> None:
    """Clear all pre-configured caches."""
    molecule_cache.clear()
    conversion_cache.clear()
    api_cache.clear()
    logger.info("All caches cleared")


# Standard library LRU cache re-export for simple cases
__all__ = [
    "TTLCache",
    "cached",
    "cache_molecule",
    "cache_conversion",
    "molecule_cache",
    "conversion_cache",
    "api_cache",
    "get_all_cache_stats",
    "clear_all_caches",
    "lru_cache",  # Re-export from functools
]
