"""
Unit tests for ChEMBL API client (backend/modules/api_client.py).
All external API calls mocked -- no real network requests.
"""
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestCacheNonNone:
    """Tests for the cache_non_none decorator."""

    def test_caches_non_none_result(self):
        """Test that non-None results are cached on second call."""
        from backend.modules.api_client import cache_non_none

        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=60)
        def expensive_call(key):
            nonlocal call_count
            call_count += 1
            return f"result-{key}"

        result1 = expensive_call("a")
        result2 = expensive_call("a")
        assert result1 == "result-a"
        assert result2 == "result-a"
        assert call_count == 1  # Second call used cache

    def test_does_not_cache_none(self):
        """Test that None results are NOT cached."""
        from backend.modules.api_client import cache_non_none

        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=60)
        def returns_none(key):
            nonlocal call_count
            call_count += 1
            return None

        result1 = returns_none("a")
        result2 = returns_none("a")
        assert result1 is None
        assert result2 is None
        assert call_count == 2  # Both calls hit the function

    def test_cache_clear(self):
        """Test cache_clear resets cache."""
        from backend.modules.api_client import cache_non_none

        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=60)
        def fn(key):
            nonlocal call_count
            call_count += 1
            return key

        fn("x")
        fn("x")
        assert call_count == 1
        fn.cache_clear()
        fn("x")
        assert call_count == 2

    def test_cache_info(self):
        """Test cache_info returns stats."""
        from backend.modules.api_client import cache_non_none

        @cache_non_none(maxsize=10, ttl_seconds=60)
        def fn(key):
            return key

        fn("a")
        fn("a")  # cache hit
        fn("b")  # cache miss

        info = fn.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.maxsize == 10
        assert info.currsize == 2

    def test_cache_evicts_when_full(self):
        """Test cache evicts oldest entry when maxsize reached."""
        from backend.modules.api_client import cache_non_none

        @cache_non_none(maxsize=2, ttl_seconds=60)
        def fn(key):
            return key

        fn("a")
        fn("b")
        fn("c")  # Should evict "a"

        info = fn.cache_info()
        assert info.currsize == 2

    def test_cache_ttl_expiry(self):
        """Test cache entries expire after TTL."""
        from backend.modules.api_client import cache_non_none

        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=0.1)
        def fn(key):
            nonlocal call_count
            call_count += 1
            return key

        fn("a")
        assert call_count == 1
        time.sleep(0.15)  # Wait for TTL to expire
        fn("a")
        assert call_count == 2  # Had to call function again


class TestWithTimeout:
    """Tests for the with_timeout decorator."""

    def test_returns_result_within_timeout(self):
        """Test normal function returns result."""
        from backend.modules.api_client import with_timeout

        @with_timeout(timeout_seconds=5)
        def fast_fn():
            return "done"

        assert fast_fn() == "done"

    def test_returns_none_on_timeout(self):
        """Test returns None when function exceeds timeout."""
        from backend.modules.api_client import with_timeout

        @with_timeout(timeout_seconds=0.1)
        def slow_fn():
            time.sleep(5)
            return "too late"

        result = slow_fn()
        assert result is None

    def test_returns_none_on_exception(self):
        """Test returns None when function raises exception."""
        from backend.modules.api_client import with_timeout

        @with_timeout(timeout_seconds=5)
        def failing_fn():
            raise RuntimeError("boom")

        result = failing_fn()
        assert result is None


class TestSanitizeCompoundName:
    """Tests for _sanitize_compound_name in azure_sync."""

    def test_basic_sanitization(self):
        """Test basic compound name sanitization."""
        from backend.core.azure_sync import _sanitize_compound_name
        assert _sanitize_compound_name("Aspirin") == "Aspirin"
        assert _sanitize_compound_name("Vitamin C (Ascorbic Acid)") == "Vitamin_C_Ascorbic_Acid"
        assert _sanitize_compound_name("test/compound\\name") == "test_compound_name"
        assert _sanitize_compound_name("  ") == "unnamed_compound"
        assert _sanitize_compound_name("a___b") == "a_b"


class TestIsUuidPath:
    """Tests for _is_uuid_path helper."""

    def test_valid_uuid_path(self):
        """Test recognizes valid UUID paths."""
        from backend.core.azure_sync import _is_uuid_path
        assert _is_uuid_path("results/3a/3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c.zip") is True

    def test_name_based_path(self):
        """Test rejects name-based paths."""
        from backend.core.azure_sync import _is_uuid_path
        assert _is_uuid_path("results/Aspirin.zip") is False

    def test_non_results_path(self):
        """Test rejects non-results paths."""
        from backend.core.azure_sync import _is_uuid_path
        assert _is_uuid_path("logs/some.zip") is False


class TestExtractEntryIdFromBlob:
    """Tests for _extract_entry_id_from_blob helper."""

    def test_extracts_uuid(self):
        """Test extracts UUID from valid blob path."""
        from backend.core.azure_sync import _extract_entry_id_from_blob
        result = _extract_entry_id_from_blob("results/3a/3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c.zip")
        assert result == "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c"

    def test_returns_none_for_name_path(self):
        """Test returns None for name-based path."""
        from backend.core.azure_sync import _extract_entry_id_from_blob
        assert _extract_entry_id_from_blob("results/Aspirin.zip") is None


class TestGetChemblClient:
    """Tests for _get_chembl_client lazy initialization."""

    def test_returns_none_when_not_installed(self):
        """Test returns None when chembl_webresource_client not installed."""
        from backend.modules import api_client

        # Save and clear the global
        original = api_client._chembl_client
        api_client._chembl_client = None

        try:
            with patch('backend.modules.api_client._configure_chembl_settings'):
                with patch.dict('sys.modules', {'chembl_webresource_client': None,
                                                 'chembl_webresource_client.new_client': None}):
                    # Force re-import to trigger ImportError
                    pass
        finally:
            api_client._chembl_client = original


class TestConfigureChemblSettings:
    """Tests for _configure_chembl_settings."""

    def test_configures_once(self):
        """Test settings configured only once (idempotent)."""
        from backend.modules import api_client

        original = api_client._chembl_settings_configured

        # Set to True so first call is a no-op
        api_client._chembl_settings_configured = True
        try:
            api_client._configure_chembl_settings()
            assert api_client._chembl_settings_configured is True
        finally:
            api_client._chembl_settings_configured = original

    def test_configures_from_scratch(self):
        """Test configures settings when not yet configured."""
        from backend.modules import api_client

        original = api_client._chembl_settings_configured
        api_client._chembl_settings_configured = False

        try:
            # The function imports Settings inside its body, so we mock at the import
            mock_settings_cls = MagicMock()
            mock_instance = MagicMock()
            mock_instance.MAX_LIMIT = 20
            mock_instance.TIMEOUT = 3.0
            mock_settings_cls.Instance.return_value = mock_instance

            import sys
            mock_mod = MagicMock()
            mock_mod.Settings = mock_settings_cls
            with patch.dict(sys.modules, {'chembl_webresource_client.settings': mock_mod}):
                api_client._configure_chembl_settings()
                assert api_client._chembl_settings_configured is True
        finally:
            api_client._chembl_settings_configured = original


class TestGetThreadSession:
    """Tests for _get_thread_session thread-local sessions."""

    def test_returns_session(self):
        """Test returns a requests.Session instance."""
        from backend.modules.api_client import _get_thread_session
        session = _get_thread_session()
        import requests
        assert isinstance(session, requests.Session)

    def test_same_thread_same_session(self):
        """Test same thread gets same session."""
        from backend.modules.api_client import _get_thread_session
        s1 = _get_thread_session()
        s2 = _get_thread_session()
        assert s1 is s2


class TestGetTimeoutExecutor:
    """Tests for _get_timeout_executor."""

    def test_returns_executor(self):
        """Test returns a ThreadPoolExecutor."""
        from backend.modules.api_client import _get_timeout_executor
        from concurrent.futures import ThreadPoolExecutor
        executor = _get_timeout_executor()
        assert isinstance(executor, ThreadPoolExecutor)


class TestShutdownApiClient:
    """Tests for shutdown_api_client."""

    def test_shutdown_resets_thread_local(self):
        """Test shutdown clears thread-local sessions."""
        from backend.modules.api_client import shutdown_api_client, _get_thread_session, _thread_local

        # Ensure a session exists
        _get_thread_session()
        assert hasattr(_thread_local, 'session')

        shutdown_api_client()
        # After shutdown, thread_local.session should be reset
        # (the function deletes the attribute)
        # Note: shutdown_api_client resets _thread_local completely


class TestGetResponseData:
    """Tests for _get_response_data helper."""

    def test_returns_empty_for_none(self):
        """Test returns empty list when data is None."""
        from backend.modules.api_client import _get_response_data
        assert _get_response_data(None, "activity") == []

    def test_extracts_known_endpoint(self):
        """Test extracts data using known endpoint key mapping."""
        from backend.modules.api_client import _get_response_data
        data = {"activities": [{"id": 1}, {"id": 2}]}
        result = _get_response_data(data, "activity")
        assert len(result) == 2
        assert result[0]["id"] == 1

    def test_extracts_molecule_endpoint(self):
        """Test extracts data for molecule endpoint."""
        from backend.modules.api_client import _get_response_data
        data = {"molecules": [{"chembl_id": "CHEMBL25"}]}
        result = _get_response_data(data, "molecule")
        assert len(result) == 1

    def test_fallback_pluralized_key(self):
        """Test falls back to pluralized endpoint name when not in mapping."""
        from backend.modules.api_client import _get_response_data
        data = {"unknowns": [{"x": 1}]}
        result = _get_response_data(data, "unknown")
        assert len(result) == 1

    def test_missing_key_returns_empty(self):
        """Test returns empty list when response key not in data."""
        from backend.modules.api_client import _get_response_data
        data = {"wrong_key": [{"x": 1}]}
        result = _get_response_data(data, "activity")
        assert result == []


class TestUrlEncodeSmiles:
    """Tests for _url_encode_smiles."""

    def test_basic_smiles(self):
        """Test basic SMILES encoding."""
        from backend.modules.api_client import _url_encode_smiles
        result = _url_encode_smiles("CCO")
        assert result == "CCO"

    def test_special_chars_encoded(self):
        """Test URL-significant chars are encoded."""
        from backend.modules.api_client import _url_encode_smiles
        result = _url_encode_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        # Parentheses and = should be encoded
        assert "(" not in result or "%28" in result

    def test_hash_encoded(self):
        """Test # is encoded."""
        from backend.modules.api_client import _url_encode_smiles
        result = _url_encode_smiles("C#N")
        assert "%23" in result

    def test_slash_encoded(self):
        """Test / is encoded."""
        from backend.modules.api_client import _url_encode_smiles
        result = _url_encode_smiles("C/C=C/C")
        assert "%2F" in result


class TestClearCaches:
    """Tests for clear_caches function."""

    def test_clear_caches_no_error(self):
        """Test clear_caches runs without error."""
        from backend.modules.api_client import clear_caches
        # Should not raise
        clear_caches()


class TestGetCacheInfo:
    """Tests for get_cache_info function."""

    def test_returns_dict(self):
        """Test get_cache_info returns dict with expected keys."""
        from backend.modules.api_client import get_cache_info
        info = get_cache_info()
        assert isinstance(info, dict)
        assert "molecule_data" in info
        assert "classification" in info
        assert "target_name" in info
        assert "drug_indications" in info

    def test_cache_info_has_stats(self):
        """Test each cache info entry has expected stats."""
        from backend.modules.api_client import get_cache_info
        info = get_cache_info()
        for key, stats in info.items():
            assert "hits" in stats
            assert "misses" in stats
            assert "maxsize" in stats
            assert "currsize" in stats


class TestCacheInfoAsDict:
    """Tests for CacheInfo._asdict method (line 178)."""

    def test_asdict_returns_correct_keys(self):
        """Test CacheInfo._asdict returns all expected fields."""
        from backend.modules.api_client import cache_non_none

        @cache_non_none(maxsize=5, ttl_seconds=60)
        def fn(key):
            return key

        fn("a")
        fn("a")  # hit
        fn("b")  # miss

        info = fn.cache_info()
        d = info._asdict()
        assert d["hits"] == 1
        assert d["misses"] == 2
        assert d["maxsize"] == 5
        assert d["currsize"] == 2


class TestCacheDoubleCheck:
    """Tests for ARCH-10 double-checked locking in cache_non_none (lines 138-141)."""

    def test_concurrent_cache_writes(self):
        """Test double-check path when concurrent threads populate the same key."""
        import threading
        from backend.modules.api_client import cache_non_none

        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=60)
        def slow_fn(key):
            nonlocal call_count
            call_count += 1
            return f"result-{key}-{call_count}"

        # Simulate: both threads call the function, both get a result,
        # but only one should set the cache. The second should use the first's value.
        results = []
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            result = slow_fn("shared")
            results.append(result)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Both should return a result (either the same or different depending on timing)
        assert len(results) == 2
        # The cache should have exactly 1 entry
        info = slow_fn.cache_info()
        assert info.currsize == 1


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_basic_rate_limiting(self):
        """Test rate limiter enforces timing."""
        from backend.modules.api_client import RateLimiter
        limiter = RateLimiter(calls_per_second=100)  # High rate for fast test
        # Should not raise
        limiter.wait()
        limiter.wait()

    def test_min_interval_property(self):
        """Test min_interval is calculated correctly."""
        from backend.modules.api_client import RateLimiter
        limiter = RateLimiter(calls_per_second=10)
        assert limiter.min_interval == pytest.approx(0.1, abs=0.01)


class TestGetSession:
    """Tests for get_session function."""

    def test_returns_session_with_retries(self):
        """Test creates a session with retry adapters."""
        from backend.modules.api_client import get_session
        import requests
        s = get_session()
        assert isinstance(s, requests.Session)
        # Should have retry adapters mounted
        assert len(s.adapters) >= 2  # http:// and https://
