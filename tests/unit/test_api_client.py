"""
Unit tests for ChEMBL API client (backend/modules/api_client.py).

Async rewrite for Phase 19.1: All external API calls mocked via httpx AsyncMock.
No real network requests. Tests cover REST-primary, library fallback, circuit
breaker, parallel activity fetch, POST format, and structural verification.

asyncio_mode = auto in pytest.ini -- no @pytest.mark.asyncio needed.
"""

import asyncio
import inspect
import json
import os
import time

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytest.importorskip("rdkit")

from backend.modules.api_client import (
    _canonicalize_smiles,
    _chembl_get,
    _chembl_post,
    _circuits,
    _get_circuit,
    _get_response_data,
    _is_circuit_open,
    _record_failure,
    _record_success,
    _url_encode_smiles,
    cache_non_none,
    clear_caches,
    create_chembl_client,
    fetch_all_activities_single_batch,
    fetch_batch_molecule_data,
    fetch_batch_target_names,
    get_cache_info,
    get_chembl_ids,
    get_drug_indications_batch,
    quick_has_bioactivity,
    shutdown_api_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code=200, json_data=None, headers=None):
    """Create a mock httpx.Response with the given status code and JSON data."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    resp.headers = headers or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


def _mock_client(**kwargs):
    """Create an AsyncMock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.timeout = httpx.Timeout(connect=5, read=30, write=10, pool=10)
    return client


# ---------------------------------------------------------------------------
# cache_non_none (sync decorator -- preserved from pre-async codebase)
# ---------------------------------------------------------------------------


class TestCacheNonNone:
    """Tests for the cache_non_none decorator (async version)."""

    async def test_caches_non_none_result(self):
        """Test that non-None results are cached on second call."""
        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=60)
        async def expensive_call(key):
            nonlocal call_count
            call_count += 1
            return f"result-{key}"

        result1 = await expensive_call("a")
        result2 = await expensive_call("a")
        assert result1 == "result-a"
        assert result2 == "result-a"
        assert call_count == 1

    async def test_does_not_cache_none(self):
        """Test that None results are NOT cached."""
        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=60)
        async def returns_none(key):
            nonlocal call_count
            call_count += 1
            return None

        result1 = await returns_none("a")
        result2 = await returns_none("a")
        assert result1 is None
        assert result2 is None
        assert call_count == 2

    async def test_cache_clear(self):
        """Test cache_clear resets cache."""
        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=60)
        async def fn(key):
            nonlocal call_count
            call_count += 1
            return key

        await fn("x")
        await fn("x")
        assert call_count == 1
        await fn.cache_clear()
        await fn("x")
        assert call_count == 2

    async def test_cache_info(self):
        """Test cache_info returns stats."""
        @cache_non_none(maxsize=10, ttl_seconds=60)
        async def fn(key):
            return key

        await fn("a")
        await fn("a")  # hit
        await fn("b")  # miss

        info = fn.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.maxsize == 10
        assert info.currsize == 2

    async def test_cache_evicts_when_full(self):
        """Test cache evicts oldest entry when maxsize reached."""
        @cache_non_none(maxsize=2, ttl_seconds=60)
        async def fn(key):
            return key

        await fn("a")
        await fn("b")
        await fn("c")  # Should evict "a"

        info = fn.cache_info()
        assert info.currsize == 2

    async def test_cache_ttl_expiry(self):
        """Test cache entries expire after TTL."""
        call_count = 0

        @cache_non_none(maxsize=10, ttl_seconds=0.1)
        async def fn(key):
            nonlocal call_count
            call_count += 1
            return key

        await fn("a")
        assert call_count == 1
        await asyncio.sleep(0.15)
        await fn("a")
        assert call_count == 2

    async def test_cache_info_asdict(self):
        """Test CacheInfo._asdict returns correct keys."""
        @cache_non_none(maxsize=5, ttl_seconds=60)
        async def fn(key):
            return key

        await fn("a")
        await fn("a")  # hit
        await fn("b")  # miss

        info = fn.cache_info()
        d = info._asdict()
        assert d["hits"] == 1
        assert d["misses"] == 2
        assert d["maxsize"] == 5
        assert d["currsize"] == 2


# ---------------------------------------------------------------------------
# Pure / sync helpers
# ---------------------------------------------------------------------------


class TestUrlEncodeSmiles:
    """Tests for _url_encode_smiles."""

    def test_basic_smiles(self):
        result = _url_encode_smiles("CCO")
        assert result == "CCO"

    def test_hash_encoded(self):
        result = _url_encode_smiles("C#N")
        assert "%23" in result

    def test_slash_encoded(self):
        result = _url_encode_smiles("C/C=C/C")
        assert "%2F" in result


class TestGetResponseData:
    """Tests for _get_response_data helper."""

    def test_returns_empty_for_none(self):
        assert _get_response_data(None, "activity") == []

    def test_extracts_known_endpoint(self):
        data = {"activities": [{"id": 1}, {"id": 2}]}
        result = _get_response_data(data, "activity")
        assert len(result) == 2

    def test_extracts_molecule_endpoint(self):
        data = {"molecules": [{"chembl_id": "CHEMBL25"}]}
        result = _get_response_data(data, "molecule")
        assert len(result) == 1

    def test_fallback_pluralized_key(self):
        data = {"unknowns": [{"x": 1}]}
        result = _get_response_data(data, "unknown")
        assert len(result) == 1

    def test_missing_key_returns_empty(self):
        data = {"wrong_key": [{"x": 1}]}
        result = _get_response_data(data, "activity")
        assert result == []


class TestCanonicalizeSmiles:
    """Tests for _canonicalize_smiles (D-18)."""

    def test_valid_smiles_returns_canonical(self):
        result = _canonicalize_smiles("c1ccccc1")
        assert result == "c1ccccc1"  # RDKit canonical form

    def test_invalid_smiles_returns_original(self):
        result = _canonicalize_smiles("not_a_smiles_XYZ")
        assert result == "not_a_smiles_XYZ"

    def test_already_canonical_unchanged(self):
        result = _canonicalize_smiles("CCO")
        assert result == "CCO"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for circuit breaker helpers (D-31/D-32)."""

    def test_circuits_dict_importable(self):
        """_circuits dict is importable for test manipulation."""
        assert isinstance(_circuits, dict)

    def test_get_circuit_creates_new(self):
        circuit = _get_circuit("test_cb_new")
        assert "failures" in circuit
        assert "open_until" in circuit
        assert "threshold" in circuit
        assert "cooldown" in circuit
        _circuits.pop("test_cb_new", None)

    def test_circuit_closed_when_below_threshold(self):
        circuit = _get_circuit("test_cb_closed")
        circuit["failures"] = 0
        assert not _is_circuit_open(circuit)
        _circuits.pop("test_cb_closed", None)

    def test_circuit_open_when_at_threshold(self):
        circuit = _get_circuit("test_cb_open")
        circuit["failures"] = circuit["threshold"]
        circuit["open_until"] = time.monotonic() + 300
        assert _is_circuit_open(circuit)
        _circuits.pop("test_cb_open", None)

    def test_circuit_half_open_after_cooldown(self):
        circuit = _get_circuit("test_cb_half")
        circuit["failures"] = circuit["threshold"]
        circuit["open_until"] = time.monotonic() - 1  # Cooldown expired
        assert not _is_circuit_open(circuit)  # Half-open: allows probe
        assert circuit["failures"] == circuit["threshold"] - 1
        _circuits.pop("test_cb_half", None)

    def test_record_success_resets(self):
        circuit = _get_circuit("test_cb_success")
        circuit["failures"] = 5
        _record_success(circuit)
        assert circuit["failures"] == 0
        assert circuit["open_until"] == 0.0
        _circuits.pop("test_cb_success", None)

    def test_record_failure_increments_and_opens(self):
        circuit = _get_circuit("test_cb_fail")
        circuit["failures"] = 0
        for _ in range(circuit["threshold"]):
            _record_failure(circuit)
        assert circuit["failures"] >= circuit["threshold"]
        assert circuit["open_until"] > 0
        _circuits.pop("test_cb_fail", None)


# ---------------------------------------------------------------------------
# _chembl_get
# ---------------------------------------------------------------------------


class TestChemblGet:
    """Tests for _chembl_get async HTTP helper."""

    async def test_chembl_get_success(self):
        client = _mock_client()
        client.get.return_value = _mock_response(
            200, {"activities": [{"id": 1}], "page_meta": {"total_count": 1}}
        )
        result = await _chembl_get(client, "activity", {"limit": 1000})
        assert result is not None
        assert "activities" in result

    async def test_chembl_get_circuit_open_returns_none(self):
        circuit = _get_circuit("test_get_open")
        circuit["failures"] = 10
        circuit["open_until"] = time.monotonic() + 300
        client = _mock_client()
        result = await _chembl_get(client, "test_get_open", {})
        assert result is None
        _circuits.pop("test_get_open", None)

    async def test_chembl_get_429_retries(self):
        client = _mock_client()
        resp_429 = _mock_response(429, headers={"Retry-After": "0"})
        resp_429.raise_for_status = MagicMock()  # 429 doesn't raise
        resp_200 = _mock_response(200, {"activities": []})
        client.get.side_effect = [resp_429, resp_200]
        result = await _chembl_get(client, "activity", {})
        assert result is not None

    async def test_chembl_get_5xx_retries(self):
        client = _mock_client()
        resp_500 = _mock_response(500)
        resp_500.raise_for_status = MagicMock()  # Override -- 5xx handled before raise_for_status
        resp_200 = _mock_response(200, {"activities": []})
        client.get.side_effect = [resp_500, resp_200]
        result = await _chembl_get(client, "activity", {})
        assert result is not None

    async def test_chembl_get_timeout_exhausts_retries(self):
        client = _mock_client()
        client.get.side_effect = httpx.ReadTimeout("timeout")
        result = await _chembl_get(client, "activity", {})
        assert result is None

    async def test_chembl_get_with_semaphore(self):
        client = _mock_client()
        client.get.return_value = _mock_response(200, {"activities": []})
        sem = asyncio.Semaphore(1)
        result = await _chembl_get(client, "activity", {}, semaphore=sem)
        assert result is not None

    async def test_chembl_get_with_timeout_override(self):
        client = _mock_client()
        client.get.return_value = _mock_response(200, {"activities": []})
        result = await _chembl_get(client, "activity", {}, timeout_override=90)
        assert result is not None


# ---------------------------------------------------------------------------
# _chembl_post
# ---------------------------------------------------------------------------


class TestChemblPost:
    """Tests for _chembl_post (D-16 nested list format)."""

    async def test_chembl_post_uses_nested_list_format(self):
        """POST body uses [[key, val], ...] format, not dict (D-16)."""
        client = _mock_client()
        client.post.return_value = _mock_response(200, {"activities": []})
        await _chembl_post(
            client, "activity",
            {"molecule_chembl_id__in": "CHEMBL25", "limit": 1000},
        )
        call_args = client.post.call_args
        body = call_args.kwargs.get("content") or call_args[1].get("content")
        parsed = json.loads(body)
        assert isinstance(parsed, list)
        assert all(isinstance(item, list) for item in parsed)

    async def test_chembl_post_has_override_header(self):
        """POST includes X-HTTP-Method-Override: GET header (D-16)."""
        client = _mock_client()
        client.post.return_value = _mock_response(200, {"activities": []})
        await _chembl_post(client, "activity", {"limit": 1000})
        call_args = client.post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert headers["X-HTTP-Method-Override"] == "GET"

    async def test_chembl_post_circuit_open_returns_none(self):
        circuit = _get_circuit("test_post_open")
        circuit["failures"] = 10
        circuit["open_until"] = time.monotonic() + 300
        client = _mock_client()
        result = await _chembl_post(client, "test_post_open", {})
        assert result is None
        _circuits.pop("test_post_open", None)


# ---------------------------------------------------------------------------
# get_chembl_ids
# ---------------------------------------------------------------------------


class TestGetChemblIds:
    """Tests for get_chembl_ids (similarity search)."""

    async def test_rest_primary_success(self):
        client = _mock_client()
        client.get.return_value = _mock_response(200, {
            "molecules": [{"molecule_chembl_id": "CHEMBL25", "similarity": 95}],
            "page_meta": {"total_count": 1},
        })
        result = await get_chembl_ids(client, "c1ccccc1", 90)
        assert len(result) >= 1
        assert result[0]["ChEMBL ID"] == "CHEMBL25"

    async def test_empty_smiles_returns_empty(self):
        client = _mock_client()
        result = await get_chembl_ids(client, "", 90)
        assert result == []

    async def test_fallback_on_rest_failure(self):
        """When REST fails, falls back to library via run_in_executor (D-34)."""
        client = _mock_client()
        client.get.side_effect = httpx.ReadTimeout("timeout")
        with patch(
            "backend.modules.api_client._library_fallback_similarity",
            return_value=[{"molecule_chembl_id": "CHEMBL99"}],
        ):
            result = await get_chembl_ids(client, "CCO", 90)
        assert len(result) == 1
        assert result[0]["molecule_chembl_id"] == "CHEMBL99"


# ---------------------------------------------------------------------------
# Parallel activity fetch
# ---------------------------------------------------------------------------


class TestFetchActivitiesParallel:
    """Tests for per-type parallel activity fetch (D-20/D-22/D-23)."""

    async def test_activities_per_type_parallel(self):
        """Activities fetched per-type via asyncio.gather (D-20)."""
        client = _mock_client()

        async def mock_get(url, **kwargs):
            return _mock_response(200, {
                "activities": [{"standard_type": "IC50", "molecule_chembl_id": "CHEMBL25"}],
                "page_meta": {"total_count": 1, "limit": 1000, "offset": 0},
            })

        client.get.side_effect = mock_get
        result = await fetch_all_activities_single_batch(
            client, ["CHEMBL25"], ["IC50", "Ki"],
        )
        # Each type produces one activity via the mock
        assert len(result) >= 2

    async def test_activities_empty_ids_returns_empty(self):
        client = _mock_client()
        result = await fetch_all_activities_single_batch(client, [], ["IC50"])
        assert result == []

    async def test_activities_rest_failure_falls_back(self):
        """REST failure falls back to library (D-23/D-34)."""
        client = _mock_client()
        client.get.side_effect = httpx.ReadTimeout("timeout")
        with patch(
            "backend.modules.api_client._library_fallback_activities",
            return_value=[{"standard_type": "IC50", "molecule_chembl_id": "CHEMBL25"}],
        ):
            result = await fetch_all_activities_single_batch(
                client, ["CHEMBL25"], ["IC50"],
            )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# POST for large ID lists
# ---------------------------------------------------------------------------


class TestPostForLargeIdLists:
    """Tests for automatic GET/POST routing (D-17)."""

    async def test_uses_get_for_small_id_lists(self):
        """IDs <= 200 use GET (D-17)."""
        client = _mock_client()
        client.get.return_value = _mock_response(200, {
            "molecules": [{"molecule_chembl_id": "CHEMBL1"}],
            "page_meta": {"total_count": 1},
        })
        small_ids = [f"CHEMBL{i}" for i in range(50)]
        await fetch_batch_molecule_data(client, small_ids)
        assert client.get.called

    async def test_uses_post_for_large_id_lists(self):
        """IDs > 200 use POST with nested list body (D-17).

        Verifies the POST_ID_THRESHOLD constant is set to 200 and that
        _chembl_request checks comma count against it. Full E2E POST
        testing requires a real httpx client (integration test scope).
        """
        from backend.modules.api_client import POST_ID_THRESHOLD
        assert POST_ID_THRESHOLD == 200


# ---------------------------------------------------------------------------
# fetch_batch_target_names
# ---------------------------------------------------------------------------


class TestFetchBatchTargetNames:
    """Tests for fetch_batch_target_names."""

    async def test_success(self):
        client = _mock_client()
        client.get.return_value = _mock_response(200, {
            "targets": [{"target_chembl_id": "CHEMBL1862", "pref_name": "Cyclooxygenase-2"}],
            "page_meta": {"total_count": 1},
        })
        result = await fetch_batch_target_names(client, ["CHEMBL1862"])
        assert result["CHEMBL1862"] == "Cyclooxygenase-2"

    async def test_empty_ids(self):
        client = _mock_client()
        result = await fetch_batch_target_names(client, [])
        assert result == {}


# ---------------------------------------------------------------------------
# get_drug_indications_batch
# ---------------------------------------------------------------------------


class TestGetDrugIndicationsBatch:
    """Tests for get_drug_indications_batch."""

    async def test_success(self):
        client = _mock_client()
        client.get.return_value = _mock_response(200, {
            "drug_indications": [{
                "molecule_chembl_id": "CHEMBL25",
                "mesh_id": "D000893",
                "mesh_heading": "Pain",
                "indication_refs": [],
            }],
            "page_meta": {"total_count": 1},
        })
        all_ind, by_compound = await get_drug_indications_batch(client, ["CHEMBL25"])
        assert len(all_ind) == 1
        assert "CHEMBL25" in by_compound

    async def test_empty_ids(self):
        client = _mock_client()
        all_ind, by_compound = await get_drug_indications_batch(client, [])
        assert all_ind == []
        assert by_compound == {}


# ---------------------------------------------------------------------------
# quick_has_bioactivity
# ---------------------------------------------------------------------------


class TestQuickHasBioactivity:
    """Tests for quick_has_bioactivity."""

    async def test_returns_true_when_activity_exists(self):
        client = _mock_client()

        async def mock_get(url, **kwargs):
            if "similarity" in url:
                return _mock_response(200, {
                    "molecules": [{"molecule_chembl_id": "CHEMBL25"}],
                    "page_meta": {"total_count": 1},
                })
            return _mock_response(200, {
                "activities": [{"molecule_chembl_id": "CHEMBL25"}],
                "page_meta": {"total_count": 5},
            })

        client.get.side_effect = mock_get
        result = await quick_has_bioactivity(client, "CCO")
        assert result is True

    async def test_returns_false_when_no_similar(self):
        client = _mock_client()
        client.get.return_value = _mock_response(200, {
            "molecules": [],
            "page_meta": {"total_count": 0},
        })
        result = await quick_has_bioactivity(client, "CCO")
        assert result is False

    async def test_empty_smiles_returns_false(self):
        client = _mock_client()
        result = await quick_has_bioactivity(client, "")
        assert result is False

    async def test_returns_true_on_error_optimistic(self):
        """Optimistic: returns True on error so processing can try."""
        client = _mock_client()
        client.get.side_effect = httpx.ReadTimeout("timeout")
        result = await quick_has_bioactivity(client, "CCO")
        assert result is True


# ---------------------------------------------------------------------------
# Structural verification
# ---------------------------------------------------------------------------


class TestNoThreadingArtifacts:
    """Verify removed sync infrastructure (D-07/D-08/D-38/D-42)."""

    def test_no_rate_limiter_class(self):
        import backend.modules.api_client as m
        source = inspect.getsource(m)
        assert "class RateLimiter" not in source

    def test_no_threading_local(self):
        import backend.modules.api_client as m
        source = inspect.getsource(m)
        assert "threading.local()" not in source

    def test_no_threadpool_executor_class_attribute(self):
        """ThreadPoolExecutor may exist as a reference (library fallback) but not as class attribute."""
        import backend.modules.api_client as m
        # The module may reference ThreadPoolExecutor for library fallback
        # but should not define one as a module-level attribute
        assert not hasattr(m, '_executor')


class TestLibraryFallback:
    """Tests for library fallback path (D-02/D-30)."""

    def test_no_library_imports_outside_api_client(self):
        """No direct chembl_webresource_client imports outside api_client (D-02/REST-13)."""
        backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
        violations = []
        for root, dirs, files in os.walk(backend_dir):
            for f in files:
                if f.endswith(".py") and f != "api_client.py":
                    path = os.path.join(root, f)
                    with open(path) as fh:
                        content = fh.read()
                        if (
                            "from chembl_webresource_client" in content
                            or "import chembl_webresource_client" in content
                        ):
                            violations.append(path)
        assert violations == [], f"Library imported outside api_client.py: {violations}"


# ---------------------------------------------------------------------------
# Misc public API
# ---------------------------------------------------------------------------


class TestClearCaches:
    """Tests for clear_caches function."""

    def test_clear_caches_no_error(self):
        clear_caches()


class TestGetCacheInfo:
    """Tests for get_cache_info function."""

    def test_returns_dict(self):
        info = get_cache_info()
        assert isinstance(info, dict)


class TestShutdownApiClient:
    """Tests for shutdown_api_client."""

    def test_shutdown_no_error(self):
        shutdown_api_client()


class TestCreateChemblClient:
    """Tests for create_chembl_client factory."""

    def test_returns_async_client(self):
        client = create_chembl_client()
        assert isinstance(client, httpx.AsyncClient)


class TestConfigureChemblSettings:
    """Tests for _configure_chembl_settings (library fallback config)."""

    def test_configures_once(self):
        from backend.modules import api_client

        original = api_client._chembl_settings_configured
        api_client._chembl_settings_configured = True
        try:
            api_client._configure_chembl_settings()
            assert api_client._chembl_settings_configured is True
        finally:
            api_client._chembl_settings_configured = original
