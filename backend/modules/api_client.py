"""
Async ChEMBL API client with REST-primary access and library fallback.

All public functions are async def with client: httpx.AsyncClient as first parameter.
REST API is the primary path for similarity, activity, molecule, target, and drug
indication fetching. The ChEMBL library (chembl_webresource_client) is fallback only,
wrapped in run_in_executor.

Phase 19.1: Full async rewrite. Proactive rate limiting removed (proven unnecessary).
Circuit breakers protect each endpoint. Per-type parallel activity fetch with parallel
pagination. POST support for >200 IDs.
"""

import asyncio
import json as _json
import time
from functools import wraps
from typing import Any
from collections.abc import Callable

import httpx
import structlog
from rdkit import Chem

from backend.config import settings
from backend.core.metrics import metrics

__all__ = [
    "cache_non_none",
    "cascade_similarity_counts",
    "clear_caches",
    "create_chembl_client",
    "fetch_all_activities_single_batch",
    "fetch_batch_molecule_data",
    "fetch_batch_target_names",
    "get_cache_info",
    "get_chembl_ids",
    "get_drug_indications_batch",
    "probe_all_thresholds",
    "quick_has_bioactivity",
    "shutdown_api_client",
]

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_SIZE = 500  # Bounded: 500 x ~10KB = 5MB ceiling (STAB-04)
MAX_BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 0.5
ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]
DEFAULT_ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50", "AC50", "GI50", "MIC"]

# ChEMBL REST API response key mapping
CHEMBL_RESPONSE_KEYS = {
    "activity": "activities",
    "molecule": "molecules",
    "target": "targets",
    "similarity": "molecules",
    "drug_indication": "drug_indications",
}

# POST when ID list exceeds this threshold (GET fits ~250-300 IDs in URL,
# 200 provides safety margin) -- per D-17
POST_ID_THRESHOLD = 200

# Only request the fields the pipeline actually uses (D-24/D-43).
# ~70% smaller activity responses.  Added assay_type, document_year,
# activity_comment for richer Activity tab context (v2.2).
ACTIVITY_ONLY_FIELDS = (
    "molecule_chembl_id,standard_type,standard_value,standard_units,"
    "pchembl_value,target_chembl_id,assay_chembl_id,data_validity_comment,"
    "assay_type,document_year,activity_comment"
)

CHEMBL_MAX_LIMIT = 1000  # Server-enforced hard cap across ALL endpoints

# Progress callback type
ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# SMILES helpers
# ---------------------------------------------------------------------------

def _url_encode_smiles(smiles: str) -> str:
    """URL-encode SMILES for use in ChEMBL REST API URL paths.

    SMILES can contain URL-significant characters like /, #, +, @, [, ].
    """
    from urllib.parse import quote as _url_quote
    return _url_quote(smiles, safe="")


def _canonicalize_smiles(smiles: str) -> str:
    """Canonicalize SMILES via RDKit before REST API calls.

    The ChEMBL library auto-canonicalizes internally, but REST does exact matching.
    This eliminates the #1 reason REST could fail where library succeeds (D-18).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return smiles  # Return original if canonicalization fails


# ---------------------------------------------------------------------------
# cache_non_none decorator (preserved for 19.2 -- D-08/D-12)
# ---------------------------------------------------------------------------

def cache_non_none(maxsize: int = CACHE_SIZE, ttl_seconds: int = 3600):
    """LRU cache that only caches successful (non-None) results with TTL support.

    This prevents caching of API failures, allowing retry on subsequent calls.
    Cached entries expire after ttl_seconds (default: 1 hour).

    Uses asyncio.Lock for async-safe concurrency (single event loop, Phase 19.2).
    Applied to async functions -- wrapper is async def.

    Args:
        maxsize: Maximum number of entries to cache
        ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)
    """
    def decorator(func):
        cache: dict[Any, Any] = {}  # key -> (value, timestamp)
        cache_hits = [0]
        cache_misses = [0]
        cache_lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            current_time = time.time()

            async with cache_lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if current_time - timestamp < ttl_seconds:
                        cache_hits[0] += 1
                        metrics.increment("cache_hits")
                        return value
                    else:
                        del cache[key]

                cache_misses[0] += 1
                metrics.increment("cache_misses")

            # Call function outside lock to avoid holding lock during I/O
            result = await func(*args, **kwargs)

            # Only cache non-None results
            if result is not None:
                async with cache_lock:
                    # Double-check: another task may have cached this key
                    if key in cache:
                        existing_value, existing_ts = cache[key]
                        if time.time() - existing_ts < ttl_seconds:
                            return existing_value

                    now = time.time()

                    # Evict expired entries first
                    expired_keys = [
                        k for k, (_, ts) in cache.items()
                        if now - ts >= ttl_seconds
                    ]
                    for k in expired_keys:
                        del cache[k]

                    # Evict oldest entry if still at capacity
                    if len(cache) >= maxsize:
                        oldest_key = next(iter(cache))
                        del cache[oldest_key]

                    cache[key] = (result, now)

            return result

        async def cache_clear():
            async with cache_lock:
                cache.clear()
                cache_hits[0] = 0
                cache_misses[0] = 0

        def cache_info():
            class CacheInfo:
                def __init__(self, hits, misses, maxsize_, currsize):
                    self.hits = hits
                    self.misses = misses
                    self.maxsize = maxsize_
                    self.currsize = currsize

                def _asdict(self):
                    return {
                        "hits": self.hits,
                        "misses": self.misses,
                        "maxsize": self.maxsize,
                        "currsize": self.currsize,
                    }

            # cache_info is sync -- reads are atomic in single event loop
            return CacheInfo(cache_hits[0], cache_misses[0], maxsize, len(cache))

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# ChEMBL library singleton (fallback only -- D-02/D-03)
# ---------------------------------------------------------------------------

_chembl_client: dict[str, Any] | None = None
_chembl_settings_configured = False


def _configure_chembl_settings() -> None:
    """Configure ChEMBL client settings for optimal performance.

    IMPORTANT: This must be called BEFORE importing new_client!

    Settings changed:
    - MAX_LIMIT: 20 -> 1000 (reduces API calls by 50x)
    - TIMEOUT: 3.0 -> 60 (prevents timeouts on large queries)
    """
    global _chembl_settings_configured
    if _chembl_settings_configured:
        return

    try:
        from chembl_webresource_client.settings import Settings as ChEMBLSettings
        chembl_settings = ChEMBLSettings.Instance()
        chembl_settings.MAX_LIMIT = CHEMBL_MAX_LIMIT
        chembl_settings.TIMEOUT = 60
        try:
            chembl_settings.CACHING = False
        except Exception:
            pass
        _chembl_settings_configured = True
        logger.info("chembl_library_configured", max_limit=CHEMBL_MAX_LIMIT)
    except ImportError:
        logger.warning("chembl_webresource_client not installed, cannot configure settings")
    except Exception as exc:
        logger.warning("chembl_settings_config_failed", error=str(exc))


def _get_chembl_client() -> dict[str, Any] | None:  # pragma: no cover
    """Lazy initialization of ChEMBL library client for fallback path.

    In async context, only one task calls fallback at a time via circuit breaker,
    so no threading lock needed.
    """
    global _chembl_client

    if _chembl_client is None:
        _configure_chembl_settings()
        try:
            from chembl_webresource_client.new_client import new_client  # type: ignore[import-untyped]
            _chembl_client = {
                "similarity": new_client.similarity,  # type: ignore[attr-defined]
                "molecule": new_client.molecule,  # type: ignore[attr-defined]
                "activity": new_client.activity,  # type: ignore[attr-defined]
                "target": new_client.target,  # type: ignore[attr-defined]
                "drug_indication": new_client.drug_indication,  # type: ignore[attr-defined]
            }
            logger.info("chembl_library_initialized", endpoints=list(_chembl_client.keys()))
        except ImportError:
            logger.warning("chembl_webresource_client not installed")
        except Exception as exc:
            logger.error("chembl_library_init_failed", error=str(exc))

    return _chembl_client


# ---------------------------------------------------------------------------
# httpx client factory (D-10/D-11/D-14/D-26)
# ---------------------------------------------------------------------------

def create_chembl_client() -> httpx.AsyncClient:
    """Create httpx client for ChEMBL API calls.

    One client per job in 19.1 (D-11). Module-level in 19.2.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=30, write=10, pool=10),
        limits=httpx.Limits(max_connections=settings.CHEMBL_MAX_CONNECTIONS),
        headers={"Accept": "application/json"},
    )


# ---------------------------------------------------------------------------
# Circuit breaker infrastructure (D-31/D-32/D-33)
# ---------------------------------------------------------------------------

_circuits: dict[str, dict[str, Any]] = {}


def _make_circuit(threshold: int = 3, cooldown: int = 300) -> dict[str, Any]:
    """Create a new circuit breaker state dict."""
    return {
        "failures": 0,
        "open_until": 0.0,
        "threshold": threshold,
        "cooldown": cooldown,
    }


def _get_circuit(endpoint: str) -> dict[str, Any]:
    """Get or create circuit breaker for an endpoint."""
    if endpoint not in _circuits:
        _circuits[endpoint] = _make_circuit()
    return _circuits[endpoint]


def _is_circuit_open(circuit: dict[str, Any]) -> bool:
    """Check if a circuit breaker is open (blocking requests)."""
    if circuit["failures"] < circuit["threshold"]:
        return False
    if time.monotonic() >= circuit["open_until"]:
        # Half-open: allow one test request
        circuit["failures"] = circuit["threshold"] - 1
        return False
    return True


def _record_success(circuit: dict[str, Any]) -> None:
    """Record a successful request, resetting the circuit."""
    circuit["failures"] = 0
    circuit["open_until"] = 0.0


def _record_failure(circuit: dict[str, Any]) -> None:
    """Record a failed request, potentially opening the circuit."""
    circuit["failures"] += 1
    if circuit["failures"] >= circuit["threshold"]:
        circuit["open_until"] = time.monotonic() + circuit["cooldown"]


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _get_response_data(data: dict | None, endpoint: str) -> list[dict]:
    """Extract the data list from a ChEMBL REST API response.

    Args:
        data: Raw JSON response from _chembl_get/_chembl_post
        endpoint: API endpoint name (used to look up the response key)

    Returns:
        List of data dictionaries, or empty list if data is None/missing
    """
    if data is None:
        return []
    response_key = CHEMBL_RESPONSE_KEYS.get(endpoint, f"{endpoint}s")
    return data.get(response_key, [])


# ---------------------------------------------------------------------------
# Core HTTP helpers (D-28/D-29)
# ---------------------------------------------------------------------------

async def _chembl_get(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, Any],
    *,
    semaphore: asyncio.Semaphore | None = None,
    timeout_override: float | None = None,
) -> dict | None:
    """Make a GET request to ChEMBL REST API with retry and circuit breaker.

    Args:
        client: httpx.AsyncClient instance
        endpoint: API endpoint (e.g., 'activity', 'molecule')
        params: Query parameters
        semaphore: Optional concurrency limiter
        timeout_override: Override the default read timeout

    Returns:
        JSON response as dict, or None on exhausted retries / open circuit
    """
    circuit = _get_circuit(endpoint)
    if _is_circuit_open(circuit):
        logger.debug("circuit_open", endpoint=endpoint)
        return None

    url = f"{settings.CHEMBL_API_URL}/{endpoint}.json"
    request_timeout = (
        httpx.Timeout(connect=5, read=timeout_override, write=10, pool=10)
        if timeout_override
        else client.timeout
    )

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            if semaphore is not None:
                async with semaphore:
                    _start = time.time()
                    response = await client.get(url, params=params, timeout=request_timeout)
            else:
                _start = time.time()
                response = await client.get(url, params=params, timeout=request_timeout)

            metrics.increment("api_calls_total")
            metrics.record_latency("chembl", (time.time() - _start) * 1000)

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2))
                logger.warning("chembl_429", endpoint=endpoint, retry_after=retry_after)
                await asyncio.sleep(retry_after)
                continue

            if response.status_code >= 500:
                logger.warning("chembl_5xx", endpoint=endpoint, status=response.status_code)
                await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
                continue

            response.raise_for_status()
            _record_success(circuit)
            return response.json()

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            metrics.increment("api_calls_failed")
            logger.warning(
                "chembl_request_error",
                endpoint=endpoint,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            metrics.increment("api_calls_failed")
            logger.warning(
                "chembl_http_error",
                endpoint=endpoint,
                status=exc.response.status_code,
                attempt=attempt + 1,
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
        except Exception as exc:
            last_exc = exc
            metrics.increment("api_calls_failed")
            logger.error("chembl_unexpected_error", endpoint=endpoint, error=str(exc))
            break  # Don't retry on unexpected errors

    _record_failure(circuit)
    logger.error(
        "chembl_request_exhausted",
        endpoint=endpoint,
        last_error=str(last_exc) if last_exc else "unknown",
    )
    return None


async def _chembl_post(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, Any],
    *,
    semaphore: asyncio.Semaphore | None = None,
    timeout_override: float | None = None,
) -> dict | None:
    """Make a POST request to ChEMBL REST API (for large parameter lists).

    Uses X-HTTP-Method-Override: GET header and nested list body format
    per D-16: ``json.dumps([[key, value], ...])`` -- NOT dict format.

    Args:
        client: httpx.AsyncClient instance
        endpoint: API endpoint
        params: Parameters to send as POST body
        semaphore: Optional concurrency limiter
        timeout_override: Override the default read timeout

    Returns:
        JSON response as dict, or None on failure
    """
    circuit = _get_circuit(endpoint)
    if _is_circuit_open(circuit):
        logger.debug("circuit_open_post", endpoint=endpoint)
        return None

    url = f"{settings.CHEMBL_API_URL}/{endpoint}.json"
    body = _json.dumps([[k, v] for k, v in params.items()])
    headers = {
        "X-HTTP-Method-Override": "GET",
        "Content-Type": "application/json",
    }
    request_timeout = (
        httpx.Timeout(connect=5, read=timeout_override, write=10, pool=10)
        if timeout_override
        else client.timeout
    )

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            if semaphore is not None:
                async with semaphore:
                    _start = time.time()
                    response = await client.post(
                        url, content=body, headers=headers, timeout=request_timeout,
                    )
            else:
                _start = time.time()
                response = await client.post(
                    url, content=body, headers=headers, timeout=request_timeout,
                )

            metrics.increment("api_calls_total")
            metrics.record_latency("chembl", (time.time() - _start) * 1000)

            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2))
                logger.warning("chembl_429_post", endpoint=endpoint, retry_after=retry_after)
                await asyncio.sleep(retry_after)
                continue

            if response.status_code >= 500:
                logger.warning("chembl_5xx_post", endpoint=endpoint, status=response.status_code)
                await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
                continue

            response.raise_for_status()
            _record_success(circuit)
            return response.json()

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            metrics.increment("api_calls_failed")
            logger.warning(
                "chembl_post_error",
                endpoint=endpoint,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            metrics.increment("api_calls_failed")
            logger.warning(
                "chembl_post_http_error",
                endpoint=endpoint,
                status=exc.response.status_code,
                attempt=attempt + 1,
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
        except Exception as exc:
            last_exc = exc
            metrics.increment("api_calls_failed")
            logger.error("chembl_post_unexpected", endpoint=endpoint, error=str(exc))
            break

    _record_failure(circuit)
    logger.error(
        "chembl_post_exhausted",
        endpoint=endpoint,
        last_error=str(last_exc) if last_exc else "unknown",
    )
    return None


async def _chembl_request(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, Any],
    *,
    semaphore: asyncio.Semaphore | None = None,
    timeout_override: float | None = None,
) -> dict | None:
    """Smart router: use POST for >200 comma-separated IDs, GET otherwise (D-17).

    Args:
        client: httpx.AsyncClient instance
        endpoint: API endpoint
        params: Query parameters
        semaphore: Optional concurrency limiter
        timeout_override: Override the default read timeout

    Returns:
        JSON response as dict, or None on failure
    """
    # Check if any parameter value has >POST_ID_THRESHOLD comma-separated IDs
    use_post = False
    for v in params.values():
        if isinstance(v, str) and v.count(",") >= POST_ID_THRESHOLD:
            use_post = True
            break

    if use_post:
        return await _chembl_post(
            client, endpoint, params,
            semaphore=semaphore, timeout_override=timeout_override,
        )
    return await _chembl_get(
        client, endpoint, params,
        semaphore=semaphore, timeout_override=timeout_override,
    )


# ---------------------------------------------------------------------------
# Library fallback helpers (sync, run via run_in_executor -- D-30/D-34)
# ---------------------------------------------------------------------------

def _sync_similarity_search(smiles: str, threshold: int) -> list[dict[str, str]] | None:
    """Synchronous similarity search via ChEMBL library (fallback)."""
    client = _get_chembl_client()
    if client is None or "similarity" not in client:
        return None
    try:
        results = client["similarity"].filter(
            smiles=smiles, similarity=threshold,
        ).only(["molecule_chembl_id", "similarity"])
        return [
            {
                "ChEMBL ID": r["molecule_chembl_id"],
                "Similarity": r.get("similarity", 0),
            }
            for r in list(results)
            if "molecule_chembl_id" in r
        ]
    except Exception as exc:
        logger.warning("library_similarity_failed", error=str(exc))
        return None


async def _library_fallback_similarity(
    smiles: str, threshold: int,
) -> list[dict[str, str]] | None:
    """Run synchronous library similarity search in executor (D-30)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _sync_similarity_search, smiles, threshold,
    )


def _sync_activity_fetch(
    chembl_ids: list[str],
    activity_types: list[str],
) -> list[dict] | None:
    """Synchronous activity fetch via ChEMBL library (fallback)."""
    client = _get_chembl_client()
    if client is None or "activity" not in client:
        return None
    try:
        all_activities = []
        activity_types_set = set(activity_types)
        activities = client["activity"].filter(
            molecule_chembl_id__in=chembl_ids,
        ).only([
            "molecule_chembl_id",
            "standard_type",
            "standard_value",
            "standard_units",
            "pchembl_value",
            "target_chembl_id",
            "assay_chembl_id",
            "data_validity_comment",
            "assay_type",
            "document_year",
            "activity_comment",
        ])
        raw = list(activities)
        filtered = [a for a in raw if a.get("standard_type") in activity_types_set]
        all_activities.extend(filtered)
        return all_activities
    except Exception as exc:
        logger.warning("library_activity_failed", error=str(exc))
        return None


async def _library_fallback_activities(
    chembl_ids: list[str],
    activity_types: list[str],
) -> list[dict] | None:
    """Run synchronous library activity fetch in executor (D-30)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _sync_activity_fetch, chembl_ids, activity_types,
    )


def _sync_molecule_fetch(chembl_ids: list[str]) -> dict[str, dict] | None:
    """Synchronous batch molecule fetch via ChEMBL library (fallback)."""
    client = _get_chembl_client()
    if client is None or "molecule" not in client:
        return None
    try:
        molecules = client["molecule"].filter(
            molecule_chembl_id__in=chembl_ids,
        ).only([
            "molecule_chembl_id",
            "pref_name",
            "molecule_properties",
            "molecule_structures",
        ])
        result = {}
        for mol in list(molecules):
            cid = mol.get("molecule_chembl_id")
            if cid:
                result[cid] = mol
        return result
    except Exception as exc:
        logger.warning("library_molecule_failed", error=str(exc))
        return None


async def _library_fallback_molecule_data(
    chembl_ids: list[str],
) -> dict[str, dict] | None:
    """Run synchronous library molecule fetch in executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_molecule_fetch, chembl_ids)


def _sync_target_fetch(target_ids: list[str]) -> dict[str, str] | None:
    """Synchronous batch target fetch via ChEMBL library (fallback)."""
    client = _get_chembl_client()
    if client is None or "target" not in client:
        return None
    try:
        targets = client["target"].filter(
            target_chembl_id__in=target_ids,
        ).only(["target_chembl_id", "pref_name"])
        result = {}
        for t in list(targets):
            tid = t.get("target_chembl_id")
            if tid:
                result[tid] = t.get("pref_name", "") or ""
        return result
    except Exception as exc:
        logger.warning("library_target_failed", error=str(exc))
        return None


async def _library_fallback_target_names(
    target_ids: list[str],
) -> dict[str, str] | None:
    """Run synchronous library target fetch in executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_target_fetch, target_ids)


def _sync_drug_indication_fetch(
    chembl_ids: list[str],
) -> list[dict] | None:
    """Synchronous drug indication fetch via ChEMBL library (fallback)."""
    client = _get_chembl_client()
    if client is None or "drug_indication" not in client:
        return None
    try:
        all_indications: list[dict] = []
        for cid in chembl_ids:
            try:
                indications = client["drug_indication"].filter(molecule_chembl_id=cid)
                for ind in list(indications):
                    ind["molecule_chembl_id"] = cid
                    all_indications.append(ind)
            except Exception:
                pass
        return all_indications
    except Exception as exc:
        logger.warning("library_drug_indication_failed", error=str(exc))
        return None


async def _library_fallback_drug_indications(
    chembl_ids: list[str],
) -> list[dict] | None:
    """Run synchronous library drug indication fetch in executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _sync_drug_indication_fetch, chembl_ids,
    )


# ---------------------------------------------------------------------------
# Public async functions
# ---------------------------------------------------------------------------

async def get_chembl_ids(
    client: httpx.AsyncClient,
    smiles: str,
    similarity_threshold: int = 90,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict[str, str]]:
    """Perform similarity search against ChEMBL.

    REST primary: URL path ``/similarity/{SMILES}/{threshold}.json``
    with ``limit=1000`` and 90s timeout override (D-15).

    On REST failure, falls back to library via run_in_executor (D-34).

    Args:
        client: httpx.AsyncClient instance
        smiles: SMILES string to search
        similarity_threshold: Similarity threshold (0-100)
        semaphore: Optional concurrency limiter

    Returns:
        List of dicts with ``molecule_chembl_id`` keys
    """
    if not smiles:
        return []

    canonical = _canonicalize_smiles(smiles)
    encoded = _url_encode_smiles(canonical)

    # REST primary path -- similarity uses URL path, not query params
    circuit = _get_circuit("similarity")
    if not _is_circuit_open(circuit):
        url = f"{settings.CHEMBL_API_URL}/similarity/{encoded}/{similarity_threshold}.json"
        all_results: list[dict[str, str]] = []
        offset = 0

        while True:
            params: dict[str, Any] = {"limit": CHEMBL_MAX_LIMIT, "offset": offset}
            request_timeout = httpx.Timeout(connect=5, read=90, write=10, pool=10)

            try:
                if semaphore is not None:
                    async with semaphore:
                        _start = time.time()
                        response = await client.get(url, params=params, timeout=request_timeout)
                else:
                    _start = time.time()
                    response = await client.get(url, params=params, timeout=request_timeout)

                metrics.increment("api_calls_total")
                metrics.record_latency("chembl", (time.time() - _start) * 1000)

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 2))
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                data = response.json()
                _record_success(circuit)

                molecules = _get_response_data(data, "similarity")
                if not molecules:
                    break

                for mol in molecules:
                    cid = mol.get("molecule_chembl_id")
                    if cid:
                        similarity = mol.get("similarity", 0)
                        all_results.append({
                            "ChEMBL ID": cid,
                            "Similarity": similarity,
                        })

                if len(molecules) < CHEMBL_MAX_LIMIT:
                    break
                offset += CHEMBL_MAX_LIMIT

            except Exception as exc:
                _record_failure(circuit)
                logger.warning("similarity_rest_failed", error=str(exc))
                break

        if all_results or offset > 0:
            logger.info("similarity_search_complete", count=len(all_results))
            return all_results

    # Fallback to library (D-34)
    logger.info("similarity_falling_back_to_library")
    result = await _library_fallback_similarity(canonical, similarity_threshold)
    if result is not None:
        logger.info("similarity_library_fallback_ok", count=len(result))
        return result

    logger.error("similarity_search_all_failed")
    return []


async def _fetch_activities_for_type(
    client: httpx.AsyncClient,
    chembl_ids: list[str],
    activity_type: str,
    semaphore: asyncio.Semaphore | None,
) -> list[dict]:
    """Fetch activities for a single activity type with parallel pagination.

    First page reveals ``total_count``, remaining pages fetched via
    ``asyncio.gather()`` for parallel pagination (D-22).

    Args:
        client: httpx.AsyncClient instance
        chembl_ids: ChEMBL IDs to query
        activity_type: Single activity type (e.g., 'IC50')
        semaphore: Optional concurrency limiter

    Returns:
        List of activity dicts for this type

    Raises:
        RuntimeError: If REST request fails (all-or-nothing per D-23)
    """
    ids_param = ",".join(chembl_ids)
    params: dict[str, Any] = {
        "molecule_chembl_id__in": ids_param,
        "standard_type": activity_type,
        "only": ACTIVITY_ONLY_FIELDS,
        "limit": CHEMBL_MAX_LIMIT,
        "offset": 0,
    }

    # First page
    data = await _chembl_request(
        client, "activity", params,
        semaphore=semaphore, timeout_override=60,
    )
    if data is None:
        raise RuntimeError(f"Activity fetch failed for type {activity_type}")

    activities = _get_response_data(data, "activity")
    total_count = data.get("page_meta", {}).get("total_count", 0)

    if total_count <= CHEMBL_MAX_LIMIT:
        return activities

    # Parallel pagination for remaining pages
    remaining_offsets = list(range(CHEMBL_MAX_LIMIT, total_count, CHEMBL_MAX_LIMIT))

    async def _fetch_page(offset: int) -> list[dict]:
        page_params = {**params, "offset": offset}
        page_data = await _chembl_request(
            client, "activity", page_params,
            semaphore=semaphore, timeout_override=60,
        )
        if page_data is None:
            raise RuntimeError(
                f"Activity page failed for type {activity_type} at offset {offset}"
            )
        return _get_response_data(page_data, "activity")

    page_results = await asyncio.gather(
        *[_fetch_page(off) for off in remaining_offsets],
    )
    for page_activities in page_results:
        activities.extend(page_activities)

    return activities


async def fetch_all_activities_single_batch(
    client: httpx.AsyncClient,
    chembl_ids: list[str],
    activity_types: list[str] | None = None,
    cancellation_check: Callable[[], None] | None = None,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict]:
    """Fetch ALL activities for multiple ChEMBL IDs using per-type parallel strategy.

    Per-type parallel: one ``_fetch_activities_for_type()`` per activity type,
    combined via ``asyncio.gather()`` (D-20). Each type function does parallel
    pagination internally (D-22).

    All-or-nothing: if any type fails after retries, falls back to library (D-23/D-34).

    Args:
        client: httpx.AsyncClient instance
        chembl_ids: List of ChEMBL IDs to fetch
        activity_types: Activity types to filter (default: DEFAULT_ACTIVITY_TYPES)
        cancellation_check: Optional callable -- call between type fetches, raise if cancelled
        semaphore: Optional concurrency limiter

    Returns:
        List of activity dictionaries filtered to specified types
    """
    if not chembl_ids:
        return []

    if activity_types is None:
        activity_types = DEFAULT_ACTIVITY_TYPES

    # REST primary: per-type parallel strategy
    try:
        tasks = [
            _fetch_activities_for_type(client, chembl_ids, atype, semaphore)
            for atype in activity_types
        ]
        type_results = await asyncio.gather(*tasks)

        all_activities: list[dict] = []
        for i, result in enumerate(type_results):
            all_activities.extend(result)
            if cancellation_check is not None:
                cancellation_check()

        logger.info(
            "activity_fetch_complete",
            total=len(all_activities),
            types=len(activity_types),
        )
        return all_activities

    except RuntimeError as exc:
        logger.warning("activity_rest_failed", error=str(exc))
    except Exception as exc:
        logger.warning("activity_rest_unexpected", error=str(exc))

    # Fallback to library (D-34)
    logger.info("activity_falling_back_to_library")
    result = await _library_fallback_activities(chembl_ids, activity_types)
    if result is not None:
        logger.info("activity_library_fallback_ok", count=len(result))
        return result

    logger.error("activity_fetch_all_failed")
    return []


async def fetch_batch_molecule_data(
    client: httpx.AsyncClient,
    chembl_ids: list[str],
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, dict]:
    """Fetch molecule data for multiple ChEMBL IDs.

    Chunks IDs at POST_ID_THRESHOLD (200) for safety. Uses ``_chembl_request``
    which automatically switches to POST for large ID lists.

    On REST failure, falls back to library via run_in_executor.

    Args:
        client: httpx.AsyncClient instance
        chembl_ids: List of ChEMBL IDs
        semaphore: Optional concurrency limiter

    Returns:
        Dict mapping ChEMBL ID -> molecule data dict
    """
    if not chembl_ids:
        return {}

    unique_ids = list(dict.fromkeys(chembl_ids))
    all_molecules: list[dict] = []

    # Chunk IDs for manageable request sizes
    id_chunks = [
        unique_ids[i : i + POST_ID_THRESHOLD]
        for i in range(0, len(unique_ids), POST_ID_THRESHOLD)
    ]

    rest_failed = False
    for chunk in id_chunks:
        ids_param = ",".join(chunk)
        offset = 0

        while True:
            params: dict[str, Any] = {
                "molecule_chembl_id__in": ids_param,
                "limit": CHEMBL_MAX_LIMIT,
                "offset": offset,
            }

            data = await _chembl_request(
                client, "molecule", params, semaphore=semaphore,
            )
            if data is None:
                rest_failed = True
                break

            molecules = _get_response_data(data, "molecule")
            if not molecules:
                break

            all_molecules.extend(molecules)

            if len(molecules) < CHEMBL_MAX_LIMIT:
                break
            offset += CHEMBL_MAX_LIMIT

        if rest_failed:
            break

    if not rest_failed:
        result: dict[str, dict] = {}
        for mol in all_molecules:
            cid = mol.get("molecule_chembl_id")
            if cid:
                result[cid] = mol
        logger.info("molecule_batch_complete", count=len(result), requested=len(unique_ids))
        return result

    # Fallback to library
    logger.info("molecule_falling_back_to_library")
    lib_result = await _library_fallback_molecule_data(unique_ids)
    if lib_result is not None:
        logger.info("molecule_library_fallback_ok", count=len(lib_result))
        return lib_result

    # Return whatever REST managed to get
    result = {}
    for mol in all_molecules:
        cid = mol.get("molecule_chembl_id")
        if cid:
            result[cid] = mol
    return result


async def fetch_batch_target_names(
    client: httpx.AsyncClient,
    target_ids: list[str],
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, str]:
    """Fetch target names for multiple ChEMBL Target IDs.

    Chunks IDs and uses ``_chembl_request`` for automatic GET/POST routing.

    Args:
        client: httpx.AsyncClient instance
        target_ids: List of ChEMBL Target IDs
        semaphore: Optional concurrency limiter

    Returns:
        Dict mapping target_chembl_id -> target preferred name
    """
    if not target_ids:
        return {}

    unique_ids = list(dict.fromkeys(target_ids))
    all_targets: list[dict] = []

    id_chunks = [
        unique_ids[i : i + POST_ID_THRESHOLD]
        for i in range(0, len(unique_ids), POST_ID_THRESHOLD)
    ]

    rest_failed = False
    for chunk in id_chunks:
        ids_param = ",".join(chunk)
        offset = 0

        while True:
            params: dict[str, Any] = {
                "target_chembl_id__in": ids_param,
                "only": "target_chembl_id,pref_name",
                "limit": CHEMBL_MAX_LIMIT,
                "offset": offset,
            }

            data = await _chembl_request(
                client, "target", params, semaphore=semaphore,
            )
            if data is None:
                rest_failed = True
                break

            targets = _get_response_data(data, "target")
            if not targets:
                break

            all_targets.extend(targets)

            if len(targets) < CHEMBL_MAX_LIMIT:
                break
            offset += CHEMBL_MAX_LIMIT

        if rest_failed:
            break

    if not rest_failed:
        result: dict[str, str] = {}
        for t in all_targets:
            tid = t.get("target_chembl_id")
            if tid:
                result[tid] = t.get("pref_name", "") or ""
        logger.info("target_batch_complete", count=len(result), requested=len(unique_ids))
        return result

    # Fallback to library
    logger.info("target_falling_back_to_library")
    lib_result = await _library_fallback_target_names(unique_ids)
    if lib_result is not None:
        logger.info("target_library_fallback_ok", count=len(lib_result))
        return lib_result

    result = {}
    for t in all_targets:
        tid = t.get("target_chembl_id")
        if tid:
            result[tid] = t.get("pref_name", "") or ""
    return result


async def get_drug_indications_batch(
    client: httpx.AsyncClient,
    chembl_ids: list[str],
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple:
    """Fetch drug indications for multiple ChEMBL IDs using batch REST API.

    Returns ``(indication_list, indication_by_compound_dict)`` to match the
    existing caller contract in compound_service.

    Args:
        client: httpx.AsyncClient instance
        chembl_ids: List of ChEMBL molecule IDs
        semaphore: Optional concurrency limiter

    Returns:
        Tuple of (all_indications_list, dict mapping ChEMBL ID -> list of indications)
    """
    if not chembl_ids:
        return ([], {})

    unique_ids = list(dict.fromkeys(chembl_ids))
    all_indications: list[dict] = []

    id_chunks = [
        unique_ids[i : i + POST_ID_THRESHOLD]
        for i in range(0, len(unique_ids), POST_ID_THRESHOLD)
    ]

    rest_failed = False
    for chunk in id_chunks:
        ids_param = ",".join(chunk)
        offset = 0

        while True:
            params: dict[str, Any] = {
                "molecule_chembl_id__in": ids_param,
                "limit": CHEMBL_MAX_LIMIT,
                "offset": offset,
            }

            data = await _chembl_request(
                client, "drug_indication", params, semaphore=semaphore,
            )
            if data is None:
                rest_failed = True
                break

            indications = _get_response_data(data, "drug_indication")
            if not indications:
                break

            all_indications.extend(indications)

            if len(indications) < CHEMBL_MAX_LIMIT:
                break
            offset += CHEMBL_MAX_LIMIT

        if rest_failed:
            break

    if rest_failed:
        logger.info("drug_indication_falling_back_to_library")
        lib_result = await _library_fallback_drug_indications(unique_ids)
        if lib_result is not None:
            all_indications = lib_result

    # Build per-compound dict
    by_compound: dict[str, list[dict]] = {cid: [] for cid in unique_ids}
    for ind in all_indications:
        cid = ind.get("molecule_chembl_id")
        if cid and cid in by_compound:
            # Extract clinical trial info
            clinical_trials_url = ""
            clinical_trials_ids = ""
            indication_refs = ind.get("indication_refs", [])
            if indication_refs:
                for ref in indication_refs:
                    if ref.get("ref_type") == "ClinicalTrials":
                        clinical_trials_url = ref.get("ref_url", "")
                        clinical_trials_ids = ref.get("ref_id", "")
                        break

            by_compound[cid].append({
                "ChEMBL_ID": cid,
                "MESH_ID": ind.get("mesh_id", ""),
                "MESH_Heading": ind.get("mesh_heading", ""),
                "EFO_ID": ind.get("efo_id", ""),
                "EFO_Term": ind.get("efo_term", ""),
                "Max_Phase": ind.get("max_phase_for_ind", 0),
                "Clinical_Trials_URL": clinical_trials_url,
                "Clinical_Trials_IDs": clinical_trials_ids,
            })

    logger.info("drug_indication_batch_complete", total=len(all_indications), compounds=len(unique_ids))
    return (all_indications, by_compound)


async def cascade_similarity_counts(
    client: httpx.AsyncClient,
    smiles: str,
    start_threshold: int,
    min_threshold: int = 40,
    step: int = 10,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict[str, int]]:
    """Probe lower similarity thresholds and return compound count per tier.

    Sequential threshold probing (each depends on previous for early abort).
    Uses REST API with ``limit=1`` so only ``page_meta.total_count`` is transferred.

    Args:
        client: httpx.AsyncClient instance
        smiles: Query SMILES string
        start_threshold: The user's original threshold (probing starts one step below)
        min_threshold: Lowest threshold to probe (default 40%)
        step: How much to decrease the threshold each iteration
        semaphore: Optional concurrency limiter

    Returns:
        List of ``{"threshold": int, "count": int}`` dicts, one per probed tier.
    """
    if not smiles:
        return []

    canonical = _canonicalize_smiles(smiles)
    encoded = _url_encode_smiles(canonical)

    results: list[dict[str, int]] = []
    threshold = start_threshold - step
    consecutive_failures = 0

    while threshold >= min_threshold:
        url = f"{settings.CHEMBL_API_URL}/similarity/{encoded}/{threshold}.json"
        try:
            request_timeout = httpx.Timeout(connect=5, read=90, write=10, pool=10)
            if semaphore is not None:
                async with semaphore:
                    response = await client.get(
                        url,
                        params={"limit": 1, "only": "molecule_chembl_id"},
                        timeout=request_timeout,
                    )
            else:
                response = await client.get(
                    url,
                    params={"limit": 1, "only": "molecule_chembl_id"},
                    timeout=request_timeout,
                )
            response.raise_for_status()
            data = response.json()
            count = data.get("page_meta", {}).get("total_count", 0)
            results.append({"threshold": threshold, "count": count})
            consecutive_failures = 0
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            consecutive_failures += 1
            logger.warning("cascade_probe_failed", threshold=threshold, error=str(exc))
            results.append({"threshold": threshold, "count": 0})
            if consecutive_failures >= 2:
                logger.info("cascade_probe_aborted", reason="unreachable after 2 consecutive failures")
                break
        except Exception as exc:
            logger.warning("cascade_probe_error", threshold=threshold, error=str(exc))
            results.append({"threshold": threshold, "count": 0})

        threshold -= step

    return results


async def probe_all_thresholds(
    client: httpx.AsyncClient,
    smiles: str,
    start_threshold: int,
    min_threshold: int = 40,
    step: int = 10,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict[str, int]]:
    """Probe ALL thresholds from start down to min, returning count per tier.

    Unlike cascade_similarity_counts, this:
    - Includes the start_threshold itself
    - Probes ALL tiers in parallel via asyncio.gather()
    - Returns ALL tiers including 0-count ones

    Args:
        client: httpx.AsyncClient instance
        smiles: Query SMILES string
        start_threshold: The user's requested threshold (included in probing)
        min_threshold: Lowest threshold to probe (default 40%)
        step: How much to decrease the threshold each iteration
        semaphore: Optional concurrency limiter

    Returns:
        List of ``{"threshold": int, "count": int}`` dicts for every tier.
    """
    if not smiles:
        return []

    canonical = _canonicalize_smiles(smiles)
    encoded = _url_encode_smiles(canonical)

    thresholds = list(range(start_threshold, min_threshold - 1, -step))

    async def _probe(threshold: int) -> dict[str, int]:
        url = f"{settings.CHEMBL_API_URL}/similarity/{encoded}/{threshold}.json"
        try:
            request_timeout = httpx.Timeout(connect=5, read=90, write=10, pool=10)
            if semaphore is not None:
                async with semaphore:
                    response = await client.get(
                        url,
                        params={"limit": 1, "only": "molecule_chembl_id"},
                        timeout=request_timeout,
                    )
            else:
                response = await client.get(
                    url,
                    params={"limit": 1, "only": "molecule_chembl_id"},
                    timeout=request_timeout,
                )
            response.raise_for_status()
            data = response.json()
            count = data.get("page_meta", {}).get("total_count", 0)
            return {"threshold": threshold, "count": count}
        except Exception as exc:
            logger.warning("probe_threshold_failed", threshold=threshold, error=str(exc))
            return {"threshold": threshold, "count": 0}

    results = await asyncio.gather(*[_probe(t) for t in thresholds])
    return list(results)


async def quick_has_bioactivity(
    client: httpx.AsyncClient,
    smiles: str,
    threshold: int = 90,
    activity_types: list[str] | None = None,
    *,
    semaphore: asyncio.Semaphore | None = None,
) -> bool:
    """Fast pre-flight check: are there similar compounds WITH bioactivity data?

    Two-step check:
    1. Similarity search to get ChEMBL IDs (limit=5 for speed)
    2. Activity check on those IDs for requested activity types

    Returns True if at least one similar compound has usable activity data.
    On any error, returns True (optimistic -- let processing handle it).

    Args:
        client: httpx.AsyncClient instance
        smiles: SMILES string
        threshold: Similarity threshold
        activity_types: Optional list of activity types to check
        semaphore: Optional concurrency limiter

    Returns:
        True if bioactivity data exists, True on error (optimistic)
    """
    if not smiles:
        return False

    canonical = _canonicalize_smiles(smiles)
    encoded = _url_encode_smiles(canonical)

    try:
        # Step 1: Get a few similar compound ChEMBL IDs
        url = f"{settings.CHEMBL_API_URL}/similarity/{encoded}/{threshold}.json"
        request_timeout = httpx.Timeout(connect=5, read=90, write=10, pool=10)

        if semaphore is not None:
            async with semaphore:
                resp = await client.get(
                    url,
                    params={"limit": 5, "only": "molecule_chembl_id"},
                    timeout=request_timeout,
                )
        else:
            resp = await client.get(
                url,
                params={"limit": 5, "only": "molecule_chembl_id"},
                timeout=request_timeout,
            )
        resp.raise_for_status()
        data = resp.json()

        total = data.get("page_meta", {}).get("total_count", 0)
        if total == 0:
            return False

        molecules = data.get("molecules", [])
        chembl_ids = [m["molecule_chembl_id"] for m in molecules if "molecule_chembl_id" in m]
        if not chembl_ids:
            return False

        # Step 2: Check if ANY of these have activity data
        ids_param = ",".join(chembl_ids)
        act_params: dict[str, Any] = {
            "molecule_chembl_id__in": ids_param,
            "limit": 1,
            "only": "molecule_chembl_id",
        }
        if activity_types:
            act_params["standard_type__in"] = ",".join(activity_types)

        act_data = await _chembl_request(
            client, "activity", act_params, semaphore=semaphore,
        )
        if act_data is None:
            return True  # Optimistic on REST failure

        act_count = act_data.get("page_meta", {}).get("total_count", 0)
        return act_count > 0

    except Exception as exc:
        logger.warning("quick_bioactivity_check_failed", error=str(exc))
        return True  # Optimistic on error


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def clear_caches() -> None:
    """Clear all caches.

    No-op placeholder that maintains API compatibility.
    cache_non_none uses asyncio.Lock (Phase 19.2).
    """
    logger.info("api_client_caches_cleared")


def get_cache_info() -> dict[str, Any]:
    """Get cache statistics.

    cache_non_none uses asyncio.Lock (Phase 19.2).
    """
    return {
        "note": "cache_non_none with asyncio.Lock available",
    }


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown_api_client() -> None:
    """Shutdown the API client.

    In 19.1+, no ThreadPoolExecutor or thread-local sessions to clean up.
    httpx.AsyncClient is closed by its caller (per-job lifecycle in 19.1,
    module-level in 19.2).
    """
    logger.info("api_client_shutdown")
