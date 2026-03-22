"""
RCSB PDB Client Module (Async)

Provides integration with RCSB Protein Data Bank for structural evidence scoring.

Component 4 of IMP scoring: PDB Structural Evidence Score (5% weight)
- Query PDB for compound or close analogs
- Extract resolution data (X-ray crystallography quality)
- Count structures with binding affinity data
- Score based on structural validation

Resolution Quality Classes:
- *** Best: < 2.0 A (high confidence)
- ** Medium: 2.0-3.0 A (moderate confidence)
- * Poor: > 3.0 A (low confidence)

All I/O functions are async and accept an httpx.AsyncClient as first parameter.
Circuit breaker helpers are self-contained (no imports from api_client.py).
"""

import asyncio
import time
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.metrics import metrics
from backend.modules.api_client import cache_non_none

logger = structlog.get_logger()

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API_URL = "https://data.rcsb.org/rest/v1/core"
GRAPHQL_URL = "https://data.rcsb.org/graphql"

DEFAULT_SIMILARITY_THRESHOLD = 0.9

# --------------------------------------------------------------------------- #
# Client factory
# --------------------------------------------------------------------------- #


def create_pdb_client() -> httpx.AsyncClient:
    """Create httpx client for PDB API calls."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=30, write=10, pool=10),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )


# --------------------------------------------------------------------------- #
# Self-contained circuit breaker helpers (D-31, D-33)
# --------------------------------------------------------------------------- #

_circuits: dict[str, dict] = {}


def _make_circuit(threshold: int = 3, cooldown: int = 300) -> dict:
    """Create a new circuit breaker state dict."""
    return {"failures": 0, "open_until": 0.0, "threshold": threshold, "cooldown": cooldown}


def _get_circuit(endpoint: str) -> dict:
    """Get or create a circuit for the given endpoint."""
    if endpoint not in _circuits:
        _circuits[endpoint] = _make_circuit()
    return _circuits[endpoint]


def _is_circuit_open(circuit: dict) -> bool:
    """Return True if the circuit is open (should skip call)."""
    if circuit["failures"] < circuit["threshold"]:
        return False
    if time.monotonic() >= circuit["open_until"]:
        circuit["failures"] = circuit["threshold"] - 1  # Half-open
        return False
    return True


def _record_success(circuit: dict) -> None:
    """Reset circuit on success."""
    circuit["failures"] = 0
    circuit["open_until"] = 0.0


def _record_failure(circuit: dict) -> None:
    """Increment failures; open circuit if threshold reached."""
    circuit["failures"] += 1
    if circuit["failures"] >= circuit["threshold"]:
        circuit["open_until"] = time.monotonic() + circuit["cooldown"]


# --------------------------------------------------------------------------- #
# Retry decorator factory
# --------------------------------------------------------------------------- #

_RETRYABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)


def _pdb_retry(max_attempts: int = 5):
    """Tenacity retry for PDB calls: 5x exponential backoff (D-29)."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )


# --------------------------------------------------------------------------- #
# Async I/O functions
# --------------------------------------------------------------------------- #


@cache_non_none(maxsize=500, ttl_seconds=86400)
async def search_similar_ligands(
    client: httpx.AsyncClient,
    smiles: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Search RCSB PDB for ligands similar to the query compound.

    Uses chemical similarity search via the RCSB Search API v2.

    Args:
        client: httpx.AsyncClient instance
        smiles: SMILES string of query compound
        similarity_threshold: Tanimoto similarity threshold (0.0-1.0)
            Note: graph-relaxed match type is always used.

    Returns:
        List of PDB IDs containing similar ligands (max 100)
    """
    circuit = _get_circuit("pdb_search")
    if _is_circuit_open(circuit):
        logger.warning("pdb_search_circuit_open", smiles=smiles[:50])
        return []

    query_payload = {
        "query": {
            "type": "terminal",
            "service": "chemical",
            "parameters": {
                "value": smiles,
                "type": "descriptor",
                "descriptor_type": "SMILES",
                "match_type": "graph-relaxed",
            },
        },
        "request_options": {"return_all_hits": True},
        "return_type": "entry",
    }

    @_pdb_retry(max_attempts=5)
    async def _do_search() -> list[str]:
        _start = time.time()
        response = await client.post(SEARCH_API_URL, json=query_payload, timeout=45)
        latency_ms = (time.time() - _start) * 1000

        metrics.increment("api_calls_total")
        metrics.record_latency("pdb", latency_ms)

        if response.status_code == 204:
            logger.info("pdb_search_no_results", smiles=smiles[:50])
            return []

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "5"))
            await asyncio.sleep(retry_after)
            raise httpx.ReadTimeout(f"PDB 429 retry-after {retry_after}s")

        if response.status_code >= 500:
            metrics.increment("api_calls_failed")
            raise httpx.ReadTimeout(f"PDB server error {response.status_code}")

        if response.status_code != 200:
            logger.warning("pdb_search_unexpected_status", status=response.status_code)
            metrics.increment("api_calls_failed")
            return []

        result = response.json()
        pdb_ids = [entry["identifier"] for entry in result.get("result_set", [])]
        pdb_ids = pdb_ids[:100]
        logger.info("pdb_search_found", count=len(pdb_ids), smiles=smiles[:50])
        return pdb_ids

    try:
        result = await _do_search()
        _record_success(circuit)
        return result
    except Exception as exc:
        _record_failure(circuit)
        logger.warning("pdb_search_failed", smiles=smiles[:50], error=str(exc))
        metrics.increment("api_calls_failed")
        return []


@cache_non_none(maxsize=500, ttl_seconds=604800)
async def get_structure_details(
    client: httpx.AsyncClient,
    pdb_id: str,
) -> dict[str, Any]:
    """
    Retrieve detailed information for a PDB structure.

    Args:
        client: httpx.AsyncClient instance
        pdb_id: PDB identifier (e.g., "4HHB")

    Returns:
        Dictionary with pdb_id, title, resolution, doi, uniprot_ids, url,
        experimental_method.
    """
    circuit = _get_circuit("pdb_details")
    result: dict[str, Any] = {
        "pdb_id": pdb_id,
        "title": None,
        "resolution": None,
        "doi": None,
        "uniprot_ids": [],
        "url": f"https://www.rcsb.org/structure/{pdb_id}",
        "experimental_method": None,
    }

    if _is_circuit_open(circuit):
        logger.warning("pdb_details_circuit_open", pdb_id=pdb_id)
        return result

    @_pdb_retry(max_attempts=5)
    async def _fetch_entry() -> dict:
        resp = await client.get(f"{DATA_API_URL}/entry/{pdb_id}", timeout=30)
        if resp.status_code != 200:
            return {}
        return resp.json()

    @_pdb_retry(max_attempts=5)
    async def _fetch_entity() -> dict:
        resp = await client.get(f"{DATA_API_URL}/polymer_entity/{pdb_id}/1", timeout=30)
        if resp.status_code != 200:
            return {}
        return resp.json()

    try:
        entry_data, entity_data = await asyncio.gather(
            _fetch_entry(), _fetch_entity(), return_exceptions=True
        )

        # Process entry data
        if isinstance(entry_data, dict) and entry_data:
            if "struct" in entry_data and "title" in entry_data["struct"]:
                result["title"] = entry_data["struct"]["title"]

            if "rcsb_entry_info" in entry_data:
                res_list = entry_data["rcsb_entry_info"].get("resolution_combined", [])
                if res_list:
                    result["resolution"] = float(res_list[0])

            if "exptl" in entry_data and entry_data["exptl"]:
                result["experimental_method"] = entry_data["exptl"][0].get("method")

            if "rcsb_primary_citation" in entry_data:
                result["doi"] = entry_data["rcsb_primary_citation"].get(
                    "pdbx_database_id_DOI"
                )

        # Process entity data
        if isinstance(entity_data, dict) and entity_data:
            ids_container = entity_data.get("rcsb_polymer_entity_container_identifiers", {})
            for ref in ids_container.get("reference_sequence_identifiers", []):
                if ref.get("database_name") == "UniProt":
                    uid = ref.get("database_accession")
                    if uid and uid not in result["uniprot_ids"]:
                        result["uniprot_ids"].append(uid)

        _record_success(circuit)
    except Exception as exc:
        _record_failure(circuit)
        logger.error("pdb_details_failed", pdb_id=pdb_id, error=str(exc))

    return result


async def get_batch_structure_resolutions_graphql(
    client: httpx.AsyncClient,
    pdb_ids: list[str],
) -> dict[str, float | None]:
    """
    Fetch resolutions via a single GraphQL query (9.5x faster than REST).

    Args:
        client: httpx.AsyncClient instance
        pdb_ids: List of PDB identifiers

    Returns:
        Dict mapping PDB ID -> resolution (float) or None
    """
    if not pdb_ids:
        return {}

    circuit = _get_circuit("pdb_graphql")
    if _is_circuit_open(circuit):
        logger.warning("pdb_graphql_circuit_open")
        return {}

    pdb_ids_normalized = [pid.upper() for pid in pdb_ids]

    graphql_query = """
    query($ids: [String!]!) {
        entries(entry_ids: $ids) {
            rcsb_id
            rcsb_entry_info {
                resolution_combined
            }
        }
    }
    """

    try:
        _gql_start = time.time()
        response = await client.post(
            GRAPHQL_URL,
            json={"query": graphql_query, "variables": {"ids": pdb_ids_normalized}},
            timeout=60,
        )
        latency_ms = (time.time() - _gql_start) * 1000

        metrics.increment("api_calls_total")
        metrics.record_latency("pdb", latency_ms)

        resolutions: dict[str, float | None] = {}
        if response.status_code == 200:
            data = response.json()
            for entry in data.get("data", {}).get("entries", []) or []:
                pid = entry.get("rcsb_id")
                res_list = entry.get("rcsb_entry_info", {}).get("resolution_combined", [])
                resolutions[pid] = res_list[0] if res_list else None
            logger.info(
                "pdb_graphql_resolutions",
                fetched=len(resolutions),
                requested=len(pdb_ids),
            )
            _record_success(circuit)
        else:
            metrics.increment("api_calls_failed")
            logger.warning("pdb_graphql_error", status=response.status_code)
            _record_failure(circuit)

        return resolutions

    except Exception as exc:
        metrics.increment("api_calls_failed")
        _record_failure(circuit)
        logger.error("pdb_graphql_failed", error=str(exc))
        return {}


async def _fetch_single_resolution(
    client: httpx.AsyncClient,
    pdb_id: str,
) -> tuple[str, float | None]:
    """Fetch resolution for a single PDB ID via REST (used as fallback)."""

    @_pdb_retry(max_attempts=5)
    async def _do_fetch() -> float | None:
        resp = await client.get(f"{DATA_API_URL}/entry/{pdb_id}", timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        res_list = data.get("rcsb_entry_info", {}).get("resolution_combined", [])
        return float(res_list[0]) if res_list else None

    try:
        resolution = await _do_fetch()
        return (pdb_id.upper(), resolution)
    except Exception as exc:
        logger.debug("pdb_rest_resolution_failed", pdb_id=pdb_id, error=str(exc))
        return (pdb_id.upper(), None)


async def _fetch_resolutions_parallel_rest(
    client: httpx.AsyncClient,
    pdb_ids: list[str],
) -> dict[str, float | None]:
    """Fetch resolutions via parallel async REST calls using asyncio.gather."""
    results = await asyncio.gather(
        *[_fetch_single_resolution(client, pid) for pid in pdb_ids],
        return_exceptions=True,
    )
    out: dict[str, float | None] = {}
    for item in results:
        if isinstance(item, tuple):
            out[item[0]] = item[1]
        # Exceptions are silently dropped (PDB is non-critical)
    return out


async def get_batch_structure_resolutions(
    client: httpx.AsyncClient,
    pdb_ids: list[str],
) -> dict[str, float | None]:
    """
    Retrieve resolutions for multiple PDB structures.

    Uses GraphQL primary, falls back to parallel REST if GraphQL fails (D-35).

    Args:
        client: httpx.AsyncClient instance
        pdb_ids: List of PDB identifiers (e.g., ["4HHB", "3WHM", "2CPK"])

    Returns:
        Dictionary mapping PDB ID to resolution (float or None)
    """
    if not pdb_ids:
        return {}

    logger.info("pdb_batch_resolutions_start", count=len(pdb_ids))

    # Try GraphQL first (fast)
    resolutions = await get_batch_structure_resolutions_graphql(client, pdb_ids)

    if resolutions:
        pdb_ids_upper = [pid.upper() for pid in pdb_ids]
        missing = [pid for pid in pdb_ids_upper if pid not in resolutions]

        if missing:
            logger.debug("pdb_graphql_missing", count=len(missing))
            # Retry GraphQL for missing IDs
            retry_results = await get_batch_structure_resolutions_graphql(client, missing)
            resolutions.update(retry_results)

            # Still missing? Use parallel REST as final fallback
            still_missing = [pid for pid in missing if pid not in resolutions]
            if still_missing:
                logger.debug("pdb_rest_fallback", count=len(still_missing))
                rest_results = await _fetch_resolutions_parallel_rest(client, still_missing)
                resolutions.update(rest_results)

        resolved = len([r for r in resolutions.values() if r is not None])
        logger.info("pdb_batch_resolutions_done", resolved=resolved, total=len(pdb_ids))
        return resolutions

    # GraphQL failed completely, fall back to parallel REST
    logger.warning("pdb_graphql_total_fail_rest_fallback")
    resolutions = await _fetch_resolutions_parallel_rest(client, pdb_ids)

    resolved = len([r for r in resolutions.values() if r is not None])
    logger.info("pdb_batch_resolutions_done", resolved=resolved, total=len(pdb_ids))
    return resolutions


# --------------------------------------------------------------------------- #
# Pure functions (sync, no I/O)
# --------------------------------------------------------------------------- #


def classify_resolution_quality(resolution: float) -> tuple[str, float]:
    """
    Classify resolution quality and return quality score.

    Args:
        resolution: Resolution in Angstroms

    Returns:
        Tuple of (quality_class, quality_multiplier)
        - quality_class: "***" (best), "**" (medium), "*" (poor)
        - quality_multiplier: 1.0, 0.75, or 0.5
    """
    if resolution < 2.0:
        return ("***", 1.0)  # Best quality
    elif resolution <= 3.0:
        return ("**", 0.75)  # Medium quality
    else:
        return ("*", 0.5)  # Poor quality


# --------------------------------------------------------------------------- #
# High-level scoring functions
# --------------------------------------------------------------------------- #


async def get_pdb_evidence_score(
    client: httpx.AsyncClient,
    smiles: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Calculate PDB Structural Evidence Score (Component 4 of IMP scoring).

    Args:
        client: httpx.AsyncClient instance
        smiles: SMILES string of query compound
        similarity_threshold: Tanimoto similarity threshold (default 0.9)

    Returns:
        Dictionary containing pdb_score, num_structures, num_high_quality,
        num_medium_quality, num_poor_quality, pdb_ids, resolutions, quality_classes.

    Scoring Logic:
        Base score = min(num_structures_with_resolution / 5.0, 1.0)
        Quality-adjusted = (sum of quality_multipliers) / num_with_resolution
        Final = average of base and quality-adjusted
    """
    logger.info("pdb_evidence_score_start", smiles=smiles[:50])

    empty_result: dict[str, Any] = {
        "pdb_score": 0.0,
        "num_structures": 0,
        "num_high_quality": 0,
        "num_medium_quality": 0,
        "num_poor_quality": 0,
        "pdb_ids": [],
        "resolutions": [],
        "quality_classes": [],
    }

    # Step 1: Search for similar ligands
    pdb_ids = await search_similar_ligands(client, smiles, similarity_threshold)
    if not pdb_ids:
        logger.info("pdb_no_similar_structures")
        return empty_result

    # Step 2: Fetch resolutions (GraphQL primary, REST fallback)
    resolution_dict = await get_batch_structure_resolutions(client, pdb_ids)

    resolutions: list[float | None] = []
    quality_classes: list[str] = []
    quality_multipliers: list[float] = []

    for pdb_id in pdb_ids:
        resolution = resolution_dict.get(pdb_id.upper())
        if resolution is not None:
            resolutions.append(resolution)
            quality_class, quality_mult = classify_resolution_quality(resolution)
            quality_classes.append(quality_class)
            quality_multipliers.append(quality_mult)
        else:
            resolutions.append(None)
            quality_classes.append("N/A")
            quality_multipliers.append(0.0)

    # Step 3: Counts
    num_high = sum(1 for q in quality_classes if q == "***")
    num_medium = sum(1 for q in quality_classes if q == "**")
    num_poor = sum(1 for q in quality_classes if q == "*")
    num_with_resolution = num_high + num_medium + num_poor

    # Step 4: Score
    if num_with_resolution == 0:
        pdb_score = 0.0
    else:
        base_score = min(num_with_resolution / 5.0, 1.0)
        quality_weighted = sum(quality_multipliers) / num_with_resolution
        pdb_score = (base_score + quality_weighted) / 2.0

    logger.info(
        "pdb_evidence_score_done",
        score=round(pdb_score, 3),
        structures=num_with_resolution,
        high=num_high,
        medium=num_medium,
        poor=num_poor,
    )

    return {
        "pdb_score": pdb_score,
        "num_structures": len(pdb_ids),
        "num_high_quality": num_high,
        "num_medium_quality": num_medium,
        "num_poor_quality": num_poor,
        "pdb_ids": pdb_ids,
        "resolutions": resolutions,
        "quality_classes": quality_classes,
    }


async def get_detailed_pdb_structures(
    client: httpx.AsyncClient,
    smiles: str,
) -> list[dict[str, Any]]:
    """
    Get detailed information for all PDB structures matching a compound.

    Args:
        client: httpx.AsyncClient instance
        smiles: SMILES string of query compound

    Returns:
        List of dicts sorted by quality (*** first) then resolution (best first).
    """
    logger.info("pdb_detailed_structures_start", smiles=smiles[:50])

    pdb_result = await get_pdb_evidence_score(client, smiles)
    if not pdb_result["pdb_ids"]:
        return []

    # Fetch details in parallel via asyncio.gather
    details_list = await asyncio.gather(
        *[get_structure_details(client, pid) for pid in pdb_result["pdb_ids"]],
        return_exceptions=True,
    )

    detailed_structures: list[dict[str, Any]] = []
    for i, details in enumerate(details_list):
        pdb_id = pdb_result["pdb_ids"][i]
        resolution = pdb_result["resolutions"][i]
        quality_class = pdb_result["quality_classes"][i]

        if isinstance(details, Exception):
            details = {
                "pdb_id": pdb_id,
                "title": "Error fetching details",
                "uniprot_ids": [],
                "experimental_method": None,
                "url": f"https://www.rcsb.org/structure/{pdb_id}",
            }

        detailed_structures.append({
            "PDB_ID": details.get("pdb_id", pdb_id),
            "Title": details.get("title") or "N/A",
            "Resolution": resolution if resolution is not None else 999.0,
            "Quality": quality_class,
            "UniProt_IDs": ",".join(details.get("uniprot_ids", [])) or "N/A",
            "Experimental_Method": details.get("experimental_method") or "N/A",
            "URL": details.get("url", f"https://www.rcsb.org/structure/{pdb_id}"),
        })

    # Sort by quality then resolution
    quality_order = {"***": 1, "**": 2, "*": 3, "N/A": 4}
    detailed_structures.sort(
        key=lambda x: (quality_order.get(x["Quality"], 4), x["Resolution"])
    )

    # Convert sentinel back to N/A for display
    for s in detailed_structures:
        if s["Resolution"] == 999.0:
            s["Resolution"] = "N/A"

    logger.info("pdb_detailed_structures_done", count=len(detailed_structures))
    return detailed_structures


async def get_pdb_summary_for_compound(
    client: httpx.AsyncClient,
    smiles: str,
) -> str:
    """
    Generate a human-readable summary of PDB evidence for a compound.

    Args:
        client: httpx.AsyncClient instance
        smiles: SMILES string of query compound

    Returns:
        Formatted string summarizing PDB evidence
    """
    result = await get_pdb_evidence_score(client, smiles)

    if result["num_structures"] == 0:
        return "No experimental structures found in PDB for this compound or close analogs."

    parts: list[str] = []
    parts.append(f"Found {result['num_structures']} similar structure(s) in PDB")
    parts.append(f"PDB Evidence Score: {result['pdb_score']:.3f}/1.0")

    if result["num_high_quality"] > 0:
        parts.append(f"- {result['num_high_quality']} high-quality (*** < 2.0 A)")
    if result["num_medium_quality"] > 0:
        parts.append(f"- {result['num_medium_quality']} medium-quality (** 2.0-3.0 A)")
    if result["num_poor_quality"] > 0:
        parts.append(f"- {result['num_poor_quality']} poor-quality (* > 3.0 A)")

    parts.append("\nTop PDB Entries:")
    for pdb_id, resolution, quality in zip(
        result["pdb_ids"][:5],
        result["resolutions"][:5],
        result["quality_classes"][:5],
    ):
        if resolution is not None:
            parts.append(f"  {pdb_id}: {resolution:.2f} A ({quality})")
        else:
            parts.append(f"  {pdb_id}: Resolution N/A")

    return "\n".join(parts)

