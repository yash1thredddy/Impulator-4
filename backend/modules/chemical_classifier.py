"""
Chemical Classification Module (Async)

Integrates ClassyFire and NPClassifier for comprehensive chemical taxonomy.

ClassyFire: General chemical taxonomy (Kingdom -> Subclass)
NPClassifier: Natural product-specific classification (Pathway -> Class)

All I/O functions are async and accept an httpx.AsyncClient as first parameter.
Circuit breaker helpers are self-contained (no imports from api_client.py).

Usage:
    from backend.modules.chemical_classifier import get_complete_classification

    async with create_classifier_client() as client:
        classification = await get_complete_classification(client, smiles="...", inchikey="...")
        print(classification['Class'])       # ClassyFire class
        print(classification['NP_Pathway'])  # NPClassifier pathway
"""

import asyncio
import time

import httpx
import structlog
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)

from backend.core.metrics import metrics

logger = structlog.get_logger()

# ClassyFire triple-endpoint rotation (revised 2026-05-14, ported from Nobs_Classification).
# WishartLab (the original sole endpoint) suffers frequent outages; Fiehn lab and GNPS
# are academic mirrors. Rotation order: Fiehn → GNPS → WishartLab. Each mirror has its
# own circuit breaker so a flapping endpoint cannot poison the others.
_CLASSYFIRE_FIEHN = "https://cfb.fiehnlab.ucdavis.edu/entities/{key}.json"
_CLASSYFIRE_GNPS = "https://gnps-classyfire.ucsd.edu/entities/{key}.json"
_CLASSYFIRE_WISHART = "http://classyfire.wishartlab.com/entities/{key}.json"

# --------------------------------------------------------------------------- #
# Client factory
# --------------------------------------------------------------------------- #


def create_classifier_client() -> httpx.AsyncClient:
    """Create httpx client for ClassyFire/NPClassifier."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=15, write=10, pool=10),
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


# Pre-create circuits with higher threshold for non-critical classifiers (D-34).
# Each ClassyFire mirror has its own breaker so a flapping endpoint can't poison the others.
_circuits["classyfire_fiehn"] = _make_circuit(threshold=5, cooldown=300)
_circuits["classyfire_gnps"] = _make_circuit(threshold=5, cooldown=300)
_circuits["classyfire_wishart"] = _make_circuit(threshold=5, cooldown=300)
_circuits["npclassifier"] = _make_circuit(threshold=5, cooldown=300)


# --------------------------------------------------------------------------- #
# Retry decorator
# --------------------------------------------------------------------------- #

# Sentinel returned by request handlers on 429/503 so tenacity retries the whole call
# without raising (HTTPStatusError on rate-limit responses isn't ideal because we want
# to honour upstream backpressure rather than treat it as a hard failure).
_RATE_LIMITED = object()


def _classifier_retry(max_attempts: int = 5):
    """Tenacity wrapper with full-jitter backoff for shared-infra friendliness.

    Retries on httpx network errors, HTTPStatusError (5xx via raise_for_status), and
    when the wrapped function returns the _RATE_LIMITED sentinel (429/503 responses).
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=(
            retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
            | retry_if_result(lambda r: r is _RATE_LIMITED)
        ),
        reraise=True,
    )


# --------------------------------------------------------------------------- #
# Async I/O functions
# --------------------------------------------------------------------------- #


@_classifier_retry()
async def _classyfire_request(
    client: httpx.AsyncClient, url: str
) -> dict | object | None:
    """Single-endpoint ClassyFire request.

    Returns dict on 200, _RATE_LIMITED sentinel on 429/503 (tenacity retries),
    None on 404, raises on other 5xx (handled by tenacity).
    """
    _start = time.perf_counter()
    response = await client.get(url)
    latency_ms = (time.perf_counter() - _start) * 1000
    metrics.increment("api_calls_total")
    metrics.record_latency("classyfire", latency_ms)

    if response.status_code in (429, 503):
        return _RATE_LIMITED
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


async def get_classyfire_classification(
    client: httpx.AsyncClient,
    inchikey: str,
) -> dict | None:
    """
    Get ClassyFire classification with triple-endpoint rotation.

    Rotates through Fiehn lab → GNPS → WishartLab, each guarded by its own circuit
    breaker. WishartLab (the historic single endpoint) is unreliable; the other two
    are academic mirrors that serve the same dataset.

    Args:
        client: httpx.AsyncClient instance (used for Fiehn + WishartLab; GNPS gets a
                separate inline client because it presents a self-signed certificate
                and requires verify=False).
        inchikey: Standard InChIKey identifier

    Returns:
        Dict with complete ClassyFire response, or None if all mirrors are unavailable.

    Example:
        >>> data = await get_classyfire_classification(client, "REFJWTPEDVJJIY-UHFFFAOYSA-N")
        >>> print(data['kingdom']['name'])  # "Organic compounds"
    """
    endpoints = (
        (_CLASSYFIRE_FIEHN, "classyfire_fiehn", False),
        (_CLASSYFIRE_GNPS, "classyfire_gnps", True),  # self-signed TLS
        (_CLASSYFIRE_WISHART, "classyfire_wishart", False),
    )

    for url_template, label, needs_insecure in endpoints:
        circuit = _get_circuit(label)
        if _is_circuit_open(circuit):
            logger.info("classyfire_circuit_open", endpoint=label, inchikey=inchikey)
            continue

        url = url_template.format(key=inchikey)
        try:
            if needs_insecure:
                async with httpx.AsyncClient(
                    verify=False,
                    timeout=client.timeout,
                    limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                ) as insecure_client:
                    data = await _classyfire_request(insecure_client, url)
            else:
                data = await _classyfire_request(client, url)
        except (httpx.HTTPError, RetryError) as exc:
            _record_failure(circuit)
            metrics.increment("api_calls_failed")
            logger.warning(
                "classyfire_endpoint_failed",
                endpoint=label,
                inchikey=inchikey,
                error=str(exc),
            )
            continue

        if data is _RATE_LIMITED:
            _record_failure(circuit)
            logger.warning("classyfire_rate_limited", endpoint=label, inchikey=inchikey)
            continue

        # Success path: 200 with payload, or 404 (compound not in ClassyFire).
        # 404 is not a circuit failure — it's a definitive "no data for this key".
        _record_success(circuit)
        if isinstance(data, dict):
            logger.info("classyfire_obtained", endpoint=label, inchikey=inchikey)
            return data
        # data is None (404) — definitive negative answer, don't try other mirrors.
        return None

    logger.warning("classyfire_all_endpoints_unavailable", inchikey=inchikey)
    return None


async def get_npclassifier_classification(
    client: httpx.AsyncClient,
    smiles: str,
    max_retries: int = 5,
) -> dict | None:
    """
    Get NPClassifier classification for natural products with retry logic.

    NPClassifier is a deep learning tool for natural product classification.
    API: https://npclassifier.gnps2.org/

    Args:
        client: httpx.AsyncClient instance
        smiles: SMILES string
        max_retries: Maximum retry attempts (via tenacity)

    Returns:
        Dict with NP_Pathway, NP_Superclass, NP_Class, NP_isglycoside, or None

    Example:
        >>> data = await get_npclassifier_classification(client, "c1ccc(cc1)O")
        >>> print(data['NP_Pathway'])  # "Shikimates and Phenylpropanoids"
    """
    circuit = _get_circuit("npclassifier")
    if _is_circuit_open(circuit):
        logger.info("npclassifier_circuit_open")
        return None

    url = "https://npclassifier.gnps2.org/classify"
    smiles_preview = smiles[:50] + "..." if len(smiles) > 50 else smiles

    @_classifier_retry(max_attempts=max_retries)
    async def _do_fetch() -> dict | None:
        _start = time.time()
        response = await client.get(url, params={"smiles": smiles}, timeout=15)
        latency_ms = (time.time() - _start) * 1000

        metrics.increment("api_calls_total")
        metrics.record_latency("npclassifier", latency_ms)

        if response.status_code == 200:
            data = response.json()
            return {
                "NP_Pathway": data.get("pathway_results", [None])[0]
                if data.get("pathway_results")
                else None,
                "NP_Superclass": data.get("superclass_results", [None])[0]
                if data.get("superclass_results")
                else None,
                "NP_Class": data.get("class_results", [None])[0]
                if data.get("class_results")
                else None,
                "NP_isglycoside": data.get("isglycoside", False),
            }

        if response.status_code >= 500:
            metrics.increment("api_calls_failed")
            raise httpx.ReadTimeout(f"NPClassifier {response.status_code}")

        # 4xx: don't retry, don't trip circuit breaker
        logger.warning("npclassifier_client_error", status=response.status_code)
        return None

    try:
        result = await _do_fetch()
        if result is not None:
            _record_success(circuit)
        return result
    except Exception as exc:
        _record_failure(circuit)
        metrics.increment("api_calls_failed")
        logger.warning("npclassifier_failed", smiles=smiles_preview, error=str(exc))
        return None


async def get_complete_classification(
    client: httpx.AsyncClient,
    smiles: str,
    inchikey: str,
) -> dict[str, str]:
    """
    Get complete chemical classification from both ClassyFire and NPClassifier.

    ClassyFire and NPClassifier are fetched in parallel via asyncio.gather.
    If either fails, that section returns empty strings (graceful degradation).

    Args:
        client: httpx.AsyncClient instance
        smiles: SMILES string
        inchikey: InChIKey

    Returns:
        Dict with all classification fields (14 total):
            ClassyFire: Kingdom, Superclass, Class, Subclass, Direct_Parent,
                       Molecular_Framework, Description, ChEMONT_ID_Class, ChEMONT_ID_Subclass
            NPClassifier: NP_Pathway, NP_Superclass, NP_Class, NP_isglycoside
            classification_available: bool

    Example:
        >>> classification = await get_complete_classification(
        ...     client,
        ...     smiles="C1=CC(=C(C=C1O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O",
        ...     inchikey="REFJWTPEDVJJIY-UHFFFAOYSA-N",
        ... )
        >>> print(f"Class: {classification['Class']}")  # "Flavonoids"
    """
    # Initialize with empty fields
    classification: dict[str, str] = {
        # ClassyFire fields
        "Kingdom": "",
        "Superclass": "",
        "Class": "",
        "Subclass": "",
        "Direct_Parent": "",
        "Molecular_Framework": "",
        "Description": "",
        "ChEMONT_ID_Class": "",
        "ChEMONT_ID_Subclass": "",
        # NPClassifier fields
        "NP_Pathway": "",
        "NP_Superclass": "",
        "NP_Class": "",
        "NP_isglycoside": False,
    }

    # Fetch both in parallel (D-34: both non-critical, graceful degradation)
    cf_result, np_result = await asyncio.gather(
        get_classyfire_classification(client, inchikey),
        get_npclassifier_classification(client, smiles),
        return_exceptions=True,
    )

    # Process ClassyFire
    cf_data = cf_result if isinstance(cf_result, dict) else None
    if cf_data:
        cf_fields = extract_classyfire_fields(cf_data)
        classification.update(cf_fields)
        logger.info("classyfire_obtained", class_name=cf_fields.get("Class", "Unknown"))
    else:
        logger.warning("classyfire_unavailable", inchikey=inchikey)

    # Process NPClassifier
    np_data = np_result if isinstance(np_result, dict) else None
    if np_data:
        classification.update(np_data)
        logger.info("npclassifier_obtained", pathway=np_data.get("NP_Pathway", "Unknown"))
    else:
        logger.warning("npclassifier_unavailable")

    classification["classification_available"] = bool(cf_data or np_data)
    return classification


# --------------------------------------------------------------------------- #
# Pure functions (sync, no I/O)
# --------------------------------------------------------------------------- #


def extract_classyfire_fields(cf_data: dict | None) -> dict[str, str]:
    """
    Extract enhanced ClassyFire fields from API response.

    Extracts 9 fields including descriptions and ChEMONT IDs.

    Args:
        cf_data: Full ClassyFire API response

    Returns:
        Dict with Kingdom, Superclass, Class, Subclass, Direct_Parent,
        Molecular_Framework, Description, ChEMONT_ID_Class, ChEMONT_ID_Subclass
    """
    empty = {
        "Kingdom": "",
        "Superclass": "",
        "Class": "",
        "Subclass": "",
        "Direct_Parent": "",
        "Molecular_Framework": "",
        "Description": "",
        "ChEMONT_ID_Class": "",
        "ChEMONT_ID_Subclass": "",
    }

    if cf_data is None:
        return empty

    try:

        def safe_get(d, key1, key2="name", default=""):
            """Safely get nested dict value, handling None at any level."""
            val = d.get(key1) if d else None
            if val is None:
                return default
            if isinstance(val, dict):
                return val.get(key2, default) or default
            return str(val) if val else default

        return {
            "Kingdom": safe_get(cf_data, "kingdom", "name"),
            "Superclass": safe_get(cf_data, "superclass", "name"),
            "Class": safe_get(cf_data, "class", "name"),
            "Subclass": safe_get(cf_data, "subclass", "name"),
            "Direct_Parent": safe_get(cf_data, "direct_parent", "name"),
            "Molecular_Framework": cf_data.get("molecular_framework", "") or "",
            "Description": cf_data.get("description", "") or "",
            "ChEMONT_ID_Class": safe_get(cf_data, "class", "chemont_id"),
            "ChEMONT_ID_Subclass": safe_get(cf_data, "subclass", "chemont_id"),
        }
    except Exception as exc:
        logger.error("classyfire_extract_error", error=str(exc))
        return empty


def classify_compound_type(classification: dict) -> str:
    """
    Determine if compound is natural product, synthetic, or semi-synthetic.

    Uses both ClassyFire and NPClassifier data to infer compound origin.

    Args:
        classification: Classification dict from get_complete_classification()

    Returns:
        str: "Natural Product", "Synthetic", or "Semi-Synthetic"

    Example:
        >>> compound_type = classify_compound_type(classification)
        >>> print(compound_type)  # "Natural Product"
    """
    # Has NPClassifier pathway -> likely natural product
    if classification.get("NP_Pathway"):
        return "Natural Product"

    # Check ClassyFire for natural product indicators
    np_keywords = [
        "alkaloid",
        "terpenoid",
        "flavonoid",
        "polyketide",
        "phenylpropanoid",
        "steroid",
        "glycoside",
        "saponin",
        "tannin",
        "coumarin",
        "quinone",
        "lignan",
    ]

    for field in ["Superclass", "Class", "Subclass", "Direct_Parent"]:
        value = (classification.get(field) or "").lower()
        if any(keyword in value for keyword in np_keywords):
            return "Natural Product"

    return "Synthetic"


def get_classification_summary(classification: dict) -> str:
    """
    Generate human-readable summary of classification.

    Args:
        classification: Classification dict from get_complete_classification()

    Returns:
        str: Multi-line summary text
    """
    lines = []

    lines.append("Chemical Classification Summary")
    lines.append("-" * 50)

    if classification.get("Class"):
        lines.append(
            f"ClassyFire: {classification['Kingdom']} > "
            f"{classification['Superclass']} > "
            f"{classification['Class']} > "
            f"{classification['Subclass']}"
        )
    else:
        lines.append("ClassyFire: No classification available")

    if classification.get("NP_Pathway"):
        lines.append(
            f"NPClassifier: {classification['NP_Pathway']} > "
            f"{classification['NP_Superclass']} > "
            f"{classification['NP_Class']}"
        )
        if classification.get("NP_isglycoside"):
            lines.append("  - Contains glycoside moiety")

    compound_type = classify_compound_type(classification)
    lines.append(f"Compound Type: {compound_type}")

    if classification.get("Molecular_Framework"):
        lines.append(f"Molecular Framework: {classification['Molecular_Framework']}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #


def shutdown_classifier() -> None:
    """Log classifier shutdown. No resources to clean up (httpx client managed externally)."""
    logger.info("chemical_classifier_shutdown")
