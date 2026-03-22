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
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.metrics import metrics

logger = structlog.get_logger()

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


# Pre-create circuits with higher threshold for non-critical classifiers (D-34)
_circuits["classyfire"] = _make_circuit(threshold=5, cooldown=300)
_circuits["npclassifier"] = _make_circuit(threshold=5, cooldown=300)


# --------------------------------------------------------------------------- #
# Retry decorator
# --------------------------------------------------------------------------- #

_RETRYABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)


def _classifier_retry(max_attempts: int = 5):
    """Tenacity retry: 5x exponential backoff (D-29)."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )


# --------------------------------------------------------------------------- #
# Async I/O functions
# --------------------------------------------------------------------------- #


async def get_classyfire_classification(
    client: httpx.AsyncClient,
    inchikey: str,
    max_retries: int = 5,
) -> dict | None:
    """
    Get comprehensive ClassyFire classification with retry logic.

    ClassyFire provides chemical taxonomy based on structural features.
    API: http://classyfire.wishartlab.com/

    Args:
        client: httpx.AsyncClient instance
        inchikey: Standard InChIKey identifier
        max_retries: Maximum retry attempts (via tenacity)

    Returns:
        Dict with complete ClassyFire response, or None if failed

    Example:
        >>> data = await get_classyfire_classification(client, "REFJWTPEDVJJIY-UHFFFAOYSA-N")
        >>> print(data['kingdom']['name'])  # "Organic compounds"
    """
    circuit = _get_circuit("classyfire")
    if _is_circuit_open(circuit):
        logger.info("classyfire_circuit_open", inchikey=inchikey)
        return None

    url = f"http://classyfire.wishartlab.com/entities/{inchikey}.json"

    @_classifier_retry(max_attempts=max_retries)
    async def _do_fetch() -> dict | None:
        _start = time.time()
        response = await client.get(url, timeout=30)
        latency_ms = (time.time() - _start) * 1000

        metrics.increment("api_calls_total")
        metrics.record_latency("classyfire", latency_ms)

        if response.status_code == 200:
            return response.json()

        if response.status_code >= 500:
            metrics.increment("api_calls_failed")
            # 5xx trips circuit breaker via exception -> retry
            raise httpx.ReadTimeout(f"ClassyFire {response.status_code}")

        # 4xx: don't retry, don't trip circuit breaker
        logger.warning("classyfire_client_error", status=response.status_code, inchikey=inchikey)
        return None

    try:
        result = await _do_fetch()
        if result is not None:
            _record_success(circuit)
        return result
    except Exception as exc:
        _record_failure(circuit)
        metrics.increment("api_calls_failed")
        logger.warning("classyfire_failed", inchikey=inchikey, error=str(exc))
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


def classify_compound_type(classification: Dict) -> str:
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


def get_classification_summary(classification: Dict) -> str:
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
