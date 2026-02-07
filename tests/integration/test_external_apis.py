"""
External API connectivity and response validation tests.

Tests all external APIs used by IMPULATOR with the current config:
  1. ChEMBL REST API     - similarity search, molecule data, activities, targets
  2. ChEMBL Python client - chembl_webresource_client
  3. RCSB PDB Search API  - chemical similarity search
  4. RCSB PDB Data API    - structure metadata (REST + GraphQL)
  5. ClassyFire           - chemical classification
  6. NPClassifier         - natural product classification

Usage:
    python tests/integration/test_external_apis.py
    python tests/integration/test_external_apis.py --verbose
    python tests/integration/test_external_apis.py --api chembl
"""

import sys
import time
import json
import argparse
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ── Test compounds ──────────────────────────────────────────────────────────
# Aspirin: well-characterized, guaranteed to exist in all databases
ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
ASPIRIN_INCHIKEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
ASPIRIN_CHEMBL_ID = "CHEMBL25"

# Quercetin: natural product (good for NPClassifier)
QUERCETIN_SMILES = "O=C1C(O)=C(O)C2=CC(O)=C(O)C=C2OC1=C3C=CC(O)=CC3"
QUERCETIN_INCHIKEY = "REFJWTPEDVJJIY-UHFFFAOYSA-N"

# Known PDB structures with aspirin-like ligands
KNOWN_PDB_ID = "4PH9"

# ── Config ──────────────────────────────────────────────────────────────────
CHEMBL_REST_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DATA_URL = "https://data.rcsb.org/rest/v1/core"
PDB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
CLASSYFIRE_URL = "http://classyfire.wishartlab.com/entities"  # HTTP only - no TLS support
NPCLASSIFIER_URL = "https://npclassifier.ucsd.edu/classify"

# Timeouts per API (ChEMBL similarity is notoriously slow)
TIMEOUT = 30                  # default
TIMEOUT_CHEMBL_SIMILARITY = 90  # similarity search can take 60-90s
TIMEOUT_PDB_SEARCH = 45        # chemical search can be slow


# ── Helpers ─────────────────────────────────────────────────────────────────
class TestResult:
    def __init__(self, name: str, api: str):
        self.name = name
        self.api = api
        self.passed = False
        self.error: Optional[str] = None
        self.latency_ms: float = 0
        self.details: Dict = {}

    def pass_(self, latency_ms: float, **details):
        self.passed = True
        self.latency_ms = latency_ms
        self.details = details

    def fail(self, error: str, latency_ms: float = 0):
        self.passed = False
        self.error = error
        self.latency_ms = latency_ms


def timed_request(method: str, url: str, **kwargs) -> Tuple[Optional[requests.Response], float, Optional[str]]:
    """Make a request and return (response, latency_ms, error_type).

    error_type is None on success, 'timeout' on timeout, 'connection' on connection error,
    or the exception class name for other errors.
    """
    kwargs.setdefault('timeout', TIMEOUT)
    start = time.perf_counter()
    try:
        resp = requests.request(method, url, **kwargs)
        latency = (time.perf_counter() - start) * 1000
        return resp, latency, None
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
        latency = (time.perf_counter() - start) * 1000
        return None, latency, f"Timeout after {latency/1000:.0f}s"
    except requests.exceptions.ConnectionError as e:
        latency = (time.perf_counter() - start) * 1000
        return None, latency, f"Connection refused/failed"
    except requests.exceptions.SSLError as e:
        latency = (time.perf_counter() - start) * 1000
        return None, latency, f"SSL error: {e}"
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return None, latency, str(e)


# ═══════════════════════════════════════════════════════════════════════════
#  1. ChEMBL REST API
# ═══════════════════════════════════════════════════════════════════════════

def test_chembl_rest_similarity() -> TestResult:
    """ChEMBL REST: Similarity search for Aspirin."""
    t = TestResult("Similarity Search", "ChEMBL REST")
    url = f"{CHEMBL_REST_BASE}/similarity/{ASPIRIN_SMILES}/90.json"
    resp, ms, err = timed_request("GET", url, timeout=TIMEOUT_CHEMBL_SIMILARITY)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        molecules = data.get("molecules", [])
        t.pass_(ms, count=len(molecules), has_results=len(molecules) > 0)
    return t


def test_chembl_rest_molecule() -> TestResult:
    """ChEMBL REST: Fetch molecule data for Aspirin."""
    t = TestResult("Molecule Data", "ChEMBL REST")
    url = f"{CHEMBL_REST_BASE}/molecule/{ASPIRIN_CHEMBL_ID}.json"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        name = data.get("pref_name", "")
        t.pass_(ms, pref_name=name, has_smiles=bool(data.get("molecule_structures")))
    return t


def test_chembl_rest_activities() -> TestResult:
    """ChEMBL REST: Fetch activities for Aspirin."""
    t = TestResult("Activities", "ChEMBL REST")
    url = f"{CHEMBL_REST_BASE}/activity.json?molecule_chembl_id={ASPIRIN_CHEMBL_ID}&limit=5"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        activities = data.get("activities", [])
        t.pass_(ms, count=len(activities), has_results=len(activities) > 0)
    return t


def test_chembl_rest_target() -> TestResult:
    """ChEMBL REST: Fetch target info (COX-2)."""
    t = TestResult("Target Info", "ChEMBL REST")
    url = f"{CHEMBL_REST_BASE}/target/CHEMBL230.json"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        name = data.get("pref_name", "")
        t.pass_(ms, target_name=name)
    return t


def test_chembl_rest_drug_indication() -> TestResult:
    """ChEMBL REST: Fetch drug indications for Aspirin."""
    t = TestResult("Drug Indications", "ChEMBL REST")
    url = f"{CHEMBL_REST_BASE}/drug_indication.json?molecule_chembl_id={ASPIRIN_CHEMBL_ID}&limit=5"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        indications = data.get("drug_indications", [])
        t.pass_(ms, count=len(indications), has_results=len(indications) > 0)
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  2. ChEMBL Python Client
# ═══════════════════════════════════════════════════════════════════════════

def test_chembl_client_similarity() -> TestResult:
    """ChEMBL Client: Similarity search via chembl_webresource_client."""
    t = TestResult("Client Similarity", "ChEMBL Client")
    try:
        start = time.perf_counter()
        from chembl_webresource_client.new_client import new_client
        results = new_client.similarity.filter(
            smiles=ASPIRIN_SMILES, similarity=90
        ).only(['molecule_chembl_id', 'pref_name'])
        result_list = list(results[:5])
        ms = (time.perf_counter() - start) * 1000
        t.pass_(ms, count=len(result_list), has_results=len(result_list) > 0)
    except ImportError:
        t.fail("chembl_webresource_client not installed")
    except Exception as e:
        t.fail(str(e))
    return t


def test_chembl_client_molecule() -> TestResult:
    """ChEMBL Client: Molecule lookup via chembl_webresource_client."""
    t = TestResult("Client Molecule", "ChEMBL Client")
    try:
        start = time.perf_counter()
        from chembl_webresource_client.new_client import new_client
        mol = new_client.molecule.get(ASPIRIN_CHEMBL_ID)
        ms = (time.perf_counter() - start) * 1000
        if mol:
            t.pass_(ms, pref_name=mol.get('pref_name', ''))
        else:
            t.fail("No molecule returned", ms)
    except ImportError:
        t.fail("chembl_webresource_client not installed")
    except Exception as e:
        t.fail(str(e))
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  3. RCSB PDB Search API
# ═══════════════════════════════════════════════════════════════════════════

def test_pdb_search_chemical() -> TestResult:
    """PDB Search: Chemical similarity search for Aspirin."""
    t = TestResult("Chemical Search", "PDB Search")
    query = {
        "query": {
            "type": "terminal",
            "service": "chemical",
            "parameters": {
                "value": ASPIRIN_SMILES,
                "type": "descriptor",
                "descriptor_type": "SMILES",
                "match_type": "graph-relaxed-stereo",
            }
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"], "return_all_hits": False, "paginate": {"start": 0, "rows": 5}},
    }
    resp, ms, err = timed_request("POST", PDB_SEARCH_URL, json=query, timeout=TIMEOUT_PDB_SEARCH)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code == 204:
        t.pass_(ms, count=0, note="No results (204)")
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}: {resp.text[:200]}", ms)
    else:
        data = resp.json()
        total = data.get("total_count", 0)
        ids = [r.get("identifier", "") for r in data.get("result_set", [])[:5]]
        t.pass_(ms, total=total, sample_ids=ids)
    return t


def test_pdb_data_entry() -> TestResult:
    """PDB Data: Fetch entry metadata for a known structure."""
    t = TestResult("Entry Metadata", "PDB Data")
    url = f"{PDB_DATA_URL}/entry/{KNOWN_PDB_ID}"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        title = data.get("struct", {}).get("title", "")[:80]
        t.pass_(ms, title=title)
    return t


def test_pdb_graphql_resolution() -> TestResult:
    """PDB GraphQL: Batch resolution fetch."""
    t = TestResult("GraphQL Resolution", "PDB GraphQL")
    query = """
    query($ids: [String!]!) {
        entries(entry_ids: $ids) {
            rcsb_id
            rcsb_entry_info {
                resolution_combined
            }
        }
    }
    """
    resp, ms, err = timed_request(
        "POST", PDB_GRAPHQL_URL,
        json={"query": query, "variables": {"ids": [KNOWN_PDB_ID]}},
        headers={"Content-Type": "application/json"},
    )
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        entries = data.get("data", {}).get("entries", [])
        if entries:
            res = entries[0].get("rcsb_entry_info", {}).get("resolution_combined", [])
            t.pass_(ms, pdb_id=KNOWN_PDB_ID, resolution=res[0] if res else None)
        else:
            t.fail("No entries returned", ms)
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  4. ClassyFire
# ═══════════════════════════════════════════════════════════════════════════

def test_classyfire() -> TestResult:
    """ClassyFire: Chemical classification for Aspirin."""
    t = TestResult("Classification", "ClassyFire")
    url = f"{CLASSYFIRE_URL}/{ASPIRIN_INCHIKEY}.json"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code == 404:
        t.fail("Compound not found (404) - may need to submit first", ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        kingdom = data.get("kingdom", {}).get("name", "") if data.get("kingdom") else ""
        superclass = data.get("superclass", {}).get("name", "") if data.get("superclass") else ""
        t.pass_(ms, kingdom=kingdom, superclass=superclass)
    return t


def test_classyfire_quercetin() -> TestResult:
    """ClassyFire: Classification for Quercetin (natural product)."""
    t = TestResult("Quercetin Classification", "ClassyFire")
    url = f"{CLASSYFIRE_URL}/{QUERCETIN_INCHIKEY}.json"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code == 404:
        t.fail("Compound not found (404)", ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        cls = data.get("class", {}).get("name", "") if data.get("class") else ""
        t.pass_(ms, class_name=cls)
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  5. NPClassifier
# ═══════════════════════════════════════════════════════════════════════════

def test_npclassifier() -> TestResult:
    """NPClassifier: Natural product classification for Quercetin."""
    t = TestResult("NP Classification", "NPClassifier")
    resp, ms, err = timed_request("GET", NPCLASSIFIER_URL, params={"smiles": QUERCETIN_SMILES})
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        pathway = data.get("pathway_results", [])
        superclass = data.get("superclass_results", [])
        t.pass_(
            ms,
            pathways=pathway[:3] if pathway else [],
            superclasses=superclass[:3] if superclass else [],
        )
    return t


def test_npclassifier_aspirin() -> TestResult:
    """NPClassifier: Classification for Aspirin (synthetic drug)."""
    t = TestResult("Aspirin NP Class", "NPClassifier")
    resp, ms, err = timed_request("GET", NPCLASSIFIER_URL, params={"smiles": ASPIRIN_SMILES})
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        data = resp.json()
        pathway = data.get("pathway_results", [])
        t.pass_(ms, pathways=pathway[:3] if pathway else [])
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  6. HTTPS Verification
# ═══════════════════════════════════════════════════════════════════════════

def test_https_tls_chembl() -> TestResult:
    """TLS: Verify HTTPS certificate for ChEMBL."""
    t = TestResult("TLS Certificate", "ChEMBL HTTPS")
    try:
        start = time.perf_counter()
        resp = requests.head(f"{CHEMBL_REST_BASE}/status.json", timeout=10, verify=True)
        ms = (time.perf_counter() - start) * 1000
        t.pass_(ms, status=resp.status_code, tls_verified=True)
    except requests.exceptions.SSLError as e:
        t.fail(f"SSL verification failed: {e}")
    except Exception as e:
        t.fail(str(e))
    return t


def test_https_tls_pdb() -> TestResult:
    """TLS: Verify HTTPS certificate for RCSB PDB."""
    t = TestResult("TLS Certificate", "PDB HTTPS")
    try:
        start = time.perf_counter()
        resp = requests.head("https://data.rcsb.org", timeout=10, verify=True)
        ms = (time.perf_counter() - start) * 1000
        t.pass_(ms, status=resp.status_code, tls_verified=True)
    except requests.exceptions.SSLError as e:
        t.fail(f"SSL verification failed: {e}")
    except Exception as e:
        t.fail(str(e))
    return t


def test_http_classyfire() -> TestResult:
    """HTTP: Verify ClassyFire is reachable over HTTP (no TLS support)."""
    t = TestResult("HTTP Reachability", "ClassyFire HTTP")
    try:
        start = time.perf_counter()
        resp = requests.head("http://classyfire.wishartlab.com", timeout=10)
        ms = (time.perf_counter() - start) * 1000
        t.pass_(ms, status=resp.status_code)
    except Exception as e:
        t.fail(str(e))
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

ALL_TESTS = {
    "chembl": [
        test_chembl_rest_similarity,
        test_chembl_rest_molecule,
        test_chembl_rest_activities,
        test_chembl_rest_target,
        test_chembl_rest_drug_indication,
        test_chembl_client_similarity,
        test_chembl_client_molecule,
    ],
    "pdb": [
        test_pdb_search_chemical,
        test_pdb_data_entry,
        test_pdb_graphql_resolution,
    ],
    "classyfire": [
        test_classyfire,
        test_classyfire_quercetin,
    ],
    "npclassifier": [
        test_npclassifier,
        test_npclassifier_aspirin,
    ],
    "https": [
        test_https_tls_chembl,
        test_https_tls_pdb,
        test_http_classyfire,
    ],
}


def run_tests(api_filter: Optional[str] = None, verbose: bool = False) -> bool:
    """Run all tests and print results. Returns True if all pass."""
    print("=" * 72)
    print("  IMPULATOR External API Test Suite")
    print(f"  Timeouts: default={TIMEOUT}s, ChEMBL similarity={TIMEOUT_CHEMBL_SIMILARITY}s, PDB search={TIMEOUT_PDB_SEARCH}s")
    print("=" * 72)

    results: List[TestResult] = []
    api_groups = {api_filter: ALL_TESTS[api_filter]} if api_filter else ALL_TESTS

    for api_group, tests in api_groups.items():
        print(f"\n--- {api_group.upper()} ---")
        for test_fn in tests:
            try:
                result = test_fn()
            except Exception as e:
                result = TestResult(test_fn.__doc__ or test_fn.__name__, api_group)
                result.fail(f"Unhandled: {e}")

            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            symbol = "+" if result.passed else "x"
            latency = f"{result.latency_ms:7.0f}ms" if result.latency_ms else "      -"

            print(f"  [{symbol}] {status}  {latency}  {result.api}: {result.name}")

            if result.passed and verbose and result.details:
                for k, v in result.details.items():
                    print(f"                            {k}: {v}")

            if not result.passed and result.error:
                print(f"                            Error: {result.error}")

        # Small delay between API groups to be polite
        time.sleep(0.5)

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    total_latency = sum(r.latency_ms for r in results)

    print("\n" + "=" * 72)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"  Total latency: {total_latency:.0f}ms ({total_latency/1000:.1f}s)")

    if failed:
        print("\n  Failed tests:")
        for r in results:
            if not r.passed:
                print(f"    - {r.api}: {r.name} -> {r.error}")

    # Per-API summary
    print("\n  Per-API status:")
    result_idx = 0
    for api_group, tests in api_groups.items():
        group_results = results[result_idx:result_idx + len(tests)]
        result_idx += len(tests)
        group_passed = sum(1 for r in group_results if r.passed)
        group_total = len(group_results)
        avg_latency = sum(r.latency_ms for r in group_results) / max(group_total, 1)
        status = "OK" if group_passed == group_total else "DEGRADED"
        print(f"    {api_group:15s}  {status:8s}  {group_passed}/{group_total} tests  avg {avg_latency:.0f}ms")

    print("=" * 72)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Test all external APIs used by IMPULATOR"
    )
    parser.add_argument(
        "--api", choices=list(ALL_TESTS.keys()),
        help="Test only a specific API group"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed response data for passing tests"
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Request timeout in seconds (default: 30)"
    )
    args = parser.parse_args()

    global TIMEOUT
    TIMEOUT = args.timeout

    success = run_tests(api_filter=args.api, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
