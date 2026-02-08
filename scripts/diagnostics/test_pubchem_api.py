"""
PubChem PUG REST API tests for InChIKey -> SMILES resolution.

Validates that PubChem can resolve InChIKeys to SMILES strings,
which is needed for adding InChIKey as a third input type in IMPULATOR.

Usage:
    python scripts/diagnostics/test_pubchem_api.py
    python scripts/diagnostics/test_pubchem_api.py --verbose
    python scripts/diagnostics/test_pubchem_api.py --debug
"""

import json
import sys
import time
import argparse
import requests
from typing import Optional, Dict, List, Tuple

# ── Test compounds ──────────────────────────────────────────────────────────
ASPIRIN_INCHIKEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
QUERCETIN_INCHIKEY = "REFJWTPEDVJJIY-UHFFFAOYSA-N"
CAFFEINE_INCHIKEY = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
IBUPROFEN_INCHIKEY = "HEFNNWSXXWATRW-UHFFFAOYSA-N"

# Syntactically valid but non-existent InChIKey
FAKE_INCHIKEY = "ZZZZZZZZZZZZZA-ZZZZZZZZZZ-Z"

# ── Config ──────────────────────────────────────────────────────────────────
PUBCHEM_PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 30
DEBUG = False


# ── Helpers (same pattern as test_external_apis.py) ─────────────────────────
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


def timed_request(
    method: str, url: str, **kwargs
) -> Tuple[Optional[requests.Response], float, Optional[str]]:
    """Make a request and return (response, latency_ms, error_or_none)."""
    kwargs.setdefault('timeout', TIMEOUT)
    start = time.perf_counter()
    try:
        resp = requests.request(method, url, **kwargs)
        latency = (time.perf_counter() - start) * 1000
        return resp, latency, None
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
        latency = (time.perf_counter() - start) * 1000
        return None, latency, f"Timeout after {latency/1000:.0f}s"
    except requests.exceptions.ConnectionError:
        latency = (time.perf_counter() - start) * 1000
        return None, latency, "Connection refused/failed"
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return None, latency, str(e)


def _extract_smiles(props: dict) -> str:
    """Extract SMILES from a PubChem property dict, trying all known field names.

    PubChem renamed response fields (old URL params still accepted):
      CanonicalSMILES  -> ConnectivitySMILES (response JSON key)
      IsomericSMILES   -> SMILES             (response JSON key)
    """
    for key in ("ConnectivitySMILES", "SMILES", "CanonicalSMILES", "IsomericSMILES"):
        val = props.get(key, "")
        if val:
            return val
    return ""


def _debug_response(label: str, resp: requests.Response):
    """Print raw response when --debug is active."""
    if not DEBUG:
        return
    print(f"\n    [DEBUG] {label}")
    print(f"    Status: {resp.status_code}")
    print(f"    Headers: {dict(resp.headers)}")
    try:
        raw = resp.json()
        print(f"    JSON: {json.dumps(raw, indent=2)[:2000]}")
    except Exception:
        print(f"    Body: {resp.text[:500]}")


# ═══════════════════════════════════════════════════════════════════════════
#  0. Raw Response Diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def check_raw_response() -> TestResult:
    """PubChem: Dump raw response for Aspirin to diagnose field names."""
    t = TestResult("Raw Response Check", "PubChem Diagnostic")
    url = f"{PUBCHEM_PUG_BASE}/compound/inchikey/{ASPIRIN_INCHIKEY}/property/ConnectivitySMILES/JSON"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        _debug_response("Aspirin single lookup", resp)
        try:
            raw = resp.json()
            props = raw.get("PropertyTable", {}).get("Properties", [])
            if props:
                # Show all keys and values from the response
                fields = {k: repr(v) for k, v in props[0].items()}
                smiles = _extract_smiles(props[0])
                if smiles:
                    t.pass_(ms, smiles=smiles, fields=fields)
                else:
                    t.fail(f"No SMILES found. Fields: {fields}", ms)
            else:
                # Check if response has a different structure
                t.fail(f"No Properties in response. Keys: {list(raw.keys())}", ms)
        except Exception as e:
            t.fail(f"Parse error: {e}. Raw: {resp.text[:200]}", ms)
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  1. Single InChIKey -> SMILES Lookups
# ═══════════════════════════════════════════════════════════════════════════

def _check_single(name: str, inchikey: str) -> TestResult:
    """Generic single InChIKey lookup test."""
    t = TestResult(f"{name} Lookup", "PubChem Single")
    url = f"{PUBCHEM_PUG_BASE}/compound/inchikey/{inchikey}/property/ConnectivitySMILES/JSON"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        _debug_response(f"{name} single lookup", resp)
        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            t.fail("No properties returned", ms)
        else:
            smiles = _extract_smiles(props[0])
            cid = props[0].get("CID")
            if not smiles:
                fields = {k: repr(v) for k, v in props[0].items()}
                t.fail(f"Empty SMILES. Fields: {fields}", ms)
            else:
                t.pass_(ms, smiles=smiles, cid=cid)
    return t


def check_single_aspirin() -> TestResult:
    return _check_single("Aspirin", ASPIRIN_INCHIKEY)

def check_single_quercetin() -> TestResult:
    return _check_single("Quercetin", QUERCETIN_INCHIKEY)

def check_single_caffeine() -> TestResult:
    return _check_single("Caffeine", CAFFEINE_INCHIKEY)


# ═══════════════════════════════════════════════════════════════════════════
#  2. Extended Property Lookup
# ═══════════════════════════════════════════════════════════════════════════

def check_extended_properties() -> TestResult:
    """PubChem: Fetch multiple properties (SMILES, formula, InChIKey) for Aspirin."""
    t = TestResult("Extended Properties", "PubChem Properties")
    url = (
        f"{PUBCHEM_PUG_BASE}/compound/inchikey/{ASPIRIN_INCHIKEY}"
        "/property/ConnectivitySMILES,SMILES,InChIKey,MolecularFormula/JSON"
    )
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        _debug_response("Extended properties", resp)
        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            t.fail("No properties returned", ms)
        else:
            p = props[0]
            smiles = _extract_smiles(p)
            if not smiles:
                fields = {k: repr(v) for k, v in p.items()}
                t.fail(f"No SMILES in extended response. Fields: {fields}", ms)
            else:
                t.pass_(
                    ms,
                    connectivity_smiles=p.get("ConnectivitySMILES", ""),
                    smiles=p.get("SMILES", ""),
                    inchikey=p.get("InChIKey", ""),
                    formula=p.get("MolecularFormula", ""),
                    cid=p.get("CID"),
                    inchikey_matches=(p.get("InChIKey") == ASPIRIN_INCHIKEY),
                )
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  3. Batch InChIKey Lookup (POST)
# ═══════════════════════════════════════════════════════════════════════════

def check_batch_lookup() -> TestResult:
    """PubChem: Batch resolve 4 InChIKeys via POST."""
    t = TestResult("Batch Lookup (4)", "PubChem Batch")
    url = f"{PUBCHEM_PUG_BASE}/compound/inchikey/property/ConnectivitySMILES,InChIKey/JSON"
    keys = [ASPIRIN_INCHIKEY, QUERCETIN_INCHIKEY, CAFFEINE_INCHIKEY, IBUPROFEN_INCHIKEY]
    body = {"inchikey": ",".join(keys)}

    resp, ms, err = timed_request("POST", url, data=body)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code != 200:
        t.fail(f"HTTP {resp.status_code}", ms)
    else:
        _debug_response("Batch lookup", resp)
        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            t.fail("No properties returned", ms)
        else:
            # Build mapping from response
            resolved = {}
            for p in props:
                ik = p.get("InChIKey", "")
                smi = _extract_smiles(p)
                if ik and smi:
                    resolved[ik] = smi

            resolved_count = sum(1 for k in keys if k in resolved)

            if resolved_count == 0:
                # Dump first entry for diagnostics
                sample_fields = {k: repr(v) for k, v in props[0].items()} if props else {}
                t.fail(
                    f"0/{len(keys)} resolved. Got {len(props)} entries. "
                    f"First entry fields: {sample_fields}",
                    ms,
                )
            elif resolved_count < len(keys):
                missing = [k for k in keys if k not in resolved]
                t.pass_(
                    ms,
                    requested=len(keys),
                    resolved=resolved_count,
                    all_resolved=False,
                    missing=missing,
                    sample=list(resolved.items())[:2],
                )
            else:
                t.pass_(
                    ms,
                    requested=len(keys),
                    resolved=resolved_count,
                    all_resolved=True,
                    sample=list(resolved.items())[:2],
                )
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  4. Invalid InChIKey Handling
# ═══════════════════════════════════════════════════════════════════════════

def check_invalid_inchikey() -> TestResult:
    """PubChem: Verify graceful failure for non-existent InChIKey."""
    t = TestResult("Invalid InChIKey", "PubChem Error")
    url = f"{PUBCHEM_PUG_BASE}/compound/inchikey/{FAKE_INCHIKEY}/property/ConnectivitySMILES/JSON"
    resp, ms, err = timed_request("GET", url)
    if resp is None:
        t.fail(err, ms)
    elif resp.status_code == 404:
        t.pass_(ms, status=404, behavior="404 Not Found (expected)")
    elif resp.status_code == 400:
        t.pass_(ms, status=400, behavior="400 Bad Request (acceptable)")
    elif resp.status_code == 200:
        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            t.pass_(ms, status=200, behavior="200 with empty results (acceptable)")
        else:
            t.fail(f"Fake InChIKey unexpectedly resolved to {props}", ms)
    else:
        t.pass_(ms, status=resp.status_code, behavior=f"HTTP {resp.status_code}")
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  5. Rate Limit Test
# ═══════════════════════════════════════════════════════════════════════════

def check_rapid_requests() -> TestResult:
    """PubChem: 5 rapid sequential requests (rate limit check).

    PubChem limits: 5 req/s, 400 req/min, 300s server time/min.
    Throttle header: X-Throttling-Control (Green/Yellow/Red/Black).
    """
    t = TestResult("Rapid Requests (5)", "PubChem Rate")
    url = f"{PUBCHEM_PUG_BASE}/compound/inchikey/{ASPIRIN_INCHIKEY}/property/ConnectivitySMILES/JSON"

    successes = 0
    total_ms = 0
    errors = []
    last_throttle = ""

    for i in range(5):
        resp, ms, err = timed_request("GET", url)
        total_ms += ms
        if resp is not None:
            last_throttle = resp.headers.get("X-Throttling-Control", "")
            if resp.status_code == 200:
                successes += 1
            elif resp.status_code == 503:
                errors.append(f"Request {i+1}: 503 throttled")
            else:
                errors.append(f"Request {i+1}: HTTP {resp.status_code}")
        elif err:
            errors.append(f"Request {i+1}: {err}")

    if successes == 5:
        t.pass_(total_ms, successes=5, avg_ms=round(total_ms / 5, 1),
                throttle_status=last_throttle or "N/A")
    elif successes >= 3:
        t.pass_(total_ms, successes=successes, note=f"{5-successes} throttled",
                throttle_status=last_throttle or "N/A", errors=errors)
    else:
        t.fail(f"Only {successes}/5 succeeded: {'; '.join(errors)}", total_ms)
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  6. HTTPS/TLS Verification
# ═══════════════════════════════════════════════════════════════════════════

def check_https_tls() -> TestResult:
    """TLS: Verify HTTPS certificate for PubChem."""
    t = TestResult("TLS Certificate", "PubChem HTTPS")
    try:
        start = time.perf_counter()
        resp = requests.head(f"{PUBCHEM_PUG_BASE}/compound/cid/2244/JSON", timeout=10, verify=True)
        ms = (time.perf_counter() - start) * 1000
        t.pass_(ms, status=resp.status_code, tls_verified=True)
    except requests.exceptions.SSLError as e:
        t.fail(f"SSL verification failed: {e}")
    except Exception as e:
        t.fail(str(e))
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

ALL_TESTS = {
    "diagnostic": [
        check_raw_response,
    ],
    "single": [
        check_single_aspirin,
        check_single_quercetin,
        check_single_caffeine,
    ],
    "properties": [
        check_extended_properties,
    ],
    "batch": [
        check_batch_lookup,
    ],
    "error": [
        check_invalid_inchikey,
    ],
    "rate": [
        check_rapid_requests,
    ],
    "https": [
        check_https_tls,
    ],
}


def run_tests(api_filter: Optional[str] = None, verbose: bool = False) -> bool:
    """Run all tests and print results. Returns True if all pass."""
    print("=" * 72)
    print("  PubChem PUG REST API Test Suite")
    print("  InChIKey -> SMILES Resolution")
    print("=" * 72)

    results: List[TestResult] = []

    for group, tests in ALL_TESTS.items():
        if api_filter and group != api_filter:
            continue

        print(f"\n-- {group.upper()} {'─' * (60 - len(group))}")

        for test_fn in tests:
            result = test_fn()
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            icon = "+" if result.passed else "x"
            latency = f"{result.latency_ms:>7.0f}ms" if result.latency_ms else "       "

            print(f"  [{icon}] {status}  {latency}  {result.api}: {result.name}")

            if not result.passed:
                print(f"         Error: {result.error}")

            if verbose and result.details:
                for key, val in result.details.items():
                    val_str = str(val)
                    if len(val_str) > 120:
                        val_str = val_str[:117] + "..."
                    print(f"         {key}: {val_str}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    all_passed = passed == total

    print(f"\n{'=' * 72}")
    print(f"  {passed}/{total} PASSED")
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        failed_names = [r.name for r in results if not r.passed]
        print(f"  FAILED: {', '.join(failed_names)}")
    print(f"{'=' * 72}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="PubChem API Test Suite")
    parser.add_argument(
        "--api", choices=list(ALL_TESTS.keys()),
        help="Only run tests for a specific group"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed response data for passing tests"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Dump raw HTTP responses for debugging"
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Request timeout in seconds (default: 30)"
    )
    args = parser.parse_args()

    global TIMEOUT, DEBUG
    TIMEOUT = args.timeout
    DEBUG = args.debug

    success = run_tests(api_filter=args.api, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
