"""
PubChem PUG REST API Endpoint Exploration & Benchmark.

Tests PubChem endpoints for similarity search, compound properties,
and bioactivity data. Evaluates PubChem as a complementary data source
alongside ChEMBL in IMPULATOR.

Endpoints tested:
  1. Similarity search (fastsimilarity_2d)
  2. Compound properties (batch)
  3. Bioactivity data (assaysummary)
  4. Rate limit monitoring (X-Throttling-Control)

Usage:
    python scripts/test_pubchem_endpoints.py
    python scripts/test_pubchem_endpoints.py --compound aspirin --verbose
    python scripts/test_pubchem_endpoints.py --benchmark
    python scripts/test_pubchem_endpoints.py --json
"""
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_SDQ = "https://pubchem.ncbi.nlm.nih.gov/sdq/sphinxql.cgi"

SIMILARITY_TIMEOUT = 60
PROPERTY_TIMEOUT = 30
BIOACTIVITY_TIMEOUT = 60
GENERAL_TIMEOUT = 30

# PubChem allows 5 req/s; use 4 req/s for safety
RATE_LIMIT_INTERVAL = 0.25

# App default activity types (matches ChEMBL comparison script)
APP_ACTIVITY_TYPES = {'IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'GI50', 'MIC'}

# All properties we care about
STANDARD_PROPERTIES = (
    "MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,"
    "RotatableBondCount,HeavyAtomCount,Complexity,"
    "ConnectivitySMILES,SMILES,InChIKey,IUPACName"
)

# Test compounds with known PubChem CIDs
COMPOUNDS = {
    "aspirin":    {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",                    "name": "Aspirin",    "cid": 2244},
    "caffeine":   {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",                 "name": "Caffeine",   "cid": 2519},
    "kaempferol": {"smiles": "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",    "name": "Kaempferol", "cid": 5280863},
    "quercetin":  {"smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "name": "Quercetin",  "cid": 5280343},
    "ibuprofen":  {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",              "name": "Ibuprofen",  "cid": 3672},
    "ethanol":    {"smiles": "CCO",                                          "name": "Ethanol",    "cid": 702},
}


# ═══════════════════════════════════════════════════════════════════════════
#  HTTP Session & Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════

def create_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = create_session()
_last_pubchem_time = 0.0


def _rate_limit():
    """Enforce PubChem rate limiting (0.25s = 4 req/s)."""
    global _last_pubchem_time
    elapsed = time.perf_counter() - _last_pubchem_time
    if elapsed < RATE_LIMIT_INTERVAL:
        time.sleep(RATE_LIMIT_INTERVAL - elapsed)
    _last_pubchem_time = time.perf_counter()


def pubchem_get(url: str, params: dict = None, timeout: int = GENERAL_TIMEOUT
                ) -> Tuple[Optional[requests.Response], Dict[str, str]]:
    """Rate-limited GET. Returns (response, throttle_info)."""
    _rate_limit()
    throttle_info = {}
    try:
        resp = SESSION.get(url, params=params, timeout=timeout)
        throttle = resp.headers.get("X-Throttling-Control", "")
        if throttle:
            throttle_info = _parse_throttle_header(throttle)
        resp.raise_for_status()
        return resp, throttle_info
    except requests.exceptions.RequestException as e:
        print(f"    [ERROR] {e}")
        return None, throttle_info


def pubchem_post(url: str, data: dict = None, params: dict = None,
                 timeout: int = GENERAL_TIMEOUT
                 ) -> Tuple[Optional[requests.Response], Dict[str, str]]:
    """Rate-limited POST. Returns (response, throttle_info)."""
    _rate_limit()
    throttle_info = {}
    try:
        resp = SESSION.post(url, data=data, params=params, timeout=timeout)
        throttle = resp.headers.get("X-Throttling-Control", "")
        if throttle:
            throttle_info = _parse_throttle_header(throttle)
        resp.raise_for_status()
        return resp, throttle_info
    except requests.exceptions.RequestException as e:
        print(f"    [ERROR] {e}")
        return None, throttle_info


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_throttle_header(header: str) -> Dict[str, str]:
    """Parse X-Throttling-Control header.

    Format: "Request Count status: Green (12%), Request Time status: Green (0%),
             Service status: Green (0%)"
    """
    result = {}
    for part in header.split(","):
        part = part.strip()
        if "Request Count" in part:
            result["request_count"] = part.split(":", 1)[-1].strip()
        elif "Request Time" in part:
            result["request_time"] = part.split(":", 1)[-1].strip()
        elif "Service" in part:
            result["service"] = part.split(":", 1)[-1].strip()
    return result


def _extract_smiles(props: dict) -> str:
    """Extract SMILES from PubChem property dict, handling renamed fields.

    PubChem renamed response fields (old URL params still accepted):
      CanonicalSMILES  -> ConnectivitySMILES
      IsomericSMILES   -> SMILES
    """
    for key in ("ConnectivitySMILES", "SMILES", "CanonicalSMILES", "IsomericSMILES"):
        val = props.get(key, "")
        if val:
            return val
    return ""


def print_header(text: str):
    width = 80
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimilarityResult:
    cids: List[int] = field(default_factory=list)
    time_ms: float = 0.0
    threshold: int = 90
    max_records: int = 100
    error: Optional[str] = None


@dataclass
class PropertyResult:
    properties: List[Dict] = field(default_factory=list)
    time_ms: float = 0.0
    batch_size: int = 0
    error: Optional[str] = None


@dataclass
class BioactivityResult:
    # Unfiltered totals
    total_assays: int = 0
    active_count: int = 0
    inactive_count: int = 0
    inconclusive_count: int = 0
    # All activity names with counts
    by_activity_name: Dict[str, int] = field(default_factory=dict)
    # Filtered by app activity types (IC50, Ki, Kd, EC50, AC50, GI50, MIC)
    type_filtered_total: int = 0
    type_filtered_active: int = 0
    by_type_filtered: Dict[str, int] = field(default_factory=dict)
    by_type_filtered_active: Dict[str, int] = field(default_factory=dict)
    # Targets
    unique_targets: int = 0
    target_names: List[str] = field(default_factory=list)
    # Meta
    response_size_bytes: int = 0
    time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ThrottleSnapshot:
    timestamp: float = 0.0
    request_count_status: str = ""
    request_time_status: str = ""
    service_status: str = ""


@dataclass
class DrugIndicationResult:
    total: int = 0
    by_phase: Dict[str, int] = field(default_factory=dict)
    indications: List[Dict[str, str]] = field(default_factory=list)
    time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class CompoundBenchmark:
    name: str = ""
    smiles: str = ""
    cid: int = 0
    similarity: SimilarityResult = field(default_factory=SimilarityResult)
    properties: PropertyResult = field(default_factory=PropertyResult)
    bioactivity: BioactivityResult = field(default_factory=BioactivityResult)
    indications: DrugIndicationResult = field(default_factory=DrugIndicationResult)


# ═══════════════════════════════════════════════════════════════════════════
#  API Functions
# ═══════════════════════════════════════════════════════════════════════════

def pubchem_similarity_search(
    smiles: str, threshold: int = 90, max_records: int = 100
) -> SimilarityResult:
    """PubChem 2D similarity search via fastsimilarity_2d.

    Uses POST (handles complex SMILES). Returns CIDs only (no Tanimoto scores).
    PubChem uses 881-bit substructure fingerprint, NOT Morgan/ECFP like ChEMBL.
    """
    result = SimilarityResult(threshold=threshold, max_records=max_records)
    start = time.perf_counter()

    url = f"{PUBCHEM_BASE}/compound/fastsimilarity_2d/smiles/cids/JSON"
    resp, _ = pubchem_post(
        url,
        data={"smiles": smiles},
        params={"Threshold": threshold, "MaxRecords": max_records},
        timeout=SIMILARITY_TIMEOUT,
    )
    result.time_ms = (time.perf_counter() - start) * 1000

    if resp is None:
        result.error = "Request failed"
        return result

    try:
        cids = resp.json().get("IdentifierList", {}).get("CID", [])
        result.cids = cids
    except Exception as e:
        result.error = f"Parse error: {e}"

    return result


def pubchem_fetch_properties(
    cids: List[int],
    properties: str = STANDARD_PROPERTIES,
    batch_size: int = 100,
) -> PropertyResult:
    """Fetch compound properties for CIDs via batch POST.

    Recommended batch: 100 CIDs. Max practical: ~500.
    """
    result = PropertyResult(batch_size=len(cids))
    if not cids:
        return result

    start = time.perf_counter()
    url = f"{PUBCHEM_BASE}/compound/cid/property/{properties}/JSON"
    all_props = []
    chunks = [cids[i:i + batch_size] for i in range(0, len(cids), batch_size)]

    for chunk in chunks:
        cid_str = ",".join(str(c) for c in chunk)
        resp, _ = pubchem_post(url, data={"cid": cid_str}, timeout=PROPERTY_TIMEOUT)

        if resp is None:
            result.error = f"Failed at chunk starting CID {chunk[0]}"
            break

        try:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
            all_props.extend(props)
        except Exception as e:
            result.error = f"Parse error: {e}"
            break

    result.properties = all_props
    result.time_ms = (time.perf_counter() - start) * 1000
    return result


def pubchem_fetch_bioactivity(cid: int) -> BioactivityResult:
    """Fetch all bioactivity data for a CID via assaysummary.

    Endpoint: GET /compound/cid/{CID}/assaysummary/JSON
    WARNING: Can return MBs for well-studied compounds (aspirin, caffeine).
    Data is depositor-submitted, NOT curated like ChEMBL.

    Response format:
      {"Table": {"Columns": {"Column": [...]}, "Row": [{"Cell": [...]}, ...]}}

    Columns typically include:
      AID, Assay Name, Activity Outcome, Activity Name, Activity Value,
      Activity Qualifier, Target GI, Target Name, etc.
    """
    result = BioactivityResult()
    start = time.perf_counter()

    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/assaysummary/JSON"
    resp, _ = pubchem_get(url, timeout=BIOACTIVITY_TIMEOUT)
    result.time_ms = (time.perf_counter() - start) * 1000

    if resp is None:
        result.error = "Request failed"
        return result

    result.response_size_bytes = len(resp.content)

    try:
        data = resp.json()
        table = data.get("Table", {})
        columns = table.get("Columns", {}).get("Column", [])
        rows = table.get("Row", [])

        result.total_assays = len(rows)

        # Build column index map (positions vary per response)
        col_idx = {col: i for i, col in enumerate(columns)}
        outcome_idx = col_idx.get("Activity Outcome")
        name_idx = col_idx.get("Activity Name")
        target_gi_idx = col_idx.get("Target GI")

        targets_seen = set()

        for row in rows:
            cells = row.get("Cell", [])

            # Get outcome and activity name
            outcome = ""
            if outcome_idx is not None and outcome_idx < len(cells):
                outcome = cells[outcome_idx] or ""

            aname = ""
            if name_idx is not None and name_idx < len(cells):
                aname = cells[name_idx] or "(empty)"

            # Track unique targets
            target_id = ""
            if target_gi_idx is not None and target_gi_idx < len(cells):
                target_id = cells[target_gi_idx] or ""
            if target_id:
                targets_seen.add(target_id)

            # Count outcomes (unfiltered)
            if outcome == "Active":
                result.active_count += 1
            elif outcome == "Inactive":
                result.inactive_count += 1
            else:
                result.inconclusive_count += 1

            # Count all activity names (raw breakdown)
            if aname:
                result.by_activity_name[aname] = result.by_activity_name.get(aname, 0) + 1

            # Filter by app activity types (IC50, Ki, Kd, EC50, AC50, GI50, MIC)
            if aname in APP_ACTIVITY_TYPES:
                result.type_filtered_total += 1
                result.by_type_filtered[aname] = result.by_type_filtered.get(aname, 0) + 1
                if outcome == "Active":
                    result.type_filtered_active += 1
                    result.by_type_filtered_active[aname] = (
                        result.by_type_filtered_active.get(aname, 0) + 1
                    )

        result.unique_targets = len(targets_seen)

    except Exception as e:
        result.error = f"Parse error: {e}"

    return result


def pubchem_fetch_drug_indications(cid: int) -> DrugIndicationResult:
    """Fetch drug indication data from PubChem's SDQ API (Open Targets source).

    Uses the SphinxQL endpoint to query the opentargetsdrugindication collection.
    Returns disease names with max clinical trial phases (Phase I-IV).
    """
    result = DrugIndicationResult()
    start = time.perf_counter()

    query = json.dumps({
        "select": "*",
        "collection": "opentargetsdrugindication",
        "order": ["maxphase,desc"],
        "start": 1,
        "limit": 10000,
        "where": {"ands": [{"cid": str(cid)}]},
        "width": 1000000,
    })

    _rate_limit()
    try:
        resp = SESSION.get(
            PUBCHEM_SDQ,
            params={"infmt": "json", "outfmt": "json", "query": query},
            timeout=GENERAL_TIMEOUT,
        )
        result.time_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            result.error = f"HTTP {resp.status_code}"
            return result

        data = resp.json()
        output = data.get("SDQOutputSet", [{}])[0]
        rows = output.get("rows", [])
        result.total = output.get("totalCount", len(rows))

        for row in rows:
            disease = row.get("srcdiseasename", "")
            phase = row.get("maxphase", "")
            if disease:
                result.indications.append({"disease": disease, "phase": phase})
                result.by_phase[phase] = result.by_phase.get(phase, 0) + 1

    except Exception as e:
        result.time_ms = (time.perf_counter() - start) * 1000
        result.error = f"Parse error: {e}"

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_similarity_thresholds(smiles: str, name: str,
                                     thresholds: List[int] = None):
    """Test similarity at multiple thresholds."""
    if thresholds is None:
        thresholds = [70, 80, 90, 95]

    print(f"\n  Similarity Threshold Sweep: {name}")
    print(f"  {'Threshold':>10} | {'CIDs':>8} | {'Time (ms)':>10} | {'Status'}")
    print(f"  {'-' * 50}")

    results = {}
    for thr in thresholds:
        r = pubchem_similarity_search(smiles, threshold=thr, max_records=500)
        status = r.error or "OK"
        print(f"  {thr:>10}% | {len(r.cids):>8} | {r.time_ms:>10.0f} | {status}")
        results[thr] = r

    return results


def benchmark_property_batch_sizes(cids: List[int], name: str,
                                    sizes: List[int] = None):
    """Test property fetching at different batch sizes."""
    if sizes is None:
        sizes = [10, 50, 100, 200]

    print(f"\n  Property Batch Size Sweep: {name} ({len(cids)} CIDs available)")
    print(f"  {'Batch':>10} | {'Fetched':>8} | {'Time (ms)':>10} | {'ms/cmpd':>10} | {'Status'}")
    print(f"  {'-' * 60}")

    results = {}
    for size in sizes:
        subset = cids[:size]
        if not subset:
            continue
        r = pubchem_fetch_properties(subset, batch_size=size)
        per = r.time_ms / len(subset) if subset else 0
        status = r.error or "OK"
        print(f"  {size:>10} | {len(r.properties):>8} | {r.time_ms:>10.0f} | {per:>10.1f} | {status}")
        results[size] = r

    return results


def benchmark_rate_limits(n_requests: int = 10):
    """Rapid requests to monitor X-Throttling-Control header."""
    print(f"\n  Rate Limit Monitor ({n_requests} rapid requests, NO rate limiting)")
    print(f"  {'#':>4} | {'ms':>8} | {'HTTP':>5} | {'Req Count':>20} | {'Req Time':>20} | {'Service':>20}")
    print(f"  {'-' * 90}")

    url = f"{PUBCHEM_BASE}/compound/cid/2244/property/MolecularWeight/JSON"
    snapshots = []

    for i in range(n_requests):
        start = time.perf_counter()
        try:
            resp = SESSION.get(url, timeout=10)  # Skip rate limiter intentionally
            ms = (time.perf_counter() - start) * 1000
            throttle = resp.headers.get("X-Throttling-Control", "")
            info = _parse_throttle_header(throttle) if throttle else {}

            snap = ThrottleSnapshot(
                timestamp=time.perf_counter(),
                request_count_status=info.get("request_count", "N/A"),
                request_time_status=info.get("request_time", "N/A"),
                service_status=info.get("service", "N/A"),
            )
            snapshots.append(snap)

            flag = "  THROTTLED!" if resp.status_code == 503 else ""
            print(f"  {i+1:>4} | {ms:>8.0f} | {resp.status_code:>5} | "
                  f"{snap.request_count_status:>20} | {snap.request_time_status:>20} | "
                  f"{snap.service_status:>20}{flag}")
        except Exception as e:
            ms = (time.perf_counter() - start) * 1000
            print(f"  {i+1:>4} | {ms:>8.0f} |   ERR | {e}")

    return snapshots


# ═══════════════════════════════════════════════════════════════════════════
#  Compound Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_compound(name: str, smiles: str, cid: int, threshold: int,
                 max_records: int, prop_limit: int,
                 verbose: bool = False) -> CompoundBenchmark:
    """Run full PubChem analysis for one compound."""
    bench = CompoundBenchmark(name=name, smiles=smiles, cid=cid)

    # 1. Similarity
    print(f"\n  [SIMILARITY] Searching ({threshold}%, max {max_records})...",
          end="", flush=True)
    bench.similarity = pubchem_similarity_search(smiles, threshold, max_records)
    if bench.similarity.error:
        print(f" ERROR: {bench.similarity.error}")
    else:
        print(f" {len(bench.similarity.cids)} CIDs ({bench.similarity.time_ms:.0f} ms)")
        if len(bench.similarity.cids) > 100:
            print("         NOTE: PubChem has 118M+ compounds (vs ChEMBL ~2.5M) and uses")
            print("         881-bit substructure fingerprints (vs Morgan/ECFP). High CID")
            print("         counts include salts, stereoisomers, and vendor catalog entries.")
        if verbose and bench.similarity.cids:
            print(f"         First 10: {bench.similarity.cids[:10]}")

    # 2. Properties for similar compounds (capped to avoid slow bulk fetch)
    if bench.similarity.cids:
        total_cids = len(bench.similarity.cids)
        subset = bench.similarity.cids[:prop_limit]
        cap_note = f" (showing {len(subset)}/{total_cids})" if total_cids > prop_limit else ""
        print(f"  [PROPERTIES] Fetching for {len(subset)} CIDs{cap_note}...",
              end="", flush=True)
        bench.properties = pubchem_fetch_properties(subset)
        if bench.properties.error:
            print(f" ERROR: {bench.properties.error}")
        else:
            print(f" {len(bench.properties.properties)} compounds "
                  f"({bench.properties.time_ms:.0f} ms)")
            if verbose and bench.properties.properties:
                p = bench.properties.properties[0]
                print(f"         Sample: CID={p.get('CID')}, "
                      f"MW={p.get('MolecularWeight')}, "
                      f"XLogP={p.get('XLogP')}")

    # 3. Bioactivity for the query compound
    print(f"  [BIOACTIVITY] Fetching assay summary for CID {cid}...",
          end="", flush=True)
    bench.bioactivity = pubchem_fetch_bioactivity(cid)
    if bench.bioactivity.error:
        print(f" ERROR: {bench.bioactivity.error}")
    else:
        size_kb = bench.bioactivity.response_size_bytes / 1024
        print(f" {bench.bioactivity.total_assays} assays, "
              f"{bench.bioactivity.active_count} active "
              f"({size_kb:.0f} KB, {bench.bioactivity.time_ms:.0f} ms)")
        print("         NOTE: PubChem assays are depositor-submitted, not curated like ChEMBL.")
        print("         Activity names (IC50, etc.) don't map 1:1 to ChEMBL standard_type.")

    # 4. Drug indications (Open Targets via SDQ)
    if cid:
        print(f"  [INDICATIONS] Fetching drug indications for CID {cid}...",
              end="", flush=True)
        bench.indications = pubchem_fetch_drug_indications(cid)
        if bench.indications.error:
            print(f" ERROR: {bench.indications.error}")
        else:
            print(f" {bench.indications.total} indications "
                  f"({bench.indications.time_ms:.0f} ms)")
            if bench.indications.by_phase:
                phases = sorted(bench.indications.by_phase.items(),
                                key=lambda x: x[0], reverse=True)
                phase_str = ", ".join(f"{p}: {c}" for p, c in phases)
                print(f"         By phase: {phase_str}")

    # Summary table
    _print_summary(bench, verbose)
    return bench


def _print_summary(bench: CompoundBenchmark, verbose: bool = False):
    """Print summary table for one compound."""
    sep = "-" * 72
    sim = bench.similarity
    props = bench.properties
    bio = bench.bioactivity

    print(f"\n  {sep}")
    print(f"  SUMMARY: {bench.name} (CID {bench.cid})")
    print(f"  {sep}")
    print(f"  {'Metric':<35} | {'Value':>12} | {'Time (ms)':>10}")
    print(f"  {sep}")
    print(f"  {'Similar CIDs found':<35} | {len(sim.cids):>12} | {sim.time_ms:>10.0f}")
    print(f"  {'Properties fetched':<35} | {len(props.properties):>12} | {props.time_ms:>10.0f}")
    print(f"  {'Total assays (all types)':<35} | {bio.total_assays:>12} | {bio.time_ms:>10.0f}")
    print(f"  {'Active (all types)':<35} | {bio.active_count:>12} |")
    print(f"  {'Inactive (all types)':<35} | {bio.inactive_count:>12} |")
    print(f"  {'Other/Inconclusive':<35} | {bio.inconclusive_count:>12} |")
    types_str = ",".join(sorted(APP_ACTIVITY_TYPES))
    print(f"  {sep}")
    print(f"  App-filtered ({types_str}):")
    print(f"  {'  Assays matching app types':<35} | {bio.type_filtered_total:>12} |")
    print(f"  {'  Active (app types only)':<35} | {bio.type_filtered_active:>12} |")
    print(f"  {'Unique targets':<35} | {bio.unique_targets:>12} |")
    ind = bench.indications
    print(f"  {sep}")
    print(f"  {'Drug indications (Open Targets)':<35} | {ind.total:>12} | {ind.time_ms:>10.0f}")
    if ind.by_phase:
        for phase in sorted(ind.by_phase.keys(), reverse=True):
            label = f"  {phase}"
            print(f"  {label:<35} | {ind.by_phase[phase]:>12} |")
    print(f"  {'Response size':<35} | {bio.response_size_bytes/1024:>9.0f} KB |")
    print(f"  {sep}")

    # App activity type breakdown (filtered)
    if bio.by_type_filtered:
        print("\n  App Activity Type Breakdown (filtered):")
        print(f"  {'Type':<12} | {'Total':>8} | {'Active':>8}")
        print(f"  {'-' * 35}")
        for atype in sorted(APP_ACTIVITY_TYPES):
            total = bio.by_type_filtered.get(atype, 0)
            active = bio.by_type_filtered_active.get(atype, 0)
            if total > 0:
                print(f"  {atype:<12} | {total:>8} | {active:>8}")

    # All activity names (verbose)
    if verbose and bio.by_activity_name:
        top = sorted(bio.by_activity_name.items(), key=lambda x: -x[1])[:20]
        print("\n  All Activity Names (top 20, unfiltered):")
        print(f"  {'Name':<35} | {'Count':>8} | {'In App?':>8}")
        print(f"  {'-' * 58}")
        for aname, count in top:
            in_app = "YES" if aname in APP_ACTIVITY_TYPES else ""
            print(f"  {aname:<35} | {count:>8} | {in_app:>8}")
        if len(bio.by_activity_name) > 20:
            print(f"  ... and {len(bio.by_activity_name) - 20} more")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PubChem PUG REST API endpoint exploration and benchmark."
    )
    parser.add_argument("--threshold", "-t", type=int, default=90,
                        help="Similarity threshold (default: 90)")
    parser.add_argument("--compound", "-c", type=str, nargs="+", default=None,
                        help=f"Built-in: {', '.join(COMPOUNDS.keys())}")
    parser.add_argument("--smiles", "-s", type=str, default=None,
                        help="Custom SMILES input")
    parser.add_argument("--cid", type=int, default=None,
                        help="PubChem CID for bioactivity (with --smiles)")
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Display name for custom compound")
    parser.add_argument("--benchmark", "-b", action="store_true",
                        help="Run threshold/batch/rate benchmarks")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show CID lists, property values, activity names")
    parser.add_argument("--max-records", type=int, default=10000,
                        help="Max CIDs from similarity search (default: 10000)")
    parser.add_argument("--prop-limit", type=int, default=200,
                        help="Max CIDs for property fetch (default: 200, use 0 for all)")
    args = parser.parse_args()

    threshold = args.threshold

    print_header("PubChem PUG REST API Exploration")
    print(f"  Threshold:       {threshold}%")
    print(f"  Rate limit:      {RATE_LIMIT_INTERVAL}s between requests "
          f"({1/RATE_LIMIT_INTERVAL:.0f} req/s)")
    print(f"  Max records:     {args.max_records}")
    print(f"  Prop limit:      {args.prop_limit if args.prop_limit else 'all'}")
    print("  Fingerprint:     PubChem 2D (881-bit substructure keys)")

    # Collect compounds
    compounds_to_run = []

    if args.smiles:
        cid = args.cid or 0
        compounds_to_run.append((args.name or "Custom", args.smiles, cid))

    default_compounds = ["aspirin", "caffeine", "quercetin"]
    compound_keys = (args.compound if args.compound
                     else (default_compounds if not compounds_to_run else []))
    for key in compound_keys:
        k = key.lower()
        if k not in COMPOUNDS:
            print(f"\n  [WARN] Unknown '{key}'. "
                  f"Available: {', '.join(COMPOUNDS.keys())}")
            continue
        info = COMPOUNDS[k]
        compounds_to_run.append((info["name"], info["smiles"], info["cid"]))

    if not compounds_to_run:
        print("\n  No compounds. Use --compound or --smiles.")
        return

    # Run compounds
    all_results = []
    for name, smiles, cid in compounds_to_run:
        print_header(f"{name}  |  CID: {cid}  |  "
                     f"SMILES: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
        prop_limit = args.prop_limit if args.prop_limit else len(COMPOUNDS) * 10000
        bench = run_compound(name, smiles, cid, threshold,
                             args.max_records, prop_limit, args.verbose)
        all_results.append(bench)

    # Benchmarks
    if args.benchmark and compounds_to_run:
        print_header("BENCHMARKS")
        first_name, first_smiles, _ = compounds_to_run[0]

        benchmark_similarity_thresholds(first_smiles, first_name)

        wide = pubchem_similarity_search(first_smiles, threshold=70,
                                          max_records=500)
        if wide.cids:
            benchmark_property_batch_sizes(wide.cids, first_name)

        benchmark_rate_limits(n_requests=10)

    # JSON
    if args.json:
        json_out = {}
        for bench in all_results:
            json_out[bench.name] = {
                "smiles": bench.smiles,
                "cid": bench.cid,
                "similarity": {
                    "cid_count": len(bench.similarity.cids),
                    "threshold": bench.similarity.threshold,
                    "time_ms": round(bench.similarity.time_ms, 1),
                    "cids_sample": bench.similarity.cids[:20],
                },
                "properties": {
                    "count": len(bench.properties.properties),
                    "time_ms": round(bench.properties.time_ms, 1),
                },
                "bioactivity": {
                    "total_assays": bench.bioactivity.total_assays,
                    "active_all_types": bench.bioactivity.active_count,
                    "inactive_all_types": bench.bioactivity.inactive_count,
                    "type_filtered_total": bench.bioactivity.type_filtered_total,
                    "type_filtered_active": bench.bioactivity.type_filtered_active,
                    "by_type_filtered": bench.bioactivity.by_type_filtered,
                    "by_type_filtered_active": bench.bioactivity.by_type_filtered_active,
                    "unique_targets": bench.bioactivity.unique_targets,
                    "response_size_kb": round(
                        bench.bioactivity.response_size_bytes / 1024, 1),
                    "time_ms": round(bench.bioactivity.time_ms, 1),
                    "top_activity_names": dict(sorted(
                        bench.bioactivity.by_activity_name.items(),
                        key=lambda x: -x[1],
                    )[:15]),
                },
                "indications": {
                    "total": bench.indications.total,
                    "by_phase": bench.indications.by_phase,
                    "top_diseases": [
                        {"disease": i["disease"], "phase": i["phase"]}
                        for i in bench.indications.indications[:20]
                    ],
                    "time_ms": round(bench.indications.time_ms, 1),
                },
            }
        print("\n\n--- JSON Output ---")
        print(json.dumps(json_out, indent=2))

    print()


if __name__ == "__main__":
    main()