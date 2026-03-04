"""
ChEMBL Similarity & Activity Count Comparison (ChEMBL REST vs PubChem)

Compares counts of similar compounds, activities, targets, PDB entries,
and drug indications — using ChEMBL REST API and PubChem side-by-side
to understand data coverage differences between the two sources.

Supports multiple input types:
  - SMILES:    CC(=O)OC1=CC=CC=C1C(=O)O
  - InChIKey:  WCGUUGGRBIKTOS-GPOJBZKASA-N
  - InChI:     InChI=1S/C8H10N4O2/...

Usage:
    python scripts/check_similarity_activities.py
    python scripts/check_similarity_activities.py --compound caffeine
    python scripts/check_similarity_activities.py --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    python scripts/check_similarity_activities.py --inchikey WCGUUGGRBIKTOS-GPOJBZKASA-N --name "Ursolic Acid"
    python scripts/check_similarity_activities.py --json
"""
import argparse
import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_SDQ = "https://pubchem.ncbi.nlm.nih.gov/sdq/sphinxql.cgi"

MAX_LIMIT = 1000
SIMILARITY_TIMEOUT = 120
ACTIVITY_TIMEOUT = 60
PDB_TIMEOUT = 45
GENERAL_TIMEOUT = 30
RATE_LIMIT_INTERVAL = 0.15

# App default activity types (matches api_client.py + jobs.py + frontend config)
APP_ACTIVITY_TYPES = ['IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'GI50', 'MIC']

# Valid units for nM conversion (from compound_service.py:478-489)
VALID_UNITS = {'nM', 'uM', 'mM', 'pM', 'M'}

# Built-in test compounds
COMPOUNDS = {
    "ursolic":      {"smiles": None, "inchikey": "WCGUUGGRBIKTOS-GPOJBZKASA-N",  "name": "Ursolic Acid"},
    "caffeine":     {"smiles": None, "inchi": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3", "name": "Caffeine"},
    "kaempferol":   {"smiles": "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",       "name": "Kaempferol"},
    "quercetin":    {"smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",   "name": "Quercetin"},
    "aspirin":      {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",                      "name": "Aspirin"},
    "ibuprofen":    {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",                "name": "Ibuprofen"},
    "ethanol":      {"smiles": "CCO",                                            "name": "Ethanol"},
}

INCHIKEY_PATTERN = re.compile(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$')
INCHI_PATTERN = re.compile(r'^InChI=')


# ═══════════════════════════════════════════════════════════════════════════
#  HTTP session
# ═══════════════════════════════════════════════════════════════════════════

def create_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = create_session()
_last_chembl_time = 0.0


def chembl_get(url: str, params: dict, timeout: int) -> Optional[requests.Response]:
    global _last_chembl_time
    elapsed = time.perf_counter() - _last_chembl_time
    if elapsed < RATE_LIMIT_INTERVAL:
        time.sleep(RATE_LIMIT_INTERVAL - elapsed)
    _last_chembl_time = time.perf_counter()
    try:
        resp = SESSION.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.exceptions.RequestException as e:
        print(f"    [ERROR] {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Input Resolution
# ═══════════════════════════════════════════════════════════════════════════

def resolve_inchikey_to_smiles(inchikey: str) -> Optional[str]:
    # Keep property list aligned with app-side PubChem handling.
    url = (
        f"{PUBCHEM_BASE}/compound/inchikey/{inchikey}/property/"
        "ConnectivitySMILES,CanonicalSMILES,IsomericSMILES,SMILES/JSON"
    )
    try:
        resp = SESSION.get(url, timeout=GENERAL_TIMEOUT)
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            return (
                props.get("ConnectivitySMILES")
                or props.get("SMILES")
                or props.get("CanonicalSMILES")
                or props.get("IsomericSMILES")
            )
    except Exception as e:
        print(f"    [PubChem] {e}")
    # Fallback: ChEMBL
    try:
        params = {"molecule_structures__standard_inchi_key": inchikey,
                  "only": "molecule_chembl_id,molecule_structures", "limit": 1}
        resp = chembl_get(f"{CHEMBL_BASE}/molecule.json", params, GENERAL_TIMEOUT)
        if resp:
            mols = resp.json().get("molecules", [])
            if mols:
                return mols[0].get("molecule_structures", {}).get("canonical_smiles")
    except Exception as e:
        print(f"    [ChEMBL] {e}")
    return None


def resolve_inchi_to_smiles(inchi: str) -> Optional[str]:
    url = (
        f"{PUBCHEM_BASE}/compound/inchi/property/"
        "ConnectivitySMILES,CanonicalSMILES,IsomericSMILES,SMILES/JSON"
    )
    try:
        resp = SESSION.post(url, data={"inchi": inchi}, timeout=GENERAL_TIMEOUT)
        if resp.status_code == 200:
            props = resp.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            return (
                props.get("ConnectivitySMILES")
                or props.get("SMILES")
                or props.get("CanonicalSMILES")
                or props.get("IsomericSMILES")
            )
    except Exception as e:
        print(f"    [PubChem] {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Counts:
    """Activity counts with filtering breakdown."""
    raw_total: int = 0               # All activities returned by API
    type_filtered: int = 0           # After activity type filter
    value_filtered: int = 0          # After standard_value + unit filter (= app count)
    by_type_raw: Dict[str, int] = field(default_factory=dict)
    by_type_filtered: Dict[str, int] = field(default_factory=dict)
    dropped_no_value: int = 0
    dropped_bad_value: int = 0
    dropped_bad_units: int = 0
    dropped_wrong_type: int = 0
    unique_targets: int = 0
    time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class CompoundResult:
    name: str
    smiles: str
    threshold: int
    # Similarity
    rest_compounds: List[str] = field(default_factory=list)
    sim_rest_ms: float = 0.0
    sim_rest_error: Optional[str] = None
    # Activities (REST)
    rest_counts: Counts = field(default_factory=Counts)
    # PDB
    pdb_entries: int = 0
    pdb_time_ms: float = 0.0
    pdb_error: Optional[str] = None
    # Drug Indications
    drug_indications: int = 0
    indication_details: List[str] = field(default_factory=list)
    ind_time_ms: float = 0.0
    ind_error: Optional[str] = None
    # PubChem Similarity
    pubchem_cids: List[int] = field(default_factory=list)
    sim_pubchem_ms: float = 0.0
    sim_pubchem_error: Optional[str] = None
    # PubChem Bioactivity
    pubchem_total_assays: int = 0
    pubchem_active_assays: int = 0
    pubchem_type_filtered_total: int = 0
    pubchem_type_filtered_active: int = 0
    pubchem_by_type_filtered: Dict[str, int] = field(default_factory=dict)
    pubchem_by_type_filtered_active: Dict[str, int] = field(default_factory=dict)
    pubchem_unique_targets: int = 0
    pubchem_response_kb: float = 0.0
    pubchem_bio_time_ms: float = 0.0
    pubchem_bio_error: Optional[str] = None
    # PubChem Drug Indications (Open Targets)
    pubchem_indications: int = 0
    pubchem_indication_details: List[str] = field(default_factory=list)
    pubchem_indication_by_phase: Dict[str, int] = field(default_factory=dict)
    pubchem_ind_time_ms: float = 0.0
    pubchem_ind_error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
#  Activity value filter (mirrors compound_service.py:468-489)
# ═══════════════════════════════════════════════════════════════════════════

def is_valid_activity(act: dict) -> bool:
    """Check if activity passes the app's value/unit filter."""
    std_value = act.get("standard_value")
    std_units = act.get("standard_units")

    if not std_value:
        return False
    try:
        value = float(std_value)
        if value <= 0:
            return False
    except (ValueError, TypeError):
        return False

    if std_units not in VALID_UNITS:
        return False

    return True


def classify_drop_reason(act: dict) -> str:
    """Why was this activity dropped?"""
    std_value = act.get("standard_value")
    std_units = act.get("standard_units")

    if not std_value:
        return "no_value"
    try:
        value = float(std_value)
        if value <= 0:
            return "bad_value"
    except (ValueError, TypeError):
        return "bad_value"

    if std_units not in VALID_UNITS:
        return "bad_units"

    return "ok"


# ═══════════════════════════════════════════════════════════════════════════
#  REST API functions
# ═══════════════════════════════════════════════════════════════════════════

def rest_similarity_search(smiles: str, threshold: int) -> Tuple[List[str], float, Optional[str]]:
    start = time.perf_counter()
    all_ids = []
    offset = 0
    error = None

    while True:
        params = {"smiles": smiles, "similarity": threshold,
                  "only": "molecule_chembl_id", "limit": MAX_LIMIT, "offset": offset}
        resp = chembl_get(f"{CHEMBL_BASE}/similarity.json", params, SIMILARITY_TIMEOUT)
        if resp is None:
            error = "Request failed"
            break
        molecules = resp.json().get("molecules", [])
        if not molecules:
            break
        for mol in molecules:
            cid = mol.get("molecule_chembl_id")
            if cid:
                all_ids.append(cid)
        if len(molecules) < MAX_LIMIT:
            break
        offset += MAX_LIMIT

    ms = (time.perf_counter() - start) * 1000
    return all_ids, ms, error


def rest_fetch_activities(chembl_ids: List[str], activity_types: List[str]) -> Counts:
    """Fetch activities via REST and apply both type and value filters."""
    counts = Counts()
    if not chembl_ids:
        return counts

    start = time.perf_counter()
    types_set = set(activity_types)
    targets = set()
    max_per_req = 50
    chunks = [chembl_ids[i:i + max_per_req] for i in range(0, len(chembl_ids), max_per_req)]

    for ci, chunk in enumerate(chunks):
        ids_param = ",".join(chunk)
        offset = 0
        while True:
            params = {
                "molecule_chembl_id__in": ids_param,
                "only": "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id",
                "limit": MAX_LIMIT, "offset": offset,
            }
            resp = chembl_get(f"{CHEMBL_BASE}/activity.json", params, ACTIVITY_TIMEOUT)
            if resp is None:
                counts.error = f"Failed at chunk {ci + 1}/{len(chunks)}"
                break

            activities = resp.json().get("activities", [])
            if not activities:
                break

            for act in activities:
                counts.raw_total += 1
                stype = act.get("standard_type", "")

                # Track ALL types in raw breakdown
                counts.by_type_raw[stype] = counts.by_type_raw.get(stype, 0) + 1

                # Filter 1: activity type
                if stype not in types_set:
                    counts.dropped_wrong_type += 1
                    continue
                counts.type_filtered += 1

                # Filter 2: value + units (same as app)
                reason = classify_drop_reason(act)
                if reason == "no_value":
                    counts.dropped_no_value += 1
                    continue
                elif reason == "bad_value":
                    counts.dropped_bad_value += 1
                    continue
                elif reason == "bad_units":
                    counts.dropped_bad_units += 1
                    continue

                counts.value_filtered += 1
                counts.by_type_filtered[stype] = counts.by_type_filtered.get(stype, 0) + 1
                tgt = act.get("target_chembl_id")
                if tgt:
                    targets.add(tgt)

            if len(activities) < MAX_LIMIT:
                break
            offset += MAX_LIMIT

        if counts.error:
            break

    counts.unique_targets = len(targets)
    counts.time_ms = (time.perf_counter() - start) * 1000
    return counts


# ═══════════════════════════════════════════════════════════════════════════
#  PDB + Drug Indications (REST only)
# ═══════════════════════════════════════════════════════════════════════════

def search_pdb(smiles: str) -> Tuple[int, float, Optional[str]]:
    start = time.perf_counter()
    try:
        query = {
            "query": {"type": "terminal", "service": "chemical",
                      "parameters": {"value": smiles, "type": "descriptor",
                                     "descriptor_type": "SMILES", "match_type": "graph-relaxed"}},
            "request_options": {"return_all_hits": True},
            "return_type": "entry"
        }
        resp = SESSION.post(PDB_SEARCH_URL, json=query, timeout=PDB_TIMEOUT,
                            headers={"Content-Type": "application/json"})
        ms = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            return len(resp.json().get("result_set", [])), ms, None
        elif resp.status_code == 204:
            return 0, ms, None
        return 0, ms, f"HTTP {resp.status_code}"
    except Exception as e:
        return 0, (time.perf_counter() - start) * 1000, str(e)


def fetch_drug_indications(chembl_ids: List[str]) -> Tuple[int, List[str], float, Optional[str]]:
    if not chembl_ids:
        return 0, [], 0.0, None
    start = time.perf_counter()
    indications = []
    max_per_req = 50
    chunks = [chembl_ids[i:i + max_per_req] for i in range(0, len(chembl_ids), max_per_req)]
    error = None

    for ci, chunk in enumerate(chunks):
        ids_param = ",".join(chunk)
        offset = 0
        while True:
            params = {"molecule_chembl_id__in": ids_param, "limit": MAX_LIMIT, "offset": offset}
            resp = chembl_get(f"{CHEMBL_BASE}/drug_indication.json", params, ACTIVITY_TIMEOUT)
            if resp is None:
                error = f"Failed at chunk {ci + 1}/{len(chunks)}"
                break
            data = resp.json().get("drug_indications", [])
            if not data:
                break
            for ind in data:
                indications.append(ind.get("mesh_heading", "Unknown"))
            if len(data) < MAX_LIMIT:
                break
            offset += MAX_LIMIT
        if error:
            break

    ms = (time.perf_counter() - start) * 1000
    return len(indications), sorted(set(indications)), ms, error


# ═══════════════════════════════════════════════════════════════════════════
#  PubChem Functions
# ═══════════════════════════════════════════════════════════════════════════

PUBCHEM_RATE_LIMIT = 0.25
_last_pubchem_time = 0.0


def pubchem_rate_limited():
    """Enforce PubChem rate limiting (0.25s = 4 req/s)."""
    global _last_pubchem_time
    elapsed = time.perf_counter() - _last_pubchem_time
    if elapsed < PUBCHEM_RATE_LIMIT:
        time.sleep(PUBCHEM_RATE_LIMIT - elapsed)
    _last_pubchem_time = time.perf_counter()


def resolve_smiles_to_cid(smiles: str) -> Optional[int]:
    """Resolve SMILES to PubChem CID."""
    pubchem_rate_limited()
    url = f"{PUBCHEM_BASE}/compound/smiles/cids/JSON"
    try:
        resp = SESSION.post(url, data={"smiles": smiles}, timeout=GENERAL_TIMEOUT)
        if resp.status_code == 200:
            cids = resp.json().get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
    except Exception:
        pass
    return None


def pc_similarity_search(smiles: str, threshold: int
                         ) -> Tuple[List[int], float, Optional[str]]:
    """PubChem 2D similarity search. Returns (cids, ms, error)."""
    start = time.perf_counter()
    pubchem_rate_limited()
    url = f"{PUBCHEM_BASE}/compound/fastsimilarity_2d/smiles/cids/JSON"
    try:
        resp = SESSION.post(
            url,
            data={"smiles": smiles},
            params={"Threshold": threshold, "MaxRecords": 10000},
            timeout=SIMILARITY_TIMEOUT,
        )
        ms = (time.perf_counter() - start) * 1000
        if resp.status_code != 200:
            return [], ms, f"HTTP {resp.status_code}"
        cids = resp.json().get("IdentifierList", {}).get("CID", [])
        return cids, ms, None
    except Exception as e:
        return [], (time.perf_counter() - start) * 1000, str(e)


@dataclass
class PubChemBioResult:
    """Full PubChem bioactivity result with type filtering."""
    total: int = 0
    active: int = 0
    type_filtered_total: int = 0
    type_filtered_active: int = 0
    by_type_filtered: Dict[str, int] = field(default_factory=dict)
    by_type_filtered_active: Dict[str, int] = field(default_factory=dict)
    unique_targets: int = 0
    size_kb: float = 0.0
    ms: float = 0.0
    error: Optional[str] = None


def pc_fetch_bioactivity(cid: int) -> PubChemBioResult:
    """PubChem assay summary with activity type filtering and target tracking."""
    result = PubChemBioResult()
    start = time.perf_counter()
    pubchem_rate_limited()
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/assaysummary/JSON"
    try:
        resp = SESSION.get(url, timeout=SIMILARITY_TIMEOUT)
        result.ms = (time.perf_counter() - start) * 1000
        result.size_kb = len(resp.content) / 1024
        if resp.status_code != 200:
            result.error = f"HTTP {resp.status_code}"
            return result
        data = resp.json()
        rows = data.get("Table", {}).get("Row", [])
        columns = data.get("Table", {}).get("Columns", {}).get("Column", [])

        col_idx = {col: i for i, col in enumerate(columns)}
        outcome_idx = col_idx.get("Activity Outcome")
        name_idx = col_idx.get("Activity Name")
        target_gi_idx = col_idx.get("Target GI")

        result.total = len(rows)
        types_set = set(APP_ACTIVITY_TYPES)
        targets_seen = set()

        for row in rows:
            cells = row.get("Cell", [])
            outcome = ""
            if outcome_idx is not None and outcome_idx < len(cells):
                outcome = cells[outcome_idx] or ""
            aname = ""
            if name_idx is not None and name_idx < len(cells):
                aname = cells[name_idx] or ""

            # Track targets
            if target_gi_idx is not None and target_gi_idx < len(cells):
                tgt = cells[target_gi_idx] or ""
                if tgt:
                    targets_seen.add(tgt)

            if outcome == "Active":
                result.active += 1

            # Filter by app activity types
            if aname in types_set:
                result.type_filtered_total += 1
                result.by_type_filtered[aname] = result.by_type_filtered.get(aname, 0) + 1
                if outcome == "Active":
                    result.type_filtered_active += 1
                    result.by_type_filtered_active[aname] = (
                        result.by_type_filtered_active.get(aname, 0) + 1
                    )

        result.unique_targets = len(targets_seen)
        return result
    except Exception as e:
        result.ms = (time.perf_counter() - start) * 1000
        result.error = str(e)
        return result


def pc_fetch_drug_indications(cid: int
                              ) -> Tuple[int, List[str], Dict[str, int], float, Optional[str]]:
    """Fetch drug indications from PubChem SDQ (Open Targets).

    Returns (total, disease_names, by_phase, ms, error).
    """
    start = time.perf_counter()
    pubchem_rate_limited()
    query = json.dumps({
        "select": "*",
        "collection": "opentargetsdrugindication",
        "order": ["maxphase,desc"],
        "start": 1,
        "limit": 10000,
        "where": {"ands": [{"cid": str(cid)}]},
        "width": 1000000,
    })
    try:
        resp = SESSION.get(
            PUBCHEM_SDQ,
            params={"infmt": "json", "outfmt": "json", "query": query},
            timeout=GENERAL_TIMEOUT,
        )
        ms = (time.perf_counter() - start) * 1000
        if resp.status_code != 200:
            return 0, [], {}, ms, f"HTTP {resp.status_code}"
        data = resp.json()
        output = data.get("SDQOutputSet", [{}])[0]
        rows = output.get("rows", [])
        total = output.get("totalCount", len(rows))
        diseases = []
        by_phase: Dict[str, int] = {}
        for row in rows:
            disease = row.get("srcdiseasename", "")
            phase = row.get("maxphase", "")
            if disease:
                diseases.append(f"{disease} ({phase})" if phase else disease)
                by_phase[phase] = by_phase.get(phase, 0) + 1
        return total, sorted(set(diseases)), by_phase, ms, None
    except Exception as e:
        return 0, [], {}, (time.perf_counter() - start) * 1000, str(e)


# ═══════════════════════════════════════════════════════════════════════════
#  Display
# ═══════════════════════════════════════════════════════════════════════════

def print_header(text: str):
    width = 80
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def run_compound(name: str, smiles: str, threshold: int) -> CompoundResult:
    """Run full analysis for one compound."""
    r = CompoundResult(name=name, smiles=smiles, threshold=threshold)

    # ── Similarity: REST ──
    print(f"\n  [REST] Similarity search ({threshold}%)...", end="", flush=True)
    r.rest_compounds, r.sim_rest_ms, r.sim_rest_error = rest_similarity_search(smiles, threshold)
    if r.sim_rest_error:
        print(f" ERROR: {r.sim_rest_error}")
    else:
        print(f" {len(r.rest_compounds)} compounds ({r.sim_rest_ms:.0f} ms)")
        if r.rest_compounds:
            print(f"         IDs: {', '.join(r.rest_compounds[:5])}{'...' if len(r.rest_compounds) > 5 else ''}")

    # ── Activities: REST ──
    if r.rest_compounds and not r.sim_rest_error:
        print(f"  [REST] Fetching activities for {len(r.rest_compounds)} compounds...", end="", flush=True)
        r.rest_counts = rest_fetch_activities(r.rest_compounds, APP_ACTIVITY_TYPES)
        if r.rest_counts.error:
            print(f" ERROR: {r.rest_counts.error}")
        else:
            print(f" {r.rest_counts.value_filtered} activities (app-filtered), "
                  f"{r.rest_counts.raw_total} raw ({r.rest_counts.time_ms:.0f} ms)")

    # ── PDB ──
    print("  [PDB] Searching structural entries...", end="", flush=True)
    r.pdb_entries, r.pdb_time_ms, r.pdb_error = search_pdb(smiles)
    if r.pdb_error:
        print(f" ERROR: {r.pdb_error}")
    else:
        print(f" {r.pdb_entries} entries ({r.pdb_time_ms:.0f} ms)")

    # ── Drug Indications ──
    ids_for_ind = r.rest_compounds
    if ids_for_ind:
        print("  [REST] Fetching drug indications...", end="", flush=True)
        r.drug_indications, r.indication_details, r.ind_time_ms, r.ind_error = fetch_drug_indications(ids_for_ind)
        if r.ind_error:
            print(f" ERROR: {r.ind_error}")
        else:
            print(f" {r.drug_indications} indications ({r.ind_time_ms:.0f} ms)")
            if r.indication_details:
                print(f"         {', '.join(r.indication_details[:10])}")
                if len(r.indication_details) > 10:
                    print(f"         ... and {len(r.indication_details) - 10} more")

    # ── PubChem Similarity ──
    print(f"  [PUBCHEM] Similarity search ({threshold}%)...", end="", flush=True)
    r.pubchem_cids, r.sim_pubchem_ms, r.sim_pubchem_error = pc_similarity_search(smiles, threshold)
    if r.sim_pubchem_error:
        print(f" ERROR: {r.sim_pubchem_error}")
    else:
        print(f" {len(r.pubchem_cids)} CIDs ({r.sim_pubchem_ms:.0f} ms)")

    # ── PubChem Bioactivity ──
    print("  [PUBCHEM] Resolving SMILES to CID...", end="", flush=True)
    pc_cid = resolve_smiles_to_cid(smiles)
    if pc_cid:
        print(f" CID {pc_cid}")
        print(f"  [PUBCHEM] Fetching assay summary for CID {pc_cid}...", end="", flush=True)
        bio = pc_fetch_bioactivity(pc_cid)
        r.pubchem_total_assays = bio.total
        r.pubchem_active_assays = bio.active
        r.pubchem_type_filtered_total = bio.type_filtered_total
        r.pubchem_type_filtered_active = bio.type_filtered_active
        r.pubchem_by_type_filtered = bio.by_type_filtered
        r.pubchem_by_type_filtered_active = bio.by_type_filtered_active
        r.pubchem_unique_targets = bio.unique_targets
        r.pubchem_response_kb = bio.size_kb
        r.pubchem_bio_time_ms = bio.ms
        r.pubchem_bio_error = bio.error
        if r.pubchem_bio_error:
            print(f" ERROR: {r.pubchem_bio_error}")
        else:
            print(f" {r.pubchem_total_assays} assays, "
                  f"{r.pubchem_type_filtered_active} active (app-filtered), "
                  f"{r.pubchem_unique_targets} targets "
                  f"({r.pubchem_response_kb:.0f} KB, {r.pubchem_bio_time_ms:.0f} ms)")
        # ── PubChem Drug Indications ──
        print(f"  [PUBCHEM] Fetching drug indications for CID {pc_cid}...", end="", flush=True)
        (r.pubchem_indications, r.pubchem_indication_details,
         r.pubchem_indication_by_phase, r.pubchem_ind_time_ms,
         r.pubchem_ind_error) = pc_fetch_drug_indications(pc_cid)
        if r.pubchem_ind_error:
            print(f" ERROR: {r.pubchem_ind_error}")
        else:
            print(f" {r.pubchem_indications} indications ({r.pubchem_ind_time_ms:.0f} ms)")
            if r.pubchem_indication_by_phase:
                phases = sorted(r.pubchem_indication_by_phase.items(),
                                key=lambda x: x[0], reverse=True)
                phase_str = ", ".join(f"{p}: {c}" for p, c in phases)
                print(f"         By phase: {phase_str}")
    else:
        print(" not found in PubChem")

    # ── Summary ──
    sep = "-" * 68
    print(f"\n  {sep}")
    print(f"  SUMMARY: {name} @ {threshold}% similarity")
    print(f"  {sep}")
    print(f"  {'Metric':<30} | {'ChEMBL REST':>14} | {'PubChem':>14}")
    print(f"  {sep}")
    print(f"  {'Similar compounds':<30} | {len(r.rest_compounds):>14} | {len(r.pubchem_cids):>14}")
    print(f"  {'Activities (raw/all types)':<30} | {r.rest_counts.raw_total:>14} | {r.pubchem_total_assays:>14}")
    print(f"  {'Activities (type-filtered)':<30} | {r.rest_counts.type_filtered:>14} | {r.pubchem_type_filtered_total:>14}")
    print(f"  {'Activities (app-filtered)':<30} | {r.rest_counts.value_filtered:>14} | {r.pubchem_type_filtered_active:>14}")
    print(f"  {'Unique targets':<30} | {r.rest_counts.unique_targets:>14} | {r.pubchem_unique_targets:>14}")
    print(f"  {'PDB entries':<30} | {r.pdb_entries:>14} | {'':>14}")
    print(f"  {'Drug indications':<30} | {r.drug_indications:>14} | {r.pubchem_indications:>14}")
    print(f"  {sep}")

    # ── ChEMBL REST vs PubChem Activity Type Comparison ──
    all_types = sorted(set(
        list(r.rest_counts.by_type_filtered.keys()) +
        list(r.pubchem_by_type_filtered.keys())
    ))
    if all_types:
        print("\n  Activity Type Breakdown (type-filtered, ChEMBL REST vs PubChem):")
        print(f"  {'Type':<12} | {'ChEMBL REST':>12} | {'PubChem':>10} | {'PC Active':>10}")
        print(f"  {'-' * 52}")
        for t in all_types:
            rv = r.rest_counts.by_type_filtered.get(t, 0)
            pv = r.pubchem_by_type_filtered.get(t, 0)
            pa = r.pubchem_by_type_filtered_active.get(t, 0)
            print(f"  {t:<12} | {rv:>12} | {pv:>10} | {pa:>10}")

    return r


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare ChEMBL REST vs PubChem at given similarity threshold."
    )
    parser.add_argument("--threshold", "-t", type=int, default=90, help="Similarity threshold (default: 90)")
    parser.add_argument("--compound", "-c", type=str, nargs="+", default=None,
                        help=f"Built-in compounds ({', '.join(COMPOUNDS.keys())})")
    parser.add_argument("--smiles", "-s", type=str, default=None, help="SMILES input")
    parser.add_argument("--inchikey", "-k", type=str, default=None, help="InChIKey input")
    parser.add_argument("--inchi", "-i", type=str, default=None, help="InChI input")
    parser.add_argument("--name", "-n", type=str, default=None, help="Display name")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    threshold = args.threshold

    print_header("ChEMBL REST vs PubChem Comparison")
    print(f"  Threshold:       {threshold}%")
    print(f"  App filter:      {', '.join(APP_ACTIVITY_TYPES)} ({len(APP_ACTIVITY_TYPES)} types, matching app default)")
    print(f"  Value filter:    standard_value > 0, units in {VALID_UNITS}")
    print("  Sources:         ChEMBL REST, RCSB PDB, PubChem")

    # Collect compounds
    compounds_to_run: List[Tuple[str, str]] = []

    if args.smiles:
        compounds_to_run.append((args.name or "Custom", args.smiles))
    if args.inchikey:
        name = args.name or args.inchikey
        print(f"\n  Resolving InChIKey: {args.inchikey}...", end="", flush=True)
        smiles = resolve_inchikey_to_smiles(args.inchikey)
        if smiles:
            print(f" -> {smiles[:60]}")
            compounds_to_run.append((name, smiles))
        else:
            print(" FAILED")
    if args.inchi:
        name = args.name or "InChI compound"
        print(f"\n  Resolving InChI: {args.inchi[:60]}...", end="", flush=True)
        smiles = resolve_inchi_to_smiles(args.inchi)
        if smiles:
            print(f" -> {smiles[:60]}")
            compounds_to_run.append((name, smiles))
        else:
            print(" FAILED")

    # Built-in compounds
    default_compounds = ["ursolic", "caffeine", "kaempferol", "quercetin"]
    compound_keys = args.compound if args.compound else (default_compounds if not compounds_to_run else [])
    for key in compound_keys:
        k = key.lower()
        if k not in COMPOUNDS:
            print(f"\n  [WARN] Unknown '{key}'. Available: {', '.join(COMPOUNDS.keys())}")
            continue
        info = COMPOUNDS[k]
        if info.get("smiles"):
            compounds_to_run.append((info["name"], info["smiles"]))
        elif info.get("inchikey"):
            print(f"\n  Resolving {info['name']} InChIKey: {info['inchikey']}...", end="", flush=True)
            smiles = resolve_inchikey_to_smiles(info["inchikey"])
            if smiles:
                print(f" -> {smiles[:60]}")
                compounds_to_run.append((info["name"], smiles))
            else:
                print(" FAILED")
        elif info.get("inchi"):
            print(f"\n  Resolving {info['name']} InChI...", end="", flush=True)
            smiles = resolve_inchi_to_smiles(info["inchi"])
            if smiles:
                print(f" -> {smiles[:60]}")
                compounds_to_run.append((info["name"], smiles))
            else:
                print(" FAILED")

    if not compounds_to_run:
        print("\n  No compounds. Use --compound, --smiles, --inchikey, or --inchi.")
        return

    all_results = []
    all_json = {}
    for name, smiles in compounds_to_run:
        print_header(f"{name}  |  SMILES: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
        result = run_compound(name, smiles, threshold)
        all_results.append(result)
        all_json[name] = {
            "smiles": smiles,
            "threshold": threshold,
            "rest": {
                "similar_compounds": len(result.rest_compounds),
                "activities_raw": result.rest_counts.raw_total,
                "activities_type_filtered": result.rest_counts.type_filtered,
                "activities_app_filtered": result.rest_counts.value_filtered,
                "activity_by_type": result.rest_counts.by_type_filtered,
                "unique_targets": result.rest_counts.unique_targets,
            },
            "pubchem": {
                "similar_cids": len(result.pubchem_cids),
                "total_assays": result.pubchem_total_assays,
                "active_all_types": result.pubchem_active_assays,
                "type_filtered_total": result.pubchem_type_filtered_total,
                "type_filtered_active": result.pubchem_type_filtered_active,
                "by_type_filtered": result.pubchem_by_type_filtered,
                "by_type_filtered_active": result.pubchem_by_type_filtered_active,
                "unique_targets": result.pubchem_unique_targets,
                "response_kb": round(result.pubchem_response_kb, 1),
                "similarity_time_ms": round(result.sim_pubchem_ms, 1),
                "bioactivity_time_ms": round(result.pubchem_bio_time_ms, 1),
                "indications": result.pubchem_indications,
                "indication_by_phase": result.pubchem_indication_by_phase,
                "indication_names": result.pubchem_indication_details[:20],
            },
            "pdb_entries": result.pdb_entries,
            "drug_indications": result.drug_indications,
            "indication_names": result.indication_details,
        }

    if args.json:
        print("\n\n--- JSON Output ---")
        print(json.dumps(all_json, indent=2))

    print()


if __name__ == "__main__":
    main()
