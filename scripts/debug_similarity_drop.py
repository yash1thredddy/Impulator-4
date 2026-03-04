"""
Debug script: Trace where similar compounds get dropped in the IMPULATOR pipeline.

Compares ChEMBL REST API raw results vs what our app pipeline retains at each stage.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json

SMILES = "O=C1O[C@H]([C@@H](O)CO)[C@@H](O)C1=O"
THRESHOLD = 40
ACTIVITY_TYPES = ['IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'GI50', 'MIC']
VALID_UNITS = {'nM', 'uM', 'mM', 'pM', 'M'}

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
session = requests.Session()

def step(n, label):
    print(f"\n{'='*70}")
    print(f"  STEP {n}: {label}")
    print(f"{'='*70}")

# ── Step 1: Raw similarity search ──
step(1, "ChEMBL Similarity Search (raw)")
url = f"{CHEMBL_BASE}/similarity/{SMILES}/{THRESHOLD}.json?limit=1000"
resp = session.get(url, timeout=120)
resp.raise_for_status()
data = resp.json()
sim_molecules = data.get("molecules", [])
sim_ids = [m["molecule_chembl_id"] for m in sim_molecules]
print(f"  Raw similar compounds from ChEMBL: {len(sim_ids)}")
for mid in sim_ids:
    print(f"    - {mid}")

# ── Step 2: Fetch activities for these IDs ──
step(2, "Fetch activities (all types)")
all_activities = []
# Chunk to 50 per request (matches our app MAX_IDS_PER_REQUEST)
chunk_size = 50
for i in range(0, len(sim_ids), chunk_size):
    chunk = sim_ids[i:i+chunk_size]
    id_filter = ";".join(chunk)
    offset = 0
    while True:
        act_url = f"{CHEMBL_BASE}/activity.json?molecule_chembl_id__in={id_filter}&limit=1000&offset={offset}"
        r = session.get(act_url, timeout=60)
        r.raise_for_status()
        page = r.json()
        activities = page.get("activities", [])
        all_activities.extend(activities)
        if not page.get("page_meta", {}).get("next"):
            break
        offset += 1000

print(f"  Total raw activities: {len(all_activities)}")

# Which compounds have ANY activity?
ids_with_any_activity = set(a["molecule_chembl_id"] for a in all_activities)
ids_without_activity = set(sim_ids) - ids_with_any_activity
print(f"  Compounds with activities: {len(ids_with_any_activity)}")
print(f"  Compounds with NO activities: {len(ids_without_activity)}")
if ids_without_activity:
    for mid in sorted(ids_without_activity):
        print(f"    DROPPED (no activities): {mid}")

# ── Step 3: Filter by activity type ──
step(3, f"Filter by activity types: {ACTIVITY_TYPES}")
type_filtered = [a for a in all_activities if a.get("standard_type") in ACTIVITY_TYPES]
print(f"  Activities after type filter: {len(type_filtered)} (from {len(all_activities)})")

ids_after_type = set(a["molecule_chembl_id"] for a in type_filtered)
dropped_by_type = ids_with_any_activity - ids_after_type
print(f"  Compounds with matching type: {len(ids_after_type)}")
if dropped_by_type:
    for mid in sorted(dropped_by_type):
        # Show what types this compound actually had
        types = set(a["standard_type"] for a in all_activities if a["molecule_chembl_id"] == mid)
        print(f"    DROPPED (wrong types only: {types}): {mid}")

# ── Step 4: Filter by value + units (app filter) ──
step(4, f"Filter by value > 0 + units in {VALID_UNITS}")
app_filtered = []
for a in type_filtered:
    val = a.get("standard_value")
    units = a.get("standard_units")
    if val is None:
        continue
    try:
        v = float(val)
        if v <= 0:
            continue
    except (ValueError, TypeError):
        continue
    if units not in VALID_UNITS:
        continue
    app_filtered.append(a)

print(f"  Activities after app filter: {len(app_filtered)} (from {len(type_filtered)})")

ids_after_app = set(a["molecule_chembl_id"] for a in app_filtered)
dropped_by_value = ids_after_type - ids_after_app
print(f"  Compounds surviving app filter: {len(ids_after_app)}")
if dropped_by_value:
    for mid in sorted(dropped_by_value):
        print(f"    DROPPED (no valid value/units): {mid}")

# ── Step 5: Molecule data fetch ──
step(5, "Fetch molecule data (batch)")
mol_ids_to_fetch = list(ids_after_app)
mol_cache = {}
for i in range(0, len(mol_ids_to_fetch), chunk_size):
    chunk = mol_ids_to_fetch[i:i+chunk_size]
    id_filter = ";".join(chunk)
    offset = 0
    while True:
        mol_url = f"{CHEMBL_BASE}/molecule.json?molecule_chembl_id__in={id_filter}&limit=1000&offset={offset}"
        r = session.get(mol_url, timeout=60)
        r.raise_for_status()
        page = r.json()
        for mol in page.get("molecules", []):
            mol_cache[mol["molecule_chembl_id"]] = mol
        if not page.get("page_meta", {}).get("next"):
            break
        offset += 1000

ids_with_mol_data = set(mol_cache.keys())
ids_missing_mol = ids_after_app - ids_with_mol_data
print(f"  Molecule data fetched: {len(ids_with_mol_data)}")
if ids_missing_mol:
    for mid in sorted(ids_missing_mol):
        print(f"    DROPPED (no molecule data): {mid}")

# ── Step 6: Final count (unique ChEMBL IDs in output) ──
step(6, "FINAL: Unique compounds in processed output")
final_ids = ids_after_app & ids_with_mol_data
print(f"  Final compound count: {len(final_ids)}")
print(f"\n  Pipeline summary:")
print(f"    Similarity search:  {len(sim_ids):>3} compounds")
print(f"    Have activities:    {len(ids_with_any_activity):>3} compounds  (dropped {len(ids_without_activity)})")
print(f"    Type filter:        {len(ids_after_type):>3} compounds  (dropped {len(dropped_by_type)})")
print(f"    Value/unit filter:  {len(ids_after_app):>3} compounds  (dropped {len(dropped_by_value)})")
print(f"    Molecule data:      {len(final_ids):>3} compounds  (dropped {len(ids_missing_mol)})")
print(f"\n  {'='*50}")
print(f"  ChEMBL web UI: 20  |  App output: {len(final_ids)}")
print(f"  {'='*50}")

# Show all dropped compounds with reasons
all_dropped = set(sim_ids) - final_ids
if all_dropped:
    print(f"\n  ALL DROPPED COMPOUNDS ({len(all_dropped)}):")
    for mid in sorted(all_dropped):
        if mid in ids_without_activity:
            reason = "no activities at all"
        elif mid in dropped_by_type:
            types = set(a["standard_type"] for a in all_activities if a["molecule_chembl_id"] == mid)
            reason = f"no matching activity types (had: {types})"
        elif mid in dropped_by_value:
            reason = "no valid value/units"
        elif mid in ids_missing_mol:
            reason = "molecule data fetch failed"
        else:
            reason = "unknown"
        print(f"    {mid}: {reason}")
