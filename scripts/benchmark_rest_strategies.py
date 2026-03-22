#!/usr/bin/env python3
"""
Benchmark ChEMBL REST API strategies for Phase 19.1 decision-making.

Tests:
  Q1: Sequential vs parallel pagination for large activity sets
  Q2: GET vs POST requests (if supported)
  Q4: "Fetch all + filter locally" vs "split by activity type + parallel"

Usage:
  python scripts/benchmark_rest_strategies.py
"""

import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote as url_quote

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
LIMIT = 1000

# Test compounds with known high activity counts
# Aspirin (CHEMBL25) - moderate activities
# Quercetin (CHEMBL159) - high activities
# Ibuprofen (CHEMBL521) - moderate activities
TEST_COMPOUNDS = {
    "quercetin": ["CHEMBL159"],
    "aspirin": ["CHEMBL25"],
    "caffeine": ["CHEMBL113"],
}

# For batch tests, use a set of compounds from a real similarity search
BATCH_IDS = [
    "CHEMBL159", "CHEMBL25", "CHEMBL113", "CHEMBL521",
    "CHEMBL1171837", "CHEMBL428647", "CHEMBL288304", "CHEMBL1213063",
    "CHEMBL1489", "CHEMBL98", "CHEMBL1201087", "CHEMBL553025",
    "CHEMBL325041", "CHEMBL569998", "CHEMBL576155", "CHEMBL1276308",
]

ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50", "AC50", "GI50", "MIC"]


def timed(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = func(*args, **kwargs)
        elapsed = time.monotonic() - start
        return result, elapsed
    return wrapper


def rest_get(endpoint: str, params: dict, timeout: int = 60) -> Optional[dict]:
    """Make a GET request to ChEMBL REST API."""
    url = f"{BASE_URL}/{endpoint}.json"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def rest_post(endpoint: str, data: dict, timeout: int = 60) -> Optional[dict]:
    """Make a POST request to ChEMBL REST API."""
    url = f"{BASE_URL}/{endpoint}.json"
    try:
        resp = requests.post(url, data=data, timeout=timeout,
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
        if resp.status_code == 405:
            return {"error": "Method Not Allowed", "status": 405}
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        return {"error": str(e), "status": getattr(e.response, 'status_code', None)}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Q1: Sequential vs Parallel Pagination
# ============================================================================

def fetch_activities_sequential(chembl_ids: List[str]) -> Tuple[int, int]:
    """Fetch all activities sequentially, page by page."""
    ids_param = ",".join(chembl_ids)
    all_activities = []
    offset = 0
    request_count = 0
    total_count = None

    while True:
        params = {
            "molecule_chembl_id__in": ids_param,
            "only": "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id",
            "limit": LIMIT,
            "offset": offset,
        }
        data = rest_get("activity", params)
        request_count += 1

        if data is None:
            break

        if total_count is None:
            total_count = data.get("page_meta", {}).get("total_count", 0)

        activities = data.get("activities", [])
        if not activities:
            break

        all_activities.extend(activities)

        if len(activities) < LIMIT:
            break
        offset += LIMIT

    return len(all_activities), request_count, total_count or 0


def fetch_activities_parallel(chembl_ids: List[str], max_workers: int = 4) -> Tuple[int, int]:
    """Fetch first page to get total_count, then parallelize remaining pages."""
    ids_param = ",".join(chembl_ids)

    # First page (sequential - need total_count)
    params = {
        "molecule_chembl_id__in": ids_param,
        "only": "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id",
        "limit": LIMIT,
        "offset": 0,
    }
    first_page = rest_get("activity", params)
    request_count = 1

    if first_page is None:
        return 0, 1, 0

    total_count = first_page.get("page_meta", {}).get("total_count", 0)
    first_activities = first_page.get("activities", [])
    all_activities = list(first_activities)

    if len(first_activities) < LIMIT or total_count <= LIMIT:
        return len(all_activities), request_count, total_count

    # Calculate remaining pages
    remaining_offsets = list(range(LIMIT, total_count, LIMIT))

    def fetch_page(offset):
        p = {
            "molecule_chembl_id__in": ids_param,
            "only": "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id",
            "limit": LIMIT,
            "offset": offset,
        }
        return rest_get("activity", p)

    # Parallel fetch remaining pages
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_page, off): off for off in remaining_offsets}
        for future in as_completed(futures):
            request_count += 1
            result = future.result()
            if result:
                activities = result.get("activities", [])
                all_activities.extend(activities)

    return len(all_activities), request_count, total_count


# ============================================================================
# Q2: GET vs POST
# ============================================================================

def test_post_support():
    """Test if ChEMBL REST API supports POST requests."""
    print("\n" + "=" * 70)
    print("Q2: GET vs POST Support Test")
    print("=" * 70)

    endpoints_to_test = [
        ("activity", {"molecule_chembl_id__in": "CHEMBL25", "limit": 5}),
        ("molecule", {"molecule_chembl_id": "CHEMBL25", "limit": 5}),
        ("similarity", {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "similarity": 70, "limit": 5}),
    ]

    for endpoint, params in endpoints_to_test:
        print(f"\n--- {endpoint} endpoint ---")

        # GET
        start = time.monotonic()
        get_result = rest_get(endpoint, params, timeout=30)
        get_time = time.monotonic() - start

        if get_result:
            key = next((k for k in ["activities", "molecules", "targets"] if k in get_result), None)
            get_count = len(get_result.get(key, [])) if key else "?"
            print(f"  GET:  {get_time:.3f}s — {get_count} results")
        else:
            print(f"  GET:  FAILED")

        # POST with same params
        start = time.monotonic()
        post_result = rest_post(endpoint, params, timeout=30)
        post_time = time.monotonic() - start

        if post_result and "error" not in post_result:
            key = next((k for k in ["activities", "molecules", "targets"] if k in post_result), None)
            post_count = len(post_result.get(key, [])) if key else "?"
            print(f"  POST: {post_time:.3f}s — {post_count} results")
        else:
            error = post_result.get("error", "Unknown") if post_result else "No response"
            status = post_result.get("status", "?") if post_result else "?"
            print(f"  POST: Status {status} — {error}")

    # Test POST with large ID list (the real use case)
    print(f"\n--- Large ID list test (POST vs GET with {len(BATCH_IDS)} IDs) ---")
    ids_str = ",".join(BATCH_IDS)

    start = time.monotonic()
    get_result = rest_get("activity", {"molecule_chembl_id__in": ids_str, "limit": 5}, timeout=30)
    get_time = time.monotonic() - start
    if get_result and "activities" in get_result:
        total = get_result.get("page_meta", {}).get("total_count", "?")
        print(f"  GET:  {get_time:.3f}s — total_count={total}")

    start = time.monotonic()
    post_result = rest_post("activity", {"molecule_chembl_id__in": ids_str, "limit": 5}, timeout=30)
    post_time = time.monotonic() - start
    if post_result and "error" not in post_result:
        total = post_result.get("page_meta", {}).get("total_count", "?")
        print(f"  POST: {post_time:.3f}s — total_count={total}")
    else:
        error = post_result.get("error", "Unknown") if post_result else "No response"
        status = post_result.get("status", "?") if post_result else "?"
        print(f"  POST: Status {status} — {error}")


# ============================================================================
# Q4: "Fetch all + filter locally" vs "Split by type + parallel"
# ============================================================================

def fetch_all_filter_locally(chembl_ids: List[str], activity_types: List[str]) -> Tuple[int, int, int]:
    """Strategy A: Fetch ALL activities, filter by type locally."""
    ids_param = ",".join(chembl_ids)
    all_activities = []
    offset = 0
    request_count = 0

    while True:
        params = {
            "molecule_chembl_id__in": ids_param,
            "only": "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id",
            "limit": LIMIT,
            "offset": offset,
        }
        data = rest_get("activity", params)
        request_count += 1

        if data is None:
            break

        activities = data.get("activities", [])
        if not activities:
            break

        all_activities.extend(activities)

        if len(activities) < LIMIT:
            break
        offset += LIMIT

    types_set = set(activity_types)
    filtered = [a for a in all_activities if a.get("standard_type") in types_set]
    return len(filtered), len(all_activities), request_count


def fetch_by_type_parallel(chembl_ids: List[str], activity_types: List[str], max_workers: int = 4) -> Tuple[int, int, int]:
    """Strategy B: One query per activity type, run in parallel."""
    ids_param = ",".join(chembl_ids)
    all_filtered = []
    total_raw = 0
    total_requests = 0

    def fetch_type(atype):
        activities = []
        offset = 0
        reqs = 0

        while True:
            params = {
                "molecule_chembl_id__in": ids_param,
                "standard_type": atype,
                "only": "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id",
                "limit": LIMIT,
                "offset": offset,
            }
            data = rest_get("activity", params)
            reqs += 1

            if data is None:
                break

            acts = data.get("activities", [])
            if not acts:
                break

            activities.extend(acts)

            if len(acts) < LIMIT:
                break
            offset += LIMIT

        return activities, reqs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_type, t): t for t in activity_types}
        for future in as_completed(futures):
            acts, reqs = future.result()
            all_filtered.extend(acts)
            total_raw += len(acts)  # Already filtered server-side
            total_requests += reqs

    return len(all_filtered), total_raw, total_requests


def fetch_all_filter_locally_parallel_pages(chembl_ids: List[str], activity_types: List[str], max_workers: int = 4) -> Tuple[int, int, int]:
    """Strategy C: Fetch ALL activities with parallel pagination, filter locally."""
    ids_param = ",".join(chembl_ids)

    # First page to get total
    params = {
        "molecule_chembl_id__in": ids_param,
        "only": "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id",
        "limit": LIMIT,
        "offset": 0,
    }
    first = rest_get("activity", params)
    request_count = 1

    if first is None:
        return 0, 0, 1

    total_count = first.get("page_meta", {}).get("total_count", 0)
    all_activities = list(first.get("activities", []))

    if len(all_activities) < LIMIT:
        types_set = set(activity_types)
        filtered = [a for a in all_activities if a.get("standard_type") in types_set]
        return len(filtered), len(all_activities), request_count

    remaining_offsets = list(range(LIMIT, total_count, LIMIT))

    def fetch_page(offset):
        p = {
            "molecule_chembl_id__in": ids_param,
            "only": "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id",
            "limit": LIMIT,
            "offset": offset,
        }
        return rest_get("activity", p)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_page, off): off for off in remaining_offsets}
        for future in as_completed(futures):
            request_count += 1
            result = future.result()
            if result:
                all_activities.extend(result.get("activities", []))

    types_set = set(activity_types)
    filtered = [a for a in all_activities if a.get("standard_type") in types_set]
    return len(filtered), len(all_activities), request_count


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("ChEMBL REST API Strategy Benchmarks")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Q1: Sequential vs Parallel Pagination
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q1: Sequential vs Parallel Pagination (Activity Fetch)")
    print("=" * 70)

    # Use quercetin - known to have lots of activities
    test_ids = ["CHEMBL159"]

    print(f"\nTest compound: Quercetin (CHEMBL159)")

    # First, check how many activities exist
    probe = rest_get("activity", {
        "molecule_chembl_id__in": ",".join(test_ids),
        "limit": 1,
    })
    if probe:
        total = probe.get("page_meta", {}).get("total_count", 0)
        print(f"Total activities: {total} (will need {(total + LIMIT - 1) // LIMIT} pages)")

    print(f"\n--- Sequential pagination ---")
    start = time.monotonic()
    seq_count, seq_reqs, seq_total = fetch_activities_sequential(test_ids)
    seq_time = time.monotonic() - start
    print(f"  Time: {seq_time:.3f}s | Activities: {seq_count} | Requests: {seq_reqs} | Total: {seq_total}")

    time.sleep(1)  # Avoid rate limiting

    print(f"\n--- Parallel pagination (4 workers) ---")
    start = time.monotonic()
    par_count, par_reqs, par_total = fetch_activities_parallel(test_ids, max_workers=4)
    par_time = time.monotonic() - start
    print(f"  Time: {par_time:.3f}s | Activities: {par_count} | Requests: {par_reqs} | Total: {par_total}")

    if seq_time > 0:
        print(f"\n  Speedup: {seq_time / par_time:.2f}x {'faster' if par_time < seq_time else 'slower'}")
        print(f"  Data integrity: {'MATCH' if seq_count == par_count else f'MISMATCH seq={seq_count} par={par_count}'}")

    # Test with batch of compounds (more realistic)
    print(f"\n\nTest: Batch of {len(BATCH_IDS)} compounds")

    probe = rest_get("activity", {
        "molecule_chembl_id__in": ",".join(BATCH_IDS),
        "limit": 1,
    })
    if probe:
        total = probe.get("page_meta", {}).get("total_count", 0)
        print(f"Total activities: {total} (will need {(total + LIMIT - 1) // LIMIT} pages)")

    time.sleep(1)

    print(f"\n--- Sequential ---")
    start = time.monotonic()
    seq_count, seq_reqs, seq_total = fetch_activities_sequential(BATCH_IDS)
    seq_time = time.monotonic() - start
    print(f"  Time: {seq_time:.3f}s | Activities: {seq_count} | Requests: {seq_reqs} | Total: {seq_total}")

    time.sleep(1)

    print(f"\n--- Parallel (4 workers) ---")
    start = time.monotonic()
    par_count, par_reqs, par_total = fetch_activities_parallel(BATCH_IDS, max_workers=4)
    par_time = time.monotonic() - start
    print(f"  Time: {par_time:.3f}s | Activities: {par_count} | Requests: {par_reqs} | Total: {par_total}")

    if seq_time > 0:
        print(f"\n  Speedup: {seq_time / par_time:.2f}x {'faster' if par_time < seq_time else 'slower'}")
        print(f"  Data integrity: {'MATCH' if seq_count == par_count else f'MISMATCH seq={seq_count} par={par_count}'}")

    # ------------------------------------------------------------------
    # Q2: POST support test
    # ------------------------------------------------------------------
    test_post_support()

    # ------------------------------------------------------------------
    # Q4: Fetch strategy comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q4: Activity Fetch Strategy Comparison")
    print("=" * 70)

    test_batch = BATCH_IDS[:8]  # Use 8 compounds for manageable test
    print(f"\nTest: {len(test_batch)} compounds, {len(ACTIVITY_TYPES)} activity types")

    time.sleep(1)

    print(f"\n--- Strategy A: Fetch ALL + filter locally (sequential pages) ---")
    start = time.monotonic()
    a_filtered, a_raw, a_reqs = fetch_all_filter_locally(test_batch, ACTIVITY_TYPES)
    a_time = time.monotonic() - start
    print(f"  Time: {a_time:.3f}s | Filtered: {a_filtered}/{a_raw} | Requests: {a_reqs}")

    time.sleep(1)

    print(f"\n--- Strategy B: Split by type + parallel (server-side filter) ---")
    start = time.monotonic()
    b_filtered, b_raw, b_reqs = fetch_by_type_parallel(test_batch, ACTIVITY_TYPES, max_workers=4)
    b_time = time.monotonic() - start
    print(f"  Time: {b_time:.3f}s | Filtered: {b_filtered}/{b_raw} | Requests: {b_reqs}")

    time.sleep(1)

    print(f"\n--- Strategy C: Fetch ALL + parallel pages + filter locally ---")
    start = time.monotonic()
    c_filtered, c_raw, c_reqs = fetch_all_filter_locally_parallel_pages(test_batch, ACTIVITY_TYPES, max_workers=4)
    c_time = time.monotonic() - start
    print(f"  Time: {c_time:.3f}s | Filtered: {c_filtered}/{c_raw} | Requests: {c_reqs}")

    print(f"\n--- Summary ---")
    print(f"  Strategy A (all + local filter, sequential): {a_time:.3f}s, {a_reqs} requests, {a_filtered} results")
    print(f"  Strategy B (per-type parallel, server filter): {b_time:.3f}s, {b_reqs} requests, {b_filtered} results")
    print(f"  Strategy C (all + local filter, parallel pages): {c_time:.3f}s, {c_reqs} requests, {c_filtered} results")

    fastest = min(a_time, b_time, c_time)
    winner = "A" if fastest == a_time else ("B" if fastest == b_time else "C")
    print(f"\n  Winner: Strategy {winner}")
    print(f"  Data integrity: A={a_filtered}, B={b_filtered}, C={c_filtered} " +
          f"{'ALL MATCH' if a_filtered == b_filtered == c_filtered else 'MISMATCH - investigate!'}")

    print("\n" + "=" * 70)
    print("DONE — All benchmarks complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
