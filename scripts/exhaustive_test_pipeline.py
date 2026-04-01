"""Pipeline-focused test: Submit compounds that exercise the full processing pipeline.

Since common compounds (Aspirin, Caffeine) already exist in the DB, this test uses
the resolve-duplicate endpoint with action="duplicate" to force processing.
It then verifies ALL result_summary fields.
"""
import asyncio
import sys
import time
import uuid

import httpx

BASE = "http://localhost:8001"
API = f"{BASE}/api/v1"

PASS = 0
FAIL = 0
BUGS = []


def log_result(test_name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  PASS  {test_name}")
    else:
        FAIL += 1
        msg = f"  FAIL  {test_name}: {detail}"
        print(msg)
        BUGS.append(msg)


async def poll_job(client: httpx.AsyncClient, job_id: str, session_id: str,
                   timeout: float = 600, interval: float = 3.0) -> dict:
    """Poll a job until terminal state."""
    start = time.time()
    while time.time() - start < timeout:
        r = await client.get(f"{API}/jobs/{job_id}", headers={"X-Session-ID": session_id})
        if r.status_code != 200:
            await asyncio.sleep(interval)
            continue
        data = r.json()
        status = data.get("status", "")
        progress = data.get("progress", 0)
        step = data.get("current_step", "")
        elapsed = int(time.time() - start)
        if elapsed % 15 < interval:
            print(f"    [{elapsed}s] {status} {progress}% - {step}")
        if status in ("completed", "failed", "cancelled", "pending_upload"):
            return data
        await asyncio.sleep(interval)
    return {"status": "timeout"}


async def submit_with_duplicate_bypass(client, name, smiles, threshold, session_id):
    """Submit a compound, handling duplicates by choosing 'duplicate' action."""
    headers = {"X-Session-ID": session_id}

    r = await client.post(f"{API}/jobs", json={
        "compound_name": name,
        "author_name": "Pipeline Tester",
        "smiles": smiles,
        "similarity_threshold": threshold,
    }, headers=headers)

    if r.status_code == 201:
        # No duplicate
        return r.json()["id"]

    if r.status_code == 200:
        # Duplicate detected - resolve with "duplicate" action
        dup_data = r.json()
        existing_entry_id = dup_data.get("existing_compound", {}).get("entry_id")
        suggested_name = dup_data.get("suggested_name", f"{name}_dup")

        r2 = await client.post(f"{API}/jobs/resolve-duplicate", json={
            "action": "duplicate",
            "compound_name": suggested_name,
            "author_name": "Pipeline Tester",
            "smiles": smiles,
            "similarity_threshold": threshold,
            "existing_entry_id": str(existing_entry_id) if existing_entry_id else None,
        }, headers=headers)

        if r2.status_code in (200, 201):
            data = r2.json()
            return data.get("id")
        else:
            print(f"    resolve-duplicate failed: {r2.status_code} {r2.text[:200]}")
            return None

    print(f"    submit failed: {r.status_code} {r.text[:200]}")
    return None


async def test_full_pipeline(client, name, smiles, threshold, session_id,
                              expect_similar_min=1, expect_activities_min=10):
    """Submit, wait, and verify all fields for a compound."""
    print(f"\n  --- {name} ({threshold}%) ---")

    job_id = await submit_with_duplicate_bypass(client, name, smiles, threshold, session_id)
    log_result(f"{name}: submitted", job_id is not None, f"job_id={job_id}")
    if not job_id:
        return None

    # Poll
    print(f"    Polling {name} (job {str(job_id)[:8]}...)...")
    result = await poll_job(client, job_id, session_id, timeout=600)
    status = result.get("status", "unknown")
    log_result(f"{name}: terminal state", status in ("completed", "pending_upload", "failed"),
               f"status={status}, error={result.get('error_message', 'none')}")

    if status == "failed":
        log_result(f"{name}: pipeline succeeded", False, f"FAILED: {result.get('error_message')}")
        return result

    if status not in ("completed", "pending_upload"):
        return result

    # Get detail
    headers = {"X-Session-ID": session_id}
    r = await client.get(f"{API}/jobs/{job_id}/detail", headers=headers)
    log_result(f"{name}: job detail fetch", r.status_code == 200, f"status={r.status_code}")
    if r.status_code != 200:
        return result

    detail = r.json()
    rs = detail.get("result_summary", {})

    # ---- Verify ALL result_summary fields ----

    # Core metadata
    log_result(f"{name}: schema_version", rs.get("schema_version") == 1, f"got {rs.get('schema_version')}")
    log_result(f"{name}: compound_name", rs.get("compound_name") is not None, f"got {rs.get('compound_name')}")
    log_result(f"{name}: author_name populated", rs.get("author_name") not in (None, "", "N/A"),
               f"got '{rs.get('author_name')}'")
    log_result(f"{name}: query_smiles", bool(rs.get("query_smiles")), f"got {str(rs.get('query_smiles'))[:30]}")
    log_result(f"{name}: similarity_threshold", rs.get("similarity_threshold") == threshold,
               f"expected {threshold}, got {rs.get('similarity_threshold')}")

    # Counts
    similar = rs.get("similar_count", 0) or 0
    total_activities = rs.get("total_activities", 0) or 0
    total_compounds = rs.get("total_compounds", 0) or 0
    log_result(f"{name}: similar_count >= {expect_similar_min}", similar >= expect_similar_min,
               f"got {similar}")
    log_result(f"{name}: total_activities >= {expect_activities_min}",
               total_activities >= expect_activities_min, f"got {total_activities}")
    log_result(f"{name}: total_compounds > 0", total_compounds > 0, f"got {total_compounds}")
    log_result(f"{name}: total_bioactivity_rows", rs.get("total_bioactivity_rows", 0) > 0,
               f"got {rs.get('total_bioactivity_rows')}")

    # IMP scoring
    imp = rs.get("imp_score")
    log_result(f"{name}: imp_score present", imp is not None, f"got {imp}")
    if imp is not None:
        log_result(f"{name}: imp_score in [0,1]", 0 <= float(imp) <= 1, f"got {imp}")

    # QED
    qed = rs.get("qed")
    log_result(f"{name}: qed present", qed is not None, f"got {qed}")
    if qed is not None:
        log_result(f"{name}: qed in [0,1]", 0 <= float(qed) <= 1, f"got {qed}")

    # PDB data
    pdb_count = rs.get("pdb_structures_count")
    pdb_unavailable = rs.get("pdb_unavailable")
    log_result(f"{name}: PDB data present", pdb_count is not None or pdb_unavailable is True,
               f"pdb_structures_count={pdb_count}, pdb_unavailable={pdb_unavailable}")

    # Drug indications
    drug_ind = rs.get("drug_indications_count")
    log_result(f"{name}: drug_indications_count field exists", drug_ind is not None,
               f"got {drug_ind}")

    # Classification
    class_avail = rs.get("classification_available")
    log_result(f"{name}: classification_available field exists", class_avail is not None,
               f"got {class_avail}")

    # Entry ID and storage
    log_result(f"{name}: entry_id present", bool(rs.get("entry_id")), f"got {rs.get('entry_id')}")
    log_result(f"{name}: storage_path present", bool(rs.get("storage_path")),
               f"got {rs.get('storage_path')}")

    # SMILES preserved
    log_result(f"{name}: smiles in result", bool(rs.get("smiles")),
               f"got {str(rs.get('smiles'))[:30]}")

    # Interference flags
    log_result(f"{name}: num_outliers field", "num_outliers" in rs, f"got {rs.get('num_outliers')}")

    # Check the compound also appears in the compounds list
    entry_id = rs.get("entry_id")
    if entry_id:
        r_compound = await client.get(f"{API}/compounds/{entry_id}", headers=headers)
        log_result(f"{name}: compound visible in DB", r_compound.status_code == 200,
                   f"status={r_compound.status_code}")
        if r_compound.status_code == 200:
            c_data = r_compound.json()
            log_result(f"{name}: compound.author_name", bool(c_data.get("author_name")),
                       f"got '{c_data.get('author_name')}'")
            log_result(f"{name}: compound.imp_score", c_data.get("imp_score") is not None,
                       f"got {c_data.get('imp_score')}")
            log_result(f"{name}: compound.similar_compounds", c_data.get("similar_compounds", 0) >= 0,
                       f"got {c_data.get('similar_compounds')}")

    return detail


async def main():
    print("=" * 70)
    print("PIPELINE-FOCUSED EXHAUSTIVE TEST")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=60, write=30, pool=30)) as client:
        # Quick connectivity check
        try:
            r = await client.get(f"{API}/health/live")
            if r.status_code != 200:
                print(f"FATAL: Backend not responding. status={r.status_code}")
                return
        except Exception as e:
            print(f"FATAL: Cannot connect: {e}")
            return

        session_id = str(uuid.uuid4())

        # Test 1: Aspirin at 90% -- well-studied NSAID, should have lots of data
        await test_full_pipeline(client, "Aspirin", "CC(=O)Oc1ccccc1C(=O)O", 90, session_id,
                                  expect_similar_min=3, expect_activities_min=100)

        # Test 2: Caffeine at 80% -- alkaloid
        await test_full_pipeline(client, "Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", 80, session_id,
                                  expect_similar_min=1, expect_activities_min=50)

        # Test 3: Metformin at 70% -- small molecule biguanide
        await test_full_pipeline(client, "Metformin", "CN(C)C(=N)NC(=N)N", 70, session_id,
                                  expect_similar_min=1, expect_activities_min=10)

        # Test 4: Doxorubicin at 70% -- complex anticancer
        await test_full_pipeline(client, "Doxorubicin",
                                  "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H]1O",
                                  70, session_id,
                                  expect_similar_min=1, expect_activities_min=10)

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} PASSED, {FAIL} FAILED")
    print("=" * 70)

    if BUGS:
        print("\nFAILURES:")
        for bug in BUGS:
            print(bug)

    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
