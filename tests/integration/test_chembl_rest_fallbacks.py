#!/usr/bin/env python
"""
Test ChEMBL REST API fallback functions with multiple compounds.

This script tests all ChEMBL API functions with REST API fallbacks:
1. Similarity search (get_chembl_ids)
2. Batch molecule data (fetch_batch_molecule_data)
3. Batch target names (fetch_batch_target_names)
4. Batch activities (fetch_all_activities_single_batch)
5. Batch drug indications (get_drug_indications_batch)

All functions should work with both library and REST API fallback.
"""

import sys
import time
import logging

from backend.modules.api_client import (
    # Main functions (library with REST fallback)
    get_chembl_ids,
    get_molecule_data,
    get_target_name,
    get_drug_indications,  # Single compound
    get_drug_indications_batch,
    fetch_batch_molecule_data,
    fetch_batch_target_names,
    fetch_all_activities_single_batch,
    batch_fetch_activities,  # Older batch function
    fetch_compound_activities,  # Single compound activities
    # Direct REST API functions
    rest_api_similarity_search,
    rest_api_fetch_molecule,
    rest_api_fetch_molecules_batch,
    rest_api_fetch_target,
    rest_api_fetch_targets_batch,
    rest_api_fetch_activities,
    rest_api_fetch_drug_indications_batch,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test data - compounds with varying similarity thresholds
TEST_COMPOUNDS = [
    {
        "name": "QUERCETIN",
        "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "inchikey": "REFJWTPEDVJJIY-UHFFFAOYSA-N",
        "similarity": 80,
    },
    {
        "name": "GENISTEIN",
        "smiles": "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12",
        "inchikey": "TZBJGXHYKVUXJN-UHFFFAOYSA-N",
        "similarity": 85,
    },
    {
        "name": "GOSSYPOL",
        "smiles": "Cc1cc2c(C(C)C)c(O)c(O)c(C=O)c2c(O)c1-c1c(C)cc2c(C(C)C)c(O)c(O)c(C=O)c2c1O",
        "inchikey": "NPOFMQLMZUHXFY-UHFFFAOYSA-N",
        "similarity": 70,
    },
    {
        "name": "CURCUMIN",
        "smiles": "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
        "inchikey": "VFLDPWHFBUODDF-FCXRPNKRSA-N",
        "similarity": 75,
    },
    {
        "name": "RUTIN",
        "smiles": "C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O",
        "inchikey": "IKGXIBQKXVIIEB-MHWRPJGESA-N",
        "similarity": 90,
    },
]

# Legacy test data for backward compatibility
TEST_SMILES = [compound["smiles"] for compound in TEST_COMPOUNDS]

TEST_CHEMBL_IDS = [
    "CHEMBL25",    # Aspirin
    "CHEMBL159",   # Quercetin
    "CHEMBL521",   # Ibuprofen
    "CHEMBL192",   # Metformin
    "CHEMBL411",   # Naproxen
    "CHEMBL3310837",  # Curcumin
    "CHEMBL1255",   # Rutin
]

TEST_TARGET_IDS = [
    "CHEMBL220",   # Acetylcholinesterase
    "CHEMBL204",   # Thrombin
    "CHEMBL205",   # Carbonic anhydrase II
    "CHEMBL251",   # Adenosine receptor A2a
    "CHEMBL2111389",  # Sodium channel
]


def check_chembl_status():
    """Check if ChEMBL API is available."""
    import requests
    try:
        response = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/status.json",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            logger.info(f"ChEMBL API status: {data.get('status', 'unknown')}")
            return True
        else:
            logger.error(f"ChEMBL API returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"ChEMBL API check failed: {e}")
        return False


def test_similarity_search():
    """Test similarity search with multiple compounds at different thresholds."""
    print("\n" + "="*70)
    print("TEST 1: Similarity Search (get_chembl_ids)")
    print("="*70)
    print("\nTesting each compound with its specific similarity threshold...")

    total_found = 0

    for compound in TEST_COMPOUNDS:
        name = compound["name"]
        smiles = compound["smiles"]
        similarity = compound["similarity"]

        print(f"\n--- {name} (similarity={similarity}%) ---")
        print(f"SMILES: {smiles[:60]}{'...' if len(smiles) > 60 else ''}")

        # Test main function (library + fallback)
        start = time.time()
        results = get_chembl_ids(smiles, similarity_threshold=similarity)
        elapsed = time.time() - start

        print(f"  Main function: {len(results)} compounds in {elapsed:.2f}s")
        if results:
            print(f"    First 3: {[r['ChEMBL ID'] for r in results[:3]]}")
            total_found += len(results)

        # Test REST API directly for first compound only (to save time)
        if compound == TEST_COMPOUNDS[0]:
            start = time.time()
            rest_results = rest_api_similarity_search(smiles, similarity_threshold=similarity)
            rest_elapsed = time.time() - start
            print(f"  REST API direct: {len(rest_results)} compounds in {rest_elapsed:.2f}s")

        if len(results) == 0:
            print(f"  WARNING: No results for {name}")

    print("\n--- Summary ---")
    print(f"Total similar compounds found across all queries: {total_found}")

    assert total_found > 0, "No similar compounds found across all queries"
    return True


def test_molecule_data():
    """Test molecule data fetch with multiple compounds."""
    print("\n" + "="*70)
    print("TEST 2: Molecule Data (single + batch)")
    print("="*70)

    # Test single molecule
    print("\nFetching single molecule data for CHEMBL25 (Aspirin)...")
    start = time.time()
    mol = get_molecule_data("CHEMBL25")
    elapsed = time.time() - start

    print("\nSingle molecule (library + REST fallback):")
    print(f"  Name: {mol.get('pref_name') if mol else 'None'}")
    print(f"  Time: {elapsed:.2f}s")

    # Test REST API directly
    start = time.time()
    rest_mol = rest_api_fetch_molecule("CHEMBL25")
    rest_elapsed = time.time() - start

    print("\nDirect REST API (single):")
    print(f"  Name: {rest_mol.get('pref_name') if rest_mol else 'None'}")
    print(f"  Time: {rest_elapsed:.2f}s")

    # Test batch molecule data
    print(f"\n--- Batch molecule fetch for {len(TEST_CHEMBL_IDS)} compounds ---")

    start = time.time()
    batch_results = fetch_batch_molecule_data(TEST_CHEMBL_IDS)
    batch_elapsed = time.time() - start

    print("\nBatch function (library + REST fallback):")
    print(f"  Found: {len(batch_results)}/{len(TEST_CHEMBL_IDS)} molecules")
    print(f"  Time: {batch_elapsed:.2f}s")
    for cid, mol in list(batch_results.items())[:3]:
        print(f"    {cid}: {mol.get('pref_name', 'N/A')}")

    # Test REST API batch directly
    start = time.time()
    rest_batch = rest_api_fetch_molecules_batch(TEST_CHEMBL_IDS)
    rest_batch_elapsed = time.time() - start

    print("\nDirect REST API (batch):")
    print(f"  Found: {len(rest_batch)}/{len(TEST_CHEMBL_IDS)} molecules")
    print(f"  Time: {rest_batch_elapsed:.2f}s")

    assert len(batch_results) > 0 or len(rest_batch) > 0, "No molecule data found"
    return True


def test_target_names():
    """Test target name fetch with multiple targets."""
    print("\n" + "="*70)
    print("TEST 3: Target Names (single + batch)")
    print("="*70)

    # Test single target
    target_id = TEST_TARGET_IDS[0]
    print(f"\nFetching single target name for {target_id}...")

    start = time.time()
    name = get_target_name(target_id)
    elapsed = time.time() - start

    print("\nSingle target (library + REST fallback):")
    print(f"  Name: {name}")
    print(f"  Time: {elapsed:.2f}s")

    # Test REST API directly
    start = time.time()
    rest_name = rest_api_fetch_target(target_id)
    rest_elapsed = time.time() - start

    print("\nDirect REST API (single):")
    print(f"  Name: {rest_name}")
    print(f"  Time: {rest_elapsed:.2f}s")

    # Test batch target names
    print(f"\n--- Batch target fetch for {len(TEST_TARGET_IDS)} targets ---")

    start = time.time()
    batch_results = fetch_batch_target_names(TEST_TARGET_IDS)
    batch_elapsed = time.time() - start

    print("\nBatch function (library + REST fallback):")
    print(f"  Found: {len(batch_results)}/{len(TEST_TARGET_IDS)} targets")
    print(f"  Time: {batch_elapsed:.2f}s")
    for tid, name in list(batch_results.items())[:3]:
        print(f"    {tid}: {name}")

    # Test REST API batch directly
    start = time.time()
    rest_batch = rest_api_fetch_targets_batch(TEST_TARGET_IDS)
    rest_batch_elapsed = time.time() - start

    print("\nDirect REST API (batch):")
    print(f"  Found: {len(rest_batch)}/{len(TEST_TARGET_IDS)} targets")
    print(f"  Time: {rest_batch_elapsed:.2f}s")

    assert len(batch_results) > 0 or len(rest_batch) > 0, "No target names found"
    return True


def test_activities():
    """Test activity fetch with multiple compounds."""
    print("\n" + "="*70)
    print("TEST 4: Activities (batch)")
    print("="*70)

    chembl_ids = TEST_CHEMBL_IDS[:3]  # Use fewer for speed
    print(f"\nFetching activities for {len(chembl_ids)} compounds: {chembl_ids}")

    # Test main batch function
    start = time.time()
    results = fetch_all_activities_single_batch(chembl_ids)
    elapsed = time.time() - start

    print("\nBatch function (library + REST fallback):")
    print(f"  Found: {len(results)} activities")
    print(f"  Time: {elapsed:.2f}s")

    # Count by compound
    by_compound = {}
    for act in results:
        cid = act.get('molecule_chembl_id')
        by_compound[cid] = by_compound.get(cid, 0) + 1
    for cid, count in list(by_compound.items())[:3]:
        print(f"    {cid}: {count} activities")

    # Test REST API directly
    start = time.time()
    rest_results = rest_api_fetch_activities(chembl_ids)
    rest_elapsed = time.time() - start

    print("\nDirect REST API:")
    print(f"  Found: {len(rest_results)} activities")
    print(f"  Time: {rest_elapsed:.2f}s")

    assert len(results) > 0 or len(rest_results) > 0, "No activities found"
    return True


def test_drug_indications():
    """Test drug indication fetch with multiple compounds."""
    print("\n" + "="*70)
    print("TEST 5: Drug Indications (batch)")
    print("="*70)

    # Use compounds likely to have drug indications
    chembl_ids = ["CHEMBL25", "CHEMBL521", "CHEMBL192"]  # Aspirin, Ibuprofen, Metformin
    print(f"\nFetching drug indications for {len(chembl_ids)} compounds: {chembl_ids}")

    # Test main batch function
    start = time.time()
    results = get_drug_indications_batch(chembl_ids)
    elapsed = time.time() - start

    total_indications = sum(len(v) for v in results.values())

    print("\nBatch function (REST API primary, library fallback):")
    print(f"  Compounds: {len(results)}")
    print(f"  Total indications: {total_indications}")
    print(f"  Time: {elapsed:.2f}s")
    for cid, indications in results.items():
        if indications:
            sample = indications[0]
            print(f"    {cid}: {len(indications)} indications (e.g., {sample.get('MESH_Heading', 'N/A')})")
        else:
            print(f"    {cid}: 0 indications")

    # Test REST API directly
    start = time.time()
    rest_results = rest_api_fetch_drug_indications_batch(chembl_ids)
    rest_elapsed = time.time() - start

    print("\nDirect REST API:")
    print(f"  Found: {len(rest_results)} raw indications")
    print(f"  Time: {rest_elapsed:.2f}s")

    assert total_indications > 0 or len(rest_results) > 0, "No drug indications found"
    return True


def test_single_drug_indications():
    """Test single compound drug indication fetch."""
    print("\n" + "="*70)
    print("TEST 6: Single Drug Indications (get_drug_indications)")
    print("="*70)

    # Test with Aspirin (known to have indications)
    chembl_id = "CHEMBL25"
    print(f"\nFetching drug indications for {chembl_id} (Aspirin)...")

    start = time.time()
    results = get_drug_indications(chembl_id)
    elapsed = time.time() - start

    print("\nSingle compound function:")
    print(f"  Found: {len(results)} indications")
    print(f"  Time: {elapsed:.2f}s")
    if results:
        for ind in list(results)[:3]:
            print(f"    - {ind.get('MESH_Heading', 'N/A')} (Phase {ind.get('Max_Phase', 'N/A')})")

    assert len(results) > 0, "No single drug indications found"
    return True


def test_single_compound_activities():
    """Test single compound activity fetch."""
    print("\n" + "="*70)
    print("TEST 7: Single Compound Activities (fetch_compound_activities)")
    print("="*70)

    chembl_id = "CHEMBL25"  # Aspirin
    print(f"\nFetching activities for {chembl_id} (Aspirin)...")

    start = time.time()
    results = fetch_compound_activities(chembl_id)
    elapsed = time.time() - start

    print("\nSingle compound function:")
    print(f"  Found: {len(results)} activities")
    print(f"  Time: {elapsed:.2f}s")

    # Count by activity type
    by_type = {}
    for act in results:
        act_type = act.get('standard_type', 'Unknown')
        by_type[act_type] = by_type.get(act_type, 0) + 1
    for act_type, count in list(by_type.items())[:5]:
        print(f"    {act_type}: {count} activities")

    assert len(results) > 0, "No single compound activities found"
    return True


def test_batch_fetch_activities_legacy():
    """Test legacy batch_fetch_activities function."""
    print("\n" + "="*70)
    print("TEST 8: Legacy Batch Activities (batch_fetch_activities)")
    print("="*70)

    chembl_ids = TEST_CHEMBL_IDS[:3]
    print(f"\nFetching activities for {len(chembl_ids)} compounds: {chembl_ids}")

    start = time.time()
    results = batch_fetch_activities(chembl_ids, batch_size=50)
    elapsed = time.time() - start

    print("\nLegacy batch function:")
    print(f"  Found: {len(results)} activities")
    print(f"  Time: {elapsed:.2f}s")

    # Count by compound
    by_compound = {}
    for act in results:
        cid = act.get('molecule_chembl_id')
        by_compound[cid] = by_compound.get(cid, 0) + 1
    for cid, count in list(by_compound.items())[:3]:
        print(f"    {cid}: {count} activities")

    assert len(results) > 0, "No legacy batch activities found"
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("ChEMBL REST API Fallback Tests")
    print("Testing all ChEMBL functions with multiple compounds")
    print("="*70)

    # Check API status
    if not check_chembl_status():
        print("\nWARNING: ChEMBL API may be unavailable. Tests may fail.")
        print("Continuing anyway to test fallback mechanisms...\n")

    results = {}

    # Run all tests
    try:
        results['similarity_search'] = test_similarity_search()
    except Exception as e:
        logger.error(f"Similarity search test failed: {e}")
        results['similarity_search'] = False

    try:
        results['molecule_data'] = test_molecule_data()
    except Exception as e:
        logger.error(f"Molecule data test failed: {e}")
        results['molecule_data'] = False

    try:
        results['target_names'] = test_target_names()
    except Exception as e:
        logger.error(f"Target names test failed: {e}")
        results['target_names'] = False

    try:
        results['activities'] = test_activities()
    except Exception as e:
        logger.error(f"Activities test failed: {e}")
        results['activities'] = False

    try:
        results['drug_indications'] = test_drug_indications()
    except Exception as e:
        logger.error(f"Drug indications test failed: {e}")
        results['drug_indications'] = False

    try:
        results['single_drug_indications'] = test_single_drug_indications()
    except Exception as e:
        logger.error(f"Single drug indications test failed: {e}")
        results['single_drug_indications'] = False

    try:
        results['single_compound_activities'] = test_single_compound_activities()
    except Exception as e:
        logger.error(f"Single compound activities test failed: {e}")
        results['single_compound_activities'] = False

    try:
        results['legacy_batch_activities'] = test_batch_fetch_activities_legacy()
    except Exception as e:
        logger.error(f"Legacy batch activities test failed: {e}")
        results['legacy_batch_activities'] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll ChEMBL REST API fallback tests passed!")
        return 0
    else:
        print("\nSome tests failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
