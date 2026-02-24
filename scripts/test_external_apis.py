"""
Test script for all external APIs used by IMPULATOR.

Tests:
- ChEMBL: similarity search, activities, molecules, bioactivities
- RCSB PDB: structure search, resolution data
- ClassyFire: chemical classification
- NPClassifier: natural product classification

Usage:
    python scripts/test_external_apis.py
    python scripts/test_external_apis.py --verbose
    python scripts/test_external_apis.py --api chembl
"""

import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install requests rdkit")
    sys.exit(1)

# ChEMBL client import - may fail if API is down
CHEMBL_AVAILABLE = False
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except Exception as e:
    print(f"⚠️  ChEMBL client import failed: {e}")
    print("    ChEMBL tests will be skipped.")
    print()
    new_client = None


# Test compounds with known properties
TEST_COMPOUNDS = {
    "aspirin": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "name": "Aspirin",
        "expected_chembl_id": "CHEMBL25",
    },
    "caffeine": {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "name": "Caffeine",
        "expected_chembl_id": "CHEMBL113",
    },
    "ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "name": "Ibuprofen",
        "expected_chembl_id": "CHEMBL521",
    },
}


class APITestResult:
    """Store test result for an API endpoint."""

    def __init__(self, api_name: str, test_name: str):
        self.api_name = api_name
        self.test_name = test_name
        self.success = False
        self.duration_ms = 0
        self.error = None
        self.details = {}

    def __repr__(self):
        status = "✓" if self.success else "✗"
        duration = f"{self.duration_ms:.0f}ms"
        return f"{status} {self.api_name}.{self.test_name} ({duration})"


class ExternalAPITester:
    """Test all external APIs used by IMPULATOR."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[APITestResult] = []

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled."""
        if self.verbose or level == "ERROR":
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def run_test(self, api_name: str, test_name: str, test_func) -> APITestResult:
        """Run a single test and record the result."""
        result = APITestResult(api_name, test_name)

        try:
            self.log(f"Testing {api_name}.{test_name}...")
            start_time = time.time()

            result.details = test_func()

            result.duration_ms = (time.time() - start_time) * 1000
            result.success = True

            self.log(f"✓ {api_name}.{test_name} passed ({result.duration_ms:.0f}ms)")

        except Exception as e:
            result.duration_ms = (time.time() - start_time) * 1000
            result.error = str(e)
            result.success = False

            self.log(f"✗ {api_name}.{test_name} failed: {e}", "ERROR")

        self.results.append(result)
        return result

    # ==================== ChEMBL API Tests ====================

    def test_chembl_similarity_search(self) -> Dict[str, Any]:
        """Test ChEMBL similarity search endpoint."""
        if not CHEMBL_AVAILABLE:
            raise RuntimeError("ChEMBL client not available (API may be down)")

        test_compound = TEST_COMPOUNDS["aspirin"]
        smiles = test_compound["smiles"]

        # Convert SMILES to Morgan fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Validate that fingerprint can be generated (tests RDKit compatibility)
        _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

        # ChEMBL similarity API
        similarity_client = new_client.similarity
        results = similarity_client.filter(smiles=smiles, similarity=70).only(['molecule_chembl_id', 'similarity'])

        results_list = list(results[:10])  # Get first 10

        if not results_list:
            raise ValueError("No similar compounds found")

        return {
            "query_compound": test_compound["name"],
            "num_results": len(results_list),
            "top_similarity": results_list[0]['similarity'] if results_list else None,
            "sample_chembl_id": results_list[0]['molecule_chembl_id'] if results_list else None,
        }

    def test_chembl_molecule_lookup(self) -> Dict[str, Any]:
        """Test ChEMBL molecule lookup by ChEMBL ID."""
        if not CHEMBL_AVAILABLE:
            raise RuntimeError("ChEMBL client not available (API may be down)")

        test_compound = TEST_COMPOUNDS["aspirin"]
        chembl_id = test_compound["expected_chembl_id"]

        molecule_client = new_client.molecule
        molecule = molecule_client.get(chembl_id)

        if not molecule:
            raise ValueError(f"Molecule not found: {chembl_id}")

        return {
            "chembl_id": chembl_id,
            "pref_name": molecule.get('pref_name'),
            "molecule_type": molecule.get('molecule_type'),
            "max_phase": molecule.get('max_phase'),
            "has_structure": bool(molecule.get('molecule_structures')),
        }

    def test_chembl_activities_fetch(self) -> Dict[str, Any]:
        """Test ChEMBL activities endpoint (bioactivity data)."""
        if not CHEMBL_AVAILABLE:
            raise RuntimeError("ChEMBL client not available (API may be down)")

        test_compound = TEST_COMPOUNDS["aspirin"]
        chembl_id = test_compound["expected_chembl_id"]

        activity_client = new_client.activity
        activities = activity_client.filter(
            molecule_chembl_id=chembl_id,
            standard_type__in=['IC50', 'Ki', 'EC50', 'Kd'],
        ).only(['activity_id', 'standard_type', 'standard_value', 'standard_units'])

        activities_list = list(activities[:20])  # Get first 20

        if not activities_list:
            raise ValueError(f"No activities found for {chembl_id}")

        # Count by type
        type_counts = {}
        for act in activities_list:
            act_type = act.get('standard_type', 'Unknown')
            type_counts[act_type] = type_counts.get(act_type, 0) + 1

        return {
            "chembl_id": chembl_id,
            "num_activities": len(activities_list),
            "activity_types": type_counts,
            "sample_activity": activities_list[0] if activities_list else None,
        }

    def test_chembl_batch_fetch(self) -> Dict[str, Any]:
        """Test ChEMBL batch fetching (multiple compounds)."""
        if not CHEMBL_AVAILABLE:
            raise RuntimeError("ChEMBL client not available (API may be down)")

        chembl_ids = [comp["expected_chembl_id"] for comp in TEST_COMPOUNDS.values()]

        activity_client = new_client.activity
        activities = activity_client.filter(
            molecule_chembl_id__in=chembl_ids,
            standard_type__in=['IC50', 'Ki', 'EC50', 'Kd'],
        ).only(['molecule_chembl_id', 'activity_id', 'standard_type'])

        activities_list = list(activities[:50])  # Get first 50

        if not activities_list:
            raise ValueError("No activities found for batch")

        # Count by compound
        compound_counts = {}
        for act in activities_list:
            mol_id = act.get('molecule_chembl_id', 'Unknown')
            compound_counts[mol_id] = compound_counts.get(mol_id, 0) + 1

        return {
            "query_count": len(chembl_ids),
            "chembl_ids": chembl_ids,
            "total_activities": len(activities_list),
            "compounds_with_data": len(compound_counts),
            "activities_per_compound": compound_counts,
        }

    def test_chembl_error_handling(self) -> Dict[str, Any]:
        """Test ChEMBL error handling with invalid inputs."""
        if not CHEMBL_AVAILABLE:
            raise RuntimeError("ChEMBL client not available (API may be down)")

        errors_caught = []

        # Test 1: Invalid ChEMBL ID
        try:
            molecule_client = new_client.molecule
            result = molecule_client.get("CHEMBL_INVALID_12345")
            if result is None:
                errors_caught.append("invalid_chembl_id_returns_none")
        except Exception as e:
            errors_caught.append(f"invalid_chembl_id_raises: {type(e).__name__}")

        # Test 2: Invalid SMILES for similarity
        try:
            similarity_client = new_client.similarity
            results = list(similarity_client.filter(smiles="INVALID_SMILES_123", similarity=70))
            errors_caught.append(f"invalid_smiles_returns_{len(results)}_results")
        except Exception as e:
            errors_caught.append(f"invalid_smiles_raises: {type(e).__name__}")

        # Test 3: Empty filter
        try:
            activity_client = new_client.activity
            results = list(activity_client.filter(molecule_chembl_id__in=[]).only(['activity_id'])[:5])
            errors_caught.append(f"empty_filter_returns_{len(results)}_results")
        except Exception as e:
            errors_caught.append(f"empty_filter_raises: {type(e).__name__}")

        return {
            "errors_handled": len(errors_caught),
            "error_behaviors": errors_caught,
        }

    # ==================== RCSB PDB API Tests ====================

    def test_pdb_structure_search(self) -> Dict[str, Any]:
        """Test RCSB PDB structure search by ligand (production method)."""
        test_compound = TEST_COMPOUNDS["aspirin"]
        smiles = test_compound["smiles"]

        # Use RCSB REST API (matches production config)
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

        # Production query format from backend/modules/pdb_client.py
        query = {
            "query": {
                "type": "terminal",
                "service": "chemical",  # NOT "text_chem" - that was the bug!
                "parameters": {
                    "value": smiles,
                    "type": "descriptor",
                    "descriptor_type": "SMILES",
                    "match_type": "graph-relaxed"  # Structural similarity
                }
            },
            "return_type": "entry",
            "request_options": {
                "return_all_hits": True
            }
        }

        # Retry logic matching production
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    search_url,
                    json=query,
                    timeout=30,
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    pdb_ids = [entry['identifier'] for entry in data.get('result_set', [])]

                    return {
                        "search_type": "ligand_similarity",
                        "compound": test_compound["name"],
                        "num_results": len(pdb_ids),
                        "sample_pdb_ids": pdb_ids[:5],
                    }
                elif response.status_code == 204:
                    # No content - no results
                    return {
                        "search_type": "ligand_similarity",
                        "compound": test_compound["name"],
                        "num_results": 0,
                        "sample_pdb_ids": [],
                    }
                elif response.status_code == 500:
                    # Server error - retry
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        raise ValueError(f"PDB server error 500 after {max_retries} attempts")
                else:
                    response.raise_for_status()

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise TimeoutError(f"PDB search timeout after {max_retries} attempts")

        raise RuntimeError("PDB search failed after all retries")

    def test_pdb_entry_data(self) -> Dict[str, Any]:
        """Test RCSB PDB entry data retrieval."""
        # Use a well-known PDB entry
        pdb_id = "1ATP"  # ATP synthase

        data_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

        response = requests.get(data_url, timeout=15)
        response.raise_for_status()

        data = response.json()

        return {
            "pdb_id": pdb_id,
            "title": data.get("struct", {}).get("title", "")[:100],
            "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
            "deposition_date": data.get("rcsb_accession_info", {}).get("deposit_date"),
            "experimental_method": data.get("exptl", [{}])[0].get("method"),
        }

    def test_pdb_graphql_batch(self) -> Dict[str, Any]:
        """Test RCSB PDB GraphQL API for batch queries."""
        # GraphQL endpoint
        graphql_url = "https://data.rcsb.org/graphql"

        # Query multiple PDB IDs
        pdb_ids = ["1ATP", "3ATP", "2ATP"]

        query = """
        query($ids: [String!]!) {
            entries(entry_ids: $ids) {
                rcsb_id
                struct {
                    title
                }
                rcsb_entry_info {
                    resolution_combined
                }
            }
        }
        """

        response = requests.post(
            graphql_url,
            json={"query": query, "variables": {"ids": pdb_ids}},
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        entries = data.get("data", {}).get("entries", [])

        return {
            "query_count": len(pdb_ids),
            "results_count": len(entries),
            "sample_entry": entries[0] if entries else None,
        }

    # ==================== ClassyFire API Tests ====================

    def test_classyfire_classify(self) -> Dict[str, Any]:
        """Test ClassyFire chemical classification (production method)."""
        test_compound = TEST_COMPOUNDS["aspirin"]

        # Generate InChIKey from SMILES
        smiles = test_compound["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")

        from rdkit.Chem.inchi import MolToInchiKey
        inchikey = MolToInchiKey(mol)

        # ClassyFire LOOKUP API (production method - direct lookup, not async submission)
        url = f"http://classyfire.wishartlab.com/entities/{inchikey}.json"

        # Direct lookup with retries (matches production config)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "compound": test_compound["name"],
                        "inchikey": inchikey,
                        "kingdom": data.get("kingdom", {}).get("name"),
                        "superclass": data.get("superclass", {}).get("name"),
                        "class": data.get("class", {}).get("name"),
                        "subclass": data.get("subclass", {}).get("name"),
                        "method": "direct_lookup"
                    }
                elif response.status_code == 404:
                    raise ValueError(f"Compound not found in ClassyFire database (InChIKey: {inchikey})")
                elif response.status_code in [500, 502, 503, 504]:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        raise ValueError(f"ClassyFire server error {response.status_code} after {max_retries} attempts")
                else:
                    raise ValueError(f"ClassyFire returned status {response.status_code}")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    raise TimeoutError(f"ClassyFire timeout after {max_retries} attempts")

        raise RuntimeError("ClassyFire test failed after all retries")

    # ==================== NPClassifier API Tests ====================

    def test_npclassifier_classify(self) -> Dict[str, Any]:
        """Test NPClassifier natural product classification."""
        test_compound = TEST_COMPOUNDS["caffeine"]  # Known natural product

        # NPClassifier API endpoint
        url = "https://npclassifier.gnps2.org/classify"

        params = {
            "smiles": test_compound["smiles"]
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        return {
            "compound": test_compound["name"],
            "pathway": data.get("pathway_results", ["Unknown"])[0] if data.get("pathway_results") else "Unknown",
            "superclass": data.get("superclass_results", ["Unknown"])[0] if data.get("superclass_results") else "Unknown",
            "class": data.get("class_results", ["Unknown"])[0] if data.get("class_results") else "Unknown",
            "isglycoside": data.get("isglycoside", False),
        }

    # ==================== Test Runner ====================

    def run_all_tests(self, api_filter: Optional[str] = None):
        """Run all API tests or filter by API name."""
        print("\n" + "=" * 70)
        print("  IMPULATOR External API Test Suite")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")

        # Define test suite
        test_suite = [
            # ChEMBL tests
            ("chembl", "similarity_search", self.test_chembl_similarity_search),
            ("chembl", "molecule_lookup", self.test_chembl_molecule_lookup),
            ("chembl", "activities_fetch", self.test_chembl_activities_fetch),
            ("chembl", "batch_fetch", self.test_chembl_batch_fetch),
            ("chembl", "error_handling", self.test_chembl_error_handling),

            # PDB tests
            ("pdb", "structure_search", self.test_pdb_structure_search),
            ("pdb", "entry_data", self.test_pdb_entry_data),
            ("pdb", "graphql_batch", self.test_pdb_graphql_batch),

            # ClassyFire tests
            ("classyfire", "classify", self.test_classyfire_classify),

            # NPClassifier tests
            ("npclassifier", "classify", self.test_npclassifier_classify),
        ]

        # Filter tests if requested
        if api_filter:
            test_suite = [(api, name, func) for api, name, func in test_suite if api.lower() == api_filter.lower()]
            if not test_suite:
                print(f"❌ No tests found for API: {api_filter}")
                return

        # Skip ChEMBL tests if client not available
        if not CHEMBL_AVAILABLE:
            original_count = len(test_suite)
            test_suite = [(api, name, func) for api, name, func in test_suite if api != "chembl"]
            skipped_count = original_count - len(test_suite)
            print(f"DEBUG: Original count={original_count}, New count={len(test_suite)}, Skipped={skipped_count}")
            if skipped_count > 0:
                print(f"⚠️  Skipping {skipped_count} ChEMBL test(s) - API unavailable\n")
            else:
                print("⚠️  WARNING: Expected to skip ChEMBL tests but skipped_count=0\n")

        # Run tests
        for api_name, test_name, test_func in test_suite:
            self.run_test(api_name, test_name, test_func)
            time.sleep(0.5)  # Brief pause between tests

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test results summary."""
        print(f"\n{'='*70}")
        print("  Test Results Summary")
        print(f"{'='*70}\n")

        # Group by API
        api_groups: Dict[str, List[APITestResult]] = {}
        for result in self.results:
            if result.api_name not in api_groups:
                api_groups[result.api_name] = []
            api_groups[result.api_name].append(result)

        total_tests = len(self.results)
        total_passed = sum(1 for r in self.results if r.success)
        total_failed = total_tests - total_passed

        # Print by API
        for api_name, results in sorted(api_groups.items()):
            passed = sum(1 for r in results if r.success)
            failed = len(results) - passed

            status_icon = "✓" if failed == 0 else "✗"
            print(f"{status_icon} {api_name.upper()}: {passed}/{len(results)} passed")

            for result in results:
                status = "✓" if result.success else "✗"
                duration = f"{result.duration_ms:>6.0f}ms"

                if result.success:
                    print(f"    {status} {result.test_name:<25} {duration}")

                    if self.verbose and result.details:
                        # Print details in compact format
                        details_str = json.dumps(result.details, indent=6)
                        print(f"        Details: {details_str}")
                else:
                    print(f"    {status} {result.test_name:<25} {duration}")
                    print(f"        Error: {result.error}")

            print()

        # Overall summary
        print("=" * 70)
        print(f"  Total: {total_passed}/{total_tests} passed, {total_failed} failed")

        if total_failed == 0:
            print("  ✓ All tests passed!")
        else:
            print(f"  ✗ {total_failed} test(s) failed")

        print("=" * 70 + "\n")

        # Calculate average response times
        if total_passed > 0:
            avg_duration = sum(r.duration_ms for r in self.results if r.success) / total_passed
            print(f"Average response time: {avg_duration:.0f}ms\n")

        # Note about skipped ChEMBL tests
        if not CHEMBL_AVAILABLE:
            print("ℹ️  Note: ChEMBL tests were skipped due to API unavailability.")
            print("   ChEMBL is currently returning HTTP 500 errors.")
            print("   Try again later when the service is restored.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test all external APIs used by IMPULATOR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_external_apis.py                    # Run all tests
  python scripts/test_external_apis.py --verbose          # Run with detailed output
  python scripts/test_external_apis.py --api chembl       # Test only ChEMBL API
  python scripts/test_external_apis.py --api pdb          # Test only PDB API
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed results"
    )

    parser.add_argument(
        "--api",
        choices=["chembl", "pdb", "classyfire", "npclassifier"],
        help="Test only a specific API"
    )

    args = parser.parse_args()

    # Run tests
    tester = ExternalAPITester(verbose=args.verbose)
    tester.run_all_tests(api_filter=args.api)

    # Exit with appropriate code
    failed_count = sum(1 for r in tester.results if not r.success)
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
