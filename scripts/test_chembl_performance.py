"""
Comprehensive ChEMBL API Performance Test.

Tests all ChEMBL API endpoints used by IMPULATOR:
1. Similarity search
2. Activity fetching
3. Molecule data
4. Target names
5. Drug indications

Compares three approaches:
- chembl_webresource_client library with DEFAULT settings (MAX_LIMIT=20)
- chembl_webresource_client library with MODIFIED settings (MAX_LIMIT=1000)
- Direct REST API with limit=1000

Reference Documentation:
- ChEMBL REST API: https://www.ebi.ac.uk/chembl/api/data/docs
- ChEMBL Web Services: https://chembl.gitbook.io/chembl-interface-documentation/web-services
- Python Client GitHub: https://github.com/chembl/chembl_webresource_client

Key findings from documentation:
- REST API maximum limit is 1000 per request
- Library default MAX_LIMIT is 20 (configurable via Settings)
- Settings is a singleton: Settings.Instance().MAX_LIMIT = 1000

Run: python scripts/test_chembl_performance.py
"""
import time
import logging
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Test compounds (from typical IMPULATOR usage)
TEST_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
TEST_SIMILARITY_THRESHOLD = 70

# ChEMBL IDs for testing (mix of compounds with varying activity counts)
TEST_CHEMBL_IDS = [
    'CHEMBL25',       # Aspirin - many activities
    'CHEMBL1697753',
    'CHEMBL267936',
    'CHEMBL553025',
    'CHEMBL137773',
    'CHEMBL57489',
    'CHEMBL1799713',
    'CHEMBL538426',
    'CHEMBL357134',
]

# Target IDs for testing
TEST_TARGET_IDS = [
    'CHEMBL2096904',
    'CHEMBL4036',
    'CHEMBL2094127',
    'CHEMBL3712794',
    'CHEMBL2364679',
]

# Activity types to filter
ACTIVITY_TYPES = {'IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'GI50', 'MIC'}

# API configuration
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_TIMEOUT = 60
MAX_LIMIT = 1000  # Maximum allowed by ChEMBL API


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class TestResult:
    """Container for test results."""
    method: str
    endpoint: str
    time_seconds: float
    request_count: int
    result_count: int
    error: Optional[str] = None

    def __str__(self):
        if self.error:
            return f"{self.method:30} | ERROR: {self.error}"
        return f"{self.method:30} | {self.time_seconds:8.2f}s | {self.request_count:3} reqs | {self.result_count:6} results"


# =============================================================================
# HTTP SESSION SETUP
# =============================================================================

def create_session() -> requests.Session:
    """Create requests session with retry configuration."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


SESSION = create_session()


# =============================================================================
# API STATUS CHECK
# =============================================================================

def check_api_status() -> Tuple[bool, str]:
    """
    Check if ChEMBL API is available.

    Returns:
        Tuple of (is_available, status_message)
    """
    try:
        response = SESSION.get(f"{BASE_URL}/status.json", timeout=15)
        if response.status_code == 200:
            data = response.json()
            version = data.get('chembl_db_version', 'unknown')
            return True, f"ChEMBL {version} - API is available"
        return False, f"API returned status code {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "API request timed out"
    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


# =============================================================================
# REST API TESTS (Direct HTTP requests)
# =============================================================================

def rest_api_activities(chembl_ids: List[str], limit: int = MAX_LIMIT) -> TestResult:
    """
    Fetch activities using direct REST API.

    Args:
        chembl_ids: List of ChEMBL IDs
        limit: Page size (max 1000)
    """
    url = f"{BASE_URL}/activity.json"
    ids_param = ",".join(chembl_ids)
    fields = "molecule_chembl_id,standard_type,standard_value,standard_units,target_chembl_id"

    all_activities = []
    offset = 0
    request_count = 0

    start = time.time()
    try:
        while True:
            params = {
                "molecule_chembl_id__in": ids_param,
                "only": fields,
                "limit": limit,
                "offset": offset
            }

            response = SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            request_count += 1

            activities = data.get('activities', [])
            if not activities:
                break

            filtered = [a for a in activities if a.get('standard_type') in ACTIVITY_TYPES]
            all_activities.extend(filtered)

            if len(activities) < limit:
                break
            offset += limit

        elapsed = time.time() - start
        return TestResult(
            method=f"REST API (limit={limit})",
            endpoint="activity",
            time_seconds=elapsed,
            request_count=request_count,
            result_count=len(all_activities)
        )
    except Exception as e:
        return TestResult(
            method=f"REST API (limit={limit})",
            endpoint="activity",
            time_seconds=time.time() - start,
            request_count=request_count,
            result_count=0,
            error=str(e)
        )


def rest_api_molecules(chembl_ids: List[str]) -> TestResult:
    """Fetch molecule data using direct REST API."""
    url = f"{BASE_URL}/molecule.json"
    ids_param = ",".join(chembl_ids)
    fields = "molecule_chembl_id,pref_name,molecule_properties,molecule_structures"

    start = time.time()
    try:
        params = {
            "molecule_chembl_id__in": ids_param,
            "only": fields,
            "limit": MAX_LIMIT
        }

        response = SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        molecules = data.get('molecules', [])
        elapsed = time.time() - start

        return TestResult(
            method="REST API (limit=1000)",
            endpoint="molecule",
            time_seconds=elapsed,
            request_count=1,
            result_count=len(molecules)
        )
    except Exception as e:
        return TestResult(
            method="REST API (limit=1000)",
            endpoint="molecule",
            time_seconds=time.time() - start,
            request_count=1,
            result_count=0,
            error=str(e)
        )


def rest_api_targets(target_ids: List[str]) -> TestResult:
    """Fetch target names using direct REST API."""
    url = f"{BASE_URL}/target.json"
    ids_param = ",".join(target_ids)

    start = time.time()
    try:
        params = {
            "target_chembl_id__in": ids_param,
            "only": "target_chembl_id,pref_name",
            "limit": MAX_LIMIT
        }

        response = SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        targets = data.get('targets', [])
        elapsed = time.time() - start

        return TestResult(
            method="REST API (limit=1000)",
            endpoint="target",
            time_seconds=elapsed,
            request_count=1,
            result_count=len(targets)
        )
    except Exception as e:
        return TestResult(
            method="REST API (limit=1000)",
            endpoint="target",
            time_seconds=time.time() - start,
            request_count=1,
            result_count=0,
            error=str(e)
        )


def rest_api_drug_indications(chembl_ids: List[str]) -> TestResult:
    """Fetch drug indications using direct REST API (batch query)."""
    url = f"{BASE_URL}/drug_indication.json"
    ids_param = ",".join(chembl_ids)

    start = time.time()
    try:
        params = {
            "molecule_chembl_id__in": ids_param,
            "limit": MAX_LIMIT
        }

        response = SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        indications = data.get('drug_indications', [])
        elapsed = time.time() - start

        return TestResult(
            method="REST API batch (limit=1000)",
            endpoint="drug_indication",
            time_seconds=elapsed,
            request_count=1,
            result_count=len(indications)
        )
    except Exception as e:
        return TestResult(
            method="REST API batch (limit=1000)",
            endpoint="drug_indication",
            time_seconds=time.time() - start,
            request_count=1,
            result_count=0,
            error=str(e)
        )


def rest_api_similarity(smiles: str, threshold: int) -> TestResult:
    """Perform similarity search using direct REST API."""
    url = f"{BASE_URL}/similarity/{smiles}/{threshold}.json"

    start = time.time()
    try:
        params = {
            "only": "molecule_chembl_id",
            "limit": MAX_LIMIT
        }

        response = SESSION.get(url, params=params, timeout=90)
        response.raise_for_status()
        data = response.json()

        molecules = data.get('molecules', [])
        elapsed = time.time() - start

        return TestResult(
            method="REST API (limit=1000)",
            endpoint="similarity",
            time_seconds=elapsed,
            request_count=1,
            result_count=len(molecules)
        )
    except Exception as e:
        return TestResult(
            method="REST API (limit=1000)",
            endpoint="similarity",
            time_seconds=time.time() - start,
            request_count=1,
            result_count=0,
            error=str(e)
        )


# =============================================================================
# LIBRARY TESTS (chembl_webresource_client)
# =============================================================================

def get_library_client(max_limit: int = 20):
    """
    Get ChEMBL client with specified MAX_LIMIT setting.

    IMPORTANT: Settings must be modified BEFORE importing new_client!

    Args:
        max_limit: Page size limit (20=default, 1000=optimized)
    """
    # Modify settings BEFORE importing the client
    from chembl_webresource_client.settings import Settings
    settings = Settings.Instance()

    # Store original and set new limit
    original_limit = settings.MAX_LIMIT
    settings.MAX_LIMIT = max_limit

    # Also increase timeout for large requests
    settings.TIMEOUT = 60
    if hasattr(settings, 'NEW_CLIENT_TIMEOUT'):
        settings.NEW_CLIENT_TIMEOUT = 60

    # Now import the client (it will use updated settings)
    from chembl_webresource_client.new_client import new_client

    return new_client, original_limit


def library_activities(chembl_ids: List[str], max_limit: int = 20) -> TestResult:
    """Fetch activities using library with specified limit."""
    method_name = f"Library (MAX_LIMIT={max_limit})"

    start = time.time()
    try:
        client, _ = get_library_client(max_limit)

        activities = client.activity.filter(
            molecule_chembl_id__in=chembl_ids
        ).only([
            'molecule_chembl_id',
            'standard_type',
            'standard_value',
            'standard_units',
            'target_chembl_id'
        ])

        all_raw = list(activities)
        filtered = [a for a in all_raw if a.get('standard_type') in ACTIVITY_TYPES]

        elapsed = time.time() - start

        # Estimate request count based on limit
        total = len(all_raw)
        est_requests = (total // max_limit) + 1 if total > 0 else 1

        return TestResult(
            method=method_name,
            endpoint="activity",
            time_seconds=elapsed,
            request_count=est_requests,
            result_count=len(filtered)
        )
    except Exception as e:
        return TestResult(
            method=method_name,
            endpoint="activity",
            time_seconds=time.time() - start,
            request_count=0,
            result_count=0,
            error=str(e)
        )


def library_molecules(chembl_ids: List[str], max_limit: int = 20) -> TestResult:
    """Fetch molecule data using library."""
    method_name = f"Library (MAX_LIMIT={max_limit})"

    start = time.time()
    try:
        client, _ = get_library_client(max_limit)

        molecules = client.molecule.filter(
            molecule_chembl_id__in=chembl_ids
        ).only([
            'molecule_chembl_id',
            'pref_name',
            'molecule_properties',
            'molecule_structures'
        ])

        result = list(molecules)
        elapsed = time.time() - start

        est_requests = (len(result) // max_limit) + 1 if result else 1

        return TestResult(
            method=method_name,
            endpoint="molecule",
            time_seconds=elapsed,
            request_count=est_requests,
            result_count=len(result)
        )
    except Exception as e:
        return TestResult(
            method=method_name,
            endpoint="molecule",
            time_seconds=time.time() - start,
            request_count=0,
            result_count=0,
            error=str(e)
        )


def library_targets(target_ids: List[str], max_limit: int = 20) -> TestResult:
    """Fetch target names using library."""
    method_name = f"Library (MAX_LIMIT={max_limit})"

    start = time.time()
    try:
        client, _ = get_library_client(max_limit)

        targets = client.target.filter(
            target_chembl_id__in=target_ids
        ).only([
            'target_chembl_id',
            'pref_name'
        ])

        result = list(targets)
        elapsed = time.time() - start

        est_requests = (len(result) // max_limit) + 1 if result else 1

        return TestResult(
            method=method_name,
            endpoint="target",
            time_seconds=elapsed,
            request_count=est_requests,
            result_count=len(result)
        )
    except Exception as e:
        return TestResult(
            method=method_name,
            endpoint="target",
            time_seconds=time.time() - start,
            request_count=0,
            result_count=0,
            error=str(e)
        )


def library_drug_indications_sequential(chembl_ids: List[str]) -> TestResult:
    """
    Fetch drug indications using library (sequential per-compound).

    This is the CURRENT approach in IMPULATOR - one query per compound.
    """
    method_name = "Library sequential (1 per ID)"

    start = time.time()
    all_indications = []
    request_count = 0

    try:
        client, _ = get_library_client(1000)  # Limit doesn't matter for single-ID queries

        for chembl_id in chembl_ids:
            try:
                indications = client.drug_indication.filter(
                    molecule_chembl_id=chembl_id
                )
                result = list(indications)
                all_indications.extend(result)
                request_count += 1
            except Exception:
                request_count += 1  # Count failed requests too

        elapsed = time.time() - start

        return TestResult(
            method=method_name,
            endpoint="drug_indication",
            time_seconds=elapsed,
            request_count=request_count,
            result_count=len(all_indications)
        )
    except Exception as e:
        return TestResult(
            method=method_name,
            endpoint="drug_indication",
            time_seconds=time.time() - start,
            request_count=request_count,
            result_count=0,
            error=str(e)
        )


def library_similarity(smiles: str, threshold: int, max_limit: int = 20) -> TestResult:
    """Perform similarity search using library."""
    method_name = f"Library (MAX_LIMIT={max_limit})"

    start = time.time()
    try:
        client, _ = get_library_client(max_limit)

        results = client.similarity.filter(
            smiles=smiles,
            similarity=threshold
        ).only(['molecule_chembl_id'])

        result_list = list(results)
        elapsed = time.time() - start

        est_requests = (len(result_list) // max_limit) + 1 if result_list else 1

        return TestResult(
            method=method_name,
            endpoint="similarity",
            time_seconds=elapsed,
            request_count=est_requests,
            result_count=len(result_list)
        )
    except Exception as e:
        return TestResult(
            method=method_name,
            endpoint="similarity",
            time_seconds=time.time() - start,
            request_count=0,
            result_count=0,
            error=str(e)
        )


# =============================================================================
# TEST RUNNER
# =============================================================================

def print_header(text: str):
    """Print formatted header."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f" {text}")
    logger.info("=" * 80)


def print_results_table(results: List[TestResult], endpoint: str):
    """Print results in a formatted table."""
    logger.info(f"\n{endpoint.upper()} Results:")
    logger.info("-" * 80)
    logger.info(f"{'Method':<35} | {'Time':>10} | {'Reqs':>6} | {'Results':>8}")
    logger.info("-" * 80)

    for r in results:
        if r.error:
            logger.info(f"{r.method:<35} | {'ERROR':>10} | {'-':>6} | {r.error[:20]}")
        else:
            logger.info(f"{r.method:<35} | {r.time_seconds:>9.2f}s | {r.request_count:>6} | {r.result_count:>8}")

    # Calculate speedup if we have both REST and Library results
    rest_results = [r for r in results if 'REST' in r.method and not r.error]
    lib_results = [r for r in results if 'Library' in r.method and 'MAX_LIMIT=20' in r.method and not r.error]

    if rest_results and lib_results:
        speedup = lib_results[0].time_seconds / rest_results[0].time_seconds
        logger.info("-" * 80)
        logger.info(f"{'Speedup (REST vs Library default):':<35} | {speedup:>9.1f}x |")


def run_all_tests(skip_slow_library: bool = False):
    """
    Run all performance tests.

    Args:
        skip_slow_library: If True, skip library tests with default MAX_LIMIT=20
    """
    print_header("IMPULATOR ChEMBL API Performance Test")
    logger.info(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Test ChEMBL IDs: {len(TEST_CHEMBL_IDS)}")
    logger.info(f"Test Target IDs: {len(TEST_TARGET_IDS)}")
    logger.info(f"Test SMILES: {TEST_SMILES[:30]}...")

    # Check API status
    print_header("1. API Status Check")
    is_available, status_msg = check_api_status()
    logger.info(f"  Status: {status_msg}")

    if not is_available:
        logger.error("\n  ChEMBL API is not available. Cannot run tests.")
        logger.error("  Please try again later when the API is back online.")
        return None

    all_results = {}

    # ==========================================================================
    # TEST 1: SIMILARITY SEARCH
    # ==========================================================================
    print_header("2. Similarity Search")
    logger.info(f"  SMILES: {TEST_SMILES}")
    logger.info(f"  Threshold: {TEST_SIMILARITY_THRESHOLD}%")

    similarity_results = []

    # REST API
    logger.info("\n  Testing REST API...")
    similarity_results.append(rest_api_similarity(TEST_SMILES, TEST_SIMILARITY_THRESHOLD))

    # Library with MAX_LIMIT=1000
    logger.info("  Testing Library (MAX_LIMIT=1000)...")
    similarity_results.append(library_similarity(TEST_SMILES, TEST_SIMILARITY_THRESHOLD, max_limit=1000))

    # Library with default MAX_LIMIT=20 (slow)
    if not skip_slow_library:
        logger.info("  Testing Library (MAX_LIMIT=20) - may be slow...")
        similarity_results.append(library_similarity(TEST_SMILES, TEST_SIMILARITY_THRESHOLD, max_limit=20))

    print_results_table(similarity_results, "Similarity Search")
    all_results['similarity'] = similarity_results

    # ==========================================================================
    # TEST 2: ACTIVITY FETCHING
    # ==========================================================================
    print_header("3. Activity Fetching (Main Bottleneck)")
    logger.info(f"  ChEMBL IDs: {TEST_CHEMBL_IDS}")

    activity_results = []

    # REST API with limit=1000
    logger.info("\n  Testing REST API (limit=1000)...")
    activity_results.append(rest_api_activities(TEST_CHEMBL_IDS, limit=1000))

    # REST API with limit=500 (for comparison)
    logger.info("  Testing REST API (limit=500)...")
    activity_results.append(rest_api_activities(TEST_CHEMBL_IDS, limit=500))

    # Library with MAX_LIMIT=1000
    logger.info("  Testing Library (MAX_LIMIT=1000)...")
    activity_results.append(library_activities(TEST_CHEMBL_IDS, max_limit=1000))

    # Library with default MAX_LIMIT=20 (slow)
    if not skip_slow_library:
        logger.info("  Testing Library (MAX_LIMIT=20) - THIS WILL BE SLOW...")
        activity_results.append(library_activities(TEST_CHEMBL_IDS, max_limit=20))

    print_results_table(activity_results, "Activity Fetching")
    all_results['activity'] = activity_results

    # ==========================================================================
    # TEST 3: MOLECULE DATA
    # ==========================================================================
    print_header("4. Molecule Data")

    molecule_results = []

    logger.info("\n  Testing REST API...")
    molecule_results.append(rest_api_molecules(TEST_CHEMBL_IDS))

    logger.info("  Testing Library (MAX_LIMIT=1000)...")
    molecule_results.append(library_molecules(TEST_CHEMBL_IDS, max_limit=1000))

    if not skip_slow_library:
        logger.info("  Testing Library (MAX_LIMIT=20)...")
        molecule_results.append(library_molecules(TEST_CHEMBL_IDS, max_limit=20))

    print_results_table(molecule_results, "Molecule Data")
    all_results['molecule'] = molecule_results

    # ==========================================================================
    # TEST 4: TARGET NAMES
    # ==========================================================================
    print_header("5. Target Names")

    target_results = []

    logger.info("\n  Testing REST API...")
    target_results.append(rest_api_targets(TEST_TARGET_IDS))

    logger.info("  Testing Library (MAX_LIMIT=1000)...")
    target_results.append(library_targets(TEST_TARGET_IDS, max_limit=1000))

    if not skip_slow_library:
        logger.info("  Testing Library (MAX_LIMIT=20)...")
        target_results.append(library_targets(TEST_TARGET_IDS, max_limit=20))

    print_results_table(target_results, "Target Names")
    all_results['target'] = target_results

    # ==========================================================================
    # TEST 5: DRUG INDICATIONS
    # ==========================================================================
    print_header("6. Drug Indications")

    indication_results = []

    logger.info("\n  Testing REST API (batch)...")
    indication_results.append(rest_api_drug_indications(TEST_CHEMBL_IDS))

    logger.info("  Testing Library (sequential per compound)...")
    indication_results.append(library_drug_indications_sequential(TEST_CHEMBL_IDS))

    print_results_table(indication_results, "Drug Indications")
    all_results['drug_indication'] = indication_results

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_header("SUMMARY & RECOMMENDATIONS")

    logger.info("\nKey Findings:")
    logger.info("-" * 80)

    # Calculate total times
    rest_total = sum(
        r.time_seconds for results in all_results.values()
        for r in results if 'REST' in r.method and 'limit=1000' in r.method and not r.error
    )

    lib_1000_total = sum(
        r.time_seconds for results in all_results.values()
        for r in results if 'MAX_LIMIT=1000' in r.method and not r.error
    )

    lib_20_total = sum(
        r.time_seconds for results in all_results.values()
        for r in results if 'MAX_LIMIT=20' in r.method and not r.error
    )

    logger.info(f"  REST API (limit=1000) total:      {rest_total:>8.2f}s")
    logger.info(f"  Library (MAX_LIMIT=1000) total:   {lib_1000_total:>8.2f}s")
    if lib_20_total > 0:
        logger.info(f"  Library (MAX_LIMIT=20) total:     {lib_20_total:>8.2f}s")
        logger.info(f"\n  Speedup (REST vs default Library): {lib_20_total/rest_total:.1f}x")

    logger.info("\nRecommendations:")
    logger.info("-" * 80)
    logger.info("  1. Use Settings.Instance().MAX_LIMIT = 1000 before importing new_client")
    logger.info("  2. For drug_indications, use batch REST API query instead of sequential")
    logger.info("  3. Consider switching to direct REST API for better control")
    logger.info("  4. The ChEMBL API maximum limit is 1000 per request")

    logger.info("\nCode to fix in api_client.py:")
    logger.info("-" * 80)
    logger.info("""
    # Add this BEFORE importing new_client:
    from chembl_webresource_client.settings import Settings
    Settings.Instance().MAX_LIMIT = 1000
    Settings.Instance().TIMEOUT = 60

    # Then import the client:
    from chembl_webresource_client.new_client import new_client
    """)

    return all_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Check for --skip-slow flag
    skip_slow = "--skip-slow" in sys.argv

    if skip_slow:
        logger.info("Skipping slow library tests (MAX_LIMIT=20)")

    results = run_all_tests(skip_slow_library=skip_slow)

    if results:
        logger.info("\n" + "=" * 80)
        logger.info(" Test completed successfully!")
        logger.info("=" * 80)
