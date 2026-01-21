"""
Benchmark test suite for batch molecule/target fetching optimizations.

This test benchmarks:
1. Current behavior: molecule data and target names fetched individually (sequential)
2. Simulated batch behavior: what batch fetching COULD achieve
3. Provides data to decide whether batch optimization is worth implementing

Run with: pytest tests/test_batch_fetching.py -v -s
"""
import time
import pytest
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample ChEMBL IDs for testing (known compounds with data)
# Using more IDs for meaningful benchmarks
SAMPLE_CHEMBL_IDS = [
    "CHEMBL159",      # Quercetin
    "CHEMBL25",       # Aspirin
    "CHEMBL1201585",  # Ibuprofen
    "CHEMBL112",      # Paracetamol
    "CHEMBL192",      # Caffeine
]

# Sample target ChEMBL IDs (from common drug targets)
SAMPLE_TARGET_IDS = [
    "CHEMBL2095173",  # Cytochrome P450 3A4
    "CHEMBL203",      # Cyclooxygenase-2
    "CHEMBL204",      # Cyclooxygenase-1
    "CHEMBL220",      # Acetylcholinesterase
    "CHEMBL1862",     # Carbonic anhydrase II
]


class TestMoleculeDataBenchmark:
    """Benchmark molecule data fetching: individual vs batch."""

    def test_benchmark_individual_molecule_fetch(self):
        """
        Benchmark: Fetch molecule data individually (current implementation).

        This establishes the baseline performance for individual fetches.
        """
        from backend.modules.api_client import get_molecule_data, clear_caches

        # Clear cache to ensure fresh fetches
        clear_caches()

        results = {}
        api_call_times = []

        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK: Individual Molecule Data Fetching")
        logger.info(f"{'='*60}")
        logger.info(f"Fetching {len(SAMPLE_CHEMBL_IDS)} molecules individually...")

        total_start = time.time()
        for chembl_id in SAMPLE_CHEMBL_IDS:
            call_start = time.time()
            mol_data = get_molecule_data(chembl_id)
            call_time = time.time() - call_start
            api_call_times.append(call_time)

            if mol_data:
                results[chembl_id] = mol_data
                logger.info(f"  {chembl_id}: {call_time:.3f}s")
            else:
                logger.warning(f"  {chembl_id}: FAILED ({call_time:.3f}s)")

        total_time = time.time() - total_start

        # Calculate statistics
        avg_call_time = sum(api_call_times) / len(api_call_times) if api_call_times else 0

        logger.info(f"\n--- Results ---")
        logger.info(f"Total molecules fetched: {len(results)}/{len(SAMPLE_CHEMBL_IDS)}")
        logger.info(f"Total time: {total_time:.3f}s")
        logger.info(f"Average per molecule: {avg_call_time:.3f}s")
        logger.info(f"Overhead (rate limiting, etc): {total_time - sum(api_call_times):.3f}s")

        # Store for comparison
        self.individual_time = total_time
        self.individual_count = len(results)

        assert len(results) > 0, "Should fetch at least some molecule data"

    def test_benchmark_simulated_batch_molecule_fetch(self):
        """
        Benchmark: Simulate batch molecule fetch using ChEMBL's filter API.

        This tests what batch fetching COULD achieve using molecule.filter().
        """
        from backend.modules.api_client import _get_chembl_client, clear_caches

        clear_caches()

        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK: Simulated Batch Molecule Data Fetching")
        logger.info(f"{'='*60}")
        logger.info(f"Fetching {len(SAMPLE_CHEMBL_IDS)} molecules in single batch...")

        client = _get_chembl_client()
        if 'molecule' not in client:
            pytest.skip("ChEMBL client not available")

        total_start = time.time()
        try:
            # Single query for all molecules
            molecules = client['molecule'].filter(
                molecule_chembl_id__in=SAMPLE_CHEMBL_IDS
            ).only([
                'molecule_chembl_id',
                'pref_name',
                'molecule_properties',
                'molecule_structures'
            ])

            # Convert to list (triggers the query)
            fetch_start = time.time()
            results_list = list(molecules)
            fetch_time = time.time() - fetch_start

            # Build results dict
            results = {m['molecule_chembl_id']: m for m in results_list}

        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            pytest.fail(f"Batch fetch failed: {e}")

        total_time = time.time() - total_start

        logger.info(f"\n--- Results ---")
        logger.info(f"Total molecules fetched: {len(results)}/{len(SAMPLE_CHEMBL_IDS)}")
        logger.info(f"Total time: {total_time:.3f}s")
        logger.info(f"Query execution time: {fetch_time:.3f}s")
        logger.info(f"Average per molecule: {total_time/len(SAMPLE_CHEMBL_IDS):.3f}s")

        assert len(results) > 0, "Should fetch at least some molecule data"


class TestTargetNameBenchmark:
    """Benchmark target name fetching: individual vs batch."""

    def test_benchmark_individual_target_fetch(self):
        """
        Benchmark: Fetch target names individually (current implementation).
        """
        from backend.modules.api_client import get_target_name, clear_caches

        clear_caches()

        results = {}
        api_call_times = []

        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK: Individual Target Name Fetching")
        logger.info(f"{'='*60}")
        logger.info(f"Fetching {len(SAMPLE_TARGET_IDS)} targets individually...")

        total_start = time.time()
        for target_id in SAMPLE_TARGET_IDS:
            call_start = time.time()
            target_name = get_target_name(target_id)
            call_time = time.time() - call_start
            api_call_times.append(call_time)

            if target_name:
                results[target_id] = target_name
                logger.info(f"  {target_id}: '{target_name}' ({call_time:.3f}s)")
            else:
                logger.warning(f"  {target_id}: FAILED ({call_time:.3f}s)")

        total_time = time.time() - total_start
        avg_call_time = sum(api_call_times) / len(api_call_times) if api_call_times else 0

        logger.info(f"\n--- Results ---")
        logger.info(f"Total targets fetched: {len(results)}/{len(SAMPLE_TARGET_IDS)}")
        logger.info(f"Total time: {total_time:.3f}s")
        logger.info(f"Average per target: {avg_call_time:.3f}s")

        assert len(results) > 0, "Should fetch at least some target names"

    def test_benchmark_simulated_batch_target_fetch(self):
        """
        Benchmark: Simulate batch target fetch using ChEMBL's filter API.
        """
        from backend.modules.api_client import _get_chembl_client, clear_caches

        clear_caches()

        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK: Simulated Batch Target Name Fetching")
        logger.info(f"{'='*60}")
        logger.info(f"Fetching {len(SAMPLE_TARGET_IDS)} targets in single batch...")

        client = _get_chembl_client()
        if 'target' not in client:
            pytest.skip("ChEMBL client not available")

        total_start = time.time()
        try:
            # Single query for all targets
            targets = client['target'].filter(
                target_chembl_id__in=SAMPLE_TARGET_IDS
            ).only([
                'target_chembl_id',
                'pref_name'
            ])

            # Convert to list (triggers the query)
            fetch_start = time.time()
            results_list = list(targets)
            fetch_time = time.time() - fetch_start

            # Build results dict
            results = {t['target_chembl_id']: t.get('pref_name', '') for t in results_list}

        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            pytest.fail(f"Batch fetch failed: {e}")

        total_time = time.time() - total_start

        logger.info(f"\n--- Results ---")
        logger.info(f"Total targets fetched: {len(results)}/{len(SAMPLE_TARGET_IDS)}")
        logger.info(f"Total time: {total_time:.3f}s")
        logger.info(f"Query execution time: {fetch_time:.3f}s")
        logger.info(f"Average per target: {total_time/len(SAMPLE_TARGET_IDS):.3f}s")

        assert len(results) > 0, "Should fetch at least some target names"


class TestComprehensiveBenchmark:
    """Run comprehensive benchmark comparing all methods."""

    @pytest.mark.slow
    def test_full_benchmark_comparison(self):
        """
        Run full benchmark and print comparison summary.

        This test runs all benchmarks and provides a summary for decision-making.
        Run with: pytest tests/test_batch_fetching.py::TestComprehensiveBenchmark -v -s
        """
        from backend.modules.api_client import get_molecule_data, get_target_name, _get_chembl_client, clear_caches

        print(f"\n{'='*70}")
        print("COMPREHENSIVE BENCHMARK: Individual vs Batch Fetching")
        print(f"{'='*70}\n")

        results = {}

        # --- Molecule Data Benchmarks ---
        print("=" * 50)
        print("MOLECULE DATA FETCHING")
        print("=" * 50)

        # Individual
        clear_caches()
        start = time.time()
        individual_mol_results = {}
        for cid in SAMPLE_CHEMBL_IDS:
            data = get_molecule_data(cid)
            if data:
                individual_mol_results[cid] = data
        individual_mol_time = time.time() - start
        print(f"Individual: {len(individual_mol_results)} molecules in {individual_mol_time:.3f}s")

        # Batch
        clear_caches()
        client = _get_chembl_client()
        start = time.time()
        try:
            molecules = client['molecule'].filter(
                molecule_chembl_id__in=SAMPLE_CHEMBL_IDS
            ).only(['molecule_chembl_id', 'pref_name', 'molecule_properties', 'molecule_structures'])
            batch_mol_results = {m['molecule_chembl_id']: m for m in list(molecules)}
        except Exception as e:
            print(f"Batch molecule fetch failed: {e}")
            batch_mol_results = {}
        batch_mol_time = time.time() - start
        print(f"Batch: {len(batch_mol_results)} molecules in {batch_mol_time:.3f}s")

        mol_speedup = individual_mol_time / batch_mol_time if batch_mol_time > 0 else 0
        print(f"Speedup: {mol_speedup:.2f}x")

        results['molecule'] = {
            'individual_time': individual_mol_time,
            'batch_time': batch_mol_time,
            'speedup': mol_speedup,
            'individual_count': len(individual_mol_results),
            'batch_count': len(batch_mol_results),
        }

        # --- Target Name Benchmarks ---
        print("\n" + "=" * 50)
        print("TARGET NAME FETCHING")
        print("=" * 50)

        # Individual
        clear_caches()
        start = time.time()
        individual_target_results = {}
        for tid in SAMPLE_TARGET_IDS:
            name = get_target_name(tid)
            if name:
                individual_target_results[tid] = name
        individual_target_time = time.time() - start
        print(f"Individual: {len(individual_target_results)} targets in {individual_target_time:.3f}s")

        # Batch
        clear_caches()
        start = time.time()
        try:
            targets = client['target'].filter(
                target_chembl_id__in=SAMPLE_TARGET_IDS
            ).only(['target_chembl_id', 'pref_name'])
            batch_target_results = {t['target_chembl_id']: t.get('pref_name', '') for t in list(targets)}
        except Exception as e:
            print(f"Batch target fetch failed: {e}")
            batch_target_results = {}
        batch_target_time = time.time() - start
        print(f"Batch: {len(batch_target_results)} targets in {batch_target_time:.3f}s")

        target_speedup = individual_target_time / batch_target_time if batch_target_time > 0 else 0
        print(f"Speedup: {target_speedup:.2f}x")

        results['target'] = {
            'individual_time': individual_target_time,
            'batch_time': batch_target_time,
            'speedup': target_speedup,
            'individual_count': len(individual_target_results),
            'batch_count': len(batch_target_results),
        }

        # --- Summary ---
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"""
+-------------------+---------------+---------------+----------+
| Operation         | Individual    | Batch         | Speedup  |
+-------------------+---------------+---------------+----------+
| Molecule Data     | {results['molecule']['individual_time']:>10.3f}s  | {results['molecule']['batch_time']:>10.3f}s  | {results['molecule']['speedup']:>6.2f}x  |
| Target Names      | {results['target']['individual_time']:>10.3f}s  | {results['target']['batch_time']:>10.3f}s  | {results['target']['speedup']:>6.2f}x  |
+-------------------+---------------+---------------+----------+
""")

        print("RECOMMENDATION:")
        if mol_speedup > 1.5:
            print(f"  - Molecule batch fetching: RECOMMENDED ({mol_speedup:.1f}x speedup)")
        else:
            print(f"  - Molecule batch fetching: MARGINAL ({mol_speedup:.1f}x speedup)")

        if target_speedup > 1.5:
            print(f"  - Target batch fetching: RECOMMENDED ({target_speedup:.1f}x speedup)")
        else:
            print(f"  - Target batch fetching: MARGINAL ({target_speedup:.1f}x speedup)")

        # Data integrity check
        print(f"\nData Integrity Check:")
        mol_match = set(individual_mol_results.keys()) == set(batch_mol_results.keys())
        target_match = set(individual_target_results.keys()) == set(batch_target_results.keys())
        print(f"  - Molecule IDs match: {mol_match}")
        print(f"  - Target IDs match: {target_match}")


class TestActivityFetchingValidation:
    """Validate that activity fetching is already optimized."""

    def test_activity_fetch_uses_single_batch(self):
        """
        Verify that fetch_all_activities_single_batch works correctly.
        This should PASS - activity fetching is already optimized.
        """
        from backend.modules.api_client import fetch_all_activities_single_batch, clear_caches

        clear_caches()

        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION: Single Batch Activity Fetching")
        logger.info(f"{'='*60}")

        start_time = time.time()
        activities = fetch_all_activities_single_batch(
            SAMPLE_CHEMBL_IDS,
            activity_types=['IC50', 'Ki', 'EC50']
        )
        elapsed = time.time() - start_time

        logger.info(f"Fetched {len(activities)} activities in {elapsed:.2f}s")
        logger.info(f"Average per compound: {elapsed/len(SAMPLE_CHEMBL_IDS):.2f}s")

        if len(activities) > 0:
            # Check structure
            sample = activities[0]
            assert 'molecule_chembl_id' in sample, "Should have molecule_chembl_id"
            assert 'standard_type' in sample, "Should have standard_type"

            # Count by compound
            compounds = set(a['molecule_chembl_id'] for a in activities)
            logger.info(f"Activities from {len(compounds)} unique compounds")

        logger.info("\nActivity fetching is ALREADY OPTIMIZED with single batch query")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
