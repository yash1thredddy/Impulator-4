"""
Performance benchmarks to prevent regression.

These tests verify that optimizations are working and
performance hasn't degraded.

Run with: pytest tests/performance/ -v
"""
import pytest
import time
import pandas as pd
import numpy as np


class TestEfficiencyMetricsPerformance:
    """Benchmark tests for efficiency metric calculations."""

    @pytest.mark.benchmark
    def test_efficiency_metrics_vectorized_performance(self):
        """Efficiency metrics should process 10k rows in < 2 seconds.

        Uses the vectorized implementation which should be 50-100x faster
        than the iterrows approach.
        """
        # Create large test dataset
        n_rows = 10000
        df = pd.DataFrame({
            'pActivity': np.random.uniform(4, 10, n_rows),
            'Molecular_Weight': np.random.uniform(200, 600, n_rows),
            'TPSA': np.random.uniform(20, 150, n_rows),
            'NPOL': np.random.randint(2, 10, n_rows),
            'Heavy_Atoms': np.random.randint(10, 50, n_rows),
        })

        from backend.modules.efficiency_metrics import calculate_efficiency_metrics_dataframe

        # Time the operation
        start = time.time()
        result = calculate_efficiency_metrics_dataframe(df)
        elapsed = time.time() - start

        # Assertions
        assert elapsed < 2.0, f"Too slow: {elapsed:.2f}s (should be < 2s for 10k rows)"
        assert 'SEI' in result.columns
        assert 'BEI' in result.columns
        assert 'NSEI' in result.columns
        assert 'NBEI' in result.columns

        # Verify data integrity
        assert len(result) == n_rows
        assert result['SEI'].notna().sum() > n_rows * 0.9  # At least 90% should be valid

    @pytest.mark.benchmark
    def test_single_metric_calculation_performance(self):
        """Single metric calculations should be fast."""
        from backend.modules.efficiency_metrics import (
            calculate_sei, calculate_bei, calculate_nsei, calculate_nbei
        )

        # Time 10000 individual calculations
        n_iterations = 10000
        start = time.time()

        for _ in range(n_iterations):
            calculate_sei(7.5, 85.0)
            calculate_bei(7.5, 350.0)
            calculate_nsei(7.5, 5)
            calculate_nbei(7.5, 25)

        elapsed = time.time() - start

        # Should complete 40k calculations in < 1 second
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for {n_iterations * 4} calculations"


class TestOQPLAScoringPerformance:
    """Benchmark tests for OQPLA scoring."""

    @pytest.mark.benchmark
    def test_oqpla_scoring_performance(self):
        """OQPLA scoring should process 1k rows in < 30 seconds.

        Note: This test doesn't make actual API calls, testing the
        calculation logic only.
        """
        from backend.modules.oqpla_scoring import calculate_distance_to_best_score

        # Create test dataset with typical values
        n_rows = 1000
        df = pd.DataFrame({
            'pActivity': np.random.uniform(4, 10, n_rows),
            'Molecular_Weight': np.random.uniform(200, 600, n_rows),
            'TPSA': np.random.uniform(20, 150, n_rows),
            'Heavy_Atoms': np.random.randint(10, 50, n_rows),
            'SEI': np.random.uniform(5, 30, n_rows),
            'BEI': np.random.uniform(10, 40, n_rows),
        })

        # Calculate Modulus for distance score
        df['Modulus_SEI_BEI'] = np.sqrt(df['SEI']**2 + df['BEI']**2)

        start = time.time()
        distance_scores = calculate_distance_to_best_score(df)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Distance score too slow: {elapsed:.2f}s"
        assert len(distance_scores) == n_rows

        # Verify scores are in valid range
        assert (distance_scores >= 0).all()
        assert (distance_scores <= 1).all()


class TestDataFrameOperationsPerformance:
    """Benchmark tests for DataFrame operations."""

    @pytest.mark.benchmark
    def test_large_dataframe_filtering(self):
        """Large DataFrame filtering should be efficient."""
        n_rows = 50000

        df = pd.DataFrame({
            'ChEMBL_ID': [f'CHEMBL{i}' for i in range(n_rows)],
            'Activity_nM': np.random.exponential(100, n_rows),
            'pActivity': np.random.uniform(4, 10, n_rows),
            'Status': np.random.choice(['active', 'inactive'], n_rows),
        })

        start = time.time()

        # Common filtering operations
        active = df[df['Status'] == 'active']
        high_activity = df[df['pActivity'] > 7.0]
        low_nm = df[df['Activity_nM'] < 100]

        elapsed = time.time() - start

        assert elapsed < 0.5, f"DataFrame filtering too slow: {elapsed:.2f}s"
        assert len(active) + len(df[df['Status'] == 'inactive']) == n_rows

    @pytest.mark.benchmark
    def test_groupby_aggregation_performance(self):
        """GroupBy aggregation should be efficient."""
        n_rows = 50000
        n_groups = 100

        df = pd.DataFrame({
            'Target_ID': np.random.choice([f'TARGET{i}' for i in range(n_groups)], n_rows),
            'Activity_nM': np.random.exponential(100, n_rows),
            'pActivity': np.random.uniform(4, 10, n_rows),
        })

        start = time.time()

        # Common aggregations
        grouped = df.groupby('Target_ID').agg({
            'Activity_nM': ['mean', 'min', 'max', 'count'],
            'pActivity': ['mean', 'std']
        })

        elapsed = time.time() - start

        assert elapsed < 0.5, f"GroupBy too slow: {elapsed:.2f}s"
        assert len(grouped) == n_groups


class TestCachePerformance:
    """Benchmark tests for caching behavior."""

    @pytest.mark.benchmark
    def test_cache_decorator_operations(self):
        """Cache decorator operations should be fast."""
        from backend.modules.api_client import cache_non_none

        # Test the cache decorator performance
        call_count = [0]

        @cache_non_none(maxsize=1000, ttl_seconds=3600)
        def cached_function(key):
            call_count[0] += 1
            return f'value_{key}'

        # Time initial cache population
        start = time.time()
        for i in range(1000):
            cached_function(f'key_{i}')
        populate_elapsed = time.time() - start

        assert populate_elapsed < 1.0, f"Cache populate too slow: {populate_elapsed:.2f}s"

        # Time cache hits
        start = time.time()
        for i in range(1000):
            cached_function(f'key_{i}')  # These should be cache hits
        hit_elapsed = time.time() - start

        assert hit_elapsed < 0.5, f"Cache hit too slow: {hit_elapsed:.2f}s"

        # Verify caching worked (function called only 1000 times, not 2000)
        assert call_count[0] == 1000, f"Expected 1000 calls, got {call_count[0]}"


class TestAPIRateLimiterPerformance:
    """Benchmark tests for rate limiter."""

    @pytest.mark.benchmark
    def test_rate_limiter_check_performance(self):
        """Rate limiter checks should be fast even with many sessions."""
        from backend.api.v1.jobs import RateLimiter

        limiter = RateLimiter(window_seconds=60)

        # Add many sessions
        start = time.time()
        for i in range(1000):
            allowed, remaining = limiter.check_rate_limit(f'session_{i}', 10)
            assert allowed  # First request for each session should be allowed
        elapsed = time.time() - start

        assert elapsed < 0.5, f"Rate limiter too slow: {elapsed:.2f}s for 1000 sessions"

        # Check active session count
        assert limiter.active_session_count <= 1000

    @pytest.mark.benchmark
    def test_rate_limiter_cleanup_performance(self):
        """Rate limiter cleanup should not block."""
        from backend.api.v1.jobs import RateLimiter

        limiter = RateLimiter(window_seconds=1)  # 1 second window

        # Add sessions
        for i in range(100):
            limiter.check_rate_limit(f'session_{i}', 10)

        # Wait for window to expire
        time.sleep(1.1)

        # Check should trigger cleanup
        start = time.time()
        allowed, remaining = limiter.check_rate_limit('new_session', 10)
        elapsed = time.time() - start

        assert elapsed < 0.1, f"Cleanup too slow: {elapsed:.2f}s"
        assert allowed  # New session should be allowed
