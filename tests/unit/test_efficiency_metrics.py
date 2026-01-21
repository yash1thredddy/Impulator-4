"""
Tests for efficiency_metrics module.

Tests the following fixes:
- 3.3: Vectorized DataFrame operations (instead of iterrows)
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.modules.efficiency_metrics import (
    calculate_sei,
    calculate_bei,
    calculate_nsei,
    calculate_nbei,
    calculate_nbei_visualization,
    calculate_all_efficiency_metrics,
    calculate_efficiency_metrics_dataframe,
    validate_efficiency_metrics,
)


class TestIndividualMetrics:
    """Tests for individual metric calculation functions."""

    def test_calculate_sei(self):
        """Test SEI calculation: pActivity / (PSA / 100)."""
        # Normal case
        result = calculate_sei(pActivity=7.5, psa=85.2)
        expected = 7.5 / (85.2 / 100)  # = 8.80
        assert abs(result - expected) < 0.01

        # Zero PSA should return NaN
        result = calculate_sei(pActivity=7.5, psa=0)
        assert np.isnan(result)

        # NaN pActivity should return NaN
        result = calculate_sei(pActivity=np.nan, psa=85.2)
        assert np.isnan(result)

    def test_calculate_bei(self):
        """Test BEI calculation: pActivity / (MW / 1000)."""
        # Normal case
        result = calculate_bei(pActivity=7.5, molecular_weight=342.1)
        expected = 7.5 / (342.1 / 1000)  # = 21.93
        assert abs(result - expected) < 0.01

        # Zero MW should return NaN
        result = calculate_bei(pActivity=7.5, molecular_weight=0)
        assert np.isnan(result)

    def test_calculate_nsei(self):
        """Test NSEI calculation: pActivity / NPOL."""
        result = calculate_nsei(pActivity=7.5, npol=5)
        expected = 7.5 / 5  # = 1.5
        assert abs(result - expected) < 0.01

        # Zero NPOL should return NaN
        result = calculate_nsei(pActivity=7.5, npol=0)
        assert np.isnan(result)

    def test_calculate_nbei(self):
        """Test NBEI calculation: pActivity / Heavy_Atoms."""
        result = calculate_nbei(pActivity=7.5, heavy_atoms=24)
        expected = 7.5 / 24  # = 0.3125
        assert abs(result - expected) < 0.001

    def test_calculate_nbei_visualization(self):
        """Test nBEI_viz calculation: pActivity + log10(Heavy_Atoms)."""
        result = calculate_nbei_visualization(pActivity=7.5, heavy_atoms=24)
        expected = 7.5 + np.log10(24)  # = 7.5 + 1.38 = 8.88
        assert abs(result - expected) < 0.01


class TestAllMetricsFunction:
    """Tests for calculate_all_efficiency_metrics function."""

    def test_returns_all_metrics(self):
        """Test that all metrics are calculated correctly."""
        metrics = calculate_all_efficiency_metrics(
            pActivity=7.5,
            psa=85.2,
            molecular_weight=342.1,
            npol=5,
            heavy_atoms=24
        )

        assert 'SEI' in metrics
        assert 'BEI' in metrics
        assert 'NSEI' in metrics
        assert 'NBEI' in metrics
        assert 'nBEI_viz' in metrics

        # Verify values
        assert abs(metrics['SEI'] - 8.80) < 0.1
        assert abs(metrics['NBEI'] - 0.3125) < 0.01


class TestVectorizedDataFrame:
    """Tests for vectorized DataFrame efficiency metrics calculation (fix for 3.3)."""

    def test_basic_vectorized_calculation(self):
        """Test that vectorized calculation produces correct results."""
        df = pd.DataFrame({
            'pActivity': [7.5, 8.2, 6.8],
            'TPSA': [85.2, 92.1, 78.5],
            'Molecular_Weight': [342.1, 385.7, 298.3],
            'NPOL': [5, 6, 4],
            'Heavy_Atoms': [24, 27, 21]
        })

        result = calculate_efficiency_metrics_dataframe(df)

        # Check that all metric columns were added
        assert 'SEI' in result.columns
        assert 'BEI' in result.columns
        assert 'NSEI' in result.columns
        assert 'NBEI' in result.columns
        assert 'nBEI_viz' in result.columns

        # Verify first row calculations
        assert abs(result.loc[0, 'SEI'] - 7.5 / 0.852) < 0.1
        assert abs(result.loc[0, 'NBEI'] - 7.5 / 24) < 0.01

    def test_handles_nan_values(self):
        """Test that NaN values are handled correctly."""
        df = pd.DataFrame({
            'pActivity': [7.5, np.nan, 6.8],
            'TPSA': [85.2, 92.1, np.nan],
            'Molecular_Weight': [342.1, 385.7, 298.3],
            'NPOL': [5, 6, 4],
            'Heavy_Atoms': [24, 27, 21]
        })

        result = calculate_efficiency_metrics_dataframe(df)

        # Row with NaN pActivity should have NaN metrics
        assert np.isnan(result.loc[1, 'SEI'])
        assert np.isnan(result.loc[1, 'BEI'])

        # Row with NaN TPSA should have NaN SEI but valid other metrics
        assert np.isnan(result.loc[2, 'SEI'])
        assert not np.isnan(result.loc[2, 'BEI'])

    def test_handles_zero_values(self):
        """Test that zero values produce NaN (not infinity)."""
        df = pd.DataFrame({
            'pActivity': [7.5],
            'TPSA': [0],  # Zero - should produce NaN
            'Molecular_Weight': [342.1],
            'NPOL': [0],  # Zero - should produce NaN
            'Heavy_Atoms': [24]
        })

        result = calculate_efficiency_metrics_dataframe(df)

        assert np.isnan(result.loc[0, 'SEI'])  # Zero TPSA
        assert np.isnan(result.loc[0, 'NSEI'])  # Zero NPOL
        assert not np.isnan(result.loc[0, 'BEI'])  # Valid

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame({
            'pActivity': [7.5],
            'TPSA': [85.2],
            # Missing other required columns
        })

        with pytest.raises(ValueError) as excinfo:
            calculate_efficiency_metrics_dataframe(df)

        assert "Missing required columns" in str(excinfo.value)

    def test_preserves_existing_columns(self):
        """Test that existing DataFrame columns are preserved."""
        df = pd.DataFrame({
            'pActivity': [7.5],
            'TPSA': [85.2],
            'Molecular_Weight': [342.1],
            'NPOL': [5],
            'Heavy_Atoms': [24],
            'extra_column': ['preserved']
        })

        result = calculate_efficiency_metrics_dataframe(df)

        assert 'extra_column' in result.columns
        assert result.loc[0, 'extra_column'] == 'preserved'

    def test_does_not_modify_input(self):
        """Test that input DataFrame is not modified."""
        df = pd.DataFrame({
            'pActivity': [7.5],
            'TPSA': [85.2],
            'Molecular_Weight': [342.1],
            'NPOL': [5],
            'Heavy_Atoms': [24]
        })

        original_columns = set(df.columns)

        _ = calculate_efficiency_metrics_dataframe(df)

        # Original DataFrame should not have new columns
        assert set(df.columns) == original_columns

    def test_large_dataframe_performance(self):
        """Test that vectorized operations handle large DataFrames efficiently."""
        import time

        n_rows = 10000
        df = pd.DataFrame({
            'pActivity': np.random.uniform(5, 10, n_rows),
            'TPSA': np.random.uniform(50, 150, n_rows),
            'Molecular_Weight': np.random.uniform(200, 600, n_rows),
            'NPOL': np.random.randint(1, 10, n_rows),
            'Heavy_Atoms': np.random.randint(10, 50, n_rows)
        })

        start = time.time()
        result = calculate_efficiency_metrics_dataframe(df)
        elapsed = time.time() - start

        # Should complete in under 1 second for 10k rows (vectorized is fast)
        assert elapsed < 1.0, f"Took {elapsed:.2f}s for {n_rows} rows"
        assert len(result) == n_rows


class TestValidateMetrics:
    """Tests for validate_efficiency_metrics function."""

    def test_valid_metrics(self):
        """Test that valid metrics pass validation."""
        metrics = {
            'SEI': 8.0,
            'BEI': 20.0,
            'NSEI': 1.5,
            'NBEI': 0.3
        }
        assert validate_efficiency_metrics(metrics) is True

    def test_invalid_metrics_out_of_range(self):
        """Test that out-of-range metrics fail validation."""
        metrics = {
            'SEI': 8.0,
            'BEI': 150.0,  # Out of range (max 100)
            'NSEI': 1.5,
            'NBEI': 0.3
        }
        assert validate_efficiency_metrics(metrics) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
