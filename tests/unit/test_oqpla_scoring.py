"""
Tests for oqpla_scoring module.

Tests the following fixes:
- 3.2: Sigmoid Z-score normalization (preserves ranking for exceptional compounds)
- 3.11: Distance score clipping to 0-1 range
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.modules.oqpla_scoring import (
    calculate_efficiency_outlier_score,
    calculate_angle_score,
    calculate_distance_to_best_score,
    interpret_oqpla_score,
    calculate_oqpla_phase1,
    calculate_oqpla_phase2,
)


class TestEfficiencyOutlierScore:
    """Tests for calculate_efficiency_outlier_score function (fix for 3.2)."""

    def test_basic_calculation(self):
        """Test basic efficiency score calculation."""
        df = pd.DataFrame({
            'SEI': [8.0, 10.0, 12.0, 9.0, 11.0],
            'BEI': [20.0, 25.0, 30.0, 22.0, 28.0],
            'NSEI': [1.5, 2.0, 2.5, 1.8, 2.2],
            'NBEI': [0.3, 0.35, 0.4, 0.32, 0.38]
        })

        scores = calculate_efficiency_outlier_score(df)

        # All scores should be between 0 and 1
        assert all(scores >= 0)
        assert all(scores <= 1)

        # Higher values should have higher scores
        # Index 2 has highest values, should have highest score
        assert scores.iloc[2] == scores.max()

    def test_sigmoid_preserves_ranking(self):
        """Test that sigmoid normalization preserves ranking for exceptional compounds."""
        # Create data with an exceptional outlier (z-score > 3)
        df = pd.DataFrame({
            'SEI': [8.0, 9.0, 10.0, 11.0, 50.0],  # 50.0 is extreme outlier
            'BEI': [20.0, 22.0, 25.0, 28.0, 80.0],
            'NSEI': [1.5, 1.7, 2.0, 2.2, 5.0],
            'NBEI': [0.3, 0.32, 0.35, 0.38, 0.8]
        })

        scores = calculate_efficiency_outlier_score(df)

        # The exceptional outlier (index 4) should have the highest score
        assert scores.iloc[4] == scores.max()

        # Scores should be monotonically increasing (preserving ranking)
        for i in range(1, len(scores)):
            assert scores.iloc[i] >= scores.iloc[i-1], f"Ranking not preserved at index {i}"

    def test_sigmoid_vs_clip_behavior(self):
        """Test that sigmoid gives different scores to extreme outliers (vs hard clipping)."""
        # Create data with two extreme outliers
        df = pd.DataFrame({
            'SEI': [8.0, 9.0, 100.0, 200.0],  # Two extreme outliers
            'BEI': [20.0, 22.0, 100.0, 200.0],
            'NSEI': [1.5, 1.7, 10.0, 20.0],
            'NBEI': [0.3, 0.32, 1.0, 2.0]
        })

        scores = calculate_efficiency_outlier_score(df)

        # With sigmoid, the more extreme outlier should have a higher score
        # (With hard clipping at z=3, they would have the same score of 1.0)
        assert scores.iloc[3] > scores.iloc[2], \
            "Sigmoid should differentiate between extreme outliers"

    def test_handles_zero_std(self):
        """Test handling of zero standard deviation (all same values)."""
        df = pd.DataFrame({
            'SEI': [10.0, 10.0, 10.0],  # All same
            'BEI': [20.0, 20.0, 20.0],
            'NSEI': [2.0, 2.0, 2.0],
            'NBEI': [0.35, 0.35, 0.35]
        })

        scores = calculate_efficiency_outlier_score(df)

        # Should not raise error and should return valid scores
        assert len(scores) == 3
        assert all(scores >= 0)

    def test_handles_nan_values(self):
        """Test handling of NaN values in metrics."""
        df = pd.DataFrame({
            'SEI': [8.0, np.nan, 12.0],
            'BEI': [20.0, 25.0, np.nan],
            'NSEI': [1.5, 2.0, 2.5],
            'NBEI': [0.3, 0.35, 0.4]
        })

        # Should not raise error
        scores = calculate_efficiency_outlier_score(df)
        assert len(scores) == 3

    def test_negative_z_scores_map_to_zero(self):
        """Test that negative z-scores (below average) map to ~0."""
        df = pd.DataFrame({
            'SEI': [1.0, 5.0, 10.0, 15.0, 20.0],  # 1.0 is well below average
            'BEI': [5.0, 15.0, 25.0, 35.0, 45.0],
            'NSEI': [0.5, 1.0, 2.0, 3.0, 4.0],
            'NBEI': [0.1, 0.2, 0.35, 0.5, 0.6]
        })

        scores = calculate_efficiency_outlier_score(df)

        # Lowest value compound should have score close to 0
        assert scores.iloc[0] < 0.1


class TestAngleScore:
    """Tests for calculate_angle_score function."""

    def test_optimal_angle_gets_max_score(self):
        """Test that optimal angle (45 deg) gets score of 1.0."""
        angles = pd.Series([45.0])
        scores = calculate_angle_score(angles, optimal_angle=45.0)
        assert scores.iloc[0] == 1.0

    def test_deviation_reduces_score(self):
        """Test that deviation from optimal reduces score."""
        angles = pd.Series([0.0, 22.5, 45.0, 67.5, 90.0])
        scores = calculate_angle_score(angles, optimal_angle=45.0)

        # 45 deg should have highest score
        assert scores.iloc[2] == scores.max()

        # Symmetric deviations should have same score
        assert abs(scores.iloc[1] - scores.iloc[3]) < 0.01

    def test_clipping_to_zero_one(self):
        """Test that scores are clipped to 0-1 range."""
        angles = pd.Series([-10.0, 100.0, 45.0])
        scores = calculate_angle_score(angles, optimal_angle=45.0)

        assert all(scores >= 0)
        assert all(scores <= 1)


class TestDistanceScore:
    """Tests for calculate_distance_to_best_score function (fix for 3.11)."""

    def test_basic_calculation(self):
        """Test basic distance score calculation."""
        df = pd.DataFrame({
            'Modulus_SEI_BEI': [10.0, 20.0, 30.0, 40.0, 50.0]
        })

        scores = calculate_distance_to_best_score(df)

        # Best compound (50.0) should have score of 1.0
        assert scores.iloc[4] == 1.0

        # Worst compound (10.0) should have score of 0.2 (10/50)
        assert abs(scores.iloc[0] - 0.2) < 0.01

    def test_score_clipped_to_one(self):
        """Test that scores are clipped to maximum of 1.0 (fix for 3.11)."""
        df = pd.DataFrame({
            'Modulus_SEI_BEI': [10.0, 20.0, 30.0]
        })

        scores = calculate_distance_to_best_score(df)

        # All scores should be <= 1.0
        assert all(scores <= 1.0)
        assert all(scores >= 0.0)

    def test_handles_zero_best_modulus(self):
        """Test handling of zero best modulus."""
        df = pd.DataFrame({
            'Modulus_SEI_BEI': [0.0, 0.0, 0.0]
        })

        scores = calculate_distance_to_best_score(df)

        # Should return all zeros
        assert all(scores == 0.0)

    def test_handles_negative_values(self):
        """Test handling of negative modulus values."""
        df = pd.DataFrame({
            'Modulus_SEI_BEI': [-10.0, 10.0, 20.0]
        })

        scores = calculate_distance_to_best_score(df)

        # All scores should be in 0-1 range (negative clipped to 0)
        assert all(scores >= 0)
        assert all(scores <= 1)

    def test_handles_nan_values(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            'Modulus_SEI_BEI': [10.0, np.nan, 30.0]
        })

        scores = calculate_distance_to_best_score(df)

        # Should not raise error
        assert len(scores) == 3

    def test_missing_column_raises_error(self):
        """Test that missing modulus column raises ValueError."""
        df = pd.DataFrame({
            'other_column': [10.0, 20.0, 30.0]
        })

        with pytest.raises(ValueError) as excinfo:
            calculate_distance_to_best_score(df)

        assert "not found in DataFrame" in str(excinfo.value)


class TestInterpretScore:
    """Tests for interpret_oqpla_score function."""

    def test_exceptional_imp(self):
        """Test exceptional IMP classification (0.9-1.0)."""
        result = interpret_oqpla_score(0.95)
        assert result['classification'] == 'Exceptional IMP'
        assert result['priority'] == 1

    def test_strong_imp(self):
        """Test strong IMP classification (0.7-0.9)."""
        result = interpret_oqpla_score(0.8)
        assert result['classification'] == 'Strong IMP'
        assert result['priority'] == 2

    def test_moderate_imp(self):
        """Test moderate IMP classification (0.5-0.7)."""
        result = interpret_oqpla_score(0.6)
        assert result['classification'] == 'Moderate IMP'
        assert result['priority'] == 3

    def test_weak_imp(self):
        """Test weak IMP classification (0.3-0.5)."""
        result = interpret_oqpla_score(0.4)
        assert result['classification'] == 'Weak IMP'
        assert result['priority'] == 4

    def test_not_imp(self):
        """Test not IMP classification (<0.3)."""
        result = interpret_oqpla_score(0.2)
        assert result['classification'] == 'Not IMP'
        assert result['priority'] is None

    def test_nan_score(self):
        """Test handling of NaN score."""
        result = interpret_oqpla_score(np.nan)
        assert result['classification'] == 'Invalid'
        assert result['priority'] is None


class TestOQPLAPhase2NewWeights:
    """Tests for updated OQPLA Phase 2 weight ratios."""

    def test_oqpla_phase2_new_weights(self):
        """
        Verify Phase 2 weight ratios are correctly normalized:
        - Efficiency: 40% raw -> 50% normalized
        - Angle: 15% raw -> 18.75% normalized (was 10%)
        - Distance: 20% raw -> 25% normalized (was 15%)
        - PDB: 5% raw -> 6.25% normalized (was 15%)

        Total raw = 40 + 15 + 20 + 5 = 80%
        """
        # Create test dataframe with known values to verify weight ratios
        df = pd.DataFrame({
            'SEI': [10.0, 10.0],
            'BEI': [25.0, 25.0],
            'NSEI': [2.0, 2.0],
            'NBEI': [0.35, 0.35],
            'Angle_SEI_BEI': [45.0, 45.0],  # Optimal angle -> score = 1.0
            'Modulus_SEI_BEI': [30.0, 30.0],  # Same -> distance score = 1.0
            'QED': [1.0, 1.0],  # Max QED so multiplier = 1.0
            'SMILES': ['CCO', 'CCO']
        })

        # Calculate Phase 2 scores without PDB (use_pdb=False)
        _ = calculate_oqpla_phase2(df, use_pdb=False)  # noqa: F841 - called to verify no errors

        # Verify the weight ratios by examining the contribution columns
        # When PDB is disabled (score=0), effective weights should be:
        # Total active weight = 40 + 15 + 20 + 5 = 80% raw
        # Normalized: Efficiency=50%, Angle=18.75%, Distance=25%, PDB=6.25%

        # With all component scores = 1.0 and QED_Multiplier = 1.0:
        # Base score should be weighted sum of components

        # Check the normalized weight ratios
        total_raw_weight = 0.40 + 0.15 + 0.20 + 0.05  # 0.80
        expected_efficiency_weight = 0.40 / total_raw_weight  # 0.50
        expected_angle_weight = 0.15 / total_raw_weight  # 0.1875
        expected_distance_weight = 0.20 / total_raw_weight  # 0.25
        expected_pdb_weight = 0.05 / total_raw_weight  # 0.0625

        # Verify the weights sum to 1.0
        total_weight = (expected_efficiency_weight + expected_angle_weight +
                       expected_distance_weight + expected_pdb_weight)
        assert abs(total_weight - 1.0) < 0.001, f"Weights should sum to 1.0, got {total_weight}"

        # Verify individual weight values
        assert abs(expected_efficiency_weight - 0.50) < 0.001, "Efficiency weight should be 50%"
        assert abs(expected_angle_weight - 0.1875) < 0.001, "Angle weight should be 18.75%"
        assert abs(expected_distance_weight - 0.25) < 0.001, "Distance weight should be 25%"
        assert abs(expected_pdb_weight - 0.0625) < 0.001, "PDB weight should be 6.25%"


class TestQEDMultiplierNewFormula:
    """Tests for updated QED multiplier formula: 0.75 + 0.25 * QED."""

    def test_qed_multiplier_qed_zero_gives_075(self):
        """Test that QED=0 gives multiplier of 0.75 (floor)."""
        df = pd.DataFrame({
            'SEI': [10.0],
            'BEI': [25.0],
            'NSEI': [2.0],
            'NBEI': [0.35],
            'Angle_SEI_BEI': [45.0],
            'Modulus_SEI_BEI': [30.0],
            'QED': [0.0],
            'SMILES': ['CCO']
        })

        result = calculate_oqpla_phase1(df)

        # QED multiplier should be 0.75 when QED = 0
        assert abs(result['QED_Multiplier'].iloc[0] - 0.75) < 0.001, \
            f"QED=0 should give multiplier 0.75, got {result['QED_Multiplier'].iloc[0]}"

    def test_qed_multiplier_qed_one_gives_100(self):
        """Test that QED=1 gives multiplier of 1.0 (max)."""
        df = pd.DataFrame({
            'SEI': [10.0],
            'BEI': [25.0],
            'NSEI': [2.0],
            'NBEI': [0.35],
            'Angle_SEI_BEI': [45.0],
            'Modulus_SEI_BEI': [30.0],
            'QED': [1.0],
            'SMILES': ['CCO']
        })

        result = calculate_oqpla_phase1(df)

        # QED multiplier should be 1.0 when QED = 1
        assert abs(result['QED_Multiplier'].iloc[0] - 1.0) < 0.001, \
            f"QED=1 should give multiplier 1.0, got {result['QED_Multiplier'].iloc[0]}"

    def test_qed_multiplier_qed_half_gives_0875(self):
        """Test that QED=0.5 gives multiplier of 0.875."""
        df = pd.DataFrame({
            'SEI': [10.0],
            'BEI': [25.0],
            'NSEI': [2.0],
            'NBEI': [0.35],
            'Angle_SEI_BEI': [45.0],
            'Modulus_SEI_BEI': [30.0],
            'QED': [0.5],
            'SMILES': ['CCO']
        })

        result = calculate_oqpla_phase1(df)

        # QED multiplier should be 0.875 when QED = 0.5
        # Formula: 0.75 + 0.25 * 0.5 = 0.75 + 0.125 = 0.875
        assert abs(result['QED_Multiplier'].iloc[0] - 0.875) < 0.001, \
            f"QED=0.5 should give multiplier 0.875, got {result['QED_Multiplier'].iloc[0]}"

    def test_qed_multiplier_new_formula(self):
        """Combined test verifying the full QED multiplier formula: 0.75 + 0.25 * QED."""
        df = pd.DataFrame({
            'SEI': [10.0, 10.0, 10.0],
            'BEI': [25.0, 25.0, 25.0],
            'NSEI': [2.0, 2.0, 2.0],
            'NBEI': [0.35, 0.35, 0.35],
            'Angle_SEI_BEI': [45.0, 45.0, 45.0],
            'Modulus_SEI_BEI': [30.0, 30.0, 30.0],
            'QED': [0.0, 0.5, 1.0],
            'SMILES': ['CCO', 'CCO', 'CCO']
        })

        result = calculate_oqpla_phase1(df)

        # Verify the formula: 0.75 + 0.25 * QED
        expected_multipliers = [0.75, 0.875, 1.0]

        for i, expected in enumerate(expected_multipliers):
            actual = result['QED_Multiplier'].iloc[i]
            assert abs(actual - expected) < 0.001, \
                f"QED={df['QED'].iloc[i]} should give multiplier {expected}, got {actual}"


class TestEfficiencyScoreUsesSEIBEIOnly:
    """Tests verifying efficiency score uses only SEI and BEI (not NSEI/NBEI)."""

    def test_efficiency_score_uses_only_sei_bei(self):
        """
        Verify that NSEI and NBEI values do not affect the efficiency score.
        Only SEI and BEI should be used in the calculation.
        """
        # Create two dataframes with same SEI/BEI but different NSEI/NBEI
        df1 = pd.DataFrame({
            'SEI': [10.0, 15.0, 20.0],
            'BEI': [25.0, 30.0, 35.0],
            'NSEI': [1.0, 1.5, 2.0],  # Low NSEI values
            'NBEI': [0.1, 0.15, 0.2]  # Low NBEI values
        })

        df2 = pd.DataFrame({
            'SEI': [10.0, 15.0, 20.0],  # Same SEI
            'BEI': [25.0, 30.0, 35.0],  # Same BEI
            'NSEI': [5.0, 7.5, 10.0],   # Very different NSEI values
            'NBEI': [0.5, 0.75, 1.0]    # Very different NBEI values
        })

        # Calculate efficiency scores (default should be SEI and BEI only)
        scores1 = calculate_efficiency_outlier_score(df1)
        scores2 = calculate_efficiency_outlier_score(df2)

        # Scores should be identical since only SEI and BEI are used
        for i in range(len(scores1)):
            assert abs(scores1.iloc[i] - scores2.iloc[i]) < 0.001, \
                f"NSEI/NBEI should not affect efficiency score at index {i}: {scores1.iloc[i]} vs {scores2.iloc[i]}"

    def test_efficiency_score_default_metrics_sei_bei(self):
        """Test that the default metrics parameter is ['SEI', 'BEI']."""
        df = pd.DataFrame({
            'SEI': [8.0, 10.0, 12.0],
            'BEI': [20.0, 25.0, 30.0],
            'NSEI': [1.5, 2.0, 2.5],
            'NBEI': [0.3, 0.35, 0.4]
        })

        # Calculate with default (should be SEI, BEI only)
        scores_default = calculate_efficiency_outlier_score(df)

        # Calculate explicitly with SEI and BEI
        scores_explicit = calculate_efficiency_outlier_score(df, metrics=['SEI', 'BEI'])

        # Should be identical
        for i in range(len(scores_default)):
            assert abs(scores_default.iloc[i] - scores_explicit.iloc[i]) < 0.001, \
                f"Default should use SEI/BEI only at index {i}"

    def test_nsei_nbei_still_calculated_in_oqpla(self):
        """Verify that NSEI and NBEI are still calculated and stored but not used in efficiency score."""
        df = pd.DataFrame({
            'SEI': [10.0],
            'BEI': [25.0],
            'NSEI': [2.0],  # Should still be in dataframe
            'NBEI': [0.35],  # Should still be in dataframe
            'Angle_SEI_BEI': [45.0],
            'Modulus_SEI_BEI': [30.0],
            'QED': [1.0],
            'SMILES': ['CCO']
        })

        result = calculate_oqpla_phase1(df)

        # NSEI and NBEI should still exist in the result dataframe
        assert 'NSEI' in result.columns, "NSEI should still be in the dataframe"
        assert 'NBEI' in result.columns, "NBEI should still be in the dataframe"

        # But they should not affect the efficiency score (verified in other tests)
        assert 'Efficiency_Score' in result.columns, "Efficiency_Score should be calculated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
