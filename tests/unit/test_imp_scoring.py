"""
Tests for imp_scoring module.

Tests the following fixes:
- 3.2: Sigmoid Z-score normalization (preserves ranking for exceptional compounds)
- 3.11: Distance score clipping to 0-1 range
"""

import pytest
import numpy as np
import pandas as pd

from backend.modules.imp_scoring import (
    calculate_efficiency_outlier_score,
    calculate_angle_score,
    calculate_distance_to_best_score,
    calculate_interference_score,
    interpret_imp_score,
    calculate_imp_score,
    calculate_imp_score_phase1,
    WEIGHT_EFFICIENCY,
    WEIGHT_DISTANCE,
    WEIGHT_ANGLE,
    WEIGHT_INTERFERENCE,
    WEIGHT_PDB,
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
    """Tests for interpret_imp_score function."""

    def test_exceptional_imp(self):
        """Test exceptional IMP classification (0.9-1.0)."""
        result = interpret_imp_score(0.95)
        assert result['classification'] == 'Exceptional IMP'
        assert result['priority'] == 1

    def test_strong_imp(self):
        """Test strong IMP classification (0.7-0.9)."""
        result = interpret_imp_score(0.8)
        assert result['classification'] == 'Strong IMP'
        assert result['priority'] == 2

    def test_moderate_imp(self):
        """Test moderate IMP classification (0.5-0.7)."""
        result = interpret_imp_score(0.6)
        assert result['classification'] == 'Moderate IMP'
        assert result['priority'] == 3

    def test_weak_imp(self):
        """Test weak IMP classification (0.3-0.5)."""
        result = interpret_imp_score(0.4)
        assert result['classification'] == 'Weak IMP'
        assert result['priority'] == 4

    def test_not_imp(self):
        """Test not IMP classification (<0.3)."""
        result = interpret_imp_score(0.2)
        assert result['classification'] == 'Not IMP'
        assert result['priority'] is None

    def test_nan_score(self):
        """Test handling of NaN score."""
        result = interpret_imp_score(np.nan)
        assert result['classification'] == 'Invalid'
        assert result['priority'] is None


class TestIMPScoreNewWeights:
    """Tests for new IMP Score weight system (5 components, direct percentages)."""

    def test_imp_score_new_weights(self):
        """Verify new weight system: 45/20/15/15/5, no normalization."""
        df = pd.DataFrame({
            'SEI': [5.0, 10.0, 15.0],
            'BEI': [20.0, 25.0, 30.0],
            'NSEI': [1.0, 2.0, 3.0],
            'NBEI': [0.25, 0.35, 0.45],
            'Angle_SEI_BEI': [45.0, 45.0, 45.0],
            'Modulus_SEI_BEI': [20.0, 30.0, 40.0],
            'QED': [1.0, 1.0, 1.0],
            'SMILES': ['CCO', 'CCCO', 'CCCCO'],
            'PAINS_Violation': [0, 0, 0],
            'Aggregator_Risk': [0, 0, 0],
            'Redox_Reactive': [0, 0, 0],
            'Fluorescence_Interference': [0, 0, 0],
            'Thiol_Reactive': [0, 0, 0],
        })

        result = calculate_imp_score(df, use_pdb=False)

        assert 'Interference_Score' in result.columns
        assert 'Interference_Contribution' in result.columns
        assert 'IMP_Final_Score' in result.columns

        # Angle is optimal (45 deg) -> Angle_Score = 1.0
        # Angle_Contribution = 0.15 * 1.0 * 1.0 (QED=1) = 0.15
        for i, contrib in enumerate(result['Angle_Contribution']):
            assert abs(contrib - 0.15) < 0.01, \
                f"Row {i}: Angle_Contribution should be ~0.15, got {contrib}"

        # Interference is clean -> Interference_Score = 0.0
        # Interference_Contribution = 0.15 * 0.0 * 1.0 = 0.0
        for i, contrib in enumerate(result['Interference_Contribution']):
            assert abs(contrib - 0.0) < 0.01, \
                f"Row {i}: Interference_Contribution should be ~0.0, got {contrib}"

        # All scores in valid range
        for score in result['IMP_Final_Score']:
            assert 0.0 <= score <= 1.0


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

        result = calculate_imp_score_phase1(df)

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

        result = calculate_imp_score_phase1(df)

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

        result = calculate_imp_score_phase1(df)

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

        result = calculate_imp_score_phase1(df)

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

    def test_nsei_nbei_still_calculated_in_imp_score(self):
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

        result = calculate_imp_score_phase1(df)

        # NSEI and NBEI should still exist in the result dataframe
        assert 'NSEI' in result.columns, "NSEI should still be in the dataframe"
        assert 'NBEI' in result.columns, "NBEI should still be in the dataframe"

        # But they should not affect the efficiency score (verified in other tests)
        assert 'Efficiency_Score' in result.columns, "Efficiency_Score should be calculated"


class TestInterferenceScore:
    """Tests for calculate_interference_score function."""

    def test_all_scored_flags_triggered(self):
        """All 5 scored flags -> score = 1.0 (BRENK/NIH ignored)."""
        df = pd.DataFrame({
            'PAINS_Violation': [1],
            'Aggregator_Risk': [1],
            'Redox_Reactive': [1],
            'Fluorescence_Interference': [1],
            'Thiol_Reactive': [1],
            'BRENK_Alerts': [1],  # display-only, not counted
            'NIH_Alerts': [1],    # display-only, not counted
        })
        result = calculate_interference_score(df)
        assert abs(result['Interference_Score'].iloc[0] - 1.0) < 0.001

    def test_no_flags(self):
        """No flags -> score = 0.0."""
        df = pd.DataFrame({
            'PAINS_Violation': [0],
            'Aggregator_Risk': [0],
            'Redox_Reactive': [0],
            'Fluorescence_Interference': [0],
            'Thiol_Reactive': [0],
            'BRENK_Alerts': [0],
            'NIH_Alerts': [0],
        })
        result = calculate_interference_score(df)
        assert abs(result['Interference_Score'].iloc[0] - 0.0) < 0.001

    def test_two_flags(self):
        """2 of 5 scored flags -> score = 2/5 = 0.4."""
        df = pd.DataFrame({
            'PAINS_Violation': [1],
            'Aggregator_Risk': [1],
            'Redox_Reactive': [0],
            'Fluorescence_Interference': [0],
            'Thiol_Reactive': [0],
        })
        result = calculate_interference_score(df)
        assert abs(result['Interference_Score'].iloc[0] - 2.0/5.0) < 0.001

    def test_brenk_nih_not_counted(self):
        """BRENK and NIH should NOT affect interference score."""
        df_without = pd.DataFrame({
            'PAINS_Violation': [1],
            'Aggregator_Risk': [0],
            'Redox_Reactive': [0],
            'Fluorescence_Interference': [0],
            'Thiol_Reactive': [0],
            'BRENK_Alerts': [0],
            'NIH_Alerts': [0],
        })
        df_with = pd.DataFrame({
            'PAINS_Violation': [1],
            'Aggregator_Risk': [0],
            'Redox_Reactive': [0],
            'Fluorescence_Interference': [0],
            'Thiol_Reactive': [0],
            'BRENK_Alerts': [1],  # should not matter
            'NIH_Alerts': [1],    # should not matter
        })
        result_without = calculate_interference_score(df_without)
        result_with = calculate_interference_score(df_with)
        assert abs(result_without['Interference_Score'].iloc[0] - result_with['Interference_Score'].iloc[0]) < 0.001

    def test_missing_columns_defaults_to_zero(self):
        """Missing interference columns -> score = 0.0."""
        df = pd.DataFrame({'some_col': [1, 2, 3]})
        result = calculate_interference_score(df)
        assert all(result['Interference_Score'] == 0.0)

    def test_multiple_rows(self):
        """Test vectorized calculation across multiple rows."""
        df = pd.DataFrame({
            'PAINS_Violation': [1, 0, 1],
            'Aggregator_Risk': [0, 0, 1],
            'Redox_Reactive': [0, 0, 1],
            'Fluorescence_Interference': [0, 0, 1],
            'Thiol_Reactive': [0, 0, 1],
        })
        result = calculate_interference_score(df)
        assert abs(result['Interference_Score'].iloc[0] - 1.0/5.0) < 0.001  # 1 of 5
        assert abs(result['Interference_Score'].iloc[1] - 0.0) < 0.001      # 0 of 5
        assert abs(result['Interference_Score'].iloc[2] - 1.0) < 0.001      # 5 of 5


class TestNewWeights:
    """Tests for new weight system (5 components, no normalization)."""

    def test_weights_sum_to_100(self):
        """All 5 weights must sum to exactly 1.0."""
        total = WEIGHT_EFFICIENCY + WEIGHT_DISTANCE + WEIGHT_ANGLE + WEIGHT_INTERFERENCE + WEIGHT_PDB
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_individual_weights(self):
        """Verify each weight value."""
        assert abs(WEIGHT_EFFICIENCY - 0.45) < 0.001
        assert abs(WEIGHT_DISTANCE - 0.20) < 0.001
        assert abs(WEIGHT_ANGLE - 0.15) < 0.001
        assert abs(WEIGHT_INTERFERENCE - 0.15) < 0.001
        assert abs(WEIGHT_PDB - 0.05) < 0.001


class TestCalculateImpScore:
    """Tests for unified calculate_imp_score function."""

    def _make_test_df(self, interference_flags=None):
        """Helper to create test DataFrame with all required columns."""
        data = {
            'SEI': [5.0, 10.0, 15.0],
            'BEI': [20.0, 25.0, 30.0],
            'NSEI': [1.0, 2.0, 3.0],
            'NBEI': [0.25, 0.35, 0.45],
            'Angle_SEI_BEI': [45.0, 45.0, 45.0],
            'Modulus_SEI_BEI': [20.0, 30.0, 40.0],
            'QED': [1.0, 1.0, 1.0],
            'SMILES': ['CCO', 'CCCO', 'CCCCO'],
        }
        if interference_flags:
            data.update(interference_flags)
        else:
            for col in ['PAINS_Violation', 'Aggregator_Risk', 'Redox_Reactive',
                        'Fluorescence_Interference', 'Thiol_Reactive']:
                data[col] = [0, 0, 0]
        return pd.DataFrame(data)

    def test_has_interference_contribution(self):
        """New function should produce Interference_Score and Interference_Contribution columns."""
        df = self._make_test_df()
        result = calculate_imp_score(df, use_pdb=False)
        assert 'Interference_Score' in result.columns
        assert 'Interference_Contribution' in result.columns

    def test_interference_affects_final_score(self):
        """Compounds with interference flags should have higher IMP scores than clean ones."""
        df_clean = self._make_test_df()
        df_flagged = self._make_test_df(interference_flags={
            'PAINS_Violation': [1, 1, 1],
            'Aggregator_Risk': [1, 1, 1],
            'Redox_Reactive': [1, 1, 1],
            'Fluorescence_Interference': [0, 0, 0],
            'Thiol_Reactive': [0, 0, 0],
        })
        result_clean = calculate_imp_score(df_clean, use_pdb=False)
        result_flagged = calculate_imp_score(df_flagged, use_pdb=False)

        for i in range(len(result_clean)):
            assert result_flagged['IMP_Final_Score'].iloc[i] > result_clean['IMP_Final_Score'].iloc[i]

    def test_no_pdb_sets_pdb_zero(self):
        """With use_pdb=False, PDB_Score and PDB_Contribution should be 0."""
        df = self._make_test_df()
        result = calculate_imp_score(df, use_pdb=False)
        assert all(result['PDB_Score'] == 0.0)
        assert all(result['PDB_Contribution'] == 0.0)

    def test_scores_in_valid_range(self):
        """All final scores should be in [0, 1]."""
        df = self._make_test_df()
        result = calculate_imp_score(df, use_pdb=False)
        assert all(result['IMP_Final_Score'] >= 0.0)
        assert all(result['IMP_Final_Score'] <= 1.0)

    def test_qed_multiplier_still_works(self):
        """QED multiplier formula should still be 0.75 + 0.25 * QED."""
        df = self._make_test_df()
        df['QED'] = [0.0, 0.5, 1.0]
        result = calculate_imp_score(df, use_pdb=False)
        assert abs(result['QED_Multiplier'].iloc[0] - 0.75) < 0.001
        assert abs(result['QED_Multiplier'].iloc[1] - 0.875) < 0.001
        assert abs(result['QED_Multiplier'].iloc[2] - 1.0) < 0.001

    def test_base_score_formula(self):
        """Base score should equal sum of weighted components (verifiable with known inputs)."""
        df = self._make_test_df()
        df['QED'] = [1.0, 1.0, 1.0]
        result = calculate_imp_score(df, use_pdb=False)

        for i in range(len(result)):
            expected = (
                0.45 * result['Efficiency_Score'].iloc[i] +
                0.20 * result['Distance_Score'].iloc[i] +
                0.15 * result['Angle_Score'].iloc[i] +
                0.15 * result['Interference_Score'].iloc[i] +
                0.05 * result['PDB_Score'].iloc[i]
            )
            assert abs(result['IMP_Base_Score'].iloc[i] - expected) < 0.001


class TestEfficiencyOutlierScoreMissingMetrics:
    """Tests for missing metric column validation."""

    def test_missing_sei_raises(self):
        df = pd.DataFrame({'BEI': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing efficiency metrics"):
            calculate_efficiency_outlier_score(df, metrics=['SEI', 'BEI'])

    def test_missing_bei_raises(self):
        df = pd.DataFrame({'SEI': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing efficiency metrics"):
            calculate_efficiency_outlier_score(df, metrics=['SEI', 'BEI'])


class TestCalculateImpScoreMissingColumns:
    """Tests for missing required columns in calculate_imp_score."""

    def test_missing_smiles_raises(self):
        df = pd.DataFrame({
            'SEI': [10], 'BEI': [25], 'NSEI': [2], 'NBEI': [0.35],
            'Angle_SEI_BEI': [45], 'Modulus_SEI_BEI': [30], 'QED': [1.0],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_imp_score(df, use_pdb=False)

    def test_missing_all_raises(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_imp_score(df, use_pdb=False)

    def test_phase1_missing_columns_raises(self):
        df = pd.DataFrame({'SEI': [10], 'BEI': [25]})
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_imp_score_phase1(df)


class TestAddImpScoreInterpretation:
    """Tests for add_imp_score_interpretation."""

    def test_adds_classification_column(self):
        from backend.modules.imp_scoring import add_imp_score_interpretation
        df = pd.DataFrame({'IMP_Final_Score': [0.95, 0.75, 0.55, 0.35, 0.15]})
        result = add_imp_score_interpretation(df)
        assert 'IMP_Classification' in result.columns
        assert 'IMP_Priority' in result.columns
        assert result['IMP_Classification'].iloc[0] == 'Exceptional IMP'
        assert result['IMP_Classification'].iloc[4] == 'Not IMP'

    def test_missing_score_raises(self):
        from backend.modules.imp_scoring import add_imp_score_interpretation
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(ValueError, match="IMP_Final_Score column not found"):
            add_imp_score_interpretation(df)

    def test_nan_scores_classified_invalid(self):
        from backend.modules.imp_scoring import add_imp_score_interpretation
        df = pd.DataFrame({'IMP_Final_Score': [np.nan]})
        result = add_imp_score_interpretation(df)
        assert result['IMP_Classification'].iloc[0] == 'Invalid'


class TestGetImpScoreSummary:
    """Tests for get_imp_score_summary."""

    def test_no_scores_returns_error(self):
        from backend.modules.imp_scoring import get_imp_score_summary
        df = pd.DataFrame({'x': [1]})
        summary = get_imp_score_summary(df)
        assert 'error' in summary

    def test_basic_summary(self):
        from backend.modules.imp_scoring import get_imp_score_summary
        df = pd.DataFrame({'IMP_Final_Score': [0.9, 0.7, 0.5, 0.3, 0.1]})
        summary = get_imp_score_summary(df)
        assert summary['total_compounds'] == 5
        assert summary['scored_compounds'] == 5
        assert summary['mean_score'] == pytest.approx(0.5)
        assert summary['min_score'] == pytest.approx(0.1)
        assert summary['max_score'] == pytest.approx(0.9)

    def test_with_classifications(self):
        from backend.modules.imp_scoring import get_imp_score_summary
        df = pd.DataFrame({
            'IMP_Final_Score': [0.95, 0.75, 0.55],
            'IMP_Classification': ['Exceptional IMP', 'Strong IMP', 'Moderate IMP'],
        })
        summary = get_imp_score_summary(df)
        assert summary['exceptional_imps'] == 1
        assert summary['strong_imps'] == 1
        assert summary['moderate_imps'] == 1

    def test_with_priorities(self):
        from backend.modules.imp_scoring import get_imp_score_summary
        df = pd.DataFrame({
            'IMP_Final_Score': [0.95, 0.75],
            'IMP_Priority': [1, 2],
        })
        summary = get_imp_score_summary(df)
        assert 'priority_counts' in summary

    def test_empty_scores(self):
        from backend.modules.imp_scoring import get_imp_score_summary
        df = pd.DataFrame({'IMP_Final_Score': pd.Series([], dtype=float)})
        summary = get_imp_score_summary(df)
        assert summary['scored_compounds'] == 0


class TestCreatePdbSummary:
    """Tests for create_pdb_summary (pure DataFrame operation)."""

    def test_basic_summary(self):
        from backend.modules.imp_scoring import create_pdb_summary
        df = pd.DataFrame({
            'ChEMBL_ID': ['C1', 'C2'],
            'Molecule_Name': ['A', 'B'],
            'SMILES': ['CCO', 'CCCO'],
            'PDB_Score': [0.8, 0.3],
            'PDB_Num_Structures': [5, 2],
            'PDB_High_Quality': [3, 0],
            'PDB_Medium_Quality': [2, 1],
            'PDB_Poor_Quality': [0, 1],
            'PDB_IDs': ['1ABC,2DEF', '3GHI'],
            'PDB_Best_Resolution': [1.5, 3.0],
        })
        result = create_pdb_summary(df)
        assert len(result) == 2
        assert result.iloc[0]['PDB_Score'] == 0.8  # Sorted descending
        assert 'PDB_High_Quality_Pct' in result.columns

    def test_zero_structures_safe_division(self):
        from backend.modules.imp_scoring import create_pdb_summary
        df = pd.DataFrame({
            'ChEMBL_ID': ['C1'],
            'Molecule_Name': ['A'],
            'SMILES': ['CCO'],
            'PDB_Score': [0.0],
            'PDB_Num_Structures': [0],
            'PDB_High_Quality': [0],
            'PDB_Medium_Quality': [0],
            'PDB_Poor_Quality': [0],
            'PDB_IDs': [''],
            'PDB_Best_Resolution': [np.nan],
        })
        result = create_pdb_summary(df)
        assert result['PDB_High_Quality_Pct'].iloc[0] == 0.0

    def test_missing_pdb_columns_returns_empty(self):
        from backend.modules.imp_scoring import create_pdb_summary
        df = pd.DataFrame({'x': [1]})
        result = create_pdb_summary(df)
        assert result.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
