"""
Integration tests for IMP Score scoring system.

Tests the complete IMP Score pipeline with weights (sum to 100%):
- Efficiency: 45%
- Distance: 20%
- Angle: 15%
- Interference: 15%
- PDB: 5%
- QED Multiplier: 0.75 + 0.25 * QED
"""

import pytest
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.modules.imp_scoring import (
    calculate_imp_score_phase1,
    calculate_imp_score_phase2,
    get_imp_score_breakdown,
    interpret_imp_score,
    WEIGHT_EFFICIENCY,
    WEIGHT_DISTANCE,
    WEIGHT_ANGLE,
    WEIGHT_INTERFERENCE,
    WEIGHT_PDB,
    QED_MULTIPLIER_FLOOR,
    QED_MULTIPLIER_SCALE,
)


class TestIMPScoreWeightConstants:
    """Test that weight constants are correct (5 components, direct percentages)."""

    def test_weights_sum_to_100_percent(self):
        """Verify all 5 weights sum to 100%."""
        total = WEIGHT_EFFICIENCY + WEIGHT_DISTANCE + WEIGHT_ANGLE + WEIGHT_INTERFERENCE + WEIGHT_PDB
        assert abs(total - 1.0) < 0.001, f"Weights should sum to 100%, got {total * 100}%"

    def test_individual_weights(self):
        """Verify individual weight values."""
        assert abs(WEIGHT_EFFICIENCY - 0.45) < 0.001, "Efficiency should be 45%"
        assert abs(WEIGHT_DISTANCE - 0.20) < 0.001, "Distance should be 20%"
        assert abs(WEIGHT_ANGLE - 0.15) < 0.001, "Angle should be 15%"
        assert abs(WEIGHT_INTERFERENCE - 0.15) < 0.001, "Interference should be 15%"
        assert abs(WEIGHT_PDB - 0.05) < 0.001, "PDB should be 5%"

    def test_qed_multiplier_constants(self):
        """Verify QED multiplier formula constants."""
        assert abs(QED_MULTIPLIER_FLOOR - 0.75) < 0.001, "QED floor should be 75%"
        assert abs(QED_MULTIPLIER_SCALE - 0.25) < 0.001, "QED scale should be 25%"


class TestIMPScoreFullPipeline:
    """Integration test for complete IMP Score scoring pipeline."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create test DataFrame with all required columns."""
        return pd.DataFrame({
            'Molecule_Name': ['TestDrug'],
            'SMILES': ['CCO'],
            'pActivity': [7.0],
            'SEI': [15.0],
            'BEI': [20.0],
            'NSEI': [1.5],
            'NBEI': [0.35],
            'Angle_SEI_BEI': [45.0],  # Optimal angle
            'Modulus_SEI_BEI': [25.0],
            'QED': [0.75],
            'PSA': [46.5],
            'MW': [350.0],
        })

    def test_full_imp_score_pipeline(self, sample_dataframe):
        """Integration test for complete IMP Score scoring pipeline."""
        # Run Phase 2 without actual PDB calls
        result = calculate_imp_score_phase2(sample_dataframe, use_pdb=False)

        # Verify all expected columns exist
        expected_cols = [
            'Efficiency_Score', 'Angle_Score', 'Distance_Score', 'PDB_Score',
            'Efficiency_Contribution', 'Angle_Contribution',
            'Distance_Contribution', 'PDB_Contribution',
            'IMP_Base_Score', 'QED_Multiplier', 'IMP_Final_Score',
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Verify score is in valid range
        assert 0 <= result['IMP_Final_Score'].iloc[0] <= 1, "Final score out of range"

    def test_contributions_sum_to_final_score(self, sample_dataframe):
        """Verify contributions sum correctly to final score (contributions include QED multiplier)."""
        result = calculate_imp_score_phase2(sample_dataframe, use_pdb=False)

        # Contributions include QED multiplier, so they sum to FINAL score
        total_contrib = (
            result['Efficiency_Contribution'].iloc[0] +
            result['Angle_Contribution'].iloc[0] +
            result['Distance_Contribution'].iloc[0] +
            result['PDB_Contribution'].iloc[0]
        )

        assert abs(total_contrib - result['IMP_Final_Score'].iloc[0]) < 0.001, \
            f"Contributions should sum to final score: {total_contrib} vs {result['IMP_Final_Score'].iloc[0]}"

    def test_qed_multiplier_formula(self, sample_dataframe):
        """Verify QED multiplier uses correct formula: 0.75 + 0.25 * QED."""
        result = calculate_imp_score_phase2(sample_dataframe, use_pdb=False)

        # With QED=0.75: 0.75 + 0.25 * 0.75 = 0.9375
        expected_mult = 0.75 + 0.25 * 0.75
        assert abs(result['QED_Multiplier'].iloc[0] - expected_mult) < 0.001, \
            f"QED multiplier should be {expected_mult}, got {result['QED_Multiplier'].iloc[0]}"

    def test_final_score_equals_base_times_qed(self, sample_dataframe):
        """Verify final score = base score * QED multiplier."""
        result = calculate_imp_score_phase2(sample_dataframe, use_pdb=False)

        expected_final = result['IMP_Base_Score'].iloc[0] * result['QED_Multiplier'].iloc[0]
        assert abs(result['IMP_Final_Score'].iloc[0] - expected_final) < 0.001, \
            "Final score should equal base * QED multiplier"

    def test_score_breakdown_function(self, sample_dataframe):
        """Test score breakdown helper function."""
        result = calculate_imp_score_phase2(sample_dataframe, use_pdb=False)
        breakdown = get_imp_score_breakdown(result.iloc[0])

        # Verify structure
        assert 'efficiency_metrics' in breakdown
        assert 'plane_geometry' in breakdown
        assert 'component_scores' in breakdown
        assert 'final_calculation' in breakdown
        assert 'pdb_details' in breakdown

        # Verify efficiency metrics structure
        assert 'SEI' in breakdown['efficiency_metrics']
        assert 'BEI' in breakdown['efficiency_metrics']
        assert breakdown['efficiency_metrics']['SEI']['used_in_score'] is True
        assert breakdown['efficiency_metrics']['BEI']['used_in_score'] is True
        assert breakdown['efficiency_metrics']['NSEI']['used_in_score'] is False
        assert breakdown['efficiency_metrics']['NBEI']['used_in_score'] is False

        # Verify component weights (direct percentages, no normalization)
        assert breakdown['component_scores']['efficiency']['weight'] == '45%'
        assert breakdown['component_scores']['distance']['weight'] == '20%'
        assert breakdown['component_scores']['angle']['weight'] == '15%'
        assert breakdown['component_scores']['interference']['weight'] == '15%'
        assert breakdown['component_scores']['pdb']['weight'] == '5%'


class TestDirectWeights:
    """Test that direct weights sum to 100%."""

    def test_direct_weights(self):
        """Verify direct weights: 45/20/15/15/5 = 100%."""
        assert abs(WEIGHT_EFFICIENCY - 0.45) < 0.001
        assert abs(WEIGHT_DISTANCE - 0.20) < 0.001
        assert abs(WEIGHT_ANGLE - 0.15) < 0.001
        assert abs(WEIGHT_INTERFERENCE - 0.15) < 0.001
        assert abs(WEIGHT_PDB - 0.05) < 0.001

        total = WEIGHT_EFFICIENCY + WEIGHT_DISTANCE + WEIGHT_ANGLE + WEIGHT_INTERFERENCE + WEIGHT_PDB
        assert abs(total - 1.0) < 0.001, f"Weights should sum to 100%, got {total * 100}%"


class TestQEDMultiplierRange:
    """Test QED multiplier across full range."""

    @pytest.fixture
    def base_dataframe(self):
        """Base DataFrame for QED testing."""
        return pd.DataFrame({
            'SEI': [10.0],
            'BEI': [25.0],
            'NSEI': [2.0],
            'NBEI': [0.35],
            'Angle_SEI_BEI': [45.0],
            'Modulus_SEI_BEI': [30.0],
            'QED': [0.0],  # Will be overwritten
            'SMILES': ['CCO']
        })

    def test_qed_zero_gives_75_percent(self, base_dataframe):
        """Test QED=0 gives multiplier of 0.75."""
        base_dataframe['QED'] = [0.0]
        result = calculate_imp_score_phase1(base_dataframe)
        assert abs(result['QED_Multiplier'].iloc[0] - 0.75) < 0.001

    def test_qed_half_gives_875_percent(self, base_dataframe):
        """Test QED=0.5 gives multiplier of 0.875."""
        base_dataframe['QED'] = [0.5]
        result = calculate_imp_score_phase1(base_dataframe)
        # 0.75 + 0.25 * 0.5 = 0.875
        assert abs(result['QED_Multiplier'].iloc[0] - 0.875) < 0.001

    def test_qed_one_gives_100_percent(self, base_dataframe):
        """Test QED=1.0 gives multiplier of 1.0."""
        base_dataframe['QED'] = [1.0]
        result = calculate_imp_score_phase1(base_dataframe)
        assert abs(result['QED_Multiplier'].iloc[0] - 1.0) < 0.001


class TestScoreInterpretation:
    """Test score interpretation thresholds."""

    def test_exceptional_imp_threshold(self):
        """Test exceptional IMP classification (0.9-1.0)."""
        result = interpret_imp_score(0.95)
        assert result['classification'] == 'Exceptional IMP'
        assert result['priority'] == 1

    def test_strong_imp_threshold(self):
        """Test strong IMP classification (0.7-0.9)."""
        result = interpret_imp_score(0.8)
        assert result['classification'] == 'Strong IMP'
        assert result['priority'] == 2

    def test_moderate_imp_threshold(self):
        """Test moderate IMP classification (0.5-0.7)."""
        result = interpret_imp_score(0.6)
        assert result['classification'] == 'Moderate IMP'
        assert result['priority'] == 3

    def test_weak_imp_threshold(self):
        """Test weak IMP classification (0.3-0.5)."""
        result = interpret_imp_score(0.4)
        assert result['classification'] == 'Weak IMP'
        assert result['priority'] == 4

    def test_not_imp_threshold(self):
        """Test not IMP classification (<0.3)."""
        result = interpret_imp_score(0.2)
        assert result['classification'] == 'Not IMP'
        assert result['priority'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
