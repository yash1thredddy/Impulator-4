"""
Integration tests for OQPLA scoring system.

Tests the complete OQPLA pipeline with new weights:
- Efficiency: 40% raw / 50% normalized (uses SEI and BEI only)
- Angle: 15% raw / 18.75% normalized
- Distance: 20% raw / 25% normalized
- PDB: 5% raw / 6.25% normalized
- QED Multiplier: 0.75 + 0.25 * QED
"""

import pytest
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.modules.oqpla_scoring import (
    calculate_oqpla_phase1,
    calculate_oqpla_phase2,
    get_oqpla_score_breakdown,
    interpret_oqpla_score,
    WEIGHT_EFFICIENCY_RAW,
    WEIGHT_ANGLE_RAW,
    WEIGHT_DISTANCE_RAW,
    WEIGHT_PDB_RAW,
    QED_MULTIPLIER_FLOOR,
    QED_MULTIPLIER_SCALE,
)


class TestOQPLAWeightConstants:
    """Test that weight constants are correct."""

    def test_raw_weights_sum_to_80_percent(self):
        """Verify raw weights sum to 80%."""
        total = WEIGHT_EFFICIENCY_RAW + WEIGHT_ANGLE_RAW + WEIGHT_DISTANCE_RAW + WEIGHT_PDB_RAW
        assert abs(total - 0.80) < 0.001, f"Raw weights should sum to 80%, got {total * 100}%"

    def test_individual_raw_weights(self):
        """Verify individual raw weight values."""
        assert abs(WEIGHT_EFFICIENCY_RAW - 0.40) < 0.001, "Efficiency should be 40%"
        assert abs(WEIGHT_ANGLE_RAW - 0.15) < 0.001, "Angle should be 15%"
        assert abs(WEIGHT_DISTANCE_RAW - 0.20) < 0.001, "Distance should be 20%"
        assert abs(WEIGHT_PDB_RAW - 0.05) < 0.001, "PDB should be 5%"

    def test_qed_multiplier_constants(self):
        """Verify QED multiplier formula constants."""
        assert abs(QED_MULTIPLIER_FLOOR - 0.75) < 0.001, "QED floor should be 75%"
        assert abs(QED_MULTIPLIER_SCALE - 0.25) < 0.001, "QED scale should be 25%"


class TestOQPLAFullPipeline:
    """Integration test for complete OQPLA scoring pipeline."""

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

    def test_full_oqpla_pipeline(self, sample_dataframe):
        """Integration test for complete OQPLA scoring pipeline."""
        # Run Phase 2 without actual PDB calls
        result = calculate_oqpla_phase2(sample_dataframe, use_pdb=False)

        # Verify all expected columns exist
        expected_cols = [
            'Efficiency_Score', 'Angle_Score', 'Distance_Score', 'PDB_Score',
            'Efficiency_Contribution', 'Angle_Contribution',
            'Distance_Contribution', 'PDB_Contribution',
            'OQPLA_Base_Score', 'QED_Multiplier', 'OQPLA_Final_Score',
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Verify score is in valid range
        assert 0 <= result['OQPLA_Final_Score'].iloc[0] <= 1, "Final score out of range"

    def test_contributions_sum_to_final_score(self, sample_dataframe):
        """Verify contributions sum correctly to final score (contributions include QED multiplier)."""
        result = calculate_oqpla_phase2(sample_dataframe, use_pdb=False)

        # Contributions include QED multiplier, so they sum to FINAL score
        total_contrib = (
            result['Efficiency_Contribution'].iloc[0] +
            result['Angle_Contribution'].iloc[0] +
            result['Distance_Contribution'].iloc[0] +
            result['PDB_Contribution'].iloc[0]
        )

        assert abs(total_contrib - result['OQPLA_Final_Score'].iloc[0]) < 0.001, \
            f"Contributions should sum to final score: {total_contrib} vs {result['OQPLA_Final_Score'].iloc[0]}"

    def test_qed_multiplier_formula(self, sample_dataframe):
        """Verify QED multiplier uses correct formula: 0.75 + 0.25 * QED."""
        result = calculate_oqpla_phase2(sample_dataframe, use_pdb=False)

        # With QED=0.75: 0.75 + 0.25 * 0.75 = 0.9375
        expected_mult = 0.75 + 0.25 * 0.75
        assert abs(result['QED_Multiplier'].iloc[0] - expected_mult) < 0.001, \
            f"QED multiplier should be {expected_mult}, got {result['QED_Multiplier'].iloc[0]}"

    def test_final_score_equals_base_times_qed(self, sample_dataframe):
        """Verify final score = base score * QED multiplier."""
        result = calculate_oqpla_phase2(sample_dataframe, use_pdb=False)

        expected_final = result['OQPLA_Base_Score'].iloc[0] * result['QED_Multiplier'].iloc[0]
        assert abs(result['OQPLA_Final_Score'].iloc[0] - expected_final) < 0.001, \
            "Final score should equal base * QED multiplier"

    def test_score_breakdown_function(self, sample_dataframe):
        """Test score breakdown helper function."""
        result = calculate_oqpla_phase2(sample_dataframe, use_pdb=False)
        breakdown = get_oqpla_score_breakdown(result.iloc[0])

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

        # Verify component weights
        assert breakdown['component_scores']['efficiency']['weight_normalized'] == '50%'
        assert breakdown['component_scores']['angle']['weight_normalized'] == '18.75%'
        assert breakdown['component_scores']['distance']['weight_normalized'] == '25%'
        assert breakdown['component_scores']['pdb']['weight_normalized'] == '6.25%'


class TestNormalizedWeightRatios:
    """Test that normalized weight ratios are correct."""

    def test_normalized_weights(self):
        """Verify normalized weights when all components active."""
        total_raw = WEIGHT_EFFICIENCY_RAW + WEIGHT_ANGLE_RAW + WEIGHT_DISTANCE_RAW + WEIGHT_PDB_RAW

        norm_eff = WEIGHT_EFFICIENCY_RAW / total_raw
        norm_angle = WEIGHT_ANGLE_RAW / total_raw
        norm_dist = WEIGHT_DISTANCE_RAW / total_raw
        norm_pdb = WEIGHT_PDB_RAW / total_raw

        # Verify normalized weights
        assert abs(norm_eff - 0.50) < 0.001, f"Normalized efficiency should be 50%, got {norm_eff * 100}%"
        assert abs(norm_angle - 0.1875) < 0.001, f"Normalized angle should be 18.75%, got {norm_angle * 100}%"
        assert abs(norm_dist - 0.25) < 0.001, f"Normalized distance should be 25%, got {norm_dist * 100}%"
        assert abs(norm_pdb - 0.0625) < 0.001, f"Normalized PDB should be 6.25%, got {norm_pdb * 100}%"

        # Verify they sum to 1.0
        total_norm = norm_eff + norm_angle + norm_dist + norm_pdb
        assert abs(total_norm - 1.0) < 0.001, f"Normalized weights should sum to 100%, got {total_norm * 100}%"


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
        result = calculate_oqpla_phase1(base_dataframe)
        assert abs(result['QED_Multiplier'].iloc[0] - 0.75) < 0.001

    def test_qed_half_gives_875_percent(self, base_dataframe):
        """Test QED=0.5 gives multiplier of 0.875."""
        base_dataframe['QED'] = [0.5]
        result = calculate_oqpla_phase1(base_dataframe)
        # 0.75 + 0.25 * 0.5 = 0.875
        assert abs(result['QED_Multiplier'].iloc[0] - 0.875) < 0.001

    def test_qed_one_gives_100_percent(self, base_dataframe):
        """Test QED=1.0 gives multiplier of 1.0."""
        base_dataframe['QED'] = [1.0]
        result = calculate_oqpla_phase1(base_dataframe)
        assert abs(result['QED_Multiplier'].iloc[0] - 1.0) < 0.001


class TestScoreInterpretation:
    """Test score interpretation thresholds."""

    def test_exceptional_imp_threshold(self):
        """Test exceptional IMP classification (0.9-1.0)."""
        result = interpret_oqpla_score(0.95)
        assert result['classification'] == 'Exceptional IMP'
        assert result['priority'] == 1

    def test_strong_imp_threshold(self):
        """Test strong IMP classification (0.7-0.9)."""
        result = interpret_oqpla_score(0.8)
        assert result['classification'] == 'Strong IMP'
        assert result['priority'] == 2

    def test_moderate_imp_threshold(self):
        """Test moderate IMP classification (0.5-0.7)."""
        result = interpret_oqpla_score(0.6)
        assert result['classification'] == 'Moderate IMP'
        assert result['priority'] == 3

    def test_weak_imp_threshold(self):
        """Test weak IMP classification (0.3-0.5)."""
        result = interpret_oqpla_score(0.4)
        assert result['classification'] == 'Weak IMP'
        assert result['priority'] == 4

    def test_not_imp_threshold(self):
        """Test not IMP classification (<0.3)."""
        result = interpret_oqpla_score(0.2)
        assert result['classification'] == 'Not IMP'
        assert result['priority'] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
