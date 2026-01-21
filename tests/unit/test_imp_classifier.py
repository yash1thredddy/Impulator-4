"""
Unit tests for IMP Classification Module.

Tests the core IMP vs Non-IMP classification logic including:
- Classification based on outlier count
- Confidence level assignment (OQPLA-based and simple)
- Summary generation
- Filtering and ranking
- Edge cases and error handling
"""
import pytest
import pandas as pd
import numpy as np


class TestClassifyIMPCandidates:
    """Tests for classify_imp_candidates function."""

    def test_classify_with_oqpla_high_confidence(self):
        """Test classification with high OQPLA score gives High confidence."""
        from backend.modules.imp_classifier import classify_imp_candidates

        df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL1', 'CHEMBL2'],
            'Outlier_Count': [3, 1],
            'OQPLA_Final_Score': [0.85, 0.3]
        })

        result = classify_imp_candidates(df)

        assert result.loc[0, 'Is_IMP_Candidate'] == True
        assert result.loc[0, 'IMP_Confidence'] == 'High'
        assert result.loc[1, 'Is_IMP_Candidate'] == False
        assert result.loc[1, 'IMP_Confidence'] == 'Not IMP'

    def test_classify_with_oqpla_medium_confidence(self):
        """Test classification with medium OQPLA score gives Medium confidence."""
        from backend.modules.imp_classifier import classify_imp_candidates

        df = pd.DataFrame({
            'Outlier_Count': [2],
            'OQPLA_Final_Score': [0.6]
        })

        result = classify_imp_candidates(df)

        assert result.loc[0, 'Is_IMP_Candidate'] == True
        assert result.loc[0, 'IMP_Confidence'] == 'Medium'

    def test_classify_with_oqpla_low_confidence(self):
        """Test classification with low OQPLA score gives Low confidence."""
        from backend.modules.imp_classifier import classify_imp_candidates

        df = pd.DataFrame({
            'Outlier_Count': [2],
            'OQPLA_Final_Score': [0.4]
        })

        result = classify_imp_candidates(df)

        assert result.loc[0, 'Is_IMP_Candidate'] == True
        assert result.loc[0, 'IMP_Confidence'] == 'Low'

    def test_classify_without_oqpla_simple_confidence(self):
        """Test simple confidence assignment when OQPLA not available."""
        from backend.modules.imp_classifier import classify_imp_candidates

        df = pd.DataFrame({
            'Outlier_Count': [4, 3, 2, 1]
        })

        result = classify_imp_candidates(df, use_oqpla=False)

        assert result.loc[0, 'IMP_Confidence'] == 'High'  # 4 outliers
        assert result.loc[1, 'IMP_Confidence'] == 'Medium'  # 3 outliers
        assert result.loc[2, 'IMP_Confidence'] == 'Low'  # 2 outliers
        assert result.loc[3, 'IMP_Confidence'] == 'Not IMP'  # 1 outlier

    def test_classify_custom_min_outlier_count(self):
        """Test classification with custom minimum outlier count."""
        from backend.modules.imp_classifier import classify_imp_candidates

        df = pd.DataFrame({
            'Outlier_Count': [3, 2, 1],
            'OQPLA_Final_Score': [0.8, 0.8, 0.8]
        })

        result = classify_imp_candidates(df, min_outlier_count=3)

        assert result.loc[0, 'Is_IMP_Candidate'] == True
        assert result.loc[1, 'Is_IMP_Candidate'] == False
        assert result.loc[2, 'Is_IMP_Candidate'] == False

    def test_classify_missing_outlier_count_raises(self):
        """Test that missing Outlier_Count column raises ValueError."""
        from backend.modules.imp_classifier import classify_imp_candidates

        df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL1'],
            'OQPLA_Final_Score': [0.8]
        })

        with pytest.raises(ValueError, match="Outlier_Count column not found"):
            classify_imp_candidates(df)

    def test_classify_with_nan_oqpla(self):
        """Test classification handles NaN OQPLA scores."""
        from backend.modules.imp_classifier import classify_imp_candidates

        df = pd.DataFrame({
            'Outlier_Count': [3],
            'OQPLA_Final_Score': [np.nan]
        })

        result = classify_imp_candidates(df)

        assert result.loc[0, 'Is_IMP_Candidate'] == True
        assert result.loc[0, 'IMP_Confidence'] == 'Unknown'

    def test_classify_empty_dataframe(self):
        """Test classification with empty DataFrame."""
        from backend.modules.imp_classifier import classify_imp_candidates

        df = pd.DataFrame({
            'Outlier_Count': [],
            'OQPLA_Final_Score': []
        })

        result = classify_imp_candidates(df)

        assert len(result) == 0
        assert 'Is_IMP_Candidate' in result.columns
        assert 'IMP_Confidence' in result.columns


class TestGetIMPSummary:
    """Tests for get_imp_summary function."""

    def test_summary_basic(self):
        """Test basic summary generation."""
        from backend.modules.imp_classifier import get_imp_summary

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True, True, False, False, False],
            'IMP_Confidence': ['High', 'Medium', 'Not IMP', 'Not IMP', 'Not IMP']
        })

        summary = get_imp_summary(df)

        assert summary['total_compounds'] == 5
        assert summary['total_imp_candidates'] == 2
        assert summary['total_non_imps'] == 3
        assert summary['imp_percentage'] == 40.0
        assert summary['high_confidence'] == 1
        assert summary['medium_confidence'] == 1

    def test_summary_no_classification(self):
        """Test summary when no classification exists."""
        from backend.modules.imp_classifier import get_imp_summary

        df = pd.DataFrame({'ChEMBL_ID': ['CHEMBL1']})
        summary = get_imp_summary(df)

        assert 'error' in summary

    def test_summary_all_imps(self):
        """Test summary when all compounds are IMPs."""
        from backend.modules.imp_classifier import get_imp_summary

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True, True, True],
            'IMP_Confidence': ['High', 'High', 'Medium']
        })

        summary = get_imp_summary(df)

        assert summary['total_imp_candidates'] == 3
        assert summary['imp_percentage'] == 100.0

    def test_summary_no_imps(self):
        """Test summary when no IMPs found."""
        from backend.modules.imp_classifier import get_imp_summary

        df = pd.DataFrame({
            'Is_IMP_Candidate': [False, False],
            'IMP_Confidence': ['Not IMP', 'Not IMP']
        })

        summary = get_imp_summary(df)

        assert summary['total_imp_candidates'] == 0
        assert summary['imp_percentage'] == 0.0


class TestFilterIMPCandidates:
    """Tests for filter_imp_candidates function."""

    def test_filter_by_min_confidence_high(self):
        """Test filtering by High confidence level."""
        from backend.modules.imp_classifier import filter_imp_candidates

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True, True, True, False],
            'IMP_Confidence': ['High', 'Medium', 'Low', 'Not IMP'],
            'OQPLA_Final_Score': [0.9, 0.6, 0.4, 0.3]
        })

        result = filter_imp_candidates(df, min_confidence='High')

        assert len(result) == 1
        assert result.iloc[0]['IMP_Confidence'] == 'High'

    def test_filter_by_min_confidence_medium(self):
        """Test filtering by Medium confidence level (includes High)."""
        from backend.modules.imp_classifier import filter_imp_candidates

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True, True, True, False],
            'IMP_Confidence': ['High', 'Medium', 'Low', 'Not IMP'],
            'OQPLA_Final_Score': [0.9, 0.6, 0.4, 0.3]
        })

        result = filter_imp_candidates(df, min_confidence='Medium')

        assert len(result) == 2
        assert set(result['IMP_Confidence']) == {'High', 'Medium'}

    def test_filter_by_min_oqpla_score(self):
        """Test filtering by minimum OQPLA score."""
        from backend.modules.imp_classifier import filter_imp_candidates

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True, True, True],
            'IMP_Confidence': ['High', 'Medium', 'Low'],
            'OQPLA_Final_Score': [0.9, 0.6, 0.4]
        })

        result = filter_imp_candidates(df, min_oqpla_score=0.65)

        assert len(result) == 1
        assert result.iloc[0]['OQPLA_Final_Score'] == 0.9

    def test_filter_combined_criteria(self):
        """Test filtering with both confidence and OQPLA criteria."""
        from backend.modules.imp_classifier import filter_imp_candidates

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True, True, True, True],
            'IMP_Confidence': ['High', 'High', 'Medium', 'Low'],
            'OQPLA_Final_Score': [0.9, 0.75, 0.6, 0.4]
        })

        result = filter_imp_candidates(df, min_confidence='High', min_oqpla_score=0.8)

        assert len(result) == 1
        assert result.iloc[0]['OQPLA_Final_Score'] == 0.9

    def test_filter_invalid_confidence_raises(self):
        """Test that invalid confidence level raises ValueError."""
        from backend.modules.imp_classifier import filter_imp_candidates

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True],
            'IMP_Confidence': ['High']
        })

        with pytest.raises(ValueError, match="Invalid confidence level"):
            filter_imp_candidates(df, min_confidence='VeryHigh')

    def test_filter_no_classification_raises(self):
        """Test that missing classification raises ValueError."""
        from backend.modules.imp_classifier import filter_imp_candidates

        df = pd.DataFrame({'ChEMBL_ID': ['CHEMBL1']})

        with pytest.raises(ValueError, match="IMP classification not found"):
            filter_imp_candidates(df)


class TestRankIMPCandidates:
    """Tests for rank_imp_candidates function."""

    def test_rank_by_oqpla_descending(self):
        """Test ranking by OQPLA score (highest first)."""
        from backend.modules.imp_classifier import rank_imp_candidates

        df = pd.DataFrame({
            'ChEMBL_ID': ['A', 'B', 'C'],
            'OQPLA_Final_Score': [0.5, 0.9, 0.7]
        })

        result = rank_imp_candidates(df, rank_by='OQPLA_Final_Score')

        assert result.iloc[0]['ChEMBL_ID'] == 'B'  # Highest score
        assert result.iloc[0]['Rank'] == 1
        assert result.iloc[2]['ChEMBL_ID'] == 'A'  # Lowest score
        assert result.iloc[2]['Rank'] == 3

    def test_rank_by_oqpla_ascending(self):
        """Test ranking by OQPLA score (lowest first)."""
        from backend.modules.imp_classifier import rank_imp_candidates

        df = pd.DataFrame({
            'ChEMBL_ID': ['A', 'B', 'C'],
            'OQPLA_Final_Score': [0.5, 0.9, 0.7]
        })

        result = rank_imp_candidates(df, rank_by='OQPLA_Final_Score', ascending=True)

        assert result.iloc[0]['ChEMBL_ID'] == 'A'  # Lowest score
        assert result.iloc[0]['Rank'] == 1

    def test_rank_missing_column_raises(self):
        """Test that ranking by missing column raises ValueError."""
        from backend.modules.imp_classifier import rank_imp_candidates

        df = pd.DataFrame({'ChEMBL_ID': ['A']})

        with pytest.raises(ValueError, match="Column 'MissingColumn' not found"):
            rank_imp_candidates(df, rank_by='MissingColumn')


class TestCompareIMPvsNonIMP:
    """Tests for compare_imp_vs_non_imp function."""

    def test_compare_basic(self):
        """Test basic comparison between IMP and Non-IMP groups."""
        from backend.modules.imp_classifier import compare_imp_vs_non_imp

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True, True, False, False],
            'SEI': [15.0, 20.0, 5.0, 8.0],
            'BEI': [10.0, 12.0, 3.0, 4.0]
        })

        result = compare_imp_vs_non_imp(df, metrics=['SEI', 'BEI'])

        assert len(result) == 2  # Two metrics

        sei_row = result[result['Metric'] == 'SEI'].iloc[0]
        assert sei_row['IMP_Mean'] == 17.5  # (15 + 20) / 2
        assert sei_row['NonIMP_Mean'] == 6.5  # (5 + 8) / 2

    def test_compare_no_classification_raises(self):
        """Test that missing classification raises ValueError."""
        from backend.modules.imp_classifier import compare_imp_vs_non_imp

        df = pd.DataFrame({'SEI': [1, 2, 3]})

        with pytest.raises(ValueError, match="IMP classification not found"):
            compare_imp_vs_non_imp(df)

    def test_compare_with_empty_group(self):
        """Test comparison when one group is empty."""
        from backend.modules.imp_classifier import compare_imp_vs_non_imp

        df = pd.DataFrame({
            'Is_IMP_Candidate': [True, True],
            'SEI': [15.0, 20.0]
        })

        result = compare_imp_vs_non_imp(df, metrics=['SEI'])

        sei_row = result[result['Metric'] == 'SEI'].iloc[0]
        assert np.isnan(sei_row['NonIMP_Mean'])  # No Non-IMPs


class TestGenerateIMPReport:
    """Tests for generate_imp_report function."""

    def test_report_basic(self):
        """Test basic report generation."""
        from backend.modules.imp_classifier import generate_imp_report

        df = pd.DataFrame({
            'ChEMBL_ID': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'Molecule_Name': ['Drug1', 'Drug2', 'Drug3'],
            'Is_IMP_Candidate': [True, True, False],
            'IMP_Confidence': ['High', 'Medium', 'Not IMP'],
            'Outlier_Count': [4, 2, 1],
            'OQPLA_Final_Score': [0.9, 0.6, 0.3]
        })

        report = generate_imp_report(df, "Test Compound")

        assert "Test Compound" in report
        assert "Total compounds analyzed: 3" in report
        assert "IMP candidates identified: 2" in report
        assert "High confidence: 1" in report
        assert "CHEMBL1" in report

    def test_report_no_imps(self):
        """Test report when no IMPs found."""
        from backend.modules.imp_classifier import generate_imp_report

        df = pd.DataFrame({
            'Is_IMP_Candidate': [False],
            'IMP_Confidence': ['Not IMP'],
            'OQPLA_Final_Score': [0.3]
        })

        report = generate_imp_report(df)

        assert "IMP candidates identified: 0" in report


class TestConfidenceAssignment:
    """Tests for internal confidence assignment functions."""

    def test_assign_confidence_oqpla_boundary_cases(self):
        """Test OQPLA confidence at exact boundaries."""
        from backend.modules.imp_classifier import _assign_confidence_oqpla

        # Test boundaries
        assert _assign_confidence_oqpla(True, 0.7) == 'Medium'  # Exactly 0.7
        assert _assign_confidence_oqpla(True, 0.71) == 'High'  # Just above
        assert _assign_confidence_oqpla(True, 0.5) == 'Low'  # Exactly 0.5
        assert _assign_confidence_oqpla(True, 0.51) == 'Medium'  # Just above

    def test_assign_confidence_simple_boundary_cases(self):
        """Test simple confidence at exact boundaries."""
        from backend.modules.imp_classifier import _assign_confidence_simple

        assert _assign_confidence_simple(True, 4) == 'High'
        assert _assign_confidence_simple(True, 3) == 'Medium'
        assert _assign_confidence_simple(True, 2) == 'Low'
