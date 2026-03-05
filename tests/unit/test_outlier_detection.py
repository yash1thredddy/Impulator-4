"""
Unit tests for outlier_detection module.

Tests all pure-computation statistical functions:
- IQR threshold calculation
- Single-metric outlier flagging
- Percentile ranks
- Efficiency outlier detection
- Cohort statistics
- Outlier summary
- Z-scores
- Outlier filtering
"""
import numpy as np
import pandas as pd
import pytest

from backend.modules.outlier_detection import (
    calculate_iqr_threshold,
    flag_outliers_single_metric,
    calculate_percentile_ranks,
    detect_efficiency_outliers,
    calculate_cohort_statistics,
    get_outlier_summary,
    calculate_z_scores,
    filter_outliers,
)


class TestCalculateIQRThreshold:
    def test_normal_data(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        q1, q3, iqr, lower, upper = calculate_iqr_threshold(data)
        assert q1 == pytest.approx(3.25)
        assert q3 == pytest.approx(7.75)
        assert iqr == pytest.approx(4.5)

    def test_empty_series(self):
        data = pd.Series([], dtype=float)
        q1, q3, iqr, lower, upper = calculate_iqr_threshold(data)
        assert np.isnan(q1)
        assert np.isnan(upper)

    def test_all_nan(self):
        data = pd.Series([np.nan, np.nan, np.nan])
        q1, q3, iqr, lower, upper = calculate_iqr_threshold(data)
        assert np.isnan(q1)

    def test_single_value(self):
        data = pd.Series([5.0])
        q1, q3, iqr, _, _ = calculate_iqr_threshold(data)
        assert iqr == pytest.approx(0.0)

    def test_custom_multiplier(self):
        data = pd.Series(range(1, 21))
        _, _, iqr, lower_15, upper_15 = calculate_iqr_threshold(data, multiplier=1.5)
        _, _, _, lower_20, upper_20 = calculate_iqr_threshold(data, multiplier=2.0)
        assert upper_20 > upper_15


class TestFlagOutliersSingleMetric:
    @pytest.fixture
    def data_with_outlier(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

    def test_upper_direction(self, data_with_outlier):
        flags = flag_outliers_single_metric(data_with_outlier, direction='upper')
        assert flags.iloc[-1] is True or flags.iloc[-1] == True  # 100 is outlier

    def test_lower_direction(self):
        # Value must be below Q1 - 1.5*IQR to be a lower outlier
        data = pd.Series([500, 500, 500, 500, 500, 500, 500, 500, 500, -1000])
        flags = flag_outliers_single_metric(data, direction='lower')
        assert flags.iloc[-1] == True  # -1000 is clearly a lower outlier

    def test_both_direction(self, data_with_outlier):
        flags = flag_outliers_single_metric(data_with_outlier, direction='both')
        assert isinstance(flags, pd.Series)
        assert len(flags) == len(data_with_outlier)

    def test_invalid_direction_raises(self, data_with_outlier):
        with pytest.raises(ValueError, match="Invalid direction"):
            flag_outliers_single_metric(data_with_outlier, direction='invalid')

    def test_nan_threshold_returns_all_false(self):
        data = pd.Series([np.nan, np.nan])
        flags = flag_outliers_single_metric(data)
        assert not flags.any()

    def test_empty_series(self):
        data = pd.Series([], dtype=float)
        flags = flag_outliers_single_metric(data)
        assert len(flags) == 0


class TestCalculatePercentileRanks:
    def test_adds_percentile_columns(self):
        df = pd.DataFrame({'SEI': [1, 2, 3, 4, 5]})
        result = calculate_percentile_ranks(df, ['SEI'])
        assert 'SEI_Percentile' in result.columns

    def test_missing_column_skipped(self):
        df = pd.DataFrame({'SEI': [1, 2, 3]})
        result = calculate_percentile_ranks(df, ['SEI', 'MISSING'])
        assert 'SEI_Percentile' in result.columns
        assert 'MISSING_Percentile' not in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({'SEI': [1, 2, 3]})
        original_cols = set(df.columns)
        calculate_percentile_ranks(df, ['SEI'])
        assert set(df.columns) == original_cols


class TestDetectEfficiencyOutliers:
    @pytest.fixture
    def metrics_df(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            'SEI': np.random.normal(10, 2, n),
            'BEI': np.random.normal(15, 3, n),
            'NSEI': np.random.normal(1.0, 0.3, n),
            'NBEI': np.random.normal(0.5, 0.1, n),
        })
        # Add one clear outlier
        df.loc[0, 'SEI'] = 50.0
        df.loc[0, 'BEI'] = 50.0
        return df

    def test_adds_outlier_columns(self, metrics_df):
        result = detect_efficiency_outliers(metrics_df)
        assert 'Is_SEI_Outlier' in result.columns
        assert 'Is_BEI_Outlier' in result.columns
        assert 'Outlier_Count' in result.columns
        assert 'Is_Efficiency_Outlier' in result.columns

    def test_detects_clear_outlier(self, metrics_df):
        result = detect_efficiency_outliers(metrics_df)
        assert result.loc[0, 'Is_Efficiency_Outlier'] == True

    def test_missing_metric_raises(self):
        df = pd.DataFrame({'SEI': [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing metric columns"):
            detect_efficiency_outliers(df, metrics=['SEI', 'BEI'])

    def test_custom_metrics(self):
        df = pd.DataFrame({'SEI': [1, 2, 3, 4, 5, 100]})
        result = detect_efficiency_outliers(df, metrics=['SEI'])
        assert 'Is_SEI_Outlier' in result.columns


class TestCalculateCohortStatistics:
    def test_returns_all_stats(self):
        df = pd.DataFrame({'SEI': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        stats = calculate_cohort_statistics(df, metrics=['SEI'])
        sei_stats = stats['SEI']
        assert sei_stats['count'] == 10
        assert 'mean' in sei_stats
        assert 'median' in sei_stats
        assert 'std' in sei_stats
        assert 'min' in sei_stats
        assert 'max' in sei_stats
        assert 'q1' in sei_stats
        assert 'q3' in sei_stats
        assert 'iqr' in sei_stats
        assert 'outlier_threshold' in sei_stats

    def test_empty_metric(self):
        df = pd.DataFrame({'SEI': pd.Series([], dtype=float)})
        stats = calculate_cohort_statistics(df, metrics=['SEI'])
        assert stats['SEI']['count'] == 0
        assert np.isnan(stats['SEI']['mean'])

    def test_missing_metric_skipped(self):
        df = pd.DataFrame({'SEI': [1, 2, 3]})
        stats = calculate_cohort_statistics(df, metrics=['SEI', 'MISSING'])
        assert 'SEI' in stats
        assert 'MISSING' not in stats

    def test_all_nan_metric(self):
        df = pd.DataFrame({'SEI': [np.nan, np.nan, np.nan]})
        stats = calculate_cohort_statistics(df, metrics=['SEI'])
        assert stats['SEI']['count'] == 0


class TestGetOutlierSummary:
    def test_with_outlier_flags(self):
        df = pd.DataFrame({
            'Is_Efficiency_Outlier': [True, True, False, False, False],
            'Is_SEI_Outlier': [True, False, False, False, False],
            'Is_BEI_Outlier': [True, True, False, False, False],
            'Outlier_Count': [2, 1, 0, 0, 0],
        })
        summary = get_outlier_summary(df)
        assert summary['total_compounds'] == 5
        assert summary['efficiency_outliers'] == 2
        assert summary['outlier_percentage'] == pytest.approx(40.0)
        assert summary['outliers_per_metric']['SEI'] == 1
        assert summary['outliers_per_metric']['BEI'] == 2

    def test_no_outlier_flags(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        summary = get_outlier_summary(df)
        assert summary['total_compounds'] == 3
        assert 'efficiency_outliers' not in summary

    def test_distribution_counts(self):
        df = pd.DataFrame({
            'Is_Efficiency_Outlier': [True, True, False],
            'Outlier_Count': [2, 1, 0],
        })
        summary = get_outlier_summary(df)
        assert 0 in summary['outlier_count_distribution']


class TestCalculateZScores:
    def test_normal_z_scores(self):
        df = pd.DataFrame({'SEI': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = calculate_z_scores(df, metrics=['SEI'])
        assert 'SEI_Zscore' in result.columns
        assert result['SEI_Zscore'].mean() == pytest.approx(0.0, abs=1e-10)

    def test_zero_variance(self):
        df = pd.DataFrame({'SEI': [5.0, 5.0, 5.0]})
        result = calculate_z_scores(df, metrics=['SEI'])
        assert (result['SEI_Zscore'] == 0.0).all()

    def test_all_nan(self):
        df = pd.DataFrame({'SEI': [np.nan, np.nan]})
        result = calculate_z_scores(df, metrics=['SEI'])
        assert 'SEI_Zscore' in result.columns

    def test_missing_metric_skipped(self):
        df = pd.DataFrame({'SEI': [1, 2, 3]})
        result = calculate_z_scores(df, metrics=['SEI', 'MISSING'])
        assert 'SEI_Zscore' in result.columns
        assert 'MISSING_Zscore' not in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({'SEI': [1, 2, 3]})
        original_cols = set(df.columns)
        calculate_z_scores(df, metrics=['SEI'])
        assert set(df.columns) == original_cols


class TestFilterOutliers:
    def test_filters_by_count(self):
        df = pd.DataFrame({'Outlier_Count': [0, 1, 2, 3, 4]})
        result = filter_outliers(df, min_outlier_count=2)
        assert len(result) == 3
        assert (result['Outlier_Count'] >= 2).all()

    def test_missing_column_raises(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        with pytest.raises(ValueError, match="Outlier_Count column not found"):
            filter_outliers(df)

    def test_no_outliers(self):
        df = pd.DataFrame({'Outlier_Count': [0, 0, 0]})
        result = filter_outliers(df, min_outlier_count=1)
        assert len(result) == 0
