"""
Unit tests for efficiency_planes module.

Tests all pure-computation geometry functions:
- Modulus, angle, slope/intercept calculations
- DataFrame vectorized operations
- Angle interpretation
- Distance between points
- Best-in-class finding
"""
import numpy as np
import pandas as pd
import pytest

from backend.modules.efficiency_planes import (
    calculate_modulus,
    calculate_angle,
    calculate_sei_bei_plane_metrics,
    calculate_nsei_nbei_plane_metrics,
    calculate_all_plane_metrics,
    calculate_plane_metrics_dataframe,
    interpret_angle,
    calculate_distance_between_points,
    find_best_in_class,
)


class TestCalculateModulus:
    def test_normal_calculation(self):
        assert calculate_modulus(3.0, 4.0) == pytest.approx(5.0)

    def test_zero_inputs(self):
        assert calculate_modulus(0.0, 0.0) == pytest.approx(0.0)

    def test_nan_x(self):
        assert np.isnan(calculate_modulus(np.nan, 4.0))

    def test_nan_y(self):
        assert np.isnan(calculate_modulus(3.0, np.nan))

    def test_both_nan(self):
        assert np.isnan(calculate_modulus(np.nan, np.nan))


class TestCalculateAngle:
    def test_45_degrees(self):
        assert calculate_angle(1.0, 1.0) == pytest.approx(45.0)

    def test_zero_degrees(self):
        assert calculate_angle(1.0, 0.0) == pytest.approx(0.0)

    def test_90_degrees(self):
        assert calculate_angle(0.0, 1.0) == pytest.approx(90.0)

    def test_nan_input(self):
        assert np.isnan(calculate_angle(np.nan, 1.0))

    def test_origin(self):
        assert calculate_angle(0.0, 0.0) == pytest.approx(0.0)


class TestSEIBEIPlaneMetrics:
    def test_normal_calculation(self):
        result = calculate_sei_bei_plane_metrics(3.0, 4.0, 85.0, 342.0)
        assert result['Modulus_SEI_BEI'] == pytest.approx(5.0)
        assert 'Angle_SEI_BEI' in result
        assert result['Slope_SEI_BEI'] == pytest.approx(10 * (85.0 / 342.0))

    def test_zero_mw_gives_nan_slope(self):
        result = calculate_sei_bei_plane_metrics(3.0, 4.0, 85.0, 0.0)
        assert np.isnan(result['Slope_SEI_BEI'])
        assert result['Modulus_SEI_BEI'] == pytest.approx(5.0)

    def test_nan_mw_gives_nan_slope(self):
        result = calculate_sei_bei_plane_metrics(3.0, 4.0, 85.0, np.nan)
        assert np.isnan(result['Slope_SEI_BEI'])


class TestNSEINBEIPlaneMetrics:
    def test_normal_calculation(self):
        result = calculate_nsei_nbei_plane_metrics(1.5, 0.3, 5.0, 24.0)
        assert result['Modulus_NSEI_NBEI'] == pytest.approx(np.sqrt(1.5**2 + 0.3**2))
        assert result['Slope_NSEI_NBEI'] == pytest.approx(5.0 / 24.0)
        assert result['Intercept_NSEI_NBEI'] == pytest.approx(np.log10(24.0))

    def test_zero_heavy_atoms(self):
        result = calculate_nsei_nbei_plane_metrics(1.5, 0.3, 5.0, 0.0)
        assert np.isnan(result['Slope_NSEI_NBEI'])
        assert np.isnan(result['Intercept_NSEI_NBEI'])

    def test_nan_heavy_atoms(self):
        result = calculate_nsei_nbei_plane_metrics(1.5, 0.3, 5.0, np.nan)
        assert np.isnan(result['Slope_NSEI_NBEI'])
        assert np.isnan(result['Intercept_NSEI_NBEI'])


class TestCalculateAllPlaneMetrics:
    def test_returns_all_seven_keys(self):
        result = calculate_all_plane_metrics(
            sei=15.0, bei=18.0, nsei=1.5, nbei=0.3,
            psa=85.0, molecular_weight=342.0, npol=5.0, heavy_atoms=24.0
        )
        expected_keys = {
            'Modulus_SEI_BEI', 'Angle_SEI_BEI', 'Slope_SEI_BEI',
            'Modulus_NSEI_NBEI', 'Angle_NSEI_NBEI', 'Slope_NSEI_NBEI',
            'Intercept_NSEI_NBEI',
        }
        assert set(result.keys()) == expected_keys


class TestCalculatePlaneMetricsDataFrame:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'SEI': [15.0, 20.0],
            'BEI': [18.0, 12.0],
            'NSEI': [1.5, 2.0],
            'NBEI': [0.3, 0.4],
            'TPSA': [85.0, 60.0],
            'Molecular_Weight': [342.0, 250.0],
            'NPOL': [5.0, 3.0],
            'Heavy_Atoms': [24.0, 18.0],
        })

    def test_adds_all_columns(self, sample_df):
        result = calculate_plane_metrics_dataframe(sample_df)
        new_cols = {
            'Modulus_SEI_BEI', 'Angle_SEI_BEI',
            'Modulus_NSEI_NBEI', 'Angle_NSEI_NBEI',
            'Slope_SEI_BEI', 'Slope_NSEI_NBEI', 'Intercept_NSEI_NBEI',
        }
        assert new_cols.issubset(set(result.columns))

    def test_vectorized_modulus(self, sample_df):
        result = calculate_plane_metrics_dataframe(sample_df)
        expected = np.sqrt(15.0**2 + 18.0**2)
        assert result['Modulus_SEI_BEI'].iloc[0] == pytest.approx(expected)

    def test_zero_mw_gives_nan_slope(self):
        df = pd.DataFrame({
            'SEI': [10.0], 'BEI': [10.0], 'NSEI': [1.0], 'NBEI': [0.5],
            'TPSA': [50.0], 'Molecular_Weight': [0.0],
            'NPOL': [3.0], 'Heavy_Atoms': [15.0],
        })
        result = calculate_plane_metrics_dataframe(df)
        assert np.isnan(result['Slope_SEI_BEI'].iloc[0])

    def test_zero_heavy_atoms_gives_nan(self):
        df = pd.DataFrame({
            'SEI': [10.0], 'BEI': [10.0], 'NSEI': [1.0], 'NBEI': [0.5],
            'TPSA': [50.0], 'Molecular_Weight': [300.0],
            'NPOL': [3.0], 'Heavy_Atoms': [0.0],
        })
        result = calculate_plane_metrics_dataframe(df)
        assert np.isnan(result['Slope_NSEI_NBEI'].iloc[0])
        assert np.isnan(result['Intercept_NSEI_NBEI'].iloc[0])

    def test_missing_column_raises(self):
        df = pd.DataFrame({'SEI': [10.0], 'BEI': [10.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_plane_metrics_dataframe(df)

    def test_does_not_modify_original(self, sample_df):
        original_cols = set(sample_df.columns)
        calculate_plane_metrics_dataframe(sample_df)
        assert set(sample_df.columns) == original_cols


class TestInterpretAngle:
    def test_nan_returns_invalid(self):
        cat, interp = interpret_angle(np.nan)
        assert cat == "Invalid"

    def test_optimal_45(self):
        cat, _ = interpret_angle(45.0)
        assert cat == "Excellent"

    def test_good_35(self):
        cat, _ = interpret_angle(35.0)
        assert cat == "Good"

    def test_good_55(self):
        cat, _ = interpret_angle(55.0)
        assert cat == "Good"

    def test_fair_25(self):
        cat, _ = interpret_angle(25.0)
        assert cat == "Fair"

    def test_fair_65(self):
        cat, _ = interpret_angle(65.0)
        assert cat == "Fair"

    def test_poor_hydrophobic(self):
        cat, interp = interpret_angle(10.0)
        assert cat == "Poor"
        assert "hydrophobic" in interp.lower()

    def test_poor_polar(self):
        cat, interp = interpret_angle(80.0)
        assert cat == "Poor"
        assert "polar" in interp.lower()


class TestDistanceBetweenPoints:
    def test_normal_distance(self):
        assert calculate_distance_between_points(0, 0, 3, 4) == pytest.approx(5.0)

    def test_same_point(self):
        assert calculate_distance_between_points(5, 5, 5, 5) == pytest.approx(0.0)

    def test_nan_returns_nan(self):
        assert np.isnan(calculate_distance_between_points(np.nan, 0, 3, 4))


class TestFindBestInClass:
    def test_finds_highest_modulus(self):
        df = pd.DataFrame({
            'Modulus_SEI_BEI': [10.0, 25.0, 15.0],
            'ChEMBL_ID': ['C1', 'C2', 'C3'],
            'Molecule_Name': ['A', 'B', 'C'],
            'SEI': [5.0, 10.0, 7.0],
            'BEI': [8.0, 23.0, 13.0],
        })
        best = find_best_in_class(df)
        assert best['ChEMBL_ID'] == 'C2'
        assert best['modulus'] == pytest.approx(25.0)

    def test_all_nan_returns_none(self):
        df = pd.DataFrame({'Modulus_SEI_BEI': [np.nan, np.nan]})
        best = find_best_in_class(df)
        assert best['ChEMBL_ID'] is None

    def test_missing_column_raises(self):
        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            find_best_in_class(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame({'Modulus_SEI_BEI': pd.Series([], dtype=float)})
        best = find_best_in_class(df)
        assert best['ChEMBL_ID'] is None
