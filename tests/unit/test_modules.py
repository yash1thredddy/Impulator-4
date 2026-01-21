"""
Unit tests for decoupled chemistry modules.
"""
import pytest
import pandas as pd
import numpy as np


class TestEfficiencyMetrics:
    """Tests for efficiency metrics calculations."""

    def test_calculate_sei(self):
        """Test SEI calculation."""
        from backend.modules.efficiency_metrics import calculate_sei

        # Normal case
        result = calculate_sei(pActivity=7.5, psa=100.0)
        assert result == 7.5  # 7.5 / (100/100) = 7.5

        # Zero PSA should return NaN
        result = calculate_sei(pActivity=7.5, psa=0.0)
        assert np.isnan(result)

    def test_calculate_bei(self):
        """Test BEI calculation."""
        from backend.modules.efficiency_metrics import calculate_bei

        # Normal case: BEI = pActivity / (MW / 1000)
        result = calculate_bei(pActivity=10.0, molecular_weight=500.0)
        assert result == 20.0  # 10.0 / (500/1000) = 20.0

    def test_calculate_nsei(self):
        """Test NSEI calculation."""
        from backend.modules.efficiency_metrics import calculate_nsei

        # Normal case
        result = calculate_nsei(pActivity=6.0, npol=3.0)
        assert result == 2.0  # 6.0 / 3.0 = 2.0

    def test_calculate_nbei(self):
        """Test NBEI calculation."""
        from backend.modules.efficiency_metrics import calculate_nbei

        # Normal case
        result = calculate_nbei(pActivity=6.0, heavy_atoms=20.0)
        assert result == 0.3  # 6.0 / 20.0 = 0.3

    def test_calculate_all_efficiency_metrics(self):
        """Test all metrics calculation."""
        from backend.modules.efficiency_metrics import calculate_all_efficiency_metrics

        metrics = calculate_all_efficiency_metrics(
            pActivity=7.0,
            psa=70.0,
            molecular_weight=350.0,
            npol=5,
            heavy_atoms=25
        )

        assert 'SEI' in metrics
        assert 'BEI' in metrics
        assert 'NSEI' in metrics
        assert 'NBEI' in metrics
        assert metrics['SEI'] == 10.0  # 7.0 / (70/100) = 10.0
        assert metrics['NBEI'] == 0.28  # 7.0 / 25 = 0.28


class TestEfficiencyPlanes:
    """Tests for efficiency plane calculations."""

    def test_calculate_modulus(self):
        """Test modulus calculation."""
        from backend.modules.efficiency_planes import calculate_modulus

        # 3-4-5 triangle
        result = calculate_modulus(3.0, 4.0)
        assert result == 5.0

    def test_calculate_angle(self):
        """Test angle calculation."""
        from backend.modules.efficiency_planes import calculate_angle

        # 45 degree angle (x = y)
        result = calculate_angle(1.0, 1.0)
        assert abs(result - 45.0) < 0.01

        # 0 degree angle (y = 0)
        result = calculate_angle(1.0, 0.0)
        assert result == 0.0

    def test_calculate_all_plane_metrics(self):
        """Test all plane metrics calculation."""
        from backend.modules.efficiency_planes import calculate_all_plane_metrics

        metrics = calculate_all_plane_metrics(
            sei=10.0, bei=10.0,
            nsei=1.0, nbei=0.3,
            psa=80.0, molecular_weight=400.0,
            npol=5, heavy_atoms=25
        )

        assert 'Modulus_SEI_BEI' in metrics
        assert 'Angle_SEI_BEI' in metrics
        assert 'Slope_SEI_BEI' in metrics
        assert abs(metrics['Angle_SEI_BEI'] - 45.0) < 0.01


class TestOutlierDetection:
    """Tests for outlier detection."""

    def test_calculate_iqr_threshold(self):
        """Test IQR threshold calculation."""
        from backend.modules.outlier_detection import calculate_iqr_threshold

        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        q1, q3, iqr, lower, upper = calculate_iqr_threshold(data)

        assert q1 == 3.25
        assert q3 == 7.75
        assert iqr == 4.5

    def test_detect_efficiency_outliers(self):
        """Test efficiency outlier detection."""
        from backend.modules.outlier_detection import detect_efficiency_outliers

        # Create sample data with one obvious outlier
        df = pd.DataFrame({
            'SEI': [10.0, 11.0, 12.0, 13.0, 50.0],  # 50 is outlier
            'BEI': [15.0, 16.0, 17.0, 18.0, 19.0],
            'NSEI': [1.0, 1.1, 1.2, 1.3, 1.4],
            'NBEI': [0.3, 0.31, 0.32, 0.33, 0.34]
        })

        result = detect_efficiency_outliers(df)

        assert 'Is_SEI_Outlier' in result.columns
        assert 'Outlier_Count' in result.columns
        assert 'Is_Efficiency_Outlier' in result.columns


class TestOQPLAScoring:
    """Tests for O[Q/P/L]A scoring."""

    def test_calculate_efficiency_outlier_score(self):
        """Test efficiency outlier score calculation."""
        from backend.modules.oqpla_scoring import calculate_efficiency_outlier_score

        df = pd.DataFrame({
            'SEI': [10.0, 15.0, 20.0],
            'BEI': [15.0, 20.0, 25.0],
            'NSEI': [1.0, 1.5, 2.0],
            'NBEI': [0.3, 0.4, 0.5]
        })

        scores = calculate_efficiency_outlier_score(df)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores.dropna())

    def test_calculate_angle_score(self):
        """Test angle score calculation."""
        from backend.modules.oqpla_scoring import calculate_angle_score

        angles = pd.Series([45.0, 30.0, 60.0, 0.0, 90.0])
        scores = calculate_angle_score(angles)

        # 45 degrees should give score of 1.0
        assert scores.iloc[0] == 1.0
        # 0 and 90 should give score of 0.0
        assert scores.iloc[3] == 0.0
        assert scores.iloc[4] == 0.0

    def test_interpret_oqpla_score(self):
        """Test O[Q/P/L]A score interpretation."""
        from backend.modules.oqpla_scoring import interpret_oqpla_score

        # Exceptional IMP
        result = interpret_oqpla_score(0.95)
        assert result['classification'] == 'Exceptional IMP'
        assert result['priority'] == 1

        # Strong IMP
        result = interpret_oqpla_score(0.75)
        assert result['classification'] == 'Strong IMP'
        assert result['priority'] == 2

        # Not IMP
        result = interpret_oqpla_score(0.2)
        assert result['classification'] == 'Not IMP'
        assert result['priority'] is None


class TestAPIClient:
    """Tests for API client functions."""

    def test_get_cache_info(self):
        """Test cache info retrieval."""
        from backend.modules.api_client import get_cache_info

        info = get_cache_info()
        assert 'molecule_data' in info
        assert 'classification' in info
        assert 'target_name' in info

    def test_clear_caches(self):
        """Test cache clearing."""
        from backend.modules.api_client import clear_caches

        # Should not raise
        clear_caches()

    def test_get_chembl_ids_empty(self):
        """Test similarity search with no client."""
        from backend.modules.api_client import get_chembl_ids

        # This will likely return empty list without chembl_webresource_client
        result = get_chembl_ids("CCO", similarity_threshold=90)
        assert isinstance(result, list)


class TestModuleImports:
    """Tests to verify module imports work correctly."""

    def test_import_api_client(self):
        """Test api_client module imports."""
        from backend.modules import api_client
        assert hasattr(api_client, 'get_chembl_ids')
        assert hasattr(api_client, 'batch_fetch_activities')

    def test_import_efficiency_metrics(self):
        """Test efficiency_metrics module imports."""
        from backend.modules import efficiency_metrics
        assert hasattr(efficiency_metrics, 'calculate_sei')
        assert hasattr(efficiency_metrics, 'calculate_bei')

    def test_import_oqpla_scoring(self):
        """Test oqpla_scoring module imports."""
        from backend.modules import oqpla_scoring
        assert hasattr(oqpla_scoring, 'calculate_oqpla_phase1')
        assert hasattr(oqpla_scoring, 'interpret_oqpla_score')

    def test_import_config(self):
        """Test config module imports."""
        from backend.modules import config
        assert hasattr(config, 'ACTIVITY_TYPES')
        assert hasattr(config, 'CACHE_SIZE')
