"""
Unit tests for admin API key verification.

Tests verify_admin_api_key() in backend/core/auth.py:
- Missing key returns 401
- Unconfigured key returns 503
- Wrong key returns 403
- Correct key returns True
- Constant-time comparison (uses hmac.compare_digest)
"""
import pytest
from unittest.mock import patch
from fastapi import HTTPException
from pydantic import SecretStr

from backend.core.auth import verify_admin_api_key


class TestVerifyAdminApiKey:
    """Tests for admin API key authentication."""

    @patch('backend.core.auth.settings')
    def test_missing_api_key_returns_401(self, mock_settings):
        """Request without X-Admin-API-Key header should return 401."""
        mock_settings.ADMIN_API_KEY = SecretStr("configured-secret-key")

        with pytest.raises(HTTPException) as exc_info:
            verify_admin_api_key(api_key=None)

        assert exc_info.value.status_code == 401
        assert "required" in exc_info.value.detail.lower()

    @patch('backend.core.auth.settings')
    def test_unconfigured_key_returns_503(self, mock_settings):
        """When ADMIN_API_KEY is not configured, should return 503."""
        mock_settings.ADMIN_API_KEY = SecretStr("")

        with pytest.raises(HTTPException) as exc_info:
            verify_admin_api_key(api_key="some-key")

        assert exc_info.value.status_code == 503
        assert "disabled" in exc_info.value.detail.lower()

    @patch('backend.core.auth.settings')
    def test_none_configured_key_returns_503(self, mock_settings):
        """When ADMIN_API_KEY is None, should return 503."""
        mock_settings.ADMIN_API_KEY = SecretStr("")

        with pytest.raises(HTTPException) as exc_info:
            verify_admin_api_key(api_key="some-key")

        assert exc_info.value.status_code == 503

    @patch('backend.core.auth.settings')
    def test_wrong_key_returns_403(self, mock_settings):
        """Incorrect API key should return 403."""
        mock_settings.ADMIN_API_KEY = SecretStr("correct-key")

        with pytest.raises(HTTPException) as exc_info:
            verify_admin_api_key(api_key="wrong-key")

        assert exc_info.value.status_code == 403
        assert "invalid" in exc_info.value.detail.lower()

    @patch('backend.core.auth.settings')
    def test_correct_key_returns_true(self, mock_settings):
        """Correct API key should return True."""
        mock_settings.ADMIN_API_KEY = SecretStr("my-secret-admin-key")

        result = verify_admin_api_key(api_key="my-secret-admin-key")
        assert result is True

    @patch('backend.core.auth.settings')
    def test_key_comparison_is_exact(self, mock_settings):
        """Key comparison should be exact — no prefix/suffix matching."""
        mock_settings.ADMIN_API_KEY = SecretStr("secret")

        # Prefix
        with pytest.raises(HTTPException) as exc_info:
            verify_admin_api_key(api_key="secret-extra")
        assert exc_info.value.status_code == 403

        # Substring
        with pytest.raises(HTTPException) as exc_info:
            verify_admin_api_key(api_key="secre")
        assert exc_info.value.status_code == 403

    @patch('backend.core.auth.settings')
    def test_uses_constant_time_comparison(self, mock_settings):
        """Verify that hmac.compare_digest is used (timing-safe)."""
        mock_settings.ADMIN_API_KEY = SecretStr("secret-key")

        # Patch hmac.compare_digest to verify it's called
        with patch('backend.core.auth.hmac.compare_digest', return_value=True) as mock_hmac:
            result = verify_admin_api_key(api_key="any-key")
            assert result is True
            mock_hmac.assert_called_once_with("any-key", "secret-key")
