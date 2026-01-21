"""
Unit tests for Azure sync utilities.
Works with or without Azure connection.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestAzureSync:
    """Tests for Azure sync functions."""

    def test_is_azure_configured(self):
        """Test is_azure_configured returns correct value based on settings."""
        from backend.core.azure_sync import is_azure_configured
        from backend.config import settings

        result = is_azure_configured()

        # Should match whether connection string is set
        expected = bool(settings.AZURE_CONNECTION_STRING)
        assert result == expected

    def test_download_db_graceful(self):
        """Test download_db_from_azure works gracefully."""
        from backend.core.azure_sync import download_db_from_azure

        # Should not raise, returns True
        result = download_db_from_azure()
        assert result is True

    def test_sync_db_graceful(self):
        """Test sync_db_to_azure works gracefully."""
        from backend.core.azure_sync import sync_db_to_azure

        # Should not raise, returns True or False depending on db existence
        result = sync_db_to_azure()
        assert isinstance(result, bool)

    def test_upload_result_graceful(self):
        """Test upload_result_to_azure works gracefully."""
        from backend.core.azure_sync import upload_result_to_azure
        from backend.core.azure_sync import is_azure_configured

        # Should not raise
        result = upload_result_to_azure("/nonexistent/path.zip", "TestCompound")
        # Returns True if Azure not configured, False if Azure configured but file doesn't exist
        if is_azure_configured():
            assert result is False  # File doesn't exist
        else:
            assert result is True  # Azure not configured, returns True


class TestAzureConnection:
    """Tests that verify actual Azure connection (when configured)."""

    def test_azure_connection_works(self):
        """Test that Azure connection is successful when configured."""
        from backend.core.azure_sync import is_azure_configured, _get_blob_service
        from backend.config import settings

        if not is_azure_configured():
            pytest.skip("Azure not configured, skipping connection test")

        # This should not raise an exception
        service = _get_blob_service()
        assert service is not None

        # Verify we can access the container
        from backend.core.azure_sync import _get_container_client
        container = _get_container_client()
        assert container is not None
        assert container.exists()

    def test_list_results_works(self):
        """Test listing results from Azure."""
        from backend.core.azure_sync import is_azure_configured, list_results_in_azure

        if not is_azure_configured():
            pytest.skip("Azure not configured, skipping list test")

        # Should return a list (possibly empty)
        results = list_results_in_azure()
        assert isinstance(results, list)


class TestAzureSyncWithMockedClient:
    """Tests with mocked Azure Blob client."""

    def test_download_db_blob_exists(self):
        """Test download when blob exists."""
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = True
        mock_blob_client.download_blob.return_value.readall.return_value = b"test data"

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob_client), \
             patch('builtins.open', MagicMock()):
            from backend.core.azure_sync import download_db_from_azure
            result = download_db_from_azure()
            assert result is True

    def test_download_db_blob_not_exists(self):
        """Test download when blob doesn't exist."""
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = False

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob_client):
            from backend.core.azure_sync import download_db_from_azure
            result = download_db_from_azure()
            assert result is True  # Still successful, just nothing to download
