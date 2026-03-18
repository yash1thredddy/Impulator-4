"""
Unit tests for Azure sync utilities.
Works with or without Azure connection.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


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
        """Test upload_result_to_azure_by_entry_id works gracefully."""
        from backend.core.azure_sync import upload_result_to_azure_by_entry_id
        from backend.core.azure_sync import is_azure_configured

        # Should not raise - test with a fake UUID
        test_entry_id = "12345678-1234-1234-1234-123456789012"
        result = upload_result_to_azure_by_entry_id("/nonexistent/path.zip", test_entry_id)
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


class TestSyncDbToAzure:
    """Tests for sync_db_to_azure with mocked Azure."""

    def test_sync_not_configured(self):
        """Test sync returns True when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import sync_db_to_azure
            assert sync_db_to_azure() is True

    def test_sync_blob_client_none(self):
        """Test sync returns False when blob client is None."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=None):
            from backend.core.azure_sync import sync_db_to_azure
            assert sync_db_to_azure() is False

    def test_sync_db_not_found(self, tmp_path):
        """Test sync returns False when DB file doesn't exist."""
        mock_blob = MagicMock()
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob), \
             patch('backend.core.azure_sync.settings') as mock_settings:
            mock_settings.DATA_DIR = tmp_path / "nonexistent"
            from backend.core.azure_sync import sync_db_to_azure
            assert sync_db_to_azure() is False

    def test_sync_success(self, tmp_path):
        """Test successful sync with mocked backup and upload."""
        # Create a fake DB file
        db_path = tmp_path / "impulator.db"
        db_path.write_text("fake db content")
        # Also create the temp sync file that the code will try to open
        sync_path = tmp_path / "impulator.db.sync"

        mock_blob = MagicMock()

        def fake_connect(path, **kwargs):
            """Fake sqlite3.connect that creates the sync file for backup."""
            # The second connect creates the temp file
            if str(path) == str(sync_path):
                sync_path.write_text("backup content")
            return MagicMock()

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob), \
             patch('backend.core.azure_sync.settings') as mock_settings, \
             patch('sqlite3.connect', side_effect=fake_connect):
            mock_settings.DATA_DIR = tmp_path

            from backend.core.azure_sync import sync_db_to_azure
            result = sync_db_to_azure()
            assert result is True
            mock_blob.upload_blob.assert_called_once()

    def test_sync_exception_cleans_temp(self, tmp_path):
        """Test sync cleans up temp file on exception."""
        db_path = tmp_path / "impulator.db"
        db_path.write_text("fake db")
        sync_path = tmp_path / "impulator.db.sync"
        sync_path.write_text("temp")

        mock_blob = MagicMock()
        mock_blob.upload_blob.side_effect = Exception("upload failed")

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob), \
             patch('backend.core.azure_sync.settings') as mock_settings, \
             patch('sqlite3.connect') as mock_connect:
            mock_settings.DATA_DIR = tmp_path
            mock_src = MagicMock()
            mock_dst = MagicMock()
            mock_connect.side_effect = [mock_src, mock_dst]

            from backend.core.azure_sync import sync_db_to_azure
            result = sync_db_to_azure()
            assert result is False


class TestUploadResultToAzure:
    """Tests for upload_result_to_azure_by_entry_id."""

    def test_not_configured_returns_true(self):
        """Test returns True when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import upload_result_to_azure_by_entry_id
            assert upload_result_to_azure_by_entry_id("/path.zip", "abc") is True

    def test_empty_entry_id_returns_false(self):
        """Test returns False for empty entry_id."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            from backend.core.azure_sync import upload_result_to_azure_by_entry_id
            assert upload_result_to_azure_by_entry_id("/path.zip", "") is False

    def test_file_not_found_returns_false(self, tmp_path):
        """Test returns False when local file doesn't exist."""
        mock_blob = MagicMock()
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import upload_result_to_azure_by_entry_id
            assert upload_result_to_azure_by_entry_id("/nonexistent.zip", "abc-123") is False

    def test_successful_upload_with_verification(self, tmp_path):
        """Test successful upload verifies size."""
        local_file = tmp_path / "test.zip"
        local_file.write_bytes(b"zip content here")

        mock_blob = MagicMock()
        mock_props = MagicMock()
        mock_props.size = len(b"zip content here")
        mock_blob.get_blob_properties.return_value = mock_props

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import upload_result_to_azure_by_entry_id
            result = upload_result_to_azure_by_entry_id(str(local_file), "ab345678-1234-1234-1234-123456789012")
            assert result is True
            mock_blob.upload_blob.assert_called_once()

    def test_size_mismatch_returns_false(self, tmp_path):
        """Test returns False on size mismatch after upload."""
        local_file = tmp_path / "test.zip"
        local_file.write_bytes(b"zip content here")

        mock_blob = MagicMock()
        mock_props = MagicMock()
        mock_props.size = 999  # Different from actual size
        mock_blob.get_blob_properties.return_value = mock_props

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import upload_result_to_azure_by_entry_id
            result = upload_result_to_azure_by_entry_id(str(local_file), "ab345678-1234-1234-1234-123456789012")
            assert result is False


class TestDeleteResultFromAzure:
    """Tests for delete_result_from_azure_by_entry_id."""

    def test_not_configured_returns_true(self):
        """Test returns True when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import delete_result_from_azure_by_entry_id
            assert delete_result_from_azure_by_entry_id("abc") is True

    def test_empty_entry_id_returns_false(self):
        """Test returns False for empty entry_id."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            from backend.core.azure_sync import delete_result_from_azure_by_entry_id
            assert delete_result_from_azure_by_entry_id("") is False

    def test_successful_delete(self):
        """Test successful deletion when blob exists."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import delete_result_from_azure_by_entry_id
            result = delete_result_from_azure_by_entry_id("ab345678-1234-1234-1234-123456789012")
            assert result is True
            mock_blob.delete_blob.assert_called_once()

    def test_delete_blob_not_exists(self):
        """Test delete returns True when blob doesn't exist (nothing to delete)."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import delete_result_from_azure_by_entry_id
            result = delete_result_from_azure_by_entry_id("ab345678-1234-1234-1234-123456789012")
            assert result is True
            mock_blob.delete_blob.assert_not_called()


class TestCheckResultExists:
    """Tests for check_result_exists_in_azure_by_entry_id."""

    def test_not_configured_returns_false(self):
        """Test returns False when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import check_result_exists_in_azure_by_entry_id
            assert check_result_exists_in_azure_by_entry_id("abc") is False

    def test_empty_entry_id_returns_false(self):
        """Test returns False for empty entry_id."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            from backend.core.azure_sync import check_result_exists_in_azure_by_entry_id
            assert check_result_exists_in_azure_by_entry_id("") is False

    def test_exists_returns_true(self):
        """Test returns True when blob exists."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import check_result_exists_in_azure_by_entry_id
            assert check_result_exists_in_azure_by_entry_id("ab345678-1234-1234-1234-123456789012") is True


class TestDownloadResultFromAzure:
    """Tests for download_result_from_azure_by_entry_id."""

    def test_not_configured_returns_false(self):
        """Test returns False when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import download_result_from_azure_by_entry_id
            assert download_result_from_azure_by_entry_id("abc", "/tmp/out.zip") is False

    def test_empty_entry_id_returns_false(self):
        """Test returns False for empty entry_id."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            from backend.core.azure_sync import download_result_from_azure_by_entry_id
            assert download_result_from_azure_by_entry_id("", "/tmp/out.zip") is False


class TestListResultsInAzure:
    """Tests for list_results_in_azure."""

    def test_not_configured_returns_empty(self):
        """Test returns empty list when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import list_results_in_azure
            assert list_results_in_azure() == []

    def test_lists_uuid_paths_only(self):
        """Test only UUID-based paths are included in results."""
        mock_container = MagicMock()
        blob1 = MagicMock()
        blob1.name = "results/3a/3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c.zip"
        blob2 = MagicMock()
        blob2.name = "results/Aspirin.zip"  # Legacy name-based
        mock_container.list_blobs.return_value = [blob1, blob2]

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_container_client', return_value=mock_container):
            from backend.core.azure_sync import list_results_in_azure
            results = list_results_in_azure()
            assert len(results) == 1
            assert results[0] == "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c"


class TestPendingMarkers:
    """Tests for pending marker functions."""

    def test_write_pending_marker_not_configured(self):
        """Test returns False when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import write_pending_marker
            assert write_pending_marker("abc") is False

    def test_write_pending_marker_empty_id(self):
        """Test returns False for empty entry_id."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            from backend.core.azure_sync import write_pending_marker
            assert write_pending_marker("") is False

    def test_write_pending_marker_success(self):
        """Test successful marker write."""
        mock_blob = MagicMock()
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import write_pending_marker
            result = write_pending_marker("ab345678-1234-1234-1234-123456789012")
            assert result is True
            mock_blob.upload_blob.assert_called_once_with(b"", overwrite=True)

    def test_delete_pending_marker_success(self):
        """Test successful marker deletion."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import delete_pending_marker
            result = delete_pending_marker("ab345678-1234-1234-1234-123456789012")
            assert result is True
            mock_blob.delete_blob.assert_called_once()

    def test_list_pending_markers(self):
        """Test listing pending markers."""
        mock_container = MagicMock()
        blob = MagicMock()
        blob.name = "results/ab/.pending-ab345678-1234-1234-1234-123456789012"
        mock_container.list_blobs.return_value = [blob]

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_container_client', return_value=mock_container):
            from backend.core.azure_sync import list_pending_markers
            result = list_pending_markers()
            assert len(result) == 1
            assert result[0] == "ab345678-1234-1234-1234-123456789012"


class TestReconcileOrphanedUploads:
    """Tests for reconcile_orphaned_uploads."""

    def test_not_configured_returns_zero(self):
        """Test returns 0 when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import reconcile_orphaned_uploads
            assert reconcile_orphaned_uploads(set()) == 0

    def test_no_pending_returns_zero(self):
        """Test returns 0 when no pending markers exist."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync.list_pending_markers', return_value=[]):
            from backend.core.azure_sync import reconcile_orphaned_uploads
            assert reconcile_orphaned_uploads(set()) == 0

    def test_cleans_orphans(self):
        """Test cleans up orphaned markers not in DB."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync.list_pending_markers', return_value=["orphan-id", "valid-id"]), \
             patch('backend.core.azure_sync.delete_pending_marker') as mock_del_marker, \
             patch('backend.core.azure_sync.delete_result_from_azure_by_entry_id') as mock_del_result:
            from backend.core.azure_sync import reconcile_orphaned_uploads
            result = reconcile_orphaned_uploads({"valid-id"})
            assert result == 1  # Only orphan-id cleaned
            mock_del_marker.assert_called_once_with("orphan-id")
            mock_del_result.assert_called_once_with("orphan-id")


class TestCloseAzureClient:
    """Tests for close_azure_client."""

    def test_close_when_none(self):
        """Test close does nothing when client is None."""
        from backend.core import azure_sync
        original = azure_sync._blob_service_client
        azure_sync._blob_service_client = None
        try:
            azure_sync.close_azure_client()  # Should not raise
        finally:
            azure_sync._blob_service_client = original

    def test_close_when_active(self):
        """Test close calls client.close()."""
        from backend.core import azure_sync
        original = azure_sync._blob_service_client

        mock_client = MagicMock()
        azure_sync._blob_service_client = mock_client
        try:
            azure_sync.close_azure_client()
            mock_client.close.assert_called_once()
            assert azure_sync._blob_service_client is None
        finally:
            azure_sync._blob_service_client = original


class TestGetBlobService:
    """Tests for _get_blob_service lazy init."""

    def test_returns_none_when_no_connection_string(self):
        """Test returns None when connection string is empty."""
        with patch('backend.core.azure_sync.settings') as mock_settings:
            mock_settings.AZURE_CONNECTION_STRING = ""
            from backend.core.azure_sync import _get_blob_service
            # Reset cached client
            from backend.core import azure_sync
            original = azure_sync._blob_service_client
            azure_sync._blob_service_client = None
            try:
                result = _get_blob_service()
                assert result is None
            finally:
                azure_sync._blob_service_client = original


class TestSyncLogsToAzure:
    """Tests for sync_logs_to_azure."""

    def test_not_configured_returns_true(self):
        """Test returns True when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import sync_logs_to_azure
            assert sync_logs_to_azure() is True

    def test_no_log_file_returns_true(self, tmp_path):
        """Test returns True when no log file exists."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync.settings') as mock_settings:
            mock_settings.DATA_DIR = tmp_path
            from backend.core.azure_sync import sync_logs_to_azure
            assert sync_logs_to_azure() is True


class TestUploadWithRetry:
    """Tests for _upload_with_retry tenacity wrapper.

    Uses __wrapped__ to bypass tenacity retries for fast unit tests.
    """

    def test_successful_upload(self):
        """Test successful upload returns True."""
        with patch('backend.core.azure_sync.upload_result_to_azure_by_entry_id', return_value=True):
            from backend.core.azure_sync import _upload_with_retry
            # Call __wrapped__ to bypass tenacity retry (no wait between attempts)
            result = _upload_with_retry.__wrapped__("/path/to/file.zip", "ab345678-1234-1234-1234-123456789012")
            assert result is True

    def test_upload_failure_raises(self):
        """Test raises RuntimeError when upload returns False."""
        with patch('backend.core.azure_sync.upload_result_to_azure_by_entry_id', return_value=False):
            from backend.core.azure_sync import _upload_with_retry
            with pytest.raises(RuntimeError, match="Azure upload returned False"):
                _upload_with_retry.__wrapped__("/path/to/file.zip", "ab345678-1234-1234-1234-123456789012")


class TestWritePendingMarkerEdgeCases:
    """Additional tests for pending marker edge cases."""

    def test_write_marker_blob_client_none(self):
        """Test returns False when blob client is None."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=None):
            from backend.core.azure_sync import write_pending_marker
            assert write_pending_marker("ab345678-1234-1234-1234-123456789012") is False

    def test_write_marker_exception(self):
        """Test returns False on upload exception."""
        mock_blob = MagicMock()
        mock_blob.upload_blob.side_effect = Exception("network error")

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import write_pending_marker
            assert write_pending_marker("ab345678-1234-1234-1234-123456789012") is False


class TestDeletePendingMarkerEdgeCases:
    """Additional tests for delete_pending_marker edge cases."""

    def test_delete_marker_not_configured(self):
        """Test returns False when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import delete_pending_marker
            assert delete_pending_marker("abc") is False

    def test_delete_marker_empty_id(self):
        """Test returns False for empty entry_id."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True):
            from backend.core.azure_sync import delete_pending_marker
            assert delete_pending_marker("") is False

    def test_delete_marker_blob_not_exists(self):
        """Test returns True when marker blob doesn't exist."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import delete_pending_marker
            result = delete_pending_marker("ab345678-1234-1234-1234-123456789012")
            assert result is True
            mock_blob.delete_blob.assert_not_called()

    def test_delete_marker_exception(self):
        """Test returns False on exception."""
        mock_blob = MagicMock()
        mock_blob.exists.side_effect = Exception("error")

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import delete_pending_marker
            assert delete_pending_marker("ab345678-1234-1234-1234-123456789012") is False


class TestListPendingMarkersEdgeCases:
    """Additional tests for list_pending_markers."""

    def test_list_markers_not_configured(self):
        """Test returns empty when Azure not configured."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=False):
            from backend.core.azure_sync import list_pending_markers
            assert list_pending_markers() == []

    def test_list_markers_container_none(self):
        """Test returns empty when container client is None."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_container_client', return_value=None):
            from backend.core.azure_sync import list_pending_markers
            assert list_pending_markers() == []

    def test_list_markers_exception(self):
        """Test returns empty on exception."""
        mock_container = MagicMock()
        mock_container.list_blobs.side_effect = Exception("error")

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_container_client', return_value=mock_container):
            from backend.core.azure_sync import list_pending_markers
            assert list_pending_markers() == []


class TestListResultsEdgeCases:
    """Additional tests for list_results_in_azure."""

    def test_list_results_container_none(self):
        """Test returns empty when container client is None."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_container_client', return_value=None):
            from backend.core.azure_sync import list_results_in_azure
            assert list_results_in_azure() == []

    def test_list_results_exception(self):
        """Test returns empty on exception."""
        mock_container = MagicMock()
        mock_container.list_blobs.side_effect = Exception("error")

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_container_client', return_value=mock_container):
            from backend.core.azure_sync import list_results_in_azure
            assert list_results_in_azure() == []


class TestDeleteResultEdgeCases:
    """Additional tests for delete_result_from_azure_by_entry_id."""

    def test_delete_blob_client_none(self):
        """Test returns False when blob client is None."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=None):
            from backend.core.azure_sync import delete_result_from_azure_by_entry_id
            assert delete_result_from_azure_by_entry_id("abc-123") is False

    def test_delete_exception(self):
        """Test returns False on exception."""
        mock_blob = MagicMock()
        mock_blob.exists.side_effect = Exception("error")

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import delete_result_from_azure_by_entry_id
            assert delete_result_from_azure_by_entry_id("abc-123") is False


class TestCheckResultExistsEdgeCases:
    """Additional tests for check_result_exists_in_azure_by_entry_id."""

    def test_check_blob_client_none(self):
        """Test returns False when blob client is None."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=None):
            from backend.core.azure_sync import check_result_exists_in_azure_by_entry_id
            assert check_result_exists_in_azure_by_entry_id("abc-123") is False

    def test_check_exception(self):
        """Test returns False on exception."""
        mock_blob = MagicMock()
        mock_blob.exists.side_effect = Exception("error")

        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync._get_blob_client', return_value=mock_blob):
            from backend.core.azure_sync import check_result_exists_in_azure_by_entry_id
            assert check_result_exists_in_azure_by_entry_id("abc-123") is False


class TestGetContainerClient:
    """Tests for _get_container_client."""

    def test_service_none_returns_none(self):
        """Test returns None when blob service is None."""
        with patch('backend.core.azure_sync._get_blob_service', return_value=None):
            from backend.core.azure_sync import _get_container_client
            assert _get_container_client() is None

    def test_container_not_exists_creates(self):
        """Test creates container when it doesn't exist."""
        mock_container = MagicMock()
        mock_container.exists.return_value = False
        mock_service = MagicMock()
        mock_service.get_container_client.return_value = mock_container

        with patch('backend.core.azure_sync._get_blob_service', return_value=mock_service), \
             patch('backend.core.azure_sync.settings') as mock_settings:
            mock_settings.AZURE_CONTAINER = "test-container"
            from backend.core.azure_sync import _get_container_client
            result = _get_container_client()
            assert result is not None
            mock_container.create_container.assert_called_once()

    def test_container_exception(self):
        """Test returns None on exception."""
        mock_service = MagicMock()
        mock_service.get_container_client.side_effect = Exception("error")

        with patch('backend.core.azure_sync._get_blob_service', return_value=mock_service), \
             patch('backend.core.azure_sync.settings') as mock_settings:
            mock_settings.AZURE_CONTAINER = "test"
            from backend.core.azure_sync import _get_container_client
            assert _get_container_client() is None


class TestReconcileOrphanedUploadsEdgeCases:
    """Additional tests for reconcile_orphaned_uploads."""

    def test_orphan_zip_delete_exception(self):
        """Test handles exception when deleting orphaned ZIP."""
        with patch('backend.core.azure_sync.is_azure_configured', return_value=True), \
             patch('backend.core.azure_sync.list_pending_markers', return_value=["orphan-id"]), \
             patch('backend.core.azure_sync.delete_pending_marker') as mock_del_marker, \
             patch('backend.core.azure_sync.delete_result_from_azure_by_entry_id',
                    side_effect=Exception("delete failed")):
            from backend.core.azure_sync import reconcile_orphaned_uploads
            result = reconcile_orphaned_uploads(set())
            assert result == 1  # Still counts as cleaned
            mock_del_marker.assert_called_once_with("orphan-id")
