"""
Azure Blob Storage utilities.
Handles result ZIP storage and log archival in Azure Blob storage.
"""
import gzip
import re
import logging
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from backend.config import settings

logger = logging.getLogger(__name__)


class AzureSyncRotatingFileHandler(RotatingFileHandler):  # pragma: no cover -- log rotation lifecycle
    """
    RotatingFileHandler that uploads rotated log files to Azure Blob storage.

    When a log file is full and rotates, the old file is compressed and
    uploaded to Azure before being replaced. This ensures logs are preserved
    even if the container crashes.
    """

    def doRollover(self):
        """
        Override to compress and upload the rotated file to Azure asynchronously.

        Compression is synchronous (fast, local I/O). Azure upload runs in a
        daemon thread so log rotation is never blocked by network I/O.
        """
        # Get the file that's about to be rotated (current log file)
        if self.stream:
            self.stream.close()
            self.stream = None

        # Compress and upload current log to Azure before rotation
        if self.baseFilename and Path(self.baseFilename).exists():
            # Compress synchronously (fast, local I/O)
            gz_path = self._compress_to_gz(self.baseFilename)
            if gz_path:
                # Upload to Azure in background daemon thread (non-blocking)
                upload_thread = threading.Thread(
                    target=self._upload_gz_to_azure,
                    args=(gz_path,),
                    daemon=True,
                    name="azure-log-upload",
                )
                upload_thread.start()

        # Call parent rollover (handles file rotation)
        super().doRollover()

    def _compress_to_gz(self, filepath: str) -> Path | None:
        """Compress a log file to gzip format.

        Args:
            filepath: Path to the log file to compress.

        Returns:
            Path to the compressed file, or None on failure.
        """
        try:
            path = Path(filepath)
            if not path.exists() or path.stat().st_size == 0:
                return None

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            gz_name = f"backend_{timestamp}.log.gz"
            gz_path = path.parent / gz_name

            with open(path, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    f_out.writelines(f_in)

            return gz_path
        except Exception as e:
            print(f"[AzureSync] Failed to compress log file: {e}")
            return None

    def _upload_gz_to_azure(self, gz_path: Path) -> None:
        """Upload a compressed log file to Azure Blob storage.

        Runs in a daemon thread -- must not raise exceptions.

        Args:
            gz_path: Path to the .gz file to upload.
        """
        try:
            if not settings.AZURE_CONNECTION_STRING:
                gz_path.unlink(missing_ok=True)
                return

            blob_name = f"logs/{gz_path.name}"
            blob = _get_blob_client(blob_name)
            if blob is None:
                gz_path.unlink(missing_ok=True)
                return

            with open(gz_path, 'rb') as f:
                blob.upload_blob(f, overwrite=True)

            # Cleanup local compressed file
            gz_path.unlink(missing_ok=True)

            # Use print since logger might cause recursion in file handler
            print(f"[AzureSync] Uploaded rotated log to Azure: {blob_name}")

            # Cleanup old archives (keep last 5)
            _cleanup_old_log_archives(keep_count=5)

        except Exception as e:
            print(f"[AzureSync] Failed to upload log to Azure: {e}")
            # Cleanup on failure
            try:
                gz_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _upload_to_azure(self, filepath: str) -> bool:
        """Compress and upload a log file to Azure (synchronous).

        Kept for backward compatibility -- used by sync_logs_to_azure().
        """
        try:
            if not settings.AZURE_CONNECTION_STRING:
                return False

            gz_path = self._compress_to_gz(filepath)
            if gz_path is None:
                return False

            blob_name = f"logs/{gz_path.name}"
            blob = _get_blob_client(blob_name)
            if blob is None:
                gz_path.unlink(missing_ok=True)
                return False

            with open(gz_path, 'rb') as f:
                blob.upload_blob(f, overwrite=True)

            gz_path.unlink(missing_ok=True)
            print(f"[AzureSync] Uploaded rotated log to Azure: {blob_name}")
            _cleanup_old_log_archives(keep_count=5)
            return True

        except Exception as e:
            print(f"[AzureSync] Failed to upload log to Azure: {e}")
            return False


def _sanitize_compound_name(name: str) -> str:
    """
    Sanitize compound name for filesystem and Azure storage.

    Internal function - use backend.core.sanitize_compound_name for external use.
    Duplicated here to avoid circular imports (backend.core.__init__ imports from this file).
    Must stay in sync with backend.core.sanitize_compound_name.
    """
    safe = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe = re.sub(r'[^a-zA-Z0-9\-_]', '_', safe)
    safe = re.sub(r'_+', '_', safe)
    safe = safe.strip('_')
    return safe if safe else 'unnamed_compound'

# Lazy import to avoid startup failures if azure not installed
_blob_service_client = None
_blob_service_lock = threading.Lock()


def _get_blob_service():
    """Lazy initialization of Azure Blob service client."""
    global _blob_service_client

    if not settings.AZURE_CONNECTION_STRING:
        return None

    if _blob_service_client is None:
        with _blob_service_lock:
            if _blob_service_client is None:
                try:
                    from azure.storage.blob import BlobServiceClient
                    _blob_service_client = BlobServiceClient.from_connection_string(
                        settings.AZURE_CONNECTION_STRING
                    )
                    logger.info("Azure Blob service client initialized")
                except ImportError:
                    logger.warning("azure-storage-blob not installed, Azure sync disabled")
                    return None
                except Exception as e:
                    logger.warning(
                        "Azure Blob client init failed, uploads will be skipped "
                        "-- check AZURE_CONNECTION_STRING: %s", e
                    )
                    return None

    return _blob_service_client


def close_azure_client():
    """Close the global Azure Blob client. Call on application shutdown."""
    global _blob_service_client
    with _blob_service_lock:
        if _blob_service_client is not None:
            try:
                _blob_service_client.close()
                logger.info("Azure Blob client closed")
            except Exception as e:
                logger.warning(f"Error closing Azure client: {e}")
            finally:
                _blob_service_client = None


def _get_container_client():
    """Get Azure container client."""
    service = _get_blob_service()
    if service is None:
        return None

    try:
        container = service.get_container_client(settings.AZURE_CONTAINER)
        # Create container if it doesn't exist
        if not container.exists():
            container.create_container()
            logger.info(f"Created Azure container: {settings.AZURE_CONTAINER}")
        return container
    except Exception as e:
        logger.error(f"Failed to get container client: {e}")
        return None


def _get_blob_client(blob_name: str):
    """Get Azure Blob client for a specific blob."""
    container = _get_container_client()
    if container is None:
        return None

    return container.get_blob_client(blob_name)


def is_azure_configured() -> bool:
    """Check if Azure Blob storage is configured."""
    return bool(settings.AZURE_CONNECTION_STRING)


def _is_uuid_path(blob_name: str) -> bool:
    """
    Check if a blob name is a UUID-based path.

    UUID paths have format: results/xx/uuid.zip
    where xx is a 2-char prefix and uuid is a 36-char UUID.

    Name-based paths have format: results/compound_name.zip
    """
    import re
    # UUID pattern: 8-4-4-4-12 hex characters
    uuid_pattern = re.compile(
        r'^results/[0-9a-f]{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.zip$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(blob_name))


def _extract_entry_id_from_blob(blob_name: str) -> str | None:
    """
    Extract entry_id (UUID) from a UUID-based blob path.

    Args:
        blob_name: e.g., "results/3a/3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c.zip"

    Returns:
        UUID string or None if not a UUID path
    """
    if not _is_uuid_path(blob_name):
        return None

    # Extract UUID from path: results/xx/uuid.zip -> uuid
    parts = blob_name.replace("results/", "").replace(".zip", "").split("/")
    if len(parts) == 2:
        return parts[1]  # The UUID part
    return None


def list_results_in_azure() -> list:
    """
    List all result entry_ids in Azure Blob storage (UUID-based only).

    Returns:
        List of entry_ids (UUIDs) for stored results
    """
    if not is_azure_configured():
        return []

    container = _get_container_client()
    if container is None:
        return []

    try:
        blobs = container.list_blobs(name_starts_with="results/")
        results = []
        for blob in blobs:
            if _is_uuid_path(blob.name):
                entry_id = _extract_entry_id_from_blob(blob.name)
                if entry_id:
                    results.append(entry_id)
        return results

    except Exception as e:
        logger.error(f"Failed to list results from Azure: {e}")
        return []


def sync_logs_to_azure() -> bool:  # pragma: no cover -- shutdown log upload
    """
    Upload current (non-rotated) log file to Azure on shutdown.

    Only uploads the main backend.log file since rotated logs (.log.1, .log.2)
    are already uploaded by AzureSyncRotatingFileHandler during rotation.

    Returns:
        True if sync successful or Azure not configured
    """
    if not is_azure_configured():
        return True

    log_file = Path(settings.DATA_DIR) / "logs" / "backend.log"
    if not log_file.exists() or log_file.stat().st_size == 0:
        logger.info("No current log to sync to Azure")
        return True

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gz_name = f"backend_{timestamp}_shutdown.log.gz"
    gz_path = log_file.parent / gz_name

    try:
        # Compress current log
        with open(log_file, 'rb') as f_in:
            with gzip.open(gz_path, 'wb') as f_out:
                f_out.writelines(f_in)

        # Upload to Azure
        blob_name = f"logs/{gz_name}"
        blob = _get_blob_client(blob_name)
        if blob is None:
            gz_path.unlink(missing_ok=True)
            return False

        with open(gz_path, 'rb') as f:
            blob.upload_blob(f, overwrite=True)

        logger.info(f"Uploaded current log to Azure: {blob_name}")

        # Cleanup local compressed file
        gz_path.unlink(missing_ok=True)

        # Cleanup old log archives in Azure (keep last 5)
        _cleanup_old_log_archives(keep_count=5)

        return True

    except Exception as e:
        logger.error(f"Failed to sync logs to Azure: {e}")
        if gz_path.exists():
            gz_path.unlink(missing_ok=True)
        return False


def _cleanup_old_log_archives(keep_count: int = 10) -> None:  # pragma: no cover -- Azure log cleanup
    """
    Remove old log archives from Azure, keeping only the most recent ones.

    Args:
        keep_count: Number of recent archives to keep
    """
    container = _get_container_client()
    if container is None:
        return

    try:
        blobs = list(container.list_blobs(name_starts_with="logs/"))
        if len(blobs) <= keep_count:
            return

        # Sort by name (timestamp-based, so alphabetical = chronological)
        blobs.sort(key=lambda b: b.name, reverse=True)

        # Delete older archives
        for blob in blobs[keep_count:]:
            try:
                container.delete_blob(blob.name)
                logger.info(f"Deleted old log archive: {blob.name}")
            except Exception as e:
                logger.warning(f"Failed to delete old log archive {blob.name}: {e}")

    except Exception as e:
        logger.warning(f"Failed to cleanup old log archives: {e}")


# ============================================================================
# UUID-BASED STORAGE FUNCTIONS (New - Phase 5)
# ============================================================================
# These functions use entry_id (UUID) instead of compound_name for storage paths.
# This avoids issues with special characters in names and enables true duplicate support.


# Re-export from canonical location for backward compatibility (ARCH-19)
from backend.core.storage_paths import get_storage_path_from_entry_id  # noqa: E402, F401 -- re-export for backward compat (ARCH-19)


def upload_result_to_azure_by_entry_id(local_path: str, entry_id: str) -> bool:
    """
    Upload result ZIP file to Azure Blob storage using entry_id for path.

    Args:
        local_path: Path to the local ZIP file
        entry_id: UUID for the compound entry

    Returns:
        True if upload successful and verified, or Azure not configured
    """
    if not is_azure_configured():
        return True

    if not entry_id:
        logger.error("entry_id is required for upload")
        return False

    blob_name = get_storage_path_from_entry_id(entry_id)

    blob = _get_blob_client(blob_name)
    if blob is None:
        return False

    try:
        local_path_obj = Path(local_path)
        if not local_path_obj.exists():
            logger.error(f"Local file not found: {local_path}")
            return False

        local_size = local_path_obj.stat().st_size

        # Upload the file
        with open(local_path, "rb") as f:
            blob.upload_blob(f, overwrite=True)

        # Verify upload
        blob_properties = blob.get_blob_properties()
        uploaded_size = blob_properties.size

        if uploaded_size != local_size:
            logger.error(
                f"Upload verification failed for entry_id {entry_id}: "
                f"local size={local_size}, uploaded size={uploaded_size}"
            )
            return False

        logger.info(f"Uploaded and verified {entry_id}.zip to Azure ({blob_name}, {uploaded_size} bytes)")
        return True

    except Exception as e:
        logger.error(f"Failed to upload result to Azure by entry_id: {e}")
        return False


def download_result_from_azure_by_entry_id(entry_id: str, local_path: str) -> bool:  # pragma: no cover -- Azure download with path traversal guard
    """
    Download result ZIP file from Azure Blob storage using entry_id.

    Args:
        entry_id: UUID for the compound entry
        local_path: Where to save the downloaded file

    Returns:
        True if download successful
    """
    if not is_azure_configured():
        return False

    if not entry_id:
        logger.error("entry_id is required for download")
        return False

    blob_name = get_storage_path_from_entry_id(entry_id)

    blob = _get_blob_client(blob_name)
    if blob is None:
        return False

    try:
        # Security: Validate path to prevent path traversal attacks.
        # Note: tempfile.gettempdir() returns the platform's real temp dir
        # (e.g. /var/folders/.../T on macOS, /tmp on Linux, C:\Users\...\Temp
        # on Windows). Hardcoding "/tmp" wrongly rejected legitimate macOS
        # temp paths created by tempfile.TemporaryDirectory() (see Phase 21
        # backfill script which uses the OS temp dir for transient ZIPs).
        resolved_path = Path(local_path).resolve()
        allowed_dirs = [
            Path(settings.RESULTS_DIR).resolve(),
            Path(settings.DATA_DIR).resolve(),
            Path(tempfile.gettempdir()).resolve(),
        ]

        # Use Path.is_relative_to() for proper containment check (not string startswith)
        # This prevents bypasses like /tmp-evil matching /tmp
        def is_path_within(path: Path, parent: Path) -> bool:
            """Check if path is contained within parent directory (path traversal guard)."""
            try:
                path.relative_to(parent)
                return True
            except ValueError:
                return False

        path_is_safe = any(
            is_path_within(resolved_path, allowed_dir)
            for allowed_dir in allowed_dirs
        )

        if not path_is_safe:
            logger.error(f"Path traversal attempt blocked: {local_path}")
            try:
                from backend.core.audit import log_path_traversal_blocked
                log_path_traversal_blocked(local_path)
            except ImportError:
                pass
            return False

        if not blob.exists():
            logger.warning(f"Result {blob_name} not found in Azure")
            return False

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as f:
            download_stream = blob.download_blob()
            f.write(download_stream.readall())

        logger.info(f"Downloaded {entry_id}.zip from Azure")
        return True

    except Exception as e:
        logger.error(f"Failed to download result from Azure by entry_id: {e}")
        return False


def delete_result_from_azure_by_entry_id(entry_id: str) -> bool:
    """
    Delete a result ZIP file from Azure Blob storage using entry_id.

    Args:
        entry_id: UUID for the compound entry to delete

    Returns:
        True if deletion successful
    """
    if not is_azure_configured():
        return True

    if not entry_id:
        logger.warning("entry_id is required for deletion")
        return False

    blob_name = get_storage_path_from_entry_id(entry_id)

    blob = _get_blob_client(blob_name)
    if blob is None:
        return False

    try:
        if blob.exists():
            blob.delete_blob()
            logger.info(f"Deleted {entry_id}.zip from Azure")
        else:
            logger.debug(f"Result {blob_name} not found in Azure (nothing to delete)")
        return True

    except Exception as e:
        logger.error(f"Failed to delete result from Azure by entry_id: {e}")
        return False


def check_result_exists_in_azure_by_entry_id(entry_id: str) -> bool:
    """
    Check if a compound result exists in Azure Blob storage using entry_id.

    Args:
        entry_id: UUID for the compound entry

    Returns:
        True if result exists in Azure
    """
    if not is_azure_configured():
        return False

    if not entry_id:
        return False

    blob_name = get_storage_path_from_entry_id(entry_id)

    blob = _get_blob_client(blob_name)
    if blob is None:
        return False

    try:
        return blob.exists()
    except Exception as e:
        logger.error(f"Failed to check if result exists in Azure by entry_id: {e}")
        return False


# ============================================================================
# AZURE UPLOAD RETRY + TWO-PHASE COMMIT MARKERS
# ============================================================================

_upload_log = structlog.get_logger("azure_sync")


def _on_upload_retry(retry_state):
    """Tenacity before_sleep callback — only fires on actual retries, not the first attempt."""
    from backend.core.metrics import metrics
    metrics.increment('azure_upload_retried')
    _upload_log.warning("azure_upload_retrying", attempt=retry_state.attempt_number)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((OSError, ConnectionError, RuntimeError)),
    reraise=True,
    before_sleep=_on_upload_retry,
)
def _upload_with_retry(local_path: str, entry_id: str) -> bool:
    """Upload result ZIP to Azure with tenacity retry (3 attempts, exponential backoff).

    Retries only transient errors (OSError, ConnectionError, RuntimeError).
    Non-retriable errors (ValueError, FileNotFoundError) fail immediately.
    On permanent failure (after 3 attempts), the exception is reraised.

    Args:
        local_path: Path to the local ZIP file.
        entry_id: UUID for the compound entry.

    Returns:
        True if upload successful and verified.

    Raises:
        Exception: After 3 failed attempts (reraised from last attempt).
    """
    _upload_log.info("azure_upload_attempt", entry_id=entry_id)

    success = upload_result_to_azure_by_entry_id(local_path, entry_id)
    if not success:
        raise RuntimeError(f"Azure upload returned False for entry_id={entry_id}")
    return True


def write_pending_marker(entry_id: str) -> bool:
    """Write a zero-byte .pending marker blob for two-phase commit tracking.

    The marker signals that an upload is in progress. It is deleted after
    successful upload. Orphaned markers (no matching DB record) are cleaned
    up on startup by reconcile_orphaned_uploads().

    Args:
        entry_id: UUID for the compound entry.

    Returns:
        True if marker written successfully, False otherwise.
    """
    if not is_azure_configured() or not entry_id:
        return False

    entry_id = str(entry_id).lower()
    prefix = entry_id[:2]
    blob_name = f"results/{prefix}/.pending-{entry_id}"

    try:
        blob = _get_blob_client(blob_name)
        if blob is None:
            return False
        blob.upload_blob(b"", overwrite=True)
        _upload_log.debug("pending_marker_written", entry_id=entry_id)
        return True
    except Exception as e:
        _upload_log.warning("pending_marker_write_failed", entry_id=entry_id, error=str(e))
        return False


def delete_pending_marker(entry_id: str) -> bool:
    """Delete the .pending marker blob after successful upload.

    Args:
        entry_id: UUID for the compound entry.

    Returns:
        True if deleted or blob doesn't exist, False on error.
    """
    if not is_azure_configured() or not entry_id:
        return False

    entry_id = str(entry_id).lower()
    prefix = entry_id[:2]
    blob_name = f"results/{prefix}/.pending-{entry_id}"

    try:
        blob = _get_blob_client(blob_name)
        if blob is None:
            return False
        if blob.exists():
            blob.delete_blob()
        _upload_log.debug("pending_marker_deleted", entry_id=entry_id)
        return True
    except Exception as e:
        _upload_log.warning("pending_marker_delete_failed", entry_id=entry_id, error=str(e))
        return False


def list_pending_markers() -> list[str]:
    """List all .pending marker blobs and extract entry_ids.

    Returns:
        List of entry_id strings that have pending markers.
    """
    if not is_azure_configured():
        return []

    container = _get_container_client()
    if container is None:
        return []

    try:
        blobs = container.list_blobs(name_starts_with="results/")
        pending_ids = []
        for blob in blobs:
            # Match pattern: results/xx/.pending-<uuid>
            name = blob.name
            if "/.pending-" in name:
                # Extract entry_id from results/xx/.pending-<entry_id>
                parts = name.rsplit("/.pending-", 1)
                if len(parts) == 2:
                    pending_ids.append(parts[1])
        return pending_ids
    except Exception as e:
        _upload_log.error("list_pending_markers_failed", error=str(e))
        return []


def reconcile_orphaned_uploads(max_age_hours: int = 24) -> int:
    """Clean up orphaned .pending markers older than max_age_hours.

    A .pending marker older than max_age_hours indicates a failed or
    abandoned upload. The marker and its corresponding ZIP (if any)
    are deleted.

    Args:
        max_age_hours: Delete markers older than this (default 24).

    Returns:
        Number of orphaned markers cleaned up.
    """
    if not is_azure_configured():
        return 0

    container = _get_container_client()
    if container is None:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    cleaned = 0

    try:
        blobs = container.list_blobs(name_starts_with="results/")
        for blob in blobs:
            if "/.pending-" not in blob.name:
                continue
            # blob.last_modified is timezone-aware UTC datetime from Azure SDK
            if blob.last_modified < cutoff:
                parts = blob.name.rsplit("/.pending-", 1)
                if len(parts) != 2:
                    continue
                entry_id = parts[1]
                age_hours = (datetime.now(timezone.utc) - blob.last_modified).total_seconds() / 3600
                _upload_log.info("orphan_cleanup", entry_id=entry_id, age_hours=round(age_hours, 1))
                delete_pending_marker(entry_id)
                try:
                    delete_result_from_azure_by_entry_id(entry_id)
                except Exception as e:
                    _upload_log.warning("orphan_zip_delete_failed", entry_id=entry_id, error=str(e))
                cleaned += 1
    except Exception as e:
        _upload_log.error("reconcile_orphaned_uploads_failed", error=str(e))

    return cleaned
