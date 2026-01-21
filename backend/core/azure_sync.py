"""
Azure Blob Storage sync utilities.
Handles database backup/restore and result uploads.
Azure Blob serves as the single source of truth for data persistence.
"""
import gzip
import os
import re
import shutil
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class AzureSyncRotatingFileHandler(RotatingFileHandler):
    """
    RotatingFileHandler that uploads rotated log files to Azure Blob storage.

    When a log file is full and rotates, the old file is compressed and
    uploaded to Azure before being replaced. This ensures logs are preserved
    even if the container crashes.
    """

    def doRollover(self):
        """
        Override to upload the rotated file to Azure before rotation.
        """
        # Get the file that's about to be rotated (current log file)
        if self.stream:
            self.stream.close()
            self.stream = None

        # Upload current log to Azure before rotation
        if self.baseFilename and Path(self.baseFilename).exists():
            self._upload_to_azure(self.baseFilename)

        # Call parent rollover (handles file rotation)
        super().doRollover()

    def _upload_to_azure(self, filepath: str) -> bool:
        """Compress and upload a log file to Azure."""
        try:
            # Check if Azure is configured (avoid circular import)
            if not settings.AZURE_CONNECTION_STRING:
                return False

            path = Path(filepath)
            if not path.exists() or path.stat().st_size == 0:
                return False

            # Create compressed version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gz_name = f"backend_{timestamp}.log.gz"
            gz_path = path.parent / gz_name

            with open(path, 'rb') as f_in:
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

            # Cleanup local compressed file
            gz_path.unlink(missing_ok=True)

            # Use print since logger might cause recursion
            print(f"[AzureSync] Uploaded rotated log to Azure: {blob_name}")

            # Cleanup old archives (keep last 5)
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


def _get_blob_service():
    """Lazy initialization of Azure Blob service client."""
    global _blob_service_client

    if not settings.AZURE_CONNECTION_STRING:
        return None

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
            logger.error(f"Failed to initialize Azure client: {e}")
            return None

    return _blob_service_client


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


def download_db_from_azure() -> bool:
    """
    Download SQLite database from Azure on container startup.
    This restores state from the single source of truth.

    Returns:
        True if download successful or Azure not configured
    """
    if not is_azure_configured():
        logger.info("Azure not configured, using local database")
        return True

    blob = _get_blob_client("impulator.db")
    if blob is None:
        return False

    db_path = Path(settings.DATA_DIR) / "impulator.db"

    try:
        # Ensure data directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if blob.exists():
            with open(db_path, "wb") as f:
                download_stream = blob.download_blob()
                f.write(download_stream.readall())
            logger.info(f"Downloaded SQLite from Azure ({db_path})")
            return True
        else:
            logger.info("No existing database in Azure, starting fresh")
            return True

    except Exception as e:
        logger.error(f"Failed to download database from Azure: {e}")
        return False


def sync_db_to_azure() -> bool:
    """
    Upload SQLite database to Azure immediately.
    Called after every job completion to ensure no data loss.

    Returns:
        True if sync successful or Azure not configured
    """
    if not is_azure_configured():
        return True

    blob = _get_blob_client("impulator.db")
    if blob is None:
        return False

    db_path = Path(settings.DATA_DIR) / "impulator.db"
    temp_path = Path(settings.DATA_DIR) / "impulator.db.tmp"

    try:
        if not db_path.exists():
            logger.warning("Database file not found, skipping sync")
            return False

        # Copy to temp file to avoid locking issues
        shutil.copy2(db_path, temp_path)

        with open(temp_path, "rb") as f:
            blob.upload_blob(f, overwrite=True)

        # Cleanup temp file
        temp_path.unlink(missing_ok=True)

        logger.info("Synced SQLite to Azure")
        return True

    except Exception as e:
        logger.error(f"Failed to sync database to Azure: {e}")
        # Cleanup temp file on error
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False


# Legacy upload/download functions removed - use upload_result_to_azure_by_entry_id
# and download_result_from_azure_by_entry_id instead


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


def _extract_entry_id_from_blob(blob_name: str) -> Optional[str]:
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


# Legacy delete_result_from_azure and sync_compound_table_from_azure removed
# - Use delete_result_from_azure_by_entry_id instead
# - Database is the source of truth, no legacy sync needed


def sync_logs_to_azure() -> bool:
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def _cleanup_old_log_archives(keep_count: int = 10) -> None:
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


def get_storage_path_from_entry_id(entry_id: str) -> str:
    """
    Generate storage path from entry_id (UUID).

    Uses first 2 characters as prefix for directory distribution.
    This helps avoid having too many files in a single directory.

    Args:
        entry_id: UUID string (e.g., "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c")

    Returns:
        Path like "results/3a/3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c.zip"
    """
    if not entry_id:
        raise ValueError("entry_id cannot be empty")

    # Normalize to lowercase for consistent paths
    entry_id = entry_id.lower()
    prefix = entry_id[:2]
    return f"results/{prefix}/{entry_id}.zip"


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


def download_result_from_azure_by_entry_id(entry_id: str, local_path: str) -> bool:
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
        # Security: Validate path to prevent path traversal attacks
        resolved_path = Path(local_path).resolve()
        allowed_dirs = [
            Path(settings.RESULTS_DIR).resolve(),
            Path(settings.DATA_DIR).resolve(),
            Path("/tmp").resolve() if not os.name == 'nt' else Path(os.environ.get('TEMP', 'C:\\Temp')).resolve(),
        ]

        path_is_safe = any(
            str(resolved_path).startswith(str(allowed_dir))
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
