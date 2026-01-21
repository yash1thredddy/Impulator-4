"""
Azure Blob Storage client for frontend direct access.

Frontend reads directly from Azure (same credentials as backend).
Backend writes, frontend reads - CQRS pattern.

Local caching:
- Results are cached locally in data/results/
- Only downloads from Azure if not cached locally
- Reduces Azure API calls significantly
"""

import os
import io
import json
import zipfile
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache
from datetime import datetime, timezone

import pandas as pd

# Note: sanitize_compound_name removed - only UUID-based storage paths are supported now

logger = logging.getLogger(__name__)

# Lazy import Azure SDK
_blob_service_client = None

# Local cache directory - use absolute path relative to this module
# This ensures frontend and backend share the same data directory
_MODULE_DIR = Path(__file__).parent.parent.parent  # Impulator/
LOCAL_CACHE_DIR = Path(os.getenv("DATA_DIR", str(_MODULE_DIR / "data"))) / "results"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Maximum number of cached compounds (LRU eviction)
MAX_CACHE_ITEMS = int(os.getenv("MAX_CACHE_ITEMS", "1000"))


def _is_uuid(value: str) -> bool:
    """Check if a string looks like a UUID."""
    import re
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
    return bool(uuid_pattern.match(value))


def _get_connection_string() -> Optional[str]:
    """Get Azure connection string from environment."""
    return os.getenv("AZURE_CONNECTION_STRING", "")


def _get_container_name() -> str:
    """Get Azure container name from environment."""
    return os.getenv("AZURE_CONTAINER", "impulator")


def is_azure_configured() -> bool:
    """Check if Azure is configured."""
    return bool(_get_connection_string())


def _get_blob_service():
    """Lazy initialization of Azure Blob service client."""
    global _blob_service_client

    conn_str = _get_connection_string()
    if not conn_str:
        return None

    if _blob_service_client is None:
        try:
            from azure.storage.blob import BlobServiceClient
            _blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            logger.info("Frontend Azure Blob client initialized")
        except ImportError:
            logger.warning("azure-storage-blob not installed")
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
        return service.get_container_client(_get_container_name())
    except Exception as e:
        logger.error(f"Failed to get container client: {e}")
        return None


def list_local_results() -> List[Dict[str, Any]]:
    """
    List all locally cached result files.

    Scans subfolders for UUID-based files (results/{prefix}/{uuid}.zip).
    Only UUID-based storage is supported.

    Returns:
        List of compound info dictionaries
    """
    results = []
    try:
        # Scan subfolders for UUID-based files (results/{prefix}/{uuid}.zip)
        for subdir in LOCAL_CACHE_DIR.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for zip_path in subdir.glob("*.zip"):
                    entry_id = zip_path.stem
                    if _is_uuid(entry_id):
                        stat = zip_path.stat()
                        results.append({
                            "compound_name": entry_id,  # Will be replaced with actual name from DB
                            "entry_id": entry_id,
                            "blob_name": f"results/{subdir.name}/{entry_id}.zip",
                            "size": stat.st_size,
                            "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                            "source": "local",
                        })

        logger.debug(f"Found {len(results)} results in local cache")
    except Exception as e:
        logger.error(f"Failed to list local results: {e}")
    return results


def list_results(use_local_cache: bool = True) -> List[Dict[str, Any]]:
    """
    List all result files, checking local cache first.

    Args:
        use_local_cache: If True, return local results without checking Azure

    Returns:
        List of compound info dictionaries with:
        - compound_name: str
        - blob_name: str
        - size: int
        - last_modified: datetime
        - source: "local" or "azure"
    """
    # If local cache has results and we're not forcing Azure check
    local_results = list_local_results()
    if use_local_cache and local_results:
        logger.debug("Using local cache for result listing")
        return local_results

    # Check Azure for authoritative list
    if not is_azure_configured():
        logger.warning("Azure not configured, using local cache only")
        return local_results

    container = _get_container_client()
    if container is None:
        return local_results

    try:
        results = []
        blobs = container.list_blobs(name_starts_with="results/")

        for blob in blobs:
            if blob.name.endswith(".zip"):
                # Extract compound name from path: results/compound_name.zip
                compound_name = blob.name.replace("results/", "").replace(".zip", "")
                results.append({
                    "compound_name": compound_name,
                    "blob_name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified,
                    "source": "azure",
                })

        logger.info(f"Found {len(results)} results in Azure")
        return results

    except Exception as e:
        logger.error(f"Failed to list results from Azure: {e}")
        # Fallback to local cache
        return local_results


def _get_local_cache_path(entry_id: str) -> Path:
    """Get local cache path for a compound by entry_id (UUID).

    Only UUID-based storage is supported: results/{prefix}/{entry_id}.zip

    Args:
        entry_id: UUID of the compound entry
    """
    if not _is_uuid(entry_id):
        raise ValueError(f"entry_id must be a valid UUID, got: {entry_id}")

    # UUID-based path with subfolder
    prefix = entry_id[:2].lower()
    subdir = LOCAL_CACHE_DIR / prefix
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{entry_id}.zip"


def _is_cached_locally(entry_id: str) -> bool:
    """Check if compound result is cached locally by entry_id (UUID)."""
    if not _is_uuid(entry_id):
        return False

    cache_path = _get_local_cache_path(entry_id)
    return cache_path.exists() and cache_path.stat().st_size > 0


def _read_from_local_cache(entry_id: str) -> Optional[bytes]:
    """Read result from local cache by entry_id (UUID).

    Updates modification time on read (LRU behavior).
    """
    if not _is_uuid(entry_id):
        return None

    cache_path = _get_local_cache_path(entry_id)
    if cache_path.exists():
        try:
            data = cache_path.read_bytes()
            # Touch file to update mtime (LRU: recently accessed stays)
            cache_path.touch()
            return data
        except Exception as e:
            logger.warning(f"Failed to read from cache: {e}")

    return None


def _write_to_local_cache(entry_id: str, data: bytes) -> bool:
    """Write result to local cache by entry_id (UUID)."""
    if not _is_uuid(entry_id):
        logger.warning(f"Cannot cache: entry_id must be a valid UUID, got: {entry_id}")
        return False

    cache_path = _get_local_cache_path(entry_id)
    try:
        cache_path.write_bytes(data)
        logger.info(f"Cached {entry_id} locally ({len(data)} bytes)")
        # Evict oldest files if cache exceeds limit
        _evict_oldest_from_cache()
        return True
    except Exception as e:
        logger.warning(f"Failed to write to cache: {e}")
        return False


def _evict_oldest_from_cache() -> int:
    """
    Evict oldest files from cache if it exceeds MAX_CACHE_ITEMS.

    Uses LRU strategy based on file modification time.
    Scans subfolders for UUID-based files.

    Returns:
        Number of files evicted
    """
    try:
        # Get all cached ZIP files with their modification times
        cached_files = []

        # Scan subfolders (UUID-based paths)
        for subdir in LOCAL_CACHE_DIR.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for zip_path in subdir.glob("*.zip"):
                    try:
                        stat = zip_path.stat()
                        cached_files.append((zip_path, stat.st_mtime))
                    except OSError:
                        continue

        # Check if eviction is needed
        if len(cached_files) <= MAX_CACHE_ITEMS:
            return 0

        # Sort by modification time (oldest first)
        cached_files.sort(key=lambda x: x[1])

        # Calculate how many to evict
        evict_count = len(cached_files) - MAX_CACHE_ITEMS
        evicted = 0

        for zip_path, _ in cached_files[:evict_count]:
            try:
                zip_path.unlink()
                evicted += 1
                logger.debug(f"Evicted {zip_path.stem} from cache")
            except OSError as e:
                logger.warning(f"Failed to evict {zip_path}: {e}")

        if evicted > 0:
            logger.info(f"Evicted {evicted} oldest files from cache (limit: {MAX_CACHE_ITEMS})")

        return evicted

    except Exception as e:
        logger.error(f"Cache eviction failed: {e}")
        return 0


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.

    Scans subfolders for UUID-based files.

    Returns:
        Dict with cache size, count, and limit info
    """
    try:
        total_size = 0
        count = 0
        oldest_mtime = None
        newest_mtime = None

        def _process_zip(zip_path: Path):
            nonlocal total_size, count, oldest_mtime, newest_mtime
            try:
                stat = zip_path.stat()
                total_size += stat.st_size
                count += 1
                if oldest_mtime is None or stat.st_mtime < oldest_mtime:
                    oldest_mtime = stat.st_mtime
                if newest_mtime is None or stat.st_mtime > newest_mtime:
                    newest_mtime = stat.st_mtime
            except OSError:
                pass

        # Scan subfolders (UUID-based paths)
        for subdir in LOCAL_CACHE_DIR.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for zip_path in subdir.glob("*.zip"):
                    _process_zip(zip_path)

        return {
            "count": count,
            "max_items": MAX_CACHE_ITEMS,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest": datetime.fromtimestamp(oldest_mtime, tz=timezone.utc) if oldest_mtime else None,
            "newest": datetime.fromtimestamp(newest_mtime, tz=timezone.utc) if newest_mtime else None,
            "cache_dir": str(LOCAL_CACHE_DIR),
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {"error": str(e)}


def _get_uuid_blob_path(entry_id: str) -> str:
    """Get the Azure blob path for UUID-based storage.

    Format: results/{first-2-chars}/{entry_id}.zip
    Example: results/3a/3a4f8c9e-1b2d-4567-8901-234567890abc.zip
    """
    prefix = entry_id[:2].lower()
    return f"results/{prefix}/{entry_id}.zip"


def download_result(entry_id: str, force_refresh: bool = False) -> Optional[bytes]:
    """
    Download a result ZIP file by entry_id (UUID), using local cache when available.

    Only UUID-based storage is supported: results/{prefix}/{entry_id}.zip

    Args:
        entry_id: UUID of the compound entry
        force_refresh: If True, skip cache and download from Azure

    Returns:
        ZIP file bytes, or None if not found
    """
    if not _is_uuid(entry_id):
        logger.warning(f"download_result requires a valid UUID, got: {entry_id}")
        return None

    # Check local cache first (unless force refresh)
    if not force_refresh and _is_cached_locally(entry_id):
        logger.debug(f"Using cached result for {entry_id}")
        return _read_from_local_cache(entry_id)

    # Download from Azure
    if not is_azure_configured():
        # Fallback: try local cache even without Azure
        return _read_from_local_cache(entry_id)

    container = _get_container_client()
    if container is None:
        return _read_from_local_cache(entry_id)

    # UUID-based storage path
    blob_name = _get_uuid_blob_path(entry_id)

    try:
        blob_client = container.get_blob_client(blob_name)

        if not blob_client.exists():
            logger.warning(f"Result {blob_name} not found in Azure")
            return None

        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        logger.info(f"Downloaded {entry_id} from Azure ({len(data)} bytes)")

        # Cache locally for future use
        _write_to_local_cache(entry_id, data)

        return data

    except Exception as e:
        logger.error(f"Failed to download result: {e}")
        # Try local cache as fallback
        return _read_from_local_cache(entry_id)


def load_result_dataframe(entry_id: str, filename: str = "similar_compounds.csv") -> Optional[pd.DataFrame]:
    """
    Load a specific CSV from a result ZIP file by entry_id (UUID).

    Args:
        entry_id: UUID of the compound entry
        filename: CSV filename within the ZIP (default: similar_compounds.csv)

    Returns:
        DataFrame or None if not found
    """
    zip_data = download_result(entry_id)
    if zip_data is None:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            if filename in zf.namelist():
                with zf.open(filename) as f:
                    return pd.read_csv(f)
            else:
                logger.warning(f"{filename} not found in {entry_id}.zip")
                return None

    except Exception as e:
        logger.error(f"Failed to extract {filename}: {e}")
        return None


def load_result_json(entry_id: str, filename: str = "summary.json") -> Optional[Dict]:
    """
    Load a specific JSON from a result ZIP file by entry_id (UUID).

    Args:
        entry_id: UUID of the compound entry
        filename: JSON filename within the ZIP

    Returns:
        Dict or None if not found
    """
    zip_data = download_result(entry_id)
    if zip_data is None:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            if filename in zf.namelist():
                with zf.open(filename) as f:
                    return json.load(f)
            else:
                logger.warning(f"{filename} not found in {entry_id}.zip")
                return None

    except Exception as e:
        logger.error(f"Failed to extract {filename}: {e}")
        return None


def get_result_files(entry_id: str) -> List[str]:
    """
    List all files in a result ZIP by entry_id (UUID).

    Args:
        entry_id: UUID of the compound entry

    Returns:
        List of filenames in the ZIP
    """
    zip_data = download_result(entry_id)
    if zip_data is None:
        return []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            return zf.namelist()

    except Exception as e:
        logger.error(f"Failed to list ZIP contents: {e}")
        return []


@lru_cache(maxsize=50)
def get_cached_result(entry_id: str) -> Optional[Dict]:
    """
    Get cached result summary for a compound by entry_id (UUID).

    Uses LRU cache to avoid repeated Azure downloads.

    Args:
        entry_id: UUID of the compound entry

    Returns:
        Result summary dict or None
    """
    return load_result_json(entry_id, "summary.json")


def clear_cache():
    """Clear the result cache."""
    get_cached_result.cache_clear()
    logger.info("Azure result cache cleared")


def delete_from_cache(entry_id: str) -> bool:
    """
    Delete a compound from the local cache by entry_id (UUID).

    Called when a compound is deleted from the backend.

    Args:
        entry_id: UUID of the compound entry to delete

    Returns:
        True if deleted, False if not found or error
    """
    if not _is_uuid(entry_id):
        logger.warning(f"delete_from_cache requires a valid UUID, got: {entry_id}")
        return False

    cache_path = _get_local_cache_path(entry_id)
    try:
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Deleted {entry_id} from local cache")
            # Also clear from LRU cache
            get_cached_result.cache_clear()
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete from cache: {e}")
        return False


def delete_from_azure(entry_id: str) -> bool:
    """
    Delete a compound result from Azure Blob Storage by entry_id (UUID).

    This allows deletion even when backend is unavailable.

    Args:
        entry_id: UUID of the compound entry to delete

    Returns:
        True if deleted, False on error
    """
    if not _is_uuid(entry_id):
        logger.warning(f"delete_from_azure requires a valid UUID, got: {entry_id}")
        return False

    container = _get_container_client()
    if container is None:
        logger.warning("Azure not configured, cannot delete from Azure")
        return False

    blob_name = _get_uuid_blob_path(entry_id)

    try:
        blob_client = container.get_blob_client(blob_name)
        if blob_client.exists():
            blob_client.delete_blob()
            logger.info(f"Deleted {entry_id} from Azure: {blob_name}")
            return True
        else:
            logger.debug(f"Result {blob_name} not found in Azure (nothing to delete)")
            return True
    except Exception as e:
        logger.warning(f"Failed to delete from Azure: {e}")
        return False


def delete_compound(entry_id: str) -> bool:
    """
    Delete a compound from all storage locations by entry_id (UUID).

    Deletes from:
    - Azure Blob Storage
    - Local cache

    Note: Does NOT delete from backend database (use backend API for that).

    Args:
        entry_id: UUID of the compound entry to delete

    Returns:
        True if deleted from at least one location
    """
    if not _is_uuid(entry_id):
        logger.warning(f"delete_compound requires a valid UUID, got: {entry_id}")
        return False

    azure_deleted = delete_from_azure(entry_id)
    cache_deleted = delete_from_cache(entry_id)

    if azure_deleted or cache_deleted:
        logger.info(f"Deleted compound {entry_id}: azure={azure_deleted}, cache={cache_deleted}")
        return True

    logger.warning(f"Compound {entry_id} not found in any storage")
    return False


# ============================================================================
# UUID-BASED STORAGE FUNCTIONS (Phase 5 - New)
# ============================================================================
# These functions support the new UUID-based storage paths.
# Use these when the Compound record has entry_id and storage_path set.


def get_storage_path_from_entry_id(entry_id: str) -> str:
    """
    Generate storage path from entry_id (UUID).

    Args:
        entry_id: UUID string (e.g., "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c")

    Returns:
        Path like "results/3a/3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c.zip"
    """
    if not entry_id:
        raise ValueError("entry_id cannot be empty")

    entry_id = entry_id.lower()
    prefix = entry_id[:2]
    return f"results/{prefix}/{entry_id}.zip"


def _get_local_cache_path_by_entry_id(entry_id: str) -> Path:
    """Get local cache path for a compound by entry_id.

    Uses subfolder structure: results/{prefix}/{entry_id}.zip
    """
    entry_id = entry_id.lower()
    prefix = entry_id[:2]
    subdir = LOCAL_CACHE_DIR / prefix
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{entry_id}.zip"


def download_result_by_entry_id(entry_id: str, force_refresh: bool = False) -> Optional[bytes]:
    """
    Download a result ZIP file by entry_id (UUID).

    Args:
        entry_id: UUID of the compound entry
        force_refresh: If True, skip cache and download from Azure

    Returns:
        ZIP file bytes, or None if not found
    """
    if not entry_id:
        return None

    cache_path = _get_local_cache_path_by_entry_id(entry_id)

    # Check local cache first (unless force refresh)
    if not force_refresh and cache_path.exists():
        logger.debug(f"Using cached result for entry_id {entry_id}")
        try:
            data = cache_path.read_bytes()
            cache_path.touch()  # LRU: update access time
            return data
        except Exception as e:
            logger.warning(f"Failed to read from cache: {e}")

    # Download from Azure
    if not is_azure_configured():
        return None

    container = _get_container_client()
    if container is None:
        return None

    blob_name = get_storage_path_from_entry_id(entry_id)

    try:
        blob_client = container.get_blob_client(blob_name)

        if not blob_client.exists():
            logger.warning(f"Result {blob_name} not found in Azure")
            return None

        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        logger.info(f"Downloaded entry_id {entry_id} from Azure ({len(data)} bytes)")

        # Cache locally for future use
        try:
            cache_path.write_bytes(data)
            logger.info(f"Cached entry_id {entry_id} locally ({len(data)} bytes)")
            _evict_oldest_from_cache()
        except Exception as e:
            logger.warning(f"Failed to write to cache: {e}")

        return data

    except Exception as e:
        logger.error(f"Failed to download result by entry_id: {e}")
        return None


def download_result_by_storage_path(storage_path: str, entry_id: str = None, force_refresh: bool = False) -> Optional[bytes]:
    """
    Download a result ZIP file by storage path.

    Only UUID-based paths are supported: results/{prefix}/{entry_id}.zip

    Args:
        storage_path: Storage path (e.g., "results/3a/3a4f8c9e-....zip")
        entry_id: Optional entry_id for local caching (if not provided, extracts from path)
        force_refresh: If True, skip cache and download from Azure

    Returns:
        ZIP file bytes, or None if not found
    """
    if not storage_path:
        return None

    # Extract entry_id (UUID) from storage path for local caching
    # storage_path format: results/{prefix}/{entry_id}.zip
    cache_key = Path(storage_path).stem  # The UUID
    if _is_uuid(cache_key):
        # Use proper subfolder structure for UUID-based paths
        prefix = cache_key[:2].lower()
        cache_subdir = LOCAL_CACHE_DIR / prefix
        cache_subdir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_subdir / f"{cache_key}.zip"
    else:
        # Non-UUID path - this shouldn't happen with UUID-only storage
        logger.warning(f"Non-UUID storage path detected: {storage_path}")
        return None

    # Check local cache first (unless force refresh)
    if not force_refresh and cache_path.exists():
        logger.debug(f"Using cached result for storage_path {storage_path}")
        try:
            data = cache_path.read_bytes()
            cache_path.touch()  # LRU: update access time
            return data
        except Exception as e:
            logger.warning(f"Failed to read from cache: {e}")

    # Download from Azure
    if not is_azure_configured():
        return None

    container = _get_container_client()
    if container is None:
        return None

    try:
        blob_client = container.get_blob_client(storage_path)

        if not blob_client.exists():
            logger.warning(f"Result {storage_path} not found in Azure")
            return None

        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        logger.info(f"Downloaded {storage_path} from Azure ({len(data)} bytes)")

        # Cache locally for future use
        try:
            cache_path.write_bytes(data)
            logger.info(f"Cached {storage_path} locally ({len(data)} bytes)")
            _evict_oldest_from_cache()
        except Exception as e:
            logger.warning(f"Failed to write to cache: {e}")

        return data

    except Exception as e:
        logger.error(f"Failed to download result by storage_path: {e}")
        return None


def load_result_dataframe_by_entry_id(entry_id: str, filename: str = "similar_compounds.csv") -> Optional[pd.DataFrame]:
    """
    Load a specific CSV from a result ZIP file by entry_id.

    Args:
        entry_id: UUID of the compound entry
        filename: CSV filename within the ZIP

    Returns:
        DataFrame or None if not found
    """
    zip_data = download_result_by_entry_id(entry_id)
    if zip_data is None:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            if filename in zf.namelist():
                with zf.open(filename) as f:
                    return pd.read_csv(f)
            else:
                logger.warning(f"{filename} not found in {entry_id}.zip")
                return None

    except Exception as e:
        logger.error(f"Failed to extract {filename}: {e}")
        return None


def load_result_json_by_entry_id(entry_id: str, filename: str = "summary.json") -> Optional[Dict]:
    """
    Load a specific JSON from a result ZIP file by entry_id.

    Args:
        entry_id: UUID of the compound entry
        filename: JSON filename within the ZIP

    Returns:
        Dict or None if not found
    """
    zip_data = download_result_by_entry_id(entry_id)
    if zip_data is None:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            if filename in zf.namelist():
                with zf.open(filename) as f:
                    return json.load(f)
            else:
                logger.warning(f"{filename} not found in {entry_id}.zip")
                return None

    except Exception as e:
        logger.error(f"Failed to extract {filename}: {e}")
        return None


def smart_download_result(
    entry_id: str = None,
    storage_path: str = None,
    force_refresh: bool = False
) -> Optional[bytes]:
    """
    Smart download that tries multiple strategies.

    Priority:
    1. storage_path (if provided - most reliable, from database)
    2. entry_id (if provided - UUID-based path)

    Args:
        entry_id: UUID of the compound entry
        storage_path: Full storage path from database (most reliable)
        force_refresh: If True, skip cache

    Returns:
        ZIP file bytes, or None if not found
    """
    # Try storage_path first (most reliable)
    if storage_path:
        data = download_result_by_storage_path(storage_path, entry_id, force_refresh)
        if data:
            return data

    # Try entry_id-based path
    if entry_id:
        data = download_result_by_entry_id(entry_id, force_refresh)
        if data:
            return data

    logger.warning("smart_download_result requires either storage_path or entry_id")
    return None


def smart_load_summary(
    entry_id: str = None,
    storage_path: str = None,
    force_refresh: bool = False
) -> Optional[Dict]:
    """
    Smart summary loader that tries multiple strategies.

    Priority:
    1. storage_path (if provided - most reliable, from database)
    2. entry_id (if provided - UUID-based path)

    Args:
        entry_id: UUID of the compound entry
        storage_path: Full storage path from database (most reliable)
        force_refresh: If True, skip cache

    Returns:
        Summary dict, or None if not found
    """
    zip_data = smart_download_result(
        entry_id=entry_id,
        storage_path=storage_path,
        force_refresh=force_refresh
    )

    if zip_data is None:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            if "summary.json" in zf.namelist():
                with zf.open("summary.json") as f:
                    return json.load(f)
            else:
                logger.warning(f"summary.json not found in ZIP (storage_path={storage_path}, entry_id={entry_id})")
                return None

    except Exception as e:
        logger.error(f"Failed to extract summary.json: {e}")
        return None


def smart_load_dataframe(
    filename: str,
    entry_id: str = None,
    storage_path: str = None,
    force_refresh: bool = False
) -> Optional[pd.DataFrame]:
    """
    Smart dataframe loader that tries multiple strategies.

    Priority:
    1. storage_path (if provided - most reliable, from database)
    2. entry_id (if provided - UUID-based path)

    Args:
        filename: CSV filename within the ZIP
        entry_id: UUID of the compound entry
        storage_path: Full storage path from database (most reliable)
        force_refresh: If True, skip cache

    Returns:
        DataFrame, or None if not found
    """
    zip_data = smart_download_result(
        entry_id=entry_id,
        storage_path=storage_path,
        force_refresh=force_refresh
    )

    if zip_data is None:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
            if filename in zf.namelist():
                with zf.open(filename) as f:
                    return pd.read_csv(f)
            else:
                logger.warning(f"{filename} not found in ZIP (storage_path={storage_path}, entry_id={entry_id})")
                return None

    except Exception as e:
        logger.error(f"Failed to extract {filename}: {e}")
        return None
