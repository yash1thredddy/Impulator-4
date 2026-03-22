"""
Core backend modules.
- database: Postgres with SQLAlchemy
- executor: Async task executor for background jobs
- azure_sync: Azure Blob storage sync utilities
"""
import hashlib
import logging
import re

from backend.core.database import get_db, get_db_session
from backend.core.azure_sync import (
    # UUID-based storage functions (only storage method supported)
    upload_result_to_azure_by_entry_id,
    download_result_from_azure_by_entry_id,
    delete_result_from_azure_by_entry_id,
    check_result_exists_in_azure_by_entry_id,
    get_storage_path_from_entry_id,
    list_results_in_azure,
    is_azure_configured,
)

logger = logging.getLogger(__name__)

# Maximum length for sanitized compound names (Windows path limit consideration)
MAX_SANITIZED_NAME_LENGTH = 100


def sanitize_compound_name(name: str, add_hash_suffix: bool = False) -> str:
    """
    Sanitize compound name for filesystem and Azure storage.

    Consistently handles special characters to ensure files can be:
    - Saved to local filesystem
    - Uploaded to Azure blob storage
    - Retrieved correctly by name lookup

    Args:
        name: Raw compound name (e.g., "Aspirin (acetyl)", "Test/Compound")
        add_hash_suffix: If True, adds a short hash to prevent collisions

    Returns:
        Safe name with only alphanumeric, dash, and underscore characters

    Note:
        Different names can map to the same sanitized name (collision).
        E.g., "Hello@World" and "Hello#World" both become "Hello_World".
        Use add_hash_suffix=True for critical paths where collisions matter.
    """
    if not name:
        return 'unnamed_compound'

    # Replace common separators with underscore
    safe = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    # Replace any remaining non-alphanumeric chars (except - and _) with underscore
    safe = re.sub(r'[^a-zA-Z0-9\-_]', '_', safe)
    # Collapse multiple underscores
    safe = re.sub(r'_+', '_', safe)
    # Strip leading/trailing underscores
    safe = safe.strip('_')

    if not safe:
        safe = 'unnamed_compound'

    # Log warning if significant transformation occurred
    if safe.lower() != name.lower().replace(' ', '_').replace('-', '_'):
        logger.debug(f"Compound name sanitized: '{name}' -> '{safe}'")

    # Enforce max length to avoid path issues
    if len(safe) > MAX_SANITIZED_NAME_LENGTH:
        # Truncate and add hash to maintain uniqueness
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        safe = safe[:MAX_SANITIZED_NAME_LENGTH - 9] + '_' + name_hash
        logger.warning(f"Compound name truncated: '{name}' -> '{safe}'")

    # Optionally add hash suffix to prevent collisions
    if add_hash_suffix:
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        if len(safe) + 9 <= MAX_SANITIZED_NAME_LENGTH:
            safe = f"{safe}_{name_hash}"
        else:
            safe = safe[:MAX_SANITIZED_NAME_LENGTH - 9] + '_' + name_hash

    return safe


__all__ = [
    # Database
    "get_db",
    "get_db_session",
    # Azure (blob storage)
    "is_azure_configured",
    # Azure (UUID-based storage - only storage method supported)
    "upload_result_to_azure_by_entry_id",
    "download_result_from_azure_by_entry_id",
    "delete_result_from_azure_by_entry_id",
    "check_result_exists_in_azure_by_entry_id",
    "get_storage_path_from_entry_id",
    "list_results_in_azure",
    # Utilities
    "sanitize_compound_name",
]
