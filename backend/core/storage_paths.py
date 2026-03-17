"""Single source of truth for storage path computation (ARCH-19).

All modules that need to convert an entry_id to a storage path should import
from here. backend.core.azure_sync re-exports this for backward compatibility.
"""
import structlog

logger = structlog.get_logger(__name__)


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
