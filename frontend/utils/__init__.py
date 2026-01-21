"""Utilities for IMPULATOR frontend."""
import re

from frontend.utils.session_state import (
    SessionState,
    ActiveJob,
    VIEW_HOME,
    VIEW_ANALYZE,
    VIEW_COMPOUND_DETAILS,
)


def sanitize_compound_name(name: str) -> str:
    """
    Sanitize compound name for filesystem and Azure storage.

    Must match backend.core.sanitize_compound_name exactly for consistency.

    Args:
        name: Raw compound name (e.g., "Aspirin (acetyl)", "Test/Compound")

    Returns:
        Safe name with only alphanumeric, dash, and underscore characters
    """
    # Replace common separators with underscore
    safe = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    # Replace any remaining non-alphanumeric chars (except - and _) with underscore
    safe = re.sub(r'[^a-zA-Z0-9\-_]', '_', safe)
    # Collapse multiple underscores
    safe = re.sub(r'_+', '_', safe)
    # Strip leading/trailing underscores
    safe = safe.strip('_')
    return safe if safe else 'unnamed_compound'


from frontend.utils.exceptions import (
    ImpulatorError,
    InvalidSMILESError,
    ProcessingError,
    FileValidationError,
    ConversionError,
    APIError,
    RateLimitError,
    JobError,
    JobSubmissionError,
    JobTimeoutError,
    JobCancelledError,
    BackendUnavailableError,
)
from frontend.utils.validators import (
    ValidationResult,
    InputValidator,
    FileValidator,
    DataFrameValidator,
)
from frontend.utils.cache import (
    TTLCache,
    cached,
    cache_molecule,
    cache_conversion,
    molecule_cache,
    conversion_cache,
    api_cache,
    get_all_cache_stats,
    clear_all_caches,
)

__all__ = [
    # Utilities
    "sanitize_compound_name",
    # Session state
    "SessionState",
    "ActiveJob",
    "VIEW_HOME",
    "VIEW_ANALYZE",
    "VIEW_COMPOUND_DETAILS",
    # Exceptions
    "ImpulatorError",
    "InvalidSMILESError",
    "ProcessingError",
    "FileValidationError",
    "ConversionError",
    "APIError",
    "RateLimitError",
    "JobError",
    "JobSubmissionError",
    "JobTimeoutError",
    "JobCancelledError",
    "BackendUnavailableError",
    # Validators
    "ValidationResult",
    "InputValidator",
    "FileValidator",
    "DataFrameValidator",
    # Cache
    "TTLCache",
    "cached",
    "cache_molecule",
    "cache_conversion",
    "molecule_cache",
    "conversion_cache",
    "api_cache",
    "get_all_cache_stats",
    "clear_all_caches",
]
