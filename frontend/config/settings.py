"""
IMPULATOR Frontend Configuration.

Frozen dataclass for immutable configuration with environment overrides.
All magic numbers and configuration values should be defined here.

Environment variables can override defaults (read at module import time):
- API_BASE_URL: Backend API URL
- API_TIMEOUT_SECONDS: Override API timeout
- APP_VERSION: Override version string
- JOB_POLL_INTERVAL_MS: Job polling interval
"""

import os
from dataclasses import dataclass, field
from typing import FrozenSet


def _get_int_env(name: str, default: int) -> int:
    """Get integer environment variable or return default."""
    val = os.getenv(name)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _get_str_env(name: str, default: str) -> str:
    """Get string environment variable or return default."""
    return os.getenv(name, default)


def _get_bool_env(name: str, default: bool) -> bool:
    """Get boolean environment variable or return default."""
    val = os.getenv(name)
    if val is not None:
        return val.lower() in ('true', '1', 'yes')
    return default


@dataclass(frozen=True)
class ImpulatorConfig:
    """Immutable IMPULATOR configuration.

    frozen=True ensures config values cannot be accidentally modified.
    Environment variables are read at module import time.
    """

    # Application
    APP_NAME: str = "IMPULATOR"
    APP_ICON: str = "🧬"
    APP_VERSION: str = field(
        default_factory=lambda: _get_str_env('APP_VERSION', "2.0.0")
    )

    # Backend API
    API_BASE_URL: str = field(
        default_factory=lambda: _get_str_env('API_BASE_URL', 'http://localhost:8000')
    )
    API_TIMEOUT_SECONDS: int = field(
        default_factory=lambda: _get_int_env('API_TIMEOUT_SECONDS', 30)
    )
    MAX_RETRY_ATTEMPTS: int = 3

    # Job polling (6 sec to match backend scheduler interval)
    JOB_POLL_INTERVAL_MS: int = field(
        default_factory=lambda: _get_int_env('JOB_POLL_INTERVAL_MS', 6000)
    )
    JOB_TIMEOUT_SECONDS: int = field(
        default_factory=lambda: _get_int_env('JOB_TIMEOUT_SECONDS', 3600)
    )

    # Default activity types for ChEMBL (matching old code)
    DEFAULT_ACTIVITY_TYPES: tuple = ("IC50", "EC50", "Ki", "Kd", "AC50", "GI50", "MIC")

    # Default similarity threshold
    DEFAULT_SIMILARITY_THRESHOLD: int = 90

    # Cache settings
    CACHE_TTL_SECONDS: int = field(
        default_factory=lambda: _get_int_env('CACHE_TTL_SECONDS', 3600)
    )
    MAX_CACHE_SIZE: int = 1000

    # UI settings
    RESULTS_PER_PAGE: int = 20
    MAX_CHART_POINTS: int = 5000

    # Chart defaults
    DEFAULT_MARKER_SIZE: int = 8
    DEFAULT_CHART_HEIGHT: int = 600

    # Molecule viewer
    MOLECULE_2D_SIZE: tuple = (300, 300)

    # Input validation
    MAX_SMILES_LENGTH: int = 2000
    MAX_COMPOUND_NAME_LENGTH: int = 200

    # File handling
    MAX_CSV_SIZE_MB: int = field(
        default_factory=lambda: _get_int_env('MAX_CSV_SIZE_MB', 10)
    )
    MAX_FILE_SIZE_MB: int = field(
        default_factory=lambda: _get_int_env('MAX_FILE_SIZE_MB', 10)
    )
    ALLOWED_EXTENSIONS: FrozenSet[str] = frozenset({'.csv', '.xlsx'})

    # Batch processing limits
    MAX_ROWS_LIMIT: int = 1000
    MAX_ROWS_WARNING: int = 100
    MAX_CATEGORICAL_CARDINALITY: int = 50

    @property
    def MAX_CSV_SIZE_BYTES(self) -> int:
        """Get maximum file size in bytes."""
        return self.MAX_CSV_SIZE_MB * 1024 * 1024

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        """Get maximum file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def JOB_POLL_INTERVAL_SECONDS(self) -> float:
        """Get poll interval in seconds."""
        return self.JOB_POLL_INTERVAL_MS / 1000.0


# Global immutable config instance
config = ImpulatorConfig()

# Chart type definitions
CHART_TYPES = [
    'Scatter Plot',
    'Box Plot',
    'Violin Plot',
    'Bar Chart',
    'Histogram',
    'Heatmap',
    '3D Scatter',
]

# Color scales for continuous data
COLOR_SCALES = [
    'Viridis',
    'Plasma',
    'Inferno',
    'Magma',
    'Blues',
    'Reds',
    'Greens',
    'RdBu',
    'Spectral',
]

# O[Q/P/L]A Score classifications
OQPLA_CLASSIFICATIONS = {
    'Exceptional IMP': (0.9, 1.0),
    'Strong IMP': (0.7, 0.9),
    'Moderate IMP': (0.5, 0.7),
    'Weak IMP': (0.3, 0.5),
    'Not IMP': (0.0, 0.3),
}
