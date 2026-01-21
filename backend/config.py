"""
Backend configuration with environment variable support.
Simplified for single-container deployment (local, HF Spaces, Streamlit Cloud, etc.)
"""
from pathlib import Path
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Impulator"
    APP_VERSION: str = "2.1.0"  # Updated version after fixes
    DEBUG: bool = False

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    FRONTEND_PORT: int = 7860

    # Database
    DATABASE_URL: str = "sqlite:///./data/impulator.db"
    DB_POOL_TIMEOUT: int = 30  # Connection timeout in seconds
    DB_ECHO: bool = False  # SQL logging (disabled in production)

    # Executor (ThreadPoolExecutor for background jobs)
    MAX_WORKERS: int = 2  # Concurrent job limit
    JOB_TIMEOUT: int = 3600  # 1 hour max per job

    # Security
    REQUIRE_SESSION_VALIDATION: bool = True  # Validate session IDs
    SESSION_TOKEN_EXPIRY: int = 86400  # 24 hours
    MAX_CONCURRENT_SESSIONS: int = 10000  # Rate limiter max sessions

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW: int = 60  # Window in seconds
    RATE_LIMIT_MAX_JOBS: int = 10  # Max single jobs per window
    RATE_LIMIT_MAX_BATCH: int = 3  # Max batch submissions per window
    MAX_BATCH_SIZE: int = 1000  # Max compounds per batch

    # Cache (in-memory with TTL)
    CACHE_SIZE: int = 2000  # Per function cache size
    CACHE_TTL_SECONDS: int = 3600  # 1 hour TTL

    # External APIs
    CHEMBL_API_URL: str = "https://www.ebi.ac.uk/chembl/api/data"
    PDB_API_URL: str = "https://search.rcsb.org/rcsbsearch/v2/query"
    CHEMBL_RATE_LIMIT: float = 10.0  # Requests per second
    PDB_RATE_LIMIT: float = 5.0  # Requests per second
    API_TIMEOUT: int = 60  # Timeout for external API calls
    API_RETRY_COUNT: int = 3  # Retry count for failed requests

    # Storage
    DATA_DIR: Path = Path("./data")
    RESULTS_DIR: Path = Path("./data/results")

    # Azure Blob (single source of truth)
    AZURE_CONNECTION_STRING: str = ""
    AZURE_CONTAINER: str = "impulator"

    # CORS (comma-separated string in .env, parsed to list)
    # Includes HF Spaces domain pattern and local development
    CORS_ORIGINS: str = "http://localhost:7860,http://localhost:8501,https://*.hf.space"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS string into list."""
        if not self.CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience exports
settings = get_settings()
