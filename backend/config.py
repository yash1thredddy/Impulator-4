"""
Backend configuration with environment variable support.
Simplified for single-container deployment (local, HF Spaces, Streamlit Cloud, etc.)
"""
from pathlib import Path
from functools import lru_cache
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    
    APP_NAME: str = "Impulator"
    APP_VERSION: str = "2.2.0-dev"
    DEBUG: bool = False

    # Logging
    LOG_LEVEL: str = "INFO"        # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "auto"       # "json", "console", or "auto" (json when not DEBUG, console when DEBUG)

    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    FRONTEND_PORT: int = 7860

    # Database
    DATABASE_URL: str = ""  # Must be set via env var (postgresql://...)
    DB_POOL_SIZE: int = 5  # Connection pool size
    DB_MAX_OVERFLOW: int = 5  # Max overflow connections beyond pool_size
    DB_POOL_TIMEOUT: int = 10  # Connection pool checkout timeout in seconds
    DB_ECHO: bool = False  # SQL logging (disabled in production)
    DIRECT_DATABASE_URL: str = ""  # Direct Supabase connection (port 5432) for DDL/migrations. Falls back to DATABASE_URL if empty.

    # Testing mode (CI compatibility -- allows SQLite)
    TESTING: bool = False

    # Executor (async job processing)
    MAX_CONCURRENT_JOBS: int = 10  # Max concurrent async jobs (asyncio.Semaphore)
    # D-04: a COLLECTION job holds exactly ONE global executor slot and fans its
    # members out under this LOCAL asyncio.Semaphore -- members are awaited
    # coroutines, NEVER executor.submit (re-entry deadlock against the 1 global slot).
    COLLECTION_MEMBER_CONCURRENCY: int = 4
    JOB_TIMEOUT: int = 3600  # 1 hour max per job
    SHUTDOWN_TIMEOUT: int = 25  # Seconds to wait for in-flight jobs (HF Spaces has 30s grace)

    # Security
    REQUIRE_SESSION_VALIDATION: bool = True  # Validate session IDs
    ADMIN_API_KEY: SecretStr = SecretStr("")  # Required for admin endpoints (migrate, etc.)

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW: int = 60  # Window in seconds
    RATE_LIMIT_MAX_JOBS: int = 10  # Max single jobs per window
    RATE_LIMIT_MAX_BATCH: int = 3  # Max batch submissions per window
    MAX_BATCH_SIZE: int = 1000  # Max compounds per batch

    # Cache (in-memory with TTL)
    CACHE_SIZE: int = 500  # Bounded at 500 to limit memory (STAB-04)
    CACHE_TTL_SECONDS: int = 3600  # 1 hour TTL

    # ClassyFire pause toggle (2026-05-30).
    # TEMPORARY OPERATIONAL PAUSE -- not a permanent feature default. All three
    # ClassyFire mirrors (Fiehn/GNPS/Wishart) are flapping/429-ing, so collection
    # and single jobs thrash on retry-backoff. Default False pauses ClassyFire in
    # ALL environments via code alone (we don't edit .env). NPClassifier is a
    # different, healthy endpoint and stays live. Re-enable when mirrors recover:
    # set env CLASSYFIRE_ENABLED=true, or flip this default back to True.
    CLASSYFIRE_ENABLED: bool = False

    # External APIs
    CHEMBL_API_URL: str = "https://www.ebi.ac.uk/chembl/api/data"
    PDB_API_URL: str = "https://search.rcsb.org/rcsbsearch/v2/query"
    API_TIMEOUT: int = 60  # Timeout for external API calls

    # httpx connection limits (Phase 19.1)
    CHEMBL_MAX_CONNECTIONS: int = 30   # httpx.Limits max_connections -- global TCP cap to ChEMBL (D-26)
    CHEMBL_MAX_PER_JOB: int = 10      # asyncio.Semaphore per compound job -- limits concurrent ChEMBL requests (D-25/D-27)

    # Storage
    DATA_DIR: Path = Path("./data")
    RESULTS_DIR: Path = Path("./data/results")

    # Azure Blob (single source of truth)
    AZURE_CONNECTION_STRING: str = ""
    AZURE_CONTAINER: str = "impulator"

    # CORS (comma-separated string in .env, parsed to list)
    # HF Spaces origins are handled by allow_origin_regex in CORS middleware,
    # not this list (Starlette does exact matching on allow_origins)
    CORS_ORIGINS: str = "http://localhost:7860,http://localhost:8501"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS string into list."""
        if not self.CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v: str, info) -> str:
        """Validate DATABASE_URL is a Postgres connection string.

        In TESTING mode (CI), allow SQLite URLs for backward compatibility
        until Phase 20 migrates tests to Postgres.
        """
        # Check if TESTING mode -- allow SQLite for CI
        # info.data contains already-validated fields; TESTING may not be there yet
        # so also check the environment variable directly
        import os
        testing = os.environ.get("TESTING", "").lower() in ("true", "1", "yes")
        if testing:
            # In TESTING mode, skip postgresql:// validation.
            # Tests provide their own DATABASE_URL via session fixtures.
            # Return a valid-looking placeholder so create_engine() doesn't
            # crash at import time (fixtures override it before any DB access).
            return v if v else "postgresql://test:test@localhost:5432/test_placeholder"

        if not v:
            raise ValueError(
                "DATABASE_URL must be set. "
                "Get your connection string from Supabase Dashboard > Settings > Database."
            )
        # Normalize postgres:// to postgresql:// (Supabase provides postgres://)
        if v.startswith("postgres://"):
            v = "postgresql://" + v[len("postgres://"):]
        if not v.startswith("postgresql://"):
            raise ValueError(
                "DATABASE_URL must be a PostgreSQL connection string (postgresql://...)"
            )
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore unknown env vars (e.g. SUPABASE_URL still in .env)
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience exports
settings = get_settings()
