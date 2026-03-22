"""Alembic migration environment.

Reads DATABASE_URL from backend.config.settings (single source of truth).
Supports both online (live DB) and offline (--sql) modes.
Does NOT call fileConfig() -- preserves structlog configuration.
"""
import logging

from alembic import context
from sqlalchemy import engine_from_config, pool, text

# Import settings for DATABASE_URL (single source of truth)
from backend.config import settings

# Skip fileConfig() -- structlog is already configured by backend.core.logging
logger = logging.getLogger("alembic.env")

# Alembic Config object (provides access to alembic.ini values)
config = context.config

# Set sqlalchemy.url from settings (overrides alembic.ini placeholder)
database_url = (
    getattr(settings, "DIRECT_DATABASE_URL", None) or settings.DATABASE_URL
)
if database_url.startswith("postgres://"):
    database_url = "postgresql://" + database_url[len("postgres://"):]
config.set_main_option("sqlalchemy.url", database_url)

# target_metadata -- PGBase contains all Postgres ORM model metadata
from backend.models._pg_base import PGBase
target_metadata = PGBase.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (--sql output).

    Generates SQL statements without connecting to the database.
    Useful for generating migration scripts for review or manual execution.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,              # D-45: detect column type drift
        compare_server_default=True,    # D-45: detect server default drift
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (live database).

    Supports two connection modes:
    1. Programmatic: connection passed via config.attributes (from lifespan)
    2. Standalone: creates engine from alembic.ini config (CLI usage)
    """
    # Check if a connection was passed programmatically (from lifespan)
    connectable = config.attributes.get("connection", None)

    if connectable is not None:
        # Connection provided by caller (e.g., FastAPI lifespan)
        # Disable statement_timeout for DDL -- migrations may take longer than 60s
        connectable.execute(text("SET statement_timeout = 0"))
        context.configure(
            connection=connectable,
            target_metadata=target_metadata,
            compare_type=True,              # D-45: detect column type drift
            compare_server_default=True,    # D-45: detect server default drift
        )
        with context.begin_transaction():
            context.run_migrations()
    else:
        # Standalone mode: create engine from config
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,  # Single-use connection for DDL
        )
        with connectable.connect() as connection:
            # Disable statement_timeout for DDL -- migrations may take longer than 60s
            connection.execute(text("SET statement_timeout = 0"))
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,              # D-45: detect column type drift
                compare_server_default=True,    # D-45: detect server default drift
            )
            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
