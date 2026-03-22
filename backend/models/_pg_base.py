"""Temporary home for PGBase (moves to __init__.py in Plan 03).

PGBase is the declarative base for all Postgres ORM models (v2.2+).
It is intentionally separate from the old ``Base`` in ``backend.core.database``
so old and new models can coexist during migration.
"""

from sqlalchemy.orm import DeclarativeBase

__all__ = ["PGBase"]


class PGBase(DeclarativeBase):
    """Base class for all Postgres ORM models (v2.2+)."""

    pass
