"""Backfill empty activity_types with default 7 types.

Compounds processed with "default" (all activity types) stored [''] or NULL
instead of the actual list. This migration sets them to the explicit defaults.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-22
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

# The 7 default activity types used by IMPULATOR
DEFAULT_TYPES = ["AC50", "EC50", "GI50", "IC50", "Kd", "Ki", "MIC"]


def upgrade() -> None:
    # Fix compounds with [''] (single empty string element)
    op.execute(
        sa.text(
            "UPDATE compounds SET activity_types = :defaults "
            "WHERE activity_types = ARRAY['']::text[]"
        ).bindparams(defaults=DEFAULT_TYPES)
    )
    # Fix compounds with NULL activity_types
    op.execute(
        sa.text(
            "UPDATE compounds SET activity_types = :defaults "
            "WHERE activity_types IS NULL"
        ).bindparams(defaults=DEFAULT_TYPES)
    )
    # Fix jobs with NULL activity_types
    op.execute(
        sa.text(
            "UPDATE jobs SET activity_types = :defaults "
            "WHERE activity_types IS NULL"
        ).bindparams(defaults=DEFAULT_TYPES)
    )
    # Fix jobs with ['']
    op.execute(
        sa.text(
            "UPDATE jobs SET activity_types = :defaults "
            "WHERE activity_types = ARRAY['']::text[]"
        ).bindparams(defaults=DEFAULT_TYPES)
    )


def downgrade() -> None:
    # No downgrade — data was incorrect before
    pass
