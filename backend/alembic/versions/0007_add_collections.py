"""Add JobType.COLLECTION enum value + collections table.

Lands the Phase 23 schema foundation (HC-2): the ``collection`` value on the
``job_type`` Postgres ENUM and the dedicated ``collections`` table whose member
input-definitions live in a ``members_config`` JSONB column (D-02), with
soft-delete columns (D-11) and a 1:1 unique FK to its job.

Follows the ``0003_add_pending_upload`` enum-add pattern verbatim: end the
current transaction (``COMMIT``) before ``ALTER TYPE ... ADD VALUE`` (a Postgres
restriction), then create the table in a fresh transaction. No row using the new
``collection`` value is inserted in the same transaction, and no ``server_default``
references the new enum value.

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-29
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ALTER TYPE ADD VALUE cannot run inside a transaction (Postgres restriction).
    # We must end the current transaction, add the value, then create the table
    # in a fresh transaction.
    op.execute("COMMIT")
    op.execute("ALTER TYPE job_type ADD VALUE IF NOT EXISTS 'collection'")

    op.create_table(
        "collections",
        # Identity
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # Descriptive metadata
        sa.Column("name", sa.String(255), nullable=False),  # D-07: indexed, not unique
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("author_name", sa.String(100), nullable=False),
        # Linked job (1:1). Members are compounds sharing this job_id.
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        # D-02: member input-definitions (NOT job.result_summary)
        sa.Column("members_config", postgresql.JSONB, nullable=True),
        # Storage
        sa.Column("storage_path", sa.String(500), nullable=True),
        # Summary statistics
        sa.Column("compound_count", sa.Integer, server_default="0", nullable=False),
        # D-09: members that failed compute without failing the parent job
        sa.Column("member_failed_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("avg_imp_score", sa.Float, nullable=True),
        sa.Column("imp_candidate_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("unique_targets", sa.Integer, server_default="0", nullable=False),
        # Soft delete (D-11)
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by", postgresql.UUID(as_uuid=True), nullable=True),
        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    # D-07: name is indexed but NOT unique.
    op.create_index("ix_collections_name", "collections", ["name"])

    # Note: the ORM declares ``updated_at`` with ``onupdate=func.now()``; since all
    # writes go through the sync ORM/repository layer, that suffices and we do NOT
    # attach the 0002 update_updated_at_column() trigger to this table.


def downgrade() -> None:
    op.drop_index("ix_collections_name", table_name="collections")
    op.drop_table("collections")
    # Cannot remove the 'collection' enum value in Postgres -- leave it
