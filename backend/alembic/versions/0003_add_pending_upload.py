"""Add pending_upload enum value, upload_attempts, requeue_count columns.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-21
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ALTER TYPE ADD VALUE cannot run inside a transaction (Postgres restriction).
    # We must end the current transaction, add the value, then add columns
    # in a fresh transaction.
    op.execute("COMMIT")
    op.execute("ALTER TYPE job_status ADD VALUE IF NOT EXISTS 'pending_upload'")

    # Add columns outside the Alembic-managed transaction.
    # Each op.add_column runs its own implicit transaction.
    op.add_column(
        "jobs",
        sa.Column("upload_attempts", sa.Integer, server_default="0", nullable=False),
    )
    op.add_column(
        "jobs",
        sa.Column("requeue_count", sa.Integer, server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("jobs", "requeue_count")
    op.drop_column("jobs", "upload_attempts")
    # Cannot remove enum value in Postgres -- leave it
