"""Add updated_at auto-update trigger for jobs table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-20
"""
from alembic import op

# revision identifiers
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create reusable trigger function
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    # Attach trigger to jobs table
    op.execute("""
        CREATE TRIGGER update_jobs_updated_at
            BEFORE UPDATE ON jobs
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS update_jobs_updated_at ON jobs;")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
