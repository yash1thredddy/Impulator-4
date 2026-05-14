"""add imp_zip_backfill_status tracking table

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-14

Phase 21 / Plan 21-05 — IMP Score Presentation Overhaul (column-shape backfill).

Introduces a tracking table that records the per-compound state of a one-shot
backfill job that rewrites historical Azure-Blob result ZIPs to the new
column shape (drops ``IMP_Classification`` / ``IMP_Confidence``; adds
``IMP_Score_Integer``).

Design notes
------------
- The tracking table is keyed on ``entry_id`` (FK to ``compounds.entry_id``)
  rather than ``job_id`` because:

  * Result ZIPs are addressed by ``entry_id`` (see
    ``backend.core.storage_paths.get_storage_path_from_entry_id``) -- the
    same key used by the Azure helpers we reuse
    (``download_result_from_azure_by_entry_id`` /
    ``upload_result_to_azure_by_entry_id``).
  * Each compound has its own blob; per-compound state gives finer
    idempotence than per-job state when a job produced multiple compounds
    in earlier-design schemas.

- State machine: ``{pending, done, failed, skipped}`` (T-21-12).
  ``pending`` rows are LEFT-JOINed at runtime (a missing row == pending)
  so we don't need to pre-seed every existing compound on migrate.

- Implemented as a Postgres ENUM type for consistency with the project's
  other state machines (``job_status``, ``job_type``, ``audit_event_type``).

- Failure rows persist a short ``error_message`` (``str(e)``, not a full
  traceback -- T-21-11). Full stack goes to structlog only.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Step 1: Create ENUM (idempotent guard for pre-Alembic databases)
    # ------------------------------------------------------------------
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE imp_zip_backfill_status_enum AS ENUM (
                'pending', 'done', 'failed', 'skipped'
            );
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
        """
    )

    # ------------------------------------------------------------------
    # Step 2: Create tracking table
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS imp_zip_backfill_status (
            entry_id UUID PRIMARY KEY
                REFERENCES compounds(entry_id) ON DELETE CASCADE,
            status imp_zip_backfill_status_enum
                NOT NULL DEFAULT 'pending',
            error_message TEXT,
            processed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # ------------------------------------------------------------------
    # Step 3: Indexes
    # ------------------------------------------------------------------
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_imp_zip_backfill_status_status "
        "ON imp_zip_backfill_status (status)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS imp_zip_backfill_status")
    op.execute("DROP TYPE IF EXISTS imp_zip_backfill_status_enum")
