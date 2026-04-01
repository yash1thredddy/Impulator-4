"""baseline schema from sql-first design (phases 10-12)

Revision ID: 0001
Revises:
Create Date: 2026-03-20

Schema inventory at baseline:
- Tables (4): jobs, compounds, deleted_compounds, audit_events
- ENUMs (3): job_status, job_type, audit_event_type
- Triggers (1): trg_reparent_on_delete ON compounds
- Functions (1): reparent_compound_children()
- Indexes (22): see DDL below
- CHECK constraints (6): chk_threshold_range, chk_no_self_parent, chk_compound_threshold_range,
  chk_version_positive, chk_root_has_no_parent, uix_job_session_idempotency
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ================================================================
    # Step 0: Ensure gen_random_uuid() is available
    # Built-in on Postgres 13+, but pgcrypto covers older versions
    # ================================================================
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ================================================================
    # Step 1: Create ENUMs (IF NOT EXISTS for compatibility with
    # pre-Alembic databases that already have the schema)
    # ================================================================
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE job_status AS ENUM (
                'pending', 'processing', 'completed', 'failed', 'cancelled'
            );
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE job_type AS ENUM ('single', 'batch');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE audit_event_type AS ENUM (
            'rate_limit_exceeded',
            'job_cancelled',
            'job_deleted',
            'batch_cancelled',
            'validation_failed',
            'path_traversal_blocked',
            'invalid_smiles',
            'authentication_failed',
            'authorization_failed',
            'suspicious_input',
            'compound_created',
            'compound_deleted',
            'job_created'
        );
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$
    """)

    # ================================================================
    # Step 2: Create tables (in FK order)
    # ================================================================

    # --- jobs table (20 columns) ---
    op.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL,
            compound_name VARCHAR(255) NOT NULL,
            smiles TEXT,
            similarity_threshold INTEGER NOT NULL DEFAULT 90,
            activity_types TEXT[],
            status job_status NOT NULL DEFAULT 'pending',
            job_type job_type NOT NULL DEFAULT 'single',
            batch_id UUID,
            batch_index INTEGER,
            idempotency_key VARCHAR(255),
            current_step TEXT,
            progress REAL DEFAULT 0.0,
            error_message TEXT,
            result_summary JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            cancelled_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT chk_threshold_range CHECK (similarity_threshold BETWEEN 40 AND 100),
            CONSTRAINT uix_job_session_idempotency UNIQUE (session_id, idempotency_key)
        )
    """)

    # --- compounds table ---
    op.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            entry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
            compound_name VARCHAR(255) NOT NULL,
            chembl_id VARCHAR(50),
            smiles TEXT,
            canonical_smiles TEXT,
            inchikey VARCHAR(27),
            inchikey_structure_key VARCHAR(25),
            parent_id UUID REFERENCES compounds(entry_id) ON DELETE NO ACTION,
            version INTEGER NOT NULL DEFAULT 1,
            config_diff JSONB,
            imp_score REAL,
            similar_compounds INTEGER NOT NULL DEFAULT 0,
            total_activities INTEGER NOT NULL DEFAULT 0,
            imp_candidates INTEGER NOT NULL DEFAULT 0,
            qed REAL,
            num_outliers INTEGER NOT NULL DEFAULT 0,
            similarity_threshold INTEGER NOT NULL DEFAULT 90,
            activity_types TEXT[],
            author_name VARCHAR(100),
            storage_path TEXT,
            processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT chk_no_self_parent CHECK (entry_id != parent_id),
            CONSTRAINT chk_compound_threshold_range CHECK (similarity_threshold BETWEEN 40 AND 100),
            CONSTRAINT chk_version_positive CHECK (version >= 1),
            CONSTRAINT chk_root_has_no_parent CHECK (
                (version = 1 AND parent_id IS NULL) OR (version > 1 AND parent_id IS NOT NULL)
            )
        )
    """)

    # --- deleted_compounds table ---
    op.execute("""
        CREATE TABLE IF NOT EXISTS deleted_compounds (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            entry_id UUID NOT NULL,
            job_id UUID,
            compound_name VARCHAR(255) NOT NULL,
            chembl_id VARCHAR(50),
            smiles TEXT,
            canonical_smiles TEXT,
            inchikey VARCHAR(27),
            inchikey_structure_key VARCHAR(25),
            parent_id UUID,
            version INTEGER,
            config_diff JSONB,
            imp_score REAL,
            similar_compounds INTEGER,
            total_activities INTEGER,
            imp_candidates INTEGER,
            qed REAL,
            num_outliers INTEGER,
            similarity_threshold INTEGER,
            activity_types TEXT[],
            author_name VARCHAR(100),
            storage_path TEXT,
            original_processed_at TIMESTAMPTZ,
            deleted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            deleted_by UUID,
            deletion_reason VARCHAR(255)
        )
    """)

    # --- audit_events table ---
    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            event_type audit_event_type NOT NULL,
            session_id UUID,
            severity VARCHAR(20) NOT NULL DEFAULT 'warning',
            details JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # ================================================================
    # Step 3: Create indexes (22 total)
    # ================================================================

    # --- Jobs indexes (6) ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_session_id ON jobs (session_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_batch_id ON jobs (batch_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_pending ON jobs (status) WHERE status = 'pending'")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_processing ON jobs (status) WHERE status = 'processing'")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_completed_at ON jobs (status, completed_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_jobs_activity_types ON jobs USING gin (activity_types)")

    # --- Compounds indexes (10) ---
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_compounds_job_id ON compounds (job_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_compounds_inchikey ON compounds (inchikey)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_compounds_structure_key ON compounds (inchikey_structure_key)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_compounds_parent_id ON compounds (parent_id) WHERE parent_id IS NOT NULL")
    op.execute("CREATE INDEX IF NOT EXISTS idx_compounds_name ON compounds (compound_name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_compounds_roots ON compounds (processed_at DESC) WHERE parent_id IS NULL")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_compounds_parent_version ON compounds (parent_id, version) WHERE parent_id IS NOT NULL")
    op.execute("CREATE INDEX IF NOT EXISTS idx_compounds_activity_types ON compounds USING gin (activity_types)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_compounds_name_lower ON compounds (lower(trim(compound_name)))")
    op.execute("CREATE INDEX IF NOT EXISTS idx_compounds_chembl_id ON compounds (chembl_id) WHERE chembl_id IS NOT NULL")

    # --- Deleted_compounds indexes (3) ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_deleted_compounds_entry_id ON deleted_compounds (entry_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_deleted_compounds_deleted_at ON deleted_compounds (deleted_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_deleted_compounds_inchikey ON deleted_compounds (inchikey)")

    # --- Audit_events indexes (3) ---
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_type_created ON audit_events (event_type, created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_session_id ON audit_events (session_id) WHERE session_id IS NOT NULL")
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON audit_events (created_at)")

    # ================================================================
    # Step 4: Create reparent function + trigger
    # ================================================================
    op.execute("""
        CREATE OR REPLACE FUNCTION reparent_compound_children()
        RETURNS TRIGGER AS $$
        DECLARE
            v_promoted_id UUID;
        BEGIN
            -- Only act if the deleted compound is a parent (root of version tree)
            IF OLD.parent_id IS NOT NULL THEN
                RETURN OLD;  -- Not a parent, nothing to reparent
            END IF;

            -- Find the next oldest child to promote (lowest version, then earliest processed)
            SELECT entry_id INTO v_promoted_id
            FROM compounds
            WHERE parent_id = OLD.entry_id
            ORDER BY version ASC, processed_at ASC
            LIMIT 1;

            -- No children, nothing to do
            IF v_promoted_id IS NULL THEN
                RETURN OLD;
            END IF;

            -- Reparent remaining children to the promoted compound
            UPDATE compounds
            SET parent_id = v_promoted_id
            WHERE parent_id = OLD.entry_id
              AND entry_id != v_promoted_id;

            -- Promote the chosen child to parent status
            UPDATE compounds
            SET parent_id = NULL, version = 1
            WHERE entry_id = v_promoted_id;

            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql
    """)

    op.execute("DROP TRIGGER IF EXISTS trg_reparent_on_delete ON compounds")
    op.execute("""
        CREATE TRIGGER trg_reparent_on_delete
            BEFORE DELETE ON compounds
            FOR EACH ROW
            EXECUTE FUNCTION reparent_compound_children()
    """)

    # ================================================================
    # Step 5: Enforce flat version trees
    # Prevents deep nesting (grandchildren). A compound's parent must
    # be a root (parent_id IS NULL). This makes the reparent trigger
    # safe — it only needs to handle one level of children.
    # ================================================================
    op.execute("""
        CREATE OR REPLACE FUNCTION enforce_flat_version_tree()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.parent_id IS NOT NULL THEN
                IF EXISTS (
                    SELECT 1 FROM compounds
                    WHERE entry_id = NEW.parent_id
                      AND parent_id IS NOT NULL
                ) THEN
                    RAISE EXCEPTION 'Cannot create nested version: parent is not a root compound';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)

    op.execute("DROP TRIGGER IF EXISTS trg_enforce_flat_tree ON compounds")
    op.execute("""
        CREATE TRIGGER trg_enforce_flat_tree
            BEFORE INSERT OR UPDATE OF parent_id ON compounds
            FOR EACH ROW
            EXECUTE FUNCTION enforce_flat_version_tree()
    """)


def downgrade() -> None:
    raise RuntimeError("Cannot downgrade past baseline")
