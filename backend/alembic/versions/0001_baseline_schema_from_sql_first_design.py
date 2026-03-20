"""baseline schema from sql-first design (phases 10-12)

Revision ID: 0001
Revises:
Create Date: 2026-03-20

Schema inventory at baseline:
- Tables (4): jobs, compounds, deleted_compounds, audit_events
- ENUMs (3): job_status, job_type, audit_event_type
- Triggers (1): trg_reparent_on_delete ON compounds
- Functions (1): reparent_compound_children()
- Indexes (22): [see commented SQL below]
- CHECK constraints (6): chk_threshold_range, chk_no_self_parent, chk_compound_threshold_range,
  chk_version_positive, chk_root_has_no_parent, uix_job_session_idempotency

This is an empty baseline revision. The schema was created via raw SQL
in Supabase during Phases 10-12. This revision marks the starting point
for Alembic-managed migrations.

See commented SQL below for full schema reference.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Schema already exists in Supabase (created via SQL in Phases 10-12).
    # This is a baseline marker -- no DDL operations.

    # === JOBS TABLE (Phase 10, reconstructed from live schema) ===
    # CREATE TABLE jobs (
    #     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    #     session_id UUID NOT NULL,
    #     compound_name VARCHAR(255) NOT NULL,
    #     smiles TEXT,
    #     similarity_threshold INTEGER NOT NULL DEFAULT 90,
    #     activity_types TEXT[],
    #     status job_status NOT NULL DEFAULT 'pending',
    #     job_type job_type NOT NULL DEFAULT 'single',
    #     batch_id UUID,
    #     batch_index INTEGER,
    #     idempotency_key VARCHAR(255),
    #     current_step TEXT,
    #     progress REAL DEFAULT 0.0,
    #     error_message TEXT,
    #     result_summary JSONB,
    #     created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    #     started_at TIMESTAMPTZ,
    #     completed_at TIMESTAMPTZ,
    #     cancelled_at TIMESTAMPTZ,
    #     updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    # );
    # -- CONSTRAINT chk_threshold_range CHECK (similarity_threshold BETWEEN 40 AND 100)
    # -- UNIQUE (session_id, idempotency_key) = uix_job_session_idempotency
    # -- ENUMs: job_status (pending,processing,completed,failed,cancelled), job_type (single,batch)
    # -- Indexes (6): idx_jobs_session_id, idx_jobs_batch_id, idx_jobs_pending,
    # --   idx_jobs_processing, idx_jobs_status_completed_at, idx_jobs_activity_types

    # === COMPOUNDS TABLE (Phase 11, from migrations/002_create_compounds_table.sql) ===
    # -- Migration 002: Create compounds table with versioning + deleted_compounds archive
    # -- Phase 11: Schema -- Compounds + Versioning
    # -- Date: 2026-03-19
    # --
    # -- Replaces SQLite Compound + DeletedCompound models. Key changes:
    # -- 1. entry_id becomes UUID PK (was VARCHAR(36) with separate integer PK)
    # -- 2. is_duplicate/duplicate_of -> parent_id UUID self-FK + version INTEGER
    # -- 3. activity_types comma-separated TEXT -> TEXT[] (Postgres native array)
    # -- 4. New job_id UUID FK -> jobs.id ON DELETE CASCADE (1:1 job link)
    # -- 5. New config_diff JSONB for version diffs
    # -- 6. BEFORE DELETE trigger for atomic reparenting
    # -- 7. deleted_compounds mirrors all compound columns + deletion metadata
    #
    # -- ============================================================
    # -- Step 1: Create compounds table
    # -- ============================================================
    # CREATE TABLE compounds (
    #     -- Identity (entry_id is the primary key, replaces integer id)
    #     entry_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    #
    #     -- Job link (1:1 -- one job creates one compound)
    #     job_id              UUID REFERENCES jobs(id) ON DELETE CASCADE,
    #
    #     -- Compound data
    #     compound_name       VARCHAR(255) NOT NULL,
    #     chembl_id           VARCHAR(50),
    #     smiles              TEXT,
    #     canonical_smiles    TEXT,
    #
    #     -- InChIKey (both full and structure key for different query patterns)
    #     inchikey            VARCHAR(27),
    #     inchikey_structure_key VARCHAR(25),
    #
    #     -- Versioning (replaces is_duplicate/duplicate_of)
    #     parent_id           UUID REFERENCES compounds(entry_id) ON DELETE NO ACTION,
    #     version             INTEGER NOT NULL DEFAULT 1,
    #     config_diff         JSONB,
    #
    #     -- Analysis results (normalized from result_summary)
    #     imp_score           REAL,
    #     similar_compounds   INTEGER NOT NULL DEFAULT 0,
    #     total_activities    INTEGER NOT NULL DEFAULT 0,
    #     imp_candidates      INTEGER NOT NULL DEFAULT 0,
    #     qed                 REAL,
    #     num_outliers        INTEGER NOT NULL DEFAULT 0,
    #
    #     -- Analysis config
    #     similarity_threshold INTEGER NOT NULL DEFAULT 90,
    #     activity_types      TEXT[],
    #
    #     -- Metadata
    #     author_name         VARCHAR(100),
    #     storage_path        TEXT,
    #
    #     -- Timestamps
    #     processed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    #
    #     -- Constraints
    #     CONSTRAINT chk_no_self_parent CHECK (entry_id != parent_id),
    #     CONSTRAINT chk_compound_threshold_range CHECK (similarity_threshold BETWEEN 40 AND 100),
    #     CONSTRAINT chk_version_positive CHECK (version >= 1),
    #     CONSTRAINT chk_root_has_no_parent CHECK (
    #         (version = 1 AND parent_id IS NULL) OR (version > 1 AND parent_id IS NOT NULL)
    #     )
    # );
    #
    # -- ============================================================
    # -- Step 2: Create indexes on compounds
    # -- ============================================================
    #
    # -- Job lookup (1:1 link, enforces uniqueness, CASCADE performance)
    # CREATE UNIQUE INDEX idx_compounds_job_id ON compounds (job_id);
    #
    # -- InChIKey exact match (duplicate detection)
    # CREATE INDEX idx_compounds_inchikey ON compounds (inchikey);
    #
    # -- Structure key for version queries (protonation-insensitive sibling lookup)
    # CREATE INDEX idx_compounds_structure_key ON compounds (inchikey_structure_key);
    #
    # -- Parent lookup (for finding children of a compound)
    # CREATE INDEX idx_compounds_parent_id ON compounds (parent_id)
    #     WHERE parent_id IS NOT NULL;
    #
    # -- Compound name search
    # CREATE INDEX idx_compounds_name ON compounds (compound_name);
    #
    # -- Original compounds only (parent_id IS NULL = root versions)
    # CREATE INDEX idx_compounds_roots ON compounds (processed_at DESC)
    #     WHERE parent_id IS NULL;
    #
    # -- Unique version per parent (children only -- roots are unique by PK)
    # CREATE UNIQUE INDEX idx_compounds_parent_version ON compounds (parent_id, version)
    #     WHERE parent_id IS NOT NULL;
    #
    # -- ============================================================
    # -- Step 3: Create reparenting trigger function
    # -- ============================================================
    # CREATE OR REPLACE FUNCTION reparent_compound_children()
    # RETURNS TRIGGER AS $$
    # DECLARE
    #     v_promoted_id UUID;
    # BEGIN
    #     -- Only act if the deleted compound is a parent (root of version tree)
    #     IF OLD.parent_id IS NOT NULL THEN
    #         RETURN OLD;  -- Not a parent, nothing to reparent
    #     END IF;
    #
    #     -- Find the next oldest child to promote (lowest version, then earliest processed)
    #     SELECT entry_id INTO v_promoted_id
    #     FROM compounds
    #     WHERE parent_id = OLD.entry_id
    #     ORDER BY version ASC, processed_at ASC
    #     LIMIT 1;
    #
    #     -- No children, nothing to do
    #     IF v_promoted_id IS NULL THEN
    #         RETURN OLD;
    #     END IF;
    #
    #     -- Reparent remaining children to the promoted compound
    #     UPDATE compounds
    #     SET parent_id = v_promoted_id
    #     WHERE parent_id = OLD.entry_id
    #       AND entry_id != v_promoted_id;
    #
    #     -- Promote the chosen child to parent status
    #     UPDATE compounds
    #     SET parent_id = NULL, version = 1
    #     WHERE entry_id = v_promoted_id;
    #
    #     RETURN OLD;
    # END;
    # $$ LANGUAGE plpgsql;
    #
    # -- ============================================================
    # -- Step 4: Attach trigger to compounds table
    # -- ============================================================
    # CREATE TRIGGER trg_reparent_on_delete
    #     BEFORE DELETE ON compounds
    #     FOR EACH ROW
    #     EXECUTE FUNCTION reparent_compound_children();
    #
    # -- ============================================================
    # -- Step 5: Create deleted_compounds archive table
    # -- ============================================================
    # CREATE TABLE deleted_compounds (
    #     -- Archive identity (separate from original compound ID)
    #     id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    #
    #     -- Original compound data (mirrored from compounds table)
    #     entry_id                UUID NOT NULL,
    #     job_id                  UUID,
    #     compound_name           VARCHAR(255) NOT NULL,
    #     chembl_id               VARCHAR(50),
    #     smiles                  TEXT,
    #     canonical_smiles        TEXT,
    #     inchikey                VARCHAR(27),
    #     inchikey_structure_key  VARCHAR(25),
    #
    #     -- Versioning state at time of deletion
    #     parent_id               UUID,
    #     version                 INTEGER,
    #     config_diff             JSONB,
    #
    #     -- Analysis results
    #     imp_score               REAL,
    #     similar_compounds       INTEGER,
    #     total_activities        INTEGER,
    #     imp_candidates          INTEGER,
    #     qed                     REAL,
    #     num_outliers            INTEGER,
    #
    #     -- Analysis config
    #     similarity_threshold    INTEGER,
    #     activity_types          TEXT[],
    #
    #     -- Metadata
    #     author_name             VARCHAR(100),
    #     storage_path            TEXT,
    #
    #     -- Original timestamp
    #     original_processed_at   TIMESTAMPTZ,
    #
    #     -- Deletion metadata
    #     deleted_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    #     deleted_by              UUID,
    #     deletion_reason         VARCHAR(255)
    # );
    #
    # -- ============================================================
    # -- Step 6: Create indexes on deleted_compounds
    # -- ============================================================
    #
    # -- Lookup by original entry_id (recovery, audit queries)
    # CREATE INDEX idx_deleted_compounds_entry_id ON deleted_compounds (entry_id);
    #
    # -- Deletion timeline (audit queries)
    # CREATE INDEX idx_deleted_compounds_deleted_at ON deleted_compounds (deleted_at);
    #
    # -- InChIKey lookup in archive
    # CREATE INDEX idx_deleted_compounds_inchikey ON deleted_compounds (inchikey);
    #
    # -- ============================================================
    # -- Step 7: Table and column comments
    # -- ============================================================
    #
    # -- Compounds table comments
    # COMMENT ON TABLE compounds IS 'Processed compound metadata...';
    # COMMENT ON COLUMN compounds.entry_id IS 'Primary key UUID, auto-generated if not provided';
    # COMMENT ON COLUMN compounds.job_id IS '1:1 link to the job that created this compound. CASCADE.';
    # COMMENT ON COLUMN compounds.parent_id IS 'Self-FK for versioning. NULL = root/original.';
    # COMMENT ON COLUMN compounds.version IS 'Version number. 1 = original, 2+ = re-analysis.';
    # COMMENT ON COLUMN compounds.config_diff IS 'JSONB diff of analysis config vs parent';
    # COMMENT ON COLUMN compounds.inchikey IS 'Full 27-char InChIKey';
    # COMMENT ON COLUMN compounds.inchikey_structure_key IS 'First two blocks (25 chars) for protonation-insensitive lookup';
    # COMMENT ON COLUMN compounds.activity_types IS 'Postgres TEXT array of bioactivity types';
    # COMMENT ON COLUMN compounds.similarity_threshold IS 'ChEMBL similarity threshold (40-100)';
    #
    # -- Deleted compounds table comments
    # COMMENT ON TABLE deleted_compounds IS 'Archive table for deleted compounds.';
    # COMMENT ON COLUMN deleted_compounds.deleted_by IS 'UUID of the session that performed deletion';
    # COMMENT ON COLUMN deleted_compounds.deletion_reason IS 'Why deleted: user_request, replaced, job_cascade, admin_cleanup';

    # === AUDIT_EVENTS TABLE (Phase 12, from migrations/003_create_audit_events_table.sql) ===
    # -- Migration 003: Create audit_events table + gap-filling indexes
    # -- Phase 12: Schema -- Audit Trail + Indexes
    # -- Date: 2026-03-20
    # --
    # -- Creates:
    # -- 1. audit_event_type ENUM (13 values: 10 security + 3 lifecycle)
    # -- 2. audit_events table (6 columns)
    # -- 3. 3 indexes on audit_events
    # -- 4. Gap-filling indexes on existing tables (jobs, compounds)
    #
    # -- ============================================================
    # -- Step 1: Create audit_event_type ENUM
    # -- ============================================================
    # CREATE TYPE audit_event_type AS ENUM (
    #     -- Security events (from backend/core/audit.py AuditEvent)
    #     'rate_limit_exceeded',
    #     'job_cancelled',
    #     'job_deleted',
    #     'batch_cancelled',
    #     'validation_failed',
    #     'path_traversal_blocked',
    #     'invalid_smiles',
    #     'authentication_failed',
    #     'authorization_failed',
    #     'suspicious_input',
    #     -- Lifecycle events (new)
    #     'compound_created',
    #     'compound_deleted',
    #     'job_created'
    # );
    #
    # -- ============================================================
    # -- Step 2: Create audit_events table
    # -- ============================================================
    # CREATE TABLE audit_events (
    #     id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    #     event_type  audit_event_type NOT NULL,
    #     session_id  UUID,
    #     severity    VARCHAR(20) NOT NULL DEFAULT 'warning',
    #     details     JSONB,
    #     created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    # );
    #
    # -- ============================================================
    # -- Step 3: Create indexes on audit_events
    # -- ============================================================
    #
    # -- Composite: filter by event_type + time range (dashboard queries)
    # CREATE INDEX idx_audit_events_type_created ON audit_events (event_type, created_at);
    #
    # -- Session lookup (who did what) -- partial, only non-null sessions
    # CREATE INDEX idx_audit_events_session_id ON audit_events (session_id)
    #     WHERE session_id IS NOT NULL;
    #
    # -- Time-based cleanup and recent events queries
    # CREATE INDEX idx_audit_events_created_at ON audit_events (created_at);
    #
    # -- ============================================================
    # -- Step 4: Gap-filling indexes on existing tables
    # -- ============================================================
    #
    # -- GIN index on compounds.activity_types for TEXT[] containment (@>, &&)
    # CREATE INDEX IF NOT EXISTS idx_compounds_activity_types
    #     ON compounds USING gin (activity_types);
    #
    # -- GIN index on jobs.activity_types for TEXT[] containment (@>, &&)
    # CREATE INDEX IF NOT EXISTS idx_jobs_activity_types
    #     ON jobs USING gin (activity_types);
    #
    # -- Expression index for case-insensitive compound name search
    # -- Must match CompoundRepository queries: lower(trim(compound_name))
    # CREATE INDEX IF NOT EXISTS idx_compounds_name_lower
    #     ON compounds (lower(trim(compound_name)));
    #
    # -- ChEMBL ID lookup (used in get_compounds_paginated search)
    # CREATE INDEX IF NOT EXISTS idx_compounds_chembl_id
    #     ON compounds (chembl_id)
    #     WHERE chembl_id IS NOT NULL;
    #
    # -- ============================================================
    # -- Step 5: Table and column comments
    # -- ============================================================
    # COMMENT ON TABLE audit_events IS 'Structured audit trail for security and lifecycle events.';
    # COMMENT ON COLUMN audit_events.id IS 'Auto-incrementing BIGINT identity';
    # COMMENT ON COLUMN audit_events.event_type IS 'Postgres ENUM: 10 security + 3 lifecycle events';
    # COMMENT ON COLUMN audit_events.session_id IS 'Session UUID of the actor (nullable)';
    # COMMENT ON COLUMN audit_events.severity IS 'Log severity: info, warning, error, critical';
    # COMMENT ON COLUMN audit_events.details IS 'Flexible JSONB payload';
    # COMMENT ON COLUMN audit_events.created_at IS 'When the event occurred (TIMESTAMPTZ)';

    pass


def downgrade() -> None:
    raise RuntimeError("Cannot downgrade past baseline")
