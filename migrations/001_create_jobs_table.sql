-- Migration 001: Create jobs table with Postgres-native types
-- Phase 10: Schema -- Jobs Table
-- Date: 2026-03-19
--
-- Replaces SQLite Job model. Key changes:
-- 1. input_params JSON blob -> 4 normalized columns (compound_name, smiles, threshold, activity_types)
-- 2. String status -> Postgres ENUM (job_status, job_type)
-- 3. DateTime -> TIMESTAMPTZ with server defaults
-- 4. String(36) UUIDs -> native UUID type
-- 5. result_summary TEXT -> JSONB (queryable)
-- 6. SYNC_PENDING status excluded (Azure DB sync being removed)

-- Step 1: Create ENUM types
CREATE TYPE job_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed',
    'cancelled'
);

CREATE TYPE job_type AS ENUM (
    'single',
    'batch'
);

-- Step 2: Create jobs table
CREATE TABLE jobs (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type        job_type NOT NULL DEFAULT 'single',
    status          job_status NOT NULL DEFAULT 'pending',

    -- Session / grouping
    session_id      UUID,
    batch_id        UUID,
    idempotency_key VARCHAR(64),

    -- Input (normalized from input_params JSON)
    compound_name   VARCHAR(255) NOT NULL,
    smiles          TEXT,
    threshold       INTEGER NOT NULL DEFAULT 90,
    activity_types  TEXT[],

    -- Progress tracking
    progress        REAL NOT NULL DEFAULT 0.0,
    current_step    VARCHAR(255),

    -- Results
    result_path     TEXT,
    result_summary  JSONB,

    -- Error handling
    error_message   TEXT,
    request_id      UUID,
    error_code      VARCHAR(50),

    -- Timestamps (all TIMESTAMPTZ, created_at with server default)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT uix_job_session_idempotency UNIQUE (session_id, idempotency_key),
    CONSTRAINT chk_threshold_range CHECK (threshold BETWEEN 40 AND 100)
);

-- Step 3: Create indexes
-- Session filtering (most queries filter by session_id)
CREATE INDEX idx_jobs_session_id ON jobs (session_id);

-- Batch lookups (batch summary, batch cancellation)
CREATE INDEX idx_jobs_batch_id ON jobs (batch_id);

-- Partial index for scheduler hot path: only pending jobs
CREATE INDEX idx_jobs_pending ON jobs (created_at)
    WHERE status = 'pending';

-- Partial index for timeout watchdog: only processing jobs
CREATE INDEX idx_jobs_processing ON jobs (started_at)
    WHERE status = 'processing';

-- Composite index for recently completed/failed queries
CREATE INDEX idx_jobs_status_completed_at ON jobs (status, completed_at);

-- Step 4: Add table and column comments
COMMENT ON TABLE jobs IS 'Job queue for compound analysis processing. Each row tracks one compound analysis from submission through completion.';
COMMENT ON COLUMN jobs.id IS 'Primary key UUID, auto-generated if not provided by application';
COMMENT ON COLUMN jobs.threshold IS 'ChEMBL similarity search threshold percentage (40-100, default 90)';
COMMENT ON COLUMN jobs.activity_types IS 'Array of bioactivity types to filter (e.g., IC50, Ki, EC50)';
COMMENT ON COLUMN jobs.result_summary IS 'JSONB blob with processing results (normalized to compounds table in Phase 11)';
COMMENT ON COLUMN jobs.request_id IS 'Correlation ID from originating HTTP request for log traceability';
COMMENT ON COLUMN jobs.error_code IS 'Machine-readable failure classification (ErrorCode enum value)';
