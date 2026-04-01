#!/usr/bin/env python3
"""
Schema verification script for IMPULATOR Supabase Postgres database.

Verifies all 4 tables (jobs, compounds, deleted_compounds, audit_events),
3 ENUM types, all indexes, constraints, and triggers against expected schema.

Usage:
    python scripts/verify_schema.py

Requires DATABASE_URL environment variable (loaded from .env if present).

Exit codes:
    0 = all checks pass
    1 = one or more checks failed
"""
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def verify_schema(database_url: str) -> bool:
    """Verify full Supabase schema. Returns True if all checks pass."""
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]

    engine = create_engine(database_url)
    results: list[tuple[str, bool, str]] = []

    with engine.connect() as conn:
        # ============================================================
        # 1. Check all 4 tables exist
        # ============================================================
        tables = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('jobs', 'compounds', 'deleted_compounds', 'audit_events')
            ORDER BY table_name
        """)).fetchall()
        table_names = {r[0] for r in tables}
        results.append(("Tables exist (4)", len(tables) == 4,
                        f"Found {len(tables)}/4: {sorted(table_names)}"))

        # ============================================================
        # 2. Check ENUM types exist with correct values
        # ============================================================
        enums = conn.execute(text("""
            SELECT typname FROM pg_type
            WHERE typname IN ('job_status', 'job_type', 'audit_event_type')
            ORDER BY typname
        """)).fetchall()
        enum_names = {r[0] for r in enums}
        results.append(("ENUM types (3)", len(enums) == 3,
                        f"Found {len(enums)}/3: {sorted(enum_names)}"))

        # Check job_status has 5 values
        job_status_vals = conn.execute(text("""
            SELECT enumlabel FROM pg_enum
            JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
            WHERE pg_type.typname = 'job_status'
            ORDER BY enumsortorder
        """)).fetchall()
        job_status_labels = [r[0] for r in job_status_vals]
        expected_job_status = ['pending', 'processing', 'completed', 'failed', 'cancelled']
        results.append(("job_status values (5)", job_status_labels == expected_job_status,
                        f"Got: {job_status_labels}"))

        # Check job_type has 2 values
        job_type_vals = conn.execute(text("""
            SELECT enumlabel FROM pg_enum
            JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
            WHERE pg_type.typname = 'job_type'
            ORDER BY enumsortorder
        """)).fetchall()
        job_type_labels = [r[0] for r in job_type_vals]
        expected_job_type = ['single', 'batch']
        results.append(("job_type values (2)", job_type_labels == expected_job_type,
                        f"Got: {job_type_labels}"))

        # Check audit_event_type has 13 values
        audit_vals = conn.execute(text("""
            SELECT enumlabel FROM pg_enum
            JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
            WHERE pg_type.typname = 'audit_event_type'
            ORDER BY enumsortorder
        """)).fetchall()
        audit_labels = [r[0] for r in audit_vals]
        expected_audit = [
            'rate_limit_exceeded', 'job_cancelled', 'job_deleted',
            'batch_cancelled', 'validation_failed', 'path_traversal_blocked',
            'invalid_smiles', 'authentication_failed', 'authorization_failed',
            'suspicious_input', 'compound_created', 'compound_deleted', 'job_created'
        ]
        results.append(("audit_event_type values (13)", audit_labels == expected_audit,
                        f"Got {len(audit_labels)}/13"))

        # ============================================================
        # 3. Check column counts per table
        # ============================================================
        expected_columns = {
            "jobs": 20,
            "compounds": 22,
            "deleted_compounds": 26,
            "audit_events": 6,
        }
        for table, expected in expected_columns.items():
            cols = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = :table
            """), {"table": table}).scalar()
            results.append((f"{table} columns ({expected})", cols == expected,
                            f"Found {cols}/{expected}"))

        # ============================================================
        # 4. Check all required indexes exist
        # ============================================================
        indexes = conn.execute(text("""
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY indexname
        """)).fetchall()
        index_names = {r[0] for r in indexes}

        required_indexes = [
            # Jobs (5 from Phase 10)
            "idx_jobs_session_id",
            "idx_jobs_batch_id",
            "idx_jobs_pending",
            "idx_jobs_processing",
            "idx_jobs_status_completed_at",
            # Jobs (1 gap-fill from Phase 12)
            "idx_jobs_activity_types",
            # Compounds (7 from Phase 11)
            "idx_compounds_job_id",
            "idx_compounds_inchikey",
            "idx_compounds_structure_key",
            "idx_compounds_parent_id",
            "idx_compounds_name",
            "idx_compounds_roots",
            "idx_compounds_parent_version",
            # Compounds (3 gap-fill from Phase 12)
            "idx_compounds_activity_types",
            "idx_compounds_name_lower",
            "idx_compounds_chembl_id",
            # Deleted compounds (3 from Phase 11)
            "idx_deleted_compounds_entry_id",
            "idx_deleted_compounds_deleted_at",
            "idx_deleted_compounds_inchikey",
            # Audit events (3 from Phase 12)
            "idx_audit_events_type_created",
            "idx_audit_events_session_id",
            "idx_audit_events_created_at",
        ]
        for idx in required_indexes:
            results.append((f"Index {idx}", idx in index_names,
                            "present" if idx in index_names else "MISSING"))

        # ============================================================
        # 5. Check key constraints
        # ============================================================
        constraints = conn.execute(text("""
            SELECT conname, contype FROM pg_constraint
            WHERE conrelid IN (
                SELECT oid FROM pg_class
                WHERE relname IN ('jobs', 'compounds', 'deleted_compounds', 'audit_events')
            )
            ORDER BY conname
        """)).fetchall()
        constraint_names = {r[0] for r in constraints}

        required_constraints = [
            # Jobs
            "chk_threshold_range",
            "uix_job_session_idempotency",
            # Compounds
            "chk_no_self_parent",
            "chk_compound_threshold_range",
            "chk_version_positive",
            "chk_root_has_no_parent",
        ]
        for con in required_constraints:
            results.append((f"Constraint {con}", con in constraint_names,
                            "present" if con in constraint_names else "MISSING"))

        # ============================================================
        # 6. Check FK constraints
        # ============================================================
        fk_constraints = conn.execute(text("""
            SELECT conname, pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE contype = 'f'
            AND conrelid IN (
                SELECT oid FROM pg_class WHERE relname = 'compounds'
            )
            ORDER BY conname
        """)).fetchall()
        fk_names = {r[0] for r in fk_constraints}
        results.append(("FK compounds.job_id -> jobs.id",
                        any("job_id" in (r[1] or "") for r in fk_constraints),
                        f"FKs: {sorted(fk_names)}"))
        results.append(("FK compounds.parent_id -> compounds.entry_id",
                        any("parent_id" in (r[1] or "") for r in fk_constraints),
                        f"FKs: {sorted(fk_names)}"))

        # ============================================================
        # 7. Check reparenting trigger
        # ============================================================
        triggers = conn.execute(text("""
            SELECT tgname, proname
            FROM pg_trigger t
            JOIN pg_proc p ON t.tgfoid = p.oid
            WHERE t.tgrelid = 'compounds'::regclass
              AND NOT t.tgisinternal
        """)).fetchall()
        trigger_names = {r[0] for r in triggers}
        results.append(("Trigger trg_reparent_on_delete",
                        "trg_reparent_on_delete" in trigger_names,
                        f"Triggers: {sorted(trigger_names)}"))

        # ============================================================
        # 8. Check alembic_version table
        # ============================================================
        alembic_table = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'alembic_version'
        """)).fetchone()
        results.append(("alembic_version table exists",
                        alembic_table is not None,
                        "present" if alembic_table else "MISSING"))

        if alembic_table:
            version = conn.execute(text(
                "SELECT version_num FROM alembic_version"
            )).fetchone()
            results.append(("alembic_version has revision",
                            version is not None,
                            f"revision: {version[0]}" if version else "EMPTY"))

    # ============================================================
    # Print report
    # ============================================================
    all_pass = True
    passed_count = 0
    failed_count = 0

    print("=" * 60)
    print("IMPULATOR Schema Verification Report")
    print("=" * 60)

    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if passed:
            passed_count += 1
        else:
            failed_count += 1
            all_pass = False
        print(f"  [{status}] {name}: {detail}")

    print("=" * 60)
    print(f"Results: {passed_count} passed, {failed_count} failed, {passed_count + failed_count} total")
    if all_pass:
        print("STATUS: ALL CHECKS PASSED")
    else:
        print("STATUS: SOME CHECKS FAILED")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    success = verify_schema(database_url)
    sys.exit(0 if success else 1)
