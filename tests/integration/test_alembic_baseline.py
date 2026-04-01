"""Verify Alembic baseline migration creates complete schema on fresh Postgres.

Success criteria #6: alembic upgrade head on empty Postgres creates all
tables/indexes/triggers that the application expects.
"""
from sqlalchemy import create_engine, inspect, text


class TestAlembicBaseline:
    """Verify alembic upgrade head produces full schema."""

    def test_alembic_upgrade_head_creates_all_tables(self, postgres_url):
        """All 4 application tables exist after alembic upgrade head."""
        # The pg_engine fixture already ran alembic upgrade head.
        engine = create_engine(postgres_url)
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        expected = {"jobs", "compounds", "deleted_compounds", "audit_events"}
        missing = expected - tables
        assert not missing, f"Missing tables after alembic upgrade: {missing}"
        engine.dispose()

    def test_alembic_creates_enum_types(self, pg_engine):
        """Postgres ENUM types exist: job_status, job_type, audit_event_type."""
        with pg_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT typname FROM pg_type WHERE typname IN "
                "('job_status', 'job_type', 'audit_event_type') ORDER BY typname"
            ))
            enums = [row[0] for row in result]
        assert "audit_event_type" in enums
        assert "job_status" in enums
        assert "job_type" in enums

    def test_alembic_creates_job_indexes(self, pg_engine):
        """Key job indexes exist after migration."""
        inspector = inspect(pg_engine)
        idx_names = {idx["name"] for idx in inspector.get_indexes("jobs")}
        expected = {
            "idx_jobs_session_id",
            "idx_jobs_batch_id",
            "idx_jobs_pending",
            "idx_jobs_processing",
        }
        missing = expected - idx_names
        assert not missing, f"Missing job indexes: {missing}"

    def test_alembic_creates_compound_indexes(self, pg_engine):
        """Key compound indexes exist after migration."""
        inspector = inspect(pg_engine)
        idx_names = {idx["name"] for idx in inspector.get_indexes("compounds")}
        expected = {
            "idx_compounds_job_id",
            "idx_compounds_inchikey",
            "idx_compounds_structure_key",
            "idx_compounds_parent_id",
            "idx_compounds_name",
        }
        missing = expected - idx_names
        assert not missing, f"Missing compound indexes: {missing}"

    def test_alembic_creates_reparent_trigger(self, pg_engine):
        """Reparent trigger exists on compounds table."""
        with pg_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT tgname FROM pg_trigger "
                "WHERE tgrelid = 'compounds'::regclass "
                "AND tgname = 'trg_reparent_on_delete'"
            ))
            triggers = [row[0] for row in result]
        assert "trg_reparent_on_delete" in triggers

    def test_alembic_creates_updated_at_trigger(self, pg_engine):
        """updated_at trigger exists on jobs table (from migration 0002)."""
        with pg_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT tgname FROM pg_trigger "
                "WHERE tgrelid = 'jobs'::regclass "
                "AND tgname = 'update_jobs_updated_at'"
            ))
            triggers = [row[0] for row in result]
        assert "update_jobs_updated_at" in triggers

    def test_alembic_ini_has_no_hardcoded_url(self):
        """alembic.ini must not hardcode sqlalchemy.url (SC-8)."""
        with open("backend/alembic.ini") as f:
            content = f.read()
        # sqlalchemy.url should be empty or placeholder -- env.py sets it from settings
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("sqlalchemy.url") and "=" in stripped:
                value = stripped.split("=", 1)[1].strip()
                assert not value.startswith("postgresql://"), (
                    f"alembic.ini has hardcoded URL: {value}"
                )
