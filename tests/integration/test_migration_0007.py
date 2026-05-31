"""
Integration tests for Alembic migration 0007 (Phase 23).

The ``pg_engine`` fixture (root conftest) has already run ``alembic upgrade head``
against a disposable Postgres (testcontainer locally, the CI Postgres service via
``TEST_DATABASE_URL``), so these assert the *effect* of migration 0007 on the live
schema rather than re-running the migration.

Migration 0007 (see ``backend/alembic/versions/0007_add_collections.py``):
  op.execute("COMMIT") -> ALTER TYPE job_type ADD VALUE 'collection'
  -> create_table("collections", ... members_config JSONB ...).
"""
from sqlalchemy import inspect, text


def test_enum_value_added(pg_engine):
    """The job_type Postgres enum gains the 'collection' value after 0007."""
    with pg_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT e.enumlabel FROM pg_enum e "
            "JOIN pg_type t ON e.enumtypid = t.oid "
            "WHERE t.typname = 'job_type'"
        ))
        labels = {row[0] for row in result}
    assert "collection" in labels, (
        f"job_type enum is missing the 'collection' value after 0007: {labels}"
    )


def test_collections_table_has_members_config(pg_engine):
    """The collections table exists with a members_config JSONB column (D-02)."""
    inspector = inspect(pg_engine)

    assert "collections" in inspector.get_table_names(), (
        "migration 0007 did not create the 'collections' table"
    )

    columns = {col["name"]: col for col in inspector.get_columns("collections")}
    assert "members_config" in columns, (
        f"'collections' is missing the members_config column: {sorted(columns)}"
    )
    # D-02 mandates JSONB specifically (not plain JSON) for the member set.
    type_str = str(columns["members_config"]["type"]).upper()
    assert "JSONB" in type_str, (
        f"members_config should be JSONB, got: {type_str}"
    )
