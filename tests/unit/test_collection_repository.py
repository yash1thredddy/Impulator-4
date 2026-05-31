"""
Unit tests for the SYNC collection repository (Phase 23, plan 23-03).

CRITICAL OVERRIDE (23-CONTEXT.md): the DB layer is SYNC SQLAlchemy
(`def ...(db: Session)`), NOT async. The repo mirrors compound_repository.py /
job_repository.py exactly -- do NOT generate AsyncSession from CLAUDE.md drift.

DB-backed tests use the session-scoped ``pg_engine`` fixture from the root
``tests/conftest.py`` (testcontainer-backed, Alembic-provisioned). Each test
runs inside its own transaction-wrapped session that is rolled back on teardown,
so they are self-isolating without an integration-only ``db_session`` fixture.
``test_collection_repo_is_sync`` is pure introspection and needs no database.
"""
import inspect
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.models.enums import JobStatus, JobType
from backend.models.job import Job
from backend.repositories import collection_repo
from backend.repositories.collection_repository import CollectionRepository
from backend.repositories.job_repository import job_repo


@pytest.fixture
def db(pg_engine):
    """A rolled-back Session for one test (isolation without TRUNCATE)."""
    connection = pg_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


def _make_collection_job(db: Session) -> Job:
    """Insert a COLLECTION job so a collection can hang off it (1:1 FK)."""
    return job_repo.create_job(
        db,
        id=uuid.uuid4(),
        job_type=JobType.COLLECTION,
        compound_name="Test Collection",
        session_id=uuid.uuid4(),
    )


def test_create_and_get_collection(db):
    """create persists a row; get_by_id round-trips it (D-02)."""
    job = _make_collection_job(db)
    members = {"members": [{"name": "m1", "smiles": "C"}, {"name": "m2", "smiles": "CC"}]}
    created = collection_repo.create(
        db,
        name="Flavonoids",
        author_name="Jane",
        job_id=job.id,
        members_config=members,
        description="desc",
    )
    assert created.id is not None

    fetched = collection_repo.get_by_id(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "Flavonoids"
    assert fetched.author_name == "Jane"
    # D-02: member input-definitions round-trip through members_config JSONB
    assert fetched.members_config == members
    assert collection_repo.get_by_id(db, uuid.uuid4()) is None


def test_get_by_job_id(db):
    """get_by_job_id loads the collection row keyed on its 1:1 job_id (D-02)."""
    job = _make_collection_job(db)
    created = collection_repo.create(
        db,
        name="ByJob",
        author_name="Jane",
        job_id=job.id,
        members_config={"members": [{"name": "a", "smiles": "C"}]},
    )
    loaded = collection_repo.get_by_job_id(db, job.id)
    assert loaded is not None
    assert loaded.id == created.id
    assert loaded.job_id == job.id
    # Unknown job_id yields nothing
    assert collection_repo.get_by_job_id(db, uuid.uuid4()) is None


def test_collection_repo_is_sync():
    """Repo methods are sync `def ...(db: Session)` -- no AsyncSession (override)."""
    methods = [
        "create",
        "get_by_id",
        "get_by_job_id",
        "list_all",
        "update_stats",
        "update_storage_path",
        "soft_delete",
    ]
    for name in methods:
        fn = getattr(CollectionRepository, name)
        # Not a coroutine function (sync, not async)
        assert not inspect.iscoroutinefunction(fn), f"{name} must be sync"
        params = list(inspect.signature(fn).parameters)
        # First param is self, second is db
        assert params[0] == "self", f"{name} first param must be self"
        assert params[1] == "db", f"{name} second param must be db (Session)"

    # No AsyncSession anywhere in the module source
    src = inspect.getsource(CollectionRepository)
    assert "AsyncSession" not in src


def test_list_all_excludes_soft_deleted(db):
    """list_all is GLOBAL (D-05) and excludes soft-deleted (deleted_at) rows."""
    job_a = _make_collection_job(db)
    job_b = _make_collection_job(db)
    keep = collection_repo.create(
        db, name="Keep", author_name="Jane", job_id=job_a.id,
        members_config={"members": []},
    )
    gone = collection_repo.create(
        db, name="Gone", author_name="Jane", job_id=job_b.id,
        members_config={"members": []},
    )

    # Soft-delete one
    assert collection_repo.soft_delete(db, gone.id) is True
    # Double soft-delete is a no-op (already deleted)
    assert collection_repo.soft_delete(db, gone.id) is False

    listed = collection_repo.list_all(db)
    listed_ids = {c.id for c in listed}
    assert keep.id in listed_ids
    assert gone.id not in listed_ids


def test_list_all_orders_newest_first(db):
    """list_all returns collections ordered by created_at desc (D-05)."""
    job1 = _make_collection_job(db)
    job2 = _make_collection_job(db)
    first = collection_repo.create(
        db, name="First", author_name="J", job_id=job1.id,
        members_config={"members": []},
    )
    db.flush()
    second = collection_repo.create(
        db, name="Second", author_name="J", job_id=job2.id,
        members_config={"members": []},
    )
    # Force a strictly-later created_at on the second row
    db.execute(
        text("UPDATE collections SET created_at = now() + interval '1 second' WHERE id = :i"),
        {"i": second.id},
    )
    listed = collection_repo.list_all(db)
    ids_in_order = [c.id for c in listed if c.id in {first.id, second.id}]
    assert ids_in_order[0] == second.id
    assert ids_in_order[1] == first.id


def test_update_stats_and_storage_path(db):
    """update_stats / update_storage_path mutate the row; None on missing."""
    job = _make_collection_job(db)
    c = collection_repo.create(
        db, name="Stats", author_name="J", job_id=job.id,
        members_config={"members": []},
    )
    updated = collection_repo.update_stats(
        db, c.id, compound_count=5, member_failed_count=1, avg_imp_score=42.5
    )
    assert updated is not None
    assert updated.compound_count == 5
    assert updated.member_failed_count == 1  # D-09
    assert updated.avg_imp_score == 42.5

    sp = collection_repo.update_storage_path(db, c.id, "collections/ab/x.zip")
    assert sp is not None
    assert sp.storage_path == "collections/ab/x.zip"

    # Missing id -> None for both
    assert collection_repo.update_stats(db, uuid.uuid4(), compound_count=1) is None
    assert collection_repo.update_storage_path(db, uuid.uuid4(), "x") is None


def test_job_status_enum_available():
    """Sanity: JobStatus import resolves (guards against import drift)."""
    assert JobStatus.PENDING is not None
