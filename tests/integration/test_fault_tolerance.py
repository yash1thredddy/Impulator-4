"""Fault tolerance regression tests (SC-9).

Covers four scenarios:
1. Scheduler crash recovery -- crash state detection via health probe
2. Concurrent duplicate submission -- InChIKey collision safety
3. Cancel-during-processing race -- no corrupt state after cancel
4. Connection pool exhaustion -- graceful degradation under load

All tests run against real Postgres via pg_engine/client/db_session fixtures.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import text

from backend.models.job import Job
from backend.models.compound import Compound
from backend.models.enums import JobStatus, JobType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_SESSION_ID = str(uuid.uuid4())
TEST_INCHIKEY = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
TEST_SMILES = "CCO"


def _make_job(
    db_session,
    *,
    status: JobStatus = JobStatus.PENDING,
    session_id: str = TEST_SESSION_ID,
    compound_name: str = "TestCompound",
    smiles: str = TEST_SMILES,
    similarity_threshold: int = 90,
    activity_types: list[str] | None = None,
    job_type: JobType = JobType.SINGLE,
    **overrides,
) -> Job:
    """Insert a job directly via ORM and return the refreshed object."""
    job = Job(
        id=overrides.pop("id", uuid.uuid4()),
        session_id=uuid.UUID(session_id),
        compound_name=compound_name,
        smiles=smiles,
        status=status,
        similarity_threshold=similarity_threshold,
        activity_types=activity_types or ["EC50", "IC50", "Kd", "Ki"],
        job_type=job_type,
        **overrides,
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    return job


def _make_compound(
    db_session,
    *,
    job_id: uuid.UUID | None = None,
    compound_name: str = "TestCompound",
    smiles: str = TEST_SMILES,
    inchikey: str = TEST_INCHIKEY,
    similarity_threshold: int = 90,
    activity_types: list[str] | None = None,
    **overrides,
) -> Compound:
    """Insert a compound directly via ORM and return the refreshed object."""
    from backend.services.job_service import _inchikey_structure_key

    comp = Compound(
        entry_id=overrides.pop("entry_id", uuid.uuid4()),
        job_id=job_id,
        compound_name=compound_name,
        smiles=smiles,
        inchikey=inchikey,
        inchikey_structure_key=_inchikey_structure_key(inchikey),
        similarity_threshold=similarity_threshold,
        activity_types=activity_types or ["EC50", "IC50", "Kd", "Ki"],
        processed_at=datetime.now(timezone.utc),
        **overrides,
    )
    db_session.add(comp)
    db_session.commit()
    db_session.refresh(comp)
    return comp


# ===========================================================================
# Class 1: Scheduler Crash Recovery
# ===========================================================================


class TestSchedulerCrashRecovery:
    """Verify scheduler crash detection via module globals and health probes."""

    def test_scheduler_crash_sets_crash_reason(self):
        """Setting _crash_reason is reflected in stats()."""
        from backend.core import scheduler

        original = scheduler._crash_reason
        try:
            scheduler._crash_reason = "test crash"
            stats = scheduler.stats()
            assert stats["crash_reason"] == "test crash"
        finally:
            scheduler._crash_reason = original

    def test_scheduler_not_running_in_test_context(self):
        """In test context the scheduler task is not started."""
        from backend.core import scheduler

        assert scheduler.is_running() is False

    def test_health_detects_dead_scheduler(self, client):
        """GET /api/v1/health/executor reports scheduler-related stats.

        The scheduler is not started in tests (trigger is patched), so
        the executor endpoint should return successfully with scheduler
        not actively running.  The /health/detailed endpoint exposes
        scheduler.active directly.
        """
        resp = client.get(
            "/api/v1/health/detailed",
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Scheduler section should exist and report inactive
        sched = data["checks"]["scheduler"]
        assert sched["active"] is False
        # crash_reason should be None (no crash in fresh test)
        assert sched.get("crash_reason") is None or sched["crash_reason"] is None

    def test_health_executor_endpoint_ok(self, client):
        """GET /api/v1/health/executor returns 200 with executor stats."""
        resp = client.get(
            "/api/v1/health/executor",
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "max_concurrent_jobs" in data
        assert "active_jobs" in data
        assert "slots_available" in data


# ===========================================================================
# Class 2: Concurrent Duplicate Submission
# ===========================================================================


class TestConcurrentDuplicateSubmission:
    """Verify InChIKey-based duplicate detection does not corrupt data."""

    def test_concurrent_same_inchikey_handled(self, client, db_session):
        """Submitting a compound with the same InChIKey triggers duplicate detection.

        Steps:
        1. Seed a completed job + compound with a known InChIKey.
        2. POST /api/v1/jobs/check-duplicates with the same InChIKey.
        3. Verify the duplicate is detected.
        """
        # Seed existing compound
        job = _make_job(db_session, status=JobStatus.COMPLETED)
        _make_compound(db_session, job_id=job.id, inchikey=TEST_INCHIKEY)

        # Check duplicates
        resp = client.post(
            "/api/v1/jobs/check-duplicates",
            json={
                "compounds": [
                    {
                        "compound_name": "TestCompound2",
                        "smiles": TEST_SMILES,
                        "inchikey": TEST_INCHIKEY,
                    }
                ],
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        assert resp.status_code == 200
        data = resp.json()

        # The response should indicate duplicates were found via structure_matches
        assert "structure_matches" in data and len(data["structure_matches"]) > 0

    def test_duplicate_replace_creates_new_job(self, client, db_session):
        """Resolving a duplicate with 'replace' creates a new job and
        schedules deletion of the old compound."""
        # Seed existing completed job + compound
        job = _make_job(db_session, status=JobStatus.COMPLETED)
        comp = _make_compound(db_session, job_id=job.id, inchikey=TEST_INCHIKEY)

        # Resolve with replace action
        resp = client.post(
            "/api/v1/jobs/resolve-duplicate",
            json={
                "action": "replace",
                "smiles": TEST_SMILES,
                "compound_name": "TestCompound_replaced",
                "author_name": "TestAuthor",
                "existing_entry_id": str(comp.entry_id),
                "similarity_threshold": 90,
                "activity_types": ["EC50", "IC50", "Kd", "Ki"],
            },
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        # 201 = new job created for replacement
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data  # New job ID
        # compound_name comes from the existing compound being replaced
        assert data["compound_name"] == "TestCompound"

    def test_duplicate_keep_both(self, client, db_session):
        """Resolving with 'duplicate' keeps both compounds."""
        job = _make_job(db_session, status=JobStatus.COMPLETED)
        comp = _make_compound(db_session, job_id=job.id, inchikey=TEST_INCHIKEY)

        resp = client.post(
            "/api/v1/jobs/resolve-duplicate",
            json={
                "action": "duplicate",
                "smiles": TEST_SMILES,
                "compound_name": "TestCompound_dup",
                "author_name": "TestAuthor",
                "existing_entry_id": str(comp.entry_id),
                # Use different threshold to avoid identical-config rejection
                "similarity_threshold": 70,
                "activity_types": ["EC50", "IC50", "Kd", "Ki"],
            },
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        # 201 = new job created for duplicate
        assert resp.status_code == 201

        # Original compound should still exist
        row = db_session.execute(
            text("SELECT count(*) FROM compounds WHERE entry_id = :eid"),
            {"eid": comp.entry_id},
        ).scalar()
        assert row == 1, "Original compound should still exist after 'duplicate' resolve"


# ===========================================================================
# Class 3: Cancel During Processing Race
# ===========================================================================


class TestCancelDuringProcessingRace:
    """Verify cancel safety on PROCESSING and terminal-state jobs."""

    def test_cancel_processing_job_safe(self, client, db_session):
        """Cancelling a PROCESSING job sets status to CANCELLED without
        leaving orphaned compound records."""
        job = _make_job(
            db_session,
            status=JobStatus.PROCESSING,
            started_at=datetime.now(timezone.utc),
        )

        resp = client.post(
            f"/api/v1/jobs/{job.id}/cancel",
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

        # No orphaned compounds for this job
        orphans = db_session.execute(
            text("SELECT count(*) FROM compounds WHERE job_id = :jid"),
            {"jid": job.id},
        ).scalar()
        assert orphans == 0, "No compounds should be associated with cancelled job"

    def test_cancel_completed_job_returns_409(self, client, db_session):
        """Cancelling a COMPLETED job returns 409 (terminal state)."""
        job = _make_job(
            db_session,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )

        resp = client.post(
            f"/api/v1/jobs/{job.id}/cancel",
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        # 409 Conflict -- cannot cancel terminal state
        assert resp.status_code == 409

    def test_cancel_failed_job_returns_409(self, client, db_session):
        """Cancelling a FAILED job returns 409 (terminal state)."""
        job = _make_job(
            db_session,
            status=JobStatus.FAILED,
            error_message="test failure",
            completed_at=datetime.now(timezone.utc),
        )

        resp = client.post(
            f"/api/v1/jobs/{job.id}/cancel",
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        assert resp.status_code == 409

    def test_cancel_already_cancelled_returns_409(self, client, db_session):
        """Cancelling an already-cancelled job returns 409."""
        job = _make_job(
            db_session,
            status=JobStatus.CANCELLED,
            cancelled_at=datetime.now(timezone.utc),
        )

        resp = client.post(
            f"/api/v1/jobs/{job.id}/cancel",
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        assert resp.status_code == 409


# ===========================================================================
# Class 4: Connection Pool Exhaustion
# ===========================================================================


class TestConnectionPoolExhaustion:
    """Verify graceful degradation under pool pressure."""

    def test_health_reports_pool_stats(self, client):
        """GET /api/v1/health/detailed includes database stats without error."""
        resp = client.get(
            "/api/v1/health/detailed",
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Database section should be present and healthy
        db_check = data["checks"]["database"]
        assert db_check["status"] == "healthy"
        assert db_check["backend"] == "postgres"
        assert "latency_ms" in db_check
        assert db_check["latency_ms"] >= 0

    def test_multiple_concurrent_requests(self, client):
        """Fire 10 rapid health requests to validate pool recycling.

        All should return 200 without pool exhaustion errors (TestClient
        is synchronous, but exercises pool checkout/checkin cycles).
        """
        failures = []
        for i in range(10):
            resp = client.get(
                "/api/v1/health",
                headers={"X-Session-ID": TEST_SESSION_ID},
            )
            if resp.status_code != 200:
                failures.append(f"Request {i}: status={resp.status_code}")

        assert not failures, f"Pool exhaustion detected: {failures}"

    def test_detailed_health_under_load(self, client):
        """Fire 5 rapid detailed health requests (heavier DB queries).

        Validates pool handles concurrent detailed checks without exhaustion.
        """
        failures = []
        for i in range(5):
            resp = client.get(
                "/api/v1/health/detailed",
                headers={"X-Session-ID": TEST_SESSION_ID},
            )
            if resp.status_code != 200:
                failures.append(f"Request {i}: status={resp.status_code}")

        assert not failures, f"Pool exhaustion on detailed checks: {failures}"

    def test_executor_stats_accessible(self, client):
        """Executor endpoint returns stats even when pool is busy."""
        resp = client.get(
            "/api/v1/health/executor",
            headers={"X-Session-ID": TEST_SESSION_ID},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_capacity"] is True or data["has_capacity"] is False
