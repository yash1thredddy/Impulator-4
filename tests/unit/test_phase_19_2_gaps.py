"""
Phase 19.2 gap coverage tests.

Requirements covered:
  ASYNC-08 — job_service.check_availability_batch_service uses asyncio.gather
  ASYNC-10 — azure_sync upload uses asyncio.create_task (via upload_worker)
  ASYNC-16 — DB access stays sync SQLAlchemy (no AsyncSession)

These are behavioral tests that can fail.  They do NOT modify implementation files.
"""
import asyncio
import inspect
import uuid
from unittest.mock import AsyncMock, MagicMock, patch



# ---------------------------------------------------------------------------
# ASYNC-08: check_availability_batch_service is async and uses asyncio.gather
# ---------------------------------------------------------------------------

class TestBatchAvailabilityIsAsync:
    """ASYNC-08: batch availability uses asyncio.gather (not ThreadPoolExecutor)."""

    def test_check_availability_batch_service_is_coroutine(self):
        """The method must be an async coroutine function, not a plain function."""
        from backend.services.job_service import JobService
        js = JobService()
        assert inspect.iscoroutinefunction(js.check_availability_batch_service), (
            "check_availability_batch_service must be async def — ASYNC-08"
        )

    def test_job_service_source_uses_asyncio_gather(self):
        """The source code must use asyncio.gather for batch availability."""
        import backend.services.job_service as mod
        src = inspect.getsource(mod)
        assert "asyncio.gather" in src, (
            "job_service.py must use asyncio.gather for batch availability — ASYNC-08"
        )
        # Only actual instantiation of ThreadPoolExecutor is forbidden; comments are OK
        import re
        # Remove comments before checking
        src_no_comments = re.sub(r'#[^\n]*', '', src)
        assert "ThreadPoolExecutor" not in src_no_comments, (
            "job_service.py must not instantiate ThreadPoolExecutor — ASYNC-08"
        )

    async def test_batch_availability_result_shape_with_multiple_compounds(self):
        """asyncio.gather must aggregate results for multiple compounds in one call."""
        from backend.models.schemas import CheckAvailabilityBatchRequest
        from backend.services.job_service import JobService

        request = CheckAvailabilityBatchRequest(
            compounds=[
                {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                {"compound_name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
            ],
            similarity_threshold=90,
            activity_types=["IC50"],
        )

        class _FakeCtx:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        mock_probe = AsyncMock(return_value=[{"threshold": 90, "count": 2}])
        mock_bio = AsyncMock(return_value=True)

        with patch("backend.modules.api_client.create_chembl_client", return_value=_FakeCtx()), \
             patch("backend.modules.api_client.probe_all_thresholds", mock_probe), \
             patch("backend.modules.api_client.quick_has_bioactivity", mock_bio), \
             patch("backend.services.job_service.generate_inchikey", return_value=None):
            js = JobService()
            result = await js.check_availability_batch_service(MagicMock(), request)

        assert len(result.results) == 2, (
            "gather must return one result per compound — ASYNC-08"
        )
        assert result.available_count == 2


# ---------------------------------------------------------------------------
# ASYNC-08 / ASYNC-10: VALID_TRANSITIONS enforces PENDING_UPLOAD two-phase flow
# ---------------------------------------------------------------------------

class TestValidTransitionsShape:
    """VALID_TRANSITIONS must enforce the two-phase PENDING_UPLOAD flow (ASYNC-08)."""

    def test_pending_upload_in_valid_transitions(self):
        """PENDING_UPLOAD must be a key in VALID_TRANSITIONS."""
        from backend.services.job_service import VALID_TRANSITIONS
        from backend.models.enums import JobStatus
        assert JobStatus.PENDING_UPLOAD in VALID_TRANSITIONS, (
            "PENDING_UPLOAD must be in VALID_TRANSITIONS — ASYNC-08"
        )

    def test_processing_does_not_transition_directly_to_completed(self):
        """PROCESSING must NOT transition directly to COMPLETED (two-phase required)."""
        from backend.services.job_service import VALID_TRANSITIONS
        from backend.models.enums import JobStatus
        allowed_from_processing = VALID_TRANSITIONS[JobStatus.PROCESSING]
        assert JobStatus.COMPLETED not in allowed_from_processing, (
            "PROCESSING -> COMPLETED direct transition violates D-47 / ASYNC-08. "
            "Jobs must pass through PENDING_UPLOAD."
        )

    def test_processing_transitions_to_pending_upload(self):
        """PROCESSING must be able to transition to PENDING_UPLOAD."""
        from backend.services.job_service import VALID_TRANSITIONS
        from backend.models.enums import JobStatus
        assert JobStatus.PENDING_UPLOAD in VALID_TRANSITIONS[JobStatus.PROCESSING], (
            "PROCESSING must transition to PENDING_UPLOAD — ASYNC-08"
        )

    def test_pending_upload_transitions_to_completed(self):
        """PENDING_UPLOAD must be able to transition to COMPLETED."""
        from backend.services.job_service import VALID_TRANSITIONS
        from backend.models.enums import JobStatus
        assert JobStatus.COMPLETED in VALID_TRANSITIONS[JobStatus.PENDING_UPLOAD], (
            "PENDING_UPLOAD must transition to COMPLETED — ASYNC-08"
        )

    def test_all_job_statuses_present_in_transitions(self):
        """Every JobStatus value must appear as a key in VALID_TRANSITIONS."""
        from backend.services.job_service import VALID_TRANSITIONS
        from backend.models.enums import JobStatus
        for status in JobStatus:
            assert status in VALID_TRANSITIONS, (
                f"JobStatus.{status.name} is missing from VALID_TRANSITIONS"
            )


# ---------------------------------------------------------------------------
# ASYNC-08: mark_completed gates on PENDING_UPLOAD status
# ---------------------------------------------------------------------------

class TestMarkCompletedGatesOnPendingUpload:
    """mark_completed must return None if job is not PENDING_UPLOAD — ASYNC-08."""

    def _make_job_mock(self, status):
        job = MagicMock()
        job.id = uuid.uuid4()
        job.status = status
        return job

    def test_mark_completed_returns_none_for_processing_job(self):
        """mark_completed must reject a PROCESSING job (not PENDING_UPLOAD)."""
        from backend.services.job_service import JobService
        from backend.models.enums import JobStatus

        js = JobService()
        db = MagicMock()
        job = self._make_job_mock(JobStatus.PROCESSING)

        with patch("backend.repositories.job_repo.get_by_job_id", return_value=job):
            result = js.mark_completed(db, str(job.id))

        assert result is None, (
            "mark_completed must return None when job is not PENDING_UPLOAD — ASYNC-08. "
            "Direct PROCESSING->COMPLETED would bypass the two-phase flow."
        )

    def test_mark_completed_returns_none_for_completed_job(self):
        """mark_completed must reject an already-COMPLETED job."""
        from backend.services.job_service import JobService
        from backend.models.enums import JobStatus

        js = JobService()
        db = MagicMock()
        job = self._make_job_mock(JobStatus.COMPLETED)

        with patch("backend.repositories.job_repo.get_by_job_id", return_value=job):
            result = js.mark_completed(db, str(job.id))

        assert result is None, (
            "mark_completed must return None when job is already COMPLETED — ASYNC-08"
        )

    def test_mark_completed_succeeds_for_pending_upload_job(self):
        """mark_completed must succeed when job is PENDING_UPLOAD."""
        from backend.services.job_service import JobService
        from backend.models.enums import JobStatus

        js = JobService()
        db = MagicMock()
        job = self._make_job_mock(JobStatus.PENDING_UPLOAD)

        with patch("backend.repositories.job_repo.get_by_job_id", return_value=job):
            result = js.mark_completed(db, str(job.id))

        assert result is not None, (
            "mark_completed must return the updated job when status is PENDING_UPLOAD — ASYNC-08"
        )
        assert job.status == JobStatus.COMPLETED, (
            "mark_completed must set job.status to COMPLETED — ASYNC-08"
        )
        assert job.progress == 100.0, "mark_completed must set progress to 100"


# ---------------------------------------------------------------------------
# ASYNC-10: upload_worker.start() uses asyncio.create_task
# ---------------------------------------------------------------------------

class TestUploadWorkerUsesAsyncioCreateTask:
    """ASYNC-10: azure upload path uses asyncio.create_task (via upload_worker)."""

    def test_upload_worker_start_source_uses_create_task(self):
        """upload_worker.start() must use asyncio.create_task to spawn the loop."""
        import backend.core.upload_worker as uw
        src = inspect.getsource(uw.start)
        assert "asyncio.create_task" in src, (
            "upload_worker.start() must use asyncio.create_task — ASYNC-10"
        )

    def test_upload_worker_has_async_loop(self):
        """_upload_loop must be an async coroutine function."""
        from backend.core.upload_worker import _upload_loop
        assert inspect.iscoroutinefunction(_upload_loop), (
            "_upload_loop must be async def — ASYNC-10"
        )

    async def test_upload_worker_start_creates_task_in_event_loop(self):
        """start() must create an asyncio.Task visible in the running loop."""
        from backend.core import upload_worker

        upload_worker._reset()

        with patch("backend.core.upload_worker.is_azure_configured", return_value=True), \
             patch("backend.core.upload_worker._upload_loop", new=AsyncMock()) as mock_loop:

            # Override _upload_loop with a long-running coroutine so task doesn't finish instantly
            async def _never_finish():
                await asyncio.sleep(999)

            mock_loop.side_effect = None
            mock_loop.return_value = None

            with patch.object(upload_worker, "_upload_loop", _never_finish):
                upload_worker.start()
                assert upload_worker.is_active(), (
                    "upload_worker must be active after start() — ASYNC-10"
                )
                # Cleanup
                await upload_worker.stop()

        upload_worker._reset()


# ---------------------------------------------------------------------------
# ASYNC-16: DB access stays sync SQLAlchemy (no AsyncSession anywhere)
# ---------------------------------------------------------------------------

class TestDbAccessStaysSyncSqlAlchemy:
    """ASYNC-16: DB repositories use sync Session, never AsyncSession."""

    def test_no_async_session_in_repositories(self):
        """Repositories must use sqlalchemy.orm.Session, not AsyncSession."""
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "AsyncSession", "backend/repositories/"],
            capture_output=True, text=True
        )
        assert result.stdout == "", (
            f"AsyncSession found in repositories — violates ASYNC-16:\n{result.stdout}"
        )

    def test_no_async_session_in_database_module(self):
        """database.py must not use AsyncSession."""
        src = open("backend/core/database.py").read()
        assert "AsyncSession" not in src, (
            "database.py must not use AsyncSession — ASYNC-16"
        )

    def test_get_db_session_returns_sync_session(self):
        """get_db_session must be a regular generator, not an async generator."""
        from backend.core.database import get_db_session
        assert not inspect.iscoroutinefunction(get_db_session), (
            "get_db_session must be a sync generator — ASYNC-16"
        )
        assert not inspect.isasyncgenfunction(get_db_session), (
            "get_db_session must not be an async generator — ASYNC-16"
        )

    def test_job_repository_methods_are_sync(self):
        """All job_repo methods must be sync (not coroutines)."""
        from backend.repositories import job_repo
        for name in ("get_by_job_id", "create_job", "get_active_jobs"):
            method = getattr(job_repo, name, None)
            if method is not None:
                assert not inspect.iscoroutinefunction(method), (
                    f"job_repo.{name} must be sync — ASYNC-16"
                )


# ---------------------------------------------------------------------------
# ASYNC-11: threading.Lock replaced in rate_limiter and api_client cache
# ---------------------------------------------------------------------------

class TestThreadingLockReplaced:
    """ASYNC-11: threading.Lock replaced by asyncio.Lock in rate_limiter and cache_non_none."""

    def test_rate_limiter_uses_asyncio_lock(self):
        """RateLimiter._lock must be asyncio.Lock, not threading.Lock."""
        from backend.core.rate_limiter import RateLimiter
        rl = RateLimiter()
        assert isinstance(rl._lock, asyncio.Lock), (
            "RateLimiter._lock must be asyncio.Lock — ASYNC-11"
        )

    def test_rate_limiter_check_is_async(self):
        """check_rate_limit must be async def."""
        from backend.core.rate_limiter import RateLimiter
        rl = RateLimiter()
        assert inspect.iscoroutinefunction(rl.check_rate_limit), (
            "check_rate_limit must be async def — ASYNC-11"
        )

    def test_no_threading_lock_in_rate_limiter_source(self):
        """rate_limiter.py must not import or use threading.Lock."""
        src = open("backend/core/rate_limiter.py").read()
        assert "threading.Lock" not in src, (
            "threading.Lock found in rate_limiter.py — ASYNC-11"
        )
        assert "threading" not in src, (
            "threading import found in rate_limiter.py — ASYNC-11"
        )

    def test_api_client_cache_uses_asyncio_lock(self):
        """api_client cache_non_none must use asyncio.Lock, not threading.Lock."""
        src = open("backend/modules/api_client.py").read()
        assert "asyncio.Lock" in src, (
            "asyncio.Lock missing from api_client.py cache_non_none — REST-12/ASYNC-11"
        )
        assert "threading.Lock" not in src, (
            "threading.Lock still present in api_client.py — REST-12/ASYNC-11"
        )
