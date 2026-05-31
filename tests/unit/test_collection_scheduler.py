"""
Unit tests for the scheduler COLLECTION branch (Phase 23, plan 23-05).

asyncio_mode = auto in pytest.ini -- no @pytest.mark.asyncio needed.

Scheduler BLOCKERS (23-CONTEXT.md), now wired in backend/core/scheduler.py:
  - BLOCKER-1: `if not job.compound_name or not job.smiles:` instant-fails any job
    without job-level smiles. COLLECTION jobs have none -> a
    `job.job_type == JobType.COLLECTION` branch runs BEFORE this guard and bypasses it.
  - BLOCKER-2: the submit loop branches on job_type -> COLLECTION dispatches
    process_collection_job (which loads members_config by job_id, D-02), once,
    with no smiles kwargs.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from backend.core import executor, scheduler
from backend.models.enums import JobType
from backend.services.collection_service import process_collection_job


def _make_collection_job():
    """A PENDING COLLECTION job carrying NO job-level smiles/compound_name.

    Both are explicitly None: a bare MagicMock attribute is truthy, which would
    let `not job.smiles` pass vacuously and make the bypass untestable. With them
    None, deleting the COLLECTION branch makes the job hit the guard and
    instant-fail (no submit) -- that is the discriminator.
    """
    job = MagicMock()
    job.id = uuid.uuid4()
    job.job_type = JobType.COLLECTION
    job.compound_name = None
    job.smiles = None
    return job


def _mock_db_session():
    """Context-manager DB session (mirrors test_scheduler.py)."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


async def test_collection_not_instant_failed():
    """A COLLECTION job (no job-level smiles) is NOT instant-failed by the
    `not job.smiles` guard -- the COLLECTION branch runs before it.

    Discriminator: with compound_name/smiles=None, removing the COLLECTION
    branch makes this job hit the guard and instant-fail (no submit). The branch
    must route it to executor.submit instead.
    """
    job = _make_collection_job()
    session = _mock_db_session()

    # int (not MagicMock) so `MAX_CONCURRENT_JOBS - get_active_count()` stays > 0.
    with patch("backend.core.scheduler.get_db_session", return_value=session), \
         patch("backend.core.scheduler.job_repo") as mock_repo, \
         patch.object(executor, "submit", new_callable=AsyncMock) as mock_submit, \
         patch.object(executor, "get_active_count", return_value=0), \
         patch("backend.config.settings") as mock_settings:
        mock_settings.MAX_CONCURRENT_JOBS = 10
        mock_repo.claim_pending_jobs.return_value = [job]

        had_work = await scheduler._process_pending()

    # The job was dispatched (not instant-failed): submit was reached exactly once.
    assert had_work is True
    mock_submit.assert_awaited_once()


async def test_collection_dispatched_to_process_collection_job():
    """A COLLECTION job is dispatched to process_collection_job, not
    process_compound_job (D-02) -- submitted exactly once with the collection
    coroutine and no smiles/compound_name kwargs.
    """
    job = _make_collection_job()
    session = _mock_db_session()

    with patch("backend.core.scheduler.get_db_session", return_value=session), \
         patch("backend.core.scheduler.job_repo") as mock_repo, \
         patch.object(executor, "submit", new_callable=AsyncMock) as mock_submit, \
         patch.object(executor, "get_active_count", return_value=0), \
         patch("backend.config.settings") as mock_settings:
        mock_settings.MAX_CONCURRENT_JOBS = 10
        mock_repo.claim_pending_jobs.return_value = [job]

        await scheduler._process_pending()

    # Submitted exactly once, with the collection coroutine (identity compare).
    mock_submit.assert_awaited_once()
    call = mock_submit.await_args
    assert call.args[0] == job.id
    assert call.args[1] is process_collection_job
    # No single-compound kwargs leaked into the collection dispatch (D-02:
    # members are loaded inside the coroutine, not passed here).
    assert "smiles" not in call.kwargs
    assert "compound_name" not in call.kwargs
