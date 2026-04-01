"""Async task executor for background job processing.

Manages concurrent job processing via asyncio.Task dict with
Semaphore-based admission control. Replaces ThreadPoolExecutor.

Thread budget: ~2 (only run_in_executor default pool for CPU work).

Note: job_id is uuid.UUID throughout (Postgres migration v2.2). Dict keys
use str(job_id) for serialization safety.
"""
import asyncio
from typing import Any
from collections.abc import Awaitable, Callable

import structlog

from backend.config import settings

logger = structlog.get_logger(__name__)

_tasks: dict[str, asyncio.Task] = {}
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Lazy-init semaphore (must create after event loop starts)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)
    return _semaphore


async def submit(job_id, coro_func: Callable[..., Awaitable], **kwargs) -> Any:
    """Submit async job for background execution (per D-04).

    Creates an asyncio.Task wrapping the coroutine. Semaphore limits
    concurrency to MAX_CONCURRENT_JOBS.

    Args:
        job_id: UUID job identifier
        coro_func: Async function (must accept job_id as first arg)
        **kwargs: Keyword arguments for coro_func
    Returns:
        job_id
    """
    key = str(job_id)
    if key in _tasks:
        logger.warning("job_already_tracked", job_id=key)
        return job_id
    task = asyncio.create_task(
        _run_with_semaphore(key, coro_func, job_id, **kwargs),
        name=f"job-{key[:8]}",
    )
    _tasks[key] = task
    logger.info("job_submitted", job_id=key)
    return job_id


async def _run_with_semaphore(key: str, coro_func, job_id, **kwargs):
    """Run job coroutine with semaphore admission control."""
    try:
        async with _get_semaphore():
            await coro_func(job_id, **kwargs)
    except asyncio.CancelledError:
        logger.info("job_task_cancelled", job_id=key)
    except Exception:
        logger.exception("job_task_error", job_id=key)
    finally:
        _tasks.pop(key, None)


def cancel(job_id: str) -> bool:
    """Cancel a running job task (per D-05). Returns True if cancel signal sent."""
    key = str(job_id)
    task = _tasks.get(key)
    if task is None or task.done():
        return False
    task.cancel()
    return True


def has_capacity() -> bool:
    """Check if executor can accept more jobs."""
    return len(_tasks) < settings.MAX_CONCURRENT_JOBS


def get_active_count() -> int:
    """Get number of active job tasks."""
    return len(_tasks)


def get_active_job_ids() -> list[str]:
    """Get list of active job ID strings."""
    return list(_tasks.keys())


def is_active(job_id: str) -> bool:
    """Check if a specific job task is still running."""
    task = _tasks.get(str(job_id))
    return task is not None and not task.done()


def stats() -> dict:
    """Get executor statistics (per D-61)."""
    return {
        "max_concurrent_jobs": settings.MAX_CONCURRENT_JOBS,
        "active_jobs": len(_tasks),
        "slots_available": max(0, settings.MAX_CONCURRENT_JOBS - len(_tasks)),
        "has_capacity": len(_tasks) < settings.MAX_CONCURRENT_JOBS,
        "jobs": [
            {"job_id": k, "state": "running" if not t.done() else "done"}
            for k, t in _tasks.items()
        ],
    }


async def shutdown(timeout: float | None = None):
    """Graceful shutdown: cancel all tasks, gather with timeout (per D-06)."""
    t = timeout if timeout is not None else settings.SHUTDOWN_TIMEOUT
    if not _tasks:
        return
    logger.info("executor_shutdown_start", active=len(_tasks), timeout=t)
    tasks = list(_tasks.values())
    for task in tasks:
        task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=t,
        )
    except asyncio.TimeoutError:
        logger.warning("executor_shutdown_timeout", remaining=len(_tasks))
    _tasks.clear()
    logger.info("executor_shutdown_complete")


def _reset():
    """Reset module state for testing."""
    global _semaphore
    _tasks.clear()
    _semaphore = None
