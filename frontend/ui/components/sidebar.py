"""Sidebar component for IMPULATOR.

Displays active jobs with independent polling (no full page rerun).
Uses @st.fragment for partial updates of the jobs section.

Smart Polling:
- Only polls when there are active jobs (pending/processing)
- Stops polling when all jobs complete
- Polling resumes when new jobs are submitted
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

from frontend.services import get_api_client, delete_from_cache
from frontend.utils import SessionState, VIEW_HOME, VIEW_ANALYZE
from frontend.config.settings import config

logger = logging.getLogger(__name__)

# Session state keys for polling control
_POLLING_ACTIVE_KEY = "polling_active"
_LAST_ACTIVE_JOBS_KEY = "last_active_job_count"
_VIEWED_JOBS_KEY = "viewed_job_ids"


# Maximum number of viewed job IDs to track (prevents memory leak)
_MAX_VIEWED_JOBS = 100


def _get_viewed_jobs() -> set:
    """Get set of job IDs that have been viewed."""
    return st.session_state.get(_VIEWED_JOBS_KEY, set())


def _mark_job_viewed(job_id: str) -> None:
    """Mark a job as viewed so it disappears from sidebar.

    Limits the set size to prevent memory leak in long-running sessions.
    """
    viewed = _get_viewed_jobs()
    viewed.add(job_id)

    # Prevent unbounded growth - keep only most recent entries
    if len(viewed) > _MAX_VIEWED_JOBS:
        # Convert to list, keep last N entries, convert back to set
        viewed_list = list(viewed)
        viewed = set(viewed_list[-_MAX_VIEWED_JOBS:])

    st.session_state[_VIEWED_JOBS_KEY] = viewed


def render_sidebar() -> None:
    """Render the sidebar with active jobs and navigation."""
    with st.sidebar:
        st.markdown("## IMPULATOR")
        st.markdown("*IMP Navigator*")

        st.divider()

        # Navigation buttons - these need full rerun for navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Home", key="nav_home", width='stretch'):
                SessionState.navigate_to_home()
                st.rerun()
        with col2:
            if st.button("+ New", key="nav_analyze", width='stretch'):
                SessionState.navigate_to_analyze()
                st.rerun()

        st.divider()

        # Smart polling: only use polling fragment when jobs are active
        if is_polling_active():
            render_active_jobs_polling()
        else:
            render_active_jobs_static()

        st.divider()

        # Backend status
        render_backend_status()


def is_polling_active() -> bool:
    """Check if polling should be active based on session state."""
    return st.session_state.get(_POLLING_ACTIVE_KEY, False)


def start_polling():
    """Enable polling - call this when submitting a new job."""
    st.session_state[_POLLING_ACTIVE_KEY] = True
    logger.debug("Polling started")


def stop_polling():
    """Disable polling - called automatically when no active jobs."""
    st.session_state[_POLLING_ACTIVE_KEY] = False
    logger.debug("Polling stopped")


@st.fragment(run_every=config.JOB_POLL_INTERVAL_SECONDS)
def render_active_jobs_polling() -> None:
    """Render active jobs section with automatic polling.

    Uses @st.fragment to poll for job updates without triggering
    full page reruns. Only this fragment reruns on the interval.
    """
    st.markdown("### Active Jobs")

    active_jobs, has_active = _fetch_and_check_jobs()

    if active_jobs is None:
        st.caption("Connection error")
        return

    if not active_jobs:
        st.caption("No active jobs")
        # No jobs at all - stop polling
        stop_polling()
        return

    for job in active_jobs:
        render_job_card(job)

    if has_active:
        st.caption(f"Polling every {config.JOB_POLL_INTERVAL_SECONDS}s")
    else:
        # All jobs completed - stop polling
        stop_polling()
        st.success("✅ All jobs completed!")
        # Add refresh button to update home page with new compounds
        if st.button("🔄 Refresh Home", key="refresh_home_completed", width='stretch'):
            SessionState.navigate_to_home()
            st.rerun()


def render_active_jobs_static() -> None:
    """Render active jobs section without polling.

    Used when no jobs are active to avoid unnecessary API calls.
    Does NOT fetch from backend - just shows "No active jobs".
    Polling is started by the job submission flow, not by checking.
    """
    st.markdown("### Active Jobs")
    st.caption("No active jobs")


def _fetch_and_check_jobs():
    """Fetch active jobs and check if any are in progress.

    Filters out jobs that have already been viewed by the user.

    Returns:
        Tuple of (jobs_list, has_active_jobs) or (None, False) on error
    """
    client = get_api_client()
    try:
        active_jobs = client.get_active_jobs()

        # Filter out jobs that user has already viewed
        viewed_ids = _get_viewed_jobs()
        active_jobs = [j for j in active_jobs if j.get('id') not in viewed_ids]

        has_active = any(j.get('status') in ('pending', 'processing') for j in active_jobs)
        return active_jobs, has_active
    except Exception as e:
        logger.error(f"Failed to fetch active jobs: {e}")
        return None, False


def render_job_card(job: Dict[str, Any]) -> None:
    """Render a single job card in the sidebar.

    Args:
        job: Job dictionary from the API
    """
    job_id = job.get('id', 'unknown')
    compound_name = job.get('compound_name', 'Unknown')
    entry_id = job.get('entry_id')  # UUID for storage lookup
    status = job.get('status', 'pending')
    progress = job.get('progress', 0.0)
    current_step = job.get('current_step', '')

    with st.container(border=True):
        # Compound name with status emoji
        status_emoji = get_status_emoji(status)
        st.markdown(f"{status_emoji} **{_truncate(compound_name, 20)}**")

        # Progress display based on status
        if status == 'processing':
            st.progress(progress / 100.0, text=_truncate(current_step, 20) or f"{progress:.0f}%")
        elif status == 'pending':
            st.caption("Queued...")
        elif status == 'completed':
            st.caption("Ready to view")
        elif status == 'failed':
            st.caption("Processing failed")

        # Action buttons - use callbacks to avoid rerun issues in fragment
        if status in ('pending', 'processing'):
            st.button(
                "Cancel",
                key=f"cancel_{job_id}",
                width='stretch',
                on_click=_on_cancel_job,
                args=(job_id,)
            )
        elif status == 'completed':
            # View button - marks job as viewed so it disappears from sidebar
            if st.button("View Results", key=f"view_{job_id}", type="primary", width='stretch'):
                _mark_job_viewed(job_id)
                SessionState.navigate_to_compound(compound_name, entry_id=entry_id)
                st.rerun()
        elif status == 'failed':
            st.button(
                "Delete",
                key=f"delete_{job_id}",
                width='stretch',
                on_click=_on_delete_job,
                args=(job_id, compound_name)
            )


def _truncate(text: str, max_len: int = 25) -> str:
    """Truncate text to fit in sidebar."""
    if not text:
        return ""
    if len(text) > max_len:
        return text[:max_len-3] + "..."
    return text


def get_status_emoji(status: str) -> str:
    """Get emoji for job status."""
    return {
        'pending': '⏳',
        'processing': '⚙️',
        'completed': '✅',
        'failed': '❌',
        'cancelled': '🚫',
    }.get(status, '❓')


def _on_cancel_job(job_id: str) -> None:
    """Callback for cancel button - runs within fragment context."""
    client = get_api_client()
    try:
        response = client.cancel_job(job_id)
        if response.success:
            st.toast("Job cancelled")
        else:
            st.toast(f"Failed: {response.error}", icon="")
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        st.toast(f"Error: {e}", icon="")


def _on_delete_job(job_id: str, compound_name: str = None) -> None:
    """Callback for delete button - runs within fragment context.

    Deletes job from backend (which cleans up Azure) and also
    clears the local frontend cache.
    """
    client = get_api_client()
    try:
        response = client.delete_job(job_id)
        if response.success:
            # Also clear from local frontend cache
            if compound_name:
                delete_from_cache(compound_name)
            st.toast("Job and results deleted")
        else:
            st.toast(f"Failed: {response.error}", icon="")
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        st.toast(f"Error: {e}", icon="")


def render_backend_status() -> None:
    """Render backend health status."""
    client = get_api_client()

    try:
        is_healthy = client.health_check()
        if is_healthy:
            st.caption(" Backend connected")
        else:
            st.caption(" Backend unavailable")
    except Exception:
        st.caption(" Backend error")
