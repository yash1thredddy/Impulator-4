"""Sidebar component for IMPULATOR.

Displays active jobs with independent polling (no full page rerun).
Uses @st.fragment for partial updates of the jobs section.

Smart Polling:
- Only polls when there are active jobs (pending/processing)
- Stops polling when all jobs complete
- Polling resumes when new jobs are submitted
"""

import html as html_mod
import logging
from typing import Any

import base64
from pathlib import Path

import requests
import streamlit as st

from frontend.services import get_api_client, get_compounds_cached, delete_from_cache, set_session_id
from frontend.utils import SessionState
from frontend.config.settings import config

logger = logging.getLogger(__name__)

# Session state keys for polling control
_POLLING_ACTIVE_KEY = "polling_active"
_LAST_ACTIVE_JOBS_KEY = "last_active_job_count"
_VIEWED_JOBS_KEY = "viewed_job_ids"


# Maximum number of viewed job IDs to track (prevents memory leak)
_MAX_VIEWED_JOBS = 100

_SIDEBAR_STATUS_PRIORITY = {
    'processing': 0,
    'pending_upload': 1,
    'pending': 2,
    'completed': 3,
    'failed': 4,
    'cancelled': 5,
}


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


def _sort_jobs_for_sidebar(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep actively running jobs ahead of completed items in the sidebar."""
    jobs = sorted(jobs, key=lambda job: job.get('created_at', ''), reverse=True)
    return sorted(
        jobs,
        key=lambda job: _SIDEBAR_STATUS_PRIORITY.get(job.get('status', ''), 99),
    )


def _render_sidebar_logo() -> None:
    """Render the IMPULATOR logo in the sidebar — uses PNG logos with theme detection."""
    static_dir = Path(__file__).parent.parent.parent / "static"
    light_path = static_dir / "Imp_Logo_Light.png"
    dark_path = static_dir / "Imp_Logo_Dark.png"

    light_b64 = ""
    dark_b64 = ""
    if light_path.exists():
        light_b64 = base64.b64encode(light_path.read_bytes()).decode()
    if dark_path.exists():
        dark_b64 = base64.b64encode(dark_path.read_bytes()).decode()

    if light_b64 and dark_b64:
        st.components.v1.html(
            f"""
            <div id="sidebar-logo" style="text-align:center;padding:8px 0;">
                <img id="logo-l" src="data:image/png;base64,{light_b64}"
                     style="width:100%;max-width:260px;object-fit:contain;display:none;">
                <img id="logo-d" src="data:image/png;base64,{dark_b64}"
                     style="width:100%;max-width:260px;object-fit:contain;display:none;">
            </div>
            <script>
            (function() {{
                function update() {{
                    var app = window.parent.document.querySelector('[data-testid="stSidebar"]');
                    if (!app) return;
                    var bg = getComputedStyle(app).backgroundColor;
                    var m = bg.match(/\\d+/g);
                    var dark = m && (0.299*m[0] + 0.587*m[1] + 0.114*m[2]) < 128;
                    document.getElementById('logo-l').style.display = dark ? 'none' : 'block';
                    document.getElementById('logo-d').style.display = dark ? 'block' : 'none';
                }}
                update();
                setInterval(update, 1000);
            }})();
            </script>
            """,
            height=90,
        )
    else:
        st.markdown("## IMPULATOR")
        st.markdown("*IMP Navigator*")


def render_sidebar() -> None:
    """Render the sidebar with active jobs and navigation."""
    with st.sidebar:
        # Logo — load icon + text logo, theme-aware
        _render_sidebar_logo()

        st.divider()

        # Navigation buttons - these need full rerun for navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Home", key="nav_home", width='stretch'):
                SessionState.navigate_to_home()
                st.query_params.clear()
                st.rerun()
        with col2:
            if st.button("+ New", key="nav_analyze", width='stretch'):
                SessionState.navigate_to_analyze()
                st.rerun()

        # Select mode toggle (only on home page)
        if SessionState.get_current_view() == "home":
            select_mode = SessionState.get('compound_select_mode', False)
            if select_mode:
                if st.button("Cancel Selection", key="nav_cancel_select", width='stretch'):
                    _exit_select_mode()
                    st.rerun()
            else:
                if st.button("Select", key="nav_select", width='stretch'):
                    SessionState.set('compound_select_mode', True)
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

    active_jobs, has_active, failed_jobs = _fetch_and_check_jobs()

    if active_jobs is None:
        st.caption("Connection error")
        _render_failed_jobs_section(failed_jobs)
        return

    if not active_jobs:
        st.caption("No active jobs")
        stop_polling()
    else:
        for job in active_jobs:
            render_job_card(job)

        if has_active:
            st.caption(f"Polling every {config.JOB_POLL_INTERVAL_SECONDS}s")
        else:
            # All jobs completed - stop polling and invalidate compound list cache
            stop_polling()
            get_compounds_cached.clear()
            st.success("✅ All jobs completed!")
            if st.button("🔄 Refresh Home", key="refresh_home_completed", width='stretch'):
                SessionState.navigate_to_home()
                st.rerun()

    _render_failed_jobs_section(failed_jobs)


def render_active_jobs_static() -> None:
    """Render active jobs section without polling.

    Does a single fetch so persistent failed jobs show in their own section.
    Starts polling if active (pending/processing) jobs are discovered.
    """
    st.markdown("### Active Jobs")
    active_jobs, has_active, failed_jobs = _fetch_and_check_jobs()

    if active_jobs is None:
        st.caption("Connection error")
    elif not active_jobs:
        st.caption("No active jobs")
    elif has_active:
        start_polling()
        st.rerun()
        return
    else:
        for job in active_jobs:
            render_job_card(job)

    _render_failed_jobs_section(failed_jobs)


def _fetch_and_check_jobs():
    """Fetch active jobs and check if any are in progress.

    Separates failed jobs into their own list. Filters out viewed jobs
    from active/completed (failed jobs are only dismissable via Remove).

    Returns:
        Tuple of (non_failed_jobs, has_active, failed_jobs)
        or (None, False, []) on error
    """
    # Ensure session_id is set in this context — @st.fragment runs in a
    # separate context where the ContextVar from the main page is not inherited
    set_session_id(SessionState.get_session_id())
    client = get_api_client()
    try:
        all_jobs = client.get_active_jobs()

        # Separate failed jobs from active/completed
        failed_jobs = [j for j in all_jobs if j.get('status') == 'failed']
        non_failed = [j for j in all_jobs if j.get('status') != 'failed']

        # Filter out viewed jobs (only applies to completed, not failed)
        viewed_ids = _get_viewed_jobs()
        non_failed = [j for j in non_failed if j.get('id') not in viewed_ids]

        # Home should only show currently active work, not stale "View Results" cards.
        if SessionState.get_current_view() == "home":
            non_failed = [
                job for job in non_failed
                if job.get('status') in ('pending', 'processing', 'pending_upload')
            ]

        non_failed = _sort_jobs_for_sidebar(non_failed)
        failed_jobs = _sort_jobs_for_sidebar(failed_jobs)

        has_active = any(j.get('status') in ('pending', 'processing', 'pending_upload') for j in non_failed)
        return non_failed, has_active, failed_jobs
    except Exception as e:
        logger.error(f"Failed to fetch active jobs: {e}")
        return None, False, []


def _render_failed_jobs_section(failed_jobs: list) -> None:
    """Render a separate 'Failed Jobs' section if any exist."""
    dismissed = SessionState.get('dismissed_failed_jobs', set())
    visible = [j for j in failed_jobs if j.get('id') not in dismissed]
    if not visible:
        return
    st.markdown("### Failed Jobs")

    # Collect cascade-eligible jobs
    cascade_jobs = []
    for job in visible:
        cascade = job.get('cascade_results')
        if cascade:
            tiers_with_data = [t for t in cascade if t.get('count', 0) > 0]
            if tiers_with_data and job.get('input_params', {}).get('smiles'):
                cascade_jobs.append(job)

    # "Resubmit all" — each job at its own best (highest) available threshold
    if len(cascade_jobs) >= 2:
        st.button(
            f"Retry all {len(cascade_jobs)} at best threshold",
            key="cascade_resubmit_all",
            type="primary",
            width='stretch',
            on_click=_resubmit_all_at_best_threshold,
            args=(cascade_jobs,),
        )

    for job in visible:
        render_job_card(job)


def render_job_card(job: dict[str, Any]) -> None:
    """Render a single job card in the sidebar.

    Args:
        job: Job dictionary from the API
    """
    job_id = job.get('id', 'unknown')
    compound_name = job.get('compound_name', 'Unknown')
    entry_id = job.get('entry_id')  # UUID for storage lookup
    storage_path = job.get('storage_path')  # Full Azure storage path
    status = job.get('status', 'pending')
    progress = job.get('progress', 0.0)
    current_step = job.get('current_step', '')

    with st.container(border=True):
        # Compound name with status emoji
        status_emoji = get_status_emoji(status)
        safe_name = _escape_md(html_mod.escape(_truncate(compound_name, 20)))
        st.markdown(f"{status_emoji} **{safe_name}**")

        # Progress display based on status
        if status == 'processing':
            st.progress(progress / 100.0, text=_truncate(current_step, 20) or f"{progress:.0f}%")
        elif status == 'pending':
            st.caption("Queued...")
        elif status == 'completed':
            st.caption("Ready to view")
        elif status == 'pending_upload':
            st.caption("Uploading results...")
        elif status == 'failed':
            error_msg = job.get('error_message', '')
            if error_msg:
                st.caption(f"Failed: {_truncate(error_msg, 45)}")
            else:
                st.caption("Processing failed")

            # Show cascade similarity results as a dropdown with all thresholds
            cascade = job.get('cascade_results')
            if cascade:
                tiers_with_data = [t for t in cascade if t.get('count', 0) > 0]
                if tiers_with_data:
                    # Sort descending by threshold (highest = most similar first)
                    tiers_with_data.sort(key=lambda t: t['threshold'], reverse=True)
                    options = [f"{t['threshold']}% — {t['count']} compounds" for t in tiers_with_data]
                    selected = st.selectbox(
                        "Retry at",
                        options,
                        key=f"cascade_select_{job_id}",
                        label_visibility="hidden",
                    )
                    # Extract threshold from selection
                    selected_idx = options.index(selected)
                    selected_threshold = tiers_with_data[selected_idx]['threshold']
                    st.button(
                        f"Retry at {selected_threshold}%",
                        key=f"cascade_btn_{job_id}",
                        width='stretch',
                        on_click=_resubmit_at_threshold,
                        args=(job, selected_threshold),
                    )

        # Action buttons - use callbacks to avoid rerun issues in fragment
        if status in ('pending', 'processing', 'pending_upload'):
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
                SessionState.navigate_to_compound(compound_name, entry_id=entry_id, storage_path=storage_path)
                # Deep linking: update URL on compound selection
                if entry_id:
                    st.query_params["compound_id"] = entry_id
                    st.query_params["tab"] = "overview"
                    SessionState.set('_last_deep_link_id', entry_id)
                st.rerun()
        elif status == 'failed':
            st.button(
                "Remove",
                key=f"dismiss_{job_id}",
                width='stretch',
                on_click=_dismiss_failed_job,
                args=(job_id,)
            )


def _truncate(text: str, max_len: int = 25) -> str:
    """Truncate text to fit in sidebar."""
    if not text:
        return ""
    if len(text) > max_len:
        return text[:max_len-3] + "..."
    return text


def _escape_md(text: str) -> str:
    """Escape markdown special characters so text renders literally in st.markdown."""
    # Escape characters that have special meaning inside markdown inline markup
    for ch in ("\\", "`", "*", "_", "[", "]", "(", ")", "#", "+", "-", ".", "!"):
        text = text.replace(ch, "\\" + ch)
    return text


def get_status_emoji(status: str) -> str:
    """Get emoji for job status."""
    return {
        'pending': '⏳',
        'processing': '⚙️',
        'completed': '✅',
        'failed': '❌',
        'cancelled': '🚫',
        'pending_upload': '📤',
    }.get(status, '❓')


def _on_cancel_job(job_id: str) -> None:
    """Callback for cancel button - runs within fragment context."""
    client = get_api_client()
    try:
        response = client.cancel_job(job_id)
        if response.success:
            st.toast("Job cancelled")
        else:
            st.toast(f"Failed: {response.error}", icon="⚠️")
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        st.toast(f"Error: {e}", icon="⚠️")


def _dismiss_failed_job(job_id: str) -> None:
    """Delete a failed job from the backend and hide from sidebar."""
    client = get_api_client()
    try:
        response = client.delete_job(job_id)
        if response.success:
            st.toast("Failed job removed")
        else:
            logger.warning(f"Backend delete failed for job {job_id}: {response.error}")
            st.toast(f"Remove failed: {response.error}", icon="⚠️")
    except Exception as e:
        logger.error(f"Error removing failed job {job_id}: {e}")
        st.toast(f"Error: {e}", icon="⚠️")
    # Always hide locally so the UI updates even if backend delete fails
    dismissed = SessionState.get('dismissed_failed_jobs', set())
    dismissed.add(job_id)
    SessionState.set('dismissed_failed_jobs', dismissed)


def _resubmit_all_at_best_threshold(jobs: list) -> None:
    """Resubmit all cascade-eligible failed jobs, each at its highest available threshold."""
    client = get_api_client()
    submitted = 0
    failed = 0
    failure_reasons = []
    for job in jobs:
        params = job.get('input_params', {})
        compound_name = params.get('compound_name') or job.get('compound_name', 'Unknown')
        smiles = params.get('smiles', '')
        if not smiles:
            continue
        # Find this job's best (highest) threshold with data
        cascade = job.get('cascade_results', [])
        tiers_with_data = [t for t in cascade if t.get('count', 0) > 0]
        if not tiers_with_data:
            continue
        best_threshold = max(t['threshold'] for t in tiers_with_data)
        try:
            response = client.submit_job(
                compound_name=compound_name,
                smiles=smiles,
                similarity_threshold=best_threshold,
                activity_types=params.get('activity_types'),
                author_name=params.get('author_name', ''),
                duplicate_action="duplicate",
            )
            if response.success:
                job_id = job.get('id')
                if job_id:
                    _dismiss_failed_job(job_id)
                submitted += 1
            else:
                failed += 1
                reason = getattr(response, 'error', None) or 'submission rejected'
                failure_reasons.append(f"{compound_name}: {reason}")
                logger.warning(f"Resubmit failed for {compound_name}: {reason}")
        except Exception as e:
            failed += 1
            failure_reasons.append(f"{compound_name}: {e}")
            logger.error(f"Error resubmitting {compound_name}: {e}")
    if submitted:
        start_polling()
    if submitted and not failed:
        st.toast(f"Resubmitted {submitted} jobs at their best thresholds")
    elif submitted and failed:
        st.toast(f"Resubmitted {submitted} jobs, {failed} failed")
    elif failed:
        st.toast(f"Failed to resubmit {failed} jobs")


def _resubmit_at_threshold(job: dict[str, Any], threshold: int) -> None:
    """Resubmit a failed job at a lower similarity threshold."""
    client = get_api_client()
    params = job.get('input_params', {})
    compound_name = params.get('compound_name') or job.get('compound_name', 'Unknown')
    smiles = params.get('smiles', '')
    author_name = params.get('author_name', '')
    activity_types = params.get('activity_types')

    if not smiles:
        st.toast("Cannot resubmit: missing SMILES data", icon="⚠️")
        return

    try:
        response = client.submit_job(
            compound_name=compound_name,
            smiles=smiles,
            similarity_threshold=threshold,
            activity_types=activity_types,
            author_name=author_name,
            duplicate_action="duplicate",
        )
        if response.success:
            # Dismiss the old failed job
            job_id = job.get('id')
            if job_id:
                _dismiss_failed_job(job_id)
            start_polling()
            st.toast(f"Resubmitted at {threshold}%")
        else:
            st.toast(f"Resubmit failed: {response.error or response.message}", icon="⚠️")
    except Exception as e:
        logger.error(f"Error resubmitting job at {threshold}%: {e}")
        st.toast(f"Error: {e}", icon="⚠️")


def _on_delete_job(job_id: str, entry_id: str = None) -> None:
    """Callback for delete button - runs within fragment context.

    Deletes job from backend (which cleans up Azure) and also
    clears the local frontend cache.

    Args:
        job_id: Job ID to delete
        entry_id: UUID entry_id for local cache cleanup
    """
    client = get_api_client()
    try:
        response = client.delete_job(job_id)
        if response.success:
            # Also clear from local frontend cache (uses entry_id UUID)
            if entry_id:
                delete_from_cache(entry_id)
            st.toast("Job and results deleted")
        else:
            st.toast(f"Failed: {response.error}", icon="⚠️")
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        st.toast(f"Error: {e}", icon="⚠️")


def _exit_select_mode() -> None:
    """Exit compound selection mode and clear all selection state."""
    SessionState.set('compound_select_mode', False)
    SessionState.set('confirm_batch_delete', False)
    SessionState.set('batch_delete_ids', [])
    SessionState.set('batch_delete_names', [])
    keys_to_clear = [k for k in st.session_state if k.startswith("select_")]
    for k in keys_to_clear:
        del st.session_state[k]


@st.cache_data(ttl=10)
def _cached_health_check() -> bool:
    """Cached health check to avoid blocking every rerun.

    TTL=10s means we check at most once every 10 seconds.
    """
    client = get_api_client()
    return client.health_check()


def render_backend_status() -> None:
    """Render backend health status.

    Uses cached health check (TTL=10s) to avoid blocking every rerun.
    Distinguishes between backend unreachable and unexpected errors.
    """
    try:
        is_healthy = _cached_health_check()
        if is_healthy:
            st.caption(" Backend connected")
        else:
            st.caption(" Backend unavailable")
    except requests.exceptions.RequestException:
        # Backend unreachable (network error, connection refused, timeout)
        st.error("Backend unreachable")
    except Exception as e:
        logger.error(f"Health check error: {e}")
        st.warning(f"Backend status error: {type(e).__name__}")
