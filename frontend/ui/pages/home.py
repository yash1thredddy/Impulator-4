"""Home page for IMPULATOR.

Displays the compound browser with search and grid view.

Data flow:
- Fetches compound list from database via backend API (authoritative source)
- Falls back to local/Azure storage for legacy compatibility
"""

import html
import logging
from typing import Optional, List, Dict, Any

import streamlit as st
import streamlit.components.v1 as components

from frontend.services import get_api_client, get_compounds_cached
from frontend.utils import SessionState
from frontend.ui.components import render_compound_grid, render_compound_list

logger = logging.getLogger(__name__)


@st.cache_data
def _load_logo_b64(filename: str) -> str | None:
    """Load and cache base64-encoded logo PNG."""
    from pathlib import Path
    import base64
    path = Path("frontend/static") / filename
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def _render_logo():
    """Render the IMPULATOR logo from PNG — theme-aware using CSS media query."""
    light_b64 = _load_logo_b64("Imp_Logo_Light.png")
    dark_b64 = _load_logo_b64("Imp_Logo_Dark.png")

    if light_b64 and dark_b64:
        # Use components.html — renders in iframe with full JS support
        components.html(
            f'<div style="text-align:center;margin:0;padding:0;">'
            f'<img id="logo-light" src="data:image/png;base64,{light_b64}" '
            f'style="height:96px;object-fit:contain;" alt="IMPULATOR">'
            f'<img id="logo-dark" src="data:image/png;base64,{dark_b64}" '
            f'style="height:96px;object-fit:contain;display:none;" alt="IMPULATOR">'
            f'</div>'
            f'<script>'
            f'function update(){{'
            f'  var p=window.parent.document.querySelector("[data-testid=\\"stApp\\"]");'
            f'  if(!p)return;'
            f'  var bg=getComputedStyle(p).backgroundColor;'
            f'  var m=bg.match(/\\d+/g);'
            f'  var dark=m&&(0.299*m[0]+0.587*m[1]+0.114*m[2])<128;'
            f'  document.getElementById("logo-light").style.display=dark?"none":"block";'
            f'  document.getElementById("logo-dark").style.display=dark?"block":"none";'
            f'}}'
            f'update();setInterval(update,500);'
            f'</script>',
            height=110,
        )
    elif light_b64:
        st.markdown(
            f'<div style="text-align: center;">'
            f'<img src="data:image/png;base64,{light_b64}" '
            f'style="height: 64px; object-fit: contain;" alt="IMPULATOR">'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align: center;">'
            '<span style="font-size: 32px; font-weight: 800; font-family: Inter, sans-serif;">'
            '<span style="color: #5B21B6;">IMP</span>'
            '<span style="color: var(--text-color);">ULATOR</span>'
            '</span></div>',
            unsafe_allow_html=True
        )


def render_home_page() -> None:
    """Render the home page with compound browser."""
    # Gradient style for the New Analysis button only.
    # WARNING: Targets Streamlit internal selectors — may break on Streamlit upgrade
    st.markdown("""
    <style>
    #new-analysis-marker {
        display: none;
    }
    .stElementContainer:has(#new-analysis-marker) + .stElementContainer button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.5rem !important;
        border-radius: 0.5rem !important;
        color: white !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stElementContainer:has(#new-analysis-marker) + .stElementContainer button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    btn_col, spacer, title_col, spacer2 = st.columns([1, 1, 3, 1], vertical_alignment="center")
    with btn_col:
        # Hidden marker so CSS can target only this button
        st.markdown('<span id="new-analysis-marker"></span>', unsafe_allow_html=True)
        if st.button("+ New Analysis", type="primary", width='stretch'):
            SessionState.navigate_to_analyze()
            st.rerun()
    with title_col:
        # 3D interlocking hexagons logo — matching Recraft design
        _render_logo()

    st.subheader("Available Analyses")

    # Search and filter section
    render_search_section()

    st.divider()

    # Compound grid
    render_compound_browser()


def render_search_section() -> None:
    """Render the search and filter controls."""
    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        search_query = st.text_input(
            "Search compounds",
            value=SessionState.get('compound_search_query', ''),
            placeholder="Search by name...",
            label_visibility="hidden"
        )
        SessionState.set('compound_search_query', search_query)

    with col2:
        sort_options = ["Latest", "A-Z", "Z-A"]
        current_sort = SessionState.get('compound_sort', 'Latest')
        sort_index = sort_options.index(current_sort) if current_sort in sort_options else 0
        sort_mode = st.selectbox(
            "Sort",
            sort_options,
            index=sort_index,
            label_visibility="hidden"
        )
        SessionState.set('compound_sort', sort_mode)

    with col3:
        view_mode = st.selectbox(
            "View",
            ["Grid", "List"],
            index=0,
            label_visibility="hidden"
        )
        SessionState.set('compound_view_mode', view_mode)


def render_compound_browser() -> None:
    """Render the compound browser grid or list."""
    # Fetch compounds from backend
    compounds = _fetch_compounds()

    if compounds is None:
        st.error("Could not load compounds. Check backend connection.")
        return

    # Apply search filter
    search_query = SessionState.get('compound_search_query', '').strip().lower()
    if search_query:
        compounds = [
            c for c in compounds
            if search_query in c.get('compound_name', '').lower()
        ]

    # Apply sorting
    sort_mode = SessionState.get('compound_sort', 'Latest')
    if sort_mode == "A-Z":
        compounds = sorted(compounds, key=lambda x: x.get('compound_name', '').lower())
    elif sort_mode == "Z-A":
        compounds = sorted(compounds, key=lambda x: x.get('compound_name', '').lower(), reverse=True)
    elif sort_mode == "Latest":
        # Sort by created_at descending (newest first)
        compounds = sorted(compounds, key=lambda x: x.get('created_at', ''), reverse=True)

    select_mode = SessionState.get('compound_select_mode', False)

    # Show count
    st.caption(f"Showing {len(compounds)} compound(s)")

    if not compounds:
        if search_query:
            st.info(f"No compounds matching '{search_query}'")
        else:
            st.info("No compounds yet. Click '+ New Analysis' to get started.")
        return

    # Selection action bar (when in select mode)
    if select_mode:
        _render_selection_action_bar(compounds)

    # Render based on view mode
    view_mode = SessionState.get('compound_view_mode', 'Grid')

    if view_mode == "Grid":
        clicked = render_compound_grid(compounds, columns=3, select_mode=select_mode)
    else:
        clicked = render_compound_list(compounds, select_mode=select_mode)

    # Handle navigation (only when not in select mode)
    if clicked and not select_mode:
        SessionState.navigate_to_compound(
            clicked.get('compound_name'),
            entry_id=clicked.get('entry_id'),
            storage_path=clicked.get('storage_path'),
            is_duplicate=clicked.get('is_duplicate', False),
            duplicate_of_name=clicked.get('parent_name'),
        )
        # Deep linking: update URL on compound selection (D-05, D-06, D-08)
        entry_id = clicked.get('entry_id')
        if entry_id:
            st.query_params["compound_id"] = entry_id
            st.query_params["tab"] = "overview"
            # Mark as already applied so app.py deep link doesn't re-trigger with UUID
            SessionState.set('_last_deep_link_id', entry_id)
        st.rerun()


def _render_selection_action_bar(compounds: List[Dict[str, Any]]) -> None:
    """Render the selection action bar with Select All, Deselect All, and Delete Selected."""
    # Collect entry_ids of all visible compounds
    all_entry_ids = [c.get('entry_id') for c in compounds if c.get('entry_id')]

    # Count currently selected compounds (from live checkbox state)
    selected_ids = _get_selected_entry_ids(all_entry_ids)
    selected_count = len(selected_ids)

    # Check if we're in the confirmation stage
    confirm_delete = SessionState.get('confirm_batch_delete', False)

    if confirm_delete:
        # Read from snapshot saved when "Delete Selected" was clicked
        # (checkbox widget state may be lost across st.rerun())
        snapshot_ids = SessionState.get('batch_delete_ids', [])
        snapshot_names = SessionState.get('batch_delete_names', [])
        snapshot_count = len(snapshot_ids)

        st.warning(f"Are you sure you want to delete {snapshot_count} compound(s)? This cannot be undone.")

        # List compound names being deleted
        names_display = ", ".join(html.escape(n) for n in snapshot_names[:10])
        if len(snapshot_names) > 10:
            names_display += f" and {len(snapshot_names) - 10} more..."
        st.markdown(f"**Compounds:** {names_display}", unsafe_allow_html=True)

        confirm_col1, confirm_col2, _ = st.columns([1, 1, 3])
        with confirm_col1:
            if st.button("Confirm Delete", type="primary", width='stretch'):
                _execute_batch_delete(snapshot_ids)
        with confirm_col2:
            if st.button("Cancel Delete", width='stretch'):
                SessionState.set('confirm_batch_delete', False)
                SessionState.set('batch_delete_ids', [])
                SessionState.set('batch_delete_names', [])
                st.rerun()
    else:
        # Show action bar
        act_col1, act_col2, act_col3, act_col4 = st.columns([1, 1, 1, 2])

        with act_col1:
            if st.button("Select All", width='stretch'):
                for eid in all_entry_ids:
                    st.session_state[f"select_{eid}"] = True
                st.rerun()

        with act_col2:
            if st.button("Deselect All", width='stretch'):
                for eid in all_entry_ids:
                    st.session_state[f"select_{eid}"] = False
                st.rerun()

        with act_col3:
            delete_label = f"Delete Selected ({selected_count})"
            if st.button(delete_label, type="primary", disabled=(selected_count == 0), width='stretch'):
                # Snapshot selected IDs and names before rerun
                # (checkbox widget state doesn't survive st.rerun reliably)
                selected_names = []
                for c in compounds:
                    if c.get('entry_id') in selected_ids:
                        selected_names.append(c.get('compound_name', 'Unknown'))
                SessionState.set('batch_delete_ids', list(selected_ids))
                SessionState.set('batch_delete_names', selected_names)
                SessionState.set('confirm_batch_delete', True)
                st.rerun()

        with act_col4:
            if selected_count > 0:
                st.caption(f"{selected_count} of {len(all_entry_ids)} selected")


def _get_selected_entry_ids(all_entry_ids: List[str]) -> set:
    """Get the set of currently selected entry IDs from session state."""
    selected = set()
    for eid in all_entry_ids:
        if st.session_state.get(f"select_{eid}", False):
            selected.add(eid)
    return selected


def _execute_batch_delete(entry_ids: List[str]) -> None:
    """Execute batch deletion and show result."""
    api_client = get_api_client()
    result = api_client.delete_compounds_batch(entry_ids)

    if result.success:
        total_deleted = result.data.get('total_deleted', len(entry_ids)) if result.data else len(entry_ids)
        st.toast(f"✓ Deleted {total_deleted} compound(s)", icon="✅")
        # Show partial failure warning if some deletions failed
        if result.error:
            st.warning(result.error)
        # Clear compound list cache so UI updates immediately
        try:
            from frontend.services import get_compounds_cached
            get_compounds_cached.clear()
        except Exception:
            pass
        _exit_select_mode()
        st.rerun()
    else:
        st.error(f"Failed to delete compounds: {result.error}")
        SessionState.set('confirm_batch_delete', False)
        st.rerun()


def _exit_select_mode() -> None:
    """Exit selection mode and clear all selection state."""
    SessionState.set('compound_select_mode', False)
    SessionState.set('confirm_batch_delete', False)
    SessionState.set('batch_delete_ids', [])
    SessionState.set('batch_delete_names', [])
    # Clear all selection checkboxes
    keys_to_clear = [k for k in st.session_state if k.startswith("select_")]
    for k in keys_to_clear:
        del st.session_state[k]


def _fetch_compounds() -> Optional[List[Dict[str, Any]]]:
    """Fetch completed compounds from database.

    Uses database values only - no ZIP downloads on home page.
    This makes the home page load instantly without network I/O.

    Returns:
        List of compound dictionaries, or None on error
    """
    try:
        # Fetch from database (authoritative source) with st.cache_data TTL=60s
        # Include duplicates and tag them in the UI
        response = get_compounds_cached(per_page=100, include_duplicates=True)

        if response.success and response.compounds:
            compounds = []
            for compound in response.compounds:
                # Use database values only - no ZIP download needed
                compounds.append({
                    "compound_name": compound.get("compound_name", "Unknown"),
                    "entry_id": compound.get("entry_id"),
                    "storage_path": compound.get("storage_path"),
                    "smiles": compound.get("smiles", ""),
                    "created_at": compound.get("processed_at"),
                    "similarity_threshold": compound.get("similarity_threshold", 90) or 90,
                    "has_imp_warning": (compound.get("imp_candidates") or 0) > 0,
                    "chembl_id": compound.get("chembl_id", ""),
                    "total_activities": compound.get("total_activities", 0) or 0,
                    "num_outliers": compound.get("num_outliers", 0) or 0,
                    "qed": compound.get("qed", 0.0) or 0.0,
                    "imp_score": compound.get("imp_score"),
                    "is_duplicate": compound.get("is_duplicate", False),
                    "parent_id": compound.get("parent_id"),
                    "parent_name": compound.get("parent_name"),
                    "version_count": compound.get("version_count", 1),
                })
            return compounds

        # Database is empty - show message
        logger.info("Database returned no compounds")
        return []

    except Exception as e:
        logger.error(f"Error fetching compounds: {e}")
        return None
