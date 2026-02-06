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

from frontend.services import get_api_client
from frontend.utils import SessionState
from frontend.ui.components import render_compound_grid, render_compound_list

logger = logging.getLogger(__name__)


def render_home_page() -> None:
    """Render the home page with compound browser."""
    st.title(" IMPULATOR")
    st.caption("IMP Navigator - Identify Invalid Metabolic Panaceas")

    # Search and filter section
    render_search_section()

    st.divider()

    # Compound grid
    render_compound_browser()


def render_search_section() -> None:
    """Render the search and filter controls."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    with col1:
        search_query = st.text_input(
            "Search compounds",
            value=SessionState.get('compound_search_query', ''),
            placeholder="Search by name...",
            label_visibility="collapsed"
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
            label_visibility="collapsed"
        )
        SessionState.set('compound_sort', sort_mode)

    with col3:
        view_mode = st.selectbox(
            "View",
            ["Grid", "List"],
            index=0,
            label_visibility="collapsed"
        )
        SessionState.set('compound_view_mode', view_mode)

    with col4:
        if st.button("+ New Analysis", type="primary", use_container_width=True):
            SessionState.navigate_to_analyze()
            st.rerun()


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
            storage_path=clicked.get('storage_path')
        )
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
            if st.button("Confirm Delete", type="primary", use_container_width=True):
                _execute_batch_delete(snapshot_ids)
        with confirm_col2:
            if st.button("Cancel Delete", use_container_width=True):
                SessionState.set('confirm_batch_delete', False)
                SessionState.set('batch_delete_ids', [])
                SessionState.set('batch_delete_names', [])
                st.rerun()
    else:
        # Show action bar
        act_col1, act_col2, act_col3, act_col4 = st.columns([1, 1, 1, 2])

        with act_col1:
            if st.button("Select All", use_container_width=True):
                for eid in all_entry_ids:
                    st.session_state[f"select_{eid}"] = True
                st.rerun()

        with act_col2:
            if st.button("Deselect All", use_container_width=True):
                for eid in all_entry_ids:
                    st.session_state[f"select_{eid}"] = False
                st.rerun()

        with act_col3:
            delete_label = f"Delete Selected ({selected_count})"
            if st.button(delete_label, type="primary", disabled=(selected_count == 0), use_container_width=True):
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
        st.success(f"Successfully deleted {total_deleted} compound(s)")
        _exit_select_mode()
        st.rerun()
    else:
        st.error(f"Failed to delete compounds: {result.error}")
        SessionState.set('confirm_batch_delete', False)


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
        # Fetch from database (authoritative source)
        # Include duplicates and tag them in the UI
        api_client = get_api_client()
        response = api_client.get_compounds_from_db(per_page=100, include_duplicates=True)

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
                    "duplicate_of": compound.get("duplicate_of"),
                })
            return compounds

        # Database is empty - show message
        logger.info("Database returned no compounds")
        return []

    except Exception as e:
        logger.error(f"Error fetching compounds: {e}")
        return None
