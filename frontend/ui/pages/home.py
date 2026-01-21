"""Home page for IMPULATOR.

Displays the compound browser with search and grid view.

Data flow:
- Fetches compound list from database via backend API (authoritative source)
- Falls back to local/Azure storage for legacy compatibility
"""

import logging
from typing import Optional, List, Dict, Any

import streamlit as st

from frontend.services import list_results, is_azure_configured, get_cached_result, get_api_client
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

    # Show count
    st.caption(f"Showing {len(compounds)} compound(s)")

    if not compounds:
        if search_query:
            st.info(f"No compounds matching '{search_query}'")
        else:
            st.info("No compounds yet. Click '+ New Analysis' to get started.")
        return

    # Render based on view mode
    view_mode = SessionState.get('compound_view_mode', 'Grid')

    if view_mode == "Grid":
        clicked = render_compound_grid(compounds, columns=3)  # 3 columns for better sizing
    else:
        clicked = render_compound_list(compounds)

    # Handle navigation
    if clicked:
        SessionState.navigate_to_compound(
            clicked.get('compound_name'),
            entry_id=clicked.get('entry_id')
        )
        st.rerun()


def _fetch_compounds() -> Optional[List[Dict[str, Any]]]:
    """Fetch completed compounds from database.

    Data flow:
    - Primary: Fetches from database via /api/v1/compounds (proper compound names)
    - Fallback: Local/Azure storage for legacy compatibility

    Returns:
        List of compound dictionaries, or None on error
    """
    try:
        # Try to get compounds from database (authoritative source with proper names)
        api_client = get_api_client()
        response = api_client.get_compounds_from_db(per_page=100)

        if response.success and response.compounds:
            compounds = []
            for compound in response.compounds:
                # Use entry_id for storage lookup, compound_name for display
                entry_id = compound.get("entry_id")
                compound_name = compound.get("compound_name", "Unknown")

                # Try to get cached summary using entry_id (UUID-based storage)
                summary = None
                if entry_id:
                    summary = get_cached_result(entry_id)
                # Fall back to compound name for legacy data
                if not summary:
                    summary = get_cached_result(compound_name)

                compounds.append({
                    "compound_name": compound_name,
                    "entry_id": entry_id,
                    "smiles": compound.get("smiles") or (summary.get("smiles", "") if summary else ""),
                    "created_at": compound.get("processed_at"),
                    "similar_count": summary.get("similar_count", 0) if summary else 0,
                    "has_imp_warning": compound.get("imp_candidates", 0) > 0 if compound.get("imp_candidates") else False,
                    "chembl_id": compound.get("chembl_id", ""),
                    "total_activities": compound.get("total_activities", 0),
                    "num_outliers": summary.get("num_outliers", 0) if summary else 0,
                    "qed": summary.get("qed", 0.0) if summary else 0.0,
                    "avg_oqpla_score": compound.get("avg_oqpla_score"),
                    "is_duplicate": compound.get("is_duplicate", False),
                })
            return compounds

        # Fallback to legacy blob-based listing if database is empty or API fails
        logger.warning("Database returned no compounds, falling back to blob listing")
        results = list_results()

        if not results and not is_azure_configured():
            st.caption("Azure not configured - results will only persist locally")

        compounds = []
        for result in results:
            compound_name = result.get("compound_name", "Unknown")
            summary = get_cached_result(compound_name)

            compounds.append({
                "compound_name": compound_name,
                "smiles": summary.get("smiles", "") if summary else "",
                "created_at": result.get("last_modified"),
                "similar_count": summary.get("similar_count", 0) if summary else 0,
                "has_imp_warning": summary.get("has_imp_candidates", False) if summary else False,
                "chembl_id": summary.get("chembl_id", "") if summary else "",
                "total_activities": summary.get("total_activities", 0) if summary else 0,
                "num_outliers": summary.get("num_outliers", 0) if summary else 0,
                "qed": summary.get("qed", 0.0) if summary else 0.0,
            })

        return compounds

    except Exception as e:
        logger.error(f"Error fetching compounds: {e}")
        return None
