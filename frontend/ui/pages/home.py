"""Home page for IMPULATOR.

Displays the compound browser with search and grid view.

Data flow:
- Fetches compound list from database via backend API (authoritative source)
- Falls back to local/Azure storage for legacy compatibility
"""

import html
import logging
from typing import Optional, Any

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

    /* WARNING: Targets Streamlit internal selectors — may break on Streamlit upgrade */
    div[data-testid="stVerticalBlock"]:has(#compound-grid-marker) div[data-testid="stVerticalBlockBorderWrapper"]:has(.imp-compound-card-marker) {
        height: 100%;
    }

    /* WARNING: Targets Streamlit internal selectors — may break on Streamlit upgrade */
    div[data-testid="stVerticalBlock"]:has(#compound-grid-marker) div[data-testid="stVerticalBlockBorderWrapper"]:has(.imp-compound-card-marker) > div {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    /* WARNING: Targets Streamlit internal selectors — may break on Streamlit upgrade */
    div[data-testid="stVerticalBlock"]:has(#compound-grid-marker) div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }

    /* WARNING: Targets Streamlit internal selectors — may break on Streamlit upgrade */
    div[data-testid="stVerticalBlock"]:has(#compound-grid-marker) div[data-testid="stVerticalBlockBorderWrapper"]:has(.imp-compound-card-marker) div[data-testid="stButton"] {
        margin-top: auto;
    }

    /* WARNING: Targets Streamlit internal selectors — may break on Streamlit upgrade */
    @media (max-width: 1100px) {
        div[data-testid="stVerticalBlock"]:has(#compound-grid-marker) div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }
        div[data-testid="stVerticalBlock"]:has(#compound-grid-marker) div[data-testid="column"] {
            min-width: calc(50% - 1rem) !important;
            flex: 1 1 calc(50% - 1rem) !important;
        }
    }

    /* WARNING: Targets Streamlit internal selectors — may break on Streamlit upgrade */
    @media (max-width: 700px) {
        div[data-testid="stVerticalBlock"]:has(#compound-grid-marker) div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
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

    # Fetch compounds once for stats banner + browser
    compounds = _fetch_compounds()

    if compounds is None:
        st.error("Could not load compounds. Check backend connection.")
        return

    if not compounds:
        # Issue 1: Better empty state
        _render_empty_state()
        return

    # Search and filter section
    render_search_section()

    # Compound grid (pass pre-fetched compounds)
    render_compound_browser(compounds)


def render_search_section() -> None:
    """Render the search and filter controls."""
    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        search_query = st.text_input(
            "Search compounds",
            value=SessionState.get('compound_search_query', ''),
            placeholder="Search by name, ChEMBL ID, or SMILES substructure...",
            label_visibility="hidden"
        )
        SessionState.set('compound_search_query', search_query)

    with col2:
        sort_options = ["Latest", "A-Z", "Z-A", "IMP Score \u2193", "Activities \u2193", "QED \u2193"]
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


def render_compound_browser(compounds: list[dict[str, Any]]) -> None:
    """Render the compound browser grid or list."""
    # Apply search filter: text match on name/chembl/smiles, or RDKit substructure match
    search_query = SessionState.get('compound_search_query', '').strip()
    if search_query:
        compounds = _filter_compounds(compounds, search_query)

    # Apply sorting (Issue 2: additional sort options)
    sort_mode = SessionState.get('compound_sort', 'Latest')
    if sort_mode == "A-Z":
        compounds = sorted(compounds, key=lambda x: x.get('compound_name', '').lower())
    elif sort_mode == "Z-A":
        compounds = sorted(compounds, key=lambda x: x.get('compound_name', '').lower(), reverse=True)
    elif sort_mode == "Latest":
        compounds = sorted(compounds, key=lambda x: x.get('created_at', ''), reverse=True)
    elif sort_mode == "IMP Score \u2193":
        # Sort by imp_score descending, None/0 values last
        compounds = sorted(
            compounds,
            key=lambda x: (x.get('imp_score') is None or x.get('imp_score') == 0, -(x.get('imp_score') or 0)),
        )
    elif sort_mode == "Activities \u2193":
        compounds = sorted(
            compounds,
            key=lambda x: (not x.get('total_activities'), -(x.get('total_activities') or 0)),
        )
    elif sort_mode == "QED \u2193":
        compounds = sorted(
            compounds,
            key=lambda x: (x.get('qed') is None or x.get('qed') == 0, -(x.get('qed') or 0)),
        )

    select_mode = SessionState.get('compound_select_mode', False)

    if not compounds:
        if search_query:
            escaped_query = html.escape(search_query)
            st.info(f"No compounds matching '{escaped_query}'")
        return

    # Selection action bar (when in select mode)
    if select_mode:
        _render_selection_action_bar(compounds)

    # Render based on view mode
    view_mode = SessionState.get('compound_view_mode', 'Grid')

    if view_mode == "Grid":
        with st.container():
            st.markdown('<div id="compound-grid-marker"></div>', unsafe_allow_html=True)
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


def _render_selection_action_bar(compounds: list[dict[str, Any]]) -> None:
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


def _get_selected_entry_ids(all_entry_ids: list[str]) -> set:
    """Get the set of currently selected entry IDs from session state."""
    selected = set()
    for eid in all_entry_ids:
        if st.session_state.get(f"select_{eid}", False):
            selected.add(eid)
    return selected


def _execute_batch_delete(entry_ids: list[str]) -> None:
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


def _render_empty_state() -> None:
    """Render an informative empty state for new users."""
    st.markdown(
        '<div style="text-align:center;padding:40px 20px;">'
        '<h2 style="margin-bottom:8px;">Welcome to IMPULATOR</h2>'
        '<p style="opacity:0.7;font-size:16px;margin-bottom:24px;">'
        'Analyze chemical compounds for Invalid Metabolic Panaceas (IMPs)</p>'
        '<div style="display:flex;justify-content:center;gap:32px;margin-bottom:32px;flex-wrap:wrap;">'
        '<div style="text-align:center;max-width:160px;">'
        '<div style="font-size:28px;margin-bottom:4px;">1</div>'
        '<div style="font-size:14px;opacity:0.8;">Submit SMILES or CSV</div></div>'
        '<div style="text-align:center;max-width:160px;">'
        '<div style="font-size:28px;margin-bottom:4px;">2</div>'
        '<div style="font-size:14px;opacity:0.8;">ChEMBL similarity + IMP scoring</div></div>'
        '<div style="text-align:center;max-width:160px;">'
        '<div style="font-size:28px;margin-bottom:4px;">3</div>'
        '<div style="font-size:14px;opacity:0.8;">Browse results here</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([2, 1, 2])
    with center:
        st.markdown('<span id="new-analysis-marker"></span>', unsafe_allow_html=True)
        if st.button("+ New Analysis", type="primary", use_container_width=True):
            SessionState.navigate_to_analyze()
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


def _filter_compounds(compounds: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Filter compounds by text match or RDKit substructure search.

    If the query looks like a SMILES/SMARTS pattern (contains chemistry characters
    like =, #, @, [, or ring digits after letters), attempt substructure matching
    via RDKit. Falls back to text search if RDKit fails or isn't available.
    """
    query_lower = query.lower()

    # Always do text matching first
    text_matches = [
        c for c in compounds
        if query_lower in c.get('compound_name', '').lower()
        or query_lower in c.get('chembl_id', '').lower()
        or query_lower in c.get('smiles', '').lower()
    ]

    # Attempt substructure match if query looks like a structure pattern
    # Heuristic: contains chemistry-specific characters unlikely in plain names
    structure_chars = set('=#@[]/()')
    looks_like_structure = bool(structure_chars & set(query))

    if not looks_like_structure:
        return text_matches

    try:
        from rdkit import Chem

        # Try as SMARTS first (more expressive), then SMILES
        pattern = Chem.MolFromSmarts(query)
        if pattern is None:
            pattern = Chem.MolFromSmiles(query)
        if pattern is None:
            return text_matches

        substruct_matches = set()
        for i, c in enumerate(compounds):
            smiles = c.get('smiles', '')
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None and mol.HasSubstructMatch(pattern):
                substruct_matches.add(i)

        # Merge: text matches + substructure matches (deduplicated, preserve order)
        text_ids = {id(c) for c in text_matches}
        result = list(text_matches)
        for i, c in enumerate(compounds):
            if i in substruct_matches and id(c) not in text_ids:
                result.append(c)
        return result

    except ImportError:
        return text_matches
    except Exception as e:
        logger.debug(f"Substructure search failed: {e}")
        return text_matches


def _compound_from_api(compound: dict) -> dict:
    """Convert a backend API compound dict to the home page format."""
    return {
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
    }


def _fetch_compounds() -> Optional[list[dict[str, Any]]]:
    """Fetch ALL compounds from database, paginating through the API.

    Backend caps at 100 per page, so we loop until all pages are fetched.
    First page uses the cached wrapper (60s TTL), subsequent pages hit the API directly.

    Returns:
        List of compound dictionaries, or None on error
    """
    try:
        PAGE_SIZE = 100

        # First page — cached
        response = get_compounds_cached(page=1, per_page=PAGE_SIZE, include_duplicates=True)
        if not response.success:
            logger.error(f"Failed to fetch compounds: {response.error}")
            return None

        if not response.compounds:
            return []

        compounds = [_compound_from_api(c) for c in response.compounds]

        # Fetch remaining pages if there are more
        total = response.total
        if total > PAGE_SIZE:
            api = get_api_client()
            pages_needed = (total + PAGE_SIZE - 1) // PAGE_SIZE
            for page in range(2, pages_needed + 1):
                resp = api.get_compounds_from_db(
                    page=page, per_page=PAGE_SIZE, include_duplicates=True
                )
                if resp.success and resp.compounds:
                    compounds.extend(_compound_from_api(c) for c in resp.compounds)
                else:
                    break

        return compounds

    except Exception as e:
        logger.error(f"Error fetching compounds: {e}")
        return None
