"""Compound card component for IMPULATOR.

Displays a compound card in the grid view with summary info.
Uses RDKit for 2D structure rendering (like the old code).
"""

import html
import logging
import base64
from io import BytesIO
from typing import Dict, Any, Optional

import streamlit as st


logger = logging.getLogger(__name__)

# Try to import RDKit for 2D structure rendering
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - 2D structure rendering disabled")


def _imp_score_badge_css(score: float | None) -> str:
    """Return a CSS border-left style string for IMP score color coding.

    Args:
        score: IMP score (0.0-1.0) or None.

    Returns:
        CSS style string for border-left, or empty string if no score.
    """
    if score is None:
        return ""
    if score >= 0.9:
        return "border-left: 4px solid #4CAF50;"   # Green - Exceptional
    elif score >= 0.7:
        return "border-left: 4px solid #2196F3;"   # Blue - Strong
    elif score >= 0.5:
        return "border-left: 4px solid #FF9800;"   # Orange - Moderate
    elif score >= 0.3:
        return "border-left: 4px solid #9E9E9E;"   # Gray - Weak
    return ""


def render_compound_card(compound: Dict[str, Any], key_prefix: str = "", select_mode: bool = False) -> bool:
    """Render a compound card in the grid view (matching old UI style).

    Args:
        compound: Compound data dictionary with fields:
            - compound_name: Name of the compound
            - smiles: SMILES string
            - created_at: Creation timestamp
            - similarity_threshold: Similarity threshold % used for search
            - has_imp_warning: Whether IMP warning exists
            - chembl_id: Optional ChEMBL ID
            - total_activities: Optional activity count
            - num_outliers: Optional outlier count
            - qed: Optional QED score
        key_prefix: Prefix for widget keys to ensure uniqueness
        select_mode: Whether selection mode is active (shows checkbox instead of View button)

    Returns:
        bool: True if the "View" button was clicked (always False in select mode)
    """
    compound_name = compound.get('compound_name', 'Unknown')
    entry_id = compound.get('entry_id', '')  # Unique identifier for key generation
    smiles = compound.get('smiles', '')
    is_duplicate = compound.get('is_duplicate', False)
    version_count = compound.get('version_count', 1)

    # Optional fields from metadata
    chembl_id = compound.get('chembl_id', '')
    total_activities = compound.get('total_activities', 0)
    imp_score = compound.get('imp_score')
    qed = compound.get('qed', 0.0)
    similarity_threshold = compound.get('similarity_threshold', 90)

    # Escape for XSS prevention (CSS text-overflow handles truncation dynamically)
    safe_compound_name = html.escape(str(compound_name))

    with st.container(border=True):
        st.markdown('<div class="imp-compound-card-marker"></div>', unsafe_allow_html=True)

        # Selection checkbox in select mode
        if select_mode and entry_id:
            cb_key = f"select_{entry_id}"
            st.checkbox(
                safe_compound_name,
                key=cb_key,
                label_visibility="collapsed",
            )

        # Compound name (clean, no badge — badge moves to ChEMBL line)
        st.markdown(
            f"<div style='text-align: center; margin: 0 0 10px 0; min-height: 3.6rem; display: flex; align-items: center; justify-content: center;' title='{safe_compound_name}'>"
            f"<div style='font-size: clamp(0.9rem, 2.5vw, 1.4rem); font-weight: 600; line-height: 1.25; "
            f"display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; "
            f"overflow: hidden; text-overflow: ellipsis;'>"
            f"{safe_compound_name}</div></div>",
            unsafe_allow_html=True
        )

        # 2D Structure preview using RDKit
        if smiles:
            _render_rdkit_structure(smiles, compound_name)
        else:
            # Fixed height placeholder when no structure
            st.markdown(
                "<div style='height: 200px; display: flex; align-items: center; justify-content: center; "
                "color: #888; font-size: 16px;'>Structure not available</div>",
                unsafe_allow_html=True
            )

        # ChEMBL ID + optional DUPLICATE badge (right corner)
        dup_badge = ""
        if is_duplicate:
            dup_badge = (
                "<span style='background: #ff6b35; color: white; "
                "padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; "
                "letter-spacing: 0.5px;'>DUPLICATE</span>"
            )
        elif version_count > 1:
            dup_badge = (
                f"<span style='background: #667eea; color: white; "
                f"padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; "
                f"letter-spacing: 0.5px;'>{version_count} VERSIONS</span>"
            )
        if chembl_id and str(chembl_id) != 'nan':
            safe_chembl_id = html.escape(str(chembl_id))
            st.markdown(
                f"<div style='display: flex; justify-content: space-between; align-items: center; "
                f"margin: 8px 0; min-height: 1.5rem;'>"
                f"<span style='color: var(--text-color); opacity: 0.5; font-size: 14px;'>ChEMBL: {safe_chembl_id}</span>"
                f"{dup_badge}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='display: flex; justify-content: flex-end; margin: 8px 0; min-height: 1.2rem;'>"
                f"{dup_badge}</div>",
                unsafe_allow_html=True
            )

        # Stats using HTML flexbox for consistent layout
        # Escape all values for XSS prevention
        qed_display = f"{qed:.2f}" if qed and qed > 0 else "N/A"
        imp_score_display = f"{imp_score:.2f}" if imp_score is not None else "N/A"
        safe_total_activities = html.escape(str(total_activities))
        safe_imp_score = html.escape(imp_score_display)
        safe_qed_display = html.escape(str(qed_display))
        safe_similarity = html.escape(str(similarity_threshold))

        st.markdown(
            f"""<div style='display: flex; justify-content: space-between; font-size: clamp(12px, 1.4vw, 16px); margin: 8px 0;'>
                <span style='min-height: 1.5rem; display: inline-flex; align-items: center; gap: 0.2rem;'><b>Activities:</b><span>{safe_total_activities}</span></span>
                <span style='min-height: 1.5rem; display: inline-flex; align-items: center; gap: 0.2rem;'><b>IMP Score:</b><span>{safe_imp_score}</span></span>
            </div>
            <div style='display: flex; justify-content: space-between; font-size: clamp(12px, 1.4vw, 16px); margin: 8px 0;'>
                <span style='min-height: 1.5rem; display: inline-flex; align-items: center; gap: 0.2rem;'><b>QED:</b><span>{safe_qed_display}</span></span>
                <span style='min-height: 1.5rem; display: inline-flex; align-items: center; gap: 0.2rem;'><b>Similarity:</b><span>{safe_similarity}%</span></span>
            </div>""",
            unsafe_allow_html=True
        )

        # View button (hidden in select mode)
        if not select_mode:
            # key_prefix already contains unique grid position
            # Use entry_id if available for extra uniqueness with duplicate compound names
            if entry_id:
                button_key = f"{key_prefix}view_{entry_id}"
            else:
                # For legacy compounds without entry_id, key_prefix alone is unique per grid position
                button_key = f"{key_prefix}view"
            if st.button("View Details", key=button_key, type="primary", width='stretch'):
                return True

    return False


def _render_rdkit_structure(smiles: str, compound_name: str, size: tuple = (300, 200)) -> None:
    """Render 2D structure using RDKit.

    Args:
        smiles: SMILES string
        compound_name: Compound name for logging
        size: Image size (width, height)
    """
    if not smiles or smiles == 'nan' or not str(smiles).strip():
        # Fixed height placeholder
        st.markdown(
            "<div style='height: 200px; display: flex; align-items: center; justify-content: center; "
            "color: #888; font-size: 16px;'>Structure not available</div>",
            unsafe_allow_html=True
        )
        return

    if not RDKIT_AVAILABLE:
        # Fallback to SmilesDrawer if RDKit not available
        render_structure_thumbnail(smiles, compound_name, "")
        return

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            st.markdown(
                "<div style='height: 200px; display: flex; align-items: center; justify-content: center; "
                "color: #888; font-size: 16px;'>Invalid structure</div>",
                unsafe_allow_html=True
            )
            return

        # Generate the molecular image
        img = Draw.MolToImage(mol, size=size)

        # Convert image to base64 for HTML display
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Display the image centered with fixed container height
        safe_name = html.escape(compound_name)
        st.markdown(
            f'<div style="display: flex; justify-content: center; align-items: center; '
            f'height: 200px; padding: 8px; background: white; border-radius: 6px;">'
            f'<img src="data:image/png;base64,{img_str}" alt="{safe_name}" '
            f'style="max-height: 190px; max-width: 100%; object-fit: contain;" />'
            f'</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        logger.error(f"Error rendering molecule for {compound_name}: {e}")
        st.markdown(
            "<div style='height: 200px; display: flex; align-items: center; justify-content: center; "
            "color: #888; font-size: 16px;'>Structure not available</div>",
            unsafe_allow_html=True
        )


def render_structure_thumbnail(smiles: str, compound_name: str, key_prefix: str = "") -> None:
    """Render a small 2D structure thumbnail using SmilesDrawer.

    This uses JavaScript to render without triggering a Streamlit rerun.

    Args:
        smiles: SMILES string
        compound_name: Name for the canvas ID
        key_prefix: Key prefix for uniqueness
    """
    import html
    import re as _re
    safe_smiles = html.escape(smiles)
    # Sanitize canvas_id to prevent HTML/JS injection via compound_name
    canvas_id = _re.sub(r'[^a-zA-Z0-9_]', '', f"{key_prefix}struct_{compound_name}")

    # SmilesDrawer rendering via JS (doesn't trigger rerun)
    html_content = f'''
    <div style="width: 100%; height: 100px; display: flex; justify-content: center; align-items: center;">
        <canvas id="{canvas_id}" style="max-width: 100%; max-height: 100px;"></canvas>
    </div>
    <script>
        (function() {{
            function renderSmiles() {{
                if (typeof SmilesDrawer === 'undefined') {{
                    // Load SmilesDrawer if not already loaded
                    var script = document.createElement('script');
                    script.src = 'https://cdn.jsdelivr.net/npm/smiles-drawer@2.0.1/dist/smiles-drawer.min.js';
                    script.onload = function() {{
                        doRender();
                    }};
                    document.head.appendChild(script);
                }} else {{
                    doRender();
                }}
            }}

            function doRender() {{
                var drawer = new SmilesDrawer.SmiDrawer({{
                    width: 150,
                    height: 100
                }});
                drawer.draw("{safe_smiles}", "#{canvas_id}", "light");
            }}

            renderSmiles();
        }})();
    </script>
    '''

    st.components.v1.html(html_content, height=110)


def render_compound_grid(compounds: list, columns: int = 3, select_mode: bool = False) -> Optional[str]:
    """Render a grid of compound cards (3 columns for better sizing).

    Args:
        compounds: List of compound dictionaries
        columns: Number of columns in the grid (default 3)
        select_mode: Whether selection mode is active

    Returns:
        Optional[dict]: Dict with compound_name and entry_id of clicked compound, or None
    """
    if not compounds:
        st.info("No compounds found. Submit a new analysis to get started.")
        return None

    clicked_compound = None

    # Equal-height cards: force Streamlit column children to stretch
    st.markdown(
        """<style>
        /* Make compound cards in each row equal height */
        div[data-testid="stHorizontalBlock"]:has(.imp-compound-card-marker) {
            align-items: stretch;
        }
        div[data-testid="stHorizontalBlock"]:has(.imp-compound-card-marker) > div[data-testid="stColumn"] {
            display: flex;
            flex-direction: column;
        }
        div[data-testid="stHorizontalBlock"]:has(.imp-compound-card-marker) > div[data-testid="stColumn"] > div {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        div[data-testid="stHorizontalBlock"]:has(.imp-compound-card-marker) > div[data-testid="stColumn"] > div > div[data-testid="stLayoutWrapper"] {
            flex: 1;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    # ── Pagination (must be multiple of columns for full rows) ──
    PAGE_SIZE = columns * 16  # 48 for 3-column grid
    total = len(compounds)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page_key = "compound_grid_page"

    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    current_page = min(st.session_state[page_key], total_pages)

    st.caption(f"{total} compounds")
    if total_pages > 1:
        _list_pagination(page_key, current_page, total_pages, "grid_top")

    start = (current_page - 1) * PAGE_SIZE
    page_compounds = compounds[start:start + PAGE_SIZE]

    # Create grid
    for row_start in range(0, len(page_compounds), columns):
        row_compounds = page_compounds[row_start:row_start + columns]
        cols = st.columns(columns)

        for i, compound in enumerate(row_compounds):
            with cols[i]:
                global_idx = start + row_start + i
                if render_compound_card(compound, key_prefix=f"grid_{global_idx}_", select_mode=select_mode):
                    clicked_compound = {
                        'compound_name': compound.get('compound_name'),
                        'entry_id': compound.get('entry_id'),
                        'storage_path': compound.get('storage_path'),
                        'is_duplicate': compound.get('is_duplicate', False),
                        'parent_id': compound.get('parent_id'),
                        'parent_name': compound.get('parent_name'),
                    }

    if total_pages > 1:
        _list_pagination(page_key, current_page, total_pages, "grid_bot")

    return clicked_compound


def _get_structure_base64(smiles: str, size: tuple = (120, 90)) -> str:
    """Generate a base64-encoded PNG of the 2D structure, or empty string on failure."""
    if not smiles or smiles == 'nan' or not RDKIT_AVAILABLE:
        return ""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return ""
        img = Draw.MolToImage(mol, size=size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception:
        return ""


def _list_pagination(page_key: str, current_page: int, total_pages: int, pos: str) -> None:
    """PDB-style pagination bar: First Prev | Page X of Y | Next Last."""
    _, c_first, c_prev, c_label, c_next, c_last, _ = st.columns([2, 1, 1, 2, 1, 1, 2])
    with c_first:
        if st.button("⟪ First", key=f"list_first_{pos}_{page_key}", disabled=current_page <= 1,
                      use_container_width=True):
            st.session_state[page_key] = 1
            st.rerun()
    with c_prev:
        if st.button("◁ Prev", key=f"list_prev_{pos}_{page_key}", disabled=current_page <= 1,
                      use_container_width=True):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    c_label.markdown(
        f"<div style='text-align:center;padding:8px 0;font-size:15px;font-weight:500;'>"
        f"Page {current_page} of {total_pages}</div>",
        unsafe_allow_html=True,
    )
    with c_next:
        if st.button("Next ▷", key=f"list_next_{pos}_{page_key}", disabled=current_page >= total_pages,
                      use_container_width=True):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    with c_last:
        if st.button("Last ⟫", key=f"list_last_{pos}_{page_key}", disabled=current_page >= total_pages,
                      use_container_width=True):
            st.session_state[page_key] = total_pages
            st.rerun()


def render_compound_list(compounds: list, select_mode: bool = False) -> Optional[dict]:
    """Render a paginated list view of compounds as PDB-style card rows.

    Args:
        compounds: List of compound dictionaries
        select_mode: Whether selection mode is active

    Returns:
        Optional[dict]: Dict with compound_name and entry_id of clicked compound, or None
    """
    if not compounds:
        st.info("No compounds found. Submit a new analysis to get started.")
        return None

    clicked_compound = None

    # ── Pagination setup ──
    PAGE_SIZE = 50
    total = len(compounds)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page_key = "compound_list_page"

    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    current_page = min(st.session_state[page_key], total_pages)

    st.caption(f"{total} compounds")
    if total_pages > 1:
        _list_pagination(page_key, current_page, total_pages, "top")

    start = (current_page - 1) * PAGE_SIZE
    page_compounds = compounds[start:start + PAGE_SIZE]

    for i, compound in enumerate(page_compounds):
        global_idx = start + i
        compound_name = compound.get('compound_name', 'Unknown')
        entry_id = compound.get('entry_id', '')
        smiles = compound.get('smiles', '')
        similarity_threshold = compound.get('similarity_threshold', 90)
        is_duplicate = compound.get('is_duplicate', False)
        version_count = compound.get('version_count', 1)
        imp_score = compound.get('imp_score')
        qed = compound.get('qed', 0.0)
        total_activities = compound.get('total_activities', 0)
        chembl_id = compound.get('chembl_id', '')

        safe_name = html.escape(str(compound_name))
        safe_chembl = html.escape(str(chembl_id)) if chembl_id and str(chembl_id) != 'nan' else ''

        # ── Badges ──
        badges_html = ""
        if is_duplicate:
            badges_html += (
                ' <span style="display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;'
                'font-weight:700;background:#ff6b3522;color:#ff6b35;border:1px solid #ff6b3544;'
                'vertical-align:middle;">DUPLICATE</span>'
            )
        elif version_count > 1:
            badges_html += (
                f' <span style="display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;'
                f'font-weight:700;background:#667eea22;color:#667eea;border:1px solid #667eea44;'
                f'vertical-align:middle;">{version_count} VERSIONS</span>'
            )

        # ── IMP score pill ──
        imp_display = f"{imp_score:.2f}" if imp_score is not None else "N/A"
        if imp_score is not None and imp_score >= 0.7:
            imp_color = "#22c55e"
        elif imp_score is not None and imp_score >= 0.5:
            imp_color = "#FF9800"
        elif imp_score is not None and imp_score >= 0.3:
            imp_color = "#9E9E9E"
        else:
            imp_color = "#666"
        imp_pill = (
            f'<span style="display:inline-block;padding:3px 12px;border-radius:12px;font-size:12px;'
            f'font-weight:700;background:{imp_color}22;color:{imp_color};'
            f'border:1px solid {imp_color}44;">{html.escape(imp_display)}</span>'
        )

        # ── 2D structure thumbnail (links to compound detail page) ──
        img_b64 = _get_structure_base64(smiles, size=(160, 120))
        detail_url = f"?compound_id={entry_id}" if entry_id else ""
        if img_b64:
            img_tag = (
                f'<img src="data:image/png;base64,{img_b64}" '
                f'style="width:90px;height:90px;border-radius:6px;object-fit:contain;'
                f'background:white;flex-shrink:0;" '
                f'onerror="this.style.display=\'none\'" />'
            )
            if detail_url:
                img_html = (
                    f'<a href="{detail_url}" target="_self" '
                    f'style="flex-shrink:0;" title="View {safe_name}">{img_tag}</a>'
                )
            else:
                img_html = img_tag
        else:
            img_html = (
                '<div style="width:90px;height:90px;border-radius:6px;background:#333;'
                'display:flex;align-items:center;justify-content:center;flex-shrink:0;'
                'font-size:10px;color:#888;">No structure</div>'
            )

        # ── Metadata lines ──
        qed_display = f"{qed:.2f}" if qed and qed > 0 else "N/A"

        line1_parts = []
        if safe_chembl:
            line1_parts.append(
                f'<b>ChEMBL:</b> <span style="color:#3b82f6;font-weight:600;">{safe_chembl}</span>'
            )
        line1_parts.append(f'<b>Activities:</b> {html.escape(str(total_activities))}')
        line1_parts.append(f'<b>IMP Score:</b> {imp_pill}')
        line1 = ' &nbsp;&nbsp; '.join(line1_parts)

        line2 = (
            f'<b>Similarity:</b> {html.escape(str(similarity_threshold))}%'
            f' &nbsp;&nbsp; <b>QED:</b> {html.escape(qed_display)}'
        )

        # ── Card HTML ──
        card_html = (
            f'<div style="display:flex;gap:16px;padding:14px 16px;'
            f'border-bottom:1px solid rgba(128,128,128,0.2);align-items:flex-start;">'
            f'{img_html}'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-size:17px;font-weight:500;margin-bottom:6px;">'
            f'{safe_name}{badges_html}</div>'
            f'<div style="font-size:14px;opacity:0.85;line-height:1.8;">{line1}</div>'
            f'<div style="font-size:14px;opacity:0.85;line-height:1.8;">{line2}</div>'
            f'</div>'
            f'</div>'
        )

        if select_mode and entry_id:
            col_cb, col_card = st.columns([0.3, 9.7])
            with col_cb:
                st.checkbox("", key=f"select_{entry_id}", label_visibility="collapsed")
            with col_card:
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.markdown(card_html, unsafe_allow_html=True)
            button_key = f"lv_{global_idx}_{entry_id}" if entry_id else f"lv_{global_idx}"
            if st.button("View Details", key=button_key, type="primary", use_container_width=True):
                clicked_compound = {
                    'compound_name': compound_name,
                    'entry_id': entry_id,
                    'storage_path': compound.get('storage_path'),
                    'is_duplicate': is_duplicate,
                    'parent_id': compound.get('parent_id'),
                    'parent_name': compound.get('parent_name'),
                }

    # Bottom pagination
    if total_pages > 1:
        _list_pagination(page_key, current_page, total_pages, "bot")

    return clicked_compound


def _format_date(date_input) -> str:
    """Format a date string or datetime object for display.

    Args:
        date_input: Either a datetime object or ISO date string

    Returns:
        Formatted date string (e.g., "Dec 31")
    """
    from datetime import datetime

    try:
        # Handle datetime objects directly
        if isinstance(date_input, datetime):
            return date_input.strftime("%b %d")

        # Handle None or empty
        if not date_input:
            return ""

        # Handle string input
        date_str = str(date_input)
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%b %d")
    except Exception:
        # Fallback: return truncated string if possible
        try:
            return str(date_input)[:10]
        except Exception:
            return ""
