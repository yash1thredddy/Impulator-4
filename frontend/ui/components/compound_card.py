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

    # IMP score color badge
    badge_css = _imp_score_badge_css(imp_score)
    if badge_css:
        st.markdown(
            f"<div style='{badge_css} padding-left: 4px;'>",
            unsafe_allow_html=True,
        )

    with st.container(border=True):
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
            f"<div style='text-align: center; margin: 0 0 10px 0;' title='{safe_compound_name}'>"
            f"<div style='font-size: clamp(0.9rem, 2.5vw, 1.4rem); font-weight: 600; "
            f"white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>"
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
                f"margin: 8px 0;'>"
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
                <span><b>Activities:</b> {safe_total_activities}</span>
                <span><b>IMP Score:</b> {safe_imp_score}</span>
            </div>
            <div style='display: flex; justify-content: space-between; font-size: clamp(12px, 1.4vw, 16px); margin: 8px 0;'>
                <span><b>QED:</b> {safe_qed_display}</span>
                <span><b>Similarity:</b> {safe_similarity}%</span>
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

    # Close IMP score badge div if it was opened
    if badge_css:
        st.markdown("</div>", unsafe_allow_html=True)

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

    # Create grid
    for row_start in range(0, len(compounds), columns):
        row_compounds = compounds[row_start:row_start + columns]
        cols = st.columns(columns)

        for i, compound in enumerate(row_compounds):
            with cols[i]:
                # Use row_start + i to create unique keys per grid position
                if render_compound_card(compound, key_prefix=f"grid_{row_start + i}_", select_mode=select_mode):
                    clicked_compound = {
                        'compound_name': compound.get('compound_name'),
                        'entry_id': compound.get('entry_id'),
                        'storage_path': compound.get('storage_path'),
                        'is_duplicate': compound.get('is_duplicate', False),
                        'parent_id': compound.get('parent_id'),
                        'parent_name': compound.get('parent_name'),
                    }

    return clicked_compound


def render_compound_list(compounds: list, select_mode: bool = False) -> Optional[dict]:
    """Render a list view of compounds (alternative to grid).

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

    for i, compound in enumerate(compounds):
        compound_name = compound.get('compound_name', 'Unknown')
        entry_id = compound.get('entry_id')
        smiles = compound.get('smiles', '')[:50]  # Truncate
        similarity_threshold = compound.get('similarity_threshold', 90)
        has_imp_warning = compound.get('has_imp_warning', False)
        is_duplicate = compound.get('is_duplicate', False)
        imp_score = compound.get('imp_score')

        # IMP score color indicator for list rows
        badge_css = _imp_score_badge_css(imp_score)

        if select_mode and entry_id:
            # Selection mode: checkbox + name + smiles + similarity
            col0, col1, col2, col3 = st.columns([0.5, 3, 4, 2])

            with col0:
                cb_key = f"select_{entry_id}"
                st.checkbox("Select", key=cb_key, label_visibility="collapsed")

            with col1:
                safe_name = html.escape(compound_name)
                name_style = f"style='{badge_css} padding-left: 6px;'" if badge_css else ""
                if is_duplicate:
                    st.markdown(f"<div {name_style}>**{safe_name}** <span style='color: #ff6b35; font-size: 12px;'>[DUP]</span></div>", unsafe_allow_html=True)
                elif has_imp_warning:
                    st.markdown(f"<div {name_style}>**{safe_name}** </div>" if badge_css else f"**{safe_name}** ", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div {name_style}>**{safe_name}**</div>" if badge_css else f"**{safe_name}**", unsafe_allow_html=True)

            with col2:
                st.code(smiles + "..." if len(compound.get('smiles', '')) > 50 else smiles)

            with col3:
                st.caption(f"Sim: {similarity_threshold}%")
        else:
            # Normal mode: name + smiles + similarity + view button
            col1, col2, col3, col4 = st.columns([3, 4, 2, 2])

            with col1:
                safe_name = html.escape(compound_name)
                name_style = f"style='{badge_css} padding-left: 6px;'" if badge_css else ""
                if is_duplicate:
                    st.markdown(f"<div {name_style}>**{safe_name}** <span style='color: #ff6b35; font-size: 12px;'>[DUP]</span></div>", unsafe_allow_html=True)
                elif has_imp_warning:
                    st.markdown(f"<div {name_style}>**{safe_name}** </div>" if badge_css else f"**{safe_name}** ", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div {name_style}>**{safe_name}**</div>" if badge_css else f"**{safe_name}**", unsafe_allow_html=True)

            with col2:
                st.code(smiles + "..." if len(compound.get('smiles', '')) > 50 else smiles)

            with col3:
                st.caption(f"Sim: {similarity_threshold}%")

            with col4:
                button_key = f"list_view_{i}_{entry_id}" if entry_id else f"list_view_{i}"
                if st.button("View", key=button_key):
                    clicked_compound = {
                        'compound_name': compound_name,
                        'entry_id': entry_id,
                        'storage_path': compound.get('storage_path'),
                        'is_duplicate': is_duplicate,
                        'parent_id': compound.get('parent_id'),
                        'parent_name': compound.get('parent_name'),
                    }

        st.divider()

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
