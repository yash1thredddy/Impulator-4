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

from frontend.utils import SessionState

logger = logging.getLogger(__name__)

# Try to import RDKit for 2D structure rendering
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - 2D structure rendering disabled")


def render_compound_card(compound: Dict[str, Any], key_prefix: str = "") -> bool:
    """Render a compound card in the grid view (matching old UI style).

    Args:
        compound: Compound data dictionary with fields:
            - compound_name: Name of the compound
            - smiles: SMILES string
            - created_at: Creation timestamp
            - similar_count: Number of similar compounds found
            - has_imp_warning: Whether IMP warning exists
            - chembl_id: Optional ChEMBL ID
            - total_activities: Optional activity count
            - num_outliers: Optional outlier count
            - qed: Optional QED score
            - similarity_threshold: Optional similarity threshold
        key_prefix: Prefix for widget keys to ensure uniqueness

    Returns:
        bool: True if the "View" button was clicked
    """
    compound_name = compound.get('compound_name', 'Unknown')
    smiles = compound.get('smiles', '')
    similar_count = compound.get('similar_count', 0)
    has_imp_warning = compound.get('has_imp_warning', False)

    # Optional fields from metadata
    chembl_id = compound.get('chembl_id', '')
    total_activities = compound.get('total_activities', 0)
    num_outliers = compound.get('num_outliers', 0)
    qed = compound.get('qed', 0.0)
    similarity_threshold = compound.get('similarity_threshold', 90)

    # Truncate long compound names (max 20 chars for display)
    display_name = compound_name if len(compound_name) <= 20 else compound_name[:18] + "..."

    # Escape for XSS prevention
    safe_display_name = html.escape(display_name)
    safe_compound_name = html.escape(compound_name)

    with st.container(border=True):
        # Compound name centered - larger font, minimal top margin
        st.markdown(
            f"<h2 style='text-align: center; margin: 0 0 10px 0; padding-top: 0; overflow: hidden; "
            f"text-overflow: ellipsis; white-space: nowrap; font-size: 1.5rem; font-weight: 600;' "
            f"title='{safe_compound_name}'>{safe_display_name}</h2>",
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

        # ChEMBL ID if available
        if chembl_id and str(chembl_id) != 'nan':
            safe_chembl_id = html.escape(str(chembl_id))
            st.markdown(f"<p style='color: #888; font-size: 14px; margin: 8px 0;'>ChEMBL: {safe_chembl_id}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='margin: 8px 0;'>&nbsp;</p>", unsafe_allow_html=True)

        # Stats using HTML flexbox for consistent layout
        # Escape all values for XSS prevention
        qed_display = f"{qed:.2f}" if qed and qed > 0 else "N/A"
        safe_total_activities = html.escape(str(total_activities))
        safe_num_outliers = html.escape(str(num_outliers))
        safe_qed_display = html.escape(str(qed_display))
        safe_similarity = html.escape(str(similarity_threshold))

        st.markdown(
            f"""<div style='display: flex; justify-content: space-between; font-size: 16px; margin: 8px 0;'>
                <span><b>Activities:</b> {safe_total_activities}</span>
                <span><b>Outliers:</b> {safe_num_outliers}</span>
            </div>
            <div style='display: flex; justify-content: space-between; font-size: 16px; margin: 8px 0;'>
                <span><b>QED:</b> {safe_qed_display}</span>
                <span><b>Similarity:</b> {safe_similarity}%</span>
            </div>""",
            unsafe_allow_html=True
        )

        # View button
        if st.button("View Details", key=f"{key_prefix}view_{compound_name}", type="primary", use_container_width=True):
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
    safe_smiles = html.escape(smiles)
    canvas_id = f"{key_prefix}struct_{compound_name.replace(' ', '_')}"

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


def render_compound_grid(compounds: list, columns: int = 3) -> Optional[str]:
    """Render a grid of compound cards (3 columns for better sizing).

    Args:
        compounds: List of compound dictionaries
        columns: Number of columns in the grid (default 3)

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
                if render_compound_card(compound, key_prefix=f"grid_{row_start}_"):
                    clicked_compound = {
                        'compound_name': compound.get('compound_name'),
                        'entry_id': compound.get('entry_id'),
                    }

    return clicked_compound


def render_compound_list(compounds: list) -> Optional[dict]:
    """Render a list view of compounds (alternative to grid).

    Args:
        compounds: List of compound dictionaries

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
        similar_count = compound.get('similar_count', 0)
        has_imp_warning = compound.get('has_imp_warning', False)

        col1, col2, col3, col4 = st.columns([3, 4, 2, 2])

        with col1:
            if has_imp_warning:
                st.markdown(f"**{compound_name}** ")
            else:
                st.markdown(f"**{compound_name}**")

        with col2:
            st.code(smiles + "..." if len(compound.get('smiles', '')) > 50 else smiles)

        with col3:
            st.caption(f" {similar_count}")

        with col4:
            if st.button("View", key=f"list_view_{i}_{compound_name}"):
                clicked_compound = {
                    'compound_name': compound_name,
                    'entry_id': entry_id,
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
