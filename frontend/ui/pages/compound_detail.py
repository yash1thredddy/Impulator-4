"""Compound detail page for IMPULATOR.

Displays full analysis results with improved UX and organization.
"""

import hashlib
import html
import logging
import math
import re
from typing import Any, Optional
from urllib.parse import quote_plus

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from backend.modules.imp_gmm import (
    DEFAULT_COMPONENTS,
    DEFAULT_RANDOM_STATE,
    MAX_COMPONENTS,
    MIN_COMPONENTS,
    REFERENCE_CORPUS_KEY,
    best_fit_k,
    cluster_membership,
    fit_gmm,
    gmm_sentinel_message,
    load_reference_corpus,
)
from backend.modules.imp_presentation import (
    IMP_SCORE_FLOOR,
    IMP_SCORE_CEILING,
    format_imp_score,
    render_imp_range_bar_global,
    render_imp_range_bar_dynamic,
)
from frontend.services import (
    get_api_client,
    delete_from_cache,
    smart_load_summary,
    smart_load_dataframe,
)
from frontend.utils import SessionState, sanitize_compound_name
from frontend.ui.components import render_2d_structure, embed_structure_viewer
from frontend.ui.components.charts import (
    apply_impulator_theme,
    create_gmm_density_overlay,
    create_gmm_probability_bar,
    get_plotly_theme,
)
from frontend.ui.components.plotly_legend import plotly_legend_monitor

logger = logging.getLogger(__name__)

# Import scipy for regression statistics
from scipy import stats as scipy_stats  # noqa: E402


def _maybe_embed_structure_viewer(chart_id, df, x_col=None, y_col=None, z_col=None):
    """Conditionally embed hover structure viewer with inline toggle."""
    if "SMILES" not in df.columns:
        return

    # Session-persisted toggle (default on)
    if "hover_structure_preview" not in st.session_state:
        st.session_state["hover_structure_preview"] = True

    enabled = st.toggle(
        "🧬 Hover 2D Preview",
        value=st.session_state["hover_structure_preview"],
        help="Show 2D molecular structure when hovering over data points",
        key=f"hover_toggle_{chart_id}",
    )
    st.session_state["hover_structure_preview"] = enabled

    if not enabled:
        return

    embed_structure_viewer(
        chart_id=chart_id,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        name_col="Molecule_Name" if "Molecule_Name" in df.columns else None,
    )


def render_compound_detail_page() -> None:
    """Render the compound detail page."""
    compound_name = SessionState.get("selected_compound")
    entry_id = SessionState.get("selected_compound_entry_id")
    storage_path = SessionState.get("selected_compound_storage_path")
    is_duplicate = SessionState.get("selected_compound_is_duplicate", False)
    parent_name = SessionState.get("selected_compound_duplicate_of_name")
    # D-12 (COLL-14): when drilling into a collection member, the page points at
    # the collection ZIP + a "compounds/{safe_name}/" prefix so the UNMODIFIED
    # renderer reads that member's section. Default "" keeps the single-compound
    # behavior byte-identical.
    internal_prefix = SessionState.get("selected_compound_internal_prefix", "") or ""

    # Fix UUID-as-name: if compound_name looks like a UUID, fetch the real name
    import re

    _uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
    )
    if compound_name and _uuid_pattern.match(compound_name) and entry_id:
        try:
            api = get_api_client()
            response = api._request("GET", f"/api/v1/compounds/{entry_id}")
            if response.status_code == 200:
                data = response.json()
                if data.get("compound_name"):
                    compound_name = data["compound_name"]
                    SessionState.set("selected_compound", compound_name)
        except Exception:
            pass  # Keep UUID as fallback

    if not compound_name:
        st.error("No compound selected")
        if st.button("Go to Home"):
            SessionState.navigate_to_home()
            st.rerun()
        return

    # Header
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("← Back", width="stretch"):
            SessionState.navigate_to_home()
            st.query_params.clear()
            st.rerun()
    with col2:
        safe_compound_name = html.escape(compound_name)
        st.markdown(
            f"<h2 style='text-align: center; margin: 0;'>{safe_compound_name}</h2>",
            unsafe_allow_html=True,
        )
        if is_duplicate:
            dup_label = "Duplicate compound"
            if parent_name:
                safe_parent = html.escape(parent_name)
                dup_label = f"Duplicate of {safe_parent}"
            st.markdown(
                f"<p style='text-align: center; margin: 2px 0 0 0; color: #ff6b35; "
                f"font-size: 13px; font-weight: 600;'>&#9888; {dup_label}</p>",
                unsafe_allow_html=True,
            )
    with col3:
        if st.button("🗑️", width="stretch", help="Delete compound"):
            _show_delete_confirmation(compound_name, entry_id)

    # Show success toast after delete + navigate
    if st.session_state.pop("_delete_success", None):
        st.toast("✓ Compound deleted successfully", icon="✅")

    # Load data using storage_path (most reliable), fallback to entry_id, then compound_name
    data = _load_compound_data(
        compound_name=compound_name,
        entry_id=entry_id,
        storage_path=storage_path,
        internal_prefix=internal_prefix,
    )
    if data is None or "_error" in data:
        error_type = data.get("_error", "unknown") if data else "unknown"
        if error_type == "not_found":
            st.error(
                f"Compound '{compound_name}' not found. It may have been deleted or the storage path is invalid."
            )
        elif error_type == "parse_error":
            st.error(
                f"Could not parse data for '{compound_name}'. The stored data may be corrupted."
            )
        else:
            st.error(f"Server error loading '{compound_name}'. Please try again later.")
        return

    # Quick stats row
    _render_quick_stats(data)

    # Fetch versions for conditional tab (Streamlit requires fixed tab count per render)
    # Re-fetch on first visit to this compound (cache_key changes per entry_id)
    versions = []
    has_versions = False
    cache_key = f"_versions_{entry_id}"
    if entry_id:
        # Invalidate cache if we navigated to a different compound
        prev_key = st.session_state.get("_versions_last_compound")
        if prev_key != entry_id:
            st.session_state.pop(cache_key, None)
            st.session_state["_versions_last_compound"] = entry_id
        if cache_key not in st.session_state:
            api = get_api_client()
            result = api.get_compound_versions(entry_id)
            st.session_state[cache_key] = (
                result.get("versions", []) if result.get("success") else []
            )
        versions = st.session_state[cache_key]
        has_versions = len(versions) > 1  # >1 because list includes self

    # Consume tab query param for deep linking (D-46)
    # Only apply once per compound to avoid overriding user tab clicks
    tab_param = st.query_params.get("tab")
    if tab_param and st.session_state.get("_last_deep_link_id") != entry_id:
        tab_map = {
            "overview": 0,
            "imp_score": 1,
            "bioactivity": 2,
            "efficiency": 3,
            "report": 4,
            "versions": 5,
        }
        tab_index = tab_map.get(tab_param.lower())
        if tab_index is not None:
            st.session_state["compound_details_tab"] = tab_index
        st.session_state["_last_deep_link_id"] = entry_id

    # Main content with tabs
    # After navigating from Versions tab, temporarily hide it so the tab count
    # changes -- this resets Streamlit's internal tab index back to 0 (Overview).
    show_versions_tab = has_versions and not st.session_state.pop(
        "_versions_nav_reset", False
    )

    tab_labels = [
        "📊 Overview",
        "📈 Visualizations",
        "🧬 Molecules",
        "📋 Data",
        "📄 Report",
    ]
    if show_versions_tab:
        tab_labels.append(f"🔀 Versions ({len(versions) - 1})")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_overview_tab(data)

    with tabs[1]:
        _render_visualizations_tab(data)

    with tabs[2]:
        _render_structures_tab(data)

    with tabs[3]:
        _render_data_tab(data)

    with tabs[4]:
        _render_report_tab(data)

    if show_versions_tab:
        with tabs[5]:
            _render_versions_tab(versions, entry_id)


def _render_quick_stats(data: dict[str, Any]) -> None:
    """Render compact stats bar."""
    df = data.get("results")
    summary = data.get("summary", {})

    cols = st.columns(5)

    similar = summary.get("similar_count", 0)
    activities = summary.get("total_activities", len(df) if df is not None else 0)
    qed = summary.get("qed", 0)

    # Count unique IMP compounds (not activity rows)
    imp_count = 0
    if (
        df is not None
        and "Is_IMP_Candidate" in df.columns
        and "ChEMBL_ID" in df.columns
    ):
        imp_count = df[df["Is_IMP_Candidate"]]["ChEMBL_ID"].nunique()
    elif summary.get("has_imp_candidates", False):
        imp_count = summary.get("imp_candidates", 0)

    with cols[0]:
        total_similar = summary.get("total_similar", 0)
        compounds_with_data = summary.get("compounds_with_data", similar)
        if total_similar > 0 and total_similar > compounds_with_data:
            st.metric(
                "Similar Compounds",
                total_similar,
                help=f"{compounds_with_data} with activity, {total_similar - compounds_with_data} without",
            )
        else:
            st.metric("Similar Compounds", similar)
    with cols[1]:
        st.metric("Activities", activities)
    with cols[2]:
        st.metric("QED", f"{qed:.2f}" if qed else "N/A")
    with cols[3]:
        imp_score = None
        if df is not None and "IMP_Final_Score" in df.columns:
            imp_score = df["IMP_Final_Score"].max()
        _score_int = format_imp_score(imp_score) if pd.notna(imp_score) else None
        # Compact overview surface: use st.metric for visual parity with sibling
        # columns (Similar Compounds, Activities, QED). The bar-rich score-card
        # stack lives on the IMP Score tab; UI-SPEC anti-req #9's no-st.metric
        # rule applied when the bar was inline here. With the bar removed for
        # the overview row, st.metric is the consistent widget.
        st.metric(
            "IMP Score",
            _score_int if _score_int is not None else "N/A",
            help="IMP Score on the 0–100 integer scale. Full visualization on the IMP Score tab.",
        )
    with cols[4]:
        st.metric(
            "IMP candidates",
            imp_count,
            help="Unique compounds flagged Is_IMP_Candidate (boolean column from the scoring pipeline).",
        )


# =============================================================================
# OVERVIEW TAB - Using sub-tabs for organization
# =============================================================================


def _render_overview_tab(data: dict[str, Any]) -> None:
    """Overview with sub-tabs for different analysis sections."""
    df = data.get("results")
    summary = data.get("summary", {})
    compound_name = data.get("compound_name", "")
    entry_id = data.get("entry_id")
    storage_path = data.get("storage_path")

    # Sub-tabs for overview sections
    sub_tabs = st.tabs(
        [
            "🧪 Compound",
            "🔢 Properties",
            "📈 Activity",
            "🎯 Efficiency",
            "🔬 PDB Evidence",
            "⚠️ Assay Interference",
            "🔍 IMP Score",
            "💊 Drug Indications",
        ]
    )

    # Compound Info + Classification (combined)
    with sub_tabs[0]:
        _render_compound_info(data, df, summary)
        st.markdown("---")
        _render_classification_compact(df)

    # Computed Properties
    with sub_tabs[1]:
        _render_computed_properties(df)

    # Activity Analysis
    with sub_tabs[2]:
        _render_activity_analysis(df)

    # Efficiency Metrics
    with sub_tabs[3]:
        _render_efficiency_analysis(df)

    # PDB Evidence
    with sub_tabs[4]:
        _render_pdb_evidence(
            compound_name, df, entry_id=entry_id, storage_path=storage_path
        )

    # PAINS/Assay Interference (separated)
    with sub_tabs[5]:
        _render_pains_analysis(df)

    # IMP Score Analysis (without PAINS)
    with sub_tabs[6]:
        _render_imp_score_analysis(df)

    # Drug Indications
    with sub_tabs[7]:
        _render_drug_indications(data)


def _render_compound_info(
    data: dict[str, Any], df: pd.DataFrame, summary: dict
) -> None:
    """Compound information section."""
    # Get unique IDs early for use in both columns
    unique_ids = []
    unique_count = 0
    if df is not None and "ChEMBL_ID" in df.columns:
        unique_ids = [str(x) for x in df["ChEMBL_ID"].dropna().unique().tolist()]
        unique_count = len(unique_ids)

    col1, col2 = st.columns([1, 2])

    with col1:
        smiles = data.get("smiles", "")
        if smiles:
            render_2d_structure(smiles, size=(380, 300))
            _, _ec = st.columns([1, 2])
            with _ec:
                if st.button("⛶ Expand Structure", key="expand_overview_struct"):
                    _show_expanded_structure(
                        smiles, label=data.get("compound_name", "")
                    )

        # Metadata as styled cards
        author_name = data.get("author_name", "")
        proc_date = summary.get("processing_date", "")
        sim_threshold = summary.get("similarity_threshold", 90)

        meta_items = [
            ("Similarity", f"{sim_threshold}%", "#2563eb"),
        ]
        if proc_date:
            meta_items.append(("Processed", html.escape(str(proc_date)), "#7c3aed"))
        if author_name and author_name != "N/A":
            meta_items.append(("Author", html.escape(str(author_name)), "#16a34a"))

        cards_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;">'
        for label, value, color in meta_items:
            cards_html += (
                f'<div style="padding:6px 14px;border-radius:8px;border-left:3px solid {color};'
                f'background:rgba(128,128,128,0.08);">'
                f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:0.5px;opacity:0.6;">{label}</div>'
                f'<div style="font-size:15px;font-weight:600;">{value}</div>'
                f"</div>"
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        # ChEMBL IDs — show first 5, rest in expander
        if unique_count > 0:
            show_count = min(5, unique_count)
            pills_html = "".join(
                f'<a href="https://www.ebi.ac.uk/chembl/explore/compound/{html.escape(cid)}" '
                f'target="_blank" style="display:inline-block;padding:3px 8px;margin:2px;'
                f"border-radius:4px;font-size:12px;font-weight:600;color:#3b82f6;"
                f'border:1px solid rgba(59,130,246,0.3);text-decoration:none;">'
                f"{html.escape(cid)}</a>"
                for cid in unique_ids[:show_count]
            )
            st.markdown(
                f'<div style="margin-top:4px;">'
                f'<span style="font-size:13px;font-weight:600;opacity:0.7;">ChEMBL IDs ({unique_count}): </span>'
                f"{pills_html}</div>",
                unsafe_allow_html=True,
            )
            if unique_count > show_count:
                with st.expander(f"Show all {unique_count} ChEMBL IDs"):
                    # Render in columns of 4
                    for row_start in range(0, unique_count, 4):
                        row_ids = unique_ids[row_start : row_start + 4]
                        row_cols = st.columns(4)
                        for j, cid in enumerate(row_ids):
                            row_cols[j].markdown(
                                f'<a href="https://www.ebi.ac.uk/chembl/explore/compound/{html.escape(cid)}" '
                                f'target="_blank" style="color:#3b82f6;text-decoration:none;font-size:13px;'
                                f'font-weight:600;">{html.escape(cid)}</a>',
                                unsafe_allow_html=True,
                            )

    with col2:
        # SMILES + identifiers
        smiles_value = data.get("smiles", "")
        st.markdown("**Query SMILES**")
        st.code(smiles_value if smiles_value else "N/A", language=None)

        id_col1, id_col2 = st.columns(2)
        inchikey = data.get("inchikey")
        inchi = data.get("inchi")
        with id_col1:
            if inchikey:
                st.markdown("**InChIKey**")
                st.code(inchikey, language=None)
        with id_col2:
            formula = data.get("mol_formula")
            if formula:
                st.markdown("**Molecular Formula**")
                st.markdown(
                    f'<div style="font-size:20px;font-weight:700;font-family:monospace;'
                    f'color:#16a34a;padding:8px 0;">{html.escape(formula)}</div>',
                    unsafe_allow_html=True,
                )

        # InChI (full width — always long)
        inchi = data.get("inchi")
        if inchi:
            st.markdown("**InChI**")
            st.code(inchi, language=None)

        # Activity types as colored pills
        activity_types = summary.get("activity_types", "")
        if isinstance(activity_types, str):
            types = [t.strip() for t in activity_types.split(",") if t.strip()]
        elif isinstance(activity_types, list):
            types = [t for t in activity_types if t and t.strip()]
        else:
            types = []
        if types:
            type_colors = [
                "#2563eb",
                "#16a34a",
                "#d97706",
                "#dc2626",
                "#7c3aed",
                "#0e7490",
                "#ea580c",
            ]
            pills = "".join(
                f'<span style="display:inline-block;padding:4px 12px;margin:2px;border-radius:12px;'
                f"font-size:13px;font-weight:700;background:{type_colors[i % len(type_colors)]}18;"
                f'color:{type_colors[i % len(type_colors)]};border:1.5px solid {type_colors[i % len(type_colors)]}55;">'
                f"{html.escape(t)}</span>"
                for i, t in enumerate(types[:7])
            )
            st.markdown(
                f'<div style="margin-top:8px;margin-bottom:16px;">'
                f'<span style="font-size:13px;font-weight:600;opacity:0.7;">Activity Types </span>'
                f"{pills}</div>",
                unsafe_allow_html=True,
            )

        # View Compound Details - shows ALL similar compounds (not just those with bioactivity)
        if unique_count > 0 and df is not None:
            with st.expander("📋 View Compound Details", expanded=False):
                all_similar_df = data.get("all_similar")

                if all_similar_df is not None and not all_similar_df.empty:
                    # Build activity count from main results
                    activity_counts = (
                        df.groupby("ChEMBL_ID").size().to_dict()
                        if "ChEMBL_ID" in df.columns
                        else {}
                    )
                    display = all_similar_df[
                        ["ChEMBL_ID", "Molecule_Name", "Similarity"]
                    ].copy()
                    display["Biological_Activity"] = (
                        display["ChEMBL_ID"].map(activity_counts).fillna(0).astype(int)
                    )
                else:
                    # Fallback for older compounds without all_similar catalog
                    id_cols = ["ChEMBL_ID"]
                    if "Molecule_Name" in df.columns:
                        id_cols.append("Molecule_Name")
                    if "Similarity" in df.columns:
                        id_cols.append("Similarity")
                    display = (
                        df[id_cols].drop_duplicates("ChEMBL_ID").reset_index(drop=True)
                    )
                    # Add activity count
                    if "ChEMBL_ID" in df.columns:
                        activity_counts = df.groupby("ChEMBL_ID").size().to_dict()
                        display["Biological_Activity"] = (
                            display["ChEMBL_ID"]
                            .map(activity_counts)
                            .fillna(0)
                            .astype(int)
                        )
                    if "Similarity" in display.columns:
                        display = display.sort_values(
                            "Similarity", ascending=False
                        ).reset_index(drop=True)

                if "Molecule_Name" in display.columns:
                    display["Molecule_Name"] = display["Molecule_Name"].apply(
                        lambda x: x if isinstance(x, str) else ""
                    )

                # Make ChEMBL IDs clickable
                display["ChEMBL_ID"] = display["ChEMBL_ID"].apply(
                    lambda x: (
                        f"https://www.ebi.ac.uk/chembl/explore/compound/{x}"
                        if x
                        else ""
                    )
                )
                col_config = {
                    "ChEMBL_ID": st.column_config.LinkColumn(
                        "ChEMBL_ID",
                        display_text=r"https://www\.ebi\.ac\.uk/chembl/explore/compound/(.*)",
                    )
                }

                st.dataframe(
                    display,
                    width="stretch",
                    hide_index=True,
                    height=min(300, len(display) * 35 + 40),
                    column_config=col_config,
                )


def _render_classification_compact(df: pd.DataFrame) -> None:
    """Compact classification display - ClassyFire + NPClassifier side by side."""
    if df is None:
        return

    # ClassyFire columns
    classyfire_cols = ["Kingdom", "Superclass", "Class", "Subclass"]
    # NPClassifier columns
    npclass_cols = ["NP_Pathway", "NP_Superclass", "NP_Class"]

    cf_avail = [c for c in classyfire_cols if c in df.columns]
    np_avail = [c for c in npclass_cols if c in df.columns]

    if not cf_avail and not np_avail:
        return

    st.markdown("**Chemical Classification**")

    col1, col2 = st.columns(2)

    with col1:
        if cf_avail:
            # Breadcrumb taxonomy path
            breadcrumb_parts = []
            for col_name in cf_avail:
                val_counts = df[col_name].value_counts()
                if len(val_counts) > 0:
                    top_val = str(val_counts.index[0])
                    if top_val and top_val != "nan":
                        breadcrumb_parts.append((col_name, top_val))
            if breadcrumb_parts:
                crumbs_html = ' <span style="color:#2563eb;font-size:16px;">→</span> '.join(
                    f'<span style="display:inline-block;padding:4px 10px;border-radius:6px;'
                    f"font-size:13px;font-weight:600;background:rgba(37,99,235,{0.08 + i * 0.05});"
                    f'border:1.5px solid rgba(37,99,235,{0.2 + i * 0.08});">'
                    f'<span style="font-size:10px;opacity:0.6;text-transform:uppercase;">{html.escape(name)}</span><br>'
                    f"{html.escape(val)}</span>"
                    for i, (name, val) in enumerate(breadcrumb_parts)
                )
                st.markdown(
                    f'<div style="margin:8px 0;">'
                    f'<span style="font-size:14px;font-weight:600;">🧬 ClassyFire</span><br>'
                    f'<div style="margin-top:6px;display:flex;align-items:center;flex-wrap:wrap;gap:4px;">'
                    f"{crumbs_html}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("ClassyFire: Not available")

    with col2:
        if np_avail:
            breadcrumb_parts = []
            for col_name in np_avail:
                val_counts = df[col_name].value_counts()
                if len(val_counts) > 0:
                    top_val = str(val_counts.index[0])
                    if top_val and top_val != "nan":
                        label = col_name.replace("NP_", "")
                        breadcrumb_parts.append((label, top_val))
            if breadcrumb_parts:
                crumbs_html = ' <span style="color:#16a34a;font-size:16px;">→</span> '.join(
                    f'<span style="display:inline-block;padding:4px 10px;border-radius:6px;'
                    f"font-size:13px;font-weight:600;background:rgba(22,163,74,{0.08 + i * 0.05});"
                    f'border:1.5px solid rgba(22,163,74,{0.2 + i * 0.08});">'
                    f'<span style="font-size:10px;opacity:0.6;text-transform:uppercase;">{html.escape(name)}</span><br>'
                    f"{html.escape(val)}</span>"
                    for i, (name, val) in enumerate(breadcrumb_parts)
                )
                st.markdown(
                    f'<div style="margin:8px 0;">'
                    f'<span style="font-size:14px;font-weight:600;">🌿 NPClassifier</span><br>'
                    f'<div style="margin-top:6px;display:flex;align-items:center;flex-wrap:wrap;gap:4px;">'
                    f"{crumbs_html}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("NPClassifier: Not available")

    # Expandable full table
    with st.expander("📋 View Full Classification Table"):
        id_cols = ["ChEMBL_ID"] if "ChEMBL_ID" in df.columns else []
        display_cols = id_cols + cf_avail + np_avail
        display_cols = [c for c in display_cols if c in df.columns]

        if display_cols:
            class_df = df[display_cols].drop_duplicates()
            st.dataframe(class_df, width="stretch", hide_index=True, height=250)


def _render_computed_properties(df: pd.DataFrame) -> None:
    """Computed molecular properties display."""
    if df is None:
        st.info("No data available")
        return

    st.markdown("**Computed Molecular Properties**")

    # Get unique compounds for property display
    unique_df = df.drop_duplicates("ChEMBL_ID") if "ChEMBL_ID" in df.columns else df
    num_compounds = len(unique_df)
    st.caption(
        f"Properties for {num_compounds} similar compound{'s' if num_compounds != 1 else ''}"
    )

    # Identify all numeric property columns (exclude metadata columns)
    exclude_cols = {
        "ChEMBL_ID",
        "Molecule_Name",
        "SMILES",
        "Canonical_SMILES",
        "Standard_SMILES",
        "InChI",
        "InChI_Key",
        "Target",
        "Assay_ID",
        "Assay_Description",
        "Activity_Type",
        "Activity_Value",
        "Activity_Units",
        "Activity_Relation",
        "Document_ID",
        "Document_Year",
        "Activity_Comment",
        "Pchembl_Value",
        "Assay_Type",
        "Data_Quality",
        "Kingdom",
        "Superclass",
        "Class",
        "Subclass",
        "Parent_Level",
        "NP_Pathway",
        "NP_Superclass",
        "NP_Class",
        "Index",
        "_row_index",
        "Query_SMILES",
        "Similarity",
        "IMP_Candidate",
        "IMP_Reason",
        "PAINS_Alert",
        "Aggregator_Alert",
        "Redox_Alert",
        "Fluorescent_Alert",
        "IMP_Final_Score",
        "IMP_Grade",
        "O_Score",
        "Q_Score",
        "P_Score",
        "L_Score",
        "A_Score",
    }

    # Get all numeric columns that aren't excluded
    numeric_cols = []
    for col in unique_df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(unique_df[col]):
            # Check if column has any non-null values
            if unique_df[col].notna().any():
                numeric_cols.append(col)

    if not numeric_cols:
        st.info("No computed properties available in the dataset")
        return

    # Property category hints
    physchem_hints = [
        "MW",
        "Weight",
        "LogP",
        "ALogP",
        "TPSA",
        "PSA",
        "HBD",
        "HBA",
        "Rotatable",
        "Donors",
        "Acceptors",
        "CSP3",
        "Rings",
        "Aromatic",
        "Heavy",
        "Hetero",
        "NumAtoms",
        "NumBonds",
        "MolLogP",
        "MolMR",
        "NPOL",
    ]
    druglike_hints = [
        "QED",
        "Lipinski",
        "Ro5",
        "RO5",
        "Veber",
        "Ghose",
        "Muegge",
        "Egan",
        "Brenk",
        "NP_Likeness",
    ]

    # Categorize properties
    physchem_cols = [
        c for c in numeric_cols if any(h.lower() in c.lower() for h in physchem_hints)
    ]
    druglike_cols = [
        c for c in numeric_cols if any(h.lower() in c.lower() for h in druglike_hints)
    ]
    other_cols = [
        c for c in numeric_cols if c not in physchem_cols and c not in druglike_cols
    ]

    # View mode toggle (scoped to entry_id to prevent stale state across compounds)
    _eid = SessionState.get("selected_compound_entry_id", "")
    view_mode = st.radio(
        "View mode",
        ["Summary Statistics", "Individual Compounds"],
        horizontal=True,
        key=f"prop_view_mode_{_eid}",
    )

    if view_mode == "Summary Statistics":
        property_display = {
            "Molecular_Weight": ("Molecular Weight", "g/mol", 500, True),
            "MolLogP": ("LogP", "", 5, True),
            "LogP": ("LogP", "", 5, True),
            "HBD": ("H-Bond Donors", "", 5, True),
            "HBA": ("H-Bond Acceptors", "", 10, True),
            "TPSA": ("Topological PSA", "Å²", 140, True),
            "Heavy_Atoms": ("Heavy Atoms", "", 50, False),
            "Rotatable_Bonds": ("Rotatable Bonds", "", 10, True),
            "Aromatic_Rings": ("Aromatic Rings", "", 5, False),
            "NPOL": ("Polar Atoms", "", 20, False),
        }

        # Two columns: property table LEFT, drug-likeness RIGHT
        prop_col, dl_col = st.columns([3, 2])

        with prop_col:
            st.markdown("**Physicochemical Properties**")
            table_html = (
                '<table style="width:100%;border-collapse:collapse;font-size:15px;margin:8px 0;'
                'table-layout:auto;">'
                '<thead><tr style="border-bottom:2px solid rgba(128,128,128,0.3);">'
                '<th style="text-align:left;padding:7px 10px;font-weight:600;opacity:0.7;">Property</th>'
                '<th style="text-align:right;padding:7px 10px;font-weight:600;opacity:0.7;white-space:nowrap;">Mean</th>'
                '<th style="text-align:right;padding:7px 10px;font-weight:600;opacity:0.7;white-space:nowrap;">Range</th>'
                '<th style="text-align:center;padding:7px 4px;width:16px;"></th>'
                "</tr></thead><tbody>"
            )
            row_idx = 0
            for col_name in physchem_cols:
                vals = unique_df[col_name].dropna()
                if len(vals) == 0:
                    continue
                mean_val = vals.mean()
                min_val = vals.min()
                max_val = vals.max()
                info = property_display.get(col_name)
                if info:
                    disp_name, unit, rule_max, has_rule = info
                else:
                    disp_name, unit, rule_max, has_rule = col_name, "", 100, False

                if has_rule:
                    dot = "●" if mean_val <= rule_max else "●"
                    dot_color = "#22c55e" if mean_val <= rule_max else "#ef4444"
                else:
                    dot, dot_color = "●", "#3b82f6"

                unit_str = f" {unit}" if unit else ""
                range_str = (
                    f"{min_val:.2f}–{max_val:.2f}" if min_val != max_val else "—"
                )
                bg = "rgba(128,128,128,0.04)" if row_idx % 2 == 0 else "transparent"

                table_html += (
                    f'<tr style="background:{bg};border-bottom:1px solid rgba(128,128,128,0.08);">'
                    f'<td style="padding:6px 10px;font-weight:500;">{html.escape(disp_name)}'
                    f'<span style="font-size:11px;opacity:0.4;">{html.escape(unit_str)}</span></td>'
                    f'<td style="padding:6px 10px;text-align:right;font-weight:700;font-family:monospace;white-space:nowrap;">'
                    f"{mean_val:.2f}</td>"
                    f'<td style="padding:6px 10px;text-align:right;opacity:0.5;font-family:monospace;font-size:12px;white-space:nowrap;">'
                    f"{range_str}</td>"
                    f'<td style="padding:6px 4px;text-align:center;color:{dot_color};">{dot}</td>'
                    f"</tr>"
                )
                row_idx += 1
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)

        with dl_col:
            st.markdown("**Drug-likeness**")
            # QED first, then others, Lipinski last — with help tooltips
            dl_metrics = [
                (
                    "QED",
                    "QED Score",
                    "{:.3f}",
                    "Quantitative Estimate of Drug-likeness (0-1, higher is better)",
                    lambda v: (
                        ("#22c55e", "🟢")
                        if v >= 0.5
                        else ("#f59e0b", "🟡")
                        if v >= 0.3
                        else ("#ef4444", "🔴")
                    ),
                ),
                (
                    "RO5_Violations",
                    "RO5 Violations",
                    "{:.1f}",
                    "Lipinski Rule of 5 violations (0-4, lower is better)",
                    lambda v: (
                        ("#22c55e", "🟢")
                        if v <= 1
                        else ("#f59e0b", "🟡")
                        if v <= 2
                        else ("#ef4444", "🔴")
                    ),
                ),
                (
                    "QED_Multiplier",
                    "QED Multiplier",
                    "{:.3f}",
                    "IMP Score QED multiplier (0.75 + 0.25 × QED)",
                    lambda v: (
                        ("#22c55e", "🟢")
                        if v >= 0.85
                        else ("#f59e0b", "🟡")
                        if v >= 0.75
                        else ("#ef4444", "🔴")
                    ),
                ),
                (
                    "Aromatic_Rings",
                    "Aromatic Rings",
                    "{:.1f}",
                    "Number of aromatic ring systems (≤3 preferred)",
                    lambda v: (
                        ("#22c55e", "🟢")
                        if v <= 3
                        else ("#f59e0b", "🟡")
                        if v <= 4
                        else ("#ef4444", "🔴")
                    ),
                ),
                (
                    "QED_Impact",
                    "QED Impact",
                    "{:.3f}",
                    "QED penalty on IMP score (0 = best, negative = penalty)",
                    lambda v: (
                        ("#22c55e", "🟢")
                        if v >= -0.1
                        else ("#f59e0b", "🟡")
                        if v >= -0.2
                        else ("#ef4444", "🔴")
                    ),
                ),
                (
                    "NP_Likeness_Score",
                    "NP Likeness",
                    "{:.2f}",
                    "Natural Product Likeness (-5 to +5, positive = more NP-like)",
                    lambda v: (
                        ("#22c55e", "🟢")
                        if v > 0
                        else ("#f59e0b", "🟡")
                        if v > -1
                        else ("#94a3b8", "⚪")
                    ),
                ),
            ]

            # Use Streamlit metrics with help tooltips (native ? icon)
            q1, q2 = st.columns(2)
            for i, (col_name, label, fmt, tooltip, color_fn) in enumerate(dl_metrics):
                if col_name not in unique_df.columns:
                    continue
                vals = unique_df[col_name].dropna()
                if len(vals) == 0:
                    continue
                mean_val = vals.mean()
                color, emoji = color_fn(mean_val)
                formatted = fmt.format(mean_val)
                target_col = q1 if i % 2 == 0 else q2
                target_col.metric(f"{emoji} {label}", formatted, help=tooltip)

            # Lipinski RO5 last
            logp_col_name = (
                "LogP"
                if "LogP" in unique_df.columns
                else "MolLogP"
                if "MolLogP" in unique_df.columns
                else None
            )
            lip_checks = [
                ("Molecular_Weight", lambda v: v <= 500),
                ("HBD", lambda v: v <= 5),
                ("HBA", lambda v: v <= 10),
                ("TPSA", lambda v: v <= 140),
            ]
            if logp_col_name:
                lip_checks.append((logp_col_name, lambda v: v <= 5))
            lp = sum(
                1
                for c, fn in lip_checks
                if c in unique_df.columns and fn(unique_df[c].dropna().mean())
            )
            lt = sum(1 for c, _ in lip_checks if c in unique_df.columns)
            le = "🟢" if lp == lt else "🟡" if lp >= lt - 1 else "🔴"
            st.metric(
                f"{le} Lipinski RO5",
                f"{lp}/{lt} passed",
                help="Lipinski Rule of 5 — oral bioavailability filter (MW≤500, LogP≤5, HBD≤5, HBA≤10)",
            )

        # Key visualizations
        st.markdown("---")
        st.markdown("**📈 Key Property Visualizations**")

        # Determine which LogP column to use
        logp_col = None
        if "LogP" in unique_df.columns and unique_df["LogP"].notna().any():
            logp_col = "LogP"
        elif "MolLogP" in unique_df.columns and unique_df["MolLogP"].notna().any():
            logp_col = "MolLogP"

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # MW vs LogP scatter plot (Lipinski space)
            if "Molecular_Weight" in unique_df.columns and logp_col:
                plot_data = unique_df[["Molecular_Weight", logp_col]].dropna()
                if len(plot_data) > 0:
                    hover_cols = [
                        c
                        for c in ["ChEMBL_ID", "Molecule_Name"]
                        if c in unique_df.columns
                    ]
                    mw_plot_df = unique_df.dropna(
                        subset=["Molecular_Weight", logp_col]
                    ).copy()
                    cd_cols = [
                        c
                        for c in ["SMILES", "Molecule_Name", "ChEMBL_ID"]
                        if c in mw_plot_df.columns
                    ]
                    fig = px.scatter(
                        mw_plot_df,
                        x="Molecular_Weight",
                        y=logp_col,
                        color="QED"
                        if "QED" in mw_plot_df.columns
                        and mw_plot_df["QED"].notna().any()
                        else None,
                        hover_data=hover_cols if hover_cols else None,
                        custom_data=cd_cols if cd_cols else None,
                        title="MW vs LogP",
                        color_continuous_scale="RdYlGn",
                    )
                    fig.update_layout(
                        title=dict(
                            text="MW vs LogP",
                            subtitle=dict(
                                text="Lipinski Rule of 5 space — dashed lines = boundaries"
                            ),
                        )
                    )
                    # Add Lipinski rule boundaries
                    fig.add_hline(
                        y=5,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="LogP ≤ 5",
                    )
                    fig.add_vline(
                        x=500,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="MW ≤ 500",
                    )
                    fig.update_layout(height=300, margin=dict(t=55, b=30, l=30, r=30))
                    apply_impulator_theme(fig)
                    st.plotly_chart(fig, width="stretch", key="mw_logp_scatter")
                    _maybe_embed_structure_viewer(
                        "mw_logp_scatter",
                        mw_plot_df,
                        x_col="Molecular_Weight",
                        y_col=logp_col,
                    )
                else:
                    st.caption("No MW/LogP data available for visualization")
            else:
                st.caption(
                    "MW vs LogP plot requires LogP data (reprocess compounds to generate)"
                )

        with viz_col2:
            # TPSA vs HBD+HBA — absorption/permeability space
            has_hbd = "HBD" in unique_df.columns and unique_df["HBD"].notna().any()
            has_hba = "HBA" in unique_df.columns and unique_df["HBA"].notna().any()
            has_tpsa = "TPSA" in unique_df.columns and unique_df["TPSA"].notna().any()

            if has_tpsa and has_hbd and has_hba:
                abs_plot_df = unique_df.copy()
                abs_plot_df["HBD+HBA"] = abs_plot_df["HBD"].fillna(0) + abs_plot_df[
                    "HBA"
                ].fillna(0)
                abs_plot_df = abs_plot_df.dropna(subset=["TPSA"])
                cd_cols = [
                    c
                    for c in ["SMILES", "Molecule_Name", "ChEMBL_ID"]
                    if c in abs_plot_df.columns
                ]
                fig = px.scatter(
                    abs_plot_df,
                    x="TPSA",
                    y="HBD+HBA",
                    color="QED"
                    if "QED" in abs_plot_df.columns and abs_plot_df["QED"].notna().any()
                    else None,
                    hover_data=["ChEMBL_ID"]
                    if "ChEMBL_ID" in abs_plot_df.columns
                    else None,
                    custom_data=cd_cols if cd_cols else None,
                    color_continuous_scale="RdYlGn",
                )
                # Veber rule boundaries
                fig.add_hline(
                    y=10,
                    line_dash="dash",
                    line_color="#ef4444",
                    line_width=1.5,
                    annotation_text="HBD+HBA ≤ 10",
                    annotation_font_color="#ef4444",
                )
                fig.add_vline(
                    x=140,
                    line_dash="dash",
                    line_color="#ef4444",
                    line_width=1.5,
                    annotation_text="TPSA ≤ 140 Å²",
                    annotation_font_color="#ef4444",
                )
                # Good oral absorption zone
                fig.add_vrect(x0=0, x1=140, fillcolor="#22c55e", opacity=0.03)
                fig.update_layout(
                    title=dict(
                        text="Absorption Space",
                        subtitle=dict(
                            text="TPSA vs H-bond count — oral bioavailability zone"
                        ),
                    ),
                    height=300,
                    margin=dict(t=55, b=30, l=30, r=30),
                    xaxis_title="TPSA (Å²)",
                    yaxis_title="HBD + HBA",
                )
                apply_impulator_theme(fig)
                st.plotly_chart(fig, width="stretch", key="absorption_scatter")
                _maybe_embed_structure_viewer(
                    "absorption_scatter", abs_plot_df, x_col="TPSA", y_col="HBD+HBA"
                )
            elif has_tpsa:
                fig = px.histogram(
                    unique_df["TPSA"].dropna(),
                    nbins=25,
                    color_discrete_sequence=["#3b82f6"],
                )
                fig.add_vline(
                    x=140,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="TPSA ≤ 140",
                    annotation_font_color="#ef4444",
                )
                fig.update_layout(
                    title="TPSA Distribution",
                    height=300,
                    margin=dict(t=40, b=30, l=30, r=30),
                )
                apply_impulator_theme(fig)
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Absorption space plot requires TPSA + HBD/HBA data")

        # Second row: Lipinski Radar + 10xPSA/MW vs NPOL/NHA
        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            # Lipinski spider/radar chart — normalized to rule limits
            radar_props = [
                ("Molecular_Weight", "MW", 500),
                ("HBD", "HBD", 5),
                ("HBA", "HBA", 10),
                ("TPSA", "TPSA", 140),
                ("Rotatable_Bonds", "Rot. Bonds", 10),
            ]
            if logp_col:
                radar_props.insert(1, (logp_col, "LogP", 5))

            radar_names = []
            radar_vals = []
            for col_name, label, limit in radar_props:
                if col_name in unique_df.columns and unique_df[col_name].notna().any():
                    val = unique_df[col_name].dropna().mean()
                    radar_names.append(label)
                    radar_vals.append(min(val / limit, 1.5))

            if len(radar_names) >= 3:
                fig = go.Figure()
                # Rule limit circle (1.0 = at limit)
                fig.add_trace(
                    go.Scatterpolar(
                        r=[1.0] * len(radar_names) + [1.0],
                        theta=radar_names + [radar_names[0]],
                        line=dict(color="rgba(239,68,68,0.5)", width=2, dash="dot"),
                        fill="toself",
                        fillcolor="rgba(239,68,68,0.04)",
                        name="Lipinski Limit",
                    )
                )
                # Compound values
                fig.add_trace(
                    go.Scatterpolar(
                        r=radar_vals + [radar_vals[0]],
                        theta=radar_names + [radar_names[0]],
                        fill="toself",
                        fillcolor="rgba(99,102,241,0.15)",
                        line=dict(color="#6366f1", width=2.5),
                        marker=dict(
                            size=7,
                            color=[
                                "#22c55e" if v <= 1.0 else "#ef4444" for v in radar_vals
                            ]
                            + ["#6366f1"],
                        ),
                        name="Mean Values",
                        text=[
                            f"{radar_names[i]}: {radar_vals[i] * 100:.0f}% of limit"
                            for i in range(len(radar_names))
                        ]
                        + [""],
                        hoverinfo="text",
                    )
                )
                fig.update_layout(
                    title=dict(
                        text="Lipinski Profile",
                        subtitle=dict(
                            text="Normalized to rule limits — inside circle = pass"
                        ),
                    ),
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1.5],
                            tickvals=[0.5, 1.0, 1.5],
                            ticktext=["50%", "Limit", "150%"],
                            tickfont=dict(size=10),
                            gridcolor="rgba(128,128,128,0.2)",
                        ),
                        angularaxis=dict(tickfont=dict(size=13)),
                    ),
                    height=320,
                    margin=dict(t=55, b=40, l=50, r=50),
                    legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
                )
                apply_impulator_theme(fig)
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Lipinski profile requires ≥3 properties")

        with viz_col4:
            # 10xPSA_MW vs NPOLoNHA scatter plot (replaces QED distribution)
            has_psa_mw = (
                "10xPSA_MW" in unique_df.columns
                and unique_df["10xPSA_MW"].notna().any()
            )
            has_npol_nha = (
                "NPOLoNHA" in unique_df.columns and unique_df["NPOLoNHA"].notna().any()
            )

            if has_psa_mw and has_npol_nha:
                plot_df = unique_df.dropna(subset=["10xPSA_MW", "NPOLoNHA"])
                # Need >=2 points AND variance in x values for regression
                x_vals = (
                    plot_df["NPOLoNHA"].values if len(plot_df) >= 2 else np.array([])
                )
                y_vals = (
                    plot_df["10xPSA_MW"].values if len(plot_df) >= 2 else np.array([])
                )
                can_regress = len(plot_df) >= 2 and len(np.unique(x_vals)) > 1

                if can_regress:
                    # Calculate R² statistics
                    slope, intercept, r_value, p_value, std_err = (
                        scipy_stats.linregress(x_vals, y_vals)
                    )
                    r_squared = r_value**2
                    title = f"10×PSA/MW vs NPOL/NHA (R²={r_squared:.3f})"
                    show_trendline = True
                    stats_caption = (
                        f"R²={r_squared:.4f}, slope={slope:.4f}, p={p_value:.2e}"
                    )
                elif len(plot_df) >= 1:
                    # Can show scatter but not trendline (all x values identical or only 1 point)
                    title = "10×PSA/MW vs NPOL/NHA"
                    show_trendline = False
                    stats_caption = (
                        "Insufficient variance for regression"
                        if len(plot_df) >= 2
                        else ""
                    )
                else:
                    show_trendline = False
                    title = None

                if len(plot_df) >= 1:
                    # Build customdata for structure viewer
                    customdata_cols = None
                    if "SMILES" in plot_df.columns:
                        customdata_cols = ["SMILES"]
                        if "Molecule_Name" in plot_df.columns:
                            customdata_cols.append("Molecule_Name")
                        if "ChEMBL_ID" in plot_df.columns:
                            customdata_cols.append("ChEMBL_ID")

                    fig = px.scatter(
                        plot_df,
                        x="NPOLoNHA",
                        y="10xPSA_MW",
                        color="QED"
                        if "QED" in plot_df.columns and plot_df["QED"].notna().any()
                        else None,
                        hover_data=["ChEMBL_ID", "Molecule_Name"]
                        if all(
                            c in plot_df.columns for c in ["ChEMBL_ID", "Molecule_Name"]
                        )
                        else None,
                        title=title,
                        trendline="ols" if show_trendline else None,
                        color_continuous_scale="Viridis",
                        custom_data=customdata_cols,
                    )
                    fig.update_layout(
                        height=300,
                        margin=dict(t=40, b=30, l=30, r=30),
                        xaxis_title="NPOL/NHA",
                        yaxis_title="10 × PSA/MW",
                    )
                    apply_impulator_theme(fig)
                    st.plotly_chart(fig, width="stretch", key="psa_npol_scatter_chart")
                    if stats_caption:
                        st.caption(stats_caption)

                    _maybe_embed_structure_viewer(
                        "psa_npol_scatter_chart",
                        plot_df,
                        x_col="10xPSA_MW",
                        y_col="NPOLoNHA",
                    )
                else:
                    st.caption("No data points for 10×PSA/MW vs NPOL/NHA plot")
            else:
                st.caption("10×PSA/MW vs NPOL/NHA requires reprocessing compounds")

    else:
        # Individual compounds view
        st.markdown("**Individual Compound Properties**")

        # Compound selector
        if "ChEMBL_ID" in unique_df.columns:
            options = unique_df["ChEMBL_ID"].tolist()
            if "Molecule_Name" in unique_df.columns:
                labels = [
                    f"{cid} — {name}" if pd.notna(name) and name else cid
                    for cid, name in zip(
                        unique_df["ChEMBL_ID"], unique_df["Molecule_Name"]
                    )
                ]
            else:
                labels = options
            selected_idx = st.selectbox(
                "Select compound",
                range(len(labels)),
                format_func=lambda i: labels[i],
                key=f"ind_prop_select_{_eid}",
            )
            row = unique_df.iloc[selected_idx]
        else:
            row = unique_df.iloc[0]

        # Two columns: property table LEFT, drug-likeness RIGHT
        ind_prop_col, ind_dl_col = st.columns([3, 2])

        with ind_dl_col:
            st.markdown("**Drug-likeness**")
            ind_dl_metrics = [
                (
                    "QED",
                    "QED Score",
                    "{:.3f}",
                    "Quantitative Estimate of Drug-likeness (0-1)",
                    lambda v: ("🟢") if v >= 0.5 else ("🟡") if v >= 0.3 else ("🔴"),
                ),
                (
                    "RO5_Violations",
                    "RO5 Violations",
                    "{:.0f}",
                    "Lipinski Rule of 5 violations (0-4)",
                    lambda v: ("🟢") if v <= 1 else ("🟡") if v <= 2 else ("🔴"),
                ),
                (
                    "QED_Multiplier",
                    "QED Multiplier",
                    "{:.3f}",
                    "IMP Score QED multiplier (0.75 + 0.25×QED)",
                    lambda v: ("🟢") if v >= 0.85 else ("🟡") if v >= 0.75 else ("🔴"),
                ),
                (
                    "Aromatic_Rings",
                    "Aromatic Rings",
                    "{:.0f}",
                    "Aromatic ring systems (≤3 preferred)",
                    lambda v: ("🟢") if v <= 3 else ("🟡") if v <= 4 else ("🔴"),
                ),
                (
                    "NP_Likeness_Score",
                    "NP Likeness",
                    "{:.2f}",
                    "Natural Product Likeness (-5 to +5)",
                    lambda v: ("🟢") if v > 0 else ("🟡") if v > -1 else ("⚪"),
                ),
            ]
            iq1, iq2 = st.columns(2)
            idx_m = 0
            for col_name, label, fmt, tooltip, emoji_fn in ind_dl_metrics:
                if col_name not in row.index or pd.isna(row[col_name]):
                    continue
                v = float(row[col_name])
                emoji = emoji_fn(v)
                target = iq1 if idx_m % 2 == 0 else iq2
                target.metric(f"{emoji} {label}", fmt.format(v), help=tooltip)
                idx_m += 1

            # Lipinski last
            logp_col_name = (
                "LogP"
                if "LogP" in row.index
                else "MolLogP"
                if "MolLogP" in row.index
                else None
            )
            lip_checks = [
                ("Molecular_Weight", lambda v: v <= 500),
                ("HBD", lambda v: v <= 5),
                ("HBA", lambda v: v <= 10),
                ("TPSA", lambda v: v <= 140),
            ]
            if logp_col_name:
                lip_checks.append((logp_col_name, lambda v: v <= 5))
            lp = sum(
                1
                for c, fn in lip_checks
                if c in row.index and pd.notna(row[c]) and fn(row[c])
            )
            lt = sum(
                1 for c, _ in lip_checks if c in row.index and pd.notna(row.get(c))
            )
            le = "🟢" if lp == lt else "🟡" if lp >= lt - 1 else "🔴"
            st.metric(
                f"{le} Lipinski RO5",
                f"{lp}/{lt} passed",
                help="MW≤500, LogP≤5, HBD≤5, HBA≤10, TPSA≤140",
            )

        with ind_prop_col:
            st.markdown("**Computed Properties**")
            ind_props = [
                ("Molecular_Weight", "Molecular Weight", "g/mol"),
                ("MolLogP", "LogP", ""),
                ("LogP", "LogP", ""),
                ("TPSA", "Topological PSA", "Å²"),
                ("HBD", "H-Bond Donors", ""),
                ("HBA", "H-Bond Acceptors", ""),
                ("Heavy_Atoms", "Heavy Atoms", ""),
                ("Rotatable_Bonds", "Rotatable Bonds", ""),
                ("Aromatic_Rings", "Aromatic Rings", ""),
                ("NPOL", "Polar Atoms", ""),
                ("PSAoMW", "PSA/MW", ""),
                ("10xPSA_MW", "10×PSA/MW", ""),
                ("NPOLoNHA", "NPOL/NHA", ""),
            ]
            seen = set()
            table_html = (
                '<table style="width:100%;border-collapse:collapse;font-size:15px;margin:8px 0;">'
                '<thead><tr style="border-bottom:2px solid rgba(128,128,128,0.3);">'
                '<th style="text-align:left;padding:6px 10px;font-weight:600;opacity:0.7;">Property</th>'
                '<th style="text-align:right;padding:6px 10px;font-weight:600;opacity:0.7;">Value</th>'
                "</tr></thead><tbody>"
            )
            row_idx = 0
            for col_name, label, unit in ind_props:
                if col_name not in row.index or label in seen:
                    continue
                val = row[col_name]
                if pd.isna(val):
                    continue
                seen.add(label)
                unit_str = f" {unit}" if unit else ""
                bg = "rgba(128,128,128,0.04)" if row_idx % 2 == 0 else "transparent"
                table_html += (
                    f'<tr style="background:{bg};border-bottom:1px solid rgba(128,128,128,0.08);">'
                    f'<td style="padding:6px 10px;font-weight:500;">{html.escape(label)}'
                    f'<span style="font-size:11px;opacity:0.4;">{html.escape(unit_str)}</span></td>'
                    f'<td style="padding:6px 10px;text-align:right;font-weight:700;font-family:monospace;">'
                    f"{val:.2f}</td>"
                    f"</tr>"
                )
                row_idx += 1
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)

        # Full comparison table in expander
        with st.expander("Compare All Compounds"):
            all_props = [c for c in (physchem_cols + druglike_cols + other_cols)]
            display_cols = (
                (["ChEMBL_ID"] if "ChEMBL_ID" in unique_df.columns else [])
                + (["Molecule_Name"] if "Molecule_Name" in unique_df.columns else [])
                + all_props[:15]
            )
            display_df = unique_df[display_cols].copy()
            for c in display_df.columns:
                if pd.api.types.is_numeric_dtype(display_df[c]):
                    display_df[c] = display_df[c].round(2)
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                height=min(400, len(display_df) * 35 + 40),
            )


def _render_activity_analysis(df: pd.DataFrame) -> None:
    """Bioactivity Profile — shows activity evidence and its trustworthiness for IMP detection."""
    if df is None or "Activity_Type" not in df.columns:
        st.info("No activity data available")
        return

    theme = get_plotly_theme()
    has_pact = "pActivity" in df.columns
    has_targets = "Target_Name" in df.columns or "Target_ChEMBL_ID" in df.columns
    has_assay_type = "Assay_Type" in df.columns
    has_quality = "Data_Quality" in df.columns
    target_col = (
        "Target_Name"
        if "Target_Name" in df.columns
        else ("Target_ChEMBL_ID" if "Target_ChEMBL_ID" in df.columns else None)
    )

    # ── Section 1: Summary Cards ──────────────────────────────────────────
    total_records = len(df)
    unique_compounds = df["ChEMBL_ID"].nunique() if "ChEMBL_ID" in df.columns else 0
    unique_targets = df[target_col].nunique() if target_col else 0
    activity_types = df["Activity_Type"].nunique()
    assay_types_crossed = df["Assay_Type"].nunique() if has_assay_type else 0

    cols = st.columns(5 if has_assay_type else 4)
    with cols[0]:
        st.metric(
            "Measurements",
            f"{total_records:,}",
            help="Total bioactivity data points from ChEMBL",
        )
    with cols[1]:
        st.metric(
            "Compounds",
            unique_compounds,
            help="Unique similar compounds with activity data",
        )
    with cols[2]:
        st.metric(
            "Targets",
            unique_targets,
            help="Unique biological targets tested. Many unrelated targets = pan-active (IMP risk)",
        )
    with cols[3]:
        st.metric(
            "Activity Types",
            activity_types,
            help="Distinct assay types (IC50, Ki, Kd, EC50, etc.)",
        )
    if has_assay_type:
        with cols[4]:
            # Color code: >3 assay type categories = promiscuity warning
            at_delta = "⚠️ promiscuous" if assay_types_crossed >= 4 else None
            st.metric(
                "Assay Modalities",
                assay_types_crossed,
                delta=at_delta,
                delta_color="inverse" if assay_types_crossed >= 4 else "off",
                help="Binding, Functional, ADMET, Toxicity, etc. Compounds active across many modalities are more likely artifacts",
            )

    # Data quality badge
    if has_quality:
        flagged = (df["Data_Quality"] == "Flagged").sum()
        if flagged > 0:
            pct = flagged / total_records * 100
            st.caption(
                f"⚠️ {flagged} of {total_records} records ({pct:.0f}%) flagged by ChEMBL quality checks (transcription errors, out-of-range values)"
            )

    st.markdown("---")

    # ── Section 2: Activity Type Breakdown (donut + stats) ────────────────
    counts = df["Activity_Type"].value_counts().reset_index()
    counts.columns = ["Type", "Count"]
    counts["%"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)

    st.markdown("**Bioactivity Distribution**")
    st.caption(
        "Distribution of activity measurements across assay types (IC50, Ki, Kd, EC50, etc.)"
    )

    donut_col, stats_col = st.columns([3, 2])

    with donut_col:
        fig = px.pie(
            counts,
            values="Count",
            names="Type",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            template=theme["template"],
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                title_text="Activity Types",
                bgcolor=theme["legend_bgcolor"],
                bordercolor=theme["legend_bordercolor"],
                borderwidth=1,
            ),
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        apply_impulator_theme(fig)
        st.plotly_chart(fig, width="stretch")

    with stats_col:
        if has_pact:
            stats = (
                df.groupby("Activity_Type")["pActivity"]
                .agg(["count", "mean", "std", "min", "max"])
                .round(2)
            )
            stats.columns = ["Count", "Mean pAct", "Std", "Min", "Max"]
            st.dataframe(stats, width="stretch", height=min(300, len(stats) * 35 + 40))
        else:
            st.dataframe(
                counts,
                width="stretch",
                hide_index=True,
                height=min(300, len(counts) * 35 + 40),
            )

    st.markdown("---")

    # ── Section 3: Potency by Activity Type (box plot) ────────────────────
    if has_pact:
        st.markdown("**Potency Distribution by Assay Type**")
        st.caption(
            "pActivity = −log₁₀(activity in M). Higher values = more potent. Consistent potency across types supports genuine activity."
        )

        plot_df = df.dropna(subset=["pActivity"]).copy()
        if not plot_df.empty:
            # Build customdata for structure viewer
            customdata_cols = []
            if "SMILES" in plot_df.columns:
                customdata_cols.append("SMILES")
                if "Molecule_Name" in plot_df.columns:
                    customdata_cols.append("Molecule_Name")
                if "ChEMBL_ID" in plot_df.columns:
                    customdata_cols.append("ChEMBL_ID")

            fig = px.box(
                plot_df,
                x="Activity_Type",
                y="pActivity",
                color="Activity_Type",
                points="all",
                hover_data=["ChEMBL_ID", "Molecule_Name"]
                if all(c in plot_df.columns for c in ["ChEMBL_ID", "Molecule_Name"])
                else None,
                custom_data=customdata_cols if customdata_cols else None,
            )
            fig.update_layout(
                template=theme["template"],
                height=420,
                margin=dict(t=40, b=60, l=10, r=10),
                showlegend=True,
                xaxis_title="",
                yaxis_title="pActivity",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    bgcolor=theme["legend_bgcolor"],
                    bordercolor=theme["legend_bordercolor"],
                    borderwidth=1,
                ),
            )
            apply_impulator_theme(fig)
            st.plotly_chart(fig, width="stretch", key="activity_potency_box")

            _maybe_embed_structure_viewer(
                "activity_potency_box",
                plot_df,
                x_col="Activity_Type",
                y_col="pActivity",
            )

        st.markdown("---")

    # ── Section 3: Target Selectivity Profile ─────────────────────────────
    if has_targets and has_pact and target_col:
        st.markdown("**Target Selectivity Profile**")
        st.caption(
            "Mean potency per target. Compounds active against many unrelated targets are IMP red flags."
        )

        target_stats = (
            df.groupby(target_col)
            .agg(
                Mean_pActivity=("pActivity", "mean"),
                Count=("pActivity", "count"),
                Compounds=("ChEMBL_ID", "nunique")
                if "ChEMBL_ID" in df.columns
                else ("pActivity", "count"),
            )
            .round(2)
            .sort_values("Mean_pActivity", ascending=True)
        )

        # Show top 12 targets by mean potency
        plot_targets = target_stats.tail(12).copy()

        if not plot_targets.empty:
            fig = go.Figure()

            # Lollipop chart: line + marker
            for i, (tgt, row) in enumerate(plot_targets.iterrows()):
                color = (
                    "#3b82f6"
                    if row["Mean_pActivity"] < 7
                    else "#ef4444"
                    if row["Mean_pActivity"] >= 8
                    else "#f59e0b"
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, row["Mean_pActivity"]],
                        y=[str(tgt), str(tgt)],
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[row["Mean_pActivity"]],
                        y=[str(tgt)],
                        mode="markers+text",
                        marker=dict(
                            size=12, color=color, line=dict(width=1, color="white")
                        ),
                        text=[f" {row['Mean_pActivity']:.1f}"],
                        textposition="middle right",
                        textfont=dict(size=11),
                        showlegend=False,
                        hovertemplate=f"<b>{html.escape(str(tgt))}</b><br>Mean pActivity: {row['Mean_pActivity']:.2f}<br>Records: {row['Count']}<extra></extra>",
                    )
                )

            fig.update_layout(
                template=theme["template"],
                height=max(280, len(plot_targets) * 32 + 60),
                margin=dict(t=10, b=30, l=10, r=60),
                xaxis=dict(title="Mean pActivity", range=[0, None]),
                yaxis=dict(title=""),
            )
            apply_impulator_theme(fig)
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")

    # ── Section 4: Assay Type Coverage ────────────────────────────────────
    if has_assay_type and has_pact:
        st.markdown("**Assay Modality Breakdown**")
        st.caption(
            "How bioactivity evidence is distributed across assay types. Cross-modality activity strengthens (or weakens) confidence."
        )

        at_col1, at_col2 = st.columns([1, 2])

        with at_col1:
            # Assay type counts table
            at_counts = df["Assay_Type"].value_counts().reset_index()
            at_counts.columns = ["Assay Type", "Count"]
            at_counts["%"] = (
                at_counts["Count"] / at_counts["Count"].sum() * 100
            ).round(1)
            st.dataframe(
                at_counts,
                width="stretch",
                hide_index=True,
                height=min(250, len(at_counts) * 35 + 40),
            )

        with at_col2:
            # Donut chart
            _assay_colors = {
                "Binding": "#3b82f6",
                "Functional": "#22c55e",
                "ADMET": "#f59e0b",
                "Toxicity": "#ef4444",
                "Physicochemical": "#8b5cf6",
                "Unclassified": "#6b7280",
                "Unknown": "#9ca3af",
            }
            fig = px.pie(
                at_counts,
                values="Count",
                names="Assay Type",
                hole=0.45,
                color="Assay Type",
                color_discrete_map=_assay_colors,
            )
            fig.update_layout(
                template=theme["template"],
                margin=dict(t=10, b=10, l=10, r=10),
                height=250,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    bgcolor=theme["legend_bgcolor"],
                    bordercolor=theme["legend_bordercolor"],
                    borderwidth=1,
                ),
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            apply_impulator_theme(fig)
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")

    # ── Section 5: Activity Data Table ────────────────────────────────────
    # (Target × Activity Type heatmap removed — IMP records table provides
    # per-(target, activity-record) detail in tabular form. Phase 21 design
    # decision: rely on integer IMP Score + records table for the pan-activity
    # signal; remove the standalone heatmap to declutter the page.)
    with st.expander("📋 Full Activity Data", expanded=False):
        display_cols = [
            "ChEMBL_ID",
            "Molecule_Name",
            "Activity_Type",
            "Activity_nM",
            "pActivity",
        ]
        if has_targets and target_col:
            display_cols.append(target_col)
        if has_assay_type:
            display_cols.append("Assay_Type")
        if has_quality:
            display_cols.append("Data_Quality")
        if "Document_Year" in df.columns:
            display_cols.append("Document_Year")
        if "Activity_Comment" in df.columns:
            display_cols.append("Activity_Comment")

        avail_cols = [c for c in display_cols if c in df.columns]
        table_df = df[avail_cols].copy()
        for c in ["Activity_nM", "pActivity"]:
            if c in table_df.columns:
                table_df[c] = table_df[c].round(2)

        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
            height=min(500, len(table_df) * 35 + 40),
        )


def _render_efficiency_analysis(df: pd.DataFrame) -> None:
    """Ligand Efficiency & IMP Evidence — shows how efficiency metrics feed into IMP scoring."""
    metrics = ["SEI", "BEI", "NSEI", "NBEI"]
    avail = [m for m in metrics if m in df.columns]

    if not avail:
        st.info("No efficiency metrics available")
        return

    theme = get_plotly_theme()
    has_sei_bei = "SEI" in df.columns and "BEI" in df.columns
    has_pact = "pActivity" in df.columns
    has_targets = "Target_Name" in df.columns or "Target_ChEMBL_ID" in df.columns
    target_col = (
        "Target_Name"
        if "Target_Name" in df.columns
        else ("Target_ChEMBL_ID" if "Target_ChEMBL_ID" in df.columns else None)
    )
    has_angle = "Angle_SEI_BEI" in df.columns
    has_modulus = "Modulus_SEI_BEI" in df.columns

    # ── Section 1: Efficiency Summary Cards ───────────────────────────────
    # Color thresholds: green = normal, amber = elevated, red = outlier
    def _eff_color(metric_name, value):
        """Return color based on typical ranges for each metric."""
        if pd.isna(value):
            return "off"
        return "off"  # st.metric delta_color, we use help text instead

    card_cols = st.columns(4 if has_sei_bei else len(avail))
    scored_metrics = [
        ("SEI", "pActivity × 100 / PSA", True),
        ("BEI", "pActivity × 1000 / MW", True),
    ]
    ref_metrics = [
        ("NSEI", "pActivity / NPOL", False),
        ("NBEI", "pActivity / NHA", False),
    ]

    for i, (m, formula, used_in_score) in enumerate(scored_metrics + ref_metrics):
        if m not in df.columns or i >= len(card_cols):
            continue
        vals = df[m].dropna()
        if len(vals) == 0:
            continue

        mean_val = vals.mean()
        outlier_col = f"Is_{m}_Outlier"
        n_outliers = int(df[outlier_col].sum()) if outlier_col in df.columns else 0

        with card_cols[i]:
            delta_txt = f"{'✓ Used in score' if used_in_score else 'Reference only'}"
            if n_outliers > 0:
                delta_txt = f"⚠️ {n_outliers} outlier{'s' if n_outliers > 1 else ''}"
            st.metric(
                m,
                f"{mean_val:.2f}",
                delta=delta_txt,
                delta_color="inverse" if n_outliers > 0 else "off",
                help=f"**{m}** = {formula}\nMean ± Std: {mean_val:.2f} ± {vals.std():.2f}\nRange: {vals.min():.2f} – {vals.max():.2f}\n{'**Contributes to IMP Score (45% weight)**' if used_in_score else 'Display only — not used in IMP scoring'}",
            )

    # Angle + Modulus summary if available
    if has_angle and has_modulus:
        angle_vals = df["Angle_SEI_BEI"].dropna()
        mod_vals = df["Modulus_SEI_BEI"].dropna()
        if len(angle_vals) > 0 and len(mod_vals) > 0:
            mean_angle = angle_vals.mean()
            mean_mod = mod_vals.mean()

            # Angle interpretation
            if 35 <= mean_angle <= 55:
                angle_label = "Balanced"
            elif mean_angle < 35:
                angle_label = "Hydrophobic-biased"
            else:
                angle_label = "Polar-biased"

            geo_cols = st.columns(2)
            with geo_cols[0]:
                st.metric(
                    "Mean Angle",
                    f"{mean_angle:.1f}°",
                    delta=angle_label,
                    delta_color="off",
                    help="**Efficiency Plane Angle** = arctan(BEI/SEI)\n0° = hydrophobic optimization only\n**45° = optimal** (balanced SEI + BEI)\n90° = polar optimization only\n\nAngles 35–55° indicate balanced drug development trajectory.",
                )
            with geo_cols[1]:
                st.metric(
                    "Mean Modulus",
                    f"{mean_mod:.2f}",
                    delta=f"Best: {mod_vals.max():.2f}",
                    delta_color="off",
                    help="**Efficiency Modulus** = √(SEI² + BEI²)\nOverall efficiency magnitude. Higher = more efficient.\nThe best-in-class compound sets the distance benchmark for IMP scoring (20% weight).",
                )

    st.markdown("---")

    # ── Section 2: SEI-BEI Efficiency Plane ───────────────────────────────
    if has_sei_bei:
        st.markdown("**SEI–BEI Efficiency Plane**")
        st.caption(
            "The canonical IMP visualization. 45° angle = balanced development. Outliers in this space drive 45% of the IMP score."
        )

        plane_df = df.dropna(subset=["SEI", "BEI"]).copy()
        if len(plane_df) >= 1:
            # Color by Activity_Type (5-7 categories) — targets can be 20+ which overflows legend
            color_col = None
            if "Activity_Type" in plane_df.columns:
                color_col = "Activity_Type"
            elif "IMP_Score_Interpretation" in plane_df.columns:
                color_col = "IMP_Score_Interpretation"

            fig = go.Figure()

            # 45° optimal angle reference line
            max_val = max(plane_df["SEI"].max(), plane_df["BEI"].max()) * 1.15
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    line=dict(color="rgba(128,128,128,0.4)", dash="dash", width=1.5),
                    name="45° Optimal",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # Compound points
            hover_cols = []
            if "ChEMBL_ID" in plane_df.columns:
                hover_cols.append("ChEMBL_ID")
            if "Molecule_Name" in plane_df.columns:
                hover_cols.append("Molecule_Name")
            if has_angle:
                hover_cols.append("Angle_SEI_BEI")
            if has_modulus:
                hover_cols.append("Modulus_SEI_BEI")

            # customdata layout: [SMILES, Molecule_Name, ChEMBL_ID, Angle, Modulus]
            # Index 0-2 = structure viewer expects SMILES/Name/ID
            # Index 3+ = angle/modulus for hover display
            has_smiles = "SMILES" in plane_df.columns
            angle_idx = 3 if has_smiles else 0
            mod_idx = angle_idx + (1 if has_angle else 0)

            hover_template = "<b>%{text}</b><br>SEI: %{x:.2f}<br>BEI: %{y:.2f}"
            if has_angle:
                hover_template += (
                    "<br>Angle: %{customdata[" + str(angle_idx) + "]:.1f}" + chr(176)
                )
            if has_modulus:
                hover_template += "<br>Modulus: %{customdata[" + str(mod_idx) + "]:.2f}"
            hover_template += "<extra></extra>"

            def _build_cd(sub_df):
                """Build customdata array with SMILES first for structure viewer."""
                cols = []
                if has_smiles:
                    cols.append(
                        sub_df["SMILES"].values
                        if "SMILES" in sub_df.columns
                        else [""] * len(sub_df)
                    )
                    cols.append(
                        sub_df["Molecule_Name"].values
                        if "Molecule_Name" in sub_df.columns
                        else [""] * len(sub_df)
                    )
                    cols.append(
                        sub_df["ChEMBL_ID"].values
                        if "ChEMBL_ID" in sub_df.columns
                        else [""] * len(sub_df)
                    )
                if has_angle:
                    cols.append(sub_df["Angle_SEI_BEI"].values)
                if has_modulus:
                    cols.append(sub_df["Modulus_SEI_BEI"].values)
                return list(zip(*cols)) if cols else None

            if color_col and color_col in plane_df.columns:
                for grp_name, grp_df in plane_df.groupby(color_col):
                    fig.add_trace(
                        go.Scatter(
                            x=grp_df["SEI"],
                            y=grp_df["BEI"],
                            mode="markers",
                            marker=dict(
                                size=9, opacity=0.8, line=dict(width=1, color="white")
                            ),
                            name=str(grp_name),
                            text=grp_df["Molecule_Name"]
                            if "Molecule_Name" in grp_df.columns
                            else grp_df.index.astype(str),
                            customdata=_build_cd(grp_df),
                            hovertemplate=hover_template,
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=plane_df["SEI"],
                        y=plane_df["BEI"],
                        mode="markers",
                        marker=dict(
                            size=9,
                            color="#3b82f6",
                            opacity=0.8,
                            line=dict(width=1, color="white"),
                        ),
                        name="Compounds",
                        text=plane_df["Molecule_Name"]
                        if "Molecule_Name" in plane_df.columns
                        else plane_df.index.astype(str),
                        customdata=_build_cd(plane_df),
                        hovertemplate=hover_template,
                    )
                )

            # Mean point marker
            mean_sei = plane_df["SEI"].mean()
            mean_bei = plane_df["BEI"].mean()
            fig.add_trace(
                go.Scatter(
                    x=[mean_sei],
                    y=[mean_bei],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="#ef4444",
                        symbol="diamond",
                        line=dict(width=2, color="white"),
                    ),
                    name=f"Mean ({mean_sei:.1f}, {mean_bei:.1f})",
                    hovertemplate=f"Mean SEI: {mean_sei:.2f}<br>Mean BEI: {mean_bei:.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                template=theme["template"],
                height=460,
                margin=dict(t=10, b=40, l=10, r=10),
                # 1:1 axis scaling intentionally dropped here — keeping it
                # forced the chart into a small square. The diagonal is
                # still y=x (balance line); the "angle" in the caption
                # refers to the chemical angle atan2(BEI, SEI) in data
                # space, not the visual pixel angle.
                xaxis=dict(title="SEI (Surface Efficiency)", range=[0, max_val]),
                yaxis=dict(title="BEI (Binding Efficiency)", range=[0, max_val]),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=1.02,
                    bgcolor=theme["legend_bgcolor"],
                    bordercolor=theme["legend_bordercolor"],
                    borderwidth=1,
                ),
            )
            apply_impulator_theme(fig)
            st.plotly_chart(fig, width="stretch", key="sei_bei_plane")
            _maybe_embed_structure_viewer(
                "sei_bei_plane", plane_df, x_col="SEI", y_col="BEI"
            )

        st.markdown("---")

    # ── Section 3: Efficiency vs Potency ──────────────────────────────────
    if has_sei_bei and has_pact:
        st.markdown("**Efficiency vs Potency**")
        st.caption(
            "Do efficiency and potency agree? High potency with low efficiency may indicate non-specific binding (aggregation)."
        )

        ep_col1, ep_col2 = st.columns(2)

        for col_widget, metric, title in [
            (ep_col1, "SEI", "SEI vs pActivity"),
            (ep_col2, "BEI", "BEI vs pActivity"),
        ]:
            with col_widget:
                ep_df = df.dropna(subset=[metric, "pActivity"]).copy()
                if len(ep_df) >= 2:
                    ep_cd = [
                        c
                        for c in ["SMILES", "Molecule_Name", "ChEMBL_ID"]
                        if c in ep_df.columns
                    ]
                    ep_chart_key = f"eff_potency_{metric.lower()}"
                    fig = px.scatter(
                        ep_df,
                        x=metric,
                        y="pActivity",
                        color="Activity_Type"
                        if "Activity_Type" in ep_df.columns
                        else None,
                        hover_data=["ChEMBL_ID", "Molecule_Name"]
                        if all(
                            c in ep_df.columns for c in ["ChEMBL_ID", "Molecule_Name"]
                        )
                        else None,
                        custom_data=ep_cd if ep_cd else None,
                        opacity=0.7,
                        trendline="ols",
                    )
                    fig.update_traces(marker=dict(size=7))
                    fig.update_layout(
                        template=theme["template"],
                        height=370,
                        margin=dict(t=30, b=40, l=10, r=10),
                        xaxis_title=metric,
                        yaxis_title="pActivity",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0,
                            bgcolor=theme["legend_bgcolor"],
                            bordercolor=theme["legend_bordercolor"],
                            borderwidth=1,
                            font=dict(size=13),
                        ),
                    )
                    apply_impulator_theme(fig)
                    st.plotly_chart(fig, width="stretch", key=ep_chart_key)
                    _maybe_embed_structure_viewer(
                        ep_chart_key, ep_df, x_col=metric, y_col="pActivity"
                    )
                else:
                    st.info(f"Not enough data for {title}")

        st.markdown("---")

    # ── Section 4: Efficiency by Target ───────────────────────────────────
    if has_targets and target_col and has_sei_bei:
        st.markdown("**Efficiency by Target**")
        st.caption(
            "Which biological targets produce the most efficient ligands? Targets with consistently high efficiency may warrant closer IMP scrutiny."
        )

        # Build target efficiency summary
        target_eff = (
            df.groupby(target_col)
            .agg(
                SEI_Mean=("SEI", "mean"),
                BEI_Mean=("BEI", "mean"),
                Records=("SEI", "count"),
            )
            .dropna()
            .round(2)
            .sort_values("SEI_Mean", ascending=True)
        )

        # Show top 10
        plot_tgt = target_eff.tail(10).copy()
        if not plot_tgt.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=plot_tgt.index.astype(str),
                    x=plot_tgt["SEI_Mean"],
                    orientation="h",
                    name="SEI",
                    marker_color="#3b82f6",
                    hovertemplate="<b>%{y}</b><br>Mean SEI: %{x:.2f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    y=plot_tgt.index.astype(str),
                    x=plot_tgt["BEI_Mean"],
                    orientation="h",
                    name="BEI",
                    marker_color="#22c55e",
                    hovertemplate="<b>%{y}</b><br>Mean BEI: %{x:.2f}<extra></extra>",
                )
            )
            fig.update_layout(
                template=theme["template"],
                barmode="group",
                height=max(280, len(plot_tgt) * 40 + 60),
                margin=dict(t=10, b=30, l=10, r=10),
                xaxis_title="Mean Efficiency",
                yaxis_title="",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                ),
            )
            apply_impulator_theme(fig)
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")

    # ── Section 5: Outlier Status Panel ───────────────────────────────────
    outlier_cols = [f"Is_{m}_Outlier" for m in avail if f"Is_{m}_Outlier" in df.columns]
    if outlier_cols:
        st.markdown("**Outlier Detection (IQR Method)**")
        st.caption(
            "Compounds flagged as statistical outliers in efficiency metrics. Outlier status feeds directly into the IMP Efficiency Score (45% of total)."
        )

        outlier_data = []
        for m in avail:
            oc = f"Is_{m}_Outlier"
            if oc not in df.columns:
                continue
            vals = df[m].dropna()
            n_out = int(df[oc].sum())
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
            outlier_data.append(
                {
                    "Metric": m,
                    "Used in Score": "✓" if m in ["SEI", "BEI"] else "—",
                    "Q1": round(q1, 2),
                    "Q3": round(q3, 2),
                    "IQR": round(iqr, 2),
                    "Threshold": round(threshold, 2),
                    "Outliers": n_out,
                    "% Flagged": f"{n_out / len(vals) * 100:.0f}%"
                    if len(vals) > 0
                    else "0%",
                }
            )

        if outlier_data:
            st.dataframe(pd.DataFrame(outlier_data), width="stretch", hide_index=True)

        st.markdown("---")

    # ── Section 6: Efficiency Metrics by Target Table ─────────────────────
    if target_col and any(m in df.columns for m in metrics):
        with st.expander("📋 Efficiency Metrics by Target", expanded=False):
            target_metrics = []
            for target in df[target_col].dropna().unique():
                target_df = df[df[target_col] == target]
                row = {
                    "Target_ChEMBL_ID": target
                    if target_col == "Target_ChEMBL_ID"
                    else (
                        target_df["Target_ChEMBL_ID"].iloc[0]
                        if "Target_ChEMBL_ID" in target_df.columns
                        else ""
                    ),
                    "Target_Name": target if target_col == "Target_Name" else "",
                }
                for m in ["SEI", "BEI", "NSEI", "NBEI"]:
                    if m in target_df.columns:
                        vals = target_df[m].dropna()
                        row[f"{m} Mean"] = (
                            round(vals.mean(), 3) if len(vals) > 0 else None
                        )
                        row[f"{m} Median"] = (
                            round(vals.median(), 3) if len(vals) > 0 else None
                        )
                target_metrics.append(row)

            if target_metrics:
                target_metrics_df = pd.DataFrame(target_metrics)
                if "SEI Mean" in target_metrics_df.columns:
                    target_metrics_df = target_metrics_df.sort_values(
                        "SEI Mean", ascending=False
                    )

                # Clickable ChEMBL target links
                col_config = {}
                if "Target_ChEMBL_ID" in target_metrics_df.columns:
                    target_metrics_df["Target_ChEMBL_ID"] = target_metrics_df[
                        "Target_ChEMBL_ID"
                    ].apply(
                        lambda v: (
                            f"https://www.ebi.ac.uk/chembl/explore/target/{str(v).strip()}"
                            if pd.notna(v)
                            and str(v).strip()
                            and str(v).strip().lower() != "nan"
                            else ""
                        )
                    )
                    col_config["Target_ChEMBL_ID"] = st.column_config.LinkColumn(
                        "Target ChEMBL ID",
                        display_text=r"https://www\.ebi\.ac\.uk/chembl/explore/target/(.*)",
                    )

                st.dataframe(
                    target_metrics_df,
                    width="stretch",
                    hide_index=True,
                    height=300,
                    column_config=col_config,
                )

    # ── Explanation ───────────────────────────────────────────────────────
    with st.expander("📖 Understanding Efficiency Metrics", expanded=False):
        st.markdown("""
**Efficiency metrics normalize potency by molecular size — critical for IMP detection.**

| Metric | Formula | IMP Role |
|--------|---------|----------|
| **SEI** | pActivity × 100 / PSA | ✓ Used in score (45% weight) |
| **BEI** | pActivity × 1000 / MW | ✓ Used in score (45% weight) |
| **NSEI** | pActivity / NPOL | Reference only |
| **NBEI** | pActivity / NHA | Reference only |

**Efficiency Plane Geometry:**
- **Angle** (0–90°): Development trajectory. 45° = optimal balanced development. <35° = hydrophobic-biased (aggregation risk). >55° = polar-biased (permeability issues).
- **Modulus** √(SEI² + BEI²): Overall efficiency magnitude. Best-in-class sets the distance benchmark (20% of IMP score).

**Outlier Detection:** IQR method flags compounds >Q3 + 1.5×IQR. Outliers in SEI/BEI → high efficiency Z-scores → high IMP Efficiency Score → higher IMP risk.
        """)


def _render_pains_analysis(df: pd.DataFrame) -> None:
    """Assay Interference Flags analysis - dedicated section."""
    if df is None:
        st.info("No data available")
        return

    has_assay = "PAINS_Violation" in df.columns

    if not has_assay:
        st.info(
            "No assay interference data available. Re-run analysis to generate PAINS screening."
        )
        return

    unique_df = df.drop_duplicates("ChEMBL_ID") if "ChEMBL_ID" in df.columns else df
    total = len(unique_df)

    st.markdown("**Assay Interference Flags**")
    st.caption("Detection of compounds with known assay interference mechanisms")

    # Summary metrics row — 7 flags with help tooltips
    # (column_name, emoji, short_description, help_tooltip)
    flags = {
        "PAINS": (
            "PAINS_Violation",
            "🔴",
            "Pan-Assay Interference",
            "Pan-Assay Interference Compounds (PAINS) — 480 substructure filters that identify "
            "compounds prone to false positives across multiple assay types. Baell & Holloway (2010).",
        ),
        "Aggregator": (
            "Aggregator_Risk",
            "🟠",
            "Colloidal Aggregation",
            "Compounds that form colloidal aggregates in aqueous solution, causing non-specific "
            "enzyme inhibition. Shoichet Lab criteria: >=3 aromatic rings, >300 Da, <=2 rotatable bonds, >3 LogP.",
        ),
        "Redox": (
            "Redox_Reactive",
            "🟡",
            "Redox Cycling",
            "Redox-active compounds (quinones, catechols, hydroquinones, nitroaromatics) that generate "
            "H2O2/ROS in assay buffers, causing false activity signals. 10 SMARTS patterns.",
        ),
        "Fluorescence": (
            "Fluorescence_Interference",
            "🔵",
            "Fluorescence Interference",
            "Autofluorescent scaffolds (coumarins, xanthenes, PAHs, stilbenes, flavonoids, acridines) "
            "that interfere with fluorescence-based assay readouts. 13 SMARTS patterns.",
        ),
        "Thiol": (
            "Thiol_Reactive",
            "🟣",
            "Thiol Reactivity",
            "Electrophilic compounds (Michael acceptors, acylating agents, epoxides, aldehydes) that "
            "react non-specifically with cysteine residues in target proteins. 15 SMARTS patterns.",
        ),
        "BRENK": (
            "BRENK_Alerts",
            "🟤",
            "Unwanted Substructures",
            "BRENK filter — 104 unwanted substructure patterns including reactive groups, toxic moieties, "
            "and metabolic liabilities. Used in screening library design. Brenk et al. (2008).",
        ),
        "NIH": (
            "NIH_Alerts",
            "⚪",
            "NIH Problematic Groups",
            "NIH-defined problematic functional groups that are frequently associated with assay artifacts "
            "or poor drug-likeness. RDKit FilterCatalog.NIH. Doveston et al. (2015).",
        ),
    }

    # Detail column mapping for building combined Details
    detail_col_map = {
        "PAINS": "PAINS_Details",
        "Aggregator": "Aggregator_Details",
        "Redox": "Redox_Details",
        "Fluorescence": "Fluorescence_Details",
        "Thiol": "Thiol_Details",
        "BRENK": "BRENK_Details",
        "NIH": "NIH_Details",
    }

    # Display all 7 flags as styled cards in a single row
    flag_data = []
    flag_items = list(flags.items())

    cards_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">'
    for name, (col, emoji, desc, helptext) in flag_items:
        count = int(unique_df[col].sum()) if col in unique_df.columns else 0
        pct = count / total * 100 if total > 0 else 0
        flag_data.append(
            {
                "Flag": f"{emoji} {name}",
                "Count": count,
                "%": f"{pct:.0f}%",
                "Description": desc,
            }
        )
        is_flagged = count > 0
        border_color = "#dc3545" if is_flagged else "#28a745"
        count_color = "#dc3545" if is_flagged else "#28a745"
        status_text = (
            f"&#9888; Flagged ({pct:.0f}%)" if is_flagged else "&#10003; Clean"
        )
        status_color = "#ffa94d" if is_flagged else "#51cf66"
        escaped_help = html.escape(helptext)
        cards_html += f'''
        <div title="{escaped_help}" style="flex:1;min-width:110px;background:var(--secondary-background-color);border-left:4px solid {border_color};
            border-radius:6px;padding:12px 10px;text-align:center;cursor:help;">
            <div style="font-size:1.8em;font-weight:bold;color:{count_color};">{count}</div>
            <div style="font-size:0.9em;color:var(--text-color);opacity:0.8;margin:4px 0;font-weight:600;">{html.escape(name)}</div>
            <div style="font-size:0.7em;color:{status_color};">{status_text}</div>
        </div>'''
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("---")

    # Flagged compounds as cards
    flag_col_names = {name: info[0] for name, info in flags.items()}
    flag_colors = {
        "PAINS": "#ef4444",
        "Aggregator": "#f97316",
        "Redox": "#eab308",
        "Fluorescence": "#3b82f6",
        "Thiol": "#a855f7",
        "BRENK": "#d97706",
        "NIH": "#0891b2",
    }
    available_flag_cols = [
        name for name, col in flag_col_names.items() if col in unique_df.columns
    ]

    if not available_flag_cols:
        st.success("No compounds flagged for assay interference")
    else:
        any_flagged_mask = unique_df[
            [flag_col_names[n] for n in available_flag_cols]
        ].any(axis=1)
        flagged_rows = unique_df[any_flagged_mask]

        if flagged_rows.empty:
            st.success("No compounds flagged for assay interference")
        else:
            # Build card data
            compound_cards = []
            for _, row in flagged_rows.iterrows():
                mol_name = row.get("Molecule_Name", "")
                if pd.isna(mol_name) or not isinstance(mol_name, str):
                    mol_name = ""
                chembl_id = str(row.get("ChEMBL_ID", "Unknown"))

                active = []
                for name in available_flag_cols:
                    if row.get(flag_col_names[name], False):
                        dcol = detail_col_map.get(name, "")
                        detail = ""
                        if dcol and dcol in row.index:
                            val = row.get(dcol, "")
                            if val and pd.notna(val) and str(val).strip():
                                detail = str(val)
                        active.append({"name": name, "detail": detail})

                compound_cards.append(
                    {
                        "chembl_id": chembl_id,
                        "mol_name": mol_name,
                        "flags": active,
                    }
                )

            # Filter
            all_flag_names = sorted(
                set(f["name"] for c in compound_cards for f in c["flags"])
            )
            fc1, fc2 = st.columns([3, 1])
            fc1.markdown(f"**{len(compound_cards)} Flagged Compounds**")
            selected_filter = fc2.selectbox(
                "Filter",
                ["All"] + all_flag_names,
                key="assay_flag_filter",
                label_visibility="collapsed",
            )
            if selected_filter != "All":
                compound_cards = [
                    c
                    for c in compound_cards
                    if any(f["name"] == selected_filter for f in c["flags"])
                ]

            # Pagination — 2 columns × 10 rows = 20 per page
            PAGE_SIZE = 20
            total = len(compound_cards)
            total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
            assay_page_key = "assay_page"
            if assay_page_key not in st.session_state:
                st.session_state[assay_page_key] = 1
            current_page = min(st.session_state[assay_page_key], total_pages)

            if total_pages > 1:
                _pdb_pagination(
                    "assay", assay_page_key, current_page, total_pages, "top"
                )

            start = (current_page - 1) * PAGE_SIZE
            page_cards = compound_cards[start : start + PAGE_SIZE]

            # Render cards in 2 columns
            left_col, right_col = st.columns(2)
            cols = [left_col, right_col]
            for i, c in enumerate(page_cards):
                chembl_id = html.escape(c["chembl_id"])
                mol_name = html.escape(c["mol_name"]) if c["mol_name"] else ""
                chembl_url = (
                    f"https://www.ebi.ac.uk/chembl/explore/compound/{chembl_id}"
                )

                # Flag pills
                pills_html = ""
                for f in c["flags"]:
                    color = flag_colors.get(f["name"], "#666")
                    pills_html += (
                        f'<span style="display:inline-block;padding:3px 10px;margin:2px 4px 2px 0;'
                        f"border-radius:12px;font-size:13px;font-weight:600;"
                        f'background:{color}22;color:{color};border:1px solid {color}44;">'
                        f"{html.escape(f['name'])}</span>"
                    )

                # Details
                details_html = ""
                for f in c["flags"]:
                    if f["detail"]:
                        details_html += (
                            f'<div style="font-size:14px;opacity:0.8;margin-top:4px;">'
                            f'<b style="color:{flag_colors.get(f["name"], "#666")}">'
                            f"{html.escape(f['name'])}:</b> {html.escape(f['detail'])}</div>"
                        )

                with cols[i % 2]:
                    st.markdown(
                        f'<div style="padding:12px;border:1px solid rgba(128,128,128,0.2);'
                        f'border-radius:8px;margin-bottom:8px;">'
                        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                        f'<a href="{chembl_url}" target="_blank" '
                        f'style="font-size:15px;font-weight:700;color:#3b82f6;text-decoration:none;">'
                        f"{chembl_id}</a>"
                        f"{('<span style=&quot;font-size:14px;opacity:0.7;&quot;>' + mol_name + '</span>') if mol_name else ''}"
                        f"</div>"
                        f'<div style="margin-bottom:4px;">{pills_html}</div>'
                        f"{details_html}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Bottom pagination
            if total_pages > 1:
                _pdb_pagination(
                    "assay", assay_page_key, current_page, total_pages, "bot"
                )

    # PAINS patterns breakdown (if available)
    if "PAINS_Pattern" in unique_df.columns:
        st.markdown("---")
        st.markdown("**PAINS Patterns Detected**")
        patterns = unique_df[unique_df["PAINS_Pattern"].notna()][
            "PAINS_Pattern"
        ].value_counts()
        if not patterns.empty:
            fig = px.bar(x=patterns.values, y=patterns.index, orientation="h")
            fig.update_layout(
                height=min(250, len(patterns) * 30 + 50),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            apply_impulator_theme(fig)
            st.plotly_chart(fig, width="stretch")

    # Methodology & References section
    with st.expander("Methodology & References", expanded=False):
        st.markdown("""
**All detections use peer-reviewed, mechanism-specific methods (96.2% overall accuracy):**

| Flag | Method | Patterns | Accuracy | Reference |
|------|--------|----------|----------|-----------|
| **PAINS** | RDKit FilterCatalog.PAINS | 480 | Industry std | Baell & Holloway (2010) |
| **Aggregator** | Shoichet Lab heuristics | 4 criteria | Published | Irwin et al. (2015) |
| **Thiol** | HTS electrophile SMARTS | 15 | 97.5% | Dahlin et al. (2015) |
| **Redox** | Quinone/catechol SMARTS | 10 | 91.4% | Proj et al. (2022) |
| **Fluorescence** | Fluorophore scaffold SMARTS | 13 | 97.7% | Su et al. (2015) |
| **BRENK** | RDKit FilterCatalog.BRENK | 104 | Industry std | Brenk et al. (2008) |
| **NIH** | RDKit FilterCatalog.NIH | Published | Published | Doveston et al. (2015) |

---

**Detection Methods:**

- **PAINS**: Uses RDKit's built-in PAINS FilterCatalog (480 patterns from Baell & Holloway)
- **Aggregator**: Published Shoichet Lab criteria (>=3 aromatic rings, >300 Da MW, <=2 rotatable bonds, >3 LogP)
- **Thiol**: 15 SMARTS patterns for electrophilic chemotypes (Michael acceptors including substituted acrylamides/crotonamides, acylating agents, SN2 electrophiles, aldehydes, isocyanates)
- **Redox**: 10 SMARTS patterns for quinones, catechols, hydroquinones, nitroaromatics that generate H2O2/ROS
- **Fluorescence**: 13 SMARTS patterns for autofluorescent scaffolds (coumarins, xanthenes, PAHs, stilbenes, flavonoids, acridines)
- **BRENK**: RDKit FilterCatalog with 104 unwanted substructure patterns (reactive groups, toxic moieties, metabolic liabilities)
- **NIH**: RDKit FilterCatalog for NIH-defined problematic functional groups

---

**Full Citations:**

- Baell, J.B. & Holloway, G.A. (2010). New substructure filters for removal of pan-assay interference compounds (PAINS). *J. Med. Chem.* 53, 2719-2740. DOI: [10.1021/jm901137j](https://doi.org/10.1021/jm901137j)
- Irwin, J.J. et al. (2015). An aggregation advisor for ligand discovery. *J. Med. Chem.* 58, 7076-7087. DOI: [10.1021/acs.jmedchem.5b01105](https://doi.org/10.1021/acs.jmedchem.5b01105)
- Dahlin, J.L. et al. (2015). PAINS in the assay: chemical mechanisms of assay interference. *J. Med. Chem.* 58, 2091-2113. DOI: [10.1021/jm5019093](https://doi.org/10.1021/jm5019093)
- Proj, M. et al. (2022). Redox-active compounds in drug discovery. *Antioxidants* 11, 1245. DOI: [10.3390/antiox11071245](https://doi.org/10.3390/antiox11071245)
- Su, Y. et al. (2015). High-throughput identification of compounds targeting autofluorescence. *Assay Drug Dev. Technol.* 13, 476-487. DOI: [10.1089/adt.2015.659](https://doi.org/10.1089/adt.2015.659)
- Brenk, R. et al. (2008). Lessons learnt from assembling screening libraries for drug discovery. *ChemMedChem* 3, 435-444. DOI: [10.1002/cmdc.200700139](https://doi.org/10.1002/cmdc.200700139)
- Doveston, R. et al. (2015). A unified lead-oriented synthesis of over eighty new scaffolds. *Org. Biomol. Chem.* 13, 859-865. DOI: [10.1039/C4OB02287D](https://doi.org/10.1039/C4OB02287D)
        """)

    # Important interpretation note at the bottom
    st.markdown("---")
    st.info("""
**Important Note:** These flags identify compounds with known assay interference mechanisms (PAINS, aggregation, redox activity, fluorescence, thiol reactivity, BRENK unwanted substructures, NIH problematic groups). However, **flags do NOT automatically disqualify compounds**. Many flagged compounds (e.g., quercetin with catechol groups) exhibit genuine polypharmacology validated by extensive PDB structural evidence.

**Interpretation:** Use PDB scores and structural evidence to distinguish genuine multi-target binders from assay artifacts. **High IMP scores + interference flags + high PDB scores = likely genuine polypharmacology.**
    """)


def _render_imp_score_breakdown(df: pd.DataFrame) -> None:
    """
    Render detailed IMP score breakdown for a representative compound.

    Shows all individual scores, efficiency metrics, and contribution breakdown.
    """
    if df is None or df.empty:
        return

    # Check if we have the required columns
    required_cols = [
        "IMP_Final_Score",
        "Efficiency_Score",
        "Angle_Score",
        "Distance_Score",
    ]
    if not all(col in df.columns for col in required_cols):
        return

    st.markdown("---")

    with st.expander("🎯 Detailed Score Breakdown", expanded=True):
        # Get representative row (highest scoring or first valid row)
        valid_df = df[df["IMP_Final_Score"].notna()]
        if valid_df.empty:
            st.info("No valid IMP scores available for breakdown")
            return

        # Use highest scoring compound for breakdown
        row = valid_df.loc[valid_df["IMP_Final_Score"].idxmax()]

        # Final Score Hero Section — locked score-card stack
        # (UI-SPEC §Component Visual Contract item #1: integer → global bar → dynamic bar)
        final_score = row.get("IMP_Final_Score", 0)
        score_int = format_imp_score(final_score) if pd.notna(final_score) else None

        # Observed bounds for the per-query dynamic range bar
        valid_scores = df["IMP_Final_Score"].dropna()
        if not valid_scores.empty:
            observed_min = float(valid_scores.min())
            observed_max = float(valid_scores.max())
        else:
            observed_min = None
            observed_max = None

        if score_int is None:
            st.html(
                '<div style="font-size:28px;font-weight:600;color:#6b7280;">IMP Score: N/A</div>'
            )
        else:
            obs_min_int = (
                format_imp_score(observed_min) if observed_min is not None else None
            )
            obs_max_int = (
                format_imp_score(observed_max) if observed_max is not None else None
            )
            global_bar_svg = render_imp_range_bar_global(final_score)
            dynamic_bar_svg = render_imp_range_bar_dynamic(
                final_score, observed_min, observed_max
            )
            obs_min_str = str(obs_min_int) if obs_min_int is not None else "—"
            obs_max_str = str(obs_max_int) if obs_max_int is not None else "—"
            # st._main._html (iframe) bypasses DOMPurify which would strip <defs>/<linearGradient>
            st._main._html(
                "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;\">"
                '<div style="font-size:14px; font-weight:500; color:#6b7280; letter-spacing:0.02em; text-transform:uppercase;">IMP Score</div>'
                f'<div style="font-size:48px; font-weight:700; color:#111827; line-height:1; margin-top:2px;">{score_int}</div>'
                '<div style="font-size:14px; font-weight:400; color:#6b7280; margin-top:8px;">Global reference (10–80)</div>'
                f"{global_bar_svg}"
                f'<div style="display:flex; justify-content:space-between; max-width:240px; font-size:14px; color:#6b7280; margin-top:4px;"><span>{IMP_SCORE_FLOOR}</span><span>{IMP_SCORE_CEILING}</span></div>'
                f'<div style="font-size:14px; font-weight:400; color:#6b7280; margin-top:16px;">This query\'s range ({obs_min_str}–{obs_max_str})</div>'
                f"{dynamic_bar_svg}"
                '<div style="display:flex; justify-content:space-between; max-width:240px; font-size:14px; color:#6b7280; margin-top:4px;"><span>0</span><span>100</span></div>'
                "</div>",
                height=240,
            )

        # Efficiency Metrics Section
        st.markdown("#### 📊 Efficiency Metrics")
        st.caption(
            "All four metrics are calculated. Only **SEI** and **BEI** are used in the score."
        )

        eff_cols = st.columns(4)

        with eff_cols[0]:
            sei = row.get("SEI")
            st.metric(
                "SEI",
                f"{sei:.2f}" if pd.notna(sei) else "N/A",
                help="Surface Efficiency Index = pActivity ÷ (PSA/100). **Used in score.**",
            )

        with eff_cols[1]:
            bei = row.get("BEI")
            st.metric(
                "BEI",
                f"{bei:.2f}" if pd.notna(bei) else "N/A",
                help="Binding Efficiency Index = pActivity ÷ (MW/1000). **Used in score.**",
            )

        with eff_cols[2]:
            nsei = row.get("NSEI")
            st.metric(
                "NSEI",
                f"{nsei:.2f}" if pd.notna(nsei) else "N/A",
                help="Normalized SEI = pActivity ÷ NPOL. Display only.",
            )

        with eff_cols[3]:
            nbei = row.get("NBEI")
            st.metric(
                "NBEI",
                f"{nbei:.3f}" if pd.notna(nbei) else "N/A",
                help="Normalized BEI = pActivity ÷ NHA. Display only.",
            )

        # Plane Geometry Section
        st.markdown("#### 📐 Efficiency Plane Geometry")

        geom_cols = st.columns(2)

        with geom_cols[0]:
            modulus = row.get("Modulus_SEI_BEI")
            st.metric("Modulus", f"{modulus:.2f}" if pd.notna(modulus) else "N/A")
            st.caption("""
            The modulus measures the distance of the combined efficiency vector (SEI, BEI)
            from the origin on the efficiency plane. It represents the overall efficiency
            magnitude. While derived from SEI and BEI, the modulus is independent of
            the development angle—the angle defines direction, not magnitude.
            """)

        with geom_cols[1]:
            angle = row.get("Angle_SEI_BEI")
            if pd.notna(angle):
                angle_deviation = abs(angle - 45)
                if angle_deviation < 10:
                    angle_status = "✅ Optimal"
                elif angle_deviation < 20:
                    angle_status = "⚠️ Moderate"
                else:
                    angle_status = "❌ Unbalanced"
                st.metric(
                    "Development Angle",
                    f"{angle:.1f}°",
                    delta=angle_status,
                    delta_color="off",
                )
            else:
                st.metric("Development Angle", "N/A")
            st.caption(
                "Optimal angle is 45°. <30° = too hydrophobic, >60° = too polar."
            )

        # Component Scores Section
        st.markdown("#### 🎯 Component Scores & Contributions")

        comp_cols = st.columns(5)

        with comp_cols[0]:
            eff_score = row.get("Efficiency_Score", 0)
            eff_contrib = row.get("Efficiency_Contribution", 0)
            st.metric(
                "Efficiency",
                f"{eff_score:.3f}" if pd.notna(eff_score) else "N/A",
                help="Weight: 45%",
            )
            if pd.notna(eff_score):
                st.progress(max(0.0, min(1.0, float(eff_score))))
            st.caption(
                f"Contribution: {eff_contrib:.3f}" if pd.notna(eff_contrib) else ""
            )
            sei_z = row.get("SEI_zscore", None)
            bei_z = row.get("BEI_zscore", None)
            if (
                sei_z is not None
                and bei_z is not None
                and pd.notna(sei_z)
                and pd.notna(bei_z)
            ):
                st.caption(f"SEI z={sei_z:.2f} · BEI z={bei_z:.2f}")

        with comp_cols[1]:
            dist_score = row.get("Distance_Score", 0)
            dist_contrib = row.get("Distance_Contribution", 0)
            st.metric(
                "Distance",
                f"{dist_score:.3f}" if pd.notna(dist_score) else "N/A",
                help="Weight: 20%",
            )
            if pd.notna(dist_score):
                st.progress(max(0.0, min(1.0, float(dist_score))))
            st.caption(
                f"Contribution: {dist_contrib:.3f}" if pd.notna(dist_contrib) else ""
            )
            modulus = row.get("Modulus_SEI_BEI", None)
            if modulus is not None and pd.notna(modulus):
                st.caption(f"Modulus: {modulus:.2f}")

        with comp_cols[2]:
            ang_score = row.get("Angle_Score", 0)
            ang_contrib = row.get("Angle_Contribution", 0)
            st.metric(
                "Angle",
                f"{ang_score:.3f}" if pd.notna(ang_score) else "N/A",
                help="Weight: 15%",
            )
            if pd.notna(ang_score):
                st.progress(max(0.0, min(1.0, float(ang_score))))
            st.caption(
                f"Contribution: {ang_contrib:.3f}" if pd.notna(ang_contrib) else ""
            )
            angle = row.get("Angle_SEI_BEI", None)
            if angle is not None and pd.notna(angle):
                st.caption(f"Angle: {angle:.1f}° (optimal: 45°)")

        with comp_cols[3]:
            int_score = row.get("Interference_Score", 0)
            int_contrib = row.get("Interference_Contribution", 0)
            st.metric(
                "Interference",
                f"{int_score:.3f}" if pd.notna(int_score) else "N/A",
                help="Weight: 15% — Scored flags / 5 (BRENK/NIH display-only)",
            )
            if pd.notna(int_score):
                st.progress(max(0.0, min(1.0, float(int_score))))
            st.caption(
                f"Contribution: {int_contrib:.3f}" if pd.notna(int_contrib) else ""
            )

            # Show which flags triggered
            scored_flags = [
                ("PAINS_Violation", "PAINS"),
                ("Aggregator_Risk", "Aggregator"),
                ("Redox_Reactive", "Redox"),
                ("Fluorescence_Interference", "Fluorescence"),
                ("Thiol_Reactive", "Thiol"),
            ]
            triggered = sum(1 for col, _ in scored_flags if row.get(col, 0) == 1)
            flag_parts = []
            for col, label in scored_flags:
                val = row.get(col, 0)
                flag_parts.append(f"{'🔴' if val == 1 else '🟢'} {label}")
            st.caption(f"{triggered}/5 flags triggered")
            st.caption(" · ".join(flag_parts))

        with comp_cols[4]:
            pdb_score = row.get("PDB_Score", 0)
            pdb_contrib = row.get("PDB_Contribution", 0)
            st.metric(
                "PDB Evidence",
                f"{pdb_score:.3f}" if pd.notna(pdb_score) else "N/A",
                help="Weight: 5%",
            )
            if pd.notna(pdb_score):
                st.progress(max(0.0, min(1.0, float(pdb_score))))
            st.caption(
                f"Contribution: {pdb_contrib:.3f}" if pd.notna(pdb_contrib) else ""
            )
            pdb_hits = row.get("PDB_Hits", None)
            if pdb_hits is not None and pd.notna(pdb_hits):
                st.caption(f"PDB hits: {int(pdb_hits)}")

        # Final Calculation Section
        st.markdown("#### 🧮 Final Calculation")

        base_score = row.get("IMP_Base_Score", 0)
        qed = row.get("QED", 0)
        qed_mult = row.get("QED_Multiplier", 0)
        qed_impact = row.get("QED_Impact", 0)

        calc_cols = st.columns(3)

        with calc_cols[0]:
            st.metric(
                "Base Score",
                f"{base_score:.3f}" if pd.notna(base_score) else "N/A",
                help="Sum of weighted component contributions",
            )

        with calc_cols[1]:
            st.metric(
                "QED",
                f"{qed:.3f}" if pd.notna(qed) else "N/A",
                help="Quantitative Estimate of Drug-likeness (0-1)",
            )

        with calc_cols[2]:
            st.metric(
                "QED Multiplier",
                f"{qed_mult:.3f}" if pd.notna(qed_mult) else "N/A",
                delta=f"Impact: {qed_impact:+.3f}" if pd.notna(qed_impact) else None,
                help="Formula: 0.75 + 0.25 × QED. Floor at 75%.",
            )

        # Formula display with actual values
        if pd.notna(base_score) and pd.notna(qed_mult):
            eff_s = row.get("Efficiency_Score", 0)
            dist_s = row.get("Distance_Score", 0)
            ang_s = row.get("Angle_Score", 0)
            int_s = row.get("Interference_Score", 0)
            pdb_s = row.get("PDB_Score", 0)
            st.markdown(
                f"""
<div style="background-color: var(--secondary-background-color); padding: 16px 20px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.95rem; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word;">
<span style="color: #3b82f6; font-weight: 600;">Base Score</span>
  = 0.45 x Eff + 0.20 x Dist + 0.15 x Angle + 0.15 x Interf + 0.05 x PDB
  = 0.45 x <span style="color: #22c55e; font-weight: 600;">{eff_s:.3f}</span> + 0.20 x <span style="color: #22c55e; font-weight: 600;">{dist_s:.3f}</span> + 0.15 x <span style="color: #22c55e; font-weight: 600;">{ang_s:.3f}</span> + 0.15 x <span style="color: #22c55e; font-weight: 600;">{int_s:.3f}</span> + 0.05 x <span style="color: #22c55e; font-weight: 600;">{pdb_s:.3f}</span>
  = <span style="color: #f59e0b; font-weight: 700;">{base_score:.3f}</span>

<span style="color: #3b82f6; font-weight: 600;">QED Multiplier</span>
  = 0.75 + 0.25 x {qed:.3f} = <span style="color: #f59e0b; font-weight: 700;">{qed_mult:.3f}</span>

<span style="color: #3b82f6; font-weight: 600;">Final Score</span>
  = Base Score x QED Multiplier
  = {base_score:.3f} x {qed_mult:.3f}
  = <span style="color: #ef4444; font-size: 1.1em; font-weight: 700;">{final_score:.3f}</span>
</div>
            """,
                unsafe_allow_html=True,
            )

        # Contribution Pie Chart
        _render_contribution_chart(row)

        # PDB Details (if available)
        pdb_structures = row.get("PDB_Num_Structures", 0)
        if pd.notna(pdb_structures) and pdb_structures > 0:
            st.markdown("#### 🔬 PDB Structural Evidence")

            pdb_cols = st.columns(4)

            with pdb_cols[0]:
                st.metric("Total Structures", int(pdb_structures))
            with pdb_cols[1]:
                st.metric("High Quality (<2.0Å)", int(row.get("PDB_High_Quality", 0)))
            with pdb_cols[2]:
                st.metric(
                    "Medium Quality (2-3Å)", int(row.get("PDB_Medium_Quality", 0))
                )
            with pdb_cols[3]:
                st.metric("Poor Quality (>3.0Å)", int(row.get("PDB_Poor_Quality", 0)))

            st.caption("""
            **Interpretation:**
            High quality structures (<2.0Å) provide strongest validation.
            Multiple structures increase confidence.
            Low PDB score with high efficiency = potential artifact (RED FLAG).
            """)


# =============================================================================
# Phase 22 — GMM Cluster Membership widget (helpers + main render)
# =============================================================================


def _corpus_key_for_query(df: pd.DataFrame) -> str:
    """SHA-1 cache discriminator for the 'This query' corpus.

    Hashes the sorted list of compound identifiers so the cache hits remain
    stable across DataFrame row reorderings. Falls back through
    ``entry_id`` → ``Entry_ID`` → ``ChEMBL_ID`` → ``df.index`` depending on
    which identifier column the host DataFrame carries.
    """
    if "entry_id" in df.columns:
        id_col = "entry_id"
    elif "Entry_ID" in df.columns:
        id_col = "Entry_ID"
    elif "ChEMBL_ID" in df.columns:
        id_col = "ChEMBL_ID"
    else:
        id_col = None

    if id_col is None:
        ids = sorted(str(i) for i in df.index.tolist())
    else:
        ids = sorted(df[id_col].astype(str).tolist())

    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()


@st.cache_data
def _fit_gmm_cached(
    scores_tuple: tuple,
    n_components: int,
    random_state: int,
    corpus_key: str,
    grain: str,
):
    """Cached wrapper around :func:`backend.modules.imp_gmm.fit_gmm`.

    ``corpus_key`` and ``grain`` are unused inside the body — they exist
    purely as cache-key discriminators per CONTEXT.md D-11.
    """
    scores = np.array(scores_tuple, dtype=float)
    return fit_gmm(scores, n_components=n_components, random_state=random_state)


@st.cache_data
def _best_fit_k_cached(
    scores_tuple: tuple,
    random_state: int,
    corpus_key: str,
    grain: str,
) -> int:
    """Cached wrapper around :func:`backend.modules.imp_gmm.best_fit_k`.

    Same cache-key discipline as ``_fit_gmm_cached`` — ``corpus_key`` and
    ``grain`` are unused inside the body but distinguish equivalent
    score-tuples across corpus/grain choices so the wrong K never bleeds
    across views.
    """
    scores = np.array(scores_tuple, dtype=float)
    return best_fit_k(scores, random_state=random_state)


def _gmm_can_fit(
    scores: np.ndarray, n_components: int, corpus_choice_label: str
) -> tuple[bool, Optional[str]]:
    """Sentinel-decision helper for the GMM widget.

    Returns ``(True, None)`` when fitting is possible; otherwise
    ``(False, locked_sentinel_message)`` with the appropriate variant
    string interpolated.
    """
    if scores.size <= n_components:
        return False, gmm_sentinel_message(
            n_components, int(scores.size), corpus_choice_label, variant="small_corpus"
        )
    if float(np.var(scores)) == 0.0:
        return False, gmm_sentinel_message(
            n_components, int(scores.size), corpus_choice_label, variant="zero_variance"
        )
    n_unique = int(len(np.unique(scores)))
    if n_unique < n_components:
        return False, gmm_sentinel_message(
            n_components,
            int(scores.size),
            corpus_choice_label,
            variant="few_unique",
            n_unique=n_unique,
        )
    return True, None


def _render_gmm_widget(df: pd.DataFrame, job_id: str) -> None:
    """Render the GMM Cluster Membership section.

    Inserted between Phase 21's Detailed Score Breakdown and the IMP
    Candidates section inside :func:`_render_imp_score_analysis`. Renders
    inline (no expander wrapper); the GMM density overlay + cluster
    probability bar always show when the corpus has enough scored compounds.
    """
    st.markdown("---")
    st.markdown("### 🔀 GMM Cluster Membership")

    # Guard: missing data or missing IMP_Final_Score column → sentinel.
    if df is None or df.empty or "IMP_Final_Score" not in df.columns:
        st.info(
            "IMP scores are not available for this job. The GMM widget "
            "needs scored compounds."
        )
        return

    st.caption(
        "Fit a Gaussian Mixture Model to the IMP score distribution. "
        "The bar below shows how this compound's score breaks across "
        "the clusters."
    )

    # Flash any auto-fallback notice set on the previous render (e.g. when
    # Per-compound max was too small for K and the widget switched to
    # Per-record on the user's behalf). Pop so it shows exactly once.
    fallback_notice_key = f"gmm_fallback_notice_{job_id}"
    if fallback_notice_key in st.session_state:
        st.info(st.session_state.pop(fallback_notice_key))

    # Apply any pending grain override set by a prior auto-fallback run BEFORE
    # the segmented_control instantiates. Streamlit forbids writing to a
    # widget key AFTER the widget renders in the same script execution, so
    # the fallback writes to a separate "force" key and st.rerun()s; we
    # transfer the value here, before the toggle below claims its key.
    force_grain_key = f"gmm_force_grain_{job_id}"
    if force_grain_key in st.session_state:
        st.session_state[f"gmm_grain_{job_id}"] = st.session_state.pop(
            force_grain_key
        )

    # --- Controls row A: corpus + (conditional) grain -----------------------
    seg = getattr(st, "segmented_control", None)
    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        if seg is not None:
            corpus_choice = seg(
                "Fit on",
                options=["This query", "Reference corpus"],
                default="This query",
                key=f"gmm_corpus_{job_id}",
            )
        else:
            corpus_choice = st.radio(
                "Fit on",
                options=["This query", "Reference corpus"],
                horizontal=True,
                key=f"gmm_corpus_{job_id}",
            )

    grain_choice = None
    with col_a2:
        if corpus_choice == "This query":
            if seg is not None:
                grain_choice = seg(
                    "Grain",
                    options=["Per-compound max", "Per-record"],
                    default="Per-compound max",
                    key=f"gmm_grain_{job_id}",
                )
            else:
                grain_choice = st.radio(
                    "Grain",
                    options=["Per-compound max", "Per-record"],
                    horizontal=True,
                    key=f"gmm_grain_{job_id}",
                )

    # --- Corpus + grain → scores array --------------------------------------
    grain: Optional[str] = None
    if corpus_choice == "This query":
        if "entry_id" in df.columns:
            group_col = "entry_id"
        elif "Entry_ID" in df.columns:
            group_col = "Entry_ID"
        elif "ChEMBL_ID" in df.columns:
            group_col = "ChEMBL_ID"
        else:
            group_col = None

        if grain_choice == "Per-compound max" and group_col is not None:
            scores_raw = (
                df.groupby(group_col)["IMP_Final_Score"].max().dropna().values
            )
        else:
            scores_raw = df["IMP_Final_Score"].dropna().values

        corpus_key = _corpus_key_for_query(df)
        corpus_choice_label = "query"
    else:
        corpus = load_reference_corpus()
        if not corpus:
            st.info(
                "Reference corpus is not available in this build. "
                "Falling back to 'This query'."
            )
            st.session_state[f"gmm_corpus_{job_id}"] = "This query"
            st.rerun()
            return
        scores_raw = np.array(
            [
                c["imp_final_score"]
                for c in corpus
                if c.get("imp_final_score") is not None
            ],
            dtype=float,
        )
        corpus_key = REFERENCE_CORPUS_KEY
        grain = "reference"
        corpus_choice_label = "reference corpus"

    # Rescale to integer space [0, 100] BEFORE the sentinel check so
    # _gmm_can_fit and the fit see the same units.
    scores_arr = np.asarray(scores_raw, dtype=float).ravel()
    if scores_arr.size > 0 and scores_arr.max() <= 1.0:
        scores = scores_arr * 100.0
    else:
        scores = scores_arr

    # --- Grain literal (used for cache keys + best_fit_k cache discriminator)
    if corpus_choice == "This query":
        grain = (
            (grain_choice or "Per-compound max")
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    # sorted for row-order-independent cache hits (GMM fits are
    # sample-order-invariant; R7 fix from REVIEWS.md)
    scores_tuple = tuple(sorted(float(s) for s in scores))

    # --- Auto-select K via BIC and re-seed the slider on context changes ---
    # We compute a "fingerprint" of (job, active compound, corpus, grain) and
    # snap the slider to the BIC-suggested K whenever the fingerprint changes
    # (opening a different compound, switching corpus, switching grain). When
    # the fingerprint stays the same, the user's slider choice sticks — so
    # they can override BIC and the override survives any number of reruns
    # within the same context.
    suggested_k: Optional[int] = None
    if scores.size >= MIN_COMPONENTS:
        suggested_k = _best_fit_k_cached(
            scores_tuple, DEFAULT_RANDOM_STATE, corpus_key, grain
        )

    slider_key = f"gmm_n_components_{job_id}"
    fingerprint_key = f"gmm_fingerprint_{job_id}"
    active_id_for_fingerprint = str(
        SessionState.get("selected_compound_entry_id")
        or SessionState.get("selected_compound")
        or "no_compound"
    )
    fingerprint = (
        job_id,
        active_id_for_fingerprint,
        corpus_key,
        grain or "",
    )
    if (
        suggested_k is not None
        and st.session_state.get(fingerprint_key) != fingerprint
    ):
        st.session_state[slider_key] = suggested_k
        st.session_state[fingerprint_key] = fingerprint

    # --- Controls row B: slider + refit + reset ----------------------------
    col_b1, col_b2, col_b3 = st.columns([3, 1, 1])
    with col_b1:
        n_components = st.slider(
            "Number of components",
            min_value=MIN_COMPONENTS,
            max_value=MAX_COMPONENTS,
            value=DEFAULT_COMPONENTS,
            step=1,
            key=slider_key,
        )
    with col_b2:
        if st.button(
            "↻ Refit (new seed)",
            type="secondary",
            key=f"gmm_refit_{job_id}",
        ):
            st.session_state[f"gmm_seed_{job_id}"] = int(
                np.random.randint(0, 2**31)
            )
            st.rerun()
    with col_b3:
        seed_key = f"gmm_seed_{job_id}"
        if (
            seed_key in st.session_state
            and st.session_state[seed_key] != DEFAULT_RANDOM_STATE
        ):
            if st.button(
                "Reset to default seed",
                type="secondary",
                key=f"gmm_reset_{job_id}",
            ):
                del st.session_state[seed_key]
                st.rerun()

    random_state = st.session_state.get(seed_key, DEFAULT_RANDOM_STATE)

    if suggested_k is not None:
        marker = " ← in use" if int(n_components) == int(suggested_k) else ""
        st.caption(f"BIC suggests K = {suggested_k}{marker}.")

    # --- Sentinel decision --------------------------------------------------
    can_fit, message = _gmm_can_fit(scores, n_components, corpus_choice_label)
    if not can_fit:
        # Auto-fallback: if Per-compound max yields too few points for K, but
        # Per-record would have enough activity rows, flip the grain toggle and
        # rerun. The toggle visibly snaps to Per-record so the UI state matches
        # what's being fit; the fingerprint logic above re-seeds the slider
        # with the new BIC suggestion for the per-record corpus.
        if (
            corpus_choice == "This query"
            and grain_choice == "Per-compound max"
        ):
            per_record_size = int(df["IMP_Final_Score"].dropna().size)
            if per_record_size > int(n_components):
                # Use a "force" key — the grain widget already rendered in
                # this run, so we can't write to its key directly. The
                # transfer happens on the next run before the widget
                # instantiates (see force_grain_key handling above).
                st.session_state[f"gmm_force_grain_{job_id}"] = "Per-record"
                st.session_state[f"gmm_fallback_notice_{job_id}"] = (
                    f"Per-compound corpus has only {int(scores.size)} compound(s) — "
                    f"too few for K={int(n_components)}. Switched to Per-record "
                    f"({per_record_size} activity records)."
                )
                st.rerun()
                return

        st.info(message)
        return

    model = _fit_gmm_cached(
        scores_tuple, n_components, random_state, corpus_key, grain
    )

    # R6 fix: refuse to render a misleading chart when GMM did not converge.
    if not getattr(model, "converged_", True):
        st.info(
            "GMM did not fully converge for this corpus + component "
            "count. Try fewer components, or switch corpus."
        )
        return

    # --- Resolve active compound (for probability bar only — no vline) ------
    compound_name = SessionState.get("selected_compound")
    entry_id_active = SessionState.get("selected_compound_entry_id")

    active_rows = None
    if entry_id_active is not None and "entry_id" in df.columns:
        active_rows = df[df["entry_id"].astype(str) == str(entry_id_active)]
    elif entry_id_active is not None and "Entry_ID" in df.columns:
        active_rows = df[df["Entry_ID"].astype(str) == str(entry_id_active)]
    if (
        (active_rows is None or active_rows.empty)
        and compound_name
        and "Molecule_Name" in df.columns
    ):
        active_rows = df[df["Molecule_Name"] == compound_name]
    if (
        (active_rows is None or active_rows.empty)
        and compound_name
        and "ChEMBL_ID" in df.columns
    ):
        active_rows = df[df["ChEMBL_ID"] == compound_name]

    if active_rows is None or active_rows.empty:
        this_score = None
    else:
        valid_active = active_rows[active_rows["IMP_Final_Score"].notna()]
        if valid_active.empty:
            this_score = None
        else:
            row = valid_active.loc[valid_active["IMP_Final_Score"].idxmax()]
            this_score_raw = row.get("IMP_Final_Score", None)
            if this_score_raw is None or (
                isinstance(this_score_raw, float) and math.isnan(this_score_raw)
            ):
                this_score = None
            else:
                this_score = (
                    float(this_score_raw) * 100.0
                    if float(this_score_raw) <= 1.0
                    else float(this_score_raw)
                )

    # --- Density overlay (no vline — chart shows corpus structure only) -----
    density_fig = create_gmm_density_overlay(scores, model)
    st.plotly_chart(density_fig, width="stretch")

    # --- Probability bar (conditional) -------------------------------------
    if this_score is None:
        st.caption(
            "This compound has no IMP score in the current corpus; "
            "cluster membership is undefined."
        )
    else:
        memberships = cluster_membership(model, this_score)
        cluster_means_sorted = sorted(float(m) for m in model.means_.flatten())
        prob_fig = create_gmm_probability_bar(
            memberships, cluster_means=np.array(cluster_means_sorted)
        )
        st.plotly_chart(prob_fig, width="stretch")

    # --- Explanation panel --------------------------------------------------
    with st.expander("What is GMM and how does this widget work?", expanded=False):
        st.markdown(
            """
**Gaussian Mixture Models (GMMs)** assume a dataset is generated from a
mixture of several Gaussian (normal) distributions. Each Gaussian — a
"component" — has its own mean (μ), variance (σ²), and weight. Fitting a
GMM means estimating those parameters via the **Expectation-Maximization
(EM) algorithm**. Given a new score, the model returns the probability
that the score was generated by each component
(`P(cluster_k | score)`) — those probabilities are what the stacked bar
above shows.

Why GMMs for IMP scores: the IMP score distribution often shows a few
latent sub-populations (e.g., "drug-like", "borderline", "promiscuous /
PAINS-like"). A GMM lets us model that structure explicitly rather than
collapsing it into a single histogram.

**How "best fit K" works.** We don't pick the number of components by
hand. The widget fits the GMM for every K in [2, 6], scores each model
with the **Bayesian Information Criterion (BIC)**, and reports the K
that minimizes BIC.

```
BIC = -2 · log-likelihood  +  k · log(n)
```

Lower is better. The first term rewards a model that explains the data
well; the second term penalizes complexity (more components = more
parameters). BIC strikes a balance between fit and parsimony, and it's
the standard model-selection criterion for GMMs in the scikit-learn
documentation. The "BIC suggests K = N" caption above the slider always
shows the current recommendation — you can override it by moving the
slider.

**Sources.**

- Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR 12,
  2825–2830 (2011) — the `GaussianMixture` implementation used here.
  ([link](https://jmlr.org/papers/v12/pedregosa11a.html))
- Schwarz, G. *Estimating the dimension of a model.* The Annals of
  Statistics 6 (2), 461–464 (1978) — the BIC criterion.
  ([link](https://doi.org/10.1214/aos/1176344136))
- Scikit-learn User Guide §2.1 — Gaussian mixture models.
  ([link](https://scikit-learn.org/stable/modules/mixture.html))

**Implementation notes.**

- All fits use `covariance_type='full'` (each component gets its own
  covariance matrix) with `random_state=42` so that re-runs with the
  same corpus + same K yield identical clusters. The **Refit (new seed)**
  button is for sensitivity-checking how much the cluster structure
  depends on the random initialization.
- Cluster ordering is fixed by ascending mean (so C₀ is always the
  lowest-scoring cluster). The colors in the density chart and the
  probability bar are matched (ColorBrewer Set2 palette).
- The histogram shows the **corpus distribution** the model was fit on.
  We do NOT mark the active compound's position on it, because the
  corpus is built from the active compound's Tanimoto neighborhood — a
  "this compound" vline on a histogram of its own neighbors would be
  visually circular.
"""
        )


def _render_contribution_chart(row: pd.Series) -> None:
    """Render a radar chart showing component scores + weighted contribution bar chart."""
    components = [
        ("Efficiency", 0.45, "Efficiency_Score"),
        ("Distance", 0.20, "Distance_Score"),
        ("Angle", 0.15, "Angle_Score"),
        ("Interference", 0.15, "Interference_Score"),
        ("PDB", 0.05, "PDB_Score"),
    ]

    names = []
    raw_scores = []
    weighted_scores = []
    weights = []
    for name, weight, col in components:
        score = row.get(col, 0)
        score = float(score) if pd.notna(score) else 0.0
        names.append(name)
        raw_scores.append(score)
        weighted_scores.append(weight * score)
        weights.append(weight)

    if sum(weighted_scores) <= 0.005:
        return

    radar_col, bar_col = st.columns(2)

    with radar_col:
        # Radar/spider chart — raw component scores (0-1 scale)
        fig_radar = go.Figure()
        radar_colors = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#a855f7"]
        # Filled area
        fig_radar.add_trace(
            go.Scatterpolar(
                r=raw_scores + [raw_scores[0]],
                theta=names + [names[0]],
                fill="toself",
                fillcolor="rgba(99, 102, 241, 0.2)",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(
                    size=8,
                    color=radar_colors + [radar_colors[0]],
                    line=dict(color="#fff", width=1.5),
                ),
                text=[f"<b>{n}</b>: {s:.3f}" for n, s in zip(names, raw_scores)] + [""],
                hoverinfo="text",
            )
        )
        # Max reference ring
        fig_radar.add_trace(
            go.Scatterpolar(
                r=[1, 1, 1, 1, 1, 1],
                theta=names + [names[0]],
                line=dict(color="rgba(255,255,255,0.1)", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_radar.update_layout(
            title="Component Scores",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10),
                    gridcolor="rgba(128,128,128,0.2)",
                ),
                angularaxis=dict(tickfont=dict(size=13, color="#ddd")),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            height=300,
            margin=dict(t=40, b=20, l=50, r=50),
        )
        apply_impulator_theme(fig_radar)
        st.plotly_chart(fig_radar, width="stretch")

    with bar_col:
        # Horizontal bar — weighted contributions
        qed_mult = row.get("QED_Multiplier", 1.0)
        qed_mult = float(qed_mult) if pd.notna(qed_mult) else 1.0
        base_score = sum(weighted_scores)

        contrib_colors = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#a855f7"]
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                y=names,
                x=weighted_scores,
                orientation="h",
                marker_color=contrib_colors,
                text=[
                    f"{w * 100:.0f}% × {s:.3f} = {ws:.3f}"
                    for w, s, ws in zip(weights, raw_scores, weighted_scores)
                ],
                textposition="auto",
                textfont=dict(size=12),
            )
        )
        fig_bar.update_layout(
            title=dict(
                text="Weighted Contributions",
                subtitle=dict(
                    text=f"Base: {base_score:.3f} × QED {qed_mult:.3f} = {base_score * qed_mult:.3f}"
                ),
            ),
            xaxis_title="Contribution to Base Score",
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(t=55, b=30, l=10, r=10),
            showlegend=False,
        )
        apply_impulator_theme(fig_bar)
        st.plotly_chart(fig_bar, width="stretch")


def _render_imp_score_analysis(df: pd.DataFrame) -> None:
    """IMP Score analysis with full explanations."""
    if df is None:
        st.info("No data available")
        return

    has_imp_score = "IMP_Final_Score" in df.columns
    has_imp = "Is_IMP_Candidate" in df.columns

    if not (has_imp_score or has_imp):
        st.info("No IMP Score analysis data available")
        return

    # IMP Score Explanation
    with st.expander("📖 What is IMP Scoring?", expanded=False):
        st.markdown("""
**IMP Score** is a multi-criteria scoring system for evaluating compound quality and IMP (Invalid Metabolic Panacea) likelihood.

**Scoring Components:**
| Component | Weight | Description |
|-----------|--------|-------------|
| **Efficiency Outlier** | 45% | How exceptional are SEI and BEI metrics? |
| **Distance to Best** | 20% | How close to the best performer? |
| **Development Angle** | 15% | Is the compound balanced? 45° is optimal |
| **Assay Interference** | 15% | Scored flags / 5 (PAINS, Aggregator, Thiol, Redox, Fluorescence). BRENK/NIH display-only. |
| **PDB Evidence** | 5% | Structural validation from crystallography |

**QED Multiplier:** `0.75 + 0.25 × QED`
- Floor at 75% (even QED=0 retains most of score)
- Maximum impact of 25% (QED=1 gives full score)

**Score Interpretation:**
The IMP score is a continuous 0–100 measure; the team has chosen not to define qualitative thresholds.

**Note on Efficiency Metrics:**
All four efficiency metrics (SEI, BEI, NSEI, NBEI) are calculated and displayed for reference.
However, only **SEI and BEI** are used in the Efficiency Outlier Score to avoid redundancy,
since NSEI and NBEI are derived from the same underlying activity data.
        """)

    # IMP Scoring
    if has_imp_score:
        st.caption(
            "IMP Score rates each **activity record** using a composite of efficiency, distance, angle, interference, PDB evidence, and QED. "
            "This is different from IMP Candidates below, which flags unique **compounds** by efficiency outlier detection."
        )

        score_cols = st.columns(4)
        scores = df["IMP_Final_Score"].dropna()

        def _name_for_score_idx(idx) -> str:
            """Look up Molecule_Name (or ChEMBL_ID fallback) for a row index."""
            if idx is None or "Molecule_Name" not in df.columns:
                fallback_col = "ChEMBL_ID" if "ChEMBL_ID" in df.columns else None
                if fallback_col is None:
                    return ""
                fallback_val = df.loc[idx].get(fallback_col, "")
                return str(fallback_val) if pd.notna(fallback_val) else ""
            name = df.loc[idx].get("Molecule_Name", "")
            if pd.notna(name) and isinstance(name, str) and name.strip():
                return name
            # Fall back to ChEMBL_ID if Molecule_Name is missing
            if "ChEMBL_ID" in df.columns:
                cid = df.loc[idx].get("ChEMBL_ID", "")
                return str(cid) if pd.notna(cid) else ""
            return ""

        with score_cols[0]:
            avg = scores.mean() if len(scores) > 0 else None
            _avg_int = format_imp_score(avg)
            st.metric("Average Score", _avg_int if _avg_int is not None else "N/A")
        with score_cols[1]:
            max_val = scores.max() if len(scores) > 0 else None
            _max_int = format_imp_score(max_val)
            _max_name = (
                _name_for_score_idx(scores.idxmax()) if len(scores) > 0 else ""
            )
            st.metric(
                "Best Score",
                _max_int if _max_int is not None else "N/A",
                delta=_max_name or None,
                delta_color="off",
                help="Highest IMP score across all activity records in this query, with the source compound name.",
            )
        with score_cols[2]:
            min_val = scores.min() if len(scores) > 0 else None
            _min_int = format_imp_score(min_val)
            _min_name = (
                _name_for_score_idx(scores.idxmin()) if len(scores) > 0 else ""
            )
            st.metric(
                "Lowest Score",
                _min_int if _min_int is not None else "N/A",
                delta=_min_name or None,
                delta_color="off",
                help="Lowest IMP score across all activity records in this query, with the source compound name.",
            )
        with score_cols[3]:
            # Higher IMP score = MORE IMP risk (worse, not better).
            # Threshold 0.5 on raw float == 50 on the new 0-100 integer scale.
            moderate_plus_imp = len(scores[scores >= 0.5]) if len(scores) > 0 else 0
            st.metric(
                "Records with IMP ≥ 50",
                moderate_plus_imp,
                help="Number of activity records with IMP Score ≥ 50 (on the 0–100 integer scale). "
                "Each compound can have many activity records, so this count is per-record, not per-compound.",
            )

        # Score histogram — full width (verbal-classification donut removed per
        # Plan 21-02; right column was just a caption, moved below the chart).
        # Single neutral color (UI-SPEC §Color). Integer-space x-axis with
        # locked floor/ceiling reference lines (PRES-07, PRES-08).
        hist_scores = df["IMP_Final_Score"].apply(format_imp_score).dropna()
        fig = px.histogram(
            x=hist_scores,
            nbins=30,
            color_discrete_sequence=["#6b7280"],
        )
        fig.add_vline(
            x=IMP_SCORE_FLOOR,
            line_dash="dot",
            line_color="#6b7280",
            annotation_text="Observed floor (10)",
            annotation_position="top",
        )
        fig.add_vline(
            x=IMP_SCORE_CEILING,
            line_dash="dot",
            line_color="#6b7280",
            annotation_text="Observed ceiling (80)",
            annotation_position="top right",
        )
        fig.update_layout(
            title=dict(text="IMP Score Distribution"),
            height=360,
            margin=dict(t=55, b=30, l=30, r=10),
            xaxis_title="IMP Score",
            yaxis_title="Count",
            xaxis=dict(range=[0, 100]),
            bargap=0.05,
            showlegend=False,
        )
        apply_impulator_theme(fig)
        st.plotly_chart(fig, width="stretch")

        # Locked neutral caption (UI-SPEC §Copywriting Contract; Plan 21-02 Site B2).
        st.caption(
            "The IMP score is a continuous 0–100 measure; the team has chosen not to define qualitative thresholds."
        )

        # Detailed Score Breakdown Section
        _render_imp_score_breakdown(df)

        # Phase 22: GMM Cluster Membership Section
        job_id = SessionState.get("selected_compound_entry_id", "default") or "default"
        _render_gmm_widget(df, job_id)

    # IMP Candidates section
    if has_imp:
        st.markdown("---")
        st.markdown("**IMP Candidates Analysis**")
        st.caption(
            "Counts unique **compounds** with ≥2 efficiency metric outliers (z-score). "
            "This is different from IMP Score above, which scores individual activity records using a composite formula."
        )

        # Get IMP candidate records and unique compounds
        imp_df = df[df["Is_IMP_Candidate"]]
        unique_imp_compounds = (
            imp_df.drop_duplicates("ChEMBL_ID")
            if "ChEMBL_ID" in imp_df.columns
            else imp_df
        )
        total_unique = (
            df.drop_duplicates("ChEMBL_ID")["ChEMBL_ID"].nunique()
            if "ChEMBL_ID" in df.columns
            else len(df)
        )

        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric(
                "IMP Candidates",
                len(unique_imp_compounds),
                help="Unique compounds with ≥2 efficiency metrics flagged as statistical outliers (z-score detection on SEI, BEI, NSEI, NBEI)",
            )
        with info_cols[1]:
            st.metric(
                "Total Compounds",
                total_unique,
                help="Total unique compounds in this analysis",
            )
        with info_cols[2]:
            # % IMP sourced from Is_IMP_Candidate boolean (same source the donut uses).
            pct = (
                len(unique_imp_compounds) / total_unique * 100
                if total_unique > 0
                else 0
            )
            st.metric("% IMP", f"{pct:.1f}%")
        with info_cols[3]:
            # Show affected records (activity rows from IMP compounds)
            st.metric(
                "Affected Records",
                len(imp_df),
                help="Activity records from IMP compounds",
            )

        if not unique_imp_compounds.empty:
            st.markdown("**IMP Candidates with Target Mapping:**")

            # Build display table - one row per compound+target combination
            display_data = []
            for _, row in unique_imp_compounds.iterrows():
                chembl_id = row.get("ChEMBL_ID", "Unknown")
                mol_name = row.get("Molecule_Name", "")
                if pd.isna(mol_name) or not isinstance(mol_name, str):
                    mol_name = ""

                # Get all records for this compound to find targets
                compound_records = (
                    imp_df[imp_df["ChEMBL_ID"] == chembl_id]
                    if "ChEMBL_ID" in imp_df.columns
                    else pd.DataFrame()
                )

                # Per-compound fallback IMP score (max across all this compound's records;
                # matches the "Best Score" widget semantic). Used when target grouping is
                # unavailable. Inside the target loop we override with per-target max.
                _compound_imp_max = (
                    compound_records["IMP_Final_Score"].max()
                    if "IMP_Final_Score" in compound_records.columns
                    and not compound_records["IMP_Final_Score"].isna().all()
                    else None
                )
                _compound_imp_int = format_imp_score(_compound_imp_max)
                compound_imp_display = (
                    _compound_imp_int if _compound_imp_int is not None else "N/A"
                )

                # Check for target columns
                has_target_name = "Target_Name" in compound_records.columns
                has_target_id = "Target_ChEMBL_ID" in compound_records.columns

                if has_target_id or has_target_name:
                    # Get unique target IDs (prefer ID for grouping)
                    target_id_col = (
                        "Target_ChEMBL_ID" if has_target_id else "Target_Name"
                    )
                    target_ids = compound_records[target_id_col].dropna().unique()

                    if len(target_ids) > 0:
                        for target_id in target_ids:
                            target_records = compound_records[
                                compound_records[target_id_col] == target_id
                            ]
                            # Get average activity for this compound-target pair
                            avg_activity = (
                                target_records["pActivity"].mean()
                                if "pActivity" in target_records.columns
                                else None
                            )
                            # Per-target IMP score: max IMP_Final_Score across this
                            # (compound, target) group's activity records — matches
                            # the "Best Score" widget at the top, but scoped to the
                            # group so each table row reflects its own data.
                            _target_imp_max = (
                                target_records["IMP_Final_Score"].max()
                                if "IMP_Final_Score" in target_records.columns
                                and not target_records["IMP_Final_Score"].isna().all()
                                else None
                            )
                            _target_imp_int = format_imp_score(_target_imp_max)
                            target_imp_display = (
                                _target_imp_int
                                if _target_imp_int is not None
                                else "N/A"
                            )

                            # Get target name if available
                            target_name = ""
                            if has_target_name:
                                names = target_records["Target_Name"].dropna().unique()
                                target_name = (
                                    str(names[0])[:35] if len(names) > 0 else ""
                                )

                            # Get target ChEMBL ID for link
                            target_chembl_id = ""
                            target_link = ""
                            if has_target_id:
                                ids = (
                                    target_records["Target_ChEMBL_ID"].dropna().unique()
                                )
                                if len(ids) > 0:
                                    target_chembl_id = str(ids[0])
                                    target_link = f"https://www.ebi.ac.uk/chembl/explore/target/{target_chembl_id}"

                            display_data.append(
                                {
                                    "ChEMBL_ID": chembl_id,
                                    "Molecule": mol_name[:20] if mol_name else "",
                                    "Target": target_name
                                    if target_name
                                    else str(target_id)[:35],
                                    "Target_Link": target_link,
                                    "Avg_pActivity": f"{avg_activity:.2f}"
                                    if pd.notna(avg_activity)
                                    else "N/A",
                                    "IMP Score": target_imp_display,
                                    "Records": len(target_records),
                                }
                            )
                    else:
                        # No targets found — fall back to compound-level max
                        display_data.append(
                            {
                                "ChEMBL_ID": chembl_id,
                                "Molecule": mol_name[:20] if mol_name else "",
                                "Target": "N/A",
                                "Target_Link": "",
                                "Avg_pActivity": "N/A",
                                "IMP Score": compound_imp_display,
                                "Records": len(compound_records),
                            }
                        )
                else:
                    # No target column — fall back to compound-level max
                    display_data.append(
                        {
                            "ChEMBL_ID": chembl_id,
                            "Molecule": mol_name[:20] if mol_name else "",
                            "Target": "N/A",
                            "Target_Link": "",
                            "Avg_pActivity": "N/A",
                            "IMP Score": compound_imp_display,
                            "Records": len(compound_records),
                        }
                    )

            # Sort by IMP Score descending — N/A rows go to the bottom.
            # IMP Score is either an int (from format_imp_score) or "N/A" string.
            display_data.sort(
                key=lambda d: d["IMP Score"] if isinstance(d["IMP Score"], int) else -1,
                reverse=True,
            )
            imp_table = pd.DataFrame(display_data)
            st.dataframe(
                imp_table,
                column_config={
                    "ChEMBL_ID": st.column_config.TextColumn(
                        "ChEMBL ID", width="small"
                    ),
                    "Molecule": st.column_config.TextColumn("Molecule", width="small"),
                    "Target": st.column_config.TextColumn(
                        "Target Name", width="medium"
                    ),
                    "Target_Link": st.column_config.LinkColumn(
                        "Target ChEMBL ID",
                        display_text=r"https://www\.ebi\.ac\.uk/chembl/explore/target/(CHEMBL\d+)",
                        width="small",
                    ),
                    "Avg_pActivity": st.column_config.TextColumn(
                        "Avg pActivity", width="small"
                    ),
                    "IMP Score": st.column_config.TextColumn(
                        "IMP Score", width="small"
                    ),
                    "Records": st.column_config.NumberColumn("Records", width="small"),
                },
                hide_index=True,
                height=min(500, len(imp_table) * 35 + 40),
            )

            st.caption(
                "💡 **Note:** IMP candidates may still be valid if they have high PDB structural evidence. Cross-reference with the PDB Evidence tab."
            )
        else:
            st.success(
                "✓ No IMP candidates detected - all compounds show normal activity patterns"
            )


# =============================================================================
# VISUALIZATIONS TAB
# =============================================================================


def _render_visualizations_tab(data: dict[str, Any]) -> None:
    """Interactive visualizations."""
    df = data.get("results")

    if df is None or df.empty:
        st.warning("No data available for visualization")
        return

    # Plot type selector — Activity Distribution moved to Activity tab.
    # label_visibility="collapsed" removes the label slot entirely;
    # "hidden" would leave it as invisible whitespace above the radio.
    plot_type = st.radio(
        "Select Plot",
        ["Efficiency Scatter", "Efficiency by Compound", "Custom Plot"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if plot_type == "Efficiency Scatter":
        _plot_efficiency_scatter(df)
    elif plot_type == "Efficiency by Compound":
        _plot_efficiency_by_compound(df)
    elif plot_type == "Custom Plot":
        _plot_custom(df)


def _plot_activity_distribution(df: pd.DataFrame) -> None:
    """Activity distribution box plot with interactive legend and structure viewer."""
    if "Activity_Type" not in df.columns or "pActivity" not in df.columns:
        st.info("Activity columns not available")
        return

    plot_df = df.copy()

    # Build customdata for structure viewer
    customdata_cols = []
    if "SMILES" in plot_df.columns:
        customdata_cols.append("SMILES")
        if "Molecule_Name" in plot_df.columns:
            customdata_cols.append("Molecule_Name")
        if "ChEMBL_ID" in plot_df.columns:
            customdata_cols.append("ChEMBL_ID")

    fig = px.box(
        plot_df,
        x="Activity_Type",
        y="pActivity",
        color="Activity_Type",
        points="all",  # Show all points for structure viewer clicks
        hover_data=["ChEMBL_ID", "Molecule_Name"]
        if all(c in plot_df.columns for c in ["ChEMBL_ID", "Molecule_Name"])
        else None,
        custom_data=customdata_cols if customdata_cols else None,
    )
    theme = get_plotly_theme()
    fig.update_layout(
        title=dict(
            text="Bioactivity Distribution",
            subtitle=dict(text="pActivity = -log10(M) — higher = more potent"),
        ),
        template=theme["template"],
        height=520,
        showlegend=True,
        yaxis=dict(exponentformat="SI"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            title_text="",
            bgcolor=theme["legend_bgcolor"],
            bordercolor=theme["legend_bordercolor"],
            borderwidth=1,
        ),
    )
    apply_impulator_theme(fig)
    st.plotly_chart(fig, width="stretch", height=400, key="activity_dist_chart")
    st.caption(
        "💡 **Click legend items** to show/hide activity types. Double-click to isolate."
    )

    _maybe_embed_structure_viewer(
        "activity_dist_chart", plot_df, x_col="Activity_Type", y_col="pActivity"
    )


def _plot_efficiency_scatter(df: pd.DataFrame) -> None:
    """Efficiency scatter plot with full controls and structure viewer."""
    # Get all columns for color/size options
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Filter out internal columns
    numeric_cols = [
        c for c in numeric_cols if not (c.startswith("Is_") and c.endswith("_Outlier"))
    ]
    categorical_cols = [
        c for c in categorical_cols if c not in ["SMILES", "Direct_Parent"]
    ]

    # All columns for color (categorical first, then numeric)
    all_color_cols = categorical_cols + numeric_cols

    # Row 1: data selection (Plot / Color / Size) + colour-scale options.
    row1 = st.columns([1, 1, 1, 1])

    with row1[0]:
        plot_choice = st.selectbox(
            "Plot", ["SEI vs BEI", "NSEI vs NBEI"], key="scatter_choice"
        )

    with row1[1]:
        color_by = st.selectbox(
            "Color by", ["None"] + all_color_cols, key="scatter_color"
        )

    with row1[2]:
        size_by = st.selectbox("Size by", ["None"] + numeric_cols, key="scatter_size")

    # Numeric colour → gradient controls live where they're contextually
    # relevant (right of "Color by"). When colour isn't numeric, this slot
    # stays empty rather than pushing other controls around.
    is_numeric_color = color_by != "None" and color_by in numeric_cols

    with row1[3]:
        if is_numeric_color:
            color_scale = st.selectbox(
                "Color Scale",
                [
                    "Viridis",
                    "Plasma",
                    "Inferno",
                    "Turbo",
                    "Blues",
                    "Reds",
                    "RdBu",
                    "Spectral",
                ],
                key="scatter_colorscale",
            )
            reverse_scale = st.checkbox(
                "Reverse Scale", value=False, key="scatter_reverse"
            )
        else:
            color_scale = "Viridis"
            reverse_scale = False

    # Row 2: analytical overlays + visual tweaks.
    row2 = st.columns([1, 1, 1, 1])

    with row2[0]:
        show_trendline = st.checkbox("Trendline", value=False, key="scatter_trendline")

    with row2[1]:
        if plot_choice == "SEI vs BEI":
            show_balance = st.checkbox(
                "Balance line",
                value=False,
                key="scatter_balance",
                help="Diagonal where BEI = SEI, i.e. 10·PSA/MW = 1. "
                "Above the line: polar-favored (10·PSA/MW > 1). "
                "Below: lipophilic-favored.",
            )
        else:
            show_balance = False

    with row2[2]:
        opacity = st.slider("Opacity", 0.3, 1.0, 0.7, key="scatter_opacity")

    with row2[3]:
        point_size = st.slider("Base Size", 5, 20, 10, key="scatter_pointsize")

    st.markdown("---")

    x_col, y_col = ("SEI", "BEI") if plot_choice == "SEI vs BEI" else ("NSEI", "NBEI")

    if x_col not in df.columns or y_col not in df.columns:
        st.warning(f"Columns {x_col} or {y_col} not available")
        return

    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    if plot_df.empty:
        st.warning("No valid data for plotting")
        return

    is_categorical_color = color_by != "None" and color_by not in numeric_cols

    # Build customdata for structure viewer (SMILES first, then name, then index)
    if "SMILES" in plot_df.columns:
        customdata_cols = ["SMILES"]
        if "Molecule_Name" in plot_df.columns:
            customdata_cols.append("Molecule_Name")
        if "ChEMBL_ID" in plot_df.columns:
            customdata_cols.append("ChEMBL_ID")
        plot_df["_row_idx"] = range(len(plot_df))
        customdata_cols.append("_row_idx")
    else:
        customdata_cols = None

    # Build scatter plot
    scatter_args = {
        "x": x_col,
        "y": y_col,
        "opacity": opacity,
        "hover_data": ["ChEMBL_ID", "Molecule_Name"]
        if all(c in plot_df.columns for c in ["ChEMBL_ID", "Molecule_Name"])
        else None,
    }

    # Add customdata for structure viewer
    if customdata_cols:
        scatter_args["custom_data"] = customdata_cols

    # Trendline (per-group when colored, single when not)
    if show_trendline:
        scatter_args["trendline"] = "ols"

    # Color handling
    if color_by != "None":
        scatter_args["color"] = color_by
        if is_numeric_color:
            scatter_args["color_continuous_scale"] = (
                color_scale if not reverse_scale else f"{color_scale}_r"
            )

    # Size handling
    if size_by != "None" and size_by in plot_df.columns:
        scatter_args["size"] = size_by
        scatter_args["size_max"] = point_size * 2

    fig = px.scatter(plot_df, **scatter_args)

    # Update marker size if no size_by
    if size_by == "None":
        fig.update_traces(marker=dict(size=point_size))

    # Balance reference line (BEI = SEI ⇔ 10·PSA/MW = 1).
    # Spans the union of x/y ranges so the diagonal is visible even when
    # all points sit on one side of the balance.
    if show_balance:
        lo = float(min(plot_df[x_col].min(), plot_df[y_col].min()))
        hi = float(max(plot_df[x_col].max(), plot_df[y_col].max()))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        fig.add_trace(
            go.Scatter(
                x=[lo - pad, hi + pad],
                y=[lo - pad, hi + pad],
                mode="lines",
                line=dict(color="#8a8a8a", width=1.5, dash="dash"),
                name="Balance: 10·PSA/MW = 1",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Layout with meta for legend monitor identification
    theme = get_plotly_theme()
    fig.update_layout(
        template=theme["template"],
        height=520,
        meta="eff_scatter",
        showlegend=(color_by != "None" and not is_numeric_color) or show_balance,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            title_text="",
            bgcolor=theme["legend_bgcolor"],
            bordercolor=theme["legend_bordercolor"],
            borderwidth=1,
        ),
    )

    # Interactive stats bar above chart — computes regression client-side
    # via jStat, updates instantly on legend clicks (no Python round-trip).
    # JS polls for the chart so it's safe to render before st.plotly_chart().
    if show_trendline and len(plot_df) >= 2:
        x_vals = plot_df[x_col].values
        y_vals = plot_df[y_col].values
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            x_vals, y_vals
        )
        initial = {
            "r2": r_value**2,
            "slope": slope,
            "intercept": intercept,
            "p": p_value,
            "n": int(len(x_vals)),
        }
        plotly_legend_monitor(
            chart_meta="eff_scatter",
            key=f"eff_legend_{color_by}",
            initial_stats=initial,
            x_col=x_col,
            y_col=y_col,
        )
        if is_categorical_color:
            st.caption(
                "Click legend items to show/hide groups — stats update instantly."
            )

    apply_impulator_theme(fig)
    st.plotly_chart(fig, width="stretch", key="efficiency_scatter_chart")

    # Embed structure viewer for click-to-view molecules
    _maybe_embed_structure_viewer(
        "efficiency_scatter_chart", plot_df, x_col=x_col, y_col=y_col
    )


def _plot_efficiency_by_compound(df: pd.DataFrame) -> None:
    """Grouped efficiency boxplots."""
    col1, col2 = st.columns([1, 1])

    with col1:
        metric = st.selectbox(
            "Metric", ["SEI", "BEI", "NSEI", "NBEI"], key="box_metric"
        )
    with col2:
        group_size = st.slider("Compounds per view", 3, 10, 5, key="group_size")

    if metric not in df.columns or "ChEMBL_ID" not in df.columns:
        st.warning("Required columns not available")
        return

    unique_ids = df["ChEMBL_ID"].unique()
    num_groups = max(1, (len(unique_ids) + group_size - 1) // group_size)

    group_num = st.number_input("Group", 1, num_groups, 1, key="group_num")
    start = (group_num - 1) * group_size
    group_ids = unique_ids[start : start + group_size]

    group_df = df[df["ChEMBL_ID"].isin(group_ids)].dropna(subset=[metric])

    if not group_df.empty:
        eby_cd = [
            c for c in ["SMILES", "Molecule_Name", "ChEMBL_ID"] if c in group_df.columns
        ]
        fig = px.box(
            group_df,
            x="ChEMBL_ID",
            y=metric,
            color="ChEMBL_ID",
            points="all",
            custom_data=eby_cd if eby_cd else None,
        )
        fig.update_layout(height=450, xaxis_tickangle=-45, showlegend=False)
        apply_impulator_theme(fig)
        st.plotly_chart(fig, width="stretch", key="eff_by_compound_box")
        _maybe_embed_structure_viewer(
            "eff_by_compound_box", group_df, x_col="ChEMBL_ID", y_col=metric
        )
        st.caption(
            f"Group {group_num} of {num_groups} ({len(unique_ids)} total compounds)"
        )


def _plot_custom(df: pd.DataFrame) -> None:
    """Fully customizable plot - users can select X, Y, color, plot type."""
    st.markdown("**🎨 Custom Visualization**")
    st.caption("Create your own plots by selecting axes and options")

    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Filter out internal columns
    numeric_cols = [
        c for c in numeric_cols if not c.startswith("Is_") or not c.endswith("_Outlier")
    ]
    categorical_cols = [
        c for c in categorical_cols if c not in ["SMILES", "Direct_Parent"]
    ]

    if not numeric_cols:
        st.warning("No numeric columns available for plotting")
        return

    # Control row 1: Plot type and axes
    ctrl_row1 = st.columns([1, 1, 1, 1])

    with ctrl_row1[0]:
        plot_type = st.selectbox(
            "Plot Type",
            ["Scatter", "Box", "Histogram", "Violin"],
            key="custom_plot_type",
        )

    with ctrl_row1[1]:
        # X axis - for histogram/box can also be categorical
        x_options = numeric_cols + (
            categorical_cols if plot_type in ["Box", "Violin"] else []
        )
        x_axis = st.selectbox("X Axis", x_options, key="custom_x")

    with ctrl_row1[2]:
        if plot_type in ["Scatter", "Box", "Violin"]:
            y_options = numeric_cols
            y_axis = st.selectbox("Y Axis", y_options, key="custom_y")
        else:
            y_axis = None

    with ctrl_row1[3]:
        color_options = (
            ["None"]
            + categorical_cols
            + [c for c in numeric_cols if df[c].nunique() < 20]
        )
        color_by = st.selectbox("Color By", color_options, key="custom_color")

    # Control row 2: Additional options
    ctrl_row2 = st.columns([1, 1, 1, 1])

    with ctrl_row2[0]:
        if plot_type == "Scatter":
            show_trendline = st.checkbox(
                "Trendline", value=False, key="custom_trendline"
            )
            show_identity = st.checkbox(
                "y = x reference",
                value=False,
                key="custom_identity",
                help="Diagonal y = x reference line. "
                "Most useful when X and Y share the same units/scale.",
            )
        else:
            show_trendline = False
            show_identity = False

    with ctrl_row2[1]:
        if plot_type == "Scatter":
            size_by = st.selectbox(
                "Size by",
                ["None"] + numeric_cols + categorical_cols,
                key="custom_size_by",
            )
        elif plot_type == "Histogram":
            nbins = st.slider("Bins", 10, 50, 30, key="custom_bins")
            size_by = "None"
        else:
            size_by = "None"
            nbins = 30

    with ctrl_row2[2]:
        if plot_type == "Scatter":
            point_size = st.slider("Point Size", 3, 15, 8, key="custom_size")
        else:
            point_size = 8

    with ctrl_row2[3]:
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7, key="custom_opacity")

    st.markdown("---")

    # Prepare data
    if y_axis:
        plot_df = df.dropna(subset=[x_axis, y_axis]).copy()
    else:
        plot_df = df.dropna(subset=[x_axis]).copy()

    if plot_df.empty:
        st.warning("No valid data for selected columns")
        return

    is_custom_categorical = color_by != "None" and color_by in categorical_cols

    # Build customdata for structure viewer (for scatter plots)
    customdata_cols = None
    if plot_type == "Scatter" and "SMILES" in plot_df.columns:
        customdata_cols = ["SMILES"]
        if "Molecule_Name" in plot_df.columns:
            customdata_cols.append("Molecule_Name")
        if "ChEMBL_ID" in plot_df.columns:
            customdata_cols.append("ChEMBL_ID")

    # Create plot based on type
    try:
        if plot_type == "Scatter":
            scatter_kw = dict(
                x=x_axis,
                y=y_axis,
                color=color_by if color_by != "None" else None,
                hover_data=["ChEMBL_ID", "Molecule_Name"]
                if all(c in plot_df.columns for c in ["ChEMBL_ID", "Molecule_Name"])
                else None,
                trendline="ols" if show_trendline else None,
                opacity=opacity,
                custom_data=customdata_cols,
            )
            if size_by != "None" and size_by in plot_df.columns:
                if size_by in categorical_cols:
                    # Convert categorical → group frequency for sizing
                    plot_df["_size_num"] = plot_df.groupby(size_by)[size_by].transform(
                        "count"
                    )
                    scatter_kw["size"] = "_size_num"
                else:
                    scatter_kw["size"] = size_by
                scatter_kw["size_max"] = point_size * 2
            fig = px.scatter(plot_df, **scatter_kw)
            if size_by == "None":
                fig.update_traces(marker=dict(size=point_size))

            # y = x reference line. Spans the union of x/y ranges so it
            # remains visible even when X and Y are on different scales.
            if show_identity and y_axis:
                lo = float(min(plot_df[x_axis].min(), plot_df[y_axis].min()))
                hi = float(max(plot_df[x_axis].max(), plot_df[y_axis].max()))
                pad = (hi - lo) * 0.05 if hi > lo else 1.0
                fig.add_trace(
                    go.Scatter(
                        x=[lo - pad, hi + pad],
                        y=[lo - pad, hi + pad],
                        mode="lines",
                        line=dict(color="#8a8a8a", width=1.5, dash="dash"),
                        name="y = x",
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )

        elif plot_type == "Box":
            # Build customdata for box plots too
            box_customdata = None
            if "SMILES" in plot_df.columns:
                box_customdata = ["SMILES"]
                if "Molecule_Name" in plot_df.columns:
                    box_customdata.append("Molecule_Name")
                if "ChEMBL_ID" in plot_df.columns:
                    box_customdata.append("ChEMBL_ID")

            fig = px.box(
                plot_df,
                x=x_axis,
                y=y_axis,
                color=color_by if color_by != "None" else None,
                points="all",  # Show all points for structure viewer clicks
                custom_data=box_customdata,
            )

        elif plot_type == "Violin":
            # Build customdata for violin plots too
            violin_customdata = None
            if "SMILES" in plot_df.columns:
                violin_customdata = ["SMILES"]
                if "Molecule_Name" in plot_df.columns:
                    violin_customdata.append("Molecule_Name")
                if "ChEMBL_ID" in plot_df.columns:
                    violin_customdata.append("ChEMBL_ID")

            fig = px.violin(
                plot_df,
                x=x_axis,
                y=y_axis,
                color=color_by if color_by != "None" else None,
                box=True,
                points="all",  # Show all points for structure viewer clicks
                custom_data=violin_customdata,
            )

        elif plot_type == "Histogram":
            fig = px.histogram(
                plot_df,
                x=x_axis,
                color=color_by if color_by != "None" else None,
                nbins=nbins,
                opacity=opacity,
            )

        # Common layout updates with meta for legend monitor
        theme = get_plotly_theme()
        fig.update_layout(
            template=theme["template"],
            height=550,
            meta="custom_scatter",
            showlegend=(color_by != "None") or show_identity,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                title_text="",
                bgcolor=theme["legend_bgcolor"],
                bordercolor=theme["legend_bordercolor"],
                borderwidth=1,
            ),
        )

        # Interactive stats bar above chart
        if plot_type == "Scatter" and show_trendline and y_axis and len(plot_df) >= 2:
            x_vals = plot_df[x_axis].values
            y_vals = plot_df[y_axis].values
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                x_vals, y_vals
            )
            initial = {
                "r2": r_value**2,
                "slope": slope,
                "intercept": intercept,
                "p": p_value,
                "n": int(len(x_vals)),
            }
            plotly_legend_monitor(
                chart_meta="custom_scatter",
                key=f"custom_legend_{color_by}",
                initial_stats=initial,
                x_col=x_axis,
                y_col=y_axis,
            )
            if is_custom_categorical:
                st.caption(
                    "Click legend items to show/hide groups — stats update instantly."
                )

        apply_impulator_theme(fig)
        st.plotly_chart(fig, width="stretch", key="custom_plot_chart")

        # Embed structure viewer for hover-to-view molecules (for scatter, box, violin)
        if plot_type in ["Scatter", "Box", "Violin"]:
            _maybe_embed_structure_viewer(
                "custom_plot_chart",
                plot_df,
                x_col=x_axis,
                y_col=y_axis if y_axis else x_axis,
            )

    except Exception as e:
        st.error(f"Error creating plot: {e}")


# =============================================================================
# STRUCTURES TAB - Molecule Viewer
# =============================================================================


def _render_structures_tab(data: dict[str, Any]) -> None:
    """Molecular structures viewer (2D/3D)."""
    df = data.get("results")
    _render_molecule_viewer(df)


@st.dialog("Molecule Structure", width="large")
def _show_expanded_structure(smiles: str, label: str = "") -> None:
    """Expanded molecule view with atom numbers and molecular properties."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES — cannot render")
            return

        drawer = rdMolDraw2D.MolDraw2DSVG(700, 450)
        opts = drawer.drawOptions()
        opts.addAtomIndices = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        if label:
            st.markdown(f"**{html.escape(label)}**")

        st.markdown(
            f'<div style="display:flex;justify-content:center;background:#fff;'
            f'border-radius:8px;padding:12px;">{svg}</div>',
            unsafe_allow_html=True,
        )

        isomeric = Chem.MolToSmiles(mol, isomericSmiles=True)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = Descriptors.ExactMolWt(mol)
        atoms = mol.GetNumAtoms()
        bonds = mol.GetNumBonds()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Formula", formula)
        c2.metric("Mol. Weight", f"{mw:.2f}")
        c3.metric("Atoms", atoms)
        c4.metric("Bonds", bonds)

        st.markdown("**Isomeric SMILES**")
        st.code(isomeric, language=None)
    except ImportError:
        st.warning("RDKit not available for expanded view")
    except Exception as e:
        st.error(f"Render error: {e}")


def _render_3d_viewer(smiles: str, entry_id: str) -> None:
    """Interactive 3D molecule viewer using 3Dmol.js."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            st.caption("Could not parse SMILES for 3D rendering")
            return

        mol_3d = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if result == -1:
            AllChem.EmbedMolecule(mol_3d, useRandomCoords=True, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
        pdb_block = Chem.MolToPDBBlock(mol_3d)

    except Exception as e:
        st.caption(f"3D generation failed: {e}")
        return

    safe_pdb = pdb_block.replace("`", "\\`").replace("${", "\\${")

    viewer_html = f"""
    <style>
        .viewer3d-wrapper {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
        .viewer3d-wrapper * {{ box-sizing:border-box; }}
        #viewer3d {{ width:100%; height:380px; position:relative; background:#fff; border-radius:8px 8px 0 0; }}
        .viewer3d-wrapper #controls {{ display:flex; gap:6px; padding:8px 12px; align-items:center; flex-wrap:wrap;
            background:#1a1a2e; border-radius:0 0 8px 8px; position:relative; z-index:10; }}
        .viewer3d-wrapper .btn {{ padding:6px 12px; border-radius:6px; border:1px solid rgba(255,255,255,0.2);
            background:rgba(255,255,255,0.1); color:#eee; font-size:13px; cursor:pointer;
            outline:none; user-select:none; display:inline-block; }}
        .viewer3d-wrapper .btn:hover {{ background:rgba(255,255,255,0.18); }}
        .viewer3d-wrapper .btn.active {{ background:rgba(99,102,241,0.5); border-color:rgba(99,102,241,0.7); color:#fff; }}
        .viewer3d-wrapper .btn-group {{ position:relative; }}
        .viewer3d-wrapper .btn-group .menu {{ display:none; position:absolute; bottom:100%; left:0; margin-bottom:4px;
            background:#2a2a3e; border:1px solid rgba(255,255,255,0.15); border-radius:6px;
            overflow:hidden; min-width:130px; z-index:100; box-shadow:0 4px 12px rgba(0,0,0,0.4); }}
        .viewer3d-wrapper .btn-group.open .menu {{ display:block; }}
        .viewer3d-wrapper .menu-item {{ padding:8px 14px; color:#ddd; font-size:13px; cursor:pointer; }}
        .viewer3d-wrapper .menu-item:hover {{ background:rgba(255,255,255,0.1); }}
        .viewer3d-wrapper .menu-item.selected {{ background:rgba(99,102,241,0.3); }}
        .viewer3d-wrapper .lbl {{ font-size:11px; color:#999; text-transform:uppercase; letter-spacing:0.5px; }}
    </style>
    <div class="viewer3d-wrapper">
    <div id="viewer3d"></div>
    <div id="controls">
        <span class="lbl">Style</span>
        <div class="btn-group" id="styleGroup">
            <div class="btn" id="styleBtn">Ball &amp; Stick ▾</div>
            <div class="menu">
                <div class="menu-item selected" data-val="ballstick">Ball &amp; Stick</div>
                <div class="menu-item" data-val="stick">Stick</div>
                <div class="menu-item" data-val="sphere">Sphere</div>
                <div class="menu-item" data-val="line">Line</div>
            </div>
        </div>
        <span class="lbl">Color</span>
        <div class="btn-group" id="colorGroup">
            <div class="btn" id="colorBtn">Element (Jmol) ▾</div>
            <div class="menu">
                <div class="menu-item selected" data-val="default">Element (Jmol)</div>
                <div class="menu-item" data-val="greenCarbon">Green Carbon</div>
                <div class="menu-item" data-val="cyanCarbon">Cyan Carbon</div>
                <div class="menu-item" data-val="spectrum">Spectrum</div>
            </div>
        </div>
        <div class="btn" id="spinBtn">Spin</div>
        <div class="btn" id="resetBtn">Reset</div>
    </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
    <script>
    (function() {{
        var wrapper = document.querySelector(".viewer3d-wrapper");
        var container = wrapper.querySelector("#viewer3d");

        function initViewer() {{
            var viewer = $3Dmol.createViewer(container, {{
                backgroundColor: "white", antialias: true
            }});
            var pdb = `{safe_pdb}`;
            viewer.addModel(pdb, "pdb");

            var styles = {{
                ballstick: {{stick: {{radius: 0.12}}, sphere: {{scale: 0.25}}}},
                stick: {{stick: {{radius: 0.15}}}},
                sphere: {{sphere: {{scale: 0.4}}}},
                line: {{line: {{}}}}
            }};

            var curStyle = "ballstick";
            var curColor = "default";

            // Spectrum coloring: rainbow by atom index (red → green → blue)
            var totalAtoms = viewer.getModel().selectedAtoms({{}}).length || 1;
            function spectrumColor(atom) {{
                var frac = (atom.serial || 0) / totalAtoms;
                var r, g, b;
                if (frac < 0.25)      {{ r = 1; g = frac * 4; b = 0; }}
                else if (frac < 0.5)  {{ r = 1 - (frac - 0.25) * 4; g = 1; b = 0; }}
                else if (frac < 0.75) {{ r = 0; g = 1; b = (frac - 0.5) * 4; }}
                else                  {{ r = 0; g = 1 - (frac - 0.75) * 4; b = 1; }}
                return "rgb(" + Math.round(r * 255) + "," + Math.round(g * 255) + "," + Math.round(b * 255) + ")";
            }}

            function applyStyle() {{
                var styleObj = JSON.parse(JSON.stringify(styles[curStyle] || styles.ballstick));
                if (curColor === "spectrum") {{
                    Object.keys(styleObj).forEach(function(k) {{ styleObj[k].colorfunc = spectrumColor; }});
                }} else if (curColor !== "default") {{
                    // Named 3Dmol schemes: greenCarbon, cyanCarbon, etc.
                    Object.keys(styleObj).forEach(function(k) {{ styleObj[k].colorscheme = curColor; }});
                }}
                viewer.setStyle({{}}, styleObj);
                viewer.render();
            }}

            applyStyle();
            viewer.zoomTo();
            viewer.render();

            // Save initial view state (position + rotation) for reset
            var initialView = viewer.getView();

            // Hover labels
            viewer.setHoverable({{}}, true,
                function(atom, viewer, event, container) {{
                    if (!atom.label) {{
                        atom.label = viewer.addLabel(
                            atom.elem + atom.serial,
                            {{position: atom, backgroundColor: "rgba(0,0,0,0.75)",
                             fontColor: "white", fontSize: 12, borderRadius: 4, padding: 4}});
                    }}
                }},
                function(atom, viewer) {{
                    if (atom.label) {{ viewer.removeLabel(atom.label); delete atom.label; }}
                }}
            );
            viewer.render();

            // Custom dropdown logic — all queries scoped to wrapper
            function setupDropdown(groupId, btnId, onSelect) {{
                var group = wrapper.querySelector("#" + groupId);
                var btn = wrapper.querySelector("#" + btnId);
                btn.addEventListener("click", function(e) {{
                    e.stopPropagation();
                    wrapper.querySelectorAll(".btn-group.open").forEach(function(g) {{
                        if (g !== group) g.classList.remove("open");
                    }});
                    group.classList.toggle("open");
                }});
                group.querySelectorAll(".menu-item").forEach(function(item) {{
                    item.addEventListener("click", function(e) {{
                        e.stopPropagation();
                        group.querySelectorAll(".menu-item").forEach(function(i) {{ i.classList.remove("selected"); }});
                        item.classList.add("selected");
                        btn.textContent = item.textContent + " \\u25BE";
                        group.classList.remove("open");
                        onSelect(item.getAttribute("data-val"));
                    }});
                }});
            }}

            setupDropdown("styleGroup", "styleBtn", function(val) {{
                curStyle = val;
                applyStyle();
            }});
            setupDropdown("colorGroup", "colorBtn", function(val) {{
                curColor = val;
                applyStyle();
            }});

            // Close dropdowns on click outside
            document.addEventListener("click", function() {{
                wrapper.querySelectorAll(".btn-group.open").forEach(function(g) {{
                    g.classList.remove("open");
                }});
            }});

            // Spin
            var spinning = false;
            var spinBtn = wrapper.querySelector("#spinBtn");
            spinBtn.addEventListener("click", function() {{
                spinning = !spinning;
                viewer.spin(spinning);
                spinBtn.classList.toggle("active", spinning);
            }});

            // Reset — restore full view state (position + rotation + zoom)
            wrapper.querySelector("#resetBtn").addEventListener("click", function() {{
                spinning = false;
                viewer.spin(false);
                spinBtn.classList.remove("active");
                curStyle = "ballstick";
                curColor = "default";
                wrapper.querySelector("#styleBtn").textContent = "Ball & Stick \\u25BE";
                wrapper.querySelector("#colorBtn").textContent = "Element (Jmol) \\u25BE";
                wrapper.querySelectorAll(".menu-item").forEach(function(i) {{ i.classList.remove("selected"); }});
                wrapper.querySelector('[data-val="ballstick"]').classList.add("selected");
                wrapper.querySelector('[data-val="default"]').classList.add("selected");
                applyStyle();
                viewer.setView(initialView);
                viewer.render();
            }});
        }}

        // Ensure container is laid out before initializing 3Dmol
        if (container.offsetHeight > 0) {{
            initViewer();
        }} else {{
            var checkReady = setInterval(function() {{
                if (container.offsetHeight > 0) {{
                    clearInterval(checkReady);
                    initViewer();
                }}
            }}, 50);
        }}
    }})();
    </script>
    """
    st.html(viewer_html, unsafe_allow_javascript=True)


def _render_molecule_viewer(df: pd.DataFrame) -> None:
    """2D/3D molecule viewer."""
    if df is None or "SMILES" not in df.columns:
        st.warning("No SMILES data available")
        return

    # Get unique molecules
    id_col = "ChEMBL_ID" if "ChEMBL_ID" in df.columns else None
    name_col = "Molecule_Name" if "Molecule_Name" in df.columns else None

    cols = ["SMILES"]
    if id_col:
        cols.insert(0, id_col)
    if name_col:
        cols.append(name_col)

    unique_mols = df[cols].drop_duplicates().reset_index(drop=True)

    # Grid view for molecule selection (show first 12)
    st.markdown(f"**{len(unique_mols)} unique molecules**")

    # Molecule selector
    if id_col and name_col:
        options = [
            f"{row[id_col]} - {row[name_col]}" for _, row in unique_mols.iterrows()
        ]
    elif id_col:
        options = list(unique_mols[id_col])
    else:
        options = [f"Mol {i + 1}" for i in range(len(unique_mols))]

    _eid = SessionState.get("selected_compound_entry_id", "")
    selected = st.selectbox(
        "Select", options, key=f"mol_select_{_eid}", label_visibility="hidden"
    )
    idx = options.index(selected)
    row = unique_mols.iloc[idx]

    # Display
    col1, col2 = st.columns([1, 1])

    with col1:
        show_nums = st.checkbox(
            "Show atom numbers", value=True, key=f"atom_nums_{_eid}"
        )
        render_2d_structure(row["SMILES"], size=(350, 280), show_atom_numbers=show_nums)

        mol_label = row.get(id_col, "") if id_col else ""
        if st.button("⛶ Expand", key=f"expand_struct_{_eid}", type="primary"):
            _show_expanded_structure(row["SMILES"], label=mol_label)

    with col2:
        if id_col:
            st.markdown(f"**{row[id_col]}**")
        if name_col and row[name_col] != row.get(id_col, ""):
            st.caption(row[name_col])

        st.code(row["SMILES"], language=None)

        # Activity summary for this molecule
        if id_col:
            mol_data = df[df[id_col] == row[id_col]]
            if "Activity_Type" in mol_data.columns:
                st.markdown(
                    f"**Activities:** {mol_data['Activity_Type'].nunique()} types"
                )
            if "pActivity" in mol_data.columns:
                st.markdown(
                    f"**pActivity:** {mol_data['pActivity'].min():.1f} - {mol_data['pActivity'].max():.1f}"
                )
            if "IMP_Final_Score" in mol_data.columns:
                avg = mol_data["IMP_Final_Score"].mean()
                avg_text = f"{avg:.3f}" if pd.notna(avg) else "N/A"
                st.markdown(f"**IMP Score:** {avg_text}")

    # 3D Viewer
    st.markdown("#### 3D Structure")
    _render_3d_viewer(row["SMILES"], _eid)


def _pdb_pagination(
    entry_id: str, page_key: str, current_page: int, total_pages: int, pos: str
) -> None:
    """Render centered ⟪ First ◁ Prev | Page X of Y | Next ▷ Last ⟫ pagination bar."""
    _, c_first, c_prev, c_label, c_next, c_last, _ = st.columns([2, 1, 1, 2, 1, 1, 2])
    with c_first:
        if st.button(
            "⟪ First",
            key=f"pdb_first_{pos}_{entry_id}",
            disabled=current_page <= 1,
            width="stretch",
        ):
            st.session_state[page_key] = 1
            st.rerun()
    with c_prev:
        if st.button(
            "◁ Prev",
            key=f"pdb_prev_{pos}_{entry_id}",
            disabled=current_page <= 1,
            width="stretch",
        ):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    c_label.markdown(
        f"<div style='text-align:center;padding:8px 0;font-size:15px;font-weight:500;'>"
        f"Page {current_page} of {total_pages}</div>",
        unsafe_allow_html=True,
    )
    with c_next:
        if st.button(
            "Next ▷",
            key=f"pdb_next_{pos}_{entry_id}",
            disabled=current_page >= total_pages,
            width="stretch",
        ):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    with c_last:
        if st.button(
            "Last ⟫",
            key=f"pdb_last_{pos}_{entry_id}",
            disabled=current_page >= total_pages,
            width="stretch",
        ):
            st.session_state[page_key] = total_pages
            st.rerun()


def _render_pdb_evidence(
    compound_name: str, df: pd.DataFrame, entry_id: str = None, storage_path: str = None
) -> None:
    """PDB structural evidence from DataFrame columns."""
    if df is None:
        st.info("No data available")
        return

    # Check for PDB columns in main DataFrame
    pdb_cols = [
        "PDB_Score",
        "PDB_Num_Structures",
        "PDB_IDs",
        "PDB_Best_Resolution",
        "PDB_High_Quality",
        "PDB_Medium_Quality",
        "PDB_Poor_Quality",
    ]
    has_pdb = any(col in df.columns for col in pdb_cols)

    if not has_pdb:
        st.info(
            "No PDB structural evidence available. Re-run analysis with PDB enabled."
        )
        st.caption(
            "PDB scoring queries RCSB PDB for experimental crystal structures of similar compounds."
        )
        return

    # Get unique compounds with PDB data
    unique_df = df.drop_duplicates("ChEMBL_ID") if "ChEMBL_ID" in df.columns else df

    # Try to load detailed PDB summary file FIRST to get accurate counts
    pdb_summary_df = None
    try:
        safe_name = sanitize_compound_name(compound_name)
        for filename in [
            "pdb_summary.csv",
            f"{safe_name}_pdb_summary.csv",
            f"{safe_name}_pdb_details.csv",
        ]:
            pdb_summary_df = smart_load_dataframe(
                filename, entry_id=entry_id, storage_path=storage_path
            )
            if pdb_summary_df is not None and not pdb_summary_df.empty:
                break
    except Exception:
        pdb_summary_df = None

    # Calculate summary statistics - use pdb_summary_df if available for accurate counts
    if pdb_summary_df is not None and not pdb_summary_df.empty:
        # Use actual unique PDB structures from the detailed file
        total_structs = len(pdb_summary_df)
        # Count quality from the Quality column
        if "Quality" in pdb_summary_df.columns:
            high_q = int((pdb_summary_df["Quality"] == "***").sum())
            med_q = int((pdb_summary_df["Quality"] == "**").sum())
            poor_q = int((pdb_summary_df["Quality"] == "*").sum())
        else:
            # Fallback to resolution-based counting
            if "Resolution" in pdb_summary_df.columns:
                pdb_summary_df["_res"] = pd.to_numeric(
                    pdb_summary_df["Resolution"], errors="coerce"
                )
                high_q = int((pdb_summary_df["_res"] < 2.0).sum())
                med_q = int(
                    (
                        (pdb_summary_df["_res"] >= 2.0)
                        & (pdb_summary_df["_res"] <= 3.0)
                    ).sum()
                )
                poor_q = int((pdb_summary_df["_res"] > 3.0).sum())
            else:
                high_q = med_q = poor_q = 0
    else:
        # Fallback to summing from unique compounds (less accurate)
        total_structs = (
            int(unique_df["PDB_Num_Structures"].sum())
            if "PDB_Num_Structures" in unique_df.columns
            else 0
        )
        high_q = (
            int(unique_df["PDB_High_Quality"].sum())
            if "PDB_High_Quality" in unique_df.columns
            else 0
        )
        med_q = (
            int(unique_df["PDB_Medium_Quality"].sum())
            if "PDB_Medium_Quality" in unique_df.columns
            else 0
        )
        poor_q = (
            int(unique_df["PDB_Poor_Quality"].sum())
            if "PDB_Poor_Quality" in unique_df.columns
            else 0
        )

    avg_score = (
        unique_df["PDB_Score"].mean() if "PDB_Score" in unique_df.columns else None
    )
    compounds_with_pdb = (
        len(unique_df[unique_df["PDB_Num_Structures"] > 0])
        if "PDB_Num_Structures" in unique_df.columns
        else 0
    )
    pct_with_pdb = (
        (compounds_with_pdb / len(unique_df) * 100) if len(unique_df) > 0 else 0
    )

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average PDB Score", f"{avg_score:.3f}" if pd.notna(avg_score) else "N/A"
        )

    with col2:
        st.metric("Total Structures", total_structs)

    with col3:
        st.metric("High Quality (⭐⭐⭐)", high_q)

    with col4:
        st.metric("% with PDB Data", f"{pct_with_pdb:.1f}%")

    st.caption(f"Summary across {len(unique_df)} unique compounds")

    # Visualizations: Quality donut + Resolution histogram side by side
    total_q = high_q + med_q + poor_q
    if total_q > 0 and pdb_summary_df is not None and not pdb_summary_df.empty:
        viz1, viz2 = st.columns(2)

        with viz1:
            # Quality donut chart
            quality_df = pd.DataFrame(
                {
                    "Quality": [
                        "High (< 2.0 Å)",
                        "Medium (2.0-3.0 Å)",
                        "Poor (> 3.0 Å)",
                    ],
                    "Count": [high_q, med_q, poor_q],
                }
            )
            quality_df = quality_df[quality_df["Count"] > 0]
            fig_donut = px.pie(
                quality_df,
                values="Count",
                names="Quality",
                hole=0.5,
                color="Quality",
                color_discrete_map={
                    "High (< 2.0 Å)": "#22c55e",
                    "Medium (2.0-3.0 Å)": "#f59e0b",
                    "Poor (> 3.0 Å)": "#ef4444",
                },
            )
            fig_donut.update_layout(
                title="Quality Distribution",
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
                margin=dict(t=40, b=40, l=10, r=10),
                height=280,
            )
            fig_donut.update_traces(textinfo="value+percent", textfont_size=13)
            apply_impulator_theme(fig_donut)
            st.plotly_chart(fig_donut, width="stretch")

        with viz2:
            # Resolution distribution histogram
            if "Resolution" in pdb_summary_df.columns:
                res_vals = pd.to_numeric(
                    pdb_summary_df["Resolution"], errors="coerce"
                ).dropna()
                if len(res_vals) > 0:
                    fig_res = px.histogram(
                        res_vals,
                        nbins=15,
                        labels={"value": "Resolution (Å)", "count": "Structures"},
                        color_discrete_sequence=["#3b82f6"],
                    )
                    # Add quality zone annotations
                    fig_res.add_vrect(
                        x0=0,
                        x1=2.0,
                        fillcolor="#22c55e",
                        opacity=0.08,
                        annotation_text="High",
                        annotation_position="top left",
                    )
                    fig_res.add_vrect(
                        x0=2.0,
                        x1=3.0,
                        fillcolor="#f59e0b",
                        opacity=0.08,
                        annotation_text="Medium",
                        annotation_position="top left",
                    )
                    fig_res.add_vrect(
                        x0=3.0,
                        x1=res_vals.max() + 0.5,
                        fillcolor="#ef4444",
                        opacity=0.08,
                        annotation_text="Poor",
                        annotation_position="top left",
                    )
                    fig_res.update_layout(
                        title="Resolution Distribution",
                        xaxis_title="Resolution (Å)",
                        yaxis_title="Structures",
                        margin=dict(t=40, b=40, l=10, r=10),
                        height=280,
                        showlegend=False,
                    )
                    apply_impulator_theme(fig_res)
                    st.plotly_chart(fig_res, width="stretch")

    st.markdown("---")

    # pdb_summary_df was already loaded earlier for accurate counts
    # Card-based PDB structure browser with thumbnails
    if pdb_summary_df is not None and not pdb_summary_df.empty:
        # Build structured data for cards
        card_data = []
        for _, row in pdb_summary_df.iterrows():
            pdb_id = str(row.get("PDB_ID", ""))
            chembl_id = str(row.get("ChEMBL_ID", ""))
            mol_name = row.get("Molecule_Name", "")
            if pd.isna(mol_name):
                mol_name = ""
            title = row.get("Title", "")
            if pd.isna(title):
                title = ""
            resolution = row.get("Resolution", "")
            quality = row.get("Quality", "")
            if pd.isna(quality):
                quality = ""
            exp_method = row.get("Experimental_Method", "")
            if pd.isna(exp_method):
                exp_method = ""
            uniprot = row.get("UniProt_IDs", "")
            if pd.isna(uniprot):
                uniprot = ""

            uniprot_list = [
                u.strip()
                for u in str(uniprot).split(",")
                if u.strip() and u.strip() != "N/A"
            ]

            try:
                res_val = (
                    float(resolution)
                    if resolution and resolution != "N/A" and str(resolution) != "nan"
                    else 999.0
                )
            except (ValueError, TypeError):
                res_val = 999.0

            card_data.append(
                {
                    "pdb_id": pdb_id,
                    "chembl_id": chembl_id,
                    "mol_name": str(mol_name),
                    "title": str(title),
                    "resolution": f"{res_val:.2f} Å" if res_val < 999 else "N/A",
                    "resolution_val": res_val,
                    "quality": quality,
                    "exp_method": exp_method,
                    "uniprot_ids": uniprot_list,
                }
            )

        # Sort by quality then resolution
        quality_order = {"***": 1, "**": 2, "*": 3, "": 4, "N/A": 4}
        card_data.sort(
            key=lambda x: (quality_order.get(x["quality"], 4), x["resolution_val"])
        )

        # Search and sort controls
        sc1, sc2 = st.columns([3, 1])
        search = sc1.text_input(
            "Search",
            placeholder="Search PDB ID, title, ChEMBL ID...",
            key=f"pdb_search_{entry_id}",
            label_visibility="collapsed",
        )
        sort_opt = sc2.selectbox(
            "Sort",
            ["Resolution ↑", "Resolution ↓", "Quality ↓"],
            key=f"pdb_sort_{entry_id}",
            label_visibility="collapsed",
        )

        # Filter
        if search:
            q = search.lower()
            card_data = [
                c
                for c in card_data
                if q in c["pdb_id"].lower()
                or q in c["title"].lower()
                or q in c["chembl_id"].lower()
                or q in c["mol_name"].lower()
            ]

        # Sort
        if sort_opt == "Resolution ↑":
            card_data.sort(key=lambda x: x["resolution_val"])
        elif sort_opt == "Resolution ↓":
            card_data.sort(key=lambda x: x["resolution_val"], reverse=True)
        elif sort_opt == "Quality ↓":
            card_data.sort(
                key=lambda x: (quality_order.get(x["quality"], 4), x["resolution_val"])
            )

        # Pagination
        PAGE_SIZE = 25
        total = len(card_data)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        page_key = f"pdb_page_{entry_id}"

        if page_key not in st.session_state:
            st.session_state[page_key] = 1
        current_page = st.session_state[page_key]
        # Clamp
        if current_page > total_pages:
            current_page = total_pages
            st.session_state[page_key] = current_page

        st.caption(f"{total} structures")
        if total_pages > 1:
            _pdb_pagination(entry_id, page_key, current_page, total_pages, "top")

        start = (current_page - 1) * PAGE_SIZE
        page_data = card_data[start : start + PAGE_SIZE]

        # Render cards
        for c in page_data:
            pdb_id = html.escape(c["pdb_id"])
            title_text = html.escape(c["title"]) if c["title"] else "Untitled"
            star_count = c["quality"].count("*")
            quality_stars = (
                '<span style="color:#f59e0b;font-size:16px;">'
                + "★" * star_count
                + "☆" * (3 - star_count)
                + "</span>"
            )
            chembl_text = html.escape(c["chembl_id"])
            method_text = html.escape(c["exp_method"])
            thumb_url = f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_assembly-1.jpeg"

            # UniProt links
            uniprot_html = ""
            for uid in c["uniprot_ids"][:3]:
                safe_uid = html.escape(uid)
                uniprot_html += (
                    f'<a href="https://www.uniprot.org/uniprotkb/{safe_uid}" '
                    f'target="_blank" style="color:#3b82f6;text-decoration:none;font-weight:600;">{safe_uid}</a> '
                )

            st.markdown(
                f'<div style="display:flex;gap:16px;padding:14px;border-bottom:1px solid rgba(128,128,128,0.2);'
                f'align-items:flex-start;">'
                f'<a href="https://www.rcsb.org/structure/{pdb_id}" target="_blank">'
                f'<img src="{thumb_url}" style="width:90px;height:90px;border-radius:6px;'
                f'object-fit:cover;background:#222;cursor:pointer;flex-shrink:0;" '
                f'onerror="this.style.display=\'none\'" width="90" height="90">'
                f"</a>"
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:17px;font-weight:500;margin-bottom:6px;">{title_text}</div>'
                f'<div style="font-size:15px;opacity:0.85;">'
                f'<b>PDB:</b> <a href="https://www.rcsb.org/structure/{pdb_id}" target="_blank" '
                f'style="color:#3b82f6;text-decoration:none;font-weight:600;">{pdb_id}</a>'
                f" &nbsp; <b>ChEMBL:</b> {chembl_text}"
                f" &nbsp; <b>Quality:</b> {quality_stars}"
                f"</div>"
                f'<div style="font-size:15px;opacity:0.85;margin-top:4px;">'
                f"<b>Resolution:</b> {html.escape(c['resolution'])}"
                f" &nbsp; <b>Method:</b> {method_text}"
                f"{(' &nbsp; <b>UniProt:</b> ' + uniprot_html) if uniprot_html else ''}"
                f"</div>"
                f"</div>"
                f'<div style="flex-shrink:0;display:flex;flex-direction:column;gap:6px;align-self:center;">'
                f'<a href="https://www.rcsb.org/structure/{pdb_id}" target="_blank" '
                f'style="display:block;padding:8px 16px;font-size:14px;text-align:center;'
                f"color:#fff;text-decoration:none;font-weight:600;"
                f"background:linear-gradient(135deg,#3b82f6,#6366f1);border-radius:8px;white-space:nowrap;"
                f"box-shadow:0 2px 8px rgba(59,130,246,0.3);"
                f'transition:transform 0.15s,box-shadow 0.15s;"'
                f" onmouseover=\"this.style.transform='translateY(-1px)';this.style.boxShadow='0 4px 12px rgba(59,130,246,0.4)'\""
                f" onmouseout=\"this.style.transform='none';this.style.boxShadow='0 2px 8px rgba(59,130,246,0.3)'\">"
                f"View in PDB</a>"
                f'<a href="https://www.rcsb.org/3d-view/{pdb_id}" target="_blank" '
                f'style="display:block;padding:8px 16px;font-size:14px;text-align:center;'
                f"color:#fff;text-decoration:none;font-weight:600;"
                f"background:linear-gradient(135deg,#22c55e,#06b6d4);border-radius:8px;white-space:nowrap;"
                f"box-shadow:0 2px 8px rgba(34,197,94,0.3);"
                f'transition:transform 0.15s,box-shadow 0.15s;"'
                f" onmouseover=\"this.style.transform='translateY(-1px)';this.style.boxShadow='0 4px 12px rgba(34,197,94,0.4)'\""
                f" onmouseout=\"this.style.transform='none';this.style.boxShadow='0 2px 8px rgba(34,197,94,0.3)'\">"
                f"Explore in 3D</a>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Bottom pagination
        if total_pages > 1:
            _pdb_pagination(entry_id, page_key, current_page, total_pages, "bot")

    else:
        # Fallback: PDB summary file not found - show basic info from DataFrame
        # Note: For newly processed compounds, pdb_summary.csv should exist
        if "PDB_IDs" in unique_df.columns:
            # Collect all PDB IDs with associated ChEMBL data
            all_pdb_ids = []
            pdb_compound_map = {}  # Map PDB ID -> list of (chembl_id, mol_name)

            for _, row in unique_df.iterrows():
                pdb_str = row.get("PDB_IDs", "")
                chembl_id = row.get("ChEMBL_ID", "Unknown")
                mol_name = row.get("Molecule_Name", "")
                if pd.isna(mol_name) or not isinstance(mol_name, str):
                    mol_name = ""

                if pd.notna(pdb_str) and pdb_str:
                    pdb_list = [p.strip() for p in str(pdb_str).split(",") if p.strip()]
                    for pdb_id in pdb_list:
                        pdb_id_upper = pdb_id.upper()
                        all_pdb_ids.append(pdb_id_upper)
                        if pdb_id_upper not in pdb_compound_map:
                            pdb_compound_map[pdb_id_upper] = []
                        pdb_compound_map[pdb_id_upper].append((chembl_id, mol_name))

            unique_pdb_ids = list(set(all_pdb_ids))

            if unique_pdb_ids:
                st.markdown(f"**{len(unique_pdb_ids)} Unique PDB Structures**")
                st.caption(
                    "⚠️ Detailed PDB info not available. Re-process the compound to fetch PDB details."
                )
                st.caption("Click on PDB ID to view structure on RCSB PDB.")

                # Show basic info from DataFrame without API calls
                pdb_data = []
                for pdb_id in sorted(unique_pdb_ids):
                    compounds = pdb_compound_map.get(pdb_id, [])
                    chembl_ids = list(set([c[0] for c in compounds if c[0]]))
                    mol_names = list(set([c[1] for c in compounds if c[1]]))
                    pdb_data.append(
                        {
                            "PDB_Link": f"https://www.rcsb.org/structure/{pdb_id}",
                            "ChEMBL_IDs": ", ".join(chembl_ids)
                            if chembl_ids
                            else "N/A",
                            "Molecule_Name": ", ".join(mol_names[:3])
                            + (
                                f" (+{len(mol_names) - 3})"
                                if len(mol_names) > 3
                                else ""
                            )
                            if mol_names
                            else "N/A",
                        }
                    )

                pdb_table = pd.DataFrame(pdb_data)
                st.dataframe(
                    pdb_table,
                    width="stretch",
                    hide_index=True,
                    height=400,
                    column_config={
                        "PDB_Link": st.column_config.LinkColumn(
                            "PDB ID",
                            display_text=r"https://www\.rcsb\.org/structure/(.+)",
                            width=80,
                        ),
                        "ChEMBL_IDs": st.column_config.TextColumn(
                            "ChEMBL IDs", width=200
                        ),
                        "Molecule_Name": st.column_config.TextColumn(
                            "Molecule Names", width=250
                        ),
                    },
                )
            else:
                st.info("No PDB IDs found in the data")


# =============================================================================
# DATA TAB
# =============================================================================


def _render_data_tab(data: dict[str, Any]) -> None:
    """Data tables with downloads."""
    df = data.get("results")
    compound_name = data.get("compound_name", "compound")

    if df is None or df.empty:
        st.warning("No data available")
        return

    # Use pre-loaded all similar molecules catalog from data dict
    all_similar_df = data.get("all_similar")

    # View selector
    view = st.radio(
        "View",
        ["Core Analysis", "Interpretation", "All Similar", "Full Data"],
        horizontal=True,
        label_visibility="hidden",
    )

    st.markdown("---")

    if view == "Core Analysis":
        # Include SMILES for structure data in CSV downloads
        # Columns: Identifiers, Activity, Target, Efficiency Metrics, Properties
        cols = [
            "ChEMBL_ID",
            "Molecule_Name",
            "Similarity",
            "SMILES",
            "Activity_Type",
            "Activity_nM",
            "pActivity",
            "Target_ChEMBL_ID",
            "Target_Name",
            "SEI",
            "BEI",
            "NSEI",
            "NBEI",
            "Molecular_Weight",
            "LogP",
            "TPSA",
            "QED",
            "HBA",
            "HBD",
            "Heavy_Atoms",
            "PSAoMW",
            "10xPSA_MW",
            "NPOLoNHA",
        ]
        cols = [c for c in cols if c in df.columns]
        display_df = df[cols]
        st.dataframe(display_df, width="stretch", height=450, hide_index=True)

        # Deferred download - generates CSV on-demand (non-blocking)
        st.download_button(
            "📥 Download",
            data=lambda df=display_df: df.to_csv(index=False),
            file_name=f"{compound_name}_analysis.csv",
            mime="text/csv",
        )

    elif view == "Interpretation":
        # Include SMILES for structure data in CSV downloads.
        # PRES-09: drop IMP_Classification, IMP_Confidence. PRES-10: inject
        # IMP_Score_Integer (defensive — Plan 03 backend will eventually emit
        # this column natively).
        if "IMP_Final_Score" in df.columns and "IMP_Score_Integer" not in df.columns:
            df["IMP_Score_Integer"] = df["IMP_Final_Score"].apply(format_imp_score)
        cols = [
            "ChEMBL_ID",
            "Molecule_Name",
            "SMILES",
            "IMP_Final_Score",
            "IMP_Score_Integer",
            "Is_IMP_Candidate",
            "PDB_Score",
            "Efficiency_Score",
        ]
        cols = [c for c in cols if c in df.columns]

        if cols:
            display_df = df[cols].drop_duplicates()
            st.dataframe(display_df, width="stretch", height=450, hide_index=True)

            # Deferred download - generates CSV on-demand (non-blocking)
            st.download_button(
                "📥 Download",
                data=lambda df=display_df: df.to_csv(index=False),
                file_name=f"{compound_name}_interpretation.csv",
                mime="text/csv",
            )
        else:
            st.info("No interpretation columns available")

    elif view == "All Similar":
        if all_similar_df is not None and not all_similar_df.empty:
            # Build activity count lookup from main results
            activity_counts = {}
            if df is not None and "ChEMBL_ID" in df.columns:
                activity_counts = df.groupby("ChEMBL_ID").size().to_dict()

            display_df = all_similar_df.copy()
            display_df["Biological_Activity"] = (
                display_df["ChEMBL_ID"].map(activity_counts).fillna(0).astype(int)
            )

            cols = [
                "ChEMBL_ID",
                "Molecule_Name",
                "Similarity",
                "Biological_Activity",
                "Molecular_Weight",
                "QED",
                "LogP",
                "PAINS_Violation",
                "Aggregator_Risk",
                "BRENK_Alerts",
                "Kingdom",
                "Superclass",
            ]
            cols = [c for c in cols if c in display_df.columns]

            total = len(display_df)
            with_data = (display_df["Biological_Activity"] > 0).sum()
            st.caption(
                f"{total} similar compounds found — {with_data} with biological activity, {total - with_data} without"
            )

            st.dataframe(display_df[cols], width="stretch", height=450, hide_index=True)
            st.download_button(
                "📥 Download All Similar",
                data=display_df[cols].to_csv(index=False),
                file_name=f"{compound_name}_all_similar.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "All similar molecules catalog not available for this compound. Re-run the analysis to generate it."
            )

    else:  # Full Data
        # Remove internal columns
        hide = [c for c in df.columns if c.startswith("Is_") and c.endswith("_Outlier")]
        display_df = df[[c for c in df.columns if c not in hide]]

        st.caption(f"{len(display_df)} rows × {len(display_df.columns)} columns")
        st.dataframe(display_df, width="stretch", height=450, hide_index=True)

        # Deferred download - generates CSV on-demand (non-blocking)
        st.download_button(
            "📥 Download Full",
            data=lambda df=display_df: df.to_csv(index=False),
            file_name=f"{compound_name}_complete.csv",
            mime="text/csv",
        )


# =============================================================================
# DATA LOADING & DELETE
# =============================================================================


def _render_drug_indications(data: dict[str, Any]) -> None:
    """
    Render drug indications tab with clickable links to MESH, EFO, and Clinical Trials.

    Shows disease associations and clinical trial phases for similar compounds.
    """
    indications_df = data.get("indications")

    st.markdown("### 💊 Drug Indications")
    st.caption("Disease associations and clinical trial phases from ChEMBL")

    if indications_df is None or indications_df.empty:
        st.info(
            "No drug indications found for these compounds. This is common for research compounds not yet in clinical trials."
        )
        st.markdown("""
        **Note:** Drug indications are only available for compounds that:
        - Have entered clinical trials
        - Are approved drugs
        - Have documented therapeutic uses in ChEMBL
        """)
        return

    # Summary metrics
    total_indications = len(indications_df)
    unique_compounds = indications_df["ChEMBL_ID"].nunique()
    unique_diseases = (
        indications_df["MESH_Heading"].nunique()
        if "MESH_Heading" in indications_df.columns
        else 0
    )

    # Get max phase
    max_phase = 0
    if "Max_Phase" in indications_df.columns:
        max_phase = indications_df["Max_Phase"].max()

    # Phase badge
    def get_phase_badge(phase):
        if phase >= 4:
            return "🟢 Approved"
        elif phase >= 3:
            return "🔵 Phase 3"
        elif phase >= 2:
            return "🟡 Phase 2"
        elif phase >= 1:
            return "🟠 Phase 1"
        elif phase >= 0.5:
            return "⚪ Early Phase 1"
        else:
            return "⚫ Unknown"

    # Metrics row
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Indications", total_indications)
    with cols[1]:
        st.metric("Compounds with Data", unique_compounds)
    with cols[2]:
        st.metric("Unique Diseases", unique_diseases)
    with cols[3]:
        st.metric("Max Phase", get_phase_badge(max_phase))

    # Visualizations — right after metrics for attention
    if len(indications_df) > 1:
        viz1, viz2 = st.columns(2)

        # Left: Phase pipeline donut
        if "Max_Phase" in indications_df.columns:
            phase_labels_map = {
                4.0: "Approved",
                3.0: "Phase 3",
                2.0: "Phase 2",
                1.0: "Phase 1",
                0.5: "Early Phase 1",
                -1.0: "Unknown",
            }
            phase_colors_map = {
                "Approved": "#22c55e",
                "Phase 3": "#3b82f6",
                "Phase 2": "#eab308",
                "Phase 1": "#f97316",
                "Early Phase 1": "#94a3b8",
                "Unknown": "#525252",
            }
            phase_counts = (
                indications_df["Max_Phase"]
                .map(lambda p: phase_labels_map.get(p, f"Phase {p}"))
                .value_counts()
            )

            with viz1:
                fig_phase = px.pie(
                    values=phase_counts.values,
                    names=phase_counts.index,
                    hole=0.45,
                    color=phase_counts.index,
                    color_discrete_map=phase_colors_map,
                )
                fig_phase.update_traces(textinfo="value+percent", textfont_size=13)
                fig_phase.update_layout(
                    title="Clinical Phase Pipeline",
                    legend=dict(orientation="h", y=-0.15),
                    margin=dict(t=40, b=50, l=10, r=10),
                    height=300,
                )
                apply_impulator_theme(fig_phase)
                st.plotly_chart(fig_phase, width="stretch")

        # Right: Top diseases by indication count
        if "MESH_Heading" in indications_df.columns:
            disease_counts = indications_df["MESH_Heading"].value_counts().head(10)
            with viz2:
                fig_disease = px.bar(
                    x=disease_counts.values,
                    y=disease_counts.index,
                    orientation="h",
                    color=disease_counts.values,
                    color_continuous_scale=["#6366f1", "#3b82f6", "#06b6d4"],
                    labels={"x": "Indications", "y": ""},
                )
                fig_disease.update_layout(
                    title="Top Diseases",
                    showlegend=False,
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=40, b=30, l=10, r=10),
                    height=300,
                )
                apply_impulator_theme(fig_disease)
                st.plotly_chart(fig_disease, width="stretch")

        # Compound × Disease heatmap (if multiple compounds)
        if (
            unique_compounds > 1
            and "MESH_Heading" in indications_df.columns
            and "ChEMBL_ID" in indications_df.columns
        ):
            top_diseases = (
                indications_df["MESH_Heading"].value_counts().head(15).index.tolist()
            )
            heat_df = indications_df[indications_df["MESH_Heading"].isin(top_diseases)]
            if "Max_Phase" in heat_df.columns:
                pivot = heat_df.pivot_table(
                    index="ChEMBL_ID",
                    columns="MESH_Heading",
                    values="Max_Phase",
                    aggfunc="max",
                    fill_value=0,
                )
                if len(pivot) > 1 and len(pivot.columns) > 1:
                    fig_heat = px.imshow(
                        pivot,
                        aspect="auto",
                        color_continuous_scale=["#1e1b4b", "#3b82f6", "#22c55e"],
                        labels=dict(x="Disease", y="Compound", color="Max Phase"),
                    )
                    fig_heat.update_layout(
                        title="Compound × Disease Matrix (Max Phase)",
                        margin=dict(t=40, b=10, l=10, r=10),
                        height=max(250, len(pivot) * 30 + 80),
                    )
                    apply_impulator_theme(fig_heat)
                    st.plotly_chart(fig_heat, width="stretch")

    st.markdown("---")

    # Search/filter
    sc1, sc2 = st.columns([3, 1])
    search_term = sc1.text_input(
        "Search",
        placeholder="Search diseases, compounds...",
        key="indication_search",
        label_visibility="collapsed",
    )
    sort_opt = sc2.selectbox(
        "Sort",
        ["Phase (highest)", "Phase (lowest)", "Disease A-Z"],
        key="indication_sort",
        label_visibility="collapsed",
    )

    # Filter DataFrame
    display_df = indications_df.copy()
    if search_term:
        mask = pd.Series(False, index=display_df.index)
        for col in ["MESH_Heading", "EFO_Term", "ChEMBL_ID"]:
            if col in display_df.columns:
                mask |= display_df[col].str.contains(search_term, case=False, na=False)
        display_df = display_df[mask]

    if display_df.empty:
        st.warning(
            f"No indications found matching '{search_term}'"
            if search_term
            else "No indications"
        )
        return

    # Sort
    if "Max_Phase" in display_df.columns:
        if sort_opt == "Phase (highest)":
            display_df = display_df.sort_values("Max_Phase", ascending=False)
        elif sort_opt == "Phase (lowest)":
            display_df = display_df.sort_values("Max_Phase", ascending=True)
    if sort_opt == "Disease A-Z" and "MESH_Heading" in display_df.columns:
        display_df = display_df.sort_values("MESH_Heading", na_position="last")

    # Phase colors and labels
    phase_styles = {
        4: ("#22c55e", "Approved"),
        3: ("#3b82f6", "Phase 3"),
        2: ("#eab308", "Phase 2"),
        1: ("#f97316", "Phase 1"),
        0.5: ("#94a3b8", "Early Phase 1"),
        0: ("#666", "Unknown"),
    }

    def get_phase_style(val):
        for threshold in [4, 3, 2, 1, 0.5]:
            if val >= threshold:
                return phase_styles[threshold]
        return phase_styles[0]

    nct_pattern = re.compile(r"^NCT\d+$")

    # Pagination — 2 cols × 20 rows = 40 per page
    indication_rows = list(display_df.iterrows())
    PAGE_SIZE = 40
    total = len(indication_rows)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    ind_page_key = "indication_page"
    if ind_page_key not in st.session_state:
        st.session_state[ind_page_key] = 1
    current_page = min(st.session_state[ind_page_key], total_pages)

    st.caption(f"{total} indications")
    if total_pages > 1:
        _pdb_pagination("indication", ind_page_key, current_page, total_pages, "top")

    start = (current_page - 1) * PAGE_SIZE
    page_rows = indication_rows[start : start + PAGE_SIZE]

    # Render cards in 2 columns
    left_col, right_col = st.columns(2)
    ind_cols = [left_col, right_col]
    for card_idx, (_, row) in enumerate(page_rows):
        mesh_id = str(row.get("MESH_ID", "")) if pd.notna(row.get("MESH_ID")) else ""
        mesh_heading = (
            str(row.get("MESH_Heading", ""))
            if pd.notna(row.get("MESH_Heading"))
            else "Unknown Disease"
        )
        efo_id = str(row.get("EFO_ID", "")) if pd.notna(row.get("EFO_ID")) else ""
        chembl_id = (
            str(row.get("ChEMBL_ID", "")) if pd.notna(row.get("ChEMBL_ID")) else ""
        )
        max_phase_val = row.get("Max_Phase", 0)
        if pd.isna(max_phase_val):
            max_phase_val = 0

        phase_color, phase_label = get_phase_style(max_phase_val)

        # Build URLs
        mesh_url = f"https://id.nlm.nih.gov/mesh/{mesh_id}.html" if mesh_id else ""
        efo_url = (
            (
                f"https://www.ebi.ac.uk/ols4/ontologies/efo/classes/"
                f"http%253A%252F%252Fwww.ebi.ac.uk%252Fefo%252F{efo_id.replace(':', '_')}"
            )
            if efo_id
            else ""
        )
        chembl_url = (
            f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"
            if chembl_id
            else ""
        )

        # ClinicalTrials URL
        nct_ids_raw = (
            str(row.get("Clinical_Trials_IDs", ""))
            if pd.notna(row.get("Clinical_Trials_IDs"))
            else ""
        )
        ct_url = ""
        ct_label = ""
        if nct_ids_raw:
            nct_ids = [
                n.strip()
                for n in nct_ids_raw.replace(",", " ").split()
                if n.strip() and nct_pattern.match(n.strip())
            ]
            if nct_ids:
                ct_url = f"https://clinicaltrials.gov/search?term={'%20'.join(nct_ids)}"
                ct_label = f"{len(nct_ids)} Trial{'s' if len(nct_ids) > 1 else ''}"
        if not ct_url and mesh_heading:
            ct_url = (
                f"https://clinicaltrials.gov/search?cond={quote_plus(mesh_heading)}"
            )
            ct_label = "Search Trials"

        # Phase badge
        phase_badge = (
            f'<span style="display:inline-block;padding:4px 12px;border-radius:12px;font-size:13px;'
            f"font-weight:700;background:{phase_color}22;color:{phase_color};"
            f'border:1px solid {phase_color}44;">{html.escape(phase_label)}</span>'
        )

        # Link buttons with gradients
        def _grad_btn(url, label, id_text, grad_from, grad_to, shadow_color):
            return (
                f'<a href="{url}" target="_blank" '
                f'style="display:inline-block;padding:6px 14px;font-size:13px;font-weight:600;'
                f"color:#fff;text-decoration:none;border-radius:6px;margin:3px 4px 3px 0;"
                f"background:linear-gradient(135deg,{grad_from},{grad_to});"
                f"box-shadow:0 2px 6px {shadow_color};"
                f'transition:transform 0.15s,box-shadow 0.15s;"'
                f" onmouseover=\"this.style.transform='translateY(-1px)';this.style.boxShadow='0 4px 10px {shadow_color}'\""
                f" onmouseout=\"this.style.transform='none';this.style.boxShadow='0 2px 6px {shadow_color}'\">"
                f"{label}: <b>{id_text}</b></a>"
            )

        links_html = ""
        if mesh_url and mesh_id:
            links_html += _grad_btn(
                mesh_url,
                "MESH",
                html.escape(mesh_id),
                "#3b82f6",
                "#6366f1",
                "rgba(59,130,246,0.3)",
            )
        if efo_url and efo_id:
            links_html += _grad_btn(
                efo_url,
                "EFO",
                html.escape(efo_id),
                "#8b5cf6",
                "#a855f7",
                "rgba(139,92,246,0.3)",
            )
        if chembl_url and chembl_id:
            links_html += _grad_btn(
                chembl_url,
                "ChEMBL",
                html.escape(chembl_id),
                "#f59e0b",
                "#f97316",
                "rgba(245,158,11,0.3)",
            )
        if ct_url:
            links_html += _grad_btn(
                ct_url,
                "Trials",
                html.escape(ct_label),
                "#22c55e",
                "#06b6d4",
                "rgba(34,197,94,0.3)",
            )

        with ind_cols[card_idx % 2]:
            st.markdown(
                f'<div style="padding:12px;border:1px solid rgba(128,128,128,0.2);'
                f'border-radius:8px;margin-bottom:8px;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap;">'
                f'<span style="font-size:16px;font-weight:600;">{html.escape(mesh_heading)}</span>'
                f"{phase_badge}"
                f"</div>"
                f'<div style="font-size:14px;opacity:0.85;margin-bottom:6px;">'
                f'<b>Compound:</b> <a href="{chembl_url}" target="_blank" '
                f'style="color:#3b82f6;text-decoration:none;font-weight:600;">{html.escape(chembl_id)}</a>'
                f"</div>"
                f'<div style="display:flex;flex-wrap:wrap;">{links_html}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Bottom pagination
    if total_pages > 1:
        _pdb_pagination("indication", ind_page_key, current_page, total_pages, "bot")


def _load_compound_data(
    compound_name: str = None,
    entry_id: str = None,
    storage_path: str = None,
    internal_prefix: str = "",
) -> Optional[dict[str, Any]]:
    """Load compound data from storage.

    Uses smart loaders that prioritize storage_path (from database), then entry_id.
    Only UUID-based storage paths are supported.

    Args:
        compound_name: Display name of the compound (for logging only)
        entry_id: UUID entry_id for storage lookup
        storage_path: Full Azure storage path from database (most reliable)
        internal_prefix: Optional ZIP-entry prefix (D-12, COLL-14). Default "" keeps
            the single-compound behavior byte-identical. A collection member is
            nested under ``compounds/{safe_name}/`` inside the collection ZIP;
            passing that prefix lets this UNMODIFIED renderer drill into the
            member section via the plan-06 ``smart_load_*`` seam.
    """
    try:
        # Use smart loader with storage_path from database (UUID-based)
        summary = smart_load_summary(
            entry_id=entry_id, storage_path=storage_path, internal_prefix=internal_prefix
        )
        if summary is None:
            logger.debug(
                f"Could not load summary for {compound_name} (entry_id={entry_id}, storage_path={storage_path})"
            )
            return {"_error": "not_found"}

        # Load results DataFrame using smart loader
        df = smart_load_dataframe(
            "similar_compounds.csv",
            entry_id=entry_id,
            storage_path=storage_path,
            internal_prefix=internal_prefix,
        )

        if df is None:
            # Try alternate filename format
            safe_name = sanitize_compound_name(compound_name or entry_id or "unknown")
            df = smart_load_dataframe(
                f"{safe_name}_complete_results.csv",
                entry_id=entry_id,
                storage_path=storage_path,
                internal_prefix=internal_prefix,
            )

        # Load drug indications (separate file)
        indications_df = smart_load_dataframe(
            "drug_indications.csv",
            entry_id=entry_id,
            storage_path=storage_path,
            internal_prefix=internal_prefix,
        )

        # Load all similar molecules catalog (may not exist for older compounds)
        all_similar_df = smart_load_dataframe(
            "all_similar_molecules.csv",
            entry_id=entry_id,
            storage_path=storage_path,
            internal_prefix=internal_prefix,
        )

        # Get display name from summary (compound_name is in summary.json)
        display_name = summary.get("compound_name", compound_name or entry_id)

        # Compute InChI, InChIKey, and molecular formula once for reuse across all tabs
        smiles = summary.get("smiles", summary.get("query_smiles", ""))
        inchi = None
        inchikey = None
        mol_formula = None
        if smiles:
            try:
                from rdkit import Chem
                from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
                from rdkit.Chem import rdMolDescriptors

                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    inchi = MolToInchi(mol)
                    inchikey = MolToInchiKey(mol)
                    mol_formula = rdMolDescriptors.CalcMolFormula(mol)
            except Exception as e:
                logger.warning(
                    f"InChI/InChIKey conversion failed for '{display_name}' (SMILES={smiles[:50]}): {e}"
                )

        return {
            "compound_name": display_name,
            "author_name": summary.get("author_name", "N/A"),
            "entry_id": summary.get("entry_id", entry_id),
            "storage_path": storage_path,
            "smiles": smiles,
            "inchi": inchi,
            "inchikey": inchikey,
            "mol_formula": mol_formula,
            "similar_count": summary.get(
                "similar_count", summary.get("total_compounds", 0)
            ),
            "has_imp_warning": summary.get("has_imp_candidates", False),
            "summary": summary,
            "results": df,
            "indications": indications_df,
            "all_similar": all_similar_df,
        }

    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing compound data for '{compound_name}': {e}")
        return {"_error": "parse_error", "_detail": str(e)}
    except Exception as e:
        logger.error(f"Error loading compound data for '{compound_name}': {e}")
        return {"_error": "server_error", "_detail": str(e)}


@st.dialog("Confirm Delete")
def _show_delete_confirmation(
    compound_name: str, entry_id: Optional[str] = None
) -> None:
    """Delete confirmation dialog (modal overlay).

    Calls backend API to delete compound from database, Azure storage, and local cache.
    """
    st.warning(
        f"Are you sure you want to delete **{compound_name}**? This cannot be undone."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", width="stretch"):
            st.rerun()  # Closes the dialog
    with col2:
        if st.button("Delete", type="primary", width="stretch"):
            try:
                if not entry_id:
                    st.error("Cannot delete: compound entry_id not found")
                    return

                api_client = get_api_client()
                result = api_client.delete_compound(entry_id)

                if result.success:
                    if entry_id:
                        delete_from_cache(entry_id)
                    st.session_state["_delete_success"] = compound_name
                    st.query_params.clear()
                    # Clear compound list cache so deleted compound disappears
                    try:
                        from frontend.services import get_compounds_cached

                        get_compounds_cached.clear()
                    except Exception:
                        pass
                    SessionState.navigate_to_home()
                    st.rerun()
                else:
                    st.error(f"Delete failed: {result.error}")
            except Exception as e:
                logger.error(f"Error deleting compound: {e}")
                st.error(f"Error: {e}")


# =============================================================================
# REPORT TAB FUNCTIONS
# =============================================================================


def _render_report_tab(data: dict[str, Any]) -> None:
    """Render the Report tab with comprehensive analysis and export."""
    df = data.get("results")
    if df is None or df.empty:
        st.warning("No data available for report generation.")
        return

    compound_name = data.get("compound_name", "Unknown")
    smiles = data.get("smiles", "")
    summary = data.get("summary", {})

    # Calculate scores - use MAX (best scoring compound) to match Overview tab behavior
    # The Overview shows "Best scoring compound" so we match that for consistency
    mean_score = df["IMP_Final_Score"].max() if "IMP_Final_Score" in df.columns else 0
    mean_qed = df["QED"].mean() if "QED" in df.columns else 0

    # On-demand HTML generation using session state to save memory
    # HTML is only generated when user clicks "Generate Report", not on every page load
    # Key prefix matches evict_report_cache() pattern: "_report_"
    report_key = f"_report_{compound_name}"

    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        # Generate button - only creates HTML when clicked
        _eid = data.get("entry_id", "")
        if st.button(
            "🔄 Generate HTML Report",
            key=f"generate_report_btn_{_eid}",
            help="Click to generate the HTML report for download",
        ):
            with st.spinner("Generating report with charts..."):
                html_content = _generate_html_report(data, df)
                # Evict old reports before caching new one (limit to 5)
                from frontend.utils.session_state import evict_report_cache

                evict_report_cache()
                st.session_state[report_key] = html_content
                st.success("Report ready for download!")

    with col2:
        # Download button - only shown if HTML has been generated
        if report_key in st.session_state:
            st.download_button(
                "📄 Download HTML",
                st.session_state[report_key],
                f"{compound_name.replace(' ', '_')}_report.html",
                "text/html",
                key=f"download_report_btn_{_eid}",
            )
        else:
            st.markdown(
                "<small style='color: var(--text-color); opacity: 0.5;'>Click 'Generate' first</small>",
                unsafe_allow_html=True,
            )

    with col3:
        st.info(
            "💡 **Tip:** Generate the report, download the HTML, then use Ctrl+P to print to PDF"
        )

    st.markdown("---")

    # Render all report sections
    _render_report_header(data, smiles)
    _render_report_executive_summary(df, mean_score, mean_qed, summary)
    _render_report_properties_table(df)
    _render_report_imp_score_calculation(df)
    _render_report_red_flags(df)
    _render_report_bioactivity_donut(df)
    _render_report_efficiency_boxplots(df)
    _render_report_efficiency_plane(df)
    _render_report_pdb_evidence(df, data)
    _render_report_classification(df)
    _render_report_indications(data)
    _render_report_recommendation(df, mean_score)


def _render_report_header(data: dict[str, Any], smiles: str) -> None:
    """Render report header with 2D structure, compound name, SMILES."""
    st.markdown("## 📋 IMPULATOR Compound Analysis Report")

    compound_name = data.get("compound_name", "Unknown")
    summary = data.get("summary", {})

    col1, col2 = st.columns([1, 2])

    with col1:
        # Render 2D structure
        if smiles:
            try:
                from rdkit import Chem
                from rdkit.Chem import Draw
                import io
                import base64

                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # High-quality image (2x size for better resolution)
                    img = Draw.MolToImage(mol, size=(600, 500))
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG", optimize=False, quality=95)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    # Display at 300x250 but use 600x500 source for crisp rendering
                    st.markdown(
                        f'<img src="data:image/png;base64,{img_b64}" style="width: 300px; height: 250px; border: 1px solid #ddd; border-radius: 8px;">',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("Could not render structure")
            except Exception as e:
                st.warning(f"Structure rendering unavailable: {e}")
        else:
            st.info("No SMILES available")

    with col2:
        st.markdown(f"### {html.escape(compound_name)}")

        # Use pre-computed InChI and InChIKey from data
        inchikey = data.get("inchikey") or "N/A"
        inchi = data.get("inchi") or "N/A"

        st.markdown(f"**InChIKey:** `{inchikey}`")
        if inchi and inchi != "N/A":
            st.markdown(f"**InChI:** `{inchi}`")
        smiles_display = smiles[:80] + "..." if len(smiles) > 80 else smiles
        st.markdown(f"**SMILES:** `{smiles_display}`")
        st.markdown(f"**Analysis Date:** {summary.get('processing_date', 'N/A')}")
        author_name = data.get("author_name", "N/A")
        if author_name and author_name != "N/A":
            st.markdown(f"**Author:** {html.escape(author_name)}")

    st.markdown("---")

    # Add summary stats row matching the Overview header
    df = data.get("results")
    if df is not None and not df.empty:
        # Get stats from summary (same source as Overview header)
        summary = data.get("summary", {})
        similar_count = summary.get(
            "similar_count",
            df["ChEMBL_ID"].nunique() if "ChEMBL_ID" in df.columns else len(df),
        )
        activities_count = summary.get("total_activities", len(df))
        avg_qed = summary.get("qed") or (
            df["QED"].mean() if "QED" in df.columns else None
        )
        best_imp_score = (
            df["IMP_Final_Score"].max() if "IMP_Final_Score" in df.columns else None
        )

        # Count unique IMP compounds (not activity rows)
        imp_count = 0
        if "Is_IMP_Candidate" in df.columns and "ChEMBL_ID" in df.columns:
            imp_count = df[df["Is_IMP_Candidate"]]["ChEMBL_ID"].nunique()
        elif summary.get("has_imp_candidates", False):
            imp_count = summary.get("imp_candidates", 0)

        # Display stats in columns
        stat_cols = st.columns(5)
        with stat_cols[0]:
            total_similar = summary.get("total_similar", 0)
            compounds_with_data = summary.get("compounds_with_data", similar_count)
            if total_similar > 0 and total_similar > compounds_with_data:
                st.metric(
                    "Similar Compounds",
                    total_similar,
                    help=f"{compounds_with_data} with activity, {total_similar - compounds_with_data} without",
                )
            else:
                st.metric("Similar Compounds", similar_count)
        with stat_cols[1]:
            st.metric("Activities", activities_count)
        with stat_cols[2]:
            st.metric("QED", f"{avg_qed:.2f}" if pd.notna(avg_qed) else "N/A")
        with stat_cols[3]:
            _score_int = (
                format_imp_score(best_imp_score) if pd.notna(best_imp_score) else None
            )
            # Compact surface: st.metric matches sibling columns visually. Full
            # score-card stack with bars lives on the IMP Score tab.
            st.metric(
                "IMP Score",
                _score_int if _score_int is not None else "N/A",
                help="IMP Score on the 0–100 integer scale. Full visualization on the IMP Score tab.",
            )
        with stat_cols[4]:
            st.metric(
                "IMP candidates",
                imp_count,
                help="Unique compounds flagged Is_IMP_Candidate (boolean column from the scoring pipeline).",
            )

    st.markdown("---")


def _render_report_executive_summary(
    df: pd.DataFrame, mean_score: float, mean_qed: float, summary: dict
) -> None:
    """Render executive summary with compact score-card stack (global bar)."""
    st.markdown("### 📊 Executive Summary")

    # Count red flags - use SAME column names as Overview
    # Count UNIQUE COMPOUNDS with flags (not all rows)
    red_flag_cols = [
        "PAINS_Violation",
        "Aggregator_Risk",
        "Redox_Reactive",
        "Fluorescence_Interference",
        "Thiol_Reactive",
        "BRENK_Alerts",
        "NIH_Alerts",
    ]
    unique_df = df.drop_duplicates("ChEMBL_ID") if "ChEMBL_ID" in df.columns else df
    red_flag_count = 0
    for col in red_flag_cols:
        if col in unique_df.columns:
            red_flag_count += int(
                unique_df[col].sum()
                if unique_df[col].dtype == bool
                else unique_df[col].astype(bool).sum()
            )

    # Compact score-card stack (UI-SPEC Component Visual Contract item #2 — global bar only)
    mean_score_int = format_imp_score(mean_score) if pd.notna(mean_score) else None
    if mean_score_int is None:
        st.html(
            "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; padding: 16px; border-left: 4px solid #6b7280; background:#f9fafb;\">"
            '<div style="font-size:28px; font-weight:600; color:#6b7280;">IMP Score: N/A</div>'
            '<div style="font-size:14px; color:#6b7280; margin-top:4px;">Mean IMP score across this query (0–100 global scale)</div>'
            "</div>"
        )
    else:
        # st._main._html (iframe) bypasses DOMPurify which would strip <defs>/<linearGradient>
        st._main._html(
            "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; padding: 16px; border-left: 4px solid #6b7280; background:#f9fafb;\">"
            f'<div style="font-size:32px; font-weight:700; color:#111827; line-height:1;">{mean_score_int}</div>'
            '<div style="font-size:14px; color:#6b7280; margin-top:4px;">Mean IMP score across this query (0–100 global scale)</div>'
            f"{render_imp_range_bar_global(mean_score)}"
            "</div>",
            height=140,
        )

    mean_qed_str = f"{mean_qed:.3f}" if pd.notna(mean_qed) else "N/A"
    st.caption(f"QED: {mean_qed_str} | Red Flags: {red_flag_count} active")

    st.markdown("---")


def _render_report_properties_table(df: pd.DataFrame) -> None:
    """Render compound properties table for BEST scoring compound."""
    st.markdown("### 🧪 Compound Properties")

    # Get best scoring compound to match Overview behavior
    if "IMP_Final_Score" not in df.columns:
        st.info("Property data not available")
        st.markdown("---")
        return

    valid_df = df.dropna(subset=["IMP_Final_Score"])
    if valid_df.empty:
        st.info("Property data not available")
        st.markdown("---")
        return

    best_row = valid_df.loc[valid_df["IMP_Final_Score"].idxmax()]

    # Get values from best compound
    def get_val(col):
        return best_row.get(col) if col in best_row.index else None

    # Compute 10PSA/MW if possible
    tpsa_val = get_val("TPSA")
    mw_val = get_val("Molecular_Weight")
    psa_mw_ratio = (
        (10 * tpsa_val / mw_val)
        if tpsa_val is not None
        and mw_val is not None
        and mw_val > 0
        and not pd.isna(tpsa_val)
        and not pd.isna(mw_val)
        else None
    )

    props = {
        "pActivity": (get_val("pActivity"), "-log10(IC50), higher = more potent"),
        "Molecular Weight": (get_val("Molecular_Weight"), "g/mol"),
        "PSA (TPSA)": (get_val("TPSA"), "Polar surface area (Å²)"),
        "Heavy Atoms": (get_val("Heavy_Atoms"), "Non-hydrogen atom count"),
        "N+O Atoms (NPOL)": (get_val("NPOL"), "Heteroatom count (Num of Polar Atoms)"),
        "QED": (get_val("QED"), "Drug-likeness (0-1)"),
        "LogP": (get_val("MolLogP") or get_val("LogP"), "Lipophilicity"),
        "10PSA/MW": (psa_mw_ratio, "Compound Polarity vs Atom Fingerprint (= BEI/SEI)"),
    }

    # Create table
    table_data = []
    for prop_name, (value, description) in props.items():
        if value is not None and not pd.isna(value):
            table_data.append(
                {
                    "Property": prop_name,
                    "Value": f"{value:.2f}" if isinstance(value, float) else str(value),
                    "Description": description,
                }
            )

    if table_data:
        st.table(pd.DataFrame(table_data))
    else:
        st.info("Property data not available")

    st.markdown("---")


def _render_report_imp_score_calculation(df: pd.DataFrame) -> None:
    """Render step-by-step IMP Score calculation breakdown for BEST scoring compound."""
    st.markdown("### 🔢 IMP Score Calculation")

    # Get the best scoring compound (matches Overview tab behavior)
    if "IMP_Final_Score" not in df.columns:
        st.info("IMP Score data not available")
        st.markdown("---")
        return

    valid_df = df.dropna(subset=["IMP_Final_Score"])
    if valid_df.empty:
        st.info("No valid IMP scores available")
        st.markdown("---")
        return

    # Get the row with highest IMP_Final_Score (best compound)
    best_row = valid_df.loc[valid_df["IMP_Final_Score"].idxmax()]

    st.caption(
        "**Showing calculation for best scoring compound** (matches Overview tab)"
    )

    # Step 1: Efficiency Metrics
    st.markdown("#### Step 1: Efficiency Metrics")

    metrics_data = []
    metric_cols = [
        ("SEI", "pActivity × 100 / PSA", True),
        ("BEI", "pActivity × 1000 / MW", True),
        ("NSEI", "pActivity / (N+O Atoms)", False),
        ("NBEI", "pActivity / Heavy Atoms", False),
    ]

    for col, formula, used in metric_cols:
        if col in best_row.index:
            value = best_row[col]
            metrics_data.append(
                {
                    "Metric": col,
                    "Formula": formula,
                    "Value": f"{value:.3f}" if not pd.isna(value) else "N/A",
                    "Used in Score": "✓ YES" if used else "○ Display only",
                }
            )

    if metrics_data:
        st.table(pd.DataFrame(metrics_data))

    st.caption(
        "**Note:** Only SEI and BEI contribute to the Efficiency Score (NSEI/NBEI are for reference)"
    )

    # Step 2: Component Scores
    st.markdown("#### Step 2: Component Scores")

    component_data = []
    components = [
        ("Efficiency_Score", "Efficiency", "45%"),
        ("Distance_Score", "Distance", "20%"),
        ("Angle_Score", "Angle", "15%"),
        ("Interference_Score", "Interference", "15%"),
        ("PDB_Score", "PDB Evidence", "5%"),
    ]

    for col, name, weight in components:
        if col in best_row.index:
            score = best_row[col]
            contrib_col = col.replace("_Score", "_Contribution")
            contrib = best_row[contrib_col] if contrib_col in best_row.index else None
            component_data.append(
                {
                    "Component": name,
                    "Score": f"{score:.3f}" if not pd.isna(score) else "N/A",
                    "Weight": weight,
                    "Contribution": f"{contrib:.3f}"
                    if contrib is not None and not pd.isna(contrib)
                    else "N/A",
                }
            )

    if component_data:
        st.table(pd.DataFrame(component_data))

    # Step 3: Final Calculation
    st.markdown("#### Step 3: Final Calculation")

    # Use direct indexing for pandas Series (not .get())
    base_score = (
        best_row["IMP_Base_Score"] if "IMP_Base_Score" in best_row.index else None
    )
    qed = best_row["QED"] if "QED" in best_row.index else None
    qed_mult = (
        best_row["QED_Multiplier"] if "QED_Multiplier" in best_row.index else None
    )
    final_score = (
        best_row["IMP_Final_Score"] if "IMP_Final_Score" in best_row.index else None
    )

    if all(
        v is not None and not pd.isna(v)
        for v in [base_score, qed, qed_mult, final_score]
    ):
        # Extract component scores for the formula
        eff_s = (
            best_row["Efficiency_Score"] if "Efficiency_Score" in best_row.index else 0
        )
        dist_s = best_row["Distance_Score"] if "Distance_Score" in best_row.index else 0
        ang_s = best_row["Angle_Score"] if "Angle_Score" in best_row.index else 0
        int_s = (
            best_row["Interference_Score"]
            if "Interference_Score" in best_row.index
            else 0
        )
        pdb_s = best_row["PDB_Score"] if "PDB_Score" in best_row.index else 0

        # Display as formatted text box (not code block to avoid scrolling)
        st.markdown(
            f"""
<div style="background-color: var(--secondary-background-color); padding: 15px; border-radius: 8px; font-family: monospace; white-space: pre-wrap; color: var(--text-color);">
<strong>Base Score</strong> = 0.45×Eff + 0.20×Dist + 0.15×Angle + 0.15×Interf + 0.05×PDB
         = 0.45×<span style="color: #2ca02c; font-weight: bold;">{eff_s:.3f}</span> + 0.20×<span style="color: #2ca02c; font-weight: bold;">{dist_s:.3f}</span> + 0.15×<span style="color: #2ca02c; font-weight: bold;">{ang_s:.3f}</span> + 0.15×<span style="color: #2ca02c; font-weight: bold;">{int_s:.3f}</span> + 0.05×<span style="color: #2ca02c; font-weight: bold;">{pdb_s:.3f}</span>
         = <span style="color: #2ca02c; font-weight: bold;">{base_score:.3f}</span>

<strong>QED Value:</strong> <span style="color: #2ca02c; font-weight: bold;">{qed:.3f}</span>
<strong>QED Multiplier</strong> = 0.75 + 0.25 × QED
             = 0.75 + 0.25 × {qed:.3f}
             = <span style="color: #2ca02c; font-weight: bold;">{qed_mult:.3f}</span>

<strong>FINAL SCORE</strong> = Base Score × QED Multiplier
            = {base_score:.3f} × {qed_mult:.3f}
            = <span style="color: #e67e22; font-size: 1.2em;"><strong>{final_score:.3f}</strong></span>
</div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Complete IMP Score calculation data not available")

    st.markdown("---")


def _render_report_red_flags(df: pd.DataFrame) -> None:
    """Render red flags assessment section using SAME column names as Overview."""
    st.markdown("### ⚠️ Red Flags Assessment")

    # Use SAME column names as Overview tab
    flags = [
        ("PAINS_Violation", "PAINS", "Pan-Assay Interference compounds detected"),
        ("Aggregator_Risk", "Aggregator", "May form colloidal aggregates"),
        ("Redox_Reactive", "Redox", "May interfere via redox cycling"),
        (
            "Fluorescence_Interference",
            "Fluorescence",
            "May interfere with fluorescence assays",
        ),
        ("Thiol_Reactive", "Thiol", "May react with cysteine residues"),
        ("BRENK_Alerts", "BRENK", "Unwanted substructures detected"),
        ("NIH_Alerts", "NIH", "Problematic functional groups detected"),
    ]

    total_flags = 0
    flag_data = []

    # Count UNIQUE COMPOUNDS with flags (not all rows) - same as Overview tab
    unique_df = df.drop_duplicates("ChEMBL_ID") if "ChEMBL_ID" in df.columns else df

    for col, name, description in flags:
        if col in unique_df.columns:
            count = int(
                unique_df[col].sum()
                if unique_df[col].dtype == bool
                else unique_df[col].astype(bool).sum()
            )
            total_flags += count
            flag_data.append((name, count, description))

    # Overall assessment
    if total_flags == 0:
        overall = "LOW CONCERN - No red flags detected"
        overall_color = "#28a745"
    elif total_flags <= 5:
        overall = f"MODERATE CONCERN - {total_flags} flags detected"
        overall_color = "#fd7e14"
    else:
        overall = f"HIGH CONCERN - {total_flags} flags detected"
        overall_color = "#dc3545"

    st.markdown(
        f"""
    <div style="background-color: var(--secondary-background-color); padding: 12px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {overall_color};">
        <strong style="color: {overall_color};">Overall Assessment: {overall}</strong>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display each flag with counts
    cols = st.columns(len(flag_data)) if flag_data else []
    for i, (name, count, description) in enumerate(flag_data):
        with cols[i]:
            if count > 0:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #dc3545;">
                    <div style="font-size: 1.5em; color: #dc3545; font-weight: bold;">{count}</div>
                    <div style="color: var(--text-color);">{name}</div>
                    <div style="font-size: 0.7em; color: #dc3545;">⚠️ Flagged</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #28a745;">
                    <div style="font-size: 1.5em; color: #28a745; font-weight: bold;">0</div>
                    <div style="color: var(--text-color);">{name}</div>
                    <div style="font-size: 0.7em; color: #28a745;">✓ Clean</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")


def _render_report_bioactivity_donut(df: pd.DataFrame) -> None:
    """Render bioactivity distribution donut chart."""
    st.markdown("### 🎯 Bioactivity Distribution")

    if "Activity_Type" not in df.columns:
        st.info("Activity type data not available")
        st.markdown("---")
        return

    # Count activity types
    activity_counts = df["Activity_Type"].value_counts()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Create donut chart
        fig = px.pie(
            values=activity_counts.values,
            names=activity_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            height=350,
            margin=dict(t=30, b=30, l=30, r=30),
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02
            ),
        )
        apply_impulator_theme(fig)
        st.plotly_chart(fig, width="stretch", key="report_activity_donut")

    with col2:
        # Summary table
        st.markdown(f"**Total Activities:** {len(df)}")
        st.markdown(f"**Activity Types:** {len(activity_counts)}")

        table_data = []
        for activity_type, count in activity_counts.head(10).items():
            pct = (count / len(df)) * 100
            table_data.append(
                {"Type": activity_type, "Count": count, "Percentage": f"{pct:.1f}%"}
            )
        st.table(pd.DataFrame(table_data))

    st.markdown("---")


def _render_report_efficiency_boxplots(df: pd.DataFrame) -> None:
    """Render efficiency metrics box plots with enhanced statistics cards."""
    st.markdown("### 📈 Efficiency Metrics Distribution")

    metrics = ["SEI", "BEI", "NSEI", "NBEI"]
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        st.info("Efficiency metrics not available")
        st.markdown("---")
        return

    st.caption(
        "**Note:** Only SEI and BEI are used in IMP scoring. NSEI and NBEI are shown for additional context."
    )

    # Calculate statistics and create metric cards
    metric_colors = {
        "SEI": "#1f77b4",
        "BEI": "#2ca02c",
        "NSEI": "#ff7f0e",
        "NBEI": "#9467bd",
    }

    metric_descriptions = {
        "SEI": "Surface Efficiency Index",
        "BEI": "Binding Efficiency Index",
        "NSEI": "Normalized SEI (display only)",
        "NBEI": "Normalized BEI (display only)",
    }

    # Display metric cards
    cols = st.columns(len(available_metrics))
    for i, metric in enumerate(available_metrics):
        vals = df[metric].dropna()
        if len(vals) > 0:
            with cols[i]:
                mean_val = vals.mean()
                used_in_score = " ✓" if metric in ["SEI", "BEI"] else ""
                description = metric_descriptions.get(metric, metric)
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid {metric_colors.get(metric, "#636EFA")};">
                    <div style="color: var(--text-color); font-size: 0.9em; margin-bottom: 5px;">{metric}{used_in_score}</div>
                    <div style="font-size: 1.5em; color: {metric_colors.get(metric, "#636EFA")}; font-weight: bold;">{mean_val:.2f}</div>
                    <div style="color: var(--text-color); opacity: 0.6; font-size: 0.75em;">{description}</div>
                    <div style="color: var(--text-color); opacity: 0.5; font-size: 0.7em; margin-top: 3px;">Range: {vals.min():.1f}-{vals.max():.1f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("")

    # Prepare data for box plots
    plot_data = []
    for metric in available_metrics:
        for value in df[metric].dropna():
            plot_data.append({"Metric": metric, "Value": value})

    plot_df = pd.DataFrame(plot_data)

    # Create box plot
    fig = px.box(
        plot_df,
        x="Metric",
        y="Value",
        color="Metric",
        points="all",
        color_discrete_map=metric_colors,
    )
    fig.update_layout(
        height=400,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=False,
        xaxis_title="Efficiency Metric",
        yaxis_title="Value",
    )
    apply_impulator_theme(fig)
    st.plotly_chart(fig, width="stretch", key="report_efficiency_box")

    # Enhanced descriptions
    st.markdown("#### Metric Descriptions")

    desc_col1, desc_col2 = st.columns(2)

    with desc_col1:
        st.markdown("""
        **SEI (Surface Efficiency Index)** ✓ *Used in IMP Score*
        - Formula: `pActivity × 100 / PSA`
        - Measures potency efficiency relative to polar surface area
        - Higher values indicate better size efficiency
        - Typical range: 5-20 (good), >20 (exceptional)

        **NSEI (Normalized SEI)**
        - Formula: `pActivity / N+O Atoms`
        - Alternative normalization by heteroatom count
        - Not used in IMP Score but provides additional context
        """)

    with desc_col2:
        st.markdown("""
        **BEI (Binding Efficiency Index)** ✓ *Used in IMP Score*
        - Formula: `pActivity × 1000 / MW`
        - Measures potency efficiency relative to molecular weight
        - Higher values indicate better binding efficiency
        - Typical range: 15-25 (good), >25 (exceptional)

        **NBEI (Normalized BEI)**
        - Formula: `pActivity / Heavy Atoms`
        - Alternative normalization by heavy atom count
        - Not used in IMP Score but provides additional context
        """)

    st.markdown("---")


def _render_report_efficiency_plane(df: pd.DataFrame) -> None:
    """Render SEI vs BEI scatter plot with EQUAL AXIS SCALING for accurate angle visualization."""
    st.markdown("### 📐 Efficiency Plane: SEI vs BEI")

    if "SEI" not in df.columns or "BEI" not in df.columns:
        st.info("SEI/BEI data not available")
        st.markdown("---")
        return

    plot_df = df.dropna(subset=["SEI", "BEI"])

    if plot_df.empty:
        st.info("No valid SEI/BEI data points")
        st.markdown("---")
        return

    # Calculate mean angle and modulus
    mean_sei = plot_df["SEI"].mean()
    mean_bei = plot_df["BEI"].mean()
    mean_angle = (
        plot_df["Angle_SEI_BEI"].mean()
        if "Angle_SEI_BEI" in plot_df.columns
        else np.arctan2(mean_bei, mean_sei) * 180 / np.pi
    )
    mean_modulus = (
        plot_df["Modulus_SEI_BEI"].mean()
        if "Modulus_SEI_BEI" in plot_df.columns
        else np.sqrt(mean_sei**2 + mean_bei**2)
    )

    # Angle assessment
    if 40 <= mean_angle <= 50:
        angle_status = "OPTIMAL ✓"
        angle_color = "#28a745"
    elif 30 <= mean_angle < 40 or 50 < mean_angle <= 60:
        angle_status = "ACCEPTABLE"
        angle_color = "#fd7e14"
    else:
        angle_status = "UNBALANCED ⚠️"
        angle_color = "#dc3545"

    # Display angle assessment banner
    st.markdown(
        f"""
    <div style="background-color: var(--secondary-background-color); padding: 12px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {angle_color};">
        <strong style="color: {angle_color};">Development Trajectory: {angle_status} (Angle: {mean_angle:.1f}°)</strong>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create scatter plot
    fig = go.Figure()

    # Add data points
    color_col = "IMP_Final_Score" if "IMP_Final_Score" in plot_df.columns else None

    fig.add_trace(
        go.Scatter(
            x=plot_df["SEI"],
            y=plot_df["BEI"],
            mode="markers",
            marker=dict(
                size=8,
                color=plot_df[color_col] if color_col else "#636EFA",
                colorscale="RdYlGn_r" if color_col else None,  # Red = high IMP (bad)
                showscale=True if color_col else False,
                colorbar=dict(title="IMP Score") if color_col else None,
                opacity=0.7,
            ),
            text=plot_df["Molecule_Name"]
            if "Molecule_Name" in plot_df.columns
            else None,
            hovertemplate="<b>%{text}</b><br>SEI: %{x:.2f}<br>BEI: %{y:.2f}<extra></extra>"
            if "Molecule_Name" in plot_df.columns
            else "SEI: %{x:.2f}<br>BEI: %{y:.2f}<extra></extra>",
            name="Compounds",
        )
    )

    # Add 45° reference line (optimal development angle)
    max_val = max(plot_df["SEI"].max(), plot_df["BEI"].max()) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="green", dash="dash", width=2),
            name="45° Optimal",
            hoverinfo="skip",
        )
    )

    # Add mean angle line from origin
    angle_rad = mean_angle * np.pi / 180
    line_length = mean_modulus * 1.2
    fig.add_trace(
        go.Scatter(
            x=[0, line_length * np.cos(angle_rad)],
            y=[0, line_length * np.sin(angle_rad)],
            mode="lines",
            line=dict(color="red", width=2),
            name=f"Mean Angle ({mean_angle:.1f}°)",
            hoverinfo="skip",
        )
    )

    # Add mean point marker
    fig.add_trace(
        go.Scatter(
            x=[mean_sei],
            y=[mean_bei],
            mode="markers",
            marker=dict(
                size=15,
                color="orange",
                symbol="star",
                line=dict(width=2, color="white"),
            ),
            name=f"Mean Point ({mean_sei:.1f}, {mean_bei:.1f})",
            hovertemplate="Mean SEI: %{x:.2f}<br>Mean BEI: %{y:.2f}<extra></extra>",
        )
    )

    # CRITICAL: Equal axis scaling so visual angle matches calculated angle
    fig.update_layout(
        height=500,
        margin=dict(t=30, b=30, l=30, r=30),
        xaxis=dict(
            title="SEI (Surface Efficiency Index)",
            scaleanchor="y",  # CRITICAL: Link x to y
            scaleratio=1,  # CRITICAL: 1:1 ratio
            range=[0, max_val],  # Start at 0 (SEI/BEI always positive)
            autorange=False,  # Disable autorange to enforce range
            constrain="domain",  # Constrain to specified range
        ),
        yaxis=dict(
            title="BEI (Binding Efficiency Index)",
            range=[0, max_val],  # Start at 0 (SEI/BEI always positive)
            autorange=False,  # Disable autorange to enforce range
            constrain="domain",  # Constrain to specified range
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    apply_impulator_theme(fig)
    st.plotly_chart(fig, width="stretch", key="report_efficiency_plane")

    # Enhanced interpretation with metrics cards
    st.markdown("#### Efficiency Plane Analysis")

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f"""
        <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #636EFA;">
            <div style="font-size: 1.2em; color: #636EFA; font-weight: bold;">{mean_angle:.1f}°</div>
            <div style="color: var(--text-color); font-size: 0.9em;">Mean Angle</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            f"""
        <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #ff7f0e;">
            <div style="font-size: 1.2em; color: #ff7f0e; font-weight: bold;">{mean_modulus:.1f}</div>
            <div style="color: var(--text-color); font-size: 0.9em;">Mean Modulus</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        st.markdown(
            f"""
        <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #2ca02c;">
            <div style="font-size: 1.2em; color: #2ca02c; font-weight: bold;">{mean_sei:.1f}</div>
            <div style="color: var(--text-color); font-size: 0.9em;">Mean SEI</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[3]:
        st.markdown(
            f"""
        <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #d62728;">
            <div style="font-size: 1.2em; color: #d62728; font-weight: bold;">{mean_bei:.1f}</div>
            <div style="color: var(--text-color); font-size: 0.9em;">Mean BEI</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("""
    **Angle Interpretation:**
    - **< 45°:** Compound favors **size efficiency** (SEI) - efficient use of polar surface area
    - **= 45°:** **Balanced development** (OPTIMAL) - equal efficiency in size and binding
    - **> 45°:** Compound favors **binding efficiency** (BEI) - efficient use of molecular weight

    **Note:** Most approved drugs have angles between 40-60°. The green dashed line shows the optimal 45° trajectory. The orange star marks the mean efficiency point.
    """)

    st.markdown("""
    **10PSA/MW — Compound Polarity Fingerprint:**

    The ratio 10PSA/MW is the fingerprint of the compound that corresponds to the profile of the binding pocket.

    - **Polar pockets:** The most fitting compounds will have **high PSA/MW** — indicating a larger proportion of polar surface area relative to molecular weight.
    - **Hydrophobic pockets:** The most fitting compounds will have **low PSA/MW** — indicating a compact, non-polar molecular profile.
    """)

    st.markdown("---")


def _render_report_pdb_evidence(df: pd.DataFrame, data: dict[str, Any]) -> None:
    """Render PDB structural evidence section."""
    st.markdown("### 🔬 PDB Structural Evidence")

    # Check for PDB columns
    pdb_cols = [
        "PDB_Score",
        "PDB_Num_Structures",
        "PDB_High_Quality",
        "PDB_Medium_Quality",
        "PDB_Poor_Quality",
    ]
    has_pdb = any(col in df.columns for col in pdb_cols)

    if not has_pdb:
        st.info("PDB structural evidence data not available")
        st.markdown("---")
        return

    # Try to load detailed PDB summary file for accurate counts
    pdb_summary_df = None
    compound_name = data.get("compound_name", "")
    entry_id = data.get("entry_id")
    storage_path = data.get("storage_path")

    try:
        safe_name = sanitize_compound_name(compound_name)
        for filename in [
            "pdb_summary.csv",
            f"{safe_name}_pdb_summary.csv",
            f"{safe_name}_pdb_details.csv",
        ]:
            pdb_summary_df = smart_load_dataframe(
                filename, entry_id=entry_id, storage_path=storage_path
            )
            if pdb_summary_df is not None and not pdb_summary_df.empty:
                break
    except Exception:
        pdb_summary_df = None

    # Calculate stats - use pdb_summary_df if available for accurate counts
    if pdb_summary_df is not None and not pdb_summary_df.empty:
        total_structures = len(pdb_summary_df)
        if "Quality" in pdb_summary_df.columns:
            high_quality = int((pdb_summary_df["Quality"] == "***").sum())
            medium_quality = int((pdb_summary_df["Quality"] == "**").sum())
            poor_quality = int((pdb_summary_df["Quality"] == "*").sum())
        else:
            if "Resolution" in pdb_summary_df.columns:
                pdb_summary_df["_res"] = pd.to_numeric(
                    pdb_summary_df["Resolution"], errors="coerce"
                )
                high_quality = int((pdb_summary_df["_res"] < 2.0).sum())
                medium_quality = int(
                    (
                        (pdb_summary_df["_res"] >= 2.0)
                        & (pdb_summary_df["_res"] <= 3.0)
                    ).sum()
                )
                poor_quality = int((pdb_summary_df["_res"] > 3.0).sum())
            else:
                high_quality = medium_quality = poor_quality = 0
    else:
        # Fallback to summing from dataframe (less accurate)
        total_structures = (
            int(df["PDB_Num_Structures"].sum())
            if "PDB_Num_Structures" in df.columns
            else 0
        )
        high_quality = (
            int(df["PDB_High_Quality"].sum()) if "PDB_High_Quality" in df.columns else 0
        )
        medium_quality = (
            int(df["PDB_Medium_Quality"].sum())
            if "PDB_Medium_Quality" in df.columns
            else 0
        )
        poor_quality = (
            int(df["PDB_Poor_Quality"].sum()) if "PDB_Poor_Quality" in df.columns else 0
        )

    mean_pdb_score = df["PDB_Score"].mean() if "PDB_Score" in df.columns else 0

    # Confidence assessment banner
    if mean_pdb_score >= 0.7:
        confidence = "HIGH CONFIDENCE"
        conf_color = "#28a745"
        conf_icon = "✓"
    elif mean_pdb_score >= 0.4:
        confidence = "MEDIUM CONFIDENCE"
        conf_color = "#fd7e14"
        conf_icon = "●"
    else:
        confidence = "LOW CONFIDENCE"
        conf_color = "#dc3545"
        conf_icon = "⚠️"

    st.markdown(
        f"""
    <div style="background-color: var(--secondary-background-color); padding: 12px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {conf_color};">
        <strong style="color: {conf_color};">{conf_icon} Structural Validation: {confidence} (PDB Score: {mean_pdb_score:.3f})</strong>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Quality distribution cards
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f"""
        <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #636EFA;">
            <div style="font-size: 1.5em; color: #636EFA; font-weight: bold;">{total_structures}</div>
            <div style="color: var(--text-color); font-size: 0.9em;">Total Structures</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        star_display = "⭐⭐⭐" if high_quality > 0 else ""
        st.markdown(
            f"""
        <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #28a745;">
            <div style="font-size: 1.5em; color: #28a745; font-weight: bold;">{high_quality}</div>
            <div style="color: var(--text-color); font-size: 0.9em;">High Quality</div>
            <div style="color: #28a745; font-size: 0.8em;">{star_display}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        star_display = "⭐⭐" if medium_quality > 0 else ""
        st.markdown(
            f"""
        <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #ffc107;">
            <div style="font-size: 1.5em; color: #ffc107; font-weight: bold;">{medium_quality}</div>
            <div style="color: var(--text-color); font-size: 0.9em;">Medium Quality</div>
            <div style="color: #ffc107; font-size: 0.8em;">{star_display}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cols[3]:
        star_display = "⭐" if poor_quality > 0 else ""
        st.markdown(
            f"""
        <div style="text-align: center; padding: 10px; background: var(--secondary-background-color); border-radius: 8px; border-left: 4px solid #dc3545;">
            <div style="font-size: 1.5em; color: #dc3545; font-weight: bold;">{poor_quality}</div>
            <div style="color: var(--text-color); font-size: 0.9em;">Poor Quality</div>
            <div style="color: #dc3545; font-size: 0.8em;">{star_display}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if total_structures > 0:
        st.markdown("")

        # Create horizontal bar chart for quality distribution (full-width)
        quality_data = pd.DataFrame(
            {
                "Quality": ["High (<2.0Å)", "Medium (2-3Å)", "Poor (>3Å)"],
                "Count": [high_quality, medium_quality, poor_quality],
                "Percentage": [
                    f"{high_quality / total_structures * 100:.1f}%"
                    if total_structures > 0
                    else "0%",
                    f"{medium_quality / total_structures * 100:.1f}%"
                    if total_structures > 0
                    else "0%",
                    f"{poor_quality / total_structures * 100:.1f}%"
                    if total_structures > 0
                    else "0%",
                ],
            }
        )

        fig = px.bar(
            quality_data,
            x="Count",
            y="Quality",
            orientation="h",
            color="Quality",
            text="Percentage",
            color_discrete_map={
                "High (<2.0Å)": "#28a745",
                "Medium (2-3Å)": "#ffc107",
                "Poor (>3Å)": "#dc3545",
            },
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            xaxis_title="Number of Structures",
            yaxis_title="",
        )
        apply_impulator_theme(fig)
        st.plotly_chart(fig, width="stretch", key="report_pdb_quality")

        # List high-quality PDB codes as clickable links
        high_q_pdb_ids = []
        if (
            pdb_summary_df is not None
            and not pdb_summary_df.empty
            and "PDB_ID" in pdb_summary_df.columns
        ):
            if "Quality" in pdb_summary_df.columns:
                high_q_pdb_ids = (
                    pdb_summary_df[pdb_summary_df["Quality"] == "***"]["PDB_ID"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            elif "Resolution" in pdb_summary_df.columns:
                res_col = pd.to_numeric(pdb_summary_df["Resolution"], errors="coerce")
                high_q_pdb_ids = (
                    pdb_summary_df[res_col < 2.0]["PDB_ID"].dropna().unique().tolist()
                )

        if high_q_pdb_ids:
            pdb_links = ", ".join(
                f"[{pid}](https://www.rcsb.org/structure/{pid})"
                for pid in sorted(high_q_pdb_ids)
            )
            st.markdown(f"**High-Quality PDB Structures (<2.0Å):** {pdb_links}")

        # List medium-quality PDB codes as clickable links
        med_q_pdb_ids = []
        if (
            pdb_summary_df is not None
            and not pdb_summary_df.empty
            and "PDB_ID" in pdb_summary_df.columns
        ):
            if "Quality" in pdb_summary_df.columns:
                med_q_pdb_ids = (
                    pdb_summary_df[pdb_summary_df["Quality"] == "**"]["PDB_ID"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            elif "Resolution" in pdb_summary_df.columns:
                res_col = pd.to_numeric(pdb_summary_df["Resolution"], errors="coerce")
                med_q_pdb_ids = (
                    pdb_summary_df[(res_col >= 2.0) & (res_col <= 3.0)]["PDB_ID"]
                    .dropna()
                    .unique()
                    .tolist()
                )

        if med_q_pdb_ids:
            med_pdb_links = ", ".join(
                f"[{pid}](https://www.rcsb.org/structure/{pid})"
                for pid in sorted(med_q_pdb_ids)
            )
            st.markdown(f"**Medium-Resolution PDB Structures (2-3Å):** {med_pdb_links}")

        # Resolution Quality info box below
        st.markdown(
            """
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #1976d2; margin-top: 15px; color: #0d47a1;">
            <strong style="color: #0d47a1;">📊 Resolution Quality:</strong><br>
            High-resolution structures (&lt;2.0Å) provide the most reliable structural validation.
            PDB Score component contributes 5% to final IMP score.
            <ul style="margin-top: 10px; margin-bottom: 5px; color: #0d47a1;">
                <li><strong>⭐⭐⭐⭐⭐ High (&lt;2.0Å):</strong> Excellent resolution - high confidence in binding mode</li>
                <li><strong>⭐⭐⭐ Medium (2-3Å):</strong> Good resolution - reliable structural information</li>
                <li><strong>⭐ Poor (&gt;3Å):</strong> Lower resolution - general binding information only</li>
            </ul>
            <strong style="color: #0d47a1;">Note:</strong> High PDB scores (&gt;0.7) indicate strong structural validation with multiple high-resolution crystal structures, providing confidence that the compound genuinely binds to the target (not an assay artifact).
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No PDB structures found for these compounds. This is common for early-stage research compounds not yet structurally characterized."
        )

    st.markdown("---")


def _render_report_classification(df: pd.DataFrame) -> None:
    """Render chemical classification section (ClassyFire + NPClassifier)."""
    st.markdown("### 🧬 Chemical Classification")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ClassyFire Taxonomy:**")
        classyfire_cols = ["Kingdom", "Superclass", "Class", "Subclass"]
        has_classyfire = any(col in df.columns for col in classyfire_cols)

        if has_classyfire:
            for col in classyfire_cols:
                if col in df.columns:
                    # Get most common value
                    mode_vals = df[col].mode()
                    value = mode_vals.iloc[0] if not mode_vals.empty else "N/A"
                    st.markdown(f"- **{col}:** {value}")
        else:
            st.info("ClassyFire data not available")

    with col2:
        st.markdown("**NPClassifier (Natural Products):**")
        np_cols = ["NP_Pathway", "NP_Superclass", "NP_Class"]
        has_np = any(col in df.columns for col in np_cols)

        if has_np:
            for col in np_cols:
                if col in df.columns:
                    display_name = col.replace("NP_", "")
                    mode_vals = df[col].mode()
                    value = mode_vals.iloc[0] if not mode_vals.empty else "N/A"
                    st.markdown(f"- **{display_name}:** {value}")
        else:
            st.info("NPClassifier data not available")

    st.markdown("---")


def _render_report_indications(data: dict[str, Any]) -> None:
    """Render drug indications section."""
    st.markdown("### 💊 Drug Indications")

    indications_df = data.get("indications")

    if indications_df is None or (
        isinstance(indications_df, pd.DataFrame) and indications_df.empty
    ):
        st.info("No drug indication data available")
        st.markdown("---")
        return

    # Get max phase
    max_phase = (
        indications_df["Max_Phase"].max()
        if "Max_Phase" in indications_df.columns
        else "N/A"
    )
    unique_indications = (
        indications_df["MESH_Heading"].nunique()
        if "MESH_Heading" in indications_df.columns
        else 0
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        **Maximum Clinical Phase:** {max_phase}

        **Unique Indications:** {unique_indications}

        **Compounds with Data:** {indications_df["ChEMBL_ID"].nunique() if "ChEMBL_ID" in indications_df.columns else "N/A"}
        """)

    with col2:
        # Top indications table
        if (
            "MESH_Heading" in indications_df.columns
            and "Max_Phase" in indications_df.columns
        ):
            top_indications = (
                indications_df.groupby("MESH_Heading")["Max_Phase"]
                .max()
                .sort_values(ascending=False)
                .head(5)
            )

            if not top_indications.empty:
                st.markdown("**Top Indications by Phase:**")
                table_data = [
                    {"Indication": ind, "Max Phase": phase}
                    for ind, phase in top_indications.items()
                ]
                st.table(pd.DataFrame(table_data))

    st.markdown("---")


def _render_report_recommendation(df: pd.DataFrame, mean_score: float) -> None:
    """Render final recommendation section with full score-card stack."""
    st.markdown("### 🎯 Final Recommendation")

    # Full score-card stack (UI-SPEC Component Visual Contract item #1)
    mean_score_int = format_imp_score(mean_score) if pd.notna(mean_score) else None
    if "IMP_Final_Score" in df.columns:
        valid_scores = df["IMP_Final_Score"].dropna()
        observed_min = float(valid_scores.min()) if not valid_scores.empty else None
        observed_max = float(valid_scores.max()) if not valid_scores.empty else None
    else:
        observed_min = None
        observed_max = None

    if mean_score_int is None:
        st.html(
            '<div style="font-size:28px;font-weight:600;color:#6b7280;">IMP Score: N/A</div>'
        )
    else:
        obs_min_int = (
            format_imp_score(observed_min) if observed_min is not None else None
        )
        obs_max_int = (
            format_imp_score(observed_max) if observed_max is not None else None
        )
        obs_min_str = str(obs_min_int) if obs_min_int is not None else "—"
        obs_max_str = str(obs_max_int) if obs_max_int is not None else "—"
        global_bar_svg = render_imp_range_bar_global(mean_score)
        dynamic_bar_svg = render_imp_range_bar_dynamic(
            mean_score, observed_min, observed_max
        )
        # st._main._html (iframe) bypasses DOMPurify which would strip <defs>/<linearGradient>
        st._main._html(
            "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;\">"
            '<div style="font-size:14px; font-weight:500; color:#6b7280; letter-spacing:0.02em; text-transform:uppercase;">IMP Score</div>'
            f'<div style="font-size:48px; font-weight:700; color:#111827; line-height:1; margin-top:2px;">{mean_score_int}</div>'
            '<div style="font-size:14px; font-weight:400; color:#6b7280; margin-top:8px;">Global reference (10–80)</div>'
            f"{global_bar_svg}"
            f'<div style="display:flex; justify-content:space-between; max-width:240px; font-size:14px; color:#6b7280; margin-top:4px;"><span>{IMP_SCORE_FLOOR}</span><span>{IMP_SCORE_CEILING}</span></div>'
            f'<div style="font-size:14px; font-weight:400; color:#6b7280; margin-top:16px;">This query\'s range ({obs_min_str}–{obs_max_str})</div>'
            f"{dynamic_bar_svg}"
            '<div style="display:flex; justify-content:space-between; max-width:240px; font-size:14px; color:#6b7280; margin-top:4px;"><span>0</span><span>100</span></div>'
            "</div>",
            height=240,
        )

    # Recommended actions based on score and flags
    st.markdown("**Recommended Actions:**")

    actions = []

    if mean_score >= 0.7:
        actions.append(("HIGH", "Validate with orthogonal binding assay (SPR/ITC/MST)"))
        actions.append(("HIGH", "Counter-screen against aggregation"))

    # Check for PAINS - use correct column name
    if "PAINS_Violation" in df.columns and df["PAINS_Violation"].any():
        actions.append(("HIGH", "Counter-screen PAINS-flagged compounds"))

    if "QED" in df.columns and df["QED"].mean() < 0.5:
        actions.append(("LOW", "Consider SAR optimization to improve drug-likeness"))

    if "PDB_Score" in df.columns and df["PDB_Score"].mean() < 0.3:
        actions.append(
            ("LOW", "Obtain structural evidence (X-ray/cryo-EM) before advancing")
        )

    # Display actions
    for priority, action in actions:
        if priority == "HIGH":
            st.markdown(f"🔴 **[{priority}]** {action}")
        elif priority == "MEDIUM":
            st.markdown(f"🟠 **[{priority}]** {action}")
        elif priority == "LOW":
            st.markdown(f"🟡 **[{priority}]** {action}")
        else:
            st.markdown(f"🟢 **[{priority}]** {action}")

    # Interpretation guide
    st.markdown("---")
    st.markdown("**IMP Interpretation Guide:**")
    st.caption(
        "The IMP score is a continuous 0–100 measure; the team has chosen not to define qualitative thresholds."
    )


def _export_plotly_to_base64(
    fig, width: int = 700, height: int = 400, scale: float = 3.0
) -> str:
    """Export a Plotly figure to base64 PNG string using kaleido.

    Args:
        fig: Plotly figure to export
        width: Width in pixels (logical size)
        height: Height in pixels (logical size)
        scale: Scale factor for high-DPI export (default 3.0 = ~288 DPI)
               Higher values = better quality but larger file size
               1.0 = 96 DPI, 2.0 = 192 DPI, 3.0 = 288 DPI
    """
    try:
        import base64

        # Update figure for static export (white background, anti-aliased text)
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#333",
            font_size=12,  # Slightly larger font for better readability in print
        )

        # Export with high DPI (scale parameter increases resolution)
        # Plotly v6 uses kaleido v1 as default engine
        img_bytes = fig.to_image(
            format="png",
            width=width,
            height=height,
            scale=scale,  # High-quality export (3x = 288 DPI)
        )
        img_b64 = base64.b64encode(img_bytes).decode()
        return f'<img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto;">'
    except Exception as e:
        return f"<p style='color: #999;'>Chart unavailable: {html.escape(str(e))}</p>"


def _create_html_bioactivity_donut(df: pd.DataFrame) -> str:
    """Create bioactivity donut chart for HTML export."""
    if "Activity_Type" not in df.columns:
        return "<p>Activity type data not available</p>"

    type_counts = df["Activity_Type"].value_counts().head(6)
    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    theme = get_plotly_theme()
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        template=theme["template"],
        height=400,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=True,
    )
    return _export_plotly_to_base64(fig, 800, 400)


def _create_html_efficiency_boxplots(df: pd.DataFrame) -> str:
    """Create efficiency metrics box plots for HTML export."""
    metrics = ["SEI", "BEI", "NSEI", "NBEI"]
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return "<p>Efficiency metrics not available</p>"

    fig = go.Figure()
    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for i, metric in enumerate(available_metrics):
        values = df[metric].dropna()
        if len(values) > 0:
            fig.add_trace(
                go.Box(
                    y=values,
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    boxpoints="outliers",
                )
            )

    fig.update_layout(
        height=400,
        margin=dict(t=30, b=50, l=50, r=30),
        showlegend=False,
        yaxis_title="Value",
    )
    return _export_plotly_to_base64(fig, 900, 400)


def _create_html_efficiency_scatter(df: pd.DataFrame) -> str:
    """Create SEI vs BEI scatter plot with equal axis scaling for HTML export."""
    if "SEI" not in df.columns or "BEI" not in df.columns:
        return "<p>SEI/BEI data not available</p>"

    plot_df = df[["SEI", "BEI"]].dropna()
    if plot_df.empty:
        return "<p>No valid SEI/BEI data</p>"

    # Get color data if available
    if "IMP_Final_Score" in df.columns:
        plot_df = df[["SEI", "BEI", "IMP_Final_Score"]].dropna()
        color_col = "IMP_Final_Score"
    else:
        color_col = None

    # Calculate mean values
    mean_sei = plot_df["SEI"].mean()
    mean_bei = plot_df["BEI"].mean()
    mean_angle = np.degrees(np.arctan2(mean_bei, mean_sei))
    mean_modulus = np.sqrt(mean_sei**2 + mean_bei**2)

    fig = go.Figure()

    # Add data points
    if color_col:
        fig.add_trace(
            go.Scatter(
                x=plot_df["SEI"],
                y=plot_df["BEI"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=plot_df[color_col],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="IMP Score"),
                ),
                name="Compounds",
                hovertemplate="SEI: %{x:.2f}<br>BEI: %{y:.2f}<br>IMP Score: %{marker.color:.3f}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_df["SEI"],
                y=plot_df["BEI"],
                mode="markers",
                marker=dict(size=8, color="#3498db"),
                name="Compounds",
                hovertemplate="SEI: %{x:.2f}<br>BEI: %{y:.2f}<extra></extra>",
            )
        )

    # Add 45° reference line
    max_val = max(plot_df["SEI"].max(), plot_df["BEI"].max()) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name="45° Optimal Line",
            hovertemplate="45° Optimal (Balanced Development)<extra></extra>",
        )
    )

    # Add mean point marker (orange star)
    fig.add_trace(
        go.Scatter(
            x=[mean_sei],
            y=[mean_bei],
            mode="markers",
            marker=dict(
                size=15,
                color="orange",
                symbol="star",
                line=dict(width=2, color="white"),
            ),
            name=f"Mean Point ({mean_sei:.1f}, {mean_bei:.1f})",
            hovertemplate=f"Mean SEI: {mean_sei:.2f}<br>Mean BEI: {mean_bei:.2f}<br>Angle: {mean_angle:.1f}°<br>Modulus: {mean_modulus:.2f}<extra></extra>",
        )
    )

    theme = get_plotly_theme()
    fig.update_layout(
        template=theme["template"],
        height=600,
        margin=dict(t=40, b=50, l=60, r=30),
        xaxis=dict(
            title="SEI (Surface Efficiency Index)",
            scaleanchor="y",  # CRITICAL: Link x to y
            scaleratio=1,  # CRITICAL: 1:1 ratio
            range=[0, max_val],  # Start at 0 (SEI/BEI always positive)
            autorange=False,  # Disable autorange to enforce range
            constrain="domain",  # Constrain to specified range
        ),
        yaxis=dict(
            title="BEI (Binding Efficiency Index)",
            range=[0, max_val],  # Start at 0
            autorange=False,
            constrain="domain",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor=theme["legend_bgcolor"],
            bordercolor=theme["legend_bordercolor"],
            borderwidth=1,
        ),
    )

    return _export_plotly_to_base64(fig, 900, 600)


def _create_html_pdb_quality_bar(df: pd.DataFrame, data: dict[str, Any]) -> str:
    """Create PDB quality distribution bar chart for HTML export."""
    # Try to load pdb_summary for accurate counts
    pdb_summary_df = None
    compound_name = data.get("compound_name", "")
    entry_id = data.get("entry_id")
    storage_path = data.get("storage_path")

    try:
        safe_name = sanitize_compound_name(compound_name)
        for filename in ["pdb_summary.csv", f"{safe_name}_pdb_summary.csv"]:
            pdb_summary_df = smart_load_dataframe(
                filename, entry_id=entry_id, storage_path=storage_path
            )
            if pdb_summary_df is not None and not pdb_summary_df.empty:
                break
    except Exception:
        pass

    if (
        pdb_summary_df is not None
        and not pdb_summary_df.empty
        and "Quality" in pdb_summary_df.columns
    ):
        high_q = int((pdb_summary_df["Quality"] == "***").sum())
        med_q = int((pdb_summary_df["Quality"] == "**").sum())
        poor_q = int((pdb_summary_df["Quality"] == "*").sum())
    elif "PDB_High_Quality" in df.columns:
        high_q = (
            int(df["PDB_High_Quality"].max())
            if df["PDB_High_Quality"].notna().any()
            else 0
        )
        med_q = (
            int(df["PDB_Medium_Quality"].max())
            if "PDB_Medium_Quality" in df.columns
            and df["PDB_Medium_Quality"].notna().any()
            else 0
        )
        poor_q = (
            int(df["PDB_Poor_Quality"].max())
            if "PDB_Poor_Quality" in df.columns and df["PDB_Poor_Quality"].notna().any()
            else 0
        )
    else:
        return "<p>PDB quality data not available</p>"

    if high_q + med_q + poor_q == 0:
        return "<p>No PDB structures found</p>"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["High (<2.0Å)", "Medium (2-3Å)", "Poor (>3Å)"],
            x=[high_q, med_q, poor_q],
            orientation="h",
            marker_color=["#28a745", "#ffc107", "#dc3545"],
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(t=20, b=30, l=100, r=30),
        xaxis_title="Count",
        showlegend=False,
    )
    return _export_plotly_to_base64(fig, 700, 250)


def _generate_html_report(data: dict[str, Any], df: pd.DataFrame) -> str:
    """Generate comprehensive HTML report with ALL sections matching Report tab."""
    import base64
    import io
    from datetime import datetime

    compound_name = data.get("compound_name", "Unknown")
    smiles = data.get("smiles", "")
    summary = data.get("summary", {})

    # Get BEST scoring compound row (matches Overview behavior)
    best_row = None
    if "IMP_Final_Score" in df.columns:
        valid_df = df.dropna(subset=["IMP_Final_Score"])
        if not valid_df.empty:
            best_row = valid_df.loc[valid_df["IMP_Final_Score"].idxmax()]

    # Calculate scores from best compound
    final_score = best_row["IMP_Final_Score"] if best_row is not None else 0
    qed_val = best_row["QED"] if best_row is not None and "QED" in best_row.index else 0

    # Observed bounds for the dynamic range bar (visual parity with Streamlit page).
    if "IMP_Final_Score" in df.columns:
        _valid_scores = df["IMP_Final_Score"].dropna()
        observed_min = float(_valid_scores.min()) if not _valid_scores.empty else None
        observed_max = float(_valid_scores.max()) if not _valid_scores.empty else None
    else:
        observed_min = None
        observed_max = None

    final_score_int = format_imp_score(final_score) if pd.notna(final_score) else None
    obs_min_int = format_imp_score(observed_min) if observed_min is not None else None
    obs_max_int = format_imp_score(observed_max) if observed_max is not None else None
    obs_min_str = str(obs_min_int) if obs_min_int is not None else "—"
    obs_max_str = str(obs_max_int) if obs_max_int is not None else "—"
    # Identical SVG strings flow into both Streamlit and HTML report (UI-SPEC Visual Parity Contract item #1)
    global_bar_svg = render_imp_range_bar_global(final_score)
    dynamic_bar_svg = render_imp_range_bar_dynamic(
        final_score, observed_min, observed_max
    )

    # Generate 2D structure image
    structure_img_html = "<p>Structure unavailable</p>"
    if smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # High-quality image (2x size for better resolution in HTML/PDF)
                img = Draw.MolToImage(mol, size=(600, 500))
                buffered = io.BytesIO()
                # Save with high quality
                img.save(buffered, format="PNG", optimize=False, quality=95)
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                # Display at 300x250 but use 600x500 source for crisp rendering
                structure_img_html = f'<img src="data:image/png;base64,{img_b64}" style="width: 300px; height: 250px; border: 1px solid #ddd; border-radius: 8px;">'
        except Exception:
            pass

    # Use pre-computed InChI and InChIKey from data (escaped for HTML safety)
    inchikey = html.escape(data.get("inchikey") or "N/A")
    inchi = html.escape(data.get("inchi") or "N/A")

    # Compute summary stats for header (matching Overview quick stats)
    # Use summary data as source of truth (same as Overview header)
    similar_count = summary.get(
        "similar_count",
        df["ChEMBL_ID"].nunique() if "ChEMBL_ID" in df.columns else len(df),
    )
    activities_count = summary.get("total_activities", len(df))
    avg_qed = summary.get("qed") or (df["QED"].mean() if "QED" in df.columns else None)
    best_imp_score = (
        df["IMP_Final_Score"].max() if "IMP_Final_Score" in df.columns else None
    )

    # Count unique IMP compounds (not activity rows)
    imp_count = 0
    has_warning = False
    if "Is_IMP_Candidate" in df.columns and "ChEMBL_ID" in df.columns:
        imp_count = df[df["Is_IMP_Candidate"]]["ChEMBL_ID"].nunique()
        has_warning = imp_count > 0
    elif summary.get("has_imp_candidates", False):
        imp_count = summary.get("imp_candidates", 0)
        has_warning = True

    # Format stats — IMP score as integer (PRES-04)
    avg_qed_str = (
        f"{avg_qed:.2f}" if avg_qed is not None and not pd.isna(avg_qed) else "N/A"
    )
    _best_score_int = (
        format_imp_score(best_imp_score)
        if best_imp_score is not None and not pd.isna(best_imp_score)
        else None
    )
    avg_imp_score_str = str(_best_score_int) if _best_score_int is not None else "N/A"
    # Pre-rendered header score-card global bar (matches Streamlit overview compact variant)
    header_global_bar_svg = (
        render_imp_range_bar_global(best_imp_score)
        if _best_score_int is not None
        else ""
    )

    # Build properties table from best compound
    props_html = ""
    if best_row is not None:
        prop_cols = [
            ("pActivity", "pActivity", "-log10(IC50)"),
            ("Molecular_Weight", "Molecular Weight", "g/mol"),
            ("TPSA", "PSA (TPSA)", "Å²"),
            ("Heavy_Atoms", "Heavy Atoms", "count"),
            ("NPOL", "N+O Atoms (NPOL)", "Heteroatom count (Num of Polar Atoms)"),
            ("QED", "QED", "Drug-likeness"),
        ]
        for col, label, unit in prop_cols:
            if col in best_row.index and not pd.isna(best_row[col]):
                props_html += f"<tr><td>{label}</td><td>{best_row[col]:.3f}</td><td>{unit}</td></tr>"

        # Add computed 10PSA/MW
        tpsa_br = best_row["TPSA"] if "TPSA" in best_row.index else None
        mw_br = (
            best_row["Molecular_Weight"]
            if "Molecular_Weight" in best_row.index
            else None
        )
        if (
            tpsa_br is not None
            and mw_br is not None
            and not pd.isna(tpsa_br)
            and not pd.isna(mw_br)
            and mw_br > 0
        ):
            psa_mw_val = 10 * tpsa_br / mw_br
            props_html += f"<tr><td>10PSA/MW</td><td>{psa_mw_val:.3f}</td><td>Compound Polarity vs Atom Fingerprint (= BEI/SEI)</td></tr>"

    # Build efficiency metrics table from best compound
    efficiency_html = ""
    if best_row is not None:
        eff_cols = [
            ("SEI", "SEI", "pActivity × 100 / PSA", True),
            ("BEI", "BEI", "pActivity × 1000 / MW", True),
            ("NSEI", "NSEI", "pActivity / NPOL", False),
            ("NBEI", "NBEI", "pActivity / Heavy Atoms", False),
        ]
        for col, label, formula, used in eff_cols:
            if col in best_row.index and not pd.isna(best_row[col]):
                used_text = "✓ Used" if used else "Display only"
                efficiency_html += f"<tr><td>{label}</td><td>{best_row[col]:.3f}</td><td>{formula}</td><td>{used_text}</td></tr>"

    # Build component scores table from best compound
    components_html = ""
    if best_row is not None:
        components = [
            ("Efficiency_Score", "Efficiency", "45%"),
            ("Distance_Score", "Distance", "20%"),
            ("Angle_Score", "Angle", "15%"),
            ("Interference_Score", "Interference", "15%"),
            ("PDB_Score", "PDB Evidence", "5%"),
        ]
        for col, name, weight in components:
            if col in best_row.index and not pd.isna(best_row[col]):
                contrib_col = col.replace("_Score", "_Contribution")
                contrib = (
                    best_row[contrib_col] if contrib_col in best_row.index else None
                )
                contrib_str = (
                    f"{contrib:.3f}"
                    if contrib is not None and not pd.isna(contrib)
                    else "N/A"
                )
                components_html += f"<tr><td>{name}</td><td>{best_row[col]:.3f}</td><td>{weight}</td><td>{contrib_str}</td></tr>"

    # Red flags section - count UNIQUE COMPOUNDS (not all rows)
    red_flags_html = ""
    flag_cols = [
        ("PAINS_Violation", "PAINS", "Pan-Assay Interference"),
        ("Aggregator_Risk", "Aggregator", "Colloidal Aggregation"),
        ("Redox_Reactive", "Redox", "Redox Cycling"),
        ("Fluorescence_Interference", "Fluorescence", "Fluorescence Interference"),
        ("Thiol_Reactive", "Thiol", "Thiol Reactivity"),
        ("BRENK_Alerts", "BRENK", "Unwanted Substructures"),
        ("NIH_Alerts", "NIH", "NIH Problematic Groups"),
    ]
    total_flags = 0
    unique_df_flags = (
        df.drop_duplicates("ChEMBL_ID") if "ChEMBL_ID" in df.columns else df
    )
    for col, name, desc in flag_cols:
        if col in unique_df_flags.columns:
            count = int(
                unique_df_flags[col].sum()
                if unique_df_flags[col].dtype == bool
                else unique_df_flags[col].astype(bool).sum()
            )
            total_flags += count
            status = f"⚠️ {count} flagged" if count > 0 else "✓ Clean"
            color_style = "color: #dc3545;" if count > 0 else "color: #28a745;"
            red_flags_html += f"<tr><td>{name}</td><td style='{color_style}'>{status}</td><td>{desc}</td></tr>"

    flag_assessment = (
        "LOW CONCERN"
        if total_flags == 0
        else ("MODERATE CONCERN" if total_flags <= 5 else "HIGH CONCERN")
    )
    flag_color = (
        "#28a745"
        if total_flags == 0
        else ("#fd7e14" if total_flags <= 5 else "#dc3545")
    )

    # Bioactivity distribution
    bioactivity_html = ""
    if "Activity_Type" in df.columns:
        type_counts = df["Activity_Type"].value_counts().head(5)
        for stype, count in type_counts.items():
            pct = count / len(df) * 100
            bioactivity_html += f"<tr><td>{html.escape(str(stype))}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"

    # Efficiency metrics statistics
    efficiency_metrics_stats = {}
    metric_colors_html = {
        "SEI": "#1f77b4",
        "BEI": "#2ca02c",
        "NSEI": "#ff7f0e",
        "NBEI": "#9467bd",
    }
    for metric in ["SEI", "BEI", "NSEI", "NBEI"]:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                efficiency_metrics_stats[metric] = {
                    "mean": vals.mean(),
                    "min": vals.min(),
                    "max": vals.max(),
                    "used": metric in ["SEI", "BEI"],
                }

    # Efficiency plane summary - compute from full dataset
    mean_sei = df["SEI"].mean() if "SEI" in df.columns else None
    mean_bei = df["BEI"].mean() if "BEI" in df.columns else None

    if (
        mean_sei is not None
        and mean_bei is not None
        and not pd.isna(mean_sei)
        and not pd.isna(mean_bei)
    ):
        angle_val = np.degrees(np.arctan2(mean_bei, mean_sei))
        modulus_val = np.sqrt(mean_sei**2 + mean_bei**2)
    else:
        angle_val = (
            best_row["Angle_SEI_BEI"]
            if best_row is not None and "Angle_SEI_BEI" in best_row.index
            else None
        )
        modulus_val = (
            best_row["Modulus_SEI_BEI"]
            if best_row is not None and "Modulus_SEI_BEI" in best_row.index
            else None
        )

    # Pre-compute formatted strings for angle and modulus
    angle_str = (
        f"{angle_val:.1f}"
        if angle_val is not None and not pd.isna(angle_val)
        else "N/A"
    )
    modulus_str = (
        f"{modulus_val:.2f}"
        if modulus_val is not None and not pd.isna(modulus_val)
        else "N/A"
    )
    mean_sei_str = (
        f"{mean_sei:.2f}" if mean_sei is not None and not pd.isna(mean_sei) else "N/A"
    )
    mean_bei_str = (
        f"{mean_bei:.2f}" if mean_bei is not None and not pd.isna(mean_bei) else "N/A"
    )

    # Compute angle assessment
    if angle_val is not None and not pd.isna(angle_val):
        if 40 <= angle_val <= 50:
            angle_status = "OPTIMAL ✓"
            angle_status_color = "#28a745"
            angle_status_bg = "#d4edda"
        elif 35 <= angle_val <= 55:
            angle_status = "ACCEPTABLE"
            angle_status_color = "#17a2b8"
            angle_status_bg = "#d1ecf1"
        else:
            angle_status = "UNBALANCED"
            angle_status_color = "#fd7e14"
            angle_status_bg = "#fff3cd"
    else:
        angle_status = "N/A"
        angle_status_color = "#6c757d"
        angle_status_bg = "#e9ecef"

    # PDB evidence - try to load pdb_summary for accurate counts
    pdb_total = 0
    high_q = med_q = poor_q = 0

    # Try to load pdb_summary file for accurate counts
    pdb_summary_df_html = None
    try:
        safe_name = sanitize_compound_name(compound_name)
        entry_id = data.get("entry_id")
        storage_path = data.get("storage_path")
        for filename in [
            "pdb_summary.csv",
            f"{safe_name}_pdb_summary.csv",
            f"{safe_name}_pdb_details.csv",
        ]:
            pdb_summary_df_html = smart_load_dataframe(
                filename, entry_id=entry_id, storage_path=storage_path
            )
            if pdb_summary_df_html is not None and not pdb_summary_df_html.empty:
                break
    except Exception:
        pdb_summary_df_html = None

    if pdb_summary_df_html is not None and not pdb_summary_df_html.empty:
        pdb_total = len(pdb_summary_df_html)
        if "Quality" in pdb_summary_df_html.columns:
            high_q = int((pdb_summary_df_html["Quality"] == "***").sum())
            med_q = int((pdb_summary_df_html["Quality"] == "**").sum())
            poor_q = int((pdb_summary_df_html["Quality"] == "*").sum())
        elif "Resolution" in pdb_summary_df_html.columns:
            pdb_summary_df_html["_res"] = pd.to_numeric(
                pdb_summary_df_html["Resolution"], errors="coerce"
            )
            high_q = int((pdb_summary_df_html["_res"] < 2.0).sum())
            med_q = int(
                (
                    (pdb_summary_df_html["_res"] >= 2.0)
                    & (pdb_summary_df_html["_res"] <= 3.0)
                ).sum()
            )
            poor_q = int((pdb_summary_df_html["_res"] > 3.0).sum())
    else:
        # Fallback to dataframe columns
        if "PDB_Num_Structures" in df.columns:
            pdb_total = (
                int(df["PDB_Num_Structures"].max())
                if df["PDB_Num_Structures"].notna().any()
                else 0
            )
        if (
            "PDB_High_Quality" in df.columns
            and "PDB_Medium_Quality" in df.columns
            and "PDB_Poor_Quality" in df.columns
        ):
            high_q = (
                int(df["PDB_High_Quality"].max())
                if df["PDB_High_Quality"].notna().any()
                else 0
            )
            med_q = (
                int(df["PDB_Medium_Quality"].max())
                if df["PDB_Medium_Quality"].notna().any()
                else 0
            )
            poor_q = (
                int(df["PDB_Poor_Quality"].max())
                if df["PDB_Poor_Quality"].notna().any()
                else 0
            )

    # Calculate PDB confidence
    mean_pdb_score = (
        df["PDB_Score"].mean()
        if "PDB_Score" in df.columns and df["PDB_Score"].notna().any()
        else 0
    )
    if mean_pdb_score >= 0.7:
        pdb_confidence = "HIGH CONFIDENCE"
        pdb_conf_icon = "✓✓✓"
        pdb_conf_color = "#28a745"
        pdb_conf_bg = "#d4edda"
    elif mean_pdb_score >= 0.4:
        pdb_confidence = "MODERATE CONFIDENCE"
        pdb_conf_icon = "✓✓"
        pdb_conf_color = "#17a2b8"
        pdb_conf_bg = "#d1ecf1"
    elif mean_pdb_score > 0:
        pdb_confidence = "LOW CONFIDENCE"
        pdb_conf_icon = "✓"
        pdb_conf_color = "#fd7e14"
        pdb_conf_bg = "#fff3cd"
    else:
        pdb_confidence = "NO STRUCTURAL DATA"
        pdb_conf_icon = "✗"
        pdb_conf_color = "#6c757d"
        pdb_conf_bg = "#e9ecef"

    # PDB quality percentages
    high_q_pct = (high_q / pdb_total * 100) if pdb_total > 0 else 0
    med_q_pct = (med_q / pdb_total * 100) if pdb_total > 0 else 0
    poor_q_pct = (poor_q / pdb_total * 100) if pdb_total > 0 else 0

    # Extract high-quality PDB codes for clickable links
    high_q_pdb_links_html = ""
    if (
        pdb_summary_df_html is not None
        and not pdb_summary_df_html.empty
        and "PDB_ID" in pdb_summary_df_html.columns
    ):
        if "Quality" in pdb_summary_df_html.columns:
            hq_ids = (
                pdb_summary_df_html[pdb_summary_df_html["Quality"] == "***"]["PDB_ID"]
                .dropna()
                .unique()
                .tolist()
            )
        elif "Resolution" in pdb_summary_df_html.columns:
            res_col = pd.to_numeric(pdb_summary_df_html["Resolution"], errors="coerce")
            hq_ids = (
                pdb_summary_df_html[res_col < 2.0]["PDB_ID"].dropna().unique().tolist()
            )
        else:
            hq_ids = []
        if hq_ids:
            links = ", ".join(
                f'<a href="https://www.rcsb.org/structure/{html.escape(str(pid))}" target="_blank">'
                f"{html.escape(str(pid))}</a>"
                for pid in sorted(hq_ids)
            )
            high_q_pdb_links_html = f"<p><strong>High-Quality PDB Structures (&lt;2.0Å):</strong> {links}</p>"

    # Extract medium-quality PDB codes for clickable links
    med_q_pdb_links_html = ""
    if (
        pdb_summary_df_html is not None
        and not pdb_summary_df_html.empty
        and "PDB_ID" in pdb_summary_df_html.columns
    ):
        if "Quality" in pdb_summary_df_html.columns:
            mq_ids = (
                pdb_summary_df_html[pdb_summary_df_html["Quality"] == "**"]["PDB_ID"]
                .dropna()
                .unique()
                .tolist()
            )
        elif "Resolution" in pdb_summary_df_html.columns:
            res_col = pd.to_numeric(pdb_summary_df_html["Resolution"], errors="coerce")
            mq_ids = (
                pdb_summary_df_html[(res_col >= 2.0) & (res_col <= 3.0)]["PDB_ID"]
                .dropna()
                .unique()
                .tolist()
            )
        else:
            mq_ids = []
        if mq_ids:
            med_links = ", ".join(
                f'<a href="https://www.rcsb.org/structure/{html.escape(str(pid))}" target="_blank">'
                f"{html.escape(str(pid))}</a>"
                for pid in sorted(mq_ids)
            )
            med_q_pdb_links_html = f"<p><strong>Medium-Resolution PDB Structures (2-3Å):</strong> {med_links}</p>"

    # Classification - ClassyFire and NPClassifier
    classyfire_html = ""
    class_cols = ["Kingdom", "Superclass", "Class", "Subclass"]
    for col in class_cols:
        if col in df.columns and df[col].notna().any():
            val = html.escape(str(df[col].iloc[0]))
            classyfire_html += f"<tr><td>{col}</td><td>{val}</td></tr>"

    # NPClassifier
    npclassifier_html = ""
    np_cols = ["NP_Pathway", "NP_Superclass", "NP_Class"]
    np_labels = ["Pathway", "Superclass", "Class"]
    for col, label in zip(np_cols, np_labels):
        if col in df.columns and df[col].notna().any():
            val = html.escape(str(df[col].iloc[0]))
            npclassifier_html += f"<tr><td>{label}</td><td>{val}</td></tr>"

    # Drug indications
    indications_html = ""
    max_clinical_phase = 0
    unique_indications = 0
    compounds_with_indications = 0

    indications_df = data.get("indications_df")
    if indications_df is not None and not indications_df.empty:
        # Calculate summary statistics
        max_clinical_phase = (
            indications_df["Max_Phase"].max()
            if "Max_Phase" in indications_df.columns
            else 0
        )
        unique_indications = (
            indications_df["MESH_Heading"].nunique()
            if "MESH_Heading" in indications_df.columns
            else 0
        )
        compounds_with_indications = (
            indications_df["ChEMBL_ID"].nunique()
            if "ChEMBL_ID" in indications_df.columns
            else len(df)
        )

        # Get top indications
        top_indications = (
            indications_df.groupby("MESH_Heading")["Max_Phase"]
            .max()
            .sort_values(ascending=False)
            .head(10)
        )
        for indication, phase in top_indications.items():
            indications_html += f"<tr><td>{html.escape(str(indication))}</td><td>Phase {int(phase)}</td></tr>"

    # Get base score and QED multiplier from best compound
    base_score = (
        best_row["IMP_Base_Score"]
        if best_row is not None and "IMP_Base_Score" in best_row.index
        else None
    )
    qed_mult = (
        best_row["QED_Multiplier"]
        if best_row is not None and "QED_Multiplier" in best_row.index
        else None
    )

    # Pre-compute formatted strings (can't use conditionals inside f-string format specifiers)
    base_score_str = (
        f"{base_score:.3f}"
        if base_score is not None and not pd.isna(base_score)
        else "N/A"
    )
    qed_mult_str = (
        f"{qed_mult:.3f}" if qed_mult is not None and not pd.isna(qed_mult) else "N/A"
    )
    qed_val_str = (
        f"{qed_val:.3f}" if qed_val is not None and not pd.isna(qed_val) else "N/A"
    )
    final_score_str = (
        f"{final_score:.3f}"
        if final_score is not None and not pd.isna(final_score)
        else "N/A"
    )

    # Component score values for formula display
    def _fmt_cs(col):
        v = best_row[col] if best_row is not None and col in best_row.index else None
        return f"{v:.3f}" if v is not None and not pd.isna(v) else "N/A"

    eff_s_str = _fmt_cs("Efficiency_Score")
    dist_s_str = _fmt_cs("Distance_Score")
    ang_s_str = _fmt_cs("Angle_Score")
    int_s_str = _fmt_cs("Interference_Score")
    pdb_s_str = _fmt_cs("PDB_Score")

    # Escape compound name for HTML
    safe_compound_name = html.escape(compound_name)
    smiles_display = html.escape(smiles[:60] + "..." if len(smiles) > 60 else smiles)

    # Generate chart images for HTML export
    bioactivity_chart_html = _create_html_bioactivity_donut(df)
    efficiency_boxplots_html = _create_html_efficiency_boxplots(df)
    efficiency_scatter_html = _create_html_efficiency_scatter(df)
    pdb_quality_chart_html = _create_html_pdb_quality_bar(df, data)

    # Build comprehensive HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>IMPULATOR Report - {safe_compound_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 950px; margin: 0 auto; padding: 20px; color: #333; }}
        h1 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        h2 {{ color: #444; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        .header {{ display: flex; gap: 30px; align-items: flex-start; margin-bottom: 20px; }}
        .header-info {{ flex: 1; }}
        .verdict {{ background-color: #f9fafb; border-left: 4px solid #6b7280; padding: 15px; margin: 20px 0; border-radius: 5px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
        .verdict h3 {{ color: #111827; margin: 0 0 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #f5f5f5; font-weight: bold; }}
        .warning {{ background-color: #f8d7da; border: 1px solid #dc3545; padding: 12px; margin: 10px 0; border-radius: 5px; }}
        .info {{ background-color: #e3f2fd; border-left: 4px solid #1976d2; padding: 15px; margin: 10px 0; border-radius: 8px; color: #0d47a1; }}
        .success {{ background-color: #d4edda; border: 1px solid #28a745; padding: 12px; margin: 10px 0; border-radius: 5px; }}
        .section {{ margin-bottom: 30px; }}
        .calc-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; }}
        .guide {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .two-col {{ display: flex; gap: 30px; }}
        .two-col > div {{ flex: 1; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        @media print {{
            body {{ max-width: 100%; }}
            .no-print {{ display: none; }}
            h2 {{ page-break-before: auto; }}
        }}
    </style>
</head>
<body>
    <h1>IMPULATOR Compound Analysis Report</h1>

    <!-- 1. HEADER -->
    <div class="header">
        <div>{structure_img_html}</div>
        <div class="header-info">
            <h2 style="margin-top: 0; border: none;">{safe_compound_name}</h2>
            <p><strong>InChIKey:</strong> <code>{inchikey}</code></p>
            <p><strong>InChI:</strong> <code style="word-break: break-all; font-size: 0.85em;">{
        inchi
    }</code></p>
            <p><strong>SMILES:</strong> <code>{smiles_display}</code></p>
            <p><strong>Analysis Date:</strong> {
        summary.get("processing_date", "N/A")
    }</p>
            <p><strong>Author:</strong> {
        html.escape(data.get("author_name", "N/A"))
    }</p>
            <p><strong>Report Generated:</strong> {
        datetime.now().strftime("%Y-%m-%d %H:%M")
    }</p>
        </div>
    </div>

    <!-- Summary Stats Row -->
    <div style="display: flex; justify-content: space-around; background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">Similar Compounds</div>
            <div style="font-size: 1.8em; font-weight: bold;">{
        summary.get("total_similar", similar_count)
    }</div>
            <div style="font-size: 0.7em; color: #888;">{
        summary.get("compounds_with_data", similar_count)
    } with activity</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">Activities</div>
            <div style="font-size: 1.8em; font-weight: bold;">{activities_count}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">QED</div>
            <div style="font-size: 1.8em; font-weight: bold;">{avg_qed_str}</div>
        </div>
        <div style="text-align: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;">
            <div style="font-size: 0.9em; color: #666;">IMP Score</div>
            <div style="font-size: 1.8em; font-weight: 700; color: #111827; line-height: 1;">{
        avg_imp_score_str
    }</div>
            <div style="margin-top: 6px;">{header_global_bar_svg}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">IMP Candidates</div>
            <div style="font-size: 1.8em; font-weight: bold; color: #111827;">{
        imp_count if has_warning else 0
    }</div>
            <div style="font-size: 0.65em; color: #6b7280;">unique compounds (outlier detection)</div>
        </div>
    </div>

    <!-- 2. EXECUTIVE SUMMARY -->
    <h2>📊 Executive Summary</h2>
    <div class="verdict">
        <div style="font-size: 2em; font-weight: 700; color: #111827; line-height: 1;">{
        final_score_int if final_score_int is not None else "N/A"
    }</div>
        <div style="font-size: 14px; color: #6b7280; margin-top: 4px;">Mean IMP score across this query (0–100 global scale)</div>
        {global_bar_svg if final_score_int is not None else ""}
        <p style="margin-top: 10px;"><strong>QED:</strong> {
        qed_val:.3f} | <strong>Red Flags:</strong> {total_flags} active</p>
    </div>

    <!-- 3. COMPOUND PROPERTIES -->
    <h2>🧪 Compound Properties</h2>
    <table>
        <tr><th>Property</th><th>Value</th><th>Unit/Description</th></tr>
        {
        props_html
        if props_html
        else "<tr><td colspan='3'>No property data available</td></tr>"
    }
    </table>

    <!-- 4. EFFICIENCY METRICS -->
    <h2>📈 Efficiency Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Formula</th><th>Used in Score</th></tr>
        {
        efficiency_html
        if efficiency_html
        else "<tr><td colspan='4'>No efficiency data available</td></tr>"
    }
    </table>
    <p><em>Only SEI and BEI contribute to the Efficiency Score. NSEI/NBEI are for reference.</em></p>

    <!-- 5. IMP SCORE CALCULATION -->
    <h2>🔢 IMP Score Calculation</h2>
    <h3>Component Scores</h3>
    <table>
        <tr><th>Component</th><th>Score</th><th>Weight</th><th>Contribution</th></tr>
        {
        components_html
        if components_html
        else "<tr><td colspan='5'>No component data available</td></tr>"
    }
    </table>

    <h3>Final Calculation</h3>
    <div class="calc-box">
<strong>Base Score</strong> = 0.45×Eff + 0.20×Dist + 0.15×Angle + 0.15×Interf + 0.05×PDB
         = 0.45×{eff_s_str} + 0.20×{dist_s_str} + 0.15×{ang_s_str} + 0.15×{
        int_s_str
    } + 0.05×{pdb_s_str}
         = {base_score_str}

<strong>QED Value:</strong> {qed_val_str}
<strong>QED Multiplier</strong> = 0.75 + 0.25 × QED
             = 0.75 + 0.25 × {qed_val_str}
             = {qed_mult_str}

<strong>FINAL SCORE</strong> = Base Score × QED Multiplier
            = {base_score_str} × {qed_mult_str}
            = <strong>{final_score_str}</strong>
    </div>

    <!-- 6. RED FLAGS -->
    <h2>⚠️ Red Flags Assessment</h2>
    <div style="background-color: {
        "#d4edda"
        if total_flags == 0
        else ("#fff3cd" if total_flags <= 5 else "#f8d7da")
    }; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {
        flag_color
    };">
        <strong style="color: {flag_color};">{flag_assessment} - {
        total_flags
    } flags detected</strong>
    </div>
    <table>
        <tr><th>Flag Type</th><th>Status</th><th>Description</th></tr>
        {
        red_flags_html
        if red_flags_html
        else "<tr><td colspan='3'>No flag data available</td></tr>"
    }
    </table>

    <!-- 7. BIOACTIVITY DISTRIBUTION -->
    <h2>🎯 Bioactivity Distribution</h2>
    <div class="two-col">
        <div>{bioactivity_chart_html}</div>
        <div>
            <table>
                <tr><th>Activity Type</th><th>Count</th><th>Percentage</th></tr>
                {
        bioactivity_html
        if bioactivity_html
        else "<tr><td colspan='3'>No bioactivity data available</td></tr>"
    }
            </table>
            <p><strong>Total Activities:</strong> {len(df)}</p>
        </div>
    </div>

    <!-- 8. EFFICIENCY METRICS -->
    <h2>📈 Efficiency Metrics Distribution</h2>

    <!-- Metric Cards -->
    <div style="display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap;">
        {
        "".join(
            [
                f'''
        <div style="flex: 1; min-width: 200px; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid {metric_colors_html[metric]};">
            <div style="color: #fff; font-size: 0.95em; margin-bottom: 5px;">{metric}{"" if not stats["used"] else " ✓"}</div>
            <div style="font-size: 1.5em; color: {metric_colors_html[metric]}; font-weight: bold;">{stats["mean"]:.2f}</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Mean Value</div>
            <div style="color: #888; font-size: 0.75em; margin-top: 3px;">Range: {stats["min"]:.1f}-{stats["max"]:.1f}</div>
        </div>
        '''
                for metric, stats in efficiency_metrics_stats.items()
            ]
        )
    }
    </div>

    <!-- Description -->
    <div class="info" style="margin-bottom: 15px;">
        <strong>📊 Efficiency Metrics Explained:</strong><br>
        • <strong>SEI (Surface Efficiency Index)</strong> = pActivity × 100 / PSA — Potency per unit polar surface area (✓ used in IMP Score)<br>
        • <strong>BEI (Binding Efficiency Index)</strong> = pActivity × 1000 / MW — Potency per unit molecular weight (✓ used in IMP Score)<br>
        • <strong>NSEI</strong> = pActivity / (N+O atoms) — Potency per heteroatom count (informational only)<br>
        • <strong>NBEI</strong> = pActivity / Heavy atoms — Potency per heavy atom count (informational only)
    </div>

    {efficiency_boxplots_html}
    <p><em>Box plots show the distribution of all efficiency metrics across all bioactivities. Only SEI and BEI contribute to the IMP Efficiency Score.</em></p>

    <!-- 9. EFFICIENCY PLANE -->
    <h2>📐 Efficiency Plane (SEI vs BEI)</h2>

    <!-- Angle Status Banner -->
    <div style="background-color: {
        angle_status_bg
    }; padding: 12px; border-radius: 5px; margin-bottom: 15px; border: 1px solid {
        angle_status_color
    };">
        <strong style="color: {angle_status_color};">Development Trajectory: {
        angle_status
    } (Angle: {angle_str}°)</strong>
    </div>

    <!-- Metric Cards Row -->
    <div style="display: flex; gap: 15px; margin-bottom: 15px;">
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #636EFA;">
            <div style="font-size: 1.4em; color: #636EFA; font-weight: bold;">{
        angle_str
    }°</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Mean Angle</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Development trajectory</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #00CC96;">
            <div style="font-size: 1.4em; color: #00CC96; font-weight: bold;">{
        modulus_str
    }</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Modulus</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Overall efficiency</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #EF553B;">
            <div style="font-size: 1.4em; color: #EF553B; font-weight: bold;">{
        mean_sei_str
    }</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Mean SEI</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Surface efficiency</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #FFA15A;">
            <div style="font-size: 1.4em; color: #FFA15A; font-weight: bold;">{
        mean_bei_str
    }</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Mean BEI</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Binding efficiency</div>
        </div>
    </div>

    <!-- Chart and Interpretation -->
    <div class="two-col">
        <div>{efficiency_scatter_html}</div>
        <div>
            <h3>Angle Interpretation</h3>
            <ul style="line-height: 1.8;">
                <li><strong>&lt; 45°:</strong> Favors size efficiency (SEI) - Compound optimized more for surface area efficiency</li>
                <li><strong>≈ 45°:</strong> Balanced development (OPTIMAL) - Ideal balance between size and binding efficiency</li>
                <li><strong>&gt; 45°:</strong> Favors binding efficiency (BEI) - Compound optimized more for binding potency</li>
            </ul>
            <div class="info" style="margin-top: 15px;">
                <strong>📊 What This Means:</strong><br>
                The 45° optimal angle represents balanced development where compounds achieve efficiency improvements
                through both size reduction (SEI) and potency enhancement (BEI). Most drugs exhibit angles between 50-70°
                due to typical PSA/MW ratios.
            </div>
            <p style="margin-top: 10px;"><strong>Modulus Formula:</strong> <code>sqrt(SEI² + BEI²)</code></p>
            <p><strong>Angle Formula:</strong> <code>arctan2(BEI, SEI)</code> in degrees</p>
        </div>
    </div>

    <h3>10PSA/MW — Compound Polarity Fingerprint</h3>
    <p>The ratio 10PSA/MW is the fingerprint of the compound that corresponds to the profile of the binding pocket.</p>
    <ul>
        <li><strong>Polar pockets:</strong> The most fitting compounds will have <strong>high PSA/MW</strong> — indicating a larger proportion of polar surface area relative to molecular weight.</li>
        <li><strong>Hydrophobic pockets:</strong> The most fitting compounds will have <strong>low PSA/MW</strong> — indicating a compact, non-polar molecular profile.</li>
    </ul>

    <!-- 10. PDB EVIDENCE -->
    <h2>🔬 PDB Structural Evidence</h2>

    <!-- Confidence Banner -->
    <div style="background-color: {
        pdb_conf_bg
    }; padding: 12px; border-radius: 5px; margin-bottom: 15px; border: 1px solid {
        pdb_conf_color
    };">
        <strong style="color: {pdb_conf_color};">{
        pdb_conf_icon
    } Structural Validation: {pdb_confidence} (PDB Score: {mean_pdb_score:.3f})</strong>
    </div>

    <!-- Quality Distribution Cards -->
    <div style="display: flex; gap: 15px; margin-bottom: 15px;">
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #636EFA;">
            <div style="font-size: 1.5em; color: #636EFA; font-weight: bold;">{
        pdb_total
    }</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Total Structures</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">PDB entries found</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #28a745;">
            <div style="font-size: 1.5em; color: #28a745; font-weight: bold;">{
        high_q
    }</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">High Quality</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">&lt;2.0Å ({
        high_q_pct:.0f}%) ★★★★★</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #fd7e14;">
            <div style="font-size: 1.5em; color: #fd7e14; font-weight: bold;">{
        med_q
    }</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Medium Quality</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">2-3Å ({
        med_q_pct:.0f}%) ★★★</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #dc3545;">
            <div style="font-size: 1.5em; color: #dc3545; font-weight: bold;">{
        poor_q
    }</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Poor Quality</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">&gt;3Å ({
        poor_q_pct:.0f}%) ★</div>
        </div>
    </div>

    <!-- Quality Distribution Chart (Full Width) -->
    <div style="margin-bottom: 20px;">
        {pdb_quality_chart_html}
    </div>

    {high_q_pdb_links_html}
    {med_q_pdb_links_html}

    <!-- Resolution Quality Info -->
    <div class="info">
        <strong>📊 Resolution Quality:</strong><br>
        High-resolution structures (&lt;2.0Å) provide the most reliable structural validation.
        PDB Score component contributes 5% to final IMP score.
        <ul style="margin-top: 10px; margin-bottom: 5px;">
            <li><strong>⭐⭐⭐⭐⭐ High (&lt;2.0Å):</strong> Excellent resolution - high confidence in binding mode</li>
            <li><strong>⭐⭐⭐ Medium (2-3Å):</strong> Good resolution - reliable structural information</li>
            <li><strong>⭐ Poor (&gt;3Å):</strong> Lower resolution - general binding information only</li>
        </ul>
    </div>

    <!-- 11. CLASSIFICATION -->
    <h2>🏷️ Chemical Classification</h2>

    <div class="two-col">
        <!-- ClassyFire Taxonomy -->
        <div>
            <h3>ClassyFire Taxonomy:</h3>
            <table>
                <tr><th>Level</th><th>Classification</th></tr>
                {
        classyfire_html
        if classyfire_html
        else "<tr><td colspan='2'>No ClassyFire data available</td></tr>"
    }
            </table>
        </div>

        <!-- NPClassifier (Natural Products) -->
        <div>
            <h3>NPClassifier (Natural Products):</h3>
            <table>
                <tr><th>Level</th><th>Classification</th></tr>
                {
        npclassifier_html
        if npclassifier_html
        else "<tr><td colspan='2'>No NPClassifier data available</td></tr>"
    }
            </table>
        </div>
    </div>

    <!-- 12. DRUG INDICATIONS -->
    <h2>💊 Drug Indications</h2>

    {
        f'''
    <div style="display: flex; gap: 20px; margin-bottom: 15px;">
        <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea;">
            <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Maximum Clinical Phase</div>
            <div style="font-size: 1.8em; color: #667eea; font-weight: bold;">{max_clinical_phase:.1f}</div>
        </div>
        <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
            <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Unique Indications</div>
            <div style="font-size: 1.8em; color: #28a745; font-weight: bold;">{unique_indications}</div>
        </div>
        <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;">
            <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Compounds with Data</div>
            <div style="font-size: 1.8em; color: #17a2b8; font-weight: bold;">{compounds_with_indications}</div>
        </div>
    </div>

    <h3>Top Indications by Phase:</h3>
    <table>
        <tr><th>Indication</th><th>Max Phase</th></tr>
        {indications_html}
    </table>
    '''
        if indications_html
        else "<p>No drug indication data available</p>"
    }

    <!-- 13. FINAL RECOMMENDATION -->
    <h2>🎯 Final Recommendation</h2>
    <div class="verdict">
        <div style="font-size: 48px; font-weight: 700; color: #111827; line-height: 1;">{
        final_score_int if final_score_int is not None else "N/A"
    }</div>
        <div style="font-size: 14px; font-weight: 400; color: #6b7280; margin-top: 8px;">Global reference (10–80)</div>
        {global_bar_svg if final_score_int is not None else ""}
        <div style="display:flex; justify-content:space-between; max-width:240px; font-size:14px; color:#6b7280; margin-top:4px;"><span>{
        IMP_SCORE_FLOOR
    }</span><span>{IMP_SCORE_CEILING}</span></div>
        <div style="font-size:14px; font-weight:400; color:#6b7280; margin-top:16px;">This query's range</div>
        {dynamic_bar_svg if final_score_int is not None else ""}
        <div style="display:flex; justify-content:space-between; max-width:240px; font-size:14px; color:#6b7280; margin-top:4px;"><span>{
        obs_min_str
    }</span><span>{obs_max_str}</span></div>
    </div>

    <!-- IMP GUIDE -->
    <h2>📚 IMP Interpretation Guide</h2>
    <div class="guide">
        <p style="color:#6b7280; font-size:14px;">The IMP score is a continuous 0–100 measure; the team has chosen not to define qualitative thresholds.</p>
    </div>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #667eea; color: #666; font-size: 12px; text-align: center;">
        <p>Generated by <strong>IMPULATOR</strong> | {
        datetime.now().strftime("%Y-%m-%d %H:%M")
    }</p>
        <p>💡 <em>Tip: Use Ctrl+P (Cmd+P on Mac) to print this report to PDF</em></p>
    </footer>
</body>
</html>
"""
    return html_content


# =============================================================================
# VERSIONS TAB - Structural siblings with config diff highlighting
# =============================================================================


def _render_versions_tab(versions: list, current_entry_id: str) -> None:
    """Render the Versions tab showing all structural siblings.

    Args:
        versions: List of version dicts from the API
        current_entry_id: Entry ID of the compound being viewed
    """
    if not versions:
        st.info("No other versions found.")
        return

    current = None
    siblings = []
    for v in versions:
        if v.get("is_current"):
            current = v
        else:
            siblings.append(v)

    # Fallback: if is_current flag missing, match by entry_id
    if current is None and current_entry_id:
        for v in versions:
            if v.get("entry_id") == current_entry_id:
                current = v
                siblings = [
                    s for s in versions if s.get("entry_id") != current_entry_id
                ]
                break

    if not current or not siblings:
        st.info("No other versions found.")
        return

    st.markdown(
        f"**{len(siblings)}** other version{'s' if len(siblings) != 1 else ''} "
        f"of this structure (same InChIKey)"
    )

    # Summary table — compare each version's config against current
    current_threshold = current.get("similarity_threshold")
    current_act_set = set(current.get("activity_types") or [])
    rows = []
    for v in versions:
        marker = ""
        if v.get("is_current"):
            marker = " ◀"
        if v.get("is_original"):
            marker += " ★"
        qed = v.get("qed")

        # Compare activity types with current compound
        v_act = set(v.get("activity_types") or [])
        v_threshold = v.get("similarity_threshold")
        same_config = v_act == current_act_set and v_threshold == current_threshold

        if v.get("is_current"):
            act_display = ", ".join(sorted(v_act)) if v_act else "all (default)"
        elif same_config:
            act_display = "✓ Same Config"
        else:
            act_display = ", ".join(sorted(v_act)) if v_act else "all (default)"

        rows.append(
            {
                "Name": html.escape(v.get("compound_name", "")) + marker,
                "Threshold": f"{v.get('similarity_threshold', '?')}%",
                "Activity Types": act_display,
                "Similar Compounds": v.get("similar_compounds") or 0,
                "QED": f"{qed:.2f}" if qed is not None else "—",
                "Activities": v.get("total_activities") or 0,
                "Author": html.escape(v.get("author_name") or "—"),
                "Processed": (v.get("processed_at") or "")[:10],
            }
        )

    table_df = pd.DataFrame(rows)

    # Highlight current row with green background
    def _highlight_current(row):
        if "◀" in str(row["Name"]):
            return ["background-color: rgba(0, 180, 100, 0.15)"] * len(row)
        return [""] * len(row)

    styled = table_df.style.apply(_highlight_current, axis=1)
    st.markdown("##### All Versions")
    st.markdown("★ = original &nbsp;&nbsp; ◀ = currently viewing")
    st.dataframe(styled, width="stretch", hide_index=True)

    # Per-sibling expandable cards with config diffs
    st.markdown("##### Version Details")
    current_threshold = current.get("similarity_threshold")
    _raw_act = current.get("activity_types") or []
    current_activities = (
        set(_raw_act)
        if isinstance(_raw_act, list)
        else {s.strip() for s in _raw_act.split(",")} - {""}
    )

    for idx, sib in enumerate(siblings):
        sib_name = html.escape(sib.get("compound_name", "Unknown"))
        sib_threshold = sib.get("similarity_threshold")
        _raw_sib_act = sib.get("activity_types") or []
        sib_activities = (
            set(_raw_sib_act)
            if isinstance(_raw_sib_act, list)
            else {s.strip() for s in _raw_sib_act.split(",")} - {""}
        )

        label = f"**{sib_name}**"
        if sib.get("is_original"):
            label += " ★ original"

        with st.expander(label, expanded=False):
            # Config diff
            diffs = []
            if sib_threshold != current_threshold:
                diffs.append(
                    f"🔶 **Threshold**: {sib_threshold}% "
                    f"(current: {current_threshold}%)"
                )

            if sib_activities != current_activities:
                added = sib_activities - current_activities
                removed = current_activities - sib_activities
                shared = sib_activities & current_activities
                parts = []
                for a in sorted(shared):
                    parts.append(a)
                for a in sorted(added):
                    parts.append(f"**+{a}**")
                for a in sorted(removed):
                    parts.append(f"~~{a}~~")
                diffs.append(f"**Activity Types**: {', '.join(parts)}")

            if diffs:
                st.markdown("**Config differences** (vs. current):")
                for d in diffs:
                    st.markdown(f"- {d}")
            else:
                st.markdown("*Same configuration as current compound*")

            # Stats row
            qed = sib.get("qed")
            cols = st.columns(3)
            with cols[0]:
                st.metric("QED", f"{qed:.2f}" if qed is not None else "—")
            with cols[1]:
                st.metric("Activities", sib.get("total_activities") or 0)
            with cols[2]:
                st.metric("Author", html.escape(sib.get("author_name") or "—"))

            if sib.get("parent_name"):
                st.caption(f"Duplicate of: {html.escape(sib['parent_name'])}")

            # Navigate button
            if st.button(
                f"View →  {sib_name}",
                key=f"versions_nav_{idx}",
                width="stretch",
            ):
                # Reset tab index to Overview on next render
                st.session_state["_versions_nav_reset"] = True
                # Clear versions cache for the target compound
                target_cache_key = f"_versions_{sib['entry_id']}"
                st.session_state.pop(target_cache_key, None)
                # Also clear current cache (stale after navigation)
                current_cache_key = f"_versions_{current_entry_id}"
                st.session_state.pop(current_cache_key, None)

                SessionState.navigate_to_compound(
                    compound_name=sib.get("compound_name", ""),
                    entry_id=sib.get("entry_id"),
                    storage_path=sib.get("storage_path"),
                    is_duplicate=sib.get("parent_id") is not None,
                    duplicate_of_name=sib.get("parent_name"),
                )
                st.rerun()
