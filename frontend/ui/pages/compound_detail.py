"""Compound detail page for IMPULATOR.

Displays full analysis results with improved UX and organization.
"""

import html
import logging
import re
from typing import Dict, Any, Optional
from urllib.parse import quote_plus

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from frontend.services import (
    get_api_client,
    delete_from_cache,
    smart_load_summary,
    smart_load_dataframe,
)
from frontend.utils import SessionState, sanitize_compound_name
from frontend.ui.components import render_2d_structure, embed_structure_viewer, render_structure_viewer_hint

logger = logging.getLogger(__name__)

# Import scipy for regression statistics
from scipy import stats as scipy_stats  # noqa: E402


def render_compound_detail_page() -> None:
    """Render the compound detail page."""
    compound_name = SessionState.get('selected_compound')
    entry_id = SessionState.get('selected_compound_entry_id')
    storage_path = SessionState.get('selected_compound_storage_path')

    if not compound_name:
        st.error("No compound selected")
        if st.button("Go to Home"):
            SessionState.navigate_to_home()
            st.rerun()
        return

    # Header
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("← Back", width='stretch'):
            SessionState.navigate_to_home()
            st.rerun()
    with col2:
        safe_compound_name = html.escape(compound_name)
        st.markdown(f"<h2 style='text-align: center; margin: 0;'>{safe_compound_name}</h2>", unsafe_allow_html=True)
    with col3:
        if st.button("🗑️", width='stretch', help="Delete compound"):
            SessionState.set('show_delete_confirmation', True)
            st.rerun()

    if SessionState.get('show_delete_confirmation'):
        _show_delete_confirmation(compound_name, entry_id)
        return

    # Load data using storage_path (most reliable), fallback to entry_id, then compound_name
    data = _load_compound_data(
        compound_name=compound_name,
        entry_id=entry_id,
        storage_path=storage_path
    )
    if data is None:
        st.error(f"Could not load data for '{compound_name}'")
        return

    # Quick stats row
    _render_quick_stats(data)

    # Main content with tabs
    tabs = st.tabs(["📊 Overview", "📈 Visualizations", "🧬 Molecules", "📋 Data", "📄 Report"])

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


def _render_quick_stats(data: Dict[str, Any]) -> None:
    """Render compact stats bar."""
    df = data.get('results')
    summary = data.get('summary', {})

    cols = st.columns(5)

    similar = summary.get('similar_count', 0)
    activities = summary.get('total_activities', len(df) if df is not None else 0)
    qed = summary.get('qed', 0)

    # Count unique IMP compounds (not activity rows)
    imp_count = 0
    has_warning = False
    if df is not None and 'Is_IMP_Candidate' in df.columns and 'ChEMBL_ID' in df.columns:
        imp_count = df[df['Is_IMP_Candidate']]['ChEMBL_ID'].nunique()
        has_warning = imp_count > 0
    elif summary.get('has_imp_candidates', False):
        imp_count = summary.get('imp_candidates', 0)
        has_warning = True

    with cols[0]:
        st.metric("Similar Compounds", similar)
    with cols[1]:
        st.metric("Activities", activities)
    with cols[2]:
        st.metric("QED", f"{qed:.2f}" if qed else "N/A")
    with cols[3]:
        imp_score = None
        if df is not None and 'IMP_Final_Score' in df.columns:
            imp_score = df['IMP_Final_Score'].max()
        st.metric("IMP Score", f"{imp_score:.2f}" if pd.notna(imp_score) else "N/A",
                  help="Best scoring compound (highest IMP risk)")
    with cols[4]:
        if has_warning:
            st.error(f"⚠️ {imp_count} IMP")
        else:
            st.success("✓ Clean")


# =============================================================================
# OVERVIEW TAB - Using sub-tabs for organization
# =============================================================================

def _render_overview_tab(data: Dict[str, Any]) -> None:
    """Overview with sub-tabs for different analysis sections."""
    df = data.get('results')
    summary = data.get('summary', {})
    compound_name = data.get('compound_name', '')
    entry_id = data.get('entry_id')
    storage_path = data.get('storage_path')

    # Sub-tabs for overview sections
    sub_tabs = st.tabs([
        "🧪 Compound",
        "🔢 Properties",
        "📈 Activity",
        "🎯 Efficiency",
        "🔬 PDB Evidence",
        "⚠️ Assay Interference",
        "🔍 IMP Score",
        "💊 Drug Indications"
    ])

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
        _render_pdb_evidence(compound_name, df, entry_id=entry_id, storage_path=storage_path)

    # PAINS/Assay Interference (separated)
    with sub_tabs[5]:
        _render_pains_analysis(df)

    # IMP Score Analysis (without PAINS)
    with sub_tabs[6]:
        _render_imp_score_analysis(df, compound_name)

    # Drug Indications
    with sub_tabs[7]:
        _render_drug_indications(data)


def _render_compound_info(data: Dict[str, Any], df: pd.DataFrame, summary: Dict) -> None:
    """Compound information section."""
    # Get unique IDs early for use in both columns
    unique_ids = []
    unique_count = 0
    if df is not None and 'ChEMBL_ID' in df.columns:
        unique_ids = [str(x) for x in df['ChEMBL_ID'].dropna().unique().tolist()]
        unique_count = len(unique_ids)

    col1, col2 = st.columns([1, 2])

    with col1:
        smiles = data.get('smiles', '')
        if smiles:
            render_2d_structure(smiles, size=(380, 300))
        st.caption(f"Similarity: {summary.get('similarity_threshold', 90)}%")

        # Processed date below structure
        if summary.get('processing_date'):
            st.markdown(f"**Processed:** {summary['processing_date']}")

        # Author name
        author_name = data.get('author_name', 'N/A')
        if author_name and author_name != 'N/A':
            st.markdown(f"**Author:** {html.escape(author_name)}")

        # Similar Compounds info below Processed (show max 3 IDs to prevent overflow)
        if unique_count > 0:
            ids_display = ", ".join(unique_ids[:3]) if unique_ids else "None"
            if unique_count > 3:
                ids_display += f" (+{unique_count - 3})"
            st.markdown(f"**Similar Compounds ({unique_count}):** `{ids_display}`")

    with col2:
        # Key info in a clean grid
        info_cols = st.columns(2)

        with info_cols[0]:
            st.markdown("**Query SMILES**")
            # Show full SMILES (scrollable code block) - don't truncate for copy-ability
            smiles_value = data.get('smiles', '')
            st.code(smiles_value if smiles_value else 'N/A', language=None)

            # Display pre-computed InChI and InChIKey
            inchikey = data.get('inchikey')
            inchi = data.get('inchi')
            if inchikey:
                st.markdown("**InChIKey**")
                st.code(inchikey, language=None)
            if inchi:
                st.markdown("**InChI**")
                st.code(inchi, language=None)

        with info_cols[1]:
            activity_types = summary.get('activity_types', '')
            if activity_types:
                st.markdown("#### Activity Types")
                # Show as larger pills/tags
                types = activity_types.split(',') if isinstance(activity_types, str) else activity_types
                st.markdown(" ".join([f"**`{t.strip()}`**" for t in types[:7]]))

            # Chemical Formula from SMILES (larger display)
            smiles_for_formula = data.get('smiles', '')
            if smiles_for_formula:
                try:
                    from rdkit import Chem
                    from rdkit.Chem import rdMolDescriptors
                    mol = Chem.MolFromSmiles(smiles_for_formula)
                    if mol:
                        formula = rdMolDescriptors.CalcMolFormula(mol)
                        if formula:
                            st.markdown("#### Chemical Formula")
                            st.markdown(f"### `{formula}`")
                except Exception:
                    pass  # Skip if RDKit not available

        # View Compound Details - full width of col2 (spans across both info_cols)
        if unique_count > 0 and df is not None:
            with st.expander("📋 View Compound Details", expanded=False):
                id_cols = ['ChEMBL_ID']
                if 'Molecule_Name' in df.columns:
                    id_cols.append('Molecule_Name')
                if 'Similarity' in df.columns:
                    id_cols.append('Similarity')

                unique_compounds = df[id_cols].drop_duplicates('ChEMBL_ID').reset_index(drop=True)

                if 'Molecule_Name' in unique_compounds.columns:
                    unique_compounds['Molecule_Name'] = unique_compounds['Molecule_Name'].apply(
                        lambda x: x if isinstance(x, str) else ''
                    )

                st.dataframe(
                    unique_compounds,
                    width='stretch',
                    hide_index=True,
                    height=min(200, len(unique_compounds) * 35 + 40)
                )


def _render_classification_compact(df: pd.DataFrame) -> None:
    """Compact classification display - ClassyFire + NPClassifier side by side."""
    if df is None:
        return

    # ClassyFire columns
    classyfire_cols = ['Kingdom', 'Superclass', 'Class', 'Subclass']
    # NPClassifier columns
    npclass_cols = ['NP_Pathway', 'NP_Superclass', 'NP_Class']

    cf_avail = [c for c in classyfire_cols if c in df.columns]
    np_avail = [c for c in npclass_cols if c in df.columns]

    if not cf_avail and not np_avail:
        return

    st.markdown("**Chemical Classification**")

    # Side by side: ClassyFire | NPClassifier
    col1, col2 = st.columns(2)

    with col1:
        if cf_avail:
            st.markdown("🧬 **ClassyFire**")
            # Get unique values for each level (most common)
            for col in cf_avail:
                val_counts = df[col].value_counts()
                if len(val_counts) > 0:
                    top_val = val_counts.index[0]
                    count = val_counts.iloc[0]
                    unique = df[col].nunique()
                    if unique > 1:
                        st.caption(f"**{col}**: {top_val} ({count}, +{unique-1} more)")
                    else:
                        st.caption(f"**{col}**: {top_val}")
        else:
            st.caption("ClassyFire: Not available")

    with col2:
        if np_avail:
            st.markdown("🌿 **NPClassifier**")
            for col in np_avail:
                val_counts = df[col].value_counts()
                if len(val_counts) > 0:
                    top_val = val_counts.index[0]
                    count = val_counts.iloc[0]
                    unique = df[col].nunique()
                    label = col.replace('NP_', '')
                    if unique > 1:
                        st.caption(f"**{label}**: {top_val} ({count}, +{unique-1} more)")
                    else:
                        st.caption(f"**{label}**: {top_val}")
        else:
            st.caption("NPClassifier: Not available")

    # Expandable full table
    with st.expander("📋 View Full Classification Table"):
        id_cols = ['ChEMBL_ID'] if 'ChEMBL_ID' in df.columns else []
        display_cols = id_cols + cf_avail + np_avail
        display_cols = [c for c in display_cols if c in df.columns]

        if display_cols:
            class_df = df[display_cols].drop_duplicates()
            st.dataframe(class_df, width='stretch', hide_index=True, height=250)


def _render_computed_properties(df: pd.DataFrame) -> None:
    """Computed molecular properties display."""
    if df is None:
        st.info("No data available")
        return

    st.markdown("**Computed Molecular Properties**")

    # Get unique compounds for property display
    unique_df = df.drop_duplicates('ChEMBL_ID') if 'ChEMBL_ID' in df.columns else df
    num_compounds = len(unique_df)
    st.caption(f"Properties for {num_compounds} similar compound{'s' if num_compounds != 1 else ''}")

    # Identify all numeric property columns (exclude metadata columns)
    exclude_cols = {
        'ChEMBL_ID', 'Molecule_Name', 'SMILES', 'Canonical_SMILES', 'Standard_SMILES',
        'InChI', 'InChI_Key', 'Target', 'Assay_ID', 'Assay_Description',
        'Activity_Type', 'Activity_Value', 'Activity_Units', 'Activity_Relation',
        'Document_ID', 'Document_Year', 'Activity_Comment', 'Pchembl_Value',
        'Kingdom', 'Superclass', 'Class', 'Subclass', 'Parent_Level',
        'NP_Pathway', 'NP_Superclass', 'NP_Class', 'Index', '_row_index',
        'Query_SMILES', 'Similarity', 'IMP_Candidate', 'IMP_Reason',
        'PAINS_Alert', 'Aggregator_Alert', 'Redox_Alert', 'Fluorescent_Alert',
        'IMP_Final_Score', 'IMP_Grade', 'O_Score', 'Q_Score', 'P_Score', 'L_Score', 'A_Score'
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
    physchem_hints = ['MW', 'Weight', 'LogP', 'ALogP', 'TPSA', 'PSA', 'HBD', 'HBA', 'Rotatable',
                      'Donors', 'Acceptors', 'CSP3', 'Rings', 'Aromatic', 'Heavy', 'Hetero',
                      'NumAtoms', 'NumBonds', 'MolLogP', 'MolMR', 'NPOL']
    druglike_hints = ['QED', 'Lipinski', 'Ro5', 'RO5', 'Veber', 'Ghose', 'Muegge', 'Egan', 'Brenk', 'NP_Likeness']

    # Categorize properties
    physchem_cols = [c for c in numeric_cols if any(h.lower() in c.lower() for h in physchem_hints)]
    druglike_cols = [c for c in numeric_cols if any(h.lower() in c.lower() for h in druglike_hints)]
    other_cols = [c for c in numeric_cols if c not in physchem_cols and c not in druglike_cols]

    # View mode toggle
    view_mode = st.radio(
        "View mode",
        ["Summary Statistics", "Individual Compounds"],
        horizontal=True,
        key="prop_view_mode"
    )

    if view_mode == "Summary Statistics":
        # Summary statistics view
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**📊 Physicochemical Properties**")
            if physchem_cols:
                # Build a clean property table like PubChem
                prop_data = []
                # Define property display names and order
                property_display = {
                    'Molecular_Weight': 'Molecular Weight',
                    'MolLogP': 'XLogP3',
                    'LogP': 'LogP',
                    'HBD': 'H-Bond Donors',
                    'HBA': 'H-Bond Acceptors',
                    'TPSA': 'Topological PSA',
                    'Heavy_Atoms': 'Heavy Atom Count',
                    'Rotatable_Bonds': 'Rotatable Bonds',
                    'Aromatic_Rings': 'Aromatic Rings',
                    'NPOL': 'Polar Atoms (N+O)',
                    'RO5_Violations': 'RO5 Violations',
                    'NP_Likeness_Score': 'NP Likeness Score',
                }

                for col in physchem_cols:
                    vals = unique_df[col].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        std_val = vals.std() if len(vals) > 1 else 0
                        min_val = vals.min()
                        max_val = vals.max()
                        display_name = property_display.get(col, col)
                        prop_data.append({
                            'Property': display_name,
                            'Mean': round(mean_val, 2),
                            'Min': round(min_val, 2),
                            'Max': round(max_val, 2),
                            'Std Dev': round(std_val, 2) if std_val > 0 else 0
                        })
                if prop_data:
                    st.dataframe(pd.DataFrame(prop_data), width='stretch', hide_index=True, height=min(300, len(prop_data) * 35 + 40))
            else:
                st.caption("No physicochemical properties found")

        with col2:
            st.markdown("**💊 Drug-likeness**")
            metrics_shown = False

            # Two columns side by side, each with vertical stacks
            qed_col, other_col = st.columns(2)

            # QED metrics stacked vertically
            with qed_col:
                # QED Score (0-1 range, higher is better)
                if 'QED' in unique_df.columns:
                    vals = unique_df['QED'].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        color = "🟢" if mean_val >= 0.5 else "🟡" if mean_val >= 0.3 else "🔴"
                        st.metric(f"{color} QED Score", f"{mean_val:.3f}",
                                  help="Quantitative Estimate of Drug-likeness (0-1, higher is better)")
                        metrics_shown = True

                # QED Multiplier (from IMP scoring: 0.75 + 0.25*QED)
                if 'QED_Multiplier' in unique_df.columns:
                    vals = unique_df['QED_Multiplier'].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        color = "🟢" if mean_val >= 0.75 else "🟡" if mean_val >= 0.65 else "🔴"
                        st.metric(f"{color} QED Multiplier", f"{mean_val:.3f}",
                                  help="IMP Score QED multiplier (0.75 + 0.25×QED)")
                        metrics_shown = True

                # QED Impact (from IMP scoring)
                if 'QED_Impact' in unique_df.columns:
                    vals = unique_df['QED_Impact'].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        color = "🟢" if mean_val >= -0.1 else "🟡" if mean_val >= -0.2 else "🔴"
                        st.metric(f"{color} QED Impact", f"{mean_val:.3f}",
                                  help="QED penalty on IMP score (0=best)")
                        metrics_shown = True

            # Other metrics stacked vertically beside QED
            with other_col:
                # RO5 Violations (0-4, lower is better)
                if 'RO5_Violations' in unique_df.columns:
                    vals = unique_df['RO5_Violations'].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        color = "🟢" if mean_val <= 1 else "🟡" if mean_val <= 2 else "🔴"
                        st.metric(f"{color} RO5 Violations", f"{mean_val:.1f}",
                                  help="Lipinski Rule of 5 violations (0-4)")
                        metrics_shown = True

                # Aromatic Rings
                if 'Aromatic_Rings' in unique_df.columns:
                    vals = unique_df['Aromatic_Rings'].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        color = "🟢" if mean_val <= 3 else "🟡" if mean_val <= 4 else "🔴"
                        st.metric(f"{color} Aromatic Rings", f"{mean_val:.1f}",
                                  help="Number of aromatic ring systems (≤3 preferred)")
                        metrics_shown = True

                # NP Likeness Score (-5 to +5, positive = more natural product-like)
                if 'NP_Likeness_Score' in unique_df.columns:
                    vals = unique_df['NP_Likeness_Score'].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        color = "🟢" if mean_val > 0 else "🟡" if mean_val > -1 else "⚪"
                        st.metric(f"{color} NP Likeness", f"{mean_val:.2f}",
                                  help="Natural Product Likeness (-5 to +5)")
                        metrics_shown = True
                    else:
                        st.metric("⚪ NP Likeness", "N/A",
                                  help="Reprocess to calculate NP Score")

            if not metrics_shown:
                st.caption("No drug-likeness properties found")

        # Key visualizations
        st.markdown("---")
        st.markdown("**📈 Key Property Visualizations**")

        # Determine which LogP column to use
        logp_col = None
        if 'LogP' in unique_df.columns and unique_df['LogP'].notna().any():
            logp_col = 'LogP'
        elif 'MolLogP' in unique_df.columns and unique_df['MolLogP'].notna().any():
            logp_col = 'MolLogP'

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # MW vs LogP scatter plot (Lipinski space)
            if 'Molecular_Weight' in unique_df.columns and logp_col:
                plot_data = unique_df[['Molecular_Weight', logp_col]].dropna()
                if len(plot_data) > 0:
                    hover_cols = [c for c in ['ChEMBL_ID', 'Molecule_Name'] if c in unique_df.columns]
                    fig = px.scatter(
                        unique_df.dropna(subset=['Molecular_Weight', logp_col]),
                        x='Molecular_Weight',
                        y=logp_col,
                        color='QED' if 'QED' in unique_df.columns and unique_df['QED'].notna().any() else None,
                        hover_data=hover_cols if hover_cols else None,
                        title='MW vs LogP',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(title=dict(text='MW vs LogP', subtitle=dict(text='Lipinski Rule of 5 space — dashed lines = boundaries')))
                    # Add Lipinski rule boundaries
                    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="LogP ≤ 5")
                    fig.add_vline(x=500, line_dash="dash", line_color="red", annotation_text="MW ≤ 500")
                    fig.update_layout(height=300, margin=dict(t=55, b=30, l=30, r=30))
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.caption("No MW/LogP data available for visualization")
            else:
                st.caption("MW vs LogP plot requires LogP data (reprocess compounds to generate)")

        with viz_col2:
            # TPSA vs HBD+HBA scatter plot
            has_hbd = 'HBD' in unique_df.columns and unique_df['HBD'].notna().any()
            has_hba = 'HBA' in unique_df.columns and unique_df['HBA'].notna().any()
            has_tpsa = 'TPSA' in unique_df.columns and unique_df['TPSA'].notna().any()

            if has_tpsa and has_hbd and has_hba:
                plot_df = unique_df.copy()
                plot_df['HBD+HBA'] = plot_df['HBD'].fillna(0) + plot_df['HBA'].fillna(0)
                fig = px.scatter(
                    plot_df.dropna(subset=['TPSA']),
                    x='TPSA',
                    y='HBD+HBA',
                    color='QED' if 'QED' in unique_df.columns and unique_df['QED'].notna().any() else None,
                    hover_data=['ChEMBL_ID'] if 'ChEMBL_ID' in unique_df.columns else None,
                    title='TPSA vs H-Bond Donors+Acceptors',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(title=dict(text='TPSA vs H-Bond Donors+Acceptors', subtitle=dict(text='Dashed lines = Lipinski boundaries')))
                fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="HBD+HBA ≤ 10")
                fig.add_vline(x=140, line_dash="dash", line_color="red", annotation_text="TPSA ≤ 140")
                fig.update_layout(height=300, margin=dict(t=55, b=30, l=30, r=30))
                st.plotly_chart(fig, width='stretch')
            elif has_tpsa:
                # Fallback: TPSA distribution
                fig = px.histogram(unique_df['TPSA'].dropna(), nbins=25, title='TPSA Distribution')
                fig.add_vline(x=140, line_dash="dash", line_color="red", annotation_text="TPSA ≤ 140")
                fig.update_layout(height=300, margin=dict(t=40, b=30, l=30, r=30))
                st.plotly_chart(fig, width='stretch')
            else:
                st.caption("TPSA vs HBD+HBA requires HBD/HBA data (reprocess compounds)")

        # Second row: TPSA vs QED and QED Distribution
        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            # TPSA vs QED scatter plot
            has_qed = 'QED' in unique_df.columns and unique_df['QED'].notna().any()
            if has_tpsa and has_qed:
                fig = px.scatter(
                    unique_df.dropna(subset=['TPSA', 'QED']),
                    x='TPSA',
                    y='QED',
                    color='RO5_Violations' if 'RO5_Violations' in unique_df.columns else None,
                    hover_data=['ChEMBL_ID'] if 'ChEMBL_ID' in unique_df.columns else None,
                    title='TPSA vs QED',
                    color_continuous_scale='RdYlGn_r'  # Reversed: lower violations = greener
                )
                fig.update_layout(title=dict(text='TPSA vs QED', subtitle=dict(text='Drug-likeness — green dashed = good QED threshold')))
                fig.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Good QED (≥0.5)")
                fig.add_vline(x=140, line_dash="dash", line_color="red", annotation_text="TPSA ≤ 140")
                fig.update_layout(height=300, margin=dict(t=55, b=30, l=30, r=30))
                st.plotly_chart(fig, width='stretch')

        with viz_col4:
            # 10xPSA_MW vs NPOLoNHA scatter plot (replaces QED distribution)
            has_psa_mw = '10xPSA_MW' in unique_df.columns and unique_df['10xPSA_MW'].notna().any()
            has_npol_nha = 'NPOLoNHA' in unique_df.columns and unique_df['NPOLoNHA'].notna().any()

            if has_psa_mw and has_npol_nha:
                plot_df = unique_df.dropna(subset=['10xPSA_MW', 'NPOLoNHA'])
                # Need >=2 points AND variance in x values for regression
                x_vals = plot_df['10xPSA_MW'].values if len(plot_df) >= 2 else np.array([])
                y_vals = plot_df['NPOLoNHA'].values if len(plot_df) >= 2 else np.array([])
                can_regress = len(plot_df) >= 2 and len(np.unique(x_vals)) > 1

                if can_regress:
                    # Calculate R² statistics
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_vals, y_vals)
                    r_squared = r_value ** 2
                    title = f'10×PSA/MW vs NPOL/NHA (R²={r_squared:.3f})'
                    show_trendline = True
                    stats_caption = f"R²={r_squared:.4f}, slope={slope:.4f}, p={p_value:.2e}"
                elif len(plot_df) >= 1:
                    # Can show scatter but not trendline (all x values identical or only 1 point)
                    title = '10×PSA/MW vs NPOL/NHA'
                    show_trendline = False
                    stats_caption = "Insufficient variance for regression" if len(plot_df) >= 2 else ""
                else:
                    show_trendline = False
                    title = None

                if len(plot_df) >= 1:
                    # Build customdata for structure viewer
                    customdata_cols = None
                    if 'SMILES' in plot_df.columns:
                        customdata_cols = ['SMILES']
                        if 'Molecule_Name' in plot_df.columns:
                            customdata_cols.append('Molecule_Name')
                        if 'ChEMBL_ID' in plot_df.columns:
                            customdata_cols.append('ChEMBL_ID')

                    fig = px.scatter(
                        plot_df,
                        x='10xPSA_MW',
                        y='NPOLoNHA',
                        color='QED' if 'QED' in plot_df.columns and plot_df['QED'].notna().any() else None,
                        hover_data=['ChEMBL_ID', 'Molecule_Name'] if all(c in plot_df.columns for c in ['ChEMBL_ID', 'Molecule_Name']) else None,
                        title=title,
                        trendline="ols" if show_trendline else None,
                        color_continuous_scale='Viridis',
                        custom_data=customdata_cols
                    )
                    fig.update_layout(
                        height=300,
                        margin=dict(t=40, b=30, l=30, r=30),
                        xaxis_title="10 × PSA/MW",
                        yaxis_title="NPOL/NHA"
                    )
                    st.plotly_chart(fig, width='stretch', key="psa_npol_scatter_chart")
                    if stats_caption:
                        st.caption(stats_caption)

                    # Embed structure viewer for click-to-view molecules
                    if 'SMILES' in plot_df.columns:
                        render_structure_viewer_hint()
                        embed_structure_viewer(
                            chart_id="psa_npol_scatter_chart",
                            x_col='10xPSA_MW',
                            y_col='NPOLoNHA',
                            name_col='Molecule_Name' if 'Molecule_Name' in plot_df.columns else None
                        )
                else:
                    st.caption("No data points for 10×PSA/MW vs NPOL/NHA plot")
            else:
                st.caption("10×PSA/MW vs NPOL/NHA requires reprocessing compounds")

    else:
        # Individual compounds view - PubChem-style table
        st.markdown("**Computed Properties by Compound**")
        st.caption("Each row represents one compound with its computed properties (like PubChem format)")

        # Define key properties to show (PubChem-like order)
        key_props = ['Molecular_Weight', 'MolLogP', 'LogP', 'TPSA', 'HBD', 'HBA',
                     'Heavy_Atoms', 'Rotatable_Bonds', 'Aromatic_Rings', 'NPOL',
                     'QED', 'RO5_Violations', 'NP_Likeness_Score',
                     'PSAoMW', '10xPSA_MW', 'NPOLoNHA']

        # Filter to only available properties
        available_key_props = [p for p in key_props if p in unique_df.columns]

        # Add other properties not in the key list
        all_props = available_key_props + [c for c in (physchem_cols + druglike_cols + other_cols) if c not in available_key_props]

        # Build display columns
        display_cols = []
        if 'ChEMBL_ID' in unique_df.columns:
            display_cols.append('ChEMBL_ID')
        if 'Molecule_Name' in unique_df.columns:
            display_cols.append('Molecule_Name')

        # Add property columns
        display_cols.extend(all_props[:12])  # Limit to 12 properties for readability

        display_df = unique_df[display_cols].copy()

        # Clean up molecule names
        if 'Molecule_Name' in display_df.columns:
            display_df['Molecule_Name'] = display_df['Molecule_Name'].apply(
                lambda x: x[:25] if isinstance(x, str) else ''
            )

        # Round numeric columns
        for col in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = display_df[col].round(2)

        # Rename columns for display
        column_renames = {
            'Molecular_Weight': 'MW (g/mol)',
            'MolLogP': 'XLogP3',
            'LogP': 'LogP',
            'TPSA': 'TPSA (Å²)',
            'HBD': 'HBD',
            'HBA': 'HBA',
            'Heavy_Atoms': 'Heavy Atoms',
            'Rotatable_Bonds': 'Rot. Bonds',
            'Aromatic_Rings': 'Arom. Rings',
            'NPOL': 'Polar Atoms',
            'RO5_Violations': '#RO5 Viol.',
            'NP_Likeness_Score': 'NP Likeness',
        }
        display_df = display_df.rename(columns=column_renames)

        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            height=min(500, len(display_df) * 35 + 40)
        )


def _render_activity_analysis(df: pd.DataFrame) -> None:
    """Activity analysis with charts."""
    if df is None or 'Activity_Type' not in df.columns:
        st.info("No activity data available")
        return

    st.markdown("**Bioactivity Distribution**")
    st.caption("Distribution of activity measurements across different assay types (IC50, Ki, Kd, EC50, etc.)")

    # Activity distribution table and pie chart
    counts = df['Activity_Type'].value_counts().reset_index()
    counts.columns = ['Type', 'Count']
    counts['%'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(counts, width='stretch', hide_index=True, height=300)

    with col2:
        # Larger pie chart with legend
        fig = px.pie(counts, values='Count', names='Type', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(
            title=dict(text='Bioactivity Distribution', subtitle=dict(text=f'{len(counts)} activity types')),
            margin=dict(t=55, b=30, l=30, r=30),
            height=370,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                title_text="Activity Types"
            )
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, width='stretch')

    # Statistics by type
    if 'pActivity' in df.columns:
        st.markdown("---")
        st.markdown("**Statistics by Activity Type**")
        st.caption("pActivity = -log10(Activity in M). Higher values indicate more potent compounds.")
        stats = df.groupby('Activity_Type')['pActivity'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        stats.columns = ['Count', 'Mean pActivity', 'Std Dev', 'Min', 'Max']
        st.dataframe(stats, width='stretch')

    # Target distribution
    if 'Target_Name' in df.columns or 'Target_ChEMBL_ID' in df.columns:
        st.markdown("---")
        st.markdown("**Target Distribution**")
        st.caption("Top 10 biological targets with most activity data points")
        target_col = 'Target_Name' if 'Target_Name' in df.columns else 'Target_ChEMBL_ID'
        target_counts = df[target_col].value_counts().head(10)

        fig = px.bar(x=target_counts.values, y=target_counts.index, orientation='h',
                     color=target_counts.values, color_continuous_scale='Blues')
        fig.update_layout(
            title=dict(text='Top Targets', subtitle=dict(text='By number of activity records')),
            height=min(370, len(target_counts) * 35 + 70),
            margin=dict(t=55, b=10, l=10, r=10),
            xaxis_title="Number of Activity Records",
            yaxis_title="",
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, width='stretch')


def _render_efficiency_analysis(df: pd.DataFrame) -> None:
    """Efficiency metrics analysis with interactive filtering."""
    metrics = ['SEI', 'BEI', 'NSEI', 'NBEI']
    avail = [m for m in metrics if m in df.columns]

    if not avail:
        st.info("No efficiency metrics available")
        return

    # Overall stats table
    st.markdown("**Overall Efficiency Metrics Summary**")
    stats_data = []
    for m in avail:
        vals = df[m].dropna()
        if len(vals) > 0:
            stats_data.append({
                'Metric': m,
                'Count': len(vals),
                'Mean': round(vals.mean(), 3),
                'Std': round(vals.std(), 3),
                'Min': round(vals.min(), 3),
                'Max': round(vals.max(), 3)
            })

    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), width='stretch', hide_index=True)

    # Controls for interactive plot
    st.markdown("---")
    ctrl_cols = st.columns([1, 1, 2])

    with ctrl_cols[0]:
        metric_choice = st.selectbox(
            "Metric",
            avail,
            key="eff_metric_choice",
            help="Select efficiency metric to display"
        )

    with ctrl_cols[1]:
        # Color by options - categorical columns that make sense for grouping
        color_options = ['None']
        categorical_cols = ['Activity_Type', 'ChEMBL_ID', 'IMP_Classification', 'Target_Name']
        color_options += [c for c in categorical_cols if c in df.columns]

        color_by = st.selectbox(
            "Color by",
            color_options,
            key="eff_color_by",
            help="Click legend to show/hide groups"
        )

    # Prepare data for plotting
    plot_df = df.dropna(subset=[metric_choice]).copy()

    if plot_df.empty:
        st.warning(f"No data available for {metric_choice}")
        return

    # Build customdata for structure viewer (box plots support click events)
    customdata_cols = None
    if 'SMILES' in plot_df.columns:
        customdata_cols = ['SMILES']
        if 'Molecule_Name' in plot_df.columns:
            customdata_cols.append('Molecule_Name')
        if 'ChEMBL_ID' in plot_df.columns:
            customdata_cols.append('ChEMBL_ID')

    # Layout: Chart takes most space, compact sidebar for outliers/groups
    col1, col2 = st.columns([5, 1])

    with col1:
        # Box plot with optional coloring
        if color_by != "None":
            fig = px.box(
                plot_df,
                x=color_by,
                y=metric_choice,
                color=color_by,
                points='all',  # Show all points for structure viewer clicks
                hover_data=['ChEMBL_ID', 'Molecule_Name'] if all(c in plot_df.columns for c in ['ChEMBL_ID', 'Molecule_Name']) else None,
                custom_data=customdata_cols
            )
            fig.update_layout(
                title=dict(text=f'{metric_choice} Distribution', subtitle=dict(text=f'Grouped by {color_by}')),
                height=470,
                margin=dict(t=55, b=80, r=10),
                xaxis_tickangle=-45,
                # Vertical legend on right side, inside chart area
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=1.02,
                    title_text="",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                )
            )
            st.plotly_chart(fig, width='stretch', key="eff_box_chart")
            st.caption("💡 **Click legend items** to show/hide groups. Double-click to isolate.")
        else:
            # Simple histogram without grouping
            fig = px.histogram(plot_df, x=metric_choice, nbins=30)
            fig.update_layout(
                title=dict(text=f'{metric_choice} Distribution', subtitle=dict(text='Frequency distribution across compounds')),
                height=420, margin=dict(t=55, b=30)
            )
            st.plotly_chart(fig, width='stretch', key="eff_hist_chart")

        # Embed structure viewer for click-to-view molecules (box plots)
        if color_by != "None" and 'SMILES' in plot_df.columns:
            render_structure_viewer_hint()
            embed_structure_viewer(
                chart_id="eff_box_chart",
                x_col=color_by,
                y_col=metric_choice,
                name_col='Molecule_Name' if 'Molecule_Name' in plot_df.columns else None
            )

    with col2:
        # Compact outlier summary - smaller text, tighter spacing
        st.markdown("<p style='font-size: 13px; font-weight: 600; margin-bottom: 4px;'>Outliers</p>", unsafe_allow_html=True)
        for m in avail:
            outlier_col = f'Is_{m}_Outlier'
            if outlier_col in df.columns:
                count = int(df[outlier_col].sum())
                color = "#ff6b6b" if count > 0 else "#51cf66"
                st.markdown(
                    f"<div style='background: {color}; color: white; padding: 2px 6px; "
                    f"border-radius: 4px; font-size: 11px; margin: 2px 0; text-align: center;'>"
                    f"{m}: {count}</div>",
                    unsafe_allow_html=True
                )

        # Show group counts when colored - compact
        if color_by != "None":
            st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 11px; font-weight: 600; margin-bottom: 2px;'>Groups</p>", unsafe_allow_html=True)
            group_counts = plot_df[color_by].value_counts()
            for grp, cnt in group_counts.head(6).items():
                st.markdown(f"<p style='font-size: 10px; margin: 0; color: #666;'>{str(grp)[:12]}: {cnt}</p>", unsafe_allow_html=True)

    # Efficiency Metrics by Target table (after visualization)
    st.markdown("---")
    st.markdown("**Efficiency Metrics by Target**")

    # Determine target column
    target_col = None
    if 'Target_Name' in df.columns:
        target_col = 'Target_Name'
    elif 'Target_ChEMBL_ID' in df.columns:
        target_col = 'Target_ChEMBL_ID'

    if target_col and any(m in df.columns for m in metrics):
        # Build target metrics table
        target_metrics = []
        for target in df[target_col].dropna().unique():
            target_df = df[df[target_col] == target]
            row = {
                'Target_ChEMBL_ID': target if target_col == 'Target_ChEMBL_ID' else target_df['Target_ChEMBL_ID'].iloc[0] if 'Target_ChEMBL_ID' in target_df.columns else '',
                'Target_Name': target if target_col == 'Target_Name' else ''
            }
            for m in ['SEI', 'BEI', 'NSEI', 'NBEI']:
                if m in target_df.columns:
                    vals = target_df[m].dropna()
                    row[f'{m} Count'] = len(vals)
                    row[f'{m} Mean'] = round(vals.mean(), 3) if len(vals) > 0 else None
                    row[f'{m} Median'] = round(vals.median(), 3) if len(vals) > 0 else None
            target_metrics.append(row)

        if target_metrics:
            target_metrics_df = pd.DataFrame(target_metrics)
            # Sort by SEI Count if available
            if 'SEI Count' in target_metrics_df.columns:
                target_metrics_df = target_metrics_df.sort_values('SEI Count', ascending=False)

            # Make Target_ChEMBL_ID a clickable link to ChEMBL
            if 'Target_ChEMBL_ID' in target_metrics_df.columns:
                target_metrics_df['Target_ChEMBL_ID'] = target_metrics_df['Target_ChEMBL_ID'].apply(
                    lambda x: f"https://www.ebi.ac.uk/chembl/explore/target/{x}" if x else ""
                )
                col_config = {
                    "Target_ChEMBL_ID": st.column_config.LinkColumn(
                        "Target ChEMBL ID",
                        display_text=r"https://www\.ebi\.ac\.uk/chembl/explore/target/(.*)",
                    )
                }
            else:
                col_config = {}

            st.dataframe(target_metrics_df, width='stretch', hide_index=True, height=300, column_config=col_config)

    # Explanation box
    with st.expander("📖 Understanding Efficiency Metrics by Target", expanded=False):
        st.markdown("""
**This table shows efficiency metrics calculated for each target in the dataset.**

- **SEI (Surface Efficiency Index)**: Measures activity relative to polar surface area. Formula: `SEI = pActivity / (PSA / 100)`
- **BEI (Binding Efficiency Index)**: Measures activity relative to molecular weight. Formula: `BEI = pActivity / (MW / 1000)`
- **NSEI (Normalized Surface Efficiency Index)**: Measures activity relative to polar atom count. Formula: `NSEI = pActivity / NPOL` (where NPOL = N + O atom count)
- **NBEI (Normalized Binding Efficiency Index)**: Measures activity relative to heavy atom count. Formula: `NBEI = pActivity / NHA` (where NHA = heavy atom count)

**Interpretation:** Higher values indicate more efficient compounds for that target. Compounds with high efficiency metrics achieve strong binding without excessive molecular size or polarity.
        """)


def _render_pains_analysis(df: pd.DataFrame) -> None:
    """Assay Interference Flags analysis - dedicated section."""
    if df is None:
        st.info("No data available")
        return

    has_assay = 'PAINS_Violation' in df.columns

    if not has_assay:
        st.info("No assay interference data available. Re-run analysis to generate PAINS screening.")
        return

    unique_df = df.drop_duplicates('ChEMBL_ID') if 'ChEMBL_ID' in df.columns else df
    total = len(unique_df)

    st.markdown("**Assay Interference Flags**")
    st.caption("Detection of compounds with known assay interference mechanisms")

    # Summary metrics row — 7 flags with help tooltips
    # (column_name, emoji, short_description, help_tooltip)
    flags = {
        'PAINS': ('PAINS_Violation', '🔴', 'Pan-Assay Interference',
                  'Pan-Assay Interference Compounds (PAINS) — 480 substructure filters that identify '
                  'compounds prone to false positives across multiple assay types. Baell & Holloway (2010).'),
        'Aggregator': ('Aggregator_Risk', '🟠', 'Colloidal Aggregation',
                       'Compounds that form colloidal aggregates in aqueous solution, causing non-specific '
                       'enzyme inhibition. Shoichet Lab criteria: >=3 aromatic rings, >300 Da, <=2 rotatable bonds, >3 LogP.'),
        'Redox': ('Redox_Reactive', '🟡', 'Redox Cycling',
                  'Redox-active compounds (quinones, catechols, hydroquinones, nitroaromatics) that generate '
                  'H2O2/ROS in assay buffers, causing false activity signals. 10 SMARTS patterns.'),
        'Fluorescence': ('Fluorescence_Interference', '🔵', 'Fluorescence Interference',
                         'Autofluorescent scaffolds (coumarins, xanthenes, PAHs, stilbenes, flavonoids, acridines) '
                         'that interfere with fluorescence-based assay readouts. 13 SMARTS patterns.'),
        'Thiol': ('Thiol_Reactive', '🟣', 'Thiol Reactivity',
                  'Electrophilic compounds (Michael acceptors, acylating agents, epoxides, aldehydes) that '
                  'react non-specifically with cysteine residues in target proteins. 15 SMARTS patterns.'),
        'BRENK': ('BRENK_Alerts', '🟤', 'Unwanted Substructures',
                  'BRENK filter — 104 unwanted substructure patterns including reactive groups, toxic moieties, '
                  'and metabolic liabilities. Used in screening library design. Brenk et al. (2008).'),
        'NIH': ('NIH_Alerts', '⚪', 'NIH Problematic Groups',
                'NIH-defined problematic functional groups that are frequently associated with assay artifacts '
                'or poor drug-likeness. RDKit FilterCatalog.NIH. Doveston et al. (2015).'),
    }

    # Detail column mapping for building combined Details
    detail_col_map = {
        'PAINS': 'PAINS_Details',
        'Aggregator': 'Aggregator_Details',
        'Redox': 'Redox_Details',
        'Fluorescence': 'Fluorescence_Details',
        'Thiol': 'Thiol_Details',
        'BRENK': 'BRENK_Details',
        'NIH': 'NIH_Details',
    }

    # Display all 7 flags as styled cards in a single row
    flag_data = []
    flag_items = list(flags.items())

    cards_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">'
    for name, (col, emoji, desc, helptext) in flag_items:
        count = int(unique_df[col].sum()) if col in unique_df.columns else 0
        pct = count / total * 100 if total > 0 else 0
        flag_data.append({
            'Flag': f"{emoji} {name}",
            'Count': count,
            '%': f"{pct:.0f}%",
            'Description': desc
        })
        is_flagged = count > 0
        border_color = '#dc3545' if is_flagged else '#28a745'
        count_color = '#dc3545' if is_flagged else '#28a745'
        status_text = f'&#9888; Flagged ({pct:.0f}%)' if is_flagged else '&#10003; Clean'
        status_color = '#ffa94d' if is_flagged else '#51cf66'
        escaped_help = html.escape(helptext)
        cards_html += f'''
        <div title="{escaped_help}" style="flex:1;min-width:110px;background:#1e1e2e;border-left:4px solid {border_color};
            border-radius:6px;padding:12px 10px;text-align:center;cursor:help;">
            <div style="font-size:1.8em;font-weight:bold;color:{count_color};">{count}</div>
            <div style="font-size:0.9em;color:#ccc;margin:4px 0;font-weight:600;">{html.escape(name)}</div>
            <div style="font-size:0.7em;color:{status_color};">{status_text}</div>
        </div>'''
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("---")

    # Detailed table
    if flag_data:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Flag Summary**")
            st.dataframe(pd.DataFrame(flag_data), width='stretch', hide_index=True, height=290)

        with col2:
            # Build unique compound rows with combined Flags and Details
            flag_col_names = {name: info[0] for name, info in flags.items()}
            available_flag_cols = [name for name, col in flag_col_names.items() if col in unique_df.columns]

            if not available_flag_cols:
                st.success("No compounds flagged for assay interference")
            else:
                # Build mask: any flag is True
                any_flagged_mask = unique_df[[flag_col_names[n] for n in available_flag_cols]].any(axis=1)
                flagged_rows = unique_df[any_flagged_mask]

                if flagged_rows.empty:
                    st.success("No compounds flagged for assay interference")
                else:
                    # Build one row per unique compound
                    compound_records = []
                    for _, row in flagged_rows.iterrows():
                        mol_name = row.get('Molecule_Name', '')
                        if pd.isna(mol_name) or not isinstance(mol_name, str):
                            mol_name = ''

                        # Collect active flags
                        active_flags = []
                        for name in available_flag_cols:
                            if row.get(flag_col_names[name], False):
                                active_flags.append(name)

                        # Collect details from detail columns
                        details_parts = []
                        for flag_name in active_flags:
                            dcol = detail_col_map.get(flag_name, '')
                            if dcol and dcol in row.index:
                                val = row.get(dcol, '')
                                if val and pd.notna(val) and str(val).strip():
                                    details_parts.append(f"{flag_name}: {val}")

                        compound_records.append({
                            'ChEMBL_ID': row.get('ChEMBL_ID', 'Unknown'),
                            'Molecule': mol_name[:25] if mol_name else '',
                            'Flags': ', '.join(active_flags),
                            'Details': '; '.join(details_parts),
                        })

                    flagged_df = pd.DataFrame(compound_records)

                    # Filter by flag type
                    active_flag_names = sorted(set(
                        f for rec in compound_records for f in rec['Flags'].split(', ') if f
                    ))
                    st.markdown("**Flagged Compounds**")
                    if active_flag_names:
                        selected_flags = st.multiselect(
                            "Filter by flag type",
                            options=active_flag_names,
                            default=[],
                            key="assay_interference_flag_filter",
                            help="Select one or more flags to filter. Leave empty to show all."
                        )
                        if selected_flags:
                            mask = flagged_df['Flags'].apply(
                                lambda x: any(f in x for f in selected_flags)
                            )
                            flagged_df = flagged_df[mask]

                    st.dataframe(
                        flagged_df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'ChEMBL_ID': st.column_config.TextColumn('ChEMBL_ID', width='small'),
                            'Molecule': st.column_config.TextColumn('Molecule', width='small'),
                            'Flags': st.column_config.TextColumn('Flags', width='medium'),
                            'Details': st.column_config.TextColumn('Details', width='large'),
                        },
                    )

    # PAINS patterns breakdown (if available)
    if 'PAINS_Pattern' in unique_df.columns:
        st.markdown("---")
        st.markdown("**PAINS Patterns Detected**")
        patterns = unique_df[unique_df['PAINS_Pattern'].notna()]['PAINS_Pattern'].value_counts()
        if not patterns.empty:
            fig = px.bar(x=patterns.values, y=patterns.index, orientation='h')
            fig.update_layout(height=min(250, len(patterns) * 30 + 50), margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, width='stretch')

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


def _render_imp_score_breakdown(df: pd.DataFrame, compound_name: str) -> None:
    """
    Render detailed IMP score breakdown for a representative compound.

    Shows all individual scores, efficiency metrics, and contribution breakdown.
    """
    if df is None or df.empty:
        return

    # Check if we have the required columns
    required_cols = ['IMP_Final_Score', 'Efficiency_Score', 'Angle_Score', 'Distance_Score']
    if not all(col in df.columns for col in required_cols):
        return

    st.markdown("---")

    with st.expander("🎯 Detailed Score Breakdown", expanded=True):
        # Get representative row (highest scoring or first valid row)
        valid_df = df[df['IMP_Final_Score'].notna()]
        if valid_df.empty:
            st.info("No valid IMP scores available for breakdown")
            return

        # Use highest scoring compound for breakdown
        row = valid_df.loc[valid_df['IMP_Final_Score'].idxmax()]

        # Final Score Hero Section
        final_score = row.get('IMP_Final_Score', 0)
        classification = row.get('IMP_Classification', 'Unknown')
        priority = row.get('IMP_Priority', 'N/A')

        # Color based on score - Higher IMP = MORE DANGEROUS (red)
        if final_score >= 0.9:
            score_color = "#721c24"  # Dark Red - Exceptional IMP
        elif final_score >= 0.7:
            score_color = "#dc3545"  # Red - Strong IMP
        elif final_score >= 0.5:
            score_color = "#fd7e14"  # Orange - Moderate IMP
        elif final_score >= 0.3:
            score_color = "#28a745"  # Green - Weak IMP
        else:
            score_color = "#155724"  # Dark Green - Not IMP

        st.markdown(f"""
        <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, {score_color}22, {score_color}11); border-radius: 10px; border: 2px solid {score_color}; margin-bottom: 15px;">
            <h2 style="color: {score_color}; margin: 0; font-size: 2.5em;">{final_score:.3f}</h2>
            <p style="color: {score_color}; margin: 5px 0; font-size: 1.1em; font-weight: bold;">{classification}</p>
            <p style="color: #666; margin: 0; font-size: 0.9em;">Priority: {priority} | Best scoring compound shown</p>
        </div>
        """, unsafe_allow_html=True)

        # Efficiency Metrics Section
        st.markdown("#### 📊 Efficiency Metrics")
        st.caption("All four metrics are calculated. Only **SEI** and **BEI** are used in the score.")

        eff_cols = st.columns(4)

        with eff_cols[0]:
            sei = row.get('SEI')
            st.metric("SEI", f"{sei:.2f}" if pd.notna(sei) else "N/A",
                      help="Surface Efficiency Index = pActivity ÷ (PSA/100). **Used in score.**")

        with eff_cols[1]:
            bei = row.get('BEI')
            st.metric("BEI", f"{bei:.2f}" if pd.notna(bei) else "N/A",
                      help="Binding Efficiency Index = pActivity ÷ (MW/1000). **Used in score.**")

        with eff_cols[2]:
            nsei = row.get('NSEI')
            st.metric("NSEI", f"{nsei:.2f}" if pd.notna(nsei) else "N/A",
                      help="Normalized SEI = pActivity ÷ NPOL. Display only.")

        with eff_cols[3]:
            nbei = row.get('NBEI')
            st.metric("NBEI", f"{nbei:.3f}" if pd.notna(nbei) else "N/A",
                      help="Normalized BEI = pActivity ÷ NHA. Display only.")

        # Plane Geometry Section
        st.markdown("#### 📐 Efficiency Plane Geometry")

        geom_cols = st.columns(2)

        with geom_cols[0]:
            modulus = row.get('Modulus_SEI_BEI')
            st.metric("Modulus", f"{modulus:.2f}" if pd.notna(modulus) else "N/A")
            st.caption("""
            The modulus measures the distance of the combined efficiency vector (SEI, BEI)
            from the origin on the efficiency plane. It represents the overall efficiency
            magnitude. While derived from SEI and BEI, the modulus is independent of
            the development angle—the angle defines direction, not magnitude.
            """)

        with geom_cols[1]:
            angle = row.get('Angle_SEI_BEI')
            if pd.notna(angle):
                angle_deviation = abs(angle - 45)
                if angle_deviation < 10:
                    angle_status = "✅ Optimal"
                elif angle_deviation < 20:
                    angle_status = "⚠️ Moderate"
                else:
                    angle_status = "❌ Unbalanced"
                st.metric("Development Angle", f"{angle:.1f}°", delta=angle_status, delta_color="off")
            else:
                st.metric("Development Angle", "N/A")
            st.caption("Optimal angle is 45°. <30° = too hydrophobic, >60° = too polar.")

        # Component Scores Section
        st.markdown("#### 🎯 Component Scores & Contributions")

        comp_cols = st.columns(5)

        with comp_cols[0]:
            eff_score = row.get('Efficiency_Score', 0)
            eff_contrib = row.get('Efficiency_Contribution', 0)
            st.metric("Efficiency", f"{eff_score:.3f}" if pd.notna(eff_score) else "N/A",
                      help="Weight: 45%")
            if pd.notna(eff_score):
                st.progress(max(0.0, min(1.0, float(eff_score))))
            st.caption(f"Contribution: {eff_contrib:.3f}" if pd.notna(eff_contrib) else "")
            sei_z = row.get('SEI_zscore', None)
            bei_z = row.get('BEI_zscore', None)
            if sei_z is not None and bei_z is not None and pd.notna(sei_z) and pd.notna(bei_z):
                st.caption(f"SEI z={sei_z:.2f} · BEI z={bei_z:.2f}")

        with comp_cols[1]:
            dist_score = row.get('Distance_Score', 0)
            dist_contrib = row.get('Distance_Contribution', 0)
            st.metric("Distance", f"{dist_score:.3f}" if pd.notna(dist_score) else "N/A",
                      help="Weight: 20%")
            if pd.notna(dist_score):
                st.progress(max(0.0, min(1.0, float(dist_score))))
            st.caption(f"Contribution: {dist_contrib:.3f}" if pd.notna(dist_contrib) else "")
            modulus = row.get('Modulus_SEI_BEI', None)
            if modulus is not None and pd.notna(modulus):
                st.caption(f"Modulus: {modulus:.2f}")

        with comp_cols[2]:
            ang_score = row.get('Angle_Score', 0)
            ang_contrib = row.get('Angle_Contribution', 0)
            st.metric("Angle", f"{ang_score:.3f}" if pd.notna(ang_score) else "N/A",
                      help="Weight: 15%")
            if pd.notna(ang_score):
                st.progress(max(0.0, min(1.0, float(ang_score))))
            st.caption(f"Contribution: {ang_contrib:.3f}" if pd.notna(ang_contrib) else "")
            angle = row.get('Angle_SEI_BEI', None)
            if angle is not None and pd.notna(angle):
                st.caption(f"Angle: {angle:.1f}° (optimal: 45°)")

        with comp_cols[3]:
            int_score = row.get('Interference_Score', 0)
            int_contrib = row.get('Interference_Contribution', 0)
            st.metric("Interference", f"{int_score:.3f}" if pd.notna(int_score) else "N/A",
                      help="Weight: 15% — Scored flags / 5 (BRENK/NIH display-only)")
            if pd.notna(int_score):
                st.progress(max(0.0, min(1.0, float(int_score))))
            st.caption(f"Contribution: {int_contrib:.3f}" if pd.notna(int_contrib) else "")

            # Show which flags triggered
            scored_flags = [
                ('PAINS_Violation', 'PAINS'),
                ('Aggregator_Risk', 'Aggregator'),
                ('Redox_Reactive', 'Redox'),
                ('Fluorescence_Interference', 'Fluorescence'),
                ('Thiol_Reactive', 'Thiol'),
            ]
            triggered = sum(1 for col, _ in scored_flags if row.get(col, 0) == 1)
            flag_parts = []
            for col, label in scored_flags:
                val = row.get(col, 0)
                flag_parts.append(f"{'🔴' if val == 1 else '🟢'} {label}")
            st.caption(f"{triggered}/5 flags triggered")
            st.caption(" · ".join(flag_parts))

        with comp_cols[4]:
            pdb_score = row.get('PDB_Score', 0)
            pdb_contrib = row.get('PDB_Contribution', 0)
            st.metric("PDB Evidence", f"{pdb_score:.3f}" if pd.notna(pdb_score) else "N/A",
                      help="Weight: 5%")
            if pd.notna(pdb_score):
                st.progress(max(0.0, min(1.0, float(pdb_score))))
            st.caption(f"Contribution: {pdb_contrib:.3f}" if pd.notna(pdb_contrib) else "")
            pdb_hits = row.get('PDB_Hits', None)
            if pdb_hits is not None and pd.notna(pdb_hits):
                st.caption(f"PDB hits: {int(pdb_hits)}")

        # Final Calculation Section
        st.markdown("#### 🧮 Final Calculation")

        base_score = row.get('IMP_Base_Score', 0)
        qed = row.get('QED', 0)
        qed_mult = row.get('QED_Multiplier', 0)
        qed_impact = row.get('QED_Impact', 0)

        calc_cols = st.columns(3)

        with calc_cols[0]:
            st.metric("Base Score", f"{base_score:.3f}" if pd.notna(base_score) else "N/A",
                      help="Sum of weighted component contributions")

        with calc_cols[1]:
            st.metric("QED", f"{qed:.3f}" if pd.notna(qed) else "N/A",
                      help="Quantitative Estimate of Drug-likeness (0-1)")

        with calc_cols[2]:
            st.metric("QED Multiplier", f"{qed_mult:.3f}" if pd.notna(qed_mult) else "N/A",
                      delta=f"Impact: {qed_impact:+.3f}" if pd.notna(qed_impact) else None,
                      help="Formula: 0.75 + 0.25 × QED. Floor at 75%.")

        # Formula display with actual values
        if pd.notna(base_score) and pd.notna(qed_mult):
            eff_s = row.get('Efficiency_Score', 0)
            dist_s = row.get('Distance_Score', 0)
            ang_s = row.get('Angle_Score', 0)
            int_s = row.get('Interference_Score', 0)
            pdb_s = row.get('PDB_Score', 0)
            st.markdown(f"""
<div style="background-color: #1a1a2e; padding: 16px 20px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.95rem; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word;">
<span style="color: #82aaff; font-weight: 600;">Base Score</span>
  = 0.45 x Eff + 0.20 x Dist + 0.15 x Angle + 0.15 x Interf + 0.05 x PDB
  = 0.45 x <span style="color: #c3e88d;">{eff_s:.3f}</span> + 0.20 x <span style="color: #c3e88d;">{dist_s:.3f}</span> + 0.15 x <span style="color: #c3e88d;">{ang_s:.3f}</span> + 0.15 x <span style="color: #c3e88d;">{int_s:.3f}</span> + 0.05 x <span style="color: #c3e88d;">{pdb_s:.3f}</span>
  = <span style="color: #ffcb6b; font-weight: 600;">{base_score:.3f}</span>

<span style="color: #82aaff; font-weight: 600;">QED Multiplier</span>
  = 0.75 + 0.25 x {qed:.3f} = <span style="color: #ffcb6b; font-weight: 600;">{qed_mult:.3f}</span>

<span style="color: #82aaff; font-weight: 600;">Final Score</span>
  = Base Score x QED Multiplier
  = {base_score:.3f} x {qed_mult:.3f}
  = <span style="color: #f78c6c; font-size: 1.1em; font-weight: 700;">{final_score:.3f}</span>
</div>
            """, unsafe_allow_html=True)

        # Contribution Pie Chart
        _render_contribution_chart(row)

        # PDB Details (if available)
        pdb_structures = row.get('PDB_Num_Structures', 0)
        if pd.notna(pdb_structures) and pdb_structures > 0:
            st.markdown("#### 🔬 PDB Structural Evidence")

            pdb_cols = st.columns(4)

            with pdb_cols[0]:
                st.metric("Total Structures", int(pdb_structures))
            with pdb_cols[1]:
                st.metric("High Quality (<2.0Å)", int(row.get('PDB_High_Quality', 0)))
            with pdb_cols[2]:
                st.metric("Medium Quality (2-3Å)", int(row.get('PDB_Medium_Quality', 0)))
            with pdb_cols[3]:
                st.metric("Poor Quality (>3.0Å)", int(row.get('PDB_Poor_Quality', 0)))

            st.caption("""
            **Interpretation:**
            High quality structures (<2.0Å) provide strongest validation.
            Multiple structures increase confidence.
            Low PDB score with high efficiency = potential artifact (RED FLAG).
            """)


def _render_contribution_chart(row: pd.Series) -> None:
    """Render a pie chart showing base score composition + QED multiplier annotation."""
    components = [
        ('Efficiency', 0.45, 'Efficiency_Score'),
        ('Distance', 0.20, 'Distance_Score'),
        ('Angle', 0.15, 'Angle_Score'),
        ('Interference', 0.15, 'Interference_Score'),
        ('PDB', 0.05, 'PDB_Score'),
    ]

    names = []
    values = []
    custom_text = []
    for name, weight, col in components:
        score = row.get(col, 0)
        score = float(score) if pd.notna(score) else 0.0
        base_contrib = weight * score
        names.append(name)
        values.append(max(base_contrib, 0.001))
        custom_text.append(f"{name} ({weight*100:.0f}%)<br>{base_contrib:.3f}")

    if sum(values) <= 0.005:
        return

    qed_mult = row.get('QED_Multiplier', 1.0)
    qed_mult = float(qed_mult) if pd.notna(qed_mult) else 1.0
    base_score = sum(v for v in values if v > 0.001)
    subtitle_text = f'Base Score = {base_score:.3f} | QED Multiplier = {qed_mult:.3f} | Final = {base_score * qed_mult:.3f}'

    fig = px.pie(
        values=values,
        names=names,
        title='Base Score Breakdown',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(title=dict(text='Base Score Breakdown', subtitle=dict(text=subtitle_text)))

    fig.update_traces(
        textposition='inside',
        text=custom_text,
        textinfo='text',
    )
    fig.update_layout(showlegend=False, height=300, margin=dict(t=55, b=10, l=10, r=10))

    st.plotly_chart(fig, width='stretch')


def _render_imp_score_analysis(df: pd.DataFrame, compound_name: str) -> None:
    """IMP Score analysis with full explanations."""
    if df is None:
        st.info("No data available")
        return

    has_imp_score = 'IMP_Final_Score' in df.columns
    has_imp = 'Is_IMP_Candidate' in df.columns

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
| Score | Classification | Priority | Recommended Action |
|-------|---------------|----------|-------------------|
| 0.90-1.00 | Exceptional IMP | 1 (Highest) | Immediate experimental validation |
| 0.70-0.89 | Strong IMP | 2 | Validate within 1 month |
| 0.50-0.69 | Moderate IMP | 3 | Monitor, gather more data |
| 0.30-0.49 | Weak IMP | 4 | Deprioritize (unless novel) |
| < 0.30 | Not IMP | None | Exclude from pipeline |

**Note on Efficiency Metrics:**
All four efficiency metrics (SEI, BEI, NSEI, NBEI) are calculated and displayed for reference.
However, only **SEI and BEI** are used in the Efficiency Outlier Score to avoid redundancy,
since NSEI and NBEI are derived from the same underlying activity data.
        """)

    # IMP Scoring
    if has_imp_score:
        st.markdown("**IMP Score Distribution**")
        st.caption("IMP Score rates each **activity record** using a composite of efficiency, distance, angle, interference, PDB evidence, and QED. "
                   "This is different from IMP Candidates below, which flags unique **compounds** by efficiency outlier detection.")

        score_cols = st.columns(4)
        scores = df['IMP_Final_Score'].dropna()
        with score_cols[0]:
            avg = scores.mean() if len(scores) > 0 else None
            st.metric("Average Score", f"{avg:.3f}" if pd.notna(avg) else "N/A")
        with score_cols[1]:
            max_val = scores.max() if len(scores) > 0 else None
            st.metric("Best Score", f"{max_val:.3f}" if pd.notna(max_val) else "N/A")
        with score_cols[2]:
            min_val = scores.min() if len(scores) > 0 else None
            st.metric("Lowest Score", f"{min_val:.3f}" if pd.notna(min_val) else "N/A")
        with score_cols[3]:
            # Higher IMP score = MORE IMP risk (worse, not better)
            moderate_plus_imp = len(scores[scores >= 0.5]) if len(scores) > 0 else 0
            st.metric("Moderate+ IMP (≥0.5)", moderate_plus_imp,
                      help="Number of activity records with IMP Score ≥ 0.5. "
                           "Each compound can have many activity records, so this count is per-record, not per-compound.")

        # Score histogram with better styling
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(df, x='IMP_Final_Score', nbins=25, color_discrete_sequence=['#636EFA'])
            fig.update_layout(
                title=dict(text='IMP Score Distribution', subtitle=dict(text='Higher score = higher false positive risk')),
                height=300,
                margin=dict(t=55, b=30, l=30, r=10),
                xaxis_title="IMP Score",
                yaxis_title="Count"
            )
            # Add vertical lines for thresholds - higher score = MORE IMP risk
            fig.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="IMP Threshold")
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Moderate+ IMP")
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Classification breakdown with color coding
            if 'IMP_Classification' in df.columns:
                st.markdown("**Quality Classification**")
                st.caption("Per activity record (IMP Score)")
                class_counts = df['IMP_Classification'].value_counts()
                for cls, count in class_counts.items():
                    pct = count / len(df) * 100
                    # Color code based on classification
                    if 'Not IMP' in str(cls) or 'Excellent' in str(cls) or 'Good' in str(cls):
                        st.success(f"**{cls}**: {count} ({pct:.0f}%)")
                    elif 'Weak' in str(cls) or 'Moderate' in str(cls):
                        st.warning(f"**{cls}**: {count} ({pct:.0f}%)")
                    else:
                        st.error(f"**{cls}**: {count} ({pct:.0f}%)")

        # Detailed Score Breakdown Section
        _render_imp_score_breakdown(df, compound_name)

    # IMP Candidates section
    if has_imp:
        st.markdown("---")
        st.markdown("**IMP Candidates Analysis**")
        st.caption("Counts unique **compounds** with ≥2 efficiency metric outliers (z-score). "
                   "This is different from IMP Score above, which scores individual activity records using a composite formula.")

        # Get IMP candidate records and unique compounds
        imp_df = df[df['Is_IMP_Candidate']]
        unique_imp_compounds = imp_df.drop_duplicates('ChEMBL_ID') if 'ChEMBL_ID' in imp_df.columns else imp_df
        total_unique = df.drop_duplicates('ChEMBL_ID')['ChEMBL_ID'].nunique() if 'ChEMBL_ID' in df.columns else len(df)

        # Also count IMP Score-classified weak/moderate IMP records for context
        imp_score_records = 0
        if 'IMP_Classification' in df.columns:
            imp_score_records = len(df[df['IMP_Classification'].str.contains('IMP', case=False, na=False)])

        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("IMP Candidates", len(unique_imp_compounds),
                      help="Unique compounds with ≥2 efficiency metrics flagged as statistical outliers (z-score detection on SEI, BEI, NSEI, NBEI)")
        with info_cols[1]:
            st.metric("Total Compounds", total_unique, help="Total unique compounds in this analysis")
        with info_cols[2]:
            pct = len(unique_imp_compounds) / total_unique * 100 if total_unique > 0 else 0
            if pct > 20:
                st.metric("% IMP", f"{pct:.1f}%", delta="High Risk", delta_color="inverse")
            elif pct > 10:
                st.metric("% IMP", f"{pct:.1f}%", delta="Moderate", delta_color="off")
            else:
                st.metric("% IMP", f"{pct:.1f}%", delta="Low", delta_color="normal")
        with info_cols[3]:
            # Show affected records (activity rows from IMP compounds)
            st.metric("Affected Records", len(imp_df), help=f"Activity records from IMP compounds (IMP Score flagged: {imp_score_records})")

        if not unique_imp_compounds.empty:
            st.markdown("**IMP Candidates with Target Mapping:**")

            # Build display table - one row per compound+target combination
            display_data = []
            for _, row in unique_imp_compounds.iterrows():
                chembl_id = row.get('ChEMBL_ID', 'Unknown')
                mol_name = row.get('Molecule_Name', '')
                if pd.isna(mol_name) or not isinstance(mol_name, str):
                    mol_name = ''

                imp_score_val = round(row.get('IMP_Final_Score', 0), 3) if pd.notna(row.get('IMP_Final_Score')) else 'N/A'
                imp_confidence = row.get('IMP_Confidence', 'N/A')

                # Get all records for this compound to find targets
                compound_records = imp_df[imp_df['ChEMBL_ID'] == chembl_id] if 'ChEMBL_ID' in imp_df.columns else pd.DataFrame()

                # Check for target columns
                has_target_name = 'Target_Name' in compound_records.columns
                has_target_id = 'Target_ChEMBL_ID' in compound_records.columns

                if has_target_id or has_target_name:
                    # Get unique target IDs (prefer ID for grouping)
                    target_id_col = 'Target_ChEMBL_ID' if has_target_id else 'Target_Name'
                    target_ids = compound_records[target_id_col].dropna().unique()

                    if len(target_ids) > 0:
                        for target_id in target_ids:
                            target_records = compound_records[compound_records[target_id_col] == target_id]
                            # Get average activity for this compound-target pair
                            avg_activity = target_records['pActivity'].mean() if 'pActivity' in target_records.columns else None

                            # Get target name if available
                            target_name = ''
                            if has_target_name:
                                names = target_records['Target_Name'].dropna().unique()
                                target_name = str(names[0])[:35] if len(names) > 0 else ''

                            # Get target ChEMBL ID for link
                            target_chembl_id = ''
                            target_link = ''
                            if has_target_id:
                                ids = target_records['Target_ChEMBL_ID'].dropna().unique()
                                if len(ids) > 0:
                                    target_chembl_id = str(ids[0])
                                    target_link = f"https://www.ebi.ac.uk/chembl/explore/target/{target_chembl_id}"

                            display_data.append({
                                'ChEMBL_ID': chembl_id,
                                'Molecule': mol_name[:20] if mol_name else '',
                                'Target': target_name if target_name else str(target_id)[:35],
                                'Target_Link': target_link,
                                'Avg_pActivity': f"{avg_activity:.2f}" if pd.notna(avg_activity) else 'N/A',
                                'IMP Score': imp_score_val,
                                'Confidence': imp_confidence,
                                'Records': len(target_records)
                            })
                    else:
                        # No targets found
                        display_data.append({
                            'ChEMBL_ID': chembl_id,
                            'Molecule': mol_name[:20] if mol_name else '',
                            'Target': 'N/A',
                            'Target_Link': '',
                            'Avg_pActivity': 'N/A',
                            'IMP Score': imp_score_val,
                            'Confidence': imp_confidence,
                            'Records': len(compound_records)
                        })
                else:
                    # No target column
                    display_data.append({
                        'ChEMBL_ID': chembl_id,
                        'Molecule': mol_name[:20] if mol_name else '',
                        'Target': 'N/A',
                        'Target_Link': '',
                        'Avg_pActivity': 'N/A',
                        'IMP Score': imp_score_val,
                        'Confidence': imp_confidence,
                        'Records': len(compound_records)
                    })

            imp_table = pd.DataFrame(display_data)
            st.dataframe(
                imp_table,
                column_config={
                    'ChEMBL_ID': st.column_config.TextColumn('ChEMBL ID', width='small'),
                    'Molecule': st.column_config.TextColumn('Molecule', width='small'),
                    'Target': st.column_config.TextColumn('Target Name', width='medium'),
                    'Target_Link': st.column_config.LinkColumn(
                        'Target ChEMBL ID',
                        display_text=r'https://www\.ebi\.ac\.uk/chembl/explore/target/(CHEMBL\d+)',
                        width='small'
                    ),
                    'Avg_pActivity': st.column_config.TextColumn('Avg pActivity', width='small'),
                    'IMP Score': st.column_config.TextColumn('IMP Score', width='small'),
                    'Confidence': st.column_config.TextColumn('Confidence', width='small'),
                    'Records': st.column_config.NumberColumn('Records', width='small'),
                },
                hide_index=True,
                height=min(500, len(imp_table) * 35 + 40)
            )

            st.caption("💡 **Note:** IMP candidates may still be valid if they have high PDB structural evidence. Cross-reference with the PDB Evidence tab.")
        else:
            st.success("✓ No IMP candidates detected - all compounds show normal activity patterns")


# =============================================================================
# VISUALIZATIONS TAB
# =============================================================================

def _render_visualizations_tab(data: Dict[str, Any]) -> None:
    """Interactive visualizations."""
    df = data.get('results')

    if df is None or df.empty:
        st.warning("No data available for visualization")
        return

    # Plot type selector
    plot_type = st.radio(
        "Select Plot",
        ["Activity Distribution", "Efficiency Scatter", "Efficiency by Compound", "Custom Plot"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    if plot_type == "Activity Distribution":
        _plot_activity_distribution(df)
    elif plot_type == "Efficiency Scatter":
        _plot_efficiency_scatter(df)
    elif plot_type == "Efficiency by Compound":
        _plot_efficiency_by_compound(df)
    elif plot_type == "Custom Plot":
        _plot_custom(df)


def _plot_activity_distribution(df: pd.DataFrame) -> None:
    """Activity distribution box plot with interactive legend and structure viewer."""
    if 'Activity_Type' not in df.columns or 'pActivity' not in df.columns:
        st.info("Activity columns not available")
        return

    plot_df = df.copy()

    # Build customdata for structure viewer
    customdata_cols = []
    if 'SMILES' in plot_df.columns:
        customdata_cols.append('SMILES')
        if 'Molecule_Name' in plot_df.columns:
            customdata_cols.append('Molecule_Name')
        if 'ChEMBL_ID' in plot_df.columns:
            customdata_cols.append('ChEMBL_ID')

    fig = px.box(
        plot_df, x='Activity_Type', y='pActivity',
        color='Activity_Type', points='all',  # Show all points for structure viewer clicks
        hover_data=['ChEMBL_ID', 'Molecule_Name'] if all(c in plot_df.columns for c in ['ChEMBL_ID', 'Molecule_Name']) else None,
        custom_data=customdata_cols if customdata_cols else None
    )
    fig.update_layout(
        title=dict(text='Bioactivity Distribution', subtitle=dict(text='pActivity = -log10(M) — higher = more potent')),
        template='plotly_white',
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
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        )
    )
    st.plotly_chart(fig, width='stretch', height=400, key="activity_dist_chart")
    st.caption("💡 **Click legend items** to show/hide activity types. Double-click to isolate.")

    # Embed structure viewer for click-to-view molecules
    if 'SMILES' in plot_df.columns:
        render_structure_viewer_hint()
        embed_structure_viewer(
            chart_id="activity_dist_chart",
            x_col='Activity_Type',
            y_col='pActivity',
            name_col='Molecule_Name' if 'Molecule_Name' in plot_df.columns else None
        )


def _plot_efficiency_scatter(df: pd.DataFrame) -> None:
    """Efficiency scatter plot with full controls and structure viewer."""
    # Get all columns for color/size options
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Filter out internal columns
    numeric_cols = [c for c in numeric_cols if not (c.startswith('Is_') and c.endswith('_Outlier'))]
    categorical_cols = [c for c in categorical_cols if c not in ['SMILES', 'Direct_Parent']]

    # All columns for color (categorical first, then numeric)
    all_color_cols = categorical_cols + numeric_cols

    # Row 1: Plot choice and Color by
    row1 = st.columns([1, 1, 1, 1])

    with row1[0]:
        plot_choice = st.selectbox("Plot", ["SEI vs BEI", "NSEI vs NBEI"], key="scatter_choice")

    with row1[1]:
        color_by = st.selectbox("Color by", ["None"] + all_color_cols, key="scatter_color")

    with row1[2]:
        size_by = st.selectbox("Size by", ["None"] + numeric_cols, key="scatter_size")

    with row1[3]:
        show_trendline = st.checkbox("Trendline", value=False, key="scatter_trendline")

    # Row 2: Additional options (show when relevant)
    row2 = st.columns([1, 1, 1, 1])

    # Check if color_by is numeric for gradient options
    is_numeric_color = color_by != "None" and color_by in numeric_cols

    with row2[0]:
        if is_numeric_color:
            color_scale = st.selectbox(
                "Color Scale",
                ["Viridis", "Plasma", "Inferno", "Turbo", "Blues", "Reds", "RdBu", "Spectral"],
                key="scatter_colorscale"
            )
        else:
            color_scale = "Viridis"

    with row2[1]:
        if is_numeric_color:
            reverse_scale = st.checkbox("Reverse Scale", value=False, key="scatter_reverse")
        else:
            reverse_scale = False

    with row2[2]:
        opacity = st.slider("Opacity", 0.3, 1.0, 0.7, key="scatter_opacity")

    with row2[3]:
        point_size = st.slider("Base Size", 5, 20, 10, key="scatter_pointsize")

    st.markdown("---")

    x_col, y_col = ('SEI', 'BEI') if plot_choice == "SEI vs BEI" else ('NSEI', 'NBEI')

    if x_col not in df.columns or y_col not in df.columns:
        st.warning(f"Columns {x_col} or {y_col} not available")
        return

    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    if plot_df.empty:
        st.warning("No valid data for plotting")
        return

    # Show R² and regression statistics at TOP (before chart) if trendline is enabled
    if show_trendline:
        try:
            x_vals = plot_df[x_col].values
            y_vals = plot_df[y_col].values
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_vals, y_vals)
            r_squared = r_value ** 2

            # Display stats at top in a compact row
            stats_cols = st.columns([1, 1, 1, 1, 2])
            with stats_cols[0]:
                st.metric("R²", f"{r_squared:.4f}")
            with stats_cols[1]:
                st.metric("Slope", f"{slope:.4f}")
            with stats_cols[2]:
                st.metric("Intercept", f"{intercept:.4f}")
            with stats_cols[3]:
                st.metric("p-value", f"{p_value:.2e}")
            with stats_cols[4]:
                # Show equation inline
                sign = "+" if intercept >= 0 else ""
                st.markdown("**Equation:**")
                st.caption(f"{y_col} = {slope:.4f} × {x_col} {sign} {intercept:.4f}")
        except Exception as e:
            st.caption(f"Could not calculate regression stats: {e}")

    # Build customdata for structure viewer (SMILES first, then name, then index)
    if 'SMILES' in plot_df.columns:
        customdata_cols = ['SMILES']
        if 'Molecule_Name' in plot_df.columns:
            customdata_cols.append('Molecule_Name')
        if 'ChEMBL_ID' in plot_df.columns:
            customdata_cols.append('ChEMBL_ID')
        plot_df['_row_idx'] = range(len(plot_df))
        customdata_cols.append('_row_idx')
    else:
        customdata_cols = None

    # Build scatter plot
    scatter_args = {
        'x': x_col,
        'y': y_col,
        'opacity': opacity,
        'hover_data': ['ChEMBL_ID', 'Molecule_Name'] if all(c in plot_df.columns for c in ['ChEMBL_ID', 'Molecule_Name']) else None,
    }

    # Add customdata for structure viewer
    if customdata_cols:
        scatter_args['custom_data'] = customdata_cols

    # Color handling
    if color_by != "None":
        scatter_args['color'] = color_by
        if is_numeric_color:
            scatter_args['color_continuous_scale'] = color_scale if not reverse_scale else f"{color_scale}_r"

    # Size handling
    if size_by != "None" and size_by in plot_df.columns:
        scatter_args['size'] = size_by
        scatter_args['size_max'] = point_size * 2

    # Trendline
    if show_trendline:
        scatter_args['trendline'] = "ols"

    fig = px.scatter(plot_df, **scatter_args)

    # Update marker size if no size_by
    if size_by == "None":
        fig.update_traces(marker=dict(size=point_size))

    # Layout
    fig.update_layout(
        template='plotly_white',
        height=520,
        showlegend=color_by != "None" and not is_numeric_color,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            title_text="",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        )
    )

    st.plotly_chart(fig, width='stretch', height=400, key="efficiency_scatter_chart")

    # Hints
    if color_by != "None" and not is_numeric_color:
        st.caption("💡 **Click legend items** to show/hide groups. Double-click to isolate.")

    # Embed structure viewer for click-to-view molecules
    if 'SMILES' in plot_df.columns:
        render_structure_viewer_hint()
        embed_structure_viewer(
            chart_id="efficiency_scatter_chart",
            x_col=x_col,
            y_col=y_col,
            name_col='Molecule_Name' if 'Molecule_Name' in plot_df.columns else None
        )


def _plot_efficiency_by_compound(df: pd.DataFrame) -> None:
    """Grouped efficiency boxplots."""
    col1, col2 = st.columns([1, 1])

    with col1:
        metric = st.selectbox("Metric", ['SEI', 'BEI', 'NSEI', 'NBEI'], key="box_metric")
    with col2:
        group_size = st.slider("Compounds per view", 3, 10, 5, key="group_size")

    if metric not in df.columns or 'ChEMBL_ID' not in df.columns:
        st.warning("Required columns not available")
        return

    unique_ids = df['ChEMBL_ID'].unique()
    num_groups = max(1, (len(unique_ids) + group_size - 1) // group_size)

    group_num = st.number_input("Group", 1, num_groups, 1, key="group_num")
    start = (group_num - 1) * group_size
    group_ids = unique_ids[start:start + group_size]

    group_df = df[df['ChEMBL_ID'].isin(group_ids)].dropna(subset=[metric])

    if not group_df.empty:
        fig = px.box(group_df, x='ChEMBL_ID', y=metric, color='ChEMBL_ID', points='all')
        fig.update_layout(height=450, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Group {group_num} of {num_groups} ({len(unique_ids)} total compounds)")


def _plot_custom(df: pd.DataFrame) -> None:
    """Fully customizable plot - users can select X, Y, color, plot type."""
    st.markdown("**🎨 Custom Visualization**")
    st.caption("Create your own plots by selecting axes and options")

    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Filter out internal columns
    numeric_cols = [c for c in numeric_cols if not c.startswith('Is_') or not c.endswith('_Outlier')]
    categorical_cols = [c for c in categorical_cols if c not in ['SMILES', 'Direct_Parent']]

    if not numeric_cols:
        st.warning("No numeric columns available for plotting")
        return

    # Control row 1: Plot type and axes
    ctrl_row1 = st.columns([1, 1, 1, 1])

    with ctrl_row1[0]:
        plot_type = st.selectbox(
            "Plot Type",
            ["Scatter", "Box", "Histogram", "Violin"],
            key="custom_plot_type"
        )

    with ctrl_row1[1]:
        # X axis - for histogram/box can also be categorical
        x_options = numeric_cols + (categorical_cols if plot_type in ["Box", "Violin"] else [])
        x_axis = st.selectbox("X Axis", x_options, key="custom_x")

    with ctrl_row1[2]:
        if plot_type in ["Scatter", "Box", "Violin"]:
            y_options = numeric_cols
            y_axis = st.selectbox("Y Axis", y_options, key="custom_y")
        else:
            y_axis = None

    with ctrl_row1[3]:
        color_options = ["None"] + categorical_cols + [c for c in numeric_cols if df[c].nunique() < 20]
        color_by = st.selectbox("Color By", color_options, key="custom_color")

    # Control row 2: Additional options
    ctrl_row2 = st.columns([1, 1, 1, 1])

    with ctrl_row2[0]:
        if plot_type == "Scatter":
            show_trendline = st.checkbox("Trendline", value=False, key="custom_trendline")
        else:
            show_trendline = False

    with ctrl_row2[1]:
        if plot_type in ["Scatter"]:
            point_size = st.slider("Point Size", 3, 15, 8, key="custom_size")
        else:
            point_size = 8

    with ctrl_row2[2]:
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7, key="custom_opacity")

    with ctrl_row2[3]:
        if plot_type == "Histogram":
            nbins = st.slider("Bins", 10, 50, 30, key="custom_bins")
        else:
            nbins = 30

    st.markdown("---")

    # Prepare data
    if y_axis:
        plot_df = df.dropna(subset=[x_axis, y_axis]).copy()
    else:
        plot_df = df.dropna(subset=[x_axis]).copy()

    if plot_df.empty:
        st.warning("No valid data for selected columns")
        return

    # Show R² at TOP (before chart) if trendline is enabled for scatter
    if plot_type == "Scatter" and show_trendline and y_axis:
        try:
            x_vals = plot_df[x_axis].values
            y_vals = plot_df[y_axis].values
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_vals, y_vals)
            r_squared = r_value ** 2

            # Display stats at top in a compact row
            stats_cols = st.columns([1, 1, 1, 1, 2])
            with stats_cols[0]:
                st.metric("R²", f"{r_squared:.4f}")
            with stats_cols[1]:
                st.metric("Slope", f"{slope:.4f}")
            with stats_cols[2]:
                st.metric("Intercept", f"{intercept:.4f}")
            with stats_cols[3]:
                st.metric("p-value", f"{p_value:.2e}")
            with stats_cols[4]:
                sign = "+" if intercept >= 0 else ""
                st.markdown("**Equation:**")
                st.caption(f"{y_axis} = {slope:.4f} × {x_axis} {sign} {intercept:.4f}")
        except Exception as e:
            st.caption(f"Could not calculate regression stats: {e}")

    # Build customdata for structure viewer (for scatter plots)
    customdata_cols = None
    if plot_type == "Scatter" and 'SMILES' in plot_df.columns:
        customdata_cols = ['SMILES']
        if 'Molecule_Name' in plot_df.columns:
            customdata_cols.append('Molecule_Name')
        if 'ChEMBL_ID' in plot_df.columns:
            customdata_cols.append('ChEMBL_ID')

    # Create plot based on type
    try:
        if plot_type == "Scatter":
            fig = px.scatter(
                plot_df, x=x_axis, y=y_axis,
                color=color_by if color_by != "None" else None,
                hover_data=['ChEMBL_ID', 'Molecule_Name'] if all(c in plot_df.columns for c in ['ChEMBL_ID', 'Molecule_Name']) else None,
                opacity=opacity,
                trendline="ols" if show_trendline else None,
                custom_data=customdata_cols
            )
            fig.update_traces(marker=dict(size=point_size))

        elif plot_type == "Box":
            # Build customdata for box plots too
            box_customdata = None
            if 'SMILES' in plot_df.columns:
                box_customdata = ['SMILES']
                if 'Molecule_Name' in plot_df.columns:
                    box_customdata.append('Molecule_Name')
                if 'ChEMBL_ID' in plot_df.columns:
                    box_customdata.append('ChEMBL_ID')

            fig = px.box(
                plot_df, x=x_axis, y=y_axis,
                color=color_by if color_by != "None" else None,
                points="all",  # Show all points for structure viewer clicks
                custom_data=box_customdata
            )

        elif plot_type == "Violin":
            # Build customdata for violin plots too
            violin_customdata = None
            if 'SMILES' in plot_df.columns:
                violin_customdata = ['SMILES']
                if 'Molecule_Name' in plot_df.columns:
                    violin_customdata.append('Molecule_Name')
                if 'ChEMBL_ID' in plot_df.columns:
                    violin_customdata.append('ChEMBL_ID')

            fig = px.violin(
                plot_df, x=x_axis, y=y_axis,
                color=color_by if color_by != "None" else None,
                box=True, points="all",  # Show all points for structure viewer clicks
                custom_data=violin_customdata
            )

        elif plot_type == "Histogram":
            fig = px.histogram(
                plot_df, x=x_axis,
                color=color_by if color_by != "None" else None,
                nbins=nbins, opacity=opacity
            )

        # Common layout updates
        fig.update_layout(
            template='plotly_white',
            height=550,
            showlegend=color_by != "None",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                title_text="",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            )
        )

        st.plotly_chart(fig, width='stretch', height=400, key="custom_plot_chart")

        if color_by != "None":
            st.caption("💡 **Click legend items** to show/hide groups. Double-click to isolate.")

        # Embed structure viewer for click-to-view molecules (for scatter, box, violin)
        if plot_type in ["Scatter", "Box", "Violin"] and 'SMILES' in plot_df.columns:
            render_structure_viewer_hint()
            embed_structure_viewer(
                chart_id="custom_plot_chart",
                x_col=x_axis,
                y_col=y_axis if y_axis else x_axis,
                name_col='Molecule_Name' if 'Molecule_Name' in plot_df.columns else None
            )

    except Exception as e:
        st.error(f"Error creating plot: {e}")


# =============================================================================
# STRUCTURES TAB - Molecule Viewer
# =============================================================================

def _render_structures_tab(data: Dict[str, Any]) -> None:
    """Molecular structures viewer (2D/3D)."""
    df = data.get('results')
    _render_molecule_viewer(df)


def _render_molecule_viewer(df: pd.DataFrame) -> None:
    """2D/3D molecule viewer."""
    if df is None or 'SMILES' not in df.columns:
        st.warning("No SMILES data available")
        return

    # Get unique molecules
    id_col = 'ChEMBL_ID' if 'ChEMBL_ID' in df.columns else None
    name_col = 'Molecule_Name' if 'Molecule_Name' in df.columns else None

    cols = ['SMILES']
    if id_col:
        cols.insert(0, id_col)
    if name_col:
        cols.append(name_col)

    unique_mols = df[cols].drop_duplicates().reset_index(drop=True)

    # Grid view for molecule selection (show first 12)
    st.markdown(f"**{len(unique_mols)} unique molecules**")

    # Molecule selector
    if id_col and name_col:
        options = [f"{row[id_col]} - {row[name_col]}" for _, row in unique_mols.iterrows()]
    elif id_col:
        options = list(unique_mols[id_col])
    else:
        options = [f"Mol {i+1}" for i in range(len(unique_mols))]

    selected = st.selectbox("Select", options, key="mol_select", label_visibility="collapsed")
    idx = options.index(selected)
    row = unique_mols.iloc[idx]

    # Display
    col1, col2 = st.columns([1, 1])

    with col1:
        render_2d_structure(row['SMILES'], size=(350, 280))

    with col2:
        if id_col:
            st.markdown(f"**{row[id_col]}**")
        if name_col and row[name_col] != row.get(id_col, ''):
            st.caption(row[name_col])

        st.code(row['SMILES'], language=None)

        # Activity summary for this molecule
        if id_col:
            mol_data = df[df[id_col] == row[id_col]]
            if 'Activity_Type' in mol_data.columns:
                st.markdown(f"**Activities:** {mol_data['Activity_Type'].nunique()} types")
            if 'pActivity' in mol_data.columns:
                st.markdown(f"**pActivity:** {mol_data['pActivity'].min():.1f} - {mol_data['pActivity'].max():.1f}")
            if 'IMP_Final_Score' in mol_data.columns:
                avg = mol_data['IMP_Final_Score'].mean()
                st.markdown(f"**IMP Score:** {avg:.3f}")

    # 3D Viewer
    with st.expander("🧬 Generate 3D Structure"):
        if st.button("Render 3D", key="render_3d"):
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem

                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol:
                    mol_3d = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol_3d)
                    pdb_block = Chem.MolToPDBBlock(mol_3d)

                    html = f"""
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
                    <div id="viewer" style="width:100%;height:350px;background:#f8f9fa;border-radius:8px;"></div>
                    <script>
                        let viewer = $3Dmol.createViewer(document.getElementById("viewer"), {{backgroundColor: "white"}});
                        viewer.addModel(`{pdb_block}`, "pdb");
                        viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{radius: 0.3}}}});
                        viewer.zoomTo();
                        viewer.render();
                    </script>
                    """
                    st.components.v1.html(html, height=370)
                else:
                    st.error("Could not parse SMILES")
            except Exception as e:
                st.error(f"Error: {e}")


def _render_pdb_evidence(
    compound_name: str,
    df: pd.DataFrame,
    entry_id: str = None,
    storage_path: str = None
) -> None:
    """PDB structural evidence from DataFrame columns."""
    if df is None:
        st.info("No data available")
        return

    # Check for PDB columns in main DataFrame
    pdb_cols = ['PDB_Score', 'PDB_Num_Structures', 'PDB_IDs', 'PDB_Best_Resolution',
                'PDB_High_Quality', 'PDB_Medium_Quality', 'PDB_Poor_Quality']
    has_pdb = any(col in df.columns for col in pdb_cols)

    if not has_pdb:
        st.info("No PDB structural evidence available. Re-run analysis with PDB enabled.")
        st.caption("PDB scoring queries RCSB PDB for experimental crystal structures of similar compounds.")
        return

    # Get unique compounds with PDB data
    unique_df = df.drop_duplicates('ChEMBL_ID') if 'ChEMBL_ID' in df.columns else df

    # Try to load detailed PDB summary file FIRST to get accurate counts
    pdb_summary_df = None
    try:
        safe_name = sanitize_compound_name(compound_name)
        for filename in ["pdb_summary.csv", f"{safe_name}_pdb_summary.csv", f"{safe_name}_pdb_details.csv"]:
            pdb_summary_df = smart_load_dataframe(
                filename,
                entry_id=entry_id,
                storage_path=storage_path
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
        if 'Quality' in pdb_summary_df.columns:
            high_q = int((pdb_summary_df['Quality'] == '***').sum())
            med_q = int((pdb_summary_df['Quality'] == '**').sum())
            poor_q = int((pdb_summary_df['Quality'] == '*').sum())
        else:
            # Fallback to resolution-based counting
            if 'Resolution' in pdb_summary_df.columns:
                pdb_summary_df['_res'] = pd.to_numeric(pdb_summary_df['Resolution'], errors='coerce')
                high_q = int((pdb_summary_df['_res'] < 2.0).sum())
                med_q = int(((pdb_summary_df['_res'] >= 2.0) & (pdb_summary_df['_res'] <= 3.0)).sum())
                poor_q = int((pdb_summary_df['_res'] > 3.0).sum())
            else:
                high_q = med_q = poor_q = 0
    else:
        # Fallback to summing from unique compounds (less accurate)
        total_structs = int(unique_df['PDB_Num_Structures'].sum()) if 'PDB_Num_Structures' in unique_df.columns else 0
        high_q = int(unique_df['PDB_High_Quality'].sum()) if 'PDB_High_Quality' in unique_df.columns else 0
        med_q = int(unique_df['PDB_Medium_Quality'].sum()) if 'PDB_Medium_Quality' in unique_df.columns else 0
        poor_q = int(unique_df['PDB_Poor_Quality'].sum()) if 'PDB_Poor_Quality' in unique_df.columns else 0

    avg_score = unique_df['PDB_Score'].mean() if 'PDB_Score' in unique_df.columns else None
    compounds_with_pdb = len(unique_df[unique_df['PDB_Num_Structures'] > 0]) if 'PDB_Num_Structures' in unique_df.columns else 0
    pct_with_pdb = (compounds_with_pdb / len(unique_df) * 100) if len(unique_df) > 0 else 0

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Average PDB Score", f"{avg_score:.3f}" if pd.notna(avg_score) else "N/A")

    with col2:
        st.metric("Total Structures", total_structs)

    with col3:
        st.metric("High Quality (⭐⭐⭐)", high_q)

    with col4:
        st.metric("% with PDB Data", f"{pct_with_pdb:.1f}%")

    st.caption(f"📊 Summary across {len(unique_df)} unique compounds")

    st.markdown("---")

    # Structure Quality Distribution table
    st.markdown("**Structure Quality Distribution:**")
    quality_data = []
    total_q = high_q + med_q + poor_q
    if total_q > 0:
        quality_data.append({
            'Quality Tier': '⭐⭐⭐ High (< 2.0 Å)',
            'Count': high_q,
            'Avg %': f"{high_q/total_q*100:.1f}%"
        })
        quality_data.append({
            'Quality Tier': '⭐⭐ Medium (2.0-3.0 Å)',
            'Count': med_q,
            'Avg %': f"{med_q/total_q*100:.1f}%"
        })
        quality_data.append({
            'Quality Tier': '⭐ Poor (> 3.0 Å)',
            'Count': poor_q,
            'Avg %': f"{poor_q/total_q*100:.1f}%"
        })
        st.dataframe(pd.DataFrame(quality_data), width='stretch', hide_index=True, height=150)
    else:
        st.caption("No quality distribution data available")

    st.info("💡 **Tip:** Higher PDB scores indicate more experimental validation. Compounds with ⭐⭐⭐ structures (< 2.0 Å resolution) have the strongest structural evidence.")

    st.markdown("---")

    # pdb_summary_df was already loaded earlier for accurate counts
    # If we have detailed PDB summary, display it in the exact format
    if pdb_summary_df is not None and not pdb_summary_df.empty:
        st.markdown("**Detailed PDB Structures:**")
        st.caption("*Sorted by quality (⭐⭐⭐ first) and resolution (best first)*")

        # Build display table with clickable links
        display_data = []
        for _, row in pdb_summary_df.iterrows():
            pdb_id = str(row.get('PDB_ID', ''))
            chembl_id = str(row.get('ChEMBL_ID', ''))
            mol_name = row.get('Molecule_Name', '')
            if pd.isna(mol_name):
                mol_name = ''
            title = row.get('Title', '')
            if pd.isna(title):
                title = ''
            resolution = row.get('Resolution', '')
            quality = row.get('Quality', '')
            if pd.isna(quality):
                quality = ''
            exp_method = row.get('Experimental_Method', '')
            if pd.isna(exp_method):
                exp_method = ''
            uniprot = row.get('UniProt_IDs', '')
            if pd.isna(uniprot):
                uniprot = ''

            # Get first UniProt ID for link
            uniprot_list = [u.strip() for u in str(uniprot).split(',') if u.strip() and u.strip() != 'N/A']
            uniprot_link = f"https://www.uniprot.org/uniprotkb/{uniprot_list[0]}" if uniprot_list else ''

            # Parse resolution for sorting
            try:
                res_val = float(resolution) if resolution and resolution != 'N/A' and str(resolution) != 'nan' else 999.0
            except (ValueError, TypeError):
                res_val = 999.0

            display_data.append({
                'PDB_Link': f"https://www.rcsb.org/structure/{pdb_id}",
                'ChEMBL_ID': chembl_id,
                'Molecule_Name': str(mol_name) if mol_name else '',
                'Title': str(title)[:70] + '...' if len(str(title)) > 70 else str(title),
                'Resolution': f"{float(resolution):.2f}" if resolution and resolution != 'N/A' and str(resolution) != 'nan' else 'N/A',
                'Resolution_Sort': res_val,
                'Quality': quality,
                'Experimental_Method': exp_method,
                'UniProt_IDs': uniprot_link
            })

        pdb_table = pd.DataFrame(display_data)

        # Sort by quality (*** first) then by resolution (lowest first)
        quality_order = {'***': 1, '**': 2, '*': 3, '': 4, 'N/A': 4}
        pdb_table['Quality_Sort'] = pdb_table['Quality'].map(lambda x: quality_order.get(x, 4))
        pdb_table = pdb_table.sort_values(['Quality_Sort', 'Resolution_Sort']).drop(columns=['Quality_Sort', 'Resolution_Sort'])

        # Display with column config for clickable links
        st.dataframe(
            pdb_table,
            width='stretch',
            hide_index=True,
            height=400,
            column_config={
                "PDB_Link": st.column_config.LinkColumn(
                    "PDB_Link",
                    display_text=r"https://www\.rcsb\.org/structure/(.+)",
                    width="small"
                ),
                "ChEMBL_ID": st.column_config.TextColumn("ChEMBL_ID", width="small"),
                "Molecule_Name": st.column_config.TextColumn("Molecule_Name", width="medium"),
                "Title": st.column_config.TextColumn("Title", width="large"),
                "Resolution": st.column_config.TextColumn("Resolution", width="small"),
                "Quality": st.column_config.TextColumn("Quality", width="small"),
                "Experimental_Method": st.column_config.TextColumn("Experimental_Method", width="medium"),
                "UniProt_IDs": st.column_config.LinkColumn(
                    "UniProt_IDs",
                    display_text=r"https://www\.uniprot\.org/uniprotkb/(.+)",
                    width="small"
                )
            }
        )

        st.caption(f"📋 {len(pdb_table)} total PDB structures sorted by quality (⭐⭐⭐ → ⭐⭐ → ⭐) and resolution (best first). Click PDB_Link to view structure at RCSB PDB. 📜 Scroll to see all.")

    else:
        # Fallback: PDB summary file not found - show basic info from DataFrame
        # Note: For newly processed compounds, pdb_summary.csv should exist
        if 'PDB_IDs' in unique_df.columns:
            # Collect all PDB IDs with associated ChEMBL data
            all_pdb_ids = []
            pdb_compound_map = {}  # Map PDB ID -> list of (chembl_id, mol_name)

            for _, row in unique_df.iterrows():
                pdb_str = row.get('PDB_IDs', '')
                chembl_id = row.get('ChEMBL_ID', 'Unknown')
                mol_name = row.get('Molecule_Name', '')
                if pd.isna(mol_name) or not isinstance(mol_name, str):
                    mol_name = ''

                if pd.notna(pdb_str) and pdb_str:
                    pdb_list = [p.strip() for p in str(pdb_str).split(',') if p.strip()]
                    for pdb_id in pdb_list:
                        pdb_id_upper = pdb_id.upper()
                        all_pdb_ids.append(pdb_id_upper)
                        if pdb_id_upper not in pdb_compound_map:
                            pdb_compound_map[pdb_id_upper] = []
                        pdb_compound_map[pdb_id_upper].append((chembl_id, mol_name))

            unique_pdb_ids = list(set(all_pdb_ids))

            if unique_pdb_ids:
                st.markdown(f"**{len(unique_pdb_ids)} Unique PDB Structures**")
                st.caption("⚠️ Detailed PDB info not available. Re-process the compound to fetch PDB details.")
                st.caption("Click on PDB ID to view structure on RCSB PDB.")

                # Show basic info from DataFrame without API calls
                pdb_data = []
                for pdb_id in sorted(unique_pdb_ids):
                    compounds = pdb_compound_map.get(pdb_id, [])
                    chembl_ids = list(set([c[0] for c in compounds if c[0]]))
                    mol_names = list(set([c[1] for c in compounds if c[1]]))
                    pdb_data.append({
                        'PDB_Link': f"https://www.rcsb.org/structure/{pdb_id}",
                        'ChEMBL_IDs': ', '.join(chembl_ids) if chembl_ids else 'N/A',
                        'Molecule_Name': ', '.join(mol_names[:3]) + (f' (+{len(mol_names)-3})' if len(mol_names) > 3 else '') if mol_names else 'N/A'
                    })

                pdb_table = pd.DataFrame(pdb_data)
                st.dataframe(
                    pdb_table,
                    width='stretch',
                    hide_index=True,
                    height=400,
                    column_config={
                        "PDB_Link": st.column_config.LinkColumn(
                            "PDB ID",
                            display_text=r"https://www\.rcsb\.org/structure/(.+)",
                            width=80
                        ),
                        "ChEMBL_IDs": st.column_config.TextColumn("ChEMBL IDs", width=200),
                        "Molecule_Name": st.column_config.TextColumn("Molecule Names", width=250)
                    }
                )
            else:
                st.info("No PDB IDs found in the data")

    # PDB Score distribution
    if 'PDB_Score' in unique_df.columns:
        st.markdown("---")
        st.markdown("**PDB Score Distribution**")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(unique_df, x='PDB_Score', nbins=20)
            fig.update_layout(height=250, margin=dict(t=20, b=30))
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown("**Quality Breakdown**")
            if all(c in unique_df.columns for c in ['PDB_High_Quality', 'PDB_Medium_Quality', 'PDB_Poor_Quality']):
                high = unique_df['PDB_High_Quality'].sum()
                med = unique_df['PDB_Medium_Quality'].sum()
                poor = unique_df['PDB_Poor_Quality'].sum()
                total = high + med + poor
                if total > 0:
                    st.caption(f"⭐⭐⭐ High: {int(high)} ({high/total*100:.0f}%)")
                    st.caption(f"⭐⭐ Medium: {int(med)} ({med/total*100:.0f}%)")
                    st.caption(f"⭐ Poor: {int(poor)} ({poor/total*100:.0f}%)")


# =============================================================================
# DATA TAB
# =============================================================================

def _render_data_tab(data: Dict[str, Any]) -> None:
    """Data tables with downloads."""
    df = data.get('results')
    compound_name = data.get('compound_name', 'compound')

    if df is None or df.empty:
        st.warning("No data available")
        return

    # View selector
    view = st.radio(
        "View",
        ["Core Analysis", "Interpretation", "Full Data"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    if view == "Core Analysis":
        # Include SMILES for structure data in CSV downloads
        # Columns: Identifiers, Activity, Target, Efficiency Metrics, Properties
        cols = [
            'ChEMBL_ID', 'Molecule_Name', 'SMILES',
            'Activity_Type', 'Activity_nM', 'pActivity',
            'Target_ChEMBL_ID', 'Target_Name',
            'SEI', 'BEI', 'NSEI', 'NBEI',
            'Molecular_Weight', 'LogP', 'TPSA', 'QED',
            'HBA', 'HBD', 'Heavy_Atoms',
            'PSAoMW', '10xPSA_MW', 'NPOLoNHA'
        ]
        cols = [c for c in cols if c in df.columns]
        display_df = df[cols]
        st.dataframe(display_df, width='stretch', height=450, hide_index=True)

        # Deferred download - generates CSV on-demand (non-blocking)
        st.download_button(
            "📥 Download",
            data=lambda df=display_df: df.to_csv(index=False),
            file_name=f"{compound_name}_analysis.csv",
            mime="text/csv"
        )

    elif view == "Interpretation":
        # Include SMILES for structure data in CSV downloads
        cols = ['ChEMBL_ID', 'Molecule_Name', 'SMILES', 'IMP_Final_Score', 'IMP_Classification',
                'Is_IMP_Candidate', 'IMP_Confidence', 'PDB_Score', 'Efficiency_Score']
        cols = [c for c in cols if c in df.columns]

        if cols:
            display_df = df[cols].drop_duplicates()
            st.dataframe(display_df, width='stretch', height=450, hide_index=True)

            # Deferred download - generates CSV on-demand (non-blocking)
            st.download_button(
                "📥 Download",
                data=lambda df=display_df: df.to_csv(index=False),
                file_name=f"{compound_name}_interpretation.csv",
                mime="text/csv"
            )
        else:
            st.info("No interpretation columns available")

    else:  # Full Data
        # Remove internal columns
        hide = [c for c in df.columns if c.startswith('Is_') and c.endswith('_Outlier')]
        display_df = df[[c for c in df.columns if c not in hide]]

        st.caption(f"{len(display_df)} rows × {len(display_df.columns)} columns")
        st.dataframe(display_df, width='stretch', height=450, hide_index=True)

        # Deferred download - generates CSV on-demand (non-blocking)
        st.download_button(
            "📥 Download Full",
            data=lambda df=display_df: df.to_csv(index=False),
            file_name=f"{compound_name}_complete.csv",
            mime="text/csv"
        )


# =============================================================================
# DATA LOADING & DELETE
# =============================================================================

def _render_drug_indications(data: Dict[str, Any]) -> None:
    """
    Render drug indications tab with clickable links to MESH, EFO, and Clinical Trials.

    Shows disease associations and clinical trial phases for similar compounds.
    """
    indications_df = data.get('indications')

    st.markdown("### 💊 Drug Indications")
    st.caption("Disease associations and clinical trial phases from ChEMBL")

    if indications_df is None or indications_df.empty:
        st.info("No drug indications found for these compounds. This is common for research compounds not yet in clinical trials.")
        st.markdown("""
        **Note:** Drug indications are only available for compounds that:
        - Have entered clinical trials
        - Are approved drugs
        - Have documented therapeutic uses in ChEMBL
        """)
        return

    # Summary metrics
    total_indications = len(indications_df)
    unique_compounds = indications_df['ChEMBL_ID'].nunique()
    unique_diseases = indications_df['MESH_Heading'].nunique() if 'MESH_Heading' in indications_df.columns else 0

    # Get max phase
    max_phase = 0
    if 'Max_Phase' in indications_df.columns:
        max_phase = indications_df['Max_Phase'].max()

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

    st.markdown("---")

    # Search/filter
    search_term = st.text_input("🔍 Search diseases", placeholder="Type to filter by disease name...")

    # Filter DataFrame
    display_df = indications_df.copy()
    if search_term:
        mask = (
            display_df['MESH_Heading'].str.contains(search_term, case=False, na=False) |
            display_df['EFO_Term'].str.contains(search_term, case=False, na=False)
        )
        display_df = display_df[mask]

    if display_df.empty:
        st.warning(f"No indications found matching '{search_term}'")
        return

    st.markdown(f"**Showing {len(display_df)} indications:**")

    # Build display DataFrame with URLs for clickable links (like PDB table)
    # Pre-compile NCT ID pattern for validation (NCT followed by digits)
    nct_pattern = re.compile(r'^NCT\d+$')
    table_data = []
    for _, row in display_df.iterrows():
        mesh_id = str(row.get('MESH_ID', '')) if pd.notna(row.get('MESH_ID')) else ''
        mesh_heading = str(row.get('MESH_Heading', '')) if pd.notna(row.get('MESH_Heading')) else ''
        efo_id = str(row.get('EFO_ID', '')) if pd.notna(row.get('EFO_ID')) else ''
        max_phase_val = row.get('Max_Phase', 0)
        if pd.isna(max_phase_val):
            max_phase_val = 0
        chembl_id = str(row.get('ChEMBL_ID', '')) if pd.notna(row.get('ChEMBL_ID')) else ''

        # Phase badge
        phase_badge = get_phase_badge(max_phase_val)

        # Build URLs directly (like PDB table does)
        mesh_url = f"https://id.nlm.nih.gov/mesh/{mesh_id}.html" if mesh_id else ''
        efo_url = f"https://www.ebi.ac.uk/ols4/ontologies/efo/classes/http%253A%252F%252Fwww.ebi.ac.uk%252Fefo%252F{efo_id.replace(':', '_')}" if efo_id else ''
        chembl_url = f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/" if chembl_id else ''

        # ClinicalTrials.gov - use NCT ID(s) if available, fallback to disease search
        # Embed count/marker in URL fragment for display extraction via regex
        nct_ids_raw = str(row.get('Clinical_Trials_IDs', '')) if pd.notna(row.get('Clinical_Trials_IDs')) else ''
        ct_url = ''

        if nct_ids_raw:
            # Handle multiple NCT IDs (may be comma or space separated)
            nct_ids = [
                nct.strip() for nct in nct_ids_raw.replace(',', ' ').split()
                if nct.strip() and nct_pattern.match(nct.strip())
            ]
            if nct_ids:
                nct_search = '%20'.join(nct_ids)
                # Add count as fragment: #[N] - will display as "[N]" via regex
                ct_url = f"https://clinicaltrials.gov/search?term={nct_search}#[{len(nct_ids)}]"

        # Fallback to disease search if no valid NCT IDs
        if not ct_url and mesh_heading:
            # Properly URL-encode disease name (handles special chars like & ' ( ) etc.)
            ct_search = quote_plus(mesh_heading)
            ct_url = f"https://clinicaltrials.gov/search?cond={ct_search}#[🔬]"

        table_data.append({
            'MESH_Link': mesh_url,
            'Disease': mesh_heading[:60] + ('...' if len(mesh_heading) > 60 else '') if mesh_heading else 'N/A',
            'EFO_Link': efo_url,
            'Phase': phase_badge,
            'ChEMBL_Link': chembl_url,
            'ClinicalTrials': ct_url,
        })

    # Display as scrollable dataframe (matching PDB table style)
    if table_data:
        df_display = pd.DataFrame(table_data)

        st.dataframe(
            df_display,
            width='stretch',
            hide_index=True,
            height=400,
            column_order=["MESH_Link", "Disease", "EFO_Link", "Phase", "ChEMBL_Link", "ClinicalTrials"],
            column_config={
                "MESH_Link": st.column_config.LinkColumn(
                    "MESH ID",
                    help="Click to view MESH entry",
                    display_text=r"https://id\.nlm\.nih\.gov/mesh/(.+)\.html",
                    width="small"
                ),
                "Disease": st.column_config.TextColumn("Disease", width="large"),
                "EFO_Link": st.column_config.LinkColumn(
                    "EFO ID",
                    help="Click to view EFO ontology entry",
                    display_text=r".*efo%252F(.+)",
                    width="small"
                ),
                "Phase": st.column_config.TextColumn("Phase", width="small"),
                "ChEMBL_Link": st.column_config.LinkColumn(
                    "Compound",
                    help="Click to view ChEMBL entry",
                    display_text=r"https://www\.ebi\.ac\.uk/chembl/compound_report_card/(.+)/",
                    width="small"
                ),
                "ClinicalTrials": st.column_config.LinkColumn(
                    "Trials",
                    help="Click to search ClinicalTrials.gov ([N] = N linked trials, [🔬] = disease search)",
                    display_text=r".*#(\[.+\])$",
                    width="small"
                ),
            }
        )

        st.caption(f"📋 {len(table_data)} indications. Click links to view MESH, EFO, ChEMBL entries or search ClinicalTrials.gov. 📜 Scroll to see all.")

    # Phase distribution chart
    if 'Max_Phase' in indications_df.columns and len(indications_df) > 1:
        st.markdown("---")
        st.markdown("#### Phase Distribution")

        phase_counts = indications_df['Max_Phase'].value_counts().sort_index()
        phase_labels = {
            4.0: 'Approved (4)',
            3.0: 'Phase 3',
            2.0: 'Phase 2',
            1.0: 'Phase 1',
            0.5: 'Early Phase 1',
            -1.0: 'Unknown'
        }

        fig = px.bar(
            x=[phase_labels.get(p, f'Phase {p}') for p in phase_counts.index],
            y=phase_counts.values,
            color=[phase_labels.get(p, f'Phase {p}') for p in phase_counts.index],
            color_discrete_map={
                'Approved (4)': '#28a745',
                'Phase 3': '#007bff',
                'Phase 2': '#ffc107',
                'Phase 1': '#fd7e14',
                'Early Phase 1': '#6c757d',
                'Unknown': '#343a40',
            },
            labels={'x': 'Clinical Phase', 'y': 'Number of Indications'},
        )
        fig.update_layout(
            title=dict(text='Clinical Trial Phases', subtitle=dict(text='Drug indication progression status')),
            showlegend=False,
            height=320,
            margin=dict(t=55, b=40),
        )
        st.plotly_chart(fig, width='stretch')


def _load_compound_data(
    compound_name: str = None,
    entry_id: str = None,
    storage_path: str = None
) -> Optional[Dict[str, Any]]:
    """Load compound data from storage.

    Uses smart loaders that prioritize storage_path (from database), then entry_id.
    Only UUID-based storage paths are supported.

    Args:
        compound_name: Display name of the compound (for logging only)
        entry_id: UUID entry_id for storage lookup
        storage_path: Full Azure storage path from database (most reliable)
    """
    try:
        # Use smart loader with storage_path from database (UUID-based)
        summary = smart_load_summary(
            entry_id=entry_id,
            storage_path=storage_path
        )
        if summary is None:
            logger.warning(f"Could not load summary for {compound_name} (entry_id={entry_id}, storage_path={storage_path})")
            return None

        # Load results DataFrame using smart loader
        df = smart_load_dataframe(
            "similar_compounds.csv",
            entry_id=entry_id,
            storage_path=storage_path
        )

        if df is None:
            # Try alternate filename format
            safe_name = sanitize_compound_name(compound_name or entry_id or "unknown")
            df = smart_load_dataframe(
                f"{safe_name}_complete_results.csv",
                entry_id=entry_id,
                storage_path=storage_path
            )

        # Load drug indications (separate file)
        indications_df = smart_load_dataframe(
            "drug_indications.csv",
            entry_id=entry_id,
            storage_path=storage_path
        )

        # Get display name from summary (compound_name is in summary.json)
        display_name = summary.get('compound_name', compound_name or entry_id)

        # Compute InChI and InChIKey once for reuse across all tabs
        smiles = summary.get('smiles', summary.get('query_smiles', ''))
        inchi = None
        inchikey = None
        if smiles:
            try:
                from rdkit import Chem
                from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    inchi = MolToInchi(mol)
                    inchikey = MolToInchiKey(mol)
            except Exception:
                pass

        return {
            'compound_name': display_name,
            'author_name': summary.get('author_name', 'N/A'),
            'entry_id': summary.get('entry_id', entry_id),
            'storage_path': storage_path,
            'smiles': smiles,
            'inchi': inchi,
            'inchikey': inchikey,
            'similar_count': summary.get('similar_count', summary.get('total_compounds', 0)),
            'has_imp_warning': summary.get('has_imp_candidates', False),
            'summary': summary,
            'results': df,
            'indications': indications_df,
        }

    except Exception as e:
        logger.error(f"Error loading compound data: {e}")
        return None


def _show_delete_confirmation(compound_name: str, entry_id: Optional[str] = None) -> None:
    """Delete confirmation dialog.

    Calls backend API to delete compound from database, Azure storage, and local cache.

    Args:
        compound_name: Display name of the compound
        entry_id: UUID of the compound (required for proper deletion)
    """
    st.warning(f"Delete **{compound_name}**?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", width='stretch'):
            SessionState.set('show_delete_confirmation', False)
            st.rerun()
    with col2:
        if st.button("Delete", type="primary", width='stretch'):
            try:
                if not entry_id:
                    st.error("Cannot delete: compound entry_id not found")
                    return

                # Call backend API to delete (handles DB, Azure, and local cache)
                api_client = get_api_client()
                result = api_client.delete_compound(entry_id)

                if result.success:
                    # Also clear frontend cache (uses entry_id UUID)
                    if entry_id:
                        delete_from_cache(entry_id)

                    # Show toast notification (persists across rerun)
                    st.toast(f"✓ Deleted '{compound_name}' successfully", icon="✅")
                    SessionState.set('show_delete_confirmation', False)
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

def _get_imp_color(score: float) -> str:
    """Return color based on IMP score - Higher IMP = MORE DANGEROUS (RED)."""
    if pd.isna(score):
        return "#6c757d"  # Gray for N/A
    if score >= 0.9:
        return "#721c24"  # Dark Red - Exceptional IMP
    elif score >= 0.7:
        return "#dc3545"  # Red - Strong IMP
    elif score >= 0.5:
        return "#fd7e14"  # Orange - Moderate IMP
    elif score >= 0.3:
        return "#28a745"  # Light Green - Weak IMP
    else:
        return "#155724"  # Green - Not IMP


def _get_imp_bg_color(score: float) -> str:
    """Return background color (lighter version) for IMP badges."""
    if pd.isna(score):
        return "#e9ecef"  # Light gray for N/A
    if score >= 0.9:
        return "#f8d7da"  # Light red bg
    elif score >= 0.7:
        return "#f8d7da"  # Light red bg
    elif score >= 0.5:
        return "#fff3cd"  # Light orange bg
    elif score >= 0.3:
        return "#d4edda"  # Light green bg
    else:
        return "#d4edda"  # Light green bg


def _get_imp_interpretation(score: float) -> dict:
    """Return IMP interpretation - Strong IMP = FALSE POSITIVE risk."""
    if pd.isna(score):
        return {
            "label": "N/A",
            "risk": "Score not available",
            "action": "Check data quality",
            "priority": None
        }
    if score >= 0.9:
        return {
            "label": "Exceptional IMP",
            "risk": "VERY HIGH false positive risk",
            "action": "Immediate validation required - likely assay artifact",
            "priority": 1
        }
    elif score >= 0.7:
        return {
            "label": "Strong IMP",
            "risk": "HIGH false positive risk",
            "action": "Validate within 1 month with orthogonal assay",
            "priority": 2
        }
    elif score >= 0.5:
        return {
            "label": "Moderate IMP",
            "risk": "Moderate false positive risk",
            "action": "Monitor and gather additional validation data",
            "priority": 3
        }
    elif score >= 0.3:
        return {
            "label": "Weak IMP",
            "risk": "Lower false positive risk",
            "action": "More likely genuine - standard follow-up",
            "priority": 4
        }
    else:
        return {
            "label": "Not IMP",
            "risk": "Likely genuine activity",
            "action": "Proceed with development",
            "priority": None
        }


def _render_report_tab(data: Dict[str, Any]) -> None:
    """Render the Report tab with comprehensive analysis and export."""
    df = data.get('results')
    if df is None or df.empty:
        st.warning("No data available for report generation.")
        return

    compound_name = data.get('compound_name', 'Unknown')
    smiles = data.get('smiles', '')
    summary = data.get('summary', {})

    # Calculate scores - use MAX (best scoring compound) to match Overview tab behavior
    # The Overview shows "Best scoring compound" so we match that for consistency
    mean_score = df['IMP_Final_Score'].max() if 'IMP_Final_Score' in df.columns else 0
    mean_qed = df['QED'].mean() if 'QED' in df.columns else 0

    # On-demand HTML generation using session state to save memory
    # HTML is only generated when user clicks "Generate Report", not on every page load
    report_key = f"html_report_{compound_name}"

    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        # Generate button - only creates HTML when clicked
        if st.button("🔄 Generate HTML Report", key="generate_report_btn", help="Click to generate the HTML report for download"):
            with st.spinner("Generating report with charts..."):
                html_content = _generate_html_report(data, df)
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
                key="download_report_btn"
            )
        else:
            st.markdown("<small style='color: #888;'>Click 'Generate' first</small>", unsafe_allow_html=True)

    with col3:
        st.info("💡 **Tip:** Generate the report, download the HTML, then use Ctrl+P to print to PDF")

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


def _render_report_header(data: Dict[str, Any], smiles: str) -> None:
    """Render report header with 2D structure, compound name, SMILES."""
    st.markdown("## 📋 IMPULATOR Compound Analysis Report")

    compound_name = data.get('compound_name', 'Unknown')
    summary = data.get('summary', {})

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
                        unsafe_allow_html=True
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
        inchikey = data.get('inchikey') or "N/A"
        inchi = data.get('inchi') or "N/A"

        st.markdown(f"**InChIKey:** `{inchikey}`")
        if inchi and inchi != "N/A":
            st.markdown(f"**InChI:** `{inchi}`")
        smiles_display = smiles[:80] + '...' if len(smiles) > 80 else smiles
        st.markdown(f"**SMILES:** `{smiles_display}`")
        st.markdown(f"**Analysis Date:** {summary.get('processing_date', 'N/A')}")
        author_name = data.get('author_name', 'N/A')
        if author_name and author_name != 'N/A':
            st.markdown(f"**Author:** {html.escape(author_name)}")

    st.markdown("---")

    # Add summary stats row matching the Overview header
    df = data.get('results')
    if df is not None and not df.empty:
        # Get stats from summary (same source as Overview header)
        summary = data.get('summary', {})
        similar_count = summary.get('similar_count', df['ChEMBL_ID'].nunique() if 'ChEMBL_ID' in df.columns else len(df))
        activities_count = summary.get('total_activities', len(df))
        avg_qed = summary.get('qed') or (df['QED'].mean() if 'QED' in df.columns else None)
        best_imp_score = df['IMP_Final_Score'].max() if 'IMP_Final_Score' in df.columns else None

        # Count unique IMP compounds (not activity rows)
        imp_count = 0
        has_warning = False
        if 'Is_IMP_Candidate' in df.columns and 'ChEMBL_ID' in df.columns:
            imp_count = df[df['Is_IMP_Candidate']]['ChEMBL_ID'].nunique()
            has_warning = imp_count > 0
        elif summary.get('has_imp_candidates', False):
            imp_count = summary.get('imp_candidates', 0)
            has_warning = True

        # Display stats in columns
        stat_cols = st.columns(5)
        with stat_cols[0]:
            st.metric("Similar Compounds", similar_count)
        with stat_cols[1]:
            st.metric("Activities", activities_count)
        with stat_cols[2]:
            st.metric("QED", f"{avg_qed:.2f}" if avg_qed else "N/A")
        with stat_cols[3]:
            st.metric("IMP Score", f"{best_imp_score:.2f}" if best_imp_score else "N/A",
                      help="Best scoring compound (highest IMP risk)")
        with stat_cols[4]:
            if has_warning and imp_count > 0:
                st.markdown(f"""
                <div style="background-color: #721c24; color: white; padding: 10px 15px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 0.8em;">⚠️</div>
                    <div style="font-size: 1.5em; font-weight: bold;">{imp_count} IMP</div>
                    <div style="font-size: 0.65em;">unique compounds</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✓ Clean")

    st.markdown("---")


def _render_report_executive_summary(df: pd.DataFrame, mean_score: float, mean_qed: float, summary: dict) -> None:
    """Render executive summary with verdict badge."""
    st.markdown("### 📊 Executive Summary")

    interpretation = _get_imp_interpretation(mean_score)
    border_color = _get_imp_color(mean_score)

    # Count red flags - use SAME column names as Overview
    # Count UNIQUE COMPOUNDS with flags (not all rows)
    red_flag_cols = ['PAINS_Violation', 'Aggregator_Risk', 'Redox_Reactive', 'Fluorescence_Interference', 'Thiol_Reactive', 'BRENK_Alerts', 'NIH_Alerts']
    unique_df = df.drop_duplicates('ChEMBL_ID') if 'ChEMBL_ID' in df.columns else df
    red_flag_count = 0
    for col in red_flag_cols:
        if col in unique_df.columns:
            red_flag_count += int(unique_df[col].sum() if unique_df[col].dtype == bool else unique_df[col].astype(bool).sum())

    # Verdict badge with DARK background for good contrast
    priority_text = f"Priority {interpretation['priority']}" if interpretation['priority'] else "N/A"
    warning_icon = '⚠️' if mean_score >= 0.5 else '✓'

    # Use dark background with colored border and text for better contrast
    st.markdown(f"""
    <div style="background-color: #1e1e1e; border-left: 5px solid {border_color}; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <h4 style="color: {border_color}; margin: 0 0 10px 0;">
            {warning_icon} {interpretation['label'].upper()} - {priority_text}
        </h4>
        <p style="margin: 5px 0; color: #e0e0e0;"><strong style="color: #fff;">IMP Score:</strong> {mean_score:.3f} | <strong style="color: #fff;">QED:</strong> {mean_qed:.3f} | <strong style="color: #fff;">Red Flags:</strong> {red_flag_count} active</p>
        <p style="margin: 5px 0; color: #e0e0e0;"><strong style="color: #fff;">Risk Level:</strong> {interpretation['risk']}</p>
        <p style="margin: 5px 0; color: #e0e0e0;"><strong style="color: #fff;">Recommended Action:</strong> {interpretation['action']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Warning for high IMP scores
    if mean_score >= 0.7:
        st.warning("""
        **⚠️ HIGH FALSE POSITIVE RISK**

        This compound shows strong evidence of being an assay artifact (Invalid Metabolic Panacea).
        **DEPRIORITIZE** unless validated with orthogonal assays (SPR, ITC, or similar).
        """)

    st.markdown("---")


def _render_report_properties_table(df: pd.DataFrame) -> None:
    """Render compound properties table for BEST scoring compound."""
    st.markdown("### 🧪 Compound Properties")

    # Get best scoring compound to match Overview behavior
    if 'IMP_Final_Score' not in df.columns:
        st.info("Property data not available")
        st.markdown("---")
        return

    valid_df = df.dropna(subset=['IMP_Final_Score'])
    if valid_df.empty:
        st.info("Property data not available")
        st.markdown("---")
        return

    best_row = valid_df.loc[valid_df['IMP_Final_Score'].idxmax()]

    # Get values from best compound
    def get_val(col):
        return best_row.get(col) if col in best_row.index else None

    props = {
        "pActivity": (get_val('pActivity'), "-log10(IC50), higher = more potent"),
        "Molecular Weight": (get_val('Molecular_Weight'), "g/mol"),
        "PSA (TPSA)": (get_val('TPSA'), "Polar surface area (Å²)"),
        "Heavy Atoms": (get_val('Heavy_Atoms'), "Non-hydrogen atom count"),
        "N+O Atoms (NPOL)": (get_val('NPOL'), "Heteroatom count"),
        "QED": (get_val('QED'), "Drug-likeness (0-1)"),
        "LogP": (get_val('MolLogP') or get_val('LogP'), "Lipophilicity"),
    }

    # Create table
    table_data = []
    for prop_name, (value, description) in props.items():
        if value is not None and not pd.isna(value):
            table_data.append({
                "Property": prop_name,
                "Value": f"{value:.2f}" if isinstance(value, float) else str(value),
                "Description": description
            })

    if table_data:
        st.table(pd.DataFrame(table_data))
    else:
        st.info("Property data not available")

    st.markdown("---")


def _render_report_imp_score_calculation(df: pd.DataFrame) -> None:
    """Render step-by-step IMP Score calculation breakdown for BEST scoring compound."""
    st.markdown("### 🔢 IMP Score Calculation")

    # Get the best scoring compound (matches Overview tab behavior)
    if 'IMP_Final_Score' not in df.columns:
        st.info("IMP Score data not available")
        st.markdown("---")
        return

    valid_df = df.dropna(subset=['IMP_Final_Score'])
    if valid_df.empty:
        st.info("No valid IMP scores available")
        st.markdown("---")
        return

    # Get the row with highest IMP_Final_Score (best compound)
    best_row = valid_df.loc[valid_df['IMP_Final_Score'].idxmax()]

    st.caption("**Showing calculation for best scoring compound** (matches Overview tab)")

    # Step 1: Efficiency Metrics
    st.markdown("#### Step 1: Efficiency Metrics")

    metrics_data = []
    metric_cols = [
        ('SEI', 'pActivity × 100 / PSA', True),
        ('BEI', 'pActivity × 1000 / MW', True),
        ('NSEI', 'pActivity / (N+O Atoms)', False),
        ('NBEI', 'pActivity / Heavy Atoms', False),
    ]

    for col, formula, used in metric_cols:
        if col in best_row.index:
            value = best_row[col]
            metrics_data.append({
                "Metric": col,
                "Formula": formula,
                "Value": f"{value:.3f}" if not pd.isna(value) else "N/A",
                "Used in Score": "✓ YES" if used else "○ Display only"
            })

    if metrics_data:
        st.table(pd.DataFrame(metrics_data))

    st.caption("**Note:** Only SEI and BEI contribute to the Efficiency Score (NSEI/NBEI are for reference)")

    # Step 2: Component Scores
    st.markdown("#### Step 2: Component Scores")

    component_data = []
    components = [
        ('Efficiency_Score', 'Efficiency', '45%'),
        ('Distance_Score', 'Distance', '20%'),
        ('Angle_Score', 'Angle', '15%'),
        ('Interference_Score', 'Interference', '15%'),
        ('PDB_Score', 'PDB Evidence', '5%'),
    ]

    for col, name, weight in components:
        if col in best_row.index:
            score = best_row[col]
            contrib_col = col.replace('_Score', '_Contribution')
            contrib = best_row[contrib_col] if contrib_col in best_row.index else None
            component_data.append({
                "Component": name,
                "Score": f"{score:.3f}" if not pd.isna(score) else "N/A",
                "Weight": weight,
                "Contribution": f"{contrib:.3f}" if contrib and not pd.isna(contrib) else "N/A"
            })

    if component_data:
        st.table(pd.DataFrame(component_data))

    # Step 3: Final Calculation
    st.markdown("#### Step 3: Final Calculation")

    # Use direct indexing for pandas Series (not .get())
    base_score = best_row['IMP_Base_Score'] if 'IMP_Base_Score' in best_row.index else None
    qed = best_row['QED'] if 'QED' in best_row.index else None
    qed_mult = best_row['QED_Multiplier'] if 'QED_Multiplier' in best_row.index else None
    final_score = best_row['IMP_Final_Score'] if 'IMP_Final_Score' in best_row.index else None

    if all(v is not None and not pd.isna(v) for v in [base_score, qed, qed_mult, final_score]):
        # Extract component scores for the formula
        eff_s = best_row['Efficiency_Score'] if 'Efficiency_Score' in best_row.index else 0
        dist_s = best_row['Distance_Score'] if 'Distance_Score' in best_row.index else 0
        ang_s = best_row['Angle_Score'] if 'Angle_Score' in best_row.index else 0
        int_s = best_row['Interference_Score'] if 'Interference_Score' in best_row.index else 0
        pdb_s = best_row['PDB_Score'] if 'PDB_Score' in best_row.index else 0

        # Display as formatted text box (not code block to avoid scrolling)
        st.markdown(f"""
<div style="background-color: #1e1e1e; padding: 15px; border-radius: 8px; font-family: monospace; white-space: pre-wrap;">
<strong>Base Score</strong> = 0.45×Eff + 0.20×Dist + 0.15×Angle + 0.15×Interf + 0.05×PDB
         = 0.45×<span style="color: #4ec9b0;">{eff_s:.3f}</span> + 0.20×<span style="color: #4ec9b0;">{dist_s:.3f}</span> + 0.15×<span style="color: #4ec9b0;">{ang_s:.3f}</span> + 0.15×<span style="color: #4ec9b0;">{int_s:.3f}</span> + 0.05×<span style="color: #4ec9b0;">{pdb_s:.3f}</span>
         = <span style="color: #4ec9b0;">{base_score:.3f}</span>

<strong>QED Value:</strong> <span style="color: #4ec9b0;">{qed:.3f}</span>
<strong>QED Multiplier</strong> = 0.75 + 0.25 × QED
             = 0.75 + 0.25 × {qed:.3f}
             = <span style="color: #4ec9b0;">{qed_mult:.3f}</span>

<strong>FINAL SCORE</strong> = Base Score × QED Multiplier
            = {base_score:.3f} × {qed_mult:.3f}
            = <span style="color: #dcdcaa; font-size: 1.2em;"><strong>{final_score:.3f}</strong></span>
</div>
        """, unsafe_allow_html=True)
    else:
        st.info("Complete IMP Score calculation data not available")

    st.markdown("---")


def _render_report_red_flags(df: pd.DataFrame) -> None:
    """Render red flags assessment section using SAME column names as Overview."""
    st.markdown("### ⚠️ Red Flags Assessment")

    # Use SAME column names as Overview tab
    flags = [
        ('PAINS_Violation', 'PAINS', 'Pan-Assay Interference compounds detected'),
        ('Aggregator_Risk', 'Aggregator', 'May form colloidal aggregates'),
        ('Redox_Reactive', 'Redox', 'May interfere via redox cycling'),
        ('Fluorescence_Interference', 'Fluorescence', 'May interfere with fluorescence assays'),
        ('Thiol_Reactive', 'Thiol', 'May react with cysteine residues'),
        ('BRENK_Alerts', 'BRENK', 'Unwanted substructures detected'),
        ('NIH_Alerts', 'NIH', 'Problematic functional groups detected'),
    ]

    total_flags = 0
    flag_data = []

    # Count UNIQUE COMPOUNDS with flags (not all rows) - same as Overview tab
    unique_df = df.drop_duplicates('ChEMBL_ID') if 'ChEMBL_ID' in df.columns else df

    for col, name, description in flags:
        if col in unique_df.columns:
            count = int(unique_df[col].sum() if unique_df[col].dtype == bool else unique_df[col].astype(bool).sum())
            total_flags += count
            flag_data.append((name, count, description))

    # Overall assessment
    if total_flags == 0:
        overall = "LOW CONCERN - No red flags detected"
        overall_color = "#28a745"
        overall_bg = "#d4edda"
    elif total_flags <= 5:
        overall = f"MODERATE CONCERN - {total_flags} flags detected"
        overall_color = "#856404"
        overall_bg = "#fff3cd"
    else:
        overall = f"HIGH CONCERN - {total_flags} flags detected"
        overall_color = "#721c24"
        overall_bg = "#f8d7da"

    st.markdown(f"""
    <div style="background-color: {overall_bg}; padding: 12px; border-radius: 5px; margin-bottom: 15px; border: 1px solid {overall_color};">
        <strong style="color: {overall_color};">Overall Assessment: {overall}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Display each flag with counts
    cols = st.columns(len(flag_data)) if flag_data else []
    for i, (name, count, description) in enumerate(flag_data):
        with cols[i]:
            if count > 0:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #dc3545;">
                    <div style="font-size: 1.5em; color: #dc3545; font-weight: bold;">{count}</div>
                    <div style="color: #fff;">{name}</div>
                    <div style="font-size: 0.7em; color: #dc3545;">⚠️ Flagged</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #28a745;">
                    <div style="font-size: 1.5em; color: #28a745; font-weight: bold;">0</div>
                    <div style="color: #fff;">{name}</div>
                    <div style="font-size: 0.7em; color: #28a745;">✓ Clean</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")


def _render_report_bioactivity_donut(df: pd.DataFrame) -> None:
    """Render bioactivity distribution donut chart."""
    st.markdown("### 🎯 Bioactivity Distribution")

    if 'Activity_Type' not in df.columns:
        st.info("Activity type data not available")
        st.markdown("---")
        return

    # Count activity types
    activity_counts = df['Activity_Type'].value_counts()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Create donut chart
        fig = px.pie(
            values=activity_counts.values,
            names=activity_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=350,
            margin=dict(t=30, b=30, l=30, r=30),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig, width='stretch', key="report_activity_donut")

    with col2:
        # Summary table
        st.markdown(f"**Total Activities:** {len(df)}")
        st.markdown(f"**Activity Types:** {len(activity_counts)}")

        table_data = []
        for activity_type, count in activity_counts.head(10).items():
            pct = (count / len(df)) * 100
            table_data.append({
                "Type": activity_type,
                "Count": count,
                "Percentage": f"{pct:.1f}%"
            })
        st.table(pd.DataFrame(table_data))

    st.markdown("---")


def _render_report_efficiency_boxplots(df: pd.DataFrame) -> None:
    """Render efficiency metrics box plots with enhanced statistics cards."""
    st.markdown("### 📈 Efficiency Metrics Distribution")

    metrics = ['SEI', 'BEI', 'NSEI', 'NBEI']
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        st.info("Efficiency metrics not available")
        st.markdown("---")
        return

    st.caption("**Note:** Only SEI and BEI are used in IMP scoring. NSEI and NBEI are shown for additional context.")

    # Calculate statistics and create metric cards
    metric_colors = {
        'SEI': '#1f77b4',
        'BEI': '#2ca02c',
        'NSEI': '#ff7f0e',
        'NBEI': '#9467bd'
    }

    metric_descriptions = {
        'SEI': 'Surface Efficiency Index',
        'BEI': 'Binding Efficiency Index',
        'NSEI': 'Normalized SEI (display only)',
        'NBEI': 'Normalized BEI (display only)'
    }

    # Display metric cards
    cols = st.columns(len(available_metrics))
    for i, metric in enumerate(available_metrics):
        vals = df[metric].dropna()
        if len(vals) > 0:
            with cols[i]:
                mean_val = vals.mean()
                used_in_score = " ✓" if metric in ['SEI', 'BEI'] else ""
                description = metric_descriptions.get(metric, metric)
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid {metric_colors.get(metric, '#636EFA')};">
                    <div style="color: #fff; font-size: 0.9em; margin-bottom: 5px;">{metric}{used_in_score}</div>
                    <div style="font-size: 1.5em; color: {metric_colors.get(metric, '#636EFA')}; font-weight: bold;">{mean_val:.2f}</div>
                    <div style="color: #aaa; font-size: 0.75em;">{description}</div>
                    <div style="color: #888; font-size: 0.7em; margin-top: 3px;">Range: {vals.min():.1f}-{vals.max():.1f}</div>
                </div>
                """, unsafe_allow_html=True)

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
        color_discrete_map=metric_colors
    )
    fig.update_layout(
        height=400,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=False,
        xaxis_title="Efficiency Metric",
        yaxis_title="Value"
    )
    st.plotly_chart(fig, width='stretch', key="report_efficiency_box")

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

    if 'SEI' not in df.columns or 'BEI' not in df.columns:
        st.info("SEI/BEI data not available")
        st.markdown("---")
        return

    plot_df = df.dropna(subset=['SEI', 'BEI'])

    if plot_df.empty:
        st.info("No valid SEI/BEI data points")
        st.markdown("---")
        return

    # Calculate mean angle and modulus
    mean_sei = plot_df['SEI'].mean()
    mean_bei = plot_df['BEI'].mean()
    mean_angle = plot_df['Angle_SEI_BEI'].mean() if 'Angle_SEI_BEI' in plot_df.columns else np.arctan2(mean_bei, mean_sei) * 180 / np.pi
    mean_modulus = plot_df['Modulus_SEI_BEI'].mean() if 'Modulus_SEI_BEI' in plot_df.columns else np.sqrt(mean_sei**2 + mean_bei**2)

    # Angle assessment
    if 40 <= mean_angle <= 50:
        angle_status = "OPTIMAL ✓"
        angle_color = "#28a745"
        angle_bg = "#d4edda"
    elif 30 <= mean_angle < 40 or 50 < mean_angle <= 60:
        angle_status = "ACCEPTABLE"
        angle_color = "#856404"
        angle_bg = "#fff3cd"
    else:
        angle_status = "UNBALANCED ⚠️"
        angle_color = "#721c24"
        angle_bg = "#f8d7da"

    # Display angle assessment banner
    st.markdown(f"""
    <div style="background-color: {angle_bg}; padding: 12px; border-radius: 5px; margin-bottom: 15px; border: 1px solid {angle_color};">
        <strong style="color: {angle_color};">Development Trajectory: {angle_status} (Angle: {mean_angle:.1f}°)</strong>
    </div>
    """, unsafe_allow_html=True)

    # Create scatter plot
    fig = go.Figure()

    # Add data points
    color_col = 'IMP_Final_Score' if 'IMP_Final_Score' in plot_df.columns else None

    fig.add_trace(go.Scatter(
        x=plot_df['SEI'],
        y=plot_df['BEI'],
        mode='markers',
        marker=dict(
            size=8,
            color=plot_df[color_col] if color_col else '#636EFA',
            colorscale='RdYlGn_r' if color_col else None,  # Red = high IMP (bad)
            showscale=True if color_col else False,
            colorbar=dict(title="IMP Score") if color_col else None,
            opacity=0.7
        ),
        text=plot_df['Molecule_Name'] if 'Molecule_Name' in plot_df.columns else None,
        hovertemplate="<b>%{text}</b><br>SEI: %{x:.2f}<br>BEI: %{y:.2f}<extra></extra>" if 'Molecule_Name' in plot_df.columns else "SEI: %{x:.2f}<br>BEI: %{y:.2f}<extra></extra>",
        name='Compounds'
    ))

    # Add 45° reference line (optimal development angle)
    max_val = max(plot_df['SEI'].max(), plot_df['BEI'].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='green', dash='dash', width=2),
        name='45° Optimal',
        hoverinfo='skip'
    ))

    # Add mean angle line from origin
    angle_rad = mean_angle * np.pi / 180
    line_length = mean_modulus * 1.2
    fig.add_trace(go.Scatter(
        x=[0, line_length * np.cos(angle_rad)],
        y=[0, line_length * np.sin(angle_rad)],
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Mean Angle ({mean_angle:.1f}°)',
        hoverinfo='skip'
    ))

    # Add mean point marker
    fig.add_trace(go.Scatter(
        x=[mean_sei],
        y=[mean_bei],
        mode='markers',
        marker=dict(size=15, color='orange', symbol='star', line=dict(width=2, color='white')),
        name=f'Mean Point ({mean_sei:.1f}, {mean_bei:.1f})',
        hovertemplate="Mean SEI: %{x:.2f}<br>Mean BEI: %{y:.2f}<extra></extra>"
    ))

    # CRITICAL: Equal axis scaling so visual angle matches calculated angle
    fig.update_layout(
        height=500,
        margin=dict(t=30, b=30, l=30, r=30),
        xaxis=dict(
            title="SEI (Surface Efficiency Index)",
            scaleanchor="y",  # CRITICAL: Link x to y
            scaleratio=1,     # CRITICAL: 1:1 ratio
            range=[0, max_val],  # Start at 0 (SEI/BEI always positive)
            autorange=False,  # Disable autorange to enforce range
            constrain="domain"  # Constrain to specified range
        ),
        yaxis=dict(
            title="BEI (Binding Efficiency Index)",
            range=[0, max_val],  # Start at 0 (SEI/BEI always positive)
            autorange=False,  # Disable autorange to enforce range
            constrain="domain"  # Constrain to specified range
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, width='stretch', key="report_efficiency_plane")

    # Enhanced interpretation with metrics cards
    st.markdown("#### Efficiency Plane Analysis")

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #636EFA;">
            <div style="font-size: 1.2em; color: #636EFA; font-weight: bold;">{mean_angle:.1f}°</div>
            <div style="color: #fff; font-size: 0.9em;">Mean Angle</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #ff7f0e;">
            <div style="font-size: 1.2em; color: #ff7f0e; font-weight: bold;">{mean_modulus:.1f}</div>
            <div style="color: #fff; font-size: 0.9em;">Mean Modulus</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #2ca02c;">
            <div style="font-size: 1.2em; color: #2ca02c; font-weight: bold;">{mean_sei:.1f}</div>
            <div style="color: #fff; font-size: 0.9em;">Mean SEI</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #d62728;">
            <div style="font-size: 1.2em; color: #d62728; font-weight: bold;">{mean_bei:.1f}</div>
            <div style="color: #fff; font-size: 0.9em;">Mean BEI</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    **Angle Interpretation:**
    - **< 45°:** Compound favors **size efficiency** (SEI) - efficient use of polar surface area
    - **= 45°:** **Balanced development** (OPTIMAL) - equal efficiency in size and binding
    - **> 45°:** Compound favors **binding efficiency** (BEI) - efficient use of molecular weight

    **Note:** Most approved drugs have angles between 40-60°. The green dashed line shows the optimal 45° trajectory. The orange star marks the mean efficiency point.
    """)

    st.markdown("---")


def _render_report_pdb_evidence(df: pd.DataFrame, data: Dict[str, Any]) -> None:
    """Render PDB structural evidence section."""
    st.markdown("### 🔬 PDB Structural Evidence")

    # Check for PDB columns
    pdb_cols = ['PDB_Score', 'PDB_Num_Structures', 'PDB_High_Quality', 'PDB_Medium_Quality', 'PDB_Poor_Quality']
    has_pdb = any(col in df.columns for col in pdb_cols)

    if not has_pdb:
        st.info("PDB structural evidence data not available")
        st.markdown("---")
        return

    # Try to load detailed PDB summary file for accurate counts
    pdb_summary_df = None
    compound_name = data.get('compound_name', '')
    entry_id = data.get('entry_id')
    storage_path = data.get('storage_path')

    try:
        safe_name = sanitize_compound_name(compound_name)
        for filename in ["pdb_summary.csv", f"{safe_name}_pdb_summary.csv", f"{safe_name}_pdb_details.csv"]:
            pdb_summary_df = smart_load_dataframe(
                filename,
                entry_id=entry_id,
                storage_path=storage_path
            )
            if pdb_summary_df is not None and not pdb_summary_df.empty:
                break
    except Exception:
        pdb_summary_df = None

    # Calculate stats - use pdb_summary_df if available for accurate counts
    if pdb_summary_df is not None and not pdb_summary_df.empty:
        total_structures = len(pdb_summary_df)
        if 'Quality' in pdb_summary_df.columns:
            high_quality = int((pdb_summary_df['Quality'] == '***').sum())
            medium_quality = int((pdb_summary_df['Quality'] == '**').sum())
            poor_quality = int((pdb_summary_df['Quality'] == '*').sum())
        else:
            if 'Resolution' in pdb_summary_df.columns:
                pdb_summary_df['_res'] = pd.to_numeric(pdb_summary_df['Resolution'], errors='coerce')
                high_quality = int((pdb_summary_df['_res'] < 2.0).sum())
                medium_quality = int(((pdb_summary_df['_res'] >= 2.0) & (pdb_summary_df['_res'] <= 3.0)).sum())
                poor_quality = int((pdb_summary_df['_res'] > 3.0).sum())
            else:
                high_quality = medium_quality = poor_quality = 0
    else:
        # Fallback to summing from dataframe (less accurate)
        total_structures = int(df['PDB_Num_Structures'].sum()) if 'PDB_Num_Structures' in df.columns else 0
        high_quality = int(df['PDB_High_Quality'].sum()) if 'PDB_High_Quality' in df.columns else 0
        medium_quality = int(df['PDB_Medium_Quality'].sum()) if 'PDB_Medium_Quality' in df.columns else 0
        poor_quality = int(df['PDB_Poor_Quality'].sum()) if 'PDB_Poor_Quality' in df.columns else 0

    mean_pdb_score = df['PDB_Score'].mean() if 'PDB_Score' in df.columns else 0

    # Confidence assessment banner
    if mean_pdb_score >= 0.7:
        confidence = "HIGH CONFIDENCE"
        conf_color = "#28a745"
        conf_bg = "#d4edda"
        conf_icon = "✓"
    elif mean_pdb_score >= 0.4:
        confidence = "MEDIUM CONFIDENCE"
        conf_color = "#856404"
        conf_bg = "#fff3cd"
        conf_icon = "●"
    else:
        confidence = "LOW CONFIDENCE"
        conf_color = "#721c24"
        conf_bg = "#f8d7da"
        conf_icon = "⚠️"

    st.markdown(f"""
    <div style="background-color: {conf_bg}; padding: 12px; border-radius: 5px; margin-bottom: 15px; border: 1px solid {conf_color};">
        <strong style="color: {conf_color};">{conf_icon} Structural Validation: {confidence} (PDB Score: {mean_pdb_score:.3f})</strong>
    </div>
    """, unsafe_allow_html=True)

    # Quality distribution cards
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #636EFA;">
            <div style="font-size: 1.5em; color: #636EFA; font-weight: bold;">{total_structures}</div>
            <div style="color: #fff; font-size: 0.9em;">Total Structures</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        star_display = '⭐⭐⭐' if high_quality > 0 else ''
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #28a745;">
            <div style="font-size: 1.5em; color: #28a745; font-weight: bold;">{high_quality}</div>
            <div style="color: #fff; font-size: 0.9em;">High Quality</div>
            <div style="color: #28a745; font-size: 0.8em;">{star_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        star_display = '⭐⭐' if medium_quality > 0 else ''
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #ffc107;">
            <div style="font-size: 1.5em; color: #ffc107; font-weight: bold;">{medium_quality}</div>
            <div style="color: #fff; font-size: 0.9em;">Medium Quality</div>
            <div style="color: #ffc107; font-size: 0.8em;">{star_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        star_display = '⭐' if poor_quality > 0 else ''
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #dc3545;">
            <div style="font-size: 1.5em; color: #dc3545; font-weight: bold;">{poor_quality}</div>
            <div style="color: #fff; font-size: 0.9em;">Poor Quality</div>
            <div style="color: #dc3545; font-size: 0.8em;">{star_display}</div>
        </div>
        """, unsafe_allow_html=True)

    if total_structures > 0:
        st.markdown("")

        # Create horizontal bar chart for quality distribution (full-width)
        quality_data = pd.DataFrame({
            'Quality': ['High (<2.0Å)', 'Medium (2-3Å)', 'Poor (>3Å)'],
            'Count': [high_quality, medium_quality, poor_quality],
            'Percentage': [
                f"{high_quality/total_structures*100:.1f}%" if total_structures > 0 else "0%",
                f"{medium_quality/total_structures*100:.1f}%" if total_structures > 0 else "0%",
                f"{poor_quality/total_structures*100:.1f}%" if total_structures > 0 else "0%"
            ]
        })

        fig = px.bar(
            quality_data,
            x='Count',
            y='Quality',
            orientation='h',
            color='Quality',
            text='Percentage',
            color_discrete_map={
                'High (<2.0Å)': '#28a745',
                'Medium (2-3Å)': '#ffc107',
                'Poor (>3Å)': '#dc3545'
            }
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            xaxis_title="Number of Structures",
            yaxis_title=""
        )
        st.plotly_chart(fig, width='stretch', key="report_pdb_quality")

        # Resolution Quality info box below
        st.markdown("""
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
        """, unsafe_allow_html=True)
    else:
        st.info("No PDB structures found for these compounds. This is common for early-stage research compounds not yet structurally characterized.")

    st.markdown("---")


def _render_report_classification(df: pd.DataFrame) -> None:
    """Render chemical classification section (ClassyFire + NPClassifier)."""
    st.markdown("### 🧬 Chemical Classification")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ClassyFire Taxonomy:**")
        classyfire_cols = ['Kingdom', 'Superclass', 'Class', 'Subclass']
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
        np_cols = ['NP_Pathway', 'NP_Superclass', 'NP_Class']
        has_np = any(col in df.columns for col in np_cols)

        if has_np:
            for col in np_cols:
                if col in df.columns:
                    display_name = col.replace('NP_', '')
                    mode_vals = df[col].mode()
                    value = mode_vals.iloc[0] if not mode_vals.empty else "N/A"
                    st.markdown(f"- **{display_name}:** {value}")
        else:
            st.info("NPClassifier data not available")

    st.markdown("---")


def _render_report_indications(data: Dict[str, Any]) -> None:
    """Render drug indications section."""
    st.markdown("### 💊 Drug Indications")

    indications_df = data.get('indications')

    if indications_df is None or (isinstance(indications_df, pd.DataFrame) and indications_df.empty):
        st.info("No drug indication data available")
        st.markdown("---")
        return

    # Get max phase
    max_phase = indications_df['Max_Phase'].max() if 'Max_Phase' in indications_df.columns else "N/A"
    unique_indications = indications_df['MESH_Heading'].nunique() if 'MESH_Heading' in indications_df.columns else 0

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        **Maximum Clinical Phase:** {max_phase}

        **Unique Indications:** {unique_indications}

        **Compounds with Data:** {indications_df['ChEMBL_ID'].nunique() if 'ChEMBL_ID' in indications_df.columns else 'N/A'}
        """)

    with col2:
        # Top indications table
        if 'MESH_Heading' in indications_df.columns and 'Max_Phase' in indications_df.columns:
            top_indications = indications_df.groupby('MESH_Heading')['Max_Phase'].max().sort_values(ascending=False).head(5)

            if not top_indications.empty:
                st.markdown("**Top Indications by Phase:**")
                table_data = [{"Indication": ind, "Max Phase": phase} for ind, phase in top_indications.items()]
                st.table(pd.DataFrame(table_data))

    st.markdown("---")


def _render_report_recommendation(df: pd.DataFrame, mean_score: float) -> None:
    """Render final recommendation section with IMP interpretation guide."""
    st.markdown("### 🎯 Final Recommendation")

    interpretation = _get_imp_interpretation(mean_score)
    color = _get_imp_color(mean_score)

    # Verdict box
    warning_icon = '⚠️' if mean_score >= 0.5 else '✓'
    st.markdown(f"""
    <div style="border: 2px solid {color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="color: {color}; margin: 0 0 15px 0;">
            {warning_icon} VERDICT: {interpretation['label'].upper()}
        </h4>
        <p style="margin: 10px 0;">{interpretation['risk']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Recommended actions based on score and flags
    st.markdown("**Recommended Actions:**")

    actions = []

    if mean_score >= 0.7:
        actions.append(("HIGH", "Validate with orthogonal binding assay (SPR/ITC/MST)"))
        actions.append(("HIGH", "Counter-screen against aggregation"))

    # Check for PAINS - use correct column name
    if 'PAINS_Violation' in df.columns and df['PAINS_Violation'].any():
        actions.append(("HIGH", "Counter-screen PAINS-flagged compounds"))

    if 0.5 <= mean_score < 0.7:
        actions.append(("MEDIUM", "Moderate IMP risk - validation recommended before advancing"))

    if 'QED' in df.columns and df['QED'].mean() < 0.5:
        actions.append(("LOW", "Consider SAR optimization to improve drug-likeness"))

    if 'PDB_Score' in df.columns and df['PDB_Score'].mean() < 0.3:
        actions.append(("LOW", "Obtain structural evidence (X-ray/cryo-EM) before advancing"))

    if mean_score < 0.3:
        actions.append(("INFO", "Low IMP risk - compound more likely genuine, proceed with development"))

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
    st.markdown("""
    | Score Range | Classification | Meaning | Action |
    |-------------|----------------|---------|--------|
    | 0.90+ | Exceptional IMP | ⚠️ VERY HIGH false positive risk | Immediate validation |
    | 0.70-0.89 | Strong IMP | ⚠️ HIGH false positive risk | DEPRIORITIZE unless validated |
    | 0.50-0.69 | Moderate IMP | ⚠️ Moderate risk | Validate carefully |
    | 0.30-0.49 | Weak IMP | Lower risk - more likely genuine | Standard follow-up |
    | < 0.30 | Not IMP | ✓ Likely genuine activity | Proceed with development |
    """)

    st.caption("**Remember:** IMP = Invalid Metabolic Panacea = Likely FALSE POSITIVE (assay artifact)")


def _export_plotly_to_base64(fig, width: int = 700, height: int = 400, scale: float = 3.0) -> str:
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
            paper_bgcolor='white',
            plot_bgcolor='white',
            font_color='#333',
            font_size=12  # Slightly larger font for better readability in print
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
        return f"<p style='color: #999;'>Chart unavailable: {str(e)}</p>"


def _create_html_bioactivity_donut(df: pd.DataFrame) -> str:
    """Create bioactivity donut chart for HTML export."""
    if 'Activity_Type' not in df.columns:
        return "<p>Activity type data not available</p>"

    type_counts = df['Activity_Type'].value_counts().head(6)
    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=400,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=True
    )
    return _export_plotly_to_base64(fig, 800, 400)


def _create_html_efficiency_boxplots(df: pd.DataFrame) -> str:
    """Create efficiency metrics box plots for HTML export."""
    metrics = ['SEI', 'BEI', 'NSEI', 'NBEI']
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        return "<p>Efficiency metrics not available</p>"

    fig = go.Figure()
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, metric in enumerate(available_metrics):
        values = df[metric].dropna()
        if len(values) > 0:
            fig.add_trace(go.Box(
                y=values,
                name=metric,
                marker_color=colors[i % len(colors)],
                boxpoints='outliers'
            ))

    fig.update_layout(
        height=400,
        margin=dict(t=30, b=50, l=50, r=30),
        showlegend=False,
        yaxis_title="Value"
    )
    return _export_plotly_to_base64(fig, 900, 400)


def _create_html_efficiency_scatter(df: pd.DataFrame) -> str:
    """Create SEI vs BEI scatter plot with equal axis scaling for HTML export."""
    if 'SEI' not in df.columns or 'BEI' not in df.columns:
        return "<p>SEI/BEI data not available</p>"

    plot_df = df[['SEI', 'BEI']].dropna()
    if plot_df.empty:
        return "<p>No valid SEI/BEI data</p>"

    # Get color data if available
    if 'IMP_Final_Score' in df.columns:
        plot_df = df[['SEI', 'BEI', 'IMP_Final_Score']].dropna()
        color_col = 'IMP_Final_Score'
    else:
        color_col = None

    # Calculate mean values
    mean_sei = plot_df['SEI'].mean()
    mean_bei = plot_df['BEI'].mean()
    mean_angle = np.degrees(np.arctan2(mean_bei, mean_sei))
    mean_modulus = np.sqrt(mean_sei**2 + mean_bei**2)

    fig = go.Figure()

    # Add data points
    if color_col:
        fig.add_trace(go.Scatter(
            x=plot_df['SEI'],
            y=plot_df['BEI'],
            mode='markers',
            marker=dict(
                size=8,
                color=plot_df[color_col],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="IMP Score")
            ),
            name='Compounds',
            hovertemplate='SEI: %{x:.2f}<br>BEI: %{y:.2f}<br>IMP Score: %{marker.color:.3f}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_df['SEI'],
            y=plot_df['BEI'],
            mode='markers',
            marker=dict(size=8, color='#3498db'),
            name='Compounds',
            hovertemplate='SEI: %{x:.2f}<br>BEI: %{y:.2f}<extra></extra>'
        ))

    # Add 45° reference line
    max_val = max(plot_df['SEI'].max(), plot_df['BEI'].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='45° Optimal Line',
        hovertemplate='45° Optimal (Balanced Development)<extra></extra>'
    ))

    # Add mean point marker (orange star)
    fig.add_trace(go.Scatter(
        x=[mean_sei],
        y=[mean_bei],
        mode='markers',
        marker=dict(size=15, color='orange', symbol='star', line=dict(width=2, color='white')),
        name=f'Mean Point ({mean_sei:.1f}, {mean_bei:.1f})',
        hovertemplate=f'Mean SEI: {mean_sei:.2f}<br>Mean BEI: {mean_bei:.2f}<br>Angle: {mean_angle:.1f}°<br>Modulus: {mean_modulus:.2f}<extra></extra>'
    ))

    fig.update_layout(
        template='plotly_white',
        height=600,
        margin=dict(t=40, b=50, l=60, r=30),
        xaxis=dict(
            title="SEI (Surface Efficiency Index)",
            scaleanchor="y",  # CRITICAL: Link x to y
            scaleratio=1,     # CRITICAL: 1:1 ratio
            range=[0, max_val],  # Start at 0 (SEI/BEI always positive)
            autorange=False,  # Disable autorange to enforce range
            constrain="domain"  # Constrain to specified range
        ),
        yaxis=dict(
            title="BEI (Binding Efficiency Index)",
            range=[0, max_val],  # Start at 0
            autorange=False,
            constrain="domain"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return _export_plotly_to_base64(fig, 900, 600)


def _create_html_pdb_quality_bar(df: pd.DataFrame, data: Dict[str, Any]) -> str:
    """Create PDB quality distribution bar chart for HTML export."""
    # Try to load pdb_summary for accurate counts
    pdb_summary_df = None
    compound_name = data.get('compound_name', '')
    entry_id = data.get('entry_id')
    storage_path = data.get('storage_path')

    try:
        safe_name = sanitize_compound_name(compound_name)
        for filename in ["pdb_summary.csv", f"{safe_name}_pdb_summary.csv"]:
            pdb_summary_df = smart_load_dataframe(filename, entry_id=entry_id, storage_path=storage_path)
            if pdb_summary_df is not None and not pdb_summary_df.empty:
                break
    except Exception:
        pass

    if pdb_summary_df is not None and not pdb_summary_df.empty and 'Quality' in pdb_summary_df.columns:
        high_q = int((pdb_summary_df['Quality'] == '***').sum())
        med_q = int((pdb_summary_df['Quality'] == '**').sum())
        poor_q = int((pdb_summary_df['Quality'] == '*').sum())
    elif 'PDB_High_Quality' in df.columns:
        high_q = int(df['PDB_High_Quality'].max()) if df['PDB_High_Quality'].notna().any() else 0
        med_q = int(df['PDB_Medium_Quality'].max()) if 'PDB_Medium_Quality' in df.columns and df['PDB_Medium_Quality'].notna().any() else 0
        poor_q = int(df['PDB_Poor_Quality'].max()) if 'PDB_Poor_Quality' in df.columns and df['PDB_Poor_Quality'].notna().any() else 0
    else:
        return "<p>PDB quality data not available</p>"

    if high_q + med_q + poor_q == 0:
        return "<p>No PDB structures found</p>"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=['High (<2.0Å)', 'Medium (2-3Å)', 'Poor (>3Å)'],
        x=[high_q, med_q, poor_q],
        orientation='h',
        marker_color=['#28a745', '#ffc107', '#dc3545']
    ))
    fig.update_layout(
        height=250,
        margin=dict(t=20, b=30, l=100, r=30),
        xaxis_title="Count",
        showlegend=False
    )
    return _export_plotly_to_base64(fig, 700, 250)


def _generate_html_report(data: Dict[str, Any], df: pd.DataFrame) -> str:
    """Generate comprehensive HTML report with ALL sections matching Report tab."""
    import base64
    import io
    from datetime import datetime

    compound_name = data.get('compound_name', 'Unknown')
    smiles = data.get('smiles', '')
    summary = data.get('summary', {})

    # Get BEST scoring compound row (matches Overview behavior)
    best_row = None
    if 'IMP_Final_Score' in df.columns:
        valid_df = df.dropna(subset=['IMP_Final_Score'])
        if not valid_df.empty:
            best_row = valid_df.loc[valid_df['IMP_Final_Score'].idxmax()]

    # Calculate scores from best compound
    final_score = best_row['IMP_Final_Score'] if best_row is not None else 0
    qed_val = best_row['QED'] if best_row is not None and 'QED' in best_row.index else 0
    interpretation = _get_imp_interpretation(final_score)
    color = _get_imp_color(final_score)
    bg_color = _get_imp_bg_color(final_score)

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
    inchikey = html.escape(data.get('inchikey') or "N/A")
    inchi = html.escape(data.get('inchi') or "N/A")

    # Compute summary stats for header (matching Overview quick stats)
    # Use summary data as source of truth (same as Overview header)
    similar_count = summary.get('similar_count', df['ChEMBL_ID'].nunique() if 'ChEMBL_ID' in df.columns else len(df))
    activities_count = summary.get('total_activities', len(df))
    avg_qed = summary.get('qed') or (df['QED'].mean() if 'QED' in df.columns else None)
    best_imp_score = df['IMP_Final_Score'].max() if 'IMP_Final_Score' in df.columns else None

    # Count unique IMP compounds (not activity rows)
    imp_count = 0
    has_warning = False
    if 'Is_IMP_Candidate' in df.columns and 'ChEMBL_ID' in df.columns:
        imp_count = df[df['Is_IMP_Candidate']]['ChEMBL_ID'].nunique()
        has_warning = imp_count > 0
    elif summary.get('has_imp_candidates', False):
        imp_count = summary.get('imp_candidates', 0)
        has_warning = True

    # Format stats
    avg_qed_str = f"{avg_qed:.2f}" if avg_qed is not None and not pd.isna(avg_qed) else "N/A"
    avg_imp_score_str = f"{best_imp_score:.2f}" if best_imp_score is not None and not pd.isna(best_imp_score) else "N/A"

    # Build properties table from best compound
    props_html = ""
    if best_row is not None:
        prop_cols = [
            ('pActivity', 'pActivity', '-log10(IC50)'),
            ('Molecular_Weight', 'Molecular Weight', 'g/mol'),
            ('TPSA', 'PSA (TPSA)', 'Å²'),
            ('Heavy_Atoms', 'Heavy Atoms', 'count'),
            ('NPOL', 'N+O Atoms', 'count'),
            ('QED', 'QED', 'Drug-likeness'),
        ]
        for col, label, unit in prop_cols:
            if col in best_row.index and not pd.isna(best_row[col]):
                props_html += f"<tr><td>{label}</td><td>{best_row[col]:.3f}</td><td>{unit}</td></tr>"

    # Build efficiency metrics table from best compound
    efficiency_html = ""
    if best_row is not None:
        eff_cols = [
            ('SEI', 'SEI', 'pActivity × 100 / PSA', True),
            ('BEI', 'BEI', 'pActivity × 1000 / MW', True),
            ('NSEI', 'NSEI', 'pActivity / NPOL', False),
            ('NBEI', 'NBEI', 'pActivity / Heavy Atoms', False),
        ]
        for col, label, formula, used in eff_cols:
            if col in best_row.index and not pd.isna(best_row[col]):
                used_text = "✓ Used" if used else "Display only"
                efficiency_html += f"<tr><td>{label}</td><td>{best_row[col]:.3f}</td><td>{formula}</td><td>{used_text}</td></tr>"

    # Build component scores table from best compound
    components_html = ""
    if best_row is not None:
        components = [
            ('Efficiency_Score', 'Efficiency', '45%'),
            ('Distance_Score', 'Distance', '20%'),
            ('Angle_Score', 'Angle', '15%'),
            ('Interference_Score', 'Interference', '15%'),
            ('PDB_Score', 'PDB Evidence', '5%'),
        ]
        for col, name, weight in components:
            if col in best_row.index and not pd.isna(best_row[col]):
                contrib_col = col.replace('_Score', '_Contribution')
                contrib = best_row[contrib_col] if contrib_col in best_row.index else None
                contrib_str = f"{contrib:.3f}" if contrib and not pd.isna(contrib) else "N/A"
                components_html += f"<tr><td>{name}</td><td>{best_row[col]:.3f}</td><td>{weight}</td><td>{contrib_str}</td></tr>"

    # Red flags section - count UNIQUE COMPOUNDS (not all rows)
    red_flags_html = ""
    flag_cols = [
        ('PAINS_Violation', 'PAINS', 'Pan-Assay Interference'),
        ('Aggregator_Risk', 'Aggregator', 'Colloidal Aggregation'),
        ('Redox_Reactive', 'Redox', 'Redox Cycling'),
        ('Fluorescence_Interference', 'Fluorescence', 'Fluorescence Interference'),
        ('Thiol_Reactive', 'Thiol', 'Thiol Reactivity'),
        ('BRENK_Alerts', 'BRENK', 'Unwanted Substructures'),
        ('NIH_Alerts', 'NIH', 'NIH Problematic Groups'),
    ]
    total_flags = 0
    unique_df_flags = df.drop_duplicates('ChEMBL_ID') if 'ChEMBL_ID' in df.columns else df
    for col, name, desc in flag_cols:
        if col in unique_df_flags.columns:
            count = int(unique_df_flags[col].sum() if unique_df_flags[col].dtype == bool else unique_df_flags[col].astype(bool).sum())
            total_flags += count
            status = f"⚠️ {count} flagged" if count > 0 else "✓ Clean"
            color_style = "color: #dc3545;" if count > 0 else "color: #28a745;"
            red_flags_html += f"<tr><td>{name}</td><td style='{color_style}'>{status}</td><td>{desc}</td></tr>"

    flag_assessment = "LOW CONCERN" if total_flags == 0 else ("MODERATE CONCERN" if total_flags <= 5 else "HIGH CONCERN")
    flag_color = "#28a745" if total_flags == 0 else ("#fd7e14" if total_flags <= 5 else "#dc3545")

    # Bioactivity distribution
    bioactivity_html = ""
    if 'Activity_Type' in df.columns:
        type_counts = df['Activity_Type'].value_counts().head(5)
        for stype, count in type_counts.items():
            pct = count / len(df) * 100
            bioactivity_html += f"<tr><td>{html.escape(str(stype))}</td><td>{count}</td><td>{pct:.1f}%</td></tr>"

    # Efficiency metrics statistics
    efficiency_metrics_stats = {}
    metric_colors_html = {
        'SEI': '#1f77b4',
        'BEI': '#2ca02c',
        'NSEI': '#ff7f0e',
        'NBEI': '#9467bd'
    }
    for metric in ['SEI', 'BEI', 'NSEI', 'NBEI']:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                efficiency_metrics_stats[metric] = {
                    'mean': vals.mean(),
                    'min': vals.min(),
                    'max': vals.max(),
                    'used': metric in ['SEI', 'BEI']
                }

    # Efficiency plane summary - compute from full dataset
    mean_sei = df['SEI'].mean() if 'SEI' in df.columns else None
    mean_bei = df['BEI'].mean() if 'BEI' in df.columns else None

    if mean_sei is not None and mean_bei is not None and not pd.isna(mean_sei) and not pd.isna(mean_bei):
        angle_val = np.degrees(np.arctan2(mean_bei, mean_sei))
        modulus_val = np.sqrt(mean_sei**2 + mean_bei**2)
    else:
        angle_val = best_row['Angle_SEI_BEI'] if best_row is not None and 'Angle_SEI_BEI' in best_row.index else None
        modulus_val = best_row['Modulus_SEI_BEI'] if best_row is not None and 'Modulus_SEI_BEI' in best_row.index else None

    # Pre-compute formatted strings for angle and modulus
    angle_str = f"{angle_val:.1f}" if angle_val is not None and not pd.isna(angle_val) else "N/A"
    modulus_str = f"{modulus_val:.2f}" if modulus_val is not None and not pd.isna(modulus_val) else "N/A"
    mean_sei_str = f"{mean_sei:.2f}" if mean_sei is not None and not pd.isna(mean_sei) else "N/A"
    mean_bei_str = f"{mean_bei:.2f}" if mean_bei is not None and not pd.isna(mean_bei) else "N/A"

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
        entry_id = data.get('entry_id')
        storage_path = data.get('storage_path')
        for filename in ["pdb_summary.csv", f"{safe_name}_pdb_summary.csv", f"{safe_name}_pdb_details.csv"]:
            pdb_summary_df_html = smart_load_dataframe(
                filename,
                entry_id=entry_id,
                storage_path=storage_path
            )
            if pdb_summary_df_html is not None and not pdb_summary_df_html.empty:
                break
    except Exception:
        pdb_summary_df_html = None

    if pdb_summary_df_html is not None and not pdb_summary_df_html.empty:
        pdb_total = len(pdb_summary_df_html)
        if 'Quality' in pdb_summary_df_html.columns:
            high_q = int((pdb_summary_df_html['Quality'] == '***').sum())
            med_q = int((pdb_summary_df_html['Quality'] == '**').sum())
            poor_q = int((pdb_summary_df_html['Quality'] == '*').sum())
        elif 'Resolution' in pdb_summary_df_html.columns:
            pdb_summary_df_html['_res'] = pd.to_numeric(pdb_summary_df_html['Resolution'], errors='coerce')
            high_q = int((pdb_summary_df_html['_res'] < 2.0).sum())
            med_q = int(((pdb_summary_df_html['_res'] >= 2.0) & (pdb_summary_df_html['_res'] <= 3.0)).sum())
            poor_q = int((pdb_summary_df_html['_res'] > 3.0).sum())
    else:
        # Fallback to dataframe columns
        if 'PDB_Num_Structures' in df.columns:
            pdb_total = int(df['PDB_Num_Structures'].max()) if df['PDB_Num_Structures'].notna().any() else 0
        if 'PDB_High_Quality' in df.columns and 'PDB_Medium_Quality' in df.columns and 'PDB_Poor_Quality' in df.columns:
            high_q = int(df['PDB_High_Quality'].max()) if df['PDB_High_Quality'].notna().any() else 0
            med_q = int(df['PDB_Medium_Quality'].max()) if df['PDB_Medium_Quality'].notna().any() else 0
            poor_q = int(df['PDB_Poor_Quality'].max()) if df['PDB_Poor_Quality'].notna().any() else 0

    # Calculate PDB confidence
    mean_pdb_score = df['PDB_Score'].mean() if 'PDB_Score' in df.columns and df['PDB_Score'].notna().any() else 0
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

    # Classification - ClassyFire and NPClassifier
    classyfire_html = ""
    class_cols = ['Kingdom', 'Superclass', 'Class', 'Subclass']
    for col in class_cols:
        if col in df.columns and df[col].notna().any():
            val = html.escape(str(df[col].iloc[0]))
            classyfire_html += f"<tr><td>{col}</td><td>{val}</td></tr>"

    # NPClassifier
    npclassifier_html = ""
    np_cols = ['NP_Pathway', 'NP_Superclass', 'NP_Class']
    np_labels = ['Pathway', 'Superclass', 'Class']
    for col, label in zip(np_cols, np_labels):
        if col in df.columns and df[col].notna().any():
            val = html.escape(str(df[col].iloc[0]))
            npclassifier_html += f"<tr><td>{label}</td><td>{val}</td></tr>"

    # Drug indications
    indications_html = ""
    max_clinical_phase = 0
    unique_indications = 0
    compounds_with_indications = 0

    indications_df = data.get('indications_df')
    if indications_df is not None and not indications_df.empty:
        # Calculate summary statistics
        max_clinical_phase = indications_df['Max_Phase'].max() if 'Max_Phase' in indications_df.columns else 0
        unique_indications = indications_df['MESH_Heading'].nunique() if 'MESH_Heading' in indications_df.columns else 0
        compounds_with_indications = indications_df['ChEMBL_ID'].nunique() if 'ChEMBL_ID' in indications_df.columns else len(df)

        # Get top indications
        top_indications = indications_df.groupby('MESH_Heading')['Max_Phase'].max().sort_values(ascending=False).head(10)
        for indication, phase in top_indications.items():
            indications_html += f"<tr><td>{html.escape(str(indication))}</td><td>Phase {int(phase)}</td></tr>"

    # Get base score and QED multiplier from best compound
    base_score = best_row['IMP_Base_Score'] if best_row is not None and 'IMP_Base_Score' in best_row.index else None
    qed_mult = best_row['QED_Multiplier'] if best_row is not None and 'QED_Multiplier' in best_row.index else None

    # Pre-compute formatted strings (can't use conditionals inside f-string format specifiers)
    base_score_str = f"{base_score:.3f}" if base_score is not None and not pd.isna(base_score) else "N/A"
    qed_mult_str = f"{qed_mult:.3f}" if qed_mult is not None and not pd.isna(qed_mult) else "N/A"
    qed_val_str = f"{qed_val:.3f}" if qed_val is not None and not pd.isna(qed_val) else "N/A"
    final_score_str = f"{final_score:.3f}" if final_score is not None and not pd.isna(final_score) else "N/A"

    # Component score values for formula display
    def _fmt_cs(col):
        v = best_row[col] if best_row is not None and col in best_row.index else None
        return f"{v:.3f}" if v is not None and not pd.isna(v) else "N/A"
    eff_s_str = _fmt_cs('Efficiency_Score')
    dist_s_str = _fmt_cs('Distance_Score')
    ang_s_str = _fmt_cs('Angle_Score')
    int_s_str = _fmt_cs('Interference_Score')
    pdb_s_str = _fmt_cs('PDB_Score')

    # Escape compound name for HTML
    safe_compound_name = html.escape(compound_name)
    smiles_display = html.escape(smiles[:60] + '...' if len(smiles) > 60 else smiles)

    # Priority text
    priority_text = f"Priority {interpretation['priority']}" if interpretation['priority'] else "N/A"
    warning_icon = '⚠️' if final_score >= 0.5 else '✓'

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
        .verdict {{ background-color: {bg_color}; border-left: 5px solid {color}; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .verdict h3 {{ color: {color}; margin: 0 0 10px 0; }}
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
            <p><strong>InChI:</strong> <code style="word-break: break-all; font-size: 0.85em;">{inchi}</code></p>
            <p><strong>SMILES:</strong> <code>{smiles_display}</code></p>
            <p><strong>Analysis Date:</strong> {summary.get('processing_date', 'N/A')}</p>
            <p><strong>Author:</strong> {html.escape(data.get('author_name', 'N/A'))}</p>
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
    </div>

    <!-- Summary Stats Row -->
    <div style="display: flex; justify-content: space-around; background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">Similar Compounds</div>
            <div style="font-size: 1.8em; font-weight: bold;">{similar_count}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">Activities</div>
            <div style="font-size: 1.8em; font-weight: bold;">{activities_count}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">QED</div>
            <div style="font-size: 1.8em; font-weight: bold;">{avg_qed_str}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.9em; color: #666;">IMP Score</div>
            <div style="font-size: 1.8em; font-weight: bold;">{avg_imp_score_str}</div>
        </div>
        <div style="text-align: center; background-color: {'#721c24' if has_warning and imp_count > 0 else '#28a745'}; color: white; padding: 10px 20px; border-radius: 5px;">
            <div style="font-size: 0.8em;">{'⚠️' if has_warning and imp_count > 0 else '✓'}</div>
            <div style="font-size: 1.5em; font-weight: bold;">{imp_count if has_warning else 0} IMP</div>
            <div style="font-size: 0.65em;">unique compounds (outlier detection)</div>
        </div>
    </div>

    <!-- 2. EXECUTIVE SUMMARY -->
    <h2>📊 Executive Summary</h2>
    <div class="verdict">
        <h3>{warning_icon} {interpretation['label'].upper()} - {priority_text}</h3>
        <p><strong>IMP Score:</strong> {final_score:.3f} | <strong>QED:</strong> {qed_val:.3f} | <strong>Red Flags:</strong> {total_flags} active</p>
        <p><strong>Risk Level:</strong> {interpretation['risk']}</p>
        <p><strong>Recommended Action:</strong> {interpretation['action']}</p>
    </div>
    {"<div class='warning'><strong>⚠️ HIGH FALSE POSITIVE RISK:</strong> This compound shows strong evidence of being an assay artifact. DEPRIORITIZE unless validated with orthogonal assays.</div>" if final_score >= 0.7 else ""}

    <!-- 3. COMPOUND PROPERTIES -->
    <h2>🧪 Compound Properties</h2>
    <table>
        <tr><th>Property</th><th>Value</th><th>Unit/Description</th></tr>
        {props_html if props_html else "<tr><td colspan='3'>No property data available</td></tr>"}
    </table>

    <!-- 4. EFFICIENCY METRICS -->
    <h2>📈 Efficiency Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Formula</th><th>Used in Score</th></tr>
        {efficiency_html if efficiency_html else "<tr><td colspan='4'>No efficiency data available</td></tr>"}
    </table>
    <p><em>Only SEI and BEI contribute to the Efficiency Score. NSEI/NBEI are for reference.</em></p>

    <!-- 5. IMP SCORE CALCULATION -->
    <h2>🔢 IMP Score Calculation</h2>
    <h3>Component Scores</h3>
    <table>
        <tr><th>Component</th><th>Score</th><th>Weight</th><th>Contribution</th></tr>
        {components_html if components_html else "<tr><td colspan='5'>No component data available</td></tr>"}
    </table>

    <h3>Final Calculation</h3>
    <div class="calc-box">
<strong>Base Score</strong> = 0.45×Eff + 0.20×Dist + 0.15×Angle + 0.15×Interf + 0.05×PDB
         = 0.45×{eff_s_str} + 0.20×{dist_s_str} + 0.15×{ang_s_str} + 0.15×{int_s_str} + 0.05×{pdb_s_str}
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
    <div style="background-color: {'#d4edda' if total_flags == 0 else ('#fff3cd' if total_flags <= 5 else '#f8d7da')}; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid {flag_color};">
        <strong style="color: {flag_color};">{flag_assessment} - {total_flags} flags detected</strong>
    </div>
    <table>
        <tr><th>Flag Type</th><th>Status</th><th>Description</th></tr>
        {red_flags_html if red_flags_html else "<tr><td colspan='3'>No flag data available</td></tr>"}
    </table>

    <!-- 7. BIOACTIVITY DISTRIBUTION -->
    <h2>🎯 Bioactivity Distribution</h2>
    <div class="two-col">
        <div>{bioactivity_chart_html}</div>
        <div>
            <table>
                <tr><th>Activity Type</th><th>Count</th><th>Percentage</th></tr>
                {bioactivity_html if bioactivity_html else "<tr><td colspan='3'>No bioactivity data available</td></tr>"}
            </table>
            <p><strong>Total Activities:</strong> {len(df)}</p>
        </div>
    </div>

    <!-- 8. EFFICIENCY METRICS -->
    <h2>📈 Efficiency Metrics Distribution</h2>

    <!-- Metric Cards -->
    <div style="display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap;">
        {"".join([f'''
        <div style="flex: 1; min-width: 200px; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid {metric_colors_html[metric]};">
            <div style="color: #fff; font-size: 0.95em; margin-bottom: 5px;">{metric}{"" if not stats["used"] else " ✓"}</div>
            <div style="font-size: 1.5em; color: {metric_colors_html[metric]}; font-weight: bold;">{stats["mean"]:.2f}</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Mean Value</div>
            <div style="color: #888; font-size: 0.75em; margin-top: 3px;">Range: {stats["min"]:.1f}-{stats["max"]:.1f}</div>
        </div>
        ''' for metric, stats in efficiency_metrics_stats.items()])}
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
    <div style="background-color: {angle_status_bg}; padding: 12px; border-radius: 5px; margin-bottom: 15px; border: 1px solid {angle_status_color};">
        <strong style="color: {angle_status_color};">Development Trajectory: {angle_status} (Angle: {angle_str}°)</strong>
    </div>

    <!-- Metric Cards Row -->
    <div style="display: flex; gap: 15px; margin-bottom: 15px;">
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #636EFA;">
            <div style="font-size: 1.4em; color: #636EFA; font-weight: bold;">{angle_str}°</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Mean Angle</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Development trajectory</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #00CC96;">
            <div style="font-size: 1.4em; color: #00CC96; font-weight: bold;">{modulus_str}</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Modulus</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Overall efficiency</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #EF553B;">
            <div style="font-size: 1.4em; color: #EF553B; font-weight: bold;">{mean_sei_str}</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Mean SEI</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">Surface efficiency</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #FFA15A;">
            <div style="font-size: 1.4em; color: #FFA15A; font-weight: bold;">{mean_bei_str}</div>
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

    <!-- 10. PDB EVIDENCE -->
    <h2>🔬 PDB Structural Evidence</h2>

    <!-- Confidence Banner -->
    <div style="background-color: {pdb_conf_bg}; padding: 12px; border-radius: 5px; margin-bottom: 15px; border: 1px solid {pdb_conf_color};">
        <strong style="color: {pdb_conf_color};">{pdb_conf_icon} Structural Validation: {pdb_confidence} (PDB Score: {mean_pdb_score:.3f})</strong>
    </div>

    <!-- Quality Distribution Cards -->
    <div style="display: flex; gap: 15px; margin-bottom: 15px;">
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #636EFA;">
            <div style="font-size: 1.5em; color: #636EFA; font-weight: bold;">{pdb_total}</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Total Structures</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">PDB entries found</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #28a745;">
            <div style="font-size: 1.5em; color: #28a745; font-weight: bold;">{high_q}</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">High Quality</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">&lt;2.0Å ({high_q_pct:.0f}%) ★★★★★</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #fd7e14;">
            <div style="font-size: 1.5em; color: #fd7e14; font-weight: bold;">{med_q}</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Medium Quality</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">2-3Å ({med_q_pct:.0f}%) ★★★</div>
        </div>
        <div style="flex: 1; text-align: center; padding: 15px; background: #2d2d2d; border-radius: 8px; border-left: 4px solid #dc3545;">
            <div style="font-size: 1.5em; color: #dc3545; font-weight: bold;">{poor_q}</div>
            <div style="color: #fff; font-size: 0.95em; margin-top: 5px;">Poor Quality</div>
            <div style="color: #aaa; font-size: 0.8em; margin-top: 3px;">&gt;3Å ({poor_q_pct:.0f}%) ★</div>
        </div>
    </div>

    <!-- Quality Distribution Chart (Full Width) -->
    <div style="margin-bottom: 20px;">
        {pdb_quality_chart_html}
    </div>

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
                {classyfire_html if classyfire_html else "<tr><td colspan='2'>No ClassyFire data available</td></tr>"}
            </table>
        </div>

        <!-- NPClassifier (Natural Products) -->
        <div>
            <h3>NPClassifier (Natural Products):</h3>
            <table>
                <tr><th>Level</th><th>Classification</th></tr>
                {npclassifier_html if npclassifier_html else "<tr><td colspan='2'>No NPClassifier data available</td></tr>"}
            </table>
        </div>
    </div>

    <!-- 12. DRUG INDICATIONS -->
    <h2>💊 Drug Indications</h2>

    {f'''
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
    ''' if indications_html else "<p>No drug indication data available</p>"}

    <!-- 13. FINAL RECOMMENDATION -->
    <h2>🎯 Final Recommendation</h2>
    <div class="verdict">
        <h3>{warning_icon} VERDICT: {interpretation['label'].upper()}</h3>
        <p><strong>{interpretation['risk']}</strong></p>
        <p>{interpretation['action']}</p>
    </div>

    <!-- IMP GUIDE -->
    <h2>📚 IMP Interpretation Guide</h2>
    <div class="guide">
        <table>
            <tr><th>Score Range</th><th>Classification</th><th>Meaning</th><th>Action</th></tr>
            <tr><td>0.90+</td><td>Exceptional IMP</td><td>⚠️ VERY HIGH false positive risk</td><td>Immediate validation</td></tr>
            <tr><td>0.70-0.89</td><td>Strong IMP</td><td>⚠️ HIGH false positive risk</td><td>DEPRIORITIZE unless validated</td></tr>
            <tr><td>0.50-0.69</td><td>Moderate IMP</td><td>⚠️ Moderate risk</td><td>Validate carefully</td></tr>
            <tr><td>0.30-0.49</td><td>Weak IMP</td><td>Lower risk - more likely genuine</td><td>Standard follow-up</td></tr>
            <tr><td>&lt; 0.30</td><td>Not IMP</td><td>✓ Likely genuine activity</td><td>Proceed with development</td></tr>
        </table>
        <p style="margin-top: 15px;"><em><strong>IMP = Invalid Metabolic Panacea = Likely FALSE POSITIVE (assay artifact)</strong></em></p>
    </div>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #667eea; color: #666; font-size: 12px; text-align: center;">
        <p>Generated by <strong>IMPULATOR</strong> | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p>💡 <em>Tip: Use Ctrl+P (Cmd+P on Mac) to print this report to PDF</em></p>
    </footer>
</body>
</html>
"""
    return html_content
