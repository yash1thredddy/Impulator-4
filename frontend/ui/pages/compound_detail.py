"""Compound detail page for IMPULATOR.

Displays full analysis results with improved UX and organization.
"""

import html
import logging
import re
from typing import Dict, Any, Optional, List
from urllib.parse import quote_plus

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from frontend.services import (
    get_api_client,
    delete_from_cache,
    load_result_dataframe,
    get_cached_result,
    get_result_files,
)
from frontend.utils import SessionState, sanitize_compound_name
from frontend.ui.components import render_2d_structure, embed_structure_viewer, render_structure_viewer_hint
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)


def render_compound_detail_page() -> None:
    """Render the compound detail page."""
    compound_name = SessionState.get('selected_compound')
    entry_id = SessionState.get('selected_compound_entry_id')

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

    # Load data using entry_id if available (UUID-based storage), otherwise compound_name
    data = _load_compound_data(entry_id or compound_name)
    if data is None:
        st.error(f"Could not load data for '{compound_name}'")
        return

    # Quick stats row
    _render_quick_stats(data)

    # Main content with tabs
    tabs = st.tabs(["📊 Overview", "📈 Visualizations", "🧬 Molecules", "📋 Data"])

    with tabs[0]:
        _render_overview_tab(data)

    with tabs[1]:
        _render_visualizations_tab(data)

    with tabs[2]:
        _render_structures_tab(data)

    with tabs[3]:
        _render_data_tab(data)


def _render_quick_stats(data: Dict[str, Any]) -> None:
    """Render compact stats bar."""
    df = data.get('results')
    summary = data.get('summary', {})

    cols = st.columns(5)

    similar = summary.get('similar_count', 0)
    activities = summary.get('total_activities', len(df) if df is not None else 0)
    qed = summary.get('qed', 0)
    imp_count = summary.get('imp_candidates', 0)
    has_warning = summary.get('has_imp_candidates', False)

    with cols[0]:
        st.metric("Similar Compounds", similar)
    with cols[1]:
        st.metric("Activities", activities)
    with cols[2]:
        st.metric("QED", f"{qed:.2f}" if qed else "N/A")
    with cols[3]:
        oqpla = None
        if df is not None and 'OQPLA_Final_Score' in df.columns:
            oqpla = df['OQPLA_Final_Score'].mean()
        st.metric("Avg OQPLA", f"{oqpla:.2f}" if pd.notna(oqpla) else "N/A")
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

    # Sub-tabs for overview sections
    sub_tabs = st.tabs([
        "🧪 Compound",
        "🔢 Properties",
        "📈 Activity",
        "🎯 Efficiency",
        "🔬 PDB Evidence",
        "⚠️ Assay Interference",
        "🔍 IMP/OQPLA",
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
        _render_pdb_evidence(compound_name, df)

    # PAINS/Assay Interference (separated)
    with sub_tabs[5]:
        _render_pains_analysis(df)

    # IMP/OQPLA Analysis (without PAINS)
    with sub_tabs[6]:
        _render_oqpla_analysis(df, compound_name)

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
            st.code(data.get('smiles', 'N/A')[:60] + "..." if len(data.get('smiles', '')) > 60 else data.get('smiles', 'N/A'), language=None)

            # Calculate and display InChI and InChIKey from SMILES
            smiles = data.get('smiles', '')
            if smiles:
                try:
                    from rdkit import Chem
                    from rdkit.Chem.inchi import MolToInchi, MolToInchiKey
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        inchi = MolToInchi(mol)
                        inchikey = MolToInchiKey(mol)
                        if inchikey:
                            st.markdown("**InChIKey**")
                            st.code(inchikey, language=None)
                        if inchi:
                            with st.expander("📋 View InChI", expanded=False):
                                st.code(inchi, language=None)
                except Exception:
                    pass  # Skip if RDKit not available

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
        'OQPLA_Final_Score', 'OQPLA_Grade', 'O_Score', 'Q_Score', 'P_Score', 'L_Score', 'A_Score'
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

                # QED Multiplier (from OQPLA scoring: 0.5 + 0.5*QED)
                if 'QED_Multiplier' in unique_df.columns:
                    vals = unique_df['QED_Multiplier'].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        color = "🟢" if mean_val >= 0.75 else "🟡" if mean_val >= 0.65 else "🔴"
                        st.metric(f"{color} QED Multiplier", f"{mean_val:.3f}",
                                  help="OQPLA QED multiplier (0.5 + 0.5×QED)")
                        metrics_shown = True

                # QED Impact (from OQPLA scoring)
                if 'QED_Impact' in unique_df.columns:
                    vals = unique_df['QED_Impact'].dropna()
                    if len(vals) > 0:
                        mean_val = vals.mean()
                        color = "🟢" if mean_val >= -0.1 else "🟡" if mean_val >= -0.2 else "🔴"
                        st.metric(f"{color} QED Impact", f"{mean_val:.3f}",
                                  help="QED penalty on OQPLA score (0=best)")
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
                        title='MW vs LogP (Lipinski Space)',
                        color_continuous_scale='RdYlGn'
                    )
                    # Add Lipinski rule boundaries
                    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="LogP ≤ 5")
                    fig.add_vline(x=500, line_dash="dash", line_color="red", annotation_text="MW ≤ 500")
                    fig.update_layout(height=300, margin=dict(t=40, b=30, l=30, r=30))
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
                fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="HBD+HBA ≤ 10")
                fig.add_vline(x=140, line_dash="dash", line_color="red", annotation_text="TPSA ≤ 140")
                fig.update_layout(height=300, margin=dict(t=40, b=30, l=30, r=30))
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
                    title='TPSA vs QED (Drug-likeness)',
                    color_continuous_scale='RdYlGn_r'  # Reversed: lower violations = greener
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Good QED (≥0.5)")
                fig.add_vline(x=140, line_dash="dash", line_color="red", annotation_text="TPSA ≤ 140")
                fig.update_layout(height=300, margin=dict(t=40, b=30, l=30, r=30))
                st.plotly_chart(fig, width='stretch')

        with viz_col4:
            # QED distribution
            if has_qed:
                qed_vals = unique_df['QED'].dropna()
                fig = px.histogram(qed_vals, nbins=20, title='QED Score Distribution',
                                   color_discrete_sequence=['#636EFA'])
                fig.add_vline(x=0.5, line_dash="dash", line_color="green", annotation_text="Good (≥0.5)")
                fig.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Moderate (≥0.3)")
                fig.update_layout(height=300, margin=dict(t=40, b=30, l=30, r=30))
                st.plotly_chart(fig, width='stretch')

    else:
        # Individual compounds view - PubChem-style table
        st.markdown("**Computed Properties by Compound**")
        st.caption("Each row represents one compound with its computed properties (like PubChem format)")

        # Define key properties to show (PubChem-like order)
        key_props = ['Molecular_Weight', 'MolLogP', 'LogP', 'TPSA', 'HBD', 'HBA',
                     'Heavy_Atoms', 'Rotatable_Bonds', 'Aromatic_Rings', 'NPOL',
                     'QED', 'RO5_Violations', 'NP_Likeness_Score']

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
            margin=dict(t=30, b=30, l=30, r=30),
            height=350,
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
            height=min(350, len(target_counts) * 35 + 50),
            margin=dict(t=10, b=10, l=10, r=10),
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
        categorical_cols = ['Activity_Type', 'ChEMBL_ID', 'OQPLA_Classification', 'Target_Name']
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

    # Two-column layout for chart and outliers
    col1, col2 = st.columns([3, 1])

    with col1:
        # Box plot with optional coloring
        if color_by != "None":
            fig = px.box(
                plot_df,
                x=color_by,
                y=metric_choice,
                color=color_by,
                points='outliers',
                hover_data=['ChEMBL_ID', 'Molecule_Name'] if all(c in plot_df.columns for c in ['ChEMBL_ID', 'Molecule_Name']) else None
            )
            fig.update_layout(
                height=400,
                margin=dict(t=30, b=80),
                xaxis_tickangle=-45,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_text=""
                )
            )
            st.plotly_chart(fig, width='stretch')
            st.caption("💡 **Click legend items** to show/hide groups. Double-click to isolate one group.")
        else:
            # Simple histogram without grouping
            fig = px.histogram(plot_df, x=metric_choice, nbins=30)
            fig.update_layout(height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig, width='stretch')

    with col2:
        # Outlier summary
        st.markdown("**Outliers Detected**")
        for m in avail:
            outlier_col = f'Is_{m}_Outlier'
            if outlier_col in df.columns:
                count = df[outlier_col].sum()
                if count > 0:
                    st.warning(f"{m}: {int(count)}")
                else:
                    st.success(f"{m}: 0")

        # Show count when colored
        if color_by != "None":
            st.markdown("---")
            st.markdown(f"**Groups ({color_by})**")
            group_counts = plot_df[color_by].value_counts()
            for grp, cnt in group_counts.head(6).items():
                st.caption(f"{grp[:15]}: {cnt}")

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
            st.dataframe(target_metrics_df, width='stretch', hide_index=True, height=300)

    # Explanation box
    with st.expander("📖 Understanding Efficiency Metrics by Target", expanded=False):
        st.markdown("""
**This table shows efficiency metrics calculated for each target in the dataset.**

- **SEI (Surface Efficiency Index)**: Measures activity relative to polar surface area. Formula: pActivity / PSA × 100
- **BEI (Binding Efficiency Index)**: Measures activity relative to molecular weight. Formula: pActivity / MW × 1000
- **NSEI (Normalized Surface Efficiency Index)**: SEI normalized by the number of polar atoms (N + O atoms)
- **nBEI (Normalized Binding Efficiency Index)**: BEI normalized considering heavy atoms

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

    # Summary metrics row
    flags = {
        'PAINS': ('PAINS_Violation', '🔴', 'Pan-Assay Interference'),
        'Aggregator': ('Aggregator_Risk', '🟠', 'Colloidal Aggregation'),
        'Redox': ('Redox_Reactive', '🟡', 'Redox Cycling'),
        'Fluorescence': ('Fluorescence_Interference', '🔵', 'Fluorescence Interference'),
        'Thiol': ('Thiol_Reactive', '🟣', 'Thiol Reactivity')
    }

    # Display metrics in a row
    metric_cols = st.columns(5)
    flag_data = []

    for i, (name, (col, emoji, desc)) in enumerate(flags.items()):
        if col in unique_df.columns:
            count = int(unique_df[col].sum())
            pct = count / total * 100 if total > 0 else 0
            flag_data.append({
                'Flag': f"{emoji} {name}",
                'Count': count,
                '%': f"{pct:.0f}%",
                'Description': desc
            })
            with metric_cols[i]:
                if count > 0:
                    st.metric(name, count, delta=f"{pct:.0f}%", delta_color="inverse")
                else:
                    st.metric(name, "0", delta="Clean", delta_color="normal")

    st.markdown("---")

    # Detailed table
    if flag_data:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Flag Summary**")
            st.dataframe(pd.DataFrame(flag_data), width='stretch', hide_index=True, height=220)

        with col2:
            # Show compounds with any flag
            flagged_compounds = []
            for name, (col, _, _) in flags.items():
                if col in unique_df.columns:
                    flagged = unique_df[unique_df[col] == True]
                    for _, row in flagged.iterrows():
                        mol_name = row.get('Molecule_Name', '')
                        # Handle NaN/float values
                        if pd.isna(mol_name) or not isinstance(mol_name, str):
                            mol_name = ''
                        flagged_compounds.append({
                            'ChEMBL_ID': row.get('ChEMBL_ID', 'Unknown'),
                            'Flag': name,
                            'Molecule': mol_name[:20] if mol_name else ''
                        })

            if flagged_compounds:
                st.markdown("**Flagged Compounds**")
                flagged_df = pd.DataFrame(flagged_compounds).drop_duplicates()
                st.dataframe(flagged_df.head(15), width='stretch', hide_index=True, height=220)
            else:
                st.success("✓ No compounds flagged for assay interference")

    # PAINS patterns breakdown (if available)
    if 'PAINS_Pattern' in unique_df.columns:
        st.markdown("---")
        st.markdown("**PAINS Patterns Detected**")
        patterns = unique_df[unique_df['PAINS_Pattern'].notna()]['PAINS_Pattern'].value_counts()
        if not patterns.empty:
            fig = px.bar(x=patterns.values, y=patterns.index, orientation='h')
            fig.update_layout(height=min(250, len(patterns) * 30 + 50), margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, width='stretch')

    # Important interpretation note at the bottom
    st.markdown("---")
    st.info("""
💡 **Important Note:** These flags identify compounds with known assay interference mechanisms (PAINS, aggregation, redox activity, fluorescence, thiol reactivity). However, **flags do NOT automatically disqualify compounds**. Many flagged compounds (e.g., quercetin with catechol groups) exhibit genuine polypharmacology validated by extensive PDB structural evidence.

**Interpretation:** Use PDB scores and structural evidence to distinguish genuine multi-target binders from assay artifacts. **High O[Q/P/L]A scores + interference flags + high PDB scores = likely genuine polypharmacology.**
    """)


def _render_oqpla_analysis(df: pd.DataFrame, compound_name: str) -> None:
    """IMP and OQPLA analysis with full explanations."""
    if df is None:
        st.info("No data available")
        return

    has_oqpla = 'OQPLA_Final_Score' in df.columns
    has_imp = 'Is_IMP_Candidate' in df.columns

    if not (has_oqpla or has_imp):
        st.info("No OQPLA/IMP analysis data available")
        return

    # OQPLA Explanation
    with st.expander("📖 What is O[Q/P/L]A Scoring?", expanded=False):
        st.markdown("""
**O[Q/P/L]A (Overall Quality/Promise/Likelihood Assessment)** is a multi-criteria scoring system for evaluating compound quality and IMP (Invalid Metabolic Panacea) likelihood.

**Scoring Components:**
1. **Efficiency Outlier Score (40%)** - How exceptional are the compound's efficiency metrics (SEI, BEI, NSEI, NBEI) compared to the cohort? Higher outlier scores suggest unusual activity.
2. **Development Angle Score (10%)** - Is the compound balanced between surface and binding efficiency? An angle of 45° is optimal.
3. **Distance to Best-in-Class (15%)** - How close is the compound to the best performer in the dataset?
4. **PDB Structural Evidence (15%)** - Does experimental crystallography data support the binding? High-resolution structures (< 2.0 Å) provide strongest evidence.
5. **QED Multiplier** - Drug-likeness adjustment based on Quantitative Estimate of Drug-likeness.

**Score Interpretation:**
- **0.0 - 0.3**: Low quality - likely noise or artifacts
- **0.3 - 0.5**: Moderate quality - needs further validation
- **0.5 - 0.7**: Good quality - promising candidates
- **0.7 - 1.0**: Excellent quality - strong candidates with structural support

**IMP Classification:**
- **Not IMP**: Compounds showing normal behavior within expected ranges
- **Weak IMP**: Compounds with some outlier characteristics (requires attention)
- **Moderate IMP**: Compounds with multiple outlier flags (high suspicion)
- **Strong IMP**: Compounds with extreme outlier behavior (likely false positives)
        """)

    # OQPLA Scoring
    if has_oqpla:
        st.markdown("**O[Q/P/L]A Score Distribution**")

        score_cols = st.columns(4)
        scores = df['OQPLA_Final_Score'].dropna()
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
            high_quality = len(scores[scores >= 0.5]) if len(scores) > 0 else 0
            st.metric("High Quality (≥0.5)", high_quality)

        # Score histogram with better styling
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(df, x='OQPLA_Final_Score', nbins=25, color_discrete_sequence=['#636EFA'])
            fig.update_layout(
                height=280,
                margin=dict(t=10, b=30, l=30, r=10),
                xaxis_title="O[Q/P/L]A Score",
                yaxis_title="Count"
            )
            # Add vertical lines for thresholds
            fig.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Low/Moderate")
            fig.add_vline(x=0.5, line_dash="dash", line_color="green", annotation_text="Good")
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Classification breakdown with color coding
            if 'OQPLA_Classification' in df.columns:
                st.markdown("**Quality Classification**")
                class_counts = df['OQPLA_Classification'].value_counts()
                for cls, count in class_counts.items():
                    pct = count / len(df) * 100
                    # Color code based on classification
                    if 'Not IMP' in str(cls) or 'Excellent' in str(cls) or 'Good' in str(cls):
                        st.success(f"**{cls}**: {count} ({pct:.0f}%)")
                    elif 'Weak' in str(cls) or 'Moderate' in str(cls):
                        st.warning(f"**{cls}**: {count} ({pct:.0f}%)")
                    else:
                        st.error(f"**{cls}**: {count} ({pct:.0f}%)")

    # IMP Candidates section
    if has_imp:
        st.markdown("---")
        st.markdown("**IMP Candidates Analysis**")
        st.caption("Invalid Metabolic Panaceas (IMPs) are compounds that appear exceptionally active but may be assay artifacts")

        # Get IMP candidate records and unique compounds
        imp_df = df[df['Is_IMP_Candidate'] == True]
        unique_imp_compounds = imp_df.drop_duplicates('ChEMBL_ID') if 'ChEMBL_ID' in imp_df.columns else imp_df
        total_unique = df.drop_duplicates('ChEMBL_ID')['ChEMBL_ID'].nunique() if 'ChEMBL_ID' in df.columns else len(df)

        # Also count OQPLA-classified weak/moderate IMP records for context
        oqpla_imp_records = 0
        if 'OQPLA_Classification' in df.columns:
            oqpla_imp_records = len(df[df['OQPLA_Classification'].str.contains('IMP', case=False, na=False)])

        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("IMP Candidates", len(unique_imp_compounds), help="Unique compounds flagged as potential IMPs based on multiple criteria")
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
            st.metric("Affected Records", len(imp_df), help=f"Activity records from IMP compounds (OQPLA flagged: {oqpla_imp_records})")

        if not unique_imp_compounds.empty:
            st.markdown("**IMP Candidates with Target Mapping:**")

            # Build display table - one row per compound+target combination
            display_data = []
            for _, row in unique_imp_compounds.iterrows():
                chembl_id = row.get('ChEMBL_ID', 'Unknown')
                mol_name = row.get('Molecule_Name', '')
                if pd.isna(mol_name) or not isinstance(mol_name, str):
                    mol_name = ''

                oqpla_score = round(row.get('OQPLA_Final_Score', 0), 3) if pd.notna(row.get('OQPLA_Final_Score')) else 'N/A'
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
                                'OQPLA': oqpla_score,
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
                            'OQPLA': oqpla_score,
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
                        'OQPLA': oqpla_score,
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
                    'OQPLA': st.column_config.TextColumn('OQPLA', width='small'),
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
        color='Activity_Type', points='outliers',
        hover_data=['ChEMBL_ID', 'Molecule_Name'] if all(c in plot_df.columns for c in ['ChEMBL_ID', 'Molecule_Name']) else None,
        custom_data=customdata_cols if customdata_cols else None
    )
    fig.update_layout(
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text=""
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
            from scipy import stats as scipy_stats
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
                st.markdown(f"**Equation:**")
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
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text=""
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
            from scipy import stats as scipy_stats
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
                st.markdown(f"**Equation:**")
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
                points="outliers",
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
                box=True, points="outliers",
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
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
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
            if 'OQPLA_Final_Score' in mol_data.columns:
                avg = mol_data['OQPLA_Final_Score'].mean()
                st.markdown(f"**OQPLA:** {avg:.3f}")

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


def _render_pdb_evidence(compound_name: str, df: pd.DataFrame) -> None:
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

    # Calculate summary statistics
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

    # Try to load detailed PDB summary file first
    pdb_summary_df = None
    try:
        safe_name = sanitize_compound_name(compound_name)
        # Try different possible filenames for PDB summary
        for filename in ["pdb_summary.csv", f"{safe_name}_pdb_summary.csv", f"{safe_name}_pdb_details.csv"]:
            pdb_summary_df = load_result_dataframe(compound_name, filename)
            if pdb_summary_df is not None and not pdb_summary_df.empty:
                break
    except Exception:
        pdb_summary_df = None

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
            except:
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
        cols = ['ChEMBL_ID', 'Molecule_Name', 'Activity_Type', 'Activity_nM', 'pActivity',
                'Target_Name', 'SEI', 'BEI', 'NSEI', 'NBEI', 'QED']
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
        cols = ['ChEMBL_ID', 'Molecule_Name', 'OQPLA_Final_Score', 'OQPLA_Classification',
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
        efo_term = str(row.get('EFO_Term', '')) if pd.notna(row.get('EFO_Term')) else ''
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
            showlegend=False,
            height=300,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, width='stretch')


def _load_compound_data(identifier: str) -> Optional[Dict[str, Any]]:
    """Load compound data from storage.

    Args:
        identifier: Either entry_id (UUID) for new storage or compound_name for legacy
    """
    try:
        summary = get_cached_result(identifier)
        if summary is None:
            return None

        # Load results DataFrame
        df = load_result_dataframe(identifier, "similar_compounds.csv")

        if df is None:
            # Try legacy filename format
            safe_name = sanitize_compound_name(identifier)
            df = load_result_dataframe(identifier, f"{safe_name}_complete_results.csv")

        if df is None:
            files = get_result_files(identifier)
            for f in files:
                if f.endswith('.csv') and 'pdb' not in f.lower() and 'indication' not in f.lower():
                    df = load_result_dataframe(identifier, f)
                    if df is not None:
                        break

        # Load drug indications (separate file)
        indications_df = load_result_dataframe(identifier, "drug_indications.csv")

        # Get display name from summary (compound_name is in summary.json)
        display_name = summary.get('compound_name', identifier)

        return {
            'compound_name': display_name,
            'entry_id': summary.get('entry_id', identifier),
            'smiles': summary.get('smiles', summary.get('query_smiles', '')),
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
                    # Also clear frontend caches
                    delete_from_cache(compound_name)
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
