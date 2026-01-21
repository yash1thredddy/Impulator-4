"""Job submission form component for IMPULATOR.

Provides the input form for submitting new compound analysis jobs.
"""

import html
import logging
from typing import Dict, List, Optional, Tuple

import streamlit as st

from frontend.services import get_api_client
from frontend.config.settings import config
from frontend.utils import SessionState, InputValidator, sanitize_compound_name
from frontend.ui.components.sidebar import start_polling
from frontend.ui.components.duplicate_dialog import render_duplicate_dialog, clear_duplicate_dialog_state

logger = logging.getLogger(__name__)


def render_job_form() -> Optional[str]:
    """Render the job submission form.

    Returns:
        Optional[str]: Job ID if submitted successfully, None otherwise
    """
    # Check if we need to show the duplicate dialog
    if st.session_state.get('show_duplicate_dialog'):
        duplicate_info = st.session_state.get('pending_duplicate_info', {})
        action, new_name = render_duplicate_dialog(duplicate_info)

        if action == "cancel":
            # User cancelled - clear state and return to form
            clear_duplicate_dialog_state()
            st.rerun()
            return None
        elif action is not None:
            # User made a choice - resolve the duplicate
            return _resolve_duplicate_action(action, new_name)

        # Dialog still showing, don't render the form below it
        return None

    st.subheader("Compound Information")

    # Compound name input
    compound_name = st.text_input(
        "Compound Name",
        placeholder="e.g., Aspirin",
        help="Name to identify this compound in results"
    )

    # Input type selection
    input_type = st.radio(
        "Structure Input Type",
        ["SMILES", "InChI"],
        horizontal=True,
        help="Choose the format of your chemical structure input"
    )

    # Structure input
    if input_type == "SMILES":
        structure_input = st.text_area(
            "SMILES String",
            height=80,
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
            help="Simplified Molecular Input Line Entry System notation"
        )
    else:
        structure_input = st.text_area(
            "InChI String",
            height=80,
            placeholder="e.g., InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
            help="International Chemical Identifier"
        )

    # Configuration section
    st.subheader("Analysis Configuration")

    col1, col2 = st.columns(2)

    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold (%)",
            min_value=50,
            max_value=100,
            value=config.DEFAULT_SIMILARITY_THRESHOLD,
            help="Minimum similarity for ChEMBL compound search"
        )

    with col2:
        st.markdown("**Activity Types**")
        selected_activities = render_activity_checkboxes()

    # Validation feedback
    validation_passed = True
    if compound_name and structure_input:
        if input_type == "SMILES":
            result = InputValidator.validate_smiles(structure_input)
        else:
            result = InputValidator.validate_inchi(structure_input)

        if not result.is_valid:
            st.error(f"Invalid {input_type}: {result.errors[0]}")
            validation_passed = False

    # Submit button
    st.divider()

    # Check if we're already processing
    is_processing = SessionState.is_processing()

    if st.button(
        "Submit Analysis Job",
        type="primary",
        disabled=is_processing,
        width='stretch'
    ):
        return _submit_job(
            compound_name=compound_name,
            structure_input=structure_input,
            input_type=input_type.lower(),
            similarity_threshold=similarity_threshold,
            activity_types=selected_activities
        )

    if is_processing:
        st.info("A job is currently being submitted...")

    return None


def render_activity_checkboxes(key_prefix: str = "single") -> List[str]:
    """Render activity type checkboxes and return selected types.

    Args:
        key_prefix: Prefix for widget keys to avoid duplicates (e.g., "single", "batch")
    """
    # Get default activity types from config
    default_types = list(config.DEFAULT_ACTIVITY_TYPES)

    # Initialize session state for checkboxes (use prefix to separate single vs batch)
    state_key = f'{key_prefix}_activity_checkboxes'
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            activity: True for activity in default_types
        }

    selected = []
    cols = st.columns(2)

    for i, activity in enumerate(default_types):
        with cols[i % 2]:
            is_checked = st.checkbox(
                activity,
                value=st.session_state[state_key].get(activity, True),
                key=f"{key_prefix}_activity_cb_{activity}"
            )
            st.session_state[state_key][activity] = is_checked
            if is_checked:
                selected.append(activity)

    if not selected:
        st.warning("Select at least one activity type")

    return selected


def _submit_job(
    compound_name: str,
    structure_input: str,
    input_type: str,
    similarity_threshold: int,
    activity_types: List[str]
) -> Optional[str]:
    """Submit the job to the backend.

    Returns:
        Optional[str]: Job ID if successful, None otherwise
    """
    # Validate inputs
    if not compound_name or not compound_name.strip():
        st.error("Please enter a compound name")
        return None

    if not structure_input or not structure_input.strip():
        st.error(f"Please enter a {input_type.upper()} string")
        return None

    if not activity_types:
        st.error("Please select at least one activity type")
        return None

    # Sanitize compound name (consistent with backend)
    sanitized_name = _sanitize_and_limit_name(compound_name.strip())

    if sanitized_name != compound_name.strip():
        st.info(f"Compound name sanitized: '{compound_name}' -> '{sanitized_name}'")

    # Convert InChI to SMILES if needed (backend expects SMILES)
    smiles = structure_input.strip()
    if input_type == "inchi":
        with st.spinner("Converting InChI to SMILES..."):
            smiles = _inchi_to_smiles(structure_input.strip())
            if not smiles:
                st.error("Failed to convert InChI to SMILES")
                return None
            st.success(f"Converted to SMILES: {smiles[:50]}...")

    # Submit to backend
    SessionState.start_processing(sanitized_name)

    try:
        client = get_api_client()
        response = client.submit_job(
            compound_name=sanitized_name,
            smiles=smiles,
            similarity_threshold=similarity_threshold,
            activity_types=activity_types
        )

        if response.success:
            st.success(f"Job submitted! ID: {response.job_id}")
            SessionState.add_active_job(
                job_id=response.job_id,
                compound_name=sanitized_name,
                status="pending"
            )
            # Start polling for job updates
            start_polling()
            return response.job_id
        elif response.is_duplicate:
            # Duplicate detected - store info and show dialog
            st.session_state['show_duplicate_dialog'] = True
            st.session_state['pending_duplicate_info'] = response.duplicate_info
            # Store job params for later resolution
            st.session_state['duplicate_smiles'] = smiles
            st.session_state['duplicate_compound_name'] = sanitized_name
            st.session_state['duplicate_similarity_threshold'] = similarity_threshold
            st.session_state['duplicate_activity_types'] = activity_types
            st.rerun()
            return None
        else:
            st.error(f"Failed to submit job: {response.error}")
            return None

    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        st.error(f"Error: {e}")
        return None

    finally:
        SessionState.set('is_processing', False)


def _resolve_duplicate_action(action: str, new_name: Optional[str]) -> Optional[str]:
    """Resolve a duplicate compound based on user's action choice.

    Args:
        action: 'replace', 'duplicate', or 'skip'
        new_name: New compound name if user changed it

    Returns:
        Job ID if a new job was created, None otherwise
    """
    # Get stored job parameters
    smiles = st.session_state.get('duplicate_smiles')
    compound_name = st.session_state.get('duplicate_compound_name')
    similarity_threshold = st.session_state.get('duplicate_similarity_threshold')
    activity_types = st.session_state.get('duplicate_activity_types')
    duplicate_info = st.session_state.get('pending_duplicate_info', {})

    existing_entry_id = duplicate_info.get('existing_compound', {}).get('entry_id')

    try:
        client = get_api_client()
        response = client.resolve_duplicate(
            action=action,
            smiles=smiles,
            compound_name=compound_name,
            existing_entry_id=existing_entry_id,
            new_compound_name=new_name,
            similarity_threshold=similarity_threshold,
            activity_types=activity_types
        )

        # Clear duplicate dialog state
        clear_duplicate_dialog_state()

        if response.success:
            if response.status == 'skipped':
                st.info(response.message or "Compound processing skipped")
                st.rerun()
                return None
            else:
                # Job was created
                final_name = new_name or compound_name
                st.success(f"Job submitted! ID: {response.job_id}")
                SessionState.add_active_job(
                    job_id=response.job_id,
                    compound_name=final_name,
                    status="pending"
                )
                start_polling()
                return response.job_id
        else:
            st.error(f"Failed to resolve duplicate: {response.error}")
            st.rerun()
            return None

    except Exception as e:
        logger.error(f"Error resolving duplicate: {e}")
        st.error(f"Error: {e}")
        clear_duplicate_dialog_state()
        st.rerun()
        return None


def _detect_column_mappings(df) -> Dict[str, Optional[str]]:
    """
    Detect likely column mappings based on column names.

    Returns dict with suggested original column name for each required field.
    Does NOT rename columns - just suggests mappings for dropdown pre-selection.

    Args:
        df: pandas DataFrame

    Returns:
        Dict like {'compound_name': 'Molecule', 'smiles': 'SMILES', 'inchi': None}
    """
    # Column name variants (lowercase -> field type)
    compound_name_variants = [
        'compound_name', 'compoundname', 'compound', 'name', 'molecule',
        'molecule_name', 'mol_name', 'molname', 'mol', 'title', 'id',
        'cdd num', 'cdd_num', 'cddnum',
    ]

    smiles_variants = [
        'smiles', 'canonical_smiles', 'canonicalsmiles', 'canonical smiles',
        'smi', 'structure', 'mol_smiles',
    ]

    inchi_variants = [
        'inchi', 'inchikey', 'inchi_key', 'standard_inchi', 'standardinchi',
    ]

    result = {'compound_name': None, 'smiles': None, 'inchi': None}

    for col in df.columns:
        col_lower = col.lower().strip()

        # Check compound name variants (first match wins)
        if result['compound_name'] is None and col_lower in compound_name_variants:
            result['compound_name'] = col

        # Check SMILES variants
        if result['smiles'] is None and col_lower in smiles_variants:
            result['smiles'] = col

        # Check InChI variants
        if result['inchi'] is None and col_lower in inchi_variants:
            result['inchi'] = col

    return result


def _render_column_mapping_ui(df) -> Optional[Dict[str, str]]:
    """
    Render dropdown selectors for column mapping.

    Shows dropdowns with auto-detected suggestions that users can override.

    Args:
        df: pandas DataFrame with CSV data

    Returns:
        Mapping dict if valid selection, None if incomplete
    """
    st.markdown("**Column Mapping**")
    st.caption("Select which columns to use (auto-detected, you can change)")

    columns = list(df.columns)
    columns_with_none = ["-- Select --"] + columns

    # Get auto-detected suggestions
    suggestions = _detect_column_mappings(df)

    # Calculate default index based on suggestion or existing selection
    def get_default_index(field_key: str, suggestion_key: str) -> int:
        # First check if widget already has a selection (from previous render)
        widget_key = f"csv_col_{field_key}_select"
        if widget_key in st.session_state:
            selected = st.session_state[widget_key]
            if selected in columns:
                return columns.index(selected) + 1
        # Otherwise use auto-detected suggestion
        suggested = suggestions.get(suggestion_key)
        if suggested and suggested in columns:
            return columns.index(suggested) + 1
        return 0

    col1, col2 = st.columns(2)

    with col1:
        # Compound Name dropdown (required)
        selected_name = st.selectbox(
            "Compound Name Column *",
            columns_with_none,
            index=get_default_index('name', 'compound_name'),
            key="csv_col_name_select",
            help="Column containing compound identifiers"
        )

    with col2:
        # SMILES dropdown
        selected_smiles = st.selectbox(
            "SMILES Column",
            columns_with_none,
            index=get_default_index('smiles', 'smiles'),
            key="csv_col_smiles_select",
            help="Column containing SMILES strings"
        )

    # InChI dropdown
    selected_inchi = st.selectbox(
        "InChI Column (optional if SMILES selected)",
        columns_with_none,
        index=get_default_index('inchi', 'inchi'),
        key="csv_col_inchi_select",
        help="Column containing InChI strings"
    )

    # Validate selections
    has_name = selected_name != "-- Select --"
    has_smiles = selected_smiles != "-- Select --"
    has_inchi = selected_inchi != "-- Select --"

    if not has_name:
        st.warning("Please select a Compound Name column")
        return None

    if not has_smiles and not has_inchi:
        st.warning("Please select either a SMILES or InChI column")
        return None

    # Build mapping
    mapping = {'compound_name': selected_name}
    if has_smiles:
        mapping['smiles'] = selected_smiles
    if has_inchi:
        mapping['inchi'] = selected_inchi

    return mapping


def _apply_column_mapping(df, mapping: Dict[str, str]):
    """
    Apply user-selected column mapping to dataframe.

    Creates a new dataframe with standardized column names.

    Args:
        df: Original DataFrame
        mapping: Dict mapping standard names to original column names

    Returns:
        New DataFrame with standardized columns
    """
    import pandas as pd

    result = pd.DataFrame()
    result['compound_name'] = df[mapping['compound_name']]

    if mapping.get('smiles'):
        result['smiles'] = df[mapping['smiles']]
    if mapping.get('inchi'):
        result['inchi'] = df[mapping['inchi']]

    return result


def _sanitize_and_limit_name(name: str) -> str:
    """Sanitize compound name for filesystem safety with length limit."""
    # Use shared sanitization function for consistency
    safe_name = sanitize_compound_name(name)
    # Limit length for display/filesystem
    return safe_name[:100] if len(safe_name) > 100 else safe_name


def _inchi_to_smiles(inchi: str) -> Optional[str]:
    """Convert InChI to SMILES using RDKit."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromInchi(inchi)
        if mol:
            return Chem.MolToSmiles(mol)
        return None
    except Exception as e:
        logger.error(f"InChI to SMILES conversion failed: {e}")
        return None


def render_csv_upload_form() -> Optional[str]:
    """Render the CSV batch upload form with duplicate confirmation.

    Returns:
        Optional[str]: Batch job ID if submitted, None otherwise
    """
    st.subheader("Batch Upload")
    st.info("Upload a CSV file with compound names and SMILES/InChI structures")

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV with any column names - you'll map them below"
    )

    if not uploaded_file:
        # Clear state when no file
        _clear_duplicate_check_state()
        _clear_column_mapping_state()
        return None

    # Check if file changed
    if SessionState.file_changed(uploaded_file):
        # Parse and validate CSV
        import pandas as pd
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['csv_preview'] = df
            # Clear state for new file
            _clear_duplicate_check_state()
            _clear_column_mapping_state()
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return None

    df = st.session_state.get('csv_preview')
    if df is None:
        return None

    # Interactive column mapping UI
    column_mapping = _render_column_mapping_ui(df)

    if column_mapping is None:
        # User hasn't selected required columns yet
        return None

    # Apply user's column mapping
    df_mapped = _apply_column_mapping(df, column_mapping)

    # Store mapped dataframe for submission
    st.session_state['csv_mapped'] = df_mapped
    has_smiles = 'smiles' in df_mapped.columns
    has_inchi = 'inchi' in df_mapped.columns

    # Preview mapped data
    st.write("Preview (mapped):")
    st.dataframe(df_mapped.head(5))
    st.caption(f"{len(df_mapped)} compounds in file")

    # Configuration
    st.subheader("Batch Configuration")

    similarity_threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=50,
        max_value=100,
        value=config.DEFAULT_SIMILARITY_THRESHOLD,
        key="batch_similarity"
    )

    selected_activities = render_activity_checkboxes(key_prefix="batch")

    # Check for duplicates before showing submit
    duplicate_check_done = st.session_state.get('batch_duplicate_check_done', False)
    user_confirmed = st.session_state.get('batch_user_confirmed', False)

    if not duplicate_check_done:
        # Step 1: Check for duplicates first
        if st.button("Check & Submit Batch", type="primary", width='stretch'):
            # Build compounds list with structures for InChIKey-based duplicate detection
            df_has_smiles = 'smiles' in df_mapped.columns
            df_has_inchi = 'inchi' in df_mapped.columns

            compounds_for_check = []
            for _, row in df_mapped.iterrows():
                compound_name = str(row.get('compound_name', '')).strip()
                if not compound_name:
                    continue

                safe_name = _sanitize_and_limit_name(compound_name)
                compound_data = {"compound_name": safe_name}

                # Add structure data for InChIKey generation
                if df_has_smiles:
                    smiles_val = str(row.get('smiles', '')).strip()
                    if smiles_val and smiles_val.lower() not in ('nan', 'none', ''):
                        compound_data["smiles"] = smiles_val

                if df_has_inchi:
                    inchi_val = str(row.get('inchi', '')).strip()
                    if inchi_val and inchi_val.lower() not in ('nan', 'none', ''):
                        compound_data["inchi"] = inchi_val

                compounds_for_check.append(compound_data)

            if not compounds_for_check:
                st.error("No valid compound names found in file")
                return None

            with st.spinner("Checking for existing compounds (by structure)..."):
                api_client = get_api_client()
                # Use new structure-based checking for InChIKey duplicate detection
                result = api_client.check_duplicates(compounds=compounds_for_check)

                if result.get("success"):
                    st.session_state['batch_duplicate_check_done'] = True
                    st.session_state['batch_existing'] = result.get('existing', [])
                    st.session_state['batch_processing'] = result.get('processing', [])
                    st.session_state['batch_new'] = result.get('new', [])
                    # Store structure matches for enhanced duplicate handling
                    st.session_state['batch_structure_matches'] = result.get('structure_matches', [])
                    st.rerun()
                else:
                    st.error(f"Failed to check duplicates: {result.get('error', 'Unknown error')}")
                    return None

    else:
        # Step 2: Show duplicate confirmation dialog
        existing = st.session_state.get('batch_existing', [])
        processing = st.session_state.get('batch_processing', [])
        new_compounds = st.session_state.get('batch_new', [])
        structure_matches = st.session_state.get('batch_structure_matches', [])

        # Show summary with colored boxes
        st.divider()
        st.markdown("### Duplicate Check Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("New Compounds", len(new_compounds))
        with col2:
            st.metric("Already Processed", len(existing))
        with col3:
            st.metric("Currently Processing", len(processing))
        with col4:
            st.metric("Structure Matches", len(structure_matches))

        # Show structure matches (InChIKey-based, more accurate)
        if structure_matches:
            with st.expander(f"🔬 Structure Matches - same molecule, different name ({len(structure_matches)})", expanded=True):
                st.caption("These compounds have the **same chemical structure** (InChIKey) as existing compounds:")
                for match in structure_matches[:20]:
                    match_type = match.get('match_type', 'structure_only')
                    your_name = html.escape(match.get('compound_name', ''))
                    existing_name = html.escape(match.get('existing_compound_name', ''))

                    if match_type == 'exact':
                        st.markdown(f"- `{your_name}` matches exactly")
                    else:
                        st.markdown(f"- `{your_name}` → same structure as **{existing_name}**")
                if len(structure_matches) > 20:
                    st.caption(f"...and {len(structure_matches) - 20} more")

        # Show currently processing (these are always skipped)
        if processing:
            with st.expander(f"⏳ Currently processing - will be skipped ({len(processing)})", expanded=False):
                safe_names = [html.escape(name) for name in processing[:20]]
                st.markdown(", ".join(f"`{name}`" for name in safe_names))
                if len(processing) > 20:
                    st.caption(f"...and {len(processing) - 20} more")

        # Per-compound duplicate handling for already processed compounds
        # Combine name-based existing and structure matches
        duplicate_decisions = {}  # compound_name -> action ('skip', 'replace', 'duplicate')
        duplicate_new_names = {}  # compound_name -> new_name (for 'duplicate' action)

        # Get compound names from structure matches that aren't in existing list
        structure_match_names = [m.get('compound_name') for m in structure_matches]
        all_existing = list(set(existing) | set(structure_match_names))

        if all_existing:
            st.markdown("#### Handle Existing Compounds")
            st.caption("Choose what to do with each compound that already has results:")

            # Default action selector (applies to all not individually set)
            default_action = st.selectbox(
                "Default action for all existing compounds:",
                options=["skip", "replace", "duplicate"],
                format_func=lambda x: {
                    "skip": "⏭️ Skip (don't reprocess)",
                    "replace": "🔄 Replace (delete and reprocess)",
                    "duplicate": "📋 Keep both (save with new name)",
                }.get(x, x),
                key="batch_default_duplicate_action",
                index=0
            )

            # Show individual compound controls in an expander
            with st.expander(f"📦 Customize per compound ({len(all_existing)} compounds)", expanded=len(all_existing) <= 10):
                # For smaller lists, show individual selects with name input for duplicates
                if len(all_existing) <= 20:
                    for i, compound_name in enumerate(all_existing):
                        safe_name = html.escape(compound_name)
                        st.markdown(f"**{safe_name}**")
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            action = st.selectbox(
                                f"Action for {compound_name}",
                                options=["default", "skip", "replace", "duplicate"],
                                format_func=lambda x: {
                                    "default": f"Use default ({default_action})",
                                    "skip": "⏭️ Skip",
                                    "replace": "🔄 Replace",
                                    "duplicate": "📋 Duplicate",
                                }.get(x, x),
                                key=f"dup_action_{i}",
                                label_visibility="collapsed"
                            )
                            if action != "default":
                                duplicate_decisions[compound_name] = action

                        # Show name input when duplicate is selected
                        effective_action = action if action != "default" else default_action
                        with col2:
                            if effective_action == "duplicate":
                                new_name = st.text_input(
                                    f"New name for {compound_name}",
                                    value=f"{compound_name}_v2",
                                    key=f"dup_name_{i}",
                                    label_visibility="collapsed",
                                    placeholder="Enter new name"
                                )
                                if new_name and new_name != compound_name:
                                    duplicate_new_names[compound_name] = new_name
                            else:
                                st.empty()  # Placeholder for alignment
                        st.divider()
                else:
                    # For larger lists, show summary with option to expand
                    st.info(f"All {len(all_existing)} compounds will use the default action: **{default_action}**")
                    if default_action == "duplicate":
                        st.warning("⚠️ Duplicates will be auto-named with '_v2' suffix. For custom names, process in smaller batches.")
                    st.caption("For individual control with large batches, consider processing in smaller groups.")

            # Apply default action to compounds not individually configured
            for compound_name in all_existing:
                if compound_name not in duplicate_decisions:
                    duplicate_decisions[compound_name] = default_action
                # Auto-generate new name for duplicates without custom names
                if duplicate_decisions.get(compound_name) == "duplicate" and compound_name not in duplicate_new_names:
                    duplicate_new_names[compound_name] = f"{compound_name}_v2"

            # Store decisions in session state
            st.session_state['batch_duplicate_decisions'] = duplicate_decisions
            st.session_state['batch_duplicate_new_names'] = duplicate_new_names

            # Show summary of actions
            skip_count = sum(1 for a in duplicate_decisions.values() if a == 'skip')
            replace_count = sum(1 for a in duplicate_decisions.values() if a == 'replace')
            dup_count = sum(1 for a in duplicate_decisions.values() if a == 'duplicate')

            if replace_count > 0 or dup_count > 0:
                st.divider()
                action_summary = []
                if skip_count > 0:
                    action_summary.append(f"⏭️ {skip_count} skipped")
                if replace_count > 0:
                    action_summary.append(f"🔄 {replace_count} replaced")
                if dup_count > 0:
                    action_summary.append(f"📋 {dup_count} as duplicates")
                st.info(f"Existing compounds: {' | '.join(action_summary)}")

        # Determine compounds to process based on decisions
        compounds_to_skip = [name for name, action in duplicate_decisions.items() if action == 'skip']
        compounds_to_replace = [name for name, action in duplicate_decisions.items() if action == 'replace']
        compounds_as_duplicates = [name for name, action in duplicate_decisions.items() if action == 'duplicate']

        compounds_to_process = list(new_compounds) + compounds_to_replace + compounds_as_duplicates

        if not compounds_to_process:
            st.info("All compounds already exist or are being processed. Nothing new to submit.")
            if st.button("↩️ Upload Different File", width='stretch'):
                _clear_duplicate_check_state()
                st.session_state.pop('csv_preview', None)
                st.rerun()
            return None

        # Show final summary
        st.markdown("#### Summary")
        st.success(f"**{len(compounds_to_process)}** compounds will be processed")

        with st.expander("View all compounds to process", expanded=False):
            if new_compounds:
                st.markdown(f"**New compounds ({len(new_compounds)}):** " +
                           ", ".join(f"`{html.escape(n)}`" for n in new_compounds[:20]) +
                           (f"... +{len(new_compounds)-20} more" if len(new_compounds) > 20 else ""))
            if compounds_to_replace:
                st.markdown(f"**Replacing ({len(compounds_to_replace)}):** " +
                           ", ".join(f"`{html.escape(n)}`" for n in compounds_to_replace[:20]) +
                           (f"... +{len(compounds_to_replace)-20} more" if len(compounds_to_replace) > 20 else ""))
            if compounds_as_duplicates:
                st.markdown(f"**As duplicates ({len(compounds_as_duplicates)}):** " +
                           ", ".join(f"`{html.escape(n)}`" for n in compounds_as_duplicates[:20]) +
                           (f"... +{len(compounds_as_duplicates)-20} more" if len(compounds_as_duplicates) > 20 else ""))

        # Confirmation buttons
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Confirm & Submit", type="primary", width='stretch'):
                st.session_state['batch_user_confirmed'] = True
                return _submit_batch(
                    df=df_mapped,
                    has_smiles=has_smiles,
                    similarity_threshold=similarity_threshold,
                    activity_types=selected_activities,
                    duplicate_decisions=duplicate_decisions,
                    duplicate_new_names=duplicate_new_names
                )

        with col2:
            if st.button("❌ Cancel", width='stretch'):
                _clear_duplicate_check_state()
                st.session_state.pop('csv_preview', None)
                st.rerun()

    return None


def _clear_duplicate_check_state():
    """Clear duplicate check related session state."""
    keys_to_clear = [
        'batch_duplicate_check_done',
        'batch_user_confirmed',
        'batch_existing',
        'batch_processing',
        'batch_new',
        'batch_structure_matches',  # InChIKey-based structure matches
        'batch_duplicate_decisions',
        'batch_duplicate_new_names',
        'batch_default_duplicate_action',
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    # Also clear per-compound action and name keys
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith('dup_action_') or k.startswith('dup_name_')]
    for key in keys_to_remove:
        st.session_state.pop(key, None)


def _clear_column_mapping_state():
    """Clear column mapping related session state."""
    keys_to_clear = [
        'csv_col_name_select',
        'csv_col_smiles_select',
        'csv_col_inchi_select',
        'csv_mapped',
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _submit_batch(
    df,
    has_smiles: bool,
    similarity_threshold: int,
    activity_types: List[str],
    duplicate_decisions: Dict[str, str] = None,
    duplicate_new_names: Dict[str, str] = None
) -> Optional[str]:
    """Submit batch of compounds to backend.

    Args:
        df: DataFrame with compound_name and smiles/inchi columns
        has_smiles: True if df has 'smiles' column, False if 'inchi'
        similarity_threshold: Similarity threshold for all compounds
        activity_types: Activity types for all compounds
        duplicate_decisions: Dict mapping compound_name -> action ('skip', 'replace', 'duplicate')
                            for each existing compound
        duplicate_new_names: Dict mapping original compound_name -> new_name for duplicates

    Returns:
        batch_id if successful, None otherwise
    """
    if duplicate_decisions is None:
        duplicate_decisions = {}
    if duplicate_new_names is None:
        duplicate_new_names = {}
    if df is None or df.empty:
        st.error("No compounds to submit")
        return None

    # Check which columns are available
    df_has_smiles = 'smiles' in df.columns
    df_has_inchi = 'inchi' in df.columns

    # Build compounds list for batch submission
    # Include the per-compound duplicate action and new names
    compounds = []
    skipped_no_structure = []

    for _, row in df.iterrows():
        compound_name = str(row.get('compound_name', '')).strip()

        if not compound_name:
            continue

        # Sanitize compound name
        safe_name = _sanitize_and_limit_name(compound_name)

        # Get per-compound duplicate action (if any)
        compound_action = duplicate_decisions.get(safe_name, None)

        # Skip compounds marked as 'skip' - don't include them in submission
        if compound_action == 'skip':
            continue

        # Try to get SMILES - with fallback from SMILES -> InChI conversion
        smiles = None

        # First try SMILES column if available
        if df_has_smiles:
            smiles_val = str(row.get('smiles', '')).strip()
            if smiles_val and smiles_val.lower() not in ('nan', 'none', ''):
                smiles = smiles_val

        # If no SMILES, try InChI column and convert
        if not smiles and df_has_inchi:
            inchi_val = str(row.get('inchi', '')).strip()
            if inchi_val and inchi_val.lower() not in ('nan', 'none', ''):
                converted = _inchi_to_smiles(inchi_val)
                if converted:
                    smiles = converted
                else:
                    logger.warning(f"Could not convert InChI for {compound_name}")

        # Skip if no valid structure found
        if not smiles:
            skipped_no_structure.append(safe_name)
            logger.warning(f"No valid SMILES or InChI for {compound_name}, skipping")
            continue

        # For duplicates, use the new name if provided
        final_name = safe_name
        if compound_action == 'duplicate' and safe_name in duplicate_new_names:
            final_name = _sanitize_and_limit_name(duplicate_new_names[safe_name])

        compound_data = {
            "compound_name": final_name,
            "smiles": smiles,
            "similarity_threshold": similarity_threshold,
            "activity_types": activity_types,
        }

        # Add duplicate action for this specific compound if it's an existing compound
        if compound_action:
            compound_data["duplicate_action"] = compound_action
            # Store original name for reference when marking as duplicate
            if compound_action == 'duplicate':
                compound_data["original_compound_name"] = safe_name

        compounds.append(compound_data)

    # Warn user about skipped compounds
    if skipped_no_structure:
        st.warning(f"Skipped {len(skipped_no_structure)} compounds with no valid SMILES or InChI: {', '.join(skipped_no_structure[:5])}{'...' if len(skipped_no_structure) > 5 else ''}")

    if not compounds:
        st.error("No valid compounds found in file (all may have been skipped)")
        return None

    # Count actions for display
    replace_count = sum(1 for c in compounds if c.get('duplicate_action') == 'replace')
    duplicate_count = sum(1 for c in compounds if c.get('duplicate_action') == 'duplicate')
    new_count = sum(1 for c in compounds if not c.get('duplicate_action'))

    # Build action label
    action_parts = []
    if new_count > 0:
        action_parts.append(f"{new_count} new")
    if replace_count > 0:
        action_parts.append(f"{replace_count} replacing")
    if duplicate_count > 0:
        action_parts.append(f"{duplicate_count} as duplicates")
    action_label = ", ".join(action_parts) if action_parts else f"{len(compounds)} compounds"

    with st.spinner(f"Submitting batch ({action_label})..."):
        try:
            api_client = get_api_client()

            result = api_client.submit_batch_job(
                compounds,
                duplicate_decisions=duplicate_decisions
            )

            if result.get("success"):
                batch_id = result.get("batch_id")
                jobs = result.get("jobs", [])
                skipped_existing = result.get("skipped_existing", [])
                skipped_processing = result.get("skipped_processing", [])
                replaced = result.get("replaced", [])

                # Show summary
                st.success(f"Batch submitted: {len(jobs)} jobs queued")

                if replaced:
                    st.info(f"🔄 Replaced {len(replaced)} existing compounds")

                if skipped_existing:
                    st.info(f"⏭️ Skipped {len(skipped_existing)} already processed: {', '.join(skipped_existing[:5])}{'...' if len(skipped_existing) > 5 else ''}")

                if skipped_processing:
                    st.info(f"⏳ Skipped {len(skipped_processing)} currently processing: {', '.join(skipped_processing[:5])}{'...' if len(skipped_processing) > 5 else ''}")

                # Start polling for job updates
                start_polling()

                # Clear state after successful submission
                _clear_duplicate_check_state()
                st.session_state.pop('csv_preview', None)
                st.session_state.pop('csv_mapped', None)

                return batch_id
            else:
                st.error(f"Batch submission failed: {result.get('error', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Batch submission error: {e}")
            st.error(f"Error submitting batch: {e}")
            return None
