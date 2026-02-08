"""Job submission form component for IMPULATOR.

Provides the input form for submitting new compound analysis jobs.
"""

import html
import logging
from typing import Dict, List, Optional

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
    # Check if we have a pending success from duplicate resolution
    # Just return the job_id - analyze.py will handle the success display
    if st.session_state.get('duplicate_resolution_success'):
        success_info = st.session_state.pop('duplicate_resolution_success')
        return success_info['job_id']

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

    # Author name input
    author_name = st.text_input(
        "Author Name",
        placeholder="e.g., Dr. Jane Smith",
        help="Your name (required for attribution in reports)"
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
            min_value=30,
            max_value=100,
            value=config.DEFAULT_SIMILARITY_THRESHOLD,
            help="Minimum similarity for ChEMBL compound search"
        )

    with col2:
        st.markdown("**Activity Types**")
        selected_activities = render_activity_checkboxes()

    # Validation feedback
    if compound_name and structure_input:
        if input_type == "SMILES":
            result = InputValidator.validate_smiles(structure_input)
        else:
            result = InputValidator.validate_inchi(structure_input)

        if not result.is_valid:
            st.error(f"Invalid {input_type}: {result.errors[0]}")

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
            author_name=author_name,
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
    author_name: str,
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

    if not author_name or not author_name.strip():
        st.error("Please enter an author name")
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
            activity_types=activity_types,
            author_name=author_name.strip(),
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
            # First clear any previous success state to prevent it showing behind dialog
            SessionState.set('just_submitted_job', False)
            SessionState.set('last_submitted_job_id', None)

            st.session_state['show_duplicate_dialog'] = True
            st.session_state['pending_duplicate_info'] = response.duplicate_info
            # Store job params for later resolution
            st.session_state['duplicate_smiles'] = smiles
            st.session_state['duplicate_compound_name'] = sanitized_name
            st.session_state['duplicate_author_name'] = author_name.strip()
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
    author_name = st.session_state.get('duplicate_author_name')
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
            activity_types=activity_types,
            author_name=author_name,
        )

        # Clear duplicate dialog state BEFORE rerun
        clear_duplicate_dialog_state()

        if response.success:
            if response.status == 'skipped':
                st.info(response.message or "Compound processing skipped")
                st.rerun()
                return None
            else:
                # Job was created - store success info and rerun to close dialog
                final_name = new_name or compound_name
                SessionState.add_active_job(
                    job_id=response.job_id,
                    compound_name=final_name,
                    status="pending"
                )
                start_polling()

                # Store success state for display after rerun (closes the dialog first)
                st.session_state['duplicate_resolution_success'] = {
                    'job_id': response.job_id,
                    'compound_name': final_name
                }
                st.rerun()
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


def _generate_next_version_name(compound_name: str, existing_names: List[str]) -> str:
    """
    Generate next version name for a duplicate compound.

    First normalizes compound_name by stripping any existing _vN suffix to get base_name,
    then finds the highest existing version and returns base_name_v{max+1}.

    Examples:
        - "Aspirin" with no existing versions -> "Aspirin_v2"
        - "Aspirin" with "Aspirin_v2" existing -> "Aspirin_v3"
        - "Aspirin_v2" (already versioned) -> "Aspirin_v3" (strips suffix first)
        - "Aspirin" with "Aspirin_v2", "Aspirin_v3" existing -> "Aspirin_v4"

    Args:
        compound_name: Original compound name (may already have _vN suffix)
        existing_names: List of existing compound names to check against

    Returns:
        Next available version name (e.g., "Aspirin_v3")
    """
    import re

    # First, normalize compound_name by stripping any existing _vN suffix
    version_suffix_pattern = re.compile(r'^(.+?)_v(\d+)$', re.IGNORECASE)
    suffix_match = version_suffix_pattern.match(compound_name)

    if suffix_match:
        # compound_name already has version suffix - extract base and seed max_version
        base_name = suffix_match.group(1)
        max_version = int(suffix_match.group(2))
    else:
        # No suffix - use as-is, original counts as version 1
        base_name = compound_name
        max_version = 1

    # Find all existing versions matching pattern {base_name}_v{number}
    pattern = re.compile(rf'^{re.escape(base_name)}_v(\d+)$', re.IGNORECASE)

    for name in existing_names:
        match = pattern.match(name)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)

    return f"{base_name}_v{max_version + 1}"


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
        st.session_state.pop('uploaded_file_hash', None)
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
            # Clear stale state to prevent showing old data on error
            _clear_duplicate_check_state()
            _clear_column_mapping_state()
            if 'csv_preview' in st.session_state:
                del st.session_state['csv_preview']
            if 'csv_mapped' in st.session_state:
                del st.session_state['csv_mapped']
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

    # Preview mapped data
    st.write("Preview (mapped):")
    st.dataframe(df_mapped.head(5))
    st.caption(f"{len(df_mapped)} compounds in file")

    # Configuration
    st.subheader("Batch Configuration")

    batch_author_name = st.text_input(
        "Author Name",
        placeholder="e.g., Dr. Jane Smith",
        help="Your name (required, applied to all compounds in this batch)",
        key="batch_author_name"
    )

    similarity_threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=30,
        max_value=100,
        value=config.DEFAULT_SIMILARITY_THRESHOLD,
        key="batch_similarity"
    )

    selected_activities = render_activity_checkboxes(key_prefix="batch")

    def _build_duplicate_check_signature(
        threshold: int,
        activities: List[str],
        mapping: Dict[str, str],
    ) -> tuple:
        """Create a stable signature for duplicate-check configuration."""
        normalized_activities = tuple(sorted({a.strip() for a in activities if a and a.strip()}))
        mapping_signature = tuple(
            sorted((key, (value or "").strip()) for key, value in (mapping or {}).items())
        )
        return (threshold, normalized_activities, mapping_signature)

    current_check_signature = _build_duplicate_check_signature(
        similarity_threshold,
        selected_activities,
        column_mapping,
    )

    # Check for duplicates before showing submit
    duplicate_check_done = st.session_state.get('batch_duplicate_check_done', False)
    checked_signature = st.session_state.get('batch_duplicate_check_signature')
    if duplicate_check_done and checked_signature != current_check_signature:
        _clear_duplicate_check_state()
        duplicate_check_done = False
        st.info(
            "Batch configuration changed (threshold/activity types). "
            "Duplicate check results were cleared; run **Check & Submit Batch** again."
        )

    if not duplicate_check_done:
        # Step 1: Check for duplicates first
        if st.button("Check & Submit Batch", type="primary", width='stretch'):
            # Validate author name before making any API calls
            if not batch_author_name or not batch_author_name.strip():
                st.error("Please enter an author name before submitting")
                return None

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
                result = api_client.check_duplicates(
                    compounds=compounds_for_check,
                    similarity_threshold=similarity_threshold,
                    activity_types=selected_activities,
                )

                if result.get("success"):
                    st.session_state['batch_duplicate_check_done'] = True
                    st.session_state['batch_duplicate_check_signature'] = current_check_signature
                    st.session_state['batch_existing'] = result.get('existing', [])
                    st.session_state['batch_processing'] = result.get('processing', [])
                    st.session_state['batch_new'] = result.get('new', [])
                    # Store structure matches for enhanced duplicate handling
                    st.session_state['batch_structure_matches'] = result.get('structure_matches', [])
                    # Store backend-computed suggested version names (avoids collision issues)
                    st.session_state['batch_suggested_versions'] = result.get('suggested_versions', {})
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

        def _normalize_name(name: str) -> str:
            return (name or "").strip().lower()

        # Case-insensitive exact matches should count as already processed.
        exact_match_names = [
            m.get('compound_name')
            for m in structure_matches
            if m.get('match_type') == 'exact'
        ]
        already_processed = []
        seen_processed = set()
        for name in list(existing) + exact_match_names:
            normalized = _normalize_name(name)
            if normalized and normalized not in seen_processed:
                seen_processed.add(normalized)
                already_processed.append(name)

        # Show summary with colored boxes
        st.divider()
        st.markdown("### Duplicate Check Results")

        exact_count = len(exact_match_names)
        structure_only_matches = [m for m in structure_matches if m.get('match_type') != 'exact']
        structure_only_count = len(structure_only_matches)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("New Compounds", len(new_compounds))
        with col2:
            st.metric("Already Processed", len(already_processed))
        with col3:
            st.metric("Currently Processing", len(processing))
        with col4:
            st.metric("Structure-only Matches", structure_only_count)
        if exact_count > 0:
            st.caption(
                f"{exact_count} exact name+structure match(es) are counted in **Already Processed**."
            )

        # Show structure matches (InChIKey-based, more accurate)
        if structure_matches:
            with st.expander(f"🔬 Structure Matches (InChIKey) ({len(structure_matches)})", expanded=True):
                st.caption(
                    "These compounds share structure with existing records "
                    f"({exact_count} exact name+structure, {structure_only_count} structure-only)."
                )
                for match in structure_matches[:20]:
                    match_type = match.get('match_type', 'structure_only')
                    config_match = match.get('config_match', '')
                    your_name = html.escape(match.get('compound_name', ''))
                    existing_name = html.escape(match.get('existing_compound_name', ''))

                    if match_type == 'exact':
                        if config_match == 'identical':
                            st.markdown(
                                f"- `{your_name}` matches **{existing_name}** exactly "
                                "(same name, structure, and configuration)"
                            )
                        else:
                            st.markdown(
                                f"- `{your_name}` matches **{existing_name}** by name+structure "
                                "(configuration differs)"
                            )
                    else:
                        if config_match == 'identical':
                            st.markdown(
                                f"- `{your_name}` has the same structure as **{existing_name}** "
                                "(same configuration)"
                            )
                        else:
                            st.markdown(
                                f"- `{your_name}` → same structure as **{existing_name}** "
                                "(configuration differs)"
                            )
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
        skip_count = 0
        replace_count = 0
        dup_count = 0

        # Build a case-insensitive lookup for structure match metadata.
        structure_match_by_name = {}
        for match in structure_matches:
            normalized = _normalize_name(match.get('compound_name', ''))
            if normalized and normalized not in structure_match_by_name:
                structure_match_by_name[normalized] = match

        # Get compound names from structure matches that aren't in existing list.
        # Preserve order: first existing, then structure matches.
        structure_match_names = [m.get('compound_name') for m in structure_matches if m.get('compound_name')]
        all_existing = []
        seen_existing = set()
        for name in list(existing) + structure_match_names:
            normalized = _normalize_name(name)
            if normalized and normalized not in seen_existing:
                seen_existing.add(normalized)
                all_existing.append(name)

        def _render_match_details(compound_name: str, match: dict, allow_duplicate: bool) -> None:
            """Render per-compound duplicate context near the action selector."""
            if not match:
                st.markdown("**Match Details**")
                st.caption("Name-based duplicate match found. Structure/config details are unavailable for this row.")
                return

            existing_name = match.get('existing_compound_name', 'Unknown')
            match_type = match.get('match_type', 'structure_only')
            config_match = match.get('config_match')
            processed_at = match.get('existing_processed_at')
            existing_threshold = match.get('existing_similarity_threshold')
            existing_activities = match.get('existing_activity_types')
            existing_author = match.get('existing_author_name')
            your_name = compound_name

            def _format_processed_datetime(raw_timestamp: str) -> str:
                """Format ISO-like timestamp into readable local format."""
                if not raw_timestamp:
                    return "N/A"
                ts = str(raw_timestamp).strip()
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(ts)
                    return dt.strftime("%b %d, %Y at %I:%M:%S %p")
                except (ValueError, TypeError):
                    return str(raw_timestamp).replace("T", " ")

            st.markdown("**Match Details**")

            # Row 1 (left): structure match summary with chip styling
            if match_type == 'exact':
                summary_text = f"`{your_name}` matches `{existing_name}` by name and structure."
            else:
                summary_text = f"`{your_name}` shares structure with `{existing_name}`."

            st.caption(summary_text)

            activities_display = ""
            if existing_activities:
                activities_display = ", ".join(
                    part.strip() for part in str(existing_activities).split(",") if part.strip()
                )

            # Row 2: existing config (left) and activity types (right)
            existing_config_text = (
                f"Existing config: threshold {existing_threshold}%"
                if existing_threshold is not None
                else "Existing config: N/A"
            )
            activity_types_text = (
                f"Activity types: `{activities_display}`"
                if activities_display
                else "Activity types: N/A"
            )
            row2_left, row2_right = st.columns(2, gap="medium")
            with row2_left:
                st.caption(existing_config_text)
            with row2_right:
                st.caption(activity_types_text)

            # Row 3: allowed actions (left) and processed info (right)
            allowed_text = "skip or replace" if not allow_duplicate else "skip, replace, or duplicate"
            processed_display = _format_processed_datetime(processed_at) if processed_at else "N/A"
            author_label = (existing_author or "Unknown").strip()
            row3_left, row3_right = st.columns(2, gap="medium")
            with row3_left:
                st.caption(f"Allowed actions: {allowed_text}")
            with row3_right:
                st.caption(f"Processed: {processed_display} by {author_label}")

        if all_existing:
            st.markdown("#### Handle Existing Compounds")
            st.caption("Choose what to do with each compound that already has results:")

            duplicate_blocked = {
                name: (
                    structure_match_by_name.get(_normalize_name(name), {}).get("config_match") == "identical"
                )
                for name in all_existing
            }
            blocked_count = sum(1 for blocked in duplicate_blocked.values() if blocked)
            if blocked_count > 0:
                st.info(
                    f"{blocked_count} compounds already exist with identical configuration. "
                    "For those, duplicate is disabled (replace or skip only)."
                )

            # Keep all default options available globally. Per-compound controls and
            # the post-selection safety guard handle blocked identical-config rows.
            default_options = ["skip", "replace", "duplicate"]
            if st.session_state.get("batch_default_duplicate_action") not in default_options:
                st.session_state.pop("batch_default_duplicate_action", None)

            # Default action selector (applies to all not individually set)
            default_action = st.selectbox(
                "Default action for all existing compounds:",
                options=default_options,
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
                def _get_config_status_note(match: dict) -> tuple[str, str]:
                    """Return (level, text) for config status note in batch per-compound rows."""
                    if not match:
                        return ("none", "")
                    config_match = match.get("config_match")
                    existing_threshold = match.get("existing_similarity_threshold", 90)
                    if config_match == "identical":
                        return (
                            "warning",
                            f"Same threshold ({existing_threshold}%) and activity types — results would be identical.",
                        )
                    if config_match:
                        if config_match == "different_threshold":
                            return ("info", "Different threshold selected — results may differ from existing analysis.")
                        if config_match == "different_activities":
                            return ("info", "Different activity types selected — results may differ from existing analysis.")
                        if config_match == "different_both":
                            return ("info", "Threshold and activity types differ — results will likely differ.")
                        return ("info", "Configuration differs from existing analysis — results may differ.")
                    return ("caption", "Configuration details unavailable.")

                # For smaller lists, show individual selects with name input for duplicates
                if len(all_existing) <= 20:
                    for i, compound_name in enumerate(all_existing):
                        safe_name = html.escape(compound_name)
                        match_details = structure_match_by_name.get(_normalize_name(compound_name), {})
                        note_level, note_text = _get_config_status_note(match_details)

                        st.markdown(f"**{safe_name}**")

                        col1, col2 = st.columns([2, 3])
                        allow_duplicate = not duplicate_blocked.get(compound_name, False)
                        action_options = ["default", "skip", "replace"] + (["duplicate"] if allow_duplicate else [])
                        action_key = f"dup_action_{i}"
                        if st.session_state.get(action_key) not in action_options:
                            st.session_state.pop(action_key, None)

                        with col1:
                            action = st.selectbox(
                                f"Action for {compound_name}",
                                options=action_options,
                                format_func=lambda x: {
                                    "default": f"Use default ({default_action})",
                                    "skip": "⏭️ Skip",
                                    "replace": "🔄 Replace",
                                    "duplicate": "📋 Duplicate",
                                }.get(x, x),
                                key=action_key,
                                label_visibility="collapsed"
                            )
                            if action != "default":
                                duplicate_decisions[compound_name] = action
                            # Show config-status note under action control (left column)
                            if note_text:
                                if note_level == "warning":
                                    st.warning(note_text, icon="⚠️")
                                elif note_level == "info":
                                    st.info(note_text, icon="ℹ️")
                                else:
                                    st.caption(note_text)

                        # Show name input when duplicate is selected
                        effective_action = action if action != "default" else default_action
                        with col2:
                            if effective_action == "duplicate" and allow_duplicate:
                                new_name = st.text_input(
                                    f"New name for {compound_name}",
                                    value=f"{compound_name}_v2",
                                    key=f"dup_name_{i}",
                                    label_visibility="collapsed",
                                    placeholder="Enter new name"
                                )
                                if new_name and new_name != compound_name:
                                    duplicate_new_names[compound_name] = new_name
                            _render_match_details(compound_name, match_details, allow_duplicate)
                        st.divider()
                else:
                    # For larger lists, show summary with option to expand
                    st.info(f"All {len(all_existing)} compounds will use the default action: **{default_action}**")
                    if default_action == "duplicate":
                        st.warning("⚠️ Duplicates will be auto-named with '_v2' suffix. For custom names, process in smaller batches.")
                    st.caption("For individual control with large batches, consider processing in smaller groups.")

            # Apply default action to compounds not individually configured
            # Get backend-computed suggested versions (avoids collision issues)
            suggested_versions = st.session_state.get('batch_suggested_versions', {})

            for compound_name in all_existing:
                if compound_name not in duplicate_decisions:
                    duplicate_decisions[compound_name] = default_action
                if duplicate_blocked.get(compound_name) and duplicate_decisions.get(compound_name) == "duplicate":
                    duplicate_decisions[compound_name] = "skip"
                # Use backend-provided version name for duplicates (computed from full database state)
                if duplicate_decisions.get(compound_name) == "duplicate" and compound_name not in duplicate_new_names:
                    # Prefer backend-computed version name, fallback to local generation
                    if compound_name in suggested_versions:
                        duplicate_new_names[compound_name] = suggested_versions[compound_name]
                    else:
                        # Fallback for structure matches not in existing list
                        duplicate_new_names[compound_name] = _generate_next_version_name(compound_name, existing)

            # Store decisions in session state
            st.session_state['batch_duplicate_decisions'] = duplicate_decisions
            st.session_state['batch_duplicate_new_names'] = duplicate_new_names

            # Show summary of actions
            skip_count = sum(1 for a in duplicate_decisions.values() if a == 'skip')
            replace_count = sum(1 for a in duplicate_decisions.values() if a == 'replace')
            dup_count = sum(1 for a in duplicate_decisions.values() if a == 'duplicate')

            if skip_count > 0 or replace_count > 0 or dup_count > 0:
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
        compounds_to_replace = [name for name, action in duplicate_decisions.items() if action == 'replace']
        compounds_as_duplicates = [name for name, action in duplicate_decisions.items() if action == 'duplicate']

        compounds_to_process = list(new_compounds) + compounds_to_replace + compounds_as_duplicates

        if not compounds_to_process:
            st.info("All compounds already exist or are being processed. Nothing new to submit.")
            if st.button("↩️ Upload Different File", width='stretch'):
                _clear_duplicate_check_state()
                _clear_column_mapping_state()
                st.session_state.pop('csv_preview', None)
                st.session_state.pop('uploaded_file_hash', None)
                st.rerun()
            return None

        # Show final summary
        st.markdown("#### Summary")
        st.success(f"**{len(compounds_to_process)}** compounds will be processed")
        skipped_existing_names = [name for name, action in duplicate_decisions.items() if action == 'skip']
        if skipped_existing_names or processing:
            skip_parts = []
            if skipped_existing_names:
                skip_parts.append(f"⏭️ {len(skipped_existing_names)} existing skipped")
            if processing:
                skip_parts.append(f"⏳ {len(processing)} currently processing skipped")
            st.info("Will be skipped: " + " | ".join(skip_parts))

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
                if not batch_author_name or not batch_author_name.strip():
                    st.error("Please enter an author name")
                    return None
                return _submit_batch(
                    df=df_mapped,
                    has_smiles=has_smiles,
                    similarity_threshold=similarity_threshold,
                    activity_types=selected_activities,
                    duplicate_decisions=duplicate_decisions,
                    duplicate_new_names=duplicate_new_names,
                    author_name=batch_author_name.strip(),
                )

        with col2:
            if st.button("❌ Cancel", width='stretch'):
                _clear_duplicate_check_state()
                _clear_column_mapping_state()
                st.session_state.pop('csv_preview', None)
                st.session_state.pop('uploaded_file_hash', None)
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
        'batch_suggested_versions',
        'batch_duplicate_check_signature',
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
    duplicate_new_names: Dict[str, str] = None,
    author_name: str = "",
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
        author_name: Name of the author submitting the batch

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
            "author_name": author_name,
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
                st.session_state.pop('uploaded_file_hash', None)

                return batch_id
            else:
                st.error(f"Batch submission failed: {result.get('error', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Batch submission error: {e}")
            st.error(f"Error submitting batch: {e}")
            return None
