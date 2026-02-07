"""Duplicate compound detection dialog for IMPULATOR.

Displays a dialog when a user submits a compound that already exists,
allowing them to choose how to handle the duplicate.

Config-aware: distinguishes same-config duplicates (pointless re-run)
from different-config duplicates (genuinely different analysis).
"""

import logging
from typing import Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)


def _render_config_comparison(config_diff: dict) -> None:
    """Render a side-by-side config comparison table.

    Args:
        config_diff: Dict with 'similarity_threshold' and 'activity_types',
                     each containing 'existing' and 'submitted' values.
    """
    threshold = config_diff.get("similarity_threshold", {})
    activities = config_diff.get("activity_types", {})

    existing_threshold = threshold.get("existing", "N/A")
    submitted_threshold = threshold.get("submitted", "N/A")
    existing_activities = activities.get("existing", "N/A")
    submitted_activities = activities.get("submitted", "N/A")

    col_existing, col_submitted = st.columns(2)

    with col_existing:
        st.markdown("**Existing**")
        st.markdown(f"Threshold: **{existing_threshold}%**")
        # Format activity types as readable list
        if isinstance(existing_activities, str):
            st.markdown(f"Activities: {existing_activities}")

    with col_submitted:
        st.markdown("**New Submission**")
        st.markdown(f"Threshold: **{submitted_threshold}%**")
        if isinstance(submitted_activities, str):
            st.markdown(f"Activities: {submitted_activities}")


def render_duplicate_dialog(duplicate_info: dict) -> Tuple[Optional[str], Optional[str]]:
    """Render the duplicate detection dialog.

    Shows different options based on the type of duplicate and config match:
    - identical config: Only Replace/Skip (duplicate would be pointless)
    - different config: Show config diff, allow Replace/Duplicate/Skip

    Args:
        duplicate_info: Dict with duplicate_type, config_match, config_diff,
                       existing_compound, submitted

    Returns:
        Tuple of (action, new_name):
        - action: 'replace', 'duplicate', 'skip', or None if not yet decided
        - new_name: New compound name if user changed it, None otherwise
    """
    dup_type = duplicate_info.get("duplicate_type", "exact")
    existing = duplicate_info.get("existing_compound", {})
    submitted = duplicate_info.get("submitted", {})
    config_match = duplicate_info.get("config_match", "identical")
    config_diff = duplicate_info.get("config_diff")

    existing_name = existing.get("compound_name", "Unknown")
    submitted_name = submitted.get("compound_name", "Unknown")
    processed_at = existing.get("processed_at", "Unknown")
    # Get suggested name from backend (calculates next available version, e.g., _v3 if _v2 exists)
    suggested_name = duplicate_info.get("suggested_name", f"{existing_name}_v2")

    is_identical_config = config_match == "identical"

    # Container for the dialog
    with st.container(border=True):
        # Header based on config match
        if is_identical_config:
            st.warning("**Exact Duplicate Found**")
            if dup_type == "exact":
                st.markdown(
                    f"**{existing_name}** with this exact structure and "
                    f"configuration already exists."
                )
            else:
                st.markdown(
                    f"This structure already exists as **{existing_name}** "
                    f"with the same configuration."
                )
                st.markdown(f"You entered: **{submitted_name}**")

            st.info(
                f"Same threshold ({existing.get('similarity_threshold', 90)}%) "
                f"and activity types — results would be identical.",
                icon="ℹ️",
            )
        else:
            st.warning("**Structure Already Exists**")
            if dup_type == "exact":
                st.markdown(
                    f"**{existing_name}** with this structure already exists, "
                    f"but with a **different configuration**."
                )
            else:
                st.markdown(
                    f"This structure already exists as **{existing_name}** "
                    f"with a different configuration."
                )
                st.markdown(f"You entered: **{submitted_name}**")

            # Show config comparison
            if config_diff:
                with st.expander("Configuration Comparison", expanded=True):
                    _render_config_comparison(config_diff)

        st.caption(f"Processed: {processed_at if processed_at != 'Unknown' else 'Previously processed'}")

        st.divider()

        # Options depend on config match
        st.markdown("**What would you like to do?**")

        if is_identical_config:
            # Same config: only Replace or Skip (no duplicate option - would be pointless)
            if dup_type == "exact":
                action = st.radio(
                    "Choose an action:",
                    options=["replace", "skip"],
                    format_func=lambda x: {
                        "replace": "Replace (reprocess and overwrite existing results)",
                        "skip": "Skip (don't process)",
                    }.get(x, x),
                    key="duplicate_action_exact",
                    label_visibility="collapsed"
                )
            else:
                action = st.radio(
                    "Choose an action:",
                    options=["replace", "skip"],
                    format_func=lambda x: {
                        "replace": f"Replace existing '{existing_name}' with new results",
                        "skip": "Skip (don't process)",
                    }.get(x, x),
                    key="duplicate_action_structure",
                    label_visibility="collapsed"
                )
            new_name = None

        else:
            # Different config: allow all three options
            if dup_type == "exact":
                action = st.radio(
                    "Choose an action:",
                    options=["replace", "change_name", "skip"],
                    format_func=lambda x: {
                        "replace": "Replace existing (reprocess with new config)",
                        "change_name": "Keep both (save as separate analysis)",
                        "skip": "Skip (don't process)",
                    }.get(x, x),
                    key="duplicate_action_exact",
                    label_visibility="collapsed"
                )

                # Show name input if user wants to keep both
                new_name = None
                if action == "change_name":
                    new_name = st.text_input(
                        "New compound name:",
                        value=suggested_name,
                        key="duplicate_new_name",
                        help="Enter a unique name for this compound"
                    )
                    if new_name and new_name.strip() == existing_name:
                        st.error("Please enter a different name than the existing one.")
                        new_name = None

            else:
                # Structure-only + different config: full options
                action = st.radio(
                    "Choose an action:",
                    options=["replace", "duplicate", "skip"],
                    format_func=lambda x: {
                        "replace": f"Replace existing '{existing_name}' (reprocess with new config)",
                        "duplicate": f"Keep both (save as separate analysis of {existing_name})",
                        "skip": "Skip (don't process)",
                    }.get(x, x),
                    key="duplicate_action_structure",
                    label_visibility="collapsed"
                )
                new_name = None

        st.divider()

        # Action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Cancel", width="stretch", key="duplicate_cancel"):
                return "cancel", None

        with col2:
            if st.button("Continue", type="primary", width="stretch", key="duplicate_continue"):
                # Map change_name to duplicate action with new name
                if action == "change_name":
                    if new_name and new_name.strip():
                        return "duplicate", new_name.strip()
                    else:
                        st.error("Please enter a valid name")
                        return None, None
                return action, new_name

    return None, None


def clear_duplicate_dialog_state():
    """Clear all duplicate dialog related session state.

    Note: 'duplicate_resolution_success' is NOT cleared here because it's set
    AFTER clearing dialog state and needs to persist through the rerun to
    display success messages after the dialog closes.
    """
    keys_to_clear = [
        'pending_duplicate_info',
        'show_duplicate_dialog',
        'duplicate_action_exact',
        'duplicate_action_structure',
        'duplicate_new_name',
        'duplicate_smiles',
        'duplicate_compound_name',
        'duplicate_author_name',
        'duplicate_similarity_threshold',
        'duplicate_activity_types',
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
