"""Duplicate compound detection dialog for IMPULATOR.

Displays a dialog when a user submits a compound that already exists,
allowing them to choose how to handle the duplicate.
"""

import logging
from typing import Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)


def render_duplicate_dialog(duplicate_info: dict) -> Tuple[Optional[str], Optional[str]]:
    """Render the duplicate detection dialog.

    Shows different options based on the type of duplicate:
    - exact: Both structure (InChIKey) and name match
    - structure_only: Structure matches but name is different

    Args:
        duplicate_info: Dict with duplicate_type, existing_compound, submitted

    Returns:
        Tuple of (action, new_name):
        - action: 'replace', 'duplicate', 'skip', or None if not yet decided
        - new_name: New compound name if user changed it, None otherwise
    """
    dup_type = duplicate_info.get("duplicate_type", "exact")
    existing = duplicate_info.get("existing_compound", {})
    submitted = duplicate_info.get("submitted", {})

    existing_name = existing.get("compound_name", "Unknown")
    submitted_name = submitted.get("compound_name", "Unknown")
    processed_at = existing.get("processed_at", "Unknown")
    # Get suggested name from backend (calculates next available version, e.g., _v3 if _v2 exists)
    suggested_name = duplicate_info.get("suggested_name", f"{existing_name}_v2")

    # Container for the dialog
    with st.container(border=True):
        # Header based on duplicate type
        if dup_type == "exact":
            st.warning("**Exact Duplicate Found**")
            st.markdown(
                f"**{existing_name}** with this exact structure already exists."
            )
        else:
            st.warning("**Structure Already Exists**")
            st.markdown(
                f"This structure already exists as **{existing_name}**."
            )
            st.markdown(f"You entered: **{submitted_name}**")

        st.caption(f"Processed: {processed_at if processed_at != 'Unknown' else 'Previously processed'}")

        st.divider()

        # Different options based on duplicate type
        if dup_type == "exact":
            # Exact duplicate: same structure AND same name
            st.markdown("**What would you like to do?**")

            action = st.radio(
                "Choose an action:",
                options=["replace", "change_name", "skip"],
                format_func=lambda x: {
                    "replace": "Replace existing results (reprocess and overwrite)",
                    "change_name": "Change name and save as duplicate",
                    "skip": "Skip (don't process)",
                }.get(x, x),
                key="duplicate_action_exact",
                label_visibility="collapsed"
            )

            # Show name input if user wants to change name
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
            # Structure-only duplicate: same structure, different name
            st.markdown("**What would you like to do?**")

            action = st.radio(
                "Choose an action:",
                options=["replace", "duplicate", "skip"],
                format_func=lambda x: {
                    "replace": f"Replace existing '{existing_name}' with new name",
                    "duplicate": f"Proceed as duplicate (tagged as duplicate of {existing_name})",
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
    """Clear all duplicate dialog related session state."""
    keys_to_clear = [
        'pending_duplicate_info',
        'show_duplicate_dialog',
        'duplicate_action_exact',
        'duplicate_action_structure',
        'duplicate_new_name',
        'duplicate_smiles',
        'duplicate_compound_name',
        'duplicate_similarity_threshold',
        'duplicate_activity_types',
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
