"""Duplicate compound detection dialog for IMPULATOR.

Displays a modal dialog when a user submits a compound that already exists,
allowing them to choose how to handle the duplicate.

Config-aware: distinguishes same-config duplicates (pointless re-run)
from different-config duplicates (genuinely different analysis).

Uses @st.dialog decorator for native modal behavior. Results are communicated
via session_state (not return values) because @st.dialog functions cannot
return values to the caller.
"""

import logging
from datetime import datetime
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Session state key for dialog result communication
DUPLICATE_RESULT_KEY = "duplicate_result"


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


def _format_processed_datetime(raw_timestamp: Optional[str]) -> str:
    """Format ISO-like timestamp into a human-readable form."""
    if not raw_timestamp:
        return "N/A"

    ts = str(raw_timestamp).strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%b %d, %Y at %I:%M:%S %p")
    except (ValueError, TypeError):
        return str(raw_timestamp).replace("T", " ")


def _get_allowed_actions_text(dup_type: str, is_identical_config: bool) -> str:
    """Get human-readable allowed-actions summary for the current duplicate context."""
    if is_identical_config:
        return "replace or skip"
    if dup_type == "exact":
        return "replace, change name, or skip"
    return "replace, duplicate, or skip"


def _render_match_details(
    existing: dict,
    submitted: dict,
    dup_type: str,
    config_match: str,
    is_identical_config: bool,
    config_diff: Optional[dict],
) -> None:
    """Render compact match details in 3 aligned lines."""
    existing_name = existing.get("compound_name", "Unknown")
    submitted_name = submitted.get("compound_name", "Unknown")
    processed_at = _format_processed_datetime(existing.get("processed_at"))
    author_name = (existing.get("author_name") or "Unknown").strip()
    existing_threshold = existing.get("similarity_threshold", 90)
    existing_activities = existing.get("activity_types") or "N/A"

    # Header + config status
    st.markdown("**Match Details**")
    if is_identical_config:
        st.info(
            f"Same threshold ({existing_threshold}%) and activity types — results would be identical.",
            icon="ℹ️",
        )
    else:
        # Use suggestion-style info note for config differences.
        if config_match == "different_threshold":
            note_text = "Different threshold selected — results may differ from the existing analysis."
        elif config_match == "different_activities":
            note_text = "Different activity types selected — results may differ from the existing analysis."
        elif config_match == "different_both":
            note_text = "Threshold and activity types differ — results will likely differ."
        else:
            note_text = "Configuration differs from the existing analysis — results may differ."
        st.info(note_text, icon="ℹ️")

    # Row 1: match sentence
    if dup_type == "exact":
        st.caption(f"`{submitted_name}` matches `{existing_name}` by name and structure.")
    else:
        st.caption(f"`{submitted_name}` shares structure with `{existing_name}`.")

    # Row 2: existing config + activity types
    row2_left, row2_right = st.columns(2, gap="medium")
    with row2_left:
        st.caption(f"Existing config: threshold {existing_threshold}%")
    with row2_right:
        st.caption(f"Activity types: `{existing_activities}`")

    # Row 3: allowed actions + processed info
    row3_left, row3_right = st.columns(2, gap="medium")
    with row3_left:
        st.caption(f"Allowed actions: {_get_allowed_actions_text(dup_type, is_identical_config)}")
    with row3_right:
        st.caption(f"Processed: {processed_at} by {author_name}")

    # Optional concise diff details for non-identical configs.
    if not is_identical_config and config_diff:
        threshold = config_diff.get("similarity_threshold", {})
        activities = config_diff.get("activity_types", {})
        parts = []
        if threshold:
            parts.append(f"threshold {threshold.get('existing')}% -> {threshold.get('submitted')}%")
        if activities:
            parts.append(f"activity types `{activities.get('existing')}` -> `{activities.get('submitted')}`")
        if parts:
            st.caption("Submitted vs existing: " + " | ".join(parts))


@st.dialog("Duplicate Compound Found", width="large")
def duplicate_dialog(duplicate_info: dict):
    """Render the duplicate detection dialog as a native @st.dialog modal.

    Shows different options based on the type of duplicate and config match:
    - identical config: Only Replace/Skip (duplicate would be pointless)
    - different config: Show config diff, allow Replace/Duplicate/Skip

    Results are written to st.session_state[DUPLICATE_RESULT_KEY] and the
    dialog is closed via st.rerun(). The caller must .pop() the result key
    to consume it.

    Args:
        duplicate_info: Dict with duplicate_type, config_match, config_diff,
                       existing_compound, submitted
    """
    dup_type = duplicate_info.get("duplicate_type", "exact")
    existing = duplicate_info.get("existing_compound", {})
    submitted = duplicate_info.get("submitted", {})
    config_match = duplicate_info.get("config_match", "identical")
    config_diff = duplicate_info.get("config_diff")

    existing_name = existing.get("compound_name", "Unknown")
    submitted_name = submitted.get("compound_name", "Unknown")
    # Get suggested name from backend (calculates next available version, e.g., _v3 if _v2 exists)
    suggested_name = duplicate_info.get("suggested_name", f"{existing_name}_v2")

    is_identical_config = config_match == "identical"

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

    _render_match_details(
        existing=existing,
        submitted=submitted,
        dup_type=dup_type,
        config_match=config_match,
        is_identical_config=is_identical_config,
        config_diff=config_diff,
    )

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
            st.session_state[DUPLICATE_RESULT_KEY] = {
                "action": "cancel",
                "new_name": None,
            }
            st.rerun()

    with col2:
        if st.button("Continue", type="primary", width="stretch", key="duplicate_continue"):
            # Map change_name to duplicate action with new name
            if action == "change_name":
                if new_name and new_name.strip():
                    st.session_state[DUPLICATE_RESULT_KEY] = {
                        "action": "duplicate",
                        "new_name": new_name.strip(),
                    }
                    st.rerun()
                else:
                    st.error("Please enter a valid name")
            else:
                st.session_state[DUPLICATE_RESULT_KEY] = {
                    "action": action,
                    "new_name": new_name,
                }
                st.rerun()


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
