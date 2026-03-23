"""Duplicate compound detection dialog for IMPULATOR.

Displays a modal dialog when a user submits a compound that already exists,
allowing them to choose how to handle the duplicate.

Config-aware: distinguishes same-config duplicates (pointless re-run)
from different-config duplicates (genuinely different analysis).

Uses @st.dialog decorator for native modal behavior. Results are communicated
via session_state (not return values) because @st.dialog functions cannot
return values to the caller.
"""

import html
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
    """Render clear, visually distinct match details with side-by-side comparison."""
    existing_name = existing.get("compound_name", "Unknown")
    submitted_name = submitted.get("compound_name", "Unknown")
    processed_at = _format_processed_datetime(existing.get("processed_at"))
    author_name = (existing.get("author_name") or "Unknown").strip()
    existing_threshold = existing.get("similarity_threshold", 90)
    submitted_threshold = submitted.get("similarity_threshold", 90)

    # Normalize activity types for display
    existing_act = existing.get("activity_types") or []
    if isinstance(existing_act, str):
        existing_act = [a.strip() for a in existing_act.split(",") if a.strip()]
    submitted_act = submitted.get("activity_types") or []
    if isinstance(submitted_act, str):
        submitted_act = [a.strip() for a in submitted_act.split(",") if a.strip()]

    _ = config_diff  # consumed via direct comparison below

    # Clear comparison table — designed for readability (large text, obvious colors)
    existing_act_str = ", ".join(sorted(existing_act)) if existing_act else "All types (default)"
    submitted_act_str = ", ".join(sorted(submitted_act)) if submitted_act else "All types (default)"

    threshold_same = existing_threshold == submitted_threshold
    activities_same = set(existing_act) == set(submitted_act)

    # Build HTML table with clear color coding
    threshold_existing_style = ""
    threshold_submitted_style = ""
    act_existing_style = ""
    act_submitted_style = ""

    if not threshold_same:
        threshold_existing_style = "color: #d32f2f; text-decoration: line-through;"
        threshold_submitted_style = "color: #2e7d32; font-weight: bold;"
    if not activities_same:
        act_existing_style = "color: #d32f2f;"
        act_submitted_style = "color: #2e7d32; font-weight: bold;"

    table_html = f"""
    <table style="width: 100%; border-collapse: collapse; font-size: 16px; margin: 10px 0;">
        <thead>
            <tr style="border-bottom: 2px solid var(--text-color); opacity: 0.8;">
                <th style="text-align: left; padding: 8px 12px; width: 30%;"></th>
                <th style="text-align: left; padding: 8px 12px; width: 35%; color: #1565c0;">
                    📋 Existing Analysis</th>
                <th style="text-align: left; padding: 8px 12px; width: 35%; color: #e65100;">
                    🆕 Your Submission</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid var(--secondary-background-color);">
                <td style="padding: 8px 12px; font-weight: 600;">Threshold</td>
                <td style="padding: 8px 12px; {threshold_existing_style}">{existing_threshold}%</td>
                <td style="padding: 8px 12px; {threshold_submitted_style}">{submitted_threshold}%
                    {'<span style="color: #2e7d32;"> ✓ same</span>' if threshold_same else ' <span style="color: #e65100;">← changed</span>'}</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--secondary-background-color);">
                <td style="padding: 8px 12px; font-weight: 600;">Activity Types</td>
                <td style="padding: 8px 12px; {act_existing_style}">{html.escape(existing_act_str)}</td>
                <td style="padding: 8px 12px; {act_submitted_style}">{html.escape(submitted_act_str)}
                    {'<span style="color: #2e7d32;"> ✓ same</span>' if activities_same else ' <span style="color: #e65100;">← changed</span>'}</td>
            </tr>
            <tr>
                <td style="padding: 8px 12px; font-weight: 600;">Author</td>
                <td style="padding: 8px 12px;">{html.escape(author_name)}</td>
                <td style="padding: 8px 12px; color: var(--text-color); opacity: 0.5;">You</td>
            </tr>
            <tr>
                <td style="padding: 8px 12px; font-weight: 600;">Processed</td>
                <td style="padding: 8px 12px;">{processed_at}</td>
                <td style="padding: 8px 12px; color: var(--text-color); opacity: 0.5;">Now</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # Clear summary
    if is_identical_config:
        st.info("Same configuration — reprocessing would produce identical results.", icon="ℹ️")
    else:
        changes = []
        if not threshold_same:
            changes.append(f"threshold ({existing_threshold}% → {submitted_threshold}%)")
        if not activities_same:
            added = set(submitted_act) - set(existing_act)
            removed = set(existing_act) - set(submitted_act)
            if added:
                changes.append(f"added types: {', '.join(sorted(added))}")
            if removed:
                changes.append(f"removed types: {', '.join(sorted(removed))}")
        st.warning(f"Configuration differs: {'; '.join(changes)}. Results will differ.", icon="⚠️")


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
    suggested_name = duplicate_info.get("suggested_name", f"{existing_name}_v2")
    is_identical_config = config_match == "identical"

    # ─── CONFIRMATION MODE ───────────────────────────────────────────
    # If user already clicked Continue, show ONLY the compact confirmation
    confirm_key = "_dup_confirm_step"
    if st.session_state.get(confirm_key, False):
        pending_action = st.session_state.get("_dup_pending_action", "")
        pending_new_name = st.session_state.get("_dup_pending_new_name")

        if pending_action == "replace":
            st.error(
                f"**⚠️ Replace** will permanently delete **{existing_name}** "
                f"and reprocess with your new configuration. "
                f"Existing results will be **lost**."
            )
            confirm_btn_text = "🔴 Yes, Replace"
        elif pending_action in ("change_name", "duplicate"):
            display_name = pending_new_name or suggested_name
            st.success(
                f"**✅ Keep Both** — Save as **{display_name}** "
                f"alongside existing **{existing_name}**."
            )
            confirm_btn_text = "✅ Yes, Keep Both"
        elif pending_action == "skip":
            st.info(
                f"**Skip** — **{submitted_name}** will not be processed. "
                f"Nothing changes."
            )
            confirm_btn_text = "Skip"
        else:
            confirm_btn_text = "Confirm"

        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Go Back", width="stretch", key="dup_go_back"):
                st.session_state[confirm_key] = False
                st.rerun(scope="fragment")
        with c2:
            if st.button(confirm_btn_text, type="primary", width="stretch", key="dup_final_confirm"):
                st.session_state.pop(confirm_key, None)
                final_action = pending_action
                st.session_state.pop("_dup_pending_action", None)
                st.session_state.pop("_dup_pending_new_name", None)
                if final_action == "change_name":
                    st.session_state[DUPLICATE_RESULT_KEY] = {
                        "action": "duplicate",
                        "new_name": (pending_new_name or "").strip(),
                    }
                else:
                    st.session_state[DUPLICATE_RESULT_KEY] = {
                        "action": final_action,
                        "new_name": pending_new_name,
                    }
                st.rerun()
        return  # Don't render the full details below

    # ─── MAIN DIALOG CONTENT ─────────────────────────────────────────
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
        # Different config: allow all three options — default to "Keep both" (safest)
        if dup_type == "exact":
            action = st.radio(
                "Choose an action:",
                options=["change_name", "replace", "skip"],
                index=0,  # Default: Keep both (safest)
                format_func=lambda x: {
                    "change_name": "✅ Keep both (save as separate analysis)",
                    "replace": "⚠️ Replace existing (reprocess with new config — overwrites!)",
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
            # Structure-only + different config: full options — default to "Keep both"
            action = st.radio(
                "Choose an action:",
                options=["duplicate", "replace", "skip"],
                index=0,  # Default: Keep both (safest)
                format_func=lambda x: {
                    "duplicate": f"✅ Keep both (save as separate analysis of {existing_name})",
                    "replace": f"⚠️ Replace existing '{existing_name}' (overwrites!)",
                    "skip": "Skip (don't process)",
                }.get(x, x),
                key="duplicate_action_structure",
                label_visibility="collapsed"
            )
            new_name = None

    st.divider()

    # Action buttons — Continue goes to confirmation (rendered at top of dialog on next render)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", width="stretch", key="duplicate_cancel"):
            st.session_state[DUPLICATE_RESULT_KEY] = {"action": "cancel", "new_name": None}
            st.rerun()
    with col2:
        can_continue = True
        if action == "change_name" and (not new_name or not new_name.strip()):
            can_continue = False
        if st.button("Continue →", type="primary", width="stretch", key="duplicate_continue", disabled=not can_continue):
            st.session_state["_dup_confirm_step"] = True
            st.session_state["_dup_pending_action"] = action
            st.session_state["_dup_pending_new_name"] = new_name
            st.rerun(scope="fragment")


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
