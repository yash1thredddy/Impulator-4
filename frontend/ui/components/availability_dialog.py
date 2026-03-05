"""Availability dialog component for IMPULATOR.

Shows ChEMBL data availability at different similarity thresholds
when no data is found at the user's requested threshold.
Allows user to pick an alternative threshold or view existing compounds.

This dialog replaces both the availability check AND the duplicate dialog
when a compound has no ChEMBL data at the requested threshold — existing
compound details are shown inline with full config comparison.
"""

import html
import logging
from datetime import datetime
from typing import Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# CSS to increase radio button label font size for readability
_RADIO_CSS = """
<style>
    div[data-testid="stRadio"] label {
        font-size: 1.1rem !important;
        padding: 0.4rem 0 !important;
    }
    div[data-testid="stRadio"] label p {
        font-size: 1.1rem !important;
    }
</style>
"""


def _recompute_config_match(
    existing_compound: dict,
    selected_threshold: int,
    submitted_activity_types: str,
) -> str:
    """Re-evaluate config match against the user's SELECTED threshold.

    Returns one of: 'identical', 'different_threshold', 'different_activities', 'different_both'
    """
    ec_threshold = existing_compound.get("similarity_threshold") or 90
    ec_activities = existing_compound.get("activity_types", "") or ""

    threshold_same = ec_threshold == selected_threshold

    # Backend default activity types — treat empty/missing as this set
    _DEFAULT_ACTIVITIES = "AC50,EC50,GI50,IC50,Kd,Ki,MIC"

    submitted_norm = submitted_activity_types.strip() if submitted_activity_types else ""
    existing_norm = ec_activities.strip() if ec_activities else ""

    # Normalize empty to default set for comparison
    submitted_set = frozenset(a.strip() for a in (submitted_norm or _DEFAULT_ACTIVITIES).split(",") if a.strip())
    existing_set = frozenset(a.strip() for a in (existing_norm or _DEFAULT_ACTIVITIES).split(",") if a.strip())
    activities_same = submitted_set == existing_set

    if threshold_same and activities_same:
        return "identical"
    elif not threshold_same and activities_same:
        return "different_threshold"
    elif threshold_same and not activities_same:
        return "different_activities"
    return "different_both"


def _format_processed_datetime(raw_timestamp: Optional[str]) -> str:
    """Format ISO timestamp into a human-readable form."""
    if not raw_timestamp:
        return "N/A"
    ts = str(raw_timestamp).strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%b %d, %Y at %I:%M %p")
    except (ValueError, TypeError):
        return str(raw_timestamp).replace("T", " ")


def _render_existing_compound_section(
    existing_compound: dict,
    selected_threshold: int,
    config_match: str,
    submitted_at: str,
) -> None:
    """Render detailed existing compound info with config comparison.

    Styled similarly to the duplicate dialog for consistency.
    """
    ec_name = html.escape(existing_compound.get("compound_name", "Unknown"))
    ec_threshold = existing_compound.get("similarity_threshold") or 90
    ec_activities = existing_compound.get("activity_types", "") or "N/A"
    ec_author = existing_compound.get("author_name") or "Unknown"
    imp_score = existing_compound.get("imp_score")
    processed_at = _format_processed_datetime(existing_compound.get("processed_at"))

    with st.container(border=True):
        st.markdown(f"#### Existing Analysis: {ec_name}")

        # Config match status
        is_identical = config_match == "identical"

        if is_identical:
            score_text = f" — IMP Score: **{imp_score:.2f}**" if imp_score is not None else ""
            st.info(
                f"This compound was already analyzed with **identical settings** "
                f"(threshold {ec_threshold}%, same activity types).{score_text}\n\n"
                f"Re-running would produce the same results.",
                icon="ℹ️",
            )
        else:
            diff_labels = {
                "different_threshold": "a **different similarity threshold**",
                "different_activities": "**different activity types**",
                "different_both": "a **different threshold and activity types**",
            }
            diff_desc = diff_labels.get(config_match, "a **different configuration**")
            st.warning(
                f"This compound was previously analyzed with {diff_desc}.",
                icon="⚠️",
            )

        # Side-by-side config comparison
        col_existing, col_new = st.columns(2)

        with col_existing:
            st.markdown("**Existing**")
            th_style = "🔴 " if config_match in ("different_threshold", "different_both") else ""
            at_style = "🔴 " if config_match in ("different_activities", "different_both") else ""
            st.markdown(f"{th_style}Threshold: **{ec_threshold}%**")
            st.markdown(f"{at_style}Activities: **{ec_activities}**")

        with col_new:
            st.markdown("**Your Submission**")
            st.markdown(f"{th_style}Threshold: **{selected_threshold}%**")
            st.markdown(f"{at_style}Activities: **{submitted_at or 'N/A'}**")

        # Author and processed info
        st.caption(f"Author: {html.escape(ec_author)}  ·  Analyzed: {processed_at}")


def render_availability_dialog(
    avail_info: dict,
    compound_name: str,
) -> Tuple[Optional[str], Optional[int], Optional[dict], Optional[str], Optional[str]]:
    """Render the availability dialog for a single compound.

    Combines availability check AND duplicate information in one dialog.
    Shows all thresholds at a glance with existing compound details.

    Returns:
        Tuple of (action, threshold, existing_compound_dict, duplicate_action, new_compound_name):
        - ("submit", chosen_threshold, existing_or_None, "duplicate"|"replace"|None, new_name_or_None)
        - ("view_existing", None, existing_compound_dict, None, None) — navigate to existing
        - ("cancel", None, None, None, None) — abort
        - (None, None, None, None, None) — dialog still showing
    """
    requested_threshold = st.session_state.get('availability_requested_threshold', 90)
    submitted_at = st.session_state.get('availability_activity_types_str', "")
    safe_name = html.escape(compound_name)

    thresholds = avail_info.get("thresholds", [])
    existing_compounds = avail_info.get("existing_compounds", [])

    # Build threshold options (only those with data)
    available_thresholds = [t for t in thresholds if t.get("count", 0) > 0]

    if not available_thresholds:
        st.error("No similar compounds found in ChEMBL at any threshold (40%-100%).")
        if st.button("OK", key="avail_no_data_ok"):
            return ("cancel", None, None, None, None)
        return (None, None, None, None, None)

    # --- Inject CSS for larger radio labels ---
    st.markdown(_RADIO_CSS, unsafe_allow_html=True)

    # --- Header ---
    st.markdown(
        f"### No ChEMBL data found for {safe_name} at {requested_threshold}% similarity"
    )

    # If there's an existing compound, show duplicate notice prominently
    if existing_compounds:
        ec = existing_compounds[0]
        ec_name = html.escape(ec.get("compound_name", "Unknown"))
        st.markdown(
            f"This compound shares its structure with **{ec_name}** "
            f"(analyzed at {ec.get('similarity_threshold', '?')}% threshold)."
        )
    else:
        st.markdown(
            "Choose an alternative similarity threshold below. "
            "Lower thresholds find more, but less similar, compounds."
        )

    # Show thresholds with no data
    zero_thresholds = [t for t in thresholds if t.get("count", 0) == 0]
    if zero_thresholds:
        zero_info = ", ".join(
            f"{t['threshold']}%"
            for t in sorted(zero_thresholds, key=lambda x: x["threshold"], reverse=True)
        )
        st.markdown(f"**No ChEMBL data can be found at:** {zero_info}")

    # --- Threshold Selection (radio — all visible at once) ---
    st.markdown("#### Select Similarity Threshold")

    sorted_thresholds = sorted(available_thresholds, key=lambda x: x["threshold"], reverse=True)

    radio_labels = []
    threshold_values = []

    for t in sorted_thresholds:
        th = t["threshold"]
        count = t["count"]
        label = f"**{th}%** — {count} compound{'s' if count != 1 else ''}"

        # Check for existing compound at this threshold (re-evaluated config)
        # Prefer identical match over different-config matches
        best_ec = None
        best_match = None
        for ec in existing_compounds:
            match = _recompute_config_match(ec, th, submitted_at)
            if match == "identical":
                best_ec = ec
                best_match = match
                break  # identical is the best possible — stop early
            elif best_ec is None:
                best_ec = ec
                best_match = match

        if best_ec is not None:
            ec_name = best_ec.get("compound_name", "?")
            if best_match == "identical":
                label += f"  ·  ✅ {ec_name} (same config — already done)"
            else:
                diff_short = {
                    "different_threshold": "different threshold",
                    "different_activities": "different activities",
                    "different_both": "different config",
                }.get(best_match, "different config")
                label += f"  ·  ⚠️ {ec_name} ({diff_short})"

        radio_labels.append(label)
        threshold_values.append(th)

    selected_idx = st.radio(
        "Select a threshold",
        range(len(radio_labels)),
        format_func=lambda i: radio_labels[i],
        key="avail_threshold_radio",
        label_visibility="collapsed",
    )

    selected_threshold = threshold_values[selected_idx]

    st.markdown("")  # spacer

    # --- Existing Compound Details (if any) ---
    identical_match = None
    different_match = None
    different_match_type = None

    for ec in existing_compounds:
        match = _recompute_config_match(ec, selected_threshold, submitted_at)
        if match == "identical":
            identical_match = ec
            break
        elif different_match is None:
            different_match = ec
            different_match_type = match

    active_match = identical_match or different_match
    active_match_type = "identical" if identical_match else different_match_type

    if active_match and active_match_type:
        _render_existing_compound_section(
            active_match, selected_threshold, active_match_type, submitted_at
        )

    st.markdown("")  # spacer

    # --- Action Buttons (direct — one click per action) ---
    if active_match and active_match_type == "identical":
        # Same config: View existing | Replace | Cancel
        ec_btn = html.escape(active_match.get('compound_name', '?')[:25])
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Cancel", key="avail_cancel", use_container_width=True):
                return ("cancel", None, None, None, None)
        with col2:
            if st.button(
                f"Replace at {selected_threshold}%",
                key="avail_replace",
                use_container_width=True,
            ):
                return ("submit", selected_threshold, active_match, "replace", None)
        with col3:
            if st.button(
                f"View {ec_btn}",
                key="avail_view_existing",
                type="primary",
                use_container_width=True,
            ):
                return ("view_existing", None, active_match, None, None)

    elif active_match:
        # Different config: Keep Both | Replace | Cancel
        # For "Keep Both", check if name collides and ask for a new name
        ec_name = active_match.get("compound_name", "")
        names_collide = compound_name.strip().lower() == ec_name.strip().lower()

        new_keep_both_name = None
        if names_collide:
            # Count existing versions to suggest a unique name
            existing_compounds = avail_info.get("existing_compounds", [])
            version_count = len(existing_compounds) + 1
            suggested_name = f"{compound_name}_v{version_count}"
            new_keep_both_name = st.text_input(
                "Name for new copy (must differ from existing)",
                value=st.session_state.get("avail_keep_both_name", suggested_name),
                key="avail_keep_both_name_input",
                help="Since both compounds share the same name, provide a distinct name for the new analysis.",
            )
            st.session_state["avail_keep_both_name"] = new_keep_both_name

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Cancel", key="avail_cancel", use_container_width=True):
                return ("cancel", None, None, None, None)
        with col2:
            if st.button(
                f"Replace at {selected_threshold}%",
                key="avail_replace",
                use_container_width=True,
            ):
                return ("submit", selected_threshold, active_match, "replace", None)
        with col3:
            # Disable Keep Both if names collide and no valid new name provided
            # Also block if user typed the same name as the existing compound
            keep_both_disabled = names_collide and (
                not (new_keep_both_name and new_keep_both_name.strip())
                or (new_keep_both_name and new_keep_both_name.strip().lower() == ec_name.strip().lower())
            )
            if st.button(
                f"Keep Both at {selected_threshold}%",
                key="avail_submit",
                type="primary",
                use_container_width=True,
                disabled=keep_both_disabled,
            ):
                final_new_name = new_keep_both_name.strip() if names_collide and new_keep_both_name else None
                return ("submit", selected_threshold, active_match, "duplicate", final_new_name)

    else:
        # No existing compound: Submit | Cancel
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", key="avail_cancel", use_container_width=True):
                return ("cancel", None, None, None, None)
        with col2:
            if st.button(
                f"Submit at {selected_threshold}%",
                key="avail_submit",
                type="primary",
                use_container_width=True,
            ):
                return ("submit", selected_threshold, None, None, None)

    return (None, None, None, None, None)


def clear_availability_state():
    """Clear all availability dialog session state."""
    keys_to_clear = [
        'show_availability_dialog',
        'pending_availability_info',
        'availability_smiles',
        'availability_compound_name',
        'availability_author_name',
        'availability_activity_types',
        'availability_requested_threshold',
        'availability_activity_types_str',
        'avail_keep_both_name',
        'avail_threshold_radio',
        'avail_keep_both_name_input',
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
