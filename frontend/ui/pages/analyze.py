"""Analyze page for IMPULATOR.

Provides the interface for submitting new compound analysis jobs.
"""

import logging
import re

import streamlit as st

from frontend.services import get_api_client
from frontend.config.settings import config
from frontend.utils import SessionState, sanitize_compound_name
from frontend.ui.components import render_job_form, render_csv_upload_form
from frontend.ui.components.job_form import (
    render_activity_checkboxes,
    _render_column_mapping_ui,
    _apply_column_mapping,
    _build_compact_preview,
    _inchi_to_smiles,
)
from frontend.ui.components.collection_preflight import (
    compute_inchikey as _compute_inchikey,
    group_in_file_duplicates,
    build_preflight_plan,
    apply_preflight_decisions,
    distinct_thresholds,
)

logger = logging.getLogger(__name__)

# Client-side UX mirror of backend.models.schemas.COMPOUND_NAME_PATTERN (D-03).
# The backend schema is AUTHORITATIVE — this is a fast UX guard only so the user
# sees an inline error before the round-trip. Keep the character class identical
# to backend/models/schemas.py::COMPOUND_NAME_PATTERN.
COMPOUND_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9\-_\s\(\)\[\]',\.]+$")

# D-06 member-count bounds (mirror backend COLLECTION_MIN_MEMBERS / MAX_MEMBERS).
COLLECTION_MIN_MEMBERS = 2
COLLECTION_MAX_MEMBERS = 100

# Legibility soft-cap: overlaid chart series get hard to read past ~8 (UI-SPEC).
COLLECTION_SOFT_CAP = 8


def render_analyze_page() -> None:
    """Render the analyze page with job submission form."""
    if st.button("⬅ Back", key="analyze_back_btn"):
        SessionState.navigate_to_home()
        st.rerun()

    st.markdown(
        "<h1 style='text-align:center;margin:-1rem 0 0 0;padding:0;'>New Analysis</h1>"
        "<p style='text-align:center;color:#888;font-size:14px;margin:0 0 4px 0;'>"
        "Submit a compound for IMP analysis</p>",
        unsafe_allow_html=True,
    )

    # Tabs for single vs batch vs collection
    tab1, tab2, tab3 = st.tabs(["Single Compound", "Batch Upload", "Collection"])

    with tab1:
        render_single_analysis()

    with tab2:
        render_batch_analysis()

    with tab3:
        render_collection_analysis()


def render_single_analysis() -> None:
    """Render the single compound analysis form."""
    # Check if we just submitted a job (stored in session state)
    just_submitted = SessionState.get('just_submitted_job', False)
    last_job_id = SessionState.get('last_submitted_job_id', None)

    job_id = render_job_form()

    # If a new job was submitted, store it in session state and rerun
    # so the sidebar fragment picks up the polling flag immediately
    if job_id:
        SessionState.set('just_submitted_job', True)
        SessionState.set('last_submitted_job_id', job_id)
        st.rerun()

    if just_submitted and last_job_id:
        # Job submitted successfully
        st.success("Analysis job submitted!")
        st.info("The job is now processing. You can monitor progress in the sidebar.")

        # Option to view status or start another
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏠 Go to Home", key="go_home_after_submit", width='stretch'):
                # Clear submission state
                SessionState.set('just_submitted_job', False)
                SessionState.set('last_submitted_job_id', None)
                SessionState.navigate_to_home()
                st.rerun()

        with col2:
            if st.button("➕ Submit Another", key="submit_another_btn", width='stretch'):
                # Clear form state
                SessionState.set('just_submitted_job', False)
                SessionState.set('last_submitted_job_id', None)
                SessionState.reset_processing_state()
                st.rerun()


def render_batch_analysis() -> None:
    """Render the batch upload form."""
    # Check if we just submitted a batch (stored in session state)
    just_submitted_batch = SessionState.get('just_submitted_batch', False)
    last_batch_id = SessionState.get('last_submitted_batch_id', None)

    # Show post-submission UI if batch was just submitted
    if just_submitted_batch and last_batch_id:
        st.success("Batch submitted successfully!")
        st.info("Jobs are now processing. You can monitor progress in the sidebar.")

        # Option to go home or submit another batch
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏠 Go to Home", key="go_home_after_batch", width='stretch'):
                # Clear submission state
                SessionState.set('just_submitted_batch', False)
                SessionState.set('last_submitted_batch_id', None)
                _clear_batch_form_state()
                SessionState.navigate_to_home()
                st.rerun()

        with col2:
            if st.button("➕ Submit Another Batch", key="submit_another_batch_btn", width='stretch'):
                # Clear form state for new submission
                SessionState.set('just_submitted_batch', False)
                SessionState.set('last_submitted_batch_id', None)
                _clear_batch_form_state()
                st.rerun()

        return

    # Normal batch upload form
    st.markdown("### Batch Processing")
    st.info("""
    Upload a CSV file to analyze multiple compounds at once.

    **Required columns:**
    - `compound_name`: Unique name for each compound
    - `smiles` or `inchi`: Chemical structure

    Each compound will be submitted as a separate job.
    """)

    batch_id = render_csv_upload_form()

    # If a batch was submitted, store it in session state
    if batch_id:
        SessionState.set('just_submitted_batch', True)
        SessionState.set('last_submitted_batch_id', batch_id)
        st.rerun()


def _validate_collection_name(value: str, field_label: str) -> str | None:
    """Client-side UX validation of a collection name / author_name (D-03).

    Mirrors backend.models.schemas COMPOUND_NAME_PATTERN + path-traversal guard.
    The backend re-validates authoritatively — this only surfaces the error
    inline so the user does not need a round-trip to learn the name is invalid.

    Returns an error message string, or None if the value is valid.
    """
    v = (value or "").strip()
    if not v:
        return f"Please enter a {field_label}."
    if not COMPOUND_NAME_PATTERN.match(v):
        return f"{field_label} contains invalid characters."
    # COMPOUND_NAME_PATTERN allows '.', so '..' passes the regex — guard separately.
    if ".." in v or "/" in v or "\\" in v or "\x00" in v:
        return f"{field_label} contains invalid path characters."
    return None


def _dedupe_members_by_structure(
    members: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int, list[str]]:
    """Drop intra-set InChIKey collisions, keeping the first occurrence (D-08).

    Reuses `group_in_file_duplicates` so the surfacing path (Plan 2 pre-flight)
    and this legacy non-preflight fallback share one grouping implementation.
    Members whose SMILES cannot be parsed are kept (reported as invalid).
    """
    invalid: list[str] = [
        str(m.get("name", "(unnamed)"))
        for m in members
        if _compute_inchikey(m.get("smiles", "")) is None
    ]
    groups = group_in_file_duplicates(members)
    drop: set[int] = set()
    for g in groups:
        drop.update(g.member_indices[1:])  # keep first
    deduped = [m for i, m in enumerate(members) if i not in drop]
    return deduped, len(drop), invalid


def _build_members_from_mapped(
    df_mapped,
    *,
    similarity_threshold: int,
    activity_types: list[str],
    inchikey_smiles_map: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    """Turn a mapped CSV dataframe into stamped collection-member dicts.

    Each row's structure is resolved with a SMILES → InChI → InChIKey fallback
    (mirroring the Batch upload). ``CollectionMember`` requires a SMILES, so
    InChI is converted via RDKit and InChIKey via the caller-supplied
    ``inchikey_smiles_map`` (resolved upstream through PubChem).

    The chosen shared ``similarity_threshold`` / ``activity_types`` are STAMPED
    onto every member dict. This is load-bearing: the backend reads them
    per-member (``process_collection_member`` →
    ``int(member_input.get("similarity_threshold", 90))``), and an unstamped
    member is stored with ``similarity_threshold=None`` and crashes that member
    on ``int(None)``. Stamping is what makes the shared config actually apply.

    Returns ``(members, report)`` with three exclusion buckets:
    ``report["invalid_names"]`` lists rows whose name fails the D-03 name
    whitelist (would 422 the whole POST), ``report["skipped_no_structure"]``
    lists rows that had no resolvable structure, and
    ``report["invalid_structure"]`` lists rows whose SMILES is present but
    unparseable by RDKit (would also 422 the whole availability batch).
    """
    inchikey_smiles_map = inchikey_smiles_map or {}
    members: list[dict[str, object]] = []
    skipped_no_structure: list[str] = []
    invalid_names: list[str] = []
    invalid_structure: list[str] = []

    has_smiles = "smiles" in df_mapped.columns
    has_inchi = "inchi" in df_mapped.columns
    has_inchikey = "inchikey" in df_mapped.columns

    def _clean(value) -> str:
        text = str(value).strip()
        return "" if text.lower() in ("nan", "none") else text

    for row in df_mapped.to_dict("records"):
        name = _clean(row.get("compound_name", ""))
        if not name:
            continue

        # D-03 name whitelist: the backend rejects the ENTIRE collection on one
        # bad name, so flag invalid rows here per-row instead of round-tripping
        # to a single opaque 422.
        if _validate_collection_name(name, "Member name") is not None:
            invalid_names.append(name)
            continue

        smiles = _clean(row.get("smiles", "")) if has_smiles else ""
        if not smiles and has_inchi:
            inchi = _clean(row.get("inchi", ""))
            if inchi:
                smiles = _inchi_to_smiles(inchi) or ""
        if not smiles and has_inchikey:
            key = _clean(row.get("inchikey", "")).upper()
            if key:
                smiles = inchikey_smiles_map.get(key, "") or ""

        if not smiles:
            skipped_no_structure.append(name)
            continue

        # A present-but-unparseable SMILES clears the empty check above yet 422s
        # the WHOLE availability batch: the backend runs RDKit on every
        # CompoundInput and rejects the entire POST on one bad structure. Drop +
        # report it per-row (same _compute_inchikey gate the dedupe path uses) so
        # one junk cell can't sink the collection before the rest is probed.
        if _compute_inchikey(smiles) is None:
            invalid_structure.append(name)
            continue

        members.append(
            {
                "name": name,
                "smiles": smiles,
                "similarity_threshold": int(similarity_threshold),
                "activity_types": list(activity_types),
            }
        )

    return members, {
        "skipped_no_structure": skipped_no_structure,
        "invalid_names": invalid_names,
        "invalid_structure": invalid_structure,
    }


def clear_collection_preflight_state() -> None:
    """Clear all pre-flight session keys (cancel / submit / new upload)."""
    for k in list(st.session_state.keys()):
        if k.startswith("collection_preflight"):
            del st.session_state[k]


def _run_collection_preflight(
    name: str,
    author_name: str,
    df_mapped,
    similarity_threshold: int,
    activity_types: list[str],
) -> None:
    """Phase 1: validate, build members, resolve InChIKeys, probe ChEMBL
    availability, build the pre-flight plan, and stash it for rendering.

    Blocks (D-PF-5) if the availability probe fails wholesale — a dead ChEMBL
    means every member's compute would fail too, so submitting is pointless.
    """
    # --- validation (mirror _submit_collection) ---
    name_error = _validate_collection_name(name, "Collection name")
    if name_error:
        st.error(name_error)
        return
    author_error = _validate_collection_name(author_name, "Author name")
    if author_error:
        st.error(author_error)
        return
    if not activity_types:
        st.error("Please select at least one activity type.")
        return

    # --- resolve InChIKey-only rows (reuse existing block) ---
    inchikey_smiles_map: dict[str, str] = {}
    if "inchikey" in df_mapped.columns:
        has_smiles_col = "smiles" in df_mapped.columns
        has_inchi_col = "inchi" in df_mapped.columns
        keys_to_resolve: list[str] = []
        for row in df_mapped.to_dict("records"):
            smiles_val = str(row.get("smiles", "")).strip().lower() if has_smiles_col else ""
            inchi_val = str(row.get("inchi", "")).strip().lower() if has_inchi_col else ""
            if smiles_val in ("", "nan", "none") and inchi_val in ("", "nan", "none"):
                key_val = str(row.get("inchikey", "")).strip().upper()
                if key_val and key_val.lower() not in ("nan", "none"):
                    keys_to_resolve.append(key_val)
        if keys_to_resolve:
            with st.spinner(f"Resolving {len(set(keys_to_resolve))} InChIKey(s) via PubChem..."):
                try:
                    inchikey_smiles_map = get_api_client().resolve_inchikeys_batch(
                        list(set(keys_to_resolve))
                    )
                except Exception as e:
                    logger.warning(f"Collection InChIKey resolution failed: {e}")

    members, build_report = _build_members_from_mapped(
        df_mapped, similarity_threshold=similarity_threshold,
        activity_types=activity_types, inchikey_smiles_map=inchikey_smiles_map,
    )

    invalid_names = build_report.get("invalid_names", [])
    if invalid_names:
        st.error(
            f"{len(invalid_names)} member name(s) contain invalid characters. "
            "Fix these in your CSV and re-upload: "
            f"{', '.join(invalid_names[:5])}" + ("…" if len(invalid_names) > 5 else "")
        )
        return
    if build_report.get("skipped_no_structure"):
        skipped = build_report["skipped_no_structure"]
        st.warning(
            f"{len(skipped)} row(s) had no usable structure and were left out: "
            f"{', '.join(skipped[:5])}" + ("…" if len(skipped) > 5 else "")
        )
    if build_report.get("invalid_structure"):
        bad = build_report["invalid_structure"]
        st.warning(
            f"{len(bad)} row(s) had an unreadable SMILES and were left out: "
            f"{', '.join(bad[:5])}" + ("…" if len(bad) > 5 else "")
        )
    if not members:
        st.error("No members with a usable structure were found in the uploaded CSV.")
        return

    # --- ChEMBL availability probe (D-PF-5: block on wholesale failure) ---
    members_for_check = [{"compound_name": m["name"], "smiles": m["smiles"]} for m in members]
    with st.spinner(f"Checking ChEMBL data availability for {len(members)} members..."):
        try:
            avail = get_api_client().check_availability_batch(
                compounds=members_for_check,
                similarity_threshold=similarity_threshold,
                activity_types=activity_types,
            )
        except Exception as e:
            logger.warning(f"Collection availability probe failed: {e}")
            avail = {"success": False, "error": str(e)}

    # success==False only catches the *backend* being down (wrapper sets it on
    # a non-200 / network error). It does NOT catch ChEMBL being down — the
    # backend swallows ChEMBL probe errors as count=0 and returns 200.
    if not avail.get("success"):
        st.error(
            "Couldn't reach the analysis service to check data availability. "
            "Please try again later."
        )
        return

    # NOTE: avail["results"] are plain dicts (HTTP .json()), already the
    # list[dict] build_preflight_plan expects — no model_dump needed.
    plan = build_preflight_plan(members, avail.get("results") or [], similarity_threshold)

    # D-PF-5 heuristic block: probe errors are swallowed as count=0 upstream
    # (api_client.probe_all_thresholds:1439), so an ALL-no_data result means
    # either ChEMBL is down OR none of these compounds have ChEMBL data. Either
    # way the collection can't run — block instead of silently excluding all.
    if len(members) > 0 and plan.no_data_count == len(members):
        st.error(
            "Couldn't verify ChEMBL data — ChEMBL may be down, or none of these "
            "compounds have ChEMBL data. Please try again later."
        )
        return

    st.session_state["collection_preflight"] = {
        "plan": plan,
        "members": members,
        "name": name.strip(),
        "author_name": author_name.strip(),
        "similarity_threshold": similarity_threshold,
        "activity_types": activity_types,
        "skipped_no_structure": len(build_report.get("skipped_no_structure", [])),
    }
    st.rerun()


def _confirm_collection_submit() -> None:
    """Phase 2 submit: gather widget decisions, apply, re-check D-06, POST."""
    pf = st.session_state["collection_preflight"]
    plan = pf["plan"]
    members = pf["members"]

    # dup decisions: {inchikey: "first"|"both"}
    dup_decisions: dict[str, str] = {}
    for g in plan.dup_groups:
        choice = st.session_state.get(f"collection_preflight_dup_{g.inchikey}", "Keep first only")
        dup_decisions[g.inchikey] = "both" if choice == "Keep both" else "first"

    # threshold + exclusion decisions keyed by INDEX (D-PF-7)
    threshold_decisions: dict[int, int] = {}
    excluded_indices: set[int] = set()
    for idx, m in enumerate(plan.members):
        if m.status == "needs_lower":
            label = st.session_state.get(f"collection_preflight_thr_{idx}")
            if label:
                threshold_decisions[idx] = int(label.split("%")[0])
        elif m.status == "no_data":
            excluded_indices.add(idx)

    final_members = apply_preflight_decisions(
        members, dup_decisions, threshold_decisions, excluded_indices
    )

    # duplicates actually removed = kept-first drops NOT already excluded
    from frontend.ui.components.collection_preflight import group_in_file_duplicates
    dup_dropped: set[int] = set()
    for g in group_in_file_duplicates(members):
        if dup_decisions.get(g.inchikey, "first") != "both":
            dup_dropped.update(g.member_indices[1:])
    removed_dups = len(dup_dropped - excluded_indices)

    # --- D-06 bounds re-checked AFTER exclusions/dedupe ---
    count = len(final_members)
    if count < COLLECTION_MIN_MEMBERS:
        st.error(
            f"Only {count} member(s) remain after excluding no-data compounds and "
            f"duplicates — a collection needs at least {COLLECTION_MIN_MEMBERS}."
        )
        return
    if count > COLLECTION_MAX_MEMBERS:
        st.error(f"A collection may have at most {COLLECTION_MAX_MEMBERS} members (got {count}).")
        return

    tiers = distinct_thresholds(final_members)
    mixed_note = (
        f"Members analyzed at mixed thresholds ({tiers[0]}–{tiers[-1]}%); "
        "results may not be directly comparable." if len(tiers) > 1 else ""
    )

    with st.spinner("Creating collection..."):
        try:
            result = get_api_client().create_collection(
                name=pf["name"], members=final_members, author_name=pf["author_name"],
                similarity_threshold=pf["similarity_threshold"],
                activity_types=pf["activity_types"],
            )
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            st.error(f"Error: {e}")
            return

    if result.get("success"):
        SessionState.set('just_submitted_collection', True)
        SessionState.set('last_submitted_collection', {
            "id": result.get("id"), "job_id": result.get("job_id"),
            "name": sanitize_compound_name(pf["name"]),
            "deduped": removed_dups,
            "skipped_no_structure": pf.get("skipped_no_structure", 0),
            "mixed_note": mixed_note,
        })
        clear_collection_preflight_state()
        st.rerun()
    else:
        st.error(f"Error: {result.get('error', 'unknown')}")


def _render_collection_preflight() -> None:
    """Phase 2 view: in-file duplicates, ChEMBL availability, summary, confirm."""
    pf = st.session_state["collection_preflight"]
    plan = pf["plan"]

    st.markdown("### Pre-flight check")

    # --- In-File Duplicates ---
    if plan.dup_groups:
        st.subheader("In-File Duplicates")
        st.caption("Members sharing an identical structure. Default: keep the first.")
        for g in plan.dup_groups:
            st.radio(
                f"{g.names[0]} ↔ {', '.join(g.names[1:])} (same structure)",
                options=["Keep first only", "Keep both"],
                index=0,
                key=f"collection_preflight_dup_{g.inchikey}",
                horizontal=True,
            )

    # --- ChEMBL Data Availability ---
    st.subheader("ChEMBL Data Availability")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ready", plan.ready_count)
    c2.metric("Need lower %", plan.needs_lower_count)
    c3.metric("Excluded (no data)", plan.no_data_count)

    # Widgets keyed by member INDEX (D-PF-7) — names aren't unique.
    for idx, m in enumerate(plan.members):
        if m.status == "needs_lower":
            labels = [f"{t['threshold']}% ({t['count']} compounds)" for t in m.tiers]
            default_idx = next(
                (i for i, t in enumerate(m.tiers) if t["threshold"] == m.suggested_threshold), 0
            )
            st.selectbox(
                f"**{m.name}** — no data at {m.requested_threshold}%; pick a lower threshold",
                options=labels,
                index=default_idx,
                key=f"collection_preflight_thr_{idx}",
            )
        elif m.status == "unknown":
            st.caption(f"**{m.name}** — availability unknown; will attempt at {m.requested_threshold}%")

    no_data = [m.name for m in plan.members if m.status == "no_data"]
    if no_data:
        st.info(
            "Excluded (no ChEMBL data at any threshold): " + ", ".join(no_data[:10])
            + ("…" if len(no_data) > 10 else "")
        )

    # --- Summary (pre-submit, spec §3.2) ---
    st.subheader("Summary")
    will_process = plan.ready_count + plan.needs_lower_count
    # in-file dups that WILL be removed given current radio state (default keep-first)
    dup_removed = sum(
        len(g.member_indices) - 1
        for g in plan.dup_groups
        if st.session_state.get(f"collection_preflight_dup_{g.inchikey}", "Keep first only")
        != "Keep both"
    )
    st.success(f"**{will_process - dup_removed}** member(s) will be processed")
    if plan.no_data_count:
        st.info(f"{plan.no_data_count} excluded — no ChEMBL data at any threshold")
    if dup_removed:
        st.info(f"{dup_removed} in-file duplicate(s) will be removed")
    if plan.needs_lower_count:
        st.warning(
            "Some members will run at a lower threshold — members may be compared "
            "at mixed thresholds; results may not be directly comparable."
        )

    # --- Confirm / Cancel ---
    st.divider()
    col_ok, col_cancel = st.columns(2)
    with col_ok:
        confirm = st.button("✅ Confirm & Submit", type="primary", width='stretch',
                            key="collection_preflight_confirm")
    with col_cancel:
        if st.button("❌ Cancel", width='stretch', key="collection_preflight_cancel"):
            clear_collection_preflight_state()
            st.rerun()

    if confirm:
        _confirm_collection_submit()  # Task 4


def render_collection_analysis() -> None:
    """Render the Collection input tab.

    Collects a named/authored member set (paste "Name, SMILES" per line),
    runs the shared availability pre-check, auto-dedupes intra-set InChIKey
    collisions (D-08), enforces the D-06 member bounds, and submits ONE
    collection job via ``backend_client.create_collection`` (POST /collections).
    """
    # Post-submission view
    just_submitted = SessionState.get('just_submitted_collection', False)
    last_collection = SessionState.get('last_submitted_collection', None)

    if just_submitted and last_collection:
        st.success("Collection created!")
        st.info(
            "The collection is now processing. Collection ID: "
            f"`{last_collection.get('id', 'unknown')}`"
        )

        # Re-emit the D-08 dedupe warning here: it was drawn before the submit
        # st.rerun() (which discards pre-rerun elements), so the success view
        # is the only place the user reliably sees it.
        removed = last_collection.get("deduped", 0)
        if removed > 0:
            st.warning(
                f"{removed} duplicate compound(s) removed from this collection (identical structure)."
            )
        mixed_note = last_collection.get("mixed_note", "")
        if mixed_note:
            st.warning(mixed_note)
        skipped = last_collection.get("skipped_no_structure", 0)
        if skipped > 0:
            st.warning(
                f"{skipped} row(s) had no usable structure (SMILES/InChI/InChIKey) "
                "and were left out of this collection."
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Go to Home", key="go_home_after_collection", width='stretch'):
                SessionState.set('just_submitted_collection', False)
                SessionState.set('last_submitted_collection', None)
                _clear_collection_form_state()
                SessionState.navigate_to_home()
                st.rerun()
        with col2:
            if st.button("➕ Create Another", key="create_another_collection_btn", width='stretch'):
                SessionState.set('just_submitted_collection', False)
                SessionState.set('last_submitted_collection', None)
                _clear_collection_form_state()
                st.rerun()
        return

    st.markdown("### Collection")
    st.info(
        "Compare a related set of compounds side by side. Upload a CSV and map "
        "its columns below (minimum 2 members). The chosen similarity threshold "
        "and activity types apply to every member."
    )

    name = st.text_input(
        "Collection Name",
        placeholder="e.g., Flavonoid Comparison",
        help="Name to identify this collection",
        key="collection_name",
    )
    author_name = st.text_input(
        "Author Name",
        placeholder="e.g., Dr. Jane Smith",
        help="Your name (required for attribution in reports)",
        key="collection_author_name",
    )

    # --- CSV upload + auto/manual column mapping (reuses the Batch widgets) ---
    # Collection-prefixed keys keep this uploader/mapping independent of the
    # Batch tab's, which renders in the same script run (st.tabs renders all
    # tab bodies every run) and would otherwise collide on widget/session keys.
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        key="collection_csv_uploader",
        help="CSV with any column names — you'll map them below.",
    )

    if not uploaded_file:
        _clear_collection_form_state()
        return

    if SessionState.file_changed(uploaded_file, key="collection_uploaded_file_hash"):
        import pandas as pd

        clear_collection_preflight_state()
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['collection_csv_preview'] = df
            _clear_collection_mapping_state()
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            _clear_collection_mapping_state()
            st.session_state.pop('collection_csv_preview', None)
            return

    df = st.session_state.get('collection_csv_preview')
    if df is None:
        return

    column_mapping = _render_column_mapping_ui(df, key_prefix="collection_")
    if column_mapping is None:
        # User has not selected the required columns yet.
        return

    df_mapped = _apply_column_mapping(df, column_mapping)

    # Drop rows with no usable compound name (blank / NaN — e.g. a trailing empty
    # CSV line reads as `nan`). A nameless row can never be a member, so excluding
    # it here keeps the preview and the "N members" count honest (the builder also
    # skips these at submit; this just stops them being shown/counted).
    _names = df_mapped['compound_name'].astype(str).str.strip().str.lower()
    df_mapped = df_mapped[~_names.isin(['', 'nan', 'none'])].reset_index(drop=True)

    st.session_state['collection_csv_mapped'] = df_mapped

    if df_mapped.empty:
        st.warning("No rows with a compound name were found in the uploaded CSV.")
        return

    st.write("Preview (mapped):")
    preview_height = min(max(220, 36 * min(len(df_mapped), 20) + 40), 760)
    st.dataframe(
        _build_compact_preview(df_mapped),
        width='stretch',
        hide_index=True,
        height=preview_height,
    )
    st.caption(f"{len(df_mapped)} members in file")

    st.subheader("Collection Configuration")
    similarity_threshold = st.slider(
        "Similarity Threshold (%)",
        min_value=40,
        max_value=100,
        value=config.DEFAULT_SIMILARITY_THRESHOLD,
        help="Shared similarity threshold applied to every member",
        key="collection_similarity",
    )
    selected_activities = render_activity_checkboxes(key_prefix="collection")

    # Phase 1 trigger (only when no pre-flight is pending)
    if "collection_preflight" not in st.session_state:
        if st.button("🔍 Check collection", type="primary", width='stretch',
                     key="check_collection_btn"):
            _run_collection_preflight(
                name=name,
                author_name=author_name,
                df_mapped=df_mapped,
                similarity_threshold=similarity_threshold,
                activity_types=selected_activities,
            )
        return

    # Phase 2: a pre-flight exists -> render sections + Confirm/Cancel (Task 3)
    _render_collection_preflight()


def _submit_collection(
    name: str,
    author_name: str,
    df_mapped,
    similarity_threshold: int,
    activity_types: list[str],
) -> None:
    """Validate, build members from the mapped CSV, pre-check, dedupe (D-08),
    and submit ONE collection job.

    Each member is stamped with the shared ``similarity_threshold`` /
    ``activity_types`` (see :func:`_build_members_from_mapped`) so the backend's
    per-member config read applies the user's choice instead of silently
    defaulting (an unstamped member would also crash on ``int(None)``).
    """
    # --- Name / author validation (client-side UX mirror of D-03) ---
    name_error = _validate_collection_name(name, "Collection name")
    if name_error:
        st.error(name_error)
        return

    author_error = _validate_collection_name(author_name, "Author name")
    if author_error:
        st.error(author_error)
        return

    if not activity_types:
        st.error("Please select at least one activity type.")
        return

    # --- Resolve InChIKey-only rows to SMILES via PubChem (CollectionMember
    # requires a SMILES). Only rows lacking BOTH a SMILES and an InChI need it. ---
    inchikey_smiles_map: dict[str, str] = {}
    if "inchikey" in df_mapped.columns:
        has_smiles_col = "smiles" in df_mapped.columns
        has_inchi_col = "inchi" in df_mapped.columns
        keys_to_resolve: list[str] = []
        for row in df_mapped.to_dict("records"):
            smiles_val = str(row.get("smiles", "")).strip().lower() if has_smiles_col else ""
            inchi_val = str(row.get("inchi", "")).strip().lower() if has_inchi_col else ""
            if smiles_val in ("", "nan", "none") and inchi_val in ("", "nan", "none"):
                key_val = str(row.get("inchikey", "")).strip().upper()
                if key_val and key_val.lower() not in ("nan", "none"):
                    keys_to_resolve.append(key_val)
        if keys_to_resolve:
            unique_keys = list(set(keys_to_resolve))
            with st.spinner(f"Resolving {len(unique_keys)} InChIKey(s) via PubChem..."):
                try:
                    client = get_api_client()
                    inchikey_smiles_map = client.resolve_inchikeys_batch(unique_keys)
                except Exception as e:
                    logger.warning(f"Collection InChIKey resolution failed: {e}")

    # --- Build stamped members from the mapped CSV ---
    members, build_report = _build_members_from_mapped(
        df_mapped,
        similarity_threshold=similarity_threshold,
        activity_types=activity_types,
        inchikey_smiles_map=inchikey_smiles_map,
    )

    # --- Invalid member names BLOCK the submit (D-03): one bad name 422s the
    # whole collection, so surface the offending rows and let the user fix the
    # CSV rather than silently dropping members or hitting an opaque backend error.
    invalid_names = build_report.get("invalid_names", [])
    if invalid_names:
        st.error(
            f"{len(invalid_names)} member name(s) contain invalid characters "
            "(letters, numbers, spaces, and - _ ( ) [ ] ' , . only). "
            "Fix these names in your CSV and re-upload: "
            f"{', '.join(invalid_names[:5])}"
            + ("…" if len(invalid_names) > 5 else "")
        )
        return

    skipped_no_structure = build_report.get("skipped_no_structure", [])
    if skipped_no_structure:
        st.warning(
            f"{len(skipped_no_structure)} row(s) had no usable structure "
            "(SMILES/InChI/InChIKey) and were left out: "
            f"{', '.join(skipped_no_structure[:5])}"
            + ("…" if len(skipped_no_structure) > 5 else "")
        )

    # A present-but-unparseable SMILES would 422 the whole availability batch;
    # _build_members_from_mapped already excluded these — surface them so the
    # user can fix the offending cells rather than wonder why members vanished.
    invalid_structure = build_report.get("invalid_structure", [])
    if invalid_structure:
        st.warning(
            f"{len(invalid_structure)} row(s) had an unreadable SMILES and were "
            "left out (check for typos or non-structure text in those cells): "
            f"{', '.join(invalid_structure[:5])}"
            + ("…" if len(invalid_structure) > 5 else "")
        )
    if not members:
        st.error(
            "No members with a usable structure were found in the uploaded CSV."
        )
        return

    # --- Intra-set InChIKey dedupe (D-08) ---
    members, removed, invalid_smiles = _dedupe_members_by_structure(members)
    if removed > 0:
        # Inline warning covers the submit-FAILURE path (no rerun); the success
        # path re-emits it from session state after st.rerun() (see the
        # post-submission branch in render_collection_analysis).
        st.warning(
            f"{removed} duplicate compound(s) removed from this collection (identical structure)."
        )
    if invalid_smiles:
        st.warning(
            f"{len(invalid_smiles)} member(s) have a SMILES that could not be "
            f"parsed and may fail during analysis: {', '.join(invalid_smiles[:5])}"
            + ("…" if len(invalid_smiles) > 5 else "")
        )

    # --- D-06 member-count bounds (after dedupe) ---
    count = len(members)
    if count < COLLECTION_MIN_MEMBERS:
        st.error(
            f"A collection needs at least {COLLECTION_MIN_MEMBERS} members to "
            f"compare (got {count})."
        )
        return
    if count > COLLECTION_MAX_MEMBERS:
        st.error(
            f"A collection may have at most {COLLECTION_MAX_MEMBERS} members "
            f"(got {count})."
        )
        return
    if count > COLLECTION_SOFT_CAP:
        st.warning(
            f"{count} compounds selected. Charts get hard to read past ~8 — "
            "series opacity is reduced. Narrow the selection for clearer "
            "comparison."
        )

    # --- Shared availability pre-check (non-blocking; warns, never opens a dialog) ---
    members_for_check = [
        {"compound_name": m["name"], "smiles": m["smiles"]} for m in members
    ]
    with st.spinner(f"Checking ChEMBL data availability for {count} members..."):
        try:
            client = get_api_client()
            avail = client.check_availability_batch(
                compounds=members_for_check,
                similarity_threshold=similarity_threshold,
                activity_types=activity_types,
            )
        except Exception as e:
            logger.warning(f"Collection availability pre-check failed, proceeding: {e}")
            avail = {"success": False, "error": str(e)}

    if avail.get("success"):
        unavailable = [
            r.get("compound_name", "")
            for r in (avail.get("results") or [])
            if isinstance(r, dict) and r.get("available") is False
        ]
        if unavailable:
            st.warning(
                f"{len(unavailable)} member(s) may have no ChEMBL data at this "
                "threshold and could be reported as failed: "
                f"{', '.join([u for u in unavailable[:5] if u])}"
                + ("…" if len(unavailable) > 5 else "")
            )
    elif avail.get("error"):
        st.warning(
            f"Could not verify data availability: {avail['error']}. Proceeding anyway."
        )

    # --- Submit ONE collection job (POST /collections) ---
    # `members` already carry the stamped per-member similarity_threshold /
    # activity_types — send them as-is so the backend applies the shared config
    # instead of defaulting/crashing on a missing per-member threshold.
    safe_name = sanitize_compound_name(name.strip())

    with st.spinner("Creating collection..."):
        try:
            client = get_api_client()
            result = client.create_collection(
                name=name.strip(),
                members=members,
                author_name=author_name.strip(),
                similarity_threshold=similarity_threshold,
                activity_types=activity_types,
            )
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            st.error(f"Error: {e}")
            return

    if result.get("success"):
        # CollectionResponse carries `id` + `job_id` (NOT `collection_id`) — see
        # 23-06-SUMMARY: the interfaces note's `collection_id` is stale.
        collection_id = result.get("id")
        job_id = result.get("job_id")
        SessionState.set('just_submitted_collection', True)
        SessionState.set(
            'last_submitted_collection',
            {
                "id": collection_id,
                "job_id": job_id,
                "name": safe_name,
                "deduped": removed,
                "skipped_no_structure": len(skipped_no_structure),
            },
        )
        st.rerun()
    else:
        st.error(f"Failed to create collection: {result.get('error', 'Unknown error')}")


def _clear_collection_mapping_state():
    """Clear the Collection tab's CSV column-mapping selections + mapped frame.

    Mirrors the Batch ``_clear_column_mapping_state`` but for the
    ``collection_``-prefixed keys (the two tabs render in the same script run,
    so their CSV state must be kept independent).
    """
    keys_to_clear = [
        'collection_csv_col_name_select',
        'collection_csv_col_smiles_select',
        'collection_csv_col_inchi_select',
        'collection_csv_col_inchikey_select',
        'collection_csv_mapped',
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _clear_collection_form_state():
    """Clear all Collection tab CSV/upload session state (mapping + file hash)."""
    _clear_collection_mapping_state()
    for key in ('collection_csv_preview', 'collection_uploaded_file_hash'):
        st.session_state.pop(key, None)


def _clear_batch_form_state():
    """Clear batch form related session state."""
    keys_to_clear = [
        'uploaded_file_hash',
        'csv_preview',
        'csv_mapped',
        'csv_col_name_select',
        'csv_col_smiles_select',
        'csv_col_inchi_select',
        'batch_duplicate_check_done',
        'batch_duplicate_check_signature',
        'batch_user_confirmed',
        'batch_existing',
        'batch_processing',
        'batch_new',
        'batch_structure_matches',
        'batch_duplicate_decisions',
        'batch_duplicate_new_names',
        'batch_default_duplicate_action',
        'batch_suggested_versions',
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
