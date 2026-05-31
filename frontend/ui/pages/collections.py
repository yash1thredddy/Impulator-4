"""Collections page — global list + axis-flip comparison detail (Phase 23).

The presentation payoff for the Collections MVP. Two views, switched on
session state:

* **List** — every collection (GLOBAL, D-05) rendered as
  ``st.container(border=True)`` cards in a ``st.columns(3)`` grid (mirrors
  ``home.py``). Each card shows the name, author, member count, integer
  average IMP, a job-derived status badge, and an orange
  "completed (N failed)" partial-success badge (D-09). Empty state shows the
  UI-SPEC copy.

* **Detail** — the locked "axis flip" (UI-SPEC §1): a 5× ``st.metric`` header
  then ONE ``st.tabs`` strip driven by a single member ``st.multiselect``
  above it. Members are series/facets inside one chart, never a
  tab→subtab tree × N. Comparison charts overlay members via
  ``color_col="compound_name"``; the headline bioactivity heatmap is consumed
  from :func:`frontend.ui.components.charts.create_bioactivity_heatmap`
  (built in plan 23-07 — this page does NOT define it, D-01/D-10). The
  Overview rank table defaults to IMP-descending with 🔬 candidate emphasis
  and NO verbal IMP labels (Phase 21). The efficiency plane draws the
  golden-triangle LE/LLE diagonal and flags Lipinski/Veber violations
  (D-13). While processing, an ``@st.fragment(run_every="2s")`` polls the
  linked job and shows the aggregate "{completed}/{total} members analyzed".
  Members drill into the UNMODIFIED ``compound_detail`` renderer via the
  ``internal_prefix`` seam (D-12 / COLL-14). Full-ZIP download via
  ``st.link_button``; delete is confirm-gated (D-11).

SECURITY (HC-3, T-23-09-I1): member names and target labels are untrusted.
They are rendered ONLY through Plotly / native Streamlit widgets (escaped by
default) — this page NEVER enables raw-HTML rendering for any member-derived
string. The heatmap top-K (D-10) always shows the "others (N more)" note so a
candidate is never silently dropped (T-23-09-I2).
"""

import io
import logging
import re
import zipfile
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.services import get_api_client
from frontend.ui.components.charts import (
    HEATMAP_TOP_K,
    create_bioactivity_heatmap,
    create_box_plot,
    create_scatter_plot,
)
from frontend.utils import SessionState, sanitize_compound_name

logger = logging.getLogger(__name__)

# Session-state keys (single ownership for the member multiselect, no default=).
_SELECTED_COLLECTION_KEY = "selected_collection_id"
_MEMBER_SELECT_KEY = "collection_member_select"
_CONFIRM_DELETE_KEY = "collection_confirm_delete"

# Past this many overlaid series the charts become hard to read; we keep the
# overlay (no faceting — out of scope this phase) but reduce opacity (UI-SPEC §2).
_OVERLAY_LEGIBILITY_LIMIT = 8
_OVERLAY_REDUCED_OPACITY = 0.45

# Terminal job statuses (when reached, polling stops + the cache is cleared).
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "error"}

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


# ---------------------------------------------------------------------------
# Cached data loaders (TTL ~20s while processing; .clear()ed on terminal status)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=20, show_spinner=False)
def _list_collections_cached() -> dict[str, Any]:
    """Fetch the GLOBAL collection list (D-05). Cached ~20s."""
    return get_api_client().list_collections()


@st.cache_data(ttl=20, show_spinner=False)
def _get_collection_cached(collection_id: str) -> dict[str, Any]:
    """Fetch a single collection's detail. Cached ~20s, keyed by id."""
    return get_api_client().get_collection(collection_id)


def _get_collection_uncached(collection_id: str) -> dict[str, Any]:
    """Fetch a single collection's detail FRESH (no cache) — for the 2s poll.

    The collection payload now carries the linked job's ``status`` / ``progress``
    / ``message`` (folded in server-side), so the processing fragment polls the
    GLOBAL collection endpoint instead of the session-scoped ``/jobs/{id}`` —
    which 403s for any session that did not submit the collection, leaving a
    finished collection stuck "processing" and polling forever.
    """
    return get_api_client().get_collection(collection_id)


_PROGRESS_MSG_RE = re.compile(r"(?:Processed|Processing)\s+(\d+)\s*/\s*(\d+)\s+members", re.I)


@st.cache_data(ttl=20, show_spinner=False)
def _load_collection_tables(storage_path: str) -> dict[str, Any]:
    """Load the root tables from the collection ZIP (internal_prefix="").

    ``collection_summary.csv`` (one row per member; the rank-table source) and
    ``combined_activities.csv`` (the overlay + heatmap source) live at the ZIP
    ROOT and are addressed via the collection's own ``storage_path`` with
    ``internal_prefix=""`` — distinct from the per-member drill-in prefix.
    """
    # The collection ZIP is fetched via the same smart download funnel the
    # single-compound page uses; we read the two root CSVs directly.
    from frontend.services.azure_storage import smart_download_result

    out: dict[str, Any] = {"summary": None, "combined": None, "truncated": False}
    try:
        zip_bytes = smart_download_result(storage_path=storage_path)
    except Exception as e:  # pragma: no cover - network/IO
        logger.warning(f"Could not download collection ZIP {storage_path}: {e}")
        return out
    if not zip_bytes:
        return out

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = set(zf.namelist())
            if "collection_summary.csv" in names:
                with zf.open("collection_summary.csv") as f:
                    out["summary"] = pd.read_csv(f)
            if "combined_activities.csv" in names:
                with zf.open("combined_activities.csv") as f:
                    out["combined"] = pd.read_csv(f)
            out["truncated"] = "WARNING.txt" in names
    except Exception as e:  # pragma: no cover - parse error
        logger.warning(f"Could not parse collection tables {storage_path}: {e}")
    return out


# ---------------------------------------------------------------------------
# Small helpers (pure / presentation)
# ---------------------------------------------------------------------------

def _imp_int(raw: Optional[float]) -> Optional[int]:
    """Convert a raw IMP score (0..1 float) to an integer 0-100 (Phase 21).

    Returns ``None`` for missing values so the caller renders "N/A".
    """
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    # collection_summary.csv stores the raw 0..1 score; the API avg is the same
    # scale. Multiply only when clearly fractional to stay robust to either.
    return int(round(val * 100)) if 0.0 <= val <= 1.0 else int(round(val))


def _job_status_badge(status: Optional[str]) -> None:
    """Render a job-derived status badge (icon + text, never color-alone)."""
    s = (status or "").lower()
    if s == "completed":
        st.success("✓ completed")
    elif s in ("failed", "error"):
        st.error("🔴 failed")
    elif s == "cancelled":
        st.warning("⚠ cancelled")
    else:
        st.info("⏳ processing")


def _safe_member_folders(member_names: list[str]) -> list[str]:
    """Reproduce the backend ``_safe_member_folder`` mapping (D-12 drill-in).

    The collection ZIP nests each member under ``compounds/{safe_name}/``; the
    folder is the backend's ``sanitize_compound_name`` output with a ``_{n}``
    de-dup suffix applied IN MEMBER ORDER (mirrors
    ``backend.services.collection_service._safe_member_folder``). The page
    replays that exact algorithm so drill-in addresses the right ZIP section.

    Returns a list POSITION-ALIGNED with ``member_names`` (NOT a name-keyed dict)
    so two members sharing a display name still get their distinct folders —
    matching the backend's per-position de-dup (Rule 1: a name-keyed dict would
    collapse same-named members onto one folder).
    """
    used: set[str] = set()
    folders: list[str] = []
    for name in member_names:
        safe = sanitize_compound_name(name or "member")
        safe = safe.replace("..", "_").replace("/", "_").replace("\\", "_").strip()
        if not safe or safe in (".", ".."):
            safe = "member"
        candidate = safe
        n = 1
        while candidate in used:
            n += 1
            candidate = f"{safe}_{n}"
        used.add(candidate)
        folders.append(candidate)
    return folders


def _activity_target_col(df: pd.DataFrame) -> Optional[str]:
    """Pick the target label column for the heatmap / overlays."""
    for col in ("Target_Name", "Target_ChEMBL_ID", "Target"):
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_collections_page() -> None:
    """Route between the collections list and a single collection's detail."""
    selected = SessionState.get(_SELECTED_COLLECTION_KEY)
    if selected:
        _render_collection_detail(selected)
    else:
        _render_collections_list()


# ---------------------------------------------------------------------------
# List view (Task 1)
# ---------------------------------------------------------------------------

def _render_collections_list() -> None:
    """Render the GLOBAL collections list as cards (D-05 / D-09)."""
    st.markdown("## :violet[IMP] Collections")
    st.caption("Compare a related set of compounds side by side.")

    result = _list_collections_cached()
    if not result.get("success"):
        st.error(
            f"Could not load collections: {result.get('error', 'unknown error')}"
        )
        return

    items = result.get("items", []) or []

    if not items:
        # Empty state — UI-SPEC copy verbatim.
        st.info("No collections yet")
        st.write(
            "Create a collection from the Analyze page to compare a related set "
            "of compounds side by side."
        )
        if st.button("Go to Analyze", width="stretch"):
            SessionState.navigate_to_analyze()
            st.rerun()
        return

    # 3-card grid (mirror home.py density).
    cols = st.columns(3)
    for i, item in enumerate(items):
        with cols[i % 3]:
            _render_collection_card(item)


def _render_collection_card(item: dict[str, Any]) -> None:
    """Render one collection card (member-derived strings as plain text, HC-3)."""
    collection_id = str(item.get("id", ""))
    name = item.get("name", "Untitled")
    author = item.get("author_name", "")
    compound_count = int(item.get("compound_count", 0) or 0)
    failed = int(item.get("member_failed_count", 0) or 0)
    avg_imp = _imp_int(item.get("avg_imp_score"))

    with st.container(border=True):
        # Name + author as plain text via native widgets (no unsafe_allow_html).
        st.subheader(name)
        if author:
            st.caption(f"by {author}")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Members", compound_count)
        with m2:
            st.metric("Avg IMP", avg_imp if avg_imp is not None else "N/A")

        # Job-derived status, folded into the (global) collection summary
        # server-side — no session-scoped /jobs/{id} call (which would 403 for
        # non-owner sessions and blank the badge).
        status = item.get("status")
        _job_status_badge(status)

        # Partial-success badge (D-09): ⚠ + text, never color-alone.
        if failed > 0 and (status or "").lower() == "completed":
            st.warning(f"⚠ completed ({failed} failed)")

        if st.button("Open", key=f"open_coll_{collection_id}", width="stretch"):
            SessionState.set(_SELECTED_COLLECTION_KEY, collection_id)
            # Reset the per-collection member selection so it re-seeds to all.
            st.session_state.pop(_MEMBER_SELECT_KEY, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Detail view (Task 2)
# ---------------------------------------------------------------------------

def _render_failed_members(detail: dict[str, Any]) -> None:
    """Surface per-member failures + lower-tier cascade hints (D-PF-6).

    ``detail`` is the CollectionDetailResponse dict from the API, whose
    ``failed_members`` field carries the linked job's per-member failure rows
    ([{name, error, cascade_results}]). For each failed member we show the
    error and, when the diagnostic probe found data at lower thresholds, a
    "data available at 50% (1), 40% (29)" hint — diagnostic only, no auto-retry.
    """
    failed_members = detail.get("failed_members") or []
    if not failed_members:
        return
    with st.expander(f"Failed members ({len(failed_members)})", expanded=True):
        for fm in failed_members:
            line = f"**{fm['name']}** — {fm['error']}"
            cascade = fm.get("cascade_results") or []
            hits = [f"{t['threshold']}% ({t['count']})" for t in cascade if t.get("count")]
            if hits:
                line += " · data available at " + ", ".join(hits)
            st.markdown(line)


def _render_collection_detail(collection_id: str) -> None:
    """Render a single collection: header, axis-flip tabs, polling, delete."""
    if st.button("← Back to collections", width="stretch"):
        SessionState.set(_SELECTED_COLLECTION_KEY, None)
        st.session_state.pop(_MEMBER_SELECT_KEY, None)
        st.session_state.pop(_CONFIRM_DELETE_KEY, None)
        st.rerun()

    detail = _get_collection_cached(collection_id)
    if not detail.get("success"):
        st.error(
            f"Could not load collection: {detail.get('error', 'not found')}"
        )
        return

    name = detail.get("name", "Untitled")
    storage_path = detail.get("storage_path")

    st.markdown(f"## {name}")
    if detail.get("author_name"):
        st.caption(f"by {detail['author_name']}")

    # The true member total is the members_config length (D-02) — available from
    # the first render, unlike compound_count which is only written at finalize.
    members_config_early = detail.get("members_config") or {}
    member_total = (
        len(members_config_early.get("members", []))
        if isinstance(members_config_early, dict)
        else 0
    )

    # ---- Job status (folded into the collection payload server-side) ----
    status = (detail.get("status") or "").lower()

    if status and status not in _TERMINAL_STATUSES:
        # Still processing — poll the (global) collection every 2s in a fragment.
        _render_processing_fragment(collection_id, member_total)
        return

    # ---- Fully-failed collection -> error copy, no tabs (UI-SPEC states) ----
    compound_count = int(detail.get("compound_count", 0) or 0)
    failed = int(detail.get("member_failed_count", 0) or 0)
    if status in ("failed", "error") or (compound_count == 0 and status == "completed"):
        job_err = (detail.get("message") or "").strip()
        st.error(
            "Collection failed — no members could be analyzed. "
            f"{job_err} "
            "Re-create the collection or check the input compounds."
        )
        _render_failed_members(detail)
        _render_delete_control(collection_id, name)
        return

    # ---- Header metrics (5×) ----
    avg_imp = _imp_int(detail.get("avg_imp_score"))
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Members", compound_count)
    h2.metric("Avg IMP", avg_imp if avg_imp is not None else "N/A")
    h3.metric("IMP candidates", int(detail.get("imp_candidate_count", 0) or 0))
    h4.metric("Unique targets", int(detail.get("unique_targets", 0) or 0))
    h5.metric("Failed", failed)

    if failed > 0:
        st.warning(f"⚠ completed ({failed} failed)")
        _render_failed_members(detail)

    # ---- Load root tables (internal_prefix="" — root of the collection ZIP) ----
    tables = _load_collection_tables(storage_path) if storage_path else {}
    summary_df = tables.get("summary")
    combined_df = tables.get("combined")
    truncated = bool(tables.get("truncated"))

    # ---- Member set (from members_config, D-02) + the single multiselect ----
    members_config = detail.get("members_config") or {}
    member_defs = (
        members_config.get("members", []) if isinstance(members_config, dict) else []
    )
    all_member_names = [
        m.get("name") for m in member_defs if isinstance(m, dict) and m.get("name")
    ]
    # Fall back to the summary table's members if members_config is sparse.
    if (
        not all_member_names
        and summary_df is not None
        and "compound_name" in summary_df.columns
    ):
        all_member_names = [str(x) for x in summary_df["compound_name"].tolist()]

    selected_members = _render_member_control(all_member_names)

    st.divider()

    tabs = st.tabs(
        [
            "📊 Overview",
            "📈 Visualizations",
            "🔬 Evidence",
            "🧬 Molecules",
            "📋 Data",
            "👥 Members",
        ]
    )

    with tabs[0]:
        _render_overview_tab(summary_df, selected_members)
    with tabs[1]:
        _render_visualizations_tab(combined_df, selected_members)
    with tabs[2]:
        _render_evidence_tab(combined_df, selected_members)
    with tabs[3]:
        _render_molecules_tab(member_defs, selected_members)
    with tabs[4]:
        _render_data_tab(combined_df, selected_members, truncated, collection_id)
    with tabs[5]:
        _render_members_tab(all_member_names, storage_path)

    st.divider()
    _render_delete_control(collection_id, name)


@st.fragment(run_every="2s")
def _render_processing_fragment(
    collection_id: str, member_total: int
) -> None:
    """Poll the (global) collection every 2s and show aggregate progress (UI-SPEC §4).

    The backend folds the linked job's ``status`` / ``progress`` / ``message``
    onto the collection payload, and reports aggregate progress (D-09) as the job
    ``current_step`` string ``"Processed {done}/{total} members"`` plus a
    percentage — it does NOT expose completed/total as structured keys. We parse
    the step string (authoritative for ``done``), falling back to
    ``round(progress% * member_total)`` when absent. ``total`` is the
    members_config length (known from the first render, unlike ``compound_count``
    which is only written at finalize). Polling the collection (not the
    session-scoped ``/jobs/{id}``) avoids the 403 that left a finished collection
    stuck "processing" and polling forever for non-owner sessions.

    On a terminal status the per-id caches are cleared and the whole page
    reruns so the detail view renders the tabs.
    """
    detail = _get_collection_uncached(collection_id)
    status = (detail.get("status") or "").lower() if detail.get("success") else ""

    if status in _TERMINAL_STATUSES:
        # Clear the caches so the next render reads fresh terminal data.
        _get_collection_cached.clear()
        _list_collections_cached.clear()
        st.rerun()
        return

    total = int(member_total or 0)
    completed = 0
    message = detail.get("message") or ""
    match = _PROGRESS_MSG_RE.search(str(message))
    if match:
        completed = int(match.group(1))
        # Prefer the step's own total if members_config was unavailable.
        total = total or int(match.group(2))
    elif total:
        # Fall back to deriving done from the progress percentage.
        pct = float(detail.get("progress", 0.0) or 0.0)
        completed = int(round(pct / 100.0 * total))

    with st.spinner("Processing collection…"):
        if total:
            st.info(
                f"Processing collection… {completed}/{total} members analyzed"
            )
        else:
            st.info("Processing collection… members analyzed")


def _render_member_control(all_member_names: list[str]) -> list[str]:
    """The single member multiselect (axis flip) + Select-all/Clear callbacks.

    Keyed on session_state with NO ``default=`` (single ownership), seeded to
    all members on first load (UI-SPEC §2).
    """
    if not all_member_names:
        return []

    # Seed once: all members selected on first visit to this collection.
    if _MEMBER_SELECT_KEY not in st.session_state:
        st.session_state[_MEMBER_SELECT_KEY] = list(all_member_names)

    def _select_all() -> None:
        st.session_state[_MEMBER_SELECT_KEY] = list(all_member_names)

    def _clear() -> None:
        st.session_state[_MEMBER_SELECT_KEY] = []

    c1, c2, c3 = st.columns([6, 1, 1])
    with c1:
        st.multiselect(
            "Members to compare",
            options=all_member_names,
            key=_MEMBER_SELECT_KEY,
        )
    with c2:
        st.button("Select all", on_click=_select_all, width="stretch")
    with c3:
        st.button("Clear", on_click=_clear, width="stretch")

    selected = st.session_state.get(_MEMBER_SELECT_KEY, [])

    # >8 series: keep the overlay, reduce opacity, warn (NO faceting, UI-SPEC §2).
    if len(selected) > _OVERLAY_LEGIBILITY_LIMIT:
        st.warning(
            f"{len(selected)} compounds selected. Charts get hard to read past "
            "~8 — series opacity is reduced. Narrow the selection for clearer "
            "comparison."
        )
    return selected


def _filter_combined(
    combined_df: Optional[pd.DataFrame], selected_members: list[str]
) -> Optional[pd.DataFrame]:
    """Restrict the combined activities table to the selected members."""
    if (
        combined_df is None
        or combined_df.empty
        or "compound_name" not in combined_df.columns
    ):
        return combined_df
    if not selected_members:
        return combined_df.iloc[0:0]
    return combined_df[combined_df["compound_name"].isin(selected_members)]


def _overlay_opacity(selected_members: list[str]) -> float:
    """Reduced opacity past the legibility limit (UI-SPEC §2)."""
    return (
        _OVERLAY_REDUCED_OPACITY
        if len(selected_members) > _OVERLAY_LEGIBILITY_LIMIT
        else 1.0
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def _render_overview_tab(
    summary_df: Optional[pd.DataFrame], selected_members: list[str]
) -> None:
    """Rank table: default sort IMP DESC, 🔬 candidate emphasis, NO verbal labels."""
    if summary_df is None or summary_df.empty:
        st.info("No data for the selected members.")
        return

    df = summary_df.copy()
    if "compound_name" in df.columns and selected_members:
        df = df[df["compound_name"].isin(selected_members)]
    if df.empty:
        st.info("No data for the selected members.")
        return

    # IMP score as integer 0-100 (Phase 21).
    if "imp_score" in df.columns:
        df["IMP"] = df["imp_score"].apply(_imp_int)
    candidates_col = "imp_candidates" if "imp_candidates" in df.columns else None

    # 🔬 candidate marker on the candidate count (no verbal labels).
    if candidates_col:
        df["Candidates"] = df[candidates_col].apply(
            lambda v: f"🔬 {int(v)}"
            if pd.notna(v) and int(v) > 0
            else str(int(v) if pd.notna(v) else 0)
        )

    display_cols = []
    rename = {}
    if "compound_name" in df.columns:
        display_cols.append("compound_name")
        rename["compound_name"] = "Member"
    if "IMP" in df.columns:
        display_cols.append("IMP")
    if "total_compounds" in df.columns:
        display_cols.append("total_compounds")
        rename["total_compounds"] = "Similar compounds"
    if "total_activities" in df.columns:
        display_cols.append("total_activities")
        rename["total_activities"] = "Activities"
    if "Candidates" in df.columns:
        display_cols.append("Candidates")

    out = df[display_cols].rename(columns=rename)

    # Default sort IMP descending (D-11).
    if "IMP" in out.columns:
        out = out.sort_values(by="IMP", ascending=False, na_position="last")

    def _flag_candidate_rows(row: pd.Series):
        # Orange (#ffa94d family) tint for candidate rows, dark text (HC-3).
        is_candidate = isinstance(row.get("Candidates"), str) and row[
            "Candidates"
        ].startswith("🔬")
        return [
            "background-color: #fff4e6; color: #1a1a1a" if is_candidate else ""
            for _ in row
        ]

    styled = out.style.apply(_flag_candidate_rows, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)
    st.caption("Sorted by IMP score (descending). 🔬 marks IMP candidates.")


def _render_visualizations_tab(
    combined_df: Optional[pd.DataFrame], selected_members: list[str]
) -> None:
    """Overlaid comparison charts + the bioactivity heatmap + golden triangle."""
    df = _filter_combined(combined_df, selected_members)
    if df is None or df.empty:
        st.info("No activity data for the selected members.")
        return

    opacity = _overlay_opacity(selected_members)

    # ---- Bioactivity heatmap (D-01 / D-10) — built from combined_activities ----
    st.subheader("📊 Bioactivity heatmap")
    target_col = _activity_target_col(df)
    value_col = "pActivity" if "pActivity" in df.columns else None
    if target_col and value_col:
        # Build the FULL members × targets matrix; create_bioactivity_heatmap
        # trims to top-K internally. The "others" note is THIS page's job
        # (T-23-09-I2): count the full target set so no candidate is silently
        # dropped.
        pivot = df.pivot_table(
            index="compound_name",
            columns=target_col,
            values=value_col,
            aggfunc="max",
        )
        total_targets = int(pivot.shape[1])
        fig = create_bioactivity_heatmap(pivot)
        if len(fig.data) == 0:
            st.info("No shared targets across the selected members.")
        else:
            st.plotly_chart(fig, width="stretch")
            shown_k = min(HEATMAP_TOP_K, total_targets)
            others = max(0, total_targets - HEATMAP_TOP_K)
            st.caption(
                f"Showing top {shown_k} targets by member hit-count. "
                f"+{others} more targets in the full data."
            )
    else:
        st.info("No shared targets across the selected members.")

    st.divider()

    # ---- Overlaid activity comparison (members as series, color_col) ----
    st.subheader("📈 Activity comparison")
    if "pActivity" in df.columns:
        fig_box = create_box_plot(
            df,
            y_col="pActivity",
            color_col="compound_name",
            title="pActivity distribution by member",
        )
        fig_box.update_traces(opacity=opacity)
        st.plotly_chart(fig_box, width="stretch")
    else:
        st.info("No activity data for the selected members.")

    st.divider()

    # ---- Golden-triangle efficiency plane (D-13) ----
    _render_golden_triangle(df, opacity)


def _render_golden_triangle(df: pd.DataFrame, opacity: float) -> None:
    """LE/LLE golden-triangle scatter + Lipinski/Veber flags (D-13)."""
    st.subheader("🔺 Efficiency plane (golden triangle)")
    # Use BEI vs SEI (ligand-efficiency family) when present; else MW vs LogP.
    if "BEI" in df.columns and "SEI" in df.columns:
        plot_df = df.dropna(subset=["BEI", "SEI"])
        if plot_df.empty:
            st.info("No efficiency data for the selected members.")
            return
        fig = create_scatter_plot(
            plot_df,
            x_col="SEI",
            y_col="BEI",
            color_col="compound_name",
            title="Ligand efficiency (BEI vs SEI)",
        )
        fig.update_traces(opacity=opacity)
        # Golden-triangle LE/LLE diagonal guide.
        lo = float(min(plot_df["SEI"].min(), plot_df["BEI"].min()))
        hi = float(max(plot_df["SEI"].max(), plot_df["BEI"].max()))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="#ffa94d", dash="dash"),
                name="LE/LLE diagonal",
            )
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No efficiency data for the selected members.")

    # Lipinski / Veber violation flags per member (D-13).
    _render_property_flags(df)


def _render_property_flags(df: pd.DataFrame) -> None:
    """Per-member Lipinski / Veber violation flags in a properties table."""
    if "compound_name" not in df.columns:
        return
    rows = []
    for member, g in df.groupby("compound_name"):
        mw = (
            g["Molecular_Weight"].dropna().max()
            if "Molecular_Weight" in g.columns
            else None
        )
        logp = g["LogP"].dropna().max() if "LogP" in g.columns else None
        hbd = g["HBD"].dropna().max() if "HBD" in g.columns else None
        hba = g["HBA"].dropna().max() if "HBA" in g.columns else None
        tpsa = g["TPSA"].dropna().max() if "TPSA" in g.columns else None

        lipinski_violations = 0
        if mw is not None and mw > 500:
            lipinski_violations += 1
        if logp is not None and logp > 5:
            lipinski_violations += 1
        if hbd is not None and hbd > 5:
            lipinski_violations += 1
        if hba is not None and hba > 10:
            lipinski_violations += 1

        # Veber: TPSA <= 140 (rotatable-bond count not in this table; TPSA proxy).
        veber_ok = tpsa is None or tpsa <= 140

        rows.append(
            {
                "Member": member,
                "MW": round(mw, 1) if mw is not None else None,
                "LogP": round(logp, 2) if logp is not None else None,
                "Lipinski violations": (
                    f"⚠ {lipinski_violations}" if lipinski_violations > 0 else "✓ 0"
                ),
                "Veber": "✓ pass" if veber_ok else "⚠ fail",
            }
        )
    if rows:
        st.caption("Druglikeness flags (Lipinski Rule of 5 / Veber)")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_evidence_tab(
    combined_df: Optional[pd.DataFrame], selected_members: list[str]
) -> None:
    """Per-member overlaid evidence views (activity by target)."""
    df = _filter_combined(combined_df, selected_members)
    if df is None or df.empty:
        st.info("No activity data for the selected members.")
        return

    target_col = _activity_target_col(df)
    if target_col and "pActivity" in df.columns:
        fig = create_scatter_plot(
            df,
            x_col=target_col,
            y_col="pActivity",
            color_col="compound_name",
            title="Activity by target (members overlaid)",
        )
        fig.update_traces(opacity=_overlay_opacity(selected_members))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No activity data for the selected members.")


def _render_molecules_tab(
    member_defs: list[dict[str, Any]], selected_members: list[str]
) -> None:
    """2D structure grid, one cell per selected member (names as plain text)."""
    from frontend.ui.components import render_2d_structure

    selected_defs = [
        m
        for m in member_defs
        if isinstance(m, dict) and m.get("name") in selected_members
    ]
    if not selected_defs:
        st.info("No molecules for the selected members.")
        return

    cols = st.columns(4)
    for i, m in enumerate(selected_defs):
        with cols[i % 4]:
            smiles = m.get("smiles", "")
            if smiles:
                try:
                    render_2d_structure(smiles)
                except Exception as e:  # pragma: no cover - render fallback
                    logger.debug(f"2D render failed: {e}")
            # Member name as plain text (HC-3).
            st.caption(m.get("name", ""))


def _render_data_tab(
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
    truncated: bool,
    collection_id: str,
) -> None:
    """Combined data table (250k-cap warning) + ZIP download link."""
    df = _filter_combined(combined_df, selected_members)

    if truncated:
        n = len(combined_df) if combined_df is not None else 0
        st.warning(
            f"Showing first 250,000 of {n} rows. Full data is in the per-member "
            "sections of the collection ZIP."
        )

    if df is not None and not df.empty:
        st.dataframe(df, width="stretch", height=450, hide_index=True)
    else:
        st.info("No data for the selected members.")

    # Full-ZIP download via st.link_button (NOT a buffered download_button).
    url = get_api_client().collection_download_url(collection_id)
    st.link_button("Download collection ZIP", url, width="stretch")


def _render_members_tab(
    all_member_names: list[str], storage_path: Optional[str]
) -> None:
    """Drill into the UNMODIFIED compound_detail renderer via internal_prefix (D-12)."""
    if not all_member_names or not storage_path:
        st.info("No members to drill into.")
        return

    folders = _safe_member_folders(all_member_names)

    st.caption("Open a member to view its full single-compound analysis.")
    # Enumerate so widget keys are index-based — two members sharing a display
    # name would otherwise collide into a DuplicateWidgetID crash (Rule 1).
    for i, member in enumerate(all_member_names):
        c1, c2 = st.columns([6, 1])
        with c1:
            # Member name as plain text (HC-3).
            st.write(member)
        with c2:
            if st.button("Open", key=f"drill_{i}", width="stretch"):
                # Point the unmodified compound_detail renderer at this member's
                # section inside the collection ZIP (internal_prefix seam, D-12).
                safe = folders[i]
                SessionState.navigate_to_compound(
                    member,
                    storage_path=storage_path,
                )
                SessionState.set(
                    "selected_compound_internal_prefix",
                    f"compounds/{safe}/",
                )
                st.rerun()


def _render_delete_control(collection_id: str, name: str) -> None:
    """Confirm-gated delete (D-11): st.warning + explicit button."""
    if st.session_state.get(_CONFIRM_DELETE_KEY) == collection_id:
        # Name rendered as plain text inside st.warning (HC-3).
        st.warning(
            f"Delete collection '{name}'? This removes the collection and its "
            "ZIP. This cannot be undone."
        )
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            if st.button("Delete collection", type="primary", width="stretch"):
                result = get_api_client().delete_collection(collection_id)
                if result.get("success"):
                    st.toast("✓ Collection deleted", icon="✅")
                    _list_collections_cached.clear()
                    _get_collection_cached.clear()
                    SessionState.set(_SELECTED_COLLECTION_KEY, None)
                    st.session_state.pop(_CONFIRM_DELETE_KEY, None)
                    st.session_state.pop(_MEMBER_SELECT_KEY, None)
                    st.rerun()
                else:
                    st.error(
                        f"Could not delete: {result.get('error', 'unknown error')}"
                    )
        with c2:
            if st.button("Cancel", width="stretch"):
                st.session_state.pop(_CONFIRM_DELETE_KEY, None)
                st.rerun()
    else:
        if st.button("🗑️ Delete collection", width="stretch"):
            st.session_state[_CONFIRM_DELETE_KEY] = collection_id
            st.rerun()
