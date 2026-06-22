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

import html
import io
import logging
import re
import zipfile
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.services import get_api_client
from frontend.ui.components.charts import (
    HEATMAP_TOP_K,
    apply_impulator_theme,
    create_bioactivity_heatmap,
    create_box_plot,
    create_chemical_space_scatter,
    create_compare_radar,
    create_decision_map,
    create_efficiency_plane,
    create_gmm_density_overlay,
    create_imp_component_breakdown,
    create_imp_component_radar,
    create_imp_contribution_bar,
    create_pareto_plot,
    create_promise_contribution_bar,
    create_sar_matrix,
    create_scatter_plot,
)
from frontend.utils import SessionState, sanitize_compound_name

logger = logging.getLogger(__name__)

# Session-state keys (single ownership for the member multiselect, no default=).
_SELECTED_COLLECTION_KEY = "selected_collection_id"
_MEMBER_SELECT_KEY = "collection_member_select"
_CONFIRM_DELETE_KEY = "collection_confirm_delete"
# Mode toggles for the global control row (UI-SPEC Interaction §2). The active
# mode is exposed under the CANONICAL flag names `small_multiples` (mode B) and
# `compare` (mode C); downstream view plans (24-08/09/12) gate on those exact
# identifiers, so these session keys back the same contract.
_SMALL_MULTIPLES_KEY = "collection_small_multiples"
_COMPARE_KEY = "collection_compare"

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


@st.cache_data(ttl=20, show_spinner=False)
def _load_collection_aggregate(storage_path: str) -> dict[str, Any]:
    """Load + parse ``collection_aggregate.json`` from the collection ZIP (24-04).

    The Evidence & Annotations view is single-sourced from exactly this one
    artifact (D-S2-ARCH): a dict keyed by member ``entry_id`` ->
    ``{member_name, indications, all_similar, pdb, classification}``, written at
    finalize by ``collection_service._build_collection_aggregate``. The heavy ZIP
    download is cached here (mirrors :func:`_load_collection_tables`); the pure
    normalization stays in ``collection_aggregate.parse_aggregate`` (24-05).

    Resilience (T-24-10-01): a missing artifact, an unreadable ZIP, or a malformed
    payload all degrade to an empty dict (logged, never raised) so the Evidence
    view falls back to the D-12 per-member drill-in (UI-SPEC artifact-missing
    warning). Returns the entry_id-keyed dict of
    :class:`~frontend.ui.components.collection_aggregate.AggregateEntry`.
    """
    import json

    from frontend.services.azure_storage import smart_download_result
    from frontend.ui.components import collection_aggregate as agg_reader

    try:
        zip_bytes = smart_download_result(storage_path=storage_path)
    except Exception as e:  # pragma: no cover - network/IO
        logger.warning(f"Could not download collection ZIP {storage_path}: {e}")
        return {}
    if not zip_bytes:
        return {}

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            if "collection_aggregate.json" not in set(zf.namelist()):
                return {}
            with zf.open("collection_aggregate.json") as f:
                raw = json.load(f)
    except Exception as e:  # pragma: no cover - parse/IO error
        logger.warning(f"Could not parse collection aggregate {storage_path}: {e}")
        return {}

    return agg_reader.parse_aggregate(raw)


# ---------------------------------------------------------------------------
# Cached heavy compute (Chemical Space / IMP Analysis — UI-SPEC §6).
# Keyed on (collection_id, tuple(sorted(members))[, k/method]) so a re-select
# within an already-computed selection is instant. The big combined frame is
# passed underscore-prefixed (`_combined_df`) so st.cache_data EXCLUDES it from
# the hash — the (collection_id, members) tuple is the real cache key. `.clear()`
# is wired into the terminal-status branch of the processing fragment.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=20, show_spinner=False)
def _project_members(
    collection_id: str,
    members: tuple[str, ...],
    method: str,
    _combined_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Cached PCA/UMAP projection → one row per member coords_df, joined by name.

    Builds the per-member descriptor matrix from the selected subset, projects it
    to 2D (k-independent), and joins the per-member IMP score + distinct-target
    count BY NAME (descriptor matrix groups sort=True; never assume positional
    alignment). Returns a frame with x/y/IMP_Final_Score/n_targets/compound_name,
    or ``None`` when there is not enough data to project (caller renders the
    ≥3-members sentinel). Keyed on (collection_id, sorted members, method).
    """
    from frontend.ui.components import collection_aggregation as agg
    from frontend.ui.components import collection_projection as proj

    if _combined_df is None or _combined_df.empty:
        return None
    if "compound_name" not in _combined_df.columns:
        return None
    filtered = _combined_df[_combined_df["compound_name"].isin(members)]
    members_sorted = sorted(set(filtered["compound_name"].dropna().astype(str)))
    if len(members_sorted) < proj.MIN_MEMBERS_FOR_PROJECTION:
        return None

    try:
        matrix = proj.build_descriptor_matrix(filtered)
        coords = proj.project(matrix, method=method, n_members=len(members_sorted))
    except ValueError:
        return None

    coords_df = pd.DataFrame(
        {"compound_name": members_sorted, "x": coords[:, 0], "y": coords[:, 1]}
    )

    # Join IMP score + distinct-target count BY NAME (reindex onto members_sorted).
    # IMP_Final_Score on combined.csv is the raw 0..1 score (weights sum to 1,
    # component scores 0..1, QED multiplier <= 1 — verified imp_scoring.py:654-664).
    # The scatter's hover formats the color as %{...:.0f} expecting INTEGER 0–100
    # (Phase 21 — IMP-as-color always co-presents the integer value), so rescale
    # the raw 0..1 score to integer space here.
    comp = agg.per_member_components(filtered)
    if not comp.empty and agg.IMP_SCORE_COL in comp.columns:
        imp_raw = pd.to_numeric(
            comp[agg.IMP_SCORE_COL].reindex(members_sorted), errors="coerce"
        )
        # Rescale raw [0, 1] -> integer space [0, 100]; leave an already-0..100
        # column untouched (robust to either scale).
        scale = 100.0 if (imp_raw.dropna().le(1.0).all()) else 1.0
        coords_df["IMP_Final_Score"] = (imp_raw * scale).to_numpy()
    else:
        coords_df["IMP_Final_Score"] = 0.0

    promisc = agg.promiscuity(filtered)
    if not promisc.empty:
        coords_df["n_targets"] = promisc.reindex(members_sorted).fillna(0).to_numpy()
    else:
        coords_df["n_targets"] = 0
    return coords_df


@st.cache_data(ttl=20, show_spinner=False)
def _promise_frame(
    collection_id: str,
    members: tuple[str, ...],
    weights: tuple[tuple[str, float], ...],
    _combined_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Cached per-member Promise frame for the Triage verdict table + decision map.

    Mirrors :func:`_project_members` (D-25-CACHE, RESTATED INLINE): the heavy
    combined frame is passed underscore-prefixed (``_combined_df``) so
    ``st.cache_data`` EXCLUDES it from hashing; the real key is
    ``(collection_id, sorted members, frozen weights)``. The frozen weights go IN
    the key with no functional effect in P0a (weights are fixed) — it pre-empts the
    P0b Promise-slider stale-cache bug.

    Calls :func:`collection_promise.compute_promise` for the 0–100 Promise +
    per-component [0,1] contributions, then joins BY NAME (never positionally):

    * the per-member raw IMP score (``IMP_Final_Score``, kept on its native 0–1
      scale — the band/verb helpers operate on 0–1, NOT 0–100), and
    * the three concrete red-flag inputs computed from the same frame — distinct
      active-target count (>10 → promiscuity flag), any scored interference
      (``Interference_Score`` > 0), and the argmax(``Distance_Score``)
      near-best-in-class "(watch)" member.

    Returns a frame with one row per member carrying ``compound_name``, ``Promise``
    (0–100 float, NaN where insufficient data), the Promise component columns,
    ``IMP_Final_Score`` (0–1), ``SMILES``, ``n_targets``, ``_interference``, and
    ``_is_watch``. Returns ``None`` when there is nothing to score (caller renders
    the empty-data sentinel).
    """
    from frontend.ui.components import collection_promise as cp

    if _combined_df is None or _combined_df.empty:
        return None
    if cp.GROUP_COL not in _combined_df.columns:
        return None
    filtered = _combined_df[_combined_df[cp.GROUP_COL].isin(members)]
    if filtered.empty:
        return None

    promise = cp.compute_promise(filtered, weights=weights)
    if promise is None or promise.empty:
        return None
    promise = promise.reset_index()  # compound_name from index -> real column

    # Per-member raw IMP (0–1) + red-flag inputs, joined BY NAME.
    grp = filtered.groupby(cp.GROUP_COL)
    extras = pd.DataFrame(index=promise[cp.GROUP_COL])

    if cp.IMP_SCORE_COL in filtered.columns:
        extras["IMP_Final_Score"] = grp[cp.IMP_SCORE_COL].mean()
    else:
        extras["IMP_Final_Score"] = np.nan

    # First SMILES per member for the decision-map hover customdata.
    if "SMILES" in filtered.columns:
        extras["SMILES"] = grp["SMILES"].first()
    else:
        extras["SMILES"] = ""

    # Red flag 1: distinct active-target count (>10 = promiscuity flag, §0.6).
    if cp.TARGET_COL in filtered.columns:
        extras["n_targets"] = grp[cp.TARGET_COL].nunique()
    else:
        extras["n_targets"] = 0

    # Red flag 2: any scored interference flag set (Interference_Score > 0).
    if cp.INTERFERENCE_COL in filtered.columns:
        extras["_interference"] = grp[cp.INTERFERENCE_COL].max().fillna(0.0) > 0.0
    else:
        extras["_interference"] = False

    out = promise.merge(extras, on=cp.GROUP_COL, how="left")

    # Red flag 3: the argmax(Distance_Score) near-best-in-class "(watch)" member.
    out["_is_watch"] = False
    if "Distance_Score" in filtered.columns:
        dist = grp["Distance_Score"].max()
        if dist.notna().any():
            watch_name = dist.idxmax()
            out.loc[out[cp.GROUP_COL] == watch_name, "_is_watch"] = True

    return out


@st.cache_data(ttl=20, show_spinner=False)
def _cluster_members(
    collection_id: str,
    members: tuple[str, ...],
    k: int,
    _coords_df: pd.DataFrame,
) -> Optional[list[int]]:
    """Cached 2D GMM cluster labels (DIRECT sklearn — Pitfall 1, never imp_gmm).

    Clustering depends on the user-chosen ``k`` so it is keyed SEPARATELY from the
    k-independent projection: (collection_id, sorted members, k). Returns the
    per-row integer cluster labels, or ``None`` when k is out of range (caller
    renders the GMM-insufficient sentinel).
    """
    from frontend.ui.components import collection_projection as proj

    if _coords_df is None or _coords_df.empty:
        return None
    coords = _coords_df[["x", "y"]].to_numpy(dtype=float)
    try:
        labels = proj.cluster_2d(coords, k)
    except ValueError:
        return None
    return [int(v) for v in labels]


@st.cache_data(ttl=20, show_spinner=False)
def _member_imp_components(
    collection_id: str,
    members: tuple[str, ...],
    _combined_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Cached per-member IMP component breakdown (24-02 groupby — never recompute).

    Returns the per-member mean of the IMP component contributions (+ IMP score),
    one row per member indexed by ``compound_name``, or ``None`` when there is no
    component data. Keyed on (collection_id, sorted members).
    """
    from frontend.ui.components import collection_aggregation as agg

    if _combined_df is None or _combined_df.empty:
        return None
    if "compound_name" not in _combined_df.columns:
        return None
    filtered = _combined_df[_combined_df["compound_name"].isin(members)]
    comp = agg.per_member_components(filtered)
    if comp.empty:
        return None
    return comp.reset_index()


@st.cache_data(ttl=20, show_spinner=False)
def _member_raw_frame(
    collection_id: str,
    members: tuple[str, ...],
    _combined_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Cached one-row-per-member frame carrying the RAW IMP/efficiency columns.

    Shared by the ⚗️ Properties & Efficiency SEI–BEI plane AND the 🔬 IMP Analysis
    radar + top-1 contribution bar. Mirrors :func:`_promise_frame` /
    :func:`_member_imp_components`: the heavy combined frame is passed
    underscore-prefixed (``_combined_df``) so ``st.cache_data`` EXCLUDES it from the
    hash; the real key is ``(collection_id, sorted members)``.

    The combined frame has MULTIPLE rows per member (one per activity record), so we
    aggregate to ONE row per member BEFORE the downstream helpers run — feeding the
    raw combined frame straight into ``select_radar_members`` / the efficiency plane
    would silently mix rows from a handful of compounds (the advisor trap). Numeric
    raw columns are mean-aggregated; ``SMILES`` takes the first per member (hover
    customdata). The raw ``*_Score`` / ``BEI`` / ``SEI`` / ``Distance_Score`` /
    ``QED_Multiplier`` / ``IMP_Final_Score`` names are PRESERVED verbatim — the
    Plan-02 radar-prep + Plan-03 contribution bar look up these exact raw names
    (``per_member_components`` renames them to ``*_Contribution`` / ``QED_Impact``,
    so that frame is NOT reusable here).

    Returns a ``compound_name``-column frame (one row per member), or ``None`` when
    there is nothing to aggregate (caller renders the insufficient-data sentinel).
    """
    if _combined_df is None or _combined_df.empty:
        return None
    if "compound_name" not in _combined_df.columns:
        return None
    filtered = _combined_df[_combined_df["compound_name"].isin(members)]
    if filtered.empty:
        return None

    # Raw numeric columns the efficiency plane + radar + contribution bar consume,
    # by their EXACT raw names (omit any the schema does not carry).
    raw_numeric = [
        "IMP_Final_Score",
        "BEI",
        "SEI",
        "Efficiency_Score",
        "Distance_Score",
        "Angle_Score",
        "Interference_Score",
        "PDB_Score",
        "QED_Multiplier",
    ]
    present = [c for c in raw_numeric if c in filtered.columns]
    if not present:
        return None

    grp = filtered.groupby("compound_name")
    out = grp[present].mean()
    if "SMILES" in filtered.columns:
        out["SMILES"] = grp["SMILES"].first()
    return out.reset_index()


@st.cache_data(ttl=20, show_spinner=False)
def _member_pareto_frame(
    collection_id: str,
    members: tuple[str, ...],
    _combined_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Cached per-member Pareto trade-off frame (efficiency axes + IMP).

    One row per selected member with the BEI / SEI ligand-efficiency axes
    (per-member mean, 24-02 aggregate — NEVER recomputed) plus the IMP score in
    integer space [0, 100]. The Stage-3 Pareto view (24-12) flips lower-is-better
    axes and calls :func:`collection_pareto.pareto_front` over these axes. All
    three axes here (BEI, SEI, IMP) are HIGHER-is-better — greater ligand
    efficiency and a higher IMP score are both the "desirable" corner — so the
    view needs NO sign flip. Keyed on (collection_id, sorted members).
    """
    from frontend.ui.components import collection_aggregation as agg

    if _combined_df is None or _combined_df.empty:
        return None
    if "compound_name" not in _combined_df.columns:
        return None
    filtered = _combined_df[_combined_df["compound_name"].isin(members)]
    if filtered.empty:
        return None

    eff = agg.member_efficiency_stats(filtered)  # BEI_mean/max, SEI_mean/max
    if eff is None or eff.empty:
        return None
    rows = pd.DataFrame(index=eff.index)
    if "BEI_mean" in eff.columns:
        rows["BEI"] = pd.to_numeric(eff["BEI_mean"], errors="coerce")
    if "SEI_mean" in eff.columns:
        rows["SEI"] = pd.to_numeric(eff["SEI_mean"], errors="coerce")
    if not {"BEI", "SEI"}.issubset(rows.columns):
        return None

    comp = agg.per_member_components(filtered)
    if not comp.empty and agg.IMP_SCORE_COL in comp.columns:
        imp_raw = pd.to_numeric(
            comp[agg.IMP_SCORE_COL].reindex(rows.index), errors="coerce"
        )
        scale = 100.0 if (imp_raw.dropna().le(1.0).all()) else 1.0
        # Fill any per-member NaN IMP with 0.0: a NaN row is never "dominated" by
        # pareto_front, so it would be falsely marked on the front (advisor catch).
        rows["IMP_Final_Score"] = (imp_raw * scale).fillna(0.0).to_numpy()
    else:
        rows["IMP_Final_Score"] = 0.0

    out = rows.dropna(subset=["BEI", "SEI"]).reset_index()
    out = out.rename(columns={out.columns[0]: "compound_name"})
    return out if not out.empty else None


@st.cache_data(ttl=20, show_spinner=False)
def _member_sar(
    collection_id: str,
    members: tuple[str, ...],
    sim_threshold: float,
    delta_threshold: float,
    _combined_df: pd.DataFrame,
) -> Optional[dict[str, Any]]:
    """Cached SAR-lite compute over the SELECTED subset ONLY (O(N²), T-24-12-01).

    Builds one SMILES per member (``groupby('compound_name')['SMILES'].first()``
    — combined.csv carries a per-row ``SMILES`` column) over the sorted selected
    members, then computes the pairwise Tanimoto matrix
    (:func:`collection_sar.tanimoto_matrix`, RDKit Morgan, None-mol-guarded
    T-24-12-02) and the activity-cliff pairs
    (:func:`collection_sar.activity_cliffs`) at the given thresholds. The O(N²)
    cost is bounded by the multiselect (compute over the selected subset only).
    Keyed on (collection_id, sorted members, sim_threshold, delta_threshold).

    Returns a dict ``{labels, matrix, imp, cliffs}`` (matrix/imp aligned to the
    same sorted-member order as ``labels``), or ``None`` when there is no SMILES
    data to compute over.
    """
    from frontend.ui.components import collection_aggregation as agg
    from frontend.ui.components import collection_sar as sar

    if _combined_df is None or _combined_df.empty:
        return None
    if (
        "compound_name" not in _combined_df.columns
        or "SMILES" not in _combined_df.columns
    ):
        return None
    filtered = _combined_df[_combined_df["compound_name"].isin(members)]
    if filtered.empty:
        return None

    # One SMILES per member, sorted member order (matrix labels + imp align here).
    smiles_by_member = (
        filtered.dropna(subset=["SMILES"])
        .groupby("compound_name")["SMILES"]
        .first()
        .sort_index()
    )
    labels = [str(m) for m in smiles_by_member.index]
    if len(labels) < 2:
        return None
    smiles_list = [str(s) for s in smiles_by_member.to_numpy()]

    matrix = sar.tanimoto_matrix(smiles_list)

    # Per-member IMP in integer space [0, 100], aligned to the SAME sorted order.
    comp = agg.per_member_components(filtered)
    if not comp.empty and agg.IMP_SCORE_COL in comp.columns:
        imp_raw = pd.to_numeric(
            comp[agg.IMP_SCORE_COL].reindex(smiles_by_member.index), errors="coerce"
        )
        scale = 100.0 if (imp_raw.dropna().le(1.0).all()) else 1.0
        imp = (imp_raw * scale).fillna(0.0).to_numpy().tolist()
    else:
        imp = [0.0] * len(labels)

    cliffs = sar.activity_cliffs(
        matrix, imp, sim_threshold=sim_threshold, delta_threshold=delta_threshold
    )
    return {
        "labels": labels,
        "matrix": matrix,
        "imp": imp,
        "cliffs": cliffs,
    }


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

    # The global control row is the SINGLE selection source for every tab and
    # mode. It returns the selected members PLUS the canonical mode flags
    # `small_multiples` (mode B) and `compare` (mode C) — every _render_*_tab
    # branches on these exact identifiers (24-08/09/12 grep `if small_multiples`
    # / `if compare`; do NOT rename them).
    selected_members, small_multiples, compare = _render_member_control(
        all_member_names
    )

    st.divider()

    # Phase 24 9-tab strip (UI-SPEC §1) — 8 core views + Report. ONE strip,
    # members-as-series inside each view (the axis flip — never a tab→subtab tree
    # × N members). Short labels (emoji + 1-2 words) keep 9 tabs comfortable.
    tabs = st.tabs(
        [
            "📊 Triage",
            "🧭 Chemical Space",
            "🎯 Bioactivity & Targets",
            "⚗️ Properties & Efficiency",
            "🔬 IMP Analysis",
            "🖼️ Structures",
            "🧬 Evidence",
            "📋 Data",
            "📄 Report",
        ]
    )

    with tabs[0]:
        # Triage (landing): the Phase-23 Overview rank table folds in here, and
        # the D-12 per-member drill-in (the unmodified compound_detail renderer
        # via internal_prefix) stays reachable as a member index below it
        # (UI-SPEC §7 — drill-in lives in Triage now that there is no Members tab).
        _render_overview_tab(
            collection_id, summary_df, combined_df, selected_members
        )
        st.divider()
        _render_members_tab(all_member_names, storage_path)
    with tabs[1]:
        _render_chemical_space_tab(
            collection_id, combined_df, selected_members, small_multiples, compare
        )
    with tabs[2]:
        # Bioactivity & Targets (24-09): heatmap + promiscuity ranking + overlaid
        # activity distributions. Mode B re-lays the activity dist out as a
        # per-member st.columns(3) mini-grid via the shared helper.
        _render_bioactivity_tab(
            combined_df, selected_members, small_multiples, compare
        )
    with tabs[3]:
        # Properties & Efficiency (24-09 + 24-12): golden-triangle efficiency
        # plane + druglikeness flags + per-member efficiency stats + Pareto
        # front + Compare radar. Mode B re-lays the efficiency plane AND the
        # Pareto front out as per-member mini-grids.
        _render_properties_tab(
            collection_id,
            combined_df,
            selected_members,
            small_multiples,
            compare,
        )
    with tabs[4]:
        _render_imp_analysis_tab(
            collection_id, combined_df, selected_members, small_multiples, compare
        )
    with tabs[5]:
        _render_molecules_tab(member_defs, selected_members)
    with tabs[6]:
        _render_evidence_tab(storage_path, combined_df, selected_members)
    with tabs[7]:
        _render_data_tab(combined_df, selected_members, truncated, collection_id)
    with tabs[8]:
        _render_report_tab(
            collection_id, name, summary_df, combined_df, selected_members
        )

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
        # Clear the caches so the next render reads fresh terminal data — including
        # the heavy Chemical Space / IMP Analysis compute caches (UI-SPEC §6).
        _get_collection_cached.clear()
        _list_collections_cached.clear()
        _project_members.clear()
        _cluster_members.clear()
        _member_imp_components.clear()
        _member_pareto_frame.clear()
        _member_sar.clear()
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


def _render_member_control(
    all_member_names: list[str],
) -> tuple[list[str], bool, bool]:
    """The single member multiselect (axis flip) + Select-all/Clear + mode toggles.

    Keyed on session_state with NO ``default=`` (single ownership), seeded to
    all members on first load (UI-SPEC §2). Returns the selected members PLUS the
    canonical mode flags ``small_multiples`` (mode B) and ``compare`` (mode C) so
    every ``_render_*_tab`` branches on the SAME identifiers (24-08/09/12
    grep-assert ``if small_multiples`` / ``if compare``; do NOT rename them).
    """
    if not all_member_names:
        return [], False, False

    # Seed once: all members selected on first visit to this collection.
    if _MEMBER_SELECT_KEY not in st.session_state:
        st.session_state[_MEMBER_SELECT_KEY] = list(all_member_names)

    def _select_all() -> None:
        st.session_state[_MEMBER_SELECT_KEY] = list(all_member_names)

    def _clear() -> None:
        st.session_state[_MEMBER_SELECT_KEY] = []

    # ONE global control row (UI-SPEC Spacing): multiselect · Select all · Clear
    # · mode toggles. This row is the single selection source for every tab/mode.
    c1, c2, c3, c4 = st.columns([6, 1, 1, 2])
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
    with c4:
        # Mode B (small-multiples ⇄ overlay default) + Mode C (Compare). Each is
        # backed by a session_state key; the canonical `small_multiples` /
        # `compare` flags below read them. Off = mode A (overlay+rank backbone).
        st.toggle(
            "Small multiples",
            key=_SMALL_MULTIPLES_KEY,
            help="Small multiples: per-member mini-panels. Off = Overlay.",
        )
        st.toggle(
            "Compare",
            key=_COMPARE_KEY,
            help="Compare 2–5 members side by side.",
        )

    selected = st.session_state.get(_MEMBER_SELECT_KEY, [])

    # >8 series: keep the overlay, reduce opacity, warn (NO faceting, UI-SPEC §2).
    if len(selected) > _OVERLAY_LEGIBILITY_LIMIT:
        st.warning(
            f"{len(selected)} compounds selected. Charts get hard to read past "
            "~8 — series opacity is reduced. Narrow the selection for clearer "
            "comparison."
        )

    # Canonical mode flags (mode B / mode C) — backed by the toggle widgets in the
    # control row; default off (overlay+rank backbone, mode A).
    small_multiples = bool(st.session_state.get(_SMALL_MULTIPLES_KEY, False))
    compare = bool(st.session_state.get(_COMPARE_KEY, False))

    # Compare-mode 2–5 cap (D-COMPARE-CAP, UI-SPEC Copywriting). Under-2 → info
    # prompt; over-5 → narrow-the-selection warning. The cap copy guides without
    # mutating the selection (the multiselect stays the single source of truth).
    if compare:
        n = len(selected)
        if n < 2:
            st.info("Pick at least 2 members to compare. Select up to 5.")
        elif n > 5:
            st.warning(
                "Compare mode shows up to 5 members. Narrow your selection to "
                "5 or fewer."
            )
    return selected, small_multiples, compare


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


def _render_small_multiples(member_fig_fn, members: list[str], ncols: int = 3) -> None:
    """Shared mode-B layout: an ``st.columns(3)`` per-member mini-panel grid.

    The single small-multiples helper reused across every visual view (24-08
    Chemical Space + IMP Analysis here; 24-09 / 24-12 reuse it). For each selected
    member it calls ``member_fig_fn(member)`` → a COMPACT Plotly figure, renders it
    ``width="stretch"`` with the member-name caption, ``ncols`` panels per row
    (3/row desktop, UI-SPEC §2 Spacing). This is a re-LAYOUT of the ACTIVE visual
    view — NEVER a new tab and NEVER an alteration of the 9-tab strip.

    The per-member ``member_fig_fn`` builder is contractually expected to end with
    :func:`apply_impulator_theme` so the 14px-min theme holds on the mini charts;
    this helper does NOT re-theme (the builder owns the final theme step).
    """
    if not members:
        st.info("No members to display.")
        return
    for row_start in range(0, len(members), ncols):
        row_members = members[row_start : row_start + ncols]
        cols = st.columns(ncols)
        for i, member in enumerate(row_members):
            with cols[i]:
                try:
                    fig = member_fig_fn(member)
                    st.plotly_chart(
                        fig, width="stretch", key=f"sm_{row_start}_{i}_{member}"
                    )
                except Exception as e:  # pragma: no cover - per-panel render guard
                    logger.debug(f"small-multiple render failed for {member}: {e}")
                # Member name as plain text (HC-3).
                st.caption(member)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def _build_red_flags(row: pd.Series) -> str:
    """Concrete red-flag chips for a verdict row (D-25 §0.6, RESTATED INLINE).

    Flags a member when ANY of: distinct active-target count > 10 (promiscuity —
    "high count is a flag to CHECK, not proof"); any scored interference flag set
    (``Interference_Score`` > 0); the member is the argmax(``Distance_Score``)
    near-best-in-class "(watch)" point. Empty string when none match.
    """
    chips: list[str] = []
    n_targets = row.get("n_targets")
    if pd.notna(n_targets) and float(n_targets) > 10:
        chips.append(">10 targets")
    if bool(row.get("_interference")):
        chips.append("interference")
    if bool(row.get("_is_watch")):
        chips.append("near best-in-class")
    return " · ".join(chips)


def _render_overview_tab(
    collection_id: str,
    summary_df: Optional[pd.DataFrame],
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
) -> None:
    """📊 Triage decision dashboard (Phase 25 redesign, D-25-ENTRY-VIEW).

    The band-gated ranked **verdict table is the PRIMARY/default surface** (the
    persona thinks in compounds, not coordinates, and the table is the
    empty-PRIORITIZE-quadrant safety-net). The Promise-vs-IMP **decision map is the
    SUPPORTING "see the spread" view** with hover-2D-structure, the always-visible
    inversion warning, and the Promise transparency stacked bar.

    Replaces the old Phase-21 IMP-descending leaderboard: the sort is now the
    Plan-02 ``verdict_sort_key`` — IMP threshold BAND ascending-risk first (5 bins,
    LEFT-inclusive boundaries 0.30/0.50/0.70/0.90 on the 0–1 IMP scale), THEN
    Promise descending within band — so a high-Promise+high-IMP member routes into
    the VALIDATE bucket, NEVER the top. Promise is shown as a ROUNDED INTEGER
    (D-25-DECODE-PRECISION) and the action column carries exactly the four
    USER-LOCKED verbs PRIORITIZE/MONITOR/VALIDATE/DEPRIORITIZE (D-25-ACTION-VERBS,
    NEVER severity nouns, NEVER banded IMP color). Insufficient-data members
    (Promise NaN) render greyed "insufficient data", never a silent 0.
    """
    from frontend.ui.components import collection_promise as cp
    from frontend.ui.components import embed_structure_viewer

    st.subheader("📊 Triage")

    if not selected_members:
        st.info("No data for the selected members.")
        return

    # Promise frame (cached): 0–100 Promise + components + 0–1 IMP + red-flag
    # inputs, keyed on (collection_id, sorted members, frozen weights).
    promise_df = _promise_frame(
        collection_id,
        tuple(sorted(selected_members)),
        cp.DEFAULT_PROMISE_WEIGHTS,
        combined_df,
    )
    if promise_df is None or promise_df.empty:
        st.info("No data for the selected members.")
        return

    # ---- PRIMARY surface: band-gated ranked verdict table ----
    # Band-gated precedence (band asc-risk, Promise desc within band). IMP stays on
    # the native 0–1 scale for the sort + verb mapping (the helpers expect 0–1).
    ranked = cp.verdict_sort_key(promise_df)

    imp_raw = ranked.get("IMP_Final_Score")

    table = pd.DataFrame(index=ranked.index)
    table["Member"] = ranked[cp.GROUP_COL]
    # Promise as a ROUNDED INTEGER string; greyed "insufficient data" on NaN.
    table["Promise"] = [
        str(int(round(p))) if pd.notna(p) else "insufficient data"
        for p in ranked["Promise"]
    ]
    # Display-only integer IMP 0–100 (separate from the 0–1 banding value).
    table["IMP"] = [
        _imp_int(v) for v in (imp_raw if imp_raw is not None else [np.nan] * len(ranked))
    ]
    # Action verb from the 0–1 IMP band (USER-LOCKED 4-verb mapping).
    table["Action"] = [
        cp.imp_band_verb(v) if pd.notna(v) else "—"
        for v in (imp_raw if imp_raw is not None else [np.nan] * len(ranked))
    ]
    # Concrete red-flag chips.
    table["Key red flags"] = [_build_red_flags(r) for _, r in ranked.iterrows()]

    column_config = {
        "IMP": st.column_config.ProgressColumn(
            "IMP",
            help="IMP score (integer 0–100). Higher = more likely a false-positive artifact.",
            format="%d",
            min_value=0,
            max_value=100,
        ),
    }
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )
    st.caption(
        "Ranked by IMP risk band (lowest-risk first), then Promise within band — "
        "a high-Promise but high-IMP member routes to VALIDATE, never the top. "
        "Use Promise to rank, not to split near-ties."
    )

    st.divider()

    # ---- SUPPORTING view: Promise-vs-IMP decision map (see the spread) ----
    st.markdown("#### Decision map — see the spread")

    # ALWAYS-VISIBLE inversion warning (D-25-DECODE-INVERSION [HIGHEST]) — a
    # visually distinct st.warning block, NOT a collapsed expander.
    st.warning(
        "⚠️ This is NOT the efficiency plane. Here UP = MORE suspicious "
        "(deprioritize); the best leads are LOWER-right (high Promise, low "
        "artifact-risk)."
    )

    # Decision-map frame: rescale 0–1 IMP to the 0–100 plane the factory expects;
    # members with Promise=NaN keep a NaN x-coordinate and are dropped SILENTLY by
    # Plotly (they already surface as greyed "insufficient data" above — no impute).
    map_df = ranked.copy()
    if imp_raw is not None:
        map_df["IMP_display"] = pd.to_numeric(imp_raw, errors="coerce") * 100.0
    else:
        map_df["IMP_display"] = np.nan

    fig = create_decision_map(
        map_df,
        promise_col="Promise",
        imp_col="IMP_display",
        smiles_col="SMILES" if "SMILES" in map_df.columns else None,
        name_col=cp.GROUP_COL,
    )
    st.plotly_chart(fig, width="stretch", key="decision_map")
    # Hover-2D-structure: chart_id MUST be unique ("decision_map") so the namespaced
    # panel #sv-panel-decision_map never collides with the efficiency plane
    # ("eff_plane", Plan 05). name_col is a STRING naming the column the JS reads;
    # passing it does NOT mutate the frame (LOCKED name_col="compound_name").
    embed_structure_viewer(
        chart_id="decision_map",
        x_col="Promise",
        y_col="IMP_display",
        name_col=cp.GROUP_COL,
    )

    # Quadrant legend (fixed Promise=50 divider) + routine decode in a COLLAPSED
    # expander (D-25-DECODE-DISCLOSURE — only the inversion warning stays always-on).
    with st.expander("How to read this decision map"):
        st.markdown(
            "- **What it shows:** every member as one point — Promise (x) vs "
            "IMP-suspicion (y).\n"
            "- **How to read it:** UP = more suspicious; the divider at Promise=50 "
            "splits high/low Promise.\n"
            "- **What to look for (quadrants):**\n"
            "  - high-Promise + high-IMP → **TOO GOOD TO BE TRUE → VALIDATE**\n"
            "  - low-Promise + high-IMP → **DEPRIORITIZE**\n"
            "  - high-Promise + low-IMP → **PRIORITIZE (genuine leads)**\n"
            "  - low-Promise + low-IMP → **Low priority / unremarkable**"
        )

    # ---- Promise transparency stacked bar (plain-language component labels) ----
    component_names = [name for name, _ in cp.DEFAULT_PROMISE_WEIGHTS]
    _component_labels = {
        "potency": "Potency",
        "ligand_efficiency": "Ligand efficiency",
        "promiscuity": "Apparent promiscuity (recorded active targets)",
        "cleanliness": "Cleanliness",
        "druglikeness": "Druglikeness",
    }
    present_components = [c for c in component_names if c in ranked.columns]
    if present_components:
        contrib_df = ranked[[cp.GROUP_COL, *present_components]].rename(
            columns=_component_labels
        )
        bar = create_promise_contribution_bar(contrib_df, name_col=cp.GROUP_COL)
        st.plotly_chart(bar, width="stretch", key="promise_contribution_bar")
        st.caption(
            "Promise is a transparent weighted blend — each segment is a component's "
            "contribution (not a hidden number)."
        )


def _render_bioactivity_tab(
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
    small_multiples: bool = False,
    compare: bool = False,
) -> None:
    """🎯 Bioactivity & Targets (UI-SPEC §1 row 3): the bioactivity heatmap +
    per-member promiscuity ranking + overlaid activity distributions.

    Reuse-heavy — the headline heatmap is consumed from
    :func:`create_bioactivity_heatmap` (D-01 / D-10, NOT redefined here); the
    overlaid activity distribution is the Phase-23 ``create_box_plot``
    (members-as-series via ``color_col="compound_name"`` → qualitative palette,
    Color §). The promiscuity ranking is a view-time aggregation
    (``collection_aggregation.promiscuity`` — distinct-target count per member).

    Mode B (``small_multiples``): the overlaid activity distribution re-lays out
    as an ``st.columns(3)`` per-member mini-grid via the shared
    :func:`_render_small_multiples` helper (each mini-fig themed). The heatmap +
    promiscuity ranking stay single-figure (a heatmap is already a member×target
    matrix — it is not re-gridded per member).
    """
    df = _filter_combined(combined_df, selected_members)
    if df is None or df.empty:
        st.info("No activity data for the selected members.")
        return

    # ---- Activity_Type filter (correctness gate, SPEC §0.3 / D-25-ACTIVITY-TYPE) ----
    # LOCAL to THIS tab ONLY (review #10): the mask is applied to the frame that
    # drives THIS tab's potency/target charts — it is NEVER threaded into the shared
    # `_filter_combined`/dispatcher, so it does NOT subset Triage/Promise/efficiency/
    # radar/Chemical Space. The mask keys off `Activity_Type` (NOT `Standard_Type` —
    # the wrong key silently matches nothing and re-pools assay types, the exact bug
    # this redesign kills). The pure predicate lives in collection_filters.
    from frontend.ui.components import collection_filters as cf

    if "Activity_Type" in df.columns:
        type_counts = (
            df["Activity_Type"].dropna().astype(str).value_counts()
        )
        if not type_counts.empty:
            type_options = list(type_counts.index)
            # Default to the MOST COMMON Activity_Type present.
            active_type = st.selectbox(
                "Activity type",
                options=type_options,
                index=0,
                key=f"collection_activity_type_{id(combined_df)}",
                help=(
                    "Filters THIS tab only. Each assay type (IC50, Ki, EC50…) is a "
                    "different measurement — pooling them re-introduces the exact "
                    "mix-up this redesign removes."
                ),
            )
            mask, available = cf.activity_type_mask(df, active_type)
            if available:
                total_members = int(df["compound_name"].nunique())
                df = df[mask]
                shown_members = (
                    int(df["compound_name"].nunique()) if not df.empty else 0
                )
                # Verbatim filter-state caption (D-25-DECODE-FRAMING / RR-9d) —
                # member counts (distinct compounds), not row counts.
                st.caption(
                    f"Showing {active_type} only — {shown_members} of "
                    f"{total_members} compounds. Switch type or facet to see the rest."
                )
                if df.empty:
                    st.info(
                        f"No members have a {active_type} activity record. "
                        "Switch the activity type above."
                    )
                    return
    else:
        # Fail-OPEN (review #7): absent column → all members shown + a VISIBLE
        # notice, NEVER a silently-empty page.
        st.info(
            "Activity_Type unavailable — showing all members "
            "(this dataset carries no activity-type column to filter on)."
        )

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

    # ---- Promiscuity ranking (per-member distinct-target count, 24-02) ----
    st.subheader("🎯 Target promiscuity")
    from frontend.ui.components import collection_aggregation as agg

    promisc = agg.promiscuity(df)
    if promisc is not None and not promisc.empty:
        rank_df = (
            promisc.sort_values(ascending=False)
            .reset_index()
            .rename(
                columns={
                    "compound_name": "Member",
                    "distinct_targets": "Distinct targets",
                }
            )
        )
        st.dataframe(rank_df, width="stretch", hide_index=True)
        st.caption(
            "Distinct targets hit per member (higher = more promiscuous). "
            "Click the column header to re-sort."
        )
        # Promiscuity framing (verbatim, D-25-DECODE-FRAMING) — a high count is a
        # flag to investigate, not proof of artifact; few targets may be few assays.
        st.caption(
            "recorded active targets — high count is a flag to CHECK, not proof; "
            "few targets may just mean few assays run."
        )
    else:
        st.info("No target data for the selected members.")

    st.divider()

    # ---- Overlaid activity comparison (members as series, color_col) ----
    st.subheader("📈 Activity comparison")
    if "pActivity" not in df.columns:
        st.info("No activity data for the selected members.")
        return

    # ---- Mode B (small multiples): per-member mini-grid re-layout ----
    if small_multiples:
        def _member_activity(member: str) -> go.Figure:
            one = df[df["compound_name"] == member]
            fig = create_box_plot(
                one,
                y_col="pActivity",
                color_col="compound_name",
                title=None,
            )
            fig.update_layout(height=260, showlegend=False)
            # Re-theme as the literal last step so the 14px-min theme holds on
            # the compact mini-chart (UI-SPEC §2 Typography).
            return apply_impulator_theme(fig)

        _render_small_multiples(_member_activity, list(selected_members))
        return

    # ---- Mode A (overlay): single combined activity distribution ----
    fig_box = create_box_plot(
        df,
        y_col="pActivity",
        color_col="compound_name",
        title="pActivity distribution by member",
    )
    fig_box.update_traces(opacity=opacity)
    # Potency y-axis relabel (verbatim) + the µM/nM reference annotation so the
    # −log₁₀ M scale reads in chemist units (D-25-DECODE-FRAMING).
    fig_box.update_layout(
        yaxis_title="Potency (pActivity = −log₁₀ M; higher = more potent)"
    )
    fig_box.add_annotation(
        text="6 ≈ 1 µM · 7 ≈ 100 nM · 8 ≈ 10 nM · 9 ≈ 1 nM",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.08,
        showarrow=False,
        font=dict(size=12),
        align="left",
    )
    st.plotly_chart(fig_box, width="stretch")
    # One-line box-plot legend so the glyphs are self-explanatory.
    st.caption(
        "Box plot: line = median · box = inter-quartile range (25–75%) · "
        "whiskers = spread · points = outliers."
    )


def _render_properties_tab(
    collection_id: str,
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
    small_multiples: bool = False,
    compare: bool = False,
) -> None:
    """⚗️ Properties & Efficiency (UI-SPEC §1 row 4): the golden-triangle
    efficiency plane + Lipinski/Veber druglikeness flags + per-member BEI/SEI
    efficiency stats + the Stage-3 Pareto front + the Compare-mode radar.

    Reuse-heavy — re-slots the Phase-23 :func:`_render_golden_triangle` (which
    draws the LE/LLE diagonal and calls :func:`_render_property_flags`) under
    the new label, plus a per-member efficiency-stats table
    (``collection_aggregation.member_efficiency_stats``). 24-12 adds the Pareto
    trade-off front (potency/efficiency vs IMP, non-dominated members marked) and
    the Compare-mode 2–5 member radar.

    Mode B (``small_multiples``): the efficiency plane AND the Pareto front
    re-lay out as ``st.columns(3)`` per-member compact mini-grids via the shared
    :func:`_render_small_multiples` helper (each mini-fig themed); the flags +
    efficiency-stats tables stay single tables (not chart-bearing).
    """
    df = _filter_combined(combined_df, selected_members)
    if df is None or df.empty:
        st.info("No property data for the selected members.")
        return

    opacity = _overlay_opacity(selected_members)

    # ---- HEADLINE: SEI–BEI efficiency plane (Plan 05, D-25-EFFPLANE-*) ----
    # The promoted efficiency plane is the HEADLINE chart INSIDE this tab — NOT a
    # new tab (the 9-tab strip is fixed). It consumes the one-row-per-member RAW
    # frame (ALL selected members, NOT subset by the Bioactivity Activity_Type
    # filter, which is LOCAL to that tab). The factory recolors to Viridis
    # CONTINUOUS, draws the 45° optimal line + angle banner, and marks the member
    # at argmax(Distance_Score) as "Closest to best-in-class (watch)".
    from frontend.ui.components import embed_structure_viewer

    raw = _member_raw_frame(
        collection_id, tuple(sorted(selected_members)), combined_df
    )
    if raw is not None and not raw.empty and "BEI" in raw.columns and "SEI" in raw.columns:
        st.subheader("⚖️ Efficiency plane — SEI vs BEI (headline)")
        eff_fig = create_efficiency_plane(
            raw,
            sei_col="SEI",
            bei_col="BEI",
            imp_col="IMP_Final_Score",
            distance_col="Distance_Score",
            smiles_col="SMILES" if "SMILES" in raw.columns else None,
            name_col="compound_name",
        )
        if eff_fig.data:
            st.plotly_chart(eff_fig, width="stretch", key="eff_plane")
            # Hover-2D-structure: chart_id "eff_plane" is DISTINCT from the Triage
            # decision map's "decision_map" (Plan 04) so #sv-panel-eff_plane never
            # collides. name_col is a STRING naming the column the JS reads; passing
            # it does NOT mutate the frame (LOCKED name_col="compound_name").
            embed_structure_viewer(
                chart_id="eff_plane",
                x_col="SEI",
                y_col="BEI",
                name_col="compound_name",
            )
            # ALWAYS-VISIBLE marker caption (verbatim, D-25-EFFPLANE-MARKER).
            st.caption(
                "Marked point = Closest to best-in-class (watch) — the member "
                "nearest the best-in-class corner (max Distance_Score)."
            )
            # ALWAYS-VISIBLE tie caption (verbatim, §0.7).
            st.caption(
                "≈80% of the IMP base score is read on this plane (Efficiency 45 "
                "+ Distance 20 + Angle 15 of the base) — high efficiency + near "
                "best-in-class + balanced 45° angle is exactly the 'too good to be "
                "true' pattern."
            )
            # Routine glosses → COLLAPSED expander (D-25-DECODE-DISCLOSURE).
            with st.expander("How to read this efficiency plane"):
                st.markdown(
                    "- **What it shows:** each member as a point — SEI (x) vs BEI "
                    "(y) ligand efficiency, colored by IMP artifact-risk.\n"
                    "- **How to read it:** *modulus* = overall efficiency magnitude; "
                    "*angle* = hydrophobic↔polar balance (45° = optimal).\n"
                    "- **What to look for:** high efficiency + near best-in-class + "
                    "a balanced 45° angle is the 'too good to be true' pattern."
                )
        st.divider()

    # ---- Mode B (small multiples): per-member efficiency-plane mini-grid ----
    if small_multiples and "BEI" in df.columns and "SEI" in df.columns:
        st.subheader("🔺 Efficiency plane (golden triangle)")

        def _member_efficiency(member: str) -> go.Figure:
            one = df[df["compound_name"] == member].dropna(subset=["BEI", "SEI"])
            fig = create_scatter_plot(
                one,
                x_col="SEI",
                y_col="BEI",
                color_col="compound_name",
                title=None,
            )
            fig.update_layout(height=260, showlegend=False)
            # Re-theme as the literal last step (14px-min on the compact mini-chart).
            return apply_impulator_theme(fig)

        _render_small_multiples(_member_efficiency, list(selected_members))
        st.divider()
        # The flags table is still useful in mode B — it is not a chart, so it
        # is not re-gridded; render it below the mini-grid.
        _render_property_flags(df)
    else:
        # ---- Mode A (overlay): golden-triangle efficiency plane (D-13) ----
        _render_golden_triangle(df, opacity)

    st.divider()

    # ---- Per-member efficiency stats (BEI/SEI mean+max, 24-02) ----
    st.subheader("📐 Efficiency stats")
    from frontend.ui.components import collection_aggregation as agg

    eff = agg.member_efficiency_stats(df)
    if eff is not None and not eff.empty:
        st.dataframe(
            eff.reset_index().rename(columns={"compound_name": "Member"}),
            width="stretch",
            hide_index=True,
        )
        st.caption("Per-member ligand-efficiency (BEI/SEI) mean and max.")
    else:
        st.info("No efficiency (BEI/SEI) data for the selected members.")

    # ---- Stage-3 Pareto / trade-off front (24-12) ----
    st.divider()
    _render_pareto_section(
        collection_id, combined_df, selected_members, small_multiples
    )

    # ---- Stage-3 Compare-mode radar (24-12, 2–5 members) ----
    if compare:
        st.divider()
        _render_compare_radar(collection_id, combined_df, selected_members)


def _render_pareto_section(
    collection_id: str,
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
    small_multiples: bool,
) -> None:
    """Pareto trade-off front over (SEI, BEI, −IMP) — non-dominated members marked.

    Desirability directions (``pareto_front`` treats every axis HIGHER-is-better,
    so each lower-is-better axis is negated upstream):
    * SEI / BEI — ligand efficiency, HIGHER is better → passed as-is.
    * IMP_Final_Score — "Higher IMP Score = higher probability the compound is a
      false positive" (IMP_Score.md §Overview / §Score Interpretation, verified),
      so LOWER is better → the IMP axis is NEGATED before the front compute. The
      desirable corner is therefore high-efficiency + low-IMP (efficient, NOT a
      suspected panacea).

    The 2-D plot stays SEI-vs-BEI (colored by IMP, Viridis continuous); the front
    computation considers all three (sign-corrected) axes so a member that wins on
    low-IMP alone still surfaces. Single member → the trade-off info prompt.

    Mode B (``small_multiples``): the front re-lays out as a per-member
    ``st.columns(3)`` mini-grid via the shared helper — each mini-panel plots the
    member's point on the SHARED SEI/BEI axes (a per-member front is degenerate),
    each mini-fig themed.
    """
    from frontend.ui.components import collection_pareto as pareto

    st.subheader("📐 Trade-off front (Pareto)")
    members = tuple(sorted(selected_members))
    if len(members) < 2:
        st.info("Select 2 or more members to see the trade-off front.")
        return

    frame = _member_pareto_frame(collection_id, members, combined_df)
    if frame is None or frame.empty or len(frame) < 2:
        st.info("Select 2 or more members to see the trade-off front.")
        return

    # SEI/BEI are higher-is-better (as-is); IMP is lower-is-better (higher IMP =
    # more likely an invalid panacea — IMP_Score.md), so NEGATE the IMP axis so
    # pareto_front (higher-is-better on every axis) marks the efficient + low-IMP
    # corner as the front.
    sei = frame["SEI"].to_numpy(dtype=float)
    bei = frame["BEI"].to_numpy(dtype=float)
    imp = frame["IMP_Final_Score"].to_numpy(dtype=float)
    points = np.column_stack([sei, bei, -imp])
    front_mask = pareto.pareto_front(points)

    if small_multiples:
        # Shared SEI/BEI axis ranges so the per-member panels are comparable.
        x_pad = (frame["SEI"].max() - frame["SEI"].min()) * 0.1 or 1.0
        y_pad = (frame["BEI"].max() - frame["BEI"].min()) * 0.1 or 1.0
        x_range = [frame["SEI"].min() - x_pad, frame["SEI"].max() + x_pad]
        y_range = [frame["BEI"].min() - y_pad, frame["BEI"].max() + y_pad]

        def _member_pareto_point(member: str) -> go.Figure:
            one = frame[frame["compound_name"] == member]
            mask = [True] * len(one)  # the member is its own (degenerate) front
            fig = create_pareto_plot(
                one, "SEI", "BEI", mask, title=None
            )
            fig.update_layout(
                height=260,
                showlegend=False,
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
            )
            return apply_impulator_theme(fig)

        _render_small_multiples(
            _member_pareto_point, list(frame["compound_name"])
        )
    else:
        pareto_fig = create_pareto_plot(frame, "SEI", "BEI", front_mask)
        st.plotly_chart(pareto_fig, width="stretch")

    # Verbatim Pareto-front caption (Plan 05 Task 3, D-25 spec_lock). The IMP axis
    # is already correctly negated (24-12), so the front = high-efficiency + low-IMP.
    st.caption(
        "non-dominated = best trade-off; nothing beats them on all axes at once"
    )

    # Ranked front list (front members, IMP-desc).
    front_df = frame.loc[front_mask].copy()
    if not front_df.empty:
        front_df = front_df.sort_values("IMP_Final_Score", ascending=False)
        ranked = front_df[["compound_name", "SEI", "BEI", "IMP_Final_Score"]].rename(
            columns={
                "compound_name": "Member",
                "IMP_Final_Score": "IMP",
            }
        )
        ranked["IMP"] = ranked["IMP"].round().astype(int)
        ranked["SEI"] = ranked["SEI"].round(2)
        ranked["BEI"] = ranked["BEI"].round(2)
        st.markdown("**On the front**")
        st.dataframe(ranked, width="stretch", hide_index=True)


def _render_compare_radar(
    collection_id: str,
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
) -> None:
    """Compare-mode 2–5 member radar overlay (member-identity qualitative).

    Builds a per-member IMP-component frame (24-02 aggregate), min-max
    NORMALIZES each axis to ``[0, 1]`` across the selected members (the radar
    factory is dumb — normalization is owned here), then overlays 2–5 members via
    :func:`create_compare_radar`. The 2–5 cap copy already fires in
    :func:`_render_member_control`, so this only renders for an in-range
    selection and stays silent otherwise (no double-warning).
    """
    st.subheader("🕸️ Compare (radar)")
    members = tuple(sorted(selected_members))
    n = len(members)
    if n < 2 or n > 5:
        # The 2–5 cap messaging is emitted once by _render_member_control.
        return

    comp_df = _member_imp_components(collection_id, members, combined_df)
    if comp_df is None or comp_df.empty:
        st.info("No IMP component data for the selected members.")
        return

    from frontend.ui.components.charts import IMP_COMPONENT_COLS

    axis_cols = [c for c in IMP_COMPONENT_COLS if c in comp_df.columns]
    if not axis_cols:
        st.info("No IMP component data for the selected members.")
        return

    norm = comp_df[["compound_name", *axis_cols]].copy()
    for col in axis_cols:
        series = pd.to_numeric(norm[col], errors="coerce")
        lo = float(series.min())
        hi = float(series.max())
        span = hi - lo
        # Constant axis → all members sit at the mid (0.5) so the polygon closes.
        norm[col] = ((series - lo) / span) if span > 0 else 0.5
    norm = norm.fillna(0.0)

    radar_fig = create_compare_radar(norm)
    st.plotly_chart(radar_fig, width="stretch")
    st.caption("Axes are min-max normalized across the selected members.")


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
        # Routine decode → COLLAPSED expander (D-25-DECODE-DISCLOSURE).
        with st.expander("How to read the golden triangle"):
            st.markdown(
                "- **What it shows:** SEI (x) vs BEI (y) ligand efficiency; the "
                "dashed LE/LLE diagonal marks the drug-like region.\n"
                "- **How to read it:** upper-right of the diagonal = efficient on "
                "both axes (the golden triangle).\n"
                "- **What to look for:** the *angle* of a member's efficiency "
                "trajectory reads its hydrophobic ↔ polar balance."
            )
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


def _name_to_entry_ids(combined_df: Optional[pd.DataFrame]) -> dict[str, list[str]]:
    """Build a member-name -> [entry_id, ...] map from ``combined_activities``.

    The Evidence view selects by member NAME (the multiselect, like every other
    tab) but the aggregate artifact is keyed by ``entry_id``. The combined frame
    stamps BOTH columns on every row (``collection_service._stamp_and_concat``),
    so it is the join bridge. A name maps to a LIST of entry_ids so two members
    sharing a display name (distinct entry_ids) both resolve — never collapsed
    onto one (mirrors the index-based de-dup elsewhere on this page, HC-3).
    """
    if (
        combined_df is None
        or combined_df.empty
        or "compound_name" not in combined_df.columns
        or "entry_id" not in combined_df.columns
    ):
        return {}
    out: dict[str, list[str]] = {}
    pairs = (
        combined_df[["compound_name", "entry_id"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    for name, entry_id in pairs.itertuples(index=False):
        out.setdefault(name, []).append(entry_id)
    return out


def _render_evidence_tab(
    storage_path: Optional[str],
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
) -> None:
    """🧬 Evidence & Annotations (UI-SPEC §1 row 7): drug indications + PDB
    structure evidence + chemical classification, combined and comparable across
    the selected members.

    Single-sourced from ``collection_aggregate.json`` (D-S2-ARCH) via the cached
    :func:`_load_collection_aggregate` loader — there is NO on-demand per-member
    fallback fetch. The artifact is keyed by ``entry_id``; the selection is by
    member NAME, so :func:`_name_to_entry_ids` bridges the two via the combined
    frame (which carries both columns).

    Resilience (T-24-10-01): when the artifact is entirely absent (missing/
    unreadable/malformed → empty dict) the view shows the UI-SPEC artifact-missing
    ``st.warning`` and keeps the D-12 per-member drill-in (in the Triage tab)
    reachable for full evidence — it never crashes or shows a blank panel. Each
    of the three sections shows its own empty-sentinel when present-but-empty.

    Member-derived strings (names, indication/PDB labels) render ONLY through
    native Streamlit widgets (escaped by default) — never raw HTML (HC-3).
    """
    parsed = _load_collection_aggregate(storage_path) if storage_path else {}

    # Artifact entirely absent → defensive warning + degrade to the D-12 drill-in
    # (reachable in the Triage tab). Do NOT crash or blank the panel.
    if not parsed:
        st.warning(
            "Combined evidence data isn't available for this collection. "
            "Open an individual member for its full evidence."
        )
        return

    name_to_ids = _name_to_entry_ids(combined_df)

    # Resolve the selected member NAMES to their aggregate entries (by entry_id).
    # Preserve selection order; a name with no resolvable entry_id is skipped here
    # (its absence simply means no combined-frame row — the section sentinels
    # below cover the present-but-empty case).
    resolved: list[tuple[str, Any]] = []  # (display_name, AggregateEntry)
    for name in selected_members:
        for entry_id in name_to_ids.get(name, []):
            entry = parsed.get(entry_id)
            if entry is not None:
                resolved.append((name, entry))

    # ---- Drug indications (combined across members) ----
    st.subheader("💊 Drug indications")
    indication_rows: list[dict[str, Any]] = []
    for name, entry in resolved:
        for row in entry.indications:
            if isinstance(row, dict):
                indication_rows.append({"Member": name, **row})
    if indication_rows:
        st.dataframe(
            pd.DataFrame(indication_rows), width="stretch", hide_index=True
        )
    else:
        st.info("No drug indications recorded for the selected members.")

    st.divider()

    # ---- PDB structure evidence (with scores) ----
    st.subheader("🧬 PDB structure evidence")
    pdb_rows: list[dict[str, Any]] = []
    for name, entry in resolved:
        for row in entry.pdb:
            if isinstance(row, dict):
                pdb_rows.append({"Member": name, **row})
    if pdb_rows:
        st.dataframe(pd.DataFrame(pdb_rows), width="stretch", hide_index=True)
    else:
        st.info("No PDB structure evidence for the selected members.")

    st.divider()

    # ---- Chemical classification (per-member rollup) ----
    st.subheader("🏷️ Chemical classification")
    class_rows: list[dict[str, Any]] = []
    for name, entry in resolved:
        c = entry.classification or {}
        available = c.get("classification_available")
        # Only surface members that actually carry a classification rollup.
        if available is None and not c.get("imp_score") and not c.get("imp_candidates"):
            continue
        class_rows.append(
            {
                "Member": name,
                "Classified": "✓" if available else "—",
                "IMP score": _imp_int(c.get("imp_score")),
                "IMP candidates": c.get("imp_candidates"),
            }
        )
    if class_rows:
        st.dataframe(pd.DataFrame(class_rows), width="stretch", hide_index=True)
    else:
        st.info("No chemical classification available for the selected members.")


def _render_molecules_tab(
    member_defs: list[dict[str, Any]], selected_members: list[str]
) -> None:
    """🖼️ Structures (UI-SPEC §1 row 6 UPGRADE): an ``st.columns(4)`` grid of
    member structure tiles (3-4 molecules/row) PLUS an OpenChemLib hover-preview
    scatter (the upgrade from the static-only Phase-23 2D grid).

    The static tiles reuse :func:`render_2d_structure` (RDKit SVG, with built-in
    None/invalid-SMILES degradation — T-24-09-02). The hover-preview upgrade
    reuses the existing ``structure_viewer.py`` component (OpenChemLib) the
    proven way it is wired in ``compound_detail.py``: a single Plotly scatter
    carrying ``SMILES`` in customdata + :func:`embed_structure_viewer` attached
    to it. The viewer attaches to the NEAREST single Plotly chart, so it is wired
    to ONE scatter (not per-tile) — hovering a point pops the OCL 2D panel.

    Member-derived names are rendered ONLY as plain ``st.caption`` text (HC-3).
    Bounded by the selected members (multiselect) so a large collection never
    renders an unbounded tile wall (T-24-09-01).
    """
    from frontend.ui.components import embed_structure_viewer, render_2d_structure

    selected_defs = [
        m
        for m in member_defs
        if isinstance(m, dict) and m.get("name") in selected_members
    ]
    if not selected_defs:
        st.info("No molecules for the selected members.")
        return

    # ---- OpenChemLib hover-preview scatter (the upgrade) ----
    # One point per member; SMILES rides in customdata so the reused OCL viewer
    # pops the 2D structure on hover/click (UI-SPEC §1: hover-preview upgrade).
    struct_rows = [
        {
            "SMILES": m.get("smiles", ""),
            "Molecule_Name": m.get("name", ""),
            "member_index": i,
        }
        for i, m in enumerate(selected_defs)
        if m.get("smiles")
    ]
    if struct_rows:
        struct_df = pd.DataFrame(struct_rows)
        st.caption("Hover a point to preview its 2D structure (click to pin).")
        fig = create_scatter_plot(
            struct_df,
            x_col="member_index",
            y_col="member_index",
            smiles_col="SMILES",
            name_col="Molecule_Name",
            title="Structure overview (hover to preview)",
        )
        st.plotly_chart(fig, width="stretch", key="collection_structure_scatter")
        embed_structure_viewer(
            chart_id="collection_structure_scatter",
            x_col="member_index",
            y_col="member_index",
            name_col="Molecule_Name",
        )
        st.divider()

    # ---- Static 2D structure tile grid (3-4 molecules/row) ----
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
    """📋 Data (UI-SPEC §1 row 8 UPGRADE): the combined data table with faceted
    filters (numeric range / categorical) + a column chooser, plus the inherited
    250k-cap warning and the full-collection ZIP download.

    Upgrade of the Phase-23 flat dump: the filters + column chooser operate over
    the ALREADY-loaded combined frame (no new backend path — UI filtering only,
    T-24-09 trust boundary). The "Download collection ZIP" link is preserved
    verbatim (always points at the full, unfiltered collection ZIP).
    """
    df = _filter_combined(combined_df, selected_members)

    if truncated:
        n = len(combined_df) if combined_df is not None else 0
        st.warning(
            f"Showing first 250,000 of {n} rows. Full data is in the per-member "
            "sections of the collection ZIP."
        )

    if df is None or df.empty:
        st.info("No data for the selected members.")
        # Full-ZIP download via st.link_button (preserve the inherited control).
        url = get_api_client().collection_download_url(collection_id)
        st.link_button("Download collection ZIP", url, width="stretch")
        return

    # ---- Faceted filters (numeric range + categorical) over the loaded frame ----
    with st.expander("Filters & columns", expanded=False):
        # Numeric facet: pick a numeric column, then a value range slider.
        numeric_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
        ]
        num_facet = st.selectbox(
            "Filter by numeric column (range)",
            options=["(none)"] + numeric_cols,
            key="collection_data_num_facet",
        )
        if num_facet != "(none)":
            series = pd.to_numeric(df[num_facet], errors="coerce")
            lo, hi = float(series.min()), float(series.max())
            if lo < hi:
                sel_lo, sel_hi = st.slider(
                    f"{num_facet} range",
                    min_value=lo,
                    max_value=hi,
                    value=(lo, hi),
                    # Namespace the key by the chosen column: switching facet
                    # columns must NOT carry the previous column's persisted
                    # range (which would be out of the new column's min/max and
                    # crash the slider on rerun — Rule 1 bug).
                    key=f"collection_data_num_range_{num_facet}",
                )
                df = df[series.between(sel_lo, sel_hi)]
            else:
                st.caption(f"{num_facet} has a single value ({lo:g}) — no range to filter.")

        # Categorical facet: low-cardinality string/object column → multiselect.
        cat_cols = [
            c
            for c in df.columns
            if (df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype))
            and df[c].nunique(dropna=True) <= 50
            and df[c].nunique(dropna=True) > 1
        ]
        cat_facet = st.selectbox(
            "Filter by category column",
            options=["(none)"] + cat_cols,
            key="collection_data_cat_facet",
        )
        if cat_facet != "(none)":
            options = sorted(df[cat_facet].dropna().astype(str).unique().tolist())
            chosen = st.multiselect(
                f"{cat_facet} values",
                options=options,
                default=options,
                # Namespace by the chosen column so a previous column's
                # persisted selections (not in the new column's options) do
                # not crash the multiselect on rerun (Rule 1 bug).
                key=f"collection_data_cat_values_{cat_facet}",
            )
            if chosen:
                df = df[df[cat_facet].astype(str).isin(chosen)]

        # Column chooser: subset the displayed columns (multiselect over columns).
        all_cols = list(df.columns)
        shown_cols = st.multiselect(
            "Columns to show",
            options=all_cols,
            default=all_cols,
            key="collection_data_columns",
        )
        if shown_cols:
            df = df[shown_cols]

    if df is not None and not df.empty:
        st.caption(f"{len(df):,} rows after filters.")
        st.dataframe(df, width="stretch", height=450, hide_index=True)
    else:
        st.info("No rows match the current filters.")

    # Full-ZIP download via st.link_button (NOT a buffered download_button).
    # Always the FULL collection ZIP — filters only narrow the on-screen table.
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


# ---------------------------------------------------------------------------
# Net-new Phase 24 tabs (thin stubs — Stage-1/3 view plans fill these in).
# Each delegates to a placeholder showing "view coming in this phase" so the
# 9-tab skeleton is complete; later plans (24-08 Chemical Space / IMP Analysis,
# 24-12 Report) replace these bodies. Stubs accept the same selection + mode
# flags every other tab receives so swapping in the real body is a body-only
# edit (no call-site change).
# ---------------------------------------------------------------------------

def _render_chemical_space_tab(
    collection_id: str,
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
    small_multiples: bool,
    compare: bool,
) -> None:
    """Chemical Space: PCA scatter (member=point, IMP=Viridis) + collection-scale
    2D GMM, gated on >=3 members (UI-SPEC §1 + Copywriting).

    Heavy compute (projection, clustering) is cached on (collection_id, sorted
    members[, k]). Pitfall 1: 2D clustering uses ``collection_projection.cluster_2d``
    (direct sklearn) — NEVER the 1D ``imp_gmm.fit_gmm`` (that belongs in IMP
    Analysis). When the small-multiples mode (B) is active, the ACTIVE view re-lays
    out as a per-member st.columns(3) mini-grid (re-layout, NOT a new tab).
    """
    st.subheader("🧭 Chemical Space")
    # DEMOTED, clarified (Plan 05 Task 3). Verbatim title + caption; the real axis
    # titles ("Similarity axis 1/2 (no units)"), the separated IMP colorbar/legend,
    # and the "Structural group A/B/C" relabel all come from the fixed factory.
    st.markdown(
        "**Structural similarity map — PCA of MW, LogP, H-bond donors/acceptors, "
        "polar surface area**"
    )
    st.caption(
        "Each point is a compound; close = similar physicochemical profile, "
        "distance has no units. Color = IMP artifact-risk; size = number of "
        "targets. Tight clusters of yellow (high-IMP) compounds = a scaffold that "
        "may be an interference class."
    )
    members = tuple(sorted(selected_members))

    # ---- PCA / UMAP projection toggle (Stage-3, UI-SPEC §1 + Copywriting) ----
    # UMAP path is deterministic (seed=42, n_neighbors-clamped) and DEGRADES to
    # PCA inside collection_projection.project when umap-learn is unavailable.
    method = st.segmented_control(
        "Projection",
        options=["PCA", "UMAP"],
        default="PCA",
        key=f"chem_space_method_{collection_id}",
    )
    method = (method or "PCA").lower()
    if method == "umap":
        st.caption(
            "UMAP layout is stochastic; a fixed random seed keeps it "
            "reproducible across reruns."
        )

    coords_df = _project_members(collection_id, members, method, combined_df)
    if coords_df is None or coords_df.empty:
        # ≥3-members projection gate (UI-SPEC copy, verbatim).
        st.info(
            "Need at least 3 members to map the chemical space. Select more members."
        )
        return

    n_members = int(coords_df.shape[0])

    # Collection-scale 2D GMM component-count control. Capped at the member count
    # (a GMM cannot have more components than samples). Default = min(3, N).
    max_k = max(1, n_members)
    default_k = min(3, max_k)
    k = st.slider(
        "GMM clusters",
        min_value=1,
        max_value=max_k,
        value=default_k,
        help="Collection-scale 2D Gaussian-Mixture clusters over the projection.",
    )

    cluster_labels = _cluster_members(collection_id, members, int(k), coords_df)
    cluster_col: Optional[str] = None
    if cluster_labels is not None and len(cluster_labels) == n_members:
        coords_df = coords_df.copy()
        coords_df["cluster"] = [str(c) for c in cluster_labels]
        cluster_col = "cluster"
        if int(k) < n_members:
            st.caption(f"Components reduced to {int(k)} to fit the selected member count.")
    else:
        # GMM could not fit (need more members than components) — sentinel copy.
        st.info(
            "Not enough members to cluster (need more members than components). "
            "Add members or reduce the component count."
        )

    # ---- Mode B (small multiples): per-member mini-grid re-layout of THIS view ----
    if small_multiples:
        def _member_scatter(member: str) -> go.Figure:
            one = coords_df[coords_df["compound_name"] == member]
            fig = create_chemical_space_scatter(
                one, cluster_col=cluster_col, title=None
            )
            fig.update_layout(height=260, showlegend=False)
            # Re-theme as the literal last step so the 14px-min theme holds on
            # the compact mini-chart (UI-SPEC §2 Typography).
            return apply_impulator_theme(fig)

        _render_small_multiples(_member_scatter, list(coords_df["compound_name"]))
    else:
        # ---- Mode A (overlay): single combined scatter + in-chart lasso ----
        # Best-effort in-chart selection highlighting (UI-SPEC §4 / RESEARCH
        # Pattern 1). on_select="rerun" means the selection is known only AFTER
        # st.plotly_chart returns, so we read the PRIOR rerun's selection out of
        # session_state, build a highlight OVERLAY trace, then render. The global
        # multiselect stays the master filter — cross-chart propagation is NOT
        # promised (Pitfall 6 / T-24-13-03: degrade + inform, never over-promise).
        chart_key = f"chem_space_lasso_{collection_id}"
        overlay_fig = create_chemical_space_scatter(coords_df, cluster_col=cluster_col)
        _apply_lasso_highlight(overlay_fig, coords_df, chart_key)

        # selection_mode carries box+lasso+points; the event re-styles within THIS
        # chart only (we never feed it back into the multiselect — that would break
        # the master-filter invariant and risk a rerun loop).
        st.plotly_chart(
            overlay_fig,
            width="stretch",
            key=chart_key,
            on_select="rerun",
            selection_mode=["points", "box", "lasso"],
        )
        st.caption(
            "Tip: the member selector above is the master filter — narrowing it "
            "updates every view. In-chart lasso highlights points within this "
            "chart."
        )

    # ---- SAR-lite section (NOT a 10th tab — folded into Chemical Space) ----
    st.divider()
    _render_sar_section(collection_id, combined_df, selected_members)


def _apply_lasso_highlight(
    fig: go.Figure,
    coords_df: pd.DataFrame,
    chart_key: str,
) -> None:
    """Add an in-chart highlight overlay for the prior rerun's lasso/box selection.

    Best-effort selection highlighting (UI-SPEC §4, RESEARCH Pattern 1). The
    Plotly selection event is read out of ``st.session_state[chart_key]`` (it is
    only populated AFTER ``st.plotly_chart`` returns, hence read-before-render).

    Trace-count-independent by design: ``create_chemical_space_scatter`` emits a
    SINGLE trace when ungrouped but ONE-TRACE-PER-CLUSTER when ``cluster_col`` is
    set, which makes ``point_indices`` ambiguous across traces. So instead of
    index-mapping, we read each selected point's ``x``/``y`` coordinates directly
    and draw a single ring-marker overlay trace at those positions — correct
    regardless of how many cluster traces the figure was split into. The selection
    is NEVER fed back into the multiselect (that stays the master filter — Pitfall
    6 / T-24-13-03: degrade + inform, no cross-chart brushing promise).
    """
    event = st.session_state.get(chart_key)
    if event is None:
        return
    # The stored event mirrors the st.plotly_chart return: attribute access with a
    # mapping fallback (Streamlit returns an attr-dict; guard both shapes).
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None) if selection is not None else None
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return

    xs: list[float] = []
    ys: list[float] = []
    for pt in points:
        px_ = pt.get("x") if isinstance(pt, dict) else getattr(pt, "x", None)
        py_ = pt.get("y") if isinstance(pt, dict) else getattr(pt, "y", None)
        if px_ is not None and py_ is not None:
            xs.append(px_)
            ys.append(py_)
    if not xs:
        return

    # Single ring-marker overlay at the selected coordinates — the Viridis IMP
    # fill underneath is preserved (the rings are an outline accent, never a
    # recolor of the member points).
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(
                size=18,
                color="rgba(0,0,0,0)",
                line=dict(color="#5B21B6", width=3),
            ),
            name="Lasso selection",
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _render_sar_section(
    collection_id: str,
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
) -> None:
    """SAR-lite: pairwise Tanimoto matrix (Viridis) + activity-cliff list.

    Computed over the SELECTED member subset ONLY (the multiselect bounds the
    O(N²) cost — T-24-12-01), cached on (collection_id, sorted members,
    thresholds). The feasibility-cap caption and the no-similar-pairs sentinel
    use the UI-SPEC Copywriting strings verbatim.
    """
    from frontend.ui.components import collection_sar as sar

    st.subheader("🔗 SAR-lite — structural similarity & activity cliffs")
    st.caption(
        "Pairwise similarity is computed across selected members; narrow the "
        "selection for faster results on large collections."
    )

    members = tuple(sorted(selected_members))
    if len(members) < 2:
        st.info(
            "No structurally-similar member pairs above the similarity "
            "threshold. Activity-cliff analysis needs at least one similar pair."
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        sim_th = st.slider(
            "Similarity threshold",
            min_value=0.5,
            max_value=1.0,
            value=float(sar.DEFAULT_SIM_THRESHOLD),
            step=0.05,
            key=f"sar_sim_th_{collection_id}",
            help="Pairs above this Tanimoto similarity count as structurally similar.",
        )
    with c2:
        delta_th = st.slider(
            "IMP gap (cliff)",
            min_value=5.0,
            max_value=50.0,
            value=float(sar.DEFAULT_DELTA_THRESHOLD),
            step=5.0,
            key=f"sar_delta_th_{collection_id}",
            help="A similar pair whose IMP scores differ by more than this is a cliff.",
        )

    result = _member_sar(
        collection_id, members, float(sim_th), float(delta_th), combined_df
    )
    if result is None:
        st.info(
            "No structurally-similar member pairs above the similarity "
            "threshold. Activity-cliff analysis needs at least one similar pair."
        )
        return

    labels = result["labels"]
    matrix = result["matrix"]
    cliffs = result["cliffs"]

    sar_fig = create_sar_matrix(matrix, labels)
    st.plotly_chart(sar_fig, width="stretch")

    # ---- Activity-cliff list (similar pairs with divergent IMP) ----
    if cliffs:
        imp = result["imp"]
        cliff_rows = [
            {
                "Member A": labels[i],
                "Member B": labels[j],
                "Tanimoto": round(float(matrix[i][j]), 3),
                "ΔIMP": int(round(abs(float(imp[i]) - float(imp[j])))),
            }
            for (i, j) in cliffs
        ]
        st.markdown("**Activity cliffs** (structurally similar, divergent IMP)")
        st.dataframe(
            pd.DataFrame(cliff_rows), width="stretch", hide_index=True
        )
    else:
        st.info(
            "No structurally-similar member pairs above the similarity "
            "threshold. Activity-cliff analysis needs at least one similar pair."
        )


def _render_imp_analysis_tab(
    collection_id: str,
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
    small_multiples: bool,
    compare: bool,
) -> None:
    """IMP Analysis: 1D IMP-score population density (imp_gmm, reused as-is) +
    per-member component-breakdown comparison (24-02 → create_imp_component_breakdown).

    Pitfall 1: the 1D ``imp_gmm.fit_gmm`` here is the IMP-SCORE population density
    ONLY — it is NOT the 2D chemical-space clustering (that uses cluster_2d). The
    component breakdown re-lays-out per-member (st.columns(3) mini-grid) under mode B.
    """
    st.subheader("🔬 IMP Analysis")
    members = tuple(sorted(selected_members))

    comp_df = _member_imp_components(collection_id, members, combined_df)
    if comp_df is None or comp_df.empty:
        st.info("No IMP component data for the selected members.")
        return

    # ---- 1D IMP-score population density (imp_gmm, reused as-is) ----
    st.markdown("**IMP score distribution**")
    from backend.modules import imp_gmm

    score_col = "IMP_Final_Score" if "IMP_Final_Score" in comp_df.columns else None
    if score_col is not None:
        raw_scores = pd.to_numeric(comp_df[score_col], errors="coerce").dropna()
        # Per-member IMP scores are raw 0..1 → rescale to integer space [0, 100]
        # for BOTH the model fit and the histogram (create_gmm_density_overlay
        # plots a fixed 0-100 axis and uses the raw array for the histogram).
        scores = np.asarray(raw_scores, dtype=float)
        if scores.size and scores.max() <= 1.0:
            scores = scores * 100.0
        if scores.size >= imp_gmm.MIN_COMPONENTS:
            best_k = imp_gmm.best_fit_k(scores)
            model = imp_gmm.fit_gmm(scores, n_components=best_k)
            density_fig = create_gmm_density_overlay(scores, model)
            # Label the IMP-density bands PRIORITIZE → DEPRIORITIZE at the ×100
            # integer scale (Plan 05 Task 2). The shared Phase-22 factory is left
            # untouched; the action-direction annotations are added at THIS call
            # site on the 0–100 IMP x-axis (low IMP = genuine = prioritize; high
            # IMP = artifact-suspicious = deprioritize).
            density_fig.add_annotation(
                text="◀ PRIORITIZE (low IMP = genuine)",
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.12,
                showarrow=False,
                font=dict(size=12),
                align="left",
            )
            density_fig.add_annotation(
                text="DEPRIORITIZE (high IMP = artifact-suspicious) ▶",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.12,
                showarrow=False,
                font=dict(size=12),
                align="right",
            )
            st.plotly_chart(density_fig, width="stretch")
            # Routine decode → COLLAPSED expander (D-25-DECODE-DISCLOSURE).
            with st.expander("How to read the IMP-score density"):
                st.markdown(
                    "- **What it shows:** the IMP-score population (0–100) of the "
                    "selected members as a histogram with the fitted Gaussian "
                    "components overlaid.\n"
                    "- **How to read it:** the x-axis runs PRIORITIZE (low IMP, "
                    "left) → DEPRIORITIZE (high IMP, right).\n"
                    "- **What to look for:** a cluster pushed to the high-IMP right "
                    "is a sub-population worth scrutinizing for artifacts."
                )
        else:
            st.info(
                "Not enough members to model the IMP-score distribution "
                "(need at least 2). Add members to see the density."
            )
    else:
        st.info("No IMP scores for the selected members.")

    st.divider()

    # ---- Per-member component breakdown (member = series → qualitative color) ----
    st.markdown("**IMP component breakdown**")
    if small_multiples:
        def _member_breakdown(member: str) -> go.Figure:
            one = comp_df[comp_df["compound_name"] == member]
            fig = create_imp_component_breakdown(one, title=None)
            fig.update_layout(height=260, showlegend=False)
            # Re-theme as the literal last step (14px-min on the compact mini-chart).
            return apply_impulator_theme(fig)

        _render_small_multiples(_member_breakdown, list(comp_df["compound_name"]))
        return

    breakdown_fig = create_imp_component_breakdown(comp_df)
    st.plotly_chart(breakdown_fig, width="stretch")

    # ---- IMP component radar overlay (top-5-by-IMP) + top-1 contribution bar ----
    # (Plan 05, D-25-RADAR / D-25-PLAN-RESTATE). Both consume the one-row-per-member
    # RAW frame so select_radar_members sorts by per-MEMBER IMP (not raw combined
    # rows — feeding the combined frame would grab 5 rows from 1-2 compounds).
    from frontend.ui.components import collection_promise as cp

    raw = _member_raw_frame(collection_id, members, combined_df)
    # GUARD #1: _member_raw_frame returns None when there is nothing to aggregate.
    if raw is None or raw.empty:
        st.info("Insufficient data — no members to break down.")
        return

    members5 = cp.select_radar_members(raw, n=5)
    # GUARD #2: select_radar_members can return an EMPTY frame (0 members) —
    # NEVER call .iloc[0] on it (IndexError crashes the app).
    if members5 is None or members5.empty:
        st.info("Insufficient data — no members to break down.")
        return

    st.divider()
    st.markdown("**IMP component radar (most-suspicious by IMP)**")
    # Normalize the 5 raw *_Score axes to [0,1] caller-side (the factory is dumb),
    # then RENAME the axis columns to the plain-language spoke labels (the radar
    # factory derives theta from column names). EXACTLY 5 spokes — NO 6th
    # QED-impact spoke (QED is the bar multiplier, never a radar axis).
    norm = cp.normalize_radar_axes(members5)
    _radar_axis_labels = {
        "Efficiency_Score": "Efficiency-outlier",
        "Distance_Score": "Distance-to-best",
        "Angle_Score": "Development-angle",
        "Interference_Score": "Assay-interference",
        "PDB_Score": "PDB-evidence",
    }
    norm = norm.rename(columns=_radar_axis_labels)
    radar_fig = create_imp_component_radar(norm, name_col="compound_name")
    if radar_fig.data:
        st.plotly_chart(radar_fig, width="stretch", key="imp_component_radar")
        with st.expander("How to read this radar"):
            st.markdown(
                "- **What it shows:** the top-5 most-IMP-suspicious members overlaid "
                "across the 5 IMP base components, each axis min-max normalized "
                "[0,1] across the set.\n"
                "- **How to read it:** spokes are Efficiency-outlier · "
                "Distance-to-best · Development-angle · Assay-interference · "
                "PDB-evidence, each min-max normalized [0,1] across the set "
                "(outer = higher relative to the other members).\n"
                "- **What to look for:** a member that pushes outward on several "
                "spokes at once is the most artifact-suspicious."
            )

    # ---- Top-1-by-IMP weighted-contribution bar (member #1 of the cap-5 set) ----
    # REUSE the cap-5 selection (do NOT recompute). Pass the RAW top-1 row (NOT the
    # normalized frame — normalization discards the raw *_Score weights and the
    # QED_Multiplier the Base×QED subtitle reads). This is the ONLY place QED
    # appears in IMP Analysis.
    top1 = members5.iloc[0]
    member_label = str(top1.get("compound_name", "this member"))
    st.divider()
    st.markdown("**Why the worst one is suspicious — weighted contributions**")
    # ALWAYS-VISIBLE member-naming caption (verbatim grammar, Plan 05 Task 2).
    st.caption(
        f"Breakdown for {member_label} — the most IMP-suspicious of this set."
    )
    bar_fig = create_imp_contribution_bar(top1)
    if bar_fig.data:
        st.plotly_chart(bar_fig, width="stretch", key="imp_contribution_bar")
    else:
        st.info(
            "Insufficient data — no component scores to break down for "
            f"{member_label}."
        )


def _render_report_tab(
    collection_id: str,
    collection_name: str,
    summary_df: Optional[pd.DataFrame],
    combined_df: Optional[pd.DataFrame],
    selected_members: list[str],
) -> None:
    """Report tab (9th) — lazy combined collection-scale HTML report.

    Mirrors ``compound_detail._render_report_tab`` at COLLECTION scale: an
    explicit **"Generate combined report"** primary CTA (the one new ``#5B21B6``
    accent on this page) builds the combined HTML ONLY on click, stashes it in
    ``st.session_state[report_key]``, then offers it via ``st.download_button``.
    Sections combine across the selected members (one per-member summary table +
    aggregate metrics). It is NEVER auto-generated on load — that lazy build is
    the T-24-13-02 DoS mitigation (no HTML on every rerun). Member-supplied names
    are HTML-escaped before interpolation (T-24-13-01).
    """
    st.subheader("📄 Combined report")

    if summary_df is None or summary_df.empty:
        st.info("No data available for report generation.")
        return

    df = summary_df.copy()
    if "compound_name" in df.columns and selected_members:
        df = df[df["compound_name"].isin(selected_members)]
    if df.empty:
        st.info("No data for the selected members.")
        return

    report_key = f"_report_collection_{collection_id}"

    c1, c2, c3 = st.columns([2, 2, 4])
    with c1:
        # The single new primary CTA — type="primary" renders the brand accent
        # (#5B21B6); we never hand-roll the color. Lazy: HTML built only on click.
        if st.button(
            "Generate combined report",
            type="primary",
            key=f"generate_collection_report_{collection_id}",
            help="Build a combined collection-scale HTML report for download.",
            width="stretch",
        ):
            with st.spinner("Generating combined report…"):
                html_content = _build_combined_report_html(
                    collection_name, df, combined_df
                )
                # Cap cached reports (shared "_report_*" eviction with the
                # per-compound report tab).
                from frontend.utils.session_state import evict_report_cache

                evict_report_cache()
                st.session_state[report_key] = html_content
            st.success("Combined report ready for download.")

    with c2:
        if report_key in st.session_state:
            safe_file = re.sub(r"[^\w\-]+", "_", collection_name).strip("_") or "collection"
            st.download_button(
                "📄 Download HTML",
                st.session_state[report_key],
                f"{safe_file}_combined_report.html",
                "text/html",
                key=f"download_collection_report_{collection_id}",
                width="stretch",
            )
        else:
            st.markdown(
                "<small style='color: var(--text-color); opacity: 0.5;'>"
                "Click 'Generate combined report' first</small>",
                unsafe_allow_html=True,
            )

    with c3:
        st.info(
            "💡 **Tip:** Generate the report, download the HTML, then use Ctrl+P "
            "to print to PDF."
        )


def _build_combined_report_html(
    collection_name: str,
    summary_df: pd.DataFrame,
    combined_df: Optional[pd.DataFrame],
) -> str:
    """Build the combined collection-scale HTML report (sections combine members).

    Every member-supplied string (collection name, member names) is passed through
    :func:`html.escape` before interpolation — never raw-interpolated — so a
    crafted member name cannot inject markup into the produced document
    (T-24-13-01, mirrors ``compound_detail``'s report escaping). The figures are
    aggregated from the already-loaded ``collection_summary.csv`` columns; nothing
    is recomputed here.
    """
    safe_name = html.escape(str(collection_name or "Collection"))
    n_members = int(summary_df.shape[0])

    # ---- Aggregate metrics (combine across members) ----
    imp_ints = [
        v
        for v in (
            _imp_int(x) for x in summary_df.get("imp_score", pd.Series(dtype=float))
        )
        if v is not None
    ]
    avg_imp = int(round(sum(imp_ints) / len(imp_ints))) if imp_ints else None
    max_imp = max(imp_ints) if imp_ints else None

    def _sum_col(col: str) -> Optional[int]:
        if col not in summary_df.columns:
            return None
        vals = pd.to_numeric(summary_df[col], errors="coerce").dropna()
        return int(vals.sum()) if not vals.empty else 0

    total_candidates = _sum_col("imp_candidates")
    total_pains = _sum_col("pains_count")
    total_pdb = _sum_col("pdb_structures_count")
    total_indications = _sum_col("drug_indications_count")

    metric_rows = [
        ("Members", str(n_members)),
        ("Average IMP", str(avg_imp) if avg_imp is not None else "N/A"),
        ("Best IMP", str(max_imp) if max_imp is not None else "N/A"),
        (
            "IMP candidates",
            str(total_candidates) if total_candidates is not None else "N/A",
        ),
        ("PAINS flags", str(total_pains) if total_pains is not None else "N/A"),
        ("PDB structures", str(total_pdb) if total_pdb is not None else "N/A"),
        (
            "Drug indications",
            str(total_indications) if total_indications is not None else "N/A",
        ),
    ]
    metric_html = "".join(
        f'<div style="display:inline-block;min-width:150px;margin:6px 12px 6px 0;'
        f'padding:10px 14px;border:1px solid #e5e7eb;border-radius:8px;">'
        f'<div style="font-size:11px;text-transform:uppercase;opacity:0.6;">'
        f"{html.escape(label)}</div>"
        f'<div style="font-size:22px;font-weight:700;color:#5B21B6;">'
        f"{html.escape(value)}</div></div>"
        for label, value in metric_rows
    )

    # ---- Per-member combined table (member = a row) ----
    table_cols = [
        ("compound_name", "Member"),
        ("imp_score", "IMP"),
        ("imp_candidates", "IMP candidates"),
        ("pains_count", "PAINS"),
        ("pdb_structures_count", "PDB"),
        ("drug_indications_count", "Indications"),
        ("qed", "QED"),
    ]
    present = [(c, lab) for c, lab in table_cols if c in summary_df.columns]
    header_html = "".join(
        f'<th style="text-align:left;padding:8px 10px;border-bottom:2px solid '
        f'#5B21B6;">{html.escape(lab)}</th>'
        for _, lab in present
    )

    sorted_df = summary_df
    if "imp_score" in summary_df.columns:
        sorted_df = summary_df.sort_values(
            by="imp_score", ascending=False, na_position="last"
        )

    body_rows: list[str] = []
    for _, row in sorted_df.iterrows():
        cells: list[str] = []
        for col, _lab in present:
            raw = row.get(col)
            if col == "imp_score":
                disp = _imp_int(raw)
                text = str(disp) if disp is not None else "N/A"
            elif col == "qed":
                try:
                    text = f"{float(raw):.2f}" if pd.notna(raw) else "N/A"
                except (TypeError, ValueError):
                    text = "N/A"
            elif col == "compound_name":
                # Member-supplied name — ESCAPED (T-24-13-01).
                text = str(raw) if pd.notna(raw) else ""
            else:
                try:
                    text = str(int(float(raw))) if pd.notna(raw) else "0"
                except (TypeError, ValueError):
                    text = "0"
            cells.append(
                f'<td style="padding:6px 10px;border-bottom:1px solid #eee;">'
                f"{html.escape(text)}</td>"
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    table_body = "".join(body_rows)

    total_activities = (
        int(combined_df.shape[0]) if combined_df is not None else 0
    )

    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{safe_name} — Combined IMP Report</title>"
        "<style>"
        "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
        "max-width:1000px;margin:24px auto;padding:0 16px;color:#1f2937;}"
        "h1{color:#5B21B6;} h2{margin-top:28px;border-bottom:1px solid #eee;"
        "padding-bottom:6px;} table{border-collapse:collapse;width:100%;"
        "font-size:14px;} .muted{opacity:0.6;font-size:12px;}"
        "</style></head><body>"
        f"<h1>IMPULATOR — Combined Collection Report</h1>"
        f"<h2>{safe_name}</h2>"
        f"<p class='muted'>{n_members} member(s) · {total_activities} "
        "combined activity row(s)</p>"
        "<h2>Aggregate metrics</h2>"
        f"<div>{metric_html}</div>"
        "<h2>Per-member summary</h2>"
        f"<table><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{table_body}</tbody></table>"
        "<p class='muted'>IMP shown as an integer 0–100 (higher is more "
        "IMP-like). Generated by IMPULATOR.</p>"
        "</body></html>"
    )


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
