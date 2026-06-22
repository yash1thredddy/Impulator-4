"""Pure Promise-composite + radar-prep + band-gated verdict logic (no Streamlit, no IO).

This is the correctness-critical logic layer for the Collections decision dashboard
(Phase 25). It mirrors the structure of ``collection_aggregation.py`` (Stage-1 pure-pandas
over ``combined_activities.csv`` — D-25-MODULE-TEMPLATE): Streamlit-free, lazy, structlog
only, ``GROUP_COL="compound_name"``, guard-and-omit for absent columns (never crash).

It exposes four genuinely net-new logic units (RR-3/RR-4/§0.2/§0.3):

1. ``compute_promise`` — a 0–100 per-member Promise score: a transparent weighted blend of
   5 normalized [0,1] components (potency, ligand-efficiency, apparent-promiscuity,
   cleanliness, druglikeness) with FROZEN default weights summing to 1.0. A MISSING component
   (absent column OR per-member NaN value) is dropped and the remaining weights reweight to
   re-sum 1.0; ALL-missing → Promise = NaN ("insufficient data", never a silent 0).
2. ``select_radar_members`` — top-5-by-IMP-descending selection (crash-free over 0/1/<5/5/>5).
3. ``normalize_radar_axes`` — per-axis min-max [0,1] over the FIVE raw ``*_Score`` columns
   (constant/all-NaN axis → 0.5 neutral midpoint, never divide-by-zero). EXACTLY 5 axes.
4. ``verdict_sort_key`` + ``imp_band_index`` / ``imp_band_verb`` — the §0.2 5-bin
   LEFT-inclusive band-gated sort (band asc-risk first, Promise desc within band) and the
   USER-LOCKED 4-verb action mapping, sharing one boundary rule so they cannot drift.

⚠ Display rounding (D-25-DECODE-PRECISION) is CALLER-side; this module returns floats.

⚠ Name-collision: this is NOT ``collection_aggregate.py`` (Stage-2 JSON reader) nor
``collection_aggregation.py`` (Stage-1 stats) — it is the Promise/decision logic layer.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Member key column stamped by build_combined_activities (col0).
GROUP_COL = "compound_name"

# ---------------------------------------------------------------------------
# Promise composite (SPEC §3.1) — FROZEN default weights (tuple-of-pairs so it is
# immutable AND can go IN a cache key). Sum == 1.0.
# ---------------------------------------------------------------------------
DEFAULT_PROMISE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("potency", 0.30),
    ("ligand_efficiency", 0.25),
    ("promiscuity", 0.20),
    ("cleanliness", 0.15),
    ("druglikeness", 0.10),
)

# Per-component source columns on combined_activities.csv.
POTENCY_COL = "pActivity"
BEI_COL = "BEI"
SEI_COL = "SEI"
TARGET_COL = "Target_Name"
INTERFERENCE_COL = "Interference_Score"  # already scored-flags/5 (D-25-CLEANLINESS)
QED_COL = "QED"  # REAL RDKit druglikeness 0–1; NOT QED_Impact (the IMP multiplier-impact)

IMP_SCORE_COL = "IMP_Final_Score"

# Raw radar axes — read DIRECTLY (D-25-RAW-SCORES). EXACTLY FIVE; no 6th QED spoke.
RADAR_AXES: tuple[str, ...] = (
    "Efficiency_Score",
    "Distance_Score",
    "Angle_Score",
    "Interference_Score",
    "PDB_Score",
)

# §0.2 IMP threshold bands (0–1). LEFT-inclusive / right-EXCLUSIVE.
# 5 bins: [0.00,0.30) [0.30,0.50) [0.50,0.70) [0.70,0.90) [0.90,1.00].
BAND_EDGES: tuple[float, ...] = (0.30, 0.50, 0.70, 0.90)

# USER-LOCKED 4-verb action mapping (D-25-ACTION-VERBS). The [0.30,0.50) and [0.50,0.70)
# sort bins BOTH fold into MONITOR — do NOT add a 5th proceed-style verb.
_BAND_VERBS: tuple[str, ...] = (
    "PRIORITIZE",  # band 0: [0.00,0.30)
    "MONITOR",     # band 1: [0.30,0.50)
    "MONITOR",     # band 2: [0.50,0.70)
    "VALIDATE",    # band 3: [0.70,0.90)
    "DEPRIORITIZE",  # band 4: [0.90,1.00]
)


def _clip01(x: pd.Series | float) -> pd.Series | float:
    """Clip to [0,1]."""
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Promise composite
# ---------------------------------------------------------------------------
def _potency_component(grouped_mean: pd.Series) -> pd.Series:
    """pActivity 5→0.0 … 9→1.0, clipped to [0,1]."""
    return pd.Series(_clip01((grouped_mean - 5.0) / 4.0), index=grouped_mean.index)


def _ligand_efficiency_component(bei: pd.Series, sei: pd.Series) -> pd.Series:
    """BEI 'good' range 10–30 and SEI 'good' range 5–25 (each → [0,1]), averaged.

    Either metric may be absent (NaN) per member; the average is taken over the present
    metrics only (so a member with only BEI still gets a ligand-efficiency component).
    """
    bei_n = pd.Series(_clip01((bei - 10.0) / 20.0), index=bei.index) if bei is not None else None
    sei_n = pd.Series(_clip01((sei - 5.0) / 20.0), index=sei.index) if sei is not None else None
    parts = [p for p in (bei_n, sei_n) if p is not None]
    if not parts:
        return pd.Series(np.nan, index=bei.index if bei is not None else sei.index)
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1, skipna=True)


def _promiscuity_component(distinct_targets: pd.Series) -> pd.Series:
    """INVERSE of distinct Target_Name count: 1 target → 1.0, ≥10 → 0.0, clipped.

    Coverage-confounded count, NOT a virtue. Linear map: (10 - n) / 9 over [1,10].
    """
    val = (10.0 - distinct_targets) / 9.0
    return pd.Series(_clip01(val), index=distinct_targets.index)


def _cleanliness_component(interference: pd.Series) -> pd.Series:
    """1 − Interference_Score (read the frame's Interference_Score DIRECTLY). D-25-CLEANLINESS."""
    return 1.0 - interference


def _druglikeness_component(qed: pd.Series) -> pd.Series:
    """QED as-is (already 0–1). Uses the REAL QED column, NOT QED_Impact."""
    return qed


def compute_promise(
    df: pd.DataFrame, *, weights: tuple[tuple[str, float], ...] = DEFAULT_PROMISE_WEIGHTS
) -> pd.DataFrame:
    """Per-member 0–100 Promise score + per-component normalized [0,1] contributions.

    Aggregates multi-row members by ``compound_name`` (mean of scalar inputs, ``nunique``
    for the target count — mirrors collection_aggregation's groupby), then blends 5 normalized
    components with ``weights``.

    Missing-component contract (D-25-PROMISE-EDGECASE):
      - a component whose source column is ABSENT, OR whose per-member value is NaN, is
        DROPPED for that member and the remaining component weights reweight to re-sum 1.0;
      - a member with ALL 5 components missing → Promise = NaN (insufficient data, never 0).

    Returns a DataFrame indexed by ``compound_name`` with columns:
      ``Promise`` (0–100 float) and one column per component name (the [0,1] contribution,
      NaN where that component was dropped for the member).
    Display rounding stays CALLER-side.
    """
    weight_map = dict(weights)
    component_names = [name for name, _ in weights]

    if df is None or df.empty or GROUP_COL not in (df.columns if df is not None else []):
        logger.warning("compute_promise_no_group_col")
        return pd.DataFrame(columns=["Promise", *component_names])

    members = pd.Index(df[GROUP_COL].drop_duplicates(), name=GROUP_COL)

    def _grouped_mean(col: str) -> pd.Series | None:
        if col not in df.columns:
            logger.warning("promise_component_column_absent", column=col)
            return None
        return df.groupby(GROUP_COL)[col].mean().reindex(members)

    # Potency.
    pact = _grouped_mean(POTENCY_COL)
    potency = _potency_component(pact) if pact is not None else None

    # Ligand efficiency (average of present BEI/SEI).
    bei = _grouped_mean(BEI_COL)
    sei = _grouped_mean(SEI_COL)
    ligeff = (
        _ligand_efficiency_component(bei, sei) if (bei is not None or sei is not None) else None
    )

    # Promiscuity (inverse distinct-target count).
    if TARGET_COL in df.columns:
        distinct = df.groupby(GROUP_COL)[TARGET_COL].nunique().reindex(members)
        # nunique() returns 0 for a member whose Target_Name is entirely missing.
        # Left as 0 it would map to a *perfect* promiscuity score; treat "no
        # target data" as missing (NaN) so the member is excluded, not rewarded.
        distinct = distinct.mask(distinct == 0)
        promisc = _promiscuity_component(distinct)
    else:
        logger.warning("promise_component_column_absent", column=TARGET_COL)
        promisc = None

    # Cleanliness (1 - Interference_Score).
    inter = _grouped_mean(INTERFERENCE_COL)
    cleanliness = _cleanliness_component(inter) if inter is not None else None

    # Druglikeness (QED as-is).
    qed = _grouped_mean(QED_COL)
    druglikeness = _druglikeness_component(qed) if qed is not None else None

    component_series: dict[str, pd.Series | None] = {
        "potency": potency,
        "ligand_efficiency": ligeff,
        "promiscuity": promisc,
        "cleanliness": cleanliness,
        "druglikeness": druglikeness,
    }

    # Assemble a per-member component frame (NaN where dropped/absent).
    out = pd.DataFrame(index=members)
    for name in component_names:
        s = component_series.get(name)
        out[name] = s if s is not None else np.nan

    # Per-member weighted blend with reweight-on-missing.
    weight_vec = pd.Series({name: weight_map[name] for name in component_names})
    present_mask = out[component_names].notna()  # member × component
    # weighted value (NaN treated as 0 contribution, but its weight is excluded below).
    weighted = out[component_names].fillna(0.0).mul(weight_vec, axis=1)
    numer = weighted.sum(axis=1)
    denom = present_mask.mul(weight_vec, axis=1).sum(axis=1)  # sum of present weights

    promise01 = numer / denom  # reweight: divide by present-weight sum
    promise01[denom == 0.0] = np.nan  # ALL components missing → NaN, never 0

    out.insert(0, "Promise", promise01 * 100.0)
    out.index.name = GROUP_COL
    return out


# ---------------------------------------------------------------------------
# Radar-prep
# ---------------------------------------------------------------------------
def select_radar_members(df: pd.DataFrame, *, n: int = 5) -> pd.DataFrame:
    """Top-``n`` members by ``IMP_Final_Score`` DESCENDING (most-suspicious first).

    Crash-free over 0/1/<n/==n/>n members (D-25-RADAR): empty frame for 0; all available
    (capped at ``n``, no padding) for 1..n; the top-``n`` for >n. STABLE sort so IMP ties
    preserve input order deterministically. A MISSING ``IMP_Final_Score`` column → return the
    available members in input order (no raise).
    """
    if df is None or df.empty:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()

    if IMP_SCORE_COL not in df.columns:
        logger.warning("select_radar_members_no_imp_column")
        return df.head(n).copy()

    ordered = df.sort_values(by=IMP_SCORE_COL, ascending=False, kind="stable", na_position="last")
    return ordered.head(n).copy()


def normalize_radar_axes(
    df: pd.DataFrame, axes: Iterable[str] = RADAR_AXES
) -> pd.DataFrame:
    """Per-axis min-max [0,1] normalization over the FIVE raw ``*_Score`` columns.

    min→0.0, max→1.0 per axis. CONSTANT or ALL-NaN axis (max==min) → every member maps to
    0.5 (neutral midpoint), never divide-by-zero, never NaN/inf (review #2). Returns a frame
    carrying ``compound_name`` (if present) + EXACTLY the requested axis columns (no 6th
    QED/QED_Impact spoke — D-25-PLAN-RESTATE). Absent axes are omitted (warned).
    """
    axes = tuple(axes)
    out = pd.DataFrame(index=df.index if df is not None else None)
    if df is None or df.empty:
        return out

    if GROUP_COL in df.columns:
        out[GROUP_COL] = df[GROUP_COL].values

    for ax in axes:
        if ax not in df.columns:
            logger.warning("radar_axis_column_absent", axis=ax)
            continue
        col = df[ax]
        amin = col.min(skipna=True)
        amax = col.max(skipna=True)
        if pd.isna(amin) or pd.isna(amax) or amax == amin:
            # constant or all-NaN axis → neutral midpoint
            out[ax] = 0.5
        else:
            normed = (col - amin) / (amax - amin)
            out[ax] = normed.fillna(0.5)  # any residual NaN (shouldn't occur) → neutral
    return out


# ---------------------------------------------------------------------------
# §0.2 band-gated verdict sort + 4-verb action mapping
# ---------------------------------------------------------------------------
def imp_band_index(imp: float) -> int:
    """LEFT-inclusive / right-EXCLUSIVE 5-bin index for an IMP on the 0–1 scale.

    [0.00,0.30)→0  [0.30,0.50)→1  [0.50,0.70)→2  [0.70,0.90)→3  [0.90,1.00]→4.
    So IMP==0.30→1, IMP==0.70→3 (the higher-risk band's LOWER edge).
    """
    if imp is None or (isinstance(imp, float) and np.isnan(imp)):
        return len(BAND_EDGES)  # NaN → highest-risk bin (sorts last)
    idx = 0
    for edge in BAND_EDGES:
        if imp >= edge:
            idx += 1
        else:
            break
    return idx


def imp_band_verb(imp: float) -> str:
    """USER-LOCKED 4-verb action mapping (D-25-ACTION-VERBS), same boundary rule as the sort.

    PRIORITIZE / MONITOR / VALIDATE / DEPRIORITIZE — NEVER a severity noun, NEVER a 5th verb.
    IMP==0.30→MONITOR, IMP==0.70→VALIDATE.
    """
    return _BAND_VERBS[imp_band_index(imp)]


def verdict_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` sorted by the §0.2 band-gated precedence.

    (1) IMP threshold BAND ascending-RISK FIRST (5 bins, LEFT-inclusive/right-EXCLUSIVE via
    the band edges 0.30/0.50/0.70/0.90 on the 0–1 IMP scale), THEN (2) Promise DESCENDING
    WITHIN each band. A high-Promise + high-IMP member never sorts to the top — it lands in
    its (higher-risk) band below every lower-risk member regardless of Promise.

    Reads ``IMP_Final_Score`` (0–1) and ``Promise`` columns. Does not mutate the input.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    work = df.copy()
    if IMP_SCORE_COL in work.columns:
        work["_band"] = work[IMP_SCORE_COL].apply(imp_band_index)
    else:
        logger.warning("verdict_sort_no_imp_column")
        work["_band"] = 0

    promise = work["Promise"] if "Promise" in work.columns else pd.Series(0.0, index=work.index)
    work["_neg_promise"] = -promise.fillna(-np.inf)  # desc within band; NaN promise sorts last

    work = work.sort_values(by=["_band", "_neg_promise"], kind="stable")
    return work.drop(columns=["_band", "_neg_promise"])
