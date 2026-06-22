"""Stage-1 view-time aggregation for the Collections combined view (no Streamlit, no IO).

Pure pandas groupby aggregations over the columns already present on
``combined_activities.csv`` (the output of ``build_combined_activities``). This is the
substrate for the Triage and IMP Analysis views: per-member IMP component breakdown,
efficiency (BEI/SEI) stats, druglikeness / assay-interference flag counts, and
distinct-target promiscuity.

⚠ NO RECOMPUTE (RESEARCH anti-pattern): every IMP component contribution is ALREADY a
column on ``combined.csv`` (verified ``imp_scoring.py:664-673``). This module aggregates
those columns; it never calls ``imp_scoring`` / RDKit to recompute them.

Streamlit-free so it is unit-testable without a Streamlit runtime (CLAUDE.md + design §7
invariant); the thin ``_render_*`` functions in ``collections.py`` call into it.

⚠ Name-collision: this is ``collection_aggregat**ion**.py`` (Stage-1 stats), distinct from
``collection_aggregat**e**.py`` (Stage-2 JSON artifact reader, plan 24-05).
"""

from __future__ import annotations

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Member key column stamped by build_combined_activities (col0).
GROUP_COL = "compound_name"

# IMP component contribution columns — ALREADY present on combined.csv
# (verified imp_scoring.py:664-673). Aggregate, never recompute.
COMPONENT_COLS = [
    "Efficiency_Contribution",
    "Distance_Contribution",
    "Angle_Contribution",
    "Interference_Contribution",
    "PDB_Contribution",
    "QED_Impact",
]

# Final IMP score column, also aggregated alongside the components.
IMP_SCORE_COL = "IMP_Final_Score"

# Efficiency metric columns (mean + max per member).
EFFICIENCY_COLS = ["BEI", "SEI"]

# Druglikeness / assay-interference flag columns (verified compound_service.py:415-421,
# imp_scoring.py:65-68). Boolean flags are summed (flagged-row count); RO5_Violations is
# an integer count, so its "flag" is rows with violation count > 0.
INTERFERENCE_FLAG_COLS = [
    "PAINS_Violation",
    "Aggregator_Risk",
    "Redox_Reactive",
    "Fluorescence_Interference",
    "Thiol_Reactive",
    "BRENK_Alerts",
    "NIH_Alerts",
]
# Lipinski Rule-of-Five violation count column (integer; >0 == flagged).
RO5_VIOLATION_COL = "RO5_Violations"

# Distinct-target column for promiscuity.
TARGET_COL = "Target_Name"


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return the subset of ``cols`` present in ``df``; warn on any absent (T-24-02-02)."""
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("aggregation_columns_absent", missing=missing)
    return present


def per_member_components(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Per-member mean of the IMP component contributions + IMP_Final_Score.

    Pure groupby over ``compound_name`` — NO recompute (the components are columns).
    Absent component/score columns are omitted (warned), never assumed present.
    Returns a DataFrame indexed by ``compound_name``.
    """
    cols = _present(combined_df, [*COMPONENT_COLS, IMP_SCORE_COL])
    if GROUP_COL not in combined_df.columns or not cols:
        logger.warning("per_member_components_no_data", group_present=GROUP_COL in combined_df.columns)
        return pd.DataFrame()
    # combined_df.groupby('compound_name')[COMPONENT_COLS].mean() — aggregate, never recompute
    return combined_df.groupby(GROUP_COL)[cols].mean()


def member_efficiency_stats(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Per-member BEI/SEI mean and max.

    Columns out: ``{metric}_mean`` / ``{metric}_max`` for each present efficiency metric.
    Absent metrics are omitted (warned).
    """
    cols = _present(combined_df, EFFICIENCY_COLS)
    if GROUP_COL not in combined_df.columns or not cols:
        logger.warning("member_efficiency_stats_no_data")
        return pd.DataFrame()
    grouped = combined_df.groupby(GROUP_COL)[cols]
    out = grouped.agg(["mean", "max"])
    # Flatten the (metric, stat) MultiIndex columns to "{metric}_{stat}".
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out


def druglikeness_flag_counts(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Per-member count of druglikeness / interference flags, from existing columns only.

    Boolean interference flags (PAINS/Aggregator/Redox/Fluorescence/Thiol, BRENK/NIH
    alerts) are summed → flagged-row count per member. ``RO5_Violations`` is an integer
    violation count, so its flag is the number of rows with violations > 0.

    Zero recompute. If NO flag columns are present, returns an empty frame (the toy
    fixture carries none — this guard is what the fixture exercises).
    """
    if GROUP_COL not in combined_df.columns:
        logger.warning("druglikeness_flag_counts_no_group_col")
        return pd.DataFrame()

    bool_cols = [c for c in INTERFERENCE_FLAG_COLS if c in combined_df.columns]
    has_ro5 = RO5_VIOLATION_COL in combined_df.columns
    if not bool_cols and not has_ro5:
        logger.warning("druglikeness_flag_counts_no_flag_columns")
        return pd.DataFrame()

    parts: list[pd.Series] = []
    for col in bool_cols:
        # fillna(False) first: astype(bool) maps NaN -> True (NumPy truthiness),
        # which would inflate flag counts for rows missing the flag. Then 1/0 or
        # True/False both count as flagged rows.
        parts.append(
            combined_df[col].fillna(False).astype(bool).groupby(combined_df[GROUP_COL]).sum().rename(col)
        )
    if has_ro5:
        flagged = (combined_df[RO5_VIOLATION_COL] > 0)
        parts.append(flagged.groupby(combined_df[GROUP_COL]).sum().rename(RO5_VIOLATION_COL))

    out = pd.concat(parts, axis=1)
    out.index.name = GROUP_COL
    return out.astype(int)


def promiscuity(combined_df: pd.DataFrame) -> pd.Series:
    """Per-member distinct-target count (promiscuity).

    Pure ``nunique`` over ``Target_Name`` grouped by member. If the target column is
    absent, returns an empty Series (warned).
    """
    if GROUP_COL not in combined_df.columns or TARGET_COL not in combined_df.columns:
        logger.warning("promiscuity_missing_columns", target_present=TARGET_COL in combined_df.columns)
        return pd.Series(dtype="int64", name="distinct_targets")
    out = combined_df.groupby(GROUP_COL)[TARGET_COL].nunique()
    out.name = "distinct_targets"
    return out
