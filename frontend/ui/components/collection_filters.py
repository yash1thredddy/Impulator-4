"""Pure Activity_Type filter predicate for the Collections combined view (no Streamlit, no IO).

SPEC §6 names this module; widget rendering stays in ``collections.py``. The predicate keys
off the ``Activity_Type`` column — NOT ``Standard_Type`` (D-25-ACTIVITY-TYPE / §0.3): keying
on the wrong column silently re-pools heterogeneous assay types, reintroducing the exact bug
this redesign kills. This is a correctness gate, not a rename.

FAIL-OPEN contract (review #7): if ``Activity_Type`` is ABSENT, return an ALL-TRUE mask
(keep every member — never fail-closed to an empty all-False mask that would silently blank
the page) ALONGSIDE an ``available: bool`` flag so the caller can surface a visible
"Activity_Type unavailable" notice. Never raises.

Streamlit-free so it is unit-testable without a Streamlit runtime (CLAUDE.md + design §7).
"""

from __future__ import annotations

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# The correct key column (D-25-ACTIVITY-TYPE). NOT Standard_Type.
ACTIVITY_TYPE_COL = "Activity_Type"


def activity_type_mask(
    df: pd.DataFrame, activity_type: str
) -> tuple[pd.Series, bool]:
    """Boolean mask selecting rows whose ``Activity_Type`` equals ``activity_type``.

    Returns ``(mask, available)``:
      - ``mask`` — a boolean ``pd.Series`` aligned to ``df.index``.
      - ``available`` — ``True`` when the ``Activity_Type`` column is present; ``False`` when
        absent (FAIL-OPEN: ``mask`` is then all-True so no member is silently dropped, and the
        caller can render an "Activity_Type unavailable" notice).

    Never raises; an empty frame yields an empty all-True mask with ``available`` reflecting
    whether the column exists.
    """
    if df is None:
        return pd.Series([], dtype=bool), False

    if ACTIVITY_TYPE_COL not in df.columns:
        logger.warning("activity_type_column_absent")
        # FAIL-OPEN: keep every row, signal unavailable.
        return pd.Series(True, index=df.index, dtype=bool), False

    mask = df[ACTIVITY_TYPE_COL] == activity_type
    return mask, True
