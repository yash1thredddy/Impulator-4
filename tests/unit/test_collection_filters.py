"""Unit tests for the pure Activity_Type filter predicate (no Streamlit).

Co-located with frontend/ui/components/collection_filters.py. The predicate keys off
``Activity_Type`` (NOT ``Standard_Type`` — wrong key silently re-pools assay types, the
exact bug this redesign kills; D-25-ACTIVITY-TYPE / §0.3). Absent ``Activity_Type`` column
→ FAIL-OPEN (all-True mask + an unavailable signal), never fail-closed to an empty page.
"""

from pathlib import Path

import pandas as pd

from frontend.ui.components.collection_filters import activity_type_mask

FIXTURE = Path(__file__).parent / "fixtures" / "collection_toy_combined.csv"


def test_activity_type_filter():
    # The fixture has a discriminating row: ETHANOL/TargetB has Standard_Type=Ki but
    # Activity_Type=IC50. Masking on Activity_Type vs Standard_Type yields DIFFERENT sets.
    df = pd.read_csv(FIXTURE)

    mask, available = activity_type_mask(df, "IC50")
    assert available is True

    # Activity_Type=="IC50" keeps ETHANOL (both rows), BENZENE, ASPIRIN; excludes CAFFEINE.
    kept = set(df.loc[mask, "compound_name"])
    excluded = set(df.loc[~mask, "compound_name"])
    assert "CAFFEINE" not in kept  # CAFFEINE's Activity_Type is Ki
    assert "CAFFEINE" in excluded

    # Discriminating: the Activity_Type mask differs from a Standard_Type mask.
    std_mask = df["Standard_Type"] == "IC50"
    assert not mask.equals(std_mask)
    # Specifically the ETHANOL/TargetB row: Activity_Type=IC50 (kept) but Standard_Type=Ki.
    eth_b = (df["compound_name"] == "ETHANOL") & (df["Target_Name"] == "TargetB")
    assert bool(mask[eth_b].iloc[0]) is True  # included by Activity_Type
    assert bool(std_mask[eth_b].iloc[0]) is False  # would be excluded by Standard_Type


def test_activity_type_absent_column_fail_open():
    # Activity_Type column ABSENT -> all-True mask (keep every member) + unavailable signal.
    df = pd.DataFrame(
        {
            "compound_name": ["A", "B", "C"],
            "Standard_Type": ["IC50", "Ki", "IC50"],
        }
    )
    mask, available = activity_type_mask(df, "IC50")
    assert available is False  # caller can render "Activity_Type unavailable"
    assert mask.all()  # fail-open: keep everyone, never blank the page
    assert len(mask) == len(df)
