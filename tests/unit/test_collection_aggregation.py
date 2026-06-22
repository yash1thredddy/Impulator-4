"""Unit tests for the pure collection Stage-1 aggregation logic (no Streamlit).

Co-located with frontend/ui/components/collection_aggregation.py. The module is a
Streamlit-free pure-pandas groupby aggregator over combined_activities.csv columns
— it NEVER recomputes IMP components (RESEARCH anti-pattern: they are already columns).
"""

from pathlib import Path

import pandas as pd
import pytest

from frontend.ui.components.collection_aggregation import (
    COMPONENT_COLS,
    druglikeness_flag_counts,
    member_efficiency_stats,
    per_member_components,
    promiscuity,
)

FIXTURE = Path(__file__).parent / "fixtures" / "collection_toy_combined.csv"


@pytest.fixture
def toy_combined() -> pd.DataFrame:
    """The shared 24-01 toy combined_activities frame."""
    return pd.read_csv(FIXTURE)


class TestComponentCols:
    def test_component_cols_are_the_six_contributions(self):
        assert COMPONENT_COLS == [
            "Efficiency_Contribution",
            "Distance_Contribution",
            "Angle_Contribution",
            "Interference_Contribution",
            "PDB_Contribution",
            "QED_Impact",
        ]


class TestPerMemberComponents:
    def test_index_is_compound_name(self, toy_combined):
        out = per_member_components(toy_combined)
        assert out.index.name == "compound_name"
        assert set(out.index) == {"ETHANOL", "BENZENE", "ASPIRIN", "CAFFEINE"}

    def test_columns_include_components_and_final_score(self, toy_combined):
        out = per_member_components(toy_combined)
        for col in COMPONENT_COLS:
            assert col in out.columns
        assert "IMP_Final_Score" in out.columns

    def test_means_match_hand_computed_groupby(self, toy_combined):
        out = per_member_components(toy_combined)
        # ETHANOL has 2 rows: IMP_Final_Score (40,50) -> mean 45.0
        assert out.loc["ETHANOL", "IMP_Final_Score"] == pytest.approx(45.0)
        # Efficiency_Contribution (12,16) -> mean 14.0
        assert out.loc["ETHANOL", "Efficiency_Contribution"] == pytest.approx(14.0)
        # QED_Impact (8,6) -> mean 7.0
        assert out.loc["ETHANOL", "QED_Impact"] == pytest.approx(7.0)
        # Single-row members equal their row value
        assert out.loc["BENZENE", "IMP_Final_Score"] == pytest.approx(60.0)
        assert out.loc["ASPIRIN", "IMP_Final_Score"] == pytest.approx(70.0)
        assert out.loc["CAFFEINE", "IMP_Final_Score"] == pytest.approx(30.0)

    def test_missing_columns_omitted_not_crash(self):
        # Only one component column present + the score; others absent
        df = pd.DataFrame(
            {
                "compound_name": ["A", "A", "B"],
                "Efficiency_Contribution": [10.0, 20.0, 5.0],
                "IMP_Final_Score": [1.0, 3.0, 9.0],
            }
        )
        out = per_member_components(df)
        assert "Efficiency_Contribution" in out.columns
        assert "IMP_Final_Score" in out.columns
        # Absent component columns are simply not present (no crash, no NaN columns)
        assert "PDB_Contribution" not in out.columns
        assert out.loc["A", "Efficiency_Contribution"] == pytest.approx(15.0)


class TestMemberEfficiencyStats:
    def test_bei_sei_mean_and_max(self, toy_combined):
        out = member_efficiency_stats(toy_combined)
        # ETHANOL BEI (18,22) -> mean 20.0, max 22.0 ; SEI (9,11) -> mean 10.0, max 11.0
        assert out.loc["ETHANOL", "BEI_mean"] == pytest.approx(20.0)
        assert out.loc["ETHANOL", "BEI_max"] == pytest.approx(22.0)
        assert out.loc["ETHANOL", "SEI_mean"] == pytest.approx(10.0)
        assert out.loc["ETHANOL", "SEI_max"] == pytest.approx(11.0)
        assert out.index.name == "compound_name"

    def test_missing_efficiency_columns_omitted(self):
        df = pd.DataFrame(
            {"compound_name": ["A", "B"], "BEI": [1.0, 2.0]}
        )  # no SEI column
        out = member_efficiency_stats(df)
        assert "BEI_mean" in out.columns
        assert "SEI_mean" not in out.columns


class TestDruglikenessFlagCounts:
    def test_counts_present_flag_columns(self):
        # Inline frame WITH flag columns (toy fixture has none — only its guard is testable there)
        df = pd.DataFrame(
            {
                "compound_name": ["A", "A", "B"],
                "PAINS_Violation": [True, False, True],
                "Aggregator_Risk": [False, False, True],
                "RO5_Violations": [0, 2, 1],
            }
        )
        out = druglikeness_flag_counts(df)
        # A: 1 PAINS row flagged, 0 Aggregator, 1 row with RO5 violation (>0)
        assert out.loc["A", "PAINS_Violation"] == 1
        assert out.loc["A", "Aggregator_Risk"] == 0
        assert out.loc["A", "RO5_Violations"] == 1
        # B: 1 PAINS, 1 Aggregator, 1 RO5-violating row
        assert out.loc["B", "PAINS_Violation"] == 1
        assert out.loc["B", "Aggregator_Risk"] == 1
        assert out.loc["B", "RO5_Violations"] == 1

    def test_no_flag_columns_returns_empty_no_crash(self):
        # Inline frame that deliberately OMITS all flag columns — guard must omit,
        # not crash. (The shared fixture now carries flag columns, so this branch
        # is exercised via an explicit no-column frame instead — mirrors the
        # inline-frame style of test_counts_present_flag_columns above.)
        df = pd.DataFrame(
            {
                "compound_name": ["A", "A", "B"],
                "IMP_Final_Score": [1.0, 3.0, 9.0],
            }
        )
        out = druglikeness_flag_counts(df)
        assert out.empty or out.shape[1] == 0


class TestPromiscuity:
    def test_distinct_target_count_per_member(self, toy_combined):
        out = promiscuity(toy_combined)
        # ETHANOL hits TargetA + TargetB -> 2 distinct; others -> 1
        assert out.loc["ETHANOL"] == 2
        assert out.loc["BENZENE"] == 1
        assert out.loc["ASPIRIN"] == 1
        assert out.loc["CAFFEINE"] == 1
        assert out.index.name == "compound_name"

    def test_missing_target_column_returns_empty(self):
        df = pd.DataFrame({"compound_name": ["A", "B"]})  # no Target_Name
        out = promiscuity(df)
        assert out.empty
