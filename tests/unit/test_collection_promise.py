"""Unit tests for the pure Promise-composite + radar-prep + band-gated verdict logic.

Co-located with frontend/ui/components/collection_promise.py. The module is
Streamlit-free pure-pandas logic over combined_activities.csv columns. It gates the
correctness-critical Promise math, radar data-prep, the §0.2 band-gated verdict sort,
and the 4-verb action mapping BEFORE Wave 2 wires them into collections.py.

⚠ IMP-SCALE NOTE: the fixture's ``IMP_Final_Score`` is on the 0–100 scale (BENZENE=60,
ASPIRIN=70). The band edges (0.30/0.50/0.70/0.90) and every band/verb literal are on the
0–1 scale, so the band/verb/verdict tests use INLINE 0–1 data, never the raw fixture IMP.
``select_radar_members`` is a pure ordering (scale-independent) so the fixture is fine there.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frontend.ui.components.collection_promise import (
    DEFAULT_PROMISE_WEIGHTS,
    compute_promise,
    imp_band_index,
    imp_band_verb,
    normalize_radar_axes,
    select_radar_members,
    verdict_sort_key,
)

FIXTURE = Path(__file__).parent / "fixtures" / "collection_toy_combined.csv"


@pytest.fixture
def toy_combined() -> pd.DataFrame:
    """The shared 25-01 real-schema toy combined_activities frame."""
    return pd.read_csv(FIXTURE)


# ---------------------------------------------------------------------------
# Task 1: Promise composite
# ---------------------------------------------------------------------------


def _promise_for(result: pd.DataFrame, member: str) -> float:
    """Pull the scalar Promise score for a member out of compute_promise's output."""
    return float(result.loc[member, "Promise"])


class TestPromiseWeights:
    def test_weights_frozen(self):
        # Sums to 1.0
        total = sum(w for _, w in DEFAULT_PROMISE_WEIGHTS)
        assert total == pytest.approx(1.0)
        # Immutable: a tuple-of-pairs (so it can go IN a cache key); not a mutable dict.
        assert isinstance(DEFAULT_PROMISE_WEIGHTS, tuple)
        with pytest.raises((TypeError, AttributeError)):
            DEFAULT_PROMISE_WEIGHTS[0] = ("x", 0.5)  # type: ignore[index]
        # compute output is bounded [0,100] for the full fixture.
        out = compute_promise(pd.read_csv(FIXTURE))
        finite = out["Promise"].dropna()
        assert (finite >= 0.0).all() and (finite <= 100.0).all()


class TestPromiseComposite:
    def test_all_five_components(self, toy_combined):
        # ASPIRIN: all 5 components present -> full unreduced weight vector (no reweight).
        # potency=(7.3-5)/4=0.575; ligeff=avg((25-10)/20,(18-5)/20)=avg(0.75,0.65)=0.70;
        # promiscuity=1 target->1.0; cleanliness=1-0.20=0.80; druglikeness=0.55
        # blend = .30*.575 + .25*.70 + .20*1.0 + .15*.80 + .10*.55 = 0.7225 -> 72.25
        out = compute_promise(toy_combined)
        assert _promise_for(out, "ASPIRIN") == pytest.approx(72.25, abs=1e-6)

    def test_druglikeness(self, toy_combined):
        # The real QED column (0-1) maps to the druglikeness component as-is (NOT QED_Impact).
        # If QED_Impact (8/6/4/2) were used, ASPIRIN's druglikeness would blow past 1.0 and
        # the 72.25 all-5 blend would not hold.
        out = compute_promise(toy_combined)
        assert _promise_for(out, "ASPIRIN") == pytest.approx(72.25, abs=1e-6)

    def test_potency_scale(self):
        # pActivity 5 -> 0.0 ; 9 -> 1.0 ; clipped outside.
        df = pd.DataFrame(
            {
                "compound_name": ["LO", "HI", "UNDER", "OVER"],
                "pActivity": [5.0, 9.0, 3.0, 11.0],
                "Target_Name": ["T", "T", "T", "T"],
            }
        )
        # Drop every other component so Promise == potency component (reweighted to 1.0
        # over the present {potency, promiscuity}). To isolate potency, give a constant
        # promiscuity (1 target each -> 1.0) and read the potency contribution directly.
        out = compute_promise(df)
        comp = out["potency"]
        assert comp.loc["LO"] == pytest.approx(0.0)
        assert comp.loc["HI"] == pytest.approx(1.0)
        assert comp.loc["UNDER"] == pytest.approx(0.0)  # clipped
        assert comp.loc["OVER"] == pytest.approx(1.0)  # clipped

    def test_selectivity_inverse(self):
        # distinct Target_Name count 1 -> 1.0 ; >=10 -> 0.0 ; clipped.
        rows = []
        for i in range(1):
            rows.append(("ONE", f"T{i}"))
        for i in range(10):
            rows.append(("TEN", f"T{i}"))
        df = pd.DataFrame(rows, columns=["compound_name", "Target_Name"])
        out = compute_promise(df)
        assert out.loc["ONE", "promiscuity"] == pytest.approx(1.0)
        assert out.loc["TEN", "promiscuity"] == pytest.approx(0.0)

    def test_cleanliness_matches_interference(self, toy_combined):
        # Cleanliness input == frame's Interference_Score 1:1 (before the 1- flip).
        # ETHANOL rows Interference_Score 0.20/0.40 -> mean 0.30 -> cleanliness = 1-0.30 = 0.70
        out = compute_promise(toy_combined)
        assert out.loc["ETHANOL", "cleanliness"] == pytest.approx(0.70, abs=1e-9)
        # BENZENE single row Interference_Score 0.60 -> cleanliness 0.40
        assert out.loc["BENZENE", "cleanliness"] == pytest.approx(0.40, abs=1e-9)

    def test_reweight(self):
        # A member missing ONE component -> that component is dropped and the remaining
        # weights renormalize to re-sum 1.0. Member with NO QED (druglikeness missing):
        # present {potency .30, ligeff .25, promiscuity .20, cleanliness .15} sum=0.90.
        df = pd.DataFrame(
            {
                "compound_name": ["X"],
                "pActivity": [7.0],  # potency=(7-5)/4=0.5
                "BEI": [20.0],  # (20-10)/20=0.5
                "SEI": [15.0],  # (15-5)/20=0.5 -> ligeff avg 0.5
                "Interference_Score": [0.2],  # cleanliness 0.8
                "Target_Name": ["T"],  # promiscuity 1.0
                # QED absent for this member -> druglikeness dropped
            }
        )
        out = compute_promise(df)
        # reweighted blend over the 4 present, weights renormalized by /0.90:
        # (.30*.5 + .25*.5 + .20*1.0 + .15*.8) / 0.90 = (.15+.125+.20+.12)/0.90
        # = 0.595/0.90 = 0.661111... -> *100 = 66.1111
        assert _promise_for(out, "X") == pytest.approx(0.595 / 0.90 * 100, abs=1e-6)

    def test_missing_qed_reweights(self, toy_combined):
        # BENZENE has the QED COLUMN present but a BLANK (NaN) value -> druglikeness DROPPED
        # for BENZENE and the remaining 4 weights reweight to re-sum 1.0. Promise finite, not NaN.
        # BENZENE: pActivity=6.3 -> potency=(6.3-5)/4=0.325; BEI=30->(30-10)/20=1.0,
        # SEI=15->(15-5)/20=0.5 -> ligeff avg 0.75; 1 target -> promiscuity 1.0;
        # Interference_Score 0.60 -> cleanliness 0.40. Present weights sum 0.90.
        out = compute_promise(toy_combined)
        b = _promise_for(out, "BENZENE")
        assert math.isfinite(b)
        expected = (0.30 * 0.325 + 0.25 * 0.75 + 0.20 * 1.0 + 0.15 * 0.40) / 0.90 * 100
        assert b == pytest.approx(expected, abs=1e-6)

    def test_all_missing_is_nan(self):
        # A member with ALL components missing -> Promise == NaN (insufficient data), never 0.
        df = pd.DataFrame({"compound_name": ["EMPTY"]})
        out = compute_promise(df)
        assert np.isnan(out.loc["EMPTY", "Promise"])


# ---------------------------------------------------------------------------
# Task 2: radar-prep + band-gated verdict sort
# ---------------------------------------------------------------------------

RADAR_AXES = (
    "Efficiency_Score",
    "Distance_Score",
    "Angle_Score",
    "Interference_Score",
    "PDB_Score",
)


def _radar_frame(n: int) -> pd.DataFrame:
    """n members with descending IMP and arbitrary axis values."""
    return pd.DataFrame(
        {
            "compound_name": [f"M{i}" for i in range(n)],
            "IMP_Final_Score": [float(n - i) for i in range(n)],  # M0 highest
            **{ax: [float(i + 1) for i in range(n)] for ax in RADAR_AXES},
        }
    )


class TestRadarSelection:
    def test_radar_selection_top5_by_imp(self):
        df = _radar_frame(8)
        sel = select_radar_members(df, n=5)
        assert len(sel) == 5
        # Top 5 by IMP DESCENDING: M0..M4 (IMP 8..4).
        assert list(sel["compound_name"]) == ["M0", "M1", "M2", "M3", "M4"]

    def test_radar_selection_edge_counts(self):
        # 0 members -> empty frame, no crash.
        empty = select_radar_members(_radar_frame(0), n=5)
        assert empty.empty
        # 1, 3 -> all of them (capped at available, no padding).
        assert len(select_radar_members(_radar_frame(1), n=5)) == 1
        assert len(select_radar_members(_radar_frame(3), n=5)) == 3
        # exactly 5 -> all 5.
        assert len(select_radar_members(_radar_frame(5), n=5)) == 5
        # >5 -> exactly 5.
        assert len(select_radar_members(_radar_frame(9), n=5)) == 5
        # TIES in IMP -> stable/deterministic (preserve input order).
        tied = pd.DataFrame(
            {
                "compound_name": ["A", "B", "C", "D"],
                "IMP_Final_Score": [5.0, 5.0, 5.0, 5.0],
                **{ax: [1.0, 2.0, 3.0, 4.0] for ax in RADAR_AXES},
            }
        )
        sel = select_radar_members(tied, n=3)
        assert list(sel["compound_name"]) == ["A", "B", "C"]
        # MISSING IMP_Final_Score column -> graceful, returns available in input order.
        no_imp = pd.DataFrame(
            {
                "compound_name": ["P", "Q"],
                **{ax: [1.0, 2.0] for ax in RADAR_AXES},
            }
        )
        sel2 = select_radar_members(no_imp, n=5)
        assert list(sel2["compound_name"]) == ["P", "Q"]


class TestRadarNormalize:
    def test_radar_normalize_0_1(self):
        df = _radar_frame(4)  # each axis = [1,2,3,4]
        out = normalize_radar_axes(df, axes=RADAR_AXES)
        for ax in RADAR_AXES:
            col = out[ax]
            assert col.min() == pytest.approx(0.0)
            assert col.max() == pytest.approx(1.0)
            # min->0, max->1 explicitly
            assert col.iloc[0] == pytest.approx(0.0)
            assert col.iloc[-1] == pytest.approx(1.0)

    def test_radar_normalize_constant_axis(self):
        # Constant axis (no spread) AND all-NaN axis -> every member maps to 0.5 (neutral).
        df = pd.DataFrame(
            {
                "compound_name": ["A", "B", "C"],
                "Efficiency_Score": [7.0, 7.0, 7.0],  # constant
                "Distance_Score": [np.nan, np.nan, np.nan],  # all-NaN
                "Angle_Score": [0.0, 1.0, 2.0],
                "Interference_Score": [1.0, 1.0, 1.0],  # constant
                "PDB_Score": [3.0, 4.0, 5.0],
            }
        )
        out = normalize_radar_axes(df, axes=RADAR_AXES)
        assert (out["Efficiency_Score"] == 0.5).all()
        assert (out["Distance_Score"] == 0.5).all()
        assert (out["Interference_Score"] == 0.5).all()
        # No NaN / inf anywhere.
        for ax in RADAR_AXES:
            assert out[ax].notna().all()
            assert np.isfinite(out[ax]).all()

    def test_radar_has_exactly_5_axes(self):
        df = _radar_frame(3)
        df["QED"] = [0.5, 0.6, 0.7]
        df["QED_Impact"] = [8.0, 6.0, 4.0]
        out = normalize_radar_axes(df, axes=RADAR_AXES)
        axis_cols = [c for c in out.columns if c != "compound_name"]
        assert set(axis_cols) == set(RADAR_AXES)
        assert len(set(axis_cols)) == 5
        assert "QED" not in axis_cols
        assert "QED_Impact" not in axis_cols


class TestVerdictSort:
    def test_verdict_sort_band_then_promise(self):
        # IMP on the 0-1 scale. A high-Promise + high-IMP member (IMP 0.80, Promise 95)
        # must NOT sort to the top; it lands in the VALIDATE band [0.70,0.90), below
        # lower-risk members regardless of Promise.
        df = pd.DataFrame(
            {
                "compound_name": ["GENUINE", "TOO_GOOD", "MID"],
                "IMP_Final_Score": [0.10, 0.80, 0.45],
                "Promise": [40.0, 95.0, 60.0],
            }
        )
        sorted_df = verdict_sort_key(df)
        order = list(sorted_df["compound_name"])
        # band asc-risk first: GENUINE (band0) < MID (band1) < TOO_GOOD (band3)
        assert order == ["GENUINE", "MID", "TOO_GOOD"]
        # TOO_GOOD never at the top despite Promise 95.
        assert order[0] != "TOO_GOOD"

    def test_band_boundary_inclusivity(self):
        # LEFT-inclusive/right-EXCLUSIVE: IMP==0.30 -> band [0.30,0.50) -> MONITOR;
        # IMP==0.70 -> band [0.70,0.90) -> VALIDATE.
        assert imp_band_index(0.30) == 1
        assert imp_band_index(0.70) == 3
        assert imp_band_verb(0.30) == "MONITOR"
        assert imp_band_verb(0.70) == "VALIDATE"

    def test_verb_helper_output_set(self):
        # Across the full IMP range, the verb helper emits ONLY the 4 action verbs.
        vals = [i / 100.0 for i in range(0, 101)]
        verbs = {imp_band_verb(v) for v in vals}
        assert verbs == {"PRIORITIZE", "MONITOR", "VALIDATE", "DEPRIORITIZE"}
        # No PROCEED, no severity nouns.
        assert "PROCEED" not in verbs
        for noun in ("Weak", "Moderate", "Strong", "Exceptional", "Not-IMP"):
            assert noun not in verbs
