"""Unit tests for the pure collection pre-flight logic (no Streamlit)."""

from frontend.ui.components.collection_preflight import (
    apply_preflight_decisions,
    build_preflight_plan,
    compute_inchikey,
    distinct_thresholds,
    group_in_file_duplicates,
)


class TestComputeInchikey:
    def test_valid_smiles_returns_key(self):
        # Ethanol
        key = compute_inchikey("CCO")
        assert key == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"

    def test_invalid_smiles_returns_none(self):
        assert compute_inchikey("not-a-smiles") is None

    def test_empty_returns_none(self):
        assert compute_inchikey("") is None


class TestGroupInFileDuplicates:
    def test_no_duplicates_returns_empty(self):
        members = [
            {"name": "A", "smiles": "CCO"},
            {"name": "B", "smiles": "c1ccccc1"},
        ]
        assert group_in_file_duplicates(members) == []

    def test_one_group_of_two(self):
        members = [
            {"name": "A", "smiles": "CCO"},
            {"name": "B", "smiles": "OCC"},   # same structure as ethanol
        ]
        groups = group_in_file_duplicates(members)
        assert len(groups) == 1
        g = groups[0]
        assert g.member_indices == [0, 1]
        assert g.names == ["A", "B"]
        assert g.inchikey == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"

    def test_unparseable_smiles_not_grouped(self):
        members = [
            {"name": "A", "smiles": "CCO"},
            {"name": "bad", "smiles": "xxx"},
            {"name": "B", "smiles": "OCC"},
        ]
        groups = group_in_file_duplicates(members)
        assert len(groups) == 1
        assert groups[0].member_indices == [0, 2]

    def test_group_order_is_input_order(self):
        members = [
            {"name": "A", "smiles": "CCO"},
            {"name": "X", "smiles": "c1ccccc1"},
            {"name": "Y", "smiles": "c1ccccc1"},
            {"name": "B", "smiles": "OCC"},
        ]
        groups = group_in_file_duplicates(members)
        # ethanol group appears first (index 0), benzene group second (index 1)
        assert [g.member_indices for g in groups] == [[0, 3], [1, 2]]


def _avail(name, smiles, *, available, has_any_data=True, thresholds=None):
    return {
        "compound_name": name,
        "smiles": smiles,
        "available": available,
        "has_any_data": has_any_data,
        "thresholds": thresholds or [],
    }


class TestBuildPreflightPlan:
    def test_ready_member(self):
        members = [{"name": "A", "smiles": "CCO", "similarity_threshold": 90}]
        results = [_avail("A", "CCO", available=True,
                          thresholds=[{"threshold": 90, "count": 5}])]
        plan = build_preflight_plan(members, results, 90)
        assert plan.ready_count == 1
        assert plan.members[0].status == "ready"
        assert plan.members[0].suggested_threshold == 90

    def test_needs_lower_picks_lowest_tier_with_data(self):
        members = [{"name": "B", "smiles": "c1ccccc1", "similarity_threshold": 90}]
        results = [_avail("B", "c1ccccc1", available=False, thresholds=[
            {"threshold": 90, "count": 0},
            {"threshold": 70, "count": 0},
            {"threshold": 50, "count": 3},
            {"threshold": 40, "count": 29},
        ])]
        plan = build_preflight_plan(members, results, 90)
        m = plan.members[0]
        assert m.status == "needs_lower"
        assert plan.needs_lower_count == 1
        assert m.suggested_threshold == 40           # lowest with data
        assert m.tiers == [                            # count>0, descending
            {"threshold": 50, "count": 3},
            {"threshold": 40, "count": 29},
        ]

    def test_no_data_member(self):
        members = [{"name": "C", "smiles": "CCO", "similarity_threshold": 90}]
        results = [_avail("C", "CCO", available=False, has_any_data=False,
                          thresholds=[{"threshold": 90, "count": 0}])]
        plan = build_preflight_plan(members, results, 90)
        assert plan.no_data_count == 1
        assert plan.members[0].status == "no_data"
        assert plan.members[0].suggested_threshold is None

    def test_member_missing_from_results_is_unknown(self):
        members = [{"name": "Z", "smiles": "CCO", "similarity_threshold": 90}]
        plan = build_preflight_plan(members, [], 90)
        assert plan.members[0].status == "unknown"
        assert plan.members[0].suggested_threshold is None

    def test_dup_groups_included(self):
        members = [
            {"name": "A", "smiles": "CCO", "similarity_threshold": 90},
            {"name": "B", "smiles": "OCC", "similarity_threshold": 90},
        ]
        results = [
            _avail("A", "CCO", available=True, thresholds=[{"threshold": 90, "count": 5}]),
            _avail("B", "OCC", available=True, thresholds=[{"threshold": 90, "count": 5}]),
        ]
        plan = build_preflight_plan(members, results, 90)
        assert len(plan.dup_groups) == 1
        assert plan.dup_groups[0].member_indices == [0, 1]


class TestApplyPreflightDecisions:
    def _members(self):
        return [
            {"name": "A", "smiles": "CCO", "similarity_threshold": 90, "activity_types": ["IC50"]},
            {"name": "B", "smiles": "OCC", "similarity_threshold": 90, "activity_types": ["IC50"]},  # dup of A (idx 1)
            {"name": "C", "smiles": "c1ccccc1", "similarity_threshold": 90, "activity_types": ["IC50"]},  # idx 2
        ]

    def test_keep_first_drops_other_group_members(self):
        out = apply_preflight_decisions(
            self._members(),
            dup_decisions={"LFQSCWFLJHTTHZ-UHFFFAOYSA-N": "first"},
            threshold_decisions={},
            excluded_indices=set(),
        )
        assert [m["name"] for m in out] == ["A", "C"]  # idx 1 (B) dropped

    def test_keep_both_retains_all(self):
        out = apply_preflight_decisions(
            self._members(),
            dup_decisions={"LFQSCWFLJHTTHZ-UHFFFAOYSA-N": "both"},
            threshold_decisions={},
            excluded_indices=set(),
        )
        assert [m["name"] for m in out] == ["A", "B", "C"]

    def test_threshold_decisions_stamped_by_index(self):
        out = apply_preflight_decisions(
            self._members(),
            dup_decisions={"LFQSCWFLJHTTHZ-UHFFFAOYSA-N": "both"},
            threshold_decisions={2: 50},   # index 2 == "C"
            excluded_indices=set(),
        )
        by_name = {m["name"]: m for m in out}
        assert by_name["C"]["similarity_threshold"] == 50
        assert by_name["A"]["similarity_threshold"] == 90  # untouched
        assert by_name["C"]["activity_types"] == ["IC50"]  # preserved

    def test_excluded_indices_dropped(self):
        out = apply_preflight_decisions(
            self._members(),
            dup_decisions={"LFQSCWFLJHTTHZ-UHFFFAOYSA-N": "both"},
            threshold_decisions={},
            excluded_indices={2},   # drop "C"
        )
        assert [m["name"] for m in out] == ["A", "B"]

    def test_same_name_members_do_not_collide(self):
        # Two members named "Dup" with DIFFERENT structures; a per-index
        # threshold decision must affect only one of them.
        members = [
            {"name": "Dup", "smiles": "CCO", "similarity_threshold": 90, "activity_types": ["IC50"]},
            {"name": "Dup", "smiles": "c1ccccc1", "similarity_threshold": 90, "activity_types": ["IC50"]},
        ]
        out = apply_preflight_decisions(
            members, dup_decisions={}, threshold_decisions={1: 50}, excluded_indices=set()
        )
        assert out[0]["similarity_threshold"] == 90
        assert out[1]["similarity_threshold"] == 50

    def test_default_decision_is_keep_first(self):
        out = apply_preflight_decisions(
            self._members(),
            dup_decisions={},
            threshold_decisions={},
            excluded_indices=set(),
        )
        assert [m["name"] for m in out] == ["A", "C"]


class TestDistinctThresholds:
    def test_single_threshold(self):
        members = [
            {"name": "A", "similarity_threshold": 90},
            {"name": "B", "similarity_threshold": 90},
        ]
        assert distinct_thresholds(members) == [90]

    def test_mixed_thresholds_sorted_desc(self):
        members = [
            {"name": "A", "similarity_threshold": 90},
            {"name": "B", "similarity_threshold": 50},
            {"name": "C", "similarity_threshold": 90},
        ]
        assert distinct_thresholds(members) == [90, 50]
