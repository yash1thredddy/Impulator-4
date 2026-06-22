"""Unit tests for the Stage-2 aggregate-artifact reader (no Streamlit, no IO).

Exercises the pure parse/normalize/filter functions over a fixture that mirrors
the ``collection_aggregate.json`` schema written by plan 24-04: a dict keyed by
member ``entry_id`` -> {indications, pdb, all_similar, classification}.
"""

from frontend.ui.components.collection_aggregate import (
    AggregateEntry,
    filter_by_members,
    parse_aggregate,
)

# Schema per D-S2-ARCH: dict keyed by entry_id.
SAMPLE = {
    "m1": {
        "indications": [{"name": "hypertension", "phase": 4}],
        "pdb": [{"pdb_id": "1ABC", "resolution": 1.8}],
        "all_similar": [{"chembl_id": "CHEMBL1", "similarity": 0.92}],
        "classification": {"superclass": "Organic acids"},
    },
    "m2": {
        "indications": [],
        "pdb": [],
        "all_similar": [{"chembl_id": "CHEMBL2", "similarity": 0.71}],
        "classification": {"superclass": "Benzenoids"},
    },
}


class TestParseAggregate:
    def test_empty_dict_returns_empty_without_crash(self):
        result = parse_aggregate({})
        assert result == {}

    def test_none_returns_empty_without_crash(self):
        result = parse_aggregate(None)
        assert result == {}

    def test_non_dict_input_returns_empty(self):
        # A malformed/partial artifact (e.g. a list) must not crash.
        assert parse_aggregate([1, 2, 3]) == {}
        assert parse_aggregate("garbage") == {}

    def test_parses_each_member_into_entry(self):
        result = parse_aggregate(SAMPLE)
        assert set(result.keys()) == {"m1", "m2"}
        assert isinstance(result["m1"], AggregateEntry)

    def test_entry_fields_normalized(self):
        result = parse_aggregate(SAMPLE)
        m1 = result["m1"]
        assert m1.entry_id == "m1"
        assert m1.indications == [{"name": "hypertension", "phase": 4}]
        assert m1.pdb == [{"pdb_id": "1ABC", "resolution": 1.8}]
        assert m1.all_similar == [{"chembl_id": "CHEMBL1", "similarity": 0.92}]
        assert m1.classification == {"superclass": "Organic acids"}

    def test_missing_keys_default_to_empty_containers(self):
        # A member with a partial record still parses; absent dimensions default.
        partial = {"m3": {"indications": [{"name": "x"}]}}
        result = parse_aggregate(partial)
        m3 = result["m3"]
        assert m3.indications == [{"name": "x"}]
        assert m3.pdb == []
        assert m3.all_similar == []
        assert m3.classification == {}

    def test_malformed_member_record_skipped_not_crash(self):
        # A member whose record is not a dict is skipped gracefully.
        mixed = {"good": {"indications": []}, "bad": "not-a-record"}
        result = parse_aggregate(mixed)
        assert "good" in result
        assert "bad" not in result


class TestFilterByMembers:
    def test_subset_selection(self):
        parsed = parse_aggregate(SAMPLE)
        subset = filter_by_members(parsed, ["m1"])
        assert set(subset.keys()) == {"m1"}

    def test_empty_selection_returns_empty(self):
        parsed = parse_aggregate(SAMPLE)
        assert filter_by_members(parsed, []) == {}

    def test_unknown_member_ignored(self):
        parsed = parse_aggregate(SAMPLE)
        subset = filter_by_members(parsed, ["m1", "does-not-exist"])
        assert set(subset.keys()) == {"m1"}

    def test_filter_on_empty_parsed_returns_empty(self):
        assert filter_by_members({}, ["m1"]) == {}
