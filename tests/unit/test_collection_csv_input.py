"""Tests for the Collection CSV-input member builder (Phase 23 gap closure).

Covers ``frontend.ui.pages.analyze._build_members_from_mapped`` — the pure core
that turns a mapped CSV dataframe into stamped collection-member dicts.

The load-bearing assertion is :func:`test_stamps_shared_config_onto_every_member`:
collection members MUST carry the chosen ``similarity_threshold`` /
``activity_types`` per-member, because the backend reads them per-member
(``process_collection_member`` does ``int(member_input.get("similarity_threshold", 90))``
and a stored ``None`` would crash every member). This test is the regression guard
for that bug.
"""

import pandas as pd

from frontend.ui.pages.analyze import _build_members_from_mapped


def test_stamps_shared_config_onto_every_member():
    """Every built member carries the shared threshold + activity types (bug guard)."""
    df = pd.DataFrame(
        {
            "compound_name": ["Quercetin", "Kaempferol"],
            "smiles": [
                "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
                "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
            ],
        }
    )

    members, _report = _build_members_from_mapped(
        df,
        similarity_threshold=75,
        activity_types=["IC50", "Ki"],
    )

    assert len(members) == 2
    for member in members:
        assert member["similarity_threshold"] == 75
        assert member["activity_types"] == ["IC50", "Ki"]


def test_uses_smiles_column_directly():
    """A SMILES column is used verbatim as the member structure."""
    df = pd.DataFrame(
        {"compound_name": ["Ethanol"], "smiles": ["CCO"]}
    )

    members, report = _build_members_from_mapped(
        df, similarity_threshold=90, activity_types=["IC50"]
    )

    assert members[0]["name"] == "Ethanol"
    assert members[0]["smiles"] == "CCO"
    assert report["skipped_no_structure"] == []


def test_resolves_inchi_to_smiles():
    """An InChI-only row is converted to canonical SMILES via RDKit."""
    df = pd.DataFrame(
        {
            "compound_name": ["Ethanol"],
            "inchi": ["InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"],
        }
    )

    members, report = _build_members_from_mapped(
        df, similarity_threshold=90, activity_types=["IC50"]
    )

    assert len(members) == 1
    assert members[0]["smiles"] == "CCO"
    assert report["skipped_no_structure"] == []


def test_resolves_inchikey_via_provided_map():
    """An InChIKey-only row is resolved through the caller-supplied map."""
    df = pd.DataFrame(
        {
            "compound_name": ["Ethanol"],
            "inchikey": ["LFQSCWFLJHTTHZ-UHFFFAOYSA-N"],
        }
    )

    members, report = _build_members_from_mapped(
        df,
        similarity_threshold=90,
        activity_types=["IC50"],
        inchikey_smiles_map={"LFQSCWFLJHTTHZ-UHFFFAOYSA-N": "CCO"},
    )

    assert len(members) == 1
    assert members[0]["smiles"] == "CCO"


def test_skips_rows_with_no_resolvable_structure():
    """Rows whose structure cannot be resolved are skipped and reported."""
    df = pd.DataFrame(
        {
            "compound_name": ["Good", "NoStructure"],
            "smiles": ["CCO", ""],
        }
    )

    members, report = _build_members_from_mapped(
        df, similarity_threshold=90, activity_types=["IC50"]
    )

    assert [m["name"] for m in members] == ["Good"]
    assert report["skipped_no_structure"] == ["NoStructure"]


def test_skips_nan_compound_name_rows():
    """A trailing all-empty CSV row (NaN name + NaN smiles) is excluded."""
    import numpy as np

    df = pd.DataFrame(
        {
            "compound_name": ["Real", np.nan],
            "smiles": ["CCO", np.nan],
        }
    )

    members, report = _build_members_from_mapped(
        df, similarity_threshold=90, activity_types=["IC50"]
    )

    assert [m["name"] for m in members] == ["Real"]
    # The NaN-name row is dropped outright (no name), not flagged as bad data.
    assert report["invalid_names"] == []
    assert report["skipped_no_structure"] == []


def test_skips_blank_compound_names():
    """Rows with a blank/NaN compound name are dropped entirely."""
    df = pd.DataFrame(
        {
            "compound_name": ["Real", "  "],
            "smiles": ["CCO", "CCC"],
        }
    )

    members, _report = _build_members_from_mapped(
        df, similarity_threshold=90, activity_types=["IC50"]
    )

    assert [m["name"] for m in members] == ["Real"]


def test_reports_rows_with_invalid_member_names():
    """Names outside COMPOUND_NAME_PATTERN are reported and excluded (not 422'd)."""
    df = pd.DataFrame(
        {
            "compound_name": ["Quercetin", "α-Pinene", "Bad/Name"],
            "smiles": ["CCO", "CCC", "CCCC"],
        }
    )

    members, report = _build_members_from_mapped(
        df, similarity_threshold=90, activity_types=["IC50"]
    )

    assert [m["name"] for m in members] == ["Quercetin"]
    assert set(report["invalid_names"]) == {"α-Pinene", "Bad/Name"}


def test_allows_names_with_permitted_punctuation():
    """Parens, brackets, hyphen, comma, apostrophe, dot, space are all valid."""
    df = pd.DataFrame(
        {
            "compound_name": ["Compound (A)", "Drug-1', [B]"],
            "smiles": ["CCO", "CCC"],
        }
    )

    members, report = _build_members_from_mapped(
        df, similarity_threshold=90, activity_types=["IC50"]
    )

    assert len(members) == 2
    assert report["invalid_names"] == []


def test_smiles_takes_priority_over_inchi_and_inchikey():
    """When multiple structure columns exist, SMILES wins (no resolution needed)."""
    df = pd.DataFrame(
        {
            "compound_name": ["Ethanol"],
            "smiles": ["CCO"],
            "inchi": ["InChI=1S/C3H8/c1-3-2/h3H2,1-2H3"],
            "inchikey": ["ATUOYWHBWRKTHZ-UHFFFAOYSA-N"],
        }
    )

    members, _report = _build_members_from_mapped(
        df,
        similarity_threshold=90,
        activity_types=["IC50"],
        inchikey_smiles_map={"ATUOYWHBWRKTHZ-UHFFFAOYSA-N": "CCC"},
    )

    assert members[0]["smiles"] == "CCO"
