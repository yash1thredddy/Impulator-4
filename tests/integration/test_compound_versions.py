"""
Integration tests for compound versions endpoint.

Tests GET /api/v1/compounds/{entry_id}/versions which finds all structural
siblings (same InChIKey structure key) for a compound.
"""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy.orm import sessionmaker


def _make_compound(session, **overrides):
    """Helper to create a Compound in the database.

    Returns a SimpleNamespace (not ORM object) to avoid DetachedInstanceError
    when the session is closed after creation.

    Translates legacy is_duplicate/duplicate_of to parent_id/version.
    """
    from backend.models.compound import Compound
    from backend.services.job_service import _inchikey_structure_key

    # Extract and translate legacy fields
    is_duplicate = overrides.pop("is_duplicate", False)
    duplicate_of = overrides.pop("duplicate_of", None)

    defaults = {
        "entry_id": str(uuid.uuid4()),
        "compound_name": "TestCompound",
        "smiles": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "similarity_threshold": 90,
        "activity_types": ["IC50"],
        "processed_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)

    # Auto-compute inchikey_structure_key if not provided
    if "inchikey_structure_key" not in defaults and defaults.get("inchikey"):
        defaults["inchikey_structure_key"] = _inchikey_structure_key(defaults["inchikey"])

    # Translate legacy fields
    if is_duplicate and duplicate_of:
        try:
            defaults["parent_id"] = uuid.UUID(str(duplicate_of))
        except (ValueError, AttributeError):
            pass
        defaults.setdefault("version", 2)

    comp = Compound(**defaults)
    session.add(comp)
    session.commit()

    return SimpleNamespace(
        entry_id=defaults["entry_id"],
        compound_name=defaults["compound_name"],
        inchikey=defaults["inchikey"],
    )


class TestCompoundVersionsEndpoint:
    """Tests for GET /api/v1/compounds/{entry_id}/versions."""

    def test_nonexistent_compound_returns_404(self, client):
        """Requesting versions for a nonexistent compound returns 404."""
        response = client.get("/api/v1/compounds/00000000-0000-4000-8000-000000000008/versions")
        assert response.status_code == 404

    def test_single_compound_returns_empty_versions(self, test_engine, client):
        """A compound with no siblings returns empty versions list."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        comp = _make_compound(session)
        session.close()

        response = client.get(f"/api/v1/compounds/{comp.entry_id}/versions")
        assert response.status_code == 200
        data = response.json()
        assert data["versions"] == []
        assert data["current_entry_id"] == comp.entry_id

    def test_compound_without_inchikey_returns_empty(self, test_engine, client):
        """A compound with no InChIKey returns empty versions."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        comp = _make_compound(session, inchikey=None)
        session.close()

        response = client.get(f"/api/v1/compounds/{comp.entry_id}/versions")
        assert response.status_code == 200
        data = response.json()
        assert data["versions"] == []

    def test_two_siblings_same_inchikey(self, test_engine, client):
        """Two compounds with same InChIKey structure key appear as versions."""
        Session = sessionmaker(bind=test_engine)
        session = Session()

        # Same InChIKey structure key (first two blocks match)
        inchikey = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        comp1 = _make_compound(
            session,
            compound_name="Ethanol_v1",
            inchikey=inchikey,
            similarity_threshold=90,
            processed_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        comp2 = _make_compound(
            session,
            compound_name="Ethanol_v2",
            inchikey=inchikey,
            similarity_threshold=70,
            processed_at=datetime.now(timezone.utc),
        )
        session.close()

        response = client.get(f"/api/v1/compounds/{comp1.entry_id}/versions")
        assert response.status_code == 200
        data = response.json()

        assert len(data["versions"]) == 2
        assert data["current_entry_id"] == comp1.entry_id

        # Verify version items have expected fields
        v1 = next(v for v in data["versions"] if v["entry_id"] == comp1.entry_id)
        v2 = next(v for v in data["versions"] if v["entry_id"] == comp2.entry_id)

        assert v1["is_current"] is True
        assert v2["is_current"] is False
        assert v1["similarity_threshold"] == 90
        assert v2["similarity_threshold"] == 70

    def test_original_is_oldest_non_duplicate(self, test_engine, client):
        """The original compound should be the oldest non-duplicate sibling."""
        Session = sessionmaker(bind=test_engine)
        session = Session()

        inchikey = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        # Second oldest is not a duplicate — should be original (create first to serve as parent)
        original = _make_compound(
            session,
            compound_name="Aspirin",
            inchikey=inchikey,
            is_duplicate=False,
            processed_at=datetime.now(timezone.utc) - timedelta(days=2),
        )
        # Oldest is a duplicate (child of original)
        oldest = _make_compound(
            session,
            compound_name="Aspirin_dup",
            inchikey=inchikey,
            is_duplicate=True,
            duplicate_of=original.entry_id,
            processed_at=datetime.now(timezone.utc) - timedelta(days=3),
        )
        # Newest
        newest = _make_compound(
            session,
            compound_name="Aspirin_v2",
            inchikey=inchikey,
            is_duplicate=False,
            processed_at=datetime.now(timezone.utc),
        )
        session.close()

        response = client.get(f"/api/v1/compounds/{newest.entry_id}/versions")
        data = response.json()
        assert len(data["versions"]) == 3

        original_version = next(v for v in data["versions"] if v["entry_id"] == original.entry_id)
        oldest_version = next(v for v in data["versions"] if v["entry_id"] == oldest.entry_id)

        assert original_version["is_original"] is True
        assert oldest_version["is_original"] is False

    def test_different_inchikey_structure_not_sibling(self, test_engine, client):
        """Compounds with different InChIKey structure keys are not siblings."""
        Session = sessionmaker(bind=test_engine)
        session = Session()

        comp1 = _make_compound(
            session,
            compound_name="Ethanol",
            inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        )
        # Different structure key
        _make_compound(
            session,
            compound_name="Aspirin",
            inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        )
        session.close()

        response = client.get(f"/api/v1/compounds/{comp1.entry_id}/versions")
        data = response.json()
        # Only Ethanol, no Aspirin — so empty since single compound
        assert data["versions"] == []

    def test_protonation_insensitive_matching(self, test_engine, client):
        """Compounds with different protonation blocks (3rd block) should match."""
        Session = sessionmaker(bind=test_engine)
        session = Session()

        # Same first two blocks, different third block (protonation)
        comp1 = _make_compound(
            session,
            compound_name="Compound_acid",
            inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            processed_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        _make_compound(
            session,
            compound_name="Compound_base",
            inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-M",  # Different protonation
            processed_at=datetime.now(timezone.utc),
        )
        session.close()

        response = client.get(f"/api/v1/compounds/{comp1.entry_id}/versions")
        data = response.json()
        assert len(data["versions"]) == 2

    def test_version_items_have_all_fields(self, test_engine, client):
        """Version items should include all required fields."""
        Session = sessionmaker(bind=test_engine)
        session = Session()

        inchikey = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        comp1 = _make_compound(
            session,
            compound_name="Compound_v1",
            inchikey=inchikey,
            similarity_threshold=90,
            activity_types=["IC50", "Ki"],
            imp_score=0.75,
            qed=0.8,
            similar_compounds=15,
            total_activities=200,
            author_name="Test Author",
            processed_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        _make_compound(
            session,
            compound_name="Compound_v2",
            inchikey=inchikey,
            processed_at=datetime.now(timezone.utc),
        )
        session.close()

        response = client.get(f"/api/v1/compounds/{comp1.entry_id}/versions")
        data = response.json()
        v1 = next(v for v in data["versions"] if v["entry_id"] == comp1.entry_id)

        expected_fields = {
            "entry_id", "compound_name", "similarity_threshold", "activity_types",
            "imp_score", "qed", "similar_compounds", "total_activities",
            "parent_id", "version", "config_diff", "parent_name", "author_name",
            "processed_at", "storage_path", "is_original", "is_current",
        }
        assert expected_fields.issubset(set(v1.keys())), f"Missing fields: {expected_fields - set(v1.keys())}"

    def test_duplicate_of_name_resolved(self, test_engine, client):
        """Duplicate compounds should have their parent name resolved."""
        Session = sessionmaker(bind=test_engine)
        session = Session()

        inchikey = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        parent = _make_compound(
            session,
            compound_name="Aspirin_Original",
            inchikey=inchikey,
            processed_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        child = _make_compound(
            session,
            compound_name="Aspirin_Dup",
            inchikey=inchikey,
            is_duplicate=True,
            duplicate_of=parent.entry_id,
            processed_at=datetime.now(timezone.utc),
        )
        session.close()

        response = client.get(f"/api/v1/compounds/{child.entry_id}/versions")
        data = response.json()

        child_version = next(v for v in data["versions"] if v["entry_id"] == child.entry_id)
        assert child_version["parent_id"] == parent.entry_id
        assert child_version["parent_name"] == "Aspirin_Original"
