"""
Integration tests for compound CRUD endpoints.

Tests:
- GET /api/v1/compounds (list with pagination, search, duplicate filtering)
- GET /api/v1/compounds/{entry_id} (single compound detail)
- DELETE /api/v1/compounds/{entry_id} (delete with audit trail + child reparenting)
- POST /api/v1/compounds/batch-delete (batch deletion)
"""
import uuid
from datetime import datetime, timezone

import pytest
from unittest.mock import patch
from sqlalchemy.orm import sessionmaker


def _seed_compound(session, name="TestCompound", smiles="CCO", **overrides):
    """Seed a compound into the database and return its entry_id."""
    from backend.models.database import Compound

    entry_id = str(uuid.uuid4())
    defaults = {
        "entry_id": entry_id,
        "compound_name": name,
        "smiles": smiles,
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "similarity_threshold": 90,
        "activity_types": "IC50",
        "total_activities": 50,
        "imp_score": 0.75,
        "qed": 0.8,
        "author_name": "Test Author",
        "processed_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    comp = Compound(**defaults)
    session.add(comp)
    session.commit()
    return entry_id


# ─────────────────────────────────────────────────
# GET /compounds (list)
# ─────────────────────────────────────────────────

class TestListCompounds:
    """Tests for GET /api/v1/compounds."""

    def test_empty_list(self, client):
        """No compounds returns empty paginated response."""
        response = client.get("/api/v1/compounds")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    def test_returns_compounds(self, test_engine, client):
        """Seeded compounds appear in list response."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        _seed_compound(session, name="Aspirin")
        _seed_compound(session, name="Caffeine")
        session.close()

        response = client.get("/api/v1/compounds")
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        names = {item["compound_name"] for item in data["items"]}
        assert names == {"Aspirin", "Caffeine"}

    def test_pagination(self, test_engine, client):
        """Pagination returns correct page size and total."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        for i in range(5):
            _seed_compound(session, name=f"Compound_{i}")
        session.close()

        response = client.get("/api/v1/compounds?page=1&per_page=2")
        data = response.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["pages"] == 3

    def test_search_filter(self, test_engine, client):
        """Search by compound name filters results."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        _seed_compound(session, name="Aspirin")
        _seed_compound(session, name="Caffeine")
        _seed_compound(session, name="Aspirin_v2")
        session.close()

        response = client.get("/api/v1/compounds?search=Aspirin")
        data = response.json()
        assert data["total"] == 2
        names = {item["compound_name"] for item in data["items"]}
        assert names == {"Aspirin", "Aspirin_v2"}

    def test_duplicates_excluded_by_default(self, test_engine, client):
        """Duplicate compounds are excluded by default."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        parent_id = _seed_compound(session, name="Aspirin")
        _seed_compound(session, name="Aspirin_dup", is_duplicate=True, duplicate_of=parent_id)
        session.close()

        response = client.get("/api/v1/compounds")
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["compound_name"] == "Aspirin"

    def test_include_duplicates(self, test_engine, client):
        """include_duplicates=true shows all compounds."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        parent_id = _seed_compound(session, name="Aspirin")
        _seed_compound(session, name="Aspirin_dup", is_duplicate=True, duplicate_of=parent_id)
        session.close()

        response = client.get("/api/v1/compounds?include_duplicates=true")
        data = response.json()
        assert data["total"] == 2

    def test_response_fields(self, test_engine, client):
        """Each compound item has all expected fields."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        _seed_compound(session, name="Aspirin")
        session.close()

        response = client.get("/api/v1/compounds")
        item = response.json()["items"][0]

        expected_fields = {
            "entry_id", "compound_name", "chembl_id", "smiles", "inchikey",
            "total_activities", "imp_candidates", "imp_score",
            "similarity_threshold", "qed", "num_outliers", "author_name",
            "storage_path", "processed_at", "is_duplicate", "duplicate_of",
        }
        assert set(item.keys()) == expected_fields

    def test_search_sql_injection_escaped(self, test_engine, client):
        """SQL wildcards in search are escaped to prevent pattern injection."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        _seed_compound(session, name="Aspirin")
        _seed_compound(session, name="A%test")
        session.close()

        # The % should be escaped, not treated as wildcard
        response = client.get("/api/v1/compounds?search=%25")
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["compound_name"] == "A%test"


# ─────────────────────────────────────────────────
# GET /compounds/{entry_id}
# ─────────────────────────────────────────────────

class TestGetCompound:
    """Tests for GET /api/v1/compounds/{entry_id}."""

    def test_get_existing_compound(self, test_engine, client):
        """Returns compound data for valid entry_id."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        entry_id = _seed_compound(session, name="Aspirin", smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
        session.close()

        response = client.get(f"/api/v1/compounds/{entry_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["compound_name"] == "Aspirin"
        assert data["entry_id"] == entry_id

    def test_get_nonexistent_compound_returns_404(self, client):
        """Nonexistent entry_id returns 404."""
        response = client.get("/api/v1/compounds/nonexistent-id")
        assert response.status_code == 404


# ─────────────────────────────────────────────────
# DELETE /compounds/{entry_id}
# ─────────────────────────────────────────────────

class TestDeleteCompound:
    """Tests for DELETE /api/v1/compounds/{entry_id}."""

    def test_delete_existing_compound(self, test_engine, client):
        """Deleting a compound removes it and returns confirmation."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        entry_id = _seed_compound(session, name="ToDelete")
        session.close()

        response = client.delete(f"/api/v1/compounds/{entry_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["entry_id"] == entry_id
        assert data["compound_name"] == "ToDelete"
        assert "message" in data

        # Verify compound is gone
        get_response = client.get(f"/api/v1/compounds/{entry_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        """Deleting nonexistent compound returns 404."""
        response = client.delete("/api/v1/compounds/nonexistent-id")
        assert response.status_code == 404

    def test_delete_creates_audit_record(self, test_engine, client):
        """Deletion creates a record in deleted_compounds table."""
        from backend.models.database import DeletedCompound

        Session = sessionmaker(bind=test_engine)
        session = Session()
        entry_id = _seed_compound(session, name="AuditMe")
        session.close()

        client.delete(f"/api/v1/compounds/{entry_id}")

        # Check audit table
        session = Session()
        audit = session.query(DeletedCompound).filter(
            DeletedCompound.entry_id == entry_id
        ).first()
        assert audit is not None
        assert audit.compound_name == "AuditMe"
        assert audit.deletion_reason == "user_request"
        session.close()

    def test_delete_main_promotes_child(self, test_engine, client):
        """Deleting a main compound promotes oldest child to main."""
        from backend.models.database import Compound

        Session = sessionmaker(bind=test_engine)
        session = Session()
        parent_id = _seed_compound(session, name="Parent")
        child1_id = _seed_compound(
            session, name="Child1", is_duplicate=True, duplicate_of=parent_id,
        )
        child2_id = _seed_compound(
            session, name="Child2", is_duplicate=True, duplicate_of=parent_id,
        )
        session.close()

        # Delete parent
        client.delete(f"/api/v1/compounds/{parent_id}")

        # Child1 should be promoted (oldest child)
        session = Session()
        child1 = session.query(Compound).filter(Compound.entry_id == child1_id).first()
        child2 = session.query(Compound).filter(Compound.entry_id == child2_id).first()
        assert child1.is_duplicate is False
        assert child1.duplicate_of is None
        assert child2.duplicate_of == child1_id  # Re-pointed to promoted child
        session.close()

    def test_delete_duplicate_does_not_promote(self, test_engine, client):
        """Deleting a duplicate doesn't affect parent or siblings."""
        from backend.models.database import Compound

        Session = sessionmaker(bind=test_engine)
        session = Session()
        parent_id = _seed_compound(session, name="Parent")
        dup_id = _seed_compound(
            session, name="Duplicate", is_duplicate=True, duplicate_of=parent_id,
        )
        session.close()

        client.delete(f"/api/v1/compounds/{dup_id}")

        # Parent should be unchanged
        session = Session()
        parent = session.query(Compound).filter(Compound.entry_id == parent_id).first()
        assert parent is not None
        assert parent.is_duplicate is False
        session.close()


# ─────────────────────────────────────────────────
# POST /compounds/batch-delete
# ─────────────────────────────────────────────────

class TestBatchDeleteCompounds:
    """Tests for POST /api/v1/compounds/batch-delete."""

    def test_batch_delete_multiple(self, test_engine, client):
        """Batch deletes multiple compounds."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        id1 = _seed_compound(session, name="Del1")
        id2 = _seed_compound(session, name="Del2")
        session.close()

        response = client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": [id1, id2]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_deleted"] == 2
        assert len(data["deleted"]) == 2
        assert len(data["not_found"]) == 0

    def test_batch_delete_partial_not_found(self, test_engine, client):
        """Batch delete with some nonexistent IDs reports not_found."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        real_id = _seed_compound(session, name="Real")
        session.close()

        response = client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": [real_id, "fake-id"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_deleted"] == 1
        assert len(data["not_found"]) == 1
        assert data["not_found"][0] == "fake-id"

    def test_batch_delete_empty_list_returns_400(self, client):
        """Empty entry_ids list returns 400."""
        response = client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": []},
        )
        assert response.status_code == 400

    def test_batch_delete_over_50_returns_400(self, client):
        """More than 50 entry_ids returns 400."""
        ids = [str(uuid.uuid4()) for _ in range(51)]
        response = client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": ids},
        )
        assert response.status_code == 400

    def test_batch_delete_creates_audit_records(self, test_engine, client):
        """Batch delete creates audit records for each deleted compound."""
        from backend.models.database import DeletedCompound

        Session = sessionmaker(bind=test_engine)
        session = Session()
        id1 = _seed_compound(session, name="BatchDel1")
        id2 = _seed_compound(session, name="BatchDel2")
        session.close()

        client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": [id1, id2]},
        )

        session = Session()
        audits = session.query(DeletedCompound).all()
        assert len(audits) == 2
        reasons = {a.deletion_reason for a in audits}
        assert reasons == {"batch_delete"}
        session.close()

    def test_batch_delete_deduplicates_ids(self, test_engine, client):
        """Duplicate entry_ids in request are deduplicated."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        entry_id = _seed_compound(session, name="DupId")
        session.close()

        response = client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": [entry_id, entry_id]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_deleted"] == 1

    def test_batch_delete_promotes_children(self, test_engine, client):
        """Batch deleting a parent promotes children."""
        from backend.models.database import Compound

        Session = sessionmaker(bind=test_engine)
        session = Session()
        parent_id = _seed_compound(session, name="BatchParent")
        child_id = _seed_compound(
            session, name="BatchChild", is_duplicate=True, duplicate_of=parent_id,
        )
        session.close()

        client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": [parent_id]},
        )

        session = Session()
        child = session.query(Compound).filter(Compound.entry_id == child_id).first()
        assert child.is_duplicate is False
        assert child.duplicate_of is None
        session.close()


# ─────────────────────────────────────────────────
# Search edge cases
# ─────────────────────────────────────────────────

class TestCompoundSearchEdgeCases:
    """Tests for search wildcard escaping in GET /api/v1/compounds."""

    def test_search_underscore_escaped(self, test_engine, client):
        """Literal underscore in search is escaped, not treated as SQL wildcard."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        _seed_compound(session, name="Aspirin_v2")
        _seed_compound(session, name="AspirinXv2")  # Would match if _ is wildcard
        session.close()

        # Search for literal underscore — should only match "Aspirin_v2"
        response = client.get("/api/v1/compounds?search=_v2")
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["compound_name"] == "Aspirin_v2"

    def test_search_backslash_escaped(self, test_engine, client):
        r"""Backslash in search does not break the LIKE escape sequence."""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        _seed_compound(session, name=r"Test\Compound")
        _seed_compound(session, name="TestXCompound")
        session.close()

        response = client.get(r"/api/v1/compounds?search=\C")
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["compound_name"] == r"Test\Compound"


# ─────────────────────────────────────────────────
# Batch delete validation edge cases
# ─────────────────────────────────────────────────

class TestBatchDeleteValidation:
    """Tests for input validation in POST /api/v1/compounds/batch-delete."""

    def test_non_string_entry_ids_returns_400(self, client):
        """Non-string values in entry_ids list returns 400."""
        response = client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": [123, None]},
        )
        # FastAPI may return 422 for type validation or 400 from our check
        assert response.status_code in (400, 422)

    def test_empty_string_entry_id_returns_400(self, client):
        """Empty string in entry_ids list returns 400."""
        response = client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": [""]},
        )
        assert response.status_code == 400
        assert "non-empty strings" in response.json()["detail"]

    def test_whitespace_only_entry_id_returns_400(self, client):
        """Whitespace-only string in entry_ids list returns 400."""
        response = client.post(
            "/api/v1/compounds/batch-delete",
            json={"entry_ids": ["   "]},
        )
        assert response.status_code == 400
        assert "non-empty strings" in response.json()["detail"]
