"""Verify compound reparenting trigger on parent deletion.

Success criteria #7: create parent with children, delete parent,
assert children reparented to promoted child.
"""
import uuid
from datetime import datetime, timezone

from backend.models.compound import Compound


class TestReparentTrigger:
    """Test the trg_reparent_on_delete trigger on compounds."""

    def _make_compound(self, session, name, parent_id=None, version=1, **kw):
        """Helper to create a compound."""
        entry_id = uuid.uuid4()
        defaults = dict(
            entry_id=entry_id,
            compound_name=name,
            smiles="CCO",
            inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            inchikey_structure_key="LFQSCWFLJHTTHZ-UHFFFAOYSA",
            similarity_threshold=90,
            activity_types=["IC50"],
            parent_id=parent_id,
            version=version,
            processed_at=datetime.now(timezone.utc),
        )
        defaults.update(kw)
        comp = Compound(**defaults)
        session.add(comp)
        session.flush()
        return comp

    def test_delete_parent_reparents_children(self, db_session):
        """Deleting a root parent promotes the lowest-version child."""
        parent = self._make_compound(db_session, "Parent")
        child1 = self._make_compound(
            db_session, "Child1", parent_id=parent.entry_id, version=2
        )
        child2 = self._make_compound(
            db_session, "Child2", parent_id=parent.entry_id, version=3
        )
        db_session.commit()

        # Delete parent -- trigger should reparent
        db_session.delete(parent)
        db_session.commit()

        # Refresh children from DB
        db_session.expire_all()
        c1 = db_session.get(Compound, child1.entry_id)
        c2 = db_session.get(Compound, child2.entry_id)

        # child1 (lowest version) should be promoted to root
        assert c1.parent_id is None, "child1 should be promoted to root"
        assert c1.version == 1, "promoted child should have version 1"

        # child2 should be reparented to child1
        assert c2.parent_id == child1.entry_id, "child2 should point to promoted child1"

    def test_delete_leaf_does_not_trigger_reparent(self, db_session):
        """Deleting a child (non-root) does not trigger reparenting."""
        parent = self._make_compound(db_session, "Parent")
        child = self._make_compound(
            db_session, "Child", parent_id=parent.entry_id, version=2
        )
        db_session.commit()

        db_session.delete(child)
        db_session.commit()

        db_session.expire_all()
        p = db_session.get(Compound, parent.entry_id)
        assert p is not None, "Parent should still exist"
        assert p.parent_id is None, "Parent should still be root"

    def test_delete_parent_no_children(self, db_session):
        """Deleting a root with no children is a no-op for the trigger."""
        parent = self._make_compound(db_session, "Alone")
        db_session.commit()

        db_session.delete(parent)
        db_session.commit()
        # No assertion needed -- just verify no error

    def test_delete_parent_with_multiple_children_promotes_earliest(self, db_session):
        """With multiple children, lowest version (then earliest processed_at) wins."""
        parent = self._make_compound(db_session, "Parent")
        # Children with different versions; child_early has lower version
        child_early = self._make_compound(
            db_session, "ChildEarly", parent_id=parent.entry_id, version=2,
            processed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        child_late = self._make_compound(
            db_session, "ChildLate", parent_id=parent.entry_id, version=3,
            processed_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        db_session.commit()

        db_session.delete(parent)
        db_session.commit()

        db_session.expire_all()
        ce = db_session.get(Compound, child_early.entry_id)
        cl = db_session.get(Compound, child_late.entry_id)

        assert ce.parent_id is None, "Earlier child should be promoted"
        assert ce.version == 1
        assert cl.parent_id == child_early.entry_id
