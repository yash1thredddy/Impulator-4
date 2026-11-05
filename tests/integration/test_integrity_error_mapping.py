# tests/integration/test_integrity_error_mapping.py
"""_handle_integrity_error maps driver-level violations to domain errors.

The mapping is coupled to the installed PG driver's ``errors`` module; these
tests are the acceptance gate for the PG driver swap (v2 -> v3) and must pass
unchanged on both drivers.
"""
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from backend.repositories.compound_repository import compound_repo
from backend.repositories.job_repository import (
    DuplicateEntryError,
    ReferenceError,
    ValidationError,
)

ETHANOL_IK = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def _kwargs(**over):
    base = dict(
        entry_id=uuid.uuid4(),
        compound_name="IntegrityProbe",
        smiles="CCO",
        inchikey=ETHANOL_IK,
        similarity_threshold=90,
    )
    base.update(over)
    return base


def test_unique_violation_maps_to_duplicate_entry(db_session, seed_compound):
    existing = seed_compound(name="Existing")
    existing_id = existing.entry_id
    db_session.expunge_all()
    with pytest.raises(DuplicateEntryError) as exc_info:
        compound_repo.create_compound(db_session, **_kwargs(entry_id=existing_id))
    # Domain error must chain the original (raise ... from e).
    assert isinstance(exc_info.value.__cause__, IntegrityError)


def test_fk_violation_maps_to_reference_error(db_session):
    with pytest.raises(ReferenceError):
        compound_repo.create_compound(db_session, **_kwargs(job_id=uuid.uuid4()))


def test_check_violation_maps_to_validation_error(db_session):
    with pytest.raises(ValidationError):
        compound_repo.create_compound(db_session, **_kwargs(similarity_threshold=5))
