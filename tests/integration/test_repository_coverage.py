"""
Integration tests for repository layer coverage.
Tests repositories with real SQLite (in-memory) via existing fixtures.
"""
import uuid
from datetime import datetime, timezone


class TestJobRepository:
    """Tests for JobRepository methods."""

    def _create_job(self, db_session, **overrides):
        """Helper to create a job record."""
        from backend.models.database import Job, JobStatus, JobType

        defaults = {
            "id": str(uuid.uuid4()),
            "status": JobStatus.PENDING,
            "session_id": str(uuid.uuid4()),
            "job_type": JobType.SINGLE,
        }
        defaults.update(overrides)
        job = Job(**defaults)
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        return job

    def test_get_by_job_id(self, db_session):
        """Test finding a job by ID."""
        from backend.repositories.job_repository import job_repo
        job = self._create_job(db_session)
        found = job_repo.get_by_job_id(db_session, job.id)
        assert found is not None
        assert found.id == job.id

    def test_get_by_job_id_not_found(self, db_session):
        """Test returns None for nonexistent job."""
        from backend.repositories.job_repository import job_repo
        assert job_repo.get_by_job_id(db_session, "nonexistent") is None

    def test_get_active_jobs(self, db_session):
        """Test getting active (pending/processing) jobs."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        sid = str(uuid.uuid4())
        self._create_job(db_session, status=JobStatus.PENDING, session_id=sid)
        self._create_job(db_session, status=JobStatus.COMPLETED, session_id=sid)

        active = job_repo.get_active_jobs(db_session, session_id=sid)
        assert len(active) == 1
        assert active[0].status == JobStatus.PENDING

    def test_get_active_jobs_no_session(self, db_session):
        """Test getting active jobs without session filter."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        self._create_job(db_session, status=JobStatus.PENDING)
        self._create_job(db_session, status=JobStatus.PROCESSING)

        active = job_repo.get_active_jobs(db_session)
        assert len(active) == 2

    def test_get_jobs_paginated(self, db_session):
        """Test paginated job listing."""
        from backend.repositories.job_repository import job_repo

        sid = str(uuid.uuid4())
        for _ in range(5):
            self._create_job(db_session, session_id=sid)

        jobs, total = job_repo.get_jobs_paginated(db_session, session_id=sid, offset=0, limit=3)
        assert len(jobs) == 3
        assert total == 5

    def test_get_jobs_paginated_with_status_filter(self, db_session):
        """Test paginated jobs with status filter."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        self._create_job(db_session, status=JobStatus.PENDING)
        self._create_job(db_session, status=JobStatus.COMPLETED)

        jobs, total = job_repo.get_jobs_paginated(
            db_session, status_filter=[JobStatus.COMPLETED]
        )
        assert total == 1
        assert jobs[0].status == JobStatus.COMPLETED

    def test_find_by_idempotency_key(self, db_session):
        """Test finding job by idempotency key."""
        from backend.repositories.job_repository import job_repo

        sid = str(uuid.uuid4())
        idem_key = "idem-123"
        job = self._create_job(db_session, session_id=sid, idempotency_key=idem_key)

        found = job_repo.find_by_idempotency_key(db_session, sid, idem_key)
        assert found is not None
        assert found.id == job.id

    def test_get_batch_jobs(self, db_session):
        """Test getting all jobs in a batch."""
        from backend.repositories.job_repository import job_repo

        batch_id = str(uuid.uuid4())
        self._create_job(db_session, batch_id=batch_id)
        self._create_job(db_session, batch_id=batch_id)
        self._create_job(db_session)  # Different batch

        jobs = job_repo.get_batch_jobs(db_session, batch_id)
        assert len(jobs) == 2

    def test_get_batch_summary(self, db_session):
        """Test batch summary aggregation."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        batch_id = str(uuid.uuid4())
        self._create_job(db_session, batch_id=batch_id, status=JobStatus.COMPLETED,
                         input_params='{"compound_name": "Aspirin", "smiles": "CC"}')
        self._create_job(db_session, batch_id=batch_id, status=JobStatus.PENDING,
                         input_params='{"compound_name": "Caffeine", "smiles": "CN"}')

        summary = job_repo.get_batch_summary(db_session, batch_id)
        assert summary["total_jobs"] == 2
        assert summary["completed"] == 1
        assert summary["pending"] == 1
        assert len(summary["compound_names"]) == 2

    def test_get_batch_summary_empty(self, db_session):
        """Test batch summary for nonexistent batch."""
        from backend.repositories.job_repository import job_repo
        summary = job_repo.get_batch_summary(db_session, "nonexistent-batch")
        assert summary == {}

    def test_count_by_status(self, db_session):
        """Test counting jobs by status."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        self._create_job(db_session, status=JobStatus.PENDING)
        self._create_job(db_session, status=JobStatus.PENDING)
        self._create_job(db_session, status=JobStatus.COMPLETED)

        count = job_repo.count_by_status(db_session, [JobStatus.PENDING])
        assert count == 2

    def test_get_pending_processing_count(self, db_session):
        """Test pending+processing count."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        self._create_job(db_session, status=JobStatus.PENDING)
        self._create_job(db_session, status=JobStatus.PROCESSING)
        self._create_job(db_session, status=JobStatus.COMPLETED)

        count = job_repo.get_pending_processing_count(db_session)
        assert count == 2

    def test_claim_next_pending_job(self, db_session):
        """Test round-robin fair scheduling claim."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        job = self._create_job(db_session, status=JobStatus.PENDING)

        claimed = job_repo.claim_next_pending_job(db_session)
        assert claimed is not None
        assert claimed.id == job.id

    def test_claim_next_pending_no_jobs(self, db_session):
        """Test claim returns None when no pending jobs."""
        from backend.repositories.job_repository import job_repo
        assert job_repo.claim_next_pending_job(db_session) is None

    def test_create_job(self, db_session):
        """Test write-locked job creation."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus, JobType

        job_id = str(uuid.uuid4())
        job = job_repo.create_job(
            db_session,
            id=job_id,
            job_type=JobType.SINGLE,
            session_id="test-session",
            input_params='{"compound_name": "test"}',
        )
        db_session.commit()
        assert job.id == job_id
        assert job.status == JobStatus.PENDING

    def test_update_status(self, db_session):
        """Test write-locked status update."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        job = self._create_job(db_session, status=JobStatus.PENDING)
        updated = job_repo.update_status(
            db_session, job.id, JobStatus.PROCESSING,
            current_step="Processing..."
        )
        db_session.commit()
        assert updated.status == JobStatus.PROCESSING
        assert updated.current_step == "Processing..."

    def test_update_status_resurrection_guard(self, db_session):
        """Test SD-13: cannot resurrect CANCELLED/FAILED jobs."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        job = self._create_job(db_session, status=JobStatus.CANCELLED)
        result = job_repo.update_status(db_session, job.id, JobStatus.PENDING)
        assert result is None  # Blocked by resurrection guard

    def test_update_progress(self, db_session):
        """Test write-locked progress update."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        job = self._create_job(db_session, status=JobStatus.PROCESSING)
        updated = job_repo.update_progress(db_session, job.id, 50.0, "Halfway")
        db_session.commit()
        assert updated.progress == 50.0
        assert updated.current_step == "Halfway"

    def test_cancel_batch_jobs(self, db_session):
        """Test cancelling all pending/processing jobs in a batch."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        batch_id = str(uuid.uuid4())
        self._create_job(db_session, batch_id=batch_id, status=JobStatus.PENDING)
        self._create_job(db_session, batch_id=batch_id, status=JobStatus.PROCESSING)
        self._create_job(db_session, batch_id=batch_id, status=JobStatus.COMPLETED)

        cancelled = job_repo.cancel_batch_jobs(db_session, batch_id)
        db_session.commit()
        assert cancelled == 2

    def test_delete_job(self, db_session):
        """Test write-locked job deletion."""
        from backend.repositories.job_repository import job_repo

        job = self._create_job(db_session)
        result = job_repo.delete_job(db_session, job.id)
        db_session.commit()
        assert result is True
        assert job_repo.get_by_job_id(db_session, job.id) is None

    def test_delete_job_not_found(self, db_session):
        """Test delete returns False for nonexistent job."""
        from backend.repositories.job_repository import job_repo
        assert job_repo.delete_job(db_session, "nonexistent") is False

    def test_get_by_status(self, db_session):
        """Test getting all jobs by status."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        self._create_job(db_session, status=JobStatus.FAILED)
        self._create_job(db_session, status=JobStatus.FAILED)

        jobs = job_repo.get_by_status(db_session, JobStatus.FAILED)
        assert len(jobs) == 2

    def test_get_completed_jobs_since(self, db_session):
        """Test getting completed jobs after cutoff."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._create_job(db_session, status=JobStatus.COMPLETED,
                         completed_at=datetime(2026, 2, 1, tzinfo=timezone.utc))

        jobs = job_repo.get_completed_jobs_since(db_session, cutoff)
        assert len(jobs) == 1

    def test_get_failed_jobs_since(self, db_session):
        """Test getting failed jobs after cutoff."""
        from backend.repositories.job_repository import job_repo
        from backend.models.database import JobStatus

        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._create_job(db_session, status=JobStatus.FAILED,
                         completed_at=datetime(2026, 2, 1, tzinfo=timezone.utc))

        jobs = job_repo.get_failed_jobs_since(db_session, cutoff)
        assert len(jobs) == 1


class TestCompoundRepository:
    """Tests for CompoundRepository methods."""

    def test_get_by_entry_id(self, db_session, seed_compound):
        """Test finding compound by entry_id."""
        from backend.repositories.compound_repository import compound_repo
        comp = seed_compound(name="TestFind")
        found = compound_repo.get_by_entry_id(db_session, comp.entry_id)
        assert found is not None
        assert found.compound_name == "TestFind"

    def test_get_by_entry_id_not_found(self, db_session):
        """Test returns None for nonexistent entry_id."""
        from backend.repositories.compound_repository import compound_repo
        assert compound_repo.get_by_entry_id(db_session, "nonexistent") is None

    def test_get_compounds_paginated(self, db_session, seed_compound):
        """Test paginated compound listing."""
        from backend.repositories.compound_repository import compound_repo
        for i in range(5):
            seed_compound(name=f"Compound{i}", entry_id=str(uuid.uuid4()))

        compounds, total = compound_repo.get_compounds_paginated(db_session, offset=0, limit=3)
        assert len(compounds) == 3
        assert total == 5

    def test_get_compounds_paginated_with_search(self, db_session, seed_compound):
        """Test paginated listing with search filter."""
        from backend.repositories.compound_repository import compound_repo
        seed_compound(name="Aspirin")
        seed_compound(name="Caffeine", entry_id=str(uuid.uuid4()))

        compounds, total = compound_repo.get_compounds_paginated(db_session, search="Aspirin")
        assert total == 1
        assert compounds[0].compound_name == "Aspirin"

    def test_get_compounds_paginated_sort_asc(self, db_session, seed_compound):
        """Test paginated listing with ascending sort."""
        from backend.repositories.compound_repository import compound_repo
        seed_compound(name="Alpha", entry_id=str(uuid.uuid4()))
        seed_compound(name="Beta", entry_id=str(uuid.uuid4()))

        compounds, _ = compound_repo.get_compounds_paginated(
            db_session, sort_by="compound_name", sort_order="asc"
        )
        assert compounds[0].compound_name == "Alpha"

    def test_get_versions_with_siblings(self, db_session, seed_compound):
        """Test getting structural siblings by InChIKey."""
        from backend.repositories.compound_repository import compound_repo

        inchikey = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        c1 = seed_compound(name="Aspirin", inchikey=inchikey)
        seed_compound(name="Aspirin (v2)", inchikey=inchikey, entry_id=str(uuid.uuid4()))  # sibling row

        versions = compound_repo.get_versions(db_session, c1.entry_id)
        assert len(versions) == 2

    def test_get_versions_not_found(self, db_session):
        """Test get_versions returns empty for nonexistent compound."""
        from backend.repositories.compound_repository import compound_repo
        assert compound_repo.get_versions(db_session, "nonexistent") == []

    def test_get_versions_no_inchikey(self, db_session, seed_compound):
        """Test get_versions returns empty when compound has no InChIKey."""
        from backend.repositories.compound_repository import compound_repo
        comp = seed_compound(name="NoKey", inchikey=None,
                             inchikey_structure_key=None)
        assert compound_repo.get_versions(db_session, comp.entry_id) == []

    def test_find_by_structure_key(self, db_session, seed_compound):
        """Test finding canonical compound by structure key."""
        from backend.repositories.compound_repository import compound_repo

        comp = seed_compound(name="Canon")
        found = compound_repo.find_by_structure_key(db_session, comp.inchikey_structure_key)
        assert found is not None
        assert found.entry_id == comp.entry_id

    def test_find_duplicates_by_structure_key(self, db_session, seed_compound):
        """Test finding all compounds matching a structure key."""
        from backend.repositories.compound_repository import compound_repo

        c1 = seed_compound(name="Main")
        seed_compound(name="Dup", is_duplicate=True, duplicate_of=c1.entry_id,
                     entry_id=str(uuid.uuid4()))  # duplicate row

        dups = compound_repo.find_duplicates_by_structure_key(db_session, c1.inchikey_structure_key)
        assert len(dups) == 2

    def test_create_compound_auto_structure_key(self, db_session):
        """Test create_compound auto-computes inchikey_structure_key."""
        from backend.repositories.compound_repository import compound_repo

        comp = compound_repo.create_compound(
            db_session,
            entry_id=str(uuid.uuid4()),
            compound_name="AutoKey",
            smiles="CCO",
            inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            similarity_threshold=90,
        )
        db_session.commit()
        assert comp.inchikey_structure_key == "LFQSCWFLJHTTHZ-UHFFFAOYSA"

    def test_update_compound(self, db_session, seed_compound):
        """Test write-locked compound update."""
        from backend.repositories.compound_repository import compound_repo

        comp = seed_compound(name="Original")
        updated = compound_repo.update_compound(
            db_session, comp.entry_id, compound_name="Updated"
        )
        db_session.commit()
        assert updated.compound_name == "Updated"

    def test_update_compound_not_found(self, db_session):
        """Test update returns None for nonexistent compound."""
        from backend.repositories.compound_repository import compound_repo
        assert compound_repo.update_compound(db_session, "nonexistent", compound_name="X") is None

    def test_archive_compound(self, db_session, seed_compound):
        """Test archiving a compound to DeletedCompound table."""
        from backend.repositories.compound_repository import compound_repo

        comp = seed_compound(name="ToArchive")
        deleted_record = compound_repo.archive_compound(
            db_session, comp, session_id="test-session", deletion_reason="test"
        )
        db_session.commit()

        assert deleted_record.compound_name == "ToArchive"
        assert deleted_record.deleted_by_session == "test-session"
        assert deleted_record.deletion_reason == "test"

    def test_handle_children_before_delete(self, db_session, seed_compound):
        """Test child promotion and reparenting on parent deletion."""
        from backend.repositories.compound_repository import compound_repo

        parent = seed_compound(name="Parent")
        child1 = seed_compound(name="Child1", is_duplicate=True, duplicate_of=parent.entry_id,
                               entry_id=str(uuid.uuid4()))
        child2 = seed_compound(name="Child2", is_duplicate=True, duplicate_of=parent.entry_id,
                               entry_id=str(uuid.uuid4()))

        count = compound_repo.handle_children_before_delete(db_session, parent.entry_id)
        db_session.commit()

        assert count >= 2  # promoted + reparented

        # Refresh to see changes
        db_session.refresh(child1)
        db_session.refresh(child2)

        # One child should be promoted (is_duplicate=False)
        promoted = [c for c in [child1, child2] if not c.is_duplicate]
        assert len(promoted) == 1

    def test_handle_children_no_children(self, db_session, seed_compound):
        """Test handle_children returns 0 when no children exist."""
        from backend.repositories.compound_repository import compound_repo
        comp = seed_compound(name="NoChildren")
        assert compound_repo.handle_children_before_delete(db_session, comp.entry_id) == 0

    def test_handle_children_duplicate_compound(self, db_session, seed_compound):
        """Test handle_children returns 0 for duplicate compounds."""
        from backend.repositories.compound_repository import compound_repo
        parent = seed_compound(name="Parent")
        child = seed_compound(name="Child", is_duplicate=True, duplicate_of=parent.entry_id,
                              entry_id=str(uuid.uuid4()))
        assert compound_repo.handle_children_before_delete(db_session, child.entry_id) == 0

    def test_find_by_name_case_insensitive(self, db_session, seed_compound):
        """Test case-insensitive name search."""
        from backend.repositories.compound_repository import compound_repo
        seed_compound(name="Aspirin")
        results = compound_repo.find_by_name_case_insensitive(db_session, "aspirin")
        assert len(results) == 1

    def test_find_names_by_prefix(self, db_session, seed_compound):
        """Test finding names by prefix."""
        from backend.repositories.compound_repository import compound_repo
        seed_compound(name="Aspirin")
        seed_compound(name="Ascorbic Acid", entry_id=str(uuid.uuid4()))
        names = compound_repo.find_names_by_prefix(db_session, "asp")
        assert len(names) == 1  # "Aspirin" matches "asp" prefix

    def test_count_compounds(self, db_session, seed_compound):
        """Test counting compounds."""
        from backend.repositories.compound_repository import compound_repo
        seed_compound(name="A")
        seed_compound(name="B", entry_id=str(uuid.uuid4()))
        assert compound_repo.count_compounds(db_session) == 2

    def test_count_compounds_with_search(self, db_session, seed_compound):
        """Test counting compounds with search filter."""
        from backend.repositories.compound_repository import compound_repo
        seed_compound(name="Aspirin")
        seed_compound(name="Caffeine", entry_id=str(uuid.uuid4()))
        assert compound_repo.count_compounds(db_session, search="Aspirin") == 1


class TestBaseRepository:
    """Tests for BaseRepository generic methods."""

    def test_get_by_id(self, db_session, seed_compound):
        """Test get_by_id on compound."""
        from backend.repositories.compound_repository import compound_repo
        comp = seed_compound(name="GetById")
        found = compound_repo.get_by_id(db_session, comp.id, id_column="id")
        assert found is not None
        assert found.compound_name == "GetById"

    def test_get_all_with_limit(self, db_session, seed_compound):
        """Test get_all with pagination."""
        from backend.repositories.compound_repository import compound_repo
        for i in range(4):
            seed_compound(name=f"C{i}", entry_id=str(uuid.uuid4()))

        results = compound_repo.get_all(db_session, offset=0, limit=2)
        assert len(results) == 2

    def test_count(self, db_session, seed_compound):
        """Test count method."""
        from backend.repositories.compound_repository import compound_repo
        seed_compound(name="CountMe")
        assert compound_repo.count(db_session) >= 1

    def test_add_entity(self, db_session):
        """Test add entity with write lock."""
        from backend.repositories.compound_repository import compound_repo
        from backend.models.database import Compound

        comp = Compound(
            entry_id=str(uuid.uuid4()),
            compound_name="Added",
            smiles="CCO",
        )
        compound_repo.add(db_session, comp)
        db_session.commit()
        assert comp.id is not None

    def test_delete_entity(self, db_session, seed_compound):
        """Test delete entity with write lock."""
        from backend.repositories.compound_repository import compound_repo
        comp = seed_compound(name="ToDelete")
        compound_repo.delete(db_session, comp)
        db_session.commit()
        assert compound_repo.get_by_entry_id(db_session, comp.entry_id) is None
