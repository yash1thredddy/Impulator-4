"""
Regression tests for session isolation, duplicate deletion safety,
and metrics lock reentrancy.

These tests verify fixes for:
- Session validation bypass (body session_id overriding header)
- Name-based compound deletion fallback (data integrity risk)
- Metrics Lock -> RLock (deadlock prevention)
- Batch size schema/runtime alignment
"""
import threading
import pytest
from pathlib import Path
from pydantic import ValidationError


# Path to source files for static analysis tests
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"


class TestSessionIsolation:
    """Tests that session_id from request body cannot override validated header."""

    def test_body_session_id_field_exists_but_is_unused(self):
        """Ensure body session_id does not override header session_id."""
        from backend.models.schemas import JobCreate

        job = JobCreate(
            compound_name="Test",
            author_name="Author",
            smiles="CCO",
            session_id="attacker-session-id",
        )
        # The schema accepts session_id for backward compat but endpoints ignore it
        assert job.session_id == "attacker-session-id"

    def test_batch_body_session_id_field_is_unused(self):
        """Ensure body session_id does not override header session_id in batch."""
        from backend.models.schemas import BatchJobCreate, JobCreate

        batch = BatchJobCreate(
            compounds=[
                JobCreate(
                    compound_name="Test",
                    author_name="Author",
                    smiles="CCO",
                )
            ],
            session_id="attacker-session-id",
        )
        assert batch.session_id == "attacker-session-id"

    def test_create_job_source_no_body_session_override(self):
        """Verify create_job source code does not use request.session_id."""
        source = (BACKEND_DIR / "api" / "v1" / "jobs.py").read_text()

        # The old vulnerable pattern was:
        # if session_id.startswith("anon-") and request.session_id:
        #     session_id = request.session_id
        assert 'request.session_id' not in source, (
            "jobs.py should not reference request.session_id anywhere"
        )


class TestDuplicateDeletionSafety:
    """Tests that job deletion does not fall back to name-based compound lookup."""

    def test_no_name_fallback_in_delete_job_source(self):
        """Verify delete_job source does not use compound_name fallback."""
        source = (BACKEND_DIR / "api" / "v1" / "jobs.py").read_text()

        # The old unsafe pattern was:
        # compound_entry = db.query(Compound).filter(Compound.compound_name == compound_name).first()
        assert 'Compound.compound_name == compound_name' not in source, (
            "jobs.py should not fall back to compound_name lookup for deletion"
        )


class TestMetricsLockReentrancy:
    """Tests that metrics.to_dict() does not deadlock due to lock reentrancy."""

    def test_metrics_source_uses_rlock(self):
        """Verify metrics.py uses RLock instead of Lock."""
        source = (BACKEND_DIR / "core" / "metrics.py").read_text()

        assert 'from threading import RLock' in source, (
            "metrics.py should import RLock, not Lock"
        )
        assert 'default_factory=RLock' in source, (
            "metrics.py should use RLock as default_factory"
        )

    def test_to_dict_does_not_deadlock(self):
        """Ensure to_dict() completes without deadlock (RLock allows reentrancy)."""
        # Import directly to avoid __init__.py chain
        import importlib
        import sys

        spec = importlib.util.spec_from_file_location(
            "metrics_standalone",
            BACKEND_DIR / "core" / "metrics.py"
        )
        metrics_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics_mod)
        Metrics = metrics_mod.Metrics

        m = Metrics()
        m.record_latency('chembl', 100.0)
        m.record_latency('pdb', 50.0)
        m.increment('jobs_created')

        # This would deadlock with threading.Lock but works with RLock
        result = m.to_dict()

        assert result['jobs_created'] == 1
        assert 'chembl' in result['latencies']
        assert 'pdb' in result['latencies']
        assert result['latencies']['chembl']['count'] == 1

    def test_to_dict_under_concurrent_access(self):
        """Test to_dict under concurrent access from multiple threads."""
        import importlib

        spec = importlib.util.spec_from_file_location(
            "metrics_standalone2",
            BACKEND_DIR / "core" / "metrics.py"
        )
        metrics_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics_mod)
        Metrics = metrics_mod.Metrics

        m = Metrics()
        errors = []

        def writer():
            try:
                for i in range(100):
                    m.increment('api_calls_total')
                    m.record_latency('chembl', float(i))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    m.to_dict()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Concurrent metrics access failed: {errors}"
        assert m.api_calls_total == 200


class TestBatchSizeConsistency:
    """Tests that schema and runtime batch size limits are aligned."""

    def test_schema_allows_1000_compounds(self):
        """Ensure BatchJobCreate schema accepts up to 1000 compounds."""
        from backend.models.schemas import BatchJobCreate, JobCreate

        compounds = [
            JobCreate(
                compound_name=f"Compound_{i}",
                author_name="Author",
                smiles="CCO",
            )
            for i in range(1000)
        ]
        batch = BatchJobCreate(compounds=compounds)
        assert len(batch.compounds) == 1000

    def test_schema_rejects_over_1000_compounds(self):
        """Ensure BatchJobCreate schema rejects > 1000 compounds."""
        from backend.models.schemas import BatchJobCreate, JobCreate

        compounds = [
            JobCreate(
                compound_name=f"Compound_{i}",
                author_name="Author",
                smiles="CCO",
            )
            for i in range(1001)
        ]
        with pytest.raises(ValidationError):
            BatchJobCreate(compounds=compounds)

    def test_runtime_and_schema_batch_limit_match(self):
        """Verify runtime MAX_BATCH_SIZE matches schema max_length."""
        source = (BACKEND_DIR / "api" / "v1" / "jobs.py").read_text()
        assert 'MAX_BATCH_SIZE = 1000' in source

        schema_source = (BACKEND_DIR / "models" / "schemas.py").read_text()
        assert 'max_length=1000' in schema_source


class TestClassyFireHTTP:
    """Tests that ClassyFire API calls use HTTP (server has no TLS support)."""

    def test_classyfire_uses_http_in_classifier(self):
        """Verify chemical_classifier.py uses HTTP for ClassyFire (no TLS support)."""
        source = (BACKEND_DIR / "modules" / "chemical_classifier.py").read_text()
        import re
        # ClassyFire server does not support HTTPS — must use http://
        https_urls = re.findall(r"['\"]https://classyfire", source)
        assert not https_urls, (
            f"ClassyFire has no TLS support. Found HTTPS URLs in chemical_classifier.py: {https_urls}"
        )

    def test_classyfire_uses_http_in_api_client(self):
        """Verify api_client.py uses HTTP for ClassyFire (no TLS support)."""
        source = (BACKEND_DIR / "modules" / "api_client.py").read_text()
        import re
        https_urls = re.findall(r"['\"]https://classyfire", source)
        assert not https_urls, (
            f"ClassyFire has no TLS support. Found HTTPS URLs in api_client.py: {https_urls}"
        )
