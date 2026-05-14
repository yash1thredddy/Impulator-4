"""
Phase 19.4 Gap Coverage Tests
==============================
Validates behavioral requirements for tasks shipped in plans 01–09 of
Phase 19.4 (frontend-sync-wire-backend-changes-to-ui) that lack direct
automated test assertions.

These requirements have NO formal REQ-IDs in REQUIREMENTS.md because Phase
19.4 was inserted after the last REQUIREMENTS.md update (2026-03-21).  IDs
are derived post-hoc from SUMMARY accomplishments.

Gap Map
-------
FE-01  PENDING_UPLOAD included in get_active_jobs status filter
FE-02  PENDING_UPLOAD jobs counted as completed in batch summary (SQL CASE)
FE-03  mark_pending_upload sets current_step="Processing complete" (not "Uploading results...")
FE-04  cancel_job() guard: PENDING_UPLOAD job triggers orphaned compound cleanup
FE-05  CompoundListItem schema has is_duplicate (bool) and parent_name (str|None)
FE-06  CompoundVersionItem schema has parent_name field
FE-07  ActiveJobResponse schema has input_params field
FE-08  Dead CompoundResponse schema removed from schemas.py
FE-09  Config: MAX_SMILES_LENGTH=5000, MAX_COMPOUND_NAME_LENGTH=255 match backend
FE-10  Frontend APP_VERSION updated to "2.2.0-dev"
FE-11  pyproject.toml version is "2.2.0-dev", test extras group exists
FE-12  CORS_ORIGINS default does not contain "*.hf.space" literal
FE-13  recover_on_startup: per-job db.commit() + db.rollback() on failure
FE-14  upload_worker: per-job get_db_session() (two sessions, not one batch session)
FE-15  compound_service.process_compound_job uses async with for all three httpx clients
FE-16  get_active_jobs() returns None on error (not empty list [])
FE-17  Session state: 4 navigation keys + dismissed_failed_jobs in _DEFAULT_FACTORIES
FE-18  Session state: evict_version_cache evicts when _MAX_VERSION_CACHE >= 20 keys
FE-19  Session state: evict_report_cache evicts when _MAX_REPORT_CACHE >= 5 keys
FE-20  Dead list_compounds() and poll_job_until_complete() removed from backend_client.py
FE-21  DUPLICATE_RESULT_KEY exported from duplicate_dialog.py; no render_duplicate_dialog
FE-22  AVAILABILITY_RESULT_KEY exported from availability_dialog.py
FE-23  compound_repository.get_compounds_paginated uses aliased self-join for parent_name
FE-24  validators._MockConfig limits match backend schema limits

Manual-only (require live Streamlit runtime):
  FE-M1  @st.dialog modal for duplicate_dialog renders correctly in browser
  FE-M2  @st.dialog modal for availability_dialog renders correctly in browser
  FE-M3  st.pills activity selection pre-selects all 7 types
  FE-M4  PENDING_UPLOAD jobs display as "completed" with View Results in sidebar
  FE-M5  Duplicate compound badge shows orange DUP badge on compound cards
  FE-M6  IMP score badge (border-left color) shows on compound grid/list rows
  FE-M7  Tab deep-linking via ?tab=report opens Report tab in compound_detail
  FE-M8  Dark mode: no illegible hardcoded colors in app.py/sidebar.py
  FE-M9  st.form wraps compound name + author name + submit button only
"""
import inspect
import re
from unittest.mock import patch
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# FE-01: PENDING_UPLOAD in get_active_jobs status filter
# ─────────────────────────────────────────────────────────────────────────────

class TestFE01PendingUploadInActiveJobsFilter:
    """FE-01: job_repository.get_active_jobs() must include PENDING_UPLOAD."""

    def test_job_repository_get_active_jobs_includes_pending_upload(self):
        """PENDING_UPLOAD must appear in the get_active_jobs status filter."""
        from backend.repositories.job_repository import JobRepository

        src = inspect.getsource(JobRepository.get_active_jobs)
        assert "PENDING_UPLOAD" in src, (
            "get_active_jobs() must include JobStatus.PENDING_UPLOAD in its "
            "status filter so uploading jobs appear in the active jobs list — FE-01"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-02: PENDING_UPLOAD counted as completed in batch summary SQL CASE
# ─────────────────────────────────────────────────────────────────────────────

class TestFE02BatchSummaryCountsPendingUploadAsCompleted:
    """FE-02: get_batch_summary must count PENDING_UPLOAD as completed."""

    def test_batch_summary_includes_pending_upload_in_completed_count(self):
        """get_batch_summary SQL CASE must include PENDING_UPLOAD alongside COMPLETED."""
        from backend.repositories.job_repository import JobRepository

        src = inspect.getsource(JobRepository.get_batch_summary)
        assert "PENDING_UPLOAD" in src, (
            "get_batch_summary() must count PENDING_UPLOAD jobs as completed "
            "in the SQL CASE expression — FE-02"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-03: mark_pending_upload step text is "Processing complete" not "Uploading results..."
# ─────────────────────────────────────────────────────────────────────────────

class TestFE03MarkPendingUploadStepText:
    """FE-03: mark_pending_upload must set current_step to 'Processing complete'."""

    def test_mark_pending_upload_uses_processing_complete_step_text(self):
        """mark_pending_upload must NOT use 'Uploading results...' step text (D-07)."""
        from backend.services.job_service import JobService

        src = inspect.getsource(JobService.mark_pending_upload)
        assert "Processing complete" in src, (
            "mark_pending_upload() must set current_step='Processing complete', "
            "not 'Uploading results...' — FE-03/D-07"
        )
        assert "Uploading results" not in src, (
            "mark_pending_upload() must not contain the old 'Uploading results...' "
            "step text — user-facing display must show completion — FE-03/D-07"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-04: cancel_job() PENDING_UPLOAD guard deletes orphaned compound
# ─────────────────────────────────────────────────────────────────────────────

class TestFE04CancelJobPendingUploadCleanup:
    """FE-04: cancel_job() must clean up orphaned compound on PENDING_UPLOAD cancel."""

    def test_cancel_job_has_pending_upload_cleanup_guard(self):
        """cancel_job() must contain a PENDING_UPLOAD branch for compound cleanup."""
        from backend.services.job_service import JobService

        src = inspect.getsource(JobService.cancel_job)
        assert "PENDING_UPLOAD" in src, (
            "cancel_job() must handle PENDING_UPLOAD status to delete the "
            "orphaned Compound row created by mark_pending_upload — FE-04/D-05"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-05: CompoundListItem has is_duplicate (bool) and parent_name (str|None)
# ─────────────────────────────────────────────────────────────────────────────

class TestFE05CompoundListItemComputedFields:
    """FE-05: CompoundListItem must expose is_duplicate and parent_name fields."""

    def test_compound_list_item_has_is_duplicate_field(self):
        """CompoundListItem must have is_duplicate: bool field."""
        from backend.models.schemas import CompoundListItem
        fields = CompoundListItem.model_fields
        assert "is_duplicate" in fields, (
            "CompoundListItem must declare is_duplicate field — FE-05/D-08"
        )
        # Verify it's a bool type (annotation check)
        annotation = fields["is_duplicate"].annotation
        assert annotation is bool or str(annotation) == "bool", (
            f"is_duplicate must be bool, got {annotation}"
        )

    def test_compound_list_item_has_parent_name_field(self):
        """CompoundListItem must have parent_name: str | None field."""
        from backend.models.schemas import CompoundListItem
        fields = CompoundListItem.model_fields
        assert "parent_name" in fields, (
            "CompoundListItem must declare parent_name field — FE-05/D-08"
        )

    def test_compound_list_item_is_duplicate_defaults_false(self):
        """CompoundListItem.is_duplicate defaults to False for root compounds."""
        from backend.models.schemas import CompoundListItem
        import uuid
        item = CompoundListItem(
            entry_id=uuid.uuid4(),
            compound_name="TestCompound",
            smiles="CCO",
            status="completed",
            created_at="2026-01-01T00:00:00Z",
        )
        assert item.is_duplicate is False, (
            "is_duplicate must default to False for root compounds — FE-05"
        )

    def test_compound_list_item_parent_name_defaults_none(self):
        """CompoundListItem.parent_name defaults to None for root compounds."""
        from backend.models.schemas import CompoundListItem
        import uuid
        item = CompoundListItem(
            entry_id=uuid.uuid4(),
            compound_name="TestCompound",
            smiles="CCO",
            status="completed",
            created_at="2026-01-01T00:00:00Z",
        )
        assert item.parent_name is None, (
            "parent_name must default to None for root compounds — FE-05"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-06: CompoundVersionItem has parent_name field
# ─────────────────────────────────────────────────────────────────────────────

class TestFE06CompoundVersionItemParentName:
    """FE-06: CompoundVersionItem must have parent_name field."""

    def test_compound_version_item_has_parent_name(self):
        """CompoundVersionItem must declare parent_name: str | None — FE-06/D-09."""
        from backend.models.schemas import CompoundVersionItem
        fields = CompoundVersionItem.model_fields
        assert "parent_name" in fields, (
            "CompoundVersionItem must have parent_name field — currently "
            "the frontend versions tab reads sib.get('parent_name') — FE-06/D-09"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-07: ActiveJobResponse has input_params field
# ─────────────────────────────────────────────────────────────────────────────

class TestFE07ActiveJobResponseInputParams:
    """FE-07: ActiveJobResponse must expose input_params for resubmission."""

    def test_active_job_response_has_input_params(self):
        """ActiveJobResponse must declare input_params: dict | None — FE-07/D-70."""
        from backend.models.schemas import ActiveJobResponse
        fields = ActiveJobResponse.model_fields
        assert "input_params" in fields, (
            "ActiveJobResponse must have input_params field so sidebar can "
            "populate author_name on resubmit — FE-07/D-70"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-08: Dead CompoundResponse schema removed from schemas.py
# ─────────────────────────────────────────────────────────────────────────────

class TestFE08CompoundResponseRemoved:
    """FE-08: CompoundResponse dead schema must be removed."""

    def test_compound_response_not_in_schemas_module(self):
        """CompoundResponse must NOT exist in backend.models.schemas — FE-08/D-74."""
        import backend.models.schemas as schemas_module
        assert not hasattr(schemas_module, "CompoundResponse"), (
            "CompoundResponse is dead code (no API consumers) and must be "
            "removed from schemas.py — FE-08/D-74"
        )

    def test_compound_response_not_in_backend_models_init(self):
        """CompoundResponse must not be re-exported from backend.models.__init__."""
        import backend.models as models_module
        assert not hasattr(models_module, "CompoundResponse"), (
            "CompoundResponse must not appear in backend.models.__all__ — FE-08"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-09: Frontend config limits match backend schema constraints
# ─────────────────────────────────────────────────────────────────────────────

class TestFE09FrontendConfigLimitsMatchBackend:
    """FE-09: Frontend config limits must match backend Pydantic field max_length."""

    def test_max_smiles_length_matches_backend(self):
        """MAX_SMILES_LENGTH must equal backend SmilesString max_length (5000) — FE-09/D-48."""
        from frontend.config.settings import ImpulatorConfig
        cfg = ImpulatorConfig()
        assert cfg.MAX_SMILES_LENGTH == 5000, (
            f"Frontend MAX_SMILES_LENGTH={cfg.MAX_SMILES_LENGTH} but backend "
            f"SmilesString max_length=5000 — FE-09/D-48"
        )

    def test_max_compound_name_length_matches_backend(self):
        """MAX_COMPOUND_NAME_LENGTH must equal backend CompoundName max_length (255) — FE-09/D-49."""
        from frontend.config.settings import ImpulatorConfig
        cfg = ImpulatorConfig()
        assert cfg.MAX_COMPOUND_NAME_LENGTH == 255, (
            f"Frontend MAX_COMPOUND_NAME_LENGTH={cfg.MAX_COMPOUND_NAME_LENGTH} "
            f"but backend CompoundName max_length=255 — FE-09/D-49"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-10: Frontend APP_VERSION = "2.2.0-dev"
# ─────────────────────────────────────────────────────────────────────────────

class TestFE10FrontendAppVersion:
    """FE-10: Frontend APP_VERSION must be updated to 2.2.0-dev."""

    def test_frontend_app_version_is_2_2_0_dev(self):
        """APP_VERSION must be '2.2.0-dev' in frontend config — FE-10/D-47."""
        from frontend.config.settings import ImpulatorConfig
        cfg = ImpulatorConfig()
        assert cfg.APP_VERSION == "2.2.0-dev", (
            f"Frontend APP_VERSION='{cfg.APP_VERSION}', expected '2.2.0-dev' — FE-10/D-47"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-11: pyproject.toml version = "2.2.0-dev" and test extras group exists
# ─────────────────────────────────────────────────────────────────────────────

class TestFE11PyprojectVersion:
    """FE-11: pyproject.toml must declare version 2.2.0-dev and test extras."""

    def test_pyproject_version_is_2_2_0_dev(self):
        """pyproject.toml version must be '2.2.0-dev' — FE-11/D-63."""
        import tomllib
        from pathlib import Path
        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        version = data["project"]["version"]
        assert version == "2.2.0-dev", (
            f"pyproject.toml version='{version}', expected '2.2.0-dev' — FE-11/D-63"
        )

    def test_pyproject_has_test_optional_dependencies(self):
        """pyproject.toml must have [project.optional-dependencies] test group — FE-11/D-62."""
        import tomllib
        from pathlib import Path
        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        assert "test" in optional_deps, (
            "pyproject.toml missing [project.optional-dependencies] test group — FE-11/D-62"
        )
        test_deps = optional_deps["test"]
        assert any("pytest" in dep for dep in test_deps), (
            "test extras group must contain pytest — FE-11/D-62"
        )

    def test_pytest_asyncio_not_in_main_dependencies(self):
        """pytest-asyncio must be in test extras, not main dependencies — FE-11/D-62."""
        import tomllib
        from pathlib import Path
        pyproject = Path(__file__).parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        main_deps = data.get("project", {}).get("dependencies", [])
        main_dep_names = [d.split(">=")[0].split("==")[0].strip().lower() for d in main_deps]
        assert "pytest-asyncio" not in main_dep_names, (
            "pytest-asyncio must be in test extras group, not main dependencies — FE-11/D-62"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-12: CORS_ORIGINS default does not contain "*.hf.space" literal
# ─────────────────────────────────────────────────────────────────────────────

class TestFE12CorsNoHfSpaceLiteral:
    """FE-12: *.hf.space must be removed from CORS_ORIGINS default (was a no-op)."""

    def test_cors_origins_default_excludes_hf_space_literal(self):
        """CORS_ORIGINS default must not contain '*.hf.space' — FE-12/D-72."""
        from backend import config as config_module
        src = inspect.getsource(config_module)
        assert "*.hf.space" not in src, (
            "'*.hf.space' is a no-op CORS origin (Starlette does exact matching "
            "not glob matching) and must be removed — FE-12/D-72"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-13: recover_on_startup uses per-job commit + rollback on failure
# ─────────────────────────────────────────────────────────────────────────────

class TestFE13RecoverOnStartupPerJobCommit:
    """FE-13: recover_on_startup must commit per job inside a try/except."""

    def test_recover_on_startup_has_per_job_try_except_with_rollback(self):
        """recover_on_startup must have db.rollback() for error isolation — FE-13/D-68."""
        from backend.services.job_service import JobService

        src = inspect.getsource(JobService.recover_on_startup)
        assert "db.rollback()" in src, (
            "recover_on_startup() must call db.rollback() on per-job failure "
            "to isolate errors — FE-13/D-68"
        )
        assert "db.commit()" in src, (
            "recover_on_startup() must commit per-job inside the loop — FE-13/D-68"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-14: upload_worker fetches IDs first, then opens per-job session
# ─────────────────────────────────────────────────────────────────────────────

class TestFE14UploadWorkerPerJobSession:
    """FE-14: _process_pending_uploads must use per-job get_db_session — D-69."""

    def test_upload_worker_has_two_get_db_session_calls(self):
        """_process_pending_uploads must call get_db_session() twice: once for IDs, once per job."""
        from backend.core import upload_worker

        src = inspect.getsource(upload_worker._process_pending_uploads)
        occurrences = src.count("get_db_session()")
        assert occurrences >= 2, (
            f"_process_pending_uploads must call get_db_session() at least twice "
            f"(once for ID fetch, once per job), found {occurrences} — FE-14/D-69"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-15: compound_service uses async with for all three httpx clients
# ─────────────────────────────────────────────────────────────────────────────

class TestFE15CompoundServiceAsyncWithHttpx:
    """FE-15: process_compound_job must use async with for all three httpx clients."""

    def test_compound_service_no_manual_aclose_calls(self):
        """compound_service must not call .aclose() manually on httpx clients — FE-15/D-52."""
        from backend.services import compound_service as cs_module

        src = inspect.getsource(cs_module)
        assert ".aclose()" not in src, (
            "compound_service.py must not call .aclose() manually — "
            "all httpx clients must use async with context managers — FE-15/D-52"
        )

    def test_compound_service_uses_async_with_for_httpx_clients(self):
        """process_compound_job must use async with create_*_client() — FE-15/D-52."""
        from backend.services.compound_service import process_compound_job

        src = inspect.getsource(process_compound_job)
        assert "async with" in src, (
            "process_compound_job must use async with for httpx clients — FE-15/D-52"
        )
        assert "create_chembl_client" in src, (
            "process_compound_job must use create_chembl_client — FE-15"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-16: get_active_jobs() returns None on error, not []
# ─────────────────────────────────────────────────────────────────────────────

class TestFE16GetActiveJobsReturnsNoneOnError:
    """FE-16: ImpulatorAPIClient.get_active_jobs() must return None on error, not []."""

    def test_get_active_jobs_source_has_no_empty_list_return_on_exception(self):
        """get_active_jobs must not return [] on RequestException — FE-16/D-28."""
        from frontend.services.backend_client import ImpulatorAPIClient

        src = inspect.getsource(ImpulatorAPIClient.get_active_jobs)
        # Look for "return []" patterns — these must not exist; only "return None" allowed
        lines = src.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "return []":
                pytest.fail(
                    f"get_active_jobs() has 'return []' at source line ~{i+1}. "
                    f"Must return None on error so callers can distinguish "
                    f"'empty' from 'unreachable' — FE-16/D-28"
                )

    def test_get_active_jobs_return_type_annotation_is_optional(self):
        """get_active_jobs return type must be Optional[list[dict]] not list[dict] — FE-16/D-28."""
        from frontend.services.backend_client import ImpulatorAPIClient

        hints = ImpulatorAPIClient.get_active_jobs.__annotations__
        return_annotation = hints.get("return", "")
        annotation_str = str(return_annotation)
        assert "None" in annotation_str or "Optional" in annotation_str, (
            f"get_active_jobs return annotation '{annotation_str}' must allow None "
            f"— FE-16/D-28"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-17: Session state _DEFAULT_FACTORIES has all 5 new keys
# ─────────────────────────────────────────────────────────────────────────────

class TestFE17SessionStateDefaultFactories:
    """FE-17: _DEFAULT_FACTORIES must include the 5 keys added in D-32/D-34."""

    REQUIRED_KEYS = [
        "selected_compound_entry_id",
        "selected_compound_storage_path",
        "selected_compound_is_duplicate",
        "selected_compound_duplicate_of_name",
        "dismissed_failed_jobs",
    ]

    def test_all_five_new_keys_present_in_default_factories(self):
        """All 5 new session state keys must be in _DEFAULT_FACTORIES — FE-17/D-32/D-34."""
        from frontend.utils.session_state import SessionState

        factories = SessionState._DEFAULT_FACTORIES
        for key in self.REQUIRED_KEYS:
            assert key in factories, (
                f"Key '{key}' missing from _DEFAULT_FACTORIES — FE-17"
            )

    def test_navigation_keys_default_to_none(self):
        """The 4 navigation keys must default to None — FE-17/D-32."""
        from frontend.utils.session_state import SessionState

        nav_keys = [
            "selected_compound_entry_id",
            "selected_compound_storage_path",
            "selected_compound_is_duplicate",
            "selected_compound_duplicate_of_name",
        ]
        factories = SessionState._DEFAULT_FACTORIES
        for key in nav_keys:
            val = factories[key]()
            assert val is None, (
                f"_DEFAULT_FACTORIES['{key}']() must return None, got {val!r} — FE-17/D-32"
            )

    def test_dismissed_failed_jobs_defaults_to_empty_set(self):
        """dismissed_failed_jobs must default to set() — FE-17/D-34."""
        from frontend.utils.session_state import SessionState

        val = SessionState._DEFAULT_FACTORIES["dismissed_failed_jobs"]()
        assert isinstance(val, set), (
            f"dismissed_failed_jobs factory must return set(), got {type(val).__name__} — FE-17/D-34"
        )
        assert len(val) == 0, (
            f"dismissed_failed_jobs factory must return empty set, got {val!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-18: evict_version_cache evicts entries when at or above _MAX_VERSION_CACHE
# ─────────────────────────────────────────────────────────────────────────────

class TestFE18VersionCacheEviction:
    """FE-18: evict_version_cache must evict oldest key when >= _MAX_VERSION_CACHE entries exist."""

    def test_evict_version_cache_removes_oldest_when_at_limit(self):
        """When 20 _versions_ keys exist, evict_version_cache must delete the oldest."""
        from frontend.utils.session_state import evict_version_cache, _MAX_VERSION_CACHE, SessionState

        assert _MAX_VERSION_CACHE == 20, (
            f"_MAX_VERSION_CACHE must be 20, got {_MAX_VERSION_CACHE} — FE-18/D-31"
        )

        # Patch _get_session_state to return a plain dict (Streamlit is installed,
        # so the ImportError fallback never fires — we must patch the method directly)
        mock_state = {f"_versions_compound_{i:02d}": f"data_{i}" for i in range(_MAX_VERSION_CACHE)}

        with patch.object(SessionState, "_get_session_state", return_value=mock_state):
            evict_version_cache()

        remaining_keys = [k for k in mock_state if k.startswith("_versions_")]
        assert len(remaining_keys) == _MAX_VERSION_CACHE - 1, (
            f"After eviction, must have {_MAX_VERSION_CACHE - 1} _versions_ keys, "
            f"got {len(remaining_keys)} — FE-18/D-31"
        )
        assert "_versions_compound_00" not in mock_state, (
            "Oldest (lexicographically first) version cache key must be evicted — FE-18/D-31"
        )

    def test_evict_version_cache_noop_when_below_limit(self):
        """evict_version_cache must not evict when fewer than _MAX_VERSION_CACHE keys exist."""
        from frontend.utils.session_state import evict_version_cache, _MAX_VERSION_CACHE, SessionState

        mock_state = {f"_versions_compound_{i}": f"data_{i}" for i in range(5)}

        with patch.object(SessionState, "_get_session_state", return_value=mock_state):
            evict_version_cache()

        remaining_keys = [k for k in mock_state if k.startswith("_versions_")]
        assert len(remaining_keys) == 5, (
            f"No eviction expected with 5 keys (limit={_MAX_VERSION_CACHE}), "
            f"got {len(remaining_keys)} — FE-18/D-31"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-19: evict_report_cache evicts when >= _MAX_REPORT_CACHE entries
# ─────────────────────────────────────────────────────────────────────────────

class TestFE19ReportCacheEviction:
    """FE-19: evict_report_cache must evict oldest key when >= _MAX_REPORT_CACHE entries exist."""

    def test_evict_report_cache_removes_oldest_when_at_limit(self):
        """When 5 _report_ keys exist, evict_report_cache must delete the oldest."""
        from frontend.utils.session_state import evict_report_cache, _MAX_REPORT_CACHE, SessionState

        assert _MAX_REPORT_CACHE == 5, (
            f"_MAX_REPORT_CACHE must be 5, got {_MAX_REPORT_CACHE} — FE-19/D-41"
        )

        mock_state = {f"_report_compound_{i:02d}": f"html_{i}" for i in range(_MAX_REPORT_CACHE)}

        with patch.object(SessionState, "_get_session_state", return_value=mock_state):
            evict_report_cache()

        remaining = [k for k in mock_state if k.startswith("_report_")]
        assert len(remaining) == _MAX_REPORT_CACHE - 1, (
            f"After eviction, must have {_MAX_REPORT_CACHE - 1} _report_ keys, "
            f"got {len(remaining)} — FE-19/D-41"
        )
        assert "_report_compound_00" not in mock_state, (
            "Oldest _report_ key must be evicted — FE-19/D-41"
        )

    def test_evict_report_cache_noop_when_below_limit(self):
        """evict_report_cache must not evict when fewer than _MAX_REPORT_CACHE keys exist."""
        from frontend.utils.session_state import evict_report_cache, _MAX_REPORT_CACHE, SessionState

        mock_state = {f"_report_compound_{i}": f"html_{i}" for i in range(3)}

        with patch.object(SessionState, "_get_session_state", return_value=mock_state):
            evict_report_cache()

        remaining = [k for k in mock_state if k.startswith("_report_")]
        assert len(remaining) == 3, (
            f"No eviction expected with 3 keys (limit={_MAX_REPORT_CACHE}), "
            f"got {len(remaining)} — FE-19/D-41"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-20: Dead list_compounds() and poll_job_until_complete() removed
# ─────────────────────────────────────────────────────────────────────────────

class TestFE20DeadCodeRemoved:
    """FE-20: list_compounds() and poll_job_until_complete() must be removed from ImpulatorAPIClient."""

    def test_list_compounds_method_removed(self):
        """ImpulatorAPIClient.list_compounds must not exist — FE-20/D-73."""
        from frontend.services.backend_client import ImpulatorAPIClient
        assert not hasattr(ImpulatorAPIClient, "list_compounds"), (
            "list_compounds() is dead code (replaced by get_compounds_from_db) "
            "and must be removed from ImpulatorAPIClient — FE-20/D-73"
        )

    def test_poll_job_until_complete_method_removed(self):
        """ImpulatorAPIClient.poll_job_until_complete must not exist — FE-20/D-53."""
        from frontend.services.backend_client import ImpulatorAPIClient
        assert not hasattr(ImpulatorAPIClient, "poll_job_until_complete"), (
            "poll_job_until_complete() blocks the Streamlit session thread and "
            "must be removed — FE-20/D-53"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-21: DUPLICATE_RESULT_KEY exported; render_duplicate_dialog removed
# ─────────────────────────────────────────────────────────────────────────────

class TestFE21DuplicateDialogModernized:
    """FE-21: duplicate_dialog.py must export DUPLICATE_RESULT_KEY; old render_ removed."""

    def test_duplicate_result_key_exported(self):
        """DUPLICATE_RESULT_KEY constant must be importable from duplicate_dialog — FE-21."""
        from frontend.ui.components.duplicate_dialog import DUPLICATE_RESULT_KEY
        assert isinstance(DUPLICATE_RESULT_KEY, str), (
            "DUPLICATE_RESULT_KEY must be a string constant — FE-21"
        )
        assert len(DUPLICATE_RESULT_KEY) > 0, (
            "DUPLICATE_RESULT_KEY must be a non-empty string — FE-21"
        )

    def test_render_duplicate_dialog_removed(self):
        """render_duplicate_dialog() returning a tuple must no longer exist — FE-21/D-16."""
        import frontend.ui.components.duplicate_dialog as dd
        assert not hasattr(dd, "render_duplicate_dialog"), (
            "render_duplicate_dialog() must be replaced by @st.dialog duplicate_dialog() — FE-21/D-16"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-22: AVAILABILITY_RESULT_KEY exported from availability_dialog.py
# ─────────────────────────────────────────────────────────────────────────────

class TestFE22AvailabilityDialogModernized:
    """FE-22: availability_dialog.py must export AVAILABILITY_RESULT_KEY."""

    def test_availability_result_key_exported(self):
        """AVAILABILITY_RESULT_KEY constant must be importable — FE-22/D-16."""
        from frontend.ui.components.availability_dialog import AVAILABILITY_RESULT_KEY
        assert isinstance(AVAILABILITY_RESULT_KEY, str), (
            "AVAILABILITY_RESULT_KEY must be a string constant — FE-22"
        )
        assert len(AVAILABILITY_RESULT_KEY) > 0

    def test_render_availability_dialog_removed(self):
        """render_availability_dialog() returning a tuple must no longer exist — FE-22/D-16."""
        import frontend.ui.components.availability_dialog as ad
        assert not hasattr(ad, "render_availability_dialog"), (
            "render_availability_dialog() must be replaced by @st.dialog availability_dialog() — FE-22/D-16"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-23: compound_repository.get_compounds_paginated uses aliased self-join
# ─────────────────────────────────────────────────────────────────────────────

class TestFE23CompoundRepositoryParentNameJoin:
    """FE-23: get_compounds_paginated must use aliased self-join for parent_name resolution."""

    def test_get_compounds_paginated_uses_aliased(self):
        """get_compounds_paginated source must import and use aliased() — FE-23/D-08."""
        from backend.repositories.compound_repository import CompoundRepository

        src = inspect.getsource(CompoundRepository.get_compounds_paginated)
        assert "aliased" in src, (
            "get_compounds_paginated must use sqlalchemy.orm.aliased() for "
            "self-join parent name resolution — FE-23/D-08"
        )

    def test_get_compounds_paginated_uses_outerjoin_for_parent(self):
        """get_compounds_paginated must use outerjoin for parent name resolution — FE-23/D-08."""
        from backend.repositories.compound_repository import CompoundRepository

        src = inspect.getsource(CompoundRepository.get_compounds_paginated)
        assert "outerjoin" in src, (
            "get_compounds_paginated must use outerjoin to resolve parent_name "
            "so root compounds still appear (not inner join) — FE-23/D-08"
        )

    def test_get_compounds_paginated_returns_tuple_with_parent_name(self):
        """get_compounds_paginated return type must include parent_name as second tuple element."""
        from backend.repositories.compound_repository import CompoundRepository

        src = inspect.getsource(CompoundRepository.get_compounds_paginated)
        assert "parent_name" in src, (
            "get_compounds_paginated must label parent name in the SELECT — FE-23/D-08"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FE-24: validators._MockConfig limits match backend schema limits
# ─────────────────────────────────────────────────────────────────────────────

class TestFE24ValidatorsMockConfigAlignment:
    """FE-24: validators.py _MockConfig limits must match backend schema max_length."""

    def test_validators_max_smiles_length_is_5000(self):
        """validators._MockConfig MAX_SMILES_LENGTH must be 5000 — FE-24."""
        import frontend.utils.validators as validators_module

        src = inspect.getsource(validators_module)
        # Find the _MockConfig block and check the value
        match = re.search(r"MAX_SMILES_LENGTH\s*=\s*(\d+)", src)
        assert match is not None, "MAX_SMILES_LENGTH not found in validators.py — FE-24"
        value = int(match.group(1))
        assert value == 5000, (
            f"validators._MockConfig.MAX_SMILES_LENGTH={value}, expected 5000 — FE-24"
        )

    def test_validators_max_compound_name_length_is_255(self):
        """validators._MockConfig MAX_COMPOUND_NAME_LENGTH must be 255 — FE-24."""
        import frontend.utils.validators as validators_module

        src = inspect.getsource(validators_module)
        match = re.search(r"MAX_COMPOUND_NAME_LENGTH\s*=\s*(\d+)", src)
        assert match is not None, "MAX_COMPOUND_NAME_LENGTH not found in validators.py — FE-24"
        value = int(match.group(1))
        assert value == 255, (
            f"validators._MockConfig.MAX_COMPOUND_NAME_LENGTH={value}, expected 255 — FE-24"
        )
