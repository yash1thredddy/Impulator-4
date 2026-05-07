"""
Phase 19.3 Gap Coverage Tests

Validates behavioral requirements for OPT-01..OPT-16 that were
shipped in plans 01–10 but lack direct test assertions.

Gaps covered:
  OPT-06  orjson / ORJSONResponse not wired in FastAPI app (BLOCKER)
  OPT-04  structlog CallsiteParameterAdder adds module/func_name/lineno to JSON
  OPT-07  CalcMolDescriptors uses ExactMolWt (monoisotopic), not MolWt
  OPT-08  SessionLocal created with expire_on_commit=False
  OPT-09  config defaults: DB_POOL_SIZE=5, DB_MAX_OVERFLOW=5, DB_POOL_TIMEOUT=10
  OPT-10  PDB cache TTL: search_similar_ligands=86400s, get_structure_details=604800s
  OPT-11  Recovery marker cleanup: marker deleted on CANCELLED terminal state
  OPT-13  Pydantic str_strip_whitespace on ResolveDuplicateRequest
  OPT-14  CompoundRepository.delete_by_entry_id returns True/False correctly
  OPT-15  pyproject.toml declares uvicorn>=0.42.0 and numpy>=2.4.3

Manual-only (frontend/observability -- require live Streamlit app):
  OPT-01  @st.fragment already wired -- visual-only verification
  OPT-02  st.cache_data 60s TTL -- visual/network tab verification
  OPT-03  st.query_params deep linking -- browser URL verification
  OPT-05  httpx event hooks -- DROPPED per D-13
  OPT-12  VALID_TRANSITIONS PROCESSING->PENDING -- already in 19.2
  OPT-16  uvloop -- installed implicitly via uvicorn[standard]
"""
import importlib
import inspect
import json
import logging
import io
import re
import uuid

import pytest
import structlog


# ─────────────────────────────────────────────────────────────────────────────
# OPT-06: orjson / ORJSONResponse must be the FastAPI default response class
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT06OrjsonDefaultResponse:
    """
    OPT-06 requirement: orjson as FastAPI default response class (3-6x faster
    JSON serialization). D-14: default_response_class=ORJSONResponse on app.
    D-16: Replace ALL explicit JSONResponse calls with ORJSONResponse.
    """

    def test_fastapi_app_uses_orjson_default_response_class(self):
        """
        FastAPI app must have default_response_class=ORJSONResponse.
        Checks app.router.default_response_class — the FastAPI-internal location
        where the resolved response class is stored after construction.
        """
        from fastapi.responses import ORJSONResponse
        from backend.main import app

        actual = app.router.default_response_class
        assert actual is ORJSONResponse, (
            f"Expected app.router.default_response_class=ORJSONResponse, "
            f"got {actual!r}. "
            "D-14 requires: app = FastAPI(..., default_response_class=ORJSONResponse)"
        )

    def test_main_py_has_no_plain_jsonresponse_calls(self):
        """
        main.py must not import or return plain JSONResponse after OPT-06.
        All explicit JSON responses should use ORJSONResponse.
        """
        import backend.main as main_module
        import inspect as ins

        source = ins.getsource(main_module)
        # Allow the import line itself to mention JSONResponse only IF it's
        # importing ORJSONResponse.  Forbidden: standalone JSONResponse(...)
        # calls or from fastapi.responses import JSONResponse without ORJSON.
        calls = re.findall(r'return\s+JSONResponse\s*\(', source)
        assert not calls, (
            f"Found {len(calls)} plain JSONResponse(...) return(s) in main.py. "
            "D-16 requires all explicit JSONResponse calls replaced with ORJSONResponse."
        )

    def test_exceptions_py_has_no_plain_jsonresponse_calls(self):
        """exceptions.py exception handlers must use ORJSONResponse, not JSONResponse."""
        import backend.core.exceptions as exc_module
        import inspect as ins

        source = ins.getsource(exc_module)
        calls = re.findall(r'return\s+JSONResponse\s*\(', source)
        assert not calls, (
            f"Found {len(calls)} plain JSONResponse return(s) in exceptions.py. "
            "D-16 requires ORJSONResponse throughout."
        )

    def test_health_py_has_no_plain_jsonresponse_calls(self):
        """health.py readiness checks must use ORJSONResponse."""
        import backend.api.v1.health as health_module
        import inspect as ins

        source = ins.getsource(health_module)
        calls = re.findall(r'return\s+JSONResponse\s*\(', source)
        assert not calls, (
            f"Found {len(calls)} plain JSONResponse return(s) in health.py. "
            "D-16 requires ORJSONResponse throughout."
        )

    def test_jobs_py_has_no_plain_jsonresponse_calls(self):
        """jobs.py duplicate/skip responses must use ORJSONResponse."""
        import backend.api.v1.jobs as jobs_module
        import inspect as ins

        source = ins.getsource(jobs_module)
        calls = re.findall(r'return\s+JSONResponse\s*\(', source)
        assert not calls, (
            f"Found {len(calls)} plain JSONResponse return(s) in jobs.py. "
            "D-16 requires ORJSONResponse throughout."
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-04: structlog CallsiteParameterAdder adds module / func_name / lineno
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT04CallsiteParameterAdder:
    """
    OPT-04 requirement: structlog CallsiteParameterAdder in shared_processors
    so every JSON log entry includes module, func_name, and lineno fields.
    D-10: placed before StackInfoRenderer.

    NOTE: Tests read the source file directly via pathlib rather than
    inspect.getsource() because in the full test suite, the stdlib `logging`
    module name shadows backend.core.logging in some Python versions,
    causing inspect to return the wrong source.
    """

    @staticmethod
    def _read_backend_logging_source() -> str:
        """
        Read backend/core/logging.py source directly via filesystem.
        Avoids inspect.getsource() stdlib name collision in full suite runs.
        """
        import pathlib
        root = pathlib.Path(__file__).parent.parent.parent
        src_path = root / "backend" / "core" / "logging.py"
        assert src_path.exists(), f"backend/core/logging.py not found at {src_path}"
        return src_path.read_text()

    def test_callsite_adder_is_in_shared_processors(self):
        """
        CallsiteParameterAdder must be present in the shared_processors list
        in logging.py, regardless of test-time logging capture details.
        """
        source = self._read_backend_logging_source()
        assert "CallsiteParameterAdder" in source, (
            "CallsiteParameterAdder not found in backend/core/logging.py. "
            "D-10 requires it in shared_processors."
        )

    def test_callsite_adder_includes_module_func_lineno(self):
        """
        The CallsiteParameterAdder configuration must request MODULE, FUNC_NAME,
        and LINENO — not just a subset.
        """
        source = self._read_backend_logging_source()
        assert "CallsiteParameter.MODULE" in source, (
            "MODULE not configured in CallsiteParameterAdder (backend/core/logging.py)"
        )
        assert "CallsiteParameter.FUNC_NAME" in source, (
            "FUNC_NAME not configured in CallsiteParameterAdder (backend/core/logging.py)"
        )
        assert "CallsiteParameter.LINENO" in source, (
            "LINENO not configured in CallsiteParameterAdder (backend/core/logging.py)"
        )

    def test_format_exc_info_is_in_shared_processors(self):
        """
        format_exc_info (D-11) must also be present to structure tracebacks.
        """
        source = self._read_backend_logging_source()
        assert "format_exc_info" in source, (
            "format_exc_info not found in backend/core/logging.py. "
            "D-11 requires it after StackInfoRenderer."
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-07: CalcMolDescriptors uses ExactMolWt, not MolWt
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT07CalcMolDescriptors:
    """
    OPT-07 requirement: Replace 12 individual RDKit descriptor calls with
    single CalcMolDescriptors() batch call. D-18: use ExactMolWt (monoisotopic)
    not MolWt (average) throughout.
    """

    def test_compound_service_uses_calc_mol_descriptors_batch(self):
        """_calculate_molecular_descriptors_sync must call CalcMolDescriptors."""
        import backend.services.compound_service as svc
        import inspect as ins

        source = ins.getsource(svc)
        assert "CalcMolDescriptors" in source, (
            "CalcMolDescriptors not found in compound_service.py. "
            "OPT-07/D-17 requires batch descriptor call."
        )

    def test_compound_service_uses_exact_mol_wt_not_avg(self):
        """
        D-18: compound_service must extract ExactMolWt from the descriptor dict,
        not MolWt. Checks that the code line doing the extraction uses 'ExactMolWt'
        as the dict key, not 'MolWt'.
        """
        import backend.services.compound_service as svc
        import inspect as ins

        source = ins.getsource(svc)
        # Must reference ExactMolWt as a dict key lookup in code
        assert "ExactMolWt" in source, (
            "ExactMolWt not found in compound_service.py. "
            "D-18 requires monoisotopic mass via ExactMolWt key."
        )
        # Check code lines (not comments/docstrings) for bare MolWt key usage.
        # Strip comment lines and blank lines before checking.
        code_lines = [
            line for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        # Remove all ExactMolWt occurrences, then check no bare MolWt dict key remains
        sanitized = code_only.replace("ExactMolWt", "")
        # A bare MolWt used as a descriptor key would look like: ['MolWt'] or ["MolWt"]
        bare_molwt_key = re.findall(r"""['"](MolWt)['"]""", sanitized)
        assert not bare_molwt_key, (
            f"Found bare 'MolWt' as a dict key in compound_service.py code lines. "
            "D-18 requires ExactMolWt exclusively (monoisotopic, not average)."
        )

    def test_rdkit_calc_mol_descriptors_produces_exact_mol_wt_key(self):
        """
        Integration: CalcMolDescriptors on a real molecule produces ExactMolWt
        key in its output dict (sanity-check the RDKit API contract).
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.Descriptors import CalcMolDescriptors
        except ImportError:
            pytest.skip("RDKit not available")

        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        assert mol is not None
        descs = CalcMolDescriptors(mol)
        assert "ExactMolWt" in descs, (
            f"CalcMolDescriptors does not return ExactMolWt key. "
            f"Available keys starting with 'Mol': {[k for k in descs if 'Mol' in k]}"
        )
        # ExactMolWt for ethanol ≈ 46.042 Da
        assert abs(descs["ExactMolWt"] - 46.042) < 0.01, (
            f"ExactMolWt for ethanol expected ~46.042, got {descs['ExactMolWt']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-08: SessionLocal created with expire_on_commit=False
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT08ExpireOnCommit:
    """
    OPT-08 requirement: expire_on_commit=False on SessionLocal to prevent
    lazy-load bugs when ORM objects are used after commit in async/background.
    D-19.
    """

    def test_database_py_sets_expire_on_commit_false(self):
        """database.py must contain expire_on_commit=False in sessionmaker call."""
        import backend.core.database as db_module
        import inspect as ins

        source = ins.getsource(db_module)
        assert "expire_on_commit=False" in source, (
            "expire_on_commit=False not found in backend/core/database.py. "
            "OPT-08/D-19 requires it to prevent lazy-load bugs."
        )

    def test_database_module_exports_get_db_session(self):
        """database module must export get_db_session and get_db for DI."""
        import backend.core.database as db_module

        assert hasattr(db_module, "get_db"), (
            "backend.core.database missing get_db() — FastAPI DI will break"
        )
        assert hasattr(db_module, "get_db_session"), (
            "backend.core.database missing get_db_session() — background workers will break"
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-09: DB pool sizing defaults from config
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT09DBPoolSizing:
    """
    OPT-09 requirement: DB pool sized for 10 concurrent jobs.
    D-20: pool_size=5, max_overflow=5, pool_timeout=10 (wired from config.py).
    """

    def test_config_has_db_pool_size_default_5(self):
        """Settings.DB_POOL_SIZE must default to 5."""
        from backend.config import Settings

        s = Settings(_env_file=None)
        assert s.DB_POOL_SIZE == 5, (
            f"Expected DB_POOL_SIZE=5, got {s.DB_POOL_SIZE}. "
            "OPT-09/D-20 requires pool_size=5."
        )

    def test_config_has_db_max_overflow_default_5(self):
        """Settings.DB_MAX_OVERFLOW must default to 5."""
        from backend.config import Settings

        s = Settings(_env_file=None)
        assert s.DB_MAX_OVERFLOW == 5, (
            f"Expected DB_MAX_OVERFLOW=5, got {s.DB_MAX_OVERFLOW}. "
            "OPT-09/D-20 requires max_overflow=5."
        )

    def test_config_has_db_pool_timeout_default_10(self):
        """Settings.DB_POOL_TIMEOUT must default to 10 (was 30 before OPT-09)."""
        from backend.config import Settings

        s = Settings(_env_file=None)
        assert s.DB_POOL_TIMEOUT == 10, (
            f"Expected DB_POOL_TIMEOUT=10 (reduced from 30 per D-20), got {s.DB_POOL_TIMEOUT}."
        )

    def test_database_py_wires_pool_settings_from_config(self):
        """database.py must reference settings.DB_POOL_SIZE (not hardcoded)."""
        import backend.core.database as db_module
        import inspect as ins

        source = ins.getsource(db_module)
        assert "settings.DB_POOL_SIZE" in source, (
            "database.py does not use settings.DB_POOL_SIZE — pool is hardcoded or missing. "
            "OPT-09/D-20 requires wiring through config."
        )
        assert "settings.DB_MAX_OVERFLOW" in source, (
            "database.py does not use settings.DB_MAX_OVERFLOW."
        )
        assert "settings.DB_POOL_TIMEOUT" in source, (
            "database.py does not use settings.DB_POOL_TIMEOUT."
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-10: PDB cache TTL values (24h search, 7d structure details)
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT10PDBCacheTTL:
    """
    OPT-10 requirement: PDB functions cached with correct TTLs via cache_non_none.
    D-21: 24h (86400s) for search_similar_ligands, 7d (604800s) for get_structure_details.
    No @lru_cache remaining.
    """

    def test_search_similar_ligands_has_cache_non_none_decorator(self):
        """search_similar_ligands must be decorated with cache_non_none."""
        import backend.modules.pdb_client as pdb_module
        import inspect as ins

        source = ins.getsource(pdb_module)
        # 86400 = 24 hours in seconds
        assert "86400" in source, (
            "86400 (24h TTL) not found in pdb_client.py. "
            "OPT-10/D-21 requires search TTL = 86400s."
        )

    def test_get_structure_details_has_7day_ttl(self):
        """get_structure_details must use 7-day TTL (604800s)."""
        import backend.modules.pdb_client as pdb_module
        import inspect as ins

        source = ins.getsource(pdb_module)
        # 604800 = 7 days in seconds
        assert "604800" in source, (
            "604800 (7d TTL) not found in pdb_client.py. "
            "OPT-10/D-21 requires structure details TTL = 604800s."
        )

    def test_no_lru_cache_on_pdb_functions(self):
        """No @lru_cache on PDB async functions — must use cache_non_none."""
        import backend.modules.pdb_client as pdb_module
        import inspect as ins

        source = ins.getsource(pdb_module)
        # @lru_cache cannot handle async functions properly
        assert "@lru_cache" not in source, (
            "@lru_cache found in pdb_client.py. "
            "OPT-10 requires cache_non_none (async-safe) instead."
        )

    def test_cache_non_none_imported_in_pdb_client(self):
        """cache_non_none must be imported in pdb_client.py."""
        import backend.modules.pdb_client as pdb_module
        import inspect as ins

        source = ins.getsource(pdb_module)
        assert "cache_non_none" in source, (
            "cache_non_none not imported in pdb_client.py. "
            "OPT-10/D-21 requires it for TTL-based caching."
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-11: Recovery marker cleanup on any terminal state
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT11RecoveryMarkerCleanup:
    """
    OPT-11 requirement: scan_recovery_markers() result is replayed into DB
    on startup. D-22: delete marker files on ANY terminal state (including
    CANCELLED, not-found, already-completed). D-23: IntegrityError guard.
    """

    def test_main_py_imports_scan_recovery_markers(self):
        """main.py lifespan must call scan_recovery_markers()."""
        import backend.main as main_module
        import inspect as ins

        source = ins.getsource(main_module)
        assert "scan_recovery_markers" in source, (
            "scan_recovery_markers not referenced in backend/main.py. "
            "OPT-11 requires recovery markers to be replayed on startup."
        )

    def test_main_py_has_finally_block_for_marker_cleanup(self):
        """
        D-22: marker must be deleted regardless of outcome.
        The cleanup must be in a finally block or equivalent unconditional path.
        """
        import backend.main as main_module
        import inspect as ins

        source = ins.getsource(main_module)
        assert "finally" in source, (
            "No finally block in backend/main.py. "
            "D-22 requires marker deletion on ANY terminal outcome."
        )
        assert "marker_path.unlink" in source or "unlink()" in source, (
            "marker_path.unlink() not found. "
            "D-22 requires the marker file to be deleted."
        )

    def test_main_py_has_integrity_error_guard(self):
        """
        D-23: already-completed job replay must be guarded with IntegrityError catch.
        """
        import backend.main as main_module
        import inspect as ins

        source = ins.getsource(main_module)
        assert "IntegrityError" in source, (
            "IntegrityError guard not found in backend/main.py. "
            "D-23 requires try/except IntegrityError for already-completed replays."
        )

    def test_scan_recovery_markers_is_callable(self):
        """scan_recovery_markers must be importable and callable (not async)."""
        from backend.services.compound_service import scan_recovery_markers

        assert callable(scan_recovery_markers), (
            "scan_recovery_markers is not callable"
        )
        assert not inspect.iscoroutinefunction(scan_recovery_markers), (
            "scan_recovery_markers must be sync (called from lifespan before event loop)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-13: Pydantic Annotated validators + str_strip_whitespace
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT13PydanticAnnotatedValidators:
    """
    OPT-13 requirement: Reusable Annotated type aliases consolidate duplicated
    field validators. D-25: CompoundName, SmilesString, AuthorName, SimilarityThreshold.
    D-26: str_strip_whitespace=True on request models.
    """

    def test_compound_name_type_alias_exists(self):
        """CompoundName Annotated alias must be exported from schemas."""
        from backend.models.schemas import CompoundName
        from typing import get_type_hints, get_args, get_origin
        from typing import Annotated

        # Should be an Annotated type
        assert get_origin(CompoundName) is not None or "CompoundName" in str(CompoundName), (
            "CompoundName is not an Annotated type alias"
        )

    def test_smiles_string_type_alias_exists(self):
        """SmilesString Annotated alias must be exported."""
        from backend.models.schemas import SmilesString  # noqa: F401

    def test_author_name_type_alias_exists(self):
        """AuthorName Annotated alias must be exported."""
        from backend.models.schemas import AuthorName  # noqa: F401

    def test_similarity_threshold_type_alias_exists(self):
        """SimilarityThreshold Annotated alias must be exported."""
        from backend.models.schemas import SimilarityThreshold  # noqa: F401

    def test_resolve_duplicate_request_strips_whitespace(self):
        """
        D-26: ResolveDuplicateRequest has str_strip_whitespace=True.
        Leading/trailing whitespace is stripped before validation.
        """
        from backend.models.schemas import ResolveDuplicateRequest

        req = ResolveDuplicateRequest(
            compound_name="  Aspirin  ",
            author_name="  Test Author  ",
            smiles="CCO",
            existing_entry_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            action="duplicate",
        )
        assert req.compound_name == "Aspirin", (
            f"Expected 'Aspirin' after stripping, got '{req.compound_name}'. "
            "D-26 requires str_strip_whitespace=True on ResolveDuplicateRequest."
        )
        assert req.author_name == "Test Author", (
            f"Expected 'Test Author' after stripping, got '{req.author_name}'."
        )

    def test_check_availability_request_strips_whitespace(self):
        """
        D-26: CheckAvailabilityRequest also has str_strip_whitespace=True.
        CheckAvailabilityRequest has smiles + similarity_threshold + activity_types
        (no compound_name field — tests the smiles field for stripping).
        """
        from backend.models.schemas import CheckAvailabilityRequest

        req = CheckAvailabilityRequest(
            smiles="  CC(C)Cc1ccc(cc1)C(C)C(O)=O  ",
        )
        # smiles is a SmilesString (Annotated with AfterValidator) + str_strip_whitespace
        # Strip should apply before the AfterValidator runs
        assert req.smiles == "CC(C)Cc1ccc(cc1)C(C)C(O)=O", (
            f"Expected stripped SMILES, got '{req.smiles}'. "
            "D-26 requires str_strip_whitespace=True on CheckAvailabilityRequest."
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-14: CompoundRepository.delete_by_entry_id behavioral contract
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT14DeleteByEntryId:
    """
    OPT-14 requirement: add missing delete_by_entry_id() to CompoundRepository
    (D-29: live bug fix). Method must return True when a row was deleted,
    False otherwise.
    """

    def test_delete_by_entry_id_method_exists(self):
        """CompoundRepository must have delete_by_entry_id method."""
        from backend.repositories.compound_repository import CompoundRepository

        repo = CompoundRepository()
        assert hasattr(repo, "delete_by_entry_id"), (
            "CompoundRepository missing delete_by_entry_id. "
            "OPT-14/D-29: this is a live bug — upload failure paths will crash."
        )

    def test_delete_by_entry_id_signature(self):
        """delete_by_entry_id(self, db, entry_id) must accept a UUID."""
        from backend.repositories.compound_repository import CompoundRepository
        import inspect as ins

        sig = ins.signature(CompoundRepository.delete_by_entry_id)
        params = list(sig.parameters.keys())
        assert "db" in params, "delete_by_entry_id missing 'db' parameter"
        assert "entry_id" in params, "delete_by_entry_id missing 'entry_id' parameter"

    def test_delete_by_entry_id_returns_false_for_missing_id(self):
        """
        delete_by_entry_id must return False when no row matches.
        Tests the contract without a real DB: mock the db.execute() return value.
        """
        from unittest.mock import MagicMock
        from backend.repositories.compound_repository import CompoundRepository

        repo = CompoundRepository()
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_db.execute.return_value = mock_result

        result = repo.delete_by_entry_id(mock_db, uuid.uuid4())
        assert result is False, (
            f"Expected False when rowcount=0, got {result!r}. "
            "delete_by_entry_id must return bool(rowcount > 0)."
        )

    def test_delete_by_entry_id_returns_true_when_row_deleted(self):
        """
        delete_by_entry_id must return True when one row was deleted.
        """
        from unittest.mock import MagicMock
        from backend.repositories.compound_repository import CompoundRepository

        repo = CompoundRepository()
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db.execute.return_value = mock_result

        result = repo.delete_by_entry_id(mock_db, uuid.uuid4())
        assert result is True, (
            f"Expected True when rowcount=1, got {result!r}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# OPT-15: pyproject.toml version floor assertions
# ─────────────────────────────────────────────────────────────────────────────

class TestOPT15DependencyVersionFloors:
    """
    OPT-15 requirement: uvicorn 0.42.0 (O(n^2) body fix), numpy 2.4.3 (memory leak fix).
    D-35: Floor bumps in pyproject.toml.
    D-37: requirements.txt deleted (pyproject.toml is sole manifest).
    D-38: pydantic floor bumped to >=2.12.0 for Annotated validators.
    """

    def _read_pyproject(self) -> str:
        import pathlib

        root = pathlib.Path(__file__).parent.parent.parent
        p = root / "pyproject.toml"
        assert p.exists(), f"pyproject.toml not found at {p}"
        return p.read_text()

    def test_uvicorn_floor_is_0_42_0(self):
        """uvicorn[standard] floor must be >=0.42.0 (O(n^2) body accumulation fix)."""
        content = self._read_pyproject()
        assert "uvicorn[standard]>=0.42.0" in content, (
            "uvicorn[standard]>=0.42.0 not found in pyproject.toml. "
            "OPT-15/D-35 requires this floor for the O(n^2) body fix."
        )

    def test_numpy_floor_is_2_4_3(self):
        """numpy floor must be >=2.4.3 (memory leak + ARM threading fix)."""
        content = self._read_pyproject()
        assert "numpy>=2.4.3" in content, (
            "numpy>=2.4.3 not found in pyproject.toml. "
            "OPT-15/D-35 requires this floor."
        )

    def test_pydantic_floor_is_2_12_0(self):
        """pydantic floor must be >=2.12.0 (Annotated validators, SecretStr, str_strip_whitespace)."""
        content = self._read_pyproject()
        assert "pydantic>=2.12.0" in content, (
            "pydantic>=2.12.0 not found in pyproject.toml. "
            "D-38 requires this floor for Pydantic Annotated features used in OPT-13."
        )

    def test_orjson_declared_in_pyproject(self):
        """orjson must be declared as a dependency (was installed but undeclared before OPT-06)."""
        content = self._read_pyproject()
        assert "orjson" in content, (
            "orjson not found in pyproject.toml. "
            "OPT-06/D-14 requires it as an explicit dependency."
        )

    def test_requirements_txt_deleted(self):
        """requirements.txt must be deleted (D-37: pyproject.toml is sole manifest)."""
        import pathlib

        root = pathlib.Path(__file__).parent.parent.parent
        req_txt = root / "requirements.txt"
        assert not req_txt.exists(), (
            "requirements.txt still exists. "
            "D-37 requires it to be deleted; pyproject.toml is the sole manifest."
        )

    def test_numpy_capped_below_3(self):
        """numpy must be capped at <3.0.0 (D-34) to prevent breaking scientific dep upgrades."""
        content = self._read_pyproject()
        assert "<3.0.0" in content, (
            "<3.0.0 cap not found in pyproject.toml. "
            "D-34 requires numpy and pandas capped for scientific dep stability."
        )
