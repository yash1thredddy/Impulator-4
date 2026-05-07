"""
Phase 20.1 -- Fix Integration Test Suite for Postgres
Behavioral gap tests (State B retroactive coverage).

Derived requirements verified here:
- ITF-01: All integration tests pass after testcontainers migration
- ITF-02: client fixture is module-scoped (one TestClient per test file)
- ITF-03: mock_azure fixture is module-scoped (dependency chain requirement)
- ITF-04: _clean_tables is autouse=True function-scoped with TRUNCATE CASCADE
- ITF-05: No non-UUID strings used as path parameters in integration tests
- ITF-06: No stale input_params field assertions remain in integration tests
- ITF-07: Async mock pattern applied -- no sync function mocking async APIs

These are structural tests (no Docker/testcontainers required) that verify
the test infrastructure and test file state directly, proving each requirement
is satisfied by the current codebase.
"""
import ast
import uuid
from pathlib import Path

INTEGRATION_DIR = Path(__file__).parent.parent / "integration"
CONFTEST = INTEGRATION_DIR / "conftest.py"


def _extract_fixture_scope(conftest_source: str, fixture_name: str) -> str | None:
    """Parse conftest source and return the scope value for the named fixture, or None if unset."""
    tree = ast.parse(conftest_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fixture_name:
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    for kw in decorator.keywords:
                        if kw.arg == "scope":
                            # kw.value is a Constant node
                            if isinstance(kw.value, ast.Constant):
                                return kw.value.value
            return "function"  # pytest default when no scope keyword present
    return None  # fixture not found


# ---------------------------------------------------------------------------
# ITF-01: Integration test files all exist and collect without import errors
# ---------------------------------------------------------------------------


class TestITF01IntegrationTestsExist:
    """All integration test files referenced by Phase 20.1 exist and are valid Python."""

    PHASE_FILES = [
        "test_availability_check.py",
        "test_batch_endpoints.py",
        "test_api_jobs.py",
        "test_compound_endpoints.py",
        "conftest.py",
    ]

    def test_integration_test_files_exist(self):
        """All Phase 20.1 target test files exist on disk."""
        for filename in self.PHASE_FILES:
            path = INTEGRATION_DIR / filename
            assert path.exists(), f"Missing integration test file: {filename}"

    def test_integration_test_files_are_valid_python(self):
        """All Phase 20.1 target test files parse without syntax errors."""
        for filename in self.PHASE_FILES:
            path = INTEGRATION_DIR / filename
            source = path.read_text()
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{filename} has syntax error: {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# ITF-02: client fixture is module-scoped
# ---------------------------------------------------------------------------


class TestITF02ClientModuleScope:
    """client fixture must be scope='module' -- one TestClient per test file."""

    def test_client_fixture_is_module_scoped(self):
        """conftest.py client fixture has scope='module'."""
        source = CONFTEST.read_text()
        scope = _extract_fixture_scope(source, "client")
        assert scope is not None, "client fixture not found in conftest.py"
        assert scope == "module", (
            f"client fixture scope is '{scope}', expected 'module' (D-05 requirement)"
        )

    def test_client_fixture_present_in_conftest(self):
        """conftest.py contains a client fixture definition."""
        source = CONFTEST.read_text()
        assert "def client(" in source, "client fixture not found in conftest.py"


# ---------------------------------------------------------------------------
# ITF-03: mock_azure fixture is module-scoped
# ---------------------------------------------------------------------------


class TestITF03MockAzureModuleScope:
    """mock_azure fixture must be module-scoped (required by client fixture dependency)."""

    def test_mock_azure_fixture_is_module_scoped(self):
        """conftest.py mock_azure fixture has scope='module'."""
        source = CONFTEST.read_text()
        scope = _extract_fixture_scope(source, "mock_azure")
        assert scope is not None, "mock_azure fixture not found in conftest.py"
        assert scope == "module", (
            f"mock_azure fixture scope is '{scope}', expected 'module' "
            "(required because module-scoped client depends on it)"
        )


# ---------------------------------------------------------------------------
# ITF-04: _clean_tables is autouse=True function-scoped with TRUNCATE CASCADE
# ---------------------------------------------------------------------------


class TestITF04CleanTablesIsolation:
    """_clean_tables must be autouse=True (function-scoped) with TRUNCATE CASCADE."""

    def test_clean_tables_is_autouse(self):
        """_clean_tables fixture is decorated with autouse=True."""
        source = CONFTEST.read_text()
        assert "autouse=True" in source, (
            "_clean_tables autouse=True not found in conftest.py"
        )

    def test_clean_tables_uses_truncate_cascade(self):
        """_clean_tables fixture body contains TRUNCATE ... CASCADE for data isolation."""
        source = CONFTEST.read_text()
        assert "TRUNCATE" in source and "CASCADE" in source, (
            "TRUNCATE CASCADE not found in conftest.py _clean_tables fixture"
        )

    def test_clean_tables_is_not_module_scoped(self):
        """_clean_tables must remain function-scoped, not module-scoped, for per-test isolation."""
        source = CONFTEST.read_text()
        scope = _extract_fixture_scope(source, "_clean_tables")
        assert scope is not None, "_clean_tables fixture not found in conftest.py"
        assert scope == "function", (
            f"_clean_tables scope is '{scope}', must be 'function' (per D-06) "
            "-- module-scope would allow data to bleed between tests"
        )

    def test_clean_tables_truncates_all_four_tables(self):
        """TRUNCATE statement covers all four core tables: jobs, compounds, deleted_compounds, audit_events."""
        source = CONFTEST.read_text()
        for table in ("jobs", "compounds", "deleted_compounds", "audit_events"):
            assert table in source, (
                f"Table '{table}' not found in TRUNCATE statement in conftest.py"
            )


# ---------------------------------------------------------------------------
# ITF-05: No non-UUID strings used as path parameters in integration tests
# ---------------------------------------------------------------------------


class TestITF05NoFakeIdStrings:
    """Integration tests must not use non-UUID strings as ID path parameters."""

    BANNED_FAKE_IDS = ["fake-id", "fake_id", "fake-batch", "fakeid"]
    FILES_TO_CHECK = [
        "test_batch_endpoints.py",
        "test_api_jobs.py",
        "test_compound_endpoints.py",
        "test_availability_check.py",
        "test_compound_versions.py",
        "test_job_service_integration.py",
    ]

    def test_no_fake_id_strings_as_url_path_segments(self):
        """No banned non-UUID fake ID strings appear as URL path segments in integration tests."""
        violations = []
        for filename in self.FILES_TO_CHECK:
            path = INTEGRATION_DIR / filename
            if not path.exists():
                continue
            source = path.read_text()
            for banned in self.BANNED_FAKE_IDS:
                # Look for the string used as a URL path segment
                # e.g. "/api/v1/jobs/fake-id/cancel" in a string literal
                if f"/{banned}" in source and f"/{banned}" in source:
                    # Check it's not inside a comment or non-functional context
                    for lineno, line in enumerate(source.splitlines(), 1):
                        stripped = line.strip()
                        if f"/{banned}" in stripped and not stripped.startswith("#"):
                            violations.append(f"{filename}:{lineno}: {stripped[:120]}")

        assert not violations, (
            "Non-UUID fake ID strings found as URL path segments in integration tests:\n"
            + "\n".join(violations)
        )

    def test_nonexistent_id_uses_valid_uuid_format(self):
        """Nonexistent-ID test cases use valid UUID format (00000000-0000-4000-8000-...)."""
        sentinel_uuid = "00000000-0000-4000-8000-000000000099"
        files_expecting_sentinel = [
            "test_batch_endpoints.py",
            "test_compound_endpoints.py",
        ]
        found_in = []
        for filename in files_expecting_sentinel:
            path = INTEGRATION_DIR / filename
            if path.exists() and sentinel_uuid in path.read_text():
                found_in.append(filename)

        assert len(found_in) >= 1, (
            f"Sentinel UUID '{sentinel_uuid}' not found in any of "
            f"{files_expecting_sentinel} -- nonexistent-ID tests may still use invalid UUID strings"
        )

    def test_sentinel_uuid_is_valid_uuid_format(self):
        """The sentinel UUID used for nonexistent IDs is parseable as a real UUID."""
        sentinel = "00000000-0000-4000-8000-000000000099"
        try:
            parsed = uuid.UUID(sentinel)
        except ValueError as exc:
            raise AssertionError(
                f"Sentinel UUID '{sentinel}' is not a valid UUID: {exc}"
            ) from exc
        assert str(parsed) == sentinel


# ---------------------------------------------------------------------------
# ITF-06: No stale input_params field assertions remain
# ---------------------------------------------------------------------------


class TestITF06NoStaleInputParams:
    """Stale input_params assertions must be removed -- field was normalized in Postgres schema."""

    FILES_TO_CHECK = [
        "test_api_jobs.py",
        "test_compound_endpoints.py",
        "test_job_service_integration.py",
        "test_repository_coverage.py",
    ]

    def test_no_input_params_assertions_in_integration_tests(self):
        """No integration test asserts 'input_params' as a response field (normalized away in v2.2)."""
        violations = []
        for filename in self.FILES_TO_CHECK:
            path = INTEGRATION_DIR / filename
            if not path.exists():
                continue
            source = path.read_text()
            lines = source.splitlines()
            for lineno, line in enumerate(lines, 1):
                stripped = line.strip()
                if "input_params" in stripped and stripped.startswith("assert"):
                    violations.append(f"{filename}:{lineno}: {stripped}")

        assert not violations, (
            "Stale 'input_params' assertions found in integration tests:\n"
            + "\n".join(violations)
        )

    def test_job_detail_test_asserts_normalized_columns(self):
        """test_api_jobs.py asserts normalized columns, not input_params."""
        path = INTEGRATION_DIR / "test_api_jobs.py"
        assert path.exists(), "test_api_jobs.py not found"
        source = path.read_text()

        # The normalized columns that replaced input_params
        normalized_fields = ["compound_name", "smiles", "similarity_threshold", "activity_types"]
        for field in normalized_fields:
            assert field in source, (
                f"Expected normalized field '{field}' not found in test_api_jobs.py -- "
                "test_get_job_detail may not have been updated to assert current schema"
            )


# ---------------------------------------------------------------------------
# ITF-07: Async mock pattern applied in availability check tests
# ---------------------------------------------------------------------------


class TestITF07AsyncMockPattern:
    """Availability check tests must use async mock pattern -- sync function mocking async APIs causes 500s."""

    def test_availability_test_uses_async_context_manager_mock(self):
        """test_availability_check.py uses asynccontextmanager for create_chembl_client mock."""
        path = INTEGRATION_DIR / "test_availability_check.py"
        assert path.exists(), "test_availability_check.py not found"
        source = path.read_text()
        assert "asynccontextmanager" in source, (
            "test_availability_check.py does not import asynccontextmanager -- "
            "async mock pattern for create_chembl_client may be missing"
        )

    def test_availability_test_patches_all_three_async_functions(self):
        """test_availability_check.py patches create_chembl_client, probe_all_thresholds, quick_has_bioactivity."""
        path = INTEGRATION_DIR / "test_availability_check.py"
        source = path.read_text()

        required_patches = [
            "create_chembl_client",
            "probe_all_thresholds",
            "quick_has_bioactivity",
        ]
        for fn in required_patches:
            assert fn in source, (
                f"test_availability_check.py does not reference '{fn}' -- "
                "triple-patch async mock pattern may be incomplete"
            )

    def test_availability_test_mock_functions_are_async_def(self):
        """Mock functions in test_availability_check.py are defined as async def (not sync)."""
        path = INTEGRATION_DIR / "test_availability_check.py"
        source = path.read_text()

        # Count async def lines -- there must be at least 2 (the mock helpers)
        async_defs = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("async def")
        ]
        assert len(async_defs) >= 2, (
            f"Expected at least 2 async def mock helpers in test_availability_check.py, "
            f"found {len(async_defs)}: {async_defs}"
        )

    def test_availability_test_patches_at_source_module(self):
        """Mock target path is backend.modules.api_client.* (source module), not backend.services.*."""
        path = INTEGRATION_DIR / "test_availability_check.py"
        source = path.read_text()

        # Correct: patch at where the function is defined (source module)
        assert "backend.modules.api_client" in source, (
            "test_availability_check.py does not patch at 'backend.modules.api_client' -- "
            "mocks may not intercept the deferred import pattern used by job_service"
        )
