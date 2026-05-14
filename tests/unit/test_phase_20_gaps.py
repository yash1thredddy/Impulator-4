"""
Phase 20 — Test Infrastructure Gap Coverage

Behavioral tests verifying that Phase 20 (test-infrastructure) requirements
are actually met by the current repo state.

Requirements covered:
  TST-01  Root conftest provides Postgres-backed fixtures (no SQLite fallback)
  TST-02  CI workflow has Postgres service container + Alembic migration step
  TST-03  Integration test suite exists with real DB tests
  TST-04  Coverage gate is enforced at >=70% in CI
  TST-05  Golden fixture file is valid; IMP scoring module is importable

These are structural/contract tests — they can fail if the infra is removed
or regressed. They do NOT require a live Postgres connection.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent


def _read_ci_workflow() -> dict:
    ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), f"CI workflow not found: {ci_path}"
    with open(ci_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# TST-01: Root conftest provides Postgres fixtures — no SQLite fallback
# ---------------------------------------------------------------------------


class TestRootConftestPostgresFixtures:
    """TST-01: Tests/conftest.py must be Postgres-only with CI-aware fixture."""

    def test_root_conftest_exists(self):
        """Root conftest.py must be present."""
        conftest = REPO_ROOT / "tests" / "conftest.py"
        assert conftest.exists(), "tests/conftest.py is missing"

    def test_postgres_url_fixture_defined(self):
        """Root conftest must define a postgres_url fixture."""
        conftest_text = (REPO_ROOT / "tests" / "conftest.py").read_text()
        assert "def postgres_url" in conftest_text, (
            "postgres_url fixture not found in tests/conftest.py"
        )

    def test_pg_engine_fixture_defined(self):
        """Root conftest must define a pg_engine fixture with Alembic upgrade."""
        conftest_text = (REPO_ROOT / "tests" / "conftest.py").read_text()
        assert "def pg_engine" in conftest_text, (
            "pg_engine fixture not found in tests/conftest.py"
        )
        assert "alembic" in conftest_text.lower(), (
            "pg_engine fixture must run Alembic migrations"
        )

    def test_no_sqlite_create_engine_in_root_conftest(self):
        """Root conftest must not create SQLite engines."""
        conftest_text = (REPO_ROOT / "tests" / "conftest.py").read_text()
        assert "sqlite" not in conftest_text.lower(), (
            "Root conftest still references SQLite — must be Postgres-only"
        )

    def test_ci_aware_postgres_url_reads_test_database_url(self):
        """postgres_url fixture must detect CI via TEST_DATABASE_URL env var."""
        conftest_text = (REPO_ROOT / "tests" / "conftest.py").read_text()
        assert "TEST_DATABASE_URL" in conftest_text, (
            "postgres_url fixture must read TEST_DATABASE_URL for CI detection"
        )

    def test_testcontainers_in_dependencies(self):
        """testcontainers[postgres] must be declared as a test dependency."""
        pyproject = REPO_ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml is missing"
        pyproject_text = pyproject.read_text()
        assert "testcontainers" in pyproject_text, (
            "testcontainers dependency not found in pyproject.toml"
        )

    def test_integration_conftest_uses_pg_engine_from_root(self):
        """Integration conftest must reuse pg_engine from root — no local engine."""
        int_conftest = REPO_ROOT / "tests" / "integration" / "conftest.py"
        assert int_conftest.exists(), "tests/integration/conftest.py is missing"
        conftest_text = int_conftest.read_text()
        # Must use pg_engine (from root conftest), not create its own engine
        assert "pg_engine" in conftest_text, (
            "Integration conftest must use pg_engine fixture from root conftest"
        )
        # Must not create inline SQLite engines
        assert "sqlite" not in conftest_text.lower(), (
            "Integration conftest must not reference SQLite"
        )

    def test_integration_conftest_has_truncate_cascade_cleanup(self):
        """Integration conftest must have TRUNCATE CASCADE for per-test isolation."""
        conftest_text = (REPO_ROOT / "tests" / "integration" / "conftest.py").read_text()
        assert "TRUNCATE" in conftest_text, (
            "Integration conftest must TRUNCATE tables for per-test isolation"
        )
        assert "CASCADE" in conftest_text, (
            "TRUNCATE must use CASCADE to handle FK dependencies"
        )


# ---------------------------------------------------------------------------
# TST-02: CI workflow has Postgres service container + Alembic migration step
# ---------------------------------------------------------------------------


class TestCIWorkflowPostgresConfig:
    """TST-02: .github/workflows/ci.yml must have Postgres service + Alembic step."""

    def test_ci_workflow_exists(self):
        """CI workflow file must exist."""
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        assert ci_path.exists(), ".github/workflows/ci.yml is missing"

    def test_ci_has_postgres_service(self):
        """CI workflow must define a Postgres 15 service container."""
        ci = _read_ci_workflow()
        jobs = ci.get("jobs", {})
        found_postgres = False
        for job_name, job_def in jobs.items():
            services = job_def.get("services", {})
            for svc_name, svc_def in services.items():
                image = svc_def.get("image", "")
                if "postgres" in image:
                    found_postgres = True
        assert found_postgres, (
            "CI workflow has no Postgres service container — TST-02 requires "
            "a live Postgres service for integration tests"
        )

    def test_ci_postgres_has_health_check(self):
        """CI Postgres service must have pg_isready health check."""
        ci = _read_ci_workflow()
        jobs = ci.get("jobs", {})
        found_health = False
        for job_name, job_def in jobs.items():
            services = job_def.get("services", {})
            for svc_name, svc_def in services.items():
                options = svc_def.get("options", "")
                if "pg_isready" in str(options):
                    found_health = True
        assert found_health, (
            "CI Postgres service has no pg_isready health check — tests may "
            "start before Postgres is ready"
        )

    def test_ci_has_alembic_migration_step(self):
        """CI workflow must run 'alembic upgrade head' before tests."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "alembic upgrade head" in ci_text, (
            "CI workflow has no Alembic migration step — schema won't be "
            "provisioned before integration tests run"
        )

    def test_ci_alembic_step_uses_database_url(self):
        """The Alembic migration step must set DATABASE_URL env var."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        # Alembic env.py reads DATABASE_URL — verify CI provides it
        assert "DATABASE_URL" in ci_text, (
            "CI workflow must set DATABASE_URL for the Alembic step"
        )

    def test_ci_test_step_uses_test_database_url(self):
        """The test step must use TEST_DATABASE_URL (not production DATABASE_URL)."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "TEST_DATABASE_URL" in ci_text, (
            "CI test step must set TEST_DATABASE_URL — conftest reads this "
            "to route to the test Postgres, not real Supabase"
        )

    def test_ci_no_sqlite_references(self):
        """CI workflow must not configure or reference SQLite."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "sqlite" not in ci_text.lower(), (
            "CI workflow references SQLite — must be Postgres-only"
        )


# ---------------------------------------------------------------------------
# TST-03: Integration test suite exists with real DB tests
# ---------------------------------------------------------------------------


class TestIntegrationSuiteExists:
    """TST-03: Integration tests must exist and be collectable."""

    def test_integration_directory_exists(self):
        """tests/integration/ directory must exist."""
        int_dir = REPO_ROOT / "tests" / "integration"
        assert int_dir.is_dir(), "tests/integration/ directory is missing"

    def test_alembic_baseline_test_exists(self):
        """Alembic baseline verification test (SC-6) must exist."""
        test_file = REPO_ROOT / "tests" / "integration" / "test_alembic_baseline.py"
        assert test_file.exists(), (
            "tests/integration/test_alembic_baseline.py is missing — "
            "SC-6 (schema verification) has no coverage"
        )

    def test_reparent_trigger_test_exists(self):
        """Reparent trigger test (SC-7) must exist."""
        test_file = REPO_ROOT / "tests" / "integration" / "test_reparent_trigger.py"
        assert test_file.exists(), (
            "tests/integration/test_reparent_trigger.py is missing — "
            "SC-7 (compound versioning trigger) has no coverage"
        )

    def test_fault_tolerance_test_exists(self):
        """Fault tolerance regression tests (SC-9) must exist."""
        test_file = REPO_ROOT / "tests" / "integration" / "test_fault_tolerance.py"
        assert test_file.exists(), (
            "tests/integration/test_fault_tolerance.py is missing — "
            "SC-9 (fault tolerance) has no coverage"
        )

    def test_alembic_baseline_test_has_sufficient_methods(self):
        """Alembic baseline test must verify all 4 tables + ENUMs + indexes."""
        test_text = (
            REPO_ROOT / "tests" / "integration" / "test_alembic_baseline.py"
        ).read_text()
        # Plan 02 required 7 test methods
        test_methods = [
            line for line in test_text.splitlines() if line.strip().startswith("def test_")
        ]
        assert len(test_methods) >= 6, (
            f"test_alembic_baseline.py has only {len(test_methods)} test methods — "
            "expected at least 6 to cover tables, ENUMs, indexes, triggers, alembic.ini"
        )

    def test_fault_tolerance_test_has_multiple_scenarios(self):
        """Fault tolerance test must cover multiple SC-9 scenarios."""
        test_text = (
            REPO_ROOT / "tests" / "integration" / "test_fault_tolerance.py"
        ).read_text()
        test_methods = [
            line for line in test_text.splitlines() if line.strip().startswith("def test_")
        ]
        assert len(test_methods) >= 6, (
            f"test_fault_tolerance.py has only {len(test_methods)} test methods — "
            "expected at least 6 to cover scheduler, duplicates, cancel race, pool exhaustion"
        )

    def test_integration_tests_are_collectable(self):
        """pytest must be able to collect integration tests without syntax errors."""
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/integration/",
                "--collect-only",
                "-q",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        # Check that collection succeeded (exit code 0 = success, 5 = no tests found)
        assert result.returncode not in (1, 2, 3, 4), (
            f"Integration test collection failed:\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        # Must collect at least some tests
        collected_line = [
            line for line in result.stdout.splitlines() if "collected" in line
        ]
        assert collected_line, "No 'collected' line in pytest output"
        # Parse count
        import re
        match = re.search(r"(\d+) test", collected_line[-1])
        assert match, f"Could not parse test count from: {collected_line[-1]}"
        count = int(match.group(1))
        assert count >= 10, (
            f"Only {count} integration tests collected — expected at least 10"
        )


# ---------------------------------------------------------------------------
# TST-04: Coverage gate enforced at >=70% in CI (user override from 85%)
# ---------------------------------------------------------------------------


class TestCoveragGateEnforced:
    """TST-04: CI must enforce a coverage gate; gate must be >=70%."""

    def test_coverage_gate_present_in_ci(self):
        """CI workflow must have --cov-fail-under flag."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "--cov-fail-under" in ci_text, (
            "CI workflow has no --cov-fail-under flag — coverage gate is not enforced"
        )

    def test_coverage_gate_is_at_least_70(self):
        """Coverage gate must be >=70% (user-approved override from plan's 85%)."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        import re
        match = re.search(r"--cov-fail-under[=\s]+(\d+)", ci_text)
        assert match, "Could not parse --cov-fail-under value from CI workflow"
        gate = int(match.group(1))
        assert gate >= 70, (
            f"Coverage gate is {gate}% — must be at least 70% (user override). "
            "The plan originally specified 85%; 70% is the approved minimum."
        )

    def test_coverage_gate_is_not_trivially_low(self):
        """Coverage gate must not be set below 50% (sanity bound)."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        import re
        match = re.search(r"--cov-fail-under[=\s]+(\d+)", ci_text)
        assert match, "Could not parse --cov-fail-under value from CI workflow"
        gate = int(match.group(1))
        assert gate >= 50, (
            f"Coverage gate is only {gate}% — this is too low to be meaningful"
        )

    def test_coverage_target_is_backend(self):
        """Coverage measurement must target the 'backend' package."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "--cov=backend" in ci_text, (
            "CI workflow does not measure coverage for the 'backend' package"
        )

    def test_ruff_lint_step_present_in_ci(self):
        """CI workflow must run ruff lint (TST-04 requires 'ruff passes clean')."""
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "ruff" in ci_text, (
            "CI workflow has no ruff lint step — TST-04 requires ruff to pass"
        )


# ---------------------------------------------------------------------------
# TST-05: Golden fixture file valid; IMP scoring importable; 86 tests collect
# ---------------------------------------------------------------------------


class TestGoldenFixturesAndIMPScoring:
    """TST-05: Golden fixtures must be valid; IMP scoring module must be importable."""

    def test_golden_fixture_file_exists(self):
        """Golden fixture file must exist at tests/fixtures/golden_compounds.json."""
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "golden_compounds.json"
        assert fixture_path.exists(), (
            "tests/fixtures/golden_compounds.json is missing — "
            "IMP scoring golden regression tests have no fixture data"
        )

    def test_golden_fixture_is_valid_json(self):
        """Golden fixture file must be valid JSON."""
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "golden_compounds.json"
        try:
            data = json.loads(fixture_path.read_text())
        except json.JSONDecodeError as e:
            pytest.fail(f"golden_compounds.json is not valid JSON: {e}")
        assert isinstance(data, dict), "Golden fixture root must be a JSON object"

    def test_golden_fixture_has_compounds_key(self):
        """Golden fixture must have a 'compounds' list."""
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "golden_compounds.json"
        data = json.loads(fixture_path.read_text())
        assert "compounds" in data, (
            "golden_compounds.json has no 'compounds' key — "
            "fixture schema is wrong or file is empty"
        )
        compounds = data["compounds"]
        assert isinstance(compounds, list), "'compounds' must be a JSON array"
        assert len(compounds) >= 1, (
            "'compounds' array is empty — golden fixtures provide no regression coverage"
        )

    def test_golden_fixture_compound_has_expected_score_fields(self):
        """Each golden compound must have 'name', 'input', and 'expected' fields.

        Fixture schema uses nested structure:
          { "name": "...", "input": {...}, "expected": { "IMP_Final_Score": ..., ... } }
        """
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "golden_compounds.json"
        data = json.loads(fixture_path.read_text())
        compounds = data["compounds"]
        for i, compound in enumerate(compounds):
            assert "name" in compound, (
                f"Compound at index {i} has no 'name' field"
            )
            assert "expected" in compound, (
                f"Compound '{compound.get('name', i)}' has no 'expected' block — "
                "golden fixture cannot validate IMP scoring accuracy"
            )
            expected = compound["expected"]
            assert "IMP_Final_Score" in expected, (
                f"Compound '{compound.get('name', i)}' expected block has no "
                "'IMP_Final_Score' — 4dp accuracy requirement (TST-05) is untestable"
            )

    def test_imp_scoring_test_file_exists(self):
        """tests/unit/test_imp_scoring.py must exist."""
        test_file = REPO_ROOT / "tests" / "unit" / "test_imp_scoring.py"
        assert test_file.exists(), (
            "tests/unit/test_imp_scoring.py is missing — TST-05 golden fixture "
            "tests have no test file"
        )

    def test_imp_scoring_collects_minimum_tests(self):
        """pytest must collect at least 80 tests from test_imp_scoring.py."""
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/unit/test_imp_scoring.py",
                "--collect-only",
                "-q",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode in (0, 5), (
            f"test_imp_scoring.py collection failed:\n{result.stdout}\n{result.stderr}"
        )
        import re
        match = re.search(r"(\d+) test", result.stdout)
        assert match, f"Could not parse test count from: {result.stdout}"
        count = int(match.group(1))
        # Plan 21-04 deleted ~20 qualitative-label producer tests
        # (interpret_imp_score/_assign_confidence_* helpers were removed),
        # bringing the baseline from 86 down to ~66. Threshold lowered to 60
        # to keep the gate alive as a regression guard at the new realistic
        # baseline; raise this if substantial new IMP-scoring tests are added.
        assert count >= 60, (
            f"test_imp_scoring.py only collects {count} tests — "
            "expected at least 60 (post-21-04 baseline ~66)"
        )

    def test_imp_scoring_module_importable(self):
        """backend.modules.imp_scoring must be importable without errors."""
        import importlib
        try:
            mod = importlib.import_module("backend.modules.imp_scoring")
        except ImportError as e:
            pytest.fail(f"backend.modules.imp_scoring is not importable: {e}")
        assert hasattr(mod, "calculate_imp_score"), (
            "imp_scoring module has no 'calculate_imp_score' function"
        )

    def test_imp_scoring_golden_fixture_tolerance_documented(self):
        """Golden fixture must document tolerance for 4dp accuracy requirement."""
        fixture_path = REPO_ROOT / "tests" / "fixtures" / "golden_compounds.json"
        data = json.loads(fixture_path.read_text())
        # The fixture metadata should document tolerance
        metadata = data.get("metadata", {})
        assert "tolerance" in metadata, (
            "golden_compounds.json metadata has no 'tolerance' field — "
            "4dp accuracy requirement (TST-05) is not formally documented in fixture"
        )
        tolerance = metadata["tolerance"]
        assert tolerance <= 0.001, (
            f"Golden fixture tolerance is {tolerance} — "
            "must be <=0.001 to enforce 4dp accuracy"
        )
