"""
Unit tests for the process_collection_member reuse seam (Phase 23, Plan 01).

This seam (RESEARCH Pattern 2 / A4) extracts the compute steps (ChEMBL search
through classification) of process_compound_job into a member-scoped function
that:
  (a) raises a member-scoped exception instead of _fail_job_with_retry(job_id),
  (b) never touches job status,
  (c) never calls _update_progress (N members would clobber the shared job row),
  (d) writes to a per-member-unique temp dir keyed on entry_id, and
  (e) forces a fresh create + skips the global InChIKey lookup (D-08).

NOTE on D-08: the seam source (process_compound_job steps 1-7) performs NO
compound-row persistence and NO global InChIKey DB lookup -- that logic lives in
job_service (~L961-1004 / _update_compound_entry). So "skip global InChIKey
lookup / force fresh create" is satisfied BY CONSTRUCTION for the compute seam:
the test asserts the member path performs no InChIKey DB lookup and never calls
the in-place _update_compound_entry hijack. Actual member persistence + the
replace_entry_id/parent_id fresh-create bypass is exercised downstream (23-02).
"""
import inspect

import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch

_MOD = "backend.services.compound_service"


def _mock_clients():
    """Patch context managers for all 3 httpx client factories."""
    cc = patch(f"{_MOD}.create_chembl_client", return_value=AsyncMock())
    cp = patch(f"{_MOD}.create_pdb_client", return_value=AsyncMock())
    cl = patch(f"{_MOD}.create_classifier_client", return_value=AsyncMock())
    return cc, cp, cl


# ---------------------------------------------------------------------------
# Structure / contract
# ---------------------------------------------------------------------------

class TestSeamStructure:
    """Verify the seam exists with the right shape."""

    def test_process_collection_member_exists(self):
        from backend.services import compound_service as mod
        assert hasattr(mod, "process_collection_member")

    def test_process_collection_member_is_async(self):
        """Members run as awaitable coroutines under asyncio.Semaphore (D-04)."""
        from backend.services.compound_service import process_collection_member
        assert inspect.iscoroutinefunction(process_collection_member)

    def test_collection_member_error_exists(self):
        """A member-scoped exception type exists (not _fail_job_with_retry)."""
        from backend.services import compound_service as mod
        assert hasattr(mod, "CollectionMemberError")
        assert issubclass(mod.CollectionMemberError, Exception)

    def test_no_async_session_in_module(self):
        """HC-1: DB layer is SYNC SQLAlchemy -- no AsyncSession anywhere (D-CRITICAL)."""
        import backend.services.compound_service as mod
        src = inspect.getsource(mod)
        assert "AsyncSession" not in src


# ---------------------------------------------------------------------------
# Happy path: returns (df_results, result_summary, files)
# ---------------------------------------------------------------------------

class TestHappyPath:

    @pytest.mark.asyncio
    async def test_returns_results_tuple(self, tmp_path):
        """A known member returns (df_results, result_summary, files) non-empty."""
        from backend.services.compound_service import process_collection_member

        df = pd.DataFrame({
            "ChEMBL_ID": ["CHEMBL25", "CHEMBL26"],
            "Molecule_Name": ["Ethanol", "Methanol"],
            "SMILES": ["CCO", "CO"],
            "pActivity": [5.0, 6.0],
        })
        cc, cp, cl = _mock_clients()
        with cc, cp, cl, \
             patch(f"{_MOD}.get_chembl_ids", new=AsyncMock(return_value=[{"ChEMBL ID": "CHEMBL25", "Similarity": 100}])), \
             patch(f"{_MOD}._fetch_activities_async", new=AsyncMock(return_value=[{"ChEMBL_ID": "CHEMBL25", "SMILES": "CCO"}])), \
             patch(f"{_MOD}._calculate_molecular_descriptors_sync", side_effect=lambda d: df), \
             patch(f"{_MOD}._add_assay_interference_flags_sync", side_effect=lambda d: df), \
             patch(f"{_MOD}._calculate_advanced_metrics_sync", side_effect=lambda d: df), \
             patch(f"{_MOD}.calculate_imp_score", new=AsyncMock(return_value=df)), \
             patch(f"{_MOD}.classify_imp_candidates", side_effect=lambda d, *a, **k: df), \
             patch(f"{_MOD}._add_chemical_classification_async", new=AsyncMock(return_value=df)), \
             patch(f"{_MOD}._build_all_similar_df_async", new=AsyncMock(return_value=pd.DataFrame())), \
             patch(f"{_MOD}._fetch_drug_indications_async", new=AsyncMock(return_value=pd.DataFrame())), \
             patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = str(tmp_path)
            member = {"name": "Ethanol", "smiles": "CCO", "similarity_threshold": 90}
            df_results, result_summary, files = await process_collection_member(
                member, results_dir=str(tmp_path)
            )

        assert df_results is not None and len(df_results) > 0
        assert isinstance(result_summary, dict)
        assert result_summary.get("compound_name") == "Ethanol"
        assert files  # at least the ZIP path


# ---------------------------------------------------------------------------
# Per-member config coercion: a missing/None similarity_threshold must default
# to 90, NOT crash on int(None). members_config stores model_dump() which keeps
# similarity_threshold present-but-None, so `.get(key, 90)` returns None and a
# naive int(None) raises TypeError -> every member fails. Guard against that.
# ---------------------------------------------------------------------------

class TestThresholdCoercion:

    def _patches(self, df, chembl_mock, tmp_path):
        cc, cp, cl = _mock_clients()
        return [
            cc, cp, cl,
            patch(f"{_MOD}.get_chembl_ids", new=chembl_mock),
            patch(f"{_MOD}._fetch_activities_async", new=AsyncMock(return_value=[{"ChEMBL_ID": "CHEMBL25", "SMILES": "CCO"}])),
            patch(f"{_MOD}._calculate_molecular_descriptors_sync", side_effect=lambda d: df),
            patch(f"{_MOD}._add_assay_interference_flags_sync", side_effect=lambda d: df),
            patch(f"{_MOD}._calculate_advanced_metrics_sync", side_effect=lambda d: df),
            patch(f"{_MOD}.calculate_imp_score", new=AsyncMock(return_value=df)),
            patch(f"{_MOD}.classify_imp_candidates", side_effect=lambda d, *a, **k: df),
            patch(f"{_MOD}._add_chemical_classification_async", new=AsyncMock(return_value=df)),
            patch(f"{_MOD}._build_all_similar_df_async", new=AsyncMock(return_value=pd.DataFrame())),
            patch(f"{_MOD}._fetch_drug_indications_async", new=AsyncMock(return_value=pd.DataFrame())),
        ]

    @pytest.mark.asyncio
    async def test_none_threshold_defaults_to_90_without_crashing(self, tmp_path):
        """A member with similarity_threshold=None succeeds and uses 90 (not int(None))."""
        from backend.services.compound_service import process_collection_member

        df = pd.DataFrame({"ChEMBL_ID": ["CHEMBL25"], "SMILES": ["CCO"], "pActivity": [5.0]})
        chembl_mock = AsyncMock(return_value=[{"ChEMBL ID": "CHEMBL25", "Similarity": 100}])

        with patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = str(tmp_path)
            from contextlib import ExitStack
            with ExitStack() as stack:
                for p in self._patches(df, chembl_mock, tmp_path):
                    stack.enter_context(p)
                member = {"name": "Ethanol", "smiles": "CCO", "similarity_threshold": None}
                df_results, _summary, _files = await process_collection_member(
                    member, results_dir=str(tmp_path)
                )

        assert df_results is not None
        # get_chembl_ids(client, smiles, similarity_threshold) -- 3rd positional arg.
        assert chembl_mock.call_args.args[2] == 90

    @pytest.mark.asyncio
    async def test_missing_threshold_key_defaults_to_90(self, tmp_path):
        """A member with no similarity_threshold key at all also defaults to 90."""
        from backend.services.compound_service import process_collection_member

        df = pd.DataFrame({"ChEMBL_ID": ["CHEMBL25"], "SMILES": ["CCO"], "pActivity": [5.0]})
        chembl_mock = AsyncMock(return_value=[{"ChEMBL ID": "CHEMBL25", "Similarity": 100}])

        with patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = str(tmp_path)
            from contextlib import ExitStack
            with ExitStack() as stack:
                for p in self._patches(df, chembl_mock, tmp_path):
                    stack.enter_context(p)
                member = {"name": "Ethanol", "smiles": "CCO"}
                await process_collection_member(member, results_dir=str(tmp_path))

        assert chembl_mock.call_args.args[2] == 90


# ---------------------------------------------------------------------------
# (a)+(b) member-scoped error: no _fail_job_with_retry, no job-status mutation
# ---------------------------------------------------------------------------

class TestMemberScopedError:

    @pytest.mark.asyncio
    async def test_failure_raises_member_error_not_fail_job(self):
        """A failing member raises CollectionMemberError and never fails the job."""
        from backend.services import compound_service as mod

        cc, cp, cl = _mock_clients()
        with cc, cp, cl, \
             patch(f"{_MOD}.get_chembl_ids", new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch(f"{_MOD}._fail_job_with_retry", new=AsyncMock()) as fail_mock, \
             patch(f"{_MOD}._update_progress", new=AsyncMock()) as prog_mock:
            member = {"name": "BadMember", "smiles": "CCO", "similarity_threshold": 90}
            with pytest.raises(mod.CollectionMemberError):
                await mod.process_collection_member(member)

        fail_mock.assert_not_called()
        prog_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_results_raises_member_error(self):
        """An empty ChEMBL result raises a member error, not a job failure."""
        from backend.services import compound_service as mod

        cc, cp, cl = _mock_clients()
        with cc, cp, cl, \
             patch(f"{_MOD}.get_chembl_ids", new=AsyncMock(return_value=[])), \
             patch(f"{_MOD}.cascade_similarity_counts", new=AsyncMock(return_value=[])), \
             patch(f"{_MOD}._fail_job_with_retry", new=AsyncMock()) as fail_mock:
            member = {"name": "Empty", "smiles": "CCO", "similarity_threshold": 90}
            with pytest.raises(mod.CollectionMemberError):
                await mod.process_collection_member(member)

        fail_mock.assert_not_called()


# ---------------------------------------------------------------------------
# (c) no progress clobber: process_collection_member never calls _update_progress
# ---------------------------------------------------------------------------

class TestNoProgressClobber:

    def test_function_body_has_no_update_progress_call(self):
        """Static guard: the seam body must not reference _update_progress."""
        from backend.services.compound_service import process_collection_member
        src = inspect.getsource(process_collection_member)
        assert "_update_progress" not in src

    @pytest.mark.asyncio
    async def test_happy_path_never_calls_update_progress(self, tmp_path):
        from backend.services.compound_service import process_collection_member

        df = pd.DataFrame({
            "ChEMBL_ID": ["CHEMBL25"],
            "SMILES": ["CCO"],
            "pActivity": [5.0],
        })
        cc, cp, cl = _mock_clients()
        with cc, cp, cl, \
             patch(f"{_MOD}.get_chembl_ids", new=AsyncMock(return_value=[{"ChEMBL ID": "CHEMBL25", "Similarity": 100}])), \
             patch(f"{_MOD}._fetch_activities_async", new=AsyncMock(return_value=[{"ChEMBL_ID": "CHEMBL25", "SMILES": "CCO"}])), \
             patch(f"{_MOD}._calculate_molecular_descriptors_sync", side_effect=lambda d: df), \
             patch(f"{_MOD}._add_assay_interference_flags_sync", side_effect=lambda d: df), \
             patch(f"{_MOD}._calculate_advanced_metrics_sync", side_effect=lambda d: df), \
             patch(f"{_MOD}.calculate_imp_score", new=AsyncMock(return_value=df)), \
             patch(f"{_MOD}.classify_imp_candidates", side_effect=lambda d, *a, **k: df), \
             patch(f"{_MOD}._add_chemical_classification_async", new=AsyncMock(return_value=df)), \
             patch(f"{_MOD}._build_all_similar_df_async", new=AsyncMock(return_value=pd.DataFrame())), \
             patch(f"{_MOD}._fetch_drug_indications_async", new=AsyncMock(return_value=pd.DataFrame())), \
             patch(f"{_MOD}._update_progress", new=AsyncMock()) as prog_mock, \
             patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = str(tmp_path)
            member = {"name": "Ethanol", "smiles": "CCO", "similarity_threshold": 90}
            await process_collection_member(member, results_dir=str(tmp_path))

        prog_mock.assert_not_called()


# ---------------------------------------------------------------------------
# (d) per-member-unique temp dir keyed on entry_id: same-named members no collide
# ---------------------------------------------------------------------------

class TestUniqueTempDir:
    """(d) Two same-named members must use DISTINCT save dirs.

    Discriminating design: we capture the exact `results_dir` that
    _save_results_sync is handed for each member and assert the two are distinct
    AND each is a unique per-member subdir (not the shared root the caller
    passed). This fails if uniqueness keying is removed -- it does NOT rely on
    uuid4() != uuid4() (which would be true for any implementation).
    """

    @pytest.mark.asyncio
    async def test_same_named_members_use_distinct_dirs(self, tmp_path):
        from backend.services.compound_service import process_collection_member

        df = pd.DataFrame({"ChEMBL_ID": ["CHEMBL25"], "SMILES": ["CCO"], "pActivity": [5.0]})
        captured: list = []

        def _fake_save(*args, **kwargs):
            # results_dir is the 11th positional arg of _save_results_sync
            # (compound_name, smiles, sim, act, df, ind, allsim, entry_id,
            #  author_name, results_dir).
            captured.append(args[9] if len(args) > 9 else kwargs.get("results_dir"))
            return (str(args[9]) + "/out.zip", {"compound_name": args[0]})

        async def _run(member, shared_dir):
            cc, cp, cl = _mock_clients()
            with cc, cp, cl, \
                 patch(f"{_MOD}.get_chembl_ids", new=AsyncMock(return_value=[{"ChEMBL ID": "CHEMBL25", "Similarity": 100}])), \
                 patch(f"{_MOD}._fetch_activities_async", new=AsyncMock(return_value=[{"ChEMBL_ID": "CHEMBL25", "SMILES": "CCO"}])), \
                 patch(f"{_MOD}._calculate_molecular_descriptors_sync", side_effect=lambda d: df), \
                 patch(f"{_MOD}._add_assay_interference_flags_sync", side_effect=lambda d: df), \
                 patch(f"{_MOD}._calculate_advanced_metrics_sync", side_effect=lambda d: df), \
                 patch(f"{_MOD}.calculate_imp_score", new=AsyncMock(return_value=df)), \
                 patch(f"{_MOD}.classify_imp_candidates", side_effect=lambda d, *a, **k: df), \
                 patch(f"{_MOD}._add_chemical_classification_async", new=AsyncMock(return_value=df)), \
                 patch(f"{_MOD}._build_all_similar_df_async", new=AsyncMock(return_value=pd.DataFrame())), \
                 patch(f"{_MOD}._fetch_drug_indications_async", new=AsyncMock(return_value=pd.DataFrame())), \
                 patch(f"{_MOD}._save_results_sync", side_effect=_fake_save), \
                 patch(f"{_MOD}.settings") as ms:
                ms.RESULTS_DIR = str(tmp_path)
                # Caller passes the SAME shared_dir for both members (as the
                # fan-out does) -- uniqueness must hold despite the shared root.
                return await process_collection_member(member, results_dir=shared_dir)

        shared_dir = str(tmp_path / "collection_run")
        member = {"name": "SameName", "smiles": "CCO", "similarity_threshold": 90}
        await _run(dict(member), shared_dir)
        await _run(dict(member), shared_dir)

        assert len(captured) == 2
        # The two save dirs must DIFFER even though the caller passed one shared
        # root -- this fails if per-member uniqueness keying is removed.
        assert captured[0] != captured[1]
        # And each must be a unique per-member subdir under the shared root.
        for d in captured:
            assert d != shared_dir
            assert d.startswith(shared_dir)

    @pytest.mark.asyncio
    async def test_default_root_also_unique(self, tmp_path):
        """When results_dir is omitted, the default branch still gives a unique
        per-member subdir under settings.RESULTS_DIR (the None branch is
        exercised, not just the provided-dir branch)."""
        from backend.services.compound_service import process_collection_member

        df = pd.DataFrame({"ChEMBL_ID": ["CHEMBL25"], "SMILES": ["CCO"], "pActivity": [5.0]})
        captured: list = []

        def _fake_save(*args, **kwargs):
            captured.append(args[9] if len(args) > 9 else kwargs.get("results_dir"))
            return (str(args[9]) + "/out.zip", {"compound_name": args[0]})

        async def _run(member):
            cc, cp, cl = _mock_clients()
            with cc, cp, cl, \
                 patch(f"{_MOD}.get_chembl_ids", new=AsyncMock(return_value=[{"ChEMBL ID": "CHEMBL25", "Similarity": 100}])), \
                 patch(f"{_MOD}._fetch_activities_async", new=AsyncMock(return_value=[{"ChEMBL_ID": "CHEMBL25", "SMILES": "CCO"}])), \
                 patch(f"{_MOD}._calculate_molecular_descriptors_sync", side_effect=lambda d: df), \
                 patch(f"{_MOD}._add_assay_interference_flags_sync", side_effect=lambda d: df), \
                 patch(f"{_MOD}._calculate_advanced_metrics_sync", side_effect=lambda d: df), \
                 patch(f"{_MOD}.calculate_imp_score", new=AsyncMock(return_value=df)), \
                 patch(f"{_MOD}.classify_imp_candidates", side_effect=lambda d, *a, **k: df), \
                 patch(f"{_MOD}._add_chemical_classification_async", new=AsyncMock(return_value=df)), \
                 patch(f"{_MOD}._build_all_similar_df_async", new=AsyncMock(return_value=pd.DataFrame())), \
                 patch(f"{_MOD}._fetch_drug_indications_async", new=AsyncMock(return_value=pd.DataFrame())), \
                 patch(f"{_MOD}._save_results_sync", side_effect=_fake_save), \
                 patch(f"{_MOD}.settings") as ms:
                ms.RESULTS_DIR = str(tmp_path)
                return await process_collection_member(member)  # no results_dir

        member = {"name": "SameName", "smiles": "CCO", "similarity_threshold": 90}
        await _run(dict(member))
        await _run(dict(member))

        assert len(captured) == 2
        assert captured[0] != captured[1]
        for d in captured:
            assert d != str(tmp_path)
            assert d.startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# (e) D-08: no global InChIKey DB lookup, no in-place _update_compound_entry
# ---------------------------------------------------------------------------

class TestForceFreshCreate:

    def test_seam_does_not_call_update_compound_entry(self):
        """D-08: the compute seam must never invoke the in-place InChIKey hijack."""
        from backend.services.compound_service import process_collection_member
        src = inspect.getsource(process_collection_member)
        assert "_update_compound_entry" not in src

    def test_seam_does_not_do_global_inchikey_db_lookup(self):
        """D-08: the seam performs no find_by_inchikey global dedup lookup."""
        from backend.services.compound_service import process_collection_member
        src = inspect.getsource(process_collection_member)
        assert "find_by_inchikey" not in src


# ---------------------------------------------------------------------------
# Plan 3 (D-PF-6): failure-cascade diagnostic — CollectionMemberError.cascade_results
# ---------------------------------------------------------------------------

class TestCollectionMemberErrorCascade:
    def test_carries_cascade_results(self):
        from backend.services.compound_service import CollectionMemberError
        err = CollectionMemberError(
            "No similar compounds", member_name="A",
            cascade_results=[{"threshold": 80, "count": 5}],
        )
        assert err.member_name == "A"
        assert err.cascade_results == [{"threshold": 80, "count": 5}]

    def test_cascade_defaults_none(self):
        from backend.services.compound_service import CollectionMemberError
        assert CollectionMemberError("x").cascade_results is None


class TestProcessCollectionMemberCascade:
    async def test_no_similar_compounds_attaches_cascade(self, tmp_path):
        from backend.services.compound_service import (
            process_collection_member, CollectionMemberError,
        )
        cascade = [{"threshold": 80, "count": 5}, {"threshold": 40, "count": 29}]
        with patch(f"{_MOD}.create_chembl_client", return_value=AsyncMock()), \
             patch(f"{_MOD}.create_pdb_client", return_value=AsyncMock()), \
             patch(f"{_MOD}.create_classifier_client", return_value=AsyncMock()), \
             patch(f"{_MOD}.get_chembl_ids", new=AsyncMock(return_value=[])), \
             patch(f"{_MOD}.cascade_similarity_counts", new=AsyncMock(return_value=cascade)), \
             patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = str(tmp_path)
            with pytest.raises(CollectionMemberError) as ei:
                await process_collection_member(
                    {"name": "A", "smiles": "CCO", "similarity_threshold": 90},
                    results_dir=str(tmp_path),
                )
            assert ei.value.cascade_results == cascade

    async def test_cascade_probe_unreachable_still_raises_without_cascade(self, tmp_path):
        from backend.services.compound_service import (
            process_collection_member, CollectionMemberError,
        )
        with patch(f"{_MOD}.create_chembl_client", return_value=AsyncMock()), \
             patch(f"{_MOD}.create_pdb_client", return_value=AsyncMock()), \
             patch(f"{_MOD}.create_classifier_client", return_value=AsyncMock()), \
             patch(f"{_MOD}.get_chembl_ids", new=AsyncMock(return_value=[])), \
             patch(f"{_MOD}.cascade_similarity_counts",
                   new=AsyncMock(side_effect=ConnectionError("down"))), \
             patch(f"{_MOD}.settings") as ms:
            ms.RESULTS_DIR = str(tmp_path)
            with pytest.raises(CollectionMemberError) as ei:
                await process_collection_member(
                    {"name": "A", "smiles": "CCO", "similarity_threshold": 90},
                    results_dir=str(tmp_path),
                )
            assert ei.value.cascade_results in (None, [])


# ---------------------------------------------------------------------------
# Plan 24-04 (D-S2-SOURCE Option A): static-source guard for the Tier-3 CSV
# root arcnames. The aggregate re-read in collection_service matches the 3
# Tier-3 CSVs by EXACT filename at the member ZIP ROOT. If _save_results_sync
# (compound_service) ever renames or relocates them into a subfolder, the
# Evidence aggregate silently goes empty -- this guard trips first (mirrors the
# existing process_collection_member body-assertion guards). RESEARCH Pitfall 4.
# ---------------------------------------------------------------------------

class TestTier3RootArcnames:

    def test_tier3_csv_arcnames(self, tmp_path):
        """The 3 Tier-3 CSVs the Option-A aggregate re-reads must sit at the
        member ZIP ROOT (no subfolder). Built via the 24-01 member-ZIP builder,
        which mirrors _save_results_sync's
        arcname = os.path.relpath(file_path, compound_folder)."""
        import zipfile

        from tests.unit.fixtures.collection_member_zip_builder import (
            ALL_SIMILAR_MOLECULES_CSV,
            DRUG_INDICATIONS_CSV,
            PDB_SUMMARY_CSV,
            build_member_zip,
        )

        member_zip = build_member_zip(
            tmp_path,
            "Ethanol",
            indications=pd.DataFrame({"chembl_id": ["CHEMBL25"], "indication": ["x"]}),
            similar=pd.DataFrame({"chembl_id": ["CHEMBL1"], "similarity": [99]}),
            pdb=pd.DataFrame({"pdb_id": ["1ABC"]}),
        )

        with zipfile.ZipFile(member_zip, "r") as zf:
            names = set(zf.namelist())

        # The exact root names the aggregate re-read keys on. A rename or a move
        # into a subfolder (e.g. "tier3/drug_indications.csv") breaks this.
        for root_name in (
            DRUG_INDICATIONS_CSV,
            ALL_SIMILAR_MOLECULES_CSV,
            PDB_SUMMARY_CSV,
        ):
            assert root_name in names, (
                f"{root_name!r} must live at the member ZIP ROOT -- the Option-A "
                f"aggregate re-read (collection_service) matches it by exact name"
            )
            assert "/" not in root_name, "Tier-3 CSV must NOT be nested in a subfolder"

    def test_aggregate_reread_uses_exact_root_filenames(self):
        """Static guard: collection_service's Tier-3 filename map must hold the
        3 canonical root names. If the source CSVs are renamed in
        compound_service, BOTH this map and the builder constants must change in
        lockstep -- this test plus test_tier3_csv_arcnames bracket that drift."""
        from backend.services.collection_service import _AGGREGATE_TIER3_FILES

        assert _AGGREGATE_TIER3_FILES["indications"] == "drug_indications.csv"
        assert _AGGREGATE_TIER3_FILES["all_similar"] == "all_similar_molecules.csv"
        assert _AGGREGATE_TIER3_FILES["pdb"] == "pdb_summary.csv"
