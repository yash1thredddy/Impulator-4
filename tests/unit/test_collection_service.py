"""
Unit tests for the collection service fan-out (Phase 23, plan 23-04).

Fan-out contract (D-04/D-09): members run as awaited coroutines under a LOCAL
asyncio.Semaphore -- NEVER executor.submit (re-entry deadlock against the 1 global
slot the collection job already holds). Any member failure becomes a MemberResult
sentinel so a bad member can never abort its siblings (D-09 partial success). The
fan-out writes AGGREGATE progress (completed/total) only; members never write
per-member progress.

These tests exercise the pure / injectable seams directly (no DB, no network, no
Docker) so they run green here AND in CI:
  - _run_member_fanout(member_processor=...) -- injectable member compute
  - build_combined_activities(...)            -- PURE 250k row-cap helper (D-03)
"""
import asyncio
import inspect

import pandas as pd
import pytest

from backend.config import settings
from backend.services.collection_service import (
    COMBINED_ROW_CAP,
    MemberResult,
    _run_member_fanout,
    build_combined_activities,
    process_collection_job,
)


# ---------------------------------------------------------------------------
# D-09: partial success -- one failing member must not abort the others
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fanout_partial_success():
    """One failing member -> caught as a MemberResult sentinel; the other members
    still succeed; member_failed_count reflects exactly the one failure (D-09)."""
    members = [
        {"name": "Ethanol", "smiles": "CCO"},
        {"name": "BoomCompound", "smiles": "CC"},  # this one raises
        {"name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    ]

    async def fake_processor(member_input, *, results_dir=None):
        name = member_input["name"]
        if name == "BoomCompound":
            # A NON-CollectionMemberError to prove the wrapper catches broadly.
            raise RuntimeError("simulated member explosion")
        df = pd.DataFrame({"ChEMBL_ID": ["CHEMBL1"], "Value": [1.0]})
        return df, {"entry_id": f"id-{name}", "imp_score": 0.5}, [f"/tmp/{name}.zip"]

    results = await _run_member_fanout(
        members, shared_root="/tmp/x", member_processor=fake_processor
    )

    assert len(results) == 3
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    # Exactly one failure; the other two succeeded -- gather did NOT abort.
    assert len(ok) == 2
    assert len(failed) == 1
    assert failed[0].member_name == "BoomCompound"
    assert "explosion" in (failed[0].error or "")
    # member_failed_count is total - completed.
    member_failed_count = len(members) - len(ok)
    assert member_failed_count == 1
    # Surviving members carried their payloads through.
    assert {r.member_name for r in ok} == {"Ethanol", "Aspirin"}
    assert all(r.df is not None for r in ok)


@pytest.mark.asyncio
async def test_fanout_uses_local_semaphore_never_blocks_more_than_cap():
    """D-04: members run under a LOCAL semaphore sized to
    COLLECTION_MEMBER_CONCURRENCY -- never more than `cap` members run at once."""
    cap = settings.COLLECTION_MEMBER_CONCURRENCY
    members = [{"name": f"m{i}", "smiles": "CCO"} for i in range(cap * 3)]

    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def fake_processor(member_input, *, results_dir=None):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return pd.DataFrame({"v": [1]}), {"entry_id": "e"}, []

    await _run_member_fanout(
        members, shared_root="/tmp/x", member_processor=fake_processor
    )
    # The local semaphore caps concurrency; never exceeds the configured cap.
    assert peak <= cap


# ---------------------------------------------------------------------------
# D-03: 250k row cap on the combined activities table
# ---------------------------------------------------------------------------

def test_row_cap_truncates_and_warns():
    """D-03: combined_activities past the 250k cap truncates + flags a warning;
    per-member sections (the source frames) are untouched -- only the merged
    convenience table is capped."""
    # Two members whose combined rows exceed the cap.
    half = COMBINED_ROW_CAP // 2 + 100  # 2 * this > COMBINED_ROW_CAP
    df_a = pd.DataFrame({"ChEMBL_ID": ["A"] * half, "Value": range(half)})
    df_b = pd.DataFrame({"ChEMBL_ID": ["B"] * half, "Value": range(half)})

    combined, truncated = build_combined_activities(
        [df_a, df_b],
        ["MemberA", "MemberB"],
        ["id-a", "id-b"],
    )

    assert truncated is True
    assert len(combined) == COMBINED_ROW_CAP
    # Per-row stamping is present (compound_name + entry_id).
    assert "compound_name" in combined.columns
    assert "entry_id" in combined.columns
    # Source frames were NOT mutated (per-member ZIP sections remain complete).
    assert len(df_a) == half
    assert len(df_b) == half
    assert "compound_name" not in df_a.columns


def test_row_cap_no_truncation_under_cap():
    """Under the cap: no truncation, no warning; all rows + stamps present."""
    df_a = pd.DataFrame({"ChEMBL_ID": ["A", "A"], "Value": [1, 2]})
    df_b = pd.DataFrame({"ChEMBL_ID": ["B"], "Value": [3]})

    combined, truncated = build_combined_activities(
        [df_a, df_b], ["A", "B"], ["id-a", "id-b"]
    )

    assert truncated is False
    assert len(combined) == 3
    # entry_id stamped correctly per source.
    assert set(combined["entry_id"]) == {"id-a", "id-b"}
    assert set(combined["compound_name"]) == {"A", "B"}


def test_combined_activities_empty_members():
    """No member frames -> empty combined table, no truncation."""
    combined, truncated = build_combined_activities([], [], [])
    assert truncated is False
    assert combined.empty


# ---------------------------------------------------------------------------
# D-09: aggregate-only progress -- members never write per-member progress
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_aggregate_progress_not_per_member():
    """D-09: the fan-out invokes the aggregate progress callback once per
    completed member with a monotonically increasing completed-count (1..N), and
    the member compute seam itself never reports per-member job progress."""
    members = [{"name": f"m{i}", "smiles": "CCO"} for i in range(5)]
    progress_calls: list[int] = []

    async def fake_processor(member_input, *, results_dir=None):
        return pd.DataFrame({"v": [1]}), {"entry_id": "e"}, []

    await _run_member_fanout(
        members,
        shared_root="/tmp/x",
        member_processor=fake_processor,
        progress_cb=lambda done: progress_calls.append(done),
    )

    # One aggregate progress tick per member, covering 1..N exactly once each
    # (completed/total semantics) -- not a per-member STEP write.
    assert sorted(progress_calls) == [1, 2, 3, 4, 5]
    assert len(progress_calls) == len(members)


def test_member_seam_never_calls_update_progress():
    """D-09 (static): the real per-member compute seam contains no per-member
    job-progress write -- aggregate progress is the fan-out's job only."""
    from backend.services import compound_service

    src = inspect.getsource(compound_service.process_collection_member)
    # The seam must not invoke the per-member job progress writer.
    assert "_update_progress(" not in src
    assert "update_progress(" not in src


def test_process_collection_job_is_coroutine_not_executor_submit():
    """D-04 (static): process_collection_job is an async coroutine that uses a
    LOCAL asyncio.Semaphore and NEVER executor.submit per member."""
    assert inspect.iscoroutinefunction(process_collection_job)
    from backend.services import collection_service

    src = inspect.getsource(collection_service)
    assert "asyncio.Semaphore" in src  # LOCAL fan-out semaphore (D-04)
    assert "executor.submit" not in src  # never re-enter the global executor


def test_member_result_sentinel_shape():
    """The MemberResult sentinel carries ok/error so members never raise out."""
    ok = MemberResult(ok=True, member_name="x", df=pd.DataFrame(), member_summary={})
    bad = MemberResult(ok=False, member_name="y", error="boom")
    assert ok.ok and ok.error is None
    assert not bad.ok and bad.error == "boom"


@pytest.mark.asyncio
async def test_process_collection_job_drives_processing_then_finalizes_once(
    monkeypatch, tmp_path
):
    """Regression: process_collection_job must set PENDING -> PROCESSING up front
    (mirroring process_compound_job) and then finalize the job exactly ONCE.

    Without the explicit PROCESSING transition the finalize chain
    (PROCESSING -> PENDING_UPLOAD -> COMPLETED) is rejected by VALID_TRANSITIONS
    and a real collection job hangs at PENDING forever. This patches the DB /
    storage seams so the orchestration runs with no DB, network, or Docker.
    """
    import uuid as _uuid

    from backend.services import collection_service as cs

    job_id = _uuid.uuid4()
    collection_id = _uuid.uuid4()
    members = [{"name": "Ethanol", "smiles": "CCO"}, {"name": "Aspirin", "smiles": "CCO"}]
    calls: list[str] = []

    monkeypatch.setattr(cs, "_load_members_sync", lambda jid: (collection_id, members))

    async def fake_member(member_input, *, results_dir=None):
        return pd.DataFrame({"ChEMBL_ID": ["X"], "v": [1]}), {"entry_id": "e", "imp_score": 1.0}, []

    monkeypatch.setattr(
        "backend.services.compound_service.process_collection_member", fake_member
    )
    # Stub the ZIP/upload so no disk/Azure work happens.
    monkeypatch.setattr(
        cs,
        "_assemble_and_upload_zip_sync",
        lambda cid, succeeded: {"storage_path": f"collections/x/{cid}.zip"},
    )

    def rec_progress(jid, progress, step):
        calls.append(f"progress:{progress}")

    def rec_finalize(jid, cid, stats):
        calls.append("finalize")

    def rec_fail(jid, msg, *a):
        calls.append(f"fail:{msg}")

    monkeypatch.setattr(cs, "_update_progress_sync", rec_progress)
    monkeypatch.setattr(cs, "_finalize_job_sync", rec_finalize)
    monkeypatch.setattr(cs, "_fail_job_sync", rec_fail)

    await cs.process_collection_job(job_id)

    # The FIRST status write is the PROCESSING transition (progress 0.0), and the
    # job is finalized exactly once. No fail path on an all-success run.
    assert calls[0] == "progress:0.0"
    assert calls.count("finalize") == 1
    assert not any(c.startswith("fail:") for c in calls)


@pytest.mark.asyncio
async def test_all_members_failed_auto_deletes_collection(monkeypatch):
    """When EVERY member fails, the collection is auto-soft-deleted (user-chosen
    cleanup) so a fully-failed collection (0 usable compounds, no ZIP) never
    lingers on the Collections page. The job is still failed and is NOT finalized;
    the failure still surfaces in the sidebar's Failed Jobs section.
    """
    import uuid as _uuid

    from backend.services import collection_service as cs
    from backend.services.compound_service import CollectionMemberError

    job_id = _uuid.uuid4()
    collection_id = _uuid.uuid4()
    members = [{"name": "A", "smiles": "CCO"}, {"name": "B", "smiles": "CCO"}]
    deleted: list = []
    failed: list = []

    monkeypatch.setattr(cs, "_load_members_sync", lambda jid: (collection_id, members))

    async def fail_member(member_input, *, results_dir=None):
        raise CollectionMemberError("no data", member_name=member_input["name"])

    monkeypatch.setattr(
        "backend.services.compound_service.process_collection_member", fail_member
    )
    monkeypatch.setattr(cs, "_update_progress_sync", lambda *a: None)
    monkeypatch.setattr(cs, "_fail_job_sync", lambda jid, msg, *a: failed.append(msg))
    monkeypatch.setattr(
        cs, "_finalize_job_sync",
        lambda *a: pytest.fail("must not finalize when all members failed"),
    )
    monkeypatch.setattr(
        cs, "_auto_delete_failed_collection_sync",
        lambda cid: deleted.append(cid), raising=False,
    )

    await cs.process_collection_job(job_id)

    assert failed, "job must be failed when all members fail"
    assert deleted == [collection_id], (
        "a fully-failed collection must be auto-deleted for cleanup"
    )


# ---------------------------------------------------------------------------
# Plan 3 (D-PF-6): MemberResult carries cascade; fan-out copies it
# ---------------------------------------------------------------------------

class TestMemberResultCascade:
    async def test_member_error_copies_cascade_into_result(self):
        from backend.services.collection_service import _run_member_fanout
        from backend.services.compound_service import CollectionMemberError

        async def boom(member_input, *, results_dir=None):
            raise CollectionMemberError(
                "no data", member_name=member_input["name"],
                cascade_results=[{"threshold": 50, "count": 3}],
            )

        results = await _run_member_fanout(
            [{"name": "A", "smiles": "CCO"}],
            shared_root="/tmp",
            member_processor=boom,
        )
        assert results[0].ok is False
        assert results[0].cascade_results == [{"threshold": 50, "count": 3}]

    async def test_non_member_exception_has_no_cascade(self):
        from backend.services.collection_service import _run_member_fanout

        async def boom(member_input, *, results_dir=None):
            raise RuntimeError("unexpected bug")

        results = await _run_member_fanout(
            [{"name": "B", "smiles": "CCO"}],
            shared_root="/tmp",
            member_processor=boom,
        )
        assert results[0].ok is False
        assert results[0].cascade_results is None


# ---------------------------------------------------------------------------
# Plan 3 (D-PF-6): aggregate failed_members into stats + failed-job result_summary
# ---------------------------------------------------------------------------

class TestFailedMembersAggregation:
    def test_build_failed_members_payload(self):
        from backend.services.collection_service import _failed_members_payload
        from backend.services.collection_service import MemberResult
        results = [
            MemberResult(ok=True, member_name="A"),
            MemberResult(ok=False, member_name="B", error="No bioactivity data found",
                         cascade_results=[{"threshold": 50, "count": 3}]),
            MemberResult(ok=False, member_name="C", error="boom", cascade_results=None),
        ]
        payload = _failed_members_payload(results)
        assert payload == [
            {"name": "B", "error": "No bioactivity data found",
             "cascade_results": [{"threshold": 50, "count": 3}]},
            {"name": "C", "error": "boom", "cascade_results": None},
        ]
