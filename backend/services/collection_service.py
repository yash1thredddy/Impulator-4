"""Collection service -- create, delete, and the deadlock-safe resilient fan-out.

A *collection* is ONE ``JobType.COLLECTION`` job plus a row in the ``collections``
table whose members are persisted in ``members_config`` JSONB (D-02). Processing a
collection means running the per-member compute pipeline for every member, folding
the per-member artifacts into ONE nested ZIP, and finalising the single job row.

Concurrency model (D-04 -- the load-bearing invariant):
    The COLLECTION job runs as a single coroutine and therefore holds exactly ONE
    slot of the global executor semaphore (``backend.core.executor``). Its members
    fan out *inside* that coroutine under a LOCAL ``asyncio.Semaphore`` and are
    ``await``ed directly. They are NEVER submitted back into the global executor --
    doing so would re-enter the semaphore the parent already holds and deadlock.

Resilience (D-09 -- partial success):
    Each member runs inside a try/except that converts ANY failure into a
    :class:`MemberResult` sentinel (``ok=False``). Members never raise out of the
    fan-out, so ``asyncio.gather`` always returns a full result list -- one bad
    member can never abort its siblings.

Progress (D-09 -- aggregate only):
    The job row's progress is written by the fan-out coroutine in aggregate terms
    (completed / total) ONLY. Members never touch job status or per-member
    progress (the seam in ``compound_service`` enforces this by construction).

Storage (D-12):
    The assembled collection ZIP is stored under the dedicated ``collections/``
    prefix via :func:`backend.core.storage_paths.get_collection_storage_path`,
    never mixed with the per-compound ``results/`` artifacts.

Layering / typing: this service is sync-first at the DB layer (HC-1 -- the
SQLAlchemy session type is the synchronous ``Session``, never an async one). The
fan-out coroutine is ``async`` (members are awaitable coroutines), and every sync
DB call it makes is wrapped in ``run_in_executor`` with its own short-lived
session. Service methods raise ``ValueError`` for domain errors; HTTP translation
is the route's job (no HTTP error types here).
"""

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.config import settings
from backend.core import sanitize_compound_name
from backend.core.azure_sync import is_azure_configured
from backend.core.database import get_db_session
from backend.core.storage_paths import get_collection_storage_path
from backend.models.enums import AuditEventType, JobStatus, JobType
from backend.repositories import collection_repo

logger = logging.getLogger(__name__)

__all__ = [
    "MemberResult",
    "create_collection",
    "delete_collection",
    "process_collection_job",
]

# D-03: the combined activities table is hard-capped to protect the event loop /
# memory from a pathological collection (many members x many rows). Per-member ZIP
# sections are ALWAYS complete -- only the merged convenience table is truncated.
COMBINED_ROW_CAP = 250_000

# D-S2-ARCH / T-24-04-02: the Option-A aggregate re-reads the per-member
# all_similar_molecules.csv catalog into collection_aggregate.json at finalize.
# Bound the per-member row count so a pathological member ZIP cannot blow up
# finalize RAM (the 250k cap only protects combined_activities.csv). Indications
# and PDB frames are small by construction and left uncapped.
AGGREGATE_ALL_SIMILAR_ROW_CAP = 5_000

# D-S2-SOURCE (Option A): the 3 Tier-3 CSVs written by ``_save_results_sync`` at
# the ROOT of every member ZIP (arcname = os.path.relpath(file, compound_folder)).
# The aggregate re-reads ONLY these known filenames -- never arbitrary entries.
_AGGREGATE_TIER3_FILES = {
    "indications": "drug_indications.csv",
    "all_similar": "all_similar_molecules.csv",
    "pdb": "pdb_summary.csv",
}


# ---------------------------------------------------------------------------
# Member result sentinel (D-09 partial success)
# ---------------------------------------------------------------------------

@dataclass
class MemberResult:
    """Outcome of one member's compute -- members NEVER raise out of the fan-out.

    ``ok=True``  -> ``df`` / ``member_summary`` / ``files`` are populated.
    ``ok=False`` -> ``error`` holds the message; the member is counted as failed.
    """

    ok: bool
    member_name: str
    df: pd.DataFrame | None = None
    member_summary: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    error: str | None = None
    cascade_results: list[dict] | None = None


def _failed_members_payload(results: list["MemberResult"]) -> list[dict]:
    """Per-member failure rows {name, error, cascade_results} for surfacing (D-PF-6)."""
    return [
        {"name": r.member_name, "error": r.error, "cascade_results": r.cascade_results}
        for r in results
        if not r.ok
    ]


# ---------------------------------------------------------------------------
# create / delete (sync, HC-1)
# ---------------------------------------------------------------------------

def create_collection(db, payload) -> tuple[str, str]:
    """Create a collection: persist a ``collections`` row + a COLLECTION job.

    Mirrors ``job_service.submit_batch``'s ``create_job`` usage but with
    ``JobType.COLLECTION``. The member set is stored in ``members_config`` (D-02),
    never the job summary JSONB.

    Args:
        db: Active sync SQLAlchemy ``Session``.
        payload: A ``CollectionJobCreate`` (``name``, ``author_name``,
            ``description``, ``members``, optional ``session_id``).

    Returns:
        ``(collection_id, job_id)`` as strings.

    Raises:
        ValueError: on a domain failure (empty member set, persistence conflict).
    """
    from backend.services.job_service import job_service

    members = list(getattr(payload, "members", []) or [])
    if not members:
        raise ValueError("A collection must contain at least one member")

    # D-02: the canonical member-set load path is members_config, keyed on job_id.
    members_config = {
        "members": [
            m.model_dump() if hasattr(m, "model_dump") else dict(m)
            for m in members
        ]
    }

    session_id = getattr(payload, "session_id", None)

    # Create the linked COLLECTION job (mirror submit_batch's JobType.BATCH call).
    # compound_name/smiles carry the collection name so the jobs list stays
    # human-readable; the real member set lives in members_config.
    job = job_service.create_job(
        db,
        JobType.COLLECTION,
        compound_name=payload.name,
        smiles=None,
        session_id=str(session_id) if session_id else None,
        author_name=payload.author_name,
    )

    collection = collection_repo.create(
        db,
        name=payload.name,
        author_name=payload.author_name,
        job_id=job.id,
        members_config=members_config,
        description=getattr(payload, "description", None),
    )
    db.commit()

    logger.info(
        f"Created collection {collection.id} (job={job.id}) "
        f"with {len(members)} members"
    )
    return str(collection.id), str(job.id)


def _auto_delete_failed_collection_sync(collection_id) -> None:
    """Soft-delete a fully-failed collection (all members failed) for cleanup.

    Opens its own sync session (called via run_in_executor from the async job
    coroutine). Best-effort: any failure here is swallowed — the job is already
    marked FAILED, so a cleanup miss must never raise out of the coroutine.
    """
    try:
        with get_db_session() as db:
            delete_collection(db, collection_id)
            logger.info(f"Auto-deleted fully-failed collection {collection_id}")
    except Exception as e:
        logger.warning(
            f"Auto-delete of failed collection {collection_id} failed: {e}"
        )


def delete_collection(db, collection_id) -> bool:
    """Soft-delete a collection, emit an audit event, and remove its ZIP (D-11).

    NEVER routes through the single-compound job-cleanup path -- that path is for
    single-compound jobs and would mishandle a collection's nested ZIP / member
    set. Collection deletion is soft-delete + audit + ZIP removal only.

    Args:
        db: Active sync SQLAlchemy ``Session``.
        collection_id: Collection UUID (``str`` or ``uuid.UUID``).

    Returns:
        ``True`` if a live collection was found and soft-deleted; ``False`` if it
        was missing or already deleted (idempotent no-op).

    Raises:
        ValueError: if ``collection_id`` is not a valid UUID.
    """
    cid = _coerce_uuid(collection_id, "collection_id")

    collection = collection_repo.get_by_id(db, cid)
    storage_path = getattr(collection, "storage_path", None) if collection else None

    deleted = collection_repo.soft_delete(db, cid)
    if not deleted:
        logger.info(f"Collection {cid} not found or already deleted (no-op)")
        return False

    # D-11: repudiation defence -- every delete leaves an audit trail. There is no
    # COLLECTION-specific AuditEventType; JOB_DELETED with a collection marker is
    # the closest existing event.
    from backend.services._audit import log_audit_event

    log_audit_event(
        db,
        AuditEventType.JOB_DELETED,
        severity="info",
        details={
            "resource": "collection",
            "collection_id": str(cid),
        },
    )
    db.commit()

    # Remove the collections/ ZIP (best-effort -- DB state is the source of truth).
    _delete_collection_zip(str(cid), storage_path)

    logger.info(f"Soft-deleted collection {cid} and removed its ZIP")
    return True


# ---------------------------------------------------------------------------
# Fan-out coroutine (D-02 / D-03 / D-04 / D-09 / D-12)
# ---------------------------------------------------------------------------

async def process_collection_job(job_id, **kwargs) -> None:
    """Process a COLLECTION job end-to-end: fan-out -> ZIP -> aggregate -> finalise.

    Steps (RESEARCH Pattern 1):
        1. Load the collection by ``job_id`` and read members from
           ``members_config`` (D-02).
        2. Fan members out under a LOCAL ``asyncio.Semaphore`` (D-04); each member
           is the awaited :func:`compound_service.process_collection_member`
           coroutine wrapped so it NEVER raises out (D-09 sentinel).
        3. ``asyncio.gather`` (NOT TaskGroup) -> a full ``MemberResult`` list.
        4. Assemble ONE nested ZIP (``compounds/{sanitized_name}/...`` +
           ``collection_summary.csv`` + ``combined_activities.csv``) with the
           250k row cap (D-03) and path-traversal-guarded arcnames.
        5. Compute aggregate stats and upload the ZIP under ``collections/`` (D-12).
        6. Mark the job completed (or failed if zero members succeeded) ONCE, with
           aggregate progress only (D-09).

    ``job_id`` is the only required argument; everything else is loaded from the DB
    (the scheduler may pass extra kwargs, which are ignored). Members never touch
    job status or per-member progress.
    """
    loop = asyncio.get_running_loop()

    # ---- 1. Load members (D-02) -- sync DB via run_in_executor ----
    members = await loop.run_in_executor(None, _load_members_sync, job_id)
    if members is None:
        logger.error(f"Collection for job {job_id} not found; failing job")
        await loop.run_in_executor(
            None, _fail_job_sync, job_id, "Collection not found"
        )
        return
    collection_id, member_inputs = members
    total = len(member_inputs)

    # Drive PENDING -> PROCESSING up front (the process_* function owns this
    # transition, mirroring process_compound_job's first _update_progress call).
    # Without it the finalize chain (PROCESSING -> PENDING_UPLOAD -> COMPLETED) is
    # rejected by VALID_TRANSITIONS and the job would silently hang at PENDING.
    await loop.run_in_executor(
        None,
        _update_progress_sync,
        job_id,
        0.0,
        f"Processing {total} collection members...",
    )

    # ---- 2/3. Fan out under a LOCAL semaphore + gather (D-04, D-09) ----
    completed = 0
    shared_root = tempfile.mkdtemp(prefix="collection_")
    try:
        results = await _run_member_fanout(
            member_inputs,
            shared_root=shared_root,
            progress_cb=lambda done: _report_aggregate_progress(
                loop, job_id, done, total
            ),
        )
        completed = sum(1 for r in results if r.ok)
        failed = total - completed

        succeeded = [r for r in results if r.ok]
        if not succeeded:
            failed_payload = _failed_members_payload(results)  # D-PF-6
            errors = "; ".join(r.error or "unknown" for r in results)[:500]
            logger.warning(
                f"Collection job {job_id}: all {total} members failed ({errors})"
            )
            await loop.run_in_executor(
                None, _fail_job_sync, job_id,
                f"All {total} collection members failed",
                failed_payload,  # NEW arg — persist per-member cascade in result_summary
            )
            # User-chosen cleanup: a fully-failed collection has 0 usable compounds
            # and no ZIP, so auto-soft-delete it (D-11) — it must not linger on the
            # Collections page. The job stays FAILED and still surfaces in the
            # sidebar's Failed Jobs section.
            await loop.run_in_executor(
                None, _auto_delete_failed_collection_sync, collection_id
            )
            return

        # ---- 4. Assemble ONE nested ZIP (D-03 cap, arcname guard) ----
        stats = await loop.run_in_executor(
            None,
            _assemble_and_upload_zip_sync,
            str(collection_id),
            succeeded,
        )
        stats["member_failed_count"] = failed  # D-09
        stats["compound_count"] = completed
        stats["failed_members"] = _failed_members_payload(results)  # D-PF-6

        # ---- 5/6. Persist stats + finalise the job ONCE (aggregate progress) ----
        await loop.run_in_executor(
            None, _finalize_job_sync, job_id, str(collection_id), stats
        )
        logger.info(
            f"Collection job {job_id} completed: {completed}/{total} members ok, "
            f"{failed} failed, combined_truncated={stats.get('combined_truncated')}"
        )
    finally:
        shutil.rmtree(shared_root, ignore_errors=True)


async def _run_member_fanout(
    member_inputs: list[dict],
    *,
    shared_root: str,
    progress_cb: Callable[[int], None] | None = None,
    member_processor: Callable | None = None,
) -> list[MemberResult]:
    """Run members under a LOCAL semaphore (D-04); return one sentinel per member.

    ``member_processor`` is the injectable seam (defaults to the real
    ``compound_service.process_collection_member`` coroutine) so the fan-out
    control flow is unit-testable WITHOUT network / DB. Members are ``await``ed
    directly -- never submitted back into the global executor (the parent already
    holds the 1 global slot; re-entry would deadlock). Any failure becomes a
    ``MemberResult`` sentinel so a bad member can never abort its siblings (D-09).
    """
    # Imported unconditionally (not inside the ``if`` below): the nested ``_member``
    # except clause needs ``CollectionMemberError`` in scope on EVERY call, including
    # when an injected ``member_processor`` skips the default-binding branch.
    from backend.services.compound_service import (
        CollectionMemberError,
        process_collection_member,
    )

    if member_processor is None:
        member_processor = process_collection_member

    # D-04: LOCAL semaphore, NOT the global executor semaphore.
    sem = asyncio.Semaphore(settings.COLLECTION_MEMBER_CONCURRENCY)
    done_count = 0
    done_lock = asyncio.Lock()

    async def _member(member_input: dict) -> MemberResult:
        nonlocal done_count
        name = (member_input.get("name") if isinstance(member_input, dict) else None) or "member"
        async with sem:
            try:
                df, member_summary, files = await member_processor(
                    member_input, results_dir=shared_root
                )
                result = MemberResult(
                    ok=True,
                    member_name=name,
                    df=df,
                    member_summary=member_summary or {},
                    files=list(files or []),
                )
            except CollectionMemberError as e:
                logger.warning(f"Collection member '{name}' failed: {e}")
                result = MemberResult(
                    ok=False, member_name=name, error=str(e),
                    cascade_results=e.cascade_results,
                )
            except Exception as e:  # broad: a non-CollectionMemberError bug in one
                # member must not abort gather (D-09 partial success).
                logger.warning(f"Collection member '{name}' failed: {e}")
                result = MemberResult(ok=False, member_name=name, error=str(e))
        # Aggregate progress (D-09) -- completed/total only, never per-member step.
        async with done_lock:
            done_count += 1
            if progress_cb is not None:
                progress_cb(done_count)
        return result

    # gather (NOT TaskGroup) -- sentinels mean members never raise out (D-09).
    return await asyncio.gather(*[_member(m) for m in member_inputs])


# ---------------------------------------------------------------------------
# Pure / sync helpers (unit-testable; called via run_in_executor where DB-bound)
# ---------------------------------------------------------------------------

def build_combined_activities(
    member_dfs: list[pd.DataFrame],
    member_names: list[str],
    member_entry_ids: list[str],
    *,
    row_cap: int = COMBINED_ROW_CAP,
) -> tuple[pd.DataFrame, bool]:
    """Concatenate per-member activity frames into ONE table (PURE; D-03).

    Stamps ``compound_name`` + ``entry_id`` on every row, concats with
    ``ignore_index``, and truncates to ``row_cap`` rows -- returning a
    ``(combined_df, truncated)`` tuple. Truncation NEVER touches the per-member ZIP
    sections (those are written whole); only this merged convenience table is
    capped (D-03). Takes in-memory DataFrames -- no CSV round-trip.
    """
    stamped: list[pd.DataFrame] = []
    for df, name, entry_id in zip(member_dfs, member_names, member_entry_ids):
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d.insert(0, "compound_name", name)
        d.insert(1, "entry_id", entry_id)
        stamped.append(d)

    if not stamped:
        return pd.DataFrame(), False

    combined = pd.concat(stamped, ignore_index=True)
    truncated = False
    if len(combined) > row_cap:
        combined = combined.iloc[:row_cap].copy()
        truncated = True
    return combined, truncated


def _safe_member_folder(member_name: str, used: set[str]) -> str:
    """Path-traversal-guarded, de-duplicated archive folder for a member (T-23-04-T1).

    User member names become archive paths -- untrusted. ``sanitize_compound_name``
    strips traversal sequences; we additionally reject ``..`` / separators and
    suffix a counter so two same-named members never collide inside the ZIP.
    """
    safe = sanitize_compound_name(member_name or "member")
    # Defence in depth: never allow traversal / separators into an arcname.
    safe = safe.replace("..", "_").replace("/", "_").replace("\\", "_").strip()
    if not safe or safe in (".", ".."):
        safe = "member"
    candidate = safe
    n = 1
    while candidate in used:
        n += 1
        candidate = f"{safe}_{n}"
    used.add(candidate)
    return candidate


def _assemble_and_upload_zip_sync(
    collection_id: str,
    succeeded: list[MemberResult],
) -> dict[str, Any]:
    """Build ONE nested ZIP, upload it under ``collections/`` (D-12), return stats.

    Layout: ``compounds/{sanitized_name}/...`` (each member's ZIP contents),
    ``collection_summary.csv`` (one row per member), ``combined_activities.csv``
    (250k-capped merged table, D-03). All arcnames are path-traversal-guarded
    and folder names de-duplicated (T-23-04-T1). Runs in a thread (CPU/disk I/O).
    """
    member_dfs: list[pd.DataFrame] = []
    member_names: list[str] = []
    member_entry_ids: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    used_folders: set[str] = set()
    # Map each member's archive folder so ZIP sections + summary agree.
    member_folders: list[str] = []

    for r in succeeded:
        rs = r.member_summary or {}
        entry_id = str(rs.get("entry_id", uuid.uuid4()))
        folder = _safe_member_folder(r.member_name, used_folders)
        member_folders.append(folder)
        member_dfs.append(r.df)
        member_names.append(r.member_name)
        member_entry_ids.append(entry_id)
        summary_rows.append(
            {
                "compound_name": r.member_name,
                "entry_id": entry_id,
                "imp_score": rs.get("imp_score"),
                "imp_candidates": rs.get("imp_candidates", 0),
                "total_compounds": rs.get("total_compounds", 0),
                "total_activities": rs.get("total_activities", 0),
                # D-S2 free Triage leaderboard columns -- already in member_summary
                # at finalize (zero recompute). .get() defaults so a missing key
                # degrades to None/0, never KeyError (T-24-04-03).
                "pains_count": rs.get("pains_count", 0),
                "brenk_count": rs.get("brenk_count", 0),
                "nih_count": rs.get("nih_count", 0),
                "qed": rs.get("qed"),
                "num_outliers": rs.get("num_outliers", 0),
                "pdb_structures_count": rs.get("pdb_structures_count", 0),
                "drug_indications_count": rs.get("drug_indications_count", 0),
                "compounds_with_indications": rs.get("compounds_with_indications", 0),
                "total_similar": rs.get("total_similar", 0),
                "classification_available": rs.get("classification_available"),
            }
        )

    combined_df, truncated = build_combined_activities(
        member_dfs, member_names, member_entry_ids
    )
    summary_df = pd.DataFrame(summary_rows)

    # Aggregate stats over the successful members.
    imp_scores = [
        s["imp_score"] for s in summary_rows if s.get("imp_score") is not None
    ]
    avg_imp = float(sum(imp_scores) / len(imp_scores)) if imp_scores else None
    imp_candidate_count = int(sum(int(s.get("imp_candidates") or 0) for s in summary_rows))
    unique_targets = 0
    if not combined_df.empty:
        for col in ("Target_ChEMBL_ID", "ChEMBL_ID", "Target"):
            if col in combined_df.columns:
                unique_targets = int(combined_df[col].nunique())
                break

    # Stage the ZIP INSIDE the destination dir (same filesystem) then atomically
    # os.replace -- avoids the cross-device link error os.replace raises when the
    # temp dir and RESULTS_DIR live on different mounts (common in Docker).
    storage_path = get_collection_storage_path(collection_id)  # collections/{shard}/{id}.zip
    dest = os.path.join(str(settings.RESULTS_DIR), storage_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    zip_tmp = dest + ".tmp"
    try:
        with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            # Top-level tables.
            zf.writestr("collection_summary.csv", summary_df.to_csv(index=False))
            zf.writestr(
                "combined_activities.csv", combined_df.to_csv(index=False)
            )
            # D-S2-ARCH: ONE structured aggregate artifact the Evidence view loads
            # directly (no on-demand fallback). Option A re-reads the 3 Tier-3 CSVs
            # from each member ZIP ROOT under the same arcname guard (D-S2-SOURCE).
            aggregate = _build_collection_aggregate(succeeded, member_entry_ids)
            zf.writestr(
                "collection_aggregate.json",
                json.dumps(aggregate, default=str),
            )
            if truncated:
                zf.writestr(
                    "WARNING.txt",
                    "combined_activities.csv was truncated to "
                    f"{COMBINED_ROW_CAP} rows (D-03). Per-member sections under "
                    "compounds/ are complete.\n",
                )
            # Each member's own ZIP contents nested under compounds/{folder}/.
            for r, folder in zip(succeeded, member_folders):
                for member_zip in r.files:
                    if not member_zip or not os.path.exists(member_zip):
                        continue
                    _nest_member_zip(zf, member_zip, folder)
        os.replace(zip_tmp, dest)
    except Exception:
        if os.path.exists(zip_tmp):
            try:
                os.unlink(zip_tmp)
            except OSError:
                pass
        raise

    _upload_collection_zip(dest, collection_id)

    return {
        "storage_path": storage_path,
        "avg_imp_score": avg_imp,
        "imp_candidate_count": imp_candidate_count,
        "unique_targets": unique_targets,
        "combined_truncated": truncated,
    }


def _nest_member_zip(zf: zipfile.ZipFile, member_zip: str, folder: str) -> None:
    """Copy a member ZIP's entries under ``compounds/{folder}/`` (arcname-guarded)."""
    with zipfile.ZipFile(member_zip, "r") as src:
        for info in src.infolist():
            inner = info.filename
            # T-23-04-T1: never let a member ZIP entry escape its folder.
            if inner.startswith("/") or ".." in inner.split("/"):
                logger.warning(
                    f"Skipping unsafe member ZIP arcname: {inner!r}"
                )
                continue
            arcname = f"compounds/{folder}/{inner}"
            zf.writestr(arcname, src.read(info.filename))


def _read_tier3_csv(src: zipfile.ZipFile, names: set[str], filename: str) -> list[dict]:
    """Re-read ONE known Tier-3 CSV from a member ZIP ROOT (Option A, D-S2-SOURCE).

    Returns a list of row dicts, or ``[]`` when the CSV is absent (a member with
    no drug indications writes no ``drug_indications.csv``). Honors the SAME
    arcname guard ``_nest_member_zip`` uses (reject leading ``/`` or ``..``) as
    defence in depth even though only the 3 KNOWN root filenames are ever read --
    a traversal-named impostor (e.g. ``../drug_indications.csv``) never matches
    the exact root name AND would be rejected by the guard regardless (T-24-04-01).
    """
    if filename not in names:
        return []
    # Defence in depth: never read an entry whose stored name escapes the root.
    if filename.startswith("/") or ".." in filename.split("/"):
        logger.warning("Skipping unsafe Tier-3 arcname: %r", filename)
        return []
    try:
        with src.open(filename) as fh:
            frame = pd.read_csv(io.BytesIO(fh.read()))
    except Exception as e:  # a malformed CSV in one member must not abort finalize
        logger.warning("Could not parse Tier-3 CSV %r: %s", filename, e)
        return []
    if frame.empty:
        return []
    # JSON-safe: collection_aggregate.json is the SINGLE source the Evidence view
    # (24-05) loads with no fallback, so numbers must arrive as numbers. Replace
    # NaN -> None (bare ``NaN`` is invalid strict JSON) and let _json_safe coerce
    # any numpy scalar to a native int/float (np.int64 is NOT an int subclass, so
    # json.dumps(default=str) would otherwise silently STRINGIFY it).
    frame = frame.where(pd.notnull(frame), None)
    return [
        {k: _json_safe(v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _json_safe(value: Any) -> Any:
    """Coerce a numpy/pandas scalar to a JSON-native type (no silent stringify).

    ``np.int64`` is not an ``int`` subclass, so ``json.dumps(..., default=str)``
    would turn it into a string; ``np.float64`` is a ``float`` subclass and is
    fine. Float ``NaN`` -> ``None`` (bare ``NaN`` is invalid strict JSON).
    Anything already native passes through unchanged.
    """
    if value is None:
        return None
    # numpy scalars expose .item() -> the native Python equivalent.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (ValueError, TypeError):
            return value
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def _build_collection_aggregate(
    succeeded: list["MemberResult"],
    member_entry_ids: list[str],
) -> dict[str, dict]:
    """Build the ``collection_aggregate.json`` payload via the Option-A re-read.

    For each succeeded member, open each member ZIP and re-read ONLY the 3 KNOWN
    Tier-3 CSVs at the ROOT (drug_indications / all_similar_molecules / pdb_summary)
    -- never iterating arbitrary entries (T-24-04-01). Absent CSVs degrade to ``[]``
    (T-24-04-03). The result is a dict keyed by member ``entry_id`` ->
    ``{indications, all_similar, pdb, classification}`` (RESEARCH Open Q2). The
    per-member ``all_similar`` catalog is row-capped (T-24-04-02 DoS bound).

    ``member_entry_ids`` is the SAME list ``_assemble_and_upload_zip_sync`` already
    computed for ``collection_summary.csv`` / ``combined_activities.csv`` -- passed
    in (not recomputed) so the aggregate key joins to those tables on entry_id even
    when a member lacks ``entry_id`` (a recomputed ``uuid4()`` would diverge).
    """
    aggregate: dict[str, dict] = {}
    for r, entry_id in zip(succeeded, member_entry_ids):
        rs = r.member_summary or {}
        indications: list[dict] = []
        all_similar: list[dict] = []
        pdb: list[dict] = []
        for member_zip in r.files:
            if not member_zip or not os.path.exists(member_zip):
                continue
            try:
                with zipfile.ZipFile(member_zip, "r") as src:
                    names = set(src.namelist())
                    indications.extend(
                        _read_tier3_csv(src, names, _AGGREGATE_TIER3_FILES["indications"])
                    )
                    all_similar.extend(
                        _read_tier3_csv(src, names, _AGGREGATE_TIER3_FILES["all_similar"])
                    )
                    pdb.extend(
                        _read_tier3_csv(src, names, _AGGREGATE_TIER3_FILES["pdb"])
                    )
            except zipfile.BadZipFile as e:
                logger.warning("Skipping unreadable member ZIP %r: %s", member_zip, e)
                continue
        # T-24-04-02: bound the per-member all_similar catalog (DoS guard).
        if len(all_similar) > AGGREGATE_ALL_SIMILAR_ROW_CAP:
            all_similar = all_similar[:AGGREGATE_ALL_SIMILAR_ROW_CAP]
        # Classification / IMP dims already ride in member_summary at finalize
        # (combined.csv carries the per-row detail; this is the headline rollup).
        classification = {
            "classification_available": rs.get("classification_available"),
            "imp_score": rs.get("imp_score"),
            "imp_candidates": rs.get("imp_candidates"),
        }
        aggregate[entry_id] = {
            "member_name": r.member_name,
            "indications": indications,
            "all_similar": all_similar,
            "pdb": pdb,
            "classification": classification,
        }
    return aggregate


# ---------------------------------------------------------------------------
# Sync DB / storage seams (called via run_in_executor or directly from sync code)
# ---------------------------------------------------------------------------

def _load_members_sync(job_id) -> tuple[uuid.UUID, list[dict]] | None:
    """Load a collection's (collection_id, member list) by job_id (D-02)."""
    jid = _coerce_uuid(job_id, "job_id")
    with get_db_session() as db:
        collection = collection_repo.get_by_job_id(db, jid)
        if collection is None:
            return None
        config = collection.members_config or {}
        member_inputs = list(config.get("members", []))
        return collection.id, member_inputs


def _report_aggregate_progress(loop, job_id, done: int, total: int) -> None:
    """Schedule an aggregate progress write (D-09) without blocking the member.

    Fire-and-forget on the executor so member coroutines aren't serialised behind a
    DB write. Aggregate completed/total ONLY -- members never report per-member
    progress.
    """
    pct = (done / total * 100.0) if total else 100.0
    loop.run_in_executor(
        None,
        _update_progress_sync,
        job_id,
        round(pct, 2),
        f"Processed {done}/{total} members",
    )


def _update_progress_sync(job_id, progress: float, step: str) -> None:
    with get_db_session() as db:
        from backend.services.job_service import job_service

        job_service.update_progress(
            db, str(job_id), progress, step, status=JobStatus.PROCESSING
        )


def _finalize_job_sync(job_id, collection_id: str, stats: dict[str, Any]) -> None:
    """Persist collection stats + mark the job COMPLETED ONCE (D-09 aggregate)."""
    with get_db_session() as db:
        from backend.services.job_service import job_service

        collection_repo.update_stats(
            db,
            _coerce_uuid(collection_id, "collection_id"),
            compound_count=stats.get("compound_count", 0),
            member_failed_count=stats.get("member_failed_count", 0),
            avg_imp_score=stats.get("avg_imp_score"),
            imp_candidate_count=stats.get("imp_candidate_count", 0),
            unique_targets=stats.get("unique_targets", 0),
        )
        if stats.get("storage_path"):
            collection_repo.update_storage_path(
                db,
                _coerce_uuid(collection_id, "collection_id"),
                stats["storage_path"],
            )
        # D-PF-6: bridge the per-member failure rows from the aggregate ``stats``
        # into the LINKED JOB's result_summary so the detail endpoint can surface
        # them on a COMPLETED-with-failures collection (the collection columns hold
        # only counts/scores, not the {name, error, cascade_results} payload). The
        # collection finalize never routes through ``mark_pending_upload`` (that
        # path is single-compound-only), so we merge result_summary here directly.
        failed_members = stats.get("failed_members")
        if failed_members:  # only genuine partial failures write the key (no empty [])
            job = job_service.get_job(db, str(job_id))
            if job is not None:
                summary = dict(job.result_summary or {})
                summary["failed_members"] = failed_members
                job.result_summary = summary
        db.commit()
        # PROCESSING -> PENDING_UPLOAD -> COMPLETED (single aggregate finalise).
        job_service.update_progress(
            db, str(job_id), 100.0, "Finalizing collection",
            status=JobStatus.PENDING_UPLOAD,
        )
        job_service.mark_completed(db, str(job_id))


def _fail_job_sync(job_id, error_message: str, failed_members: list | None = None) -> None:
    with get_db_session() as db:
        from backend.services.job_service import job_service

        job_service.fail_job(
            db, str(job_id), error_message, failed_members=failed_members
        )


def _upload_collection_zip(local_path: str, collection_id: str) -> bool:
    """Upload the collection ZIP to the ``collections/`` blob prefix (D-12).

    Uses the generic blob client with :func:`get_collection_storage_path` (the
    entry_id-keyed azure_sync helpers compute the ``results/`` prefix and cannot
    target ``collections/``). No-op when Azure is not configured (the ZIP already
    lives on the local results tree).
    """
    if not is_azure_configured():
        return True
    blob_name = get_collection_storage_path(collection_id)
    try:
        from backend.core.azure_sync import _get_blob_client

        blob = _get_blob_client(blob_name)
        if blob is None:
            return False
        with open(local_path, "rb") as f:
            blob.upload_blob(f, overwrite=True)
        logger.info(f"Uploaded collection ZIP to Azure ({blob_name})")
        return True
    except Exception as e:
        logger.error(f"Failed to upload collection ZIP {collection_id}: {e}")
        return False


def _delete_collection_zip(collection_id: str, storage_path: str | None) -> None:
    """Remove the collection ZIP locally and from Azure (best-effort, D-11)."""
    rel = storage_path or get_collection_storage_path(collection_id)
    # Local copy.
    local_path = os.path.join(str(settings.RESULTS_DIR), rel)
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
    except OSError as e:
        logger.warning(f"Could not remove local collection ZIP {local_path}: {e}")
    # Azure copy.
    if is_azure_configured():
        try:
            from backend.core.azure_sync import _get_blob_client

            blob = _get_blob_client(rel)
            if blob is not None and blob.exists():
                blob.delete_blob()
        except Exception as e:
            logger.warning(
                f"Could not remove Azure collection ZIP {rel}: {e}"
            )


def _coerce_uuid(value, name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"Invalid {name}: {value!r}") from e
