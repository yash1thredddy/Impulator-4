"""One-shot backfill: rewrite historical result ZIPs to the new IMP column shape.

Phase 21 / Plan 21-05 -- IMP Score Presentation Overhaul.

Old result ZIPs (produced before Plans 21-01..21-04) carry the legacy
``IMP_Classification`` (qualitative bucket) and ``IMP_Confidence`` columns
in their embedded CSV(s) and JSON metadata. The new presentation layer
emits ``IMP_Score_Integer`` (banker's-rounded ``round(raw * 100)``) instead.

This script reconciles historical artefacts with the new contract by:

1. Iterating compounds (each compound owns one entry-id-keyed Azure blob).
2. Downloading each ZIP into a tempdir via
   ``download_result_from_azure_by_entry_id`` (T-21-09: re-uses the
   path-traversal guard at ``azure_sync.py:511-517`` -- we do NOT call
   ``_get_blob_client`` directly).
3. Rewriting every ``*.csv`` inside the ZIP:
       - ``df.drop(columns=["IMP_Classification", "IMP_Confidence"], errors="ignore")``
       - Add ``IMP_Score_Integer`` derived from ``IMP_Final_Score`` (or
         the existing ``IMP_Score`` / ``imp_score`` column) via
         ``backend.modules.imp_presentation.format_imp_score``.
4. Rewriting every ``*.json`` inside the ZIP:
       - ``payload.pop("IMP_Classification", None)``
       - ``payload.pop("IMP_Confidence", None)``
   For nested structures we recurse but only ever ``pop`` the two
   forbidden keys -- no other transformation.
5. Re-zipping the contents with ``zipfile.ZIP_DEFLATED`` and uploading
   via ``upload_result_to_azure_by_entry_id`` (same path-traversal-safe
   wrapper).
6. Marking ``imp_zip_backfill_status.status = 'done'`` (or ``'failed'``
   with ``error_message = str(e)`` -- T-21-11).

Threat model
------------
- T-21-09: only call the *by_entry_id* helpers; both wrap ``is_path_within``.
- T-21-10: ZIPs contain CSV + JSON only -- use ``pandas.read_csv`` and
  ``json.load`` exclusively. No unsafe deserialisation formats.
- T-21-11: on failure, persist ``str(e)`` (short, user-safe) to
  ``error_message``; full stack goes to structlog ``logger.exception``.
- T-21-12: idempotence via the state machine
  ``{pending, done, failed, skipped}``; per-compound try/except keeps a
  single failure from aborting the batch.

Usage
-----
    .impenv/bin/python -m backend.scripts.backfill_imp_columns [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import text

# Add project root to path for ``python -m backend.scripts.backfill_imp_columns``
# invocations from any cwd.
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.azure_sync import (  # noqa: E402
    download_result_from_azure_by_entry_id,
    upload_result_to_azure_by_entry_id,
)
from backend.core.database import get_db_session  # noqa: E402
from backend.modules.imp_presentation import format_imp_score  # noqa: E402

logger = structlog.get_logger(__name__)

# Legacy columns to drop (CSV + JSON).
LEGACY_COLUMNS: tuple[str, ...] = ("IMP_Classification", "IMP_Confidence")

# Candidate raw-score columns (we prefer the first present, case-sensitive,
# in source-of-truth order).
SCORE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "IMP_Final_Score",
    "IMP_Score",
    "imp_score",
)

# Status values mirror the Postgres enum imp_zip_backfill_status_enum
# (alembic 0006). Kept here as plain strings -- no ORM model needed for
# a one-shot script.
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# In-ZIP transforms
# ---------------------------------------------------------------------------


def _rewrite_csv(csv_path: Path) -> None:
    """Drop legacy columns and add ``IMP_Score_Integer`` in-place.

    Uses ``df.drop(columns=..., errors='ignore')`` so a CSV missing one or
    both legacy columns is treated as a no-op rather than an error. The
    new ``IMP_Score_Integer`` column is only added when a recognised raw
    score column is present.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=list(LEGACY_COLUMNS), errors="ignore")

    score_col = next(
        (c for c in SCORE_COLUMN_CANDIDATES if c in df.columns),
        None,
    )
    if score_col is not None:
        # format_imp_score handles None/NaN by returning None, which
        # round-trips as an empty cell in the rewritten CSV (Int64 nullable).
        df["IMP_Score_Integer"] = (
            df[score_col]
            .map(lambda v: format_imp_score(v) if pd.notna(v) else None)
            .astype("Int64")
        )

    df.to_csv(csv_path, index=False)


def _strip_legacy_keys(node: Any) -> Any:
    """Recursively pop the two legacy keys from any nested dict.

    Only ``dict.pop`` is used (T-21-10) -- no other mutation of the JSON
    payload. Lists are recursed into; scalars are returned unchanged.
    """
    if isinstance(node, dict):
        for key in LEGACY_COLUMNS:
            node.pop(key, None)
        for v in node.values():
            _strip_legacy_keys(v)
    elif isinstance(node, list):
        for item in node:
            _strip_legacy_keys(item)
    return node


def _rewrite_json(json_path: Path) -> None:
    """Drop legacy keys from a JSON file in-place."""
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    _strip_legacy_keys(payload)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _rewrite_zip(src_zip: Path, dst_zip: Path) -> None:
    """Extract -> rewrite CSV+JSON -> re-zip with DEFLATED compression.

    Walks the staged tree and applies ``_rewrite_csv`` to every ``*.csv``
    and ``_rewrite_json`` to every ``*.json``. All other files (rare in
    our result ZIPs, but possible) pass through unchanged.
    """
    with tempfile.TemporaryDirectory(prefix="imp_backfill_stage_") as stage_str:
        stage = Path(stage_str)
        with zipfile.ZipFile(src_zip, "r") as zf:
            zf.extractall(stage)

        for path in stage.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".csv":
                _rewrite_csv(path)
            elif path.suffix.lower() == ".json":
                _rewrite_json(path)

        with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as out:
            for path in stage.rglob("*"):
                if path.is_file():
                    out.write(path, arcname=path.relative_to(stage))


# ---------------------------------------------------------------------------
# Per-compound driver
# ---------------------------------------------------------------------------


def process_one(entry_id: str, work_dir: Path) -> None:
    """Download -> rewrite -> upload one compound's result ZIP.

    Raises on any failure -- the caller catches and persists ``str(e)`` to
    ``imp_zip_backfill_status.error_message``.
    """
    src_zip = work_dir / f"{entry_id}.src.zip"
    dst_zip = work_dir / f"{entry_id}.rewritten.zip"

    if not download_result_from_azure_by_entry_id(entry_id, str(src_zip)):
        raise RuntimeError(f"download failed for entry_id={entry_id}")

    _rewrite_zip(src_zip, dst_zip)

    if not upload_result_to_azure_by_entry_id(str(dst_zip), entry_id):
        raise RuntimeError(f"upload failed for entry_id={entry_id}")


# ---------------------------------------------------------------------------
# Discovery + state-machine I/O (raw SQL -- no ORM model for the tracking
# table; this is a one-shot script and adding a model would inflate the
# import graph for production code).
# ---------------------------------------------------------------------------


def _select_pending(session, limit: int | None) -> list[uuid.UUID]:
    """Return compound entry_ids whose backfill row is missing or non-terminal.

    LEFT JOIN ensures fresh compounds (no tracking row yet) are picked
    up as ``pending``. ``done`` rows are skipped on re-run; ``failed``
    rows are retried (operator can manually mark them ``skipped`` if a
    permanent failure is identified).
    """
    sql = (
        "SELECT c.entry_id "
        "FROM compounds c "
        "LEFT JOIN imp_zip_backfill_status s ON s.entry_id = c.entry_id "
        "WHERE c.storage_path IS NOT NULL "
        "  AND (s.status IS NULL OR s.status IN ('pending', 'failed')) "
        "ORDER BY c.processed_at"
    )
    if limit is not None:
        sql = sql + f" LIMIT {int(limit)}"
    rows = session.execute(text(sql)).scalars().all()
    return list(rows)


def _mark_done(session, entry_id: uuid.UUID) -> None:
    session.execute(
        text(
            """
            INSERT INTO imp_zip_backfill_status (entry_id, status, error_message, processed_at)
            VALUES (:entry_id, 'done', NULL, now())
            ON CONFLICT (entry_id) DO UPDATE
                SET status = 'done',
                    error_message = NULL,
                    processed_at = now()
            """
        ),
        {"entry_id": entry_id},
    )


def _mark_failed(session, entry_id: uuid.UUID, error_message: str) -> None:
    # Truncate defensively -- ``str(e)`` should be short but we keep the
    # column from unbounded growth on pathological errors.
    short_msg = (error_message or "")[:2000]
    session.execute(
        text(
            """
            INSERT INTO imp_zip_backfill_status (entry_id, status, error_message, processed_at)
            VALUES (:entry_id, 'failed', :msg, now())
            ON CONFLICT (entry_id) DO UPDATE
                SET status = 'failed',
                    error_message = :msg,
                    processed_at = now()
            """
        ),
        {"entry_id": entry_id, "msg": short_msg},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_backfill(limit: int | None = None, dry_run: bool = False) -> dict[str, int]:
    """Iterate pending compounds and rewrite their result ZIPs.

    Returns a counts dict ``{discovered, done, failed, skipped}``. The
    ``skipped`` bucket only fires in ``--dry-run`` mode (where no work is
    performed) so historical-state idempotence is preserved.

    Per-compound exceptions are caught and recorded in the tracking
    table; the batch always runs to completion (T-21-12).
    """
    counts = {"discovered": 0, "done": 0, "failed": 0, "skipped": 0}

    with get_db_session() as session:
        entry_ids = _select_pending(session, limit)
        counts["discovered"] = len(entry_ids)
        logger.info(
            "backfill_discovered",
            count=len(entry_ids),
            limit=limit,
            dry_run=dry_run,
        )

        if dry_run:
            counts["skipped"] = len(entry_ids)
            for eid in entry_ids:
                logger.info("backfill_would_process", entry_id=str(eid))
            return counts

        with tempfile.TemporaryDirectory(prefix="imp_backfill_") as work_str:
            work_dir = Path(work_str)
            for eid in entry_ids:
                eid_str = str(eid)
                try:
                    process_one(eid_str, work_dir)
                    _mark_done(session, eid)
                    session.commit()
                    counts["done"] += 1
                    logger.info("backfill_done", entry_id=eid_str)
                except Exception as e:  # noqa: BLE001 -- per-job isolation (T-21-12)
                    # Full stack to structlog only (T-21-11)
                    logger.exception("backfill_failed", entry_id=eid_str, error=str(e))
                    # Short str(e) only to error_message (T-21-11)
                    session.rollback()
                    _mark_failed(session, eid, str(e))
                    session.commit()
                    counts["failed"] += 1
                finally:
                    # Best-effort cleanup of per-compound staged files so
                    # the work_dir doesn't accumulate gigabytes mid-batch.
                    for stale in work_dir.glob(f"{eid_str}.*.zip"):
                        try:
                            stale.unlink()
                        except OSError:
                            pass
                    # Defensive: also clean any orphaned subdirs
                    for stale in work_dir.iterdir():
                        if stale.is_dir():
                            shutil.rmtree(stale, ignore_errors=True)

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot backfill of historical result ZIPs to the new IMP "
            "column shape (drops IMP_Classification + IMP_Confidence; "
            "adds IMP_Score_Integer)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pending compounds (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be processed without downloading or uploading.",
    )
    args = parser.parse_args(argv)

    counts = run_backfill(limit=args.limit, dry_run=args.dry_run)
    logger.info("backfill_complete", **counts)
    # Non-zero exit if any failures so cron / CI can flag it.
    return 1 if counts["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
