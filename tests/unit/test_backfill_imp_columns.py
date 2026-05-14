"""Unit tests for backend.scripts.backfill_imp_columns.

Strategy
--------
Pure-Python coverage of the in-ZIP transforms plus the per-compound
driver, with Azure and the DB fully mocked. We never touch real Azure
and never spin up a Postgres container -- the script's correctness is
expressible at the transform layer.

Threat-model coverage
---------------------
- T-21-10: assert legacy columns / keys are gone, no other transformation.
- T-21-11: assert error_message receives ``str(e)`` only (no traceback
  string).
- T-21-12: assert one failure does NOT abort the batch; assert idempotence
  (a ``done`` row is left untouched on re-run).
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.scripts import backfill_imp_columns as bf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_zip_bytes() -> bytes:
    """Build an in-memory ZIP with a CSV + JSON carrying the legacy columns."""
    df = pd.DataFrame(
        {
            "compound_name": ["A", "B", "C"],
            "IMP_Final_Score": [0.10, 0.55, None],
            "IMP_Classification": ["Low", "Mid", "High"],
            "IMP_Confidence": [0.9, 0.8, 0.7],
            "Other_Column": [1, 2, 3],
        }
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    json_payload = {
        "summary": {
            "name": "X",
            "IMP_Classification": "Mid",
            "IMP_Confidence": 0.8,
        },
        "rows": [
            {"name": "row1", "IMP_Classification": "Low", "Other": "keep"},
            {"name": "row2", "Other": "keep"},
        ],
    }
    json_bytes = json.dumps(json_payload).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", csv_bytes)
        zf.writestr("metadata.json", json_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# In-ZIP transform tests (no Azure, no DB)
# ---------------------------------------------------------------------------


class TestRewriteCsv:
    def test_drops_legacy_and_adds_integer(self, tmp_path: Path) -> None:
        csv = tmp_path / "r.csv"
        pd.DataFrame(
            {
                "IMP_Final_Score": [0.10, 0.55, None],
                "IMP_Classification": ["Low", "Mid", "High"],
                "IMP_Confidence": [0.9, 0.8, 0.7],
                "keep_me": [1, 2, 3],
            }
        ).to_csv(csv, index=False)

        bf._rewrite_csv(csv)

        out = pd.read_csv(csv)
        assert "IMP_Classification" not in out.columns
        assert "IMP_Confidence" not in out.columns
        assert "keep_me" in out.columns  # untouched
        assert "IMP_Score_Integer" in out.columns
        # Banker's rounding: 0.10 -> 10, 0.55 -> 55 (round-half-to-even applies
        # to .5 ties; 55 isn't a tie). None -> NaN.
        assert int(out["IMP_Score_Integer"].iloc[0]) == 10
        assert int(out["IMP_Score_Integer"].iloc[1]) == 55
        assert pd.isna(out["IMP_Score_Integer"].iloc[2])

    def test_missing_legacy_columns_is_noop(self, tmp_path: Path) -> None:
        """errors='ignore' on df.drop -- absent legacy cols must not raise."""
        csv = tmp_path / "r.csv"
        pd.DataFrame({"IMP_Final_Score": [0.42], "keep": [1]}).to_csv(csv, index=False)

        bf._rewrite_csv(csv)

        out = pd.read_csv(csv)
        assert "IMP_Score_Integer" in out.columns
        assert int(out["IMP_Score_Integer"].iloc[0]) == 42
        assert "keep" in out.columns

    def test_no_score_column_skips_integer(self, tmp_path: Path) -> None:
        csv = tmp_path / "r.csv"
        pd.DataFrame(
            {"IMP_Classification": ["Mid"], "IMP_Confidence": [0.5], "keep": [1]}
        ).to_csv(csv, index=False)

        bf._rewrite_csv(csv)

        out = pd.read_csv(csv)
        assert "IMP_Classification" not in out.columns
        assert "IMP_Confidence" not in out.columns
        assert "IMP_Score_Integer" not in out.columns
        assert "keep" in out.columns


class TestRewriteJson:
    def test_drops_legacy_keys_recursively(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps(
                {
                    "top": "v",
                    "IMP_Classification": "Mid",
                    "IMP_Confidence": 0.8,
                    "nested": {
                        "IMP_Classification": "Low",
                        "keep": "yes",
                    },
                    "list": [
                        {"IMP_Confidence": 0.1, "keep": 1},
                        {"keep": 2},
                    ],
                }
            )
        )

        bf._rewrite_json(path)

        out = json.loads(path.read_text())
        assert "IMP_Classification" not in out
        assert "IMP_Confidence" not in out
        assert out["top"] == "v"
        assert "IMP_Classification" not in out["nested"]
        assert out["nested"]["keep"] == "yes"
        assert "IMP_Confidence" not in out["list"][0]
        assert out["list"][0]["keep"] == 1
        assert out["list"][1]["keep"] == 2


class TestRewriteZip:
    def test_end_to_end(self, tmp_path: Path, sample_zip_bytes: bytes) -> None:
        src = tmp_path / "src.zip"
        dst = tmp_path / "dst.zip"
        src.write_bytes(sample_zip_bytes)

        bf._rewrite_zip(src, dst)

        # Inspect the rewritten ZIP
        with zipfile.ZipFile(dst, "r") as zf:
            names = sorted(zf.namelist())
            assert names == ["metadata.json", "results.csv"]

            csv_text = zf.read("results.csv").decode("utf-8")
            df_out = pd.read_csv(io.StringIO(csv_text))
            assert "IMP_Classification" not in df_out.columns
            assert "IMP_Confidence" not in df_out.columns
            assert "IMP_Score_Integer" in df_out.columns

            meta = json.loads(zf.read("metadata.json"))
            assert "IMP_Classification" not in meta["summary"]
            assert "IMP_Confidence" not in meta["summary"]
            assert "IMP_Classification" not in meta["rows"][0]


# ---------------------------------------------------------------------------
# Driver tests (mock Azure + DB session)
# ---------------------------------------------------------------------------


def _make_fake_db_session(entry_ids: list[uuid.UUID]) -> MagicMock:
    """Build a fake sync session that returns the given entry_ids from select().

    Used as the value yielded by a patched ``get_db_session()`` context
    manager. Records execute() calls so the test can assert state-machine
    writes were performed with the expected (entry_id, msg) bindings.
    """
    session = MagicMock(name="FakeSession")
    session.execute_calls = []

    def execute(stmt, params=None):
        session.execute_calls.append((str(stmt), params))
        result = MagicMock()
        # The first execute() call is the discovery SELECT.
        # We detect it by absence of params (the marker writes pass params).
        if params is None:
            result.scalars.return_value.all.return_value = list(entry_ids)
        return result

    session.execute.side_effect = execute
    return session


def _ctx(session: MagicMock):
    """Return a context manager that yields ``session``."""

    class _CM:
        def __enter__(self_inner):
            return session

        def __exit__(self_inner, exc_type, exc, tb):
            return False

    return _CM()


def _stage_zip(src_path: Path, sample_zip_bytes: bytes) -> bool:
    """Side-effect for download_result_from_azure_by_entry_id: write a fake zip."""
    Path(src_path).write_bytes(sample_zip_bytes)
    return True


class TestRunBackfillSuccess:
    def test_happy_path_marks_done_and_uploads_rewritten(
        self, sample_zip_bytes: bytes
    ) -> None:
        eid = uuid.uuid4()
        session = _make_fake_db_session([eid])

        uploaded_paths: list[str] = []

        def fake_upload(local_path: str, entry_id: str) -> bool:
            uploaded_paths.append(local_path)
            # Sanity: uploaded zip must have the new column shape
            with zipfile.ZipFile(local_path, "r") as zf:
                csv_text = zf.read("results.csv").decode("utf-8")
                df = pd.read_csv(io.StringIO(csv_text))
                assert "IMP_Classification" not in df.columns
                assert "IMP_Confidence" not in df.columns
                assert "IMP_Score_Integer" in df.columns
            return True

        with (
            patch.object(bf, "get_db_session", return_value=_ctx(session)),
            patch.object(
                bf,
                "download_result_from_azure_by_entry_id",
                side_effect=lambda entry_id, local_path: _stage_zip(
                    local_path, sample_zip_bytes
                ),
            ),
            patch.object(
                bf, "upload_result_to_azure_by_entry_id", side_effect=fake_upload
            ),
        ):
            counts = bf.run_backfill()

        assert counts == {"discovered": 1, "done": 1, "failed": 0, "skipped": 0}
        assert len(uploaded_paths) == 1
        # state-machine INSERT for the 'done' status, with the entry_id bind
        marker_writes = [c for c in session.execute_calls if c[1] is not None]
        assert any(
            params.get("entry_id") == eid and "'done'" in stmt
            for stmt, params in marker_writes
        )


class TestRunBackfillFailureIsolated:
    def test_one_failure_does_not_abort_batch_and_records_str_e(
        self, sample_zip_bytes: bytes
    ) -> None:
        eid_ok = uuid.uuid4()
        eid_bad = uuid.uuid4()
        session = _make_fake_db_session([eid_bad, eid_ok])

        def fake_download(entry_id: str, local_path: str) -> bool:
            if entry_id == str(eid_bad):
                # Simulate a download failure by returning False -- the
                # script translates that into a RuntimeError.
                return False
            _stage_zip(local_path, sample_zip_bytes)
            return True

        with (
            patch.object(bf, "get_db_session", return_value=_ctx(session)),
            patch.object(
                bf, "download_result_from_azure_by_entry_id", side_effect=fake_download
            ),
            patch.object(bf, "upload_result_to_azure_by_entry_id", return_value=True),
        ):
            counts = bf.run_backfill()

        # The good compound still got processed (T-21-12 batch isolation)
        assert counts["discovered"] == 2
        assert counts["done"] == 1
        assert counts["failed"] == 1

        marker_writes = [c for c in session.execute_calls if c[1] is not None]

        # 'failed' marker carries the short str(e), NOT a traceback
        failed_writes = [
            (stmt, params)
            for stmt, params in marker_writes
            if "'failed'" in stmt and params.get("entry_id") == eid_bad
        ]
        assert len(failed_writes) == 1
        msg = failed_writes[0][1]["msg"]
        assert "download failed" in msg
        # Traceback strings contain "Traceback (most recent call last)";
        # we must not see that in the persisted error_message.
        assert "Traceback" not in msg
        # And short (<= 2000 chars per script's defensive truncation)
        assert len(msg) <= 2000

        # 'done' marker was still emitted for the good compound
        done_writes = [
            (stmt, params)
            for stmt, params in marker_writes
            if "'done'" in stmt and params.get("entry_id") == eid_ok
        ]
        assert len(done_writes) == 1


class TestRunBackfillDryRun:
    def test_dry_run_skips_everything_and_does_not_touch_azure(self) -> None:
        eids = [uuid.uuid4(), uuid.uuid4()]
        session = _make_fake_db_session(eids)

        with (
            patch.object(bf, "get_db_session", return_value=_ctx(session)),
            patch.object(bf, "download_result_from_azure_by_entry_id") as mock_dl,
            patch.object(bf, "upload_result_to_azure_by_entry_id") as mock_up,
        ):
            counts = bf.run_backfill(dry_run=True)

        assert counts == {"discovered": 2, "done": 0, "failed": 0, "skipped": 2}
        mock_dl.assert_not_called()
        mock_up.assert_not_called()
        # Only the discovery SELECT should have run -- no marker writes.
        marker_writes = [c for c in session.execute_calls if c[1] is not None]
        assert marker_writes == []
