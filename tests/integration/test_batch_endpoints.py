"""
Integration tests for batch job endpoints and availability checks.

Tests:
- POST /api/v1/jobs/batch (batch submission)
- GET /api/v1/jobs/batch/{batch_id} (batch summary)
- POST /api/v1/jobs/batch/{batch_id}/cancel (batch cancellation)
- POST /api/v1/jobs/check-availability (single availability)
- POST /api/v1/jobs/check-availability/batch (batch availability)

All external APIs are mocked -- no network calls.
Responses validated through Pydantic models (TEST-04).
"""
import uuid as _uuid
from unittest.mock import patch, MagicMock

from backend.models.schemas import (
    BatchResponse,
    BatchSummary,
    CancelResponse,
    CheckAvailabilityResponse,
    CheckAvailabilityBatchResponse,
)


def _sid():
    """Generate a valid UUID v4 session ID."""
    return str(_uuid.uuid4())


# ============================================================================
# Batch Submit
# ============================================================================


class TestBatchSubmit:
    """Tests for POST /api/v1/jobs/batch."""

    def test_submit_batch_success(self, client):
        """Batch submission with 3 compounds returns 201 with batch_id and jobs list."""
        sid = _sid()
        resp = client.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": [
                    {"compound_name": "BatchA", "author_name": "Author", "smiles": "CCO"},
                    {"compound_name": "BatchB", "author_name": "Author", "smiles": "CCCO"},
                    {"compound_name": "BatchC", "author_name": "Author", "smiles": "CCCCO"},
                ],
            },
            headers={"X-Session-ID": sid},
        )
        assert resp.status_code == 201

        result = BatchResponse(**resp.json())
        assert result.batch_id is not None
        assert len(result.jobs) == 3
        assert result.total_submitted == 3

    def test_submit_batch_empty_list(self, client):
        """POST with empty compounds list returns 422."""
        resp = client.post(
            "/api/v1/jobs/batch",
            json={"compounds": []},
            headers={"X-Session-ID": _sid()},
        )
        assert resp.status_code == 422

    def test_submit_batch_validates_smiles(self, client):
        """POST with invalid SMILES returns 422."""
        resp = client.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": [
                    {"compound_name": "Bad", "author_name": "A", "smiles": "INVALID!!!"},
                ],
            },
            headers={"X-Session-ID": _sid()},
        )
        assert resp.status_code == 422


# ============================================================================
# Batch Operations
# ============================================================================


class TestBatchOperations:
    """Tests for batch summary and cancellation."""

    def test_get_batch_summary(self, client):
        """GET /api/v1/jobs/batch/{batch_id} returns BatchSummary."""
        sid = _sid()
        batch_resp = client.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": [
                    {"compound_name": "Sum1", "author_name": "A", "smiles": "CCO"},
                    {"compound_name": "Sum2", "author_name": "A", "smiles": "CCCO"},
                ],
            },
            headers={"X-Session-ID": sid},
        )
        batch_id = batch_resp.json()["batch_id"]

        resp = client.get(
            f"/api/v1/jobs/batch/{batch_id}", headers={"X-Session-ID": sid}
        )
        assert resp.status_code == 200

        summary = BatchSummary(**resp.json())
        assert summary.batch_id == batch_id
        assert summary.total_jobs == 2
        assert summary.pending >= 0

    def test_cancel_batch(self, client):
        """POST /api/v1/jobs/batch/{batch_id}/cancel cancels pending jobs."""
        sid = _sid()
        batch_resp = client.post(
            "/api/v1/jobs/batch",
            json={
                "compounds": [
                    {"compound_name": "Can1", "author_name": "A", "smiles": "CCO"},
                    {"compound_name": "Can2", "author_name": "A", "smiles": "CCCO"},
                ],
            },
            headers={"X-Session-ID": sid},
        )
        batch_id = batch_resp.json()["batch_id"]

        resp = client.post(
            f"/api/v1/jobs/batch/{batch_id}/cancel", headers={"X-Session-ID": sid}
        )
        assert resp.status_code == 200

        result = CancelResponse(**resp.json())
        assert result.batch_id == batch_id
        assert result.cancelled_count == 2

    def test_get_nonexistent_batch_returns_404(self, client):
        """GET /api/v1/jobs/batch/fake returns 404."""
        resp = client.get(
            "/api/v1/jobs/batch/nonexistent-batch", headers={"X-Session-ID": _sid()}
        )
        assert resp.status_code == 404

    def test_cancel_nonexistent_batch_returns_404(self, client):
        """POST cancel on nonexistent batch returns 404."""
        resp = client.post(
            "/api/v1/jobs/batch/fake-batch/cancel", headers={"X-Session-ID": _sid()}
        )
        assert resp.status_code == 404


# ============================================================================
# Check Availability (mocked external APIs)
# ============================================================================


class TestCheckAvailability:
    """Tests for availability check endpoints with mocked ChEMBL."""

    @patch("backend.modules.api_client.probe_all_thresholds")
    def test_check_availability_single(self, mock_probe, client):
        """POST /api/v1/jobs/check-availability returns availability result."""
        mock_probe.return_value = [
            {"threshold": 90, "count": 5},
            {"threshold": 80, "count": 12},
            {"threshold": 70, "count": 30},
        ]

        resp = client.post(
            "/api/v1/jobs/check-availability",
            json={"smiles": "CCO", "similarity_threshold": 90},
            headers={"X-Session-ID": _sid()},
        )
        assert resp.status_code == 200

        result = CheckAvailabilityResponse(**resp.json())
        assert result.result.available is True
        assert result.result.count_at_threshold == 5

    @patch("backend.modules.api_client.probe_all_thresholds")
    def test_check_availability_no_data(self, mock_probe, client):
        """Availability check with no data returns available=False."""
        mock_probe.return_value = [
            {"threshold": 90, "count": 0},
            {"threshold": 80, "count": 0},
        ]

        resp = client.post(
            "/api/v1/jobs/check-availability",
            json={"smiles": "CCO", "similarity_threshold": 90},
            headers={"X-Session-ID": _sid()},
        )
        assert resp.status_code == 200

        result = CheckAvailabilityResponse(**resp.json())
        assert result.result.available is False
        assert result.result.count_at_threshold == 0

    @patch("backend.modules.api_client.probe_all_thresholds")
    def test_check_availability_batch(self, mock_probe, client):
        """POST /api/v1/jobs/check-availability/batch returns batch results."""
        mock_probe.return_value = [
            {"threshold": 90, "count": 5},
        ]

        resp = client.post(
            "/api/v1/jobs/check-availability/batch",
            json={
                "compounds": [
                    {"compound_name": "C1", "smiles": "CCO"},
                    {"compound_name": "C2", "smiles": "CCCO"},
                ],
                "similarity_threshold": 90,
            },
            headers={"X-Session-ID": _sid()},
        )
        assert resp.status_code == 200

        result = CheckAvailabilityBatchResponse(**resp.json())
        assert len(result.results) == 2
        assert result.available_count >= 0
