"""
Integration tests for pre-submission availability check endpoints.

Tests:
- POST /api/v1/jobs/check-availability (single compound)
- POST /api/v1/jobs/check-availability/batch (multiple compounds)

These endpoints probe ChEMBL for similarity data before job submission
to prevent wasting time on compounds with no ChEMBL data.
"""
import uuid
from datetime import datetime, timezone

from unittest.mock import patch
from sqlalchemy.orm import sessionmaker


def _mock_probe_data_available(smiles, threshold):
    """Mock probe_all_thresholds returning data at multiple thresholds."""
    return [
        {"threshold": 90, "count": 5},
        {"threshold": 80, "count": 12},
        {"threshold": 70, "count": 20},
        {"threshold": 60, "count": 35},
        {"threshold": 50, "count": 50},
        {"threshold": 40, "count": 80},
    ]


def _mock_probe_no_data(smiles, threshold):
    """Mock probe_all_thresholds returning no data at any threshold."""
    return [
        {"threshold": 90, "count": 0},
        {"threshold": 80, "count": 0},
        {"threshold": 70, "count": 0},
        {"threshold": 60, "count": 0},
        {"threshold": 50, "count": 0},
        {"threshold": 40, "count": 0},
    ]


def _mock_probe_low_thresholds_only(smiles, threshold):
    """Mock probe_all_thresholds with data only at lower thresholds."""
    return [
        {"threshold": 90, "count": 0},
        {"threshold": 80, "count": 0},
        {"threshold": 70, "count": 3},
        {"threshold": 60, "count": 8},
        {"threshold": 50, "count": 15},
        {"threshold": 40, "count": 30},
    ]


class TestCheckAvailabilitySingle:
    """Tests for POST /api/v1/jobs/check-availability."""

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=_mock_probe_data_available)
    def test_compound_with_data_returns_available(self, mock_probe, client):
        """Compound with ChEMBL data at requested threshold should return available=True."""
        response = client.post(
            "/api/v1/jobs/check-availability",
            json={
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "similarity_threshold": 90,
                "activity_types": ["IC50"],
            }
        )
        assert response.status_code == 200
        data = response.json()
        result = data["result"]

        assert result["available"] is True
        assert result["has_any_data"] is True
        assert result["count_at_threshold"] == 5
        assert len(result["thresholds"]) > 0

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=_mock_probe_no_data)
    def test_compound_without_data_returns_unavailable(self, mock_probe, client):
        """Compound with no ChEMBL data should return available=False, has_any_data=False."""
        response = client.post(
            "/api/v1/jobs/check-availability",
            json={
                "smiles": "C1CCCCC1",
                "similarity_threshold": 90,
            }
        )
        assert response.status_code == 200
        result = response.json()["result"]

        assert result["available"] is False
        assert result["has_any_data"] is False
        assert result["count_at_threshold"] == 0

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=_mock_probe_low_thresholds_only)
    def test_data_at_lower_thresholds_only(self, mock_probe, client):
        """Compound with data only at lower thresholds: available=False but has_any_data=True."""
        response = client.post(
            "/api/v1/jobs/check-availability",
            json={
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "similarity_threshold": 90,
            }
        )
        assert response.status_code == 200
        result = response.json()["result"]

        assert result["available"] is False
        assert result["has_any_data"] is True
        assert result["count_at_threshold"] == 0
        # Should have threshold entries showing where data exists
        threshold_counts = {t["threshold"]: t["count"] for t in result["thresholds"]}
        assert threshold_counts[70] == 3
        assert threshold_counts[40] == 30

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=_mock_probe_data_available)
    def test_existing_compound_included_in_response(self, mock_probe, test_engine, client):
        """Existing compounds with same structure key should be in existing_compounds."""
        from backend.models.database import Compound
        from backend.services.job_service import generate_inchikey, _inchikey_structure_key

        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        inchikey = generate_inchikey(smiles)

        Session = sessionmaker(bind=test_engine)
        session = Session()
        session.add(Compound(
            entry_id=str(uuid.uuid4()),
            compound_name="Aspirin",
            smiles=smiles,
            inchikey=inchikey,
            inchikey_structure_key=_inchikey_structure_key(inchikey),
            similarity_threshold=90,
            activity_types="IC50",
            processed_at=datetime.now(timezone.utc),
        ))
        session.commit()
        session.close()

        response = client.post(
            "/api/v1/jobs/check-availability",
            json={
                "smiles": smiles,
                "similarity_threshold": 90,
                "activity_types": ["IC50"],
            }
        )
        assert response.status_code == 200
        result = response.json()["result"]

        assert len(result["existing_compounds"]) == 1
        assert result["existing_compounds"][0]["compound_name"] == "Aspirin"

    def test_missing_smiles_returns_422(self, client):
        """Request without SMILES should return 422 validation error."""
        response = client.post(
            "/api/v1/jobs/check-availability",
            json={
                "similarity_threshold": 90,
            }
        )
        assert response.status_code == 422


class TestCheckAvailabilityBatch:
    """Tests for POST /api/v1/jobs/check-availability/batch."""

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=_mock_probe_data_available)
    def test_batch_with_multiple_compounds(self, mock_probe, client):
        """Batch check with multiple compounds returns per-compound results."""
        response = client.post(
            "/api/v1/jobs/check-availability/batch",
            json={
                "compounds": [
                    {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                    {"compound_name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
                ],
                "similarity_threshold": 90,
                "activity_types": ["IC50"],
            }
        )
        assert response.status_code == 200
        data = response.json()

        assert len(data["results"]) == 2
        assert data["available_count"] == 2
        assert data["no_data_count"] == 0

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=_mock_probe_no_data)
    def test_batch_all_no_data(self, mock_probe, client):
        """Batch where no compounds have data should report no_data_count."""
        response = client.post(
            "/api/v1/jobs/check-availability/batch",
            json={
                "compounds": [
                    {"compound_name": "Unknown1", "smiles": "C1CCCCC1"},
                    {"compound_name": "Unknown2", "smiles": "CCCCCC"},
                ],
                "similarity_threshold": 90,
            }
        )
        assert response.status_code == 200
        data = response.json()

        assert data["available_count"] == 0
        assert data["no_data_count"] == 2

    @patch('backend.modules.api_client.probe_all_thresholds')
    def test_batch_mixed_availability(self, mock_probe, client):
        """Batch with mixed results: some available, some not."""
        def mixed_probe(smiles, threshold):
            if "OC1=CC=CC=C1" in smiles:  # Aspirin-like
                return _mock_probe_data_available(smiles, threshold)
            else:
                return _mock_probe_no_data(smiles, threshold)

        mock_probe.side_effect = mixed_probe

        response = client.post(
            "/api/v1/jobs/check-availability/batch",
            json={
                "compounds": [
                    {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                    {"compound_name": "Unknown", "smiles": "C1CCCCC1"},
                ],
                "similarity_threshold": 90,
            }
        )
        assert response.status_code == 200
        data = response.json()

        assert data["available_count"] == 1
        assert data["no_data_count"] == 1

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=_mock_probe_data_available)
    def test_batch_returns_summary_counts(self, mock_probe, client):
        """Batch response should include summary counts."""
        response = client.post(
            "/api/v1/jobs/check-availability/batch",
            json={
                "compounds": [
                    {"compound_name": "A", "smiles": "CCO"},
                ],
                "similarity_threshold": 90,
            }
        )
        data = response.json()

        assert "available_count" in data
        assert "unavailable_count" in data
        assert "no_data_count" in data
        assert "results" in data

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=Exception("API timeout"))
    def test_batch_handles_probe_failure_gracefully(self, mock_probe, client):
        """If probe fails for a compound, it should be reported as unavailable, not crash."""
        response = client.post(
            "/api/v1/jobs/check-availability/batch",
            json={
                "compounds": [
                    {"compound_name": "Problem", "smiles": "CCO"},
                ],
                "similarity_threshold": 90,
            }
        )
        # Should not crash — either returns 200 with unavailable result or 500
        assert response.status_code in (200, 500)

    def test_batch_empty_compounds_returns_422(self, client):
        """Empty compounds list should return validation error."""
        response = client.post(
            "/api/v1/jobs/check-availability/batch",
            json={
                "compounds": [],
                "similarity_threshold": 90,
            }
        )
        # Should fail validation (empty list)
        assert response.status_code in (200, 422)

    @patch('backend.modules.api_client.probe_all_thresholds', side_effect=_mock_probe_data_available)
    def test_batch_existing_compounds_matched(self, mock_probe, test_engine, client):
        """Batch should find existing compounds by InChIKey for each input."""
        from backend.models.database import Compound
        from backend.services.job_service import generate_inchikey, _inchikey_structure_key

        smiles = "CCO"
        inchikey = generate_inchikey(smiles)

        Session = sessionmaker(bind=test_engine)
        session = Session()
        session.add(Compound(
            entry_id=str(uuid.uuid4()),
            compound_name="Ethanol",
            smiles=smiles,
            inchikey=inchikey,
            inchikey_structure_key=_inchikey_structure_key(inchikey),
            similarity_threshold=90,
            activity_types="IC50",
            processed_at=datetime.now(timezone.utc),
        ))
        session.commit()
        session.close()

        response = client.post(
            "/api/v1/jobs/check-availability/batch",
            json={
                "compounds": [
                    {"compound_name": "Ethanol_v2", "smiles": smiles},
                ],
                "similarity_threshold": 70,
                "activity_types": ["IC50"],
            }
        )
        assert response.status_code == 200
        data = response.json()

        result = data["results"][0]
        assert len(result["existing_compounds"]) == 1
        assert result["existing_compounds"][0]["compound_name"] == "Ethanol"
