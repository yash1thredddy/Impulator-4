"""Regression tests for batch availability service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeChemblClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_batch_availability_uses_chembl_client_for_probe_and_bioactivity():
    """Batch availability should call probe helpers with the created client."""
    from backend.models.schemas import CheckAvailabilityBatchRequest
    from backend.services.job_service import JobService

    request = CheckAvailabilityBatchRequest(
        compounds=[{"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}],
        similarity_threshold=90,
        activity_types=["IC50"],
    )

    fake_client = _FakeChemblClient()
    mock_probe = AsyncMock(return_value=[{"threshold": 90, "count": 3}, {"threshold": 80, "count": 5}])
    mock_bio = AsyncMock(return_value=True)

    with patch("backend.modules.api_client.create_chembl_client", return_value=fake_client), \
         patch("backend.modules.api_client.probe_all_thresholds", mock_probe), \
         patch("backend.modules.api_client.quick_has_bioactivity", mock_bio), \
         patch("backend.services.job_service.generate_inchikey", return_value=None):
        service = JobService()
        result = await service.check_availability_batch_service(MagicMock(), request)

    assert result.available_count == 1
    assert result.no_data_count == 0
    assert result.results[0].available is True
    mock_probe.assert_awaited_once_with(fake_client, "CC(=O)OC1=CC=CC=C1C(=O)O", 90)
    mock_bio.assert_awaited_once_with(fake_client, "CC(=O)OC1=CC=CC=C1C(=O)O", 90, ["IC50"])
