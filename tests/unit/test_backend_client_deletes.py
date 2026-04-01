"""Regression tests for frontend delete APIs."""

from unittest.mock import patch

import requests

from frontend.services.backend_client import ImpulatorAPIClient


def test_delete_compound_fails_closed_when_backend_unavailable():
    """Delete must not fall back to direct storage writes."""
    client = ImpulatorAPIClient()

    with patch.object(
        client,
        "_request",
        side_effect=requests.ConnectionError("backend down"),
    ):
        result = client.delete_compound("123e4567-e89b-12d3-a456-426614174000")

    assert result.success is False
    assert result.data is None
    assert result.error == "Backend unavailable. Compound was not deleted."


def test_delete_compounds_batch_fails_closed_when_backend_unavailable():
    """Batch delete must not fall back to direct storage writes."""
    client = ImpulatorAPIClient()

    with patch.object(
        client,
        "_request",
        side_effect=requests.Timeout("backend down"),
    ):
        result = client.delete_compounds_batch(
            [
                "123e4567-e89b-12d3-a456-426614174000",
                "123e4567-e89b-12d3-a456-426614174001",
            ]
        )

    assert result.success is False
    assert result.data is None
    assert result.error == "Backend unavailable. Compounds were not deleted."
