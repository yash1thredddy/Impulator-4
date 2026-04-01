"""
Unit tests for RCSB PDB Client Module (async rewrite).

Tests the PDB integration for structural evidence scoring including:
- Similar ligand search (async)
- Structure details retrieval (async)
- Resolution fetching -- REST and GraphQL (async)
- GraphQL-to-REST fallback (D-35)
- Quality classification (sync)
- PDB evidence scoring (async)
- Circuit breaker behavior (D-31/D-33)
- Error handling and edge cases
- No threading artifacts

asyncio_mode = auto in pytest.ini -- no @pytest.mark.asyncio needed.
"""

import inspect
import time

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytest.importorskip("rdkit")

from backend.modules.pdb_client import (
    _circuits,
    _get_circuit,
    _is_circuit_open,
    _record_failure,
    _record_success,
    classify_resolution_quality,
    create_pdb_client,
    get_batch_structure_resolutions,
    get_batch_structure_resolutions_graphql,
    get_detailed_pdb_structures,
    get_pdb_evidence_score,
    get_pdb_summary_for_compound,
    get_structure_details,
    search_similar_ligands,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code=200, json_data=None, headers=None):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    resp.headers = headers or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestPdbCircuitBreaker:
    """Tests for PDB circuit breaker helpers (D-31/D-33)."""

    def test_circuits_dict_importable(self):
        assert isinstance(_circuits, dict)

    def test_get_circuit_creates_new(self):
        circuit = _get_circuit("pdb_test_cb")
        assert "failures" in circuit
        assert "threshold" in circuit
        _circuits.pop("pdb_test_cb", None)

    def test_circuit_opens_after_failures(self):
        circuit = _get_circuit("pdb_test_open")
        for _ in range(circuit["threshold"]):
            _record_failure(circuit)
        assert _is_circuit_open(circuit)
        _circuits.pop("pdb_test_open", None)

    def test_circuit_half_open_after_cooldown(self):
        circuit = _get_circuit("pdb_test_half")
        circuit["failures"] = circuit["threshold"]
        circuit["open_until"] = time.monotonic() - 1  # Expired
        assert not _is_circuit_open(circuit)
        _circuits.pop("pdb_test_half", None)

    def test_record_success_resets(self):
        circuit = _get_circuit("pdb_test_success")
        circuit["failures"] = 5
        _record_success(circuit)
        assert circuit["failures"] == 0
        _circuits.pop("pdb_test_success", None)


# ---------------------------------------------------------------------------
# search_similar_ligands
# ---------------------------------------------------------------------------


class TestSearchSimilarLigands:
    """Tests for search_similar_ligands (async)."""

    async def test_search_returns_pdb_ids(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = _mock_response(200, {
            "result_set": [
                {"identifier": "4HHB"},
                {"identifier": "3WHM"},
                {"identifier": "2CPK"},
            ]
        })
        result = await search_similar_ligands(client, "CCO")
        assert result == ["4HHB", "3WHM", "2CPK"]

    async def test_search_no_results_204(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = _mock_response(204)
        result = await search_similar_ligands(client, "CCCCCCCCCC")
        assert result == []

    async def test_search_empty_result_set(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = _mock_response(200, {"result_set": []})
        result = await search_similar_ligands(client, "CCO")
        assert result == []

    async def test_search_limits_to_100(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_results = [{"identifier": f"PDB{i:04d}"} for i in range(150)]
        client.post.return_value = _mock_response(200, {"result_set": mock_results})
        result = await search_similar_ligands(client, "CCO")
        assert len(result) == 100

    async def test_search_timeout_returns_empty(self):
        """PDB is non-critical -- failure returns empty list (D-34)."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.side_effect = httpx.TimeoutException("timeout")
        result = await search_similar_ligands(client, "CCO")
        assert result == []

    async def test_search_400_returns_empty(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = _mock_response(400)
        result = await search_similar_ligands(client, "invalid_smiles")
        assert result == []

    async def test_search_circuit_open_returns_empty(self):
        circuit = _get_circuit("pdb_search")
        old = circuit["failures"]
        circuit["failures"] = 10
        circuit["open_until"] = time.monotonic() + 300
        try:
            client = AsyncMock(spec=httpx.AsyncClient)
            result = await search_similar_ligands(client, "CCO")
            assert result == []
        finally:
            circuit["failures"] = old
            circuit["open_until"] = 0.0


# ---------------------------------------------------------------------------
# get_structure_details
# ---------------------------------------------------------------------------


class TestGetStructureDetails:
    """Tests for get_structure_details (async)."""

    async def test_details_success(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        entry_resp = _mock_response(200, {
            "struct": {"title": "Test Structure"},
            "rcsb_entry_info": {"resolution_combined": [1.74]},
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "rcsb_primary_citation": {"pdbx_database_id_DOI": "10.1234/test"},
        })
        entity_resp = _mock_response(200, {
            "rcsb_polymer_entity_container_identifiers": {
                "reference_sequence_identifiers": [
                    {"database_name": "UniProt", "database_accession": "P12345"}
                ]
            }
        })
        client.get.side_effect = [entry_resp, entity_resp]
        result = await get_structure_details(client, "4HHB")
        assert result["pdb_id"] == "4HHB"
        assert result["title"] == "Test Structure"
        assert result["resolution"] == 1.74
        assert "P12345" in result["uniprot_ids"]

    async def test_details_minimal_data(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = _mock_response(200, {})
        result = await get_structure_details(client, "XXXX")
        assert result["pdb_id"] == "XXXX"
        assert result["title"] is None
        assert result["resolution"] is None

    async def test_details_failure_returns_defaults(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ConnectError("connection refused")
        result = await get_structure_details(client, "4HHB")
        assert result["pdb_id"] == "4HHB"
        assert "url" in result


# ---------------------------------------------------------------------------
# Batch structure resolutions (GraphQL + REST)
# ---------------------------------------------------------------------------


class TestGetBatchStructureResolutionsGraphQL:
    """Tests for get_batch_structure_resolutions_graphql."""

    async def test_graphql_returns_resolutions(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = _mock_response(200, {
            "data": {
                "entries": [
                    {"rcsb_id": "4HHB", "rcsb_entry_info": {"resolution_combined": [1.74]}},
                    {"rcsb_id": "3WHM", "rcsb_entry_info": {"resolution_combined": [2.10]}},
                    {"rcsb_id": "2CPK", "rcsb_entry_info": {"resolution_combined": None}},
                ]
            }
        })
        result = await get_batch_structure_resolutions_graphql(client, ["4HHB", "3WHM", "2CPK"])
        assert result["4HHB"] == 1.74
        assert result["3WHM"] == 2.10
        assert result["2CPK"] is None

    async def test_graphql_empty_input(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        result = await get_batch_structure_resolutions_graphql(client, [])
        assert result == {}

    async def test_graphql_normalizes_case(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = _mock_response(200, {
            "data": {
                "entries": [
                    {"rcsb_id": "4HHB", "rcsb_entry_info": {"resolution_combined": [1.74]}}
                ]
            }
        })
        result = await get_batch_structure_resolutions_graphql(client, ["4hhb"])
        assert "4HHB" in result

    async def test_graphql_error_returns_empty(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.side_effect = httpx.TimeoutException("timeout")
        result = await get_batch_structure_resolutions_graphql(client, ["4HHB"])
        assert result == {}

    async def test_graphql_circuit_open(self):
        circuit = _get_circuit("pdb_graphql")
        old = circuit["failures"]
        circuit["failures"] = 10
        circuit["open_until"] = time.monotonic() + 300
        try:
            client = AsyncMock(spec=httpx.AsyncClient)
            result = await get_batch_structure_resolutions_graphql(client, ["4HHB"])
            assert result == {}
        finally:
            circuit["failures"] = old
            circuit["open_until"] = 0.0


class TestGetBatchStructureResolutions:
    """Tests for get_batch_structure_resolutions (GraphQL primary, REST fallback)."""

    async def test_graphql_primary_success(self):
        """GraphQL is primary path (D-35)."""
        client = AsyncMock(spec=httpx.AsyncClient)
        graphql_resp = _mock_response(200, {
            "data": {"entries": [
                {"rcsb_id": "4HHB", "rcsb_entry_info": {"resolution_combined": [1.74]}},
            ]}
        })
        client.post.return_value = graphql_resp
        result = await get_batch_structure_resolutions(client, ["4HHB"])
        assert result["4HHB"] == 1.74

    async def test_graphql_failure_falls_back_to_rest(self):
        """When GraphQL fails, falls back to parallel REST (D-35)."""
        client = AsyncMock(spec=httpx.AsyncClient)
        # GraphQL POST returns empty (failure)
        client.post.return_value = _mock_response(200, {"data": {"entries": []}})
        # REST GET succeeds
        client.get.return_value = _mock_response(200, {
            "rcsb_entry_info": {"resolution_combined": [2.1]}
        })
        result = await get_batch_structure_resolutions(client, ["4HHB"])
        assert "4HHB" in result

    async def test_empty_input(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        result = await get_batch_structure_resolutions(client, [])
        assert result == {}


# ---------------------------------------------------------------------------
# classify_resolution_quality (sync pure function)
# ---------------------------------------------------------------------------


class TestClassifyResolutionQuality:
    """Tests for classify_resolution_quality."""

    def test_high_quality(self):
        quality, multiplier = classify_resolution_quality(1.5)
        assert quality == "***"
        assert multiplier == 1.0

    def test_medium_quality(self):
        quality, multiplier = classify_resolution_quality(2.5)
        assert quality == "**"
        assert multiplier == 0.75

    def test_poor_quality(self):
        quality, multiplier = classify_resolution_quality(3.5)
        assert quality == "*"
        assert multiplier == 0.5

    def test_boundary_2_angstrom(self):
        quality, multiplier = classify_resolution_quality(2.0)
        assert quality == "**"

    def test_boundary_3_angstrom(self):
        quality, multiplier = classify_resolution_quality(3.0)
        assert quality == "**"

    def test_very_high_resolution(self):
        quality, multiplier = classify_resolution_quality(0.5)
        assert quality == "***"

    def test_very_low_resolution(self):
        quality, multiplier = classify_resolution_quality(10.0)
        assert quality == "*"


# ---------------------------------------------------------------------------
# PDB evidence score
# ---------------------------------------------------------------------------


class TestGetPDBEvidenceScore:
    """Tests for get_pdb_evidence_score."""

    async def test_score_with_high_quality_structures(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        # Mock search
        with patch(
            "backend.modules.pdb_client.search_similar_ligands",
            return_value=["4HHB", "3WHM", "2CPK", "1ABC", "5XYZ"],
        ):
            with patch(
                "backend.modules.pdb_client.get_batch_structure_resolutions",
                return_value={
                    "4HHB": 1.5, "3WHM": 1.8, "2CPK": 1.9,
                    "1ABC": 1.7, "5XYZ": 1.6,
                },
            ):
                result = await get_pdb_evidence_score(client, "CCO")

        assert result["pdb_score"] == 1.0
        assert result["num_structures"] == 5
        assert result["num_high_quality"] == 5

    async def test_score_with_mixed_quality(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with patch(
            "backend.modules.pdb_client.search_similar_ligands",
            return_value=["4HHB", "3WHM", "2CPK"],
        ):
            with patch(
                "backend.modules.pdb_client.get_batch_structure_resolutions",
                return_value={"4HHB": 1.5, "3WHM": 2.5, "2CPK": 3.5},
            ):
                result = await get_pdb_evidence_score(client, "CCO")

        assert result["num_high_quality"] == 1
        assert result["num_medium_quality"] == 1
        assert result["num_poor_quality"] == 1
        assert 0 < result["pdb_score"] < 1.0

    async def test_score_no_structures(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with patch(
            "backend.modules.pdb_client.search_similar_ligands",
            return_value=[],
        ):
            result = await get_pdb_evidence_score(client, "CCCCCCCCCC")

        assert result["pdb_score"] == 0.0
        assert result["num_structures"] == 0

    async def test_score_structures_without_resolution(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with patch(
            "backend.modules.pdb_client.search_similar_ligands",
            return_value=["4HHB", "3WHM"],
        ):
            with patch(
                "backend.modules.pdb_client.get_batch_structure_resolutions",
                return_value={"4HHB": None, "3WHM": None},
            ):
                result = await get_pdb_evidence_score(client, "CCO")

        assert result["pdb_score"] == 0.0
        assert result["num_structures"] == 2


# ---------------------------------------------------------------------------
# PDB summary
# ---------------------------------------------------------------------------


class TestGetPDBSummary:
    """Tests for get_pdb_summary_for_compound."""

    async def test_summary_with_structures(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with patch("backend.modules.pdb_client.get_pdb_evidence_score", return_value={
            "pdb_score": 0.8,
            "num_structures": 3,
            "num_high_quality": 2,
            "num_medium_quality": 1,
            "num_poor_quality": 0,
            "pdb_ids": ["4HHB", "3WHM", "2CPK"],
            "resolutions": [1.74, 1.9, 2.5],
            "quality_classes": ["***", "***", "**"],
        }):
            summary = await get_pdb_summary_for_compound(client, "CCO")

        assert "Found 3 similar structure(s)" in summary
        assert "PDB Evidence Score: 0.800" in summary
        assert "2 high-quality" in summary

    async def test_summary_no_structures(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with patch("backend.modules.pdb_client.get_pdb_evidence_score", return_value={
            "pdb_score": 0.0,
            "num_structures": 0,
            "num_high_quality": 0,
            "num_medium_quality": 0,
            "num_poor_quality": 0,
            "pdb_ids": [],
            "resolutions": [],
            "quality_classes": [],
        }):
            summary = await get_pdb_summary_for_compound(client, "CCCCCCCCCC")

        assert "No experimental structures found" in summary


# ---------------------------------------------------------------------------
# Detailed structures
# ---------------------------------------------------------------------------


class TestGetDetailedPDBStructures:
    """Tests for get_detailed_pdb_structures."""

    async def test_detailed_structures_sorted(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with patch("backend.modules.pdb_client.get_pdb_evidence_score", return_value={
            "pdb_ids": ["4HHB", "3WHM", "2CPK"],
            "resolutions": [3.5, 1.5, 2.5],
            "quality_classes": ["*", "***", "**"],
        }):
            with patch("backend.modules.pdb_client.get_structure_details") as mock_details:
                async def fake_details(client_, pdb_id):
                    return {
                        "pdb_id": pdb_id,
                        "title": f"Structure {pdb_id}",
                        "uniprot_ids": [],
                        "experimental_method": "X-RAY",
                        "url": f"https://www.rcsb.org/structure/{pdb_id}",
                    }
                mock_details.side_effect = fake_details
                result = await get_detailed_pdb_structures(client, "CCO")

        assert result[0]["Quality"] == "***"
        assert result[1]["Quality"] == "**"
        assert result[2]["Quality"] == "*"

    async def test_detailed_structures_empty(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        with patch("backend.modules.pdb_client.get_pdb_evidence_score", return_value={
            "pdb_ids": [],
            "resolutions": [],
            "quality_classes": [],
        }):
            result = await get_detailed_pdb_structures(client, "CCCCCCCCCC")

        assert result == []


# ---------------------------------------------------------------------------
# Structural verification
# ---------------------------------------------------------------------------


class TestNoThreadingArtifacts:
    """Verify removed sync infrastructure (D-36/D-37/D-39/D-42)."""

    def test_no_rate_limiter(self):
        import backend.modules.pdb_client as m
        source = inspect.getsource(m)
        assert "class RateLimiter" not in source

    def test_no_threading_local(self):
        import backend.modules.pdb_client as m
        source = inspect.getsource(m)
        assert "threading.local()" not in source

    def test_no_threadpool_executor(self):
        import backend.modules.pdb_client as m
        source = inspect.getsource(m)
        assert "ThreadPoolExecutor" not in source

    def test_no_use_official_api(self):
        import backend.modules.pdb_client as m
        source = inspect.getsource(m)
        assert "USE_OFFICIAL_API" not in source

    def test_no_data_query(self):
        import backend.modules.pdb_client as m
        source = inspect.getsource(m)
        assert "DataQuery" not in source


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestCreatePdbClient:
    """Tests for create_pdb_client factory."""

    def test_returns_async_client(self):
        client = create_pdb_client()
        assert isinstance(client, httpx.AsyncClient)

