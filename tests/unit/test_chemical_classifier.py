"""
Unit tests for Chemical Classification Module (async rewrite).

Tests ClassyFire and NPClassifier integration including:
- ClassyFire API classification (async)
- NPClassifier API classification (async)
- Complete classification combination (parallel asyncio.gather)
- Circuit breaker behavior (D-31/D-33/D-34)
- Compound type inference (sync)
- Error handling and graceful degradation
- No threading artifacts

asyncio_mode = auto in pytest.ini -- no @pytest.mark.asyncio needed.
"""

import inspect
import time

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.modules.chemical_classifier import (
    _circuits,
    _get_circuit,
    _is_circuit_open,
    _record_failure,
    _record_success,
    classify_compound_type,
    create_classifier_client,
    extract_classyfire_fields,
    get_classyfire_classification,
    get_classification_summary,
    get_complete_classification,
    get_npclassifier_classification,
    shutdown_classifier,
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
# Circuit breaker importability
# ---------------------------------------------------------------------------


class TestCircuitBreakerImportability:
    """Verify circuit breaker state is accessible for test manipulation (W4)."""

    def test_circuits_dict_importable(self):
        assert isinstance(_circuits, dict)

    def test_get_circuit_importable(self):
        circuit = _get_circuit("test_classifier_cb")
        assert "failures" in circuit
        assert "open_until" in circuit
        assert "threshold" in circuit
        _circuits.pop("test_classifier_cb", None)

    def test_circuit_breaker_helpers_work(self):
        circuit = _get_circuit("test_classifier_helpers")
        assert not _is_circuit_open(circuit)
        _record_failure(circuit)
        assert circuit["failures"] == 1
        _record_success(circuit)
        assert circuit["failures"] == 0
        _circuits.pop("test_classifier_helpers", None)

    def test_classyfire_circuit_pre_created(self):
        """ClassyFire circuit is pre-created with threshold=5 (D-34)."""
        circuit = _get_circuit("classyfire")
        assert circuit["threshold"] == 5

    def test_npclassifier_circuit_pre_created(self):
        """NPClassifier circuit is pre-created with threshold=5 (D-34)."""
        circuit = _get_circuit("npclassifier")
        assert circuit["threshold"] == 5


# ---------------------------------------------------------------------------
# get_classyfire_classification
# ---------------------------------------------------------------------------


class TestGetClassyfireClassification:
    """Tests for get_classyfire_classification (async)."""

    async def test_success(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = _mock_response(200, {
            "kingdom": {"name": "Organic compounds"},
            "superclass": {"name": "Phenylpropanoids and polyketides"},
            "class": {"name": "Flavonoids", "chemont_id": "CHEMONTID:0000001"},
            "subclass": {"name": "Flavones", "chemont_id": "CHEMONTID:0000002"},
            "direct_parent": {"name": "Hydroxyflavones"},
            "molecular_framework": "Aromatic homomonocyclic compounds",
            "description": "A flavonoid compound",
        })
        result = await get_classyfire_classification(client, "REFJWTPEDVJJIY-UHFFFAOYSA-N")
        assert result is not None
        assert result["kingdom"]["name"] == "Organic compounds"
        assert result["class"]["name"] == "Flavonoids"

    async def test_404_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        resp = _mock_response(404)
        resp.raise_for_status = MagicMock()  # 4xx doesn't trip retry
        client.get.return_value = resp
        result = await get_classyfire_classification(client, "UNKNOWN-INCHIKEY")
        assert result is None

    async def test_timeout_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ReadTimeout("timeout")
        result = await get_classyfire_classification(client, "REFJWTPEDVJJIY-UHFFFAOYSA-N")
        assert result is None

    async def test_circuit_open_returns_none(self):
        circuit = _get_circuit("classyfire")
        old_failures = circuit["failures"]
        old_open = circuit["open_until"]
        circuit["failures"] = 10
        circuit["open_until"] = time.monotonic() + 300
        try:
            client = AsyncMock(spec=httpx.AsyncClient)
            result = await get_classyfire_classification(client, "REFJWTPEDVJJIY-UHFFFAOYSA-N")
            assert result is None
        finally:
            circuit["failures"] = old_failures
            circuit["open_until"] = old_open


# ---------------------------------------------------------------------------
# get_npclassifier_classification
# ---------------------------------------------------------------------------


class TestGetNpclassifierClassification:
    """Tests for get_npclassifier_classification (async)."""

    async def test_success(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = _mock_response(200, {
            "pathway_results": ["Shikimates and Phenylpropanoids"],
            "superclass_results": ["Flavonoids"],
            "class_results": ["Flavones"],
            "isglycoside": False,
        })
        result = await get_npclassifier_classification(client, "c1ccc(cc1)O")
        assert result is not None
        assert result["NP_Pathway"] == "Shikimates and Phenylpropanoids"
        assert result["NP_Superclass"] == "Flavonoids"
        assert result["NP_Class"] == "Flavones"
        assert not result["NP_isglycoside"]

    async def test_glycoside_detection(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = _mock_response(200, {
            "pathway_results": ["Carbohydrates"],
            "superclass_results": ["Glycosides"],
            "class_results": ["O-glycosides"],
            "isglycoside": True,
        })
        result = await get_npclassifier_classification(client, "glycoside_smiles")
        assert result is not None
        assert result["NP_isglycoside"]

    async def test_empty_results(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = _mock_response(200, {
            "pathway_results": [],
            "superclass_results": [],
            "class_results": [],
            "isglycoside": False,
        })
        result = await get_npclassifier_classification(client, "CCO")
        assert result is not None
        assert result["NP_Pathway"] is None
        assert result["NP_Superclass"] is None

    async def test_timeout_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ReadTimeout("timeout")
        result = await get_npclassifier_classification(client, "c1ccc(cc1)O")
        assert result is None

    async def test_circuit_open_returns_none(self):
        circuit = _get_circuit("npclassifier")
        old_failures = circuit["failures"]
        old_open = circuit["open_until"]
        circuit["failures"] = 10
        circuit["open_until"] = time.monotonic() + 300
        try:
            client = AsyncMock(spec=httpx.AsyncClient)
            result = await get_npclassifier_classification(client, "CCO")
            assert result is None
        finally:
            circuit["failures"] = old_failures
            circuit["open_until"] = old_open


# ---------------------------------------------------------------------------
# get_complete_classification
# ---------------------------------------------------------------------------


class TestGetCompleteClassification:
    """Tests for get_complete_classification (parallel asyncio.gather)."""

    async def test_both_succeed(self):
        """ClassyFire and NPClassifier called in parallel via asyncio.gather."""
        client = AsyncMock(spec=httpx.AsyncClient)
        cf_resp = _mock_response(200, {
            "kingdom": {"name": "Organic compounds"},
            "class": {"name": "Flavonoids", "chemont_id": "CHEMONTID:0001"},
        })
        np_resp = _mock_response(200, {
            "pathway_results": ["Shikimates"],
            "superclass_results": ["Flavonoids"],
            "class_results": ["Flavones"],
            "isglycoside": False,
        })
        client.get.side_effect = [cf_resp, np_resp]
        result = await get_complete_classification(
            client, "c1ccc(cc1)O", "REFJWTPEDVJJIY-UHFFFAOYSA-N",
        )
        assert result["Kingdom"] == "Organic compounds"
        assert result["Class"] == "Flavonoids"
        assert result["NP_Pathway"] == "Shikimates"
        assert result["classification_available"] is True

    async def test_classyfire_fails_gracefully(self):
        """If ClassyFire fails, NPClassifier result still returned (D-34)."""
        client = AsyncMock(spec=httpx.AsyncClient)
        np_resp = _mock_response(200, {
            "pathway_results": ["Shikimates"],
            "superclass_results": ["Flavonoids"],
            "class_results": ["Flavones"],
            "isglycoside": False,
        })
        # ClassyFire times out, NPClassifier succeeds
        client.get.side_effect = [httpx.ReadTimeout("timeout"), np_resp]
        result = await get_complete_classification(
            client, "CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        )
        assert result["NP_Pathway"] == "Shikimates"
        assert result["classification_available"] is True

    async def test_npclassifier_fails_gracefully(self):
        """If NPClassifier fails, ClassyFire result still returned (D-34)."""
        client = AsyncMock(spec=httpx.AsyncClient)
        cf_resp = _mock_response(200, {
            "kingdom": {"name": "Organic compounds"},
            "class": {"name": "Alkanes"},
        })
        # ClassyFire succeeds, NPClassifier times out
        client.get.side_effect = [cf_resp, httpx.ReadTimeout("timeout")]
        result = await get_complete_classification(
            client, "CCCC", "IJDNQMDRQITEOD-UHFFFAOYSA-N",
        )
        assert result["Kingdom"] == "Organic compounds"
        assert result["Class"] == "Alkanes"
        assert result["NP_Pathway"] == ""
        assert result["classification_available"] is True

    async def test_both_fail_gracefully(self):
        """If both fail, returns dict with empty strings (D-34)."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ReadTimeout("timeout")
        result = await get_complete_classification(
            client, "CCCC", "UNKNOWN",
        )
        assert result["Kingdom"] == ""
        assert result["Class"] == ""
        assert result["NP_Pathway"] == ""
        assert result["classification_available"] is False

    async def test_parallel_gather_both_endpoints_called(self):
        """Verify ClassyFire and NPClassifier are called concurrently."""
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = _mock_response(200, {})
        result = await get_complete_classification(
            client, "CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        )
        # Both endpoints should be called
        assert client.get.call_count >= 2


# ---------------------------------------------------------------------------
# extract_classyfire_fields (sync pure function)
# ---------------------------------------------------------------------------


class TestExtractClassyfireFields:
    """Tests for extract_classyfire_fields."""

    def test_full_extraction(self):
        cf_data = {
            "kingdom": {"name": "Organic compounds"},
            "superclass": {"name": "Phenylpropanoids"},
            "class": {"name": "Flavonoids", "chemont_id": "CHEMONTID:0000001"},
            "subclass": {"name": "Flavones", "chemont_id": "CHEMONTID:0000002"},
            "direct_parent": {"name": "Hydroxyflavones"},
            "molecular_framework": "Aromatic",
            "description": "A flavonoid",
        }
        result = extract_classyfire_fields(cf_data)
        assert result["Kingdom"] == "Organic compounds"
        assert result["Superclass"] == "Phenylpropanoids"
        assert result["Class"] == "Flavonoids"
        assert result["Subclass"] == "Flavones"
        assert result["Direct_Parent"] == "Hydroxyflavones"
        assert result["Molecular_Framework"] == "Aromatic"
        assert result["Description"] == "A flavonoid"
        assert result["ChEMONT_ID_Class"] == "CHEMONTID:0000001"
        assert result["ChEMONT_ID_Subclass"] == "CHEMONTID:0000002"

    def test_none_input(self):
        result = extract_classyfire_fields(None)
        assert result["Kingdom"] == ""
        assert result["Class"] == ""
        assert len(result) == 9

    def test_partial_data(self):
        cf_data = {
            "kingdom": {"name": "Organic compounds"},
            "class": {"name": "Flavonoids"},
        }
        result = extract_classyfire_fields(cf_data)
        assert result["Kingdom"] == "Organic compounds"
        assert result["Class"] == "Flavonoids"
        assert result["Superclass"] == ""


# ---------------------------------------------------------------------------
# classify_compound_type (sync pure function)
# ---------------------------------------------------------------------------


class TestClassifyCompoundType:
    """Tests for classify_compound_type."""

    def test_natural_product_by_np_pathway(self):
        classification = {"NP_Pathway": "Terpenoids", "Class": "Organic compounds"}
        assert classify_compound_type(classification) == "Natural Product"

    def test_natural_product_by_classyfire(self):
        classification = {"NP_Pathway": "", "Class": "Flavonoids"}
        assert classify_compound_type(classification) == "Natural Product"

    def test_natural_product_alkaloid(self):
        classification = {"NP_Pathway": "", "Superclass": "Alkaloids and derivatives"}
        assert classify_compound_type(classification) == "Natural Product"

    def test_synthetic_compound(self):
        classification = {
            "NP_Pathway": "",
            "Class": "Organic acids",
            "Superclass": "Organic acids and derivatives",
            "Subclass": "Carboxylic acids",
        }
        assert classify_compound_type(classification) == "Synthetic"

    def test_empty_classification(self):
        classification = {"NP_Pathway": "", "Class": "", "Superclass": "", "Subclass": ""}
        assert classify_compound_type(classification) == "Synthetic"

    def test_none_values_handled(self):
        classification = {"NP_Pathway": None, "Class": None, "Superclass": None, "Subclass": None}
        assert classify_compound_type(classification) == "Synthetic"


# ---------------------------------------------------------------------------
# get_classification_summary (sync pure function)
# ---------------------------------------------------------------------------


class TestGetClassificationSummary:
    """Tests for get_classification_summary."""

    def test_summary_with_full_data(self):
        classification = {
            "Kingdom": "Organic compounds",
            "Superclass": "Phenylpropanoids",
            "Class": "Flavonoids",
            "Subclass": "Flavones",
            "NP_Pathway": "Shikimates",
            "NP_Superclass": "Flavonoids",
            "NP_Class": "Flavones",
            "NP_isglycoside": False,
            "Molecular_Framework": "Aromatic",
        }
        summary = get_classification_summary(classification)
        assert "Chemical Classification Summary" in summary
        assert "Flavonoids" in summary
        assert "Shikimates" in summary
        assert "Natural Product" in summary

    def test_summary_with_glycoside(self):
        classification = {
            "Kingdom": "",
            "Superclass": "",
            "Class": "",
            "Subclass": "",
            "NP_Pathway": "Carbohydrates",
            "NP_Superclass": "Glycosides",
            "NP_Class": "O-glycosides",
            "NP_isglycoside": True,
            "Molecular_Framework": "",
        }
        summary = get_classification_summary(classification)
        assert "glycoside moiety" in summary

    def test_summary_no_classification(self):
        classification = {
            "Kingdom": "",
            "Superclass": "",
            "Class": "",
            "Subclass": "",
            "NP_Pathway": "",
            "NP_Superclass": "",
            "NP_Class": "",
            "NP_isglycoside": False,
            "Molecular_Framework": "",
        }
        summary = get_classification_summary(classification)
        assert "No classification available" in summary


# ---------------------------------------------------------------------------
# Structural verification
# ---------------------------------------------------------------------------


class TestNoThreadingArtifacts:
    """Verify removed sync infrastructure."""

    def test_no_threading_local(self):
        import backend.modules.chemical_classifier as m
        source = inspect.getsource(m)
        assert "threading.local()" not in source

    def test_no_import_requests(self):
        import backend.modules.chemical_classifier as m
        source = inspect.getsource(m)
        assert "import requests" not in source

    def test_no_import_threading(self):
        import backend.modules.chemical_classifier as m
        source = inspect.getsource(m)
        assert "import threading" not in source

    def test_no_create_session(self):
        import backend.modules.chemical_classifier as m
        source = inspect.getsource(m)
        assert "def _create_session(" not in source

    def test_no_get_classification_dead_function(self):
        """Legacy get_classification function should be removed."""
        import backend.modules.chemical_classifier as m
        source = inspect.getsource(m)
        assert "def get_classification(" not in source


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestCreateClassifierClient:
    """Tests for create_classifier_client factory."""

    def test_returns_async_client(self):
        client = create_classifier_client()
        assert isinstance(client, httpx.AsyncClient)


class TestShutdownClassifier:
    """Tests for shutdown_classifier."""

    def test_shutdown_no_error(self):
        shutdown_classifier()
