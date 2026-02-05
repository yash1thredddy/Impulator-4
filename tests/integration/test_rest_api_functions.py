#!/usr/bin/env python
"""
Test ChEMBL REST API functions and response key consistency.

Tests:
1. CHEMBL_RESPONSE_KEYS mapping constant
2. _get_response_data helper function
3. All REST API functions with real API calls
4. Edge cases (empty responses, None data)
"""

import sys
import os
import time
import pytest
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# API Connectivity Test (runs first to diagnose issues)
# =============================================================================

class TestApiConnectivity:
    """Test basic API connectivity - runs first to diagnose issues."""

    def test_chembl_api_reachable(self):
        """Test that ChEMBL API is reachable from Python requests."""
        try:
            response = requests.get(
                "https://www.ebi.ac.uk/chembl/api/data/status.json",
                timeout=30
            )
            print(f"\nAPI Response Status: {response.status_code}")
            print(f"API Response: {response.text[:200]}")
            assert response.status_code == 200, f"API returned {response.status_code}"
            data = response.json()
            assert data.get('status') == 'UP', f"API status is {data.get('status')}"
        except requests.exceptions.Timeout:
            pytest.fail("ChEMBL API request timed out after 30s")
        except requests.exceptions.ConnectionError as e:
            pytest.fail(f"ChEMBL API connection error: {e}")
        except Exception as e:
            pytest.fail(f"ChEMBL API check failed: {type(e).__name__}: {e}")

from backend.modules.api_client import (
    # Constants
    CHEMBL_RESPONSE_KEYS,
    CHEMBL_MAX_LIMIT,
    DEFAULT_ACTIVITY_TYPES,
    SIMILARITY_SEARCH_TIMEOUT,
    # Helper functions
    _get_response_data,
    _rest_api_get,
    # REST API functions
    rest_api_similarity_search,
    rest_api_fetch_molecule,
    rest_api_fetch_molecules_batch,
    rest_api_fetch_target,
    rest_api_fetch_targets_batch,
    rest_api_fetch_activities,
    rest_api_fetch_drug_indications_batch,
)


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Test that constants are properly defined."""

    def test_chembl_response_keys_defined(self):
        """Test CHEMBL_RESPONSE_KEYS mapping exists and has expected keys."""
        assert isinstance(CHEMBL_RESPONSE_KEYS, dict)
        assert 'activity' in CHEMBL_RESPONSE_KEYS
        assert 'molecule' in CHEMBL_RESPONSE_KEYS
        assert 'target' in CHEMBL_RESPONSE_KEYS
        assert 'similarity' in CHEMBL_RESPONSE_KEYS
        assert 'drug_indication' in CHEMBL_RESPONSE_KEYS

    def test_chembl_response_keys_values(self):
        """Test CHEMBL_RESPONSE_KEYS has correct response key values."""
        assert CHEMBL_RESPONSE_KEYS['activity'] == 'activities'
        assert CHEMBL_RESPONSE_KEYS['molecule'] == 'molecules'
        assert CHEMBL_RESPONSE_KEYS['target'] == 'targets'
        assert CHEMBL_RESPONSE_KEYS['similarity'] == 'molecules'
        assert CHEMBL_RESPONSE_KEYS['drug_indication'] == 'drug_indications'

    def test_chembl_max_limit(self):
        """Test CHEMBL_MAX_LIMIT is set to 1000."""
        assert CHEMBL_MAX_LIMIT == 1000

    def test_similarity_search_timeout(self):
        """Test SIMILARITY_SEARCH_TIMEOUT is defined."""
        assert SIMILARITY_SEARCH_TIMEOUT == 90

    def test_default_activity_types(self):
        """Test DEFAULT_ACTIVITY_TYPES contains expected types."""
        assert isinstance(DEFAULT_ACTIVITY_TYPES, list)
        assert 'IC50' in DEFAULT_ACTIVITY_TYPES
        assert 'Ki' in DEFAULT_ACTIVITY_TYPES
        assert 'Kd' in DEFAULT_ACTIVITY_TYPES
        assert 'EC50' in DEFAULT_ACTIVITY_TYPES
        assert len(DEFAULT_ACTIVITY_TYPES) >= 4


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestGetResponseData:
    """Test _get_response_data helper function."""

    def test_returns_empty_list_for_none_data(self):
        """Test that None data returns empty list."""
        result = _get_response_data(None, 'activity')
        assert result == []

    def test_returns_empty_list_for_missing_key(self):
        """Test that missing key returns empty list."""
        data = {'other_key': [1, 2, 3]}
        result = _get_response_data(data, 'activity')
        assert result == []

    def test_extracts_activities(self):
        """Test extracting activities from response."""
        data = {'activities': [{'id': 1}, {'id': 2}]}
        result = _get_response_data(data, 'activity')
        assert result == [{'id': 1}, {'id': 2}]

    def test_extracts_molecules(self):
        """Test extracting molecules from response."""
        data = {'molecules': [{'chembl_id': 'CHEMBL25'}]}
        result = _get_response_data(data, 'molecule')
        assert result == [{'chembl_id': 'CHEMBL25'}]

    def test_extracts_targets(self):
        """Test extracting targets from response."""
        data = {'targets': [{'target_chembl_id': 'CHEMBL220'}]}
        result = _get_response_data(data, 'target')
        assert result == [{'target_chembl_id': 'CHEMBL220'}]

    def test_extracts_similarity_results(self):
        """Test extracting similarity results (uses 'molecules' key)."""
        data = {'molecules': [{'molecule_chembl_id': 'CHEMBL50'}]}
        result = _get_response_data(data, 'similarity')
        assert result == [{'molecule_chembl_id': 'CHEMBL50'}]

    def test_extracts_drug_indications(self):
        """Test extracting drug indications from response."""
        data = {'drug_indications': [{'mesh_heading': 'Fever'}]}
        result = _get_response_data(data, 'drug_indication')
        assert result == [{'mesh_heading': 'Fever'}]

    def test_fallback_for_unknown_endpoint(self):
        """Test fallback pluralization for unknown endpoint."""
        data = {'unknowns': [{'data': 'test'}]}
        result = _get_response_data(data, 'unknown')
        assert result == [{'data': 'test'}]

    def test_empty_dict_returns_empty_list(self):
        """Test that empty dict returns empty list."""
        result = _get_response_data({}, 'activity')
        assert result == []


# =============================================================================
# Test REST API Functions (Integration Tests)
# =============================================================================

@pytest.fixture(scope="module")
def check_api_available():
    """Check if ChEMBL API is available before running integration tests."""
    import requests
    try:
        response = requests.get(
            "https://www.ebi.ac.uk/chembl/api/data/status.json",
            timeout=30
        )
        if response.status_code != 200:
            pytest.skip(f"ChEMBL API returned status {response.status_code}")
        data = response.json()
        if data.get('status') != 'UP':
            pytest.skip(f"ChEMBL API status is {data.get('status')}")
        # Return the status data for potential use in tests
        return data
    except requests.exceptions.Timeout:
        pytest.skip("ChEMBL API request timed out")
    except requests.exceptions.ConnectionError as e:
        pytest.skip(f"ChEMBL API connection error: {e}")
    except Exception as e:
        pytest.skip(f"ChEMBL API check failed: {type(e).__name__}: {e}")


class TestRestApiGet:
    """Test _rest_api_get function."""

    def test_returns_dict_for_valid_request(self, check_api_available):
        """Test that valid request returns a dict."""
        params = {"molecule_chembl_id": "CHEMBL25"}
        result = _rest_api_get("molecule", params, timeout=30)
        assert result is not None
        assert isinstance(result, dict)

    def test_returns_none_for_invalid_endpoint(self, check_api_available):
        """Test that invalid endpoint returns None."""
        result = _rest_api_get("invalid_endpoint_xyz", {}, timeout=10)
        assert result is None

    def test_response_contains_expected_keys(self, check_api_available):
        """Test that response contains page_meta and data keys."""
        params = {"molecule_chembl_id": "CHEMBL25", "limit": 1}
        result = _rest_api_get("molecule", params, timeout=30)
        assert result is not None
        assert 'page_meta' in result or 'molecules' in result


class TestRestApiSimilaritySearch:
    """Test rest_api_similarity_search function."""

    def test_returns_list(self, check_api_available):
        """Test that function returns a list."""
        # Aspirin SMILES
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        result = rest_api_similarity_search(smiles, similarity_threshold=90)
        assert isinstance(result, list)

    def test_results_have_chembl_id_key(self, check_api_available):
        """Test that results contain 'ChEMBL ID' key."""
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        result = rest_api_similarity_search(smiles, similarity_threshold=90)
        if result:  # Only check if we got results
            assert 'ChEMBL ID' in result[0]

    def test_empty_smiles_returns_empty_list(self, check_api_available):
        """Test that empty SMILES returns empty list."""
        result = rest_api_similarity_search("", similarity_threshold=90)
        assert result == []


class TestRestApiFetchMolecule:
    """Test rest_api_fetch_molecule function."""

    def test_returns_dict_for_valid_id(self, check_api_available):
        """Test that valid ChEMBL ID returns molecule data."""
        result = rest_api_fetch_molecule("CHEMBL25")
        assert result is not None
        assert isinstance(result, dict)

    def test_returns_none_for_invalid_id(self, check_api_available):
        """Test that invalid ChEMBL ID returns None."""
        result = rest_api_fetch_molecule("CHEMBL_INVALID_12345678")
        assert result is None

    def test_result_contains_molecule_chembl_id(self, check_api_available):
        """Test that result contains molecule_chembl_id."""
        result = rest_api_fetch_molecule("CHEMBL25")
        assert result is not None
        assert 'molecule_chembl_id' in result


class TestRestApiFetchMoleculesBatch:
    """Test rest_api_fetch_molecules_batch function."""

    def test_returns_dict(self, check_api_available):
        """Test that function returns a dict."""
        chembl_ids = ["CHEMBL25", "CHEMBL521"]
        result = rest_api_fetch_molecules_batch(chembl_ids)
        assert isinstance(result, dict)

    def test_returns_data_for_valid_ids(self, check_api_available):
        """Test that valid IDs return molecule data."""
        chembl_ids = ["CHEMBL25", "CHEMBL521", "CHEMBL192"]
        result = rest_api_fetch_molecules_batch(chembl_ids)
        assert len(result) > 0
        assert "CHEMBL25" in result or "CHEMBL521" in result

    def test_empty_list_returns_empty_dict(self, check_api_available):
        """Test that empty list returns empty dict."""
        result = rest_api_fetch_molecules_batch([])
        assert result == {}


class TestRestApiFetchTarget:
    """Test rest_api_fetch_target function."""

    def test_returns_string_for_valid_id(self, check_api_available):
        """Test that valid target ID returns target name."""
        result = rest_api_fetch_target("CHEMBL220")
        assert result is not None
        assert isinstance(result, str)

    def test_returns_none_for_invalid_id(self, check_api_available):
        """Test that invalid target ID returns None."""
        result = rest_api_fetch_target("CHEMBL_TARGET_INVALID")
        assert result is None

    def test_returns_acetylcholinesterase(self, check_api_available):
        """Test that CHEMBL220 returns Acetylcholinesterase."""
        result = rest_api_fetch_target("CHEMBL220")
        assert result is not None
        assert "Acetylcholinesterase" in result


class TestRestApiFetchTargetsBatch:
    """Test rest_api_fetch_targets_batch function."""

    def test_returns_dict(self, check_api_available):
        """Test that function returns a dict."""
        target_ids = ["CHEMBL220", "CHEMBL204"]
        result = rest_api_fetch_targets_batch(target_ids)
        assert isinstance(result, dict)

    def test_returns_data_for_valid_ids(self, check_api_available):
        """Test that valid target IDs return target names."""
        target_ids = ["CHEMBL220", "CHEMBL204", "CHEMBL205"]
        result = rest_api_fetch_targets_batch(target_ids)
        assert len(result) > 0

    def test_empty_list_returns_empty_dict(self, check_api_available):
        """Test that empty list returns empty dict."""
        result = rest_api_fetch_targets_batch([])
        assert result == {}


class TestRestApiFetchActivities:
    """Test rest_api_fetch_activities function."""

    def test_returns_list(self, check_api_available):
        """Test that function returns a list."""
        chembl_ids = ["CHEMBL25"]
        result = rest_api_fetch_activities(chembl_ids)
        assert isinstance(result, list)

    def test_returns_activities_for_aspirin(self, check_api_available):
        """Test that Aspirin (CHEMBL25) returns activities."""
        chembl_ids = ["CHEMBL25"]
        result = rest_api_fetch_activities(chembl_ids)
        assert len(result) > 0

    def test_filters_by_activity_type(self, check_api_available):
        """Test that activity type filtering works."""
        chembl_ids = ["CHEMBL25"]
        result = rest_api_fetch_activities(chembl_ids, activity_types=['IC50'])
        if result:
            for activity in result:
                assert activity.get('standard_type') == 'IC50'

    def test_empty_list_returns_empty_list(self, check_api_available):
        """Test that empty list returns empty list."""
        result = rest_api_fetch_activities([])
        assert result == []


class TestRestApiFetchDrugIndicationsBatch:
    """Test rest_api_fetch_drug_indications_batch function."""

    def test_returns_list(self, check_api_available):
        """Test that function returns a list."""
        chembl_ids = ["CHEMBL25"]
        result = rest_api_fetch_drug_indications_batch(chembl_ids)
        assert isinstance(result, list)

    def test_returns_indications_for_aspirin(self, check_api_available):
        """Test that Aspirin (CHEMBL25) returns drug indications."""
        chembl_ids = ["CHEMBL25"]
        result = rest_api_fetch_drug_indications_batch(chembl_ids)
        assert len(result) > 0

    def test_indications_have_expected_fields(self, check_api_available):
        """Test that indications have expected fields."""
        chembl_ids = ["CHEMBL25"]
        result = rest_api_fetch_drug_indications_batch(chembl_ids)
        if result:
            indication = result[0]
            assert 'molecule_chembl_id' in indication

    def test_empty_list_returns_empty_list(self, check_api_available):
        """Test that empty list returns empty list."""
        result = rest_api_fetch_drug_indications_batch([])
        assert result == []


# =============================================================================
# Test Response Key Consistency Across Functions
# =============================================================================

class TestResponseKeyConsistency:
    """Test that all REST API functions use consistent response key handling."""

    def test_all_endpoints_in_mapping(self):
        """Test that all used endpoints are in the mapping."""
        expected_endpoints = ['activity', 'molecule', 'target', 'similarity', 'drug_indication']
        for endpoint in expected_endpoints:
            assert endpoint in CHEMBL_RESPONSE_KEYS, f"Missing endpoint: {endpoint}"

    def test_mapping_covers_all_rest_functions(self):
        """Test that the mapping covers all REST API function needs."""
        # These are the endpoints used by our REST API functions
        used_endpoints = {
            'rest_api_fetch_activities': 'activity',
            'rest_api_similarity_search': 'similarity',
            'rest_api_fetch_molecule': 'molecule',
            'rest_api_fetch_molecules_batch': 'molecule',
            'rest_api_fetch_target': 'target',
            'rest_api_fetch_targets_batch': 'target',
            'rest_api_fetch_drug_indications_batch': 'drug_indication',
        }

        for func_name, endpoint in used_endpoints.items():
            assert endpoint in CHEMBL_RESPONSE_KEYS, \
                f"Endpoint '{endpoint}' used by {func_name} not in CHEMBL_RESPONSE_KEYS"


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ChEMBL REST API Function Tests")
    print("="*70)

    # Run pytest with verbose output
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
