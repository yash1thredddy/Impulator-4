"""Unit tests for PubChem InChIKey resolution methods."""
import pytest
from unittest.mock import patch, MagicMock
from frontend.services.backend_client import ImpulatorAPIClient


@pytest.fixture
def client():
    return ImpulatorAPIClient()


class TestExtractPubchemSmiles:
    """Test the SMILES extraction helper that handles PubChem field name changes."""

    def test_extracts_connectivity_smiles(self, client):
        props = {"CID": 2244, "ConnectivitySMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}
        assert client._extract_pubchem_smiles(props) == "CC(=O)OC1=CC=CC=C1C(=O)O"

    def test_extracts_smiles_field(self, client):
        props = {"CID": 2244, "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}
        assert client._extract_pubchem_smiles(props) == "CC(=O)OC1=CC=CC=C1C(=O)O"

    def test_prefers_connectivity_over_old_names(self, client):
        props = {"ConnectivitySMILES": "NEW", "CanonicalSMILES": "OLD"}
        assert client._extract_pubchem_smiles(props) == "NEW"

    def test_falls_back_to_canonical(self, client):
        props = {"CID": 2244, "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}
        assert client._extract_pubchem_smiles(props) == "CC(=O)OC1=CC=CC=C1C(=O)O"

    def test_returns_empty_for_no_smiles(self, client):
        props = {"CID": 2244}
        assert client._extract_pubchem_smiles(props) == ""


class TestResolveSingleInchikey:
    """Test single InChIKey -> SMILES resolution via PubChem."""

    @patch('frontend.services.backend_client.requests.get')
    def test_resolves_valid_inchikey(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "PropertyTable": {
                "Properties": [{"CID": 2244, "ConnectivitySMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
            }
        }
        mock_get.return_value = mock_resp

        result = client.resolve_inchikey_to_smiles("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        assert result == "CC(=O)OC1=CC=CC=C1C(=O)O"

    @patch('frontend.services.backend_client.requests.get')
    def test_returns_none_for_404(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = client.resolve_inchikey_to_smiles("AAAAAAAAAAAAA-BBBBBBBBBB-C")
        assert result is None

    @patch('frontend.services.backend_client.requests.get')
    def test_returns_none_on_network_error(self, mock_get, client):
        mock_get.side_effect = Exception("Connection error")
        result = client.resolve_inchikey_to_smiles("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
        assert result is None


class TestResolveBatchInchikeys:
    """Test batch InChIKey -> SMILES resolution via PubChem POST."""

    @patch('frontend.services.backend_client.requests.post')
    def test_resolves_batch(self, mock_post, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "PropertyTable": {
                "Properties": [
                    {"CID": 2244, "ConnectivitySMILES": "CC(=O)OC1=CC=CC=C1C(=O)O", "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
                    {"CID": 5280343, "ConnectivitySMILES": "C1=CC(=C(C=C1O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O", "InChIKey": "REFJWTPEDVJJIY-UHFFFAOYSA-N"},
                ]
            }
        }
        mock_post.return_value = mock_resp

        keys = ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "REFJWTPEDVJJIY-UHFFFAOYSA-N"]
        result = client.resolve_inchikeys_batch(keys)

        assert len(result) == 2
        assert result["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"] == "CC(=O)OC1=CC=CC=C1C(=O)O"

    @patch('frontend.services.backend_client.requests.post')
    def test_returns_empty_dict_on_failure(self, mock_post, client):
        mock_post.side_effect = Exception("Connection error")
        result = client.resolve_inchikeys_batch(["KEY1", "KEY2"])
        assert result == {}
