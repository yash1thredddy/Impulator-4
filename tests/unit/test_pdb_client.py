"""
Unit tests for RCSB PDB Client Module.

Tests the PDB integration for structural evidence scoring including:
- Similar ligand search
- Structure details retrieval
- Resolution fetching (REST and GraphQL)
- Quality classification
- PDB evidence scoring
- Error handling and edge cases
"""
import pytest
from unittest.mock import patch, MagicMock
import json


class TestSearchSimilarLigands:
    """Tests for search_similar_ligands function."""

    def test_search_returns_pdb_ids(self):
        """Test successful search returns list of PDB IDs."""
        from backend.modules.pdb_client import search_similar_ligands

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'result_set': [
                {'identifier': '4HHB'},
                {'identifier': '3WHM'},
                {'identifier': '2CPK'}
            ]
        }

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response):
            # Clear cache before test
            search_similar_ligands.cache_clear()
            result = search_similar_ligands('CCO')

        assert result == ['4HHB', '3WHM', '2CPK']

    def test_search_no_results(self):
        """Test search with no results returns empty list."""
        from backend.modules.pdb_client import search_similar_ligands

        mock_response = MagicMock()
        mock_response.status_code = 204  # No content

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response):
            search_similar_ligands.cache_clear()
            result = search_similar_ligands('CCCCCCCCCC')

        assert result == []

    def test_search_limits_results(self):
        """Test that search limits results to 100."""
        from backend.modules.pdb_client import search_similar_ligands

        # Create 150 mock results
        mock_results = [{'identifier': f'PDB{i:04d}'} for i in range(150)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'result_set': mock_results}

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response):
            search_similar_ligands.cache_clear()
            result = search_similar_ligands('CCO')

        assert len(result) == 100

    def test_search_handles_timeout(self):
        """Test search handles timeout gracefully."""
        from backend.modules.pdb_client import search_similar_ligands
        import requests

        with patch('backend.modules.pdb_client.requests.post', side_effect=requests.exceptions.Timeout()):
            search_similar_ligands.cache_clear()
            result = search_similar_ligands('CCO')

        assert result == []

    def test_search_handles_server_error_with_retry(self):
        """Test search retries on 500 error."""
        from backend.modules.pdb_client import search_similar_ligands

        mock_error_response = MagicMock()
        mock_error_response.status_code = 500

        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            'result_set': [{'identifier': '4HHB'}]
        }

        with patch('backend.modules.pdb_client.requests.post',
                   side_effect=[mock_error_response, mock_success_response]):
            with patch('backend.modules.pdb_client.time.sleep'):  # Skip delays
                search_similar_ligands.cache_clear()
                result = search_similar_ligands('CCO')

        assert result == ['4HHB']

    def test_search_caches_results(self):
        """Test that results are cached."""
        from backend.modules.pdb_client import search_similar_ligands

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'result_set': [{'identifier': '4HHB'}]
        }

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response) as mock_post:
            search_similar_ligands.cache_clear()
            result1 = search_similar_ligands('CCO')
            result2 = search_similar_ligands('CCO')

        # Should only call API once due to caching
        assert mock_post.call_count == 1
        assert result1 == result2


class TestGetStructureResolution:
    """Tests for get_structure_resolution function."""

    def test_resolution_returned(self):
        """Test successful resolution retrieval."""
        from backend.modules.pdb_client import get_structure_resolution

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'rcsb_entry_info': {
                'resolution_combined': [1.74]
            }
        }

        with patch.object(
            __import__('backend.modules.pdb_client', fromlist=['session']).session,
            'get',
            return_value=mock_response
        ):
            get_structure_resolution.cache_clear()
            result = get_structure_resolution('4HHB')

        assert result == 1.74

    def test_resolution_none_when_missing(self):
        """Test returns None when resolution not available."""
        from backend.modules.pdb_client import get_structure_resolution

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'rcsb_entry_info': {}
        }

        with patch.object(
            __import__('backend.modules.pdb_client', fromlist=['session']).session,
            'get',
            return_value=mock_response
        ):
            get_structure_resolution.cache_clear()
            result = get_structure_resolution('XXXX')

        assert result is None

    def test_resolution_handles_error(self):
        """Test handles API errors gracefully."""
        from backend.modules.pdb_client import get_structure_resolution
        import requests

        with patch.object(
            __import__('backend.modules.pdb_client', fromlist=['session']).session,
            'get',
            side_effect=requests.exceptions.RequestException("Network error")
        ):
            get_structure_resolution.cache_clear()
            result = get_structure_resolution('4HHB')

        assert result is None


class TestGetBatchStructureResolutionsGraphQL:
    """Tests for get_batch_structure_resolutions_graphql function."""

    def test_graphql_returns_resolutions(self):
        """Test GraphQL returns resolutions for multiple IDs."""
        from backend.modules.pdb_client import get_batch_structure_resolutions_graphql

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'entries': [
                    {'rcsb_id': '4HHB', 'rcsb_entry_info': {'resolution_combined': [1.74]}},
                    {'rcsb_id': '3WHM', 'rcsb_entry_info': {'resolution_combined': [2.10]}},
                    {'rcsb_id': '2CPK', 'rcsb_entry_info': {'resolution_combined': None}}
                ]
            }
        }

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response):
            result = get_batch_structure_resolutions_graphql(['4HHB', '3WHM', '2CPK'])

        assert result['4HHB'] == 1.74
        assert result['3WHM'] == 2.10
        assert result['2CPK'] is None

    def test_graphql_empty_input(self):
        """Test GraphQL with empty input returns empty dict."""
        from backend.modules.pdb_client import get_batch_structure_resolutions_graphql

        result = get_batch_structure_resolutions_graphql([])

        assert result == {}

    def test_graphql_normalizes_case(self):
        """Test GraphQL normalizes PDB IDs to uppercase."""
        from backend.modules.pdb_client import get_batch_structure_resolutions_graphql

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'entries': [
                    {'rcsb_id': '4HHB', 'rcsb_entry_info': {'resolution_combined': [1.74]}}
                ]
            }
        }

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response):
            result = get_batch_structure_resolutions_graphql(['4hhb'])

        assert '4HHB' in result

    def test_graphql_handles_error(self):
        """Test GraphQL handles errors gracefully."""
        from backend.modules.pdb_client import get_batch_structure_resolutions_graphql
        import requests

        with patch('backend.modules.pdb_client.requests.post',
                   side_effect=requests.exceptions.RequestException()):
            result = get_batch_structure_resolutions_graphql(['4HHB'])

        assert result == {}


class TestGetBatchStructureResolutions:
    """Tests for get_batch_structure_resolutions function."""

    def test_batch_uses_graphql_first(self):
        """Test that batch function uses GraphQL first."""
        from backend.modules.pdb_client import get_batch_structure_resolutions

        with patch('backend.modules.pdb_client.get_batch_structure_resolutions_graphql',
                   return_value={'4HHB': 1.74}) as mock_graphql:
            result = get_batch_structure_resolutions(['4HHB'])

        mock_graphql.assert_called_once()
        assert result['4HHB'] == 1.74

    def test_batch_falls_back_to_rest(self):
        """Test that batch falls back to REST on GraphQL failure."""
        from backend.modules.pdb_client import get_batch_structure_resolutions

        with patch('backend.modules.pdb_client.get_batch_structure_resolutions_graphql',
                   return_value={}):
            with patch('backend.modules.pdb_client.get_structure_resolution',
                       return_value=1.74) as mock_rest:
                with patch('backend.modules.pdb_client.time.sleep'):
                    result = get_batch_structure_resolutions(['4HHB'])

        mock_rest.assert_called()
        assert result['4HHB'] == 1.74

    def test_batch_empty_input(self):
        """Test batch with empty input returns empty dict."""
        from backend.modules.pdb_client import get_batch_structure_resolutions

        result = get_batch_structure_resolutions([])

        assert result == {}


class TestClassifyResolutionQuality:
    """Tests for classify_resolution_quality function."""

    def test_high_quality(self):
        """Test classification of high-quality resolution (< 2.0)."""
        from backend.modules.pdb_client import classify_resolution_quality

        quality, multiplier = classify_resolution_quality(1.5)

        assert quality == "***"
        assert multiplier == 1.0

    def test_medium_quality(self):
        """Test classification of medium-quality resolution (2.0-3.0)."""
        from backend.modules.pdb_client import classify_resolution_quality

        quality, multiplier = classify_resolution_quality(2.5)

        assert quality == "**"
        assert multiplier == 0.75

    def test_poor_quality(self):
        """Test classification of poor-quality resolution (> 3.0)."""
        from backend.modules.pdb_client import classify_resolution_quality

        quality, multiplier = classify_resolution_quality(3.5)

        assert quality == "*"
        assert multiplier == 0.5

    def test_boundary_2_angstrom(self):
        """Test boundary at exactly 2.0 Angstrom (medium)."""
        from backend.modules.pdb_client import classify_resolution_quality

        quality, multiplier = classify_resolution_quality(2.0)

        assert quality == "**"  # 2.0 is medium quality
        assert multiplier == 0.75

    def test_boundary_3_angstrom(self):
        """Test boundary at exactly 3.0 Angstrom (medium)."""
        from backend.modules.pdb_client import classify_resolution_quality

        quality, multiplier = classify_resolution_quality(3.0)

        assert quality == "**"  # 3.0 is still medium quality
        assert multiplier == 0.75


class TestGetPDBEvidenceScore:
    """Tests for get_pdb_evidence_score function."""

    def test_score_with_high_quality_structures(self):
        """Test score calculation with high-quality structures."""
        from backend.modules.pdb_client import get_pdb_evidence_score

        with patch('backend.modules.pdb_client.search_similar_ligands',
                   return_value=['4HHB', '3WHM', '2CPK', '1ABC', '5XYZ']):
            with patch('backend.modules.pdb_client.get_batch_structure_resolutions',
                       return_value={
                           '4HHB': 1.5, '3WHM': 1.8, '2CPK': 1.9,
                           '1ABC': 1.7, '5XYZ': 1.6
                       }):
                result = get_pdb_evidence_score('CCO')

        assert result['pdb_score'] == 1.0  # 5+ high-quality = max score
        assert result['num_structures'] == 5
        assert result['num_high_quality'] == 5
        assert result['num_medium_quality'] == 0
        assert result['num_poor_quality'] == 0

    def test_score_with_mixed_quality(self):
        """Test score calculation with mixed-quality structures."""
        from backend.modules.pdb_client import get_pdb_evidence_score

        with patch('backend.modules.pdb_client.search_similar_ligands',
                   return_value=['4HHB', '3WHM', '2CPK']):
            with patch('backend.modules.pdb_client.get_batch_structure_resolutions',
                       return_value={
                           '4HHB': 1.5,  # High
                           '3WHM': 2.5,  # Medium
                           '2CPK': 3.5   # Poor
                       }):
                result = get_pdb_evidence_score('CCO')

        assert result['num_high_quality'] == 1
        assert result['num_medium_quality'] == 1
        assert result['num_poor_quality'] == 1
        assert 0 < result['pdb_score'] < 1.0

    def test_score_no_structures(self):
        """Test score when no structures found."""
        from backend.modules.pdb_client import get_pdb_evidence_score

        with patch('backend.modules.pdb_client.search_similar_ligands', return_value=[]):
            result = get_pdb_evidence_score('CCCCCCCCCC')

        assert result['pdb_score'] == 0.0
        assert result['num_structures'] == 0
        assert result['pdb_ids'] == []

    def test_score_structures_without_resolution(self):
        """Test score when structures lack resolution data."""
        from backend.modules.pdb_client import get_pdb_evidence_score

        with patch('backend.modules.pdb_client.search_similar_ligands',
                   return_value=['4HHB', '3WHM']):
            with patch('backend.modules.pdb_client.get_batch_structure_resolutions',
                       return_value={'4HHB': None, '3WHM': None}):
                result = get_pdb_evidence_score('CCO')

        assert result['pdb_score'] == 0.0
        assert result['num_structures'] == 2


class TestGetStructureDetails:
    """Tests for get_structure_details function."""

    def test_details_returned(self):
        """Test successful details retrieval."""
        from backend.modules.pdb_client import get_structure_details

        mock_entry_response = MagicMock()
        mock_entry_response.status_code = 200
        mock_entry_response.json.return_value = {
            'struct': {'title': 'Test Structure'},
            'rcsb_entry_info': {'resolution_combined': [1.74]},
            'exptl': [{'method': 'X-RAY DIFFRACTION'}],
            'rcsb_primary_citation': {'pdbx_database_id_DOI': '10.1234/test'}
        }

        mock_entity_response = MagicMock()
        mock_entity_response.status_code = 200
        mock_entity_response.json.return_value = {
            'rcsb_polymer_entity_container_identifiers': {
                'reference_sequence_identifiers': [
                    {'database_name': 'UniProt', 'database_accession': 'P12345'}
                ]
            }
        }

        with patch.object(
            __import__('backend.modules.pdb_client', fromlist=['session']).session,
            'get',
            side_effect=[mock_entry_response, mock_entity_response]
        ):
            get_structure_details.cache_clear()
            result = get_structure_details('4HHB')

        assert result['pdb_id'] == '4HHB'
        assert result['title'] == 'Test Structure'
        assert result['resolution'] == 1.74
        assert result['doi'] == '10.1234/test'
        assert 'P12345' in result['uniprot_ids']
        assert result['experimental_method'] == 'X-RAY DIFFRACTION'

    def test_details_minimal_data(self):
        """Test details with minimal data available."""
        from backend.modules.pdb_client import get_structure_details

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        with patch.object(
            __import__('backend.modules.pdb_client', fromlist=['session']).session,
            'get',
            return_value=mock_response
        ):
            get_structure_details.cache_clear()
            result = get_structure_details('XXXX')

        assert result['pdb_id'] == 'XXXX'
        assert result['title'] is None
        assert result['resolution'] is None
        assert 'url' in result


class TestGetPDBSummary:
    """Tests for get_pdb_summary_for_compound function."""

    def test_summary_with_structures(self):
        """Test summary generation with structures found."""
        from backend.modules.pdb_client import get_pdb_summary_for_compound

        with patch('backend.modules.pdb_client.get_pdb_evidence_score', return_value={
            'pdb_score': 0.8,
            'num_structures': 3,
            'num_high_quality': 2,
            'num_medium_quality': 1,
            'num_poor_quality': 0,
            'pdb_ids': ['4HHB', '3WHM', '2CPK'],
            'resolutions': [1.74, 1.9, 2.5],
            'quality_classes': ['***', '***', '**']
        }):
            summary = get_pdb_summary_for_compound('CCO')

        assert 'Found 3 similar structure(s)' in summary
        assert 'PDB Evidence Score: 0.800' in summary
        assert '2 high-quality' in summary
        assert '4HHB' in summary

    def test_summary_no_structures(self):
        """Test summary when no structures found."""
        from backend.modules.pdb_client import get_pdb_summary_for_compound

        with patch('backend.modules.pdb_client.get_pdb_evidence_score', return_value={
            'pdb_score': 0.0,
            'num_structures': 0,
            'num_high_quality': 0,
            'num_medium_quality': 0,
            'num_poor_quality': 0,
            'pdb_ids': [],
            'resolutions': [],
            'quality_classes': []
        }):
            summary = get_pdb_summary_for_compound('CCCCCCCCCC')

        assert 'No experimental structures found' in summary


class TestGetDetailedPDBStructures:
    """Tests for get_detailed_pdb_structures function."""

    def test_detailed_structures_sorted(self):
        """Test that detailed structures are sorted by quality."""
        from backend.modules.pdb_client import get_detailed_pdb_structures

        mock_evidence = {
            'pdb_ids': ['4HHB', '3WHM', '2CPK'],
            'resolutions': [3.5, 1.5, 2.5],
            'quality_classes': ['*', '***', '**']
        }

        mock_details = {
            '4HHB': {'pdb_id': '4HHB', 'title': 'Poor', 'resolution': 3.5,
                     'uniprot_ids': [], 'experimental_method': 'X-RAY',
                     'url': 'https://rcsb.org/4HHB', 'doi': None},
            '3WHM': {'pdb_id': '3WHM', 'title': 'Best', 'resolution': 1.5,
                     'uniprot_ids': [], 'experimental_method': 'X-RAY',
                     'url': 'https://rcsb.org/3WHM', 'doi': None},
            '2CPK': {'pdb_id': '2CPK', 'title': 'Medium', 'resolution': 2.5,
                     'uniprot_ids': [], 'experimental_method': 'X-RAY',
                     'url': 'https://rcsb.org/2CPK', 'doi': None}
        }

        with patch('backend.modules.pdb_client.get_pdb_evidence_score', return_value=mock_evidence):
            with patch('backend.modules.pdb_client.get_structure_details',
                       side_effect=lambda x: mock_details[x]):
                with patch('backend.modules.pdb_client.time.sleep'):
                    result = get_detailed_pdb_structures('CCO')

        # Should be sorted: *** first, then ** , then *
        assert result[0]['Quality'] == '***'
        assert result[1]['Quality'] == '**'
        assert result[2]['Quality'] == '*'

    def test_detailed_structures_empty(self):
        """Test detailed structures with no PDB IDs."""
        from backend.modules.pdb_client import get_detailed_pdb_structures

        with patch('backend.modules.pdb_client.get_pdb_evidence_score', return_value={
            'pdb_ids': [],
            'resolutions': [],
            'quality_classes': []
        }):
            result = get_detailed_pdb_structures('CCCCCCCCCC')

        assert result == []


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_smiles_search(self):
        """Test search with invalid SMILES."""
        from backend.modules.pdb_client import search_similar_ligands

        mock_response = MagicMock()
        mock_response.status_code = 400  # Bad request

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response):
            search_similar_ligands.cache_clear()
            result = search_similar_ligands('invalid_smiles')

        assert result == []

    def test_empty_smiles_search(self):
        """Test search with empty SMILES."""
        from backend.modules.pdb_client import search_similar_ligands

        mock_response = MagicMock()
        mock_response.status_code = 400

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response):
            search_similar_ligands.cache_clear()
            result = search_similar_ligands('')

        assert result == []

    def test_json_decode_error(self):
        """Test handling of JSON decode error."""
        from backend.modules.pdb_client import search_similar_ligands

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Error", "", 0)

        with patch('backend.modules.pdb_client.requests.post', return_value=mock_response):
            search_similar_ligands.cache_clear()
            result = search_similar_ligands('CCO')

        assert result == []

    def test_resolution_very_high_value(self):
        """Test resolution classification with very high value."""
        from backend.modules.pdb_client import classify_resolution_quality

        quality, multiplier = classify_resolution_quality(10.0)

        assert quality == "*"
        assert multiplier == 0.5

    def test_resolution_very_low_value(self):
        """Test resolution classification with very low value."""
        from backend.modules.pdb_client import classify_resolution_quality

        quality, multiplier = classify_resolution_quality(0.5)

        assert quality == "***"
        assert multiplier == 1.0
