"""
Unit tests for Chemical Classification Module.

Tests ClassyFire and NPClassifier integration including:
- ClassyFire API classification
- NPClassifier API classification
- Complete classification combination
- Compound type inference
- Error handling
"""
import pytest
from unittest.mock import patch, MagicMock


class TestGetClassyFireClassification:
    """Tests for get_classyfire_classification function."""

    def test_successful_classification(self):
        """Test successful ClassyFire classification."""
        from backend.modules.chemical_classifier import get_classyfire_classification

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'kingdom': {'name': 'Organic compounds'},
            'superclass': {'name': 'Phenylpropanoids and polyketides'},
            'class': {'name': 'Flavonoids', 'chemont_id': 'CHEMONTID:0000001'},
            'subclass': {'name': 'Flavones', 'chemont_id': 'CHEMONTID:0000002'},
            'direct_parent': {'name': 'Hydroxyflavones'},
            'molecular_framework': 'Aromatic homomonocyclic compounds',
            'description': 'A flavonoid compound'
        }

        with patch('backend.modules.chemical_classifier.session.get', return_value=mock_response):
            result = get_classyfire_classification('REFJWTPEDVJJIY-UHFFFAOYSA-N')

        assert result is not None
        assert result['kingdom']['name'] == 'Organic compounds'
        assert result['class']['name'] == 'Flavonoids'

    def test_not_found(self):
        """Test ClassyFire returns None for unknown InChIKey."""
        from backend.modules.chemical_classifier import get_classyfire_classification

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch('backend.modules.chemical_classifier.session.get', return_value=mock_response):
            result = get_classyfire_classification('UNKNOWN-INCHIKEY')

        assert result is None

    def test_timeout_handling(self):
        """Test ClassyFire handles timeout gracefully."""
        from backend.modules.chemical_classifier import get_classyfire_classification
        import requests

        with patch('backend.modules.chemical_classifier.session.get',
                   side_effect=requests.exceptions.Timeout()):
            result = get_classyfire_classification('REFJWTPEDVJJIY-UHFFFAOYSA-N')

        assert result is None

    def test_error_handling(self):
        """Test ClassyFire handles general errors gracefully."""
        from backend.modules.chemical_classifier import get_classyfire_classification
        import requests

        with patch('backend.modules.chemical_classifier.session.get',
                   side_effect=requests.exceptions.RequestException("Network error")):
            result = get_classyfire_classification('REFJWTPEDVJJIY-UHFFFAOYSA-N')

        assert result is None


class TestGetNPClassifierClassification:
    """Tests for get_npclassifier_classification function."""

    def test_successful_classification(self):
        """Test successful NPClassifier classification."""
        from backend.modules.chemical_classifier import get_npclassifier_classification

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'pathway_results': ['Shikimates and Phenylpropanoids'],
            'superclass_results': ['Flavonoids'],
            'class_results': ['Flavones'],
            'isglycoside': False
        }

        with patch('backend.modules.chemical_classifier.session.get', return_value=mock_response):
            result = get_npclassifier_classification('c1ccc(cc1)O')

        assert result is not None
        assert result['NP_Pathway'] == 'Shikimates and Phenylpropanoids'
        assert result['NP_Superclass'] == 'Flavonoids'
        assert result['NP_Class'] == 'Flavones'
        assert result['NP_isglycoside'] == False

    def test_glycoside_detection(self):
        """Test NPClassifier detects glycosides."""
        from backend.modules.chemical_classifier import get_npclassifier_classification

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'pathway_results': ['Carbohydrates'],
            'superclass_results': ['Glycosides'],
            'class_results': ['O-glycosides'],
            'isglycoside': True
        }

        with patch('backend.modules.chemical_classifier.session.get', return_value=mock_response):
            result = get_npclassifier_classification('glycoside_smiles')

        assert result is not None
        assert result['NP_isglycoside'] == True

    def test_empty_results(self):
        """Test NPClassifier handles empty results."""
        from backend.modules.chemical_classifier import get_npclassifier_classification

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'pathway_results': [],
            'superclass_results': [],
            'class_results': [],
            'isglycoside': False
        }

        with patch('backend.modules.chemical_classifier.session.get', return_value=mock_response):
            result = get_npclassifier_classification('CCO')

        assert result is not None
        assert result['NP_Pathway'] is None
        assert result['NP_Superclass'] is None

    def test_timeout_handling(self):
        """Test NPClassifier handles timeout gracefully."""
        from backend.modules.chemical_classifier import get_npclassifier_classification
        import requests

        with patch('backend.modules.chemical_classifier.session.get',
                   side_effect=requests.exceptions.Timeout()):
            result = get_npclassifier_classification('c1ccc(cc1)O')

        assert result is None


class TestExtractClassyFireFields:
    """Tests for extract_classyfire_fields function."""

    def test_full_extraction(self):
        """Test extraction of all ClassyFire fields."""
        from backend.modules.chemical_classifier import extract_classyfire_fields

        cf_data = {
            'kingdom': {'name': 'Organic compounds'},
            'superclass': {'name': 'Phenylpropanoids'},
            'class': {'name': 'Flavonoids', 'chemont_id': 'CHEMONTID:0000001'},
            'subclass': {'name': 'Flavones', 'chemont_id': 'CHEMONTID:0000002'},
            'direct_parent': {'name': 'Hydroxyflavones'},
            'molecular_framework': 'Aromatic',
            'description': 'A flavonoid'
        }

        result = extract_classyfire_fields(cf_data)

        assert result['Kingdom'] == 'Organic compounds'
        assert result['Superclass'] == 'Phenylpropanoids'
        assert result['Class'] == 'Flavonoids'
        assert result['Subclass'] == 'Flavones'
        assert result['Direct_Parent'] == 'Hydroxyflavones'
        assert result['Molecular_Framework'] == 'Aromatic'
        assert result['Description'] == 'A flavonoid'
        assert result['ChEMONT_ID_Class'] == 'CHEMONTID:0000001'
        assert result['ChEMONT_ID_Subclass'] == 'CHEMONTID:0000002'

    def test_none_input(self):
        """Test extraction with None input returns empty fields."""
        from backend.modules.chemical_classifier import extract_classyfire_fields

        result = extract_classyfire_fields(None)

        assert result['Kingdom'] == ''
        assert result['Class'] == ''
        assert len(result) == 9  # All 9 fields present

    def test_partial_data(self):
        """Test extraction with partial data."""
        from backend.modules.chemical_classifier import extract_classyfire_fields

        cf_data = {
            'kingdom': {'name': 'Organic compounds'},
            'class': {'name': 'Flavonoids'}
            # Missing superclass, subclass, etc.
        }

        result = extract_classyfire_fields(cf_data)

        assert result['Kingdom'] == 'Organic compounds'
        assert result['Class'] == 'Flavonoids'
        assert result['Superclass'] == ''


class TestGetCompleteClassification:
    """Tests for get_complete_classification function."""

    def test_combined_classification(self):
        """Test complete classification combines both sources."""
        from backend.modules.chemical_classifier import get_complete_classification

        cf_response = MagicMock()
        cf_response.status_code = 200
        cf_response.json.return_value = {
            'kingdom': {'name': 'Organic compounds'},
            'class': {'name': 'Flavonoids', 'chemont_id': 'CHEMONTID:0001'}
        }

        np_response = MagicMock()
        np_response.status_code = 200
        np_response.json.return_value = {
            'pathway_results': ['Shikimates'],
            'superclass_results': ['Flavonoids'],
            'class_results': ['Flavones'],
            'isglycoside': False
        }

        with patch('backend.modules.chemical_classifier.session.get',
                   side_effect=[cf_response, np_response]):
            result = get_complete_classification(
                smiles='c1ccc(cc1)O',
                inchikey='REFJWTPEDVJJIY-UHFFFAOYSA-N'
            )

        # ClassyFire fields
        assert result['Kingdom'] == 'Organic compounds'
        assert result['Class'] == 'Flavonoids'

        # NPClassifier fields
        assert result['NP_Pathway'] == 'Shikimates'
        assert result['NP_Class'] == 'Flavones'

    def test_classyfire_only(self):
        """Test classification when only ClassyFire succeeds."""
        from backend.modules.chemical_classifier import get_complete_classification

        cf_response = MagicMock()
        cf_response.status_code = 200
        cf_response.json.return_value = {
            'kingdom': {'name': 'Organic compounds'},
            'class': {'name': 'Alkanes'}
        }

        np_response = MagicMock()
        np_response.status_code = 500

        with patch('backend.modules.chemical_classifier.session.get',
                   side_effect=[cf_response, np_response]):
            result = get_complete_classification('CCCC', 'IJDNQMDRQITEOD-UHFFFAOYSA-N')

        assert result['Class'] == 'Alkanes'
        assert result['NP_Pathway'] == ''  # Empty from NPClassifier failure

    def test_neither_source(self):
        """Test classification when both sources fail."""
        from backend.modules.chemical_classifier import get_complete_classification

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch('backend.modules.chemical_classifier.session.get', return_value=mock_response):
            result = get_complete_classification('CCCC', 'UNKNOWN')

        assert result['Kingdom'] == ''
        assert result['Class'] == ''
        assert result['NP_Pathway'] == ''


class TestClassifyCompoundType:
    """Tests for classify_compound_type function."""

    def test_natural_product_by_np_pathway(self):
        """Test natural product detection via NPClassifier pathway."""
        from backend.modules.chemical_classifier import classify_compound_type

        classification = {
            'NP_Pathway': 'Terpenoids',
            'Class': 'Organic compounds'
        }

        result = classify_compound_type(classification)

        assert result == "Natural Product"

    def test_natural_product_by_classyfire(self):
        """Test natural product detection via ClassyFire keywords."""
        from backend.modules.chemical_classifier import classify_compound_type

        classification = {
            'NP_Pathway': '',
            'Class': 'Flavonoids'
        }

        result = classify_compound_type(classification)

        assert result == "Natural Product"

    def test_natural_product_alkaloid(self):
        """Test natural product detection for alkaloids."""
        from backend.modules.chemical_classifier import classify_compound_type

        classification = {
            'NP_Pathway': '',
            'Superclass': 'Alkaloids and derivatives'
        }

        result = classify_compound_type(classification)

        assert result == "Natural Product"

    def test_synthetic_compound(self):
        """Test synthetic compound detection."""
        from backend.modules.chemical_classifier import classify_compound_type

        classification = {
            'NP_Pathway': '',
            'Class': 'Organic acids',
            'Superclass': 'Organic acids and derivatives',
            'Subclass': 'Carboxylic acids'
        }

        result = classify_compound_type(classification)

        assert result == "Synthetic"

    def test_empty_classification(self):
        """Test compound type with empty classification."""
        from backend.modules.chemical_classifier import classify_compound_type

        classification = {
            'NP_Pathway': '',
            'Class': '',
            'Superclass': '',
            'Subclass': ''
        }

        result = classify_compound_type(classification)

        assert result == "Synthetic"  # Default to synthetic

    def test_none_values_handled(self):
        """Test that None values in classification are handled."""
        from backend.modules.chemical_classifier import classify_compound_type

        classification = {
            'NP_Pathway': None,
            'Class': None,
            'Superclass': None,
            'Subclass': None
        }

        result = classify_compound_type(classification)

        assert result == "Synthetic"


class TestGetClassificationSummary:
    """Tests for get_classification_summary function."""

    def test_summary_with_full_data(self):
        """Test summary generation with complete data."""
        from backend.modules.chemical_classifier import get_classification_summary

        classification = {
            'Kingdom': 'Organic compounds',
            'Superclass': 'Phenylpropanoids',
            'Class': 'Flavonoids',
            'Subclass': 'Flavones',
            'NP_Pathway': 'Shikimates',
            'NP_Superclass': 'Flavonoids',
            'NP_Class': 'Flavones',
            'NP_isglycoside': False,
            'Molecular_Framework': 'Aromatic'
        }

        summary = get_classification_summary(classification)

        assert 'Chemical Classification Summary' in summary
        assert 'Flavonoids' in summary
        assert 'Shikimates' in summary
        assert 'Natural Product' in summary

    def test_summary_with_glycoside(self):
        """Test summary shows glycoside information."""
        from backend.modules.chemical_classifier import get_classification_summary

        classification = {
            'Kingdom': '',
            'Superclass': '',
            'Class': '',
            'Subclass': '',
            'NP_Pathway': 'Carbohydrates',
            'NP_Superclass': 'Glycosides',
            'NP_Class': 'O-glycosides',
            'NP_isglycoside': True,
            'Molecular_Framework': ''
        }

        summary = get_classification_summary(classification)

        assert 'glycoside moiety' in summary

    def test_summary_no_classification(self):
        """Test summary when no classification available."""
        from backend.modules.chemical_classifier import get_classification_summary

        classification = {
            'Kingdom': '',
            'Superclass': '',
            'Class': '',
            'Subclass': '',
            'NP_Pathway': '',
            'NP_Superclass': '',
            'NP_Class': '',
            'NP_isglycoside': False,
            'Molecular_Framework': ''
        }

        summary = get_classification_summary(classification)

        assert 'No classification available' in summary


class TestLegacyGetClassification:
    """Tests for legacy get_classification function."""

    def test_legacy_function(self):
        """Test legacy function returns same as classyfire function."""
        from backend.modules.chemical_classifier import get_classification, get_classyfire_classification

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'class': {'name': 'Test'}}

        with patch('backend.modules.chemical_classifier.session.get', return_value=mock_response):
            result = get_classification('TEST-INCHIKEY')

        assert result is not None
        assert result['class']['name'] == 'Test'
