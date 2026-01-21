"""
Chemical Classification Module

Integrates ClassyFire and NPClassifier for comprehensive chemical taxonomy.

ClassyFire: General chemical taxonomy (Kingdom → Subclass)
NPClassifier: Natural product-specific classification (Pathway → Class)

Usage:
    from backend.modules.chemical_classifier import get_complete_classification

    classification = get_complete_classification(smiles="...", inchikey="...")
    print(classification['Class'])  # ClassyFire class
    print(classification['NP_Pathway'])  # NPClassifier pathway
"""

import logging
from typing import Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Configure session with retries
session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# API timeout
API_TIMEOUT = 10


def get_classyfire_classification(inchikey: str, max_retries: int = 2) -> Optional[Dict]:
    """
    Get comprehensive ClassyFire classification with retry logic.

    ClassyFire provides chemical taxonomy based on structural features.
    API: http://classyfire.wishartlab.com/

    Includes retry logic for transient failures (timeouts, connection errors).

    Args:
        inchikey: Standard InChIKey identifier
        max_retries: Maximum retry attempts for transient failures

    Returns:
        Dict with complete ClassyFire response, or None if failed

    Example:
        >>> data = get_classyfire_classification("REFJWTPEDVJJIY-UHFFFAOYSA-N")
        >>> print(data['kingdom']['name'])  # "Organic compounds"
    """
    import time

    url = f'http://classyfire.wishartlab.com/entities/{inchikey}.json'

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=API_TIMEOUT)

            if response.status_code == 200:
                return response.json()
            elif response.status_code in [500, 502, 503, 504]:
                # Server error - retry
                if attempt < max_retries - 1:
                    logger.warning(f"ClassyFire returned {response.status_code} (attempt {attempt + 1}), retrying...")
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    logger.error(f"ClassyFire returned {response.status_code} after {max_retries} attempts")
                    return None
            else:
                # 4xx errors - don't retry
                logger.warning(f"ClassyFire returned status {response.status_code} for {inchikey}")
                return None

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"ClassyFire timeout for {inchikey} (attempt {attempt + 1}), retrying...")
                time.sleep(1 * (attempt + 1))
            else:
                logger.error(f"ClassyFire timeout for {inchikey} after {max_retries} attempts")
                return None
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                logger.warning(f"ClassyFire connection error for {inchikey} (attempt {attempt + 1}), retrying...")
                time.sleep(1 * (attempt + 1))
            else:
                logger.error(f"ClassyFire connection error for {inchikey} after {max_retries} attempts")
                return None
        except Exception as e:
            logger.error(f"ClassyFire error for {inchikey}: {str(e)}")
            return None

    return None


def get_npclassifier_classification(smiles: str, max_retries: int = 2) -> Optional[Dict]:
    """
    Get NPClassifier classification for natural products with retry logic.

    NPClassifier is a deep learning tool specialized for natural product classification.
    API: https://npclassifier.ucsd.edu/

    Includes retry logic for transient failures (timeouts, connection errors).

    Args:
        smiles: SMILES string
        max_retries: Maximum retry attempts for transient failures

    Returns:
        Dict with NP_Pathway, NP_Superclass, NP_Class, NP_isglycoside, or None if failed

    Example:
        >>> data = get_npclassifier_classification("c1ccc(cc1)O")
        >>> print(data['NP_Pathway'])  # "Shikimates and Phenylpropanoids"
    """
    import time

    url = 'https://npclassifier.ucsd.edu/classify'
    params = {'smiles': smiles}
    smiles_preview = smiles[:50] + "..." if len(smiles) > 50 else smiles

    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=API_TIMEOUT)

            if response.status_code == 200:
                data = response.json()

                # Extract first prediction for each level
                # NPClassifier returns arrays of predictions
                return {
                    'NP_Pathway': data.get('pathway_results', [None])[0] if data.get('pathway_results') else None,
                    'NP_Superclass': data.get('superclass_results', [None])[0] if data.get('superclass_results') else None,
                    'NP_Class': data.get('class_results', [None])[0] if data.get('class_results') else None,
                    'NP_isglycoside': data.get('isglycoside', False)
                }
            elif response.status_code in [500, 502, 503, 504]:
                # Server error - retry
                if attempt < max_retries - 1:
                    logger.warning(f"NPClassifier returned {response.status_code} (attempt {attempt + 1}), retrying...")
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    logger.error(f"NPClassifier returned {response.status_code} after {max_retries} attempts")
                    return None
            else:
                # 4xx errors - don't retry
                logger.warning(f"NPClassifier returned status {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"NPClassifier timeout (attempt {attempt + 1}) for SMILES: {smiles_preview}")
                time.sleep(1 * (attempt + 1))
            else:
                logger.error(f"NPClassifier timeout for SMILES: {smiles_preview} after {max_retries} attempts")
                return None
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                logger.warning(f"NPClassifier connection error (attempt {attempt + 1}) for SMILES: {smiles_preview}")
                time.sleep(1 * (attempt + 1))
            else:
                logger.error(f"NPClassifier connection error for SMILES: {smiles_preview} after {max_retries} attempts")
                return None
        except Exception as e:
            logger.error(f"NPClassifier error for SMILES: {str(e)}")
            return None

    return None


def extract_classyfire_fields(cf_data: Optional[Dict]) -> Dict[str, str]:
    """
    Extract enhanced ClassyFire fields from API response.

    Extracts 9 fields including descriptions and ChEMONT IDs.

    Args:
        cf_data: Full ClassyFire API response

    Returns:
        Dict with Kingdom, Superclass, Class, Subclass, Direct_Parent,
        Molecular_Framework, Description, ChEMONT_ID_Class, ChEMONT_ID_Subclass
    """
    if cf_data is None:
        return {
            'Kingdom': '',
            'Superclass': '',
            'Class': '',
            'Subclass': '',
            'Direct_Parent': '',
            'Molecular_Framework': '',
            'Description': '',
            'ChEMONT_ID_Class': '',
            'ChEMONT_ID_Subclass': ''
        }

    try:
        # Helper to safely extract nested values (handles None values)
        def safe_get(d, key1, key2='name', default=''):
            """Safely get nested dict value, handling None at any level."""
            val = d.get(key1) if d else None
            if val is None:
                return default
            if isinstance(val, dict):
                return val.get(key2, default) or default
            return str(val) if val else default

        return {
            # Standard taxonomy levels
            'Kingdom': safe_get(cf_data, 'kingdom', 'name'),
            'Superclass': safe_get(cf_data, 'superclass', 'name'),
            'Class': safe_get(cf_data, 'class', 'name'),
            'Subclass': safe_get(cf_data, 'subclass', 'name'),

            # Enhanced fields
            'Direct_Parent': safe_get(cf_data, 'direct_parent', 'name'),
            'Molecular_Framework': cf_data.get('molecular_framework', '') or '',
            'Description': cf_data.get('description', '') or '',

            # ChEMONT ontology IDs
            'ChEMONT_ID_Class': safe_get(cf_data, 'class', 'chemont_id'),
            'ChEMONT_ID_Subclass': safe_get(cf_data, 'subclass', 'chemont_id')
        }
    except Exception as e:
        logger.error(f"Error extracting ClassyFire fields: {str(e)}")
        return {
            'Kingdom': '',
            'Superclass': '',
            'Class': '',
            'Subclass': '',
            'Direct_Parent': '',
            'Molecular_Framework': '',
            'Description': '',
            'ChEMONT_ID_Class': '',
            'ChEMONT_ID_Subclass': ''
        }


def get_complete_classification(smiles: str, inchikey: str) -> Dict[str, str]:
    """
    Get complete chemical classification from both ClassyFire and NPClassifier.

    This is the main function to use for comprehensive chemical taxonomy.

    Args:
        smiles: SMILES string
        inchikey: InChIKey

    Returns:
        Dict with all classification fields (13-14 total):
            - ClassyFire: Kingdom, Superclass, Class, Subclass, Direct_Parent,
                         Molecular_Framework, Description, ChEMONT_ID_Class, ChEMONT_ID_Subclass
            - NPClassifier: NP_Pathway, NP_Superclass, NP_Class, NP_isglycoside

    Example:
        >>> classification = get_complete_classification(
        ...     smiles="C1=CC(=C(C=C1O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O",
        ...     inchikey="REFJWTPEDVJJIY-UHFFFAOYSA-N"
        ... )
        >>> print(f"Class: {classification['Class']}")  # "Flavonoids"
        >>> print(f"Pathway: {classification['NP_Pathway']}")  # "Shikimates and Phenylpropanoids"
    """
    # Initialize with empty fields
    classification = {
        # ClassyFire fields
        'Kingdom': '',
        'Superclass': '',
        'Class': '',
        'Subclass': '',
        'Direct_Parent': '',
        'Molecular_Framework': '',
        'Description': '',
        'ChEMONT_ID_Class': '',
        'ChEMONT_ID_Subclass': '',

        # NPClassifier fields
        'NP_Pathway': '',
        'NP_Superclass': '',
        'NP_Class': '',
        'NP_isglycoside': False
    }

    # Get ClassyFire data
    cf_data = get_classyfire_classification(inchikey)
    if cf_data:
        cf_fields = extract_classyfire_fields(cf_data)
        classification.update(cf_fields)
        logger.info(f"ClassyFire classification obtained: {cf_fields.get('Class', 'Unknown')}")
    else:
        logger.warning(f"No ClassyFire data for {inchikey}")

    # Get NPClassifier data
    np_data = get_npclassifier_classification(smiles)
    if np_data:
        classification.update(np_data)
        logger.info(f"NPClassifier classification obtained: {np_data.get('NP_Pathway', 'Unknown')}")
    else:
        logger.warning(f"No NPClassifier data for SMILES")

    return classification


def classify_compound_type(classification: Dict) -> str:
    """
    Determine if compound is natural product, synthetic, or semi-synthetic.

    Uses both ClassyFire and NPClassifier data to infer compound origin.

    Args:
        classification: Classification dict from get_complete_classification()

    Returns:
        str: "Natural Product", "Synthetic", or "Semi-Synthetic"

    Example:
        >>> classification = get_complete_classification(smiles="...", inchikey="...")
        >>> compound_type = classify_compound_type(classification)
        >>> print(compound_type)  # "Natural Product"
    """
    # Has NPClassifier pathway → likely natural product
    if classification.get('NP_Pathway'):
        return "Natural Product"

    # Check ClassyFire for natural product indicators
    np_keywords = [
        'alkaloid', 'terpenoid', 'flavonoid', 'polyketide',
        'phenylpropanoid', 'steroid', 'glycoside', 'saponin',
        'tannin', 'coumarin', 'quinone', 'lignan'
    ]

    # Check all ClassyFire levels
    for field in ['Superclass', 'Class', 'Subclass', 'Direct_Parent']:
        value = classification.get(field) or ''  # Guard against None
        value = value.lower()
        if any(keyword in value for keyword in np_keywords):
            return "Natural Product"

    # Default to synthetic
    return "Synthetic"


def get_classification_summary(classification: Dict) -> str:
    """
    Generate human-readable summary of classification.

    Args:
        classification: Classification dict from get_complete_classification()

    Returns:
        str: Multi-line summary text

    Example:
        >>> classification = get_complete_classification(smiles="...", inchikey="...")
        >>> print(get_classification_summary(classification))
    """
    lines = []

    lines.append("Chemical Classification Summary")
    lines.append("-" * 50)

    # ClassyFire taxonomy
    if classification.get('Class'):
        lines.append(f"ClassyFire: {classification['Kingdom']} > {classification['Superclass']} > {classification['Class']} > {classification['Subclass']}")
    else:
        lines.append("ClassyFire: No classification available")

    # NPClassifier (if natural product)
    if classification.get('NP_Pathway'):
        lines.append(f"NPClassifier: {classification['NP_Pathway']} > {classification['NP_Superclass']} > {classification['NP_Class']}")
        if classification.get('NP_isglycoside'):
            lines.append("  • Contains glycoside moiety")

    # Compound type
    compound_type = classify_compound_type(classification)
    lines.append(f"Compound Type: {compound_type}")

    # Molecular framework
    if classification.get('Molecular_Framework'):
        lines.append(f"Molecular Framework: {classification['Molecular_Framework']}")

    return "\n".join(lines)


# For backward compatibility with existing code
def get_classification(inchikey: str) -> Optional[Dict]:
    """
    Legacy function for backward compatibility with existing api_client.py.

    This maintains the same interface as the old get_classification() function
    but returns the enhanced data.

    Args:
        inchikey: InChIKey

    Returns:
        Full ClassyFire API response (same as before)
    """
    return get_classyfire_classification(inchikey)
