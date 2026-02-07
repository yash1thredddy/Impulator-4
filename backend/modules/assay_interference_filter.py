"""
Assay Interference Detection Module

Detects compounds with known assay interference mechanisms using
ONLY peer-reviewed, industry-standard methods from published literature.

Each detection category uses SPECIFIC published SMARTS patterns or methods,
NOT generic mappings to catch-all filters.

=============================================================================
DETECTION METHODS AND PEER-REVIEWED REFERENCES
=============================================================================

1. PAINS (Pan-Assay Interference Substructures)
   - Method: RDKit FilterCatalog.PAINS (480 SMARTS patterns)
   - Reference: Baell JB, Holloway GA. J Med Chem. 2010;53(7):2719-2740.
   - DOI: 10.1021/jm901137j

2. Aggregator (Colloidal Aggregation Risk)
   - Method: Published Shoichet laboratory heuristics
   - References:
     * Irwin JJ, et al. J Med Chem. 2015;58(17):7076-7087. DOI: 10.1021/acs.jmedchem.5b01105
     * McGovern SL, et al. J Med Chem. 2002;45(8):1712-1722. DOI: 10.1021/jm010533y

3. Thiol-Reactive (Electrophilic compounds)
   - Method: SMARTS patterns for electrophilic chemotypes causing HTS interference
   - Reference: Dahlin JL, et al. J Med Chem. 2015;58(5):2091-2113.
   - DOI: 10.1021/jm5019093

4. Redox-Active (Redox cycling compounds)
   - Method: SMARTS patterns for redox-active functional groups
   - References:
     * Baell JB, Holloway GA (2010) - quinones in PAINS
     * Proj M, et al. Drug Discov Today. 2022;27(6):1733-1742.
   - DOI: 10.1016/j.drudis.2022.03.008

5. Autofluorescent (Fluorescence interference)
   - Method: SMARTS patterns for known fluorophore scaffolds
   - Reference: Su BH, et al. J Chem Inf Model. 2015;55(2):434-445.
   - DOI: 10.1021/ci5007432

6. BRENK (Unwanted substructures)
   - Method: RDKit FilterCatalog.BRENK (104 patterns)
   - Reference: Brenk R, et al. ChemMedChem. 2008;3(3):435-444.
   - DOI: 10.1002/cmdc.200700139

7. NIH (Problematic functional groups)
   - Method: RDKit FilterCatalog.NIH
   - Reference: Jadhav A, et al. J Med Chem. 2010;53(1):37-51.
   - DOI: 10.1021/jm901070c

=============================================================================
KEY PRINCIPLE: Each flag uses SPECIFIC detection, not generic catch-all
=============================================================================

| Flag           | Detection Method                | NOT Using              |
|----------------|--------------------------------|------------------------|
| PAINS          | RDKit PAINS FilterCatalog      | -                      |
| Aggregator     | Shoichet heuristics            | -                      |
| Thiol          | Dahlin HTS electrophile SMARTS | Generic BRENK filter   |
| Redox          | Quinone/catechol SMARTS        | Generic BRENK filter   |
| Fluorescence   | Fluorophore scaffold SMARTS    | Generic NIH filter     |

IMPORTANT: These flags are displayed as ORTHOGONAL INFORMATION alongside IMP scores.
They do NOT penalize the IMP score. Compounds with interference flags but strong
PDB structural evidence (e.g., quercetin) are retained as valid IMPs exhibiting genuine
polypharmacology rather than assay artifacts.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from rdkit import Chem
from rdkit.Chem import FilterCatalog, rdMolDescriptors, Descriptors

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLISHED SMARTS PATTERNS FOR THIOL-REACTIVE COMPOUNDS
# =============================================================================
# Reference: Dahlin JL, et al. (2015) J Med Chem. 58:2091-2113
# DOI: 10.1021/jm5019093
# Additional: NCBI Assay Guidance Manual NBK326709 - thiol-reactive groups

THIOL_REACTIVE_SMARTS = {
    # Michael Acceptors (α,β-unsaturated carbonyls)
    'michael_acceptor': '[C;$(C=C)]-[C;$(C=O)]',
    'acrylamide': 'C=CC(=O)N',
    'acrylate': 'C=CC(=O)O',
    'enone': 'C=CC(=O)[#6]',
    'maleimide': 'O=C1C=CC(=O)N1',

    # Acylating Agents
    'acyl_halide': 'C(=O)[F,Cl,Br,I]',
    'anhydride': 'C(=O)OC(=O)',
    'activated_ester': '[C;$(C(=O)O)][F,Cl,Br,I,$(OS(=O)(=O))]',

    # SN2 Electrophiles (Alkylating agents)
    'epoxide': 'C1OC1',
    'aziridine': 'C1NC1',

    # Schiff Base Formers
    'aldehyde': '[CH1](=O)',

    # Isocyanates/Isothiocyanates
    'isocyanate': 'N=C=O',
    'isothiocyanate': 'N=C=S',

    # Additional reactive groups
    'vinyl_sulfone': 'C=CS(=O)(=O)',
    'sulfonyl_fluoride': 'S(=O)(=O)F',
}

# =============================================================================
# PUBLISHED SMARTS PATTERNS FOR REDOX-ACTIVE COMPOUNDS
# =============================================================================
# References:
# - Baell JB, Holloway GA (2010) J Med Chem - quinones are major PAINS
# - NCBI Assay Guidance Manual NBK326709 - HRP-PR assay validation compounds
# - Proj M et al. (2022) Drug Discov Today. DOI: 10.1016/j.drudis.2022.03.008

REDOX_ACTIVE_SMARTS = {
    # Quinones (most common redox-cycling compounds)
    'para_quinone': 'O=C1C=CC(=O)C=C1',
    'ortho_quinone': 'O=C1C(=O)C=CC=C1',
    'naphthoquinone': 'O=C1C=CC2=CC=CC=C2C1=O',
    'anthraquinone': 'O=C1c2ccccc2C(=O)c3ccccc13',

    # Catechols (oxidize to quinones in air)
    'catechol': 'c1ccc(O)c(O)c1',
    'catechol_substituted': '[cH]1[cH][cH]c(O)c(O)[cH]1',

    # Hydroquinones (redox pair with quinones)
    'hydroquinone': 'Oc1ccc(O)cc1',

    # Other redox-active groups
    'hydroxylamine': '[NH2]O',
    'nitroso': '[N;X2]=O',
    'nitro_aromatic': 'c[N+](=O)[O-]',
}

# =============================================================================
# PUBLISHED SMARTS PATTERNS FOR AUTOFLUORESCENT COMPOUNDS
# =============================================================================
# Reference: Su BH et al. (2015) J Chem Inf Model. 55(2):434-445
# DOI: 10.1021/ci5007432

FLUORESCENT_SMARTS = {
    # Coumarins (excitation ~340-405nm)
    'coumarin': 'O=c1ccc2ccccc2o1',
    'coumarin_keto': 'O=C1C=Cc2ccccc2O1',
    'coumarin_7amino': 'Nc1ccc2ccc(=O)oc2c1',

    # Xanthenes (fluorescein, rhodamine scaffolds)
    'xanthene': 'c1ccc2c(c1)Cc1ccccc1O2',
    'fluorescein_core': 'O=C1OC2(c3ccc(O)cc3Oc3cc(O)ccc23)c2ccccc12',
    'rhodamine_core': 'c1cc2c(cc1)C(=C1C=CC(=[NH2])C=C1)c1cc(N)ccc1O2',

    # Polycyclic aromatic hydrocarbons (PAHs)
    'naphthalene': 'c1ccc2ccccc2c1',
    'anthracene': 'c1ccc2cc3ccccc3cc2c1',
    'pyrene': 'c1cc2ccc3cccc4ccc(c1)c2c34',

    # Stilbenes
    'stilbene': 'c1ccccc1C=Cc1ccccc1',

    # Flavonoids (common natural product fluorophores)
    'flavone': 'O=c1cc(-c2ccccc2)oc2ccccc12',
    'flavonol': 'O=c1c(O)c(-c2ccccc2)oc2ccccc12',

    # Acridines
    'acridine': 'c1ccc2nc3ccccc3cc2c1',
}


# =============================================================================
# Pre-compile SMARTS patterns for performance (lazy-initialized cache)
# =============================================================================

_compiled_patterns: Dict[str, Dict[str, Chem.Mol]] = {}


def _get_compiled_patterns(category: str) -> Dict[str, Chem.Mol]:
    """Get compiled SMARTS patterns for a category (cached).

    Args:
        category: One of 'thiol', 'redox', 'fluorescent'

    Returns:
        Dictionary mapping pattern names to compiled Mol objects
    """
    if category not in _compiled_patterns:
        pattern_dict = {
            'thiol': THIOL_REACTIVE_SMARTS,
            'redox': REDOX_ACTIVE_SMARTS,
            'fluorescent': FLUORESCENT_SMARTS,
        }.get(category, {})

        compiled = {}
        for name, smarts in pattern_dict.items():
            try:
                mol = Chem.MolFromSmarts(smarts)
                if mol is not None:
                    compiled[name] = mol
                else:
                    logger.warning(f"Failed to compile SMARTS for {name}: {smarts}")
            except Exception as e:
                logger.warning(f"Error compiling SMARTS {name}: {e}")

        _compiled_patterns[category] = compiled

    return _compiled_patterns[category]


# =============================================================================
# FilterCatalog cache (lazy-initialized)
# =============================================================================

_filter_catalogs: Dict[str, FilterCatalog.FilterCatalog] = {}


def _get_filter_catalog(catalog_type: str) -> FilterCatalog.FilterCatalog:
    """Get or create a cached FilterCatalog instance.

    Args:
        catalog_type: One of 'PAINS', 'BRENK', 'NIH', 'ZINC', 'ALL'

    Returns:
        FilterCatalog instance
    """
    if catalog_type not in _filter_catalogs:
        params = FilterCatalog.FilterCatalogParams()

        catalog_map = {
            'PAINS': FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS,
            'BRENK': FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
            'NIH': FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
            'ZINC': FilterCatalog.FilterCatalogParams.FilterCatalogs.ZINC,
            'ALL': FilterCatalog.FilterCatalogParams.FilterCatalogs.ALL,
        }

        if catalog_type in catalog_map:
            params.AddCatalog(catalog_map[catalog_type])
        else:
            raise ValueError(f"Unknown catalog type: {catalog_type}")

        _filter_catalogs[catalog_type] = FilterCatalog.FilterCatalog(params)

    return _filter_catalogs[catalog_type]


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class InterferenceFlags:
    """Container for all assay interference flags.

    Each flag uses SPECIFIC peer-reviewed detection methods:
    - PAINS: RDKit FilterCatalog.PAINS (Baell & Holloway 2010)
    - Aggregator: Shoichet lab heuristics (Irwin et al. 2015)
    - Thiol: HTS electrophile SMARTS (Dahlin et al. 2015, NCBI AGM)
    - Redox: Quinone/catechol SMARTS (Baell 2010, NCBI Manual)
    - Fluorescence: Fluorophore SMARTS (Su et al. 2015)
    - BRENK: RDKit FilterCatalog.BRENK (Brenk et al. 2008)
    - NIH: RDKit FilterCatalog.NIH (Jadhav et al. 2009)
    """
    # Primary flags with specific detection
    pains: bool = False
    aggregator: bool = False
    thiol: bool = False
    redox: bool = False
    fluorescence: bool = False
    brenk: bool = False
    nih: bool = False

    # Details for each flag
    pains_details: List[str] = field(default_factory=list)
    aggregator_reason: str = ""
    thiol_details: List[str] = field(default_factory=list)
    redox_details: List[str] = field(default_factory=list)
    fluorescence_details: List[str] = field(default_factory=list)
    brenk_details: List[str] = field(default_factory=list)
    nih_details: List[str] = field(default_factory=list)

    @property
    def total_flags(self) -> int:
        """Count total number of flags raised."""
        return sum([
            self.pains,
            self.aggregator,
            self.thiol,
            self.redox,
            self.fluorescence,
            self.brenk,
            self.nih,
        ])

    @property
    def is_clean(self) -> bool:
        """Check if compound has no interference flags."""
        return self.total_flags == 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary with integer flags (0/1) for DataFrame usage."""
        return {
            'PAINS': 1 if self.pains else 0,
            'Aggregator': 1 if self.aggregator else 0,
            'Thiol': 1 if self.thiol else 0,
            'Redox': 1 if self.redox else 0,
            'Fluorescence': 1 if self.fluorescence else 0,
            'BRENK': 1 if self.brenk else 0,
            'NIH': 1 if self.nih else 0,
        }

    def to_detailed_dict(self) -> Dict[str, Any]:
        """Convert to detailed dictionary including reasons.

        All detail columns use consistent '_Details' suffix.
        """
        return {
            'PAINS': 1 if self.pains else 0,
            'PAINS_Details': ', '.join(self.pains_details) if self.pains_details else '',
            'Aggregator': 1 if self.aggregator else 0,
            'Aggregator_Details': self.aggregator_reason,
            'Thiol': 1 if self.thiol else 0,
            'Thiol_Details': ', '.join(self.thiol_details) if self.thiol_details else '',
            'Redox': 1 if self.redox else 0,
            'Redox_Details': ', '.join(self.redox_details) if self.redox_details else '',
            'Fluorescence': 1 if self.fluorescence else 0,
            'Fluorescence_Details': ', '.join(self.fluorescence_details) if self.fluorescence_details else '',
            'BRENK': 1 if self.brenk else 0,
            'BRENK_Details': ', '.join(self.brenk_details) if self.brenk_details else '',
            'NIH': 1 if self.nih else 0,
            'NIH_Details': ', '.join(self.nih_details) if self.nih_details else '',
            'Total_Flags': self.total_flags,
        }


# =============================================================================
# Aggregator Detection Thresholds
# =============================================================================
# Published heuristics from Shoichet Laboratory
# Reference: Irwin et al. (2015) J. Med. Chem. 58, 7076-7087

AGGREGATOR_MIN_AROMATIC_RINGS = 3
AGGREGATOR_MIN_MW = 300
AGGREGATOR_MAX_ROTATABLE_BONDS = 2
AGGREGATOR_MIN_LOGP = 3


# =============================================================================
# PAINS Detection - RDKit FilterCatalog.PAINS
# =============================================================================

def check_pains_violations(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """Check for PAINS using RDKit's built-in FilterCatalog.PAINS.

    Uses the peer-reviewed PAINS filter set (480 patterns) as published
    by Baell & Holloway (2010) and implemented in RDKit.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (has_pains, list_of_pains_names)
    """
    if mol is None:
        return False, []

    try:
        catalog = _get_filter_catalog('PAINS')
        pains_names = []

        matches = catalog.GetMatches(mol)
        for match in matches:
            desc = match.GetDescription()
            if desc not in pains_names:
                pains_names.append(desc)

        has_pains = len(pains_names) > 0

        if has_pains:
            logger.debug(f"PAINS violations detected: {', '.join(pains_names)}")

        return has_pains, pains_names

    except Exception as e:
        logger.warning(f"Error in PAINS detection: {e}")
        return False, []


# =============================================================================
# BRENK Filter Detection - RDKit FilterCatalog.BRENK
# =============================================================================

def check_brenk_alerts(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """Check for BRENK filter alerts using RDKit's built-in FilterCatalog.BRENK.

    BRENK filter contains 104 unwanted substructures as published by
    Brenk et al. (2008). Includes reactive groups, toxic groups,
    and groups with unfavorable pharmacokinetics.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (has_alerts, list_of_alert_names)
    """
    if mol is None:
        return False, []

    try:
        catalog = _get_filter_catalog('BRENK')
        alert_names = []

        matches = catalog.GetMatches(mol)
        for match in matches:
            desc = match.GetDescription()
            if desc not in alert_names:
                alert_names.append(desc)

        has_alerts = len(alert_names) > 0

        if has_alerts:
            logger.debug(f"BRENK alerts detected: {', '.join(alert_names)}")

        return has_alerts, alert_names

    except Exception as e:
        logger.warning(f"Error in BRENK detection: {e}")
        return False, []


# =============================================================================
# NIH Filter Detection - RDKit FilterCatalog.NIH
# =============================================================================

def check_nih_alerts(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """Check for NIH filter alerts using RDKit's built-in FilterCatalog.NIH.

    NIH filter identifies compounds with problematic functional groups
    as published by Jadhav et al. (2009).

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (has_alerts, list_of_alert_names)
    """
    if mol is None:
        return False, []

    try:
        catalog = _get_filter_catalog('NIH')
        alert_names = []

        matches = catalog.GetMatches(mol)
        for match in matches:
            desc = match.GetDescription()
            if desc not in alert_names:
                alert_names.append(desc)

        has_alerts = len(alert_names) > 0

        if has_alerts:
            logger.debug(f"NIH alerts detected: {', '.join(alert_names)}")

        return has_alerts, alert_names

    except Exception as e:
        logger.warning(f"Error in NIH detection: {e}")
        return False, []


# =============================================================================
# Aggregation Risk Detection - Shoichet Lab Heuristics
# =============================================================================

def check_aggregator_risk(mol: Chem.Mol) -> Tuple[bool, str]:
    """Detect aggregation risk using published Shoichet laboratory heuristics.

    Risk factors (ALL must be met):
    - >= 3 aromatic rings (flat, hydrophobic)
    - > 300 Da molecular weight
    - <= 2 rotatable bonds (rigid)
    - > 3 LogP (lipophilic)

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (is_aggregator_risk, reason)
    """
    if mol is None:
        return False, ""

    try:
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)

        risk_factors = []
        criteria_met = 0

        if num_aromatic_rings >= AGGREGATOR_MIN_AROMATIC_RINGS:
            risk_factors.append(f"{num_aromatic_rings} aromatic rings (>=3)")
            criteria_met += 1

        if mw > AGGREGATOR_MIN_MW:
            risk_factors.append(f"MW={mw:.1f} (>300)")
            criteria_met += 1

        if num_rotatable_bonds <= AGGREGATOR_MAX_ROTATABLE_BONDS:
            risk_factors.append(f"{num_rotatable_bonds} rotatable bonds (<=2)")
            criteria_met += 1

        if logp > AGGREGATOR_MIN_LOGP:
            risk_factors.append(f"LogP={logp:.2f} (>3)")
            criteria_met += 1

        is_risk = criteria_met >= 4
        reason = "; ".join(risk_factors) if is_risk else ""

        if is_risk:
            logger.debug(f"Aggregator risk detected: {reason}")

        return is_risk, reason

    except Exception as e:
        logger.warning(f"Error in aggregator detection: {e}")
        return False, ""


# =============================================================================
# THIOL-REACTIVE Detection - Dahlin HTS electrophile SMARTS
# =============================================================================

def check_thiol_reactive(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """Check for thiol-reactive electrophiles using published HTS interference patterns.

    Detects electrophilic chemotypes causing HTS interference:
    - Michael acceptors (alpha,beta-unsaturated carbonyls)
    - Acylating agents (acyl halides, anhydrides)
    - SN2 electrophiles (epoxides, aziridines)
    - Schiff base formers (aldehydes)
    - Isocyanates/isothiocyanates
    - Vinyl sulfones, sulfonyl fluorides

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (has_thiol_reactive, list_of_matched_patterns)
    """
    if mol is None:
        return False, []

    try:
        patterns = _get_compiled_patterns('thiol')
        matched_names = []

        for name, pattern in patterns.items():
            if mol.HasSubstructMatch(pattern):
                matched_names.append(name)

        has_matches = len(matched_names) > 0

        if has_matches:
            logger.debug(f"Thiol-reactive patterns detected: {', '.join(matched_names)}")

        return has_matches, matched_names

    except Exception as e:
        logger.warning(f"Error in thiol-reactive detection: {e}")
        return False, []


# =============================================================================
# REDOX-ACTIVE Detection - Quinone/Catechol SMARTS
# =============================================================================

def check_redox_active(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """Check for redox-active compounds using published SMARTS patterns.

    Detects compounds that generate H2O2/ROS via redox cycling:
    - Quinones (p-quinone, o-quinone, naphthoquinone, anthraquinone)
    - Catechols (oxidize to quinones)
    - Hydroquinones
    - Hydroxylamines, nitroso, nitroaromatics

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (has_redox_active, list_of_matched_patterns)
    """
    if mol is None:
        return False, []

    try:
        patterns = _get_compiled_patterns('redox')
        matched_names = []

        for name, pattern in patterns.items():
            if mol.HasSubstructMatch(pattern):
                matched_names.append(name)

        has_matches = len(matched_names) > 0

        if has_matches:
            logger.debug(f"Redox-active patterns detected: {', '.join(matched_names)}")

        return has_matches, matched_names

    except Exception as e:
        logger.warning(f"Error in redox-active detection: {e}")
        return False, []


# =============================================================================
# AUTOFLUORESCENCE Detection - Fluorophore SMARTS
# =============================================================================

def check_fluorescence_interference(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """Check for autofluorescent scaffolds using published SMARTS patterns.

    Detects common fluorophore scaffolds:
    - Coumarins (340-405nm excitation)
    - Xanthenes (fluorescein, rhodamine)
    - PAHs (naphthalene, anthracene, pyrene)
    - Stilbenes, flavonoids, acridines

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple of (has_fluorescence, list_of_matched_scaffolds)
    """
    if mol is None:
        return False, []

    try:
        patterns = _get_compiled_patterns('fluorescent')
        matched_names = []

        for name, pattern in patterns.items():
            if mol.HasSubstructMatch(pattern):
                matched_names.append(name)

        has_matches = len(matched_names) > 0

        if has_matches:
            logger.debug(f"Fluorescent scaffolds detected: {', '.join(matched_names)}")

        return has_matches, matched_names

    except Exception as e:
        logger.warning(f"Error in fluorescence detection: {e}")
        return False, []


# =============================================================================
# Main Interface Functions
# =============================================================================

def calculate_interference_flags(mol: Chem.Mol) -> InterferenceFlags:
    """Calculate all assay interference flags for a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        InterferenceFlags dataclass with all detection results
    """
    flags = InterferenceFlags()

    if mol is None:
        return flags

    try:
        flags.pains, flags.pains_details = check_pains_violations(mol)
        flags.aggregator, flags.aggregator_reason = check_aggregator_risk(mol)
        flags.thiol, flags.thiol_details = check_thiol_reactive(mol)
        flags.redox, flags.redox_details = check_redox_active(mol)
        flags.fluorescence, flags.fluorescence_details = check_fluorescence_interference(mol)
        flags.brenk, flags.brenk_details = check_brenk_alerts(mol)
        flags.nih, flags.nih_details = check_nih_alerts(mol)

    except Exception as e:
        logger.error(f"Error calculating interference flags: {e}")

    return flags


def get_interference_flags_from_smiles(smiles: str) -> InterferenceFlags:
    """Calculate all assay interference flags from a SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        InterferenceFlags dataclass with all detection results
    """
    if not smiles or smiles == 'N/A':
        return InterferenceFlags()

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES (could not parse): {smiles}")
            return InterferenceFlags()
        return calculate_interference_flags(mol)
    except Exception as e:
        logger.error(f"Error processing SMILES '{smiles}': {e}")
        return InterferenceFlags()


def get_interference_summary(flags: InterferenceFlags) -> Dict[str, Any]:
    """Get a summary of interference flags for display.

    Args:
        flags: InterferenceFlags object

    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_flags': flags.total_flags,
        'is_clean': flags.is_clean,
        'flags': flags.to_dict(),
        'details': flags.to_detailed_dict(),
    }


def get_all_filter_matches(mol: Chem.Mol) -> Dict[str, List[str]]:
    """Get all matches from all available RDKit FilterCatalogs for a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Dictionary mapping catalog name to list of match descriptions
    """
    if mol is None:
        return {}

    results = {}
    for catalog_name in ['PAINS', 'BRENK', 'NIH', 'ZINC']:
        try:
            catalog = _get_filter_catalog(catalog_name)
            matches = catalog.GetMatches(mol)
            results[catalog_name] = [m.GetDescription() for m in matches]
        except Exception as e:
            logger.warning(f"Error getting {catalog_name} matches: {e}")
            results[catalog_name] = []

    return results


# =============================================================================
# Flag descriptions for UI display (with full citations)
# =============================================================================

FLAG_DESCRIPTIONS = {
    'PAINS': {
        'name': 'PAINS',
        'full_name': 'Pan-Assay Interference Substructures',
        'description': (
            'Pan-Assay Interference Substructures - compounds that show activity '
            'in many assays due to interference rather than genuine binding.'
        ),
        'reference': 'Baell JB, Holloway GA. J Med Chem. 2010;53(7):2719-2740',
        'doi': '10.1021/jm901137j',
        'source': 'RDKit FilterCatalog.PAINS',
        'method': 'Substructure matching against 480 PAINS patterns from published filter set',
        'color': '#ff6b6b',
    },
    'Aggregator': {
        'name': 'Aggregator',
        'full_name': 'Colloidal Aggregation Risk',
        'description': (
            'Compounds that may form colloidal aggregates in aqueous solution, '
            'leading to non-specific protein inhibition.'
        ),
        'reference': 'Irwin JJ, et al. J Med Chem. 2015;58(17):7076-7087',
        'doi': '10.1021/acs.jmedchem.5b01105',
        'source': 'Shoichet Lab heuristics (https://advisor.bkslab.org/)',
        'method': (
            'Published heuristics: >=3 aromatic rings, >300 Da MW, '
            '<=2 rotatable bonds, >3 LogP (all criteria must be met)'
        ),
        'color': '#ffa94d',
    },
    'BRENK': {
        'name': 'BRENK',
        'full_name': 'BRENK Unwanted Substructures',
        'description': (
            '104 unwanted substructures including reactive groups (Michael acceptors, '
            'aldehydes, epoxides), toxic groups, and groups with unfavorable pharmacokinetics.'
        ),
        'reference': 'Brenk R, et al. ChemMedChem. 2008;3(3):435-444',
        'doi': '10.1002/cmdc.200700139',
        'source': 'RDKit FilterCatalog.BRENK',
        'method': 'Substructure matching against 104 BRENK patterns from published filter set',
        'color': '#ffd43b',
    },
    'NIH': {
        'name': 'NIH',
        'full_name': 'NIH Problematic Functional Groups',
        'description': (
            'Compounds with problematic functional groups identified in NIH screening. '
            'Includes aggregation, autofluorescence, and reactivity artifacts.'
        ),
        'reference': 'Jadhav A, et al. J Med Chem. 2010;53(1):37-51',
        'doi': '10.1021/jm901070c',
        'source': 'RDKit FilterCatalog.NIH',
        'method': 'Substructure matching against NIH patterns from published filter set',
        'color': '#74c0fc',
    },
    'Redox': {
        'name': 'Redox',
        'full_name': 'Redox-Active Compounds',
        'description': (
            'Compounds containing redox-active functional groups that generate H2O2/ROS '
            'via redox cycling, causing false positives in HTS assays.'
        ),
        'reference': 'Proj M, et al. Drug Discov Today. 2022;27(6):1733-1742',
        'doi': '10.1016/j.drudis.2022.03.008',
        'additional_refs': [
            'Baell JB, Holloway GA. J Med Chem. 2010;53(7):2719-2740 (quinones as PAINS)',
            'NCBI Assay Guidance Manual NBK326709 (HRP-PR assay compounds)',
        ],
        'source': 'SMARTS patterns: quinones, catechols, hydroquinones, nitroaromatics',
        'method': (
            'Substructure matching against 10 validated SMARTS patterns: '
            'para-quinone, ortho-quinone, naphthoquinone, anthraquinone, catechol, '
            'catechol_substituted, hydroquinone, hydroxylamine, nitroso, nitroaromatic'
        ),
        'color': '#ffd43b',
    },
    'Fluorescence': {
        'name': 'Fluorescence',
        'full_name': 'Autofluorescent Scaffolds',
        'description': (
            'Compounds containing fluorophore scaffolds that exhibit autofluorescence, '
            'interfering with fluorescence-based HTS assays.'
        ),
        'reference': 'Su BH, et al. J Chem Inf Model. 2015;55(2):434-445',
        'doi': '10.1021/ci5007432',
        'source': 'SMARTS patterns: coumarins, xanthenes, PAHs, stilbenes, flavonoids, acridines',
        'method': (
            'Substructure matching against 13 validated SMARTS patterns: '
            'coumarin (2 forms), 7-aminocoumarin, xanthene, fluorescein, rhodamine, naphthalene, '
            'anthracene, pyrene, stilbene, flavone, flavonol, acridine'
        ),
        'color': '#74c0fc',
    },
    'Thiol': {
        'name': 'Thiol',
        'full_name': 'Thiol-Reactive Electrophiles',
        'description': (
            'Electrophilic compounds that covalently modify cysteine thiol groups in proteins, '
            'causing non-specific HTS interference.'
        ),
        'reference': 'Dahlin JL, et al. J Med Chem. 2015;58(5):2091-2113',
        'doi': '10.1021/jm5019093',
        'additional_refs': [
            'NCBI Assay Guidance Manual NBK326709 (thiol-reactive functional groups)',
        ],
        'source': 'SMARTS patterns: Michael acceptors, acylating agents, SN2 electrophiles, Schiff base formers',
        'method': (
            'Substructure matching against 15 validated SMARTS patterns: '
            'Michael acceptors, acylating agents (acyl halides, anhydrides, activated esters), '
            'SN2 electrophiles (epoxides, aziridines), aldehydes, isocyanates/isothiocyanates, '
            'vinyl sulfones, sulfonyl fluorides'
        ),
        'color': '#b197fc',
    },
}


# =============================================================================
# References for external citation
# =============================================================================

METHODOLOGY_REFERENCES = {
    'pains': {
        'citation': (
            'Baell JB, Holloway GA. New substructure filters for removal of pan assay '
            'interference compounds (PAINS) from screening libraries and for their '
            'exclusion in bioassays. J Med Chem. 2010;53(7):2719-2740.'
        ),
        'doi': '10.1021/jm901137j',
        'url': 'https://doi.org/10.1021/jm901137j',
    },
    'brenk': {
        'citation': (
            'Brenk R, Schipani A, James D, et al. Lessons learnt from assembling '
            'screening libraries for drug discovery for neglected diseases. '
            'ChemMedChem. 2008;3(3):435-444.'
        ),
        'doi': '10.1002/cmdc.200700139',
        'url': 'https://doi.org/10.1002/cmdc.200700139',
    },
    'nih': {
        'citation': (
            'Jadhav A, Ferreira RS, Klumpp C, et al. Quantitative analyses of aggregation, '
            'autofluorescence, and reactivity artifacts in a screen for inhibitors of a '
            'thiol protease. J Med Chem. 2010;53(1):37-51.'
        ),
        'doi': '10.1021/jm901070c',
        'url': 'https://doi.org/10.1021/jm901070c',
    },
    'aggregator': {
        'citation': (
            'Irwin JJ, Duan D, Torosyan H, et al. An Aggregation Advisor for Ligand '
            'Discovery. J Med Chem. 2015;58(17):7076-7087.'
        ),
        'doi': '10.1021/acs.jmedchem.5b01105',
        'url': 'https://doi.org/10.1021/acs.jmedchem.5b01105',
    },
    'thiol': {
        'citation': (
            'Dahlin JL, Nissink JWM, Strasser JM, et al. PAINS in the Assay: Chemical '
            'Mechanisms of Assay Interference and Promiscuous Enzymatic Inhibition '
            'Observed during a Sulfhydryl-Scavenging HTS. J Med Chem. 2015;58(5):2091-2113.'
        ),
        'doi': '10.1021/jm5019093',
        'url': 'https://doi.org/10.1021/jm5019093',
        'additional': [
            {
                'citation': (
                    'NCBI Assay Guidance Manual. Interference with Assay Signal: '
                    'Thiol-Reactive Compounds. NBK326709.'
                ),
                'url': 'https://www.ncbi.nlm.nih.gov/books/NBK326709/',
            },
        ],
    },
    'redox': {
        'citation': (
            'Proj M, Knez D, Sosic I, Gobec S. Redox active or thiol reactive? Optimization '
            'of rapid screens to identify less evident nuisance compounds. '
            'Drug Discov Today. 2022;27(6):1733-1742.'
        ),
        'doi': '10.1016/j.drudis.2022.03.008',
        'url': 'https://doi.org/10.1016/j.drudis.2022.03.008',
        'additional': [
            {
                'citation': 'NCBI Assay Guidance Manual. Chapter: Assay Guidance Manual Glossary. NBK326709.',
                'url': 'https://www.ncbi.nlm.nih.gov/books/NBK326709/',
            },
        ],
    },
    'fluorescence': {
        'citation': (
            'Su BH, Tu YS, Lin C, Shao CY, Lin OA, Tseng YJ. Rule-based classification models '
            'of molecular autofluorescence. J Chem Inf Model. 2015;55(2):434-445.'
        ),
        'doi': '10.1021/ci5007432',
        'url': 'https://doi.org/10.1021/ci5007432',
    },
}


def get_methodology_citation(flag_name: str) -> str:
    """Get the full citation for a flag's detection methodology."""
    flag_lower = flag_name.lower()
    if flag_lower in METHODOLOGY_REFERENCES:
        return METHODOLOGY_REFERENCES[flag_lower]['citation']
    return f"No citation available for {flag_name}"


def get_methodology_doi(flag_name: str) -> str:
    """Get the DOI for a flag's detection methodology."""
    flag_lower = flag_name.lower()
    if flag_lower in METHODOLOGY_REFERENCES:
        return METHODOLOGY_REFERENCES[flag_lower].get('doi', '')
    return ""


def get_all_methodology_references() -> Dict[str, Dict[str, Any]]:
    """Get all methodology references for documentation."""
    return METHODOLOGY_REFERENCES.copy()
