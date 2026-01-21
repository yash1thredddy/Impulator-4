"""
Assay Interference Detection Module

This module detects compounds with known assay interference mechanisms identified
in the 2016 Bisson et al. "Invalid Metabolic Panaceas (IMPs)" paper.

Five core interference mechanisms are detected:
1. PAINS (Pan-Assay Interference Substructures)
2. Aggregation risk (colloidal aggregators)
3. Redox reactivity (redox-active functional groups)
4. Fluorescence interference (autofluorescent compounds)
5. Thiol reactivity (cysteine-reactive electrophiles)

IMPORTANT: These flags are displayed as ORTHOGONAL INFORMATION alongside O[Q/P/L]A scores.
They do NOT penalize the O[Q/P/L]A score. Compounds with interference flags but strong
PDB structural evidence (e.g., quercetin) are retained as valid IMPs exhibiting genuine
polypharmacology rather than assay artifacts.

References:
- Bisson et al. (2016) J. Med. Chem. 59, 1671-1690 (Invalid Metabolic Panaceas)
- Baell & Holloway (2010) J. Med. Chem. 53, 2719-2740 (PAINS filters)
- Shoichet laboratory aggregator research (http://www.bkslab.org)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import FilterCatalog, rdMolDescriptors, Descriptors

logger = logging.getLogger(__name__)


# =============================================================================
# CONSOLIDATED SMARTS PATTERNS
# All patterns centralized here for easier maintenance and consistency
# =============================================================================

# Redox-active functional groups that can cause assay interference
REDOX_PATTERNS = {
    'catechol': 'c1c(O)c(O)ccc1',  # ortho-diphenol
    'quinone': 'C1(=O)C=CC(=O)C=C1',  # benzoquinone
    'disulfide': '[S;D2]-[S;D2]',  # S-S bond
    'thiol': '[SH1]',  # free thiol
    'hydroquinone': 'c1c(O)ccc(O)c1',  # para-diphenol
    'anthraquinone': 'c1ccc2c(c1)C(=O)c1ccccc1C2=O',
    'naphthoquinone': 'C1=CC2=C(C=C1)C(=O)C=CC2=O',
}

# Fluorescent scaffolds that can interfere with fluorescence-based assays
FLUORESCENT_PATTERNS = {
    'flavonoid': 'O=c1cc(-c2ccccc2)oc2ccccc12',  # flavone core
    'coumarin': 'O=C1C=Cc2ccccc2O1',
    'xanthene': 'c1ccc2c(c1)Cc1ccccc1O2',
    'naphthalene': 'c1ccc2ccccc2c1',
    'anthracene': 'c1ccc2cc3ccccc3cc2c1',
    'stilbene': 'c1ccccc1C=Cc1ccccc1',  # extended conjugation
}

# Thiol-reactive electrophiles that can modify cysteine residues
THIOL_REACTIVE_PATTERNS = {
    'michael_acceptor': '[C;$(C=C)]-[C;$(C=O)]',  # α,β-unsaturated carbonyl
    'acrylamide': 'C=CC(=O)N',
    'maleimide': 'O=C1C=CC(=O)N1',
    'aldehyde': '[CH1](=O)',  # not part of carboxylic acid
    'activated_ester': '[C;$(C(=O)O)][F,Cl,Br,I]',
    'epoxide': 'C1OC1',
    'isothiocyanate': 'N=C=S',
    'vinyl_sulfone': 'C=CS(=O)(=O)',
}


# ============================================================================
# PAINS (Pan-Assay Interference Substructures) Detection
# ============================================================================

def check_pains_violations(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    Check for PAINS (Pan-Assay Interference Substructures) using RDKit FilterCatalog.

    PAINS are substructures associated with promiscuous bioactivity and assay interference.
    Originally identified by Baell & Holloway (2010) from HTS campaigns.

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple[bool, List[str]]: (has_pains, list_of_pains_names)

    Example:
        >>> mol = Chem.MolFromSmiles('c1c(O)c(O)ccc1')  # catechol
        >>> has_pains, names = check_pains_violations(mol)
        >>> print(has_pains)  # True
    """
    if mol is None:
        return False, []

    try:
        # Initialize PAINS filter catalog (RDKit built-in)
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        fc = FilterCatalog.FilterCatalog(params)

        # Check for matches
        pains_names = []
        entry = fc.GetFirstMatch(mol)

        if entry is not None:
            pains_names.append(entry.GetDescription())

            # Check for additional matches
            matches = fc.GetMatches(mol)
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


# ============================================================================
# Aggregation Risk Detection
# ============================================================================

def check_aggregator_risk(mol: Chem.Mol) -> Tuple[bool, str]:
    """
    Detect aggregation risk using Shoichet laboratory heuristics.

    Aggregators are compounds that form colloidal aggregates in aqueous solution,
    leading to non-specific protein inhibition (not genuine binding).

    Risk factors (Shoichet lab criteria):
    - Multiple aromatic rings (≥3)
    - Moderate molecular weight (>300 Da)
    - Low rotatable bonds (≤2, rigid structure)
    - High lipophilicity (LogP > 3)

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple[bool, str]: (is_aggregator_risk, reason)

    Example:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2ccc4c3cccc4')  # anthracene
        >>> is_risk, reason = check_aggregator_risk(mol)
    """
    if mol is None:
        return False, ""

    try:
        # Calculate molecular descriptors
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)

        # Aggregator risk heuristic
        risk_factors = []

        if num_aromatic_rings >= 3:
            risk_factors.append(f"{num_aromatic_rings} aromatic rings")

        if mw > 300:
            risk_factors.append(f"MW={mw:.1f}")

        if num_rotatable_bonds <= 2:
            risk_factors.append(f"{num_rotatable_bonds} rotatable bonds")

        if logp > 3:
            risk_factors.append(f"LogP={logp:.2f}")

        # Risk if meets ALL four criteria (conservative)
        is_risk = len(risk_factors) >= 4
        reason = "; ".join(risk_factors) if is_risk else ""

        if is_risk:
            logger.debug(f"Aggregator risk detected: {reason}")

        return is_risk, reason

    except Exception as e:
        logger.warning(f"Error in aggregator detection: {e}")
        return False, ""


# ============================================================================
# Redox Reactivity Detection
# ============================================================================

def check_redox_reactive(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    Detect redox-active functional groups that can cause assay interference.

    Redox-active compounds can:
    - Generate reactive oxygen species (ROS)
    - Form quinone/semiquinone intermediates
    - Oxidize assay components
    - Reduce metal-containing enzymes

    Common redox-active groups:
    - Catechols (o-diphenols)
    - Quinones
    - Disulfides
    - Thiols
    - Hydroquinones

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple[bool, List[str]]: (is_redox_reactive, list_of_groups)

    Example:
        >>> mol = Chem.MolFromSmiles('Oc1ccc(O)c(O)c1')  # quercetin-like catechol
        >>> is_redox, groups = check_redox_reactive(mol)
        >>> print('catechol' in groups)  # True
    """
    if mol is None:
        return False, []

    try:
        detected_groups = []

        # Use consolidated REDOX_PATTERNS from module-level constant
        for group_name, smarts in REDOX_PATTERNS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                detected_groups.append(group_name)

        is_redox = len(detected_groups) > 0

        if is_redox:
            logger.debug(f"Redox-reactive groups detected: {', '.join(detected_groups)}")

        return is_redox, detected_groups

    except Exception as e:
        logger.warning(f"Error in redox detection: {e}")
        return False, []


# ============================================================================
# Fluorescence Interference Detection
# ============================================================================

def check_fluorescence_interference(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    Detect compounds likely to cause fluorescence interference in assays.

    Autofluorescent compounds can:
    - Interfere with fluorescence-based assays
    - Quench fluorescent reporters
    - Generate false positive/negative signals

    Common autofluorescent scaffolds:
    - Flavonoids (quercetin, kaempferol, etc.)
    - Coumarins
    - Xanthenes (rhodamines, fluoresceins)
    - Extended conjugated systems (>4 double bonds)
    - Naphthalenes and anthracenes

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple[bool, List[str]]: (is_fluorescent, list_of_scaffold_types)

    Example:
        >>> mol = Chem.MolFromSmiles('O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12')
        >>> is_fluor, scaffolds = check_fluorescence_interference(mol)
        >>> print('flavonoid' in scaffolds)  # True (quercetin)
    """
    if mol is None:
        return False, []

    try:
        detected_scaffolds = []

        # Use consolidated FLUORESCENT_PATTERNS from module-level constant
        for scaffold_name, smarts in FLUORESCENT_PATTERNS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                detected_scaffolds.append(scaffold_name)

        # Check for extended conjugation (>4 conjugated double bonds)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        if num_aromatic_rings >= 3:
            if 'extended_conjugation' not in detected_scaffolds:
                detected_scaffolds.append('extended_conjugation')

        is_fluorescent = len(detected_scaffolds) > 0

        if is_fluorescent:
            logger.debug(f"Fluorescent scaffolds detected: {', '.join(detected_scaffolds)}")

        return is_fluorescent, detected_scaffolds

    except Exception as e:
        logger.warning(f"Error in fluorescence detection: {e}")
        return False, []


# ============================================================================
# Thiol Reactivity Detection
# ============================================================================

def check_thiol_reactive(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    Detect electrophilic groups that react with cysteine thiols in proteins.

    Thiol-reactive compounds can:
    - Covalently modify cysteine residues
    - Cause non-specific protein inhibition
    - Show promiscuous activity across many targets

    Common thiol-reactive electrophiles:
    - Michael acceptors (α,β-unsaturated carbonyls)
    - Acrylamides
    - Maleimides
    - Aldehydes
    - Activated esters
    - Epoxides
    - Isothiocyanates

    Args:
        mol: RDKit Mol object

    Returns:
        Tuple[bool, List[str]]: (is_thiol_reactive, list_of_electrophiles)

    Example:
        >>> mol = Chem.MolFromSmiles('C=CC(=O)N')  # acrylamide
        >>> is_reactive, groups = check_thiol_reactive(mol)
        >>> print('michael_acceptor' in groups)  # True
    """
    if mol is None:
        return False, []

    try:
        detected_electrophiles = []

        # Use consolidated THIOL_REACTIVE_PATTERNS from module-level constant
        for group_name, smarts in THIOL_REACTIVE_PATTERNS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                detected_electrophiles.append(group_name)

        is_reactive = len(detected_electrophiles) > 0

        if is_reactive:
            logger.debug(f"Thiol-reactive groups detected: {', '.join(detected_electrophiles)}")

        return is_reactive, detected_electrophiles

    except Exception as e:
        logger.warning(f"Error in thiol reactivity detection: {e}")
        return False, []


# ============================================================================
# Main Interface Functions
# ============================================================================

def get_all_interference_flags(smiles: str) -> Dict[str, bool]:
    """
    Run all assay interference checks on a SMILES string.

    This is the main interface function used by the data processing pipeline.

    Args:
        smiles: SMILES string representing the molecule

    Returns:
        Dict[str, bool]: Dictionary with 5 boolean flags:
            - 'PAINS': Pan-Assay Interference Substructures
            - 'Aggregator': Aggregation risk
            - 'Redox': Redox reactivity
            - 'Fluorescence': Fluorescence interference
            - 'Thiol_Reactive': Thiol reactivity

    Example:
        >>> flags = get_all_interference_flags('O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12')
        >>> print(flags)
        {'PAINS': True, 'Aggregator': False, 'Redox': True,
         'Fluorescence': True, 'Thiol_Reactive': False}
    """
    # Initialize all flags to False
    flags = {
        'PAINS': False,
        'Aggregator': False,
        'Redox': False,
        'Fluorescence': False,
        'Thiol_Reactive': False
    }

    if not smiles or smiles == 'N/A':
        return flags

    try:
        # Parse SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return flags

        # Run all checks
        flags['PAINS'], _ = check_pains_violations(mol)
        flags['Aggregator'], _ = check_aggregator_risk(mol)
        flags['Redox'], _ = check_redox_reactive(mol)
        flags['Fluorescence'], _ = check_fluorescence_interference(mol)
        flags['Thiol_Reactive'], _ = check_thiol_reactive(mol)

    except Exception as e:
        logger.error(f"Error in get_all_interference_flags for SMILES '{smiles}': {e}")

    return flags


def calculate_assay_quality_score(flags_dict: Dict[str, bool]) -> float:
    """
    Calculate an assay quality score based on interference flags.

    Score ranges from 0.0 (all 5 flags) to 1.0 (no flags).
    Formula: 1.0 - (num_flags / 5)

    IMPORTANT: This score is for DISPLAY purposes only. It does NOT affect
    the O[Q/P/L]A score. Compounds with low assay quality but strong PDB
    evidence are still valid IMPs.

    Args:
        flags_dict: Dictionary with boolean flags (from get_all_interference_flags)

    Returns:
        float: Assay quality score (0.0 - 1.0)

    Example:
        >>> flags = {'PAINS': True, 'Aggregator': False, 'Redox': True,
        ...          'Fluorescence': True, 'Thiol_Reactive': False}
        >>> score = calculate_assay_quality_score(flags)
        >>> print(score)  # 0.4 (3 flags out of 5)
    """
    num_flags = sum(1 for flag in flags_dict.values() if flag)
    assay_quality_score = 1.0 - (num_flags / 5.0)

    return assay_quality_score


def get_detailed_interference_report(smiles: str) -> Dict[str, Any]:
    """
    Generate a detailed report of all interference mechanisms for a compound.

    This function provides detailed information about which specific substructures
    or properties triggered each flag, useful for debugging and interpretation.

    Args:
        smiles: SMILES string

    Returns:
        Dict with detailed information for each mechanism

    Example:
        >>> report = get_detailed_interference_report('Oc1ccc(O)c(O)c1')
        >>> print(report['redox']['detected_groups'])
        ['catechol']
    """
    report = {
        'smiles': smiles,
        'pains': {'flag': False, 'violations': []},
        'aggregator': {'flag': False, 'reason': ''},
        'redox': {'flag': False, 'detected_groups': []},
        'fluorescence': {'flag': False, 'scaffolds': []},
        'thiol_reactive': {'flag': False, 'electrophiles': []},
        'num_flags': 0,
        'assay_quality_score': 1.0
    }

    if not smiles or smiles == 'N/A':
        return report

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return report

        # PAINS
        has_pains, pains_names = check_pains_violations(mol)
        report['pains']['flag'] = has_pains
        report['pains']['violations'] = pains_names

        # Aggregator
        is_agg, reason = check_aggregator_risk(mol)
        report['aggregator']['flag'] = is_agg
        report['aggregator']['reason'] = reason

        # Redox
        is_redox, groups = check_redox_reactive(mol)
        report['redox']['flag'] = is_redox
        report['redox']['detected_groups'] = groups

        # Fluorescence
        is_fluor, scaffolds = check_fluorescence_interference(mol)
        report['fluorescence']['flag'] = is_fluor
        report['fluorescence']['scaffolds'] = scaffolds

        # Thiol reactive
        is_reactive, electrophiles = check_thiol_reactive(mol)
        report['thiol_reactive']['flag'] = is_reactive
        report['thiol_reactive']['electrophiles'] = electrophiles

        # Calculate totals
        flags = {
            'PAINS': has_pains,
            'Aggregator': is_agg,
            'Redox': is_redox,
            'Fluorescence': is_fluor,
            'Thiol_Reactive': is_reactive
        }
        report['num_flags'] = sum(flags.values())
        report['assay_quality_score'] = calculate_assay_quality_score(flags)

    except Exception as e:
        logger.error(f"Error generating interference report: {e}")

    return report
