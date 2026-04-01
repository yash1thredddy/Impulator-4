"""
Unit tests for Assay Interference Detection Module.

Tests the seven interference detection mechanisms:
1. PAINS (Pan-Assay Interference Substructures) - RDKit FilterCatalog (480 patterns)
2. Aggregation risk - Shoichet lab heuristics
3. Thiol-reactive - Dahlin et al. (2015) HTS electrophile SMARTS (15 patterns)
4. Redox-active - Quinone/catechol SMARTS (10 patterns)
5. Autofluorescent - Fluorophore scaffold SMARTS (13 patterns)
6. BRENK - RDKit FilterCatalog (104 unwanted substructures)
7. NIH - RDKit FilterCatalog (problematic functional groups)

Also tests InterferenceFlags dataclass, main interface functions, and scoring.

References:
- Baell & Holloway (2010) J. Med. Chem. 53, 2719-2740 (PAINS)
- Irwin et al. (2015) J. Med. Chem. 58, 7076-7087 (Aggregator)
- Dahlin et al. (2015) J. Med. Chem. 58, 2091-2113 (Thiol-reactive)
- Proj et al. (2022) Drug Discov. Today 27, 1733-1742 (Redox)
- Su et al. (2015) J. Chem. Inf. Model. 55, 434-445 (Fluorescence)
- Brenk et al. (2008) ChemMedChem 3, 435-444 (BRENK)
- Jadhav et al. (2009) J. Med. Chem. 53, 37-51 (NIH)
"""
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from backend.modules.assay_interference_filter import (
    InterferenceFlags,
    check_pains_violations,
    check_aggregator_risk,
    check_brenk_alerts,
    check_nih_alerts,
    check_thiol_reactive,
    check_redox_active,
    check_fluorescence_interference,
    calculate_interference_flags,
    get_interference_flags_from_smiles,
    get_interference_summary,
    get_all_filter_matches,
    REDOX_ACTIVE_SMARTS,
    FLUORESCENT_SMARTS,
    THIOL_REACTIVE_SMARTS,
)


# =============================================================================
# InterferenceFlags Dataclass Tests
# =============================================================================

class TestInterferenceFlags:
    """Tests for the InterferenceFlags dataclass."""

    def test_default_initialization(self):
        """Test default initialization with all flags False."""
        flags = InterferenceFlags()
        assert not flags.pains
        assert not flags.aggregator
        assert not flags.thiol
        assert not flags.redox
        assert not flags.fluorescence
        assert not flags.brenk
        assert not flags.nih
        assert flags.total_flags == 0
        assert flags.is_clean

    def test_total_flags_count(self):
        """Test total_flags property counts correctly."""
        flags = InterferenceFlags(pains=True, thiol=True, redox=True)
        assert flags.total_flags == 3
        assert not flags.is_clean

    def test_total_flags_all_seven(self):
        """Test total_flags with all 7 flags set."""
        flags = InterferenceFlags(
            pains=True, aggregator=True, thiol=True,
            redox=True, fluorescence=True, brenk=True, nih=True
        )
        assert flags.total_flags == 7
        assert not flags.is_clean

    def test_to_dict(self):
        """Test conversion to dictionary with integer flags."""
        flags = InterferenceFlags(pains=True, aggregator=False, thiol=True)
        d = flags.to_dict()
        assert d['PAINS'] == 1
        assert d['Aggregator'] == 0
        assert d['Thiol'] == 1
        assert d['Redox'] == 0
        assert d['Fluorescence'] == 0
        assert d['BRENK'] == 0
        assert d['NIH'] == 0

    def test_to_detailed_dict(self):
        """Test conversion to detailed dictionary with reasons."""
        flags = InterferenceFlags(
            pains=True,
            pains_details=['catechol_A(92)'],
            thiol=True,
            thiol_details=['aldehyde', 'michael_acceptor'],
        )
        d = flags.to_detailed_dict()
        assert d['PAINS'] == 1
        assert d['PAINS_Details'] == 'catechol_A(92)'
        assert d['Thiol'] == 1
        assert 'aldehyde' in d['Thiol_Details']
        assert 'michael_acceptor' in d['Thiol_Details']
        assert d['Total_Flags'] == 2

    def test_to_detailed_dict_empty(self):
        """Test detailed dict for clean compound."""
        flags = InterferenceFlags()
        d = flags.to_detailed_dict()
        assert d['PAINS'] == 0
        assert d['PAINS_Details'] == ''
        assert d['Total_Flags'] == 0


# =============================================================================
# PAINS Detection Tests
# =============================================================================

class TestPAINSDetection:
    """Tests for PAINS violation detection.

    Uses RDKit FilterCatalog.PAINS (480 patterns).
    Reference: Baell & Holloway (2010) J. Med. Chem. 53, 2719-2740
    """

    def test_pains_catechol_detected(self):
        """Test that catechol (common PAINS) is detected."""
        mol = Chem.MolFromSmiles('c1ccc(O)c(O)c1')
        has_pains, names = check_pains_violations(mol)
        assert has_pains
        assert len(names) > 0

    def test_pains_clean_molecule(self):
        """Test that clean molecule has no PAINS."""
        mol = Chem.MolFromSmiles('CCO')
        has_pains, names = check_pains_violations(mol)
        assert not has_pains
        assert len(names) == 0

    def test_pains_none_molecule(self):
        """Test handling of None molecule."""
        has_pains, names = check_pains_violations(None)
        assert not has_pains
        assert names == []

    def test_pains_rhodanine_detected(self):
        """Test that rhodanine (known PAINS) is detected."""
        mol = Chem.MolFromSmiles('O=C1NC(=S)SC1')
        has_pains, names = check_pains_violations(mol)
        assert has_pains

    def test_pains_quinone_detected(self):
        """Test that 1,4-benzoquinone triggers PAINS."""
        mol = Chem.MolFromSmiles('O=C1C=CC(=O)C=C1')
        has_pains, names = check_pains_violations(mol)
        assert has_pains

    def test_pains_aspirin_clean(self):
        """Test that aspirin (well-behaved drug) has no PAINS."""
        mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
        has_pains, names = check_pains_violations(mol)
        assert not has_pains


# =============================================================================
# BRENK Filter Detection Tests
# =============================================================================

class TestBRENKDetection:
    """Tests for BRENK filter detection (104 unwanted substructures).

    Reference: Brenk et al. (2008) ChemMedChem 3, 435-444
    """

    def test_brenk_clean_molecule(self):
        """Test that clean molecules don't trigger BRENK."""
        mol = Chem.MolFromSmiles('CCO')
        has_alerts, names = check_brenk_alerts(mol)
        assert not has_alerts

    def test_brenk_aldehyde_detected(self):
        """Test aldehyde detection (known BRENK alert)."""
        mol = Chem.MolFromSmiles('c1ccccc1C=O')
        has_alerts, names = check_brenk_alerts(mol)
        assert has_alerts
        assert len(names) > 0

    def test_brenk_epoxide_detected(self):
        """Test epoxide detection (reactive group in BRENK)."""
        mol = Chem.MolFromSmiles('C1OC1')
        has_alerts, names = check_brenk_alerts(mol)
        assert has_alerts

    def test_brenk_michael_acceptor_detected(self):
        """Test Michael acceptor detection."""
        mol = Chem.MolFromSmiles('C=CC(=O)N')
        has_alerts, names = check_brenk_alerts(mol)
        assert has_alerts

    def test_brenk_thiol_detected(self):
        """Test free thiol detection (BRENK includes thiols)."""
        mol = Chem.MolFromSmiles('N[C@@H](CS)C(=O)O')
        has_alerts, names = check_brenk_alerts(mol)
        assert has_alerts
        assert len(names) > 0

    def test_brenk_disulfide_detected(self):
        """Test disulfide bond detection."""
        mol = Chem.MolFromSmiles('N[C@@H](CSSC[C@H](N)C(=O)O)C(=O)O')
        has_alerts, names = check_brenk_alerts(mol)
        assert has_alerts

    def test_brenk_maleimide_detected(self):
        """Test maleimide detection."""
        mol = Chem.MolFromSmiles('O=C1C=CC(=O)N1')
        has_alerts, names = check_brenk_alerts(mol)
        assert has_alerts

    def test_brenk_isothiocyanate_detected(self):
        """Test isothiocyanate detection."""
        mol = Chem.MolFromSmiles('c1ccccc1N=C=S')
        has_alerts, names = check_brenk_alerts(mol)
        assert has_alerts

    def test_brenk_none_molecule(self):
        """Test handling of None molecule."""
        has_alerts, names = check_brenk_alerts(None)
        assert not has_alerts
        assert names == []


# =============================================================================
# NIH Filter Detection Tests
# =============================================================================

class TestNIHDetection:
    """Tests for NIH filter detection (problematic functional groups).

    Reference: Jadhav et al. (2009) J. Med. Chem. 53, 37-51
    """

    def test_nih_clean_molecule(self):
        """Test that clean molecules don't trigger NIH alerts."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        has_alerts, names = check_nih_alerts(mol)
        assert not has_alerts

    def test_nih_none_molecule(self):
        """Test handling of None molecule."""
        has_alerts, names = check_nih_alerts(None)
        assert not has_alerts
        assert names == []


# =============================================================================
# Aggregator Risk Detection Tests
# =============================================================================

class TestAggregatorRiskDetection:
    """Tests for aggregation risk detection.

    Uses Shoichet lab heuristics (all 4 criteria must be met).
    Reference: Irwin et al. (2015) J. Med. Chem. 58, 7076-7087
    """

    def test_aggregator_risk_aromatic_lipophilic(self):
        """Test aggregator detection for highly aromatic, lipophilic molecule."""
        mol = Chem.MolFromSmiles('c1ccc2cc3cc4ccccc4cc3cc2c1')
        is_risk, reason = check_aggregator_risk(mol)
        assert isinstance(is_risk, bool)
        assert isinstance(reason, str)

    def test_aggregator_risk_small_polar_molecule(self):
        """Test that small polar molecule has no aggregator risk."""
        mol = Chem.MolFromSmiles('NCC(=O)O')
        is_risk, reason = check_aggregator_risk(mol)
        assert not is_risk

    def test_aggregator_none_molecule(self):
        """Test handling of None molecule."""
        is_risk, reason = check_aggregator_risk(None)
        assert not is_risk
        assert reason == ""

    def test_aggregator_drug_like_no_risk(self):
        """Test that typical drug-like molecules don't trigger risk."""
        mol = Chem.MolFromSmiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
        is_risk, reason = check_aggregator_risk(mol)
        assert not is_risk

    def test_aggregator_large_aromatic(self):
        """Test that large, rigid, lipophilic aromatics trigger risk."""
        mol = Chem.MolFromSmiles('c1cc2ccc3ccc4ccc5ccc6ccc1c7c2c3c4c5c67')
        is_risk, reason = check_aggregator_risk(mol)
        assert is_risk

    def test_aggregator_naphthalene_no_risk(self):
        """Test that naphthalene (only 2 rings) does not trigger risk."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        is_risk, reason = check_aggregator_risk(mol)
        assert not is_risk


# =============================================================================
# Redox-Active Detection Tests
# =============================================================================

class TestRedoxActiveDetection:
    """Tests for redox-active group detection.

    Reference: Proj et al. (2022) Drug Discov. Today 27, 1733-1742
    """

    def test_redox_catechol_detected(self):
        """Test that catechol is detected as redox-active."""
        mol = Chem.MolFromSmiles('Oc1ccccc1O')
        is_redox, groups = check_redox_active(mol)
        assert is_redox
        assert 'catechol' in groups

    def test_redox_hydroquinone_detected(self):
        """Test that hydroquinone is detected as redox-active."""
        mol = Chem.MolFromSmiles('Oc1ccc(O)cc1')
        is_redox, groups = check_redox_active(mol)
        assert is_redox
        assert 'hydroquinone' in groups

    def test_redox_quinone_detected(self):
        """Test that p-benzoquinone is detected as redox-active."""
        mol = Chem.MolFromSmiles('O=C1C=CC(=O)C=C1')
        is_redox, groups = check_redox_active(mol)
        assert is_redox

    def test_redox_hydroxylamine_detected(self):
        """Test that hydroxylamine is detected as redox-active."""
        mol = Chem.MolFromSmiles('NO')
        is_redox, groups = check_redox_active(mol)
        assert is_redox
        assert 'hydroxylamine' in groups

    def test_redox_nitroso_detected(self):
        """Test that nitroso compounds are detected as redox-active."""
        mol = Chem.MolFromSmiles('O=Nc1ccccc1')
        is_redox, groups = check_redox_active(mol)
        assert is_redox
        assert 'nitroso' in groups

    def test_redox_nitro_aromatic_detected(self):
        """Test that nitroaromatics are detected as redox-active."""
        mol = Chem.MolFromSmiles('c1ccc([N+](=O)[O-])cc1')
        is_redox, groups = check_redox_active(mol)
        assert is_redox
        assert 'nitro_aromatic' in groups

    def test_redox_clean_molecule(self):
        """Test that clean molecule has no redox groups."""
        mol = Chem.MolFromSmiles('CCCCCC')
        is_redox, groups = check_redox_active(mol)
        assert not is_redox
        assert len(groups) == 0

    def test_redox_none_molecule(self):
        """Test handling of None molecule."""
        is_redox, groups = check_redox_active(None)
        assert not is_redox
        assert groups == []


# =============================================================================
# Fluorescence Interference Detection Tests
# =============================================================================

class TestFluorescenceInterferenceDetection:
    """Tests for fluorescence interference detection.

    Reference: Su et al. (2015) J. Chem. Inf. Model. 55, 434-445
    """

    def test_fluorescence_naphthalene_detected(self):
        """Test that naphthalene is detected as fluorescent."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        is_fluor, scaffolds = check_fluorescence_interference(mol)
        assert is_fluor
        assert 'naphthalene' in scaffolds

    def test_fluorescence_anthracene_detected(self):
        """Test that anthracene is detected as fluorescent."""
        mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        is_fluor, scaffolds = check_fluorescence_interference(mol)
        assert is_fluor
        assert 'anthracene' in scaffolds

    def test_fluorescence_pyrene_detected(self):
        """Test that pyrene is detected as fluorescent."""
        mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')
        is_fluor, scaffolds = check_fluorescence_interference(mol)
        assert is_fluor
        assert 'pyrene' in scaffolds

    def test_fluorescence_coumarin_detected(self):
        """Test that coumarin is detected as fluorescent."""
        mol = Chem.MolFromSmiles('O=c1ccc2ccccc2o1')
        is_fluor, scaffolds = check_fluorescence_interference(mol)
        assert is_fluor

    def test_fluorescence_acridine_detected(self):
        """Test that acridine is detected as fluorescent."""
        mol = Chem.MolFromSmiles('c1ccc2nc3ccccc3cc2c1')
        is_fluor, scaffolds = check_fluorescence_interference(mol)
        assert is_fluor
        assert 'acridine' in scaffolds

    def test_fluorescence_clean_molecule(self):
        """Test that simple molecule has no fluorescence."""
        mol = Chem.MolFromSmiles('CCO')
        is_fluor, scaffolds = check_fluorescence_interference(mol)
        assert not is_fluor

    def test_fluorescence_none_molecule(self):
        """Test handling of None molecule."""
        is_fluor, scaffolds = check_fluorescence_interference(None)
        assert not is_fluor
        assert scaffolds == []


# =============================================================================
# Thiol Reactivity Detection Tests
# =============================================================================

class TestThiolReactivityDetection:
    """Tests for thiol-reactive electrophile detection.

    Reference: Dahlin et al. (2015) J. Med. Chem. 58, 2091-2113
    """

    def test_thiol_acrylamide_detected(self):
        """Test that acrylamide is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('C=CC(=O)N')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'acrylamide' in groups

    def test_thiol_epoxide_detected(self):
        """Test that epoxide is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('C1OC1')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'epoxide' in groups

    def test_thiol_isothiocyanate_detected(self):
        """Test that isothiocyanate is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('CN=C=S')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'isothiocyanate' in groups

    def test_thiol_aldehyde_detected(self):
        """Test that aldehyde is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('CC=O')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'aldehyde' in groups

    def test_thiol_acrylate_detected(self):
        """Test that acrylate is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('C=CC(=O)O')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'acrylate' in groups

    def test_thiol_maleimide_detected(self):
        """Test that maleimide is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('O=C1C=CC(=O)N1')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'maleimide' in groups

    def test_thiol_aziridine_detected(self):
        """Test that aziridine is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('C1NC1')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'aziridine' in groups

    def test_thiol_isocyanate_detected(self):
        """Test that isocyanate is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('CN=C=O')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'isocyanate' in groups

    def test_thiol_acyl_halide_detected(self):
        """Test that acyl halide is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('CC(=O)Cl')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'acyl_halide' in groups

    def test_thiol_anhydride_detected(self):
        """Test that anhydride is detected as thiol-reactive."""
        mol = Chem.MolFromSmiles('CC(=O)OC(=O)C')
        is_reactive, groups = check_thiol_reactive(mol)
        assert is_reactive
        assert 'anhydride' in groups

    def test_thiol_clean_molecule(self):
        """Test that clean molecule has no thiol reactivity."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        is_reactive, groups = check_thiol_reactive(mol)
        assert not is_reactive

    def test_thiol_none_molecule(self):
        """Test handling of None molecule."""
        is_reactive, groups = check_thiol_reactive(None)
        assert not is_reactive
        assert groups == []


# =============================================================================
# Main Interface: calculate_interference_flags
# =============================================================================

class TestCalculateInterferenceFlags:
    """Tests for the main calculate_interference_flags function."""

    def test_clean_drug_molecule(self):
        """Test a clean drug molecule (ibuprofen)."""
        mol = Chem.MolFromSmiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
        flags = calculate_interference_flags(mol)
        assert not flags.pains
        assert not flags.aggregator

    def test_quercetin_multiple_flags(self):
        """Test quercetin (known to have multiple interference mechanisms)."""
        mol = Chem.MolFromSmiles('O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12')
        flags = calculate_interference_flags(mol)
        assert flags.pains
        assert flags.total_flags > 0

    def test_none_molecule_returns_empty_flags(self):
        """Test that None molecule returns all-False flags."""
        flags = calculate_interference_flags(None)
        assert flags.is_clean
        assert flags.total_flags == 0

    def test_aldehyde_triggers_thiol(self):
        """Test that aldehyde triggers thiol-reactive flag."""
        mol = Chem.MolFromSmiles('c1ccccc1C=O')
        flags = calculate_interference_flags(mol)
        assert flags.thiol

    def test_epoxide_triggers_thiol(self):
        """Test that epoxide triggers thiol-reactive flag."""
        mol = Chem.MolFromSmiles('C1OC1')
        flags = calculate_interference_flags(mol)
        assert flags.thiol

    def test_quinone_triggers_redox(self):
        """Test that quinone triggers redox flag."""
        mol = Chem.MolFromSmiles('O=C1C=CC(=O)C=C1')
        flags = calculate_interference_flags(mol)
        assert flags.redox

    def test_catechol_triggers_redox(self):
        """Test that catechol triggers redox flag."""
        mol = Chem.MolFromSmiles('Oc1ccccc1O')
        flags = calculate_interference_flags(mol)
        assert flags.redox


# =============================================================================
# SMILES Interface: get_interference_flags_from_smiles
# =============================================================================

class TestGetInterferenceFlagsFromSmiles:
    """Tests for the SMILES-based interface function."""

    def test_valid_smiles_clean(self):
        """Test with valid clean SMILES."""
        flags = get_interference_flags_from_smiles('CCO')
        assert isinstance(flags, InterferenceFlags)
        assert flags.is_clean

    def test_invalid_smiles(self):
        """Test with invalid SMILES."""
        flags = get_interference_flags_from_smiles('invalid_smiles')
        assert isinstance(flags, InterferenceFlags)
        assert flags.is_clean

    def test_empty_smiles(self):
        """Test with empty SMILES."""
        flags = get_interference_flags_from_smiles('')
        assert flags.is_clean

    def test_na_smiles(self):
        """Test with 'N/A' SMILES."""
        flags = get_interference_flags_from_smiles('N/A')
        assert flags.is_clean

    def test_catechol_smiles(self):
        """Test catechol detection from SMILES."""
        flags = get_interference_flags_from_smiles('Oc1ccccc1O')
        assert flags.pains
        assert flags.redox

    def test_quercetin_smiles(self):
        """Test quercetin flags from SMILES."""
        flags = get_interference_flags_from_smiles(
            'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'
        )
        assert flags.pains or flags.redox
        assert flags.fluorescence


# =============================================================================
# get_interference_summary Tests
# =============================================================================

class TestGetInterferenceSummary:
    """Tests for get_interference_summary function."""

    def test_summary_structure(self):
        """Test that summary has correct structure."""
        flags = InterferenceFlags(pains=True, thiol=True)
        summary = get_interference_summary(flags)

        assert 'total_flags' in summary
        assert 'is_clean' in summary
        assert 'flags' in summary
        assert 'details' in summary
        assert summary['total_flags'] == 2
        assert not summary['is_clean']

    def test_summary_clean(self):
        """Test summary for clean compound."""
        flags = InterferenceFlags()
        summary = get_interference_summary(flags)

        assert summary['total_flags'] == 0
        assert summary['is_clean']

    def test_summary_flags_dict(self):
        """Test that flags dict in summary uses integer values."""
        flags = InterferenceFlags(pains=True, brenk=True)
        summary = get_interference_summary(flags)
        assert summary['flags']['PAINS'] == 1
        assert summary['flags']['BRENK'] == 1
        assert summary['flags']['Aggregator'] == 0


# =============================================================================
# get_all_filter_matches Tests
# =============================================================================

class TestGetAllFilterMatches:
    """Tests for get_all_filter_matches function."""

    def test_returns_all_catalogs(self):
        """Test that all catalogs are returned."""
        mol = Chem.MolFromSmiles('CCO')
        results = get_all_filter_matches(mol)
        assert 'PAINS' in results
        assert 'BRENK' in results
        assert 'NIH' in results
        assert 'ZINC' in results

    def test_none_molecule(self):
        """Test handling of None molecule."""
        results = get_all_filter_matches(None)
        assert results == {}

    def test_catechol_matches_pains(self):
        """Test that catechol triggers PAINS in all filter matches."""
        mol = Chem.MolFromSmiles('Oc1ccccc1O')
        results = get_all_filter_matches(mol)
        assert len(results['PAINS']) > 0


# =============================================================================
# SMARTS Pattern Validation Tests
# =============================================================================

class TestSMARTSPatterns:
    """Test that SMARTS patterns are valid and compile correctly."""

    def test_redox_patterns_valid(self):
        """Test that all REDOX_ACTIVE_SMARTS are valid SMARTS."""
        for name, smarts in REDOX_ACTIVE_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            assert pattern is not None, f"Invalid SMARTS for {name}: {smarts}"

    def test_fluorescent_patterns_valid(self):
        """Test that all FLUORESCENT_SMARTS are valid SMARTS."""
        for name, smarts in FLUORESCENT_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            assert pattern is not None, f"Invalid SMARTS for {name}: {smarts}"

    def test_thiol_reactive_patterns_valid(self):
        """Test that all THIOL_REACTIVE_SMARTS are valid SMARTS."""
        for name, smarts in THIOL_REACTIVE_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            assert pattern is not None, f"Invalid SMARTS for {name}: {smarts}"

    def test_pattern_counts(self):
        """Verify expected number of patterns per category."""
        assert len(THIOL_REACTIVE_SMARTS) == 15
        assert len(REDOX_ACTIVE_SMARTS) == 10
        assert len(FLUORESCENT_SMARTS) == 13


# =============================================================================
# SMARTS Pattern Matching Tests
# =============================================================================

class TestSMARTSPatternMatching:
    """Test that SMARTS patterns correctly match molecules."""

    def test_catechol_pattern_matches_catechol(self):
        """Test catechol SMARTS matches actual catechol."""
        pattern = Chem.MolFromSmarts(REDOX_ACTIVE_SMARTS['catechol'])
        assert pattern is not None
        catechol = Chem.MolFromSmiles('Oc1ccccc1O')
        assert catechol.HasSubstructMatch(pattern)

    def test_epoxide_pattern_matches_epoxide(self):
        """Test epoxide SMARTS matches three-membered ring with oxygen."""
        pattern = Chem.MolFromSmarts(THIOL_REACTIVE_SMARTS['epoxide'])
        assert pattern is not None
        ethylene_oxide = Chem.MolFromSmiles('C1OC1')
        assert ethylene_oxide.HasSubstructMatch(pattern)

    def test_aldehyde_pattern_matches_aldehyde(self):
        """Test aldehyde SMARTS matches aldehydes."""
        pattern = Chem.MolFromSmarts(THIOL_REACTIVE_SMARTS['aldehyde'])
        assert pattern is not None
        benzaldehyde = Chem.MolFromSmiles('c1ccccc1C=O')
        assert benzaldehyde.HasSubstructMatch(pattern)

    def test_quinone_pattern_matches_quinone(self):
        """Test para-quinone SMARTS matches p-benzoquinone."""
        pattern = Chem.MolFromSmarts(REDOX_ACTIVE_SMARTS['para_quinone'])
        assert pattern is not None
        benzoquinone = Chem.MolFromSmiles('O=C1C=CC(=O)C=C1')
        assert benzoquinone.HasSubstructMatch(pattern)

    def test_naphthalene_pattern_matches_naphthalene(self):
        """Test naphthalene SMARTS matches naphthalene."""
        pattern = Chem.MolFromSmarts(FLUORESCENT_SMARTS['naphthalene'])
        assert pattern is not None
        naphthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert naphthalene.HasSubstructMatch(pattern)


# =============================================================================
# Known Compound Tests
# =============================================================================

class TestKnownCompounds:
    """Tests with well-characterized compounds from literature."""

    def test_ibuprofen_clean(self):
        """Test ibuprofen - should be clean (no flags)."""
        flags = get_interference_flags_from_smiles('CC(C)Cc1ccc(cc1)C(C)C(=O)O')
        assert flags.is_clean

    def test_quercetin_problematic(self):
        """Test quercetin - catechol, flavonoid, redox-active."""
        flags = get_interference_flags_from_smiles(
            'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'
        )
        assert flags.pains or flags.redox
        assert flags.fluorescence
        assert not flags.is_clean

    def test_dopamine_catechol(self):
        """Test dopamine (catechol - PAINS)."""
        mol = Chem.MolFromSmiles('NCCc1ccc(O)c(O)c1')
        flags = calculate_interference_flags(mol)
        assert flags.pains

    def test_acrylamide_michael_acceptor(self):
        """Test acrylamide (Michael acceptor - thiol-reactive)."""
        mol = Chem.MolFromSmiles('C=CC(=O)N')
        flags = calculate_interference_flags(mol)
        assert flags.thiol

    def test_maleimide_thiol_reactive(self):
        """Test maleimide triggers thiol-reactive flag."""
        mol = Chem.MolFromSmiles('O=C1C=CC(=O)N1')
        flags = calculate_interference_flags(mol)
        assert flags.thiol or flags.pains


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_large_molecule(self):
        """Test handling of very large molecule."""
        large_smiles = 'C' * 100 + 'O'
        flags = get_interference_flags_from_smiles(large_smiles)
        assert isinstance(flags, InterferenceFlags)

    def test_charged_molecule(self):
        """Test handling of charged molecule."""
        flags = get_interference_flags_from_smiles('CC(=O)[O-].[Na+]')
        assert isinstance(flags, InterferenceFlags)

    def test_stereochemistry(self):
        """Test handling of molecules with stereochemistry."""
        flags = get_interference_flags_from_smiles('C[C@H](N)C(=O)O')
        assert isinstance(flags, InterferenceFlags)

    def test_aromatic_heterocycle(self):
        """Test handling of aromatic heterocycles."""
        flags = get_interference_flags_from_smiles('c1ccncc1')
        assert isinstance(flags, InterferenceFlags)

    def test_single_atom(self):
        """Test single atom molecule."""
        mol = Chem.MolFromSmiles('[Na]')
        flags = calculate_interference_flags(mol)
        assert isinstance(flags, InterferenceFlags)

    def test_radical(self):
        """Test molecule with radical."""
        mol = Chem.MolFromSmiles('[CH3]')
        flags = calculate_interference_flags(mol)
        assert isinstance(flags, InterferenceFlags)

    def test_multiple_fused_rings_fluorescence(self):
        """Test molecule with multiple fused rings detects fluorescence."""
        flags = get_interference_flags_from_smiles('c1cc2ccc3cccc4ccc(c1)c2c34')
        assert isinstance(flags, InterferenceFlags)
        assert flags.fluorescence
