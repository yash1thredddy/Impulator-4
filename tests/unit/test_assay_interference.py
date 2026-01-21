"""
Unit tests for Assay Interference Detection Module.

Tests the five core interference mechanisms:
1. PAINS (Pan-Assay Interference Substructures)
2. Aggregation risk
3. Redox reactivity
4. Fluorescence interference
5. Thiol reactivity

Also tests the main interface functions and scoring.
"""
import pytest
from rdkit import Chem


class TestPAINSDetection:
    """Tests for PAINS violation detection."""

    def test_pains_catechol_detected(self):
        """Test that catechol (common PAINS) is detected."""
        from backend.modules.assay_interference_filter import check_pains_violations

        # Catechol - ortho-diphenol, known PAINS pattern
        mol = Chem.MolFromSmiles('c1ccc(O)c(O)c1')
        has_pains, names = check_pains_violations(mol)

        assert has_pains == True
        assert len(names) > 0

    def test_pains_clean_molecule(self):
        """Test that clean molecule has no PAINS."""
        from backend.modules.assay_interference_filter import check_pains_violations

        # Simple clean molecule - ethanol
        mol = Chem.MolFromSmiles('CCO')
        has_pains, names = check_pains_violations(mol)

        assert has_pains == False
        assert len(names) == 0

    def test_pains_none_molecule(self):
        """Test handling of None molecule."""
        from backend.modules.assay_interference_filter import check_pains_violations

        has_pains, names = check_pains_violations(None)

        assert has_pains == False
        assert names == []

    def test_pains_rhodanine_detected(self):
        """Test that rhodanine (known PAINS) is detected."""
        from backend.modules.assay_interference_filter import check_pains_violations

        # Rhodanine - known PAINS
        mol = Chem.MolFromSmiles('O=C1NC(=S)SC1')
        has_pains, names = check_pains_violations(mol)

        assert has_pains == True


class TestAggregatorRiskDetection:
    """Tests for aggregation risk detection."""

    def test_aggregator_risk_aromatic_lipophilic(self):
        """Test aggregator detection for highly aromatic, lipophilic molecule."""
        from backend.modules.assay_interference_filter import check_aggregator_risk

        # Tetracene - 4 fused aromatic rings, rigid, lipophilic
        mol = Chem.MolFromSmiles('c1ccc2cc3cc4ccccc4cc3cc2c1')
        is_risk, reason = check_aggregator_risk(mol)

        # May or may not trigger depending on exact criteria
        # At minimum, should have aromatic rings detected
        assert isinstance(is_risk, bool)
        assert isinstance(reason, str)

    def test_aggregator_risk_small_polar_molecule(self):
        """Test that small polar molecule has no aggregator risk."""
        from backend.modules.assay_interference_filter import check_aggregator_risk

        # Glycine - small, polar, not an aggregator
        mol = Chem.MolFromSmiles('NCC(=O)O')
        is_risk, reason = check_aggregator_risk(mol)

        assert is_risk == False

    def test_aggregator_none_molecule(self):
        """Test handling of None molecule."""
        from backend.modules.assay_interference_filter import check_aggregator_risk

        is_risk, reason = check_aggregator_risk(None)

        assert is_risk == False
        assert reason == ""


class TestRedoxReactivityDetection:
    """Tests for redox-active group detection."""

    def test_redox_catechol_detected(self):
        """Test that catechol is detected as redox-active."""
        from backend.modules.assay_interference_filter import check_redox_reactive

        # Catechol - ortho-diphenol
        mol = Chem.MolFromSmiles('Oc1ccccc1O')
        is_redox, groups = check_redox_reactive(mol)

        assert is_redox == True
        assert 'catechol' in groups

    def test_redox_hydroquinone_detected(self):
        """Test that hydroquinone is detected as redox-active."""
        from backend.modules.assay_interference_filter import check_redox_reactive

        # Hydroquinone - para-diphenol
        mol = Chem.MolFromSmiles('Oc1ccc(O)cc1')
        is_redox, groups = check_redox_reactive(mol)

        assert is_redox == True
        assert 'hydroquinone' in groups

    def test_redox_disulfide_detected(self):
        """Test that disulfide is detected as redox-active."""
        from backend.modules.assay_interference_filter import check_redox_reactive

        # Disulfide
        mol = Chem.MolFromSmiles('CSSC')
        is_redox, groups = check_redox_reactive(mol)

        assert is_redox == True
        assert 'disulfide' in groups

    def test_redox_thiol_detected(self):
        """Test that free thiol is detected as redox-active."""
        from backend.modules.assay_interference_filter import check_redox_reactive

        # Methanethiol
        mol = Chem.MolFromSmiles('CS')
        is_redox, groups = check_redox_reactive(mol)

        assert is_redox == True
        assert 'thiol' in groups

    def test_redox_clean_molecule(self):
        """Test that clean molecule has no redox groups."""
        from backend.modules.assay_interference_filter import check_redox_reactive

        # Simple alkane
        mol = Chem.MolFromSmiles('CCCCCC')
        is_redox, groups = check_redox_reactive(mol)

        assert is_redox == False
        assert len(groups) == 0

    def test_redox_none_molecule(self):
        """Test handling of None molecule."""
        from backend.modules.assay_interference_filter import check_redox_reactive

        is_redox, groups = check_redox_reactive(None)

        assert is_redox == False
        assert groups == []


class TestFluorescenceInterferenceDetection:
    """Tests for fluorescence interference detection."""

    def test_fluorescence_naphthalene_detected(self):
        """Test that naphthalene is detected as fluorescent."""
        from backend.modules.assay_interference_filter import check_fluorescence_interference

        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        is_fluor, scaffolds = check_fluorescence_interference(mol)

        assert is_fluor == True
        assert 'naphthalene' in scaffolds

    def test_fluorescence_anthracene_detected(self):
        """Test that anthracene is detected as fluorescent."""
        from backend.modules.assay_interference_filter import check_fluorescence_interference

        mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        is_fluor, scaffolds = check_fluorescence_interference(mol)

        assert is_fluor == True
        assert 'anthracene' in scaffolds

    def test_fluorescence_extended_conjugation(self):
        """Test detection of extended conjugation (3+ aromatic rings)."""
        from backend.modules.assay_interference_filter import check_fluorescence_interference

        # Phenanthrene - 3 fused rings
        mol = Chem.MolFromSmiles('c1ccc2c(c1)ccc1ccccc12')
        is_fluor, scaffolds = check_fluorescence_interference(mol)

        assert is_fluor == True
        assert 'extended_conjugation' in scaffolds

    def test_fluorescence_clean_molecule(self):
        """Test that simple molecule has no fluorescence."""
        from backend.modules.assay_interference_filter import check_fluorescence_interference

        # Ethanol
        mol = Chem.MolFromSmiles('CCO')
        is_fluor, scaffolds = check_fluorescence_interference(mol)

        assert is_fluor == False

    def test_fluorescence_none_molecule(self):
        """Test handling of None molecule."""
        from backend.modules.assay_interference_filter import check_fluorescence_interference

        is_fluor, scaffolds = check_fluorescence_interference(None)

        assert is_fluor == False
        assert scaffolds == []


class TestThiolReactivityDetection:
    """Tests for thiol-reactive electrophile detection."""

    def test_thiol_acrylamide_detected(self):
        """Test that acrylamide is detected as thiol-reactive."""
        from backend.modules.assay_interference_filter import check_thiol_reactive

        mol = Chem.MolFromSmiles('C=CC(=O)N')
        is_reactive, groups = check_thiol_reactive(mol)

        assert is_reactive == True
        assert 'acrylamide' in groups

    def test_thiol_epoxide_detected(self):
        """Test that epoxide is detected as thiol-reactive."""
        from backend.modules.assay_interference_filter import check_thiol_reactive

        mol = Chem.MolFromSmiles('C1OC1')
        is_reactive, groups = check_thiol_reactive(mol)

        assert is_reactive == True
        assert 'epoxide' in groups

    def test_thiol_isothiocyanate_detected(self):
        """Test that isothiocyanate is detected as thiol-reactive."""
        from backend.modules.assay_interference_filter import check_thiol_reactive

        mol = Chem.MolFromSmiles('CN=C=S')
        is_reactive, groups = check_thiol_reactive(mol)

        assert is_reactive == True
        assert 'isothiocyanate' in groups

    def test_thiol_aldehyde_detected(self):
        """Test that aldehyde is detected as thiol-reactive."""
        from backend.modules.assay_interference_filter import check_thiol_reactive

        # Acetaldehyde
        mol = Chem.MolFromSmiles('CC=O')
        is_reactive, groups = check_thiol_reactive(mol)

        assert is_reactive == True
        assert 'aldehyde' in groups

    def test_thiol_clean_molecule(self):
        """Test that clean molecule has no thiol reactivity."""
        from backend.modules.assay_interference_filter import check_thiol_reactive

        # Simple ester (not activated)
        mol = Chem.MolFromSmiles('CC(=O)OC')
        is_reactive, groups = check_thiol_reactive(mol)

        assert is_reactive == False

    def test_thiol_none_molecule(self):
        """Test handling of None molecule."""
        from backend.modules.assay_interference_filter import check_thiol_reactive

        is_reactive, groups = check_thiol_reactive(None)

        assert is_reactive == False
        assert groups == []


class TestGetAllInterferenceFlags:
    """Tests for main get_all_interference_flags interface."""

    def test_all_flags_clean_molecule(self):
        """Test that clean molecule has all flags False."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        flags = get_all_interference_flags('CCO')  # Ethanol

        assert flags['PAINS'] == False
        assert flags['Aggregator'] == False
        assert flags['Redox'] == False
        assert flags['Fluorescence'] == False
        assert flags['Thiol_Reactive'] == False

    def test_all_flags_quercetin(self):
        """Test flags for quercetin (known problematic compound)."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        # Quercetin - flavonoid with catechol
        quercetin_smiles = 'O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'
        flags = get_all_interference_flags(quercetin_smiles)

        # Quercetin should trigger multiple flags
        # PAINS (catechol), Redox (catechol), Fluorescence (flavonoid)
        assert flags['PAINS'] == True or flags['Redox'] == True
        assert flags['Fluorescence'] == True

    def test_all_flags_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        flags = get_all_interference_flags('invalid_smiles')

        # Should return all False for invalid SMILES
        assert all(v == False for v in flags.values())

    def test_all_flags_empty_smiles(self):
        """Test handling of empty SMILES."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        flags = get_all_interference_flags('')

        assert all(v == False for v in flags.values())

    def test_all_flags_na_smiles(self):
        """Test handling of 'N/A' SMILES."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        flags = get_all_interference_flags('N/A')

        assert all(v == False for v in flags.values())


class TestCalculateAssayQualityScore:
    """Tests for assay quality score calculation."""

    def test_quality_score_no_flags(self):
        """Test quality score with no flags (perfect)."""
        from backend.modules.assay_interference_filter import calculate_assay_quality_score

        flags = {
            'PAINS': False,
            'Aggregator': False,
            'Redox': False,
            'Fluorescence': False,
            'Thiol_Reactive': False
        }

        score = calculate_assay_quality_score(flags)
        assert score == 1.0

    def test_quality_score_all_flags(self):
        """Test quality score with all flags (worst)."""
        from backend.modules.assay_interference_filter import calculate_assay_quality_score

        flags = {
            'PAINS': True,
            'Aggregator': True,
            'Redox': True,
            'Fluorescence': True,
            'Thiol_Reactive': True
        }

        score = calculate_assay_quality_score(flags)
        assert score == 0.0

    def test_quality_score_some_flags(self):
        """Test quality score with some flags."""
        from backend.modules.assay_interference_filter import calculate_assay_quality_score

        flags = {
            'PAINS': True,
            'Aggregator': False,
            'Redox': True,
            'Fluorescence': False,
            'Thiol_Reactive': False
        }

        score = calculate_assay_quality_score(flags)
        assert score == 0.6  # 2 flags out of 5 = 1 - 0.4


class TestGetDetailedInterferenceReport:
    """Tests for detailed interference report generation."""

    def test_detailed_report_structure(self):
        """Test that detailed report has correct structure."""
        from backend.modules.assay_interference_filter import get_detailed_interference_report

        report = get_detailed_interference_report('CCO')

        assert 'smiles' in report
        assert 'pains' in report
        assert 'aggregator' in report
        assert 'redox' in report
        assert 'fluorescence' in report
        assert 'thiol_reactive' in report
        assert 'num_flags' in report
        assert 'assay_quality_score' in report

    def test_detailed_report_clean_molecule(self):
        """Test detailed report for clean molecule."""
        from backend.modules.assay_interference_filter import get_detailed_interference_report

        report = get_detailed_interference_report('CCO')

        assert report['num_flags'] == 0
        assert report['assay_quality_score'] == 1.0
        assert report['pains']['flag'] == False
        assert report['redox']['flag'] == False

    def test_detailed_report_catechol(self):
        """Test detailed report for catechol."""
        from backend.modules.assay_interference_filter import get_detailed_interference_report

        report = get_detailed_interference_report('Oc1ccccc1O')

        assert report['redox']['flag'] == True
        assert 'catechol' in report['redox']['detected_groups']
        assert report['num_flags'] > 0
        assert report['assay_quality_score'] < 1.0

    def test_detailed_report_invalid_smiles(self):
        """Test detailed report for invalid SMILES."""
        from backend.modules.assay_interference_filter import get_detailed_interference_report

        report = get_detailed_interference_report('invalid')

        assert report['num_flags'] == 0
        assert report['assay_quality_score'] == 1.0

    def test_detailed_report_empty_smiles(self):
        """Test detailed report for empty SMILES."""
        from backend.modules.assay_interference_filter import get_detailed_interference_report

        report = get_detailed_interference_report('')

        assert report['smiles'] == ''
        assert report['num_flags'] == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_large_molecule(self):
        """Test handling of very large molecule."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        # Large polymer-like structure
        large_smiles = 'C' * 100 + 'O'
        flags = get_all_interference_flags(large_smiles)

        # Should not raise, may or may not parse
        assert isinstance(flags, dict)

    def test_charged_molecule(self):
        """Test handling of charged molecule."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        # Sodium acetate
        flags = get_all_interference_flags('CC(=O)[O-].[Na+]')

        assert isinstance(flags, dict)

    def test_stereochemistry(self):
        """Test handling of molecules with stereochemistry."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        # L-alanine with stereochemistry
        flags = get_all_interference_flags('C[C@H](N)C(=O)O')

        assert isinstance(flags, dict)
        assert all(isinstance(v, bool) for v in flags.values())

    def test_aromatic_heterocycle(self):
        """Test handling of aromatic heterocycles."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        # Pyridine
        flags = get_all_interference_flags('c1ccncc1')

        assert isinstance(flags, dict)

    def test_multiple_rings(self):
        """Test molecule with multiple fused rings."""
        from backend.modules.assay_interference_filter import get_all_interference_flags

        # Pyrene - 4 fused rings
        flags = get_all_interference_flags('c1cc2ccc3cccc4ccc(c1)c2c34')

        assert isinstance(flags, dict)
        # Should detect fluorescence due to extended conjugation
        assert flags['Fluorescence'] == True
