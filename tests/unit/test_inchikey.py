"""
Unit tests for InChIKey generation and duplicate detection utilities.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestInChIKeyGeneration:
    """Tests for InChIKey generation functionality."""

    def test_generate_inchikey_valid_smiles(self):
        """Test InChIKey generation with valid SMILES."""
        from backend.services.job_service import generate_inchikey

        # Aspirin SMILES
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        result = generate_inchikey(smiles)

        assert result is not None
        assert len(result) == 27  # InChIKey is always 27 characters
        assert "-" in result  # InChIKey contains hyphens

    def test_generate_inchikey_caffeine(self):
        """Test InChIKey generation with caffeine SMILES."""
        from backend.services.job_service import generate_inchikey

        smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        result = generate_inchikey(smiles)

        assert result is not None
        assert len(result) == 27

    def test_generate_inchikey_same_structure_same_key(self):
        """Test that same structure always produces same InChIKey."""
        from backend.services.job_service import generate_inchikey

        # Different SMILES representations of same molecule (ethanol)
        smiles1 = "CCO"
        smiles2 = "OCC"  # Same molecule, different SMILES notation
        smiles3 = "C(C)O"  # Another notation

        key1 = generate_inchikey(smiles1)
        key2 = generate_inchikey(smiles2)
        key3 = generate_inchikey(smiles3)

        # All should produce the same InChIKey
        assert key1 == key2 == key3

    def test_generate_inchikey_empty_smiles(self):
        """Test InChIKey generation with empty SMILES."""
        from backend.services.job_service import generate_inchikey

        assert generate_inchikey("") is None
        assert generate_inchikey("   ") is None
        assert generate_inchikey(None) is None

    def test_generate_inchikey_invalid_smiles(self):
        """Test InChIKey generation with invalid SMILES."""
        from backend.services.job_service import generate_inchikey

        # Invalid SMILES strings
        assert generate_inchikey("invalid") is None
        assert generate_inchikey("XYZ123") is None
        assert generate_inchikey("not_a_molecule") is None

    def test_generate_inchikey_different_structures(self):
        """Test that different structures produce different InChIKeys."""
        from backend.services.job_service import generate_inchikey

        # Different molecules
        ethanol = generate_inchikey("CCO")
        methanol = generate_inchikey("CO")
        propanol = generate_inchikey("CCCO")

        assert ethanol != methanol
        assert ethanol != propanol
        assert methanol != propanol


class TestCanonicalSmilesGeneration:
    """Tests for canonical SMILES generation."""

    def test_generate_canonical_smiles_valid(self):
        """Test canonical SMILES generation."""
        from backend.services.job_service import generate_canonical_smiles

        smiles = "CCO"
        result = generate_canonical_smiles(smiles)

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_canonical_smiles_normalizes(self):
        """Test that canonical SMILES normalizes different notations."""
        from backend.services.job_service import generate_canonical_smiles

        # Different notations for same molecule
        smiles1 = "CCO"
        smiles2 = "OCC"

        canonical1 = generate_canonical_smiles(smiles1)
        canonical2 = generate_canonical_smiles(smiles2)

        # Should produce same canonical form
        assert canonical1 == canonical2

    def test_generate_canonical_smiles_empty(self):
        """Test canonical SMILES with empty input."""
        from backend.services.job_service import generate_canonical_smiles

        assert generate_canonical_smiles("") is None
        assert generate_canonical_smiles("   ") is None
        assert generate_canonical_smiles(None) is None

    def test_generate_canonical_smiles_invalid(self):
        """Test canonical SMILES with invalid input."""
        from backend.services.job_service import generate_canonical_smiles

        assert generate_canonical_smiles("invalid") is None
        assert generate_canonical_smiles("XYZ123") is None


class TestInChIKeyEdgeCases:
    """Tests for InChIKey edge cases."""

    def test_generate_inchikey_complex_molecule(self):
        """Test InChIKey generation with a complex molecule."""
        from backend.services.job_service import generate_inchikey

        # Quercetin - a more complex flavonoid
        smiles = "O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C=C2O"
        result = generate_inchikey(smiles)

        assert result is not None
        assert len(result) == 27

    def test_generate_inchikey_with_stereochemistry(self):
        """Test InChIKey generation with stereochemistry."""
        from backend.services.job_service import generate_inchikey

        # L-Alanine with stereochemistry
        smiles = "C[C@H](N)C(=O)O"
        result = generate_inchikey(smiles)

        assert result is not None
        assert len(result) == 27


class TestCompoundNameSanitization:
    """Tests for compound name sanitization."""

    def test_sanitize_compound_name_basic(self):
        """Test basic compound name sanitization."""
        from backend.core import sanitize_compound_name

        # Basic names should be unchanged
        assert sanitize_compound_name("Aspirin") == "Aspirin"
        assert sanitize_compound_name("Ibuprofen") == "Ibuprofen"

    def test_sanitize_compound_name_spaces(self):
        """Test sanitization of names with spaces."""
        from backend.core import sanitize_compound_name

        result = sanitize_compound_name("Vitamin B12")
        assert " " not in result
        assert "_" in result

    def test_sanitize_compound_name_special_chars(self):
        """Test sanitization of names with special characters."""
        from backend.core import sanitize_compound_name

        # Special characters should be replaced with underscore
        result = sanitize_compound_name("Test/Compound")
        assert "/" not in result

        result = sanitize_compound_name("Test\\Compound")
        assert "\\" not in result

        result = sanitize_compound_name("Test@Compound#123")
        assert "@" not in result
        assert "#" not in result

    def test_sanitize_compound_name_empty(self):
        """Test sanitization of empty names."""
        from backend.core import sanitize_compound_name

        assert sanitize_compound_name("") == "unnamed_compound"
        assert sanitize_compound_name("   ") == "unnamed_compound"

    def test_sanitize_compound_name_long(self):
        """Test sanitization of very long names."""
        from backend.core import sanitize_compound_name

        # Create a very long name
        long_name = "A" * 200
        result = sanitize_compound_name(long_name)

        # Should be truncated to max length (100 by default)
        assert len(result) <= 100

    def test_sanitize_compound_name_unicode(self):
        """Test sanitization of names with unicode characters."""
        from backend.core import sanitize_compound_name

        result = sanitize_compound_name("Café-Molecule")
        # Should handle unicode gracefully
        assert isinstance(result, str)
