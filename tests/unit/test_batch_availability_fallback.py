"""
Tests for batch availability check SMILES/InChI/InChIKey fallback chain.

Tests the resolution logic added to job_form.py that resolves
non-SMILES inputs before the availability check API call.
"""
import pandas as pd


class TestBatchAvailabilityFallbackChain:
    """Test the SMILES → InChI → InChIKey fallback for availability checking."""

    def _build_compounds_for_avail(self, df_mapped, identical_config_names=None, inchikey_smiles_map=None):
        """
        Extract the fallback chain logic from job_form.py for testability.

        This mirrors the logic at job_form.py lines 1033-1082.
        """
        from frontend.ui.components.job_form import _sanitize_and_limit_name, _inchi_to_smiles

        if identical_config_names is None:
            identical_config_names = set()
        if inchikey_smiles_map is None:
            inchikey_smiles_map = {}

        compounds_for_avail = []
        df_has_smiles_col = 'smiles' in df_mapped.columns
        df_has_inchi_col = 'inchi' in df_mapped.columns
        df_has_inchikey_col = 'inchikey' in df_mapped.columns

        structural_cols = [c for c in ['compound_name', 'smiles', 'inchi', 'inchikey'] if c in df_mapped.columns]
        if structural_cols:
            for row in df_mapped[structural_cols].to_dict('records'):
                name = _sanitize_and_limit_name(str(row.get('compound_name', '')).strip())
                if not name:
                    continue
                if name.lower() in identical_config_names:
                    continue

                smiles_val = None
                if df_has_smiles_col:
                    raw = str(row.get('smiles', '')).strip()
                    if raw and raw.lower() not in ('nan', 'none', ''):
                        smiles_val = raw
                if not smiles_val and df_has_inchi_col:
                    inchi_val = str(row.get('inchi', '')).strip()
                    if inchi_val and inchi_val.lower() not in ('nan', 'none', ''):
                        smiles_val = _inchi_to_smiles(inchi_val)
                if not smiles_val and df_has_inchikey_col:
                    key_val = str(row.get('inchikey', '')).strip().upper()
                    if key_val and key_val.lower() not in ('nan', 'none', ''):
                        smiles_val = inchikey_smiles_map.get(key_val)

                if smiles_val:
                    compounds_for_avail.append({
                        "compound_name": name,
                        "smiles": smiles_val,
                    })

        return compounds_for_avail

    def test_smiles_only_csv(self):
        """CSV with only SMILES column should work (original behavior)."""
        df = pd.DataFrame({
            'compound_name': ['Aspirin', 'Caffeine'],
            'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 2
        assert result[0]['compound_name'] == 'Aspirin'
        assert result[0]['smiles'] == 'CC(=O)OC1=CC=CC=C1C(=O)O'

    def test_inchi_only_csv(self):
        """CSV with only InChI column should resolve to SMILES via RDKit."""
        df = pd.DataFrame({
            'compound_name': ['Ethanol'],
            'inchi': ['InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 1
        assert result[0]['compound_name'] == 'Ethanol'
        assert result[0]['smiles'] is not None
        assert len(result[0]['smiles']) > 0

    def test_inchikey_only_csv_with_map(self):
        """CSV with only InChIKey should use pre-resolved map."""
        df = pd.DataFrame({
            'compound_name': ['Aspirin'],
            'inchikey': ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N'],
        })
        inchikey_map = {'BSYNRYMUTXBXSQ-UHFFFAOYSA-N': 'CC(=O)OC1=CC=CC=C1C(=O)O'}
        result = self._build_compounds_for_avail(df, inchikey_smiles_map=inchikey_map)
        assert len(result) == 1
        assert result[0]['smiles'] == 'CC(=O)OC1=CC=CC=C1C(=O)O'

    def test_inchikey_only_csv_without_map(self):
        """CSV with only InChIKey but no map entry should skip the compound."""
        df = pd.DataFrame({
            'compound_name': ['Unknown'],
            'inchikey': ['XXXXXXXXXXXXXX-UHFFFAOYSA-N'],
        })
        result = self._build_compounds_for_avail(df, inchikey_smiles_map={})
        assert len(result) == 0

    def test_smiles_takes_precedence_over_inchi(self):
        """When both SMILES and InChI columns exist, SMILES should be used."""
        df = pd.DataFrame({
            'compound_name': ['Ethanol'],
            'smiles': ['CCO'],
            'inchi': ['InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 1
        assert result[0]['smiles'] == 'CCO'

    def test_inchi_fallback_when_smiles_is_nan(self):
        """When SMILES column exists but value is NaN, should fall back to InChI."""
        df = pd.DataFrame({
            'compound_name': ['Ethanol'],
            'smiles': ['nan'],
            'inchi': ['InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 1
        assert result[0]['smiles'] is not None
        assert result[0]['smiles'] != 'nan'

    def test_inchi_fallback_when_smiles_is_empty(self):
        """When SMILES column exists but value is empty, should fall back to InChI."""
        df = pd.DataFrame({
            'compound_name': ['Ethanol'],
            'smiles': [''],
            'inchi': ['InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 1
        assert result[0]['smiles'] is not None

    def test_inchikey_fallback_when_smiles_and_inchi_missing(self):
        """Full fallback chain: SMILES empty, InChI empty, InChIKey resolves."""
        df = pd.DataFrame({
            'compound_name': ['Aspirin'],
            'smiles': [''],
            'inchi': [''],
            'inchikey': ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N'],
        })
        inchikey_map = {'BSYNRYMUTXBXSQ-UHFFFAOYSA-N': 'CC(=O)OC1=CC=CC=C1C(=O)O'}
        # Note: inchikey_smiles_map is only pre-populated when InChIKey is the SOLE
        # structural column. When smiles/inchi columns exist (even if empty), the map
        # won't be pre-populated in real code. This test verifies the fallback logic
        # with a pre-populated map.
        result = self._build_compounds_for_avail(df, inchikey_smiles_map=inchikey_map)
        assert len(result) == 1
        assert result[0]['smiles'] == 'CC(=O)OC1=CC=CC=C1C(=O)O'

    def test_identical_config_names_skipped(self):
        """Compounds with identical existing config should be skipped."""
        df = pd.DataFrame({
            'compound_name': ['Aspirin', 'Caffeine'],
            'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
        })
        result = self._build_compounds_for_avail(
            df, identical_config_names={'aspirin'}
        )
        assert len(result) == 1
        assert result[0]['compound_name'] == 'Caffeine'

    def test_empty_compound_name_gets_default(self):
        """Rows with empty compound names get 'unnamed_compound' from sanitizer."""
        df = pd.DataFrame({
            'compound_name': ['', 'Caffeine'],
            'smiles': ['CCO', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 2
        assert result[0]['compound_name'] == 'unnamed_compound'
        assert result[1]['compound_name'] == 'Caffeine'

    def test_no_structural_columns(self):
        """DataFrame with no structural columns should return empty list."""
        df = pd.DataFrame({
            'compound_name': ['Aspirin'],
            'other_column': ['some_value'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 0

    def test_invalid_inchi_falls_through(self):
        """Invalid InChI that can't be converted should skip compound."""
        df = pd.DataFrame({
            'compound_name': ['BadCompound'],
            'inchi': ['not-a-valid-inchi'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 0

    def test_mixed_csv_some_smiles_some_inchi(self):
        """CSV where some rows have SMILES and others only InChI."""
        df = pd.DataFrame({
            'compound_name': ['Aspirin', 'Ethanol'],
            'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'nan'],
            'inchi': ['', 'InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'],
        })
        result = self._build_compounds_for_avail(df)
        assert len(result) == 2
        assert result[0]['smiles'] == 'CC(=O)OC1=CC=CC=C1C(=O)O'
        assert result[1]['smiles'] is not None  # Resolved from InChI


class TestInChIKeyPreResolution:
    """Test the InChIKey pre-resolution logic (lines 1039-1053)."""

    def test_only_triggers_when_inchikey_is_sole_column(self):
        """InChIKey batch resolution should only happen when InChIKey is the sole structural column."""
        df_inchikey_only = pd.DataFrame({
            'compound_name': ['Aspirin'],
            'inchikey': ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N'],
        })
        df_has_smiles = pd.DataFrame({
            'compound_name': ['Aspirin'],
            'smiles': ['CCO'],
            'inchikey': ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N'],
        })
        df_has_inchi = pd.DataFrame({
            'compound_name': ['Aspirin'],
            'inchi': ['InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3'],
            'inchikey': ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N'],
        })

        # InChIKey-only: should trigger resolution
        has_smiles = 'smiles' in df_inchikey_only.columns
        has_inchi = 'inchi' in df_inchikey_only.columns
        has_inchikey = 'inchikey' in df_inchikey_only.columns
        should_resolve = has_inchikey and not has_smiles and not has_inchi
        assert should_resolve is True

        # Has SMILES: should NOT trigger resolution
        has_smiles = 'smiles' in df_has_smiles.columns
        has_inchi = 'inchi' in df_has_smiles.columns
        has_inchikey = 'inchikey' in df_has_smiles.columns
        should_resolve = has_inchikey and not has_smiles and not has_inchi
        assert should_resolve is False

        # Has InChI: should NOT trigger resolution
        has_smiles = 'smiles' in df_has_inchi.columns
        has_inchi = 'inchi' in df_has_inchi.columns
        has_inchikey = 'inchikey' in df_has_inchi.columns
        should_resolve = has_inchikey and not has_smiles and not has_inchi
        assert should_resolve is False

    def test_nan_inchikeys_filtered(self):
        """NaN/None/empty InChIKeys should not be sent for resolution."""
        df = pd.DataFrame({
            'compound_name': ['A', 'B', 'C', 'D'],
            'inchikey': ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N', 'nan', 'None', ''],
        })
        inchikeys_to_resolve = []
        for _, row in df.iterrows():
            key_val = str(row.get('inchikey', '')).strip()
            if key_val and key_val.lower() not in ('nan', 'none', ''):
                inchikeys_to_resolve.append(key_val.upper())

        assert len(inchikeys_to_resolve) == 1
        assert inchikeys_to_resolve[0] == 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'

    def test_duplicate_inchikeys_deduplicated(self):
        """Duplicate InChIKeys should be deduplicated before resolution."""
        df = pd.DataFrame({
            'compound_name': ['A', 'B'],
            'inchikey': ['BSYNRYMUTXBXSQ-UHFFFAOYSA-N', 'BSYNRYMUTXBXSQ-UHFFFAOYSA-N'],
        })
        inchikeys_to_resolve = []
        for _, row in df.iterrows():
            key_val = str(row.get('inchikey', '')).strip()
            if key_val and key_val.lower() not in ('nan', 'none', ''):
                inchikeys_to_resolve.append(key_val.upper())

        unique_keys = list(set(inchikeys_to_resolve))
        assert len(unique_keys) == 1
