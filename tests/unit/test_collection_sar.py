"""Unit tests for the pure SAR module: Tanimoto matrix + activity cliffs.

No Streamlit, no IO. RDKit is exercised on real, parseable SMILES.
"""

import numpy as np

from frontend.ui.components.collection_sar import activity_cliffs, tanimoto_matrix


class TestTanimotoMatrix:
    def test_symmetric_with_unit_diagonal(self):
        smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
        m = tanimoto_matrix(smiles)
        assert m.shape == (3, 3)
        # Unit diagonal.
        assert np.allclose(np.diag(m), 1.0)
        # Symmetric.
        assert np.allclose(m, m.T)

    def test_identical_structures_score_one(self):
        # Two spellings of ethanol -> Tanimoto 1.0 off-diagonal.
        m = tanimoto_matrix(["CCO", "OCC"])
        assert m[0, 1] == 1.0
        assert m[1, 0] == 1.0

    def test_known_pair_value(self):
        # ethanol vs propanol -> 0.5556 (verified against RDKit 2025.09.6).
        m = tanimoto_matrix(["CCO", "CCCO"])
        assert m[0, 1] == m[1, 0]
        assert abs(m[0, 1] - 0.5556) < 1e-3

    def test_dissimilar_pair_near_zero(self):
        # ethanol vs benzene share no Morgan bits.
        m = tanimoto_matrix(["CCO", "c1ccccc1"])
        assert m[0, 1] == 0.0

    def test_none_mol_handled_gracefully(self):
        # A malformed SMILES yields a None mol; its row/col stays at the
        # identity default (diagonal 1.0, off-diagonal 0.0) without crashing.
        m = tanimoto_matrix(["CCO", "not-a-smiles", "OCC"])
        assert m.shape == (3, 3)
        assert np.allclose(np.diag(m), 1.0)
        # The two valid ethanol spellings still match each other.
        assert m[0, 2] == 1.0
        # The bad SMILES contributes no similarity.
        assert m[0, 1] == 0.0
        assert m[1, 2] == 0.0
        assert np.allclose(m, m.T)

    def test_empty_list_returns_empty_matrix(self):
        m = tanimoto_matrix([])
        assert m.shape == (0, 0)


class TestActivityCliffs:
    def test_high_sim_large_delta_flagged(self):
        # Two identical structures (sim 1.0) with a large IMP gap -> cliff.
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        imp = [10.0, 90.0]
        cliffs = activity_cliffs(matrix, imp, sim_threshold=0.85, delta_threshold=20.0)
        assert (0, 1) in cliffs

    def test_high_sim_small_delta_not_flagged(self):
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        imp = [50.0, 55.0]
        cliffs = activity_cliffs(matrix, imp, sim_threshold=0.85, delta_threshold=20.0)
        assert cliffs == []

    def test_low_sim_large_delta_not_flagged(self):
        matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
        imp = [10.0, 90.0]
        cliffs = activity_cliffs(matrix, imp, sim_threshold=0.85, delta_threshold=20.0)
        assert cliffs == []

    def test_returns_upper_triangle_pairs_only(self):
        # No (j, i) duplicates and no (i, i) self-pairs.
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
        imp = [0.0, 100.0]
        cliffs = activity_cliffs(matrix, imp, sim_threshold=0.85, delta_threshold=20.0)
        assert cliffs == [(0, 1)]

    def test_multiple_cliffs(self):
        matrix = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        imp = [10.0, 90.0, 95.0]
        cliffs = activity_cliffs(matrix, imp, sim_threshold=0.85, delta_threshold=20.0)
        # (0,1) high-sim+large-delta; (0,2) high-sim+large-delta; (1,2) low-sim.
        assert (0, 1) in cliffs
        assert (0, 2) in cliffs
        assert (1, 2) not in cliffs
