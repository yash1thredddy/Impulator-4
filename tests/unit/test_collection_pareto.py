"""Unit tests for the pure Pareto non-dominated sort (no Streamlit, no IO)."""

import numpy as np

from frontend.ui.components.collection_pareto import pareto_front


class TestParetoFront:
    def test_single_point_is_on_front(self):
        pts = np.array([[1.0, 1.0]])
        mask = pareto_front(pts)
        assert mask.tolist() == [True]

    def test_known_toy_set_exact_mask(self):
        # HIGHER-is-better on both axes.
        #   A (3,1) non-dominated (best x)
        #   B (1,3) non-dominated (best y)
        #   C (2,2) non-dominated (no point dominates it)
        #   D (1,1) dominated by A, B and C
        #   E (2,1) dominated by A and C
        pts = np.array(
            [
                [3.0, 1.0],  # A
                [1.0, 3.0],  # B
                [2.0, 2.0],  # C
                [1.0, 1.0],  # D dominated
                [2.0, 1.0],  # E dominated
            ]
        )
        mask = pareto_front(pts)
        assert mask.tolist() == [True, True, True, False, False]

    def test_strict_domination_excludes_inner_point(self):
        pts = np.array([[5.0, 5.0], [1.0, 1.0]])
        mask = pareto_front(pts)
        assert mask.tolist() == [True, False]

    def test_equal_points_both_kept(self):
        # Identical points do not strictly dominate one another -> both on front.
        pts = np.array([[2.0, 2.0], [2.0, 2.0]])
        mask = pareto_front(pts)
        assert mask.tolist() == [True, True]

    def test_three_axis_front(self):
        pts = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],  # dominated by every other
            ]
        )
        mask = pareto_front(pts)
        assert mask.tolist() == [True, True, True, False]

    def test_returns_boolean_array_of_correct_length(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 0.5]])
        mask = pareto_front(pts)
        assert mask.dtype == bool
        assert mask.shape == (3,)

    def test_lower_is_better_axis_flipped_upstream(self):
        # Caller flips sign of lower-is-better axes before calling. Here axis 1 is
        # "lower-is-better" (e.g. toxicity), so we negate it; the survivor is the
        # point with highest x AND lowest original y.
        x = np.array([3.0, 1.0, 2.0])
        tox = np.array([0.1, 0.9, 0.5])  # lower is better
        pts = np.column_stack([x, -tox])
        mask = pareto_front(pts)
        # (3, -0.1) dominates (2, -0.5) and (1, -0.9) on both flipped axes.
        assert mask.tolist() == [True, False, False]
