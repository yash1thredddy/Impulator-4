"""Tests for frontend.ui.components.charts — Plotly chart helpers."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

from backend.modules.imp_gmm import cluster_membership, fit_gmm
from frontend.ui.components.charts import (
    HEATMAP_TARGET_LABEL_MAXLEN,
    HEATMAP_TOP_K,
    create_bar_chart,
    create_bioactivity_heatmap,
    create_gmm_density_overlay,
    create_gmm_probability_bar,
)


def _make_model(K: int, seed: int = 0):
    """Return a fitted ``GaussianMixture`` on a deterministic synth corpus."""
    rng = np.random.default_rng(seed=seed)
    # 60-sample three-cluster cloud — enough for K up to 6.
    scores = np.concatenate(
        [
            rng.normal(20, 5, 20),
            rng.normal(50, 5, 20),
            rng.normal(80, 5, 20),
        ]
    )
    return scores, fit_gmm(scores, n_components=K)


# =============================================================================
# create_gmm_density_overlay
# =============================================================================


class TestCreateGmmDensityOverlay:
    """Smoke tests for create_gmm_density_overlay (GMM-06)."""

    def test_returns_figure_instance(self):
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        assert isinstance(fig, go.Figure)

    def test_has_one_histogram_trace(self):
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        hist_count = sum(1 for t in fig.data if t.type == "histogram")
        assert hist_count == 1

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
    def test_has_k_density_curve_traces_for_k_components(self, K):
        scores, model = _make_model(K)
        fig = create_gmm_density_overlay(scores, model)
        scatter_count = sum(1 for t in fig.data if t.type == "scatter")
        assert scatter_count == K

    def test_no_vline_shapes_drawn(self):
        # The corpus is built from the active compound's neighborhood, so
        # marking "this compound" on the histogram of its own neighbors would
        # be visually circular. The chart MUST NOT draw such a vline.
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        shapes = list(fig.layout.shapes)
        assert len(shapes) == 0

    def test_xaxis_range_is_0_100(self):
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        assert tuple(fig.layout.xaxis.range) == (0, 100)

    def test_has_twin_yaxis(self):
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        assert fig.layout.yaxis2 is not None
        assert fig.layout.yaxis2.overlaying == "y"
        assert fig.layout.yaxis2.side == "right"

    def test_applies_impulator_theme(self):
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        # apply_impulator_theme sets hoverlabel.namelength = -1
        assert fig.layout.hoverlabel.namelength == -1

    def test_density_traces_use_set2_palette(self):
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        for i, trace in enumerate(scatter_traces):
            assert trace.line.color == px.colors.qualitative.Set2[i]

    def test_height_is_280(self):
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        assert fig.layout.height == 280

    def test_density_traces_have_translucent_fill(self):
        scores, model = _make_model(3)
        fig = create_gmm_density_overlay(scores, model)
        scatter_traces = [t for t in fig.data if t.type == "scatter"]
        for i, trace in enumerate(scatter_traces):
            assert trace.fill == "tozeroy"
            # Set2 returns rgb(r,g,b) strings; fillcolor builds rgba(r, g, b, 0.15)
            r, g, b = px.colors.unlabel_rgb(px.colors.qualitative.Set2[i])
            assert trace.fillcolor == (
                f"rgba({int(r)}, {int(g)}, {int(b)}, 0.15)"
            )


# =============================================================================
# create_gmm_probability_bar
# =============================================================================


class TestCreateGmmProbabilityBar:
    """Smoke tests for create_gmm_probability_bar (GMM-04 visual half)."""

    def test_returns_figure_instance(self):
        fig = create_gmm_probability_bar(
            [0.5, 0.3, 0.2], cluster_means=[20.0, 50.0, 80.0]
        )
        assert isinstance(fig, go.Figure)

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
    def test_has_k_bar_traces(self, K):
        mem = [1.0 / K] * K
        means = list(np.linspace(10, 90, K))
        fig = create_gmm_probability_bar(mem, cluster_means=means)
        assert len(fig.data) == K
        for t in fig.data:
            assert t.type == "bar"

    def test_barmode_is_stack(self):
        fig = create_gmm_probability_bar(
            [0.5, 0.3, 0.2], cluster_means=[20.0, 50.0, 80.0]
        )
        assert fig.layout.barmode == "stack"

    def test_orientation_is_horizontal(self):
        fig = create_gmm_probability_bar(
            [0.5, 0.3, 0.2], cluster_means=[20.0, 50.0, 80.0]
        )
        for t in fig.data:
            assert t.orientation == "h"

    def test_xaxis_range_is_0_to_1(self):
        fig = create_gmm_probability_bar(
            [0.5, 0.3, 0.2], cluster_means=[20.0, 50.0, 80.0]
        )
        assert tuple(fig.layout.xaxis.range) == (0, 1)

    def test_xaxis_tickformat_is_percent(self):
        fig = create_gmm_probability_bar(
            [0.5, 0.3, 0.2], cluster_means=[20.0, 50.0, 80.0]
        )
        assert fig.layout.xaxis.tickformat == ".0%"

    def test_text_is_integer_percent_no_decimals(self):
        fig = create_gmm_probability_bar(
            [0.42, 0.35, 0.23], cluster_means=[20.0, 50.0, 80.0]
        )
        texts = [t.text for t in fig.data]
        assert texts == ["42%", "35%", "23%"]

    def test_height_is_140(self):
        # Bumped from 80 → 140 so the horizontal legend at y=-0.6 has room
        # inside the figure bounds (the 80px version was clipping percentage
        # labels at the bottom of the page).
        fig = create_gmm_probability_bar(
            [0.5, 0.3, 0.2], cluster_means=[20.0, 50.0, 80.0]
        )
        assert fig.layout.height == 140

    def test_segment_colors_match_density_chart_for_same_k(self):
        """Cross-chart consistency: k → identical Set2[k] in both figures."""
        scores, model = _make_model(3)
        mem = cluster_membership(model, 50.0)
        means = np.sort(model.means_.flatten())

        density = create_gmm_density_overlay(scores, model)
        prob = create_gmm_probability_bar(mem, cluster_means=means)

        density_scatters = [t for t in density.data if t.type == "scatter"]
        for k in range(3):
            assert density_scatters[k].line.color == prob.data[k].marker.color

    def test_defensive_normalization_handles_imperfect_sum(self):
        # Sum slightly above 1.0; after defensive normalization texts should
        # sum to exactly 100.
        fig = create_gmm_probability_bar(
            [0.4200000001, 0.3499999998, 0.18, 0.05],
            cluster_means=[10.0, 30.0, 60.0, 90.0],
        )
        total = sum(int(t.text.rstrip("%")) for t in fig.data)
        assert total == 100

    def test_applies_impulator_theme(self):
        fig = create_gmm_probability_bar(
            [0.5, 0.3, 0.2], cluster_means=[20.0, 50.0, 80.0]
        )
        assert fig.layout.hoverlabel.namelength == -1


# =============================================================================
# create_bioactivity_heatmap (COLL-01 / D-01 headline viz, D-10 top-K)
# =============================================================================


def _make_matrix(n_members: int = 3, n_targets: int = 5, seed: int = 0):
    """Return a member×target activity DataFrame (members on the index)."""
    rng = np.random.default_rng(seed=seed)
    members = [f"Member {i}" for i in range(n_members)]
    targets = [f"Target {j}" for j in range(n_targets)]
    data = rng.uniform(0.0, 100.0, size=(n_members, n_targets))
    return pd.DataFrame(data, index=members, columns=targets)


class TestCreateBioactivityHeatmap:
    """Smoke + contract tests for create_bioactivity_heatmap (D-01 / D-10)."""

    def test_returns_figure_instance(self):
        fig = create_bioactivity_heatmap(_make_matrix())
        assert isinstance(fig, go.Figure)

    def test_builds_a_single_heatmap_trace(self):
        fig = create_bioactivity_heatmap(_make_matrix())
        heatmaps = [t for t in fig.data if t.type == "heatmap"]
        assert len(heatmaps) == 1

    def test_members_are_rows_targets_are_columns(self):
        matrix = _make_matrix(n_members=3, n_targets=4)
        fig = create_bioactivity_heatmap(matrix)
        trace = fig.data[0]
        assert set(trace.y) == set(matrix.index)
        assert set(trace.x) == set(matrix.columns)

    def test_uses_viridis_sequential_scale(self):
        fig = create_bioactivity_heatmap(_make_matrix())
        # Plotly expands "Viridis" into its [(stop, color), ...] stop list;
        # the canonical Viridis endpoints are dark purple -> yellow.
        colorscale = fig.data[0].colorscale
        assert colorscale is not None
        first_color = colorscale[0][1].lower()
        last_color = colorscale[-1][1].lower()
        assert first_color == "#440154"  # Viridis start (dark purple)
        assert last_color == "#fde725"  # Viridis end (yellow)

    def test_applies_impulator_theme(self):
        fig = create_bioactivity_heatmap(_make_matrix())
        assert fig.layout.hoverlabel.namelength == -1

    def test_height_grows_with_member_count(self):
        small = create_bioactivity_heatmap(_make_matrix(n_members=2))
        large = create_bioactivity_heatmap(_make_matrix(n_members=10))
        assert large.layout.height > small.layout.height

    def test_caps_targets_at_top_k(self):
        # More targets than the cap -> only HEATMAP_TOP_K columns retained.
        matrix = _make_matrix(n_members=3, n_targets=HEATMAP_TOP_K + 8)
        fig = create_bioactivity_heatmap(matrix)
        assert len(fig.data[0].x) == HEATMAP_TOP_K

    def test_top_k_override_respected(self):
        matrix = _make_matrix(n_members=3, n_targets=10)
        fig = create_bioactivity_heatmap(matrix, top_k=4)
        assert len(fig.data[0].x) == 4

    def test_targets_ranked_by_member_hit_count(self):
        # Target "Hot" hit by all 3 members; "Cold" hit by only 1. With top_k=1
        # only the most-hit target survives.
        matrix = pd.DataFrame(
            {
                "Cold": [10.0, 0.0, 0.0],
                "Hot": [5.0, 5.0, 5.0],
            },
            index=["A", "B", "C"],
        )
        fig = create_bioactivity_heatmap(matrix, top_k=1)
        assert list(fig.data[0].x) == ["Hot"]

    def test_long_target_label_truncated_in_ticktext_full_name_in_data(self):
        long_name = "X" * (HEATMAP_TARGET_LABEL_MAXLEN + 20)
        matrix = pd.DataFrame({long_name: [1.0, 2.0]}, index=["A", "B"])
        fig = create_bioactivity_heatmap(matrix)
        # Full name preserved in trace x (drives the hover tooltip)...
        assert long_name in list(fig.data[0].x)
        # ...but the displayed ticktext is truncated.
        ticktext = list(fig.layout.xaxis.ticktext)
        assert any(len(t) <= HEATMAP_TARGET_LABEL_MAXLEN for t in ticktext)
        assert all(t != long_name for t in ticktext)

    def test_empty_matrix_returns_empty_themed_figure(self):
        fig = create_bioactivity_heatmap(pd.DataFrame())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        # Still themed so the caller can branch without losing styling.
        assert fig.layout.hoverlabel.namelength == -1

    def test_single_member_still_renders_one_row(self):
        matrix = _make_matrix(n_members=1, n_targets=3)
        fig = create_bioactivity_heatmap(matrix)
        assert len(fig.data) == 1
        assert len(fig.data[0].y) == 1


# =============================================================================
# create_bar_chart — grouped barmode (collection member comparison)
# =============================================================================


class TestCreateBarChartBarmode:
    """create_bar_chart gains barmode='group' without breaking callers."""

    def _df(self):
        return pd.DataFrame(
            {
                "compound": ["A", "A", "B", "B"],
                "metric": ["x", "y", "x", "y"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def test_returns_figure_instance(self):
        fig = create_bar_chart(self._df(), x_col="metric", y_col="value")
        assert isinstance(fig, go.Figure)

    def test_default_barmode_is_group(self):
        fig = create_bar_chart(
            self._df(), x_col="metric", y_col="value", color_col="compound"
        )
        assert fig.layout.barmode == "group"

    def test_stack_barmode_still_available(self):
        fig = create_bar_chart(
            self._df(),
            x_col="metric",
            y_col="value",
            color_col="compound",
            barmode="stack",
        )
        assert fig.layout.barmode == "stack"
