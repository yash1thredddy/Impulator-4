"""Tests for frontend.ui.components.charts — Plotly chart helpers."""

import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

from backend.modules.imp_gmm import cluster_membership, fit_gmm
from frontend.ui.components.charts import (
    HEATMAP_TARGET_LABEL_MAXLEN,
    HEATMAP_TOP_K,
    IMP_COMPONENT_COLS,
    create_bar_chart,
    create_bioactivity_heatmap,
    create_chemical_space_scatter,
    create_compare_radar,
    create_decision_map,
    create_efficiency_plane,
    create_gmm_density_overlay,
    create_gmm_probability_bar,
    create_imp_component_breakdown,
    create_imp_component_radar,
    create_imp_contribution_bar,
    create_pareto_plot,
    create_promise_contribution_bar,
    create_sar_matrix,
)

# Path to the Plan-01 real-schema collection toy fixture (one row per
# (member, target); per-member factories receive a member-deduplicated frame).
_FIXTURE = (
    pathlib.Path(__file__).parent / "fixtures" / "collection_toy_combined.csv"
)


def _load_member_frame() -> pd.DataFrame:
    """Load the toy fixture collapsed to one row per member.

    The decision map / efficiency plane factories plot one point per member,
    so collapse the (member, target) grain to per-member means for the
    coordinate columns while keeping the first SMILES / name.
    """
    raw = pd.read_csv(_FIXTURE)
    agg = (
        raw.groupby("compound_name", as_index=False)
        .agg(
            {
                "SMILES": "first",
                "SEI": "mean",
                "BEI": "mean",
                "IMP_Final_Score": "mean",
                "Distance_Score": "mean",
                "Angle_Score": "mean",
            }
        )
    )
    # A synthetic Promise axis (0-100) so the decision map has an x channel.
    agg["Promise"] = (agg["IMP_Final_Score"] * 0.8 + 10.0).clip(0, 100)
    return agg


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


# =============================================================================
# create_chemical_space_scatter (Stage-1 — member=point, IMP=Viridis color)
# =============================================================================


def _make_coords(n: int = 5, seed: int = 0, with_clusters: bool = False):
    """Return a toy projection frame (one row per member)."""
    rng = np.random.default_rng(seed=seed)
    df = pd.DataFrame(
        {
            "compound_name": [f"Member {i}" for i in range(n)],
            "x": rng.normal(0.0, 1.0, n),
            "y": rng.normal(0.0, 1.0, n),
            "IMP_Final_Score": rng.uniform(0.0, 100.0, n),
            "n_targets": rng.integers(1, 12, n).astype(float),
        }
    )
    if with_clusters:
        df["cluster"] = rng.integers(0, 2, n)
    return df


class TestCreateChemicalSpaceScatter:
    """Render-without-error + color-channel contract (member=point → IMP Viridis)."""

    def test_returns_figure_instance(self):
        fig = create_chemical_space_scatter(_make_coords())
        assert isinstance(fig, go.Figure)

    def test_renders_without_error_on_toy_data(self):
        # No exception on the happy path; at least one trace produced.
        fig = create_chemical_space_scatter(_make_coords(n=6))
        assert len(fig.data) >= 1

    def test_uses_viridis_continuous_for_imp(self):
        # member = point -> IMP owns the color channel on a CONTINUOUS Viridis
        # gradient (no discrete bands — Phase 21 VIZ-03).
        fig = create_chemical_space_scatter(_make_coords())
        colorscale = fig.layout.coloraxis.colorscale
        assert colorscale is not None
        assert colorscale[0][1].lower() == "#440154"  # Viridis start
        assert colorscale[-1][1].lower() == "#fde725"  # Viridis end

    def test_no_banded_color_axis(self):
        # A banded/discrete IMP scale would surface as coloraxis tick bands;
        # the continuous gradient must NOT set a stepped colorscale of named
        # band colors. Endpoints already asserted continuous above; here we
        # assert the scale has the full multi-stop Viridis ramp (>2 stops),
        # never a 2-stop binary band.
        fig = create_chemical_space_scatter(_make_coords())
        assert len(fig.layout.coloraxis.colorscale) > 2

    def test_hover_shows_integer_imp(self):
        fig = create_chemical_space_scatter(_make_coords())
        assert "%{marker.color:.0f}" in fig.data[0].hovertemplate

    def test_axis_titles_are_human_readable_not_literal_xy(self):
        # The legend/axis bugfix: axis titles are real similarity-axis labels,
        # NOT the literal "x"/"y" the raw default column names produced
        # (UI-SPEC §"🧭 Chemical Space").
        fig = create_chemical_space_scatter(_make_coords())
        assert fig.layout.xaxis.title.text == "Similarity axis 1 (no units)"
        assert fig.layout.yaxis.title.text == "Similarity axis 2 (no units)"
        assert fig.layout.xaxis.title.text not in ("x", "y")
        assert fig.layout.yaxis.title.text not in ("x", "y")

    def test_colorbar_and_legend_have_distinct_positions(self):
        # The IMP colorbar and the symbol legend no longer overlap: the
        # colorbar is pushed flush-right, the symbol legend floats top-left.
        fig = create_chemical_space_scatter(
            _make_coords(with_clusters=True), cluster_col="cluster"
        )
        assert fig.layout.coloraxis.colorbar.x == 1.02
        assert fig.layout.legend.x == 0.01

    def test_cluster_outline_renamed_to_structural_group(self):
        # Cryptic "cluster 0/1/2" labels are remapped to "Structural group A/B/C".
        fig = create_chemical_space_scatter(
            _make_coords(with_clusters=True), cluster_col="cluster"
        )
        symbol_names = [t.name for t in fig.data if t.name]
        assert any(
            n and n.startswith("Structural group ") for n in symbol_names
        )

    def test_cluster_outline_does_not_displace_imp_color(self):
        # Optional cluster column drives the marker symbol (outline), IMP still
        # owns the continuous color channel.
        fig = create_chemical_space_scatter(
            _make_coords(with_clusters=True), cluster_col="cluster"
        )
        assert fig.layout.coloraxis.colorscale is not None

    def test_nan_size_column_does_not_crash(self):
        df = _make_coords()
        df.loc[0, "n_targets"] = np.nan
        fig = create_chemical_space_scatter(df)  # must not raise
        assert isinstance(fig, go.Figure)

    def test_missing_size_column_falls_back(self):
        df = _make_coords().drop(columns=["n_targets"])
        fig = create_chemical_space_scatter(df)  # size_col absent -> skipped
        assert isinstance(fig, go.Figure)

    def test_applies_impulator_theme(self):
        fig = create_chemical_space_scatter(_make_coords())
        assert fig.layout.hoverlabel.namelength == -1

    def test_empty_frame_returns_empty_themed_figure(self):
        fig = create_chemical_space_scatter(pd.DataFrame())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1

    def test_none_frame_returns_empty_themed_figure(self):
        fig = create_chemical_space_scatter(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


# =============================================================================
# create_imp_component_breakdown (Stage-1 — member=series → qualitative color)
# =============================================================================


def _make_per_member(n: int = 3, seed: int = 0):
    """Return a per-member IMP-component aggregation frame."""
    rng = np.random.default_rng(seed=seed)
    data = {"compound_name": [f"Member {i}" for i in range(n)]}
    for col in IMP_COMPONENT_COLS:
        data[col] = rng.uniform(-5.0, 30.0, n)
    return pd.DataFrame(data)


class TestCreateImpComponentBreakdown:
    """Render-without-error + member=series qualitative color contract."""

    def test_returns_figure_instance(self):
        fig = create_imp_component_breakdown(_make_per_member())
        assert isinstance(fig, go.Figure)

    def test_renders_without_error_on_toy_data(self):
        fig = create_imp_component_breakdown(_make_per_member(n=4))
        assert len(fig.data) >= 1

    def test_grouped_barmode(self):
        fig = create_imp_component_breakdown(_make_per_member())
        assert fig.layout.barmode == "group"

    def test_one_bar_trace_per_member_series(self):
        # member = series -> one trace per member (qualitative identity).
        fig = create_imp_component_breakdown(_make_per_member(n=3))
        assert len(fig.data) == 3
        for t in fig.data:
            assert t.type == "bar"

    def test_does_not_use_viridis_continuous_color(self):
        # member = series -> member identity owns color (qualitative), NOT the
        # IMP Viridis gradient. A grouped px.bar colored by member produces no
        # continuous coloraxis.
        fig = create_imp_component_breakdown(_make_per_member())
        assert fig.layout.coloraxis.colorscale is None

    def test_uses_default_component_cols(self):
        # x-axis categories derive from IMP_COMPONENT_COLS (underscores spaced).
        fig = create_imp_component_breakdown(_make_per_member())
        x_vals = set()
        for t in fig.data:
            x_vals.update(t.x)
        expected = {c.replace("_", " ") for c in IMP_COMPONENT_COLS}
        assert x_vals == expected

    def test_uses_index_when_name_col_absent(self):
        df = _make_per_member(n=2).set_index("compound_name")
        fig = create_imp_component_breakdown(df)  # name_col not a column
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_skips_absent_component_columns(self):
        df = _make_per_member()[["compound_name", "QED_Impact"]]
        fig = create_imp_component_breakdown(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_applies_impulator_theme(self):
        fig = create_imp_component_breakdown(_make_per_member())
        assert fig.layout.hoverlabel.namelength == -1

    def test_empty_frame_returns_empty_themed_figure(self):
        fig = create_imp_component_breakdown(pd.DataFrame())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1

    def test_none_frame_returns_empty_themed_figure(self):
        fig = create_imp_component_breakdown(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_no_component_columns_returns_empty_themed_figure(self):
        df = pd.DataFrame({"compound_name": ["A", "B"], "unrelated": [1.0, 2.0]})
        fig = create_imp_component_breakdown(df)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1


# =============================================================================
# create_compare_radar (Stage-3 Compare — member=trace → qualitative identity)
# =============================================================================


def _make_radar_norm(n: int = 3, seed: int = 0):
    """Return a per-member [0,1]-normalized radar frame (one row per member)."""
    rng = np.random.default_rng(seed=seed)
    data = {"compound_name": [f"Member {i}" for i in range(n)]}
    for axis in ("Potency", "Efficiency", "Druglikeness", "Selectivity"):
        data[axis] = rng.uniform(0.0, 1.0, n)
    return pd.DataFrame(data)


class TestCreateCompareRadar:
    """Render-without-error + member=trace qualitative + Scatterpolar contract."""

    def test_returns_figure_instance(self):
        fig = create_compare_radar(_make_radar_norm())
        assert isinstance(fig, go.Figure)

    def test_one_scatterpolar_trace_per_member(self):
        fig = create_compare_radar(_make_radar_norm(n=4))
        assert len(fig.data) == 4
        for t in fig.data:
            assert isinstance(t, go.Scatterpolar)

    def test_radial_axis_locked_to_unit_range(self):
        # Axes are pre-normalized [0, 1]; the radial axis is locked accordingly.
        fig = create_compare_radar(_make_radar_norm())
        assert tuple(fig.layout.polar.radialaxis.range) == (0, 1)

    def test_member_identity_qualitative_color_is_stable(self):
        # member = trace -> qualitative palette by SORTED member order, so the
        # same member keeps the same color across reruns / selections.
        a = create_compare_radar(_make_radar_norm(n=3))
        b = create_compare_radar(_make_radar_norm(n=3))
        assert [t.line.color for t in a.data] == [t.line.color for t in b.data]

    def test_polygon_is_closed(self):
        # The first axis is repeated to close the polygon (Scatterpolar idiom):
        # r/theta length == axis count + 1.
        fig = create_compare_radar(_make_radar_norm())
        n_axes = 4  # Potency / Efficiency / Druglikeness / Selectivity
        assert len(fig.data[0].r) == n_axes + 1
        assert len(fig.data[0].theta) == n_axes + 1

    def test_applies_impulator_theme(self):
        fig = create_compare_radar(_make_radar_norm())
        assert fig.layout.hoverlabel.namelength == -1

    def test_empty_frame_returns_empty_themed_figure(self):
        fig = create_compare_radar(pd.DataFrame())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1

    def test_no_axis_columns_returns_empty_themed_figure(self):
        df = pd.DataFrame({"compound_name": ["A", "B"]})
        fig = create_compare_radar(df)
        assert len(fig.data) == 0


# =============================================================================
# create_pareto_plot (Stage-3 — member=point → IMP Viridis continuous)
# =============================================================================


def _make_pareto_df(n: int = 5, seed: int = 0):
    """Return a per-member trade-off frame (x/y axes + IMP)."""
    rng = np.random.default_rng(seed=seed)
    return pd.DataFrame(
        {
            "compound_name": [f"Member {i}" for i in range(n)],
            "SEI": rng.uniform(1.0, 20.0, n),
            "BEI": rng.uniform(5.0, 30.0, n),
            "IMP_Final_Score": rng.uniform(0.0, 100.0, n),
        }
    )


class TestCreateParetoPlot:
    """Render-without-error + IMP Viridis continuous + front-overlay contract."""

    def test_returns_figure_instance(self):
        df = _make_pareto_df()
        mask = [True, False, False, True, False]
        fig = create_pareto_plot(df, "SEI", "BEI", mask)
        assert isinstance(fig, go.Figure)

    def test_uses_viridis_continuous_for_imp(self):
        # member = point -> IMP owns a CONTINUOUS Viridis color channel.
        df = _make_pareto_df()
        mask = [False] * len(df)
        fig = create_pareto_plot(df, "SEI", "BEI", mask)
        colorscale = fig.layout.coloraxis.colorscale
        assert colorscale is not None
        assert colorscale[0][1].lower() == "#440154"  # Viridis start
        assert colorscale[-1][1].lower() == "#fde725"  # Viridis end

    def test_front_overlay_trace_added_when_mask_has_members(self):
        df = _make_pareto_df()
        mask = [True, False, False, True, False]
        fig = create_pareto_plot(df, "SEI", "BEI", mask)
        names = [t.name for t in fig.data]
        assert "Pareto front" in names

    def test_no_front_overlay_when_mask_all_false(self):
        df = _make_pareto_df()
        mask = [False] * len(df)
        fig = create_pareto_plot(df, "SEI", "BEI", mask)
        names = [t.name for t in fig.data]
        assert "Pareto front" not in names

    def test_applies_impulator_theme(self):
        df = _make_pareto_df()
        fig = create_pareto_plot(df, "SEI", "BEI", [False] * len(df))
        assert fig.layout.hoverlabel.namelength == -1

    def test_empty_frame_returns_empty_themed_figure(self):
        fig = create_pareto_plot(pd.DataFrame(), "SEI", "BEI", [])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1

    def test_missing_axis_returns_empty_themed_figure(self):
        df = _make_pareto_df().drop(columns=["BEI"])
        fig = create_pareto_plot(df, "SEI", "BEI", [False] * len(df))
        assert len(fig.data) == 0


# =============================================================================
# create_sar_matrix (Stage-3 SAR-lite — matrix cell → Viridis sequential)
# =============================================================================


def _make_sar_matrix(n: int = 4, seed: int = 0):
    """Return a symmetric unit-diagonal Tanimoto-like matrix + labels."""
    rng = np.random.default_rng(seed=seed)
    m = rng.uniform(0.0, 1.0, (n, n))
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 1.0)
    labels = [f"Member {i}" for i in range(n)]
    return m, labels


class TestCreateSarMatrix:
    """Render-without-error + Viridis sequential + heatmap contract."""

    def test_returns_figure_instance(self):
        m, labels = _make_sar_matrix()
        fig = create_sar_matrix(m, labels)
        assert isinstance(fig, go.Figure)

    def test_renders_a_heatmap(self):
        m, labels = _make_sar_matrix()
        fig = create_sar_matrix(m, labels)
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_uses_viridis_sequential(self):
        m, labels = _make_sar_matrix()
        fig = create_sar_matrix(m, labels)
        # Plotly resolves the named "Viridis" scale to its stop list; the
        # endpoints are the Viridis sequential ramp (no diverging midpoint).
        colorscale = fig.data[0].colorscale
        assert colorscale[0][1].lower() == "#440154"
        assert colorscale[-1][1].lower() == "#fde725"

    def test_similarity_axis_locked_to_unit_range(self):
        m, labels = _make_sar_matrix()
        fig = create_sar_matrix(m, labels)
        assert fig.data[0].zmin == 0.0
        assert fig.data[0].zmax == 1.0

    def test_applies_impulator_theme(self):
        m, labels = _make_sar_matrix()
        fig = create_sar_matrix(m, labels)
        assert fig.layout.hoverlabel.namelength == -1

    def test_empty_matrix_returns_empty_themed_figure(self):
        fig = create_sar_matrix(np.zeros((0, 0)), [])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1


# =============================================================================
# create_decision_map (P0a — member=point, IMP=Viridis; Promise×IMP plane)
# =============================================================================


def _make_decision_df(n: int = 5, seed: int = 0):
    """Return a per-member Promise×IMP frame (one row per member)."""
    rng = np.random.default_rng(seed=seed)
    return pd.DataFrame(
        {
            "compound_name": [f"Member {i}" for i in range(n)],
            "SMILES": ["CCO"] * n,
            "Promise": rng.uniform(0.0, 100.0, n),
            "IMP_Final_Score": rng.uniform(0.0, 100.0, n),
        }
    )


def _vline_at(fig, x):
    """True if any layout shape is a vertical line at ``x`` (x0==x1==x)."""
    return any(
        s.type == "line" and s.x0 == x and s.x1 == x for s in fig.layout.shapes
    )


def _hlines_y(fig):
    """Return the set of y-values of horizontal-line layout shapes."""
    return {
        s.y0
        for s in fig.layout.shapes
        if s.type == "line" and s.y0 == s.y1
    }


class TestCreateDecisionMap:
    """Promise×IMP decision map — IMP is the only color channel (PRES-07)."""

    def test_returns_figure_instance(self):
        fig = create_decision_map(
            _make_decision_df(), promise_col="Promise", imp_col="IMP_Final_Score"
        )
        assert isinstance(fig, go.Figure)

    def test_decision_map_uses_viridis_continuous_imp(self):
        # IMP owns a CONTINUOUS Viridis color channel.
        fig = create_decision_map(
            _make_decision_df(), promise_col="Promise", imp_col="IMP_Final_Score"
        )
        colorscale = fig.layout.coloraxis.colorscale
        assert colorscale is not None
        assert colorscale[0][1].lower() == "#440154"  # Viridis start
        assert colorscale[-1][1].lower() == "#fde725"  # Viridis end
        assert len(colorscale) > 2  # full ramp, never a 2-stop band

    def test_decision_map_fixed_promise_divider_at_50(self):
        # A FIXED vertical divider at Promise=50 (D-25-MAP-SPLIT), not a median.
        fig = create_decision_map(
            _make_decision_df(), promise_col="Promise", imp_col="IMP_Final_Score"
        )
        assert _vline_at(fig, 50)

    def test_decision_map_grey_band_hlines(self):
        # IMP band lines at 30/50/70/90 (= 0.30/0.50/0.70/0.90 × 100).
        fig = create_decision_map(
            _make_decision_df(), promise_col="Promise", imp_col="IMP_Final_Score"
        )
        ys = _hlines_y(fig)
        for band in (30, 50, 70, 90):
            assert band in ys

    def test_decision_map_grey_quadrant_rects(self):
        # Neutral grey quadrant shading rects (never tinted by IMP band).
        fig = create_decision_map(
            _make_decision_df(), promise_col="Promise", imp_col="IMP_Final_Score"
        )
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) >= 1
        for r in rects:
            assert "128" in str(r.fillcolor)  # grey rgba(128,128,128,...)

    def test_decision_map_text_yaxis_annotations(self):
        # Persistent TEXT y-axis annotations (text, NOT banded color).
        fig = create_decision_map(
            _make_decision_df(), promise_col="Promise", imp_col="IMP_Final_Score"
        )
        texts = " ".join(a.text for a in fig.layout.annotations)
        assert "more suspicious" in texts
        assert "more genuine" in texts

    def test_decision_map_hover_customdata_unique_id(self):
        # SMILES rides in customdata for downstream hover wiring
        # (VALIDATION anchor: hover_customdata_unique_id). The unique
        # key/chart_id half is Plan 04's page-wiring concern, not the factory's.
        fig = create_decision_map(
            _make_decision_df(),
            promise_col="Promise",
            imp_col="IMP_Final_Score",
            smiles_col="SMILES",
        )
        customdata = fig.data[0].customdata
        assert customdata is not None
        assert any("CCO" in str(row[0]) for row in customdata)

    def test_decision_map_applies_impulator_theme(self):
        fig = create_decision_map(
            _make_decision_df(), promise_col="Promise", imp_col="IMP_Final_Score"
        )
        assert fig.layout.hoverlabel.namelength == -1

    def test_decision_map_empty_frame_returns_empty_themed_figure(self):
        fig = create_decision_map(
            pd.DataFrame(), promise_col="Promise", imp_col="IMP_Final_Score"
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1

    def test_decision_map_none_frame_returns_empty_themed_figure(self):
        fig = create_decision_map(
            None, promise_col="Promise", imp_col="IMP_Final_Score"
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


# =============================================================================
# create_efficiency_plane (P0a — SEI×BEI; Viridis; argmax(Distance) watch marker)
# =============================================================================


class TestCreateEfficiencyPlane:
    """SEI×BEI efficiency plane — Viridis recolor + argmax(Distance) marker."""

    def test_returns_figure_instance(self):
        fig = create_efficiency_plane(
            _load_member_frame(), imp_col="IMP_Final_Score"
        )
        assert isinstance(fig, go.Figure)

    def test_efficiency_plane_viridis(self):
        # The plane's continuous colorscale is Viridis — NOT RdYlGn_r (RR-2).
        fig = create_efficiency_plane(
            _load_member_frame(), imp_col="IMP_Final_Score"
        )
        colorscale = fig.layout.coloraxis.colorscale
        assert colorscale is not None
        assert colorscale[0][1].lower() == "#440154"  # Viridis start
        assert colorscale[-1][1].lower() == "#fde725"  # Viridis end
        # No trace uses the source's RdYlGn_r scale.
        for t in fig.data:
            marker = getattr(t, "marker", None)
            if marker is not None and getattr(marker, "colorscale", None):
                assert marker.colorscale != "RdYlGn_r"

    def test_efficiency_plane_structure(self):
        # Equal-axis scaling + a 45° reference line + an angle banner annotation.
        fig = create_efficiency_plane(
            _load_member_frame(), imp_col="IMP_Final_Score"
        )
        # Equal-axis scaling (scaleanchor links x to y).
        assert fig.layout.xaxis.scaleanchor == "y"
        # 45° optimal reference line trace present.
        names = [t.name for t in fig.data]
        assert any(n and "45" in n for n in names)
        # Angle banner annotation present (one of the three status words).
        ann_text = " ".join(a.text for a in fig.layout.annotations)
        assert any(
            word in ann_text for word in ("OPTIMAL", "ACCEPTABLE", "UNBALANCED")
        )

    def test_watch_marker_is_argmax_distance(self):
        # The watch marker is the member with MAX Distance_Score, NOT the member
        # with MAX modulus sqrt(SEI^2 + BEI^2). Both argmaxes are derived from
        # the loaded frame at runtime (they differ by fixture design).
        frame = _load_member_frame()
        dist_argmax = frame.loc[
            frame["Distance_Score"].idxmax(), "compound_name"
        ]
        modulus = np.sqrt(frame["SEI"] ** 2 + frame["BEI"] ** 2)
        mod_argmax = frame.loc[modulus.idxmax(), "compound_name"]
        assert dist_argmax != mod_argmax  # fixture sanity: they discriminate

        fig = create_efficiency_plane(frame, imp_col="IMP_Final_Score")
        watch = [
            t for t in fig.data if t.name and "watch" in t.name.lower()
        ]
        assert len(watch) == 1
        # Marker sits at the Distance argmax's (SEI, BEI), not the modulus one.
        dist_row = frame[frame["compound_name"] == dist_argmax].iloc[0]
        mod_row = frame[frame["compound_name"] == mod_argmax].iloc[0]
        assert float(watch[0].x[0]) == pytest.approx(float(dist_row["SEI"]))
        assert float(watch[0].y[0]) == pytest.approx(float(dist_row["BEI"]))
        assert not (
            float(watch[0].x[0]) == pytest.approx(float(mod_row["SEI"]))
            and float(watch[0].y[0]) == pytest.approx(float(mod_row["BEI"]))
        )

    def test_efficiency_plane_hover_customdata_unique_id(self):
        # SMILES rides in customdata for downstream hover wiring
        # (VALIDATION anchor: hover_customdata_unique_id).
        fig = create_efficiency_plane(
            _load_member_frame(), imp_col="IMP_Final_Score", smiles_col="SMILES"
        )
        # The member point trace carries SMILES in customdata.
        has_smiles = any(
            t.customdata is not None
            and any("C" in str(row[0]) for row in t.customdata)
            for t in fig.data
            if t.customdata is not None
        )
        assert has_smiles

    def test_efficiency_plane_applies_impulator_theme(self):
        fig = create_efficiency_plane(
            _load_member_frame(), imp_col="IMP_Final_Score"
        )
        assert fig.layout.hoverlabel.namelength == -1

    def test_efficiency_plane_empty_frame_returns_empty_themed_figure(self):
        fig = create_efficiency_plane(pd.DataFrame(), imp_col="IMP_Final_Score")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1

    def test_efficiency_plane_none_frame_returns_empty_themed_figure(self):
        fig = create_efficiency_plane(None, imp_col="IMP_Final_Score")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


# =============================================================================
# create_imp_component_radar (P0a — 5-spoke overlay, NO 6th QED-impact spoke)
# =============================================================================


# The 5 plain-language radar axis labels (verbatim, UI-SPEC §"🔬 IMP Analysis").
_RADAR_AXES = (
    "Efficiency-outlier",
    "Distance-to-best",
    "Development-angle",
    "Assay-interference",
    "PDB-evidence",
)


def _make_radar5_norm(n: int = 3, seed: int = 0):
    """Return a per-member [0,1]-normalized 5-axis radar frame."""
    rng = np.random.default_rng(seed=seed)
    data = {"compound_name": [f"Member {i}" for i in range(n)]}
    for axis in _RADAR_AXES:
        data[axis] = rng.uniform(0.0, 1.0, n)
    return pd.DataFrame(data)


class TestCreateImpComponentRadar:
    """5-spoke IMP component radar overlay (reuses create_compare_radar)."""

    def test_imp_component_radar_returns_figure_instance(self):
        fig = create_imp_component_radar(_make_radar5_norm())
        assert isinstance(fig, go.Figure)

    def test_imp_component_radar_one_trace_per_member(self):
        fig = create_imp_component_radar(_make_radar5_norm(n=4))
        assert len(fig.data) == 4
        for t in fig.data:
            assert isinstance(t, go.Scatterpolar)

    def test_imp_component_radar_has_5_spokes(self):
        # The radar trace theta has EXACTLY 5 unique categories — NO 6th
        # "QED-impact" spoke. Assert on the FIGURE (review #6).
        fig = create_imp_component_radar(_make_radar5_norm())
        theta = fig.data[0].theta
        # theta is closed (first axis repeated); unique count must be 5.
        assert len(set(theta)) == 5
        assert "QED" not in " ".join(str(t) for t in theta)

    def test_radar_has_exactly_5_axes(self):
        # The rendered polar radar exposes EXACTLY 5 angular categories (the
        # 5-axis guard — the normalized frame has 5 axis columns, never 6).
        norm = _make_radar5_norm()
        axis_cols = [c for c in norm.columns if c != "compound_name"]
        assert len(axis_cols) == 5
        fig = create_imp_component_radar(norm)
        assert len(set(fig.data[0].theta)) == 5

    def test_imp_component_radar_applies_impulator_theme(self):
        fig = create_imp_component_radar(_make_radar5_norm())
        assert fig.layout.hoverlabel.namelength == -1

    def test_imp_component_radar_empty_frame_returns_empty_themed_figure(self):
        fig = create_imp_component_radar(pd.DataFrame())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1

    def test_imp_component_radar_none_returns_empty_themed_figure(self):
        fig = create_imp_component_radar(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


# =============================================================================
# create_promise_contribution_bar (P0a — STACKED bar by Promise component)
# =============================================================================


# Plain-language Promise component labels (verbatim, UI-SPEC §"Promise display").
_PROMISE_COMPONENTS = (
    "Potency",
    "Ligand efficiency",
    "Apparent promiscuity (recorded active targets)",
    "Cleanliness",
    "Druglikeness",
)


def _make_promise_df(n: int = 3, seed: int = 0):
    """Return a per-member Promise-component contribution frame."""
    rng = np.random.default_rng(seed=seed)
    data = {"compound_name": [f"Member {i}" for i in range(n)]}
    for comp in _PROMISE_COMPONENTS:
        data[comp] = rng.uniform(0.0, 25.0, n)
    return pd.DataFrame(data)


class TestCreatePromiseContributionBar:
    """STACKED Promise-decomposition bar colored by Promise COMPONENT."""

    def test_promise_contribution_returns_figure_instance(self):
        fig = create_promise_contribution_bar(_make_promise_df())
        assert isinstance(fig, go.Figure)

    def test_promise_contribution_renders_without_error(self):
        fig = create_promise_contribution_bar(_make_promise_df(n=4))
        assert len(fig.data) >= 1

    def test_promise_contribution_stacked_barmode(self):
        fig = create_promise_contribution_bar(_make_promise_df())
        assert fig.layout.barmode == "stack"

    def test_promise_contribution_colored_by_component(self):
        # color = Promise COMPONENT (not member) -> one trace per component.
        fig = create_promise_contribution_bar(_make_promise_df(n=3))
        names = {t.name for t in fig.data}
        assert names == set(_PROMISE_COMPONENTS)

    def test_promise_contribution_applies_impulator_theme(self):
        fig = create_promise_contribution_bar(_make_promise_df())
        assert fig.layout.hoverlabel.namelength == -1

    def test_promise_contribution_empty_frame_returns_empty_themed_figure(self):
        fig = create_promise_contribution_bar(pd.DataFrame())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
        assert fig.layout.hoverlabel.namelength == -1

    def test_promise_contribution_none_returns_empty_themed_figure(self):
        fig = create_promise_contribution_bar(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


# =============================================================================
# create_imp_contribution_bar (P0a — single-member weighted Base×QED bar)
# =============================================================================


def _make_member_row(qed_mult: float = 0.7):
    """Return a single member row with the 5 raw IMP component scores."""
    return pd.Series(
        {
            "compound_name": "ETHANOL",
            "Efficiency_Score": 0.60,
            "Distance_Score": 0.50,
            "Angle_Score": 0.55,
            "Interference_Score": 0.20,
            "PDB_Score": 0.40,
            "QED_Multiplier": qed_mult,
        }
    )


class TestCreateImpContributionBar:
    """Single-member weighted-contribution bar with Base×QED subtitle."""

    def test_imp_contribution_bar_returns_figure_instance(self):
        fig = create_imp_contribution_bar(_make_member_row())
        assert isinstance(fig, go.Figure)

    def test_imp_contribution_bar_structure(self):
        # EXACTLY 5 horizontal bars; subtitle contains "Base" AND "× QED"
        # (U+00D7). Assert on the FIGURE (review #6).
        fig = create_imp_contribution_bar(_make_member_row())
        bar_traces = [t for t in fig.data if t.type == "bar"]
        assert len(bar_traces) == 1
        assert len(bar_traces[0].y) == 5  # 5 component bars
        assert bar_traces[0].orientation == "h"
        subtitle = fig.layout.title.subtitle.text
        assert "Base" in subtitle
        assert "× QED" in subtitle  # Unicode multiplication sign

    def test_imp_contribution_bar_reads_qed_multiplier(self):
        # The Base×QED subtitle reads the QED_Multiplier column exactly.
        fig = create_imp_contribution_bar(_make_member_row(qed_mult=0.5))
        # base = 0.45*.60 + 0.20*.50 + 0.15*.55 + 0.15*.20 + 0.05*.40
        base = 0.45 * 0.60 + 0.20 * 0.50 + 0.15 * 0.55 + 0.15 * 0.20 + 0.05 * 0.40
        subtitle = fig.layout.title.subtitle.text
        assert f"{base:.3f}" in subtitle
        assert "0.500" in subtitle  # the QED multiplier value

    def test_imp_contribution_bar_title(self):
        fig = create_imp_contribution_bar(_make_member_row())
        assert fig.layout.title.text == "Weighted Contributions"

    def test_imp_contribution_bar_applies_impulator_theme(self):
        fig = create_imp_contribution_bar(_make_member_row())
        assert fig.layout.hoverlabel.namelength == -1

    def test_imp_contribution_bar_empty(self):
        # None AND an all-NaN-score row each return an empty themed figure with
        # no crash (review #1 — pinned empty/None guard).
        fig_none = create_imp_contribution_bar(None)
        assert isinstance(fig_none, go.Figure)
        assert len(fig_none.data) == 0
        assert fig_none.layout.hoverlabel.namelength == -1

        nan_row = pd.Series(
            {
                "compound_name": "X",
                "Efficiency_Score": np.nan,
                "Distance_Score": np.nan,
                "Angle_Score": np.nan,
                "Interference_Score": np.nan,
                "PDB_Score": np.nan,
                "QED_Multiplier": 1.0,
            }
        )
        fig_nan = create_imp_contribution_bar(nan_row)
        assert isinstance(fig_nan, go.Figure)
        assert len(fig_nan.data) == 0

    def test_imp_contribution_bar_empty_row_missing_cols(self):
        # A row missing the score columns entirely must not raise.
        fig = create_imp_contribution_bar(pd.Series({"compound_name": "X"}))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
