"""Chart components for molecular data visualization.

This module provides reusable chart components built on Plotly,
with support for the structure viewer integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from typing import Optional, Any

from backend.modules.imp_gmm import DENSITY_GRID, component_curves
from frontend.ui.components.molecule_viewer import (
    embed_structure_viewer,
    render_structure_viewer_hint,
    prepare_chart_customdata,
)

# Number of target columns retained in the collection bioactivity heatmap
# (D-10). Targets are ranked by member-hit-count (how many members have any
# activity against the target) then total activity, and the top-K are kept;
# the page surfaces an "+N more targets" caption for the remainder.
HEATMAP_TOP_K = 25

# Max characters shown on a heatmap target tick label before truncation. The
# full target name is preserved in the hover tooltip (D-01 / UI-SPEC §3).
HEATMAP_TARGET_LABEL_MAXLEN = 24

# Approximate pixel height allotted per member row so the heatmap grows with
# member count instead of squashing rows (UI-SPEC §3 "~32px/row").
HEATMAP_ROW_HEIGHT_PX = 32


def _subscript(k: int) -> str:
    """Return the Unicode subscript digit for ``k``.

    Single-digit subscript only (k ∈ [0, 9]). Safe for Phase 22 because
    ``MAX_COMPONENTS = 6``.
    """
    return chr(0x2080 + k)


def _rgba_with_alpha(color: str, alpha: float) -> str:
    """Convert a Plotly color string to an ``rgba(r, g, b, a)`` translucent variant.

    Accepts either ``rgb(r,g,b)`` (Plotly qualitative palettes like ``Set2``
    return this format) or ``#rrggbb`` hex strings. Used for translucent fills
    under the GMM density-overlay component curves so the qualitative-palette
    identity is preserved while reducing visual weight relative to the solid
    component lines.
    """
    if color.startswith("rgb"):
        r, g, b = px.colors.unlabel_rgb(color)
    else:
        r, g, b = px.colors.hex_to_rgb(color)
    return f"rgba({int(r)}, {int(g)}, {int(b)}, {alpha})"


def get_plotly_theme() -> dict:
    """Return Plotly layout kwargs for the IMPULATOR theme.

    The app is locked to light mode (see ``.streamlit/config.toml``),
    so this returns a single fixed palette — no detection branching.
    """
    return {
        "template": "plotly_white",
        "legend_bgcolor": "rgba(255,255,255,0.8)",
        "legend_bordercolor": "rgba(0,0,0,0.1)",
    }


def apply_impulator_theme(fig: go.Figure) -> go.Figure:
    """Apply standard IMPULATOR theme to a Plotly figure.

    Ensures consistent hover behavior and styling across all charts.
    Call this on every figure before rendering with st.plotly_chart.

    Font sizes are bumped over Plotly's 12px default for accessibility
    (primary audience is 50+ users). ``layout.font`` covers axis labels,
    tick labels, legend entries, and annotations. Chart titles have
    their own ``title.font`` and are intentionally left untouched.
    """
    fig.update_layout(
        hoverlabel=dict(namelength=-1, font=dict(size=14)),
        font=dict(size=14),
    )
    return fig


def _apply_subtitle(fig: go.Figure, subtitle: str) -> None:
    """Apply a subtitle to a Plotly figure (Plotly v6+).

    Must be called after figure creation since px.* constructors
    only accept string titles.
    """
    fig.update_layout(
        title=dict(
            text=fig.layout.title.text if fig.layout.title and fig.layout.title.text else "",
            subtitle=dict(text=subtitle)
        )
    )


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    smiles_col: Optional[str] = None,
    name_col: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    trendline: bool = False,
    color_scale: str = "Viridis",
    marker_size: int = 8,
    opacity: float = 0.7
) -> go.Figure:
    """Create a scatter plot with structure viewer support.

    Args:
        df: DataFrame with data
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Optional column for color encoding
        size_col: Optional column for size encoding
        smiles_col: SMILES column for structure viewer
        name_col: Name/ID column for display
        title: Chart title
        subtitle: Chart subtitle (Plotly v6)
        trendline: Whether to add a trendline
        color_scale: Color scale for continuous color
        marker_size: Base marker size
        opacity: Marker opacity

    Returns:
        Plotly Figure
    """
    # Prepare customdata for structure viewer
    if smiles_col:
        df, customdata_cols = prepare_chart_customdata(df, smiles_col, name_col)
    else:
        customdata_cols = None

    # Build scatter plot
    default_title = f"{y_col.replace('_', ' ')} vs {x_col.replace('_', ' ')}"
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title or default_title,
        color_continuous_scale=color_scale,
        opacity=opacity,
        custom_data=customdata_cols if customdata_cols else None,
        trendline="ols" if trendline else None
    )

    # Update marker size if not using size column
    if not size_col:
        fig.update_traces(marker=dict(size=marker_size))

    # Update layout
    theme = get_plotly_theme()
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' '),
        yaxis_title=y_col.replace('_', ' '),
        template=theme["template"],
        hovermode="closest"
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def add_original_compound_marker(
    fig: go.Figure,
    x_val: float,
    y_val: float,
    label: str = "Original",
    hover_label: str = "Original compound (query)",
) -> go.Figure:
    """Overlay a distinct marker at the original (query) compound's position.

    Drawn as a gold star with a dark outline so it pops against the Viridis
    analog cloud and is unmistakable from the existing mean-marker. The marker
    is a native Plotly trace, so it is included in the figure's exported image
    (camera icon) — i.e. it travels into manuscript figures, not just the live
    view. Caller supplies coordinates already on the plot's axes (either the
    original's existing cloud row or computed via
    ``compute_query_structural_descriptors``).
    """
    fig.add_trace(
        go.Scatter(
            x=[x_val],
            y=[y_val],
            mode="markers+text",
            marker=dict(
                symbol="star",
                size=20,
                color="#FFD700",
                line=dict(color="#1a1a1a", width=1.6),
            ),
            text=[label],
            textposition="top center",
            textfont=dict(color="#1a1a1a", size=12),
            name=hover_label,
            hovertemplate=f"<b>{hover_label}</b><br>%{{x}}, %{{y}}<extra></extra>",
            showlegend=True,
        )
    )
    return fig


def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None
) -> go.Figure:
    """Create a histogram.

    Args:
        df: DataFrame with data
        column: Column to plot
        bins: Number of bins
        color_col: Optional column for color grouping
        title: Chart title
        subtitle: Chart subtitle (Plotly v6)

    Returns:
        Plotly Figure
    """
    theme = get_plotly_theme()
    fig = px.histogram(
        df,
        x=column,
        color=color_col,
        nbins=bins,
        title=title or f"Distribution of {column}",
        template=theme["template"]
    )

    fig.update_layout(
        xaxis_title=column.replace('_', ' '),
        yaxis_title="Count"
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_gmm_density_overlay(
    scores: np.ndarray,
    model: GaussianMixture,
    *,
    x_axis_label: str = "IMP Score",
) -> go.Figure:
    """Histogram + GMM component-density overlay for the Phase 22 widget.

    The histogram (corpus counts) sits on the left y-axis; the K per-component
    weighted Gaussian PDFs (consumed from :func:`backend.modules.imp_gmm.component_curves`)
    overlay on a twin right y-axis with translucent fills below each curve so
    the qualitative palette stays readable when components overlap.

    The chart deliberately does NOT mark the active compound's position with a
    vertical line. Because the corpus is built from the active compound's
    Tanimoto neighborhood (and per-record grain spreads a single compound
    across many bioactivity rows), pinning a "this compound" marker on the
    distribution it generated would be visually circular. The active
    compound's cluster membership is communicated via the separate stacked
    probability bar (see :func:`create_gmm_probability_bar`).

    Args:
        scores: 1-D ``np.ndarray`` of corpus IMP scores in INTEGER space
            ``[0, 100]``. The Streamlit caller rescales raw ``[0, 1]`` scores
            before passing.
        model: Fitted ``GaussianMixture`` (from Plan 02 ``fit_gmm``). Cluster
            ordering is sourced from ``component_curves``; this helper does
            NOT re-sort.
        x_axis_label: Override for the x-axis title. Defaults to ``"IMP Score"``.

    Returns:
        A ``go.Figure`` with trace z-order ``[histogram, K density curves]``.
        Calls :func:`apply_impulator_theme` as the LAST step before returning.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=30,
            marker_color="#6b7280",
            opacity=0.6,
            name="Corpus",
            yaxis="y",
        )
    )

    means, _weights, _sigmas, pdfs = component_curves(model, DENSITY_GRID)
    for k in range(len(means)):
        cluster_hex = px.colors.qualitative.Set2[k]
        fig.add_trace(
            go.Scatter(
                x=DENSITY_GRID,
                y=pdfs[k],
                mode="lines",
                line=dict(width=2.5, color=cluster_hex),
                fill="tozeroy",
                fillcolor=_rgba_with_alpha(cluster_hex, 0.15),
                name=f"C{_subscript(k)}: μ={int(round(float(means[k])))}",
                yaxis="y2",
            )
        )

    fig.update_layout(
        height=280,
        margin=dict(t=40, b=30, l=30, r=10),
        xaxis=dict(title=x_axis_label, range=[0, 100]),
        yaxis=dict(title="Count"),
        yaxis2=dict(title="Density", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.25),
        showlegend=True,
    )

    return apply_impulator_theme(fig)


def create_gmm_probability_bar(
    memberships,
    *,
    cluster_means,
) -> go.Figure:
    """Stacked horizontal probability bar for ``P(cluster_k | score)``.

    Args:
        memberships: Sequence of length K, ``P(cluster_k | score)`` already
            sorted by ascending cluster mean (consumed from
            :func:`backend.modules.imp_gmm.cluster_membership`).
        cluster_means: Sequence of length K, cluster means in INTEGER space,
            used for legend labels ``Cₖ: μ=<int>``.

    Returns:
        A ``go.Figure`` with K stacked horizontal ``go.Bar`` traces.
        Memberships are defensively normalized at construction time
        (``mem / mem.sum()`` — Pitfall 2) so segments always sum to 100%.
        Calls :func:`apply_impulator_theme` as the LAST step before returning.
    """
    mem = np.asarray(memberships, dtype=float)
    total = mem.sum()
    if total > 0:
        mem = mem / total

    means_arr = np.asarray(cluster_means, dtype=float)

    fig = go.Figure()
    for k in range(len(mem)):
        p_k = float(mem[k])
        fig.add_trace(
            go.Bar(
                x=[p_k],
                y=["Membership"],
                orientation="h",
                name=f"C{_subscript(k)}: μ={int(round(float(means_arr[k])))}",
                marker=dict(color=px.colors.qualitative.Set2[k]),
                text=f"{int(round(p_k * 100))}%",
                textposition="inside",
                hovertemplate=f"P(C{k}) = {p_k:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        height=140,
        margin=dict(t=10, b=70, l=10, r=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False),
        yaxis=dict(showticklabels=False),
        showlegend=True,
        legend=dict(orientation="h", y=-0.6, yanchor="top"),
    )

    return apply_impulator_theme(fig)


def create_box_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None
) -> go.Figure:
    """Create a box plot.

    Args:
        df: DataFrame with data
        y_col: Y-axis column (values)
        x_col: Optional X-axis column (categories)
        color_col: Optional column for color grouping
        title: Chart title
        subtitle: Chart subtitle (Plotly v6)

    Returns:
        Plotly Figure
    """
    theme = get_plotly_theme()
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title or f"Distribution of {y_col}",
        template=theme["template"]
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_violin_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None
) -> go.Figure:
    """Create a violin plot.

    Args:
        df: DataFrame with data
        y_col: Y-axis column (values)
        x_col: Optional X-axis column (categories)
        color_col: Optional column for color grouping
        title: Chart title
        subtitle: Chart subtitle (Plotly v6)

    Returns:
        Plotly Figure
    """
    theme = get_plotly_theme()
    fig = px.violin(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        box=True,
        title=title or f"Distribution of {y_col}",
        template=theme["template"]
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    orientation: str = "v",
    barmode: str = "group"
) -> go.Figure:
    """Create a bar chart.

    Args:
        df: DataFrame with data
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Optional column for color encoding
        title: Chart title
        subtitle: Chart subtitle (Plotly v6)
        orientation: 'v' for vertical, 'h' for horizontal
        barmode: How bars sharing an axis position are laid out — ``'group'``
            (side-by-side, the default for collection member comparison) or
            ``'stack'``. Passed through to Plotly's ``layout.barmode``.

    Returns:
        Plotly Figure
    """
    theme = get_plotly_theme()
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title or f"{y_col} by {x_col}",
        template=theme["template"],
        orientation=orientation,
        barmode=barmode
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def _truncate_label(label: str, maxlen: int = HEATMAP_TARGET_LABEL_MAXLEN) -> str:
    """Truncate a long axis label to ``maxlen`` chars with an ellipsis.

    The full label is preserved separately for the hover tooltip; only the
    displayed tick text is shortened so the heatmap x-axis stays legible.
    """
    label = str(label)
    if len(label) <= maxlen:
        return label
    return label[: maxlen - 1].rstrip() + "…"


def create_bioactivity_heatmap(
    matrix: pd.DataFrame,
    *,
    top_k: int = HEATMAP_TOP_K,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    color_scale: str = "Viridis",
) -> go.Figure:
    """Build the collection bioactivity heatmap (D-01 headline viz / D-10).

    Encoding contract (UI-SPEC §3):

    * **rows (y)** = collection members / input compound names (the
      ``matrix`` index),
    * **columns (x)** = the top-``K`` ≈ 25 targets, ranked by
      member-hit-count (how many members have any activity against the
      target) then total activity, descending. ``K`` defaults to the
      module-level :data:`HEATMAP_TOP_K`,
    * **cell color** = the activity / hit value, on the **Viridis
      sequential** scale (activity has no meaningful midpoint, so a
      diverging scale would be misleading) — matching the scatter default
      for visual consistency.

    Target tick labels show the target name truncated to
    :data:`HEATMAP_TARGET_LABEL_MAXLEN` chars, with the **full** name kept
    in the hover tooltip. Figure height grows ``~32px`` per member row.
    Styled via :func:`apply_impulator_theme` as the LAST step.

    Empty / sparse handling: if ``matrix`` is empty (no members or no shared
    targets), an empty themed :class:`~plotly.graph_objects.Figure` is
    returned (``len(fig.data) == 0``) so the page can branch and render the
    ``st.info`` "No shared targets" copy instead of an empty grid.

    Args:
        matrix: DataFrame indexed by member name, one column per target,
            cell values = activity / hit count (NaN / 0 where a member has no
            activity against a target).
        top_k: Number of top targets to retain (D-10). Defaults to
            :data:`HEATMAP_TOP_K`.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).
        color_scale: Continuous color scale. Defaults to ``"Viridis"``
            (sequential) — overriding is discouraged for this chart.

    Returns:
        A themed Plotly ``go.Figure`` (a ``go.Heatmap``), or an empty themed
        figure when there is nothing to plot.
    """
    # Empty / no-member / no-target input -> caller branches on len(fig.data).
    if matrix is None or matrix.empty or matrix.shape[1] == 0:
        return apply_impulator_theme(go.Figure())

    numeric = matrix.apply(pd.to_numeric, errors="coerce")

    # Rank targets (columns) by member-hit-count (number of members with any
    # non-null, non-zero activity) then total activity — both descending —
    # and keep the top-K (D-10).
    hit_count = (numeric.fillna(0) != 0).sum(axis=0)
    total_activity = numeric.fillna(0).sum(axis=0)
    ranking = pd.DataFrame({"hits": hit_count, "activity": total_activity})
    ranking = ranking.sort_values(
        by=["hits", "activity"], ascending=[False, False]
    )
    top_targets = list(ranking.index[: max(int(top_k), 1)])
    top = numeric[top_targets]

    members = [str(m) for m in top.index]
    full_targets = [str(t) for t in top.columns]
    tick_labels = [_truncate_label(t) for t in full_targets]
    z = top.to_numpy(dtype=float)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=full_targets,
            y=members,
            colorscale=color_scale,  # Viridis sequential (D-01)
            colorbar=dict(title="Activity"),
            hovertemplate=(
                "Member: %{y}<br>Target: %{x}<br>Activity: %{z}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or "Bioactivity by Target",
        height=max(len(members) * HEATMAP_ROW_HEIGHT_PX + 160, 260),
        xaxis=dict(
            title="Target",
            tickmode="array",
            tickvals=full_targets,
            ticktext=tick_labels,
            tickangle=-45,
        ),
        yaxis=dict(title="Member", autorange="reversed"),
        margin=dict(t=60, b=120, l=10, r=10),
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    smiles_col: Optional[str] = None,
    name_col: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    color_scale: str = "Viridis",
    marker_size: int = 5,
    opacity: float = 0.8
) -> go.Figure:
    """Create a 3D scatter plot with structure viewer support.

    Args:
        df: DataFrame with data
        x_col: X-axis column
        y_col: Y-axis column
        z_col: Z-axis column
        color_col: Optional column for color encoding
        size_col: Optional column for size encoding
        smiles_col: SMILES column for structure viewer
        name_col: Name/ID column for display
        title: Chart title
        subtitle: Chart subtitle (Plotly v6)
        color_scale: Color scale for continuous color
        marker_size: Base marker size
        opacity: Marker opacity

    Returns:
        Plotly Figure
    """
    # Prepare customdata for structure viewer
    if smiles_col:
        df, customdata_cols = prepare_chart_customdata(df, smiles_col, name_col)
    else:
        customdata_cols = None

    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        size=size_col,
        title=title or f"3D: {x_col} vs {y_col} vs {z_col}",
        color_continuous_scale=color_scale,
        opacity=opacity,
        custom_data=customdata_cols if customdata_cols else None
    )

    # Update marker size if not using size column
    if not size_col:
        fig.update_traces(marker=dict(size=marker_size))

    fig.update_layout(
        scene=dict(
            xaxis_title=x_col.replace('_', ' '),
            yaxis_title=y_col.replace('_', ' '),
            zaxis_title=z_col.replace('_', ' '),
        )
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    title: str = "Correlation Heatmap",
    subtitle: Optional[str] = None,
    color_scale: str = "RdBu_r"
) -> go.Figure:
    """Create a correlation heatmap.

    Args:
        df: DataFrame with data
        columns: Columns to include (None for all numeric)
        title: Chart title
        subtitle: Chart subtitle (Plotly v6)
        color_scale: Color scale

    Returns:
        Plotly Figure
    """
    if columns:
        corr_df = df[columns].select_dtypes(include=[np.number])
    else:
        corr_df = df.select_dtypes(include=[np.number])

    corr_matrix = corr_df.corr()

    fig = px.imshow(
        corr_matrix,
        title=title,
        color_continuous_scale=color_scale,
        aspect="auto",
        text_auto=".2f"
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title=""
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_pair_plot(
    df: pd.DataFrame,
    columns: list[str],
    color_col: Optional[str] = None,
    title: str = "Pair Plot",
    subtitle: Optional[str] = None
) -> go.Figure:
    """Create a scatter matrix (pair plot).

    Args:
        df: DataFrame with data
        columns: Columns to include
        color_col: Optional column for color encoding
        title: Chart title
        subtitle: Chart subtitle (Plotly v6)

    Returns:
        Plotly Figure
    """
    theme = get_plotly_theme()
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        color=color_col,
        title=title,
        template=theme["template"]
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    fig.update_traces(diagonal_visible=False)

    return apply_impulator_theme(fig)


def render_chart_with_viewer(
    fig: go.Figure,
    chart_id: str = "chart",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    z_col: Optional[str] = None,
    name_col: Optional[str] = None,
    show_hint: bool = True,
    width: str = 'stretch'
) -> None:
    """Render a Plotly chart with structure viewer integration.

    Args:
        fig: Plotly Figure
        chart_id: Unique identifier for the chart
        x_col: X-axis column name
        y_col: Y-axis column name
        z_col: Z-axis column name (for 3D)
        name_col: Name/ID column
        show_hint: Whether to show the viewer hint
        width: Chart width - 'stretch' for full width, 'content' for auto
    """
    if show_hint:
        render_structure_viewer_hint()

    st.plotly_chart(fig, width=width, key=f"{chart_id}_plotly")

    embed_structure_viewer(
        chart_id=chart_id,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        name_col=name_col
    )


def get_available_chart_types() -> dict[str, str]:
    """Get dictionary of available chart types.

    Returns:
        Dictionary mapping chart type names to descriptions
    """
    return {
        "Scatter Plot": "Show relationship between two variables",
        "Histogram": "Show distribution of a single variable",
        "Box Plot": "Show distribution with quartiles",
        "Violin Plot": "Show distribution with density",
        "Bar Chart": "Compare values across categories",
        "3D Scatter": "Show relationship between three variables",
        "Correlation Heatmap": "Show correlations between all numeric columns",
        "Pair Plot": "Show all pairwise relationships",
    }


def render_chart_controls(
    df: pd.DataFrame,
    key_prefix: str = "chart"
) -> dict[str, Any]:
    """Render chart configuration controls.

    Args:
        df: DataFrame with data
        key_prefix: Prefix for widget keys

    Returns:
        Dictionary of selected options
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    config = {}

    col1, col2 = st.columns(2)

    with col1:
        config['chart_type'] = st.selectbox(
            "Chart Type",
            options=list(get_available_chart_types().keys()),
            key=f"{key_prefix}_type"
        )

        config['x_col'] = st.selectbox(
            "X Axis",
            options=numeric_cols,
            key=f"{key_prefix}_x"
        )

    with col2:
        config['y_col'] = st.selectbox(
            "Y Axis",
            options=numeric_cols,
            index=min(1, len(numeric_cols) - 1),
            key=f"{key_prefix}_y"
        )

        config['color_col'] = st.selectbox(
            "Color By",
            options=["None"] + all_cols,
            key=f"{key_prefix}_color"
        )

    if config['color_col'] == "None":
        config['color_col'] = None

    return config
