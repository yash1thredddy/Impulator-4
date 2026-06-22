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


# Canonical IMP component-contribution columns on combined_activities.csv
# (verified imp_scoring.py:664-673 / RESEARCH §"View-time IMP component
# breakdown"). These are AGGREGATED at view time (groupby mean/max), never
# recomputed in the frontend — they already exist as columns. The base
# Efficiency_Contribution + the four geometry/interference contributions +
# the QED impact sum (plus a base) into IMP_Final_Score.
IMP_COMPONENT_COLS = [
    "Efficiency_Contribution",
    "Distance_Contribution",
    "Angle_Contribution",
    "Interference_Contribution",
    "PDB_Contribution",
    "QED_Impact",
]


def create_chemical_space_scatter(
    coords_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    imp_col: str = "IMP_Final_Score",
    size_col: Optional[str] = "n_targets",
    name_col: Optional[str] = "compound_name",
    cluster_col: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    opacity: float = 0.8,
) -> go.Figure:
    """Chemical-space scatter — each collection member is a single point.

    Consumes the 2-D projection frame produced by the Stage-1 projection
    compute (24-03 — PCA / UMAP coordinates over the descriptor matrix). The
    per-geometry color rule (UI-SPEC ⭐) applies: **member = a point**, so the
    color channel is owned by the **IMP score on a Viridis CONTINUOUS
    gradient — NO discrete bands** (Phase 21 VIZ-03 / PRES-07). Marker size
    encodes the member's distinct-target count, and the hover tooltip shows
    the IMP score as an **integer 0–100** (Phase 21 — no severity word).

    An optional ``cluster_col`` outlines 2-D GMM clusters (collection-scale
    clustering) by mapping the cluster label onto the marker outline; the IMP
    color channel is preserved (clusters are an *outline*, never the fill).

    Empty / missing input: when ``coords_df`` is empty or ``None`` an empty
    themed :class:`~plotly.graph_objects.Figure` is returned
    (``len(fig.data) == 0``) so the page can branch and render the
    "Need at least 3 members to map the chemical space" ``st.info`` copy
    instead of crashing the render (mitigates T-24-07-01).

    Args:
        coords_df: One row per member, with projection coordinates, an IMP
            score column, and (optionally) a distinct-target count and a
            cluster label.
        x_col: Projection x-coordinate column. Defaults to ``"x"``.
        y_col: Projection y-coordinate column. Defaults to ``"y"``.
        imp_col: IMP score column (drives the Viridis continuous color).
            Defaults to ``"IMP_Final_Score"``.
        size_col: Distinct-target count column for marker size, or ``None``
            for a fixed marker size. Defaults to ``"n_targets"``.
        name_col: Member-name column for the hover label. Defaults to
            ``"compound_name"``.
        cluster_col: Optional 2-D GMM cluster-label column drawn as the marker
            outline. Defaults to ``None`` (no cluster outline).
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).
        opacity: Marker opacity. Defaults to ``0.8``.

    Returns:
        A themed Plotly ``go.Figure``, or an empty themed figure when there is
        nothing to plot. Calls :func:`apply_impulator_theme` as the LAST step.
    """
    # Empty / no-member input -> caller branches on len(fig.data) and shows
    # the "need ≥3 members" sentinel instead of a crash (T-24-07-01).
    if coords_df is None or coords_df.empty:
        fig = go.Figure()
    else:
        plot_df = coords_df.copy()

        # Resolve the optional size channel. px.scatter raises on NaN sizes,
        # so coerce-and-fill the distinct-target count before plotting; drop
        # the channel entirely when the column is absent.
        effective_size = size_col if (size_col and size_col in plot_df.columns) else None
        if effective_size is not None:
            plot_df[effective_size] = (
                pd.to_numeric(plot_df[effective_size], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )

        # The cluster label (if present) drives the marker OUTLINE only — the
        # IMP Viridis fill is never displaced (per-geometry color rule). The
        # cryptic "cluster 0/1/2" labels are remapped to plain-language
        # "Structural group A/B/C" so the symbol legend reads clearly
        # (UI-SPEC §"🧭 Chemical Space").
        symbol_col = (
            cluster_col if (cluster_col and cluster_col in plot_df.columns) else None
        )
        if symbol_col is not None:
            symbol_col = "Structural group"
            plot_df[symbol_col] = (
                "Structural group "
                + plot_df[cluster_col]
                .astype("category")
                .cat.codes.map(lambda i: chr(ord("A") + int(i)) if i >= 0 else "?")
            )

        hover_name = name_col if (name_col and name_col in plot_df.columns) else None

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=imp_col,  # member = point -> IMP owns color
            size=effective_size,
            symbol=symbol_col,
            hover_name=hover_name,
            color_continuous_scale="Viridis",  # CONTINUOUS, no bands (VIZ-03)
            opacity=opacity,
            title=title or "Chemical Space",
        )

        # IMP score co-presents as an integer in the hover so the gradient is
        # never the sole carrier of the score (Phase 21 — integer 0–100).
        fig.update_traces(
            hovertemplate=(
                f"{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}"
                "<br>IMP: %{marker.color:.0f}<extra></extra>"
            )
        )

        theme = get_plotly_theme()
        fig.update_layout(
            # Real human-readable axis titles — the projection axes are
            # unitless similarity coordinates, NOT the literal "x"/"y" the raw
            # default column names produced (UI-SPEC §"🧭 Chemical Space").
            xaxis_title="Similarity axis 1 (no units)",
            yaxis_title="Similarity axis 2 (no units)",
            template=theme["template"],
            hovermode="closest",
            # Separate the IMP colorbar from the symbol legend so they no longer
            # overlap: the colorbar sits flush right, the symbol legend floats
            # in the top-left plot corner.
            coloraxis_colorbar=dict(title="IMP", x=1.02, xanchor="left"),
            legend=dict(
                title_text="Structural group",
                x=0.01,
                xanchor="left",
                y=0.99,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
            ),
        )

        if subtitle:
            _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


# IMP band boundaries (Phase 21 integer 0–100 space) drawn as thin grey
# reference hlines on the decision map (SPEC §0.10 / RR-9a). These are the
# 0.30/0.50/0.70/0.90 float bands scaled to integer space; they are PURELY
# visual reference lines — never used to tint quadrants by band (PRES-07).
DECISION_MAP_BANDS = (30, 50, 70, 90)

# Fixed Promise vertical divider (D-25-MAP-SPLIT) — a stable reference at the
# midpoint of the 0–100 Promise axis, NOT a per-set median.
DECISION_MAP_PROMISE_DIVIDER = 50


def create_decision_map(
    df: pd.DataFrame,
    *,
    promise_col: str,
    imp_col: str,
    smiles_col: Optional[str] = None,
    name_col: str = "compound_name",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Promise × IMP decision map — IMP is the ONLY color channel (PRES-07).

    Each collection member is a single point on a fixed ``Promise`` (x, 0–100)
    vs ``IMP`` (y, 0–100) plane. Per the per-geometry color rule (UI-SPEC ⭐)
    a **member = a point**, so the color channel is owned by the **IMP score on
    a Viridis CONTINUOUS gradient — NO discrete bands** (Phase 21 VIZ-03 /
    PRES-07). The map composes several net-new decision-support primitives over
    the base scatter (SPEC §0.10 / D-25-MAP-SPLIT / RR-9a):

    * a **fixed** vertical divider at ``Promise = 50``
      (:data:`DECISION_MAP_PROMISE_DIVIDER`) — NOT a per-set median,
    * thin **grey** horizontal band reference lines at the IMP bands
      :data:`DECISION_MAP_BANDS` (30/50/70/90),
    * neutral low-opacity **grey** quadrant shading rectangles (never tinted by
      IMP band),
    * persistent **text** y-axis annotations ("↑ more suspicious" /
      "↓ more genuine") — text, never banded color.

    SMILES (when ``smiles_col`` is given) rides in ``customdata`` so the page
    can wire structure-viewer hover downstream (D-25-HOVER). Members with a NaN
    Promise (x) or IMP (y) coordinate are dropped SILENTLY by Plotly — that is
    the accepted behavior (they already surface as greyed "insufficient data"
    in the verdict table); coordinates are never imputed to 0.

    Empty / missing input: when ``df`` is empty or ``None`` an empty themed
    :class:`~plotly.graph_objects.Figure` is returned (``len(fig.data) == 0``)
    so the page can branch instead of crashing the render.

    Args:
        df: One row per member, with a Promise column, an IMP column, and
            (optionally) a SMILES column and a member-name column.
        promise_col: Promise (x-axis, 0–100) column.
        imp_col: IMP score (y-axis, 0–100) column — drives the Viridis color.
        smiles_col: Optional SMILES column placed in ``customdata`` for hover.
        name_col: Member-name column for the hover label. Defaults to
            ``"compound_name"``.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).

    Returns:
        A themed Plotly ``go.Figure``, or an empty themed figure when there is
        nothing to plot. Calls :func:`apply_impulator_theme` as the LAST step.
    """
    # Empty / no-member input -> caller branches on len(fig.data).
    if df is None or df.empty:
        return apply_impulator_theme(go.Figure())

    plot_df = df.copy()

    # SMILES rides in customdata for downstream structure-viewer hover wiring.
    if smiles_col and smiles_col in plot_df.columns:
        plot_df, customdata_cols = prepare_chart_customdata(
            plot_df, smiles_col, name_col if name_col in plot_df.columns else None
        )
    else:
        customdata_cols = None

    hover_name = name_col if name_col in plot_df.columns else None

    fig = px.scatter(
        plot_df,
        x=promise_col,
        y=imp_col,
        color=imp_col,  # member = point -> IMP owns color (PRES-07)
        hover_name=hover_name,
        color_continuous_scale="Viridis",  # CONTINUOUS, no bands (VIZ-03)
        custom_data=customdata_cols if customdata_cols else None,
        title=title or "Decision map",
    )
    fig.update_traces(marker=dict(size=11, opacity=0.85))

    # Neutral grey quadrant shading (the four Promise<>50 × IMP<>50 quadrants).
    # Drawn BELOW the data; never tinted by IMP band (PRES-07).
    for x0, x1 in ((0, DECISION_MAP_PROMISE_DIVIDER), (DECISION_MAP_PROMISE_DIVIDER, 100)):
        for y0, y1 in ((0, 50), (50, 100)):
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor="rgba(128,128,128,0.06)",
                line_width=0,
                layer="below",
            )

    # Fixed Promise=50 vertical divider (D-25-MAP-SPLIT).
    fig.add_vline(
        x=DECISION_MAP_PROMISE_DIVIDER,
        line=dict(color="rgba(128,128,128,0.5)", width=1.5, dash="dash"),
    )

    # Thin grey IMP band reference hlines at 30/50/70/90.
    for band in DECISION_MAP_BANDS:
        fig.add_hline(
            y=band,
            line=dict(color="rgba(128,128,128,0.35)", width=1),
        )

    # Persistent TEXT y-axis annotations (text, NOT banded color).
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=-0.02,
        y=1.0,
        showarrow=False,
        text="↑ more suspicious",
        textangle=-90,
        font=dict(size=12, color="#6b7280"),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=-0.02,
        y=0.0,
        showarrow=False,
        text="↓ more genuine",
        textangle=-90,
        font=dict(size=12, color="#6b7280"),
    )

    theme = get_plotly_theme()
    fig.update_layout(
        xaxis_title="Promise",
        yaxis_title="IMP",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
        template=theme["template"],
        hovermode="closest",
        coloraxis_colorbar=dict(title="IMP"),
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


# Component weights + raw-score columns for the IMP base score, VERBATIM from
# the shipped _render_contribution_chart (compound_detail.py:3585-3589 /
# D-25-RAW-SCORES). Order is load-bearing for the contribution bar and the
# 5-spoke radar axis labels.
IMP_RAW_SCORE_COMPONENTS = (
    ("Efficiency", 0.45, "Efficiency_Score"),
    ("Distance", 0.20, "Distance_Score"),
    ("Angle", 0.15, "Angle_Score"),
    ("Interference", 0.15, "Interference_Score"),
    ("PDB", 0.05, "PDB_Score"),
)


def create_efficiency_plane(
    df: pd.DataFrame,
    *,
    sei_col: str = "SEI",
    bei_col: str = "BEI",
    imp_col: str,
    distance_col: str = "Distance_Score",
    smiles_col: Optional[str] = None,
    name_col: str = "compound_name",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """SEI × BEI efficiency plane — Viridis recolor + best-in-class watch marker.

    Mirrors the STRUCTURE of
    :func:`compound_detail._render_report_efficiency_plane` (equal-axis scaling;
    a 45° optimal dashed reference line; an angle banner classifying the mean
    development trajectory as OPTIMAL 40–50° / ACCEPTABLE 30–40° & 50–60° /
    UNBALANCED otherwise). Two pieces are NET-NEW relative to the source:

    * **Re-color (RR-2 / D-25-EFFPLANE-COLOR):** the source uses a red-green
      diverging colorscale; this factory uses **Viridis CONTINUOUS** (a naive
      copy of the red-green scale would violate PRES-07 by re-encoding the
      score as a qualitative good/bad color).
    * **Watch marker (RR-1 / D-25-EFFPLANE-MARKER):** the member at
      ``argmax(Distance_Score)`` is highlighted as
      **"Closest to best-in-class (watch)"** — NOT ``argmax(modulus)``
      (√(SEI²+BEI²)); §4.2 prose drift is rejected. The source's mean-point
      star is intentionally dropped; this marker does not exist in source.

    SMILES (when ``smiles_col`` is given) rides in ``customdata`` for hover
    (D-25-HOVER). Members with a NaN SEI/BEI coordinate are dropped SILENTLY by
    Plotly (no imputation).

    Empty / missing input: an empty themed
    :class:`~plotly.graph_objects.Figure` is returned (``len(fig.data) == 0``).

    Args:
        df: One row per member, with SEI / BEI / IMP / Distance_Score columns
            and (optionally) a SMILES and member-name column.
        sei_col: SEI (x-axis) column. Defaults to ``"SEI"``.
        bei_col: BEI (y-axis) column. Defaults to ``"BEI"``.
        imp_col: IMP score column driving the Viridis continuous color.
        distance_col: Distance score column whose argmax selects the watch
            marker. Defaults to ``"Distance_Score"``.
        smiles_col: Optional SMILES column placed in ``customdata`` for hover.
        name_col: Member-name column for the hover label. Defaults to
            ``"compound_name"``.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).

    Returns:
        A themed Plotly ``go.Figure``, or an empty themed figure when there is
        nothing to plot. Calls :func:`apply_impulator_theme` as the LAST step.
    """
    # Empty / no-member input -> caller branches on len(fig.data).
    if df is None or df.empty or sei_col not in df.columns or bei_col not in df.columns:
        return apply_impulator_theme(go.Figure())

    plot_df = df.copy()
    plot_df[sei_col] = pd.to_numeric(plot_df[sei_col], errors="coerce")
    plot_df[bei_col] = pd.to_numeric(plot_df[bei_col], errors="coerce")
    # Drop members with no SEI/BEI coordinate (silent drop — no imputation).
    plot_df = plot_df.dropna(subset=[sei_col, bei_col])
    if plot_df.empty:
        return apply_impulator_theme(go.Figure())

    # SMILES rides in customdata for downstream structure-viewer hover wiring.
    if smiles_col and smiles_col in plot_df.columns:
        plot_df, customdata_cols = prepare_chart_customdata(
            plot_df, smiles_col, name_col if name_col in plot_df.columns else None
        )
    else:
        customdata_cols = None

    hover_name = name_col if name_col in plot_df.columns else None
    has_imp = imp_col in plot_df.columns

    # Base scatter as px (so the Viridis scale lands on layout.coloraxis,
    # matching the chemical-space / pareto idiom — the 45° line and watch
    # marker are added as overlay go.Scatter traces).
    fig = px.scatter(
        plot_df,
        x=sei_col,
        y=bei_col,
        color=imp_col if has_imp else None,  # member = point -> IMP owns color
        hover_name=hover_name,
        color_continuous_scale="Viridis",  # CONTINUOUS red-green replaced (RR-2)
        custom_data=customdata_cols if customdata_cols else None,
        title=title or "Efficiency plane: SEI vs BEI",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8), name="Members")

    # 45° optimal development reference line (mirrors source 7137-7148).
    max_val = max(float(plot_df[sei_col].max()), float(plot_df[bei_col].max())) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="green", dash="dash", width=2),
            name="45° Optimal",
            hoverinfo="skip",
        )
    )

    # Best-in-class watch marker at argmax(Distance_Score) — RR-1.
    if distance_col in plot_df.columns:
        dist = pd.to_numeric(plot_df[distance_col], errors="coerce")
        if dist.notna().any():
            watch_row = plot_df.loc[dist.idxmax()]
            fig.add_trace(
                go.Scatter(
                    x=[float(watch_row[sei_col])],
                    y=[float(watch_row[bei_col])],
                    mode="markers",
                    marker=dict(
                        size=18,
                        symbol="star",
                        color="rgba(0,0,0,0)",
                        line=dict(width=2.5, color="#d6336c"),
                    ),
                    name="Closest to best-in-class (watch)",
                    hovertemplate=(
                        "Closest to best-in-class (watch)"
                        "<br>SEI: %{x:.2f}<br>BEI: %{y:.2f}<extra></extra>"
                    ),
                )
            )

    # Angle banner (mirrors source 7088-7096) as a figure annotation — a pure
    # factory cannot st.markdown the banner the source renders.
    mean_sei = float(plot_df[sei_col].mean())
    mean_bei = float(plot_df[bei_col].mean())
    mean_angle = float(np.arctan2(mean_bei, mean_sei) * 180.0 / np.pi)
    if 40 <= mean_angle <= 50:
        angle_status, angle_color = "OPTIMAL ✓", "#28a745"
    elif 30 <= mean_angle < 40 or 50 < mean_angle <= 60:
        angle_status, angle_color = "ACCEPTABLE", "#fd7e14"
    else:
        angle_status, angle_color = "UNBALANCED ⚠️", "#dc3545"
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.08,
        showarrow=False,
        text=f"Development trajectory: {angle_status} (Angle: {mean_angle:.1f}°)",
        font=dict(size=13, color=angle_color),
        align="left",
    )

    # Equal-axis scaling so the visual angle matches the calculated angle.
    theme = get_plotly_theme()
    fig.update_layout(
        xaxis=dict(
            title="SEI (Surface Efficiency Index)",
            scaleanchor="y",
            scaleratio=1,
            range=[0, max_val],
            autorange=False,
            constrain="domain",
        ),
        yaxis=dict(
            title="BEI (Binding Efficiency Index)",
            range=[0, max_val],
            autorange=False,
            constrain="domain",
        ),
        template=theme["template"],
        hovermode="closest",
    )
    if has_imp:
        fig.update_layout(coloraxis_colorbar=dict(title="IMP"))

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_imp_component_breakdown(
    per_member_df: pd.DataFrame,
    *,
    name_col: str = "compound_name",
    component_cols: Optional[list[str]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Per-member IMP component-breakdown comparison bar (IMP Analysis view).

    Renders the six IMP component contributions
    (:data:`IMP_COMPONENT_COLS`) per collection member as a **grouped** bar
    chart so members can be compared component-by-component. These are the
    AGGREGATED per-member contributions produced by the Stage-1 view-time
    aggregation (24-02 — ``groupby(name).mean()``); they are **never
    recomputed** here, only plotted.

    Per-geometry color rule (UI-SPEC ⭐): here a **member = a series**, so the
    color channel is owned by **member identity** (Plotly qualitative
    palette), NOT the IMP Viridis gradient — each member keeps a consistent
    qualitative color across all series-overlay charts.

    Empty / missing input: when ``per_member_df`` is empty / ``None`` or none
    of the component columns are present, an empty themed
    :class:`~plotly.graph_objects.Figure` is returned (``len(fig.data) == 0``)
    so the page can branch on it (mitigates T-24-07-01).

    Args:
        per_member_df: One row per member (indexed or with ``name_col``), one
            column per IMP component contribution.
        name_col: Member-name column. If absent, the frame index is used as
            the member identity. Defaults to ``"compound_name"``.
        component_cols: Component-contribution columns to plot. Defaults to
            :data:`IMP_COMPONENT_COLS`; absent columns are skipped.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).

    Returns:
        A themed Plotly grouped-bar ``go.Figure``, or an empty themed figure
        when there is nothing to plot. Calls :func:`apply_impulator_theme` as
        the LAST step.
    """
    cols = component_cols if component_cols is not None else IMP_COMPONENT_COLS

    # Empty / no-member / no-component input -> caller branches on
    # len(fig.data) (T-24-07-01).
    if per_member_df is None or per_member_df.empty:
        fig = go.Figure()
    else:
        df = per_member_df.copy()

        # Resolve the member-identity column; fall back to the index.
        if name_col in df.columns:
            members = df[name_col].astype(str)
        else:
            members = pd.Series(df.index.astype(str), index=df.index)

        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            fig = go.Figure()
        else:
            # Long form: one (member, component, value) row per cell so the
            # grouped bar can color by member (member = series -> qualitative).
            long_df = (
                df[present_cols]
                .assign(**{name_col: members})
                .melt(
                    id_vars=[name_col],
                    value_vars=present_cols,
                    var_name="Component",
                    value_name="Contribution",
                )
            )
            long_df["Contribution"] = pd.to_numeric(
                long_df["Contribution"], errors="coerce"
            ).fillna(0.0)
            # Human-readable component labels on the x-axis.
            long_df["Component"] = long_df["Component"].str.replace("_", " ")

            fig = px.bar(
                long_df,
                x="Component",
                y="Contribution",
                color=name_col,  # member = series -> qualitative identity
                barmode="group",
                title=title or "IMP Component Breakdown",
            )

            theme = get_plotly_theme()
            fig.update_layout(
                xaxis_title="Component",
                yaxis_title="Contribution",
                template=theme["template"],
                legend_title_text="Member",
            )

            if subtitle:
                _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_compare_radar(
    per_member_norm: pd.DataFrame,
    *,
    name_col: str = "compound_name",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Compare-mode radar / spider overlay (2–5 members, normalized axes).

    Renders one ``go.Scatterpolar`` trace **per member** overlaid on a shared
    set of normalized radial axes (D-COMPARE-CAP — Compare mode handles 2–5
    members; UI-SPEC §1 row 4 / Component Inventory "Radar / spider"). The
    factory is deliberately **dumb**: it expects a frame that is ALREADY
    min-max normalized to ``[0, 1]`` per axis (the caller in ``collections.py``
    owns the normalization so the factory stays IO-free and unit-testable).

    Per-geometry color rule (UI-SPEC ⭐): a **member = a series/trace**, so the
    color channel is owned by **member identity** (Plotly qualitative palette),
    NOT the IMP Viridis gradient. Because this is a ``go`` (not ``px``) figure,
    the qualitative color is assigned EXPLICITLY by sorted-member order so a
    given member keeps the SAME color it gets in every other series-overlay
    chart (box/violin/component-breakdown) — the "consistent member→color"
    contract.

    Empty / single-member input: an empty themed
    :class:`~plotly.graph_objects.Figure` is returned (``len(fig.data) == 0``)
    so the caller can branch and render the Compare-mode "pick 2 members" copy
    instead of a degenerate single-trace radar.

    Args:
        per_member_norm: One row per member (with ``name_col``), one column per
            radar axis, every axis value already normalized to ``[0, 1]``.
        name_col: Member-identity column. Defaults to ``"compound_name"``.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).

    Returns:
        A themed Plotly ``go.Figure`` of overlaid ``go.Scatterpolar`` traces, or
        an empty themed figure when there is nothing to plot. Calls
        :func:`apply_impulator_theme` as the LAST step.
    """
    # Empty / no-axis / single-member input -> caller branches on len(fig.data).
    if (
        per_member_norm is None
        or per_member_norm.empty
        or name_col not in per_member_norm.columns
    ):
        return apply_impulator_theme(go.Figure())

    axis_cols = [c for c in per_member_norm.columns if c != name_col]
    if not axis_cols:
        return apply_impulator_theme(go.Figure())

    # Stable, sorted member order so the qualitative palette maps the SAME member
    # to the SAME color across reruns and across every series-overlay chart.
    members = sorted(str(m) for m in per_member_norm[name_col].astype(str))
    palette = px.colors.qualitative.Plotly
    theta = [c.replace("_", " ") for c in axis_cols]

    fig = go.Figure()
    for idx, member in enumerate(members):
        row = per_member_norm[per_member_norm[name_col].astype(str) == member]
        if row.empty:
            continue
        values = pd.to_numeric(
            row[axis_cols].iloc[0], errors="coerce"
        ).fillna(0.0).tolist()
        color = palette[idx % len(palette)]
        # Close the polygon by repeating the first axis (Scatterpolar idiom).
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=theta + theta[:1],
                mode="lines+markers",
                fill="toself",
                name=member,
                line=dict(color=color),
                opacity=0.6,
            )
        )

    fig.update_layout(
        title=title or "Member comparison (radar)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        legend_title_text="Member",
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_imp_component_radar(
    per_member_norm: pd.DataFrame,
    *,
    name_col: str = "compound_name",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """5-spoke IMP component radar overlay (top-5-by-IMP members).

    A thin pass-through to :func:`create_compare_radar` (D-25-RADAR / RR-3):
    the factory expects a frame ALREADY min-max normalized to ``[0, 1]`` per
    axis (the top-5-by-IMP selection + raw-``*_Score`` → ``[0, 1]``
    normalization happen caller-side in Plan 02's collection_promise helpers;
    this factory just renders the overlay).

    The radar has EXACTLY **FIVE** spokes — the IMP base components
    (Efficiency-outlier · Distance-to-best · Development-angle ·
    Assay-interference · PDB-evidence). There is **NO 6th "QED-impact" spoke**:
    QED is the IMP MULTIPLIER (``Base × QED = IMP``), not a radar axis; its
    gloss belongs ONLY to the weighted-contribution bar
    (:func:`create_imp_contribution_bar`), never here. Ground truth: the shipped
    ``_render_contribution_chart`` radar (compound_detail.py:3584-3590) has
    exactly 5 spokes; QED is the bar multiplier at :3661,3685.

    Args:
        per_member_norm: One row per member (with ``name_col``), one column per
            of the 5 radar axes, every value already normalized to ``[0, 1]``.
        name_col: Member-identity column. Defaults to ``"compound_name"``.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).

    Returns:
        A themed Plotly ``go.Figure`` of overlaid ``go.Scatterpolar`` traces, or
        an empty themed figure when there is nothing to plot.
    """
    return create_compare_radar(
        per_member_norm,
        name_col=name_col,
        title=title or "IMP component radar",
        subtitle=subtitle,
    )


def create_promise_contribution_bar(
    contrib_df: pd.DataFrame,
    *,
    name_col: str = "compound_name",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Per-member STACKED Promise-decomposition bar, colored by Promise component.

    Mirrors the melt-to-long ``px.bar`` grammar of
    :func:`create_imp_component_breakdown` BUT uses ``barmode="stack"`` colored
    by **Promise COMPONENT** per member (the breakdown is grouped-by-member; the
    Promise transparency bar is stacked-by-component) — showing each member's
    Promise decomposition (SPEC §4.0c). The plain-language Promise component
    labels (Potency · Ligand efficiency · Apparent promiscuity (recorded active
    targets) · Cleanliness · Druglikeness) are the contribution columns of
    ``contrib_df``. The rounded-integer Promise display
    (D-25-DECODE-PRECISION) is caller-side; this factory just stacks the
    contributions.

    Per-geometry color rule (UI-SPEC ⭐): here the color channel is owned by the
    **Promise component** (the stack segments), NOT the IMP Viridis gradient.

    Empty / missing input: an empty themed
    :class:`~plotly.graph_objects.Figure` is returned (``len(fig.data) == 0``).

    Args:
        contrib_df: One row per member (with ``name_col``), one column per
            Promise component contribution.
        name_col: Member-name column. If absent, the frame index is used.
            Defaults to ``"compound_name"``.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).

    Returns:
        A themed Plotly stacked-bar ``go.Figure``, or an empty themed figure
        when there is nothing to plot. Calls :func:`apply_impulator_theme` as
        the LAST step.
    """
    if contrib_df is None or contrib_df.empty:
        return apply_impulator_theme(go.Figure())

    df = contrib_df.copy()

    if name_col in df.columns:
        members = df[name_col].astype(str)
    else:
        members = pd.Series(df.index.astype(str), index=df.index)

    component_cols = [c for c in df.columns if c != name_col]
    if not component_cols:
        return apply_impulator_theme(go.Figure())

    # Long form: one (member, component, value) row per cell so the stacked bar
    # can color by component (component = stack segment).
    long_df = (
        df[component_cols]
        .assign(**{name_col: members})
        .melt(
            id_vars=[name_col],
            value_vars=component_cols,
            var_name="Component",
            value_name="Contribution",
        )
    )
    long_df["Contribution"] = pd.to_numeric(
        long_df["Contribution"], errors="coerce"
    ).fillna(0.0)

    fig = px.bar(
        long_df,
        x=name_col,
        y="Contribution",
        color="Component",  # component = stack segment -> color by component
        barmode="stack",
        title=title or "Promise decomposition",
    )

    theme = get_plotly_theme()
    fig.update_layout(
        xaxis_title="Member",
        yaxis_title="Promise contribution",
        template=theme["template"],
        legend_title_text="Promise component",
    )

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_imp_contribution_bar(
    member_row,
    *,
    title: str = "Weighted Contributions",
) -> go.Figure:
    """Single-member IMP weighted-contribution bar with the Base × QED subtitle.

    A PURE factory mirroring the BAR half of the shipped
    ``_render_contribution_chart`` (compound_detail.py:3659-3695, which is
    Streamlit-coupled via ``st.plotly_chart``). For ONE member row it plots the
    FIVE weighted IMP contributions (``weight × raw_*_Score``) as a single
    horizontal :class:`~plotly.graph_objects.Bar`, and renders the
    ``Base × QED = IMP`` relationship in the subtitle (SPEC §4.5 /
    D-25-PLAN-RESTATE). This is WHERE the QED-impact / Base×QED multiplier
    renders — it is the bar subtitle, NEVER a radar spoke.

    Components + weights are VERBATIM from the source
    (:data:`IMP_RAW_SCORE_COMPONENTS`): Efficiency 0.45 / Distance 0.20 /
    Angle 0.15 / Interference 0.15 / PDB 0.05, reading raw cols
    ``Efficiency_Score`` … ``PDB_Score`` (NaN → 0.0). The subtitle reads the
    ``QED_Multiplier`` column EXACTLY (NaN → 1.0), using the Unicode ``×``
    (U+00D7) per source.

    Empty / missing / all-NaN row: when ``member_row`` is ``None``, or the 5
    raw scores are absent / all-NaN (``sum(weighted_scores) <= 0.005``, mirroring
    the source short-circuit at :3604), an empty themed
    :class:`~plotly.graph_objects.Figure` is returned (``len(fig.data) == 0``) —
    NEVER an IndexError/KeyError.

    Args:
        member_row: A single member row (``pd.Series`` / mapping) carrying the
            5 raw ``*_Score`` columns and (optionally) ``QED_Multiplier``.
        title: Chart title. Defaults to ``"Weighted Contributions"``.

    Returns:
        A themed Plotly ``go.Figure`` with one horizontal bar of 5 weighted
        contributions, or an empty themed figure when there is nothing to plot.
        Calls :func:`apply_impulator_theme` as the LAST step.
    """
    # Guard None / empty FIRST — never raise on a missing member.
    if member_row is None:
        return apply_impulator_theme(go.Figure())
    if hasattr(member_row, "empty") and member_row.empty:
        return apply_impulator_theme(go.Figure())

    def _get(row, key, default):
        try:
            val = row.get(key, default)
        except AttributeError:
            val = row[key] if key in row else default
        return val

    names = []
    raw_scores = []
    weighted_scores = []
    weights = []
    for name, weight, col in IMP_RAW_SCORE_COMPONENTS:
        score = _get(member_row, col, 0.0)
        score = float(score) if pd.notna(score) else 0.0
        names.append(name)
        raw_scores.append(score)
        weighted_scores.append(weight * score)
        weights.append(weight)

    base_score = sum(weighted_scores)
    # All-NaN / empty-score row short-circuit (mirrors source :3604).
    if base_score <= 0.005:
        return apply_impulator_theme(go.Figure())

    qed_mult = _get(member_row, "QED_Multiplier", 1.0)
    qed_mult = float(qed_mult) if pd.notna(qed_mult) else 1.0

    contrib_colors = ["#3b82f6", "#22c55e", "#eab308", "#f97316", "#a855f7"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names,
            x=weighted_scores,
            orientation="h",
            marker_color=contrib_colors,
            text=[
                f"{w * 100:.0f}% × {s:.3f} = {ws:.3f}"
                for w, s, ws in zip(weights, raw_scores, weighted_scores)
            ],
            textposition="auto",
            textfont=dict(size=12),
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            subtitle=dict(
                text=(
                    f"Base: {base_score:.3f} × QED {qed_mult:.3f} = "
                    f"{base_score * qed_mult:.3f}"
                )
            ),
        ),
        xaxis_title="Contribution to Base Score",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )

    return apply_impulator_theme(fig)


def create_pareto_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    front_mask,
    *,
    imp_col: str = "IMP_Final_Score",
    name_col: Optional[str] = "compound_name",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Pareto / trade-off front scatter — members as points, IMP = Viridis color.

    Consumes the non-dominated mask produced by
    :func:`frontend.ui.components.collection_pareto.pareto_front` (the page flips
    any lower-is-better axis BEFORE calling that pure module, then passes the
    resulting boolean mask here). Every member is plotted as a point on the
    chosen 2-D trade-off plane (``x_col`` vs ``y_col``); the **front members are
    overlaid with a distinct marker** so the "best available trade-offs" are
    visually called out (UI-SPEC Component Inventory "Pareto front").

    Per-geometry color rule (UI-SPEC ⭐): a **member = a point**, so the color
    channel is owned by the **IMP score on a Viridis CONTINUOUS gradient — NO
    discrete bands** (Phase 21 VIZ-03 / PRES-07) — matching the Chemical Space
    scatter for visual consistency.

    Empty / missing input: an empty themed
    :class:`~plotly.graph_objects.Figure` is returned (``len(fig.data) == 0``)
    so the caller can branch and render the "Select 2 or more members" info.

    Args:
        df: One row per member, with the two trade-off axis columns, an IMP
            score column, and (optionally) a member-name column.
        x_col: Trade-off x-axis column.
        y_col: Trade-off y-axis column.
        front_mask: Boolean sequence (length == ``len(df)``, row-aligned) — True
            where the member is on the non-dominated Pareto front.
        imp_col: IMP score column driving the Viridis continuous color. Defaults
            to ``"IMP_Final_Score"``.
        name_col: Member-name column for the hover label. Defaults to
            ``"compound_name"``.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).

    Returns:
        A themed Plotly ``go.Figure``, or an empty themed figure when there is
        nothing to plot. Calls :func:`apply_impulator_theme` as the LAST step.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return apply_impulator_theme(go.Figure())

    plot_df = df.copy()
    hover_name = name_col if (name_col and name_col in plot_df.columns) else None
    has_imp = imp_col in plot_df.columns

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=imp_col if has_imp else None,  # member = point -> IMP owns color
        hover_name=hover_name,
        color_continuous_scale="Viridis",  # CONTINUOUS, no bands (VIZ-03)
        title=title or "Trade-off front",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85))

    # Overlay the non-dominated front with a distinct (open diamond) marker so
    # the best trade-offs read at a glance without displacing the IMP color.
    mask = np.asarray(list(front_mask), dtype=bool)
    if mask.shape[0] == len(plot_df) and mask.any():
        front_df = plot_df.loc[mask]
        fig.add_trace(
            go.Scatter(
                x=front_df[x_col],
                y=front_df[y_col],
                mode="markers",
                marker=dict(
                    symbol="diamond-open",
                    size=18,
                    line=dict(width=2.5, color="#d6336c"),
                    color="rgba(0,0,0,0)",
                ),
                name="Pareto front",
                hoverinfo="skip",
            )
        )

    theme = get_plotly_theme()
    fig.update_layout(
        xaxis_title=x_col.replace("_", " "),
        yaxis_title=y_col.replace("_", " "),
        template=theme["template"],
        hovermode="closest",
    )
    if has_imp:
        fig.update_layout(coloraxis_colorbar=dict(title="IMP"))

    if subtitle:
        _apply_subtitle(fig, subtitle)

    return apply_impulator_theme(fig)


def create_sar_matrix(
    matrix,
    labels: list[str],
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> go.Figure:
    """Pairwise Tanimoto similarity matrix (SAR-lite) — Viridis SEQUENTIAL.

    Consumes the symmetric ``(N, N)`` Tanimoto matrix from
    :func:`frontend.ui.components.collection_sar.tanimoto_matrix` (computed over
    the SELECTED member subset only — the O(N²) cost is multiselect-bounded,
    T-24-12-01) and renders it as a labelled heatmap (UI-SPEC Component
    Inventory "SAR-lite / activity-cliffs").

    Per-geometry color rule (UI-SPEC ⭐): a **matrix cell = a similarity value**,
    so the cell color is **Viridis SEQUENTIAL** — structural similarity has no
    meaningful midpoint, so a diverging scale would be misleading (matching the
    bioactivity heatmap convention).

    Empty input: an empty themed :class:`~plotly.graph_objects.Figure` is
    returned (``len(fig.data) == 0``) so the caller can branch and render the
    "No structurally-similar member pairs" info instead of an empty grid.

    Args:
        matrix: ``(N, N)`` symmetric similarity array (unit diagonal).
        labels: Length-``N`` member labels (row/column ticks), row-aligned with
            ``matrix``.
        title: Optional chart title.
        subtitle: Optional chart subtitle (Plotly v6).

    Returns:
        A themed Plotly ``go.Figure`` (a ``go.Heatmap``), or an empty themed
        figure when there is nothing to plot. Calls :func:`apply_impulator_theme`
        as the LAST step.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        return apply_impulator_theme(go.Figure())

    tick_labels = [_truncate_label(str(label)) for label in labels]
    full_labels = [str(label) for label in labels]

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=full_labels,
            y=full_labels,
            colorscale="Viridis",  # SEQUENTIAL — similarity has no midpoint
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Tanimoto"),
            hovertemplate=(
                "%{y} ↔ %{x}<br>Tanimoto: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or "Pairwise structural similarity (Tanimoto)",
        xaxis=dict(
            tickmode="array",
            tickvals=full_labels,
            ticktext=tick_labels,
            tickangle=-45,
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=60, b=120, l=80, r=10),
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
