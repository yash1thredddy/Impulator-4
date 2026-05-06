"""Chart components for molecular data visualization.

This module provides reusable chart components built on Plotly,
with support for the structure viewer integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Any

from frontend.ui.components.molecule_viewer import (
    embed_structure_viewer,
    render_structure_viewer_hint,
    prepare_chart_customdata,
)


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
    orientation: str = "v"
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
        orientation=orientation
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
