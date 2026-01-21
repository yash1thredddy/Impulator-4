"""Chart components for molecular data visualization.

This module provides reusable chart components built on Plotly,
with support for the structure viewer integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any, Tuple

from frontend.ui.components.molecule_viewer import (
    embed_structure_viewer,
    render_structure_viewer_hint,
    prepare_chart_customdata,
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
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title or f"{y_col} vs {x_col}",
        color_continuous_scale=color_scale,
        opacity=opacity,
        custom_data=customdata_cols if customdata_cols else None,
        trendline="ols" if trendline else None
    )

    # Update marker size if not using size column
    if not size_col:
        fig.update_traces(marker=dict(size=marker_size))

    # Update layout
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' '),
        yaxis_title=y_col.replace('_', ' '),
        template="plotly_white",
        hovermode="closest"
    )

    return fig


def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    color_col: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """Create a histogram.

    Args:
        df: DataFrame with data
        column: Column to plot
        bins: Number of bins
        color_col: Optional column for color grouping
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = px.histogram(
        df,
        x=column,
        color=color_col,
        nbins=bins,
        title=title or f"Distribution of {column}",
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title=column.replace('_', ' '),
        yaxis_title="Count"
    )

    return fig


def create_box_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """Create a box plot.

    Args:
        df: DataFrame with data
        y_col: Y-axis column (values)
        x_col: Optional X-axis column (categories)
        color_col: Optional column for color grouping
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title or f"Distribution of {y_col}",
        template="plotly_white"
    )

    return fig


def create_violin_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """Create a violin plot.

    Args:
        df: DataFrame with data
        y_col: Y-axis column (values)
        x_col: Optional X-axis column (categories)
        color_col: Optional column for color grouping
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = px.violin(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        box=True,
        title=title or f"Distribution of {y_col}",
        template="plotly_white"
    )

    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
    orientation: str = "v"
) -> go.Figure:
    """Create a bar chart.

    Args:
        df: DataFrame with data
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Optional column for color encoding
        title: Chart title
        orientation: 'v' for vertical, 'h' for horizontal

    Returns:
        Plotly Figure
    """
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title or f"{y_col} by {x_col}",
        template="plotly_white",
        orientation=orientation
    )

    return fig


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

    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Heatmap",
    color_scale: str = "RdBu_r"
) -> go.Figure:
    """Create a correlation heatmap.

    Args:
        df: DataFrame with data
        columns: Columns to include (None for all numeric)
        title: Chart title
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

    return fig


def create_pair_plot(
    df: pd.DataFrame,
    columns: List[str],
    color_col: Optional[str] = None,
    title: str = "Pair Plot"
) -> go.Figure:
    """Create a scatter matrix (pair plot).

    Args:
        df: DataFrame with data
        columns: Columns to include
        color_col: Optional column for color encoding
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        color=color_col,
        title=title,
        template="plotly_white"
    )

    fig.update_traces(diagonal_visible=False)

    return fig


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


def get_available_chart_types() -> Dict[str, str]:
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
) -> Dict[str, Any]:
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
