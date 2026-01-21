"""
Statistical Outlier Detection Module

This module implements IQR-based outlier detection for efficiency metrics.

The Interquartile Range (IQR) method is robust to extreme values and is the
standard approach for identifying statistical outliers.

Method:
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1
- Upper threshold = Q3 + multiplier × IQR (default multiplier = 1.5)
- Outlier = value > upper threshold
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_iqr_threshold(
    data: pd.Series,
    multiplier: float = 1.5
) -> Tuple[float, float, float, float, float]:
    """
    Calculate IQR statistics and outlier threshold for a metric.

    Args:
        data: Series of metric values
        multiplier: IQR multiplier (default 1.5 = standard outlier detection)
                   - 1.5 → ~7% outliers (standard)
                   - 2.0 → ~1% outliers (stricter)
                   - 1.0 → ~15% outliers (more lenient)

    Returns:
        Tuple[float, float, float, float, float]: (Q1, Q3, IQR, lower_threshold, upper_threshold)

    Example:
        >>> q1, q3, iqr, lower, upper = calculate_iqr_threshold(df['SEI'])
        >>> print(f"Outliers: values > {upper:.2f}")
    """
    # Remove NaN values
    data_clean = data.dropna()

    if data_clean.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Calculate quartiles
    q1 = data_clean.quantile(0.25)
    q3 = data_clean.quantile(0.75)
    iqr = q3 - q1

    # Calculate thresholds
    lower_threshold = q1 - multiplier * iqr
    upper_threshold = q3 + multiplier * iqr

    return q1, q3, iqr, lower_threshold, upper_threshold


def flag_outliers_single_metric(
    data: pd.Series,
    multiplier: float = 1.5,
    direction: str = 'upper'
) -> pd.Series:
    """
    Flag outliers for a single metric using IQR method.

    Args:
        data: Series of metric values
        multiplier: IQR multiplier
        direction: 'upper' (default), 'lower', or 'both'
                  For efficiency metrics, we typically only care about upper outliers

    Returns:
        pd.Series: Boolean series (True = outlier, False = normal)

    Example:
        >>> df['Is_SEI_Outlier'] = flag_outliers_single_metric(df['SEI'])
    """
    q1, q3, iqr, lower_threshold, upper_threshold = calculate_iqr_threshold(data, multiplier)

    if np.isnan(upper_threshold):
        return pd.Series([False] * len(data), index=data.index)

    # Flag outliers based on direction
    if direction == 'upper':
        return data > upper_threshold
    elif direction == 'lower':
        return data < lower_threshold
    elif direction == 'both':
        return (data < lower_threshold) | (data > upper_threshold)
    else:
        raise ValueError(f"Invalid direction: {direction}. Use 'upper', 'lower', or 'both'")


def calculate_percentile_ranks(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate percentile ranks (0-100) for specified columns.

    Percentile rank indicates what percentage of values are below this value.

    Args:
        df: DataFrame with metrics
        columns: List of column names to calculate percentiles for

    Returns:
        pd.DataFrame: Input DataFrame with added percentile columns

    Example:
        >>> df = calculate_percentile_ranks(df, ['SEI', 'BEI', 'NSEI', 'NBEI'])
        >>> print(df[['SEI', 'SEI_Percentile']])
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame. Skipping.")
            continue

        # Calculate percentile rank (0-100)
        percentile_col = f'{col}_Percentile'
        df[percentile_col] = df[col].rank(pct=True) * 100

    return df


def detect_efficiency_outliers(
    df: pd.DataFrame,
    metrics: List[str] = None,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers for all efficiency metrics and add flags + counts.

    This is the main outlier detection function.

    Args:
        df: DataFrame with efficiency metrics
        metrics: List of metrics to check (default: ['SEI', 'BEI', 'NSEI', 'NBEI'])
        multiplier: IQR multiplier for threshold calculation

    Returns:
        pd.DataFrame: Input DataFrame with added columns:
            - Is_{metric}_Outlier: Boolean flag for each metric
            - {metric}_Percentile: Percentile rank for each metric
            - Outlier_Count: Total number of outlier flags (0-4)
            - Is_Efficiency_Outlier: True if ANY metric is an outlier

    Example:
        >>> df_with_outliers = detect_efficiency_outliers(df)
        >>> imp_candidates = df_with_outliers[df_with_outliers['Is_Efficiency_Outlier']]
        >>> print(f"Found {len(imp_candidates)} IMP candidates")
    """
    if metrics is None:
        metrics = ['SEI', 'BEI', 'NSEI', 'NBEI']

    df = df.copy()

    # Validate that metrics exist
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")

    # Flag outliers for each metric
    outlier_columns = []
    for metric in metrics:
        outlier_col = f'Is_{metric}_Outlier'
        df[outlier_col] = flag_outliers_single_metric(
            df[metric],
            multiplier=multiplier,
            direction='upper'  # High efficiency = good (outlier in positive direction)
        )
        outlier_columns.append(outlier_col)

    # Calculate percentile ranks
    df = calculate_percentile_ranks(df, metrics)

    # Count total outliers per row
    df['Outlier_Count'] = df[outlier_columns].sum(axis=1)

    # Composite flag: ANY outlier = efficiency outlier
    df['Is_Efficiency_Outlier'] = df[outlier_columns].any(axis=1)

    return df


def calculate_cohort_statistics(
    df: pd.DataFrame,
    metrics: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive statistics for each metric across the cohort.

    Useful for metadata and reporting.

    Args:
        df: DataFrame with efficiency metrics
        metrics: List of metrics to analyze (default: ['SEI', 'BEI', 'NSEI', 'NBEI'])

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with statistics

    Example:
        >>> stats = calculate_cohort_statistics(df)
        >>> print(f"SEI mean: {stats['SEI']['mean']:.2f}")
        >>> print(f"SEI outlier threshold: {stats['SEI']['outlier_threshold']:.2f}")
    """
    if metrics is None:
        metrics = ['SEI', 'BEI', 'NSEI', 'NBEI']

    statistics = {}

    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in DataFrame. Skipping.")
            continue

        data = df[metric].dropna()

        if data.empty:
            statistics[metric] = {
                'count': 0,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'q1': np.nan,
                'q3': np.nan,
                'iqr': np.nan,
                'outlier_threshold': np.nan
            }
            continue

        # Calculate IQR threshold
        q1, q3, iqr, _, upper_threshold = calculate_iqr_threshold(data, multiplier=1.5)

        statistics[metric] = {
            'count': len(data),
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(iqr),
            'outlier_threshold': float(upper_threshold)
        }

    return statistics


def get_outlier_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate summary statistics about outliers in the dataset.

    Args:
        df: DataFrame with outlier flags

    Returns:
        Dict: Summary information

    Example:
        >>> summary = get_outlier_summary(df)
        >>> print(f"Total compounds: {summary['total_compounds']}")
        >>> print(f"IMP candidates: {summary['efficiency_outliers']}")
        >>> print(f"Percentage: {summary['outlier_percentage']:.1f}%")
    """
    total = len(df)

    if 'Is_Efficiency_Outlier' not in df.columns:
        logger.warning("No outlier flags found in DataFrame")
        return {'total_compounds': total}

    efficiency_outliers = df['Is_Efficiency_Outlier'].sum()

    # Count outliers per metric
    outlier_counts = {}
    for col in df.columns:
        if col.startswith('Is_') and col.endswith('_Outlier') and col != 'Is_Efficiency_Outlier':
            metric_name = col.replace('Is_', '').replace('_Outlier', '')
            outlier_counts[metric_name] = int(df[col].sum())

    # Distribution of outlier counts
    if 'Outlier_Count' in df.columns:
        outlier_distribution = df['Outlier_Count'].value_counts().sort_index().to_dict()
    else:
        outlier_distribution = {}

    return {
        'total_compounds': total,
        'efficiency_outliers': int(efficiency_outliers),
        'outlier_percentage': (efficiency_outliers / total * 100) if total > 0 else 0,
        'outliers_per_metric': outlier_counts,
        'outlier_count_distribution': outlier_distribution
    }


def calculate_z_scores(df: pd.DataFrame, metrics: List[str] = None) -> pd.DataFrame:
    """
    Calculate Z-scores for efficiency metrics.

    Z-score = (value - mean) / std

    Used in O[Q/P/L]A scoring (Component 1).

    Args:
        df: DataFrame with efficiency metrics
        metrics: List of metrics to calculate Z-scores for

    Returns:
        pd.DataFrame: Input DataFrame with added Z-score columns

    Example:
        >>> df = calculate_z_scores(df, ['SEI', 'BEI', 'NSEI', 'NBEI'])
        >>> print(df[['SEI', 'SEI_Zscore']])
    """
    if metrics is None:
        metrics = ['SEI', 'BEI', 'NSEI', 'NBEI']

    df = df.copy()

    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found. Skipping.")
            continue

        # Calculate Z-score (guard against zero variance)
        z_col = f'{metric}_Zscore'
        std_val = df[metric].std()
        if std_val == 0 or pd.isna(std_val):
            # Constant metric - all values are the same, z-score is 0
            df[z_col] = 0.0
        else:
            df[z_col] = (df[metric] - df[metric].mean()) / std_val

    return df


def filter_outliers(
    df: pd.DataFrame,
    min_outlier_count: int = 2
) -> pd.DataFrame:
    """
    Filter DataFrame to return only rows with sufficient outlier flags.

    Default: Outlier_Count >= 2 (standard IMP candidate criterion)

    Args:
        df: DataFrame with outlier detection results
        min_outlier_count: Minimum number of outlier flags required

    Returns:
        pd.DataFrame: Filtered DataFrame with only outlier compounds

    Example:
        >>> imp_candidates = filter_outliers(df, min_outlier_count=2)
        >>> print(f"Found {len(imp_candidates)} IMP candidates")
    """
    if 'Outlier_Count' not in df.columns:
        raise ValueError("Outlier_Count column not found. Run detect_efficiency_outliers() first.")

    return df[df['Outlier_Count'] >= min_outlier_count].copy()
