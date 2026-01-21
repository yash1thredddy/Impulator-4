"""
Efficiency Plane Geometry Calculation Module

This module implements geometric calculations in efficiency space (efficiency planes).

Two efficiency planes:
1. SEI-BEI Plane: Traditional efficiency plane
2. NSEI-NBEI Plane: Atom-normalized efficiency plane

For each plane, we calculate:
- Modulus: Overall efficiency magnitude (vector length)
- Angle: Development trajectory (0° = hydrophobic, 45° = optimal, 90° = polar)
- Slope: Physicochemical balance
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_modulus(x: float, y: float) -> float:
    """
    Calculate vector modulus (magnitude) in 2D efficiency space.

    Formula: |v| = sqrt(x² + y²)

    Args:
        x: X-component (e.g., SEI or NSEI)
        y: Y-component (e.g., BEI or NBEI)

    Returns:
        float: Modulus (overall efficiency magnitude) or NaN if invalid
    """
    if not (np.isnan(x) or np.isnan(y)):
        return np.sqrt(x**2 + y**2)
    return np.nan


def calculate_angle(x: float, y: float) -> float:
    """
    Calculate angle in efficiency plane (development trajectory).

    Formula: θ = arctan2(y, x) × 180/π

    Interpretation:
    - 0°: Improvement purely in x-axis (e.g., SEI) - hydrophobic
    - 45°: Balanced improvement (OPTIMAL)
    - 90°: Improvement purely in y-axis (e.g., BEI) - polar

    Args:
        x: X-component (e.g., SEI or NSEI)
        y: Y-component (e.g., BEI or NBEI)

    Returns:
        float: Angle in degrees (0-90°) or NaN if invalid
    """
    if not (np.isnan(x) or np.isnan(y)):
        return np.arctan2(y, x) * 180 / np.pi
    return np.nan


def calculate_sei_bei_plane_metrics(
    sei: float,
    bei: float,
    psa: float,
    molecular_weight: float
) -> Dict[str, float]:
    """
    Calculate all geometric metrics for the SEI-BEI efficiency plane.

    Vector formulation: vLEI = [SEI, BEI]

    Metrics:
    - Modulus: |vLEI| = sqrt(SEI² + BEI²) - overall efficiency magnitude
    - Angle: arctan2(BEI, SEI) - development trajectory
    - Slope: 10 × (PSA / MW) - physicochemical balance

    Args:
        sei: Surface Efficiency Index
        bei: Binding Efficiency Index
        psa: Polar surface area (Ų)
        molecular_weight: Molecular weight (Da)

    Returns:
        Dict[str, float]: Dictionary with Modulus_SEI_BEI, Angle_SEI_BEI, Slope_SEI_BEI
    """
    try:
        modulus = calculate_modulus(sei, bei)
        angle = calculate_angle(sei, bei)

        # Slope calculation: 10 × (PSA / MW)
        if molecular_weight and not np.isnan(molecular_weight) and molecular_weight > 0:
            slope = 10 * (psa / molecular_weight)
        else:
            slope = np.nan

        return {
            'Modulus_SEI_BEI': modulus,
            'Angle_SEI_BEI': angle,
            'Slope_SEI_BEI': slope
        }

    except Exception as e:
        logger.error(f"Error calculating SEI-BEI plane metrics: {str(e)}")
        return {
            'Modulus_SEI_BEI': np.nan,
            'Angle_SEI_BEI': np.nan,
            'Slope_SEI_BEI': np.nan
        }


def calculate_nsei_nbei_plane_metrics(
    nsei: float,
    nbei: float,
    npol: float,
    heavy_atoms: float
) -> Dict[str, float]:
    """
    Calculate all geometric metrics for the NSEI-NBEI efficiency plane.

    Also known as: Atom-Normalized Efficiency Plane

    Metrics:
    - Modulus: sqrt(NSEI² + NBEI²) - overall normalized efficiency
    - Angle: arctan2(NBEI, NSEI) - normalized development trajectory
    - Slope: NPOL / NHA - atom-based physicochemical ratio
    - Intercept: log(NHA) - reference point

    Args:
        nsei: Normalized Surface Efficiency Index
        nbei: Normalized Binding Efficiency Index
        npol: NPOL (N + O atom count)
        heavy_atoms: Number of heavy atoms (NHA)

    Returns:
        Dict[str, float]: Dictionary with Modulus_NSEI_NBEI, Angle_NSEI_NBEI,
                         Slope_NSEI_NBEI, Intercept_NSEI_NBEI
    """
    try:
        modulus = calculate_modulus(nsei, nbei)
        angle = calculate_angle(nsei, nbei)

        # Slope calculation: NPOL / NHA
        if heavy_atoms and not np.isnan(heavy_atoms) and heavy_atoms > 0:
            slope = npol / heavy_atoms
        else:
            slope = np.nan

        # Intercept: log(NHA)
        if heavy_atoms and not np.isnan(heavy_atoms) and heavy_atoms > 0:
            intercept = np.log10(heavy_atoms)
        else:
            intercept = np.nan

        return {
            'Modulus_NSEI_NBEI': modulus,
            'Angle_NSEI_NBEI': angle,
            'Slope_NSEI_NBEI': slope,
            'Intercept_NSEI_NBEI': intercept
        }

    except Exception as e:
        logger.error(f"Error calculating NSEI-NBEI plane metrics: {str(e)}")
        return {
            'Modulus_NSEI_NBEI': np.nan,
            'Angle_NSEI_NBEI': np.nan,
            'Slope_NSEI_NBEI': np.nan,
            'Intercept_NSEI_NBEI': np.nan
        }


def calculate_all_plane_metrics(
    sei: float,
    bei: float,
    nsei: float,
    nbei: float,
    psa: float,
    molecular_weight: float,
    npol: float,
    heavy_atoms: float
) -> Dict[str, float]:
    """
    Calculate geometric metrics for BOTH efficiency planes.

    This is the main function that calculates all plane geometry metrics.

    Args:
        sei: Surface Efficiency Index
        bei: Binding Efficiency Index
        nsei: Normalized Surface Efficiency Index
        nbei: Normalized Binding Efficiency Index
        psa: Polar surface area (Ų)
        molecular_weight: Molecular weight (Da)
        npol: NPOL (N + O atom count)
        heavy_atoms: Number of heavy atoms (NHA)

    Returns:
        Dict[str, float]: Dictionary containing all plane metrics (7 metrics total)

    Example:
        >>> plane_metrics = calculate_all_plane_metrics(
        ...     sei=15.2, bei=18.9, nsei=1.5, nbei=0.31,
        ...     psa=85.2, molecular_weight=342.1, npol=5, heavy_atoms=24
        ... )
        >>> print(plane_metrics['Angle_SEI_BEI'])  # 51.2°
        >>> print(plane_metrics['Modulus_SEI_BEI'])  # 24.3
    """
    try:
        # Calculate SEI-BEI plane metrics
        sei_bei_metrics = calculate_sei_bei_plane_metrics(
            sei, bei, psa, molecular_weight
        )

        # Calculate NSEI-NBEI plane metrics
        nsei_nbei_metrics = calculate_nsei_nbei_plane_metrics(
            nsei, nbei, npol, heavy_atoms
        )

        # Combine all metrics
        all_metrics = {**sei_bei_metrics, **nsei_nbei_metrics}
        return all_metrics

    except Exception as e:
        logger.error(f"Error calculating plane metrics: {str(e)}")
        return {
            'Modulus_SEI_BEI': np.nan,
            'Angle_SEI_BEI': np.nan,
            'Slope_SEI_BEI': np.nan,
            'Modulus_NSEI_NBEI': np.nan,
            'Angle_NSEI_NBEI': np.nan,
            'Slope_NSEI_NBEI': np.nan,
            'Intercept_NSEI_NBEI': np.nan
        }


def calculate_plane_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate efficiency plane metrics for an entire DataFrame.

    Requires columns: SEI, BEI, NSEI, NBEI, TPSA, Molecular_Weight, NPOL, Heavy_Atoms

    Args:
        df: DataFrame with efficiency metrics

    Returns:
        pd.DataFrame: Input DataFrame with added plane geometry columns

    Example:
        >>> df_with_planes = calculate_plane_metrics_dataframe(df)
        >>> print(df_with_planes[['Modulus_SEI_BEI', 'Angle_SEI_BEI']])
    """
    df = df.copy()

    required_columns = [
        'SEI', 'BEI', 'NSEI', 'NBEI',
        'TPSA', 'Molecular_Weight', 'NPOL', 'Heavy_Atoms'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Calculate plane metrics for each row
    plane_metrics_list = []
    for _, row in df.iterrows():
        metrics = calculate_all_plane_metrics(
            sei=float(row['SEI']),
            bei=float(row['BEI']),
            nsei=float(row['NSEI']),
            nbei=float(row['NBEI']),
            psa=float(row['TPSA']),
            molecular_weight=float(row['Molecular_Weight']),
            npol=float(row['NPOL']),
            heavy_atoms=float(row['Heavy_Atoms'])
        )
        plane_metrics_list.append(metrics)

    # Add metrics as new columns
    plane_metrics_df = pd.DataFrame(plane_metrics_list)
    df = pd.concat([df, plane_metrics_df], axis=1)

    return df


def interpret_angle(angle: float) -> Tuple[str, str]:
    """
    Interpret the meaning of an angle in the efficiency plane.

    Args:
        angle: Angle in degrees (0-90°)

    Returns:
        Tuple[str, str]: (category, interpretation)

    Example:
        >>> category, interpretation = interpret_angle(48.5)
        >>> print(category)  # "Excellent"
        >>> print(interpretation)  # "Balanced size and polarity"
    """
    if np.isnan(angle):
        return "Invalid", "No angle calculated"

    if 40 <= angle <= 50:
        return "Excellent", "Balanced size and polarity (near optimal 45°)"
    elif 30 <= angle < 40 or 50 < angle <= 60:
        return "Good", "Moderate balance"
    elif 20 <= angle < 30 or 60 < angle <= 70:
        return "Fair", "Somewhat unbalanced"
    elif angle < 20:
        return "Poor", "Too hydrophobic (low polarity)"
    elif angle > 70:
        return "Poor", "Too polar (excessive polarity)"
    else:
        return "Unknown", "Angle outside expected range"


def calculate_distance_between_points(
    sei1: float, bei1: float,
    sei2: float, bei2: float
) -> float:
    """
    Calculate Euclidean distance between two points in SEI-BEI plane.

    Useful for measuring similarity between compounds in efficiency space.

    Args:
        sei1, bei1: Coordinates of first point
        sei2, bei2: Coordinates of second point

    Returns:
        float: Euclidean distance
    """
    if not any(np.isnan([sei1, bei1, sei2, bei2])):
        return np.sqrt((sei2 - sei1)**2 + (bei2 - bei1)**2)
    return np.nan


def find_best_in_class(df: pd.DataFrame, metric_column: str = 'Modulus_SEI_BEI') -> Dict:
    """
    Find the best-in-class compound (highest modulus) in a cohort.

    Args:
        df: DataFrame with efficiency plane metrics
        metric_column: Column name for modulus (default: 'Modulus_SEI_BEI')

    Returns:
        Dict: Information about best compound

    Example:
        >>> best = find_best_in_class(df)
        >>> print(f"Best compound: {best['ChEMBL_ID']} with modulus {best['modulus']:.2f}")
    """
    if metric_column not in df.columns:
        raise ValueError(f"Column {metric_column} not found in DataFrame")

    # Filter out NaN values
    df_valid = df[df[metric_column].notna()].copy()

    if df_valid.empty:
        return {'ChEMBL_ID': None, 'modulus': np.nan, 'index': None}

    # Find row with maximum modulus
    best_idx = df_valid[metric_column].idxmax()
    best_row = df_valid.loc[best_idx]

    return {
        'ChEMBL_ID': best_row.get('ChEMBL_ID', 'Unknown'),
        'Molecule_Name': best_row.get('Molecule_Name', 'Unknown'),
        'modulus': best_row[metric_column],
        'SEI': best_row.get('SEI', np.nan),
        'BEI': best_row.get('BEI', np.nan),
        'index': best_idx
    }
