"""
Ligand Efficiency Metrics Calculation Module

This module implements the efficiency metric calculations according to IMPs 2.0 (Reddy et al., Table 1).

Metrics calculated:
- SEI: Surface Efficiency Index (Equation 2.3)
- BEI: Binding Efficiency Index (Equation 2.2)
- NSEI: Normalized Surface Efficiency Index (Equation 2.4)
- NBEI: Normalized Binding Efficiency Index (Equation 2.6)
- nBEI_viz: Visualization-only metric (pActivity + log(NHA))
"""

import numpy as np
import pandas as pd
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def calculate_sei(pActivity: float, psa: float) -> float:
    """
    Calculate Surface Efficiency Index (SEI).

    Equation 2.3: SEI = pActivity / (PSA / 100)

    Args:
        pActivity: pActivity value (-log10 of activity in M)
        psa: Polar surface area (Ų)

    Returns:
        float: SEI value or NaN if invalid input
    """
    if psa and not np.isnan(pActivity) and psa > 0:
        return pActivity / (psa / 100)
    return np.nan


def calculate_bei(pActivity: float, molecular_weight: float) -> float:
    """
    Calculate Binding Efficiency Index (BEI).

    Equation 2.2: BEI = pActivity / (MW / 1000)

    Args:
        pActivity: pActivity value (-log10 of activity in M)
        molecular_weight: Molecular weight (Da)

    Returns:
        float: BEI value or NaN if invalid input
    """
    if molecular_weight and not np.isnan(pActivity) and molecular_weight > 0:
        return pActivity / (molecular_weight / 1000)
    return np.nan


def calculate_nsei(pActivity: float, npol: float) -> float:
    """
    Calculate Normalized Surface Efficiency Index (NSEI).

    Equation 2.4: NSEI = pActivity / NPOL
    where NPOL = count of (N + O atoms)

    Note: This is NOT "Normalized BEI" but rather an atom-related polarity metric.

    Args:
        pActivity: pActivity value (-log10 of activity in M)
        npol: NPOL value (N + O atom count)

    Returns:
        float: NSEI value or NaN if invalid input
    """
    if npol and not np.isnan(pActivity) and npol > 0:
        return pActivity / npol
    return np.nan


def calculate_nbei(pActivity: float, heavy_atoms: float) -> float:
    """
    Calculate Normalized Binding Efficiency Index (NBEI).

    Equation 2.6: NBEI = pActivity / NHA
    where NHA = Number of Heavy Atoms (non-hydrogen atoms)

    This is the PRIMARY normalized efficiency metric for calculations.

    Args:
        pActivity: pActivity value (-log10 of activity in M)
        heavy_atoms: Number of heavy atoms (NHA)

    Returns:
        float: NBEI value or NaN if invalid input
    """
    if heavy_atoms and not np.isnan(pActivity) and heavy_atoms > 0:
        return pActivity / heavy_atoms
    return np.nan


def calculate_nbei_visualization(pActivity: float, heavy_atoms: float) -> float:
    """
    Calculate nBEI for VISUALIZATION purposes only.

    Formula: nBEI_viz = pActivity + log(NHA)

    This is NOT used in any calculations or scoring - only for plotting.
    It helps in visualizing the NSEI-NBEI efficiency plane.

    Args:
        pActivity: pActivity value (-log10 of activity in M)
        heavy_atoms: Number of heavy atoms (NHA)

    Returns:
        float: nBEI_viz value or NaN if invalid input
    """
    if heavy_atoms and not np.isnan(pActivity) and heavy_atoms > 0:
        return pActivity + np.log10(heavy_atoms)
    return np.nan


def calculate_all_efficiency_metrics(
    pActivity: float,
    psa: float,
    molecular_weight: float,
    npol: float,
    heavy_atoms: float
) -> Dict[str, float]:
    """
    Calculate all ligand efficiency metrics for a single compound/bioactivity.

    This is the main function that calculates all efficiency indices according
    to IMPs 2.0 (Reddy et al., Table 1).

    Args:
        pActivity: pActivity value (-log10 of activity in M)
        psa: Polar surface area (Ų)
        molecular_weight: Molecular weight (Da)
        npol: NPOL value (N + O atom count)
        heavy_atoms: Number of heavy atoms (NHA)

    Returns:
        Dict[str, float]: Dictionary containing all efficiency metrics
            - SEI: Surface Efficiency Index
            - BEI: Binding Efficiency Index
            - NSEI: Normalized Surface Efficiency Index
            - NBEI: Normalized Binding Efficiency Index (for calculations)
            - nBEI_viz: Visualization-only metric (for plots)

    Example:
        >>> metrics = calculate_all_efficiency_metrics(
        ...     pActivity=7.5,
        ...     psa=85.2,
        ...     molecular_weight=342.1,
        ...     npol=5,
        ...     heavy_atoms=24
        ... )
        >>> print(metrics['SEI'])  # 8.80
        >>> print(metrics['NBEI'])  # 0.3125
    """
    try:
        metrics = {
            'SEI': calculate_sei(pActivity, psa),
            'BEI': calculate_bei(pActivity, molecular_weight),
            'NSEI': calculate_nsei(pActivity, npol),
            'NBEI': calculate_nbei(pActivity, heavy_atoms),
            'nBEI_viz': calculate_nbei_visualization(pActivity, heavy_atoms)
        }
        return metrics

    except Exception as e:
        logger.error(f"Error calculating efficiency metrics: {str(e)}")
        return {
            'SEI': np.nan,
            'BEI': np.nan,
            'NSEI': np.nan,
            'NBEI': np.nan,
            'nBEI_viz': np.nan
        }


def calculate_efficiency_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate efficiency metrics for an entire DataFrame of bioactivities.

    This function adds efficiency metric columns to the input DataFrame
    using vectorized operations (50-100x faster than iterrows approach).

    Requires the following columns: pActivity, TPSA, Molecular_Weight, NPOL, Heavy_Atoms

    Args:
        df: DataFrame with bioactivity data

    Returns:
        pd.DataFrame: Input DataFrame with added efficiency metric columns

    Example:
        >>> df = pd.DataFrame({
        ...     'pActivity': [7.5, 8.2, 6.8],
        ...     'TPSA': [85.2, 92.1, 78.5],
        ...     'Molecular_Weight': [342.1, 385.7, 298.3],
        ...     'NPOL': [5, 6, 4],
        ...     'Heavy_Atoms': [24, 27, 21]
        ... })
        >>> df_with_metrics = calculate_efficiency_metrics_dataframe(df)
        >>> print(df_with_metrics[['SEI', 'BEI', 'NSEI', 'NBEI']])
    """
    df = df.copy()

    required_columns = ['pActivity', 'TPSA', 'Molecular_Weight', 'NPOL', 'Heavy_Atoms']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Vectorized SEI: pActivity / (PSA / 100)
    # Only calculate where TPSA > 0 and pActivity is valid
    mask_sei = (df['TPSA'].notna()) & (df['TPSA'] > 0) & (df['pActivity'].notna())
    df['SEI'] = np.nan
    df.loc[mask_sei, 'SEI'] = df.loc[mask_sei, 'pActivity'] / (df.loc[mask_sei, 'TPSA'] / 100)

    # Vectorized BEI: pActivity / (MW / 1000)
    mask_bei = (df['Molecular_Weight'].notna()) & (df['Molecular_Weight'] > 0) & (df['pActivity'].notna())
    df['BEI'] = np.nan
    df.loc[mask_bei, 'BEI'] = df.loc[mask_bei, 'pActivity'] / (df.loc[mask_bei, 'Molecular_Weight'] / 1000)

    # Vectorized NSEI: pActivity / NPOL
    mask_nsei = (df['NPOL'].notna()) & (df['NPOL'] > 0) & (df['pActivity'].notna())
    df['NSEI'] = np.nan
    df.loc[mask_nsei, 'NSEI'] = df.loc[mask_nsei, 'pActivity'] / df.loc[mask_nsei, 'NPOL']

    # Vectorized NBEI: pActivity / NHA (Heavy_Atoms)
    mask_nbei = (df['Heavy_Atoms'].notna()) & (df['Heavy_Atoms'] > 0) & (df['pActivity'].notna())
    df['NBEI'] = np.nan
    df.loc[mask_nbei, 'NBEI'] = df.loc[mask_nbei, 'pActivity'] / df.loc[mask_nbei, 'Heavy_Atoms']

    # Vectorized nBEI_viz: pActivity + log10(NHA) (for visualization only)
    df['nBEI_viz'] = np.nan
    df.loc[mask_nbei, 'nBEI_viz'] = df.loc[mask_nbei, 'pActivity'] + np.log10(df.loc[mask_nbei, 'Heavy_Atoms'])

    return df


def validate_efficiency_metrics(metrics: Dict[str, float]) -> bool:
    """
    Validate that efficiency metrics are within expected ranges.

    Expected ranges (approximate):
    - SEI: 0-100 (typical: 5-30)
    - BEI: 0-100 (typical: 10-40)
    - NSEI: 0-10 (typical: 1-3)
    - NBEI: 0-1 (typical: 0.2-0.5)

    Args:
        metrics: Dictionary of efficiency metrics

    Returns:
        bool: True if all metrics are valid, False otherwise
    """
    ranges = {
        'SEI': (0, 100),
        'BEI': (0, 100),
        'NSEI': (0, 10),
        'NBEI': (0, 1)
    }

    for metric_name, (min_val, max_val) in ranges.items():
        if metric_name in metrics:
            value = metrics[metric_name]
            if not np.isnan(value):
                if value < min_val or value > max_val:
                    logger.warning(
                        f"{metric_name} = {value:.2f} is outside expected range "
                        f"[{min_val}, {max_val}]"
                    )
                    return False

    return True


# Backward compatibility: keep old function signature
def calculate_efficiency_metrics(
    pActivity: float,
    psa: float,
    molecular_weight: float,
    npol: float,
    heavy_atoms: float
):
    """
    Legacy function for backward compatibility.
    Returns tuple instead of dictionary.

    DEPRECATED: Use calculate_all_efficiency_metrics() instead.

    Returns:
        Tuple[float, float, float, float]: SEI, BEI, NSEI, NBEI
    """
    metrics = calculate_all_efficiency_metrics(
        pActivity, psa, molecular_weight, npol, heavy_atoms
    )
    return metrics['SEI'], metrics['BEI'], metrics['NSEI'], metrics['NBEI']
