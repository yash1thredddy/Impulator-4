"""
IMP Classification Module

This module implements the logic for classifying compounds as IMP vs Non-IMP candidates.

Classification criteria:
- IMP Candidate: Outlier_Count >= 2 (at least 2 efficiency metrics are outliers)
- Confidence Level: Based on IMP Score
  - High: IMP_Score > 0.7
  - Medium: IMP_Score > 0.5
  - Low: IMP_Score <= 0.5
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def classify_imp_candidates(
    df: pd.DataFrame,
    min_outlier_count: int = 2,
    use_imp_score: bool = True
) -> pd.DataFrame:
    """
    Classify compounds as IMP vs Non-IMP candidates.

    Classification logic:
    1. Check outlier count (Outlier_Count >= min_outlier_count)
    2. If IMP candidate, assign confidence level based on IMP score

    Args:
        df: DataFrame with outlier flags and IMP scores
        min_outlier_count: Minimum outliers required for IMP candidate (default: 2)
        use_imp_score: Use IMP score for confidence levels (default: True)

    Returns:
        pd.DataFrame: Input DataFrame with added columns:
            - Is_IMP_Candidate: Boolean (True if IMP candidate)
            - IMP_Confidence: "High", "Medium", "Low", or "Not IMP"

    Example:
        >>> df_classified = classify_imp_candidates(df)
        >>> high_confidence_imps = df_classified[
        ...     (df_classified['Is_IMP_Candidate']) &
        ...     (df_classified['IMP_Confidence'] == 'High')
        ... ]
    """
    df = df.copy()

    # Validate required columns
    if 'Outlier_Count' not in df.columns:
        raise ValueError("Outlier_Count column not found. Run detect_efficiency_outliers() first.")

    # Classify as IMP candidate based on outlier count
    df['Is_IMP_Candidate'] = df['Outlier_Count'] >= min_outlier_count

    # Assign confidence levels
    if use_imp_score and 'IMP_Final_Score' in df.columns:
        # Use IMP score for confidence
        df['IMP_Confidence'] = df.apply(
            lambda row: _assign_confidence_imp_score(
                row['Is_IMP_Candidate'],
                row['IMP_Final_Score']
            ),
            axis=1
        )
    else:
        # Simple confidence based on outlier count only
        df['IMP_Confidence'] = df.apply(
            lambda row: _assign_confidence_simple(
                row['Is_IMP_Candidate'],
                row['Outlier_Count']
            ),
            axis=1
        )

    return df


def _assign_confidence_imp_score(is_imp: bool, imp_final_score: float) -> str:
    """
    Assign confidence level based on IMP score.

    Args:
        is_imp: Whether compound is IMP candidate
        imp_final_score: IMP final score

    Returns:
        str: Confidence level
    """
    if not is_imp:
        return 'Not IMP'

    if np.isnan(imp_final_score):
        return 'Unknown'

    if imp_final_score > 0.7:
        return 'High'
    elif imp_final_score > 0.5:
        return 'Medium'
    else:
        return 'Low'


def _assign_confidence_simple(is_imp: bool, outlier_count: int) -> str:
    """
    Assign confidence level based on outlier count only.

    Simple fallback if IMP scores not available.

    Args:
        is_imp: Whether compound is IMP candidate
        outlier_count: Number of outlier flags

    Returns:
        str: Confidence level
    """
    if not is_imp:
        return 'Not IMP'

    if outlier_count >= 4:
        return 'High'
    elif outlier_count >= 3:
        return 'Medium'
    else:
        return 'Low'


def get_imp_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics about IMP classification.

    Args:
        df: DataFrame with IMP classification results

    Returns:
        Dict: Summary information

    Example:
        >>> summary = get_imp_summary(df)
        >>> print(f"Total IMPs: {summary['total_imp_candidates']}")
        >>> print(f"High confidence: {summary['high_confidence']}")
    """
    if 'Is_IMP_Candidate' not in df.columns:
        return {'error': 'No IMP classification found'}

    total_imps = df['Is_IMP_Candidate'].sum()
    total_compounds = len(df)

    summary = {
        'total_compounds': total_compounds,
        'total_imp_candidates': int(total_imps),
        'imp_percentage': (total_imps / total_compounds * 100) if total_compounds > 0 else 0,
        'total_non_imps': total_compounds - int(total_imps)
    }

    # Count by confidence level
    if 'IMP_Confidence' in df.columns:
        confidence_counts = df['IMP_Confidence'].value_counts().to_dict()
        summary['confidence_distribution'] = confidence_counts

        summary['high_confidence'] = confidence_counts.get('High', 0)
        summary['medium_confidence'] = confidence_counts.get('Medium', 0)
        summary['low_confidence'] = confidence_counts.get('Low', 0)

    return summary


def filter_imp_candidates(
    df: pd.DataFrame,
    min_confidence: str = None,
    min_imp_final_score: float = None
) -> pd.DataFrame:
    """
    Filter DataFrame to return only IMP candidates meeting specified criteria.

    Args:
        df: DataFrame with IMP classification
        min_confidence: Minimum confidence level ("Low", "Medium", "High")
        min_imp_final_score: Minimum IMP score (0-1)

    Returns:
        pd.DataFrame: Filtered DataFrame

    Example:
        >>> high_priority_imps = filter_imp_candidates(
        ...     df,
        ...     min_confidence="High",
        ...     min_imp_final_score=0.7
        ... )
    """
    if 'Is_IMP_Candidate' not in df.columns:
        raise ValueError("IMP classification not found. Run classify_imp_candidates() first.")

    # Start with all IMP candidates
    filtered_df = df[df['Is_IMP_Candidate']].copy()

    # Filter by confidence level
    if min_confidence:
        confidence_order = {'Low': 1, 'Medium': 2, 'High': 3}

        if min_confidence not in confidence_order:
            raise ValueError(f"Invalid confidence level: {min_confidence}")

        min_level = confidence_order[min_confidence]

        if 'IMP_Confidence' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['IMP_Confidence'].map(confidence_order).fillna(0) >= min_level
            ]

    # Filter by IMP score
    if min_imp_final_score is not None:
        if 'IMP_Final_Score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['IMP_Final_Score'] >= min_imp_final_score]

    return filtered_df


def rank_imp_candidates(
    df: pd.DataFrame,
    rank_by: str = 'IMP_Final_Score',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Rank IMP candidates by specified metric.

    Args:
        df: DataFrame with IMP candidates
        rank_by: Column to rank by (default: 'IMP_Final_Score')
        ascending: Sort order (default: False = highest first)

    Returns:
        pd.DataFrame: Sorted DataFrame with added Rank column

    Example:
        >>> ranked_imps = rank_imp_candidates(df, rank_by='IMP_Final_Score')
        >>> top_10 = ranked_imps.head(10)
    """
    if rank_by not in df.columns:
        raise ValueError(f"Column '{rank_by}' not found in DataFrame")

    df = df.copy()

    # Sort by specified column
    df = df.sort_values(by=rank_by, ascending=ascending)

    # Add rank column
    df['Rank'] = range(1, len(df) + 1)

    return df


def compare_imp_vs_non_imp(df: pd.DataFrame, metrics: List[str] = None) -> pd.DataFrame:
    """
    Compare efficiency metrics between IMP and Non-IMP groups.

    Args:
        df: DataFrame with IMP classification and efficiency metrics
        metrics: List of metrics to compare (default: efficiency metrics + IMP)

    Returns:
        pd.DataFrame: Comparison table with mean/median/std for each group

    Example:
        >>> comparison = compare_imp_vs_non_imp(df)
        >>> print(comparison)
    """
    if 'Is_IMP_Candidate' not in df.columns:
        raise ValueError("IMP classification not found")

    if metrics is None:
        metrics = [
            'SEI', 'BEI', 'NSEI', 'NBEI',
            'Modulus_SEI_BEI', 'Angle_SEI_BEI',
            'IMP_Final_Score', 'QED'
        ]

    # Filter available metrics
    metrics = [m for m in metrics if m in df.columns]

    # Split into groups
    imps = df[df['Is_IMP_Candidate']]
    non_imps = df[~df['Is_IMP_Candidate']]

    # Calculate statistics for each group
    comparison_data = []

    for metric in metrics:
        imp_data = imps[metric].dropna()
        non_imp_data = non_imps[metric].dropna()

        comparison_data.append({
            'Metric': metric,
            'IMP_Mean': imp_data.mean() if len(imp_data) > 0 else np.nan,
            'IMP_Median': imp_data.median() if len(imp_data) > 0 else np.nan,
            'IMP_Std': imp_data.std() if len(imp_data) > 0 else np.nan,
            'IMP_Count': len(imp_data),
            'NonIMP_Mean': non_imp_data.mean() if len(non_imp_data) > 0 else np.nan,
            'NonIMP_Median': non_imp_data.median() if len(non_imp_data) > 0 else np.nan,
            'NonIMP_Std': non_imp_data.std() if len(non_imp_data) > 0 else np.nan,
            'NonIMP_Count': len(non_imp_data),
            'Difference': (imp_data.mean() - non_imp_data.mean()) if len(imp_data) > 0 and len(non_imp_data) > 0 else np.nan
        })

    comparison_df = pd.DataFrame(comparison_data)

    return comparison_df


def generate_imp_report(df: pd.DataFrame, compound_name: str = "Query Compound") -> str:
    """
    Generate a text report summarizing IMP analysis results.

    Args:
        df: DataFrame with complete IMP analysis
        compound_name: Name of query compound

    Returns:
        str: Formatted report text

    Example:
        >>> report = generate_imp_report(df, "Quercetin")
        >>> print(report)
    """
    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append(f"IMP ANALYSIS REPORT: {compound_name}")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Overall summary
    imp_summary = get_imp_summary(df)

    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"Total compounds analyzed: {imp_summary['total_compounds']}")
    report_lines.append(f"IMP candidates identified: {imp_summary['total_imp_candidates']} ({imp_summary['imp_percentage']:.1f}%)")
    report_lines.append(f"Non-IMP compounds: {imp_summary['total_non_imps']}")
    report_lines.append("")

    # Confidence breakdown
    if 'confidence_distribution' in imp_summary:
        report_lines.append("CONFIDENCE DISTRIBUTION")
        report_lines.append("-" * 70)
        report_lines.append(f"High confidence: {imp_summary.get('high_confidence', 0)}")
        report_lines.append(f"Medium confidence: {imp_summary.get('medium_confidence', 0)}")
        report_lines.append(f"Low confidence: {imp_summary.get('low_confidence', 0)}")
        report_lines.append("")

    # Top candidates
    if 'IMP_Final_Score' in df.columns:
        top_imps = df[df['Is_IMP_Candidate']].nlargest(5, 'IMP_Final_Score')

        if len(top_imps) > 0:
            report_lines.append("TOP 5 IMP CANDIDATES (by IMP score)")
            report_lines.append("-" * 70)

            for idx, row in top_imps.iterrows():
                chembl_id = row.get('ChEMBL_ID', 'Unknown')
                molecule_name = row.get('Molecule_Name', 'Unknown')
                score = row['IMP_Final_Score']
                confidence = row.get('IMP_Confidence', 'Unknown')
                outlier_count = row.get('Outlier_Count', 0)

                report_lines.append(
                    f"{chembl_id} ({molecule_name}): "
                    f"IMP Score = {score:.3f}, "
                    f"Confidence = {confidence}, "
                    f"Outliers = {outlier_count}"
                )

            report_lines.append("")

    report_lines.append("=" * 70)

    return "\n".join(report_lines)
