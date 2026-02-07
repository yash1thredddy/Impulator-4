"""
IMP Scoring Module - Multi-criteria scoring for IMP (Invalid Metabolic Panacea) detection.
Decoupled from Streamlit for backend use.

This module implements the IMP multi-criteria scoring system.

**Components** (weights sum to 100%, no normalization):
1. Efficiency Outlier Score - 45%
2. Distance to Best-in-Class Score - 20%
3. Development Angle Score - 15%
4. Assay Interference Score - 15%
5. PDB Structural Evidence Score - 5%

When PDB is disabled, PDB_Score = 0 (max possible = 95% before QED).
Final score includes QED multiplier for drug-likeness: 0.75 + 0.25 * QED

**Efficiency Score Calculation**:
Uses only SEI and BEI (not NSEI/NBEI) to avoid redundancy since normalized
metrics are derived from the same underlying data.

**Interference Score Calculation**:
Converts 5 binary assay interference flags into a 0-1 score: flags_triggered / 5.
Only PAINS, Aggregator, Thiol, Redox, and Fluorescence are counted.
BRENK and NIH remain display-only (not counted in score).
More flags = stronger evidence the compound is a genuine IMP (assay artifact).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Progress callback type
ProgressCallback = Callable[[float, str], None]

# =============================================================================
# IMP Score Weight Constants (Direct Percentages - Sum to 100%)
# =============================================================================

WEIGHT_EFFICIENCY = 0.45      # Efficiency Outlier Score
WEIGHT_DISTANCE = 0.20        # Distance to Best-in-Class Score
WEIGHT_ANGLE = 0.15           # Development Angle Score
WEIGHT_INTERFERENCE = 0.15    # Assay Interference Score
WEIGHT_PDB = 0.05             # PDB Structural Evidence Score

# QED Multiplier constants
QED_MULTIPLIER_FLOOR = 0.75   # Minimum multiplier when QED=0
QED_MULTIPLIER_SCALE = 0.25   # Additional multiplier per QED unit

# Interference flag columns COUNTED in score (BRENK and NIH are display-only)
INTERFERENCE_FLAG_COLUMNS = [
    'PAINS_Violation', 'Aggregator_Risk', 'Redox_Reactive',
    'Fluorescence_Interference', 'Thiol_Reactive',
]
INTERFERENCE_FLAG_COUNT = len(INTERFERENCE_FLAG_COLUMNS)  # 5

# Default efficiency metrics (use only SEI and BEI, not NSEI/NBEI)
DEFAULT_EFFICIENCY_METRICS = ['SEI', 'BEI']


def calculate_efficiency_outlier_score(
    df: pd.DataFrame,
    metrics: List[str] = None
) -> pd.Series:
    """
    Component 1: Efficiency Outlier Score (40% raw weight).

    Quantifies how exceptional the compound's efficiency metrics are compared
    to the cohort using Z-score normalization with sigmoid transformation.

    Uses sigmoid function instead of hard clipping to preserve ranking
    information for exceptional compounds (z > 3).

    Note: Default metrics are SEI and BEI only (not NSEI/NBEI) to avoid
    redundancy since normalized metrics are derived from the same data.

    Args:
        df: DataFrame with efficiency metrics (SEI, BEI, NSEI, NBEI)
        metrics: List of metrics to use (default: ['SEI', 'BEI'])

    Returns:
        pd.Series: Efficiency scores (0-1) for each compound
    """
    if metrics is None:
        metrics = DEFAULT_EFFICIENCY_METRICS  # ['SEI', 'BEI']

    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing efficiency metrics: {missing_metrics}")

    normalized_scores = []

    for metric in metrics:
        std_val = df[metric].std()
        if std_val == 0 or pd.isna(std_val):
            z_score = pd.Series(0.0, index=df.index)
        else:
            z_score = (df[metric] - df[metric].mean()) / std_val

        # Use sigmoid normalization to preserve ranking for exceptional compounds
        # Sigmoid maps z-scores to 0-1 while maintaining ordering
        # z=0 -> 0.5, z=3 -> ~0.95, z=-3 -> ~0.05
        sigmoid_score = 1 / (1 + np.exp(-z_score))

        # Shift and scale so that z=0 maps to 0 and positive z-scores map to (0, 1)
        # This preserves the original behavior for normal compounds while
        # maintaining ranking for outliers (z > 3 still gets progressively higher scores)
        normalized = (sigmoid_score - 0.5) * 2
        normalized = normalized.clip(0, 1)
        normalized_scores.append(normalized)

    efficiency_score = pd.concat(normalized_scores, axis=1).mean(axis=1)
    return efficiency_score


def calculate_angle_score(angles: pd.Series, optimal_angle: float = 45.0) -> pd.Series:
    """
    Component 2: Development Angle Score (15% raw weight).

    An angle of 45deg represents optimal balance between surface efficiency
    and binding efficiency.

    Args:
        angles: Series of angles (in degrees) from efficiency plane
        optimal_angle: Target angle (default: 45deg)

    Returns:
        pd.Series: Angle scores (0-1) for each compound
    """
    angle_deviation = (angles - optimal_angle).abs()
    score = 1 - (angle_deviation / optimal_angle)
    return score.clip(0, 1)


def calculate_distance_to_best_score(
    df: pd.DataFrame,
    modulus_column: str = 'Modulus_SEI_BEI'
) -> pd.Series:
    """
    Component 3: Distance to Best-in-Class Score (20% raw weight).

    Measures how close each compound is to the best-performing compound.

    Args:
        df: DataFrame with modulus values
        modulus_column: Name of modulus column

    Returns:
        pd.Series: Distance scores (0-1) for each compound
    """
    if modulus_column not in df.columns:
        raise ValueError(f"Modulus column '{modulus_column}' not found in DataFrame")

    best_modulus = df[modulus_column].max()

    if np.isnan(best_modulus) or best_modulus <= 0:
        logger.warning("Best modulus is NaN or zero. Returning all zeros.")
        return pd.Series([0.0] * len(df), index=df.index)

    # Normalize by best modulus and clip to ensure 0-1 range
    distance_score = (df[modulus_column] / best_modulus).clip(0, 1)
    return distance_score


def calculate_interference_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Component 4: Assay Interference Score (15% weight).

    Converts 5 binary assay interference flags into a single 0-1 score.
    Score = number of flags triggered / 5.

    More flags = higher score = stronger evidence of IMP (assay artifact).

    Only these 5 flags are counted (BRENK and NIH are display-only):
        PAINS_Violation, Aggregator_Risk, Redox_Reactive,
        Fluorescence_Interference, Thiol_Reactive

    Args:
        df: DataFrame (may or may not contain interference flag columns)

    Returns:
        DataFrame with added Interference_Score column (0.0-1.0)
    """
    df = df.copy()

    present_cols = [col for col in INTERFERENCE_FLAG_COLUMNS if col in df.columns]

    if not present_cols:
        logger.warning("No interference flag columns found in DataFrame. Setting Interference_Score = 0.")
        df['Interference_Score'] = 0.0
        return df

    df['Interference_Score'] = df[present_cols].fillna(0).astype(int).sum(axis=1) / INTERFERENCE_FLAG_COUNT
    df['Interference_Score'] = df['Interference_Score'].clip(0.0, 1.0)

    return df


def calculate_imp_score(
    df: pd.DataFrame,
    use_pdb: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> pd.DataFrame:
    """
    Calculate IMP score using all 5 components.

    Weights (sum to 100%, no normalization):
    - Efficiency: 45%
    - Distance: 20%
    - Angle: 15%
    - Interference: 15%
    - PDB: 5%

    QED Multiplier: 0.75 + 0.25 * QED

    When use_pdb=False, PDB_Score = 0 (max possible = 95% before QED).

    Args:
        df: DataFrame with efficiency metrics, plane geometry, SMILES, and
            optionally interference flag columns
        use_pdb: If True, query PDB for structural evidence
        progress_callback: Optional callback for PDB progress updates

    Returns:
        pd.DataFrame: Input DataFrame with added IMP score columns
    """
    df = df.copy()

    required_columns = ['SEI', 'BEI', 'NSEI', 'NBEI', 'Angle_SEI_BEI', 'Modulus_SEI_BEI', 'QED', 'SMILES']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Calculate component scores
    df['Efficiency_Score'] = calculate_efficiency_outlier_score(df)
    df['Angle_Score'] = calculate_angle_score(df['Angle_SEI_BEI'])
    df['Distance_Score'] = calculate_distance_to_best_score(df)
    df = calculate_interference_score(df)
    df = calculate_pdb_evidence_score(df, use_pdb=use_pdb, progress_callback=progress_callback)

    # Calculate base score (direct weights, no normalization)
    df['IMP_Base_Score'] = (
        WEIGHT_EFFICIENCY * df['Efficiency_Score'] +
        WEIGHT_DISTANCE * df['Distance_Score'] +
        WEIGHT_ANGLE * df['Angle_Score'] +
        WEIGHT_INTERFERENCE * df['Interference_Score'] +
        WEIGHT_PDB * df['PDB_Score']
    )

    # Apply QED multiplier
    df['QED_Multiplier'] = QED_MULTIPLIER_FLOOR + QED_MULTIPLIER_SCALE * df['QED']
    df['IMP_Final_Score'] = df['IMP_Base_Score'] * df['QED_Multiplier']

    # Calculate individual contributions (after QED)
    df['Efficiency_Contribution'] = WEIGHT_EFFICIENCY * df['Efficiency_Score'] * df['QED_Multiplier']
    df['Angle_Contribution'] = WEIGHT_ANGLE * df['Angle_Score'] * df['QED_Multiplier']
    df['Distance_Contribution'] = WEIGHT_DISTANCE * df['Distance_Score'] * df['QED_Multiplier']
    df['Interference_Contribution'] = WEIGHT_INTERFERENCE * df['Interference_Score'] * df['QED_Multiplier']
    df['PDB_Contribution'] = WEIGHT_PDB * df['PDB_Score'] * df['QED_Multiplier']

    df['QED_Impact'] = df['IMP_Final_Score'] - df['IMP_Base_Score']

    return df


def calculate_imp_score_phase1(
    df: pd.DataFrame,
    use_normalized_weights: bool = True
) -> pd.DataFrame:
    """
    Deprecated: Use calculate_imp_score(use_pdb=False) instead.

    Calculate IMP score using Phase 1 components (1-3) only.

    Phase 1 weights (raw -> normalized):
    - Efficiency: 45% -> 56.25%
    - Angle: 15% -> 18.75%
    - Distance: 20% -> 25%

    QED Multiplier: 0.75 + 0.25 * QED

    Args:
        df: DataFrame with efficiency metrics and plane geometry
        use_normalized_weights: If True, normalize Phase 1 weights to 100%

    Returns:
        pd.DataFrame: Input DataFrame with added IMP score columns
    """
    df = df.copy()

    required_columns = ['SEI', 'BEI', 'NSEI', 'NBEI', 'Angle_SEI_BEI', 'Modulus_SEI_BEI', 'QED']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df['Efficiency_Score'] = calculate_efficiency_outlier_score(df)
    df['Angle_Score'] = calculate_angle_score(df['Angle_SEI_BEI'])
    df['Distance_Score'] = calculate_distance_to_best_score(df)

    if use_normalized_weights:
        total_phase1_weight = WEIGHT_EFFICIENCY + WEIGHT_ANGLE + WEIGHT_DISTANCE
        w1 = WEIGHT_EFFICIENCY / total_phase1_weight
        w2 = WEIGHT_ANGLE / total_phase1_weight
        w3 = WEIGHT_DISTANCE / total_phase1_weight
    else:
        w1, w2, w3 = WEIGHT_EFFICIENCY, WEIGHT_ANGLE, WEIGHT_DISTANCE

    df['IMP_Base_Score'] = (
        w1 * df['Efficiency_Score'] +
        w2 * df['Angle_Score'] +
        w3 * df['Distance_Score']
    )

    df['QED_Multiplier'] = QED_MULTIPLIER_FLOOR + QED_MULTIPLIER_SCALE * df['QED']
    df['IMP_Final_Score'] = df['IMP_Base_Score'] * df['QED_Multiplier']

    return df


def calculate_pdb_evidence_score(
    df: pd.DataFrame,
    use_pdb: bool = False,
    progress_callback: Optional[ProgressCallback] = None
) -> pd.DataFrame:
    """
    Component 4: PDB Structural Evidence Score (5% raw weight).

    Query RCSB PDB for experimental structures of the compound or close analogs.

    Args:
        df: DataFrame with SMILES column
        use_pdb: If True, query PDB API; if False, return zeros
        progress_callback: Optional callback for progress updates

    Returns:
        DataFrame with added PDB columns
    """
    df = df.copy()

    if not use_pdb:
        logger.info("PDB Evidence Score disabled. Returning zeros.")
        df['PDB_Score'] = 0.0
        df['PDB_Num_Structures'] = 0
        df['PDB_High_Quality'] = 0
        df['PDB_Medium_Quality'] = 0
        df['PDB_Poor_Quality'] = 0
        df['PDB_IDs'] = ""
        df['PDB_Best_Resolution'] = np.nan
        return df

    # Import PDB client with clear error reporting
    get_pdb_evidence_score = None
    try:
        # Try relative import first, then absolute
        try:
            from .pdb_client import get_pdb_evidence_score
        except ImportError:
            from backend.modules.pdb_client import get_pdb_evidence_score
    except ImportError as e:
        logger.error(f"Failed to import pdb_client: {e}")
        logger.error("PDB evidence scoring will be disabled - returning zeros")

    if get_pdb_evidence_score is None:
        df['PDB_Score'] = 0.0
        df['PDB_Num_Structures'] = 0
        df['PDB_High_Quality'] = 0
        df['PDB_Medium_Quality'] = 0
        df['PDB_Poor_Quality'] = 0
        df['PDB_IDs'] = ""
        df['PDB_Best_Resolution'] = np.nan
        return df

    logger.info(f"Querying RCSB PDB for {len(df)} compounds...")

    unique_smiles = df['SMILES'].dropna().unique()

    if progress_callback:
        progress_callback(0.0, f"Querying PDB for {len(unique_smiles)} unique compound(s)...")

    # Parallel PDB queries using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    pdb_results = {}
    completed_count = [0]  # Use list for mutable counter in closure
    lock = threading.Lock()

    def fetch_pdb_for_smiles(smiles: str, max_retries: int = 2) -> tuple:
        """Fetch PDB evidence for a single SMILES string with retry logic.

        Handles transient PDB API failures (timeouts, connection errors).
        """
        import time
        smiles_preview = smiles[:50] + "..." if len(smiles) > 50 else smiles

        last_error = None
        for attempt in range(max_retries):
            try:
                result = get_pdb_evidence_score(smiles, similarity_threshold=0.9)
                return smiles, result
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Check for transient errors that are worth retrying
                is_transient = any(x in error_str for x in [
                    'timeout', 'connection', 'temporary', '503', '502', '504'
                ])

                if attempt < max_retries - 1:
                    if is_transient:
                        logger.warning(f"PDB query transient error for {smiles_preview} (attempt {attempt + 1}), retrying...")
                    else:
                        logger.warning(f"PDB query attempt {attempt + 1} failed for {smiles_preview}: {e}")
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"PDB query failed for {smiles_preview} after {max_retries} attempts: {last_error}")

        # Return empty result on failure
        return smiles, {
            'pdb_score': 0.0,
            'num_structures': 0,
            'num_high_quality': 0,
            'num_medium_quality': 0,
            'num_poor_quality': 0,
            'pdb_ids': [],
            'resolutions': []
        }

    # Use 5 parallel workers (balance between speed and API rate limits)
    max_workers = min(5, len(unique_smiles)) if len(unique_smiles) > 0 else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_pdb_for_smiles, smiles): smiles
            for smiles in unique_smiles
        }

        for future in as_completed(futures):
            smiles, result = future.result()
            pdb_results[smiles] = result

            with lock:
                completed_count[0] += 1
                if progress_callback:
                    progress = completed_count[0] / len(unique_smiles)
                    progress_callback(
                        progress,
                        f"Processed {completed_count[0]}/{len(unique_smiles)} compounds "
                        f"({result['num_structures']} structures found)"
                    )

    if progress_callback:
        progress_callback(1.0, "PDB query complete")

    df['PDB_Score'] = df['SMILES'].map(lambda s: pdb_results.get(s, {}).get('pdb_score', 0.0))
    df['PDB_Num_Structures'] = df['SMILES'].map(lambda s: pdb_results.get(s, {}).get('num_structures', 0))
    df['PDB_High_Quality'] = df['SMILES'].map(lambda s: pdb_results.get(s, {}).get('num_high_quality', 0))
    df['PDB_Medium_Quality'] = df['SMILES'].map(lambda s: pdb_results.get(s, {}).get('num_medium_quality', 0))
    df['PDB_Poor_Quality'] = df['SMILES'].map(lambda s: pdb_results.get(s, {}).get('num_poor_quality', 0))

    df['PDB_IDs'] = df['SMILES'].map(
        lambda s: ",".join(pdb_results.get(s, {}).get('pdb_ids', []))
    )

    df['PDB_Best_Resolution'] = df['SMILES'].map(
        lambda s: min([r for r in pdb_results.get(s, {}).get('resolutions', []) if r is not None], default=np.nan)
    )

    total_structures = sum([result['num_structures'] for result in pdb_results.values()])
    logger.info(f"PDB query complete. Found {total_structures} total structures across {len(unique_smiles)} unique compounds.")

    return df


def calculate_imp_score_phase2(
    df: pd.DataFrame,
    use_pdb: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> pd.DataFrame:
    """
    Deprecated: Use calculate_imp_score() instead.

    Calculate IMP score using Phase 2 components (1-4, no interference).

    Phase 2 weights (raw -> normalized):
    - Efficiency: 45% -> 52.94%
    - Angle: 15% -> 17.65%
    - Distance: 20% -> 23.53%
    - PDB: 5% -> 5.88%

    QED Multiplier: 0.75 + 0.25 * QED

    Args:
        df: DataFrame with efficiency metrics, plane geometry, and SMILES
        use_pdb: If True, query PDB for structural evidence
        progress_callback: Optional callback for progress updates

    Returns:
        pd.DataFrame: Input DataFrame with added IMP score columns
    """
    df = df.copy()

    required_columns = ['SEI', 'BEI', 'NSEI', 'NBEI', 'Angle_SEI_BEI', 'Modulus_SEI_BEI', 'QED', 'SMILES']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df['Efficiency_Score'] = calculate_efficiency_outlier_score(df)
    df['Angle_Score'] = calculate_angle_score(df['Angle_SEI_BEI'])
    df['Distance_Score'] = calculate_distance_to_best_score(df)

    df = calculate_pdb_evidence_score(df, use_pdb=use_pdb, progress_callback=progress_callback)

    total_phase2_weight = WEIGHT_EFFICIENCY + WEIGHT_ANGLE + WEIGHT_DISTANCE + WEIGHT_PDB
    w1 = WEIGHT_EFFICIENCY / total_phase2_weight
    w2 = WEIGHT_ANGLE / total_phase2_weight
    w3 = WEIGHT_DISTANCE / total_phase2_weight
    w4 = WEIGHT_PDB / total_phase2_weight

    df['IMP_Base_Score'] = (
        w1 * df['Efficiency_Score'] +
        w2 * df['Angle_Score'] +
        w3 * df['Distance_Score'] +
        w4 * df['PDB_Score']
    )

    df['QED_Multiplier'] = QED_MULTIPLIER_FLOOR + QED_MULTIPLIER_SCALE * df['QED']
    df['IMP_Final_Score'] = df['IMP_Base_Score'] * df['QED_Multiplier']

    df['Efficiency_Contribution'] = w1 * df['Efficiency_Score'] * df['QED_Multiplier']
    df['Angle_Contribution'] = w2 * df['Angle_Score'] * df['QED_Multiplier']
    df['Distance_Contribution'] = w3 * df['Distance_Score'] * df['QED_Multiplier']
    df['PDB_Contribution'] = w4 * df['PDB_Score'] * df['QED_Multiplier']

    df['QED_Impact'] = df['IMP_Final_Score'] - df['IMP_Base_Score']

    return df


def interpret_imp_score(score: float) -> Dict[str, str]:
    """
    Interpret IMP score and provide classification + recommendation.

    CRITICAL: IMP = Invalid Metabolic Panacea = FALSE POSITIVE indicator

    Higher IMP scores indicate HIGHER probability of being an assay artifact.
    Lower IMP scores indicate compounds more likely to be genuine leads.

    Score Interpretation (INVERSE relationship):
    - High Score (0.9+) = High false positive risk → EXCLUDE/DEPRIORITIZE
    - Low Score (<0.3) = Low false positive risk → PROCEED with confidence
    """
    if np.isnan(score):
        return {
            'classification': 'Invalid',
            'interpretation': 'No score calculated',
            'action': 'Check data quality',
            'priority': None
        }

    # CORRECTED INTERPRETATION: Higher score = Higher false positive risk
    if 0.9 <= score <= 1.0:
        return {
            'classification': 'Exceptional IMP',
            'interpretation': '⚠️ VERY HIGH false positive risk - likely assay artifact',
            'action': 'DEPRIORITIZE - Do not pursue unless validated with orthogonal assays',
            'priority': 1  # Priority 1 = Highest concern (to exclude)
        }
    elif 0.7 <= score < 0.9:
        return {
            'classification': 'Strong IMP',
            'interpretation': '⚠️ HIGH false positive risk - requires validation',
            'action': 'VALIDATE with orthogonal assays (SPR, ITC) before advancing',
            'priority': 2  # Priority 2 = High concern (validate before proceeding)
        }
    elif 0.5 <= score < 0.7:
        return {
            'classification': 'Moderate IMP',
            'interpretation': '⚠️ MODERATE false positive risk',
            'action': 'Monitor carefully - gather additional evidence before investing resources',
            'priority': 3  # Priority 3 = Moderate concern
        }
    elif 0.3 <= score < 0.5:
        return {
            'classification': 'Weak IMP',
            'interpretation': '✓ LOW false positive risk - more likely genuine',
            'action': 'PROCEED with standard due diligence',
            'priority': 4  # Priority 4 = Low concern (proceed normally)
        }
    else:
        return {
            'classification': 'Not IMP',
            'interpretation': '✓ LOWEST false positive risk - likely genuine activity',
            'action': 'PROCEED with confidence - prioritize for development',
            'priority': None  # No concern - best candidates
        }


def add_imp_score_interpretation(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable IMP score interpretation columns to DataFrame."""
    df = df.copy()

    if 'IMP_Final_Score' not in df.columns:
        raise ValueError("IMP_Final_Score column not found. Run calculate_imp_score() first.")

    interpretations = df['IMP_Final_Score'].apply(interpret_imp_score)

    df['IMP_Classification'] = interpretations.apply(lambda x: x['classification'])
    df['IMP_Priority'] = interpretations.apply(lambda x: x['priority'])

    return df


def get_imp_score_summary(df: pd.DataFrame) -> Dict:
    """Generate summary statistics about IMP scores in the dataset."""
    if 'IMP_Final_Score' not in df.columns:
        return {'error': 'No IMP scores found'}

    scores = df['IMP_Final_Score'].dropna()

    summary = {
        'total_compounds': len(df),
        'scored_compounds': len(scores),
        'mean_score': float(scores.mean()) if len(scores) > 0 else np.nan,
        'median_score': float(scores.median()) if len(scores) > 0 else np.nan,
        'std_score': float(scores.std()) if len(scores) > 0 else np.nan,
        'min_score': float(scores.min()) if len(scores) > 0 else np.nan,
        'max_score': float(scores.max()) if len(scores) > 0 else np.nan
    }

    if 'IMP_Classification' in df.columns:
        classification_counts = df['IMP_Classification'].value_counts().to_dict()
        summary['classification_counts'] = classification_counts

        summary['exceptional_imps'] = classification_counts.get('Exceptional IMP', 0)
        summary['strong_imps'] = classification_counts.get('Strong IMP', 0)
        summary['moderate_imps'] = classification_counts.get('Moderate IMP', 0)
        summary['weak_imps'] = classification_counts.get('Weak IMP', 0)
        summary['not_imps'] = classification_counts.get('Not IMP', 0)

    if 'IMP_Priority' in df.columns:
        priority_counts = df['IMP_Priority'].value_counts().sort_index().to_dict()
        summary['priority_counts'] = priority_counts

    return summary


def create_pdb_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create compound-level PDB summary from bioactivity dataframe."""
    compound_cols = ['ChEMBL_ID', 'Molecule_Name', 'SMILES']
    pdb_cols = [
        'PDB_Score', 'PDB_Num_Structures',
        'PDB_High_Quality', 'PDB_Medium_Quality', 'PDB_Poor_Quality',
        'PDB_IDs', 'PDB_Best_Resolution'
    ]

    if 'PDB_Score' not in df.columns:
        logger.warning("PDB columns not found in dataframe. Cannot create PDB summary.")
        return pd.DataFrame()

    summary_df = df[compound_cols + pdb_cols].drop_duplicates(subset=['SMILES']).copy()
    summary_df = summary_df.sort_values('PDB_Score', ascending=False).reset_index(drop=True)

    # Safe division - replace inf values from division by zero
    summary_df['PDB_High_Quality_Pct'] = (
        summary_df['PDB_High_Quality'] / summary_df['PDB_Num_Structures'].replace(0, float('nan')) * 100
    ).fillna(0).round(1)

    summary_df['PDB_Medium_Quality_Pct'] = (
        summary_df['PDB_Medium_Quality'] / summary_df['PDB_Num_Structures'].replace(0, float('nan')) * 100
    ).fillna(0).round(1)

    summary_df['PDB_Poor_Quality_Pct'] = (
        summary_df['PDB_Poor_Quality'] / summary_df['PDB_Num_Structures'].replace(0, float('nan')) * 100
    ).fillna(0).round(1)

    logger.info(f"Created PDB summary for {len(summary_df)} unique compounds.")

    return summary_df


# =============================================================================
# IMP Score Breakdown Helper
# =============================================================================

# List of all IMP Score output columns
IMP_SCORE_OUTPUT_COLUMNS = [
    # Raw efficiency metrics (all calculated, displayed)
    'SEI', 'BEI', 'NSEI', 'NBEI',

    # Plane geometry
    'Modulus_SEI_BEI', 'Angle_SEI_BEI',

    # Component scores (0-1)
    'Efficiency_Score', 'Angle_Score', 'Distance_Score', 'Interference_Score', 'PDB_Score',

    # Weighted contributions
    'Efficiency_Contribution', 'Angle_Contribution',
    'Distance_Contribution', 'Interference_Contribution', 'PDB_Contribution',

    # Final calculations
    'IMP_Base_Score', 'QED', 'QED_Multiplier', 'QED_Impact',
    'IMP_Final_Score', 'IMP_Classification', 'IMP_Priority',

    # PDB details (if available)
    'PDB_Num_Structures', 'PDB_High_Quality', 'PDB_Medium_Quality', 'PDB_Poor_Quality'
]


def _build_component_scores(row: pd.Series) -> dict:
    """Build component scores dict with direct weights (no normalization)."""
    result = {
        'efficiency': {
            'value': row.get('Efficiency_Score'),
            'weight': f'{WEIGHT_EFFICIENCY * 100:.0f}%',
            'contribution': row.get('Efficiency_Contribution'),
            'description': 'Outlier score based on SEI and BEI z-scores'
        },
        'distance': {
            'value': row.get('Distance_Score'),
            'weight': f'{WEIGHT_DISTANCE * 100:.0f}%',
            'contribution': row.get('Distance_Contribution'),
            'description': 'Proximity to best-in-class compound (highest modulus)'
        },
        'angle': {
            'value': row.get('Angle_Score'),
            'weight': f'{WEIGHT_ANGLE * 100:.0f}%',
            'contribution': row.get('Angle_Contribution'),
            'description': 'Proximity to optimal 45° development angle'
        },
        'interference': {
            'value': row.get('Interference_Score'),
            'weight': f'{WEIGHT_INTERFERENCE * 100:.0f}%',
            'contribution': row.get('Interference_Contribution'),
            'description': 'Assay interference flags (PAINS, Aggregator, Thiol, Redox, Fluorescence). BRENK/NIH display-only.'
        },
        'pdb': {
            'value': row.get('PDB_Score'),
            'weight': f'{WEIGHT_PDB * 100:.0f}%',
            'contribution': row.get('PDB_Contribution'),
            'description': 'Structural validation from RCSB PDB'
        },
    }
    return result


def get_imp_score_breakdown(row: pd.Series) -> dict:
    """
    Get a complete breakdown of IMP score components for a single compound.

    Returns dict with all individual scores and their interpretations,
    organized by category for easy frontend display.

    Args:
        row: A pandas Series containing IMP score columns

    Returns:
        dict with sections: efficiency_metrics, plane_geometry, component_scores,
                          final_calculation, pdb_details
    """
    return {
        'efficiency_metrics': {
            'SEI': {
                'value': row.get('SEI'),
                'description': 'Surface Efficiency Index (pActivity / PSA×100)',
                'used_in_score': True
            },
            'BEI': {
                'value': row.get('BEI'),
                'description': 'Binding Efficiency Index (pActivity / MW×1000)',
                'used_in_score': True
            },
            'NSEI': {
                'value': row.get('NSEI'),
                'description': 'Normalized SEI (pActivity / NPOL) - display only',
                'used_in_score': False
            },
            'NBEI': {
                'value': row.get('NBEI'),
                'description': 'Normalized BEI (pActivity / NHA) - display only',
                'used_in_score': False
            },
        },
        'plane_geometry': {
            'modulus': {
                'value': row.get('Modulus_SEI_BEI'),
                'description': (
                    'The modulus measures the distance of the combined efficiency vector '
                    '(SEI, BEI) from the origin on the efficiency plane. It represents the '
                    'overall efficiency magnitude of a compound. While derived from SEI and BEI, '
                    'the modulus is independent of the development angle—the angle only defines '
                    "the vector's direction, not its magnitude."
                )
            },
            'angle': {
                'value': row.get('Angle_SEI_BEI'),
                'optimal': 45.0,
                'description': 'Development trajectory angle. 45° is optimal (balanced). <30° = too hydrophobic, >60° = too polar.'
            },
        },
        'component_scores': _build_component_scores(row),
        'final_calculation': {
            'base_score': row.get('IMP_Base_Score'),
            'qed': row.get('QED'),
            'qed_multiplier': row.get('QED_Multiplier'),
            'qed_formula': '0.75 + 0.25 × QED',
            'qed_impact': row.get('QED_Impact'),
            'final_score': row.get('IMP_Final_Score'),
            'classification': row.get('IMP_Classification'),
            'priority': row.get('IMP_Priority'),
        },
        'pdb_details': {
            'num_structures': row.get('PDB_Num_Structures', 0),
            'high_quality': row.get('PDB_High_Quality', 0),
            'medium_quality': row.get('PDB_Medium_Quality', 0),
            'poor_quality': row.get('PDB_Poor_Quality', 0),
            'pdb_ids': row.get('PDB_IDs', ''),
            'best_resolution': row.get('PDB_Best_Resolution'),
        }
    }


def create_detailed_pdb_summary(df: pd.DataFrame, progress_callback: Optional[ProgressCallback] = None) -> pd.DataFrame:
    """
    Create detailed PDB summary with Title, Resolution, Quality, Experimental Method, UniProt IDs.

    This fetches additional details from RCSB PDB API for each unique PDB ID found in the data.

    Args:
        df: DataFrame with PDB_IDs column (comma-separated PDB IDs per compound)
        progress_callback: Optional callback for progress updates

    Returns:
        DataFrame with columns: PDB_ID, ChEMBL_ID, Molecule_Name, Title, Resolution,
                               Quality, Experimental_Method, UniProt_IDs
    """
    if 'PDB_IDs' not in df.columns:
        logger.warning("PDB_IDs column not found in dataframe. Cannot create detailed PDB summary.")
        return pd.DataFrame()

    try:
        # Try relative import first, then absolute
        try:
            from .pdb_client import get_structure_details, classify_resolution_quality
        except ImportError:
            from backend.modules.pdb_client import get_structure_details, classify_resolution_quality
    except ImportError:
        logger.error("PDB client module not found. Cannot create detailed PDB summary.")
        return pd.DataFrame()

    # Collect all unique PDB IDs with their associated compounds
    pdb_compound_map = {}  # PDB_ID -> list of (ChEMBL_ID, Molecule_Name)

    for _, row in df.iterrows():
        pdb_str = row.get('PDB_IDs', '')
        chembl_id = row.get('ChEMBL_ID', '')
        mol_name = row.get('Molecule_Name', '')

        if pd.isna(pdb_str) or not pdb_str:
            continue

        pdb_list = [p.strip().upper() for p in str(pdb_str).split(',') if p.strip()]
        for pdb_id in pdb_list:
            if pdb_id not in pdb_compound_map:
                pdb_compound_map[pdb_id] = []
            pdb_compound_map[pdb_id].append((chembl_id, mol_name if pd.notna(mol_name) else ''))

    unique_pdb_ids = list(pdb_compound_map.keys())

    if not unique_pdb_ids:
        logger.info("No PDB IDs found in data.")
        return pd.DataFrame()

    logger.info(f"Fetching detailed information for {len(unique_pdb_ids)} unique PDB structures...")

    if progress_callback:
        progress_callback(0.0, f"Fetching details for {len(unique_pdb_ids)} PDB structures...")

    detailed_data = []

    for i, pdb_id in enumerate(unique_pdb_ids):
        try:
            # Fetch PDB details from API
            pdb_info = get_structure_details(pdb_id)

            # Get resolution and quality
            resolution = pdb_info.get('resolution')
            if resolution is not None:
                quality, _ = classify_resolution_quality(resolution)
                resolution_str = f"{resolution:.2f}"
            else:
                quality = 'N/A'
                resolution_str = 'N/A'

            # Get associated compounds
            compounds = pdb_compound_map.get(pdb_id, [])
            chembl_ids = list(set([c[0] for c in compounds if c[0]]))
            mol_names = list(set([c[1] for c in compounds if c[1]]))

            # Get UniProt IDs from API
            api_uniprots = pdb_info.get('uniprot_ids', [])

            detailed_data.append({
                'PDB_ID': pdb_id,
                'ChEMBL_ID': ', '.join(chembl_ids) if chembl_ids else 'N/A',
                'Molecule_Name': ', '.join(mol_names) if mol_names else 'N/A',
                'Title': pdb_info.get('title') or 'N/A',
                'Resolution': resolution_str,
                'Quality': quality,
                'Experimental_Method': pdb_info.get('experimental_method') or 'N/A',
                'UniProt_IDs': ', '.join(api_uniprots) if api_uniprots else 'N/A'
            })

        except Exception as e:
            logger.warning(f"Error fetching details for {pdb_id}: {e}")
            compounds = pdb_compound_map.get(pdb_id, [])
            chembl_ids = list(set([c[0] for c in compounds if c[0]]))
            mol_names = list(set([c[1] for c in compounds if c[1]]))
            detailed_data.append({
                'PDB_ID': pdb_id,
                'ChEMBL_ID': ', '.join(chembl_ids) if chembl_ids else 'N/A',
                'Molecule_Name': ', '.join(mol_names) if mol_names else 'N/A',
                'Title': 'N/A',
                'Resolution': 'N/A',
                'Quality': 'N/A',
                'Experimental_Method': 'N/A',
                'UniProt_IDs': 'N/A'
            })

        if progress_callback and i % 10 == 0:
            progress = (i + 1) / len(unique_pdb_ids)
            progress_callback(progress, f"Processed {i + 1}/{len(unique_pdb_ids)} PDB structures")

    if progress_callback:
        progress_callback(1.0, "PDB detail fetch complete")

    # Create DataFrame and sort by quality then resolution
    pdb_df = pd.DataFrame(detailed_data)

    if not pdb_df.empty:
        # Sort by quality (*** first) then by resolution (lowest first)
        quality_order = {'***': 1, '**': 2, '*': 3, 'N/A': 4}
        pdb_df['Quality_Sort'] = pdb_df['Quality'].map(lambda x: quality_order.get(x, 4))
        pdb_df['Resolution_Sort'] = pdb_df['Resolution'].apply(
            lambda x: float(x) if x != 'N/A' else 999.0
        )
        pdb_df = pdb_df.sort_values(['Quality_Sort', 'Resolution_Sort']).drop(
            columns=['Quality_Sort', 'Resolution_Sort']
        ).reset_index(drop=True)

    logger.info(f"Created detailed PDB summary for {len(pdb_df)} structures.")

    return pdb_df
