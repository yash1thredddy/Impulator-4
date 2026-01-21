"""
O[Q/P/L]A Scoring Module - Overall Quality/Promise/Likelihood Assessment
Decoupled from Streamlit for backend use.

This module implements the O[Q/P/L]A multi-criteria scoring system for IMPs 2.0.

**Phase 1 Components**:
1. Efficiency Outlier Score (40%)
2. Development Angle Score (10%)
3. Distance to Best-in-Class Score (15%)

**Phase 2 Components (WITH PDB INTEGRATION)**:
4. PDB Structural Evidence Score (15%)
5. Target Prediction Confidence Score (10%) - DEFERRED
6. Analog Support Score (10%) - FUTURE

When PDB is enabled, weights are renormalized.
Final score includes QED multiplier for drug-likeness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Progress callback type
ProgressCallback = Callable[[float, str], None]


def calculate_efficiency_outlier_score(
    df: pd.DataFrame,
    metrics: List[str] = None
) -> pd.Series:
    """
    Component 1: Efficiency Outlier Score (40% weight).

    Quantifies how exceptional the compound's efficiency metrics are compared
    to the cohort using Z-score normalization with sigmoid transformation.

    Uses sigmoid function instead of hard clipping to preserve ranking
    information for exceptional compounds (z > 3).

    Args:
        df: DataFrame with efficiency metrics (SEI, BEI, NSEI, NBEI)
        metrics: List of metrics to use (default: ['SEI', 'BEI', 'NSEI', 'NBEI'])

    Returns:
        pd.Series: Efficiency scores (0-1) for each compound
    """
    if metrics is None:
        metrics = ['SEI', 'BEI', 'NSEI', 'NBEI']

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
    Component 2: Development Angle Score (10% weight).

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
    Component 3: Distance to Best-in-Class Score (15% weight).

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


def calculate_oqpla_phase1(
    df: pd.DataFrame,
    use_normalized_weights: bool = True
) -> pd.DataFrame:
    """
    Calculate O[Q/P/L]A score using Phase 1 components (1-3) only.

    Args:
        df: DataFrame with efficiency metrics and plane geometry
        use_normalized_weights: If True, normalize Phase 1 weights to 100%

    Returns:
        pd.DataFrame: Input DataFrame with added O[Q/P/L]A columns
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
        total_phase1_weight = 0.40 + 0.10 + 0.15
        w1 = 0.40 / total_phase1_weight
        w2 = 0.10 / total_phase1_weight
        w3 = 0.15 / total_phase1_weight
    else:
        w1, w2, w3 = 0.40, 0.10, 0.15

    df['OQPLA_Base_Score'] = (
        w1 * df['Efficiency_Score'] +
        w2 * df['Angle_Score'] +
        w3 * df['Distance_Score']
    )

    df['QED_Multiplier'] = 0.5 + 0.5 * df['QED']
    df['OQPLA_Final_Score'] = df['OQPLA_Base_Score'] * df['QED_Multiplier']

    return df


def calculate_pdb_evidence_score(
    df: pd.DataFrame,
    use_pdb: bool = False,
    progress_callback: Optional[ProgressCallback] = None
) -> pd.DataFrame:
    """
    Component 4: PDB Structural Evidence Score (15% weight).

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


def calculate_oqpla_phase2(
    df: pd.DataFrame,
    use_pdb: bool = True,
    progress_callback: Optional[ProgressCallback] = None
) -> pd.DataFrame:
    """
    Calculate O[Q/P/L]A score using Phase 2 components (1-4).

    Args:
        df: DataFrame with efficiency metrics, plane geometry, and SMILES
        use_pdb: If True, query PDB for structural evidence
        progress_callback: Optional callback for progress updates

    Returns:
        pd.DataFrame: Input DataFrame with added O[Q/P/L]A columns
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

    total_phase2_weight = 0.40 + 0.10 + 0.15 + 0.15
    w1 = 0.40 / total_phase2_weight
    w2 = 0.10 / total_phase2_weight
    w3 = 0.15 / total_phase2_weight
    w4 = 0.15 / total_phase2_weight

    df['OQPLA_Base_Score'] = (
        w1 * df['Efficiency_Score'] +
        w2 * df['Angle_Score'] +
        w3 * df['Distance_Score'] +
        w4 * df['PDB_Score']
    )

    df['QED_Multiplier'] = 0.5 + 0.5 * df['QED']
    df['OQPLA_Final_Score'] = df['OQPLA_Base_Score'] * df['QED_Multiplier']

    df['Efficiency_Contribution'] = w1 * df['Efficiency_Score'] * df['QED_Multiplier']
    df['Angle_Contribution'] = w2 * df['Angle_Score'] * df['QED_Multiplier']
    df['Distance_Contribution'] = w3 * df['Distance_Score'] * df['QED_Multiplier']
    df['PDB_Contribution'] = w4 * df['PDB_Score'] * df['QED_Multiplier']

    df['QED_Impact'] = df['OQPLA_Final_Score'] - df['OQPLA_Base_Score']

    return df


def interpret_oqpla_score(score: float) -> Dict[str, str]:
    """Interpret O[Q/P/L]A score and provide classification + recommendation."""
    if np.isnan(score):
        return {
            'classification': 'Invalid',
            'interpretation': 'No score calculated',
            'action': 'Check data quality',
            'priority': None
        }

    if 0.9 <= score <= 1.0:
        return {
            'classification': 'Exceptional IMP',
            'interpretation': 'Highest confidence - multiple validation streams confirm',
            'action': 'Priority 1: Immediate experimental validation',
            'priority': 1
        }
    elif 0.7 <= score < 0.9:
        return {
            'classification': 'Strong IMP',
            'interpretation': 'High confidence - most validation criteria met',
            'action': 'Priority 2: Validate within 1 month',
            'priority': 2
        }
    elif 0.5 <= score < 0.7:
        return {
            'classification': 'Moderate IMP',
            'interpretation': 'Potential lead - some validation, needs more evidence',
            'action': 'Priority 3: Monitor and gather more data',
            'priority': 3
        }
    elif 0.3 <= score < 0.5:
        return {
            'classification': 'Weak IMP',
            'interpretation': 'Low confidence - outlier but lacking validation',
            'action': 'Priority 4: Deprioritize unless novel scaffold',
            'priority': 4
        }
    else:
        return {
            'classification': 'Not IMP',
            'interpretation': 'Likely artifact or not druggable',
            'action': 'Exclude: Do not pursue',
            'priority': None
        }


def add_oqpla_interpretation(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable O[Q/P/L]A interpretation columns to DataFrame."""
    df = df.copy()

    if 'OQPLA_Final_Score' not in df.columns:
        raise ValueError("OQPLA_Final_Score column not found. Run calculate_oqpla_phase1() first.")

    interpretations = df['OQPLA_Final_Score'].apply(interpret_oqpla_score)

    df['OQPLA_Classification'] = interpretations.apply(lambda x: x['classification'])
    df['OQPLA_Priority'] = interpretations.apply(lambda x: x['priority'])

    return df


def get_oqpla_summary(df: pd.DataFrame) -> Dict:
    """Generate summary statistics about O[Q/P/L]A scores in the dataset."""
    if 'OQPLA_Final_Score' not in df.columns:
        return {'error': 'No O[Q/P/L]A scores found'}

    scores = df['OQPLA_Final_Score'].dropna()

    summary = {
        'total_compounds': len(df),
        'scored_compounds': len(scores),
        'mean_score': float(scores.mean()) if len(scores) > 0 else np.nan,
        'median_score': float(scores.median()) if len(scores) > 0 else np.nan,
        'std_score': float(scores.std()) if len(scores) > 0 else np.nan,
        'min_score': float(scores.min()) if len(scores) > 0 else np.nan,
        'max_score': float(scores.max()) if len(scores) > 0 else np.nan
    }

    if 'OQPLA_Classification' in df.columns:
        classification_counts = df['OQPLA_Classification'].value_counts().to_dict()
        summary['classification_counts'] = classification_counts

        summary['exceptional_imps'] = classification_counts.get('Exceptional IMP', 0)
        summary['strong_imps'] = classification_counts.get('Strong IMP', 0)
        summary['moderate_imps'] = classification_counts.get('Moderate IMP', 0)
        summary['weak_imps'] = classification_counts.get('Weak IMP', 0)
        summary['not_imps'] = classification_counts.get('Not IMP', 0)

    if 'OQPLA_Priority' in df.columns:
        priority_counts = df['OQPLA_Priority'].value_counts().sort_index().to_dict()
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
