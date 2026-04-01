"""
Async Compound Processing Service for IMPULATOR.

Module-level functions (no class) for compound processing pipeline.
All pipeline steps use await for network I/O and run_in_executor for CPU work.
Cancellation via asyncio.CancelledError (no JobCancelledException).

Key features:
- Async pipeline via asyncio tasks
- PENDING_UPLOAD completion flow (user sees results before Azure upload)
- Progress callbacks via async DB writes
- Integration with async chemistry modules (19.1)
"""
import asyncio
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy.exc import DBAPIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from backend.config import settings
from backend.core.database import get_db_session
from backend.core.storage_paths import get_storage_path_from_entry_id
from backend.core import sanitize_compound_name
from backend.core.azure_sync import is_azure_configured

from backend.models.enums import JobStatus

# Import async chemistry modules (19.1 async versions)
from backend.modules.api_client import (
    get_chembl_ids,
    get_drug_indications_batch,
    cascade_similarity_counts,
    fetch_all_activities_single_batch,
    fetch_batch_molecule_data,
    fetch_batch_target_names,
    create_chembl_client,
)
from backend.modules.efficiency_metrics import calculate_efficiency_metrics_dataframe
from backend.modules.efficiency_planes import calculate_plane_metrics_dataframe
from backend.modules.outlier_detection import detect_efficiency_outliers
from backend.modules.imp_scoring import (
    calculate_imp_score,
    add_imp_score_interpretation,
    create_detailed_pdb_summary,
)
from backend.modules.imp_classifier import classify_imp_candidates
from backend.modules.assay_interference_filter import get_interference_flags_from_smiles, InterferenceFlags
from backend.modules.chemical_classifier import (
    get_complete_classification,
    create_classifier_client,
)
from backend.modules.pdb_client import create_pdb_client

logger = logging.getLogger(__name__)

# Type alias for progress callback (pct: float 0-1, message: str)
ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------

def _is_connection_error(e: Exception) -> bool:
    """Retry predicate: only retry actual disconnections, not constraint violations."""
    return isinstance(e, DBAPIError) and e.connection_invalidated


import structlog as _structlog

_retry_logger = _structlog.get_logger("compound_service.retry")


def _log_db_retry(retry_state) -> None:
    """Tenacity before_sleep callback — log DB retry attempts (D-55)."""
    _retry_logger.warning(
        "db_retry",
        attempt=retry_state.attempt_number,
        wait=getattr(retry_state.next_action, "sleep", None),
        fn=getattr(retry_state.fn, "__name__", str(retry_state.fn)),
    )


# ---------------------------------------------------------------------------
# Recovery markers (sync -- filesystem only)
# ---------------------------------------------------------------------------

def _write_recovery_marker(entry_id: str, job_id, compound_name: str,
                            result_summary: dict, completed_at: str):
    """Write recovery marker for DB-down scenario (D-08, D-09).

    When the database is unreachable after 3 retries, we write the completed
    job result to a JSON file on disk so it can be replayed on next startup.
    """
    marker_path = settings.DATA_DIR / f".recovery-{entry_id}.json"
    data = {
        "job_id": str(job_id),
        "entry_id": entry_id,
        "compound_name": compound_name,
        "status": "COMPLETED",
        "result_summary": result_summary,
        "completed_at": completed_at,
    }
    tmp_path = marker_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, default=str))
    os.replace(str(tmp_path), str(marker_path))  # Atomic on POSIX
    logger.warning(f"Recovery marker written for entry_id={entry_id} (DB unreachable after retries)")


def scan_recovery_markers() -> list:
    """Scan for recovery markers from previous DB-down crashes (D-10).

    Returns:
        List of marker dicts with keys: job_id, entry_id, compound_name,
        status, result_summary, completed_at.
    """
    markers = []
    for f in settings.DATA_DIR.glob(".recovery-*.json"):
        try:
            markers.append(json.loads(f.read_text()))
        except Exception as e:
            logger.warning(f"Corrupt recovery marker {f}: {e}")
    # Also clean up orphaned .tmp files
    for f in settings.DATA_DIR.glob(".recovery-*.tmp"):
        try:
            f.unlink()
            logger.info(f"Cleaned orphaned recovery tmp: {f.name}")
        except Exception:
            pass
    return markers


# ---------------------------------------------------------------------------
# Cleanup (sync -- filesystem only)
# ---------------------------------------------------------------------------

def cleanup_stale_folders() -> int:
    """Remove stale compound processing folders from data/results/.

    During processing, compound_service creates folders like data/results/Aspirin/
    which are converted to ZIPs and deleted. If the process crashes mid-processing,
    these folders remain as orphans. Since recover_on_startup() has already reset
    PROCESSING jobs to PENDING, any remaining folders are stale.

    Only removes directories (not ZIP files or UUID-prefix subdirs).

    Returns:
        Number of cleaned folders.
    """
    results_dir = settings.RESULTS_DIR
    if not results_dir.exists():
        return 0

    cleaned = 0
    for item in results_dir.iterdir():
        if not item.is_dir():
            continue
        # Skip UUID-prefix subdirs (2-char hex: "3a", "7f", etc.) -- these contain ZIPs
        if len(item.name) == 2 and all(c in '0123456789abcdef' for c in item.name.lower()):
            continue
        # This is a compound processing folder (e.g., "Aspirin", "Caffeine")
        try:
            shutil.rmtree(item)
            cleaned += 1
            logger.info(f"Cleaned up stale compound folder: {item.name}")
        except Exception as e:
            logger.warning(f"Failed to clean up stale folder {item.name}: {e}")

    if cleaned:
        logger.info(f"Cleaned up {cleaned} stale compound folder(s)")
    return cleaned


# ---------------------------------------------------------------------------
# Async progress/status helpers (D-21)
# ---------------------------------------------------------------------------

async def _update_progress(job_id, progress, current_step, status=JobStatus.PROCESSING):
    """Async progress update wrapping sync DB write in executor (D-21)."""
    loop = asyncio.get_running_loop()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_connection_error),
        reraise=True,
        before_sleep=_log_db_retry,
    )
    def _do():
        with get_db_session() as db:
            from backend.services.job_service import job_service
            job_service.update_progress(db, job_id, progress, current_step, status)

    await loop.run_in_executor(None, _do)
    logger.debug(f"Job {job_id}: {progress}% - {current_step}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=5),
    retry=retry_if_exception(_is_connection_error),
    reraise=True,
    before_sleep=_log_db_retry,
)
def _mark_pending_upload_sync(job_id, result_summary):
    """Sync DB write wrapped for run_in_executor."""
    with get_db_session() as db:
        from backend.services.job_service import job_service
        return job_service.mark_pending_upload(db, str(job_id), result_summary)


async def _mark_pending_upload_with_retry(job_id, result_summary):
    """Async wrapper: runs sync mark_pending_upload in executor with tenacity retry."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _mark_pending_upload_sync, job_id, result_summary)


def _mark_completed_sync(job_id):
    """Sync DB write for mark_completed (<10ms)."""
    with get_db_session() as db:
        from backend.services.job_service import job_service
        job_service.mark_completed(db, str(job_id))


async def _fail_job_with_retry(job_id, error_message, cascade_results=None):
    """Async wrapper for retry-wrapped job failure (D-05, D-06)."""
    loop = asyncio.get_running_loop()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception(_is_connection_error),
        reraise=True,
        before_sleep=_log_db_retry,
    )
    def _do():
        with get_db_session() as db:
            from backend.services.job_service import job_service
            job_service.fail_job(db, job_id, error_message, cascade_results=cascade_results)

    await loop.run_in_executor(None, _do)


# ---------------------------------------------------------------------------
# Inline Azure upload attempt
# ---------------------------------------------------------------------------

async def _try_inline_upload(result_path, entry_id):
    """Try Azure upload inline with tenacity retry (D-37)."""
    from backend.core.azure_sync import (
        write_pending_marker, delete_pending_marker,
        _upload_with_retry,
    )
    loop = asyncio.get_running_loop()

    await loop.run_in_executor(None, write_pending_marker, entry_id)
    await loop.run_in_executor(None, _upload_with_retry, str(result_path), entry_id)
    await loop.run_in_executor(None, delete_pending_marker, entry_id)


# ---------------------------------------------------------------------------
# CPU-bound helper functions (called via run_in_executor)
# ---------------------------------------------------------------------------

def _calculate_molecular_descriptors_sync(  # pragma: no cover -- RDKit heavy computation
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate molecular descriptors from SMILES using RDKit.

    Adds: Heavy_Atoms, NPOL, QED, Aromatic_Rings, RO5_Violations, HBD, HBA, LogP,
          Rotatable_Bonds, and other derived columns.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import CalcMolDescriptors
    except ImportError:
        logger.warning("RDKit not available - skipping molecular descriptor calculation")
        return df

    # Try to import NP Likeness scorer
    np_scorer = None
    try:
        from rdkit.Chem import RDConfig
        from rdkit.Contrib.NP_Score import npscorer
        fscore_data = os.path.join(RDConfig.RDContribDir, 'NP_Score', 'publicnp.model.gz')
        np_scorer_obj = npscorer.readNPModel(fscore_data)

        def np_scorer(mol):
            return npscorer.scoreMol(mol, np_scorer_obj)

        logger.info("NP Likeness scorer loaded from Contrib")
    except Exception as e1:
        try:
            from rdkit.Chem.Descriptors import CalcNPScore
            np_scorer = CalcNPScore
            logger.info("NP Likeness scorer loaded from Descriptors")
        except Exception as e2:
            logger.debug(f"NP Likeness scorer not available: {e1}, {e2}")

    df = df.copy()

    descriptor_cols = [
        'Heavy_Atoms', 'NPOL', 'QED', 'TPSA',
        'Aromatic_Rings', 'Rotatable_Bonds',
        'HBD', 'HBA', 'LogP',
        'RO5_Violations', 'NP_Likeness_Score',
        'PSAoMW', '10xPSA_MW', 'NPOLoNHA'
    ]
    for col in descriptor_cols:
        if col not in df.columns:
            df[col] = np.nan

    unique_smiles = df['SMILES'].dropna().unique()
    descriptor_cache = {}

    def calculate_descriptors_for_smiles(smiles_str: str) -> dict:
        """Calculate molecular descriptors using CalcMolDescriptors batch call (D-17).

        Uses ExactMolWt (monoisotopic) instead of MolWt (average) per D-18.
        NPOL (N+O count) is not in CalcMolDescriptors — computed separately.
        """
        result = {col: np.nan for col in descriptor_cols}
        try:
            mol = Chem.MolFromSmiles(str(smiles_str))
            if mol is None:
                return result

            # Single batch call replaces 8 individual Calc*/Descriptors.* calls
            all_descs = CalcMolDescriptors(mol)

            result['Heavy_Atoms'] = all_descs['HeavyAtomCount']
            result['QED'] = all_descs['qed']
            result['TPSA'] = all_descs['TPSA']
            result['Aromatic_Rings'] = all_descs['NumAromaticRings']
            result['Rotatable_Bonds'] = all_descs['NumRotatableBonds']
            result['HBD'] = all_descs['NumHDonors']
            result['HBA'] = all_descs['NumHAcceptors']
            result['LogP'] = all_descs['MolLogP']

            # NPOL = nitrogen + oxygen count (not in CalcMolDescriptors)
            n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
            o_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
            result['NPOL'] = n_count + o_count

            # D-18: ExactMolWt (monoisotopic) instead of MolWt (average)
            mw = all_descs['ExactMolWt']
            violations = 0
            if mw > 500:
                violations += 1
            if result['LogP'] > 5:
                violations += 1
            if result['HBD'] > 5:
                violations += 1
            if result['HBA'] > 10:
                violations += 1
            result['RO5_Violations'] = violations

            tpsa_val = result['TPSA']
            if mw > 0 and tpsa_val is not None and not np.isnan(tpsa_val):
                result['PSAoMW'] = tpsa_val / mw
                result['10xPSA_MW'] = 10 * result['PSAoMW']

            heavy_atoms = result['Heavy_Atoms']
            npol = result['NPOL']
            if heavy_atoms is not None and not np.isnan(heavy_atoms) and heavy_atoms > 0:
                if npol is not None and not np.isnan(npol):
                    result['NPOLoNHA'] = npol / heavy_atoms

            if np_scorer is not None:
                try:
                    result['NP_Likeness_Score'] = np_scorer(mol)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Error calculating descriptors for {smiles_str[:30]}...: {e}")

        return result

    total_unique = len(unique_smiles)
    logger.info(f"Computing descriptors for {total_unique} unique SMILES...")

    for i, smiles in enumerate(unique_smiles):
        if not smiles or smiles == 'nan':
            continue
        descriptor_cache[smiles] = calculate_descriptors_for_smiles(smiles)

    for col in descriptor_cols:
        mask = df[col].isna()
        values = df.loc[mask, 'SMILES'].apply(
            lambda s, fn=col: descriptor_cache.get(s, {}).get(fn, np.nan) if pd.notna(s) else np.nan
        )
        df.loc[mask, col] = pd.to_numeric(values, errors='coerce')

    return df


def _add_assay_interference_flags_sync(df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover -- RDKit SMARTS matching
    """Add assay interference flags to the DataFrame (CPU-bound)."""
    if 'SMILES' not in df.columns:
        logger.warning("SMILES column not found, skipping interference analysis")
        return df

    df = df.copy()

    bool_mapping = {
        'pains': 'PAINS_Violation',
        'aggregator': 'Aggregator_Risk',
        'redox': 'Redox_Reactive',
        'fluorescence': 'Fluorescence_Interference',
        'thiol': 'Thiol_Reactive',
        'brenk': 'BRENK_Alerts',
        'nih': 'NIH_Alerts',
    }
    detail_mapping = {
        'pains_details': 'PAINS_Details',
        'aggregator_reason': 'Aggregator_Details',
        'redox_details': 'Redox_Details',
        'fluorescence_details': 'Fluorescence_Details',
        'thiol_details': 'Thiol_Details',
        'brenk_details': 'BRENK_Details',
        'nih_details': 'NIH_Details',
    }

    for col in bool_mapping.values():
        df[col] = False
    for col in detail_mapping.values():
        df[col] = ''

    unique_smiles = df['SMILES'].dropna().unique()
    flags_cache: dict[str, InterferenceFlags] = {}

    logger.info(f"Running interference analysis for {len(unique_smiles)} unique compounds...")

    for i, smiles in enumerate(unique_smiles):
        try:
            flags_cache[smiles] = get_interference_flags_from_smiles(smiles)
        except Exception as e:
            logger.warning(f"Interference analysis failed for {smiles[:30]}...: {e}")
            flags_cache[smiles] = InterferenceFlags()

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(unique_smiles)} compounds for interference")

    for field_name, col_name in bool_mapping.items():
        df[col_name] = df['SMILES'].apply(
            lambda s, fn=field_name: getattr(flags_cache.get(s, InterferenceFlags()), fn, False) if pd.notna(s) else False
        )

    for field_name, col_name in detail_mapping.items():
        def _get_detail(s, fn=field_name):
            if not pd.notna(s):
                return ''
            val = getattr(flags_cache.get(s, InterferenceFlags()), fn, '')
            if isinstance(val, list):
                return ', '.join(val)
            return val or ''
        df[col_name] = df['SMILES'].apply(_get_detail)

    pains_count = df['PAINS_Violation'].sum()
    brenk_count = df['BRENK_Alerts'].sum()
    nih_count = df['NIH_Alerts'].sum()
    logger.info(
        f"Interference analysis complete: {pains_count} PAINS, "
        f"{brenk_count} BRENK, {nih_count} NIH out of {len(df)} records"
    )

    return df


def _calculate_advanced_metrics_sync(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced efficiency metrics (CPU-bound)."""
    try:
        required_columns = [
            'SEI', 'BEI', 'NSEI', 'NBEI', 'nBEI_viz',
            'Modulus_SEI_BEI', 'Angle_SEI_BEI', 'Slope_SEI_BEI',
            'Modulus_NSEI_NBEI', 'Angle_NSEI_NBEI', 'Slope_NSEI_NBEI', 'Intercept_NSEI_NBEI',
            'QED'
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

        metrics_input_cols = ['pActivity', 'TPSA', 'Molecular_Weight', 'NPOL', 'Heavy_Atoms']
        if all(col in df.columns for col in metrics_input_cols):
            df = calculate_efficiency_metrics_dataframe(df)
        else:
            missing = [c for c in metrics_input_cols if c not in df.columns]
            logger.warning(f"Skipping efficiency metrics: missing columns {missing}")

        plane_input_cols = ['SEI', 'BEI', 'NSEI', 'NBEI', 'TPSA', 'Molecular_Weight', 'NPOL', 'Heavy_Atoms']
        if all(col in df.columns for col in plane_input_cols):
            df = calculate_plane_metrics_dataframe(df)
        else:
            missing = [c for c in plane_input_cols if c not in df.columns]
            logger.warning(f"Skipping plane metrics: missing columns {missing}")

        df = detect_efficiency_outliers(df, metrics=['SEI', 'BEI', 'NSEI', 'NBEI'])
        return df

    except Exception as e:
        logger.warning(f"Could not calculate efficiency metrics: {e}")
        return df


def _save_results_sync(
    compound_name: str,
    smiles: str,
    similarity_threshold: int,
    activity_types: list[str] | None,
    df_results: pd.DataFrame,
    indications_df: pd.DataFrame | None = None,
    all_similar_df: pd.DataFrame | None = None,
    entry_id: str | None = None,
    author_name: str | None = None,
) -> tuple:
    """Save results to disk and create ZIP archive (CPU/disk I/O bound)."""
    results_dir = settings.RESULTS_DIR
    safe_name = sanitize_compound_name(compound_name)
    compound_folder = os.path.join(results_dir, safe_name)
    os.makedirs(compound_folder, exist_ok=True)

    try:
        return _save_results_inner(
            compound_name, smiles, similarity_threshold, activity_types,
            df_results, indications_df, all_similar_df, entry_id, author_name,
            safe_name, compound_folder, results_dir,
        )
    except Exception:
        shutil.rmtree(compound_folder, ignore_errors=True)
        logger.warning(f"Cleaned up compound_folder on failure: {compound_folder}")
        raise


def _save_results_inner(
    compound_name: str,
    smiles: str,
    similarity_threshold: int,
    activity_types: list[str] | None,
    df_results: pd.DataFrame,
    indications_df: pd.DataFrame | None,
    all_similar_df: pd.DataFrame | None,
    entry_id: str | None,
    author_name: str | None,
    safe_name: str,
    compound_folder: str,
    results_dir,
) -> tuple:
    """Inner implementation of _save_results_sync (separated for cleanup wrapper)."""
    # Save CSV
    results_filename = os.path.join(compound_folder, f"{safe_name}_complete_results.csv")
    df_results.to_csv(results_filename, index=False)

    # Create metadata
    result_summary = {
        'schema_version': 1,
        'compound_name': compound_name,
        'author_name': author_name or 'N/A',
        'query_smiles': smiles,
        'similarity_threshold': similarity_threshold,
        'activity_types': ','.join(activity_types) if activity_types else 'AC50,EC50,GI50,IC50,Kd,Ki,MIC',
        'processing_date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'total_compounds': df_results['ChEMBL_ID'].nunique() if 'ChEMBL_ID' in df_results.columns else 0,
        'total_bioactivity_rows': len(df_results),
    }

    # Add IMP summary if available
    if 'Is_IMP_Candidate' in df_results.columns:
        result_summary['imp_candidates'] = int(df_results['Is_IMP_Candidate'].sum())
        result_summary['has_imp_candidates'] = result_summary['imp_candidates'] > 0

    # Add interference summary if available
    if 'PAINS_Violation' in df_results.columns:
        result_summary['pains_count'] = int(df_results['PAINS_Violation'].sum())
    if 'BRENK_Alerts' in df_results.columns:
        result_summary['brenk_count'] = int(df_results['BRENK_Alerts'].sum())
    if 'NIH_Alerts' in df_results.columns:
        result_summary['nih_count'] = int(df_results['NIH_Alerts'].sum())

    # Add similar_count for frontend
    result_summary['similar_count'] = result_summary.get('total_compounds', 0)
    result_summary['compounds_with_data'] = result_summary.get('total_compounds', 0)
    result_summary['smiles'] = smiles

    # First ChEMBL ID
    if 'ChEMBL_ID' in df_results.columns and len(df_results) > 0:
        result_summary['chembl_id'] = df_results['ChEMBL_ID'].iloc[0]
    else:
        result_summary['chembl_id'] = ''

    # Total activities count
    result_summary['total_activities'] = len(df_results)

    # Number of outliers
    outlier_cols = [c for c in df_results.columns if 'outlier' in c.lower()]
    if outlier_cols:
        outlier_mask = df_results[outlier_cols].any(axis=1)
        result_summary['num_outliers'] = int(outlier_mask.sum())
    else:
        result_summary['num_outliers'] = 0

    # QED score
    if 'QED' in df_results.columns:
        qed_values = df_results['QED'].dropna()
        if len(qed_values) > 0:
            result_summary['qed'] = float(qed_values.mean())
        else:
            result_summary['qed'] = 0.0
    else:
        result_summary['qed'] = 0.0

    # IMP score (max if available)
    if 'IMP_Final_Score' in df_results.columns:
        imp_values = df_results['IMP_Final_Score'].dropna()
        if len(imp_values) > 0:
            result_summary['imp_score'] = float(imp_values.max())
        else:
            result_summary['imp_score'] = None
    else:
        result_summary['imp_score'] = None

    # Also save CSV with standard name for frontend
    similar_csv = os.path.join(compound_folder, "similar_compounds.csv")
    df_results.to_csv(similar_csv, index=False)

    # Create detailed PDB summary if PDB data is available
    if 'PDB_IDs' in df_results.columns:
        try:
            logger.info("Creating detailed PDB summary...")
            pdb_summary_df = create_detailed_pdb_summary(df_results)

            if not pdb_summary_df.empty:
                pdb_summary_csv = os.path.join(compound_folder, "pdb_summary.csv")
                pdb_summary_df.to_csv(pdb_summary_csv, index=False)
                logger.info(f"Saved detailed PDB summary with {len(pdb_summary_df)} structures")
                result_summary['pdb_structures_count'] = len(pdb_summary_df)
        except Exception as e:
            logger.warning(f"Could not create detailed PDB summary: {e}")

    # Save drug indications as separate CSV
    if indications_df is not None and not indications_df.empty:
        try:
            indications_csv = os.path.join(compound_folder, "drug_indications.csv")
            indications_df.to_csv(indications_csv, index=False)
            logger.info(f"Saved {len(indications_df)} drug indications")
            result_summary['drug_indications_count'] = len(indications_df)
            result_summary['compounds_with_indications'] = indications_df['ChEMBL_ID'].nunique()
        except Exception as e:
            logger.warning(f"Could not save drug indications: {e}")
    else:
        result_summary['drug_indications_count'] = 0
        result_summary['compounds_with_indications'] = 0

    # Save all similar molecules CSV
    if all_similar_df is not None and not all_similar_df.empty:
        try:
            all_similar_csv = os.path.join(compound_folder, "all_similar_molecules.csv")
            all_similar_df.to_csv(all_similar_csv, index=False)
            logger.info(f"Saved {len(all_similar_df)} all similar molecules")
            result_summary['total_similar'] = len(all_similar_df)
        except Exception as e:
            logger.warning(f"Could not save all similar molecules: {e}")

    # Update similar_count to reflect total
    if result_summary.get('total_similar'):
        result_summary['similar_count'] = result_summary['total_similar']

    # Track classification availability
    result_summary['classification_available'] = df_results.attrs.get('classification_available', True)

    # Save metadata
    metadata_filename = os.path.join(compound_folder, f"{safe_name}_metadata.json")
    with open(metadata_filename, 'w') as f:
        json.dump(result_summary, f, indent=4)

    summary_filename = os.path.join(compound_folder, "summary.json")
    with open(summary_filename, 'w') as f:
        json.dump(result_summary, f, indent=4)

    # Create ZIP archive
    if entry_id:
        eid = str(entry_id).lower()
        prefix = eid[:2]
        zip_subdir = os.path.join(results_dir, prefix)
        os.makedirs(zip_subdir, exist_ok=True)
        zip_filename = f"{eid}.zip"
        zip_path = os.path.join(zip_subdir, zip_filename)
    else:
        zip_filename = f"{safe_name}.zip"
        zip_path = os.path.join(results_dir, zip_filename)

    # Write ZIP atomically
    zip_tmp_path = zip_path + ".tmp"
    try:
        with zipfile.ZipFile(zip_tmp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(compound_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, compound_folder)
                    zipf.write(file_path, arcname)
        os.replace(zip_tmp_path, zip_path)
    except Exception:
        if os.path.exists(zip_tmp_path):
            try:
                os.unlink(zip_tmp_path)
            except OSError:
                pass
        raise

    # Clean up folder
    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            shutil.rmtree(compound_folder)
            break
        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Folder cleanup attempt {attempt + 1} failed (file lock), retrying in 1s: {e}")
                time.sleep(1)
            else:
                logger.warning(f"Could not clean up folder {compound_folder} after {max_retries} attempts: {e}")
        except Exception as e:
            logger.warning(f"Error cleaning up folder {compound_folder}: {e}")
            break

    logger.info(f"Saved results to {zip_path}")
    return zip_path, result_summary


# ---------------------------------------------------------------------------
# Main async pipeline (D-14, D-19)
# ---------------------------------------------------------------------------

async def process_compound_job(  # pragma: no cover -- orchestration of 10+ external APIs, tested via integration/e2e
    job_id,
    compound_name: str,
    smiles: str,
    similarity_threshold: int = 90,
    activity_types: list[str] | None = None,
    author_name: str | None = None,
) -> None:
    """Async compound processing pipeline (D-14).

    All I/O uses await. CPU work wrapped in run_in_executor.
    Cancellation via asyncio.CancelledError (D-51 through D-53).
    Completion via PENDING_UPLOAD flow (D-32 through D-36).

    Args:
        job_id: Unique job identifier (UUID)
        compound_name: Name of the compound
        smiles: SMILES string
        similarity_threshold: Similarity threshold (50-100)
        activity_types: List of activity types to fetch
        author_name: Name of the author who submitted the analysis
    """
    entry_id = str(uuid.uuid4())
    logger.info(f"Generated entry_id {entry_id} for job {job_id}")
    loop = asyncio.get_running_loop()

    # Per-job httpx clients via async with context managers (D-52)
    async with create_chembl_client() as chembl_client, \
               create_pdb_client() as pdb_client, \
               create_classifier_client() as classifier_client:
        try:
            # Start processing
            await _update_progress(job_id, 0, "Starting...", JobStatus.PROCESSING)

            # Step 1: Search for similar compounds (20%) -- Network I/O
            await _update_progress(job_id, 5, "Searching ChEMBL for similar compounds...")
            chembl_ids = await get_chembl_ids(chembl_client, smiles, similarity_threshold)

            # Build similarity score lookup
            similarity_scores = {
                d.get('ChEMBL ID'): d.get('Similarity', 0)
                for d in chembl_ids if d.get('ChEMBL ID')
            }
            all_similar_chembl_ids = list(similarity_scores.keys())

            if not chembl_ids:
                cascade = []
                try:
                    await _update_progress(job_id, 10, "Searching lower thresholds...")
                    cascade = await cascade_similarity_counts(chembl_client, smiles, similarity_threshold)
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.debug(f"Cascade probe skipped (ChEMBL unreachable): {e}")
                error_msg = "No similar compounds found in ChEMBL"
                if cascade:
                    error_msg += f" at {similarity_threshold}% threshold"
                await _fail_job_with_retry(job_id, error_msg, cascade_results=cascade)
                return

            await _update_progress(job_id, 20, f"Found {len(chembl_ids)} similar compounds")

            # Step 2: Fetch activities (40%) -- Network I/O
            await _update_progress(job_id, 25, "Fetching bioactivity data...")
            all_results = await _fetch_activities_async(
                chembl_client, chembl_ids, activity_types,
                lambda pct, msg: _update_progress(job_id, 25 + int(pct * 0.15), msg)
            )
            await _update_progress(job_id, 40, f"Retrieved {len(all_results)} bioactivity records")

            if not all_results:
                cascade = []
                try:
                    await _update_progress(job_id, 35, "Searching lower thresholds...")
                    cascade = await cascade_similarity_counts(chembl_client, smiles, similarity_threshold)
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.debug(f"Cascade probe skipped (ChEMBL unreachable): {e}")
                error_msg = "No bioactivity data found"
                if cascade:
                    error_msg += f" at {similarity_threshold}% threshold"
                await _fail_job_with_retry(job_id, error_msg, cascade_results=cascade)
                return

            # Step 3: Process and calculate metrics (60%) -- CPU bound
            await _update_progress(job_id, 42, "Processing compounds & calculating metrics...")
            df_results = pd.DataFrame(all_results)
            df_results.replace("No data", np.nan, inplace=True)

            if 'ChEMBL_ID' in df_results.columns:
                df_results['Similarity'] = df_results['ChEMBL_ID'].map(similarity_scores).fillna(0)

            # Calculate molecular descriptors -- CPU
            await _update_progress(job_id, 44, "Calculating molecular descriptors...")
            df_results = await loop.run_in_executor(None, _calculate_molecular_descriptors_sync, df_results)
            await _update_progress(job_id, 48, "Molecular descriptors complete")

            # Add PAINS and assay interference flags -- CPU
            await _update_progress(job_id, 49, "Running PAINS and assay interference analysis...")
            df_results = await loop.run_in_executor(None, _add_assay_interference_flags_sync, df_results)
            await _update_progress(job_id, 50, "PAINS analysis complete")

            # Calculate advanced metrics -- CPU
            await _update_progress(job_id, 51, "Calculating efficiency metrics...")
            df_results = await loop.run_in_executor(None, _calculate_advanced_metrics_sync, df_results)
            await _update_progress(job_id, 65, "Efficiency metrics complete")

            # Step 4: IMP scoring + PDB (75%) -- async (mixed network + CPU)
            await _update_progress(job_id, 68, "Querying PDB & calculating IMP scores...")
            pdb_unavailable = False
            try:
                df_results = await calculate_imp_score(pdb_client, df_results, use_pdb=True)
                df_results = add_imp_score_interpretation(df_results)
            except Exception as e:
                logger.warning(f"IMP scoring failed: {e}")
                pdb_unavailable = True
            await _update_progress(job_id, 75, "IMP + PDB scoring complete")

            # Step 5: IMP classification (80%) -- CPU
            await _update_progress(job_id, 78, "Classifying IMP candidates...")
            df_results = await loop.run_in_executor(None, classify_imp_candidates, df_results, 2, True)
            await _update_progress(job_id, 80, "IMP classification complete")

            # Step 6: Chemical classification (82%) -- Network I/O
            await _update_progress(job_id, 81, "Getting chemical classifications...")
            df_results = await _add_chemical_classification_async(classifier_client, df_results)
            await _update_progress(job_id, 84, "Chemical classification complete")

            # Step 6.5: Build all similar molecules catalog -- Network I/O
            await _update_progress(job_id, 84, "Building similar molecules catalog...")
            try:
                all_similar_df = await _build_all_similar_df_async(
                    chembl_client, classifier_client, all_similar_chembl_ids,
                    similarity_scores, df_results,
                )
            except Exception as e:
                logger.warning(f"Failed to build all similar molecules catalog: {e}")
                all_similar_df = pd.DataFrame()

            # Step 6.6: Fetch drug indications -- Network I/O
            await _update_progress(job_id, 87, "Fetching drug indications...")
            indications_df = await _fetch_drug_indications_async(chembl_client, df_results)
            await _update_progress(job_id, 89, f"Drug indications complete ({len(indications_df)} found)")

            # Step 7: Save results (90%) -- Disk I/O + CPU
            await _update_progress(job_id, 89, "Saving results...")
            result_path, result_summary = await loop.run_in_executor(
                None, _save_results_sync,
                compound_name, smiles, similarity_threshold, activity_types,
                df_results, indications_df, all_similar_df, entry_id, author_name,
            )
            result_summary['entry_id'] = entry_id
            if pdb_unavailable:
                result_summary['pdb_unavailable'] = True
            await _update_progress(job_id, 90, "Results saved")

            # Explicit DataFrame cleanup
            del df_results
            del indications_df
            del all_similar_df

            # Step 8: Mark PENDING_UPLOAD (D-32) -- compound added to DB, user can browse
            result_summary['storage_path'] = get_storage_path_from_entry_id(entry_id)
            await _update_progress(job_id, 92, "Finalizing job...")
            await _mark_pending_upload_with_retry(job_id, result_summary)
            logger.info(f"Job {job_id} marked PENDING_UPLOAD")

            # Step 9: Inline Azure upload attempt (D-37)
            if is_azure_configured():
                try:
                    await _try_inline_upload(result_path, entry_id)
                    await loop.run_in_executor(None, _mark_completed_sync, job_id)
                    logger.info(f"Job {job_id} completed with Azure upload")
                except Exception:
                    logger.warning(f"Job {job_id} inline Azure upload failed, background worker will retry")
                    # Stays PENDING_UPLOAD -- upload_worker picks up
            else:
                # No Azure = immediate COMPLETED (D-35)
                await loop.run_in_executor(None, _mark_completed_sync, job_id)
                logger.info(f"Job {job_id} completed (no Azure)")

        except asyncio.CancelledError:
            # D-52: Do NOT touch DB status -- caller already set it
            logger.info(f"Job {job_id} cancelled")
        except Exception as e:
            logger.exception(f"Job {job_id} failed with unexpected error: {type(e).__name__}: {e}")
            try:
                await _fail_job_with_retry(job_id, f"Unexpected error: {type(e).__name__}: {e}")
            except Exception:
                # If we can't even fail the job, write recovery marker
                if entry_id:
                    _write_recovery_marker(
                        entry_id=entry_id,
                        job_id=job_id,
                        compound_name=compound_name,
                        result_summary={},
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )


# ---------------------------------------------------------------------------
# Async helper functions for pipeline steps
# ---------------------------------------------------------------------------

async def _fetch_activities_async(
    client,
    chembl_ids: list[dict[str, str]],
    activity_types: list[str] | None,
    progress_coro_factory,
) -> list[dict]:
    """Fetch bioactivity data using async single-batch approach."""
    if activity_types is None:
        activity_types = ['IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'GI50', 'MIC']

    all_chembl_ids = [
        d.get('ChEMBL ID') for d in chembl_ids
        if d.get('ChEMBL ID')
    ]

    if not all_chembl_ids:
        return []

    # Fetch ALL activities in one async batch call
    raw_activities = await fetch_all_activities_single_batch(
        client, all_chembl_ids,
        activity_types=activity_types,
    )

    if not raw_activities:
        return []

    # Build molecule data cache for all unique IDs
    unique_ids = list(set(a.get('molecule_chembl_id') for a in raw_activities if a.get('molecule_chembl_id')))
    unique_target_ids = list(set(a.get('target_chembl_id') for a in raw_activities if a.get('target_chembl_id')))

    # Batch fetch molecule data and target names in parallel
    mol_cache, target_name_cache = await asyncio.gather(
        fetch_batch_molecule_data(client, unique_ids),
        fetch_batch_target_names(client, unique_target_ids),
    )

    # Process activities into final format (CPU-bound but fast)
    all_results = []
    for act in raw_activities:
        chembl_id = act.get('molecule_chembl_id')
        mol_data = mol_cache.get(chembl_id)

        if not mol_data:
            continue

        mol_props = mol_data.get('molecule_properties', {}) or {}
        mol_structures = mol_data.get('molecule_structures', {}) or {}
        compound_smiles = mol_structures.get('canonical_smiles', '')
        mol_name = mol_data.get('pref_name') or 'Unknown'

        std_value = act.get('standard_value')
        std_units = act.get('standard_units')

        if not std_value:
            continue

        try:
            value = float(std_value)
            if value <= 0:
                continue

            value_nM = None
            if std_units == 'nM':
                value_nM = value
            elif std_units == 'uM':
                value_nM = value * 1000
            elif std_units == 'mM':
                value_nM = value * 1000000
            elif std_units == 'pM':
                value_nM = value / 1000
            elif std_units == 'M':
                value_nM = value * 1e9
            else:
                continue

            if value_nM <= 0:
                continue

            pActivity = -np.log10(value_nM * 1e-9)
            target_chembl_id = act.get('target_chembl_id', '')
            target_name = target_name_cache.get(target_chembl_id, '')

            # Map single-char assay_type to human-readable label
            _assay_type_raw = act.get('assay_type', '')
            _ASSAY_TYPE_MAP = {
                'B': 'Binding', 'F': 'Functional', 'A': 'ADMET',
                'T': 'Toxicity', 'P': 'Physicochemical', 'U': 'Unclassified',
            }
            assay_type_label = _ASSAY_TYPE_MAP.get(_assay_type_raw, _assay_type_raw or 'Unknown')

            # Data quality flag
            dvc = act.get('data_validity_comment')
            data_quality = 'Clean' if dvc is None or dvc == 'Manually validated' else 'Flagged'

            all_results.append({
                'ChEMBL_ID': chembl_id,
                'Molecule_Name': mol_name,
                'SMILES': compound_smiles,
                'Molecular_Weight': float(mol_props.get('full_mwt') or 0) or np.nan,
                'TPSA': float(mol_props.get('psa') or 0) or np.nan,
                'Activity_Type': act.get('standard_type', ''),
                'Activity_nM': value_nM,
                'pActivity': pActivity,
                'Target_ChEMBL_ID': target_chembl_id,
                'Target_Name': target_name,
                'Assay_Type': assay_type_label,
                'Document_Year': int(act['document_year']) if act.get('document_year') else None,
                'Data_Quality': data_quality,
                'Activity_Comment': act.get('activity_comment', ''),
            })
        except (ValueError, TypeError):
            continue

    logger.info(f"Fetched target names for {len(target_name_cache)} unique targets")
    return all_results


async def _add_chemical_classification_async(
    client,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add chemical classification from ClassyFire and NPClassifier (async)."""
    if 'SMILES' not in df.columns:
        logger.warning("SMILES column not found, skipping classification")
        return df

    df = df.copy()

    classification_cols = [
        'Kingdom', 'Superclass', 'Class', 'Subclass', 'Direct_Parent',
        'Molecular_Framework', 'Description', 'ChEMONT_ID_Class', 'ChEMONT_ID_Subclass',
        'NP_Pathway', 'NP_Superclass', 'NP_Class', 'NP_isglycoside'
    ]
    for col in classification_cols:
        if col not in df.columns:
            df[col] = ''

    unique_smiles = df['SMILES'].dropna().unique()
    classification_cache = {}
    any_classification_succeeded = False

    logger.info(f"Getting chemical classifications for {len(unique_smiles)} unique compounds...")

    for i, smiles_val in enumerate(unique_smiles):
        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import MolToInchiKey

            mol = Chem.MolFromSmiles(smiles_val)
            if mol:
                inchikey = MolToInchiKey(mol)
                if inchikey:
                    classification = await get_complete_classification(client, smiles_val, inchikey)
                    classification_cache[smiles_val] = classification
                    if classification and classification.get('classification_available', False):
                        any_classification_succeeded = True
                else:
                    classification_cache[smiles_val] = {}
            else:
                classification_cache[smiles_val] = {}

            if (i + 1) % 10 == 0:
                logger.info(f"Classified {i + 1}/{len(unique_smiles)} compounds")

        except Exception as e:
            logger.warning(f"Classification failed for SMILES {smiles_val[:30]}...: {e}")
            classification_cache[smiles_val] = {}

    for col in classification_cols:
        df[col] = df['SMILES'].apply(
            lambda s: classification_cache.get(s, {}).get(col, '') if pd.notna(s) else ''
        )

    df.attrs['classification_available'] = any_classification_succeeded

    logger.info(f"Chemical classification complete for {len(unique_smiles)} compounds (available={any_classification_succeeded})")
    return df


async def _build_all_similar_df_async(
    chembl_client,
    classifier_client,
    all_chembl_ids: list[str],
    similarity_scores: dict[str, float],
    df_results: pd.DataFrame,
) -> pd.DataFrame:
    """Build DataFrame of ALL similar compounds (async version)."""
    if not all_chembl_ids:
        return pd.DataFrame()

    loop = asyncio.get_running_loop()

    compounds_with_data = set()
    if df_results is not None and 'ChEMBL_ID' in df_results.columns:
        compounds_with_data = set(df_results['ChEMBL_ID'].dropna().unique())
    new_chembl_ids = [cid for cid in all_chembl_ids if cid not in compounds_with_data]

    # Extract rows for already-processed compounds
    reuse_cols = ['ChEMBL_ID', 'Molecule_Name', 'SMILES', 'Molecular_Weight', 'TPSA',
                  'QED', 'LogP', 'HBA', 'HBD', 'Heavy_Atoms',
                  'PAINS_Violation', 'Aggregator_Risk', 'BRENK_Alerts', 'NIH_Alerts',
                  'Kingdom', 'Superclass']
    existing_rows = pd.DataFrame()
    if compounds_with_data and df_results is not None:
        available_cols = [c for c in reuse_cols if c in df_results.columns]
        existing_rows = df_results[available_cols].drop_duplicates('ChEMBL_ID').copy()
        existing_rows['Has_Bioactivity'] = True
        existing_rows['Similarity'] = existing_rows['ChEMBL_ID'].map(similarity_scores).fillna(0)

    new_rows = pd.DataFrame()
    if new_chembl_ids:
        mol_data = await fetch_batch_molecule_data(chembl_client, new_chembl_ids)

        rows = []
        for chembl_id in new_chembl_ids:
            mol = mol_data.get(chembl_id, {})
            props = mol.get('molecule_properties', {}) or {}
            rows.append({
                'ChEMBL_ID': chembl_id,
                'Molecule_Name': (mol.get('pref_name') or chembl_id),
                'SMILES': (mol.get('molecule_structures') or {}).get('canonical_smiles', ''),
                'Molecular_Weight': props.get('full_mwt'),
                'TPSA': props.get('psa'),
                'Similarity': similarity_scores.get(chembl_id, 0),
            })

        new_rows = pd.DataFrame(rows)
        if not new_rows.empty:
            for col in ['Molecular_Weight', 'TPSA', 'Similarity']:
                if col in new_rows.columns:
                    new_rows[col] = pd.to_numeric(new_rows[col], errors='coerce')

            new_rows['Has_Bioactivity'] = False

            # CPU-bound steps via run_in_executor
            new_rows = await loop.run_in_executor(None, _calculate_molecular_descriptors_sync, new_rows)
            new_rows = await loop.run_in_executor(None, _add_assay_interference_flags_sync, new_rows)

            # Async classification
            new_rows = await _add_chemical_classification_async(classifier_client, new_rows)

    # Combine existing + new
    dfs_to_concat = [d for d in [existing_rows, new_rows] if not d.empty]
    if not dfs_to_concat:
        return pd.DataFrame()

    df = pd.concat(dfs_to_concat, ignore_index=True)
    # Ensure Similarity is numeric before sorting (API may return str or int)
    if 'Similarity' in df.columns:
        df['Similarity'] = pd.to_numeric(df['Similarity'], errors='coerce').fillna(0)
    df = df.sort_values('Similarity', ascending=False).reset_index(drop=True)
    return df


async def _fetch_drug_indications_async(
    client,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch drug indications for all unique ChEMBL IDs (async)."""
    if 'ChEMBL_ID' not in df.columns:
        logger.warning("ChEMBL_ID column not found, skipping drug indications")
        return pd.DataFrame()

    unique_ids = list(df['ChEMBL_ID'].dropna().unique())
    total = len(unique_ids)

    if total == 0:
        logger.info("No ChEMBL IDs to fetch indications for")
        return pd.DataFrame()

    logger.info(f"Fetching drug indications for {total} unique compounds (batch)...")

    try:
        # get_drug_indications_batch returns (all_list, by_compound_dict)
        _all_raw, indications_by_id = await get_drug_indications_batch(client, unique_ids)

        all_indications = []
        for chembl_id, indications in indications_by_id.items():
            for ind in indications:
                all_indications.append(dict(ind))

        if all_indications:
            indications_df = pd.DataFrame(all_indications)
            logger.info(f"Found {len(indications_df)} drug indications across {indications_df['ChEMBL_ID'].nunique()} compounds")
            return indications_df
        else:
            logger.info("No drug indications found for any compounds")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Batch drug indications fetch failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Compound API service functions (delete, versions, list)
# These are separate from the processing pipeline above. They handle
# compound CRUD orchestration for the API layer (ARCH-04).
# ---------------------------------------------------------------------------

from backend.core.azure_sync import delete_result_from_azure_by_entry_id  # noqa: E402 -- deferred to avoid circular imports
from backend.core.audit import log_job_deleted  # noqa: E402 -- deferred to avoid circular imports
from backend.core.auth import truncate_session_id  # noqa: E402 -- deferred to avoid circular imports
from backend.repositories import compound_repo  # noqa: E402 -- deferred to avoid circular imports
from backend.models.schemas import (  # noqa: E402 -- deferred to avoid circular imports
    CompoundDeleteResponse,
    BatchDeleteResponse,
)
from sqlalchemy.orm import Session  # noqa: E402


def get_compound_versions(db: Session, entry_id: str) -> dict:
    """Resolve all structural siblings (versions) of a compound.

    Uses parent_id/version for versioning (not is_duplicate/duplicate_of).
    Returns dict with 'versions' list and 'current_entry_id'.
    Raises ValueError if compound not found.
    """
    siblings = compound_repo.get_versions(db, entry_id)

    if len(siblings) <= 1:
        if not siblings:
            compound = compound_repo.get_by_entry_id(db, entry_id)
            if not compound:
                raise ValueError("Compound not found")
        return {"versions": [], "current_entry_id": entry_id}

    # Identify the original: root compound (parent_id is None), fallback to oldest overall
    original_entry_id = None
    for s in siblings:
        if s.parent_id is None:  # Root compound = original
            original_entry_id = s.entry_id
            break
    if original_entry_id is None:
        original_entry_id = siblings[0].entry_id

    # Batch-resolve parent names for child compounds
    parent_entry_ids = {s.parent_id for s in siblings if s.parent_id is not None}
    parent_names = {}
    if parent_entry_ids:
        for pid in parent_entry_ids:
            parent = compound_repo.get_by_entry_id(db, pid)
            if parent:
                parent_names[pid] = parent.compound_name

    versions = []
    for s in siblings:
        versions.append({
            "entry_id": s.entry_id,
            "compound_name": s.compound_name,
            "similarity_threshold": s.similarity_threshold,
            "activity_types": s.activity_types,  # Already a list (ARRAY)
            "imp_score": s.imp_score,
            "qed": s.qed,
            "similar_compounds": s.similar_compounds or 0,
            "total_activities": s.total_activities,
            "parent_id": str(s.parent_id) if s.parent_id else None,
            "version": s.version,
            "config_diff": s.config_diff,  # JSONB -- already a dict
            "parent_name": parent_names.get(s.parent_id) if s.parent_id else None,
            "author_name": s.author_name,
            "processed_at": s.processed_at.isoformat() if s.processed_at else None,
            "storage_path": s.storage_path,
            "is_original": s.entry_id == original_entry_id,
            "is_current": str(s.entry_id) == str(entry_id),
        })

    return {"versions": versions, "current_entry_id": entry_id}


def delete_compound_with_cleanup(
    db: Session, entry_id: str, session_id: str
) -> CompoundDeleteResponse:
    """Delete a compound with full cleanup: Azure, local files, DB archive.

    Raises ValueError if compound not found.
    """
    # Fast-fail: check compound exists before doing any I/O
    compound = compound_repo.get_by_entry_id(db, entry_id)
    if not compound:
        raise ValueError("Compound not found")

    # Delete from Azure FIRST (UUID-based storage only) -- outside lock
    azure_deleted = delete_result_from_azure_by_entry_id(entry_id)
    if azure_deleted:
        logger.info(f"Deleted result from Azure: {entry_id}")

    # Delete local ZIP file -- outside lock
    eid = str(entry_id).lower()
    prefix = eid[:2]
    local_zip = settings.RESULTS_DIR / prefix / f"{eid}.zip"
    if local_zip.exists():
        try:
            local_zip.unlink()
            logger.info(f"Deleted local result: {local_zip}")
        except Exception as e:
            logger.warning(f"Failed to delete local result {local_zip}: {e}")

    # DB mutations
    compound = compound_repo.get_by_entry_id(db, entry_id)
    if not compound:
        raise ValueError("Compound not found")

    compound_name = compound.compound_name

    # Archive to deleted_compounds table before deletion
    compound_repo.archive_compound(
        db,
        compound,
        deleted_by=uuid.UUID(session_id) if session_id else None,
        deletion_reason="user_request",
    )

    # Delete from compounds table
    compound_repo.delete_compound(db, compound)
    db.commit()

    # Audit log -- DB table + file-based
    from backend.services._audit import log_audit_event
    from backend.models.enums import AuditEventType
    log_audit_event(
        db, AuditEventType.COMPOUND_DELETED,
        session_id=uuid.UUID(session_id) if session_id else None,
        details={"entry_id": str(entry_id), "compound_name": compound_name},
    )
    db.commit()
    logger.info(f"Deleted compound: {compound_name} (entry_id={entry_id})")

    return CompoundDeleteResponse(
        status="deleted",
        entry_id=entry_id,
        message=f"Compound '{compound_name}' deleted successfully",
    )


def batch_delete_with_cleanup(
    db: Session, entry_ids: list, session_id: str
) -> BatchDeleteResponse:
    """Delete multiple compounds with full cleanup.

    Validates input, deduplicates, archives, then cleans storage.
    Raises ValueError for invalid input.
    """
    if not entry_ids:
        raise ValueError("entry_ids list cannot be empty")

    if len(entry_ids) > 50:
        raise ValueError("Cannot delete more than 50 compounds at once")

    # Validate all entry_ids are non-empty strings
    for eid in entry_ids:
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError("All entry_ids must be non-empty strings")

    deleted = []
    not_found = []

    # Deduplicate to prevent same ID appearing in both deleted and not_found
    seen = set()
    unique_entry_ids = []
    for eid in entry_ids:
        if eid not in seen:
            seen.add(eid)
            unique_entry_ids.append(eid)

    # Sort: delete children (parent_id not None) before parents to avoid FK violations.
    # Fetch all compounds first, then sort by dependency.
    compounds_to_delete = []
    for eid in unique_entry_ids:
        compound = compound_repo.get_by_entry_id(db, eid)
        if not compound:
            not_found.append(eid)
            continue
        compounds_to_delete.append(compound)

    # Children first (parent_id is not None), then parents (parent_id is None)
    compounds_to_delete.sort(key=lambda c: (c.parent_id is None, c.compound_name))

    for compound in compounds_to_delete:
        eid = str(compound.entry_id)
        compound_name = compound.compound_name

        # Archive to deleted_compounds (before delete triggers reparenting)
        compound_repo.archive_compound(
            db,
            compound,
            deleted_by=uuid.UUID(session_id) if session_id else None,
            deletion_reason="batch_delete",
        )
        compound_repo.delete_compound(db, compound)

        deleted.append({"entry_id": eid, "compound_name": compound_name})

        logger.info(f"Batch delete - archived: {compound_name} ({eid})")

    # Commit DB first -- only delete storage after successful commit
    db.commit()

    # Audit log outside lock
    for item in deleted:
        log_job_deleted(truncate_session_id(session_id), item["entry_id"], item["compound_name"])

    # Now safe to delete storage
    for item in deleted:
        eid = item["entry_id"]
        try:
            azure_ok = delete_result_from_azure_by_entry_id(eid)
            if azure_ok:
                logger.info(f"Batch delete - Azure deleted: {eid}")
        except Exception as e:
            logger.warning(f"Batch delete - Azure cleanup failed for {eid}: {e}")

        prefix = eid[:2].lower()
        local_zip = settings.RESULTS_DIR / prefix / f"{eid}.zip"
        if local_zip.exists():
            try:
                local_zip.unlink()
                logger.info(f"Batch delete - local deleted: {local_zip}")
            except Exception as e:
                logger.warning(f"Batch delete - failed to delete local {local_zip}: {e}")

    return BatchDeleteResponse(
        status="completed",
        deleted=[item["entry_id"] for item in deleted],
        failed=[{"entry_id": eid, "error": "not found"} for eid in not_found],
        total_deleted=len(deleted),
        total_failed=len(not_found),
    )
