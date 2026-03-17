"""
Compound Processing Service for IMPULATOR.

This service handles compound processing in background threads,
integrates with chemistry modules, and syncs results to Azure.

Key features:
- Background processing via ThreadPoolExecutor
- Progress callbacks (no Streamlit dependencies)
- Immediate Azure sync on completion
- Integration with existing chemistry modules
"""
import os
import sys
import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd
import numpy as np

# Ensure project root is in path for module imports (do this ONCE at module level)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import settings  # noqa: E402
from backend.core.database import get_db_session  # noqa: E402
from backend.core.azure_sync import (  # noqa: E402
    upload_result_to_azure_by_entry_id,
    get_storage_path_from_entry_id,
)
from backend.core import sanitize_compound_name  # noqa: E402
from backend.models.database import JobStatus  # noqa: E402

# Import chemistry modules (clean absolute imports)
from backend.modules.api_client import (  # noqa: E402
    get_chembl_ids,
    get_drug_indications_batch,
    cascade_similarity_counts,
)
from backend.modules.efficiency_metrics import calculate_efficiency_metrics_dataframe  # noqa: E402
from backend.modules.efficiency_planes import calculate_plane_metrics_dataframe  # noqa: E402
from backend.modules.outlier_detection import detect_efficiency_outliers  # noqa: E402
from backend.modules.imp_scoring import (  # noqa: E402
    calculate_imp_score,
    add_imp_score_interpretation,
    create_detailed_pdb_summary,
)
from backend.modules.imp_classifier import classify_imp_candidates  # noqa: E402
from backend.modules.assay_interference_filter import get_interference_flags_from_smiles, InterferenceFlags  # noqa: E402
from backend.modules.chemical_classifier import get_complete_classification  # noqa: E402

logger = logging.getLogger(__name__)


# Type alias for progress callback (pct: float 0-1, message: str)
ProgressCallback = Callable[[float, str], None]


class CompoundService:
    """
    Service for processing compounds.

    Wraps the chemistry modules and provides:
    - Progress tracking via callbacks
    - Background job execution
    - Azure sync on completion
    """

    def __init__(self):
        self.results_dir = settings.RESULTS_DIR if hasattr(settings, 'RESULTS_DIR') else "./data/results"
        os.makedirs(self.results_dir, exist_ok=True)

    def process_compound_job(
        self,
        job_id: str,
        compound_name: str,
        smiles: str,
        similarity_threshold: int = 90,
        activity_types: Optional[List[str]] = None,
        author_name: Optional[str] = None,
    ) -> None:
        """
        Main processing function. Runs in background thread.
        Updates database with progress, syncs to Azure on complete.

        This is the entry point called by the ThreadPoolExecutor.

        Args:
            job_id: Unique job identifier
            compound_name: Name of the compound
            smiles: SMILES string
            similarity_threshold: Similarity threshold (50-100)
            activity_types: List of activity types to fetch
            author_name: Name of the author who submitted the analysis
        """
        import uuid

        # Generate unique entry_id for this compound result (used for UUID-based storage)
        entry_id = str(uuid.uuid4())
        logger.info(f"Generated entry_id {entry_id} for job {job_id}")

        # Use context manager for proper resource management
        # This ensures db.close() is always called, even with early returns
        with get_db_session() as db:
            try:
                # Start processing
                self._update_progress(db, job_id, 0, "Starting...", JobStatus.PROCESSING)

                # Step 1: Search for similar compounds (20%)
                self._update_progress(db, job_id, 5, "Searching ChEMBL for similar compounds...")
                chembl_ids = self._search_similar_compounds(smiles, similarity_threshold)

                # Build similarity score lookup before any filtering
                similarity_scores = {
                    d.get('ChEMBL ID'): d.get('Similarity', 0)
                    for d in chembl_ids if d.get('ChEMBL ID')
                }
                all_similar_chembl_ids = list(similarity_scores.keys())

                if not chembl_ids:
                    # Probe lower thresholds so the user knows where data exists
                    self._update_progress(db, job_id, 10, "Searching lower thresholds...")
                    cascade = cascade_similarity_counts(smiles, similarity_threshold)
                    error_msg = "No similar compounds found in ChEMBL"
                    if cascade:
                        error_msg += f" at {similarity_threshold}% threshold"
                    self._fail_job(db, job_id, error_msg, cascade_results=cascade)
                    return

                # Check for cancellation after ChEMBL search
                if self._is_job_cancelled(db, job_id):
                    return

                self._update_progress(db, job_id, 20, f"Found {len(chembl_ids)} similar compounds")

                # Step 2: Fetch activities (40%)
                self._update_progress(db, job_id, 25, "Fetching bioactivity data...")
                all_results = self._fetch_activities(
                    chembl_ids,
                    activity_types,
                    lambda pct, msg: self._update_progress(db, job_id, 25 + int(pct * 0.15), msg)
                )
                self._update_progress(db, job_id, 40, f"Retrieved {len(all_results)} bioactivity records")

                if not all_results:
                    # Probe lower thresholds — structural matches exist but lack activity data
                    self._update_progress(db, job_id, 35, "Searching lower thresholds...")
                    cascade = cascade_similarity_counts(smiles, similarity_threshold)
                    error_msg = "No bioactivity data found"
                    if cascade:
                        error_msg += f" at {similarity_threshold}% threshold"
                    self._fail_job(db, job_id, error_msg, cascade_results=cascade)
                    return

                # Check for cancellation after fetching activities (long operation)
                if self._is_job_cancelled(db, job_id):
                    return

                # Step 3: Process and calculate metrics (60%)
                self._update_progress(db, job_id, 42, "Processing compounds & calculating metrics...")
                df_results = pd.DataFrame(all_results)
                df_results.replace("No data", np.nan, inplace=True)

                # Map similarity scores onto results
                if 'ChEMBL_ID' in df_results.columns:
                    df_results['Similarity'] = df_results['ChEMBL_ID'].map(similarity_scores).fillna(0)

                # Calculate molecular descriptors (QED, NPOL, Heavy_Atoms) from SMILES
                self._update_progress(db, job_id, 44, "Calculating molecular descriptors...")
                df_results = self._calculate_molecular_descriptors(
                    df_results,
                    lambda pct, msg: self._update_progress(db, job_id, 44 + int(pct * 4), msg)
                )
                self._update_progress(db, job_id, 48, "Molecular descriptors complete")

                # Check for cancellation after molecular descriptors
                if self._is_job_cancelled(db, job_id):
                    return

                # Add PAINS and assay interference flags
                self._update_progress(db, job_id, 49, "Running PAINS and assay interference analysis...")
                df_results = self._add_assay_interference_flags(df_results)
                self._update_progress(db, job_id, 50, "PAINS analysis complete")

                # Calculate advanced metrics
                self._update_progress(db, job_id, 51, "Calculating efficiency metrics...")
                df_results = self._calculate_advanced_metrics(
                    df_results,
                    lambda pct, msg: self._update_progress(db, job_id, 51 + int(pct * 0.14), msg)
                )
                self._update_progress(db, job_id, 65, "Efficiency metrics complete")

                # Check for cancellation after efficiency metrics
                if self._is_job_cancelled(db, job_id):
                    return

                # Step 4: IMP scoring + PDB (75%)
                self._update_progress(db, job_id, 68, "Querying PDB & calculating IMP scores...")
                df_results, pdb_unavailable = self._calculate_imp_scores(
                    df_results,
                    use_pdb=True,
                    progress_callback=lambda pct, msg: self._update_progress(db, job_id, 68 + int(pct * 7), f"PDB: {msg}")
                )
                self._update_progress(db, job_id, 75, "IMP + PDB scoring complete")

                # Check for cancellation after IMP/PDB scoring (long operation)
                if self._is_job_cancelled(db, job_id):
                    return

                # Step 5: IMP classification (80%)
                self._update_progress(db, job_id, 78, "Classifying IMP candidates...")
                df_results = self._classify_imps(df_results)
                self._update_progress(db, job_id, 80, "IMP classification complete")

                # Step 6: Add chemical classification (82%)
                self._update_progress(db, job_id, 81, "Getting chemical classifications...")
                df_results = self._add_chemical_classification(df_results)
                self._update_progress(db, job_id, 84, "Chemical classification complete")

                # Step 6.5: Build all similar molecules catalog
                self._update_progress(db, job_id, 84, "Building similar molecules catalog...")
                try:
                    all_similar_df = self._build_all_similar_df(
                        all_similar_chembl_ids, similarity_scores, df_results,
                        lambda pct, msg: self._update_progress(db, job_id, 84 + int(pct * 3), msg)
                    )
                except Exception as e:
                    logger.warning(f"Failed to build all similar molecules catalog: {e}")
                    all_similar_df = pd.DataFrame()

                # Step 6.6: Fetch drug indications (separate data, not merged with main df)
                self._update_progress(db, job_id, 87, "Fetching drug indications...")
                indications_df = self._fetch_drug_indications(
                    df_results,
                    lambda pct, msg: self._update_progress(db, job_id, 87 + int(pct * 2), msg)
                )
                self._update_progress(db, job_id, 89, f"Drug indications complete ({len(indications_df)} found)")

                # Step 7: Save results (90%)
                self._update_progress(db, job_id, 89, "Saving results...")
                result_path, result_summary = self._save_results(
                    compound_name, smiles, similarity_threshold, activity_types, df_results, indications_df,
                    all_similar_df=all_similar_df,
                    entry_id=entry_id,
                    author_name=author_name,
                )
                # Add entry_id to result_summary for database
                result_summary['entry_id'] = entry_id
                # Flag PDB unavailability in result_summary (STAB-15)
                if pdb_unavailable:
                    result_summary['pdb_unavailable'] = True
                self._update_progress(db, job_id, 90, "Results saved")

                # Explicit DataFrame cleanup to reduce memory pressure
                del df_results
                del indications_df
                del all_similar_df

                # Step 8: Commit DB to COMPLETED first (atomic-first ordering, STAB-10)
                # Store anticipated storage_path so compound entry has it before Azure upload
                result_summary['storage_path'] = get_storage_path_from_entry_id(entry_id)
                self._update_progress(db, job_id, 92, "Finalizing job...")
                self._complete_job(db, job_id, result_path, result_summary)
                logger.info(f"Job {job_id} DB committed as COMPLETED")

                # Step 9: Upload to Azure with retry (95%)
                self._update_progress(db, job_id, 95, "Uploading to Azure...")
                if is_azure_configured():
                    from backend.core.azure_sync import (
                        write_pending_marker, delete_pending_marker,
                        _upload_with_retry,
                    )
                    from backend.core.metrics import metrics as _metrics
                    from backend.core.exceptions import ErrorCode

                    try:
                        write_pending_marker(entry_id)
                        _upload_with_retry(result_path, entry_id)
                        delete_pending_marker(entry_id)
                        self._update_progress(db, job_id, 100, "Complete")
                    except Exception as azure_error:
                        logger.error(f"Azure upload failed permanently for job {job_id}: {azure_error}")
                        _metrics.increment('azure_upload_failed_permanently')
                        # Transition to SYNC_PENDING
                        from backend.services.job_service import _db_write_lock
                        with _db_write_lock:
                            job = db.query(Job).filter(Job.id == job_id).first()
                            if job and job.status == JobStatus.COMPLETED:
                                job.status = JobStatus.SYNC_PENDING
                                job.error_message = f"Azure upload failed: {azure_error}"
                                job.error_code = str(ErrorCode.SYNC_FAILED)
                                db.commit()
                                logger.warning(f"Job {job_id} moved to SYNC_PENDING")
                else:
                    self._update_progress(db, job_id, 100, "Complete")
                logger.info(f"Job {job_id} completed successfully")

            except (ConnectionError, TimeoutError) as e:
                # Network-related errors - log and fail gracefully
                error_msg = f"Network error: {type(e).__name__}: {e}"
                logger.error(f"Job {job_id} failed: {error_msg}")
                self._fail_job(db, job_id, error_msg)
            except ValueError as e:
                # Data validation errors
                error_msg = f"Data validation error: {e}"
                logger.error(f"Job {job_id} failed: {error_msg}")
                self._fail_job(db, job_id, error_msg)
            except Exception as e:
                # Unexpected errors - log full traceback
                logger.exception(f"Job {job_id} failed with unexpected error: {type(e).__name__}: {e}")
                self._fail_job(db, job_id, f"Unexpected error: {type(e).__name__}: {e}")

    def _update_progress(
        self,
        db,
        job_id: str,
        progress: float,
        current_step: str,
        status: JobStatus = JobStatus.PROCESSING
    ) -> None:
        """Update job progress in database."""
        from backend.services.job_service import job_service
        job_service.update_progress(db, job_id, progress, current_step, status)
        logger.debug(f"Job {job_id}: {progress}% - {current_step}")

    def _complete_job(
        self,
        db,
        job_id: str,
        result_path: str,
        result_summary: Dict
    ) -> None:
        """Mark job as completed."""
        from backend.services.job_service import job_service
        job_service.complete_job(db, job_id, result_path, result_summary)

    def _fail_job(self, db, job_id: str, error_message: str, cascade_results=None) -> None:
        """Mark job as failed, optionally storing cascade similarity data."""
        from backend.services.job_service import job_service
        job_service.fail_job(db, job_id, error_message, cascade_results=cascade_results)

    def _is_job_cancelled(self, db, job_id: str) -> bool:
        """Check if job has been cancelled by user.

        Returns True if job status is CANCELLED, allowing graceful early exit.
        """
        from backend.services.job_service import job_service
        job = job_service.get_job(db, job_id)
        if job and job.status == JobStatus.CANCELLED:
            logger.info(f"Job {job_id} was cancelled, stopping processing")
            return True
        return False

    def _search_similar_compounds(
        self,
        smiles: str,
        similarity_threshold: int
    ) -> List[Dict[str, str]]:
        """
        Search for similar compounds in ChEMBL.

        Args:
            smiles: Query SMILES string
            similarity_threshold: Similarity threshold (50-100)

        Returns:
            List of ChEMBL IDs
        """
        try:
            return get_chembl_ids(smiles, similarity_threshold)
        except (IndexError, ConnectionError, TimeoutError) as e:
            # Known recoverable errors - try fallback
            logger.warning(f"Similarity search failed (recoverable): {e}")
            return self._search_similar_compounds_fallback(smiles, similarity_threshold)
        except Exception as e:
            # Unexpected error - log fully and try fallback
            logger.warning(f"Similarity search failed (unexpected): {type(e).__name__}: {e}")
            return self._search_similar_compounds_fallback(smiles, similarity_threshold)

    def _search_similar_compounds_fallback(
        self,
        smiles: str,
        similarity_threshold: int,
        max_retries: int = 2
    ) -> List[Dict[str, str]]:
        """Fallback similarity search using chembl_webresource_client directly.

        Includes retry logic for ChEMBL API intermittent failures.
        """
        from chembl_webresource_client.new_client import new_client

        last_error = None
        for attempt in range(max_retries):
            try:
                similarity = new_client.similarity

                results = similarity.filter(
                    smiles=smiles,
                    similarity=similarity_threshold
                ).only(['molecule_chembl_id', 'similarity'])

                # Explicit list conversion to handle pagination issues
                result_list = list(results)
                return [{"ChEMBL ID": r['molecule_chembl_id'], "Similarity": float(r.get('similarity', 0))} for r in result_list]

            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check for ChEMBL data corruption (empty attribute errors during pagination)
                is_corruption_error = "empty attribute" in error_str or "doesn't allow a default" in error_str

                if attempt < max_retries - 1:
                    if is_corruption_error:
                        logger.warning(f"Fallback similarity search API data corruption (attempt {attempt + 1}), retrying...")
                    else:
                        logger.warning(f"Fallback similarity search attempt {attempt + 1} failed: {e}")
                    import time
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"Fallback similarity search failed after {max_retries} attempts: {last_error}")

        return []

    def _fetch_activities(
        self,
        chembl_ids: List[Dict[str, str]],
        activity_types: Optional[List[str]],
        progress_callback: Callable[[float, str], None]
    ) -> List[Dict]:
        """
        Fetch bioactivity data using OPTIMIZED single-batch approach.

        BEFORE: Loop through each compound, fetch individually (slow)
        NOW: Single query for all compounds, filter locally (fast)

        Args:
            chembl_ids: List of ChEMBL IDs (as dicts with 'ChEMBL ID' key)
            activity_types: Activity types to fetch
            progress_callback: Callback for progress updates

        Returns:
            List of processed compound data
        """
        if activity_types is None:
            activity_types = ['IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'GI50', 'MIC']

        # Extract ChEMBL IDs from dict format
        all_chembl_ids = [
            d.get('ChEMBL ID') for d in chembl_ids
            if d.get('ChEMBL ID')
        ]

        if not all_chembl_ids:
            return []

        progress_callback(0.1, f"Fetching activities for {len(all_chembl_ids)} compounds (single batch)...")

        # OPTIMIZED: Fetch ALL activities in one query
        from backend.modules.api_client import fetch_all_activities_single_batch

        raw_activities = fetch_all_activities_single_batch(
            all_chembl_ids,
            activity_types=activity_types,
            progress_callback=lambda pct, msg: progress_callback(0.1 + pct * 0.4, msg)
        )

        if not raw_activities:
            progress_callback(1.0, "No activities found")
            return []

        progress_callback(0.5, f"Processing {len(raw_activities)} activities...")

        # Build molecule data cache for all unique IDs found in activities
        # OPTIMIZED: Use batch fetching for molecules and targets
        unique_ids = list(set(a.get('molecule_chembl_id') for a in raw_activities if a.get('molecule_chembl_id')))
        unique_target_ids = list(set(a.get('target_chembl_id') for a in raw_activities if a.get('target_chembl_id')))

        # Batch fetch molecule data (3-5x faster than individual calls)
        from backend.modules.api_client import fetch_batch_molecule_data, fetch_batch_target_names

        progress_callback(0.55, f"Fetching molecule data for {len(unique_ids)} compounds (batch)...")
        mol_cache = fetch_batch_molecule_data(
            unique_ids,
            progress_callback=lambda pct, msg: progress_callback(0.55 + pct * 0.15, msg)
        )

        # Batch fetch target names (3-5x faster than individual calls)
        progress_callback(0.7, f"Fetching target names for {len(unique_target_ids)} targets (batch)...")
        target_name_cache = fetch_batch_target_names(
            unique_target_ids,
            progress_callback=lambda pct, msg: progress_callback(0.7 + pct * 0.1, msg)
        )

        progress_callback(0.8, "Building result records...")

        # Process activities into final format
        all_results = []
        for act in raw_activities:
            chembl_id = act.get('molecule_chembl_id')
            mol_data = mol_cache.get(chembl_id)

            if not mol_data:
                continue

            mol_props = mol_data.get('molecule_properties', {}) or {}
            mol_structures = mol_data.get('molecule_structures', {}) or {}
            smiles = mol_structures.get('canonical_smiles', '')
            mol_name = mol_data.get('pref_name') or 'Unknown'

            std_value = act.get('standard_value')
            std_units = act.get('standard_units')

            if not std_value:
                continue

            try:
                value = float(std_value)
                if value <= 0:
                    continue

                # Convert to nM
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

                # Get target name from batch cache
                target_chembl_id = act.get('target_chembl_id', '')
                target_name = target_name_cache.get(target_chembl_id, '')

                all_results.append({
                    'ChEMBL_ID': chembl_id,
                    'Molecule_Name': mol_name,
                    'SMILES': smiles,
                    'Molecular_Weight': float(mol_props.get('full_mwt') or 0) or np.nan,
                    'TPSA': float(mol_props.get('psa') or 0) or np.nan,
                    'Activity_Type': act.get('standard_type', ''),
                    'Activity_nM': value_nM,
                    'pActivity': pActivity,
                    'Target_ChEMBL_ID': target_chembl_id,
                    'Target_Name': target_name,
                })
            except (ValueError, TypeError):
                continue

        progress_callback(1.0, f"Processed {len(all_results)} activity records")
        logger.info(f"Fetched target names for {len(target_name_cache)} unique targets")
        return all_results

    def _add_assay_interference_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add assay interference flags to the DataFrame.

        Uses the assay_interference_filter module to detect:
        - PAINS (Pan-Assay Interference Substructures)
        - Aggregator risk
        - Redox reactivity
        - Fluorescence interference
        - Thiol reactivity
        - BRENK alerts (unwanted substructures)
        - NIH alerts (problematic functional groups)

        Each flag also has a _Details column with matched pattern names.

        Args:
            df: DataFrame with SMILES column

        Returns:
            DataFrame with interference flag columns added
        """
        if 'SMILES' not in df.columns:
            logger.warning("SMILES column not found, skipping interference analysis")
            return df

        df = df.copy()

        # Boolean flag columns (mapped from InterferenceFlags fields to frontend names)
        bool_mapping = {
            'pains': 'PAINS_Violation',
            'aggregator': 'Aggregator_Risk',
            'redox': 'Redox_Reactive',
            'fluorescence': 'Fluorescence_Interference',
            'thiol': 'Thiol_Reactive',
            'brenk': 'BRENK_Alerts',
            'nih': 'NIH_Alerts',
        }

        # Detail columns (matched pattern names)
        detail_mapping = {
            'pains_details': 'PAINS_Details',
            'aggregator_reason': 'Aggregator_Details',
            'redox_details': 'Redox_Details',
            'fluorescence_details': 'Fluorescence_Details',
            'thiol_details': 'Thiol_Details',
            'brenk_details': 'BRENK_Details',
            'nih_details': 'NIH_Details',
        }

        # Initialize all columns
        for col in bool_mapping.values():
            df[col] = False
        for col in detail_mapping.values():
            df[col] = ''

        # Get unique SMILES to avoid redundant processing
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

        # Apply boolean flags
        for field_name, col_name in bool_mapping.items():
            df[col_name] = df['SMILES'].apply(
                lambda s, fn=field_name: getattr(flags_cache.get(s, InterferenceFlags()), fn, False) if pd.notna(s) else False
            )

        # Apply detail columns
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

    def _calculate_molecular_descriptors(
        self,
        df: pd.DataFrame,
        progress_callback: Callable[[float, str], None]
    ) -> pd.DataFrame:
        """
        Calculate molecular descriptors from SMILES using RDKit.

        Adds: Heavy_Atoms, NPOL, QED, Aromatic_Rings, RO5_Violations, HBD, HBA, LogP,
              Rotatable_Bonds, and other missing columns.

        Args:
            df: DataFrame with SMILES column
            progress_callback: Callback for progress updates

        Returns:
            DataFrame with molecular descriptor columns added
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED as QEDModule, rdMolDescriptors
        except ImportError:
            logger.warning("RDKit not available - skipping molecular descriptor calculation")
            progress_callback(1.0, "Skipped descriptors (RDKit not available)")
            return df

        # Try to import NP Likeness scorer (multiple import paths for different RDKit versions)
        np_scorer = None
        try:
            from rdkit.Chem import RDConfig
            import os
            from rdkit.Contrib.NP_Score import npscorer
            fscore_data = os.path.join(RDConfig.RDContribDir, 'NP_Score', 'publicnp.model.gz')
            np_scorer_obj = npscorer.readNPModel(fscore_data)

            def np_scorer(mol):
                return npscorer.scoreMol(mol, np_scorer_obj)

            logger.info("NP Likeness scorer loaded from Contrib")
        except Exception as e1:
            try:
                # Try alternate import for newer RDKit
                from rdkit.Chem.Descriptors import CalcNPScore
                np_scorer = CalcNPScore
                logger.info("NP Likeness scorer loaded from Descriptors")
            except Exception as e2:
                logger.debug(f"NP Likeness scorer not available: {e1}, {e2}")

        df = df.copy()
        progress_callback(0.1, "Calculating molecular descriptors...")

        # Initialize columns if not present
        descriptor_cols = [
            'Heavy_Atoms', 'NPOL', 'QED', 'TPSA',
            'Aromatic_Rings', 'Rotatable_Bonds',
            'HBD', 'HBA', 'LogP',
            'RO5_Violations', 'NP_Likeness_Score',
            'PSAoMW', '10xPSA_MW', 'NPOLoNHA'  # Derived ratios for efficiency analysis
        ]
        for col in descriptor_cols:
            if col not in df.columns:
                df[col] = np.nan

        # OPTIMIZED: Pre-compute descriptors for unique SMILES using cache
        # This avoids redundant calculations for identical compounds
        unique_smiles = df['SMILES'].dropna().unique()
        descriptor_cache = {}  # smiles -> dict of descriptors

        def calculate_descriptors_for_smiles(smiles_str: str) -> dict:
            """Calculate all descriptors for a single SMILES."""
            result = {col: np.nan for col in descriptor_cols}

            try:
                mol = Chem.MolFromSmiles(str(smiles_str))
                if mol is None:
                    return result

                result['Heavy_Atoms'] = Descriptors.HeavyAtomCount(mol)

                # NPOL = number of N + O atoms
                n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
                o_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
                result['NPOL'] = n_count + o_count

                result['QED'] = QEDModule.qed(mol)
                result['TPSA'] = Descriptors.TPSA(mol)
                result['Aromatic_Rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
                result['Rotatable_Bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                result['HBD'] = rdMolDescriptors.CalcNumHBD(mol)
                result['HBA'] = rdMolDescriptors.CalcNumHBA(mol)
                result['LogP'] = Descriptors.MolLogP(mol)

                # Calculate RO5 Violations
                mw = Descriptors.MolWt(mol)
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

                # Calculate derived ratios for efficiency analysis
                # PSAoMW = PSA / MW (polarity relative to size)
                tpsa_val = result['TPSA']
                if mw > 0 and tpsa_val is not None and not np.isnan(tpsa_val):
                    result['PSAoMW'] = tpsa_val / mw
                    result['10xPSA_MW'] = 10 * result['PSAoMW']  # Scaled version

                # NPOLoNHA = NPOL / Heavy_Atoms (polar atom fraction)
                heavy_atoms = result['Heavy_Atoms']
                npol = result['NPOL']
                if heavy_atoms is not None and not np.isnan(heavy_atoms) and heavy_atoms > 0:
                    if npol is not None and not np.isnan(npol):
                        result['NPOLoNHA'] = npol / heavy_atoms

                # NP Likeness Score
                if np_scorer is not None:
                    try:
                        result['NP_Likeness_Score'] = np_scorer(mol)
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(f"Error calculating descriptors for {smiles_str[:30]}...: {e}")

            return result

        # Pre-compute descriptors for all unique SMILES
        total_unique = len(unique_smiles)
        logger.info(f"Computing descriptors for {total_unique} unique SMILES...")

        for i, smiles in enumerate(unique_smiles):
            if not smiles or smiles == 'nan':
                continue
            descriptor_cache[smiles] = calculate_descriptors_for_smiles(smiles)

            # Update progress
            if (i + 1) % max(1, total_unique // 10) == 0 or i == total_unique - 1:
                pct = (i + 1) / total_unique
                progress_callback(pct * 0.8, f"Computed descriptors for {i + 1}/{total_unique} unique compounds")

        # Apply cached results to DataFrame using vectorized operations
        progress_callback(0.85, "Applying descriptors to dataframe...")

        for col in descriptor_cols:
            # Only update where current value is NaN
            mask = df[col].isna()
            values = df.loc[mask, 'SMILES'].apply(
                lambda s: descriptor_cache.get(s, {}).get(col, np.nan) if pd.notna(s) else np.nan
            )
            # Coerce to numeric — descriptor cache can return non-scalar values
            # (e.g. empty lists) which would corrupt the column dtype to object
            df.loc[mask, col] = pd.to_numeric(values, errors='coerce')

        progress_callback(1.0, "Molecular descriptors complete")
        return df

    def _calculate_advanced_metrics(
        self,
        df: pd.DataFrame,
        progress_callback: Callable[[float, str], None]
    ) -> pd.DataFrame:
        """
        Calculate advanced efficiency metrics.

        Args:
            df: DataFrame with basic data
            progress_callback: Callback for progress updates

        Returns:
            DataFrame with efficiency metrics added
        """
        try:
            progress_callback(0.2, "Calculating efficiency metrics...")

            # Initialize ALL required columns upfront to avoid "missing columns" errors
            # This ensures IMP scoring can check column existence even if values are NaN
            required_columns = [
                'SEI', 'BEI', 'NSEI', 'NBEI', 'nBEI_viz',  # Efficiency metrics
                'Modulus_SEI_BEI', 'Angle_SEI_BEI', 'Slope_SEI_BEI',  # SEI-BEI plane
                'Modulus_NSEI_NBEI', 'Angle_NSEI_NBEI', 'Slope_NSEI_NBEI', 'Intercept_NSEI_NBEI',  # NSEI-NBEI plane
                'QED'  # Drug-likeness (may be calculated earlier, but ensure it exists)
            ]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = np.nan

            # Calculate efficiency metrics using vectorized operations
            metrics_input_cols = ['pActivity', 'TPSA', 'Molecular_Weight', 'NPOL', 'Heavy_Atoms']
            if all(col in df.columns for col in metrics_input_cols):
                df = calculate_efficiency_metrics_dataframe(df)
            else:
                missing = [c for c in metrics_input_cols if c not in df.columns]
                logger.warning(f"Skipping efficiency metrics: missing columns {missing}")

            progress_callback(0.5, "Calculating plane geometry...")

            # Calculate plane metrics using vectorized operations
            plane_input_cols = ['SEI', 'BEI', 'NSEI', 'NBEI', 'TPSA', 'Molecular_Weight', 'NPOL', 'Heavy_Atoms']
            if all(col in df.columns for col in plane_input_cols):
                df = calculate_plane_metrics_dataframe(df)
            else:
                missing = [c for c in plane_input_cols if c not in df.columns]
                logger.warning(f"Skipping plane metrics: missing columns {missing}")

            progress_callback(0.8, "Detecting outliers...")

            # Detect outliers
            df = detect_efficiency_outliers(df, metrics=['SEI', 'BEI', 'NSEI', 'NBEI'])

            progress_callback(1.0, "Advanced metrics complete")
            return df

        except Exception as e:
            logger.warning(f"Could not calculate efficiency metrics: {e}")
            progress_callback(1.0, "Skipped advanced metrics (error occurred)")
            return df

    def _calculate_imp_scores(
        self,
        df: pd.DataFrame,
        use_pdb: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple:
        """
        Calculate IMP scores with optional PDB integration.

        Args:
            df: DataFrame with efficiency metrics
            use_pdb: If True, query PDB for structural evidence (default True)
            progress_callback: Optional callback for PDB progress updates

        Returns:
            Tuple of (DataFrame with IMP scores added, pdb_unavailable flag)
        """
        pdb_unavailable = False
        try:
            df = calculate_imp_score(df, use_pdb=use_pdb, progress_callback=progress_callback)
            df = add_imp_score_interpretation(df)
            # Check if PDB data was actually retrieved (STAB-15)
            if use_pdb and 'PDB_Score' in df.columns:
                # If all PDB scores are 0 and we expected PDB data, it may indicate API failure
                # but this is not necessarily "unavailable" -- could be no structures found
                pass
            return df, pdb_unavailable

        except Exception as e:
            logger.warning(f"IMP scoring failed: {e}")
            pdb_unavailable = use_pdb  # If PDB was requested and scoring failed, flag it
            return df, pdb_unavailable

    def _add_chemical_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add chemical classification from ClassyFire and NPClassifier.

        Args:
            df: DataFrame with SMILES column

        Returns:
            DataFrame with classification columns added
        """
        if 'SMILES' not in df.columns:
            logger.warning("SMILES column not found, skipping classification")
            return df

        df = df.copy()

        # Initialize classification columns if not present
        classification_cols = [
            'Kingdom', 'Superclass', 'Class', 'Subclass', 'Direct_Parent',
            'Molecular_Framework', 'Description', 'ChEMONT_ID_Class', 'ChEMONT_ID_Subclass',
            'NP_Pathway', 'NP_Superclass', 'NP_Class', 'NP_isglycoside'
        ]
        for col in classification_cols:
            if col not in df.columns:
                df[col] = ''

        # Get unique SMILES to avoid redundant API calls
        unique_smiles = df['SMILES'].dropna().unique()
        classification_cache = {}

        logger.info(f"Getting chemical classifications for {len(unique_smiles)} unique compounds...")

        for i, smiles in enumerate(unique_smiles):
            try:
                # Get InChIKey for ClassyFire
                from rdkit import Chem
                from rdkit.Chem.inchi import MolToInchiKey

                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    inchikey = MolToInchiKey(mol)
                    if inchikey:
                        classification = get_complete_classification(smiles, inchikey)
                        classification_cache[smiles] = classification
                    else:
                        classification_cache[smiles] = {}
                else:
                    classification_cache[smiles] = {}

                if (i + 1) % 10 == 0:
                    logger.info(f"Classified {i + 1}/{len(unique_smiles)} compounds")

            except Exception as e:
                logger.warning(f"Classification failed for SMILES {smiles[:30]}...: {e}")
                classification_cache[smiles] = {}

        # Apply classifications to DataFrame
        for col in classification_cols:
            df[col] = df['SMILES'].apply(
                lambda s: classification_cache.get(s, {}).get(col, '') if pd.notna(s) else ''
            )

        logger.info(f"Chemical classification complete for {len(unique_smiles)} compounds")
        return df

    def _classify_imps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify IMP candidates.

        Args:
            df: DataFrame with IMP scores

        Returns:
            DataFrame with IMP classification added
        """
        try:
            return classify_imp_candidates(df, min_outlier_count=2, use_imp_score=True)
        except Exception as e:
            logger.warning(f"IMP classification failed: {e}")
            return df

    def _fetch_drug_indications(
        self,
        df: pd.DataFrame,
        progress_callback: Callable[[float, str], None]
    ) -> pd.DataFrame:
        """
        Fetch drug indications for all unique ChEMBL IDs using batch API.

        Uses REST API batch query which is 13.7x faster than sequential calls
        (0.59s vs 8.10s for 9 compounds in benchmarks).

        Returns a separate DataFrame with indication data including:
        - MESH ID/Heading (disease identifiers)
        - EFO ID/Term (ontology identifiers)
        - Max Phase (clinical trial phase)
        - Clinical Trials URLs

        Args:
            df: DataFrame with ChEMBL_ID column
            progress_callback: Callback for progress updates

        Returns:
            DataFrame with drug indication data (separate from main df)
        """
        if 'ChEMBL_ID' not in df.columns:
            logger.warning("ChEMBL_ID column not found, skipping drug indications")
            return pd.DataFrame()

        unique_ids = list(df['ChEMBL_ID'].dropna().unique())
        total = len(unique_ids)

        if total == 0:
            logger.info("No ChEMBL IDs to fetch indications for")
            return pd.DataFrame()

        logger.info(f"Fetching drug indications for {total} unique compounds (batch): {unique_ids[:5]}...")
        progress_callback(0.0, f"Fetching drug indications for {total} compounds (batch)...")

        # Use batch function (13.7x faster than sequential)
        try:
            indications_by_id = get_drug_indications_batch(unique_ids, progress_callback)

            # Flatten results into a list
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

    def _build_all_similar_df(
        self,
        all_chembl_ids: List[str],
        similarity_scores: Dict[str, float],
        df_results: pd.DataFrame,
        progress_callback: Callable[[float, str], None],
    ) -> pd.DataFrame:
        """Build a DataFrame of ALL similar compounds (including those without bioactivity).

        Reuses data from df_results for compounds that already have bioactivity
        (avoids duplicate API calls, descriptor calculations, and interference analysis).
        Only fetches/processes the delta for compounds WITHOUT bioactivity.

        Args:
            all_chembl_ids: All ChEMBL IDs from similarity search
            similarity_scores: Mapping of ChEMBL ID to similarity score
            df_results: Main results DataFrame (compounds WITH bioactivity)
            progress_callback: Progress callback (0.0-1.0, message)

        Returns:
            DataFrame with all similar compounds and their properties
        """
        from backend.modules.api_client import fetch_batch_molecule_data

        if not all_chembl_ids:
            return pd.DataFrame()

        # Identify which compounds already have data in df_results (avoid re-processing)
        compounds_with_data = set()
        if df_results is not None and 'ChEMBL_ID' in df_results.columns:
            compounds_with_data = set(df_results['ChEMBL_ID'].dropna().unique())
        new_chembl_ids = [cid for cid in all_chembl_ids if cid not in compounds_with_data]

        # Extract rows for already-processed compounds from df_results
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

        # Fetch molecule data only for NEW compounds (not in df_results)
        new_rows = pd.DataFrame()
        if new_chembl_ids:
            progress_callback(0.0, f"Fetching molecule data for {len(new_chembl_ids)} new compounds...")
            mol_data = fetch_batch_molecule_data(
                new_chembl_ids,
                progress_callback=lambda pct, msg: progress_callback(pct * 0.3, msg),
            )

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

                # Only run descriptors/interference/classification on NEW compounds
                progress_callback(0.3, "Calculating molecular descriptors...")
                new_rows = self._calculate_molecular_descriptors(
                    new_rows,
                    lambda pct, msg: progress_callback(0.3 + pct * 0.2, msg)
                )

                progress_callback(0.5, "Running assay interference analysis...")
                new_rows = self._add_assay_interference_flags(new_rows)

                progress_callback(0.7, "Getting chemical classifications...")
                new_rows = self._add_chemical_classification(new_rows)
        else:
            progress_callback(0.7, "All compounds already processed")

        # Combine existing + new
        dfs_to_concat = [d for d in [existing_rows, new_rows] if not d.empty]
        if not dfs_to_concat:
            return pd.DataFrame()

        df = pd.concat(dfs_to_concat, ignore_index=True)
        df = df.sort_values('Similarity', ascending=False).reset_index(drop=True)
        progress_callback(1.0, f"All similar molecules catalog complete ({len(df)} compounds)")
        return df

    def _save_results(
        self,
        compound_name: str,
        smiles: str,
        similarity_threshold: int,
        activity_types: Optional[List[str]],
        df_results: pd.DataFrame,
        indications_df: Optional[pd.DataFrame] = None,
        all_similar_df: Optional[pd.DataFrame] = None,
        entry_id: Optional[str] = None,
        author_name: Optional[str] = None,
    ) -> tuple:
        """
        Save results to disk and create ZIP archive.

        Args:
            compound_name: Name of the compound
            smiles: Query SMILES
            similarity_threshold: Similarity threshold used
            activity_types: Activity types processed
            df_results: Results DataFrame
            indications_df: Optional DataFrame with drug indications (separate file)
            all_similar_df: Optional DataFrame with ALL similar compounds (including those without bioactivity)
            entry_id: Optional UUID for the compound entry (used for ZIP filename)
            author_name: Name of the author who submitted the analysis

        Returns:
            Tuple of (zip_path, result_summary)
        """
        # Sanitize compound name for filesystem (consistent across codebase)
        safe_name = sanitize_compound_name(compound_name)
        compound_folder = os.path.join(self.results_dir, safe_name)
        os.makedirs(compound_folder, exist_ok=True)

        try:
            return self._save_results_inner(
                compound_name, smiles, similarity_threshold, activity_types,
                df_results, indications_df, all_similar_df, entry_id, author_name,
                safe_name, compound_folder,
            )
        except Exception:
            # STAB-13: Clean up compound_folder on any failure
            shutil.rmtree(compound_folder, ignore_errors=True)
            logger.warning(f"Cleaned up compound_folder on failure: {compound_folder}")
            raise  # Re-raise so caller handles the error

    def _save_results_inner(
        self,
        compound_name: str,
        smiles: str,
        similarity_threshold: int,
        activity_types: Optional[List[str]],
        df_results: pd.DataFrame,
        indications_df: Optional[pd.DataFrame],
        all_similar_df: Optional[pd.DataFrame],
        entry_id: Optional[str],
        author_name: Optional[str],
        safe_name: str,
        compound_folder: str,
    ) -> tuple:
        """Inner implementation of _save_results (separated for STAB-13 cleanup wrapper)."""
        # Save CSV
        results_filename = os.path.join(compound_folder, f"{safe_name}_complete_results.csv")
        df_results.to_csv(results_filename, index=False)

        # Create metadata
        result_summary = {
            'schema_version': 1,  # Integer, increment on structural changes to ZIP contents
            'compound_name': compound_name,
            'author_name': author_name or 'N/A',
            'query_smiles': smiles,
            'similarity_threshold': similarity_threshold,
            'activity_types': ','.join(activity_types or []),
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

        # Add fields for home page display (matching old UI)
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
            # Count rows where any outlier flag is True
            outlier_mask = df_results[outlier_cols].any(axis=1)
            result_summary['num_outliers'] = int(outlier_mask.sum())
        else:
            result_summary['num_outliers'] = 0

        # QED score (average if available, or from first row)
        if 'QED' in df_results.columns:
            qed_values = df_results['QED'].dropna()
            if len(qed_values) > 0:
                result_summary['qed'] = float(qed_values.mean())
            else:
                result_summary['qed'] = 0.0
        else:
            result_summary['qed'] = 0.0

        # IMP score (max if available) - for Compound table
        # Uses max to match the detail page display (best IMP candidate)
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

                    # Add PDB summary stats to result_summary
                    result_summary['pdb_structures_count'] = len(pdb_summary_df)
            except Exception as e:
                logger.warning(f"Could not create detailed PDB summary: {e}")

        # Save drug indications as separate CSV (not merged with main df)
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

        # Save all similar molecules CSV (includes compounds without bioactivity data)
        if all_similar_df is not None and not all_similar_df.empty:
            try:
                all_similar_csv = os.path.join(compound_folder, "all_similar_molecules.csv")
                all_similar_df.to_csv(all_similar_csv, index=False)
                logger.info(f"Saved {len(all_similar_df)} all similar molecules")
                result_summary['total_similar'] = len(all_similar_df)
            except Exception as e:
                logger.warning(f"Could not save all similar molecules: {e}")

        # Update similar_count to reflect total (not just with-data)
        if result_summary.get('total_similar'):
            result_summary['similar_count'] = result_summary['total_similar']

        # Save metadata and summary after all fields are finalized
        metadata_filename = os.path.join(compound_folder, f"{safe_name}_metadata.json")
        with open(metadata_filename, 'w') as f:
            json.dump(result_summary, f, indent=4)

        summary_filename = os.path.join(compound_folder, "summary.json")
        with open(summary_filename, 'w') as f:
            json.dump(result_summary, f, indent=4)

        # Create ZIP archive - use entry_id for filename if available (UUID-based storage)
        # This enables true duplicate support and avoids issues with special characters
        # Use subfolder structure: results/{prefix}/{uuid}.zip (matches Azure storage)
        if entry_id:
            prefix = entry_id[:2].lower()
            zip_subdir = os.path.join(self.results_dir, prefix)
            os.makedirs(zip_subdir, exist_ok=True)
            zip_filename = f"{entry_id}.zip"
            zip_path = os.path.join(zip_subdir, zip_filename)
        else:
            zip_filename = f"{safe_name}.zip"
            zip_path = os.path.join(self.results_dir, zip_filename)
        # Write ZIP atomically: write to .tmp, then rename to final path
        # This prevents partial ZIPs from existing at the final path if the process crashes
        zip_tmp_path = zip_path + ".tmp"
        try:
            with zipfile.ZipFile(zip_tmp_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(compound_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, compound_folder)
                        zipf.write(file_path, arcname)

            # Atomic move: on POSIX this is atomic; on Windows it replaces atomically
            os.replace(zip_tmp_path, zip_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(zip_tmp_path):
                try:
                    os.unlink(zip_tmp_path)
                except OSError:
                    pass
            raise  # Re-raise so the caller (process_compound) sees the failure

        # Clean up folder - keep only ZIP for space optimization
        # Use retry logic for Windows file locking issues
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
                    # Give up but don't fail the job - ZIP was created successfully
                    logger.warning(f"Could not clean up folder {compound_folder} after {max_retries} attempts: {e}")
            except Exception as e:
                logger.warning(f"Error cleaning up folder {compound_folder}: {e}")
                break

        logger.info(f"Saved results to {zip_path}")
        return zip_path, result_summary


# Singleton instance
compound_service = CompoundService()


def process_compound_job(
    job_id: str,
    compound_name: str,
    smiles: str,
    similarity_threshold: int = 90,
    activity_types: Optional[List[str]] = None,
    author_name: Optional[str] = None,
) -> None:
    """
    Wrapper function for executor.submit().

    This is the function that gets submitted to the ThreadPoolExecutor.
    It delegates to the CompoundService singleton.
    """
    compound_service.process_compound_job(
        job_id=job_id,
        compound_name=compound_name,
        smiles=smiles,
        similarity_threshold=similarity_threshold,
        activity_types=activity_types,
        author_name=author_name,
    )
