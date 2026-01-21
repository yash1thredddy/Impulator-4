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
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

import pandas as pd
import numpy as np

# Ensure project root is in path for module imports (do this ONCE at module level)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import settings
from backend.core.database import SessionLocal, get_db_session
from backend.core.azure_sync import (
    sync_db_to_azure,
    upload_result_to_azure,
    upload_result_to_azure_by_entry_id,
    get_storage_path_from_entry_id,
)
from backend.core import sanitize_compound_name
from backend.models.database import JobStatus, JobType

# Import chemistry modules (clean absolute imports)
from backend.modules.api_client import (
    get_chembl_ids,
    get_molecule_data,
    batch_fetch_activities,
    get_target_name,
    get_drug_indications,
)
from backend.modules.efficiency_metrics import calculate_all_efficiency_metrics
from backend.modules.efficiency_planes import calculate_all_plane_metrics
from backend.modules.outlier_detection import detect_efficiency_outliers
from backend.modules.oqpla_scoring import (
    calculate_oqpla_phase2,
    add_oqpla_interpretation,
    create_detailed_pdb_summary,
)
from backend.modules.imp_classifier import classify_imp_candidates
from backend.modules.assay_interference_filter import get_all_interference_flags
from backend.modules.chemical_classifier import get_complete_classification

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
        """
        from backend.services.job_service import job_service
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

                if not chembl_ids:
                    self._fail_job(db, job_id, "No similar compounds found in ChEMBL")
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
                    self._fail_job(db, job_id, "No bioactivity data found")
                    return

                # Check for cancellation after fetching activities (long operation)
                if self._is_job_cancelled(db, job_id):
                    return

                # Step 3: Process and calculate metrics (60%)
                self._update_progress(db, job_id, 42, "Processing compounds & calculating metrics...")
                df_results = pd.DataFrame(all_results)
                df_results.replace("No data", np.nan, inplace=True)

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

                # Step 4: OQPLA scoring + PDB (75%)
                self._update_progress(db, job_id, 68, "Querying PDB & calculating OQPLA scores...")
                df_results = self._calculate_oqpla_scores(
                    df_results,
                    use_pdb=True,
                    progress_callback=lambda pct, msg: self._update_progress(db, job_id, 68 + int(pct * 7), f"PDB: {msg}")
                )
                self._update_progress(db, job_id, 75, "OQPLA + PDB scoring complete")

                # Check for cancellation after OQPLA/PDB scoring (long operation)
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

                # Step 6.5: Fetch drug indications (separate data, not merged with main df)
                self._update_progress(db, job_id, 85, "Fetching drug indications...")
                indications_df = self._fetch_drug_indications(
                    df_results,
                    lambda pct, msg: self._update_progress(db, job_id, 85 + int(pct * 3), msg)
                )
                self._update_progress(db, job_id, 88, f"Drug indications complete ({len(indications_df)} found)")

                # Step 7: Save results (90%)
                self._update_progress(db, job_id, 89, "Saving results...")
                result_path, result_summary = self._save_results(
                    compound_name, smiles, similarity_threshold, activity_types, df_results, indications_df,
                    entry_id=entry_id  # Pass entry_id for UUID-based storage path
                )
                # Add entry_id to result_summary for database
                result_summary['entry_id'] = entry_id
                self._update_progress(db, job_id, 90, "Results saved")

                # Explicit DataFrame cleanup to reduce memory pressure
                del df_results
                del indications_df

                # Step 8: Upload to Azure using entry_id-based path (95%)
                self._update_progress(db, job_id, 92, "Uploading to Azure...")
                # Use UUID-based storage path for better organization and duplicate support
                try:
                    upload_result_to_azure_by_entry_id(result_path, entry_id)
                    # Store the UUID-based path in result_summary for later retrieval
                    result_summary['storage_path'] = get_storage_path_from_entry_id(entry_id)
                    self._update_progress(db, job_id, 95, "Upload complete")
                except Exception as azure_error:
                    # Log Azure upload failure but don't fail the job
                    logger.error(f"Azure upload failed for job {job_id}: {azure_error}")
                    result_summary['storage_path'] = None
                    result_summary['azure_upload_error'] = str(azure_error)
                    self._update_progress(db, job_id, 95, "Upload failed (results saved locally)")

                # Step 9: Sync database (100%)
                self._update_progress(db, job_id, 97, "Syncing database...")
                try:
                    sync_db_to_azure()
                except Exception as sync_error:
                    logger.error(f"Database sync failed for job {job_id}: {sync_error}")

                # Complete job
                self._complete_job(db, job_id, result_path, result_summary)
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

    def _fail_job(self, db, job_id: str, error_message: str) -> None:
        """Mark job as failed."""
        from backend.services.job_service import job_service
        job_service.fail_job(db, job_id, error_message)

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
                ).only(['molecule_chembl_id'])

                # Explicit list conversion to handle pagination issues
                result_list = list(results)
                return [{"ChEMBL ID": r['molecule_chembl_id']} for r in result_list]

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
            activity_types = ['IC50', 'Ki', 'Kd', 'EC50']

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
        Add PAINS and other assay interference flags to the DataFrame.

        Uses the assay_interference_filter module to detect:
        - PAINS (Pan-Assay Interference Substructures)
        - Aggregator risk
        - Redox reactivity
        - Fluorescence interference
        - Thiol reactivity

        Args:
            df: DataFrame with SMILES column

        Returns:
            DataFrame with interference flag columns added
        """
        if 'SMILES' not in df.columns:
            logger.warning("SMILES column not found, skipping PAINS analysis")
            return df

        df = df.copy()

        # Map from get_all_interference_flags keys to frontend expected column names
        # The frontend expects: PAINS_Violation, Aggregator_Risk, Redox_Reactive, Fluorescence_Interference, Thiol_Reactive
        flag_mapping = {
            'PAINS': 'PAINS_Violation',
            'Aggregator': 'Aggregator_Risk',
            'Redox': 'Redox_Reactive',
            'Fluorescence': 'Fluorescence_Interference',
            'Thiol_Reactive': 'Thiol_Reactive'
        }

        # Initialize interference columns with frontend expected names
        for col in flag_mapping.values():
            df[col] = False

        # Get unique SMILES to avoid redundant processing
        unique_smiles = df['SMILES'].dropna().unique()
        flags_cache = {}

        logger.info(f"Running PAINS analysis for {len(unique_smiles)} unique compounds...")

        for i, smiles in enumerate(unique_smiles):
            try:
                flags = get_all_interference_flags(smiles)
                flags_cache[smiles] = flags
            except Exception as e:
                logger.warning(f"PAINS analysis failed for {smiles[:30]}...: {e}")
                flags_cache[smiles] = {k: False for k in flag_mapping.keys()}

            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(unique_smiles)} compounds for PAINS")

        # Apply flags to DataFrame with mapped column names
        for api_key, col_name in flag_mapping.items():
            df[col_name] = df['SMILES'].apply(
                lambda s: flags_cache.get(s, {}).get(api_key, False) if pd.notna(s) else False
            )

        # Count how many compounds have PAINS
        pains_count = df['PAINS_Violation'].sum()
        logger.info(f"PAINS analysis complete: {pains_count}/{len(df)} records have PAINS flags")

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
            np_scorer = lambda mol: npscorer.scoreMol(mol, np_scorer_obj)
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
            'RO5_Violations', 'NP_Likeness_Score'
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
            df.loc[mask, col] = df.loc[mask, 'SMILES'].apply(
                lambda s: descriptor_cache.get(s, {}).get(col, np.nan) if pd.notna(s) else np.nan
            )

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
            # This ensures OQPLA scoring can check column existence even if values are NaN
            required_columns = [
                'SEI', 'BEI', 'NSEI', 'NBEI', 'nBEI_viz',  # Efficiency metrics
                'Modulus_SEI_BEI', 'Angle_SEI_BEI', 'Slope_SEI_BEI',  # SEI-BEI plane
                'Modulus_NSEI_NBEI', 'Angle_NSEI_NBEI', 'Slope_NSEI_NBEI', 'Intercept_NSEI_NBEI',  # NSEI-NBEI plane
                'QED'  # Drug-likeness (may be calculated earlier, but ensure it exists)
            ]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = np.nan

            # Calculate efficiency metrics if not already present
            if df['SEI'].isna().all():
                for idx, row in df.iterrows():
                    if pd.notna(row.get('pActivity')) and pd.notna(row.get('TPSA')):
                        metrics = calculate_all_efficiency_metrics(
                            pActivity=float(row['pActivity']),
                            psa=float(row.get('TPSA', 0)),
                            molecular_weight=float(row.get('Molecular_Weight', 0)),
                            npol=float(row.get('NPOL', 0)) if pd.notna(row.get('NPOL')) else 0,
                            heavy_atoms=float(row.get('Heavy_Atoms', 0)) if pd.notna(row.get('Heavy_Atoms')) else 0
                        )
                        for key, value in metrics.items():
                            df.at[idx, key] = value

            progress_callback(0.5, "Calculating plane geometry...")

            # Calculate plane metrics
            for idx, row in df.iterrows():
                if all(pd.notna(row.get(m)) for m in ['SEI', 'BEI', 'NSEI', 'NBEI']):
                    plane_metrics = calculate_all_plane_metrics(
                        sei=float(row['SEI']),
                        bei=float(row['BEI']),
                        nsei=float(row['NSEI']),
                        nbei=float(row['NBEI']),
                        psa=float(row.get('TPSA', 0)),
                        molecular_weight=float(row.get('Molecular_Weight', 0)),
                        npol=float(row.get('NPOL', 0)) if pd.notna(row.get('NPOL')) else 0,
                        heavy_atoms=float(row.get('Heavy_Atoms', 0)) if pd.notna(row.get('Heavy_Atoms')) else 0
                    )
                    for key, value in plane_metrics.items():
                        if key not in df.columns:
                            df[key] = np.nan
                        df.at[idx, key] = value

            progress_callback(0.8, "Detecting outliers...")

            # Detect outliers
            df = detect_efficiency_outliers(df, metrics=['SEI', 'BEI', 'NSEI', 'NBEI'])

            progress_callback(1.0, "Advanced metrics complete")
            return df

        except Exception as e:
            logger.warning(f"Could not calculate efficiency metrics: {e}")
            progress_callback(1.0, "Skipped advanced metrics (error occurred)")
            return df

    def _calculate_oqpla_scores(
        self,
        df: pd.DataFrame,
        use_pdb: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pd.DataFrame:
        """
        Calculate OQPLA scores with optional PDB integration.

        Args:
            df: DataFrame with efficiency metrics
            use_pdb: If True, query PDB for structural evidence (default True)
            progress_callback: Optional callback for PDB progress updates

        Returns:
            DataFrame with OQPLA scores added
        """
        try:
            # Use Phase 2 which includes PDB scoring
            df = calculate_oqpla_phase2(df, use_pdb=use_pdb, progress_callback=progress_callback)
            df = add_oqpla_interpretation(df)
            return df

        except Exception as e:
            logger.warning(f"OQPLA scoring failed: {e}")
            return df

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
            df: DataFrame with OQPLA scores

        Returns:
            DataFrame with IMP classification added
        """
        try:
            return classify_imp_candidates(df, min_outlier_count=2, use_oqpla=True)
        except Exception as e:
            logger.warning(f"IMP classification failed: {e}")
            return df

    def _fetch_drug_indications(
        self,
        df: pd.DataFrame,
        progress_callback: Callable[[float, str], None]
    ) -> pd.DataFrame:
        """
        Fetch drug indications for all unique ChEMBL IDs.

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

        unique_ids = df['ChEMBL_ID'].dropna().unique()
        total = len(unique_ids)
        all_indications = []

        logger.info(f"Fetching drug indications for {total} unique compounds: {list(unique_ids)[:5]}...")
        progress_callback(0.0, f"Fetching drug indications for {total} compounds...")

        for i, chembl_id in enumerate(unique_ids):
            try:
                indications = get_drug_indications(chembl_id)
                if indications:
                    logger.debug(f"Found {len(indications)} indications for {chembl_id}")
                    # Convert tuple of dicts to list and add to results
                    for ind in indications:
                        all_indications.append(dict(ind))
            except Exception as e:
                logger.warning(f"Could not fetch indications for {chembl_id}: {e}")

            # Update progress every 10 compounds or at the end
            if (i + 1) % 10 == 0 or i == total - 1:
                pct = (i + 1) / total
                progress_callback(pct, f"Fetched indications for {i + 1}/{total} compounds")

        if all_indications:
            indications_df = pd.DataFrame(all_indications)
            logger.info(f"Found {len(indications_df)} drug indications across {indications_df['ChEMBL_ID'].nunique()} compounds")
            return indications_df
        else:
            logger.info("No drug indications found for any compounds")
            return pd.DataFrame()

    def _save_results(
        self,
        compound_name: str,
        smiles: str,
        similarity_threshold: int,
        activity_types: Optional[List[str]],
        df_results: pd.DataFrame,
        indications_df: Optional[pd.DataFrame] = None,
        entry_id: Optional[str] = None
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
            entry_id: Optional UUID for the compound entry (used for ZIP filename)

        Returns:
            Tuple of (zip_path, result_summary)
        """
        # Sanitize compound name for filesystem (consistent across codebase)
        safe_name = sanitize_compound_name(compound_name)
        compound_folder = os.path.join(self.results_dir, safe_name)
        os.makedirs(compound_folder, exist_ok=True)

        # Save CSV
        results_filename = os.path.join(compound_folder, f"{safe_name}_complete_results.csv")
        df_results.to_csv(results_filename, index=False)

        # Create metadata
        result_summary = {
            'compound_name': compound_name,
            'query_smiles': smiles,
            'similarity_threshold': similarity_threshold,
            'activity_types': ','.join(activity_types or []),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_compounds': df_results['ChEMBL_ID'].nunique() if 'ChEMBL_ID' in df_results.columns else 0,
            'total_bioactivity_rows': len(df_results),
        }

        # Add IMP summary if available
        if 'Is_IMP_Candidate' in df_results.columns:
            result_summary['imp_candidates'] = int(df_results['Is_IMP_Candidate'].sum())
            result_summary['has_imp_candidates'] = result_summary['imp_candidates'] > 0

        # Add PAINS summary if available
        if 'PAINS_Violation' in df_results.columns:
            result_summary['pains_count'] = int(df_results['PAINS_Violation'].sum())

        # Add similar_count for frontend
        result_summary['similar_count'] = result_summary.get('total_compounds', 0)
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

        # OQPLA score (average if available) - for Compound table
        if 'OQPLA_Final_Score' in df_results.columns:
            oqpla_values = df_results['OQPLA_Final_Score'].dropna()
            if len(oqpla_values) > 0:
                result_summary['avg_oqpla_score'] = float(oqpla_values.mean())
            else:
                result_summary['avg_oqpla_score'] = None
        else:
            result_summary['avg_oqpla_score'] = None

        # Save metadata (legacy filename)
        metadata_filename = os.path.join(compound_folder, f"{safe_name}_metadata.json")
        with open(metadata_filename, 'w') as f:
            json.dump(result_summary, f, indent=4)

        # Save standardized summary.json (for frontend direct Azure access)
        summary_filename = os.path.join(compound_folder, "summary.json")
        with open(summary_filename, 'w') as f:
            json.dump(result_summary, f, indent=4)

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
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(compound_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, compound_folder)
                    zipf.write(file_path, arcname)

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
    )
