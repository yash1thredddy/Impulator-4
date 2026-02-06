"""
Chemistry modules for IMPULATOR.
Decoupled from Streamlit for backend use.
"""

# API Client
from backend.modules.api_client import (
    get_molecule_data,
    get_classification,
    get_chembl_ids,
    fetch_compound_activities,
    batch_fetch_activities,
    get_target_name,
    get_drug_indications,
    clear_caches,
    get_cache_info,
)

# Efficiency Metrics
from backend.modules.efficiency_metrics import (
    calculate_sei,
    calculate_bei,
    calculate_nsei,
    calculate_nbei,
    calculate_all_efficiency_metrics,
    calculate_efficiency_metrics_dataframe,
)

# Efficiency Planes
from backend.modules.efficiency_planes import (
    calculate_modulus,
    calculate_angle,
    calculate_all_plane_metrics,
    calculate_plane_metrics_dataframe,
    find_best_in_class,
)

# Outlier Detection
from backend.modules.outlier_detection import (
    detect_efficiency_outliers,
    calculate_cohort_statistics,
    get_outlier_summary,
    calculate_z_scores,
    filter_outliers,
)

# IMP Scoring
from backend.modules.imp_scoring import (
    calculate_imp_score_phase1,
    calculate_imp_score_phase2,
    interpret_imp_score,
    add_imp_score_interpretation,
    get_imp_score_summary,
    create_pdb_summary,
    create_detailed_pdb_summary,
)

# Configuration
from backend.modules.config import (
    ACTIVITY_TYPES,
    CACHE_SIZE,
    MAX_BATCH_SIZE,
    MAX_WORKERS,
)

# Assay Interference Filter (PAINS, BRENK, NIH, etc.)
from backend.modules.assay_interference_filter import (
    InterferenceFlags,
    calculate_interference_flags,
    get_interference_flags_from_smiles,
    get_interference_summary,
    check_pains_violations,
    check_brenk_alerts,
    check_nih_alerts,
    check_aggregator_risk,
    check_thiol_reactive,
    check_redox_active,
    check_fluorescence_interference,
    get_all_filter_matches,
    FLAG_DESCRIPTIONS,
    METHODOLOGY_REFERENCES,
    get_methodology_citation,
    get_methodology_doi,
    get_all_methodology_references,
)

# Chemical Classifier
from backend.modules.chemical_classifier import (
    get_complete_classification,
    get_classyfire_classification,
    get_npclassifier_classification,
    classify_compound_type,
)

# IMP Classifier
from backend.modules.imp_classifier import (
    classify_imp_candidates,
)

__all__ = [
    # API Client
    "get_molecule_data",
    "get_classification",
    "get_chembl_ids",
    "fetch_compound_activities",
    "batch_fetch_activities",
    "get_target_name",
    "get_drug_indications",
    "clear_caches",
    "get_cache_info",
    # Efficiency Metrics
    "calculate_sei",
    "calculate_bei",
    "calculate_nsei",
    "calculate_nbei",
    "calculate_all_efficiency_metrics",
    "calculate_efficiency_metrics_dataframe",
    # Efficiency Planes
    "calculate_modulus",
    "calculate_angle",
    "calculate_all_plane_metrics",
    "calculate_plane_metrics_dataframe",
    "find_best_in_class",
    # Outlier Detection
    "detect_efficiency_outliers",
    "calculate_cohort_statistics",
    "get_outlier_summary",
    "calculate_z_scores",
    "filter_outliers",
    # IMP Scoring
    "calculate_imp_score_phase1",
    "calculate_imp_score_phase2",
    "interpret_imp_score",
    "add_imp_score_interpretation",
    "get_imp_score_summary",
    "create_pdb_summary",
    "create_detailed_pdb_summary",
    # Configuration
    "ACTIVITY_TYPES",
    "CACHE_SIZE",
    "MAX_BATCH_SIZE",
    "MAX_WORKERS",
    # Assay Interference Filter
    "InterferenceFlags",
    "calculate_interference_flags",
    "get_interference_flags_from_smiles",
    "get_interference_summary",
    "check_pains_violations",
    "check_brenk_alerts",
    "check_nih_alerts",
    "check_aggregator_risk",
    "check_thiol_reactive",
    "check_redox_active",
    "check_fluorescence_interference",
    "get_all_filter_matches",
    "FLAG_DESCRIPTIONS",
    "METHODOLOGY_REFERENCES",
    "get_methodology_citation",
    "get_methodology_doi",
    "get_all_methodology_references",
    # Chemical Classifier
    "get_complete_classification",
    "get_classyfire_classification",
    "get_npclassifier_classification",
    "classify_compound_type",
    # IMP Classifier
    "classify_imp_candidates",
]
