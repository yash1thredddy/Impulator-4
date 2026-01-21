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

# O[Q/P/L]A Scoring
from backend.modules.oqpla_scoring import (
    calculate_oqpla_phase1,
    calculate_oqpla_phase2,
    interpret_oqpla_score,
    add_oqpla_interpretation,
    get_oqpla_summary,
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

# Assay Interference Filter (PAINS, etc.)
from backend.modules.assay_interference_filter import (
    get_all_interference_flags,
    check_pains_violations,
    check_aggregator_risk,
    check_redox_reactive,
    check_fluorescence_interference,
    check_thiol_reactive,
    get_detailed_interference_report,
    calculate_assay_quality_score,
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
    # O[Q/P/L]A Scoring
    "calculate_oqpla_phase1",
    "calculate_oqpla_phase2",
    "interpret_oqpla_score",
    "add_oqpla_interpretation",
    "get_oqpla_summary",
    "create_pdb_summary",
    "create_detailed_pdb_summary",
    # Configuration
    "ACTIVITY_TYPES",
    "CACHE_SIZE",
    "MAX_BATCH_SIZE",
    "MAX_WORKERS",
    # Assay Interference Filter
    "get_all_interference_flags",
    "check_pains_violations",
    "check_aggregator_risk",
    "check_redox_reactive",
    "check_fluorescence_interference",
    "check_thiol_reactive",
    "get_detailed_interference_report",
    "calculate_assay_quality_score",
    # Chemical Classifier
    "get_complete_classification",
    "get_classyfire_classification",
    "get_npclassifier_classification",
    "classify_compound_type",
    # IMP Classifier
    "classify_imp_candidates",
]
