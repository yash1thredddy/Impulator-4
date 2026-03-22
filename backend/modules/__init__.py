"""
Chemistry modules for IMPULATOR.
Decoupled from Streamlit for backend use.

Imports are lazy — submodules are loaded on first access, not at package import.
This avoids pulling in heavy dependencies (rdkit, pandas, numpy) during test
collection or when only a single submodule is needed.
"""

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


def __getattr__(name):  # pragma: no cover -- lazy import dispatcher, tested indirectly
    """Lazy import: only load submodules when their exports are accessed."""
    # API Client
    if name in ("get_molecule_data", "get_classification", "get_chembl_ids",
                "fetch_compound_activities", "batch_fetch_activities",
                "get_target_name", "get_drug_indications", "clear_caches",
                "get_cache_info"):
        from backend.modules import api_client
        return getattr(api_client, name)

    # Efficiency Metrics
    if name in ("calculate_sei", "calculate_bei", "calculate_nsei",
                "calculate_nbei", "calculate_all_efficiency_metrics",
                "calculate_efficiency_metrics_dataframe"):
        from backend.modules import efficiency_metrics
        return getattr(efficiency_metrics, name)

    # Efficiency Planes
    if name in ("calculate_modulus", "calculate_angle",
                "calculate_all_plane_metrics",
                "calculate_plane_metrics_dataframe", "find_best_in_class"):
        from backend.modules import efficiency_planes
        return getattr(efficiency_planes, name)

    # Outlier Detection
    if name in ("detect_efficiency_outliers", "calculate_cohort_statistics",
                "get_outlier_summary", "calculate_z_scores", "filter_outliers"):
        from backend.modules import outlier_detection
        return getattr(outlier_detection, name)

    # IMP Scoring
    if name in ("calculate_imp_score_phase1", "calculate_imp_score_phase2",
                "interpret_imp_score", "add_imp_score_interpretation",
                "get_imp_score_summary", "create_pdb_summary",
                "create_detailed_pdb_summary"):
        from backend.modules import imp_scoring
        return getattr(imp_scoring, name)

    # Configuration (consolidated into backend.config)
    if name == "ACTIVITY_TYPES":
        return ["IC50", "Ki", "Kd", "EC50"]
    if name in ("CACHE_SIZE", "MAX_BATCH_SIZE"):
        from backend.config import settings
        return getattr(settings, name.upper())

    # Assay Interference Filter
    if name in ("InterferenceFlags", "calculate_interference_flags",
                "get_interference_flags_from_smiles", "get_interference_summary",
                "check_pains_violations", "check_brenk_alerts",
                "check_nih_alerts", "check_aggregator_risk",
                "check_thiol_reactive", "check_redox_active",
                "check_fluorescence_interference", "get_all_filter_matches",
                "FLAG_DESCRIPTIONS", "METHODOLOGY_REFERENCES",
                "get_methodology_citation", "get_methodology_doi",
                "get_all_methodology_references"):
        from backend.modules import assay_interference_filter
        return getattr(assay_interference_filter, name)

    # Chemical Classifier
    if name in ("get_complete_classification", "get_classyfire_classification",
                "get_npclassifier_classification", "classify_compound_type"):
        from backend.modules import chemical_classifier
        return getattr(chemical_classifier, name)

    # IMP Classifier
    if name in ("classify_imp_candidates",):
        from backend.modules import imp_classifier
        return getattr(imp_classifier, name)

    raise AttributeError(f"module 'backend.modules' has no attribute {name!r}")
