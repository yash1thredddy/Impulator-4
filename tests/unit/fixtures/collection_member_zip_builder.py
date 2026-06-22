"""Shared test helper: build a fake collection-member ZIP for Phase 24 tests.

Streamlit-free. Mirrors how ``_save_results_sync`` (``compound_service.py``)
writes a member result ZIP: each per-compound CSV is written into the
``compound_folder`` and archived with
``arcname = os.path.relpath(file_path, compound_folder)`` so the three Tier-3
CSVs (``drug_indications.csv`` / ``all_similar_molecules.csv`` /
``pdb_summary.csv``) live at the **ROOT** of the member ZIP, not inside a
subfolder (Pitfall 4 in 24-RESEARCH).

The Stage-2 Option-A aggregate-artifact path (24-04) re-reads those three
root-level CSVs back out of each member ZIP, so this builder is the load-bearing
input for that re-read test. When a frame argument is ``None`` the corresponding
CSV is OMITTED entirely (a member with no drug indications writes no
``drug_indications.csv`` -- exactly what ``_save_results_sync`` does).
"""

from __future__ import annotations

import os
import zipfile

import pandas as pd

# Canonical Tier-3 CSV filenames, written at the ZIP ROOT by _save_results_sync.
DRUG_INDICATIONS_CSV = "drug_indications.csv"
ALL_SIMILAR_MOLECULES_CSV = "all_similar_molecules.csv"
PDB_SUMMARY_CSV = "pdb_summary.csv"


def build_member_zip(
    tmp_path,
    member_name: str,
    indications: pd.DataFrame | None = None,
    similar: pd.DataFrame | None = None,
    pdb: pd.DataFrame | None = None,
) -> str:
    """Write a fake member result ZIP and return its path.

    Args:
        tmp_path: a directory to write into (e.g. pytest's ``tmp_path``).
        member_name: used only to name the ZIP file (``{member_name}.zip``).
        indications: drug-indications frame; written as ``drug_indications.csv``
            at ZIP root. ``None`` -> CSV omitted.
        similar: all-similar-molecules frame; written as
            ``all_similar_molecules.csv`` at ZIP root. ``None`` -> CSV omitted.
        pdb: PDB-summary frame; written as ``pdb_summary.csv`` at ZIP root.
            ``None`` -> CSV omitted.

    Returns:
        Absolute path to the written ZIP.
    """
    zip_path = os.path.join(str(tmp_path), f"{member_name}.zip")

    # (frame, arcname) pairs -- arcname has NO subfolder, mirroring
    # os.path.relpath(file_path, compound_folder) for a file that sits directly
    # in compound_folder.
    entries: list[tuple[pd.DataFrame, str]] = []
    if indications is not None:
        entries.append((indications, DRUG_INDICATIONS_CSV))
    if similar is not None:
        entries.append((similar, ALL_SIMILAR_MOLECULES_CSV))
    if pdb is not None:
        entries.append((pdb, PDB_SUMMARY_CSV))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for frame, arcname in entries:
            zf.writestr(arcname, frame.to_csv(index=False))

    return zip_path
