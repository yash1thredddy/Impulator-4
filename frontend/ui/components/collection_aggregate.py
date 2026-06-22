"""Pure reader/parser for the Stage-2 ``collection_aggregate.json`` artifact.

No Streamlit, no IO. This module takes an ALREADY-LOADED dict (the heavy ZIP
download + ``st.cache_data`` wrapper live in ``collections.py``, plan 24-09) and
normalizes it for the Evidence & Annotations view. The artifact is written at
finalize by plan 24-04 with the D-S2-ARCH schema: a dict keyed by member
``entry_id`` -> {indications, pdb, all_similar, classification}.

Resilience mirrors ``imp_gmm.load_reference_corpus``: a missing, empty, or
partial/malformed artifact degrades to an empty result (no crash) so the
Evidence view falls back to the D-12 per-member drill-in per the UI-SPEC.
See app_research/DESIGN_collection_combined_view.md §1.

⚠️ Name collision (PATTERNS): this is ``collection_aggregat*e*.py`` (Stage-2
reader), distinct from ``collection_aggregat*ion*.py`` (Stage-1 stats, 24-02).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AggregateEntry:
    """Normalized per-member Evidence record from the aggregate artifact."""

    entry_id: str
    indications: list = field(default_factory=list)
    pdb: list = field(default_factory=list)
    all_similar: list = field(default_factory=list)
    classification: dict = field(default_factory=dict)


def parse_aggregate(data: object) -> dict[str, AggregateEntry]:
    """Normalize a loaded ``collection_aggregate.json`` dict into entries.

    Args:
        data: The already-loaded artifact. Expected to be a dict keyed by member
            ``entry_id``. Anything else (``None``, a list, a string, a partial
            record) degrades to an empty / partial result without raising.

    Returns:
        Dict keyed by ``entry_id`` -> :class:`AggregateEntry`. Empty dict when
        the artifact is missing/empty/malformed. Members whose record is not a
        dict are skipped (logged, not fatal).
    """
    if not isinstance(data, dict):
        if data is not None:
            logger.warning("aggregate_artifact_not_a_dict", got=type(data).__name__)
        return {}

    result: dict[str, AggregateEntry] = {}
    for entry_id, record in data.items():
        if not isinstance(record, dict):
            logger.warning("aggregate_member_record_skipped", entry_id=str(entry_id))
            continue
        try:
            result[str(entry_id)] = AggregateEntry(
                entry_id=str(entry_id),
                indications=_as_list(record.get("indications")),
                pdb=_as_list(record.get("pdb")),
                all_similar=_as_list(record.get("all_similar")),
                classification=_as_dict(record.get("classification")),
            )
        except (TypeError, ValueError):
            logger.warning("aggregate_member_parse_failed", entry_id=str(entry_id))
            continue
    return result


def filter_by_members(
    parsed: dict[str, AggregateEntry], selected_members: list[str]
) -> dict[str, AggregateEntry]:
    """Subset ``parsed`` to the selected members (the multiselect).

    Unknown member ids are ignored. An empty selection yields an empty dict.
    """
    selected = set(selected_members)
    return {eid: entry for eid, entry in parsed.items() if eid in selected}


def _as_list(value: object) -> list:
    """Return ``value`` if it is a list, else an empty list."""
    return value if isinstance(value, list) else []


def _as_dict(value: object) -> dict:
    """Return ``value`` if it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


__all__ = ["AggregateEntry", "parse_aggregate", "filter_by_members"]
