"""Pure logic for the Collection pre-flight (no Streamlit, no IO).

Grouping, availability planning, and decision application for the two-phase
collection submit. See app_research/DESIGN_collection_preflight.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def compute_inchikey(smiles: str) -> str | None:
    """InChIKey for a SMILES via RDKit, or None if unparseable/empty.

    Mirrors the legacy `_compute_inchikey` in analyze.py for grouping parity,
    with one deliberate hardening: an empty / 0-atom mol returns None. RDKit's
    `MolFromSmiles("")` yields a non-None 0-atom mol whose InChIKey is "", which
    would let empty-SMILES members silently dedupe against each other. Returning
    None keeps them OUT of duplicate groups — both group_in_file_duplicates and
    _dedupe_members_by_structure skip a None key — so they are reported invalid.
    """
    try:
        from rdkit import Chem

        if not smiles or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


@dataclass
class DupGroup:
    """A set of >1 members sharing one InChIKey (in-file duplicates)."""

    inchikey: str
    member_indices: list[int]
    names: list[str]


def group_in_file_duplicates(members: list[dict]) -> list[DupGroup]:
    """Group members by InChIKey; return only groups with >1 member.

    Groups are ordered by the input index of their FIRST member. Members whose
    SMILES cannot be parsed are not grouped (reported elsewhere).
    """
    by_key: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, member in enumerate(members):
        key = compute_inchikey(member.get("smiles", ""))
        if key is None:
            continue
        if key not in by_key:
            by_key[key] = []
            order.append(key)
        by_key[key].append(idx)

    groups: list[DupGroup] = []
    for key in order:
        indices = by_key[key]
        if len(indices) > 1:
            groups.append(
                DupGroup(
                    inchikey=key,
                    member_indices=indices,
                    names=[str(members[i].get("name", "")) for i in indices],
                )
            )
    return groups


@dataclass
class MemberPlan:
    """Per-member availability classification."""

    name: str
    smiles: str
    status: str  # "ready" | "needs_lower" | "no_data" | "unknown"
    requested_threshold: int
    tiers: list[dict] = field(default_factory=list)   # [{threshold,count}], count>0, desc
    suggested_threshold: int | None = None


@dataclass
class PreflightPlan:
    members: list[MemberPlan]
    dup_groups: list[DupGroup]
    ready_count: int
    needs_lower_count: int
    no_data_count: int


def _index_availability(results: list[dict]) -> dict[tuple[str, str], dict]:
    """Index availability rows by (compound_name, smiles)."""
    idx: dict[tuple[str, str], dict] = {}
    for r in results or []:
        if isinstance(r, dict):
            idx[(r.get("compound_name", ""), r.get("smiles", ""))] = r
    return idx


def build_preflight_plan(
    members: list[dict],
    availability_results: list[dict],
    requested_threshold: int,
) -> PreflightPlan:
    """Classify each member from the availability response + group in-file dups."""
    avail_idx = _index_availability(availability_results)
    plans: list[MemberPlan] = []
    ready = needs_lower = no_data = 0

    for member in members:
        name = str(member.get("name", ""))
        smiles = str(member.get("smiles", ""))
        avail = avail_idx.get((name, smiles))

        if avail is None:
            plans.append(MemberPlan(name, smiles, "unknown", requested_threshold))
            continue

        # count>0 tiers, descending by threshold
        tiers = sorted(
            [
                {"threshold": int(t["threshold"]), "count": int(t["count"])}
                for t in (avail.get("thresholds") or [])
                if int(t.get("count", 0)) > 0
            ],
            key=lambda t: t["threshold"],
            reverse=True,
        )

        if avail.get("available") is True:
            plans.append(
                MemberPlan(name, smiles, "ready", requested_threshold,
                           tiers=tiers, suggested_threshold=requested_threshold)
            )
            ready += 1
        elif avail.get("has_any_data") is False:
            plans.append(MemberPlan(name, smiles, "no_data", requested_threshold))
            no_data += 1
        else:
            suggested = min((t["threshold"] for t in tiers), default=None)
            plans.append(
                MemberPlan(name, smiles, "needs_lower", requested_threshold,
                           tiers=tiers, suggested_threshold=suggested)
            )
            needs_lower += 1

    return PreflightPlan(
        members=plans,
        dup_groups=group_in_file_duplicates(members),
        ready_count=ready,
        needs_lower_count=needs_lower,
        no_data_count=no_data,
    )


def apply_preflight_decisions(
    members: list[dict],
    dup_decisions: dict[str, str],
    threshold_decisions: dict[int, int],
    excluded_indices: set[int],
) -> list[dict]:
    """Produce the final member list after pre-flight decisions (INDEX-keyed, D-PF-7).

    - dup_decisions: {inchikey: "first"|"both"}; default (missing) == "first".
      "first" drops all but the first member of that in-file duplicate group.
    - threshold_decisions: {member_index: threshold}; stamps similarity_threshold.
    - excluded_indices: member indices to drop entirely (auto-excluded no-data).

    Keyed by index, NOT name: member names are not unique, so name keys would
    let one decision bleed across same-named members.
    """
    groups = group_in_file_duplicates(members)
    dropped: set[int] = set(excluded_indices)
    for g in groups:
        if dup_decisions.get(g.inchikey, "first") != "both":
            dropped.update(g.member_indices[1:])  # keep first only

    final: list[dict] = []
    for idx, member in enumerate(members):
        if idx in dropped:
            continue
        out = dict(member)
        if idx in threshold_decisions:
            out["similarity_threshold"] = int(threshold_decisions[idx])
        final.append(out)
    return final


def distinct_thresholds(members: list[dict]) -> list[int]:
    """Distinct per-member thresholds, descending. len>1 => mixed-threshold note."""
    seen = {int(m.get("similarity_threshold") or 90) for m in members}
    return sorted(seen, reverse=True)
