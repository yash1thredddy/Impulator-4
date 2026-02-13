"""
PubChem Chemical-Target Interactions Explorer.

Fetches Chemical-Target Interactions from PubChem's consolidatedcompoundtarget
SDQ collection. Shows unique gene targets and the complete interaction table.

Data sources aggregated by PubChem:
  - BindingDB     (binding affinities: IC50, Ki, Kd, EC50)
  - CTD           (Comparative Toxicogenomics Database)
  - DGIdb         (Drug Gene Interaction database)
  - DrugBank      (drug-target associations)
  - IUPHAR/BPS    (Guide to Pharmacology)
  - TTD           (Therapeutic Target Database)
  - T3DB          (Toxin and Toxin Target Database)

IMPORTANT: This collection must be queried by compound NAME (regex),
not by CID. CID-based queries return 0 for many compounds.

Usage:
    python scripts/target_interactions.py
    python scripts/target_interactions.py --compound "Quercetin"
    python scripts/target_interactions.py --compound "Aspirin" --source BindingDB
    python scripts/target_interactions.py --compound "Quercetin" --human-only
    python scripts/target_interactions.py --compound "Quercetin" --json
    python scripts/target_interactions.py --compound "Quercetin" --csv output.csv
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

PUBCHEM_SDQ = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi"
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RATE_LIMIT_INTERVAL = 0.25  # 4 req/s (PubChem allows 5)
GENERAL_TIMEOUT = 60
PAGE_SIZE = 10000  # Max rows per SDQ request


# ═══════════════════════════════════════════════════════════════════════════
#  HTTP Session & Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════

def _create_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = _create_session()
_last_request_time = 0.0


def _rate_limit():
    global _last_request_time
    elapsed = time.perf_counter() - _last_request_time
    if elapsed < RATE_LIMIT_INTERVAL:
        time.sleep(RATE_LIMIT_INTERVAL - elapsed)
    _last_request_time = time.perf_counter()


# ═══════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Interaction:
    """Single compound-target interaction row from PubChem.

    Matches PubChem's Chemical-Target Interactions table columns:
      CID, Compound, Protein Accession, Protein, Gene ID, Gene,
      Taxonomy ID, Taxonomy, Source Chemical ID, Source Chemical,
      Source Target ID, Source Target, Action, Activity Name, Activity,
      Evidence IDs
    """
    cid: int = 0
    geneid: int = 0
    genename: str = ""
    taxid: int = 0
    taxname: str = ""
    protacxn: str = ""       # UniProt accession (Protein Accession)
    protname: str = ""       # Protein name
    dsn: str = ""            # Data source name
    action: str = ""         # e.g. "Inhibitor", "Agonist"
    actname: str = ""        # e.g. "IC50 (nM)", "Ki (nM)"
    actvalue: str = ""       # e.g. "3.12e+3"
    pmids: str = ""
    dois: str = ""
    citations: str = ""
    srccmpd: str = ""        # Source Chemical ID
    srccmpdname: str = ""    # Source Chemical name
    srccmpdurl: str = ""
    srctarget: str = ""      # Source Target ID
    srctargetname: str = ""  # Source Target name
    srctargeturl: str = ""
    pclids: str = ""         # PubChem assay IDs


@dataclass
class GeneTarget:
    """Aggregated info for a unique gene target."""
    geneid: int = 0
    genename: str = ""
    sources: Set[str] = field(default_factory=set)
    actions: Set[str] = field(default_factory=set)
    species: Set[str] = field(default_factory=set)
    uniprot: Set[str] = field(default_factory=set)
    protein_names: Set[str] = field(default_factory=set)
    activity_types: Set[str] = field(default_factory=set)
    activity_values: List[Tuple[str, str]] = field(default_factory=list)
    pmid_count: int = 0
    interaction_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  API Functions
# ═══════════════════════════════════════════════════════════════════════════

def resolve_cid(compound_name: str) -> Optional[int]:
    """Resolve compound name to PubChem CID."""
    _rate_limit()
    url = f"{PUBCHEM_REST}/compound/name/{requests.utils.quote(compound_name)}/cids/JSON"
    try:
        resp = SESSION.get(url, timeout=GENERAL_TIMEOUT)
        if resp.status_code == 200:
            cids = resp.json().get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
    except Exception:
        pass
    return None


def fetch_all_interactions(compound_name: str) -> Tuple[List[Interaction], int]:
    """Fetch ALL Chemical-Target Interactions for a compound.

    Uses consolidatedcompoundtarget SDQ collection.
    MUST query by compound name (regex), not CID.

    Returns (interactions, total_count).
    """
    all_interactions = []
    start = 1
    total = 0

    while True:
        _rate_limit()
        query = json.dumps({
            "select": "*",
            "collection": "consolidatedcompoundtarget",
            "where": {"ands": [{"cmpdname": f"^{compound_name}$"}]},
            "start": start,
            "limit": PAGE_SIZE,
            "width": 1000000,
        })

        try:
            resp = SESSION.get(
                PUBCHEM_SDQ,
                params={"infmt": "json", "outfmt": "json", "query": query},
                timeout=GENERAL_TIMEOUT,
            )
            if resp.status_code != 200:
                print(f"  [ERROR] SDQ returned HTTP {resp.status_code}")
                break

            data = resp.json()
            output = data.get("SDQOutputSet", [{}])[0]
            rows = output.get("rows", [])
            total = output.get("totalCount", 0)

            if not rows:
                break

            for r in rows:
                ix = Interaction(
                    cid=r.get("cid", 0),
                    geneid=r.get("geneid", 0),
                    genename=r.get("genename", ""),
                    taxid=r.get("taxid", 0),
                    taxname=r.get("taxname", ""),
                    protacxn=r.get("protacxn", ""),
                    protname=r.get("protname", ""),
                    dsn=r.get("dsn", ""),
                    action=r.get("action", ""),
                    actname=r.get("actname", ""),
                    actvalue=r.get("actvalue", ""),
                    pmids=r.get("pmids", ""),
                    dois=r.get("dois", ""),
                    citations=r.get("citations", ""),
                    srccmpd=r.get("srccmpd", ""),
                    srccmpdname=r.get("srccmpdname", ""),
                    srccmpdurl=r.get("srccmpdurl", ""),
                    srctarget=r.get("srctarget", ""),
                    srctargetname=r.get("srctargetname", ""),
                    srctargeturl=r.get("srctargeturl", ""),
                    pclids=r.get("pclids", ""),
                )
                all_interactions.append(ix)

            if start + len(rows) > total:
                break
            start += len(rows)

        except Exception as e:
            print(f"  [ERROR] SDQ query failed: {e}")
            break

    return all_interactions, total


# ═══════════════════════════════════════════════════════════════════════════
#  Analysis Functions
# ═══════════════════════════════════════════════════════════════════════════

def build_gene_map(interactions: List[Interaction]) -> Dict[str, GeneTarget]:
    """Aggregate interactions by unique gene name."""
    gene_map: Dict[str, GeneTarget] = {}

    for ix in interactions:
        key = ix.genename or f"gene_{ix.geneid}"
        if key not in gene_map:
            gene_map[key] = GeneTarget(
                geneid=ix.geneid,
                genename=ix.genename,
            )
        g = gene_map[key]
        g.interaction_count += 1
        if ix.dsn:
            g.sources.add(ix.dsn)
        if ix.action:
            g.actions.add(ix.action)
        if ix.taxname:
            g.species.add(ix.taxname)
        if ix.protacxn:
            g.uniprot.add(ix.protacxn)
        if ix.protname:
            g.protein_names.add(ix.protname)
        if ix.actname:
            g.activity_types.add(ix.actname)
        if ix.actname and ix.actvalue:
            g.activity_values.append((ix.actname, ix.actvalue))
        if ix.pmids:
            g.pmid_count += len(ix.pmids.split(","))

    return gene_map


def filter_interactions(
    interactions: List[Interaction],
    source: Optional[str] = None,
    human_only: bool = False,
    gene: Optional[str] = None,
) -> List[Interaction]:
    """Filter interactions by source, species, or gene."""
    filtered = interactions
    if source:
        src_lower = source.lower()
        filtered = [ix for ix in filtered if src_lower in ix.dsn.lower()]
    if human_only:
        filtered = [ix for ix in filtered if ix.taxid == 9606]
    if gene:
        gene_upper = gene.upper()
        filtered = [ix for ix in filtered
                    if gene_upper in ix.genename.upper()
                    or gene_upper in ix.protname.upper()
                    or gene_upper in ix.srctargetname.upper()]
    return filtered


# ═══════════════════════════════════════════════════════════════════════════
#  Display Functions
# ═══════════════════════════════════════════════════════════════════════════

def print_header(text: str):
    width = 90
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def display_summary(compound_name: str, cid: Optional[int],
                    interactions: List[Interaction],
                    gene_map: Dict[str, GeneTarget]):
    """Print overview summary."""
    sep = "-" * 90

    sources = defaultdict(int)
    species = defaultdict(int)
    actions = defaultdict(int)
    act_types = defaultdict(int)

    for ix in interactions:
        if ix.dsn:
            sources[ix.dsn] += 1
        if ix.taxname:
            species[ix.taxname] += 1
        if ix.action:
            actions[ix.action] += 1
        if ix.actname:
            act_types[ix.actname] += 1

    print(f"\n  {sep}")
    print(f"  SUMMARY: {compound_name}" +
          (f" (CID {cid})" if cid else ""))
    print(f"  {sep}")
    print(f"  {'Total interactions':<35} {len(interactions):>8}")
    print(f"  {'Unique genes':<35} {len(gene_map):>8}")
    print(f"  {'Data sources':<35} {len(sources):>8}")
    print(f"  {'Species':<35} {len(species):>8}")
    print(f"  {sep}")

    # By source
    print("\n  Interactions by Source:")
    print(f"  {'Source':<50} {'Count':>8}")
    print(f"  {'-' * 62}")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src:<50} {count:>8}")

    # By species
    if len(species) > 1:
        print("\n  Interactions by Species:")
        print(f"  {'Species':<50} {'Count':>8}")
        print(f"  {'-' * 62}")
        for sp, count in sorted(species.items(), key=lambda x: -x[1]):
            print(f"  {sp:<50} {count:>8}")

    # By action
    if actions:
        print("\n  Interactions by Action:")
        print(f"  {'Action':<50} {'Count':>8}")
        print(f"  {'-' * 62}")
        for act, count in sorted(actions.items(), key=lambda x: -x[1]):
            print(f"  {act:<50} {count:>8}")

    # By activity type
    if act_types:
        print("\n  Activity Measurement Types:")
        print(f"  {'Type':<50} {'Count':>8}")
        print(f"  {'-' * 62}")
        for at, count in sorted(act_types.items(), key=lambda x: -x[1]):
            print(f"  {at:<50} {count:>8}")


def display_unique_genes(gene_map: Dict[str, GeneTarget]):
    """Print unique genes table sorted by interaction count."""
    print_header("UNIQUE GENE TARGETS")
    sep = "-" * 90

    sorted_genes = sorted(gene_map.values(),
                          key=lambda g: -g.interaction_count)

    print(f"\n  {'#':>4}  {'Gene':<12} {'GeneID':>8}  {'Sources':>3}  "
          f"{'Actions':<25}  {'Species':<20}  {'#Ix':>4}  {'UniProt':<12}")
    print(f"  {sep}")

    for i, g in enumerate(sorted_genes, 1):
        actions_str = ", ".join(sorted(g.actions)[:3]) if g.actions else "-"
        if len(actions_str) > 25:
            actions_str = actions_str[:22] + "..."
        species_str = "Human" if {"Homo sapiens (human)"} == g.species else (
            f"{len(g.species)} spp." if len(g.species) > 1 else
            (list(g.species)[0][:20] if g.species else "-")
        )
        uniprot_str = ", ".join(sorted(g.uniprot)[:2]) if g.uniprot else "-"
        if len(uniprot_str) > 12:
            uniprot_str = uniprot_str[:9] + "..."

        print(f"  {i:>4}  {g.genename:<12} {g.geneid:>8}  "
              f"{len(g.sources):>3}  {actions_str:<25}  "
              f"{species_str:<20}  {g.interaction_count:>4}  {uniprot_str:<12}")

    print(f"\n  Total: {len(sorted_genes)} unique genes")


def _shorten_source(dsn: str) -> str:
    """Shorten common PubChem data source names."""
    return (dsn
            .replace("Drug Gene Interaction database (DGIdb)", "DGIdb")
            .replace("Comparative Toxicogenomics Database (CTD)", "CTD")
            .replace("Therapeutic Target Database", "TTD")
            .replace("IUPHAR/BPS Guide to Pharmacology", "IUPHAR"))


def _build_evidence(ix: Interaction) -> str:
    """Build evidence ID string from PMIDs and DOIs."""
    parts = []
    if ix.pmids:
        for pmid in ix.pmids.split(","):
            pmid = pmid.strip()
            if pmid:
                parts.append(f"PMID:{pmid}")
    if ix.dois:
        for doi in ix.dois.split("|"):
            doi = doi.strip()
            if doi and not doi.startswith("PMID"):
                parts.append(f"DOI:{doi}")
    return ", ".join(parts)


def display_full_table(interactions: List[Interaction]):
    """Print complete interaction table matching PubChem's column layout.

    PubChem columns: CID, Compound, Protein Accession, Protein, Gene ID,
    Gene, Taxonomy ID, Taxonomy, Src Chem ID, Src Chemical, Src Target ID,
    Src Target, Action, Activity Name, Activity, Evidence IDs
    """
    print_header("COMPLETE INTERACTION TABLE (PubChem format)")

    # Two-row format per interaction for terminal readability
    sep = "-" * 120

    print(f"\n  {'#':>4}  {'CID':<10} {'Protein Acc':<12} {'Protein':<30} "
          f"{'GeneID':>7} {'Gene':<10} {'Taxonomy':<15}")
    print(f"  {'':>4}  {'Source':<12} {'Src Chem ID':<12} {'Src Chemical':<30} "
          f"{'Src Tgt ID':>10} {'Src Target':<15}")
    print(f"  {'':>4}  {'Action':<12} {'Act. Name':<15} {'Act. Value':<12} "
          f"{'Evidence IDs':<50}")
    print(f"  {sep}")

    for i, ix in enumerate(interactions, 1):
        protein = ix.protname or "-"
        if len(protein) > 30:
            protein = protein[:27] + "..."
        taxonomy = "Human" if ix.taxid == 9606 else (ix.taxname[:15] if ix.taxname else "-")
        src_chem = ix.srccmpdname or "-"
        if len(src_chem) > 30:
            src_chem = src_chem[:27] + "..."
        src_target = ix.srctargetname or "-"
        if len(src_target) > 15:
            src_target = src_target[:12] + "..."
        source = _shorten_source(ix.dsn) if ix.dsn else "-"
        if len(source) > 12:
            source = source[:9] + "..."
        evidence = _build_evidence(ix)
        if len(evidence) > 50:
            evidence = evidence[:47] + "..."

        # Row 1: identity
        print(f"  {i:>4}  {ix.cid:<10} {ix.protacxn or '-':<12} {protein:<30} "
              f"{ix.geneid:>7} {ix.genename or '-':<10} {taxonomy:<15}")
        # Row 2: source
        print(f"  {'':>4}  {source:<12} {ix.srccmpd or '-':<12} {src_chem:<30} "
              f"{ix.srctarget or '-':>10} {src_target:<15}")
        # Row 3: activity + evidence
        print(f"  {'':>4}  {ix.action or '-':<12} {ix.actname or '-':<15} "
              f"{ix.actvalue or '-':<12} {evidence:<50}")
        print(f"  {'-' * 100}")

    print(f"\n  Total: {len(interactions)} interactions")


def display_gene_detail(gene_map: Dict[str, GeneTarget], gene_name: str):
    """Print detailed view for a specific gene."""
    gene_upper = gene_name.upper()
    matches = {k: v for k, v in gene_map.items() if gene_upper in k.upper()}

    if not matches:
        print(f"\n  Gene '{gene_name}' not found in interactions.")
        return

    for name, g in matches.items():
        print_header(f"GENE DETAIL: {name} (GeneID: {g.geneid})")
        print(f"\n  Protein Names:    {', '.join(sorted(g.protein_names)) or '-'}")
        print(f"  UniProt:          {', '.join(sorted(g.uniprot)) or '-'}")
        print(f"  Species:          {', '.join(sorted(g.species))}")
        print(f"  Actions:          {', '.join(sorted(g.actions)) or '-'}")
        print(f"  Sources:          {', '.join(sorted(g.sources))}")
        print(f"  Interactions:     {g.interaction_count}")
        print(f"  PubMed refs:      {g.pmid_count}")

        if g.activity_values:
            print("\n  Activity Measurements:")
            print(f"  {'Type':<25} {'Value':<15}")
            print(f"  {'-' * 42}")
            for atype, aval in g.activity_values:
                print(f"  {atype:<25} {aval:<15}")


# ═══════════════════════════════════════════════════════════════════════════
#  Export Functions
# ═══════════════════════════════════════════════════════════════════════════

def export_csv(interactions: List[Interaction], filepath: str):
    """Export interactions to CSV matching PubChem's table columns exactly."""
    # Column mapping: PubChem display name -> Interaction field
    columns = [
        ("Compound CID", "cid"),
        ("Compound", "cmpdname"),  # handled specially
        ("Protein Accession", "protacxn"),
        ("Protein", "protname"),
        ("Gene ID", "geneid"),
        ("Gene", "genename"),
        ("Taxonomy ID", "taxid"),
        ("Taxonomy", "taxname"),
        ("Source Chemical ID", "srccmpd"),
        ("Source Chemical", "srccmpdname"),
        ("Source Target ID", "srctarget"),
        ("Source Target", "srctargetname"),
        ("Action", "action"),
        ("Activity Name", "actname"),
        ("Activity", "actvalue"),
        ("Evidence IDs", "evidence"),  # handled specially
        ("Data Source", "dsn"),
        ("PMIDs", "pmids"),
        ("DOIs", "dois"),
        ("PubChem Assay IDs", "pclids"),
        ("Citations", "citations"),
    ]
    headers = [c[0] for c in columns]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for ix in interactions:
            row = []
            for display_name, field_name in columns:
                if field_name == "cmpdname":
                    # Compound name not stored per row, derive from first row
                    row.append(interactions[0].srccmpdname if interactions else "")
                elif field_name == "evidence":
                    row.append(_build_evidence(ix))
                else:
                    row.append(getattr(ix, field_name, ""))
            writer.writerow(row)
    print(f"\n  Exported {len(interactions)} interactions to {filepath}")


def export_json(compound_name: str, cid: Optional[int],
                interactions: List[Interaction],
                gene_map: Dict[str, GeneTarget]) -> str:
    """Export full data as JSON."""
    data = {
        "compound": compound_name,
        "cid": cid,
        "total_interactions": len(interactions),
        "unique_genes": len(gene_map),
        "genes": {},
        "interactions": [],
    }

    for name, g in sorted(gene_map.items(),
                           key=lambda x: -x[1].interaction_count):
        data["genes"][name] = {
            "geneid": g.geneid,
            "sources": sorted(g.sources),
            "actions": sorted(g.actions),
            "species": sorted(g.species),
            "uniprot": sorted(g.uniprot),
            "protein_names": sorted(g.protein_names),
            "activity_types": sorted(g.activity_types),
            "activity_values": [
                {"type": t, "value": v} for t, v in g.activity_values
            ],
            "interaction_count": g.interaction_count,
            "pmid_count": g.pmid_count,
        }

    for ix in interactions:
        data["interactions"].append({
            "genename": ix.genename,
            "geneid": ix.geneid,
            "protname": ix.protname,
            "protacxn": ix.protacxn,
            "taxname": ix.taxname,
            "dsn": ix.dsn,
            "action": ix.action,
            "actname": ix.actname,
            "actvalue": ix.actvalue,
            "srctargetname": ix.srctargetname,
            "pmids": ix.pmids,
            "dois": ix.dois,
            "cid": ix.cid,
        })

    return json.dumps(data, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PubChem Chemical-Target Interactions Explorer"
    )
    parser.add_argument("--compound", "-c", type=str, default="Quercetin",
                        help="Compound name (default: Quercetin)")
    parser.add_argument("--source", "-s", type=str, default=None,
                        help="Filter by source (BindingDB, CTD, DGIdb, DrugBank, IUPHAR, TTD, T3DB)")
    parser.add_argument("--human-only", action="store_true",
                        help="Show only human (taxid=9606) interactions")
    parser.add_argument("--gene", "-g", type=str, default=None,
                        help="Filter or show detail for a specific gene")
    parser.add_argument("--no-table", action="store_true",
                        help="Skip the full interaction table")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export to CSV file")
    args = parser.parse_args()

    compound = args.compound

    print_header(f"Chemical-Target Interactions: {compound}")
    print("  Source: PubChem consolidatedcompoundtarget (SDQ)")
    print("  Databases: BindingDB, CTD, DGIdb, DrugBank, IUPHAR, TTD, T3DB")

    # Resolve CID
    print(f"\n  Resolving CID for '{compound}'...", end="", flush=True)
    cid = resolve_cid(compound)
    if cid:
        print(f" CID {cid}")
    else:
        print(" not found (proceeding with name-based query)")

    # Fetch all interactions
    print(f"  Fetching interactions for '{compound}'...", end="", flush=True)
    start_time = time.perf_counter()
    interactions, total = fetch_all_interactions(compound)
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f" {len(interactions)} / {total} fetched ({elapsed:.0f} ms)")

    if not interactions:
        print(f"\n  No Chemical-Target Interactions found for '{compound}'.")
        print("  Try a different compound name or check PubChem directly:")
        print(f"  https://pubchem.ncbi.nlm.nih.gov/#query={compound}")
        return

    # Apply filters
    filtered = filter_interactions(
        interactions,
        source=args.source,
        human_only=args.human_only,
        gene=args.gene if not args.gene else None,  # gene detail handled separately
    )

    if args.source or args.human_only:
        filters = []
        if args.source:
            filters.append(f"source={args.source}")
        if args.human_only:
            filters.append("human-only")
        print(f"  Filters: {', '.join(filters)}")
        print(f"  After filtering: {len(filtered)} interactions")

    # Build gene map from filtered interactions
    gene_map = build_gene_map(filtered)

    # JSON output
    if args.json:
        print(export_json(compound, cid, filtered, gene_map))
        return

    # CSV export
    if args.csv:
        export_csv(filtered, args.csv)

    # Summary
    display_summary(compound, cid, filtered, gene_map)

    # Unique genes table
    display_unique_genes(gene_map)

    # Gene detail (if requested)
    if args.gene:
        display_gene_detail(gene_map, args.gene)

    # Full table
    if not args.no_table:
        display_full_table(filtered)

    print()


if __name__ == "__main__":
    main()
