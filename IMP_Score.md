# IMP Score: Multi-Criteria Scoring for Invalid Metabolic Panacea Detection

## Table of Contents

1. [What is IMP Score?](#what-is-imp-score)
2. [Quick Reference](#quick-reference)
3. [The Five Components](#the-five-components)
   - [Component 1: Efficiency Outlier Score (45%)](#component-1-efficiency-outlier-score-45-weight)
   - [Component 2: Distance to Best-in-Class (20%)](#component-2-distance-to-best-in-class-20-weight)
   - [Component 3: Development Angle Score (15%)](#component-3-development-angle-score-15-weight)
   - [Component 4: Assay Interference Score (15%)](#component-4-assay-interference-score-15-weight)
   - [Component 5: PDB Structural Evidence (5%)](#component-5-pdb-structural-evidence-5-weight)
4. [Final Score Calculation](#final-score-calculation)
5. [Score Interpretation](#score-interpretation)
6. [Step-by-Step Calculation Example](#step-by-step-calculation-example)
7. [Efficiency Metrics Reference](#efficiency-metrics-reference)
8. [Assay Interference Flags Reference](#assay-interference-flags-reference)
9. [Red Flags & Decision Guide](#red-flags--decision-guide)
10. [Implementation](#implementation)
11. [Glossary](#glossary)
12. [References](#references)

---

## What is IMP Score?

IMP Score is a multi-criteria scoring system that identifies **Invalid Metabolic Panaceas (IMPs)** - compounds that appear to have exceptional bioactivity in screening assays but are actually assay artifacts (false positives).

**Higher IMP Score = Higher probability the compound is a false positive.**

The score evaluates five independent dimensions:

1. How exceptionally efficient the compound appears (suspiciously good = higher score)
2. How close it is to the best compound in the cohort
3. Whether its development profile is balanced
4. How many assay interference mechanisms it triggers
5. Whether structural evidence from crystallography supports genuine binding

---

## Quick Reference

```
FORMULA:
  IMP Score = Base Score x QED Multiplier

  Base Score = 0.45 x Efficiency + 0.20 x Distance + 0.15 x Angle
             + 0.15 x Interference + 0.05 x PDB

  QED Multiplier = 0.75 + 0.25 x QED

WEIGHTS (sum to 100%, no renormalization):
  Efficiency:    45%  (SEI + BEI z-scores, sigmoid-normalized)
  Distance:      20%  (compound modulus / best modulus)
  Angle:         15%  (deviation from optimal 45 degrees)
  Interference:  15%  (scored flags / 5)
  PDB:            5%  (structure count + resolution quality)

CLASSIFICATION:
  0.90 - 1.00  Exceptional IMP   Priority 1  DEPRIORITIZE
  0.70 - 0.89  Strong IMP        Priority 2  VALIDATE with orthogonal assays
  0.50 - 0.69  Moderate IMP      Priority 3  MONITOR carefully
  0.30 - 0.49  Weak IMP          Priority 4  PROCEED with standard validation
  0.00 - 0.29  Not IMP           None        PRIORITIZE for development
```

---

## The Five Components

### Component 1: Efficiency Outlier Score (45% Weight)

**Purpose**: Measures how statistically exceptional the compound's efficiency metrics are compared to the similarity cohort. Compounds that are extreme outliers are more likely to be artifacts.

**Metrics used**: Only **SEI** (Surface Efficiency Index) and **BEI** (Binding Efficiency Index).

NSEI and NBEI are calculated and displayed but **not used in the score** to avoid redundancy - they are derived from the same underlying activity data. Since pActivity is the numerator in all four formulas (SEI, BEI, NSEI, NBEI), counting all four would effectively weight the same activity signal multiple times.

#### Why Only SEI and BEI?

| Metric | Normalizes By | Captures |
|--------|--------------|----------|
| SEI | Polar Surface Area (PSA) | Surface efficiency - how effectively the compound uses its polar surface |
| BEI | Molecular Weight (MW) | Binding efficiency - how effectively the compound uses its size |
| NSEI | N+O atom count (NPOL) | Atom-level polarity efficiency (redundant with SEI) |
| NBEI | Heavy atom count (NHA) | Atom-level size efficiency (redundant with BEI) |

SEI and BEI use continuous molecular descriptors (PSA, MW) which provide finer discrimination. NSEI and NBEI use integer atom counts which are more coarsely quantized. Using all four would over-represent the same underlying activity measurement.

#### Calculation Steps

For each metric (SEI, BEI):

```
Step 1: Z-score = (compound value - cohort mean) / cohort standard deviation
Step 2: Sigmoid = 1 / (1 + exp(-Z))
Step 3: Normalized = (Sigmoid - 0.5) x 2, clipped to [0, 1]
```

The final Efficiency Score is the **average** of the two normalized scores:

```
Efficiency Score = (Normalized_SEI + Normalized_BEI) / 2
```

#### The Sigmoid Transformation

```
Input Z-score                  Sigmoid Output              Normalized Score
     |                              |                            |
  <- negative --- 0 --- positive -> |  0.0 ------- 0.5 -------- 1.0
                                    |                            |
     Z = -3  ->  sigmoid = 0.047  ->  (0.047 - 0.5) x 2 = -0.91  ->  clipped to 0.00
     Z = -1  ->  sigmoid = 0.269  ->  (0.269 - 0.5) x 2 = -0.46  ->  clipped to 0.00
     Z =  0  ->  sigmoid = 0.500  ->  (0.500 - 0.5) x 2 =  0.00  ->  0.00
     Z = +1  ->  sigmoid = 0.731  ->  (0.731 - 0.5) x 2 =  0.46  ->  0.46
     Z = +2  ->  sigmoid = 0.881  ->  (0.881 - 0.5) x 2 =  0.76  ->  0.76
     Z = +3  ->  sigmoid = 0.953  ->  (0.953 - 0.5) x 2 =  0.91  ->  0.91
     Z = +4  ->  sigmoid = 0.982  ->  (0.982 - 0.5) x 2 =  0.96  ->  0.96
```

**Why sigmoid instead of hard clipping?** A compound with Z=4 gets a higher score than Z=3 (0.96 vs 0.91), preserving ranking information for exceptional outliers without arbitrary cutoffs. Hard clipping at Z=3 would treat all extreme outliers equally, losing the ability to distinguish between "very unusual" and "extremely unusual" compounds.

**Key behavior**: Negative Z-scores (compounds below the cohort mean) are clipped to 0. Only compounds above average contribute to the efficiency outlier score. This makes sense because IMPs are compounds that appear *too good*, not too poor.

#### Complete Sigmoid Reference Table

| Z-score | Sigmoid | Normalized | Meaning |
|---------|---------|------------|---------|
| -3 | 0.05 | 0.00 | Far below average (clipped) |
| -1 | 0.27 | 0.00 | Below average (clipped) |
| 0 | 0.50 | 0.00 | Average - baseline |
| +0.5 | 0.62 | 0.24 | Slightly above average |
| +1 | 0.73 | 0.46 | Above average |
| +1.5 | 0.82 | 0.64 | Notably above average |
| +2 | 0.88 | 0.76 | Good outlier (top ~2.3%) |
| +2.5 | 0.92 | 0.85 | Strong outlier |
| +3 | 0.95 | 0.91 | Very strong outlier (top ~0.1%) |
| +4 | 0.98 | 0.96 | Extreme outlier |
| +5 | 0.99 | 0.99 | Near-maximum |

#### Edge Cases

- **Zero standard deviation**: If all compounds in the cohort have identical metric values (std = 0), Z-scores are set to 0.0 for all compounds. This yields an Efficiency Score of 0.00 for every compound - nobody stands out.
- **Single compound in cohort**: The std will be NaN. Same treatment as zero std - all scores become 0.
- **NaN metrics**: Compounds with NaN for SEI or BEI will propagate NaN through the calculation.

#### Worked Example

Cohort of 50 compounds. For one compound:
- Compound SEI = 22.5, cohort mean SEI = 14.2, cohort std SEI = 3.8
- Compound BEI = 28.1, cohort mean BEI = 18.5, cohort std BEI = 4.2

```
SEI Z-score = (22.5 - 14.2) / 3.8 = 2.18
SEI sigmoid = 1 / (1 + exp(-2.18)) = 0.899
SEI normalized = (0.899 - 0.5) x 2 = 0.797

BEI Z-score = (28.1 - 18.5) / 4.2 = 2.29
BEI sigmoid = 1 / (1 + exp(-2.29)) = 0.908
BEI normalized = (0.908 - 0.5) x 2 = 0.816

Efficiency Score = (0.797 + 0.816) / 2 = 0.807
```

This compound scores 0.807 - a strong efficiency outlier in both dimensions.

**Source**: `backend/modules/imp_scoring.py` - `calculate_efficiency_outlier_score()`

---

### Component 2: Distance to Best-in-Class (20% Weight)

**Purpose**: Measures how close the compound is to the best-performing compound in the cohort on the SEI-BEI efficiency plane. The closer a compound is to the best-in-class, the more "suspicious" it becomes from an IMP perspective.

#### The Efficiency Plane

The SEI-BEI plane is a 2D space where each compound is plotted as a point:
- **X-axis**: SEI (Surface Efficiency Index)
- **Y-axis**: BEI (Binding Efficiency Index)

```
BEI
 ^
 |           * best (highest modulus)
 |         /
 |       /  <-- modulus = distance from origin
 |     * compound
 |   /
 | /
 +----------------------------> SEI
Origin (0,0)
```

The **modulus** (vector length) captures overall efficiency magnitude regardless of direction:

```
Modulus = sqrt(SEI^2 + BEI^2)
```

A compound with SEI=15 and BEI=20 has modulus = sqrt(225 + 400) = sqrt(625) = 25.0

#### Calculation

```
Distance Score = compound Modulus / max Modulus in cohort
```

The best compound in the cohort (highest modulus) scores 1.0. All other compounds are scored proportionally.

| Compound Modulus | Best Modulus | Distance Score | Interpretation |
|-----------------|-------------|---------------|----------------|
| 38.2 | 38.2 | 1.00 | Best-in-class |
| 30.5 | 38.2 | 0.80 | Close to best |
| 19.1 | 38.2 | 0.50 | Moderate distance |
| 9.6 | 38.2 | 0.25 | Far from best |

#### Why Modulus Instead of Euclidean Distance?

Using the modulus from the origin (rather than Euclidean distance from the best compound) ensures that "distance to best" captures *absolute efficiency magnitude*, not relative positioning. A compound at (5, 5) and one at (30, 30) are both on the 45-degree line but have very different moduli (7.1 vs 42.4). The one at (30, 30) is far more efficient overall.

#### Edge Cases

- **Best modulus is 0 or NaN**: All Distance Scores are set to 0.0. This happens only with completely invalid data.
- **All moduli identical**: Every compound scores 1.0 (everyone is "best-in-class").
- **Result clipped to [0, 1]**: The score is clipped after division to ensure it stays in range.

#### Worked Example

Cohort moduli: [38.2, 35.1, 28.9, 22.4, 18.7, 15.3, 12.1]

```
Compound modulus = 28.9
Best modulus = 38.2

Distance Score = 28.9 / 38.2 = 0.757
```

**Source**: `backend/modules/imp_scoring.py` - `calculate_distance_to_best_score()`
**Geometry**: `backend/modules/efficiency_planes.py` - `calculate_modulus()`

---

### Component 3: Development Angle Score (15% Weight)

**Purpose**: Evaluates whether the compound has a balanced development trajectory on the SEI-BEI efficiency plane. An imbalanced angle suggests the compound's apparent activity comes from one physicochemical dimension rather than balanced molecular optimization.

#### What the Angle Means

The angle describes the **direction** of the efficiency vector in the SEI-BEI plane:

```
BEI
 ^
 |         90 degrees: Pure BEI (polar compounds)
 |        /
 |       /  60 degrees: BEI-dominant
 |      /
 |     / 45 degrees: OPTIMAL (balanced)
 |    /
 |   /  30 degrees: SEI-dominant
 |  /
 | / 0 degrees: Pure SEI (hydrophobic compounds)
 +----------------------------> SEI
```

**Optimal angle = 45 degrees**: This means the compound improves both surface efficiency (SEI) and binding efficiency (BEI) equally. Balanced optimization is more likely to reflect genuine structure-activity relationships than one-dimensional improvement.

#### Physicochemical Interpretation

| Angle Range | Character | What It Means |
|-------------|-----------|---------------|
| 0-20 | Very hydrophobic | Activity driven by lipophilicity, not specific binding. High LogP, low PSA. Risk of promiscuous binding, membrane disruption, aggregation. |
| 20-30 | Moderately hydrophobic | SEI-dominant. Compound relies more on surface interactions than size optimization. |
| 30-40 | Slightly SEI-biased | Good balance with slight surface efficiency preference. |
| **40-50** | **Balanced (optimal)** | **Both dimensions contribute equally. Suggests genuine, specific binding.** |
| 50-60 | Slightly BEI-biased | Good balance with slight binding efficiency preference. |
| 60-70 | Moderately polar | BEI-dominant. Activity driven by polar interactions. May face permeability issues. |
| 70-90 | Very polar | Compound is very polar for its potency. Unlikely to cross membranes. May have poor oral bioavailability. |

#### Calculation

```
Angle = arctan2(BEI, SEI) x 180 / pi   (in degrees)

Angle Score = 1 - (|Angle - 45| / 45), clipped to [0, 1]
```

This creates a linear penalty that increases with deviation from 45 degrees. Deviations greater than 45 degrees (i.e., angle < 0 or angle > 90) are clipped to 0.

#### Score Table

| Angle | |Deviation| | Raw Score | Clipped | Quality |
|-------|------------|-----------|---------|---------|
| 0 | 45 | 0.00 | 0.00 | Poor (pure hydrophobic) |
| 10 | 35 | 0.22 | 0.22 | Poor |
| 20 | 25 | 0.44 | 0.44 | Fair |
| 30 | 15 | 0.67 | 0.67 | Good |
| 35 | 10 | 0.78 | 0.78 | Good |
| 40 | 5 | 0.89 | 0.89 | Excellent |
| **45** | **0** | **1.00** | **1.00** | **Perfect** |
| 50 | 5 | 0.89 | 0.89 | Excellent |
| 55 | 10 | 0.78 | 0.78 | Good |
| 60 | 15 | 0.67 | 0.67 | Good |
| 70 | 25 | 0.44 | 0.44 | Fair |
| 80 | 35 | 0.22 | 0.22 | Poor |
| 90 | 45 | 0.00 | 0.00 | Poor (pure polar) |

#### Compound Examples

| Compound Type | Typical Angle | Angle Score | Notes |
|--------------|---------------|-------------|-------|
| Well-optimized kinase inhibitor | 42-48 | 0.93-1.00 | Balanced polar/hydrophobic features |
| Lipophilic natural product | 15-25 | 0.22-0.44 | Relies on hydrophobicity |
| Highly polar peptide mimetic | 65-75 | 0.33-0.44 | Too many polar groups |
| Balanced fragment hit | 40-50 | 0.89-1.00 | Ideal starting point for optimization |

**Source**: `backend/modules/imp_scoring.py` - `calculate_angle_score()`
**Geometry**: `backend/modules/efficiency_planes.py` - `calculate_angle()`

---

### Component 4: Assay Interference Score (15% Weight)

**Purpose**: Quantifies how many known assay interference mechanisms the compound triggers. More flags = stronger evidence the compound is an artifact rather than a genuine bioactive.

#### Calculation

```
Interference Score = (number of scored flags triggered) / 5
```

The score is a simple fraction: 0/5 = 0.00 (clean), 5/5 = 1.00 (maximum interference risk).

#### Overview: 5 Scored Flags + 2 Display-Only

| # | Flag | Patterns | Scored | Detection Source |
|---|------|----------|--------|-----------------|
| 1 | PAINS | 480 | Yes | RDKit FilterCatalog |
| 2 | Aggregator | 4 criteria | Yes | Shoichet heuristics |
| 3 | Thiol-Reactive | 15 SMARTS | Yes | Dahlin et al. 2015 |
| 4 | Redox-Active | 10 SMARTS | Yes | Proj et al. 2022 |
| 5 | Fluorescence | 13 SMARTS | Yes | Su et al. 2015 |
| 6 | BRENK | 104 | No | RDKit FilterCatalog |
| 7 | NIH | varies | No | RDKit FilterCatalog |

#### Flag 1: PAINS (Pan-Assay Interference Substructures)

**What it detects**: Substructures that produce apparent activity across many unrelated assay types through non-specific mechanisms (covalent binding, redox cycling, chelation, membrane disruption, aggregation).

**Method**: RDKit's built-in `FilterCatalog.PAINS` containing **480 validated SMARTS patterns** from Baell & Holloway (2010). Patterns are organized into three families (PAINS_A, PAINS_B, PAINS_C) based on frequency of HTS interference.

**Mechanism**: PAINS compounds produce false positive readouts because they interact with the assay system itself (reporters, coupling enzymes, detection reagents) rather than the intended target.

**Reference**: Baell JB, Holloway GA. *J Med Chem*. 2010;53(7):2719-2740. DOI: [10.1021/jm901137j](https://doi.org/10.1021/jm901137j)

#### Flag 2: Aggregator (Colloidal Aggregation Risk)

**What it detects**: Compounds that form colloidal particles in aqueous solution. These particles non-specifically sequester proteins, producing apparent inhibition that disappears with detergent counter-screens.

**Method**: Published Shoichet laboratory heuristics. **ALL four criteria must be met simultaneously**:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Aromatic rings | >= 3 | Flat, hydrophobic surfaces promote stacking |
| Molecular weight | > 300 Da | Sufficient size to form stable aggregates |
| Rotatable bonds | <= 2 | Rigid molecules stack more readily |
| LogP | > 3 | Lipophilic compounds partition out of solution |

A compound that meets only 3 of 4 criteria does **not** trigger the flag. This conservative AND-logic minimizes false positives.

**Reference**: Irwin JJ, et al. *J Med Chem*. 2015;58(17):7076-7087. DOI: [10.1021/acs.jmedchem.5b01105](https://doi.org/10.1021/acs.jmedchem.5b01105)

#### Flag 3: Thiol-Reactive (Electrophilic Compounds)

**What it detects**: Electrophilic functional groups that covalently modify cysteine thiol groups (-SH) in proteins, causing non-specific enzyme inhibition in HTS assays.

**Method**: **15 validated SMARTS patterns** organized by reaction mechanism:

| Category | Patterns | Mechanism |
|----------|----------|-----------|
| **Michael acceptors** (5) | `michael_acceptor`, `acrylamide`, `acrylate`, `enone`, `maleimide` | 1,4-conjugate addition to alpha,beta-unsaturated carbonyls |
| **Acylating agents** (3) | `acyl_halide`, `anhydride`, `activated_ester` | Nucleophilic acyl substitution on activated carbonyls |
| **SN2 electrophiles** (2) | `epoxide`, `aziridine` | Ring-opening alkylation of strained 3-membered rings |
| **Schiff base formers** (1) | `aldehyde` | Imine formation with lysine amines and cysteine thiols |
| **Isocyanates** (2) | `isocyanate`, `isothiocyanate` | Nucleophilic addition to cumulated double bonds |
| **Other reactive** (2) | `vinyl_sulfone`, `sulfonyl_fluoride` | Michael addition and sulfonylation |

**Reference**: Dahlin JL, et al. *J Med Chem*. 2015;58(5):2091-2113. DOI: [10.1021/jm5019093](https://doi.org/10.1021/jm5019093)

#### Flag 4: Redox-Active (Redox Cycling Compounds)

**What it detects**: Compounds that generate hydrogen peroxide (H2O2) and reactive oxygen species (ROS) through redox cycling. These oxidize assay components and produce false readouts, particularly in HRP-coupled assays.

**Method**: **10 validated SMARTS patterns** organized by chemotype:

| Category | Patterns | Mechanism |
|----------|----------|-----------|
| **Quinones** (4) | `para_quinone`, `ortho_quinone`, `naphthoquinone`, `anthraquinone` | Two-electron reduction/oxidation cycling generates superoxide and H2O2 |
| **Catechols** (2) | `catechol`, `catechol_substituted` | Auto-oxidize to ortho-quinones in aerobic aqueous solution |
| **Hydroquinones** (1) | `hydroquinone` | Redox pair with para-quinones, cycles between reduced/oxidized forms |
| **Other redox** (3) | `hydroxylamine`, `nitroso`, `nitro_aromatic` | N-centered redox cycling |

**Why catechols matter**: Many natural products (e.g., quercetin, EGCG) contain catechol moieties. These auto-oxidize at physiological pH, generating H2O2 that produces false positive readouts in most HTS assay formats.

**Reference**: Proj M, et al. *Drug Discov Today*. 2022;27(6):1733-1742. DOI: [10.1016/j.drudis.2022.03.008](https://doi.org/10.1016/j.drudis.2022.03.008)

#### Flag 5: Fluorescence (Autofluorescent Scaffolds)

**What it detects**: Compounds containing fluorophore scaffolds that emit light in the same wavelength ranges used by fluorescence-based HTS assays (FP, FRET, AlphaScreen). These produce false signals that are misinterpreted as target activity.

**Method**: **13 validated SMARTS patterns** organized by scaffold family:

| Family | Patterns | Excitation Range |
|--------|----------|-----------------|
| **Coumarins** (3) | `coumarin`, `coumarin_keto`, `coumarin_7amino` | 340-405 nm |
| **Xanthenes** (3) | `xanthene`, `fluorescein_core`, `rhodamine_core` | 480-570 nm |
| **PAHs** (3) | `naphthalene`, `anthracene`, `pyrene` | 250-380 nm |
| **Stilbenes** (1) | `stilbene` | 300-350 nm |
| **Flavonoids** (2) | `flavone`, `flavonol` | 340-430 nm |
| **Acridines** (1) | `acridine` | 350-430 nm |

**Clinical relevance**: Fluorescence-based assays are among the most common HTS formats. Compounds with intrinsic fluorescence can produce apparent activity even with zero target binding. This is particularly problematic for natural product screening, where many scaffolds (flavones, coumarins, PAHs) are inherently fluorescent.

**Reference**: Su BH, et al. *J Chem Inf Model*. 2015;55(2):434-445. DOI: [10.1021/ci5007432](https://doi.org/10.1021/ci5007432)

#### Display-Only Flags (NOT Counted in Score)

**BRENK** (104 patterns): Identifies unwanted substructures including reactive groups, toxic groups, and groups with unfavorable pharmacokinetics. Uses RDKit `FilterCatalog.BRENK`.
Reference: Brenk R, et al. *ChemMedChem*. 2008;3(3):435-444. DOI: [10.1002/cmdc.200700139](https://doi.org/10.1002/cmdc.200700139)

**NIH**: Identifies problematic functional groups from NIH HTS screening campaigns. Uses RDKit `FilterCatalog.NIH`.
Reference: Jadhav A, et al. *J Med Chem*. 2010;53(1):37-51. DOI: [10.1021/jm901070c](https://doi.org/10.1021/jm901070c)

**Why BRENK and NIH are excluded from scoring**: Both catalogs contain overlapping patterns with the 5 scored flags. BRENK includes Michael acceptors and aldehydes (already covered by Thiol-Reactive), quinones (covered by Redox-Active), and additional patterns that are more about drug-likeness than assay interference. NIH similarly overlaps. Counting them would double-penalize compounds for the same structural features.

#### Score Examples

| Flags triggered | Score | Meaning |
|----------------|-------|---------|
| 0 / 5 | 0.00 | Clean compound - no known interference mechanisms |
| 1 / 5 | 0.20 | Minor concern - one interference pathway |
| 2 / 5 | 0.40 | Moderate concern - multiple interference mechanisms |
| 3 / 5 | 0.60 | Significant concern - likely artifact |
| 4 / 5 | 0.80 | Strong concern - very likely artifact |
| 5 / 5 | 1.00 | Maximum interference - almost certainly artifact |

#### Real-World Examples

| Compound | PAINS | Aggr. | Thiol | Redox | Fluor. | Score | Notes |
|----------|:-----:|:-----:|:-----:|:-----:|:------:|-------|-------|
| Quercetin | Yes | No | No | Yes | Yes | 0.60 | Catechol (redox) + flavonol (fluor.) + PAINS |
| Curcumin | Yes | No | Yes | No | Yes | 0.60 | Michael acceptor (thiol) + chromophore (fluor.) |
| Ibuprofen | No | No | No | No | No | 0.00 | Clean small molecule |
| EGCG | Yes | No | No | Yes | Yes | 0.60 | Multiple catechols + chromophore |

**Source**: `backend/modules/imp_scoring.py` - `calculate_interference_score()`
**Detection**: `backend/modules/assay_interference_filter.py`

---

### Component 5: PDB Structural Evidence (5% Weight)

**Purpose**: Checks whether the compound (or close structural analogs) have been experimentally observed bound to proteins in X-ray crystal structures. Structural evidence provides the strongest validation that a compound genuinely binds to biological targets rather than producing artifacts.

#### Why Only 5% Weight?

PDB evidence is the most reliable validation signal, but it receives only 5% weight because:
1. Many genuine drug leads have no PDB structures (especially novel compounds)
2. Well-studied interference compounds (e.g., quercetin) may have many PDB structures
3. The IMP score is primarily about detecting *artifacts*, not validating *binding*

The PDB component serves as a **tiebreaker** and **confidence modifier**, not a primary driver.

#### The PDB Pipeline (4 Steps)

**Step 1: Chemical Similarity Search**

Search the RCSB PDB for structures containing ligands similar to the query compound.

```
POST https://search.rcsb.org/rcsbsearch/v2/query

{
  "query": {
    "type": "terminal",
    "service": "chemical",
    "parameters": {
      "value": "<SMILES>",
      "type": "descriptor",
      "descriptor_type": "SMILES",
      "match_type": "graph-relaxed"    <-- structural similarity
    }
  },
  "return_type": "entry"
}
```

**Match type**: `graph-relaxed` finds compounds with the same core scaffold, allowing for minor structural variations (different substituents, stereochemistry). This is more permissive than exact match but still requires substantial structural similarity.

**Limit**: Results capped at 100 PDB entries per compound.

**Step 2: Batch Resolution Retrieval (GraphQL)**

Fetch resolution data for all matching structures in a single GraphQL query:

```graphql
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_entry_info {
      resolution_combined
    }
  }
}
```

This is **9.5x faster** than individual REST calls (validated: 64 PDB IDs in ~0.16s vs 12.8s+ with REST). Falls back to parallel REST if GraphQL fails.

**Step 3: Resolution Quality Classification**

Each structure is classified by its X-ray crystallographic resolution:

| Resolution | Quality | Stars | Multiplier | What It Means |
|-----------|---------|-------|------------|---------------|
| < 2.0 A | High | *** | 1.00 | Individual atoms clearly resolved. Ligand binding pose is reliable. |
| 2.0-3.0 A | Medium | ** | 0.75 | Main chain and large side chains visible. Ligand position generally reliable. |
| > 3.0 A | Poor | * | 0.50 | Only overall shape visible. Ligand placement uncertain. |
| N/A | Unknown | - | 0.00 | No resolution data (e.g., NMR structures). Not counted. |

**Step 4: Score Calculation**

```
Base Score = min(structures_with_resolution / 5, 1.0)

Quality Score = sum(quality_multipliers) / (num_structures_with_resolution x 1.0)

PDB Score = (Base Score + Quality Score) / 2
```

The score combines two factors:
- **Base Score**: Do enough structures exist? (saturates at 5 structures)
- **Quality Score**: How good are those structures? (weighted by resolution)

#### Score Calculation Examples

**Example A: Well-studied compound (8 structures: 4 high, 3 medium, 1 poor)**

```
Base Score = min(8/5, 1.0) = 1.0

Quality Score = (4 x 1.0 + 3 x 0.75 + 1 x 0.5) / (8 x 1.0)
             = (4.0 + 2.25 + 0.5) / 8.0
             = 6.75 / 8.0 = 0.844

PDB Score = (1.0 + 0.844) / 2 = 0.922
```

**Example B: Moderate evidence (3 structures: 1 high, 2 medium)**

```
Base Score = min(3/5, 1.0) = 0.6

Quality Score = (1 x 1.0 + 2 x 0.75) / (3 x 1.0)
             = 2.5 / 3.0 = 0.833

PDB Score = (0.6 + 0.833) / 2 = 0.717
```

**Example C: Weak evidence (1 poor structure)**

```
Base Score = min(1/5, 1.0) = 0.2

Quality Score = (1 x 0.5) / (1 x 1.0)
             = 0.5 / 1.0 = 0.500

PDB Score = (0.2 + 0.5) / 2 = 0.350
```

**Example D: No structures found**

```
PDB Score = 0.0  (no structures = no evidence)
```

#### Performance Optimization

| Operation | Method | Performance |
|-----------|--------|-------------|
| Similarity search | REST POST | 1-5 seconds per compound |
| Resolution fetch | GraphQL batch | ~0.16s for 64 IDs |
| Resolution fetch (fallback) | Parallel REST (5 workers) | ~12.8s for 64 IDs |
| Per-compound caching | `@lru_cache(maxsize=500)` | Instant for repeat queries |
| Multi-compound | `ThreadPoolExecutor(max_workers=5)` | Parallel across compounds |
| Retry logic | 2 retries with 0.5s backoff | Handles transient PDB API failures |
| Rate limiting | 5 requests/second | Respects RCSB PDB fair use policy |

#### When PDB Is Disabled

If `use_pdb=False` (e.g., for offline analysis or faster processing):
- PDB Score = 0.0 for all compounds
- PDB columns are populated with zeros/empty strings
- Maximum possible IMP Base Score becomes 0.95 (missing the 5% PDB contribution)
- After QED multiplier, maximum possible Final Score = 0.95 x QED_Multiplier

**Source**: `backend/modules/imp_scoring.py` - `calculate_pdb_evidence_score()`
**API Client**: `backend/modules/pdb_client.py`

---

## Final Score Calculation

### Step 1: Base Score (weights sum to 100%)

```
IMP Base Score = 0.45 x Efficiency_Score
               + 0.20 x Distance_Score
               + 0.15 x Angle_Score
               + 0.15 x Interference_Score
               + 0.05 x PDB_Score
```

No renormalization is needed because the weights already sum to 1.00.

### Step 2: QED Multiplier

```
QED Multiplier = 0.75 + 0.25 x QED
```

QED (Quantitative Estimate of Drug-likeness) ranges from 0 to 1, based on molecular weight, LogP, HBD, HBA, PSA, rotatable bonds, and aromatic rings.

| QED | Multiplier | Effect |
|-----|-----------|--------|
| 1.0 | 1.00 | No change (highly drug-like) |
| 0.8 | 0.95 | 5% reduction |
| 0.6 | 0.90 | 10% reduction |
| 0.4 | 0.85 | 15% reduction |
| 0.2 | 0.80 | 20% reduction |
| 0.0 | 0.75 | 25% reduction (floor) |

The 75% floor ensures that even compounds with QED=0 retain most of their IMP signal. QED can only reduce the score by up to 25%.

### Step 3: Final Score

```
IMP Final Score = IMP Base Score x QED Multiplier
```

### Individual Contributions

After QED is applied, the contribution of each component is:

```
Efficiency_Contribution  = 0.45 x Efficiency_Score  x QED_Multiplier
Distance_Contribution    = 0.20 x Distance_Score    x QED_Multiplier
Angle_Contribution       = 0.15 x Angle_Score       x QED_Multiplier
Interference_Contribution= 0.15 x Interference_Score x QED_Multiplier
PDB_Contribution         = 0.05 x PDB_Score         x QED_Multiplier

QED_Impact = IMP_Final_Score - IMP_Base_Score
```

**Source**: `backend/modules/imp_scoring.py` - `calculate_imp_score()`

---

## Score Interpretation

| Score | Classification | Priority | Action |
|-------|---------------|----------|--------|
| 0.90-1.00 | Exceptional IMP | 1 (Highest concern) | DEPRIORITIZE. Very high false positive risk. Do not pursue without orthogonal validation (SPR, ITC, MST). |
| 0.70-0.89 | Strong IMP | 2 (High concern) | VALIDATE before advancing. Require counter-screens and orthogonal assays before significant investment. |
| 0.50-0.69 | Moderate IMP | 3 (Moderate concern) | MONITOR carefully. Gather additional validation data before committing resources. |
| 0.30-0.49 | Weak IMP | 4 (Low concern) | PROCEED with standard validation. Low false positive risk - more likely genuine activity. |
| 0.00-0.29 | Not IMP | None (Best) | PRIORITIZE for development. Lowest false positive risk - likely genuine activity. |

**Key principle**: IMP = Invalid Metabolic Panacea = FALSE POSITIVE indicator. Higher score = higher chance the compound is an artifact, not a genuine drug lead.

**Source**: `backend/modules/imp_scoring.py` - `interpret_imp_score()`

---

## Step-by-Step Calculation Example

**Compound**: Hypothetical kinase inhibitor in a cohort of 50 similar compounds.

### Input Data

| Property | Value |
|----------|-------|
| SEI | 18.5 |
| BEI | 22.3 |
| NSEI | 1.8 (display only) |
| NBEI | 0.35 (display only) |
| Angle_SEI_BEI | 50.3 degrees |
| Modulus_SEI_BEI | 28.97 |
| Best modulus in cohort | 38.2 |
| QED | 0.72 |
| PAINS | No |
| Aggregator | No |
| Thiol | Yes (Michael acceptor) |
| Redox | No |
| Fluorescence | No |
| PDB structures found | 8 (4 high, 3 medium, 1 poor) |

### Step 1: Component Scores

**Efficiency**: Assume SEI z-score = 1.5 and BEI z-score = 1.8
- SEI normalized: (sigmoid(1.5) - 0.5) x 2 = (0.818 - 0.5) x 2 = 0.636
- BEI normalized: (sigmoid(1.8) - 0.5) x 2 = (0.858 - 0.5) x 2 = 0.716
- Efficiency Score = (0.636 + 0.716) / 2 = **0.676**

**Distance**: 28.97 / 38.2 = **0.758**

**Angle**: 1 - (|50.3 - 45| / 45) = 1 - (5.3 / 45) = 1 - 0.118 = **0.882**

**Interference**: 1 flag (Thiol) / 5 = **0.200**

**PDB**:
- Base = min(8/5, 1.0) = 1.0
- Quality = (4x1.0 + 3x0.75 + 1x0.5) / 8 = 6.75/8 = 0.844
- PDB Score = (1.0 + 0.844) / 2 = **0.922**

### Step 2: Base Score

```
Base = 0.45 x 0.676 + 0.20 x 0.758 + 0.15 x 0.882 + 0.15 x 0.200 + 0.05 x 0.922
     = 0.304 + 0.152 + 0.132 + 0.030 + 0.046
     = 0.664
```

### Step 3: QED Multiplier & Final Score

```
QED Multiplier = 0.75 + 0.25 x 0.72 = 0.93

IMP Final Score = 0.664 x 0.93 = 0.618
```

### Result

**Score: 0.618** = Moderate IMP (Priority 3). Monitor carefully and gather additional evidence. The single thiol flag and strong PDB evidence are balanced - the compound warrants standard validation.

---

## Efficiency Metrics Reference

All four metrics are calculated for every bioactivity measurement. SEI and BEI are used in scoring; NSEI and NBEI are displayed for reference.

### Formulas (IMPs 2.0, Reddy et al.)

| Metric | Formula | Used in Score |
|--------|---------|:-------------:|
| SEI | pActivity / (PSA / 100) | Yes |
| BEI | pActivity / (MW / 1000) | Yes |
| NSEI | pActivity / NPOL | No (display only) |
| NBEI | pActivity / NHA | No (display only) |

Where:
- **pActivity** = -log10(activity in Molar). IC50 = 10 nM -> pActivity = 8.0
- **PSA** = Polar Surface Area (A^2)
- **MW** = Molecular Weight (Da)
- **NPOL** = Count of nitrogen + oxygen atoms
- **NHA** = Number of Heavy Atoms (non-hydrogen)

### Typical Ranges

| Metric | Poor | Average | Good | Excellent |
|--------|------|---------|------|-----------|
| SEI | < 5 | 5-15 | 15-25 | > 25 |
| BEI | < 10 | 10-20 | 20-30 | > 30 |
| NSEI | < 0.5 | 0.5-1.5 | 1.5-2.5 | > 2.5 |
| NBEI | < 0.15 | 0.15-0.3 | 0.3-0.45 | > 0.45 |

### Plane Geometry

Two efficiency planes are computed:

**SEI-BEI Plane** (used in scoring):
- Modulus = sqrt(SEI^2 + BEI^2) - overall efficiency magnitude
- Angle = arctan2(BEI, SEI) x 180/pi - development trajectory
- Slope = 10 x (PSA / MW) - physicochemical balance

**NSEI-NBEI Plane** (display only):
- Modulus = sqrt(NSEI^2 + NBEI^2)
- Angle = arctan2(NBEI, NSEI) x 180/pi
- Slope = NPOL / NHA
- Intercept = log10(NHA)

**Source**: `backend/modules/efficiency_metrics.py`, `backend/modules/efficiency_planes.py`

---

## Assay Interference Flags Reference

### Detection Methods

**PAINS** (Pan-Assay Interference Substructures)
- 480 SMARTS patterns from Baell & Holloway (2010)
- Detects compounds showing activity in many assays due to interference rather than genuine binding
- Source: RDKit FilterCatalog.PAINS

**Aggregator** (Colloidal Aggregation Risk)
- Published Shoichet laboratory heuristics
- ALL four criteria must be met: >=3 aromatic rings AND >300 Da AND <=2 rotatable bonds AND LogP >3
- Source: Irwin et al. (2015)

**Thiol-Reactive** (Electrophilic Compounds)
- 15 SMARTS patterns: Michael acceptors, acylating agents, SN2 electrophiles, Schiff base formers, isocyanates, vinyl sulfones
- Detects covalent modification of cysteine residues
- Source: Dahlin et al. (2015)

**Redox-Active** (Redox Cycling Compounds)
- 10 SMARTS patterns: quinones (para, ortho, naphthoquinone, anthraquinone), catechols, hydroquinones, hydroxylamines, nitroso, nitroaromatics
- Detects H2O2/ROS generation via redox cycling
- Source: Proj et al. (2022), Baell & Holloway (2010)

**Fluorescence** (Autofluorescent Scaffolds)
- 13 SMARTS patterns: coumarins, xanthenes, fluorescein, rhodamine, PAHs (naphthalene, anthracene, pyrene), stilbenes, flavonoids, acridines
- Detects compounds that interfere with fluorescence-based assays
- Source: Su et al. (2015)

### Display-Only Flags

**BRENK**: 104 unwanted substructures (reactive groups, toxic groups, unfavorable PK). Reference: Brenk et al. (2008).

**NIH**: Problematic functional groups from NIH screening. Reference: Jadhav et al. (2010).

These are shown in the UI but excluded from the interference score to avoid double-counting with the 5 scored flags.

**Source**: `backend/modules/assay_interference_filter.py`

---

## Red Flags & Decision Guide

### Warning Patterns

**High Efficiency + Low PDB**: Exceptional efficiency metrics without structural validation. The activity may be an artifact from aggregation, fluorescence interference, or redox cycling. Require orthogonal assay validation.

**Extreme Angle (<20 or >70 degrees)**: Unbalanced development trajectory. <20 degrees = too hydrophobic (relies on lipophilicity, promiscuous binding risk). >70 degrees = too polar (permeability issues, may not cross membranes).

**High Score + Low QED**: Strong IMP characteristics combined with poor drug-likeness (too large, too lipophilic, too many H-bond donors/acceptors). Even if validated, may face development challenges.

**Multi-target Activity**: Compounds active against many unrelated targets (>10) are almost always artifacts, not genuine polypharmacology. Classic examples: curcumin, quercetin, EGCG.

### Decision Flowchart

```
START: Compound has IMP Score
  |
  +-- Score >= 0.7?
  |     YES -> VALIDATE with orthogonal assays before any investment
  |     NO  -> Continue
  |
  +-- Score 0.5-0.7?
  |     YES -> Check PDB Score
  |              PDB >= 0.5? -> Likely genuine, PROCEED with monitoring
  |              PDB < 0.5?  -> VALIDATE, structural evidence is weak
  |     NO  -> Continue
  |
  +-- Score < 0.5?
        YES -> Check interference flags
                 0 flags? -> PRIORITIZE, clean compound
                 1-2 flags? -> PROCEED with standard validation
                 3+ flags? -> VALIDATE despite low score
```

### Pre-advancement Checklist

Before advancing any compound, verify:

- [ ] IMP Score < 0.7 (or validated with orthogonal assays if higher)
- [ ] PDB Score >= 0.5 (or structural validation planned)
- [ ] Angle Score >= 0.6 (balanced development trajectory)
- [ ] QED >= 0.5 (or acceptable for target class)
- [ ] Interference flags < 3 (or counter-screens completed)
- [ ] Activity against <= 5 related targets (not pan-active)

---

## Glossary

### Core Metrics

| Term | Definition |
|------|-----------|
| pActivity | -log10(activity in Molar). Higher = more potent. IC50=10nM -> pActivity=8.0 |
| SEI | Surface Efficiency Index = pActivity / (PSA/100) |
| BEI | Binding Efficiency Index = pActivity / (MW/1000) |
| NSEI | Normalized SEI = pActivity / NPOL (display only in scoring) |
| NBEI | Normalized BEI = pActivity / NHA (display only in scoring) |
| QED | Quantitative Estimate of Drug-likeness (0-1) |

### Geometric Terms

| Term | Definition |
|------|-----------|
| Modulus | Vector length: sqrt(SEI^2 + BEI^2). Overall efficiency magnitude. |
| Angle | Vector direction: arctan2(BEI, SEI). Development trajectory. 45 degrees = optimal. |
| Slope | Physicochemical balance: 10 x (PSA / MW) for SEI-BEI plane. |

### Structural Biology

| Term | Definition |
|------|-----------|
| RCSB PDB | Protein Data Bank - repository of 3D protein structures |
| Resolution | Crystal structure quality in Angstroms. Lower = better. <2.0 A = excellent. |
| GraphQL | Batch query API for PDB (9.5x faster than individual REST calls) |

### Problem Compound Terms

| Term | Definition |
|------|-----------|
| IMP | Invalid Metabolic Panacea. Compound appearing active but is an assay artifact. |
| PAINS | Pan-Assay Interference Substructures. Interfere with many assay types. |
| Aggregator | Forms colloidal particles that non-specifically sequester proteins. |
| Michael Acceptor | Reactive electrophile that forms covalent bonds with protein cysteines. |
| Catechol | 1,2-dihydroxybenzene. Redox-active, known to cause assay interference. |

---

## References

### Primary Methodology

- Reddy AS, et al. "IMPs 2.0" - Ligand efficiency metrics and scoring methodology

### Assay Interference Detection

1. Baell JB, Holloway GA. New substructure filters for removal of pan assay interference compounds (PAINS). *J Med Chem*. 2010;53(7):2719-2740. DOI: [10.1021/jm901137j](https://doi.org/10.1021/jm901137j)

2. Irwin JJ, et al. An Aggregation Advisor for Ligand Discovery. *J Med Chem*. 2015;58(17):7076-7087. DOI: [10.1021/acs.jmedchem.5b01105](https://doi.org/10.1021/acs.jmedchem.5b01105)

3. Dahlin JL, et al. PAINS in the Assay: Chemical Mechanisms of Assay Interference and Promiscuous Enzymatic Inhibition. *J Med Chem*. 2015;58(5):2091-2113. DOI: [10.1021/jm5019093](https://doi.org/10.1021/jm5019093)

4. Proj M, et al. Redox active or thiol reactive? Optimization of rapid screens to identify less evident nuisance compounds. *Drug Discov Today*. 2022;27(6):1733-1742. DOI: [10.1016/j.drudis.2022.03.008](https://doi.org/10.1016/j.drudis.2022.03.008)

5. Su BH, et al. Rule-based classification models of molecular autofluorescence. *J Chem Inf Model*. 2015;55(2):434-445. DOI: [10.1021/ci5007432](https://doi.org/10.1021/ci5007432)

6. Brenk R, et al. Lessons learnt from assembling screening libraries for drug discovery for neglected diseases. *ChemMedChem*. 2008;3(3):435-444. DOI: [10.1002/cmdc.200700139](https://doi.org/10.1002/cmdc.200700139)

7. Jadhav A, et al. Quantitative analyses of aggregation, autofluorescence, and reactivity artifacts. *J Med Chem*. 2010;53(1):37-51. DOI: [10.1021/jm901070c](https://doi.org/10.1021/jm901070c)

### Structural Biology

8. RCSB PDB Search API v2: https://search.rcsb.org/index.html
9. RCSB PDB GraphQL API: https://data.rcsb.org/graphql

---

## Implementation

- Backend: `backend/modules/imp_scoring.py`
- Frontend: `frontend/ui/pages/compound_detail.py`
- Documentation: `IMP_Score.md`

---

*Document Version: 1.0.0*
*Last Updated: February 2026*
*Based on actual implementation in `backend/modules/imp_scoring.py`*