# O[Q/P/L]A Scoring Methodology

## Overall Quality/Promise/Likelihood Assessment for Invalid Metabolic Panaceas (IMPs)

**Document Version**: 2.0
**Date**: November 2025
**System**: IMPULATOR-3 (IMPs Navigator)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scoring Philosophy](#scoring-philosophy)
3. [Phase 1 Components (65% Total Weight)](#phase-1-components)
4. [Phase 2 Components (80% Total Weight)](#phase-2-components)
5. [QED Multiplier](#qed-multiplier)
6. [Score Interpretation & Classification](#score-interpretation--classification)
7. [Complete Calculation Example](#complete-calculation-example)
8. [Implementation Details](#implementation-details)
9. [References](#references)

---

## Executive Summary

The **O[Q/P/L]A Score** (Overall Quality/Promise/Likelihood Assessment) is a multi-criteria scoring system designed to identify and prioritize Invalid Metabolic Panaceas (IMPs) - natural product compounds showing anomalously high efficiency across multiple bioactivity assays.

### Key Features:
- **Multi-component**: Integrates 4 independent validation streams (Phase 2)
- **Weighted scoring**: Components have different importance (40%, 15%, 15%, 10%, etc.)
- **Drug-likeness filter**: QED multiplier ensures compounds are developable
- **Interpretable**: Provides clear classification and priority levels
- **Extensible**: Designed for future phases (target prediction, analog support)

### Final Score Range:
- **0.0 - 1.0**: Higher scores indicate stronger IMP candidates
- **Threshold**: Scores ≥ 0.5 are considered "Moderate IMP" or better

---

## Scoring Philosophy

### Why Multi-Criteria Scoring?

IMPs are compounds that show **exceptional efficiency outliers** across multiple unrelated targets. However, not all efficiency outliers are valid IMPs. Many can be:
- **Assay artifacts** (PAINS, aggregators, fluorescence interference)
- **Promiscuous binders** (non-specific interactions)
- **Poor drug candidates** (violate Lipinski rules, high toxicity risk)

The O[Q/P/L]A score addresses this by requiring **convergent evidence** from multiple independent sources:

1. **Efficiency Outlier**: Is the compound statistically exceptional?
2. **Development Trajectory**: Does it have balanced physicochemical properties?
3. **Competitive Position**: Is it among the best in its cohort?
4. **Structural Evidence**: Do experimental structures validate binding?
5. **Drug-likeness**: Can it be developed into a drug?

### Scoring Principles:

1. **No single criterion dominates**: All components contribute
2. **Missing data is penalized**: If PDB data unavailable, weight shifts to other components
3. **Transparent**: Users can see contribution of each component
4. **Conservative**: Exceptional scores (>0.9) require strong evidence across ALL criteria

---

## Phase 1 Components

Phase 1 uses 3 components accounting for **65% of total weight**. When Phase 2 (PDB) is unavailable, these are renormalized to 100%.

### Component 1: Efficiency Outlier Score (40% weight)

**Purpose**: Quantify how exceptional the compound's efficiency is compared to the cohort.

**Method**: Z-score normalization across 4 efficiency metrics

#### Step 1: Calculate Efficiency Metrics

Four ligand efficiency metrics are calculated for each bioactivity:

##### 1.1. SEI (Surface Efficiency Index)
```
SEI = pActivity / (PSA / 100)
```

- **pActivity**: -log₁₀(Activity in M) - higher is better
- **PSA**: Polar Surface Area (Ų) - topological measure
- **Interpretation**: Binding efficiency relative to polar surface area
- **Typical range**: 5-30 (good IMPs: >20)

**Example**:
```
pActivity = 7.5 (activity = 31.6 nM)
PSA = 85.2 Ų
SEI = 7.5 / (85.2 / 100) = 8.80
```

##### 1.2. BEI (Binding Efficiency Index)
```
BEI = pActivity / (MW / 1000)
```

- **MW**: Molecular Weight (Da)
- **Interpretation**: Binding efficiency relative to molecular size
- **Typical range**: 10-40 (good IMPs: >25)

**Example**:
```
pActivity = 7.5
MW = 342.1 Da
BEI = 7.5 / (342.1 / 1000) = 21.92
```

##### 1.3. NSEI (Normalized Surface Efficiency Index)
```
NSEI = pActivity / NPOL
```

- **NPOL**: Count of N + O atoms (polarity measure)
- **Interpretation**: Binding efficiency per polar atom
- **Typical range**: 1-3 (good IMPs: >2)

**Example**:
```
pActivity = 7.5
NPOL = 5 (nitrogen + oxygen atoms)
NSEI = 7.5 / 5 = 1.50
```

##### 1.4. NBEI (Normalized Binding Efficiency Index)
```
NBEI = pActivity / NHA
```

- **NHA**: Number of Heavy Atoms (non-hydrogen atoms)
- **Interpretation**: Binding efficiency per heavy atom
- **Typical range**: 0.2-0.5 (good IMPs: >0.35)

**Example**:
```
pActivity = 7.5
NHA = 24 heavy atoms
NBEI = 7.5 / 24 = 0.3125
```

#### Step 2: Z-Score Normalization

For each metric, calculate the Z-score relative to the cohort:

```
Z = (value - mean) / std_dev
```

**Special cases**:
- If `std_dev = 0` (all values identical): Z = 0
- If `std_dev = NaN`: Z = 0

#### Step 3: Normalize to [0, 1]

```
Normalized_Score = (Z / 3.0).clip(0, 1)
```

**Rationale**: Z-scores beyond ±3σ are extremely rare (99.7% of data within ±3σ). We clip at 3 to prevent extreme outliers from dominating.

#### Step 4: Average Across Metrics

```
Efficiency_Score = mean(SEI_norm, BEI_norm, NSEI_norm, NBEI_norm)
```

**Interpretation**:
- **0.0-0.3**: Below average (not an IMP)
- **0.3-0.5**: Average (borderline IMP)
- **0.5-0.7**: Above average (potential IMP)
- **0.7-0.9**: Exceptional (strong IMP candidate)
- **0.9-1.0**: Extreme outlier (validate - possible artifact)

---

### Component 2: Development Angle Score (10% weight)

**Purpose**: Assess if the compound has balanced physicochemical properties.

#### Concept: Efficiency Plane Geometry

Compounds can be visualized in 2D "efficiency space":
- **X-axis**: Surface efficiency (SEI or NSEI)
- **Y-axis**: Binding efficiency (BEI or NBEI)

Each compound is a vector: `v = [SEI, BEI]`

The **angle** of this vector indicates the development trajectory:

```
Angle = arctan2(BEI, SEI) × 180/π
```

**Interpretation**:
- **0°**: Pure surface efficiency improvement (hydrophobic)
- **45°**: Balanced improvement (OPTIMAL)
- **90°**: Pure binding efficiency improvement (polar)

#### Scoring Formula

```
Angle_Score = 1 - |Angle - 45°| / 45°
```

This formula penalizes deviations from the optimal 45° angle.

**Angle Score Examples**:

| Angle | Deviation | Score | Interpretation |
|-------|-----------|-------|----------------|
| 45° | 0° | 1.00 | Perfect balance |
| 40° | 5° | 0.89 | Excellent |
| 30° | 15° | 0.67 | Good |
| 20° | 25° | 0.44 | Fair (too hydrophobic) |
| 70° | 25° | 0.44 | Fair (too polar) |
| 10° | 35° | 0.22 | Poor (highly hydrophobic) |
| 80° | 35° | 0.22 | Poor (highly polar) |

**Why 45° is optimal**:
- Balanced improvement in both size reduction AND polarity optimization
- Compounds at extremes (<20° or >70°) are harder to optimize
- Historical analysis shows successful drugs cluster around 30-60°

---

### Component 3: Distance to Best-in-Class Score (15% weight)

**Purpose**: Measure how close the compound is to the best-performing compound in the cohort.

#### Modulus Calculation

First, calculate the **modulus** (vector magnitude) in efficiency space:

```
Modulus = sqrt(SEI² + BEI²)
```

This represents the **overall efficiency magnitude** - a compound's total efficiency in 2D space.

**Example**:
```
SEI = 8.80
BEI = 21.92
Modulus = sqrt(8.80² + 21.92²) = 23.63
```

#### Scoring Formula

```
Distance_Score = Compound_Modulus / Best_Modulus
```

Where `Best_Modulus = max(all moduli in cohort)`

**Interpretation**:
- **1.00**: This compound IS the best
- **0.90-0.99**: Very close to best (top tier)
- **0.70-0.89**: Competitive (second tier)
- **0.50-0.69**: Moderate distance (third tier)
- **<0.50**: Far from best (lower tier)

**Example**:
```
Compound_Modulus = 23.63
Best_Modulus = 28.45
Distance_Score = 23.63 / 28.45 = 0.83 (competitive)
```

---

### Phase 1 Weighted Score

```
OQPLA_Base = w1 × Efficiency_Score + w2 × Angle_Score + w3 × Distance_Score
```

#### Weight Options:

**Option A: Normalized weights (Phase 1 only)** - Used when PDB unavailable:
```
Total = 0.40 + 0.10 + 0.15 = 0.65
w1 = 0.40 / 0.65 = 0.615 (61.5%)
w2 = 0.10 / 0.65 = 0.154 (15.4%)
w3 = 0.15 / 0.65 = 0.231 (23.1%)
```

**Option B: Original weights** - Used in development/testing:
```
w1 = 0.40 (40%)
w2 = 0.10 (10%)
w3 = 0.15 (15%)
```
(Results in max base score = 0.65)

**IMPULATOR-3 uses Option A** for Phase 1 implementation.

---

## Phase 2 Components

Phase 2 adds **Component 4: PDB Structural Evidence** (15% weight), bringing total to **80%**.

### Component 4: PDB Structural Evidence Score (15% weight)

**Purpose**: Validate compound binding with experimental structural data from RCSB Protein Data Bank.

**Why PDB Evidence Matters**:
- Confirms compound actually binds to proteins
- Validates bioactivity data (not just assay artifacts)
- Provides mechanism-of-action insights
- High-quality structures (< 2.0 Å) indicate reliable binding modes

#### Data Collection

For each compound SMILES, query RCSB PDB API to find:
1. Exact matches (same SMILES)
2. Similar compounds (≥90% Tanimoto similarity)

Extract:
- **PDB IDs**: List of structure identifiers
- **Resolution**: X-ray crystallography resolution (Ångströms)
- **Method**: Experimental method (X-RAY, NMR, CRYO-EM)

#### Quality Classification

Structures are binned by resolution quality:

| Quality Class | Resolution Range | Multiplier | Interpretation |
|---------------|------------------|------------|----------------|
| ⭐⭐⭐ High | < 2.0 Å | 1.0 | Atomic detail visible |
| ⭐⭐ Medium | 2.0 - 3.0 Å | 0.75 | Good structural detail |
| ⭐ Poor | > 3.0 Å | 0.5 | Limited detail |

**Why resolution matters**:
- **< 2.0 Å**: Individual atoms clearly resolved, high confidence in binding mode
- **2.0-3.0 Å**: Overall structure clear, some uncertainty in side chains
- **> 3.0 Å**: Coarse structure only, binding mode uncertain

#### Scoring Formula

```python
# Step 1: Base score (quantity)
base_score = min(num_structures / 5.0, 1.0)

# Step 2: Quality-weighted score
quality_score = (
    num_high_quality * 1.0 +
    num_medium_quality * 0.75 +
    num_poor_quality * 0.5
) / num_structures

# Step 3: Final PDB score (average of quantity and quality)
PDB_Score = (base_score + quality_score) / 2.0
```

**Rationale**:
- **5 structures = 1.0 base score**: Sufficient statistical evidence
- **Quality weighting**: High-resolution structures count more
- **Average**: Balance between quantity and quality

**Example Calculations**:

**Example 1: High quantity, high quality**
```
Structures found: 8
High quality (< 2.0 Å): 6
Medium quality (2.0-3.0 Å): 2
Poor quality (> 3.0 Å): 0

base_score = min(8/5, 1.0) = 1.0
quality_score = (6×1.0 + 2×0.75 + 0×0.5) / 8 = 7.5/8 = 0.9375
PDB_Score = (1.0 + 0.9375) / 2 = 0.969
```

**Example 2: Low quantity, medium quality**
```
Structures found: 2
High quality: 0
Medium quality: 2
Poor quality: 0

base_score = min(2/5, 1.0) = 0.4
quality_score = (0×1.0 + 2×0.75 + 0×0.5) / 2 = 0.75
PDB_Score = (0.4 + 0.75) / 2 = 0.575
```

**Example 3: No structures**
```
Structures found: 0
PDB_Score = 0.0
```

#### Additional PDB Metrics Exported

- `PDB_Num_Structures`: Total count
- `PDB_High_Quality`: Count of < 2.0 Å structures
- `PDB_Medium_Quality`: Count of 2.0-3.0 Å structures
- `PDB_Poor_Quality`: Count of > 3.0 Å structures
- `PDB_IDs`: Comma-separated list of all PDB IDs
- `PDB_Best_Resolution`: Lowest (best) resolution found

---

### Phase 2 Weighted Score

```
OQPLA_Base = w1 × Efficiency_Score +
             w2 × Angle_Score +
             w3 × Distance_Score +
             w4 × PDB_Score
```

**Phase 2 Weights** (with PDB enabled):
```
Total = 0.40 + 0.10 + 0.15 + 0.15 = 0.80
w1 = 0.40 / 0.80 = 0.500 (50.0%)
w2 = 0.10 / 0.80 = 0.125 (12.5%)
w3 = 0.15 / 0.80 = 0.1875 (18.75%)
w4 = 0.15 / 0.80 = 0.1875 (18.75%)
```

**Note**: When PDB is enabled, weights shift to account for the 4th component.

---

## QED Multiplier

### Purpose: Drug-Likeness Filter

Even if a compound has exceptional efficiency and structural evidence, it may be undevelopable due to poor drug-like properties. The **QED (Quantitative Estimate of Drug-likeness)** multiplier addresses this.

### What is QED?

QED is a continuous measure of "drug-likeness" ranging from 0 (non-drug-like) to 1 (highly drug-like).

QED integrates 8 molecular properties:
1. Molecular weight (MW)
2. LogP (lipophilicity)
3. Hydrogen bond donors (HBD)
4. Hydrogen bond acceptors (HBA)
5. Polar surface area (PSA)
6. Rotatable bonds
7. Aromatic rings
8. Structural alerts

**Calculation**: See Bickerton et al., Nature Chemistry (2012)

QED is calculated using RDKit's `Descriptors.qed()` function.

### QED Multiplier Formula

```
QED_Multiplier = 0.5 + 0.5 × QED
```

**Why this formula?**
- **Ensures range [0.5, 1.0]**: Compounds with QED=0 still get 50% credit
- **Rewards high QED**: Compounds with QED=1.0 get full (100%) credit
- **Balanced penalty**: Poor drug-likeness reduces score by max 50%

**QED Multiplier Examples**:

| QED | Multiplier | Interpretation |
|-----|------------|----------------|
| 1.0 | 1.00 | Excellent drug-likeness (no penalty) |
| 0.8 | 0.90 | Good drug-likeness (10% reduction) |
| 0.6 | 0.80 | Moderate drug-likeness (20% reduction) |
| 0.4 | 0.70 | Fair drug-likeness (30% reduction) |
| 0.2 | 0.60 | Poor drug-likeness (40% reduction) |
| 0.0 | 0.50 | Very poor drug-likeness (50% reduction) |

### Final O[Q/P/L]A Score

```
OQPLA_Final = OQPLA_Base × QED_Multiplier
```

**Example**:
```
OQPLA_Base = 0.85 (strong IMP candidate)
QED = 0.6 (moderate drug-likeness)
QED_Multiplier = 0.5 + 0.5 × 0.6 = 0.80

OQPLA_Final = 0.85 × 0.80 = 0.68 (still strong, but reduced)
```

**Impact Analysis**:
```
QED_Impact = OQPLA_Final - OQPLA_Base
           = 0.68 - 0.85 = -0.17
```

A negative impact indicates QED reduced the score (as expected for QED < 1.0).

---

## Score Interpretation & Classification

### Classification Thresholds

| Score Range | Classification | Priority | Action |
|-------------|----------------|----------|--------|
| 0.90 - 1.00 | **Exceptional IMP** | Priority 1 | Immediate experimental validation |
| 0.70 - 0.89 | **Strong IMP** | Priority 2 | Validate within 1 month |
| 0.50 - 0.69 | **Moderate IMP** | Priority 3 | Monitor and gather more data |
| 0.30 - 0.49 | **Weak IMP** | Priority 4 | Deprioritize unless novel scaffold |
| 0.00 - 0.29 | **Not IMP** | None | Exclude - do not pursue |

### Interpretation Guidelines

#### Exceptional IMP (0.90-1.00)
**Characteristics**:
- Top 5% efficiency outlier (Efficiency_Score > 0.85)
- Optimal development trajectory (Angle_Score > 0.85)
- Best or near-best in cohort (Distance_Score > 0.90)
- Strong PDB evidence if available (PDB_Score > 0.80)
- Good drug-likeness (QED > 0.70)

**Validation**:
- Confirm bioactivity with orthogonal assays
- Check for assay interference (PAINS, aggregation)
- Literature search for prior art
- Consider for hit-to-lead optimization

**Risk**: Scores >0.95 may indicate:
- Assay artifacts (despite PDB evidence)
- Data quality issues (outliers, errors)
- ALWAYS manually verify before investment

---

#### Strong IMP (0.70-0.89)
**Characteristics**:
- Top 10-20% efficiency outlier
- Good development trajectory
- Competitive in cohort
- Some PDB evidence or strong efficiency
- Reasonable drug-likeness

**Validation**:
- Dose-response curves (confirm potency)
- Counter-screening (specificity)
- Assess synthetic accessibility
- Patent landscape review

**Use Cases**:
- Lead optimization campaigns
- Scaffold hopping starting points
- Polypharmacology probes

---

#### Moderate IMP (0.50-0.69)
**Characteristics**:
- Above-average efficiency
- Some validation gaps
- May lack PDB evidence
- Moderate drug-likeness

**Validation**:
- Additional bioactivity data needed
- Structure determination (if not available)
- Analog searching (SAR support)

**Use Cases**:
- Chemical probe development
- Tool compounds for biology
- Low-priority screening hits

---

#### Weak IMP (0.30-0.49)
**Characteristics**:
- Marginal efficiency outlier
- Poor development trajectory OR
- Lacks structural/drug-likeness validation

**Recommendation**:
- Deprioritize unless:
  - Novel chemotype (new scaffold)
  - Underexplored target
  - Mechanistic interest

---

#### Not IMP (0.00-0.29)
**Characteristics**:
- Below-average efficiency
- Likely assay artifact
- Poor drug-likeness
- No validation

**Recommendation**: Exclude from further study

---

## Complete Calculation Example

Let's walk through a complete O[Q/P/L]A calculation for a hypothetical compound.

### Input Data

**Compound**: Natural Product X
**ChEMBL ID**: CHEMBL123456
**SMILES**: `CC(C)Cc1ccc(C(C)C(=O)O)cc1` (Ibuprofen for demo)

**Bioactivity**:
- Activity: 31.6 nM (pActivity = 7.5)

**Molecular Properties**:
- MW: 206.28 Da
- PSA: 37.3 Ų
- NPOL: 2 (N+O atoms)
- NHA: 15 (heavy atoms)
- QED: 0.72

**Cohort Statistics** (example dataset):
- Mean SEI: 15.2, StdDev: 4.8
- Mean BEI: 18.5, StdDev: 5.2
- Mean NSEI: 1.8, StdDev: 0.5
- Mean NBEI: 0.35, StdDev: 0.08
- Best Modulus: 28.45

**PDB Evidence**:
- 3 structures found
- 2 high-quality (< 2.0 Å)
- 1 medium-quality (2.5 Å)

---

### Step-by-Step Calculation

#### Step 1: Calculate Efficiency Metrics

```
SEI = 7.5 / (37.3 / 100) = 20.11
BEI = 7.5 / (206.28 / 1000) = 36.36
NSEI = 7.5 / 2 = 3.75
NBEI = 7.5 / 15 = 0.50
```

---

#### Step 2: Component 1 - Efficiency Outlier Score

**Z-scores**:
```
Z_SEI = (20.11 - 15.2) / 4.8 = 1.023
Z_BEI = (36.36 - 18.5) / 5.2 = 3.438
Z_NSEI = (3.75 - 1.8) / 0.5 = 3.900
Z_NBEI = (0.50 - 0.35) / 0.08 = 1.875
```

**Normalized** (divide by 3, clip [0,1]):
```
SEI_norm = 1.023 / 3 = 0.341
BEI_norm = 3.438 / 3 = 1.000 (clipped at 1.0)
NSEI_norm = 3.900 / 3 = 1.000 (clipped at 1.0)
NBEI_norm = 1.875 / 3 = 0.625
```

**Average**:
```
Efficiency_Score = (0.341 + 1.000 + 1.000 + 0.625) / 4 = 0.742
```

✅ **This compound is a strong efficiency outlier!**

---

#### Step 3: Component 2 - Development Angle Score

**Calculate Angle**:
```
Angle_SEI_BEI = arctan2(36.36, 20.11) × 180/π = 61.05°
```

**Score**:
```
Angle_Score = 1 - |61.05 - 45| / 45 = 1 - 0.357 = 0.643
```

✅ **Good balance, slightly polar-biased**

---

#### Step 4: Component 3 - Distance to Best Score

**Calculate Modulus**:
```
Modulus = sqrt(20.11² + 36.36²) = 41.50
```

**Score**:
```
Distance_Score = 41.50 / 28.45 = 1.000 (this IS the best!)
```

✅ **This compound is the best in the cohort!**

---

#### Step 5: Component 4 - PDB Evidence Score

**Base Score** (quantity):
```
base_score = min(3 / 5, 1.0) = 0.6
```

**Quality Score**:
```
quality_score = (2×1.0 + 1×0.75 + 0×0.5) / 3 = 2.75 / 3 = 0.917
```

**PDB Score**:
```
PDB_Score = (0.6 + 0.917) / 2 = 0.758
```

✅ **Strong PDB evidence with high-quality structures**

---

#### Step 6: Calculate Base Score (Phase 2 Weights)

**Weights**:
```
w1 = 0.500, w2 = 0.125, w3 = 0.1875, w4 = 0.1875
```

**Weighted Sum**:
```
OQPLA_Base = 0.500 × 0.742 +
             0.125 × 0.643 +
             0.1875 × 1.000 +
             0.1875 × 0.758

           = 0.371 + 0.080 + 0.188 + 0.142
           = 0.781
```

---

#### Step 7: Apply QED Multiplier

```
QED_Multiplier = 0.5 + 0.5 × 0.72 = 0.86
```

**Final Score**:
```
OQPLA_Final = 0.781 × 0.86 = 0.672
```

---

### Final Results

| Metric | Value | Contribution |
|--------|-------|--------------|
| Efficiency_Score | 0.742 | 0.500 × 0.742 × 0.86 = 0.319 |
| Angle_Score | 0.643 | 0.125 × 0.643 × 0.86 = 0.069 |
| Distance_Score | 1.000 | 0.1875 × 1.000 × 0.86 = 0.161 |
| PDB_Score | 0.758 | 0.1875 × 0.758 × 0.86 = 0.122 |
| **OQPLA_Base** | **0.781** | - |
| **QED_Multiplier** | **0.86** | - |
| **OQPLA_Final** | **0.672** | **Total** |

**Classification**: **Moderate IMP**
**Priority**: **3** (Monitor and gather more data)
**Action**: Validate with additional bioactivity data, assess synthetic accessibility

---

## Implementation Details

### Code Architecture

The O[Q/P/L]A scoring system is implemented across 3 main modules:

#### 1. `modules/efficiency_metrics.py`
Calculates SEI, BEI, NSEI, NBEI for each bioactivity row.

**Key Functions**:
- `calculate_sei(pActivity, psa)`
- `calculate_bei(pActivity, molecular_weight)`
- `calculate_nsei(pActivity, npol)`
- `calculate_nbei(pActivity, heavy_atoms)`
- `calculate_all_efficiency_metrics()` - Main entry point

---

#### 2. `modules/efficiency_planes.py`
Calculates geometric metrics (modulus, angle, slope) in efficiency space.

**Key Functions**:
- `calculate_modulus(x, y)` - Vector magnitude
- `calculate_angle(x, y)` - Vector angle (0-90°)
- `calculate_sei_bei_plane_metrics()` - Traditional plane
- `calculate_nsei_nbei_plane_metrics()` - Normalized plane

---

#### 3. `modules/oqpla_scoring.py`
Implements the complete O[Q/P/L]A scoring pipeline.

**Key Functions**:

**Phase 1**:
- `calculate_efficiency_outlier_score(df)` - Component 1
- `calculate_angle_score(angles)` - Component 2
- `calculate_distance_to_best_score(df)` - Component 3
- `calculate_oqpla_phase1(df)` - Complete Phase 1 calculation

**Phase 2**:
- `calculate_pdb_evidence_score(df)` - Component 4
- `calculate_oqpla_phase2(df)` - Complete Phase 2 calculation

**Interpretation**:
- `interpret_oqpla_score(score)` - Classification and recommendations
- `add_oqpla_interpretation(df)` - Add classification columns
- `get_oqpla_summary(df)` - Dataset-level statistics

---

### Data Flow

```
1. Raw Bioactivity Data (from ChEMBL)
   ↓
2. Calculate Molecular Properties (RDKit)
   → MW, PSA, NPOL, NHA, QED
   ↓
3. Calculate Efficiency Metrics
   → SEI, BEI, NSEI, NBEI
   ↓
4. Calculate Plane Geometry
   → Modulus, Angle, Slope
   ↓
5. Detect Efficiency Outliers
   → Z-scores, normalization
   ↓
6. Query PDB (if Phase 2 enabled)
   → PDB_Score, PDB_Num_Structures, etc.
   ↓
7. Calculate O[Q/P/L]A Score
   → Base score, QED multiplier, final score
   ↓
8. Add Interpretation
   → Classification, priority, recommendations
   ↓
9. Export Results
   → CSV files, visualizations, reports
```

---

### Configuration

**File**: `config.py`

```python
# Enable/disable PDB Evidence (Phase 2)
USE_PDB_EVIDENCE = True

# PDB similarity threshold for analog search
PDB_SIMILARITY_THRESHOLD = 0.90  # Tanimoto similarity

# Z-score normalization cap
ZSCORE_CAP = 3.0

# Optimal development angle
OPTIMAL_ANGLE = 45.0  # degrees
```

---

### Performance Considerations

#### PDB API Queries
- **Slow**: ~1-2 seconds per compound
- **Rate-limited**: RCSB PDB API has usage limits
- **Recommendation**: Enable only for high-priority compounds or small datasets

**Optimization**:
- Cache PDB results (stored in `df._pdb_results_cache`)
- Query unique SMILES only (avoid duplicate queries)
- Progress bar for user feedback

#### Memory Usage
- Large datasets (>1000 bioactivities) can consume significant memory
- Each bioactivity row gets full efficiency metrics + plane geometry
- **Recommendation**: Process in batches if >10,000 rows

---

### Error Handling

**Missing Data**:
- If any component score is NaN, it contributes 0 to the weighted sum
- If ALL scores are NaN, final score = 0.0

**Division by Zero**:
- Protected in all metric calculations
- Returns `np.nan` if denominator is 0 or NaN

**Outliers**:
- Z-scores capped at ±3σ to prevent extreme values from dominating
- QED multiplier ensures final score never exceeds 1.0

---

## Assay Quality Score (Display Only)

### Purpose

The **Assay Quality Score** is a separate metric that indicates potential assay interference issues. It is calculated but **DOES NOT affect the O[Q/P/L]A score**.

### Why Separate?

Some compounds (e.g., polyphenols) may flag as PAINS or aggregators but have strong PDB evidence confirming real binding. The philosophy:
- **O[Q/P/L]A score** = "Is this a validated IMP?"
- **Assay Quality Score** = "Is the bioactivity data trustworthy?"

Users see both scores and can make informed decisions.

### Calculation

```
Assay_Quality_Score = 1.0 - (num_flags / 5)
```

**5 Interference Mechanisms Checked**:
1. **PAINS** (Pan-Assay INterference Structures)
2. **Aggregators** (promiscuous binding via aggregation)
3. **Redox** (interfere via oxidation/reduction)
4. **Fluorescence** (interfere with fluorescence-based assays)
5. **Thiol Reactive** (covalent modifiers)

**Example**:
```
Flags: PAINS=True, Aggregator=False, Redox=True, Fluorescence=False, Thiol=False
num_flags = 2
Assay_Quality_Score = 1.0 - (2/5) = 0.6
```

**Interpretation**:
- **1.0**: No interference flags (highest confidence)
- **0.8**: 1 flag (good)
- **0.6**: 2 flags (fair - review carefully)
- **0.4**: 3 flags (poor - validate with orthogonal assays)
- **0.0**: All 5 flags (very poor - likely artifact)

**Note**: See `modules/assay_interference_filter.py` for detailed implementation.

---

## Future Enhancements & Discussion Points

This section outlines potential enhancements to the O[Q/P/L]A scoring system, with special focus on integrating assay interference flags and other validation streams. **These are discussion points for future implementation.**

---

### Enhancement 1: Integrating Assay Interference Flags into O[Q/P/L]A Score

#### Current Status: Display-Only

Currently, the **Assay Quality Score** is calculated but **does NOT affect the O[Q/P/L]A score**. This was a deliberate design decision based on the philosophy that:
- Some "interference-prone" compounds (e.g., polyphenols) have strong PDB evidence confirming real binding
- We want to avoid penalizing valid IMPs that happen to flag for interference

#### The Challenge: When to Penalize?

**Question for discussion**: Should assay interference flags reduce the O[Q/P/L]A score, and if so, under what conditions?

---

### Option A: Conditional Penalty (Recommended for Phase 3)

**Concept**: Only penalize interference if there's **no compensating structural evidence**.

#### Proposed Formula:

```python
# Step 1: Calculate Assay Quality Score
Assay_Quality_Score = 1.0 - (num_flags / 5)

# Step 2: Calculate PDB Evidence Strength
PDB_Evidence_Strength = PDB_Score  # 0-1 range

# Step 3: Conditional penalty
if PDB_Evidence_Strength >= 0.7:
    # Strong PDB evidence - minimal penalty
    Interference_Multiplier = 0.95 + 0.05 × Assay_Quality_Score
    # Range: [0.95, 1.0]

elif PDB_Evidence_Strength >= 0.4:
    # Moderate PDB evidence - moderate penalty
    Interference_Multiplier = 0.85 + 0.15 × Assay_Quality_Score
    # Range: [0.85, 1.0]

else:
    # Weak/no PDB evidence - full penalty
    Interference_Multiplier = 0.5 + 0.5 × Assay_Quality_Score
    # Range: [0.5, 1.0]

# Step 4: Apply to final score
OQPLA_Final = OQPLA_Base × QED_Multiplier × Interference_Multiplier
```

#### Example Scenarios:

**Scenario 1: Strong IMP with PDB evidence + interference flags**
```
OQPLA_Base = 0.85
QED = 0.75 → QED_Multiplier = 0.875
Assay_Quality_Score = 0.6 (2 flags)
PDB_Score = 0.85 → Strong evidence

Interference_Multiplier = 0.95 + 0.05 × 0.6 = 0.98

OQPLA_Final = 0.85 × 0.875 × 0.98 = 0.729
```
**Impact**: Only 2% reduction despite interference flags (PDB validates binding)

---

**Scenario 2: Strong IMP with NO PDB evidence + interference flags**
```
OQPLA_Base = 0.85
QED = 0.75 → QED_Multiplier = 0.875
Assay_Quality_Score = 0.4 (3 flags)
PDB_Score = 0.0 → No evidence

Interference_Multiplier = 0.5 + 0.5 × 0.4 = 0.7

OQPLA_Final = 0.85 × 0.875 × 0.7 = 0.520
```
**Impact**: 30% reduction (no PDB to validate, multiple interference flags)

---

**Scenario 3: Weak IMP + clean assay data**
```
OQPLA_Base = 0.45
QED = 0.80 → QED_Multiplier = 0.90
Assay_Quality_Score = 1.0 (no flags)
PDB_Score = 0.0

Interference_Multiplier = 0.5 + 0.5 × 1.0 = 1.0

OQPLA_Final = 0.45 × 0.90 × 1.0 = 0.405
```
**Impact**: No penalty (clean assay, but weak IMP stays weak)

---

#### Advantages of Option A:

✅ **Scientifically defensible**: PDB evidence overrides interference concerns
✅ **Conservative**: Strong IMPs with validation are protected
✅ **Penalizes high risk**: No validation + interference = lower score
✅ **Transparent**: Users see both Assay Quality and PDB scores

#### Disadvantages:

⚠️ **Complexity**: Three-tier penalty system
⚠️ **Dependency**: Requires PDB data to work optimally
⚠️ **Tuning needed**: Multiplier ranges need validation

---

### Option B: Fixed Penalty (Simpler Alternative)

**Concept**: Apply a fixed penalty based on Assay Quality Score, regardless of PDB evidence.

#### Proposed Formula:

```python
# Calculate interference multiplier
Interference_Multiplier = 0.7 + 0.3 × Assay_Quality_Score

# Apply to final score
OQPLA_Final = OQPLA_Base × QED_Multiplier × Interference_Multiplier
```

**Range**: [0.7, 1.0]
- 0 flags: 1.0 (no penalty)
- 1 flag: 0.94 (6% reduction)
- 2 flags: 0.88 (12% reduction)
- 3 flags: 0.82 (18% reduction)
- 4 flags: 0.76 (24% reduction)
- 5 flags: 0.70 (30% reduction max)

#### Advantages:

✅ **Simple**: Single formula, easy to understand
✅ **No dependencies**: Works without PDB data
✅ **Modest penalty**: Max 30% reduction protects potential IMPs

#### Disadvantages:

⚠️ **Penalizes validated IMPs**: Even with strong PDB evidence
⚠️ **May miss artifacts**: Max 30% penalty might be too lenient for 5-flag compounds

---

### Option C: Component 7 - Assay Confidence Score (New Component)

**Concept**: Add assay interference as a **7th component** with its own weight, rather than a multiplier.

#### Proposed Implementation:

**Weight allocation** (Phase 3 with 7 components):
```
Total = 0.40 + 0.10 + 0.15 + 0.15 + 0.10 + 0.10 + 0.10 = 1.10
Need to renormalize or reduce other weights
```

**Option C.1: Renormalize Phase 2 weights** (add 7th component):
```
Component 1 (Efficiency): 0.40 → 0.364 (36.4%)
Component 2 (Angle): 0.10 → 0.091 (9.1%)
Component 3 (Distance): 0.15 → 0.136 (13.6%)
Component 4 (PDB): 0.15 → 0.136 (13.6%)
Component 5 (Target Pred): 0.10 → 0.091 (9.1%) [Future]
Component 6 (Analog): 0.10 → 0.091 (9.1%) [Future]
Component 7 (Assay Confidence): 0.10 → 0.091 (9.1%) [NEW]
Total = 1.00
```

**Component 7 Scoring**:
```python
Assay_Confidence_Score = Assay_Quality_Score  # Direct use
# Range: 0.0 (all flags) to 1.0 (no flags)
```

**Weighted sum**:
```python
OQPLA_Base = (
    0.364 × Efficiency_Score +
    0.091 × Angle_Score +
    0.136 × Distance_Score +
    0.136 × PDB_Score +
    0.091 × Assay_Confidence_Score  # NEW COMPONENT
)

OQPLA_Final = OQPLA_Base × QED_Multiplier
```

#### Example:

```
Efficiency_Score = 0.85
Angle_Score = 0.70
Distance_Score = 0.90
PDB_Score = 0.80
Assay_Confidence_Score = 0.60 (2 flags)

OQPLA_Base = 0.364×0.85 + 0.091×0.70 + 0.136×0.90 + 0.136×0.80 + 0.091×0.60
           = 0.309 + 0.064 + 0.122 + 0.109 + 0.055
           = 0.659

QED_Multiplier = 0.5 + 0.5 × 0.75 = 0.875

OQPLA_Final = 0.659 × 0.875 = 0.577
```

#### Advantages:

✅ **Transparent**: Assay confidence is a visible component
✅ **Tunable**: Weight can be adjusted based on validation studies
✅ **Additive**: Integrates naturally with other components

#### Disadvantages:

⚠️ **Weight dilution**: Adding components reduces weight of others
⚠️ **Complexity**: More components = more to explain
⚠️ **May not distinguish**: A component with 9.1% weight has modest impact

---

### Option D: Flag-Specific Penalties (Most Conservative)

**Concept**: Different interference mechanisms have different implications. Weight them differently.

#### Proposed Penalty Weights:

| Flag | Penalty Weight | Rationale |
|------|----------------|-----------|
| **PAINS** | 0.30 | Highest concern - known false positives |
| **Aggregator** | 0.25 | High concern - promiscuous binding |
| **Redox** | 0.20 | Moderate - may be real for redox targets |
| **Fluorescence** | 0.15 | Lower concern - assay-specific |
| **Thiol Reactive** | 0.10 | Lowest concern - may be mechanism |

#### Formula:

```python
# Calculate weighted penalty
penalty_score = (
    PAINS_flag × 0.30 +
    Aggregator_flag × 0.25 +
    Redox_flag × 0.20 +
    Fluorescence_flag × 0.15 +
    Thiol_flag × 0.10
)

# Normalized to [0, 1]
total_weight = 0.30 + 0.25 + 0.20 + 0.15 + 0.10  # = 1.00

# Assay Confidence Score
Assay_Confidence_Score = 1.0 - penalty_score

# Apply as multiplier or component
```

#### Example:

```
PAINS = True (penalty = 0.30)
Aggregator = False (penalty = 0)
Redox = True (penalty = 0.20)
Fluorescence = False (penalty = 0)
Thiol = False (penalty = 0)

Total penalty = 0.30 + 0.20 = 0.50
Assay_Confidence_Score = 1.0 - 0.50 = 0.50
```

#### Advantages:

✅ **Nuanced**: Recognizes that not all flags are equal
✅ **Evidence-based**: Weights reflect literature consensus
✅ **Flexible**: Weights can be updated with new research

#### Disadvantages:

⚠️ **Subjective**: Weight assignment requires expert judgment
⚠️ **Controversial**: Community may disagree on weights
⚠️ **Complex**: Harder to explain to users

---

### Recommendation: Hybrid Approach (Phase 3 Implementation)

**Proposed Strategy**:

1. **Phase 2 (Current)**: Keep assay quality as **display-only** ✅ DONE
   - Users see Assay_Quality_Score alongside O[Q/P/L]A
   - No penalty applied
   - Gather user feedback and validation data

2. **Phase 3**: Implement **Option A (Conditional Penalty)** as opt-in feature
   - Add configuration flag: `USE_ASSAY_INTERFERENCE_PENALTY = False` (default)
   - When enabled, apply PDB-conditional penalty
   - Compare results with/without penalty on validation datasets

3. **Phase 4**: Based on Phase 3 validation, choose:
   - Keep conditional penalty (Option A) if effective
   - OR implement Component 7 (Option C) if community prefers transparency
   - OR implement flag-specific penalties (Option D) if evidence supports

4. **Documentation**: Clearly communicate in UI:
   - "Assay Quality Score displayed for reference only"
   - "Enable interference penalty in settings (experimental)"
   - Show impact: `Score without penalty: 0.72 | With penalty: 0.68`

---

### Enhancement 2: Component 5 - Target Prediction Confidence (Phase 3)

#### Current Status: Deferred

**Concept**: Use AI/ML target prediction to validate that observed activities align with predicted targets.

#### Proposed Implementation:

**Tools to integrate**:
1. **ChEMBL Target Prediction API**
   - Input: SMILES
   - Output: Top 5 predicted targets with confidence scores

2. **SEA (Similarity Ensemble Approach)**
   - Input: SMILES
   - Output: Target predictions based on ligand similarity

3. **DeepPurpose or similar ML models**
   - Deep learning-based target prediction

#### Scoring Logic:

```python
def calculate_target_prediction_score(smiles, observed_targets):
    # Step 1: Get predictions
    predictions = predict_targets(smiles)  # Returns [(target, confidence), ...]

    # Step 2: Check alignment
    aligned_predictions = []
    for obs_target in observed_targets:
        for pred_target, confidence in predictions:
            if obs_target == pred_target or is_related(obs_target, pred_target):
                aligned_predictions.append(confidence)

    # Step 3: Score based on alignment
    if len(aligned_predictions) > 0:
        # Average confidence of aligned predictions
        alignment_score = np.mean(aligned_predictions)
    else:
        # No alignment - penalize
        alignment_score = 0.2  # Base score for misalignment

    # Step 4: Bonus for multiple alignments
    if len(aligned_predictions) >= 3:
        alignment_score *= 1.2  # 20% bonus

    return min(alignment_score, 1.0)
```

#### Example:

```
Observed targets: [EGFR, VEGFR2, SRC]
Predicted targets: [(EGFR, 0.85), (VEGFR2, 0.72), (BRAF, 0.55), (CDK2, 0.48), (ABL, 0.40)]

Aligned: [EGFR: 0.85, VEGFR2: 0.72]
alignment_score = (0.85 + 0.72) / 2 = 0.785

Target_Prediction_Score = 0.785
```

#### Integration:

Add as Component 5 with 10% weight (Phase 3):
```python
w5 = 0.10 / (0.40 + 0.10 + 0.15 + 0.15 + 0.10)  # = 0.111

OQPLA_Base = w1×C1 + w2×C2 + w3×C3 + w4×C4 + w5×C5
```

#### Challenges:

⚠️ **API availability**: Not all tools have stable APIs
⚠️ **Computation time**: ML models can be slow
⚠️ **Target family mapping**: Need to handle target families (e.g., kinases)
⚠️ **False negatives**: IMPs may have novel mechanisms

---

### Enhancement 3: Component 6 - Analog Support Score (Phase 4)

#### Current Status: Future Work

**Concept**: Search ChEMBL for structurally similar compounds (analogs) and check if they also show high efficiency.

#### Proposed Implementation:

```python
def calculate_analog_support_score(smiles, target_activities):
    # Step 1: Find analogs (Tanimoto similarity >= 0.7)
    analogs = search_similar_compounds(smiles, threshold=0.7)

    # Step 2: For each analog, check if it's active against same targets
    consistent_analogs = 0
    total_analogs = len(analogs)

    for analog in analogs:
        analog_targets = get_targets(analog.chembl_id)

        # Check target overlap
        overlap = set(target_activities.keys()) & set(analog_targets.keys())

        if len(overlap) >= 2:  # At least 2 shared targets
            # Check if activities are consistent
            activity_correlation = calculate_activity_correlation(
                target_activities, analog_targets, overlap
            )

            if activity_correlation > 0.5:  # Consistent SAR
                consistent_analogs += 1

    # Step 3: Score based on analog support
    if total_analogs == 0:
        return 0.0  # No analogs found

    support_ratio = consistent_analogs / total_analogs

    # Step 4: Bonus for many supporting analogs
    if consistent_analogs >= 5:
        support_ratio *= 1.2

    return min(support_ratio, 1.0)
```

#### Example:

```
Query compound: CHEMBL123456
Analogs found: 12 (Tanimoto >= 0.7)

Consistent analogs (shared targets + similar activities): 8

support_ratio = 8 / 12 = 0.667

Analog_Support_Score = 0.667
```

#### Benefits:

✅ **SAR validation**: Confirms structure-activity relationships
✅ **Reduces false positives**: Random outliers won't have analog support
✅ **Literature connection**: May find published validation

#### Challenges:

⚠️ **Computation intensive**: Requires many ChEMBL queries
⚠️ **Natural product bias**: Fewer analogs for novel natural products
⚠️ **Activity threshold**: Defining "consistent" activities is subjective

---

### Enhancement 4: Confidence Intervals & Uncertainty Quantification

#### Current Status: Not Implemented

**Concept**: Provide confidence intervals for O[Q/P/L]A scores to reflect uncertainty.

#### Proposed Methods:

**Method 1: Bootstrap Resampling**
```python
# Resample bioactivity data 1000 times
bootstrap_scores = []
for i in range(1000):
    resampled_df = df.sample(frac=1.0, replace=True)
    score = calculate_oqpla_phase2(resampled_df)
    bootstrap_scores.append(score['OQPLA_Final'].mean())

# Calculate 95% confidence interval
CI_lower = np.percentile(bootstrap_scores, 2.5)
CI_upper = np.percentile(bootstrap_scores, 97.5)
```

**Method 2: Component-Level Uncertainty**
```python
# Propagate uncertainty from each component
σ_efficiency = estimate_efficiency_uncertainty(df)
σ_angle = estimate_angle_uncertainty(df)
σ_distance = estimate_distance_uncertainty(df)
σ_pdb = estimate_pdb_uncertainty(df)

# Combined uncertainty (assuming independence)
σ_total = sqrt(
    (w1 × σ_efficiency)² +
    (w2 × σ_angle)² +
    (w3 × σ_distance)² +
    (w4 × σ_pdb)²
)

# Report as: OQPLA = 0.72 ± 0.08
```

#### Visualization:

```
Compound A: O[Q/P/L]A = 0.72 [0.65 - 0.79]  95% CI
Compound B: O[Q/P/L]A = 0.68 [0.62 - 0.74]  95% CI
Compound C: O[Q/P/L]A = 0.85 [0.80 - 0.90]  95% CI
```

---

### Enhancement 5: Dynamic Weight Optimization (Machine Learning)

#### Current Status: Exploratory

**Concept**: Use machine learning to optimize component weights based on validation outcomes.

#### Proposed Approach:

1. **Collect validation data**:
   - Experimental validation results (success/failure)
   - Literature validation (published IMPs)
   - Expert annotations (curator judgments)

2. **Train optimization model**:
   ```python
   from sklearn.linear_model import LogisticRegression

   # Features: Component scores
   X = df[['Efficiency_Score', 'Angle_Score', 'Distance_Score', 'PDB_Score']]

   # Target: Validation outcome (1 = confirmed IMP, 0 = false positive)
   y = df['Validated']

   # Train model to find optimal weights
   model = LogisticRegression()
   model.fit(X, y)

   # Extract optimized weights
   optimized_weights = model.coef_[0]
   ```

3. **Compare with expert weights**:
   ```
   Expert weights: [0.50, 0.125, 0.1875, 0.1875]
   ML weights:     [0.55, 0.10, 0.20, 0.15]

   Agreement: 90% correlation
   ```

4. **Hybrid approach**: Combine expert knowledge with data-driven optimization

#### Challenges:

⚠️ **Validation data scarcity**: Need large dataset of validated IMPs
⚠️ **Overfitting risk**: May optimize for specific dataset
⚠️ **Interpretability**: ML-derived weights harder to explain

---

## Discussion Questions for Future Development

### Question 1: Assay Interference Integration

**Should assay interference flags reduce the O[Q/P/L]A score?**

**Arguments FOR**:
- ✅ Reduces false positives from assay artifacts
- ✅ Aligns with original IMPs 2016 paper philosophy
- ✅ Protects researchers from investing in artifacts

**Arguments AGAINST**:
- ❌ May penalize valid natural products (polyphenols, quinones)
- ❌ PDB evidence already validates binding
- ❌ Adds complexity to scoring system

**Proposed compromise**: Conditional penalty (Option A) - only penalize if no compensating evidence

**Community input needed**: Which option (A, B, C, or D) is most scientifically defensible?

---

### Question 2: Component Weight Distribution

**Should all components have equal weight, or should some dominate?**

**Current approach** (hierarchical):
- Efficiency outlier: **50%** (dominant)
- PDB evidence: **18.75%** (important)
- Distance to best: **18.75%** (important)
- Angle: **12.5%** (supporting)

**Alternative approach** (democratic):
- All 4 components: **25%** each

**Which is better?**
- Hierarchical: Efficiency is most important (outlier detection is core)
- Democratic: All validation streams equally important (convergent evidence)

---

### Question 3: Handling Missing Components

**What happens when PDB data is unavailable?**

**Current approach**: Renormalize other weights to 100%
**Alternative**: Penalize missing data (e.g., max score = 0.80 if no PDB)

**Trade-off**:
- Current: Doesn't penalize lack of structural data (fair to new compounds)
- Alternative: Incentivizes structural validation (encourages better science)

---

### Question 4: Score Calibration

**How should O[Q/P/L]A scores be calibrated?**

**Option A**: Absolute scale
- 0.9 = "90% confidence this is a valid IMP"
- Requires extensive validation to establish

**Option B**: Relative scale
- 0.9 = "Top 10% of compounds in database"
- Easier to interpret, but database-dependent

**Option C**: Hybrid
- Use validation data to map scores to probabilities
- Report both: "Score: 0.75 | Estimated success rate: 65%"

---

## Conclusion: Roadmap for Future Phases

### Phase 2 (Current) ✅ COMPLETE
- Components 1-4 implemented
- Assay quality displayed (not integrated)
- QED multiplier active

### Phase 3 (Target: Q2 2026)
- [ ] Component 5: Target Prediction Confidence
- [ ] Assay interference integration (Option A - conditional penalty)
- [ ] Confidence intervals for scores
- [ ] Validation study with experimental data

### Phase 4 (Target: Q4 2026)
- [ ] Component 6: Analog Support Score
- [ ] Dynamic weight optimization (ML-based)
- [ ] Full uncertainty quantification
- [ ] Publication of validation results

### Phase 5 (Target: 2027)
- [ ] Community feedback integration
- [ ] Cross-database validation (DrugBank, PubChem)
- [ ] API for external tool integration
- [ ] Standardized reporting format

---

**We welcome community input on these enhancements. Please open a GitHub issue to discuss!**

---

## References

### Scientific Literature

1. **Bisson et al. (2016)**
   "Can Invalid Bioactives Undermine Natural Product-Based Drug Discovery?"
   *J. Med. Chem.* 59(5): 1671-1690
   [DOI: 10.1021/acs.jmedchem.5b01009](https://doi.org/10.1021/acs.jmedchem.5b01009)

2. **Reddy et al. (2024)**
   "IMPs 2.0: Extended Analysis Framework"
   *(Manuscript in preparation)*

3. **Bickerton et al. (2012)**
   "Quantifying the chemical beauty of drugs"
   *Nature Chemistry* 4: 90-98
   [DOI: 10.1038/nchem.1243](https://doi.org/10.1038/nchem.1243)

4. **Hopkins et al. (2004)**
   "Ligand efficiency: a useful metric for lead selection"
   *Drug Discovery Today* 9(10): 430-431

5. **Baell & Holloway (2010)**
   "New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS)"
   *J. Med. Chem.* 53(7): 2719-2740

---

### IMPULATOR-3 Documentation

- **[CODEBASE_AUDIT_2025-11-19.md](CODEBASE_AUDIT_2025-11-19.md)**: Code quality audit
- **[CODEBASE_FIXES_2025-11-19.md](CODEBASE_FIXES_2025-11-19.md)**: Bug fixes and improvements
- **[OUTPUT_SCHEMA.md](OUTPUT_SCHEMA.md)**: Complete data schema for all exports
- **[MANUSCRIPT_METHODS_ASSAY_INTERFERENCE.md](MANUSCRIPT_METHODS_ASSAY_INTERFERENCE.md)**: Assay interference methodology

---

## Appendix: Mathematical Notation

### Symbols

| Symbol | Description | Units |
|--------|-------------|-------|
| pActivity | -log₁₀(Activity in M) | - |
| MW | Molecular Weight | Da |
| PSA | Polar Surface Area | Ų |
| NPOL | N + O atom count | atoms |
| NHA | Number of Heavy Atoms | atoms |
| SEI | Surface Efficiency Index | - |
| BEI | Binding Efficiency Index | - |
| NSEI | Normalized Surface Efficiency Index | - |
| NBEI | Normalized Binding Efficiency Index | - |
| QED | Quantitative Estimate of Drug-likeness | 0-1 |
| Z | Z-score (standard score) | σ |
| θ | Angle in efficiency plane | degrees |
| \|v\| | Modulus (vector magnitude) | - |

---

### Formulas Summary

```
Efficiency Metrics:
SEI = pActivity / (PSA / 100)
BEI = pActivity / (MW / 1000)
NSEI = pActivity / NPOL
NBEI = pActivity / NHA

Plane Geometry:
Modulus = sqrt(x² + y²)
Angle = arctan2(y, x) × 180/π

Z-Score Normalization:
Z = (value - mean) / std_dev
Normalized = (Z / 3).clip(0, 1)

O[Q/P/L]A Components:
Efficiency_Score = mean(SEI_norm, BEI_norm, NSEI_norm, NBEI_norm)
Angle_Score = 1 - |Angle - 45°| / 45°
Distance_Score = Compound_Modulus / Best_Modulus
PDB_Score = (base_score + quality_score) / 2

Phase 1 (normalized):
w1 = 0.615, w2 = 0.154, w3 = 0.231
OQPLA_Base = w1×C1 + w2×C2 + w3×C3

Phase 2 (with PDB):
w1 = 0.500, w2 = 0.125, w3 = 0.1875, w4 = 0.1875
OQPLA_Base = w1×C1 + w2×C2 + w3×C3 + w4×C4

Final Score:
QED_Multiplier = 0.5 + 0.5 × QED
OQPLA_Final = OQPLA_Base × QED_Multiplier
```

---

## Contact & Support

**IMPULATOR-3 Development Team**
GitHub: [anthropics/claude-code](https://github.com/anthropics/claude-code)

**Questions or Issues?**
Open an issue on GitHub or contact the research team.

---

**Document End**

*Last Updated: November 19, 2025*
*IMPULATOR-3 Version: 2.0*
*O[Q/P/L]A Scoring Version: Phase 2 (Components 1-4 implemented)*
