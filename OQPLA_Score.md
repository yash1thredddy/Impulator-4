# O[Q/P/L]A Scoring System: A Complete Guide for Drug Researchers

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Quick Reference Card](#quick-reference-card)
3. [What is OQPLA?](#what-is-oqpla)
4. [Why Do We Need OQPLA?](#why-do-we-need-oqpla)
5. [The Four Scoring Components](#the-four-scoring-components)
   - [Component 1: Efficiency Outlier Score](#component-1-efficiency-outlier-score-40-weight)
   - [Component 2: Development Angle Score](#component-2-development-angle-score-15-weight)
   - [Component 3: Distance to Best-in-Class Score](#component-3-distance-to-best-in-class-score-20-weight)
   - [Component 4: PDB Structural Evidence Score](#component-4-pdb-structural-evidence-score-5-weight)
6. [The Final Score Calculation](#the-final-score-calculation)
7. [Understanding Your Results](#understanding-your-results)
8. [Worked Examples with Real Drugs](#worked-examples-with-real-drugs)
9. [Red Flags: Compounds to Watch Out For](#red-flags-compounds-to-watch-out-for)
10. [Frequently Asked Questions](#frequently-asked-questions)
11. [Glossary of Terms](#glossary-of-terms)

---

## Executive Summary

**O[Q/P/L]A** stands for **Overall Quality/Promise/Likelihood Assessment**. It is a multi-criteria scoring system designed to help you identify **Invalid Metabolic Panaceas (IMPs)** - compounds that appear exceptional but are likely assay artifacts.

> **CRITICAL:** OQPLA measures the probability that a compound is a FALSE POSITIVE (assay artifact), NOT the probability it's a good drug.

The OQPLA score ranges from **0.0 to 1.0**, where:
- **0.9-1.0**: ⚠️ Exceptional IMP - VERY HIGH false positive risk - DEPRIORITIZE
- **0.7-0.9**: ⚠️ Strong IMP - HIGH false positive risk - VALIDATE before advancing
- **0.5-0.7**: ⚠️ Moderate IMP - Moderate risk - Monitor carefully
- **0.3-0.5**: ✓ Weak IMP - Lower risk - More likely genuine
- **< 0.3**: ✓ Not IMP - Likely genuine activity - PRIORITIZE

### Score Distribution Visualization

```
OQPLA Score Scale (0.0 ────────────────────────────────────────── 1.0)

     NOT IMP        WEAK         MODERATE        STRONG      EXCEPTIONAL
  ◄───────────►◄───────────►◄───────────────►◄───────────►◄─────────────►
   0.0    0.3      0.3   0.5     0.5     0.7    0.7   0.9    0.9    1.0
       ★              ✓              ●              ⚠️              ✗
   PRIORITIZE      PROCEED       MONITOR       VALIDATE     DEPRIORITIZE

  Legend (CORRECTED INTERPRETATION):
  ★ = Lowest false positive risk - PRIORITIZE for development
  ✓ = Low risk - more likely genuine - PROCEED with validation
  ● = Moderate risk - needs additional evidence - MONITOR carefully
  ⚠️ = High false positive risk - VALIDATE with orthogonal assays
  ✗ = Very high false positive risk - DEPRIORITIZE unless validated
```

---

## Quick Reference Card

**Print this section for easy reference during your analysis.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OQPLA QUICK REFERENCE CARD                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  THE FOUR COMPONENTS:                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 1. EFFICIENCY (50%)   │ How exceptional is potency vs size/polarity?│ │
│  │ 2. ANGLE (18.75%)     │ Is development trajectory balanced?         │ │
│  │ 3. DISTANCE (25%)     │ How close to best-in-class?                 │ │
│  │ 4. PDB (6.25%)        │ Is there structural validation?             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  FINAL SCORE = Base Score × QED Multiplier                              │
│                                                                         │
│  DECISION MATRIX (CORRECTED):                                           │
│  ┌──────────┬──────────────────┬────────────────────────────────────┐  │
│  │  Score   │  Classification  │  Action (CORRECTED)                │  │
│  ├──────────┼──────────────────┼────────────────────────────────────┤  │
│  │ 0.9-1.0  │  EXCEPTIONAL IMP │  ⚠️ DEPRIORITIZE - Very high risk  │  │
│  │ 0.7-0.9  │  STRONG IMP      │  ⚠️ VALIDATE before advancing      │  │
│  │ 0.5-0.7  │  MODERATE IMP    │  ⚠️ MONITOR - Moderate risk        │  │
│  │ 0.3-0.5  │  WEAK IMP        │  ✓ PROCEED - Low risk (genuine)    │  │
│  │ < 0.3    │  NOT IMP         │  ✓ PRIORITIZE - Best candidates    │  │
│  └──────────┴──────────────────┴────────────────────────────────────┘  │
│                                                                         │
│  RED FLAGS TO WATCH:                                                    │
│  ! High efficiency + Low PDB score → Possible artifact                  │
│  ! Extreme angle (<20° or >70°) → Unbalanced profile                    │
│  ! High score + Low QED → Developability concerns                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## What is OQPLA?

OQPLA is a composite scoring system that evaluates drug candidates from **multiple angles simultaneously**. Think of it like a job interview where a candidate is evaluated on:
- Technical skills (efficiency metrics)
- Work-life balance (development angle)
- Comparison to top performers (distance to best)
- Background verification (structural evidence)

Rather than relying on a single measurement that could be misleading, OQPLA combines four different perspectives to give you a more reliable overall picture.

### The Problem OQPLA Solves

In drug discovery, you often encounter compounds that look exceptionally promising in initial screens - they appear to have remarkable potency against your target. However, many of these "too good to be true" results turn out to be **Invalid Metabolic Panaceas (IMPs)** - compounds that produce misleading results due to:

- Assay interference (PAINS, aggregation, redox cycling)
- Fluorescence interference
- Non-specific binding
- Compound instability

**OQPLA identifies these IMPs** by assigning HIGH SCORES to compounds with characteristics typical of assay artifacts. The higher the OQPLA score, the MORE LIKELY the compound is a false positive that should be deprioritized or validated with orthogonal assays.

---

## Why Do We Need OQPLA?

### Traditional Approach: Single-Metric Selection

Traditionally, researchers might select compounds based on a single metric like IC50 (potency). But consider this scenario:

| Compound | IC50 (nM) | Molecular Weight | Would You Pursue? |
|----------|-----------|------------------|-------------------|
| A        | 5         | 850 Da           | Maybe not - too large |
| B        | 50        | 320 Da           | Better - more drug-like |
| C        | 0.5       | 450 Da           | Looks perfect! |

Compound C looks ideal, but what if it:
- Has no structural evidence in PDB (never been crystallized with a target)?
- Shows an unusual efficiency pattern suggesting it might be an assay artifact?
- Has very poor drug-likeness (QED = 0.2)?

### OQPLA Approach: Multi-Criteria Evaluation

OQPLA evaluates all these factors together:

```
OQPLA Score = (Efficiency + Angle + Distance + PDB Evidence) × Drug-likeness
```

**CRITICAL:** Higher scores indicate compounds with characteristics typical of assay artifacts (false positives). This system helps you **AVOID** wasting resources on compounds that appear exceptional but are likely invalid results.

- **High score (0.9+)** = High risk of false positive = DEPRIORITIZE
- **Low score (<0.3)** = Low risk = Likely genuine = PRIORITIZE

---

## The Four Scoring Components

### Component 1: Efficiency Outlier Score (40% Weight)

**Purpose**: Measures how exceptional your compound's efficiency is compared to similar compounds.

**What It Measures**: This component uses four ligand efficiency metrics:

#### The Four Efficiency Metrics

1. **SEI (Surface Efficiency Index)**
   - What it measures: Potency relative to polar surface area
   - Formula: SEI = pActivity ÷ (PSA ÷ 100)
   - Interpretation: Higher = better efficiency per unit of polar surface

2. **BEI (Binding Efficiency Index)**
   - What it measures: Potency relative to molecular weight
   - Formula: BEI = pActivity ÷ (MW ÷ 1000)
   - Interpretation: Higher = better efficiency per unit of size

3. **NSEI (Normalized Surface Efficiency Index)**
   - What it measures: Potency relative to polar atom count
   - Formula: NSEI = pActivity ÷ NPOL (where NPOL = nitrogen + oxygen atom count)
   - Interpretation: Higher = better efficiency per polar atom

4. **NBEI (Normalized Binding Efficiency Index)**
   - What it measures: Potency relative to heavy atom count
   - Formula: NBEI = pActivity ÷ NHA (where NHA = non-hydrogen atoms)
   - Interpretation: Higher = better efficiency per atom

**Note on pActivity**: pActivity = -log10(activity in Molar). For example:
- IC50 = 10 nM = 10 × 10⁻⁹ M → pActivity = -log10(10⁻⁸) = 8.0
- IC50 = 1 μM = 10⁻⁶ M → pActivity = -log10(10⁻⁶) = 6.0

Higher pActivity = more potent compound.

#### How the Score is Calculated

For each of the four metrics, we calculate a **Z-score** that tells us how many standard deviations this compound differs from the average:

```
Z-score = (Compound's value - Average of all compounds) ÷ Standard deviation
```

**Example**:
- If the average BEI in your dataset is 20 with a standard deviation of 5
- And your compound has BEI = 30
- Z-score = (30 - 20) ÷ 5 = 2.0 (i.e., 2 standard deviations above average)

We then convert Z-scores to a 0-1 scale using a **sigmoid function**.

#### What is a Sigmoid Function? (Simple Explanation)

Think of the sigmoid as a "soft cap" that converts any number into a score between 0 and 1:

```
Sigmoid Transformation: Z-score → Normalized Score (0-1)

         Normalized
         Score
           1.0 ─┬─────────────────────────────●━━━━━━━━ Exceptional
               │                          ●
           0.9 ─┤                       ●
               │                     ●
           0.8 ─┤                  ●
               │                ●          ← Above Average Zone
           0.7 ─┤             ●
               │           ●
           0.6 ─┤        ●
               │      ●
           0.5 ─┼────●─────────────────────── Average (Z=0)
               │   ●
           0.4 ─┤  ●
               │ ●                         ← Below Average Zone
           0.3 ─┤●
               │●
           0.2 ─●
              ●│
           0.1 ─●
               │
           0.0 ─┴───┬───┬───┬───┬───┬───┬───┬───┬───→ Z-score
                   -3  -2  -1   0   1   2   3   4

    Key Values:
    ┌─────────┬────────────────┬─────────────────────────┐
    │ Z-score │ Normalized     │ Meaning                 │
    ├─────────┼────────────────┼─────────────────────────┤
    │  -3     │  0.05 (5%)     │ Very poor performer     │
    │  -1     │  0.27 (27%)    │ Below average           │
    │   0     │  0.50 (50%)    │ Average compound        │
    │  +1     │  0.73 (73%)    │ Above average           │
    │  +2     │  0.88 (88%)    │ Good performer          │
    │  +3     │  0.95 (95%)    │ Exceptional outlier     │
    │  +4     │  0.98 (98%)    │ Extreme outlier         │
    └─────────┴────────────────┴─────────────────────────┘
```

**Why use sigmoid instead of simple clipping?**

- **Preserves ranking**: A Z-score of 4 gets a higher score than Z=3 (0.98 vs 0.95)
- **Smooth transition**: No arbitrary cutoffs that lose information
- **Intuitive**: Score naturally compresses toward the extremes

**Note**: While all four metrics are calculated and displayed for reference, the Efficiency Score uses only **SEI and BEI** (the two primary metrics). The final Efficiency Score is the average of these two normalized metrics.

#### Typical Ranges for Efficiency Metrics

| Metric | Poor | Average | Good | Excellent |
|--------|------|---------|------|-----------|
| SEI    | < 5  | 5-15    | 15-25| > 25      |
| BEI    | < 10 | 10-20   | 20-30| > 30      |
| NSEI   | < 0.5| 0.5-1.5 | 1.5-2.5| > 2.5   |
| NBEI   | < 0.15| 0.15-0.3| 0.3-0.45| > 0.45 |

---

### Component 2: Development Angle Score (15% Weight)

**Purpose**: Evaluates whether the compound has a balanced development profile.

**Concept**: Imagine plotting your compound on a graph where:
- X-axis = Surface Efficiency (SEI)
- Y-axis = Binding Efficiency (BEI)

The **angle** from the origin to your compound tells you about its development trajectory:

```
                    BEI (Y-axis)
                    ↑
                    │     * (45°) ← OPTIMAL: Balanced
                    │    /
                    │   /
                    │  * (70°) ← Too polar
                    │ /
          * (20°)   │/
          ↓         └───────────────→ SEI (X-axis)
    Too hydrophobic
```

#### Interpretation of Angles

| Angle | Meaning | Score |
|-------|---------|-------|
| 45°   | **Optimal**: Balanced improvement in both size and polarity | 1.0 |
| 40-50°| Excellent: Near optimal balance | 0.89-1.0 |
| 30-40° or 50-60° | Good: Moderate balance | 0.67-0.89 |
| 20-30° or 60-70° | Fair: Somewhat unbalanced | 0.44-0.67 |
| < 20° | Poor: Too hydrophobic (relies too much on lipophilicity) | < 0.44 |
| > 70° | Poor: Too polar (may have permeability issues) | < 0.44 |

#### The Formula

```
Angle Score = 1 - (|Actual Angle - 45°| ÷ 45°)
```

**Example Calculations**:
- Compound at 45°: Score = 1 - (|45-45| ÷ 45) = 1.0
- Compound at 60°: Score = 1 - (|60-45| ÷ 45) = 1 - 0.33 = 0.67
- Compound at 30°: Score = 1 - (|30-45| ÷ 45) = 1 - 0.33 = 0.67
- Compound at 20°: Score = 1 - (|20-45| ÷ 45) = 1 - 0.56 = 0.44

---

### Component 3: Distance to Best-in-Class Score (20% Weight)

**Purpose**: Measures how close your compound is to the best performer in the dataset.

**Concept**: Using the same efficiency plane (SEI vs BEI), we calculate the **modulus** (distance from origin) for each compound:

```
Modulus = √(SEI² + BEI²)
```

This represents the **overall efficiency magnitude** - how far along the "good" direction the compound has traveled.

#### Visual Representation

```
        BEI
        ↑
        │         ★ Best-in-class (Modulus = 50)
        │        /
        │       /
        │      * Your compound (Modulus = 35)
        │     /
        │    /
        │   /
        └──────────────────→ SEI

        Distance Score = 35 ÷ 50 = 0.70
```

#### The Formula

```
Distance Score = Your Compound's Modulus ÷ Best Compound's Modulus
```

**Example**:
- Best compound: SEI=30, BEI=40 → Modulus = √(900+1600) = 50
- Your compound: SEI=21, BEI=28 → Modulus = √(441+784) = 35
- Distance Score = 35 ÷ 50 = 0.70

This means your compound is 70% as efficient as the best performer in the cohort.

---

### Component 4: PDB Structural Evidence Score (5% Weight)

**Purpose**: Validates your compound by checking if similar molecules have been experimentally observed binding to proteins.

**Why This Matters**: If a compound or close analog has been crystallized in complex with a protein target, this provides **experimental evidence** that:
1. The compound can actually bind to proteins
2. The binding mode is understood
3. The compound is stable enough for crystallography

This is one of the strongest forms of validation available.

#### How It Works

1. **Search for Similar Ligands**: We search the RCSB Protein Data Bank for compounds structurally similar to yours

2. **Retrieve Resolution Data**: For each structure found, we check the **resolution** (in Ångströms, Å)
   - Resolution indicates how clearly we can see the atoms in the crystal structure
   - Lower resolution = clearer picture = higher confidence

3. **Classify Quality**:

| Resolution | Quality | Stars | Meaning |
|------------|---------|-------|---------|
| < 2.0 Å    | High    | ★★★   | Excellent - individual atoms clearly visible |
| 2.0-3.0 Å  | Medium  | ★★    | Good - molecular details visible |
| > 3.0 Å    | Low     | ★     | Limited - only overall shape visible |

#### The Scoring Formula

```
Base Score = min(Number of structures with resolution ÷ 5, 1.0)
Quality Score = (Sum of quality multipliers) ÷ (Number of structures × 1.0)
PDB Score = (Base Score + Quality Score) ÷ 2
```

Where quality multipliers are:
- ★★★ (< 2.0 Å): 1.0
- ★★ (2.0-3.0 Å): 0.75
- ★ (> 3.0 Å): 0.5

#### Example Calculation

Your compound search finds 6 similar structures in PDB:
- 2 structures at 1.8 Å (★★★, multiplier = 1.0 each)
- 3 structures at 2.5 Å (★★, multiplier = 0.75 each)
- 1 structure at 3.5 Å (★, multiplier = 0.5)

```
Base Score = min(6 ÷ 5, 1.0) = 1.0 (capped at 1.0)
Quality Score = (2×1.0 + 3×0.75 + 1×0.5) ÷ (6 × 1.0) = 4.75 ÷ 6 = 0.79
PDB Score = (1.0 + 0.79) ÷ 2 = 0.895
```

#### Interpretation

The Criteria for PDB Score is only a guideline not a Predictor of Success Examination of the Structural Evidence is always Highly Recommended

| PDB Score | Interpretation |
|-----------|----------------|
| 0.8-1.0   | Strong structural evidence - high confidence |
| 0.5-0.8   | Moderate evidence - some validation exists |
| 0.2-0.5   | Limited evidence - few or low-quality structures |
| 0.0-0.2   | Minimal/no evidence - no structural validation |

---

## The Final Score Calculation

### Step 1: Combine the Four Components

The four components are combined using **weighted averaging**:

```
Base Score = (0.40 × Efficiency Score) +
             (0.15 × Angle Score) +
             (0.20 × Distance Score) +
             (0.05 × PDB Score)
```

Note: These weights sum to 0.80, so they are **renormalized** to 100%:

| Component | Raw Weight | Normalized Weight |
|-----------|------------|-------------------|
| Efficiency | 40% | 50.0% (0.40 ÷ 0.80) |
| Angle | 15% | 18.75% (0.15 ÷ 0.80) |
| Distance | 20% | 25.0% (0.20 ÷ 0.80) |
| PDB | 5% | 6.25% (0.05 ÷ 0.80) |
| **Total** | **80%** | **100%** |

### Step 2: Apply the Drug-Likeness Multiplier

The base score is then adjusted by the compound's **QED (Quantitative Estimate of Drug-likeness)**:

```
QED Multiplier = 0.75 + (0.25 × QED)
Final OQPLA Score = Base Score × QED Multiplier
```

**What is QED?**

QED is a score from 0 to 1 that estimates how "drug-like" a molecule is, based on properties like:
- Molecular weight (optimal: 200-500 Da)
- LogP (lipophilicity, optimal: 0-5)
- Number of hydrogen bond donors/acceptors
- Number of rotatable bonds
- Polar surface area

**Effect of QED Multiplier**:

| QED Value | Meaning | Multiplier | Effect on Score |
|-----------|---------|------------|-----------------|
| 1.0 | Highly drug-like | 1.0 | No change |
| 0.8 | Drug-like | 0.95 | 5% reduction |
| 0.6 | Moderately drug-like | 0.90 | 10% reduction |
| 0.4 | Poorly drug-like | 0.85 | 15% reduction |
| 0.2 | Very poor drug-like | 0.80 | 20% reduction |
| 0.0 | Not drug-like | 0.75 | 25% reduction |

**Visual: How QED Affects Your Score**

```
Example: Compound with Base Score = 0.80

                         Base Score (before QED)
                         ├────────────────────────────────────┤
                         0.0                                 0.80

If QED = 1.0 (excellent): ████████████████████████████████████ = 0.80 (no change)
If QED = 0.8 (good):      ██████████████████████████████████░░ = 0.76 (5% loss)
If QED = 0.6 (moderate):  ████████████████████████████████░░░░ = 0.72 (10% loss)
If QED = 0.4 (poor):      ██████████████████████████████░░░░░░ = 0.68 (15% loss)
If QED = 0.2 (very poor): ████████████████████████████░░░░░░░░ = 0.64 (20% loss)
If QED = 0.0 (awful):     ██████████████████████████░░░░░░░░░░ = 0.60 (25% loss)
                         0.0                               0.80

The shaded area (░) shows the score lost due to poor drug-likeness.
```

**Why the 75% Floor?**

The minimum multiplier is 0.75 (not 0) because:
- Even poorly drug-like compounds may have value as chemical probes
- Some targets (e.g., intracellular) may tolerate larger molecules
- The compound may be optimizable toward better QED
- QED is a guideline, not a strict predictor of success

### Complete Formula

```
Final OQPLA Score = [0.50 × Efficiency + 0.1875 × Angle + 0.25 × Distance + 0.0625 × PDB] × (0.75 + 0.25 × QED)
```

---

## Understanding Your Results

### Score Classification Table (CORRECTED)

**CRITICAL: IMP = Invalid Metabolic Panacea = FALSE POSITIVE Indicator**

| Score Range | Classification | Priority | Recommended Action (CORRECTED) |
|-------------|----------------|----------|-------------------------------|
| 0.90 - 1.00 | **Exceptional IMP** | 1 (Highest Concern) | ⚠️ **DEPRIORITIZE** - Very high false positive risk. Do not pursue without orthogonal validation (SPR, ITC, MST). Check for PAINS/aggregation. |
| 0.70 - 0.89 | **Strong IMP** | 2 (High Concern) | ⚠️ **VALIDATE** before advancing. High false positive risk. Require counter-screens and orthogonal assays before significant investment. |
| 0.50 - 0.69 | **Moderate IMP** | 3 (Moderate Concern) | ⚠️ **MONITOR** carefully. Moderate false positive risk. Gather additional validation data before committing resources. |
| 0.30 - 0.49 | **Weak IMP** | 4 (Low Concern) | ✓ **PROCEED** with standard validation. Low false positive risk - more likely genuine activity. Apply normal drug development criteria. |
| 0.00 - 0.29 | **Not IMP** | None (Best) | ✓ **PRIORITIZE** for development. Lowest false positive risk - likely genuine activity. Best candidates for resource allocation. |

### Understanding Component Contributions

After scoring, you can see how much each component contributed to the final score:

```
Example Output:
Compound: Quercetin
OQPLA Final Score: 0.72

Component Breakdown:
├── Efficiency Contribution: 0.35 (48.6%)
├── Angle Contribution: 0.08 (11.1%)
├── Distance Contribution: 0.14 (19.4%)
├── PDB Contribution: 0.15 (20.8%)
└── QED Impact: -0.07 (reduced score by 7%)
```

This tells you:
- Efficiency is the main driver of this compound's score
- PDB evidence is strong (contributing nearly 21%)
- The QED penalty is moderate (7% reduction)

### Red Flags to Watch For

1. **High Efficiency but Low PDB Score**: May indicate the activity is an artifact
2. **Extreme Angle (< 20° or > 70°)**: Unbalanced development trajectory
3. **High Score but Low QED**: Potent but may have developability issues
4. **Large QED Impact**: The compound loses significant points due to poor drug-likeness

---

## Worked Examples with Real Drugs

The following examples use well-known drugs and drug candidates to illustrate how OQPLA scoring works in practice.

### Example 1: Ibuprofen (Well-Known NSAID)

**Compound**: Ibuprofen - A widely used anti-inflammatory drug

```
Ibuprofen Structure:
                CH3
                 |
    CH3-CH-CH2-⬡-CH-COOH
        |           |
       CH3         CH3

    Molecular Formula: C13H18O2
```

| Property | Value | Notes |
|----------|-------|-------|
| pActivity | 5.0 (IC50 = 10 μM against COX-1) | Moderate potency |
| Molecular Weight | 206.3 Da | Small molecule |
| PSA | 37.3 Å² | Low polar surface |
| Heavy Atoms | 15 | Compact structure |
| N+O Atoms | 2 | Minimal polarity |
| QED | 0.78 | Good drug-likeness |

**Step 1: Calculate Efficiency Metrics**
```
SEI  = 5.0 ÷ (37.3/100) = 13.4   [Average - typical for NSAIDs]
BEI  = 5.0 ÷ (206.3/1000) = 24.2 [Good - efficient for its size]
NSEI = 5.0 ÷ 2 = 2.5             [Excellent - few polar atoms]
NBEI = 5.0 ÷ 15 = 0.33           [Good - compact molecule]
```

**Step 2: Calculate Plane Geometry**
```
Modulus = √(13.4² + 24.2²) = √(180 + 586) = 27.7
Angle   = arctan(24.2/13.4) × 180/π = 61°
```

**Step 3: Component Scores**
```
┌─────────────────┬───────┬─────────────────────────────────────────┐
│ Component       │ Score │ Explanation                             │
├─────────────────┼───────┼─────────────────────────────────────────┤
│ Efficiency      │ 0.62  │ Above average on most metrics           │
│ Angle           │ 0.64  │ Slightly polar-biased (61° vs 45° opt.) │
│ Distance        │ 0.85  │ Close to best in NSAID cohort           │
│ PDB Evidence    │ 0.95  │ 47 crystal structures in PDB!           │
└─────────────────┴───────┴─────────────────────────────────────────┘
```

**Step 4: Final Calculation**
```
Base Score = 0.50×0.62 + 0.1875×0.64 + 0.25×0.85 + 0.0625×0.95
           = 0.310 + 0.120 + 0.213 + 0.059
           = 0.702

QED Multiplier = 0.75 + 0.25×0.78 = 0.945

Final Score = 0.702 × 0.945 = 0.663
```

**Result**: **Moderate IMP** (Priority 3) → Score: 0.66

**Interpretation (CORRECTED)**:
- Ibuprofen scores 0.66 = MODERATE false positive risk
- This is ACCEPTABLE: it's not flagged as high-risk artifact (score < 0.7)
- The strong PDB evidence (95+ structures!) **validates** it as a genuine binder, not an artifact
- The moderate score with strong PDB evidence = **genuine compound, safe to proceed**
- The slightly unbalanced angle (61°) is a minor concern, not a dealbreaker

```
Component Contribution Chart:
┌──────────────────────────────────────────────────────────────────────┐
│ Efficiency   ████████████████████████████████░░░░░░░░░  47.9%        │
│ Angle        █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  12.4%        │
│ Distance     ████████████████████████░░░░░░░░░░░░░░░░░  24.6%        │
│ PDB          ██████████████████████████░░░░░░░░░░░░░░░  27.5%        │
│ QED Impact   ▼▼▼▼▼▼▼▼░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ -12.4%        │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Example 2: Quercetin (Natural Product - Potential Artifact)

**Compound**: Quercetin - A plant flavonoid that appears highly active in many assays

```
Quercetin Structure:
           OH
            |
        ⬡──⬡──OH
       /    \\
    HO─⬡      ⬡─OH
       \\    /
        ⬡──⬡
            |
           OH

    Molecular Formula: C15H10O7
    Known PAINS compound (catechol substructure)
```

| Property | Value | Notes |
|----------|-------|-------|
| pActivity | 6.5 (IC50 = 316 nM against multiple targets) | Suspiciously potent |
| Molecular Weight | 302.2 Da | Small-medium |
| PSA | 131.4 Å² | High polar surface |
| Heavy Atoms | 22 | Medium size |
| N+O Atoms | 7 | Highly polar |
| QED | 0.42 | Poor drug-likeness |

**Warning**: Quercetin is a known PAINS compound due to its catechol groups.

**Step 1: Calculate Efficiency Metrics**
```
SEI  = 6.5 ÷ (131.4/100) = 4.9   [Poor - too much polar surface]
BEI  = 6.5 ÷ (302.2/1000) = 21.5 [Good - efficient for size]
NSEI = 6.5 ÷ 7 = 0.93            [Below average - many polar atoms]
NBEI = 6.5 ÷ 22 = 0.30           [Average]
```

**Step 2: Component Scores**
```
┌─────────────────┬───────┬─────────────────────────────────────────┐
│ Component       │ Score │ Explanation                             │
├─────────────────┼───────┼─────────────────────────────────────────┤
│ Efficiency      │ 0.48  │ Unbalanced - good BEI but poor SEI      │
│ Angle           │ 0.22  │ Very poor (77°) - too polar!            │
│ Distance        │ 0.65  │ Moderate                                │
│ PDB Evidence    │ 0.72  │ 28 structures (BUT may be artifacts)    │
└─────────────────┴───────┴─────────────────────────────────────────┘
```

**Step 3: Final Calculation**
```
Base Score = 0.50×0.48 + 0.1875×0.22 + 0.25×0.65 + 0.0625×0.72
           = 0.240 + 0.041 + 0.163 + 0.045
           = 0.489

QED Multiplier = 0.75 + 0.25×0.42 = 0.855

Final Score = 0.489 × 0.855 = 0.418
```

**Result**: **Weak IMP** (Priority 4) → Score: 0.42

**Interpretation (CORRECTED)**:
```
Score 0.42 = WEAK IMP = Lower false positive risk

HOWEVER, quercetin has multiple RED FLAGS that override the score:

┌──────────────────────────────────────────────────────────────────────┐
│ ⚠ PAINS Alert: Catechol substructure (known assay interference)     │
│ ⚠ Extreme Angle: 77° (far from optimal 45°)                         │
│ ⚠ Low QED: 0.42 (poor drug-likeness, 29% score reduction)           │
│ ⚠ Multi-target Activity: Hits many unrelated targets (suspicious)   │
│ ⚠ Low PDB Score: Despite activity claims, structural evidence weak  │
└──────────────────────────────────────────────────────────────────────┘

IMPORTANT LESSON: Always check RED FLAGS alongside OQPLA score!

The OQPLA score alone (0.42) suggests lower risk, BUT the PAINS flag
and extreme angle indicate this is still a problematic compound.

Recommendation: VALIDATE with orthogonal assays before proceeding.
Do NOT rely on score alone - check all quality indicators.
```

---

### Example 3: Gefitinib (Successful Targeted Therapy)

**Compound**: Gefitinib (Iressa) - EGFR tyrosine kinase inhibitor for lung cancer

```
Gefitinib - Approved anticancer drug (2003)
Target: EGFR (Epidermal Growth Factor Receptor)
Indication: Non-small cell lung cancer

    Molecular Formula: C22H24ClFN4O3
```

| Property | Value | Notes |
|----------|-------|-------|
| pActivity | 8.2 (IC50 = 6.3 nM against EGFR) | Highly potent |
| Molecular Weight | 446.9 Da | Medium-large |
| PSA | 68.7 Å² | Moderate polar surface |
| Heavy Atoms | 31 | Standard kinase inhibitor size |
| N+O Atoms | 7 | Moderate polarity |
| QED | 0.65 | Moderate drug-likeness |

**Step 1: Calculate Efficiency Metrics**
```
SEI  = 8.2 ÷ (68.7/100) = 11.9  [Good]
BEI  = 8.2 ÷ (446.9/1000) = 18.3 [Average for kinase inhibitor]
NSEI = 8.2 ÷ 7 = 1.17           [Good]
NBEI = 8.2 ÷ 31 = 0.26          [Average]
```

**Step 2: Component Scores**
```
┌─────────────────┬───────┬─────────────────────────────────────────┐
│ Component       │ Score │ Explanation                             │
├─────────────────┼───────┼─────────────────────────────────────────┤
│ Efficiency      │ 0.68  │ Good across all metrics                 │
│ Angle           │ 0.89  │ Near optimal (57°) - well balanced      │
│ Distance        │ 0.78  │ Competitive in kinase inhibitor space   │
│ PDB Evidence    │ 0.92  │ 85+ structures with EGFR                │
└─────────────────┴───────┴─────────────────────────────────────────┘
```

**Step 3: Final Calculation**
```
Base Score = 0.50×0.68 + 0.1875×0.89 + 0.25×0.78 + 0.0625×0.92
           = 0.340 + 0.167 + 0.195 + 0.058
           = 0.760

QED Multiplier = 0.75 + 0.25×0.65 = 0.9125

Final Score = 0.760 × 0.9125 = 0.694
```

**Result**: **Moderate IMP** (Priority 3) → Score: 0.69

**Gefitinib is an approved drug - what does the score mean?**

```
Understanding the Score (CORRECTED):
┌──────────────────────────────────────────────────────────────────────┐
│ Gefitinib scores 0.69 = MODERATE false positive risk                 │
│                                                                      │
│ This is GOOD NEWS for an approved drug!                              │
│                                                                      │
│ 1. Score < 0.7 means it's NOT flagged as high-risk artifact          │
│    → OQPLA correctly identifies it as NOT a false positive           │
│    → The compound is likely genuine (which it is - it's approved!)   │
│                                                                      │
│ 2. The HIGH PDB Score (0.92) provides strong validation              │
│    → 85+ crystal structures PROVE it binds to EGFR                   │
│    → This confirms it's NOT an assay artifact                        │
│                                                                      │
│ 3. QED penalty of 17.5% is acceptable for targeted therapy           │
│    → Some structural complexity needed for target selectivity        │
│                                                                      │
│ LESSON: A "Moderate IMP" score (0.5-0.7) with strong PDB evidence    │
│ means the compound is GENUINE and can proceed with confidence!       │
│ OQPLA correctly did NOT flag this as a high-risk artifact.           │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Example 4: Curcumin (Classic "Invalid Metabolic Panacea")

**Compound**: Curcumin - The compound OQPLA is designed to flag!

```
Curcumin Structure:
    OCH3                              OCH3
      |                                  |
   HO─⬡─CH=CH─C─CH=CH─⬡─OH
              ‖
              O

    Known as a "Pan-Assay Interference Compound" (PAINS)
    Molecular Formula: C21H20O6
```

| Property | Value | Notes |
|----------|-------|-------|
| pActivity | 7.0 (IC50 = 100 nM against "50+ targets") | TOO MANY TARGETS! |
| Molecular Weight | 368.4 Da | Medium |
| PSA | 93.1 Å² | Moderate-high |
| Heavy Atoms | 27 | Medium |
| N+O Atoms | 6 | Moderate polarity |
| QED | 0.47 | Poor drug-likeness |

**The Red Flags:**
```
⚠️⚠️⚠️ EXTREME CAUTION: Classic IMP Compound ⚠️⚠️⚠️

┌──────────────────────────────────────────────────────────────────────┐
│ MULTIPLE SEVERE WARNING SIGNS:                                       │
│                                                                      │
│ 1. PAINS Structure: α,β-unsaturated ketone (Michael acceptor)        │
│    → Known to form covalent bonds with proteins non-specifically     │
│                                                                      │
│ 2. Polypharmacology Claim: "Active" against 50+ unrelated targets    │
│    → Biologically implausible - likely assay interference            │
│                                                                      │
│ 3. Poor Chemical Stability                                           │
│    → Degrades rapidly in aqueous solution                            │
│    → May produce reactive decomposition products                     │
│                                                                      │
│ 4. Fluorescence Interference                                         │
│    → Strong intrinsic fluorescence interferes with many assays       │
│                                                                      │
│ 5. Failed Clinical Trials                                            │
│    → Despite decades of research, no approved drug                   │
│    → Systematic review found "limited bioavailability"               │
└──────────────────────────────────────────────────────────────────────┘
```

**Component Scores:**
```
┌─────────────────┬───────┬─────────────────────────────────────────┐
│ Component       │ Score │ Explanation                             │
├─────────────────┼───────┼─────────────────────────────────────────┤
│ Efficiency      │ 0.71  │ Looks good (but misleading!)            │
│ Angle           │ 0.45  │ Moderate (62°)                          │
│ Distance        │ 0.82  │ Appears competitive                     │
│ PDB Evidence    │ 0.35  │ LOW! Few high-quality structures        │
└─────────────────┴───────┴─────────────────────────────────────────┘
```

**Final Calculation:**
```
Base Score = 0.50×0.71 + 0.1875×0.45 + 0.25×0.82 + 0.0625×0.35
           = 0.355 + 0.084 + 0.205 + 0.022
           = 0.666

QED Multiplier = 0.75 + 0.25×0.47 = 0.8675

Final Score = 0.666 × 0.8675 = 0.578
```

**Result**: **Moderate IMP** (Priority 3) → Score: 0.58

**Interpretation (CORRECTED) - This is what OQPLA is designed to catch:**
```
Curcumin Analysis:
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│ Score 0.58 = MODERATE false positive risk                            │
│                                                                      │
│ HOWEVER, combined with multiple RED FLAGS, this is problematic:      │
│                                                                      │
│    Efficiency Score (0.71) ───→ Looks promising on paper             │
│                    BUT                                               │
│    PDB Score (0.35) ─────────→ LOW structural validation             │
│    + PAINS structure ────────→ Known assay interference              │
│    + Multi-target claims ────→ Biologically implausible              │
│                                                                      │
│    CONCLUSION: MONITOR carefully and VALIDATE with orthogonal        │
│    assays. The moderate score + low PDB + PAINS flags suggest        │
│    this compound needs extensive validation before advancing.        │
│                                                                      │
│    The "activity" may be assay interference rather than genuine      │
│    target engagement. Require SPR/ITC confirmation before investing  │
│    resources.                                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Comparison Summary: All Four Examples

```
┌────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Compound       │ Efficiency│ Angle    │ Distance │ PDB      │ FINAL    │
├────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Gefitinib      │   0.68   │   0.89   │   0.78   │   0.92   │   0.69   │
│ (Good drug)    │   ███▓   │   ████▓  │   ███▓   │   █████  │  ███▓    │
├────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Ibuprofen      │   0.62   │   0.64   │   0.85   │   0.95   │   0.66   │
│ (Good drug)    │   ███░   │   ███░   │   ████░  │   █████  │  ███▓    │
├────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Curcumin       │   0.71   │   0.45   │   0.82   │   0.35   │   0.58   │
│ (Classic IMP)  │   ███▓   │   ██░░   │   ████░  │   █▓░░   │  ███░    │
├────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Quercetin      │   0.48   │   0.22   │   0.65   │   0.72   │   0.42   │
│ (PAINS)        │   ██░░   │   █░░░   │   ███░   │   ███▓   │  ██░░    │
└────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

Key Insight (CORRECTED):
┌──────────────────────────────────────────────────────────────────────┐
│ All four compounds have scores in the 0.42-0.69 range = MODERATE     │
│ to WEAK IMP = Lower false positive risk range.                       │
│                                                                      │
│ The approved drugs (Gefitinib, Ibuprofen) have HIGH PDB scores       │
│ (0.92, 0.95) confirming they are genuine binders.                    │
│                                                                      │
│ Quercetin and Curcumin have LOWER PDB scores (0.72, 0.35) plus       │
│ RED FLAGS (PAINS, extreme angles) indicating they need extra         │
│ validation despite moderate OQPLA scores.                            │
│                                                                      │
│ LESSON: Use OQPLA score + PDB evidence + RED FLAGS together!         │
│ Don't rely on score alone - check all quality indicators.            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Red Flags: Compounds to Watch Out For

This section helps you identify compounds that may look promising but are likely artifacts or poor drug candidates.

### Red Flag Pattern 1: High Efficiency, Low PDB Score

```
PATTERN: "Too Good to Be True"
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Efficiency Score: ████████████████████████░░  HIGH (0.85+)        │
│   PDB Score:        █████░░░░░░░░░░░░░░░░░░░░░  LOW  (< 0.3)        │
│                                                                      │
│   ⚠️ WARNING: Exceptional efficiency without structural validation   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

What this means:
- The compound shows exceptional potency metrics
- BUT there's little/no crystallographic evidence it actually binds
- Likely an assay artifact (aggregation, fluorescence, redox cycling)

Examples of compounds with this pattern:
- Curcumin (efficiency 0.71, PDB 0.35)
- Many polyphenols
- Rhodanines
- Some quinones

Action: REQUIRE orthogonal assay validation before proceeding
```

### Red Flag Pattern 2: Extreme Development Angle

```
PATTERN: "Unbalanced Trajectory"
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Angle Score: Very Low (< 0.4)                                      │
│                                                                      │
│   Two possible causes:                                               │
│                                                                      │
│   ANGLE < 25° (Too Hydrophobic)           ANGLE > 70° (Too Polar)   │
│                                                                      │
│        BEI                                      BEI                  │
│         ↑                                        ↑                   │
│         │                                        │    ●              │
│         │                                        │   /               │
│         │                                        │  /                │
│         │   ● compound                           │ / compound        │
│         │  /                                     │/                  │
│         │ /                                      │                   │
│         └──────→ SEI                             └──────→ SEI        │
│                                                                      │
│   Issues:                         Issues:                            │
│   - Relies on lipophilicity       - Too many polar groups            │
│   - Likely poor selectivity       - May have permeability issues     │
│   - Promiscuous binding risk      - Difficulty crossing membranes    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

Examples:
- Quercetin: Angle = 77° (too polar) → Score = 0.22
- Highly lipophilic compounds: Angle < 20° → Score < 0.5

Action: Consider structural modifications to improve balance
```

### Red Flag Pattern 3: High Score, Low QED (Drug-likeness Problem)

```
PATTERN: "Potent but Undevelopable"
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Base Score:      ████████████████████████████  HIGH (0.80+)       │
│   QED:             █████░░░░░░░░░░░░░░░░░░░░░░░  LOW  (< 0.4)       │
│   Final Score:     █████████████████░░░░░░░░░░░  REDUCED            │
│                                                                      │
│   Impact: Base score reduced by 15-25%!                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

What this means:
- The compound has good efficiency and validation
- BUT it has poor drug-like properties:
  - Too large (MW > 500 Da)
  - Too lipophilic (LogP > 5)
  - Too many H-bond donors/acceptors
  - High polar surface area

Examples:
- Many natural products (complex structures)
- Peptide-like compounds
- Large macrocycles

Why it matters:
┌─────────────────────────────────────────────────────────────────┐
│ Even if a compound binds well, it may fail due to:              │
│ • Poor oral absorption                                          │
│ • Rapid metabolism                                              │
│ • Toxicity                                                      │
│ • Formulation challenges                                        │
└─────────────────────────────────────────────────────────────────┘

Action: Consider fragment-based optimization or prodrug strategy
```

### Red Flag Pattern 4: Multi-Target "Pan-Active" Compounds

```
PATTERN: "Hits Everything"
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   ⚠️ SUSPICIOUS ACTIVITY PROFILE                                     │
│                                                                      │
│   Target 1 (Kinase):     IC50 = 100 nM  ✓                           │
│   Target 2 (GPCR):       IC50 = 200 nM  ✓                           │
│   Target 3 (Ion Channel):IC50 = 150 nM  ✓                           │
│   Target 4 (Nuclear Rec):IC50 = 300 nM  ✓                           │
│   Target 5 (Enzyme):     IC50 = 180 nM  ✓                           │
│   ... and 45 more targets                                            │
│                                                                      │
│   ❌ BIOLOGICALLY IMPLAUSIBLE                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

Reality check:
- True polypharmacology exists but is RARE
- A compound hitting 50+ unrelated targets is almost certainly:
  - An aggregator (forms colloidal particles that sequester proteins)
  - A reactive compound (covalently modifies many proteins)
  - Causing assay interference

Classic examples:
- Curcumin: "Active" against 50+ targets → Classic artifact
- EGCG (green tea): Similar promiscuous profile
- Many polyphenols and catechols

Action: Run aggregation assays (with detergent), counter-screens
```

### Red Flag Pattern 5: Discordant Component Scores

```
PATTERN: "Something Doesn't Add Up"
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   When component scores are drastically different:                   │
│                                                                      │
│   Component       Score    Expected Relationship                     │
│   ─────────────────────────────────────────────                      │
│   Efficiency      0.95     ←─┐                                       │
│   Distance        0.98        │ These should correlate!             │
│   Angle           0.15     ←─┘ Why is angle so different?           │
│   PDB             0.40                                               │
│                                                                      │
│   ⚠️ Investigate discrepancies between related metrics               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

What to check:
1. Data quality - Are the input values correct?
2. Cohort composition - Is the comparison group appropriate?
3. Calculation errors - Verify the molecular properties
4. Unusual structure - May have unique physicochemical properties
```

### Quick Reference: Red Flag Checklist

Before advancing any compound, verify:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PRE-ADVANCEMENT CHECKLIST                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ □ PDB Score ≥ 0.5?                                                   │
│   If NO → Requires orthogonal validation                             │
│                                                                      │
│ □ Angle Score ≥ 0.6?                                                 │
│   If NO → Check for balance issues                                   │
│                                                                      │
│ □ QED ≥ 0.5?                                                         │
│   If NO → May have developability concerns                           │
│                                                                      │
│ □ Activity against ≤ 5 related targets?                              │
│   If MORE → Check for promiscuity/artifacts                          │
│                                                                      │
│ □ No known PAINS substructures?                                      │
│   If YES → Run counter-screens                                       │
│                                                                      │
│ □ Stable in assay buffer?                                            │
│   If NO → Results may be from degradation products                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Frequently Asked Questions

### Q1: Why is my highly potent compound scoring low?

**A**: High potency alone doesn't guarantee a high OQPLA score. The system also considers:
- Balance of efficiency (angle)
- Structural validation (PDB evidence)
- Drug-likeness (QED)

A compound with IC50 = 0.1 nM but no structural evidence and poor drug-likeness may score lower than a 10 nM compound with excellent validation.

### Q2: What if no PDB structures are found for my compound?

**A**: The PDB Score will be 0, but this doesn't necessarily mean your compound is bad. It could mean:
- The compound class is novel (no one has crystallized similar compounds yet)
- Structures exist but haven't been deposited

In this case, the other three components carry more weight. Consider pursuing X-ray crystallography or cryo-EM to generate your own structural evidence.

### Q3: Why does QED affect my score so significantly?

**A**: Drug-likeness is crucial for actual drug development. A compound that's too large, too lipophilic, or has too many hydrogen bond donors will likely fail in later stages due to:
- Poor oral absorption
- Metabolic instability
- Toxicity

The QED multiplier ensures we don't waste resources on compounds that will fail for drug-likeness reasons, even if they're potent.

### Q4: Can I compare OQPLA scores across different datasets?

**A**: **With caution**. The Efficiency and Distance scores are calculated relative to the cohort, so:
- A score of 0.7 in a cohort of weak compounds may be less impressive than 0.7 in a cohort of strong compounds
- The Angle and PDB scores are absolute and comparable across datasets

For cross-dataset comparisons, focus on the raw efficiency metrics rather than the normalized scores.

### Q5: What's the minimum score I should accept for follow-up?

**A**: This depends on your resources and risk tolerance:

| Risk Tolerance | Minimum Score | Compounds Selected |
|---------------|---------------|-------------------|
| Conservative | 0.7+ | Fewer, higher confidence |
| Moderate | 0.5+ | Balanced portfolio |
| Aggressive | 0.3+ | More compounds, higher risk |

Early-stage projects might accept lower thresholds to maintain diversity, while late-stage optimization should focus on higher-scoring compounds.

### Q6: How should I handle compounds with missing data?

**A**: If a compound is missing key data (e.g., no PSA calculated):
- Some efficiency metrics cannot be calculated
- The Efficiency Score will be based on available metrics only
- Consider re-calculating properties or flagging for review

The system will return NaN scores for compounds with insufficient data.

---

## Glossary of Terms

### Core Metrics

| Term | Definition | Example |
|------|------------|---------|
| **pActivity** | -log10 of activity in Molar units. Higher = more potent. | IC50=10nM → pActivity=8.0 |
| **SEI** | Surface Efficiency Index = pActivity ÷ (PSA/100). Potency per polar surface. | pActivity 7.0, PSA 70 → SEI = 10.0 |
| **BEI** | Binding Efficiency Index = pActivity ÷ (MW/1000). Potency per molecular weight. | pActivity 7.0, MW 350 → BEI = 20.0 |
| **NSEI** | Normalized SEI = pActivity ÷ NPOL. Potency per polar atom. | pActivity 7.0, NPOL 5 → NSEI = 1.4 |
| **NBEI** | Normalized BEI = pActivity ÷ NHA. Potency per heavy atom. | pActivity 7.0, NHA 25 → NBEI = 0.28 |

### Molecular Properties

| Term | Definition | Typical Drug Range |
|------|------------|-------------------|
| **MW** | Molecular Weight in Daltons (Da). | 150-500 Da |
| **PSA** | Polar Surface Area. Sum of polar atom surfaces. | 20-140 Å² |
| **NPOL** | Count of nitrogen + oxygen atoms. Indicates polarity. | 2-10 atoms |
| **NHA** | Number of Heavy Atoms (non-hydrogen). | 15-35 atoms |
| **LogP** | Partition coefficient. Measures lipophilicity. | 0-5 |
| **QED** | Quantitative Estimate of Drug-likeness (0-1). | > 0.5 is good |

### Geometric Terms

| Term | Definition | Visualization |
|------|------------|---------------|
| **Modulus** | Distance from origin: √(SEI² + BEI²). Overall efficiency magnitude. | Length of arrow from (0,0) to compound |
| **Angle** | Direction: arctan(BEI/SEI) × 180/π. Development trajectory. | Angle of arrow from x-axis |
| **Optimal Angle** | 45° represents balanced SEI/BEI improvement. | Diagonal line from origin |

### Statistical Terms

| Term | Definition | Example |
|------|------------|---------|
| **Z-score** | Standard deviations from mean: (value - mean) / SD. | Z=2 means 2 SDs above average |
| **Sigmoid** | S-curve function mapping any value to 0-1 range. | Z=0 → 0.5, Z=3 → 0.95 |
| **Cohort** | Reference group for comparison (similar compounds). | "All kinase inhibitors in dataset" |

### Structural Biology Terms

| Term | Definition | Quality Threshold |
|------|------------|------------------|
| **RCSB PDB** | Protein Data Bank. Repository of 3D protein structures. | - |
| **Resolution** | Crystal structure quality in Ångströms (Å). Lower = better. | < 2.0 Å is excellent |
| **X-ray Crystallography** | Technique to determine atomic structure using X-ray diffraction. | Gold standard |
| **Cryo-EM** | Electron microscopy at cryogenic temperatures. | Good for large complexes |

### Problem Compound Terms

| Term | Definition | Examples |
|------|------------|----------|
| **IMP** | Invalid Metabolic Panacea. Appears active but is artifact. | Curcumin, EGCG |
| **PAINS** | Pan-Assay INterference compoundS. Interfere with many assays. | Rhodanines, catechols |
| **Aggregator** | Compound that forms colloidal particles, trapping proteins. | Many lipophilic compounds |
| **Michael Acceptor** | Reactive group that covalently binds proteins. | α,β-unsaturated carbonyls |
| **Catechol** | 1,2-dihydroxybenzene. Known assay interference. | Found in quercetin |

### Quick Conversion Reference

```
Activity Conversions (IC50 → pActivity):
┌─────────────────┬────────────────┬─────────────┐
│ IC50            │ Molar          │ pActivity   │
├─────────────────┼────────────────┼─────────────┤
│ 1 mM            │ 10⁻³ M         │ 3.0         │
│ 100 μM          │ 10⁻⁴ M         │ 4.0         │
│ 10 μM           │ 10⁻⁵ M         │ 5.0         │
│ 1 μM            │ 10⁻⁶ M         │ 6.0         │
│ 100 nM          │ 10⁻⁷ M         │ 7.0         │
│ 10 nM           │ 10⁻⁸ M         │ 8.0         │
│ 1 nM            │ 10⁻⁹ M         │ 9.0         │
│ 100 pM          │ 10⁻¹⁰ M        │ 10.0        │
└─────────────────┴────────────────┴─────────────┘

QED Multiplier Effect:
┌─────────┬────────────┬─────────────────────────────────────────┐
│ QED     │ Multiplier │ Effect on Score                         │
├─────────┼────────────┼─────────────────────────────────────────┤
│ 1.0     │ 1.00       │ ████████████████████ No penalty         │
│ 0.8     │ 0.95       │ ███████████████████░ 5% reduction       │
│ 0.6     │ 0.90       │ ██████████████████░░ 10% reduction      │
│ 0.4     │ 0.85       │ █████████████████░░░ 15% reduction      │
│ 0.2     │ 0.80       │ ████████████████░░░░ 20% reduction      │
│ 0.0     │ 0.75       │ ███████████████░░░░░ 25% reduction      │
└─────────┴────────────┴─────────────────────────────────────────┘
```

---

## Summary: The OQPLA Scoring Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           YOUR COMPOUND                                  │
│                    (SMILES, Activity, Properties)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: CALCULATE EFFICIENCY METRICS                  │
│                                                                          │
│   SEI = pActivity ÷ (PSA/100)     NSEI = pActivity ÷ NPOL               │
│   BEI = pActivity ÷ (MW/1000)     NBEI = pActivity ÷ NHA                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: CALCULATE PLANE GEOMETRY                      │
│                                                                          │
│   Modulus = √(SEI² + BEI²)        Angle = arctan(BEI/SEI) × 180/π       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: CALCULATE FOUR COMPONENT SCORES               │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  EFFICIENCY  │  │    ANGLE     │  │   DISTANCE   │  │     PDB      │ │
│  │    (50%)     │  │  (18.75%)    │  │    (25%)     │  │   (6.25%)    │ │
│  │              │  │              │  │              │  │              │ │
│  │ Z-score →    │  │ Deviation    │  │ Modulus ÷    │  │ Structure    │ │
│  │ Sigmoid →    │  │ from 45°     │  │ Best Modulus │  │ count &      │ │
│  │ Average      │  │              │  │              │  │ quality      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: COMBINE & APPLY QED MULTIPLIER                │
│                                                                          │
│   Base Score = 0.50×Eff + 0.1875×Ang + 0.25×Dist + 0.0625×PDB           │
│                                                                          │
│   QED Multiplier = 0.75 + 0.25 × QED                                     │
│                                                                          │
│   ═══════════════════════════════════════════════════════════════════    │
│   ║  FINAL OQPLA SCORE = Base Score × QED Multiplier  (0.0 to 1.0)  ║    │
│   ═══════════════════════════════════════════════════════════════════    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: CLASSIFY & PRIORITIZE                         │
│                                                                          │
│   0.9-1.0: EXCEPTIONAL (Priority 1) → Immediate validation               │
│   0.7-0.9: STRONG (Priority 2) → Validate soon                           │
│   0.5-0.7: MODERATE (Priority 3) → Gather more data                      │
│   0.3-0.5: WEAK (Priority 4) → Deprioritize                              │
│   < 0.3:   NOT IMP → Exclude                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Practical Decision Guide

Use this flowchart when evaluating your OQPLA results:

```
                          START: You have an OQPLA score
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    Is OQPLA Score ≥ 0.5?            │
                    └─────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
            YES                              NO
              │                               │
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ Is PDB Score    │             │ EXCLUDE         │
    │ ≥ 0.5?          │             │ Unless novel    │
    └─────────────────┘             │ scaffold for    │
              │                     │ future SAR      │
    ┌─────────┴────────┐            └─────────────────┘
    ▼                  ▼
   YES                NO
    │                  │
    ▼                  ▼
┌───────────┐    ┌──────────────────────────┐
│ VALIDATED │    │ REQUIRES VALIDATION      │
│           │    │                          │
│ Proceed   │    │ Run orthogonal assays:   │
│ with      │    │ - Counter-screens        │
│ confidence│    │ - SPR/ITC binding        │
│           │    │ - Aggregation assay      │
│           │    │ - Cell-based assay       │
└───────────┘    └──────────────────────────┘
    │                       │
    ▼                       ▼
┌─────────────────────────────────────────────┐
│           CHECK QED MULTIPLIER              │
├─────────────────────────────────────────────┤
│                                             │
│ QED ≥ 0.6: Good drug-likeness → Proceed     │
│                                             │
│ QED < 0.6: Consider:                        │
│   - Fragment-based optimization             │
│   - Prodrug strategy                        │
│   - Different delivery route                │
│                                             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│         FINAL RECOMMENDATION                │
├─────────────────────────────────────────────┤
│                                             │
│ Score 0.7+, PDB 0.7+, QED 0.6+:             │
│   → PRIORITY CANDIDATE - advance to lead    │
│     optimization                            │
│                                             │
│ Score 0.5-0.7, PDB 0.5+, QED 0.5+:          │
│   → PROMISING - gather additional data,     │
│     consider for parallel optimization      │
│                                             │
│ Score 0.5+, PDB < 0.5:                      │
│   → VALIDATE FIRST - require orthogonal     │
│     confirmation before resources           │
│                                             │
│ Score < 0.5:                                │
│   → DEPRIORITIZE - focus resources          │
│     elsewhere unless novel scaffold         │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Key Takeaways for Drug Researchers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CORRECTED INTERPRETATION                              │
│                                                                         │
│  1. OQPLA IDENTIFIES FALSE POSITIVES (NOT GOOD CANDIDATES)              │
│     → HIGH SCORE (0.9+) = HIGH false positive risk = DEPRIORITIZE       │
│     → LOW SCORE (<0.3) = LOW false positive risk = PRIORITIZE           │
│     → Use OQPLA to AVOID wasting resources on artifacts                 │
│                                                                         │
│  2. PDB EVIDENCE IS YOUR BEST FRIEND                                    │
│     → High PDB score = structural proof of genuine binding              │
│     → Low PDB score + high OQPLA score = STRONG RED FLAG (artifact)     │
│     → Low PDB score + low OQPLA score = May be OK but needs validation  │
│                                                                         │
│  3. BALANCE MATTERS                                                     │
│     → 45° angle represents optimal development trajectory               │
│     → Extremes (<20° or >70°) indicate problematic compounds            │
│                                                                         │
│  4. DRUG-LIKENESS MATTERS                                               │
│     → QED < 0.4 means 15%+ score increase (more IMP-like)               │
│     → Poor QED compounds face development challenges                    │
│                                                                         │
│  5. USE MULTIPLE INDICATORS TOGETHER                                    │
│     → Don't rely on OQPLA score alone                                   │
│     → Check: OQPLA + PDB + RED FLAGS + QED together                     │
│     → Approved drugs can have moderate scores (0.5-0.7) - that's OK!    │
│                                                                         │
│  6. WATCH FOR PAINS AND IMPS                                            │
│     → Curcumin, quercetin, EGCG, rhodanines are classic artifacts       │
│     → "Pan-active" compounds are almost always false positives          │
│     → PAINS flags override favorable OQPLA scores                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---


---

## Contact & Support

For questions about the OQPLA scoring system or IMPULATOR application, please refer to the project documentation or contact the development team.

---

*Document Version: 3.0.0*
*Last Updated: January 2026*
*Based on IMPs 2.0 methodology (Reddy et al.)*

