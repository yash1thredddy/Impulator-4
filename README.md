---
title: IMPs Navigator
sdk: docker
app_port: 7860
license: other
---

# 🔬 IMPs Navigator (Impulator 3)

[![CI](https://github.com/yash1thredddy/Impulator-4/actions/workflows/ci.yml/badge.svg)](https://github.com/yash1thredddy/Impulator-4/actions/workflows/ci.yml) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-Proprietary-red)

**Compound Library & Analysis Tool for better Insights**

A powerful Streamlit application for analyzing chemical compounds, calculating efficiency indices (SEI, BEI, etc.), and integrating data from ChEMBL and PDB.

## What is IMPULATOR?

IMPULATOR (IMPs Navigator) is a scientific web application for analyzing chemical compounds to identify **Invalid Metabolic Panaceas (IMPs)** — compounds that appear to have exceptional bioactivity but are actually assay artifacts.

### Core Features

1. **Compound Analysis** — Process compounds via SMILES/InChI input
2. **Similarity Search** — Find similar compounds in ChEMBL
3. **Efficiency Metrics** — Calculate SEI, BEI, NSEI, NBEI
4. **IMP Scoring** — Multi-criteria quality assessment
5. **IMP Classification** — Identify potential false positives
6. **Assay Interference** — Detect PAINS, aggregation, redox issues
7. **Batch Processing** — Process CSV files with multiple compounds
8. **Visualization** — Interactive Plotly charts, 3D molecule viewer
9. **Report Generation** — Comprehensive HTML reports with charts, exportable for sharing
10. **Versions Tab** — View all structural siblings (same InChIKey) with config diff highlighting and navigation

**Requirements:** Python 3.11+, Supabase Postgres database

## 🚀 Quick Start

### Run Locally
```bash
# Install dependencies (editable install with test extras)
pip install -e ".[test]"

# Set DATABASE_URL to your Supabase Postgres connection
export DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Run the app (starts both backend + frontend)
./start.sh
```

### Documentation
*   **[Architecture](.claude/docs/architecture.md)**: System design and architecture overview.
*   **[IMP Score Methodology](IMP_Score.md)**: Multi-criteria scoring system for compound prioritization.

## 🧪 Key Features
*   **Compound Analysis**: Automated retrieval of bioactivity data.
*   **Efficiency Metrics**: Calculate SEI, BEI, NSEI, NBEI.
*   **IMP Scoring**: Multi-component scoring system for compound prioritization.
*   **Assay Interference Detection**: PAINS, aggregation, redox, fluorescence, thiol reactivity filters.
*   **PDB Structural Evidence**: Integration with RCSB Protein Data Bank for validation.
*   **Visualizations**: Interactive Plotly charts and 3D molecule viewing.
*   **Data Integration**: ChEMBL, RCSB PDB, and NPClassifier.
*   **Cloud Storage**: Azure Blob Storage integration for persistence.
