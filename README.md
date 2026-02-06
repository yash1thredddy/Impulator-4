---
title: IMPs Navigator
sdk: docker
app_port: 7860
license: other
---

# 🔬 IMPs Navigator (Impulator 3)

[![CI](https://github.com/yash1thredddy/Impulator-4/actions/workflows/ci.yml/badge.svg)](https://github.com/yash1thredddy/Impulator-4/actions/workflows/ci.yml) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-Proprietary-red)

**Compound Library & Analysis Tool for better Insights**

A powerful Streamlit application for analyzing chemical compounds, calculating efficiency indices (SEI, BEI, etc.), and integrating data from ChEMBL and PDB.

## 🚀 Quick Start

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Documentation
*   **[Output Schema](OUTPUT_SCHEMA.md)**: Complete CSV output format documentation.
*   **[Free Deployment Guide](FREE_DEPLOYMENT.md)**: Deploy to Streamlit Cloud or Local.
*   **[Hugging Face Guide](HUGGINGFACE_GUIDE.md)**: Deploy to HF Spaces (Best for performance).
*   **[Concurrency Guide](CONCURRENCY_GUIDE.md)**: Understanding multi-user support.

## 🧪 Key Features
*   **Compound Analysis**: Automated retrieval of bioactivity data.
*   **Efficiency Metrics**: Calculate SEI, BEI, NSEI, NBEI.
*   **IMP Scoring**: Multi-component scoring system for compound prioritization.
*   **Assay Interference Detection**: PAINS, aggregation, redox, fluorescence, thiol reactivity filters.
*   **PDB Structural Evidence**: Integration with RCSB Protein Data Bank for validation.
*   **Visualizations**: Interactive Plotly charts and 3D molecule viewing.
*   **Data Integration**: ChEMBL, RCSB PDB, and NPClassifier.
*   **Cloud Storage**: Azure Blob Storage integration for persistence.
