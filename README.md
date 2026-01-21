---
title: IMPs Navigator
emoji: 🔬
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🔬 IMPs Navigator (Impulator 3)

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
*   **O[Q/P/L]A Scoring**: Multi-component scoring system for compound prioritization.
*   **Assay Interference Detection**: PAINS, aggregation, redox, fluorescence, thiol reactivity filters.
*   **PDB Structural Evidence**: Integration with RCSB Protein Data Bank for validation.
*   **Visualizations**: Interactive Plotly charts and 3D molecule viewing.
*   **Data Integration**: ChEMBL, RCSB PDB, and NPClassifier.
*   **Cloud Storage**: Azure Blob Storage integration for persistence.
