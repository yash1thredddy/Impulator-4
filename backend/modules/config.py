"""
Module configuration for IMPULATOR chemistry modules.
"""
from pathlib import Path
from typing import List

# Default activity types
ACTIVITY_TYPES: List[str] = ["IC50", "Ki", "Kd", "EC50"]

# API settings
CHEMBL_API_URL = "https://www.ebi.ac.uk/chembl/api/data"
PDB_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

# Cache settings
CACHE_SIZE = 2000

# API rate limiting
API_RATE_LIMIT = 10  # requests per second
API_TIMEOUT = 30  # seconds
PDB_API_DELAY = 0.2  # seconds between PDB requests (rate limiting)

# Batch processing
MAX_BATCH_SIZE = 50
MAX_WORKERS = 4

# Directories (can be overridden by backend settings)
DATA_DIR = Path("./data")
RESULTS_DIR = Path("./data/results")
