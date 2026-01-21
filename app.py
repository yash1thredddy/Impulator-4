"""IMPULATOR Application Entry Point.

Simple redirect to the frontend app.

For development: python start.py (starts both backend and frontend)
For frontend only: streamlit run app.py (requires backend running separately)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the frontend app
from frontend.app import main

if __name__ == "__main__":
    main()
