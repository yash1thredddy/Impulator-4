"""IMPULATOR Frontend Application.

Clean Streamlit app that uses modular components and backend API.
"""

import logging
import sys
from pathlib import Path

# Load environment variables from .env file BEFORE any other imports
# This ensures Azure credentials are available to all modules
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st  # noqa: E402

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.config.settings import config  # noqa: E402
from frontend.utils import SessionState, VIEW_HOME, VIEW_ANALYZE, VIEW_COMPOUND_DETAILS  # noqa: E402
from frontend.ui.components import render_sidebar  # noqa: E402
from frontend.ui.pages import (  # noqa: E402
    render_home_page,
    render_analyze_page,
    render_compound_detail_page,
)
from frontend.services import set_session_id  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    # Page configuration - must be first Streamlit command
    st.set_page_config(
        page_title=config.APP_NAME,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS for better styling
    _apply_custom_css()

    # Initialize session state with defaults
    SessionState.init_defaults()

    # Configure API client with session ID for user isolation
    # This ensures each browser session has unique job tracking
    session_id = SessionState.get_session_id()
    set_session_id(session_id)

    # Deep linking: read URL params and navigate if present
    params = st.query_params
    if "compound_id" in params and not SessionState.get('_deep_link_applied'):
        compound_id = params["compound_id"]
        tab = params.get("tab", "overview")
        # Navigate to compound detail (shows compound list with detail auto-opened)
        SessionState.navigate_to_compound(compound_id, entry_id=compound_id)
        SessionState.set('_deep_link_applied', True)

    # Render sidebar (includes fragment-based job polling)
    render_sidebar()

    # Route to appropriate page based on current view
    current_view = SessionState.get_current_view()

    if current_view == VIEW_HOME:
        render_home_page()

    elif current_view == VIEW_ANALYZE:
        render_analyze_page()

    elif current_view == VIEW_COMPOUND_DETAILS:
        render_compound_detail_page()

    else:
        # Fallback to home
        SessionState.navigate_to_home()
        render_home_page()


def _apply_custom_css():
    """Apply custom CSS styling.

    Uses Streamlit CSS variables (var(--...)) for dark-mode compatibility.
    """
    st.markdown("""
        <style>
        /* WARNING: Targets Streamlit internal selectors -- may break on Streamlit upgrade */
        /* Improve card container styling */
        div[data-testid="stVerticalBlock"] > div[style*="border"]:has(> div > div > button) {
            padding: 0.75rem;
        }

        /* WARNING: Targets Streamlit internal selectors -- may break on Streamlit upgrade */
        /* Better sidebar styling */
        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }

        /* WARNING: Targets Streamlit internal selectors -- may break on Streamlit upgrade */
        /* Improve button consistency */
        .stButton > button {
            font-size: 0.875rem;
        }

        /* WARNING: Targets Streamlit internal selectors -- may break on Streamlit upgrade */
        /* Progress bar styling — uses CSS variable for dark mode support */
        .stProgress > div > div > div {
            background-color: var(--primary-color);
        }

        /* WARNING: Targets Streamlit internal selectors -- may break on Streamlit upgrade */
        /* Toast notifications — uses CSS variables for dark mode support */
        div[data-testid="stToast"] {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
        }

        /* Code block styling */
        pre {
            font-size: 0.75rem;
            max-height: 100px;
            overflow-y: auto;
        }

        /* WARNING: Targets Streamlit internal selectors -- may break on Streamlit upgrade */
        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }

        /* Hide Streamlit footer only (keep menu for theme settings) */
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
