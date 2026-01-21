"""Analyze page for IMPULATOR.

Provides the interface for submitting new compound analysis jobs.
"""

import logging

import streamlit as st

from frontend.utils import SessionState
from frontend.ui.components import render_job_form, render_csv_upload_form

logger = logging.getLogger(__name__)


def render_analyze_page() -> None:
    """Render the analyze page with job submission form."""
    # Header with back button
    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("⬅ Back", key="analyze_back_btn", width='stretch'):
            SessionState.navigate_to_home()
            st.rerun()

    with col2:
        st.title("New Analysis")

    st.caption("Submit a compound for IMP analysis")

    # Tabs for single vs batch
    tab1, tab2 = st.tabs(["Single Compound", "Batch Upload"])

    with tab1:
        render_single_analysis()

    with tab2:
        render_batch_analysis()


def render_single_analysis() -> None:
    """Render the single compound analysis form."""
    # Check if we just submitted a job (stored in session state)
    just_submitted = SessionState.get('just_submitted_job', False)
    last_job_id = SessionState.get('last_submitted_job_id', None)

    job_id = render_job_form()

    # If a new job was submitted, store it in session state
    if job_id:
        SessionState.set('just_submitted_job', True)
        SessionState.set('last_submitted_job_id', job_id)
        just_submitted = True
        last_job_id = job_id

    if just_submitted and last_job_id:
        # Job submitted successfully
        st.success("Analysis job submitted!")
        st.info("The job is now processing. You can monitor progress in the sidebar.")

        # Option to view status or start another
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏠 Go to Home", key="go_home_after_submit", width='stretch'):
                # Clear submission state
                SessionState.set('just_submitted_job', False)
                SessionState.set('last_submitted_job_id', None)
                SessionState.navigate_to_home()
                st.rerun()

        with col2:
            if st.button("➕ Submit Another", key="submit_another_btn", width='stretch'):
                # Clear form state
                SessionState.set('just_submitted_job', False)
                SessionState.set('last_submitted_job_id', None)
                SessionState.reset_processing_state()
                st.rerun()


def render_batch_analysis() -> None:
    """Render the batch upload form."""
    # Check if we just submitted a batch (stored in session state)
    just_submitted_batch = SessionState.get('just_submitted_batch', False)
    last_batch_id = SessionState.get('last_submitted_batch_id', None)

    # Show post-submission UI if batch was just submitted
    if just_submitted_batch and last_batch_id:
        st.success("Batch submitted successfully!")
        st.info("Jobs are now processing. You can monitor progress in the sidebar.")

        # Option to go home or submit another batch
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🏠 Go to Home", key="go_home_after_batch", width='stretch'):
                # Clear submission state
                SessionState.set('just_submitted_batch', False)
                SessionState.set('last_submitted_batch_id', None)
                _clear_batch_form_state()
                SessionState.navigate_to_home()
                st.rerun()

        with col2:
            if st.button("➕ Submit Another Batch", key="submit_another_batch_btn", width='stretch'):
                # Clear form state for new submission
                SessionState.set('just_submitted_batch', False)
                SessionState.set('last_submitted_batch_id', None)
                _clear_batch_form_state()
                st.rerun()

        return

    # Normal batch upload form
    st.markdown("### Batch Processing")
    st.info("""
    Upload a CSV file to analyze multiple compounds at once.

    **Required columns:**
    - `compound_name`: Unique name for each compound
    - `smiles` or `inchi`: Chemical structure

    Each compound will be submitted as a separate job.
    """)

    batch_id = render_csv_upload_form()

    # If a batch was submitted, store it in session state
    if batch_id:
        SessionState.set('just_submitted_batch', True)
        SessionState.set('last_submitted_batch_id', batch_id)
        st.rerun()


def _clear_batch_form_state():
    """Clear batch form related session state."""
    keys_to_clear = [
        'csv_preview',
        'batch_duplicate_check_done',
        'batch_user_confirmed',
        'batch_existing',
        'batch_processing',
        'batch_new'
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
