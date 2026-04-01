"""Streamlit-facing cached service wrappers."""

from typing import Optional

import streamlit as st

from frontend.services.backend_client import CompoundListResponse, get_api_client


@st.cache_data(ttl=60)
def get_compounds_cached(
    page: int = 1,
    per_page: int = 50,
    search: Optional[str] = None,
    include_duplicates: bool = False,
) -> CompoundListResponse:
    """Cached wrapper around ImpulatorAPIClient.get_compounds_from_db."""
    client = get_api_client()
    return client.get_compounds_from_db(
        page=page,
        per_page=per_page,
        search=search,
        include_duplicates=include_duplicates,
    )
