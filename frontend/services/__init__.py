"""Services for IMPULATOR frontend."""
from frontend.services.backend_client import (
    ImpulatorAPIClient,
    get_api_client,
    set_session_id,
    JobResponse,
    CompoundListResponse,
)
from frontend.services.azure_storage import (
    is_azure_configured,
    list_results,
    list_local_results,
    download_result,
    load_result_dataframe,
    load_result_json,
    get_result_files,
    get_cached_result,
    get_cache_stats,
    delete_from_cache,
    clear_cache as clear_azure_cache,
)

__all__ = [
    # Backend client
    "ImpulatorAPIClient",
    "get_api_client",
    "set_session_id",
    "JobResponse",
    "CompoundListResponse",
    # Azure storage (direct read with local cache)
    "is_azure_configured",
    "list_results",
    "list_local_results",
    "download_result",
    "load_result_dataframe",
    "load_result_json",
    "get_result_files",
    "get_cached_result",
    "get_cache_stats",
    "delete_from_cache",
    "clear_azure_cache",
]
