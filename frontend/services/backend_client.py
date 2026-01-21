"""
Backend API Client for IMPULATOR Frontend.

Provides type-safe access to backend API with retry logic and error handling.
"""

import json
import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from frontend.config.settings import config

logger = logging.getLogger(__name__)


@dataclass
class JobResponse:
    """Standardized response from job API."""
    success: bool
    job_id: Optional[str] = None
    status: Optional[str] = None
    progress: float = 0.0
    message: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None
    # For duplicate detection
    is_duplicate: bool = False
    duplicate_info: Optional[dict] = None


@dataclass
class CompoundListResponse:
    """Response from compound listing API."""
    success: bool
    compounds: List[dict] = None
    total: int = 0
    error: Optional[str] = None

    def __post_init__(self):
        if self.compounds is None:
            self.compounds = []


class ImpulatorAPIClient:
    """Client for IMPULATOR backend API with retry logic."""

    def __init__(
        self,
        base_url: str = None,
        timeout: int = None,
        max_retries: int = None,
        session_id: str = None
    ):
        """Initialize API client.

        Args:
            base_url: Backend API URL (default from config)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            session_id: Session ID for user isolation
        """
        self.base_url = base_url or config.API_BASE_URL
        self.timeout = timeout or config.API_TIMEOUT_SECONDS
        self.max_retries = max_retries or config.MAX_RETRY_ATTEMPTS
        self._session_id = session_id

        # Configure session with retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str):
        """Set session ID for user isolation."""
        self._session_id = value

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including session ID if set."""
        headers = {}
        if self._session_id:
            headers["X-Session-ID"] = self._session_id
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)

        # Merge session headers with any provided headers
        headers = self._get_headers()
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        kwargs['headers'] = headers

        try:
            response = self.session.request(method, url, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            raise

    # Health check
    def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            response = self._request('GET', '/api/v1/health')
            return response.status_code == 200
        except Exception:
            return False

    # Job Management
    def submit_job(
        self,
        compound_name: str,
        smiles: str,
        similarity_threshold: int = None,
        activity_types: List[str] = None
    ) -> JobResponse:
        """Submit a compound analysis job.

        Args:
            compound_name: Name of the compound
            smiles: SMILES string
            similarity_threshold: Similarity threshold (default from config)
            activity_types: Activity types to search

        Returns:
            JobResponse with job_id on success
        """
        payload = {
            "compound_name": compound_name,
            "smiles": smiles,
            "similarity_threshold": similarity_threshold or config.DEFAULT_SIMILARITY_THRESHOLD,
            "activity_types": activity_types or list(config.DEFAULT_ACTIVITY_TYPES),
        }

        try:
            response = self._request('POST', '/api/v1/jobs', json=payload)

            if response.status_code == 201:
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in success response: {e}")
                    return JobResponse(success=False, error="Invalid response from server")

                # Check for duplicate_found response
                if data.get('status') == 'duplicate_found':
                    return JobResponse(
                        success=False,
                        is_duplicate=True,
                        duplicate_info=data,
                        message=f"Duplicate compound found: {data.get('existing_compound', {}).get('compound_name', 'Unknown')}"
                    )

                return JobResponse(
                    success=True,
                    job_id=data.get('id'),
                    status=data.get('status'),
                    data=data
                )
            else:
                try:
                    error = response.json().get('detail', f"HTTP {response.status_code}")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}: {response.text[:100] if response.text else 'Unknown error'}"
                return JobResponse(success=False, error=error)

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    def check_duplicates(self, compound_names: List[str]) -> Dict[str, Any]:
        """Check which compounds already exist or are being processed.

        Args:
            compound_names: List of compound names to check

        Returns:
            Dict with 'existing', 'processing', and 'new' lists
        """
        payload = {"compound_names": compound_names}

        try:
            response = self._request('POST', '/api/v1/jobs/check-duplicates', json=payload)

            if response.status_code == 200:
                try:
                    return {"success": True, **response.json()}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in check-duplicates response: {e}")
                    return {"success": False, "error": "Invalid response from server"}
            else:
                try:
                    error = response.json().get('detail', f"HTTP {response.status_code}")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}"
                return {"success": False, "error": error}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def resolve_duplicate(
        self,
        action: str,
        smiles: str,
        compound_name: str,
        existing_entry_id: Optional[str] = None,
        new_compound_name: Optional[str] = None,
        similarity_threshold: int = None,
        activity_types: List[str] = None
    ) -> JobResponse:
        """Resolve a duplicate compound situation.

        Called after submit_job returns a duplicate_found response.

        Args:
            action: One of 'replace', 'duplicate', or 'skip'
            smiles: SMILES string of the compound
            compound_name: Original compound name
            existing_entry_id: Entry ID of existing compound (for replace/duplicate)
            new_compound_name: New name if user wants to change it
            similarity_threshold: Similarity threshold for new job
            activity_types: Activity types for new job

        Returns:
            JobResponse with job_id on success (for replace/duplicate),
            or success=True with status='skipped' (for skip)
        """
        payload = {
            "action": action,
            "smiles": smiles,
            "compound_name": compound_name,
            "existing_entry_id": existing_entry_id,
            "new_compound_name": new_compound_name,
            "similarity_threshold": similarity_threshold or config.DEFAULT_SIMILARITY_THRESHOLD,
            "activity_types": activity_types or list(config.DEFAULT_ACTIVITY_TYPES),
        }

        try:
            response = self._request('POST', '/api/v1/jobs/resolve-duplicate', json=payload)

            if response.status_code == 201:
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in resolve-duplicate response: {e}")
                    return JobResponse(success=False, error="Invalid response from server")

                # Check if skipped
                if data.get('status') == 'skipped':
                    return JobResponse(
                        success=True,
                        status='skipped',
                        message=data.get('message', 'Compound skipped'),
                        data=data
                    )

                # Job was created
                return JobResponse(
                    success=True,
                    job_id=data.get('id'),
                    status=data.get('status'),
                    data=data
                )
            else:
                try:
                    error = response.json().get('detail', f"HTTP {response.status_code}")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}"
                return JobResponse(success=False, error=error)

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    def submit_batch_job(
        self,
        compounds: List[Dict[str, Any]],
        duplicate_decisions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Submit a batch of compound analysis jobs.

        Args:
            compounds: List of compound dicts with compound_name, smiles, etc.
                      Each compound can have a 'duplicate_action' field
                      ('skip', 'replace', 'duplicate') for per-compound handling.
            duplicate_decisions: Dict mapping compound_name -> action ('skip', 'replace', 'duplicate')
                                for existing compounds. Provided for backward compatibility,
                                but actions are now included per-compound in the compounds list.

        Returns:
            Dict with batch_id, jobs list, skipped/replaced compounds info
        """
        payload = {
            "compounds": compounds,
        }

        # Include duplicate_decisions if provided (for backend processing)
        if duplicate_decisions:
            payload["duplicate_decisions"] = duplicate_decisions

        try:
            response = self._request('POST', '/api/v1/jobs/batch', json=payload)

            if response.status_code == 201:
                try:
                    data = response.json()
                    return {"success": True, **data}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in batch job response: {e}")
                    return {"success": False, "error": "Invalid response from server"}
            else:
                try:
                    error = response.json().get('detail', f"HTTP {response.status_code}")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}"
                return {"success": False, "error": error}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_batch_summary(self, batch_id: str) -> Dict[str, Any]:
        """Get summary of a batch of jobs.

        Args:
            batch_id: Batch ID to get summary for

        Returns:
            Dict with batch statistics
        """
        try:
            response = self._request('GET', f'/api/v1/jobs/batch/{batch_id}')

            if response.status_code == 200:
                try:
                    data = response.json()
                    return {"success": True, **data}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in batch summary response: {e}")
                    return {"success": False, "error": "Invalid response from server"}
            elif response.status_code == 404:
                return {"success": False, "error": "Batch not found"}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel all jobs in a batch.

        Args:
            batch_id: Batch ID to cancel

        Returns:
            Dict with cancellation result
        """
        try:
            response = self._request('POST', f'/api/v1/jobs/batch/{batch_id}/cancel')

            if response.status_code == 200:
                try:
                    data = response.json()
                    return {"success": True, **data}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in cancel batch response: {e}")
                    return {"success": False, "error": "Invalid response from server"}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_job_status(self, job_id: str) -> JobResponse:
        """Get job status.

        Args:
            job_id: Job ID to check

        Returns:
            JobResponse with current status
        """
        try:
            response = self._request('GET', f'/api/v1/jobs/{job_id}')

            if response.status_code == 200:
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in job status response: {e}")
                    return JobResponse(success=False, error="Invalid response from server")
                return JobResponse(
                    success=True,
                    job_id=job_id,
                    status=data.get('status'),
                    progress=data.get('progress', 0.0),
                    message=data.get('message'),
                    data=data
                )
            elif response.status_code == 404:
                return JobResponse(success=False, error="Job not found")
            else:
                return JobResponse(success=False, error=f"HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    def get_active_jobs(self) -> List[dict]:
        """Get list of active jobs.

        Returns:
            List of active job dictionaries
        """
        try:
            response = self._request('GET', '/api/v1/jobs/active')

            if response.status_code == 200:
                try:
                    data = response.json()
                    return data if isinstance(data, list) else []
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Invalid JSON in active jobs response")
                    return []
            return []

        except requests.exceptions.RequestException:
            return []

    def cancel_job(self, job_id: str) -> JobResponse:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            JobResponse indicating success
        """
        try:
            response = self._request('POST', f'/api/v1/jobs/{job_id}/cancel')

            if response.status_code == 200:
                return JobResponse(success=True, job_id=job_id, message="Job cancelled")
            else:
                return JobResponse(success=False, error=f"HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    def delete_job(self, job_id: str) -> JobResponse:
        """Delete a job record.

        Args:
            job_id: Job ID to delete

        Returns:
            JobResponse indicating success
        """
        try:
            response = self._request('DELETE', f'/api/v1/jobs/{job_id}')

            if response.status_code in (200, 204):
                return JobResponse(success=True, job_id=job_id, message="Job deleted")
            else:
                return JobResponse(success=False, error=f"HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    # Compound Management
    def list_compounds(
        self,
        page: int = 1,
        per_page: int = None
    ) -> CompoundListResponse:
        """List processed compounds.

        Args:
            page: Page number (1-indexed)
            per_page: Items per page

        Returns:
            CompoundListResponse with compound list
        """
        per_page = per_page or config.RESULTS_PER_PAGE

        try:
            response = self._request(
                'GET',
                '/api/v1/jobs',
                params={'page': page, 'per_page': per_page, 'status': 'completed'}
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in list compounds response: {e}")
                    return CompoundListResponse(success=False, error="Invalid response from server")
                return CompoundListResponse(
                    success=True,
                    compounds=data.get('items', []),
                    total=data.get('total', 0)
                )
            else:
                return CompoundListResponse(success=False, error=f"HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            return CompoundListResponse(success=False, error=str(e))

    def get_compounds_from_db(
        self,
        page: int = 1,
        per_page: int = 50,
        search: Optional[str] = None,
        include_duplicates: bool = False
    ) -> CompoundListResponse:
        """Get processed compounds from database (authoritative source).

        This uses the /compounds endpoint which returns proper compound names
        from the database, not parsed from blob names.

        Args:
            page: Page number (1-indexed)
            per_page: Items per page (max 100)
            search: Optional search term for compound name
            include_duplicates: Whether to include duplicate entries

        Returns:
            CompoundListResponse with compound list
        """
        try:
            params = {
                'page': page,
                'per_page': per_page,
                'include_duplicates': include_duplicates,
            }
            if search:
                params['search'] = search

            response = self._request(
                'GET',
                '/api/v1/compounds',
                params=params
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in compounds response: {e}")
                    return CompoundListResponse(success=False, error="Invalid response from server")
                return CompoundListResponse(
                    success=True,
                    compounds=data.get('items', []),
                    total=data.get('total', 0)
                )
            else:
                return CompoundListResponse(success=False, error=f"HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            return CompoundListResponse(success=False, error=str(e))

    def delete_compound(self, entry_id: str) -> JobResponse:
        """Delete a compound by entry_id.

        Deletes from database, Azure storage, and local cache.

        Args:
            entry_id: UUID of the compound to delete

        Returns:
            JobResponse indicating success/failure
        """
        try:
            response = self._request('DELETE', f'/api/v1/compounds/{entry_id}')

            if response.status_code in (200, 204):
                try:
                    data = response.json()
                    return JobResponse(
                        success=True,
                        message=data.get("message", "Compound deleted"),
                        data=data,
                    )
                except (json.JSONDecodeError, ValueError):
                    return JobResponse(success=True, message="Compound deleted")
            elif response.status_code == 404:
                return JobResponse(success=False, error="Compound not found")
            else:
                return JobResponse(success=False, error=f"HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    def poll_job_until_complete(
        self,
        job_id: str,
        callback=None,
        timeout: int = None
    ) -> JobResponse:
        """Poll job until completion or timeout.

        Args:
            job_id: Job ID to poll
            callback: Optional callback(progress, message) for updates
            timeout: Timeout in seconds (default from config)

        Returns:
            Final JobResponse
        """
        timeout = timeout or config.JOB_TIMEOUT_SECONDS
        start_time = time.time()
        poll_interval = config.JOB_POLL_INTERVAL_SECONDS

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return JobResponse(
                    success=False,
                    job_id=job_id,
                    error=f"Job timed out after {timeout} seconds"
                )

            response = self.get_job_status(job_id)

            if not response.success:
                return response

            if callback:
                callback(response.progress, response.message)

            if response.status == 'completed':
                return response
            elif response.status == 'failed':
                return JobResponse(
                    success=False,
                    job_id=job_id,
                    status='failed',
                    error=response.data.get('error', 'Job failed')
                )
            elif response.status == 'cancelled':
                return JobResponse(
                    success=False,
                    job_id=job_id,
                    status='cancelled',
                    error='Job was cancelled'
                )

            time.sleep(poll_interval)


# Singleton pattern with thread-safe initialization
_api_client: Optional[ImpulatorAPIClient] = None
_api_client_lock = threading.Lock()


def get_api_client() -> ImpulatorAPIClient:
    """Get singleton API client instance (thread-safe)."""
    global _api_client
    if _api_client is None:
        with _api_client_lock:
            if _api_client is None:
                _api_client = ImpulatorAPIClient()
    return _api_client


def set_session_id(session_id: str) -> None:
    """Set session ID on the singleton API client.

    Call this early in the app lifecycle with the user's session ID.

    Args:
        session_id: Unique session identifier
    """
    client = get_api_client()
    client.session_id = session_id
    logger.debug(f"API client session ID set: {session_id[:8]}...")
