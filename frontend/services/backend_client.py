"""
Backend API Client for IMPULATOR Frontend.

Provides type-safe access to backend API with retry logic and error handling.
"""

import json
import logging
import threading
import contextvars
from dataclasses import dataclass
from typing import Optional, Any

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
    compounds: list[dict] = None
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
        # Session ID stored per-thread via contextvars (Streamlit runs each user
        # session in a separate thread — a shared instance attribute would cause
        # one user's session_id to overwrite another's mid-request).
        if session_id:
            _session_id_var.set(session_id)

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
        """Get current session ID (thread-safe via contextvars)."""
        return _session_id_var.get(None)

    @session_id.setter
    def session_id(self, value: str):
        """Set session ID for user isolation (thread-safe via contextvars)."""
        _session_id_var.set(value)

    def _get_headers(self) -> dict[str, str]:
        """Get request headers including session ID if set."""
        headers = {}
        sid = _session_id_var.get(None)
        if sid:
            headers["X-Session-ID"] = sid
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
        except requests.exceptions.RequestException as e:
            logger.debug(f"Health check failed: {e}")
            return False

    # Job Management
    def submit_job(
        self,
        compound_name: str,
        smiles: str,
        similarity_threshold: int = None,
        activity_types: list[str] = None,
        author_name: str = "",
        duplicate_action: str = None,
    ) -> JobResponse:
        """Submit a compound analysis job.

        Args:
            compound_name: Name of the compound
            smiles: SMILES string
            similarity_threshold: Similarity threshold (default from config)
            activity_types: Activity types to search
            author_name: Name of the author submitting the analysis
            duplicate_action: Optional duplicate handling ('skip', 'replace', 'duplicate')

        Returns:
            JobResponse with job_id on success
        """
        payload = {
            "compound_name": compound_name,
            "author_name": author_name,
            "smiles": smiles,
            "similarity_threshold": similarity_threshold or config.DEFAULT_SIMILARITY_THRESHOLD,
            "activity_types": activity_types or list(config.DEFAULT_ACTIVITY_TYPES),
        }
        if duplicate_action:
            payload["duplicate_action"] = duplicate_action

        try:
            response = self._request('POST', '/api/v1/jobs', json=payload)

            if response.status_code in (200, 201):
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

    def check_duplicates(
        self,
        compound_names: list[str] = None,
        compounds: list[dict[str, Any]] = None,
        similarity_threshold: Optional[int] = None,
        activity_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Check which compounds already exist or are being processed.

        Supports two modes:
        1. Name-only (legacy): Just pass compound_names list
        2. Structure-based (recommended): Pass compounds list with SMILES/InChI
           This uses InChIKey for 100% accurate duplicate detection by structure.

        Args:
            compound_names: List of compound names to check (legacy mode)
            compounds: List of dicts with compound_name, smiles, and/or inchi (new mode)
                      Example: [{"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
            similarity_threshold: Optional submitted threshold for config-aware matching
            activity_types: Optional submitted activity types for config-aware matching

        Returns:
            Dict with:
            - 'existing': Compounds that already have results (by name)
            - 'processing': Compounds currently being processed
            - 'new': Compounds that are new
            - 'structure_matches': List of compounds that match by InChIKey (structure)
              Each match has: compound_name, inchikey, existing_compound_name,
              existing_entry_id, match_type ('exact' or 'structure_only')
        """
        if compounds:
            # New mode: structure-based checking with InChIKey
            payload = {
                "compounds": compounds,
                "similarity_threshold": similarity_threshold,
                "activity_types": activity_types,
            }
        elif compound_names:
            # Legacy mode: name-only checking
            payload = {"compound_names": compound_names}
        else:
            return {"success": False, "error": "Must provide compound_names or compounds"}

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

    def check_availability(
        self,
        smiles: str,
        similarity_threshold: int = 90,
        activity_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Check ChEMBL data availability for a single compound before submission.

        Args:
            smiles: SMILES string
            similarity_threshold: Requested similarity threshold
            activity_types: Activity types to search

        Returns:
            Dict with 'result' containing CompoundAvailability data, or 'error'
        """
        payload = {
            "smiles": smiles,
            "similarity_threshold": similarity_threshold,
        }
        if activity_types:
            payload["activity_types"] = activity_types

        try:
            response = self._request('POST', '/api/v1/jobs/check-availability', json=payload)

            if response.status_code == 200:
                try:
                    return {"success": True, **response.json()}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in availability response: {e}")
                    return {"success": False, "error": "Invalid response from server"}
            else:
                try:
                    error = response.json().get('detail', f"HTTP {response.status_code}")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}"
                return {"success": False, "error": error}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def check_availability_batch(
        self,
        compounds: list[dict[str, Any]],
        similarity_threshold: int = 90,
        activity_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Batch availability check for multiple compounds.

        Args:
            compounds: List of dicts with compound_name and smiles
            similarity_threshold: Requested similarity threshold
            activity_types: Activity types to search

        Returns:
            Dict with results, available_count, unavailable_count, no_data_count
        """
        payload = {
            "compounds": compounds,
            "similarity_threshold": similarity_threshold,
        }
        if activity_types:
            payload["activity_types"] = activity_types

        try:
            response = self._request(
                'POST', '/api/v1/jobs/check-availability/batch',
                json=payload,
                timeout=max(self.timeout, 60),  # Allow extra time for batch
            )

            if response.status_code == 200:
                try:
                    return {"success": True, **response.json()}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in batch availability response: {e}")
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
        activity_types: list[str] = None,
        author_name: Optional[str] = None,
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
            author_name: Name of the author submitting the analysis

        Returns:
            JobResponse with job_id on success (for replace/duplicate),
            or success=True with status='skipped' (for skip)
        """
        payload = {
            "action": action,
            "smiles": smiles,
            "compound_name": compound_name,
            "author_name": author_name,
            "existing_entry_id": existing_entry_id,
            "new_compound_name": new_compound_name,
            "similarity_threshold": similarity_threshold or config.DEFAULT_SIMILARITY_THRESHOLD,
            "activity_types": activity_types or list(config.DEFAULT_ACTIVITY_TYPES),
        }

        try:
            response = self._request('POST', '/api/v1/jobs/resolve-duplicate', json=payload)

            if response.status_code in (200, 201):
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
        compounds: list[dict[str, Any]],
        duplicate_decisions: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
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

            if response.status_code in (200, 201):
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in batch job response: {e}")
                    return {"success": False, "error": "Invalid response from server"}

                # Surface failed compounds if present
                failed = data.get("failed", [])
                if failed:
                    logger.warning(
                        f"Batch submission had {len(failed)} failures: "
                        f"{[f.get('compound_name', 'unknown') for f in failed[:5]]}"
                    )
                return {"success": True, **data}
            else:
                try:
                    error = response.json().get('detail', f"HTTP {response.status_code}")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}"
                return {"success": False, "error": error}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def get_batch_summary(self, batch_id: str) -> dict[str, Any]:
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

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
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
                    message=data.get('current_step') or data.get('message'),
                    data=data
                )
            elif response.status_code == 404:
                return JobResponse(success=False, error="Job not found")
            else:
                return JobResponse(success=False, error=f"HTTP {response.status_code}")

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    def get_active_jobs(self) -> Optional[list[dict]]:
        """Get list of active jobs.

        Returns:
            List of active job dictionaries, or None on error
        """
        try:
            response = self._request('GET', '/api/v1/jobs/active')

            if response.status_code == 200:
                try:
                    data = response.json()
                    return data if isinstance(data, list) else []
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Invalid JSON in active jobs response")
                    return None
            logger.debug(f"get_active_jobs returned HTTP {response.status_code}")
            return None

        except requests.exceptions.RequestException as e:
            logger.debug(f"get_active_jobs request failed: {e}")
            return None

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
                'page_size': per_page,
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

    def get_compound_versions(self, entry_id: str) -> dict:
        """Get all structural siblings (versions) of a compound.

        Args:
            entry_id: UUID of the compound entry

        Returns:
            Dict with success, versions list, and current_entry_id
        """
        try:
            response = self._request('GET', f'/api/v1/compounds/{entry_id}/versions')

            if response.status_code == 200:
                try:
                    data = response.json()
                    return {"success": True, **data}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Invalid JSON in versions response: {e}")
                    return {"success": False, "error": "Invalid response from server"}
            elif response.status_code == 404:
                return {"success": False, "error": "Compound not found"}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def delete_compound(self, entry_id: str) -> JobResponse:
        """Delete compound through the backend API.

        Deletion must remain backend-authoritative so database state, storage
        cleanup, and audit logging stay consistent.

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

        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"Backend unreachable for compound deletion: {e}")
            return JobResponse(
                success=False,
                error="Backend unavailable. Compound was not deleted.",
            )

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    def delete_compounds_batch(self, entry_ids: list[str]) -> JobResponse:
        """Delete multiple compounds through the backend API.

        Deletion must remain backend-authoritative so database state, storage
        cleanup, and audit logging stay consistent.

        Args:
            entry_ids: List of compound entry UUIDs to delete

        Returns:
            JobResponse with success/failure and deletion summary in data
        """
        try:
            response = self._request(
                'POST',
                '/api/v1/compounds/batch-delete',
                json={"entry_ids": entry_ids}
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except (json.JSONDecodeError, ValueError):
                    return JobResponse(success=True, message="Compounds deleted")

                total_failed = data.get("total_failed", 0)
                failed_list = data.get("failed", [])
                if total_failed > 0:
                    failed_names = [f.get("compound_name", f.get("entry_id", "unknown")) for f in failed_list]
                    return JobResponse(
                        success=True,
                        message=data.get("message", "Compounds deleted"),
                        data=data,
                        error=f"{total_failed} compound(s) failed to delete: {', '.join(failed_names[:5])}{'...' if len(failed_names) > 5 else ''}",
                    )
                return JobResponse(
                    success=True,
                    message=data.get("message", "Compounds deleted"),
                    data=data,
                )
            else:
                try:
                    error = response.json().get('detail', f"HTTP {response.status_code}")
                except (json.JSONDecodeError, ValueError):
                    error = f"HTTP {response.status_code}"
                return JobResponse(success=False, error=error)

        except (requests.ConnectionError, requests.Timeout) as e:
            logger.warning(f"Backend unreachable for batch compound deletion: {e}")
            return JobResponse(
                success=False,
                error="Backend unavailable. Compounds were not deleted.",
            )

        except requests.exceptions.RequestException as e:
            return JobResponse(success=False, error=str(e))

    # PubChem InChIKey Resolution

    @staticmethod
    def _extract_pubchem_smiles(props: dict) -> str:
        """Extract SMILES from a PubChem property dict.

        PubChem renamed response fields (old URL params still accepted):
          CanonicalSMILES  -> ConnectivitySMILES
          IsomericSMILES   -> SMILES
        """
        for key in ("ConnectivitySMILES", "SMILES", "CanonicalSMILES", "IsomericSMILES"):
            val = props.get(key, "")
            if val:
                return str(val)
        return ""

    def resolve_inchikey_to_smiles(self, inchikey: str) -> Optional[str]:
        """Resolve a single InChIKey to SMILES via PubChem.

        Args:
            inchikey: 27-character InChIKey string

        Returns:
            SMILES string, or None if not found / error
        """
        url = f"{config.PUBCHEM_BASE_URL}/compound/inchikey/{inchikey}/property/ConnectivitySMILES/JSON"
        try:
            resp = requests.get(url, timeout=config.PUBCHEM_TIMEOUT_SECONDS)
            if resp.status_code != 200:
                logger.warning(f"PubChem lookup failed for {inchikey}: HTTP {resp.status_code}")
                return None
            data = resp.json()
            props_list = data.get("PropertyTable", {}).get("Properties", [])
            if not props_list:
                return None
            smiles = self._extract_pubchem_smiles(props_list[0])
            return smiles or None
        except Exception as e:
            logger.error(f"PubChem resolution error for {inchikey}: {e}")
            return None

    def resolve_inchikeys_batch(self, inchikeys: list[str]) -> dict[str, str]:
        """Resolve multiple InChIKeys to SMILES via PubChem batch POST.

        Args:
            inchikeys: List of InChIKey strings

        Returns:
            Dict mapping InChIKey -> SMILES (only successful resolutions)
        """
        if not inchikeys:
            return {}

        url = f"{config.PUBCHEM_BASE_URL}/compound/inchikey/property/ConnectivitySMILES,InChIKey/JSON"
        results = {}

        # Process in chunks of PUBCHEM_BATCH_SIZE
        for i in range(0, len(inchikeys), config.PUBCHEM_BATCH_SIZE):
            chunk = inchikeys[i:i + config.PUBCHEM_BATCH_SIZE]
            try:
                resp = requests.post(
                    url,
                    data={"inchikey": ",".join(chunk)},
                    timeout=config.PUBCHEM_TIMEOUT_SECONDS,
                )
                if resp.status_code != 200:
                    logger.warning(f"PubChem batch lookup failed: HTTP {resp.status_code}")
                    continue
                data = resp.json()
                for props in data.get("PropertyTable", {}).get("Properties", []):
                    key = props.get("InChIKey", "")
                    smiles = self._extract_pubchem_smiles(props)
                    if key and smiles:
                        results[key] = smiles
            except Exception as e:
                logger.error(f"PubChem batch resolution error: {e}")

        return results


# Thread-safe session ID: each Streamlit user thread gets its own value
_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    '_session_id_var', default=None
)

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
    """Set session ID for the current thread (thread-safe via contextvars).

    Call this early in the app lifecycle with the user's session ID.
    Each Streamlit user thread gets its own isolated session ID.

    Args:
        session_id: Unique session identifier
    """
    _session_id_var.set(session_id)
    logger.debug(f"API client session ID set: {session_id[:8]}...")
