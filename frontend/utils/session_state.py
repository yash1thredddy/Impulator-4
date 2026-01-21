"""Session state management for IMPULATOR frontend.

This module provides centralized session state management for Streamlit,
with features like:
- Default value initialization
- Type-safe access
- View management
- Job state tracking
- User session isolation (via session_id)
"""

import copy
import hashlib
import logging
import uuid
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _default_factory(value: Any) -> Callable[[], Any]:
    """Create a factory function that returns a deep copy of the value.

    This prevents mutable default values from being shared across sessions.
    Uses deepcopy to handle nested mutable structures safely.
    """
    if isinstance(value, (list, dict, set)):
        return lambda: copy.deepcopy(value)
    return lambda: value


# View constants
VIEW_HOME = "home"
VIEW_ANALYZE = "analyze"
VIEW_COMPOUND_DETAILS = "compound_details"


@dataclass
class ActiveJob:
    """Represents an active job being tracked."""
    job_id: str
    compound_name: str
    status: str
    progress: float = 0.0
    message: Optional[str] = None


class SessionState:
    """Centralized session state management for IMPULATOR.

    This class provides a clean interface for managing Streamlit session state.
    It handles initialization, access, and cleanup of session state values.

    Example:
        >>> from frontend.utils import SessionState
        >>> SessionState.init_defaults()
        >>> SessionState.set('my_key', 'my_value')
        >>> value = SessionState.get('my_key')
    """

    # Default value factories for session state keys
    # Using factories prevents mutable defaults from being shared across sessions
    _DEFAULT_FACTORIES: Dict[str, Callable[[], Any]] = {
        # Session isolation - unique ID per browser session
        'session_id': lambda: str(uuid.uuid4()),

        # Navigation state
        'current_view': lambda: VIEW_HOME,
        'selected_compound': lambda: None,
        'compound_search_query': lambda: "",

        # Job tracking
        'active_jobs': lambda: {},  # Dict[job_id, ActiveJob]
        'active_batch_ids': lambda: set(),  # Set of active batch IDs
        'last_processed_compound': lambda: None,
        'show_view_results': lambda: False,

        # Processing state
        'is_processing': lambda: False,
        'processing_compound': lambda: None,

        # Form state
        'pending_compound_name': lambda: None,
        'pending_structure_input': lambda: None,
        'pending_input_format': lambda: 'smiles',
        'pending_similarity_threshold': lambda: 90,
        'pending_activity_types': lambda: None,

        # Preferences
        'selected_activity_types': lambda: ["IC50", "Ki", "Kd", "EC50"],
        'last_similarity_threshold': lambda: 90,
        'compound_details_tab': lambda: "summary",
        'molecule_viewer_tab': lambda: "3D",

        # UI state
        'show_delete_confirmation': lambda: False,
        'deletion_success': lambda: False,
        'show_batch_complete': lambda: False,
        'batch_success_count': lambda: 0,
        'batch_fail_count': lambda: 0,

        # Error state
        'last_error': lambda: None,
    }

    # Legacy DEFAULTS for backwards compatibility (read-only reference)
    DEFAULTS: Dict[str, Any] = {
        'current_view': VIEW_HOME,
        'selected_compound': None,
        'compound_search_query': "",
        'active_jobs': {},
        'is_processing': False,
        'selected_activity_types': ["IC50", "Ki", "Kd", "EC50"],
        'last_similarity_threshold': 90,
    }

    # Keys associated with each mode/view
    MODE_KEYS: Dict[str, List[str]] = {
        'analyze': [
            'pending_compound_name',
            'pending_structure_input',
            'pending_input_format',
            'pending_similarity_threshold',
            'pending_activity_types',
        ],
        'compound_details': [
            'selected_compound',
            'compound_details_tab',
            'molecule_viewer_tab',
            'show_delete_confirmation',
        ],
        'processing': [
            'is_processing',
            'processing_compound',
            'show_view_results',
            'last_processed_compound',
        ],
    }

    @classmethod
    def _get_session_state(cls):
        """Get Streamlit session state (lazy import for testing)."""
        try:
            import streamlit as st
            return st.session_state
        except ImportError:
            # Fallback for testing without Streamlit
            if not hasattr(cls, '_mock_state'):
                cls._mock_state = {}
            return cls._mock_state

    @classmethod
    def init_defaults(cls) -> None:
        """Initialize default session state values.

        Call this at the start of your Streamlit app to ensure
        all expected keys exist with sensible defaults.
        """
        session_state = cls._get_session_state()

        for key, factory in cls._DEFAULT_FACTORIES.items():
            if key not in session_state:
                session_state[key] = factory()
                logger.debug(f"Initialized session state key: {key}")

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a value from session state."""
        session_state = cls._get_session_state()
        return session_state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a value in session state."""
        session_state = cls._get_session_state()
        session_state[key] = value
        logger.debug(f"Set session state: {key} = {type(value).__name__}")

    @classmethod
    def clear(cls, key: str) -> None:
        """Clear a session state key."""
        session_state = cls._get_session_state()
        if key in session_state:
            del session_state[key]
            logger.debug(f"Cleared session state key: {key}")

    @classmethod
    def clear_mode(cls, mode: str) -> None:
        """Clear all state associated with a specific mode."""
        keys = cls.MODE_KEYS.get(mode, [])
        for key in keys:
            cls.clear(key)
        logger.debug(f"Cleared session state for mode: {mode}")

    @classmethod
    def has(cls, key: str) -> bool:
        """Check if a key exists in session state."""
        session_state = cls._get_session_state()
        return key in session_state

    @classmethod
    def get_or_set(cls, key: str, default: Any) -> Any:
        """Get value if exists, otherwise set and return default."""
        if not cls.has(key):
            cls.set(key, default)
        return cls.get(key)

    # View management helpers
    @classmethod
    def get_current_view(cls) -> str:
        """Get the current view."""
        return cls.get('current_view', VIEW_HOME)

    @classmethod
    def set_view(cls, view: str) -> None:
        """Set the current view."""
        cls.set('current_view', view)

    @classmethod
    def navigate_to_home(cls) -> None:
        """Navigate to home view."""
        cls.set_view(VIEW_HOME)
        cls.set('selected_compound', None)

    @classmethod
    def navigate_to_analyze(cls) -> None:
        """Navigate to analyze view."""
        cls.set_view(VIEW_ANALYZE)

    @classmethod
    def navigate_to_compound(cls, compound_name: str, entry_id: str = None, storage_path: str = None) -> None:
        """Navigate to compound details view.

        Args:
            compound_name: Display name of the compound
            entry_id: UUID entry_id for storage lookup (optional, for new storage format)
            storage_path: Full Azure storage path from database (most reliable for fetching results)
        """
        cls.set('selected_compound', compound_name)
        cls.set('selected_compound_entry_id', entry_id)
        cls.set('selected_compound_storage_path', storage_path)
        cls.set_view(VIEW_COMPOUND_DETAILS)

    # Job management helpers
    @classmethod
    def add_active_job(cls, job_id: str, compound_name: str, status: str = "pending") -> None:
        """Add a job to active jobs tracking."""
        jobs = cls.get('active_jobs', {})
        jobs[job_id] = ActiveJob(
            job_id=job_id,
            compound_name=compound_name,
            status=status
        )
        cls.set('active_jobs', jobs)

    @classmethod
    def update_job(cls, job_id: str, status: str, progress: float, message: str = None) -> None:
        """Update an active job's status."""
        jobs = cls.get('active_jobs', {})
        if job_id in jobs:
            job = jobs[job_id]
            job.status = status
            job.progress = progress
            job.message = message
            cls.set('active_jobs', jobs)

    @classmethod
    def remove_job(cls, job_id: str) -> None:
        """Remove a job from active tracking."""
        jobs = cls.get('active_jobs', {})
        if job_id in jobs:
            del jobs[job_id]
            cls.set('active_jobs', jobs)

    @classmethod
    def get_active_jobs(cls) -> Dict[str, ActiveJob]:
        """Get all active jobs."""
        return cls.get('active_jobs', {})

    # Processing state helpers
    @classmethod
    def reset_processing_state(cls) -> None:
        """Reset all processing-related session state variables."""
        cls.set('is_processing', False)
        cls.set('processing_compound', None)
        cls.set('show_view_results', False)
        cls.clear_mode('analyze')

    @classmethod
    def is_processing(cls) -> bool:
        """Check if processing is in progress."""
        return cls.get('is_processing', False)

    @classmethod
    def start_processing(cls, compound_name: str) -> None:
        """Mark processing as started."""
        cls.set('is_processing', True)
        cls.set('processing_compound', compound_name)

    @classmethod
    def finish_processing(cls, compound_name: str, success: bool = True) -> None:
        """Mark processing as finished."""
        cls.set('is_processing', False)
        cls.set('processing_compound', None)
        if success:
            cls.set('last_processed_compound', compound_name)
            cls.set('show_view_results', True)

    # File change detection
    @classmethod
    def file_changed(cls, file) -> bool:
        """Check if an uploaded file has changed."""
        if file is None:
            return False

        try:
            file_content = file.getvalue()
            current_hash = hashlib.sha256(file_content).hexdigest()
            stored_hash = cls.get('uploaded_file_hash')

            if current_hash != stored_hash:
                cls.set('uploaded_file_hash', current_hash)
                return True

            return False

        except Exception as e:
            logger.warning(f"Error checking file change: {e}")
            return True

    # Session ID helpers
    @classmethod
    def get_session_id(cls) -> str:
        """Get the unique session ID for this browser session.

        This ID is used to isolate jobs between different users.
        It's generated once per browser session and persists across page refreshes.

        Returns:
            Unique session ID string (UUID format)
        """
        session_id = cls.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            cls.set('session_id', session_id)
            logger.info(f"Generated new session ID: {session_id[:8]}...")
        return session_id

    # Batch tracking helpers
    @classmethod
    def add_batch(cls, batch_id: str) -> None:
        """Track an active batch of jobs."""
        batches = cls.get('active_batch_ids', set())
        batches.add(batch_id)
        cls.set('active_batch_ids', batches)

    @classmethod
    def remove_batch(cls, batch_id: str) -> None:
        """Remove a batch from active tracking."""
        batches = cls.get('active_batch_ids', set())
        batches.discard(batch_id)
        cls.set('active_batch_ids', batches)

    @classmethod
    def get_active_batches(cls) -> set:
        """Get set of active batch IDs."""
        return cls.get('active_batch_ids', set())
