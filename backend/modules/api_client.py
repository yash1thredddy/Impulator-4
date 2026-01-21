"""
ChEMBL API client with optimized batch processing and caching.
Decoupled from Streamlit for backend use.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Callable, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
CACHE_SIZE = 2000
MAX_BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 0.5
RETRY_STATUS_CODES = [500, 502, 503, 504]
API_TIMEOUT = 30  # seconds for HTTP requests
CHEMBL_API_TIMEOUT = 60  # seconds for ChEMBL client operations (can be slower)
MAX_WORKERS = 4
ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]

# Rate limiting configuration
RATE_LIMIT_CALLS = 10  # Max calls per window
RATE_LIMIT_WINDOW = 1.0  # Window size in seconds


class RateLimiter:
    """
    Simple token bucket rate limiter for API calls.

    Limits requests to RATE_LIMIT_CALLS per RATE_LIMIT_WINDOW seconds.
    Thread-safe implementation using locks.
    """
    def __init__(self, calls_per_second: float = RATE_LIMIT_CALLS):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
        import threading
        self._lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                time.sleep(wait_time)
            self.last_call_time = time.time()


# Global rate limiters for different APIs
_chembl_rate_limiter = RateLimiter(calls_per_second=10)  # ChEMBL: ~10 req/sec
_classyfire_rate_limiter = RateLimiter(calls_per_second=2)  # ClassyFire: conservative


def cache_non_none(maxsize: int = CACHE_SIZE, ttl_seconds: int = 3600):
    """
    LRU cache that only caches successful (non-None) results with TTL support.

    This prevents caching of API failures, allowing retry on subsequent calls.
    Cached entries expire after ttl_seconds (default: 1 hour).

    Args:
        maxsize: Maximum number of entries to cache
        ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)
    """
    def decorator(func):
        cache = {}  # key -> (value, timestamp)
        cache_hits = [0]
        cache_misses = [0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            current_time = time.time()

            if key in cache:
                value, timestamp = cache[key]
                # Check if entry is still valid (not expired)
                if current_time - timestamp < ttl_seconds:
                    cache_hits[0] += 1
                    return value
                else:
                    # Entry expired, remove it
                    del cache[key]

            cache_misses[0] += 1
            result = func(*args, **kwargs)

            # Only cache non-None results
            if result is not None:
                # Evict expired entries first
                expired_keys = [
                    k for k, (_, ts) in cache.items()
                    if current_time - ts >= ttl_seconds
                ]
                for k in expired_keys:
                    del cache[k]

                # Evict oldest entry if still at capacity
                if len(cache) >= maxsize:
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]

                cache[key] = (result, current_time)

            return result

        def cache_clear():
            cache.clear()
            cache_hits[0] = 0
            cache_misses[0] = 0

        def cache_info():
            class CacheInfo:
                def __init__(self, hits, misses, maxsize, currsize):
                    self.hits = hits
                    self.misses = misses
                    self.maxsize = maxsize
                    self.currsize = currsize

                def _asdict(self):
                    return {
                        'hits': self.hits,
                        'misses': self.misses,
                        'maxsize': self.maxsize,
                        'currsize': self.currsize
                    }

            return CacheInfo(cache_hits[0], cache_misses[0], maxsize, len(cache))

        wrapper.cache_clear = cache_clear
        wrapper.cache_info = cache_info
        return wrapper

    return decorator

# Thread pool for timeout wrapper (reused across calls)
_timeout_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chembl_timeout")


def with_timeout(timeout_seconds: int = CHEMBL_API_TIMEOUT):
    """
    Decorator to add timeout to functions (especially ChEMBL library calls).

    The ChEMBL client library doesn't support native timeouts, so we use
    ThreadPoolExecutor to enforce a timeout.

    Args:
        timeout_seconds: Maximum seconds to wait before timing out
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            future = _timeout_executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                logger.error(f"Timeout ({timeout_seconds}s) exceeded for {func.__name__}")
                # Return appropriate default based on return type hints
                return None
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator

# Progress callback type
ProgressCallback = Callable[[float, str], None]

# Lazy import for ChEMBL client
_chembl_client = None


def _get_chembl_client():
    """Lazy initialization of ChEMBL client."""
    global _chembl_client
    if _chembl_client is None:
        try:
            from chembl_webresource_client.new_client import new_client
            _chembl_client = {
                'similarity': new_client.similarity,
                'molecule': new_client.molecule,
                'activity': new_client.activity,
                'target': new_client.target,
                'drug_indication': new_client.drug_indication,
            }
            logger.info(f"ChEMBL client initialized with endpoints: {list(_chembl_client.keys())}")
        except ImportError:
            logger.warning("chembl_webresource_client not installed")
            _chembl_client = {}
    return _chembl_client


# Configure retry strategy
retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=RETRY_BACKOFF_FACTOR,
    status_forcelist=RETRY_STATUS_CODES,
)


def get_session():
    """Create and return a requests session with retry configuration."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# Create session
session = get_session()


def _fetch_molecule_data_with_timeout(chembl_id: str) -> Optional[Dict]:
    """Internal function to fetch molecule data with timeout."""
    _chembl_rate_limiter.wait()  # Apply rate limiting
    client = _get_chembl_client()
    if 'molecule' not in client:
        return None
    return client['molecule'].get(chembl_id)


@cache_non_none(maxsize=CACHE_SIZE)
def get_molecule_data(chembl_id: str) -> Optional[Dict]:
    """
    Fetch molecule data from ChEMBL API with caching and timeout.

    Args:
        chembl_id: ChEMBL ID to fetch

    Returns:
        Optional[Dict]: Molecule data or None if error
    """
    try:
        # Use ThreadPoolExecutor for timeout since ChEMBL client doesn't support it
        future = _timeout_executor.submit(_fetch_molecule_data_with_timeout, chembl_id)
        return future.result(timeout=CHEMBL_API_TIMEOUT)
    except FuturesTimeoutError:
        logger.error(f"Timeout fetching molecule data for {chembl_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching molecule data for {chembl_id}: {str(e)}")
        return None


@cache_non_none(maxsize=CACHE_SIZE)
def get_classification(inchikey: str) -> Optional[Dict]:
    """
    Get classification data from ClassyFire API with caching.

    Args:
        inchikey: InChIKey for the molecule

    Returns:
        Optional[Dict]: Classification data or None if error
    """
    try:
        _classyfire_rate_limiter.wait()  # Apply rate limiting
        url = f'http://classyfire.wishartlab.com/entities/{inchikey}.json'
        response = session.get(url, timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Error getting classification for {inchikey}: {str(e)}")
        return None


def _fetch_target_name_with_timeout(target_chembl_id: str) -> Optional[str]:
    """Internal function to fetch target name with timeout."""
    _chembl_rate_limiter.wait()  # Apply rate limiting
    client = _get_chembl_client()
    if 'target' not in client:
        return None
    target_data = client['target'].get(target_chembl_id)
    if target_data:
        return target_data.get('pref_name', target_chembl_id)
    return None


@cache_non_none(maxsize=CACHE_SIZE)
def get_target_name(target_chembl_id: str) -> Optional[str]:
    """
    Fetch target name from ChEMBL API with caching and timeout.

    Args:
        target_chembl_id: ChEMBL Target ID

    Returns:
        Optional[str]: Target preferred name or None if error
    """
    if not target_chembl_id:
        return None

    try:
        future = _timeout_executor.submit(_fetch_target_name_with_timeout, target_chembl_id)
        return future.result(timeout=CHEMBL_API_TIMEOUT)
    except FuturesTimeoutError:
        logger.error(f"Timeout fetching target name for {target_chembl_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching target name for {target_chembl_id}: {str(e)}")
        return None


def _fetch_drug_indications_with_timeout(chembl_id: str, max_retries: int = 2) -> tuple:
    """Internal function to fetch drug indications with timeout and retry logic.

    Handles ChEMBL API intermittent failures (e.g., empty attribute errors during pagination).
    """
    client = _get_chembl_client()
    if 'drug_indication' not in client:
        logger.warning("drug_indication endpoint not available")
        return ()

    last_error = None
    for attempt in range(max_retries):
        try:
            indications = client['drug_indication'].filter(molecule_chembl_id=chembl_id)
            indication_list = list(indications)

            results = []
            for ind in indication_list:
                # Extract clinical trial URL from indication_refs
                clinical_trials_url = ''
                clinical_trials_ids = ''
                indication_refs = ind.get('indication_refs', [])

                if indication_refs:
                    for ref in indication_refs:
                        if ref.get('ref_type') == 'ClinicalTrials':
                            clinical_trials_url = ref.get('ref_url', '')
                            clinical_trials_ids = ref.get('ref_id', '')
                            break

                results.append({
                    'ChEMBL_ID': chembl_id,
                    'MESH_ID': ind.get('mesh_id', ''),
                    'MESH_Heading': ind.get('mesh_heading', ''),
                    'EFO_ID': ind.get('efo_id', ''),
                    'EFO_Term': ind.get('efo_term', ''),
                    'Max_Phase': ind.get('max_phase_for_ind', 0),
                    'Clinical_Trials_URL': clinical_trials_url,
                    'Clinical_Trials_IDs': clinical_trials_ids,
                })

            return tuple(results)

        except Exception as e:
            last_error = e
            error_str = str(e)
            # Check for ChEMBL data corruption (empty attribute errors during pagination)
            is_corruption_error = "empty attribute" in error_str or "doesn't allow a default" in error_str

            if attempt < max_retries - 1:
                if is_corruption_error:
                    logger.warning(f"Drug indications API data corruption for {chembl_id} (attempt {attempt + 1}), retrying...")
                else:
                    logger.warning(f"Drug indications fetch attempt {attempt + 1} failed for {chembl_id}: {e}")
                time.sleep(0.5 * (attempt + 1))
            else:
                logger.error(f"Drug indications fetch failed for {chembl_id} after {max_retries} attempts: {last_error}")

    return ()


@lru_cache(maxsize=CACHE_SIZE)
def get_drug_indications(chembl_id: str) -> tuple:
    """
    Fetch drug indications for a ChEMBL ID with caching and timeout.

    Returns indication data including MESH, EFO, and clinical trial references.

    Args:
        chembl_id: ChEMBL molecule ID

    Returns:
        tuple: Tuple of indication dictionaries (for caching compatibility)
    """
    if not chembl_id:
        return ()

    try:
        future = _timeout_executor.submit(_fetch_drug_indications_with_timeout, chembl_id)
        return future.result(timeout=CHEMBL_API_TIMEOUT)
    except FuturesTimeoutError:
        logger.error(f"Timeout fetching drug indications for {chembl_id}")
        return ()
    except Exception as e:
        logger.error(f"Error fetching drug indications for {chembl_id}: {str(e)}")
        return ()


def _similarity_search_with_timeout(smiles: str, similarity_threshold: int, max_retries: int = 2) -> List[Dict[str, str]]:
    """Internal function to perform similarity search with timeout and retry logic.

    Handles ChEMBL API intermittent failures (e.g., empty attribute errors during pagination).
    """
    client = _get_chembl_client()
    if 'similarity' not in client:
        logger.error("ChEMBL client not available for similarity search")
        return []

    last_error = None
    for attempt in range(max_retries):
        try:
            results = client['similarity'].filter(
                smiles=smiles,
                similarity=similarity_threshold
            ).only(['molecule_chembl_id'])

            # Convert to list to actually fetch the data
            result_list = list(results)
            return [{"ChEMBL ID": result['molecule_chembl_id']} for result in result_list]

        except Exception as e:
            last_error = e
            error_str = str(e)
            # Check for ChEMBL data corruption (empty attribute errors during pagination)
            is_corruption_error = "empty attribute" in error_str or "doesn't allow a default" in error_str

            if attempt < max_retries - 1:
                if is_corruption_error:
                    logger.warning(f"Similarity search API data corruption (attempt {attempt + 1}), retrying...")
                else:
                    logger.warning(f"Similarity search attempt {attempt + 1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))
            else:
                # Re-raise for the outer retry logic in get_chembl_ids
                raise last_error

    return []


# Longer timeout for similarity searches (can be slow)
SIMILARITY_SEARCH_TIMEOUT = 90  # seconds


def get_chembl_ids(smiles: str, similarity_threshold: int = 90, max_retries: int = 3) -> List[Dict[str, str]]:
    """
    Perform similarity search with error handling, retries, and timeout.

    Args:
        smiles: SMILES string to search
        similarity_threshold: Similarity threshold (0-100)
        max_retries: Maximum number of retry attempts

    Returns:
        List[Dict[str, str]]: List of ChEMBL IDs
    """
    for attempt in range(max_retries):
        try:
            # Use ThreadPoolExecutor for timeout
            future = _timeout_executor.submit(
                _similarity_search_with_timeout, smiles, similarity_threshold
            )
            return future.result(timeout=SIMILARITY_SEARCH_TIMEOUT)

        except FuturesTimeoutError:
            logger.warning(f"Similarity search timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            continue
        except IndexError as e:
            # Handle "tuple index out of range" from chembl client
            logger.warning(f"ChEMBL API IndexError (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            continue
        except Exception as e:
            logger.error(f"Error in similarity search (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            continue

    logger.error(f"Similarity search failed after {max_retries} attempts")
    return []


def _fetch_activity_batch(batch_params: Dict[str, Any], max_retries: int = 2) -> List[Dict]:
    """
    Helper function to fetch a batch of activities with retry logic.

    Args:
        batch_params: Dictionary containing batch parameters
        max_retries: Maximum retry attempts

    Returns:
        List[Dict]: List of activity data
    """
    chembl_ids = batch_params['chembl_ids']
    activity_type = batch_params['activity_type']

    for attempt in range(max_retries):
        try:
            client = _get_chembl_client()
            if 'activity' not in client:
                return []

            activities = client['activity'].filter(
                molecule_chembl_id__in=chembl_ids,
                standard_type=activity_type
            ).only('molecule_chembl_id', 'standard_value',
                  'standard_units', 'standard_type',
                  'target_chembl_id')

            return list(activities)

        except IndexError as e:
            # Handle "tuple index out of range" from chembl client
            logger.warning(f"Activity fetch IndexError for {chembl_ids[:2]} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            continue
        except Exception as e:
            logger.error(f"Error fetching activities for batch {chembl_ids[:2]} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            continue

    return []


def batch_fetch_activities(
    chembl_ids: List[str],
    activity_types: List[str] = None,
    batch_size: int = MAX_BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    progress_callback: Optional[ProgressCallback] = None
) -> List[Dict]:
    """
    Fetch activities in parallel batches with optimized performance.

    Args:
        chembl_ids: List of ChEMBL IDs
        activity_types: List of activity types to fetch
        batch_size: Size of each batch
        max_workers: Maximum number of concurrent workers
        progress_callback: Optional callback for progress updates (progress: 0-1, message: str)

    Returns:
        List[Dict]: List of activity data
    """
    if activity_types is None:
        activity_types = ACTIVITY_TYPES

    if not chembl_ids:
        return []

    if batch_size > MAX_BATCH_SIZE:
        logger.warning(f"Batch size {batch_size} exceeds maximum {MAX_BATCH_SIZE}. Using maximum value.")
        batch_size = MAX_BATCH_SIZE

    all_activities = []

    # Create batches for parallel processing
    batches = []
    for i in range(0, len(chembl_ids), batch_size):
        batch = chembl_ids[i:i + batch_size]
        for activity_type in activity_types:
            batches.append({
                'chembl_ids': batch,
                'activity_type': activity_type
            })

    total_batches = len(batches)

    if progress_callback:
        progress_callback(0.0, f"Fetching activity data for {len(chembl_ids)} compounds across {len(activity_types)} activity types...")

    # Process batches in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_activity_batch, batch): i for i, batch in enumerate(batches)}

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results = future.result()
                all_activities.extend(batch_results)

                # Update progress
                completed += 1
                progress = completed / total_batches
                if progress_callback:
                    progress_callback(progress, f"Processed {completed}/{total_batches} batches ({int(progress * 100)}%)")
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")

    if progress_callback:
        progress_callback(1.0, f"Completed! Fetched {len(all_activities)} activity data points.")

    return all_activities


def fetch_batch_molecule_data(
    chembl_ids: List[str],
    progress_callback: Optional[ProgressCallback] = None,
    max_retries: int = 2
) -> Dict[str, Dict]:
    """
    Fetch molecule data for multiple ChEMBL IDs in a single batch query.

    This is the OPTIMIZED approach - single query instead of N individual calls.
    Provides ~3-5x speedup over individual get_molecule_data() calls.

    Includes retry logic for ChEMBL API intermittent failures.

    Args:
        chembl_ids: List of ChEMBL IDs to fetch
        progress_callback: Optional callback for progress updates
        max_retries: Max retries for batch fetch before fallback

    Returns:
        Dict mapping ChEMBL ID -> molecule data dict
    """
    if not chembl_ids:
        return {}

    if progress_callback:
        progress_callback(0.1, f"Fetching molecule data for {len(chembl_ids)} compounds...")

    client = _get_chembl_client()
    if 'molecule' not in client:
        logger.error("ChEMBL molecule client not available")
        return {}

    # Try batch fetch with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            if progress_callback:
                progress_callback(0.2, f"Querying ChEMBL (batch, attempt {attempt + 1})...")

            # Single batch query for all molecules
            molecules = client['molecule'].filter(
                molecule_chembl_id__in=chembl_ids
            ).only([
                'molecule_chembl_id',
                'pref_name',
                'molecule_properties',
                'molecule_structures'
            ])

            if progress_callback:
                progress_callback(0.6, "Processing molecule data...")

            # Convert to dict keyed by ChEMBL ID
            result = {}
            for mol in list(molecules):
                chembl_id = mol.get('molecule_chembl_id')
                if chembl_id:
                    result[chembl_id] = mol

            if progress_callback:
                progress_callback(1.0, f"Fetched {len(result)}/{len(chembl_ids)} molecules")

            logger.info(f"Batch molecule fetch: {len(result)}/{len(chembl_ids)} molecules retrieved")
            return result

        except Exception as e:
            last_error = e
            error_str = str(e)
            is_corruption_error = "empty attribute" in error_str or "doesn't allow a default" in error_str

            if attempt < max_retries - 1:
                if is_corruption_error:
                    logger.warning(f"Batch molecule fetch API data corruption (attempt {attempt + 1}), retrying...")
                else:
                    logger.warning(f"Batch molecule fetch attempt {attempt + 1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))
            else:
                logger.error(f"Batch molecule fetch failed after {max_retries} attempts ({type(e).__name__}): {e}")

    # Fallback to individual fetches with timeout protection
    logger.info("Falling back to individual molecule fetches...")
    result = {}
    failed_count = 0
    for i, chembl_id in enumerate(chembl_ids):
        try:
            # Use timeout executor for individual fetches in fallback
            future = _timeout_executor.submit(get_molecule_data, chembl_id)
            mol_data = future.result(timeout=CHEMBL_API_TIMEOUT)
            if mol_data:
                result[chembl_id] = mol_data
        except FuturesTimeoutError:
            logger.debug(f"Timeout fetching molecule {chembl_id} in fallback")
            failed_count += 1
        except Exception as e:
            logger.debug(f"Failed to fetch molecule {chembl_id}: {e}")
            failed_count += 1

        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(0.1 + 0.9 * (i + 1) / len(chembl_ids),
                            f"Fetched {i + 1}/{len(chembl_ids)} molecules (fallback)...")

    if failed_count > 0:
        logger.warning(f"Fallback molecule fetch: {failed_count}/{len(chembl_ids)} failed")
    return result


def fetch_batch_target_names(
    target_chembl_ids: List[str],
    progress_callback: Optional[ProgressCallback] = None,
    max_retries: int = 2
) -> Dict[str, str]:
    """
    Fetch target names for multiple ChEMBL Target IDs in a single batch query.

    This is the OPTIMIZED approach - single query instead of N individual calls.
    Provides ~3-5x speedup over individual get_target_name() calls.

    Includes retry logic for ChEMBL API intermittent failures.

    Args:
        target_chembl_ids: List of ChEMBL Target IDs to fetch
        progress_callback: Optional callback for progress updates
        max_retries: Max retries for batch fetch before fallback

    Returns:
        Dict mapping Target ChEMBL ID -> target preferred name
    """
    if not target_chembl_ids:
        return {}

    # Remove duplicates while preserving order
    unique_ids = list(dict.fromkeys(target_chembl_ids))

    if progress_callback:
        progress_callback(0.1, f"Fetching target names for {len(unique_ids)} targets...")

    client = _get_chembl_client()
    if 'target' not in client:
        logger.error("ChEMBL target client not available")
        return {}

    # Try batch fetch with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            if progress_callback:
                progress_callback(0.2, f"Querying ChEMBL targets (batch, attempt {attempt + 1})...")

            # Single batch query for all targets
            targets = client['target'].filter(
                target_chembl_id__in=unique_ids
            ).only([
                'target_chembl_id',
                'pref_name'
            ])

            if progress_callback:
                progress_callback(0.6, "Processing target data...")

            # Convert to dict keyed by Target ChEMBL ID
            result = {}
            for target in list(targets):
                target_id = target.get('target_chembl_id')
                if target_id:
                    result[target_id] = target.get('pref_name', '') or ''

            if progress_callback:
                progress_callback(1.0, f"Fetched {len(result)}/{len(unique_ids)} target names")

            logger.info(f"Batch target fetch: {len(result)}/{len(unique_ids)} targets retrieved")
            return result

        except Exception as e:
            last_error = e
            error_str = str(e)
            is_corruption_error = "empty attribute" in error_str or "doesn't allow a default" in error_str

            if attempt < max_retries - 1:
                if is_corruption_error:
                    logger.warning(f"Batch target fetch API data corruption (attempt {attempt + 1}), retrying...")
                else:
                    logger.warning(f"Batch target fetch attempt {attempt + 1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))
            else:
                logger.error(f"Batch target fetch failed after {max_retries} attempts ({type(e).__name__}): {e}")

    # Fallback to individual fetches with timeout protection
    logger.info("Falling back to individual target fetches...")
    result = {}
    failed_count = 0
    for i, target_id in enumerate(unique_ids):
        try:
            # Use timeout executor for individual fetches in fallback
            future = _timeout_executor.submit(get_target_name, target_id)
            name = future.result(timeout=CHEMBL_API_TIMEOUT)
            if name:
                result[target_id] = name
        except FuturesTimeoutError:
            logger.debug(f"Timeout fetching target {target_id} in fallback")
            failed_count += 1
        except Exception as e:
            logger.debug(f"Failed to fetch target {target_id}: {e}")
            failed_count += 1

        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(0.1 + 0.9 * (i + 1) / len(unique_ids),
                            f"Fetched {i + 1}/{len(unique_ids)} targets (fallback)...")

    if failed_count > 0:
        logger.warning(f"Fallback target fetch: {failed_count}/{len(unique_ids)} failed")
    return result


def fetch_all_activities_single_batch(
    chembl_ids: List[str],
    activity_types: List[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    max_retries: int = 2
) -> List[Dict]:
    """
    Fetch ALL activities for multiple ChEMBL IDs in a single query.

    This is the OPTIMIZED approach (validated in test_quercetin_verification.py):
    - 1 server query instead of N queries
    - Auto-pagination handles large results
    - Local filtering is instant

    Falls back to chunked fetching if single batch fails (e.g., for large ID lists
    or when ChEMBL API returns corrupted records during pagination).

    Args:
        chembl_ids: List of ChEMBL IDs to fetch
        activity_types: Activity types to filter (done locally after fetch)
        progress_callback: Optional callback for progress updates
        max_retries: Number of retries for single batch before chunked fallback

    Returns:
        List of activity dictionaries filtered to specified types
    """
    if not chembl_ids:
        return []

    if activity_types is None:
        activity_types = ['IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'GI50', 'MIC']

    activity_types_set = set(activity_types)

    if progress_callback:
        progress_callback(0.1, f"Fetching activities for {len(chembl_ids)} compounds...")

    client = _get_chembl_client()
    if 'activity' not in client:
        logger.error("ChEMBL client not available")
        return []

    # Try single batch with retries (ChEMBL API can have intermittent issues with corrupted records)
    last_error = None
    for attempt in range(max_retries):
        try:
            # Single query for ALL activities - auto-paginates
            activities = client['activity'].filter(
                molecule_chembl_id__in=chembl_ids
            ).only([
                'molecule_chembl_id',
                'standard_type',
                'standard_value',
                'standard_units',
                'target_chembl_id'
            ])

            if progress_callback:
                progress_callback(0.3, f"Fetching from ChEMBL (attempt {attempt + 1})...")

            # Convert to list (triggers pagination)
            all_raw = list(activities)

            if progress_callback:
                progress_callback(0.7, f"Filtering {len(all_raw)} activities locally...")

            # Filter locally (instant)
            filtered = [
                a for a in all_raw
                if a.get('standard_type') in activity_types_set
            ]

            if progress_callback:
                progress_callback(1.0, f"Found {len(filtered)} activities")

            logger.info(f"Single batch fetch: {len(all_raw)} raw -> {len(filtered)} filtered")
            return filtered

        except Exception as e:
            last_error = e
            error_str = str(e)
            # Check if this is a ChEMBL data corruption error (empty attribute)
            is_corruption_error = "empty attribute" in error_str or "doesn't allow a default" in error_str

            if attempt < max_retries - 1:
                if is_corruption_error:
                    logger.warning(f"ChEMBL API data corruption on attempt {attempt + 1}, retrying...")
                else:
                    logger.warning(f"Single batch attempt {attempt + 1} failed ({type(e).__name__}): {e}")
                time.sleep(1 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Single batch activity fetch failed after {max_retries} attempts ({type(e).__name__}): {e}")

    # All retries failed, fall back to chunked fetching
    logger.info("Falling back to chunked activity fetching...")

    # Fallback: fetch in smaller chunks (more resilient to bad records)
    CHUNK_SIZE = 5  # Smaller chunks for better error isolation
    all_filtered = []
    total_chunks = (len(chembl_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    failed_chunks = 0

    for i in range(0, len(chembl_ids), CHUNK_SIZE):
        chunk = chembl_ids[i:i + CHUNK_SIZE]
        chunk_num = i // CHUNK_SIZE + 1

        if progress_callback:
            progress_callback(0.1 + 0.8 * (chunk_num / total_chunks),
                            f"Fetching chunk {chunk_num}/{total_chunks}...")

        chunk_success = False
        for chunk_attempt in range(2):  # 2 attempts per chunk
            try:
                chunk_activities = client['activity'].filter(
                    molecule_chembl_id__in=chunk
                ).only([
                    'molecule_chembl_id',
                    'standard_type',
                    'standard_value',
                    'standard_units',
                    'target_chembl_id'
                ])

                chunk_raw = list(chunk_activities)
                chunk_filtered = [
                    a for a in chunk_raw
                    if a.get('standard_type') in activity_types_set
                ]
                all_filtered.extend(chunk_filtered)
                logger.debug(f"Chunk {chunk_num}: {len(chunk_raw)} raw -> {len(chunk_filtered)} filtered")
                chunk_success = True
                break

            except Exception as chunk_error:
                if chunk_attempt == 0:
                    logger.debug(f"Chunk {chunk_num} attempt 1 failed, retrying: {chunk_error}")
                    time.sleep(0.5)
                else:
                    logger.warning(f"Chunk {chunk_num} failed after 2 attempts: {chunk_error}")

        if not chunk_success:
            failed_chunks += 1
            # Try individual IDs in the failed chunk as last resort
            for chembl_id in chunk:
                try:
                    single_activities = client['activity'].filter(
                        molecule_chembl_id=chembl_id
                    ).only([
                        'molecule_chembl_id',
                        'standard_type',
                        'standard_value',
                        'standard_units',
                        'target_chembl_id'
                    ])
                    single_raw = list(single_activities)
                    single_filtered = [
                        a for a in single_raw
                        if a.get('standard_type') in activity_types_set
                    ]
                    all_filtered.extend(single_filtered)
                    logger.debug(f"Individual fetch for {chembl_id}: {len(single_filtered)} activities")
                except Exception as ind_error:
                    logger.debug(f"Individual fetch for {chembl_id} failed: {ind_error}")

    if progress_callback:
        progress_callback(1.0, f"Found {len(all_filtered)} activities (chunked, {failed_chunks} chunks needed fallback)")

    logger.info(f"Chunked fetch complete: {len(all_filtered)} activities from {len(chembl_ids)} compounds ({failed_chunks} chunks failed)")
    return all_filtered


def fetch_compound_activities(
    chembl_id: str,
    activity_types: List[str] = None,
    max_retries_per_type: int = 2
) -> List[Dict]:
    """
    Fetch activities for a single compound with retry logic.

    Handles ChEMBL API intermittent failures (e.g., empty attribute errors during pagination).

    Args:
        chembl_id: ChEMBL ID to fetch
        activity_types: List of activity types to fetch
        max_retries_per_type: Max retries per activity type

    Returns:
        List[Dict]: List of activity data
    """
    if activity_types is None:
        activity_types = ACTIVITY_TYPES

    all_activities = []
    client = _get_chembl_client()

    if 'activity' not in client:
        logger.error("ChEMBL client not available")
        return []

    for activity_type in activity_types:
        for attempt in range(max_retries_per_type):
            try:
                activities = client['activity'].filter(
                    molecule_chembl_id=chembl_id,
                    standard_type=activity_type
                ).only('standard_value', 'standard_units', 'standard_type',
                       'target_chembl_id', 'target_pref_name')

                activity_list = list(activities)
                all_activities.extend(activity_list)
                break  # Success, move to next activity type

            except Exception as e:
                error_str = str(e)
                # Check for ChEMBL data corruption (empty attribute errors during pagination)
                is_corruption_error = "empty attribute" in error_str or "doesn't allow a default" in error_str

                if attempt < max_retries_per_type - 1:
                    if is_corruption_error:
                        logger.warning(f"Activity fetch for {activity_type} API data corruption (attempt {attempt + 1}), retrying...")
                    else:
                        logger.warning(f"Activity fetch for {activity_type} attempt {attempt + 1} failed: {e}")
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"Error fetching {activity_type} for {chembl_id} after {max_retries_per_type} attempts: {e}")

    return all_activities


# Cache clearing utilities
def clear_caches():
    """Clear all LRU caches."""
    get_molecule_data.cache_clear()
    get_classification.cache_clear()
    get_target_name.cache_clear()
    get_drug_indications.cache_clear()
    logger.info("All API client caches cleared")


def get_cache_info() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        'molecule_data': get_molecule_data.cache_info()._asdict(),
        'classification': get_classification.cache_info()._asdict(),
        'target_name': get_target_name.cache_info()._asdict(),
        'drug_indications': get_drug_indications.cache_info()._asdict(),
    }


def shutdown_api_client():
    """
    Shutdown the API client and cleanup resources.

    Call this during application shutdown to properly cleanup
    the timeout executor thread pool.
    """
    global _timeout_executor
    try:
        _timeout_executor.shutdown(wait=False, cancel_futures=True)
        logger.info("API client timeout executor shutdown complete")
    except Exception as e:
        logger.warning(f"Error during API client shutdown: {e}")


# Register shutdown handler
import atexit
atexit.register(shutdown_api_client)
