"""Input validation utilities.

This module provides validation for various types of user input:
- SMILES strings
- InChI strings
- InChI Keys
- Uploaded files
- DataFrames

All validators return ValidationResult objects for consistent error handling.
"""

import re
import html
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

import pandas as pd

# Import config - use try/except for standalone testing
try:
    from frontend.config.settings import config
except ImportError:
    # Fallback for standalone testing
    class _MockConfig:
        MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
        MAX_FILE_SIZE_MB = 50
        ALLOWED_EXTENSIONS = frozenset({'.csv', '.xlsx'})
        MAX_ROWS_LIMIT = 100_000
        MAX_ROWS_WARNING = 10_000
        MAX_SMILES_LENGTH = 10_000
        MAX_CATEGORICAL_CARDINALITY = 50
    config = _MockConfig()

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid: Whether the validation passed
        errors: List of error messages (empty if valid)
        warnings: List of warning messages (non-fatal issues)

    Example:
        >>> result = ValidationResult(is_valid=True)
        >>> if result.is_valid:
        ...     process_data()
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid

    def add_error(self, error: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(warning)


class InputValidator:
    """Validates chemical structure inputs.

    Provides validation for SMILES, InChI, and InChI Key formats.
    Also provides format detection and HTML sanitization.
    """

    # SMILES allowed characters pattern
    # Includes: atoms, bonds, rings, branches, stereochemistry, charges
    SMILES_PATTERN = re.compile(
        r'^[A-Za-z0-9@+\-\[\]()\\/#=%.:\*\$]+$'
    )

    # InChI pattern - starts with InChI= followed by version and layers
    INCHI_PATTERN = re.compile(
        r'^InChI=1S?/[A-Za-z0-9/+\-(),.;?*]+$'
    )

    # InChI Key pattern - exactly 27 characters in specific format
    # Format: XXXXXXXXXXXXXX-YYYYYYYYYY-Z
    INCHI_KEY_PATTERN = re.compile(
        r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$'
    )

    @classmethod
    def validate_smiles(cls, smiles: str) -> ValidationResult:
        """Validate a SMILES string.

        Performs the following checks:
        1. Non-empty string
        2. Maximum length check (DoS prevention)
        3. Character pattern validation

        Note: This does NOT validate chemical correctness. Use RDKit
        for full SMILES parsing validation.

        Args:
            smiles: SMILES string to validate

        Returns:
            ValidationResult with is_valid flag and any errors

        Example:
            >>> result = InputValidator.validate_smiles('CCO')
            >>> result.is_valid
            True
            >>> result = InputValidator.validate_smiles('')
            >>> result.is_valid
            False
        """
        result = ValidationResult(is_valid=True)

        # Check for empty/None
        if not smiles or not isinstance(smiles, str):
            result.add_error("SMILES cannot be empty")
            return result

        # Strip whitespace
        smiles = smiles.strip()

        if not smiles:
            result.add_error("SMILES cannot be empty")
            return result

        # Check length (DoS prevention)
        if len(smiles) > config.MAX_SMILES_LENGTH:
            result.add_error(
                f"SMILES too long (max {config.MAX_SMILES_LENGTH} characters)"
            )
            return result

        # Check character pattern
        if not cls.SMILES_PATTERN.match(smiles):
            result.add_error("SMILES contains invalid characters")
            return result

        return result

    @classmethod
    def validate_inchi(cls, inchi: str) -> ValidationResult:
        """Validate an InChI string.

        Args:
            inchi: InChI string to validate

        Returns:
            ValidationResult with is_valid flag and any errors
        """
        result = ValidationResult(is_valid=True)

        if not inchi or not isinstance(inchi, str):
            result.add_error("InChI cannot be empty")
            return result

        inchi = inchi.strip()

        if not inchi.startswith('InChI='):
            result.add_error("InChI must start with 'InChI='")
            return result

        if len(inchi) > config.MAX_SMILES_LENGTH:
            result.add_error(
                f"InChI too long (max {config.MAX_SMILES_LENGTH} characters)"
            )
            return result

        if not cls.INCHI_PATTERN.match(inchi):
            result.add_error("InChI contains invalid characters or format")
            return result

        return result

    @classmethod
    def validate_inchi_key(cls, inchi_key: str) -> ValidationResult:
        """Validate an InChI Key.

        Args:
            inchi_key: InChI Key to validate

        Returns:
            ValidationResult with is_valid flag and any errors
        """
        result = ValidationResult(is_valid=True)

        if not inchi_key or not isinstance(inchi_key, str):
            result.add_error("InChI Key cannot be empty")
            return result

        inchi_key = inchi_key.strip().upper()

        if len(inchi_key) != 27:
            result.add_error("InChI Key must be exactly 27 characters")
            return result

        if not cls.INCHI_KEY_PATTERN.match(inchi_key):
            result.add_error("Invalid InChI Key format")
            return result

        return result

    @classmethod
    def detect_format(cls, input_str: str) -> str:
        """Detect the format of a chemical structure string.

        Args:
            input_str: Input string to analyze

        Returns:
            Format type: 'smiles', 'inchi', 'inchi_key', or 'unknown'

        Example:
            >>> InputValidator.detect_format('CCO')
            'smiles'
            >>> InputValidator.detect_format('InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3')
            'inchi'
            >>> InputValidator.detect_format('LFQSCWFLJHTTHZ-UHFFFAOYSA-N')
            'inchi_key'
        """
        if not input_str or not isinstance(input_str, str):
            return 'unknown'

        input_str = input_str.strip()

        # Check InChI Key first (most specific format)
        if cls.INCHI_KEY_PATTERN.match(input_str.upper()):
            return 'inchi_key'

        # Check InChI
        if input_str.startswith('InChI='):
            return 'inchi'

        # Check SMILES pattern
        if cls.SMILES_PATTERN.match(input_str):
            return 'smiles'

        return 'unknown'

    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """Sanitize string to prevent XSS attacks.

        Escapes HTML special characters to prevent script injection.

        Args:
            text: Text to sanitize

        Returns:
            HTML-escaped string safe for display

        Example:
            >>> InputValidator.sanitize_html('<script>alert("xss")</script>')
            '&lt;script&gt;alert("xss")&lt;/script&gt;'
        """
        if not isinstance(text, str):
            text = str(text)
        return html.escape(text)

    @classmethod
    def is_safe_input(cls, text: str, is_chemical_format: bool = False) -> bool:
        """Check if input is safe (no injection attempts).

        Checks for common injection patterns like SQL injection
        and script tags. When checking chemical formats (SMILES, InChI),
        allows special characters that are valid in those formats.

        Args:
            text: Text to check
            is_chemical_format: If True, skip checks that would flag
                               valid SMILES/InChI characters

        Returns:
            True if input appears safe
        """
        if not isinstance(text, str):
            return False

        # Check for script patterns (always dangerous)
        if '<script' in text.lower() or 'javascript:' in text.lower():
            return False

        # For chemical formats, only check for script injection
        # SMILES can contain: ()[]=#@+-\/. and other special chars
        # InChI can contain: / - ( ) , ; and other delimiters
        if is_chemical_format:
            return True

        # For non-chemical text, check for SQL injection patterns
        # Check for SQL keywords followed by space
        sql_keyword_pattern = r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\s+"
        if re.search(sql_keyword_pattern, text, re.IGNORECASE):
            return False

        # Check for SQL boolean injection patterns like: ' OR '1'='1
        # These use quotes with OR/AND keywords
        sql_boolean_pattern = r"['\"].*\b(OR|AND)\b.*['\"]"
        if re.search(sql_boolean_pattern, text, re.IGNORECASE):
            return False

        # Check for comment injection ending with --
        if re.search(r'--\s*$', text):
            return False

        # Check for semicolon followed by SQL keyword (statement chaining)
        if re.search(r';\s*(DROP|DELETE|INSERT|UPDATE|SELECT)\b', text, re.IGNORECASE):
            return False

        return True


class FileValidator:
    """Validates uploaded files."""

    @classmethod
    def validate_upload(cls, uploaded_file) -> ValidationResult:
        """Validate an uploaded file.

        Checks:
        1. File is not None
        2. File size is within limits
        3. File extension is allowed

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            ValidationResult with is_valid flag and any errors

        Example:
            >>> result = FileValidator.validate_upload(uploaded_file)
            >>> if result.is_valid:
            ...     df = pd.read_csv(uploaded_file)
        """
        result = ValidationResult(is_valid=True)

        if uploaded_file is None:
            result.add_error("No file uploaded")
            return result

        # Check file size
        try:
            file_size = uploaded_file.size
            if file_size > config.MAX_FILE_SIZE_BYTES:
                result.add_error(
                    f"File too large ({file_size / 1024 / 1024:.1f}MB). "
                    f"Maximum: {config.MAX_FILE_SIZE_MB}MB"
                )
        except AttributeError:
            result.add_warning("Could not determine file size")

        # Check extension
        try:
            filename = uploaded_file.name
            ext = Path(filename).suffix.lower()
            if ext not in config.ALLOWED_EXTENSIONS:
                result.add_error(
                    f"Invalid file type '{ext}'. "
                    f"Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
                )
        except AttributeError:
            result.add_error("Could not determine file type")

        return result

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize a filename to prevent path traversal.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for use

        Example:
            >>> FileValidator.sanitize_filename('../../../etc/passwd')
            'etc_passwd'
        """
        if not filename:
            return "unnamed"

        # Remove path separators and traversal attempts
        filename = filename.replace('..', '')
        filename = filename.replace('/', '_')
        filename = filename.replace('\\', '_')

        # Keep only safe characters
        filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

        # Limit length
        if len(filename) > 255:
            name, ext = (
                filename.rsplit('.', 1) if '.' in filename
                else (filename, '')
            )
            filename = name[:250] + ('.' + ext if ext else '')

        return filename or "unnamed"


class DataFrameValidator:
    """Validates pandas DataFrames."""

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        required_columns: List[str] = None
    ) -> ValidationResult:
        """Validate DataFrame structure and content.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            ValidationResult with is_valid flag and any errors/warnings
        """
        result = ValidationResult(is_valid=True)

        # Check for None or empty
        if df is None:
            result.add_error("DataFrame is None")
            return result

        if df.empty:
            result.add_error("DataFrame is empty")
            return result

        # Check row count
        if len(df) > config.MAX_ROWS_LIMIT:
            result.add_error(
                f"Too many rows ({len(df):,}). "
                f"Maximum: {config.MAX_ROWS_LIMIT:,}"
            )
        elif len(df) > config.MAX_ROWS_WARNING:
            result.add_warning(
                f"Large dataset ({len(df):,} rows). Processing may be slow."
            )

        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                result.add_error(
                    f"Missing required columns: {', '.join(sorted(missing))}"
                )

        return result

    @classmethod
    def column_exists(cls, df: pd.DataFrame, column: str) -> bool:
        """Check if column exists in DataFrame.

        Args:
            df: DataFrame to check
            column: Column name

        Returns:
            True if column exists
        """
        if df is None:
            return False
        return column in df.columns

    @classmethod
    def get_numeric_columns(cls, df: pd.DataFrame) -> List[str]:
        """Get list of numeric column names.

        Args:
            df: DataFrame to analyze

        Returns:
            List of numeric column names
        """
        if df is None or df.empty:
            return []
        return df.select_dtypes(include=['number']).columns.tolist()

    @classmethod
    def get_categorical_columns(
        cls,
        df: pd.DataFrame,
        max_cardinality: int = None
    ) -> List[str]:
        """Get list of categorical column names.

        A column is considered categorical if it has fewer unique values
        than the cardinality threshold.

        Args:
            df: DataFrame to analyze
            max_cardinality: Maximum unique values to be considered categorical

        Returns:
            List of categorical column names
        """
        if df is None or df.empty:
            return []

        if max_cardinality is None:
            max_cardinality = config.MAX_CATEGORICAL_CARDINALITY

        numeric_cols = set(cls.get_numeric_columns(df))
        categorical = []

        for col in df.columns:
            if col not in numeric_cols:
                if df[col].nunique() <= max_cardinality:
                    categorical.append(col)

        return categorical
