"""
IMP Presentation Module - pure-Python rendering of integer IMP score + inline-SVG range bars.
Decoupled from Streamlit and pandas for both Streamlit pages and the static HTML report.

This module is the single source of truth for IMP-score integer formatting and
inline-SVG range-bar rendering. Pure functions (no Streamlit, pandas, network, or
logging side-effects) so identical SVG strings flow into both the Streamlit page
and the exported static HTML report.

**Public API**:
- ``IMP_SCORE_FLOOR`` (int = 10) — empirical global lower bound for the in-range zone.
- ``IMP_SCORE_CEILING`` (int = 80) — empirical global upper bound for the in-range zone.
- ``format_imp_score(raw)`` — banker's-rounded integer in ``[0, 100]`` (or ``None``).
- ``imp_range_position(raw, *, floor, ceiling)`` — float in ``[0.0, 1.0]`` clamped position.
- ``render_imp_range_bar_global(raw, *, width_px, compound_name)`` — inline SVG string.
- ``render_imp_range_bar_dynamic(raw, observed_min, observed_max, *, width_px, compound_name)``
  — inline SVG string with sentinel rendering for degenerate observed ranges.

**Edge cases**:
- ``None`` or ``NaN`` score → ``format_imp_score`` returns ``None``; range-bar renderers
  return the empty string ``""`` so the caller can substitute the "IMP Score: N/A" text.
- ``observed_min == observed_max`` or either bound ``None``/``NaN`` → dynamic renderer
  emits a flat sentinel SVG (no gradient, no marker).
- Score outside ``[IMP_SCORE_FLOOR, IMP_SCORE_CEILING]`` → marker still renders at its
  actual x-position on the gray out-of-range segment; no "outside range" caption.

**Security** (see 21-UI-SPEC.md §Anti-Requirements lines 300-316):
- All str inputs (compound names, ARIA labels) pass through ``html.escape`` before
  f-string interpolation into SVG attributes.
- No JavaScript tags, no inline event-handler attributes, no external linked refs
  (anti-requirements 5 + 6; see 21-UI-SPEC.md for the full forbidden-token list).
- Raw ``IMP_Final_Score`` floats are never interpolated verbatim — they pass through
  ``format_imp_score`` (-> int|None) or ``imp_range_position`` (-> float clamped to
  ``[0, 1]``) first.

Reference: 21-UI-SPEC.md §SVG Anatomy (lines 204-244) and §Color (lines 88-122) are
authoritative for the visual contract; this module is the canonical implementation.
"""

import html
import math
import secrets

# =============================================================================
# IMP Score Range Constants (empirical global bounds per VIZ-07)
# =============================================================================

IMP_SCORE_FLOOR: int = 10
IMP_SCORE_CEILING: int = 80

# =============================================================================
# SVG Color Palette (21-UI-SPEC.md §Color lines 88-122 — locked)
# Gradient stops: green -> yellow -> red, smooth interpolation (no banding)
# =============================================================================

_GRADIENT_LOW: str = "#22c55e"  # Stop at 0% — low end of in-range zone
_GRADIENT_MID: str = "#eab308"  # Stop at 50% — mid of in-range zone
_GRADIENT_HIGH: str = "#dc2626"  # Stop at 100% — high end of in-range zone
_OOR_COLOR: str = "#d1d5db"  # Out-of-range bar segments (left + right)
_MARKER_FILL: str = "#111827"  # Downward triangle marker fill
_MARKER_STROKE: str = "#ffffff"  # 1px white stroke around marker for contrast
_TEXT_BODY: str = "#111827"  # Body/score-integer text
_TEXT_MUTED: str = "#6b7280"  # Captions, tick labels, muted text


# =============================================================================
# Format helpers
# =============================================================================


def format_imp_score(raw: float | None) -> int | None:
    """
    Convert a raw IMP score (float in ``[0, 1]``) into an integer in ``[0, 100]``.

    Uses Python's built-in ``round`` (banker's rounding — half-to-even) per
    CONTEXT.md decision #1. Returns ``None`` for ``None`` or ``NaN`` input so the
    caller can render the "IMP Score: N/A" sentinel.

    Args:
        raw: Raw IMP score float in ``[0, 1]``, or ``None``.

    Returns:
        Integer in ``[0, 100]``, or ``None`` when input is ``None`` or ``NaN``.

    Example:
        >>> format_imp_score(0.5)
        50
        >>> format_imp_score(1.0)
        100
        >>> format_imp_score(None) is None
        True
    """
    if raw is None:
        return None
    if math.isnan(raw):
        return None
    return round(raw * 100)


def imp_range_position(
    raw: float | None,
    *,
    floor: int = IMP_SCORE_FLOOR,
    ceiling: int = IMP_SCORE_CEILING,
) -> float | None:
    """
    Compute the marker's position in ``[0.0, 1.0]`` within the ``[floor, ceiling]`` band.

    Used to place the triangle marker inside the in-range gradient zone. Scores
    below ``floor`` clamp to ``0.0``; scores above ``ceiling`` clamp to ``1.0``.
    Returns ``None`` for ``None`` or ``NaN`` input.

    Args:
        raw: Raw IMP score float in ``[0, 1]``, or ``None``.
        floor: Lower bound integer score (default ``IMP_SCORE_FLOOR`` = 10).
        ceiling: Upper bound integer score (default ``IMP_SCORE_CEILING`` = 80).

    Returns:
        Float in ``[0.0, 1.0]`` (clamped), or ``None``.

    Example:
        >>> imp_range_position(0.10)
        0.0
        >>> imp_range_position(0.80)
        1.0
        >>> imp_range_position(None) is None
        True
    """
    score_int = format_imp_score(raw)
    if score_int is None:
        return None
    span = ceiling - floor
    if span <= 0:
        return 0.0
    position = (score_int - floor) / span
    if position < 0.0:
        return 0.0
    if position > 1.0:
        return 1.0
    return position


# =============================================================================
# SVG renderers (inline, stdlib-only, no JS, no external refs)
# =============================================================================


def _build_gradient_defs(grad_id: str) -> str:
    """Return the ``<defs>`` block with a single three-stop linear gradient."""
    return (
        f'<defs><linearGradient id="{grad_id}" x1="0" x2="1">'
        f'<stop offset="0%" stop-color="{_GRADIENT_LOW}"/>'
        f'<stop offset="50%" stop-color="{_GRADIENT_MID}"/>'
        f'<stop offset="100%" stop-color="{_GRADIENT_HIGH}"/>'
        f"</linearGradient></defs>"
    )


def _format_marker_polygon(marker_x: float) -> str:
    """Render the downward-triangle marker polygon at ``marker_x``."""
    # Triangle apex at y=12 (touching top of bar at y=12); base at y=4.
    # We render integer coordinates to keep the SVG stable across renders.
    mx = int(round(marker_x))
    return (
        f'<polygon points="{mx - 5},4 {mx + 5},4 {mx},12" '
        f'fill="{_MARKER_FILL}" stroke="{_MARKER_STROKE}" stroke-width="1"/>'
    )


def _escape_label(label: str) -> str:
    """Escape a user-derived string for safe interpolation into an SVG attribute."""
    return html.escape(str(label), quote=True)


def render_imp_range_bar_global(
    raw: float | None,
    *,
    width_px: int = 240,
    compound_name: str | None = None,
) -> str:
    """
    Render the global-reference range bar as an inline SVG string.

    The global bar uses the empirical bounds ``[IMP_SCORE_FLOOR, IMP_SCORE_CEILING]``
    (10..80). Scores outside this range place the marker on the gray
    out-of-range segments at the bar's left or right.

    Args:
        raw: Raw IMP score float in ``[0, 1]``, or ``None`` / ``NaN``.
        width_px: Maximum rendered width in CSS pixels (default 240).
        compound_name: Optional untrusted string appended to the ARIA label.
            Escaped via ``html.escape`` before interpolation.

    Returns:
        Inline SVG string starting with ``<svg``, or ``""`` when ``raw`` is
        ``None`` or ``NaN`` (caller renders "IMP Score: N/A" text instead).

    Example:
        >>> svg = render_imp_range_bar_global(0.5)
        >>> svg.startswith('<svg')
        True
        >>> render_imp_range_bar_global(None)
        ''
    """
    if raw is None or math.isnan(raw):
        return ""

    score_int = format_imp_score(raw)
    # format_imp_score guards None/NaN above, so score_int is guaranteed int.
    assert score_int is not None  # for type-narrowing

    floor_x = width_px * (IMP_SCORE_FLOOR / 100)
    ceil_x = width_px * (IMP_SCORE_CEILING / 100)

    # Marker position in actual score space (0..100), clamped to the visible bar.
    raw_marker_x = width_px * (score_int / 100)
    marker_x = max(0.0, min(float(width_px), raw_marker_x))

    grad_id = f"impGrad-global-{secrets.token_hex(3)}"

    aria_label = (
        f"IMP score {score_int} on global reference scale "
        f"{IMP_SCORE_FLOOR} to {IMP_SCORE_CEILING}: {raw:.4f}"
    )
    if compound_name is not None:
        aria_label = f"{aria_label} for {compound_name}"
    safe_aria_label = _escape_label(aria_label)

    defs = _build_gradient_defs(grad_id)
    rect_oor_left = (
        f'<rect x="0" y="12" width="{floor_x:g}" height="16" fill="{_OOR_COLOR}"/>'
    )
    rect_in_range = (
        f'<rect x="{floor_x:g}" y="12" width="{ceil_x - floor_x:g}" height="16" '
        f'fill="url(#{grad_id})"/>'
    )
    rect_oor_right = (
        f'<rect x="{ceil_x:g}" y="12" width="{width_px - ceil_x:g}" height="16" '
        f'fill="{_OOR_COLOR}"/>'
    )
    marker = _format_marker_polygon(marker_x)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width_px} 28" width="100%" height="28" '
        f'style="max-width:{width_px}px;height:28px;display:block" '
        f'role="img" aria-label="{safe_aria_label}">'
        f"{defs}"
        f"{rect_oor_left}"
        f"{rect_in_range}"
        f"{rect_oor_right}"
        f"{marker}"
        f"</svg>"
    )


def _render_sentinel_dynamic(
    width_px: int,
    sentinel_message: str,
    compound_name: str | None,
) -> str:
    """Render the flat-gray sentinel SVG for degenerate dynamic ranges."""
    aria_label = sentinel_message
    if compound_name is not None:
        aria_label = f"{aria_label} for {compound_name}"
    safe_aria_label = _escape_label(aria_label)

    rect = f'<rect x="0" y="12" width="{width_px}" height="16" fill="{_OOR_COLOR}"/>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width_px} 28" width="100%" height="28" '
        f'style="max-width:{width_px}px;height:28px;display:block" '
        f'role="img" aria-label="{safe_aria_label}">'
        f"{rect}"
        f"</svg>"
    )


def render_imp_range_bar_dynamic(
    raw: float | None,
    observed_min: float | None,
    observed_max: float | None,
    *,
    width_px: int = 240,
    compound_name: str | None = None,
) -> str:
    """
    Render the dynamic per-query range bar as an inline SVG string.

    The dynamic bar uses the observed corpus bounds ``[observed_min, observed_max]``
    (both raw floats in ``[0, 1]``). The entire bar is in-range by construction
    (endpoints == observed bounds), so there are no gray out-of-range segments
    in normal mode.

    Sentinel mode (flat gray bar, no gradient, no marker) is rendered when:
    - ``observed_min`` is ``None`` or ``NaN``, OR
    - ``observed_max`` is ``None`` or ``NaN``, OR
    - ``observed_min == observed_max`` (single-compound query).

    Args:
        raw: Raw IMP score float in ``[0, 1]``, or ``None`` / ``NaN``.
        observed_min: Observed lower bound (raw float in ``[0, 1]``) or ``None``.
        observed_max: Observed upper bound (raw float in ``[0, 1]``) or ``None``.
        width_px: Maximum rendered width in CSS pixels (default 240).
        compound_name: Optional untrusted string appended to the ARIA label.

    Returns:
        Inline SVG string starting with ``<svg``, or ``""`` when ``raw`` is
        ``None`` or ``NaN``.

    Example:
        >>> svg = render_imp_range_bar_dynamic(0.5, 0.3, 0.7)
        >>> '<linearGradient' in svg
        True
        >>> sentinel = render_imp_range_bar_dynamic(0.5, 0.5, 0.5)
        >>> '<linearGradient' in sentinel
        False
    """
    if raw is None or math.isnan(raw):
        return ""

    # Sentinel-mode detection
    if observed_min is None or observed_max is None:
        return _render_sentinel_dynamic(
            width_px, "This query's range is unavailable", compound_name
        )
    if math.isnan(observed_min) or math.isnan(observed_max):
        return _render_sentinel_dynamic(
            width_px, "This query's range is unavailable", compound_name
        )
    if observed_min == observed_max:
        return _render_sentinel_dynamic(
            width_px, "Only one compound in this query's range", compound_name
        )

    # Normal mode
    score_int = format_imp_score(raw)
    min_int = format_imp_score(observed_min)
    max_int = format_imp_score(observed_max)
    assert score_int is not None
    assert min_int is not None
    assert max_int is not None

    # After format_imp_score, the integer representations could collapse to equal
    # (e.g. 0.500 and 0.504 both -> 50); treat that the same as the sentinel
    # min==max case to keep the gradient meaningful.
    span = max_int - min_int
    if span == 0:
        return _render_sentinel_dynamic(
            width_px, "Only one compound in this query's range", compound_name
        )

    # 0-100 absolute scale (matches global bar). Gray out-of-observed-range
    # zones flank the gradient-filled [min_int, max_int] band, so a score at
    # the dataset max appears at its true global position (not pegged to the
    # right edge). Marker is at the score's 0-100 position.
    min_x = width_px * (min_int / 100.0)
    max_x = width_px * (max_int / 100.0)
    marker_x = max(0.0, min(float(width_px), width_px * (score_int / 100.0)))

    grad_id = f"impGrad-dynamic-{secrets.token_hex(3)}"

    aria_label = (
        f"IMP score {score_int} on 0-100 absolute scale; "
        f"this query's observed range is {min_int} to {max_int}: {raw:.4f}"
    )
    if compound_name is not None:
        aria_label = f"{aria_label} for {compound_name}"
    safe_aria_label = _escape_label(aria_label)

    defs = _build_gradient_defs(grad_id)
    rect_oor_left = (
        f'<rect x="0" y="12" width="{min_x:.4f}" height="16" fill="{_OOR_COLOR}"/>'
    )
    rect_in_range = (
        f'<rect x="{min_x:.4f}" y="12" width="{(max_x - min_x):.4f}" height="16" '
        f'fill="url(#{grad_id})"/>'
    )
    rect_oor_right = (
        f'<rect x="{max_x:.4f}" y="12" width="{(width_px - max_x):.4f}" height="16" '
        f'fill="{_OOR_COLOR}"/>'
    )
    marker = _format_marker_polygon(marker_x)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width_px} 28" width="100%" height="28" '
        f'style="max-width:{width_px}px;height:28px;display:block" '
        f'role="img" aria-label="{safe_aria_label}">'
        f"{defs}"
        f"{rect_oor_left}"
        f"{rect_in_range}"
        f"{rect_oor_right}"
        f"{marker}"
        f"</svg>"
    )


__all__ = [
    "IMP_SCORE_FLOOR",
    "IMP_SCORE_CEILING",
    "format_imp_score",
    "imp_range_position",
    "render_imp_range_bar_global",
    "render_imp_range_bar_dynamic",
]
