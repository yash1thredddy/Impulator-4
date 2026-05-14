"""Tests for backend.modules.imp_presentation — pure SVG range-bar + format helpers."""

import math
import re

import pytest

from backend.modules.imp_presentation import (
    IMP_SCORE_CEILING,
    IMP_SCORE_FLOOR,
    format_imp_score,
    imp_range_position,
    render_imp_range_bar_dynamic,
    render_imp_range_bar_global,
)


class TestConstants:
    """Tests for IMP_SCORE_FLOOR / IMP_SCORE_CEILING constants (VIZ-07)."""

    def test_floor_is_10(self):
        assert IMP_SCORE_FLOOR == 10

    def test_ceiling_is_80(self):
        assert IMP_SCORE_CEILING == 80

    def test_floor_is_int(self):
        assert isinstance(IMP_SCORE_FLOOR, int)

    def test_ceiling_is_int(self):
        assert isinstance(IMP_SCORE_CEILING, int)


class TestFormatImpScore:
    """Tests for format_imp_score (PRES-01)."""

    def test_zero_returns_zero(self):
        assert format_imp_score(0.0) == 0

    def test_one_returns_hundred(self):
        assert format_imp_score(1.0) == 100

    def test_mid_value(self):
        assert format_imp_score(0.5) == 50

    def test_quarter_value(self):
        assert format_imp_score(0.25) == 25

    def test_none_returns_none(self):
        assert format_imp_score(None) is None

    def test_nan_returns_none(self):
        assert format_imp_score(float("nan")) is None

    def test_returns_int_type(self):
        result = format_imp_score(0.5)
        # Pure stdlib — NOT numpy int.
        assert isinstance(result, int)
        assert type(result) is int

    def test_high_precision_value(self):
        # 0.5678 -> 56.78 -> rounds to 57
        assert format_imp_score(0.5678) == 57

    def test_low_precision_value(self):
        # 0.123 -> 12.3 -> rounds to 12
        assert format_imp_score(0.123) == 12


class TestImpRangePosition:
    """Tests for imp_range_position (VIZ-04)."""

    def test_at_floor_returns_zero(self):
        # score == 10 -> position 0 on the [10, 80] band.
        assert imp_range_position(0.10) == 0.0

    def test_at_ceiling_returns_one(self):
        # score == 80 -> position 1 on the [10, 80] band.
        assert imp_range_position(0.80) == 1.0

    def test_mid_value_in_range(self):
        # score == 45 -> (45-10)/(80-10) = 0.5
        result = imp_range_position(0.45)
        assert result is not None
        assert math.isclose(result, 0.5, abs_tol=1e-9)

    def test_below_floor_clamps_to_zero(self):
        # score == 5 -> below floor, clamp to 0.0
        assert imp_range_position(0.05) == 0.0

    def test_above_ceiling_clamps_to_one(self):
        # score == 95 -> above ceiling, clamp to 1.0
        assert imp_range_position(0.95) == 1.0

    def test_none_returns_none(self):
        assert imp_range_position(None) is None

    def test_nan_returns_none(self):
        assert imp_range_position(float("nan")) is None

    def test_custom_floor_ceiling(self):
        # score == 50 with floor=0, ceiling=100 -> position 0.5
        result = imp_range_position(0.50, floor=0, ceiling=100)
        assert result is not None
        assert math.isclose(result, 0.5, abs_tol=1e-9)


class TestRenderImpRangeBarGlobal:
    """Tests for render_imp_range_bar_global (VIZ-01, VIZ-03, VIZ-05)."""

    def test_returns_svg_root(self):
        svg = render_imp_range_bar_global(0.5)
        assert svg.startswith("<svg")

    def test_contains_role_img(self):
        svg = render_imp_range_bar_global(0.5)
        assert 'role="img"' in svg

    def test_contains_viewbox_locked(self):
        svg = render_imp_range_bar_global(0.5)
        assert 'viewBox="0 0 240 28"' in svg

    def test_contains_aria_label(self):
        svg = render_imp_range_bar_global(0.5)
        assert "aria-label=" in svg

    def test_contains_linear_gradient(self):
        svg = render_imp_range_bar_global(0.5)
        assert "<linearGradient" in svg

    def test_three_gradient_stops_smooth_not_banded(self):
        """VIZ-03 anti-banding: exactly three <stop> at 0%, 50%, 100%."""
        svg = render_imp_range_bar_global(0.5)
        assert svg.count("<stop") == 3
        # Offsets locked to 0% / 50% / 100% — no banded duplicates.
        assert 'offset="0%"' in svg
        assert 'offset="50%"' in svg
        assert 'offset="100%"' in svg

    def test_contains_downward_triangle_marker(self):
        """Marker is a downward triangle <polygon> with apex pointing at the bar."""
        svg = render_imp_range_bar_global(0.5)
        # `points="{mx-5},4 {mx+5},4 {mx},12"` — base-y at 4, apex-y at 12.
        match = re.search(r'<polygon points="-?\d+,4 -?\d+,4 -?\d+,12"', svg)
        assert match is not None, f"Marker polygon not found in: {svg[:200]}"

    def test_marker_polygon_appears_exactly_once(self):
        svg = render_imp_range_bar_global(0.5)
        assert svg.count("<polygon") == 1

    def test_none_returns_empty_string(self):
        assert render_imp_range_bar_global(None) == ""

    def test_nan_returns_empty_string(self):
        assert render_imp_range_bar_global(float("nan")) == ""

    def test_below_floor_marker_on_left_oor(self):
        # score=0.05 -> int 5; floor_x = 240 * 10/100 = 24
        svg = render_imp_range_bar_global(0.05)
        match = re.search(r'<polygon points="(-?\d+),4 (-?\d+),4 (-?\d+),12"', svg)
        assert match is not None
        apex_x = int(match.group(3))
        assert apex_x < 24, f"Marker apex {apex_x} should be < floor_x=24"

    def test_above_ceiling_marker_on_right_oor(self):
        # score=0.95 -> int 95; ceil_x = 240 * 80/100 = 192
        svg = render_imp_range_bar_global(0.95)
        match = re.search(r'<polygon points="(-?\d+),4 (-?\d+),4 (-?\d+),12"', svg)
        assert match is not None
        apex_x = int(match.group(3))
        assert apex_x > 192, f"Marker apex {apex_x} should be > ceil_x=192"

    def test_each_call_unique_gradient_id(self):
        """Multiple bars on the same page must not collide on gradient id."""
        svg1 = render_imp_range_bar_global(0.5)
        svg2 = render_imp_range_bar_global(0.5)
        id1 = re.search(r'id="(impGrad-global-[a-f0-9]+)"', svg1)
        id2 = re.search(r'id="(impGrad-global-[a-f0-9]+)"', svg2)
        assert id1 is not None
        assert id2 is not None
        assert id1.group(1) != id2.group(1)

    def test_aria_label_contains_score_integer(self):
        svg = render_imp_range_bar_global(0.5)
        assert "IMP score 50" in svg

    def test_aria_label_contains_raw_float_4dp(self):
        svg = render_imp_range_bar_global(0.5)
        assert "0.5000" in svg

    def test_responsive_width_attributes(self):
        svg = render_imp_range_bar_global(0.5, width_px=240)
        assert 'width="100%"' in svg
        assert "max-width:240px" in svg


class TestRenderImpRangeBarDynamic:
    """Tests for render_imp_range_bar_dynamic (VIZ-02, VIZ-04)."""

    def test_basic_normal_mode_has_gradient(self):
        svg = render_imp_range_bar_dynamic(0.5, 0.3, 0.7)
        assert "<linearGradient" in svg

    def test_basic_normal_mode_has_polygon_marker(self):
        svg = render_imp_range_bar_dynamic(0.5, 0.3, 0.7)
        assert "<polygon" in svg

    def test_normal_mode_starts_with_svg(self):
        svg = render_imp_range_bar_dynamic(0.5, 0.3, 0.7)
        assert svg.startswith("<svg")

    def test_observed_min_equals_max_sentinel(self):
        """Single-compound query — sentinel mode (no gradient, no marker)."""
        svg = render_imp_range_bar_dynamic(0.5, 0.5, 0.5)
        assert "<linearGradient" not in svg
        assert "<polygon" not in svg
        assert svg.startswith("<svg")

    def test_observed_min_equals_max_sentinel_message(self):
        svg = render_imp_range_bar_dynamic(0.5, 0.5, 0.5)
        assert "Only one compound" in svg

    def test_observed_min_none_sentinel(self):
        svg = render_imp_range_bar_dynamic(0.5, None, 0.7)
        assert "<linearGradient" not in svg
        assert "<polygon" not in svg

    def test_observed_max_none_sentinel(self):
        svg = render_imp_range_bar_dynamic(0.5, 0.3, None)
        assert "<linearGradient" not in svg
        assert "<polygon" not in svg

    def test_observed_both_none_sentinel(self):
        svg = render_imp_range_bar_dynamic(0.5, None, None)
        assert "<linearGradient" not in svg
        assert "<polygon" not in svg

    def test_observed_nan_min_sentinel(self):
        svg = render_imp_range_bar_dynamic(0.5, float("nan"), 0.7)
        assert "<linearGradient" not in svg
        assert "<polygon" not in svg

    def test_observed_nan_max_sentinel(self):
        svg = render_imp_range_bar_dynamic(0.5, 0.3, float("nan"))
        assert "<linearGradient" not in svg
        assert "<polygon" not in svg

    def test_unavailable_message_when_bounds_none(self):
        svg = render_imp_range_bar_dynamic(0.5, None, 0.7)
        assert "range is unavailable" in svg

    def test_none_score_returns_empty_string(self):
        assert render_imp_range_bar_dynamic(None, 0.3, 0.7) == ""

    def test_nan_score_returns_empty_string(self):
        assert render_imp_range_bar_dynamic(float("nan"), 0.3, 0.7) == ""

    def test_contains_role_img(self):
        svg = render_imp_range_bar_dynamic(0.5, 0.3, 0.7)
        assert 'role="img"' in svg

    def test_normal_mode_unique_gradient_id(self):
        svg1 = render_imp_range_bar_dynamic(0.5, 0.3, 0.7)
        svg2 = render_imp_range_bar_dynamic(0.5, 0.3, 0.7)
        id1 = re.search(r'id="(impGrad-dynamic-[a-f0-9]+)"', svg1)
        id2 = re.search(r'id="(impGrad-dynamic-[a-f0-9]+)"', svg2)
        assert id1 is not None
        assert id2 is not None
        assert id1.group(1) != id2.group(1)


class TestSvgSafety:
    """Security tests for SVG output (T-21-01, anti-requirements 5 + 6)."""

    def test_no_script_tag_in_global(self):
        assert "<script" not in render_imp_range_bar_global(0.5)

    def test_no_script_tag_in_dynamic(self):
        assert "<script" not in render_imp_range_bar_dynamic(0.5, 0.3, 0.7)

    def test_no_script_tag_in_sentinel(self):
        assert "<script" not in render_imp_range_bar_dynamic(0.5, 0.5, 0.5)

    def test_no_onload_attribute(self):
        assert "onload=" not in render_imp_range_bar_global(0.5)

    def test_no_onclick_attribute(self):
        assert "onclick=" not in render_imp_range_bar_global(0.5)

    def test_no_onerror_attribute(self):
        assert "onerror=" not in render_imp_range_bar_global(0.5)

    def test_no_xlink_href_global(self):
        assert "xlink:href" not in render_imp_range_bar_global(0.5)

    def test_no_xlink_href_dynamic(self):
        assert "xlink:href" not in render_imp_range_bar_dynamic(0.5, 0.3, 0.7)

    def test_compound_name_xss_attempt_escaped_global(self):
        """T-21-01: <script>alert(1)</script> in compound_name must NOT yield <script."""
        svg = render_imp_range_bar_global(
            0.5, compound_name="<script>alert(1)</script>"
        )
        assert "<script" not in svg
        # Escaped form may appear instead.
        assert "&lt;script&gt;" in svg or "&lt;script" in svg

    def test_compound_name_xss_attempt_escaped_dynamic(self):
        svg = render_imp_range_bar_dynamic(
            0.5, 0.3, 0.7, compound_name="<script>alert(1)</script>"
        )
        assert "<script" not in svg

    def test_compound_name_xss_attempt_escaped_sentinel(self):
        svg = render_imp_range_bar_dynamic(
            0.5, 0.5, 0.5, compound_name="<script>alert(1)</script>"
        )
        assert "<script" not in svg

    def test_compound_name_quote_injection_escaped(self):
        """Closing-quote + injected attribute must not yield a working on* attr."""
        svg = render_imp_range_bar_global(0.5, compound_name='" onclick="alert(1)')
        # The injection must not produce an unescaped attribute boundary.
        assert ' onclick="alert(1)' not in svg
        # Quote should be escaped (html.escape with quote=True -> &quot;).
        assert "&quot;" in svg

    def test_compound_name_apostrophe_injection_escaped(self):
        svg = render_imp_range_bar_global(0.5, compound_name="' onclick='alert(1)")
        assert "onclick=" not in svg or "&#x27;" in svg or "&apos;" in svg


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
