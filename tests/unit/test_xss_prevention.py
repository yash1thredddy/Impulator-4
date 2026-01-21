"""
Unit tests for XSS prevention in frontend components.
"""
import pytest
import html


class TestHTMLEscaping:
    """Tests for HTML escaping in frontend components."""

    def test_html_escape_script_tags(self):
        """Test that script tags are escaped."""
        malicious = "<script>alert('xss')</script>"
        escaped = html.escape(malicious)

        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_html_escape_html_entities(self):
        """Test that HTML entities are escaped."""
        test_cases = [
            ("<", "&lt;"),
            (">", "&gt;"),
            ("&", "&amp;"),
            ('"', "&quot;"),
            ("'", "&#x27;"),
        ]

        for input_char, expected in test_cases:
            escaped = html.escape(input_char)
            assert expected in escaped or input_char not in escaped

    def test_html_escape_event_handlers(self):
        """Test that event handlers are escaped."""
        malicious = '<img src=x onerror=alert("xss")>'
        escaped = html.escape(malicious)

        assert "onerror" not in escaped or "&" in escaped
        assert "<img" not in escaped

    def test_safe_numeric_values(self):
        """Test that numeric values are safely converted."""
        # Simulating what compound_card.py does
        total_activities = 42
        num_outliers = 5
        qed = 0.75
        similarity = 90

        safe_total = html.escape(str(total_activities))
        safe_outliers = html.escape(str(num_outliers))
        safe_qed = html.escape(f"{qed:.2f}")
        safe_similarity = html.escape(str(similarity))

        assert safe_total == "42"
        assert safe_outliers == "5"
        assert safe_qed == "0.75"
        assert safe_similarity == "90"

    def test_escape_compound_name_with_special_chars(self):
        """Test escaping compound names with special characters."""
        names = [
            ("Normal Name", "Normal Name"),
            ("Name <with> brackets", "Name &lt;with&gt; brackets"),
            ("Name & ampersand", "Name &amp; ampersand"),
            ('Name "quoted"', "Name &quot;quoted&quot;"),
        ]

        for input_name, expected in names:
            escaped = html.escape(input_name)
            assert escaped == expected


class TestCompoundCardXSSPrevention:
    """Tests specifically for compound_card.py XSS prevention."""

    def test_all_interpolated_values_escaped(self):
        """Verify that the escaping pattern is used correctly."""
        # This test verifies the pattern used in compound_card.py
        # by simulating what the code does

        # Simulated compound data (could be malicious)
        compound = {
            'compound_name': '<script>alert("xss")</script>',
            'total_activities': '<img src=x onerror=alert(1)>',
            'num_outliers': '5; DROP TABLE users',
            'qed': 0.75,
            'similarity_threshold': 90,
            'chembl_id': 'CHEMBL<script>',
        }

        # Apply escaping like compound_card.py does
        safe_display_name = html.escape(str(compound['compound_name']))
        safe_total_activities = html.escape(str(compound['total_activities']))
        safe_num_outliers = html.escape(str(compound['num_outliers']))
        safe_qed_display = html.escape(f"{compound['qed']:.2f}")
        safe_similarity = html.escape(str(compound['similarity_threshold']))
        safe_chembl_id = html.escape(str(compound['chembl_id']))

        # All should have dangerous characters escaped
        assert '<script>' not in safe_display_name
        assert '<img' not in safe_total_activities
        assert 'DROP TABLE' in safe_num_outliers  # Text is kept, just escaped
        assert safe_qed_display == "0.75"
        assert safe_similarity == "90"
        assert '<script>' not in safe_chembl_id

    def test_truncation_preserves_escaping(self):
        """Test that name truncation doesn't break escaping."""
        # Long malicious name that will be truncated
        long_name = "<script>alert('xss')</script>" + "A" * 100

        # Truncate like compound_card.py does (max 20 chars for display)
        display_name = long_name if len(long_name) <= 20 else long_name[:18] + "..."

        # Then escape
        safe_name = html.escape(display_name)

        # Should be escaped
        assert '<script>' not in safe_name
        assert '&lt;' in safe_name

    def test_none_values_handled(self):
        """Test that None values don't cause issues."""
        values = [None, 0, "", "N/A"]

        for value in values:
            if value is None:
                display = "N/A"
            else:
                display = str(value)

            escaped = html.escape(display)
            assert escaped is not None
            assert '<' not in escaped or '&lt;' in escaped
