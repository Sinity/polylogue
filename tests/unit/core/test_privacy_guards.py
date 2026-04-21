"""Edge-case tests for schema inference privacy guards.

Supplements tests/unit/core/test_schema.py which already covers
the main _is_safe_enum_value heuristics extensively. This file
focuses on boundary conditions and corner cases.
"""

from __future__ import annotations

from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.schemas.schema_inference import (
    _annotate_schema,
    _collect_field_stats,
    _is_content_field,
    _is_safe_enum_value,
)
from tests.infra.schema_access import schema_property, schema_values

LETTER_CATEGORIES: tuple[Literal["L"], ...] = ("L",)


class TestSafeEnumEdgeCases:
    """Boundary conditions for _is_safe_enum_value."""

    def test_exactly_max_length_is_accepted(self) -> None:
        """A value at exactly 50 chars (the cap) should still pass if safe."""
        value = "a" * 50
        assert _is_safe_enum_value(value)

    def test_one_over_max_length_is_rejected(self) -> None:
        """A value at 51 chars is rejected."""
        value = "a" * 51
        assert not _is_safe_enum_value(value)

    def test_private_tld_mixed_case(self) -> None:
        """Domain-like values with known TLDs are rejected regardless of case."""
        assert not _is_safe_enum_value("Example.COM")
        assert not _is_safe_enum_value("test.IO")
        assert not _is_safe_enum_value("host.Org")

    def test_domain_trailing_content(self) -> None:
        """TLD pattern uses word boundary -- trailing text after TLD is handled."""
        # ".com" followed by nothing → rejected
        assert not _is_safe_enum_value("example.com")
        # ".com" with trailing slash → still has "://" path? No -- just has dot
        assert not _is_safe_enum_value("host.net")

    def test_non_public_tlds_are_allowed(self) -> None:
        """Domains with TLDs not in the denylist pass through."""
        # .xyz, .dev, .app are not in the denylist
        assert _is_safe_enum_value("x.xyz")
        assert _is_safe_enum_value("a.dev")

    def test_timestamp_prefix_rejected(self) -> None:
        """ISO-like timestamps at string start are rejected."""
        assert not _is_safe_enum_value("2024-01-15T10:30:00")
        assert not _is_safe_enum_value("2024-01-15 10:30:00")

    def test_non_ascii_rejected(self) -> None:
        """Non-ASCII values are rejected even if short."""
        assert not _is_safe_enum_value("caf\u00e9")
        assert not _is_safe_enum_value("\u00fcber")

    @given(
        st.text(
            alphabet=st.characters(whitelist_categories=LETTER_CATEGORIES, min_codepoint=128), min_size=1, max_size=20
        )
    )
    @settings(max_examples=30)
    def test_non_ascii_always_rejected(self, value: str) -> None:
        """Property: any non-ASCII string is rejected."""
        assert not _is_safe_enum_value(value)

    def test_empty_string_rejected(self) -> None:
        """Empty string is rejected."""
        assert not _is_safe_enum_value("")

    def test_slash_prefix_rejected(self) -> None:
        """Strings starting with / are rejected (path-like)."""
        assert not _is_safe_enum_value("/usr/bin")

    def test_plus_prefix_rejected(self) -> None:
        """Strings starting with + are rejected (phone-like)."""
        assert not _is_safe_enum_value("+1234567890")


class TestContentFieldDetection:
    """Edge cases for _is_content_field."""

    def test_nested_path_terminal_match(self) -> None:
        """Content field detection works on nested dotted paths."""
        assert _is_content_field("$.mapping.*.message.content.text")
        assert _is_content_field("deep.path.title")

    def test_array_marker_stripped(self) -> None:
        """Array markers like [*] are stripped before matching."""
        assert _is_content_field("items[*].description")

    def test_non_content_field(self) -> None:
        """Non-content fields are correctly identified."""
        assert not _is_content_field("$.role")
        assert not _is_content_field("type")
        assert not _is_content_field("status_code")


class TestAnnotationPrivacyIntegration:
    """Integration tests: privacy guards applied through the annotation pipeline."""

    def test_high_cardinality_values_excluded_from_annotation(self) -> None:
        """Fields with >50 distinct values don't get x-polylogue-values."""
        # Create samples with 60 distinct role values (high cardinality)
        samples = [{"status": f"status-{i}"} for i in range(60)]
        stats = _collect_field_stats(samples)

        schema: dict[str, object] = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, path="$")

        # High cardinality → not enum-like → no values annotation
        assert not schema_values(schema_property(annotated, "status"))

    def test_content_field_values_excluded(self) -> None:
        """Known content fields never get x-polylogue-values even if low cardinality."""
        samples = [{"title": "hello"}, {"title": "world"}]
        stats = _collect_field_stats(samples)

        schema: dict[str, object] = {"type": "object", "properties": {"title": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, path="$")

        # "title" is in _CONTENT_FIELD_NAMES → values suppressed
        assert not schema_values(schema_property(annotated, "title"))
