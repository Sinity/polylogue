"""Privacy guard tests for schema inference.

Covers all heuristics in _is_safe_enum_value and the field-level filters,
plus the three new guards added in this session:
  1. Cross-conversation threshold (value seen in <N distinct conversations excluded)
  2. Key denylist expansion (body, message, input, output never yield enums)
  3. Private TLD denylist (.local, .lan, .corp, .internal, .home rejected)

Tests are grouped by guard type so failures pinpoint which heuristic regressed.
"""

from __future__ import annotations

import pytest

from polylogue.schemas.schema_inference import (
    _annotate_schema,
    _collect_field_stats,
    _is_content_field,
    _is_safe_enum_value,
)

# =============================================================================
# _is_safe_enum_value — existing heuristics (regression guard)
# =============================================================================

class TestSafeEnumValueExistingGuards:
    """All existing _is_safe_enum_value filters must still hold."""

    # --- Values that MUST pass (structural enum candidates) ---

    @pytest.mark.parametrize("value", [
        "user",
        "assistant",
        "system",
        "chatgpt",
        "application/json",
        "text/plain",
        "gpt-4",
        "claude-3-opus",
        "active",
        "pending",
        "disabled",
    ])
    def test_safe_structural_values_pass(self, value: str) -> None:
        assert _is_safe_enum_value(value), f"Expected {value!r} to be safe"

    # --- URL rejection ---

    @pytest.mark.parametrize("value", [
        "https://example.com/path",
        "http://api.openai.com/v1/chat",
        "ftp://files.corp.internal/export.zip",
    ])
    def test_urls_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected URL {value!r} to be rejected"

    # --- Email rejection ---

    @pytest.mark.parametrize("value", [
        "user@example.com",
        "alice@corp.internal",
    ])
    def test_emails_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected email {value!r} to be rejected"

    # --- Natural language / whitespace rejection ---

    @pytest.mark.parametrize("value", [
        "Hello world",
        "This is a message",
        "multi\nline\ncontent",
    ])
    def test_sentences_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected sentence {value!r} to be rejected"

    # --- Capitalized words now PASS (CamelCase check removed) ---

    @pytest.mark.parametrize("value", [
        "Reasoning",
        "Thinking",
        "GitHub",
        "None",
        "Alice",  # single word — no longer blocked (whitespace check handles multi-word names)
    ])
    def test_capitalized_words_pass(self, value: str) -> None:
        assert _is_safe_enum_value(value), f"Expected {value!r} to pass (CamelCase check removed)"

    # --- Public domain / TLD rejection ---

    @pytest.mark.parametrize("value", [
        "openai.com",
        "api.anthropic.com",
        "storage.googleapis.com",
        "cdn.example.net",
    ])
    def test_public_domains_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected domain {value!r} to be rejected"

    # --- File extension rejection ---

    @pytest.mark.parametrize("value", [
        "document.pdf",
        "archive.zip",
        "data.json",
        "script.py",
        "export.csv",
    ])
    def test_file_extensions_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected filename {value!r} to be rejected"

    # --- Timestamp rejection ---

    @pytest.mark.parametrize("value", [
        "2024-01-15T10:30:00Z",
        "2024-01-15",
    ])
    def test_timestamps_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected timestamp {value!r} to be rejected"

    # --- High-entropy token rejection ---

    @pytest.mark.parametrize("value", [
        "sk-abc123XYZ789def456",
        "Bearer eyJhbGciOiJSUzI1NiJ9",
        "dQw4w9WgXcQ",  # YouTube video ID (11 chars, now caught with lowered threshold)
    ])
    def test_high_entropy_tokens_rejected(self, value: str) -> None:
        # These are opaque tokens — reject even without explicit URL indicators
        assert not _is_safe_enum_value(value), f"Expected token {value!r} to be rejected"

    # --- Quoted high-entropy tokens (Gemini format) ---

    def test_quoted_high_entropy_stripped(self) -> None:
        """Gemini exports embed values in double quotes — quotes are stripped before check."""
        # A YouTube-ID-like token wrapped in quotes
        assert not _is_safe_enum_value('"dQw4w9WgXcQ"')

    # --- Model slug exemption (dash-separated structural tokens) ---

    @pytest.mark.parametrize("value", [
        "gpt-4-code-interpreter",
        "claude-haiku-4-5-20251001",
        "gemini-2-5-pro",
        "gpt-4o-mini",
    ])
    def test_model_slugs_pass(self, value: str) -> None:
        """Model slugs with 2+ dashes and short segments should pass high-entropy check."""
        assert _is_safe_enum_value(value), f"Expected model slug {value!r} to pass"

    # --- Non-ASCII rejection ---

    def test_non_ascii_rejected(self) -> None:
        assert not _is_safe_enum_value("café"), "Non-ASCII value should be rejected"

    # --- Empty / overlength rejection ---

    def test_empty_string_rejected(self) -> None:
        assert not _is_safe_enum_value(""), "Empty string should be rejected"

    def test_overlength_rejected(self) -> None:
        long_value = "a" * 51
        assert not _is_safe_enum_value(long_value), "Value longer than 50 chars should be rejected"


# =============================================================================
# Guard 1: Cross-conversation threshold
# =============================================================================

class TestCrossConversationThreshold:
    """Values seen in fewer than N conversations are suppressed from schema enums."""

    def _make_schema_with_samples(
        self,
        values_by_conv: dict[str, list[str]],
        *,
        min_conversation_count: int = 3,
    ) -> dict:
        """Build schema annotations from samples grouped by conversation ID."""
        samples = []
        conv_ids = []
        for conv_id, values in values_by_conv.items():
            for v in values:
                samples.append({"$.status": v})
                conv_ids.append(conv_id)

        # _collect_field_stats expects flat dicts with the actual field as a key
        # Use a simple structure: {"status": value}
        flat_samples = [{"status": v} for v in [v for vals in values_by_conv.values() for v in vals]]
        flat_conv_ids = [cid for cid, vals in values_by_conv.items() for _ in vals]

        stats = _collect_field_stats(flat_samples, conversation_ids=flat_conv_ids)
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_conversation_count=min_conversation_count)
        return annotated.get("properties", {}).get("status", {})

    def test_value_in_one_conv_excluded_at_threshold_3(self) -> None:
        """A value seen in only 1 conversation is excluded when threshold=3."""
        values_by_conv = {
            "conv_A": ["rare_status"],
            "conv_B": ["common"],
            "conv_C": ["common"],
            "conv_D": ["common"],
        }
        status_schema = self._make_schema_with_samples(values_by_conv, min_conversation_count=3)
        enum_vals = status_schema.get("x-polylogue-values", [])
        assert "rare_status" not in enum_vals, "Single-conversation value should be excluded"

    def test_value_in_three_convs_included_at_threshold_3(self) -> None:
        """A value seen in exactly 3 conversations passes the threshold."""
        values_by_conv = {
            "conv_A": ["stable_role"],
            "conv_B": ["stable_role"],
            "conv_C": ["stable_role"],
        }
        status_schema = self._make_schema_with_samples(values_by_conv, min_conversation_count=3)
        enum_vals = status_schema.get("x-polylogue-values", [])
        assert "stable_role" in enum_vals, "Value in 3 conversations should pass threshold=3"

    def test_threshold_1_includes_single_conv_values(self) -> None:
        """Default threshold=1 preserves existing behaviour (no cross-conv filtering)."""
        values_by_conv = {"conv_A": ["only_here", "only_here"]}
        status_schema = self._make_schema_with_samples(values_by_conv, min_conversation_count=1)
        enum_vals = status_schema.get("x-polylogue-values", [])
        assert "only_here" in enum_vals, "threshold=1 should not filter single-conv values"

    def test_no_conversation_ids_skips_threshold(self) -> None:
        """When conversation_ids is None, cross-conv check is skipped entirely."""
        samples = [{"status": "solo_value"}] * 5
        stats = _collect_field_stats(samples)  # no conversation_ids
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_conversation_count=3)
        enum_vals = annotated.get("properties", {}).get("status", {}).get("x-polylogue-values", [])
        # No conversation tracking → no filtering
        assert "solo_value" in enum_vals


# =============================================================================
# Guard 2: Key denylist (expanded)
# =============================================================================

class TestKeyDenylist:
    """Fields in _CONTENT_FIELD_NAMES never yield enum annotations."""

    @pytest.mark.parametrize("field_name", [
        # Original denylist
        "text", "prompt", "summary", "query",
        # Newly added
        "body", "message", "input", "output",
    ])
    def test_content_field_is_detected(self, field_name: str) -> None:
        assert _is_content_field(f"$.{field_name}"), f"$.{field_name} should be a content field"

    @pytest.mark.parametrize("field_name", [
        "body", "message", "input", "output",
    ])
    def test_new_denylist_fields_suppress_enums(self, field_name: str) -> None:
        """New denylist fields produce no x-polylogue-values even with repeated values."""
        samples = [{field_name: "active"} for _ in range(20)]
        stats = _collect_field_stats(samples)
        schema = {"type": "object", "properties": {field_name: {"type": "string"}}}
        annotated = _annotate_schema(schema, stats)
        field_schema = annotated["properties"][field_name]
        assert "x-polylogue-values" not in field_schema, (
            f"Field '{field_name}' should suppress enum extraction"
        )

    def test_non_denylist_field_gets_enums(self) -> None:
        """A field not in the denylist does produce x-polylogue-values when repeated."""
        samples = [{"status": "active"} for _ in range(20)]
        stats = _collect_field_stats(samples)
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats)
        assert "x-polylogue-values" in annotated["properties"]["status"]


# =============================================================================
# Guard 3: Private TLD denylist
# =============================================================================

class TestPrivateTLDDenylist:
    """Internal network hostnames are rejected by _is_safe_enum_value."""

    @pytest.mark.parametrize("hostname", [
        "myhost.local",
        "printer.lan",
        "intranet.corp",
        "api.internal",
        "router.home",
        "nas.local",
        "dev-server.corp",
    ])
    def test_private_tld_hostnames_rejected(self, hostname: str) -> None:
        assert not _is_safe_enum_value(hostname), (
            f"Internal hostname {hostname!r} should be rejected"
        )

    @pytest.mark.parametrize("hostname", [
        # Should still be rejected by the existing public TLD filter
        "example.com",
        "api.openai.com",
    ])
    def test_public_tld_hostnames_still_rejected(self, hostname: str) -> None:
        assert not _is_safe_enum_value(hostname)

    def test_plain_word_without_dot_passes(self) -> None:
        """A plain word with no dot is unaffected by TLD filters."""
        assert _is_safe_enum_value("local"), "Bare word 'local' should not be rejected"
        assert _is_safe_enum_value("corp"), "Bare word 'corp' should not be rejected"


# =============================================================================
# Guard 4: Structural constants in identifier fields
# =============================================================================

class TestStructuralConstantsInIdentifierFields:
    """Identifier-named fields allow structural constants (lowercase tokens)."""

    @pytest.mark.parametrize("value", [
        "chatgpt_agent",
        "deep_research",
        "text_completion",
        "auto",
    ])
    def test_structural_constants_pass_identifier_fields(self, value: str) -> None:
        """Lowercase underscore-separated tokens pass even in identifier fields."""
        assert _is_safe_enum_value(value, path="$.notification_channel_id"), (
            f"Structural constant {value!r} should pass in identifier field"
        )

    @pytest.mark.parametrize("value", [
        "abc123DEF456ghi789",
        "A1b2C3d4E5f6G7h8I9j0",
    ])
    def test_random_ids_still_blocked_in_identifier_fields(self, value: str) -> None:
        """Random-looking values are still blocked in identifier fields."""
        assert not _is_safe_enum_value(value, path="$.user_id"), (
            f"Random ID {value!r} should be blocked in identifier field"
        )

    def test_uuid_still_blocked_in_identifier_field(self) -> None:
        """UUID-like values are still blocked (they don't match structural constant pattern)."""
        assert not _is_safe_enum_value(
            "550e8400-e29b-41d4-a716-446655440000",
            path="$.message_id",
        )


    def test_partial_tld_match_not_rejected(self) -> None:
        """Words containing TLD substrings without a preceding dot are not rejected."""
        # "locally-sourced" contains "local" but has no dot — TLD regex won't match
        assert _is_safe_enum_value("locally-sourced"), "'locally-sourced' has no dot, should not be rejected"
        # "corporate" contains "corp" but has no dot
        assert _is_safe_enum_value("corporate"), "'corporate' has no dot, should not be rejected"
        # "internal-api" contains "internal" as a substring, not a TLD
        assert _is_safe_enum_value("internal-api"), "'internal-api' has no dot, should not be rejected"
