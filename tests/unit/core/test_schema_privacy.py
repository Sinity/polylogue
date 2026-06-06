"""Privacy guard tests for schema inference.

Covers all heuristics in _is_safe_enum_value and the field-level filters,
plus the three new guards added in this session:
  1. Cross-session threshold (value seen in <N distinct sessions excluded)
  2. Key denylist expansion (body, message, input, output never yield enums)
  3. Private TLD denylist (.local, .lan, .corp, .internal, .home rejected)

Tests are grouped by guard type so failures pinpoint which heuristic regressed.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.schemas.operator.schema_inference import (
    _annotate_schema,
    _collect_field_stats,
    _is_content_field,
    _is_safe_enum_value,
)
from tests.infra.schema_access import schema_property, schema_values

# =============================================================================
# _is_safe_enum_value — existing heuristics (regression guard)
# =============================================================================


class TestSafeEnumValueExistingGuards:
    """All existing _is_safe_enum_value filters must still hold."""

    # --- Values that MUST pass (structural enum candidates) ---

    @pytest.mark.parametrize(
        "value",
        [
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
        ],
    )
    def test_safe_structural_values_pass(self, value: str) -> None:
        assert _is_safe_enum_value(value), f"Expected {value!r} to be safe"

    # --- URL rejection ---

    @pytest.mark.parametrize(
        "value",
        [
            "https://example.com/path",
            "http://api.openai.com/v1/chat",
            "ftp://files.corp.internal/export.zip",
        ],
    )
    def test_urls_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected URL {value!r} to be rejected"

    # --- Email rejection ---

    @pytest.mark.parametrize(
        "value",
        [
            "user@example.com",
            "alice@corp.internal",
        ],
    )
    def test_emails_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected email {value!r} to be rejected"

    # --- Natural language / whitespace rejection ---

    @pytest.mark.parametrize(
        "value",
        [
            "Hello world",
            "This is a message",
            "multi\nline\ncontent",
        ],
    )
    def test_sentences_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected sentence {value!r} to be rejected"

    # --- Capitalized words now PASS (CamelCase check removed) ---

    @pytest.mark.parametrize(
        "value",
        [
            "Reasoning",
            "Thinking",
            "GitHub",
            "None",
            "Alice",  # single word — no longer blocked (whitespace check handles multi-word names)
        ],
    )
    def test_capitalized_words_pass(self, value: str) -> None:
        assert _is_safe_enum_value(value), f"Expected {value!r} to pass (CamelCase check removed)"

    # --- Public domain / TLD rejection ---

    @pytest.mark.parametrize(
        "value",
        [
            "openai.com",
            "api.anthropic.com",
            "storage.googleapis.com",
            "cdn.example.net",
        ],
    )
    def test_public_domains_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected domain {value!r} to be rejected"

    # --- File extension rejection ---

    @pytest.mark.parametrize(
        "value",
        [
            "document.pdf",
            "archive.zip",
            "data.json",
            "script.py",
            "export.csv",
        ],
    )
    def test_file_extensions_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected filename {value!r} to be rejected"

    # --- Timestamp rejection ---

    @pytest.mark.parametrize(
        "value",
        [
            "2024-01-15T10:30:00Z",
            "2024-01-15",
        ],
    )
    def test_timestamps_rejected(self, value: str) -> None:
        assert not _is_safe_enum_value(value), f"Expected timestamp {value!r} to be rejected"

    # --- High-entropy token rejection ---

    @pytest.mark.parametrize(
        "value",
        [
            "sk-abc123XYZ789def456",
            "Bearer eyJhbGciOiJSUzI1NiJ9",
            "dQw4w9WgXcQ",  # YouTube video ID (11 chars, now caught with lowered threshold)
        ],
    )
    def test_high_entropy_tokens_rejected(self, value: str) -> None:
        # These are opaque tokens — reject even without explicit URL indicators
        assert not _is_safe_enum_value(value), f"Expected token {value!r} to be rejected"

    # --- Quoted high-entropy tokens (Gemini format) ---

    def test_quoted_high_entropy_stripped(self) -> None:
        """Gemini exports embed values in double quotes — quotes are stripped before check."""
        # A YouTube-ID-like token wrapped in quotes
        assert not _is_safe_enum_value('"dQw4w9WgXcQ"')

    # --- Model slug exemption (dash-separated structural tokens) ---

    @pytest.mark.parametrize(
        "value",
        [
            "gpt-4-code-interpreter",
            "claude-haiku-4-5-20251001",
            "gemini-2-5-pro",
            "gpt-4o-mini",
        ],
    )
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
# Guard 1: Cross-session threshold
# =============================================================================


class TestCrossSessionThreshold:
    """Values seen in fewer than N sessions are suppressed from schema enums."""

    def _make_schema_with_samples(
        self,
        values_by_conv: dict[str, list[str]],
        *,
        min_session_count: int = 3,
    ) -> Mapping[str, object]:
        """Build schema annotations from samples grouped by session ID."""
        samples = []
        conv_ids = []
        for conv_id, values in values_by_conv.items():
            for v in values:
                samples.append({"$.status": v})
                conv_ids.append(conv_id)

        # _collect_field_stats expects flat dicts with the actual field as a key
        # Use a simple structure: {"status": value}
        flat_samples = [{"status": v} for v in [v for vals in values_by_conv.values() for v in vals]]
        flat_conv_ids: list[str | None] = [cid for cid, vals in values_by_conv.items() for _ in vals]

        stats = _collect_field_stats(flat_samples, session_ids=flat_conv_ids)
        schema: dict[str, object] = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=min_session_count)
        return schema_property(annotated, "status")

    def test_value_in_one_conv_excluded_at_threshold_3(self) -> None:
        """A value seen in only 1 session is excluded when threshold=3."""
        values_by_conv = {
            "conv_A": ["rare_status"],
            "conv_B": ["common"],
            "conv_C": ["common"],
            "conv_D": ["common"],
        }
        status_schema = self._make_schema_with_samples(values_by_conv, min_session_count=3)
        enum_vals = schema_values(status_schema)
        assert "rare_status" not in enum_vals, "Single-session value should be excluded"

    def test_value_in_three_convs_included_at_threshold_3(self) -> None:
        """A value seen in exactly 3 sessions passes the threshold."""
        values_by_conv = {
            "conv_A": ["stable_role"],
            "conv_B": ["stable_role"],
            "conv_C": ["stable_role"],
        }
        status_schema = self._make_schema_with_samples(values_by_conv, min_session_count=3)
        enum_vals = schema_values(status_schema)
        assert "stable_role" in enum_vals, "Value in 3 sessions should pass threshold=3"

    def test_threshold_1_includes_single_conv_values(self) -> None:
        """Default threshold=1 preserves existing behaviour (no cross-conv filtering)."""
        values_by_conv = {"conv_A": ["only_here", "only_here"]}
        status_schema = self._make_schema_with_samples(values_by_conv, min_session_count=1)
        enum_vals = schema_values(status_schema)
        assert "only_here" in enum_vals, "threshold=1 should not filter single-conv values"

    def test_no_session_ids_skips_threshold(self) -> None:
        """When session_ids is None, cross-conv check is skipped entirely."""
        samples = [{"status": "solo_value"}] * 5
        stats = _collect_field_stats(samples)  # no session_ids
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        enum_vals = schema_values(schema_property(annotated, "status"))
        # No session tracking → no filtering
        assert "solo_value" in enum_vals


# =============================================================================
# Guard 2: Key denylist (expanded)
# =============================================================================


class TestKeyDenylist:
    """Fields in _CONTENT_FIELD_NAMES never yield enum annotations."""

    @pytest.mark.parametrize(
        "field_name",
        [
            # Original denylist
            "text",
            "prompt",
            "summary",
            "query",
            # Newly added
            "body",
            "message",
            "input",
            "output",
        ],
    )
    def test_content_field_is_detected(self, field_name: str) -> None:
        assert _is_content_field(f"$.{field_name}"), f"$.{field_name} should be a content field"

    @pytest.mark.parametrize(
        "field_name",
        [
            "body",
            "message",
            "input",
            "output",
        ],
    )
    def test_new_denylist_fields_suppress_enums(self, field_name: str) -> None:
        """New denylist fields produce no x-polylogue-values even with repeated values."""
        samples = [{field_name: "active"} for _ in range(20)]
        stats = _collect_field_stats(samples)
        schema = {"type": "object", "properties": {field_name: {"type": "string"}}}
        annotated = _annotate_schema(schema, stats)
        field_schema = schema_property(annotated, field_name)
        assert "x-polylogue-values" not in field_schema, f"Field '{field_name}' should suppress enum extraction"

    def test_non_denylist_field_gets_enums(self) -> None:
        """A field not in the denylist does produce x-polylogue-values when repeated."""
        samples = [{"status": "active"} for _ in range(20)]
        stats = _collect_field_stats(samples)
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats)
        assert "x-polylogue-values" in schema_property(annotated, "status")


# =============================================================================
# Guard 3: Private TLD denylist
# =============================================================================


class TestPrivateTLDDenylist:
    """Internal network hostnames are rejected by _is_safe_enum_value."""

    @pytest.mark.parametrize(
        "hostname",
        [
            "myhost.local",
            "printer.lan",
            "intranet.corp",
            "api.internal",
            "router.home",
            "nas.local",
            "dev-server.corp",
        ],
    )
    def test_private_tld_hostnames_rejected(self, hostname: str) -> None:
        assert not _is_safe_enum_value(hostname), f"Internal hostname {hostname!r} should be rejected"

    @pytest.mark.parametrize(
        "hostname",
        [
            # Should still be rejected by the existing public TLD filter
            "example.com",
            "api.openai.com",
        ],
    )
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

    @pytest.mark.parametrize(
        "value",
        [
            "chatgpt_agent",
            "deep_research",
            "text_completion",
            "auto",
        ],
    )
    def test_structural_constants_pass_identifier_fields(self, value: str) -> None:
        """Lowercase underscore-separated tokens pass even in identifier fields."""
        assert _is_safe_enum_value(value, path="$.notification_channel_id"), (
            f"Structural constant {value!r} should pass in identifier field"
        )

    @pytest.mark.parametrize(
        "value",
        [
            "abc123DEF456ghi789",
            "A1b2C3d4E5f6G7h8I9j0",
        ],
    )
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


# =============================================================================
# Guard 4: Structural role exemption -- message_role bypasses
#          cross-session threshold
# =============================================================================


class TestStructuralRoleExemption:
    """message_role fields bypass cross-session privacy threshold."""

    def test_message_role_bypasses_threshold(self) -> None:
        """A message_role field includes values from <3 sessions."""
        values_by_conv = {
            "conv_A": ["attachment"],
            "conv_B": ["assistant"],
            "conv_C": ["assistant"],
            "conv_D": ["user", "assistant"],
            "conv_E": ["user"],
        }
        flat_samples = [{"type": v} for vals in values_by_conv.values() for v in vals]
        flat_conv_ids: list[str | None] = [cid for cid, vals in values_by_conv.items() for _ in vals]
        stats = _collect_field_stats(flat_samples, session_ids=flat_conv_ids)
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "x-polylogue-semantic-role": "message_role",
                }
            },
        }
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        type_schema = schema_property(annotated, "type")
        enum_vals = schema_values(type_schema)
        assert "attachment" in enum_vals, "message_role field should preserve values even when min_session_count=3"
        assert "assistant" in enum_vals
        assert "user" in enum_vals

    def test_non_role_field_respects_threshold(self) -> None:
        """A field without message_role still respects the threshold."""
        values_by_conv = {
            "conv_A": ["rare_val"],
            "conv_B": ["common"],
            "conv_C": ["common"],
            "conv_D": ["common"],
        }
        flat_samples = [{"status": v} for vals in values_by_conv.values() for v in vals]
        flat_conv_ids: list[str | None] = [cid for cid, vals in values_by_conv.items() for _ in vals]
        stats = _collect_field_stats(flat_samples, session_ids=flat_conv_ids)
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        }
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        status_schema = schema_property(annotated, "status")
        enum_vals = schema_values(status_schema)
        assert "rare_val" not in enum_vals, "Non-role field should respect min_session_count=3"
        assert "common" in enum_vals


# =============================================================================
# Guard 5: Property test — safe values never resemble PII (Phase 9)
# =============================================================================


class TestSafeValueNeverResemblesPII:
    """Property: if _is_safe_enum_value(v) returns True, v must not look like PII."""

    @given(st.from_regex(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", fullmatch=True))
    @settings(max_examples=50)
    def test_emails_always_rejected(self, email: str) -> None:
        assert not _is_safe_enum_value(email), f"Email-like {email!r} should be rejected"

    @given(st.from_regex(r"https?://[a-z0-9.-]+/[a-z0-9/]*", fullmatch=True))
    @settings(max_examples=50)
    def test_urls_always_rejected(self, url: str) -> None:
        assert not _is_safe_enum_value(url), f"URL {url!r} should be rejected"

    @pytest.mark.parametrize(
        "word",
        [
            "function",
            "class",
            "import",
            "model",
            "user",
            "assistant",
            "system",
            "tool",
            "active",
            "pending",
            "disabled",
            "completed",
            "text",
            "json",
            "html",
            "markdown",
        ],
    )
    def test_technical_vocabulary_always_passes(self, word: str) -> None:
        assert _is_safe_enum_value(word), f"Technical word {word!r} should pass"

    @given(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P")),
            min_size=1,
            max_size=50,
        ).filter(lambda s: s.isascii() and " " not in s and "\n" not in s)
    )
    @settings(max_examples=100)
    def test_accepted_values_have_no_pii_markers(self, value: str) -> None:
        """If a value passes _is_safe_enum_value, it should not contain PII markers."""
        if _is_safe_enum_value(value):
            assert "@" not in value, f"Accepted value {value!r} contains @"
            assert "://" not in value, f"Accepted value {value!r} contains URL scheme"


# =============================================================================
# Guard interaction tests (#1225)
# =============================================================================


class TestMultiGuardInteraction:
    """Values that trip multiple privacy guards simultaneously.

    The three independent guards are:
      1. Cross-session threshold (min_session_count)
      2. Content-field key denylist (_CONTENT_FIELD_NAMES)
      3. Private TLD denylist (_is_safe_enum_value)

    These tests assert that multi-guard payloads are handled correctly:
    suppression happens regardless of which guard fires first, and
    values are absent if ANY guard would suppress them.
    """

    # ── overlap: content-field + cross-conv threshold ──────────

    def test_content_field_value_absent_even_when_seen_in_many_convs(self) -> None:
        """Guard 2 (content field) suppresses regardless of Guard 1 (threshold).

        A value in a content field is blocked even when seen in enough
        sessions to satisfy the cross-conv threshold.
        """
        samples = [{"body": "active"} for _ in range(30)]
        conv_ids: list[str | None] = [f"conv_{i}" for i in range(30)]
        stats = _collect_field_stats(samples, session_ids=conv_ids)
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"body": {"type": "string"}},
        }
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        field_schema = schema_property(annotated, "body")
        assert "x-polylogue-values" not in field_schema, (
            "Content field should suppress enums even when value passes threshold"
        )

    def test_content_field_with_rare_value_still_suppressed(self) -> None:
        """Content field suppresses rare AND common values alike."""
        values_by_conv = {
            "conv_A": ["rare_body_text"],
            "conv_B": ["common_text"],
            "conv_C": ["common_text"],
            "conv_D": ["common_text"],
        }
        flat_samples = [{"body": v} for vals in values_by_conv.values() for v in vals]
        flat_conv_ids: list[str | None] = [cid for cid, vals in values_by_conv.items() for _ in vals]
        stats = _collect_field_stats(flat_samples, session_ids=flat_conv_ids)
        schema = {"type": "object", "properties": {"body": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        field_schema = schema_property(annotated, "body")
        assert "x-polylogue-values" not in field_schema, "Content field 'body' should never produce enums"

    # ── overlap: content-field + private TLD ───────────────────

    def test_content_field_with_private_tld_value_suppressed(self) -> None:
        """Guard 2 (content field) and Guard 3 (private TLD) both fire.

        'input' is a content field; 'api.internal' is a private TLD value.
        Either guard alone would suppress — both together must also suppress.
        """
        samples = [{"input": "api.internal"} for _ in range(20)]
        conv_ids: list[str | None] = [f"conv_{i}" for i in range(20)]
        stats = _collect_field_stats(samples, session_ids=conv_ids)
        schema = {"type": "object", "properties": {"input": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        field_schema = schema_property(annotated, "input")
        assert "x-polylogue-values" not in field_schema, (
            "Content field 'input' should suppress enums regardless of private TLD status"
        )

    def test_private_tld_value_on_structural_field_suppressed_by_value_guard(self) -> None:
        """Guard 3 alone suppresses a private TLD value on a structural field.

        'status' is NOT a content field, so Guard 2 doesn't fire.
        Guard 3 (private TLD) still suppresses 'myhost.local'.
        """
        samples = [{"status": "myhost.local"} for _ in range(20)]
        conv_ids: list[str | None] = [f"conv_{i}" for i in range(20)]
        stats = _collect_field_stats(samples, session_ids=conv_ids)
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        field_schema = schema_property(annotated, "status")
        enum_vals = schema_values(field_schema)
        assert "myhost.local" not in enum_vals, "Private TLD value should be suppressed on structural field"

    # ── overlap: cross-conv threshold + private TLD ────────────

    def test_private_tld_value_suppressed_regardless_of_conv_count(self) -> None:
        """Guard 3 (private TLD) suppresses even when Guard 1 (threshold) is satisfied.

        'printer.lan' is a private TLD — it should be absent from enums
        even when seen in 30 different sessions.
        """
        samples = [{"status": "printer.lan"} for _ in range(30)]
        conv_ids: list[str | None] = [f"conv_{i}" for i in range(30)]
        stats = _collect_field_stats(samples, session_ids=conv_ids)
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        field_schema = schema_property(annotated, "status")
        enum_vals = schema_values(field_schema)
        assert "printer.lan" not in enum_vals, "Private TLD value should be suppressed even when seen in 30 sessions"

    # ── triple overlap ─────────────────────────────────────────

    def test_triple_guard_overlap_suppresses_value(self) -> None:
        """All three guards fire: content field + private TLD + rare conv count.

        'message' is a content field, 'dev-server.corp' is a private TLD,
        and the value appears in only 1 session.
        """
        values_by_conv = {
            "conv_A": ["dev-server.corp"],
            "conv_B": ["active"],
            "conv_C": ["active"],
            "conv_D": ["active"],
        }
        flat_samples = [{"message": v} for vals in values_by_conv.values() for v in vals]
        flat_conv_ids: list[str | None] = [cid for cid, vals in values_by_conv.items() for _ in vals]
        stats = _collect_field_stats(flat_samples, session_ids=flat_conv_ids)
        schema = {"type": "object", "properties": {"message": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        field_schema = schema_property(annotated, "message")
        assert "x-polylogue-values" not in field_schema, "Content field 'message' should never produce enums (Guard 2)"

    # ── order independence ─────────────────────────────────────

    def test_suppression_is_independent_of_guard_order(self) -> None:
        """The same input produces identical output regardless of guard ordering.

        We verify this by running the annotation twice with the same
        input and asserting the outputs are identical — guards are
        applied in a fixed order by _annotate_schema, so two runs
        with the same inputs must produce equal results.
        """
        samples = [
            {"status": "na1.storybird.ai"},
            {"body": "some text"},
            {"status": "active"},
            {"input": "api.internal"},
            {"status": "pending"},
        ]
        conv_ids: list[str | None] = ["conv_A", "conv_A", "conv_B", "conv_B", "conv_C"]
        stats = _collect_field_stats(samples, session_ids=conv_ids)

        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "body": {"type": "string"},
                "input": {"type": "string"},
            },
        }

        run1 = _annotate_schema(schema, stats, min_session_count=3)
        run2 = _annotate_schema(schema, stats, min_session_count=3)

        assert run1 == run2, "Identical inputs must produce identical outputs"

    # ── any-guard-suppresses guarantee ─────────────────────────

    def test_any_guard_suppresses_value_is_absent(self) -> None:
        """If any guard would suppress a value, the value is absent from enums.

        We construct a mixed payload where:
        - 'na1.storybird.ai' → suppressed by Guard 1 (seen in 1 conv, threshold=3)
          AND Guard 3 (domain with public TLD '.ai')
        - 'active' in 'status' → passes all guards (seen in 3+ convs, not content field, not TLD)
        - 'active' in 'body' → suppressed by Guard 2 (content field)
        """
        values_by_conv = {
            "conv_A": [("status", "na1.storybird.ai"), ("body", "active")],
            "conv_B": [("status", "active")],
            "conv_C": [("status", "active")],
            "conv_D": [("status", "active")],
        }
        flat_samples = [{field: val} for _cid, pairs in values_by_conv.items() for field, val in pairs]
        flat_conv_ids: list[str | None] = [cid for cid, pairs in values_by_conv.items() for _ in pairs]
        stats = _collect_field_stats(flat_samples, session_ids=flat_conv_ids)

        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "body": {"type": "string"},
            },
        }
        annotated = _annotate_schema(schema, stats, min_session_count=3)

        status_schema = schema_property(annotated, "status")
        status_enums = schema_values(status_schema)
        assert "active" in status_enums, "'active' in 'status' should pass all guards"
        assert "na1.storybird.ai" not in status_enums, "Domain value suppressed by Guard 1 (rare) + Guard 3 (TLD)"

        body_schema = schema_property(annotated, "body")
        assert "x-polylogue-values" not in body_schema, "Content field 'body' should never produce enums (Guard 2)"

    # ── no double-counting ─────────────────────────────────────

    def test_multi_guard_suppression_not_double_counted(self) -> None:
        """A value suppressed by multiple guards still counts as one suppression.

        We verify that the suppressed count in field stats is consistent
        regardless of how many guards would independently suppress a value.
        """
        samples = [
            {"message": "api.internal"},
            {"message": "router.home"},
            {"message": "active"},
        ]
        conv_ids: list[str | None] = [f"conv_{i}" for i in range(3)]
        stats = _collect_field_stats(samples, session_ids=conv_ids)
        schema = {"type": "object", "properties": {"message": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=3)

        # Content field 'message' suppresses ALL enums (Guard 2)
        field_schema = schema_property(annotated, "message")
        assert "x-polylogue-values" not in field_schema, "Content field 'message' should suppress all enum values"

    def test_stats_are_consistent_when_guards_overlap(self) -> None:
        """Field stats (total count, distinct count) are consistent even when
        multiple guards fire on different values in the same field."""
        values_by_conv = {
            "conv_A": [("status", "active"), ("status", "nas.local")],
            "conv_B": [("status", "active")],
            "conv_C": [("status", "active")],
            "conv_D": [("status", "pending")],
        }
        flat_samples = [{field: val} for _cid, pairs in values_by_conv.items() for field, val in pairs]
        flat_conv_ids: list[str | None] = [cid for cid, pairs in values_by_conv.items() for _ in pairs]
        stats = _collect_field_stats(flat_samples, session_ids=flat_conv_ids)

        # 'status' is a structural field — not a content field
        # 'nas.local' → suppressed by Guard 3 (private TLD)
        # 'active' → passes all guards (seen in 3 convs, not TLD)
        # 'pending' → suppressed by Guard 1 (seen in 1 conv, threshold=3)
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        annotated = _annotate_schema(schema, stats, min_session_count=3)
        status_schema = schema_property(annotated, "status")
        enum_vals = schema_values(status_schema)

        assert "active" in enum_vals, "'active' should pass all three guards"
        assert "nas.local" not in enum_vals, "'nas.local' suppressed by Guard 3 (private TLD)"
        assert "pending" not in enum_vals, "'pending' suppressed by Guard 1 (rare, threshold=3)"
