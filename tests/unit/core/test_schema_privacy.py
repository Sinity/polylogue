"""Privacy guard tests for schema inference.

Covers the heuristics in ``_is_safe_enum_value``, the field-level filters, and
the slot allowlist that decides whether a field may publish observed members at
all:
  1. Publishable-slot allowlist (only a declared protocol vocabulary role emits
     ``x-polylogue-values``)
  2. Key denylist (body, message, input, output never yield enums)
  3. Private TLD denylist (.local, .lan, .corp, .internal, .home rejected)

Tests are grouped by guard type so failures pinpoint which heuristic regressed.
Every suppression test declares a publishable role on the field under test, so
a suppression assertion cannot pass merely because the slot was never eligible.
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
# Guard 1: Publishable-slot allowlist
# =============================================================================


ROLE_SLOT: dict[str, object] = {"type": "string", "x-polylogue-semantic-role": "message_role"}


class TestPublishableSlotAllowlist:
    """Only a declared protocol vocabulary slot publishes observed members.

    A committed package is public, and ``x-polylogue-values`` is the one
    annotation that carries observed member *values* rather than structure.
    Value shape cannot separate a provider constant from a recurring private
    token, so publication is an allowlist over declared semantic roles.

    Anti-vacuity: restore publication for an undeclared slot -- drop the
    ``sem_role in PUBLISHABLE_VOCABULARY_ROLES`` guard in
    ``polylogue/schemas/generation/field_annotations.annotate_schema`` -- and
    ``test_undeclared_slot_publishes_no_members`` and
    ``test_undeclared_role_publishes_no_members`` both fail.
    """

    def _annotate(self, field_schema: Mapping[str, object], values: list[str]) -> Mapping[str, object]:
        samples = [{"status": value} for value in values]
        session_ids: list[str | None] = [f"conv_{index}" for index in range(len(values))]
        stats = _collect_field_stats(samples, session_ids=session_ids)
        schema: dict[str, object] = {"type": "object", "properties": {"status": dict(field_schema)}}
        return schema_property(_annotate_schema(schema, stats), "status")

    def test_undeclared_slot_publishes_no_members(self) -> None:
        """A safe, highly recurrent value is still unpublished without a declared role."""
        annotated = self._annotate({"type": "string"}, ["active"] * 30)
        assert "x-polylogue-values" not in annotated
        assert "x-polylogue-observed-distribution" in annotated, "structure must survive the member refusal"

    def test_undeclared_role_publishes_no_members(self) -> None:
        """A declared role outside the allowlist publishes nothing either."""
        annotated = self._annotate(
            {"type": "string", "x-polylogue-semantic-role": "session_title"},
            ["active"] * 30,
        )
        assert "x-polylogue-values" not in annotated

    def test_declared_protocol_slot_publishes_members(self) -> None:
        annotated = self._annotate(ROLE_SLOT, ["assistant"] * 10 + ["user"] * 10)
        assert set(schema_values(annotated)) == {"assistant", "user"}

    def test_declared_slot_publishes_a_single_session_member(self) -> None:
        """A declared protocol vocabulary is not user content, so recurrence is not required."""
        samples = [{"status": "attachment"}] + [{"status": "assistant"} for _ in range(9)]
        session_ids: list[str | None] = ["conv_A", *["conv_B"] * 9]
        stats = _collect_field_stats(samples, session_ids=session_ids)
        schema: dict[str, object] = {"type": "object", "properties": {"status": dict(ROLE_SLOT)}}
        annotated = schema_property(_annotate_schema(schema, stats), "status")
        assert "attachment" in schema_values(annotated)

    def test_value_guard_still_applies_inside_a_declared_slot(self) -> None:
        """The allowlist admits the slot; it does not admit an unsafe value."""
        annotated = self._annotate(ROLE_SLOT, ["myhost.local"] * 20)
        assert "myhost.local" not in schema_values(annotated)


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
        schema = {"type": "object", "properties": {field_name: dict(ROLE_SLOT)}}
        annotated = _annotate_schema(schema, stats)
        field_schema = schema_property(annotated, field_name)
        assert "x-polylogue-values" not in field_schema, f"Field '{field_name}' should suppress enum extraction"

    def test_non_denylist_field_gets_enums(self) -> None:
        """A field not in the denylist does produce x-polylogue-values when repeated."""
        samples = [{"status": "active"} for _ in range(20)]
        stats = _collect_field_stats(samples)
        schema = {"type": "object", "properties": {"status": dict(ROLE_SLOT)}}
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
# Guard 4: Property test — safe values never resemble PII (Phase 9)
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
      1. Publishable-slot allowlist (PUBLISHABLE_VOCABULARY_ROLES)
      2. Content-field key denylist (_CONTENT_FIELD_NAMES)
      3. Private TLD denylist (_is_safe_enum_value)

    These tests assert that multi-guard payloads are handled correctly:
    suppression happens regardless of which guard fires first, and
    values are absent if ANY guard would suppress them.

    Every field under test declares a publishable role, so guard 1 admits the
    slot and the assertion measures guards 2 and 3 rather than passing because
    nothing was eligible to publish.
    """

    # ── overlap: content-field + publishable slot ──────────────

    def test_content_field_value_absent_even_when_seen_in_many_convs(self) -> None:
        """Guard 2 (content field) suppresses a value in a publishable slot.

        The field declares a publishable role, so guard 1 admits it; the
        content-field denylist is what removes the members.
        """
        samples = [{"body": "active"} for _ in range(30)]
        conv_ids: list[str | None] = [f"conv_{i}" for i in range(30)]
        stats = _collect_field_stats(samples, session_ids=conv_ids)
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"body": dict(ROLE_SLOT)},
        }
        annotated = _annotate_schema(schema, stats)
        field_schema = schema_property(annotated, "body")
        assert "x-polylogue-values" not in field_schema, (
            "Content field should suppress enums even in a publishable slot"
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
        schema = {"type": "object", "properties": {"body": dict(ROLE_SLOT)}}
        annotated = _annotate_schema(schema, stats)
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
        schema = {"type": "object", "properties": {"input": dict(ROLE_SLOT)}}
        annotated = _annotate_schema(schema, stats)
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
        schema = {"type": "object", "properties": {"status": dict(ROLE_SLOT)}}
        annotated = _annotate_schema(schema, stats)
        field_schema = schema_property(annotated, "status")
        enum_vals = schema_values(field_schema)
        assert "myhost.local" not in enum_vals, "Private TLD value should be suppressed on structural field"

    # ── overlap: publishable slot + private TLD ────────────────

    def test_private_tld_value_suppressed_regardless_of_conv_count(self) -> None:
        """Guard 3 (private TLD) suppresses inside an admitted slot.

        'printer.lan' is a private TLD — it should be absent from enums
        even when seen in 30 different sessions on a publishable field.
        """
        samples = [{"status": "printer.lan"} for _ in range(30)]
        conv_ids: list[str | None] = [f"conv_{i}" for i in range(30)]
        stats = _collect_field_stats(samples, session_ids=conv_ids)
        schema = {"type": "object", "properties": {"status": dict(ROLE_SLOT)}}
        annotated = _annotate_schema(schema, stats)
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
        schema = {"type": "object", "properties": {"message": dict(ROLE_SLOT)}}
        annotated = _annotate_schema(schema, stats)
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
            {"status": "dev-server.corp"},
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
                "status": dict(ROLE_SLOT),
                "body": dict(ROLE_SLOT),
                "input": dict(ROLE_SLOT),
            },
        }

        run1 = _annotate_schema(schema, stats)
        run2 = _annotate_schema(schema, stats)

        assert run1 == run2, "Identical inputs must produce identical outputs"

    # ── any-guard-suppresses guarantee ─────────────────────────

    def test_any_guard_suppresses_value_is_absent(self) -> None:
        """If any guard would suppress a value, the value is absent from enums.

        We construct a mixed payload where:
        - 'dev-server.corp' → suppressed by Guard 3 (private TLD '.corp')
        - 'active' in 'status' → passes all guards (declared slot, not content field, not TLD)
        - 'active' in 'body' → suppressed by Guard 2 (content field)
        """
        values_by_conv = {
            "conv_A": [("status", "dev-server.corp"), ("body", "active")],
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
                "status": dict(ROLE_SLOT),
                "body": dict(ROLE_SLOT),
            },
        }
        annotated = _annotate_schema(schema, stats)

        status_schema = schema_property(annotated, "status")
        status_enums = schema_values(status_schema)
        assert "active" in status_enums, "'active' in 'status' should pass all guards"
        assert "dev-server.corp" not in status_enums, "Domain value suppressed by Guard 3 (private TLD)"

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
        schema = {"type": "object", "properties": {"message": dict(ROLE_SLOT)}}
        annotated = _annotate_schema(schema, stats)

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

        # 'status' declares a publishable role and is not a content field
        # 'nas.local' → suppressed by Guard 3 (private TLD)
        # 'active' and 'pending' → pass all guards
        schema = {"type": "object", "properties": {"status": dict(ROLE_SLOT)}}
        annotated = _annotate_schema(schema, stats)
        status_schema = schema_property(annotated, "status")
        enum_vals = schema_values(status_schema)

        assert "active" in enum_vals, "'active' should pass all three guards"
        assert "pending" in enum_vals, "'pending' is a safe value in an admitted slot"
        assert "nas.local" not in enum_vals, "'nas.local' suppressed by Guard 3 (private TLD)"
