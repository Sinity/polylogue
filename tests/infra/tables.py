"""Canonical test data tables for provider model tests.

Single source of truth for parametrized test cases that were previously
duplicated across 3-4 test files. Import these tables instead of copying.

Each table is a list of tuples suitable for ``@pytest.mark.parametrize``.
"""

from __future__ import annotations

# =============================================================================
# Claude Code: record type → role mapping (superset of all 4 files)
#
# Production code: ClaudeCodeRecord.role property
# Bug reference: a1c085e — 328K messages had wrong role
# =============================================================================

CLAUDE_CODE_TYPE_ROLE_MAPPING: list[tuple[str, str]] = [
    # Direct mappings
    ("user", "user"),
    ("assistant", "assistant"),
    # System record types
    ("summary", "system"),
    ("system", "system"),
    ("file-history-snapshot", "system"),
    ("queue-operation", "system"),
    # Tool/progress record types
    ("progress", "tool"),
    ("result", "tool"),
    # Everything else → unknown (NOT tool — that was the original bug)
    ("init", "unknown"),
    ("custom", "unknown"),
    ("", "unknown"),
    ("some-new-type", "unknown"),
]

# =============================================================================
# Claude Code: timestamp parsing edge cases
#
# Production code: ClaudeCodeRecord.parsed_timestamp property
# Bug reference: Claude Code uses Unix milliseconds, not seconds
# =============================================================================

CLAUDE_CODE_TIMESTAMP_CASES: list[tuple[str | int | float | None, int | None, str]] = [
    (1700000000000, 2023, "unix_milliseconds"),
    (1700000000, 2023, "unix_seconds"),
    (1700000000000.5, 2023, "unix_float_milliseconds"),
    ("2025-01-01T00:00:00Z", 2025, "iso_string_with_z"),
    ("2025-06-15T12:30:00+05:00", 2025, "iso_string_with_timezone"),
    (None, None, "none_timestamp"),
    ("not-a-date", None, "invalid_string_returns_none"),
    (0, 1970, "zero_timestamp"),
    # Boundary: value just above the millisecond threshold (1e11)
    (100_000_000_001, 1973, "boundary_above_ms_threshold"),
    # Boundary: value just below the millisecond threshold
    (99_999_999_999, 5138, "boundary_below_ms_threshold_treated_as_seconds"),
]

# =============================================================================
# Claude Code: boolean flag tables
#
# Production code: is_actual_message, is_context_compaction, is_tool_progress
# =============================================================================

CLAUDE_CODE_IS_ACTUAL_MESSAGE: list[tuple[str, bool]] = [
    ("user", True),
    ("assistant", True),
    ("system", False),
    ("summary", False),
    ("progress", False),
    ("result", False),
    ("file-history-snapshot", False),
    ("init", False),
    ("", False),
]

CLAUDE_CODE_IS_CONTEXT_COMPACTION: list[tuple[str, bool]] = [
    ("summary", True),
    ("user", False),
    ("assistant", False),
    ("progress", False),
    ("system", False),
    ("init", False),
]

CLAUDE_CODE_IS_TOOL_PROGRESS: list[tuple[str, bool]] = [
    ("progress", True),
    ("user", False),
    ("assistant", False),
    ("summary", False),
    ("system", False),
    ("result", False),
]

# =============================================================================
# ChatGPT: role mapping
# =============================================================================

CHATGPT_ROLE_MAPPING: list[tuple[str, str]] = [
    ("user", "user"),
    ("assistant", "assistant"),
    ("tool", "tool"),
    ("system", "system"),
    ("custom", "unknown"),
]

# =============================================================================
# Gemini: role mapping (full set including case variants)
# =============================================================================

GEMINI_ROLE_MAPPING: list[tuple[str, str]] = [
    ("user", "user"),
    ("USER", "user"),
    ("model", "assistant"),
    ("MODEL", "assistant"),
    ("assistant", "assistant"),
    ("system", "system"),
    ("SYSTEM", "system"),
    ("unknown_role", "unknown"),
    ("", "unknown"),
]

# =============================================================================
# normalize_role: canonical input→output mapping
#
# Production code: polylogue.lib.roles.normalize_role
# (re-exported as polylogue.sources.parsers.base.normalize_role)
# =============================================================================

NORMALIZE_ROLE_CANONICAL: list[tuple[str, str, str]] = [
    # (input, expected_output, description)
    ("user", "user", "standard_user"),
    ("human", "user", "human_variant"),
    ("User", "user", "capitalized"),
    ("USER", "user", "uppercase"),
    ("assistant", "assistant", "standard_assistant"),
    ("model", "assistant", "model_variant"),
    ("ai", "assistant", "ai_variant"),
    ("Assistant", "assistant", "capitalized"),
    ("MODEL", "assistant", "uppercase_model"),
    ("system", "system", "standard_system"),
    ("System", "system", "capitalized"),
    ("SYSTEM", "system", "uppercase"),
    ("tool", "tool", "standard_tool"),
    ("function", "tool", "function_variant"),
    ("Tool", "tool", "capitalized"),
    ("custom_role", "unknown", "unknown_passthrough"),
    ("gibberish", "unknown", "gibberish"),
]

# =============================================================================
# parse_timestamp: comprehensive format table
#
# Production code: polylogue.lib.timestamps.parse_timestamp
# Tuples: (input_value, expected_year, expected_month, expected_day, expected_microsecond_or_None, description)
# When expected_year is None → result must be None
# =============================================================================

PARSE_TIMESTAMP_FORMAT_TABLE: list[tuple] = [
    # Epoch integers
    (1704067200, 2024, 1, 1, None, "epoch_int"),
    (1704067200.5, 2024, 1, 1, 500000, "epoch_float"),
    # Epoch strings
    ("1704067200", 2024, 1, 1, None, "epoch_string"),
    ("1704067200.5", 2024, 1, 1, 500000, "epoch_string_decimal"),
    # ISO 8601 variants
    ("2024-01-01T00:00:00", 2024, 1, 1, None, "iso_basic"),
    ("2024-01-01T00:00:00Z", 2024, 1, 1, None, "iso_z"),
    ("2024-01-01T00:00:00+00:00", 2024, 1, 1, None, "iso_offset"),
    ("2024-01-01T00:00:00.500000", 2024, 1, 1, 500000, "iso_microseconds"),
    # Millisecond epoch (Claude Code uses these — treated as seconds by generic parser → None)
    (1704067200000, None, None, None, None, "ms_epoch_int"),
    ("1704067200000", None, None, None, None, "ms_epoch_string"),
    # None and invalid
    (None, None, None, None, None, "none_input"),
    ("", None, None, None, None, "empty_string"),
    ("not-a-date", None, None, None, None, "garbage"),
    # Boundary: below minimum epoch threshold (treated as year, returns None)
    ("2024", None, None, None, None, "year_string"),
    ("86399", None, None, None, None, "below_threshold"),
    # Boundary: epoch threshold
    ("86400", 1970, 1, 2, None, "epoch_threshold"),
]


# =============================================================================
# format_timestamp: round-trip and formatting table
#
# Production code: polylogue.lib.timestamps.format_timestamp
# Tuples: (input, expected_prefix, description)
# expected_prefix: the ISO 8601 date prefix that must appear in the output
# =============================================================================

from datetime import datetime as _datetime, timezone as _timezone

FORMAT_TIMESTAMP_TABLE: list[tuple] = [
    (1700000000, "2023-", "epoch_int_gives_2023"),
    (_datetime(2024, 6, 15, 12, 0, 0, tzinfo=_timezone.utc), "2024-06-15", "utc_datetime"),
    (_datetime(2024, 1, 1, 0, 0, 0), "2024-01-01", "naive_datetime_treated_as_utc"),
]


# =============================================================================
# Unified role normalization: all providers in one table
#
# (provider_name, raw_role_input, expected_role_normalized, description)
# =============================================================================

UNIFIED_ROLE_NORMALIZATION: list[tuple[str, str, str, str]] = [
    # Claude Code (type → role)
    ("claude-code", "user", "user", "cc_user"),
    ("claude-code", "assistant", "assistant", "cc_assistant"),
    ("claude-code", "system", "system", "cc_system"),
    ("claude-code", "summary", "system", "cc_summary"),
    ("claude-code", "progress", "tool", "cc_progress"),
    ("claude-code", "result", "tool", "cc_result"),
    ("claude-code", "init", "unknown", "cc_init"),
    # Claude AI
    ("claude-ai", "human", "user", "cai_human"),
    ("claude-ai", "assistant", "assistant", "cai_assistant"),
    ("claude-ai", "system", "system", "cai_system"),
    ("claude-ai", "", "unknown", "cai_empty"),
    # ChatGPT
    ("chatgpt", "user", "user", "cgpt_user"),
    ("chatgpt", "assistant", "assistant", "cgpt_assistant"),
    ("chatgpt", "tool", "tool", "cgpt_tool"),
    ("chatgpt", "system", "system", "cgpt_system"),
    ("chatgpt", "custom", "unknown", "cgpt_custom"),
    # Gemini
    ("gemini", "user", "user", "gem_user"),
    ("gemini", "model", "assistant", "gem_model"),
    ("gemini", "USER", "user", "gem_user_upper"),
    ("gemini", "MODEL", "assistant", "gem_model_upper"),
    # Codex
    ("codex", "user", "user", "codex_user"),
    ("codex", "assistant", "assistant", "codex_assistant"),
]
