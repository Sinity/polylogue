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
