"""Schema-drift canary for ``polylogue.sources.parsers.hermes_state``.

Pins the exact ``sessions``/``messages`` column set observed against a live
``~/.hermes/state.db`` snapshot (``SCHEMA_VERSION = 19``, hermes-agent
checkout, captured 2026-07-18) and asserts every column is either required,
mapped to a typed capability, or explicitly allowlisted as a known,
deliberately-uncaptured producer-side denormalized counter. A future Hermes
schema bump that adds a genuinely new column must fail this test loudly
instead of the column silently vanishing into no capability group and no
fidelity caveat (see the fs1.2.1/fs1.1 design's "every unsupported column
lands in the fidelity declaration, not silently dropped" acceptance
criterion).

This is a fixture-pin, not a live-database probe: it asserts about this
repo's *parser contract*, not about the operator's current ``~/.hermes``
install (which is unavailable in CI and may itself drift ahead of this pin).
Re-run ``PRAGMA table_info(sessions)`` / ``PRAGMA table_info(messages)``
against a live install and update the pinned sets below when Hermes ships a
schema bump; add a capability group in ``hermes_state.py`` for any genuinely
new evidence rather than just widening the allowlist to make this pass.
"""

from __future__ import annotations

from polylogue.sources.parsers import hermes_state

_LIVE_V19_SESSION_COLUMNS = frozenset(
    {
        "actual_cost_usd",
        "api_call_count",
        "archived",
        "billing_base_url",
        "billing_mode",
        "billing_provider",
        "cache_read_tokens",
        "cache_write_tokens",
        "chat_id",
        "chat_type",
        "compression_failure_cooldown_until",
        "compression_failure_error",
        "cost_source",
        "cost_status",
        "cwd",
        "display_name",
        "end_reason",
        "ended_at",
        "estimated_cost_usd",
        "expiry_finalized",
        "git_branch",
        "git_repo_root",
        "handoff_error",
        "handoff_platform",
        "handoff_state",
        "id",
        "input_tokens",
        "message_count",
        "model",
        "model_config",
        "origin_json",
        "output_tokens",
        "parent_session_id",
        "pricing_version",
        "reasoning_tokens",
        "rewind_count",
        "session_key",
        "source",
        "started_at",
        "system_prompt",
        "thread_id",
        "title",
        "tool_call_count",
        "user_id",
    }
)
_LIVE_V19_MESSAGE_COLUMNS = frozenset(
    {
        "active",
        "codex_message_items",
        "codex_reasoning_items",
        "compacted",
        "content",
        "finish_reason",
        "id",
        "observed",
        "platform_message_id",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "role",
        "session_id",
        "timestamp",
        "token_count",
        "tool_call_id",
        "tool_calls",
        "tool_name",
    }
)

# Columns deliberately not mapped to a capability group: read directly
# elsewhere in the parser (title), or producer-maintained denormalized
# counters that duplicate rows Polylogue already parses independently
# (message_count/tool_call_count mirror the messages table and per-message
# tool_calls JSON this parser reads row-by-row; no separate evidence lane
# would add anything a live count over the imported messages doesn't).
_SESSION_COLUMNS_HANDLED_OUTSIDE_CAPABILITY_MAP = frozenset({"title", "message_count", "tool_call_count"})


def test_every_live_session_column_is_required_capability_mapped_or_allowlisted() -> None:
    mapped: set[str] = set(hermes_state._REQUIRED_SESSION_COLUMNS)
    for fields in hermes_state._SESSION_CAPABILITIES.values():
        mapped |= fields
    mapped |= _SESSION_COLUMNS_HANDLED_OUTSIDE_CAPABILITY_MAP

    unmapped = _LIVE_V19_SESSION_COLUMNS - mapped
    assert not unmapped, (
        f"Live Hermes state.db sessions columns {sorted(unmapped)} are not covered by any "
        "capability group, _REQUIRED_SESSION_COLUMNS, or the explicit allowlist in this test "
        "-- add a capability group in hermes_state.py (see fs1.1/fs1.2.1 design) instead of "
        "leaving new evidence silently dropped."
    )


def test_every_live_message_column_is_required_or_capability_mapped() -> None:
    mapped: set[str] = set(hermes_state._REQUIRED_MESSAGE_COLUMNS)
    for fields in hermes_state._MESSAGE_CAPABILITIES.values():
        mapped |= fields

    unmapped = _LIVE_V19_MESSAGE_COLUMNS - mapped
    assert not unmapped, (
        f"Live Hermes state.db messages columns {sorted(unmapped)} are not covered by any "
        "capability group or _REQUIRED_MESSAGE_COLUMNS -- add a capability group in "
        "hermes_state.py instead of leaving new evidence silently dropped."
    )
