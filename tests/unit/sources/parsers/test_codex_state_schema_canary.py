"""Schema-drift canary for ``polylogue.sources.parsers.codex_state``.

Pins the exact ``threads``/``thread_spawn_edges`` column sets observed
against a live ``~/.codex/state_5.sqlite`` (captured 2026-07-29, 3,054
threads / 1,030 spawn edges) and asserts every column is either read by
``parse_codex_state_db`` or explicitly allowlisted as deliberately
unconsumed, with a reason. A future Codex release that adds a genuinely new
column must fail this test loudly instead of the column silently landing in
neither camp -- add a typed field to ``CodexThreadRecord``/``CodexSpawnEdge``
for any genuinely new evidence rather than widening the allowlist to make
this pass.

This is a fixture-pin, not a live-database probe: it asserts about this
repo's parser contract, not about the operator's current ``~/.codex``
install (unavailable in CI, and may itself drift ahead of this pin). Re-run
``PRAGMA table_info(threads)`` against a live install and update the pinned
sets below when Codex ships a schema bump.
"""

from __future__ import annotations

_LIVE_THREADS_COLUMNS = frozenset(
    {
        "id",
        "rollout_path",
        "created_at",
        "updated_at",
        "source",
        "model_provider",
        "cwd",
        "title",
        "sandbox_policy",
        "approval_mode",
        "tokens_used",
        "has_user_event",
        "archived",
        "archived_at",
        "git_sha",
        "git_branch",
        "git_origin_url",
        "cli_version",
        "first_user_message",
        "agent_nickname",
        "agent_role",
        "memory_mode",
        "model",
        "reasoning_effort",
        "agent_path",
        "created_at_ms",
        "updated_at_ms",
        "thread_source",
        "preview",
        "recency_at",
        "recency_at_ms",
        "history_mode",
        "name",
    }
)

_LIVE_THREAD_SPAWN_EDGES_COLUMNS = frozenset({"parent_thread_id", "child_thread_id", "status"})

# Columns ``CodexThreadRecord`` reads directly (see codex_state.py's SELECT).
_THREADS_COLUMNS_READ = frozenset(
    {
        "id",
        "title",
        "cwd",
        "created_at_ms",
        "updated_at_ms",
        "source",
        "model",
        "agent_nickname",
        "agent_role",
        "archived",
    }
)

# Columns deliberately not surfaced by this module, with reasons:
_THREADS_COLUMNS_ALLOWLISTED = frozenset(
    {
        # Superseded by the *_ms columns this parser reads (seconds vs. millis
        # dual-write; Codex keeps both for backward compat).
        "created_at",
        "updated_at",
        # Cross-referenced against the rollout path polylogue already has
        # from JSONL discovery; not additional evidence.
        "rollout_path",
        # Codex-side execution/session config, not session-content evidence.
        "model_provider",
        "sandbox_policy",
        "approval_mode",
        "memory_mode",
        "cli_version",
        "agent_path",
        "thread_source",
        "history_mode",
        # Denormalized counters/flags that duplicate content already parsed
        # independently from the JSONL rollout (usage tables, message
        # presence) -- no separate evidence lane would add anything a count
        # over the already-ingested session doesn't.
        "tokens_used",
        "has_user_event",
        "archived_at",
        # Git/environment provenance already captured per-session from the
        # JSONL rollout's own session_meta by parsers/codex.py.
        "git_sha",
        "git_branch",
        "git_origin_url",
        # Candidate future title-source evidence (assembly_codex.py /
        # polylogue-ih67's resolution ladder already reads this table's
        # `title` column; `first_user_message`/`preview`/`name` are further
        # candidate title lanes not wired by this acquisition-only change).
        "first_user_message",
        "preview",
        "name",
        # Recency bookkeeping Codex's own UI uses for sort order; not
        # session evidence.
        "recency_at",
        "recency_at_ms",
        # Reasoning-effort/model config already reflected in per-turn JSONL
        # evidence; the thread-level column is a coarse summary of it.
        "reasoning_effort",
    }
)

_SPAWN_EDGES_COLUMNS_READ = frozenset({"parent_thread_id", "child_thread_id", "status"})


def test_every_live_threads_column_is_read_or_allowlisted() -> None:
    covered = _THREADS_COLUMNS_READ | _THREADS_COLUMNS_ALLOWLISTED
    unmapped = _LIVE_THREADS_COLUMNS - covered
    assert not unmapped, (
        f"Live Codex state_5.sqlite threads columns {sorted(unmapped)} are not read by "
        "parse_codex_state_db nor allowlisted in this canary -- add a typed field to "
        "CodexThreadRecord (codex_state.py) for genuinely new evidence instead of leaving it "
        "silently dropped."
    )
    # And the inverse: nothing pinned here should be stale relative to the parser/allowlist.
    unexpected = covered - _LIVE_THREADS_COLUMNS
    assert not unexpected, f"Canary references columns not observed on the live schema: {sorted(unexpected)}"


def test_every_live_spawn_edge_column_is_read() -> None:
    unmapped = _LIVE_THREAD_SPAWN_EDGES_COLUMNS - _SPAWN_EDGES_COLUMNS_READ
    assert not unmapped, f"Live thread_spawn_edges columns not read: {sorted(unmapped)}"
