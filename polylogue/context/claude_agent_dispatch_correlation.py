"""Correlate Claude Code hook ``agent_id``/``agent_type`` against archived tool calls.

Claude Code stamps every ``PreToolUse``/``PostToolUse`` hook payload made by a
dispatched subagent with ``agent_id`` (the agent instance) and ``agent_type``
(the agent definition), keyed to the call by ``tool_use_id``. That is the
runtime's own per-call assertion of which agent instance ran a call --
evidence transcript topology cannot reconstruct, because a subagent's
``subagents/agent-*.jsonl`` transcript proves only that *some* child ran, not
which instance owns any one call.

Read-only bridge over two tiers, mirroring
``context.codex_spawn_edge_correlation``: the durable ``raw_hook_events``
spool in source.db and the ingested ``blocks`` tree in index.db. Nothing is
mutated -- the assertion is derived on read from the durable payload, which
already retains ``agent_id``/``agent_type`` verbatim.

The resolved block's session is reported alongside the session the hook fired
in: they differ exactly when the dispatched agent's calls were archived under
its own child session, which is the lineage the assertion carries.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.core.enums import Origin
from polylogue.core.hook_payload import hook_record_field
from polylogue.storage.sqlite.archive_tiers.source_write import list_hook_events

#: Hook event types whose payload can carry a dispatched agent's identity.
_AGENT_BEARING_EVENTS = frozenset({"PreToolUse", "PostToolUse"})

#: SQLite parameter batch for the ``tool_id`` resolution query.
_RESOLVE_CHUNK = 500


@dataclass(frozen=True, slots=True)
class ClaudeAgentDispatchAssertion:
    """One tool call the runtime attributed to a named agent instance."""

    agent_id: str
    agent_type: str | None
    tool_use_id: str
    hook_session_native_id: str | None
    observed_at_ms: int
    tool_use_block_id: str
    message_id: str
    resolved_session_id: str


@dataclass(frozen=True, slots=True)
class ClaudeAgentDispatchCorrelation:
    """Archive-wide reconciliation of hook-asserted agent identity against tool calls."""

    total_agent_bearing_events: int
    distinct_agent_ids: int
    resolved: tuple[ClaudeAgentDispatchAssertion, ...]
    unresolved_tool_use_ids: tuple[str, ...]
    contradicted_tool_use_ids: tuple[str, ...]

    @property
    def resolved_count(self) -> int:
        return len(self.resolved)


@dataclass(frozen=True, slots=True)
class _AgentClaim:
    agent_id: str
    agent_type: str | None
    hook_session_native_id: str | None
    observed_at_ms: int


def _agent_claims(source_conn: sqlite3.Connection) -> tuple[dict[str, _AgentClaim], set[str], int]:
    """Return ``{tool_use_id: claim}``, contradicted ids, and the events read.

    ``PreToolUse`` and ``PostToolUse`` assert the same pair for one call; a
    second, *different* ``agent_id`` for the same ``tool_use_id`` is a
    contradiction and neither claim is kept.
    """
    claims: dict[str, _AgentClaim] = {}
    contradicted: set[str] = set()
    seen = 0
    for event in list_hook_events(source_conn, origin=Origin.CLAUDE_CODE_SESSION):
        if event.event_type not in _AGENT_BEARING_EVENTS:
            continue
        # ``event.payload`` is the spooled envelope for spool-drained events and
        # the bare harness payload for evidence written directly; reading
        # envelope-before-payload covers both without branching on the producer.
        agent_id = hook_record_field(event.payload, "agent_id")
        tool_use_id = hook_record_field(event.payload, "tool_use_id")
        if not isinstance(agent_id, str) or not isinstance(tool_use_id, str):
            continue
        seen += 1
        agent_type = hook_record_field(event.payload, "agent_type")
        claim = _AgentClaim(
            agent_id=agent_id,
            agent_type=agent_type if isinstance(agent_type, str) else None,
            hook_session_native_id=event.session_native_id,
            observed_at_ms=event.observed_at_ms,
        )
        existing = claims.get(tool_use_id)
        if existing is None:
            claims[tool_use_id] = claim
        elif existing.agent_id != claim.agent_id:
            contradicted.add(tool_use_id)
    for tool_use_id in contradicted:
        claims.pop(tool_use_id, None)
    return claims, contradicted, seen


def _tool_use_blocks(index_conn: sqlite3.Connection, tool_use_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    """Return ``{tool_id: (block_id, message_id, session_id)}`` for Claude Code tool calls.

    Lineage replay can put one ``tool_id`` in both a parent's replayed prefix
    and its child's tail, so the lowest ``block_id`` wins: the assertion binds
    to one deterministic block, and ``resolved_session_id`` names which.
    """
    resolved: dict[str, tuple[str, str, str]] = {}
    for start in range(0, len(tool_use_ids), _RESOLVE_CHUNK):
        chunk = tool_use_ids[start : start + _RESOLVE_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = index_conn.execute(
            f"""
            SELECT b.tool_id, b.block_id, b.message_id, b.session_id
            FROM blocks AS b
            JOIN sessions AS s ON s.session_id = b.session_id
            WHERE b.block_type = 'tool_use'
              AND s.origin = ?
              AND b.tool_id IN ({placeholders})
            ORDER BY b.block_id
            """,
            (Origin.CLAUDE_CODE_SESSION.value, *chunk),
        ).fetchall()
        # Row-factory agnostic (positional indices): callers may pass a plain
        # tuple-factory connection, not necessarily one with sqlite3.Row set.
        for row in rows:
            resolved.setdefault(str(row[0]), (str(row[1]), str(row[2]), str(row[3])))
    return resolved


def correlate_claude_agent_dispatches(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
) -> ClaudeAgentDispatchCorrelation:
    """Resolve hook-asserted agent identity onto the tool-call blocks it names.

    ``source_conn`` reads the durable hook spool (``raw_hook_events``,
    source.db); ``index_conn`` reads the ingested block tree (index.db).
    Neither is mutated. An asserted ``tool_use_id`` with no archived
    ``tool_use`` block is reported unresolved rather than dropped -- the hook
    can outrun ingest, and the difference between "not yet ingested" and
    "never read" is the whole point of this seam.
    """
    claims, contradicted, seen = _agent_claims(source_conn)
    blocks = _tool_use_blocks(index_conn, sorted(claims))
    resolved: list[ClaudeAgentDispatchAssertion] = []
    unresolved: list[str] = []
    for tool_use_id in sorted(claims):
        claim = claims[tool_use_id]
        block = blocks.get(tool_use_id)
        if block is None:
            unresolved.append(tool_use_id)
            continue
        block_id, message_id, session_id = block
        resolved.append(
            ClaudeAgentDispatchAssertion(
                agent_id=claim.agent_id,
                agent_type=claim.agent_type,
                tool_use_id=tool_use_id,
                hook_session_native_id=claim.hook_session_native_id,
                observed_at_ms=claim.observed_at_ms,
                tool_use_block_id=block_id,
                message_id=message_id,
                resolved_session_id=session_id,
            )
        )
    return ClaudeAgentDispatchCorrelation(
        total_agent_bearing_events=seen,
        distinct_agent_ids=len({claim.agent_id for claim in claims.values()}),
        resolved=tuple(resolved),
        unresolved_tool_use_ids=tuple(unresolved),
        contradicted_tool_use_ids=tuple(sorted(contradicted)),
    )


__all__ = [
    "ClaudeAgentDispatchAssertion",
    "ClaudeAgentDispatchCorrelation",
    "correlate_claude_agent_dispatches",
]
