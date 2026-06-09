"""Shared blackboard note model and body codec (#1697).

The blackboard is a persistent, agent-addressable note surface stored in
``user.db`` (``blackboard_notes``). A note's structured fields — kind, title,
content, and an optional repo scope — are encoded into the single ``body`` text
column so the storage schema stays simple. This module owns that encoding so
the CLI, the archive store, the Python API, and the MCP server all agree on one
representation instead of each re-deriving it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Closed kind vocabulary. Adding a kind is a deliberate code change so surfaces
# (CLI choices, MCP enums, filters) stay in lockstep.
BLACKBOARD_KINDS: tuple[str, ...] = (
    "finding",
    "blocker",
    "decision",
    "handoff",
    "question",
    "observation",
)

# Kinds that represent open work — surfaced by the ``unresolved`` filter.
UNRESOLVED_KINDS: frozenset[str] = frozenset({"blocker", "question"})

_KIND_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


@dataclass(frozen=True, slots=True)
class BlackboardNote:
    """A blackboard note with its body decoded into structured fields."""

    note_id: str
    kind: str
    title: str
    content: str
    scope_repo: str | None
    target_type: str | None
    target_id: str | None
    created_at_ms: int
    updated_at_ms: int


def build_blackboard_body(
    *,
    kind: str,
    title: str,
    content: str,
    scope_repo: str | None = None,
    scope_issue: int | None = None,
    scope_path: str | None = None,
    related_sessions: tuple[str, ...] = (),
) -> str:
    """Encode structured note fields into the stored ``body`` text."""
    lines = [f"[{kind}] {title}".strip(), "", content]
    scope_lines: list[str] = []
    if scope_repo:
        scope_lines.append(f"scope_repo: {scope_repo}")
    if scope_issue is not None:
        scope_lines.append(f"scope_issue: {scope_issue}")
    if scope_path:
        scope_lines.append(f"scope_path: {scope_path}")
    if related_sessions:
        scope_lines.append(f"related_sessions: {', '.join(related_sessions)}")
    if scope_lines:
        lines.extend(["", *scope_lines])
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ParsedBlackboardBody:
    kind: str
    title: str
    content: str
    scope_repo: str | None


def parse_blackboard_body(body: str) -> ParsedBlackboardBody:
    """Decode a stored ``body`` back into structured fields.

    Unrecognized bodies (no ``[kind]`` prefix) fall back to ``observation`` with
    the whole first line as the title, so legacy or hand-written notes never
    raise.
    """
    first, _, rest = body.partition("\n")
    match = _KIND_RE.match(first)
    kind = match.group(1) if match else "observation"
    title = match.group(2) if match else first
    content_lines: list[str] = []
    scope_repo: str | None = None
    for line in rest.strip().splitlines():
        if line.startswith("scope_repo: "):
            scope_repo = line.removeprefix("scope_repo: ").strip()
            continue
        if line.startswith(("scope_issue: ", "scope_path: ", "related_sessions: ")):
            continue
        content_lines.append(line)
    content = "\n".join(content_lines).strip()
    return ParsedBlackboardBody(kind=kind, title=title, content=content, scope_repo=scope_repo)


def decode_blackboard_note(
    *,
    note_id: str,
    body: str,
    target_type: str | None,
    target_id: str | None,
    created_at_ms: int,
    updated_at_ms: int,
) -> BlackboardNote:
    """Build a structured :class:`BlackboardNote` from stored-row primitives.

    Takes primitives (not a storage envelope) so this module stays free of any
    storage import — the facade passes the row fields straight through.
    """
    parsed = parse_blackboard_body(body)
    return BlackboardNote(
        note_id=note_id,
        kind=parsed.kind,
        title=parsed.title,
        content=parsed.content,
        scope_repo=parsed.scope_repo,
        target_type=target_type,
        target_id=target_id,
        created_at_ms=created_at_ms,
        updated_at_ms=updated_at_ms,
    )


__all__ = [
    "BLACKBOARD_KINDS",
    "UNRESOLVED_KINDS",
    "BlackboardNote",
    "ParsedBlackboardBody",
    "build_blackboard_body",
    "decode_blackboard_note",
    "parse_blackboard_body",
]
