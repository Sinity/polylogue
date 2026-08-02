"""Claude Code ``~/.claude/todos/*.json`` plan-snapshot parsing (polylogue-t0p).

Claude Code persists the agent's current TODO/plan list to
``~/.claude/todos/<session-id>[-agent-<agent-id>].json`` -- a directory sibling
to ``~/.claude/projects/`` (the session transcript root), not nested under it
-- and overwrites the same path wholesale on every ``TodoWrite`` tool call
(never appended). Each file is a bare JSON array of task objects
(``content``/``status``/``priority``/``id``), unlike every other Claude Code
sidecar this package parses (those are JSONL streams or single JSON *objects*
under ``workflows/``/``jobs/``). See the ``todo_snapshot`` ``OriginArtifactRule``
in ``polylogue.sources.origin_specs`` for the admission/fidelity declaration
this module implements.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

#: Claude Code names todo files by the owning session's UUID, optionally
#: suffixed with the delegated subagent's own UUID (``agent-*.jsonl``
#: transcripts follow the identical ``-agent-<uuid>`` convention elsewhere in
#: this package, e.g. ``orchestration.py:_agent_id_from_path``).
_FILENAME_RE = re.compile(
    r"^(?P<session_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"(?:-agent-(?P<agent_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}))?$"
)


def session_and_agent_id_from_filename(source_path: str | Path) -> tuple[str | None, str | None]:
    """Recover ``(session_id, agent_id)`` from a todo snapshot's filename stem.

    Returns ``(None, None)`` when the stem doesn't match Claude Code's own
    naming convention (e.g. a hand-placed test fixture with a non-UUID name);
    callers should treat that as "cannot link this snapshot to a session",
    not fabricate an identity.
    """
    stem = Path(source_path).stem
    match = _FILENAME_RE.match(stem)
    if not match:
        return None, None
    return match.group("session_id"), match.group("agent_id")


@dataclass(frozen=True, slots=True)
class ClaudeTodoItem:
    """One task in a Claude Code plan snapshot, in the agent's own list order."""

    position: int
    content: str
    status: str
    item_id: str | None
    priority: str | None


@dataclass(frozen=True, slots=True)
class ClaudeTodoSnapshot:
    """One parsed ``todos/*.json`` plan snapshot."""

    source_path: str
    session_id: str | None
    agent_id: str | None
    items: tuple[ClaudeTodoItem, ...]
    parse_error: str | None = None

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def completed_count(self) -> int:
        return sum(1 for item in self.items if item.status == "completed")

    @property
    def completion_rate(self) -> float | None:
        """Fraction of items with ``status == "completed"``, or ``None`` for an empty plan.

        ``None`` (not ``0.0``) for zero items: an empty plan has no
        denominator to divide by, and reporting a bare ``0.0`` would read as
        "nothing done" rather than "nothing planned" -- the honest-analytics
        null-policy convention this codebase applies elsewhere (see
        ``insights/measurement/metric.py``'s ``NullPolicy``).
        """
        if not self.items:
            return None
        return self.completed_count / len(self.items)


def parse_claude_todo_artifact(source_path: str, payload: bytes | str | object) -> ClaudeTodoSnapshot | None:
    """Parse one ``todo_snapshot`` artifact's raw payload.

    Returns ``None`` only when ``source_path`` doesn't even look like a
    todos-directory file (defensive: callers should already have matched the
    ``OriginArtifactRule`` before reaching here). A malformed or
    unexpectedly-shaped payload still returns a ``ClaudeTodoSnapshot`` with
    ``parse_error`` set and ``items=()`` -- the session linkage recovered from
    the filename remains available even when the content itself didn't parse.
    """
    session_id, agent_id = session_and_agent_id_from_filename(source_path)
    try:
        loaded = _decode(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        return ClaudeTodoSnapshot(source_path, session_id, agent_id, (), parse_error=f"{type(exc).__name__}: {exc}")

    if not isinstance(loaded, list):
        return ClaudeTodoSnapshot(
            source_path,
            session_id,
            agent_id,
            (),
            parse_error=f"expected a JSON array, got {type(loaded).__name__}",
        )

    items: list[ClaudeTodoItem] = []
    for position, entry in enumerate(loaded):
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        status = entry.get("status")
        if not isinstance(content, str) or not isinstance(status, str):
            continue
        item_id = entry.get("id")
        priority = entry.get("priority")
        items.append(
            ClaudeTodoItem(
                position=position,
                content=content,
                status=status,
                item_id=item_id if isinstance(item_id, str) else None,
                priority=priority if isinstance(priority, str) else None,
            )
        )
    return ClaudeTodoSnapshot(source_path, session_id, agent_id, tuple(items))


def _decode(payload: bytes | str | object) -> object:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        return json.loads(payload)
    return payload


__all__ = [
    "ClaudeTodoItem",
    "ClaudeTodoSnapshot",
    "parse_claude_todo_artifact",
    "session_and_agent_id_from_filename",
]
