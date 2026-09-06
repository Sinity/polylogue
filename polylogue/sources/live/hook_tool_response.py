"""Third-fallback recovery of a truncated ``tool_result`` from hook evidence.

A Claude Code tool call whose output overflows the inline transcript envelope
keeps only a short preview plus a pointer into the session's ``tool-results/``
directory; ``tool_result_sidecars`` restores the full text from that file. The
file is not always reachable: a session whose project slug changes mid-run
(the process ``cwd`` moved) writes its sidecars under the old slug's
directory, and once that directory is removed the pointer resolves to nothing
and the block stays truncated. The recovery order is therefore inline text,
then sidecar file, then -- here -- the durable ``PostToolUse`` hook envelope,
which carries the same call's ``tool_response`` under its ``tool_use_id``.

The hook's copy is not always whole. Claude Code hands ``Bash`` a
``tool_response.stdout`` capped at :data:`BASH_STDOUT_CAP_CHARS` characters
together with the same ``persistedOutputPath``/``persistedOutputSize`` pointer
it wrote into the transcript, so recovery widens the preview without
completing it; responses that carry their whole result inline (WebFetch, MCP
results) recover in full. ``recovery_complete`` on the emitted session event
distinguishes the two, and a recovery that would not add text is not
performed.

Recovery is confined to blocks the sidecar join left truncated. Deriving every
captured ``tool_response`` would duplicate content the transcript already
retains in full for all but a handful of calls.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import BlockType
from polylogue.logging import get_logger
from polylogue.sources.parsers.base_models import ParsedSession, ParsedSessionEvent

logger = get_logger(__name__)

#: The event family this module appends to a recovered session's timeline.
HOOK_TOOL_RESPONSE_EVENT_TYPE = "hook_tool_response_recovery"

#: Claude Code truncates a ``Bash`` result's ``stdout`` to this many characters
#: before handing it to a hook, whatever the persisted output's real size.
BASH_STDOUT_CAP_CHARS = 30000

#: A truncation envelope always opens the block. Matching the pointer only in
#: the head keeps a tool result that merely *quotes* a transcript (an agent
#: reading a ``.jsonl``) from being read as truncated itself. The opener is
#: checked against a short prefix first so a full rebuild does not copy a
#: head-sized slice of every tool result in the archive.
_ENVELOPE_OPENER_CHARS = 64
_ENVELOPE_HEAD_CHARS = 4096
_ENVELOPE_OPENERS = ("<persisted-output>", "Error: result (")
_POINTER_RE = re.compile(r"(?:Full output saved to|Output has been saved to):?\s*(\S+)")

#: ``tool_response`` shapes this module knows how to read. Anything else is
#: left alone rather than guessed at: substituting the wrong field of a
#: structured response would replace an honest preview with unrelated text.
_STDOUT_KEY = "stdout"
_STDERR_KEY = "stderr"
_PERSISTED_SIZE_KEY = "persistedOutputSize"
_WHOLE_RESULT_KEYS = ("result", "content", "output", "text")


@dataclass(frozen=True)
class PersistedTruncation:
    """A ``tool_result`` block still carrying an unresolved overflow pointer."""

    tool_use_id: str
    pointer: str
    inline_chars: int


@dataclass(frozen=True)
class HookToolResponse:
    """Text recovered from one ``PostToolUse`` envelope, and how whole it is."""

    tool_use_id: str
    hook_event_id: str
    text: str
    #: Size the provider reported for the full output, when it reported one.
    full_size: int | None

    @property
    def complete(self) -> bool:
        return self.full_size is None or len(self.text) >= self.full_size


def unresolved_persisted_truncations(session: ParsedSession) -> tuple[PersistedTruncation, ...]:
    """Return the session's ``tool_result`` blocks the sidecar join left truncated.

    A block that the sidecar join resolved carries the sidecar's own bytes and
    no longer opens with an envelope, so this is exactly the residue.
    """
    found: dict[str, PersistedTruncation] = {}
    for message in session.messages:
        for block in message.blocks:
            if block.type is not BlockType.TOOL_RESULT or not block.tool_id or not block.text:
                continue
            if block.tool_id in found:
                continue
            if not block.text[:_ENVELOPE_OPENER_CHARS].lstrip().startswith(_ENVELOPE_OPENERS):
                continue
            pointer = _POINTER_RE.search(block.text[:_ENVELOPE_HEAD_CHARS])
            if pointer is None:
                continue
            found[block.tool_id] = PersistedTruncation(
                tool_use_id=block.tool_id,
                pointer=pointer.group(1).rstrip("."),
                inline_chars=len(block.text),
            )
    return tuple(found.values())


def hook_response_text(tool_response: object) -> tuple[str, int | None] | None:
    """Read ``(text, reported_full_size)`` out of one hook ``tool_response``."""
    if isinstance(tool_response, str):
        return (tool_response, None) if tool_response else None
    if not isinstance(tool_response, Mapping):
        return None
    stdout = tool_response.get(_STDOUT_KEY)
    if isinstance(stdout, str):
        stderr = tool_response.get(_STDERR_KEY)
        text = f"{stdout}\n{stderr}" if isinstance(stderr, str) and stderr else stdout
        full_size = tool_response.get(_PERSISTED_SIZE_KEY)
        return (text, full_size if isinstance(full_size, int) else None) if text else None
    for key in _WHOLE_RESULT_KEYS:
        value = tool_response.get(key)
        if isinstance(value, str) and value:
            return value, None
    return None


def read_hook_tool_responses(
    conn: sqlite3.Connection,
    *,
    origin: str,
    session_native_ids: Sequence[str],
    tool_use_ids: Iterable[str],
) -> dict[str, HookToolResponse]:
    """Read ``PostToolUse`` responses for ``tool_use_ids`` out of ``source.db``.

    A subagent's tool calls are journalled under the parent session's id, so
    callers pass every native id the call could have been recorded against.
    """
    wanted = {tool_use_id for tool_use_id in tool_use_ids if tool_use_id}
    natives = [native_id for native_id in dict.fromkeys(session_native_ids) if native_id]
    if not wanted or not natives:
        return {}
    has_hook_spool = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_hook_events'"
    ).fetchone()
    if has_hook_spool is None:
        # An index-only harness, or a source tier predating the hook spool.
        return {}
    placeholders = ", ".join("?" for _ in natives)
    rows = conn.execute(
        f"""
        SELECT hook_event_id, payload_json
        FROM raw_hook_events
        WHERE origin = ?
          AND session_native_id IN ({placeholders})
          AND event_type = 'PostToolUse'
        ORDER BY observed_at_ms
        """,
        (origin, *natives),
    ).fetchall()
    recovered: dict[str, HookToolResponse] = {}
    for hook_event_id, payload_json in rows:
        try:
            payload = json.loads(payload_json).get("payload")
        except (TypeError, ValueError, AttributeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        tool_use_id = payload.get("tool_use_id")
        if not isinstance(tool_use_id, str) or tool_use_id not in wanted:
            continue
        read = hook_response_text(payload.get("tool_response"))
        if read is None:
            continue
        text, full_size = read
        recovered[tool_use_id] = HookToolResponse(
            tool_use_id=tool_use_id,
            hook_event_id=str(hook_event_id),
            text=text,
            full_size=full_size,
        )
    return recovered


def resolve_hook_tool_responses(
    archive_root: Path,
    *,
    origin: str,
    session_native_ids: Sequence[str],
    tool_use_ids: Iterable[str],
) -> dict[str, HookToolResponse]:
    """``read_hook_tool_responses`` against the archive's own ``source.db``.

    Read-only so it is safe beside the daemon's single writer; an absent or
    unreadable source tier degrades to no evidence.
    """
    source_db = archive_root / "source.db"
    if not source_db.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error as exc:
        logger.debug("Failed to open source.db for hook tool responses: %s", exc)
        return {}
    try:
        return read_hook_tool_responses(
            conn,
            origin=origin,
            session_native_ids=session_native_ids,
            tool_use_ids=tool_use_ids,
        )
    except sqlite3.Error as exc:
        logger.debug("Failed to read hook tool responses: %s", exc)
        return {}
    finally:
        conn.close()


def apply_hook_tool_responses(
    session: ParsedSession,
    truncations: Sequence[PersistedTruncation],
    responses: Mapping[str, HookToolResponse],
) -> ParsedSession:
    """Substitute recovered text into its owning block and record the recovery.

    Never adds or removes a message or block. A response that would not
    lengthen the retained preview is recorded as ``matched`` but not applied,
    so the block keeps the text the provider itself put in the transcript.
    """
    if not truncations:
        return session
    replacements: dict[str, str] = {}
    events: list[ParsedSessionEvent] = []
    for truncation in truncations:
        response = responses.get(truncation.tool_use_id)
        if response is None:
            events.append(
                ParsedSessionEvent(
                    event_type=HOOK_TOOL_RESPONSE_EVENT_TYPE,
                    payload={
                        "acquisition_status": "absent",
                        "tool_use_id": truncation.tool_use_id,
                        "pointer": truncation.pointer,
                        "inline_chars": truncation.inline_chars,
                    },
                )
            )
            continue
        applied = len(response.text) > truncation.inline_chars
        if applied:
            replacements[truncation.tool_use_id] = response.text
        events.append(
            ParsedSessionEvent(
                event_type=HOOK_TOOL_RESPONSE_EVENT_TYPE,
                payload={
                    "acquisition_status": "matched",
                    "tool_use_id": truncation.tool_use_id,
                    "pointer": truncation.pointer,
                    "inline_chars": truncation.inline_chars,
                    "recovered_chars": len(response.text),
                    "recovery_complete": response.complete,
                    "reported_full_size": response.full_size,
                    "hook_event_id": response.hook_event_id,
                    "content_replaced": applied,
                },
            )
        )
    if not events:
        return session
    messages = session.messages
    if replacements:
        updated = []
        for message in messages:
            if not any(
                block.type is BlockType.TOOL_RESULT and block.tool_id in replacements for block in message.blocks
            ):
                updated.append(message)
                continue
            blocks = [
                block.model_copy(update={"text": replacements[block.tool_id]})
                if block.type is BlockType.TOOL_RESULT and block.tool_id in replacements
                else block
                for block in message.blocks
            ]
            updated.append(message.model_copy(update={"blocks": blocks}))
        messages = updated
    return session.model_copy(update={"messages": messages, "session_events": [*session.session_events, *events]})


def recover_persisted_tool_results(session: ParsedSession, *, archive_root: Path) -> ParsedSession:
    """Apply hook recovery to whatever the sidecar join left truncated.

    A no-op -- with no source-tier read at all -- for a session that carries no
    unresolved overflow pointer, which is every session but a handful.
    """
    truncations = unresolved_persisted_truncations(session)
    if not truncations:
        return session
    from polylogue.core.sources import origin_from_provider

    native_id = str(session.provider_session_id or "")
    if not native_id:
        return session
    responses = resolve_hook_tool_responses(
        archive_root,
        origin=origin_from_provider(session.source_name).value,
        # A subagent transcript's native id is ``<parent uuid>:<agent id>``,
        # while its hook envelopes are journalled under the parent uuid alone.
        session_native_ids=(native_id, native_id.split(":", 1)[0]),
        tool_use_ids=(truncation.tool_use_id for truncation in truncations),
    )
    return apply_hook_tool_responses(session, truncations, responses)


__all__ = [
    "BASH_STDOUT_CAP_CHARS",
    "HOOK_TOOL_RESPONSE_EVENT_TYPE",
    "HookToolResponse",
    "PersistedTruncation",
    "apply_hook_tool_responses",
    "hook_response_text",
    "read_hook_tool_responses",
    "recover_persisted_tool_results",
    "resolve_hook_tool_responses",
    "unresolved_persisted_truncations",
]
