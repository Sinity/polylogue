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

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import BlockType
from polylogue.sources.parsers.base_models import ParsedSession, ParsedSessionEvent

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


def hook_tool_responses_from_rows(
    rows: Iterable[tuple[object, object]], *, tool_use_ids: Iterable[str]
) -> dict[str, HookToolResponse]:
    """Decode selected durable hook rows without owning their storage read."""
    wanted = {tool_use_id for tool_use_id in tool_use_ids if tool_use_id}
    recovered: dict[str, HookToolResponse] = {}
    for hook_event_id, payload_json in rows:
        check_compute_cancelled()
        if not isinstance(payload_json, (str, bytes, bytearray)):
            continue
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


def hook_tool_response_evidence_digest(rows: Iterable[tuple[object, object]]) -> str | None:
    """Digest all selected durable hook rows without retaining their payloads."""
    digest = hashlib.sha256()
    found = False
    for hook_event_id, payload_json in rows:
        check_compute_cancelled()
        found = True
        for value in (str(hook_event_id).encode("utf-8"), str(payload_json).encode("utf-8")):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest() if found else None


def read_hook_tool_responses(
    conn: sqlite3.Connection,
    *,
    origin: str,
    session_native_ids: Sequence[str],
    tool_use_ids: Iterable[str],
) -> dict[str, HookToolResponse]:
    """Read hook responses using an already-owned Source connection.

    Retained preparation uses ``PreparedSessionSourceRead`` so its selected
    rows remain inside the original publication fence. This adapter is only
    for callers that already own a connection; it never opens an archive.
    """
    wanted = tuple(dict.fromkeys(tool_id for tool_id in tool_use_ids if tool_id))
    natives = tuple(dict.fromkeys(native_id for native_id in session_native_ids if native_id))
    if not wanted or not natives:
        return {}
    placeholders = ", ".join("?" for _ in natives)
    recovered: dict[str, HookToolResponse] = {}
    for start in range(0, len(wanted), 128):
        batch = wanted[start : start + 128]
        wanted_placeholders = ", ".join("?" for _ in batch)
        cursor = conn.execute(
            f"""SELECT hook_event_id, payload_json
            FROM raw_hook_events
            WHERE origin=? AND session_native_id IN ({placeholders}) AND event_type='PostToolUse'
              AND json_valid(payload_json)
              AND json_extract(payload_json, '$.payload.tool_use_id') IN ({wanted_placeholders})
            ORDER BY observed_at_ms, hook_event_id""",
            (origin, *natives, *batch),
        )
        try:
            while page := cursor.fetchmany(256):
                recovered.update(hook_tool_responses_from_rows(page, tool_use_ids=batch))
        finally:
            cursor.close()
    return recovered


def read_hook_tool_response_evidence_digest(
    conn: sqlite3.Connection,
    *,
    origin: str,
    session_native_ids: Sequence[str],
) -> str | None:
    """Digest a session's complete PostToolUse evidence through an owned read."""
    natives = tuple(dict.fromkeys(native_id for native_id in session_native_ids if native_id))
    if not natives:
        return None
    placeholders = ", ".join("?" for _ in natives)
    cursor = conn.execute(
        f"""SELECT hook_event_id, payload_json FROM raw_hook_events
        WHERE origin=? AND session_native_id IN ({placeholders}) AND event_type='PostToolUse'
        ORDER BY hook_event_id""",
        (origin, *natives),
    )

    def selected_rows() -> Iterable[tuple[object, object]]:
        while page := cursor.fetchmany(256):
            yield from page

    try:
        return hook_tool_response_evidence_digest(selected_rows())
    finally:
        cursor.close()


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


def recover_persisted_tool_results(
    session: ParsedSession, *, responses: Mapping[str, HookToolResponse]
) -> ParsedSession:
    """Apply responses already read through the caller's owned Source window."""
    truncations = unresolved_persisted_truncations(session)
    if not truncations:
        return session
    return apply_hook_tool_responses(session, truncations, responses)


__all__ = [
    "BASH_STDOUT_CAP_CHARS",
    "HOOK_TOOL_RESPONSE_EVENT_TYPE",
    "HookToolResponse",
    "PersistedTruncation",
    "apply_hook_tool_responses",
    "hook_response_text",
    "hook_tool_responses_from_rows",
    "read_hook_tool_responses",
    "recover_persisted_tool_results",
    "unresolved_persisted_truncations",
]
