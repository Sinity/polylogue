"""Gemini CLI content the parser used to drop (polylogue-7yji2/rlw4h/2ow9p).

Every fixture below is synthetic but shaped from a census of the real
``~/.gemini/tmp/*/chats/*.json`` corpus (26 files, 789 tool calls):

- a tool call is ``{args, description, displayName, id, name,
  renderOutputAsMarkdown, result, resultDisplay, status, timestamp}``;
  ``result`` is a one-element list of ``{functionResponse: {id, name,
  response}}`` where ``functionResponse.id == toolCall.id`` in all 789, and
  ``response`` is ``{output}`` (766), ``{error}`` (21) or both (2);
- ``resultDisplay`` is a plain string (458), a list of ANSI cell rows (235),
  or a dict (94);
- the tool id is ``<tool>_<epoch-ms>_<n>`` and the sidecar filename is either
  that stem verbatim or ``<tool>_<id>_<slug>.txt``;
- ``displayContent`` appears on 9 user messages and differs from ``content``
  in all 9.

No operator content is reproduced: the payload strings are invented.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from polylogue.core.enums import BlockType, Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.live.gemini_tool_output_sidecars import (
    join_gemini_tool_output_sidecars,
    resolve_tool_outputs_dir,
)
from polylogue.sources.parsers.base import ParsedSession

_MASK = (
    "<tool_output_masked>\n"
    "Output too large. Showing first 8,000 and last 32,000 characters. "
    "For full output see: {path}\n"
    "Output: HEAD-EXCERPT\n"
    "...\n"
    "TAIL-EXCERPT\n"
    "</tool_output_masked>"
)


def _tool_call(tool_id: str, *, output: str, result_display: JSONValue) -> JSONDocument:
    return {
        "id": tool_id,
        "name": tool_id.rsplit("_", 2)[0],
        "displayName": "Shell",
        "description": "run a command",
        "args": {"command": "echo hi"},
        "renderOutputAsMarkdown": True,
        "status": "success",
        "timestamp": "2026-03-14T21:41:00.000Z",
        "result": [
            {
                "functionResponse": {
                    "id": tool_id,
                    "name": tool_id.rsplit("_", 2)[0],
                    "response": {"output": output},
                }
            }
        ],
        "resultDisplay": result_display,
    }


def _session(messages: Sequence[JSONValue], *, session_id: str = "sess-1") -> JSONDocument:
    return {
        "sessionId": session_id,
        "projectHash": "hash-1",
        "kind": "chat",
        "startTime": "2026-03-14T21:41:00.000Z",
        "lastUpdated": "2026-03-14T21:45:00.000Z",
        "messages": list(messages),
    }


def _tool_result_texts(session: ParsedSession) -> list[str]:
    return [
        block.text
        for message in session.messages
        for block in message.blocks
        if block.type is BlockType.TOOL_RESULT and block.text
    ]


# polylogue-7yji2


def test_masked_output_yields_to_the_full_result_display_sibling() -> None:
    """A truncation notice never wins over the full text in the same record.

    Anti-vacuity: reverting ``_fullest_tool_result_text`` to ``output or error
    or _content_text(resultDisplay)`` -- the short-circuit this bead names --
    stores the envelope and this assertion fails on both counts.
    """
    full = "LINE\n" * 400
    payload = _session(
        [
            {
                "id": "a1",
                "type": "gemini",
                "timestamp": "2026-03-14T21:41:02.000Z",
                "content": "ran it",
                "toolCalls": [
                    _tool_call(
                        "run_shell_command_1773524726450_0",
                        output=_MASK.format(path="/tmp/tool-outputs/x.txt"),
                        result_display=full,
                    )
                ],
            }
        ]
    )

    [session] = parse_payload("gemini-cli", payload, "fallback")

    assert _tool_result_texts(session) == [full]
    assert "<tool_output_masked>" not in "".join(_tool_result_texts(session))


def test_unmasked_output_is_kept_even_when_result_display_is_longer() -> None:
    """The two fields are different renderings, not a truncation pair.

    ``resultDisplay`` is the terminal rendering (ANSI cell rows in 235 of the
    corpus's tool calls); it is longer than ``output`` for 240 unmasked calls.
    Preferring the longer one wholesale would replace the model-facing text
    with a newline-joined cell dump.

    Anti-vacuity: changing the rule from "masked" to "longer wins" makes this
    assert the cell text instead of ``ok``.
    """
    cells: JSONValue = [[{"text": "ok", "fg": "white", "bold": False}] for _ in range(50)]
    payload = _session(
        [
            {
                "id": "a1",
                "type": "gemini",
                "timestamp": "2026-03-14T21:41:02.000Z",
                "content": "ran it",
                "toolCalls": [_tool_call("run_shell_command_1_0", output="ok", result_display=cells)],
            }
        ]
    )

    [session] = parse_payload("gemini-cli", payload, "fallback")

    assert _tool_result_texts(session) == ["ok"]


def test_masked_output_is_kept_when_result_display_holds_no_more() -> None:
    """A masked envelope is still better than a shorter display rendering.

    Anti-vacuity: dropping the length comparison from the rule stores the
    two-word summary and loses the retained head/tail excerpt.
    """
    masked = _MASK.format(path="/tmp/tool-outputs/x.txt")
    payload = _session(
        [
            {
                "id": "a1",
                "type": "gemini",
                "timestamp": "2026-03-14T21:41:02.000Z",
                "content": "ran it",
                "toolCalls": [_tool_call("run_shell_command_1_0", output=masked, result_display="two words")],
            }
        ]
    )

    [session] = parse_payload("gemini-cli", payload, "fallback")

    assert _tool_result_texts(session) == [masked]


# polylogue-rlw4h


def _sidecar_corpus(tmp_path: Path, *, filename: str, tool_id: str = "run_shell_command_1773524726450_0") -> Path:
    project = tmp_path / "polylogue"
    chats = project / "chats"
    chats.mkdir(parents=True)
    outputs = project / "tool-outputs" / "session-sess-1"
    outputs.mkdir(parents=True)
    (outputs / filename).write_text("FULL-SIDECAR-TEXT\n" * 500, encoding="utf-8")

    snapshot = chats / "session-2026-03-14T21-41-sess1.json"
    snapshot.write_text(
        json.dumps(
            _session(
                [
                    {
                        "id": "a1",
                        "type": "gemini",
                        "timestamp": "2026-03-14T21:41:02.000Z",
                        "content": "ran it",
                        "toolCalls": [
                            _tool_call(
                                tool_id,
                                output=_MASK.format(path=str(outputs / filename)),
                                result_display="short display",
                            )
                        ],
                    }
                ]
            )
        ),
        encoding="utf-8",
    )
    return snapshot


def test_sidecar_stem_join_recovers_the_full_tool_output(tmp_path: Path) -> None:
    """The doubled-name-plus-slug filename resolves to its tool id.

    This is the on-disk spelling for 194 of the corpus's 218 sidecars; the
    pointer inside the envelope cites a *different* spelling that Gemini CLI
    never wrote, so a pointer-only join would miss it.

    Anti-vacuity: removing the ``_{tool_id}_`` branch from
    ``_tool_id_for_stem`` classifies the file as debt and the block keeps its
    masked text.
    """
    snapshot = _sidecar_corpus(
        tmp_path,
        filename="run_shell_command_run_shell_command_1773524726450_0_keyt3f.txt",
    )

    [session] = parse_payload(
        "gemini-cli",
        json.loads(snapshot.read_text(encoding="utf-8")),
        "fallback",
        source_path=str(snapshot),
    )

    [text] = _tool_result_texts(session)
    assert text.startswith("FULL-SIDECAR-TEXT")
    assert "<tool_output_masked>" not in text
    [event] = [e for e in session.session_events if e.event_type == "gemini_cli_tool_output_sidecar"]
    assert event.payload["acquisition_status"] == "matched"
    assert event.payload["content_replaced"] is True
    assert event.timestamp is not None


def test_sidecar_with_no_citing_tool_call_is_declared_debt(tmp_path: Path) -> None:
    """A file the transcript never claims is recorded, not silently skipped.

    The directory is session-scoped, so an unresolvable file has no owner
    anywhere -- 27 of the corpus's 218 files are in this state.

    Anti-vacuity: dropping the debt branch leaves zero sidecar events and this
    unpacking raises.
    """
    snapshot = _sidecar_corpus(tmp_path, filename="grep_search_9999999999999_7_zzz.txt")

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    outputs = resolve_tool_outputs_dir(snapshot, "sess-1")
    assert outputs is not None
    result = join_gemini_tool_output_sidecars(payload, outputs)

    assert not result.matched
    [debt] = result.debt
    assert debt.reason == "no_owning_tool_call"
    assert debt.byte_size > 0


def test_absent_tool_outputs_directory_is_not_an_error(tmp_path: Path) -> None:
    """Most Gemini CLI projects have no ``tool-outputs/`` at all.

    Anti-vacuity: removing the ``is_dir`` guard raises ``FileNotFoundError``
    here instead of returning an empty join.
    """
    result = join_gemini_tool_output_sidecars(_session([]), tmp_path / "nope")
    assert not result.matched and not result.debt


# polylogue-2ow9p


def test_display_content_is_admitted_alongside_content() -> None:
    """The form the user was shown survives next to the model-facing form.

    Anti-vacuity: deleting the ``displayContent`` branch in
    ``_parse_gemini_message`` leaves one text block and this assertion fails.
    """
    payload = _session(
        [
            {
                "id": "u1",
                "type": "user",
                "timestamp": "2026-03-14T21:41:01.000Z",
                "content": ["@notes.md", "<file contents injected by the CLI>"],
                "displayContent": ["@notes.md what does this say?"],
            }
        ]
    )

    [session] = parse_payload("gemini-cli", payload, "fallback")

    [message] = session.messages
    texts = [block.text for block in message.blocks if block.type is BlockType.TEXT]
    assert "@notes.md what does this say?" in texts
    assert "<file contents injected by the CLI>" in texts
    assert session.source_name is Provider.GEMINI_CLI


def test_display_content_equal_to_content_adds_no_block() -> None:
    """Only a divergent rendering is content; an identical one is a duplicate.

    Anti-vacuity: dropping the ``!= text`` guard emits a second identical
    block and doubles the message's stored words.
    """
    payload = _session(
        [
            {
                "id": "u1",
                "type": "user",
                "timestamp": "2026-03-14T21:41:01.000Z",
                "content": "same text",
                "displayContent": "same text",
            }
        ]
    )

    [session] = parse_payload("gemini-cli", payload, "fallback")

    [message] = session.messages
    assert [block.text for block in message.blocks if block.type is BlockType.TEXT] == ["same text"]
