"""Provider-shaped routes from a raw record to the public tool-outcome triple.

Every case here starts at a provider's own wire record and ends at the stored
block, the ``actions`` projection and the hydrated public envelope, so a
producer that stops deriving a structural reason cannot be rescued by a
default further down. The triple under test is
``(tool_outcome, tool_result_is_error, tool_result_outcome_unknown_reason)``.

Anti-vacuity for the whole file: delete the reason derivation at any producer
and the record either refuses at :class:`ParsedContentBlock` construction or
lands with the wrong reason, and each case asserts the reason it expects
rather than merely "some unknown". Replace ``unknown_reason`` with a constant
``not_reported`` and the ``unsupported_construct`` / ``source_truncated``
cases fail.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Origin, Provider, ToolOutcome, ToolResultUnknownReason
from polylogue.core.sources import origin_from_provider
from polylogue.sources.dispatch import detect_provider
from polylogue.sources.origin_specs import origin_specs, tool_outcome_unknown_reasons_for_origin
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.chatgpt import looks_like as chatgpt_looks_like
from polylogue.sources.parsers.chatgpt import parse as parse_chatgpt
from polylogue.sources.parsers.claude.code_parser import parse_code
from polylogue.sources.parsers.codex import looks_like as codex_looks_like
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.sources.parsers.drive import looks_like as drive_looks_like
from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.sources.parsers.local_agent import (
    looks_like_gemini_cli,
    looks_like_hermes,
    parse_gemini_cli,
    parse_hermes,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    read_archive_session_envelope,
    write_parsed_session_to_archive,
)

NOT_REPORTED = ToolResultUnknownReason.NOT_REPORTED.value
DISTRUSTED = ToolResultUnknownReason.DISTRUSTED.value
UNSUPPORTED = ToolResultUnknownReason.UNSUPPORTED_CONSTRUCT.value
TRUNCATED = ToolResultUnknownReason.SOURCE_TRUNCATED.value

Triple = tuple[str, int | None, str | None]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _stored_triples(conn: sqlite3.Connection, session_id: str) -> list[Triple]:
    rows = conn.execute(
        """
        SELECT b.tool_outcome, b.tool_result_is_error, b.tool_result_outcome_unknown_reason
        FROM blocks b JOIN messages m ON m.message_id = b.message_id
        WHERE b.session_id = ? AND b.block_type = 'tool_result'
        ORDER BY m.position, m.variant_index, b.position
        """,
        (session_id,),
    ).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


def _envelope_triples(conn: sqlite3.Connection, session_id: str) -> list[Triple]:
    envelope = read_archive_session_envelope(conn, session_id)
    return [
        (
            block.tool_outcome.value if block.tool_outcome is not None else "",
            None if block.tool_result_is_error is None else int(block.tool_result_is_error),
            block.tool_result_outcome_unknown_reason,
        )
        for message in envelope.messages
        for block in message.blocks
        if block.block_type == BlockType.TOOL_RESULT.value
    ]


def _action_states(conn: sqlite3.Connection, session_id: str) -> list[tuple[str | None, str | None]]:
    rows = conn.execute(
        "SELECT result_state, outcome_unknown_reason FROM actions WHERE session_id = ? ORDER BY tool_use_block_id",
        (session_id,),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _write(conn: sqlite3.Connection, session: ParsedSession) -> str:
    return write_parsed_session_to_archive(conn, session)


# --------------------------------------------------------------------------
# Provider wire records
# --------------------------------------------------------------------------


def _claude_code_records(result_segment: dict[str, Any], *, extra: dict[str, Any] | None = None) -> list[Any]:
    return [
        {
            "type": "assistant",
            "uuid": "assistant-1",
            "sessionId": "cc-outcome",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call-1", "name": "Bash", "input": {}}],
            },
        },
        {
            "type": "user",
            "uuid": "user-1",
            "sessionId": "cc-outcome",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call-1", **result_segment}],
            },
            **(extra or {}),
        },
    ]


def _codex_records(output: Any) -> list[dict[str, Any]]:
    return [
        {"type": "session_meta", "payload": {"id": "codex-outcome"}},
        {
            "type": "response_item",
            "payload": {"type": "function_call", "call_id": "call-1", "name": "Bash", "arguments": "{}"},
        },
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "call-1", "output": output},
        },
    ]


def _chatgpt_payload(node_status: str | None) -> dict[str, Any]:
    output_message: dict[str, Any] = {
        "id": "out1",
        "author": {"role": "tool"},
        "content": {"content_type": "execution_output", "text": "3"},
        "create_time": 1_704_067_203.0,
    }
    if node_status is not None:
        output_message["status"] = node_status
    return {
        "id": "chatgpt-outcome",
        "conversation_id": "chatgpt-outcome",
        "title": "outcome",
        "create_time": 1_704_067_200.0,
        "current_node": "out1",
        "mapping": {
            "code1": {
                "id": "code1",
                "parent": None,
                "children": ["out1"],
                "message": {
                    "id": "code1",
                    "author": {"role": "assistant"},
                    "recipient": "python",
                    "content": {"content_type": "code", "text": "1+2"},
                    "create_time": 1_704_067_202.0,
                },
            },
            "out1": {"id": "out1", "parent": "code1", "children": [], "message": output_message},
        },
    }


def _gemini_cli_payload(tool_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sessionId": "gemini-outcome",
        "projectHash": "hash",
        "startTime": "2026-02-01T08:00:00.000Z",
        "lastUpdated": "2026-02-01T08:01:00.000Z",
        "messages": [
            {"id": "g1", "timestamp": "2026-02-01T08:00:01.000Z", "type": "user", "content": ["go"]},
            {
                "id": "g2",
                "timestamp": "2026-02-01T08:00:02.000Z",
                "type": "gemini",
                "content": "done",
                "toolCalls": [tool_record],
            },
        ],
    }


def _drive_payload(outcome: str | None) -> dict[str, Any]:
    execution: dict[str, Any] = {"output": "42"}
    if outcome is not None:
        execution["outcome"] = outcome
    return {
        "id": "drive-outcome",
        "title": "outcome",
        "createTime": "2026-03-01T09:00:00Z",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "run it"},
                {"role": "model", "executableCode": {"code": "print(42)"}, "codeExecutionResult": execution},
            ]
        },
    }


def _hermes_payload(tool_content: str) -> dict[str, Any]:
    return {
        "session_id": "hermes-outcome",
        "model": "hermes-3",
        "session_start": "2026-05-01T07:00:00.000000",
        "messages": [
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": "running",
                "tool_calls": [{"id": "call-h1", "function": {"name": "shell", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-h1", "content": tool_content},
        ],
    }


# --------------------------------------------------------------------------
# Detection -> parser -> writer -> hydration -> actions -> public envelope
# --------------------------------------------------------------------------

_ROUTES: tuple[tuple[str, Provider, Callable[[], ParsedSession], Triple, str | None], ...] = (
    (
        "claude-code-success",
        Provider.CLAUDE_CODE,
        lambda: parse_code(_claude_code_records({"content": "ok", "is_error": False}), "cc-outcome"),
        (ToolOutcome.OK.value, 0, None),
        "outcome_success",
    ),
    (
        "claude-code-failure",
        Provider.CLAUDE_CODE,
        lambda: parse_code(_claude_code_records({"content": "boom", "is_error": True}), "cc-outcome"),
        (ToolOutcome.ERROR.value, 1, None),
        "outcome_error",
    ),
    (
        "claude-code-absent-report",
        Provider.CLAUDE_CODE,
        lambda: parse_code(_claude_code_records({"content": "Error: it failed, fatal"}), "cc-outcome"),
        (ToolOutcome.UNKNOWN.value, None, NOT_REPORTED),
        "outcome_unknown",
    ),
    (
        "claude-code-unsupported-is-error-shape",
        Provider.CLAUDE_CODE,
        lambda: parse_code(_claude_code_records({"content": "?", "is_error": "maybe"}), "cc-outcome"),
        (ToolOutcome.UNKNOWN.value, None, UNSUPPORTED),
        "outcome_unknown",
    ),
    (
        "claude-code-background-start-ack",
        Provider.CLAUDE_CODE,
        lambda: parse_code(
            _claude_code_records(
                {"content": "started", "is_error": False},
                extra={"toolUseResult": {"backgroundTaskId": "bg-1"}},
            ),
            "cc-outcome",
        ),
        (ToolOutcome.UNKNOWN.value, None, DISTRUSTED),
        "outcome_unknown",
    ),
    (
        "codex-exit-code-only",
        Provider.CODEX,
        lambda: parse_codex(_codex_records('{"exit_code": 3}'), "codex-outcome"),
        (ToolOutcome.ERROR.value, 1, None),
        "outcome_error",
    ),
    (
        "codex-absent-report",
        Provider.CODEX,
        lambda: parse_codex(_codex_records("plain output"), "codex-outcome"),
        (ToolOutcome.UNKNOWN.value, None, NOT_REPORTED),
        "outcome_unknown",
    ),
    (
        "codex-truncated-envelope",
        Provider.CODEX,
        lambda: parse_codex(_codex_records('{"output": "partial", "exit_c'), "codex-outcome"),
        (ToolOutcome.UNKNOWN.value, None, TRUNCATED),
        "outcome_unknown",
    ),
    (
        "codex-unsupported-exit-code-type",
        Provider.CODEX,
        lambda: parse_codex(_codex_records('{"exit_code": "0"}'), "codex-outcome"),
        (ToolOutcome.UNKNOWN.value, None, UNSUPPORTED),
        "outcome_unknown",
    ),
    (
        "chatgpt-terminal-success",
        Provider.CHATGPT,
        lambda: parse_chatgpt(_chatgpt_payload("finished_successfully"), "chatgpt-outcome"),
        (ToolOutcome.OK.value, 0, None),
        "outcome_success",
    ),
    (
        "chatgpt-terminal-partial",
        Provider.CHATGPT,
        lambda: parse_chatgpt(_chatgpt_payload("finished_partial_completion"), "chatgpt-outcome"),
        (ToolOutcome.ERROR.value, 1, None),
        "outcome_error",
    ),
    (
        "chatgpt-unconcluded",
        Provider.CHATGPT,
        lambda: parse_chatgpt(_chatgpt_payload("in_progress"), "chatgpt-outcome"),
        (ToolOutcome.UNKNOWN.value, None, NOT_REPORTED),
        "outcome_unknown",
    ),
    (
        "chatgpt-unmapped-status",
        Provider.CHATGPT,
        lambda: parse_chatgpt(_chatgpt_payload("finished_with_something_new"), "chatgpt-outcome"),
        (ToolOutcome.UNKNOWN.value, None, UNSUPPORTED),
        "outcome_unknown",
    ),
    (
        "gemini-cli-status-success",
        Provider.GEMINI_CLI,
        lambda: parse_gemini_cli(
            _gemini_cli_payload({"id": "tc-1", "name": "read_file", "status": "success", "resultDisplay": "ok"}),
            "gemini-outcome",
        ),
        (ToolOutcome.OK.value, 0, None),
        "outcome_success",
    ),
    (
        "gemini-cli-status-error",
        Provider.GEMINI_CLI,
        lambda: parse_gemini_cli(
            _gemini_cli_payload({"id": "tc-1", "name": "read_file", "status": "error", "resultDisplay": "nope"}),
            "gemini-outcome",
        ),
        (ToolOutcome.ERROR.value, 1, None),
        "outcome_error",
    ),
    (
        "gemini-cli-unmapped-status",
        Provider.GEMINI_CLI,
        lambda: parse_gemini_cli(
            _gemini_cli_payload({"id": "tc-1", "name": "read_file", "status": "awaiting_approval"}),
            "gemini-outcome",
        ),
        (ToolOutcome.UNKNOWN.value, None, UNSUPPORTED),
        "outcome_unknown",
    ),
    (
        "gemini-cli-absent-status",
        Provider.GEMINI_CLI,
        lambda: parse_gemini_cli(
            _gemini_cli_payload({"id": "tc-1", "name": "read_file", "resultDisplay": "contents"}),
            "gemini-outcome",
        ),
        (ToolOutcome.UNKNOWN.value, None, NOT_REPORTED),
        "outcome_unknown",
    ),
    (
        "hermes-exit-code",
        Provider.HERMES,
        lambda: parse_hermes(_hermes_payload('{"output": "ran", "exit_code": 0}'), "hermes-outcome"),
        (ToolOutcome.OK.value, 0, None),
        "outcome_success",
    ),
    (
        "hermes-error-envelope",
        Provider.HERMES,
        lambda: parse_hermes(_hermes_payload('{"output": "", "error": "boom"}'), "hermes-outcome"),
        (ToolOutcome.ERROR.value, 1, None),
        "outcome_error",
    ),
    (
        "hermes-absent-report",
        Provider.HERMES,
        lambda: parse_hermes(_hermes_payload("plain tool output"), "hermes-outcome"),
        (ToolOutcome.UNKNOWN.value, None, NOT_REPORTED),
        "outcome_unknown",
    ),
    (
        "hermes-truncated-envelope",
        Provider.HERMES,
        lambda: parse_hermes(_hermes_payload('\x00json:{"output": "partial", "exit_c'), "hermes-outcome"),
        (ToolOutcome.UNKNOWN.value, None, TRUNCATED),
        "outcome_unknown",
    ),
    # AI Studio / Drive results carry no tool id, so they are never paired into
    # ``actions`` -- the structural outcome still comes from the record's own
    # ``outcome`` field rather than the text it is rendered into.
    (
        "aistudio-drive-outcome-ok",
        Provider.DRIVE,
        lambda: parse_chunked_prompt(Provider.DRIVE, _drive_payload("OUTCOME_OK"), "drive-outcome"),
        (ToolOutcome.OK.value, 0, None),
        None,
    ),
    (
        "aistudio-drive-outcome-failed",
        Provider.DRIVE,
        lambda: parse_chunked_prompt(Provider.DRIVE, _drive_payload("OUTCOME_FAILED"), "drive-outcome"),
        (ToolOutcome.ERROR.value, 1, None),
        None,
    ),
    (
        "aistudio-drive-unmapped-outcome",
        Provider.DRIVE,
        lambda: parse_chunked_prompt(Provider.DRIVE, _drive_payload("OUTCOME_SOMETHING_NEW"), "drive-outcome"),
        (ToolOutcome.UNKNOWN.value, None, UNSUPPORTED),
        None,
    ),
    (
        "aistudio-drive-absent-outcome",
        Provider.DRIVE,
        lambda: parse_chunked_prompt(Provider.DRIVE, _drive_payload(None), "drive-outcome"),
        (ToolOutcome.UNKNOWN.value, None, NOT_REPORTED),
        None,
    ),
)


@pytest.mark.parametrize(
    ("label", "provider", "build", "expected", "expected_state"),
    _ROUTES,
    ids=[route[0] for route in _ROUTES],
)
def test_provider_record_reaches_the_public_envelope_with_one_triple(
    label: str,
    provider: Provider,
    build: Callable[[], ParsedSession],
    expected: Triple,
    expected_state: str | None,
    tmp_path: Path,
) -> None:
    """One provider record, one structural triple, identical at every surface."""
    session = build()
    result_blocks = [
        block for message in session.messages for block in message.blocks if block.type is BlockType.TOOL_RESULT
    ]
    assert result_blocks, f"{label}: parser produced no tool_result block"

    conn = _connect(tmp_path / f"{label}.db")
    try:
        session_id = _write(conn, session)
        stored = _stored_triples(conn, session_id)
        assert stored == [expected], f"{label}: stored {stored}"
        assert _envelope_triples(conn, session_id) == stored, f"{label}: envelope disagrees with the block"
        states = _action_states(conn, session_id)
        expected_states = [] if expected_state is None else [(expected_state, expected[2])]
        assert states == expected_states, f"{label}: actions projected {states}"
    finally:
        conn.close()


def test_detection_routes_each_wire_record_to_the_parser_that_maps_its_outcome() -> None:
    """The fixtures above are the shapes dispatch actually admits.

    Anti-vacuity: a fixture shaped so no detector claims it would still parse
    when called directly, leaving the outcome mapping proved on a record the
    pipeline never routes there.
    """
    assert detect_provider(_claude_code_records({"content": "ok", "is_error": False})) is Provider.CLAUDE_CODE
    assert codex_looks_like(_codex_records("plain output"))
    assert chatgpt_looks_like(_chatgpt_payload("in_progress"))
    assert looks_like_gemini_cli(_gemini_cli_payload({"id": "tc-1", "name": "read_file", "status": "success"}))
    assert looks_like_hermes(_hermes_payload("plain tool output"))
    assert drive_looks_like(_drive_payload("OUTCOME_OK"))


def test_unpaired_invocation_is_no_result_not_an_unknown_outcome(tmp_path: Path) -> None:
    """An interruption is a known state, and never borrows an unknown reason.

    Anti-vacuity: making ``no_result`` share the ``unknown`` state collapses
    this assertion into the unknown cases above.
    """
    session = parse_code(
        [
            {
                "type": "assistant",
                "uuid": "assistant-1",
                "sessionId": "cc-unpaired",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "call-1", "name": "Bash", "input": {}}],
                },
            }
        ],
        "cc-unpaired",
    )
    conn = _connect(tmp_path / "unpaired.db")
    try:
        session_id = _write(conn, session)
        assert _stored_triples(conn, session_id) == []
        assert _action_states(conn, session_id) == [("no_result", None)]
    finally:
        conn.close()


# --------------------------------------------------------------------------
# The tuple law and its owners
# --------------------------------------------------------------------------


def _declared_pairs() -> list[tuple[Origin, ToolResultUnknownReason]]:
    return [
        (spec.origin, reason)
        for spec in origin_specs()
        for reason in sorted(spec.tool_outcome_unknown_reasons, key=lambda item: item.value)
    ]


@pytest.mark.parametrize(
    ("origin", "reason"),
    _declared_pairs(),
    ids=[f"{origin.value}-{reason.value}" for origin, reason in _declared_pairs()],
)
def test_every_declared_reason_round_trips_to_the_public_envelope(
    origin: Origin, reason: ToolResultUnknownReason, tmp_path: Path
) -> None:
    """Each reason an origin owns survives the write and every read surface.

    Anti-vacuity: dropping the reason column from the writer or the hydrator
    turns the round-tripped reason into ``None`` here.
    """
    spec = next(item for item in origin_specs() if item.origin is origin)
    conn = _connect(tmp_path / f"{origin.value}-{reason.value}.db")
    try:
        session_id = _write(
            conn,
            ParsedSession(
                source_name=spec.provider_wires[0],
                provider_session_id=f"declared-{reason.value}",
                messages=[
                    ParsedMessage(
                        provider_message_id="use",
                        role=Role.ASSISTANT,
                        blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="call-1", tool_name="run")],
                    ),
                    ParsedMessage(
                        provider_message_id="result",
                        role=Role.TOOL,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="call-1",
                                text="no verdict",
                                outcome_unknown_reason=reason.value,
                            )
                        ],
                    ),
                ],
            ),
        )
        expected: Triple = (ToolOutcome.UNKNOWN.value, None, reason.value)
        assert _stored_triples(conn, session_id) == [expected]
        assert _envelope_triples(conn, session_id) == [expected]
        assert _action_states(conn, session_id) == [("outcome_unknown", reason.value)]
    finally:
        conn.close()


def test_an_undeclared_reason_refuses_the_write(tmp_path: Path) -> None:
    """A reason no parser of this origin owns cannot enter the archive.

    Anti-vacuity: deleting ``_require_declared_reason`` writes the row and this
    raises nothing; widening every origin's declaration to the whole vocabulary
    does the same.
    """
    undeclared = next(
        reason
        for reason in ToolResultUnknownReason
        if reason not in tool_outcome_unknown_reasons_for_origin(Origin.CLAUDE_CODE_SESSION)
    )
    conn = _connect(tmp_path / "undeclared.db")
    try:
        session = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="undeclared",
            messages=[
                ParsedMessage(
                    provider_message_id="use",
                    role=Role.ASSISTANT,
                    blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id="call-1", tool_name="run")],
                ),
                ParsedMessage(
                    provider_message_id="result",
                    role=Role.TOOL,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_RESULT,
                            tool_id="call-1",
                            text="no verdict",
                            outcome_unknown_reason=undeclared.value,
                        )
                    ],
                ),
            ],
        )
        with pytest.raises(ValueError, match="undeclared unknown reason"):
            _write(conn, session)
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_tool_result_without_a_derived_reason_cannot_be_constructed() -> None:
    """The producer boundary refuses the shape a blanket default used to mint.

    This is the mutation "drop the reason mapping at a producer": every parser
    site above builds this object, so an omitted derivation stops there rather
    than entering the archive as an unattributed unknown.
    """
    with pytest.raises(ValueError, match="must carry one ToolResultUnknownReason"):
        ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id="call-1", text="no verdict")


def test_a_known_outcome_cannot_carry_a_reason() -> None:
    """The other half of the tuple law: a verdict has nothing to explain."""
    with pytest.raises(ValueError, match="known tool-result outcomes cannot carry an unknown reason"):
        ParsedContentBlock(
            type=BlockType.TOOL_RESULT,
            tool_id="call-1",
            text="done",
            is_error=False,
            outcome_unknown_reason=NOT_REPORTED,
        )


def test_an_explicit_unknown_outcome_cannot_bypass_its_reason() -> None:
    """A pre-populated UNKNOWN value still needs the structural explanation."""
    with pytest.raises(ValueError, match="unknown tool-result outcomes must carry"):
        ParsedContentBlock(
            type=BlockType.TOOL_RESULT,
            tool_id="call-1",
            text="no verdict",
            tool_outcome=ToolOutcome.UNKNOWN,
        )


def test_hermes_off_type_success_is_unsupported_not_false() -> None:
    """Provider fields are type-sensitive; a string must not become failure."""
    session = parse_hermes(_hermes_payload('{"output": "ran", "success": "false"}'), "hermes-outcome")
    result_blocks = [
        block for message in session.messages for block in message.blocks if block.type is BlockType.TOOL_RESULT
    ]
    assert result_blocks
    assert result_blocks[0].is_error is None
    assert result_blocks[0].outcome_unknown_reason == UNSUPPORTED


def test_an_unknown_outcome_never_coerces_to_a_reported_success() -> None:
    """Coercing NULL to false is the failure this vocabulary exists to prevent.

    Anti-vacuity: replace ``is_error = None if outcome is UNKNOWN`` in
    ``derive_tool_outcomes`` with ``bool(is_error)`` and every unknown route
    below reports ``is_error == 0``.
    """
    unknown_routes = [route for route in _ROUTES if route[3][0] == ToolOutcome.UNKNOWN.value]
    assert unknown_routes, "no unknown-outcome route to check"
    for _label, _provider, build, expected, _state in unknown_routes:
        session = build()
        for message in session.messages:
            for block in message.blocks:
                if block.type is BlockType.TOOL_RESULT:
                    assert block.is_error is None
                    assert block.outcome_unknown_reason == expected[2]


# --------------------------------------------------------------------------
# Clean vs incremental ingestion, and unknown becoming known
# --------------------------------------------------------------------------


def _codex_session(output: Any) -> ParsedSession:
    return parse_codex(_codex_records(output), "codex-outcome")


def test_clean_and_incremental_ingestion_produce_identical_triples(tmp_path: Path) -> None:
    """Re-ingesting the same record twice cannot change the recorded outcome.

    Anti-vacuity: a merge path that coalesces the verdict columns
    independently pairs the fresh outcome with a stale ``is_error`` and the
    second read stops matching the first.
    """
    clean = _connect(tmp_path / "clean.db")
    incremental = _connect(tmp_path / "incremental.db")
    try:
        clean_id = _write(clean, _codex_session("plain output"))
        _write(incremental, _codex_session("plain output"))
        incremental_id = _write(incremental, _codex_session("plain output"))
        assert _stored_triples(clean, clean_id) == _stored_triples(incremental, incremental_id)
        assert _stored_triples(clean, clean_id) == [(ToolOutcome.UNKNOWN.value, None, NOT_REPORTED)]
    finally:
        clean.close()
        incremental.close()


def test_a_later_provider_report_rewrites_the_row_and_moves_content_identity(tmp_path: Path) -> None:
    """Unknown becoming known is a content change, not a silent overwrite.

    Anti-vacuity: drop ``tool_outcome``/``outcome_unknown_reason`` from the
    block content hash and the hash stays equal while the triple changes.
    """
    conn = _connect(tmp_path / "resolved.db")
    try:
        session_id = _write(conn, _codex_session("plain output"))
        before = conn.execute(
            "SELECT content_hash FROM blocks WHERE session_id = ? AND block_type = 'tool_result'",
            (session_id,),
        ).fetchone()[0]
        assert _stored_triples(conn, session_id) == [(ToolOutcome.UNKNOWN.value, None, NOT_REPORTED)]

        _write(conn, _codex_session('{"exit_code": 0, "output": "plain output"}'))
        after = conn.execute(
            "SELECT content_hash FROM blocks WHERE session_id = ? AND block_type = 'tool_result'",
            (session_id,),
        ).fetchone()[0]
        assert _stored_triples(conn, session_id) == [(ToolOutcome.OK.value, 0, None)]
        assert after != before
        assert _action_states(conn, session_id) == [("outcome_success", None)]
    finally:
        conn.close()


def test_the_vocabulary_is_a_closed_partition_with_an_owner_for_every_member() -> None:
    """No member is dead weight and no origin owns a reason it cannot derive.

    Anti-vacuity: adding a member to ``ToolResultUnknownReason`` that no origin
    declares fails here, as does removing the origin that owns one.
    """
    owned: set[ToolResultUnknownReason] = set()
    for spec in origin_specs():
        owned |= set(spec.tool_outcome_unknown_reasons)
    assert owned == set(ToolResultUnknownReason)

    exercised = {route[3][2] for route in _ROUTES if route[3][2] is not None}
    assert exercised == {reason.value for reason in ToolResultUnknownReason}


def test_every_origin_that_can_produce_a_tool_result_declares_its_reasons() -> None:
    """An executable origin whose parser can emit an unknown must own it.

    Anti-vacuity: clearing an origin's declaration makes its unknown results
    refuse at the writer, which the provider routes above then catch.
    """
    routed_origins = {Origin.from_string(name) for name in _routed_origin_names()}
    for origin in routed_origins:
        assert tool_outcome_unknown_reasons_for_origin(origin), f"{origin.value} produces unknowns but declares none"


def _routed_origin_names() -> Sequence[str]:
    return sorted({origin_from_provider(route[1]).value for route in _ROUTES})
