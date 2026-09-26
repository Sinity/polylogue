from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Origin, ToolOutcome, ToolResultUnknownReason
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSessionEvent
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.sources.tool_outcomes import derive_tool_outcomes


def _use(tool_id: str, provider_id: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=Role.ASSISTANT,
        blocks=[ParsedContentBlock(type=BlockType.TOOL_USE, tool_id=tool_id, tool_name="run")],
    )


def _result(tool_id: str, provider_id: str, *, reason: str | None = None) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=Role.TOOL,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id=tool_id,
                text="synthetic result",
                outcome_unknown_reason=reason,
            )
        ],
    )


def _dumps(messages: Iterable[ParsedMessage]) -> list[dict[str, object]]:
    return [message.model_dump(mode="json") for message in messages]


def test_disk_normalization_matches_collected_duplicate_and_owned_sidecar_outcomes(tmp_path: Path) -> None:
    messages = [
        _use("shared", "use-1"),
        _use("shared", "use-2"),
        _use("shared", "use-3"),
        _result("shared", "silent", reason=ToolResultUnknownReason.NOT_REPORTED.value),
        _result("shared", "reported", reason=ToolResultUnknownReason.NOT_REPORTED.value),
        _use("unmatched", "no-result"),
    ]
    events = [
        ParsedSessionEvent(
            event_type="claude_tool_execution_result",
            source_message_provider_id="reported",
            payload={"tool_use_id": "shared", "exit_code": 2},
        )
    ]
    original = _dumps(messages)
    expected = derive_tool_outcomes(messages, events, origin=Origin.CLAUDE_CODE_SESSION)
    assert _dumps(messages) == original

    store = SqliteMessageStore(tmp_path / "prepared.sqlite3")
    try:
        sink = store.new_sink()
        sink.extend(messages)
        returned = derive_tool_outcomes(sink, events, origin=Origin.CLAUDE_CODE_SESSION)
        assert returned is sink
        assert _dumps(list(sink)) == _dumps(expected)
        assert [message.blocks[0].tool_outcome for message in sink] == [
            ToolOutcome.UNKNOWN,
            ToolOutcome.ERROR,
            ToolOutcome.ERROR,
            ToolOutcome.UNKNOWN,
            ToolOutcome.ERROR,
            ToolOutcome.NO_RESULT,
        ]
        assert not list(tmp_path.glob("tool-outcomes-*"))
    finally:
        store.close()


def test_disk_normalization_rolls_back_late_refusal_and_cleans_state(tmp_path: Path) -> None:
    store = SqliteMessageStore(tmp_path / "prepared.sqlite3")
    try:
        sink = store.new_sink()
        sink.extend(
            [
                _use("first", "first"),
                _result(
                    "bad",
                    "later",
                    reason=ToolResultUnknownReason.DISTRUSTED.value,
                ),
            ]
        )
        original = _dumps(list(sink))
        with pytest.raises(ValueError, match="undeclared unknown reason"):
            derive_tool_outcomes(sink, [], origin=Origin.CHATGPT_EXPORT)
        assert _dumps(list(sink)) == original
        assert not list(tmp_path.glob("tool-outcomes-*"))
    finally:
        store.close()
