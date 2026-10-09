from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import BlockType, MaterialOrigin, Provider, Role
from polylogue.sources.dispatch import iter_parsed_stream, parse_payload
from polylogue.sources.parsers.base import AdmissionUnit
from polylogue.sources.parsers.chatgpt import parse as parse_chatgpt
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session
from tests.infra.retained_replay import publish_retained_payload


@pytest.mark.parametrize("reply_role", ["assistant", "user", "tool"])
def test_declared_tool_author_pairs_chained_replies_without_changing_role(tmp_path: Path, reply_role: str) -> None:
    mapping: dict[str, object] = {
        "call-node": {
            "id": "call-node",
            "parent": None,
            "children": ["reply-1"],
            "message": {
                "id": "call-message",
                "author": {"role": "assistant"},
                "recipient": "web.run",
                "content": {"content_type": "text", "parts": ['{"search_query":[{"q":"neutral"}]}']},
            },
        },
    }
    for ordinal in (1, 2):
        mapping[f"reply-{ordinal}"] = {
            "id": f"reply-{ordinal}",
            "parent": "call-node" if ordinal == 1 else "reply-1",
            "children": ["reply-2"] if ordinal == 1 else [],
            "message": {
                "id": f"result-message-{ordinal}",
                "author": {"role": reply_role, "metadata": {"real_author": "tool:web.run"}},
                "content": {"content_type": "text", "parts": [f"Neutral result {ordinal}"]},
                "status": "finished_successfully",
            },
        }
    parsed = parse_chatgpt({"id": "neutral-tool-session", "mapping": mapping, "current_node": "reply-2"}, "fallback")
    for message in parsed.messages[1:]:
        assert message.role is Role.normalize(reply_role)
        assert message.material_origin is MaterialOrigin.TOOL_RESULT
        assert message.blocks[0].type is BlockType.TOOL_RESULT
        assert message.blocks[0].tool_id == "call-message"
    with fixture_index_connection(tmp_path / "index.db") as conn:
        session_id = write_fixture_index_session(conn, parsed)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        assert conn.execute("SELECT tool_outcome FROM action_pairs WHERE session_id=?", (session_id,)).fetchall() == [
            ("ok",)
        ]
        assert (
            conn.execute(
                "SELECT role,material_origin FROM messages WHERE session_id=? ORDER BY position", (session_id,)
            ).fetchall()[1:]
            == [(reply_role, "tool_result")] * 2
        )


@pytest.mark.parametrize("provider", [Provider.CLAUDE_CODE, Provider.CLAUDE_AI])
def test_claude_nested_result_media_survives_parse_accounting_and_stored_tree(
    tmp_path: Path, provider: Provider
) -> None:
    content = [
        {
            "type": "tool_result",
            "tool_use_id": "neutral-call",
            "is_error": False,
            "content": [
                {"type": "text", "text": "Neutral result"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                    },
                },
                {
                    "type": "document",
                    "title": "Neutral document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "Neutral document body",
                    },
                },
                {"type": "future_media", "payload": "neutral opaque"},
            ],
        }
    ]
    payload: object = (
        [
            {
                "type": "user",
                "uuid": "result-message",
                "sessionId": "neutral-media-session",
                "message": {"role": "user", "content": content},
            }
        ]
        if provider is Provider.CLAUDE_CODE
        else {
            "uuid": "neutral-media-session",
            "chat_messages": [
                {
                    "uuid": "result-message",
                    "sender": "user",
                    "content": content,
                }
            ],
        }
    )
    parsed = parse_payload(provider, payload, "fallback")[0]
    message = parsed.messages[0]
    assert message.role is Role.TOOL
    assert message.material_origin is MaterialOrigin.TOOL_RESULT
    assert [block.type for block in message.blocks] == [
        BlockType.TOOL_RESULT,
        BlockType.IMAGE,
        BlockType.DOCUMENT,
        BlockType.DOCUMENT,
    ]
    assert message.blocks[0].text == "Neutral result"
    assert message.blocks[2].text == "Neutral document body"
    assert [attachment.inline_bytes for attachment in parsed.attachments] == [
        base64.b64decode(content[0]["content"][1]["source"]["data"]),
        b"Neutral document body",
    ]
    assert parsed.unit_accounting is not None
    parsed.unit_accounting.assert_conserved()
    assert parsed.unit_accounting.expected[AdmissionUnit.PART] == 5
    assert parsed.unit_accounting.expected[AdmissionUnit.BLOCK] == 4
    assert any(
        outcome.unit is AdmissionUnit.PART and outcome.key == "future_media"
        for outcome in parsed.unit_accounting.outcomes
    )
    wire = (
        "\n".join(json.dumps(record) for record in payload) if isinstance(payload, list) else json.dumps(payload)
    ).encode()
    _, session_ids = asyncio.run(
        publish_retained_payload(
            tmp_path / "archive",
            provider=provider,
            payload=wire,
            source_path="neutral-media.jsonl" if provider is Provider.CLAUDE_CODE else "neutral-media.json",
            acquired_at_ms=1,
        )
    )
    assert len(session_ids) == 1
    session_id = session_ids[0]
    with sqlite3.connect(tmp_path / "archive" / "index.db") as conn:
        rows = conn.execute(
            "SELECT block_type,text,semantic_extra_json FROM blocks WHERE session_id=? ORDER BY position", (session_id,)
        ).fetchall()
        assert [row[0] for row in rows] == ["tool_result", "image", "document", "document"]
        assert rows[2][1] == "Neutral document body"
        assert json.loads(rows[1][2])["metadata"]["source_digest"] == message.blocks[1].metadata["source_digest"]
        assert conn.execute("SELECT COUNT(*) FROM attachments WHERE session_id=?", (session_id,)).fetchone() == (2,)


def test_streamed_claude_code_media_uses_attachment_sink(tmp_path: Path) -> None:
    store = SqliteMessageStore(tmp_path / "prepared.sqlite")
    try:
        parsed = list(
            iter_parsed_stream(
                Provider.CLAUDE_CODE,
                iter(
                    [
                        {
                            "type": "user",
                            "uuid": "result-message",
                            "sessionId": "neutral-media-session",
                            "message": {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "neutral-call",
                                        "content": [
                                            {
                                                "type": "document",
                                                "source": {"type": "url", "url": "https://example.test/neutral.pdf"},
                                            }
                                        ],
                                    }
                                ],
                            },
                        }
                    ]
                ),
                "neutral-media-session",
                message_sink_factory=store.new_sink,
                event_sink_factory=store.new_event_sink,
                attachment_sink_factory=store.new_attachment_sink,
            )
        )[0]
        assert not isinstance(parsed.attachments, list)
        assert parsed.attachments[0].source_url == "https://example.test/neutral.pdf"
        assert parsed.messages[0].role is Role.TOOL
        assert parsed.messages[0].blocks[1].type is BlockType.DOCUMENT
        assert parsed.unit_accounting is not None
        parsed.unit_accounting.assert_conserved()
    finally:
        store.close()
