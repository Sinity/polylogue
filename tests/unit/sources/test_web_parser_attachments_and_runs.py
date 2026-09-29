"""Browser capture attachments, Claude web attachments and ChatGPT run evidence."""

import pytest

from polylogue.core.enums import BlockType
from polylogue.sources.parsers.browser_capture import parse as parse_capture
from polylogue.sources.parsers.chatgpt import parse as parse_chatgpt
from polylogue.sources.parsers.claude import parse_ai
from tests.infra.source_parser_cases import (
    case,
    chatgpt_parts,
    chatgpt_run,
    claude_conversation_attachment,
    native_and_envelope_files,
)


def test_typed_extracted_content_precedes_provider_metadata() -> None:
    """Stale provider metadata formerly replaced newer typed extracted text."""
    payload = case("browser")
    payload["session"]["attachments"] = [
        {
            "provider_attachment_id": "extracted",
            "message_provider_id": "u1",
            "name": "same.txt",
            "mime_type": "text/plain",
            "extracted_content": "fresh",
            "provider_meta": {"extracted_content": "old"},
        }
    ]
    session = parse_capture(payload, "fallback")
    [attachment] = session.attachments
    assert attachment.inline_bytes == b"fresh"


def test_claude_file_projection_reconciles_without_bytes() -> None:
    """Exact native file_uuid plus claude-file: id formerly yielded two rows."""
    session = parse_capture(native_and_envelope_files(), "fallback")
    [attachment] = session.attachments
    assert attachment.provider_attachment_id == "file-1"
    assert attachment.message_provider_id == "u1"
    assert attachment.inline_bytes is None


def test_projection_id_collision_revalidates_native_target() -> None:
    """Reusing the first cached target overwrote that file with the second bytes."""
    session = parse_capture(native_and_envelope_files(repeated_synthetic_id=True), "fallback")
    assert {row.provider_attachment_id: (row.name, row.inline_bytes) for row in session.attachments} == {
        "file-1": ("file-1.txt", b"aaaa"),
        "file-2": ("file-2.txt", b"bbbb"),
    }


@pytest.mark.parametrize("name,expected", [("failed_list", True), ("unknown_list", None), ("success_list", False)])
def test_list_aggregate_result_overrides_delivery_status(name: str, expected: bool | None) -> None:
    """A list-valued failed/unknown run formerly inherited successful delivery."""
    session = parse_chatgpt(chatgpt_run(name), "fallback")
    message = next(message for message in session.messages if message.provider_message_id == "result")
    result = next(block for block in message.blocks if block.type is BlockType.TOOL_RESULT)
    assert result.is_error is expected
    assert len([event for event in session.session_events if event.event_type == "chatgpt_code_interpreter_run"]) == 1


def test_project_scoped_conversation_citation_keeps_native_identity() -> None:
    """/g/g-p-.../c/... citations formerly lost their source-conversation id."""
    payload = case("chatgpt")
    payload["mapping"]["result"]["message"]["metadata"]["conversation_context_citation_metadata"] = [case("citation")]
    session = parse_chatgpt(payload, "fallback")
    constructs = [
        construct for message in session.messages for block in message.blocks for construct in block.web_constructs
    ]
    matching = [construct for construct in constructs if construct.source_id == "11111111-2222-3333-4444-555555555555"]
    assert len(matching) == 1


@pytest.mark.parametrize("name,outputs", [("direct_output", ["only in run"]), ("two_outputs", ["first", "second"])])
def test_direct_run_outputs_are_retained(name: str, outputs: list[str]) -> None:
    """Direct output/exit_code fields vanished unless repeated in node text."""
    session = parse_chatgpt(chatgpt_run(name), "fallback")
    events = [event for event in session.session_events if event.event_type == "chatgpt_code_interpreter_run"]
    assert [event.payload["output"] for event in events] == outputs
    assert [event.payload["exit_code"] for event in events] == [0] * len(outputs)


@pytest.mark.parametrize("location", ["part", "envelope"])
def test_unhashable_content_type_keeps_valid_message(location: str) -> None:
    """An object-valued discriminator formerly raised TypeError during set lookup."""
    payload = chatgpt_parts("bad_type")
    if location == "envelope":
        payload["mapping"]["result"]["message"]["content"]["content_type"] = {"malformed": True}
    session = parse_chatgpt(payload, "fallback")
    message = next(message for message in session.messages if message.provider_message_id == "result")
    assert message.blocks
    if location == "part":
        assert "kept" in (message.text or "")


def test_text_only_audio_transcription_is_available() -> None:
    """A complete text-only transcription formerly claimed unavailable media."""
    session = parse_chatgpt(chatgpt_parts("transcription"), "fallback")
    constructs = [
        construct for message in session.messages for block in message.blocks for construct in block.web_constructs
    ]
    transcription = next(construct for construct in constructs if construct.text == "complete transcript")
    assert transcription.status is None


@pytest.mark.parametrize("relation,expected_count", [("equal", 1), ("different", 2), ("absent", 1)])
def test_conflicting_bytes_keep_a_top_level_file_distinct(relation: str, expected_count: int) -> None:
    """Without the byte-conflict check, "different" collapses into one owned row and loses a payload."""
    session = parse_ai(claude_conversation_attachment(byte_relation=relation), "fallback")
    assert len(session.attachments) == expected_count
    if expected_count == 2:
        assert sum(row.message_provider_id is None for row in session.attachments) == 1
    if relation == "different":
        assert {row.inline_bytes for row in session.attachments} == {b"same", b"diff"}
