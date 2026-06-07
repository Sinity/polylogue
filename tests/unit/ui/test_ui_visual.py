"""Consolidated UI visual/rendering tests.

MERGED: test_rendering.py (renderer implementations, core formatting) +
        test_golden.py (golden/snapshot reference tests)

To regenerate golden files after intentional rendering changes:
    UPDATE_GOLDEN=1 pytest tests/unit/ui/test_ui_visual.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TypeAlias

import pytest

from polylogue.core.timestamps import canonical_timestamp_text
from tests.infra import GOLDEN_DIR
from tests.infra.archive_scenarios import archive_for_scenario_db, native_session_id_for
from tests.infra.assertions import (
    assert_contains_all,
    assert_messages_ordered,
)
from tests.infra.storage_records import (
    DbFactory,
    SessionBuilder,
    db_setup,
)


async def _format_native(
    archive_root: Path,
    db_path: Path,
    conv_id: str,
    *,
    provider: str = "test",
    output_format: str = "markdown",
) -> tuple[object, str]:
    """Read a seeded session through the archive facade and render it.

    Resolves the archive session id for ``(provider, conv_id)``, reads the
    domain ``Session`` via ``Polylogue.get_session``, and renders it
    with the production ``format_session`` entry point.

    Returns (session_object, markdown_text) tuple.
    """
    from polylogue.rendering.formatting import format_session

    archive = archive_for_scenario_db(db_path)
    async with archive:
        conv = await archive.get_session(native_session_id_for(provider, conv_id))
    if conv is None:
        raise ValueError(f"Session not found: {conv_id}")
    rendered = format_session(conv, output_format, None)
    return (conv, rendered)


JSONRecord: TypeAlias = dict[str, object]
WorkspaceEnv: TypeAlias = dict[str, Path]
MessageOrderingRow: TypeAlias = tuple[str, str, str | None]
AttachmentCaseMeta: TypeAlias = JSONRecord | list[JSONRecord] | None
AttachmentExpected: TypeAlias = str | list[str]

# =============================================================================
# RENDERER IMPLEMENTATION TESTS (from test_rendering.py)
# =============================================================================


@pytest.fixture
def sample_session_id(workspace_env: WorkspaceEnv) -> str:
    """Seed a sample session directly and return its archive session id."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "test-conv-1")
        .provider("chatgpt")
        .title("Test Session")
        .created_at("2024-01-01T10:00:00Z")
        .updated_at("2024-01-01T10:00:10Z")
        .add_message("msg1", role="user", text="Hello, can you help me?", timestamp="2024-01-01T10:00:00Z")
        .add_message(
            "msg2", role="assistant", text="Of course! How can I help you today?", timestamp="2024-01-01T10:00:05Z"
        )
        .add_message("msg3", role="user", text="I need help with Python testing", timestamp="2024-01-01T10:00:10Z")
    )
    builder.save()
    return builder.native_session_id()


@pytest.fixture
def sample_session_with_json(workspace_env: WorkspaceEnv) -> str:
    """Seed a session with JSON content directly; return its native id."""
    db_path = db_setup(workspace_env)
    builder = (
        SessionBuilder(db_path, "test-conv-json")
        .provider("chatgpt")
        .title("JSON Test")
        .created_at("2024-01-01T10:00:00Z")
        .updated_at("2024-01-01T10:00:05Z")
        .add_message("msg1", role="user", text="Search for Python testing", timestamp="2024-01-01T10:00:00Z")
        .add_message(
            "msg2",
            role="assistant",
            text='{"query": "Python testing", "results": ["pytest", "unittest"]}',
            timestamp="2024-01-01T10:00:05Z",
        )
    )
    builder.save()
    return builder.native_session_id()


FORMAT_CASES = [
    ("missing session", "nonexistent-conv", "raises ValueError"),
    ("basic session", "basic-conv", "returns rendered markdown"),
    ("null title", "no-title-conv", "uses untitled for null title"),
]


@pytest.mark.parametrize("label,conv_id,desc", FORMAT_CASES)
@pytest.mark.asyncio
async def test_formatter_format_comprehensive(workspace_env: WorkspaceEnv, label: str, conv_id: str, desc: str) -> None:
    db_path = db_setup(workspace_env)
    archive_root = workspace_env["archive_root"]

    if label == "missing session":
        with pytest.raises(ValueError, match="Session not found"):
            await _format_native(archive_root, db_path, conv_id)
    elif label == "basic session":
        (SessionBuilder(db_path, conv_id).provider("chatgpt").add_message("m1", role="user", text="Hello!").save())
        conv, markdown_text = await _format_native(archive_root, db_path, conv_id, provider="chatgpt")
        assert conv.title == "Test Session"  # type: ignore[attr-defined]
        assert str(conv.origin) == "chatgpt-export"  # type: ignore[attr-defined]
        assert "Hello!" in markdown_text
    elif label == "null title":
        (SessionBuilder(db_path, conv_id).title(None).add_message("m1", role="user", text="Body").save())
        conv, markdown_text = await _format_native(archive_root, db_path, conv_id)
        # Archive rendering uses "Untitled" for a null title
        assert conv.title is None  # type: ignore[attr-defined]
        assert "# Untitled" in markdown_text


# renders messages in source/position order (the order they
# were parsed/added), never re-sorted by timestamp (#1743: "Ordering never
# depends on timestamps"). These cases seed messages whose timestamps disagree
# with their source order and assert the source order is preserved verbatim.
MESSAGE_ORDERING_CASES = [
    (
        "source order ignores timestamps",
        "ordered-conv",
        [
            ("m3", "Third", "2024-01-01T12:00:30Z"),
            ("m1", "First", "2024-01-01T12:00:10Z"),
            ("m2", "Second", "2024-01-01T12:00:20Z"),
        ],
        "source order preserved despite out-of-order timestamps",
    ),
    (
        "null timestamps",
        "null-ts-conv",
        [
            ("m1", "Timestamped", "2024-01-01T12:00:00Z"),
            ("m2", "NoTimestamp", None),
        ],
        "source order preserved with a null timestamp",
    ),
    (
        "epoch timestamps",
        "epoch-conv",
        [
            ("m1", "LaterEpoch", "1704110400.5"),
            ("m2", "EarlierEpoch", "1704106800"),
        ],
        "epoch timestamps do not reorder source order",
    ),
]


@pytest.mark.parametrize("label,conv_id,message_data,desc", MESSAGE_ORDERING_CASES)
@pytest.mark.asyncio
async def test_message_ordering_comprehensive(
    workspace_env: WorkspaceEnv, label: str, conv_id: str, message_data: list[MessageOrderingRow], desc: str
) -> None:
    db_path = db_setup(workspace_env)
    builder = SessionBuilder(db_path, conv_id)
    for i, (msg_id, text, timestamp) in enumerate(message_data):
        role = "user" if i % 2 == 0 else "assistant"
        builder.add_message(msg_id, role=role, text=text, timestamp=timestamp)
    builder.save()
    _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id)
    if label == "source order ignores timestamps":
        assert_messages_ordered(markdown_text, "Third", "First", "Second")
    elif label == "null timestamps":
        assert_messages_ordered(markdown_text, "Timestamped", "NoTimestamp")
    elif label == "epoch timestamps":
        assert_messages_ordered(markdown_text, "LaterEpoch", "EarlierEpoch")


JSON_WRAPPING_CASES = [
    ('{"key": "value", "count": 42}', True, "JSON object wrapped"),
    ('[1, 2, 3, "four"]', True, "JSON array wrapped"),
    ("{malformed json without closing", False, "invalid JSON not wrapped"),
    ("{this is not json}", False, "JSON-like but not JSON"),
    ("This is just regular text.", False, "plain text not wrapped"),
]


@pytest.mark.parametrize("text,wrapped,desc", JSON_WRAPPING_CASES)
@pytest.mark.asyncio
async def test_json_text_wrapping_comprehensive(
    workspace_env: WorkspaceEnv, text: str, wrapped: bool, desc: str
) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"json-{hash(text) % 10000}-conv"
    (SessionBuilder(db_path, conv_id).title("Test").add_message("m1", role="tool", text=text).save())
    _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id)
    if wrapped:
        assert "```json" in markdown_text, f"Failed {desc}"
        assert "```" in markdown_text
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            assert any(key in markdown_text for key in parsed), f"Failed {desc}: JSON keys not found"
        elif isinstance(parsed, list):
            assert "[" in markdown_text and "]" in markdown_text, f"Failed {desc}: JSON array markers not found"
    else:
        assert "```json" not in markdown_text, f"Failed {desc}"
        assert text in markdown_text, f"Failed {desc}: text not found"


TIMESTAMP_RENDERING_CASES = [
    ("2024-01-15T10:30:00Z", True, "timestamps rendered"),
    (None, False, "no timestamp when null"),
]


@pytest.mark.parametrize("timestamp,rendered,desc", TIMESTAMP_RENDERING_CASES)
@pytest.mark.asyncio
async def test_timestamp_rendering_comprehensive(
    workspace_env: WorkspaceEnv, timestamp: str | None, rendered: bool, desc: str
) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"ts-{hash(str(timestamp)) % 10000}-conv"
    (
        SessionBuilder(db_path, conv_id)
        .title("Test")
        .add_message("m1", role="user", text="Hello", timestamp=timestamp)
        .save()
    )
    _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id)
    if rendered:
        # Archive rendering emits the canonical timestamp (with explicit UTC
        # offset) rather than echoing the raw ``Z``-suffixed input.
        from polylogue.core.timestamps import canonical_timestamp_text

        assert f"_Timestamp: {canonical_timestamp_text(timestamp)}_" in markdown_text, f"Failed {desc}"
    else:
        assert "_Timestamp:" not in markdown_text, f"Failed {desc}"


ATTACHMENT_CASES = [
    ("meta.name", {"name": "MyFile.pdf"}, "MyFile.pdf", "name from meta.name"),
    ("meta.provider_id", {"provider_id": "provider_file_123"}, "provider_file_123", "name from provider_id"),
    ("meta.drive_id", {"drive_id": "1ABC123XYZ"}, "1ABC123XYZ", "name from drive_id"),
    ("fallback ID", None, "att-fallback-123", "fallback to attachment_id"),
    ("empty meta", {}, "att-empty-meta", "fallback when meta empty"),
    (
        "multiple",
        [
            {"id": "att1", "meta": {"name": "File1.png"}},
            {"id": "att2", "meta": {"name": "File2.jpg"}},
            {"id": "att3", "meta": {"name": "File3.txt"}},
        ],
        ["File1.png", "File2.jpg", "File3.txt"],
        "multiple attachments",
    ),
    ("path", {"name": "Doc.pdf"}, "custom/path/to/file.pdf", "uses explicit path"),
]


@pytest.mark.parametrize("label,meta,expected,desc", ATTACHMENT_CASES)
@pytest.mark.asyncio
async def test_attachment_handling_comprehensive(
    workspace_env: WorkspaceEnv, label: str, meta: AttachmentCaseMeta, expected: AttachmentExpected, desc: str
) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"att-{label}-conv"
    builder = SessionBuilder(db_path, conv_id).title("Test").add_message("m1", role="user", text="See attachment")
    if label == "multiple":
        assert isinstance(meta, list)
        for att in meta:
            attachment_id = att["id"]
            att_meta = att.get("meta")
            assert isinstance(attachment_id, str)
            assert isinstance(att_meta, dict)
            attachment_name = att_meta["name"]
            assert isinstance(attachment_name, str)
            builder.add_attachment(
                attachment_id=attachment_id,
                message_id="m1",
                display_name=attachment_name,
            )
    elif label == "path":
        assert isinstance(expected, str)
        assert meta is None or isinstance(meta, dict)
        meta_name = meta.get("name") if isinstance(meta, dict) else None
        builder.add_attachment(
            attachment_id="att1",
            message_id="m1",
            path=expected,
            display_name=meta_name if isinstance(meta_name, str) else None,
        )
    else:
        assert isinstance(expected, str)
        assert meta is None or isinstance(meta, dict)
        is_empty_meta = meta is None or meta == {}
        att_id = expected if is_empty_meta else "att1"
        builder.add_attachment(
            attachment_id=att_id,
            message_id="m1",
            display_name=None if is_empty_meta else expected,
        )
    builder.save()
    _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id)
    if isinstance(expected, list):
        for exp in expected:
            assert exp in markdown_text, f"Failed {desc}: {exp} not found"
    else:
        assert expected in markdown_text, f"Failed {desc}"


@pytest.mark.asyncio
async def test_orphaned_attachments_section(workspace_env: WorkspaceEnv) -> None:
    db_path = db_setup(workspace_env)
    conv_id = "orphan-att-conv"
    (
        SessionBuilder(db_path, conv_id)
        .title("Test")
        .add_message("m1", role="user", text="Hello")
        .add_attachment(
            attachment_id="orphan-att",
            message_id=None,
            mime_type="image/png",
            size_bytes=2048,
            display_name="OrphanFile.png",
        )
        .save()
    )
    _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id)
    assert "## attachments" in markdown_text
    assert "OrphanFile.png" in markdown_text


METADATA_CASES = [
    ("counts", {"messages": 5, "attachments": 3}, "message and attachment counts"),
    ("timestamps", {"created": "2024-01-01T10:00:00Z", "updated": "2024-01-15T15:30:00Z"}, "created_at and updated_at"),
]


@pytest.mark.parametrize("label,data,desc", METADATA_CASES)
@pytest.mark.asyncio
async def test_metadata_comprehensive(workspace_env: WorkspaceEnv, label: str, data: JSONRecord, desc: str) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"meta-{label}-conv"
    if label == "counts":
        message_count = data["messages"]
        attachment_count = data["attachments"]
        assert isinstance(message_count, int)
        assert isinstance(attachment_count, int)
        builder = SessionBuilder(db_path, conv_id).title("Test")
        for i in range(message_count):
            role = "user" if i % 2 == 0 else "assistant"
            builder.add_message(f"m{i}", role=role, text=f"Message {i}")
        for i in range(attachment_count):
            builder.add_attachment(
                attachment_id=f"att{i}",
                message_id="m0",
                mime_type="text/plain",
                size_bytes=100,
            )
        builder.save()
        conv, _markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id)
        messages = list(conv.messages)  # type: ignore[attr-defined]
        attachments = [att for msg in messages for att in getattr(msg, "attachments", []) or []]
        assert len(messages) == message_count, f"Failed {desc}"
        assert len(attachments) == attachment_count, f"Failed {desc}"
    elif label == "timestamps":
        created = data["created"]
        updated = data["updated"]
        assert isinstance(created, str)
        assert isinstance(updated, str)
        (
            SessionBuilder(db_path, conv_id)
            .title("Test")
            .created_at(created)
            .updated_at(updated)
            .add_message("m1", role="user", text="anchor", timestamp=created)
            .add_message("m2", role="assistant", text="reply", timestamp=updated)
            .save()
        )
        conv, _markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id)
        # The archive domain Session derives created/updated from message
        # timestamps (min/max), exposed as datetime; compare canonical text.
        assert canonical_timestamp_text(str(conv.created_at)) == canonical_timestamp_text(created), (  # type: ignore[attr-defined]
            f"Failed {desc}"
        )
        assert canonical_timestamp_text(str(conv.updated_at)) == canonical_timestamp_text(updated), (  # type: ignore[attr-defined]
            f"Failed {desc}"
        )


MARKDOWN_STRUCTURE_CASES = [
    ("header", [], "header structure"),
    (
        "roles",
        [
            {"role": "user", "text": "User message"},
            {"role": "assistant", "text": "Assistant message"},
            {"role": "system", "text": "System message"},
        ],
        "role sections",
    ),
    (
        "empty messages",
        [
            {"role": "user", "text": "Real content"},
            {"role": "tool", "text": ""},
            {"role": "system", "text": "   "},
        ],
        "empty messages skipped",
    ),
    ("null role", [{"role": None, "text": "No role"}], "null role defaults to 'message'"),
]


@pytest.mark.parametrize("label,messages_data,desc", MARKDOWN_STRUCTURE_CASES)
@pytest.mark.asyncio
async def test_markdown_structure_comprehensive(
    workspace_env: WorkspaceEnv, label: str, messages_data: list[JSONRecord], desc: str
) -> None:
    db_path = db_setup(workspace_env)
    conv_id = f"md-{label}-conv"
    builder = SessionBuilder(db_path, conv_id).provider("chatgpt").title("My Chat Title")
    for i, msg in enumerate(messages_data):
        role = msg["role"]
        text = msg["text"]
        assert role is None or isinstance(role, str)
        assert isinstance(text, str)
        builder.add_message(
            f"m{i}",
            role=role,
            text=text,
            timestamp=f"2024-01-01T10:00:{i:02d}Z",
        )
    builder.save()
    _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id, provider="chatgpt")
    if label == "header":
        assert_contains_all(
            markdown_text,
            "# My Chat Title",
            "Origin: chatgpt-export",
            f"Session ID: {native_session_id_for('chatgpt', conv_id)}",
        )
    elif label == "roles":
        assert_contains_all(markdown_text, "## user", "## assistant", "## system")
    elif label == "empty messages":
        # Empty-text messages are skipped; archive keeps a whitespace-only
        # system body, so only the empty ``tool`` message drops out.
        assert "## user" in markdown_text
        assert "## tool" not in markdown_text
    elif label == "null role":
        # Archive role normalization maps a null role to ``user`` (the legacy
        # formatter emitted a synthetic ``## message`` heading).
        assert "## user" in markdown_text


# =============================================================================
# GOLDEN / SNAPSHOT TESTS (from test_golden.py)
# =============================================================================


def normalize_markdown(text: str) -> str:
    """Normalize markdown for comparison (handle whitespace differences)."""
    text = text.replace("\r\n", "\n")
    text = re.sub(
        r"(?:/tmp|/dev/shm)/(?:nix-shell\.[^/]+/)?(?:pytest-of-|pytest-polylogue[-/])[^)>\s]+",
        "$TMPDIR",
        text,
    )
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def assert_golden(name: str, actual: str) -> None:
    """Compare rendered output against a golden reference file."""
    golden_path = GOLDEN_DIR / f"{name}.md"
    normalized = normalize_markdown(actual)

    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(normalized)
        return

    if not golden_path.exists():
        pytest.fail(f"Golden file not found: {golden_path}\nRun with UPDATE_GOLDEN=1 to generate golden files.")

    expected = normalize_markdown(golden_path.read_text())
    assert normalized == expected, (
        f"Rendered output differs from golden file: {golden_path}\n"
        "Run with UPDATE_GOLDEN=1 to regenerate after intentional changes."
    )


class TestGoldenMarkdownRendering:
    """Test markdown rendering against golden reference files."""

    @pytest.mark.asyncio
    async def test_chatgpt_simple_session(self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-chatgpt-simple"
        factory.create_session(
            id=conv_id,
            provider="chatgpt",
            title="Simple ChatGPT Session",
            messages=[
                {"id": "msg1", "role": "user", "text": "Hello, how are you?", "timestamp": "2024-01-01T12:00:00Z"},
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "I'm doing well, thank you for asking! How can I help you today?",
                    "timestamp": "2024-01-01T12:00:05Z",
                },
                {
                    "id": "msg3",
                    "role": "user",
                    "text": "Can you explain what markdown is?",
                    "timestamp": "2024-01-01T12:00:15Z",
                },
                {
                    "id": "msg4",
                    "role": "assistant",
                    "text": "Markdown is a lightweight markup language for creating formatted text using a plain-text editor.",
                    "timestamp": "2024-01-01T12:00:20Z",
                },
            ],
        )
        _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id, provider="chatgpt")
        assert "# Simple ChatGPT Session" in markdown_text
        assert "Origin: chatgpt-export" in markdown_text
        assert "## user" in markdown_text
        assert "## assistant" in markdown_text
        assert "Hello, how are you?" in markdown_text
        assert "Markdown is a lightweight markup language" in markdown_text
        assert "_Timestamp: 2024-01-01T12:00:00+00:00_" in markdown_text
        assert_golden("chatgpt-simple", markdown_text)

    @pytest.mark.asyncio
    async def test_claude_with_thinking_blocks(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-claude-thinking"
        factory.create_session(
            id=conv_id,
            provider="claude-ai",
            title="Claude with Thinking",
            messages=[
                {"id": "msg1", "role": "user", "text": "What is 2+2?", "timestamp": "2024-01-15T10:00:00Z"},
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "<thinking>\nThis is a simple arithmetic question. 2+2 equals 4.\n</thinking>\n\nThe answer is 4.",
                    "timestamp": "2024-01-15T10:00:05Z",
                },
            ],
        )
        _conv, markdown_text = await _format_native(
            workspace_env["archive_root"], db_path, conv_id, provider="claude-ai"
        )
        assert "<thinking>" in markdown_text
        assert "</thinking>" in markdown_text
        assert "This is a simple arithmetic question" in markdown_text
        assert "The answer is 4" in markdown_text
        assert_golden("claude-thinking", markdown_text)

    @pytest.mark.asyncio
    async def test_json_tool_use_formatted(self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-tool-use"
        factory.create_session(
            id=conv_id,
            provider="claude-code",
            title="Tool Use Example",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "List files in current directory",
                    "timestamp": "2024-02-01T09:00:00Z",
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": '{"tool": "bash", "command": "ls -la"}',
                    "timestamp": "2024-02-01T09:00:03Z",
                },
            ],
        )
        _conv, markdown_text = await _format_native(
            workspace_env["archive_root"], db_path, conv_id, provider="claude-code"
        )
        assert "```json" in markdown_text
        assert '"tool": "bash"' in markdown_text
        assert "```" in markdown_text
        assert_golden("tool-use-json", markdown_text)

    @pytest.mark.asyncio
    async def test_empty_messages_skipped(self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-empty-messages"
        factory.create_session(
            id=conv_id,
            provider="chatgpt",
            title="Session with Empty Messages",
            messages=[
                {"id": "msg1", "role": "user", "text": "Hello", "timestamp": "2024-03-01T14:00:00Z"},
                {"id": "msg2", "role": "assistant", "text": "", "timestamp": "2024-03-01T14:00:01Z"},
                {"id": "msg3", "role": "user", "text": "Are you there?", "timestamp": "2024-03-01T14:00:10Z"},
                {"id": "msg4", "role": "assistant", "text": "Yes, I'm here!", "timestamp": "2024-03-01T14:00:15Z"},
            ],
        )
        _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id, provider="chatgpt")
        assistant_count = markdown_text.count("## assistant")
        assert assistant_count == 1, f"Expected 1 assistant section, got {assistant_count}"
        assert_golden("empty-messages", markdown_text)

    @pytest.mark.asyncio
    async def test_unicode_content_preserved(self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-unicode"
        factory.create_session(
            id=conv_id,
            provider="chatgpt",
            title="Unicode Test: \u4f60\u597d\u4e16\u754c \U0001f30d",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "Hello in Chinese: \u4f60\u597d",
                    "timestamp": "2024-04-01T08:00:00Z",
                },
                {
                    "id": "msg2",
                    "role": "assistant",
                    "text": "\u4f60\u597d! That means 'hello' in Chinese. \U0001f1e8\U0001f1f3",
                    "timestamp": "2024-04-01T08:00:05Z",
                },
                {
                    "id": "msg3",
                    "role": "user",
                    "text": "Math symbols: \u2211, \u222b, \u221a, \u03c0, \u221e",
                    "timestamp": "2024-04-01T08:00:15Z",
                },
            ],
        )
        _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id, provider="chatgpt")
        assert "\u4f60\u597d\u4e16\u754c \U0001f30d" in markdown_text
        assert "\u4f60\u597d" in markdown_text
        assert "\U0001f1e8\U0001f1f3" in markdown_text
        assert "\u2211, \u222b, \u221a, \u03c0, \u221e" in markdown_text
        assert_golden("unicode", markdown_text)

    @pytest.mark.asyncio
    async def test_attachments_formatted_as_links(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        factory = DbFactory(db_path)
        conv_id = "golden-attachments"
        factory.create_session(
            id=conv_id,
            provider="chatgpt",
            title="Session with Attachments",
            messages=[
                {
                    "id": "msg1",
                    "role": "user",
                    "text": "Here's a screenshot",
                    "timestamp": "2024-05-01T11:00:00Z",
                    "attachments": [
                        {
                            "id": "att1",
                            "mime_type": "image/png",
                            "size_bytes": 12345,
                            "meta": {"name": "screenshot.png"},
                        },
                    ],
                },
            ],
        )
        _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id, provider="chatgpt")
        assert "- Attachment:" in markdown_text
        has_name_or_id = "screenshot.png" in markdown_text or "att1" in markdown_text
        assert has_name_or_id, "Attachment reference not found in output"
        assert_golden("attachments", markdown_text)

    @pytest.mark.asyncio
    async def test_message_ordering_by_source_position(
        self, tmp_path: Path, workspace_env: WorkspaceEnv, db_path: Path
    ) -> None:
        # renders in source/position order, never re-sorted by
        # timestamp (#1743). The messages are seeded Third, First, Second with
        # disagreeing timestamps; the rendered order must match the source order.
        factory = DbFactory(db_path)
        conv_id = "golden-ordering"
        factory.create_session(
            id=conv_id,
            provider="chatgpt",
            title="Message Ordering Test",
            messages=[
                {"id": "msg3", "role": "user", "text": "Third message", "timestamp": "2024-01-01T12:00:30Z"},
                {"id": "msg1", "role": "user", "text": "First message", "timestamp": "2024-01-01T12:00:00Z"},
                {"id": "msg2", "role": "assistant", "text": "Second message", "timestamp": "2024-01-01T12:00:15Z"},
            ],
        )
        _conv, markdown_text = await _format_native(workspace_env["archive_root"], db_path, conv_id, provider="chatgpt")
        first_pos = markdown_text.find("First message")
        second_pos = markdown_text.find("Second message")
        third_pos = markdown_text.find("Third message")
        assert third_pos < first_pos < second_pos, "Messages should render in source/position order"
        assert_golden("ordering", markdown_text)
