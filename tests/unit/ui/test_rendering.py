"""Compact contracts for renderer implementations and formatter projections."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.rendering.core import ConversationFormatter, FormattedConversation
from polylogue.rendering.renderers import HTMLRenderer, MarkdownRenderer, create_renderer, list_formats
from polylogue.storage.backends.connection import open_connection
from tests.infra.assertions import assert_messages_ordered
from tests.infra.storage_records import ConversationBuilder, db_setup, make_conversation, store_records


@pytest.fixture
def sample_conversation_id(workspace_env):
    conversation = (
        ConversationBuilder(db_setup(workspace_env), "test-conv-1")
        .provider("test-provider")
        .title("Test Conversation")
        .created_at("2024-01-01T10:00:00Z")
        .updated_at("2024-01-01T10:00:10Z")
        .metadata({"source": "test"})
        .add_message("msg1", role="user", text="Hello, can you help me?", timestamp="2024-01-01T10:00:00Z")
        .add_message("msg2", role="assistant", text="Of course! How can I help you today?", timestamp="2024-01-01T10:00:05Z")
        .add_message("msg3", role="user", text="I need help with Python testing", timestamp="2024-01-01T10:00:10Z")
    )
    conversation.save()
    return "test-conv-1"


@pytest.fixture
def sample_conversation_with_json(workspace_env):
    db_path = db_setup(workspace_env)
    (
        ConversationBuilder(db_path, "test-conv-json")
        .provider("test-provider")
        .title("JSON Test")
        .created_at("2024-01-01T10:00:00Z")
        .updated_at("2024-01-01T10:00:05Z")
        .metadata({"source": "test"})
        .add_message("msg1", role="user", text="Search for Python testing", timestamp="2024-01-01T10:00:00Z")
        .add_message(
            "msg2",
            role="assistant",
            text='{"query": "Python testing", "results": ["pytest", "unittest"]}',
            timestamp="2024-01-01T10:00:05Z",
        )
        .save()
    )
    return "test-conv-json"


def _renderer_config(workspace_env):
    return Config(
        archive_root=workspace_env["archive_root"],
        render_root=workspace_env["archive_root"] / "render",
        sources=[],
    )


@pytest.mark.parametrize(
    ("fmt", "renderer_cls"),
    [("markdown", MarkdownRenderer), ("html", HTMLRenderer), ("HTML", HTMLRenderer), ("MARKDOWN", MarkdownRenderer)],
)
def test_renderer_factory_contract(workspace_env, fmt: str, renderer_cls: type[MarkdownRenderer | HTMLRenderer]) -> None:
    renderer = create_renderer(fmt, _renderer_config(workspace_env))
    assert isinstance(renderer, renderer_cls)
    assert renderer.supports_format() == renderer_cls(archive_root=workspace_env["archive_root"]).supports_format()


def test_renderer_factory_rejects_unknown_format(workspace_env) -> None:
    with pytest.raises(ValueError, match="Unsupported format: json"):
        create_renderer("json", _renderer_config(workspace_env))
    assert list_formats() == ["markdown", "html"]


@pytest.mark.parametrize(
    ("renderer_cls", "suffix"),
    [(MarkdownRenderer, ".md"), (HTMLRenderer, ".html")],
)
@pytest.mark.asyncio
async def test_renderer_basic_output_contract(workspace_env, sample_conversation_id, renderer_cls, suffix: str) -> None:
    renderer = renderer_cls(archive_root=workspace_env["archive_root"])
    result_path = await renderer.render(sample_conversation_id, workspace_env["archive_root"] / "render")
    assert result_path.exists()
    assert result_path.suffix == suffix
    content = result_path.read_text()
    assert "Test Conversation" in content
    assert "Hello, can you help me?" in content
    assert "Of course! How can I help you today?" in content


@pytest.mark.parametrize(
    ("renderer_cls", "needle"),
    [(MarkdownRenderer, '"query": "Python testing"'), (HTMLRenderer, "Python testing")],
)
@pytest.mark.asyncio
async def test_renderer_json_content_contract(workspace_env, sample_conversation_with_json, renderer_cls, needle: str) -> None:
    renderer = renderer_cls(archive_root=workspace_env["archive_root"])
    result_path = await renderer.render(sample_conversation_with_json, workspace_env["archive_root"] / "render")
    assert needle in result_path.read_text()


@pytest.mark.parametrize("renderer_cls", [MarkdownRenderer, HTMLRenderer])
@pytest.mark.asyncio
async def test_renderer_missing_conversation_contract(workspace_env, renderer_cls) -> None:
    renderer = renderer_cls(archive_root=workspace_env["archive_root"])
    with pytest.raises(ValueError, match="Conversation not found"):
        await renderer.render("missing-id", workspace_env["archive_root"] / "render")


@pytest.mark.asyncio
async def test_markdown_renderer_creates_output_dirs_contract(workspace_env, sample_conversation_id) -> None:
    renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
    output_path = workspace_env["archive_root"] / "nested" / "render"
    result_path = await renderer.render(sample_conversation_id, output_path)
    assert result_path.exists()
    assert result_path.parent.exists()


@pytest.mark.asyncio
async def test_html_renderer_escapes_html_contract(workspace_env) -> None:
    db_path = db_setup(workspace_env)
    (
        ConversationBuilder(db_path, "escaped-html-conv")
        .title("<script>alert('xss')</script>")
        .add_message("m1", role="user", text="<img src=x onerror='alert(1)'>")
        .save()
    )
    renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
    result_path = await renderer.render("escaped-html-conv", workspace_env["archive_root"] / "render")
    content = result_path.read_text()
    assert "<script>" not in content
    assert "&lt;script&gt;" in content
    assert "<img" not in content
    assert "&lt;img" in content


@pytest.mark.asyncio
async def test_renderer_protocol_contract(workspace_env, sample_conversation_id) -> None:
    from polylogue.protocols import OutputRenderer

    md_renderer = MarkdownRenderer(archive_root=workspace_env["archive_root"])
    html_renderer = HTMLRenderer(archive_root=workspace_env["archive_root"])
    md_path = await md_renderer.render(sample_conversation_id, workspace_env["archive_root"] / "render")
    html_path = await html_renderer.render(sample_conversation_id, workspace_env["archive_root"] / "render")
    assert isinstance(md_renderer, OutputRenderer)
    assert isinstance(html_renderer, OutputRenderer)
    assert md_path.exists() and html_path.exists()


def test_formatted_conversation_dataclass_contract() -> None:
    fc1 = FormattedConversation(
        title="Test",
        provider="claude",
        conversation_id="c1",
        markdown_text="# Test\n\nContent",
        metadata={"message_count": 5},
    )
    fc2 = FormattedConversation(
        title="Test",
        provider="claude",
        conversation_id="c1",
        markdown_text="# Test\n\nContent",
        metadata={"message_count": 5},
    )
    assert fc1 == fc2
    assert fc1.title == "Test"
    assert fc1.provider == "claude"
    assert fc1.conversation_id == "c1"


@pytest.mark.parametrize("use_nested_path", [False, True])
def test_formatter_initialization_contract(tmp_path: Path, use_nested_path: bool) -> None:
    archive_root = tmp_path / "archive" if use_nested_path else tmp_path
    archive_root.mkdir(parents=True, exist_ok=True)
    formatter = ConversationFormatter(archive_root)
    assert formatter.archive_root == archive_root


@pytest.mark.parametrize("title", ["Basic Title", None])
@pytest.mark.asyncio
async def test_formatter_format_contract(workspace_env, title: str | None) -> None:
    db_path = db_setup(workspace_env)
    conv_id = "format-conv"
    builder = ConversationBuilder(db_path, conv_id).provider("test")
    if title is not None:
        builder.title(title)
    else:
        builder.title(None)
    builder.add_message("m1", role="user", text="Hello!").save()

    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format(conv_id)
    assert isinstance(result, FormattedConversation)
    assert result.conversation_id == conv_id
    assert result.title == (title or conv_id)
    assert "Hello!" in result.markdown_text
    assert result.metadata["message_count"] == 1


@pytest.mark.asyncio
async def test_formatter_missing_conversation_contract(workspace_env) -> None:
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_setup(workspace_env))
    with pytest.raises(ValueError, match="Conversation not found"):
        await formatter.format("missing-conv")


@pytest.mark.parametrize(
    ("conv_id", "messages", "expected_order"),
    [
        (
            "ordered-conv",
            [
                ("m3", "Third", "2024-01-01T12:00:30Z"),
                ("m1", "First", "2024-01-01T12:00:10Z"),
                ("m2", "Second", "2024-01-01T12:00:20Z"),
            ],
            ("First", "Second", "Third"),
        ),
        (
            "epoch-conv",
            [("m1", "LaterEpoch", "1704110400.5"), ("m2", "EarlierEpoch", "1704106800")],
            ("EarlierEpoch", "LaterEpoch"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_formatter_message_ordering_contract(workspace_env, conv_id: str, messages: list[tuple[str, str, str | None]], expected_order: tuple[str, ...]) -> None:
    db_path = db_setup(workspace_env)
    builder = ConversationBuilder(db_path, conv_id)
    for idx, (msg_id, text, timestamp) in enumerate(messages):
        builder.add_message(msg_id, role="user" if idx % 2 == 0 else "assistant", text=text, timestamp=timestamp)
    builder.save()

    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format(conv_id)
    assert_messages_ordered(result.markdown_text, *expected_order)


@pytest.mark.asyncio
async def test_formatter_null_timestamps_sort_last_contract(workspace_env) -> None:
    db_path = db_setup(workspace_env)
    (
        ConversationBuilder(db_path, "null-ts-conv")
        .add_message("m1", role="user", text="Timestamped", timestamp="2024-01-01T12:00:00Z")
        .add_message("m2", role="assistant", text="NoTimestamp", timestamp=None)
        .save()
    )
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format("null-ts-conv")
    assert_messages_ordered(result.markdown_text, "Timestamped", "NoTimestamp")


@pytest.mark.asyncio
async def test_formatter_records_attachment_metadata_contract(workspace_env) -> None:
    db_path = db_setup(workspace_env)
    conversation = make_conversation("attachment-conv", title="With Attachment")
    with open_connection(db_path) as conn:
        store_records(
            conversation=conversation,
            messages=[],
            attachments=[],
            conn=conn,
        )
    formatter = ConversationFormatter(workspace_env["archive_root"], db_path=db_path)
    result = await formatter.format("attachment-conv")
    assert result.metadata["message_count"] == 0
