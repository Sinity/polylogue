"""Golden/snapshot contracts for markdown rendering."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from polylogue.rendering.core import ConversationFormatter
from polylogue.rendering.renderers.markdown import MarkdownRenderer
from tests.infra import GOLDEN_DIR
from tests.infra.storage_records import DbFactory

RenderAssertion = Callable[[str], bool]


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"/tmp/(?:nix-shell\.[^/]+/)?pytest-of-[^)>\s]+", "$TMPDIR", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def assert_golden(name: str, actual: str) -> None:
    golden_path = GOLDEN_DIR / f"{name}.md"
    normalized = normalize_markdown(actual)
    if os.environ.get("UPDATE_GOLDEN"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(normalized)
        return
    if not golden_path.exists():
        pytest.fail(
            f"Golden file not found: {golden_path}\n"
            "Run with UPDATE_GOLDEN=1 to generate golden files."
        )
    expected = normalize_markdown(golden_path.read_text())
    assert normalized == expected, (
        f"Rendered output differs from golden file: {golden_path}\n"
        "Run with UPDATE_GOLDEN=1 to regenerate after intentional changes."
    )


@dataclass(frozen=True)
class GoldenRenderCase:
    name: str
    conversation_id: str
    provider: str
    title: str
    messages: tuple[dict[str, object], ...]
    expected: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    assertion: RenderAssertion | None = None


@dataclass(frozen=True)
class RendererStructureCase:
    conversation_id: str
    provider: str
    title: str
    messages: tuple[dict[str, object], ...]


def _factory(db_path: Path) -> DbFactory:
    return DbFactory(db_path)


async def _render_markdown(factory: DbFactory, archive_root: Path, case: GoldenRenderCase) -> str:
    factory.create_conversation(
        id=case.conversation_id,
        provider=case.provider,
        title=case.title,
        messages=list(case.messages),
    )
    formatter = ConversationFormatter(archive_root)
    formatted = await formatter.format(case.conversation_id)
    return formatted.markdown_text


async def _render_path(
    factory: DbFactory,
    archive_root: Path,
    output_root: Path,
    case: RendererStructureCase,
) -> Path:
    factory.create_conversation(
        id=case.conversation_id,
        provider=case.provider,
        title=case.title,
        messages=list(case.messages),
    )
    renderer = MarkdownRenderer(archive_root)
    return await renderer.render(case.conversation_id, output_root)


_SIMPLE_CASES = (
    GoldenRenderCase(
        name="chatgpt-simple",
        conversation_id="golden-chatgpt-simple",
        provider="chatgpt",
        title="Simple ChatGPT Conversation",
        messages=(
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
        ),
        expected=(
            "# Simple ChatGPT Conversation",
            "Provider: chatgpt",
            "## user",
            "## assistant",
            "Hello, how are you?",
            "Markdown is a lightweight markup language",
            "_Timestamp: 2024-01-01T12:00:00Z_",
        ),
    ),
    GoldenRenderCase(
        name="claude-thinking",
        conversation_id="golden-claude-thinking",
        provider="claude",
        title="Claude with Thinking",
        messages=(
            {"id": "msg1", "role": "user", "text": "What is 2+2?", "timestamp": "2024-01-15T10:00:00Z"},
            {
                "id": "msg2",
                "role": "assistant",
                "text": "<thinking>\nThis is a simple arithmetic question. 2+2 equals 4.\n</thinking>\n\nThe answer is 4.",
                "timestamp": "2024-01-15T10:00:05Z",
            },
        ),
        expected=("<thinking>", "</thinking>", "This is a simple arithmetic question", "The answer is 4"),
    ),
    GoldenRenderCase(
        name="tool-use-json",
        conversation_id="golden-tool-use",
        provider="claude-code",
        title="Tool Use Example",
        messages=(
            {"id": "msg1", "role": "user", "text": "List files in current directory", "timestamp": "2024-02-01T09:00:00Z"},
            {
                "id": "msg2",
                "role": "assistant",
                "text": '{"tool": "bash", "command": "ls -la"}',
                "timestamp": "2024-02-01T09:00:03Z",
            },
        ),
        expected=("```json", '"tool": "bash"', '"command": "ls -la"'),
    ),
    GoldenRenderCase(
        name="empty-messages",
        conversation_id="golden-empty-messages",
        provider="chatgpt",
        title="Conversation with Empty Messages",
        messages=(
            {"id": "msg1", "role": "user", "text": "Hello", "timestamp": "2024-03-01T14:00:00Z"},
            {"id": "msg2", "role": "assistant", "text": "", "timestamp": "2024-03-01T14:00:01Z"},
            {"id": "msg3", "role": "user", "text": "Are you there?", "timestamp": "2024-03-01T14:00:10Z"},
            {"id": "msg4", "role": "assistant", "text": "Yes, I'm here!", "timestamp": "2024-03-01T14:00:15Z"},
        ),
        expected=("Hello", "Are you there?", "Yes, I'm here!"),
        assertion=lambda text: text.count("## assistant") == 1,
    ),
    GoldenRenderCase(
        name="unicode",
        conversation_id="golden-unicode",
        provider="chatgpt",
        title="Unicode Test: 你好世界 🌍",
        messages=(
            {"id": "msg1", "role": "user", "text": "Hello in Chinese: 你好", "timestamp": "2024-04-01T08:00:00Z"},
            {
                "id": "msg2",
                "role": "assistant",
                "text": "你好! That means 'hello' in Chinese. 🇨🇳",
                "timestamp": "2024-04-01T08:00:05Z",
            },
            {"id": "msg3", "role": "user", "text": "Math symbols: ∑, ∫, √, π, ∞", "timestamp": "2024-04-01T08:00:15Z"},
        ),
        expected=("你好世界 🌍", "你好", "🇨🇳", "∑, ∫, √, π, ∞"),
    ),
    GoldenRenderCase(
        name="attachments",
        conversation_id="golden-attachments",
        provider="chatgpt",
        title="Conversation with Attachments",
        messages=(
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
                    }
                ],
            },
        ),
        expected=("- Attachment:",),
        assertion=lambda text: "screenshot.png" in text or "att1" in text,
    ),
    GoldenRenderCase(
        name="ordering",
        conversation_id="golden-ordering",
        provider="chatgpt",
        title="Message Ordering Test",
        messages=(
            {"id": "msg3", "role": "user", "text": "Third message", "timestamp": "2024-01-01T12:00:30Z"},
            {"id": "msg1", "role": "user", "text": "First message", "timestamp": "2024-01-01T12:00:00Z"},
            {"id": "msg2", "role": "assistant", "text": "Second message", "timestamp": "2024-01-01T12:00:15Z"},
        ),
        expected=("First message", "Second message", "Third message"),
        assertion=lambda text: text.find("First message") < text.find("Second message") < text.find("Third message"),
    ),
    GoldenRenderCase(
        name="markdown-chars",
        conversation_id="golden-markdown-chars",
        provider="chatgpt",
        title="Markdown Chars Test",
        messages=(
            {
                "id": "msg1",
                "role": "user",
                "text": "This has **bold**, *italic*, `code`, [links](url), and # headers",
                "timestamp": "2024-06-01T16:00:00Z",
            },
        ),
        expected=("**bold**", "*italic*", "`code`", "[links](url)", "# headers"),
    ),
    GoldenRenderCase(
        name="timestamp",
        conversation_id="golden-with-timestamp",
        provider="chatgpt",
        title="Timestamp Test",
        messages=(
            {"id": "msg1", "role": "user", "text": "Message with timestamp", "timestamp": "2024-01-01T12:00:00Z"},
        ),
        expected=("_Timestamp: 2024-01-01T12:00:00Z_",),
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _SIMPLE_CASES, ids=lambda case: case.name)
async def test_golden_markdown_contracts(workspace_env, db_path, case: GoldenRenderCase) -> None:
    text = await _render_markdown(_factory(db_path), workspace_env["archive_root"], case)
    for needle in case.expected:
        assert needle in text
    for needle in case.excluded:
        assert needle not in text
    if case.assertion is not None:
        assert case.assertion(text)
    assert_golden(case.name, text)


@pytest.mark.asyncio
async def test_very_long_text_not_truncated(workspace_env, db_path) -> None:
    long_text = "This is a very long message. " * 1000
    case = GoldenRenderCase(
        name="golden-long-text",
        conversation_id="golden-long-text",
        provider="chatgpt",
        title="Long Text Test",
        messages=({"id": "msg1", "role": "user", "text": long_text},),
        expected=("This is a very long message.",),
        assertion=lambda text: text.count("This is a very long message.") >= 999,
    )
    text = await _render_markdown(_factory(db_path), workspace_env["archive_root"], case)
    assert len(text) >= len(long_text)
    assert case.assertion is not None and case.assertion(text)


_STRUCTURE_CASES = (
    RendererStructureCase(
        conversation_id="test-file-structure",
        provider="chatgpt",
        title="File Structure Test",
        messages=({"id": "msg1", "role": "user", "text": "Test message"},),
    ),
    RendererStructureCase(
        conversation_id="test-conv-1",
        provider="chatgpt",
        title="Conversation 1",
        messages=({"id": "msg1", "role": "user", "text": "Message 1"},),
    ),
    RendererStructureCase(
        conversation_id="test-conv-2",
        provider="claude",
        title="Conversation 2",
        messages=({"id": "msg2", "role": "user", "text": "Message 2"},),
    ),
)


@pytest.mark.asyncio
async def test_markdown_renderer_output_path_contract(workspace_env, db_path, tmp_path) -> None:
    path = await _render_path(_factory(db_path), workspace_env["archive_root"], tmp_path, _STRUCTURE_CASES[0])
    assert path.exists()
    assert path.name == "conversation.md"
    assert "chatgpt" in str(path.parent)


@pytest.mark.asyncio
async def test_multiple_conversations_render_to_isolated_directories(workspace_env, db_path, tmp_path) -> None:
    factory = _factory(db_path)
    path1 = await _render_path(factory, workspace_env["archive_root"], tmp_path, _STRUCTURE_CASES[1])
    path2 = await _render_path(factory, workspace_env["archive_root"], tmp_path, _STRUCTURE_CASES[2])
    assert path1.parent != path2.parent
    assert path1.exists() and path2.exists()
