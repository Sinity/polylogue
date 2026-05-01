from __future__ import annotations

from polylogue.lib.attachment.models import Attachment
from polylogue.lib.roles import Role
from polylogue.lib.semantic.content_projection import (
    ContentProjectionSpec,
    coerce_content_projection_spec,
    project_message_content,
)
from tests.infra.builders import make_conv, make_msg


def test_projection_removes_only_file_read_payloads_when_requested() -> None:
    conversation = make_conv(
        messages=[
            make_msg(
                id="a1",
                role="assistant",
                text="Working",
                content_blocks=[
                    {"type": "text", "text": "Working"},
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "tool_id": "tool-read",
                        "tool_input": {"path": "README.md"},
                        "semantic_type": "file_read",
                    },
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "tool_id": "tool-shell",
                        "tool_input": {"command": "pytest -q"},
                        "semantic_type": "shell",
                    },
                ],
            ),
            make_msg(
                id="t1",
                role="tool",
                text="README contents",
                content_blocks=[{"type": "tool_result", "tool_id": "tool-read", "text": "README contents"}],
            ),
            make_msg(
                id="t2",
                role="tool",
                text="pytest ok",
                content_blocks=[{"type": "tool_result", "tool_id": "tool-shell", "text": "pytest ok"}],
            ),
        ]
    )

    projected = conversation.with_content_projection(ContentProjectionSpec.from_params({"no_file_reads": True}))

    assert [message.id for message in projected.messages] == ["a1", "t2"]
    texts = [message.text or "" for message in projected.messages]
    assert any("[Tool: Read]" in text for text in texts)
    assert any("pytest ok" in text for text in texts)
    assert all("README contents" not in text for text in texts)


def test_prose_only_uses_text_fallback_and_preserves_order() -> None:
    conversation = make_conv(
        messages=[
            make_msg(
                id="fallback",
                role="assistant",
                text="Alpha\n\n```python\nprint('x')\n```\n\n<thinking>step one</thinking>\n\nOmega",
            )
        ]
    )

    projected = conversation.with_content_projection(ContentProjectionSpec.prose_only())

    assert len(projected.messages) == 1
    message = next(iter(projected.messages))
    assert message.text == "Alpha\n\nOmega"


def test_projection_filters_structured_code_and_tool_outputs_without_losing_prose() -> None:
    conversation = make_conv(
        messages=[
            make_msg(
                id="mixed",
                role="assistant",
                text="ignored",
                content_blocks=[
                    {"type": "text", "text": "Plan"},
                    {"type": "code", "text": "print('x')", "language": "python"},
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "tool_id": "tool-read",
                        "tool_input": {"path": "README.md"},
                        "semantic_type": "file_read",
                    },
                    {"type": "tool_result", "tool_id": "tool-read", "text": "README body"},
                ],
            )
        ]
    )

    projected = conversation.with_content_projection(
        ContentProjectionSpec.from_params(
            {
                "no_code_blocks": True,
                "no_tool_outputs": True,
            }
        )
    )

    message = next(iter(projected.messages))
    assert message.text == "Plan\n\n[Tool: Read] `README.md`"
    assert [block["type"] for block in message.content_blocks] == ["text", "tool_use"]


def test_prose_only_drops_attachment_only_messages() -> None:
    conversation = make_conv(
        messages=[
            make_msg(
                id="attachment-only",
                role="assistant",
                text=None,
                attachments=[Attachment(id="att-1", name="README.md", path="README.md")],
            )
        ]
    )

    projected = conversation.with_content_projection(ContentProjectionSpec.prose_only())

    assert list(projected.messages) == []


def test_projection_default_coercion_returns_unfiltered_messages() -> None:
    messages = [make_msg(id="plain", text="plain")]

    assert coerce_content_projection_spec(None).is_default()
    assert coerce_content_projection_spec({"no_code_blocks": True}).include_code is False
    assert project_message_content(messages, None) == messages


def test_projection_classifies_text_blocks_tools_attachments_and_system_noise() -> None:
    messages = [
        make_msg(id="tool-text", role=Role.TOOL, text="tool output"),
        make_msg(id="system-text", role=Role.SYSTEM, text="system noise"),
        make_msg(
            id="mixed-text",
            role=Role.ASSISTANT,
            text="Lead\n\n```python\nprint('x')\n```\n\n<antml:thinking>hidden</antml:thinking>\n\nTail",
        ),
        make_msg(
            id="blocks",
            role=Role.ASSISTANT,
            text="ignored",
            content_blocks=[
                {"type": "code", "code": "print('block')"},
                {"type": "thinking", "thinking": "reason"},
                {
                    "type": "tool_use",
                    "name": "LongCommand",
                    "tool_input": {"command": "x" * 90},
                },
                {"type": "tool_use", "name": "Search", "tool_input": {"query": "q" * 70}},
                {"type": "tool_use", "name": "Grep", "tool_input": {"pattern": "needle"}},
                {"type": "tool_use", "name": "Short", "tool_input": {"mode": "fast"}},
                {"type": "tool_use"},
                {"type": "tool_result", "text": "tool body"},
                {"type": "image", "name": "plot", "url": "https://example.test/plot.png", "media_type": "image/png"},
                {"type": "custom", "content": "custom prose"},
            ],
        ),
    ]

    projected = project_message_content(messages, ContentProjectionSpec(include_system_noise=False))
    rendered = "\n\n".join(message.text or "" for message in projected)

    assert "tool output" in rendered
    assert "system noise" not in rendered
    assert "print('x')" in rendered
    assert "hidden" in rendered
    assert "`xxxxxxxx" in rendered
    assert '"qqqqqq' in rendered
    assert "`needle`" in rendered
    assert "mode=fast" in rendered
    assert "[Tool: unknown]" in rendered
    assert "tool body" in rendered
    assert "plot https://example.test/plot.png (image/png)" in rendered
    assert "custom prose" in rendered

    system_projected = project_message_content(
        [make_msg(id="system-kept", role=Role.SYSTEM, text="system noise")],
        ContentProjectionSpec(include_code=False),
    )
    assert system_projected[0].text == "system noise"

    prose_without_noise = project_message_content(
        messages,
        ContentProjectionSpec.prose_only(),
    )
    assert [message.id for message in prose_without_noise] == ["mixed-text", "blocks"]
    assert "Lead" in (prose_without_noise[0].text or "")
    assert "custom prose" in (prose_without_noise[1].text or "")
