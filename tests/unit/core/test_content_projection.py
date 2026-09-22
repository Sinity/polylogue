from __future__ import annotations

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.semantic.content_projection import (
    ContentKind,
    ContentProjectionSpec,
    coerce_content_projection_spec,
    project_message_content,
)
from tests.infra.builders import make_conv, make_msg


def test_projection_removes_only_file_read_payloads_when_requested() -> None:
    session = make_conv(
        messages=[
            make_msg(
                id="a1",
                role="assistant",
                text="Working",
                blocks=[
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
                blocks=[{"type": "tool_result", "tool_id": "tool-read", "text": "README contents"}],
            ),
            make_msg(
                id="t2",
                role="tool",
                text="pytest ok",
                blocks=[{"type": "tool_result", "tool_id": "tool-shell", "text": "pytest ok"}],
            ),
        ]
    )

    projected = session.with_content_projection(
        ContentProjectionSpec.from_params({"exclude_content_kinds": [ContentKind.FILE_READ]})
    )

    assert [message.id for message in projected.messages] == ["a1", "t2"]
    texts = [message.text or "" for message in projected.messages]
    assert any("[Tool: Read]" in text for text in texts)
    assert any("pytest ok" in text for text in texts)
    assert all("README contents" not in text for text in texts)


def test_prose_only_uses_text_fallback_and_preserves_order() -> None:
    session = make_conv(
        messages=[
            make_msg(
                id="fallback",
                role="assistant",
                text="Alpha\n\n```python\nprint('x')\n```\n\n<thinking>step one</thinking>\n\nOmega",
            )
        ]
    )

    projected = session.with_content_projection(ContentProjectionSpec.prose_only())

    assert len(projected.messages) == 1
    message = next(iter(projected.messages))
    assert message.text == "Alpha\n\nOmega"


def test_reasoning_projection_drops_empty_inline_thinking_wrapper() -> None:
    message = make_msg(id="empty-thinking", role=Role.ASSISTANT, text="<thinking></thinking>")

    projected = project_message_content([message], ContentProjectionSpec(include_reasoning=False))

    assert projected == []


def test_reasoning_projection_uses_typed_message_without_blocks() -> None:
    message = make_msg(
        id="typed-thinking",
        role=Role.TOOL,
        text="private reasoning",
        message_type=MessageType.THINKING,
    )

    without_reasoning = project_message_content([message], ContentProjectionSpec(include_reasoning=False))
    only_reasoning = project_message_content(
        [message],
        ContentProjectionSpec.from_params({"include_content_kinds": [ContentKind.REASONING]}),
    )

    assert without_reasoning == []
    assert [projected.text for projected in only_reasoning] == ["private reasoning"]


def test_projection_filters_structured_code_and_tool_outputs_without_losing_prose() -> None:
    session = make_conv(
        messages=[
            make_msg(
                id="mixed",
                role="assistant",
                text="ignored",
                blocks=[
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

    projected = session.with_content_projection(
        ContentProjectionSpec.from_params(
            {
                "exclude_content_kinds": [ContentKind.CODE, ContentKind.TOOL_OUTPUT, ContentKind.FILE_READ],
            }
        )
    )

    message = next(iter(projected.messages))
    assert message.text == "Plan\n\n[Tool: Read] `README.md`"
    assert [block["type"] for block in message.blocks] == ["text", "tool_use"]


def test_prose_only_drops_attachment_only_messages() -> None:
    session = make_conv(
        messages=[
            make_msg(
                id="attachment-only",
                role="assistant",
                text=None,
                attachments=[Attachment(id="att-1", name="README.md", path="README.md")],
            )
        ]
    )

    projected = session.with_content_projection(ContentProjectionSpec.prose_only())

    assert list(projected.messages) == []


def test_prose_only_drops_claude_code_protocol_artifacts() -> None:
    """``prose_only`` drops messages whose stored ``message_type`` is
    PROTOCOL/CONTEXT/TOOL_RESULT and rewrites leading
    ``<system-reminder>`` blocks out of mixed prose.

    The classification is materialized at ingest by
    :func:`polylogue.archive.message.artifacts.classify_text_message_type`
    (wired into ``polylogue/sources/parsers/claude/code_parser.py``).
    The projection therefore drops these rows by stored type, not by
    re-running text heuristics — the contract from #839. A row stored
    before that contract keeps ``MessageType.MESSAGE`` and is not
    reclassified at read time.
    """
    messages = [
        make_msg(
            id="direct-user",
            role=Role.USER,
            message_type=MessageType.MESSAGE,
            text="This is a real typed request.",
            blocks=[{"type": "text", "text": "This is a real typed request."}],
        ),
        make_msg(
            id="leading-reminder",
            role=Role.USER,
            message_type=MessageType.MESSAGE,
            text="<system-reminder>model-only reminder</system-reminder>\n\nActual user prompt.",
            blocks=[
                {
                    "type": "text",
                    "text": "<system-reminder>model-only reminder</system-reminder>\n\nActual user prompt.",
                }
            ],
        ),
        make_msg(
            id="command-wrapper",
            role=Role.USER,
            message_type=MessageType.PROTOCOL,
            text="<command-name>status</command-name>\n<command-message>status</command-message>",
            blocks=[
                {
                    "type": "text",
                    "text": "<command-name>status</command-name>\n<command-message>status</command-message>",
                }
            ],
        ),
        make_msg(
            id="inline-reminder-example",
            role=Role.USER,
            message_type=MessageType.MESSAGE,
            text="Please explain literal <system-reminder> tags.",
            blocks=[{"type": "text", "text": "Please explain literal <system-reminder> tags."}],
        ),
        make_msg(
            id="local-command-caveat",
            role=Role.USER,
            message_type=MessageType.PROTOCOL,
            text=(
                "Caveat: The messages below were generated by the user while running local commands. "
                "DO NOT respond to these messages or otherwise consider them in your response unless "
                "the user explicitly asks you to."
            ),
            blocks=[
                {
                    "type": "text",
                    "text": (
                        "Caveat: The messages below were generated by the user while running local commands. "
                        "DO NOT respond to these messages or otherwise consider them in your response unless "
                        "the user explicitly asks you to."
                    ),
                }
            ],
        ),
        make_msg(
            id="task-notification",
            role=Role.USER,
            message_type=MessageType.PROTOCOL,
            text="<task-notification><status>completed</status><result>Tool payload</result></task-notification>",
            blocks=[
                {
                    "type": "text",
                    "text": (
                        "<task-notification><status>completed</status><result>Tool payload</result></task-notification>"
                    ),
                }
            ],
        ),
        make_msg(
            id="skill-body",
            role=Role.USER,
            message_type=MessageType.CONTEXT,
            text="Base directory for this skill: /home/sinity/.claude/skills/enhance\n\n# Prompt Enhancement",
            blocks=[
                {
                    "type": "text",
                    "text": (
                        "Base directory for this skill: /home/sinity/.claude/skills/enhance\n\n# Prompt Enhancement"
                    ),
                }
            ],
        ),
        make_msg(
            id="user-envelope-tool-result",
            role=Role.USER,
            text="Tool loaded.",
            message_type=MessageType.TOOL_RESULT,
        ),
    ]

    projected = project_message_content(messages, ContentProjectionSpec.prose_only())

    assert [message.id for message in projected] == ["direct-user", "leading-reminder", "inline-reminder-example"]
    assert projected[1].text == "Actual user prompt."
    assert projected[1].blocks == [{"type": "text", "text": "Actual user prompt."}]
    assert projected[2].text == "Please explain literal <system-reminder> tags."


def test_projection_default_coercion_returns_unfiltered_messages() -> None:
    messages = [make_msg(id="plain", text="plain")]

    assert coerce_content_projection_spec(None).is_default()
    assert coerce_content_projection_spec({"exclude_content_kinds": ["code"]}).include_code is False
    assert project_message_content(messages, None) == messages


def test_projection_can_include_only_named_content_kinds() -> None:
    spec = ContentProjectionSpec.from_params({"include_content_kinds": "prose,tool_call"})

    assert spec.include_prose is True
    assert spec.include_tool_calls is True
    assert spec.include_code is False
    assert spec.include_tool_outputs is False


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
            blocks=[
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


def test_reasoning_projection_suppresses_the_writer_fallback_text_block() -> None:
    """A persisted typed-thinking row must project like its pre-write self.

    A text-only ``message_type=thinking`` message reaches storage with no
    blocks, and ``archive_tiers/write.py::_message_blocks`` -- called here so
    the fixture is the production shape and not a guess -- materializes
    ``message.text`` as a plain ``text`` block.  Every hydrated read therefore
    has ``blocks`` non-empty, which used to route the fallback block through
    the ordinary prose classifier: ``include_reasoning=False`` kept the private
    reasoning after the write and dropped it before, so the projection
    disagreed with itself across the writer boundary.

    Anti-vacuity: reverting ``_segments_for_message``'s
    ``message_is_typed_thinking`` hand-off makes ``after_write`` project as
    PROSE, so ``without_reasoning`` keeps the text and the first assertion is
    red.  The reasoning-only assertions pin the opposite direction, so a
    blanket "drop every typed-thinking message" would fail too.
    """
    from polylogue.sources.parsers.base_models import ParsedMessage
    from polylogue.storage.sqlite.archive_tiers.write import _message_blocks

    text = "private chain of thought"
    parsed = ParsedMessage(
        provider_message_id="typed-thinking",
        role=Role.ASSISTANT,
        text=text,
        message_type=MessageType.THINKING,
    )
    stored_blocks = [block.model_dump(mode="json") for block in _message_blocks(parsed)]
    assert [block["type"] for block in stored_blocks] == ["text"], stored_blocks

    after_write = make_msg(
        id="after-write",
        role=Role.ASSISTANT,
        text=text,
        message_type=MessageType.THINKING,
        blocks=stored_blocks,
    )
    before_write = make_msg(
        id="before-write",
        role=Role.ASSISTANT,
        text=text,
        message_type=MessageType.THINKING,
    )

    hide = ContentProjectionSpec(include_reasoning=False)
    show = ContentProjectionSpec.from_params({"include_content_kinds": [ContentKind.REASONING]})

    assert project_message_content([after_write], hide) == []
    assert project_message_content([before_write], hide) == []
    assert [message.text for message in project_message_content([after_write], show)] == [text]
    assert [message.text for message in project_message_content([before_write], show)] == [text]


def test_typed_thinking_keeps_structural_blocks_classified_as_themselves() -> None:
    """Only the text carrier is reclassified, not the whole message.

    Anti-vacuity: widening the ``message_is_typed_thinking`` branch to every
    block type makes the tool call project as REASONING, so the tool-call
    assertion goes red.
    """
    message = make_msg(
        id="thinking-with-tool",
        role=Role.ASSISTANT,
        text="reasoning body",
        message_type=MessageType.THINKING,
        blocks=[
            {"type": "text", "text": "reasoning body"},
            {"type": "tool_use", "name": "Bash", "tool_id": "t1", "tool_input": {"command": "ls"}},
        ],
    )

    tools_only = ContentProjectionSpec.from_params({"include_content_kinds": [ContentKind.TOOL_CALL]})
    reasoning_only = ContentProjectionSpec.from_params({"include_content_kinds": [ContentKind.REASONING]})

    kept_tools = project_message_content([message], tools_only)
    assert [block["type"] for block in kept_tools[0].blocks] == ["tool_use"]
    kept_reasoning = project_message_content([message], reasoning_only)
    assert [block["type"] for block in kept_reasoning[0].blocks] == ["text"]
