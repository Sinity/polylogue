"""Conversation-level semantic projection contracts.

Low-level role/classification and record-conversion coverage lives in
``test_models.py``. This file owns the higher-level conversation/view/rendering
contracts built on top of those primitives.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pytest

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Attachment, Conversation, DialoguePair, Message
from polylogue.lib.pricing import harmonize_session_cost
from polylogue.lib.projections import ConversationProjection
from polylogue.lib.roles import Role
from tests.infra.assertions import assert_contains_all, assert_not_contains_any
from tests.infra.builders import make_conv, make_msg


@dataclass(frozen=True)
class ViewCase:
    name: str
    messages: list[Message]
    expected_ids: tuple[str, ...]
    view: str | None = None
    role_filter: tuple[Role, ...] = ()


@dataclass(frozen=True)
class RenderCase:
    name: str
    conversation: Conversation
    method: str
    expected: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    kwargs: dict[str, object] | None = None


@pytest.fixture
def substantive_pair() -> list[Message]:
    return [
        make_msg(id="u1", role="user", text="What is machine learning?"),
        make_msg(id="a1", role="assistant", text="Machine learning is a subset of AI."),
    ]


@pytest.fixture
def conversation_with_metadata() -> Conversation:
    messages = [
        make_msg(id="u1", role="user", text="Can you help with this?", provider_meta={"costUSD": 0.001}),
        make_msg(
            id="a1", role="assistant", text="Yes, I can help.", provider_meta={"costUSD": 0.005, "durationMs": 2500}
        ),
        make_msg(id="u2", role="user", text="Great, now what?", provider_meta={"costUSD": 0.001}),
        make_msg(
            id="a2",
            role="assistant",
            text="Let me explain further.",
            provider_meta={"costUSD": 0.008, "durationMs": 3000},
        ),
    ]
    return make_conv(
        id="complex-conv",
        provider="claude-ai",
        title="Complex Conversation",
        messages=MessageCollection(messages=messages),
        created_at=datetime(2024, 1, 15, 10, 0),
        updated_at=datetime(2024, 1, 15, 12, 0),
        metadata={"tags": ["test", "comprehensive"], "summary": "A test conversation"},
    )


@pytest.fixture
def dialogue_noise_mix() -> Conversation:
    messages = [
        make_msg(id="u1", role="user", text="Actual question with substance"),
        make_msg(id="a1", role="assistant", text="Actual answer with substance"),
        make_msg(id="t1", role="tool", text='{"ok": true}'),
        make_msg(id="s1", role="system", text="System prompt"),
        make_msg(
            id="a2",
            role="assistant",
            text="<thinking>Reasoning trace</thinking>",
            provider_meta={"content_blocks": [{"type": "thinking", "text": "Reasoning trace"}]},
        ),
        make_msg(
            id="a3",
            role="assistant",
            text="Calling tool",
            provider_meta={"content_blocks": [{"type": "tool_use"}]},
        ),
    ]
    return make_conv(id="mixed", provider="claude-ai", messages=MessageCollection(messages=messages))


@pytest.fixture
def projection_conversation() -> Conversation:
    return make_conv(
        id="projection",
        provider="test",
        messages=MessageCollection(
            messages=[
                make_msg(id="u1", role="user", text="First question here"),
                make_msg(id="a1", role="assistant", text="First answer here"),
                make_msg(id="u2", role="user", text="Second question here"),
                make_msg(id="a2", role="assistant", text="Second answer here"),
            ]
        ),
    )


class TestDialoguePairContracts:
    @pytest.mark.parametrize(
        ("user_role", "assistant_role", "should_pass", "error"),
        [
            ("user", "assistant", True, None),
            ("assistant", "assistant", False, "user message must have user role"),
            ("user", "system", False, "assistant message must have assistant role"),
        ],
    )
    def test_dialogue_pair_role_contract(
        self,
        user_role: str,
        assistant_role: str,
        should_pass: bool,
        error: str | None,
    ) -> None:
        user = make_msg(id="u1", role=user_role, text="Question")
        assistant = make_msg(id="a1", role=assistant_role, text="Answer")
        if should_pass:
            pair = DialoguePair(user=user, assistant=assistant)
            assert pair.user.id == "u1"
            assert pair.assistant.id == "a1"
        else:
            with pytest.raises(ValueError, match=error):
                DialoguePair(user=user, assistant=assistant)

    def test_dialogue_pair_exchange_and_semantic_payload(self) -> None:
        pair = DialoguePair(
            user=make_msg(id="u1", role="user", text="Hard problem"),
            assistant=make_msg(
                id="a1",
                role="assistant",
                text="<thinking>Complex reasoning</thinking>\nAnswer",
                provider_meta={"content_blocks": [{"type": "thinking", "text": "Complex reasoning"}]},
            ),
        )
        assert "User: Hard problem" in pair.exchange
        assert "Assistant: <thinking>Complex reasoning</thinking>" in pair.exchange
        assert pair.assistant.extract_thinking() == "Complex reasoning"


class TestMessageSemanticProjection:
    @pytest.mark.parametrize(
        ("provider_meta", "text", "expected"),
        [
            ({"content_blocks": [{"type": "thinking", "text": "step one"}]}, "visible", "step one"),
            (
                {"content_blocks": [{"type": "thinking", "text": "first"}, {"type": "thinking", "text": "second"}]},
                "visible",
                "first\n\nsecond",
            ),
            ({"isThought": True}, "gemini thinking text", "gemini thinking text"),
            ({"raw": {"content": {"content_type": "thoughts"}}}, "chatgpt thinking text", "chatgpt thinking text"),
            (None, "plain response", None),
        ],
        ids=["content_blocks", "multiple_blocks", "gemini", "chatgpt", "non_thinking"],
    )
    def test_extract_thinking_projection_contract(
        self,
        provider_meta: dict[str, object] | None,
        text: str,
        expected: str | None,
    ) -> None:
        msg = make_msg(id="m1", role="assistant", text=text, provider_meta=provider_meta)
        assert msg.extract_thinking() == expected

    @pytest.mark.parametrize(
        ("provider_meta", "expected_cost", "expected_duration"),
        [
            ({"costUSD": 0.042}, 0.042, None),
            ({"durationMs": 2500}, None, 2500),
            ({"costUSD": 0.01, "durationMs": 1000}, 0.01, 1000),
            ({"raw": {"usage": {"prompt_tokens": 10}}}, None, None),
            (None, None, None),
        ],
    )
    def test_message_metadata_projection_contract(
        self,
        provider_meta: dict[str, object] | None,
        expected_cost: float | None,
        expected_duration: int | None,
    ) -> None:
        msg = make_msg(id="m1", role="assistant", text="Response", provider_meta=provider_meta)
        assert msg.cost_usd == expected_cost
        assert msg.duration_ms == expected_duration

    def test_message_attachments_and_classification_contract(self) -> None:
        attachment = Attachment(
            id="att-1",
            name="doc.pdf",
            mime_type="application/pdf",
            size_bytes=5000,
            provider_meta={"uploaded_by": "user"},
        )
        thinking = make_msg(
            id="m-thinking",
            role="assistant",
            text="<thinking>...</thinking>",
            provider_meta={"content_blocks": [{"type": "thinking", "text": "..."}]},
        )
        tool = make_msg(
            id="m-tool",
            role="assistant",
            text="Calling tool",
            provider_meta={"content_blocks": [{"type": "tool_use"}]},
        )
        msg = make_msg(id="m-user", role="user", text="Review this", attachments=[attachment])

        assert msg.attachments[0].provider_meta is not None
        assert msg.attachments[0].provider_meta is not None
        assert msg.attachments[0].provider_meta["uploaded_by"] == "user"
        assert thinking.is_thinking is True
        assert tool.is_tool_use is True

    def test_context_wrappers_are_context_dumps(self) -> None:
        msg = make_msg(
            id="m1",
            role="user",
            text="<environment_context>\n<cwd>/workspace/polylogue</cwd>\n</environment_context>",
        )
        assert msg.is_context_dump is True

    def test_multiline_context_markers_are_context_dumps(self) -> None:
        contents_dump = make_msg(
            id="m2",
            role="user",
            text="Please inspect this.\nContents of /workspace/polylogue/README.md:\nhello",
        )
        file_path_dump = make_msg(
            id="m3",
            role="user",
            text="Captured payload:\n<file path=/workspace/polylogue/README.md>\nhello",
        )

        assert contents_dump.is_context_dump is True
        assert file_path_dump.is_context_dump is True


class TestConversationMetadataAndAggregation:
    def test_title_summary_tags_and_display_contract(self, conversation_with_metadata: Conversation) -> None:
        assert conversation_with_metadata.user_title is None
        assert conversation_with_metadata.display_title == "Complex Conversation"
        assert conversation_with_metadata.summary == "A test conversation"
        assert conversation_with_metadata.tags == ["test", "comprehensive"]

        titled = conversation_with_metadata.model_copy(
            update={"metadata": {"title": "User Override", "summary": "A test conversation", "tags": ["test"]}}
        )
        assert titled.user_title == "User Override"
        assert titled.display_title == "User Override"

        provider_labeled = conversation_with_metadata.model_copy(
            update={
                "title": "gemini-20250422-1234",
                "provider_meta": {"display_label": "Review the project plan"},
            }
        )
        assert provider_labeled.display_title == "Review the project plan"

        fallback = make_conv(id="abc123def456", provider="test", title=None, messages=MessageCollection(messages=[]))
        assert fallback.display_title == "abc123de"
        assert fallback.tags == []

    def test_cost_duration_branch_and_equality_contract(self, conversation_with_metadata: Conversation) -> None:
        assert conversation_with_metadata.total_cost_usd == 0.015
        assert conversation_with_metadata.total_duration_ms == 5500

        branched = make_conv(
            id="branchy",
            provider="test",
            messages=MessageCollection(
                messages=[
                    make_msg(id="u1", role="user", text="First question?"),
                    make_msg(id="a1", role="assistant", text="First answer."),
                    make_msg(id="u2", role="user", text="Second question?"),
                    make_msg(id="a2", role="assistant", text="Second answer."),
                    make_msg(id="u3", role="user", text="Follow-up?"),
                    make_msg(id="a3", role="assistant", text="Follow-up answer."),
                ]
            ),
        )
        assert branched.user_message_count == 3
        assert branched.assistant_message_count == 3
        assert conversation_with_metadata.model_copy() == conversation_with_metadata

    def test_cost_duration_fall_back_to_conversation_provider_meta(self) -> None:
        conversation = make_conv(
            id="claude-code-session",
            provider="claude-code",
            messages=MessageCollection(
                messages=[
                    make_msg(id="u1", role="user", text="Question"),
                    make_msg(id="a1", role="assistant", text="Answer"),
                ]
            ),
            provider_meta={"total_cost_usd": "0.75", "total_duration_ms": "3200"},
        )

        assert conversation.total_cost_usd == pytest.approx(0.75)
        assert conversation.total_duration_ms == 3200
        assert harmonize_session_cost(conversation) == (0.75, False)


VIEW_CASES = [
    ViewCase(
        name="dialogue_only",
        messages=[
            make_msg(id="u1", role="user", text="Question one with enough detail"),
            make_msg(id="a1", role="assistant", text="Answer one with enough detail"),
            make_msg(id="s1", role="system", text="System prompt"),
            make_msg(id="t1", role="tool", text="tool"),
        ],
        expected_ids=("u1", "a1"),
        view="dialogue_only",
    ),
    ViewCase(
        name="assistant_role_filter",
        messages=[
            make_msg(id="u1", role="user", text="Question one with enough detail"),
            make_msg(id="a1", role="assistant", text="Answer one with enough detail"),
            make_msg(id="a2", role="assistant", text="Answer two with enough detail"),
        ],
        expected_ids=("a1", "a2"),
        role_filter=(Role.ASSISTANT,),
    ),
    ViewCase(
        name="without_noise",
        messages=[
            make_msg(id="u1", role="user", text="Question one with enough detail"),
            make_msg(id="a1", role="assistant", text="Answer one with enough detail"),
            make_msg(id="s1", role="system", text="System prompt"),
            make_msg(id="t1", role="tool", text="Tool result"),
        ],
        view="without_noise",
        expected_ids=("u1", "a1"),
    ),
    ViewCase(
        name="substantive_only",
        messages=[
            make_msg(id="u1", role="user", text="Question one with enough detail"),
            make_msg(id="a1", role="assistant", text="Answer one with enough detail"),
            make_msg(
                id="a2",
                role="assistant",
                text="<thinking>Reasoning</thinking>",
                provider_meta={"content_blocks": [{"type": "thinking", "text": "Reasoning"}]},
            ),
            make_msg(id="t1", role="tool", text="Tool result"),
        ],
        view="substantive_only",
        expected_ids=("u1", "a1"),
    ),
]


class TestConversationViewsAndIteration:
    @pytest.mark.parametrize("case", VIEW_CASES, ids=lambda case: case.name)
    def test_view_projection_contract(self, case: ViewCase) -> None:
        conversation = make_conv(id="c1", provider="test", messages=MessageCollection(messages=case.messages))
        if case.role_filter:
            projected = conversation.with_roles(case.role_filter)
        else:
            assert case.view is not None
            projected = getattr(conversation, case.view)()
        assert tuple(message.id for message in projected.messages) == case.expected_ids

    def test_iterators_share_projection_contract(self, dialogue_noise_mix: Conversation) -> None:
        assert [message.id for message in dialogue_noise_mix.iter_dialogue()] == ["u1", "a1", "a2", "a3"]
        assert [message.id for message in dialogue_noise_mix.iter_substantive()] == ["u1", "a1"]
        assert list(dialogue_noise_mix.iter_thinking()) == ["Reasoning trace"]

    @pytest.mark.parametrize(
        ("messages", "expected_pairs"),
        [
            (
                [
                    make_msg(id="u1", role="user", text="First question here"),
                    make_msg(id="a1", role="assistant", text="First answer here"),
                    make_msg(id="u2", role="user", text="Second question here"),
                    make_msg(id="a2", role="assistant", text="Second answer here"),
                ],
                [("u1", "a1"), ("u2", "a2")],
            ),
            (
                [
                    make_msg(id="u1", role="user", text="First question here"),
                    make_msg(id="a1", role="assistant", text="First answer here"),
                    make_msg(id="u2", role="user", text="Second question orphaned no reply"),
                ],
                [("u1", "a1")],
            ),
            (
                [
                    make_msg(id="a1", role="assistant", text="assistant substantive answer"),
                    make_msg(id="u1", role="user", text="user substantive question"),
                    make_msg(id="a2", role="assistant", text="assistant substantive reply"),
                ],
                [("u1", "a2")],
            ),
        ],
        ids=["paired", "orphan_user", "out_of_order"],
    )
    def test_iter_pairs_contract(
        self,
        messages: list[Message],
        expected_pairs: list[tuple[str, str]],
    ) -> None:
        conversation = make_conv(id="c1", provider="test", messages=MessageCollection(messages=messages))
        assert [(pair.user.id, pair.assistant.id) for pair in conversation.iter_pairs()] == expected_pairs

    def test_iter_branches_contract(self) -> None:
        conversation = make_conv(
            id="c1",
            provider="claude-ai",
            messages=MessageCollection(
                messages=[
                    make_msg(id="m1", role="assistant", text="root", parent_id=None, branch_index=0),
                    make_msg(id="m3", role="assistant", text="branch-2", parent_id="m1", branch_index=2),
                    make_msg(id="m2", role="assistant", text="branch-1", parent_id="m1", branch_index=1),
                    make_msg(id="m4", role="assistant", text="single-child", parent_id="m2", branch_index=0),
                ]
            ),
        )
        branches = list(conversation.iter_branches())
        assert len(branches) == 1
        assert branches[0][0] == "m1"
        assert [message.id for message in branches[0][1]] == ["m2", "m3"]


class TestConversationProjectionContracts:
    def test_projection_count_and_execute_contract(self, projection_conversation: Conversation) -> None:
        projection = projection_conversation.project()
        assert projection.count() == 4
        assert [message.id for message in projection.to_list()] == ["u1", "a1", "u2", "a2"]
        executed = projection.execute()
        assert isinstance(executed, Conversation)
        assert [message.id for message in executed.messages] == ["u1", "a1", "u2", "a2"]

    @pytest.mark.parametrize(
        ("projector", "expected_ids"),
        [
            (lambda p: p.limit(2), ["u1", "a1"]),
            (lambda p: p.offset(1), ["a1", "u2", "a2"]),
            (lambda p: p.reverse(), ["a2", "u2", "a1", "u1"]),
            (lambda p: p.user_messages(), ["u1", "u2"]),
        ],
    )
    def test_projection_window_contract_matrix(
        self,
        projection_conversation: Conversation,
        projector: Callable[[ConversationProjection], ConversationProjection],
        expected_ids: list[str],
    ) -> None:
        projection = projector(projection_conversation.project())
        assert [message.id for message in projection.to_list()] == expected_ids


class TestConversationRendering:
    @pytest.fixture
    def render_cases(self) -> list[RenderCase]:
        unicode_conv = make_conv(
            id="unicode",
            provider="test",
            messages=MessageCollection(
                messages=[
                    make_msg(id="u1", role="user", text="What's the meaning of 🎯?"),
                    make_msg(id="a1", role="assistant", text="It means 目的 in Japanese."),
                ]
            ),
        )
        attachment_conv = make_conv(
            id="attachments",
            provider="test",
            messages=MessageCollection(
                messages=[
                    make_msg(
                        id="u1",
                        role="user",
                        text="Here's the document",
                        attachments=[Attachment(id="att1", name="doc.pdf")],
                    ),
                    make_msg(id="a1", role="assistant", text="I'll review it"),
                ]
            ),
        )
        return [
            RenderCase(
                name="to_text_default",
                conversation=make_conv(
                    id="basic",
                    provider="test",
                    messages=MessageCollection(
                        messages=[
                            make_msg(id="u1", role="user", text="Hello"),
                            make_msg(id="a1", role="assistant", text="Hi there"),
                        ]
                    ),
                ),
                method="to_text",
                expected=("user: Hello", "assistant: Hi there"),
            ),
            RenderCase(
                name="to_text_without_roles",
                conversation=make_conv(
                    id="basic-no-role",
                    provider="test",
                    messages=MessageCollection(
                        messages=[
                            make_msg(id="u1", role="user", text="Q"),
                            make_msg(id="a1", role="assistant", text="A"),
                        ]
                    ),
                ),
                method="to_text",
                kwargs={"include_role": False},
                expected=("Q", "A"),
                excluded=("user:", "assistant:"),
            ),
            RenderCase(
                name="to_clean_text_filters_noise",
                conversation=make_conv(
                    id="clean",
                    provider="test",
                    messages=MessageCollection(
                        messages=[
                            make_msg(id="u1", role="user", text="Important question with detail"),
                            make_msg(id="s1", role="system", text="System instructions"),
                            make_msg(id="a1", role="assistant", text="Important answer with detail"),
                            make_msg(id="t1", role="tool", text="Tool output"),
                        ]
                    ),
                ),
                method="to_clean_text",
                expected=("Important question", "Important answer"),
                excluded=("System instructions", "Tool output"),
            ),
            RenderCase(name="unicode", conversation=unicode_conv, method="to_text", expected=("🎯", "目的")),
            RenderCase(
                name="attachments",
                conversation=attachment_conv,
                method="to_text",
                expected=("Here's the document", "I'll review it"),
            ),
        ]

    @pytest.mark.parametrize("include_empty", [True, False])
    def test_empty_conversation_rendering_contract(self, include_empty: bool) -> None:
        conversation = make_conv(id="empty", provider="test", messages=MessageCollection(messages=[]))
        assert conversation.to_text() == ""
        assert conversation.to_clean_text() == ""

    def test_render_contract_matrix(self, render_cases: list[RenderCase]) -> None:
        for case in render_cases:
            rendered = getattr(case.conversation, case.method)(**(case.kwargs or {}))
            assert_contains_all(rendered, *case.expected)
            if case.excluded:
                assert_not_contains_any(rendered, *case.excluded)
