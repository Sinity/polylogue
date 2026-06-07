"""Session-level semantic projection contracts.

Low-level role/classification and record-conversion coverage lives in
``test_models.py``. This file owns the higher-level session/view/rendering
contracts built on top of those primitives.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.models import Attachment, DialoguePair, Message, Session
from polylogue.archive.projection.projections import SessionProjection
from polylogue.archive.semantic.pricing import harmonize_session_cost
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
    session: Session
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
def session_with_metadata() -> Session:
    # Hydrated messages have no provider_meta (#1256). Session-level
    # total_duration_ms is sourced from the session provider_meta envelope.
    messages = [
        make_msg(id="u1", role="user", text="Can you help with this?"),
        make_msg(id="a1", role="assistant", text="Yes, I can help."),
        make_msg(id="u2", role="user", text="Great, now what?"),
        make_msg(id="a2", role="assistant", text="Let me explain further."),
    ]
    return make_conv(
        id="complex-conv",
        provider="claude-ai",
        title="Complex Session",
        messages=MessageCollection(messages=messages),
        created_at=datetime(2024, 1, 15, 10, 0),
        updated_at=datetime(2024, 1, 15, 12, 0),
        metadata={"tags": ["test", "comprehensive"], "summary": "A test session"},
        provider_meta={"total_duration_ms": 5500},
    )


@pytest.fixture
def dialogue_noise_mix() -> Session:
    messages = [
        make_msg(id="u1", role="user", text="Actual question with substance"),
        make_msg(id="a1", role="assistant", text="Actual answer with substance"),
        make_msg(id="t1", role="tool", text='{"ok": true}'),
        make_msg(id="s1", role="system", text="System prompt"),
        make_msg(
            id="a2",
            role="assistant",
            text="<thinking>Reasoning trace</thinking>",
            content_blocks=[{"type": "thinking", "text": "Reasoning trace"}],
        ),
        make_msg(
            id="a3",
            role="assistant",
            text="Calling tool",
            content_blocks=[{"type": "tool_use"}],
        ),
    ]
    return make_conv(id="mixed", provider="claude-ai", messages=MessageCollection(messages=messages))


@pytest.fixture
def projection_session() -> Session:
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
                content_blocks=[{"type": "thinking", "text": "Complex reasoning"}],
            ),
        )
        assert "User: Hard problem" in pair.exchange
        assert "Assistant: <thinking>Complex reasoning</thinking>" in pair.exchange
        assert pair.assistant.extract_thinking() == "Complex reasoning"


class TestMessageSemanticProjection:
    @pytest.mark.parametrize(
        ("content_blocks", "text", "expected"),
        [
            ([{"type": "thinking", "text": "step one"}], "visible", "step one"),
            (
                [{"type": "thinking", "text": "first"}, {"type": "thinking", "text": "second"}],
                "visible",
                "first\n\nsecond",
            ),
            ([], "<thinking>xml fallback</thinking>", "xml fallback"),
            ([], "plain response", None),
        ],
        ids=["content_blocks", "multiple_blocks", "xml_fallback", "non_thinking"],
    )
    def test_extract_thinking_projection_contract(
        self,
        content_blocks: list[dict[str, object]],
        text: str,
        expected: str | None,
    ) -> None:
        # Hydrated messages source thinking from typed content_blocks; the
        # XML fallback exists for text-only providers (#1256).
        msg = make_msg(id="m1", role="assistant", text=text, content_blocks=content_blocks)
        assert msg.extract_thinking() == expected

    def test_message_attachments_and_classification_contract(self) -> None:
        attachment = Attachment(
            id="att-1",
            name="doc.pdf",
            mime_type="application/pdf",
            size_bytes=5000,
        )
        thinking = make_msg(
            id="m-thinking",
            role="assistant",
            text="<thinking>...</thinking>",
            content_blocks=[{"type": "thinking", "text": "..."}],
        )
        tool = make_msg(
            id="m-tool",
            role="assistant",
            text="Calling tool",
            content_blocks=[{"type": "tool_use"}],
        )
        msg = make_msg(id="m-user", role="user", text="Review this", attachments=[attachment])

        assert msg.attachments[0].name == "doc.pdf"
        assert msg.attachments[0].mime_type == "application/pdf"
        assert thinking.is_thinking is True
        assert tool.is_tool_use is True

    def test_context_wrappers_are_context_dumps(self) -> None:
        # Stored message_type is the source of truth post-#839 AC #3;
        # materialization (`pipeline/materialization_runtime.py`) classifies
        # context markers and persists `message_type=CONTEXT`.
        msg = make_msg(
            id="m1",
            role="user",
            text="<environment_context>\n<cwd>/workspace/polylogue</cwd>\n</environment_context>",
            message_type=MessageType.CONTEXT,
        )
        assert msg.is_context_dump is True

    def test_multiline_context_markers_are_context_dumps(self) -> None:
        contents_dump = make_msg(
            id="m2",
            role="user",
            text="Please inspect this.\nContents of /workspace/polylogue/README.md:\nhello",
            message_type=MessageType.CONTEXT,
        )
        file_path_dump = make_msg(
            id="m3",
            role="user",
            text="Captured payload:\n<file path=/workspace/polylogue/README.md>\nhello",
            message_type=MessageType.CONTEXT,
        )

        assert contents_dump.is_context_dump is True
        assert file_path_dump.is_context_dump is True


class TestSessionMetadataAndAggregation:
    def test_title_summary_tags_and_display_contract(self, session_with_metadata: Session) -> None:
        assert session_with_metadata.user_title is None
        assert session_with_metadata.display_title == "Complex Session"
        assert session_with_metadata.summary == "A test session"
        assert session_with_metadata.tags == ["test", "comprehensive"]

        titled = session_with_metadata.model_copy(
            update={"metadata": {"title": "User Override", "summary": "A test session", "tags": ["test"]}}
        )
        assert titled.user_title == "User Override"
        assert titled.display_title == "User Override"

        provider_labeled = session_with_metadata.model_copy(
            update={
                "title": "gemini-20250422-1234",
                "provider_meta": {"display_label": "Review the project plan"},
            }
        )
        assert provider_labeled.display_title == "Review the project plan"

        fallback = make_conv(id="abc123def456", provider="test", title=None, messages=MessageCollection(messages=[]))
        assert fallback.display_title == "abc123de"
        assert fallback.tags == []

    def test_cost_duration_branch_and_equality_contract(self, session_with_metadata: Session) -> None:
        # total_cost_usd is always 0.0 post-#1256 (no Message-level cost);
        # total_duration_ms reads the session provider_meta envelope.
        assert session_with_metadata.total_cost_usd == 0.0
        assert session_with_metadata.total_duration_ms == 5500

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
        assert session_with_metadata.model_copy() == session_with_metadata

    def test_total_cost_usd_does_not_fall_back_to_provider_meta(self) -> None:
        """Per #1139: Session.total_cost_usd is message-only.

        Session-level cost in ``provider_meta`` must be consumed via the
        typed ``pricing.estimate_session_cost`` /
        ``harmonize_session_cost`` extractor path, not via a silent
        runtime-property fallback.
        """

        session = make_conv(
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

        # Runtime property no longer falls back to provider_meta for cost.
        assert session.total_cost_usd == 0.0
        # Duration still falls back (out of #1139 scope).
        assert session.total_duration_ms == 3200
        # The typed extractor path still surfaces the provider-reported cost.
        assert harmonize_session_cost(session) == (0.75, False)


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
                content_blocks=[{"type": "thinking", "text": "Reasoning"}],
            ),
            make_msg(id="t1", role="tool", text="Tool result"),
        ],
        view="substantive_only",
        expected_ids=("u1", "a1"),
    ),
]


class TestSessionViewsAndIteration:
    @pytest.mark.parametrize("case", VIEW_CASES, ids=lambda case: case.name)
    def test_view_projection_contract(self, case: ViewCase) -> None:
        session = make_conv(id="c1", provider="test", messages=MessageCollection(messages=case.messages))
        if case.role_filter:
            projected = session.with_roles(case.role_filter)
        else:
            assert case.view is not None
            projected = getattr(session, case.view)()
        assert tuple(message.id for message in projected.messages) == case.expected_ids

    def test_iterators_share_projection_contract(self, dialogue_noise_mix: Session) -> None:
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
        session = make_conv(id="c1", provider="test", messages=MessageCollection(messages=messages))
        assert [(pair.user.id, pair.assistant.id) for pair in session.iter_pairs()] == expected_pairs

    def test_iter_branches_contract(self) -> None:
        session = make_conv(
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
        branches = list(session.iter_branches())
        assert len(branches) == 1
        assert branches[0][0] == "m1"
        assert [message.id for message in branches[0][1]] == ["m2", "m3"]


class TestSessionProjectionContracts:
    def test_projection_count_and_execute_contract(self, projection_session: Session) -> None:
        projection = projection_session.project()
        assert projection.count() == 4
        assert [message.id for message in projection.to_list()] == ["u1", "a1", "u2", "a2"]
        executed = projection.execute()
        assert isinstance(executed, Session)
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
        projection_session: Session,
        projector: Callable[[SessionProjection], SessionProjection],
        expected_ids: list[str],
    ) -> None:
        projection = projector(projection_session.project())
        assert [message.id for message in projection.to_list()] == expected_ids

    def test_projection_filter_helpers_cover_text_role_time_and_noise_contracts(self) -> None:
        session = make_conv(
            id="projection-rich",
            provider="claude-ai",
            messages=MessageCollection(
                messages=[
                    make_msg(
                        id="u1",
                        role="user",
                        text="Needle question with detail",
                        timestamp=datetime(2024, 1, 1, 9, 0),
                        attachments=[Attachment(id="att-1", name="spec.md")],
                    ),
                    make_msg(
                        id="a1",
                        role="assistant",
                        text="Needle answer with detail",
                        timestamp=datetime(2024, 1, 1, 9, 5),
                    ),
                    make_msg(
                        id="a2",
                        role="assistant",
                        text="Thinking step",
                        timestamp=datetime(2024, 1, 1, 9, 10),
                        content_blocks=[{"type": "thinking", "text": "step"}],
                    ),
                    make_msg(
                        id="a3",
                        role="assistant",
                        text="Calling tool",
                        timestamp=datetime(2024, 1, 1, 9, 15),
                        content_blocks=[{"type": "tool_use", "name": "Edit"}],
                    ),
                    make_msg(
                        id="s1",
                        role="system",
                        text="System notice",
                        timestamp=datetime(2024, 1, 1, 9, 20),
                    ),
                ]
            ),
        )

        assert [message.id for message in session.project().assistant_messages().to_list()] == ["a1", "a2", "a3"]
        assert [message.id for message in session.project().dialogue().to_list()] == ["u1", "a1", "a2", "a3"]
        # Post-#839 AC #3: an attachment + short text is no longer auto-classified
        # as a context dump at runtime; only stored `message_type=CONTEXT` would
        # exclude `u1` here.
        assert [message.id for message in session.project().substantive().to_list()] == ["u1", "a1"]
        assert [message.id for message in session.project().without_noise().to_list()] == ["u1", "a1", "a2"]
        assert [message.id for message in session.project().with_attachments().to_list()] == ["u1"]
        assert [message.id for message in session.project().min_words(3).to_list()] == ["u1", "a1"]
        assert [message.id for message in session.project().max_words(2).to_list()] == ["a2", "a3", "s1"]
        assert [message.id for message in session.project().contains("needle").to_list()] == ["u1", "a1"]
        assert [message.id for message in session.project().contains("Needle", case_sensitive=True).to_list()] == [
            "u1",
            "a1",
        ]
        assert [message.id for message in session.project().matches(r"Needle\s+answer").to_list()] == ["a1"]
        assert [message.id for message in session.project().since(datetime(2024, 1, 1, 9, 5)).to_list()] == [
            "a1",
            "a2",
            "a3",
            "s1",
        ]
        assert [message.id for message in session.project().until(datetime(2024, 1, 1, 9, 10)).to_list()] == [
            "u1",
            "a1",
            "a2",
        ]
        assert [
            message.id
            for message in session.project().between(datetime(2024, 1, 1, 9, 5), datetime(2024, 1, 1, 9, 15)).to_list()
        ] == ["a1", "a2", "a3"]
        assert [message.id for message in session.project().thinking_only().to_list()] == ["a2"]
        assert [message.id for message in session.project().tool_use_only().to_list()] == ["a3"]

    def test_projection_transform_and_terminal_helpers_cover_empty_and_render_paths(self) -> None:
        session = make_conv(
            id="projection-transform",
            provider="claude-ai",
            messages=MessageCollection(
                messages=[
                    make_msg(
                        id="u1",
                        role="user",
                        text="Alpha text",
                        attachments=[Attachment(id="att-1", name="draft.md")],
                    ),
                    make_msg(
                        id="a1",
                        role="assistant",
                        text="Very long assistant answer",
                    ),
                    make_msg(
                        id="a2",
                        role="assistant",
                        text="Thinking trace",
                        content_blocks=[{"type": "thinking", "text": "trace"}],
                    ),
                    make_msg(
                        id="a3",
                        role="assistant",
                        text="Tool output",
                        content_blocks=[{"type": "tool_use", "name": "Edit"}],
                    ),
                ]
            ),
        )

        stripped = session.project().strip_attachments().truncate_text(4, suffix="..").execute()
        stripped_messages = list(stripped.messages)
        assert stripped_messages[0].attachments == []
        assert stripped_messages[1].text == "Very.."

        assert [message.id for message in session.project().strip_tools().to_list()] == ["u1", "a1", "a2"]
        assert [message.id for message in session.project().strip_thinking().to_list()] == ["u1", "a1", "a3"]
        assert [message.id for message in session.project().strip_all().to_list()] == ["u1", "a1"]
        assert [message.id for message in session.project().first_n(2).to_list()] == ["u1", "a1"]
        assert [message.id for message in session.project().last_n(2).to_list()] == ["a3", "a2"]
        first_message = session.project().first()
        last_message = session.project().last()
        assert first_message is not None
        assert last_message is not None
        assert first_message.id == "u1"
        assert last_message.id == "a3"
        assert session.project().exists() is True
        assert session.project().to_text(separator=" | ") == (
            "user: Alpha text | assistant: Very long assistant answer | assistant: Thinking trace | assistant: Tool output"
        )
        assert session.project().to_text(include_role=False, separator=" | ") == (
            "Alpha text | Very long assistant answer | Thinking trace | Tool output"
        )
        assert session.project().limit(0).to_list() == []
        assert session.project().limit(0).first() is None
        assert session.project().limit(0).last() is None
        assert session.project().limit(0).exists() is False


class TestSessionRendering:
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
                session=make_conv(
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
                session=make_conv(
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
                session=make_conv(
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
            RenderCase(name="unicode", session=unicode_conv, method="to_text", expected=("🎯", "目的")),
            RenderCase(
                name="attachments",
                session=attachment_conv,
                method="to_text",
                expected=("Here's the document", "I'll review it"),
            ),
        ]

    @pytest.mark.parametrize("include_empty", [True, False])
    def test_empty_session_rendering_contract(self, include_empty: bool) -> None:
        session = make_conv(id="empty", provider="test", messages=MessageCollection(messages=[]))
        assert session.to_text() == ""
        assert session.to_clean_text() == ""

    def test_render_contract_matrix(self, render_cases: list[RenderCase]) -> None:
        for case in render_cases:
            rendered = getattr(case.session, case.method)(**(case.kwargs or {}))
            assert_contains_all(rendered, *case.expected)
            if case.excluded:
                assert_not_contains_any(rendered, *case.excluded)
