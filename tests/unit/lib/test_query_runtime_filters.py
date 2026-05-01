from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.lib.conversation.branch_type import BranchType
from polylogue.lib.conversation.models import ConversationSummary
from polylogue.lib.models import Attachment, Conversation, Message
from polylogue.lib.query.plan import ConversationQueryPlan
from polylogue.lib.query.runtime_filters import apply_common_filters, apply_full_filters
from polylogue.types import ConversationId, Provider
from tests.infra.builders import make_conv, make_msg


def _summary(
    conversation_id: str,
    *,
    provider: Provider,
    title: str,
    updated_at: datetime | None,
    tags: list[str] | None = None,
    summary: str | None = None,
    parent_id: str | None = None,
    branch_type: BranchType | None = None,
) -> ConversationSummary:
    metadata: dict[str, object] = {}
    if tags is not None:
        metadata["tags"] = tags
    if summary is not None:
        metadata["summary"] = summary
    return ConversationSummary(
        id=ConversationId(conversation_id),
        provider=provider,
        title=title,
        updated_at=updated_at,
        metadata=metadata,
        parent_id=ConversationId(parent_id) if parent_id is not None else None,
        branch_type=branch_type,
    )


def _conversation(
    conversation_id: str,
    *messages: Message,
    provider: Provider = Provider.CLAUDE_CODE,
    title: str = "Conversation",
    metadata: dict[str, object] | None = None,
    provider_meta: dict[str, object] | None = None,
    parent_id: str | None = None,
    branch_type: BranchType | None = None,
    updated_at: datetime | None = None,
) -> Conversation:
    return make_conv(
        id=conversation_id,
        provider=provider,
        title=title,
        metadata=metadata or {},
        provider_meta=provider_meta,
        updated_at=updated_at,
        parent_id=parent_id,
        branch_type=branch_type,
        messages=list(messages),
    )


def test_apply_common_filters_respects_non_sql_fields_and_shared_selection_flags() -> None:
    items = [
        _summary(
            "chatgpt:alpha",
            provider=Provider.CHATGPT,
            title="Alpha build plan",
            updated_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
            tags=["ship", "alpha"],
            summary="ready",
        ),
        _summary(
            "claude-ai:beta",
            provider=Provider.CLAUDE_AI,
            title="Beta deploy plan",
            updated_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
            tags=["beta"],
            summary="deploy",
            parent_id="chatgpt:alpha",
            branch_type=BranchType.CONTINUATION,
        ),
        _summary(
            "codex:gamma",
            provider=Provider.CODEX,
            title="Gamma sidechain",
            updated_at=datetime(2026, 4, 23, tzinfo=timezone.utc),
            tags=["ship", "ops"],
            branch_type=BranchType.SIDECHAIN,
        ),
    ]

    narrowed = apply_common_filters(
        ConversationQueryPlan(
            providers=(Provider.CHATGPT, Provider.CLAUDE_AI),
            since=datetime(2026, 4, 22, tzinfo=timezone.utc),
            until=datetime(2026, 4, 24, tzinfo=timezone.utc),
            title="deploy",
            parent_id="chatgpt:alpha",
            tags=("beta",),
            conversation_id="claude-ai",
            has_types=("summary",),
            continuation=True,
        ),
        items,
        sql_pushed=False,
    )

    assert [str(item.id) for item in narrowed] == ["claude-ai:beta"]

    not_continuations = apply_common_filters(
        ConversationQueryPlan(
            excluded_providers=(Provider.CODEX,),
            tags=("ship",),
            excluded_tags=("ops",),
            continuation=False,
            sidechain=False,
            root=True,
        ),
        items,
        sql_pushed=False,
    )

    assert [str(item.id) for item in not_continuations] == ["chatgpt:alpha"]


def test_apply_common_filters_skips_sql_pushed_predicates_but_keeps_shared_ones() -> None:
    items = [
        _summary(
            "chatgpt:root",
            provider=Provider.CHATGPT,
            title="Wrong title",
            updated_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
            tags=["keep"],
        ),
        _summary(
            "claude-ai:child",
            provider=Provider.CLAUDE_AI,
            title="Still wrong title",
            updated_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
            tags=["keep", "drop"],
            parent_id="chatgpt:root",
        ),
    ]

    preserved = apply_common_filters(
        ConversationQueryPlan(
            providers=(Provider.CODEX,),
            since=datetime(2026, 4, 25, tzinfo=timezone.utc),
            title="missing",
            parent_id="other-parent",
            excluded_tags=("missing",),
            root=False,
        ),
        items,
        sql_pushed=True,
    )

    assert [str(item.id) for item in preserved] == ["claude-ai:child"]


def test_apply_full_filters_handles_content_word_branch_predicate_and_action_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branchy = _conversation(
        "conv-branchy",
        make_msg(
            id="u1",
            role="user",
            text="Needle question with enough words",
            attachments=[Attachment(id="att-1", name="spec.txt")],
            timestamp=datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc),
        ),
        make_msg(
            id="a1",
            role="assistant",
            text="Thinking out loud",
            provider_meta={"content_blocks": [{"type": "thinking", "text": "step"}]},
            timestamp=datetime(2026, 4, 23, 10, 1, tzinfo=timezone.utc),
        ),
        make_msg(
            id="a2",
            role="assistant",
            text="Using tool",
            content_blocks=[{"type": "tool_use", "name": "Edit"}],
            branch_index=1,
            timestamp=datetime(2026, 4, 23, 10, 2, tzinfo=timezone.utc),
        ),
        metadata={"tags": ["ship"], "summary": "ship summary"},
    )
    plain = _conversation(
        "conv-plain",
        make_msg(
            id="u2",
            role="user",
            text="boring negative term here",
            timestamp=datetime(2026, 4, 23, 9, 0, tzinfo=timezone.utc),
        ),
        make_msg(
            id="a3",
            role="assistant",
            text="short reply",
            timestamp=datetime(2026, 4, 23, 9, 1, tzinfo=timezone.utc),
        ),
    )

    for name in (
        "matches_referenced_path",
        "matches_action_terms",
        "matches_action_sequence",
        "matches_action_text_terms",
        "matches_tool_terms",
    ):
        monkeypatch.setattr(
            "polylogue.lib.query.runtime_filters." + name,
            lambda _plan, conversation, *, _target="conv-branchy": str(conversation.id) == _target,
        )

    filtered = apply_full_filters(
        ConversationQueryPlan(
            has_types=("thinking", "tools", "attachments"),
            filter_has_tool_use=True,
            filter_has_thinking=True,
            min_messages=3,
            max_messages=3,
            min_words=5,
            negative_terms=("negative",),
            has_branches=True,
            predicates=(lambda conversation: "branchy" in str(conversation.id),),
            referenced_path=("/repo/src/app.py",),
            action_terms=("shell",),
            action_sequence=("search", "shell"),
            action_text_terms=("pytest",),
            tool_terms=("bash",),
        ),
        [branchy, plain],
        sql_pushed=False,
    )

    assert [str(conversation.id) for conversation in filtered] == ["conv-branchy"]

    no_branches = apply_full_filters(
        ConversationQueryPlan(
            negative_terms=("needle",),
            has_branches=False,
            max_messages=2,
            min_words=2,
        ),
        [branchy, plain],
        sql_pushed=False,
    )

    assert [str(conversation.id) for conversation in no_branches] == ["conv-plain"]


def test_apply_full_filters_handles_message_type_and_since_session_scope() -> None:
    reference_ts = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    reference = _conversation(
        "session-root",
        make_msg(id="ref-user", role="user", text="root", timestamp=reference_ts),
        provider_meta={"working_directories": ["/repo"]},
        updated_at=reference_ts,
    )
    later_same_repo = _conversation(
        "session-later",
        make_msg(
            id="later-summary",
            role="system",
            text="summary",
            timestamp=datetime(2026, 4, 23, 10, 2, tzinfo=timezone.utc),
            message_type="summary",
        ),
        provider_meta={"working_directories": ["/repo/polylogue"]},
        updated_at=datetime(2026, 4, 23, 10, 2, tzinfo=timezone.utc),
    )
    later_other_repo = _conversation(
        "session-other",
        make_msg(
            id="other-summary",
            role="system",
            text="summary",
            timestamp=datetime(2026, 4, 23, 10, 3, tzinfo=timezone.utc),
            message_type="summary",
        ),
        provider_meta={"working_directories": ["/other"]},
        updated_at=datetime(2026, 4, 23, 10, 3, tzinfo=timezone.utc),
    )
    earlier_same_repo = _conversation(
        "session-earlier",
        make_msg(
            id="earlier-summary",
            role="system",
            text="summary",
            timestamp=datetime(2026, 4, 23, 9, 59, tzinfo=timezone.utc),
            message_type="summary",
        ),
        provider_meta={"working_directories": ["/repo"]},
        updated_at=datetime(2026, 4, 23, 9, 59, tzinfo=timezone.utc),
    )

    filtered = apply_full_filters(
        ConversationQueryPlan(
            message_type="summary",
            since_session_id="session-root",
        ),
        [reference, later_same_repo, later_other_repo, earlier_same_repo],
        sql_pushed=False,
    )

    assert [str(conversation.id) for conversation in filtered] == ["session-later"]

    unchanged = apply_full_filters(
        ConversationQueryPlan(since_session_id="missing"),
        [reference, later_same_repo],
        sql_pushed=False,
    )
    assert [str(conversation.id) for conversation in unchanged] == ["session-root", "session-later"]
