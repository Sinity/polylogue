from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.archive.models import Attachment, Message, Session
from polylogue.archive.query.path_prefix import path_matches_prefix
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.runtime_filters import apply_common_filters, apply_full_filters
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.sources import origin_from_provider
from polylogue.types import Provider, SessionId
from tests.infra.builders import make_conv, make_msg


def _summary(
    session_id: str,
    *,
    provider: Provider,
    title: str,
    updated_at: datetime | None,
    tags: list[str] | None = None,
    summary: str | None = None,
    parent_id: str | None = None,
    branch_type: BranchType | None = None,
) -> SessionSummary:
    metadata: dict[str, object] = {}
    if tags is not None:
        metadata["tags"] = tags
    if summary is not None:
        metadata["summary"] = summary
    return SessionSummary(
        id=SessionId(session_id),
        origin=origin_from_provider(provider),
        title=title,
        updated_at=updated_at,
        metadata=metadata,
        parent_id=SessionId(parent_id) if parent_id is not None else None,
        branch_type=branch_type,
    )


def _session(
    session_id: str,
    *messages: Message,
    provider: Provider = Provider.CLAUDE_CODE,
    title: str = "Session",
    metadata: dict[str, object] | None = None,
    working_directories: list[str] | tuple[str, ...] | None = None,
    parent_id: str | None = None,
    branch_type: BranchType | None = None,
    updated_at: datetime | None = None,
) -> Session:
    return make_conv(
        id=session_id,
        provider=provider,
        title=title,
        metadata=metadata or {},
        working_directories=tuple(working_directories or ()),
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
        SessionQueryPlan(
            origins=("chatgpt-export", "claude-ai-export"),
            since=datetime(2026, 4, 22, tzinfo=timezone.utc),
            until=datetime(2026, 4, 24, tzinfo=timezone.utc),
            title="deploy",
            parent_id="chatgpt:alpha",
            tags=("beta",),
            session_id="claude-ai",
            has_types=("summary",),
            continuation=True,
        ),
        items,
        sql_pushed=False,
    )

    assert [str(item.id) for item in narrowed] == ["claude-ai:beta"]

    not_continuations = apply_common_filters(
        SessionQueryPlan(
            excluded_origins=("codex-session",),
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
        SessionQueryPlan(
            origins=("codex-session",),
            since=datetime(2026, 4, 25, tzinfo=timezone.utc),
            title="missing",
            # parent_id is never SQL-pushed (absent from _ArchiveFilterKwargs),
            # so it is applied as a residual filter even when sql_pushed=True
            # (#1743 follow-up). The matching parent keeps the child; the
            # pushable origins/since/title predicates are the ones being skipped.
            parent_id="chatgpt:root",
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
    branchy = _session(
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
            blocks=[{"type": "thinking", "text": "step"}],
            timestamp=datetime(2026, 4, 23, 10, 1, tzinfo=timezone.utc),
        ),
        make_msg(
            id="a2",
            role="assistant",
            text="Using tool",
            blocks=[{"type": "tool_use", "name": "Edit"}],
            branch_index=1,
            timestamp=datetime(2026, 4, 23, 10, 2, tzinfo=timezone.utc),
        ),
        metadata={"tags": ["ship"], "summary": "ship summary"},
    )
    plain = _session(
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
            "polylogue.archive.query.runtime_filters." + name,
            lambda _plan, session, *, _target="conv-branchy": str(session.id) == _target,
        )

    filtered = apply_full_filters(
        SessionQueryPlan(
            has_types=("thinking", "tools", "attachments"),
            filter_has_tool_use=True,
            filter_has_thinking=True,
            min_messages=3,
            max_messages=3,
            min_words=5,
            negative_terms=("negative",),
            has_branches=True,
            predicates=(lambda session: "branchy" in str(session.id),),
            referenced_path=("/repo/src/app.py",),
            action_terms=("shell",),
            action_sequence=("search", "shell"),
            action_text_terms=("pytest",),
            tool_terms=("bash",),
        ),
        [branchy, plain],
        sql_pushed=False,
    )

    assert [str(session.id) for session in filtered] == ["conv-branchy"]

    no_branches = apply_full_filters(
        SessionQueryPlan(
            negative_terms=("needle",),
            has_branches=False,
            max_messages=2,
            min_words=2,
        ),
        [branchy, plain],
        sql_pushed=False,
    )

    assert [str(session.id) for session in no_branches] == ["conv-plain"]


def test_apply_full_filters_handles_message_type_and_since_session_scope() -> None:
    reference_ts = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
    reference = _session(
        "session-root",
        make_msg(id="ref-user", role="user", text="root", timestamp=reference_ts),
        working_directories=["/repo"],
        updated_at=reference_ts,
    )
    later_same_repo = _session(
        "session-later",
        make_msg(
            id="later-summary",
            role="system",
            text="summary",
            timestamp=datetime(2026, 4, 23, 10, 2, tzinfo=timezone.utc),
            message_type="summary",
        ),
        working_directories=["/repo/polylogue"],
        updated_at=datetime(2026, 4, 23, 10, 2, tzinfo=timezone.utc),
    )
    later_sibling_repo = _session(
        "session-sibling",
        make_msg(
            id="sibling-summary",
            role="system",
            text="summary",
            timestamp=datetime(2026, 4, 23, 10, 4, tzinfo=timezone.utc),
            message_type="summary",
        ),
        working_directories=["/repository"],
        updated_at=datetime(2026, 4, 23, 10, 4, tzinfo=timezone.utc),
    )
    later_unknown_cwd = _session(
        "session-unknown",
        make_msg(
            id="unknown-summary",
            role="system",
            text="summary",
            timestamp=datetime(2026, 4, 23, 10, 5, tzinfo=timezone.utc),
            message_type="summary",
        ),
        working_directories=[],
        updated_at=datetime(2026, 4, 23, 10, 5, tzinfo=timezone.utc),
    )
    later_other_repo = _session(
        "session-other",
        make_msg(
            id="other-summary",
            role="system",
            text="summary",
            timestamp=datetime(2026, 4, 23, 10, 3, tzinfo=timezone.utc),
            message_type="summary",
        ),
        working_directories=["/other"],
        updated_at=datetime(2026, 4, 23, 10, 3, tzinfo=timezone.utc),
    )
    earlier_same_repo = _session(
        "session-earlier",
        make_msg(
            id="earlier-summary",
            role="system",
            text="summary",
            timestamp=datetime(2026, 4, 23, 9, 59, tzinfo=timezone.utc),
            message_type="summary",
        ),
        working_directories=["/repo"],
        updated_at=datetime(2026, 4, 23, 9, 59, tzinfo=timezone.utc),
    )

    filtered = apply_full_filters(
        SessionQueryPlan(
            message_type="summary",
            since_session_id="session-root",
        ),
        [reference, later_same_repo, later_sibling_repo, later_unknown_cwd, later_other_repo, earlier_same_repo],
        sql_pushed=False,
    )

    assert [str(session.id) for session in filtered] == ["session-later"]

    missing_reference = apply_full_filters(
        SessionQueryPlan(since_session_id="missing"),
        [reference, later_same_repo],
        sql_pushed=False,
    )
    assert missing_reference == []


def test_apply_full_filters_rejects_unknown_message_type() -> None:
    session = _session(
        "session",
        make_msg(id="msg", role="user", text="hello", message_type="message"),
    )

    with pytest.raises(ValueError, match="Unknown message type"):
        apply_full_filters(
            SessionQueryPlan(message_type="summmary"),
            [session],
            sql_pushed=False,
        )


def test_apply_full_filters_cwd_prefix_is_path_component_bounded() -> None:
    exact = _session("exact", working_directories=["/realm/project/polylogue"])
    child = _session("child", working_directories=["/realm/project/polylogue/src"])
    sibling = _session("sibling", working_directories=["/realm/project/polylogue2"])
    missing = _session("missing", working_directories=[])

    filtered = apply_full_filters(
        SessionQueryPlan(cwd_prefix="/realm/project/polylogue"),
        [exact, child, sibling, missing],
        sql_pushed=False,
    )

    assert [str(session.id) for session in filtered] == ["exact", "child"]


def test_path_prefix_matching_is_component_bounded() -> None:
    assert path_matches_prefix("/repo/foo", "/repo/foo")
    assert path_matches_prefix("/repo/foo/bar", "/repo/foo")
    assert path_matches_prefix(r"C:\\repo\\foo\\bar", r"C:\\repo\\foo")
    assert not path_matches_prefix("/repo/foobar", "/repo/foo")
    assert not path_matches_prefix("/repo/fooish/bar", "/repo/foo")
