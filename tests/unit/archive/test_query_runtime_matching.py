from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.archive.action_event.action_events import ActionEvent
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.runtime_matching import (
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_referenced_path,
    matches_tool_terms,
)
from polylogue.archive.session.domain_models import Session
from polylogue.archive.viewport.enums import ToolCategory
from polylogue.core.enums import Origin
from polylogue.types import Provider, SessionId


def _session() -> Session:
    return Session(
        id=SessionId("conv"),
        origin=Origin.CLAUDE_CODE_SESSION,
        messages=MessageCollection.empty(),
    )


def _event(
    kind: ToolCategory,
    *,
    tool_name: str = "unknown",
    affected_paths: tuple[str, ...] = (),
    search_text: str = "",
    index: int = 0,
) -> ActionEvent:
    return ActionEvent(
        event_id=f"event-{index}",
        message_id="message-1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        sequence_index=index,
        kind=kind,
        tool_name=tool_name,
        tool_id=None,
        provider=Provider.CLAUDE_CODE,
        affected_paths=affected_paths,
        cwd_path=None,
        branch_names=(),
        command=None,
        query=None,
        url=None,
        output_text=None,
        search_text=search_text,
        raw={},
    )


def _patch_events(
    monkeypatch: pytest.MonkeyPatch,
    events: tuple[ActionEvent, ...],
) -> None:
    monkeypatch.setattr("polylogue.archive.query.runtime_matching._action_events_for", lambda _session: events)


def test_matches_referenced_path_requires_each_term_across_affected_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_events(
        monkeypatch,
        (
            _event(
                ToolCategory.FILE_EDIT,
                affected_paths=("/repo/src/app.py", "/repo/tests/test_app.py"),
            ),
        ),
    )

    assert matches_referenced_path(SessionQueryPlan(referenced_path=("SRC\\APP", "tests")), _session()) is True
    assert matches_referenced_path(SessionQueryPlan(referenced_path=("missing",)), _session()) is False


def test_matches_action_and_tool_terms_handle_none_and_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.SHELL, tool_name="Bash"),
            _event(ToolCategory.FILE_EDIT, tool_name="Edit"),
        ),
    )

    assert matches_action_terms(SessionQueryPlan(action_terms=("shell",)), session) is True
    assert matches_action_terms(SessionQueryPlan(action_terms=("none",)), session) is False
    assert matches_action_terms(SessionQueryPlan(excluded_action_terms=("web",)), session) is True
    assert matches_action_terms(SessionQueryPlan(excluded_action_terms=("shell",)), session) is False

    assert matches_tool_terms(SessionQueryPlan(tool_terms=("bash",)), session) is True
    assert matches_tool_terms(SessionQueryPlan(excluded_tool_terms=("edit",)), session) is False

    _patch_events(monkeypatch, ())
    assert matches_action_terms(SessionQueryPlan(action_terms=("none",)), session) is True
    assert matches_tool_terms(SessionQueryPlan(tool_terms=("none",)), session) is True
    assert matches_tool_terms(SessionQueryPlan(excluded_tool_terms=("none",)), session) is False


def test_matches_action_sequence_and_search_text(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.SEARCH, search_text="ripgrep found schema pinning", index=0),
            _event(ToolCategory.FILE_EDIT, search_text="edited schema pinning tests", index=1),
            _event(ToolCategory.SHELL, search_text="pytest passed", index=2),
        ),
    )

    assert matches_action_sequence(SessionQueryPlan(action_sequence=("search", "shell")), session) is True
    assert matches_action_sequence(SessionQueryPlan(action_sequence=("shell", "search")), session) is False
    assert matches_action_text_terms(SessionQueryPlan(action_text_terms=("schema", "pytest")), session) is True
    assert matches_action_text_terms(SessionQueryPlan(action_text_terms=("deployment",)), session) is False
