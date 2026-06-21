from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.archive.actions.actions import Action
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.predicate import QueryBoolPredicate, QueryFieldPredicate, QueryFieldRef
from polylogue.archive.query.runtime_matching import (
    matches_action_predicate_sequence,
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_referenced_path,
    matches_tool_terms,
)
from polylogue.archive.session.domain_models import Session
from polylogue.archive.viewport.enums import ToolCategory
from polylogue.core.enums import Origin, Provider
from polylogue.types import SessionId


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
    command: str | None = None,
    output_text: str | None = None,
    search_text: str = "",
    index: int = 0,
) -> Action:
    return Action(
        action_id=f"action-{index}",
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
        command=command,
        query=None,
        url=None,
        output_text=output_text,
        search_text=search_text
        or " ".join(part for part in (kind.value, tool_name, command, output_text, *affected_paths) if part),
        raw={},
    )


def _patch_events(
    monkeypatch: pytest.MonkeyPatch,
    events: tuple[Action, ...],
) -> None:
    monkeypatch.setattr("polylogue.archive.query.runtime_matching._actions_for", lambda _session: events)


def _action_predicate(field: str, *values: str) -> QueryFieldPredicate:
    return QueryFieldPredicate(field=field, values=values).with_field_ref(
        QueryFieldRef(scope="unit", name=field, source_name=field, unit="action")
    )


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


def test_matches_action_predicate_sequence_filters_step_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT, tool_name="Edit", affected_paths=("polylogue/archive/query/expression.py",)),
            _event(
                ToolCategory.SHELL, tool_name="Bash", command="pytest", output_text="FAILED test_query_expression.py"
            ),
            _event(ToolCategory.FILE_EDIT, tool_name="Edit", affected_paths=("polylogue/archive/query/expression.py",)),
        ),
    )
    steps = (
        _action_predicate("action", "file_edit"),
        QueryBoolPredicate(
            op="and",
            children=(
                _action_predicate("tool", "bash"),
                _action_predicate("output", "failed"),
            ),
        ),
        _action_predicate("path", "archive/query"),
    )

    assert matches_action_predicate_sequence(steps, session) is True

    missed_steps = (
        _action_predicate("action", "file_edit"),
        QueryBoolPredicate(
            op="and",
            children=(
                _action_predicate("tool", "bash"),
                _action_predicate("output", "passed"),
            ),
        ),
        _action_predicate("path", "archive/query"),
    )
    assert matches_action_predicate_sequence(missed_steps, session) is False


def test_matches_action_predicate_sequence_rejects_unbound_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()
    _patch_events(monkeypatch, (_event(ToolCategory.FILE_EDIT),))

    with pytest.raises(ValueError, match="unbound query field predicate"):
        matches_action_predicate_sequence((QueryFieldPredicate(field="action", values=("file_edit",)),), session)


def test_matches_action_predicate_sequence_treats_field_values_as_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT, tool_name="Edit", affected_paths=("polylogue/archive/query/expression.py",)),
            _event(
                ToolCategory.SHELL,
                tool_name="Bash",
                command="pytest tests/unit/cli/test_query_expression.py",
                output_text="FAILED test_query_expression.py",
            ),
            _event(ToolCategory.FILE_EDIT, tool_name="Edit", affected_paths=("polylogue/archive/query/expression.py",)),
        ),
    )
    steps = (
        _action_predicate("action", "file_edit"),
        QueryBoolPredicate(
            op="and",
            children=(
                _action_predicate("command", "ruff", "pytest"),
                _action_predicate("output", "error", "failed"),
            ),
        ),
        _action_predicate("path", "missing/path", "archive/query"),
    )

    assert matches_action_predicate_sequence(steps, session) is True
