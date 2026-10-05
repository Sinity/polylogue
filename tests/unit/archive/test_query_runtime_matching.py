from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from polylogue.archive.actions.actions import Action
from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.predicate import (
    QueryBoolPredicate,
    QueryFieldPredicate,
    QueryFieldRef,
    QuerySequenceConstraint,
)
from polylogue.archive.query.runtime_matching import (
    action_predicate_sequence_witnesses,
    matches_action_predicate_sequence,
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_referenced_path,
    matches_tool_terms,
)
from polylogue.archive.session.domain_models import Session
from polylogue.archive.viewport.enums import ToolCategory
from polylogue.core.enums import Origin
from polylogue.core.types import SessionId


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
    timestamp: datetime | None = datetime(2026, 1, 1, tzinfo=timezone.utc),
) -> Action:
    return Action(
        action_id=f"action-{index}",
        message_id="message-1",
        timestamp=timestamp,
        sequence_index=index,
        kind=kind,
        tool_name=tool_name,
        tool_id=None,
        origin=Origin.CLAUDE_CODE_SESSION,
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


def test_matches_action_predicate_sequence_enforces_next(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session()
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT, index=0),
            _event(ToolCategory.SEARCH, index=1),
            _event(ToolCategory.SHELL, index=2),
        ),
    )
    steps = (_action_predicate("action", "file_edit"), _action_predicate("action", "shell"))

    assert matches_action_predicate_sequence(steps, session) is True
    assert matches_action_predicate_sequence(steps, session, (QuerySequenceConstraint(kind="next"),)) is False


def test_matches_action_predicate_sequence_enforces_known_within_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    steps = (_action_predicate("action", "file_edit"), _action_predicate("action", "shell"))
    constraint = (QuerySequenceConstraint(kind="within", within_ms=300_000),)

    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT, index=0, timestamp=start),
            _event(ToolCategory.SHELL, index=1, timestamp=start + timedelta(minutes=5)),
        ),
    )
    assert matches_action_predicate_sequence(steps, session, constraint) is True

    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT, index=0, timestamp=start),
            _event(ToolCategory.SHELL, index=1, timestamp=None),
        ),
    )
    assert matches_action_predicate_sequence(steps, session, constraint) is False


def test_sequence_witnesses_name_the_matching_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-memory matcher returns which actions matched, not only whether.

    Anti-vacuity: have ``action_predicate_sequence_witnesses`` keep only the
    last index of each partial tuple (the pre-4p1.7 ``candidates`` shape) and
    the binding assertion below goes red, because the first step's action can
    no longer be recovered.
    """
    session = _session()
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT, tool_name="Edit"),
            _event(ToolCategory.SHELL, tool_name="Bash", command="pytest"),
        ),
    )
    steps = (_action_predicate("action", "file_edit"), _action_predicate("action", "shell"))

    result = action_predicate_sequence_witnesses(steps, session)

    witnesses = tuple(result.witnesses)
    assert [witness.action_indices for witness in witnesses] == [(0, 1)]
    assert witnesses[0].span == (0, 1)
    assert bool(result) is True


def test_sequence_multiplicity_is_all_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two edits before two shells yield four matches, matching the SQL relation."""
    session = _session()
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT, tool_name="Edit"),
            _event(ToolCategory.FILE_EDIT, tool_name="Edit"),
            _event(ToolCategory.SHELL, tool_name="Bash", command="pytest"),
            _event(ToolCategory.SHELL, tool_name="Bash", command="ruff"),
        ),
    )
    steps = (_action_predicate("action", "file_edit"), _action_predicate("action", "shell"))

    result = action_predicate_sequence_witnesses(steps, session)

    assert sorted(witness.action_indices for witness in result.witnesses) == [(0, 2), (0, 3), (1, 2), (1, 3)]


def test_the_boolean_answer_is_derived_from_the_witnesses(monkeypatch: pytest.MonkeyPatch) -> None:
    """One matcher, two answers -- a reported match always has a witness.

    Anti-vacuity: reintroducing a separate traversal inside
    ``matches_action_predicate_sequence`` would let the two disagree; this
    pins them to the same call for both a matching and a non-matching pattern.
    """
    session = _session()
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT, tool_name="Edit"),
            _event(ToolCategory.SHELL, tool_name="Bash", command="pytest"),
        ),
    )
    hit = (_action_predicate("action", "file_edit"), _action_predicate("action", "shell"))
    miss = (_action_predicate("action", "shell"), _action_predicate("action", "search"))

    for steps in (hit, miss):
        witnesses = action_predicate_sequence_witnesses(steps, session)
        assert matches_action_predicate_sequence(steps, session) is bool(witnesses)


def test_sequence_retains_the_late_viable_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the last of 10,001 shells is immediately followed by search."""
    _patch_events(
        monkeypatch,
        (
            _event(ToolCategory.FILE_EDIT),
            *(_event(ToolCategory.SHELL, index=index) for index in range(1, 10_002)),
            _event(ToolCategory.SEARCH, index=10_002),
        ),
    )
    steps = tuple(_action_predicate("action", kind) for kind in ("file_edit", "shell", "search"))
    constraints = (QuerySequenceConstraint(), QuerySequenceConstraint(kind="next"))
    result = action_predicate_sequence_witnesses(steps, _session(), constraints)

    assert bool(result) is True
    assert [w.action_indices for w in result.witnesses] == [(0, 10_001, 10_002)]
    assert matches_action_predicate_sequence(steps, _session(), constraints) is True


def test_sequence_stream_preserves_every_completed_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_events(
        monkeypatch,
        (
            *(_event(ToolCategory.FILE_EDIT, index=index) for index in range(101)),
            *(_event(ToolCategory.SHELL, index=index) for index in range(101, 202)),
        ),
    )
    steps = (_action_predicate("action", "file_edit"), _action_predicate("action", "shell"))
    result = action_predicate_sequence_witnesses(steps, _session())

    assert iter(result.witnesses) is result.witnesses
    assert sum(1 for _ in result.witnesses) == 10_201
    assert bool(result) is True


def test_sequence_stream_has_no_recursive_pattern_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_events(monkeypatch, tuple(_event(ToolCategory.FILE_EDIT, index=index) for index in range(1_100)))
    steps = (_action_predicate("action", "file_edit"),) * 1_100
    constraints = (QuerySequenceConstraint(kind="next"),) * 1_099
    result = action_predicate_sequence_witnesses(steps, _session(), constraints)

    assert next(result.witnesses).action_indices == tuple(range(1_100))
    assert bool(result) is True
