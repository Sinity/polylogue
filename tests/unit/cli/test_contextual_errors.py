"""Selection refusals name a next action, and the chooser never blocks a pipe.

Two contracts, each with its own anti-vacuity condition.

*Next actions* — a refusal on a selection route is raised as a
:class:`ContextualCliError`, whose ``format_message`` renders the declared
actions.  :func:`test_every_contextual_error_class_declares_a_next_action`
walks the class tree, so a new subclass that declares no action is red;
:func:`test_selection_refusals_reach_the_terminal_with_a_next_action` reads the
rendered output, so a route that downgrades the typed error back to a bare
``click.UsageError`` loses the ``Next:`` block and is red too.

*Chooser gating* — :func:`test_a_non_interactive_caller_never_reaches_the_chooser`
replaces the chooser with one that fails on contact, so any path that consults
it off a terminal is red rather than silently hanging in a pipe.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner, Result

from polylogue.archive.session.domain_models import SessionSummary
from polylogue.cli import select as select_module
from polylogue.cli.click_app import cli
from polylogue.cli.contextual_errors import (
    AmbiguousSelectionError,
    ContextualCliError,
    EmptySelectionError,
    NextAction,
)
from polylogue.cli.select import resolve_ambiguous_selection
from polylogue.cli.verb_cardinality import (
    AmbiguousCardinalityError,
    CardinalityError,
    EmptyCardinalityError,
    check_cardinality,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder


def _all_subclasses(root: type) -> set[type]:
    found: set[type] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        for child in current.__subclasses__():
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def test_every_contextual_error_class_declares_a_next_action() -> None:
    """A refusal class with no declared action can only produce a dead end."""
    classes = {ContextualCliError, *_all_subclasses(ContextualCliError)}
    without = sorted(cls.__name__ for cls in classes if not cls.default_next_actions)
    assert without == []


def test_an_instance_without_explicit_actions_inherits_its_class_defaults() -> None:
    """Constructing the error is enough; raise sites do not restate the action."""
    error = EmptySelectionError("No sessions matched; cannot mark.")
    assert error.next_actions == EmptySelectionError.default_next_actions
    rendered = error.format_message()
    assert "Next:" in rendered
    assert rendered.count("\n  - ") >= 1


def test_explicit_actions_override_the_class_defaults() -> None:
    action = NextAction("Do the specific thing", "polylogue read id:abc")
    error = ContextualCliError("refused", next_actions=(action,))
    assert error.format_message().endswith("  - Do the specific thing: polylogue read id:abc")


def test_the_cardinality_guard_raises_typed_selection_errors() -> None:
    """Zero and many are distinct typed outcomes, not one shared message."""
    with pytest.raises(EmptyCardinalityError) as empty:
        check_cardinality(0, allow_all=False, first_only=False, operation="mark")
    assert isinstance(empty.value, EmptySelectionError | CardinalityError)

    with pytest.raises(AmbiguousCardinalityError) as many:
        check_cardinality(
            3,
            allow_all=False,
            first_only=False,
            operation="mark",
            candidates=("origin:a", "origin:b", "origin:c"),
        )
    rendered = many.value.format_message()
    assert "origin:a" in rendered and "origin:c" in rendered
    assert "Next:" in rendered


def test_the_ambiguity_refusal_marks_a_truncated_candidate_list() -> None:
    """A bounded list says so, so a consumer does not read it as the whole set."""
    error = AmbiguousSelectionError("matched many", candidates=("a", "b"), bounded=True)
    assert "First 2 candidates:" in error.format_message()


# ---------------------------------------------------------------------------
# Chooser gating
# ---------------------------------------------------------------------------


def _env(*, plain: bool) -> object:
    return SimpleNamespace(ui=SimpleNamespace(plain=plain))


def _summaries(*ids: str) -> list[SessionSummary]:
    return [SessionSummary(id=ref, origin="claude-ai-export", title=f"Session {ref}") for ref in ids]


def test_a_non_interactive_caller_never_reaches_the_chooser(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is checked before the chooser, not inside it."""

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the chooser ran without a terminal")

    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: False)
    monkeypatch.setattr(select_module, "choose_select_row", _explode)
    monkeypatch.setattr(select_module, "_choose_with_fzf", _explode)

    with pytest.raises(AmbiguousSelectionError) as refusal:
        resolve_ambiguous_selection(_env(plain=True), _summaries("o:1", "o:2"), operation="continue")  # type: ignore[arg-type]
    rendered = refusal.value.format_message()
    assert "o:1" in rendered and "o:2" in rendered
    assert "Next:" in rendered


def test_the_chooser_runs_and_decides_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambiguity plus a terminal is the only case that consults the chooser."""
    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: True)
    monkeypatch.setattr(
        select_module,
        "choose_select_row",
        lambda _env, rows: rows[1],
    )

    chosen = resolve_ambiguous_selection(_env(plain=False), _summaries("o:1", "o:2"), operation="continue")  # type: ignore[arg-type]
    assert chosen == "o:2"


def test_a_single_candidate_is_not_an_ambiguity(monkeypatch: pytest.MonkeyPatch) -> None:
    """One match resolves without a prompt even on a terminal."""

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the chooser ran with nothing to disambiguate")

    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: True)
    monkeypatch.setattr(select_module, "choose_select_row", _explode)

    assert resolve_ambiguous_selection(_env(plain=False), _summaries("o:1"), operation="continue") == "o:1"  # type: ignore[arg-type]


def test_a_cancelled_chooser_still_refuses_deterministically(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backing out of the prompt is a refusal, not an arbitrary pick."""
    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: True)
    monkeypatch.setattr(select_module, "choose_select_row", lambda _env, _rows: None)

    with pytest.raises(AmbiguousSelectionError):
        resolve_ambiguous_selection(_env(plain=False), _summaries("o:1", "o:2"), operation="continue")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Terminal output on the covered routes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def refusal_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An archive with several sessions one query matches at once."""
    root = tmp_path_factory.mktemp("contextual-errors")
    initialize_active_archive_root(root)
    index_db = root / "index.db"
    for index in range(3):
        (
            SessionBuilder(index_db, f"ambiguous-{index}")
            .provider("claude-ai")
            .title(f"Ambiguous session {index}")
            .git_repository_url("https://github.com/Sinity/polylogue")
            .add_message(f"m{index}", role="user", text="shared marker text")
            .save()
        )
    return root


def _invoke(archive_root: Path, argv: list[str]) -> Result:
    env = {
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_DB_PATH": str(archive_root / "index.db"),
        "POLYLOGUE_NO_DAEMON": "1",
    }
    return CliRunner().invoke(cli, ["--no-daemon", *argv], env=env)


#: Selection refusals the CLI must deliver with a next action.
_COVERED_REFUSALS: tuple[tuple[str, list[str]], ...] = (
    ("delete-without-yes", ["find", "repo:polylogue", "then", "delete"]),
    ("delete-ambiguous-dry-run", ["find", "repo:polylogue", "then", "delete", "--dry-run"]),
    ("delete-ambiguous", ["find", "repo:polylogue", "then", "delete", "--yes"]),
    ("mark-ambiguous", ["find", "repo:polylogue", "then", "mark", "--star"]),
    ("mark-exclusive-flags", ["find", "repo:polylogue", "then", "mark", "--star", "--all", "--first"]),
    ("mark-empty", ["find", "repo:no-such-repository", "then", "mark", "--star"]),
)


@pytest.mark.parametrize(
    ("name", "argv"),
    _COVERED_REFUSALS,
    ids=[name for name, _ in _COVERED_REFUSALS],
)
def test_selection_refusals_reach_the_terminal_with_a_next_action(
    refusal_archive: Path, name: str, argv: list[str]
) -> None:
    """A covered refusal never arrives as a bare sentence."""
    result = _invoke(refusal_archive, argv)
    assert result.exit_code != 0, f"{name} did not refuse: {result.output}"
    assert "Next:" in result.output, f"{name} refused without a next action: {result.output}"
    assert "\n  - " in result.output, f"{name} rendered no action line: {result.output}"


def test_an_ambiguous_refusal_lists_the_candidate_refs(refusal_archive: Path) -> None:
    """A non-interactive consumer can resolve the ambiguity from the refusal."""
    result = _invoke(refusal_archive, ["find", "repo:polylogue", "then", "delete", "--yes"])
    assert result.exit_code != 0
    assert "Candidates:" in result.output
    assert "ambiguous-0" in result.output
