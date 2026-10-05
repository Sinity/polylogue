"""Regression coverage for the shared latest-resolver helper (#1626, #1642).

Verifies the resolution rules apply uniformly:
an identity-only conv_id resolves directly, then narrowing filters, then
``None``. The single-session surfaces (``read --view messages``/``raw``/
``neighbors``, ``export``, ``analyze turns``) all route through
this helper, so a single test here pins the contract for all of them.

The resolution itself is the declared ``cli.query`` operation asked for one
row. It used to open a ``Polylogue`` facade in the CLI process and list
summaries against the local archive, so ``--latest`` answered from a second
executor no daemon ever saw; the seam these tests control is therefore the
operation's rows, not a patched ``list_summaries``.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from polylogue.cli import select as select_module
from polylogue.cli.contextual_errors import AmbiguousSelectionError
from polylogue.cli.operation_kernel import OperationUnavailableError
from polylogue.cli.select import SelectSessionRow
from polylogue.cli.session_rows import SessionSelection
from polylogue.cli.shared.helper_support import DaemonRequiredError
from polylogue.cli.shared.latest_resolver import resolve_session_id_from_root_params
from tests.infra.cli_selection import fixture_query_page, selection_for_ids, selection_for_rows


def _stub_ids(monkeypatch: pytest.MonkeyPatch, ids: Sequence[str], captured_limits: list[int] | None = None) -> None:
    def _query_session_selection(config: object, request: object, *, limit: int, **_kwargs: object) -> SessionSelection:
        if captured_limits is not None:
            captured_limits.append(limit)
        return selection_for_ids(ids)

    monkeypatch.setattr("polylogue.cli.session_rows.query_session_selection", _query_session_selection)


def _stub_rows(monkeypatch: pytest.MonkeyPatch, ids: Sequence[str], captured_limits: list[int] | None = None) -> None:
    def _query_session_selection(config: object, request: object, *, limit: int, **_kwargs: object) -> SessionSelection:
        if captured_limits is not None:
            captured_limits.append(limit)
        return selection_for_ids(ids[:limit])

    monkeypatch.setattr("polylogue.cli.session_rows.query_session_selection", _query_session_selection)


def _refuse_chooser(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the chooser ran without a terminal")

    monkeypatch.setattr(select_module, "choose_select_row", _explode)
    monkeypatch.setattr(select_module, "_choose_with_fzf", _explode)


def test_explicit_conv_id_preserves_identity_with_latest_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Latest ordering does not widen an identity-only selection."""
    limits: list[int] = []
    _stub_ids(monkeypatch, ["claude-code:explicit"], limits)
    result = resolve_session_id_from_root_params({"conv_id": "claude-code:explicit", "latest": True})
    assert result == "claude-code:explicit"
    assert limits == [1]


def test_no_filters_returns_none() -> None:
    """Empty params returns None — caller surfaces its own missing-id error."""
    assert resolve_session_id_from_root_params({}) is None


def test_latest_runs_the_operation_with_limit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--latest`` asks the declared read for one row and returns the top match.

    Anti-vacuity: drop the ``limit=1`` and the captured-limit assertion is red;
    resolve locally again and the patched operation is never consulted.
    """
    captured_limits: list[int] = []
    _stub_ids(monkeypatch, ["claude-code:latest-conv"], captured_limits)

    result = resolve_session_id_from_root_params({"latest": True})

    assert result == "claude-code:latest-conv"
    assert captured_limits == [1]


def test_filter_alone_resolves_when_match_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """A narrowing filter (provider, since, etc.) also triggers resolution."""
    _stub_rows(monkeypatch, ["codex:filtered"])

    assert resolve_session_id_from_root_params({"origin": "codex-session"}) == "codex:filtered"


def test_a_filter_matching_several_sessions_refuses_instead_of_taking_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--latest`` a multi-row filter is ambiguous, not "the top row".

    The probe asks for one row past the candidate bound so a truncated list is
    marked as such, and the chooser is never consulted without a terminal.

    Anti-vacuity: restore ``limit=1`` plus ``session_ids[0]`` for a bare filter
    and this returns ``codex:a`` instead of raising.
    """
    captured_limits: list[int] = []
    _stub_rows(monkeypatch, ["codex:a", "codex:b", "codex:c"], captured_limits)
    _refuse_chooser(monkeypatch)

    with pytest.raises(AmbiguousSelectionError) as refusal:
        resolve_session_id_from_root_params({"origin": "codex-session"}, operation="analyze turns")

    assert refusal.value.candidates == ("codex:a", "codex:b", "codex:c")
    assert refusal.value.bounded is False
    assert captured_limits == [11]


def test_a_plain_terminal_session_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--plain`` opts out of the chooser even when a terminal is attached."""
    _stub_rows(monkeypatch, ["codex:a", "codex:b"])
    _refuse_chooser(monkeypatch)
    terminal = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(select_module, "sys", SimpleNamespace(stdin=terminal, stdout=terminal))

    with pytest.raises(AmbiguousSelectionError):
        resolve_session_id_from_root_params(
            {"origin": "codex-session"},
            env=SimpleNamespace(ui=SimpleNamespace(plain=True)),  # type: ignore[arg-type]
        )


def test_the_terminal_chooser_decides_an_ambiguous_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a terminal the operator's pick is the resolved session."""
    _stub_rows(monkeypatch, ["codex:a", "codex:b"])
    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: True)
    monkeypatch.setattr(select_module, "choose_select_row", lambda _env, rows: rows[1])

    resolved = resolve_session_id_from_root_params(
        {"origin": "codex-session"},
        env=SimpleNamespace(ui=SimpleNamespace(plain=False)),  # type: ignore[arg-type]
    )

    assert resolved == "codex:b"


def _stub_hit_pages(monkeypatch: pytest.MonkeyPatch, hits: Sequence[str]) -> list[tuple[int, int]]:
    """Answer ``cli.query`` as a ranked page: block-grain hits, offset continuation."""
    requested: list[tuple[int, int]] = []

    def _query_page(
        config: object, request: object, *, limit: int, offset: int, daemon_disabled: bool
    ) -> tuple[dict[str, object], str]:
        requested.append((offset, limit))
        page = hits[offset : offset + limit]
        return fixture_query_page(
            {
                "snapshot_epoch": "fixture-selected-frame",
                "hits": [{"session": {"id": ref, "origin": "codex-session", "title": ref}} for ref in page],
                "total": len(set(hits)),
                "next_offset": offset + len(page) if offset + len(page) < len(hits) else None,
            }
        ), "fixture"

    monkeypatch.setattr("polylogue.cli.session_rows._query_page_with_authority", _query_page)
    return requested


def test_the_terminal_chooser_offers_every_match_beyond_the_display_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal display is bounded; the chooser walks the whole selection.

    Anti-vacuity: handing the chooser the bounded probe makes the last
    session unreachable and the assertion red.
    """
    from polylogue.cli.contextual_errors import AMBIGUITY_CANDIDATE_LIMIT
    from polylogue.cli.session_rows import COMPLETE_SELECTION_PAGE

    ids = [f"codex:{index:04d}" for index in range(AMBIGUITY_CANDIDATE_LIMIT + COMPLETE_SELECTION_PAGE * 2 + 3)]
    _stub_hit_pages(monkeypatch, ids)
    offered: list[int] = []

    def _choose_last(_env: object, rows: Sequence[SelectSessionRow]) -> SelectSessionRow:
        offered.append(len(rows))
        return rows[-1]

    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: True)
    monkeypatch.setattr(select_module, "choose_select_row", _choose_last)

    resolved = resolve_session_id_from_root_params(
        {"origin": "codex-session"},
        env=SimpleNamespace(ui=SimpleNamespace(plain=False)),  # type: ignore[arg-type]
    )

    assert offered == [len(ids)]
    assert resolved == ids[-1]


def test_ranked_hits_in_one_session_resolve_that_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ranked filter returns block-grain hits; several hits in one session are one session.

    ``--contains`` matching many blocks of one session used to count each hit
    as a session and refuse a selection that names exactly one.

    Anti-vacuity: count hits instead of distinct session ids and this raises
    ``AmbiguousSelectionError`` listing ``codex:only`` repeatedly.
    """
    from polylogue.cli.contextual_errors import AMBIGUITY_CANDIDATE_LIMIT

    _stub_hit_pages(monkeypatch, ["codex:only"] * (AMBIGUITY_CANDIDATE_LIMIT * 3))
    _refuse_chooser(monkeypatch)

    assert resolve_session_id_from_root_params({"origin": "codex-session"}) == "codex:only"


def test_ranked_hits_are_counted_and_offered_by_distinct_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cardinality and the chooser's rows follow sessions, past a first page of repeats.

    Anti-vacuity: stop after the first probe page and ``codex:b`` is never
    seen, so the selection resolves to ``codex:a`` alone; drop the dedupe and
    the chooser is offered ``codex:a`` more than once.
    """
    from polylogue.cli.contextual_errors import AMBIGUITY_CANDIDATE_LIMIT

    hits = ["codex:a"] * (AMBIGUITY_CANDIDATE_LIMIT + 5) + ["codex:b", "codex:a"]
    _stub_hit_pages(monkeypatch, hits)
    _refuse_chooser(monkeypatch)

    with pytest.raises(AmbiguousSelectionError) as refusal:
        resolve_session_id_from_root_params({"origin": "codex-session"}, operation="analyze turns")
    assert refusal.value.candidates == ("codex:a", "codex:b")

    offered: list[tuple[str, ...]] = []

    def _choose_first(_env: object, rows: Sequence[SelectSessionRow]) -> SelectSessionRow:
        offered.append(tuple(row.session_id for row in rows))
        return rows[0]

    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: True)
    monkeypatch.setattr(select_module, "choose_select_row", _choose_first)
    resolve_session_id_from_root_params(
        {"origin": "codex-session"},
        env=SimpleNamespace(ui=SimpleNamespace(plain=False)),  # type: ignore[arg-type]
    )
    assert offered == [("codex:a", "codex:b")]


@pytest.mark.parametrize("output_format", ["json", "ndjson"])
def test_machine_output_on_a_terminal_refuses_instead_of_prompting(
    monkeypatch: pytest.MonkeyPatch, output_format: str
) -> None:
    """``--format json`` on a TTY is a program's call: it gets the typed refusal.

    Both the root ``--format`` and a verb's own format, passed as
    ``machine_output``, suppress the chooser.

    Anti-vacuity: drop the ``machine_output`` gate in
    ``resolve_ambiguous_selection`` and the exploding chooser runs.
    """
    _stub_rows(monkeypatch, ["codex:a", "codex:b"])
    _refuse_chooser(monkeypatch)
    monkeypatch.setattr(select_module, "interactive_selection_available", lambda _env: True)
    terminal_env = SimpleNamespace(ui=SimpleNamespace(plain=False))

    with pytest.raises(AmbiguousSelectionError):
        resolve_session_id_from_root_params(
            {"origin": "codex-session", "output_format": output_format},
            env=terminal_env,  # type: ignore[arg-type]
        )
    with pytest.raises(AmbiguousSelectionError):
        resolve_session_id_from_root_params(
            {"origin": "codex-session"},
            env=terminal_env,  # type: ignore[arg-type]
            machine_output=select_module.machine_output_requested(output_format),
        )


def test_latest_returns_none_when_archive_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--latest`` against an empty archive returns None, not an error."""
    _stub_ids(monkeypatch, [])

    assert resolve_session_id_from_root_params({"latest": True}) is None


def test_latest_reports_typed_daemon_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable query keeps its operation in the CLI refusal."""

    def unavailable(*_args: object, **_kwargs: object) -> list[str]:
        raise OperationUnavailableError("daemon unavailable", operation="cli.query")

    monkeypatch.setattr("polylogue.cli.session_rows.query_session_selection", unavailable)

    with pytest.raises(DaemonRequiredError) as exc_info:
        resolve_session_id_from_root_params({"latest": True})

    assert exc_info.value.code == "daemon_required"
    assert exc_info.value.operation == "cli.query"
    assert "polylogued run" in exc_info.value.format_message()


@pytest.mark.parametrize(
    "predicate", [{"tag": "reviewed"}, {"repo": "project"}, {"query": ('sessions where ~"needle"',)}]
)
def test_explicit_id_and_predicate_resolve_the_canonical_selection(
    monkeypatch: pytest.MonkeyPatch, predicate: dict[str, object]
) -> None:
    from polylogue.cli.root_request import RootModeRequest

    seen: list[RootModeRequest] = []

    def rows(config: object, request: RootModeRequest, *, limit: int) -> SessionSelection:
        seen.append(request)
        return selection_for_rows([])

    monkeypatch.setattr("polylogue.cli.session_rows.query_session_selection", rows)
    assert resolve_session_id_from_root_params({"conv_id": "codex:one", **predicate}) is None
    assert len(seen) == 1
    spec = seen[0].query_spec()
    assert spec.session_id == "codex:one"
    assert not spec.is_exact_session_ref()
    assert spec.has_filters()
