"""The complete-selection walk over a real seeded archive (polylogue-w3s0q).

``tests/unit/cli/test_select.py::test_complete_selection_walks_every_page``
proves the *loop* terminates correctly, but it feeds the loop hand-written
pages that already carry ``next_offset``.  The production ``cli.query`` list
payload did not carry that key at all, so the loop returned after one page and
``select``/``mark``/``delete --all`` acted on the first ``COMPLETE_SELECTION_PAGE``
matches while the operator asked for every match.  These tests execute the real
declared operation against a real archive, which is the only shape that can see
the missing key.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from polylogue.cli.operation_kernel import OperationFailedError
from polylogue.cli.root_request import RootModeRequest
from polylogue.config import Config

SEEDED_SESSIONS = 7
PAGE = 3
LARGE_SEEDED_SESSIONS = 501
TOKEN = "complete-selection-token"


@pytest.fixture
def seeded_root(tmp_path: Path) -> Path:
    """An archive holding more sessions than one walk page."""

    from tests.infra.storage_records import SessionBuilder

    index_db = tmp_path / "index.db"
    now = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    for n in range(SEEDED_SESSIONS):
        stamp = (now - timedelta(minutes=n)).isoformat()
        (
            SessionBuilder(index_db, f"walk-{n}")
            .provider("codex")
            .title(f"{TOKEN} walk {n}")
            .created_at(stamp)
            .updated_at(stamp)
            .add_message(f"w{n}-1", role="user", text=f"{TOKEN} page walk question {n}")
            .add_message(f"w{n}-2", role="assistant", text=f"page walk answer {n}")
            .save()
        )
    return tmp_path


def _config(root: Path) -> Config:
    return Config(archive_root=root, db_path=root / "index.db", render_root=root / "render", sources=[])


@pytest.mark.parametrize("params,row_key", [({}, "items"), ({"query": (TOKEN,)}, "hits")])
def test_operation_payloads_carry_the_same_next_page_offset(
    seeded_root: Path, params: dict[str, object], row_key: str
) -> None:
    """The list and ranked operation payloads own the same continuation.

    Anti-vacuity: remove either producer's ``next_offset`` forwarding without
    changing this real archive test and its corresponding assertion goes red.
    """

    from polylogue.cli.lowering import lower_cli_query
    from polylogue.cli.operation_kernel import dispatch

    result = dispatch(
        _config(seeded_root),
        lower_cli_query(RootModeRequest.from_params(params), limit=PAGE, offset=0),
        daemon_disabled=True,
    )
    payload = result.value
    assert isinstance(payload, dict)
    assert payload["total"] == SEEDED_SESSIONS
    assert payload["next_offset"] == PAGE
    assert len(payload[row_key]) == PAGE

    last = dispatch(
        _config(seeded_root),
        lower_cli_query(RootModeRequest.from_params(params), limit=PAGE, offset=6),
        daemon_disabled=True,
    ).value
    assert isinstance(last, dict)
    assert last["next_offset"] is None


def test_complete_selection_resolves_every_seeded_session(seeded_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--all`` means every match, across page boundaries.

    Reproduces the bead's measurement: with the walk page set below the seeded
    count, the pre-fix walk returned ``PAGE`` ids for an archive holding
    ``SEEDED_SESSIONS``.

    Anti-vacuity: remove ``next_offset`` from the list payload and this returns
    ``PAGE`` ids instead of ``SEEDED_SESSIONS``.
    """

    import polylogue.cli.session_rows as session_rows

    monkeypatch.setattr(session_rows, "COMPLETE_SELECTION_PAGE", PAGE)
    ids = session_rows.query_complete_session_ids(
        _config(seeded_root), RootModeRequest.from_params({}), daemon_disabled=True
    )

    assert len(ids) == SEEDED_SESSIONS
    assert len(set(ids)) == SEEDED_SESSIONS


def test_complete_selection_walks_the_ordinary_over_500_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production producer/consumer pair crosses its ordinary 500-row page.

    Anti-vacuity: remove ``next_offset`` from the list operation payload and
    the first page is no longer accepted as a complete 501-row selection.
    """

    from polylogue.cli.session_rows import query_complete_session_ids
    from tests.infra.storage_records import SessionBuilder

    index_db = tmp_path / "index.db"
    for n in range(LARGE_SEEDED_SESSIONS):
        (
            SessionBuilder(index_db, f"large-walk-{n}")
            .provider("codex")
            .title(f"{TOKEN} large walk {n}")
            .add_message(f"large-{n}", role="user", text=f"{TOKEN} body {n}")
            .save()
        )

    import polylogue.cli.session_rows as session_rows

    monkeypatch.setattr(session_rows, "COMPLETE_SELECTION_PAGE", 500)
    ids = query_complete_session_ids(
        _config(tmp_path), RootModeRequest.from_params({"query": (TOKEN,)}), daemon_disabled=True
    )

    assert len(ids) == LARGE_SEEDED_SESSIONS
    assert len(set(ids)) == LARGE_SEEDED_SESSIONS


def test_complete_selection_all_verbs_receive_every_real_operation_id(
    seeded_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selection, delete --all, and mark --all share the real page consumer.

    The mutation transport is recorded because writes belong to the daemon;
    no producer, selector, or verb callback is mocked.  Anti-vacuity: make
    the list producer omit ``next_offset`` and resolution refuses before
    either recorder can receive the incomplete first page.
    """

    import polylogue.cli.session_rows as session_rows
    from polylogue.cli import query_verbs
    from polylogue.cli.verb_cardinality import resolve_session_ids_for_verb
    from tests.infra.app_env import make_app_env

    monkeypatch.setattr(session_rows, "COMPLETE_SELECTION_PAGE", PAGE)
    env = make_app_env(archive_root=seeded_root)
    request = RootModeRequest.from_params({"query": (TOKEN,)})
    expected = resolve_session_ids_for_verb(env, request)
    assert len(expected) == SEEDED_SESSIONS
    assert len(set(expected)) == SEEDED_SESSIONS

    parent = click.Context(click.Command("query"))
    parent.params = {"query_term": (TOKEN,)}
    parent.meta["polylogue_query_terms"] = (TOKEN,)
    child = click.Context(click.Command("verb"), parent=parent)
    child.obj = env

    deleted: list[list[str]] = []
    mark_operations: list[tuple[str, dict[str, object]]] = []

    def record_delete(_env: object, ids: list[str], *, force: bool, dry_run: bool = False) -> None:
        assert force is True
        assert dry_run is False
        deleted.append(ids)

    def record_mark(_env: object, operation: str, payload: dict[str, object]) -> dict[str, object]:
        mark_operations.append((operation, payload))
        return {"status": "ok", "affected_count": len(payload["session_ids"])}

    delete_callback = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
    mark_callback = getattr(query_verbs.mark_verb.callback, "__wrapped__", None)
    assert callable(delete_callback)
    assert callable(mark_callback)
    with patch("polylogue.cli.archive_query.execute_delete_by_session_ids", side_effect=record_delete):
        delete_callback(child, False, True, True, None)
    with patch("polylogue.cli.archive_query.submit_cli_mutation", side_effect=record_mark):
        mark_callback(child, ("reviewed",), (), False, False, False, False, False, False, None, True, False, None)

    assert deleted == [expected]
    assert mark_operations == [
        ("mutation.session.tag", {"session_ids": expected, "tags": ["reviewed"]}),
    ]


@pytest.mark.parametrize(
    ("pages", "expected_message"),
    [
        ([{"items": [{"id": "a"}], "total": 1}], "continuation"),
        ([{"items": [{"id": "a"}], "total": 2, "next_offset": 0}], "advance"),
        (
            [
                {"items": [{"id": "a"}], "total": 2, "next_offset": 1},
                OperationFailedError("read_failed", "later page failed"),
            ],
            "later page failed",
        ),
    ],
)
def test_mutating_selection_refuses_incomplete_or_failed_walks(pages: list[object], expected_message: str) -> None:
    """No --all mutation receives a prefix after a bad continuation.

    Anti-vacuity: return the accumulated ids on a missing/repeated
    ``next_offset`` (the pre-fix behavior) and the delete recorder is called.
    """

    from polylogue.cli import query_verbs
    from tests.infra.app_env import make_app_env

    calls = iter(pages)
    deleted: list[list[str]] = []

    def dispatch(*_args: object, **_kwargs: object) -> SimpleNamespace:
        page = next(calls)
        if isinstance(page, Exception):
            raise page
        assert isinstance(page, dict)
        return SimpleNamespace(value=page, authority={}, envelope=None)

    parent = click.Context(click.Command("query"))
    parent.params = {"query_term": (TOKEN,)}
    parent.meta["polylogue_query_terms"] = (TOKEN,)
    child = click.Context(click.Command("verb"), parent=parent)
    child.obj = make_app_env()
    callback = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
    assert callable(callback)

    with (
        patch("polylogue.cli.operation_kernel.dispatch", side_effect=dispatch),
        patch(
            "polylogue.cli.archive_query.execute_delete_by_session_ids",
            side_effect=lambda _env, ids, **_kwargs: deleted.append(ids),
        ),
        pytest.raises((click.ClickException, RuntimeError), match=expected_message),
    ):
        callback(child, False, True, True, None)

    assert deleted == []


def test_unknown_total_continues_only_on_a_full_page() -> None:
    """The shared producer rule preserves an honest unknown ranked total."""

    from polylogue.operations.daemon_reads import page_next_offset

    assert page_next_offset(offset=0, returned=PAGE, total=None, limit=PAGE) == PAGE
    assert page_next_offset(offset=PAGE, returned=1, total=None, limit=PAGE) is None
    assert page_next_offset(offset=0, returned=0, total=0, limit=PAGE) is None


def test_the_mutating_verb_route_resolves_through_the_complete_walk() -> None:
    """``select``/``mark``/``delete --all`` reach the walk above, not a page.

    ``resolve_session_ids_for_verb`` is the single route the mutating verbs and
    their cardinality guard share, so the completeness proved above is the
    completeness they get.  Pinning the delegation is what makes that transfer
    an assertion rather than a claim.

    Anti-vacuity: point ``resolve_session_ids_for_verb`` at the bounded
    ``query_session_ids`` probe instead and this goes red.
    """

    import inspect

    from polylogue.cli import verb_cardinality

    source = inspect.getsource(verb_cardinality.resolve_session_ids_for_verb)
    assert "query_complete_session_ids" in source
    assert "query_session_ids(" not in source
