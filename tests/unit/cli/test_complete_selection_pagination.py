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
from polylogue.surfaces.outcome import decide_outcome
from tests.infra.cli_selection import fixture_query_page
from tests.infra.daemon_operations import cli_daemon_archive

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
    seeded_root: Path, params: dict[str, object], row_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list and ranked operation payloads own the same continuation.

    Anti-vacuity: remove either producer's ``next_offset`` forwarding without
    changing this real archive test and its corresponding assertion goes red.
    """

    from polylogue.cli.lowering import lower_cli_query
    from polylogue.cli.operation_kernel import dispatch

    with cli_daemon_archive(seeded_root, monkeypatch):
        result = dispatch(
            _config(seeded_root),
            lower_cli_query(RootModeRequest.from_params(params), limit=PAGE, offset=0),
        )
        last = dispatch(
            _config(seeded_root),
            lower_cli_query(RootModeRequest.from_params(params), limit=PAGE, offset=6),
        ).value
    payload = result.value
    assert isinstance(payload, dict)
    assert payload["total"] == SEEDED_SESSIONS
    assert payload["next_offset"] == PAGE
    assert len(payload[row_key]) == PAGE
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
    with cli_daemon_archive(seeded_root, monkeypatch):
        ids = session_rows.query_complete_session_selection(_config(seeded_root), RootModeRequest.from_params({})).ids

    assert len(ids) == SEEDED_SESSIONS
    assert len(set(ids)) == SEEDED_SESSIONS


def test_complete_selection_walks_the_ordinary_over_500_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production producer/consumer pair crosses its ordinary 500-row page.

    Anti-vacuity: remove ``next_offset`` from the list operation payload and
    the first page is no longer accepted as a complete 501-row selection.
    """

    from polylogue.cli.session_rows import query_complete_session_selection
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
    with cli_daemon_archive(tmp_path, monkeypatch):
        ids = query_complete_session_selection(_config(tmp_path), RootModeRequest.from_params({"query": (TOKEN,)})).ids

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
    with cli_daemon_archive(seeded_root, monkeypatch):
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
        selection = payload["session_ids"]
        assert isinstance(selection, list)
        return {"status": "ok", "affected_count": len(selection)}

    delete_callback = getattr(query_verbs.delete_verb.callback, "__wrapped__", None)
    mark_callback = getattr(query_verbs.mark_verb.callback, "__wrapped__", None)
    assert callable(delete_callback)
    assert callable(mark_callback)
    with cli_daemon_archive(seeded_root, monkeypatch):
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
        (
            [
                {
                    "items": [{"id": "a"}],
                    "total": 1,
                    "next_offset": None,
                    "outcome": decide_outcome(matched=1, degraded=("projection_incomplete",)).to_dict(),
                }
            ],
            "selection_not_authoritative",
        ),
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
        return SimpleNamespace(
            value={
                **fixture_query_page(page),
                **({"outcome": page["outcome"]} if "outcome" in page else {}),
                "snapshot_epoch": "fixture-selected-frame",
            },
            authority={},
            envelope=None,
        )

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
    ``query_session_selection`` probe instead and this goes red.
    """

    import inspect

    from polylogue.cli import verb_cardinality

    source = inspect.getsource(verb_cardinality.resolve_session_ids_for_verb)
    assert "query_complete_session_selection" in source
    assert "query_session_selection(" not in source


@pytest.mark.parametrize("complete", [True, False])
def test_selection_refuses_user_tag_swap_with_unchanged_total(
    seeded_root: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    """Two independently correct pages cannot authorize one mixed tag population."""
    import polylogue.cli.session_rows as session_rows
    from polylogue.operations.operation_context import open_operation_read

    with open_operation_read(seeded_root) as pinned:
        ids = [row.session_id for row in pinned.archive.iter_summaries()]
    monkeypatch.setattr(session_rows, "COMPLETE_SELECTION_PAGE", PAGE)
    original = session_rows._query_page_with_authority
    pages: list[dict[str, object]] = []
    with cli_daemon_archive(seeded_root, monkeypatch) as stack:

        def tag(session_ids: list[str], *, remove: bool = False) -> None:
            result = stack.client.operation_to_completion(
                "mutation.session.tag",
                {"session_ids": session_ids, "remove_tags" if remove else "tags": ["bound-membership"]},
                archive_root=str(seeded_root),
            )
            assert result is not None and result["outcome"] == "completed"
            assert result["result"]["effect"] == "committed"

        tag(ids[:-1])

        def changed_page(
            config: Config,
            request: RootModeRequest,
            *,
            limit: int,
            offset: int,
            daemon_disabled: bool,
        ) -> tuple[dict[str, object], str]:
            payload, authority = original(config, request, limit=limit, offset=offset, daemon_disabled=daemon_disabled)
            assert isinstance(payload, dict)
            pages.append(payload)
            if len(pages) == 1:
                items = payload["items"]
                assert isinstance(items, list)
                tag([items[0]["id"]], remove=True)
                tag([ids[-1]])
            return payload, authority

        monkeypatch.setattr(session_rows, "_query_page_with_authority", changed_page)
        request = RootModeRequest.from_params({"tag": "bound-membership"})
        with pytest.raises(OperationFailedError) as refusal:
            if complete:
                session_rows.query_complete_session_selection(_config(seeded_root), request)
            else:
                session_rows.query_session_selection(_config(seeded_root), request, limit=SEEDED_SESSIONS)
    assert refusal.value.code == "query_continuation_stale"
    assert len(pages) == 2
    assert pages[0]["total"] == pages[1]["total"] == SEEDED_SESSIONS - 1
    first_epoch, second_epoch = (str(page["snapshot_epoch"]) for page in pages)
    assert first_epoch != second_epoch
    # No Index generation changed. The durable User assertion trigger alone
    # changes the selected population and therefore its continuation authority.
    first_generation, first_relations = first_epoch.rsplit(":", 1)
    second_generation, second_relations = second_epoch.rsplit(":", 1)
    assert first_generation == second_generation
    before = dict(component.split("=", 1) for component in first_relations.split(","))
    after = dict(component.split("=", 1) for component in second_relations.split(","))
    assert before.pop("assertions") != after.pop("assertions")
    assert before == after


@pytest.mark.parametrize("view", ["messages", "hooks", "dialogue", "raw"])
def test_explicit_session_read_preserves_its_text_selection(
    seeded_root: Path, monkeypatch: pytest.MonkeyPatch, view: str
) -> None:
    """An explicit identity narrows a predicate rather than bypassing it."""
    from click.testing import CliRunner

    from polylogue.cli import cli
    from polylogue.cli.verb_cardinality import EmptyCardinalityError
    from polylogue.operations.operation_context import open_operation_read

    with open_operation_read(seeded_root) as pinned:
        sid = pinned.archive.list_summaries(limit=1)[0].session_id
    with cli_daemon_archive(seeded_root, monkeypatch):
        result = CliRunner().invoke(
            cli,
            [
                "--plain",
                "--id",
                sid,
                "find",
                "absent-membership-token",
                "then",
                "read",
                "--view",
                view,
                "--format",
                "json",
            ],
        )
    assert result.exit_code == 2, result.output
    assert result.exception is not None
    assert isinstance(result.exception.__context__, EmptyCardinalityError)


def test_explicit_session_root_boolean_query_preserves_its_predicate(
    seeded_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from click.testing import CliRunner

    from polylogue.cli import cli
    from polylogue.operations.operation_context import open_operation_read

    with open_operation_read(seeded_root) as pinned:
        sid = pinned.archive.list_summaries(limit=1)[0].session_id
    with cli_daemon_archive(seeded_root, monkeypatch):
        result = CliRunner().invoke(
            cli, ["--plain", "--id", sid, "find", 'sessions where ~"absent-membership-token"', "--format", "json"]
        )
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["items"] == []


@pytest.mark.parametrize("params,row_key", [({}, "items"), ({"query": (TOKEN,)}, "hits")])
def test_missing_explicit_scope_is_a_real_pinned_empty_query(
    seeded_root: Path, monkeypatch: pytest.MonkeyPatch, params: dict[str, object], row_key: str
) -> None:
    from polylogue.cli.lowering import lower_cli_query
    from polylogue.cli.operation_kernel import dispatch
    from polylogue.cli.session_rows import query_complete_session_selection

    request = RootModeRequest.from_params({"conv_id": "codex-session:absent", **params})
    with cli_daemon_archive(seeded_root, monkeypatch):
        value = dispatch(_config(seeded_root), lower_cli_query(request, limit=PAGE, offset=0)).value
        selection = query_complete_session_selection(_config(seeded_root), request)
    assert isinstance(value, dict)
    assert value[row_key] == [] and value["total"] == 0
    assert value["outcome"]["state"] == "empty"
    assert isinstance(value["snapshot_epoch"], str) and value["snapshot_epoch"]
    assert value["next_offset"] is None
    assert selection.ids == []
    selection.require_authoritative()
