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

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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


def _seeded_session_ids(root: Path) -> set[str]:
    from polylogue.storage.sqlite.connection_profile import readonly_connection_context

    with readonly_connection_context(root / "index.db") as index:
        return {str(row[0]) for row in index.execute("SELECT session_id FROM sessions")}


def _tag_targets(root: Path, tag: str) -> set[str]:
    from polylogue.core.enums import AssertionKind
    from polylogue.storage.sqlite.connection_profile import readonly_connection_context

    with readonly_connection_context(root / "user.db") as user:
        rows = user.execute(
            "SELECT target_ref FROM assertions WHERE kind = ? AND status = 'active' AND key = ?",
            (AssertionKind.TAG.value, tag),
        ).fetchall()
    return {str(row[0]).removeprefix("session:") for row in rows}


@pytest.mark.uses_real_clock("drives delete and mark --all through a real resident daemon over its UDS socket")
def test_complete_selection_all_verbs_receive_every_real_operation_id(
    seeded_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``delete --all`` and ``mark --all`` act on every match, across page boundaries.

    The resident selection owner walks the canonical selection in pages and
    seals every distinct identity before the mutation. With its page set below
    the seeded count, a walk that stopped after one page would preview, tag or
    delete ``PAGE`` sessions instead of all of them.

    Anti-vacuity: stop the daemon-side walk after its first page and the preview
    count, the tagged set and the deleted set each shrink to ``PAGE``.
    """
    from click.testing import CliRunner

    import polylogue.operations.daemon_mutations as daemon_mutations
    from polylogue.cli import cli

    expected = _seeded_session_ids(seeded_root)
    assert len(expected) == SEEDED_SESSIONS
    monkeypatch.setattr(daemon_mutations, "_MUTATION_SELECTION_PAGE_SIZE", PAGE)
    with cli_daemon_archive(seeded_root, monkeypatch):
        preview = CliRunner().invoke(cli, ["--plain", "find", TOKEN, "then", "delete", "--dry-run", "--all"])
        marked = CliRunner().invoke(cli, ["--plain", "find", TOKEN, "then", "mark", "--tag-add", "reviewed", "--all"])
        tagged = _tag_targets(seeded_root, "reviewed")
        deleted = CliRunner().invoke(cli, ["--plain", "find", TOKEN, "then", "delete", "--yes", "--all"])

    assert preview.exit_code == 0, preview.output
    assert json.loads(preview.output)["session_count"] == SEEDED_SESSIONS
    assert marked.exit_code == 0, marked.output
    assert tagged == expected
    assert deleted.exit_code == 0, deleted.output
    assert _seeded_session_ids(seeded_root).isdisjoint(expected)


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
                RuntimeError("later page failed"),
            ],
            "later page failed",
        ),
    ],
)
@pytest.mark.uses_real_clock("submits the delete to a real resident daemon over its UDS socket")
def test_mutating_selection_refuses_incomplete_or_failed_walks(
    seeded_root: Path, monkeypatch: pytest.MonkeyPatch, pages: list[object], expected_message: str
) -> None:
    """No --all mutation receives a prefix after a bad continuation.

    The resident selection owner reads the canonical ``cli.query`` pages; each
    page here replaces one of its reads. A missing continuation, a degraded
    outcome, a non-advancing offset and a failing later page must each refuse
    before any session is deleted.

    Anti-vacuity: accept the accumulated identities on a missing or repeated
    ``next_offset`` (the pre-fix behavior) and the delete proceeds.
    """
    from click.testing import CliRunner

    import polylogue.operations.daemon_reads as daemon_reads
    from polylogue.cli import cli

    before = _seeded_session_ids(seeded_root)
    calls = iter(pages)
    original = daemon_reads.execute_read_operation

    def selection_read(operation: str, payload: dict[str, object], **kwargs: Any) -> object:
        params = payload.get("params")
        if operation != "cli.query" or not isinstance(params, dict) or TOKEN not in str(params.get("query")):
            return original(operation, payload, **kwargs)
        page = next(calls)
        if isinstance(page, Exception):
            raise page
        assert isinstance(page, dict)
        return {
            **fixture_query_page(page),
            **({"outcome": page["outcome"]} if "outcome" in page else {}),
            "snapshot_epoch": "fixture-selected-frame",
        }

    with cli_daemon_archive(seeded_root, monkeypatch):
        monkeypatch.setattr(daemon_reads, "execute_read_operation", selection_read)
        result = CliRunner().invoke(cli, ["--plain", "find", TOKEN, "then", "delete", "--yes", "--all"])

    assert result.exit_code != 0, result.output
    assert expected_message in result.output
    assert _seeded_session_ids(seeded_root) == before


def test_unknown_total_continues_only_on_a_full_page() -> None:
    """The shared producer rule preserves an honest unknown ranked total."""

    from polylogue.operations.daemon_reads import page_next_offset

    assert page_next_offset(offset=0, returned=PAGE, total=None, limit=PAGE) == PAGE
    assert page_next_offset(offset=PAGE, returned=1, total=None, limit=PAGE) is None
    assert page_next_offset(offset=0, returned=0, total=0, limit=PAGE) is None


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
