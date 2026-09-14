"""Per-flag differential: the CLI's local branch vs the ``cli.query`` operation.

``archive_query._daemon_session_page_supported`` used to refuse the declared
``cli.query`` read for thirteen groups of inputs, forking execution into a
second local ``ArchiveStore`` implementation of the same query.  Most of those
refusals were stale, and every one of them hid the possibility of a
behavioural difference between the two routes.

Each case below runs one flag twice against one seeded archive:

* ``--no-daemon`` — ``_try_emit_daemon_session_page`` declines and
  ``_execute_archive_query_stdout`` executes locally against ``ArchiveStore``.
* no flag — the request is lowered onto ``cli.query`` and executed through
  ``operation_kernel.configured_read_operation``.  No daemon socket exists in
  this workspace, so the kernel's declared ``DIRECT_READ`` fallback runs the
  identical ``operations/daemon_reads`` handler in-process.  That is the
  "direct mode" leg: same handler, no transport.

The comparison is the whole ``--format json`` envelope, minus the one
documented difference — ``source: "daemon"``, the route's own provenance
marker.  A flag the operation silently ignores shows up as different
``items``/``total``; a flag whose two implementations disagree on row shape
shows up as a different row.

Anti-vacuity.  Two guards, both required:

* ``test_flag_reaches_the_operation_route`` asserts the operation actually
  answered (``source == "daemon"``).  Without it, a case the gate still
  refuses would compare the local branch with itself and prove nothing.
* ``test_every_case_changes_the_answer`` asserts each flag selects, orders or
  bounds something different from the unflagged page.  Without it, a
  forwarding bug that widened every filter back to "the whole archive" would
  still make both routes agree.

The named mutation for the whole module: revert
``archive_query._cli_query_operation_params`` to the hand-listed rename table
and the ``has-tool-use``/``has-thinking`` cases go red (those keys were
renamed to names ``SessionQuerySpec.from_params`` does not read, so the filter
was dropped); revert the gate deletion and every case fails its
``source == "daemon"`` assertion instead.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.infra.storage_records import SessionBuilder

# Session ids as the writer computes them (``origin || ':' || native_id``),
# newest first.
LONE_C = "codex-session:ext-lone-c"
PARENT_B = "chatgpt-export:ext-parent-b"
CHILD_A = "claude-code-session:ext-child-a"
PARENT_A = "claude-code-session:ext-parent-a"


@pytest.fixture
def query_route_workspace(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A seeded archive both CLI routes read, with daemon discovery neutral."""

    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)
    _seed(cli_workspace["archive_root"] / "index.db")
    return cli_workspace


def _seed(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "parent-a")
        .provider("claude-code")
        .title("Retry budget for the ingest loop")
        .git_repository_url("polylogue")
        .provider_project_ref("proj-alpha")
        .created_at("2026-03-01T00:00:00Z")
        .updated_at("2026-03-01T00:00:00Z")
        .add_message("pa1", role="user", text="how do we bound the ingest retry budget")
        .add_message("pa2", role="assistant", text="cap the attempts and record convergence debt")
        .save()
    )
    (
        SessionBuilder(db_path, "child-a")
        .provider("claude-code")
        .title("Subagent: measure the retry budget")
        .git_repository_url("polylogue")
        .parent_session("claude-code-session:ext-parent-a")
        .branch_type("subagent")
        .created_at("2026-03-02T00:00:00Z")
        .updated_at("2026-03-02T00:00:00Z")
        .add_message("ca1", role="user", text="measure the retry budget on the seeded corpus")
        .save()
    )
    (
        SessionBuilder(db_path, "parent-b")
        .provider("chatgpt")
        .title("Ownership in Rust")
        .git_repository_url("other-repo")
        .created_at("2026-03-03T00:00:00Z")
        .updated_at("2026-03-03T00:00:00Z")
        .add_message("pb1", role="user", text="what is ownership in Rust")
        .add_message(
            "pb2",
            role="assistant",
            text="checking the borrow checker",
            blocks=[{"type": "tool_use", "tool_name": "Bash", "tool_id": "t1", "tool_input": {}}],
        )
        .save()
    )
    (
        SessionBuilder(db_path, "lone-c")
        .provider("codex")
        .title("Thinking about the retry budget")
        .created_at("2026-03-04T00:00:00Z")
        .updated_at("2026-03-04T00:00:00Z")
        .add_message(
            "lc1",
            role="assistant",
            text="think harder about the retry budget",
            blocks=[{"type": "thinking", "text": "weigh the retry budget"}],
        )
        .save()
    )


@dataclass(frozen=True)
class FlagCase:
    """One retired gate rejection, with the argv that exercises it."""

    #: Test id, which names the rejection group the case retires.
    name: str
    #: Root options placed before the ``find`` marker.
    root_args: tuple[str, ...] = ()
    #: Query terms placed after it.
    find_args: tuple[str, ...] = ()
    #: What must break this case.
    breaks_if: str = ""
    #: Session ids the case must select, in envelope order, when it names a set.
    expected_ids: tuple[str, ...] | None = None
    #: Envelope keys dropped before comparing, beyond ``source``.
    ignore_keys: frozenset[str] = field(default_factory=frozenset)


FLAG_CASES: tuple[FlagCase, ...] = (
    FlagCase(
        name="sort",
        root_args=("--sort", "messages"),
        expected_ids=(PARENT_B, PARENT_A, LONE_C, CHILD_A),
        breaks_if="the operation drops spec.sort and answers in the default date order",
    ),
    FlagCase(
        name="reverse",
        root_args=("--reverse",),
        expected_ids=(PARENT_A, CHILD_A, PARENT_B, LONE_C),
        breaks_if="the operation drops spec.reverse and answers in the default direction",
    ),
    FlagCase(
        name="boolean-predicate",
        find_args=("origin:claude-code-session AND repo:polylogue",),
        expected_ids=(CHILD_A, PARENT_A),
        breaks_if="the compiled boolean predicate is not forwarded and every session is returned",
    ),
    FlagCase(
        name="project-refs",
        root_args=("--project", "proj-alpha"),
        expected_ids=(PARENT_A,),
        breaks_if="`project` is not forwarded and the filter widens to the whole archive",
    ),
    FlagCase(
        name="latest",
        root_args=("--latest",),
        expected_ids=(LONE_C,),
        # The local branch forces ``limit = 1`` and then fetches ``limit + 1``
        # rows for its pagination probe, so it mints a continuation cursor for a
        # query that means "exactly one". The operation's ``None`` is the honest
        # answer; the difference is the local probe, not the selection.
        ignore_keys=frozenset({"next_cursor", "next_offset"}),
        breaks_if="the operation ignores spec.latest and returns a full page",
    ),
    FlagCase(
        name="has-thinking",
        root_args=("--has-thinking",),
        expected_ids=(LONE_C,),
        breaks_if="`filter_has_thinking` is renamed to a key `from_params` does not read, widening the page",
    ),
    FlagCase(
        name="has-tool-use",
        root_args=("--has-tool-use",),
        expected_ids=(PARENT_B,),
        breaks_if="`filter_has_tool_use` is renamed to a key `from_params` does not read, widening the page",
    ),
)


#: Inputs that must keep declining the operation, because ``cli.query``
#: declares no equivalent.  The parity assertion still runs: it compares the
#: local branch with itself, which is worth nothing on its own, so the
#: load-bearing claim here is the absence of ``source``.
STILL_LOCAL_CASES: tuple[FlagCase, ...] = (
    FlagCase(
        name="sample-stays-local",
        root_args=("--sample", "2"),
        ignore_keys=frozenset({"items"}),
        breaks_if="`sample` routes to `cli.query`, whose list path drops it and answers an ordered page",
    ),
    FlagCase(
        name="exact-ref-stays-local",
        root_args=("--id", PARENT_B),
        breaks_if="an exact-ref read routes to `cli.query`, which has no session-document result",
    ),
    FlagCase(
        name="semantic-stays-local",
        root_args=("--similar", "retry budget"),
        breaks_if="vector retrieval routes to `cli.query`, whose unavailable-backend refusal is a "
        "typed EmbeddingRetrievalNotReadyError rather than the CLI's UsageError",
    ),
)


def _run(args: Sequence[str], *, no_daemon: bool) -> tuple[int, dict[str, object]]:
    from polylogue.cli import cli

    root: list[str] = ["--plain"]
    if no_daemon:
        root.append("--no-daemon")
    result = CliRunner().invoke(cli, [*root, *args], catch_exceptions=True)
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    try:
        return result.exit_code, json.loads(result.output) if result.output.strip() else {}
    except json.JSONDecodeError:
        # A refusal that never reaches the renderer (Click's own UsageError
        # line).  Comparing it verbatim is the point for the cases that must
        # keep refusing the same way on both routes.
        return result.exit_code, {"_refusal": result.output}


def _argv(case: FlagCase) -> list[str]:
    return [*case.root_args, "find", *case.find_args, "--format", "json", "--limit", "10"]


def _comparable(envelope: dict[str, object], case: FlagCase) -> dict[str, object]:
    return {key: value for key, value in envelope.items() if key != "source" and key not in case.ignore_keys}


def _ids(envelope: dict[str, object]) -> tuple[str, ...]:
    rows = envelope.get("items")
    if not isinstance(rows, list):
        return ()
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("id"), str):
            ids.append(str(row["id"]))
            continue
        session = row.get("session")
        if isinstance(session, dict) and isinstance(session.get("id"), str):
            ids.append(str(session["id"]))
    return tuple(ids)


@pytest.mark.parametrize("case", FLAG_CASES, ids=lambda case: case.name)
def test_flag_reaches_the_operation_route(query_route_workspace: dict[str, Path], case: FlagCase) -> None:
    """The gate must let this flag through, or the parity case is self-comparison."""
    _, payload = _run(_argv(case), no_daemon=False)
    assert payload.get("source") == "daemon", (
        f"{case.name} is still refused by _daemon_session_page_supported: the parity case for it "
        "would compare the local branch with itself"
    )


@pytest.mark.parametrize("case", (*FLAG_CASES, *STILL_LOCAL_CASES), ids=lambda case: case.name)
def test_local_branch_and_cli_query_operation_answer_identically(
    query_route_workspace: dict[str, Path],
    case: FlagCase,
) -> None:
    """One flag, two routes, one envelope."""
    argv = _argv(case)
    local_code, local_payload = _run(argv, no_daemon=True)
    operation_code, operation_payload = _run(argv, no_daemon=False)

    if case in STILL_LOCAL_CASES:
        # ``--id`` answers with a session document whose own ``source`` field is
        # acquisition provenance, not route provenance, so the route marker is
        # read as "not the daemon envelope's marker" instead.
        assert operation_payload.get("mode") == local_payload.get("mode")
        assert operation_payload.get("source") != "daemon", (
            f"{case.name} names a capability `cli.query` does not declare; it must keep declining the operation"
        )
    else:
        assert "source" not in local_payload, "the local branch must not claim daemon provenance"
    assert operation_code == local_code
    assert _comparable(operation_payload, case) == _comparable(local_payload, case), case.breaks_if


@pytest.mark.parametrize("case", FLAG_CASES, ids=lambda case: case.name)
def test_every_case_changes_the_answer(query_route_workspace: dict[str, Path], case: FlagCase) -> None:
    """The flag under test must select, order or bound something different."""
    _, unflagged = _run(["find", "--format", "json", "--limit", "10"], no_daemon=True)
    assert set(_ids(unflagged)) == {PARENT_A, CHILD_A, PARENT_B, LONE_C}

    _, flagged = _run(_argv(case), no_daemon=True)
    if case.expected_ids is not None:
        assert _ids(flagged) == case.expected_ids
        return
    assert _ids(flagged) != _ids(unflagged), f"{case.name} produced the unflagged answer; it exercises nothing"


def test_exclude_text_is_withheld_from_the_operation_because_the_routes_disagree(
    query_route_workspace: dict[str, Path],
) -> None:
    """``--exclude-text`` is the one recognised parameter still not forwarded.

    Three answers exist for one flag and none has been chosen: ``cli.query``
    applies the content post-filter to a list page
    (``_archive_list_summaries_with_post_filters``); on a ranked page it
    applies it to the count and not to the hits, reporting a total smaller
    than the page it returned; and the CLI's local branch ignores it
    everywhere.  Forwarding it would silently change ``find --exclude-text``
    results, so ``_cli_query_operation_params`` withholds it and both routes
    keep answering the same -- today's -- way (polylogue-v1mnm).

    Anti-vacuity: drop the ``key != "exclude_text"`` guard from
    ``_cli_query_operation_params`` and this test goes red, because the two
    routes then return different rows and a different total.
    """
    argv = ["--exclude-text", "harder", "find", "--format", "json", "--limit", "10"]
    _, local_payload = _run(argv, no_daemon=True)
    _, operation_payload = _run(argv, no_daemon=False)

    assert operation_payload["source"] == "daemon"
    assert _ids(operation_payload) == _ids(local_payload)
    assert operation_payload["total"] == local_payload["total"]


def test_exclude_text_post_filter_hydrates_in_bounded_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative-text page hydrates a bounded prefix, not the whole candidate set.

    ``exclude_text`` has no SQL reduction, so every candidate must be hydrated
    to be tested.  The route must hydrate in chunks, drop each chunk, and stop
    once the requested page is full -- a page of 2 must not read 1000 sessions.

    Anti-vacuity: restore the single-pass hydration in
    ``_archive_list_summaries_with_post_filters`` (one list comprehension over
    every candidate before filtering) and ``read_session`` is called once per
    candidate, so the bound assertion goes red.
    """
    from types import SimpleNamespace

    from polylogue.api import archive as archive_api

    total = 1000
    summaries = [SimpleNamespace(session_id=f"codex-session:s{i:04d}", display_label=None) for i in range(total)]
    reads: list[str] = []

    class _Archive:
        def count_sessions(self, **kwargs: object) -> int:
            return total

        def list_summaries(self, **kwargs: object) -> list[SimpleNamespace]:
            limit = kwargs.get("limit")
            return summaries[: int(limit)] if isinstance(limit, int) else list(summaries)

        def read_session(self, session_id: str) -> SimpleNamespace:
            reads.append(session_id)
            return SimpleNamespace(session_id=session_id)

    spec = SimpleNamespace(
        to_plan=lambda: SimpleNamespace(
            _apply_full_filters=lambda sessions, sql_pushed: list(sessions),
        )
    )

    def _to_session(envelope: SimpleNamespace, *, display_label: object = None) -> SimpleNamespace:
        return SimpleNamespace(id=envelope.session_id)

    monkeypatch.setattr(archive_api, "archive_envelope_to_session", _to_session)
    page = archive_api._archive_list_summaries_with_post_filters(
        _Archive(),
        spec,  # type: ignore[arg-type]
        query_text=None,
        query_kwargs={"limit": 2},
        limit=2,
        offset=0,
    )

    assert [summary.session_id for summary in page] == ["codex-session:s0000", "codex-session:s0001"]
    assert len(reads) <= archive_api.POST_FILTER_HYDRATION_CHUNK, (
        f"hydrated {len(reads)} sessions for a 2-row page; the post-filter is unbounded"
    )
    assert len(reads) < total


def test_exclude_text_post_filter_refuses_an_over_cap_scope() -> None:
    """Over the declared cap the route refuses; it never returns a short page.

    Anti-vacuity: delete the ``POST_FILTER_HYDRATION_CAP`` check in
    ``_post_filter_candidates`` and no refusal is raised, so ``pytest.raises``
    fails.
    """
    from types import SimpleNamespace

    from polylogue.api import archive as archive_api

    class _HugeArchive:
        def count_sessions(self, **kwargs: object) -> int:
            return archive_api.POST_FILTER_HYDRATION_CAP + 1

        def list_summaries(self, **kwargs: object) -> list[SimpleNamespace]:
            raise AssertionError("candidates must not be fetched above the cap")

    spec = SimpleNamespace(to_plan=lambda: SimpleNamespace(_apply_full_filters=lambda sessions, sql_pushed: sessions))
    with pytest.raises(archive_api.PostFilterScopeTooLargeError) as excinfo:
        archive_api._archive_list_summaries_with_post_filters(
            _HugeArchive(),
            spec,  # type: ignore[arg-type]
            query_text=None,
            query_kwargs={},
            limit=1,
            offset=0,
        )
    assert excinfo.value.gap_reason.startswith("post_filter_scope_too_large:")
