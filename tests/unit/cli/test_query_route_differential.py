"""Per-flag selection laws for the root query's one declared read.

``archive_query`` once carried a second, local ``ArchiveStore`` implementation
of the root query behind ``_daemon_session_page_supported``, and this module
was a differential between the two routes.  That gate and that executor are
gone: every root page is now lowered onto ``cli.query`` (Seam A,
``polylogue/cli/lowering.py``) and dispatched by
``polylogue/cli/operation_kernel.py``.  ``--no-daemon`` selects the transport,
not the implementation -- the same declared handler answers either way -- so
a route differential can no longer say anything.

What survives is the part that was never about routes: each flag must select,
order or bound a specific set of sessions out of one seeded archive.  Those
were the assertions the differential was protecting, and the forwarding bugs it
caught (``--has-tool-use``/``--has-thinking`` renamed to keys
``SessionQuerySpec.from_params`` does not read, so the filter was dropped
silently) are exactly what ``expected_ids`` catches directly.

Anti-vacuity for the module: drop a key from
``lowering._selection_params`` -- or rename it on the way onto the payload --
and that flag's case goes red with the unflagged answer.
``test_every_case_changes_the_answer`` is the guard that makes that true: it
proves each flag's expectation differs from the whole-archive page, so a
forwarding bug that widened every filter cannot pass by agreeing with itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.infra.daemon_operations import cli_daemon_archive
from tests.infra.storage_records import SessionBuilder

# Session ids as the writer computes them (``origin || ':' || native_id``),
# newest first.
LONE_C = "codex-session:ext-lone-c"
PARENT_B = "chatgpt-export:ext-parent-b"
CHILD_A = "claude-code-session:ext-child-a"
PARENT_A = "claude-code-session:ext-parent-a"


@pytest.fixture
def query_route_workspace(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Path]]:
    """A seeded archive for the root query, served by the production daemon."""

    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)

    def seed(root: Path) -> None:
        _seed(root / "index.db")

    with cli_daemon_archive(
        cli_workspace["archive_root"],
        monkeypatch,
        seed_archive=seed,
        home=cli_workspace["state_dir"],
    ):
        yield cli_workspace


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
    """One root-query flag, with the argv that exercises it."""

    #: Test id, which names the flag group under test.
    name: str
    #: Root options placed before the ``find`` marker.
    root_args: tuple[str, ...] = ()
    #: Query terms placed after it.
    find_args: tuple[str, ...] = ()
    #: What must break this case.
    breaks_if: str = ""
    #: Session ids the case must select, in envelope order.
    expected_ids: tuple[str, ...] | None = None


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


def _run(args: Sequence[str]) -> tuple[int, dict[str, object]]:
    """Run one root query end to end against the seeded archive.

    The workspace fixture points CLI daemon discovery at its real UDS listener.
    """
    from polylogue.cli import cli

    result = CliRunner().invoke(cli, ["--plain", *args], catch_exceptions=True)
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    try:
        return result.exit_code, json.loads(result.output) if result.output.strip() else {}
    except json.JSONDecodeError:
        # A refusal that never reaches the renderer (Click's own UsageError line).
        return result.exit_code, {"_refusal": result.output}


def _argv(case: FlagCase) -> list[str]:
    return [*case.root_args, "find", *case.find_args, "--format", "json", "--limit", "10"]


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
def test_flag_selects_exactly_its_sessions(query_route_workspace: dict[str, Path], case: FlagCase) -> None:
    """One flag, one declared read, one selected set in envelope order."""
    exit_code, payload = _run(_argv(case))
    assert exit_code == 0, payload
    assert case.expected_ids is not None
    assert _ids(payload) == case.expected_ids, case.breaks_if


@pytest.mark.parametrize("case", FLAG_CASES, ids=lambda case: case.name)
def test_every_case_changes_the_answer(query_route_workspace: dict[str, Path], case: FlagCase) -> None:
    """The flag under test must select, order or bound something different."""
    _, unflagged = _run(["find", "--format", "json", "--limit", "10"])
    assert set(_ids(unflagged)) == {PARENT_A, CHILD_A, PARENT_B, LONE_C}

    _, flagged = _run(_argv(case))
    assert case.expected_ids is not None
    assert _ids(flagged) == case.expected_ids
    if case.name not in {"sort", "reverse"}:
        assert set(_ids(flagged)) != set(_ids(unflagged)), (
            f"{case.name} produced the unflagged selection; it exercises nothing"
        )
    else:
        assert _ids(flagged) != _ids(unflagged), f"{case.name} produced the unflagged order; it exercises nothing"


def test_exclude_text_is_withheld_from_the_operation_because_no_answer_is_chosen(
    query_route_workspace: dict[str, Path],
) -> None:
    """``--exclude-text`` is the one recognised parameter deliberately not forwarded.

    Three answers exist for one flag and none has been chosen: ``cli.query``
    applies the content post-filter to a list page
    (``_archive_list_summaries_with_post_filters``); on a ranked page it applies
    it to the count and not to the hits, reporting a total smaller than the page
    it returned; and the CLI's retired local branch ignored it everywhere.
    Forwarding it would silently change ``find --exclude-text`` results, so
    ``lowering._SELECTION_EXCLUDED`` withholds it and the flag stays inert
    (polylogue-v1mnm).

    Anti-vacuity: remove ``exclude_text`` from ``lowering._SELECTION_EXCLUDED``
    and this goes red -- ``harder`` appears in one seeded session, so the page
    and the total both shrink.
    """
    plain = ["find", "--format", "json", "--limit", "10"]
    excluded = ["--exclude-text", "harder", *plain]
    _, baseline = _run(plain)
    _, filtered = _run(excluded)

    assert _ids(filtered) == _ids(baseline)
    assert filtered["total"] == baseline["total"]


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
    summaries = [
        SimpleNamespace(session_id=f"codex-session:s{i:04d}", display_label=None, display_label_source=None)
        for i in range(total)
    ]
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

    def _to_session(
        envelope: SimpleNamespace,
        *,
        display_label: object = None,
        display_label_source: object = None,
    ) -> SimpleNamespace:
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
