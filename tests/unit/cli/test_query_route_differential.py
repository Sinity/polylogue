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
from typing import Any

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


def test_exclude_text_filters_before_list_pagination_and_count(
    query_route_workspace: dict[str, Path],
) -> None:
    """The excluded session leaves the population before the page is cut."""
    plain = ["find", "--format", "json", "--limit", "10"]
    excluded = ["--exclude-text", "weigh", *plain]
    _, baseline = _run(plain)
    _, filtered = _run(excluded)

    assert LONE_C in _ids(baseline)
    assert _ids(filtered) == (PARENT_B, CHILD_A, PARENT_A)
    assert isinstance(baseline["total"], int)
    assert filtered["total"] == baseline["total"] - 1

    _, page = _run(["--exclude-text", "weigh", "find", "--format", "json", "--limit", "1", "--offset", "1"])
    assert _ids(page) == (CHILD_A,)
    assert page["total"] == filtered["total"]

    _, count = _run(["--exclude-text", "weigh", "find", "then", "analyze", "count", "--format", "json"])
    assert count.get("count") == filtered["total"], count


def test_ranked_exclude_text_refuses_without_a_misleading_page(query_route_workspace: dict[str, Path]) -> None:
    exit_code, payload = _run(["--exclude-text", "harder", "find", "retry", "--format", "json"])

    assert exit_code != 0
    assert "ranked search cannot apply text exclusions before ranking" in str(payload)


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        # ``--id X`` with --exclude-text is no longer an exact read: an extra
        # predicate makes it a filtered page (#5941). A lone ref token is.
        (["find", LONE_C], "Exact session reads do not apply --exclude-text"),
        (["find", "messages where role:assistant"], "Unit queries do not apply --exclude-text"),
        (["find", "from result-set:stable-set"], "Reference reads do not apply --exclude-text"),
    ],
)
def test_exclude_text_refuses_routes_that_cannot_filter_before_read(
    query_route_workspace: dict[str, Path], query: list[str], reason: str
) -> None:
    exit_code, payload = _run(["--exclude-text", "weigh", *query, "--format", "json"])

    assert exit_code != 0
    assert reason in str(payload)


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

        def iter_summaries(self, **kwargs: object) -> Iterator[SimpleNamespace]:
            yield from summaries

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


def test_exclude_text_post_filter_pages_a_scope_of_any_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """A large post-filter scope is paged, never refused.

    Anti-vacuity: stop after the first candidate page, or refuse past a
    candidate count, and the survivor on the last page is never returned.
    """
    from types import SimpleNamespace

    from polylogue.api import archive as archive_api

    monkeypatch.setattr(archive_api, "POST_FILTER_HYDRATION_CHUNK", 2)
    ids = [f"s{index}" for index in range(10)]

    class _PagedArchive:
        def iter_summaries(self, *, limit: int | None, **kwargs: object) -> Iterator[SimpleNamespace]:
            # One forward cursor: no page size, no offset to re-walk.
            assert limit is None and "offset" not in kwargs, (limit, kwargs)
            for session_id in ids:
                yield SimpleNamespace(session_id=session_id, display_label=None, display_label_source=None)

        def read_session(self, session_id: str) -> str:
            return session_id

    monkeypatch.setattr(
        archive_api,
        "archive_envelope_to_session",
        lambda envelope, **kwargs: SimpleNamespace(id=envelope),
    )
    spec = SimpleNamespace(
        to_plan=lambda: SimpleNamespace(
            _apply_full_filters=lambda sessions, sql_pushed: [s for s in sessions if s.id == "s9"]
        )
    )
    page = archive_api._archive_list_summaries_with_post_filters(
        _PagedArchive(),
        spec,  # type: ignore[arg-type]
        query_text=None,
        query_kwargs={},
        limit=5,
        offset=0,
    )
    assert [summary.session_id for summary in page] == ["s9"]


@pytest.mark.parametrize("randomizer", [{"sample": True}, {"sort": "random"}])
def test_exclude_text_post_filter_randomizes_survivors_not_candidate_pages(
    monkeypatch: pytest.MonkeyPatch, randomizer: dict[str, object]
) -> None:
    """A sampled or randomly sorted post-filter scan terminates and keeps its size.

    A sampled store read ignores ``offset``; paging candidates with it would
    fetch a fresh random page forever.

    Anti-vacuity: forward ``sample``/``sort=random`` to the candidate query and
    the fake store raises; sample before filtering and the lone survivor on
    the last candidate page is usually missing.
    """
    from types import SimpleNamespace

    from polylogue.api import archive as archive_api

    ids = [f"s{index}" for index in range(10)]

    class _PagedArchive:
        def iter_summaries(self, *, limit: int | None, **kwargs: object) -> Iterator[SimpleNamespace]:
            assert "sample" not in kwargs and kwargs.get("sort") != "random", kwargs
            assert limit is None and "offset" not in kwargs, (limit, kwargs)
            for session_id in ids:
                yield SimpleNamespace(session_id=session_id, display_label=None, display_label_source=None)

        def read_session(self, session_id: str) -> str:
            return session_id

    monkeypatch.setattr(
        archive_api,
        "archive_envelope_to_session",
        lambda envelope, **kwargs: SimpleNamespace(id=envelope),
    )
    spec = SimpleNamespace(
        to_plan=lambda: SimpleNamespace(
            _apply_full_filters=lambda sessions, sql_pushed: [s for s in sessions if s.id in {"s8", "s9"}]
        )
    )
    page = archive_api._archive_list_summaries_with_post_filters(
        _PagedArchive(),
        spec,  # type: ignore[arg-type]
        query_text=None,
        query_kwargs={"limit": 5, **randomizer},
        limit=None,
        offset=0,
    )
    assert sorted(summary.session_id for summary in page) == ["s8", "s9"]

    one = archive_api._archive_list_summaries_with_post_filters(
        _PagedArchive(),
        spec,  # type: ignore[arg-type]
        query_text=None,
        query_kwargs={"limit": 1, **randomizer},
        limit=None,
        offset=0,
    )
    assert len(one) == 1 and one[0].session_id in {"s8", "s9"}


def test_reservoir_sample_keeps_only_the_window() -> None:
    """The random post-filter window is chosen in one pass over a generator.

    Anti-vacuity: materialize the stream and the ``len`` check on the
    generator-fed reservoir is the only guard; returning more than ``size``
    items or duplicates fails the assertions.
    """
    from polylogue.api import archive as archive_api

    chosen = archive_api._reservoir_sample((index for index in range(10_000)), 7)
    assert len(chosen) == 7 and len(set(chosen)) == 7
    assert archive_api._reservoir_sample(iter(range(3)), 7) == [0, 1, 2]
    assert archive_api._reservoir_sample(iter(range(3)), 0) == []


def test_facet_scope_post_filter_hydrates_each_candidate_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post-filtered facet scope is one pass, not a restart per facet page.

    Anti-vacuity: page the facet scope through
    ``_archive_list_summaries_for_spec`` again and each later page restarts the
    candidate scan, so candidates are hydrated more than once.
    """
    from types import SimpleNamespace

    from polylogue.api import archive as archive_api
    from polylogue.archive.query.spec import SessionQuerySpec

    ids = [f"s{index}" for index in range(10)]
    reads: list[str] = []

    class _PagedArchive:
        def iter_summaries(self, *, limit: int | None, **kwargs: object) -> Iterator[SimpleNamespace]:
            # One forward cursor: no page size, no offset to re-walk.
            assert limit is None and "offset" not in kwargs, (limit, kwargs)
            for session_id in ids:
                yield SimpleNamespace(session_id=session_id, display_label=None, display_label_source=None)

        def read_session(self, session_id: str) -> str:
            reads.append(session_id)
            return session_id

    monkeypatch.setattr(
        archive_api,
        "archive_envelope_to_session",
        lambda envelope, **kwargs: SimpleNamespace(id=envelope),
    )
    monkeypatch.setattr(
        SessionQuerySpec,
        "to_plan",
        lambda self: SimpleNamespace(
            _apply_full_filters=lambda sessions, sql_pushed: [s for s in sessions if s.id != "s0"]
        ),
    )
    spec = SessionQuerySpec(exclude_text_terms=("absent",))
    scope = [summary.session_id for summary in archive_api._iter_facet_scope(_PagedArchive(), spec)]

    assert scope == ids[1:]
    assert sorted(reads) == sorted(ids)


def test_search_post_filter_candidates_are_distinct_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session with several matching blocks is one candidate, sampled once.

    Anti-vacuity: yield one candidate per search hit and the session with
    three matching blocks is hydrated three times and can fill a sample alone.
    """
    from types import SimpleNamespace

    from polylogue.api import archive as archive_api

    reads: list[str] = []

    class _SearchArchive:
        def iter_search_summaries(self, query: str, **kwargs: object) -> Iterator[SimpleNamespace]:
            for session_id in ("s1", "s1", "s2", "s1", "s3"):
                yield SimpleNamespace(session_id=session_id)

        def read_summary(self, session_id: str) -> SimpleNamespace:
            reads.append(session_id)
            return SimpleNamespace(session_id=session_id)

    candidates = archive_api._post_filter_candidates(_SearchArchive(), query_text="foo", query_kwargs={})
    assert [candidate.session_id for candidate in candidates] == ["s1", "s2", "s3"]
    assert reads == ["s1", "s2", "s3"]


def test_post_filter_window_clamps_negative_offset_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: pass a negative offset to ``islice`` and it raises ``ValueError``."""
    from types import SimpleNamespace

    from polylogue.api import archive as archive_api

    ids = [f"s{index}" for index in range(4)]

    class _Archive:
        def iter_summaries(self, **kwargs: object) -> Iterator[SimpleNamespace]:
            for session_id in ids:
                yield SimpleNamespace(session_id=session_id, display_label=None, display_label_source=None)

        def read_session(self, session_id: str) -> str:
            return session_id

    monkeypatch.setattr(
        archive_api, "archive_envelope_to_session", lambda envelope, **kwargs: SimpleNamespace(id=envelope)
    )
    spec: Any = SimpleNamespace(
        to_plan=lambda: SimpleNamespace(_apply_full_filters=lambda sessions, sql_pushed: list(sessions))
    )
    page = archive_api._archive_list_summaries_with_post_filters(
        _Archive(),
        spec,
        query_text=None,
        query_kwargs={},
        limit=2,
        offset=-1,
    )
    assert [summary.session_id for summary in page] == ["s0", "s1"]
    empty = archive_api._archive_list_summaries_with_post_filters(
        _Archive(),
        spec,
        query_text=None,
        query_kwargs={},
        limit=-3,
        offset=0,
    )
    assert empty == []


def test_facet_search_scope_hydrates_each_session_once() -> None:
    """Anti-vacuity: hydrate one summary per search hit and a session with three
    matching blocks is read three times."""
    from types import SimpleNamespace

    from polylogue.api import archive as archive_api
    from polylogue.archive.query.spec import SessionQuerySpec

    reads: list[str] = []

    class _SearchArchive:
        def iter_search_summaries(self, query: str, **kwargs: object) -> Iterator[SimpleNamespace]:
            for session_id in ("s1", "s1", "s2", "s1"):
                yield SimpleNamespace(session_id=session_id)

        def read_summary(self, session_id: str) -> SimpleNamespace:
            reads.append(session_id)
            return SimpleNamespace(session_id=session_id)

    scope = list(archive_api._iter_facet_scope(_SearchArchive(), SessionQuerySpec(query_terms=("foo",))))
    assert [summary.session_id for summary in scope] == ["s1", "s2"]
    assert reads == ["s1", "s2"]
