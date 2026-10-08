"""SQL is the one ordering authority for lexical and structured session queries.

The executor used to re-sort every SQL-ordered result in Python by
``updated_at``, which discarded the requested sort and disagreed with SQL's
``COALESCE(updated_at, created_at)`` key. A post-filtered page therefore had
to read every candidate before it could cut, because the Python order could
move a late row ahead of a kept one (polylogue-6xrab, polylogue-ztm1t).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.hydration import archive_envelope_to_session
from polylogue.archive.query.archive_execution import list_archive, list_summaries_archive
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.transaction import run_archive_read
from polylogue.archive.session.domain_models import Session
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder


def _seed(tmp_path: Path, name: str, *, updated_at: str, messages: int, text: str = "neutral") -> None:
    builder = SessionBuilder(tmp_path / "index.db", name).provider("claude-code").title(name).updated_at(updated_at)
    for index in range(messages):
        builder = builder.add_message(f"{name}-m{index}", role="user", text=f"{text} {index}")
    builder.save()


@pytest.mark.asyncio
async def test_requested_sort_is_not_overridden_by_recency(tmp_path: Path) -> None:
    """Anti-vacuity: restore the Python re-sort by ``updated_at`` and the newer,
    shorter session is listed first despite ``sort=messages``."""
    _seed(tmp_path, "long-old", updated_at="2026-01-01T00:00:00Z", messages=3)
    _seed(tmp_path, "short-new", updated_at="2026-02-01T00:00:00Z", messages=1)

    summaries = await list_summaries_archive(SessionQueryPlan(sort="messages"), archive_root=tmp_path, config=None)

    assert [summary.title for summary in summaries] == ["long-old", "short-new"]


@pytest.mark.asyncio
async def test_count_sorted_sessions_order_by_the_composed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lineage child's stored counters cover only its tail; the page must rank
    the recomposed Session the caller receives.

    Recomposition is simulated by padding one hydrated session, the shape a
    prefix-sharing child takes once its inherited prefix is composed.

    Anti-vacuity: order count-sorted sessions by the tail-only SQL key again
    (drop ``composed_order``) and the one-message ``child`` never enters the
    one-row window, so ``standalone`` is returned.
    """
    _seed(tmp_path, "child", updated_at="2026-01-01T00:00:00Z", messages=1)
    _seed(tmp_path, "standalone", updated_at="2026-01-02T00:00:00Z", messages=2)
    _compose_child(monkeypatch)

    sessions = await list_archive(SessionQueryPlan(sort="messages", limit=1), archive_root=tmp_path, config=None)

    assert [session.title for session in sessions] == ["child"]


def _compose_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pad the hydrated ``child`` the way prefix recomposition grows it."""
    import polylogue.archive.query.archive_execution as execution

    def composing(*args: object, **kwargs: object) -> object:
        session = archive_envelope_to_session(*args, **kwargs)  # type: ignore[arg-type]
        if session.title == "child":
            return session.model_copy(update={"messages": list(session.messages) * 11})
        return session

    monkeypatch.setattr(execution, "archive_envelope_to_session", composing)


@pytest.mark.asyncio
async def test_composed_count_sort_ranks_every_candidate_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The composed sort pages through the whole candidate set, not one batch.

    Anti-vacuity: fetch a single ``default_limit`` batch again and the
    tail-sorted ``child`` (fewest stored messages) never reaches Python, so a
    standalone session is returned.
    """
    _seed(tmp_path, "child", updated_at="2026-01-01T00:00:00Z", messages=1)
    for index in range(3):
        _seed(tmp_path, f"standalone-{index}", updated_at=f"2026-01-0{index + 2}T00:00:00Z", messages=2)
    _compose_child(monkeypatch)

    sessions = await list_archive(
        SessionQueryPlan(sort="messages", limit=1), archive_root=tmp_path, config=None, default_limit=2
    )

    assert [session.title for session in sessions] == ["child"]


@pytest.mark.asyncio
async def test_custom_predicate_runs_once_per_candidate(tmp_path: Path) -> None:
    """A finite post-filtered page evaluates each predicate once per session.

    Anti-vacuity: filter the batched survivors a second time and a predicate
    that accepts a session only on first sight drops every result.
    """
    _seed(tmp_path, "only", updated_at="2026-01-01T00:00:00Z", messages=1)
    seen: list[str] = []

    def first_sight(session: Session) -> bool:
        session_id = str(session.id)
        accepted = session_id not in seen
        seen.append(session_id)
        return accepted

    sessions = await list_archive(
        SessionQueryPlan(predicates=(first_sight,), limit=1), archive_root=tmp_path, config=None
    )

    assert [session.title for session in sessions] == ["only"]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_attached_units_are_projected_one_result_page_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-filter candidate batch wider than the unit budget still serves its page.

    Anti-vacuity: project the whole ten-page candidate batch at once and
    ``fetch_attached_units`` refuses it with ``AttachedUnitPageTooWideError``.
    """
    for index in range(4):
        _seed(tmp_path, f"s{index}", updated_at=f"2026-01-0{index + 1}T00:00:00Z", messages=1)
    monkeypatch.setattr("polylogue.archive.query.attached_units._MAX_ROWS_PER_PAGE", 3)
    monkeypatch.setattr("polylogue.archive.query.archive_execution._fetch_limit", lambda plan, *, default: 4)

    sessions = await list_archive(
        SessionQueryPlan(negative_terms=("absent-term",), limit=1),
        archive_root=tmp_path,
        config=None,
        with_units=("message",),
    )

    assert [session.title for session in sessions] == ["s3"]


@pytest.mark.asyncio
async def test_early_filter_sees_the_fully_hydrated_session(tmp_path: Path) -> None:
    """A predicate evaluated during the early-stop fetch sees the session the
    caller gets, display label included.

    Anti-vacuity: hydrate without the summary's ``display_label`` in the early
    filter and the predicate rejects every row, returning nothing.
    """
    _seed(tmp_path, "labelled", updated_at="2026-01-01T00:00:00Z", messages=1)
    labels = [
        session.display_title
        for session in await list_archive(SessionQueryPlan(limit=1), archive_root=tmp_path, config=None)
    ]
    assert labels
    expected = labels[0]

    sessions = await list_archive(
        SessionQueryPlan(predicates=(lambda session: session.display_title == expected,), limit=1),
        archive_root=tmp_path,
        config=None,
    )

    assert [session.display_title for session in sessions] == [expected]


@pytest.mark.asyncio
async def test_post_filtered_page_stops_once_the_page_is_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A finite post-filtered page reads only until ``offset + limit`` rows matched.

    Anti-vacuity: drop the early stop in ``_archive_summaries`` and every
    candidate batch is fetched, so ``fetches`` exceeds one.
    """
    for index in range(12):
        _seed(tmp_path, f"s{index:02d}", updated_at=f"2026-01-{index + 1:02d}T00:00:00Z", messages=1)
    fetches: list[int] = []
    original = ArchiveStore.list_summaries

    def counting(self: ArchiveStore, *args: object, **kwargs: object) -> object:
        result = original(self, *args, **kwargs)  # type: ignore[arg-type]
        fetches.append(len(result))
        return result

    monkeypatch.setattr(ArchiveStore, "list_summaries", counting)
    monkeypatch.setattr("polylogue.archive.query.archive_execution._fetch_limit", lambda plan, *, default: 4)

    sessions = await list_archive(
        SessionQueryPlan(negative_terms=("absent-term",), limit=2), archive_root=tmp_path, config=None
    )

    assert [session.title for session in sessions] == ["s11", "s10"]
    assert len(fetches) == 1


@pytest.mark.asyncio
async def test_ranked_composed_sort_ranks_the_complete_scoped_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A requested page does not bound the semantic candidate relation."""
    from tests.infra.archive_templates import run_off_event_loop
    from tests.infra.scoped_semantic import ranking_archive

    config, provider, identities, requests = run_off_event_loop(
        lambda: ranking_archive(
            tmp_path,
            [
                ("closest", "m", "Closest purchased occurrence has enough prose", 0.0),
                ("largest", "m", "Later purchased occurrence " + "neutral " * 50, 10.0),
            ],
            query_axis=0.0,
            monkeypatch=monkeypatch,
        )
    )
    sessions = await list_archive(
        SessionQueryPlan(similar_text="anything", sort="words", limit=1, vector_provider=provider),
        archive_root=tmp_path,
        config=config,
    )
    assert [str(session.id) for session in sessions] == [identities[("largest", "m")][0]]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_predicate_sees_the_units_the_served_session_carries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit-reading predicate and the served page see the same projection.

    ``kept-late`` is filtered alone in a partial candidate chunk and served on
    a two-session page. Anti-vacuity: size the allowance by the chunk's own
    session count and the predicate sees three message rows while the served
    session carries one; reproject the served page by its own count and a
    rejected neighbour changes what a survivor receives.
    """
    _seed(tmp_path, "kept-late", updated_at="2026-01-01T00:00:00Z", messages=3)
    _seed(tmp_path, "rejected", updated_at="2026-01-02T00:00:00Z", messages=1)
    _seed(tmp_path, "kept-early", updated_at="2026-01-03T00:00:00Z", messages=3)
    monkeypatch.setattr("polylogue.archive.query.attached_units._MAX_ROWS_PER_PAGE", 5)
    monkeypatch.setattr("polylogue.archive.query.archive_execution._fetch_limit", lambda plan, *, default: 3)
    seen: dict[str, int] = {}

    def is_kept(session: Session) -> bool:
        seen[str(session.title)] = len(session.attached_units["message"])
        return str(session.title).startswith("kept")

    sessions = await list_archive(
        SessionQueryPlan(predicates=(is_kept,), limit=2),
        archive_root=tmp_path,
        config=None,
        with_units=("message",),
    )

    assert [session.title for session in sessions] == ["kept-early", "kept-late"]
    assert {str(session.title): len(session.attached_units["message"]) for session in sessions} == {
        "kept-early": seen["kept-early"],
        "kept-late": seen["kept-late"],
    }
    assert seen["kept-late"] == seen["kept-early"]


@pytest.mark.asyncio
async def test_complete_composed_sort_is_admitted_as_a_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-row composed sort reads every candidate, so it is a scan.

    Anti-vacuity: classify by the requested limit alone and this read is
    admitted as ``interactive``.
    """
    import polylogue.archive.query.archive_execution as execution

    _seed(tmp_path, "only", updated_at="2026-01-01T00:00:00Z", messages=1)
    classes: list[str] = []
    original = run_archive_read

    async def recording(*args: object, **kwargs: object) -> object:
        classes.append(str(kwargs["workload_class"]))
        return await original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(execution, "run_archive_read", recording)

    await list_archive(SessionQueryPlan(sort="messages", limit=1), archive_root=tmp_path, config=None)

    assert classes == ["scan"]


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered", [False, True])
async def test_a_complete_composed_sort_holds_only_its_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filtered: bool
) -> None:
    """A one-row composed page never sorts (and so never holds) every candidate at once.

    Anti-vacuity (Codex P2, #5695): collect every hydrated candidate before
    sorting and the sort sees all 250 sessions in one list.
    """
    from polylogue.archive.query import archive_execution
    from polylogue.archive.query import plan as plan_module

    for index in range(250):
        _seed(tmp_path, f"s{index:03d}", updated_at="2026-01-01T00:00:00Z", messages=1 + index % 7)
    widest: list[int] = []
    original = plan_module.SessionQueryPlan._sort_sessions

    def measured(self: SessionQueryPlan, sessions: list[Session]) -> list[Session]:
        widest.append(len(sessions))
        return original(self, sessions)

    monkeypatch.setattr(plan_module.SessionQueryPlan, "_sort_sessions", measured)
    predicates = (lambda session: True,) if filtered else ()
    sessions = await list_archive(
        SessionQueryPlan(sort="messages", limit=1, predicates=predicates), archive_root=tmp_path, config=None
    )

    assert len(sessions) == 1
    assert len(sessions[0].messages) == 7
    assert max(widest) <= archive_execution._COMPOSED_SORT_CHUNK + 1


@pytest.mark.asyncio
async def test_a_composed_sort_without_a_limit_serves_the_default_page(tmp_path: Path) -> None:
    """An omitted limit on a complete composed sort is the default page, not the archive.

    Anti-vacuity (Codex P1, #5695): leave the retained window unbounded when
    ``limit`` is None and all five sessions come back for a default of three.
    """
    for index in range(5):
        _seed(tmp_path, f"d{index}", updated_at="2026-01-01T00:00:00Z", messages=1 + index)

    sessions = await list_archive(
        SessionQueryPlan(sort="messages"), archive_root=tmp_path, config=None, default_limit=3
    )

    assert [len(session.messages) for session in sessions] == [5, 4, 3]


@pytest.mark.asyncio
async def test_a_sampled_composed_sort_samples_every_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sampled composed page draws from every qualified candidate, holding only the sample.

    Anti-vacuity (Codex P2, #5695): keep the top ``limit`` before sampling and
    only the largest session is ever sampled; lift the bound without a
    reservoir and ``_finalize`` receives every hydrated session.
    """
    import random

    from polylogue.archive.query import plan as plan_module

    for index in range(5):
        _seed(tmp_path, f"p{index}", updated_at="2026-01-01T00:00:00Z", messages=1 + index)
    offered: list[int] = []
    original = plan_module.SessionQueryPlan._finalize

    def finalize(self: SessionQueryPlan, items: list[Session]) -> list[Session]:
        offered.append(len(items))
        return original(self, items)

    monkeypatch.setattr(plan_module.SessionQueryPlan, "_finalize", finalize)
    drawn: set[int] = set()
    for seed in range(40):
        random.seed(seed)
        sessions = await list_archive(
            SessionQueryPlan(sort="messages", limit=1, sample=1), archive_root=tmp_path, config=None
        )
        drawn.update(len(session.messages) for session in sessions)

    assert set(offered) == {1}
    assert len(drawn) > 1


@pytest.mark.asyncio
async def test_units_of_a_default_composed_page_get_the_default_page_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Units of a limit-less composed page are projected with the default page's allowance.

    Anti-vacuity (Codex P2, #5695): leave ``unit_page`` unset when ``limit`` is
    omitted and each candidate chunk is projected with the chunk's width.
    """
    from polylogue.archive.query import archive_execution

    for index in range(4):
        _seed(tmp_path, f"u{index}", updated_at="2026-01-01T00:00:00Z", messages=1 + index)
    widths: list[int | None] = []
    original = archive_execution._attach_units_to_domain

    def attach(*args: Any, **kwargs: Any) -> Any:
        widths.append(kwargs.get("page_width"))
        return original(*args, **kwargs)

    monkeypatch.setattr(archive_execution, "_attach_units_to_domain", attach)
    await list_archive(
        SessionQueryPlan(sort="messages"), archive_root=tmp_path, config=None, default_limit=3, with_units=("messages",)
    )

    assert widths and set(widths) == {3}


@pytest.mark.asyncio
async def test_a_sampled_page_with_an_offset_samples_past_the_head(tmp_path: Path) -> None:
    """``offset`` drops the sort's head, then the sample draws from the rest.

    Anti-vacuity (Codex P2, #5695): hold only a sample-sized reservoir and the
    offset slice removes its only item, so every such page is empty.
    """
    import random

    for index in range(5):
        _seed(tmp_path, f"o{index}", updated_at="2026-01-01T00:00:00Z", messages=1 + index)
    drawn: set[int] = set()
    for seed in range(30):
        random.seed(seed)
        sessions = await list_archive(
            SessionQueryPlan(sort="messages", sample=1, offset=1, limit=1), archive_root=tmp_path, config=None
        )
        assert len(sessions) == 1
        drawn.add(len(sessions[0].messages))

    assert 5 not in drawn
    assert len(drawn) > 1


@pytest.mark.asyncio
async def test_units_of_a_sampled_page_get_the_sampled_width(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P2, #5695): size unit allowances by the pre-sample
    limit and one sampled session gets a fiftieth of its allowance."""
    from polylogue.archive.query import archive_execution

    for index in range(3):
        _seed(tmp_path, f"w{index}", updated_at="2026-01-01T00:00:00Z", messages=1 + index)
    widths: list[int | None] = []
    original = archive_execution._attach_units_to_domain

    def attach(*args: Any, **kwargs: Any) -> Any:
        widths.append(kwargs.get("page_width"))
        return original(*args, **kwargs)

    monkeypatch.setattr(archive_execution, "_attach_units_to_domain", attach)
    await list_archive(
        SessionQueryPlan(sort="messages", limit=50, sample=1),
        archive_root=tmp_path,
        config=None,
        with_units=("messages",),
    )

    assert widths and set(widths) == {1}


@pytest.mark.asyncio
async def test_a_complete_composed_sort_streams_its_candidate_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete scan hands candidate summaries to the reducer batch by batch.

    Anti-vacuity (Codex P2, #5695): accumulate every summary before reducing
    and the fetch returns all 250 rows in one list.
    """
    from polylogue.archive.query import archive_execution

    for index in range(250):
        _seed(tmp_path, f"s{index:03d}", updated_at="2026-01-01T00:00:00Z", messages=1 + index % 7)
    original = archive_execution._archive_summaries
    returned: list[int] = []
    streamed: list[int] = []

    def spied(*args: Any, **kwargs: Any) -> Any:
        on_batch = kwargs.get("on_batch")
        if on_batch is not None:

            def counting(rows: list[Any]) -> None:
                streamed.append(len(rows))
                on_batch(rows)

            kwargs["on_batch"] = counting
        result = original(*args, **kwargs)
        returned.append(len(result))
        return result

    monkeypatch.setattr(archive_execution, "_archive_summaries", spied)
    sessions = await list_archive(SessionQueryPlan(sort="messages", limit=1), archive_root=tmp_path, config=None)

    assert len(sessions) == 1 and len(sessions[0].messages) == 7
    assert returned == [0]
    assert sum(streamed) == 250


@pytest.mark.asyncio
async def test_units_of_an_unlimited_sampled_page_get_the_sample_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sample with no limit serves the whole sample, and its units that page's allowance.

    Anti-vacuity (Codex P2, #5695): fall back to the default page of 20 and a
    50-session sample is projected in 20-session chunks with a 20-page budget.
    """
    from polylogue.archive.query import archive_execution

    for index in range(60):
        _seed(tmp_path, f"u{index:02d}", updated_at="2026-01-01T00:00:00Z", messages=1 + index % 3)
    widths: list[int | None] = []
    original = archive_execution._attach_units_to_domain

    def attach(*args: Any, **kwargs: Any) -> Any:
        widths.append(kwargs.get("page_width"))
        return original(*args, **kwargs)

    monkeypatch.setattr(archive_execution, "_attach_units_to_domain", attach)
    sessions = await list_archive(
        SessionQueryPlan(sort="messages", sample=50), archive_root=tmp_path, config=None, with_units=("messages",)
    )

    assert len(sessions) == 50
    assert widths and set(widths) == {50}


@pytest.mark.asyncio
async def test_post_filtered_offset_applies_once_and_count_ignores_the_window(tmp_path: Path) -> None:
    """An offset skips survivors exactly once; a count reports the whole result.

    ``root`` is a residual post-filter. An unlimited post-filtered read used to
    pass the offset to SQL, over unfiltered candidates, and then skip that many
    survivors again, and ``count`` kept the page offset on its unbounded plan.
    Anti-vacuity: restore ``plan.limit is not None`` in ``post_filter_fetch``
    and the unlimited page loses four rows instead of two; keep the offset in
    ``count_archive`` and the count drops below five.
    """
    from polylogue.archive.query.archive_execution import count_archive

    for index in range(5):
        _seed(tmp_path, f"root-{index}", updated_at=f"2026-01-0{index + 1}T00:00:00Z", messages=1)

    everything = await list_summaries_archive(SessionQueryPlan(root=True), archive_root=tmp_path, config=None)
    unlimited = await list_summaries_archive(SessionQueryPlan(root=True, offset=2), archive_root=tmp_path, config=None)
    limited = await list_summaries_archive(
        SessionQueryPlan(root=True, offset=2, limit=10), archive_root=tmp_path, config=None
    )
    sessions = await list_archive(SessionQueryPlan(root=True, offset=2), archive_root=tmp_path, config=None)

    assert len(everything) == 5
    assert [summary.id for summary in unlimited] == [summary.id for summary in everything[2:]]
    assert [summary.id for summary in limited] == [summary.id for summary in everything[2:]]
    assert [session.id for session in sessions] == [summary.id for summary in everything[2:]]
    assert await count_archive(SessionQueryPlan(root=True, offset=2), archive_root=tmp_path, config=None) == 5
    assert await count_archive(SessionQueryPlan(root=True, offset=2, limit=1), archive_root=tmp_path, config=None) == 5
