"""SQL is the one ordering authority for lexical and structured session queries.

The executor used to re-sort every SQL-ordered result in Python by
``updated_at``, which discarded the requested sort and disagreed with SQL's
``COALESCE(updated_at, created_at)`` key. A post-filtered page therefore had
to read every candidate before it could cut, because the Python order could
move a late row ahead of a kept one (polylogue-6xrab, polylogue-ztm1t).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.query.archive_execution import list_archive, list_summaries_archive
from polylogue.archive.query.plan import SessionQueryPlan
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
    import polylogue.archive.query.archive_execution as execution

    _seed(tmp_path, "child", updated_at="2026-01-01T00:00:00Z", messages=1)
    _seed(tmp_path, "standalone", updated_at="2026-01-02T00:00:00Z", messages=2)
    original = execution.archive_envelope_to_session

    def composing(*args: object, **kwargs: object) -> object:
        session = original(*args, **kwargs)  # type: ignore[arg-type]
        if session.title == "child":
            return session.model_copy(update={"messages": list(session.messages) * 11})
        return session

    monkeypatch.setattr(execution, "archive_envelope_to_session", composing)

    sessions = await list_archive(SessionQueryPlan(sort="messages", limit=1), archive_root=tmp_path, config=None)

    assert [session.title for session in sessions] == ["child"]


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
