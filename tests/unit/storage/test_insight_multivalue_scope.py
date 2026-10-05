from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from polylogue.analysis.archive import SessionLatencyProfileInsight, SessionProfileInsight
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.frozen_clock import FrozenClock
from tests.infra.storage_records import seed_insight_scope_archive


@pytest.mark.frozen_clock_modules("tests.infra.storage_records")
@pytest.mark.parametrize("reader", ["profile", "latency", "stuck"])
def test_insight_multivalue_scope_filters_before_paging(tmp_path: Path, reader: str, frozen_clock: FrozenClock) -> None:
    seed_insight_scope_archive(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:

        def read(
            *, repo: str, tag: str, limit: int | None, offset: int = 0
        ) -> Sequence[SessionProfileInsight | SessionLatencyProfileInsight]:
            if reader == "profile":
                return archive.list_session_profile_insights(repo=repo, tag=tag, limit=limit, offset=offset)
            if reader == "latency":
                return archive.list_session_latency_profile_insights(repo=repo, tag=tag, limit=limit, offset=offset)
            return archive.find_stuck_session_latency_profile_insights(repo=repo, tag=tag, limit=limit, offset=offset)

        scoped = read(repo=" alpha, beta, ", tag="alpha,beta", limit=1)
        following = read(repo="alpha,beta", tag=" alpha, beta ", limit=1, offset=1)
        intersection = read(repo="alpha,beta", tag="beta", limit=None)
        absent = read(repo="alpha,beta", tag="other", limit=None)
        empty = read(repo=" , ", tag=" , ", limit=None)
    assert [row.session_id for row in scoped] == ["claude-code-session:ext-beta"]
    assert [row.session_id for row in following] == ["claude-code-session:ext-alpha"]
    assert [row.session_id for row in intersection] == ["claude-code-session:ext-beta"]
    assert absent == []
    assert {row.session_id for row in empty} == {
        "claude-code-session:ext-alpha",
        "claude-code-session:ext-beta",
        "claude-code-session:ext-other",
    }
