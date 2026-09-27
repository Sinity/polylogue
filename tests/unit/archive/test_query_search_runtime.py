from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.models import Session
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.retrieval import search_query_text
from polylogue.archive.query.search_hits import (
    plan_has_search_hit_evidence,
    search_hits_for_plan,
)
from polylogue.config import Config, Source
from polylogue.core.enums import Provider
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.builders import make_conv, make_msg
from tests.infra.storage_records import SessionBuilder


@dataclass(frozen=True)
class _Request:
    limit: int | None = None
    offset: int = 0

    def with_limit(self, limit: int) -> _Request:
        return replace(self, limit=limit)

    def with_offset(self, offset: int) -> _Request:
        return replace(self, offset=offset)


def _session(session_id: str, *, text: str = "needle here", updated_hour: int = 0) -> Session:
    return make_conv(
        id=session_id,
        provider=Provider.CLAUDE_CODE,
        updated_at=datetime(2026, 4, 23, updated_hour, tzinfo=timezone.utc),
        messages=[make_msg(id=f"{session_id}-m1", role="assistant", text=text)],
    )


def test_search_query_text_joins_nonblank_terms() -> None:
    plan = SessionQueryPlan(query_terms=("Alpha",), contains_terms=("Beta", ""))

    assert search_query_text(plan) == "Alpha Beta"


@pytest.mark.asyncio
async def test_search_hits_for_plan_handles_empty_and_lexical_paths(tmp_path: Path) -> None:
    """``search_hits_for_plan`` executes over the archive.

    The function now takes ``(plan, config)`` (no repository) and resolves
    hits through ``archive_search_hits``. This pins the empty-evidence,
    whitespace-only, and lexical (``dialogue``) contracts against a real
    seeded ``index.db``.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root = tmp_path / "render"
    render_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"

    (
        SessionBuilder(db_path, "conv-summary")
        .provider(Provider.CHATGPT.value)
        .title("Needle Doc")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("m1", role="user", text="needle in the haystack here")
        .save()
    )

    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[Source(name="test", path=tmp_path / "inbox")],
        db_path=db_path,
    )

    # A plan with no search-bearing fields carries no evidence.
    assert plan_has_search_hit_evidence(SessionQueryPlan()) is False

    # Whitespace-only query terms degrade to no hits.
    assert await search_hits_for_plan(SessionQueryPlan(query_terms=("   ",)), config) == []

    # Lexical (dialogue) lane returns evidence-bearing hits with the native
    # session id, the dialogue lane label, and an FTS snippet.
    lexical_hits = await search_hits_for_plan(
        SessionQueryPlan(
            query_terms=("needle",),
            origins=("chatgpt-export",),
            limit=5,
            since=datetime(2026, 4, 20, tzinfo=timezone.utc),
        ),
        config,
    )

    native_id = native_session_id_for("chatgpt", "conv-summary")
    assert [hit.session_id for hit in lexical_hits] == [native_id]
    assert lexical_hits[0].retrieval_lane == "dialogue"
    assert "needle" in (lexical_hits[0].snippet or "")


@pytest.mark.asyncio
async def test_search_hits_for_plan_refuses_semantic_and_degrades_hybrid_without_embeddings(tmp_path: Path) -> None:
    """polylogue-9onfa: unavailable semantic evidence is never an empty success."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root = tmp_path / "render"
    render_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"

    (
        SessionBuilder(db_path, "conv-semantic")
        .provider(Provider.CHATGPT.value)
        .title("Needle Doc")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("m1", role="user", text="needle in the haystack here")
        .save()
    )

    # No vector provider is configured (no Voyage key / sqlite-vec backend), so
    # create_vector_provider returns None for this archive regardless of whether
    # an empty embeddings.db was bootstrapped alongside the index.
    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[Source(name="test", path=tmp_path / "inbox")],
        db_path=db_path,
    )

    from polylogue.core.errors import EmbeddingRetrievalNotReadyError

    with pytest.raises(EmbeddingRetrievalNotReadyError):
        await search_hits_for_plan(
            SessionQueryPlan(similar_text="needle", retrieval_lane="semantic"),
            config,
        )

    # Hybrid keeps lexical evidence and exposes the missing vector lane.
    hybrid_hits = await search_hits_for_plan(
        SessionQueryPlan(query_terms=("needle",), retrieval_lane="hybrid"),
        config,
    )
    assert len(hybrid_hits) == 1
    from polylogue.archive.query.search_hits import SearchHitResults

    assert cast(SearchHitResults, hybrid_hits).execution.unavailable_lanes == ("vector",)


@pytest.mark.asyncio
async def test_search_hits_origin_filter_excludes_other_origins(tmp_path: Path) -> None:
    """An origin filter over identical FTS content selects only that origin.

    Regression for the origin/provider seam (#2820 review P1): the FTS leg
    passed provider tokens (``chatgpt``) into a SQL filter over origin tokens
    (``chatgpt-export``), silently matching nothing — or, with the filter
    dropped, everything. Two origins sharing the same searchable text pin both
    failure directions against a real seeded ``index.db``: mutating the
    production filter tokens or removing the origin restriction each flips an
    assertion.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root = tmp_path / "render"
    render_root.mkdir(parents=True, exist_ok=True)
    db_path = archive_root / "index.db"

    (
        SessionBuilder(db_path, "conv-chatgpt")
        .provider(Provider.CHATGPT.value)
        .title("Needle One")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("m1", role="user", text="shared needle payload for origin filtering")
        .save()
    )
    (
        SessionBuilder(db_path, "conv-codex")
        .provider(Provider.CODEX.value)
        .title("Needle Two")
        .updated_at("2026-04-22T13:00:00+00:00")
        .add_message("m1", role="user", text="shared needle payload for origin filtering")
        .save()
    )

    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=[Source(name="test", path=tmp_path / "inbox")],
        db_path=db_path,
    )

    async def _hit_ids(origins: tuple[str, ...] | None) -> list[str]:
        hits = await search_hits_for_plan(
            SessionQueryPlan(query_terms=("needle",), origins=origins or (), limit=10),
            config,
        )
        return sorted(hit.session_id for hit in hits)

    chatgpt_id = native_session_id_for("chatgpt", "conv-chatgpt")
    codex_id = native_session_id_for("codex", "conv-codex")

    assert await _hit_ids(("chatgpt-export",)) == [chatgpt_id]
    assert await _hit_ids(("codex-session",)) == [codex_id]
    assert await _hit_ids(None) == sorted([chatgpt_id, codex_id])
