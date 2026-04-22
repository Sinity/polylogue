"""Neighboring-conversation candidate discovery tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.lib.neighbor_candidates import NeighborDiscoveryRequest, discover_neighbor_candidates
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from tests.infra.storage_records import ConversationBuilder


async def _discover(
    db_path: Path,
    request: NeighborDiscoveryRequest,
) -> list[str]:
    async with ConversationRepository(backend=SQLiteBackend(db_path=db_path)) as repo:
        candidates = await discover_neighbor_candidates(repo, request)
    return [candidate.conversation_id for candidate in candidates]


@pytest.mark.asyncio
async def test_same_title_different_body_is_explainable_candidate(db_path: Path) -> None:
    (
        ConversationBuilder(db_path, "target")
        .provider("claude-ai")
        .title("Vault Analysis")
        .updated_at("2026-04-22T10:00:00+00:00")
        .add_message("target-user", role="user", text="Map the archive verification roadmap.")
        .save()
    )
    (
        ConversationBuilder(db_path, "candidate")
        .provider("claude-ai")
        .title("Vault Analysis")
        .updated_at("2026-04-20T10:00:00+00:00")
        .add_message("candidate-user", role="user", text="Discuss unrelated dashboard styling decisions.")
        .save()
    )

    async with ConversationRepository(backend=SQLiteBackend(db_path=db_path)) as repo:
        candidates = await discover_neighbor_candidates(
            repo,
            NeighborDiscoveryRequest(conversation_id="target", window_hours=1),
        )

    assert [candidate.conversation_id for candidate in candidates] == ["candidate"]
    assert {reason.kind for reason in candidates[0].reasons} == {"same_title"}
    assert "same normalized title" in candidates[0].reasons[0].detail


@pytest.mark.asyncio
async def test_nearby_similar_content_gets_time_and_content_reasons(db_path: Path) -> None:
    (
        ConversationBuilder(db_path, "target")
        .provider("codex")
        .title("Checkpoint Failure")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("target-user", role="user", text="Debug sqlite checkpoint lock retry recovery in archive writes.")
        .save()
    )
    (
        ConversationBuilder(db_path, "candidate")
        .provider("codex")
        .title("Archive Lock Retries")
        .updated_at("2026-04-22T14:00:00+00:00")
        .add_message("candidate-user", role="user", text="Investigate sqlite checkpoint lock retry recovery behavior.")
        .save()
    )
    (
        ConversationBuilder(db_path, "distractor")
        .provider("codex")
        .title("Visual Cleanup")
        .updated_at("2026-04-22T13:00:00+00:00")
        .add_message("distractor-user", role="user", text="Adjust typography and color spacing.")
        .save()
    )

    async with ConversationRepository(backend=SQLiteBackend(db_path=db_path)) as repo:
        candidates = await discover_neighbor_candidates(
            repo,
            NeighborDiscoveryRequest(conversation_id="target", window_hours=6),
        )

    candidate = next(candidate for candidate in candidates if candidate.conversation_id == "candidate")
    assert {"nearby_time", "content_similarity"} <= {reason.kind for reason in candidate.reasons}
    assert candidate.rank == 1


@pytest.mark.asyncio
async def test_source_content_seed_finds_similar_candidate_outside_time_window(db_path: Path) -> None:
    (
        ConversationBuilder(db_path, "target")
        .provider("codex")
        .title("Checkpoint Failure")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("target-user", role="user", text="Debug sqlite checkpoint lock retry recovery in archive writes.")
        .save()
    )
    (
        ConversationBuilder(db_path, "candidate")
        .provider("codex")
        .title("Different Title")
        .updated_at("2026-04-18T12:00:00+00:00")
        .add_message("candidate-user", role="user", text="Investigate sqlite checkpoint lock retry recovery behavior.")
        .save()
    )

    async with ConversationRepository(backend=SQLiteBackend(db_path=db_path)) as repo:
        candidates = await discover_neighbor_candidates(
            repo,
            NeighborDiscoveryRequest(conversation_id="target", window_hours=1),
        )

    candidate = next(candidate for candidate in candidates if candidate.conversation_id == "candidate")
    reason_kinds = {reason.kind for reason in candidate.reasons}
    assert {"content_search", "content_similarity"} <= reason_kinds
    assert "nearby_time" not in reason_kinds


@pytest.mark.asyncio
async def test_query_seed_returns_explainable_candidates(db_path: Path) -> None:
    (
        ConversationBuilder(db_path, "query-candidate")
        .provider("chatgpt")
        .title("Schema Audit")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("query-user", role="user", text="Run schema audit checks for provider annotations.")
        .save()
    )

    ids = await _discover(
        db_path,
        NeighborDiscoveryRequest(query="schema audit", provider="chatgpt"),
    )

    assert ids == ["query-candidate"]
