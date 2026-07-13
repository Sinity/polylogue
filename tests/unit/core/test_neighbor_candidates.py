"""Neighboring-session candidate discovery tests for the archive."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.archive.session.neighbor_candidates import (
    NeighborDiscoveryRequest,
    SessionNeighborCandidate,
)
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import SessionBuilder


async def _discover(
    db_path: Path,
    request: NeighborDiscoveryRequest,
) -> list[str]:
    archive = Polylogue(archive_root=db_path.parent, db_path=db_path)
    try:
        candidates = await archive.neighbor_candidates(
            session_id=request.session_id,
            query=request.query,
            provider=request.origin,
            limit=request.limit,
            window_hours=request.window_hours,
        )
    finally:
        await archive.close()
    return [candidate.session_id for candidate in candidates]


async def _discover_candidates(db_path: Path, request: NeighborDiscoveryRequest) -> list[SessionNeighborCandidate]:
    archive = Polylogue(archive_root=db_path.parent, db_path=db_path)
    try:
        return await archive.neighbor_candidates(
            session_id=request.session_id,
            query=request.query,
            provider=request.origin,
            limit=request.limit,
            window_hours=request.window_hours,
        )
    finally:
        await archive.close()


@pytest.mark.asyncio
async def test_same_title_different_body_is_explainable_candidate(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "target")
        .provider("claude-ai")
        .title("Vault Analysis")
        .updated_at("2026-04-22T10:00:00+00:00")
        .add_message("target-user", role="user", text="Map the archive verification roadmap.")
        .save()
    )
    (
        SessionBuilder(db_path, "candidate")
        .provider("claude-ai")
        .title("Vault Analysis")
        .updated_at("2026-04-20T10:00:00+00:00")
        .add_message("candidate-user", role="user", text="Discuss unrelated dashboard styling decisions.")
        .save()
    )

    target_id = native_session_id_for("claude-ai", "target")
    candidate_id = native_session_id_for("claude-ai", "candidate")

    candidates = await _discover_candidates(
        db_path,
        NeighborDiscoveryRequest(session_id=target_id, window_hours=1),
    )

    assert [candidate.session_id for candidate in candidates] == [candidate_id]
    assert {reason.kind for reason in candidates[0].reasons} == {"same_title"}
    assert "same normalized title" in candidates[0].reasons[0].detail


@pytest.mark.asyncio
async def test_nearby_similar_content_gets_time_and_content_reasons(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "target")
        .provider("codex")
        .title("Checkpoint Failure")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("target-user", role="user", text="Debug sqlite checkpoint lock retry recovery in archive writes.")
        .save()
    )
    (
        SessionBuilder(db_path, "candidate")
        .provider("codex")
        .title("Archive Lock Retries")
        .updated_at("2026-04-22T14:00:00+00:00")
        .add_message("candidate-user", role="user", text="Investigate sqlite checkpoint lock retry recovery behavior.")
        .save()
    )
    (
        SessionBuilder(db_path, "distractor")
        .provider("codex")
        .title("Visual Cleanup")
        .updated_at("2026-04-22T13:00:00+00:00")
        .add_message("distractor-user", role="user", text="Adjust typography and color spacing.")
        .save()
    )

    target_id = native_session_id_for("codex", "target")
    candidate_id = native_session_id_for("codex", "candidate")

    candidates = await _discover_candidates(
        db_path,
        NeighborDiscoveryRequest(session_id=target_id, window_hours=6),
    )

    candidate = next(candidate for candidate in candidates if candidate.session_id == candidate_id)
    assert {"nearby_time", "content_similarity"} <= {reason.kind for reason in candidate.reasons}
    assert candidate.rank == 1


@pytest.mark.asyncio
async def test_source_content_seed_finds_similar_candidate_outside_time_window(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "target")
        .provider("codex")
        .title("Checkpoint Failure")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("target-user", role="user", text="Debug sqlite checkpoint lock retry recovery in archive writes.")
        .save()
    )
    (
        SessionBuilder(db_path, "candidate")
        .provider("codex")
        .title("Different Title")
        .updated_at("2026-04-18T12:00:00+00:00")
        .add_message("candidate-user", role="user", text="Investigate sqlite checkpoint lock retry recovery behavior.")
        .save()
    )

    target_id = native_session_id_for("codex", "target")
    candidate_id = native_session_id_for("codex", "candidate")

    candidates = await _discover_candidates(
        db_path,
        NeighborDiscoveryRequest(session_id=target_id, window_hours=1),
    )

    candidate = next(candidate for candidate in candidates if candidate.session_id == candidate_id)
    reason_kinds = {reason.kind for reason in candidate.reasons}
    assert {"content_search", "content_similarity"} <= reason_kinds
    assert "nearby_time" not in reason_kinds


@pytest.mark.asyncio
async def test_query_seed_returns_explainable_candidates(db_path: Path) -> None:
    (
        SessionBuilder(db_path, "query-candidate")
        .provider("chatgpt")
        .title("Schema Audit")
        .updated_at("2026-04-22T12:00:00+00:00")
        .add_message("query-user", role="user", text="Run schema audit checks for provider annotations.")
        .save()
    )

    ids = await _discover(
        db_path,
        NeighborDiscoveryRequest(query="schema audit", origin="chatgpt"),
    )

    assert ids == [native_session_id_for("chatgpt", "query-candidate")]
