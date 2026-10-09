"""Facade-level contracts for the sunk session-analysis primitives (#1691 / polylogue-9e5.24).

Scalar profile analytics reduce the full matched scope on the controlled
reader without full profile hydration. Correlation still requests the full
profile scope and validates metric names before paying for a fetch.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest

from polylogue import Polylogue
from polylogue.analysis.archive import ArchiveInferenceProvenance, ArchiveInsightProvenance, SessionProfileInsight
from polylogue.analysis.archive_models import SessionEvidencePayload, SessionInferencePayload
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.session_profiles import write_session_profile
from tests.infra.storage_records import SessionBuilder


def _archive_on_writer(tmp_path: Path) -> Polylogue:
    with ArchiveStore(tmp_path):
        pass
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")


def _archive(tmp_path: Path) -> Polylogue:
    """Run the synchronous seed off any running event loop."""
    return run_off_event_loop(lambda: _archive_on_writer(tmp_path))


def _provenance() -> ArchiveInsightProvenance:
    return ArchiveInsightProvenance(materializer_version=1, materialized_at="2026-03-24T10:00:00+00:00")


def _inference_provenance() -> ArchiveInferenceProvenance:
    return ArchiveInferenceProvenance(
        materializer_version=1,
        materialized_at="2026-03-24T10:00:00+00:00",
        inference_version=1,
        inference_family="heuristic_session_semantics",
    )


def _profile(session_id: str, *, message_count: int = 10, word_count: int = 100) -> SessionProfileInsight:
    return SessionProfileInsight(
        session_id=session_id,
        logical_session_id=session_id,
        origin="claude-code",
        title=session_id,
        provenance=_provenance(),
        semantic_tier="merged",
        evidence=SessionEvidencePayload(message_count=message_count, word_count=word_count),
        inference_provenance=_inference_provenance(),
        inference=SessionInferencePayload(workflow_shape="chat", terminal_state="resolved"),
    )


def _seed_analytics(tmp_path: Path, *, count: int = 3) -> tuple[Polylogue, list[str]]:
    poly = _archive(tmp_path)
    ids: list[str] = []
    for index in range(count):
        builder = (
            SessionBuilder(tmp_path / "index.db", f"profile-{index}")
            .provider("claude-code" if index % 2 == 0 else "codex")
            .created_at("2026-05-04T12:00:00Z")
            .updated_at("2026-05-04T12:00:00Z")
            .reported_cost_usd(0.25)
            .git_repository_url("https://example.test/selected.git")
        )
        builder.save()
        ids.append(builder.native_session_id())

    def write() -> None:
        with ArchiveStore(tmp_path) as archive:
            for index, session_id in enumerate(ids):
                write_session_profile(
                    archive._conn,
                    session_id,
                    workflow_shape="chat" if index % 2 == 0 else "agentic_loop",
                    terminal_state="question_left" if index % 2 == 0 else "truncated",
                    terminal_state_confidence=0.75,
                    canonical_session_date="2026-05-04" if index % 2 == 0 else None,
                    evidence={
                        "cwd_paths": ["/workspace/a", "/workspace/b"] if index % 2 == 0 else [],
                        "terminal_state_evidence": {"last_role": "assistant"},
                        "total_cost_usd": 0.25,
                    },
                    terminal_state_evidence_json='{"last_role":"assistant"}',
                    enrichment={"intent_summary": "neutral " * 1024},
                )
            archive.add_user_tags(tuple(ids[::2]), ("selected",))
            archive.commit()

    run_off_event_loop(write)
    return poly, ids


@pytest.mark.asyncio
async def test_profile_histograms_reduce_full_native_scope_without_payloads_or_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poly, _ = _seed_analytics(tmp_path, count=64)
    SessionBuilder(tmp_path / "index.db", "unprofiled").provider("claude-code").save()
    actual = ArchiveStore._session_profile_selection

    def guarded(archive: ArchiveStore, **kwargs: object) -> tuple[str, list[object]]:
        assert archive._conn.in_transaction
        selection = actual(archive, **kwargs)  # type: ignore[arg-type]
        forbidden = {
            "evidence_payload_json",
            "inference_payload_json",
            "enrichment_payload_json",
            "materializer_version",
        }

        def authorize(action: int, table: str | None, column: str | None, *_: str | None) -> int:
            if action == sqlite3.SQLITE_READ and (
                table == "session_model_usage" or (table == "session_profiles" and column in forbidden)
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        archive._conn.set_authorizer(authorize)
        return selection

    monkeypatch.setattr(ArchiveStore, "_session_profile_selection", guarded)
    result = await poly.aggregate_sessions(group_by="workflow_shape")
    assert result == {"group_by": "workflow_shape", "total_sessions": 64, "buckets": {"chat": 32, "agentic_loop": 32}}
    result = await poly.aggregate_sessions(group_by="origin", origin="claude-code", since="2026-05-04")
    assert result == {"group_by": "origin", "total_sessions": 32, "buckets": {"claude-code-session": 32}}
    assert (await poly.aggregate_sessions(group_by="terminal_state", until="2026-05-03"))["buckets"] == {}


@pytest.mark.asyncio
async def test_workflow_distributions_preserve_week_origin_and_plural_project_buckets(tmp_path: Path) -> None:
    poly, _ = _seed_analytics(tmp_path)
    assert await poly.workflow_shape_distribution(group_by="week") == {
        "group_by": "week",
        "total_sessions": 3,
        "buckets": {"2026-W19": {"chat": 2}, "undated": {"agentic_loop": 1}},
    }
    assert (await poly.workflow_shape_distribution(group_by="origin"))["buckets"] == {
        "claude-code-session": {"chat": 2},
        "codex-session": {"agentic_loop": 1},
    }
    assert (await poly.workflow_shape_distribution(group_by="project"))["buckets"] == {
        "/workspace/a": {"chat": 2},
        "/workspace/b": {"chat": 2},
        "unattributed": {"agentic_loop": 1},
    }


@pytest.mark.asyncio
async def test_abandoned_profiles_count_scope_before_evidence_page_and_preserve_slices(tmp_path: Path) -> None:
    poly, ids = _seed_analytics(tmp_path)
    result = await poly.find_abandoned_sessions(limit=1)
    assert result["total"] == 3
    assert result["items"] == [
        {
            "session_id": ids[0],
            "origin": "claude-code-session",
            "title": "Test Session",
            "terminal_state": "question_left",
            "terminal_state_confidence": 0.75,
            "workflow_shape": "chat",
            "canonical_session_date": "2026-05-04",
            "evidence": {"last_role": "assistant"},
        }
    ]
    assert (await poly.find_abandoned_sessions(limit=0)) == {"total": 3, "items": []}
    negative = await poly.find_abandoned_sessions(limit=-1)
    assert [item["session_id"] for item in cast(list[dict[str, object]], negative["items"])] == ids[::2]
    scoped = await poly.find_abandoned_sessions(tag="selected", repo="selected", limit=10)
    assert scoped["total"] == 2
    assert [item["session_id"] for item in cast(list[dict[str, object]], scoped["items"])] == ids[::2]
    severe = await poly.find_abandoned_sessions(min_severity="tool_left", limit=10)
    assert [item["session_id"] for item in cast(list[dict[str, object]], severe["items"])] == [ids[1]]


@pytest.mark.asyncio
async def test_profile_analytics_refuse_invalid_dimensions_and_use_read_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poly = _archive(tmp_path)
    with pytest.raises(ValueError):
        await poly.aggregate_sessions(group_by="invalid")
    with pytest.raises(ValueError):
        await poly.workflow_shape_distribution(group_by="invalid")
    with pytest.raises(ValueError):
        await poly.find_abandoned_sessions(min_severity="invalid")

    class CancelledReadError(Exception):
        pass

    def cancel(_: ArchiveStore) -> None:
        raise CancelledReadError

    monkeypatch.setattr(ArchiveStore, "check_operation_read", cancel)
    with pytest.raises(CancelledReadError):
        await poly.aggregate_sessions()


@pytest.mark.asyncio
async def test_profile_insight_iterator_reads_no_unused_usage_costs_but_record_keeps_money(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.analysis.archive import SessionProfileInsightQuery

    poly, ids = _seed_analytics(tmp_path)
    selection = ArchiveStore._session_profile_selection

    def guarded(archive: ArchiveStore, **kwargs: object) -> tuple[str, list[object]]:
        result = selection(archive, **kwargs)  # type: ignore[arg-type]

        def authorize(action: int, table: str | None, *_: str | None) -> int:
            return (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and table == "session_model_usage"
                else sqlite3.SQLITE_OK
            )

        archive._conn.set_authorizer(authorize)
        return result

    monkeypatch.setattr(ArchiveStore, "_session_profile_selection", guarded)
    profiles = await poly.list_session_profile_insights(SessionProfileInsightQuery(limit=None))
    assert {profile.session_id for profile in profiles} == set(ids)
    record = await poly.get_session_profile_record(ids[0])
    assert record is not None
    assert record.total_cost_usd == 0.25


@pytest.mark.asyncio
async def test_correlate_sessions_fetches_full_scope_not_a_page_limit(tmp_path: Path) -> None:
    poly = _archive(tmp_path)
    profiles = [_profile(f"c{i}", message_count=i, word_count=i * 2) for i in range(1, 1006)]
    fetch_mock = AsyncMock(return_value=profiles)
    poly.list_session_profile_insights = fetch_mock  # type: ignore[method-assign]

    result = await poly.correlate_sessions(metric_x="message_count", metric_y="word_count")

    assert result["sample_count"] == 1005
    assert fetch_mock.await_args is not None
    assert fetch_mock.await_args.args[0].limit is None


@pytest.mark.asyncio
async def test_correlate_sessions_validates_metrics_before_fetching(tmp_path: Path) -> None:
    poly = _archive(tmp_path)
    poly.list_session_profile_insights = AsyncMock(return_value=[])  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Unknown metric_x"):
        await poly.correlate_sessions(metric_x="favorite_color", metric_y="word_count")

    poly.list_session_profile_insights.assert_not_awaited()


@pytest.mark.asyncio
async def test_compare_sessions_rejects_out_of_range_counts(tmp_path: Path) -> None:
    poly = _archive(tmp_path)

    with pytest.raises(ValueError, match="Need at least 2"):
        await poly.compare_sessions(["only-one"])

    with pytest.raises(ValueError, match="Too many"):
        await poly.compare_sessions([f"c{i}" for i in range(11)])


@pytest.mark.asyncio
async def test_find_similar_sessions_by_metadata_returns_none_for_unknown_session(tmp_path: Path) -> None:
    poly = _archive(tmp_path)
    poly.get_session_profile_insight = AsyncMock(return_value=None)  # type: ignore[method-assign]

    result = await poly.find_similar_sessions_by_metadata("nonexistent")

    assert result is None
