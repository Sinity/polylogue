"""Codex search-lane regressions through production archive and surface routes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.api import Polylogue
from polylogue.api.search_envelope_builder import build_search_envelope_for_spec
from polylogue.archive.query.archive_execution import archive_search_hits, count_archive
from polylogue.archive.query.execution_control import (
    QueryCancelledError,
    QueryTimeoutError,
    QueryWorkBudgetExceededError,
)
from polylogue.archive.query.plan import SessionQueryPlan
from polylogue.archive.query.search_hits import project_search_hits
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.query_output import format_search_envelope
from polylogue.config import Config, Source
from polylogue.core.errors import EmbeddingRetrievalNotReadyError
from polylogue.core.protocols import VectorProvider
from polylogue.mcp.archive_support import archive_search_payload
from polylogue.mcp.payloads import session_search_result_payload
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.connection_profile import ReadFrameCancelledError, ReadFrameExpiredError
from polylogue.surfaces.payloads import decode_search_cursor
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.storage_records import SessionBuilder

LaneArchive = tuple[Path, Config, dict[str, str]]


class _VectorReply:
    """Only the external vector service is substituted; SQL/hydration are real."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def query(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        del text, limit
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return []


@pytest.fixture
def lane_archive(tmp_path: Path) -> tuple[Path, Config, dict[str, str]]:
    root = tmp_path / "archive"
    bootstrap_archive_root(root)
    ids: dict[str, str] = {}
    for name in ("dialogue", "action-one", "action-two"):
        builder = SessionBuilder(root / "index.db", name).provider("codex").title(name)
        if name == "dialogue":
            builder.add_message("plain", role="user", text="needle in ordinary conversation")
        else:
            # Repeated action hits in one session must not starve a distinct
            # session from an action page. The storage lowering owns these IDs.
            for index in range(4):
                builder.add_message(
                    f"tool-{index}",
                    role="assistant",
                    text="",
                    blocks=[
                        {
                            "type": "tool_use",
                            "tool_id": f"call-{index}",
                            "tool_name": "shell",
                            "tool_input": {"command": "needle"},
                            "text": "needle in a command",
                        }
                    ],
                )
        builder.save()
        ids[name] = builder.native_session_id()
    config = Config(
        archive_root=root,
        render_root=tmp_path / "render",
        db_path=root / "index.db",
        sources=[Source(name="fixture", path=tmp_path / "inbox")],
    )
    return root, config, ids


def test_actions_without_embeddings_return_only_action_evidence(lane_archive: LaneArchive) -> None:
    """F871: treating actions as semantic refuses; generic FTS admits dialogue."""
    root, _config, ids = lane_archive
    with open_operation_read(root) as pinned:
        result = archive_search_hits(
            SessionQueryPlan(query_terms=("needle",), retrieval_lane="actions", limit=2),
            archive_root=root,
            config=None,
            archive=pinned.archive,
        )
    assert {hit.session_id for hit, _summary in result.hits} == {ids["action-one"], ids["action-two"]}
    assert result.retrieval_lane == "actions"
    assert result.execution.requested_lanes == result.execution.executed_lanes == ("action",)
    assert not result.execution.degraded


@pytest.mark.asyncio
async def test_actions_lane_count_counts_only_sessions_its_search_returns(lane_archive: LaneArchive) -> None:
    """The actions-lane count and the actions-lane search read one relation.

    Anti-vacuity: drop ``actions_only`` from the count route and the dialogue
    session, which the actions search never returns, is counted too (3).
    """
    root, _config, _ids = lane_archive
    plan = SessionQueryPlan(query_terms=("needle",), retrieval_lane="actions")
    assert await count_archive(plan, archive_root=root, config=None) == 2
    dialogue = replace(plan, retrieval_lane="dialogue")
    assert await count_archive(dialogue, archive_root=root, config=None) == 3


def test_vector_execution_failure_degrades_hybrid_and_refuses_semantic(lane_archive: LaneArchive) -> None:
    """F872: an uncaught query failure loses real lexical hits and leaks its detail."""
    root, config, ids = lane_archive
    backend = _VectorReply(RuntimeError("sensitive provider detail must not escape"))
    with open_operation_read(root) as pinned:
        hybrid = archive_search_hits(
            SessionQueryPlan(
                query_terms=("needle",),
                retrieval_lane="hybrid",
                limit=10,
                vector_provider=cast(VectorProvider, backend),
            ),
            archive_root=root,
            config=config,
            archive=pinned.archive,
        )
        with pytest.raises(EmbeddingRetrievalNotReadyError) as refusal:
            archive_search_hits(
                SessionQueryPlan(
                    similar_text="needle",
                    retrieval_lane="semantic",
                    vector_provider=cast(VectorProvider, backend),
                ),
                archive_root=root,
                config=config,
                archive=pinned.archive,
            )
    assert {hit.session_id for hit, _summary in hybrid.hits} == set(ids.values())
    assert hybrid.execution.failed_lanes[0].kind == "execution_failed"
    assert hybrid.execution.failed_lanes[0].reason == "RuntimeError"
    assert hybrid.execution.executed_lanes == ("text", "action")
    assert refusal.value.readiness_status == "failed"
    assert "sensitive provider detail" not in str(refusal.value)
    assert backend.calls == 2


@pytest.mark.parametrize(
    "error_type",
    [
        QueryCancelledError,
        QueryTimeoutError,
        QueryWorkBudgetExceededError,
        ReadFrameCancelledError,
        ReadFrameExpiredError,
    ],
)
def test_vector_read_cancellation_is_not_a_successful_hybrid_degradation(
    lane_archive: LaneArchive, error_type: type[Exception]
) -> None:
    """F872 guard: catching cancellation as an optional-lane error conceals aborted reads."""
    root, config, _ids = lane_archive
    backend = _VectorReply(error_type("cancelled read"))
    with open_operation_read(root) as pinned, pytest.raises(error_type):
        archive_search_hits(
            SessionQueryPlan(
                query_terms=("needle",),
                retrieval_lane="hybrid",
                vector_provider=cast(VectorProvider, backend),
            ),
            archive_root=root,
            config=config,
            archive=pinned.archive,
        )


@pytest.mark.parametrize("query", ["needle", "nomatchingword"])
def test_cli_and_both_mcp_builders_preserve_degraded_execution(lane_archive: LaneArchive, query: str) -> None:
    """F873: omitting execution reports an unavailable vector lane as healthy, even on empty pages."""
    root, _config, _ids = lane_archive
    spec = SessionQuerySpec.from_params(
        {"query": (query,), "retrieval_lane": "hybrid", "limit": 10},
        strict=True,
    )
    with open_operation_read(root) as pinned:
        plan = spec.to_plan()
        native = archive_search_hits(plan, archive_root=root, config=None, archive=pinned.archive)
        hits = project_search_hits(plan, native)
        mcp_native = archive_search_payload(
            pinned.archive,
            spec,
            query=query,
            limit=10,
            offset=0,
            sort=None,
            archive_root=root,
        )
    cli = json.loads(
        format_search_envelope(
            hits,
            query=query,
            retrieval_lane="hybrid",
            limit=10,
            offset=0,
            sort=None,
        )
    )
    mcp_hits = session_search_result_payload(
        hits,
        query=query,
        retrieval_lane="hybrid",
        total=None,
        limit=10,
        offset=0,
    )
    for envelope in (cli, mcp_hits.model_dump(mode="json"), mcp_native.model_dump(mode="json")):
        assert envelope["requested_lanes"] == ["text", "action", "vector"]
        assert envelope["executed_lanes"] == ["text", "action"]
        assert envelope["unavailable_lanes"] == ["vector"]
        assert envelope["advisories"]
        assert envelope["outcome"]["state"] == "degraded"
        assert bool(envelope["hits"]) is (query == "needle")


@pytest.mark.asyncio
async def test_hybrid_api_does_not_publish_semantic_only_total(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F874: the semantic-only count returns zero despite a full lexical union page."""
    root, _config, _ids = lane_archive
    backend = _VectorReply()
    monkeypatch.setattr(
        "polylogue.storage.search_providers.create_vector_provider",
        lambda *args, **kwargs: backend,
    )
    spec = SessionQuerySpec.from_params(
        {"query": ("needle",), "retrieval_lane": "hybrid", "limit": 1},
        strict=True,
    )
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
        assert len(first.hits) == 1
        assert first.total is None  # Qualified unknown, never the unrelated vector count.
        assert first.next_offset == 1
        assert first.next_cursor is not None
        second = await build_search_envelope_for_spec(facade, replace(spec, cursor=first.next_cursor))
    assert len(second.hits) == 1
    assert second.hits[0].session.id != first.hits[0].session.id
    assert backend.calls == 2  # Counts must not rerun a different vector-only relation.


@pytest.mark.asyncio
async def test_degraded_hybrid_api_cursor_keeps_request_lane(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F877: a dialogue-labelled cursor is rejected by its originating hybrid request."""
    root, _config, _ids = lane_archive
    monkeypatch.setattr("polylogue.storage.search_providers.create_vector_provider", lambda *a, **k: None)
    spec = SessionQuerySpec.from_params(
        {"query": ("needle",), "retrieval_lane": "hybrid", "limit": 1},
        strict=True,
    )
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
        assert first.next_cursor is not None
        assert decode_search_cursor(first.next_cursor).lane == "hybrid"
        second = await build_search_envelope_for_spec(facade, replace(spec, cursor=first.next_cursor))
    assert second.offset == 1
    assert second.hits and second.hits[0].session.id != first.hits[0].session.id
    assert second.executed_lanes == ("text", "action")
    assert second.unavailable_lanes == ("vector",)


def test_degraded_hybrid_daemon_cursor_keeps_request_lane(lane_archive: LaneArchive) -> None:
    """F877: the pinned daemon path must also consume its own degraded cursor."""
    root, _config, _ids = lane_archive
    params = {"query": ("needle",), "retrieval_lane": "hybrid", "limit": 1}
    with open_operation_read(root) as pinned:
        first = execute_read_operation(
            "cli.query",
            {"params": params},
            archive=pinned.archive,
            serving_identity="daemon",
        )
        assert first["next_cursor"]
        second = execute_read_operation(
            "cli.query",
            {"params": {**params, "cursor": first["next_cursor"]}},
            archive=pinned.archive,
            serving_identity="daemon",
        )
    assert second["offset"] == 1
    assert second["retrieval_lane"] == "hybrid"
    assert second["executed_lanes"] == ["text", "action"]


def test_degraded_hybrid_fuses_action_rank_contributions(lane_archive: LaneArchive) -> None:
    """F879: returning generic FTS directly omits action membership and its RRF contribution."""
    root, _config, ids = lane_archive
    with open_operation_read(root) as pinned:
        result = archive_search_hits(
            SessionQueryPlan(query_terms=("needle",), retrieval_lane="hybrid", limit=10),
            archive_root=root,
            config=None,
            archive=pinned.archive,
        )
    by_id = {hit.session_id: hit for hit, _summary in result.hits}
    assert set(by_id) == set(ids.values())
    assert result.execution.requested_lanes == ("text", "action", "vector")
    assert result.execution.executed_lanes == ("text", "action")
    for name, hit in ((name, by_id[session_id]) for name, session_id in ids.items()):
        assert hit.lane_ranks is not None
        assert (hit.lane_ranks["action"] is None) is (name == "dialogue")
    # With only three candidates, an extra positive action contribution beats
    # any text-only contribution, independently of FTS's ordering of the three.
    assert {hit.session_id for hit, _summary in result.hits[:2]} == {ids["action-one"], ids["action-two"]}


def test_archive_read_failure_is_not_relabelled_as_a_degraded_vector_lane(
    lane_archive: LaneArchive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F872 guard: wrapping ``semantic_summaries`` in the vector-leg handler hides index faults.

    The vector service answered; the archive's own hydration read failed. That
    is not an optional-lane outcome, so hybrid must raise rather than report a
    lexical page with a ``failed_lanes`` entry.
    """
    import sqlite3

    root, config, _ids = lane_archive
    backend = _VectorReply()
    with open_operation_read(root) as pinned:

        def broken_semantic_summaries(*_args: object, **_kwargs: object) -> list[object]:
            raise sqlite3.DatabaseError("index page corrupt")

        monkeypatch.setattr(pinned.archive, "semantic_summaries", broken_semantic_summaries)
        with pytest.raises(sqlite3.DatabaseError, match="index page corrupt"):
            archive_search_hits(
                SessionQueryPlan(
                    query_terms=("needle",),
                    retrieval_lane="hybrid",
                    vector_provider=cast(VectorProvider, backend),
                ),
                archive_root=root,
                config=config,
                archive=pinned.archive,
            )
    assert backend.calls == 1
