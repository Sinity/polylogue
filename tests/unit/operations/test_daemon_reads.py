"""Pinned-reader conformance for declared daemon read operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from polylogue.operations.daemon_reads import (
    DaemonReadDependencies,
    execute_read_operation,
    requires_vector_snapshot,
    vector_binding_from_config,
)
from polylogue.operations.operation_context import open_operation_read
from tests.infra.archive_templates import bootstrap_archive_root


@dataclass
class _Stats:
    def to_dict(self) -> dict[str, object]:
        return {"total_sessions": 3, "total_messages": 8, "retrieval_ready": False}


class _Archive:
    def stats(self) -> _Stats:
        return _Stats()


def test_status_preserves_runtime_contract_without_using_cached_archive_evidence(tmp_path: Path) -> None:
    """Mutation: merge the cached snapshot last and stale archive facts win."""
    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        result = execute_read_operation(
            "status",
            {},
            archive=pinned.archive,
            serving_identity="daemon",
            dependencies=DaemonReadDependencies(
                status_now_ms=1_700_000_000_000,
                runtime_status={
                    "ok": True,
                    "daemon_liveness": True,
                    "total_sessions": 99,
                    "raw_parse_failures": 88,
                    "raw_materialization_readiness": {"available": False},
                    "checked_at": "2023-11-14T22:13:19Z",
                    "browser_capture_active": True,
                },
            ),
        )

    assert result["daemon_liveness"] is True
    assert result["browser_capture_active"] is True
    assert result["total_sessions"] == 0
    assert result["archive_stats"]["total_sessions"] == 0
    assert result["raw_parse_failures"] == 0
    assert result["raw_materialization_readiness"]["available"] is True
    observations = result["status_observations"]
    assert "raw_parse_failures" in observations["archive"]["fields"]
    assert "browser_capture_active" in observations["runtime"]["fields"]
    assert observations["runtime"]["checked_at"] == "2023-11-14T22:13:19Z"


def test_completion_is_the_existing_public_completion_envelope() -> None:
    result = execute_read_operation(
        "completion",
        {"kind": "field", "incomplete": "orig"},
        archive=_Archive(),  # type: ignore[arg-type]
        serving_identity="daemon",
    )

    completion = result["query_completions"]
    assert completion["kind"] == "field"
    assert completion["incomplete"] == "orig"
    assert completion["candidates"][0]["value"] == "origin"


def test_cli_query_lowering_is_independent_of_the_cli_package() -> None:
    from polylogue.operations.daemon_reads import _lower_cli_query_params

    params, expression = _lower_cli_query_params({"query": ("repo:polylogue since:7d",), "lexical": True})

    assert params == {"retrieval_lane": "dialogue"}
    assert expression == "repo:polylogue since:7d"


def test_vector_snapshot_requirement_uses_the_canonical_cli_lowering() -> None:
    assert not requires_vector_snapshot("cli.query", {"params": {"query": ("ordinary words",)}})
    assert requires_vector_snapshot("cli.query", {"params": {"query": ("semantic words",), "semantic": True}})
    assert requires_vector_snapshot("cli.query", {"params": {"query": ("hello",), "retrieval_lane": "hybrid"}})
    assert not requires_vector_snapshot("facets", {"params": {"query": "near:hello"}})


def test_vector_binding_uses_only_explicit_resolved_config_values() -> None:
    """Mutation: fall back to ambient configuration and this misses the configured model."""

    config = SimpleNamespace(
        index_config=SimpleNamespace(voyage_api_key="test-voyage-key"),
        embedding_model="voyage-3-lite",
        embedding_dimension=512,
    )

    binding = vector_binding_from_config(config)  # type: ignore[arg-type]

    assert binding is not None
    assert (binding.voyage_key, binding.model, binding.dimension) == ("test-voyage-key", "voyage-3-lite", 512)
    assert (
        vector_binding_from_config(
            SimpleNamespace(index_config=None, embedding_model="voyage-4-lite", embedding_dimension=1024)  # type: ignore[arg-type]
        )
        is None
    )


def test_hybrid_query_names_an_absent_vector_provider_as_a_degraded_lane(tmp_path: Path) -> None:
    """Mutation: omit the synthesized unavailable failure and this reads empty instead of degraded."""

    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        result = execute_read_operation(
            "cli.query",
            {"params": {"query": ("needle",), "retrieval_lane": "hybrid"}},
            archive=pinned.archive,
            serving_identity="daemon",
        )

    assert result["outcome"]["state"] == "degraded"
    assert result["requested_lanes"] == ["text", "vector"]
    assert result["executed_lanes"] == ["text"]
    assert result["unavailable_lanes"] == ["vector"]
    assert result["failed_lanes"] == []


def test_search_projection_hydrates_storage_rows_and_describes_real_lanes() -> None:
    """Mutation: duplicate surface projection with rank-as-score or phantom lanes."""
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.query.search_hits import project_search_hits
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary

    summary = ArchiveSessionSummary(
        session_id="codex-session:fixture",
        native_id="fixture",
        origin="codex-session",
        title="Fixture",
        created_at=None,
        updated_at=None,
        message_count=1,
        word_count=2,
        tags=(),
    )
    native = ArchiveSessionSearchHit(
        rank=1,
        session_id=summary.session_id,
        block_id="block",
        message_id="message",
        origin=summary.origin,
        title=summary.title,
        snippet="needle",
        lane_ranks={"text": 2, "vector": 3},
    )
    hits = project_search_hits(
        SessionQueryPlan(query_terms=("needle",), retrieval_lane="hybrid"), [(native, summary)], "hybrid"
    )

    assert hits[0].session_id == summary.session_id
    assert hits[0].matched_terms == ("needle",)
    assert hits[0].score_components == {"text_rank": 2.0, "vector_rank": 3.0}
    assert hits[0].raw_score is None
    assert hits.execution.requested_lanes == ("text", "vector")
    assert hits.execution.executed_lanes == ("text", "vector")
