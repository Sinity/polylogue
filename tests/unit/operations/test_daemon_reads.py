"""Pinned-reader conformance for declared daemon read operations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest

from polylogue.config import Config
from polylogue.operations.daemon_reads import (
    DaemonReadDependencies,
    execute_read_operation,
    requires_vector_snapshot,
    vector_binding_from_config,
)
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


@dataclass
class _Stats:
    def to_dict(self) -> dict[str, object]:
        return {"total_sessions": 3, "total_messages": 8, "retrieval_ready": False}


class _Archive:
    def stats(self) -> _Stats:
        return _Stats()


@dataclass(frozen=True)
class _IndexConfig:
    voyage_api_key: str | None


@dataclass(frozen=True)
class _VectorConfig:
    index_config: _IndexConfig | None
    embedding_model: str
    embedding_dimension: int


class _ArchiveStatsPayload(TypedDict):
    total_sessions: int


class _RawMaterializationPayload(TypedDict):
    available: bool


class _ObservationFields(TypedDict):
    fields: list[str]


class _RuntimeObservation(TypedDict):
    checked_at: str
    fields: list[str]


class _StatusObservations(TypedDict):
    archive: _ObservationFields
    runtime: _RuntimeObservation


class _StatusResult(TypedDict):
    daemon_liveness: bool
    browser_capture_active: bool
    total_sessions: int
    archive_stats: _ArchiveStatsPayload
    raw_parse_failures: int
    raw_materialization_readiness: _RawMaterializationPayload
    status_observations: _StatusObservations


class _CompletionCandidate(TypedDict):
    value: str


class _CompletionPayload(TypedDict):
    kind: str
    incomplete: str
    candidates: list[_CompletionCandidate]


class _CompletionResult(TypedDict):
    query_completions: _CompletionPayload


class _Outcome(TypedDict):
    state: str


class _SearchResult(TypedDict):
    outcome: _Outcome
    requested_lanes: list[str]
    executed_lanes: list[str]
    unavailable_lanes: list[str]
    failed_lanes: list[str]


def test_status_preserves_runtime_contract_without_using_cached_archive_evidence(tmp_path: Path) -> None:
    """Mutation: merge the cached snapshot last and stale archive facts win."""
    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        result = cast(
            _StatusResult,
            execute_read_operation(
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
    result = cast(
        _CompletionResult,
        execute_read_operation(
            "completion",
            {"kind": "field", "incomplete": "orig"},
            archive=cast(ArchiveStore, _Archive()),
            serving_identity="daemon",
        ),
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

    config = _VectorConfig(
        index_config=_IndexConfig(voyage_api_key="test-voyage-key"),
        embedding_model="voyage-3-lite",
        embedding_dimension=512,
    )

    binding = vector_binding_from_config(cast(Config, config))

    assert binding is not None
    assert (binding.voyage_key, binding.model, binding.dimension) == ("test-voyage-key", "voyage-3-lite", 512)
    assert (
        vector_binding_from_config(
            cast(
                Config,
                _VectorConfig(index_config=None, embedding_model="voyage-4-lite", embedding_dimension=1024),
            )
        )
        is None
    )


def test_hybrid_query_names_an_absent_vector_provider_as_a_degraded_lane(tmp_path: Path) -> None:
    """Mutation: omit the synthesized unavailable failure and this reads empty instead of degraded."""

    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        result = cast(
            _SearchResult,
            execute_read_operation(
                "cli.query",
                {"params": {"query": ("needle",), "retrieval_lane": "hybrid"}},
                archive=pinned.archive,
                serving_identity="daemon",
            ),
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


def test_archive_backed_completion_answers_from_the_pinned_reader(tmp_path: Path) -> None:
    """A completion naming a source reads the operation's own archive, bounded.

    The CLI's completer used to open ``index.db`` in the shell-completion
    process. The vocabularies it read are now this operation's answer, which is
    what lets a resident daemon serve a TAB press from its already-open
    snapshot.

    Mutation: ignore ``limit`` and the bound assertion goes red; answer ``tag``
    from ``stats_by`` instead of the durable user tier and the tag vocabulary
    empties on an archive with tags but no tag aggregate.
    """

    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        for source in ("session_id", "tag", "repo", "tool"):
            result = execute_read_operation(
                "completion",
                {"source": source, "incomplete": "", "limit": 3},
                archive=pinned.archive,
                serving_identity="daemon",
            )
            values = cast("dict[str, Any]", result["value_completions"])
            assert values["source"] == source
            assert len(values["values"]) <= 3
            assert all(isinstance(row["value"], str) and row["value"] for row in values["values"])


def test_an_undeclared_completion_source_is_refused(tmp_path: Path) -> None:
    """A source the handler does not implement refuses instead of reading empty.

    Mutation: return ``[]`` for an unknown source and a typo in a completer
    silently produces no completions forever.
    """

    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        with pytest.raises(ValueError, match="completion source is not declared"):
            execute_read_operation(
                "completion",
                {"source": "not_a_source", "incomplete": ""},
                archive=pinned.archive,
                serving_identity="daemon",
            )


def test_a_grammar_completion_still_needs_no_archive() -> None:
    """The cold path stays cold: a grammar completion opens nothing.

    Mutation: make the handler read ``archive`` unconditionally and this raises,
    because the minimal archive-shaped object the public operation seam passes
    implements only ``stats``.
    """

    result = execute_read_operation(
        "completion",
        {"kind": "field", "incomplete": "orig"},
        archive=cast(ArchiveStore, _Archive()),
        serving_identity="daemon",
    )
    assert "value_completions" not in result
    assert cast("dict[str, Any]", result["query_completions"])["kind"] == "field"


class TestBoundedReadsDegradeByName:
    """A declared bound that shapes a read must reach the envelope as a gap.

    Both cases below produced a complete-looking ``ok`` envelope over a
    truncated population, which the terminal-outcome contract exists to make
    impossible: ``degraded`` means named gaps shaped the answer.
    """

    @staticmethod
    def _seed_sessions(root: Path, count: int, messages: int = 1) -> list[str]:
        from tests.infra.storage_records import SessionBuilder

        session_ids: list[str] = []
        for index in range(count):
            name = f"bounded-{index:03d}"
            builder = SessionBuilder(root / "index.db", name).provider("claude-code").title(name)
            for message in range(messages):
                builder = builder.add_message(
                    f"m-{message:04d}",
                    role="user" if message % 2 == 0 else "assistant",
                    text=f"body {message}",
                )
            builder.save()
            session_ids.append(f"claude-code-session:ext-{name}")
        return session_ids

    def test_facets_route_reports_a_capped_scope(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The daemon facets route names ``facet_scope_truncated`` like the API route.

        The cap is read through ``polylogue.api.archive``; both routes call the
        same ``_archive_facet_buckets``, but this one dropped the ``scope_gaps``
        collector, so only the API route reported it. The daemon route is the
        one the CLI reads through, so the unreported answer was the one
        operators saw.

        Anti-vacuity: stop passing ``scope_gaps`` to either
        ``_archive_facet_buckets`` call or to ``build_facets_response`` and the
        capped read returns ``outcome.state == "ok"`` with every family still
        in ``complete_families`` -- a one-session denominator for a two-session
        archive, rendered as measured and complete.
        """
        self._seed_sessions(tmp_path, 2)
        params = {"query": "origin:claude-code-session"}
        with ArchiveStore.open_existing(tmp_path) as archive:
            uncapped = execute_read_operation("facets", {"params": params}, archive=archive, serving_identity="test")
            assert cast(dict[str, Any], uncapped["outcome"])["state"] == "ok"
            assert uncapped["total_sessions"] == 2

            monkeypatch.setattr("polylogue.api.archive.FACET_SCOPE_SESSION_CAP", 1)
            capped = execute_read_operation(
                "facets", {"params": {**params, "no_idf": True}}, archive=archive, serving_identity="test"
            )

        outcome = cast(dict[str, Any], capped["outcome"])
        family_errors = cast(dict[str, str], capped["family_errors"])
        assert outcome["state"] == "degraded"
        assert outcome["reason"] == "facet_scope_truncated:1"
        assert capped["complete_families"] == []
        assert family_errors
        assert all(reason == "facet_scope_truncated:1" for reason in family_errors.values())

    def test_query_envelope_degrades_when_the_attached_projection_is_cut(self, tmp_path: Path) -> None:
        """``with messages`` over a 250-message session degrades, it does not lie.

        250 is strictly more than the 200-row per-session ceiling, so the
        projection genuinely cannot carry the session's whole row set.

        Anti-vacuity: decide the outcome before the projection runs again
        (``outcome = decide_outcome(matched=total)`` above the
        ``_attached_units_payload`` call) and this envelope is ``ok`` while
        carrying 200 of 250 rows.
        """
        session_id = self._seed_sessions(tmp_path, 1, messages=250)[0]
        with ArchiveStore.open_existing(tmp_path) as archive:
            result = execute_read_operation(
                "cli.query",
                {"params": {"query": f"id:{session_id}", "with_units": ["message"]}},
                archive=archive,
                serving_identity="test",
            )

        attached = cast(dict[str, Any], result["attached_units"])["message"][session_id]
        outcome = cast(dict[str, Any], result["outcome"])
        assert len(attached) == 200
        assert outcome["state"] == "degraded"
        assert outcome["reason"] == "attached_unit_truncated:message:200"


def _seed_lineage_child(root: Path) -> str:
    from polylogue.archive.message.roles import Role
    from polylogue.archive.session.branch_type import BranchType
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

    def _msg(pid: str, role: Role, text: str, position: int) -> ParsedMessage:
        return ParsedMessage(provider_message_id=pid, role=role, text=text, position=position)

    bootstrap_archive_root(root)
    conn = sqlite3.connect(root / "index.db")
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="parent",
                title="parent",
                messages=[
                    _msg("p0", Role.USER, "hello", 0),
                    _msg("p1", Role.ASSISTANT, "hi there", 1),
                ],
            ),
        )
        child_id = write_parsed_session_to_archive(
            conn,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="child",
                title="child",
                parent_session_provider_id="parent",
                branch_type=BranchType.FORK,
                messages=[
                    _msg("c0", Role.USER, "hello", 0),
                    _msg("c1", Role.ASSISTANT, "hi there", 1),
                    _msg("cx", Role.USER, "child diverges", 2),
                    _msg("cy", Role.ASSISTANT, "child replies", 3),
                ],
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return str(child_id)


def test_transcript_total_is_the_composed_length(tmp_path: Path) -> None:
    """Anti-vacuity: read ``total`` from ``summary.message_count`` again and
    this asserts 2 for a child whose composed transcript is 4 -- and the
    ``next_offset`` assertion below goes None, i.e. the reader never reaches
    the divergent tail.  A plain (non-lineage) session cannot see this: its
    stored count and its composed length are the same number.
    """
    child_id = _seed_lineage_child(tmp_path)
    with ArchiveStore.open_existing(tmp_path) as archive:
        stored = archive.read_summary(child_id).message_count
        first = execute_read_operation(
            "session.read",
            {"ref": f"session:{child_id}", "kind": "transcript", "limit": 2, "offset": 0},
            archive=archive,
            serving_identity="test",
        )
        messages_kind = execute_read_operation(
            "session.read",
            {"ref": f"session:{child_id}", "kind": "messages", "limit": 2, "offset": 0},
            archive=archive,
            serving_identity="test",
        )

    assert stored == 2, "fixture must be a real prefix-sharing child storing only its tail"
    assert first["total"] == 4
    assert first["next_offset"] == 2, "pagination must reach the divergent tail"
    assert messages_kind["total"] == first["total"], "two vocabularies, one window"


def test_search_continuation_survives_a_session_grain_total(tmp_path: Path) -> None:
    """A ranked page full of one session's hits still offers the next page.

    `search_summaries` selects FTS *block* rows with no DISTINCT over
    `session_id`, so several matching blocks in one session are several hits.
    `total` is deliberately session-grain -- it is what `total_unit` labels --
    so comparing `offset + len(hits)` against it is a unit error: the first
    page fills with one session's blocks while another session's hit waits at
    the next offset, and a small session total makes that page look final.

    Anti-vacuity: passing `total=total` to `page_next_offset` again makes
    `next_offset` `None` here, ending the walk before the second session's
    hit is ever returned.
    """
    from tests.infra.storage_records import SessionBuilder

    crowded = SessionBuilder(tmp_path / "index.db", "crowded").provider("claude-code").title("crowded")
    for index in range(4):
        crowded = crowded.add_message(f"m-{index:04d}", role="user", text=f"needle body {index}")
    crowded.save()
    SessionBuilder(tmp_path / "index.db", "later").provider("claude-code").title("later").add_message(
        "m-0000", role="user", text="needle body tail"
    ).save()

    with ArchiveStore.open_existing(tmp_path) as archive:
        page = execute_read_operation(
            "cli.query",
            {"params": {"query": "needle", "limit": 4, "offset": 0}},
            archive=archive,
            serving_identity="test",
        )
        hits = cast(list[dict[str, Any]], page["hits"])
        assert len(hits) == 4
        # Four block-grain hits, two sessions: the session total is smaller
        # than the hits already returned.
        assert cast(int, page["total"]) < len(hits)
        assert page["next_offset"] == 4

        second = execute_read_operation(
            "cli.query",
            {"params": {"query": "needle", "limit": 4, "offset": 4}},
            archive=archive,
            serving_identity="test",
        )
    assert cast(list[dict[str, Any]], second["hits"]), "the second page must still carry the remaining hit"


def test_a_short_ranked_page_still_terminates(tmp_path: Path) -> None:
    """The opposite direction: a page under its own bound ends the walk."""
    from tests.infra.storage_records import SessionBuilder

    SessionBuilder(tmp_path / "index.db", "only").provider("claude-code").title("only").add_message(
        "m-0000", role="user", text="needle body"
    ).save()

    with ArchiveStore.open_existing(tmp_path) as archive:
        page = execute_read_operation(
            "cli.query",
            {"params": {"query": "needle", "limit": 50, "offset": 0}},
            archive=archive,
            serving_identity="test",
        )
    assert page["next_offset"] is None


def test_declared_query_units_replays_the_http_opaque_continuation(tmp_path: Path) -> None:
    """Removing the continuation branch makes the declared operation reject page two."""
    from polylogue.archive.query.transaction import QueryContinuationInvalidError
    from tests.infra.storage_records import SessionBuilder

    SessionBuilder(tmp_path / "index.db", "page-owner").provider("claude-code").title("page owner").add_message(
        "m-0000", role="user", text="first page"
    ).add_message("m-0001", role="user", text="second page").save()

    with ArchiveStore.open_existing(tmp_path) as archive:
        first = execute_read_operation(
            "query.units",
            {"params": {"expression": "messages where words >= 0 | sort by time asc", "limit": 1}},
            archive=archive,
            serving_identity="daemon",
        )
        continuation = first["continuation"]
        assert isinstance(continuation, str) and continuation.startswith("q2.")
        second = execute_read_operation(
            "query.units",
            {"params": {"continuation": continuation}},
            archive=archive,
            serving_identity="daemon",
        )
        with pytest.raises(QueryContinuationInvalidError):
            execute_read_operation(
                "query.units",
                {"params": {"continuation": continuation, "limit": 10}},
                archive=archive,
                serving_identity="daemon",
            )

    assert first["query_ref"] == second["query_ref"]
    assert first["result_ref"] == second["result_ref"]
    assert second["offset"] == 1
    first_items = cast("list[dict[str, object]]", first["items"])
    second_items = cast("list[dict[str, object]]", second["items"])
    assert first_items[0]["message_id"] != second_items[0]["message_id"]
