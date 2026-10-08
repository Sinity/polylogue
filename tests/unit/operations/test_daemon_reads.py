"""Pinned-reader conformance for declared daemon read operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest

from polylogue.config import Config
from polylogue.operations.daemon_reads import (
    DaemonReadDependencies,
    _cacheable_read,
    execute_read_operation,
    requires_vector_snapshot,
    vector_binding_from_config,
)
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session


def test_sampled_and_moving_date_queries_are_not_cached() -> None:
    """Random and relative bounds change results while the index stays fixed.

    Anti-vacuity: treating sample or a natural-language cutoff as an ordinary
    stable query makes this predicate cacheable under the unchanged payload.
    """
    assert not _cacheable_read("cli.query", {"sample": 5})
    assert not _cacheable_read("cli.query", {"sort": "random"})
    assert not _cacheable_read("cli.query", {"since": "1 hour ago"})
    assert not _cacheable_read("cli.query", {"until": "yesterday"})
    assert _cacheable_read("cli.query", {"since": "2026-09-01"})


def test_session_read_preserves_the_declared_2000_row_window() -> None:
    """Generic session reads accept the full public request bound.

    Anti-vacuity: routing through a narrower internal ``Bound`` rejects a
    request accepted by ``SessionReadRequest`` before transcript paging.
    """
    from polylogue.operations.session_contracts import SessionRead

    assert SessionRead.model_validate({"ref": "session:sample", "limit": 1500}).limit == 1500


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
    assert not requires_vector_snapshot(
        "cli.query", {"params": {"query": ("hello",), "retrieval_lane": "hybrid"}}, acquisition_enabled=False
    )
    assert requires_vector_snapshot("cli.query", {"params": {"query": ("near:id:seed",)}}, acquisition_enabled=False)


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
    retained_binding = vector_binding_from_config(
        cast(Config, _VectorConfig(index_config=None, embedding_model="voyage-4-lite", embedding_dimension=1024))
    )
    assert retained_binding is not None
    assert retained_binding.voyage_key is None
    assert retained_binding.model == "voyage-4-lite"


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
    assert result["requested_lanes"] == ["text", "action", "vector"]
    assert result["executed_lanes"] == ["text", "action"]
    assert result["unavailable_lanes"] == ["vector"]
    assert result["failed_lanes"] == []


def test_search_projection_hydrates_storage_rows_and_describes_real_lanes() -> None:
    """Mutation: duplicate surface projection with rank-as-score or phantom lanes."""
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.query.search_contract import ArchiveSearchResult, SearchExecution
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
        SessionQueryPlan(query_terms=("needle",), retrieval_lane="hybrid"),
        ArchiveSearchResult(
            [(native, summary)],
            "hybrid",
            SearchExecution(("text", "action", "vector"), ("text", "action", "vector")),
        ),
    )

    assert hits[0].session_id == summary.session_id
    assert hits[0].matched_terms == ("needle",)
    # Hybrid hits carry each lane's recorded rank and its RRF contribution;
    # the fused score is the sum of those contributions, not a rank.
    components = hits[0].score_components
    assert {key: components[key] for key in ("text_rank", "vector_rank")} == {"text_rank": 2.0, "vector_rank": 3.0}
    assert set(components) == {"text_rank", "vector_rank", "text_rrf", "vector_rrf"}
    assert hits[0].raw_score == hits[0].score
    assert hits[0].score is not None
    assert abs(hits[0].score - (components["text_rrf"] + components["vector_rrf"])) < 1e-9
    assert hits.execution.requested_lanes == ("text", "action", "vector")
    assert hits.execution.executed_lanes == ("text", "action", "vector")


def test_single_lane_search_hit_keeps_its_native_rank() -> None:
    """Anti-vacuity: deriving lane_rank only from lane_ranks leaves every dialogue hit with lane_rank=None."""
    from polylogue.archive.query.plan import SessionQueryPlan
    from polylogue.archive.query.search_contract import ArchiveSearchResult, SearchExecution
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
        rank=4,
        session_id=summary.session_id,
        block_id="block",
        message_id="message",
        origin=summary.origin,
        title=summary.title,
        snippet="needle",
    )
    hits = project_search_hits(
        SessionQueryPlan(query_terms=("needle",)),
        ArchiveSearchResult([(native, summary)], "dialogue", SearchExecution(("text",), ("text",))),
    )

    assert hits[0].lane_rank == 4


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
        for source in ("session_id", "tag", "repo", "cwd_prefix", "tool"):
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

    def test_facets_route_counts_a_scope_larger_than_one_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The facets scope is paged, never truncated.

        Anti-vacuity: read only the first family chunk, or overwrite a
        family's counts per chunk instead of summing them, and a one-session
        chunk reports one session's families for a two-session archive.
        """
        self._seed_sessions(tmp_path, 2)
        params = {"query": "origin:claude-code-session"}
        with ArchiveStore.open_existing(tmp_path) as archive:
            uncapped = execute_read_operation("facets", {"params": params}, archive=archive, serving_identity="test")
            assert cast(dict[str, Any], uncapped["outcome"])["state"] == "ok"
            assert uncapped["total_sessions"] == 2

            monkeypatch.setattr("polylogue.api.archive._FACET_FAMILY_CHUNK", 1)
            paged = execute_read_operation(
                "facets", {"params": {**params, "no_idf": True}}, archive=archive, serving_identity="test"
            )

        assert cast(dict[str, Any], paged["outcome"])["state"] == "ok"
        assert paged["total_sessions"] == 2

        # The scoped SQL families, aggregated one session per chunk, sum to
        # the single-chunk aggregation.
        from polylogue.api import archive as archive_api
        from polylogue.archive.query.spec import SessionQuerySpec

        with ArchiveStore.open_existing(tmp_path) as archive:
            chunked = archive_api._archive_facet_buckets(archive, SessionQuerySpec())
            monkeypatch.setattr("polylogue.api.archive._FACET_FAMILY_CHUNK", 900)
            whole = archive_api._archive_facet_buckets(archive, SessionQuerySpec())
        assert chunked == whole
        assert sum(whole.role_counts.values()) > 0 and whole.total_sessions == 2

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

    def _msg(pid: str, role: Role, text: str, position: int) -> ParsedMessage:
        return ParsedMessage(provider_message_id=pid, role=role, text=text, position=position)

    # Fixture writers prepare on the measured Index connection they will
    # publish through; a bare sqlite3 handle is not that creator.
    with fixture_index_connection(root / "index.db") as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        write_fixture_index_session(
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
        child_id = write_fixture_index_session(
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
        filtered_page = execute_read_operation(
            "session.read",
            {
                "ref": f"session:{child_id}",
                "kind": "messages",
                "limit": 2,
                "projection": {"exclude_block_kinds": ["thinking"]},
            },
            archive=archive,
            serving_identity="test",
        )
        empty_page = execute_read_operation(
            "session.read",
            {"ref": f"session:{child_id}", "kind": "messages", "limit": 2, "offset": 4},
            archive=archive,
            serving_identity="test",
        )

    assert stored == 2, "fixture must be a real prefix-sharing child storing only its tail"
    assert first["total"] == 4
    assert first["next_offset"] == 2, "pagination must reach the divergent tail"
    assert messages_kind["total"] == first["total"], "two vocabularies, one window"
    assert cast(dict[str, Any], empty_page["outcome"])["state"] == "empty"
    continuation = cast(str, filtered_page["continuation"])
    from polylogue.archive.query.transaction import QueryContinuationInvalidError

    with ArchiveStore.open_existing(tmp_path) as archive:
        with pytest.raises(QueryContinuationInvalidError):
            execute_read_operation(
                "session.read",
                {"ref": f"session:{child_id}", "kind": "messages", "continuation": continuation},
                archive=archive,
                serving_identity="test",
            )


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


def test_a_cursor_page_emits_the_builder_page_its_outcome_describes(tmp_path: Path) -> None:
    """The daemon route emits the hits the envelope builder decided on.

    A cursor page fetches twice the display limit so the builder can trim
    stragglers at or before the anchor and truncate to the limit.

    Anti-vacuity: replace ``envelope["hits"]`` with the raw fetch again and
    this page carries two hits under a limit of one, rows the outcome and
    ``authority.matched`` never counted.
    """
    from tests.infra.storage_records import SessionBuilder

    for name in ("alpha", "beta", "gamma", "delta"):
        SessionBuilder(tmp_path / "index.db", name).provider("claude-code").title(name).add_message(
            "m-0000", role="user", text=f"needle body {name}"
        ).save()

    with ArchiveStore.open_existing(tmp_path) as archive:
        first = execute_read_operation(
            "cli.query",
            {"params": {"query": "needle", "limit": 1}},
            archive=archive,
            serving_identity="daemon",
        )
        cursor = first["next_cursor"]
        assert isinstance(cursor, str)
        second = execute_read_operation(
            "cli.query",
            {"params": {"query": "needle", "limit": 1, "cursor": cursor}},
            archive=archive,
            serving_identity="daemon",
        )

    hits = cast(list[dict[str, Any]], second["hits"])
    assert len(hits) == 1
    assert cast(_Outcome, second["outcome"])["state"] != "empty"
    # ``matched`` names the query's full match count (#5727), not the page.
    assert cast(dict[str, Any], second["authority"])["matched"] >= len(hits)
    first_hits = cast(list[dict[str, Any]], first["hits"])
    assert hits[0]["session"]["id"] != first_hits[0]["session"]["id"]


def test_search_authority_counts_the_match_total_and_the_returned_window(tmp_path: Path) -> None:
    """The operation route reports ``matched``/``analyzed`` as the API builder does.

    Anti-vacuity: swap the two counters back and a one-hit page over four
    matching sessions reports ``matched=1, analyzed=4``.
    """
    from tests.infra.storage_records import SessionBuilder

    for name in ("alpha", "beta", "gamma", "delta"):
        SessionBuilder(tmp_path / "index.db", name).provider("claude-code").title(name).add_message(
            "m-0000", role="user", text=f"needle body {name}"
        ).save()

    with ArchiveStore.open_existing(tmp_path) as archive:
        page = execute_read_operation(
            "cli.query",
            {"params": {"query": "needle", "limit": 1}},
            archive=archive,
            serving_identity="daemon",
        )

    authority = cast(dict[str, Any], page["authority"])
    assert page["total"] == 4
    assert (authority["matched"], authority["analyzed"]) == (4, len(cast(list[object], page["hits"]))) == (4, 1)


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


@pytest.mark.parametrize("lane", ["semantic", "hybrid"])
def test_keyless_text_search_with_retained_binding_remains_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: str
) -> None:
    """Retained-vector binding must not turn unavailable acquisition into failure."""
    from unittest.mock import MagicMock

    from polylogue.core.errors import EmbeddingRetrievalNotReadyError
    from polylogue.operations.daemon_reads import VectorReadBinding
    from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
    from polylogue.storage.search_providers.sqlite_vec_runtime import open_vector_read_snapshot
    from tests.infra.vector_archive import seed_vector_archive

    bootstrap_archive_root(tmp_path)
    seed_vector_archive(
        tmp_path,
        [("seed", "m1", "Synthetic needle prose with retained embeddings.", [1.0] + [0.0] * 1023)],
        model="voyage-4-lite",
    )
    binding = VectorReadBinding(voyage_key=None, model="voyage-4-lite", dimension=1024)
    acquisition = MagicMock(side_effect=AssertionError("keyless text must not call acquisition"))
    monkeypatch.setattr(SqliteVecProvider, "_get_embeddings", acquisition)
    connection = open_vector_read_snapshot(
        embeddings_path=tmp_path / "embeddings.db", index_path=tmp_path / "index.db", recipe=binding.recipe
    )
    try:
        assert connection.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        with open_operation_read(tmp_path) as pinned:

            def execute() -> object:
                return execute_read_operation(
                    "cli.query",
                    {"params": {"query": ("needle",), "retrieval_lane": lane}},
                    archive=pinned.archive,
                    serving_identity="daemon",
                    dependencies=DaemonReadDependencies(vector_binding=binding, vector_connection=connection),
                )

            if lane == "semantic":
                with pytest.raises(EmbeddingRetrievalNotReadyError) as refusal:
                    execute()
                assert refusal.value.readiness_status == "disabled"
            else:
                result = cast(_SearchResult, execute())
                assert result["outcome"]["state"] == "degraded"
                assert result["unavailable_lanes"] == ["vector"]
                assert result["failed_lanes"] == []
    finally:
        connection.close()
    acquisition.assert_not_called()


@pytest.mark.parametrize("name", ["cli.query", "read.temporal", "read.chronicle", "read.compact"])
@pytest.mark.parametrize("session_ref", ["codex-session:selected", "", None])
def test_only_selected_temporal_reads_skip_vector_admission(name: str, session_ref: str | None) -> None:
    payload = {"session_id": session_ref, "params": {"similar_text": "synthetic query"}}
    assert requires_vector_snapshot(name, payload) is not (name == "read.temporal" and bool(session_ref))
