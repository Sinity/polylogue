"""Direct contract tests for ``polylogue.Polylogue`` async facade methods (#1298).

The ``Polylogue`` class is the documented Python-API entry point used by
downstream consumers such as lynchpin. Most usage flows through CLI/MCP
adapters, leaving the typed facade itself sparsely covered. These tests
pin the public contract directly:

1. **Surface discovery** — every public async method on ``Polylogue`` is
   enumerated dynamically (not hard-coded). New methods automatically
   fail the signature gate until they are added to ``KNOWN_METHODS``,
   which forces the test author to opt-in deliberately.
2. **Signature contract** — each method has a real coroutine function,
   a typed return annotation, and accepts the documented positional /
   keyword-only arguments.
3. **Return-type contract** — happy-path invocations on a seeded
   archive return values that structurally match the declared model
   (Python types, container shapes, attribute presence).
4. **Empty-archive contract** — read methods return empty containers
   (``[]``, ``{}``, ``None``) rather than raising on an empty database.
5. **Error envelope** — methods that resolve session IDs raise
   :class:`SessionNotFoundError` (a typed
   :class:`PolylogueError` subclass) on unknown IDs.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue import Polylogue
from polylogue.api.archive import SessionNotFoundError
from polylogue.archive.message.roles import Role
from polylogue.core.enums import AssertionKind, AssertionStatus, BlockType, BranchType, MaterialOrigin, Origin, Provider
from polylogue.core.refs import delegation_edge_object_id
from polylogue.errors import DatabaseError, PolylogueError
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from tests.infra.storage_records import db_setup

# ---------------------------------------------------------------------------
# Surface enumeration
# ---------------------------------------------------------------------------
#
# ``KNOWN_METHODS`` is the curated list of public async methods on the
# facade that this contract suite knows about. The
# :func:`test_no_undiscovered_async_methods` test fails when the live
# ``Polylogue`` class grows a new public async method that is not in
# this list — that forces the author to either add it here (and pick the
# appropriate categorization below) or to mark it explicitly as
# out-of-scope.
#
# The categorization below drives parametrized contracts. Read methods
# are exercised on a seeded archive; mutation methods are exercised
# against an unknown session ID to assert the error envelope.

# Methods that close transport-level resources, not domain operations.
LIFECYCLE_METHODS: frozenset[str] = frozenset({"close"})

# Read-only methods that take a session_id; resolving an unknown ID
# returns ``None`` (no error). They are part of the read contract.
READ_BY_ID_NONE_METHODS: frozenset[str] = frozenset(
    {
        "get_session",
        "get_session_summary",
        "get_session_profile_insight",
        "get_session_profile_record",
        "get_session_latency_profile_insight",
        "resume_brief",
    }
)

# Read-only methods that take a session_id and return an empty
# collection / zero stats / dict for missing IDs rather than raising.
READ_BY_ID_EMPTY_METHODS: frozenset[str] = frozenset(
    {
        "get_messages_paginated",  # special: raises SessionNotFoundError
        "get_session_stats",
        "get_session_work_event_insights",
        "get_session_phase_insights",
        "get_session_tree",
        "get_raw_artifacts_for_session",
        "bulk_get_messages",
        "get_actions",
    }
)

# Read-only methods with no required arguments — should always succeed
# on an empty archive and return an empty container or zero counts.
READ_NULLARY_METHODS: frozenset[str] = frozenset(
    {
        "list_sessions",
        "list_summaries",
        "stats",
        "facets",
        "health_check",
        "list_tags",
        "list_marks",
        "list_annotations",
        "list_views",
        "list_read_view_profiles",
        "list_assertion_claims",
        "list_assertion_candidates",
        "list_recall_packs",
        "list_workspaces",
        "list_corrections",
        "get_session_insight_status",
        "rebuild_insights",
        "insight_readiness_report",
        "insight_rigor_audit",
        "list_session_profile_insights",
        "list_session_latency_profile_insights",
        "find_stuck_session_latency_profile_insights",
        "list_session_tag_rollup_insights",
        "list_session_work_event_insights",
        "list_session_phase_insights",
        "list_thread_insights",
        "list_archive_coverage_insights",
        "list_tool_usage_insights",
        "list_session_cost_insights",
        "list_cost_rollup_insights",
        "list_usage_timeline_insights",
        "list_archive_debt_insights",
        "provider_usage_report",
        "count_sessions",
        "rebuild_index",
        "get_index_status",
        "get_stats_by",
        "parse_sources",
    }
)

# Query methods that take a free-text query string and produce a search
# envelope or result set.
SEARCH_METHODS: frozenset[str] = frozenset({"search", "search_envelope", "query_units"})

# Methods that mutate state by session ID. Calling them with an
# unknown ID should raise ``SessionNotFoundError`` (a typed
# ``PolylogueError`` subclass). The bulk variant takes a list and
# returns a typed report rather than raising — see the dedicated test.
MUTATION_BY_ID_RAISES_METHODS: frozenset[str] = frozenset(
    {
        "add_tag",
        "remove_tag",
        "add_mark",
        "remove_mark",
        "record_correction",
        "delete_correction",
        "clear_corrections",
        "save_annotation",
    }
)

# Mutation methods that take only a session_id and return a typed
# envelope describing the outcome (no raise on missing IDs).
MUTATION_BY_ID_TYPED_OUTCOME: frozenset[str] = frozenset(
    {
        "delete_session",
        "delete_session_safe",
    }
)

# Mutation methods that take metadata key/value and resolve the ID.
METADATA_MUTATION_METHODS: frozenset[str] = frozenset(
    {
        "set_metadata",
        "update_metadata",
        "delete_metadata",
        "get_metadata",
    }
)

# Methods covered by dedicated bespoke tests below (not in the
# parametrized contract families because they take complex args).
BESPOKE_METHODS: frozenset[str] = frozenset(
    {
        "explain_query_expression",
        "query_completions",
        "get_sessions",
        "get_actions_batch",
        "query_sessions",
        "list_sessions_for_spec",
        "search_session_hits",
        "diagnose_query_miss",
        "storage_stats",
        "bulk_tag_sessions",
        "list_session_profile_insights",
        "get_thread_insight",
        "get_annotation",
        "save_view",
        "get_view",
        "get_view_by_name",
        "delete_view",
        "save_workspace",
        "get_workspace",
        "delete_workspace",
        "create_recall_pack",
        "get_recall_pack",
        "delete_recall_pack",
        "delete_annotation",
        "list_annotations",
        "neighbor_candidates",
        "context_image_payload",
        "context_preamble_payload",
        "compile_context",
        "find_resume_candidates",
        "export_insight_bundle",
        # Distilled postmortem bundle (#2380).
        "postmortem_bundle",
        "parse_file",
        "update_index",
        "cost_outlook",
        "archive_count_sessions",
        "archive_get_session",
        "archive_list_sessions",
        "archive_search_sessions",
        # Topology read methods exposed in tests/unit/api/test_topology_api.py.
        "get_ancestors",
        "get_descendants",
        "get_logical_session",
        "get_session_topology",
        "get_siblings",
        "get_thread",
        # Blackboard note methods exposed in tests/unit/mcp/test_mcp_edge_cases.py.
        "list_blackboard_notes",
        "post_blackboard_note",
        # Shared web/MCP payload DTO helpers covered by daemon/MCP surface tests.
        "explain_import",
        "archive_debt",
        "list_assertion_claim_payloads",
        "list_assertion_candidate_reviews",
        "judge_assertion_candidate",
        "neighbor_candidate_payloads",
        "resolve_ref",
        "export_otel",
        "session_correlation_payload",
        # Pathology and portfolio read methods added in recent PRs.
        "pathology_report",
        "materialize_pathology_assertions",
        "portfolio_bundle",
        # Session analysis primitives sunk from the MCP-only surface into the
        # shared facade (#1691 / polylogue-9e5.24) -- covered by
        # tests/unit/insights/test_archive_rollups.py,
        # tests/unit/insights/test_session_analytics.py, and
        # tests/unit/api/test_session_analytics_facade.py.
        "aggregate_sessions",
        "workflow_shape_distribution",
        "find_abandoned_sessions",
        "tool_call_latency_distribution",
        "compare_sessions",
        "find_similar_sessions_by_metadata",
        "correlate_sessions",
    }
)

# All categorized methods. The union must equal the discovered set;
# this catches drift in the public surface.
KNOWN_METHODS: frozenset[str] = (
    LIFECYCLE_METHODS
    | READ_BY_ID_NONE_METHODS
    | READ_BY_ID_EMPTY_METHODS
    | READ_NULLARY_METHODS
    | SEARCH_METHODS
    | MUTATION_BY_ID_RAISES_METHODS
    | MUTATION_BY_ID_TYPED_OUTCOME
    | METADATA_MUTATION_METHODS
    | BESPOKE_METHODS
)


def _discovered_public_async_methods() -> set[str]:
    """Discover every public async method on :class:`Polylogue` (no hardcoding)."""
    methods: set[str] = set()
    for name in dir(Polylogue):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(Polylogue, name)
        if callable(attr) and inspect.iscoroutinefunction(attr):
            methods.add(name)
    return methods


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _archive(tmp_path: Path) -> Polylogue:
    """Construct a Polylogue facade against an isolated tmp archive."""
    with ArchiveStore(tmp_path):
        pass
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")


def _materialize_run_projection(index_db: Path) -> None:
    """Run the session-insight materializer for richer digest-derived projections."""
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(index_db) as conn:
        rebuild_session_insights_sync(conn)


_HASH = b"x" * 32


def _seed_import_explain_archive(tmp_path: Path, *, source_path: str | None = None) -> tuple[str, str]:
    source_path = source_path or str(Path.home() / ".codex" / "sessions" / "session.jsonl")
    raw_id = "raw-import-1"
    source_conn = sqlite3.connect(tmp_path / "source.db")
    index_conn = sqlite3.connect(tmp_path / "index.db")
    try:
        initialize_archive_tier(source_conn, ArchiveTier.SOURCE)
        initialize_archive_tier(index_conn, ArchiveTier.INDEX)
        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms, validated_at_ms, validation_status,
                detection_warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                "codex-session",
                "native-1",
                source_path,
                0,
                _HASH,
                42,
                1_700_000_000_000,
                1_700_000_000_100,
                1_700_000_000_200,
                "passed",
                '["normalized path separators"]',
            ),
        )
        source_conn.execute(
            """
            INSERT INTO raw_artifacts (
                artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                support_status, classification_reason, parse_as_session, schema_eligible,
                first_observed_at_ms, last_observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "artifact-1",
                raw_id,
                "codex-session",
                source_path,
                0,
                "session_record_stream",
                "supported_parseable",
                "codex session stream",
                1,
                1,
                1_700_000_000_000,
                1_700_000_000_000,
            ),
        )
        index_conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, title, content_hash, message_count, tool_use_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("native-1", "codex-session", raw_id, "Import explain", _HASH, 1, 1),
        )
        index_conn.execute(
            """
            INSERT INTO messages (
                session_id, native_id, position, role, message_type, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("codex-session:native-1", "msg-1", 0, "assistant", "message", _HASH),
        )
        index_conn.execute(
            """
            INSERT INTO blocks (
                message_id, session_id, position, block_type, text, tool_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("codex-session:native-1:msg-1", "codex-session:native-1", 0, "tool_use", "pytest", "tool-1"),
        )
        source_conn.commit()
        index_conn.commit()
    finally:
        source_conn.close()
        index_conn.close()
    return raw_id, source_path


@pytest.mark.asyncio
async def test_archive_tiers_facade_reads_active_db_override_root(tmp_path: Path) -> None:
    """Archive facade reads open the active index root, not just config.archive_root."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    configured_root = tmp_path / "configured"
    active_root = tmp_path / "active"
    configured_root.mkdir()
    active_root.mkdir()
    with ArchiveStore(active_root) as archive_db:
        archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="override-root",
                title="Override root",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="override root needle",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="override root needle")],
                    )
                ],
            )
        )

    archive = Polylogue(archive_root=configured_root, db_path=active_root / "index.db")

    assert await archive.archive_count_sessions() == 1
    hits = await archive.archive_search_sessions("needle")
    assert [hit.session_id for hit in hits] == ["codex-session:override-root"]


async def _seed_two_sessions(db_path: Path) -> None:
    """Seed two minimal sessions for happy-path read assertions."""
    root = db_path.parent
    with ArchiveStore(root) as archive_db:
        archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_AI,
                provider_session_id="conv-alpha",
                title="Alpha",
                messages=[
                    ParsedMessage(
                        provider_message_id="alpha-m1",
                        role=Role.USER,
                        text="alpha body",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="alpha body")],
                    ),
                    ParsedMessage(
                        provider_message_id="alpha-m2",
                        role=Role.ASSISTANT,
                        text="alpha reply",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="alpha reply")],
                    ),
                ],
            )
        )
        archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="conv-beta",
                title="Beta",
                messages=[
                    ParsedMessage(
                        provider_message_id="beta-m1",
                        role=Role.USER,
                        text="beta body",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="beta body")],
                    )
                ],
            )
        )


# ---------------------------------------------------------------------------
# 1. Surface enumeration: discovery must stay current
# ---------------------------------------------------------------------------


def test_no_undiscovered_async_methods() -> None:
    """Every public async ``Polylogue`` method is categorized in this file.

    When a new async method is added to the facade, this test fails so
    the author has to add it to one of the ``*_METHODS`` sets above or
    to ``BESPOKE_METHODS``. This is the mechanism the issue asks for:
    "discovers methods from the Polylogue class (not hardcoded) so new
    methods fail until covered."
    """
    discovered = _discovered_public_async_methods()
    uncovered = discovered - KNOWN_METHODS
    assert not uncovered, (
        f"Polylogue gained {len(uncovered)} async method(s) without contract "
        f"coverage in tests/unit/api/test_facade_contracts.py: {sorted(uncovered)}. "
        "Add each one to the appropriate *_METHODS set."
    )
    stale = KNOWN_METHODS - discovered
    assert not stale, (
        f"tests/unit/api/test_facade_contracts.py categorizes methods that no "
        f"longer exist on Polylogue: {sorted(stale)}."
    )


# ---------------------------------------------------------------------------
# 2. Signature contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", sorted(_discovered_public_async_methods()))
def test_method_has_typed_signature(method_name: str) -> None:
    """Each public async method has a real signature and a return annotation."""
    method = getattr(Polylogue, method_name)
    assert inspect.iscoroutinefunction(method), f"{method_name} must be async"
    sig = inspect.signature(method)
    # Return annotation must be present (not ``inspect.Signature.empty``).
    assert sig.return_annotation is not inspect.Signature.empty, (
        f"{method_name} is missing a return annotation; downstream consumers rely on the typed shape of the facade."
    )
    # Every non-``self`` parameter must be annotated.
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        assert param.annotation is not inspect.Parameter.empty, (
            f"{method_name}({param_name}) is missing a type annotation; "
            "this breaks mypy --strict for downstream consumers."
        )


# ---------------------------------------------------------------------------
# 3. Lifecycle: facade is an async context manager
# ---------------------------------------------------------------------------


async def test_facade_is_async_context_manager(tmp_path: Path) -> None:
    """``Polylogue`` supports the documented ``async with`` lifecycle."""
    async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "x.db") as archive:
        assert isinstance(archive, Polylogue)
        assert archive.archive_root == tmp_path


# ---------------------------------------------------------------------------
# 4. Empty-archive contract for nullary read methods
# ---------------------------------------------------------------------------

# A representative subset that exercises every return-shape family
# (list, scalar, mapping, typed envelope). The full
# ``READ_NULLARY_METHODS`` set is too broad to invoke individually here
# because some methods (``rebuild_insights``, ``rebuild_index``,
# ``parse_sources``) have meaningful side-effects that warrant their
# own tests. Keep this list focused on pure reads.
EMPTY_ARCHIVE_LIST_METHODS: tuple[str, ...] = (
    "list_sessions",
    "list_summaries",
    "list_tags",
    "list_marks",
    "list_views",
    "list_recall_packs",
    "list_workspaces",
    "list_corrections",
    "list_session_profile_insights",
    "list_session_latency_profile_insights",
    "find_stuck_session_latency_profile_insights",
    "list_session_tag_rollup_insights",
    "list_session_work_event_insights",
    "list_session_phase_insights",
    "list_thread_insights",
    "list_archive_coverage_insights",
    "list_tool_usage_insights",
    "list_session_cost_insights",
    "list_cost_rollup_insights",
    "list_archive_debt_insights",
    "list_annotations",
)


# Insight list methods that auto-materialize default-empty rows for the
# zero-session archive (e.g. global archive-debt placeholders). For
# these, the empty-archive contract is "returns a real iterable", not
# "returns an empty iterable".
_AUTO_MATERIALIZED_INSIGHTS: frozenset[str] = frozenset(
    {
        "list_archive_debt_insights",
        "list_tool_usage_insights",
    }
)


@pytest.mark.parametrize("method_name", EMPTY_ARCHIVE_LIST_METHODS)
async def test_list_methods_return_iterable_on_empty_archive(
    tmp_path: Path,
    method_name: str,
) -> None:
    """Every ``list_*`` read method returns a real iterable on an empty archive.

    Most list methods return an empty list for an empty archive. The
    insight methods in :data:`_AUTO_MATERIALIZED_INSIGHTS` synthesize
    default placeholder rows even on an empty archive (archive-debt
    bookkeeping, tool-usage coverage skeletons, etc.) — for them the
    contract is "returns a real iterable of the declared row type",
    not "returns an empty iterable".
    """
    archive = _archive(tmp_path)
    try:
        result = await getattr(archive, method_name)()
        # ``list_tags`` returns a dict[str, int]; everything else here is a list.
        if method_name == "list_tags":
            assert isinstance(result, dict)
            assert result == {}
            return
        assert isinstance(result, Iterable)
        rows = list(result)
        if method_name in _AUTO_MATERIALIZED_INSIGHTS:
            # Contract is iterable-of-typed-rows; rows themselves may be
            # synthesized placeholders. Re-iteration must work because
            # the surface is documented as returning a ``list[...]``.
            assert isinstance(rows, list)
        else:
            assert rows == []
    finally:
        await archive.close()


async def test_count_sessions_returns_int_on_empty_archive(tmp_path: Path) -> None:
    """``count_sessions`` returns ``0`` (a real ``int``) on empty archive."""
    archive = _archive(tmp_path)
    try:
        result = await archive.count_sessions()
        assert isinstance(result, int)
        assert result == 0
    finally:
        await archive.close()


async def test_stats_returns_typed_envelope_on_empty_archive(tmp_path: Path) -> None:
    """``stats()`` returns a typed ``ArchiveStats`` with zero counts."""
    from polylogue.operations import ArchiveStats

    archive = _archive(tmp_path)
    try:
        result = await archive.stats()
        assert isinstance(result, ArchiveStats)
        assert result.session_count == 0
        assert result.message_count == 0
        assert result.word_count == 0
        assert isinstance(result.origins, dict)
        assert isinstance(result.tags, dict)
        assert isinstance(result.recent, list)
    finally:
        await archive.close()


async def test_health_check_returns_readiness_report(tmp_path: Path) -> None:
    """``health_check()`` returns a typed ``ReadinessReport``."""
    from polylogue.readiness import ReadinessReport

    archive = _archive(tmp_path)
    try:
        result = await archive.health_check()
        assert isinstance(result, ReadinessReport)
    finally:
        await archive.close()


async def test_archive_debt_returns_shared_payload_on_empty_archive(tmp_path: Path) -> None:
    """``archive_debt()`` exposes the shared operational debt payload."""
    from polylogue.surfaces.payloads import ArchiveDebtListPayload

    archive = _archive(tmp_path)
    try:
        result = await archive.archive_debt(kinds=("archive-tier",), limit=1)
        assert isinstance(result, ArchiveDebtListPayload)
        assert result.mode == "archive-debt-list"
        assert result.totals.total == 0
        assert result.rows == ()
    finally:
        await archive.close()


async def test_facets_returns_typed_envelope_on_empty_archive(tmp_path: Path) -> None:
    """``facets()`` returns the canonical ``FacetsResponse`` envelope."""
    from polylogue.surfaces.payloads import FacetsResponse

    archive = _archive(tmp_path)
    try:
        result = await archive.facets()
        assert isinstance(result, FacetsResponse)
        assert result.total_sessions == 0
        assert result.total_messages == 0
        assert isinstance(result.origins, dict)
    finally:
        await archive.close()


def test_archive_facet_buckets_count_unique_sessions_for_duplicate_hits() -> None:
    """Facet buckets count sessions, not duplicate per-message search hits."""
    from types import SimpleNamespace

    from polylogue.api.archive import _archive_facet_buckets
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSummary

    summary = ArchiveSessionSummary(
        session_id="claude-ai-export:conv-alpha",
        native_id="conv-alpha",
        origin="claude-ai-export",
        provider=Provider.CLAUDE_AI,
        title="Alpha",
        created_at=None,
        updated_at=None,
        message_count=2,
        word_count=4,
        tags=("work",),
    )
    archive = SimpleNamespace(
        list_summaries=lambda limit: [summary, summary],
        _conn=None,
    )

    result = _archive_facet_buckets(archive, None, include_deferred=False)

    assert result.total_sessions == 1
    assert result.total_messages == 2
    assert result.providers == {"claude-ai-export": 1}
    assert result.tags == {"work": 1}


# ---------------------------------------------------------------------------
# 5. Happy path on a seeded archive
# ---------------------------------------------------------------------------


async def test_list_sessions_returns_seeded_rows(tmp_path: Path) -> None:
    """``list_sessions`` returns the seeded rows as typed sessions."""
    from polylogue.archive.session.domain_models import Session

    archive = _archive(tmp_path)
    db_path = archive.config.archive_root / "index.db"
    # Ensure schema is initialized by touching the repository before seeding.
    _ = archive.repository
    await _seed_two_sessions(db_path)
    try:
        rows = await archive.list_sessions()
        assert isinstance(rows, list)
        assert len(rows) == 2
        for row in rows:
            assert isinstance(row, Session)
        ids = {str(r.id) for r in rows}
        assert ids == {"claude-ai-export:conv-alpha", "chatgpt-export:conv-beta"}
    finally:
        await archive.close()


async def test_list_sessions_respects_origin_filter(tmp_path: Path) -> None:
    """``list_sessions(origin=...)`` narrows the result set."""
    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "index.db"
    await _seed_two_sessions(db_path)
    try:
        rows = await archive.list_sessions(origin="claude-ai-export")
        ids = {str(r.id) for r in rows}
        assert ids == {"claude-ai-export:conv-alpha"}
    finally:
        await archive.close()


async def test_list_sessions_respects_limit(tmp_path: Path) -> None:
    """``list_sessions(limit=1)`` returns at most one row."""
    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "index.db"
    await _seed_two_sessions(db_path)
    try:
        rows = await archive.list_sessions(limit=1)
        assert len(rows) == 1
    finally:
        await archive.close()


async def test_stats_reflects_seeded_archive(tmp_path: Path) -> None:
    """``stats()`` reports the seeded session count."""
    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "index.db"
    await _seed_two_sessions(db_path)
    try:
        result = await archive.stats()
        assert result.session_count == 2
        # Message count is at least the three rows we seeded.
        assert result.message_count >= 3
    finally:
        await archive.close()


async def test_get_session_returns_typed_object(tmp_path: Path) -> None:
    """``get_session`` returns a typed ``Session``."""
    from polylogue.archive.session.domain_models import Session

    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "index.db"
    await _seed_two_sessions(db_path)
    try:
        conv = await archive.get_session("claude-ai-export:conv-alpha")
        assert conv is not None
        assert isinstance(conv, Session)
        assert conv.title == "Alpha"
    finally:
        await archive.close()


async def test_session_digest_compiles_seeded_session(tmp_path: Path) -> None:
    """The private session-digest helper compiles deterministic transform output."""
    archive = _archive(tmp_path)
    db_path = archive.config.archive_root / "index.db"
    await _seed_two_sessions(db_path)
    try:
        digest = await archive._session_digest("claude-ai-export:conv-alpha")

        assert digest is not None
        assert digest.session_id == "claude-ai-export:conv-alpha"
        assert digest.transform.transform_id == "session_digest_v0"
        assert digest.transform.input_session_id == "claude-ai-export:conv-alpha"
        assert digest.resume_markdown.startswith("# Resume: Alpha")
        assert digest.raw_refs
        assert await archive._session_digest("nonexistent") is None
    finally:
        await archive.close()


async def test_session_digest_resolves_subagent_child_links(tmp_path: Path) -> None:
    """Archive-backed session digests enrich raw Task child ids with topology refs."""
    from polylogue.core.enums import BranchType

    archive = _archive(tmp_path)
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session-digest-parent",
        title="Session Digest Parent",
        messages=[
            ParsedMessage(
                provider_message_id="parent-message",
                role=Role.ASSISTANT,
                text="Delegating to subagent",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Task",
                        tool_id="task-tool",
                        tool_input={
                            "taskId": "task-123",
                            "child_session_id": "codex-session:session-digest-child",
                            "prompt": "Inspect session topology.",
                        },
                    )
                ],
            )
        ],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="session-digest-child",
        parent_session_provider_id="session-digest-parent",
        branch_type=BranchType.SUBAGENT,
        title="Session Digest Child",
        messages=[
            ParsedMessage(
                provider_message_id="child-message",
                role=Role.ASSISTANT,
                text="Child report",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="Child report")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            parent_id = archive_db.write_parsed(parent)
            child_id = archive_db.write_parsed(child)

        digest = await archive._session_digest(parent_id)

        assert digest is not None
        [subagent] = digest.subagent_reports
        assert subagent.child_session_id == child_id
        assert subagent.child_link_status == "resolved"
        assert subagent.child_link_type == "subagent"
        assert subagent.resolved_child_session_id == child_id
        assert "child_link_status=resolved" in digest.resume_markdown

        alias_digest = await archive._session_digest("codex:session-digest-parent")
        assert alias_digest is not None
        [alias_subagent] = alias_digest.subagent_reports
        assert alias_subagent.resolved_child_session_id == child_id
    finally:
        await archive.close()


async def test_list_read_view_profiles_exposes_shared_profile_payloads(tmp_path: Path) -> None:
    """``list_read_view_profiles`` exposes the executable read-view registry."""
    archive = _archive(tmp_path)
    try:
        profiles = await archive.list_read_view_profiles()

        views = {}
        for profile in profiles:
            view_id = profile["view_id"]
            assert isinstance(view_id, str)
            views[view_id] = profile
        assert views["raw"]["lossiness"] == "raw"
        assert views["raw"]["evidence_policy"] == "required"
        assert "recovery" not in views
        formats = views["context-image"]["formats"]
        assert isinstance(formats, list)
        assert "markdown" in formats
    finally:
        await archive.close()


async def test_explain_query_expression_exposes_shared_query_payload(tmp_path: Path) -> None:
    """``explain_query_expression`` exposes the canonical query AST payload."""
    archive = _archive(tmp_path)
    try:
        payload = await archive.explain_query_expression("sessions where exists block(type:code) AND messages > 10")

        assert payload["source_text"] == "sessions where exists block(type:code) AND messages > 10"
        selected_units = payload["selected_units"]
        execution_legs = payload["execution_legs"]
        assert isinstance(selected_units, list)
        assert isinstance(execution_legs, list)
        assert selected_units == ["block", "session"]
        assert "exists-block" in execution_legs
        assert "sql" in execution_legs
        assert payload["predicate"]
    finally:
        await archive.close()


async def test_explain_query_expression_exposes_terminal_unit_payload(tmp_path: Path) -> None:
    """Unit-source explain output reports the terminal row unit."""
    archive = _archive(tmp_path)
    try:
        payload = await archive.explain_query_expression("actions where action:file_edit AND path:polylogue")

        assert payload["lowerer"] == "lark-query-unit-source-to-terminal-unit"
        assert payload["selected_units"] == ["action"]
        assert payload["execution_legs"] == ["sql", "terminal-action-rows"]
        assert payload["plan_description"] == [
            "terminal unit source: action",
            "compatibility session selector: exists action(...)",
        ]
        assert payload["predicate"]
    finally:
        await archive.close()


async def test_explain_query_expression_exposes_sql_unit_payload(tmp_path: Path) -> None:
    """SQL unit-source explain output names its terminal projection leg."""
    archive = _archive(tmp_path)
    try:
        payload = await archive.explain_query_expression(
            "context-snapshots where session.repo:polylogue AND boundary:session_start"
        )

        assert payload["lowerer"] == "lark-query-unit-source-to-terminal-unit"
        assert payload["selected_units"] == ["context-snapshot"]
        assert payload["execution_legs"] == ["sql", "terminal-context-snapshot-rows"]
        assert payload["plan_description"] == [
            "terminal unit source: context-snapshot",
            "compatibility session selector: exists context-snapshot(...)",
        ]
        lowering_plan = cast(dict[str, object], payload["lowering_plan"])
        assert "compatibility_selector" in lowering_plan
        assert payload["predicate"]
    finally:
        await archive.close()


async def test_explain_import_exposes_shared_import_payload(tmp_path: Path) -> None:
    """``explain_import`` exposes the same bounded payload as ``polylogue import --explain``."""
    source = Path("tests/data/codex_event_stream/text_only_stream.jsonl")
    target = tmp_path / "session.jsonl"
    shutil.copy2(source, target)

    archive = _archive(tmp_path)
    try:
        payload = await archive.explain_import(target, source_name="codex")

        assert payload.mode == "import-explain"
        assert payload.produced.sessions == 1
        assert payload.produced.messages >= 1
        assert payload.entries[0].detected_provider == "codex"
        assert payload.entries[0].detected_origin == "codex-session"
        assert payload.entries[0].parser_mode == "grouped_records"
        assert payload.entries[0].produced.session_refs
    finally:
        await archive.close()


async def test_explain_import_reports_bounded_decode_failure(tmp_path: Path) -> None:
    """``explain_import`` reports malformed local input without exposing raw bytes."""
    target = tmp_path / "broken.json"
    target.write_text("{not json", encoding="utf-8")

    archive = _archive(tmp_path)
    try:
        payload = await archive.explain_import(target)

        assert payload.produced.sessions == 0
        assert payload.skipped
        assert payload.skipped[0].reason.startswith("decode failure:")
        rendered = payload.to_json(exclude_none=True)
        assert "{not json" not in rendered
        assert "raw_bytes" not in rendered
    finally:
        await archive.close()


async def test_explain_import_reads_archived_raw_evidence(tmp_path: Path) -> None:
    """``explain_import`` can explain already-archived source/index rows."""
    raw_id, source_path = _seed_import_explain_archive(tmp_path)
    archive = _archive(tmp_path)
    try:
        payload = await archive.explain_import(raw_ref=f"raw:{raw_id}")

        assert payload.mode == "import-explain"
        assert payload.source_path == f"archive:raw:{raw_id}"
        assert payload.produced.raw_records == 1
        assert payload.produced.sessions == 1
        assert payload.produced.messages == 1
        assert payload.produced.blocks == 1
        assert payload.produced.actions == 1
        assert payload.entries[0].raw_ref == f"raw:{raw_id}"
        assert payload.entries[0].detected_origin == "codex-session"
        assert payload.entries[0].artifact_kind == "session_record_stream"
        assert payload.entries[0].parser_mode == "archived_raw_session"
        assert payload.entries[0].schema_resolution == "passed"
        assert payload.entries[0].raw_evidence_refs == (f"raw:{raw_id}", "raw-artifact:artifact-1")
        assert payload.entries[0].normalization_warnings == ("normalized path separators",)
        rendered = payload.to_json(exclude_none=True)
        assert source_path not in rendered
        assert "raw_bytes" not in rendered
    finally:
        await archive.close()


async def test_explain_import_redacts_non_home_absolute_archive_paths(tmp_path: Path) -> None:
    """Archive-backed import explain does not expose arbitrary absolute paths by default."""
    raw_id, source_path = _seed_import_explain_archive(tmp_path, source_path="/realm/private/imports/session.jsonl")
    archive = _archive(tmp_path)
    try:
        payload = await archive.explain_import(raw_ref=f"raw:{raw_id}")

        assert payload.entries[0].source_path == ".../session.jsonl"
        assert source_path not in payload.to_json(exclude_none=True)
    finally:
        await archive.close()


async def test_query_completions_exposes_shared_completion_payload(tmp_path: Path) -> None:
    """``query_completions`` exposes registry-backed query metadata."""
    archive = _archive(tmp_path)
    try:
        payload = await archive.query_completions("field", incomplete="d")

        assert payload["kind"] == "field"
        assert payload["incomplete"] == "d"
        candidates = payload["candidates"]
        assert isinstance(candidates, list)
        candidate_payloads = [cast(dict[str, object], candidate) for candidate in candidates]
        date_candidate = next(candidate for candidate in candidate_payloads if candidate["value"] == "date")
        assert date_candidate["insert"] == "date "
        assert date_candidate["source"] == "DATE_QUERY_FIELD_REGISTRY"
        description = date_candidate["description"]
        assert isinstance(description, str)
        assert "date between 2026-01-01 and 2026-02-01" in description

        terminal_payload = await archive.query_completions("terminal-field", unit="observed-events", incomplete="del")
        assert terminal_payload["kind"] == "terminal-field"
        terminal_candidates = terminal_payload["candidates"]
        assert isinstance(terminal_candidates, list)
        terminal_candidate_payloads = [cast(dict[str, object], candidate) for candidate in terminal_candidates]
        assert [candidate["value"] for candidate in terminal_candidate_payloads] == ["delivery_state"]
        assert terminal_candidate_payloads[0]["insert"] == "delivery_state:"

        pipeline_payload = await archive.query_completions("pipeline-stage", unit="messages", incomplete="g")
        assert pipeline_payload["kind"] == "pipeline-stage"
        pipeline_candidates = pipeline_payload["candidates"]
        assert isinstance(pipeline_candidates, list)
        pipeline_candidate_payloads = [cast(dict[str, object], candidate) for candidate in pipeline_candidates]
        assert "group by role" in {candidate["value"] for candidate in pipeline_candidate_payloads}
        role_candidate = next(
            candidate for candidate in pipeline_candidate_payloads if candidate["value"] == "group by role"
        )
        assert role_candidate["insert"] == "group by role"
        assert role_candidate["payload_model"] == "MessageQueryRowPayload"
    finally:
        await archive.close()


async def test_compile_context_builds_message_segments_from_refs_and_query(tmp_path: Path) -> None:
    """``compile_context`` executes ContextSpec over supported context views."""
    from polylogue.context.compiler import ContextSpec

    archive = _archive(tmp_path)
    await _seed_two_sessions(archive.config.db_path)

    try:
        image = await archive.compile_context(
            ContextSpec(
                seed_refs=("session:claude-ai-export:conv-alpha", "assertion:claim-1"),
                seed_query="beta body",
                read_views=("messages",),
                max_tokens=20_000,
            )
        )

        assert image.spec.seed_query == "beta body"
        assert [segment.payload_kind for segment in image.segments] == ["messages", "messages"]
        assert {ref.object_id for ref in image.object_refs if ref.kind == "session"} == {
            "claude-ai-export:conv-alpha",
            "chatgpt-export:conv-beta",
        }
        assert {ref.session_id for ref in image.evidence_refs} == {
            "claude-ai-export:conv-alpha",
            "chatgpt-export:conv-beta",
        }
        assert image.token_estimate > 0
        unsupported = [item for item in image.omitted if item.reason == "unsupported"]
        assert len(unsupported) == 1
        assert unsupported[0].ref == "assertion:claim-1"
        message_segments = [segment for segment in image.segments if segment.payload_kind == "messages"]
        assert len(message_segments) == 2
        assert "user: alpha body" in (message_segments[0].markdown or "")
        assert "user: beta body" in (message_segments[1].markdown or "")

        unsupported_image = await archive.compile_context(
            ContextSpec(
                seed_refs=("session:claude-ai-export:conv-alpha",),
                read_views=("unsupported-a", "unsupported-b"),
                max_tokens=20_000,
            )
        )
        assert unsupported_image.segments == ()
        assert [(item.view, item.reason) for item in unsupported_image.omitted] == [
            ("unsupported-a", "unsupported"),
            ("unsupported-b", "unsupported"),
        ]
    finally:
        await archive.close()


async def test_compile_context_composes_temporal_and_chronicle_views(tmp_path: Path) -> None:
    """Composed context images materialize non-message read views as segments."""
    from polylogue.context.compiler import ContextSpec

    archive = _archive(tmp_path)
    await _seed_two_sessions(archive.config.db_path)

    try:
        image = await archive.compile_context(
            ContextSpec(
                seed_refs=("session:claude-ai-export:conv-alpha",),
                read_views=("temporal", "chronicle"),
                max_tokens=20_000,
                include_assertions=False,
            )
        )

        assert [segment.payload_kind for segment in image.segments] == ["temporal", "chronicle"]
        assert image.omitted == ()
        temporal, chronicle = image.segments
        assert "Temporal Evidence" in (temporal.markdown or "")
        assert "Events:" in (temporal.markdown or "")
        assert "Session Chronicle" in (chronicle.markdown or "")
        assert "alpha body" in (chronicle.markdown or "")
        assert "alpha reply" in (chronicle.markdown or "")
        assert {ref.object_id for ref in image.object_refs if ref.kind == "session"} == {"claude-ai-export:conv-alpha"}
        assert {ref.session_id for ref in image.evidence_refs} == {"claude-ai-export:conv-alpha"}
    finally:
        await archive.close()


async def test_compile_context_message_view_can_opt_out_of_assertion_injection(tmp_path: Path) -> None:
    """Plain message context stays message-shaped when assertion injection is off."""
    from polylogue.context.compiler import ContextSpec
    from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

    archive = _archive(tmp_path)
    await _seed_two_sessions(archive.config.db_path)
    with sqlite3.connect(archive.config.archive_root / "user.db") as conn:
        upsert_assertion(
            conn,
            assertion_id="context-opt-out-decision",
            target_ref="session:claude-ai-export:conv-alpha",
            kind=AssertionKind.DECISION,
            body_text="This assertion must not enter opted-out context images.",
            author_ref="user:test",
            author_kind="user",
            evidence_refs=["claude-ai-export:conv-alpha"],
            status="active",
            visibility="private",
            context_policy={"inject": True},
            now_ms=1_700_000_000_000,
        )

    try:
        image = await archive.compile_context(
            ContextSpec(seed_refs=("session:claude-ai-export:conv-alpha",), include_assertions=False)
        )

        assert image.assertion_refs == ()
        assert image.segments[0].assertion_refs == ()
        markdown = image.segments[0].markdown
        assert markdown is not None
        assert "This assertion must not enter opted-out context images." not in markdown
    finally:
        await archive.close()


async def test_compile_context_records_missing_and_budget_omissions(tmp_path: Path) -> None:
    """``compile_context`` fails closed when seeds or budget do not resolve."""
    from polylogue.context.compiler import ContextSpec

    archive = _archive(tmp_path)
    await _seed_two_sessions(archive.config.db_path)

    try:
        image = await archive.compile_context(
            ContextSpec(
                seed_refs=("session:missing",),
                seed_query="no such query",
                max_tokens=1,
            )
        )

        assert image.segments == ()
        assert image.token_estimate == 0
        assert {(item.ref, item.query, item.reason) for item in image.omitted} == {
            ("session:missing", None, "not_found"),
            (None, "no such query", "not_found"),
        }

        budgeted = await archive.compile_context(
            ContextSpec(
                seed_refs=("session:claude-ai-export:conv-alpha",),
                max_tokens=1,
            )
        )
        assert len(budgeted.segments) == 1
        assert budgeted.omitted == ()
        markdown = budgeted.segments[0].markdown or ""
        assert "alpha" in markdown
        assert "omitted" in markdown
        assert budgeted.segments[0].caveats

        fallback = await archive.compile_context(
            ContextSpec(
                seed_refs=("session:claude-ai-export:conv-alpha",),
                read_views=("raw",),
                max_tokens=50,
            )
        )
        assert len(fallback.segments) == 1
        assert fallback.segments[0].payload_kind == "messages"
        assert fallback.segments[0].markdown
        assert [(item.view, item.reason) for item in fallback.omitted] == [("raw", "unsupported")]
    finally:
        await archive.close()


async def test_list_assertion_claims_filters_lifecycle_claims(tmp_path: Path) -> None:
    """``list_assertion_claims`` exposes policy-aware assertion reads."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

    archive = _archive(tmp_path)
    user_db = archive.config.archive_root / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    with sqlite3.connect(user_db) as conn:
        upsert_assertion(
            conn,
            assertion_id="claim-decision-inject",
            target_ref="session:codex-session:claim-target",
            scope_ref="repo:polylogue",
            kind=AssertionKind.DECISION,
            body_text="Use the shared query metadata module.",
            author_ref="agent:codex",
            # author_kind="user": an active, injectable decision is always
            # user-authored in production (37t.15) -- non-user writes land
            # as candidates regardless of the caller-supplied status/policy.
            author_kind="user",
            evidence_refs=["codex-session:claim-target::m1"],
            status="active",
            visibility="private",
            context_policy={"inject": True},
            now_ms=1_700_000_000_000,
        )
        upsert_assertion(
            conn,
            assertion_id="claim-caveat-private",
            target_ref="session:codex-session:claim-target",
            scope_ref="repo:polylogue",
            kind=AssertionKind.CAVEAT,
            body_text="Review evidence is still partial.",
            author_ref="agent:codex",
            author_kind="agent",
            status="candidate",
            visibility="private",
            context_policy={"inject": False},
            now_ms=1_700_000_000_100,
        )
        upsert_assertion(
            conn,
            assertion_id="claim-deleted",
            target_ref="session:codex-session:claim-target",
            scope_ref="repo:polylogue",
            kind=AssertionKind.DECISION,
            body_text="Deleted claims should not appear by default.",
            status="deleted",
            context_policy={"inject": True},
            now_ms=1_700_000_000_200,
        )
        upsert_assertion(
            conn,
            assertion_id="claim-other-target",
            target_ref="session:codex-session:other",
            scope_ref="repo:polylogue",
            kind=AssertionKind.LESSON,
            body_text="Other target.",
            status="active",
            context_policy={"inject": True},
            now_ms=1_700_000_000_300,
        )
        conn.commit()

    try:
        claims = await archive.list_assertion_claims(target_ref="session:codex-session:claim-target")
        assert [claim.assertion_id for claim in claims] == ["claim-caveat-private", "claim-decision-inject"]

        injectable = await archive.list_assertion_claims(
            target_ref="session:codex-session:claim-target",
            context_inject=True,
        )
        assert [claim.assertion_id for claim in injectable] == ["claim-decision-inject"]

        decisions = await archive.list_assertion_claims(
            kinds=(AssertionKind.DECISION,),
            target_ref="session:codex-session:claim-target",
            statuses=("active",),
        )
        assert [claim.assertion_id for claim in decisions] == ["claim-decision-inject"]

        deleted = await archive.list_assertion_claims(
            target_ref="session:codex-session:claim-target",
            statuses=("deleted",),
        )
        assert [claim.assertion_id for claim in deleted] == ["claim-deleted"]
    finally:
        await archive.close()


async def test_get_actions_derives_from_archive_blocks(tmp_path: Path) -> None:
    """``get_actions`` derives tool invocations from content blocks.

    Archives have no materialized ``actions`` table; the facade
    rebuilds actions on read from the session's tool-use blocks. This pins that
    derivation contract (single, batch, and the missing-id empty cases).
    """
    archive = _archive(tmp_path)
    with ArchiveStore(archive.config.archive_root) as archive_db:
        archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="tooluse-conv",
                title="Tooluse",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        text="running a command",
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Bash",
                                tool_id="t1",
                                tool_input={"command": "ls"},
                            )
                        ],
                    )
                ],
            )
        )
    try:
        summaries = await archive.list_summaries()
        assert len(summaries) == 1
        native_id = str(summaries[0].id)

        actions = await archive.get_actions(native_id)
        assert [action.tool_name for action in actions] == ["Bash"]

        batch = await archive.get_actions_batch([native_id])
        assert batch[native_id] == actions

        # Unknown IDs: empty tuple, and omitted from the batch mapping.
        assert await archive.get_actions("nonexistent") == ()
        assert await archive.get_actions_batch(["nonexistent"]) == {}
    finally:
        await archive.close()


# ---------------------------------------------------------------------------
# 6. Error envelope: SessionNotFoundError is a typed PolylogueError
# ---------------------------------------------------------------------------


def test_session_not_found_is_typed_polylogue_error() -> None:
    """``SessionNotFoundError`` carries an HTTP status code and inherits the typed base."""
    assert issubclass(SessionNotFoundError, PolylogueError)
    assert SessionNotFoundError.http_status_code == 404


@pytest.mark.parametrize(
    "method_name",
    sorted(MUTATION_BY_ID_RAISES_METHODS),
)
async def test_mutation_methods_raise_on_unknown_id(
    tmp_path: Path,
    method_name: str,
) -> None:
    """Mutation-by-ID methods raise ``SessionNotFoundError`` for unknown IDs."""
    archive = _archive(tmp_path)
    try:
        method = getattr(archive, method_name)
        # Construct the minimum-valid argument set per method shape.
        coro: object
        if method_name in {"add_tag", "remove_tag"}:
            coro = method("does-not-exist", "some-tag")
        elif method_name in {"add_mark", "remove_mark"}:
            coro = method("does-not-exist", "star")
        elif method_name == "record_correction":
            coro = method("does-not-exist", "tag_accept", {"value": "x"})
        elif method_name == "delete_correction":
            coro = method("does-not-exist", "tag_accept")
        elif method_name == "clear_corrections":
            coro = method("does-not-exist")
        elif method_name == "save_annotation":
            coro = method("note-1", "does-not-exist", "note text")
        else:
            raise AssertionError(f"Unhandled mutation method in parametrize set: {method_name}")
        with pytest.raises(SessionNotFoundError):
            await coro  # type: ignore[misc]
    finally:
        await archive.close()


async def test_delete_session_returns_false_on_unknown_id(tmp_path: Path) -> None:
    """``delete_session`` returns ``False`` rather than raising for unknown IDs."""
    archive = _archive(tmp_path)
    try:
        result = await archive.delete_session("nonexistent")
        assert result is False
    finally:
        await archive.close()


async def test_delete_session_safe_returns_typed_not_found(tmp_path: Path) -> None:
    """``delete_session_safe`` returns ``outcome='not_found'`` for unknown IDs."""
    from polylogue.surfaces.payloads import DeleteSessionResult

    archive = _archive(tmp_path)
    try:
        result = await archive.delete_session_safe("nonexistent")
        assert isinstance(result, DeleteSessionResult)
        assert result.outcome == "not_found"
    finally:
        await archive.close()


async def test_get_session_returns_none_for_unknown_id(tmp_path: Path) -> None:
    """Read-by-ID methods return ``None`` for unknown IDs, not an exception."""
    archive = _archive(tmp_path)
    with ArchiveStore(tmp_path):
        pass
    try:
        assert await archive.get_session("nonexistent") is None
        assert await archive.get_session_summary("nonexistent") is None
        assert await archive.get_session_profile_insight("nonexistent") is None
    finally:
        await archive.close()


async def test_get_messages_paginated_raises_for_unknown_id(tmp_path: Path) -> None:
    """``get_messages_paginated`` raises ``SessionNotFoundError`` for unknown IDs."""
    archive = _archive(tmp_path)
    try:
        with pytest.raises(SessionNotFoundError):
            await archive.get_messages_paginated("nonexistent")
    finally:
        await archive.close()


async def test_get_messages_paginated_applies_content_projection(tmp_path: Path) -> None:
    """Message reads honor the same content projection as session reads."""
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec

    archive = _archive(tmp_path)
    body = "Alpha\n\n```python\nprint('x')\n```\n\nOmega"
    with ArchiveStore(tmp_path) as archive_db:
        session_id = archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="projected-messages",
                title="Projected messages",
                messages=[
                    ParsedMessage(
                        provider_message_id="projected-m1",
                        role=Role.ASSISTANT,
                        text=body,
                        blocks=[],
                    )
                ],
            )
        )
    try:
        messages, total = await archive.get_messages_paginated(
            session_id,
            content_projection=ContentProjectionSpec.prose_only(),
        )
    finally:
        await archive.close()

    assert total == 1
    assert messages[0].text == "Alpha\n\nOmega"


async def test_message_hydration_preserves_origin_provider_and_structural_tool_outcome(tmp_path: Path) -> None:
    """Semantic readers receive the evidence fields required by shared cards."""

    archive = _archive(tmp_path)
    with ArchiveStore(tmp_path) as archive_db:
        session_id = archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="semantic-card-hydration",
                title="Semantic card hydration",
                messages=[
                    ParsedMessage(
                        provider_message_id="assistant-tool",
                        role=Role.ASSISTANT,
                        text=None,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="exec_command",
                                tool_id="call-1",
                                tool_input={"command": "pytest -q"},
                            ),
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                tool_id="call-1",
                                text="checks passed",
                                is_error=False,
                                exit_code=0,
                            ),
                        ],
                    )
                ],
            )
        )
    try:
        messages, total = await archive.get_messages_paginated(session_id, limit=10)
    finally:
        await archive.close()

    assert total == 1
    assert messages[0].provider is Provider.CODEX
    assert len(messages[0].blocks) == 2
    result = messages[0].blocks[1]
    assert result["id"]
    assert result["tool_result_is_error"] == 0
    assert result["tool_result_exit_code"] == 0


async def test_get_messages_paginated_filters_material_origin(tmp_path: Path) -> None:
    """Message reads can filter authoredness separately from provider role."""

    archive = _archive(tmp_path)
    with ArchiveStore(tmp_path) as archive_db:
        session_id = archive_db.write_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="material-origin-messages",
                title="Material origin messages",
                messages=[
                    ParsedMessage(
                        provider_message_id="protocol-user",
                        role=Role.USER,
                        text="runtime protocol envelope",
                        material_origin=MaterialOrigin.RUNTIME_PROTOCOL,
                    ),
                    ParsedMessage(
                        provider_message_id="authored-user",
                        role=Role.USER,
                        text="human prompt",
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    ),
                ],
            )
        )
    try:
        role_messages, role_total = await archive.get_messages_paginated(
            session_id,
            message_role=(Role.USER,),
        )
        authored_messages, authored_total = await archive.get_messages_paginated(
            session_id,
            material_origin=(MaterialOrigin.HUMAN_AUTHORED,),
        )
    finally:
        await archive.close()

    assert role_total == 2
    assert [str(message.id).rsplit(":", 1)[-1] for message in role_messages] == ["protocol-user", "authored-user"]
    assert authored_total == 1
    assert str(authored_messages[0].id).endswith(":authored-user")


# ---------------------------------------------------------------------------
# 7. Required-arg validation
# ---------------------------------------------------------------------------


async def test_save_annotation_rejects_blank_inputs(tmp_path: Path) -> None:
    """``save_annotation`` validates non-empty ``annotation_id`` and ``note_text``."""
    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "index.db"
    await _seed_two_sessions(db_path)
    try:
        with pytest.raises(ValueError):
            await archive.save_annotation("", "conv-alpha", "note text")
        with pytest.raises(ValueError):
            await archive.save_annotation("note-1", "conv-alpha", "")
    finally:
        await archive.close()


async def test_save_workspace_rejects_invalid_mode(tmp_path: Path) -> None:
    """``save_workspace`` rejects modes outside the documented vocabulary."""
    archive = _archive(tmp_path)
    try:
        with pytest.raises(ValueError):
            await archive.save_workspace(
                workspace_id="ws-1",
                name="My Workspace",
                mode="not-a-mode",
                open_targets_json="[]",
                layout_json="{}",
            )
    finally:
        await archive.close()


async def test_neighbor_candidates_requires_id_or_query(tmp_path: Path) -> None:
    """``neighbor_candidates`` needs at least one of ``session_id`` / ``query``.

    Asserts the underlying operations layer rejects the no-input call;
    the facade is a pass-through but the contract surface must reject
    semantically-empty calls rather than returning misleading results.
    """
    archive = _archive(tmp_path)
    try:
        with pytest.raises((ValueError, TypeError, PolylogueError)):
            await archive.neighbor_candidates()
    finally:
        await archive.close()


# ---------------------------------------------------------------------------
# 8. Search contract
# ---------------------------------------------------------------------------


async def test_search_returns_typed_result_on_empty_archive(tmp_path: Path) -> None:
    """``search()`` returns a typed ``SearchResult`` even with no hits."""
    from polylogue.storage.search.models import SearchResult

    archive = _archive(tmp_path)
    try:
        result = await archive.search("nothing here")
        assert isinstance(result, SearchResult)
    finally:
        await archive.close()


async def test_search_envelope_returns_typed_envelope_on_empty_archive(
    tmp_path: Path,
) -> None:
    """``search_envelope()`` returns a typed ``SearchEnvelope`` even with no hits."""
    from polylogue.surfaces.payloads import SearchEnvelope

    archive = _archive(tmp_path)
    try:
        envelope = await archive.search_envelope("nothing here")
        assert isinstance(envelope, SearchEnvelope)
    finally:
        await archive.close()


async def test_search_envelope_executes_role_count_boolean_predicate(tmp_path: Path) -> None:
    """``search_envelope()`` executes role-split count predicates through the shared DSL."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="role-count-hit",
                    title="Role Count Hit",
                    messages=[
                        ParsedMessage(
                            provider_message_id="u1",
                            role=Role.USER,
                            text="one question",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="one question")],
                        ),
                        ParsedMessage(
                            provider_message_id="u2",
                            role=Role.USER,
                            text="second question",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second question")],
                        ),
                        ParsedMessage(
                            provider_message_id="a1",
                            role=Role.ASSISTANT,
                            text="one two three four five",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="one two three four five")],
                        ),
                    ],
                )
            )
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="role-count-miss",
                    title="Role Count Miss",
                    messages=[
                        ParsedMessage(
                            provider_message_id="u1",
                            role=Role.USER,
                            text="one question",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="one question")],
                        ),
                        ParsedMessage(
                            provider_message_id="a1",
                            role=Role.ASSISTANT,
                            text="one two three four five",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="one two three four five")],
                        ),
                    ],
                )
            )

        envelope = await archive.search_envelope(
            "sessions where user_messages >= 2 AND assistant_words between 5 and 6"
        )

        assert [hit.session.id for hit in envelope.hits] == ["codex-session:role-count-hit"]
    finally:
        await archive.close()


async def test_search_envelope_paginates_filter_only_boolean_predicates(tmp_path: Path) -> None:
    """Filter-only DSL envelopes honor cursor offsets when fetching the next page."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            for index in range(3):
                archive_db.write_parsed(
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=f"role-count-page-{index}",
                        title=f"Role Count Page {index}",
                        messages=[
                            ParsedMessage(
                                provider_message_id=f"u{index}",
                                role=Role.USER,
                                text=f"page {index} question",
                                blocks=[
                                    ParsedContentBlock(
                                        type=BlockType.TEXT,
                                        text=f"page {index} question",
                                    )
                                ],
                            )
                        ],
                    )
                )

        query = "sessions where user_messages >= 1"
        first = await archive.search_envelope(query, limit=1)
        assert first.total == 3
        assert len(first.hits) == 1
        assert first.next_cursor is not None

        second = await archive.search_envelope(query, limit=1, cursor=first.next_cursor)

        assert second.total == 3
        assert len(second.hits) == 1
        assert second.hits[0].session.id != first.hits[0].session.id
    finally:
        await archive.close()


async def test_query_sessions_sampled_text_query_uses_search_kwargs(tmp_path: Path) -> None:
    """Sampled specs with text queries do not leak list-only kwargs into FTS search."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="sampled-text-query",
                    title="Sampled Text Query",
                    messages=[
                        ParsedMessage(
                            provider_message_id="u1",
                            role=Role.USER,
                            text="sampled needle",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="sampled needle")],
                        )
                    ],
                )
            )

        rows = await archive.query_sessions(query="sampled needle", sample=1, limit=5)

        assert [row["id"] for row in rows] == ["codex-session:sampled-text-query"]
    finally:
        await archive.close()


async def test_query_units_returns_typed_envelope_on_empty_archive(tmp_path: Path) -> None:
    """``query_units()`` returns a typed terminal-unit envelope even with no rows."""
    from polylogue.surfaces.payloads import QueryUnitEnvelope

    archive = _archive(tmp_path)
    try:
        envelope = await archive.query_units("messages where text:missing")
        assert isinstance(envelope, QueryUnitEnvelope)
        assert envelope.mode == "query-unit"
        assert envelope.unit == "message"
        assert envelope.items == ()
    finally:
        await archive.close()


async def test_query_units_applies_session_scope_filters(tmp_path: Path) -> None:
    """``query_units()`` applies surrounding session filters before returning rows."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="unit-filter-codex",
                    title="Unit filter Codex",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="shared terminal needle",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="shared terminal needle")],
                        )
                    ],
                )
            )
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CHATGPT,
                    provider_session_id="unit-filter-chatgpt",
                    title="Unit filter ChatGPT",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="shared terminal needle",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="shared terminal needle")],
                        )
                    ],
                )
            )

        envelope = await archive.query_units("messages where text:needle", origin="codex-session")

        assert [cast(Any, item).session_id for item in envelope.items] == ["codex-session:unit-filter-codex"]
    finally:
        await archive.close()


async def test_query_units_reports_pipeline_stages(tmp_path: Path) -> None:
    """``query_units()`` exposes the terminal pipeline that shaped returned rows."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="unit-pipeline-codex",
                    title="Unit pipeline Codex",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.ASSISTANT,
                            text="first terminal pipeline row",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first terminal pipeline row")],
                        ),
                        ParsedMessage(
                            provider_message_id="m2",
                            role=Role.ASSISTANT,
                            text="second terminal pipeline row",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="second terminal pipeline row")],
                        ),
                    ],
                )
            )

        envelope = await archive.query_units(
            "sessions where origin:codex-session | messages where role:assistant | limit 1 | offset 1"
        )

        assert envelope.pipeline_stages == (
            {
                "kind": "session_scope",
                "predicate": {
                    "field": "origin",
                    "field_ref": {"name": "origin", "scope": "session", "source_name": "origin"},
                    "kind": "field",
                    "op": "=",
                    "values": ["codex-session"],
                },
            },
            {"kind": "limit", "value": 1},
            {"kind": "offset", "value": 1},
            {"kind": "terminal", "action": "rows"},
        )
        assert envelope.pipeline is not None
        assert envelope.pipeline["stages"] == list(envelope.pipeline_stages)
        assert envelope.pipeline["result"] == {"limit": 1, "offset": 1}
        assert envelope.pipeline["session_scope"] == envelope.pipeline_stages[0]["predicate"]
        assert envelope.limit == 1
        assert [cast(Any, item).message_id for item in envelope.items] == ["codex-session:unit-pipeline-codex:m2"]
    finally:
        await archive.close()


async def test_query_units_returns_aggregate_envelope(tmp_path: Path) -> None:
    """``query_units()`` returns grouped aggregate counts for terminal pipelines."""
    from polylogue.surfaces.payloads import QueryUnitAggregateEnvelope

    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="unit-aggregate-codex",
                    title="Unit aggregate Codex",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m-user",
                            role=Role.USER,
                            text="aggregate facade",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="aggregate facade")],
                        ),
                        ParsedMessage(
                            provider_message_id="m-assistant-1",
                            role=Role.ASSISTANT,
                            text="aggregate facade one",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="aggregate facade one")],
                        ),
                        ParsedMessage(
                            provider_message_id="m-assistant-2",
                            role=Role.ASSISTANT,
                            text="aggregate facade two",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="aggregate facade two")],
                        ),
                    ],
                )
            )

        envelope = await archive.query_units(
            "sessions where origin:codex-session | messages where text:aggregate | group by role | count"
        )

        assert isinstance(envelope, QueryUnitAggregateEnvelope)
        assert envelope.mode == "query-unit-aggregate"
        assert [(row.group_key, row.count) for row in envelope.items] == [("assistant", 2), ("user", 1)]
    finally:
        await archive.close()


async def test_query_units_accepts_inline_session_scope(tmp_path: Path) -> None:
    """``query_units()`` accepts owning-session scope inside the shared DSL."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="unit-inline-codex",
                    title="Unit inline Codex",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="shared inline needle",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="shared inline needle")],
                        )
                    ],
                )
            )
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CHATGPT,
                    provider_session_id="unit-inline-chatgpt",
                    title="Unit inline ChatGPT",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="shared inline needle",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="shared inline needle")],
                        )
                    ],
                )
            )

        envelope = await archive.query_units("messages where session.origin:codex-session AND text:needle")

        assert [cast(Any, item).session_id for item in envelope.items] == ["codex-session:unit-inline-codex"]
    finally:
        await archive.close()


async def test_query_units_returns_assertion_rows(tmp_path: Path) -> None:
    """``query_units()`` exposes assertion terminal rows through the facade."""
    import sqlite3

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="unit-assertion-codex",
                    title="Unit assertion Codex",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="assertion target",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="assertion target")],
                        )
                    ],
                )
            )
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CHATGPT,
                    provider_session_id="unit-assertion-chatgpt",
                    title="Unit assertion ChatGPT",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="assertion target",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="assertion target")],
                        )
                    ],
                )
            )
        user_db = archive.config.archive_root / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        with sqlite3.connect(user_db) as conn:
            upsert_assertion(
                conn,
                assertion_id="facade-decision-hit",
                target_ref="session:codex-session:unit-assertion-codex",
                kind=AssertionKind.DECISION,
                key="query-unit",
                value={"rank": 1},
                body_text="Review assertion query unit facade wiring.",
                author_ref="agent:codex",
                # author_kind="user": active decisions are always user-authored
                # in production (37t.15) -- these seed rows test active-status
                # query filtering, not the write-side security invariant.
                author_kind="user",
                evidence_refs=["session:codex-session:unit-assertion-codex#message:m1"],
                status="active",
                visibility="public",
                now_ms=1000,
            )
            upsert_assertion(
                conn,
                assertion_id="facade-decision-miss",
                target_ref="session:chatgpt-export:unit-assertion-chatgpt",
                kind=AssertionKind.DECISION,
                body_text="Review another origin.",
                author_ref="agent:codex",
                author_kind="user",
                status="active",
                now_ms=500,
            )
            conn.commit()

        envelope = await archive.query_units(
            "assertions where kind:decision AND status:active AND text:facade",
            origin="codex-session",
        )

        assert envelope.unit == "assertion"
        [item] = envelope.items
        payload = item.model_dump(mode="json")
        assert payload["unit"] == "assertion"
        assert payload["assertion_id"] == "facade-decision-hit"
        assert payload["target_ref"] == "session:codex-session:unit-assertion-codex"
        assert payload["kind"] == "decision"
        assert payload["value"] == {"rank": 1}
        assert payload["evidence_refs"] == ["session:codex-session:unit-assertion-codex#message:m1"]
    finally:
        await archive.close()


async def test_query_units_returns_context_snapshot_rows(tmp_path: Path) -> None:
    """``query_units()`` exposes runtime context snapshots through the facade."""
    from polylogue.surfaces.payloads import ContextSnapshotQueryRowPayload

    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="facade-context-snapshot-v1",
                    title="Facade context snapshot",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="facade context snapshot seed",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="facade context snapshot seed")],
                        )
                    ],
                )
            )

        envelope = await archive.query_units(
            "context-snapshots where boundary:session_start AND text:facade-context-snapshot-v1"
        )

        assert envelope.unit == "context-snapshot"
        [item] = envelope.items
        assert isinstance(item, ContextSnapshotQueryRowPayload)
        assert item.session_id == "codex-session:facade-context-snapshot-v1"
        assert item.boundary == "session_start"
        assert item.evidence_refs == ("codex-session:facade-context-snapshot-v1",)
    finally:
        await archive.close()


async def test_resolve_ref_returns_bounded_session_message_block_and_runtime_payloads(tmp_path: Path) -> None:
    """``resolve_ref()`` makes public refs actionable without broad search."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="ref-resolution-v1",
                    title="Ref resolution fixture",
                    working_directories=["/realm/project/polylogue"],
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="resolve this public ref",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="resolve this public ref")],
                        )
                    ],
                )
            )

        session_payload = await archive.resolve_ref(f"session:{session_id}")
        assert session_payload.resolved is True
        assert session_payload.payload_kind == "session-summary"
        assert session_payload.payload is not None
        assert session_payload.payload["id"] == session_id

        message_id = f"{session_id}:m1"
        message_payload = await archive.resolve_ref(f"message:{session_id}:{message_id}")
        assert message_payload.resolved is True
        assert message_payload.payload_kind == "message"
        assert message_payload.payload is not None
        assert message_payload.payload["id"] == message_id

        evidence_message_payload = await archive.resolve_ref(f"{session_id}::{message_id}")
        assert evidence_message_payload.resolved is True
        assert evidence_message_payload.payload_kind == "message"
        assert evidence_message_payload.evidence_refs == (f"{session_id}::{message_id}",)

        block_payload = await archive.resolve_ref(f"block:{message_id}:0")
        assert block_payload.resolved is True
        assert block_payload.payload_kind == "block"
        assert block_payload.payload is not None
        assert block_payload.payload["block_id"] == f"{message_id}:0"

        evidence_block_payload = await archive.resolve_ref(f"{session_id}::{message_id}::0")
        assert evidence_block_payload.resolved is True
        assert evidence_block_payload.payload_kind == "block"
        assert evidence_block_payload.evidence_refs == (f"{session_id}::{message_id}::0",)

        runtime_payload = await archive.resolve_ref(f"context-snapshot:{session_id}:session_start")
        assert runtime_payload.resolved is True
        assert runtime_payload.payload_kind == "context-snapshot"
        assert runtime_payload.payload is not None
        assert runtime_payload.payload["session_id"] == session_id

        unsupported_payload = await archive.resolve_ref("commit:abc123")
        assert unsupported_payload.resolved is False
        assert unsupported_payload.kind == "commit"
        assert unsupported_payload.payload is None

        missing_payload = await archive.resolve_ref("session:missing")
        assert missing_payload.resolved is False
        assert missing_payload.payload is None
    finally:
        await archive.close()


@pytest.mark.parametrize(
    ("kind", "object_id"),
    [
        ("query", "sha256:deadbeef"),
        ("query-run", "sha256:deadbeef:run-1"),
        ("result-set", "sha256:deadbeef:run-1:result-1"),
        ("finding", "finding-hash-1"),
        ("cohort", "cohort-1"),
        ("analysis", "analysis-1"),
        ("annotation-batch", "batch-1"),
    ],
)
async def test_resolve_ref_returns_typed_pending_payload_for_analysis_provenance_kinds(
    tmp_path: Path, kind: str, object_id: str
) -> None:
    """polylogue-rxdo.1: refs land ahead of storage; resolution stubs cleanly.

    ``query``/``query-run``/``result-set``/``finding``/``cohort``/``analysis``/
    ``annotation-batch`` are registered ObjectRefKind values with no backing
    table yet (polylogue-rxdo.2/.3/.4/.7/.8). resolve_ref must not raise and
    must not silently pretend to resolve them — it returns a typed pending
    payload carrying reason=substrate-pending so a client can distinguish
    "not implemented yet" from "does not exist".
    """
    archive = _archive(tmp_path)
    try:
        ref = f"{kind}:{object_id}"
        payload = await archive.resolve_ref(ref)

        assert payload.resolved is False
        assert payload.kind == kind
        assert payload.normalized_ref == ref
        assert payload.payload_kind == "pending"
        assert payload.payload is not None
        assert payload.payload["reason"] == "substrate-pending"
        assert payload.payload["kind"] == kind
        assert payload.payload["unit"] == "pending"
        assert any("substrate-pending" in caveat for caveat in payload.caveats)
    finally:
        await archive.close()


async def test_resolve_ref_returns_assertion_payload(tmp_path: Path) -> None:
    """Assertion refs resolve through the shared assertion claim DTO."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

    archive = _archive(tmp_path)
    try:
        user_db = archive.config.archive_root / "user.db"
        initialize_archive_database(user_db, ArchiveTier.USER)
        with sqlite3.connect(user_db) as conn:
            upsert_assertion(
                conn,
                assertion_id="assertion-ref-resolution",
                target_ref="session:codex-session:ref-resolution-v1",
                kind=AssertionKind.DECISION,
                body_text="Resolve assertion refs through shared payloads.",
                evidence_refs=["codex-session:ref-resolution-v1"],
            )

        payload = await archive.resolve_ref("assertion:assertion-ref-resolution")

        assert payload.resolved is True
        assert payload.payload_kind == "assertion-claim"
        assert payload.payload is not None
        assert payload.payload["assertion_id"] == "assertion-ref-resolution"
        assert payload.evidence_refs == ("codex-session:ref-resolution-v1",)
    finally:
        await archive.close()


def _delegation_parent_session(*, provider_session_id: str, with_dispatch: bool) -> ParsedSession:
    """Ingest-shaped parent fixture: writes real session/message/block rows
    through the live archive writer (``ArchiveStore.write_parsed`` ->
    ``write_parsed_session_to_archive``), the same seam the daemon uses --
    not a hand-inserted SQL row. A Task ``tool_use``/``tool_result`` pair
    gets classified ``semantic_type='subagent'`` at write time
    (``polylogue/storage/sqlite/archive_tiers/write.py:_semantic_type``),
    which is what the polylogue-y964 `delegations` view spines on."""

    messages = [
        ParsedMessage(
            provider_message_id="turn-0",
            role=Role.USER,
            text="please look into this",
            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="please look into this")],
        )
    ]
    if with_dispatch:
        messages.append(
            ParsedMessage(
                provider_message_id="dispatch",
                role=Role.ASSISTANT,
                model_name="claude-opus-4-8",
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Task",
                        tool_id="task-1",
                        tool_input={"prompt": "audit the thing", "subagent_type": "general-purpose"},
                    )
                ],
            )
        )
        messages.append(
            ParsedMessage(
                provider_message_id="dispatch-result",
                role=Role.USER,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_id="task-1",
                        text="3 gaps found",
                        is_error=False,
                        exit_code=0,
                    )
                ],
            )
        )
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id=provider_session_id,
        title="Delegation parent fixture",
        messages=messages,
    )


async def test_resolve_ref_returns_resolved_delegation_attempt_payload(tmp_path: Path) -> None:
    """polylogue-lph4: action-observed delegation identity is
    (parent_session_id, instruction_tool_use_block_id) -- the ref carries
    only the block id since it already embeds the parent session id
    structurally (block_id = message_id:position, message_id =
    session_id:native_id)."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            parent_session_id = archive_db.write_parsed(
                _delegation_parent_session(provider_session_id="delegation-parent-v1", with_dispatch=True)
            )
            child_session_id = archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="delegation-child-v1",
                    title="Delegation child fixture",
                    messages=[
                        ParsedMessage(provider_message_id="c1", role=Role.ASSISTANT, text="on it"),
                    ],
                    parent_session_provider_id="delegation-parent-v1",
                    branch_type=BranchType.SUBAGENT,
                )
            )

        instruction_block_id = f"{parent_session_id}:dispatch:0"
        payload = await archive.resolve_ref(f"delegation:{instruction_block_id}")

        assert payload.resolved is True
        assert payload.kind == "delegation"
        assert payload.payload_kind == "delegation-attempt"
        assert payload.payload is not None
        assert payload.payload["mapping_state"] == "resolved"
        assert payload.payload["parent_session_id"] == parent_session_id
        assert payload.payload["child_session_id"] == child_session_id
        assert payload.payload["instruction_tool_use_block_id"] == instruction_block_id
        assert payload.payload["dispatch_turn_model"] == "claude-opus-4-8"
        assert payload.object_refs == (f"session:{parent_session_id}", f"session:{child_session_id}")
        assert f"block:{instruction_block_id}" in payload.evidence_refs
        assert payload.caveats == ()
    finally:
        await archive.close()


async def test_resolve_ref_returns_edge_only_delegation_attempt_payload(tmp_path: Path) -> None:
    """A resolved child with no parent-side dispatch action (e.g. a Codex
    async subagent) resolves through the deterministic edge identity and is
    typed edge_only -- never given a fabricated instruction."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            parent_session_id = archive_db.write_parsed(
                _delegation_parent_session(provider_session_id="delegation-edge-parent-v1", with_dispatch=False)
            )
            child_session_id = archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="delegation-edge-child-v1",
                    title="Delegation edge-only child fixture",
                    messages=[ParsedMessage(provider_message_id="c1", role=Role.ASSISTANT, text="on it")],
                    parent_session_provider_id="delegation-edge-parent-v1",
                    branch_type=BranchType.SUBAGENT,
                )
            )

        edge_ref = f"delegation:{delegation_edge_object_id(parent_session_id, child_session_id)}"
        payload = await archive.resolve_ref(edge_ref)

        assert payload.resolved is True
        assert payload.payload_kind == "delegation-attempt"
        assert payload.payload is not None
        assert payload.payload["mapping_state"] == "edge_only"
        assert payload.payload["parent_session_id"] == parent_session_id
        assert payload.payload["child_session_id"] == child_session_id
        assert payload.payload["instruction_tool_use_block_id"] is None
        assert payload.payload["instruction_payload"] is None
        assert any("edge_only" in caveat for caveat in payload.caveats)
    finally:
        await archive.close()


async def test_resolve_ref_returns_ambiguous_delegation_attempt_payload(tmp_path: Path) -> None:
    """Two dispatch actions but only one resolved child: rank-pairing would
    have to guess, so both dispatch rows surface as ambiguous with a real
    instruction but no fabricated child (polylogue-y964)."""
    archive = _archive(tmp_path)
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            parent_session_id = archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="delegation-ambiguous-parent-v1",
                    title="Delegation ambiguous fixture",
                    messages=[
                        ParsedMessage(
                            provider_message_id="dispatch-a",
                            role=Role.ASSISTANT,
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TOOL_USE, tool_name="Task", tool_id="task-a", tool_input={}
                                )
                            ],
                        ),
                        ParsedMessage(
                            provider_message_id="dispatch-b",
                            role=Role.ASSISTANT,
                            blocks=[
                                ParsedContentBlock(
                                    type=BlockType.TOOL_USE, tool_name="Task", tool_id="task-b", tool_input={}
                                )
                            ],
                        ),
                    ],
                )
            )
            archive_db.write_parsed(
                ParsedSession(
                    source_name=Provider.CLAUDE_CODE,
                    provider_session_id="delegation-ambiguous-child-v1",
                    title="Delegation ambiguous child fixture",
                    messages=[ParsedMessage(provider_message_id="c1", role=Role.ASSISTANT, text="on it")],
                    parent_session_provider_id="delegation-ambiguous-parent-v1",
                    branch_type=BranchType.SUBAGENT,
                )
            )

        instruction_block_id = f"{parent_session_id}:dispatch-a:0"
        payload = await archive.resolve_ref(f"delegation:{instruction_block_id}")

        assert payload.resolved is True
        assert payload.payload is not None
        assert payload.payload["mapping_state"] == "ambiguous"
        assert payload.payload["child_session_id"] is None
        assert payload.payload["instruction_tool_use_block_id"] == instruction_block_id
        assert any("ambiguous" in caveat for caveat in payload.caveats)
    finally:
        await archive.close()


async def test_resolve_ref_returns_missing_payload_for_unknown_delegation_identity(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    try:
        block_missing = await archive.resolve_ref("delegation:claude-code-session:no-such-parent:dispatch:0")
        assert block_missing.resolved is False
        assert block_missing.kind == "delegation"
        assert block_missing.payload_kind == "missing"
        assert block_missing.payload is None

        edge_missing = await archive.resolve_ref(
            f"delegation:{delegation_edge_object_id('claude-code-session:no-parent', 'claude-code-session:no-child')}"
        )
        assert edge_missing.resolved is False
        assert edge_missing.payload_kind == "missing"
    finally:
        await archive.close()


async def test_query_units_returns_run_rows(tmp_path: Path) -> None:
    """``query_units()`` exposes runtime run projection rows through the facade."""
    from polylogue.surfaces.payloads import RunQueryRowPayload
    from tests.infra.storage_records import SessionBuilder

    archive = _archive(tmp_path)
    try:
        index_db = archive.config.archive_root / "index.db"
        (
            SessionBuilder(index_db, "facade-run-v1")
            .provider("codex")
            .git_repository_url("polylogue")
            .git_branch("feature/query-runs")
            .working_directories(["/realm/project/polylogue"])
            .title("Facade run query")
            .add_message(
                "m-run",
                role="assistant",
                text="Subagent finished the facade run query wiring.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "tool-run",
                        "name": "Task",
                        "tool_input": {
                            "subagent_type": "Explore",
                            "taskId": "task-run",
                            "child_session_id": "codex-session:facade-run-child",
                            "prompt": "Map facade run query wiring.",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "tool-run",
                        "text": "Subagent done: facade run query wired.\n4 passed in 0.31s",
                    },
                ],
            )
            .save()
        )

        _materialize_run_projection(index_db)

        envelope = await archive.query_units(
            "runs where session.repo:polylogue AND role:subagent AND status:completed AND agent:Explore"
        )

        assert envelope.unit == "run"
        [item] = envelope.items
        assert isinstance(item, RunQueryRowPayload)
        assert item.session_id == "codex-session:ext-facade-run-v1"
        assert item.role == "subagent"
        assert item.status == "completed"
        assert item.agent_ref == "agent:codex/Explore"
        assert item.parent_run_ref == "run:codex-session:ext-facade-run-v1"
        assert item.run_ref == "run:codex-session:ext-facade-run-v1:subagent:0:tool-run"
    finally:
        await archive.close()


async def test_export_otel_projects_query_unit_rows(tmp_path: Path) -> None:
    """``export_otel()`` projects bounded query-unit evidence into OTel-like JSON."""
    from polylogue.surfaces.payloads import OtelProjectionPayload
    from tests.infra.storage_records import SessionBuilder

    archive = _archive(tmp_path)
    try:
        index_db = archive.config.archive_root / "index.db"
        (
            SessionBuilder(index_db, "facade-otel-v1")
            .provider("codex")
            .git_repository_url("polylogue")
            .git_branch("feature/otel")
            .working_directories(["/realm/project/polylogue"])
            .title("Facade OTel projection")
            .add_message(
                "m-run",
                role="assistant",
                text="Subagent finished the OTel projection wiring.",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "tool-run",
                        "name": "Task",
                        "tool_input": {
                            "subagent_type": "Explore",
                            "taskId": "task-otel",
                            "child_session_id": "codex-session:facade-otel-child",
                            "prompt": "Map facade OTel projection wiring.",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "tool-run",
                        "text": "Subagent done: facade OTel projection wired.\n4 passed in 0.31s",
                    },
                ],
            )
            .save()
        )

        _materialize_run_projection(index_db)

        payload = await archive.export_otel(
            source_ref="session:codex-session:ext-facade-otel-v1",
            expressions=("runs where session.repo:polylogue AND role:subagent AND agent:Explore",),
        )

        assert isinstance(payload, OtelProjectionPayload)
        assert payload.mode == "otel-projection"
        assert payload.trace_count == 1
        assert payload.span_count == 1
        assert "session:codex-session:ext-facade-otel-v1" in payload.refs
        assert "run:codex-session:ext-facade-otel-v1:subagent:0:tool-run" in payload.refs
        assert any(ref.startswith("codex-session:ext-facade-otel-v1::") for ref in payload.refs)
        assert "context-snapshot:codex-session:ext-facade-otel-v1:subagent:0:tool-run:subagent_start" in payload.refs
        [span] = payload.spans
        assert span.attributes["polylogue.run.ref"] == "run:codex-session:ext-facade-otel-v1:subagent:0:tool-run"
        assert span.attributes["polylogue.run.cwd.redacted"] is True
    finally:
        await archive.close()


async def test_query_units_rejects_session_expression(tmp_path: Path) -> None:
    """``query_units()`` is only for terminal source expressions."""
    from polylogue.archive.query.expression import ExpressionCompileError

    archive = _archive(tmp_path)
    try:
        with pytest.raises(ExpressionCompileError):
            await archive.query_units("repo:polylogue")
    finally:
        await archive.close()


async def test_archive_tiers_api_reads_native_sessions(tmp_path: Path) -> None:
    """Archive API opt-in methods read index.db files directly."""
    from polylogue.archive.message.roles import Role
    from polylogue.archive.message.types import MessageType
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import (
        ArchiveSessionSearchHit,
        ArchiveSessionSummary,
        ArchiveStore,
    )
    from polylogue.storage.sqlite.archive_tiers.write import ArchiveSessionEnvelope

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-v1-1",
        title="API archive session",
        working_directories=["/realm/project/polylogue"],
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                message_type=MessageType.TOOL_USE,
                blocks=[
                    ParsedContentBlock(type=BlockType.TEXT, text="api archive needle"),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Read",
                        tool_id="api-tool-1",
                        tool_input={"file_path": "README.md"},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Bash",
                        tool_id="api-tool-2",
                        tool_input={"command": "pytest"},
                    ),
                ],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)
            archive_db.add_user_tags((session_id,), ("api-v1",))

        count = await archive.archive_count_sessions(
            origin="codex-session",
            excluded_origins=("chatgpt-export",),
            tags=("api-v1",),
            excluded_tags=("archived",),
            repo_names=("polylogue",),
            has_types=("tool_use",),
            has_tool_use=True,
            tool_terms=("read",),
            excluded_tool_terms=("write",),
            action_terms=("file_read",),
            excluded_action_terms=("file_write",),
            action_sequence=("file_read", "shell"),
            action_text_terms=("README.md",),
            referenced_paths=("README.md",),
            cwd_prefix="/realm/project",
            typed_only=True,
            message_type="tool_use",
            title="API",
            max_words=10,
        )
        summaries = await archive.archive_list_sessions(
            origin="codex-session",
            excluded_origins=("chatgpt-export",),
            tags=("api-v1",),
            excluded_tags=("archived",),
            repo_names=("polylogue",),
            has_types=("tool_use",),
            has_tool_use=True,
            tool_terms=("read",),
            excluded_tool_terms=("write",),
            action_terms=("file_read",),
            excluded_action_terms=("file_write",),
            action_sequence=("file_read", "shell"),
            action_text_terms=("README.md",),
            referenced_paths=("README.md",),
            cwd_prefix="/realm/project",
            typed_only=True,
            message_type="tool-use",
            title="API",
            max_words=10,
            limit=5,
        )
        hits = await archive.archive_search_sessions(
            "needle",
            origin="codex-session",
            excluded_origins=("chatgpt-export",),
            tags=("api-v1",),
            excluded_tags=("archived",),
            repo_names=("polylogue",),
            has_types=("tool_use",),
            has_tool_use=True,
            tool_terms=("read",),
            excluded_tool_terms=("write",),
            action_terms=("file_read",),
            excluded_action_terms=("file_write",),
            action_sequence=("file_read", "shell"),
            action_text_terms=("README.md",),
            referenced_paths=("README.md",),
            cwd_prefix="/realm/project",
            typed_only=True,
            message_type="tool-use",
            title="API",
            limit=5,
        )
        envelope = await archive.archive_get_session("codex-session:api-v1")
        normal_rows = await archive.list_sessions(origin="codex-session")
        query_rows = await archive.query_sessions(
            origin="codex-session",
            tag="api-v1",
            has_tool_use=True,
            typed_only=True,
            min_messages=1,
            query="needle",
            limit=5,
        )
        query_count = await archive.count_sessions(
            origin="codex-session",
            tag="api-v1",
            has_tool_use=True,
            typed_only=True,
            min_messages=1,
            query="needle",
        )
        facet_spec = SessionQuerySpec.from_params(
            {"origin": "codex-session", "tag": "api-v1", "filter_has_tool_use": True},
            strict=True,
        )
        normal_facets = await archive.facets(facet_spec)
        normal_session = await archive.get_session("codex-session:api-v1")
        normal_summary = await archive.get_session_summary("codex-session:api-v1")
        normal_stats = await archive.get_session_stats("codex-session:api-v1")
        aggregate_stats = await archive.stats()
        health = await archive.health_check()
        rebuilt = await archive.rebuild_insights()
        rebuilt_profile = await archive.get_session_profile_insight(session_id)
        rebuilt_status = await archive.get_session_insight_status()
        normal_search = await archive.search("needle")
        normal_envelope = await archive.search_envelope("needle", limit=1, origin="codex-session")
        unit_envelope = await archive.query_units("messages where text:needle", limit=5)
        normal_neighbors = await archive.neighbor_candidates(
            query="needle",
            provider=Provider.CODEX.value,
            limit=3,
        )
        paged_messages, total_messages = await archive.get_messages_paginated(
            "codex-session:api-v1",
            message_role=(Role.USER,),
            message_type="tool_use",
            limit=1,
            offset=0,
        )
        bulk_messages = await archive.bulk_get_messages(("codex-session:api-v1", "missing-session"))

        assert count == 1
        assert isinstance(summaries[0], ArchiveSessionSummary)
        assert summaries[0].session_id == session_id
        assert isinstance(hits[0], ArchiveSessionSearchHit)
        assert hits[0].session_id == session_id
        assert isinstance(envelope, ArchiveSessionEnvelope)
        assert envelope.session_id == session_id
        assert [str(row.id) for row in normal_rows] == [session_id]
        assert [str(row["id"]) for row in query_rows] == [session_id]
        assert query_rows[0]["origin"] == "codex-session"
        assert "provider" not in query_rows[0]
        assert query_rows[0]["message_count"] == 1
        assert query_count == 1
        assert normal_facets.scoped_to_query is True
        assert normal_facets.total_sessions == 1
        assert normal_facets.origins == {Origin.CODEX_SESSION.value: 1}
        assert normal_facets.tags == {"api-v1": 1}
        assert normal_facets.message_types == {"tool_use": 1}
        assert normal_facets.has_flags["has_tool_use"] == 1
        assert normal_session is not None
        assert str(normal_session.id) == session_id
        assert normal_session.origin == Origin.CODEX_SESSION
        assert "api archive needle" in (normal_session.messages.to_list()[0].text or "")
        assert normal_summary is not None
        assert str(normal_summary.id) == session_id
        assert normal_summary.message_count == 1
        assert normal_stats["messages"] == 1
        assert aggregate_stats.session_count == 1
        assert aggregate_stats.message_count == 1
        assert {check.name for check in health.checks} >= {"archive_index", "archive_index_rows"}
        assert "database" not in {check.name for check in health.checks}
        assert rebuilt.profiles == 1
        assert rebuilt.threads == 1
        assert rebuilt_profile is not None
        assert rebuilt_profile.inference is not None
        assert rebuilt_profile.inference.workflow_shape
        assert rebuilt_status.profile_row_count == 1
        assert rebuilt_status.profile_rows_ready is True
        assert rebuilt_status.thread_count == 1
        assert [hit.session_id for hit in normal_search.hits] == [session_id]
        assert normal_envelope.total == 1
        assert normal_envelope.retrieval_lane == "dialogue"
        assert [hit.session.id for hit in normal_envelope.hits] == [session_id]
        assert normal_envelope.hits[0].match.message_id == f"{session_id}:m1"
        assert unit_envelope.mode == "query-unit"
        assert unit_envelope.unit == "message"
        assert unit_envelope.total == 1
        assert unit_envelope.items[0].unit == "message"
        assert unit_envelope.items[0].session_id == session_id
        assert [candidate.session_id for candidate in normal_neighbors] == [session_id]
        assert normal_neighbors[0].summary.message_count == 1
        assert {reason.kind for reason in normal_neighbors[0].reasons} >= {"query_match", "content_similarity"}
        assert total_messages == 1
        assert [message.id for message in paged_messages] == [f"{session_id}:m1"]
        assert list(bulk_messages) == [session_id]
        assert [message.id for message in bulk_messages[session_id]] == [f"{session_id}:m1"]

        with ArchiveStore.open_existing(archive.config.archive_root, read_only=False) as archive_db:
            archive_db._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
            archive_db._conn.commit()

        with pytest.raises(DatabaseError):
            await archive.search("needle")
        assert await archive.rebuild_index() is True
        assert [hit.session_id for hit in (await archive.search("needle")).hits] == [session_id]
    finally:
        await archive.close()


async def test_archive_tiers_api_reads_session_topology(tmp_path: Path) -> None:
    """Topology facade helpers project parent/session rows."""
    from polylogue.archive.message.roles import Role
    from polylogue.archive.session.branch_type import BranchType
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive = _archive(tmp_path)
    root = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="topology-root",
        title="Topology Root",
        messages=[
            ParsedMessage(
                provider_message_id="root-message",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="root")],
            )
        ],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="topology-child",
        title="Topology Child",
        parent_session_provider_id="topology-root",
        branch_type=BranchType.CONTINUATION,
        messages=[
            ParsedMessage(
                provider_message_id="child-message",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="child")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            root_id = archive_db.write_parsed(root)
            child_id = archive_db.write_parsed(child)

        topology = await archive.get_session_topology(child_id)
        ancestors = await archive.get_ancestors(child_id)
        descendants = await archive.get_descendants(root_id)
        siblings = await archive.get_siblings(child_id)
        thread = await archive.get_thread(child_id)
        logical = await archive.get_logical_session(child_id)

        assert topology is not None
        assert topology.root_id == root_id
        assert topology.target_id == child_id
        assert [str(node.session_id) for node in topology.nodes] == [root_id, child_id]
        assert [(str(edge.parent_id), str(edge.child_id), edge.kind.value) for edge in topology.edges] == [
            (root_id, child_id, "continuation")
        ]
        assert [str(ref.session_id) for ref in ancestors] == [root_id]
        assert [str(ref.session_id) for ref in descendants] == [child_id]
        assert siblings == []
        assert [str(ref.session_id) for ref in thread] == [root_id, child_id]
        assert logical is not None
        assert logical.root_id == root_id
        assert [str(ref.session_id) for ref in logical.thread] == [root_id, child_id]
    finally:
        await archive.close()


async def test_archive_tiers_parse_file_writes_source_and_index_tiers(tmp_path: Path) -> None:
    """Public parse_file writes directly to source.db/index.db under active archive."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive = _archive(tmp_path)
    payload = {
        "sessionId": "gemini-v1-parse",
        "projectHash": "project-hash",
        "startTime": "2026-04-08T20:45:00.000Z",
        "lastUpdated": "2026-04-08T20:47:00.000Z",
        "kind": "chat",
        "summary": "Native parse",
        "messages": [
            {
                "id": "u1",
                "timestamp": "2026-04-08T20:45:01.000Z",
                "type": "user",
                "content": ["parse needle"],
            },
            {"id": "a1", "timestamp": "2026-04-08T20:45:02.000Z", "type": "gemini", "content": "response"},
        ],
    }
    source_path = tmp_path / "gemini-session.json"
    source_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        with ArchiveStore(archive.config.archive_root):
            pass

        result = await archive.parse_file(source_path, source_name=Provider.GEMINI_CLI.value)
        rows = await archive.list_sessions(origin="gemini-cli-session")
        search = await archive.search("needle")

        assert result.counts["sessions"] == 1
        assert result.counts["messages"] == 2
        assert result.changed_counts["sessions"] == 1
        observation = result.batch_observations[-1]
        assert observation["primary_ingest_store"] == "archive_file_set"
        assert observation["archive_write_mode"] == "archive"
        assert observation["archive_root"] == str(archive.config.archive_root)
        assert observation["archive_write_targets"] == ["source.db", "index.db"]
        assert observation["archive_source_rows"] == 1
        assert observation["archive_index_rows"] == 1
        assert [str(row.id) for row in rows] == ["gemini-cli-session:gemini-v1-parse"]
        assert [hit.session_id for hit in search.hits] == ["gemini-cli-session:gemini-v1-parse"]
        with ArchiveStore.open_existing(archive.config.archive_root) as archive_db:
            artifacts, total = archive_db.raw_artifacts_for_session("gemini-cli-session:gemini-v1-parse")
        assert total == 1
        assert artifacts[0]["source_path"] == str(source_path)
    finally:
        await archive.close()


async def test_archive_tiers_api_user_mutations_write_user_tier(tmp_path: Path) -> None:
    """User tag/metadata facade methods write ``user.db``."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.user_write import (
        assertion_id_for_session_metadata,
        assertion_id_for_session_tag,
        read_assertion_envelope,
    )

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-user-state",
        title="API archive user state",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="user state target")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)

        added = await archive.add_tag(session_id, "Review")
        duplicate = await archive.add_tag(session_id, "review")
        tags = await archive.list_tags(origin="codex-session")
        removed = await archive.remove_tag(session_id, "REVIEW")
        missing_remove = await archive.remove_tag(session_id, "review")
        bulk = await archive.bulk_tag_sessions([session_id, "missing-session"], ["bulk-a", "bulk-b"])
        bulk_duplicate = await archive.bulk_tag_sessions([session_id, "missing-session"], ["bulk-a", "bulk-b"])
        set_result = await archive.set_metadata(session_id, "priority", "high")
        unchanged = await archive.set_metadata(session_id, "priority", "high")
        bool_changed = await archive.update_metadata(session_id, "status", "open")
        metadata = await archive.get_metadata(session_id)
        deleted = await archive.delete_metadata(session_id, "status")
        missing_delete = await archive.delete_metadata(session_id, "status")

        assert added.outcome == "added"
        assert duplicate.outcome == "no_op"
        assert tags == {"review": 1}
        assert removed.outcome == "removed"
        assert missing_remove.outcome == "not_present"
        assert bulk.affected_count == 1
        assert bulk.skipped_count == 1
        assert bulk_duplicate.affected_count == 0
        assert bulk_duplicate.skipped_count == 2
        assert set_result.outcome == "set"
        assert unchanged.outcome == "unchanged"
        assert bool_changed is True
        assert metadata == {"priority": "high", "status": "open"}
        assert deleted.outcome == "deleted"
        assert missing_delete.outcome == "not_found"

        with sqlite3.connect(tmp_path / "user.db") as conn:
            review_assertion = read_assertion_envelope(
                conn,
                assertion_id_for_session_tag(session_id, "review", "user"),
            )
            bulk_a_assertion = read_assertion_envelope(
                conn,
                assertion_id_for_session_tag(session_id, "bulk-a", "user"),
            )
            priority_assertion = read_assertion_envelope(
                conn,
                assertion_id_for_session_metadata(session_id, "priority"),
            )
            status_assertion = read_assertion_envelope(
                conn,
                assertion_id_for_session_metadata(session_id, "status"),
            )
        assert review_assertion is not None
        assert review_assertion.status == "deleted"
        assert bulk_a_assertion is not None
        assert bulk_a_assertion.status == "active"
        assert priority_assertion is not None
        assert priority_assertion.value == "high"
        assert priority_assertion.status == "active"
        assert status_assertion is not None
        assert status_assertion.status == "deleted"
    finally:
        await archive.close()


async def test_archive_tiers_api_tag_rollups_read_index_and_user_tiers(tmp_path: Path) -> None:
    """Session tag rollups aggregate index and user tags."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.archive import SessionTagRollupQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import upsert_session_tag

    archive = _archive(tmp_path)
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-rollup-v1-a",
        title="Rollup A",
        updated_at="2026-02-02T02:40:00Z",
        git_repository_url="https://example.test/polylogue.git",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="rollup one")],
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-rollup-v1-b",
        title="Rollup B",
        updated_at="2026-02-02T02:41:00Z",
        git_repository_url="https://example.test/polylogue.git",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="rollup two")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            first_id = archive_db.write_parsed(first)
            second_id = archive_db.write_parsed(second)
            assert archive_db.add_user_tags((first_id,), ("focus",)) == 1
        with sqlite3.connect(tmp_path / "index.db") as conn:
            upsert_session_tag(
                conn, session_id=second_id, tag="focus", tag_source="auto", method="test", confidence=0.8
            )

        rollups = await archive.list_session_tag_rollup_insights(
            SessionTagRollupQuery(provider=Provider.CODEX.value, query="foc", limit=10)
        )

        assert len(rollups) == 1
        rollup = rollups[0]
        assert rollup.tag == "focus"
        assert rollup.session_count == 2
        assert rollup.logical_session_count == 2
        assert rollup.explicit_count == 1
        assert rollup.auto_count == 1
        assert rollup.origin_breakdown == {Origin.CODEX_SESSION.value: 2}
        assert rollup.provider_breakdown == {Origin.CODEX_SESSION.value: 2}
        assert rollup.repo_breakdown == {"https://example.test/polylogue.git": 2}
        assert rollup.provenance.source_updated_at == "2026-02-02T02:41:00Z"
    finally:
        await archive.close()


async def test_archive_tiers_api_archive_coverage_reads_index_tier(tmp_path: Path) -> None:
    """Archive coverage aggregates sessions and profiles."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.archive import ArchiveCoverageInsightQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import upsert_session_profile_costs, upsert_session_work_event

    archive = _archive(tmp_path)
    codex = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-coverage-v1-codex",
        title="Coverage Codex",
        updated_at="2026-02-02T02:40:00Z",
        git_repository_url="https://example.test/polylogue.git",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="one two three",
                blocks=[
                    ParsedContentBlock(type=BlockType.TEXT, text="one two three"),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Read",
                        tool_id="tool-1",
                        tool_input={"file_path": "README.md"},
                    ),
                ],
            ),
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                text="four five",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="four five")],
            ),
        ],
    )
    chatgpt = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="api-coverage-v1-chatgpt",
        title="Coverage ChatGPT",
        updated_at="2026-02-03T02:40:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                text="plain",
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="plain")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            codex_id = archive_db.write_parsed(codex)
            archive_db.write_parsed(chatgpt)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            upsert_session_profile_costs(
                conn,
                codex_id,
                cost_usd=1.25,
                cost_provenance="exact",
                priced_with="fixture",
                priced_at_ms=1_770_000_000_000,
            )
            conn.execute("UPDATE session_profiles SET duration_ms = 60000 WHERE session_id = ?", (codex_id,))
            upsert_session_work_event(
                conn,
                session_id=codex_id,
                position=0,
                work_event_type="implementation",
                summary="Implemented coverage",
                confidence=0.8,
            )
            conn.commit()

        provider_rows = await archive.list_archive_coverage_insights(
            ArchiveCoverageInsightQuery(group_by="provider", provider=Provider.CODEX.value)
        )
        day_rows = await archive.list_archive_coverage_insights(
            ArchiveCoverageInsightQuery(group_by="day", provider=Provider.CODEX.value, limit=10)
        )
        week_rows = await archive.list_archive_coverage_insights(
            ArchiveCoverageInsightQuery(group_by="week", provider=Provider.CODEX.value, limit=10)
        )

        assert len(provider_rows) == 1
        provider = provider_rows[0]
        assert provider.source_name == Provider.CODEX.value
        assert provider.session_count == 1
        assert provider.message_count == 2
        assert provider.user_message_count == 1
        assert provider.assistant_message_count == 1
        assert provider.avg_messages_per_session == 2.0
        assert provider.tool_use_count == 1
        assert provider.total_sessions_with_tools == 1
        assert provider.tool_use_percentage == 100.0

        assert len(day_rows) == 1
        day = day_rows[0]
        assert day.group_by == "day"
        assert day.bucket == "2026-02-02"
        assert day.session_count == 1
        assert day.logical_session_count == 1
        assert day.message_count == 2
        assert day.total_cost_usd == 1.25
        assert day.total_duration_ms == 60000
        assert day.total_wall_duration_ms == 60000
        assert day.total_words == 5
        assert day.work_event_breakdown == {"implementation": 1}
        assert day.repos_active == ("polylogue",)
        assert day.origin_breakdown == {Origin.CODEX_SESSION.value: 1}
        assert day.provider_breakdown == {Origin.CODEX_SESSION.value: 1}
        assert day.provenance is not None
        assert day.provenance.source_updated_at == "2026-02-02T02:40:00Z"

        assert len(week_rows) == 1
        assert week_rows[0].group_by == "week"
        assert week_rows[0].session_count == 1
        assert week_rows[0].origin_breakdown == {Origin.CODEX_SESSION.value: 1}
        assert week_rows[0].provider_breakdown == {Origin.CODEX_SESSION.value: 1}
    finally:
        await archive.close()


async def test_archive_tiers_api_tool_usage_reads_index_actions(tmp_path: Path) -> None:
    """Tool-usage API reads archive ``actions`` instead of retired action tables."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.tool_usage import ToolUsageInsightQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive = _archive(tmp_path)
    codex = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-tool-v1-codex",
        title="Tool usage",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[
                    ParsedContentBlock(type=BlockType.TEXT, text="inspect file"),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Read",
                        tool_id="tool-1",
                        tool_input={"file_path": "README.md"},
                    ),
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        text="read output",
                        tool_id="tool-1",
                    ),
                ],
            )
        ],
    )
    chatgpt = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="api-tool-v1-chatgpt",
        title="No tools",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="plain chat")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            archive_db.write_parsed(codex)
            archive_db.write_parsed(chatgpt)

        [insight] = await archive.list_tool_usage_insights(
            ToolUsageInsightQuery(provider=Provider.CODEX.value, tool="read")
        )

        assert insight.total_call_count == 1
        assert insight.total_distinct_tools == 1
        assert insight.providers_with_data == 1
        assert insight.providers_without_data == 1
        assert insight.has_coverage_gaps is True
        assert len(insight.entries) == 1
        entry = insight.entries[0]
        assert entry.source_name == Provider.CODEX.value
        assert entry.normalized_tool_name == "read"
        assert entry.action_kind == "file_read"
        assert entry.call_count == 1
        assert entry.session_count == 1
        assert entry.message_count == 1
        assert entry.distinct_tool_ids == 1
        assert entry.affected_path_calls == 1
        assert entry.output_text_calls == 1
        coverage = {item.source_name: item for item in insight.provider_coverage}
        assert coverage[Provider.CODEX.value].data_available is True
        assert coverage[Provider.CODEX.value].action_count == 1
        assert coverage[Provider.CHATGPT.value].data_available is False
        assert coverage[Provider.CHATGPT.value].session_count == 1
    finally:
        await archive.close()


async def test_archive_tiers_api_delete_uses_index_tier_and_keeps_user_overlay(tmp_path: Path) -> None:
    """Archive facade delete removes index rows without dropping ``user.db`` overlays."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-delete-v1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="delete through v1")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)
        await archive.add_tag(session_id, "keep-user-state")

        deleted = await archive.delete_session_safe(session_id)
        second = await archive.delete_session_safe(session_id)
        bool_deleted = await archive.delete_session(session_id)
        missing = await archive.get_session(session_id)

        assert deleted.outcome == "deleted"
        assert second.outcome == "not_found"
        assert bool_deleted is False
        assert missing is None

        with sqlite3.connect(tmp_path / "index.db") as index_conn:
            session_count = index_conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        with sqlite3.connect(tmp_path / "user.db") as user_conn:
            tag_count = user_conn.execute(
                "SELECT COUNT(*) FROM assertions WHERE kind = 'tag' AND COALESCE(status, '') IN ('', 'active')"
            ).fetchone()[0]
        assert session_count == 0
        assert tag_count == 1
    finally:
        await archive.close()


async def test_archive_tiers_api_raw_artifacts_read_source_tier(tmp_path: Path) -> None:
    """Raw artifact facade reads ``source.db`` rows."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive = _archive(tmp_path)
    payload = b'{"session":"raw-artifact-v1"}'
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-raw-artifact-v1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="raw artifact target")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            raw_id, session_id = archive_db.write_raw_and_parsed(
                session,
                payload=payload,
                source_path="/tmp/raw-artifact-v1.jsonl",
                acquired_at_ms=1_770_000_000_000,
            )

        artifacts, total = await archive.get_raw_artifacts_for_session(session_id)
        missing_artifacts, missing_total = await archive.get_raw_artifacts_for_session("missing-session")

        assert total == 1
        assert artifacts == [
            {
                "raw_id": raw_id,
                "source_name": Provider.CODEX.value,
                "source_path": "/tmp/raw-artifact-v1.jsonl",
                "blob_size": len(payload),
                "acquired_at": "2026-02-02T02:40:00Z",
                "parsed_at": None,
                "validation_status": None,
            }
        ]
        assert missing_artifacts == []
        assert missing_total == 0
    finally:
        await archive.close()


async def test_archive_tiers_api_timeline_insights_read_index_tier(tmp_path: Path) -> None:
    """Work-event and phase insight facade methods read rows."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.archive import SessionPhaseInsightQuery, SessionWorkEventInsightQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import (
        upsert_insight_materialization,
        upsert_session_phase,
        upsert_session_work_event,
    )

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-timeline-v1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="timeline target")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            upsert_insight_materialization(
                conn,
                insight_type="work_events",
                session_id=session_id,
                materializer_version=7,
                materialized_at_ms=1_770_000_000_000,
                source_sort_key_ms=1_770_000_060_000,
                input_row_count=1,
            )
            upsert_session_work_event(
                conn,
                session_id=session_id,
                position=0,
                work_event_type="implementation",
                summary="Implemented timeline reads",
                confidence=0.82,
                start_index=0,
                end_index=1,
                started_at_ms=1_770_000_060_000,
                ended_at_ms=1_770_000_120_000,
                duration_ms=60_000,
                file_paths=("polylogue/api/insights.py",),
                tools_used=("apply_patch",),
                evidence={"canonical_session_date": "2026-02-02"},
            )
            upsert_insight_materialization(
                conn,
                insight_type="phases",
                session_id=session_id,
                materializer_version=8,
                materialized_at_ms=1_770_000_010_000,
                source_sort_key_ms=1_770_000_060_000,
                input_row_count=1,
            )
            upsert_session_phase(
                conn,
                session_id=session_id,
                position=0,
                start_index=0,
                end_index=1,
                started_at_ms=1_770_000_060_000,
                ended_at_ms=1_770_000_120_000,
                duration_ms=60_000,
                tool_counts={"apply_patch": 1},
                word_count=2,
                evidence={"canonical_session_date": "2026-02-02"},
                inference={"evidence": ("edited archive path",)},
            )

        events = await archive.get_session_work_event_insights(session_id)
        filtered_events = await archive.list_session_work_event_insights(
            SessionWorkEventInsightQuery(
                provider=Provider.CODEX.value,
                heuristic_label="implementation",
                since="2026-02-02T02:40:30Z",
                limit=10,
            )
        )
        phases = await archive.get_session_phase_insights(session_id)
        filtered_phases = await archive.list_session_phase_insights(
            SessionPhaseInsightQuery(provider=Provider.CODEX.value, limit=10)
        )

        assert len(events) == 1
        assert events == filtered_events
        assert events[0].session_id == session_id
        assert events[0].source_name == Provider.CODEX.value
        assert events[0].provenance.materializer_version == 7
        assert events[0].inference.heuristic_label == "implementation"
        assert events[0].evidence.file_paths == ("polylogue/api/insights.py",)
        assert len(phases) == 1
        assert phases == filtered_phases
        assert phases[0].session_id == session_id
        assert phases[0].source_name == Provider.CODEX.value
        assert phases[0].provenance.materializer_version == 8
        assert phases[0].evidence.tool_counts == {"apply_patch": 1}
        assert phases[0].semantic_tier == "evidence"
        assert phases[0].inference is None
        assert phases[0].inference_provenance is None
    finally:
        await archive.close()


async def test_archive_tiers_api_threads_read_index_tier(tmp_path: Path) -> None:
    """Thread facade methods project thread rows."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.archive.session.branch_type import BranchType
    from polylogue.core.enums import BlockType
    from polylogue.insights.archive import ThreadInsightQuery
    from polylogue.insights.audit import InsightRigorAuditQuery
    from polylogue.insights.export_bundles import InsightExportBundleRequest
    from polylogue.insights.readiness import InsightReadinessQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import upsert_insight_materialization

    archive = _archive(tmp_path)
    parent = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="api-thread-parent",
        title="Parent planning",
        created_at="2026-02-02T02:40:00Z",
        updated_at="2026-02-02T02:41:00Z",
        git_repository_url="https://example.test/polylogue.git",
        git_branch="archive",
        messages=[
            ParsedMessage(
                provider_message_id="p1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="thread parent")],
            )
        ],
    )
    child = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="api-thread-child",
        parent_session_provider_id="api-thread-parent",
        branch_type=BranchType.SIDECHAIN,
        title="Child implementation",
        created_at="2026-02-02T02:42:00Z",
        updated_at="2026-02-02T02:45:00Z",
        git_repository_url="https://example.test/polylogue.git",
        git_branch="archive",
        messages=[
            ParsedMessage(
                provider_message_id="c1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="thread child")],
            ),
            ParsedMessage(
                provider_message_id="c2",
                role=Role.ASSISTANT,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="thread reply")],
            ),
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            parent_id = archive_db.write_parsed(parent)
            child_id = archive_db.write_parsed(child)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            upsert_insight_materialization(
                conn,
                insight_type="thread",
                session_id=parent_id,
                materializer_version=11,
                materialized_at_ms=1_770_000_400_000,
                source_sort_key_ms=1_770_000_300_000,
                input_row_count=2,
            )
            conn.execute(
                """
                INSERT INTO session_profiles (
                    session_id, workflow_shape, workflow_shape_confidence,
                    terminal_state, terminal_state_confidence, duration_ms,
                    substantive_count, work_event_count, phase_count,
                    evidence_payload_json
                ) VALUES (?, 'agentic_loop', 0.91, 'question_left', 0.88, 180000,
                          2, 0, 0, ?)
                """,
                (
                    parent_id,
                    json.dumps(
                        {
                            "first_message_at": "2026-02-02T02:40:00Z",
                            "last_message_at": "2026-02-02T02:41:00Z",
                            "canonical_session_date": "2026-02-02",
                            "repo_paths": ["https://example.test/polylogue.git"],
                            "cwd_paths": ["/realm/project/polylogue"],
                            "file_paths_touched": ["/realm/project/polylogue/polylogue/api/archive.py"],
                            "branch_names": ["archive"],
                            "tags": ["archive"],
                        }
                    ),
                ),
            )

        thread = await archive.get_thread_insight(parent_id)
        listed = await archive.list_thread_insights(ThreadInsightQuery(query="polylogue", limit=10))
        tree = await archive.get_session_tree(child_id)
        brief = await archive.resume_brief(child_id, related_limit=5)
        candidates = await archive.find_resume_candidates(
            repo_path="/realm/project/polylogue",
            cwd="/realm/project/polylogue/polylogue/api",
            recent_files=("/realm/project/polylogue/polylogue/api/archive.py",),
            limit=5,
        )
        readiness = await archive.insight_readiness_report(
            InsightReadinessQuery(insights=("session_profiles", "threads"))
        )
        rigor = await archive.insight_rigor_audit(
            InsightRigorAuditQuery(insights=("session_profiles", "threads"), sample_limit=10)
        )
        export_target = tmp_path / "exports" / "v1-insights"
        export_result = await archive.export_insight_bundle(
            InsightExportBundleRequest(
                output_path=export_target,
                insights=("session_profiles", "threads"),
                include_readme=False,
            )
        )
        missing = await archive.get_thread_insight(child_id)

        assert thread is not None
        assert listed == [thread]
        assert [session.id for session in tree] == [parent_id, child_id]
        assert tree[0].parent_id is None
        assert tree[1].parent_id == parent_id
        assert tree[1].branch_type == BranchType.SIDECHAIN
        assert [len(session.messages) for session in tree] == [1, 2]
        assert brief is not None
        assert brief.session_id == child_id
        assert [related.session_id for related in brief.related_sessions] == [parent_id]
        assert brief.inferences.thread is not None
        assert brief.inferences.thread.thread_id == parent_id
        assert brief.provenance.cited_thread_id == parent_id
        assert candidates
        assert candidates[0].logical_session_id == parent_id
        assert candidates[0].terminal_state == "question_left"
        assert candidates[0].workflow_shape == "agentic_loop"
        assert candidates[0].file_overlap == ("/realm/project/polylogue/polylogue/api/archive.py",)
        readiness_by_name = {entry.insight_name: entry for entry in readiness.insights}
        assert readiness.total_sessions == 2
        assert readiness_by_name["session_profiles"].verdict == "partial"
        assert readiness_by_name["session_profiles"].missing_count == 1
        assert readiness_by_name["threads"].verdict == "stale"
        assert readiness_by_name["threads"].row_count == 1
        assert readiness_by_name["threads"].expected_row_count == 1
        assert readiness_by_name["threads"].stale_count == 1
        assert readiness_by_name["threads"].ready_flags == {"threads_ready": False}
        assert "threads_ready=False" in readiness_by_name["threads"].evidence
        rigor_by_name = {entry.insight_name: entry for entry in rigor.entries}
        assert rigor.sample_limit == 10
        assert rigor_by_name["session_profiles"].sample_size == 1
        assert rigor_by_name["session_profiles"].evidence_count == 1
        assert rigor_by_name["session_profiles"].inference_count == 1
        assert rigor_by_name["threads"].sample_size == 1
        assert rigor_by_name["threads"].evidence_count == 1
        assert rigor_by_name["threads"].has_evidence_payload is True
        assert rigor_by_name["threads"].version_targets
        manifest = json.loads(export_result.manifest_path.read_text(encoding="utf-8"))
        coverage = json.loads(export_result.coverage_path.read_text(encoding="utf-8"))
        exported_threads = [
            json.loads(line)
            for line in (export_target / "insights" / "threads.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert manifest["query"]["insights"] == ["session_profiles", "threads"]
        assert {entry["insight_name"] for entry in manifest["insights"]} == {"session_profiles", "threads"}
        assert coverage["total_sessions"] == 2
        # The threads insight is materialized but stale (asserted above);
        # the export bundle withholds stale/incompatible/missing insights
        # (#1743 readiness taxonomy) rather than shipping divergent rows, so
        # threads.jsonl is empty and the manifest records the withholding.
        assert exported_threads == []
        threads_summary = next(entry for entry in manifest["insights"] if entry["insight_name"] == "threads")
        assert threads_summary["readiness_verdict"] == "stale"
        assert threads_summary["row_count"] == 0
        assert any("withheld" in error for error in threads_summary["errors"])
        assert (export_target / "schemas" / "threads.schema.json").exists()
        assert not (export_target / "README.md").exists()
        assert missing is None
        assert thread.thread_id == parent_id
        assert thread.root_id == parent_id
        assert thread.dominant_repo == "https://example.test/polylogue.git"
        assert thread.provenance.materializer_version == 11
        assert thread.thread.session_ids == (parent_id, child_id)
        assert thread.thread.session_count == 2
        assert thread.thread.depth == 1
        assert thread.thread.branch_count == 1
        assert thread.thread.total_messages == 3
        assert thread.thread.origin_breakdown == {Origin.CLAUDE_CODE_SESSION.value: 2}
        assert thread.thread.provider_breakdown == {Origin.CLAUDE_CODE_SESSION.value: 2}
        assert thread.thread.member_evidence[1].parent_id == parent_id
        assert thread.thread.member_evidence[1].role == "parent_continuation"
        assert "parent_session_id" in thread.thread.member_evidence[1].support_signals
    finally:
        await archive.close()


async def test_archive_tiers_api_session_costs_read_index_tier(tmp_path: Path) -> None:
    """Session cost insight facade reads profile cost columns."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.archive import CostRollupInsightQuery, SessionCostInsightQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import (
        upsert_insight_materialization,
        upsert_session_profile_costs,
    )

    archive = _archive(tmp_path)
    priced_session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-cost-v1-priced",
        title="Priced archive cost",
        created_at="2026-02-02T02:40:00Z",
        updated_at="2026-02-02T02:45:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="priced cost")],
            )
        ],
    )
    unpriced_session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="api-cost-v1-unpriced",
        title="Unpriced archive cost",
        created_at="2026-02-01T02:40:00Z",
        updated_at="2026-02-01T02:45:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="unpriced cost")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            priced_id = archive_db.write_parsed(priced_session)
            archive_db.write_parsed(unpriced_session)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            upsert_session_profile_costs(
                conn,
                priced_id,
                cost_usd=1.25,
                cost_credits=12.5,
                cost_is_estimated=False,
                cost_provenance="priced",
                priced_with="voyage-cost-v1-test",
                priced_at_ms=1_770_000_300_000,
            )
            upsert_insight_materialization(
                conn,
                insight_type="session_profile",
                session_id=priced_id,
                materializer_version=9,
                materialized_at_ms=1_770_000_300_000,
                source_sort_key_ms=1_770_000_300_000,
                input_row_count=1,
            )

        costs = await archive.list_session_cost_insights(
            SessionCostInsightQuery(provider=Provider.CODEX.value, status="priced", limit=10)
        )
        unavailable = await archive.list_session_cost_insights(
            SessionCostInsightQuery(provider=Provider.CHATGPT.value, status="unavailable", limit=10)
        )
        model_filtered = await archive.list_session_cost_insights(SessionCostInsightQuery(model="claude-sonnet-4-5"))
        rollups = await archive.list_cost_rollup_insights(CostRollupInsightQuery(provider=Provider.CODEX.value))
        all_rollups = await archive.list_cost_rollup_insights(CostRollupInsightQuery(limit=10))
        model_rollups = await archive.list_cost_rollup_insights(CostRollupInsightQuery(model="claude-sonnet-4-5"))

        assert len(costs) == 1
        assert costs[0].session_id == priced_id
        assert costs[0].source_name == Provider.CODEX.value
        assert costs[0].title == "Priced archive cost"
        assert costs[0].estimate.status == "priced"
        assert costs[0].estimate.total_usd == 1.25
        assert costs[0].estimate.basis.catalog_priced_usd == 1.25
        assert costs[0].provenance.materializer_version == 9
        assert len(unavailable) == 1
        assert unavailable[0].estimate.status == "unavailable"
        # The facade enriches the degraded storage read (cost_enrichment.py): the
        # internal ``archive_profile_no_cost`` placeholder is replaced by the public
        # missing-reason taxonomy re-derived from the session — here a
        # message-bearing session with no token usage, matching the canonical
        # facade contract pinned by ``test_insights_costs_json``.
        assert unavailable[0].estimate.missing_reasons == ("missing_token_usage",)
        assert model_filtered == []
        assert len(rollups) == 1
        assert rollups[0].source_name == Provider.CODEX.value
        assert rollups[0].session_count == 1
        assert rollups[0].priced_session_count == 1
        assert rollups[0].unavailable_session_count == 0
        assert rollups[0].status_counts == {"priced": 1}
        assert rollups[0].total_usd == 1.25
        assert rollups[0].basis.catalog_priced_usd == 1.25
        assert rollups[0].confidence == 0.9
        assert {rollup.source_name for rollup in all_rollups} == {Provider.CODEX.value, Provider.CHATGPT.value}
        unavailable_rollup = next(rollup for rollup in all_rollups if rollup.source_name == Provider.CHATGPT.value)
        assert unavailable_rollup.unavailable_session_count == 1
        assert unavailable_rollup.unavailable_reason_counts == {"no_tokens": 1}
        assert model_rollups == []
    finally:
        await archive.close()


async def test_archive_tiers_api_latency_profiles_read_index_tier(tmp_path: Path) -> None:
    """Latency profile facade methods project message timings."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.archive import SessionLatencyProfileInsightQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import upsert_insight_materialization

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-latency-v1",
        title="Latency v1",
        updated_at="2026-02-02T02:45:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                occurred_at_ms=1_770_000_000_000,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="first prompt")],
            ),
            ParsedMessage(
                provider_message_id="m2",
                role=Role.ASSISTANT,
                occurred_at_ms=1_770_000_060_000,
                blocks=[
                    ParsedContentBlock(type=BlockType.TEXT, text="using tool"),
                    ParsedContentBlock(
                        type=BlockType.TOOL_USE,
                        tool_name="Read",
                        tool_id="tool-1",
                        tool_input={"file_path": "README.md"},
                    ),
                ],
            ),
            ParsedMessage(
                provider_message_id="m3",
                role=Role.USER,
                occurred_at_ms=1_770_000_180_000,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="follow up")],
            ),
            ParsedMessage(
                provider_message_id="m4",
                role=Role.ASSISTANT,
                occurred_at_ms=1_770_000_300_000,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="answer")],
            ),
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            upsert_insight_materialization(
                conn,
                insight_type="latency",
                session_id=session_id,
                materializer_version=12,
                materialized_at_ms=1_770_000_360_000,
                source_sort_key_ms=1_770_000_300_000,
                input_row_count=4,
            )

        profile = await archive.get_session_latency_profile_insight(session_id)
        listed = await archive.list_session_latency_profile_insights(
            SessionLatencyProfileInsightQuery(provider=Provider.CODEX.value, limit=10)
        )
        only_stuck = await archive.list_session_latency_profile_insights(
            SessionLatencyProfileInsightQuery(provider=Provider.CODEX.value, only_stuck=True, limit=10)
        )
        stuck = await archive.find_stuck_session_latency_profile_insights(
            SessionLatencyProfileInsightQuery(provider=Provider.CODEX.value, limit=10)
        )

        assert profile is not None
        assert listed == [profile]
        assert only_stuck == []
        assert stuck == []
        assert profile.session_id == session_id
        assert profile.source_name == Provider.CODEX.value
        assert profile.title == "Latency v1"
        assert profile.provenance.materializer_version == 12
        assert profile.latency.median_agent_response_ms == 90000
        assert profile.latency.median_user_response_ms == 120000
        assert profile.latency.median_tool_call_ms == 0
        assert profile.latency.stuck_tool_count == 0
        assert profile.latency.tool_call_count_by_category == {"file_read": 1}
    finally:
        await archive.close()


async def test_archive_tiers_api_archive_debt_reads_archive_consistency(tmp_path: Path) -> None:
    """Archive debt API reports consistency debt."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.archive import ArchiveDebtInsightQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-debt-v1",
        title="Debt v1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="debt target")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)
            assert archive_db.add_user_tags((session_id,), ("valid",)) == 1
            assert session_id
        with sqlite3.connect(tmp_path / "user.db") as conn:
            upsert_assertion(
                conn,
                assertion_id="assertion-orphan-tag",
                target_ref="session:missing-session",
                kind=AssertionKind.TAG,
                key="orphan",
                value={"tag_source": "user"},
                status="active",
            )
            conn.commit()

        debts = await archive.list_archive_debt_insights(ArchiveDebtInsightQuery(only_actionable=True, limit=20))
        by_name = {debt.debt_name: debt for debt in debts}

        assert by_name["archive_session_profile_rows"].category == "derived_repair"
        assert by_name["archive_session_profile_rows"].issue_count == 1
        assert by_name["archive_insight_materialization"].issue_count >= 1
        assert by_name["archive_user_overlay_orphans"].category == "archive_cleanup"
        assert by_name["archive_user_overlay_orphans"].issue_count == 1
        assert all(not debt.healthy for debt in debts)

        cleanup_debts = await archive.list_archive_debt_insights(
            ArchiveDebtInsightQuery(category="archive_cleanup", only_actionable=True, limit=20)
        )
        assert [debt.debt_name for debt in cleanup_debts] == ["archive_user_overlay_orphans"]
    finally:
        await archive.close()


async def test_archive_tiers_api_session_profiles_read_index_tier(tmp_path: Path) -> None:
    """Session profile facade methods read profile rows."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.archive import SessionProfileInsightQuery
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import upsert_insight_materialization

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-profile-v1",
        title="Native profile",
        created_at="2026-02-02T02:40:00Z",
        updated_at="2026-02-02T02:45:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="profile target")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(
                """
                INSERT INTO session_profiles (
                    session_id, workflow_shape, workflow_shape_confidence,
                    terminal_state, terminal_state_confidence, total_duration_ms,
                    substantive_count, attachment_count, work_event_count,
                    phase_count, tool_calls_per_minute, total_cost_usd,
                    cost_is_estimated, cost_provenance, evidence_payload_json,
                    inference_payload_json
                ) VALUES (?, 'implementation', 0.82, 'completed', 0.91, 120000,
                          1, 0, 2, 1, 3.5, 1.25, 0, 'priced', ?, ?)
                """,
                (
                    session_id,
                    json.dumps(
                        {
                            "canonical_session_date": "2026-02-02",
                            "total_duration_ms": 120000,
                            "total_cost_usd": 1.25,
                            "workflow_shape": "implementation",
                            "workflow_shape_confidence": 0.82,
                            "terminal_state": "completed",
                            "terminal_state_confidence": 0.91,
                        }
                    ),
                    json.dumps(
                        {
                            "work_event_count": 2,
                            "phase_count": 1,
                            "workflow_shape": "implementation",
                        }
                    ),
                ),
            )
            upsert_insight_materialization(
                conn,
                insight_type="session_profile",
                session_id=session_id,
                materializer_version=10,
                materialized_at_ms=1_770_000_300_000,
                source_sort_key_ms=1_770_000_300_000,
                input_row_count=1,
            )

        merged = await archive.get_session_profile_insight(session_id)
        evidence_only = await archive.get_session_profile_insight(session_id, tier="evidence")
        inference_only = await archive.get_session_profile_insight(session_id, tier="inference")
        listed = await archive.list_session_profile_insights(
            SessionProfileInsightQuery(
                provider=Provider.CODEX.value,
                workflow_shape="implementation",
                terminal_state="completed",
                tier="merged",
                limit=10,
            )
        )
        missing = await archive.get_session_profile_insight("missing-session")

        assert merged is not None
        assert listed == [merged]
        assert merged.session_id == session_id
        assert merged.source_name == Provider.CODEX.value
        assert merged.title == "Native profile"
        assert merged.provenance.materializer_version == 10
        assert merged.evidence is not None
        assert merged.evidence.total_duration_ms == 120000
        assert merged.evidence.canonical_session_date == "2026-02-02"
        assert merged.inference is not None
        assert merged.inference.work_event_count == 2
        assert merged.inference.phase_count == 1
        assert merged.inference.workflow_shape == "implementation"
        assert merged.inference.terminal_state == "completed"
        assert merged.enrichment is None
        assert evidence_only is not None
        assert evidence_only.evidence is not None
        assert evidence_only.inference is None
        assert inference_only is not None
        assert inference_only.evidence is None
        assert inference_only.inference is not None
        assert missing is None
    finally:
        await archive.close()


async def test_archive_tiers_api_session_insight_status_reads_index_tier(tmp_path: Path) -> None:
    """Session insight status projects readiness."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import upsert_insight_materialization

    archive = _archive(tmp_path)
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-status-v1-a",
        title="Status A",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="status target")],
            )
        ],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-status-v1-b",
        title="Status B",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="missing profile")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            first_id = archive_db.write_parsed(first)
            second_id = archive_db.write_parsed(second)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(
                """
                INSERT INTO session_profiles (
                    session_id, work_event_count, phase_count
                ) VALUES (?, 1, 1)
                """,
                (first_id,),
            )
            conn.execute(
                """
                INSERT INTO session_work_events (
                    session_id, position, work_event_type, summary, confidence,
                    start_index, end_index
                ) VALUES (?, 0, 'implementation', 'built it', 0.8, 0, 0)
                """,
                (first_id,),
            )
            conn.execute(
                """
                INSERT INTO session_phases (
                    session_id, position, start_index, end_index
                ) VALUES (?, 0, 0, 0)
                """,
                (first_id,),
            )
            conn.execute(
                "UPDATE threads SET materializer_version = ?, materialized_at = ?",
                (SESSION_INSIGHT_MATERIALIZER_VERSION, "2026-02-03T00:05:00Z"),
            )
            for insight_type in ("work_events", "phases", "thread"):
                for session_id in (first_id, second_id):
                    upsert_insight_materialization(
                        conn,
                        insight_type=insight_type,
                        session_id=session_id,
                        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
                        materialized_at_ms=1_770_000_300_000,
                    )

        status = await archive.get_session_insight_status()

        assert status.total_sessions == 2
        assert status.profile_row_count == 1
        assert status.missing_profile_row_count == 1
        assert status.profile_rows_ready is False
        assert status.work_event_inference_count == 1
        assert status.expected_work_event_inference_count == 1
        assert status.stale_work_event_inference_count == 0
        assert status.work_event_inference_rows_ready is True
        assert status.phase_count == 1
        assert status.expected_phase_count == 1
        assert status.phase_rows_ready is True
        assert status.thread_count == 2
        assert status.root_threads == 2
        assert status.threads_ready is True
        assert status.tag_rollup_count == 0
        assert status.expected_tag_rollup_count == 0
        assert status.tag_rollups_ready is True
    finally:
        await archive.close()


async def test_archive_tiers_api_marks_and_annotations_write_user_tier(tmp_path: Path) -> None:
    """Marks and annotations use archive ``user.db`` when is present."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-overlay-v1",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="overlay target")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)

        mark_added = await archive.add_mark(session_id, "star")
        mark_duplicate = await archive.add_mark(session_id, "star")
        marks = await archive.list_marks(session_id=session_id)
        mark_removed = await archive.remove_mark(session_id, "star")
        mark_missing = await archive.remove_mark(session_id, "star")
        with pytest.raises(ValueError, match="mark_type must be one of"):
            await archive.add_mark(session_id, "bogus")
        annotation_created = await archive.save_annotation("ann-v1", session_id, "needs follow-up")
        annotation_updated = await archive.save_annotation("ann-v1", session_id, "resolved")
        annotation = await archive.get_annotation("ann-v1")
        annotations = await archive.list_annotations(session_id=session_id)
        annotation_deleted = await archive.delete_annotation("ann-v1")
        annotation_missing = await archive.delete_annotation("ann-v1")

        assert mark_added is True
        assert mark_duplicate is False
        assert marks[0]["target_type"] == "session"
        assert marks[0]["target_id"] == session_id
        assert marks[0]["mark_type"] == "star"
        assert mark_removed is True
        assert mark_missing is False
        assert annotation_created is True
        assert annotation_updated is False
        assert annotation is not None
        assert annotation["note_text"] == "resolved"
        assert annotations == [annotation]
        assert annotation_deleted is True
        assert annotation_missing is False

        marks_after_delete = await archive.list_marks(session_id=session_id)
        annotations_after_delete = await archive.list_annotations(session_id=session_id)
        with sqlite3.connect(tmp_path / "user.db") as conn:
            assertion_statuses = conn.execute(
                """
                SELECT kind, key, status
                FROM assertions
                WHERE target_ref = ?
                ORDER BY kind, key
                """,
                (f"session:{session_id}",),
            ).fetchall()
        assert marks_after_delete == []
        assert annotations_after_delete == []
        assert assertion_statuses == [
            ("annotation", "ann-v1", "deleted"),
            ("mark", "star", "deleted"),
        ]
    finally:
        await archive.close()


async def test_archive_tiers_api_reader_artifacts_write_user_tier(tmp_path: Path) -> None:
    """Saved views, recall packs, and workspaces use ``user.db``."""
    import json
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-reader-artifacts-v1",
        title="Reader artifacts target",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="reader artifact target")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)

        view_created = await archive.save_view("view-v1", "Needs Review", '{"provider":"codex"}')
        view_updated = await archive.save_view("view-v1", "Needs Review", '{"provider":"codex","tag":"review"}')
        view_renamed_id = await archive.save_view("view-v2", "Needs Review", '{"provider":"codex","tag":"second"}')
        view = await archive.get_view("view-v1")
        replaced_view = await archive.get_view("view-v2")
        view_by_name = await archive.get_view_by_name("Needs Review")
        views = await archive.list_views()

        pack_created = await archive.create_recall_pack(
            "pack-v1",
            "Review Pack",
            json.dumps({"items": [{"target_type": "session", "session_id": session_id}]}),
        )
        pack_updated = await archive.create_recall_pack(
            "pack-v1",
            "Review Pack",
            json.dumps({"summary": "updated", "items": [{"target_type": "session", "session_id": session_id}]}),
        )
        pack = await archive.get_recall_pack("pack-v1")
        packs = await archive.list_recall_packs()

        workspace_created = await archive.save_workspace(
            "workspace-v1",
            "Review Workspace",
            "tabs",
            json.dumps([{"target_type": "session", "session_id": session_id}]),
            json.dumps({"panes": 1}),
            json.dumps({"target_type": "session", "session_id": session_id}),
        )
        workspace_updated = await archive.save_workspace(
            "workspace-v1",
            "Review Workspace",
            "stack",
            json.dumps([{"target_type": "session", "session_id": session_id}]),
            json.dumps({"panes": 2}),
            "{}",
        )
        workspace_renamed_id = await archive.save_workspace(
            "workspace-v2",
            "Review Workspace",
            "tabs",
            json.dumps([{"target_type": "session", "session_id": session_id}]),
            json.dumps({"panes": 3}),
            "{}",
        )
        workspace = await archive.get_workspace("workspace-v1")
        replaced_workspace = await archive.get_workspace("workspace-v2")
        workspaces = await archive.list_workspaces()

        assert view_created is True
        assert view_updated is False
        assert view_renamed_id is False
        assert view is None
        assert replaced_view is not None
        assert view_by_name == replaced_view
        assert views == [replaced_view]
        assert json.loads(replaced_view["query_json"]) == {"provider": "codex", "tag": "second"}
        assert pack_created is True
        assert pack_updated is False
        assert pack is not None
        assert packs == [pack]
        assert pack["session_ids_json"] == json.dumps([session_id], sort_keys=True)
        assert json.loads(pack["payload_json"])["resolved_count"] == 1
        assert workspace_created is True
        assert workspace_updated is False
        assert workspace_renamed_id is False
        assert workspace is None
        assert replaced_workspace is not None
        assert workspaces == [replaced_workspace]
        assert replaced_workspace["mode"] == "tabs"
        assert json.loads(replaced_workspace["layout_json"]) == {"panes": 3}

        assert await archive.delete_view("view-v2") is True
        assert await archive.delete_view("view-v2") is False
        assert await archive.delete_recall_pack("pack-v1") is True
        assert await archive.delete_recall_pack("pack-v1") is False
        assert await archive.delete_workspace("workspace-v2") is True
        assert await archive.delete_workspace("workspace-v2") is False

        assert await archive.list_views() == []
        assert await archive.list_recall_packs() == []
        assert await archive.list_workspaces() == []
        with sqlite3.connect(tmp_path / "user.db") as conn:
            assertion_statuses = conn.execute(
                """
                SELECT target_ref, kind, status
                FROM assertions
                WHERE target_ref IN ('saved_view:view-v1', 'recall_pack:pack-v1', 'workspace:workspace-v1')
                ORDER BY target_ref
                """
            ).fetchall()
        assert assertion_statuses == [
            ("recall_pack:pack-v1", "recall_pack", "deleted"),
            ("saved_view:view-v1", "saved_query", "deleted"),
            ("workspace:workspace-v1", "workspace_note", "deleted"),
        ]
    finally:
        await archive.close()


async def test_archive_tiers_api_corrections_write_user_tier(tmp_path: Path) -> None:
    """Learning corrections use ``user.db``."""
    import sqlite3

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.insights.feedback import CorrectionKind
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive = _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="api-corrections-v1",
        title="Correction target",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="correction target")],
            )
        ],
    )
    try:
        with ArchiveStore(archive.config.archive_root) as archive_db:
            session_id = archive_db.write_parsed(session)

        recorded = await archive.record_correction(session_id, "tag_accept", {"tag": "archive"}, note="operator")
        updated = await archive.record_correction(session_id, "tag_accept", {"tag": "archive-updated"})
        summary = await archive.record_correction(session_id, "summary_override", {"summary": "replacement"})
        listed = await archive.list_corrections(session_id=session_id)
        filtered = await archive.list_corrections(session_id=session_id, kind="summary_override")
        all_corrections = await archive.list_corrections()
        deleted = await archive.delete_correction(session_id, "tag_accept")
        missing_delete = await archive.delete_correction(session_id, "tag_accept")
        cleared = await archive.clear_corrections(session_id)

        assert recorded.session_id == session_id
        assert recorded.kind is CorrectionKind.TAG_ACCEPT
        assert recorded.payload == {"tag": "archive"}
        assert recorded.note == "operator"
        assert updated.kind is CorrectionKind.TAG_ACCEPT
        assert updated.payload == {"tag": "archive-updated"}
        assert updated.note is None
        assert summary.kind is CorrectionKind.SUMMARY_OVERRIDE
        assert {item.kind for item in listed} == {CorrectionKind.TAG_ACCEPT, CorrectionKind.SUMMARY_OVERRIDE}
        assert filtered == [summary]
        assert {item.kind for item in all_corrections} == {CorrectionKind.TAG_ACCEPT, CorrectionKind.SUMMARY_OVERRIDE}
        assert deleted is True
        assert missing_delete is False
        assert cleared == 1

        assert await archive.list_corrections(session_id=session_id) == []
        with sqlite3.connect(tmp_path / "user.db") as conn:
            assertion_statuses = conn.execute(
                """
                SELECT key, status, json_extract(value_json, '$.payload.tag'), json_extract(value_json, '$.payload.summary')
                FROM assertions
                WHERE target_ref = ? AND kind = 'correction'
                ORDER BY key
                """,
                (f"insight:{session_id}",),
            ).fetchall()
        assert assertion_statuses == [
            ("summary_override", "deleted", None, "replacement"),
            ("tag_accept", "deleted", "archive-updated", None),
        ]
    finally:
        await archive.close()


# ---------------------------------------------------------------------------
# 9. Metadata round-trip
# ---------------------------------------------------------------------------


async def test_get_metadata_returns_empty_dict_for_unknown_id(tmp_path: Path) -> None:
    """``get_metadata`` returns ``{}`` for an unknown session."""
    archive = _archive(tmp_path)
    try:
        result = await archive.get_metadata("nonexistent")
        assert isinstance(result, dict)
        assert result == {}
    finally:
        await archive.close()


# ---------------------------------------------------------------------------
# 10. ``workspace_env``-backed parity with the read-surface contract tests.
# ---------------------------------------------------------------------------
#
# The other tests in this module use the auto-isolated ``tmp_path``
# fixture for brevity. This test threads through the documented
# ``workspace_env`` fixture (see tests/conftest.py) to confirm the
# facade composes cleanly with the canonical XDG-isolated environment.


async def test_facade_runs_inside_workspace_env_fixture(
    workspace_env: dict[str, Path],
) -> None:
    """The facade composes with the ``workspace_env`` fixture from conftest."""
    archive_root = workspace_env["archive_root"]
    db_setup(workspace_env)
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    try:
        await _seed_two_sessions(archive_root / "index.db")
        rows = await archive.list_sessions()
        ids = {str(r.id) for r in rows}
        assert ids == {"claude-ai-export:conv-alpha", "chatgpt-export:conv-beta"}
    finally:
        await archive.close()


async def test_facade_judges_candidate_assertion_in_user_tier(workspace_env: dict[str, Path]) -> None:
    """Candidate assertion promotion is exposed through the Python facade."""

    archive_root = workspace_env["archive_root"]
    user_db = archive_root / "user.db"
    conn = sqlite3.connect(user_db)
    conn.row_factory = sqlite3.Row
    try:
        initialize_archive_tier(conn, ArchiveTier.USER)
        upsert_assertion(
            conn,
            assertion_id="candidate-api-1",
            target_ref="session:claude-ai-export:conv-alpha",
            kind=AssertionKind.TRANSFORM_CANDIDATE,
            value={"candidate_kind": "decision"},
            body_text="Use the shared assertion lifecycle.",
            evidence_refs=("session:claude-ai-export:conv-alpha",),
            status="candidate",
            visibility="private",
            context_policy={"inject": False, "promotion_required": True},
            now_ms=1_700_000_000_000,
        )
        conn.commit()
    finally:
        conn.close()

    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    try:
        candidates = await archive.list_assertion_candidates()
        assert [candidate.assertion_id for candidate in candidates] == ["candidate-api-1"]

        result = await archive.judge_assertion_candidate(
            candidate_ref="assertion:candidate-api-1",
            decision="accept",
            reason="confirmed by operator",
        )

        assert result.candidate.status == "accepted"
        assert result.judgment.decision == "accept"
        assert result.judgment.reason == "confirmed by operator"
        assert result.resulting_assertion is not None
        assert result.resulting_assertion.kind == "decision"
        assert result.resulting_assertion.supersedes == ("assertion:candidate-api-1",)
        post_candidates = await archive.list_assertion_candidates()
        assert post_candidates == []

        review = await archive.list_assertion_candidate_reviews()
        assert review.mode == "assertion-candidate-review-list"
        assert review.durable_assertions_excluded is True
        assert [item.candidate.assertion_id for item in review.items] == ["candidate-api-1"]
        assert review.items[0].review_status == "accepted"
        assert review.items[0].candidate.status is AssertionStatus.ACCEPTED
        assert review.items[0].latest_judgment is not None
        assert review.items[0].latest_judgment.decision == "accept"
        disabled = {action.availability.disabled_reason for action in review.items[0].action_affordances}
        assert disabled == {"candidate_already_accepted"}
    finally:
        await archive.close()
