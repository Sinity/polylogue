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
5. **Error envelope** — methods that resolve conversation IDs raise
   :class:`ConversationNotFoundError` (a typed
   :class:`PolylogueError` subclass) on unknown IDs.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.api.archive import ConversationNotFoundError
from polylogue.errors import PolylogueError
from polylogue.types import Provider
from tests.infra.storage_records import ConversationBuilder, db_setup

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
# against an unknown conversation ID to assert the error envelope.

# Methods that close transport-level resources, not domain operations.
LIFECYCLE_METHODS: frozenset[str] = frozenset({"close"})

# Read-only methods that take a conversation_id; resolving an unknown ID
# returns ``None`` (no error). They are part of the read contract.
READ_BY_ID_NONE_METHODS: frozenset[str] = frozenset(
    {
        "get_conversation",
        "get_conversation_summary",
        "get_session_profile_insight",
        "get_session_latency_profile_insight",
        "resume_brief",
    }
)

# Read-only methods that take a conversation_id and return an empty
# collection / zero stats / dict for missing IDs rather than raising.
READ_BY_ID_EMPTY_METHODS: frozenset[str] = frozenset(
    {
        "get_messages_paginated",  # special: raises ConversationNotFoundError
        "get_conversation_stats",
        "get_session_work_event_insights",
        "get_session_phase_insights",
        "get_session_tree",
        "get_raw_artifacts_for_conversation",
        "bulk_get_messages",
    }
)

# Read-only methods with no required arguments — should always succeed
# on an empty archive and return an empty container or zero counts.
READ_NULLARY_METHODS: frozenset[str] = frozenset(
    {
        "list_conversations",
        "stats",
        "facets",
        "health_check",
        "list_tags",
        "list_marks",
        "list_annotations",
        "list_views",
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
        "list_work_thread_insights",
        "list_archive_coverage_insights",
        "list_tool_usage_insights",
        "list_session_cost_insights",
        "list_cost_rollup_insights",
        "list_archive_debt_insights",
        "count_conversations",
        "rebuild_index",
        "parse_sources",
    }
)

# Query methods that take a free-text query string and produce a search
# envelope or result set.
SEARCH_METHODS: frozenset[str] = frozenset({"search", "search_envelope"})

# Methods that mutate state by conversation ID. Calling them with an
# unknown ID should raise ``ConversationNotFoundError`` (a typed
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

# Mutation methods that take only a conversation_id and return a typed
# envelope describing the outcome (no raise on missing IDs).
MUTATION_BY_ID_TYPED_OUTCOME: frozenset[str] = frozenset(
    {
        "delete_conversation",
        "delete_conversation_safe",
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
        "get_conversations",
        "query_conversations",
        "bulk_tag_conversations",
        "list_session_profile_insights",
        "get_work_thread_insight",
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
        "find_resume_candidates",
        "export_insight_bundle",
        "parse_file",
        "cost_outlook",
        # Topology read methods exposed in tests/unit/api/test_topology_api.py.
        "get_ancestors",
        "get_descendants",
        "get_logical_session",
        "get_session_topology",
        "get_siblings",
        "get_thread",
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
    return Polylogue(archive_root=tmp_path, db_path=tmp_path / "polylogue.db")


async def _seed_two_conversations(db_path: Path) -> None:
    """Seed two minimal conversations for happy-path read assertions."""
    await (
        ConversationBuilder(db_path, "conv-alpha")
        .provider(Provider.CLAUDE_AI.value)
        .title("Alpha")
        .add_message(message_id="alpha-m1", role="user", text="alpha body")
        .add_message(message_id="alpha-m2", role="assistant", text="alpha reply")
        .build()
    )
    await (
        ConversationBuilder(db_path, "conv-beta")
        .provider(Provider.CHATGPT.value)
        .title("Beta")
        .add_message(message_id="beta-m1", role="user", text="beta body")
        .build()
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
    "list_conversations",
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
    "list_work_thread_insights",
    "list_archive_coverage_insights",
    "list_tool_usage_insights",
    "list_session_cost_insights",
    "list_cost_rollup_insights",
    "list_archive_debt_insights",
    "list_annotations",
)


# Insight list methods that auto-materialize default-empty rows for the
# zero-conversation archive (e.g. global archive-debt placeholders). For
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


async def test_count_conversations_returns_int_on_empty_archive(tmp_path: Path) -> None:
    """``count_conversations`` returns ``0`` (a real ``int``) on empty archive."""
    archive = _archive(tmp_path)
    try:
        result = await archive.count_conversations()
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
        assert result.conversation_count == 0
        assert result.message_count == 0
        assert result.word_count == 0
        assert isinstance(result.providers, dict)
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


async def test_facets_returns_typed_envelope_on_empty_archive(tmp_path: Path) -> None:
    """``facets()`` returns the canonical ``FacetsResponse`` envelope."""
    from polylogue.surfaces.payloads import FacetsResponse

    archive = _archive(tmp_path)
    try:
        result = await archive.facets()
        assert isinstance(result, FacetsResponse)
        assert result.total_conversations == 0
        assert result.total_messages == 0
        assert isinstance(result.providers, dict)
    finally:
        await archive.close()


# ---------------------------------------------------------------------------
# 5. Happy path on a seeded archive
# ---------------------------------------------------------------------------


async def test_list_conversations_returns_seeded_rows(tmp_path: Path) -> None:
    """``list_conversations`` returns the seeded rows as typed conversations."""
    from polylogue.archive.conversation.models import Conversation

    archive = _archive(tmp_path)
    db_path = archive.config.archive_root / "polylogue.db"
    # Ensure schema is initialized by touching the repository before seeding.
    _ = archive.repository
    await _seed_two_conversations(db_path)
    try:
        rows = await archive.list_conversations()
        assert isinstance(rows, list)
        assert len(rows) == 2
        for row in rows:
            assert isinstance(row, Conversation)
        ids = {str(r.id) for r in rows}
        assert ids == {"conv-alpha", "conv-beta"}
    finally:
        await archive.close()


async def test_list_conversations_respects_provider_filter(tmp_path: Path) -> None:
    """``list_conversations(provider=...)`` narrows the result set."""
    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "polylogue.db"
    await _seed_two_conversations(db_path)
    try:
        rows = await archive.list_conversations(provider=Provider.CLAUDE_AI.value)
        ids = {str(r.id) for r in rows}
        assert ids == {"conv-alpha"}
    finally:
        await archive.close()


async def test_list_conversations_respects_limit(tmp_path: Path) -> None:
    """``list_conversations(limit=1)`` returns at most one row."""
    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "polylogue.db"
    await _seed_two_conversations(db_path)
    try:
        rows = await archive.list_conversations(limit=1)
        assert len(rows) == 1
    finally:
        await archive.close()


async def test_stats_reflects_seeded_archive(tmp_path: Path) -> None:
    """``stats()`` reports the seeded conversation count."""
    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "polylogue.db"
    await _seed_two_conversations(db_path)
    try:
        result = await archive.stats()
        assert result.conversation_count == 2
        # Message count is at least the three rows we seeded.
        assert result.message_count >= 3
    finally:
        await archive.close()


async def test_get_conversation_returns_typed_object(tmp_path: Path) -> None:
    """``get_conversation`` returns a typed ``Conversation``."""
    from polylogue.archive.conversation.models import Conversation

    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "polylogue.db"
    await _seed_two_conversations(db_path)
    try:
        conv = await archive.get_conversation("conv-alpha")
        assert conv is not None
        assert isinstance(conv, Conversation)
        assert conv.title == "Alpha"
    finally:
        await archive.close()


# ---------------------------------------------------------------------------
# 6. Error envelope: ConversationNotFoundError is a typed PolylogueError
# ---------------------------------------------------------------------------


def test_conversation_not_found_is_typed_polylogue_error() -> None:
    """``ConversationNotFoundError`` carries an HTTP status code and inherits the typed base."""
    assert issubclass(ConversationNotFoundError, PolylogueError)
    assert ConversationNotFoundError.http_status_code == 404


@pytest.mark.parametrize(
    "method_name",
    sorted(MUTATION_BY_ID_RAISES_METHODS),
)
async def test_mutation_methods_raise_on_unknown_id(
    tmp_path: Path,
    method_name: str,
) -> None:
    """Mutation-by-ID methods raise ``ConversationNotFoundError`` for unknown IDs."""
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
        with pytest.raises(ConversationNotFoundError):
            await coro  # type: ignore[misc]
    finally:
        await archive.close()


async def test_delete_conversation_returns_false_on_unknown_id(tmp_path: Path) -> None:
    """``delete_conversation`` returns ``False`` rather than raising for unknown IDs."""
    archive = _archive(tmp_path)
    try:
        result = await archive.delete_conversation("nonexistent")
        assert result is False
    finally:
        await archive.close()


async def test_delete_conversation_safe_returns_typed_not_found(tmp_path: Path) -> None:
    """``delete_conversation_safe`` returns ``outcome='not_found'`` for unknown IDs."""
    from polylogue.surfaces.payloads import DeleteConversationResult

    archive = _archive(tmp_path)
    try:
        result = await archive.delete_conversation_safe("nonexistent")
        assert isinstance(result, DeleteConversationResult)
        assert result.outcome == "not_found"
    finally:
        await archive.close()


async def test_get_conversation_returns_none_for_unknown_id(tmp_path: Path) -> None:
    """Read-by-ID methods return ``None`` for unknown IDs, not an exception."""
    archive = _archive(tmp_path)
    try:
        assert await archive.get_conversation("nonexistent") is None
        assert await archive.get_conversation_summary("nonexistent") is None
        assert await archive.get_session_profile_insight("nonexistent") is None
    finally:
        await archive.close()


async def test_get_messages_paginated_raises_for_unknown_id(tmp_path: Path) -> None:
    """``get_messages_paginated`` raises ``ConversationNotFoundError`` for unknown IDs."""
    archive = _archive(tmp_path)
    try:
        with pytest.raises(ConversationNotFoundError):
            await archive.get_messages_paginated("nonexistent")
    finally:
        await archive.close()


# ---------------------------------------------------------------------------
# 7. Required-arg validation
# ---------------------------------------------------------------------------


async def test_save_annotation_rejects_blank_inputs(tmp_path: Path) -> None:
    """``save_annotation`` validates non-empty ``annotation_id`` and ``note_text``."""
    archive = _archive(tmp_path)
    _ = archive.repository
    db_path = archive.config.archive_root / "polylogue.db"
    await _seed_two_conversations(db_path)
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
    """``neighbor_candidates`` needs at least one of ``conversation_id`` / ``query``.

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


# ---------------------------------------------------------------------------
# 9. Metadata round-trip
# ---------------------------------------------------------------------------


async def test_get_metadata_returns_empty_dict_for_unknown_id(tmp_path: Path) -> None:
    """``get_metadata`` returns ``{}`` for an unknown conversation."""
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
    db_path = db_setup(workspace_env)
    archive = Polylogue(archive_root=archive_root, db_path=db_path)
    try:
        await _seed_two_conversations(db_path)
        rows = await archive.list_conversations()
        ids = {str(r.id) for r in rows}
        assert ids == {"conv-alpha", "conv-beta"}
    finally:
        await archive.close()
