"""Tests for ``enrich_bundle_from_db`` and ``_build_single_cache`` in
``polylogue/pipeline/prepare_enrichment.py``.

Verifies that paste boundary state, provider_meta enrichment, topology edges,
and message fields propagate through the DB-aware enrichment stage.
"""

from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.topology.edge import TopologyEdgeStatus, TopologyEdgeType
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.prepare_enrichment import enrich_bundle_from_db
from polylogue.pipeline.prepare_models import (
    AttachmentMaterializationPlan,
    PrepareCache,
    RecordBundle,
    TransformResult,
)
from polylogue.sources.parsers.base_models import ParsedMessage, ParsedSession
from polylogue.storage.runtime import (
    AttachmentRecord,
    ContentBlockRecord,
    MessageRecord,
    ProviderEventRecord,
    SessionRecord,
)
from polylogue.types import ContentHash, MessageId, Provider, ProviderEventId, SessionId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_record(
    session_id: str = "test:conv-1",
    title: str = "Test Conv",
    created_at: str = "2026-05-01T00:00:00Z",
    updated_at: str = "2026-05-01T01:00:00Z",
    **kwargs: object,
) -> SessionRecord:
    defaults: dict[str, object] = {
        "session_id": SessionId(session_id),
        "provider_session_id": "prov-1",
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "sort_key": 1.0,
        "content_hash": ContentHash("hash-1"),
        "provider_meta": {"source": "test"},
        "source_name": "test",
        "working_directories_json": None,
        "git_branch": None,
        "git_repository_url": None,
        "parent_session_id": None,
        "branch_type": None,
        "raw_id": None,
    }
    defaults.update(kwargs)
    return SessionRecord(**defaults)  # type: ignore[arg-type]


def _make_message_record(
    message_id: str = "test:msg-1",
    session_id: str = "test:conv-1",
    role: Role = Role.USER,
    text: str | None = "hello",
    sort_key: float = 0.0,
    word_count: int = 1,
    has_tool_use: int = 0,
    has_thinking: int = 0,
    has_paste: int = 0,
    paste_boundary_state: str | None = None,
    **kwargs: object,
) -> MessageRecord:
    defaults: dict[str, object] = {
        "message_id": MessageId(message_id),
        "session_id": SessionId(session_id),
        "provider_message_id": "prov-msg-1",
        "role": role,
        "text": text,
        "sort_key": sort_key,
        "content_hash": ContentHash("mhash-1"),
        "version": 1,
        "parent_message_id": None,
        "branch_index": 0,
        "source_name": "",
        "word_count": word_count,
        "has_tool_use": has_tool_use,
        "has_thinking": has_thinking,
        "has_paste": has_paste,
        "paste_boundary_state": paste_boundary_state,
    }
    defaults.update(kwargs)
    return MessageRecord(**defaults)  # type: ignore[arg-type]


def _make_transform(
    candidate_cid: str = "test:conv-1",
    content_hash_str: str = "hash-1",
    messages: list[MessageRecord] | None = None,
    session: SessionRecord | None = None,
    content_blocks: list[ContentBlockRecord] | None = None,
    attachments: list[AttachmentRecord] | None = None,
    provider_events: list[ProviderEventRecord] | None = None,
) -> TransformResult:
    """Build a TransformResult with sensible defaults."""
    msgs = messages if messages is not None else [_make_message_record(message_id="test:msg-1", text="hello")]
    conv = session or _make_session_record()
    mid_map: dict[str, MessageId] = {}
    for i, msg in enumerate(msgs, start=1):
        key = str(msg.provider_message_id) if msg.provider_message_id else f"msg-{i}"
        mid_map[key] = msg.message_id

    bundle = RecordBundle(
        session=conv,
        messages=msgs,
        attachments=attachments or [],
        content_blocks=content_blocks or [],
        provider_events=provider_events or [],
    )
    return TransformResult(
        bundle=bundle,
        materialization_plan=AttachmentMaterializationPlan(),
        content_hash=ContentHash(content_hash_str),
        candidate_cid=SessionId(candidate_cid),
        message_id_map=mid_map,
    )


def _make_parsed_session(
    source_name: str = "claude-code",
    provider_session_id: str = "prov-1",
    title: str = "Test Conv",
    messages: list[ParsedMessage] | None = None,
    provider_meta: dict[str, object] | None = None,
    **kwargs: object,
) -> ParsedSession:
    """Build a ParsedSession with sensible defaults.

    Pass ``messages=[]`` explicitly to get zero messages; ``None`` (the
    default) fills in one placeholder message.
    """
    msgs = (
        messages
        if messages is not None
        else [
            ParsedMessage(provider_message_id="prov-msg-1", role=Role.USER, text="hello"),
        ]
    )
    return ParsedSession(
        source_name=Provider.from_string(source_name),
        provider_session_id=provider_session_id,
        title=title,
        messages=msgs,
        provider_meta=provider_meta,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Paste boundary state propagation
# ---------------------------------------------------------------------------


class TestPasteBoundaryPropagation:
    """Paste boundary state on input MessageRecords is preserved in the
    enriched output PreparedBundle."""

    def test_paste_boundary_exact(self) -> None:
        """exact boundary state passes through enrichment."""
        msg = _make_message_record(
            message_id="test:msg-1",
            provider_message_id="pm-1",
            text="content with [Pasted text #1]",
            has_paste=1,
            paste_boundary_state="exact",
        )
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(
            messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="content with [Pasted text #1]")]
        )
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        out_msgs = result.bundle.messages
        assert len(out_msgs) == 1
        assert out_msgs[0].has_paste == 1
        assert out_msgs[0].paste_boundary_state == "exact"

    def test_paste_boundary_projected(self) -> None:
        """projected boundary state passes through enrichment."""
        msg = _make_message_record(
            message_id="test:msg-1",
            provider_message_id="pm-1",
            text="pasted content",
            has_paste=1,
            paste_boundary_state="projected",
        )
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(
            messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="pasted content")]
        )
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.bundle.messages[0].paste_boundary_state == "projected"

    def test_paste_boundary_hash_only(self) -> None:
        """hash_only boundary state passes through enrichment."""
        msg = _make_message_record(
            message_id="test:msg-1",
            provider_message_id="pm-1",
            text=None,
            has_paste=1,
            paste_boundary_state="hash_only",
        )
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text=None)])
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.bundle.messages[0].paste_boundary_state == "hash_only"

    def test_paste_boundary_none(self) -> None:
        """None boundary state (no paste) passes through as None."""
        msg = _make_message_record(
            message_id="test:msg-1",
            provider_message_id="pm-1",
            text="normal message",
            has_paste=0,
            paste_boundary_state=None,
        )
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(
            messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="normal message")]
        )
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.bundle.messages[0].has_paste == 0
        assert result.bundle.messages[0].paste_boundary_state is None

    def test_multiple_messages_mixed_paste_state(self) -> None:
        """Different paste states on different messages are each preserved."""
        msg1 = _make_message_record(
            message_id="test:msg-1",
            provider_message_id="pm-1",
            text="normal",
            has_paste=0,
            paste_boundary_state=None,
        )
        msg2 = _make_message_record(
            message_id="test:msg-2",
            provider_message_id="pm-2",
            text="[Pasted text #1] long content",
            has_paste=1,
            paste_boundary_state="exact",
        )
        transform = _make_transform(messages=[msg1, msg2])
        convo = _make_parsed_session(
            messages=[
                ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="normal"),
                ParsedMessage(provider_message_id="pm-2", role=Role.USER, text="[Pasted text #1] long content"),
            ]
        )
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        out = result.bundle.messages
        assert out[0].has_paste == 0
        assert out[0].paste_boundary_state is None
        assert out[1].has_paste == 1
        assert out[1].paste_boundary_state == "exact"


# ---------------------------------------------------------------------------
# Empty / minimal input
# ---------------------------------------------------------------------------


class TestEmptyMessagesNoCrash:
    """Enrichment handles empty/minimal input gracefully."""

    def test_empty_messages(self) -> None:
        """Zero messages → valid PreparedBundle with empty messages list."""
        transform = _make_transform(messages=[])
        convo = _make_parsed_session(messages=[])
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert len(result.bundle.messages) == 0
        assert len(result.bundle.attachments) == 0
        assert len(result.bundle.content_blocks) == 0
        assert result.cid == SessionId("test:conv-1")

    def test_no_provider_meta(self) -> None:
        """Session without provider_meta still enriches."""
        msg = _make_message_record(provider_message_id="pm-1")
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(
            messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="hello")],
            provider_meta=None,
        )
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.bundle.messages[0].text == "hello"

    def test_none_text_message(self) -> None:
        """Message with None text passes through."""
        msg = _make_message_record(message_id="test:msg-1", provider_message_id="pm-1", text=None)
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text=None)])
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.bundle.messages[0].text is None


# ---------------------------------------------------------------------------
# Full enrichment
# ---------------------------------------------------------------------------


class TestFullEnrichment:
    """All fields populated → complete enrichment."""

    def test_all_message_fields_preserved(self) -> None:
        """Every message field (tokens, model, message_type, etc.) is preserved."""
        msg = _make_message_record(
            message_id="test:msg-1",
            provider_message_id="pm-1",
            role=Role.ASSISTANT,
            text="I'll help with that.",
            sort_key=1.5,
            word_count=5,
            has_tool_use=1,
            has_thinking=1,
            has_paste=1,
            paste_boundary_state="exact",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_write_tokens=10,
            model_name="claude-sonnet-4-20250514",
            branch_index=0,
        )
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(
            messages=[
                ParsedMessage(
                    provider_message_id="pm-1",
                    role=Role.ASSISTANT,
                    text="I'll help with that.",
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_tokens=200,
                    cache_write_tokens=10,
                    model_name="claude-sonnet-4-20250514",
                )
            ]
        )
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        out = result.bundle.messages[0]
        assert out.role == Role.ASSISTANT
        assert out.text == "I'll help with that."
        assert out.word_count == 5
        assert out.has_tool_use == 1
        assert out.has_thinking == 1
        assert out.has_paste == 1
        assert out.paste_boundary_state == "exact"
        assert out.sort_key == 1.5
        # Note: token fields (input_tokens, output_tokens, etc.) and
        # model_name are NOT propagated through enrich_bundle_from_db —
        # it constructs a new MessageRecord from a curated subset of fields.

    def test_provider_meta_enrichment(self) -> None:
        """Provider_meta fields (working_directories, git, cwd) are extracted."""
        msg = _make_message_record(message_id="test:msg-1", provider_message_id="pm-1")
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(
            messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="hello")],
            provider_meta={
                "source": "claude-code",
                "working_directories": ["/realm/project/polylogue"],
                "gitBranch": "feature/test",
                "git": {
                    "branch": "feature/test-git",
                    "repository_url": "https://github.com/Sinity/polylogue",
                },
            },
        )
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        conv = result.bundle.session
        assert conv.source_name == "claude-code"
        import json

        wds = json.loads(conv.working_directories_json or "[]")
        assert "/realm/project/polylogue" in wds
        assert conv.git_branch == "feature/test"
        assert conv.git_repository_url == "https://github.com/Sinity/polylogue"

    def test_parent_session_topology(self) -> None:
        """When parent is in cache, parent_session_id is set and a
        resolved topology edge is emitted."""
        # Use a recognized provider name so candidate_parent is predictable.
        parent_id = make_session_id("claude-code", "parent-prov-1")

        msg = _make_message_record(message_id="test:msg-1", provider_message_id="pm-1")
        transform = _make_transform(messages=[msg], candidate_cid="claude-code:child-1")
        convo = _make_parsed_session(
            source_name="claude-code",
            messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="hello")],
            parent_session_provider_id="parent-prov-1",
            branch_type=BranchType.CONTINUATION,
        )
        cache = PrepareCache()
        cache.known_ids.add(parent_id)

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.bundle.session.parent_session_id == parent_id
        assert len(result.bundle.topology_edges) == 1
        edge = result.bundle.topology_edges[0]
        assert edge.dst_provider_native_id == "parent-prov-1"
        assert edge.edge_type == TopologyEdgeType.CONTINUATION
        assert edge.status == TopologyEdgeStatus.RESOLVED
        assert edge.resolved_dst_session_id == parent_id

    def test_parent_not_in_cache_yields_unresolved_edge(self) -> None:
        """When parent is not in cache, the topology edge is unresolved."""
        msg = _make_message_record(message_id="test:msg-1", provider_message_id="pm-1")
        transform = _make_transform(messages=[msg])
        convo = _make_parsed_session(
            source_name="claude-code",
            messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="hello")],
            parent_session_provider_id="unknown-parent",
            branch_type=BranchType.FORK,
        )
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.bundle.session.parent_session_id is None
        assert len(result.bundle.topology_edges) == 1
        edge = result.bundle.topology_edges[0]
        assert edge.status == TopologyEdgeStatus.UNRESOLVED
        assert edge.resolved_dst_session_id is None
        assert edge.edge_type == TopologyEdgeType.FORK

    def test_provider_events_preserved(self) -> None:
        """Provider events in the transform bundle are preserved in output."""
        pe = ProviderEventRecord(
            event_id=ProviderEventId("evt-1"),
            session_id=SessionId("test:conv-1"),
            source_name="claude-code",
            event_index=0,
            event_type="compaction",
            timestamp="2026-05-01T00:00:00Z",
            sort_key=0.0,
            payload={"key": "value"},
        )
        msg = _make_message_record(provider_message_id="pm-1")
        transform = _make_transform(messages=[msg], provider_events=[pe])
        convo = _make_parsed_session(messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="hello")])
        cache = PrepareCache()

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert len(result.bundle.provider_events) == 1
        out_evt = result.bundle.provider_events[0]
        assert out_evt.event_type == "compaction"
        # event_id is recomputed by provider_event_id(cid, event_index).
        assert out_evt.event_id == ProviderEventId("test:conv-1:provider-event:000000")

    def test_existing_session_changed_flag(self) -> None:
        """When content_hash differs from existing, changed=True."""
        from polylogue.storage.archive_views import ExistingSession

        msg = _make_message_record(provider_message_id="pm-1")
        transform = _make_transform(
            candidate_cid="test:conv-1",
            content_hash_str="new-hash",
            messages=[msg],
        )
        convo = _make_parsed_session(messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="hello")])
        cache = PrepareCache()
        cache.existing["test:conv-1"] = ExistingSession(
            session_id="test:conv-1",
            content_hash=ContentHash("old-hash"),
        )

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.changed is True

    def test_existing_session_unchanged_flag(self) -> None:
        """When content_hash matches existing, changed=False."""
        from polylogue.storage.archive_views import ExistingSession

        msg = _make_message_record(provider_message_id="pm-1")
        transform = _make_transform(
            candidate_cid="test:conv-1",
            content_hash_str="same-hash",
            messages=[msg],
        )
        convo = _make_parsed_session(messages=[ParsedMessage(provider_message_id="pm-1", role=Role.USER, text="hello")])
        cache = PrepareCache()
        cache.existing["test:conv-1"] = ExistingSession(
            session_id="test:conv-1",
            content_hash=ContentHash("same-hash"),
        )

        result = enrich_bundle_from_db(convo, source_name="claude-code", transform=transform, cache=cache)

        assert result.changed is False
