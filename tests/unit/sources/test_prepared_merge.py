"""Parity checks for worker-side composition of retained revisions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role, TitleSource
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import merge_parsed_session_chunks
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.sources.prepared_jsonl import PreparedJsonl, _write_artifact
from polylogue.sources.prepared_merge import prepare_retained_cohort_artifact, prepared_cohort_source_hash
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from tests.infra.index_writer import write_fixture_index_session


@pytest.fixture
def event_index(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    from polylogue.core.write_lease import write_lease
    from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.index_writer import close_fixture_index_connection

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.event-archive.bootstrap", archive_root=root):
        bootstrap_archive_root(root)
        conn = open_isolated_write_connection(root / "index.db", purpose="test.event-archive", archive_root=root)
    try:
        yield conn
    finally:
        close_fixture_index_connection(conn)


def _chunk_artifact(directory: Path, session: ParsedSession, source_hash: str) -> PreparedJsonl:
    directory.mkdir(parents=True, exist_ok=True)
    store = SqliteMessageStore(directory / f"{source_hash}.db")
    try:
        session.content_hash = session_content_hash(session)
        shard = prepare_session_shard(directory, [session])
        _write_artifact(store, source_hash, [session], enrichment_digest=None, enrichment_index_path=None)
    finally:
        store.close()
    return PreparedJsonl.seal(source_hash, store.path, shard.path)


def test_prepared_cohort_preserves_merge_order_title_and_duplicate_leaf(tmp_path: Path) -> None:
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="same",
        title="weak",
        title_source=TitleSource.HEURISTIC,
        title_ref="message:dup",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:01:00Z",
        models_used=["alpha"],
        working_directories=["/first"],
        messages=[
            ParsedMessage(provider_message_id="dup", role=Role.USER, text="first", position=9, is_active_leaf=True)
        ],
        session_events=[ParsedSessionEvent(event_type="turn_context", payload={"chunk": 1})],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="same",
        title="strong",
        title_source=TitleSource.ORIGIN,
        title_ref="codex-thread-name:same",
        created_at="2026-01-02T00:00:00Z",
        updated_at="2026-01-02T00:01:00Z",
        models_used=["beta"],
        working_directories=["/second"],
        messages=[
            ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="last", position=0, is_active_leaf=True)
        ],
        session_events=[ParsedSessionEvent(event_type="turn_context", payload={"chunk": 2})],
    )
    ordered = [
        ("raw-0", _chunk_artifact(tmp_path, first, "a" * 64)),
        ("raw-1", _chunk_artifact(tmp_path, second, "b" * 64)),
    ]

    aggregate = prepare_retained_cohort_artifact(ordered, tmp_path)
    aggregate.verify_files(full=True)
    session = list(aggregate.iter_sessions())[0]
    expected = merge_parsed_session_chunks([first, second])[0]

    assert aggregate.blob_hash == prepared_cohort_source_hash(ordered)
    assert session.content_hash == session_content_hash(expected)
    assert session.title == expected.title
    assert session.title == "strong"
    assert session.title_source is TitleSource.ORIGIN
    assert session.created_at == expected.created_at
    assert session.updated_at == expected.updated_at
    assert session.models_used == expected.models_used
    assert session.working_directories == expected.working_directories
    assert [
        (message.provider_message_id, message.text, message.position, message.is_active_leaf)
        for message in session.messages
    ] == [
        ("dup", "first", 0, False),
        ("dup", "last", 1, True),
    ]
    assert [event.payload["chunk"] for event in session.session_events] == [1, 2]
    assert prepared_cohort_source_hash(list(reversed(ordered))) != aggregate.blob_hash


def test_prepared_claude_cohort_reduces_chunk_summaries(tmp_path: Path) -> None:
    first = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="same",
        updated_at="2026-01-01T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="zero")],
        session_events=[
            ParsedSessionEvent(event_type="point", payload={"chunk": 1}),
            ParsedSessionEvent(event_type="claude_session_environment", payload={"entrypoints": {"cli": 1}}),
            ParsedSessionEvent(event_type="claude_parse_coverage", payload={"sidecar_seen": {"user": 1}}),
        ],
    )
    second = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="same",
        updated_at="2026-01-02T00:00:00Z",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.ASSISTANT, text="one")],
        session_events=[
            ParsedSessionEvent(event_type="point", payload={"chunk": 2}),
            ParsedSessionEvent(event_type="claude_session_environment", payload={"entrypoints": {"cli": 2, "hook": 1}}),
            ParsedSessionEvent(event_type="claude_parse_coverage", payload={"sidecar_seen": {"user": 3}}),
        ],
    )
    ordered = [
        ("raw-0", _chunk_artifact(tmp_path, first, "a" * 64)),
        ("raw-1", _chunk_artifact(tmp_path, second, "b" * 64)),
    ]

    aggregate = prepare_retained_cohort_artifact(ordered, tmp_path)
    session = list(aggregate.iter_sessions())[0]
    expected = merge_parsed_session_chunks([first, second])[0]

    assert [event.event_type for event in session.session_events] == [
        event.event_type for event in expected.session_events
    ]
    assert [event.payload for event in session.session_events] == [event.payload for event in expected.session_events]
    assert session.content_hash == session_content_hash(expected)


def test_prepared_cohort_rejects_conflicting_session_identity(tmp_path: Path) -> None:
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="first",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="first")],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="second",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="second")],
    )
    ordered = [
        ("raw-0", _chunk_artifact(tmp_path, first, "a" * 64)),
        ("raw-1", _chunk_artifact(tmp_path, second, "b" * 64)),
    ]

    with pytest.raises(ValueError, match="provider-native session identity"):
        prepare_retained_cohort_artifact(ordered, tmp_path)


@pytest.mark.parametrize("prepared", [False, True], ids=["memory", "prepared"])
def test_chunk_references_survive_composition_and_archive_write(
    tmp_path: Path, prepared: bool, event_index: sqlite3.Connection
) -> None:
    """F016: the second chunk's compaction and attachment must not bind to the first."""
    import hashlib
    from contextlib import closing

    from polylogue.core.message_owner import MessageOwnerCoordinate
    from polylogue.sources.parsers.base import ParsedAttachment
    from tests.infra.index_writer import write_fixture_index_session

    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="coordinates",
        messages=[
            ParsedMessage(provider_message_id=f"first-{i}", role=Role.USER, text=f"first {i}", position=i)
            for i in range(3)
        ],
    )
    owner = MessageOwnerCoordinate(stable_key="second-owner", position=1)
    second_messages = [
        ParsedMessage(provider_message_id=f"second-{i}", role=Role.ASSISTANT, text=f"second {i}", position=i)
        for i in range(4)
    ]
    second_messages[1] = second_messages[1].model_copy(update={"owner_coordinate": owner})
    second_messages[2] = second_messages[2].model_copy(update={"parent_message_position": 0})
    attachment = ParsedAttachment(
        provider_attachment_id="second-file",
        message_position=1,
        owner_coordinate=owner,
        name="synthetic.txt",
        inline_bytes=b"synthetic",
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="coordinates",
        messages=second_messages,
        attachments=[attachment],
        session_events=[
            ParsedSessionEvent(
                event_type="compaction",
                payload={"source_index": 7},
                boundary_start_position=0,
                boundary_end_position=2,
                boundary_message_position=3,
            )
        ],
    )
    original_key = attachment.acquisition_key
    expected = merge_parsed_session_chunks([first, second])[0]
    artifacts = []
    try:
        if prepared:
            artifacts = [
                _chunk_artifact(tmp_path / f"chunk-{i}", chunk, str(i + 1) * 64)
                for i, chunk in enumerate((first, second))
            ]
            aggregate = prepare_retained_cohort_artifact(
                [(f"raw-{i}", artifact) for i, artifact in enumerate(artifacts)], tmp_path / "merged"
            )
            artifacts.append(aggregate)
            with closing(aggregate.iter_sessions()) as sessions:
                merged = next(sessions)
            assert session_content_hash(merged) == session_content_hash(expected)
        else:
            merged = expected
            assert merged.attachments[0].acquisition_key == original_key
        assert attachment.message_position == 1
        assert attachment.owner_coordinate == owner
        assert second.messages[2].parent_message_position == 0
        event = merged.session_events[0]
        assert (event.boundary_start_position, event.boundary_end_position, event.boundary_message_position) == (
            3,
            5,
            6,
        )
        assert event.payload == {"source_index": 7}
        assert merged.attachments[0].message_position == 4
        assert merged.attachments[0].owner_coordinate == MessageOwnerCoordinate(stable_key="second-owner", position=4)
        assert merged.messages[5].parent_message_position == 3
        blob = hashlib.sha256(b"synthetic").digest()
        key = merged.attachments[0].acquisition_key if prepared else original_key
        conn = event_index
        sid = write_fixture_index_session(
            conn, merged, preacquired_attachment_blobs={key: (blob, len(b"synthetic"), "acquired")}
        )
        stored = conn.execute(
            "SELECT boundary_start_position, boundary_end_position, boundary_message_id "
            "FROM session_events WHERE session_id = ? AND event_type = 'compaction'",
            (sid,),
        ).fetchone()
        assert tuple(stored) == (3, 5, f"{sid}:n:second-3")
        assert conn.execute("SELECT message_id FROM attachment_refs").fetchall()[0][0] == f"{sid}:n:second-1"
        assert conn.execute("SELECT blob_hash FROM attachments").fetchone()[0] == blob
        assert (
            conn.execute("SELECT parent_message_id FROM messages WHERE native_id = 'second-2'").fetchone()[0]
            == f"{sid}:n:second-0"
        )
    finally:
        for artifact in reversed(artifacts):
            artifact.discard()


@pytest.mark.parametrize("prepared", [False, True], ids=["memory", "prepared"])
def test_empty_compaction_ranges_survive_three_chunk_merge(tmp_path: Path, prepared: bool) -> None:
    from contextlib import closing

    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="empty",
        messages=[ParsedMessage(provider_message_id="a", role=Role.USER, position=9, text="a")],
    )
    empty = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="empty",
        messages=[],
        session_events=[
            ParsedSessionEvent(event_type="compaction", boundary_start_position=0, boundary_end_position=-1)
        ],
    )
    last = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="empty",
        messages=[ParsedMessage(provider_message_id="b", role=Role.USER, position=0, text="b")],
    )
    artifacts = []
    try:
        if prepared:
            artifacts = [
                _chunk_artifact(tmp_path / str(i), chunk, str(i + 1) * 64)
                for i, chunk in enumerate((first, empty, last))
            ]
            aggregate = prepare_retained_cohort_artifact(
                [(str(i), artifact) for i, artifact in enumerate(artifacts)], tmp_path / "merged"
            )
            artifacts.append(aggregate)
            with closing(aggregate.iter_sessions()) as sessions:
                merged = next(sessions)
        else:
            merged = merge_parsed_session_chunks([first, empty, last])[0]
        event = merged.session_events[0]
        assert (event.boundary_start_position, event.boundary_end_position) == (1, 0)
        assert [message.position for message in merged.messages] == [0, 1]
    finally:
        for artifact in reversed(artifacts):
            artifact.discard()


@pytest.mark.parametrize("disk", [False, True], ids=["memory", "disk"])
def test_chunk_position_uses_the_full_variant_coordinate(tmp_path: Path, disk: bool) -> None:
    import sqlite3
    from contextlib import closing

    from polylogue.core.message_owner import MessageOwnerAmbiguityError, MessageOwnerCoordinate
    from polylogue.sources.chunk_positions import ChunkPositions
    from polylogue.sources.parsers.base import ParsedAttachment

    with closing(sqlite3.connect(tmp_path / "scratch.db")) as conn:
        messages = [
            ParsedMessage(
                provider_message_id="",
                role=Role.USER,
                position=9,
                variant_index=v,
                text=str(v),
                owner_coordinate=MessageOwnerCoordinate(position=9, variant_index=v),
            )
            for v in (0, 1)
        ]
        positions = ChunkPositions(messages, 3, conn=conn if disk else None)
        attachment = ParsedAttachment(
            provider_attachment_id="variant",
            message_position=9,
            message_variant_index=1,
            owner_coordinate=MessageOwnerCoordinate(position=9, variant_index=1),
        )
        moved = positions.attachment(attachment)
        assert moved.message_position == 4
        assert moved.owner_coordinate == MessageOwnerCoordinate(position=4, variant_index=1)
        assert moved.acquisition_key == attachment.acquisition_key
        assert positions.message(messages[1], 1).owner_coordinate == moved.owner_coordinate
        with pytest.raises(MessageOwnerAmbiguityError):
            positions.attachment(attachment.model_copy(update={"message_position": 123}))
        ambiguous = ChunkPositions([messages[0], messages[0]], 3, conn=conn if disk else None)
        with pytest.raises(MessageOwnerAmbiguityError):
            ambiguous.position(9)


def test_rebased_attachment_keeps_acquisition_identity_alive() -> None:
    import gc
    import weakref

    from polylogue.sources.chunk_positions import ChunkPositions
    from polylogue.sources.parsers.base import ParsedAttachment

    original = ParsedAttachment(provider_attachment_id="alive", message_position=0)
    original_ref = weakref.ref(original)
    key = original.acquisition_key
    messages = [ParsedMessage(provider_message_id="a", role=Role.USER, text="a", position=0)]
    moved = ChunkPositions(messages, 3).attachment(original)
    del original
    gc.collect()
    assert original_ref() is not None
    assert moved.acquisition_key == key
    again = ChunkPositions([messages[0].model_copy(update={"position": 3})], 7).attachment(moved)
    del moved
    gc.collect()
    assert original_ref() is not None
    assert again.acquisition_key == key
    del again
    gc.collect()
    assert original_ref() is None


def test_prepared_cohort_leaf_is_not_a_revision_storage_default(tmp_path: Path) -> None:
    """The merge chooses the cohort's leaf, so a revision's storage-default
    marker must not ride along and veto the path the cohort leaf implies.

    Anti-vacuity: copying the revision's ``active_leaf_fallback`` onto the
    cohort's last message makes the lowering treat the merge's leaf as a
    storage default and leave both explicit ``False`` paths unset.
    """
    from contextlib import closing

    from polylogue.core.sources import origin_from_provider
    from polylogue.sources.prepared_message_sink import SqliteMessageSink

    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="same",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="first", is_active_path=False)],
    )
    second = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="same",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                parent_message_provider_id="m0",
                role=Role.ASSISTANT,
                text="second",
                is_active_path=False,
                is_active_leaf=True,
                active_leaf_fallback=True,
            )
        ],
    )
    ordered = [
        ("raw-0", _chunk_artifact(tmp_path / "first", first, "c" * 64)),
        ("raw-1", _chunk_artifact(tmp_path / "second", second, "d" * 64)),
    ]

    aggregate = prepare_retained_cohort_artifact(ordered, tmp_path)
    with closing(aggregate.iter_sessions()) as sessions:
        selected = next(sessions)
    messages = list(selected.messages)

    assert [(message.is_active_leaf, message.active_leaf_fallback) for message in messages] == [
        (False, False),
        (True, False),
    ]
    # Parser observations remain intact; publication borrows the separately sealed lowering.
    assert [message.is_active_path for message in messages] == [False, False]
    assert isinstance(selected.messages, SqliteMessageSink)
    normalized = selected.messages.normalized_messages(
        selected.session_events, origin=origin_from_provider(selected.source_name)
    )
    assert [message.is_active_path for message in normalized] == [True, True]


@pytest.mark.parametrize("prepared", [False, True], ids=["memory", "prepared"])
@pytest.mark.parametrize("end_anchor", [False, True], ids=["next-message", "fragment-end"])
def test_real_codex_instructions_anchor_keeps_its_fragment(
    tmp_path: Path,
    prepared: bool,
    end_anchor: bool,
    event_index: sqlite3.Connection,
) -> None:
    from contextlib import closing

    from polylogue.sources.parsers.codex import _codex_instructions_changed_event

    chunks = [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="instruction-fragments",
            messages=[
                ParsedMessage(provider_message_id=name, role=Role.USER, text=name, position=position)
                for position, name in enumerate(names)
            ],
        )
        for names in (("A", "B"), ("C", "D"))
    ]
    chunks[1].session_events.append(
        _codex_instructions_changed_event(
            kind="developer",
            instructions="synthetic second fragment policy",
            revision=1,
            timestamp=None,
            source_index=0,
            effective_from_message_position=2 if end_anchor else 0,
        )
    )
    artifacts = []
    try:
        if prepared:
            artifacts = [
                _chunk_artifact(tmp_path / f"chunk-{index}", chunk, str(index + 1) * 64)
                for index, chunk in enumerate(chunks)
            ]
            aggregate = prepare_retained_cohort_artifact(
                [(str(index), artifact) for index, artifact in enumerate(artifacts)],
                tmp_path / "merged",
            )
            artifacts.append(aggregate)
            with closing(aggregate.iter_sessions()) as sessions:
                merged = next(sessions)
        else:
            merged = merge_parsed_session_chunks(chunks)[0]
        assert [message.provider_message_id for message in merged.messages] == ["A", "B", "C", "D"]
        conn = event_index
        sid = write_fixture_index_session(conn, merged)
        row = conn.execute(
            "SELECT m.position FROM session_events e LEFT JOIN messages m "
            "ON m.message_id = e.boundary_message_id WHERE e.session_id = ?",
            (sid,),
        ).fetchone()
        assert row is not None
        assert row[0] == (None if end_anchor else 2)
    finally:
        for artifact in reversed(artifacts):
            artifact.discard()


@pytest.mark.parametrize("prepared", [False, True], ids=["memory", "prepared"])
def test_repeated_native_event_uses_exact_occurrence_after_composition(
    tmp_path: Path, prepared: bool, event_index: sqlite3.Connection
) -> None:
    from contextlib import closing

    from polylogue.core.message_owner import MessageOwnerCoordinate

    chunks = [
        ParsedSession(
            source_name=Provider.CLAUDE_AI,
            provider_session_id="event-occurrences",
            messages=[
                ParsedMessage(
                    provider_message_id="repeated",
                    role=Role.ASSISTANT,
                    text=text,
                    position=0,
                    owner_coordinate=MessageOwnerCoordinate(stable_key=text, position=0),
                ),
            ],
        )
        for text in ("first occurrence", "second occurrence")
    ]
    chunks[1].session_events.append(
        ParsedSessionEvent(
            event_type="model_configuration",
            source_message_provider_id="repeated",
            payload={"model": "synthetic"},
            owner_coordinate=chunks[1].messages[0].owner_coordinate,
        )
    )
    artifacts = []
    try:
        if prepared:
            artifacts = [
                _chunk_artifact(tmp_path / f"chunk-{index}", chunk, str(index + 1) * 64)
                for index, chunk in enumerate(chunks)
            ]
            aggregate = prepare_retained_cohort_artifact(
                [(str(index), artifact) for index, artifact in enumerate(artifacts)],
                tmp_path / "merged",
            )
            artifacts.append(aggregate)
            with closing(aggregate.iter_sessions()) as sessions:
                merged = next(sessions)
        else:
            merged = merge_parsed_session_chunks(chunks)[0]
        assert merged.session_events[0].owner_coordinate == MessageOwnerCoordinate(
            stable_key="second occurrence", position=1
        )
        conn = event_index
        sid = write_fixture_index_session(conn, merged)
        row = conn.execute(
            "SELECT m.position FROM session_events e JOIN messages m "
            "ON m.message_id = e.source_message_id WHERE e.session_id = ?",
            (sid,),
        ).fetchone()
        assert row is not None and row[0] == 1
    finally:
        for artifact in reversed(artifacts):
            artifact.discard()


@pytest.mark.parametrize("composition", ["unmerged", "memory", "prepared"])
def test_real_claude_repeated_message_events_keep_their_authored_occurrence(
    tmp_path: Path, event_index: sqlite3.Connection, composition: str
) -> None:
    import json
    from contextlib import closing

    from polylogue.sources.parsers.claude.ai_parser import parse_ai
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    records = [
        {
            "uuid": "repeated-native",
            "sender": "assistant",
            "content": [{"type": "text", "text": text}],
            "thinking_config": {"budget_tokens": budget},
        }
        for text, budget in (("first authored occurrence", 101), ("second authored occurrence", 202))
    ]
    envelope = {"uuid": "real-event-occurrences", "name": "Neutral event fragments"}
    artifacts: list[PreparedJsonl] = []
    try:
        if composition == "unmerged":
            merged = parse_ai({**envelope, "chat_messages": records}, "fallback")
        else:
            chunks = [parse_ai({**envelope, "chat_messages": [record]}, "fallback") for record in records]
            if composition == "memory":
                merged = merge_parsed_session_chunks(chunks)[0]
            else:
                artifacts = [
                    _chunk_artifact(tmp_path / f"real-chunk-{i}", chunk, str(i + 1) * 64)
                    for i, chunk in enumerate(chunks)
                ]
                aggregate = prepare_retained_cohort_artifact(
                    [(str(i), artifact) for i, artifact in enumerate(artifacts)], tmp_path / "real-merged"
                )
                artifacts.append(aggregate)
                with closing(aggregate.iter_sessions()) as sessions:
                    merged = next(sessions)
        # These are original Claude provider objects, not admitted replacement events.
        assert len(merged.messages) == 2
        assert all(
            event.owner_coordinate is not None
            for event in merged.session_events
            if event.event_type == "model_configuration"
        )
        sid = write_fixture_index_session(event_index, merged)
        with closing(open_readonly_connection(tmp_path / "archive" / "index.db")) as reopened:
            rows = reopened.execute(
                "SELECT b.text,e.payload_json FROM session_events e "
                "JOIN blocks b ON b.message_id=e.source_message_id "
                "WHERE e.session_id=? AND e.event_type='model_configuration'",
                (sid,),
            ).fetchall()
            actual = {row[0]: json.loads(row[1])["thinking"]["budget_tokens"] for row in rows}
            assert len(rows) == 2
            assert actual == {"first authored occurrence": 101, "second authored occurrence": 202}
    finally:
        for artifact in reversed(artifacts):
            artifact.discard()
