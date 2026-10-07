"""polylogue-bp12n.6: the stage-A shard the writer bulk-copies.

A shard moves both row construction and per-row parameter binding off the
writer thread: the parse worker inserts its tuples into a private, index-free
SQLite file and the writer copies them with ``INSERT ... SELECT``. These
tests hold the two properties that make that safe.

  1. **The rows are the same rows.** Writing a corpus through the shard and
     through the inline builder must leave byte-identical ``sessions`` /
     ``messages`` / ``blocks`` tables, and the copy must genuinely replace
     the binding loop rather than run beside it.
  2. **A partial shard is not a shard.** A builder killed mid-transaction
     leaves a file with data in it and no seal; it is refused, and no row of
     it reaches the archive.

Anti-vacuity for (1): monkeypatching ``_write_messages``/``_write_blocks`` to
raise leaves the shard write green and every other write red. For (2):
sealing before the kill, or dropping the seal check in
``open_session_shard``, makes the refusal test red.
"""

from __future__ import annotations

import gc
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import MessageOwnerResolution, session_content_hash
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    prepare_session_rows,
    prepare_session_shard,
    prepared_session_rows_from_shard,
)
from polylogue.storage.sqlite.session_shard import (
    SessionShardBuilder,
    ShardRefusedError,
    build_session_shard,
    open_session_shard,
    shard_column_signature,
)
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.revision_backfill_benchmark import build_large_parent_shared_prefix_sessions

REPO_ROOT = Path(__file__).resolve().parents[3]


def _connect(path: Path) -> sqlite3.Connection:
    # ``uri=True`` matches the archive's own write connection: it is what
    # lets the shard be ATTACHed read-only.
    conn = connect_measured(path, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _archive_index(root: Path) -> Path:
    """Give each comparison archive its own root: a root owns one active Index."""
    root.mkdir()
    return root / "index.db"


def _synthetic_sessions() -> list[ParsedSession]:
    return [
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="plain-text",
            title="Plain text session",
            messages=[
                ParsedMessage(
                    provider_message_id="m0",
                    role=Role.USER,
                    text="hello there, this is a synthetic corpus message",
                    material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    position=0,
                ),
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    text="a reply with unicode: café — 你好",
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    position=1,
                ),
            ],
        ),
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="tool-use-and-thinking",
            title="Tool use and thinking",
            messages=[
                ParsedMessage(
                    provider_message_id="t0",
                    role=Role.ASSISTANT,
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    position=0,
                    blocks=[
                        ParsedContentBlock(type=BlockType.THINKING, text="let me think about this"),
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Bash",
                            tool_id="call-1",
                            tool_input={"command": "pytest -q"},
                        ),
                    ],
                ),
                ParsedMessage(
                    provider_message_id="t1",
                    role=Role.TOOL,
                    material_origin=MaterialOrigin.TOOL_RESULT,
                    position=1,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_RESULT,
                            tool_id="call-1",
                            text="1 passed",
                            is_error=False,
                            exit_code=0,
                        ),
                    ],
                ),
            ],
        ),
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="numeric-looking-identifiers",
            title="Values a column affinity would rewrite",
            messages=[
                # ``native_id`` "0042" survives only because the shard's
                # columns carry no declared type: an INTEGER affinity would
                # store 42 and change this message's identity.
                ParsedMessage(
                    provider_message_id="0042",
                    role=Role.USER,
                    text="a leading-zero native id",
                    material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    position=0,
                ),
                ParsedMessage(
                    provider_message_id="1e3",
                    role=Role.ASSISTANT,
                    text="an exponent-shaped native id",
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    position=1,
                ),
            ],
        ),
        ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="duplicate-native-ids",
            title="Duplicate native ids",
            messages=[
                ParsedMessage(
                    provider_message_id="dup",
                    role=Role.USER,
                    text="first with a colliding native id",
                    position=0,
                    variant_index=0,
                ),
                ParsedMessage(
                    provider_message_id="dup",
                    role=Role.USER,
                    text="second with a colliding native id",
                    position=1,
                    variant_index=1,
                ),
            ],
        ),
    ]


def _dump_table(conn: sqlite3.Connection, table: str, order_by: str) -> list[tuple[object, ...]]:
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
    return [tuple(row) for row in rows]


def _write_inline(conn: sqlite3.Connection, sessions: list[ParsedSession]) -> None:
    for session in sessions:
        write_fixture_index_session(conn, session, content_hash=str(session_content_hash(session)))


def _session_key(session: ParsedSession) -> str:
    return archive_session_id(origin_from_provider(session.source_name).value, session.provider_session_id)


def _write_from_shard(conn: sqlite3.Connection, sessions: list[ParsedSession], shard_path: Path) -> None:
    """The writer's half: bind each session's sealed shard rows as its prepared carrier."""
    for session in sessions:
        write_fixture_index_session(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            prepared_rows=prepared_session_rows_from_shard(shard_path, _session_key(session)),
        )


def _write_through_shard(conn: sqlite3.Connection, sessions: list[ParsedSession], directory: Path) -> None:
    _write_from_shard(conn, sessions, prepare_session_shard(directory, sessions).path)


def test_shard_and_inline_writes_produce_identical_rows(tmp_path: Path) -> None:
    sessions = _synthetic_sessions()
    inline_conn = _connect(_archive_index(tmp_path / "inline"))
    shard_conn = _connect(_archive_index(tmp_path / "shard-written"))
    try:
        _write_inline(inline_conn, sessions)
        _write_through_shard(shard_conn, sessions, tmp_path / "shards")

        for table, order_by in (("sessions", "session_id"), ("messages", "message_id"), ("blocks", "block_id")):
            assert _dump_table(inline_conn, table, order_by) == _dump_table(shard_conn, table, order_by), table
        assert _dump_table(inline_conn, "messages", "message_id"), "corpus wrote no messages"
    finally:
        inline_conn.close()
        shard_conn.close()


def test_sealed_message_sink_replaces_same_raw_from_streamed_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="streamed-reparse",
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="old")],
    )
    revised = original.model_copy(
        update={
            "messages": [
                ParsedMessage(provider_message_id="m0", role=Role.USER, text="new"),
                ParsedMessage(provider_message_id="m1", role=Role.ASSISTANT, text="answer"),
            ]
        }
    )
    store = SqliteMessageStore(tmp_path / "prepared.db")
    sink = store.new_sink()
    sink.extend(revised.messages)
    worker_session = revised.model_copy(update={"messages": sink, "content_hash": str(session_content_hash(revised))})
    shard = prepare_session_shard(tmp_path / "shards", [worker_session])
    store.conn.commit()
    store.close()
    sealed = SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink))
    publication = worker_session.model_copy(update={"messages": sealed})

    conn = _connect(tmp_path / "index.db")
    try:
        write_fixture_index_session(conn, original, raw_id="same-acquisition")
        prepared = prepared_session_rows_from_shard(shard.path, _session_key(publication))

        def forbid_inline(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("writer rebuilt canonical rows instead of binding the shard rows")

        monkeypatch.setattr(archive_tier_write, "_iter_message_rows", forbid_inline)
        monkeypatch.setattr(archive_tier_write, "_iter_block_rows", forbid_inline)
        write_fixture_index_session(
            conn,
            publication,
            content_hash=publication.content_hash,
            raw_id="same-acquisition",
            prepared_rows=prepared,
        )
        rows = conn.execute(
            "SELECT native_id, text FROM messages JOIN blocks USING (message_id) "
            "WHERE messages.session_id = ? ORDER BY messages.position",
            (_session_key(publication),),
        ).fetchall()
        assert [tuple(row) for row in rows] == [("m0", "new"), ("m1", "answer")]
    finally:
        conn.close()


def test_sealed_message_sink_preserves_attachment_owner_projection(tmp_path: Path) -> None:
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="streamed-attachment-owner",
        messages=[
            ParsedMessage(provider_message_id="m0", role=Role.USER, text="request", position=0),
            ParsedMessage(provider_message_id="m1", role=Role.ASSISTANT, text="answer", position=1),
        ],
        attachments=[
            ParsedAttachment(
                provider_attachment_id="attachment-1",
                message_position=1,
                name="answer.txt",
                mime_type="text/plain",
            )
        ],
    )
    store = SqliteMessageStore(tmp_path / "attachment-prepared.db")
    sink = store.new_sink()
    sink.extend(session.messages)
    worker_session = session.model_copy(update={"messages": sink, "content_hash": str(session_content_hash(session))})
    shard = prepare_session_shard(tmp_path / "attachment-shards", [worker_session])
    store.conn.commit()
    store.close()
    sealed = SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink))
    publication = worker_session.model_copy(update={"messages": sealed})

    inline = _connect(_archive_index(tmp_path / "inline-attachment"))
    streamed = _connect(_archive_index(tmp_path / "streamed-attachment"))
    try:
        write_fixture_index_session(inline, session, content_hash=str(session_content_hash(session)))
        write_fixture_index_session(
            streamed,
            publication,
            content_hash=publication.content_hash,
            prepared_rows=prepared_session_rows_from_shard(shard.path, _session_key(publication)),
        )
        for table, order_by in (
            ("sessions", "session_id"),
            ("messages", "message_id"),
            ("blocks", "block_id"),
            ("attachments", "attachment_id"),
            ("attachment_refs", "message_id, position"),
        ):
            assert _dump_table(inline, table, order_by) == _dump_table(streamed, table, order_by), table
        assert streamed.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0] == 1
    finally:
        inline.close()
        streamed.close()


def test_shard_rows_replace_the_writer_row_builders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: with the row builders poisoned, only the sealed rows can be written."""
    sessions = _synthetic_sessions()[:2]
    # The shard is built first, by the stage-A half, which legitimately uses
    # the row builders. Only the writer's half is poisoned.
    shard_path = prepare_session_shard(tmp_path / "shards", sessions).path

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("a shard-carried write must not build rows on the writer thread")

    monkeypatch.setattr(archive_tier_write, "_iter_message_rows", _boom)
    monkeypatch.setattr(archive_tier_write, "_iter_block_rows", _boom)
    monkeypatch.setattr(archive_tier_write, "_build_message_rows", _boom)
    monkeypatch.setattr(archive_tier_write, "_build_block_rows", _boom)

    conn = _connect(tmp_path / "index.db")
    try:
        _write_from_shard(conn, sessions, shard_path)
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] > 0
    finally:
        conn.close()


def test_shard_rows_reuse_carried_identities_without_writer_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete shard supplies every identity needed by the writer."""
    sessions = _synthetic_sessions()[:2]
    shard_path = prepare_session_shard(tmp_path / "shards", sessions).path

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("complete shard writes must not recompute identities")

    monkeypatch.setattr(archive_tier_write, "message_content_identities", _boom)
    conn = _connect(tmp_path / "index.db")
    try:
        _write_from_shard(conn, sessions, shard_path)
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4
    finally:
        conn.close()


def test_shard_rows_preserve_search_text_and_fts(tmp_path: Path) -> None:
    """Shard-carried rows feed the same generated columns and FTS surfaces as an insert."""
    sessions = _synthetic_sessions()
    conn = _connect(tmp_path / "index.db")
    try:
        _write_through_shard(conn, sessions, tmp_path / "shards")
        hits = conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?", ("synthetic",)).fetchone()[
            0
        ]
        assert hits > 0
        generated = conn.execute(
            "SELECT block_id, search_text FROM blocks WHERE block_type = 'text' LIMIT 1"
        ).fetchone()
        assert generated["block_id"].count(":") >= 3
        assert generated["search_text"]
    finally:
        conn.close()


def test_shard_replaces_prior_rows_from_the_same_acquisition(tmp_path: Path) -> None:
    """A same-acquisition reparse replaces the old rows from the shard."""
    original = _synthetic_sessions()[0]
    conn = _connect(tmp_path / "index.db")
    try:
        _write_inline(conn, [original])
        rewritten = original.model_copy(
            update={
                "messages": [
                    ParsedMessage(
                        provider_message_id="m0",
                        role=Role.USER,
                        text="the rewritten body",
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                        position=0,
                    )
                ]
            }
        )
        _write_through_shard(conn, [rewritten], tmp_path / "shards")
        envelope = archive_tier_write.read_archive_session_envelope(conn, _session_key(original))
        texts = ["".join(block.text or "" for block in message.blocks) for message in envelope.messages]
        assert texts == ["the rewritten body"]
    finally:
        conn.close()


def test_shared_prefix_prior_rows_reconcile_shard_rows_and_preserve_finished_projection(tmp_path: Path) -> None:
    """A bounded parent/child witness reconciles a prior child rewrite.

    The child has a sizeable inherited prefix and already has stored rows when
    its changed tail arrives. The writer reconciles the sealed rows against
    the stored ones, and the completed logical and FTS projections still
    match the explicit inline control.
    """
    *children, parent = build_large_parent_shared_prefix_sessions()
    child = children[0]
    rewritten = child.model_copy(
        update={
            "messages": [
                *child.messages,
                ParsedMessage(
                    provider_message_id="measurement-fallback-tail",
                    role=Role.ASSISTANT,
                    text="measurement fallback tail after prior session rows",
                    material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
                    position=len(child.messages),
                ),
            ]
        }
    )
    control = _connect(_archive_index(tmp_path / "inline-control"))
    witness = _connect(_archive_index(tmp_path / "shard-fallback-witness"))
    try:
        _write_inline(control, [parent, child, rewritten])
        _write_inline(witness, [parent, child])
        _write_through_shard(witness, [rewritten], tmp_path / "shared-prefix-shards")

        for table, order_by in (("sessions", "session_id"), ("messages", "message_id"), ("blocks", "block_id")):
            assert _dump_table(control, table, order_by) == _dump_table(witness, table, order_by), table
        assert control.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'measurement'"
        ).fetchone() == (
            witness.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'measurement'").fetchone()
        )
    finally:
        control.close()
        witness.close()


def test_stale_shard_content_hash_falls_back_to_fresh_content(tmp_path: Path) -> None:
    original = _synthetic_sessions()[0]
    shard = prepare_session_shard(tmp_path / "shards", [original])
    mutated = original.model_copy(
        update={
            "messages": [
                ParsedMessage(
                    provider_message_id="m0",
                    role=Role.USER,
                    text="a DIFFERENT body -- the raw changed before the writer ran",
                    material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    position=0,
                )
            ]
        }
    )
    conn = _connect(tmp_path / "index.db")
    try:
        write_fixture_index_session(
            conn,
            mutated,
            content_hash=str(session_content_hash(mutated)),
            prepared_rows=prepared_session_rows_from_shard(shard.path, _session_key(original)),
        )
        envelope = archive_tier_write.read_archive_session_envelope(conn, _session_key(original))
        texts = ["".join(block.text or "" for block in message.blocks) for message in envelope.messages]
        assert texts == ["a DIFFERENT body -- the raw changed before the writer ran"]
    finally:
        conn.close()


def _kill_a_builder_mid_write(tmp_path: Path) -> Path:
    """Run a builder in a child process and SIGKILL it before it seals."""
    shard_path = tmp_path / "killed.db"
    ready = tmp_path / "ready"
    child = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from pathlib import Path
        from polylogue.archive.message.roles import Role
        from polylogue.core.enums import MaterialOrigin, Provider
        from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
        from polylogue.storage.sqlite.archive_tiers.write import prepare_session_rows
        from polylogue.storage.sqlite.session_shard import SessionShardBuilder

        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="killed-mid-write",
            title="Killed mid write",
            messages=[
                ParsedMessage(
                    provider_message_id=f"m{{i}}",
                    role=Role.USER,
                    text="x" * 3000,
                    material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    position=i,
                )
                for i in range(4000)
            ],
        )
        builder = SessionShardBuilder(Path({str(shard_path)!r}))
        builder.add(prepare_session_rows(session))
        Path({str(ready)!r}).write_text("rows written, seal pending")
        time.sleep(120)
        """
    )
    process = subprocess.Popen([sys.executable, "-c", child])
    try:
        deadline = 300
        while deadline and not ready.exists() and process.poll() is None:
            deadline -= 1
            import time as _time

            _time.sleep(0.2)
        assert ready.exists(), "child never reported its rows written"
        os.kill(process.pid, signal.SIGKILL)
    finally:
        process.wait(timeout=30)
    return shard_path


@pytest.mark.slow
def test_a_shard_from_a_killed_worker_is_never_opened(tmp_path: Path) -> None:
    shard_path = _kill_a_builder_mid_write(tmp_path)

    # The kill landed with real data on disk, not before the first page.
    on_disk = shard_path.stat().st_size + sum(
        candidate.stat().st_size for candidate in tmp_path.glob("killed.db-journal")
    )
    assert on_disk > 1_000_000, f"child died before spilling anything ({on_disk} bytes)"

    with pytest.raises(ShardRefusedError):
        open_session_shard(shard_path)

    conn = _connect(tmp_path / "index.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 0
    finally:
        conn.close()


def test_an_unsealed_shard_is_refused(tmp_path: Path) -> None:
    """The same refusal without the kill: rows present, seal absent."""
    builder = SessionShardBuilder(tmp_path / "unsealed.db")
    builder.add(prepare_session_rows(_synthetic_sessions()[0]))
    builder._conn.execute("COMMIT")  # the rows land; the seal never does
    # Keep the unsealed file for refusal, but retire its registered SQL owner.
    builder._discard_on_close = False
    builder.close()

    with pytest.raises(ShardRefusedError, match="unsealed"):
        open_session_shard(tmp_path / "unsealed.db")


_EMPTY_OWNER_RESOLUTION = MessageOwnerResolution((), {}, frozenset(), {}, frozenset(), frozenset(), {}, frozenset())


def test_many_small_sessions_keep_manifest_off_python_heap(tmp_path: Path) -> None:
    builder = SessionShardBuilder(tmp_path / "many.db")

    def add(index: int) -> None:
        builder.add(
            SimpleNamespace(
                session_id=f"session-{index}",
                session_content_hash=b"x" * 32,
                message_rows=(),
                block_rows=(),
                content_identities=(),
                owner_resolution=_EMPTY_OWNER_RESOLUTION,
            )
        )

    tracemalloc.start()
    try:
        for index in range(250):
            add(index)
        gc.collect()
        baseline, _ = tracemalloc.get_traced_memory()
        for index in range(250, 8000):
            add(index)
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
        assert retained - baseline < 1_000_000
        shard = open_session_shard(builder.seal().path)
        gc.collect()
        opened, _ = tracemalloc.get_traced_memory()
        assert opened - baseline < 1_000_000
        assert len(shard.sessions) == 8000
        assert shard.sessions[7999].session_id == "session-7999"
        assert shard.by_session_id()["session-4000"].session_id == "session-4000"
    finally:
        tracemalloc.stop()


def test_a_shard_built_against_other_columns_is_refused(tmp_path: Path) -> None:
    shard = build_session_shard(tmp_path / "shards", [prepare_session_rows(_synthetic_sessions()[0])])
    with sqlite3.connect(shard.path) as conn:
        conn.execute("UPDATE shard_seal SET column_signature = 'a-build-with-other-columns'")
    with pytest.raises(ShardRefusedError, match="column signature"):
        open_session_shard(shard.path)
    assert shard_column_signature() != "a-build-with-other-columns"


def test_a_shard_preserves_duplicate_ordinals_but_refuses_ambiguous_writer_binding(tmp_path: Path) -> None:
    """The parser carrier retains outputs; an identity cannot select either duplicate range."""
    session = _synthetic_sessions()[0]
    built = build_session_shard(tmp_path / "shards", [prepare_session_rows(session), prepare_session_rows(session)])
    shard = open_session_shard(built.path)
    assert len(shard.sessions) == 2
    first, second = shard.sessions
    assert first.session_id == second.session_id
    assert first.message_lo != second.message_lo
    with pytest.raises(ShardRefusedError, match="twice"):
        shard.by_session_id()[first.session_id]
    from polylogue.storage.sqlite.archive_tiers.write import prepared_session_rows_from_shard

    with pytest.raises(ShardRefusedError, match="twice"):
        prepared_session_rows_from_shard(shard.path, first.session_id)
