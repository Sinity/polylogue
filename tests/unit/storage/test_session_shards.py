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

import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.identity_law import session_id as archive_session_id
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    bind_session_shard,
    prepare_session_rows,
    prepare_session_shard,
    write_parsed_session_to_archive,
)
from polylogue.storage.sqlite.archive_tiers.write_shard import (
    SessionShardBuilder,
    ShardRefusedError,
    attached_session_shard,
    build_session_shard,
    open_session_shard,
    shard_column_signature,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _connect(path: Path) -> sqlite3.Connection:
    # ``uri=True`` matches the archive's own write connection: it is what
    # lets the shard be ATTACHed read-only.
    conn = sqlite3.connect(path, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


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
        write_parsed_session_to_archive(conn, session, content_hash=str(session_content_hash(session)))


def _session_key(session: ParsedSession) -> str:
    return archive_session_id(origin_from_provider(session.source_name).value, session.provider_session_id)


def _copy_from_shard(conn: sqlite3.Connection, sessions: list[ParsedSession], shard_path: Path) -> None:
    """The writer's half: attach a sealed shard and write each of its sessions."""
    shard = open_session_shard(shard_path)
    with attached_session_shard(conn, shard) as schema:
        bindings = bind_session_shard(schema, shard)
        for session in sessions:
            write_parsed_session_to_archive(
                conn,
                session,
                content_hash=str(session_content_hash(session)),
                prepared=bindings[_session_key(session)],
            )


def _write_through_shard(conn: sqlite3.Connection, sessions: list[ParsedSession], directory: Path) -> None:
    _copy_from_shard(conn, sessions, prepare_session_shard(directory, sessions).path)


def test_shard_and_inline_writes_produce_identical_rows(tmp_path: Path) -> None:
    sessions = _synthetic_sessions()
    inline_conn = _connect(tmp_path / "inline.db")
    shard_conn = _connect(tmp_path / "shard-written.db")
    try:
        _write_inline(inline_conn, sessions)
        _write_through_shard(shard_conn, sessions, tmp_path / "shards")

        for table, order_by in (("sessions", "session_id"), ("messages", "message_id"), ("blocks", "block_id")):
            assert _dump_table(inline_conn, table, order_by) == _dump_table(shard_conn, table, order_by), table
        assert _dump_table(inline_conn, "messages", "message_id"), "corpus wrote no messages"
    finally:
        inline_conn.close()
        shard_conn.close()


def test_shard_copy_replaces_the_row_binding_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: with the binding writers poisoned, only a real copy can succeed."""
    sessions = _synthetic_sessions()[:2]
    # The shard is built first, by the stage-A half, which legitimately uses
    # the row builders. Only the writer's half is poisoned.
    shard_path = prepare_session_shard(tmp_path / "shards", sessions).path

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("a shard write must not build or bind rows on the writer thread")

    monkeypatch.setattr(archive_tier_write, "_write_messages", _boom)
    monkeypatch.setattr(archive_tier_write, "_write_blocks", _boom)
    monkeypatch.setattr(archive_tier_write, "_build_message_rows", _boom)
    monkeypatch.setattr(archive_tier_write, "_build_block_rows", _boom)

    conn = _connect(tmp_path / "index.db")
    try:
        _copy_from_shard(conn, sessions, shard_path)
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] > 0
    finally:
        conn.close()


def test_shard_copy_preserves_search_text_and_fts(tmp_path: Path) -> None:
    """The copy feeds the same generated columns and FTS surfaces as an insert."""
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


def test_shard_falls_back_when_the_session_already_has_rows(tmp_path: Path) -> None:
    """A rewrite has prior rows to reconcile, which a shard cannot carry.

    The second write must land the new content anyway, through the inline
    builders, exactly as ``prepared=None`` would.
    """
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
        reopened = open_session_shard(shard.path)
        with attached_session_shard(conn, reopened) as schema:
            bindings = bind_session_shard(schema, reopened)
            write_parsed_session_to_archive(
                conn,
                mutated,
                content_hash=str(session_content_hash(mutated)),
                prepared=bindings[_session_key(original)],
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
        from polylogue.storage.sqlite.archive_tiers.write_shard import SessionShardBuilder

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
def test_a_shard_from_a_killed_worker_is_never_attached(tmp_path: Path) -> None:
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
    builder._conn.close()

    with pytest.raises(ShardRefusedError, match="unsealed"):
        open_session_shard(tmp_path / "unsealed.db")


def test_a_shard_built_against_other_columns_is_refused(tmp_path: Path) -> None:
    shard = build_session_shard(tmp_path / "shards", [prepare_session_rows(_synthetic_sessions()[0])])
    with sqlite3.connect(shard.path) as conn:
        conn.execute("UPDATE shard_seal SET column_signature = 'a-build-with-other-columns'")
    with pytest.raises(ShardRefusedError, match="column signature"):
        open_session_shard(shard.path)
    assert shard_column_signature() != "a-build-with-other-columns"


def test_a_shard_with_two_entries_for_one_session_is_refused(tmp_path: Path) -> None:
    """Bindings are keyed by session id, so a duplicate would address the wrong rows."""
    session = _synthetic_sessions()[0]
    shard = build_session_shard(tmp_path / "shards", [prepare_session_rows(session), prepare_session_rows(session)])
    with sqlite3.connect(shard.path) as conn:
        conn.execute("UPDATE shard_seal SET session_count = 2")
    with pytest.raises(ShardRefusedError, match="twice"):
        open_session_shard(shard.path)


def test_a_shard_cannot_be_attached_inside_a_transaction(tmp_path: Path) -> None:
    """SQLite will not detach a database a live transaction read, so entering
    inside one would strand the attachment. Refuse at the door instead."""
    shard = build_session_shard(tmp_path / "shards", [prepare_session_rows(_synthetic_sessions()[0])])
    conn = _connect(tmp_path / "index.db")
    try:
        conn.execute("BEGIN")
        with pytest.raises(ShardRefusedError, match="never inside one"):
            with attached_session_shard(conn, open_session_shard(shard.path)):
                pass
        conn.execute("ROLLBACK")
    finally:
        conn.close()


def test_an_attached_shard_cannot_be_written(tmp_path: Path) -> None:
    """The transport is read-only to the writer, enforced by SQLite."""
    shard = build_session_shard(tmp_path / "shards", [prepare_session_rows(_synthetic_sessions()[0])])
    conn = _connect(tmp_path / "index.db")
    try:
        with attached_session_shard(conn, open_session_shard(shard.path)) as schema:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute(f"DELETE FROM {schema}.messages")
    finally:
        conn.close()
