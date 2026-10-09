"""polylogue-623q: ``prepare_session_rows`` extraction equivalence.

Row PREPARATION (converting a ``ParsedSession`` tree into the message/block
SQL row tuples the full-replace write path inserts) was mechanically
extracted from the writer-hold-only ``_write_messages``/``_write_blocks``
row-building loops into a pure ``prepare_session_rows`` function that can run
off the writer thread (e.g. a daemon parse-prefetch worker). These tests
prove:

  1. equivalence -- writing via the inline builder and via a pre-prepared
     ``PreparedSessionRows`` produce byte-identical ``sessions``/``messages``/
     ``blocks`` rows for a corpus of synthetic sessions covering text,
     tool_use/tool_result, thinking, paste, and duplicate-native-id messages;
  2. genuine reuse -- the write path never rebuilds rows from ``messages``
     when a valid ``prepared`` is supplied (proven by monkeypatching the
     inline builders to raise);
  3. stale-prepared fallback -- when the session's content changed after
     ``prepare_session_rows`` ran (a mismatched content hash), the writer
     ignores the stale prepared rows and builds fresh ones reflecting the
     NEW content, not the stale one.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Origin, Provider
from polylogue.pipeline import ids as pipeline_ids
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import (
    PreparedSessionRows,
    PreparedSessionWrite,
    PreparedSessionWriteRefusedError,
    prepare_session_rows,
    prepare_session_shard,
    prepare_session_write,
    prepared_session_rows_from_shard,
    read_archive_session_envelope,
)
from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError
from tests.infra.index_writer import (
    fixture_index_mutation_scope,
    write_fixture_index_session,
    write_fixture_prepared_session,
)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_measured(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_disk_duplicate_native_ids_cross_thread_lookup_and_cleanup(tmp_path: Path) -> None:
    """A prepared disk lookup moves to the writer, then its owner releases it."""
    store = SqliteMessageStore(tmp_path / "duplicate-source.db")
    sink = store.new_sink()
    sink.extend(
        [
            ParsedMessage(provider_message_id="dup", role=Role.USER, text="first"),
            ParsedMessage(provider_message_id=" dup ", role=Role.USER, text="second"),
            ParsedMessage(provider_message_id="dup", role=Role.USER, text="fourth"),
            ParsedMessage(provider_message_id="unique", role=Role.USER, text="third"),
        ]
    )
    store.conn.commit()
    store.close()
    sealed = SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink))
    duplicates = archive_tier_write._duplicate_message_native_ids(sealed)
    assert isinstance(duplicates, archive_tier_write._DiskDuplicateNativeIds)
    scratch = Path(duplicates._scratch.name)

    with ThreadPoolExecutor(max_workers=1) as writer, ThreadPoolExecutor(max_workers=1) as owner:
        assert writer.submit(
            lambda: ("n:dup" in duplicates, "n:unique" in duplicates, len(duplicates), list(duplicates))
        ).result() == (
            True,
            False,
            1,
            ["n:dup"],
        )
        owner.submit(duplicates.close).result()
    duplicates.close()
    assert not scratch.exists()


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


def _write_all(conn: sqlite3.Connection, sessions: list[ParsedSession], *, prepared: bool) -> None:
    for session in sessions:
        chash = str(session_content_hash(session))
        rows = prepare_session_rows(session) if prepared else None
        write_fixture_index_session(conn, session, content_hash=chash, prepared_rows=rows)


def test_prepared_and_inline_writes_produce_identical_rows(tmp_path: Path) -> None:
    sessions = _synthetic_sessions()

    # Each comparison archive owns its own active Index.
    inline_conn = _connect(tmp_path / "inline" / "index.db")
    prepared_conn = _connect(tmp_path / "prepared" / "index.db")
    try:
        _write_all(inline_conn, sessions, prepared=False)
        _write_all(prepared_conn, sessions, prepared=True)

        assert _dump_table(inline_conn, "sessions", "session_id") == _dump_table(
            prepared_conn, "sessions", "session_id"
        )
        assert _dump_table(inline_conn, "messages", "message_id") == _dump_table(
            prepared_conn, "messages", "message_id"
        )
        assert _dump_table(inline_conn, "blocks", "block_id") == _dump_table(prepared_conn, "blocks", "block_id")
    finally:
        inline_conn.close()
        prepared_conn.close()


def test_valid_prepared_rows_are_used_verbatim_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: monkeypatch the inline row builders to raise, then prove
    a matching-content-hash write still succeeds and produces correct rows --
    the only way that can happen is if the write path used the prepared rows
    verbatim instead of calling ``_build_message_rows``/``_build_block_rows``.
    """
    session = _synthetic_sessions()[1]  # tool-use-and-thinking: exercises both tables
    prepared = prepare_session_rows(session)

    conn = _connect(tmp_path / "index.db")
    try:

        def _boom(*args: object, **kwargs: object) -> object:
            raise AssertionError("inline row builder must not run when valid prepared rows are supplied")

        monkeypatch.setattr(archive_tier_write, "_build_message_rows", _boom)
        monkeypatch.setattr(archive_tier_write, "_build_block_rows", _boom)

        session_id = write_fixture_index_session(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            prepared_rows=prepared,
        )

        stored_messages = conn.execute(
            "SELECT native_id, has_tool_use FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()
        assert [row[0] for row in stored_messages] == ["t0", "t1"]
        stored_blocks = conn.execute(
            "SELECT block_type, tool_name FROM blocks b JOIN messages m ON m.message_id = b.message_id "
            "WHERE m.session_id = ? ORDER BY m.position, b.position",
            (session_id,),
        ).fetchall()
        assert ("tool_use", "Bash") in [tuple(row) for row in stored_blocks]
    finally:
        conn.close()


def test_parse_bound_hash_is_carried_into_prepared_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepared rows consume the worker's digest without a second tree hash."""
    session = _synthetic_sessions()[0]
    expected = str(session_content_hash(session))
    bound = session.model_copy(update={"content_hash": expected})

    def _boom(*args: object, **kwargs: object) -> str:
        raise AssertionError("prepared rows must use the parse-bound hash")

    monkeypatch.setattr(pipeline_ids, "session_content_hash", _boom)
    prepared = prepare_session_rows(bound)

    assert prepared.session_content_hash.hex() == expected


def test_writer_accepts_parse_bound_hash_without_recomputing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bound hash is sufficient for the writer's prepared-row admission."""
    session = _synthetic_sessions()[0]
    bound = session.model_copy(update={"content_hash": str(session_content_hash(session))})
    prepared = prepare_session_rows(bound)

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("writer must not rebuild prepared rows")

    monkeypatch.setattr(archive_tier_write, "_build_message_rows", _boom)
    monkeypatch.setattr(archive_tier_write, "_build_block_rows", _boom)
    conn = _connect(tmp_path / "bound.db")
    try:
        session_id = write_fixture_index_session(conn, bound, prepared_rows=prepared)
        assert conn.execute("SELECT content_hash FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
    finally:
        conn.close()


def test_valid_identity_carrier_is_reused_without_writer_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid prepared carrier is the writer's identity source."""
    session = _synthetic_sessions()[0]
    prepared = prepare_session_rows(session)

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("writer must reuse the prepared identity carrier")

    monkeypatch.setattr(archive_tier_write, "message_content_identities", _boom)
    conn = _connect(tmp_path / "carrier.db")
    try:
        write_fixture_index_session(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            prepared_rows=prepared,
        )
    finally:
        conn.close()


def test_corrupt_identity_carrier_is_refused(tmp_path: Path) -> None:
    """A carrier disagreement is refused instead of regenerated on the writer."""
    session = _synthetic_sessions()[0]
    prepared = prepare_session_rows(session)
    corrupt = replace(
        prepared,
        content_identities=(("corrupt-identity", 0), *prepared.content_identities[1:]),
    )
    conn = _connect(tmp_path / "corrupt-carrier.db")
    try:
        with pytest.raises(PreparedSessionWriteRefusedError, match="disagrees with message rows"):
            write_fixture_index_session(
                conn,
                session,
                content_hash=str(session_content_hash(session)),
                prepared_rows=corrupt,
            )
    finally:
        conn.close()


def test_prepared_rows_match_identity_golden_fixture() -> None:
    """Freeze the prepared session/message/block identities for a fixed tree."""
    session = _synthetic_sessions()[1]
    prepared = prepare_session_rows(session)

    assert prepared.session_id == "codex-session:tool-use-and-thinking"
    assert prepared.session_content_hash.hex() == "b388bae80e73def1b0c39c892a27ab887bca6b7b65128e823ae698025e54123f"
    assert [(row[0], row[1], cast(bytes, row[30]).hex()) for row in prepared.message_rows] == [
        (
            "codex-session:tool-use-and-thinking",
            "t0",
            "57c72bee62c3ca3cc1fb517d1cafb1263d6e5bef6eb62389745f7b31b6b22794",
        ),
        (
            "codex-session:tool-use-and-thinking",
            "t1",
            "44ddaa18fd9e840cff5d611079f6d03f10d5e7a37d29b563c42053fa573f419a",
        ),
    ]
    assert [(row[0], cast(bytes, row[-1]).hex()) for row in prepared.block_rows] == [
        (
            "codex-session:tool-use-and-thinking:n:t0",
            "fe379f72e342a10bc02ffa074034323b87118b6a74bfc474cb3196c9e9cba039",
        ),
        (
            "codex-session:tool-use-and-thinking:n:t0",
            "9c941f49d308dcdf951a4d31ceabedc2b367bee3c7c2355604d3f6e8b60a565c",
        ),
        (
            "codex-session:tool-use-and-thinking:n:t1",
            "fc2514fe847eba2a6c7a6cbbff179087d85c0fa8d80a974afaa3808ee95d6f51",
        ),
    ]


def test_seeded_corpus_stores_identity_golden_fixture(tmp_path: Path) -> None:
    """Freeze production-stored ids and session digests for the seeded corpus.

    The corpus passes through parse-side preparation and the real SQLite writer;
    the readback is deliberately a literal fixture, not a second identity
    implementation.  It fails if canonical bytes drift (or if any fixture
    value is corrupted), including for nested tool input and non-ASCII text.
    """
    conn = _connect(tmp_path / "seeded-identity.db")
    try:
        for session in _synthetic_sessions():
            prepared = prepare_session_rows(session)
            write_fixture_index_session(
                conn,
                session,
                content_hash=str(session_content_hash(session)),
                prepared_rows=prepared,
            )

        observed = {
            "sessions": [
                (str(row["session_id"]), cast(bytes, row["content_hash"]).hex())
                for row in conn.execute("SELECT session_id, content_hash FROM sessions ORDER BY session_id")
            ],
            "messages": [
                str(row["message_id"]) for row in conn.execute("SELECT message_id FROM messages ORDER BY message_id")
            ],
            "blocks": [str(row["block_id"]) for row in conn.execute("SELECT block_id FROM blocks ORDER BY block_id")],
        }
    finally:
        conn.close()

    # Content-derived ids equal message_content_identity of the current
    # identity payload (typed lowering, python-mode model dumps).
    assert observed == {
        "sessions": [
            ("codex-session:duplicate-native-ids", "675394355f0dd0809ccac312e6c0b8258d948cd6bae827626cd44d6d207aa491"),
            ("codex-session:plain-text", "f04fe5e617574de32f9fb38ccedcf1159ce2581b14715cbc1c52516eb7d9793f"),
            ("codex-session:tool-use-and-thinking", "b388bae80e73def1b0c39c892a27ab887bca6b7b65128e823ae698025e54123f"),
        ],
        "messages": [
            "codex-session:duplicate-native-ids:c:71b9c4bb640966a23595ee589a5e1a76.0",
            "codex-session:duplicate-native-ids:c:f88028512715e01b32558cd5c33ef802.0",
            "codex-session:plain-text:n:m0",
            "codex-session:plain-text:n:m1",
            "codex-session:tool-use-and-thinking:n:t0",
            "codex-session:tool-use-and-thinking:n:t1",
        ],
        "blocks": [
            "codex-session:duplicate-native-ids:c:71b9c4bb640966a23595ee589a5e1a76.0:0",
            "codex-session:duplicate-native-ids:c:f88028512715e01b32558cd5c33ef802.0:0",
            "codex-session:plain-text:n:m0:0",
            "codex-session:plain-text:n:m1:0",
            "codex-session:tool-use-and-thinking:n:t0:0",
            "codex-session:tool-use-and-thinking:n:t0:1",
            "codex-session:tool-use-and-thinking:n:t1:0",
        ],
    }


def test_new_session_skips_field_path_union(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A proven-new session must not enter the reconciliation helper.

    This anti-vacuity guard exercises the production writer route on an empty
    index tier and makes the optimization observable: if the helper is
    consulted despite ``session_row_existed=False``, the write fails.
    """
    session = _synthetic_sessions()[0]
    prepared = prepare_session_rows(session)

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("new sessions must bypass field-path union")

    monkeypatch.setattr(archive_tier_write, "_union_with_existing_rows", _boom)
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = write_fixture_index_session(
            conn,
            session,
            content_hash=str(session_content_hash(session)),
            prepared_rows=prepared,
        )
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0] == 2
    finally:
        conn.close()


def test_stale_prepared_rows_fall_back_to_fresh_content(tmp_path: Path) -> None:
    """The session mutates AFTER prepare_session_rows() ran (a realistic
    parse-prefetch race: the archive's raw changed between warm() and the
    writer-held pass). The writer must detect the content-hash mismatch and
    build fresh rows reflecting the NEW content -- never silently write the
    stale prepared text.
    """
    original = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="mutates-after-prepare",
        title="Mutates after prepare",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="the original body computed by the prefetch worker",
                material_origin=MaterialOrigin.HUMAN_AUTHORED,
                position=0,
            ),
        ],
    )
    stale_prepared = prepare_session_rows(original)

    mutated = original.model_copy(
        update={
            "messages": [
                ParsedMessage(
                    provider_message_id="m0",
                    role=Role.USER,
                    text="a DIFFERENT body -- the raw changed before the writer ran",
                    material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    position=0,
                ),
            ]
        }
    )
    assert session_content_hash(mutated) != session_content_hash(original)

    conn = _connect(tmp_path / "index.db")
    try:
        session_id = write_fixture_index_session(
            conn,
            mutated,
            content_hash=str(session_content_hash(mutated)),
            prepared_rows=stale_prepared,
        )
        stored_text = conn.execute(
            "SELECT user_context_text FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        # user_context_text isn't populated from .text; assert via a direct
        # read of the composed message text using the archive's own reader.
        del stored_text
        from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope

        envelope = read_archive_session_envelope(conn, session_id)
        texts = ["".join(block.text or "" for block in message.blocks) for message in envelope.messages]
        assert texts == ["a DIFFERENT body -- the raw changed before the writer ran"]
    finally:
        conn.close()


def test_prepared_rows_are_ignored_when_no_content_hash_supplied(tmp_path: Path) -> None:
    """``content_hash=None`` means the writer falls back to an identity-only
    hash that can never coincide with a real ``PreparedSessionRows.session_
    content_hash`` -- prepared rows are never (mis)used in that case."""
    session = _synthetic_sessions()[0]
    prepared = prepare_session_rows(session)

    conn = _connect(tmp_path / "index.db")
    try:
        # No AssertionError from a monkeypatched builder here -- this test
        # only proves the reuse guard degrades safely, not that the builder
        # was skipped (it correctly is NOT skipped in this case).
        session_id = write_fixture_index_session(conn, session, prepared_rows=prepared)
        stored = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        assert stored == len(session.messages)
    finally:
        conn.close()


def test_prepare_session_rows_is_pure_and_reusable(tmp_path: Path) -> None:
    """Calling prepare_session_rows twice on the same immutable session
    yields identical PreparedSessionRows -- it touches no mutable state."""
    session = _synthetic_sessions()[2]
    first = prepare_session_rows(session)
    second = prepare_session_rows(session)
    assert isinstance(first, PreparedSessionRows)
    assert first.session_content_hash == second.session_content_hash
    assert first.message_rows == second.message_rows
    assert first.block_rows == second.block_rows


def test_prepared_write_preserves_prefix_sharing_context_without_writer_lowering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared fork retains only its divergent tail and composes its parent.

    Anti-vacuity: replacing normalization, prefix extraction, and both row
    builders after preparation proves publication consumes the exact prepared
    carrier rather than falling back inside the writer.
    """
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="prepared-parent",
        messages=[
            ParsedMessage(provider_message_id="a", role=Role.USER, text="A"),
            ParsedMessage(provider_message_id="b", role=Role.ASSISTANT, text="B"),
        ],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="prepared-child",
        parent_session_provider_id="prepared-parent",
        messages=[
            ParsedMessage(provider_message_id="a", role=Role.USER, text="A"),
            ParsedMessage(provider_message_id="b", role=Role.ASSISTANT, text="B"),
            ParsedMessage(provider_message_id="c", role=Role.USER, text="C"),
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        write_fixture_index_session(conn, parent, content_hash=str(session_content_hash(parent)))
        prepared = prepare_session_write(conn, child, merge_append=False)

        def _boom(*args: object, **kwargs: object) -> object:
            raise AssertionError("prepared lineage write must not lower inside the writer")

        monkeypatch.setattr(archive_tier_write, "normalize_active_branch", _boom)
        monkeypatch.setattr(archive_tier_write, "_extract_prefix_tail", _boom)
        monkeypatch.setattr(archive_tier_write, "_build_message_rows", _boom)
        monkeypatch.setattr(archive_tier_write, "_build_block_rows", _boom)
        with fixture_index_mutation_scope(conn):
            child_id = write_fixture_index_session(
                conn,
                child,
                content_hash=str(session_content_hash(child)),
                prepared_write=prepared,
            )
        assert [row[0] for row in conn.execute("SELECT native_id FROM messages WHERE session_id = ?", (child_id,))] == [
            "c"
        ]
        link = conn.execute(
            "SELECT branch_point_message_id, inheritance FROM session_links WHERE src_session_id = ?", (child_id,)
        ).fetchone()
        assert tuple(link) == ("codex-session:prepared-parent:n:b", "prefix-sharing")
        envelope = read_archive_session_envelope(conn, child_id)
        assert ["".join(block.text or "" for block in message.blocks) for message in envelope.messages] == [
            "A",
            "B",
            "C",
        ]
    finally:
        conn.close()


def test_disk_prepared_write_keeps_lineage_tail_and_rows_off_heap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="disk-parent",
        messages=[
            ParsedMessage(provider_message_id="a", role=Role.USER, text="A"),
            ParsedMessage(provider_message_id="b", role=Role.ASSISTANT, text="B"),
        ],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="disk-child",
        parent_session_provider_id="disk-parent",
        messages=[
            *parent.messages,
            ParsedMessage(provider_message_id="c", role=Role.USER, text="C"),
        ],
    )
    store = SqliteMessageStore(tmp_path / "child-prepared.db")
    sink = store.new_sink()
    sink.extend(child.messages)
    # The parse side seals its canonical normalized writer operand with the
    # messages (prepared_jsonl._append_artifact_session); a sealed sink
    # without one is refused at preparation.
    sink.normalized_messages(child.session_events, origin=Origin.CODEX_SESSION)
    store.conn.commit()
    store.close()
    sealed = SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink))
    publication = child.model_copy(update={"messages": sealed, "content_hash": str(session_content_hash(child))})
    conn = _connect(tmp_path / "index.db")
    try:
        write_fixture_index_session(conn, parent, content_hash=str(session_content_hash(parent)))

        def forbid_row_lowering(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("writer rebuilt prepared lineage rows")

        def keep_rows_off_heap(prepared: PreparedSessionWrite) -> None:
            assert not isinstance(prepared.rows.message_rows, tuple)
            assert len(prepared.context.messages) == 1
            monkeypatch.setattr(archive_tier_write, "_iter_message_rows", forbid_row_lowering)
            monkeypatch.setattr(archive_tier_write, "_iter_block_rows", forbid_row_lowering)

        child_id = write_fixture_prepared_session(
            conn,
            publication,
            inspect=keep_rows_off_heap,
            content_hash=publication.content_hash,
        )
        assert [row[0] for row in conn.execute("SELECT native_id FROM messages WHERE session_id = ?", (child_id,))] == [
            "c"
        ]
        assert [message.native_id for message in read_archive_session_envelope(conn, child_id).messages] == [
            "a",
            "b",
            "c",
        ]
    finally:
        conn.close()


def test_prepared_lineage_refuses_changed_earlier_parent_prefix(tmp_path: Path) -> None:
    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="mutable-parent",
        messages=[
            ParsedMessage(provider_message_id="a", role=Role.USER, text="old A"),
            ParsedMessage(provider_message_id="b", role=Role.ASSISTANT, text="B"),
        ],
    )
    child = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="waiting-child",
        parent_session_provider_id="mutable-parent",
        messages=[
            *parent.messages,
            ParsedMessage(provider_message_id="c", role=Role.USER, text="C"),
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        write_fixture_index_session(conn, parent, content_hash=str(session_content_hash(parent)))
        prepared = prepare_session_write(conn, child, merge_append=False)
        changed_parent = parent.model_copy(
            update={
                "messages": [
                    ParsedMessage(provider_message_id="a", role=Role.USER, text="new A"),
                    parent.messages[1],
                ]
            }
        )
        write_fixture_index_session(
            conn,
            changed_parent,
            content_hash=str(session_content_hash(changed_parent)),
            force_replace=True,
        )
        with pytest.raises(PreparedSessionWriteRefusedError, match="lineage prefix changed"):
            with fixture_index_mutation_scope(conn):
                write_fixture_index_session(
                    conn,
                    child,
                    content_hash=str(session_content_hash(child)),
                    prepared_write=prepared,
                )
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE native_id = 'waiting-child'").fetchone()[0] == 0
    finally:
        conn.close()


def test_prepared_cross_acquisition_union_matches_inline_and_skips_writer_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poorer export preserves older message and nested block evidence.

    Anti-vacuity: the old in-writer union and canonical row builders raise
    after preparation, so a prepared carrier must publish its merged rows.
    """
    rich = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="prepared-field-union",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.ASSISTANT,
                blocks=[
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text="answer",
                        tool_input={"citation": {"url": "https://example.invalid", "start": 4}},
                    ),
                ],
            ),
            ParsedMessage(
                provider_message_id="m2",
                role=Role.TOOL,
                blocks=[
                    ParsedContentBlock(type=BlockType.TOOL_RESULT, text="retained tool", is_error=False),
                ],
            ),
        ],
    )
    poor = rich.model_copy(
        update={
            "messages": [
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TEXT,
                            text="answer",
                            tool_input={"citation": {"url": "https://example.invalid"}},
                        ),
                    ],
                ),
            ]
        }
    )

    # Each comparison archive owns its own active Index.
    expected = _connect(tmp_path / "expected" / "index.db")
    actual = _connect(tmp_path / "actual" / "index.db")
    try:
        write_fixture_index_session(expected, rich, raw_id="older", content_hash=str(session_content_hash(rich)))
        write_fixture_index_session(expected, poor, raw_id="newer", content_hash=str(session_content_hash(poor)))
        session_id = write_fixture_index_session(
            actual, rich, raw_id="older", content_hash=str(session_content_hash(rich))
        )
        store = SqliteMessageStore(tmp_path / "prepared-poor.db")
        sink = store.new_sink()
        sink.extend(poor.messages)
        bound_hash = str(session_content_hash(poor))
        worker_session = poor.model_copy(update={"messages": sink, "content_hash": bound_hash})
        shard = prepare_session_shard(tmp_path, [worker_session])
        store.conn.commit()
        store.close()
        sealed = SqliteMessageSink(store.path, sink.session_ordinal, count=len(sink))
        publication = poor.model_copy(update={"messages": sealed, "content_hash": bound_hash})
        input_rows = prepared_session_rows_from_shard(shard.path, session_id)

        def forbid_inline(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("writer rebuilt or merged prepared field rows")

        def forbid_writer_merge(prepared: PreparedSessionWrite) -> None:
            assert prepared.cross_acquisition_union is not None
            assert not isinstance(prepared.cross_acquisition_union.rows.message_rows, tuple)
            monkeypatch.setattr(archive_tier_write, "_union_with_existing_rows", forbid_inline)
            monkeypatch.setattr(archive_tier_write, "_iter_message_rows", forbid_inline)
            monkeypatch.setattr(archive_tier_write, "_iter_block_rows", forbid_inline)

        # Retained preparation needs its original Source read capability, so
        # it runs on an original seal (prepare_session_write refuses raw_id
        # without one); the writer's inline builders are forbidden from
        # preparation onward.
        write_fixture_prepared_session(
            actual,
            publication,
            inspect=forbid_writer_merge,
            prepared_rows=input_rows,
            raw_id="newer",
            content_hash=publication.content_hash,
        )
        for table in ("sessions", "messages", "blocks"):
            assert [tuple(row) for row in actual.execute(f"SELECT * FROM {table} ORDER BY rowid")] == [
                tuple(row) for row in expected.execute(f"SELECT * FROM {table} ORDER BY rowid")
            ]
    finally:
        expected.close()
        actual.close()


def test_prepared_cross_acquisition_union_refuses_changed_predecessor(tmp_path: Path) -> None:
    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="union-race",
        messages=[
            ParsedMessage(provider_message_id="m1", role=Role.USER, text="first"),
        ],
    )
    second = first.model_copy(
        update={
            "messages": [
                ParsedMessage(provider_message_id="m1", role=Role.USER, text="second"),
            ]
        }
    )
    competing = first.model_copy(
        update={
            "messages": [
                ParsedMessage(provider_message_id="m1", role=Role.USER, text="competing"),
            ]
        }
    )
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = write_fixture_index_session(
            conn,
            first,
            raw_id="old",
            content_hash=str(session_content_hash(first)),
        )

        def compete(prepared: PreparedSessionWrite) -> None:
            assert prepared.cross_acquisition_union is not None
            write_fixture_index_session(
                conn,
                competing,
                raw_id="competing",
                content_hash=str(session_content_hash(competing)),
                force_replace=True,
            )

        # Retained preparation runs on its original seal. A competing commit
        # after preparation is now refused at the seal's writer validation
        # (the Index incarnation it prepared against moved) before the
        # writer's own predecessor check is reached; the refusal keeps the
        # competing row.
        with pytest.raises(ReferenceSealStaleError, match="incarnation changed"):
            write_fixture_prepared_session(
                conn,
                second,
                inspect=compete,
                raw_id="second",
                content_hash=str(session_content_hash(second)),
            )
        assert (
            conn.execute("SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0] == "competing"
        )
    finally:
        conn.close()


def test_a_dropped_prepared_union_still_refreshes_replaced_attachments(tmp_path: Path) -> None:
    """A union the writer's precedence overrules leaves no stale attachment.

    Anti-vacuity: when the union was dropped only inside the row replacement,
    the caller still saw a prepared union and skipped collecting the
    session's attachment ids, so the replaced message's attachment kept its
    reference count and pinned its blob. The prepared route must end with the
    same attachment rows as the unprepared force-replace route.
    """
    from polylogue.sources.parsers.base import ParsedAttachment

    first = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="union-overruled",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="first")],
        attachments=[ParsedAttachment(provider_attachment_id="att-old", message_provider_id="m1", name="old.txt")],
    )
    second = first.model_copy(
        update={
            "messages": [ParsedMessage(provider_message_id="m1", role=Role.USER, text="second")],
            "attachments": [],
        }
    )
    tables = ("attachments", "attachment_refs")
    snapshots = []
    for prepared_route in (False, True):
        conn = _connect(tmp_path / f"route-{prepared_route}" / "index.db")
        try:
            write_fixture_index_session(conn, first, raw_id="old", content_hash=str(session_content_hash(first)))
            if not prepared_route:
                write_fixture_index_session(
                    conn,
                    second,
                    raw_id="second",
                    content_hash=str(session_content_hash(second)),
                    force_replace=True,
                )
            else:

                def require_union(prepared: PreparedSessionWrite) -> None:
                    assert prepared.cross_acquisition_union is not None

                # Retained preparation runs on its original seal and prepares
                # the union; publication's force_replace precedence overrules it.
                write_fixture_prepared_session(
                    conn,
                    second,
                    inspect=require_union,
                    raw_id="second",
                    content_hash=str(session_content_hash(second)),
                    publish_options={"force_replace": True},
                )
            snapshots.append(
                {
                    table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
                    for table in tables
                }
            )
        finally:
            conn.close()
    assert snapshots[1] == snapshots[0]
