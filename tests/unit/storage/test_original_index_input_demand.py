"""Canonical selectors hydrate the same physical inputs they account."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite import reference_seal
from polylogue.storage.sqlite.archive_tiers.write import _branch_point_content_address_matches
from polylogue.storage.sqlite.session_identity import resolve_session_id_in_index


def test_ambiguous_session_suffix_accounts_both_selected_physical_rows() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE sessions(session_id TEXT PRIMARY KEY) STRICT")
        identifiers = ("codex-session:" + "x" * 20000, "claude-code:" + "x" * 20000)
        connection.executemany("INSERT INTO sessions VALUES (?)", ((identifier,) for identifier in identifiers))
        connection.commit()
        accounted: list[int] = []

        def before_input(table: str, columns: tuple[str, ...], sql: str, parameters: tuple[object, ...]) -> None:
            assert table == "sessions" and columns == ("session_id",)
            with closing(connection.execute(sql, parameters)) as cursor:
                accounted.extend(row[0] for row in cursor)

        with pytest.raises(ValueError):
            resolve_session_id_in_index(connection, "x" * 20000, before_input=before_input)
        assert set(accounted) == {1, 2}
        assert len(accounted) == 2
        assert not connection.in_transaction


def test_block_or_selector_hydrates_the_accounted_join_even_if_query_order_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript(
            "CREATE TABLE messages(message_id TEXT PRIMARY KEY,session_id TEXT) STRICT;"
            "CREATE TABLE blocks(block_id TEXT PRIMARY KEY,message_id TEXT,position INTEGER) STRICT;"
            "INSERT INTO messages VALUES ('message-a','session-a'),('message-b','session-b');"
            "INSERT INTO blocks VALUES ('block-a','message-a',0),('block-b','message-b',1);"
        )
        accounted: dict[str, int] = {}

        def before_input(table: str, columns: tuple[str, ...], sql: str, parameters: tuple[object, ...]) -> None:
            with closing(connection.execute(sql, parameters)) as cursor:
                accounted[table] = cursor.fetchone()[0]
            connection.execute("PRAGMA reverse_unordered_selects=ON").close()

        monkeypatch.setattr(reference_seal, "_index_input_hook", lambda original: before_input)
        row = reference_seal._block_reference_row(
            connection, "b.block_id=? OR b.message_id=?", ("block-a", "message-b"), first_only=True
        )
        assert row is not None
        with closing(connection.execute("SELECT block_id FROM blocks WHERE rowid=?", (accounted["blocks"],))) as cursor:
            assert row[3] == cursor.fetchone()[0]
        with closing(
            connection.execute("SELECT session_id FROM messages WHERE rowid=?", (accounted["messages"],))
        ) as cursor:
            assert row[0] == cursor.fetchone()[0]


def test_tied_branch_selector_compares_the_exact_accounted_link_and_message() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript(
            "CREATE TABLE messages(message_id TEXT PRIMARY KEY,content_address BLOB) STRICT;"
            "CREATE TABLE session_links(src_session_id TEXT,resolved_dst_session_id TEXT,"
            "branch_point_message_id TEXT,inheritance TEXT,branch_point_content_address BLOB) STRICT;"
            "INSERT INTO messages VALUES ('branch',X'01');"
            "INSERT INTO session_links VALUES ('child','parent','branch','prefix-sharing',X'01'),"
            "('child','parent','branch','prefix-sharing',X'02');"
        )
        accounted: dict[str, int] = {}

        def before_input(table: str, columns: tuple[str, ...], sql: str, parameters: tuple[object, ...]) -> None:
            with closing(connection.execute(sql, parameters)) as cursor:
                accounted[table] = cursor.fetchone()[0]
            connection.execute("PRAGMA reverse_unordered_selects=ON").close()

        result = _branch_point_content_address_matches(connection, "child", "parent", "branch", before_input)
        with closing(
            connection.execute(
                "SELECT branch_point_content_address FROM session_links WHERE rowid=?", (accounted["session_links"],)
            )
        ) as cursor:
            expected = cursor.fetchone()[0] == b"\x01"
        assert result is expected
        assert accounted["messages"] == 1


def test_nontransactional_composed_stream_yields_each_stored_message_once(tmp_path: Path) -> None:
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from polylogue.storage.sqlite.archive_tiers.write import _iter_composed_rows, _prefix_alignment_signature
    from tests.infra.index_writer import write_fixture_index_session
    from tests.infra.reference_sessions import reference_session

    # The fixture writer's owned Index transaction requires the production
    # measured creator; a bare sqlite3 handle is refused by the seal.
    with closing(connect_measured(tmp_path / "index.db")) as connection:
        connection.row_factory = sqlite3.Row
        initialize_archive_tier(connection, ArchiveTier.INDEX)
        session_id = write_fixture_index_session(connection, reference_session("single-composed-input"))
        # Composed rows carry the replay-alignment digest of the role and each
        # stored block (f7d0b64ef0), not the complete content address.
        with closing(
            connection.execute(
                "SELECT message_id,role FROM messages WHERE session_id=? ORDER BY position,variant_index",
                (session_id,),
            )
        ) as cursor:
            stored = [(str(row[0]), str(row[1] or "")) for row in cursor]
        expected = []
        for message_id, role in stored:
            with closing(
                connection.execute(
                    "SELECT content_hash FROM blocks WHERE session_id=? AND message_id=? ORDER BY position",
                    (session_id, message_id),
                )
            ) as cursor:
                hashes = [bytes(row[0]) for row in cursor if row[0] is not None]
            expected.append((message_id, _prefix_alignment_signature(role, hashes), session_id))
        assert len(expected) == 1
        assert not connection.in_transaction
        with closing(_iter_composed_rows(connection, session_id)) as rows:
            assert list(rows) == expected
        assert not connection.in_transaction
        connection.execute("BEGIN").close()
        with closing(_iter_composed_rows(connection, session_id)) as rows:
            assert list(rows) == expected
        assert connection.in_transaction
        connection.rollback()
