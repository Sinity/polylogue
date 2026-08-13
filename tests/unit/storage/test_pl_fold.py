"""Polish search recall: pl_fold write/query symmetry (polylogue-9jsi).

Covers the acceptance criteria on polylogue-9jsi:

- ``latwo``/``zrobilem`` queries hit seeded ``łatwo``/``zrobiłem`` content.
- ``pl_fold`` is idempotent, and the Python implementation, the registered
  SQL scalar function, and the inline ``REPLACE`` chain embedded in DDL all
  agree byte-for-byte.
- Fresh archive DDL and the repair lifecycle create the same tokenizer behavior
  from one canonical ``messages_fts`` definition.
- ``unicode61 remove_diacritics 2`` (a distinct mechanism from ``pl_fold``)
  folds ordinary combining-mark diacritics (``ó``, ``ż``, ...) on its own.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.fts.pl_fold import PL_FOLD_TABLE, pl_fold, pl_fold_sql_expr, register_pl_fold
from polylogue.storage.fts.sql import FTS_MESSAGES_TABLE_SQL, FTS_UNICODE_TOKENIZER
from polylogue.storage.search.query_support import escape_fts5_query, normalize_fts5_query
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.identity import archive_message_id

# ---------------------------------------------------------------------------
# Python fold semantics
# ---------------------------------------------------------------------------


def test_pl_fold_folds_l_stroke_only() -> None:
    assert pl_fold("łatwo") == "latwo"
    assert pl_fold("Łatwo") == "Latwo"
    assert pl_fold("zrobiłem") == "zrobilem"
    assert pl_fold("ŁATWO") == "LATWO"
    # Plain ASCII and unrelated diacritics pass through unchanged -- pl_fold
    # is deliberately narrow (unicode61 remove_diacritics handles the rest).
    assert pl_fold("latwo") == "latwo"
    assert pl_fold("różowy") == "różowy"
    assert pl_fold("") == ""
    assert pl_fold(None) is None


def test_pl_fold_is_idempotent() -> None:
    samples = [
        "łatwo",
        "Łatwo zrobiłem to ŁATWO zadanie",
        "plain ascii, no fold needed",
        "różowy słoń",
        "",
    ]
    for sample in samples:
        once = pl_fold(sample)
        twice = pl_fold(once)
        assert once == twice, f"pl_fold not idempotent for {sample!r}: {once!r} != {twice!r}"


def test_pl_fold_table_targets_never_reintroduce_a_source() -> None:
    """Idempotency invariant: no fold target character is itself a fold source."""
    sources = {source for source, _ in PL_FOLD_TABLE}
    for _, target in PL_FOLD_TABLE:
        assert target not in sources


# ---------------------------------------------------------------------------
# Python / SQL agreement (registered UDF and inline REPLACE chain)
# ---------------------------------------------------------------------------


def test_registered_sql_function_matches_python() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        register_pl_fold(conn)
        samples = ["łatwo", "Łatwo", "zrobiłem", "plain", "różowy", "", None]
        for sample in samples:
            row = conn.execute("SELECT pl_fold(?)", (sample,)).fetchone()
            assert row[0] == pl_fold(sample), f"SQL/Python disagreement for {sample!r}"
    finally:
        conn.close()


def test_inline_replace_chain_matches_python() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        expr = pl_fold_sql_expr("?")
        samples = ["łatwo", "Łatwo zrobiłem", "plain", "różowy", ""]
        for sample in samples:
            row = conn.execute(f"SELECT {expr}", (sample,)).fetchone()
            assert row[0] == pl_fold(sample), f"REPLACE chain/Python disagreement for {sample!r}"
    finally:
        conn.close()


def test_pl_fold_sql_expr_is_a_pure_replace_chain() -> None:
    """No custom-function dependency: safe to embed in DDL fired by any connection."""
    expr = pl_fold_sql_expr("new.search_text")
    assert expr.count("REPLACE(") == len(PL_FOLD_TABLE)
    assert "pl_fold(" not in expr


# ---------------------------------------------------------------------------
# DDL behavior
# ---------------------------------------------------------------------------


def test_fresh_archive_fts_surfaces_use_the_canonical_tokenizer(tmp_path: Path) -> None:
    with ArchiveStore(tmp_path):
        pass
    with sqlite3.connect(tmp_path / "index.db") as conn:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE name IN ('messages_fts', 'session_work_events_fts')"
        ).fetchall()

    definitions = {str(row[0]): str(row[1]) for row in rows}
    assert set(definitions) == {"messages_fts", "session_work_events_fts"}
    assert all(f"tokenize='{FTS_UNICODE_TOKENIZER}'" in sql for sql in definitions.values())


def test_repair_messages_fts_uses_the_same_canonical_definition() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(FTS_MESSAGES_TABLE_SQL)
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts'").fetchone()[0]
    finally:
        conn.close()

    assert f"tokenize='{FTS_UNICODE_TOKENIZER}'" in sql


# ---------------------------------------------------------------------------
# End-to-end recall (write-side fold + query-side fold, through real triggers)
# ---------------------------------------------------------------------------


def _seed_text_block(conn: sqlite3.Connection, *, native_session_id: str, native_message_id: str, text: str) -> str:
    origin = "unknown-export"
    session_id = f"{origin}:{native_session_id}"
    message_id = archive_message_id(session_id, native_message_id, position=0)
    content_hash = b"x" * 32
    conn.execute(
        "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
        (native_session_id, origin, "Polish recall fixture", content_hash),
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
        VALUES (?, ?, 0, 'user', 'message', ?)
        """,
        (session_id, native_message_id, content_hash),
    )
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text)
        VALUES (?, ?, 0, 'text', ?)
        """,
        (message_id, session_id, text),
    )
    return message_id


def _fts_hits(conn: sqlite3.Connection, query: str) -> list[str]:
    fts_query = normalize_fts5_query(query)
    assert fts_query is not None
    rows = conn.execute(
        """
        SELECT b.message_id
        FROM messages_fts
        JOIN blocks AS b ON b.rowid = messages_fts.rowid
        WHERE messages_fts MATCH ?
        """,
        (fts_query,),
    ).fetchall()
    return [row[0] for row in rows]


def test_ascii_query_finds_l_stroke_seeded_content(test_conn: sqlite3.Connection) -> None:
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-pl-fold-latwo",
        native_message_id="msg-latwo",
        text="Bardzo łatwo zrobiłem to zadanie",
    )

    assert _fts_hits(test_conn, "latwo") == [message_id]
    assert _fts_hits(test_conn, "zrobilem") == [message_id]


def test_literal_l_stroke_query_also_finds_the_same_content(test_conn: sqlite3.Connection) -> None:
    """Query-side folding must be symmetric: a literal ``ł`` in the query also matches."""
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-pl-fold-literal",
        native_message_id="msg-literal",
        text="Bardzo łatwo zrobiłem to zadanie",
    )

    assert _fts_hits(test_conn, "łatwo") == [message_id]


def test_archive_store_rebuild_index_preserves_l_stroke_recall(tmp_path: Path) -> None:
    """The public rebuild route must fold canonical text, not only fresh-write triggers."""
    with ArchiveStore(tmp_path) as archive:
        message_id = _seed_text_block(
            archive._conn,
            native_session_id="conv-pl-fold-public-rebuild",
            native_message_id="msg-public-rebuild",
            text="Bardzo łatwo zrobiłem to zadanie",
        )
        block_id = archive._conn.execute(
            "SELECT block_id FROM blocks WHERE message_id = ?",
            (message_id,),
        ).fetchone()[0]

        assert archive.rebuild_index() == 1
        assert archive.search_blocks("latwo") == [block_id]
        assert archive.search_blocks("łatwo") == [block_id]


def test_remove_diacritics_tokenizer_handles_ordinary_diacritics_without_pl_fold(
    test_conn: sqlite3.Connection,
) -> None:
    """ó/ż fold via the FTS5 tokenizer itself -- a mechanism distinct from pl_fold."""
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-pl-fold-diacritics",
        native_message_id="msg-diacritics",
        text="różowy słoń zrobię to jutro",
    )

    assert _fts_hits(test_conn, "rozowy") == [message_id]
    assert _fts_hits(test_conn, "zrobie") == [message_id]


def test_plain_ascii_query_is_unaffected_by_folding(test_conn: sqlite3.Connection) -> None:
    message_id = _seed_text_block(
        test_conn,
        native_session_id="conv-pl-fold-ascii",
        native_message_id="msg-ascii",
        text="ordinary english text with no diacritics at all",
    )

    assert _fts_hits(test_conn, "ordinary english") == [message_id]
    assert escape_fts5_query("hello world") == "hello world"
