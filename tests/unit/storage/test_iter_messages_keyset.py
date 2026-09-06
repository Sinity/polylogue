"""Equivalence checks for keyset-paginated ``iter_messages`` (#1750 F1).

The keyset stream must produce exactly the same rows in exactly the same order
as:

* the canonical single-query ordering (``get_messages``), and
* a reference ``LIMIT/OFFSET`` walk over the transcript order,

across every chunk-size boundary, every ``occurred_at_ms`` shape (non-NULL,
duplicates, NULL tail, all-NULL), sibling variants at one position, the role
filter, and the optional ``limit``.

Anti-vacuity: every scenario's clock disagrees with its content positions, so
re-keying the cursor or the ordering on ``occurred_at_ms`` makes the stream
diverge from both references. ``sibling_variants`` fails the same way if the
cursor drops ``variant_index``: at ``chunk_size=1`` the second sibling at a
position is skipped.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from polylogue.archive.message.roles import MessageRoleFilter, Role, message_role_sql_values
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.queries.message_query_reads import get_messages, iter_messages

CONV = "codex-session:keyset"

# (native_id, occurred_at seconds or None, role, position, variant_index),
# listed in transcript order.
Row = tuple[str, float | None, str, int, int]


def _linear(rows: list[tuple[str, float | None, str]]) -> list[Row]:
    """Assign one message per position, in list order."""
    return [(mid, sort_key, role, index, 0) for index, (mid, sort_key, role) in enumerate(rows)]


_SCENARIOS: dict[str, list[Row]] = {
    "all_non_null": _linear(
        [
            ("m03", 3.0, "user"),
            ("m01", 1.0, "assistant"),
            ("m02", 2.0, "user"),
            ("m04", 4.0, "assistant"),
            ("m05", 5.0, "user"),
        ]
    ),
    "duplicate_timestamps": _linear(
        [
            ("m_b", 1.0, "user"),
            ("m_a", 1.0, "assistant"),
            ("m_d", 2.0, "user"),
            ("m_c", 2.0, "assistant"),
            ("m_e", 3.0, "user"),
        ]
    ),
    "null_tail": _linear(
        [
            ("m02", 2.0, "user"),
            ("m01", 1.0, "assistant"),
            ("nz", None, "user"),
            ("na", None, "assistant"),
            ("nm", None, "user"),
        ]
    ),
    "all_null": _linear(
        [
            ("z", None, "user"),
            ("a", None, "assistant"),
            ("m", None, "user"),
            ("b", None, "assistant"),
        ]
    ),
    "nulls_interleaved": _linear(
        [
            ("s2", 9.0, "user"),
            ("s1", 8.0, "assistant"),
            ("n3", None, "user"),
            ("n1", None, "assistant"),
            ("n2", None, "user"),
            ("n4", None, "assistant"),
        ]
    ),
    # Siblings share a position and are separated only by variant_index, which
    # the keyset cursor must carry.
    "sibling_variants": [
        ("v0", 5.0, "user", 0, 0),
        ("v1a", 4.0, "assistant", 1, 0),
        ("v1b", 6.0, "assistant", 1, 1),
        ("v1c", 3.0, "assistant", 1, 2),
        ("v2", 1.0, "user", 2, 0),
    ],
    "single": _linear([("only", 1.0, "user")]),
}


def _seed(conn: sqlite3.Connection, rows: list[Row]) -> None:
    conn.execute(
        """
        INSERT INTO sessions (
            native_id, origin, title, content_hash
        ) VALUES ('keyset', 'codex-session', 'Keyset', zeroblob(32))
        """,
    )
    conn.executemany(
        """
        INSERT INTO messages (session_id, native_id, role, position, variant_index, occurred_at_ms, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, zeroblob(32))
        """,
        [
            (CONV, mid, role, position, variant_index, None if sort_key is None else int(sort_key * 1000))
            for mid, sort_key, role, position, variant_index in rows
        ],
    )
    conn.executemany(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text)
        SELECT message_id, session_id, 0, 'text', native_id
        FROM messages
        WHERE session_id = ? AND native_id = ?
        """,
        [(CONV, mid) for mid, _sort_key, _role, _position, _variant_index in rows],
    )
    conn.commit()


def _make_db(tmp_path: object, rows: list[Row]) -> str:
    db_path = str(tmp_path) + "/keyset.db"
    initialize_archive_database(Path(db_path), ArchiveTier.INDEX)
    sync_conn = sqlite3.connect(db_path)
    sync_conn.row_factory = sqlite3.Row
    _seed(sync_conn, rows)
    sync_conn.close()
    return db_path


async def _reference_offset_walk(
    conn: aiosqlite.Connection,
    session_id: str,
    chunk_size: int,
    role_values: tuple[str, ...] | list[str],
    limit: int | None,
) -> list[str]:
    """LIMIT/OFFSET walk of the transcript order -- the contract the keyset
    cursor reproduces without the O(M^2) re-scan."""
    out: list[str] = []
    offset = 0
    yielded = 0
    while True:
        query = "SELECT message_id FROM messages WHERE session_id = ?"
        params: list[str | int] = [session_id]
        if role_values:
            placeholders = ",".join("?" for _ in role_values)
            query += f" AND role IN ({placeholders})"
            params.extend(role_values)
        query += " ORDER BY position, variant_index"
        fetch_limit = chunk_size
        if limit is not None:
            remaining = limit - yielded
            if remaining <= 0:
                break
            fetch_limit = min(chunk_size, remaining)
        query += " LIMIT ? OFFSET ?"
        params.extend([fetch_limit, offset])
        cursor = await conn.execute(query, tuple(params))
        rows = list(await cursor.fetchall())
        if not rows:
            break
        for row in rows:
            out.append(str(row["message_id"]))
            yielded += 1
            if limit is not None and yielded >= limit:
                return out
        offset += len(rows)
        if len(rows) < fetch_limit:
            break
    return out


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 6, 7, 100])
async def test_keyset_matches_canonical_and_offset(tmp_path: object, scenario: str, chunk_size: int) -> None:
    rows = _SCENARIOS[scenario]
    db_path = _make_db(tmp_path, rows)
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        canonical = [m.message_id for m in await get_messages(conn, CONV)]
        keyset = [m.message_id async for m in iter_messages(conn, CONV, chunk_size=chunk_size)]
        reference = await _reference_offset_walk(conn, CONV, chunk_size, [], None)

    expected_tail = [mid for mid, _sort_key, _role, _position, _variant_index in rows]
    assert keyset == canonical
    assert keyset == reference
    assert [message_id.rsplit(":", 1)[1] for message_id in keyset] == expected_tail
    assert len(set(keyset)) == len(keyset)


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
@pytest.mark.parametrize("roles", [(Role.USER,), (Role.USER, Role.ASSISTANT)])
async def test_keyset_with_role_filter(
    tmp_path: object, scenario: str, chunk_size: int, roles: MessageRoleFilter
) -> None:
    rows = _SCENARIOS[scenario]
    db_path = _make_db(tmp_path, rows)
    role_values = message_role_sql_values(roles)
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        keyset = [m.message_id async for m in iter_messages(conn, CONV, chunk_size=chunk_size, message_roles=roles)]
        reference = await _reference_offset_walk(conn, CONV, chunk_size, role_values, None)
    assert keyset == reference


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 100])
async def test_keyset_with_limit(tmp_path: object, scenario: str, chunk_size: int, limit: int) -> None:
    rows = _SCENARIOS[scenario]
    db_path = _make_db(tmp_path, rows)
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        keyset = [m.message_id async for m in iter_messages(conn, CONV, chunk_size=chunk_size, limit=limit)]
        reference = await _reference_offset_walk(conn, CONV, chunk_size, [], limit)
    assert keyset == reference
    assert len(keyset) <= limit


async def test_keyset_empty_session(tmp_path: object) -> None:
    db_path = _make_db(tmp_path, _SCENARIOS["single"])
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        assert [m.message_id async for m in iter_messages(conn, "missing", chunk_size=2)] == []


async def test_keyset_isolates_sessions(tmp_path: object) -> None:
    """The keyset cursor must not bleed rows from a sibling session."""
    db_path = str(tmp_path) + "/iso.db"
    initialize_archive_database(Path(db_path), ArchiveTier.INDEX)
    sync_conn = sqlite3.connect(db_path)
    sync_conn.row_factory = sqlite3.Row
    for cid, rows in (
        ("codex-session:convA", [("a1", 3.0, "user"), ("a2", None, "user"), ("a3", 1.0, "user")]),
        ("codex-session:convB", [("b1", 2.0, "user"), ("b2", None, "user")]),
    ):
        sync_conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, title, content_hash
            ) VALUES (?, 'codex-session', 't', zeroblob(32))
            """,
            (cid.split(":", 1)[1],),
        )
        sync_conn.executemany(
            """
            INSERT INTO messages (session_id, native_id, role, position, occurred_at_ms, content_hash)
            VALUES (?, ?, ?, ?, ?, zeroblob(32))
            """,
            [(cid, mid, role, idx, None if sk is None else int(sk * 1000)) for idx, (mid, sk, role) in enumerate(rows)],
        )
    sync_conn.commit()
    sync_conn.close()

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        got = [m.message_id async for m in iter_messages(conn, "codex-session:convA", chunk_size=2)]
        canonical = [m.message_id for m in await get_messages(conn, "codex-session:convA")]
    assert got == canonical
    assert [message_id.rsplit(":", 1)[1] for message_id in got] == ["a1", "a2", "a3"]


__all__: list[str] = []
