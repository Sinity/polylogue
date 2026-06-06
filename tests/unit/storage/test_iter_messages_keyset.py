"""Old-vs-new equivalence for keyset-paginated ``iter_messages`` (#1750 F1).

``iter_messages`` was converted from ``LIMIT ? OFFSET ?`` chunking to keyset
pagination seeded by the previous chunk's last ``(sort_key, message_id)``. The
keyset stream must produce exactly the same rows in exactly the same order as:

* the canonical single-query ordering (``get_messages``), and
* a reference ``LIMIT/OFFSET`` walk reproducing the pre-#1750 behavior,

across every chunk-size boundary, every ``sort_key`` shape (non-NULL, duplicate
sort_key ties broken by ``message_id``, NULL tail, all-NULL), the role filter,
and the optional ``limit``.
"""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from polylogue.archive.message.roles import MessageRoleFilter, Role, message_role_sql_values
from polylogue.storage.sqlite.queries.message_query_reads import get_messages, iter_messages
from polylogue.storage.sqlite.schema import _ensure_schema

CONV = "conv-keyset"

# (message_id, sort_key, role) fixtures covering every ordering edge case.
_SCENARIOS: dict[str, list[tuple[str, float | None, str]]] = {
    "all_non_null": [
        ("m03", 3.0, "user"),
        ("m01", 1.0, "assistant"),
        ("m02", 2.0, "user"),
        ("m04", 4.0, "assistant"),
        ("m05", 5.0, "user"),
    ],
    "duplicate_sort_keys": [
        # ties on sort_key must break by message_id
        ("m_b", 1.0, "user"),
        ("m_a", 1.0, "assistant"),
        ("m_d", 2.0, "user"),
        ("m_c", 2.0, "assistant"),
        ("m_e", 3.0, "user"),
    ],
    "null_tail": [
        ("m02", 2.0, "user"),
        ("m01", 1.0, "assistant"),
        ("nz", None, "user"),
        ("na", None, "assistant"),
        ("nm", None, "user"),
    ],
    "all_null": [
        ("z", None, "user"),
        ("a", None, "assistant"),
        ("m", None, "user"),
        ("b", None, "assistant"),
    ],
    "boundary_at_null_transition": [
        ("s2", 9.0, "user"),
        ("s1", 8.0, "assistant"),
        ("n3", None, "user"),
        ("n1", None, "assistant"),
        ("n2", None, "user"),
        ("n4", None, "assistant"),
    ],
    "single": [("only", 1.0, "user")],
}


def _seed(conn: sqlite3.Connection, rows: list[tuple[str, float | None, str]]) -> None:
    conn.execute(
        """
        INSERT INTO sessions (
            session_id, provider_session_id, source_name, title, version
        ) VALUES (?, 'p1', 'codex', 'Keyset', 1)
        """,
        (CONV,),
    )
    conn.executemany(
        """
        INSERT INTO messages (message_id, session_id, role, sort_key, version, content_hash)
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        [(mid, CONV, role, sort_key, mid) for mid, sort_key, role in rows],
    )
    conn.commit()


def _make_db(tmp_path: object, rows: list[tuple[str, float | None, str]]) -> str:
    db_path = str(tmp_path) + "/keyset.db"
    sync_conn = sqlite3.connect(db_path)
    sync_conn.row_factory = sqlite3.Row
    _ensure_schema(sync_conn)
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
    """Pre-#1750 LIMIT/OFFSET walk — the contract the keyset rewrite preserves."""
    out: list[str] = []
    offset = 0
    yielded = 0
    while True:
        query = "SELECT * FROM messages WHERE session_id = ?"
        params: list[str | int] = [session_id]
        if role_values:
            placeholders = ",".join("?" for _ in role_values)
            query += f" AND role IN ({placeholders})"
            params.extend(role_values)
        query += " ORDER BY (sort_key IS NULL), sort_key, message_id"
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

    assert keyset == canonical
    assert keyset == reference
    assert len(keyset) == len(rows)
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
    sync_conn = sqlite3.connect(db_path)
    sync_conn.row_factory = sqlite3.Row
    _ensure_schema(sync_conn)
    for cid, rows in (
        ("convA", [("a1", 1.0, "user"), ("a2", None, "user"), ("a3", 2.0, "user")]),
        ("convB", [("b1", 1.0, "user"), ("b2", None, "user")]),
    ):
        sync_conn.execute(
            """
            INSERT INTO sessions (
                session_id, provider_session_id, source_name, title, version
            ) VALUES (?, 'p', 'codex', 't', 1)
            """,
            (cid,),
        )
        sync_conn.executemany(
            "INSERT INTO messages (message_id, session_id, role, sort_key, version, content_hash) VALUES (?, ?, ?, ?, 1, ?)",
            [(mid, cid, role, sk, mid) for mid, sk, role in rows],
        )
    sync_conn.commit()
    sync_conn.close()

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        got = [m.message_id async for m in iter_messages(conn, "convA", chunk_size=2)]
        canonical = [m.message_id for m in await get_messages(conn, "convA")]
    assert got == canonical
    assert set(got) == {"a1", "a2", "a3"}


__all__: list[str] = []
