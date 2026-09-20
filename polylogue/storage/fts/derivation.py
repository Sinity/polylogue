"""Domain-owned convergence for the message FTS projection.

The FTS index is a read model of ``blocks``.  This module is the one domain
owner for that relationship: it defines the session partition, derives the
expected input, inspects the persisted membership, and publishes one
partition atomically after checking that the input did not change.

The inspection deliberately reads canonical and FTS tables only.  Freshness
rows, convergence debt, startup state, and scheduler state are observations
of this operation, never its correctness authority.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.core.sqlite_introspection import table_exists
from polylogue.storage.fts.pl_fold import pl_fold_sql_expr
from polylogue.storage.fts.sql import FTS_MESSAGES_IDENTITY_RECIPE_ID


class FtsKeyStatus(StrEnum):
    """Semantic state of one independently replaceable partition."""

    VALID = "valid"
    MISSING = "missing"
    STALE = "stale"
    EXCESS = "excess"


GLOBAL_PARTITION = "__global__"

# Rowids per ``IN`` list.  Bounded by the connection's own variable limit,
# which is 999 on a default SQLite build and must never be assumed larger.
_MAX_ROWID_BATCH = 500


def _rowid_batches(conn: sqlite3.Connection, rowids: Sequence[int]) -> Iterable[tuple[int, ...]]:
    """Yield rowid batches no larger than this connection's bind-variable limit."""
    size = max(1, min(_MAX_ROWID_BATCH, conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)))
    for index in range(0, len(rowids), size):
        yield tuple(rowids[index : index + size])


@dataclass(frozen=True, slots=True)
class FtsInputRow:
    rowid: int
    block_id: str
    message_id: str
    session_id: str
    block_type: str
    search_text: str
    source_hash: bytes | None


@dataclass(frozen=True, slots=True)
class FtsPartitionInput:
    """Recipe- and generation-bound input snapshot used for publication."""

    key: str
    generation: int
    recipe_id: str
    rows: tuple[FtsInputRow, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class FtsPartitionInspection:
    """Authoritative membership facts for one partition."""

    key: str
    status: FtsKeyStatus
    generation: int
    recipe_id: str
    required_rows: int
    present_rows: int
    missing_rows: int
    excess_rows: int
    duplicate_rows: int
    wrong_identity_rows: int
    triggers_compatible: bool
    detail: str | None = None

    @property
    def valid(self) -> bool:
        return self.status is FtsKeyStatus.VALID


@dataclass(frozen=True, slots=True)
class FtsPartitionReplacement:
    """Lease-free session replacement in the kernel's structural vocabulary.

    ``payload`` is the complete value projection read from ``blocks``.  The
    binding includes its recipe and SQLite generation; publication reads the
    same projection again under ``BEGIN IMMEDIATE`` before replacing just this
    session's FTS rows.  Sessions with no searchable blocks are valid without
    a marker row because their correct replacement is empty.
    """

    key: str
    input_binding: str
    payload: FtsPartitionInput
    generation_binding: str
    empty: bool = False


@dataclass(frozen=True, slots=True)
class FtsOrphanReplacement:
    """The low-cadence global residue replacement.

    This is deliberately not an FTS rebuild.  Its only output is the absence
    of docsize-backed FTS rows whose canonical searchable block vanished (and
    their identity companions).  The digest binds the entire residue relation;
    the writer re-reads it before issuing docsize-guarded deletes.
    """

    key: str
    input_binding: str
    payload: FtsOrphanBinding
    generation_binding: str
    empty: bool = False


@dataclass(frozen=True, slots=True)
class FtsOrphanBinding:
    """Constant-memory binding of the two global orphan relations."""

    digest: str
    docsize_rows: int
    identity_rows: int


def _generation(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return 0 if row is None else int(row[0] or 0)


def _has_content_hash(conn: sqlite3.Connection) -> bool:
    return any(str(row[1]) == "content_hash" for row in conn.execute("PRAGMA table_info(blocks)"))


def _session_block_id_range(key: str) -> tuple[str, str]:
    """Half-open ``block_id`` range covering exactly one session's blocks.

    ``block_id`` is ``session_id || ':' || ...``; ``';'`` is the code point
    after ``':'``, so ``[key || ':', key || ';')`` selects the session's rows
    through the ``block_id`` UNIQUE index instead of a ``substr`` table scan.

    The range, not a join to ``blocks``, is what lets a partition see residue
    whose canonical block no longer exists — the exact class of row that must
    be retired. The range alone is not the whole rule: a colon-prefixed child
    session's block ids also fall inside it, so every caller pairs the range
    with ``b.block_id IS NULL OR b.session_id = key`` and claims only residue
    plus its own rows.
    """
    return f"{key}:", f"{key};"


def _indexable_row_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()
    return 0 if row is None else int(row[0] or 0)


_ARCHIVE_MESSAGE_FTS_TRIGGERS = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")
_SESSION_WORK_EVENT_FTS_TRIGGERS = (
    "session_work_events_fts_ai",
    "session_work_events_fts_ad",
    "session_work_events_fts_au",
)


def active_fts_triggers_sync(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Return the FTS triggers expected by the schema currently present."""
    expected: list[str] = []
    if table_exists(conn, "blocks") and table_exists(conn, "messages_fts"):
        expected.extend(_ARCHIVE_MESSAGE_FTS_TRIGGERS)
    if all(table_exists(conn, table_name) for table_name in ("session_work_events", "session_work_events_fts")):
        expected.extend(_SESSION_WORK_EVENT_FTS_TRIGGERS)
    return tuple(expected)


def _schema_compatible(conn: sqlite3.Connection) -> bool:
    if not all(
        table_exists(conn, name) for name in ("blocks", "messages_fts", "messages_fts_docsize", "messages_fts_identity")
    ):
        return False
    expected = _ARCHIVE_MESSAGE_FTS_TRIGGERS
    placeholders = ", ".join("?" for _ in expected)
    row = conn.execute(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})", expected
    ).fetchone()
    return row is not None and int(row[0]) == len(expected)


def _digest(rows: Sequence[FtsInputRow]) -> str:
    payload = [
        [
            row.rowid,
            row.block_id,
            row.message_id,
            row.session_id,
            row.block_type,
            row.search_text,
            None if row.source_hash is None else row.source_hash.hex(),
        ]
        for row in rows
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _orphan_binding(conn: sqlite3.Connection) -> FtsOrphanBinding:
    """Bind FTS residue without retaining an archive-sized rowid collection.

    ``messages_fts`` is contentless, so ``messages_fts_docsize`` is the
    durable SQLite-visible proof that an FTS row exists.  Deletion is therefore
    driven through docsize rather than an unbounded virtual-table rebuild.
    Identity-only residue is listed separately and can be retired without ever
    touching a live FTS row.
    """
    digest = hashlib.sha256()

    def update_rows(label: bytes, sql: str) -> int:
        digest.update(label)
        count = 0
        for row in conn.execute(sql):
            digest.update(f"{int(row[0])},".encode())
            count += 1
        return count

    docsize_rows = update_rows(
        b"docsize:",
        """
        SELECT d.id
        FROM messages_fts_docsize AS d
        LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
        WHERE b.rowid IS NULL
        ORDER BY d.id
        """,
    )
    identity_rows = update_rows(
        b"identity:",
        """
        SELECT i.rowid
        FROM messages_fts_identity AS i
        LEFT JOIN messages_fts_docsize AS d ON d.id = i.rowid
        WHERE d.id IS NULL
        ORDER BY i.rowid
        """,
    )
    return FtsOrphanBinding(digest.hexdigest(), docsize_rows, identity_rows)


def _has_orphan_rows(conn: sqlite3.Connection) -> bool:
    """Bound discovery for the single global residue partition."""
    return any(
        int(conn.execute(sql).fetchone()[0] or 0)
        for sql in (
            """
            SELECT EXISTS(
                SELECT 1 FROM messages_fts_docsize AS d
                LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
                WHERE b.rowid IS NULL
            )
            """,
            """
            SELECT EXISTS(
                SELECT 1 FROM messages_fts_identity AS i
                LEFT JOIN messages_fts_docsize AS d ON d.id = i.rowid
                WHERE d.id IS NULL
            )
            """,
        )
    )


class FtsDerivationAdapter:
    """Message FTS as a structural ``DerivationAdapter`` without daemon imports.

    The storage ring owns the queries and the one-session transaction.  The
    daemon's common kernel supplies only a frame, bounded key paging and a
    publication admission callback.  Keeping the connection factories here
    lets computation use read snapshots without acquiring the writer lease.
    """

    domain = "messages_fts"
    name = domain
    recipe_id = FTS_MESSAGES_IDENTITY_RECIPE_ID
    prerequisites: tuple[str, ...] = ()

    def __init__(
        self,
        read_connection: Callable[[], sqlite3.Connection] | None = None,
        write_connection: Callable[[], sqlite3.Connection] | None = None,
        *,
        generation_binding: Callable[[], str] | None = None,
        orphan_interval_s: float | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._read_connection = read_connection
        self._write_connection = write_connection
        self._generation_binding = generation_binding
        self._orphan_interval_s = orphan_interval_s
        self._monotonic = monotonic
        self._last_orphan_attempt: float | None = None

    def _connections(self) -> tuple[Callable[[], sqlite3.Connection], Callable[[], sqlite3.Connection]]:
        if self._read_connection is None or self._write_connection is None:
            raise RuntimeError("FTS derivation adapter requires read and write connection factories")
        return self._read_connection, self._write_connection

    def _frame_current(self, frame: object) -> bool:
        if self._generation_binding is None:
            return True
        return getattr(frame, "source_revision", None) == f"index-generation:{self._generation_binding()}"

    def _frame_recipe_current(self, frame: object) -> bool:
        versions = getattr(frame, "recipe_versions", {})
        return isinstance(versions, Mapping) and versions.get(self.domain) == self.recipe_id

    def required_partitions(self, conn: sqlite3.Connection) -> tuple[str, ...]:
        """Return all session keys, including sessions with valid empty output."""
        if not table_exists(conn, "blocks"):
            return ()
        keys: set[str] = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT session_id FROM blocks WHERE session_id IS NOT NULL")
        }
        if table_exists(conn, "sessions"):
            keys.update(str(row[0]) for row in conn.execute("SELECT session_id FROM sessions"))
        return tuple(sorted(keys))

    def input_for(self, conn: sqlite3.Connection, key: str) -> FtsPartitionInput:
        """Bind every canonical input value, not only its identifier."""
        if key == GLOBAL_PARTITION:
            where = "b.search_text != ''"
            params: tuple[object, ...] = ()
        else:
            where = "b.session_id = ? AND b.search_text != ''"
            params = (key,)
        hash_expr = "b.content_hash" if _has_content_hash(conn) else "NULL"
        rows = tuple(
            FtsInputRow(
                rowid=int(row[0]),
                block_id=str(row[1]),
                message_id=str(row[2]),
                session_id=str(row[3]),
                block_type=str(row[4]),
                search_text=str(row[5]),
                source_hash=None if row[6] is None else bytes(row[6]),
            )
            for row in conn.execute(
                f"""
                SELECT b.rowid, b.block_id, b.message_id, b.session_id, b.block_type,
                       b.search_text, {hash_expr}
                FROM blocks AS b
                WHERE {where}
                ORDER BY b.rowid
                """,
                params,
            ).fetchall()
        )
        return FtsPartitionInput(key, _generation(conn), self.recipe_id, rows, _digest(rows))

    def inspect_partition(self, conn: sqlite3.Connection, key: str) -> FtsPartitionInspection:
        """Inspect membership against ``blocks`` without consulting state tables."""
        generation = _generation(conn)
        compatible = _schema_compatible(conn)
        # Inspection counts the two relations; it never materializes the input
        # projection.  ``input_for`` reads and hashes every block's
        # ``search_text``, which is what a *replacement* is bound to -- an
        # inspection only has to answer whether membership and identity agree
        # with ``blocks``, and every one of those questions is a COUNT over an
        # indexed join.  Canonical writers call this on every unchanged
        # re-ingest, where hashing the session's whole text would dominate the
        # write and buy no extra evidence.
        if key == GLOBAL_PARTITION or not table_exists(conn, "blocks"):
            expected_rows = (
                _indexable_row_count(conn) if key == GLOBAL_PARTITION and table_exists(conn, "blocks") else 0
            )
        else:
            expected_rows = int(
                conn.execute(
                    "SELECT COUNT(*) FROM blocks WHERE session_id = ? AND search_text != ''",
                    (key,),
                ).fetchone()[0]
            )
        if not compatible:
            return FtsPartitionInspection(
                key,
                FtsKeyStatus.MISSING,
                generation,
                self.recipe_id,
                expected_rows,
                0,
                expected_rows,
                0,
                0,
                0,
                False,
                "FTS schema or canonical trigger set is incompatible",
            )

        if key == GLOBAL_PARTITION:
            present_rows = int(conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0])
            missing_rows = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM blocks AS b
                    LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                    WHERE b.search_text != '' AND d.id IS NULL
                    """
                ).fetchone()[0]
            )
            docsize_excess_rows = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM messages_fts_docsize AS d
                    LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
                    WHERE b.rowid IS NULL
                    """
                ).fetchone()[0]
            )
            identity_excess_rows = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM messages_fts_identity AS i
                    LEFT JOIN messages_fts_docsize AS d ON d.id = i.rowid
                    WHERE d.id IS NULL
                    """
                ).fetchone()[0]
            )
            excess_rows = docsize_excess_rows + identity_excess_rows
            wrong_rows = (
                int(
                    conn.execute(
                        """
                    SELECT COUNT(*) FROM blocks AS b
                    JOIN messages_fts_docsize AS d ON d.id = b.rowid
                    LEFT JOIN messages_fts_identity AS i ON i.rowid = b.rowid
                    WHERE b.search_text != '' AND (
                        i.rowid IS NULL OR i.block_id != b.block_id
                        OR i.source_hash IS NOT b.content_hash OR i.recipe_id != ?
                    )
                    """,
                        (self.recipe_id,),
                    ).fetchone()[0]
                )
                if _has_content_hash(conn)
                else int(
                    conn.execute(
                        """
                    SELECT COUNT(*) FROM blocks AS b
                    JOIN messages_fts_docsize AS d ON d.id = b.rowid
                    LEFT JOIN messages_fts_identity AS i ON i.rowid = b.rowid
                    WHERE b.search_text != '' AND (i.rowid IS NULL OR i.block_id != b.block_id OR i.recipe_id != ?)
                    """,
                        (self.recipe_id,),
                    ).fetchone()[0]
                )
            )
        else:
            missing_rows = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM blocks AS b
                    LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
                    WHERE b.session_id = ? AND b.search_text != '' AND d.id IS NULL
                    """,
                    (key,),
                ).fetchone()[0]
            )
            present_rows = expected_rows - missing_rows
            excess_rows = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM messages_fts_identity AS i
                    JOIN messages_fts_docsize AS d ON d.id = i.rowid
                    LEFT JOIN blocks AS b ON b.block_id = i.block_id
                    WHERE i.block_id >= ? AND i.block_id < ?
                      AND (b.block_id IS NULL OR (b.session_id = ? AND b.search_text = ''))
                    """,
                    (*_session_block_id_range(key), key),
                ).fetchone()[0]
            )
            wrong_rows = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) FROM blocks AS b
                    JOIN messages_fts_docsize AS d ON d.id = b.rowid
                    LEFT JOIN messages_fts_identity AS i ON i.rowid = b.rowid
                    WHERE b.session_id = ? AND b.search_text != '' AND (
                        i.rowid IS NULL OR i.block_id != b.block_id OR i.recipe_id != ?
                        OR {"i.source_hash IS NOT b.content_hash OR" if _has_content_hash(conn) else ""} 0
                    )
                    """,
                    (key, self.recipe_id),
                ).fetchone()[0]
            )
        duplicate_sql = (
            "SELECT COALESCE(SUM(n - 1), 0) FROM ("
            "SELECT block_id, COUNT(*) AS n FROM messages_fts_identity "
            "GROUP BY block_id HAVING n > 1)"
        )
        duplicate_params: tuple[object, ...] = ()
        if key != GLOBAL_PARTITION:
            duplicate_sql = (
                "SELECT COALESCE(SUM(n - 1), 0) FROM ("
                "SELECT i.block_id, COUNT(*) AS n FROM messages_fts_identity AS i "
                "LEFT JOIN blocks AS b ON b.block_id = i.block_id "
                "WHERE i.block_id >= ? AND i.block_id < ? "
                "AND (b.block_id IS NULL OR b.session_id = ?) "
                "GROUP BY i.block_id HAVING n > 1)"
            )
            duplicate_params = (*_session_block_id_range(key), key)
        duplicate_rows = int(conn.execute(duplicate_sql, duplicate_params).fetchone()[0])
        status = FtsKeyStatus.VALID
        detail: str | None = None
        if missing_rows or wrong_rows or duplicate_rows:
            status = FtsKeyStatus.STALE
            detail = "FTS membership or identity differs from canonical blocks"
        elif excess_rows:
            status = FtsKeyStatus.EXCESS
            detail = "FTS contains rows without canonical searchable blocks"
        return FtsPartitionInspection(
            key,
            status,
            generation,
            self.recipe_id,
            expected_rows,
            present_rows,
            missing_rows,
            excess_rows,
            duplicate_rows,
            wrong_rows,
            compatible,
            detail,
        )

    def inspect_all(
        self, conn: sqlite3.Connection, *, keys: Iterable[str] | None = None
    ) -> tuple[FtsPartitionInspection, ...]:
        selected = tuple(sorted(dict.fromkeys(keys))) if keys is not None else self.required_partitions(conn)
        inspections = [self.inspect_partition(conn, key) for key in selected]
        if keys is None:
            global_state = self.inspect_partition(conn, GLOBAL_PARTITION)
            if global_state.excess_rows or not global_state.triggers_compatible:
                inspections.append(global_state)
        return tuple(inspections)

    def publish_partition(self, conn: sqlite3.Connection, computed: FtsPartitionInput) -> bool:
        """Atomically replace one partition, returning false on revalidation drift."""
        owns_transaction = not conn.in_transaction
        if owns_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.input_for(conn, computed.key)
            if current != computed:
                if owns_transaction:
                    conn.execute("ROLLBACK")
                return False
            if not _schema_compatible(conn):
                if owns_transaction:
                    conn.execute("ROLLBACK")
                return False
            if computed.key == GLOBAL_PARTITION:
                raise ValueError("the global FTS key only retires docsize-backed orphan residue")
            else:
                rowids = {
                    int(row[0])
                    for row in conn.execute("SELECT rowid FROM blocks WHERE session_id = ?", (computed.key,))
                }
                rowids.update(
                    int(row[0])
                    for row in conn.execute(
                        """
                        SELECT i.rowid FROM messages_fts_identity AS i
                        LEFT JOIN blocks AS b ON b.block_id = i.block_id
                        WHERE i.block_id >= ? AND i.block_id < ?
                          AND (b.block_id IS NULL OR b.session_id = ?)
                        """,
                        (*_session_block_id_range(computed.key), computed.key),
                    )
                )
                for batch in _rowid_batches(conn, sorted(rowids)):
                    placeholders = ", ".join("?" for _ in batch)
                    conn.execute(f"DELETE FROM messages_fts WHERE rowid IN ({placeholders})", batch)
                    conn.execute(f"DELETE FROM messages_fts_identity WHERE rowid IN ({placeholders})", batch)
                self._insert_rows(conn, computed.key)
            if owns_transaction:
                conn.execute("COMMIT")
            return True
        except Exception:
            if owns_transaction and conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _insert_rows(self, conn: sqlite3.Connection, key: str) -> None:
        if key == GLOBAL_PARTITION:
            where = "b.search_text != ''"
            params: tuple[object, ...] = ()
        else:
            where = "b.session_id = ? AND b.search_text != ''"
            params = (key,)
        conn.execute(
            f"""
            INSERT INTO messages_fts(rowid, text)
            SELECT b.rowid, {pl_fold_sql_expr("b.search_text")}
            FROM blocks AS b WHERE {where}
            """,
            params,
        )
        if _has_content_hash(conn):
            conn.execute(
                f"""
                INSERT OR REPLACE INTO messages_fts_identity(rowid, block_id, source_hash, recipe_id)
                SELECT b.rowid, b.block_id, b.content_hash, ? FROM blocks AS b WHERE {where}
                """,
                (self.recipe_id, *params),
            )
        else:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO messages_fts_identity(rowid, block_id, source_hash, recipe_id)
                SELECT b.rowid, b.block_id, NULL, ? FROM blocks AS b WHERE {where}
                """,
                (self.recipe_id, *params),
            )

    # The following methods intentionally mirror the daemon kernel protocol
    # without importing it.  ``DerivationFrame`` and ``ReplacementLike`` are
    # structural boundaries so storage remains below daemon in the layering
    # graph.

    def required_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Keyset-page required session partitions, including valid-empty sessions."""
        if limit < 1:
            raise ValueError("FTS derivation page limit must be positive")
        read_connection, _ = self._connections()
        scope = getattr(frame, "scope", None)
        if scope is not None:
            if not isinstance(scope, tuple):
                raise TypeError("FTS derivation frame scope must be a tuple of session ids or None")
            keys = tuple(sorted(dict.fromkeys(str(key) for key in scope)))
            start = 0 if cursor is None else next((i for i, key in enumerate(keys) if key > cursor), len(keys))
            page = keys[start : start + limit]
            return page, (page[-1] if start + len(page) < len(keys) and page else None)
        conn = read_connection()
        try:
            rows = conn.execute(
                """
                SELECT session_id FROM (
                    SELECT session_id FROM sessions WHERE session_id > ?
                    UNION
                    SELECT session_id FROM blocks WHERE session_id IS NOT NULL AND session_id > ?
                )
                ORDER BY session_id
                LIMIT ?
                """,
                (cursor or "", cursor or "", limit),
            ).fetchall()
            keys = tuple(str(row[0]) for row in rows)
            return keys, (keys[-1] if len(keys) == limit else None)
        finally:
            conn.close()

    def excess_page(self, frame: object, *, cursor: str | None, limit: int) -> tuple[tuple[str, ...], str | None]:
        """Expose one global residue key only when docsize proves excess exists."""
        del frame
        if cursor is not None or limit < 1:
            return (), None
        now: float | None = None
        if self._orphan_interval_s is not None:
            if self._monotonic is None:
                raise RuntimeError("FTS orphan cadence requires a monotonic clock")
            now = self._monotonic()
            if self._last_orphan_attempt is not None and now - self._last_orphan_attempt < self._orphan_interval_s:
                return (), None
        read_connection, _ = self._connections()
        conn = read_connection()
        try:
            if not _schema_compatible(conn):
                return (), None
            found = _has_orphan_rows(conn)
        finally:
            conn.close()
        if now is not None:
            # A scheduling hint only, stamped after the probe actually ran: a
            # pass that never reached the probe must not burn the interval. It
            # is not written to SQLite and cannot certify readiness; direct
            # readiness inspection still sees residue immediately.
            self._last_orphan_attempt = now
        return ((GLOBAL_PARTITION,) if found else ()), None

    def quiet(self, frame: object, key: str) -> bool:
        """FTS has no hot-source policy; orphan cadence is discovery-only."""
        del frame
        del key
        return False

    def prerequisite_keys(self, frame: object, key: str) -> tuple[()]:
        del frame, key
        return ()

    def inspect(self, frame: object, keys: Sequence[str]) -> Mapping[str, str]:
        """Classify keys against the bound generation and each partition's output."""
        if not self._frame_current(frame) or not self._frame_recipe_current(frame):
            raise RuntimeError("FTS inspection frame generation or recipe changed")
        read_connection, _ = self._connections()
        conn = read_connection()
        try:
            conn.execute("BEGIN")
            statuses: dict[str, str] = {}
            for key in keys:
                if key == GLOBAL_PARTITION:
                    # This key owns only orphan residue. A poisoned session's
                    # missing membership cannot make successful retirement fail.
                    if not _schema_compatible(conn):
                        raise RuntimeError("FTS schema or canonical trigger set is incompatible")
                    statuses[key] = "excess" if _has_orphan_rows(conn) else "missing"
                else:
                    statuses[key] = self.inspect_partition(conn, key).status.value
            if not self._frame_current(frame):
                raise RuntimeError("FTS index generation changed during inspection")
            return statuses
        finally:
            conn.close()

    def compute(self, frame: object, key: str) -> FtsPartitionReplacement | FtsOrphanReplacement:
        """Read one replacement from a stable snapshot while no writer is held."""
        if not self._frame_current(frame):
            raise RuntimeError("FTS frame names a retired index generation")
        if not self._frame_recipe_current(frame):
            raise RuntimeError("FTS frame recipe does not match the active FTS recipe")
        read_connection, _ = self._connections()
        generation_binding = self._generation_binding() if self._generation_binding is not None else ""
        conn = read_connection()
        try:
            conn.execute("BEGIN")
            if key == GLOBAL_PARTITION:
                residue = _orphan_binding(conn)
                replacement: FtsPartitionReplacement | FtsOrphanReplacement = FtsOrphanReplacement(
                    key=key,
                    input_binding=residue.digest,
                    payload=residue,
                    generation_binding=generation_binding,
                )
            else:
                input_snapshot = self.input_for(conn, key)
                replacement = FtsPartitionReplacement(
                    key=key,
                    input_binding=hashlib.sha256(
                        f"{input_snapshot.generation}:{input_snapshot.recipe_id}:{input_snapshot.digest}".encode()
                    ).hexdigest(),
                    payload=input_snapshot,
                    generation_binding=generation_binding,
                    empty=not input_snapshot.rows,
                )
        finally:
            conn.close()
        if not self._frame_current(frame):
            raise RuntimeError("FTS index generation changed while computing a replacement")
        return replacement

    def publish(self, frame: object, replacement: object) -> bool:
        """Revalidate one frame-bound replacement and replace only its partition."""
        if not isinstance(replacement, (FtsPartitionReplacement, FtsOrphanReplacement)):
            raise TypeError(f"expected FTS replacement, got {type(replacement).__name__}")
        if not self._frame_current(frame) or not self._frame_recipe_current(frame):
            return False
        if self._generation_binding is not None and replacement.generation_binding != self._generation_binding():
            return False
        _, write_connection = self._connections()
        conn = write_connection()
        try:
            if replacement.generation_binding:
                database_row = conn.execute("PRAGMA database_list").fetchone()
                if database_row is None or Path(str(database_row[2])).resolve() != Path(replacement.generation_binding):
                    return False
            if isinstance(replacement, FtsOrphanReplacement):
                return self._publish_orphans(conn, replacement)
            return self.publish_partition(conn, replacement.payload)
        finally:
            conn.close()

    def _publish_orphans(self, conn: sqlite3.Connection, replacement: FtsOrphanReplacement) -> bool:
        """Delete global FTS residue, never rebuild the virtual table."""
        owns_transaction = not conn.in_transaction
        if owns_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            if not _schema_compatible(conn) or _orphan_binding(conn) != replacement.payload:
                if owns_transaction:
                    conn.execute("ROLLBACK")
                return False
            # The FTS deletion is guarded by its docsize shadow relation.  A
            # current canonical searchable block cannot be selected, even if a
            # stale reader prepared this global key before a rowid was reused.
            conn.execute(
                """
                DELETE FROM messages_fts
                WHERE rowid IN (
                    SELECT d.id
                    FROM messages_fts_docsize AS d
                    LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
                    WHERE b.rowid IS NULL
                )
                """
            )
            conn.execute(
                """
                DELETE FROM messages_fts_identity
                WHERE rowid NOT IN (SELECT id FROM messages_fts_docsize)
                """
            )
            if owns_transaction:
                conn.execute("COMMIT")
            return True
        except Exception:
            if owns_transaction and conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


FtsDomainAdapter = FtsDerivationAdapter


def replace_fts_partition_sync(conn: sqlite3.Connection, session_id: str) -> bool:
    """Replace one canonical-write FTS partition through the derivation SQL.

    Live writes already own their transaction and must retain immediate search
    membership.  They use the same input binding and one-partition publisher
    as daemon convergence, without a second repair SQL implementation.
    """
    adapter = FtsDerivationAdapter()
    return adapter.publish_partition(conn, adapter.input_for(conn, session_id))


def session_partition_is_valid_sync(conn: sqlite3.Connection, session_id: str) -> bool:
    """Whether one session's FTS partition already agrees with ``blocks``.

    This is the domain's own authoritative inspection, not a second probe:
    membership, identity and duplicate rules all come from
    ``inspect_partition``.  A canonical writer that re-ingested unchanged
    content asks this to decide whether it must republish the partition.
    """
    if not session_id:
        return True
    return FtsDerivationAdapter().inspect_partition(conn, session_id).valid


def converge_fts_partition_sync(conn: sqlite3.Connection, session_id: str) -> bool:
    """Bring one session's FTS partition to valid, reporting whether work ran.

    Inspection and replacement are the domain's, so a canonical write path
    carries no repair SQL or staleness rule of its own.  An incompatible FTS
    schema is a readiness failure rather than something a write repairs:
    ``publish_partition`` refuses, and the daemon's FTS derivation reports it.
    """
    if not session_id:
        return False
    adapter = FtsDerivationAdapter()
    if adapter.inspect_partition(conn, session_id).valid:
        return False
    return adapter.publish_partition(conn, adapter.input_for(conn, session_id))


__all__ = [
    "GLOBAL_PARTITION",
    "FtsDomainAdapter",
    "FtsDerivationAdapter",
    "FtsInputRow",
    "FtsKeyStatus",
    "FtsOrphanReplacement",
    "FtsPartitionInput",
    "FtsPartitionInspection",
    "FtsPartitionReplacement",
    "active_fts_triggers_sync",
    "converge_fts_partition_sync",
    "replace_fts_partition_sync",
    "session_partition_is_valid_sync",
]
