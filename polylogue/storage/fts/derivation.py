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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from polylogue.storage.fts.pl_fold import pl_fold_sql_expr
from polylogue.storage.fts.sql import FTS_MESSAGES_IDENTITY_RECIPE_ID
from polylogue.storage.introspection import table_exists


class FtsKeyStatus(StrEnum):
    """Semantic state of one independently replaceable partition."""

    VALID = "valid"
    MISSING = "missing"
    STALE = "stale"
    EXCESS = "excess"


class FtsOutcome(StrEnum):
    """Result of one bounded convergence operation."""

    DONE = "done"
    PENDING = "pending"
    FAILED = "failed"


GLOBAL_PARTITION = "__global__"


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
class FtsConvergenceResult:
    """Bounded result returned by the recurring owner."""

    outcome: FtsOutcome
    partitions: tuple[FtsPartitionInspection, ...]
    written_partitions: int = 0
    detail: str | None = None

    @property
    def ready(self) -> bool:
        return self.outcome is FtsOutcome.DONE and all(partition.valid for partition in self.partitions)


def _generation(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return 0 if row is None else int(row[0] or 0)


def _has_content_hash(conn: sqlite3.Connection) -> bool:
    return any(str(row[1]) == "content_hash" for row in conn.execute("PRAGMA table_info(blocks)"))


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


class FtsDerivationAdapter:
    """The message FTS domain adapter and its complete write protocol."""

    name = "messages_fts"
    recipe_id = FTS_MESSAGES_IDENTITY_RECIPE_ID

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

    def inspect(self, conn: sqlite3.Connection, key: str) -> FtsPartitionInspection:
        """Inspect membership against ``blocks`` without consulting state tables."""
        generation = _generation(conn)
        compatible = _schema_compatible(conn)
        expected = (
            self.input_for(conn, key)
            if table_exists(conn, "blocks")
            else FtsPartitionInput(key, generation, self.recipe_id, (), _digest(()))
        )
        if not compatible:
            return FtsPartitionInspection(
                key,
                FtsKeyStatus.MISSING,
                generation,
                self.recipe_id,
                len(expected.rows),
                0,
                len(expected.rows),
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
            excess_rows = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM messages_fts_docsize AS d
                    LEFT JOIN blocks AS b ON b.rowid = d.id AND b.search_text != ''
                    WHERE b.rowid IS NULL
                    """
                ).fetchone()[0]
            )
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
            required = tuple(row.rowid for row in expected.rows)
            placeholders = ", ".join("?" for _ in required)
            present_rows = (
                0
                if not required
                else int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM messages_fts_docsize WHERE id IN ({placeholders})", required
                    ).fetchone()[0]
                )
            )
            missing_rows = len(required) - present_rows
            excess_rows = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM messages_fts_identity AS i
                    JOIN messages_fts_docsize AS d ON d.id = i.rowid
                    LEFT JOIN blocks AS b ON b.block_id = i.block_id
                    WHERE substr(i.block_id, 1, length(?) + 1) = ? || ':'
                      AND (b.block_id IS NULL OR b.session_id != ? OR b.search_text = '')
                    """,
                    (key, key, key),
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
                "SELECT block_id, COUNT(*) AS n FROM messages_fts_identity "
                "WHERE substr(block_id, 1, length(?) + 1) = ? || ':' "
                "GROUP BY block_id HAVING n > 1)"
            )
            duplicate_params = (key, key)
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
            len(expected.rows),
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
        inspections = [self.inspect(conn, key) for key in selected]
        if keys is None:
            global_state = self.inspect(conn, GLOBAL_PARTITION)
            if global_state.excess_rows or not global_state.triggers_compatible:
                inspections.append(global_state)
        return tuple(inspections)

    def publish(self, conn: sqlite3.Connection, computed: FtsPartitionInput) -> bool:
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
                conn.execute("DELETE FROM messages_fts")
                conn.execute("DELETE FROM messages_fts_identity")
                self._insert_rows(conn, GLOBAL_PARTITION)
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
                        WHERE substr(i.block_id, 1, length(?) + 1) = ? || ':'
                        """,
                        (computed.key, computed.key),
                    )
                )
                if rowids:
                    placeholders = ", ".join("?" for _ in rowids)
                    params = tuple(sorted(rowids))
                    conn.execute(f"DELETE FROM messages_fts WHERE rowid IN ({placeholders})", params)
                    conn.execute(f"DELETE FROM messages_fts_identity WHERE rowid IN ({placeholders})", params)
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
            INSERT INTO messages_fts(rowid, block_id, message_id, session_id, block_type, text)
            SELECT b.rowid, b.block_id, b.message_id, b.session_id, b.block_type,
                   {pl_fold_sql_expr("b.search_text")}
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

    def converge(
        self, conn: sqlite3.Connection, *, keys: Sequence[str] | None = None, limit: int | None = None
    ) -> FtsConvergenceResult:
        """Inspect and publish a bounded set of stale partitions."""
        inspections = self.inspect_all(conn, keys=keys)
        candidates = [inspection for inspection in inspections if not inspection.valid]
        if limit is not None:
            candidates = candidates[: max(0, int(limit))]
        if not candidates:
            return FtsConvergenceResult(FtsOutcome.DONE, inspections)
        written = 0
        for inspection in candidates:
            computed = self.input_for(conn, inspection.key)
            if inspection.key == GLOBAL_PARTITION or computed.rows or inspection.status is not FtsKeyStatus.VALID:
                if not self.publish(conn, computed):
                    return FtsConvergenceResult(
                        FtsOutcome.PENDING,
                        self.inspect_all(conn, keys=keys),
                        written,
                        "canonical input changed before publish",
                    )
                written += 1
        remaining = self.inspect_all(conn, keys=keys)
        if any(not inspection.valid for inspection in remaining):
            return FtsConvergenceResult(FtsOutcome.PENDING, remaining, written, "bounded FTS convergence remains")
        return FtsConvergenceResult(FtsOutcome.DONE, remaining, written)


FtsDomainAdapter = FtsDerivationAdapter


__all__ = [
    "GLOBAL_PARTITION",
    "FtsConvergenceResult",
    "FtsDomainAdapter",
    "FtsDerivationAdapter",
    "FtsInputRow",
    "FtsKeyStatus",
    "FtsOutcome",
    "FtsPartitionInput",
    "FtsPartitionInspection",
    "active_fts_triggers_sync",
]
