"""Disposable SQLite lineage graph for one Claude web normalization pass."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import cast

_GRAPH_TABLES = ("claude_graph_node", "claude_graph_occurrence", "claude_graph_native", "claude_graph_summary")

_NODE_COLUMNS = (
    "evidence_key, seq, native_id, original_index, timestamp, updated_at, parent, explicit_position, "
    "explicit_branch, explicit_variant, explicit_active_path, explicit_active_leaf, has_material, "
    "has_attachments, has_compaction_summary, score, updated_score, ts_sort, updated_sort"
)


def _optional_flag(value: bool | None) -> int | None:
    return None if value is None else int(value)


def _flag(value: object) -> bool | None:
    return None if value is None else bool(value)


@dataclass(frozen=True, slots=True)
class LineageNode:
    """The lineage and ranking fields of one record, without its content."""

    evidence_key: str
    native_id: str
    original_index: int
    timestamp: str | None
    updated_at: str | None
    parent: str | None
    explicit_position: int | None
    explicit_branch: int | None
    explicit_variant: int | None
    explicit_active_path: bool | None
    explicit_active_leaf: bool | None
    has_material: bool
    has_attachments: bool
    has_compaction_summary: bool
    score: int
    updated_score: float
    ts_sort: float
    updated_sort: float


@dataclass(frozen=True, slots=True)
class _Chase:
    """Where a parent walk from one record ended.

    Without a cycle, ``steps`` records precede the stopping record and
    ``value`` is what it answered; ``terminal`` says whether that record
    still needs the value itself. With a cycle, ``steps`` is the tail before
    it and ``cycle_length`` its length.
    """

    steps: int
    value: int = 0
    terminal: bool = False
    cycle_length: int = 0


@dataclass(frozen=True, slots=True)
class GraphRow:
    evidence_key: str
    original_index: int
    position: int
    branch_index: int
    variant_index: int
    explicit_active_path: bool | None
    explicit_active_leaf: bool | None
    on_path: bool


class ClaudeLineageGraph:
    """Every record's lineage and ranking fields, in SQLite.

    The graph phase (duplicates, depth, sibling rank, variants, active path)
    reads only these fields, while each record's content stays in its
    evidence store. Parent walks keep one cursor rather than a chain, so a
    conversation of any length leaves one record and the statement cursors
    resident. The object parser runs the same statements over an in-memory
    connection; a streamed preparation passes its scratch connection.
    """

    def __init__(self, conn: sqlite3.Connection | None) -> None:
        self._owned = conn is None
        self.conn = conn if conn is not None else sqlite3.connect(":memory:")
        # ``seq`` is first-seen order: a richer repeat replaces a record's
        # fields but keeps its place, as a dict assignment would.
        self.conn.execute(
            "CREATE TABLE claude_graph_node (evidence_key TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, "
            "native_id TEXT NOT NULL, original_index INTEGER NOT NULL, timestamp TEXT, updated_at TEXT, parent TEXT, "
            "explicit_position INTEGER, explicit_branch INTEGER, explicit_variant INTEGER, "
            "explicit_active_path INTEGER, explicit_active_leaf INTEGER, has_material INTEGER NOT NULL, "
            "has_attachments INTEGER NOT NULL, has_compaction_summary INTEGER NOT NULL, score INTEGER NOT NULL, "
            "updated_score REAL NOT NULL, ts_sort REAL NOT NULL, updated_sort REAL NOT NULL, depth INTEGER, "
            "position INTEGER, branch_index INTEGER, variant_memo INTEGER, variant_index INTEGER, "
            "on_path INTEGER NOT NULL DEFAULT 0)"
        )
        self.conn.execute("CREATE TABLE claude_graph_occurrence (base TEXT PRIMARY KEY, n INTEGER NOT NULL)")
        self.conn.execute("CREATE TABLE claude_graph_native (native_id TEXT PRIMARY KEY, n INTEGER NOT NULL)")
        self.conn.execute(
            "CREATE TABLE claude_graph_summary (seq INTEGER PRIMARY KEY, base TEXT NOT NULL, position INTEGER NOT NULL, "
            "payload_json TEXT NOT NULL, timestamp TEXT, original_index INTEGER NOT NULL, evidence_key TEXT NOT NULL)"
        )
        self._seq = 0
        self.missing_native_id = False

    def close(self) -> None:
        if self._owned:
            self.conn.close()
            return
        for table in _GRAPH_TABLES:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")

    # -- intake ---------------------------------------------------------------

    def occurrence_key(self, base_evidence_key: str) -> str:
        row = self.conn.execute("SELECT n FROM claude_graph_occurrence WHERE base = ?", (base_evidence_key,)).fetchone()
        occurrence = int(row[0]) if row is not None else 0
        self.conn.execute(
            "INSERT INTO claude_graph_occurrence VALUES (?, 1) ON CONFLICT(base) DO UPDATE SET n = n + 1",
            (base_evidence_key,),
        )
        return base_evidence_key if occurrence == 0 else f"{base_evidence_key}:occurrence:{occurrence}"

    def observe(self, node: LineageNode, richer_on_tie: Callable[[int, int], bool]) -> None:
        """Keep the richest record for each evidence key.

        Records that tie on richness go to ``richer_on_tie(candidate, retained)``,
        called with both records' original indexes.
        """
        if not node.native_id:
            self.missing_native_id = True
        else:
            self.conn.execute(
                "INSERT INTO claude_graph_native VALUES (?, 1) ON CONFLICT(native_id) DO UPDATE SET n = n + 1",
                (node.native_id,),
            )
        existing = self.conn.execute(
            "SELECT seq, score, updated_score, original_index FROM claude_graph_node WHERE evidence_key = ?",
            (node.evidence_key,),
        ).fetchone()
        if existing is not None:
            if (node.score, node.updated_score) != (existing[1], existing[2]):
                richer = (node.score, node.updated_score) > (existing[1], existing[2])
            else:
                richer = richer_on_tie(node.original_index, int(existing[3]))
            if not richer:
                return
            self.conn.execute("DELETE FROM claude_graph_node WHERE evidence_key = ?", (node.evidence_key,))
            seq = int(existing[0])
        else:
            seq = self._seq
            self._seq += 1
        self.conn.execute(
            f"INSERT INTO claude_graph_node ({_NODE_COLUMNS}) VALUES ({', '.join('?' * 19)})",
            (
                node.evidence_key,
                seq,
                node.native_id,
                node.original_index,
                node.timestamp,
                node.updated_at,
                node.parent,
                node.explicit_position,
                node.explicit_branch,
                node.explicit_variant,
                _optional_flag(node.explicit_active_path),
                _optional_flag(node.explicit_active_leaf),
                int(node.has_material),
                int(node.has_attachments),
                int(node.has_compaction_summary),
                node.score,
                node.updated_score,
                node.ts_sort,
                node.updated_sort,
            ),
        )

    # -- walks ----------------------------------------------------------------

    def _parent(self, evidence_key: str) -> str | None:
        """The record's parent, when that parent is itself a record."""
        row = self.conn.execute(
            "SELECT p.evidence_key FROM claude_graph_node AS n "
            "JOIN claude_graph_node AS p ON p.evidence_key = n.parent WHERE n.evidence_key = ?",
            (evidence_key,),
        ).fetchone()
        return str(row[0]) if row is not None else None

    def _chase(self, start: str, probe: Callable[[str], tuple[str, object]]) -> _Chase:
        """Follow parents from ``start`` with Brent's cycle detection.

        ``probe(key)`` answers ``("known", value)`` or ``("terminal", value)``
        to stop at that record, or ``("next", parent)`` to continue.
        """
        power = length = 1
        tortoise = hare = start
        steps = 0
        while True:
            kind, answer = probe(hare)
            if kind != "next":
                assert isinstance(answer, int)
                return _Chase(steps, answer, terminal=kind == "terminal")
            if length == power:
                tortoise = hare
                power *= 2
                length = 0
            hare = str(answer)
            steps += 1
            length += 1
            if hare == tortoise:
                break
        tortoise = hare = start
        for _ in range(length):
            hare = cast(str, self._parent(hare))
        tail = 0
        while tortoise != hare:
            tortoise = cast(str, self._parent(tortoise))
            hare = cast(str, self._parent(hare))
            tail += 1
        return _Chase(tail, cycle_length=length)

    def _unresolved(self, column: str, *, emitted_only: bool) -> Iterator[str]:
        """Records whose ``column`` is unset, in first-seen order, rechecked as walks fill it."""
        material = "AND has_material " if emitted_only else ""
        last = -1
        while True:
            row = self.conn.execute(
                f"SELECT seq, evidence_key FROM claude_graph_node WHERE seq > ? AND {column} IS NULL "
                f"{material}ORDER BY seq LIMIT 1",
                (last,),
            ).fetchone()
            if row is None:
                return
            last = int(row[0])
            yield str(row[1])

    def _assign(self, column: str, start: str, values: Iterable[int]) -> None:
        cursor: str | None = start
        for value in values:
            assert cursor is not None
            self.conn.execute(f"UPDATE claude_graph_node SET {column} = ? WHERE evidence_key = ?", (value, cursor))
            cursor = self._parent(cursor)

    def _depth_probe(self, evidence_key: str) -> tuple[str, object]:
        row = self.conn.execute(
            "SELECT n.depth, p.evidence_key FROM claude_graph_node AS n "
            "LEFT JOIN claude_graph_node AS p ON p.evidence_key = n.parent WHERE n.evidence_key = ?",
            (evidence_key,),
        ).fetchone()
        if row[0] is not None:
            return "known", int(row[0])
        if row[1] is None:
            return "terminal", 0
        return "next", row[1]

    def _lineage_depths(self) -> bool:
        """Set every record's parent depth; a cycle's records get depth zero.

        A record whose parent is no record has depth zero; any other has its
        parent's depth plus one, and a record leading into a cycle counts
        from the cycle. Returns whether any cycle was found.
        """
        cycle_detected = False
        for start in self._unresolved("depth", emitted_only=False):
            chase = self._chase(start, self._depth_probe)
            if chase.cycle_length:
                cycle_detected = True
                values = [*range(chase.steps, 0, -1), *([0] * chase.cycle_length)]
                self._assign("depth", start, values)
            else:
                count = chase.steps + (1 if chase.terminal else 0)
                self._assign("depth", start, (chase.value + chase.steps - step for step in range(count)))
        return cycle_detected

    def _variant_probe(self, evidence_key: str) -> tuple[str, object]:
        row = self.conn.execute(
            "SELECT n.variant_memo, n.explicit_variant, n.branch_index, p.evidence_key FROM claude_graph_node AS n "
            "LEFT JOIN claude_graph_node AS p ON p.evidence_key = n.parent WHERE n.evidence_key = ?",
            (evidence_key,),
        ).fetchone()
        if row[0] is not None:
            return "known", int(row[0])
        if row[1] is not None:
            return "terminal", int(row[1])
        if row[2]:
            return "terminal", int(row[2])
        if row[3] is None:
            return "terminal", 0
        return "next", row[3]

    def _resolve_variants(self) -> None:
        """Resolve tree-mode variant indexes, composing rank-0 inheritance.

        A record with an explicit provider variant index, or a nonzero rank
        among its siblings, keeps that value -- those are real branch points.
        A rank-0 ("primary") child continues whichever variant its parent
        belongs to, so it takes the first such value up its parent chain, or
        0 at a root, a missing parent or a cycle. Without this, two sibling
        variants at the same depth each contribute a rank-0 continuation at
        the same (position, variant_index=0) coordinate -- the collision
        shape that silently drops a message via ``INSERT OR REPLACE`` on the
        messages table's ``PRIMARY KEY(session_id, position, variant_index)``.
        """
        for start in self._unresolved("variant_memo", emitted_only=True):
            chase = self._chase(start, self._variant_probe)
            if chase.cycle_length:
                self._assign("variant_memo", start, [0] * (chase.steps + chase.cycle_length))
            else:
                self._assign("variant_memo", start, [chase.value] * (chase.steps + (1 if chase.terminal else 0)))
        self.conn.execute("UPDATE claude_graph_node SET variant_index = variant_memo WHERE has_material")

    def _deduplicate_variant_collisions(self) -> None:
        """Guarantee (position, variant_index) uniqueness per session -- the safety net.

        Parser-level variant assignment (explicit provider values, sibling
        rank, or rank-0 inheritance) is a heuristic and cannot be proven
        collision-free for every exotic input tree shape. No two emitted
        messages may share (position, variant_index): the messages table is
        unique on exactly that pair, so a silent collision here becomes a
        silently dropped message downstream whose blocks then orphan against
        a foreign key that no longer resolves. A colliding position group is
        fully renumbered in order-key order (current variant index, then
        timestamp, then evidence key); other positions are left untouched.
        """
        self.conn.execute(
            "UPDATE claude_graph_node SET variant_index = ranked.rank FROM ("
            "SELECT evidence_key, ROW_NUMBER() OVER (PARTITION BY position "
            "ORDER BY variant_index, ts_sort, evidence_key) - 1 AS rank FROM claude_graph_node "
            "WHERE has_material AND position IN (SELECT position FROM claude_graph_node WHERE has_material "
            "GROUP BY position HAVING COUNT(*) != COUNT(DISTINCT variant_index))) AS ranked "
            "WHERE claude_graph_node.evidence_key = ranked.evidence_key"
        )

    # -- the graph phase --------------------------------------------------------

    def resolve(self, explicit_active_leaf_message_provider_id: str | None) -> tuple[bool, str | None, bool, bool]:
        """Assign positions, branch and variant indexes and the active path.

        Returns whether a lineage cycle was found, the active leaf's evidence
        key, and whether active-path values come from the leaf walk or are
        all true, in that order; otherwise they are each record's own.
        """
        conn = self.conn
        flat_mode = (
            conn.execute("SELECT 1 FROM claude_graph_node WHERE parent IS NOT NULL AND parent != '' LIMIT 1").fetchone()
            is None
        )
        cycle_detected = False
        if flat_mode:
            conn.execute(
                "UPDATE claude_graph_node SET position = COALESCE(explicit_position, ranked.rank), "
                "branch_index = COALESCE(explicit_branch, 0) FROM ("
                "SELECT evidence_key, ROW_NUMBER() OVER (ORDER BY COALESCE(explicit_position, 2147483648), "
                "timestamp IS NULL, ts_sort, CASE WHEN timestamp IS NULL THEN '' ELSE evidence_key END, "
                "original_index) - 1 AS rank FROM claude_graph_node WHERE has_material) AS ranked "
                "WHERE claude_graph_node.evidence_key = ranked.evidence_key"
            )
            conn.execute(
                "UPDATE claude_graph_node SET variant_index = COALESCE(explicit_variant, branch_index) "
                "WHERE has_material"
            )
        else:
            cycle_detected = self._lineage_depths()
            conn.execute(
                "UPDATE claude_graph_node SET position = COALESCE(explicit_position, MAX(0, depth - "
                "(SELECT MIN(depth) FROM claude_graph_node WHERE has_material))) WHERE has_material"
            )
            conn.execute(
                "UPDATE claude_graph_node SET branch_index = COALESCE(explicit_branch, ranked.rank) FROM ("
                "SELECT evidence_key, ROW_NUMBER() OVER (PARTITION BY parent ORDER BY explicit_variant IS NULL, "
                "COALESCE(explicit_variant, explicit_branch, 0), ts_sort, updated_sort, evidence_key) - 1 AS rank "
                "FROM claude_graph_node) AS ranked WHERE claude_graph_node.evidence_key = ranked.evidence_key"
            )
            self._resolve_variants()
        self._deduplicate_variant_collisions()

        leaf_id = explicit_active_leaf_message_provider_id
        if (
            leaf_id is not None
            and conn.execute("SELECT 1 FROM claude_graph_node WHERE evidence_key = ?", (leaf_id,)).fetchone() is None
        ):
            leaf_id = None
        if leaf_id is None:
            explicit_leaves = conn.execute(
                "SELECT evidence_key FROM claude_graph_node WHERE explicit_active_leaf = 1 LIMIT 2"
            ).fetchall()
            if len(explicit_leaves) == 1:
                leaf_id = str(explicit_leaves[0][0])
        if (
            leaf_id is None
            and conn.execute(
                "SELECT 1 FROM claude_graph_node WHERE has_material AND explicit_active_path = 1 LIMIT 1"
            ).fetchone()
            is not None
        ):
            candidates = conn.execute(
                "SELECT evidence_key FROM claude_graph_node AS a WHERE has_material AND explicit_active_path = 1 "
                "AND NOT EXISTS (SELECT 1 FROM claude_graph_node AS c WHERE c.has_material "
                "AND c.explicit_active_path = 1 AND c.parent = a.evidence_key) ORDER BY evidence_key LIMIT 2"
            ).fetchall()
            if len(candidates) == 1:
                leaf_id = str(candidates[0][0])
        if leaf_id is None and flat_mode:
            last = conn.execute(
                "SELECT evidence_key FROM claude_graph_node WHERE has_material "
                "ORDER BY position DESC, variant_index DESC, evidence_key DESC LIMIT 1"
            ).fetchone()
            if last is not None:
                leaf_id = str(last[0])
        if flat_mode:
            all_active = (
                conn.execute(
                    "SELECT 1 FROM claude_graph_node WHERE has_material AND explicit_active_path IS NOT NULL LIMIT 1"
                ).fetchone()
                is None
            )
            return cycle_detected, leaf_id, False, all_active
        if leaf_id is None:
            terminals = conn.execute(
                "SELECT evidence_key FROM claude_graph_node AS e WHERE has_material AND NOT EXISTS ("
                "SELECT 1 FROM claude_graph_node AS c WHERE c.has_material AND c.parent = e.evidence_key) LIMIT 2"
            ).fetchall()
            if len(terminals) == 1:
                leaf_id = str(terminals[0][0])
        if leaf_id is None:
            return cycle_detected, None, False, False
        cursor: str | None = leaf_id
        while cursor is not None:
            row = conn.execute(
                "SELECT on_path, parent FROM claude_graph_node WHERE evidence_key = ?", (cursor,)
            ).fetchone()
            if row is None or row[0]:
                break
            conn.execute("UPDATE claude_graph_node SET on_path = 1 WHERE evidence_key = ?", (cursor,))
            cursor = row[1]
        return cycle_detected, leaf_id, True, False

    def emitted(self, *, first_seen_with_attachments: bool = False) -> Iterator[GraphRow]:
        """Records with material in canonical order, or those carrying attachments in first-seen order."""
        selection = (
            "AND has_attachments ORDER BY seq"
            if first_seen_with_attachments
            else "ORDER BY position, variant_index, evidence_key"
        )
        for row in self.conn.execute(
            "SELECT evidence_key, original_index, position, branch_index, variant_index, explicit_active_path, "
            f"explicit_active_leaf, on_path FROM claude_graph_node WHERE has_material {selection}"
        ):
            yield GraphRow(
                evidence_key=str(row[0]),
                original_index=int(row[1]),
                position=int(row[2]),
                branch_index=int(row[3] or 0),
                variant_index=int(row[4]),
                explicit_active_path=_flag(row[5]),
                explicit_active_leaf=_flag(row[6]),
                on_path=bool(row[7]),
            )

    def unemitted_summaries(self) -> Iterator[tuple[str, int]]:
        for row in self.conn.execute(
            "SELECT evidence_key, original_index FROM claude_graph_node "
            "WHERE NOT has_material AND has_compaction_summary ORDER BY seq"
        ):
            yield str(row[0]), int(row[1])

    def native_id(self, evidence_key: str) -> str:
        return str(
            self.conn.execute(
                "SELECT native_id FROM claude_graph_node WHERE evidence_key = ?", (evidence_key,)
            ).fetchone()[0]
        )

    def is_emitted(self, evidence_key: str) -> bool:
        row = self.conn.execute(
            "SELECT has_material FROM claude_graph_node WHERE evidence_key = ?", (evidence_key,)
        ).fetchone()
        return row is not None and bool(row[0])

    def emitted_coordinate(self, evidence_key: str) -> tuple[int, int] | None:
        """Return the exact emitted occurrence, excluding summary-only records."""
        row = self.conn.execute(
            "SELECT position, variant_index FROM claude_graph_node WHERE evidence_key = ? AND has_material",
            (evidence_key,),
        ).fetchone()
        return (int(row[0]), int(row[1])) if row is not None else None

    def duplicate_native_ids(self) -> list[str]:
        return [
            str(row[0])
            for row in self.conn.execute("SELECT native_id FROM claude_graph_native WHERE n > 1 ORDER BY native_id")
        ]

    def add_summary(
        self, base: str, evidence_key: str, original_index: int, position: int, payload_json: str, timestamp: str | None
    ) -> None:
        self.conn.execute(
            "INSERT INTO claude_graph_summary (base, position, payload_json, timestamp, original_index, evidence_key) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (base, position, payload_json, timestamp, original_index, evidence_key),
        )

    def ordered_summaries(self) -> Iterator[tuple[str, int]]:
        """Summaries grouped at their identity's first position, then by content."""
        for row in self.conn.execute(
            "SELECT evidence_key, original_index FROM (SELECT *, MIN(position) OVER (PARTITION BY base) AS first "
            "FROM claude_graph_summary) ORDER BY first, base, payload_json, COALESCE(timestamp, ''), seq"
        ):
            yield str(row[0]), int(row[1])
