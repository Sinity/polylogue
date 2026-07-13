"""Durable user-tier persistence for content-addressed query objects."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

from polylogue.core.hashing import hash_payload
from polylogue.core.query_identity import JsonValue, canonical_query_plan, query_hash_for_plan, query_ref

QueryEdgeKind = Literal["operand-of", "refines", "supersedes", "derived-from", "same-as"]
ResultSetExactness = Literal["exact", "capped", "sampled", "estimate"]
ResultSetPersistence = Literal["routine", "watch", "pinned", "finding", "cohort"]

_DURABLE_MEMBER_PERSISTENCE = frozenset({"watch", "pinned", "finding", "cohort"})
_ACYCLIC_EDGE_KINDS = frozenset({"supersedes", "derived-from"})


@dataclass(frozen=True, slots=True)
class QueryObject:
    query_hash: str
    canonical_plan: dict[str, JsonValue]
    grain: str
    lane: str
    rank_policy: str

    @property
    def ref(self) -> str:
        return query_ref(self.query_hash).format()


@dataclass(frozen=True, slots=True)
class ResultSetManifest:
    result_set_id: str
    query_hash: str
    grain: str
    corpus_epoch: str
    member_count: int
    membership_merkle_root: str
    ordered_rank_hash: str
    exactness: ResultSetExactness
    persistence_class: ResultSetPersistence


def put_query(
    conn: sqlite3.Connection,
    planned_ast: dict[str, JsonValue],
    *,
    grain: str,
    lane: str,
    rank_policy: str,
    field_aliases: dict[str, str] | None = None,
    created_at_ms: int,
) -> QueryObject:
    """Idempotently persist one canonical expanded query plan."""
    canonical_plan = canonical_query_plan(
        planned_ast, grain=grain, lane=lane, rank_policy=rank_policy, field_aliases=field_aliases
    )
    query_hash = query_hash_for_plan(
        planned_ast, grain=grain, lane=lane, rank_policy=rank_policy, field_aliases=field_aliases
    )
    conn.execute(
        """
        INSERT INTO queries (query_hash, canonical_plan_json, grain, lane, rank_policy, created_at_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(query_hash) DO NOTHING
        """,
        (query_hash, _json(canonical_plan), grain, lane, rank_policy, created_at_ms),
    )
    return QueryObject(query_hash, canonical_plan, grain, lane, rank_policy)


def put_query_name(
    conn: sqlite3.Connection,
    *,
    name: str,
    query_hash: str,
    updated_at_ms: int,
    supersedes_query_hash: str | None = None,
) -> None:
    """Move a mutable human query name to an immutable query hash."""
    conn.execute(
        """
        INSERT INTO query_names (name, query_hash, supersedes_query_hash, updated_at_ms)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            query_hash = excluded.query_hash,
            supersedes_query_hash = excluded.supersedes_query_hash,
            updated_at_ms = excluded.updated_at_ms
        """,
        (name, query_hash, supersedes_query_hash, updated_at_ms),
    )


def put_result_set(
    conn: sqlite3.Connection,
    *,
    result_set_id: str,
    query_hash: str,
    grain: str,
    corpus_epoch: str,
    member_refs: tuple[str, ...],
    exactness: ResultSetExactness,
    persistence_class: ResultSetPersistence,
    created_at_ms: int,
) -> ResultSetManifest:
    """Persist a promoted result manifest and, where permitted, exact members."""
    if persistence_class not in _DURABLE_MEMBER_PERSISTENCE and member_refs:
        raise ValueError("routine result sets cannot persist exact member refs")
    if len(member_refs) != len(set(member_refs)):
        raise ValueError("result set member refs must be unique at one grain")
    manifest = ResultSetManifest(
        result_set_id=result_set_id,
        query_hash=query_hash,
        grain=grain,
        corpus_epoch=corpus_epoch,
        member_count=len(member_refs),
        membership_merkle_root=_membership_merkle_root(member_refs),
        ordered_rank_hash=hash_payload(list(member_refs)),
        exactness=exactness,
        persistence_class=persistence_class,
    )
    conn.execute(
        """
        INSERT INTO result_sets (
            result_set_id, query_hash, grain, corpus_epoch, member_count,
            membership_merkle_root, ordered_rank_hash, exactness, persistence_class, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            manifest.result_set_id,
            manifest.query_hash,
            manifest.grain,
            manifest.corpus_epoch,
            manifest.member_count,
            manifest.membership_merkle_root,
            manifest.ordered_rank_hash,
            manifest.exactness,
            manifest.persistence_class,
            created_at_ms,
        ),
    )
    if persistence_class in _DURABLE_MEMBER_PERSISTENCE:
        conn.executemany(
            "INSERT INTO result_set_members (result_set_id, rank, member_ref) VALUES (?, ?, ?)",
            ((result_set_id, rank, member_ref) for rank, member_ref in enumerate(member_refs)),
        )
    return manifest


def put_query_edge(
    conn: sqlite3.Connection,
    *,
    src_query_hash: str,
    dst_query_hash: str,
    edge_kind: QueryEdgeKind,
    created_at_ms: int,
) -> None:
    """Persist one planner-emitted query edge, rejecting semantic DAG cycles."""
    if src_query_hash == dst_query_hash and edge_kind in _ACYCLIC_EDGE_KINDS:
        raise ValueError(f"{edge_kind} query edge cannot self-reference")
    if edge_kind in _ACYCLIC_EDGE_KINDS:
        row = conn.execute(
            """
            WITH RECURSIVE reachable(query_hash) AS (
                SELECT dst_query_hash FROM query_edges WHERE src_query_hash = ? AND edge_kind = ?
                UNION
                SELECT edge.dst_query_hash
                FROM query_edges AS edge JOIN reachable ON edge.src_query_hash = reachable.query_hash
                WHERE edge.edge_kind = ?
            )
            SELECT 1 FROM reachable WHERE query_hash = ? LIMIT 1
            """,
            (dst_query_hash, edge_kind, edge_kind, src_query_hash),
        ).fetchone()
        if row is not None:
            raise ValueError(f"{edge_kind} query edge would create a cycle")
    conn.execute(
        """
        INSERT INTO query_edges (src_query_hash, dst_query_hash, edge_kind, created_at_ms)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(src_query_hash, dst_query_hash, edge_kind) DO NOTHING
        """,
        (src_query_hash, dst_query_hash, edge_kind, created_at_ms),
    )


def migrate_saved_query_assertions(conn: sqlite3.Connection) -> int:
    """Repoint legacy saved-query assertions at immutable ``query:<hash>`` refs.

    Legacy saved view payloads are already parsed JSON request specifications.
    They have no macro reference, so their dynamic request shape is the typed
    plan supplied to the shared canonicalization boundary.
    """
    rows = tuple(
        conn.execute("SELECT assertion_id, value_json, created_at_ms FROM assertions WHERE kind = 'saved_query'")
    )
    for assertion_id, value_json, created_at_ms in rows:
        try:
            value = json.loads(str(value_json or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"saved query assertion {assertion_id} has invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"saved query assertion {assertion_id} must contain an object query")
        query = put_query(
            conn,
            value,
            grain="session",
            lane="dialogue",
            rank_policy="mixed-bm25-rrf-vector",
            created_at_ms=int(created_at_ms),
        )
        conn.execute("UPDATE assertions SET target_ref = ? WHERE assertion_id = ?", (query.ref, assertion_id))
    return len(rows)


def _membership_merkle_root(member_refs: tuple[str, ...]) -> str:
    if not member_refs:
        return hash_payload([])
    level = sorted(hash_payload(member_ref) for member_ref in member_refs)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hash_payload([left, right]) for left, right in zip(level[::2], level[1::2], strict=True)]
    return level[0]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "QueryObject",
    "ResultSetManifest",
    "migrate_saved_query_assertions",
    "put_query",
    "put_query_edge",
    "put_query_name",
    "put_result_set",
]
