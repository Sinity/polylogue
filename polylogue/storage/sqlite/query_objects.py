"""Durable user-tier persistence for content-addressed query objects."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

from polylogue.core.hashing import hash_payload
from polylogue.core.query_identity import (
    QUERY_DEFINITION_PROTOCOL_VERSION,
    JsonValue,
    canonical_query_plan,
    query_hash_for_plan,
    query_ref,
    require_supported_definition_protocol_version,
)

QueryEdgeKind = Literal["operand-of", "refines", "supersedes", "derived-from", "same-as"]
ResultSetExactness = Literal["exact", "capped", "sampled", "estimate"]
ResultSetPersistence = Literal["routine", "watch", "pinned", "finding", "cohort"]

_DURABLE_MEMBER_PERSISTENCE = frozenset({"watch", "pinned", "finding", "cohort"})
_ACYCLIC_EDGE_KINDS = frozenset({"operand-of", "supersedes", "derived-from"})


@dataclass(frozen=True, slots=True)
class QueryObject:
    query_hash: str
    canonical_plan: dict[str, JsonValue]
    grain: str
    lane: str
    rank_policy: str
    definition_protocol_version: str

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


@dataclass(frozen=True, slots=True)
class RetainedQueryRun:
    run_id: str
    query_hash: str
    result_set_id: str


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    receipt_id: str
    source_generation: str
    user_generation: str
    index_generation: str
    runtime_build_ref: str
    model_refs: tuple[str, ...] = ()
    resolved_bounds: dict[str, JsonValue] | None = None
    degradation: dict[str, JsonValue] | None = None


def put_query(
    conn: sqlite3.Connection,
    planned_ast: dict[str, JsonValue],
    *,
    grain: str,
    lane: str,
    rank_policy: str,
    field_aliases: dict[str, str] | None = None,
    definition_protocol_version: str = QUERY_DEFINITION_PROTOCOL_VERSION,
    created_at_ms: int,
) -> QueryObject:
    """Idempotently persist one canonical expanded query plan."""
    definition_protocol_version = require_supported_definition_protocol_version(definition_protocol_version)
    canonical_plan = canonical_query_plan(
        planned_ast,
        grain=grain,
        lane=lane,
        rank_policy=rank_policy,
        field_aliases=field_aliases,
        definition_protocol_version=definition_protocol_version,
    )
    query_hash = query_hash_for_plan(
        planned_ast,
        grain=grain,
        lane=lane,
        rank_policy=rank_policy,
        field_aliases=field_aliases,
        definition_protocol_version=definition_protocol_version,
    )
    conn.execute(
        """
        INSERT INTO queries (
            query_hash, canonical_plan_json, grain, lane, rank_policy,
            definition_protocol_version, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(query_hash) DO NOTHING
        """,
        (query_hash, _json(canonical_plan), grain, lane, rank_policy, definition_protocol_version, created_at_ms),
    )
    return QueryObject(query_hash, canonical_plan, grain, lane, rank_policy, definition_protocol_version)


def put_query_name(
    conn: sqlite3.Connection,
    *,
    name: str,
    query_hash: str,
    updated_at_ms: int,
    supersedes_query_hash: str | None = None,
    watch: bool = False,
) -> None:
    """Move a mutable human query name to an immutable query hash."""
    conn.execute(
        """
        INSERT INTO query_names (name, query_hash, supersedes_query_hash, watch, updated_at_ms)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            query_hash = excluded.query_hash,
            supersedes_query_hash = excluded.supersedes_query_hash,
            watch = excluded.watch,
            updated_at_ms = excluded.updated_at_ms
        """,
        (name, query_hash, supersedes_query_hash, int(watch), updated_at_ms),
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
        membership_merkle_root=membership_merkle_root(member_refs),
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


def get_query(conn: sqlite3.Connection, query_hash: str) -> QueryObject | None:
    """Read one immutable query definition without decoding it as executable syntax."""
    row = conn.execute(
        """
        SELECT query_hash, canonical_plan_json, grain, lane, rank_policy, definition_protocol_version
        FROM queries WHERE query_hash = ?
        """,
        (query_hash,),
    ).fetchone()
    if row is None:
        return None
    payload = json.loads(str(row[1]))
    if not isinstance(payload, dict):
        raise ValueError(f"query {query_hash} has invalid canonical plan")
    return QueryObject(
        query_hash=str(row[0]),
        canonical_plan=payload,
        grain=str(row[2]),
        lane=str(row[3]),
        rank_policy=str(row[4]),
        definition_protocol_version=str(row[5]),
    )


def list_watched_queries(conn: sqlite3.Connection) -> tuple[QueryObject, ...]:
    """Return distinct immutable query versions selected by watched names."""
    rows = conn.execute(
        """
        SELECT DISTINCT q.query_hash, q.canonical_plan_json, q.grain, q.lane,
               q.rank_policy, q.definition_protocol_version
        FROM query_names AS n
        JOIN queries AS q ON q.query_hash = n.query_hash
        WHERE n.watch = 1
        ORDER BY q.query_hash
        """
    ).fetchall()
    return tuple(
        QueryObject(
            query_hash=str(row[0]),
            canonical_plan=_query_payload(row[1]),
            grain=str(row[2]),
            lane=str(row[3]),
            rank_policy=str(row[4]),
            definition_protocol_version=str(row[5]),
        )
        for row in rows
    )


def get_result_set(conn: sqlite3.Connection, result_set_id: str) -> ResultSetManifest | None:
    row = conn.execute(
        """
        SELECT result_set_id, query_hash, grain, corpus_epoch, member_count,
               membership_merkle_root, ordered_rank_hash, exactness, persistence_class
        FROM result_sets WHERE result_set_id = ?
        """,
        (result_set_id,),
    ).fetchone()
    return _manifest_from_row(row) if row is not None else None


def get_result_set_members(conn: sqlite3.Connection, result_set_id: str) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT member_ref FROM result_set_members WHERE result_set_id = ? ORDER BY rank",
        (result_set_id,),
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def get_latest_result_set(
    conn: sqlite3.Connection,
    *,
    query_hash: str,
    persistence_class: ResultSetPersistence,
) -> ResultSetManifest | None:
    row = conn.execute(
        """
        SELECT result_set_id, query_hash, grain, corpus_epoch, member_count,
               membership_merkle_root, ordered_rank_hash, exactness, persistence_class
        FROM result_sets
        WHERE query_hash = ? AND persistence_class = ?
        ORDER BY created_at_ms DESC, result_set_id DESC
        LIMIT 1
        """,
        (query_hash, persistence_class),
    ).fetchone()
    return _manifest_from_row(row) if row is not None else None


def get_watched_query_baseline(conn: sqlite3.Connection, query_hash: str) -> ResultSetManifest | None:
    """Return the last evaluated durable watch relation for one query."""
    row = conn.execute(
        """
        SELECT rs.result_set_id, rs.query_hash, rs.grain, rs.corpus_epoch, rs.member_count,
               rs.membership_merkle_root, rs.ordered_rank_hash, rs.exactness, rs.persistence_class
        FROM watched_query_baselines AS baseline
        JOIN result_sets AS rs ON rs.result_set_id = baseline.result_set_id
        WHERE baseline.query_hash = ?
        """,
        (query_hash,),
    ).fetchone()
    return _manifest_from_row(row) if row is not None else None


def put_watched_query_baseline(
    conn: sqlite3.Connection,
    *,
    query_hash: str,
    result_set_id: str,
    updated_at_ms: int,
) -> None:
    """Advance one query's baseline pointer, including A→B→A transitions."""
    result_set = get_result_set(conn, result_set_id)
    if result_set is None:
        raise KeyError(f"result-set:{result_set_id}")
    if result_set.query_hash != query_hash or result_set.persistence_class != "watch":
        raise ValueError("watched query baseline must reference that query's watch result set")
    conn.execute(
        """
        INSERT INTO watched_query_baselines (query_hash, result_set_id, updated_at_ms)
        VALUES (?, ?, ?)
        ON CONFLICT(query_hash) DO UPDATE SET
            result_set_id = excluded.result_set_id,
            updated_at_ms = excluded.updated_at_ms
        """,
        (query_hash, result_set_id, updated_at_ms),
    )


def put_retained_query_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    query_hash: str,
    result_set_id: str,
    retained_at_ms: int,
) -> RetainedQueryRun:
    result_set = get_result_set(conn, result_set_id)
    if result_set is None:
        raise KeyError(f"result-set:{result_set_id}")
    if result_set.query_hash != query_hash:
        raise ValueError("retained query run result set must belong to the same query")
    existing = conn.execute(
        "SELECT query_hash, result_set_id, retained_at_ms FROM retained_query_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if existing is not None:
        if tuple(existing) == (query_hash, result_set_id, retained_at_ms):
            return RetainedQueryRun(run_id, query_hash, result_set_id)
        raise ValueError(f"retained query run id conflicts with a different execution: {run_id}")
    conn.execute(
        """
        INSERT INTO retained_query_runs (run_id, query_hash, result_set_id, retained_at_ms)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, query_hash, result_set_id, retained_at_ms),
    )
    return RetainedQueryRun(run_id, query_hash, result_set_id)


def get_retained_query_run(conn: sqlite3.Connection, run_id: str) -> RetainedQueryRun | None:
    row = conn.execute(
        "SELECT run_id, query_hash, result_set_id FROM retained_query_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return RetainedQueryRun(str(row[0]), str(row[1]), str(row[2])) if row is not None else None


def put_evaluation_receipt(
    conn: sqlite3.Connection,
    *,
    query_hash: str,
    receipt: EvaluationReceipt,
    result_set_id: str | None,
    created_at_ms: int,
) -> None:
    """Persist immutable execution context for a materialized relation."""
    if result_set_id is not None:
        result_set = get_result_set(conn, result_set_id)
        if result_set is None:
            raise KeyError(f"result-set:{result_set_id}")
        if result_set.query_hash != query_hash:
            raise ValueError("evaluation receipt result set must belong to the same query")
    values = (
        query_hash,
        result_set_id,
        receipt.source_generation,
        receipt.user_generation,
        receipt.index_generation,
        receipt.runtime_build_ref,
        _json(list(receipt.model_refs)),
        _json(receipt.resolved_bounds or {}),
        _json(receipt.degradation or {}),
        created_at_ms,
    )
    existing = conn.execute(
        """
        SELECT query_hash, result_set_id, source_generation, user_generation, index_generation,
               runtime_build_ref, model_refs_json, resolved_bounds_json, degradation_json, created_at_ms
        FROM query_evaluation_receipts WHERE receipt_id = ?
        """,
        (receipt.receipt_id,),
    ).fetchone()
    if existing is not None:
        if tuple(existing) != values:
            raise ValueError(f"evaluation receipt id conflicts with a different execution: {receipt.receipt_id}")
        return
    conn.execute(
        """
        INSERT INTO query_evaluation_receipts (
            receipt_id, query_hash, result_set_id, source_generation, user_generation,
            index_generation, runtime_build_ref, model_refs_json, resolved_bounds_json,
            degradation_json, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (receipt.receipt_id, *values),
    )


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


def membership_merkle_root(member_refs: tuple[str, ...]) -> str:
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


def _query_payload(value: object) -> dict[str, JsonValue]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("query canonical plan must be an object")
    return parsed


def _manifest_from_row(row: sqlite3.Row | tuple[object, ...]) -> ResultSetManifest:
    member_count = row[4]
    if isinstance(member_count, bool) or not isinstance(member_count, int):
        raise ValueError("result set member_count must be an integer")
    return ResultSetManifest(
        result_set_id=str(row[0]),
        query_hash=str(row[1]),
        grain=str(row[2]),
        corpus_epoch=str(row[3]),
        member_count=member_count,
        membership_merkle_root=str(row[5]),
        ordered_rank_hash=str(row[6]),
        exactness=str(row[7]),  # type: ignore[arg-type]
        persistence_class=str(row[8]),  # type: ignore[arg-type]
    )


__all__ = [
    "EvaluationReceipt",
    "QueryObject",
    "RetainedQueryRun",
    "ResultSetManifest",
    "get_latest_result_set",
    "get_query",
    "get_result_set",
    "get_result_set_members",
    "get_retained_query_run",
    "get_watched_query_baseline",
    "list_watched_queries",
    "membership_merkle_root",
    "migrate_saved_query_assertions",
    "put_evaluation_receipt",
    "put_query",
    "put_query_edge",
    "put_query_name",
    "put_retained_query_run",
    "put_result_set",
    "put_watched_query_baseline",
]
