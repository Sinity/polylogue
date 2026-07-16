"""Durable immutable plans and conservation receipts for raw reconciliation.

The source tier is the authority for this ledger.  ``index.db`` may be rebuilt
and ``ops.db`` may be deleted; neither event is allowed to erase replay
fairness, stale-plan blockers, or the proof that a complete census was
accounted for.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger

RAW_AUTHORITY_PARSER_FINGERPRINT = "revision-membership-v1"
RAW_AUTHORITY_CENSUS_QUERY_PREFIX = "polylogue://raw-authority-census/"
RAW_AUTHORITY_DETAIL_QUERY_PREFIX = "polylogue://raw-authority-detail/"
RAW_AUTHORITY_DETAIL_CHUNK_CHARS = 16_384
logger = get_logger(__name__)


class RawReplayPlanStatus(StrEnum):
    EXECUTED = "executed"
    RETRYABLE = "retryable"
    DEFERRED = "deferred"
    TERMINAL = "terminal"
    REJECTED_STALE = "rejected_stale"
    CARRIED_FORWARD = "carried_forward"


@dataclass(frozen=True, slots=True)
class RawReplayPlan:
    plan_id: str
    input_digest: str
    input_raw_ids: tuple[str, ...]
    logical_keys: tuple[str, ...]
    authority_witness: JSONDocument
    source_preconditions: JSONDocument
    index_preconditions: JSONDocument

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "plan_id": self.plan_id,
                "input_digest": self.input_digest,
                "input_raw_ids": list(self.input_raw_ids),
                "logical_keys": list(self.logical_keys),
                "authority_witness": self.authority_witness,
                "source_preconditions": self.source_preconditions,
                "index_preconditions": self.index_preconditions,
            }
        )


@dataclass(frozen=True, slots=True)
class RawReplayPlanOutcome:
    plan_id: str
    input_raw_ids: tuple[str, ...]
    status: RawReplayPlanStatus
    reason: str
    next_action: str
    application_receipt: JSONDocument | None = None

    def to_dict(self) -> JSONDocument:
        payload: dict[str, object] = {
            "plan_id": self.plan_id,
            "input_raw_ids": list(self.input_raw_ids),
            "status": self.status.value,
            "reason": self.reason,
            "next_action": self.next_action,
        }
        if self.application_receipt is not None:
            payload["application_receipt"] = self.application_receipt
        return json_document(payload)

    def to_summary_dict(self) -> JSONDocument:
        """Return a bounded projection; the census handle owns full receipts."""
        raw_id_sample_limit = 8
        return json_document(
            {
                "plan_id": self.plan_id,
                "input_raw_count": len(self.input_raw_ids),
                "input_raw_id_sample": list(self.input_raw_ids[:raw_id_sample_limit]),
                "input_raw_id_sample_truncated": len(self.input_raw_ids) > raw_id_sample_limit,
                "status": self.status.value,
                "reason": self.reason,
                "next_action": self.next_action,
                "has_application_receipt": self.application_receipt is not None,
            }
        )


@dataclass(frozen=True, slots=True)
class RawAuthorityCensusReceipt:
    census_id: str
    sequence_no: int
    inventory_digest: str
    residual_digest: str
    plan_count: int
    executable_plan_count: int
    residual_plan_count: int
    post_inventory_digest: str | None
    post_residual_digest: str | None
    post_plan_count: int | None
    predecessor_census_id: str | None
    mode: str
    lifecycle_status: str
    quiescent: bool
    fixed_point: bool

    @property
    def query_handle(self) -> str:
        return raw_authority_census_query_handle(self.census_id)


def raw_authority_census_query_handle(census_id: str, *, offset: int = 0) -> str:
    """Return a directly resolvable, paginated census-ledger URI."""
    if not census_id or "/" in census_id:
        raise ValueError("raw authority census id must be non-empty and contain no slash")
    if offset < 0:
        raise ValueError("raw authority census offset must be non-negative")
    return f"{RAW_AUTHORITY_CENSUS_QUERY_PREFIX}{census_id}/{offset}"


def raw_authority_detail_query_handle(
    census_id: str,
    record_id: str,
    *,
    revision: str = "current",
    offset: int = 0,
) -> str:
    """Return a bounded-chunk URI for one complete census or plan document."""
    if not census_id or "/" in census_id or not record_id or "/" in record_id or not revision or "/" in revision:
        raise ValueError("raw authority detail identifiers must be non-empty and contain no slash")
    if offset < 0:
        raise ValueError("raw authority detail offset must be non-negative")
    return f"{RAW_AUTHORITY_DETAIL_QUERY_PREFIX}{census_id}/{record_id}/{revision}/{offset}"


def _raw_authority_census_ref(value: str, *, offset: int | None) -> tuple[str, int]:
    embedded_offset = 0
    if value.startswith(RAW_AUTHORITY_CENSUS_QUERY_PREFIX):
        suffix = value.removeprefix(RAW_AUTHORITY_CENSUS_QUERY_PREFIX)
        try:
            census_id, encoded_offset = suffix.rsplit("/", 1)
            embedded_offset = int(encoded_offset)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid raw authority census query handle") from exc
    elif value.startswith("raw-authority-census:"):
        # Read compatibility for the short-lived pre-URI receipt shape.
        census_id = value.removeprefix("raw-authority-census:")
    else:
        census_id = value
    resolved_offset = embedded_offset if offset is None else offset
    if not census_id or "/" in census_id or resolved_offset < 0:
        raise ValueError("invalid raw authority census query handle")
    return census_id, resolved_offset


def _decode_json_field(value: object) -> object:
    if not isinstance(value, str):
        raise RuntimeError("raw authority ledger contains a non-text JSON field")
    return json.loads(value)


def _raw_authority_detail_ref(value: str, *, offset: int | None) -> tuple[str, str, str, int]:
    if not value.startswith(RAW_AUTHORITY_DETAIL_QUERY_PREFIX):
        raise ValueError("invalid raw authority detail query handle")
    suffix = value.removeprefix(RAW_AUTHORITY_DETAIL_QUERY_PREFIX)
    try:
        identifiers, encoded_offset = suffix.rsplit("/", 1)
        census_id, record_id, revision = identifiers.split("/", 2)
        embedded_offset = int(encoded_offset)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid raw authority detail query handle") from exc
    resolved_offset = embedded_offset if offset is None else offset
    if (
        not census_id
        or "/" in census_id
        or not record_id
        or "/" in record_id
        or not revision
        or "/" in revision
        or resolved_offset < 0
    ):
        raise ValueError("invalid raw authority detail query handle")
    if revision == "current" and resolved_offset != 0:
        raise ValueError("unbound raw authority detail handles may only start at offset zero")
    return census_id, record_id, revision, resolved_offset


def _raw_authority_detail_document(conn: sqlite3.Connection, census_id: str, record_id: str) -> JSONDocument:
    census = conn.execute(
        """
        SELECT census_id, sequence_no, scope_json, residual_json, parser_fingerprint,
               mode, lifecycle_status, quiescent, inventory_digest, residual_digest,
               plan_count, post_inventory_digest, post_residual_json,
               post_residual_digest, post_plan_count, postflight_at_ms,
               executable_plan_count, residual_plan_count, predecessor_census_id,
               fixed_point, created_at_ms, completed_at_ms
        FROM raw_authority_censuses WHERE census_id = ?
        """,
        (census_id,),
    ).fetchone()
    if census is None:
        raise KeyError(census_id)
    if record_id == "census":
        return json_document(
            {
                "record_type": "census",
                "census_id": census_id,
                "scope": _decode_json_field(census["scope_json"]),
                "residual": _decode_json_field(census["residual_json"]),
                "post_residual": (
                    _decode_json_field(census["post_residual_json"])
                    if census["post_residual_json"] is not None
                    else None
                ),
            }
        )
    row = conn.execute(
        """
        SELECT p.plan_id, p.input_digest, p.input_raw_ids_json,
               p.logical_keys_json, p.authority_witness_json,
               p.source_preconditions_json, p.index_preconditions_json,
               p.created_at_ms, cp.ordinal, cp.selected, cp.outcome_status,
               cp.reason, cp.next_action, cp.application_receipt_json,
               cp.outcome_recorded, cp.recorded_at_ms,
               EXISTS(
                   SELECT 1 FROM raw_authority_census_post_plans AS cpp
                   WHERE cpp.census_id = ? AND cpp.plan_id = p.plan_id
               ) AS present_postflight
        FROM raw_authority_plans AS p
        LEFT JOIN raw_authority_census_plans AS cp
          ON cp.census_id = ? AND cp.plan_id = p.plan_id
        WHERE p.plan_id = ?
          AND (
              cp.plan_id IS NOT NULL
              OR EXISTS(
                  SELECT 1 FROM raw_authority_census_post_plans AS cpp
                  WHERE cpp.census_id = ? AND cpp.plan_id = p.plan_id
              )
          )
        """,
        (census_id, census_id, record_id, census_id),
    ).fetchone()
    if row is None:
        raise KeyError(f"{census_id}/{record_id}")
    blockers = [
        {
            "blocker_id": str(blocker["blocker_id"]),
            "reason": str(blocker["reason"]),
            "expected": _decode_json_field(blocker["expected_json"]),
            "observed": _decode_json_field(blocker["observed_json"]),
            "created_at_ms": int(blocker["created_at_ms"]),
            "resolved_at_ms": blocker["resolved_at_ms"],
            "resolution": (_decode_json_field(blocker["resolution"]) if blocker["resolution"] is not None else None),
        }
        for blocker in conn.execute(
            """
            SELECT blocker_id, reason, expected_json, observed_json,
                   created_at_ms, resolved_at_ms, resolution
            FROM raw_authority_blockers
            WHERE census_id = ? AND plan_id = ?
            ORDER BY created_at_ms, blocker_id
            """,
            (census_id, record_id),
        )
    ]
    return json_document(
        {
            "record_type": "plan",
            "census_id": census_id,
            "ordinal": row["ordinal"],
            "selected": bool(row["selected"]) if row["selected"] is not None else None,
            "outcome_status": row["outcome_status"],
            "reason": row["reason"],
            "next_action": row["next_action"],
            "application_receipt": (
                _decode_json_field(row["application_receipt_json"])
                if row["application_receipt_json"] is not None
                else None
            ),
            "outcome_recorded": (bool(row["outcome_recorded"]) if row["outcome_recorded"] is not None else None),
            "recorded_at_ms": row["recorded_at_ms"],
            "present_postflight": bool(row["present_postflight"]),
            "plan": {
                "plan_id": str(row["plan_id"]),
                "input_digest": str(row["input_digest"]),
                "input_raw_ids": _decode_json_field(row["input_raw_ids_json"]),
                "logical_keys": _decode_json_field(row["logical_keys_json"]),
                "authority_witness": _decode_json_field(row["authority_witness_json"]),
                "source_preconditions": _decode_json_field(row["source_preconditions_json"]),
                "index_preconditions": _decode_json_field(row["index_preconditions_json"]),
                "created_at_ms": int(row["created_at_ms"]),
            },
            "blockers": blockers,
        }
    )


def read_raw_authority_detail(
    archive_root: Path,
    query_handle: str,
    *,
    chunk_chars: int = RAW_AUTHORITY_DETAIL_CHUNK_CHARS,
    offset: int | None = None,
) -> JSONDocument:
    """Read one bounded text chunk of a complete census or plan document."""
    if not 256 <= chunk_chars <= 65_536:
        raise ValueError("raw authority detail chunk_chars must be between 256 and 65536")
    census_id, record_id, requested_revision, resolved_offset = _raw_authority_detail_ref(query_handle, offset=offset)
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        raise FileNotFoundError(source_db)
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        document = _raw_authority_detail_document(conn, census_id, record_id)
    encoded = _canonical_json(document)
    document_sha256 = hashlib.sha256(encoded.encode()).hexdigest()
    if requested_revision != "current" and requested_revision != document_sha256:
        raise RuntimeError("raw authority detail changed; restart from its current offset-zero handle")
    if resolved_offset > len(encoded):
        raise ValueError("raw authority detail offset exceeds document length")
    chunk = encoded[resolved_offset : resolved_offset + chunk_chars]
    next_offset = resolved_offset + len(chunk)
    return json_document(
        {
            "query_handle": raw_authority_detail_query_handle(
                census_id, record_id, revision=document_sha256, offset=resolved_offset
            ),
            "next_query_handle": (
                raw_authority_detail_query_handle(census_id, record_id, revision=document_sha256, offset=next_offset)
                if next_offset < len(encoded)
                else None
            ),
            "encoding": "canonical-json-text-v1",
            "document_sha256": document_sha256,
            "document_char_count": len(encoded),
            "document_byte_count": len(encoded.encode()),
            "offset": resolved_offset,
            "chunk_chars": chunk_chars,
            "chunk": chunk,
        }
    )


def read_raw_authority_census(
    archive_root: Path,
    query_handle: str,
    *,
    limit: int = 100,
    offset: int | None = None,
) -> JSONDocument:
    """Read one bounded page from an immutable source-tier census ledger."""
    if not 1 <= limit <= 500:
        raise ValueError("raw authority census limit must be between 1 and 500")
    census_id, resolved_offset = _raw_authority_census_ref(query_handle, offset=offset)
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        raise FileNotFoundError(source_db)
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        census = conn.execute(
            """
            SELECT census_id, sequence_no, scope_json, residual_json, parser_fingerprint,
                   mode, lifecycle_status, quiescent,
                   inventory_digest, residual_digest, plan_count,
                   post_inventory_digest, post_residual_json,
                   post_residual_digest, post_plan_count, postflight_at_ms,
                   executable_plan_count, residual_plan_count,
                   predecessor_census_id, fixed_point, created_at_ms,
                   completed_at_ms
            FROM raw_authority_censuses WHERE census_id = ?
            """,
            (census_id,),
        ).fetchone()
        if census is None:
            raise KeyError(census_id)
        rows = conn.execute(
            """
            SELECT cp.ordinal, cp.selected, cp.outcome_status, cp.reason,
                   cp.next_action, length(cp.application_receipt_json) AS application_receipt_chars,
                   cp.recorded_at_ms,
                   cp.outcome_recorded,
                   p.plan_id, p.input_digest,
                   json_array_length(p.input_raw_ids_json) AS input_raw_count,
                   json_array_length(p.logical_keys_json) AS logical_key_count,
                   p.authority_witness_json,
                   p.source_preconditions_json, p.index_preconditions_json,
                   p.created_at_ms,
                   (SELECT COUNT(*) FROM raw_authority_blockers AS b
                    WHERE b.census_id = cp.census_id AND b.plan_id = cp.plan_id) AS blocker_count
            FROM raw_authority_census_plans AS cp
            JOIN raw_authority_plans AS p ON p.plan_id = cp.plan_id
            WHERE cp.census_id = ?
            ORDER BY cp.ordinal
            LIMIT ? OFFSET ?
            """,
            (census_id, limit, resolved_offset),
        ).fetchall()
        post_rows = conn.execute(
            """
            SELECT cpp.ordinal, p.plan_id, p.input_digest,
                   json_array_length(p.input_raw_ids_json) AS input_raw_count,
                   json_array_length(p.logical_keys_json) AS logical_key_count
            FROM raw_authority_census_post_plans AS cpp
            JOIN raw_authority_plans AS p ON p.plan_id = cpp.plan_id
            WHERE cpp.census_id = ?
            ORDER BY cpp.ordinal
            LIMIT ? OFFSET ?
            """,
            (census_id, limit, resolved_offset),
        ).fetchall()
    plans = [
        {
            "ordinal": int(row["ordinal"]),
            "selected": bool(row["selected"]),
            "outcome_status": str(row["outcome_status"]),
            "reason_sample": str(row["reason"])[:256],
            "reason_chars": len(str(row["reason"])),
            "next_action_sample": str(row["next_action"])[:256],
            "next_action_chars": len(str(row["next_action"])),
            "application_receipt_chars": int(row["application_receipt_chars"]),
            "outcome_recorded": bool(row["outcome_recorded"]),
            "recorded_at_ms": int(row["recorded_at_ms"]),
            "blocker_count": int(row["blocker_count"]),
            "detail_query_handle": raw_authority_detail_query_handle(census_id, str(row["plan_id"])),
            "plan": {
                "plan_id": str(row["plan_id"]),
                "input_digest": str(row["input_digest"]),
                "input_raw_count": int(row["input_raw_count"]),
                "logical_key_count": int(row["logical_key_count"]),
                "authority_witness_chars": len(str(row["authority_witness_json"])),
                "source_preconditions_chars": len(str(row["source_preconditions_json"])),
                "index_preconditions_chars": len(str(row["index_preconditions_json"])),
                "created_at_ms": int(row["created_at_ms"]),
            },
        }
        for row in rows
    ]
    total = int(census["plan_count"])
    post_total = int(census["post_plan_count"] or 0)
    next_offset = resolved_offset + max(len(plans), len(post_rows))
    return json_document(
        {
            "query_handle": raw_authority_census_query_handle(census_id, offset=resolved_offset),
            "next_query_handle": (
                raw_authority_census_query_handle(census_id, offset=next_offset)
                if next_offset < max(total, post_total)
                else None
            ),
            "offset": resolved_offset,
            "limit": limit,
            "returned_count": len(plans),
            "census": {
                "census_id": str(census["census_id"]),
                "sequence_no": int(census["sequence_no"]),
                "detail_query_handle": raw_authority_detail_query_handle(census_id, "census"),
                "scope_chars": len(str(census["scope_json"])),
                "residual_chars": len(str(census["residual_json"])),
                "parser_fingerprint": str(census["parser_fingerprint"]),
                "mode": str(census["mode"]),
                "lifecycle_status": str(census["lifecycle_status"]),
                "quiescent": bool(census["quiescent"]),
                "inventory_digest": str(census["inventory_digest"]),
                "residual_digest": str(census["residual_digest"]),
                "plan_count": total,
                "post_inventory_digest": census["post_inventory_digest"],
                "post_residual_chars": (
                    len(str(census["post_residual_json"])) if census["post_residual_json"] is not None else None
                ),
                "post_residual_digest": census["post_residual_digest"],
                "post_plan_count": post_total,
                "postflight_at_ms": census["postflight_at_ms"],
                "executable_plan_count": int(census["executable_plan_count"]),
                "residual_plan_count": int(census["residual_plan_count"]),
                "predecessor_census_id": census["predecessor_census_id"],
                "fixed_point": bool(census["fixed_point"]),
                "created_at_ms": int(census["created_at_ms"]),
                "completed_at_ms": (int(census["completed_at_ms"]) if census["completed_at_ms"] is not None else None),
            },
            "plans": plans,
            "post_plans": [
                {
                    "ordinal": int(row["ordinal"]),
                    "plan_id": str(row["plan_id"]),
                    "input_digest": str(row["input_digest"]),
                    "input_raw_count": int(row["input_raw_count"]),
                    "logical_key_count": int(row["logical_key_count"]),
                    "detail_query_handle": raw_authority_detail_query_handle(census_id, str(row["plan_id"])),
                }
                for row in post_rows
            ],
            "blocker_count": sum(int(row["blocker_count"]) for row in rows),
        }
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.hex()
    return value


def _rows(conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()) -> list[dict[str, object]]:
    cursor = conn.execute(sql, tuple(params))
    names = tuple(column[0] for column in cursor.description or ())
    return [{name: _json_value(value) for name, value in zip(names, row, strict=True)} for row in cursor]


def build_raw_replay_plan(conn: sqlite3.Connection, input_raw_ids: Sequence[str]) -> RawReplayPlan:
    """Snapshot one complete component from an attached source/index pair."""
    raw_ids = tuple(sorted(dict.fromkeys(input_raw_ids)))
    if not raw_ids:
        raise ValueError("raw replay plan requires at least one input raw id")
    marks = ",".join("?" for _ in raw_ids)
    source_rows = _rows(
        conn,
        f"""
        SELECT raw_id, origin, native_id, source_path, source_index,
               hex(blob_hash) AS blob_hash, blob_size, logical_source_key,
               revision_kind, source_revision, predecessor_source_revision,
               predecessor_raw_id, baseline_raw_id, append_start_offset,
               append_end_offset, acquisition_generation, revision_authority
        FROM raw_sessions WHERE raw_id IN ({marks}) ORDER BY raw_id
        """,
        raw_ids,
    )
    if tuple(str(row["raw_id"]) for row in source_rows) != raw_ids:
        raise RuntimeError("raw replay plan input disappeared during census")
    membership_rows = _rows(
        conn,
        f"""
        SELECT raw_id, logical_source_key, provider_session_id, source_revision,
               hex(normalized_content_hash) AS normalized_content_hash,
               message_count, predecessor_raw_id, acquisition_generation,
               revision_authority, decision
        FROM raw_session_memberships
        WHERE raw_id IN ({marks})
        ORDER BY raw_id, logical_source_key
        """,
        raw_ids,
    )
    census_rows = _rows(
        conn,
        f"""
        SELECT raw_id, parser_fingerprint, status, member_count, detail
        FROM raw_membership_census
        WHERE raw_id IN ({marks}) ORDER BY raw_id
        """,
        raw_ids,
    )
    parser_census_rows = _rows(
        conn,
        f"""
        SELECT raw_id, parser_fingerprint, status, logical_keys_json, detail
        FROM raw_authority_parser_census
        WHERE raw_id IN ({marks}) ORDER BY raw_id
        """,
        raw_ids,
    )
    logical_keys = tuple(
        sorted(
            {
                str(value)
                for row in (*source_rows, *membership_rows)
                if (value := row.get("logical_source_key")) is not None
            }
        )
    )
    if logical_keys:
        key_marks = ",".join("?" for _ in logical_keys)
        head_rows = _rows(
            conn,
            f"""
            SELECT logical_source_key, session_id, accepted_raw_id,
                   accepted_source_revision, hex(accepted_content_hash) AS accepted_content_hash,
                   accepted_frontier_kind, accepted_frontier,
                   acquisition_generation, append_end_offset
            FROM index_tier.raw_revision_heads
            WHERE logical_source_key IN ({key_marks}) ORDER BY logical_source_key
            """,
            logical_keys,
        )
    else:
        head_rows = []
    session_rows = _rows(
        conn,
        f"""
        SELECT session_id, raw_id, hex(content_hash) AS content_hash, message_count
        FROM index_tier.sessions
        WHERE raw_id IN ({marks}) ORDER BY session_id
        """,
        raw_ids,
    )
    authority_witness = json_document(
        {
            "parser_census": parser_census_rows,
            "membership_census": census_rows,
            "memberships": membership_rows,
            "revision_heads": head_rows,
        }
    )
    source_preconditions = json_document(
        {"raw_sessions": source_rows, "raw_authority_parser_census": parser_census_rows}
    )
    index_preconditions = json_document({"sessions": session_rows, "revision_heads": head_rows})
    identity = {
        "schema": "polylogue.raw-replay-plan.v2",
        "input_raw_ids": list(raw_ids),
        "logical_keys": list(logical_keys),
        "authority_witness": authority_witness,
        "source_preconditions": source_preconditions,
        "index_preconditions": index_preconditions,
    }
    input_digest = _digest(identity)
    return RawReplayPlan(
        plan_id=f"raw-replay:{input_digest}",
        input_digest=input_digest,
        input_raw_ids=raw_ids,
        logical_keys=logical_keys,
        authority_witness=authority_witness,
        source_preconditions=source_preconditions,
        index_preconditions=index_preconditions,
    )


def build_raw_replay_plans(archive_root: Path, components: Sequence[tuple[str, ...]]) -> tuple[RawReplayPlan, ...]:
    if not components:
        return ()
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(archive_root / "index.db"),))
        return tuple(build_raw_replay_plan(conn, component) for component in components)


def raw_replay_plan_last_attempts(archive_root: Path) -> dict[str, int]:
    """Return durable attempt order; deleting ops.db cannot reset fairness."""
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return {}
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_authority_census_plans'"
        ).fetchone()
        if exists is None:
            return {}
        return {
            str(row[0]): int(row[1])
            for row in conn.execute(
                """
                SELECT cp.plan_id, MAX(cp.recorded_at_ms)
                FROM raw_authority_census_plans AS cp
                JOIN raw_authority_censuses AS c ON c.census_id = cp.census_id
                WHERE cp.selected = 1
                  AND c.lifecycle_status IN ('completed', 'interrupted')
                GROUP BY cp.plan_id
                """
            )
        }


def unresolved_raw_authority_blockers(archive_root: Path) -> int:
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return 0
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_authority_blockers'"
        ).fetchone()
        if exists is None:
            return 0
        return int(
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0]
        )


def record_raw_authority_census(
    archive_root: Path,
    plans: Sequence[RawReplayPlan],
    *,
    selected_plan_ids: set[str],
    executable_plan_ids: set[str] | None = None,
    mode: str,
    quiescent: bool,
    scope: Mapping[str, object],
    residual: Mapping[str, object],
) -> RawAuthorityCensusReceipt:
    """Atomically publish a plan census, finalized only when no apply is pending."""
    if mode not in {"census", "dry_run", "apply"}:
        raise ValueError(f"unsupported raw authority census mode: {mode}")
    if mode != "apply" and selected_plan_ids:
        raise ValueError(f"{mode} census cannot select plans for application")
    now = int(time.time() * 1000)
    inventory_digest = _digest([plan.plan_id for plan in plans])
    residual_digest = _digest(residual)
    scope_json = _canonical_json(scope)
    residual_json = _canonical_json(residual)
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        previous = conn.execute(
            """
            SELECT census_id, sequence_no, inventory_digest, residual_digest,
                   executable_plan_count, scope_json, parser_fingerprint,
                   mode, lifecycle_status, quiescent
            FROM raw_authority_censuses ORDER BY sequence_no DESC LIMIT 1
            """
        ).fetchone()
        sequence_no = int(previous[1]) + 1 if previous is not None else 1
        predecessor = str(previous[0]) if previous is not None else None
        executable_ids = selected_plan_ids if executable_plan_ids is None else executable_plan_ids
        unknown_ids = (selected_plan_ids | executable_ids) - {plan.plan_id for plan in plans}
        if unknown_ids:
            raise RuntimeError(f"raw authority census references unknown plans: {sorted(unknown_ids)}")
        executable_count = len(executable_ids)
        residual_count = len(plans) - executable_count
        fixed_point = bool(
            previous is not None
            and int(previous[4]) == 0
            and executable_count == 0
            and str(previous[2]) == inventory_digest
            and str(previous[3]) == residual_digest
            and str(previous[5]) == scope_json
            and str(previous[6]) == RAW_AUTHORITY_PARSER_FINGERPRINT
            and str(previous[7]) == "dry_run"
            and str(previous[8]) == "completed"
            and bool(previous[9])
            and mode == "dry_run"
            and quiescent
        )
        census_id = f"census:{sequence_no}:{inventory_digest[:16]}:{residual_digest[:16]}"
        lifecycle_status = "planned" if mode == "apply" and selected_plan_ids else "completed"
        completed_at_ms = None if lifecycle_status == "planned" else now
        post_inventory_digest = None if lifecycle_status == "planned" else inventory_digest
        post_residual_json = None if lifecycle_status == "planned" else residual_json
        post_residual_digest = None if lifecycle_status == "planned" else residual_digest
        post_plan_count = None if lifecycle_status == "planned" else len(plans)
        postflight_at_ms = None if lifecycle_status == "planned" else now
        for plan in plans:
            values = (
                plan.plan_id,
                plan.input_digest,
                _canonical_json(list(plan.input_raw_ids)),
                _canonical_json(list(plan.logical_keys)),
                _canonical_json(plan.authority_witness),
                _canonical_json(plan.source_preconditions),
                _canonical_json(plan.index_preconditions),
                now,
            )
            conn.execute(
                """
                INSERT INTO raw_authority_plans (
                    plan_id, input_digest, input_raw_ids_json, logical_keys_json,
                    authority_witness_json, source_preconditions_json,
                    index_preconditions_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO NOTHING
                """,
                values,
            )
            stored = conn.execute(
                """
                SELECT input_digest, input_raw_ids_json, logical_keys_json,
                       authority_witness_json, source_preconditions_json,
                       index_preconditions_json
                FROM raw_authority_plans WHERE plan_id = ?
                """,
                (plan.plan_id,),
            ).fetchone()
            if stored != values[1:7]:
                raise RuntimeError(f"immutable raw replay plan collision: {plan.plan_id}")
        conn.execute(
            """
            INSERT INTO raw_authority_censuses (
                census_id, sequence_no, scope_json, residual_json,
                parser_fingerprint, mode, lifecycle_status, quiescent,
                inventory_digest, residual_digest, plan_count,
                post_inventory_digest, post_residual_json,
                post_residual_digest, post_plan_count, postflight_at_ms,
                executable_plan_count, residual_plan_count,
                predecessor_census_id, fixed_point, created_at_ms, completed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                census_id,
                sequence_no,
                scope_json,
                residual_json,
                RAW_AUTHORITY_PARSER_FINGERPRINT,
                mode,
                lifecycle_status,
                int(quiescent),
                inventory_digest,
                residual_digest,
                len(plans),
                post_inventory_digest,
                post_residual_json,
                post_residual_digest,
                post_plan_count,
                postflight_at_ms,
                executable_count,
                residual_count,
                predecessor,
                int(fixed_point),
                now,
                completed_at_ms,
            ),
        )
        for ordinal, plan in enumerate(plans):
            selected = plan.plan_id in selected_plan_ids
            conn.execute(
                """
                INSERT INTO raw_authority_census_plans (
                    census_id, plan_id, ordinal, selected, outcome_status,
                    reason, next_action, application_receipt_json,
                    outcome_recorded, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?, ?)
                """,
                (
                    census_id,
                    plan.plan_id,
                    ordinal,
                    int(selected),
                    RawReplayPlanStatus.RETRYABLE.value if selected else RawReplayPlanStatus.CARRIED_FORWARD.value,
                    "selected plan application is pending"
                    if selected
                    else "bounded scheduler carried this complete plan forward unchanged",
                    "execute this plan in the current pass" if selected else "retain for a later bounded pass",
                    0 if selected else 1,
                    now,
                ),
            )
        if lifecycle_status != "planned":
            conn.executemany(
                """
                INSERT INTO raw_authority_census_post_plans (census_id, plan_id, ordinal)
                VALUES (?, ?, ?)
                """,
                ((census_id, plan.plan_id, ordinal) for ordinal, plan in enumerate(plans)),
            )
    return RawAuthorityCensusReceipt(
        census_id=census_id,
        sequence_no=sequence_no,
        inventory_digest=inventory_digest,
        residual_digest=residual_digest,
        plan_count=len(plans),
        executable_plan_count=executable_count,
        residual_plan_count=residual_count,
        post_inventory_digest=post_inventory_digest,
        post_residual_digest=post_residual_digest,
        post_plan_count=post_plan_count,
        predecessor_census_id=predecessor,
        mode=mode,
        lifecycle_status=lifecycle_status,
        quiescent=quiescent,
        fixed_point=fixed_point,
    )


def validate_raw_replay_plan(archive_root: Path, plan: RawReplayPlan) -> tuple[bool, JSONDocument]:
    try:
        observed = build_raw_replay_plans(archive_root, (plan.input_raw_ids,))[0]
    except Exception as exc:
        logger.warning("raw replay plan validation could not rebuild %s", plan.plan_id, exc_info=True)
        return False, json_document({"error": f"{type(exc).__name__}: {exc}"})
    return observed == plan, observed.to_dict()


def raw_replay_application_receipt(archive_root: Path, plan: RawReplayPlan) -> JSONDocument:
    marks = ",".join("?" for _ in plan.input_raw_ids)
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(archive_root / "index.db"),))
        source = _rows(
            conn,
            f"""
            SELECT raw_id, parsed_at_ms, parse_error
            FROM raw_sessions WHERE raw_id IN ({marks}) ORDER BY raw_id
            """,
            plan.input_raw_ids,
        )
        memberships = _rows(
            conn,
            f"""
            SELECT raw_id, logical_source_key, decision, decided_at_ms
            FROM raw_session_memberships
            WHERE raw_id IN ({marks}) ORDER BY raw_id, logical_source_key
            """,
            plan.input_raw_ids,
        )
        applications = _rows(
            conn,
            f"""
            SELECT decision_id, raw_id, session_id, logical_source_key, decision,
                   accepted_raw_id, hex(accepted_content_hash) AS accepted_content_hash,
                   decided_at_ms
            FROM index_tier.raw_revision_applications
            WHERE raw_id IN ({marks}) ORDER BY raw_id, decision_id
            """,
            plan.input_raw_ids,
        )
        if plan.logical_keys:
            key_marks = ",".join("?" for _ in plan.logical_keys)
            heads = _rows(
                conn,
                f"""
                SELECT logical_source_key, session_id, accepted_raw_id,
                       accepted_source_revision,
                       hex(accepted_content_hash) AS accepted_content_hash,
                       accepted_frontier_kind, accepted_frontier
                FROM index_tier.raw_revision_heads
                WHERE logical_source_key IN ({key_marks})
                ORDER BY logical_source_key
                """,
                plan.logical_keys,
            )
            sessions = _rows(
                conn,
                f"""
                SELECT s.session_id, s.raw_id, hex(s.content_hash) AS content_hash,
                       s.message_count
                FROM index_tier.sessions AS s
                JOIN index_tier.raw_revision_heads AS h ON h.session_id = s.session_id
                WHERE h.logical_source_key IN ({key_marks})
                ORDER BY s.session_id
                """,
                plan.logical_keys,
            )
        else:
            heads = []
            sessions = []
    return json_document(
        {
            "schema": "polylogue.raw-replay-application-receipt.v2",
            "source_rows": source,
            "membership_rows": memberships,
            "application_rows": applications,
            "head_rows": heads,
            "session_rows": sessions,
        }
    )


def validate_raw_replay_application_receipt(
    plan: RawReplayPlan,
    receipt: Mapping[str, object],
) -> tuple[bool, tuple[str, ...]]:
    """Prove exact replay postconditions; parsed timestamps are never sufficient."""
    problems: list[str] = []
    if receipt.get("schema") != "polylogue.raw-replay-application-receipt.v2":
        problems.append("application receipt schema is not v2")

    def rows(name: str) -> list[Mapping[str, object]]:
        value = receipt.get(name)
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            problems.append(f"{name} is not a row list")
            return []
        return value

    source_rows = rows("source_rows")
    membership_rows = rows("membership_rows")
    application_rows = rows("application_rows")
    head_rows = rows("head_rows")
    session_rows = rows("session_rows")
    source_ids = {str(row.get("raw_id")) for row in source_rows}
    if source_ids != set(plan.input_raw_ids):
        problems.append("source receipt raw ids do not match the immutable plan")
    if any(row.get("parsed_at_ms") is None or row.get("parse_error") is not None for row in source_rows):
        problems.append("source receipt contains an unparsed or parse-failed raw")
    expected_keys = set(plan.logical_keys)
    head_keys = {str(row.get("logical_source_key")) for row in head_rows}
    if not expected_keys:
        problems.append("executed replay plan has no logical authority keys")
    elif head_keys != expected_keys:
        problems.append("accepted head keys do not match the immutable plan")
    input_raw_ids = set(plan.input_raw_ids)
    witness = plan.authority_witness.get("memberships")
    expected_memberships = (
        {(str(row.get("raw_id")), str(row.get("logical_source_key"))) for row in witness if isinstance(row, dict)}
        if isinstance(witness, list)
        else set()
    )
    observed_memberships = {(str(row.get("raw_id")), str(row.get("logical_source_key"))) for row in membership_rows}
    if observed_memberships != expected_memberships:
        problems.append("membership receipt pairs do not match the immutable authority witness")
    terminal_membership_decisions = {"applied", "superseded_equivalent", "superseded_prefix"}
    if any(row.get("decision") not in terminal_membership_decisions for row in membership_rows):
        problems.append("membership receipt contains a non-terminal decision")
    terminal_application_decisions = {"selected_baseline", "applied_append", "superseded"}
    application_pairs = {(str(row.get("raw_id")), str(row.get("logical_source_key"))) for row in application_rows}
    if any(row.get("decision") not in terminal_application_decisions for row in application_rows):
        problems.append("application receipt contains a non-terminal decision")
    if any(raw_id not in input_raw_ids or key not in expected_keys for raw_id, key in application_pairs):
        problems.append("application receipt contains authority outside the immutable component")
    terminal_keys = {key for _, key in observed_memberships | application_pairs}
    if terminal_keys != expected_keys:
        problems.append("terminal receipt keys do not exactly match the immutable plan")
    head_session_ids = {str(row.get("session_id")) for row in head_rows}
    session_ids = {str(row.get("session_id")) for row in session_rows}
    if head_session_ids != session_ids:
        problems.append("accepted head sessions do not match materialized session rows")
    session_content = {str(row.get("session_id")): str(row.get("content_hash")) for row in session_rows}
    if any(
        str(row.get("accepted_content_hash")) != session_content.get(str(row.get("session_id"))) for row in head_rows
    ):
        problems.append("accepted head content hashes do not match materialized sessions")
    if any(str(row.get("accepted_raw_id")) not in input_raw_ids for row in head_rows):
        problems.append("accepted heads do not point into the immutable input component")
    heads_by_key = {str(row.get("logical_source_key")): row for row in head_rows}
    if len(heads_by_key) != len(head_rows):
        problems.append("accepted head receipt contains duplicate logical authority keys")
    sessions_by_id: dict[str, Mapping[str, object]] = {}
    for session in session_rows:
        session_id = str(session.get("session_id"))
        existing_session = sessions_by_id.setdefault(session_id, session)
        if existing_session != session:
            problems.append(f"materialized session receipt conflicts for {session_id}")
    application_keys: set[str] = set()
    applications_matching_current_head: set[str] = set()
    for application in application_rows:
        key = str(application.get("logical_source_key"))
        application_keys.add(key)
        head = heads_by_key.get(key)
        if head is None:
            problems.append(f"application receipt has no accepted head for {key}")
            continue
        application_authority = (
            str(application.get("session_id")),
            str(application.get("accepted_raw_id")),
            str(application.get("accepted_content_hash")),
        )
        head_authority = (
            str(head.get("session_id")),
            str(head.get("accepted_raw_id")),
            str(head.get("accepted_content_hash")),
        )
        if application_authority == head_authority:
            applications_matching_current_head.add(key)
        session = sessions_by_id.get(str(head.get("session_id")))
        if session is None:
            continue
        if str(session.get("raw_id")) != str(head.get("accepted_raw_id")) or str(session.get("content_hash")) != str(
            head.get("accepted_content_hash")
        ):
            problems.append(f"materialized session authority does not match the head for {key}")
    for key in sorted(application_keys - applications_matching_current_head):
        problems.append(f"no application accepted authority matches the current head for {key}")
    return not problems, tuple(problems)


def record_raw_replay_outcome(
    archive_root: Path,
    census_id: str,
    outcome: RawReplayPlanOutcome,
) -> None:
    now = int(time.time() * 1000)
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        updated = conn.execute(
            """
            UPDATE raw_authority_census_plans
            SET outcome_status = ?, reason = ?, next_action = ?,
                application_receipt_json = ?, outcome_recorded = 1,
                recorded_at_ms = ?
            WHERE census_id = ? AND plan_id = ? AND selected = 1
              AND outcome_recorded = 0
            """,
            (
                outcome.status.value,
                outcome.reason,
                outcome.next_action,
                _canonical_json(outcome.application_receipt or {}),
                now,
                census_id,
                outcome.plan_id,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"outcome does not conserve one selected plan: {outcome.plan_id}")


def _raw_replay_plan_from_row(row: sqlite3.Row) -> RawReplayPlan:
    return RawReplayPlan(
        plan_id=str(row["plan_id"]),
        input_digest=str(row["input_digest"]),
        input_raw_ids=tuple(str(value) for value in json.loads(str(row["input_raw_ids_json"]))),
        logical_keys=tuple(str(value) for value in json.loads(str(row["logical_keys_json"]))),
        authority_witness=json_document(json.loads(str(row["authority_witness_json"]))),
        source_preconditions=json_document(json.loads(str(row["source_preconditions_json"]))),
        index_preconditions=json_document(json.loads(str(row["index_preconditions_json"]))),
    )


def _raw_authority_census_receipt(conn: sqlite3.Connection, census_id: str) -> RawAuthorityCensusReceipt:
    row = conn.execute(
        """
        SELECT census_id, sequence_no, inventory_digest, residual_digest,
               plan_count, executable_plan_count, residual_plan_count,
               post_inventory_digest, post_residual_digest, post_plan_count,
               predecessor_census_id, mode, lifecycle_status, quiescent,
               fixed_point
        FROM raw_authority_censuses WHERE census_id = ?
        """,
        (census_id,),
    ).fetchone()
    if row is None:
        raise KeyError(census_id)
    return RawAuthorityCensusReceipt(
        census_id=str(row[0]),
        sequence_no=int(row[1]),
        inventory_digest=str(row[2]),
        residual_digest=str(row[3]),
        plan_count=int(row[4]),
        executable_plan_count=int(row[5]),
        residual_plan_count=int(row[6]),
        post_inventory_digest=str(row[7]) if row[7] is not None else None,
        post_residual_digest=str(row[8]) if row[8] is not None else None,
        post_plan_count=int(row[9]) if row[9] is not None else None,
        predecessor_census_id=str(row[10]) if row[10] is not None else None,
        mode=str(row[11]),
        lifecycle_status=str(row[12]),
        quiescent=bool(row[13]),
        fixed_point=bool(row[14]),
    )


def finalize_raw_authority_census(
    archive_root: Path,
    census_id: str,
    *,
    post_plans: Sequence[RawReplayPlan],
    post_residual: Mapping[str, object],
    interrupted: bool = False,
) -> RawAuthorityCensusReceipt:
    """Publish a census only after every selected plan has a recorded outcome."""
    now = int(time.time() * 1000)
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        status = conn.execute(
            "SELECT lifecycle_status FROM raw_authority_censuses WHERE census_id = ?",
            (census_id,),
        ).fetchone()
        if status is None:
            raise KeyError(census_id)
        if str(status[0]) != "planned":
            return _raw_authority_census_receipt(conn, census_id)
        pending = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM raw_authority_census_plans
                WHERE census_id = ? AND selected = 1 AND outcome_recorded = 0
                """,
                (census_id,),
            ).fetchone()[0]
        )
        if pending:
            raise RuntimeError(f"raw authority census still has {pending} pending selected outcome(s)")
        post_ids = {plan.plan_id for plan in post_plans}
        persistent = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT plan_id FROM raw_authority_census_plans
                WHERE census_id = ? AND outcome_status IN ('retryable', 'carried_forward')
                """,
                (census_id,),
            )
        }
        if not persistent.issubset(post_ids):
            raise RuntimeError(
                f"raw authority postflight changed a retryable/carried-forward plan: {sorted(persistent - post_ids)}"
            )
        for plan in post_plans:
            values = (
                plan.plan_id,
                plan.input_digest,
                _canonical_json(list(plan.input_raw_ids)),
                _canonical_json(list(plan.logical_keys)),
                _canonical_json(plan.authority_witness),
                _canonical_json(plan.source_preconditions),
                _canonical_json(plan.index_preconditions),
                now,
            )
            conn.execute(
                """
                INSERT INTO raw_authority_plans (
                    plan_id, input_digest, input_raw_ids_json, logical_keys_json,
                    authority_witness_json, source_preconditions_json,
                    index_preconditions_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO NOTHING
                """,
                values,
            )
            stored = conn.execute(
                """
                SELECT input_digest, input_raw_ids_json, logical_keys_json,
                       authority_witness_json, source_preconditions_json,
                       index_preconditions_json
                FROM raw_authority_plans WHERE plan_id = ?
                """,
                (plan.plan_id,),
            ).fetchone()
            if stored != values[1:7]:
                raise RuntimeError(f"immutable raw replay postflight plan collision: {plan.plan_id}")
        conn.executemany(
            """
            INSERT INTO raw_authority_census_post_plans (census_id, plan_id, ordinal)
            VALUES (?, ?, ?)
            """,
            ((census_id, plan.plan_id, ordinal) for ordinal, plan in enumerate(post_plans)),
        )
        post_inventory_digest = _digest([plan.plan_id for plan in post_plans])
        post_residual_json = _canonical_json(post_residual)
        post_residual_digest = _digest(post_residual)
        conn.execute(
            """
            UPDATE raw_authority_censuses
            SET lifecycle_status = ?, completed_at_ms = ?,
                post_inventory_digest = ?, post_residual_json = ?,
                post_residual_digest = ?, post_plan_count = ?, postflight_at_ms = ?
            WHERE census_id = ? AND lifecycle_status = 'planned'
            """,
            (
                "interrupted" if interrupted else "completed",
                now,
                post_inventory_digest,
                post_residual_json,
                post_residual_digest,
                len(post_plans),
                now,
                census_id,
            ),
        )
        return _raw_authority_census_receipt(conn, census_id)


def recover_interrupted_raw_authority_censuses(
    archive_root: Path,
) -> tuple[tuple[str, JSONDocument], ...]:
    """Reconcile unfinished apply censuses from durable postconditions."""
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return ()
    with closing(sqlite3.connect(source_db)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.census_id, p.*
            FROM raw_authority_censuses AS c
            JOIN raw_authority_census_plans AS cp ON cp.census_id = c.census_id
            JOIN raw_authority_plans AS p ON p.plan_id = cp.plan_id
            WHERE c.lifecycle_status = 'planned'
              AND cp.selected = 1 AND cp.outcome_recorded = 0
            ORDER BY c.sequence_no, cp.ordinal
            """
        ).fetchall()
        census_scopes = tuple(
            (str(row[0]), json_document(json.loads(str(row[1]))))
            for row in conn.execute(
                """
                SELECT census_id, scope_json
                FROM raw_authority_censuses
                WHERE lifecycle_status = 'planned'
                ORDER BY sequence_no
                """
            )
        )
    for row in rows:
        census_id = str(row["census_id"])
        plan = _raw_replay_plan_from_row(row)
        receipt = raw_replay_application_receipt(archive_root, plan)
        valid_receipt, problems = validate_raw_replay_application_receipt(plan, receipt)
        if valid_receipt:
            outcome = RawReplayPlanOutcome(
                plan.plan_id,
                plan.input_raw_ids,
                RawReplayPlanStatus.EXECUTED,
                "interrupted application recovered from exact durable postconditions",
                "none",
                receipt,
            )
            record_raw_replay_outcome(archive_root, census_id, outcome)
            continue
        valid_plan, observed = validate_raw_replay_plan(archive_root, plan)
        if not valid_plan:
            reject_stale_raw_replay_plan(archive_root, census_id, plan, observed)
        else:
            outcome = RawReplayPlanOutcome(
                plan.plan_id,
                plan.input_raw_ids,
                RawReplayPlanStatus.RETRYABLE,
                "interrupted before exact application postconditions were durable: " + "; ".join(problems),
                "retry the same immutable plan",
                receipt,
            )
            record_raw_replay_outcome(archive_root, census_id, outcome)
    return census_scopes


def resolve_raw_authority_blocker(archive_root: Path, blocker_id: str, *, resolution: str) -> JSONDocument:
    """Explicitly acknowledge current evidence and reopen replanning."""
    if not resolution.strip():
        raise ValueError("raw authority blocker resolution must be non-empty")
    source_db = archive_root / "source.db"
    with closing(sqlite3.connect(source_db)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(archive_root / "index.db"),))
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT b.blocker_id, b.plan_id, b.census_id, b.expected_json, p.input_raw_ids_json
            FROM raw_authority_blockers AS b
            JOIN raw_authority_plans AS p ON p.plan_id = b.plan_id
            WHERE b.blocker_id = ? AND b.resolved_at_ms IS NULL
            """,
            (blocker_id,),
        ).fetchone()
        if row is None:
            conn.rollback()
            raise KeyError(blocker_id)
        input_raw_ids = tuple(str(value) for value in json.loads(str(row["input_raw_ids_json"])))
        observed = build_raw_replay_plan(conn, input_raw_ids)
        now = int(time.time() * 1000)
        full_receipt = json_document(
            {
                "schema": "polylogue.raw-authority-blocker-resolution.v1",
                "blocker_id": blocker_id,
                "superseded_plan_id": str(row["plan_id"]),
                "current_plan": observed.to_dict(),
                "operator_resolution": resolution.strip(),
                "resolved_at_ms": now,
            }
        )
        updated = conn.execute(
            """
            UPDATE raw_authority_blockers
            SET resolved_at_ms = ?, resolution = ?
            WHERE blocker_id = ? AND resolved_at_ms IS NULL
            """,
            (now, _canonical_json(full_receipt), blocker_id),
        ).rowcount
        if updated != 1:
            conn.rollback()
            raise RuntimeError(f"raw authority blocker changed during resolution: {blocker_id}")
        conn.commit()
    return json_document(
        {
            "schema": "polylogue.raw-authority-blocker-resolution-summary.v1",
            "blocker_id": blocker_id,
            "superseded_plan_id": str(row["plan_id"]),
            "current_plan": {
                "plan_id": observed.plan_id,
                "input_digest": observed.input_digest,
                "input_raw_count": len(observed.input_raw_ids),
                "logical_key_count": len(observed.logical_keys),
            },
            "operator_resolution": resolution.strip(),
            "resolved_at_ms": now,
            "detail_query_handle": raw_authority_detail_query_handle(str(row["census_id"]), str(row["plan_id"])),
        }
    )


def reject_stale_raw_replay_plan(
    archive_root: Path,
    census_id: str,
    plan: RawReplayPlan,
    observed: JSONDocument,
) -> RawReplayPlanOutcome:
    """Persist the fail-closed blocker before returning observational output."""
    now = int(time.time() * 1000)
    blocker_id = f"raw-authority-blocker:{_digest([census_id, plan.plan_id, observed])}"
    outcome = RawReplayPlanOutcome(
        plan.plan_id,
        plan.input_raw_ids,
        RawReplayPlanStatus.REJECTED_STALE,
        "immutable source/index preconditions changed after the census",
        "resolve the durable raw-authority blocker before automatic convergence resumes",
        json_document({"expected": plan.to_dict(), "observed": observed, "blocker_id": blocker_id}),
    )
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        conn.execute(
            """
            INSERT INTO raw_authority_blockers (
                blocker_id, plan_id, census_id, reason, expected_json,
                observed_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(blocker_id) DO NOTHING
            """,
            (
                blocker_id,
                plan.plan_id,
                census_id,
                outcome.reason,
                _canonical_json(plan.to_dict()),
                _canonical_json(observed),
                now,
            ),
        )
        updated = conn.execute(
            """
            UPDATE raw_authority_census_plans
            SET outcome_status = 'rejected_stale', reason = ?, next_action = ?,
                application_receipt_json = ?, outcome_recorded = 1,
                recorded_at_ms = ?
            WHERE census_id = ? AND plan_id = ? AND selected = 1
              AND outcome_recorded = 0
            """,
            (
                outcome.reason,
                outcome.next_action,
                _canonical_json(outcome.application_receipt or {}),
                now,
                census_id,
                plan.plan_id,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"stale rejection does not conserve one selected plan: {plan.plan_id}")
        open_count = conn.execute(
            "SELECT COUNT(*) FROM raw_authority_blockers WHERE plan_id = ? AND resolved_at_ms IS NULL",
            (plan.plan_id,),
        ).fetchone()[0]
        if int(open_count) != 1:
            raise RuntimeError(f"stale rejection did not leave one open blocker: {plan.plan_id}")
    return outcome


def reject_invalid_raw_replay_application(
    archive_root: Path,
    census_id: str,
    plan: RawReplayPlan,
    receipt: JSONDocument,
    problems: Sequence[str],
) -> RawReplayPlanOutcome:
    """Fail closed when a writer returns without exact application postconditions."""
    now = int(time.time() * 1000)
    observed = json_document({"application_receipt": receipt, "problems": list(problems)})
    blocker_id = f"raw-authority-blocker:{_digest([census_id, plan.plan_id, observed])}"
    outcome = RawReplayPlanOutcome(
        plan.plan_id,
        plan.input_raw_ids,
        RawReplayPlanStatus.REJECTED_STALE,
        "raw replay application did not satisfy exact durable postconditions",
        "resolve the durable raw-authority blocker before automatic convergence resumes",
        json_document({"expected": plan.to_dict(), "observed": observed, "blocker_id": blocker_id}),
    )
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        conn.execute(
            """
            INSERT INTO raw_authority_blockers (
                blocker_id, plan_id, census_id, reason, expected_json,
                observed_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(blocker_id) DO NOTHING
            """,
            (
                blocker_id,
                plan.plan_id,
                census_id,
                outcome.reason,
                _canonical_json(plan.to_dict()),
                _canonical_json(observed),
                now,
            ),
        )
        updated = conn.execute(
            """
            UPDATE raw_authority_census_plans
            SET outcome_status = 'rejected_stale', reason = ?, next_action = ?,
                application_receipt_json = ?, outcome_recorded = 1,
                recorded_at_ms = ?
            WHERE census_id = ? AND plan_id = ? AND selected = 1
              AND outcome_recorded = 0
            """,
            (
                outcome.reason,
                outcome.next_action,
                _canonical_json(outcome.application_receipt or {}),
                now,
                census_id,
                plan.plan_id,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"invalid application does not conserve one selected plan: {plan.plan_id}")
        open_count = conn.execute(
            "SELECT COUNT(*) FROM raw_authority_blockers WHERE plan_id = ? AND resolved_at_ms IS NULL",
            (plan.plan_id,),
        ).fetchone()[0]
        if int(open_count) != 1:
            raise RuntimeError(f"invalid application did not leave one open blocker: {plan.plan_id}")
    return outcome


__all__ = [
    "RAW_AUTHORITY_CENSUS_QUERY_PREFIX",
    "RAW_AUTHORITY_DETAIL_CHUNK_CHARS",
    "RAW_AUTHORITY_DETAIL_QUERY_PREFIX",
    "RAW_AUTHORITY_PARSER_FINGERPRINT",
    "RawAuthorityCensusReceipt",
    "RawReplayPlan",
    "RawReplayPlanOutcome",
    "RawReplayPlanStatus",
    "build_raw_replay_plan",
    "build_raw_replay_plans",
    "finalize_raw_authority_census",
    "raw_replay_application_receipt",
    "raw_authority_census_query_handle",
    "raw_authority_detail_query_handle",
    "raw_replay_plan_last_attempts",
    "recover_interrupted_raw_authority_censuses",
    "read_raw_authority_census",
    "read_raw_authority_detail",
    "record_raw_authority_census",
    "record_raw_replay_outcome",
    "reject_invalid_raw_replay_application",
    "reject_stale_raw_replay_plan",
    "resolve_raw_authority_blocker",
    "unresolved_raw_authority_blockers",
    "validate_raw_replay_plan",
    "validate_raw_replay_application_receipt",
]
