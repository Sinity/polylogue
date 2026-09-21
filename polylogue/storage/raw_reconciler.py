"""Read-only census for every accepted raw-authority frontier.

This module classifies each accepted head against its durable evidence and
publishes the blocking ones as durable ``raw_authority_blockers`` obligations.
It applies nothing. A frontier state is either an explicit retryable obligation
that ordinary acquisition or derivation discharges, or a typed permanent
refusal an operator resolves through the declared daemon mutation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import time
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.config import Config
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.storage.archive_identity import archive_file_set_root
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import (
    BLOCKER_ORIGIN_FRONTIER_OBLIGATION,
    BLOCKER_ORIGIN_KEY,
    RawReplayPlan,
)

logger = get_logger(__name__)


class RawAuthorityFrontierState(StrEnum):
    """Mutually exclusive authority states for one accepted frontier.

    Every non-terminal state names who discharges it outside this module:
    ``missing_bytes_reacquire`` waits on ordinary acquisition,
    ``unresolved_provenance`` and ``corrupt`` are typed permanent refusals an
    operator resolves through ``mutation.raw-authority-blocker.resolve``. None
    of them schedules work inside the census.
    """

    PROVEN_CURRENT = "proven_current"
    SUPERSEDED = "superseded"
    MISSING_BYTES_REACQUIRE = "missing_bytes_reacquire"
    UNRESOLVED_PROVENANCE = "unresolved_provenance"
    CORRUPT = "corrupt"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, (bytes, memoryview)):
        return bytes(value).hex()
    return value


@dataclass(frozen=True, slots=True)
class RawAuthorityFrontierItem:
    """One complete, stable, evidence-bound frontier classification."""

    state: RawAuthorityFrontierState
    raw_id: str
    logical_source_key: str | None
    session_id: str | None
    reason: str
    evidence_digest: str
    input_raw_ids: tuple[str, ...]
    source_preconditions: JSONDocument
    index_preconditions: JSONDocument
    plan_id: str
    evidence_ref: str | None = None

    def to_dict(self) -> JSONDocument:
        return json_document(dataclasses.asdict(self))


@dataclass(frozen=True, slots=True)
class RawAuthorityFrontierCensus:
    """One inspection pass over accepted heads plus terminal supersessions.

    The pass itself is not durable. ``pass_id`` is a content address over the
    inspected inventory, so two passes that observe the same frontier name the
    same pass; the only rows this publishes are the durable obligations in
    ``raw_authority_blockers``, which each item points at through
    ``evidence_ref``.
    """

    pass_id: str
    inventory_digest: str
    plan_inventory_digest: str
    state_counts: JSONDocument
    accepted_head_count: int
    terminal_superseded_count: int
    plan_count: int
    items: tuple[RawAuthorityFrontierItem, ...]

    def to_dict(self, *, sample_limit: int = 100) -> JSONDocument:
        sample = self.items[:sample_limit]
        return json_document(
            {
                "schema": "polylogue.raw-authority-frontier-census.v1",
                "pass_id": self.pass_id,
                "inventory_digest": self.inventory_digest,
                "plan_inventory_digest": self.plan_inventory_digest,
                "state_counts": self.state_counts,
                "accepted_head_count": self.accepted_head_count,
                "terminal_superseded_count": self.terminal_superseded_count,
                "plan_count": self.plan_count,
                "returned_count": len(sample),
                "items_truncated": len(sample) < len(self.items),
                "items": [item.to_dict() for item in sample],
            }
        )


def _archive_root(config: Config) -> Path:
    """Return the archive file-set root housing the currently active database.

    Deliberately follows ``config.db_path`` (not ``config.archive_root``),
    matching :func:`polylogue.config.active_archive_file_set_root`:
    this reconciler inspects the database and blob store that are actually
    live right now, which ``config.db_path`` already resolves correctly
    (``.index-active-pointer``-aware, or an explicit override) inside
    ``Config.__init__``.
    """
    return archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, object]]:
    names = tuple(column[0] for column in cursor.description or ())
    return [{name: _json_value(value) for name, value in zip(names, row, strict=True)} for row in cursor.fetchall()]


def _blob_receipt_fingerprint(conn: sqlite3.Connection, hash_hex: str) -> tuple[int, int, int, int, int] | None:
    """Read the durably persisted verification-receipt fingerprint, if any."""
    row = conn.execute(
        """
        SELECT st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns
        FROM verified_blob_receipts WHERE blob_hash = ?
        """,
        (bytes.fromhex(hash_hex),),
    ).fetchone()
    if row is None:
        return None
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))


def _record_blob_receipt(conn: sqlite3.Connection, hash_hex: str, fingerprint: tuple[int, int, int, int, int]) -> None:
    """Upsert the verification receipt for *hash_hex* to its current fingerprint."""
    st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns = fingerprint
    conn.execute(
        """
        INSERT INTO verified_blob_receipts
            (blob_hash, st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns, verified_at_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(blob_hash) DO UPDATE SET
            st_dev = excluded.st_dev,
            st_ino = excluded.st_ino,
            st_size = excluded.st_size,
            st_mtime_ns = excluded.st_mtime_ns,
            st_ctime_ns = excluded.st_ctime_ns,
            verified_at_ms = excluded.verified_at_ms
        """,
        (bytes.fromhex(hash_hex), st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns, int(time.time() * 1000)),
    )


def _verified_blob_bytes(conn: sqlite3.Connection, blob_store: BlobStore, hash_hex: str) -> bool:
    """Hash once per stable on-disk inode state, then reuse the durable receipt.

    polylogue-byw3y: this used to cache verification in a process-lifetime
    dict, so every daemon restart re-hashed every accepted frontier blob from
    scratch even when nothing had changed -- tens of GiB of wasted reads given
    the corpus's size skew (top 5% of raws = 69% of bytes, polylogue-el374).
    Receipts now persist in ``verified_blob_receipts`` (source.db), so a blob
    verified in a prior census stays trusted across restarts.

    Safety invariant (deliberately conservative -- this matters more than the
    performance win): a receipt is trusted ONLY when every stat() field
    matches the persisted fingerprint exactly. Any mismatch -- a changed
    size/mtime/ctime, a different inode, or no receipt at all -- forces a
    fresh ``blob_store.verify()`` re-hash; a stale receipt is never partially
    trusted. On a successful fresh verify, the receipt is rewritten to the
    blob's current fingerprint so the next census can trust it.
    """
    path = blob_store.blob_path(hash_hex)
    try:
        stat = path.stat()
    except OSError:
        return False
    fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if _blob_receipt_fingerprint(conn, hash_hex) == fingerprint:
        return True
    if not blob_store.verify(hash_hex):
        return False
    _record_blob_receipt(conn, hash_hex, fingerprint)
    return True


def _frontier_rows(conn: sqlite3.Connection) -> list[dict[str, object]]:
    return _rows(
        conn.execute(
            """
            SELECT h.logical_source_key, h.session_id,
                   COALESCE(s.raw_id, h.accepted_raw_id) AS accepted_raw_id,
                   h.accepted_raw_id AS head_accepted_raw_id,
                   h.accepted_source_revision,
                   COALESCE(hex(s.content_hash), hex(h.accepted_content_hash)) AS accepted_content_hash,
                   h.accepted_frontier_kind, h.accepted_frontier,
                   h.decided_at_ms AS head_decided_at_ms,
                   s.origin AS session_origin, s.raw_id AS session_raw_id,
                   hex(s.content_hash) AS session_content_hash,
                   s.message_count,
                   r.origin AS raw_origin, r.capture_mode, r.native_id,
                   r.source_path, r.source_index, hex(r.blob_hash) AS blob_hash,
                   r.blob_size, r.logical_source_key AS raw_logical_source_key,
                   r.revision_kind, r.source_revision, r.predecessor_raw_id,
                   r.baseline_raw_id, r.append_start_offset, r.append_end_offset,
                   r.acquisition_generation, r.revision_authority
            FROM index_tier.raw_revision_heads AS h
            LEFT JOIN index_tier.sessions AS s ON s.session_id = h.session_id
            LEFT JOIN raw_sessions AS r ON r.raw_id = COALESCE(s.raw_id, h.accepted_raw_id)
            ORDER BY h.logical_source_key
            """
        )
    )


def _item(
    *,
    state: RawAuthorityFrontierState,
    row: dict[str, object],
    reason: str,
) -> RawAuthorityFrontierItem:
    raw_id = str(row["accepted_raw_id"])
    source = json_document(
        {
            key: row.get(key)
            for key in (
                "raw_origin",
                "capture_mode",
                "native_id",
                "source_path",
                "source_index",
                "blob_hash",
                "blob_size",
                "raw_logical_source_key",
                "revision_kind",
                "source_revision",
                "predecessor_raw_id",
                "baseline_raw_id",
                "append_start_offset",
                "append_end_offset",
                "acquisition_generation",
                "revision_authority",
            )
        }
    )
    index = json_document(
        {
            key: row.get(key)
            for key in (
                "logical_source_key",
                "session_id",
                "accepted_raw_id",
                "head_accepted_raw_id",
                "accepted_source_revision",
                "accepted_content_hash",
                "accepted_frontier_kind",
                "accepted_frontier",
                "head_decided_at_ms",
                "session_origin",
                "session_raw_id",
                "session_content_hash",
                "message_count",
            )
        }
    )
    ids = (raw_id,)
    evidence = {
        "schema": "polylogue.raw-authority-frontier-evidence.v1",
        "state": state.value,
        "input_raw_ids": ids,
        "source": source,
        "index": index,
    }
    evidence_digest = _digest(evidence)
    plan_id = f"raw-authority-frontier:{evidence_digest}"
    return RawAuthorityFrontierItem(
        state=state,
        raw_id=raw_id,
        logical_source_key=(str(row["logical_source_key"]) if row.get("logical_source_key") is not None else None),
        session_id=(str(row["session_id"]) if row.get("session_id") is not None else None),
        reason=reason,
        evidence_digest=evidence_digest,
        input_raw_ids=ids,
        source_preconditions=source,
        index_preconditions=index,
        plan_id=plan_id,
    )


def _classify_frontier(
    conn: sqlite3.Connection,
    blob_store: BlobStore,
    row: dict[str, object],
) -> RawAuthorityFrontierItem:
    """Classify one accepted head against its own durable evidence.

    Every non-``PROVEN_CURRENT`` outcome is a statement about evidence, never
    a scheduled remedy. ``MISSING_BYTES_REACQUIRE`` is the one retryable
    state, and what retries it is ordinary acquisition, not this census.
    """
    raw_id = str(row["accepted_raw_id"])
    if row.get("raw_origin") is None:
        return _item(
            state=RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE,
            row=row,
            reason="accepted head raw is absent from the durable source tier",
        )
    blob_hash = str(row["blob_hash"]).lower()
    blob_exists = blob_store.exists(blob_hash)
    reacquisition_proven = blob_exists and _verified_blob_bytes(conn, blob_store, blob_hash)
    if not blob_exists or not reacquisition_proven:
        return _item(
            state=RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE,
            row=row,
            reason="accepted head raw bytes do not prove the expected content-addressed digest",
        )
    if row.get("session_id") is None or row.get("session_origin") is None:
        return _item(
            state=RawAuthorityFrontierState.CORRUPT,
            row=row,
            reason="accepted head has no matching materialized session",
        )
    if row.get("session_raw_id") != raw_id or row.get("session_content_hash") != row.get("accepted_content_hash"):
        return _item(
            state=RawAuthorityFrontierState.CORRUPT,
            row=row,
            reason="accepted head and materialized session authority disagree",
        )
    if row.get("head_accepted_raw_id") != raw_id:
        return _item(
            state=RawAuthorityFrontierState.CORRUPT,
            row=row,
            reason="accepted revision head and materialized session select different raw authority",
        )
    if row.get("session_origin") != row.get("raw_origin"):
        return _item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            row=row,
            reason="materialized session origin and durable raw origin disagree",
        )
    if row.get("revision_authority") == "quarantined":
        return _item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            row=row,
            reason="accepted raw authority remains quarantined",
        )
    if row.get("raw_logical_source_key") != row.get("logical_source_key"):
        return _item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            row=row,
            reason="accepted raw and index head logical authority keys disagree",
        )
    return _item(
        state=RawAuthorityFrontierState.PROVEN_CURRENT,
        row=row,
        reason="accepted source bytes, identity, head, and materialized session agree",
    )


#: States that publish a durable ``raw_authority_blockers`` obligation. A
#: later pass that disproves the state tombstones its own row; nothing else
#: clears one automatically.
_OBLIGATION_STATES = {
    RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE,
    RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
    RawAuthorityFrontierState.CORRUPT,
}


def _reconcile_frontier_obligations(
    config: Config,
    pass_id: str,
    items: tuple[RawAuthorityFrontierItem, ...],
) -> dict[str, str]:
    """Publish current obligations and close only those a later pass disproved.

    Returns the durable blocker id published for each still-blocking plan, so
    the caller can bind every blocking item to the row that now carries its
    evidence.
    """
    root = _archive_root(config)
    now = int(time.time() * 1000)
    blocking = tuple(item for item in items if item.state in _OBLIGATION_STATES)
    current_ids = {item.plan_id for item in blocking}
    published: dict[str, str] = {}
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        for item in blocking:
            blocker_id = f"raw-authority-blocker:{_digest(['frontier', pass_id, item.plan_id])}"
            published[item.plan_id] = blocker_id
            observed = {
                "schema": "polylogue.raw-authority-frontier-obligation.v1",
                # Writer-declared blocker class; automatic clearing selects on
                # this positively (polylogue-l8tdh) and never clears a frontier
                # obligation.
                BLOCKER_ORIGIN_KEY: BLOCKER_ORIGIN_FRONTIER_OBLIGATION,
                "state": item.state.value,
                "reason": item.reason,
                "evidence_digest": item.evidence_digest,
            }
            conn.execute(
                """
                INSERT INTO raw_authority_blockers (
                    blocker_id, plan_input_digest, observed_pass_id, reason, expected_json,
                    observed_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    blocker_id,
                    _plan(item).input_digest,
                    pass_id,
                    item.reason,
                    _canonical_json(_plan(item).to_dict()),
                    _canonical_json(observed),
                    now,
                ),
            )
        open_rows = conn.execute(
            """
            SELECT b.blocker_id, json_extract(b.expected_json, '$.plan_id')
            FROM raw_authority_blockers AS b
            WHERE b.resolved_at_ms IS NULL
              AND json_extract(b.expected_json, '$.authority_witness.schema') =
                  'polylogue.raw-authority-frontier-plan.v1'
            """
        ).fetchall()
        for blocker_id, plan_id in open_rows:
            plan_id_text = str(plan_id)
            if plan_id_text in current_ids:
                continue
            conn.execute(
                """
                UPDATE raw_authority_blockers
                SET resolved_at_ms = ?, resolution = ?
                WHERE blocker_id = ? AND resolved_at_ms IS NULL
                """,
                (
                    now,
                    _canonical_json(
                        {
                            "schema": "polylogue.raw-authority-obligation-resolution.v1",
                            "reason": "a later complete frontier pass disproved the prior blocking state",
                            "successor_pass_id": pass_id,
                        }
                    ),
                    blocker_id,
                ),
            )
    return published


def _terminal_superseded_items(conn: sqlite3.Connection) -> list[RawAuthorityFrontierItem]:
    rows = _rows(
        conn.execute(
            """
            SELECT a.logical_source_key, a.session_id, a.raw_id AS accepted_raw_id,
                   a.source_revision AS accepted_source_revision,
                   hex(a.accepted_content_hash) AS accepted_content_hash,
                   NULL AS accepted_frontier_kind, NULL AS accepted_frontier,
                   a.decided_at_ms AS head_decided_at_ms,
                   s.origin AS session_origin, s.raw_id AS session_raw_id,
                   hex(s.content_hash) AS session_content_hash, s.message_count,
                   r.origin AS raw_origin, r.capture_mode, r.native_id,
                   r.source_path, r.source_index, hex(r.blob_hash) AS blob_hash,
                   r.blob_size, r.logical_source_key AS raw_logical_source_key,
                   r.revision_kind, r.source_revision, r.predecessor_raw_id,
                   r.baseline_raw_id, r.append_start_offset, r.append_end_offset,
                   r.acquisition_generation, r.revision_authority
            FROM index_tier.raw_revision_applications AS a
            JOIN raw_sessions AS r ON r.raw_id = a.raw_id
            LEFT JOIN index_tier.sessions AS s ON s.session_id = a.session_id
            LEFT JOIN index_tier.raw_revision_heads AS h ON h.accepted_raw_id = a.raw_id
            WHERE a.decision = 'superseded' AND h.accepted_raw_id IS NULL
            ORDER BY a.raw_id, a.logical_source_key
            """
        )
    )
    return [
        _item(
            state=RawAuthorityFrontierState.SUPERSEDED,
            row=row,
            reason="durable application receipt terminally supersedes this retained snapshot",
        )
        for row in rows
    ]


def _plan(item: RawAuthorityFrontierItem) -> RawReplayPlan:
    witness = json_document(
        {
            "schema": "polylogue.raw-authority-frontier-plan.v1",
            "state": item.state.value,
            "reason": item.reason,
            "evidence_digest": item.evidence_digest,
        }
    )
    return RawReplayPlan(
        plan_id=item.plan_id,
        input_digest=item.evidence_digest,
        input_raw_ids=item.input_raw_ids,
        logical_keys=((item.logical_source_key,) if item.logical_source_key is not None else ()),
        authority_witness=witness,
        source_preconditions=item.source_preconditions,
        index_preconditions=item.index_preconditions,
    )


def _frontier_items(config: Config) -> tuple[tuple[RawAuthorityFrontierItem, ...], int, int]:
    root = _archive_root(config)
    source_db = root / "source.db"
    index_db = config.current_db_path()
    if not source_db.is_file() or not index_db.is_file():
        raise RuntimeError("raw authority frontier census requires initialized source and index tiers")
    with closing(sqlite3.connect(source_db)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
        head_rows = _frontier_rows(conn)
        # _classify_frontier may persist a verified-blob receipt (polylogue-byw3y)
        # through this same connection; the outer ``conn`` context manager commits
        # those writes on clean exit (or rolls back on exception), so a receipt is
        # never durably recorded for bytes this pass didn't finish inspecting.
        blob_store = BlobStore(root / "blob")
        head_items = [_classify_frontier(conn, blob_store, row) for row in head_rows]
        superseded_items = _terminal_superseded_items(conn)
    all_items = (*head_items, *superseded_items)
    return (
        tuple(sorted(all_items, key=lambda item: (item.raw_id, item.plan_id))),
        len(head_items),
        len(superseded_items),
    )


def inspect_raw_authority_frontier(config: Config) -> RawAuthorityFrontierCensus:
    """Inspect the complete accepted frontier and publish its durable obligations.

    The inspection itself is not recorded: a pass that observes an unchanged
    frontier writes nothing, and its ``pass_id`` is a content address over the
    inspected inventory rather than a new ledger row (polylogue-6kur ruling
    2026-09-15). What it does publish is durable -- every blocking item gets a
    ``raw_authority_blockers`` row, and an obligation the current evidence
    disproves is tombstoned -- so this is not a read operation: offline callers
    need the same daemon exclusion boundary as an apply, and daemon convergence
    is admitted through its active write lease.
    """
    from polylogue.maintenance.offline_guard import offline_maintenance_block_reason

    block_reason = offline_maintenance_block_reason(config, active=True, dry_run=False)
    if block_reason is not None:
        raise RuntimeError(block_reason)
    all_items, accepted_head_count, terminal_superseded_count = _frontier_items(config)
    state_counts_counter = Counter(item.state.value for item in all_items)
    state_counts = json_document(dict(sorted(state_counts_counter.items())))
    inventory_digest = _digest([item.to_dict() for item in all_items])
    gap_items = tuple(item for item in all_items if item.state is not RawAuthorityFrontierState.PROVEN_CURRENT)
    plans = tuple(_plan(item) for item in gap_items)
    plan_inventory_digest = _digest([plan.to_dict() for plan in plans])
    pass_id = f"raw-authority-frontier-pass:{inventory_digest}"
    published = _reconcile_frontier_obligations(config, pass_id, all_items)
    bound_items = tuple(dataclasses.replace(item, evidence_ref=published.get(item.plan_id)) for item in all_items)
    return RawAuthorityFrontierCensus(
        pass_id=pass_id,
        inventory_digest=inventory_digest,
        plan_inventory_digest=plan_inventory_digest,
        state_counts=state_counts,
        accepted_head_count=accepted_head_count,
        terminal_superseded_count=terminal_superseded_count,
        plan_count=len(plans),
        items=bound_items,
    )


__all__ = [
    "RawAuthorityFrontierCensus",
    "RawAuthorityFrontierItem",
    "RawAuthorityFrontierState",
    "inspect_raw_authority_frontier",
]
