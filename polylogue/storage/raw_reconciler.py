"""Proof-driven census for every accepted raw-authority frontier.

This module owns the provider-neutral state machine. Historical incident
actuators remain implementation strategies in :mod:`polylogue.storage.raw_convergence`;
they do not get to define separate public notions of plan identity, evidence,
or readiness.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

from polylogue.config import Config
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import WARNING, emit, get_logger
from polylogue.storage.archive_identity import archive_file_set_root
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import (
    BLOCKER_ORIGIN_FRONTIER_OBLIGATION,
    BLOCKER_ORIGIN_KEY,
    RawReplayPlan,
)
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.storage.raw_convergence import (
        BrowserCaptureOriginRepairItem,
        DuplicateRawIdentityRepairItem,
        QuarantinedAcceptedRawRepairItem,
    )


class RawAuthorityFrontierState(StrEnum):
    """Mutually exclusive authority states for one accepted frontier."""

    PROVEN_CURRENT = "proven_current"
    SAFELY_REKEYABLE = "safely_rekeyable"
    DUPLICATE_ALIAS = "duplicate_alias"
    SUPERSEDED = "superseded"
    MISSING_BYTES_REACQUIRE = "missing_bytes_reacquire"
    CONFLICTING_AUTHORITY_NEEDS_JUDGMENT = "conflicting_authority_needs_judgment"
    UNRESOLVED_PROVENANCE = "unresolved_provenance"
    CORRUPT = "corrupt"


class RawAuthorityActuator(StrEnum):
    """Strategies admitted behind the shared plan/apply/postflight contract."""

    NONE = "none"
    REPLAY = "raw_revision_replay"
    REFINE_QUARANTINE = "refine_quarantined_raw"
    COPY_FORWARD_ORIGIN = "copy_forward_origin"
    FOLD_DUPLICATE_ALIAS = "fold_duplicate_alias"
    REACQUIRE = "reacquire"
    REQUEST_JUDGMENT = "request_judgment"
    RESOLVE_CONFLICT = "resolve_conflict"


_EXECUTABLE_STATES = {
    RawAuthorityFrontierState.SAFELY_REKEYABLE,
    RawAuthorityFrontierState.DUPLICATE_ALIAS,
}

#: polylogue-w32w: actuators with a real apply() dispatch branch in
#: ``_apply_raw_authority_plan`` -- i.e. actuators that promise "something
#: automatically executes this". Must mirror exactly the
#: ``if item.actuator is RawAuthorityActuator.<X>:`` branches there
#: (``test_apply_dispatched_actuators_match_apply_branches`` in
#: ``tests/unit/storage/test_raw_authority_ledger.py`` fails if this drifts).
#: REACQUIRE and REQUEST_JUDGMENT are deliberately excluded: they resolve
#: out-of-band (ordinary ingest re-acquisition; an operator judgment
#: assertion promoted by ``_apply_judgment_dispositions``), not through this
#: apply dispatcher, so a non-executable state pairing with them is not the
#: defect class this guards against.
_APPLY_DISPATCHED_ACTUATORS = frozenset(
    {
        RawAuthorityActuator.RESOLVE_CONFLICT,
        RawAuthorityActuator.FOLD_DUPLICATE_ALIAS,
        RawAuthorityActuator.COPY_FORWARD_ORIGIN,
        RawAuthorityActuator.REFINE_QUARANTINE,
    }
)

_ChunkT = TypeVar("_ChunkT")

_QUARANTINE_OVERRIDE_KEY_SEP = "\x00"


def _quarantine_override_key(raw_id: str, logical_source_key: str) -> str:
    """Session-scoped override key (polylogue-zaiz): distinct from bare raw_id keys.

    Browser-origin/conflict overrides stay keyed by bare raw_id (those
    proofs are properties of the raw itself, not per-session) -- only
    quarantine-refinement needs per-session scoping, so this uses a
    reserved separator no real raw_id/logical_source_key can contain
    (raw_ids are hex digests; logical_source_keys are colon-joined
    provider identifiers) to guarantee it never collides with a bare
    raw_id key.
    """
    return f"{raw_id}{_QUARANTINE_OVERRIDE_KEY_SEP}{logical_source_key}"


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
    actuator: RawAuthorityActuator
    raw_id: str
    logical_source_key: str | None
    session_id: str | None
    reason: str
    evidence_digest: str
    input_raw_ids: tuple[str, ...]
    source_preconditions: JSONDocument
    index_preconditions: JSONDocument
    strategy_witness: JSONDocument
    plan_id: str
    evidence_ref: str | None = None

    def __post_init__(self) -> None:
        # polylogue-w32w: the invariant this class enforces at construction
        # -- "every frontier state must have an actuator the executability
        # gate can admit" -- reframed as its exact contrapositive: an
        # actuator that HAS a real apply() handler must only ever be paired
        # with an executable state. polylogue-u19l was precisely a
        # violation of this: REFINE_QUARANTINE (a dispatched actuator) was
        # being assigned to UNRESOLVED_PROVENANCE (a non-executable state),
        # so daemon convergence could never select it -- 4,147 blockers
        # accumulated behind an actuator that
        # was structurally unreachable through every path that exists. This
        # makes that exact shape impossible to construct, not merely
        # undocumented.
        if self.actuator in _APPLY_DISPATCHED_ACTUATORS and self.state not in _EXECUTABLE_STATES:
            raise ValueError(
                f"raw-authority frontier item is unreachable: actuator {self.actuator.value!r} has an apply() "
                f"dispatch branch but state {self.state.value!r} is not in the executability gate "
                "(_EXECUTABLE_STATES) -- daemon convergence would never select this item for apply "
                "(polylogue-u19l/polylogue-w32w)"
            )

    def to_dict(self) -> JSONDocument:
        return json_document(dataclasses.asdict(self))

    @property
    def executable(self) -> bool:
        return self.state in _EXECUTABLE_STATES


@dataclass(frozen=True, slots=True)
class _StrategyOverride:
    state: RawAuthorityFrontierState
    actuator: RawAuthorityActuator
    reason: str
    witness: JSONDocument
    input_raw_ids: tuple[str, ...]


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
    executable_plan_count: int
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
                "executable_plan_count": self.executable_plan_count,
                "returned_count": len(sample),
                "items_truncated": len(sample) < len(self.items),
                "items": [item.to_dict() for item in sample],
            }
        )


def _archive_root(config: Config) -> Path:
    """Return the archive file-set root housing the currently active database.

    Deliberately follows ``config.db_path`` (not ``config.archive_root``),
    matching :func:`polylogue.storage.raw_convergence._raw_materialization_archive_root`:
    this reconciler inspects the database and blob store that are actually
    live right now, which ``config.db_path`` already resolves correctly
    (``.index-active-pointer``-aware, or an explicit override) inside
    ``Config.__init__``.
    """
    return archive_file_set_root(archive_root=config.archive_root, db_path=config.db_path)


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, object]]:
    names = tuple(column[0] for column in cursor.description or ())
    return [{name: _json_value(value) for name, value in zip(names, row, strict=True)} for row in cursor.fetchall()]


def _chunks(values: Sequence[_ChunkT], size: int = 100) -> Iterator[list[_ChunkT]]:
    """Yield bounded strategy-proof requests in deterministic order."""
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


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


def _browser_strategy_witness(item: BrowserCaptureOriginRepairItem) -> JSONDocument:
    from polylogue.storage.raw_convergence import _browser_origin_item_payload

    return json_document(
        {
            "schema": "polylogue.raw-authority-strategy-witness.v1",
            "kind": "browser_origin",
            "item": _browser_origin_item_payload(item),
        }
    )


def _quarantine_strategy_witness(item: QuarantinedAcceptedRawRepairItem) -> JSONDocument:
    payload = {
        key: _json_value(value)
        for key, value in dataclasses.asdict(item).items()
        if key not in {"proof_digest", "reason", "repaired", "status"}
    }
    return json_document(
        {
            "schema": "polylogue.raw-authority-strategy-witness.v1",
            "kind": "quarantine_refinement",
            "item": payload,
        }
    )


def _duplicate_strategy_witness(item: DuplicateRawIdentityRepairItem) -> JSONDocument:
    from polylogue.storage.raw_convergence import _duplicate_raw_identity_proof_digest

    return json_document(
        {
            "schema": "polylogue.raw-authority-strategy-witness.v1",
            "kind": "duplicate_alias",
            "proof_digest": _duplicate_raw_identity_proof_digest(item),
            "stale_raw_id": item.stale_raw_id,
            "canonical_raw_id": item.canonical_raw_id,
            "session_id": item.session_id,
            "logical_source_key": item.logical_source_key,
            "accepted_source_revision": item.accepted_source_revision,
            "accepted_content_hash": item.accepted_content_hash,
            "accepted_frontier_kind": item.accepted_frontier_kind,
            "accepted_frontier": item.accepted_frontier,
            "accepted_decided_at_ms": item.accepted_decided_at_ms,
        }
    )


def _browser_strategy_raw_ids(item: BrowserCaptureOriginRepairItem) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                raw_id
                for raw_id in (
                    item.raw_id,
                    item.replacement_raw_id,
                    item.copy_forward_raw_id,
                    item.semantic_canonical_raw_id,
                    *item.semantic_historical_raw_ids,
                )
                if raw_id is not None
            }
        )
    )


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


def _duplicate_alias_siblings(conn: sqlite3.Connection, row: dict[str, object]) -> tuple[str, ...]:
    if row.get("raw_origin") is None or row.get("blob_hash") is None or row.get("native_id") is None:
        return ()
    blob_hash = bytes.fromhex(cast(str, row["blob_hash"]))
    expected_accepted = deterministic_raw_session_id(
        str(row["raw_origin"]),
        str(row["source_path"]),
        int(cast(int, row["source_index"])),
        blob_hash,
        native_id=str(row["native_id"]),
    )
    if expected_accepted != row["accepted_raw_id"]:
        return ()
    siblings = conn.execute(
        """
        SELECT raw_id
        FROM raw_sessions
        WHERE origin = ? AND source_path = ? AND source_index = ?
          AND blob_hash = ? AND native_id IS NULL AND raw_id != ?
          AND NOT EXISTS (
              SELECT 1 FROM index_tier.raw_revision_heads AS h
              WHERE h.accepted_raw_id = raw_sessions.raw_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM index_tier.sessions AS s
              WHERE s.raw_id = raw_sessions.raw_id
          )
        ORDER BY raw_id
        """,
        (
            row["raw_origin"],
            row["source_path"],
            row["source_index"],
            blob_hash,
            row["accepted_raw_id"],
        ),
    ).fetchall()
    expected_canonical = deterministic_raw_session_id(
        str(row["raw_origin"]),
        str(row["source_path"]),
        int(cast(int, row["source_index"])),
        blob_hash,
        native_id=None,
    )
    return tuple(str(sibling[0]) for sibling in siblings if str(sibling[0]) == expected_canonical)


def _item(
    *,
    state: RawAuthorityFrontierState,
    actuator: RawAuthorityActuator,
    row: dict[str, object],
    reason: str,
    input_raw_ids: tuple[str, ...] | None = None,
    strategy_witness: JSONDocument | None = None,
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
    ids = tuple(sorted(set(input_raw_ids or (raw_id,))))
    evidence = {
        "schema": "polylogue.raw-authority-frontier-evidence.v1",
        "state": state.value,
        "actuator": actuator.value,
        "input_raw_ids": ids,
        "source": source,
        "index": index,
        "strategy_witness": strategy_witness or {},
    }
    evidence_digest = _digest(evidence)
    plan_id = f"raw-authority-frontier:{evidence_digest}"
    return RawAuthorityFrontierItem(
        state=state,
        actuator=actuator,
        raw_id=raw_id,
        logical_source_key=(str(row["logical_source_key"]) if row.get("logical_source_key") is not None else None),
        session_id=(str(row["session_id"]) if row.get("session_id") is not None else None),
        reason=reason,
        evidence_digest=evidence_digest,
        input_raw_ids=ids,
        source_preconditions=source,
        index_preconditions=index,
        strategy_witness=strategy_witness or json_document({}),
        plan_id=plan_id,
    )


def _classify_frontier(
    conn: sqlite3.Connection,
    blob_store: BlobStore,
    index_db: Path,
    row: dict[str, object],
    strategy_override: _StrategyOverride | None,
) -> RawAuthorityFrontierItem:
    raw_id = str(row["accepted_raw_id"])
    if row.get("raw_origin") is None:
        return _item(
            state=RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE,
            actuator=RawAuthorityActuator.REACQUIRE,
            row=row,
            reason="accepted head raw is absent from the durable source tier",
        )
    blob_hash = str(row["blob_hash"]).lower()
    blob_exists = blob_store.exists(blob_hash)
    reacquisition_proven = blob_exists and _verified_blob_bytes(conn, blob_store, blob_hash)
    if not blob_exists or not reacquisition_proven:
        return _item(
            state=RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE,
            actuator=RawAuthorityActuator.REACQUIRE,
            row=row,
            reason="accepted head raw bytes do not prove the expected content-addressed digest",
        )
    if row.get("session_id") is None or row.get("session_origin") is None:
        return _item(
            state=RawAuthorityFrontierState.CORRUPT,
            actuator=RawAuthorityActuator.NONE,
            row=row,
            reason="accepted head has no matching materialized session",
        )
    if row.get("session_raw_id") != raw_id or row.get("session_content_hash") != row.get("accepted_content_hash"):
        return _item(
            state=RawAuthorityFrontierState.CORRUPT,
            actuator=RawAuthorityActuator.NONE,
            row=row,
            reason="accepted head and materialized session authority disagree",
        )
    duplicate_siblings = _duplicate_alias_siblings(conn, row)
    if duplicate_siblings and row.get("native_id") is not None:
        from polylogue.storage.raw_convergence import _inspect_duplicate_raw_identity

        if len(duplicate_siblings) != 1:
            raise RuntimeError(f"duplicate alias classification is not injective for {raw_id}")
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as proof_conn:
            proof_conn.row_factory = sqlite3.Row
            proof_conn.execute(
                "ATTACH DATABASE ? AS source",
                (f"file:{blob_store.root.parent / 'source.db'}?mode=ro",),
            )
            duplicate_item = _inspect_duplicate_raw_identity(
                proof_conn,
                blob_store.root.parent,
                raw_id,
                duplicate_siblings[0],
                str(row["logical_source_key"]),
            )
        if duplicate_item.status == "ineligible":
            # polylogue-dmvo: a legitimate N:1 fan-out terminal state, not a
            # proof violation. Several sessions can share one stale
            # native-id-inclusive raw as their accepted head (forked/
            # subagent/resumed sessions replaying the same parent JSONL,
            # polylogue-ihc8); only ONE of them can ever fold onto the
            # single available canonical twin. Once that fold lands, every
            # other sibling's own re-inspection legitimately (and by
            # design) returns "ineligible" -- e.g. "canonical raw is
            # already an accepted head" -- from
            # ``_inspect_duplicate_raw_identity``, which never raises
            # itself. Treating that as fatal here previously crashed the
            # *entire* frontier census (every other raw's classification
            # blocked behind one RuntimeError, observed live holding the
            # writer lock for 9+ minutes before failing all queued work).
            # Classify it as a benign, non-executable terminal state
            # instead so this session's own row is skipped while every
            # other row's classification proceeds unaffected.
            return _item(
                state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
                actuator=RawAuthorityActuator.NONE,
                row=row,
                reason=f"duplicate alias fold is not eligible for this session: {duplicate_item.reason}",
            )
        if duplicate_item.status not in {"eligible", "already_repaired"}:
            raise RuntimeError(f"duplicate alias lacks an exact strategy proof: {duplicate_item.reason}")
        duplicate_witness = _duplicate_strategy_witness(duplicate_item)
        return _item(
            state=RawAuthorityFrontierState.DUPLICATE_ALIAS,
            actuator=RawAuthorityActuator.FOLD_DUPLICATE_ALIAS,
            row=row,
            reason="accepted raw uses the obsolete native-id-inclusive identity while an exact canonical twin exists",
            input_raw_ids=(raw_id, *duplicate_siblings),
            strategy_witness=duplicate_witness,
        )
    if strategy_override is not None:
        return _item(
            state=strategy_override.state,
            actuator=strategy_override.actuator,
            row=row,
            reason=strategy_override.reason,
            strategy_witness=strategy_override.witness,
            input_raw_ids=strategy_override.input_raw_ids,
        )
    if row.get("head_accepted_raw_id") != raw_id:
        return _item(
            state=RawAuthorityFrontierState.CORRUPT,
            actuator=RawAuthorityActuator.NONE,
            row=row,
            reason="accepted revision head and materialized session select different raw authority",
        )
    if row.get("session_origin") != row.get("raw_origin"):
        return _item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            actuator=RawAuthorityActuator.NONE,
            row=row,
            reason="origin mismatch lacks a strategy proof admitted by the shared reconciler",
        )
    if row.get("revision_authority") == "quarantined":
        # polylogue-u19l/w32w: reachable only when ``logical_source_key`` is
        # missing (so ``_strategy_overrides`` never had a key to inspect
        # under) -- every other quarantined row is now given an explicit
        # eligible-or-ineligible override below, and REFINE_QUARANTINE is
        # never assigned here because that actuator has an apply() dispatch
        # branch that only ever selects SAFELY_REKEYABLE items
        # (``_EXECUTABLE_STATES``); promising it for a non-executable state
        # is exactly the absorbing-state defect this bead fixed (4,147
        # blockers, fixed_point=0 on all 256 retained censuses, gap count
        # that never shrank). Actuator NONE here is an honest "nothing will
        # execute this automatically", not a broken promise.
        return _item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            actuator=RawAuthorityActuator.NONE,
            row=row,
            reason="accepted raw authority remains quarantined and has no logical source key to refine against",
        )
    if row.get("raw_logical_source_key") != row.get("logical_source_key"):
        # polylogue-w32w: REPLAY has no apply() dispatch branch at all (grep
        # confirms it), so -- like the REFINE_QUARANTINE case above -- it can
        # never be selected by the executability gate. Mirror the same fix:
        # actuator NONE, not a promise nothing discharges.
        return _item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            actuator=RawAuthorityActuator.NONE,
            row=row,
            reason="accepted raw and index head logical authority keys disagree",
        )
    return _item(
        state=RawAuthorityFrontierState.PROVEN_CURRENT,
        actuator=RawAuthorityActuator.NONE,
        row=row,
        reason="accepted source bytes, identity, head, and materialized session agree",
    )


def _strategy_overrides(
    config: Config,
    rows: list[dict[str, object]],
    *,
    index_db_path: Path,
) -> dict[str, _StrategyOverride]:
    """Ask legacy incident inspectors for proofs, never for plan identity."""
    from polylogue.storage.raw_convergence import (
        BROWSER_ORIGIN_READ_FAILED_STATUS,
        inspect_browser_canonical_authority_conflicts,
        inspect_browser_capture_origin_mismatches,
        inspect_quarantined_accepted_raws,
    )

    overrides: dict[str, _StrategyOverride] = {}
    # polylogue-roaof: raws whose durable evidence could not be read. Nothing
    # was proven about them, so no override -- conflict or otherwise -- may be
    # derived; the next pass retries once the blob is readable again.
    unread_evidence_ids: set[str] = set()
    browser_ids = sorted(
        {
            str(row["accepted_raw_id"])
            for row in rows
            if row.get("raw_origin") is not None and row.get("session_origin") != row.get("raw_origin")
        }
    )
    for browser_chunk in _chunks(browser_ids):
        browser_items = inspect_browser_capture_origin_mismatches(
            config,
            browser_chunk,
            index_db_path=index_db_path,
        )
        for browser_item in browser_items:
            if browser_item.status in {"eligible", "already_repaired"}:
                overrides[browser_item.raw_id] = _StrategyOverride(
                    state=RawAuthorityFrontierState.SAFELY_REKEYABLE,
                    actuator=RawAuthorityActuator.COPY_FORWARD_ORIGIN,
                    reason="browser-origin strategy proved an exact evidence-preserving copy-forward",
                    witness=_browser_strategy_witness(browser_item),
                    input_raw_ids=_browser_strategy_raw_ids(browser_item),
                )
            elif browser_item.status == BROWSER_ORIGIN_READ_FAILED_STATUS:
                unread_evidence_ids.add(browser_item.raw_id)
                emit(
                    "storage.raw_reconciler.browser_origin_evidence_unreadable",
                    level=WARNING,
                    outcome="degraded",
                    raw_id=browser_item.raw_id,
                    reason=browser_item.reason,
                )
            elif browser_item.terminally_ineligible:
                overrides[browser_item.raw_id] = _StrategyOverride(
                    state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
                    actuator=RawAuthorityActuator.NONE,
                    reason=f"browser-origin strategy is terminally ineligible: {browser_item.reason}",
                    witness=json_document(
                        {
                            "schema": "polylogue.raw-authority-strategy-witness.v1",
                            "kind": "browser_origin_terminal_ineligible",
                            "raw_id": browser_item.raw_id,
                            "reason": browser_item.reason,
                        }
                    ),
                    input_raw_ids=(browser_item.raw_id,),
                )
        conflicts = inspect_browser_canonical_authority_conflicts(
            config,
            browser_chunk,
            index_db_path=index_db_path,
        )
        for conflict_item in conflicts.items:
            if conflict_item.raw_id in overrides:
                continue
            if conflict_item.raw_id in unread_evidence_ids:
                continue
            if conflict_item.status == BROWSER_ORIGIN_READ_FAILED_STATUS:
                # polylogue-roaof: the durable evidence was never read, so no
                # conflict was proven and no durable judgment row may be
                # derived from it. Leave the raw in its unrecorded state so the
                # next pass retries once the blob is readable again.
                emit(
                    "storage.raw_reconciler.browser_authority_evidence_unreadable",
                    level=WARNING,
                    outcome="degraded",
                    raw_id=conflict_item.raw_id,
                    reason=conflict_item.reason,
                )
                continue
            if conflict_item.competing_raw_id is None:
                overrides[conflict_item.raw_id] = _StrategyOverride(
                    state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
                    actuator=RawAuthorityActuator.NONE,
                    reason=(
                        "browser-origin evidence has a retained membership precondition but no "
                        "canonical authority that an operator could retain"
                    ),
                    witness=json_document(
                        {
                            "schema": "polylogue.raw-authority-strategy-witness.v1",
                            "kind": "browser_membership_precondition",
                            "evidence": dataclasses.asdict(conflict_item),
                        }
                    ),
                    input_raw_ids=(conflict_item.raw_id,),
                )
                continue
            overrides[conflict_item.raw_id] = _StrategyOverride(
                state=RawAuthorityFrontierState.CONFLICTING_AUTHORITY_NEEDS_JUDGMENT,
                actuator=RawAuthorityActuator.REQUEST_JUDGMENT,
                reason=conflict_item.reason,
                witness=json_document(
                    {
                        "schema": "polylogue.raw-authority-strategy-witness.v1",
                        "kind": "browser_conflict",
                        "evidence": dataclasses.asdict(conflict_item),
                    }
                ),
                input_raw_ids=tuple(sorted({conflict_item.raw_id, conflict_item.competing_raw_id})),
            )
    # (raw_id, logical_source_key) pairs, not bare raw_ids: a fan-out raw
    # shared by several sessions (polylogue-zaiz, mirroring polylogue-ihc8)
    # needs a proof scoped to each session, not one shared proof that
    # either raises "expected one accepted head, found N" for every
    # sibling or silently proves a witness against the wrong session's
    # head. Override keys below therefore include the logical_source_key.
    quarantine_pairs = sorted(
        {
            (str(row["accepted_raw_id"]), str(row["logical_source_key"]))
            for row in rows
            if row.get("revision_authority") == "quarantined"
            and str(row["accepted_raw_id"]) not in browser_ids
            and row.get("logical_source_key") is not None
        }
    )
    for quarantine_chunk in _chunks(quarantine_pairs, size=100):
        quarantine_items = inspect_quarantined_accepted_raws(
            config,
            quarantine_chunk,
            index_db_path=index_db_path,
        )
        for (raw_id, logical_source_key), quarantine_item in zip(quarantine_chunk, quarantine_items, strict=True):
            if quarantine_item.status in {"eligible", "already_repaired"}:
                overrides[_quarantine_override_key(raw_id, logical_source_key)] = _StrategyOverride(
                    state=RawAuthorityFrontierState.SAFELY_REKEYABLE,
                    actuator=RawAuthorityActuator.REFINE_QUARANTINE,
                    reason="quarantined-raw strategy proved exact accepted-byte and semantic authority",
                    witness=_quarantine_strategy_witness(quarantine_item),
                    input_raw_ids=tuple(sorted({quarantine_item.raw_id, *quarantine_item.census_stage_raw_ids})),
                )
            else:
                # polylogue-u19l: an "ineligible" proof here is a permanent
                # structural fact about this raw's own data (missing rows,
                # mismatched hashes, competing authority, an incompatible
                # typed envelope -- see every ``_quarantined_raw_item(...)``
                # return in ``_inspect_quarantined_accepted_raw``), not a
                # transient state waiting on a retry: nothing about this
                # raw's bytes or index rows changes on its own between
                # census cycles. Previously this branch registered no
                # override at all, so the row fell through to the
                # classifier's default REFINE_QUARANTINE assignment -- an
                # actuator with a real apply() handler that the
                # executability gate (``_EXECUTABLE_STATES``) can never
                # select, because the state stayed UNRESOLVED_PROVENANCE.
                # That is the audited absorbing state: 4,147 open
                # blockers all reading "pending exact refinement proof",
                # 15,205/17,384 frontier plans residual, fixed_point=0 on
                # every one of 256 retained censuses, and a gap count that
                # only ever grew (16,874 -> 17,384). Recording the real
                # ineligibility reason with actuator NONE makes this a
                # terminal, countable (state_counts), operator-visible
                # (raw_authority_blockers) fact instead of a false promise.
                overrides[_quarantine_override_key(raw_id, logical_source_key)] = _StrategyOverride(
                    state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
                    actuator=RawAuthorityActuator.NONE,
                    reason=f"quarantined-raw refinement strategy proved this raw ineligible: {quarantine_item.reason}",
                    witness=_quarantine_strategy_witness(quarantine_item),
                    input_raw_ids=(quarantine_item.raw_id,),
                )
    return overrides


_OBLIGATION_STATES = {
    RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE,
    RawAuthorityFrontierState.CONFLICTING_AUTHORITY_NEEDS_JUDGMENT,
    RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
    RawAuthorityFrontierState.CORRUPT,
}


def _record_judgment_candidate(config: Config, item: RawAuthorityFrontierItem, *, now_ms: int) -> tuple[str, bool]:
    """Persist the conflict as a non-authoritative candidate for operator judgment.

    polylogue-rjtv: the assertion id is derived from ``item.plan_id``, which is
    itself derived from a fresh evidence digest every census cycle -- a
    census cycle that re-encounters the *same* unresolved conflict (same
    ``raw_id``/``logical_source_key``) before an operator has judged it would
    otherwise mint a brand-new candidate each time, leaving prior cycles'
    still-pending duplicates to accumulate forever (found live 2026-07-27: 24
    candidates in ``judge --list`` for what was actually 6 real conflicts).
    Look up an existing still-``candidate`` request for the same conflict
    identity first and refresh it in place instead of minting a new one. This
    only dedupes pending-vs-pending; an already accepted/rejected/deferred
    assertion is untouched, so a fresh judgment can still be requested if the
    same conflict resurfaces after a prior disposition.
    """
    from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope, upsert_assertion

    root = _archive_root(config)
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
        pending_row = conn.execute(
            """
            SELECT assertion_id FROM assertions
            WHERE kind = 'judgment' AND status = 'candidate'
              AND json_extract(value_json, '$.raw_id') = ?
              AND json_extract(value_json, '$.logical_source_key') = ?
            LIMIT 1
            """,
            (item.raw_id, item.logical_source_key),
        ).fetchone()
        assertion_id = (
            str(pending_row[0])
            if pending_row is not None
            else f"judgment:{_digest(['raw-authority-frontier', item.plan_id])}"
        )
        existing = read_assertion_envelope(conn, assertion_id)
        if existing is not None and existing.status is not AssertionStatus.CANDIDATE:
            return existing.assertion_id, False
        upsert_assertion(
            conn,
            assertion_id=assertion_id,
            scope_ref="insight:raw-authority-frontier@v1",
            target_ref=f"session:{item.session_id}" if item.session_id is not None else f"raw:{item.raw_id}",
            key=item.plan_id,
            kind=AssertionKind.JUDGMENT,
            value={
                "schema": "polylogue.raw-authority-judgment-request.v1",
                "plan_id": item.plan_id,
                "state": item.state.value,
                "actuator": item.actuator.value,
                "raw_id": item.raw_id,
                "logical_source_key": item.logical_source_key,
                "evidence_digest": item.evidence_digest,
                "reason": item.reason,
                "supported_dispositions": ["retain_canonical_authority"],
            },
            body_text=item.reason,
            author_ref="insight:raw-authority-frontier@v1",
            author_kind="detector",
            status=AssertionStatus.CANDIDATE,
            visibility=AssertionVisibility.PRIVATE,
            context_policy={"inject": False, "promotion_required": True},
            now_ms=now_ms,
        )
    return assertion_id, False


def _apply_judgment_dispositions(
    config: Config,
    items: tuple[RawAuthorityFrontierItem, ...],
) -> tuple[RawAuthorityFrontierItem, ...]:
    """Promote explicitly resolved conflict plans into executable successors."""
    root = _archive_root(config)
    with closing(sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)) as conn:
        has_blockers = conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'raw_authority_blockers'"
        ).fetchone()
        if has_blockers is None:
            return items
        resolutions = {
            str(plan_id): json_document(json.loads(str(resolution)))
            for plan_id, resolution in conn.execute(
                """
                SELECT json_extract(expected_json, '$.plan_id'), resolution
                FROM raw_authority_blockers
                WHERE resolved_at_ms IS NOT NULL AND resolution IS NOT NULL
                ORDER BY resolved_at_ms
                """
            )
        }
    promoted: list[RawAuthorityFrontierItem] = []
    for item in items:
        resolution = resolutions.get(item.plan_id)
        disposition = None if resolution is None else resolution.get("judgment_disposition")
        if (
            item.state is not RawAuthorityFrontierState.CONFLICTING_AUTHORITY_NEEDS_JUDGMENT
            or disposition != "retain_canonical_authority"
        ):
            promoted.append(item)
            continue
        assert resolution is not None
        witness = json_document(
            {
                "schema": "polylogue.raw-authority-strategy-witness.v1",
                "kind": "browser_conflict_resolution",
                "conflict": item.strategy_witness,
                "judgment": {
                    "disposition": disposition,
                    "operator_assertion_id": resolution.get("operator_assertion_id"),
                    "superseded_plan_id": item.plan_id,
                },
            }
        )
        evidence = {
            "schema": "polylogue.raw-authority-frontier-evidence.v1",
            "state": RawAuthorityFrontierState.SAFELY_REKEYABLE.value,
            "actuator": RawAuthorityActuator.RESOLVE_CONFLICT.value,
            "input_raw_ids": item.input_raw_ids,
            "source": item.source_preconditions,
            "index": item.index_preconditions,
            "strategy_witness": witness,
        }
        evidence_digest = _digest(evidence)
        promoted.append(
            dataclasses.replace(
                item,
                state=RawAuthorityFrontierState.SAFELY_REKEYABLE,
                actuator=RawAuthorityActuator.RESOLVE_CONFLICT,
                reason="accepted operator judgment retained the exact canonical authority",
                evidence_digest=evidence_digest,
                strategy_witness=witness,
                plan_id=f"raw-authority-frontier:{evidence_digest}",
                evidence_ref=None,
            )
        )
    return tuple(promoted)


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
    judgment_results = {
        item.plan_id: _record_judgment_candidate(config, item, now_ms=now)
        for item in blocking
        if item.state is RawAuthorityFrontierState.CONFLICTING_AUTHORITY_NEEDS_JUDGMENT
    }
    judgment_refs = {plan_id: result[0] for plan_id, result in judgment_results.items()}
    judged_plan_ids = {plan_id for plan_id, result in judgment_results.items() if result[1]}
    current_ids = {item.plan_id for item in blocking} - judged_plan_ids
    published: dict[str, str] = {}
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        for item in blocking:
            if item.plan_id in judged_plan_ids:
                continue
            blocker_id = f"raw-authority-blocker:{_digest(['frontier', pass_id, item.plan_id])}"
            published[item.plan_id] = blocker_id
            observed = {
                "schema": "polylogue.raw-authority-frontier-obligation.v1",
                # Writer-declared blocker class; automatic clearing selects on
                # this positively (polylogue-l8tdh) and never clears a frontier
                # obligation.
                BLOCKER_ORIGIN_KEY: BLOCKER_ORIGIN_FRONTIER_OBLIGATION,
                "state": item.state.value,
                "actuator": item.actuator.value,
                "reason": item.reason,
                "evidence_digest": item.evidence_digest,
            }
            judgment_assertion_id = judgment_refs.get(item.plan_id)
            if judgment_assertion_id is not None:
                observed["judgment_assertion_id"] = judgment_assertion_id
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
                            "reason": (
                                "an accepted operator judgment acknowledged the retained conflict"
                                if plan_id_text in judged_plan_ids
                                else "a later complete frontier pass disproved the prior blocking state"
                            ),
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
            actuator=RawAuthorityActuator.NONE,
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
            "actuator": item.actuator.value,
            "reason": item.reason,
            "evidence_digest": item.evidence_digest,
            "strategy_witness": item.strategy_witness,
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
        overrides = _strategy_overrides(config, head_rows, index_db_path=index_db)

        def _override_for(row: dict[str, object]) -> _StrategyOverride | None:
            raw_id = str(row["accepted_raw_id"])
            logical_source_key = row.get("logical_source_key")
            if logical_source_key is not None:
                scoped = overrides.get(_quarantine_override_key(raw_id, str(logical_source_key)))
                if scoped is not None:
                    return scoped
            return overrides.get(raw_id)

        # _classify_frontier may persist a verified-blob receipt (polylogue-byw3y)
        # through this same connection; the outer ``conn`` context manager commits
        # those writes on clean exit (or rolls back on exception), so a receipt is
        # never durably recorded for bytes this pass didn't finish inspecting.
        head_items = [
            _classify_frontier(conn, BlobStore(root / "blob"), index_db, row, _override_for(row)) for row in head_rows
        ]
        superseded_items = _terminal_superseded_items(conn)
    all_items = _apply_judgment_dispositions(config, (*head_items, *superseded_items))
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
    executable_ids = {item.plan_id for item in gap_items if item.executable}
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
        executable_plan_count=len(executable_ids),
        items=bound_items,
    )


__all__ = [
    "RawAuthorityActuator",
    "RawAuthorityFrontierCensus",
    "RawAuthorityFrontierItem",
    "RawAuthorityFrontierState",
    "inspect_raw_authority_frontier",
]
