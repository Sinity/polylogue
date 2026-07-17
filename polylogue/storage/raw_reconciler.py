"""Proof-driven census for every accepted raw-authority frontier.

This module owns the provider-neutral state machine. Historical incident
actuators remain implementation strategies in :mod:`polylogue.storage.repair`;
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
from typing import TYPE_CHECKING, cast

from polylogue.config import Config
from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
from polylogue.paths import archive_file_set_root_for_paths
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_authority import (
    RawAuthorityCensusReceipt,
    RawReplayPlan,
    RawReplayPlanOutcome,
    RawReplayPlanStatus,
    finalize_raw_authority_census,
    raw_authority_detail_query_handle,
    record_raw_authority_census,
    record_raw_replay_outcome,
)
from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.storage.repair import (
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

_VERIFIED_BLOB_STATS: dict[str, tuple[int, int, int, int, int]] = {}


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
    """One persisted census over accepted heads plus terminal supersessions."""

    census_id: str
    query_handle: str
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
                "census_id": self.census_id,
                "query_handle": self.query_handle,
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


@dataclass(frozen=True, slots=True)
class RawAuthorityFrontierApplyReport:
    """Bounded receipt for one shared-contract apply pass."""

    census_id: str
    preview_census_id: str
    selected_plan_count: int
    executed_plan_count: int
    retryable_plan_count: int
    post_inventory_digest: str
    post_plan_count: int
    outcome_refs: tuple[str, ...]

    @property
    def success(self) -> bool:
        return self.retryable_plan_count == 0

    def to_dict(self) -> JSONDocument:
        return json_document(dataclasses.asdict(self) | {"success": self.success})


def _archive_root(config: Config) -> Path:
    return archive_file_set_root_for_paths(archive_root_path=config.archive_root, db_anchor=config.db_path)


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, object]]:
    names = tuple(column[0] for column in cursor.description or ())
    return [{name: _json_value(value) for name, value in zip(names, row, strict=True)} for row in cursor.fetchall()]


def _chunks(values: Sequence[str], size: int = 100) -> Iterator[list[str]]:
    """Yield bounded strategy-proof requests in deterministic order."""
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _verified_blob_bytes(blob_store: BlobStore, hash_hex: str) -> bool:
    """Hash once per stable on-disk inode state, then reuse the process receipt."""
    path = blob_store.blob_path(hash_hex)
    try:
        stat = path.stat()
    except OSError:
        return False
    fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if _VERIFIED_BLOB_STATS.get(hash_hex) == fingerprint:
        return True
    if not blob_store.verify(hash_hex):
        return False
    _VERIFIED_BLOB_STATS[hash_hex] = fingerprint
    return True


def _browser_strategy_witness(item: BrowserCaptureOriginRepairItem) -> JSONDocument:
    from polylogue.storage.repair import _browser_origin_item_payload

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
    from polylogue.storage.repair import _duplicate_raw_identity_proof_digest

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
    reacquisition_proven = blob_exists and _verified_blob_bytes(blob_store, blob_hash)
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
        from polylogue.storage.repair import _inspect_duplicate_raw_identity

        if len(duplicate_siblings) != 1:
            raise RuntimeError(f"duplicate alias classification is not injective for {raw_id}")
        with closing(sqlite3.connect(f"file:{blob_store.root.parent / 'index.db'}?mode=ro", uri=True)) as proof_conn:
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
        return _item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            actuator=RawAuthorityActuator.REFINE_QUARANTINE,
            row=row,
            reason="accepted raw authority remains quarantined pending exact refinement proof",
        )
    if row.get("raw_logical_source_key") != row.get("logical_source_key"):
        return _item(
            state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
            actuator=RawAuthorityActuator.REPLAY,
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
) -> dict[str, _StrategyOverride]:
    """Ask legacy incident inspectors for proofs, never for plan identity."""
    from polylogue.storage.repair import (
        inspect_browser_canonical_authority_conflicts,
        inspect_browser_capture_origin_mismatches,
        inspect_quarantined_accepted_raws,
    )

    overrides: dict[str, _StrategyOverride] = {}
    browser_ids = sorted(
        {
            str(row["accepted_raw_id"])
            for row in rows
            if row.get("raw_origin") is not None and row.get("session_origin") != row.get("raw_origin")
        }
    )
    for browser_chunk in _chunks(browser_ids):
        browser_items = inspect_browser_capture_origin_mismatches(config, browser_chunk)
        for browser_item in browser_items:
            if browser_item.status in {"eligible", "already_repaired"}:
                overrides[browser_item.raw_id] = _StrategyOverride(
                    state=RawAuthorityFrontierState.SAFELY_REKEYABLE,
                    actuator=RawAuthorityActuator.COPY_FORWARD_ORIGIN,
                    reason="browser-origin strategy proved an exact evidence-preserving copy-forward",
                    witness=_browser_strategy_witness(browser_item),
                    input_raw_ids=_browser_strategy_raw_ids(browser_item),
                )
        conflicts = inspect_browser_canonical_authority_conflicts(config, browser_chunk)
        for conflict_item in conflicts.items:
            if conflict_item.raw_id in overrides:
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
                input_raw_ids=tuple(
                    sorted(
                        {
                            raw_id
                            for raw_id in (conflict_item.raw_id, conflict_item.competing_raw_id)
                            if raw_id is not None
                        }
                    )
                ),
            )
    quarantine_ids = sorted(
        {
            str(row["accepted_raw_id"])
            for row in rows
            if row.get("revision_authority") == "quarantined" and str(row["accepted_raw_id"]) not in browser_ids
        }
    )
    for quarantine_chunk in _chunks(quarantine_ids):
        quarantine_items = inspect_quarantined_accepted_raws(config, quarantine_chunk)
        for quarantine_item in quarantine_items:
            if quarantine_item.status in {"eligible", "already_repaired"}:
                overrides[quarantine_item.raw_id] = _StrategyOverride(
                    state=RawAuthorityFrontierState.SAFELY_REKEYABLE,
                    actuator=RawAuthorityActuator.REFINE_QUARANTINE,
                    reason="quarantined-raw strategy proved exact accepted-byte and semantic authority",
                    witness=_quarantine_strategy_witness(quarantine_item),
                    input_raw_ids=tuple(sorted({quarantine_item.raw_id, *quarantine_item.census_stage_raw_ids})),
                )
    return overrides


_OBLIGATION_STATES = {
    RawAuthorityFrontierState.MISSING_BYTES_REACQUIRE,
    RawAuthorityFrontierState.CONFLICTING_AUTHORITY_NEEDS_JUDGMENT,
    RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
    RawAuthorityFrontierState.CORRUPT,
}


def _record_judgment_candidate(config: Config, item: RawAuthorityFrontierItem, *, now_ms: int) -> tuple[str, bool]:
    """Persist the conflict as a non-authoritative candidate for operator judgment."""
    from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope, upsert_assertion

    assertion_id = f"judgment:{_digest(['raw-authority-frontier', item.plan_id])}"
    root = _archive_root(config)
    with closing(sqlite3.connect(root / "user.db")) as conn, conn:
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
        resolutions = {
            str(plan_id): json_document(json.loads(str(resolution)))
            for plan_id, resolution in conn.execute(
                """
                SELECT plan_id, resolution
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
    census_id: str,
    items: tuple[RawAuthorityFrontierItem, ...],
) -> None:
    """Publish current obligations and close only those disproven by a later census."""
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
    with closing(sqlite3.connect(root / "source.db")) as conn, conn:
        for item in blocking:
            if item.plan_id in judged_plan_ids:
                continue
            blocker_id = f"raw-authority-blocker:{_digest(['frontier', census_id, item.plan_id])}"
            observed = {
                "schema": "polylogue.raw-authority-frontier-obligation.v1",
                "state": item.state.value,
                "actuator": item.actuator.value,
                "reason": item.reason,
                "evidence_digest": item.evidence_digest,
                "judgment_assertion_id": judgment_refs.get(item.plan_id),
            }
            conn.execute(
                """
                INSERT INTO raw_authority_blockers (
                    blocker_id, plan_id, census_id, reason, expected_json,
                    observed_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (
                    blocker_id,
                    item.plan_id,
                    census_id,
                    item.reason,
                    _canonical_json(_plan(item).to_dict()),
                    _canonical_json(observed),
                    now,
                ),
            )
        open_rows = conn.execute(
            """
            SELECT b.blocker_id, b.plan_id
            FROM raw_authority_blockers AS b
            JOIN raw_authority_plans AS p ON p.plan_id = b.plan_id
            WHERE b.resolved_at_ms IS NULL
              AND json_extract(p.authority_witness_json, '$.schema') =
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
                                else "a later complete frontier census disproved the prior blocking state"
                            ),
                            "successor_census_id": census_id,
                        }
                    ),
                    blocker_id,
                ),
            )


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
    index_db = root / "index.db"
    if not source_db.is_file() or not index_db.is_file():
        raise RuntimeError("raw authority frontier census requires initialized source and index tiers")
    with closing(sqlite3.connect(source_db)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
        head_rows = _frontier_rows(conn)
        overrides = _strategy_overrides(config, head_rows)
        head_items = [
            _classify_frontier(conn, BlobStore(root / "blob"), row, overrides.get(str(row["accepted_raw_id"])))
            for row in head_rows
        ]
        superseded_items = _terminal_superseded_items(conn)
    all_items = _apply_judgment_dispositions(config, (*head_items, *superseded_items))
    return (
        tuple(sorted(all_items, key=lambda item: (item.raw_id, item.plan_id))),
        len(head_items),
        len(superseded_items),
    )


def _state_counts(items: tuple[RawAuthorityFrontierItem, ...]) -> JSONDocument:
    return json_document(dict(sorted(Counter(item.state.value for item in items).items())))


def _residual(state_counts: JSONDocument) -> JSONDocument:
    return json_document(
        {
            "schema": "polylogue.raw-authority-frontier-residual.v1",
            "state_counts": {
                state: count
                for state, count in state_counts.items()
                if state != RawAuthorityFrontierState.PROVEN_CURRENT.value
            },
        }
    )


def inspect_raw_authority_frontier(config: Config) -> RawAuthorityFrontierCensus:
    """Persist one complete accepted-frontier census without applying repairs."""
    root = _archive_root(config)
    all_items, accepted_head_count, terminal_superseded_count = _frontier_items(config)
    state_counts_counter = Counter(item.state.value for item in all_items)
    state_counts = json_document(dict(sorted(state_counts_counter.items())))
    inventory_digest = _digest([item.to_dict() for item in all_items])
    gap_items = tuple(item for item in all_items if item.state is not RawAuthorityFrontierState.PROVEN_CURRENT)
    plans = tuple(_plan(item) for item in gap_items)
    executable_ids = {item.plan_id for item in gap_items if item.executable}
    receipt: RawAuthorityCensusReceipt = record_raw_authority_census(
        root,
        plans,
        selected_plan_ids=set(),
        executable_plan_ids=executable_ids,
        mode="dry_run",
        quiescent=True,
        scope={
            "schema": "polylogue.raw-authority-frontier-scope.v1",
            "accepted_head_count": accepted_head_count,
            "terminal_superseded_count": terminal_superseded_count,
            "inventory_digest": inventory_digest,
            "state_counts": state_counts,
        },
        residual=_residual(state_counts),
    )
    _reconcile_frontier_obligations(config, receipt.census_id, all_items)
    bound_items = tuple(
        dataclasses.replace(
            item,
            evidence_ref=(
                raw_authority_detail_query_handle(receipt.census_id, item.plan_id)
                if item.state is not RawAuthorityFrontierState.PROVEN_CURRENT
                else None
            ),
        )
        for item in all_items
    )
    return RawAuthorityFrontierCensus(
        census_id=receipt.census_id,
        query_handle=receipt.query_handle,
        inventory_digest=inventory_digest,
        plan_inventory_digest=receipt.inventory_digest,
        state_counts=state_counts,
        accepted_head_count=accepted_head_count,
        terminal_superseded_count=terminal_superseded_count,
        plan_count=len(plans),
        executable_plan_count=len(executable_ids),
        items=bound_items,
    )


def _preview_plan_ids(root: Path, census_id: str) -> set[str]:
    with closing(sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)) as conn:
        row = conn.execute(
            "SELECT mode, lifecycle_status FROM raw_authority_censuses WHERE census_id = ?",
            (census_id,),
        ).fetchone()
        if row is None:
            raise KeyError(census_id)
        if tuple(row) != ("dry_run", "completed"):
            raise RuntimeError("raw authority apply requires a completed dry-run frontier census")
        return {
            str(plan_id)
            for (plan_id,) in conn.execute(
                "SELECT plan_id FROM raw_authority_census_plans WHERE census_id = ?",
                (census_id,),
            )
        }


def _apply_strategy(
    config: Config,
    item: RawAuthorityFrontierItem,
) -> JSONDocument:
    from polylogue.storage.index_generation import RebuildLease
    from polylogue.storage.repair import (
        _apply_browser_conflict_canonical_resolution,
        _apply_browser_origin_repair_item,
        _apply_duplicate_raw_identity_repair,
        _attach_repair_index,
        _browser_origin_strategy_terminal,
        _cas_refine_quarantined_accepted_raw,
        _inspect_browser_capture_origin_strategy,
        _inspect_duplicate_raw_identity,
        _inspect_quarantined_accepted_raw,
        _stage_browser_origin_copy_forward_source,
        _stage_quarantined_census_cohort,
        _validate_quarantined_raw_repair_blob_budget,
        _verify_browser_origin_copy_forward_source_stage,
    )

    root = _archive_root(config)
    source_db = root / "source.db"
    index_db = root / "index.db"

    if item.actuator is RawAuthorityActuator.RESOLVE_CONFLICT:
        conflict = item.strategy_witness.get("conflict")
        judgment = item.strategy_witness.get("judgment")
        if not isinstance(conflict, dict) or not isinstance(judgment, dict):
            raise RuntimeError("conflict-resolution strategy witness is incomplete")
        evidence = conflict.get("evidence")
        if not isinstance(evidence, dict) or judgment.get("disposition") != "retain_canonical_authority":
            raise RuntimeError("conflict-resolution strategy is not explicitly authorized")
        with RebuildLease(root), closing(sqlite3.connect(f"file:{index_db}?mode=rw", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("ATTACH DATABASE ? AS source", (f"file:{source_db}?mode=ro",))
            conn.execute("BEGIN IMMEDIATE")
            try:
                _apply_browser_conflict_canonical_resolution(root, conn, item.raw_id, evidence)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return json_document(
            {
                "strategy": item.actuator.value,
                "disposition": judgment["disposition"],
                "repaired_count": 1,
            }
        )
    if item.actuator is RawAuthorityActuator.FOLD_DUPLICATE_ALIAS:
        canonical_ids = tuple(raw_id for raw_id in item.input_raw_ids if raw_id != item.raw_id)
        if len(canonical_ids) != 1:
            raise RuntimeError("duplicate-alias plan does not identify exactly one canonical twin")
        with RebuildLease(root), closing(sqlite3.connect(f"file:{index_db}?mode=rw", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("ATTACH DATABASE ? AS source", (f"file:{source_db}?mode=ro",))
            conn.execute("BEGIN IMMEDIATE")
            try:
                duplicate_locked = _inspect_duplicate_raw_identity(conn, root, item.raw_id, canonical_ids[0])
                if _duplicate_strategy_witness(duplicate_locked) != item.strategy_witness:
                    raise RuntimeError("duplicate strategy proof changed after plan authorization")
                if duplicate_locked.status == "eligible":
                    _apply_duplicate_raw_identity_repair(conn, duplicate_locked)
                elif duplicate_locked.status != "already_repaired":
                    raise RuntimeError(f"duplicate strategy lost its exact proof: {duplicate_locked.reason}")
                after = _inspect_duplicate_raw_identity(conn, root, item.raw_id, canonical_ids[0])
                if after.status != "already_repaired":
                    raise RuntimeError("duplicate strategy did not reach its typed terminal postcondition")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return json_document(
            {
                "strategy": item.actuator.value,
                "repaired_count": int(duplicate_locked.status == "eligible"),
                "already_repaired_count": int(duplicate_locked.status == "already_repaired"),
            }
        )
    if item.actuator is RawAuthorityActuator.COPY_FORWARD_ORIGIN:
        from polylogue.storage.blob_publication import exclude_archive_blob_publishers

        with RebuildLease(root), exclude_archive_blob_publishers(source_db):
            with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as proof_conn:
                proof_conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
                preview = _inspect_browser_capture_origin_strategy(root, item.raw_id, conn=proof_conn)
            if _browser_strategy_witness(preview) != item.strategy_witness:
                raise RuntimeError("browser-origin strategy proof changed after plan authorization")
            if preview.status == "eligible" and preview.repair_strategy == "copy_forward":
                with closing(sqlite3.connect(f"file:{source_db}?mode=rw", uri=True)) as source_conn:
                    source_conn.execute("PRAGMA foreign_keys = ON")
                    source_conn.execute("BEGIN IMMEDIATE")
                    try:
                        if not preview.copy_forward_source_complete:
                            _verify_browser_origin_copy_forward_source_stage(root, source_conn, preview)
                            _stage_browser_origin_copy_forward_source(source_conn, preview)
                        source_conn.commit()
                    except Exception:
                        source_conn.rollback()
                        raise
            elif preview.status == "eligible" and preview.repair_strategy == "restore_canonical_head":
                pass
            elif preview.status != "already_repaired":
                raise RuntimeError(f"browser-origin strategy lost its exact proof: {preview.reason}")
            with closing(sqlite3.connect(f"file:{index_db}?mode=rw", uri=True)) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("ATTACH DATABASE ? AS source", (str(source_db),))
                conn.execute("BEGIN IMMEDIATE")
                try:
                    browser_locked = _inspect_browser_capture_origin_strategy(root, item.raw_id, conn=conn)
                    if _browser_strategy_witness(browser_locked) != item.strategy_witness:
                        raise RuntimeError("browser-origin strategy proof changed under the apply transaction")
                    if browser_locked.status == "eligible":
                        _apply_browser_origin_repair_item(conn, browser_locked)
                    elif browser_locked.status != "already_repaired":
                        raise RuntimeError(f"browser-origin strategy lost its locked proof: {browser_locked.reason}")
                    if not _browser_origin_strategy_terminal(conn, browser_locked):
                        raise RuntimeError("browser-origin strategy did not reach its typed terminal postcondition")
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
        return json_document(
            {
                "strategy": item.actuator.value,
                "repaired_count": int(preview.status == "eligible"),
                "already_repaired_count": int(preview.status == "already_repaired"),
            }
        )
    if item.actuator is RawAuthorityActuator.REFINE_QUARANTINE:
        with RebuildLease(root), closing(sqlite3.connect(f"file:{source_db}?mode=rw", uri=True)) as source_conn:
            source_conn.execute("PRAGMA foreign_keys = ON")
            _attach_repair_index(source_conn, index_db)
            source_conn.execute("BEGIN IMMEDIATE")
            try:
                _validate_quarantined_raw_repair_blob_budget(source_conn, [item.raw_id])
                quarantine_locked = _inspect_quarantined_accepted_raw(root, item.raw_id, conn=source_conn)
                if _quarantine_strategy_witness(quarantine_locked) != item.strategy_witness:
                    raise RuntimeError("quarantine strategy proof changed after plan authorization")
                if quarantine_locked.status == "eligible" and quarantine_locked.census_stage_raw_ids:
                    _stage_quarantined_census_cohort(source_conn, quarantine_locked)
                    quarantine_locked = _inspect_quarantined_accepted_raw(root, item.raw_id, conn=source_conn)
                if quarantine_locked.status == "eligible":
                    _cas_refine_quarantined_accepted_raw(source_conn, quarantine_locked)
                elif quarantine_locked.status != "already_repaired":
                    raise RuntimeError(f"quarantine strategy lost its exact proof: {quarantine_locked.reason}")
                quarantine_after = _inspect_quarantined_accepted_raw(root, item.raw_id, conn=source_conn)
                if quarantine_after.status != "already_repaired":
                    raise RuntimeError("quarantine strategy did not reach its typed terminal postcondition")
                source_conn.commit()
            except Exception:
                source_conn.rollback()
                raise
        return json_document(
            {
                "strategy": item.actuator.value,
                "repaired_count": int(quarantine_locked.status == "eligible"),
                "already_repaired_count": int(quarantine_locked.status == "already_repaired"),
            }
        )
    raise RuntimeError(f"raw authority plan actuator is not automatically executable: {item.actuator.value}")


def apply_raw_authority_frontier(
    config: Config,
    *,
    preview_census_id: str,
    selected_plan_ids: tuple[str, ...],
) -> RawAuthorityFrontierApplyReport:
    """Authorize, apply, receipt, and postflight exact plans through one contract."""
    from polylogue.maintenance.offline_guard import offline_maintenance_block_reason

    block_reason = offline_maintenance_block_reason(config, active=True, dry_run=False)
    if block_reason is not None:
        raise RuntimeError(block_reason)
    if not selected_plan_ids or len(set(selected_plan_ids)) != len(selected_plan_ids):
        raise ValueError("raw authority apply requires unique selected plan ids")
    root = _archive_root(config)
    preview_ids = _preview_plan_ids(root, preview_census_id)
    unknown_preview_ids = set(selected_plan_ids) - preview_ids
    if unknown_preview_ids:
        raise RuntimeError(f"selected plans are absent from the preview census: {sorted(unknown_preview_ids)}")
    before_items, accepted_head_count, terminal_superseded_count = _frontier_items(config)
    current_by_plan = {item.plan_id: item for item in before_items}
    missing_current_ids = set(selected_plan_ids) - set(current_by_plan)
    if missing_current_ids:
        raise RuntimeError(f"selected raw authority plans changed after preview: {sorted(missing_current_ids)}")
    selected_items = tuple(current_by_plan[plan_id] for plan_id in selected_plan_ids)
    if any(not item.executable for item in selected_items):
        raise RuntimeError("raw authority apply selected a non-executable judgment/reacquisition/debt plan")
    gap_items = tuple(item for item in before_items if item.state is not RawAuthorityFrontierState.PROVEN_CURRENT)
    plans = tuple(_plan(item) for item in gap_items)
    state_counts = _state_counts(before_items)
    apply_receipt = record_raw_authority_census(
        root,
        plans,
        selected_plan_ids=set(selected_plan_ids),
        executable_plan_ids={item.plan_id for item in gap_items if item.executable},
        mode="apply",
        quiescent=True,
        scope={
            "schema": "polylogue.raw-authority-frontier-scope.v1",
            "preview_census_id": preview_census_id,
            "accepted_head_count": accepted_head_count,
            "terminal_superseded_count": terminal_superseded_count,
            "inventory_digest": _digest([item.to_dict() for item in before_items]),
            "state_counts": state_counts,
        },
        residual=_residual(state_counts),
    )
    executed = 0
    retryable = 0
    outcome_refs: list[str] = []
    for item in selected_items:
        try:
            strategy_receipt = _apply_strategy(config, item)
            outcome = RawReplayPlanOutcome(
                plan_id=item.plan_id,
                input_raw_ids=item.input_raw_ids,
                status=RawReplayPlanStatus.EXECUTED,
                reason="shared raw-authority plan reached its strategy terminal state",
                next_action="none",
                application_receipt=json_document(
                    {
                        "schema": "polylogue.raw-authority-frontier-application.v1",
                        "preview_census_id": preview_census_id,
                        "frontier_evidence_digest": item.evidence_digest,
                        "strategy": strategy_receipt,
                    }
                ),
            )
            executed += 1
        except Exception as exc:
            logger.warning(
                "raw authority strategy failed plan=%s actuator=%s",
                item.plan_id,
                item.actuator.value,
                exc_info=True,
            )
            outcome = RawReplayPlanOutcome(
                plan_id=item.plan_id,
                input_raw_ids=item.input_raw_ids,
                status=RawReplayPlanStatus.RETRYABLE,
                reason=f"strategy did not reach a proven terminal state: {type(exc).__name__}: {exc}",
                next_action="reinspect the same immutable plan and retry after correcting the causal condition",
                application_receipt=json_document(
                    {
                        "schema": "polylogue.raw-authority-frontier-application.v1",
                        "preview_census_id": preview_census_id,
                        "frontier_evidence_digest": item.evidence_digest,
                    }
                ),
            )
            retryable += 1
        record_raw_replay_outcome(root, apply_receipt.census_id, outcome)
        outcome_refs.append(raw_authority_detail_query_handle(apply_receipt.census_id, item.plan_id))
    after_items, _after_head_count, _after_superseded_count = _frontier_items(config)
    after_state_counts = _state_counts(after_items)
    after_plans = tuple(
        _plan(item) for item in after_items if item.state is not RawAuthorityFrontierState.PROVEN_CURRENT
    )
    finalized = finalize_raw_authority_census(
        root,
        apply_receipt.census_id,
        post_plans=after_plans,
        post_residual=_residual(after_state_counts),
        interrupted=retryable > 0,
    )
    assert finalized.post_inventory_digest is not None
    assert finalized.post_plan_count is not None
    return RawAuthorityFrontierApplyReport(
        census_id=apply_receipt.census_id,
        preview_census_id=preview_census_id,
        selected_plan_count=len(selected_items),
        executed_plan_count=executed,
        retryable_plan_count=retryable,
        post_inventory_digest=finalized.post_inventory_digest,
        post_plan_count=finalized.post_plan_count,
        outcome_refs=tuple(outcome_refs),
    )


def recover_interrupted_raw_authority_frontier(config: Config) -> tuple[str, ...]:
    """Conserve crash-left frontier applications from current typed evidence."""
    root = _archive_root(config)
    source_db = root / "source.db"
    if not source_db.is_file():
        return ()
    with closing(sqlite3.connect(source_db)) as conn:
        conn.row_factory = sqlite3.Row
        census_ids = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT census_id
                FROM raw_authority_censuses
                WHERE lifecycle_status = 'planned'
                  AND json_extract(scope_json, '$.schema') =
                      'polylogue.raw-authority-frontier-scope.v1'
                ORDER BY sequence_no
                """
            )
        ]
        if not census_ids:
            return ()
        rows = conn.execute(
            """
            SELECT c.census_id, p.*
            FROM raw_authority_censuses AS c
            JOIN raw_authority_census_plans AS cp ON cp.census_id = c.census_id
            JOIN raw_authority_plans AS p ON p.plan_id = cp.plan_id
            WHERE c.lifecycle_status = 'planned'
              AND cp.selected = 1 AND cp.outcome_recorded = 0
              AND json_extract(p.authority_witness_json, '$.schema') =
                  'polylogue.raw-authority-frontier-plan.v1'
            ORDER BY c.sequence_no, cp.ordinal
            """
        ).fetchall()
    current_items, _head_count, _superseded_count = _frontier_items(config)
    current_by_plan = {item.plan_id: item for item in current_items}
    recovered: list[str] = []
    for row in rows:
        census_id = str(row["census_id"])
        plan = RawReplayPlan(
            plan_id=str(row["plan_id"]),
            input_digest=str(row["input_digest"]),
            input_raw_ids=tuple(str(value) for value in json.loads(str(row["input_raw_ids_json"]))),
            logical_keys=tuple(str(value) for value in json.loads(str(row["logical_keys_json"]))),
            authority_witness=json_document(json.loads(str(row["authority_witness_json"]))),
            source_preconditions=json_document(json.loads(str(row["source_preconditions_json"]))),
            index_preconditions=json_document(json.loads(str(row["index_preconditions_json"]))),
        )
        current = current_by_plan.get(plan.plan_id)
        related = tuple(item for item in current_items if set(item.input_raw_ids).intersection(plan.input_raw_ids))
        receipt = json_document(
            {
                "schema": "polylogue.raw-authority-frontier-recovery.v1",
                "current_items": [item.to_dict() for item in related],
            }
        )
        if current is not None:
            outcome = RawReplayPlanOutcome(
                plan.plan_id,
                plan.input_raw_ids,
                RawReplayPlanStatus.RETRYABLE,
                "interrupted before the immutable frontier plan reached terminal postconditions",
                "retry the same immutable plan",
                receipt,
            )
            record_raw_replay_outcome(root, census_id, outcome)
        elif related and all(
            item.state in {RawAuthorityFrontierState.PROVEN_CURRENT, RawAuthorityFrontierState.SUPERSEDED}
            for item in related
        ):
            outcome = RawReplayPlanOutcome(
                plan.plan_id,
                plan.input_raw_ids,
                RawReplayPlanStatus.EXECUTED,
                "interrupted application recovered from typed terminal frontier states",
                "none",
                receipt,
            )
            record_raw_replay_outcome(root, census_id, outcome)
        else:
            from polylogue.storage.raw_authority import reject_stale_raw_replay_plan

            reject_stale_raw_replay_plan(root, census_id, plan, receipt)
        recovered.append(plan.plan_id)
    post_state_counts = _state_counts(current_items)
    post_plans = tuple(
        _plan(item) for item in current_items if item.state is not RawAuthorityFrontierState.PROVEN_CURRENT
    )
    for census_id in census_ids:
        finalize_raw_authority_census(
            root,
            census_id,
            post_plans=post_plans,
            post_residual=_residual(post_state_counts),
            interrupted=True,
        )
    return tuple(recovered)


__all__ = [
    "RawAuthorityActuator",
    "RawAuthorityFrontierCensus",
    "RawAuthorityFrontierApplyReport",
    "RawAuthorityFrontierItem",
    "RawAuthorityFrontierState",
    "apply_raw_authority_frontier",
    "inspect_raw_authority_frontier",
    "recover_interrupted_raw_authority_frontier",
]
