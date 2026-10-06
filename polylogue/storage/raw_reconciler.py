"""Canonical classification of accepted raw-authority inputs.

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
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger
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
        fields = dataclasses.asdict(self)
        fields["input_raw_ids"] = list(self.input_raw_ids)
        fields["state"] = self.state.value
        from polylogue.core.json import require_json_document

        return require_json_document(fields, context="frontier classification")


_BLOB_RECEIPT_UPSERT_SQL = "\n        INSERT INTO verified_blob_receipts\n            (blob_hash, st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns, verified_at_ms)\n        VALUES (?, ?, ?, ?, ?, ?, ?)\n        ON CONFLICT(blob_hash) DO UPDATE SET\n            st_dev = excluded.st_dev,\n            st_ino = excluded.st_ino,\n            st_size = excluded.st_size,\n            st_mtime_ns = excluded.st_mtime_ns,\n            st_ctime_ns = excluded.st_ctime_ns,\n            verified_at_ms = excluded.verified_at_ms\n        "


def _verify_blob_bytes(
    blob_store: BlobStore,
    hash_hex: str,
    *,
    retained_fingerprint: tuple[int, int, int, int, int] | None,
    record_receipt: Callable[[str, tuple[int, int, int, int, int]], None],
) -> bool:
    """Verify actual CAS bytes against the retained original receipt."""
    path = blob_store.blob_path(hash_hex)
    try:
        stat = path.stat()
    except OSError:
        return False
    fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if retained_fingerprint == fingerprint:
        return True
    if not blob_store.verify(hash_hex):
        return False
    record_receipt(hash_hex, fingerprint)
    return True


_FRONTIER_INDEX_INPUT_SQL = """
SELECT h.logical_source_key,h.session_id,
       COALESCE(s.raw_id,h.accepted_raw_id) AS accepted_raw_id,
       h.accepted_raw_id AS head_accepted_raw_id,h.accepted_source_revision,
       hex(h.accepted_content_hash) AS accepted_content_hash,
       h.accepted_frontier_kind,h.accepted_frontier,h.decided_at_ms AS head_decided_at_ms,
       h.acquisition_generation AS head_acquisition_generation,h.append_end_offset AS head_append_end_offset,
       s.origin AS session_origin,s.raw_id AS session_raw_id,
       hex(s.content_hash) AS session_content_hash,s.message_count
FROM raw_revision_heads h LEFT JOIN sessions s ON s.session_id=h.session_id
WHERE h.logical_source_key=?
"""

_FRONTIER_SOURCE_INPUT_SQL = """
SELECT r.origin AS raw_origin,r.capture_mode,r.native_id,r.source_path,r.source_index,
       hex(r.blob_hash) AS blob_hash,r.blob_size,r.logical_source_key AS raw_logical_source_key,
       r.revision_kind,r.source_revision,r.predecessor_raw_id,r.baseline_raw_id,
       r.append_start_offset,r.append_end_offset,r.acquisition_generation,r.revision_authority,
       m.source_revision AS membership_source_revision,hex(m.normalized_content_hash) AS membership_content_hash,
       m.revision_authority AS membership_authority,m.decision AS membership_decision,
       c.status AS membership_census_status,c.parser_fingerprint AS membership_parser_fingerprint,
       pc.status AS parser_census_status,pc.parser_fingerprint AS authority_parser_fingerprint,
       EXISTS(SELECT 1 FROM json_each(pc.logical_keys_json) WHERE type='text' AND value=?) AS parser_names_key
FROM raw_sessions r
LEFT JOIN raw_session_memberships m ON m.raw_id=r.raw_id AND m.logical_source_key=?
LEFT JOIN raw_membership_census c ON c.raw_id=r.raw_id
LEFT JOIN raw_authority_parser_census pc ON pc.raw_id=r.raw_id
WHERE r.raw_id=?
"""


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
                "membership_source_revision",
                "membership_content_hash",
                "membership_authority",
                "membership_decision",
                "membership_census_status",
                "membership_parser_fingerprint",
                "parser_census_status",
                "authority_parser_fingerprint",
                "parser_names_key",
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


def _classify_frontier_row(
    row: dict[str, object],
    *,
    verified_bytes: bool,
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
    if not verified_bytes:
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
    if row.get("accepted_frontier_kind") == "semantic":
        from polylogue.archive.revision_authority import raw_authority_parser_fingerprint

        fingerprint = raw_authority_parser_fingerprint()
        # A decided-ambiguous cohort keeps its last accepted head (#3282). The
        # conflict is durable membership debt on the head's own row, not a
        # provenance gap in the head that remains materialized.
        retained_under_debt = (
            row.get("membership_decision") == "ambiguous" and row.get("membership_authority") == "quarantined"
        )
        if (
            not (
                retained_under_debt
                or (row.get("membership_decision") == "applied" and row.get("membership_authority") == "byte_proven")
            )
            or row.get("membership_source_revision") != row.get("accepted_source_revision")
            or str(row.get("membership_content_hash") or "").lower()
            != str(row.get("membership_source_revision") or "").lower()
            or row.get("membership_census_status") != "complete"
            or row.get("membership_parser_fingerprint") != fingerprint
            or row.get("parser_census_status") != "complete"
            or row.get("authority_parser_fingerprint") != fingerprint
            or not row.get("parser_names_key")
        ):
            return _item(
                state=RawAuthorityFrontierState.UNRESOLVED_PROVENANCE,
                row=row,
                reason="accepted semantic head lacks its original complete applied membership authority",
            )
        return _item(
            state=RawAuthorityFrontierState.PROVEN_CURRENT,
            row=row,
            reason=(
                "accepted head retained under ambiguous membership debt; source bytes and session agree"
                if retained_under_debt
                else "accepted source bytes, membership, head, and materialized session agree"
            ),
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


def _frontier_blocker_identity(
    resolved_at: Callable[[str], tuple[object] | sqlite3.Row | None],
    *,
    pass_id: str,
    plan_id: str,
) -> str:
    """Return the id this pass must publish its obligation under.

    Prepared blocker acknowledgement tombstones a frontier blocker on the
    operator's acknowledgement alone: it discharges nothing, and for a frontier
    witness it does not even rebuild the plan from current evidence. ``pass_id``
    is a content address over the inspected inventory, so a later pass over
    *unchanged* blocking evidence derives the identical pass, plan and blocker
    ids -- and the publishing ``INSERT ... ON CONFLICT DO NOTHING`` then found
    the tombstoned row and wrote nothing. The obligation stayed closed while the
    same missing, quarantined or corrupt evidence still existed, so
    ``raw_authority_blocker_count`` read zero and readiness called the archive
    clean (PR #5350).

    Chaining past each acknowledgement mints a fresh *unresolved* obligation
    while leaving the acknowledgement row -- the operator's durable resolution
    receipt -- intact. The successor id is derived from the row it supersedes,
    so it is as deterministic as the original: repeated passes over unchanged
    evidence converge on the same successor rather than minting one per pass.
    """

    blocker_id = f"raw-authority-blocker:{_digest(['frontier', pass_id, plan_id])}"
    while True:
        row = resolved_at(blocker_id)
        if row is None or row[0] is None:
            return blocker_id
        blocker_id = f"raw-authority-blocker:{_digest(['frontier', pass_id, plan_id, blocker_id])}"


_FRONTIER_BLOCKER_INSERT_SQL = """
INSERT INTO raw_authority_blockers (
    blocker_id, plan_input_digest, observed_pass_id, reason, expected_json,
    observed_json, created_at_ms
) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING
"""
_FRONTIER_BLOCKER_RESOLVE_SQL = """
UPDATE raw_authority_blockers SET resolved_at_ms=?, resolution=?
WHERE blocker_id=? AND resolved_at_ms IS NULL
"""


def _frontier_obligation_values(
    item: RawAuthorityFrontierItem,
    *,
    blocker_id: str,
    pass_id: str,
    observed_at_ms: int,
) -> tuple[object, ...]:
    observed = {
        "schema": "polylogue.raw-authority-frontier-obligation.v1",
        BLOCKER_ORIGIN_KEY: BLOCKER_ORIGIN_FRONTIER_OBLIGATION,
        "state": item.state.value,
        "reason": item.reason,
        "evidence_digest": item.evidence_digest,
    }
    return (
        blocker_id,
        _plan(item).input_digest,
        pass_id,
        item.reason,
        _canonical_json(_plan(item).to_dict()),
        _canonical_json(observed),
        observed_at_ms,
    )


def _frontier_resolution_values(
    blocker_id: str,
    *,
    pass_id: str,
    observed_at_ms: int,
) -> tuple[object, ...]:
    return (
        observed_at_ms,
        _canonical_json(
            {
                "schema": "polylogue.raw-authority-obligation-resolution.v1",
                "reason": "a later complete frontier pass disproved the prior blocking state",
                "successor_pass_id": pass_id,
            }
        ),
        blocker_id,
    )


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


__all__ = [
    "RawAuthorityFrontierItem",
    "RawAuthorityFrontierState",
]
