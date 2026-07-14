"""Holdout cohorts: persistence-class policy + planner enforcement (rxdo.9.4).

Holdout is an access POLICY layered on an existing promoted
:class:`~polylogue.storage.sqlite.query_objects.ResultSetManifest`, not a
second cohort/result relation type -- a result set keeps whatever
``persistence_class`` it was created with (typically ``cohort``), and
:func:`mark_holdout` additionally records that its members are reserved for
confirmation and must not be read by exploratory analysis. This gives demo
and evaluation claims a provable "held on untouched data" leg: exploratory
queries are refused read access by default (:func:`require_non_holdout_access`),
a declared confirmation run is explicitly allowed and leaves a visible
access receipt, and an undeclared/accidental access is recorded as
contamination that can never be retroactively cleared -- once contaminated,
the relation can no longer back an "untouched holdout" claim.

The actual query-planner integration (the rxdo.6 layer that would call
:func:`require_non_holdout_access` before resolving an exploratory query's
``from result-set:<id>`` operand) has not landed in this tree yet; wiring
that call site is deferred to whichever lane lands the rxdo.6 planner. This
module is the durable storage + enforcement primitive that planner will
call into.

Reset/excision durability: no excision mechanism exists for ``result_sets``
in this tree yet (see ``polylogue-layg`` for the separate source.db blob
excision cluster, which does not touch this table). The floor this module
ships is the migration's ``ON DELETE RESTRICT`` FK from
``result_set_holdout_policies`` to ``result_sets`` -- a raw ``DELETE`` of a
holdout-marked result set raises ``sqlite3.IntegrityError`` rather than
silently dropping the policy (see
``test_deleting_a_holdout_marked_result_set_is_blocked_by_the_durable_fk``).
A future excision/reset mechanism must route through an explicit
unmark-then-delete step (or a policy-aware cascade) rather than a raw
DELETE, or it will hit this same constraint; that integration is not
designed or implemented here.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.storage.sqlite.query_objects import get_result_set


class HoldoutAccessError(RuntimeError):
    """Raised when an exploratory read is attempted against a holdout relation."""


@dataclass(frozen=True, slots=True)
class HoldoutPolicy:
    result_set_id: str
    frame: str
    selection_definition: Mapping[str, object]
    intended_confirmation_use: str
    authority: str
    created_epoch: str


@dataclass(frozen=True, slots=True)
class HoldoutAccessReceipt:
    receipt_id: str
    result_set_id: str
    accessor_ref: str
    declared_confirmation: bool
    contamination: bool
    reason: str | None
    accessed_at_ms: int


def mark_holdout(
    conn: sqlite3.Connection,
    *,
    result_set_id: str,
    frame: str,
    selection_definition: Mapping[str, object],
    intended_confirmation_use: str,
    authority: str,
    created_epoch: str,
    created_at_ms: int,
) -> HoldoutPolicy:
    """Mark an existing promoted result set as a holdout relation.

    Idempotent when re-marking with byte-identical policy fields; raises if
    the result set does not exist or a *different* policy is already bound
    (a holdout's declared frame/selection/authority is fixed at creation,
    not silently mutable).
    """
    if get_result_set(conn, result_set_id) is None:
        raise KeyError(f"result-set:{result_set_id}")
    policy = HoldoutPolicy(
        result_set_id=result_set_id,
        frame=frame,
        selection_definition=dict(selection_definition),
        intended_confirmation_use=intended_confirmation_use,
        authority=authority,
        created_epoch=created_epoch,
    )
    existing = get_holdout_policy(conn, result_set_id)
    if existing is not None:
        if existing != policy:
            raise ValueError(f"result-set:{result_set_id} is already a holdout with a different declared policy")
        return existing
    conn.execute(
        """
        INSERT INTO result_set_holdout_policies (
            result_set_id, frame, selection_definition_json, intended_confirmation_use,
            authority, created_epoch, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result_set_id,
            frame,
            _json(policy.selection_definition),
            intended_confirmation_use,
            authority,
            created_epoch,
            created_at_ms,
        ),
    )
    return policy


def get_holdout_policy(conn: sqlite3.Connection, result_set_id: str) -> HoldoutPolicy | None:
    row = conn.execute(
        """
        SELECT result_set_id, frame, selection_definition_json, intended_confirmation_use,
               authority, created_epoch
        FROM result_set_holdout_policies WHERE result_set_id = ?
        """,
        (result_set_id,),
    ).fetchone()
    if row is None:
        return None
    return HoldoutPolicy(
        result_set_id=str(row[0]),
        frame=str(row[1]),
        selection_definition=json.loads(str(row[2])),
        intended_confirmation_use=str(row[3]),
        authority=str(row[4]),
        created_epoch=str(row[5]),
    )


def is_holdout(conn: sqlite3.Connection, result_set_id: str) -> bool:
    return get_holdout_policy(conn, result_set_id) is not None


def record_holdout_access(
    conn: sqlite3.Connection,
    *,
    receipt_id: str,
    result_set_id: str,
    accessor_ref: str,
    declared_confirmation: bool,
    accessed_at_ms: int,
    reason: str | None = None,
) -> HoldoutAccessReceipt:
    """Record one read of a holdout relation's members, declared or not.

    ``declared_confirmation=False`` records contamination -- this is
    permanent and cannot be cleared by a later declared access. Raises
    :class:`KeyError` if ``result_set_id`` was never marked as a holdout
    (there is nothing to declare confirmation *against*).
    """
    if get_holdout_policy(conn, result_set_id) is None:
        raise KeyError(f"result-set:{result_set_id} is not a holdout")
    contamination = not declared_confirmation
    receipt = HoldoutAccessReceipt(
        receipt_id=receipt_id,
        result_set_id=result_set_id,
        accessor_ref=accessor_ref,
        declared_confirmation=declared_confirmation,
        contamination=contamination,
        reason=reason,
        accessed_at_ms=accessed_at_ms,
    )
    conn.execute(
        """
        INSERT INTO holdout_access_receipts (
            receipt_id, result_set_id, accessor_ref, declared_confirmation,
            contamination, reason, accessed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            receipt_id,
            result_set_id,
            accessor_ref,
            int(declared_confirmation),
            int(contamination),
            reason,
            accessed_at_ms,
        ),
    )
    return receipt


def list_holdout_access_receipts(conn: sqlite3.Connection, result_set_id: str) -> tuple[HoldoutAccessReceipt, ...]:
    rows = conn.execute(
        """
        SELECT receipt_id, result_set_id, accessor_ref, declared_confirmation, contamination, reason, accessed_at_ms
        FROM holdout_access_receipts WHERE result_set_id = ? ORDER BY accessed_at_ms
        """,
        (result_set_id,),
    ).fetchall()
    return tuple(
        HoldoutAccessReceipt(
            receipt_id=str(row[0]),
            result_set_id=str(row[1]),
            accessor_ref=str(row[2]),
            declared_confirmation=bool(row[3]),
            contamination=bool(row[4]),
            reason=row[5] if row[5] is None else str(row[5]),
            accessed_at_ms=int(row[6]),
        )
        for row in rows
    )


def has_holdout_contamination(conn: sqlite3.Connection, result_set_id: str) -> bool:
    """Whether any recorded access to this holdout was undeclared -- permanent once true."""
    row = conn.execute(
        "SELECT 1 FROM holdout_access_receipts WHERE result_set_id = ? AND contamination = 1 LIMIT 1",
        (result_set_id,),
    ).fetchone()
    return row is not None


def require_non_holdout_access(conn: sqlite3.Connection, result_set_id: str, *, declared_confirmation: bool) -> None:
    """Planner-level guard: refuse an exploratory read of a holdout relation.

    Raises :class:`HoldoutAccessError` when ``result_set_id`` is a holdout
    and the caller has not declared this as a confirmation access. Not a
    holdout relation at all is always fine (no-op).
    """
    if is_holdout(conn, result_set_id) and not declared_confirmation:
        raise HoldoutAccessError(
            f"result-set:{result_set_id} is a holdout relation; exploratory queries cannot read its "
            "members. Pass declared_confirmation=True (and record the access) for an authorized "
            "confirmation run."
        )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "HoldoutAccessError",
    "HoldoutAccessReceipt",
    "HoldoutPolicy",
    "get_holdout_policy",
    "has_holdout_contamination",
    "is_holdout",
    "list_holdout_access_receipts",
    "mark_holdout",
    "record_holdout_access",
    "require_non_holdout_access",
]
