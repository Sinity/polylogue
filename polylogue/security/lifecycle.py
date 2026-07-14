"""Mirror/primary durable excision lifecycle mechanics (polylogue-27m).

Mirror/primary-mode Sinex-backed excision cannot be locally authoritative --
some other Sinex-backed replica may still hold the content. This module
implements the *local* half of that lifecycle: a durable request/outbox row
in ``user.db`` (never ``ops.db``, so an ``ops reset`` cannot erase it), and
the state machine that drives it against a **fault-injecting versioned
contract fake**. Binding this mechanism to a real Sinex confirmation, purge,
residual, rebuild, and backup proof is explicitly polylogue-303r.6's scope --
this module and its tests prove the *mechanism* only. A contract-fake-only
run never claims a real Sinex purge.

State machine (persisted as ``AssertionKind.EXCISION_REQUEST.value_json``)::

    pending -> acknowledged -> confirmed   (mirror: local hide permitted at
                                             pending; primary: local
                                             invalidation permitted only at
                                             confirmed)
    pending -> rejected                    (terminal; local content is
                                             NEVER invalidated)

A request that hits a simulated network fault stays ``pending`` and its
``attempt_count``/``history`` grow. The durable row is the single source of
truth: a "process restart" -- constructing a fresh
:class:`SinexContractFake` and driving the same ``assertion_id`` again --
recovers the exact same state because nothing about the request lives in
the contract client's memory.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.core.json import JSONValue

LifecycleMode = Literal["mirror", "primary"]
LifecycleState = Literal["pending", "acknowledged", "confirmed", "rejected"]

_TERMINAL_STATES: frozenset[str] = frozenset({"confirmed", "rejected"})


# ---------------------------------------------------------------------------
# Contract protocol + fault-injecting fake
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContractResponse:
    """One contract round-trip outcome.

    ``reachable=False`` models a transport-level fault (simulated network
    loss): the request must stay ``pending`` and retry later, never
    interpreted as a rejection.
    """

    reachable: bool
    outcome: Literal["acknowledged", "rejected", "confirmed"] | None
    detail: str = ""


class ExcisionLifecycleContract(Protocol):
    """Versioned contract a real Sinex client (or a fake) must implement."""

    contract_version: str

    def submit(
        self,
        *,
        request_id: str,
        target_ref: str,
        mode: str,
        reason: str,
        attempt: int,
    ) -> ContractResponse: ...


@dataclass
class SinexContractFake:
    """In-memory fault-injecting fake of the versioned Sinex excision contract.

    Constructing a **new** instance simulates the client process restarting
    (its transport/session state is gone); the fake deliberately shares no
    state with the durable request row in ``user.db``; tests use this to
    prove the row -- not the client -- is authoritative.

    ``drop_next_n`` requests report ``reachable=False`` (simulated network
    loss) before returning real outcomes. ``always_reject`` forces a
    terminal rejection. Otherwise the fake acknowledges on the first
    reachable attempt and confirms on the next.
    """

    contract_version: str = "sinex.excision-lifecycle.v1"
    drop_next_n: int = 0
    always_reject: bool = False
    reject_reason: str = "policy_denied"
    _dropped: int = field(default=0, init=False)
    # Counts only REACHABLE round-trips (never dropped ones) -- a request
    # that hit simulated network loss has not consumed a step of the real
    # ack -> confirm handshake, so a dropped attempt must not skip straight
    # from pending to confirmed once the network recovers.
    _reachable_attempts: int = field(default=0, init=False)

    def submit(
        self,
        *,
        request_id: str,
        target_ref: str,
        mode: str,
        reason: str,
        attempt: int,
    ) -> ContractResponse:
        del request_id, target_ref, mode, reason, attempt  # policy uses only fault-injection state
        if self._dropped < self.drop_next_n:
            self._dropped += 1
            return ContractResponse(reachable=False, outcome=None, detail="simulated network loss")
        if self.always_reject:
            return ContractResponse(reachable=True, outcome="rejected", detail=self.reject_reason)
        reachable_attempt = self._reachable_attempts
        self._reachable_attempts += 1
        if reachable_attempt == 0:
            return ContractResponse(reachable=True, outcome="acknowledged", detail="queued")
        return ContractResponse(reachable=True, outcome="confirmed", detail="")


# ---------------------------------------------------------------------------
# Durable request/outbox row
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LifecycleRequestRow:
    assertion_id: str
    target_ref: str
    mode: LifecycleMode
    state: LifecycleState
    reason: str
    actor: str
    attempt_count: int
    contract_version: str | None
    history: tuple[Mapping[str, JSONValue], ...]


def _request_assertion_id(target_ref: str, mode: str) -> str:
    digest = hashlib.sha256()
    for part in ("excision-request", target_ref, mode):
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return f"assertion-{AssertionKind.EXCISION_REQUEST}:{digest.hexdigest()}"


def submit_lifecycle_request(
    conn: sqlite3.Connection,
    *,
    target_ref: str,
    mode: LifecycleMode,
    reason: str,
    actor: str = "user:local",
    now_ms: int,
) -> str:
    """Create (or return the existing) durable outbox row for this target+mode.

    Idempotent by ``(target_ref, mode)``: a repeated submit for the same
    target returns the same ``assertion_id`` and does not reset its state --
    only :func:`drive_lifecycle_request` advances it.
    """
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope, upsert_assertion

    assertion_id = _request_assertion_id(target_ref, mode)
    if read_assertion_envelope(conn, assertion_id) is not None:
        return assertion_id
    upsert_assertion(
        conn,
        assertion_id=assertion_id,
        target_ref=target_ref,
        kind=AssertionKind.EXCISION_REQUEST,
        value={
            "mode": mode,
            "state": "pending",
            "reason": reason,
            "actor": actor,
            "attempt_count": 0,
            "contract_version": None,
            "history": [],
        },
        author_ref=actor,
        author_kind="user",
        status=AssertionStatus.ACTIVE,
        visibility=AssertionVisibility.PRIVATE,
        context_policy={"inject": False},
        now_ms=now_ms,
    )
    return assertion_id


def read_lifecycle_request(conn: sqlite3.Connection, assertion_id: str) -> LifecycleRequestRow | None:
    from polylogue.storage.sqlite.archive_tiers.user_write import read_assertion_envelope

    envelope = read_assertion_envelope(conn, assertion_id)
    if envelope is None or envelope.kind is not AssertionKind.EXCISION_REQUEST:
        return None
    value = envelope.value if isinstance(envelope.value, dict) else {}
    history_raw = value.get("history")
    history: tuple[Mapping[str, JSONValue], ...] = (
        tuple(entry for entry in history_raw if isinstance(entry, dict)) if isinstance(history_raw, list) else ()
    )
    attempt_count_raw = value.get("attempt_count", 0)
    attempt_count = int(attempt_count_raw) if isinstance(attempt_count_raw, int | float | str) else 0
    return LifecycleRequestRow(
        assertion_id=envelope.assertion_id,
        target_ref=envelope.target_ref,
        mode=str(value.get("mode", "mirror")),  # type: ignore[arg-type]
        state=str(value.get("state", "pending")),  # type: ignore[arg-type]
        reason=str(value.get("reason", "")),
        actor=str(value.get("actor", envelope.author_ref or "user:local")),
        attempt_count=attempt_count,
        contract_version=(str(value["contract_version"]) if value.get("contract_version") else None),
        history=history,
    )


def _update_lifecycle_request(
    conn: sqlite3.Connection,
    row: LifecycleRequestRow,
    *,
    state: LifecycleState,
    attempt_count: int,
    contract_version: str | None,
    history_entry: Mapping[str, JSONValue],
    now_ms: int,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

    upsert_assertion(
        conn,
        assertion_id=row.assertion_id,
        target_ref=row.target_ref,
        kind=AssertionKind.EXCISION_REQUEST,
        value={
            "mode": row.mode,
            "state": state,
            "reason": row.reason,
            "actor": row.actor,
            "attempt_count": attempt_count,
            "contract_version": contract_version or row.contract_version,
            "history": [*row.history, history_entry],
        },
        author_ref=row.actor,
        author_kind="user",
        status=AssertionStatus.ACTIVE,
        visibility=AssertionVisibility.PRIVATE,
        context_policy={"inject": False},
        now_ms=now_ms,
    )


def drive_lifecycle_request(
    conn: sqlite3.Connection,
    contract: ExcisionLifecycleContract,
    assertion_id: str,
    *,
    now_ms: int,
) -> LifecycleRequestRow:
    """Attempt one contract round-trip and persist the resulting state.

    A terminal request (``confirmed``/``rejected``) is a no-op: driving it
    again just returns the current row unchanged. A ``reachable=False``
    response leaves the request ``pending`` (retryable) and never advances
    it to ``rejected`` or ``confirmed`` -- a network fault is not a policy
    decision.
    """
    row = read_lifecycle_request(conn, assertion_id)
    if row is None:
        raise ValueError(f"unknown lifecycle request: {assertion_id}")
    if row.state in _TERMINAL_STATES:
        return row

    response = contract.submit(
        request_id=assertion_id,
        target_ref=row.target_ref,
        mode=row.mode,
        reason=row.reason,
        attempt=row.attempt_count,
    )
    history_entry: dict[str, JSONValue] = {
        "attempt": row.attempt_count,
        "reachable": response.reachable,
        "outcome": response.outcome,
        "detail": response.detail,
        "at_ms": now_ms,
    }
    if not response.reachable:
        next_state: LifecycleState = "pending"
    elif response.outcome == "rejected":
        next_state = "rejected"
    elif response.outcome == "confirmed":
        next_state = "confirmed"
    else:
        next_state = "acknowledged"

    _update_lifecycle_request(
        conn,
        row,
        state=next_state,
        attempt_count=row.attempt_count + 1,
        contract_version=contract.contract_version if response.reachable else None,
        history_entry=history_entry,
        now_ms=now_ms,
    )
    updated = read_lifecycle_request(conn, assertion_id)
    assert updated is not None
    return updated


# ---------------------------------------------------------------------------
# Local invalidation gates
# ---------------------------------------------------------------------------


def mirror_may_hide_locally(row: LifecycleRequestRow) -> bool:
    """Mirror mode: local views may hide the target once a request exists.

    The durable request itself stays pending until acknowledged/confirmed --
    this only governs the *local read-time* hide, never the durable state.
    """
    return row.mode == "mirror" and row.state != "rejected"


def primary_may_invalidate_locally(row: LifecycleRequestRow) -> bool:
    """Primary mode: local replica invalidation is allowed only after confirmation."""
    return row.mode == "primary" and row.state == "confirmed"


@dataclass(frozen=True, slots=True)
class LifecycleInvalidationOutcome:
    success: bool
    reason: str | None = None
    receipt: object | None = None


def apply_primary_invalidation_if_confirmed(
    archive_root: Path,
    conn_user: sqlite3.Connection,
    assertion_id: str,
) -> LifecycleInvalidationOutcome:
    """Invalidate the local replica for a primary-mode request -- confirmed only.

    Rejection can never report success: a rejected (or still-pending, or
    unknown, or mirror-mode) request returns ``success=False`` with an
    explicit reason and never touches the archive. Only a request whose
    durable state is exactly ``confirmed`` proceeds to
    :func:`polylogue.security.excision.apply_session_excision`.
    """
    row = read_lifecycle_request(conn_user, assertion_id)
    if row is None:
        return LifecycleInvalidationOutcome(success=False, reason="unknown_request")
    if row.mode != "primary":
        return LifecycleInvalidationOutcome(success=False, reason="not_primary_mode")
    if row.state == "rejected":
        return LifecycleInvalidationOutcome(success=False, reason="rejected")
    if row.state != "confirmed":
        return LifecycleInvalidationOutcome(success=False, reason="pending_confirmation")

    from polylogue.security.excision import LineageDependentsError, apply_session_excision

    session_id = row.target_ref.removeprefix("session:")
    try:
        receipt = apply_session_excision(archive_root, session_id, reason=row.reason, actor=row.actor)
    except LineageDependentsError:
        # The confirmed target is a prefix-sharing lineage parent: invalidate
        # only via an explicit, out-of-band `--cascade-lineage` decision, not
        # silently as a side effect of a Sinex confirmation landing. Surface
        # this as a clean failure outcome rather than letting the exception
        # propagate out of a lifecycle-drive call site.
        return LifecycleInvalidationOutcome(success=False, reason="lineage_dependents_unresolved")
    if not receipt.found:
        return LifecycleInvalidationOutcome(success=False, reason="target_not_found", receipt=receipt)
    return LifecycleInvalidationOutcome(success=True, receipt=receipt)


__all__ = [
    "ContractResponse",
    "ExcisionLifecycleContract",
    "LifecycleInvalidationOutcome",
    "LifecycleRequestRow",
    "SinexContractFake",
    "apply_primary_invalidation_if_confirmed",
    "drive_lifecycle_request",
    "mirror_may_hide_locally",
    "primary_may_invalidate_locally",
    "read_lifecycle_request",
    "submit_lifecycle_request",
]
