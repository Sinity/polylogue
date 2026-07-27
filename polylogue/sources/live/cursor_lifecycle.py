"""Declared state machine for the per-file live-ingest cursor lifecycle.

This is the first slice of polylogue-yeq.1 (safety-case + fault-injection
program): the acquisition/cursor-authority lifecycle owned by
``polylogue.sources.live.cursor.CursorStore``. It formally names the states a
live cursor can occupy, the actuators (production methods) that move a cursor
between them, and the exact set of transitions that are safe. The five
remaining hazard areas named in polylogue-yeq.1's design (materialization/
convergence, generation promotion, assertion lifecycle, deletion/excision, and
backup/restore) are NOT covered here -- see that bead's notes for the
residual scope.

Why this area first: ``CursorStore`` already isolated a real production
concurrency defect (polylogue-qug2 / #2467, a lost-update race between two
racing writers on the same ``source_path``) by moving five mutating methods
onto one ``BEGIN IMMEDIATE`` read-modify-write transaction
(``CursorStore._read_modify_write_cursor_record``). That transaction boundary
is exactly the "production seam" a SIGKILL/crash fault needs to straddle, and
the lifecycle it protects is small enough to declare exhaustively.

SIGKILL fault-injection finding (discovered building this slice's harness,
``tests/unit/sources/test_cursor_lifecycle.py``): the REAL durable-commit
boundary is narrower than ``_read_modify_write_cursor_record``'s docstring
implies. ``upsert_ingest_cursor``
(``polylogue/storage/sqlite/archive_tiers/ops_write.py``) issues its own
``conn.commit()`` immediately after the row ``INSERT ... ON CONFLICT DO
UPDATE`` -- so the transaction is durably committed well before the outer
``with conn:`` context manager (several stack frames up, at the end of
``_read_modify_write_cursor_record.write()``) ever exits. This does NOT
reopen the qug2 race (the surrounding ``BEGIN IMMEDIATE`` still holds the
write lock across the read-then-decide window, so a second concurrent
actor's own ``BEGIN IMMEDIATE`` still blocks until this one's commit lands),
but it does mean a crash-fault harness must target the actual
``conn.commit()`` call, not merely "anywhere inside the caller", to observe
an uncommitted intermediate state at all -- confirmed empirically: killing
the process anywhere after ``CursorStore.mark_failed`` returns from its
inner write call always observes the write already durable.

States
------
``ABSENT``        -- no cursor row exists for the source path yet.
``ACTIVE``        -- healthy: not excluded, no pending failure, either no
                     retry timer or a dormant one (content already
                     fingerprinted; see ``DEFERRED`` for the un-fingerprinted
                     case).
``RETRY_PENDING``  -- at least one real parse/persistence failure is
                     recorded and exponential backoff is running
                     (``failure_count > 0``).
``DEFERRED``       -- a full-capture prefix proof could not yet be
                     established (the source was busy/appending); this is a
                     *scheduling* condition, not a failure, and must not
                     consume the finite failure budget
                     (``failure_count == 0``, ``content_fingerprint is
                     None``, ``next_retry_at`` set).
``EXCLUDED``       -- quarantined ("poison pill"): the source will not be
                     retried until the exact failed file observation
                     (byte_size/st_dev/st_ino/mtime_ns) is proved replaced.

Actuators
---------
The five ``CursorStore`` methods that share the locked read-modify-write path:
``mark_failed``, ``mark_excluded``, ``revive_replaced_exclusion``,
``reset_failures``, ``defer_full_cursor_reconciliation``. ``CursorStore.set``
(the ordinary cursor-advance path) is deliberately excluded from this state
machine: it is NOT lock-protected by design (see
``tests/unit/sources/test_cursor_failure_count_race_evidence.py::
test_get_record_then_set_directly_is_still_racy_by_design``), so it has no
transactional boundary to reason about here.

Preventive invariants this table encodes
-----------------------------------------
1. Exclusion is bound to the exact failed file observation, not the pathname:
   only ``revive_replaced_exclusion`` may transition ``EXCLUDED -> ACTIVE``,
   and only when the caller proves a different file identity. No other
   actuator may lift exclusion.
2. ``reset_failures`` clears failure bookkeeping but never exclusion --
   ``(EXCLUDED, "reset_failures", EXCLUDED)`` is the only declared outcome.
   ``reset_failures`` does not accept a proof of file-identity change, so it
   must never be the actuator that revives a quarantined cursor.
3. ``defer_full_cursor_reconciliation`` may not observe/emit ``EXCLUDED`` at
   all: production callers (``sources/live/watcher.py``) only reach it after
   an explicit ``if cursor.excluded: ...`` branch has already returned, so no
   ``(EXCLUDED, "defer_full_cursor_reconciliation", *)`` transition is
   declared. If a future caller (e.g. ``sources/live/batch.py``, which does
   NOT currently gate this call on ``cursor.excluded`` -- see
   ``BatchLiveIngestor._defer_full_cursor_retry``/``_record_full_cursor``)
   ever reaches this actuator with an excluded cursor, ``validate_transition``
   raises rather than silently un-quarantining a poisoned source. This is a
   discovered, not-yet-proven-reachable hazard; see polylogue-yeq.1 notes.

``CursorLifecycleActuator`` is a closed vocabulary (``Literal``) and
``CURSOR_LIFECYCLE_TRANSITIONS`` is a ``frozenset`` of every declared
``(from_state, actuator, to_state)`` triple -- an actuator can have more than
one valid outcome from the same starting state (e.g.
``revive_replaced_exclusion`` from ``EXCLUDED`` can land on ``EXCLUDED``
itself, same identity, or ``ACTIVE``, proved-different identity), so this is
a relation, not a function.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from polylogue.sources.live.cursor import CursorRecord


class CursorLifecycleState(StrEnum):
    """States of the per-file live-ingest cursor lifecycle."""

    ABSENT = "absent"
    ACTIVE = "active"
    RETRY_PENDING = "retry_pending"
    DEFERRED = "deferred"
    EXCLUDED = "excluded"


#: The closed set of production actuators covered by this state machine.
#: Matches ``CursorStore`` method names 1:1 so a production call site can
#: name its own actuator literally.
CursorLifecycleActuator = Literal[
    "mark_failed",
    "mark_excluded",
    "revive_replaced_exclusion",
    "reset_failures",
    "defer_full_cursor_reconciliation",
]

CURSOR_LIFECYCLE_ACTUATORS: tuple[CursorLifecycleActuator, ...] = get_args(CursorLifecycleActuator)


class CursorLifecycleViolationError(RuntimeError):
    """Raised when an actuator produces a transition absent from the declared table.

    This is a fail-closed guard: an undeclared transition is refused inside
    the same locked (``BEGIN IMMEDIATE``) transaction that would otherwise
    have written it, so the write never lands and the cursor keeps its prior
    durable state.
    """


def classify_cursor_lifecycle_state(record: CursorRecord | None) -> CursorLifecycleState:
    """Classify a ``CursorRecord`` (or its absence) into a declared state.

    Mirrors the exact predicates production query paths already use:
    ``CursorStore.list_excluded`` (``excluded = 1``),
    ``CursorStore.list_failed_with_retry``/``list_retry_records``
    (``failure_count > 0`` vs. the ``content_fingerprint IS NULL AND
    next_retry_at IS NOT NULL`` deferred bucket).
    """
    if record is None:
        return CursorLifecycleState.ABSENT
    if bool(record.excluded):
        return CursorLifecycleState.EXCLUDED
    if record.failure_count > 0:
        return CursorLifecycleState.RETRY_PENDING
    if record.content_fingerprint is None and record.next_retry_at is not None:
        return CursorLifecycleState.DEFERRED
    return CursorLifecycleState.ACTIVE


_S = CursorLifecycleState

#: Every declared-safe ``(from_state, actuator, to_state)`` triple. Built from
#: reading each actuator's real implementation in
#: ``polylogue/sources/live/cursor.py`` -- not aspirational; every entry here
#: has a corresponding real code path, and every real code path in scope has
#: an entry (proved by ``test_cursor_lifecycle.py``'s soundness sweep, which
#: drives the actual ``CursorStore`` methods, not a hand-rolled replica).
CURSOR_LIFECYCLE_TRANSITIONS: frozenset[tuple[CursorLifecycleState, CursorLifecycleActuator, CursorLifecycleState]] = (
    frozenset(
        {
            # mark_failed: creates a fresh record on first failure, increments
            # failure_count with exponential backoff, and quarantines once
            # failure_count reaches the exclusion threshold. A record that is
            # already excluded stays excluded (still poisoned).
            (_S.ABSENT, "mark_failed", _S.RETRY_PENDING),
            (_S.ACTIVE, "mark_failed", _S.RETRY_PENDING),
            (_S.DEFERRED, "mark_failed", _S.RETRY_PENDING),
            (_S.RETRY_PENDING, "mark_failed", _S.RETRY_PENDING),
            (_S.RETRY_PENDING, "mark_failed", _S.EXCLUDED),
            (_S.EXCLUDED, "mark_failed", _S.EXCLUDED),
            # mark_excluded: direct quarantine; no-op when no record exists.
            (_S.ABSENT, "mark_excluded", _S.ABSENT),
            (_S.ACTIVE, "mark_excluded", _S.EXCLUDED),
            (_S.RETRY_PENDING, "mark_excluded", _S.EXCLUDED),
            (_S.DEFERRED, "mark_excluded", _S.EXCLUDED),
            (_S.EXCLUDED, "mark_excluded", _S.EXCLUDED),
            # revive_replaced_exclusion: no-op unless currently excluded; even
            # then, no-op unless the caller proves a *different* file
            # identity (byte_size/st_dev/st_ino/mtime_ns) than the poisoned
            # observation. This is the ONLY actuator allowed to clear
            # EXCLUDED.
            (_S.ABSENT, "revive_replaced_exclusion", _S.ABSENT),
            (_S.ACTIVE, "revive_replaced_exclusion", _S.ACTIVE),
            (_S.RETRY_PENDING, "revive_replaced_exclusion", _S.RETRY_PENDING),
            (_S.DEFERRED, "revive_replaced_exclusion", _S.DEFERRED),
            (_S.EXCLUDED, "revive_replaced_exclusion", _S.EXCLUDED),
            (_S.EXCLUDED, "revive_replaced_exclusion", _S.ACTIVE),
            # reset_failures: clears failure_count/next_retry_at only. Never
            # clears `excluded` -- an excluded cursor stays excluded.
            (_S.ABSENT, "reset_failures", _S.ABSENT),
            (_S.ACTIVE, "reset_failures", _S.ACTIVE),
            (_S.RETRY_PENDING, "reset_failures", _S.ACTIVE),
            (_S.DEFERRED, "reset_failures", _S.ACTIVE),
            (_S.EXCLUDED, "reset_failures", _S.EXCLUDED),
            # defer_full_cursor_reconciliation: backs off a busy full-capture
            # handoff without spending the failure budget. Deliberately NOT
            # declared from EXCLUDED (see module docstring point 3) -- an
            # attempt from EXCLUDED is refused by validate_transition rather
            # than silently un-quarantining a poisoned source.
            (_S.ABSENT, "defer_full_cursor_reconciliation", _S.ABSENT),
            (_S.ACTIVE, "defer_full_cursor_reconciliation", _S.ACTIVE),
            (_S.ACTIVE, "defer_full_cursor_reconciliation", _S.DEFERRED),
            (_S.RETRY_PENDING, "defer_full_cursor_reconciliation", _S.ACTIVE),
            (_S.RETRY_PENDING, "defer_full_cursor_reconciliation", _S.DEFERRED),
            (_S.DEFERRED, "defer_full_cursor_reconciliation", _S.DEFERRED),
        }
    )
)


def validate_cursor_lifecycle_transition(
    *,
    actuator: CursorLifecycleActuator,
    before: CursorRecord | None,
    after: CursorRecord | None,
) -> None:
    """Raise :class:`CursorLifecycleViolationError` unless this exact transition is declared.

    ``after`` may be ``None`` to mean "the actuator declined to write" (all
    five actuators can no-op); the effective post-state is then ``before``
    unchanged, matching the ``(state, actuator, state)`` no-op entries above.
    """
    prev_state = classify_cursor_lifecycle_state(before)
    next_state = classify_cursor_lifecycle_state(after if after is not None else before)
    if (prev_state, actuator, next_state) not in CURSOR_LIFECYCLE_TRANSITIONS:
        path = None
        for record in (after, before):
            if record is not None:
                path = record.source_path
                break
        raise CursorLifecycleViolationError(
            f"undeclared cursor lifecycle transition: {prev_state.value} "
            f"--{actuator}--> {next_state.value} (source_path={path!r})"
        )


__all__ = [
    "CURSOR_LIFECYCLE_ACTUATORS",
    "CURSOR_LIFECYCLE_TRANSITIONS",
    "CursorLifecycleActuator",
    "CursorLifecycleState",
    "CursorLifecycleViolationError",
    "classify_cursor_lifecycle_state",
    "validate_cursor_lifecycle_transition",
]
