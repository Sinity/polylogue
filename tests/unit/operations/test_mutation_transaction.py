"""Tests for the PREPARE -> AUTHORIZE -> EXECUTE mutation-authority protocol.

Real-dependency anti-vacuity: these tests exercise ``OperationExecutor``
against a minimal-but-real actuator whose ``prepare``/``apply`` mutate a
plain Python list standing in for archive state. Removing the plan-hash
revalidation in ``OperationExecutor.execute`` (deleting the
``fresh_plan.plan_hash != plan.plan_hash`` check) makes
``test_execute_refuses_when_live_state_moved_between_authorize_and_execute``
fail; removing the confirmation-strength floor check in ``authorize`` makes
``test_authorize_refuses_weaker_than_required_confirmation`` fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from polylogue.operations.mutation_transaction import (
    AuthorizationMismatchError,
    ConfirmationRequiredError,
    ConfirmationStrength,
    DestructiveClass,
    MutationAuthorization,
    MutationPlan,
    MutationReceipt,
    OperationExecutor,
    PlanStaleError,
    build_plan,
    compute_plan_hash,
    make_target_ref,
)


@dataclass
class _FakeStore:
    """Mutable state standing in for an archive tier."""

    live_ids: set[str]


@dataclass(frozen=True, slots=True)
class _FakeDeleteArgs:
    store: _FakeStore
    ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FakeDeleteActuator:
    operation: str = "fake-delete"
    destructive_class: DestructiveClass = "delete"
    required_confirmation: ConfirmationStrength = "confirm_flag"

    def prepare(self, args: _FakeDeleteArgs) -> MutationPlan:
        existing: tuple[str, ...] = tuple(sid for sid in args.ids if sid in args.store.live_ids)
        return build_plan(
            operation=self.operation,
            destructive_class="delete",
            target_refs=tuple(make_target_ref("session", sid) for sid in existing),
            affected_tiers=("index",),
            reversible=False,
            context={"ids": list(existing)},
        )

    def apply(self, plan: MutationPlan, args: _FakeDeleteArgs) -> MutationReceipt:
        ids: tuple[str, ...] = tuple(cast("list[str]", plan.context.get("ids") or ()))
        for sid in ids:
            args.store.live_ids.discard(sid)
        return MutationReceipt(
            operation=self.operation,
            plan_hash=plan.plan_hash,
            status="applied" if ids else "already_satisfied",
            target_refs=plan.target_refs,
            affected_count=len(ids),
            detail=None,
            receipt_ref=None,
            applied_at=plan.prepared_at,
        )


def test_plan_hash_is_stable_for_identical_inputs() -> None:
    def _hash() -> str:
        return compute_plan_hash(
            operation="op",
            target_refs=("session:a", "session:b"),
            affected_tiers=("index",),
            destructive_class="delete",
            context={"reason": "x"},
        )

    assert _hash() == _hash()


def test_plan_hash_changes_when_target_set_changes() -> None:
    def _hash(target_refs: tuple[str, ...]) -> str:
        return compute_plan_hash(
            operation="op",
            target_refs=target_refs,
            affected_tiers=("index",),
            destructive_class="delete",
            context={},
        )

    assert _hash(("session:a",)) != _hash(("session:a", "session:b"))


def test_prepare_performs_zero_mutation() -> None:
    store = _FakeStore(live_ids={"a", "b"})
    actuator = _FakeDeleteActuator()
    executor = OperationExecutor()

    plan = executor.prepare(actuator, _FakeDeleteArgs(store=store, ids=("a", "b")))

    assert store.live_ids == {"a", "b"}
    assert plan.target_refs == ("session:a", "session:b")
    assert plan.target_count == 2


def test_full_lifecycle_applies_exactly_the_prepared_targets() -> None:
    store = _FakeStore(live_ids={"a", "b", "c"})
    actuator = _FakeDeleteActuator()
    executor = OperationExecutor()
    args = _FakeDeleteArgs(store=store, ids=("a", "b"))

    plan = executor.prepare(actuator, args)
    authorization = executor.authorize(
        actuator, plan, actor="user:test", role="write", capability="test.delete", confirmation_strength="confirm_flag"
    )
    receipt = executor.execute(actuator, plan, authorization, args)

    assert receipt.status == "applied"
    assert receipt.affected_count == 2
    assert store.live_ids == {"c"}


def test_authorize_refuses_weaker_than_required_confirmation() -> None:
    store = _FakeStore(live_ids={"a"})
    actuator = _FakeDeleteActuator()
    executor = OperationExecutor()
    plan = executor.prepare(actuator, _FakeDeleteArgs(store=store, ids=("a",)))

    with pytest.raises(ConfirmationRequiredError):
        executor.authorize(
            actuator, plan, actor="user:test", role="write", capability="test.delete", confirmation_strength="role_only"
        )


def test_execute_refuses_when_authorization_bound_to_a_different_plan() -> None:
    store = _FakeStore(live_ids={"a", "b"})
    actuator = _FakeDeleteActuator()
    executor = OperationExecutor()
    args = _FakeDeleteArgs(store=store, ids=("a",))
    plan = executor.prepare(actuator, args)
    other_authorization = MutationAuthorization(
        plan_hash="not-the-real-hash",
        actor="user:test",
        role="write",
        capability="test.delete",
        confirmation_strength="confirm_flag",
        authorized_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(AuthorizationMismatchError):
        executor.execute(actuator, plan, other_authorization, args)
    # Nothing mutated.
    assert store.live_ids == {"a", "b"}


def test_execute_refuses_when_live_state_moved_between_authorize_and_execute() -> None:
    """The generalized "excision bypass" class: a stale authorization must refuse.

    This is the structural regression test proving a bound authorization
    cannot be replayed against a target set that has changed since PREPARE
    -- the mechanism that makes a TOCTOU-based bypass of destructive
    authorization impossible by construction (t46.9/kwsb.2).
    """
    store = _FakeStore(live_ids={"a", "b"})
    actuator = _FakeDeleteActuator()
    executor = OperationExecutor()
    args = _FakeDeleteArgs(store=store, ids=("a", "b"))

    plan = executor.prepare(actuator, args)
    authorization = executor.authorize(
        actuator, plan, actor="user:test", role="write", capability="test.delete", confirmation_strength="confirm_flag"
    )

    # Simulate a concurrent actor mutating live state between AUTHORIZE and
    # EXECUTE: one of the two authorized targets is already gone.
    store.live_ids.discard("b")

    with pytest.raises(PlanStaleError):
        executor.execute(actuator, plan, authorization, args)
    # The still-live target ("a") was NOT deleted by the refused execute --
    # a bypass would have silently applied a subset mutation instead.
    assert store.live_ids == {"a"}


def test_execute_succeeds_when_live_state_unchanged() -> None:
    store = _FakeStore(live_ids={"a"})
    actuator = _FakeDeleteActuator()
    executor = OperationExecutor()
    args = _FakeDeleteArgs(store=store, ids=("a",))

    plan = executor.prepare(actuator, args)
    authorization = executor.authorize(
        actuator, plan, actor="user:test", role="write", capability="test.delete", confirmation_strength="confirm_flag"
    )
    receipt = executor.execute(actuator, plan, authorization, args)

    assert receipt.status == "applied"
    assert store.live_ids == set()
