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
    MutationTarget,
    OperationExecutor,
    PlanStaleError,
    build_plan,
    build_typed_plan,
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


def test_build_plan_refuses_over_256_targets() -> None:
    """A mutation plan over the recovery-adjudication target budget must never be constructed.

    ``AuditRepository.adjudicate_recovery`` refuses any target list over 256
    as a bounded-command budget; a mutation plan with more targets than that
    has no valid adjudication request if it is ever interrupted, wedging it
    forever. Enforcing the same cap at plan-construction time keeps that
    from ever happening.

    Anti-vacuity: removing the ``len(target_refs) > MAX_MUTATION_PLAN_TARGETS``
    guard in ``build_plan`` makes this test fail -- 257 targets would build a
    plan instead of raising, and a plan of exactly 256 would also need to
    keep succeeding (checked below) so the fix doesn't over-tighten the cap.
    """

    too_many = tuple(f"session:{i}" for i in range(257))
    with pytest.raises(ValueError, match="256"):
        build_plan(
            operation="mutate-bulk-fixture",
            destructive_class="delete",
            target_refs=too_many,
            affected_tiers=("index",),
            reversible=False,
        )
    exactly_budget = tuple(f"session:{i}" for i in range(256))
    plan = build_plan(
        operation="mutate-bulk-fixture",
        destructive_class="delete",
        target_refs=exactly_budget,
        affected_tiers=("index",),
        reversible=False,
    )
    assert plan.target_count == 256


def test_build_typed_plan_refuses_over_256_targets() -> None:
    """Typed plans cannot bypass the bounded recovery-adjudication budget."""

    targets = tuple(
        MutationTarget(
            kind="session",
            ref=f"session:{index}",
            policy_key="session",
            identity_digest=f"identity:{index}",
            effect_identity=f"effect:{index}",
            durability="derived",
            recovery="none",
        )
        for index in range(257)
    )
    with pytest.raises(ValueError, match="256"):
        build_typed_plan(
            operation="mutate-bulk-fixture",
            operation_version=1,
            archive_instance_id="archive:test",
            archive_identity_digest="identity:test",
            targets=targets,
            affected_tiers=("index",),
            parameter_digest="params:test",
            required_capabilities=("test.mutate",),
            destructive_class="reversible",
            required_confirmation="role_only",
            prepared_at_ms=1,
            expires_at_ms=2,
        )


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
