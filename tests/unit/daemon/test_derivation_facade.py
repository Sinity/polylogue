"""``DaemonConverger`` drives derivations without becoming a second engine.

The facade keeps its name and its callers; what changed is that a migrated
domain converges from its own output relation instead of from stage state. The
laws here are the ones that would let a second engine creep back in: a facade
that caches the pending set, that wraps computation in an outer lease, or that
lets stage state certify a derived output.

The one thing the facade may carry between calls is a resume position, and
these tests pin what that is allowed to be: scheduling fairness, never
authority, and never durable.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import cast

import pytest

from polylogue.daemon.convergence import DaemonConverger, SessionProfileConvergenceOwner
from polylogue.daemon.derivation import (
    BaseDerivation,
    Budget,
    DerivationFrame,
    DerivationKey,
    Outcome,
    PendingReason,
    Replacement,
)
from polylogue.daemon.execution import BoundedComputeAdapter
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge

FRAME = DerivationFrame(archive_root="/archive", source_revision="r1")


class StringStatusDerivation(BaseDerivation):
    """A domain speaking the string vocabulary, as storage-side adapters do.

    Storage may not import the daemon ring, so an adapter there returns status
    values rather than enum members. This proves the kernel accepts that.
    """

    domain = "strings"
    prerequisites: tuple[str, ...] = ()

    def __init__(self, keys: Sequence[str] = ("a", "b"), *, publish_refuses: frozenset[str] = frozenset()) -> None:
        self.output: dict[str, str] = {}
        self.binding = dict.fromkeys(keys, "b0")
        self.publish_refuses = publish_refuses
        self.leases: list[str] = []

    def required_keys(self, frame: DerivationFrame) -> Iterable[str]:
        return iter(self.binding)

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, str]:
        return {
            key: "valid"
            if self.output.get(key) == self.binding[key]
            else ("missing" if key not in self.output else "stale")
            for key in keys
        }

    def compute(self, frame: DerivationFrame, key: str) -> Replacement:
        return Replacement(key=DerivationKey(self.domain, key), input_binding=self.binding[key], payload=key)

    def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
        key = str(replacement.payload)
        self.leases.append(key)
        if key in self.publish_refuses:
            return False
        self.output[key] = replacement.input_binding
        return True


def test_a_converger_with_no_derivations_reports_no_domains() -> None:
    assert DaemonConverger([]).derivation_domains == ()


def test_the_facade_converges_a_registered_domain_and_reports_its_order() -> None:
    adapter = StringStatusDerivation()
    converger = DaemonConverger([], derivations=[adapter])

    assert converger.derivation_domains == ("strings",)
    report = converger.converge_derivations(FRAME)
    assert report.done == 2
    assert sorted(adapter.output) == ["a", "b"]


def test_the_facade_routes_only_publication_through_the_owner_admission() -> None:
    """Compute stays in the caller while each publish uses the injected bridge.

    Anti-vacuity: acquire the writer around ``converge_derivations`` or ignore
    ``publisher`` and this records no per-key publication admission.
    """
    adapter = StringStatusDerivation()
    converger = DaemonConverger([], derivations=[adapter])
    admissions: list[str] = []

    def publisher(domain: str, publish: object) -> bool:
        admissions.append(domain)
        return bool(cast(Callable[[], bool], publish)())

    assert converger.converge_derivations(FRAME, publisher=publisher).done == 2
    assert admissions == ["strings", "strings"]


def test_a_required_key_that_disappears_after_discovery_is_binding_moved() -> None:
    """Correct retirement during a required pass is pending, not a false failure.

    Anti-vacuity: classify a post-publish MISSING relation as FAILED and this
    loses the retryable moved-input distinction although the old output was
    correctly removed.
    """

    class VanishingDerivation(BaseDerivation):
        domain = "vanishing"
        prerequisites: tuple[str, ...] = ()

        def required_keys(self, frame: DerivationFrame) -> Iterable[str]:
            return ("gone",)

        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, str]:
            return dict.fromkeys(keys, "missing")

        def compute(self, frame: DerivationFrame, key: str) -> Replacement:
            return Replacement(key=DerivationKey(self.domain, key), input_binding="before", payload=key)

        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
            return True

    report = DaemonConverger([], derivations=[VanishingDerivation()]).converge_derivations(FRAME)
    outcome = report.by_outcome(Outcome.PENDING)
    assert len(outcome) == 1
    assert outcome[0].reason is PendingReason.BINDING_MOVED


@pytest.mark.asyncio
async def test_session_owner_keeps_archive_resume_but_restarts_targeted_scope() -> None:
    """A targeted earlier id cannot inherit an archive sweep's page cursor.

    Anti-vacuity: pass ``resume=True`` through an incremental scope and the
    archive pass below leaves its cursor after ``a``; the targeted repair of
    ``a`` is then skipped despite the output relation reporting it missing.
    """
    adapter = StringStatusDerivation(("a", "b"))
    adapter.domain = "session_profile"
    converger = DaemonConverger([], derivations=[adapter])
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = SessionProfileConvergenceOwner(
        converger,
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )
    try:
        archive = DerivationFrame(archive_root="/archive", source_revision="r1")
        assert (await owner.converge(archive, budget=Budget(page=1, compute=1))).done == 1
        assert adapter.output == {"a": "b0"}

        adapter.output.pop("a")
        targeted = DerivationFrame(archive_root="/archive", source_revision="r2", scope=("a",))
        assert (await owner.converge(targeted, budget=Budget(page=1, compute=1))).done == 1
        assert adapter.output == {"a": "b0"}
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_session_owner_cancellation_waits_for_an_admitted_publication() -> None:
    """Cancellation cannot abandon a worker holding the bridged writer gate.

    Anti-vacuity: return immediately from ``CancelledError`` and this task
    finishes while the publisher below is still active, allowing composition
    shutdown to treat the writer as drained when it is not.
    """

    class BlockingDerivation(StringStatusDerivation):
        domain = "session_profile"

        def __init__(self) -> None:
            super().__init__(("a",))
            self.started = threading.Event()
            self.release = threading.Event()

        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
            self.started.set()
            assert self.release.wait(timeout=2.0)
            return super().publish(frame, replacement)

    adapter = BlockingDerivation()
    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator()
    owner = SessionProfileConvergenceOwner(
        DaemonConverger([], derivations=[adapter]),
        compute_adapter=compute,
        write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
    )
    task = asyncio.create_task(owner.converge(DerivationFrame(archive_root="/archive", source_revision="r1")))
    try:
        assert await asyncio.to_thread(adapter.started.wait, 1.0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert coordinator.snapshot().active_actor == "derivation.session_profile"

        adapter.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert coordinator.snapshot().active_actor is None
    finally:
        adapter.release.set()
        if not task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


def test_the_facade_reconstructs_the_pending_set_on_every_call() -> None:
    """A restart loses nothing because the facade stores nothing.

    Anti-vacuity: memoize the pending set on the converger between calls and the
    second pass here reports work that the output relation says is already
    valid, or misses work that appeared after the cache was filled.
    """
    adapter = StringStatusDerivation()
    converger = DaemonConverger([], derivations=[adapter])
    converger.converge_derivations(FRAME)

    assert converger.converge_derivations(FRAME).wrote_nothing

    adapter.output.clear()
    fresh_converger = DaemonConverger([], derivations=[adapter])
    assert fresh_converger.converge_derivations(FRAME).done == 2


def test_a_budget_bounds_the_facade_pass_without_losing_the_remainder() -> None:
    adapter = StringStatusDerivation()
    converger = DaemonConverger([], derivations=[adapter])

    assert converger.converge_derivations(FRAME, budget=1).done == 1
    assert converger.converge_derivations(FRAME, budget=1).done == 1
    assert converger.converge_derivations(FRAME).wrote_nothing


def test_the_facade_resumes_where_the_last_bounded_pass_stopped() -> None:
    """The carried position is what stops a stuck head from starving the tail.

    The first two keys always refuse publication, so they consume the pass's
    compute budget and publish nothing. Anti-vacuity: drop the cursor the facade
    carries -- ``resume=False``, as the contrast run does -- and every pass
    re-spends its whole budget on the same two keys forever.
    """
    keys = tuple(f"k{index}" for index in range(6))
    stuck = frozenset(keys[:2])
    budget = Budget(page=2, compute=2)

    starved = StringStatusDerivation(keys, publish_refuses=stuck)
    unresumed = DaemonConverger([], derivations=[starved])
    for _ in range(6):
        unresumed.converge_derivations(FRAME, budget=budget, resume=False)
    assert starved.output == {}

    fair = StringStatusDerivation(keys, publish_refuses=stuck)
    resumed = DaemonConverger([], derivations=[fair])
    for _ in range(6):
        resumed.converge_derivations(FRAME, budget=budget)
    assert sorted(fair.output) == list(keys[2:])


def test_a_restart_drops_the_resume_position_without_dropping_work() -> None:
    """The position is process-local; losing it costs a sweep, not a key."""
    adapter = StringStatusDerivation(("a", "b", "c"))
    first = DaemonConverger([], derivations=[adapter])
    first.converge_derivations(FRAME, budget=1)

    restarted = DaemonConverger([], derivations=[adapter])
    assert restarted.converge_derivations(FRAME).done == 2
    assert sorted(adapter.output) == ["a", "b", "c"]


def test_the_facade_can_select_one_domain() -> None:
    first = StringStatusDerivation()
    second = StringStatusDerivation()
    second.domain = "other"
    converger = DaemonConverger([], derivations=[first, second])

    converger.converge_derivations(FRAME, domains=["strings"])
    assert first.output and not second.output


def test_stage_state_cannot_certify_a_derived_output() -> None:
    """Stages and derivations are separate relations; neither reads the other.

    Anti-vacuity: have ``converge_derivations`` skip a domain because the stage
    map says the batch converged, and a derived table that never published
    reports as current.
    """
    adapter = StringStatusDerivation()
    converger = DaemonConverger([], derivations=[adapter])
    converger.converge_batch([], whole_archive=False)

    assert converger.converge_derivations(FRAME).done == 2
