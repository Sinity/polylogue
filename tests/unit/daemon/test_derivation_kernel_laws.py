"""The laws every derivation obeys, stated once over synthetic domains.

These are architecture-neutral: they constrain the kernel's relation between
``required_page``, ``inspect``, ``prerequisite_keys``, ``compute`` and
``publish``, not any domain's SQL. A domain adapter that passes its own tests
but breaks one of these is wrong.

Two families live here. The first is correctness: what the pending set is, what
may publish, and what a publication has to prove. The second is cost: a pass
over a domain far larger than its budget must pay for the page it looked at and
nothing more, and repeated bounded passes must reach every key.

Anti-vacuity is named per test: each says which mutation of the kernel makes it
red. A law nobody can break is not a law.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import pytest

from polylogue.daemon.derivation import (
    BaseDerivation,
    Budget,
    DerivationFrame,
    DerivationKey,
    DerivationRegistry,
    DerivationReport,
    KeyPage,
    KeyStatus,
    Outcome,
    PendingReason,
    Replacement,
    converge,
)

FRAME = DerivationFrame(archive_root="/archive", source_revision="r1")


class RecordingDerivation(BaseDerivation):
    """A domain whose whole state is an in-memory output relation.

    The output dict is the authority, exactly as a derived table is: validity is
    read back from it, never from a flag the test sets alongside it.
    """

    def __init__(
        self,
        domain: str,
        *,
        required: Sequence[str],
        prerequisites: tuple[str, ...] = (),
        bindings: Mapping[str, Sequence[tuple[str, str]]] | None = None,
        binding: Mapping[str, str] | None = None,
        poison: frozenset[str] = frozenset(),
        quiet_keys: frozenset[str] = frozenset(),
        publish_refuses: frozenset[str] = frozenset(),
    ) -> None:
        self.domain = domain
        self.prerequisites = prerequisites
        self._required = tuple(required)
        self._bindings = {key: tuple(value) for key, value in (bindings or {}).items()}
        self.binding = dict(binding or dict.fromkeys(required, "b0"))
        self.output: dict[str, str] = {}
        self.poison = poison
        self.quiet_keys = quiet_keys
        self.publish_refuses = set(publish_refuses)
        self.computed: list[str] = []
        self.published: list[str] = []

    def required_keys(self, frame: DerivationFrame) -> Iterable[str]:
        return iter(self._required)

    def excess_keys(self, frame: DerivationFrame) -> Iterable[str]:
        return (key for key in tuple(self.output) if key not in self._required)

    def prerequisite_keys(self, frame: DerivationFrame, key: str) -> Iterable[tuple[str, str]]:
        return self._bindings.get(key, ())

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
        statuses: dict[str, KeyStatus] = {}
        for key in keys:
            if key not in self.output:
                statuses[key] = KeyStatus.MISSING
            elif key not in self._required:
                statuses[key] = KeyStatus.EXCESS
            elif self.output[key] != self.binding.get(key, ""):
                statuses[key] = KeyStatus.STALE
            else:
                statuses[key] = KeyStatus.VALID
        return statuses

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        return key in self.quiet_keys

    def compute(self, frame: DerivationFrame, key: str) -> Replacement:
        self.computed.append(key)
        if key in self.poison:
            raise RuntimeError(f"poison input for {key}")
        if key not in self._required:
            return Replacement(key=DerivationKey(self.domain, key), input_binding="", payload=key, empty=True)
        return Replacement(key=DerivationKey(self.domain, key), input_binding=self.binding[key], payload=key)

    def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
        key = str(replacement.payload)
        if key in self.publish_refuses:
            return False
        self.published.append(key)
        if replacement.empty and key not in self._required:
            self.output.pop(key, None)
        else:
            self.output[key] = replacement.input_binding
        return True


class VirtualDomain(BaseDerivation):
    """A domain whose key space is generated on demand and never held.

    ``required_page`` is a keyset scan over a space orders of magnitude larger
    than any budget here, and it counts the keys it hands out. A kernel that
    enumerated the space to decide what to do would show up in
    ``materialized_keys``, not merely in the wall clock.
    """

    domain = "virtual"
    prerequisites: tuple[str, ...] = ()

    def __init__(self, total: int, *, refusing: int = 0, quiet_prefix: int = 0) -> None:
        self.total = total
        self.refusing = refusing
        self.quiet_prefix = quiet_prefix
        self.output: dict[str, str] = {}
        self.calls: Counter[str] = Counter()
        self.materialized_keys = 0
        self.inspected_keys = 0

    @staticmethod
    def key_at(index: int) -> str:
        return f"k-{index:05d}"

    @staticmethod
    def index_of(key: str) -> int:
        return int(key.split("-")[1])

    def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
        self.calls["required_page"] += 1
        start = int(cursor) if cursor else 0
        stop = min(start + limit, self.total)
        keys = tuple(self.key_at(index) for index in range(start, stop))
        self.materialized_keys += len(keys)
        return KeyPage(keys, str(stop) if stop < self.total else None)

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
        self.calls["inspect"] += 1
        self.inspected_keys += len(keys)
        return {key: (KeyStatus.VALID if key in self.output else KeyStatus.MISSING) for key in keys}

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        return self.index_of(key) < self.quiet_prefix

    def compute(self, frame: DerivationFrame, key: str) -> Replacement:
        self.calls["compute"] += 1
        return Replacement(key=DerivationKey(self.domain, key), input_binding="b0", payload=key)

    def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
        self.calls["publish"] += 1
        key = str(replacement.payload)
        if self.index_of(key) < self.refusing:
            return False
        self.output[key] = replacement.input_binding
        return True


def _drive(
    registry: DerivationRegistry,
    *,
    budget: Budget,
    passes: int,
    resume: bool,
) -> DerivationReport:
    """Run repeated bounded passes, optionally carrying the resume position."""
    cursor = None
    report = DerivationReport(frame=FRAME)
    for _ in range(passes):
        report = converge(registry, FRAME, budget=budget, cursor=cursor)
        if resume:
            cursor = report.cursor
    return report


# ── the pending set ────────────────────────────────────────────────


def test_pending_is_required_minus_valid_and_a_second_pass_writes_nothing() -> None:
    """Inspection is authoritative, so an unchanged second pass is a no-op.

    Anti-vacuity: have ``run_domain`` process every discovered key regardless of
    status, or have ``inspect`` report MISSING for a present output, and the
    zero-write assertion fails.
    """
    adapter = RecordingDerivation("d", required=("a", "b", "c"))
    registry = DerivationRegistry([adapter])

    first = converge(registry, FRAME)
    assert first.done == 3
    assert adapter.published == ["a", "b", "c"]

    second = converge(registry, FRAME)
    assert second.wrote_nothing
    assert second.outcomes == ()
    assert adapter.published == ["a", "b", "c"]


def test_valid_empty_output_is_not_missing_work() -> None:
    """A key whose correct output is empty converges once and stays valid.

    Anti-vacuity: drop the stored binding for an empty key -- treat "no rows" as
    "never computed" -- and the second pass republishes it.
    """
    adapter = RecordingDerivation("d", required=("empty",), binding={"empty": ""})
    registry = DerivationRegistry([adapter])

    assert converge(registry, FRAME).done == 1
    assert adapter.output == {"empty": ""}
    assert converge(registry, FRAME).wrote_nothing


def test_a_deleted_scheduling_hint_cannot_change_the_pending_set() -> None:
    """The pending set is derived; there is no queue to lose.

    Anti-vacuity: cache pending keys on the registry between passes and this
    test still passes -- so it also asserts the recomputed set after the output
    is cleared, which a cache would get wrong.
    """
    adapter = RecordingDerivation("d", required=("a", "b"))
    registry = DerivationRegistry([adapter])
    converge(registry, FRAME)

    adapter.output.pop("a")
    report = converge(registry, FRAME)
    assert report.done == 1
    assert adapter.published[-1] == "a"


def test_stale_binding_is_pending_even_though_the_output_row_exists() -> None:
    """Presence is not validity: a moved input value makes a present row stale.

    Anti-vacuity: compare only key presence in ``inspect`` and this goes green
    while the stale output is served forever.
    """
    adapter = RecordingDerivation("d", required=("a",))
    registry = DerivationRegistry([adapter])
    converge(registry, FRAME)

    adapter.binding["a"] = "b1"
    report = converge(registry, FRAME)
    assert report.done == 1
    assert adapter.output["a"] == "b1"


def test_excess_output_is_retired_and_certified_as_absent() -> None:
    """A key the output holds but ``required`` no longer names is retired.

    Anti-vacuity: certify a retirement with VALID like any other publication and
    a correct deletion reports FAILED forever; skip the excess phase entirely and
    the row is never noticed at all.
    """
    adapter = RecordingDerivation("d", required=("a",))
    registry = DerivationRegistry([adapter])
    converge(registry, FRAME)
    adapter.output["gone"] = "b0"

    report = converge(registry, FRAME)
    assert report.done == 1
    assert [str(item.key) for item in report.by_outcome(Outcome.DONE)] == ["d:gone"]
    assert "gone" not in adapter.output
    assert converge(registry, FRAME).wrote_nothing


# ── what may publish ───────────────────────────────────────────────


def test_one_poison_key_blocks_only_its_dependent_closure() -> None:
    """A failing key fails alone; its siblings converge in the same pass.

    Anti-vacuity: mark the whole domain failed on the first exception -- the
    stage-era behaviour -- and ``b``/``c`` never publish.
    """
    adapter = RecordingDerivation("d", required=("a", "b", "c"), poison=frozenset({"a"}))
    registry = DerivationRegistry([adapter])

    report = converge(registry, FRAME)
    assert report.failed == 1
    assert sorted(adapter.published) == ["b", "c"]
    failed = report.by_outcome(Outcome.FAILED)[0]
    assert failed.key.key == "a"
    assert failed.error is not None and "poison" in failed.error


def test_a_poisoned_prerequisite_blocks_its_dependants_and_nothing_else() -> None:
    """Order comes from declared inputs, so an unrelated domain still runs."""
    upstream = RecordingDerivation("up", required=("a",), poison=frozenset({"a"}))
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",))
    unrelated = RecordingDerivation("side", required=("y",))
    registry = DerivationRegistry([upstream, downstream, unrelated])

    report = converge(registry, FRAME)
    assert unrelated.published == ["y"]
    assert downstream.published == []
    blocked = [item for item in report.outcomes if item.reason is PendingReason.BLOCKED]
    assert [item.key.domain for item in blocked] == ["down"]


def test_a_quiet_prerequisite_key_blocks_its_dependant_and_not_its_sibling() -> None:
    """Dependency is per key, so one deferred input stops exactly what reads it.

    Anti-vacuity: gate on the declared domain edge instead of the key bindings --
    the coarse rule the stage era used -- and ``y`` is blocked with ``x``; drop
    the binding check entirely and ``x`` publishes over an input that was never
    computed.
    """
    upstream = RecordingDerivation("up", required=("a", "b"), quiet_keys=frozenset({"a"}))
    downstream = RecordingDerivation(
        "down",
        required=("x", "y"),
        prerequisites=("up",),
        bindings={"x": (("up", "a"),), "y": (("up", "b"),)},
    )
    registry = DerivationRegistry([upstream, downstream])

    report = converge(registry, FRAME)
    assert upstream.published == ["b"]
    assert downstream.published == ["y"]
    blocked = [item for item in report.outcomes if item.reason is PendingReason.BLOCKED]
    assert [str(item.key) for item in blocked] == ["down:x"]
    assert blocked[0].error is not None and "up:a" in blocked[0].error


def test_a_refused_prerequisite_publication_blocks_its_dependant_in_the_same_pass() -> None:
    """A lost publication race upstream is not a licence to publish downstream.

    Anti-vacuity: read the prerequisite's status only from the output relation
    and ignore what this pass just did, and ``x`` is computed from the pre-race
    value the upstream key still holds.
    """
    upstream = RecordingDerivation("up", required=("a",), publish_refuses=frozenset({"a"}))
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",), bindings={"x": (("up", "a"),)})
    registry = DerivationRegistry([upstream, downstream])

    report = converge(registry, FRAME)
    assert downstream.published == []
    blocked = [item for item in report.outcomes if item.reason is PendingReason.BLOCKED]
    assert [str(item.key) for item in blocked] == ["down:x"]


def test_a_prerequisite_outside_the_selected_domains_is_inspected_either_way() -> None:
    """Selection is not a claim about inputs: the binding decides, not the roster.

    Anti-vacuity: resolve bindings only against domains selected this pass and
    the first half publishes on an input that does not exist; skip the
    inspection and treat unselected as settled, and both halves publish.
    """
    upstream = RecordingDerivation("up", required=("a",))
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",), bindings={"x": (("up", "a"),)})
    registry = DerivationRegistry([upstream, downstream])

    blocked_pass = converge(registry, FRAME, domains=["down"])
    assert downstream.published == []
    assert blocked_pass.work.prerequisites_inspected == 1
    assert [item.reason for item in blocked_pass.by_outcome(Outcome.PENDING)] == [PendingReason.BLOCKED]

    converge(registry, FRAME, domains=["up"])
    permitted_pass = converge(registry, FRAME, domains=["down"])
    assert downstream.published == ["x"]
    assert permitted_pass.done == 1
    assert permitted_pass.work.prerequisites_inspected == 1


def test_a_bounded_pass_never_lets_a_dependant_run_ahead_of_its_prerequisite() -> None:
    """A bound stops the pass; it does not license publishing on pending input.

    Anti-vacuity: spend the budget in registration order instead of prerequisite
    order, or exempt a dependant from the budget check, and ``down`` publishes in
    the same pass that left its input unconverged.
    """
    upstream = RecordingDerivation("up", required=("a",))
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",), bindings={"x": (("up", "a"),)})
    registry = DerivationRegistry([upstream, downstream])

    first = converge(registry, FRAME, budget=1)
    assert upstream.published == ["a"]
    assert downstream.published == []
    assert first.done == 1

    second = converge(registry, FRAME)
    assert downstream.published == ["x"]
    assert second.done == 1


def test_publish_refusal_is_pending_not_failed() -> None:
    """A binding that moved under the computation reschedules; it is not an error.

    Anti-vacuity: map a False publish onto FAILED and the two become
    indistinguishable, which is how a lost race got reported as a defect.
    """
    adapter = RecordingDerivation("d", required=("a",), publish_refuses=frozenset({"a"}))
    registry = DerivationRegistry([adapter])

    report = converge(registry, FRAME)
    assert report.failed == 0
    assert report.pending == 1
    assert report.by_outcome(Outcome.PENDING)[0].reason is PendingReason.BINDING_MOVED
    assert adapter.output == {}


def test_a_publication_the_output_relation_does_not_confirm_is_a_failure() -> None:
    """Publishing reports a claim; the output relation certifies it.

    Anti-vacuity: delete the post-publication inspection in ``_Pass.process`` and
    a domain that writes nothing while returning True reports DONE, which is the
    precise shape of "converged" that never converged.
    """

    class LyingPublication(RecordingDerivation):
        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
            key = str(replacement.payload)
            self.published.append(key)
            # A publisher may claim success while persisting the wrong input
            # binding.  The authoritative relation must expose that as stale;
            # merely returning True must not certify the replacement.
            self.output[key] = "wrong-binding"
            return True

    adapter = LyingPublication("d", required=("a",))
    registry = DerivationRegistry([adapter])

    report = converge(registry, FRAME)
    assert report.done == 0
    assert report.failed == 1
    failure = report.by_outcome(Outcome.FAILED)[0]
    assert failure.error is not None and "stale" in failure.error
    assert adapter.output == {"a": "wrong-binding"}


def test_a_successful_publication_that_leaves_a_required_output_missing_fails() -> None:
    """A success claim cannot turn a still-required missing output into backlog.

    Anti-vacuity: classify ``MISSING`` after a successful required publication
    as ``BINDING_MOVED`` and the kernel conceals a lying publisher as an
    ordinary retry.
    """

    class LyingPublication(RecordingDerivation):
        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
            self.published.append(str(replacement.payload))
            return True

    adapter = LyingPublication("d", required=("a",))
    report = converge(DerivationRegistry([adapter]), FRAME)

    assert report.done == 0
    assert report.pending == 0
    assert report.failed == 1
    failure = report.by_outcome(Outcome.FAILED)[0]
    assert failure.error is not None and "required output remains missing" in failure.error
    assert adapter.output == {}


def test_quiet_policy_defers_a_key_without_certifying_it() -> None:
    """A deferred key stays pending, so the next pass still finds it."""
    adapter = RecordingDerivation("d", required=("hot", "cold"), quiet_keys=frozenset({"hot"}))
    registry = DerivationRegistry([adapter])

    report = converge(registry, FRAME)
    assert adapter.published == ["cold"]
    assert [item.reason for item in report.by_outcome(Outcome.PENDING)] == [PendingReason.QUIET]

    adapter.quiet_keys = frozenset()
    assert converge(registry, FRAME).done == 1


def test_compute_never_runs_for_a_key_inspection_calls_valid() -> None:
    """Work is selected from inspection, not refused after it is computed."""
    adapter = RecordingDerivation("d", required=("a",))
    registry = DerivationRegistry([adapter])
    converge(registry, FRAME)
    adapter.computed.clear()

    converge(registry, FRAME)
    assert adapter.computed == []


def test_an_inspection_failure_blocks_its_dependants_and_reports_the_domain() -> None:
    """Inspection is the correctness relation, so losing it is not silence."""

    class BrokenInspection(RecordingDerivation):
        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            raise RuntimeError("output relation unreadable")

    broken = BrokenInspection("up", required=("a",))
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",))
    registry = DerivationRegistry([broken, downstream])

    report = converge(registry, FRAME)
    assert report.failed == 1
    assert downstream.published == []


# ── bounded work ───────────────────────────────────────────────────


def test_a_bounded_pass_over_a_huge_domain_costs_its_page_and_not_the_domain() -> None:
    """Ten thousand keys, unit budgets: one key is discovered and one is written.

    Anti-vacuity: drop the ``page_limit`` clamp, or enumerate ``required`` before
    inspecting it the way the pre-paging kernel did, and ``materialized_keys``
    becomes 10,000 while the report grows a result object per unseen key.
    """
    domain = VirtualDomain(10_000)
    registry = DerivationRegistry([domain])

    report = converge(registry, FRAME, budget=Budget(page=1, discovery=1, publication=1))

    assert report.done == 1
    assert domain.materialized_keys == 1
    assert domain.calls["required_page"] == 1
    # One classification of the discovered page, plus the inspection that
    # certifies the publication -- and nothing proportional to the domain.
    assert domain.inspected_keys == 2
    assert domain.calls["compute"] == 1
    assert len(report.outcomes) == 1
    assert report.work.discovered == 1
    assert report.work.published == 1


@pytest.mark.parametrize("total", [1_000, 10_000, 100_000])
def test_the_cost_of_one_bounded_pass_does_not_move_with_the_size_of_the_domain(total: int) -> None:
    """The same budget buys the same work over any key space.

    Anti-vacuity: make discovery or reporting proportional to ``required`` and
    the asserted counters diverge across the parametrization.
    """
    domain = VirtualDomain(total)
    registry = DerivationRegistry([domain])

    report = converge(registry, FRAME, budget=Budget(page=4, publication=2))

    assert domain.materialized_keys == 4
    assert domain.calls["required_page"] == 1
    assert report.done == 2
    assert report.pending == 2
    assert len(report.outcomes) == 4
    assert {item.reason for item in report.by_outcome(Outcome.PENDING)} == {PendingReason.BUDGET}


def test_a_report_can_be_bounded_without_losing_its_totals() -> None:
    """Retention bounds the report; the counters still describe the whole pass.

    Anti-vacuity: derive ``done``/``pending`` from the retained outcomes and the
    totals silently become "the first three things that happened".
    """
    domain = VirtualDomain(10_000)
    registry = DerivationRegistry([domain])

    report = converge(registry, FRAME, budget=Budget(page=100, publication=5, retained_outcomes=3))

    assert report.done == 5
    assert report.pending == 95
    assert len(report.outcomes) == 3
    assert report.truncated


def test_repeated_bounded_passes_visit_every_key() -> None:
    """A budget delays work; it never drops it.

    Anti-vacuity: reset the domain position to the start of the space on every
    pass and the tail of the domain is never reached while a refusing prefix
    exists; stop wrapping a finished sweep and a key that becomes stale after its
    sweep is never revisited.
    """
    domain = VirtualDomain(64)
    registry = DerivationRegistry([domain])

    _drive(registry, budget=Budget(page=4, publication=2), passes=64, resume=True)

    assert len(domain.output) == 64
    assert converge(registry, FRAME, budget=Budget(page=4, publication=2)).wrote_nothing


def test_a_permanently_refusing_prefix_cannot_starve_the_keys_behind_it() -> None:
    """The resume position is what makes "eventually every key" true.

    Anti-vacuity: ignore the passed-in cursor in ``converge`` and the resumed run
    converges nothing, exactly like the unresumed one this test contrasts it
    with.
    """
    budget = Budget(page=4, compute=2)

    starved = VirtualDomain(16, refusing=2)
    _drive(DerivationRegistry([starved]), budget=budget, passes=10, resume=False)
    assert starved.output == {}

    fair = VirtualDomain(16, refusing=2)
    _drive(DerivationRegistry([fair]), budget=budget, passes=10, resume=True)
    assert sorted(fair.output) == [VirtualDomain.key_at(index) for index in range(2, 16)]


def test_a_lost_resume_position_costs_position_and_not_correctness() -> None:
    """Dropping every hint converges the same archive, one sweep later.

    Anti-vacuity: let the cursor decide that a key is valid -- skip re-inspecting
    a swept range -- and the unresumed run below converges a different set from
    the resumed one.
    """
    resumed = VirtualDomain(32)
    _drive(DerivationRegistry([resumed]), budget=Budget(page=8, publication=4), passes=8, resume=True)

    unresumed = VirtualDomain(32)
    _drive(DerivationRegistry([unresumed]), budget=Budget(page=8, publication=4), passes=8, resume=False)

    assert resumed.output == unresumed.output
    assert len(resumed.output) == 32


def test_budget_exhaustion_is_pending_by_policy() -> None:
    """A bound stops work without inventing failure or losing the remainder."""
    adapter = RecordingDerivation("d", required=("a", "b", "c"))
    registry = DerivationRegistry([adapter])

    report = converge(registry, FRAME, budget=1)
    assert report.done == 1
    assert report.failed == 0
    assert {item.reason for item in report.by_outcome(Outcome.PENDING)} == {PendingReason.BUDGET}

    assert converge(registry, FRAME).done == 2
    assert converge(registry, FRAME).wrote_nothing


def test_a_bare_integer_budget_still_means_publications() -> None:
    """The pre-paging caller's meaning survives: publish at most this many keys.

    Anti-vacuity: read a bare integer as a page or discovery bound and a caller
    asking for one publication silently gets one *key looked at* instead.
    """
    domain = VirtualDomain(10)
    registry = DerivationRegistry([domain])

    report = converge(registry, FRAME, budget=3)
    assert report.done == 3
    assert domain.materialized_keys == 10


def test_a_page_that_does_not_advance_is_refused_rather_than_looped() -> None:
    """A cursor that repeats is a broken adapter, not an infinite pass.

    Anti-vacuity: drop the advance check and this test hangs instead of failing,
    which is how the bug would reach production.
    """

    class StuckPaging(BaseDerivation):
        domain = "stuck"
        prerequisites: tuple[str, ...] = ()

        def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
            return KeyPage(("a",), "always")

        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            return dict.fromkeys(keys, KeyStatus.VALID)

        def compute(self, frame: DerivationFrame, key: str) -> Replacement:  # pragma: no cover - never reached
            raise AssertionError("a stuck page must not reach computation")

        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:  # pragma: no cover
            raise AssertionError("a stuck page must not reach publication")

    registry = DerivationRegistry([StuckPaging()])
    report = converge(registry, FRAME)

    assert report.failed == 1
    failure = report.by_outcome(Outcome.FAILED)[0]
    assert failure.error is not None and "advance" in failure.error


def test_a_page_larger_than_its_limit_is_refused() -> None:
    """The page budget is a bound on the adapter, not a suggestion to it."""

    class OversizedPaging(BaseDerivation):
        domain = "oversized"
        prerequisites: tuple[str, ...] = ()

        def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
            return KeyPage(tuple(f"k{index}" for index in range(limit + 5)), None)

        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            return dict.fromkeys(keys, KeyStatus.MISSING)

        def compute(self, frame: DerivationFrame, key: str) -> Replacement:  # pragma: no cover
            raise AssertionError("an oversized page must not reach computation")

        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:  # pragma: no cover
            raise AssertionError("an oversized page must not reach publication")

    registry = DerivationRegistry([OversizedPaging()])
    report = converge(registry, FRAME, budget=Budget(page=2))

    assert report.failed == 1
    failure = report.by_outcome(Outcome.FAILED)[0]
    assert failure.error is not None and "limit" in failure.error


def test_a_page_may_be_returned_as_a_pair_by_a_foreign_ring() -> None:
    """Storage may not import this ring, so the page contract is structural.

    This adapter inherits nothing from the kernel -- no base class, no enum, no
    page type -- exactly as a ``polylogue/storage`` adapter must. Anti-vacuity:
    require a ``KeyPage`` instance in ``_as_page`` and a domain that cannot
    import this module can no longer page at all.
    """

    class PairPaging:
        domain = "pairs"
        prerequisites: tuple[str, ...] = ()

        def __init__(self) -> None:
            self.output: dict[str, str] = {}

        def required_page(
            self, frame: DerivationFrame, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            return (("a", "b"), None) if cursor is None else ((), None)

        def excess_page(
            self, frame: DerivationFrame, *, cursor: str | None, limit: int
        ) -> tuple[tuple[str, ...], str | None]:
            return ((), None)

        def prerequisite_keys(self, frame: DerivationFrame, key: str) -> Iterable[tuple[str, str]]:
            return ()

        def quiet(self, frame: DerivationFrame, key: str) -> bool:
            return False

        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, str]:
            return {key: ("valid" if key in self.output else "missing") for key in keys}

        def compute(self, frame: DerivationFrame, key: str) -> Replacement:
            return Replacement(key=DerivationKey(self.domain, key), input_binding="b0", payload=key)

        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
            self.output[str(replacement.payload)] = replacement.input_binding
            return True

    adapter = PairPaging()
    report = converge(DerivationRegistry([adapter]), FRAME)

    assert report.done == 2
    assert sorted(adapter.output) == ["a", "b"]


def test_a_page_budget_below_one_key_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one key"):
        Budget(page=0)


# ── the graph ──────────────────────────────────────────────────────


def test_a_prerequisite_cycle_is_refused_rather_than_starving_a_domain() -> None:
    """An unrunnable graph fails at resolution, not by silently skipping work."""
    first = RecordingDerivation("a", required=("k",), prerequisites=("b",))
    second = RecordingDerivation("b", required=("k",), prerequisites=("a",))
    registry = DerivationRegistry([first, second])
    with pytest.raises(ValueError, match="cycle"):
        registry.validate()


def test_an_undeclared_prerequisite_is_refused() -> None:
    adapter = RecordingDerivation("a", required=("k",), prerequisites=("nope",))
    with pytest.raises(ValueError, match="not registered"):
        DerivationRegistry([adapter]).validate()


def test_a_binding_into_an_unregistered_domain_blocks_rather_than_publishes() -> None:
    """An input the kernel cannot reach is not an input it may assume."""
    adapter = RecordingDerivation("d", required=("x",), bindings={"x": (("ghost", "a"),)})
    registry = DerivationRegistry([adapter])

    report = converge(registry, FRAME)
    assert adapter.published == []
    blocked = report.by_outcome(Outcome.PENDING)[0]
    assert blocked.reason is PendingReason.BLOCKED
    assert blocked.error is not None and "not registered" in blocked.error


def test_registration_order_is_not_load_bearing() -> None:
    """A domain may be registered before the prerequisite it declares.

    Anti-vacuity: validate eagerly in ``register`` and this fails, because the
    dependant is registered first -- which would make a composition root's
    ordering a correctness concern rather than a listing.
    """
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",))
    upstream = RecordingDerivation("up", required=("a",))
    registry = DerivationRegistry()
    registry.register(downstream)
    registry.register(upstream)

    assert [adapter.domain for adapter in registry.ordered()] == ["up", "down"]


def test_domains_converge_in_declared_prerequisite_order() -> None:
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",))
    upstream = RecordingDerivation("up", required=("a",))
    registry = DerivationRegistry([downstream, upstream])

    assert [adapter.domain for adapter in registry.ordered()] == ["up", "down"]
