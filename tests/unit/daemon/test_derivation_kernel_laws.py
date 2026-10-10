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
from collections.abc import Iterable, Iterator, Mapping, Sequence

import pytest

from polylogue.daemon.derivation import (
    BaseDerivation,
    Budget,
    DerivationFrame,
    DerivationKey,
    DerivationRegistry,
    DerivationReport,
    KeyOutcome,
    KeyPage,
    KeyStatus,
    Outcome,
    PendingReason,
    Replacement,
    converge,
)

FRAME = DerivationFrame(archive_root="/archive", source_revision="r1")


def _page(keys: Sequence[str], *, cursor: str | None, limit: int) -> KeyPage:
    start = int(cursor) if cursor else 0
    stop = min(start + limit, len(keys))
    return KeyPage(tuple(keys[start:stop]), str(stop) if stop < len(keys) else None)


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

    def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
        return _page(self._required, cursor=cursor, limit=limit)

    def excess_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
        return _page(tuple(key for key in self.output if key not in self._required), cursor=cursor, limit=limit)

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
    assert second.made_no_publication_attempts
    assert second.work.published == 0
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
    assert converge(registry, FRAME).made_no_publication_attempts


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
    assert converge(registry, FRAME).made_no_publication_attempts


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


def test_successful_cleanup_is_pending_when_requiredness_moved_before_certification() -> None:
    """A vanished required key is a binding race, not a broken publisher.

    Anti-vacuity: omit the requiredness recheck and a successful cleanup with
    missing output is classified FAILED instead of retryable BINDING_MOVED.
    """

    class VanishingDomain(RecordingDerivation):
        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
            del frame, replacement
            self._required = ()
            return True

        def is_required_key(self, frame: DerivationFrame, key: str) -> bool:
            del frame
            return key in self._required

    report = converge(DerivationRegistry([VanishingDomain("d", required=("vanishing",))]), FRAME)

    assert report.failed == 0
    assert report.pending == 1
    assert report.by_outcome(Outcome.PENDING)[0].reason is PendingReason.BINDING_MOVED


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
    # A replacement was committed before authoritative certification rejected
    # it, so no DONE outcome cannot mean that storage was untouched.
    assert not report.made_no_publication_attempts
    assert report.work.published == 1
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


def test_a_bulk_inspection_poison_isolated_to_its_key_and_dependent_closure() -> None:
    """A bounded batch inspection retry isolates one unreadable key.

    Anti-vacuity: retain the former ``unreadable_domains.add(domain)`` after a
    bulk inspection exception and the unrelated ``good`` key plus its exact
    dependent are skipped.
    """

    class BatchPoisonedUpstream(RecordingDerivation):
        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            if "poison" in keys:
                raise RuntimeError("poison output")
            return super().inspect(frame, keys)

    upstream = BatchPoisonedUpstream("up", required=("poison", "good"))
    downstream = RecordingDerivation(
        "down",
        required=("from-good",),
        prerequisites=("up",),
        bindings={"from-good": (("up", "good"),)},
    )

    report = converge(DerivationRegistry([upstream, downstream]), FRAME)

    assert upstream.published == ["good"]
    assert downstream.published == ["from-good"]
    failed = report.by_outcome(Outcome.FAILED)
    assert [(item.key.domain, item.key.key) for item in failed] == [("up", "poison")]


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


def test_direct_report_construction_derives_counts_from_outcomes() -> None:
    """The exported positional report constructor keeps outcome properties truthful.

    Anti-vacuity: remove DerivationReport.__post_init__ and this completed
    outcome is exposed as zero done work.
    """
    report = DerivationReport(FRAME, (KeyOutcome(DerivationKey("demo", "one"), Outcome.DONE),))
    assert report.done == 1
    assert report.pending == 0


def test_outer_deadline_tightens_an_existing_budget_deadline() -> None:
    """Both relative deadlines constrain the pass to the earlier deadline.

    Anti-vacuity: ignore deadline_s when a Budget already has one and the
    returned budget keeps the later 100-second allowance.
    """
    limits = Budget.coerce(Budget(deadline_s=100), deadline_s=1)
    assert limits.deadline_s == 1


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
    assert converge(registry, FRAME, budget=Budget(page=4, publication=2)).made_no_publication_attempts


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
    assert converge(registry, FRAME).made_no_publication_attempts


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


def test_registered_domain_order_must_put_prerequisites_before_consumers() -> None:
    """Domain order is declared once; the kernel does not topologically sort it.

    Anti-vacuity: restore the generic DFS sorter and this accepts the reverse
    declaration by silently rewriting it to ``up, down``.
    """
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",))
    upstream = RecordingDerivation("up", required=("a",))

    with pytest.raises(ValueError, match="must be registered before its consumer"):
        DerivationRegistry([downstream, upstream]).ordered()

    assert [adapter.domain for adapter in DerivationRegistry([upstream, downstream]).ordered()] == ["up", "down"]


@pytest.mark.parametrize("poison_position", [0, 1])
@pytest.mark.parametrize("restart", [False, True])
def test_bulk_inspection_fallback_stops_at_the_error_limit_and_resumes_siblings(
    poison_position: int, restart: bool
) -> None:
    """Process the healthy prefix before poison stops retries; retain the suffix.

    Eagerly inspecting all fallback keys makes the middle poison consume the
    error budget before the first missing key can publish, on every pass.
    """
    calls: list[tuple[str, ...]] = []

    class PoisonedInspection(RecordingDerivation):
        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            calls.append(tuple(keys))
            if "poison" in keys:
                raise ValueError("synthetic unreadable key")
            return super().inspect(frame, keys)

    required = ("poison", "good", "later") if poison_position == 0 else ("good", "poison", "later")
    adapter = PoisonedInspection("d", required=required)
    registry = DerivationRegistry([adapter])
    report = converge(registry, FRAME, budget=Budget(max_errors=1, retained_outcomes=0))
    expected_calls = [required, (required[0],)]
    if poison_position:
        expected_calls.extend([("good",), ("poison",)])
    assert calls == expected_calls
    assert report.failed == 1 and report.pending == 2 - poison_position and report.done == poison_position
    assert report.work.inspected == 3 + poison_position
    assert adapter.computed == adapter.published == (["good"] if poison_position else [])
    assert report.cursor.position("d").pending_keys == required[poison_position + 1 :]
    assert report.cursor_unsettled_domains == frozenset({"d"})
    assert report.outcomes == () and report.truncated

    resumed = converge(registry, FRAME, budget=Budget(max_errors=1), cursor=None if restart else report.cursor)
    if restart:
        assert resumed.failed == 1
        resumed = converge(registry, FRAME, budget=Budget(max_errors=1), cursor=resumed.cursor)
    assert resumed.done == 2 - poison_position and resumed.failed == resumed.pending == 0
    assert resumed.cursor.position("d").swept
    assert adapter.output == {"good": "b0", "later": "b0"}
    assert adapter.computed == adapter.published == ["good", "later"]


def test_fallback_compute_failure_stops_before_the_next_inspection_attempt() -> None:
    """An earlier computation failure must stop a later inspection poison."""
    calls: list[tuple[str, ...]] = []

    class PoisonedInspection(RecordingDerivation):
        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            calls.append(tuple(keys))
            if "inspect-poison" in keys:
                raise ValueError("synthetic unreadable key")
            return super().inspect(frame, keys)

    required = ("compute-poison", "inspect-poison", "later")
    adapter = PoisonedInspection("d", required=required, poison=frozenset({"compute-poison"}))
    registry = DerivationRegistry([adapter])
    report = converge(registry, FRAME, budget=Budget(max_errors=1))

    assert calls == [required, ("compute-poison",)]
    assert report.failed == 1 and report.pending == 2 and report.done == 0
    assert adapter.computed == ["compute-poison"] and adapter.published == []
    assert report.cursor.position("d").pending_keys == ("inspect-poison", "later")

    poison = converge(registry, FRAME, budget=Budget(max_errors=1), cursor=report.cursor)
    assert poison.failed == poison.pending == 1 and poison.done == 0
    assert poison.cursor.position("d").pending_keys == ("later",)
    later = converge(registry, FRAME, budget=Budget(max_errors=1), cursor=poison.cursor)
    assert later.done == 1 and later.failed == later.pending == 0
    assert adapter.output == {"later": "b0"}


@pytest.mark.parametrize("deadline_phase", ["bulk", "individual"])
@pytest.mark.parametrize("restart", [False, True])
def test_bulk_inspection_fallback_deadline_retains_unvisited_keys(
    monkeypatch: pytest.MonkeyPatch, deadline_phase: str, restart: bool
) -> None:
    """Expiry stops new reads, while continuation or restart reaches the suffix."""
    clock = {"now": 0.0}
    monkeypatch.setattr("polylogue.daemon.derivation._pass_clock", lambda: clock["now"])
    calls: list[tuple[str, ...]] = []

    class SlowInspection(RecordingDerivation):
        expired = False

        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            calls.append(tuple(keys))
            if len(keys) > 1:
                if deadline_phase == "bulk" and not self.expired:
                    clock["now"] = 1.0
                    self.expired = True
                raise ValueError("synthetic page inspection fault")
            result = super().inspect(frame, keys)
            if deadline_phase == "individual" and not self.expired:
                clock["now"] = 1.0
                self.expired = True
            return result

    adapter = SlowInspection("d", required=("healthy", "later", "last"))
    adapter.output["healthy"] = "b0"
    registry = DerivationRegistry([adapter])
    report = converge(registry, FRAME, budget=Budget(deadline_at=1.0))
    expected_calls: list[tuple[str, ...]] = [("healthy", "later", "last")]
    pending_keys: tuple[str, ...] = ("healthy", "later", "last")
    if deadline_phase == "individual":
        expected_calls.append(("healthy",))
        pending_keys = ("later", "last")
    assert calls == expected_calls
    assert report.failed == report.done == 0 and report.pending == len(pending_keys)
    assert report.work.inspected == 3
    assert adapter.computed == adapter.published == []
    assert report.cursor.position("d").pending_keys == pending_keys
    assert report.cursor_unsettled_domains == frozenset()
    assert all(item.reason is PendingReason.BUDGET for item in report.by_outcome(Outcome.PENDING))

    clock["now"] = 0.0
    resumed = converge(registry, FRAME, budget=Budget(deadline_at=1.0), cursor=None if restart else report.cursor)
    assert resumed.done == 2 and resumed.failed == resumed.pending == 0
    assert resumed.cursor.position("d").swept
    assert adapter.output == {"healthy": "b0", "later": "b0", "last": "b0"}


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


def test_a_poison_prerequisite_inspection_blocks_only_its_dependent_key() -> None:
    """One unreadable upstream key does not make its whole domain unreadable.

    Anti-vacuity: add the inspected key's domain to ``unreadable_domains`` on a
    binding inspection error and ``y`` below is incorrectly blocked with ``x``.
    """

    class KeyPoisonedUpstream(RecordingDerivation):
        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            if "poison" in keys:
                raise RuntimeError("poison input")
            return super().inspect(frame, keys)

    upstream = KeyPoisonedUpstream("up", required=("poison", "good"))
    upstream.output["good"] = "b0"
    downstream = RecordingDerivation(
        "down",
        required=("x", "y"),
        prerequisites=("up",),
        bindings={"x": (("up", "poison"),), "y": (("up", "good"),)},
    )

    report = converge(DerivationRegistry([upstream, downstream]), FRAME, domains=["down"])

    assert downstream.published == ["y"]
    assert report.done == 1
    blocked = report.by_outcome(Outcome.PENDING)
    assert len(blocked) == 1
    assert blocked[0].key == DerivationKey("down", "x")
    assert blocked[0].reason is PendingReason.BLOCKED


def test_inspection_budget_allows_already_inspected_key_to_publish() -> None:
    """Anti-vacuity: treating inspection exhaustion as compute exhaustion stalls forever."""
    domain = VirtualDomain(10_000)
    report = converge(
        DerivationRegistry([domain]),
        FRAME,
        budget=Budget(page=1, discovery=1, inspection=1, compute=1, publication=1),
    )
    assert report.done == 1
    assert domain.calls["compute"] == 1
    assert report.work.discovered == 1
    # Publication certification is counted separately from discovery admission.
    assert report.work.inspected == 2


def test_a_failed_bulk_inspect_retry_does_not_report_the_key_as_valid() -> None:
    """An inspection that raised is absence of evidence, not evidence of validity.

    The kernel retries a failed bulk inspection key by key and records the
    still-failing key ``FAILED``. It used to *also* write
    ``statuses[key] = KeyStatus.VALID`` for that key, which is the one status
    that asserts the output is already up to date (polylogue-tjtua).

    Anti-vacuity: restore that assignment and ``good`` is still published while
    ``poison`` is reported VALID-by-fiat -- observable here as the excess sweep
    no longer being able to distinguish it, so assert on the recorded status
    directly through a domain that publishes on demand.
    """
    inspected_alone: list[str] = []

    class KeyPoisoned(RecordingDerivation):
        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            if len(keys) == 1:
                inspected_alone.append(keys[0])
            if "poison" in keys:
                raise RuntimeError("cannot read output relation for poison")
            return super().inspect(frame, keys)

    domain = KeyPoisoned("d", required=("poison", "good"))
    report = converge(DerivationRegistry([domain]), FRAME)

    # The per-key retry ran for both keys, and only the readable one published.
    assert set(inspected_alone) >= {"poison", "good"}
    assert domain.published == ["good"]

    failed = report.by_outcome(Outcome.FAILED)
    assert [(item.key.domain, item.key.key) for item in failed] == [("d", "poison")]
    # A key whose authority could not be inspected is not converged.
    assert report.done == 1
    assert "poison" not in domain.output


def test_the_publication_budget_bounds_attempts_not_certified_publications() -> None:
    """An adapter that always mis-publishes cannot spin inside one pass.

    ``publish`` returns True but leaves the output relation missing, so
    post-publication certification records FAILED every time.

    Anti-vacuity: move the ``published`` counter back behind the certification
    check and this pass issues one ``publish`` call per required key (4) rather
    than the two the budget allows.
    """

    class MisPublishing(RecordingDerivation):
        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
            self.published.append(str(replacement.payload))
            return True  # lies: nothing is written to ``output``

    domain = MisPublishing("d", required=("a", "b", "c", "d"))

    report = converge(DerivationRegistry([domain]), FRAME, budget=Budget(page=4, publication=2))

    assert domain.published == ["a", "b"]
    assert report.failed == 2


# ── publication barrier ────────────────────────────────────────────


class SessionKeyedDerivation(RecordingDerivation):
    """A session-derived domain: each key is its own session."""

    def barrier_sessions(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, str]:
        return {key: key for key in keys}


def test_a_barrier_held_session_stays_pending_and_its_siblings_publish() -> None:
    """A session whose newest revision awaits primary publication is not derived.

    Anti-vacuity (polylogue-wtfyv): drop the ``barrier`` argument from the pass
    (as every derivation owner did before) and ``held`` is computed and
    published ahead of its primary revision.
    """
    adapter = SessionKeyedDerivation("d", required=("free", "held"))
    registry = DerivationRegistry([adapter])

    report = converge(registry, FRAME, barrier=lambda sessions: {"held"} & set(sessions))

    assert adapter.published == ["free"]
    assert "held" not in adapter.computed
    held = [outcome for outcome in report.outcomes if outcome.key == DerivationKey("d", "held")]
    assert len(held) == 1
    assert held[0].outcome is Outcome.PENDING
    assert held[0].reason is PendingReason.BLOCKED

    released = converge(registry, FRAME, barrier=lambda sessions: set())
    assert adapter.published == ["free", "held"]
    assert released.done == 1


def test_an_unreadable_barrier_holds_every_session_derived_key() -> None:
    """Deriving past a barrier that cannot be read is the violation it prevents.

    Anti-vacuity: treat a raising barrier as "nothing blocked" and both keys
    publish.
    """
    adapter = SessionKeyedDerivation("d", required=("a", "b"))

    def unreadable(sessions: Sequence[str]) -> set[str]:
        raise RuntimeError("source tier locked")

    report = converge(DerivationRegistry([adapter]), FRAME, barrier=unreadable)

    assert adapter.published == []
    assert report.done == 0
    assert all(outcome.reason is PendingReason.BLOCKED for outcome in report.outcomes)


def test_a_domain_that_is_not_session_derived_ignores_the_barrier() -> None:
    """Only a domain that maps keys to sessions can be held."""
    adapter = RecordingDerivation("d", required=("a",))

    converge(DerivationRegistry([adapter]), FRAME, barrier=lambda sessions: set(sessions) | {"a"})

    assert adapter.published == ["a"]


class CarrierKeyedDerivation(RecordingDerivation):
    """One key per carrier, each naming several sessions."""

    def barrier_sessions(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, Iterable[str]]:
        return {"batch": ("s1", "s2")}


def test_a_key_naming_several_sessions_is_held_when_any_one_waits() -> None:
    """Anti-vacuity: holding only single-session keys lets the batch lower s2's markers."""
    adapter = CarrierKeyedDerivation("markers", required=("batch",))

    report = converge(DerivationRegistry([adapter]), FRAME, barrier=lambda sessions: {"s2"} & set(sessions))

    assert adapter.published == []
    assert report.outcomes[0].reason is PendingReason.BLOCKED


def test_barrier_consumes_session_membership_in_pages_and_stops_at_a_blocker() -> None:
    """A carrier with many sessions does not flatten its entire membership."""

    class LargeCarrier(CarrierKeyedDerivation):
        def barrier_sessions(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, Iterable[str]]:
            def sessions() -> Iterator[str]:
                try:
                    for index in range(1000):
                        yield f"s{index}"
                finally:
                    closed.append(True)

            return {"batch": sessions()}

    closed: list[bool] = []
    adapter = LargeCarrier("markers", required=("batch",))
    observed_sizes: list[int] = []

    def barrier(sessions: Sequence[str]) -> set[str]:
        observed_sizes.append(len(sessions))
        return {"s300"}.intersection(sessions)

    report = converge(DerivationRegistry([adapter]), FRAME, barrier=barrier)

    assert observed_sizes == [128, 128, 128]
    assert closed == [True]
    assert adapter.computed == []
    assert report.outcomes[0].reason is PendingReason.BLOCKED


def test_retiring_excess_output_is_never_held_by_the_barrier() -> None:
    """Retirement derives nothing from new content.

    Anti-vacuity (polylogue-wtfyv review): applying the barrier in the excess
    phase leaves a deleted session's orphaned output behind an obligation that
    may never publish.
    """
    adapter = SessionKeyedDerivation("d", required=())
    adapter.output["orphan"] = "b0"

    converge(DerivationRegistry([adapter]), FRAME, barrier=lambda sessions: set(sessions))

    assert adapter.published == ["orphan"]
    assert "orphan" not in adapter.output


def test_a_revision_staged_during_compute_is_refused_at_publication() -> None:
    """The barrier is re-decided inside the writer admission.

    Anti-vacuity (polylogue-wtfyv review): checking only before compute lets a
    session whose newer revision was staged while computing publish anyway.
    """
    adapter = SessionKeyedDerivation("d", required=("s",))
    staged: set[str] = set()
    original_compute = adapter.compute

    def compute_then_stage(frame: DerivationFrame, key: str) -> Replacement:
        replacement = original_compute(frame, key)
        staged.add(key)  # a concurrent ingest stages an unpublished revision
        return replacement

    adapter.compute = compute_then_stage  # type: ignore[method-assign]

    report = converge(DerivationRegistry([adapter]), FRAME, barrier=lambda sessions: staged & set(sessions))

    assert adapter.computed == ["s"]
    assert adapter.published == []
    assert [(item.outcome, item.reason) for item in report.outcomes] == [(Outcome.PENDING, PendingReason.BLOCKED)]


def test_a_failed_key_says_whether_its_failure_can_clear_unchanged() -> None:
    """A deterministic failure is not transient; lock contention is.

    Anti-vacuity (Codex): with no classification on the outcome, the raw
    and hook intake callbacks reported every derivation failure as transient,
    so an unchanged poison carrier was retried each cooldown forever instead
    of being isolated.
    """
    import sqlite3

    class LockedDerivation(RecordingDerivation):
        def compute(self, frame: DerivationFrame, key: str) -> Replacement:
            locked = sqlite3.OperationalError("database is locked")
            locked.sqlite_errorcode = 5  # SQLITE_BUSY
            raise locked

    poisoned = converge(DerivationRegistry([RecordingDerivation("d", required=("a",), poison=frozenset({"a"}))]), FRAME)
    locked = converge(DerivationRegistry([LockedDerivation("d", required=("a",))]), FRAME)
    assert [item.transient for item in poisoned.by_outcome(Outcome.FAILED)] == [False]
    assert [item.transient for item in locked.by_outcome(Outcome.FAILED)] == [True]


def test_a_failure_signature_names_its_exception_type() -> None:
    """Distinct exceptions in one phase are distinct failure signatures.

    Anti-vacuity (Codex): a phase-first reason such as ``compute: ...`` gave
    a KeyError and a TypeError the same signature, so alternating defects
    extended one exhaustion streak and isolated the item.
    """
    from polylogue.daemon.intake import _failure_signature

    class TypeErrorDerivation(RecordingDerivation):
        def compute(self, frame: DerivationFrame, key: str) -> Replacement:
            raise TypeError("wrong shape")

    runtime = converge(DerivationRegistry([RecordingDerivation("d", required=("a",), poison=frozenset({"a"}))]), FRAME)
    typed = converge(DerivationRegistry([TypeErrorDerivation("d", required=("a",))]), FRAME)
    (runtime_failure,) = runtime.by_outcome(Outcome.FAILED)
    (type_failure,) = typed.by_outcome(Outcome.FAILED)
    assert _failure_signature(runtime_failure.error) == "compute RuntimeError"
    assert _failure_signature(type_failure.error) == "compute TypeError"


def test_a_requiredness_recheck_failure_is_classified_and_typed() -> None:
    """The post-publication requiredness recheck classifies and names its error.

    Anti-vacuity: a FAILED outcome recorded there without ``transient`` and
    with a phase-only ``requiredness inspection: ...`` reason turns lock
    contention into a deterministic defect sharing one signature with every
    other recheck error, so the item is isolated.
    """
    import sqlite3

    from polylogue.daemon.intake import _failure_signature

    class LockedRecheck(RecordingDerivation):
        def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
            del frame, replacement
            self._required = ()
            return True

        def is_required_key(self, frame: DerivationFrame, key: str) -> bool:
            locked = sqlite3.OperationalError("database is locked")
            locked.sqlite_errorcode = 5  # SQLITE_BUSY
            raise locked

    report = converge(DerivationRegistry([LockedRecheck("d", required=("vanishing",))]), FRAME)
    (failed,) = report.by_outcome(Outcome.FAILED)
    assert failed.transient is True
    assert _failure_signature(failed.error) == "requiredness inspection OperationalError"


def test_a_per_key_inspection_failure_is_classified_and_typed() -> None:
    """The per-key inspection fallback classifies and names its exception.

    Anti-vacuity (Codex): the fallback outcome inherited ``transient=False``
    and an ``inspect: ...`` reason, so lock contention during inspection
    counted as a deterministic defect and shared one signature with every
    other inspection error.
    """
    import sqlite3

    from polylogue.daemon.intake import _failure_signature

    class LockedInspection(RecordingDerivation):
        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            locked = sqlite3.OperationalError("database is locked")
            locked.sqlite_errorcode = 5  # SQLITE_BUSY
            raise locked

    report = converge(DerivationRegistry([LockedInspection("d", required=("a",))]), FRAME)
    (failed,) = report.by_outcome(Outcome.FAILED)
    assert failed.transient is True
    assert _failure_signature(failed.error) == "inspect OperationalError"


def test_required_and_excess_pagers_may_reuse_a_cursor_value() -> None:
    """The two discovery phases are independent keysets.

    Anti-vacuity: track visited cursors per domain only and the excess pager's
    first ``next_cursor`` ("2") collides with the required pager's, raising
    "does not advance" for a valid pass.
    """
    domain = RecordingDerivation("demo", required=("a0", "a1", "a2", "a3"))
    domain.output.update(dict.fromkeys(("e0", "e1", "e2", "e3"), ""))
    registry = DerivationRegistry([domain])

    report = converge(registry, FRAME, budget=Budget(page=2))

    assert report.failed == 0
    assert {"a0", "a1", "a2", "a3"} <= set(domain.output)


def test_report_retention_does_not_change_prerequisite_verdicts() -> None:
    """A report-detail cap bounds the report, not the pass's own verdicts.

    Anti-vacuity: evict pass-local verdicts at ``retained_outcomes`` and the
    dependant re-inspects its upstream key, spending prerequisite inspection
    work the uncapped pass does not.
    """
    upstream = RecordingDerivation("up", required=("u",))
    dependant = RecordingDerivation("down", required=("d",), prerequisites=("up",), bindings={"d": (("up", "u"),)})
    registry = DerivationRegistry([upstream, dependant])

    report = converge(registry, FRAME, budget=Budget(page=10, inspection=4, retained_outcomes=0))

    assert report.done == 2
    assert report.work.prerequisites_inspected == 0
    assert dependant.output == {"d": "b0"}


def test_prerequisite_reads_do_not_starve_a_full_page_of_dependants() -> None:
    """A full page must still converge when each dependant names upstream keys.

    Anti-vacuity: meter prerequisite enumeration or inspection against the
    pass's discovery/inspection budgets and a page that fills both leaves every
    dependant blocked, pass after pass.
    """
    upstream = RecordingDerivation("up", required=("u0", "u1"))
    upstream.output.update({"u0": "b0", "u1": "b0"})
    both = (("up", "u0"), ("up", "u1"))
    dependant = RecordingDerivation(
        "down", required=("d0", "d1"), prerequisites=("up",), bindings={"d0": both, "d1": both}
    )
    registry = DerivationRegistry([upstream, dependant])

    report = converge(registry, FRAME, budget=Budget(page=2, discovery=2, inspection=2), domains=("down",))

    assert report.done == 2
    assert dependant.output == {"d0": "b0", "d1": "b0"}


@pytest.mark.parametrize("kind", ["quiet", "fault", "binding_moved", "blocked"])
def test_cursor_unsettled_evidence_survives_an_empty_outcome_sample(kind: str) -> None:
    """Only actual consumed unsettled work obliges a clean wrapped sweep."""
    adapter = RecordingDerivation(
        "d",
        required=("first", "tail"),
        quiet_keys=frozenset({"first"}) if kind == "quiet" else frozenset(),
        poison=frozenset({"first"}) if kind == "fault" else frozenset(),
        publish_refuses=frozenset({"first"}) if kind == "binding_moved" else frozenset(),
    )
    if kind == "blocked":
        adapter.prerequisites = ("up",)
        adapter._bindings = {"first": (("up", "missing"),)}
        registry = DerivationRegistry([RecordingDerivation("up", required=()), adapter])
    else:
        registry = DerivationRegistry([adapter])
    report = converge(registry, FRAME, budget=Budget(retained_outcomes=0))
    assert report.outcomes == () and report.truncated
    assert report.cursor.position("d").swept
    assert report.cursor_unsettled_domains == frozenset({"d"})


def test_fallback_budget_suffix_stays_unvisited_and_ahead_of_the_cursor() -> None:
    """Publication exhaustion prevents new fallback reads of the suffix."""
    calls: list[tuple[str, ...]] = []

    class PoisonedSuffix(RecordingDerivation):
        def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
            calls.append(tuple(keys))
            if "poison" in keys:
                raise ValueError("synthetic suffix inspection fault")
            return super().inspect(frame, keys)

    adapter = PoisonedSuffix("d", required=("published", "budget", "poison"))
    report = converge(DerivationRegistry([adapter]), FRAME, budget=Budget(publication=1, retained_outcomes=0))
    assert calls == [("published", "budget", "poison"), ("published",), ("published",)]
    assert report.failed == 0 and report.pending == 2 and report.done == 1
    assert report.cursor.position("d").pending_keys == ("budget", "poison")
    assert report.cursor_unsettled_domains == frozenset()


def test_unchanged_discovery_fault_has_no_consumed_unsettled_prefix() -> None:
    class Unreadable(RecordingDerivation):
        def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
            raise ValueError("synthetic unreadable relation")

    report = converge(DerivationRegistry([Unreadable("d", required=())]), FRAME, budget=Budget(retained_outcomes=0))
    assert report.failed == 1 and not report.cursor.position("d").swept
    assert report.cursor_unsettled_domains == frozenset()


def test_deadline_reached_during_compute_keeps_the_unattempted_suffix_ahead_of_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("polylogue.daemon.derivation._pass_clock", lambda: clock["now"])

    class SlowCompute(RecordingDerivation):
        def compute(self, frame: DerivationFrame, key: str) -> Replacement:
            result = super().compute(frame, key)
            clock["now"] = 2.0
            return result

    adapter = SlowCompute("d", required=("published", "deferred"))
    report = converge(DerivationRegistry([adapter]), FRAME, budget=Budget(deadline_at=1.0, retained_outcomes=0))
    assert report.done == 1 and report.pending == 1
    assert report.cursor.position("d").pending_keys == ("deferred",)
    assert report.cursor_unsettled_domains == frozenset()
