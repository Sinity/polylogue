"""The laws every derivation obeys, stated once over a synthetic domain.

These are architecture-neutral: they constrain the kernel's relation between
``required``, ``inspect``, ``compute`` and ``publish``, not any domain's SQL.
A domain adapter that passes its own tests but breaks one of these is wrong.

Anti-vacuity is named per test: each says which mutation of the kernel makes it
red. A law nobody can break is not a law.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pytest

from polylogue.daemon.derivation import (
    BaseDerivation,
    DerivationFrame,
    DerivationKey,
    DerivationRegistry,
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
        binding: Mapping[str, str] | None = None,
        poison: frozenset[str] = frozenset(),
        quiet_keys: frozenset[str] = frozenset(),
        publish_refuses: frozenset[str] = frozenset(),
    ) -> None:
        self.domain = domain
        self.prerequisites = prerequisites
        self._required = tuple(required)
        self.binding = dict(binding or dict.fromkeys(required, "b0"))
        self.output: dict[str, str] = {}
        self.poison = poison
        self.quiet_keys = quiet_keys
        self.publish_refuses = set(publish_refuses)
        self.computed: list[str] = []
        self.published: list[str] = []

    def required(self, frame: DerivationFrame) -> Iterable[str]:
        return self._required

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus]:
        statuses: dict[str, KeyStatus] = {}
        for key in keys:
            if key not in self.output:
                statuses[key] = KeyStatus.MISSING
            elif self.output[key] != self.binding.get(key, ""):
                statuses[key] = KeyStatus.STALE
            else:
                statuses[key] = KeyStatus.VALID
        return statuses

    def excess_candidates(self, frame: DerivationFrame) -> Iterable[str]:
        return tuple(key for key in self.output if key not in self._required)

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        return key in self.quiet_keys

    def compute(self, frame: DerivationFrame, key: str) -> Replacement:
        self.computed.append(key)
        if key in self.poison:
            raise RuntimeError(f"poison input for {key}")
        return Replacement(key=type(self)._key(self.domain, key), input_binding=self.binding[key], payload=key)

    @staticmethod
    def _key(domain: str, key: str) -> DerivationKey:
        return DerivationKey(domain, key)

    def publish(self, frame: DerivationFrame, replacement: Replacement) -> bool:
        key = str(replacement.payload)
        if key in self.publish_refuses:
            return False
        self.published.append(key)
        self.output[key] = replacement.input_binding
        return True


def test_pending_is_required_minus_valid_and_a_second_pass_writes_nothing() -> None:
    """Inspection is authoritative, so an unchanged second pass is a no-op.

    Anti-vacuity: make ``_pending_keys`` return every required key regardless of
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


def test_excess_output_is_pending_work_not_silence() -> None:
    """A key the output holds but ``required`` no longer names is reported."""
    adapter = RecordingDerivation("d", required=("a",))
    registry = DerivationRegistry([adapter])
    converge(registry, FRAME)
    adapter.output["gone"] = "b0"

    report = converge(registry, FRAME)
    keys = {str(outcome.key) for outcome in report.outcomes}
    assert "d:gone" in keys


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


def test_a_prerequisite_cycle_is_a_construction_error() -> None:
    """An unrunnable graph fails at registration, not by starving a domain."""
    first = RecordingDerivation("a", required=("k",), prerequisites=("b",))
    second = RecordingDerivation("b", required=("k",), prerequisites=("a",))
    registry = DerivationRegistry()
    registry.register(first)
    with pytest.raises(ValueError, match="cycle"):
        registry.register(second)


def test_an_undeclared_prerequisite_is_a_construction_error() -> None:
    adapter = RecordingDerivation("a", required=("k",), prerequisites=("nope",))
    with pytest.raises(ValueError, match="not registered"):
        DerivationRegistry([adapter])


def test_domains_converge_in_declared_prerequisite_order() -> None:
    downstream = RecordingDerivation("down", required=("x",), prerequisites=("up",))
    upstream = RecordingDerivation("up", required=("a",))
    registry = DerivationRegistry([downstream, upstream])

    assert [adapter.domain for adapter in registry.ordered()] == ["up", "down"]


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
