"""``DaemonConverger`` drives derivations without becoming a second engine.

The facade keeps its name and its callers; what changed is that a migrated
domain converges from its own output relation instead of from stage state. The
laws here are the ones that would let a second engine creep back in: a facade
that caches the pending set, that wraps computation in an outer lease, or that
lets stage state certify a derived output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from polylogue.daemon.convergence import DaemonConverger
from polylogue.daemon.derivation import (
    BaseDerivation,
    DerivationFrame,
    DerivationKey,
    Replacement,
)

FRAME = DerivationFrame(archive_root="/archive", source_revision="r1")


class StringStatusDerivation(BaseDerivation):
    """A domain speaking the string vocabulary, as storage-side adapters do.

    Storage may not import the daemon ring, so an adapter there returns status
    values rather than enum members. This proves the kernel accepts that.
    """

    domain = "strings"
    prerequisites: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.output: dict[str, str] = {}
        self.binding = {"a": "b0", "b": "b0"}
        self.leases: list[str] = []

    def required(self, frame: DerivationFrame) -> Iterable[str]:
        return ("a", "b")

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
        self.leases.append(str(replacement.payload))
        self.output[str(replacement.payload)] = replacement.input_binding
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
