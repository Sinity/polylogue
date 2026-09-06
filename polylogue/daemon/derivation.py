"""The derivation kernel: one correctness relation for every derived model.

A derived model is current when the authoritative output relation says so.
Nothing else may certify it — not a queue, a cursor, a debt row, a freshness
flag, or a receipt. The kernel knows five things and no domain SQL:

``required(frame)``   the stable keys that must exist, including keys whose
                      correct output is empty
``inspect(frame, k)`` each key classified VALID / MISSING / STALE / EXCESS from
                      the output relation and its own binding
``compute(frame, k)`` a typed replacement, produced outside the writer lease
``publish(frame, r)`` revalidate the bound inputs and atomically replace
                      exactly that key under the lease
``prerequisites``     the output domains this one reads

The pending set is ``required`` minus ``valid``. Deleting every scheduling hint
and restarting reconstructs it, because it was never stored.

Domains keep their own storage, SQL, and atomic replacement boundary: this is
not a generic artifact store, and adding one would create a second truth beside
the output relations that are already authoritative.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from polylogue.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "BaseDerivation",
    "DerivationAdapter",
    "DerivationFrame",
    "DerivationKey",
    "DerivationRegistry",
    "DerivationReport",
    "KeyOutcome",
    "KeyStatus",
    "Outcome",
    "PendingReason",
    "Replacement",
    "ReplacementLike",
    "converge",
]


class KeyStatus(Enum):
    """How one key's authoritative output relates to what is required."""

    VALID = "valid"
    MISSING = "missing"
    STALE = "stale"
    EXCESS = "excess"


class Outcome(Enum):
    """What one convergence attempt did with one key."""

    DONE = "done"
    PENDING = "pending"
    FAILED = "failed"


class PendingReason(Enum):
    """Why a key is pending rather than done, when nothing failed.

    Pending is a policy or race outcome and carries no error. Collapsing it
    into failure is how a quiet source or a lost publication race came to be
    reported as a defect; collapsing failure into it is how a real defect came
    to be reported as backlog.
    """

    BUDGET = "budget"
    QUIET = "quiet"
    BLOCKED = "blocked"
    BINDING_MOVED = "binding_moved"


@dataclass(frozen=True, slots=True, order=True)
class DerivationKey:
    """The smallest independently replaceable unit of one derived domain."""

    domain: str
    key: str

    def __str__(self) -> str:
        return f"{self.domain}:{self.key}"


@dataclass(frozen=True, slots=True)
class DerivationFrame:
    """The immutable input generation and recipe a pass is bound to.

    ``source_revision`` names the admitted-input boundary the pass converges
    against. Binding to a boundary rather than to "the queue looked empty" is
    what makes an archive-wide derivation reproducible: a later admission
    creates a later frame, it does not retroactively invalidate this one.
    """

    archive_root: str
    source_revision: str
    recipe_versions: Mapping[str, str] = field(default_factory=dict)

    def recipe_version(self, domain: str) -> str:
        return self.recipe_versions.get(domain, "")


class ReplacementLike(Protocol):
    """What the kernel needs from a computed replacement, and nothing more.

    Declared structurally so a domain adapter in a package that may not import
    this ring can produce its own replacement type. The kernel never reads a
    key off a replacement -- it already knows which key it asked for -- so the
    protocol does not constrain one, and the two rings cannot disagree about
    its representation.
    """

    @property
    def input_binding(self) -> str: ...

    @property
    def payload(self) -> object: ...

    @property
    def empty(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class Replacement:
    """One computed key replacement plus the input binding it was computed at.

    ``input_binding`` commits to every upstream *value* the computation read,
    not merely to identifiers, timestamps, sort keys, or row counts. A binding
    over identity alone reports VALID after a mutation that changes the output,
    which is the defect this type exists to make unrepresentable.
    """

    key: DerivationKey
    input_binding: str
    payload: object
    empty: bool = False


@dataclass(frozen=True, slots=True)
class KeyOutcome:
    """The typed result of one key's convergence attempt."""

    key: DerivationKey
    outcome: Outcome
    reason: PendingReason | None = None
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass(frozen=True, slots=True)
class DerivationReport:
    """What one bounded convergence pass did, per key and in total."""

    frame: DerivationFrame
    outcomes: tuple[KeyOutcome, ...]

    def by_outcome(self, outcome: Outcome) -> tuple[KeyOutcome, ...]:
        return tuple(item for item in self.outcomes if item.outcome is outcome)

    @property
    def done(self) -> int:
        return len(self.by_outcome(Outcome.DONE))

    @property
    def pending(self) -> int:
        return len(self.by_outcome(Outcome.PENDING))

    @property
    def failed(self) -> int:
        return len(self.by_outcome(Outcome.FAILED))

    @property
    def wrote_nothing(self) -> bool:
        """True when the pass published no replacement at all.

        A second pass over unchanged inputs must satisfy this: it is the law
        that says inspection is authoritative rather than merely advisory.
        """
        return self.done == 0


class DerivationAdapter(Protocol):
    """One domain's derivation. The domain owns its storage, SQL, and atomicity."""

    @property
    def domain(self) -> str: ...

    @property
    def prerequisites(self) -> tuple[str, ...]:
        """Output domains this derivation reads.

        Order comes from declared inputs, never from position in a stage list,
        so a poisoned key blocks its own dependants and nothing else.
        """
        ...

    def required(self, frame: DerivationFrame) -> Iterable[str]:
        """Every key that must exist at this frame, empty output included."""
        ...

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus | str]:
        """Classify each key from the authoritative output relation.

        Statuses may be returned as their string values so a domain adapter in
        a package that may not import this ring still speaks the vocabulary.
        """
        ...

    def compute(self, frame: DerivationFrame, key: str) -> ReplacementLike:
        """Produce a replacement from a stable read snapshot, lease-free."""
        ...

    def publish(self, frame: DerivationFrame, replacement: Any) -> bool:
        """Atomically replace one key under the writer lease.

        Returns False when the bound inputs moved under the computation; the
        replacement is discarded and the key stays pending. Publishing anyway
        is how stale output reaches an authoritative relation.
        """
        ...

    def excess_candidates(self, frame: DerivationFrame) -> Iterable[str]:
        """Keys present in the output that ``required`` no longer names."""
        ...

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        """Whether policy defers this key now (a hot file, a paused source)."""
        ...


class DerivationRegistry:
    """The declared derivations and the order their dependencies imply."""

    def __init__(self, adapters: Iterable[DerivationAdapter] = ()) -> None:
        self._adapters: dict[str, DerivationAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: DerivationAdapter) -> None:
        domain = adapter.domain
        if domain in self._adapters:
            raise ValueError(f"derivation domain {domain!r} is already registered")
        self._adapters[domain] = adapter
        self._resolve_order()

    def __contains__(self, domain: object) -> bool:
        return domain in self._adapters

    def __iter__(self) -> Iterator[DerivationAdapter]:
        return iter(self.ordered())

    def __len__(self) -> int:
        return len(self._adapters)

    def get(self, domain: str) -> DerivationAdapter:
        return self._adapters[domain]

    def ordered(self) -> tuple[DerivationAdapter, ...]:
        return tuple(self._adapters[domain] for domain in self._resolve_order())

    def _resolve_order(self) -> tuple[str, ...]:
        """Topological order over declared prerequisites; cycles are a bug.

        Validated on every registration so an unrunnable graph is a
        construction error rather than a convergence pass that silently
        starves one domain forever.
        """
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(domain: str, path: tuple[str, ...]) -> None:
            mark = state.get(domain, 0)
            if mark == 2:
                return
            if mark == 1:
                cycle = " -> ".join((*path, domain))
                raise ValueError(f"derivation prerequisites form a cycle: {cycle}")
            state[domain] = 1
            adapter = self._adapters.get(domain)
            if adapter is not None:
                for prerequisite in adapter.prerequisites:
                    if prerequisite not in self._adapters:
                        raise ValueError(
                            f"derivation {domain!r} declares prerequisite {prerequisite!r}, which is not registered"
                        )
                    visit(prerequisite, (*path, domain))
            state[domain] = 2
            order.append(domain)

        for domain in self._adapters:
            visit(domain, ())
        return tuple(order)


def _pending_keys(
    adapter: DerivationAdapter,
    frame: DerivationFrame,
) -> tuple[list[str], dict[str, KeyStatus]]:
    """Required minus valid, plus excess. The complete pending set, derived."""
    required = list(dict.fromkeys(adapter.required(frame)))
    statuses: dict[str, KeyStatus] = (
        {key: KeyStatus(status) for key, status in adapter.inspect(frame, required).items()} if required else {}
    )
    for key in required:
        statuses.setdefault(key, KeyStatus.MISSING)
    for key in dict.fromkeys(adapter.excess_candidates(frame)):
        if key not in statuses:
            statuses[key] = KeyStatus.EXCESS
    pending = [key for key, status in statuses.items() if status is not KeyStatus.VALID]
    return pending, statuses


def converge(
    registry: DerivationRegistry,
    frame: DerivationFrame,
    *,
    budget: int | None = None,
    deadline_s: float | None = None,
    publisher: Callable[[str, Callable[[], bool]], bool] | None = None,
    domains: Sequence[str] | None = None,
) -> DerivationReport:
    """Run one bounded convergence pass and report every key's typed outcome.

    ``publisher`` runs an adapter's publish under the caller's writer lease;
    absent, publish is called directly, which is what a synchronous facade that
    already owns its lease wants. Compute always runs outside it.

    Budget and deadline yield PENDING, never FAILED: exhausting a bound is
    policy, and the next pass recomputes the same pending set from the output
    relations.
    """
    outcomes: list[KeyOutcome] = []
    blocked_domains: set[str] = set()
    published = 0
    started = time.monotonic()

    selected = registry.ordered()
    if domains is not None:
        wanted = set(domains)
        selected = tuple(adapter for adapter in selected if adapter.domain in wanted)

    for adapter in selected:
        domain = adapter.domain
        blocking = tuple(name for name in adapter.prerequisites if name in blocked_domains)
        try:
            pending, _statuses = _pending_keys(adapter, frame)
        except Exception as exc:
            logger.warning("derivation %s: inspection failed: %s", domain, exc, exc_info=True)
            outcomes.append(
                KeyOutcome(
                    key=DerivationKey(domain, "*"),
                    outcome=Outcome.FAILED,
                    error=f"inspect: {exc}",
                )
            )
            blocked_domains.add(domain)
            continue

        if blocking and pending:
            reason = f"prerequisite {blocking[0]!r} has an unconverged key"
            outcomes.extend(
                KeyOutcome(
                    key=DerivationKey(domain, key),
                    outcome=Outcome.PENDING,
                    reason=PendingReason.BLOCKED,
                    error=reason,
                )
                for key in pending
            )
            blocked_domains.add(domain)
            continue

        domain_failed = False
        for key in pending:
            derivation_key = DerivationKey(domain, key)
            if budget is not None and published >= budget:
                outcomes.append(KeyOutcome(key=derivation_key, outcome=Outcome.PENDING, reason=PendingReason.BUDGET))
                continue
            if deadline_s is not None and time.monotonic() - started >= deadline_s:
                outcomes.append(KeyOutcome(key=derivation_key, outcome=Outcome.PENDING, reason=PendingReason.BUDGET))
                continue
            try:
                if adapter.quiet(frame, key):
                    outcomes.append(KeyOutcome(key=derivation_key, outcome=Outcome.PENDING, reason=PendingReason.QUIET))
                    continue
            except Exception as exc:
                logger.warning("derivation %s: quiet policy failed for %s: %s", domain, key, exc)
                outcomes.append(KeyOutcome(key=derivation_key, outcome=Outcome.FAILED, error=f"quiet: {exc}"))
                domain_failed = True
                continue

            started_key = time.monotonic()
            try:
                replacement = adapter.compute(frame, key)
            except Exception as exc:
                logger.warning("derivation %s: compute failed for %s: %s", domain, key, exc, exc_info=True)
                outcomes.append(
                    KeyOutcome(
                        key=derivation_key,
                        outcome=Outcome.FAILED,
                        error=f"compute: {exc}",
                        elapsed_s=time.monotonic() - started_key,
                    )
                )
                domain_failed = True
                continue

            def _publish(adapter: DerivationAdapter = adapter, replacement: ReplacementLike = replacement) -> bool:
                return adapter.publish(frame, replacement)

            try:
                accepted = publisher(domain, _publish) if publisher is not None else _publish()
            except Exception as exc:
                logger.warning("derivation %s: publish failed for %s: %s", domain, key, exc, exc_info=True)
                outcomes.append(
                    KeyOutcome(
                        key=derivation_key,
                        outcome=Outcome.FAILED,
                        error=f"publish: {exc}",
                        elapsed_s=time.monotonic() - started_key,
                    )
                )
                domain_failed = True
                continue

            elapsed = time.monotonic() - started_key
            if accepted:
                published += 1
                outcomes.append(KeyOutcome(key=derivation_key, outcome=Outcome.DONE, elapsed_s=elapsed))
            else:
                outcomes.append(
                    KeyOutcome(
                        key=derivation_key,
                        outcome=Outcome.PENDING,
                        reason=PendingReason.BINDING_MOVED,
                        elapsed_s=elapsed,
                    )
                )

        if domain_failed:
            blocked_domains.add(domain)

    return DerivationReport(frame=frame, outcomes=tuple(outcomes))


class BaseDerivation:
    """Optional base supplying the two derivation members most domains omit.

    ``excess_candidates`` and ``quiet`` are part of the contract because some
    domains genuinely need them; defaulting them here keeps a domain that does
    not from declaring empty stubs.
    """

    def excess_candidates(self, frame: DerivationFrame) -> Iterable[str]:
        return ()

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        return False
