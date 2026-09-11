"""The derivation kernel: one correctness relation for every derived model.

A derived model is current when the authoritative output relation says so.
Nothing else may certify it — not a queue, a cursor, a debt row, a freshness
flag, or a receipt. The kernel knows six things and no domain SQL:

``required_page``      one bounded page of the stable keys that must exist,
                       including keys whose correct output is empty, plus the
                       cursor that resumes enumeration after it
``inspect``            each key classified VALID / MISSING / STALE / EXCESS from
                       the output relation and its own binding
``prerequisite_keys``  the concrete upstream keys one candidate reads
``compute``            a typed replacement, produced outside the writer lease
``publish``            revalidate the bound inputs and atomically replace
                       exactly that key under the lease
``prerequisites``      the output domains this one reads

The pending set is ``required`` minus ``valid``. Deleting every scheduling hint
and restarting reconstructs it, because it was never stored.

**Nothing here is proportional to the archive.** A pass pulls pages, not
universes: discovery, inspection, computation, publication and the number of
retained per-key results are each separately bounded, and a pass that runs out
of any bound stops at a page position instead of enumerating the remainder to
label it. The position it stopped at is returned in the report as a
:class:`PassCursor` — a process-local hint with no durable authority. A caller
that keeps it gets fairness (a permanently quiet or permanently refusing prefix
cannot starve the tail); a caller that drops it loses nothing but its place,
because validity is always re-derived from the output relations.

**Publication is certified by the output relation, not by its own return
value.** ``publish`` returning True is a claim; the kernel re-inspects the key
and only reports DONE when the authoritative relation agrees. A domain that
reports success while leaving its output non-valid is a defect, and this is
where it surfaces.

Domains keep their own storage, SQL, and atomic replacement boundary: this is
not a generic artifact store, and adding one would create a second truth beside
the output relations that are already authoritative.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, cast

from polylogue.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_PAGE",
    "BaseDerivation",
    "Budget",
    "DerivationAdapter",
    "DerivationFrame",
    "DerivationKey",
    "DerivationRegistry",
    "DerivationReport",
    "DiscoveryPhase",
    "DomainCursor",
    "KeyOutcome",
    "KeyPage",
    "KeyStatus",
    "LegacyDerivationAdapter",
    "Outcome",
    "PageLike",
    "PassCursor",
    "PendingReason",
    "Replacement",
    "ReplacementLike",
    "WorkCounters",
    "converge",
]

#: Keys requested per discovery call when the caller declares no page budget.
DEFAULT_PAGE = 128


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
class KeyPage:
    """One bounded slice of a domain's key space plus how to continue it.

    ``next_cursor is None`` means the slice reached the end of the space. Any
    other value is passed back verbatim on the next request, so a domain backed
    by a table implements paging as a keyset scan (``WHERE key > ? ORDER BY key
    LIMIT ?``) and never sorts or counts the whole relation to serve one page.
    """

    keys: tuple[str, ...] = ()
    next_cursor: str | None = None


class PageLike(Protocol):
    """A page as the kernel reads it, so a foreign ring can return its own."""

    @property
    def keys(self) -> Sequence[str]: ...

    @property
    def next_cursor(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class Budget:
    """Every bound one pass may not exceed. ``None`` means unbounded.

    The bounds are separate because the work they meter is separate: a domain
    of a million valid keys costs discovery and inspection and nothing else,
    while a domain of ten stale keys costs computation and publication. A
    single "budget" number cannot bound both, and the one that mattered was
    always the one that was not being counted.

    ``retained_outcomes`` bounds the report, not the work: beyond it the pass
    still counts every outcome but stops keeping a result object per key, which
    is what keeps a bounded pass over an unbounded domain bounded in memory.
    """

    page: int = DEFAULT_PAGE
    discovery: int | None = None
    inspection: int | None = None
    compute: int | None = None
    publication: int | None = None
    retained_outcomes: int | None = None
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError("page budget must request at least one key")

    @classmethod
    def coerce(cls, value: Budget | int | None, *, deadline_s: float | None = None) -> Budget:
        """Accept a budget, a bare publication count, or nothing.

        A bare integer keeps the pre-paging caller's meaning -- "publish at most
        this many keys" -- so an existing caller does not silently acquire an
        unbounded discovery sweep it never asked for.
        """
        if isinstance(value, Budget):
            if deadline_s is None or value.deadline_s is not None:
                return value
            return replace(value, deadline_s=deadline_s)
        if value is None:
            return cls(deadline_s=deadline_s)
        return cls(publication=int(value), deadline_s=deadline_s)


@dataclass(frozen=True, slots=True)
class WorkCounters:
    """What one pass actually did, independent of how much it retained.

    These are the numbers a bound is checked against. They count calls and
    keys, not rows: a domain's own SQL cost is the domain's to report.
    """

    pages: int = 0
    discovered: int = 0
    inspected: int = 0
    prerequisites_inspected: int = 0
    computed: int = 0
    published: int = 0


class DiscoveryPhase(Enum):
    """Which key space a domain's enumeration is currently walking."""

    REQUIRED = "required"
    EXCESS = "excess"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class DomainCursor:
    """Where one domain's enumeration stopped, and how to resume it.

    ``page_cursor`` is the opaque value that *produced* the current page and
    ``offset`` is how many of that page's keys were consumed, so resuming
    re-requests one page and skips into it. Carrying the produced page's own
    cursor instead of a per-key cursor keeps the adapter contract at "give me
    the next page", which a keyset scan can serve without a stable per-key
    address.
    """

    phase: DiscoveryPhase = DiscoveryPhase.REQUIRED
    page_cursor: str | None = None
    offset: int = 0

    @property
    def swept(self) -> bool:
        """True when the last pass walked this domain's key space to the end."""
        return self.phase is DiscoveryPhase.DONE


@dataclass(frozen=True, slots=True)
class PassCursor:
    """Process-local resume positions, one per domain. Never durable.

    A cursor cannot certify anything: it only moves where a pass *starts*
    looking. Losing it re-starts every sweep at the beginning, which is
    correct and merely less fair; keeping it is what stops a permanently quiet
    or permanently refusing prefix from starving the keys behind it. A domain
    whose sweep finished is recorded as swept and starts over on the next pass,
    so every key is visited again within one sweep.
    """

    positions: Mapping[str, DomainCursor] = field(default_factory=dict)

    def position(self, domain: str) -> DomainCursor:
        return self.positions.get(domain, DomainCursor())

    def with_position(self, domain: str, cursor: DomainCursor) -> PassCursor:
        merged = dict(self.positions)
        merged[domain] = cursor
        return PassCursor(merged)


@dataclass(frozen=True, slots=True)
class DerivationReport:
    """What one bounded convergence pass did, in totals and in a bounded sample.

    ``counts`` is authoritative for how many keys reached each outcome;
    ``outcomes`` is the retained sample of per-key detail, which a
    ``retained_outcomes`` budget may truncate. Reading a total off ``outcomes``
    is the mistake this split exists to prevent.
    """

    frame: DerivationFrame
    outcomes: tuple[KeyOutcome, ...] = ()
    counts: Mapping[Outcome, int] = field(default_factory=dict)
    work: WorkCounters = WorkCounters()
    cursor: PassCursor = PassCursor()
    truncated: bool = False

    def by_outcome(self, outcome: Outcome) -> tuple[KeyOutcome, ...]:
        """The retained per-key detail for one outcome, not a count of it."""
        return tuple(item for item in self.outcomes if item.outcome is outcome)

    def count(self, outcome: Outcome) -> int:
        return int(self.counts.get(outcome, 0))

    @property
    def done(self) -> int:
        return self.count(Outcome.DONE)

    @property
    def pending(self) -> int:
        return self.count(Outcome.PENDING)

    @property
    def failed(self) -> int:
        return self.count(Outcome.FAILED)

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
        so a poisoned key blocks its own dependants and nothing else. This is
        the coarse edge; ``prerequisite_keys`` is the precise one.
        """
        ...

    def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> PageLike | Any:
        """One page of the keys that must exist at this frame, empty output included.

        Must return at most ``limit`` keys and a cursor that strictly advances,
        or ``None`` when the space is exhausted. May be returned as a
        ``(keys, next_cursor)`` pair so an adapter in a package that may not
        import this ring still speaks the contract.
        """
        ...

    def excess_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> PageLike | Any:
        """One page of keys the output holds that ``required_page`` no longer names.

        Must not repeat a key the required space contains; the kernel walks the
        two spaces in sequence and does not cross-check them, because doing so
        would mean holding one of them entire.
        """
        ...

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus | str]:
        """Classify each key from the authoritative output relation.

        Statuses may be returned as their string values so a domain adapter in
        a package that may not import this ring still speaks the vocabulary.
        """
        ...

    def prerequisite_keys(self, frame: DerivationFrame, key: str) -> Iterable[DerivationKey | tuple[str, str]]:
        """The concrete upstream keys this candidate reads.

        The kernel inspects these bindings whether or not their domain was
        selected for this pass, so a dependant is gated on the inputs it
        actually reads rather than on which domains happened to run.
        """
        ...

    def compute(self, frame: DerivationFrame, key: str) -> ReplacementLike:
        """Produce a replacement from a stable read snapshot, lease-free."""
        ...

    def publish(self, frame: DerivationFrame, replacement: Any) -> bool:
        """Atomically replace one key under the writer lease.

        Returns False when the bound inputs moved under the computation; the
        replacement is discarded and the key stays pending. Publishing anyway
        is how stale output reaches an authoritative relation. Returning True
        is a claim the kernel re-inspects, not a certification.
        """
        ...

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        """Whether policy defers this key now (a hot file, a paused source)."""
        ...


class LegacyDerivationAdapter(Protocol):
    """The pre-paging vocabulary: one call hands over the whole key space.

    Accepted so a storage-ring adapter can move to ``required_page`` in its own
    change rather than in this one. That is not politeness: the session-profile
    adapter lives inside the derived schema closure, where any edit moves the
    schema identity and obliges the archive to reconverge -- a price a dormant
    contract change must not charge.

    The kernel pages this ordinally, so the *kernel* still holds one page at a
    time. The adapter goes on materializing its own key space, and that cost is
    the adapter's to remove when it migrates.
    """

    @property
    def domain(self) -> str: ...

    @property
    def prerequisites(self) -> tuple[str, ...]: ...

    def required(self, frame: DerivationFrame) -> Iterable[str]: ...

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus | str]: ...

    def compute(self, frame: DerivationFrame, key: str) -> ReplacementLike: ...

    def publish(self, frame: DerivationFrame, replacement: Any) -> bool: ...


class _LegacyPaging:
    """Adapts the pre-paging vocabulary onto the page contract."""

    def __init__(self, adapter: LegacyDerivationAdapter) -> None:
        self._adapter = adapter

    @property
    def domain(self) -> str:
        return self._adapter.domain

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return self._adapter.prerequisites

    def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
        return _page_from_iterable(self._adapter.required(frame), cursor=cursor, limit=limit)

    def excess_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
        excess = getattr(self._adapter, "excess_candidates", None)
        return _page_from_iterable(excess(frame) if excess is not None else (), cursor=cursor, limit=limit)

    def prerequisite_keys(self, frame: DerivationFrame, key: str) -> Iterable[DerivationKey | tuple[str, str]]:
        bindings = getattr(self._adapter, "prerequisite_keys", None)
        return bindings(frame, key) if bindings is not None else ()

    def inspect(self, frame: DerivationFrame, keys: Sequence[str]) -> Mapping[str, KeyStatus | str]:
        return self._adapter.inspect(frame, keys)

    def compute(self, frame: DerivationFrame, key: str) -> ReplacementLike:
        return self._adapter.compute(frame, key)

    def publish(self, frame: DerivationFrame, replacement: Any) -> bool:
        return self._adapter.publish(frame, replacement)

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        policy = getattr(self._adapter, "quiet", None)
        return bool(policy(frame, key)) if policy is not None else False


class DerivationRegistry:
    """The declared derivations and the order their dependencies imply."""

    def __init__(self, adapters: Iterable[DerivationAdapter | LegacyDerivationAdapter] = ()) -> None:
        self._adapters: dict[str, DerivationAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: DerivationAdapter | LegacyDerivationAdapter) -> None:
        domain = adapter.domain
        if domain in self._adapters:
            raise ValueError(f"derivation domain {domain!r} is already registered")
        # Which vocabulary the object speaks is a property of the object, not a
        # flag anyone sets: an adapter that pages is used directly, one that
        # does not is paged here.
        paged: DerivationAdapter = (
            cast("DerivationAdapter", adapter)
            if callable(getattr(adapter, "required_page", None))
            else _LegacyPaging(cast("LegacyDerivationAdapter", adapter))
        )
        self._adapters[domain] = paged

    def __contains__(self, domain: object) -> bool:
        return domain in self._adapters

    def __iter__(self) -> Iterator[DerivationAdapter]:
        return iter(self.ordered())

    def __len__(self) -> int:
        return len(self._adapters)

    def get(self, domain: str) -> DerivationAdapter:
        return self._adapters[domain]

    def ordered(self) -> tuple[DerivationAdapter, ...]:
        return tuple(self._adapters[domain] for domain in self.validate())

    def validate(self) -> tuple[str, ...]:
        """Resolve the run order, refusing a graph that cannot run.

        Validation is deferred to resolution rather than performed per
        registration: a domain may legitimately be registered before the
        prerequisite it declares, and refusing that would make registration
        order load-bearing. An unrunnable graph is still a construction error,
        not a convergence pass that silently starves one domain forever.
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


_MISSING = object()


def _as_page(value: object) -> KeyPage:
    """Read a page from whatever shape the adapter's ring returns."""
    if isinstance(value, KeyPage):
        return value
    keys = getattr(value, "keys", None)
    if keys is not None and not callable(keys):
        cursor = getattr(value, "next_cursor", None)
        return KeyPage(tuple(str(item) for item in keys), None if cursor is None else str(cursor))
    if isinstance(value, tuple) and len(value) == 2:
        raw_keys, cursor = value
        if isinstance(raw_keys, (list, tuple)):
            return KeyPage(tuple(str(item) for item in raw_keys), None if cursor is None else str(cursor))
    raise TypeError(f"derivation page must be a KeyPage or a (keys, next_cursor) pair, got {type(value).__name__}")


def _as_key(value: DerivationKey | tuple[str, str]) -> DerivationKey:
    if isinstance(value, DerivationKey):
        return value
    domain, key = value
    return DerivationKey(str(domain), str(key))


def _coerce_statuses(statuses: Mapping[str, KeyStatus | str]) -> dict[str, KeyStatus]:
    return {key: KeyStatus(status) for key, status in statuses.items()}


def _page_from_iterable(source: Iterable[str], *, cursor: str | None, limit: int) -> KeyPage:
    """Page a lazy iterable by ordinal offset.

    The default paging for a domain small enough that re-walking its prefix is
    free. It is O(offset) per page and therefore quadratic over a sweep: a
    domain whose key space is large enough for that to matter implements
    ``required_page`` as a keyset scan instead. The iterable must be lazy --
    returning a materialized list here re-materializes the universe on every
    page, which is exactly the cost the page contract exists to remove.
    """
    start = int(cursor) if cursor else 0
    iterator = iter(source)
    for _ in range(start):
        if next(iterator, _MISSING) is _MISSING:
            return KeyPage((), None)
    keys: list[str] = []
    for item in iterator:
        keys.append(str(item))
        if len(keys) >= limit:
            break
    return KeyPage(tuple(keys), str(start + len(keys)) if len(keys) >= limit else None)


class BaseDerivation:
    """Optional base supplying the derivation members most domains omit.

    A domain that is small, or whose key space is naturally an iterator,
    implements ``required_keys``/``excess_keys`` and inherits ordinal paging.
    A domain backed by a large table overrides ``required_page`` directly.
    """

    def required_keys(self, frame: DerivationFrame) -> Iterable[str]:
        """Every key that must exist at this frame, as a lazy iterable."""
        raise NotImplementedError("a derivation must declare required_keys or override required_page")

    def required_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
        return _page_from_iterable(self.required_keys(frame), cursor=cursor, limit=limit)

    def excess_keys(self, frame: DerivationFrame) -> Iterable[str]:
        return ()

    def excess_page(self, frame: DerivationFrame, *, cursor: str | None, limit: int) -> KeyPage:
        return _page_from_iterable(self.excess_keys(frame), cursor=cursor, limit=limit)

    def prerequisite_keys(self, frame: DerivationFrame, key: str) -> Iterable[DerivationKey | tuple[str, str]]:
        return ()

    def quiet(self, frame: DerivationFrame, key: str) -> bool:
        return False


class _Pass:
    """The mutable state of one bounded convergence pass."""

    def __init__(
        self,
        registry: DerivationRegistry,
        frame: DerivationFrame,
        budget: Budget,
        publisher: Callable[[str, Callable[[], bool]], bool] | None,
    ) -> None:
        self.registry = registry
        self.frame = frame
        self.budget = budget
        self.publisher = publisher
        self.started = time.monotonic()
        self.counts: dict[Outcome, int] = {Outcome.DONE: 0, Outcome.PENDING: 0, Outcome.FAILED: 0}
        self.retained: list[KeyOutcome] = []
        self.truncated = False
        self.pages = 0
        self.discovered = 0
        self.inspected = 0
        self.prerequisites_inspected = 0
        self.computed = 0
        self.published = 0
        #: Every key this pass reached a verdict on, so a dependant can be gated
        #: on what just happened rather than on a re-read that would not see it.
        self.verdicts: dict[DerivationKey, Outcome] = {}
        #: Domains whose output relation could not be read at all this pass.
        self.unreadable_domains: set[str] = set()
        #: Domains this pass observed holding at least one non-valid key.
        self.unconverged_domains: set[str] = set()
        self.prerequisite_cache: dict[DerivationKey, str | None] = {}

    # ── bookkeeping ────────────────────────────────────────────────

    def record(self, outcome: KeyOutcome) -> None:
        self.counts[outcome.outcome] += 1
        self.verdicts[outcome.key] = outcome.outcome
        if outcome.outcome is not Outcome.DONE:
            self.unconverged_domains.add(outcome.key.domain)
        cap = self.budget.retained_outcomes
        if cap is not None and len(self.retained) >= cap:
            self.truncated = True
            return
        self.retained.append(outcome)

    def counters(self) -> WorkCounters:
        return WorkCounters(
            pages=self.pages,
            discovered=self.discovered,
            inspected=self.inspected,
            prerequisites_inspected=self.prerequisites_inspected,
            computed=self.computed,
            published=self.published,
        )

    # ── bounds ─────────────────────────────────────────────────────

    def out_of_time(self) -> bool:
        deadline = self.budget.deadline_s
        return deadline is not None and time.monotonic() - self.started >= deadline

    def work_exhausted(self) -> bool:
        """True when no further key in this pass can be computed or published."""
        budget = self.budget
        if self.out_of_time():
            return True
        if budget.publication is not None and self.published >= budget.publication:
            return True
        if budget.compute is not None and self.computed >= budget.compute:
            return True
        return budget.inspection is not None and self.inspected >= budget.inspection

    def page_limit(self) -> int:
        """How many keys the next discovery call may return."""
        limit = self.budget.page
        if self.budget.discovery is not None:
            limit = min(limit, max(0, self.budget.discovery - self.discovered))
        if self.budget.inspection is not None:
            limit = min(limit, max(0, self.budget.inspection - self.inspected))
        return limit

    # ── discovery ──────────────────────────────────────────────────

    def fetch(self, adapter: DerivationAdapter, position: DomainCursor, limit: int) -> KeyPage:
        if position.phase is DiscoveryPhase.REQUIRED:
            raw = adapter.required_page(self.frame, cursor=position.page_cursor, limit=limit)
        else:
            raw = adapter.excess_page(self.frame, cursor=position.page_cursor, limit=limit)
        page = _as_page(raw)
        self.pages += 1
        # A page request costs at least one unit even when it yields nothing, so
        # a discovery budget bounds calls and not merely keys.
        self.discovered += max(1, len(page.keys))
        if len(page.keys) > limit:
            raise ValueError(f"derivation {adapter.domain} returned {len(page.keys)} keys for a limit of {limit}")
        if page.next_cursor is not None and page.next_cursor == position.page_cursor:
            raise ValueError(f"derivation {adapter.domain} returned a cursor that does not advance")
        return page

    @staticmethod
    def advance(position: DomainCursor, page: KeyPage) -> DomainCursor:
        """Move past a fully consumed page: next page, next phase, or done."""
        if page.next_cursor is not None:
            return DomainCursor(position.phase, page.next_cursor, 0)
        if position.phase is DiscoveryPhase.REQUIRED:
            return DomainCursor(DiscoveryPhase.EXCESS, None, 0)
        return DomainCursor(DiscoveryPhase.DONE, None, 0)

    # ── prerequisites ──────────────────────────────────────────────

    def prerequisite_block(self, adapter: DerivationAdapter, key: str) -> str | None:
        """Why this candidate may not publish yet, or None when its inputs are ready.

        A domain that says which keys it reads is gated on exactly those keys,
        so one unconverged upstream key blocks the dependants that read it and
        no others. The declared ``prerequisites`` domain edge is the fallback
        for a domain that does not: it can only be evaluated at whole-domain
        granularity, which is coarse but never optimistic.
        """
        try:
            bindings = tuple(_as_key(item) for item in adapter.prerequisite_keys(self.frame, key))
        except Exception as exc:
            return f"prerequisite mapping failed: {exc}"
        if bindings:
            for binding in bindings:
                reason = self.binding_block(binding)
                if reason is not None:
                    return reason
            return None
        for name in adapter.prerequisites:
            if name in self.unreadable_domains:
                return f"prerequisite domain {name!r} could not be inspected"
            if name in self.unconverged_domains:
                return f"prerequisite domain {name!r} has an unconverged key"
        return None

    def binding_block(self, binding: DerivationKey) -> str | None:
        if binding.domain in self.unreadable_domains:
            return f"prerequisite {binding} could not be inspected"
        visited = self.verdicts.get(binding)
        if visited is Outcome.DONE:
            return None
        if visited is not None:
            return f"prerequisite {binding} is {visited.value} in this pass"
        if binding in self.prerequisite_cache:
            return self.prerequisite_cache[binding]
        reason = self.inspect_binding(binding)
        self.prerequisite_cache[binding] = reason
        return reason

    def inspect_binding(self, binding: DerivationKey) -> str | None:
        """Read one upstream key's authority, whatever this pass selected."""
        try:
            upstream = self.registry.get(binding.domain)
        except KeyError:
            return f"prerequisite {binding} names a domain that is not registered"
        self.prerequisites_inspected += 1
        try:
            statuses = _coerce_statuses(dict(upstream.inspect(self.frame, (binding.key,))))
        except Exception as exc:
            logger.warning("derivation: prerequisite inspection failed for %s: %s", binding, exc, exc_info=True)
            self.unreadable_domains.add(binding.domain)
            return f"prerequisite {binding} could not be inspected: {exc}"
        status = statuses.get(binding.key, KeyStatus.MISSING)
        if status is not KeyStatus.VALID:
            return f"prerequisite {binding} is {status.value}"
        try:
            if upstream.quiet(self.frame, binding.key):
                return f"prerequisite {binding} is quiet"
        except Exception as exc:
            return f"prerequisite {binding} quiet policy failed: {exc}"
        return None

    # ── one key ────────────────────────────────────────────────────

    def process(self, adapter: DerivationAdapter, key: str, *, retiring: bool = False) -> None:
        """Converge one key, certifying the result against the output relation.

        ``retiring`` marks a key discovered in the excess space: its correct end
        state is absence, so the certification requires MISSING where a required
        key requires VALID. Without the distinction a correct retirement would
        report as a failed publication and repeat forever.
        """
        derivation_key = DerivationKey(adapter.domain, key)
        expected = KeyStatus.MISSING if retiring else KeyStatus.VALID

        blocked = self.prerequisite_block(adapter, key)
        if blocked is not None:
            self.record(
                KeyOutcome(key=derivation_key, outcome=Outcome.PENDING, reason=PendingReason.BLOCKED, error=blocked)
            )
            return

        try:
            if adapter.quiet(self.frame, key):
                self.record(KeyOutcome(key=derivation_key, outcome=Outcome.PENDING, reason=PendingReason.QUIET))
                return
        except Exception as exc:
            logger.warning("derivation %s: quiet policy failed for %s: %s", adapter.domain, key, exc)
            self.record(KeyOutcome(key=derivation_key, outcome=Outcome.FAILED, error=f"quiet: {exc}"))
            return

        started_key = time.monotonic()
        self.computed += 1
        try:
            replacement = adapter.compute(self.frame, key)
        except Exception as exc:
            logger.warning("derivation %s: compute failed for %s: %s", adapter.domain, key, exc, exc_info=True)
            self.record(
                KeyOutcome(
                    key=derivation_key,
                    outcome=Outcome.FAILED,
                    error=f"compute: {exc}",
                    elapsed_s=time.monotonic() - started_key,
                )
            )
            return

        def _publish(adapter: DerivationAdapter = adapter, replacement: ReplacementLike = replacement) -> bool:
            return adapter.publish(self.frame, replacement)

        try:
            accepted = self.publisher(adapter.domain, _publish) if self.publisher is not None else _publish()
        except Exception as exc:
            logger.warning("derivation %s: publish failed for %s: %s", adapter.domain, key, exc, exc_info=True)
            self.record(
                KeyOutcome(
                    key=derivation_key,
                    outcome=Outcome.FAILED,
                    error=f"publish: {exc}",
                    elapsed_s=time.monotonic() - started_key,
                )
            )
            return

        elapsed = time.monotonic() - started_key
        if not accepted:
            self.record(
                KeyOutcome(
                    key=derivation_key,
                    outcome=Outcome.PENDING,
                    reason=PendingReason.BINDING_MOVED,
                    elapsed_s=elapsed,
                )
            )
            return

        # The output relation certifies the publication, not its return value.
        # This inspection is part of publishing and is therefore counted but
        # never budget-refused: a pass that skipped it would report DONE for a
        # key it never confirmed.
        self.inspected += 1
        try:
            after = _coerce_statuses(dict(adapter.inspect(self.frame, (key,)))).get(key, KeyStatus.MISSING)
        except Exception as exc:
            logger.warning("derivation %s: post-publication inspection failed for %s: %s", adapter.domain, key, exc)
            self.unreadable_domains.add(adapter.domain)
            self.record(
                KeyOutcome(
                    key=derivation_key,
                    outcome=Outcome.FAILED,
                    error=f"reinspect: {exc}",
                    elapsed_s=elapsed,
                )
            )
            return
        if after is not expected:
            self.record(
                KeyOutcome(
                    key=derivation_key,
                    outcome=Outcome.FAILED,
                    error=(
                        f"publish reported success but the output relation reports {after.value}, not {expected.value}"
                    ),
                    elapsed_s=elapsed,
                )
            )
            return

        self.published += 1
        self.record(KeyOutcome(key=derivation_key, outcome=Outcome.DONE, elapsed_s=elapsed))

    # ── one domain ─────────────────────────────────────────────────

    def run_domain(self, adapter: DerivationAdapter, start: DomainCursor) -> DomainCursor:
        domain = adapter.domain
        # A finished sweep starts over: wraparound is what makes "eventually
        # every key" true for a caller that keeps its cursor.
        position = DomainCursor() if start.swept else start

        while position.phase is not DiscoveryPhase.DONE and not self.work_exhausted():
            limit = self.page_limit()
            if limit <= 0:
                break
            try:
                page = self.fetch(adapter, position, limit)
            except Exception as exc:
                logger.warning("derivation %s: discovery failed: %s", domain, exc, exc_info=True)
                self.record(
                    KeyOutcome(key=DerivationKey(domain, "*"), outcome=Outcome.FAILED, error=f"discover: {exc}")
                )
                self.unreadable_domains.add(domain)
                break

            keys = page.keys[position.offset :]
            if not keys:
                position = self.advance(position, page)
                continue

            if position.phase is DiscoveryPhase.REQUIRED:
                try:
                    statuses = _coerce_statuses(dict(adapter.inspect(self.frame, keys)))
                except Exception as exc:
                    logger.warning("derivation %s: inspection failed: %s", domain, exc, exc_info=True)
                    self.record(
                        KeyOutcome(key=DerivationKey(domain, "*"), outcome=Outcome.FAILED, error=f"inspect: {exc}")
                    )
                    self.unreadable_domains.add(domain)
                    break
                self.inspected += len(keys)
            else:
                statuses = dict.fromkeys(keys, KeyStatus.EXCESS)

            stopped_at: int | None = None
            for index, key in enumerate(keys):
                if statuses.get(key, KeyStatus.MISSING) is KeyStatus.VALID:
                    continue
                if stopped_at is not None or self.work_exhausted():
                    # Already classified, so it is reported; not attempted, so
                    # the resume position stays behind it.
                    if stopped_at is None:
                        stopped_at = index
                    self.record(
                        KeyOutcome(key=DerivationKey(domain, key), outcome=Outcome.PENDING, reason=PendingReason.BUDGET)
                    )
                    continue
                self.process(adapter, key, retiring=position.phase is DiscoveryPhase.EXCESS)

            if stopped_at is not None:
                return DomainCursor(position.phase, position.page_cursor, position.offset + stopped_at)
            position = self.advance(position, page)

        return position


def converge(
    registry: DerivationRegistry,
    frame: DerivationFrame,
    *,
    budget: Budget | int | None = None,
    deadline_s: float | None = None,
    publisher: Callable[[str, Callable[[], bool]], bool] | None = None,
    domains: Sequence[str] | None = None,
    cursor: PassCursor | None = None,
) -> DerivationReport:
    """Run one bounded convergence pass and report every key it reached.

    ``publisher`` runs an adapter's publish under the caller's writer lease;
    absent, publish is called directly, which is what a synchronous facade that
    already owns its lease wants. Compute always runs outside it.

    ``cursor`` resumes the previous pass's positions. It is a hint with no
    authority: passing none re-starts each sweep at the beginning and reaches
    the same verdicts, because every verdict comes from the output relation.

    Budget and deadline yield PENDING, never FAILED: exhausting a bound is
    policy, and the next pass recomputes the same pending set from the output
    relations. Keys beyond the page a bound stopped in are not enumerated at
    all -- a pass over a domain of a million keys reports the page it looked
    at, not a million pending results.
    """
    limits = Budget.coerce(budget, deadline_s=deadline_s)
    resume = cursor or PassCursor()
    state = _Pass(registry, frame, limits, publisher)

    selected = registry.ordered()
    if domains is not None:
        wanted = set(domains)
        selected = tuple(adapter for adapter in selected if adapter.domain in wanted)

    positions = dict(resume.positions)
    for adapter in selected:
        positions[adapter.domain] = state.run_domain(adapter, resume.position(adapter.domain))

    return DerivationReport(
        frame=frame,
        outcomes=tuple(state.retained),
        counts=dict(state.counts),
        work=state.counters(),
        cursor=PassCursor(positions),
        truncated=state.truncated,
    )
