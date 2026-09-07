"""A measurement and the state it was measured in, as one inseparable value.

A read that can fail has four outcomes, and collapsing any of them into another
loses the only fact an operator needs: ``Measured`` (the store answered and the
answer is a value), ``Empty`` (the store answered and the answer is nothing),
``Unavailable`` (the store did not answer) and ``Degraded`` (the store answered
partially, and the partial answer is still worth reporting beside its reason).

The union deliberately exposes no accessor that yields the value without
consuming the state. ``resolve`` requires a handler for every case, so a
renderer that ignores unavailability does not type-check; ``match`` over the
four classes is exhaustive for the same reason. Masking an error as zero rows
therefore has to be written out in full, at the call site, where review sees it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")

__all__ = [
    "Degraded",
    "Empty",
    "Evidence",
    "Measured",
    "Unavailable",
    "evidence_state",
    "measured_or_none",
    "resolve",
]


@dataclass(frozen=True, slots=True)
class Measured(Generic[T_co]):
    """The store answered and the answer is ``value``."""

    value: T_co


@dataclass(frozen=True, slots=True)
class Empty:
    """The store answered and the answer is nothing.

    Distinct from ``Unavailable``: the absence is a measured fact.
    """


@dataclass(frozen=True, slots=True)
class Unavailable:
    """The store did not answer; ``reason`` names why in a closed vocabulary."""

    reason: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class Degraded(Generic[T_co]):
    """The store answered partially; ``value`` is that partial answer."""

    value: T_co
    reason: str
    detail: str | None = None


Evidence: TypeAlias = Measured[T] | Empty | Unavailable | Degraded[T]


def resolve(
    evidence: Evidence[T],
    *,
    measured: Callable[[T], R],
    empty: Callable[[], R],
    unavailable: Callable[[Unavailable], R],
    degraded: Callable[[Degraded[T]], R],
) -> R:
    """Consume every case. All four handlers are required, by construction."""
    match evidence:
        case Measured():
            return measured(evidence.value)
        case Empty():
            return empty()
        case Unavailable():
            return unavailable(evidence)
        case Degraded():
            return degraded(evidence)


def evidence_state(evidence: Evidence[object]) -> str:
    """Name the case for a payload field, without yielding the value."""
    return resolve(
        evidence,
        measured=lambda _value: "measured",
        empty=lambda: "empty",
        unavailable=lambda _case: "unavailable",
        degraded=lambda _case: "degraded",
    )


def measured_or_none(evidence: Evidence[T]) -> T | None:
    """Yield the value only where ``None`` is itself a reported absence.

    The one escape hatch, and it is not a default: every non-value case
    collapses to ``None``, so a caller that cannot distinguish them must say so
    by calling this instead of reaching into the union.
    """
    return resolve(
        evidence,
        measured=lambda value: value,
        empty=lambda: None,
        unavailable=lambda _case: None,
        degraded=lambda case: case.value,
    )
