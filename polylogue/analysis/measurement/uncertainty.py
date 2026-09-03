"""Uncertainty by source, gated by exactness (rxdo.9.8).

The rigor program's binding anti-goal is "no p-values or sampling CI on
enumeration-exact census counts" (docs/design/analysis-rigor.md, mechanism
H). An exact enumeration over stored rows has zero *sampling* error -- there
is nothing to bootstrap -- but it can still carry independent, *named*
uncertainty from incomplete frame coverage (some sessions were never
captured) or model-derived measurement (a classifier/judge produced the
label). Those two facts do not go away just because the count itself is
exact, and they must never be silently dressed up as a sampling interval.

This module is the enforcement gate: :func:`resolve_uncertainty` refuses to
attach a bootstrap confidence interval to anything but a genuinely sampled
or estimated result, and it always renders frame/measurement uncertainty
separately from sampling uncertainty rather than collapsing all three into
one number. It consumes a local, self-contained ``enumeration`` axis rather
than assuming rxdo.3's full result-contract envelope has landed --
``EnumerationCompleteness`` here is deliberately the same four-value
vocabulary as :data:`polylogue.storage.sqlite.query_objects.ResultSetExactness`
so a caller with a real ``ResultSetManifest`` can pass its ``exactness``
field straight through.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

EnumerationCompleteness = Literal["exact", "capped", "sampled", "estimate"]
UncertaintySource = Literal["sampling", "frame", "measurement"]

# Enumeration classes that carry zero sampling error by construction. A
# bootstrap interval over a value drawn from one of these would misrepresent
# enumeration certainty as inferential uncertainty -- refuse it outright.
# The complement, {"sampled", "estimate"}, is where a sampling interval may
# legitimately apply.
_NO_SAMPLING_ERROR: frozenset[EnumerationCompleteness] = frozenset({"exact", "capped"})


class UncertaintyRefusalError(ValueError):
    """A bootstrap/sampling interval was requested where it cannot apply."""


@dataclass(frozen=True, slots=True)
class UncertaintyRendering:
    """Every fact this result carries, each named by its own source.

    ``sampling_interval`` is populated only for ``sampled``/``estimate``
    enumeration. ``frame_note``/``measurement_note`` are populated
    independently of exactness -- an exact count can still carry both.
    """

    point_estimate: float
    n: int
    sampling_interval: tuple[float, float] | None
    sampling_method: str | None
    frame_note: str | None
    measurement_note: str | None

    @property
    def uncertainty_sources(self) -> tuple[UncertaintySource, ...]:
        sources: list[UncertaintySource] = []
        if self.sampling_interval is not None:
            sources.append("sampling")
        if self.frame_note is not None:
            sources.append("frame")
        if self.measurement_note is not None:
            sources.append("measurement")
        return tuple(sources)


def bootstrap_interval(
    samples: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval over ``samples``.

    Assumption-light by design (rxdo.9's anti-goal: no general statistics
    library) -- resampling with replacement and taking the empirical
    percentile band, no normality assumption.
    """

    if len(samples) < 2:
        raise ValueError("bootstrap interval requires at least two members")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    values = list(samples)
    n = len(values)
    rng = random.Random(seed)
    resample_stats = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    lo_index = max(0, int(alpha * n_resamples))
    hi_index = min(n_resamples - 1, int((1 - alpha) * n_resamples))
    return resample_stats[lo_index], resample_stats[hi_index]


def resolve_uncertainty(
    *,
    enumeration: EnumerationCompleteness,
    point_estimate: float,
    n: int,
    frame_complete: bool = True,
    frame_note: str | None = None,
    measurement_authority: str = "structural",
    measurement_note: str | None = None,
    samples: Sequence[float] | None = None,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
    seed: int = 0,
) -> UncertaintyRendering:
    """Render the honest uncertainty picture for one result.

    Raises:
        UncertaintyRefusalError: ``samples`` was supplied for an
            ``exact``/``capped`` enumeration (sampling CI would misrepresent
            enumeration certainty), or ``enumeration`` is
            ``sampled``/``estimate`` but fewer than two samples were given
            (nothing to bootstrap).
    """

    resolved_frame_note = None if frame_complete else (frame_note or "frame coverage incomplete")
    resolved_measurement_note = (
        None
        if measurement_authority == "structural"
        else (measurement_note or f"measurement authority: {measurement_authority}")
    )

    if enumeration in _NO_SAMPLING_ERROR:
        if samples is not None:
            raise UncertaintyRefusalError(
                f"enumeration={enumeration!r} carries no sampling error; a bootstrap interval over "
                "its members would misrepresent enumeration certainty as inferential uncertainty. "
                "Frame/measurement uncertainty (if any) render independently of exactness -- pass "
                "frame_complete=False or measurement_authority=<non-structural> instead of samples."
            )
        return UncertaintyRendering(
            point_estimate=point_estimate,
            n=n,
            sampling_interval=None,
            sampling_method=None,
            frame_note=resolved_frame_note,
            measurement_note=resolved_measurement_note,
        )

    if samples is None or len(samples) < 2:
        raise UncertaintyRefusalError(
            f"enumeration={enumeration!r} requires >=2 sample members to render a sampling interval; "
            f"got {0 if samples is None else len(samples)}"
        )
    interval = bootstrap_interval(samples, statistic=statistic, seed=seed)
    return UncertaintyRendering(
        point_estimate=point_estimate,
        n=n,
        sampling_interval=interval,
        sampling_method="bootstrap-percentile",
        frame_note=resolved_frame_note,
        measurement_note=resolved_measurement_note,
    )


__all__ = [
    "EnumerationCompleteness",
    "UncertaintyRefusalError",
    "UncertaintyRendering",
    "UncertaintySource",
    "bootstrap_interval",
    "resolve_uncertainty",
]
