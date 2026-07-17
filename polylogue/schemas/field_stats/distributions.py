"""Bounded, deterministic distribution sketches for schema observations."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field

from polylogue.core.json import JSONDocument, JSONValue

_LOG_BASE = 1.1
_LOG_BASE_LOG = math.log(_LOG_BASE)
_MAX_LOG_BUCKET = 512
_DEFAULT_QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
_CATEGORICAL_BUCKETS = 256
_HLL_PRECISION = 8


def _bucket_index(value: float) -> int:
    if value == 0.0:
        return 0
    magnitude = min(_MAX_LOG_BUCKET, max(1, int(math.floor(math.log1p(abs(value)) / _LOG_BASE_LOG)) + 1))
    return magnitude if value > 0 else -magnitude


def _bucket_midpoint(index: int) -> float:
    if index == 0:
        return 0.0
    magnitude = math.expm1((abs(index) - 0.5) * _LOG_BASE_LOG)
    return magnitude if index > 0 else -magnitude


def _json_number(value: float | None) -> JSONValue:
    if value is None:
        return None
    if value.is_integer():
        return int(value)
    return value


@dataclass
class DistributionSketch:
    """A mergeable fixed-log-histogram plus exact streaming moments.

    The number of possible buckets is fixed by the finite float exponent range
    and the explicit clamp above, so memory is independent of observation
    count.  Min/max and moments remain exact for the finite values accepted;
    quantiles are reconstructed from the histogram.
    """

    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    m2: float = 0.0
    buckets: Counter[int] = field(default_factory=Counter)
    non_finite_count: int = 0

    def observe(self, value: int | float) -> None:
        numeric = float(value)
        if not math.isfinite(numeric):
            self.non_finite_count += 1
            return
        self.count += 1
        self.minimum = numeric if self.minimum is None else min(self.minimum, numeric)
        self.maximum = numeric if self.maximum is None else max(self.maximum, numeric)
        delta = numeric - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (numeric - self.mean)
        self.buckets[_bucket_index(numeric)] += 1

    def observe_repeated(self, value: int | float, count: int) -> None:
        """Observe one finite value ``count`` times without a count-sized loop."""
        if count <= 0:
            return
        numeric = float(value)
        if not math.isfinite(numeric):
            self.non_finite_count += count
            return
        repeated = DistributionSketch(
            count=count,
            minimum=numeric,
            maximum=numeric,
            mean=numeric,
            buckets=Counter({_bucket_index(numeric): count}),
        )
        self.merge(repeated)

    def merge(self, other: DistributionSketch) -> None:
        if other.count:
            if not self.count:
                self.count = other.count
                self.minimum = other.minimum
                self.maximum = other.maximum
                self.mean = other.mean
                self.m2 = other.m2
            else:
                combined = self.count + other.count
                delta = other.mean - self.mean
                self.m2 += other.m2 + delta * delta * self.count * other.count / combined
                self.mean += delta * other.count / combined
                self.count = combined
                if other.minimum is not None:
                    self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
                if other.maximum is not None:
                    self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)
            self.buckets.update(other.buckets)
        self.non_finite_count += other.non_finite_count

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def quantile(self, probability: float) -> float | None:
        if not self.count:
            return None
        if probability <= 0:
            return self.minimum
        if probability >= 1:
            return self.maximum
        target = max(1, math.ceil(probability * self.count))
        seen = 0
        for index, bucket_count in sorted(self.buckets.items()):
            seen += bucket_count
            if seen >= target:
                midpoint = _bucket_midpoint(index)
                if self.minimum is not None:
                    midpoint = max(self.minimum, midpoint)
                if self.maximum is not None:
                    midpoint = min(self.maximum, midpoint)
                return midpoint
        return self.maximum

    def to_payload(self, *, quantiles: tuple[float, ...] = _DEFAULT_QUANTILES) -> JSONDocument:
        quantile_payload: JSONDocument = {}
        for probability in quantiles:
            value = self.quantile(probability)
            label = f"p{probability * 100:g}".replace(".", "_")
            quantile_payload[label] = _json_number(value)
        histogram: list[JSONValue] = [[index, count] for index, count in sorted(self.buckets.items())]
        return {
            "count": self.count,
            "min": _json_number(self.minimum),
            "max": _json_number(self.maximum),
            "mean": self.mean if self.count else None,
            "stddev": self.stddev if self.count else None,
            "quantiles": quantile_payload,
            "log_base": _LOG_BASE,
            "histogram": histogram,
            "non_finite_count": self.non_finite_count,
        }


@dataclass
class CategoricalSketch:
    """Fixed-memory, mergeable evidence for arbitrary categorical values.

    Values are never retained. Every observation contributes to a stable hash
    bucket, while HyperLogLog registers preserve approximate distinctness. The
    readable heavy hitters remain a separate privacy-controlled concern.
    """

    count: int = 0
    buckets: Counter[int] = field(default_factory=Counter)
    registers: dict[int, int] = field(default_factory=dict)

    def observe(self, value: str, *, count: int = 1) -> None:
        if count <= 0:
            return
        digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).digest()
        hashed = int.from_bytes(digest[:8], "big")
        bucket = hashed & (_CATEGORICAL_BUCKETS - 1)
        self.count += count
        self.buckets[bucket] += count

        remaining = hashed >> _HLL_PRECISION
        width = 64 - _HLL_PRECISION
        rank = width - remaining.bit_length() + 1 if remaining else width + 1
        self.registers[bucket] = max(self.registers.get(bucket, 0), rank)

    def merge(self, other: CategoricalSketch) -> None:
        self.count += other.count
        self.buckets.update(other.buckets)
        for bucket, rank in other.registers.items():
            self.registers[bucket] = max(self.registers.get(bucket, 0), rank)

    @property
    def estimated_distinct(self) -> int:
        register_count = 1 << _HLL_PRECISION
        alpha = 0.7213 / (1.0 + 1.079 / register_count)
        zero_count = register_count - len(self.registers)
        harmonic_sum = zero_count + sum(2.0**-register for register in self.registers.values())
        raw = alpha * register_count * register_count / harmonic_sum
        if raw <= 2.5 * register_count and zero_count:
            raw = register_count * math.log(register_count / zero_count)
        return max(0, round(raw))

    def to_payload(self) -> JSONDocument:
        bucket_loads = DistributionSketch()
        for count in self.buckets.values():
            bucket_loads.observe(count)
        return {
            "count": self.count,
            "estimated_distinct": self.estimated_distinct,
            "hash_algorithm": "sha256-64",
            "bucket_count": _CATEGORICAL_BUCKETS,
            "bucket_histogram": [[bucket, count] for bucket, count in sorted(self.buckets.items())],
            "bucket_load_distribution": bucket_loads.to_payload(),
            "values_retained": False,
        }


__all__ = ["CategoricalSketch", "DistributionSketch"]
