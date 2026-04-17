"""String-length relation detection helpers."""

from __future__ import annotations

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.relational_inference_models import StringLengthProfile


def detect_string_lengths(stats: dict[str, FieldStats]) -> list[StringLengthProfile]:
    """Extract interesting string length distributions."""
    results: list[StringLengthProfile] = []

    for path, field_stats in stats.items():
        length_stats = field_stats.string_length_stats
        if not length_stats:
            continue

        avg = length_stats["avg"]
        stddev = length_stats["stddev"]
        if avg < 5 and stddev < 2:
            continue
        if len(field_stats.string_lengths) < 3:
            continue

        results.append(
            StringLengthProfile(
                path=path,
                min_length=int(length_stats["min"]),
                max_length=int(length_stats["max"]),
                avg_length=avg,
                stddev=stddev,
                evidence={
                    "sample_count": len(field_stats.string_lengths),
                    "multiline_rate": round(field_stats.newline_rate, 3),
                },
            )
        )

    return results


__all__ = ["detect_string_lengths"]
