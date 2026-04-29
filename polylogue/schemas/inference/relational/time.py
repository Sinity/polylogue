"""Temporal relation detection helpers."""

from __future__ import annotations

from polylogue.schemas.field_stats.stats import FieldStats
from polylogue.schemas.inference.relational.models import TimeDeltaRelation


def detect_time_deltas(stats: dict[str, FieldStats]) -> list[TimeDeltaRelation]:
    """Detect temporal offsets between timestamp-like fields in the same object."""
    results: list[TimeDeltaRelation] = []
    timestamp_fields: list[tuple[str, FieldStats]] = []

    for path, field_stats in stats.items():
        fmt = field_stats.dominant_format
        if fmt in {"unix-epoch", "unix-epoch-str", "iso8601"} or (
            field_stats.num_min is not None
            and field_stats.num_max is not None
            and field_stats.num_min >= 946684800.0
            and field_stats.num_max <= 2208988800.0
        ):
            timestamp_fields.append((path, field_stats))

    for index, (path_a, stats_a) in enumerate(timestamp_fields):
        for path_b, stats_b in timestamp_fields[index + 1 :]:
            parent_a = path_a.rsplit(".", 1)[0] if "." in path_a else "$"
            parent_b = path_b.rsplit(".", 1)[0] if "." in path_b else "$"
            if parent_a != parent_b:
                continue
            if stats_a.num_min is None or stats_a.num_max is None:
                continue
            if stats_b.num_min is None or stats_b.num_max is None:
                continue

            min_delta = abs(stats_b.num_min - stats_a.num_max)
            max_delta = abs(stats_b.num_max - stats_a.num_min)
            avg_delta = (min_delta + max_delta) / 2

            if max_delta > 0:
                results.append(
                    TimeDeltaRelation(
                        field_a=path_a,
                        field_b=path_b,
                        min_delta=min_delta,
                        max_delta=max_delta,
                        avg_delta=avg_delta,
                        evidence={
                            "field_a_range": [stats_a.num_min, stats_a.num_max],
                            "field_b_range": [stats_b.num_min, stats_b.num_max],
                        },
                    )
                )

    return results


__all__ = ["detect_time_deltas"]
