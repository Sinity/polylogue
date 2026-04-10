"""Mutual-exclusion relation detection helpers."""

from __future__ import annotations

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.relational_inference_models import MutualExclusion


def detect_mutual_exclusions(stats: dict[str, FieldStats]) -> list[MutualExclusion]:
    """Detect groups of fields that never co-occur in the same object."""
    results: list[MutualExclusion] = []
    parent_children: dict[str, list[str]] = {}

    for path in stats:
        if "." not in path:
            continue
        parent, child = path.rsplit(".", 1)
        if "[*]" in child:
            continue
        parent_children.setdefault(parent, []).append(child)

    for parent, children in parent_children.items():
        if len(children) < 2:
            continue

        exclusive_pairs: list[tuple[str, str]] = []
        for index, name_a in enumerate(children):
            path_a = f"{parent}.{name_a}"
            stats_a = stats.get(path_a)
            if not stats_a or stats_a.present_count == 0:
                continue
            for name_b in children[index + 1 :]:
                path_b = f"{parent}.{name_b}"
                stats_b = stats.get(path_b)
                if not stats_b or stats_b.present_count == 0:
                    continue

                if stats_a.co_occurring_fields.get(name_b, 0) == 0:
                    exclusive_pairs.append((name_a, name_b))

        for name_a, name_b in exclusive_pairs:
            path_a = f"{parent}.{name_a}"
            path_b = f"{parent}.{name_b}"
            stats_a = stats[path_a]
            stats_b = stats[path_b]
            results.append(
                MutualExclusion(
                    parent_path=parent,
                    field_names=frozenset({name_a, name_b}),
                    sample_count=max(stats_a.total_samples, stats_b.total_samples),
                    evidence={
                        f"{name_a}_present": stats_a.present_count,
                        f"{name_b}_present": stats_b.present_count,
                    },
                )
            )

    return results


__all__ = ["detect_mutual_exclusions"]
