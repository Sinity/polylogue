"""Foreign-key-like relation detection helpers."""

from __future__ import annotations

from polylogue.schemas.field_stats import FieldStats
from polylogue.schemas.relational_inference_models import ForeignKeyRelation

_FK_MATCH_THRESHOLD = 0.6


def detect_foreign_keys(stats: dict[str, FieldStats]) -> list[ForeignKeyRelation]:
    """Detect fields whose values mostly match keys in some dict field."""
    results: list[ForeignKeyRelation] = []

    for path, field_stats in stats.items():
        ref_target = getattr(field_stats, "_ref_target", None)
        if ref_target:
            results.append(
                ForeignKeyRelation(
                    source_path=path,
                    target_path=ref_target,
                    match_ratio=1.0,
                    evidence={"source": "field_stats_ref_detection"},
                )
            )

    for path, field_stats in stats.items():
        if not field_stats.observed_values or len(field_stats.observed_values) <= 5:
            continue

        terminal = path.rsplit(".", 1)[-1].lower() if "." in path else path.lower()
        if terminal not in {
            "parent",
            "parentid",
            "parent_id",
            "parentuuid",
            "parent_uuid",
            "ref",
            "reference",
            "source_id",
        }:
            continue
        if getattr(field_stats, "_ref_target", None):
            continue

        observed = set(field_stats.observed_values.keys())
        for other_path, other_stats in stats.items():
            if other_path == path:
                continue
            other_terminal = other_path.rsplit(".", 1)[-1].lower() if "." in other_path else other_path.lower()
            if other_terminal not in {"id", "uuid", "key", "node_id"}:
                continue
            if not other_stats.observed_values:
                continue
            other_values = set(other_stats.observed_values.keys())
            if not other_values:
                continue

            overlap = len(observed & other_values)
            ratio = overlap / len(observed) if observed else 0
            if ratio >= _FK_MATCH_THRESHOLD:
                results.append(
                    ForeignKeyRelation(
                        source_path=path,
                        target_path=other_path,
                        match_ratio=ratio,
                        evidence={
                            "overlap_count": overlap,
                            "source_count": len(observed),
                            "target_count": len(other_values),
                        },
                    )
                )

    return results


__all__ = ["detect_foreign_keys"]
