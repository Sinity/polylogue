"""Foreign-key-like relation detection helpers."""

from __future__ import annotations

import hashlib

from polylogue.schemas.field_stats.models import REF_MATCH_THRESHOLD
from polylogue.schemas.field_stats.stats import FieldStats
from polylogue.schemas.inference.relational.models import ForeignKeyRelation

_FK_MATCH_THRESHOLD = 0.6


def _equality_values(field_stats: FieldStats) -> set[str]:
    """Return equality values in one private hash namespace across reloads."""
    if field_stats.equality_hash_counts:
        return set(field_stats.equality_hash_counts)
    return {
        hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
        for value in field_stats.observed_values
    }


def _append_mapping_references(stats: dict[str, FieldStats], results: list[ForeignKeyRelation]) -> None:
    """Detect references against bounded object-key witnesses after aggregation."""
    emitted: set[tuple[str, str]] = set()
    for source_path, source_stats in stats.items():
        if len(_equality_values(source_stats)) <= 5:
            continue
        for target_path, target_stats in stats.items():
            if source_path == target_path:
                continue
            overlap = source_stats.object_key_overlap(target_stats)
            if overlap is None:
                continue
            overlap_count, source_count, target_count = overlap
            ratio = overlap_count / source_count
            if ratio >= REF_MATCH_THRESHOLD:
                results.append(
                    ForeignKeyRelation(
                        source_path=source_path,
                        target_path=target_path,
                        match_ratio=ratio,
                        evidence={
                            "source": "object_key_hash_overlap",
                            "overlap_count": overlap_count,
                            "source_count": source_count,
                            "target_count": target_count,
                        },
                    )
                )
                emitted.add((source_path, target_path))

    for source_path, source_stats in stats.items():
        target_ref = source_stats.ref_target
        if target_ref is None or (source_path, target_ref) in emitted or target_ref in stats:
            continue
        results.append(
            ForeignKeyRelation(
                source_path=source_path,
                target_path=target_ref,
                match_ratio=1.0,
                evidence={"source": "field_stats_ref_detection"},
            )
        )


def detect_foreign_keys(stats: dict[str, FieldStats]) -> list[ForeignKeyRelation]:
    """Detect fields whose values mostly match keys in some dict field."""
    results: list[ForeignKeyRelation] = []

    _append_mapping_references(stats, results)

    for path, field_stats in stats.items():
        observed = _equality_values(field_stats)
        if len(observed) <= 5:
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
        for other_path, other_stats in stats.items():
            if other_path == path:
                continue
            other_terminal = other_path.rsplit(".", 1)[-1].lower() if "." in other_path else other_path.lower()
            if other_terminal not in {"id", "uuid", "key", "node_id"}:
                continue
            other_values = _equality_values(other_stats)
            if not other_values:
                continue

            overlap = len(observed & other_values)
            ratio = overlap / len(observed) if observed else 0
            if ratio >= _FK_MATCH_THRESHOLD:
                if any(relation.source_path == path and relation.target_path == other_path for relation in results):
                    continue
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
