from __future__ import annotations

from collections.abc import Sequence

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.config import Config, Source

INGEST_STAGE_SEQUENCES: dict[str, tuple[str, ...]] = {
    "acquire": ("acquire",),
    "parse": ("parse",),
    "materialize": ("materialize",),
    "index": ("index",),
    "embed": ("embed",),
    "schema": ("schema",),
    "reprocess": ("parse", "materialize", "index"),
    "all": ("acquire", "parse", "materialize", "index"),
}
INGEST_LEAF_STAGES = frozenset({stage for sequence in INGEST_STAGE_SEQUENCES.values() for stage in sequence})
INGEST_STAGES = frozenset({"parse", "reprocess", "all"})
PARSE_STAGES = frozenset({"parse", "reprocess", "all"})
MATERIALIZE_STAGES = frozenset({"materialize", "reprocess", "all"})


def select_sources(config: Config, source_names: Sequence[str] | None) -> list[Source]:
    """Select sources from config, filtering by names if provided."""
    if not source_names:
        return list(config.sources)
    name_set = set(source_names)
    return [source for source in config.sources if source.name in name_set]


def expand_requested_stage(stage: str) -> tuple[str, ...]:
    """Expand a requested ingest stage/composite into the leaf execution sequence."""
    return INGEST_STAGE_SEQUENCES[stage]


def normalize_stage_sequence(
    *,
    stage: str,
    stage_sequence: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the leaf stage sequence for an ingest request."""
    if stage_sequence is None:
        return expand_requested_stage(stage)
    normalized = tuple(stage_sequence)
    invalid = [stage_name for stage_name in normalized if stage_name not in INGEST_LEAF_STAGES]
    if invalid:
        raise ValueError(f"Unknown leaf stage(s): {', '.join(invalid)}")
    duplicates: list[str] = []
    seen: set[str] = set()
    for stage_name in normalized:
        if stage_name in seen and stage_name not in duplicates:
            duplicates.append(stage_name)
        seen.add(stage_name)
    if duplicates:
        raise ValueError(f"Duplicate leaf stage(s): {', '.join(duplicates)}")
    return normalized


__all__ = [
    "INGEST_LEAF_STAGES",
    "INGEST_STAGE_SEQUENCES",
    "INGEST_STAGES",
    "MATERIALIZE_STAGES",
    "PARSE_STAGES",
    "expand_requested_stage",
    "normalize_stage_sequence",
    "run_coroutine_sync",
    "select_sources",
]
