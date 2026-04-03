from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from polylogue.config import Config, Source
from polylogue.lib.json import dumps
from polylogue.sync_bridge import run_coroutine_sync

T = TypeVar("T")

RUN_STAGE_CHOICES: tuple[str, ...] = (
    "acquire",
    "parse",
    "materialize",
    "render",
    "index",
    "generate-schemas",
    "reprocess",
    "all",
)
RUN_STAGE_SEQUENCES: dict[str, tuple[str, ...]] = {
    "acquire": ("acquire",),
    "parse": ("parse", "index"),
    "materialize": ("materialize",),
    "render": ("render",),
    "index": ("index",),
    "generate-schemas": ("generate-schemas",),
    "reprocess": ("parse", "materialize", "render", "index"),
    "all": ("acquire", "parse", "materialize", "render", "index"),
}
RUN_LEAF_STAGES = frozenset({stage for sequence in RUN_STAGE_SEQUENCES.values() for stage in sequence})
INGEST_STAGES = frozenset({"parse", "reprocess", "all"})
PARSE_STAGES = frozenset({"parse", "reprocess", "all"})
MATERIALIZE_STAGES = frozenset({"materialize", "reprocess", "all"})
RENDER_STAGES = frozenset({"render", "reprocess", "all"})


def select_sources(config: Config, source_names: Sequence[str] | None) -> list[Source]:
    """Select sources from config, filtering by names if provided."""
    if not source_names:
        return list(config.sources)
    name_set = set(source_names)
    return [source for source in config.sources if source.name in name_set]


def expand_requested_stage(stage: str) -> tuple[str, ...]:
    """Expand a requested stage/composite into the leaf execution sequence."""
    return RUN_STAGE_SEQUENCES[stage]


def normalize_stage_sequence(
    *,
    stage: str,
    stage_sequence: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the leaf stage sequence for a run request."""
    if stage_sequence is None:
        return expand_requested_stage(stage)
    normalized = tuple(stage_sequence)
    invalid = [stage_name for stage_name in normalized if stage_name not in RUN_LEAF_STAGES]
    if invalid:
        raise ValueError(f"Unknown leaf stage(s): {', '.join(invalid)}")
    return normalized

def write_run_json(archive_root: Path, payload: dict[str, object]) -> Path:
    """Write run result JSON to the runs directory."""
    runs_dir = archive_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = payload.get("run_id", "unknown")
    run_path = runs_dir / f"run-{payload['timestamp']}-{run_id}.json"
    run_path.write_text(dumps(payload, option=None), encoding="utf-8")
    return run_path


__all__ = [
    "INGEST_STAGES",
    "MATERIALIZE_STAGES",
    "PARSE_STAGES",
    "RENDER_STAGES",
    "RUN_STAGE_CHOICES",
    "RUN_LEAF_STAGES",
    "RUN_STAGE_SEQUENCES",
    "expand_requested_stage",
    "normalize_stage_sequence",
    "run_coroutine_sync",
    "select_sources",
    "write_run_json",
]
