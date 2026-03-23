from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import TypeVar

from polylogue.config import Config, Source
from polylogue.lib.json import dumps

T = TypeVar("T")

RUN_STAGE_CHOICES: tuple[str, ...] = (
    "acquire",
    "validate",
    "parse",
    "render",
    "index",
    "generate-schemas",
    "all",
)
INGEST_STAGES = frozenset({"validate", "parse", "all"})
PARSE_STAGES = frozenset({"parse", "all"})
RENDER_STAGES = frozenset({"render", "all"})


def select_sources(config: Config, source_names: Sequence[str] | None) -> list[Source]:
    """Select sources from config, filtering by names if provided."""
    if not source_names:
        return list(config.sources)
    name_set = set(source_names)
    return [source for source in config.sources if source.name in name_set]


def run_coroutine_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code, even when already inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[T] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - re-raised on caller thread
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    if not result:
        raise RuntimeError("Coroutine thread completed without returning a result")
    return result[0]


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
    "PARSE_STAGES",
    "RENDER_STAGES",
    "RUN_STAGE_CHOICES",
    "run_coroutine_sync",
    "select_sources",
    "write_run_json",
]
