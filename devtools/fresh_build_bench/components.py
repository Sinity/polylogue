"""Component benchmarks over a sealed fresh-build corpus.

The end-to-end run proves a total; these isolate one production stage each so
an optimisation can be iterated in seconds. Every component calls the same
function the daemon calls, on the corpus's own files:

* ``parse`` -- ``live_parse_path_worker``, the off-writer preparation a parse
  worker runs per file (detection, streaming parse, prepared-carrier spill,
  session shard). ``--workers`` runs it on a thread pool the way the daemon's
  prefetch stage does, which on the free-threaded build measures scaling.
* ``blob`` -- ``BlobStore.write_from_path``, the acquisition copy + hash +
  fsync + publish each admitted file pays.

Results print as one JSON document per component with per-origin seconds,
bytes and derived MiB/s.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from devtools.fresh_build_bench.corpus import load_manifest

_PROVIDER_BY_ORIGIN = {"claude-code": "claude-code", "codex": "codex", "gemini-cli": "gemini-cli"}


def _corpus_files(corpus: Path, origins: Iterable[str] | None, limit: int | None) -> list[tuple[Path, str, int]]:
    manifest = load_manifest(corpus)
    wanted = set(origins) if origins else None
    files = [
        (corpus / relative, origin, size)
        for relative, size, _digest, origin in manifest["files"]
        if origin in _PROVIDER_BY_ORIGIN and (wanted is None or origin in wanted)
    ]
    files.sort(key=lambda item: str(item[0]))
    return files[:limit] if limit else files


def _summarise(rows: list[tuple[str, int, float]], wall: float, extra: dict[str, Any]) -> dict[str, Any]:
    by_origin: dict[str, dict[str, float]] = defaultdict(lambda: {"files": 0, "bytes": 0, "seconds": 0.0})
    for origin, size, seconds in rows:
        entry = by_origin[origin]
        entry["files"] += 1
        entry["bytes"] += size
        entry["seconds"] += seconds
    total_bytes = sum(size for _origin, size, _seconds in rows)
    return {
        **extra,
        "wall_s": round(wall, 3),
        "files": len(rows),
        "bytes": total_bytes,
        "mib_per_s_wall": round(total_bytes / 2**20 / wall, 3) if wall else None,
        "by_origin": {
            origin: {
                "files": int(entry["files"]),
                "mib": round(entry["bytes"] / 2**20, 2),
                "seconds": round(entry["seconds"], 3),
                "mib_per_s": round(entry["bytes"] / 2**20 / entry["seconds"], 3) if entry["seconds"] else None,
            }
            for origin, entry in sorted(by_origin.items())
        },
    }


def _timed_map(
    work: Callable[[tuple[Path, str, int]], dict[str, int]], files: list[tuple[Path, str, int]], workers: int
) -> tuple[list[tuple[str, int, float]], float, dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)

    def run(item: tuple[Path, str, int]) -> tuple[str, int, float, dict[str, int]]:
        began = time.perf_counter()
        produced = work(item)
        return item[1], item[2], time.perf_counter() - began, produced

    began = time.perf_counter()
    if workers <= 1:
        results = [run(item) for item in files]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="component") as pool:
            results = list(pool.map(run, files))
    wall = time.perf_counter() - began
    rows = []
    for origin, size, seconds, produced in results:
        rows.append((origin, size, seconds))
        for key, value in produced.items():
            counts[key] += value
    return rows, wall, dict(counts)


def bench_parse(
    corpus: Path, scratch: Path, *, workers: int, origins: list[str] | None, limit: int | None
) -> dict[str, Any]:
    from polylogue.sources.live.parse_prefetch import live_parse_path_worker

    files = _corpus_files(corpus, origins, limit)
    shard_root = scratch / "parse-shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    def work(item: tuple[Path, str, int]) -> dict[str, int]:
        path, origin, _size = item
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=shard_root))
        try:
            result = live_parse_path_worker(
                _PROVIDER_BY_ORIGIN[origin],
                str(path),
                path.stem,
                is_stream=path.suffix == ".jsonl",
                shard_directory=str(shard_root),
                attempt_directory=str(attempt),
            )
            if result.error is not None:
                return {"errors": 1}
            sessions = messages = 0
            for session in result.iter_sessions():
                sessions += 1
                messages += len(session.messages)
            return {"sessions": sessions, "messages": messages}
        finally:
            shutil.rmtree(attempt, ignore_errors=True)

    rows, wall, counts = _timed_map(work, files, workers)
    return _summarise(rows, wall, {"component": "parse", "workers": workers, "counts": counts})


def bench_blob(
    corpus: Path, scratch: Path, *, workers: int, origins: list[str] | None, limit: int | None
) -> dict[str, Any]:
    from polylogue.storage.blob_store import BlobStore

    files = _corpus_files(corpus, origins, limit)
    store = BlobStore(scratch / "blob")

    def work(item: tuple[Path, str, int]) -> dict[str, int]:
        store.write_from_path(item[0])
        return {}

    rows, wall, counts = _timed_map(work, files, workers)
    return _summarise(rows, wall, {"component": "blob", "workers": workers, "counts": counts})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("component", choices=("parse", "blob"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True, help="empty directory for component output")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--origin", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=True)
    bench = bench_parse if args.component == "parse" else bench_blob
    result = bench(args.corpus, args.scratch, workers=args.workers, origins=args.origin, limit=args.limit)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
