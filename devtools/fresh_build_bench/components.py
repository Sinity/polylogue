"""Component benchmarks over a sealed fresh-build corpus.

The end-to-end run proves a total; these isolate one production stage each so
an optimisation can be iterated in seconds. Every component calls the same
function the daemon calls, on the corpus's own files:

* ``blob`` -- ``BlobStore.write_from_path``, the acquisition copy + hash +
  fsync + publish each admitted file pays.

Results print as one JSON document per component with per-origin seconds,
bytes and derived MiB/s. The corpus is verified against its seal before and
after the timed work; a file that changed meanwhile fails the run, because the
summary attributes throughput to the sealed byte counts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from devtools.fresh_build_bench.corpus import load_manifest, refuse_inside_checkout, verify_manifest

_PROVIDER_BY_ORIGIN = {
    "claude-code": "claude-code",
    "codex": "codex",
    "gemini-cli": "gemini-cli",
    "chatgpt": "chatgpt",
    "claude-ai": "claude-ai",
}


def _corpus_files(
    corpus: Path,
    manifest: dict[str, Any],
    origins: Iterable[str] | None,
    limit: int | None,
) -> list[tuple[Path, str, int]]:
    verify_manifest(corpus, manifest)
    wanted = set(origins) if origins else None
    files = []
    for relative, size, _digest, origin in manifest["files"]:
        if origin not in _PROVIDER_BY_ORIGIN or (wanted is not None and origin not in wanted):
            continue
        path = corpus / relative
        # Acquisition stores every retained sidecar's bytes, so the blob
        # component keeps sidecars beside their transcripts.
        files.append((path, origin, size))
    files.sort(key=lambda item: str(item[0]))
    selected = files[:limit] if limit else files
    if not selected:
        # A timing over no files measured no production work.
        raise SystemExit(f"no corpus files match the requested selection (origins={sorted(wanted or ())})")
    return selected


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
    work: Callable[[tuple[Path, str, int]], Callable[[], dict[str, int]]],
    files: list[tuple[Path, str, int]],
    workers: int,
) -> tuple[list[tuple[str, int, float]], float, dict[str, int]]:
    """Time ``work`` per file; the callable it returns runs after the timer.

    That callable collects counts (reading prepared artifacts back, say),
    which is the caller's consumption, not the stage being measured.
    """
    counts: dict[str, int] = defaultdict(int)

    def run(item: tuple[Path, str, int]) -> tuple[str, int, float, float, Callable[[], dict[str, int]]]:
        began = time.perf_counter()
        finish = work(item)
        ended = time.perf_counter()
        return item[1], item[2], ended - began, ended, finish

    rows: list[tuple[str, int, float]] = []
    last_ended = began = time.perf_counter()

    def consume(result: tuple[str, int, float, float, Callable[[], dict[str, int]]]) -> None:
        # ``finish()`` (artifact rereads, session/message counting) is the
        # caller's own consumption, not the timed production stage. It runs
        # as each file completes, so a prepared artifact's scratch is released
        # while the rest are still being prepared, and the stage's wall is
        # read from the workers' own end times rather than from this loop.
        nonlocal last_ended
        origin, size, seconds, ended, finish = result
        rows.append((origin, size, seconds))
        last_ended = max(last_ended, ended)
        for key, value in finish().items():
            counts[key] += value

    if workers <= 1:
        for item in files:
            consume(run(item))
    else:
        # At most ``2 * workers`` files are submitted and unconsumed at once:
        # a preparation holds its attempt scratch until consumed, so faster
        # workers than one consumer must not queue the whole corpus.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="component") as pool:
            pending_items = iter(files)
            in_flight = {pool.submit(run, item) for item in itertools.islice(pending_items, 2 * workers)}
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result())
                    for item in itertools.islice(pending_items, 1):
                        in_flight.add(pool.submit(run, item))
    wall = last_ended - began
    return rows, wall, dict(counts)


def bench_blob(
    corpus: Path, scratch: Path, *, workers: int, origins: list[str] | None, limit: int | None
) -> dict[str, Any]:
    from polylogue.storage.blob_store import BlobStore

    manifest = load_manifest(corpus)
    files = _corpus_files(corpus, manifest, origins, limit)
    store = BlobStore(scratch / "blob")

    def work(item: tuple[Path, str, int]) -> Callable[[], dict[str, int]]:
        store.write_from_path(item[0])
        return dict

    rows, wall, counts = _timed_map(work, files, workers)
    verify_manifest(corpus, manifest)
    return _summarise(rows, wall, {"component": "blob", "workers": workers, "counts": counts})


def _at_least_one(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return count


def refuse_scratch_inside_corpus(scratch: Path, corpus: Path) -> None:
    """Output under the sealed input tree dirties it for good."""
    resolved, corpus_path = scratch.resolve(), corpus.resolve()
    if resolved == corpus_path or corpus_path in resolved.parents:
        raise SystemExit(f"--scratch must lie outside the corpus ({corpus_path})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("component", choices=("blob",))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True, help="empty directory for component output")
    parser.add_argument("--workers", type=_at_least_one, default=1)
    parser.add_argument("--origin", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    # Blob output is a byte copy of (possibly private) corpus files.
    # A sealed corpus is private transcript content; one under the checkout
    # can be staged by accident.
    refuse_inside_checkout(args.corpus, "--corpus")
    refuse_inside_checkout(args.scratch, "--scratch")
    refuse_scratch_inside_corpus(args.scratch, args.corpus)
    if args.scratch.exists() and any(args.scratch.iterdir()):
        # A reused blob store deduplicates and skips the publication work.
        raise SystemExit(f"--scratch must be absent or empty: {args.scratch}")
    args.scratch.mkdir(parents=True, exist_ok=True)
    result = bench_blob(args.corpus, args.scratch, workers=args.workers, origins=args.origin, limit=args.limit)
    print(json.dumps(result, indent=1))
    # A worker that failed shortened the measured work; the timing is not a
    # result for this corpus.
    return 1 if (result.get("counts") or {}).get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
