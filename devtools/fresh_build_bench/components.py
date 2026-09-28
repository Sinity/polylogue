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
bytes and derived MiB/s. The corpus is verified against its seal before and
after the timed work; a file that changed meanwhile fails the run, because the
summary attributes throughput to the sealed byte counts.
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

from devtools.fresh_build_bench.corpus import load_manifest, refuse_inside_checkout, verify_manifest
from polylogue.core.enums import Provider
from polylogue.sources.origin_specs import recognize_source_class

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
    *,
    sessions_only: bool,
) -> list[tuple[Path, str, int]]:
    verify_manifest(corpus, manifest)
    wanted = set(origins) if origins else None
    files = []
    for relative, size, _digest, origin in manifest["files"]:
        if origin not in _PROVIDER_BY_ORIGIN or (wanted is not None and origin not in wanted):
            continue
        path = corpus / relative
        # The production prefetch stage hands live_parse_path_worker only
        # candidates recognize_source_class classifies as a session (the same
        # predicate census_source_root uses); a retained sidecar
        # (tool-results/*.txt, tool-outputs/**/*.txt) is a non_session
        # candidate the walk discovers separately (source_walk.py's
        # discover_sidecars), never parsed as its own transcript. Sending
        # one here either errors the worker or measures work production
        # never performs.
        # The blob component keeps them: acquisition stores every retained
        # sidecar's bytes, so dropping them would omit real blob work.
        if sessions_only:
            provider = Provider.from_string(origin)
            recognition = recognize_source_class(provider, path)
            if recognition is not None and recognition.source_class != "session":
                continue
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

    def run(item: tuple[Path, str, int]) -> tuple[str, int, float, Callable[[], dict[str, int]]]:
        began = time.perf_counter()
        finish = work(item)
        elapsed = time.perf_counter() - began
        return item[1], item[2], elapsed, finish

    began = time.perf_counter()
    if workers <= 1:
        results = [run(item) for item in files]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="component") as pool:
            results = list(pool.map(run, files))
    wall = time.perf_counter() - began
    # ``finish()`` (artifact rereads, session/message counting) is the
    # caller's own consumption, not the timed production stage; calling it
    # here, after ``wall`` is fixed, keeps it out of both the per-file
    # ``elapsed`` boundary (already true above) and this outer wall timer --
    # `pool.map` above would otherwise not return until every `finish()` had
    # also run, folding that consumption into `wall`/`mib_per_s_wall`.
    rows = []
    for origin, size, seconds, finish in results:
        rows.append((origin, size, seconds))
        for key, value in finish().items():
            counts[key] += value
    return rows, wall, dict(counts)


def bench_parse(
    corpus: Path, scratch: Path, *, workers: int, origins: list[str] | None, limit: int | None
) -> dict[str, Any]:
    from polylogue.sources.dispatch import is_stream_record_provider
    from polylogue.sources.live.parse_prefetch import live_parse_path_worker

    manifest = load_manifest(corpus)
    files = _corpus_files(corpus, manifest, origins, limit, sessions_only=True)
    shard_root = scratch / "parse-shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    def work(item: tuple[Path, str, int]) -> Callable[[], dict[str, int]]:
        path, origin, _size = item
        provider = _PROVIDER_BY_ORIGIN[origin]
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=shard_root))
        try:
            result = live_parse_path_worker(
                provider,
                str(path),
                path.stem,
                # The production predicate: a case-insensitive suffix check.
                is_stream=is_stream_record_provider(str(path), provider),
                shard_directory=str(shard_root),
                attempt_directory=str(attempt),
            )
        except BaseException:
            shutil.rmtree(attempt, ignore_errors=True)
            raise

        def count() -> dict[str, int]:
            try:
                if result.error is not None:
                    return {"errors": 1}
                sessions = messages = 0
                for session in result.iter_sessions():
                    sessions += 1
                    messages += len(session.messages)
                return {"sessions": sessions, "messages": messages}
            finally:
                shutil.rmtree(attempt, ignore_errors=True)

        return count

    rows, wall, counts = _timed_map(work, files, workers)
    verify_manifest(corpus, manifest)
    # Threads, not the daemon's process pool: this isolates one file's
    # preparation cost; pool start-up and IPC belong to the end-to-end run.
    return _summarise(rows, wall, {"component": "parse", "workers": workers, "executor": "threads", "counts": counts})


def bench_blob(
    corpus: Path, scratch: Path, *, workers: int, origins: list[str] | None, limit: int | None
) -> dict[str, Any]:
    from polylogue.storage.blob_store import BlobStore

    manifest = load_manifest(corpus)
    files = _corpus_files(corpus, manifest, origins, limit, sessions_only=False)
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
    parser.add_argument("component", choices=("parse", "blob"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True, help="empty directory for component output")
    parser.add_argument("--workers", type=_at_least_one, default=1)
    parser.add_argument("--origin", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    # Blob and parse output are byte copies of (possibly private) corpus files.
    # A sealed corpus is private transcript content; one under the checkout
    # can be staged by accident.
    refuse_inside_checkout(args.corpus, "--corpus")
    refuse_inside_checkout(args.scratch, "--scratch")
    refuse_scratch_inside_corpus(args.scratch, args.corpus)
    if args.scratch.exists() and any(args.scratch.iterdir()):
        # A reused blob store deduplicates and skips the publication work.
        raise SystemExit(f"--scratch must be absent or empty: {args.scratch}")
    args.scratch.mkdir(parents=True, exist_ok=True)
    bench = bench_parse if args.component == "parse" else bench_blob
    result = bench(args.corpus, args.scratch, workers=args.workers, origins=args.origin, limit=args.limit)
    print(json.dumps(result, indent=1))
    # A worker that failed shortened the measured work; the timing is not a
    # result for this corpus.
    return 1 if (result.get("counts") or {}).get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
