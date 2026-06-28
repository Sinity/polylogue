"""Comprehensive ingest instrument: time, CPU, memory, I/O, SQLite, stages.

The sibling ``devtools bench ingest-amplification`` probe measures *bytes
written* per archive tier — a deterministic, host-independent quantity.  It does
not measure *time* or runtime resource cost.  This probe is the missing runtime
counterpart: it drives a real archive write path (the full ``parse_sources_archive``
pipeline for the corpus workload, or a single open ``ArchiveStore`` for the
lineage workload — both using the WAL write profile) over a deterministic
synthetic workload and captures, per run:

* **wall-clock** — total + per-batch-ms distribution (min/max/mean/p90),
* **per-stage attribution** — the pipeline's ``ParseResult.stage_timings_s``
  accumulated across batches, plus the dominant ``top_stages``,
* **CPU** — rusage user/sys seconds and ``cpu_utilization`` (cpu_total /
  wall) so a run immediately shows whether ingest is CPU-bound (~1.0),
  I/O-wait-bound (<1.0), or parallel (>1.0),
* **memory** — ``peak_rss_mb`` (process max), page faults, optional
  ``tracemalloc_peak_mb`` allocation profiling under ``--memory``,
* **storage I/O** — ``/proc/self/io`` byte/syscall deltas plus block-I/O
  rusage counters, and SQLite file growth (``index.db`` / ``-wal`` /
  ``source.db``) with ``bytes_written_per_message`` and
  ``db_growth_per_message`` efficiency ratios.

It is additive tooling.  It does **not** touch production ingest logic — it
drives the existing public write paths over a synthetic, deterministic workload
built under a temporary directory.

Operators run::

    devtools bench ingest-throughput --json > run.json
    devtools bench ingest-throughput --batches 20 --seed 2391
    devtools bench ingest-throughput --lineage          # fork-heavy workload
    devtools bench ingest-throughput --memory           # + tracemalloc (distorts timing)

Wall-clock and resource numbers are host-variable: this probe has **no CI
thresholds**.  Message/session *counts* are deterministic for a fixed
(provider/workload, batches, seed); the timings, CPU, memory, and I/O are not.

REPORT_VERSION 2 added: ``workload``, ``stage_timings_s``, ``top_stages``,
``cpu_seconds_total``, ``cpu_utilization``, ``peak_rss_mb``, ``resources``,
``proc_io``, ``proc_io_available``, ``storage``, and (under ``--memory``)
``tracemalloc_peak_mb``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

# Bumped when the JSON shape gains/changes a top-level key or field type.
REPORT_VERSION = 2

_DEFAULT_PROVIDER = "codex"
_DEFAULT_BATCHES = 16
_DEFAULT_SEED = 2391
_DEFAULT_MESSAGES_MIN = 120
_DEFAULT_MESSAGES_MAX = 320

# Short divergent tail appended after the replayed parent prefix in the
# lineage workload.  Kept small so each fork batch is dominated by the
# inherited-prefix scan + signature memoization, not by fresh tail content.
_LINEAGE_TAIL_MESSAGES = 4


# ---------------------------------------------------------------------------
# Resource / storage snapshots
# ---------------------------------------------------------------------------


def _read_proc_io() -> dict[str, int] | None:
    """Parse ``/proc/self/io`` into an int-valued dict, or ``None`` if absent."""
    try:
        text = Path("/proc/self/io").read_text(encoding="utf-8")
    except OSError:
        return None
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        with suppress(ValueError):
            parsed[key.strip()] = int(value.strip())
    return parsed


def _rusage_snapshot() -> Any:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile over a non-empty list of values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


# ---------------------------------------------------------------------------
# Workload builders / runners
# ---------------------------------------------------------------------------


def _build_fixture_files(
    workdir: Path,
    *,
    provider: str,
    batches: int,
    seed: int,
    messages_min: int,
    messages_max: int,
) -> list[Path]:
    """Write ``batches`` single-session synthetic source files deterministically."""
    from polylogue.scenarios import build_default_corpus_specs
    from polylogue.schemas.synthetic import SyntheticCorpus

    available = set(SyntheticCorpus.available_providers())
    if provider not in available:
        raise ValueError(f"provider {provider!r} not available; choose from {sorted(available)}")

    (spec,) = build_default_corpus_specs(
        providers=(provider,),
        count=batches,
        messages_min=messages_min,
        messages_max=messages_max,
        seed=seed,
    )
    corpus_dir = workdir / "corpus" / provider
    written = SyntheticCorpus.write_spec_artifacts(spec, corpus_dir, prefix="batch", index_width=3)
    # One artifact == one session == one append batch.
    return list(written.files)


async def _ingest_batch(
    archive_root: Path, provider: str, source_file: Path
) -> tuple[dict[str, int], dict[str, float]]:
    from polylogue.config import Source
    from polylogue.pipeline.services.archive_ingest import parse_sources_archive

    result = await parse_sources_archive(archive_root, [Source(name=provider, path=source_file)])
    counts = {
        "sessions": int(result.counts.get("sessions", 0)),
        "messages": int(result.counts.get("messages", 0)),
    }
    return counts, dict(result.stage_timings_s)


def _accumulate(target: dict[str, float], delta: dict[str, float]) -> None:
    for key, value in delta.items():
        target[key] = target.get(key, 0.0) + float(value)


def _run_corpus_workload(
    archive_root: Path,
    *,
    provider: str,
    source_files: list[Path],
    batches: int,
) -> tuple[list[dict[str, Any]], list[float], int, int, dict[str, float]]:
    batch_reports: list[dict[str, Any]] = []
    per_batch_ms: list[float] = []
    total_sessions = 0
    total_messages = 0
    stage_timings_s: dict[str, float] = {}

    for index in range(batches):
        source_file = source_files[index]
        batch_start = time.perf_counter()
        ingested, stage_delta = asyncio.run(_ingest_batch(archive_root, provider, source_file))
        elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
        per_batch_ms.append(elapsed_ms)
        total_sessions += ingested["sessions"]
        total_messages += ingested["messages"]
        _accumulate(stage_timings_s, stage_delta)
        batch_reports.append(
            {
                "batch_index": index,
                "sessions_ingested": ingested["sessions"],
                "messages_ingested": ingested["messages"],
                "batch_ms": round(elapsed_ms, 3),
            }
        )
    return batch_reports, per_batch_ms, total_sessions, total_messages, stage_timings_s


def _lineage_message(*, provider_message_id: str, position: int, text: str) -> Any:
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage

    role = Role.USER if position % 2 == 0 else Role.ASSISTANT
    return ParsedMessage(
        provider_message_id=provider_message_id,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _build_lineage_sessions(*, batches: int, prefix_len: int) -> tuple[Any, list[Any]]:
    """Build one parent + ``batches`` prefix-sharing fork children.

    Each fork replays the parent's full prefix verbatim (identical content,
    fresh provider ids — exactly what a real fork/resume does) and then adds a
    short divergent tail.  This exercises lineage normalization (store tail
    only) and the #2488 signature memoization across many forks of one parent.
    """
    from polylogue.archive.session.branch_type import BranchType
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedSession

    parent_id = "lineage-parent"
    # Deterministic prefix text shared by parent and every fork.
    prefix_texts = [f"shared prefix message {i} for lineage parent {parent_id}" for i in range(prefix_len)]

    parent = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=parent_id,
        title="lineage parent",
        messages=[
            _lineage_message(provider_message_id=f"p{i}", position=i, text=prefix_texts[i]) for i in range(prefix_len)
        ],
    )

    forks: list[Any] = []
    for b in range(batches):
        messages = [
            _lineage_message(provider_message_id=f"f{b}-{i}", position=i, text=prefix_texts[i])
            for i in range(prefix_len)
        ]
        for j in range(_LINEAGE_TAIL_MESSAGES):
            position = prefix_len + j
            messages.append(
                _lineage_message(
                    provider_message_id=f"f{b}-t{j}",
                    position=position,
                    text=f"fork {b} divergent tail message {j}",
                )
            )
        forks.append(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=f"lineage-fork-{b}",
                title=f"lineage fork {b}",
                parent_session_provider_id=parent_id,
                branch_type=BranchType.FORK,
                messages=messages,
            )
        )
    return parent, forks


def _run_lineage_workload(
    archive_root: Path,
    *,
    batches: int,
    prefix_len: int,
) -> tuple[list[dict[str, Any]], list[float], int, int, dict[str, float]]:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    parent, forks = _build_lineage_sessions(batches=batches, prefix_len=prefix_len)

    batch_reports: list[dict[str, Any]] = []
    per_batch_ms: list[float] = []
    stage_timings_s: dict[str, float] = {}
    total_sessions = 0
    total_messages = 0
    acquired_at_ms = 1_700_000_000_000

    store = ArchiveStore.open_existing(archive_root, read_only=False)
    try:
        # Parent write is the lineage setup, not a timed fork batch, but its
        # messages count toward the composed total.
        parent_payload = json.dumps({"provider_session_id": parent.provider_session_id}).encode("utf-8")
        store.write_raw_and_parsed(
            parent,
            payload=parent_payload,
            source_path="synthetic://lineage/parent",
            acquired_at_ms=acquired_at_ms,
            stage_timings_s=stage_timings_s,
        )
        total_sessions += 1
        total_messages += len(parent.messages)

        for index, fork in enumerate(forks):
            payload = json.dumps({"provider_session_id": fork.provider_session_id}).encode("utf-8")
            batch_start = time.perf_counter()
            store.write_raw_and_parsed(
                fork,
                payload=payload,
                source_path=f"synthetic://lineage/fork-{index}",
                acquired_at_ms=acquired_at_ms + index + 1,
                stage_timings_s=stage_timings_s,
            )
            elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
            per_batch_ms.append(elapsed_ms)
            total_sessions += 1
            total_messages += len(fork.messages)
            batch_reports.append(
                {
                    "batch_index": index,
                    "sessions_ingested": 1,
                    "messages_ingested": len(fork.messages),
                    "batch_ms": round(elapsed_ms, 3),
                }
            )
    finally:
        store.close()
    return batch_reports, per_batch_ms, total_sessions, total_messages, stage_timings_s


# ---------------------------------------------------------------------------
# Top-level measurement
# ---------------------------------------------------------------------------


def measure_ingest_throughput(
    *,
    provider: str = _DEFAULT_PROVIDER,
    batches: int = _DEFAULT_BATCHES,
    seed: int = _DEFAULT_SEED,
    messages_min: int = _DEFAULT_MESSAGES_MIN,
    messages_max: int = _DEFAULT_MESSAGES_MAX,
    lineage: bool = False,
    memory: bool = False,
    workdir: Path | None = None,
) -> dict[str, Any]:
    """Run ``batches`` ingest batches and capture time/CPU/memory/I/O/stage cost.

    Returns a stable, JSON-serializable report.  When ``workdir`` is omitted a
    private temporary directory is created and torn down automatically; the
    archive and its blob store stay entirely inside that directory.

    ``lineage=True`` replaces the single-session corpus with a fork-heavy
    workload: one parent session and ``batches`` forks that each replay the
    parent's full prefix then diverge.  ``memory=True`` adds ``tracemalloc``
    allocation profiling — its overhead distorts timings, so a ``--memory`` run
    is not time-comparable to a normal run.

    Message/session counts are deterministic for a fixed (workload, batches,
    seed); wall-clock, CPU, memory, and I/O are host-variable and carry no
    thresholds.
    """
    if batches < 1:
        raise ValueError("batches must be >= 1")

    owns_workdir = workdir is None
    base = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="plg-ingest-tput-"))
    base.mkdir(parents=True, exist_ok=True)

    # Isolate every global path (archive root + blob store) inside the workdir so
    # the probe never reads or writes real user data.
    archive_root = base / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    saved_env = {key: os.environ.get(key) for key in ("XDG_DATA_HOME", "POLYLOGUE_ARCHIVE_ROOT")}
    os.environ["XDG_DATA_HOME"] = str(base / "xdg-data")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)

    try:
        from polylogue.storage.blob_store import reset_blob_store
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        reset_blob_store()

        # Bootstrap the five-tier archive file set up front so per-batch timings
        # exclude the one-time schema-DDL cost.
        ArchiveStore.open_existing(archive_root, read_only=False).close()

        source_files: list[Path] = []
        prefix_len = messages_max
        if not lineage:
            source_files = _build_fixture_files(
                base,
                provider=provider,
                batches=batches,
                seed=seed,
                messages_min=messages_min,
                messages_max=messages_max,
            )
            if len(source_files) < batches:
                batches = len(source_files)

        index_db = archive_root / "index.db"
        index_wal = archive_root / "index.db-wal"
        source_db = archive_root / "source.db"

        index_db_before = _file_size(index_db)

        if memory:
            import tracemalloc

            tracemalloc.start()

        rusage_before = _rusage_snapshot()
        proc_io_before = _read_proc_io()

        wall_start = time.perf_counter()
        if lineage:
            (
                batch_reports,
                per_batch_ms,
                total_sessions,
                total_messages,
                stage_timings_s,
            ) = _run_lineage_workload(archive_root, batches=batches, prefix_len=prefix_len)
        else:
            (
                batch_reports,
                per_batch_ms,
                total_sessions,
                total_messages,
                stage_timings_s,
            ) = _run_corpus_workload(archive_root, provider=provider, source_files=source_files, batches=batches)
        total_wall_s = time.perf_counter() - wall_start

        rusage_after = _rusage_snapshot()
        proc_io_after = _read_proc_io()

        tracemalloc_peak_mb: float | None = None
        if memory:
            import tracemalloc

            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            tracemalloc_peak_mb = round(peak_bytes / (1024.0 * 1024.0), 3)

        # ---- per-batch ms distribution ----
        per_batch_block = {
            "min": round(min(per_batch_ms), 3) if per_batch_ms else 0.0,
            "max": round(max(per_batch_ms), 3) if per_batch_ms else 0.0,
            "mean": round(sum(per_batch_ms) / len(per_batch_ms), 3) if per_batch_ms else 0.0,
            "p90": round(_percentile(per_batch_ms, 90.0), 3) if per_batch_ms else 0.0,
        }

        # ---- per-stage attribution ----
        sorted_stages = sorted(stage_timings_s.items(), key=lambda kv: kv[1], reverse=True)
        stage_timings_sorted = {key: round(value, 4) for key, value in sorted_stages}
        total_stage_time = sum(stage_timings_s.values())
        top_stages = [
            {
                "stage": key,
                "seconds": round(value, 4),
                "pct_of_total_stage_time": (
                    round(100.0 * value / total_stage_time, 2) if total_stage_time > 0 else 0.0
                ),
            }
            for key, value in sorted_stages[:8]
        ]

        # ---- CPU ----
        ru_utime_delta = rusage_after.ru_utime - rusage_before.ru_utime
        ru_stime_delta = rusage_after.ru_stime - rusage_before.ru_stime
        cpu_seconds_total = ru_utime_delta + ru_stime_delta
        cpu_utilization = round(cpu_seconds_total / total_wall_s, 3) if total_wall_s > 0 else 0.0

        # ---- memory: ru_maxrss is process peak (KiB on Linux), not a delta ----
        peak_rss_mb = round(rusage_after.ru_maxrss / 1024.0, 2)

        resources = {
            "ru_utime_s": round(ru_utime_delta, 4),
            "ru_stime_s": round(ru_stime_delta, 4),
            "ru_minflt_delta": int(rusage_after.ru_minflt - rusage_before.ru_minflt),
            "ru_majflt_delta": int(rusage_after.ru_majflt - rusage_before.ru_majflt),
            "ru_inblock_delta": int(rusage_after.ru_inblock - rusage_before.ru_inblock),
            "ru_oublock_delta": int(rusage_after.ru_oublock - rusage_before.ru_oublock),
        }

        # ---- /proc/self/io deltas ----
        proc_io_available = proc_io_before is not None and proc_io_after is not None
        proc_io: dict[str, Any] = {}
        write_bytes_delta = 0
        if proc_io_available:
            assert proc_io_before is not None and proc_io_after is not None
            for field in ("rchar", "wchar", "read_bytes", "write_bytes", "syscr", "syscw"):
                proc_io[field] = int(proc_io_after.get(field, 0) - proc_io_before.get(field, 0))
            write_bytes_delta = proc_io["write_bytes"]
            proc_io["read_mb"] = round(proc_io["read_bytes"] / (1024.0 * 1024.0), 3)
            proc_io["write_mb"] = round(proc_io["write_bytes"] / (1024.0 * 1024.0), 3)

        # ---- SQLite storage growth ----
        index_db_after = _file_size(index_db)
        index_db_growth = index_db_after - index_db_before
        storage = {
            "index_db_bytes": index_db_after,
            "index_wal_peak_bytes": _file_size(index_wal),
            "source_db_bytes": _file_size(source_db),
            "index_db_growth_bytes": index_db_growth,
            "bytes_written_per_message": (round(write_bytes_delta / total_messages, 2) if total_messages > 0 else 0.0),
            "db_growth_per_message": (round(index_db_growth / total_messages, 2) if total_messages > 0 else 0.0),
        }

        report: dict[str, Any] = {
            "ok": True,
            "report_version": REPORT_VERSION,
            "tool": "bench ingest-throughput",
            "workload": "lineage" if lineage else "corpus",
            "provider": provider,
            "batches": len(batch_reports),
            "seed": seed,
            "messages_min": messages_min,
            "messages_max": messages_max,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_wall_s": round(total_wall_s, 4),
            "messages_per_s": round(total_messages / total_wall_s, 2) if total_wall_s > 0 else 0.0,
            "sessions_per_s": round(total_sessions / total_wall_s, 2) if total_wall_s > 0 else 0.0,
            "cpu_seconds_total": round(cpu_seconds_total, 4),
            "cpu_utilization": cpu_utilization,
            "peak_rss_mb": peak_rss_mb,
            "per_batch_ms": per_batch_block,
            "stage_timings_s": stage_timings_sorted,
            "top_stages": top_stages,
            "resources": resources,
            "proc_io": proc_io,
            "proc_io_available": proc_io_available,
            "storage": storage,
            "per_batch": batch_reports,
        }
        if tracemalloc_peak_mb is not None:
            report["tracemalloc_peak_mb"] = tracemalloc_peak_mb
            report["memory_profiling"] = True
        return report
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        with suppress(Exception):
            from polylogue.storage.blob_store import reset_blob_store as _reset

            _reset()
        if owns_workdir:
            import shutil

            with suppress(OSError):
                shutil.rmtree(base)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--provider", default=_DEFAULT_PROVIDER, help="Synthetic provider to ingest")
    parser.add_argument("--batches", type=int, default=_DEFAULT_BATCHES, help="Number of single-session append batches")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="Deterministic corpus seed")
    parser.add_argument("--messages-min", type=int, default=_DEFAULT_MESSAGES_MIN, help="Min messages per session")
    parser.add_argument("--messages-max", type=int, default=_DEFAULT_MESSAGES_MAX, help="Max messages per session")
    parser.add_argument(
        "--lineage",
        action="store_true",
        help="Fork-heavy workload: one parent + N forks that replay its prefix (exercises lineage + #2488 memoization)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Add tracemalloc allocation profiling (distorts timing; not time-comparable to a normal run)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Reuse a directory for the fixture/archive instead of a private temp dir",
    )
    return parser


def _format_human(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Ingest throughput probe (time / CPU / memory / I/O / stages)")
    lines.append(
        f"  workload={report.get('workload')} provider={report.get('provider')} "
        f"batches={report.get('batches')} seed={report.get('seed')}"
    )
    lines.append("")
    lines.append(
        f"  totals: {report.get('total_sessions', 0)} sessions, "
        f"{report.get('total_messages', 0)} messages in {report.get('total_wall_s', 0.0):.3f}s"
    )
    lines.append(
        f"  throughput: {report.get('messages_per_s', 0.0):.1f} msg/s, "
        f"{report.get('sessions_per_s', 0.0):.1f} sessions/s"
    )
    storage = report.get("storage") or {}
    lines.append(
        f"  cpu: {report.get('cpu_seconds_total', 0.0):.3f}s "
        f"(utilization {report.get('cpu_utilization', 0.0):.2f}x), "
        f"peak_rss={report.get('peak_rss_mb', 0.0):.1f} MB"
    )
    proc_io = report.get("proc_io") or {}
    write_mb = proc_io.get("write_mb", 0.0) if report.get("proc_io_available") else None
    write_str = f"{write_mb:.2f} MB" if isinstance(write_mb, (int, float)) else "n/a"
    lines.append(
        f"  io: write={write_str} "
        f"({storage.get('bytes_written_per_message', 0.0):.0f} B/msg), "
        f"index.db growth={storage.get('db_growth_per_message', 0.0):.0f} B/msg"
    )
    per_batch_ms = report.get("per_batch_ms") or {}
    lines.append(
        "  per-batch ms: "
        f"min={per_batch_ms.get('min', 0.0):.1f} "
        f"mean={per_batch_ms.get('mean', 0.0):.1f} "
        f"p90={per_batch_ms.get('p90', 0.0):.1f} "
        f"max={per_batch_ms.get('max', 0.0):.1f}"
    )
    lines.append("")
    lines.append("  top stages (share of stage time):")
    for stage in (report.get("top_stages") or [])[:3]:
        lines.append(f"    {stage['stage']}: {stage['seconds']:.4f}s ({stage['pct_of_total_stage_time']:.1f}%)")
    if report.get("memory_profiling"):
        lines.append("")
        lines.append(
            f"  tracemalloc peak={report.get('tracemalloc_peak_mb', 0.0):.2f} MB "
            "(timings not comparable to a normal run)"
        )
    lines.append("  (wall-clock / CPU / memory / I/O are host-variable; no CI thresholds)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = measure_ingest_throughput(
            provider=args.provider,
            batches=max(1, args.batches),
            seed=args.seed,
            messages_min=args.messages_min,
            messages_max=args.messages_max,
            lineage=args.lineage,
            memory=args.memory,
            workdir=args.workdir,
        )
    except ValueError as exc:
        print(f"ingest-throughput-probe failed: {exc}")
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human(report))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
