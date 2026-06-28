"""Repeatable ingest wall-clock / throughput benchmark.

The sibling ``devtools bench ingest-amplification`` probe measures *bytes
written* per archive tier — a deterministic, host-independent quantity.  It does
not measure *time*.  This probe is the missing wall-clock counterpart: it drives
the same public batch-ingest path (``parse_sources_archive`` — the real full
pipeline with the WAL write profile) over a deterministic synthetic corpus, and
times each append batch with ``time.perf_counter()``.

It is additive tooling.  It does **not** touch production ingest logic — it
drives the existing public batch-ingest path over a synthetic, deterministic
fixture corpus built under a temporary directory.

Operators run::

    devtools bench ingest-throughput --json > run.json
    devtools bench ingest-throughput --batches 20 --seed 2391

The emitted report carries the total wall, derived messages/sessions per second,
and a per-batch-millisecond distribution (min/max/mean/p90).  Wall-clock numbers
are host-variable: this probe has **no CI thresholds**.  It is a diagnostic plus
a campaign-comparable artifact — message/session *counts* are deterministic for
a fixed (provider, batches, seed), the timings are not.
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
REPORT_VERSION = 1

_DEFAULT_PROVIDER = "codex"
_DEFAULT_BATCHES = 12
_DEFAULT_SEED = 2391
_DEFAULT_MESSAGES_MIN = 4
_DEFAULT_MESSAGES_MAX = 8


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


async def _ingest_batch(archive_root: Path, provider: str, source_file: Path) -> dict[str, int]:
    from polylogue.config import Source
    from polylogue.pipeline.services.archive_ingest import parse_sources_archive

    result = await parse_sources_archive(archive_root, [Source(name=provider, path=source_file)])
    return {
        "sessions": int(result.counts.get("sessions", 0)),
        "messages": int(result.counts.get("messages", 0)),
    }


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile over a non-empty list of values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def measure_ingest_throughput(
    *,
    provider: str = _DEFAULT_PROVIDER,
    batches: int = _DEFAULT_BATCHES,
    seed: int = _DEFAULT_SEED,
    messages_min: int = _DEFAULT_MESSAGES_MIN,
    messages_max: int = _DEFAULT_MESSAGES_MAX,
    workdir: Path | None = None,
) -> dict[str, Any]:
    """Run ``batches`` single-session ingest batches and time each one.

    Returns a stable, JSON-serializable report.  When ``workdir`` is omitted a
    private temporary directory is created and torn down automatically; the
    archive and its blob store stay entirely inside that directory.

    Message/session counts are deterministic for a fixed (provider, batches,
    seed); the wall-clock timings are host-variable and carry no thresholds.
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

        batch_reports: list[dict[str, Any]] = []
        per_batch_ms: list[float] = []
        total_sessions = 0
        total_messages = 0

        wall_start = time.perf_counter()
        for index in range(batches):
            source_file = source_files[index]
            batch_start = time.perf_counter()
            ingested = asyncio.run(_ingest_batch(archive_root, provider, source_file))
            elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
            per_batch_ms.append(elapsed_ms)
            total_sessions += ingested["sessions"]
            total_messages += ingested["messages"]
            batch_reports.append(
                {
                    "batch_index": index,
                    "sessions_ingested": ingested["sessions"],
                    "messages_ingested": ingested["messages"],
                    "batch_ms": round(elapsed_ms, 3),
                }
            )
        total_wall_s = time.perf_counter() - wall_start

        per_batch_block = {
            "min": round(min(per_batch_ms), 3) if per_batch_ms else 0.0,
            "max": round(max(per_batch_ms), 3) if per_batch_ms else 0.0,
            "mean": round(sum(per_batch_ms) / len(per_batch_ms), 3) if per_batch_ms else 0.0,
            "p90": round(_percentile(per_batch_ms, 90.0), 3) if per_batch_ms else 0.0,
        }

        return {
            "ok": True,
            "report_version": REPORT_VERSION,
            "tool": "bench ingest-throughput",
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
            "per_batch_ms": per_batch_block,
            "per_batch": batch_reports,
        }
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
        "--workdir",
        type=Path,
        default=None,
        help="Reuse a directory for the fixture/archive instead of a private temp dir",
    )
    return parser


def _format_human(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Ingest throughput probe (wall-clock)")
    lines.append(
        f"  fixture: provider={report.get('provider')} batches={report.get('batches')} seed={report.get('seed')}"
    )
    lines.append("")
    lines.append("  per-batch (messages -> ms):")
    for batch in report.get("per_batch") or []:
        lines.append(
            f"    batch {batch['batch_index']}: {batch['messages_ingested']} msgs -> {batch['batch_ms']:.1f}ms"
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
    per_batch_ms = report.get("per_batch_ms") or {}
    lines.append(
        "  per-batch ms: "
        f"min={per_batch_ms.get('min', 0.0):.1f} "
        f"mean={per_batch_ms.get('mean', 0.0):.1f} "
        f"p90={per_batch_ms.get('p90', 0.0):.1f} "
        f"max={per_batch_ms.get('max', 0.0):.1f}"
    )
    lines.append("  (wall-clock is host-variable; no CI thresholds)")
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
