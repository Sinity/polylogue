"""Exercise the real pipeline on a small synthetic corpus and emit JSON metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

from polylogue.config import Config, Source
from polylogue.pipeline.runner import RUN_STAGE_CHOICES, run_sources
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.storage.backends.connection import open_connection

_EXT_MAP = {
    "chatgpt": ".json",
    "claude-ai": ".json",
    "gemini": ".json",
    "claude-code": ".jsonl",
    "codex": ".jsonl",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real pipeline against a small synthetic corpus and emit a JSON summary.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(SyntheticCorpus.available_providers()),
        default="chatgpt",
        help="Synthetic provider to generate (default: chatgpt)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Synthetic source files to generate (default: 5)",
    )
    parser.add_argument(
        "--messages-min",
        type=int,
        default=4,
        help="Minimum messages per conversation (default: 4)",
    )
    parser.add_argument(
        "--messages-max",
        type=int,
        default=12,
        help="Maximum messages per conversation (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Synthetic corpus seed (default: 42)",
    )
    parser.add_argument(
        "--stage",
        choices=RUN_STAGE_CHOICES,
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help=(
            "Probe workspace root. If omitted, a temporary workspace is created and removed after the run. "
            "Pass an explicit path when you want to keep the run/database artifacts."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for the JSON summary.",
    )
    parser.add_argument(
        "--max-total-ms",
        type=float,
        default=None,
        help="Fail if total pipeline runtime exceeds this budget in milliseconds.",
    )
    parser.add_argument(
        "--max-peak-rss-mb",
        type=float,
        default=None,
        help="Fail if peak RSS exceeds this budget in MiB.",
    )
    return parser.parse_args(argv)


@contextmanager
def _isolated_env(workdir: Path) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in (
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CONFIG_HOME",
        "POLYLOGUE_ARCHIVE_ROOT",
        "POLYLOGUE_RENDER_ROOT",
    )}
    env_updates = {
        "XDG_DATA_HOME": str(workdir / "xdg-data"),
        "XDG_STATE_HOME": str(workdir / "xdg-state"),
        "XDG_CONFIG_HOME": str(workdir / "xdg-config"),
        "POLYLOGUE_ARCHIVE_ROOT": str(workdir / "archive"),
        "POLYLOGUE_RENDER_ROOT": str(workdir / "render"),
    }
    for key, value in env_updates.items():
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _db_row_counts(db_path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not db_path.exists():
        return stats
    stats["db_size_bytes"] = db_path.stat().st_size
    with open_connection(db_path) as conn:
        for table in ("raw_conversations", "conversations", "messages", "content_blocks"):
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[f"{table}_count"] = int(row[0]) if row else 0
    return stats


def _write_probe_sources(
    *,
    provider: str,
    count: int,
    messages_min: int,
    messages_max: int,
    seed: int,
    source_root: Path,
) -> tuple[list[Path], int]:
    corpus = SyntheticCorpus.for_provider(provider)
    source_root.mkdir(parents=True, exist_ok=True)
    raw_items = corpus.generate(
        count=count,
        messages_per_conversation=range(messages_min, messages_max + 1),
        seed=seed,
    )
    total_bytes = 0
    files: list[Path] = []
    extension = _EXT_MAP.get(provider, ".json")
    for index, raw_bytes in enumerate(raw_items):
        file_path = source_root / f"{provider}-{index:03d}{extension}"
        file_path.write_bytes(raw_bytes)
        files.append(file_path)
        total_bytes += len(raw_bytes)
    return files, total_bytes


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.messages_min <= 0 or args.messages_max < args.messages_min:
        raise ValueError("--messages-min/--messages-max must define a positive inclusive range")

    workdir = args.workdir.resolve()
    source_root = workdir / "sources" / args.provider
    archive_root = workdir / "archive"
    render_root = workdir / "render"
    db_path: Path | None = None

    with _isolated_env(workdir):
        files, total_bytes = _write_probe_sources(
            provider=args.provider,
            count=args.count,
            messages_min=args.messages_min,
            messages_max=args.messages_max,
            seed=args.seed,
            source_root=source_root,
        )
        config = Config(
            sources=[Source(name=args.provider, path=source_root)],
            archive_root=archive_root,
            render_root=render_root,
        )
        db_path = config.db_path
        result = await run_sources(
            config=config,
            stage=args.stage,
            source_names=[args.provider],
        )
        run_payload: dict[str, Any] = {}
        if result.run_path:
            run_payload = json.loads(Path(result.run_path).read_text(encoding="utf-8"))

    return {
        "probe": {
            "provider": args.provider,
            "stage": args.stage,
            "count": args.count,
            "messages_min": args.messages_min,
            "messages_max": args.messages_max,
            "seed": args.seed,
        },
        "paths": {
            "workdir": str(workdir),
            "source_root": str(source_root),
            "archive_root": str(archive_root),
            "render_root": str(render_root),
            "db_path": str(db_path),
            "run_path": result.run_path,
        },
        "source_files": {
            "count": len(files),
            "total_bytes": total_bytes,
        },
        "result": result.model_dump(),
        "run_payload": run_payload,
        "db_stats": _db_row_counts(db_path) if db_path is not None else {},
    }


def _build_budget_report(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if args.max_total_ms is None and args.max_peak_rss_mb is None:
        return None

    metrics = summary.get("run_payload", {}).get("metrics", {})
    observed_total_ms = metrics.get("total_duration_ms", summary.get("result", {}).get("duration_ms"))
    observed_peak_rss_mb = metrics.get("peak_rss_mb")
    violations: list[str] = []

    if args.max_total_ms is not None:
        if observed_total_ms is None:
            violations.append("missing total runtime metric")
        elif float(observed_total_ms) > args.max_total_ms:
            violations.append(
                f"total runtime {float(observed_total_ms):.1f} ms exceeded budget {args.max_total_ms:.1f} ms"
            )

    if args.max_peak_rss_mb is not None:
        if observed_peak_rss_mb is None:
            violations.append("missing peak RSS metric")
        elif float(observed_peak_rss_mb) > args.max_peak_rss_mb:
            violations.append(
                f"peak RSS {float(observed_peak_rss_mb):.1f} MiB exceeded budget {args.max_peak_rss_mb:.1f} MiB"
            )

    return {
        "ok": not violations,
        "max_total_ms": args.max_total_ms,
        "observed_total_ms": observed_total_ms,
        "max_peak_rss_mb": args.max_peak_rss_mb,
        "observed_peak_rss_mb": observed_peak_rss_mb,
        "violations": violations,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix="polylogue-pipeline-probe-") as tempdir:
            args.workdir = Path(tempdir)
            with redirect_stdout(sys.stderr):
                summary = asyncio.run(run_probe(args))
    else:
        with redirect_stdout(sys.stderr):
            summary = asyncio.run(run_probe(args))
    budget_report = _build_budget_report(summary, args)
    if budget_report is not None:
        summary["budgets"] = budget_report
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if budget_report is not None and not budget_report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
