"""Measure compact coordination-envelope latency with stage attribution.

Use ``devtools bench coordination-latency --samples 21 --out PATH`` to write a
portable raw artifact.  Samples are intentionally taken through the production
envelope builder; each carries the complete per-stage timing map and the report
adds p50/p95 rather than hiding tail latency behind a single average.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(percentile / 100 * len(ordered))))
    return ordered[rank - 1]


def measure(*, samples: int, cwd: Path | None = None) -> dict[str, Any]:
    """Return raw compact samples and an honest distribution summary."""

    from polylogue.coordination.envelope import build_coordination_envelope
    from polylogue.paths import archive_root

    root = (cwd or Path.cwd()).resolve()
    raw: list[dict[str, object]] = []
    latencies: list[float] = []
    for number in range(samples):
        stages: dict[str, float] = {}
        started = perf_counter()
        payload = build_coordination_envelope(cwd=root, stage_timings_ms=stages)
        latency_ms = round((perf_counter() - started) * 1_000, 3)
        latencies.append(latency_ms)
        raw.append(
            {
                "sample": number,
                "latency_ms": latency_ms,
                "stages_ms": stages,
                "serialized_bytes": payload.projection.serialized_bytes,
            }
        )
    return {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_head": _git_head(root),
        "cwd": str(root),
        "archive_state": _archive_state(archive_root()),
        "mode": "warm-in-process-core",
        "samples": raw,
        "distribution_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
    }


def _archive_state(root: Path) -> dict[str, object]:
    """Record non-content archive facts needed to compare local samples."""

    index = root / "index.db"
    try:
        index_bytes: int | None = index.stat().st_size
    except OSError:
        index_bytes = None
    return {"root": str(root), "index_exists": index.exists(), "index_bytes": index_bytes}


def _git_head(cwd: Path) -> str | None:
    import subprocess

    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, capture_output=True, check=False)
    return completed.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write the full raw JSON artifact to this path.")
    args = parser.parse_args(argv)
    report = measure(samples=max(1, args.samples), cwd=args.cwd)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
