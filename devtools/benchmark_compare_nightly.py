"""Compare pytest-benchmark JSON output against a committed baseline (#1220).

Used by the nightly-scale workflow to detect benchmark regressions.
Exits 0 when within threshold, 1 when regressions exceed --fail-pct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(baseline: Path, candidate: Path, fail_pct: float) -> int:
    with open(candidate) as f:
        current = json.load(f)
    with open(baseline) as f:
        prev = json.load(f)

    baseline_map = {b["fullname"]: b["stats"]["mean"] for b in prev.get("benchmarks", [])}
    regressions: list[tuple[str, float, float, float]] = []
    for b in current.get("benchmarks", []):
        name = b["fullname"]
        bl_mean = baseline_map.get(name)
        if bl_mean and bl_mean > 0:
            delta_pct = (b["stats"]["mean"] - bl_mean) / bl_mean * 100
            if delta_pct > fail_pct:
                regressions.append((name, delta_pct, bl_mean, b["stats"]["mean"]))

    if regressions:
        regressions.sort(key=lambda r: r[1], reverse=True)
        print(f"FAIL: {len(regressions)} benchmark(s) degraded >{fail_pct:.0f}%:")
        for name, pct, prev_mean, cur_mean in regressions[:10]:
            print(f"  {name}: +{pct:.1f}% ({prev_mean:.4f}s -> {cur_mean:.4f}s)")
        return 1
    else:
        print(f"OK: no benchmark regressions exceeding {fail_pct:.0f}% threshold.")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare pytest-benchmark JSON against a baseline.")
    parser.add_argument("baseline", type=Path, help="Path to baseline JSON.")
    parser.add_argument("candidate", type=Path, help="Path to candidate JSON.")
    parser.add_argument("--fail-pct", type=float, default=20.0, help="Regression threshold in percent.")
    args = parser.parse_args(argv)
    return compare(args.baseline, args.candidate, args.fail_pct)


if __name__ == "__main__":
    raise SystemExit(main())
