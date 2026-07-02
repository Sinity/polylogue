"""Profile the shared temporal read-view builder on the active archive."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from polylogue.cli.read_views.standard import build_read_temporal_window
from polylogue.cli.root_request import RootModeRequest
from polylogue.config import Config, get_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace temporal-read-profile",
        description="Measure phase timings for read --view temporal using the shared read-view builder.",
    )
    parser.add_argument("--query", default="repo:polylogue", help="Root query expression to select sessions.")
    parser.add_argument("--limit", type=int, default=1, help="Session summary limit for the temporal read.")
    parser.add_argument("--archive-root", type=Path, default=None, help="Override the active archive root.")
    parser.add_argument("--out", type=Path, default=None, help="Write the JSON report to this path.")
    parser.add_argument("--include-window", action="store_true", help="Include the full temporal_window payload.")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout. Accepted for devtools parity.")
    return parser


def _config_with_archive_root(config: Config, archive_root: Path | None) -> Config:
    if archive_root is None:
        return config
    resolved = archive_root.expanduser().resolve()
    return Config(
        archive_root=resolved,
        render_root=config.render_root,
        sources=config.sources,
        db_path=resolved / "index.db",
        drive_config=config.drive_config,
        index_config=config.index_config,
    )


def _phase_summary(phases: list[dict[str, object]]) -> dict[str, object]:
    def elapsed_ms(phase: dict[str, object]) -> float:
        return cast(float, phase["elapsed_ms"])

    by_phase = {str(phase["name"]): elapsed_ms(phase) for phase in phases}
    slowest = max(phases, key=elapsed_ms, default=None)
    return {
        "phase_count": len(phases),
        "elapsed_by_phase_ms": by_phase,
        "slowest_phase": None if slowest is None else slowest["name"],
        "slowest_phase_ms": None if slowest is None else slowest["elapsed_ms"],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    config = _config_with_archive_root(get_config(), args.archive_root)
    phases: list[dict[str, object]] = []

    def record_phase(name: str, elapsed_ms: float, details: Mapping[str, object]) -> None:
        phases.append(
            {
                "name": name,
                "elapsed_ms": round(elapsed_ms, 3),
                "details": details,
            }
        )

    request = RootModeRequest.from_params({"query": (args.query,), "limit": args.limit})
    started = time.perf_counter()
    window = build_read_temporal_window(config, request, phase_recorder=record_phase)
    total_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    report: dict[str, Any] = {
        "report_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "command": "devtools workspace temporal-read-profile",
        "archive_root": str(config.archive_root),
        "index_db": str(config.db_path),
        "query": args.query,
        "limit": args.limit,
        "total_elapsed_ms": total_elapsed_ms,
        "phases": phases,
        "phase_summary": _phase_summary(phases),
        "temporal_window_summary": {
            "event_count": window.event_count,
            "family_counts": dict(window.family_counts),
            "kind_counts": dict(window.kind_counts),
            "caveats": list(window.caveats),
        },
    }
    if args.include_window:
        report["temporal_window"] = window.model_dump(mode="json")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
