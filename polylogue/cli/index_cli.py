from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from ..commands import CommandEnv
from ..index_health import verify_qdrant_collection, verify_sqlite_indexes
from ..schema import stamp_payload


def run_index_cli(args: argparse.Namespace, env: CommandEnv) -> None:
    # subcmd is enforced by argparse (required=True), so this always calls check
    run_index_check(args, env)


def run_index_check(args: argparse.Namespace, env: CommandEnv) -> None:
    report: Dict[str, Any] = {
        "sqlite": {"status": "ok", "messages": []},
        "qdrant": {"status": "skipped", "messages": []},
    }
    try:
        notes = verify_sqlite_indexes(attempt_rebuild=bool(getattr(args, "repair", False)))
        report["sqlite"]["messages"].extend(notes)
        if notes:
            report["sqlite"]["status"] = "updated"
    except Exception as exc:  # pragma: no cover - defensive
        report["sqlite"]["status"] = "error"
        report["sqlite"]["messages"].append(str(exc))

    if not getattr(args, "skip_qdrant", False):
        try:
            notes = verify_qdrant_collection()
            report["qdrant"]["messages"].extend(notes)
            report["qdrant"]["status"] = "ok" if not notes else "updated"
        except RuntimeError as exc:
            report["qdrant"]["status"] = "error"
            report["qdrant"]["messages"].append(str(exc))

    if getattr(args, "json", False):
        payload = stamp_payload(report)
        print(json.dumps(payload, indent=2))
        return

    ui = env.ui
    lines: List[str] = []
    lines.append(f"SQLite index: {report['sqlite']['status']}")
    for msg in report["sqlite"]["messages"]:
        lines.append(f"  - {msg}")
    if report["qdrant"]["status"] != "skipped":
        lines.append(f"Qdrant index: {report['qdrant']['status']}")
        for msg in report["qdrant"]["messages"]:
            lines.append(f"  - {msg}")
    else:
        lines.append("Qdrant index: skipped (use --skip-qdrant to suppress)")
    ui.summary("Index Check", lines)
