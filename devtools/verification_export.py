"""Read-only, versioned export of one managed verification receipt."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devtools import repo_root

SCHEMA = "polylogue.devtools.verification-export/v1"


def _receipt(root: Path, run_ref: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(run_ref)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {"", ".", ".."}:
        raise ValueError("run_ref must be one managed run id")
    path = root / ".cache" / "verify" / "runs" / candidate / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("verification receipt must be a JSON object")
    return path, payload


def normalized_export(root: Path, run_ref: str) -> dict[str, Any]:
    """Normalize one existing receipt without reading testmon or other caches."""
    path, receipt = _receipt(root, run_ref)
    steps = receipt.get("steps") if isinstance(receipt.get("steps"), list) else []
    pytest_steps = [
        step for step in steps if isinstance(step, Mapping) and str(step.get("name", "")).startswith("pytest")
    ]
    profile = next((step.get("hypothesis_profile") for step in pytest_steps if step.get("hypothesis_profile")), None)
    commands = [list(step.get("cmd", ())) for step in pytest_steps if isinstance(step.get("cmd"), list)]
    initial_sha = receipt.get("git_head")
    final_sha = receipt.get("final_git_head")
    dirty_before = receipt.get("git_dirty")
    dirty_after = receipt.get("final_git_dirty")
    tested_sha: str | None = None
    tested_sha_reason: str | None = None
    if not isinstance(initial_sha, str) or not isinstance(final_sha, str):
        tested_sha_reason = "checkout SHA unavailable"
    elif dirty_before is not False or dirty_after is not False:
        tested_sha_reason = "checkout was dirty or final cleanliness was not recorded"
    elif initial_sha != final_sha:
        tested_sha_reason = "checkout HEAD changed during verification"
    else:
        tested_sha = initial_sha
    return {
        "schema": SCHEMA,
        "checkout": str(root),
        "run_ref": str(path.relative_to(root)),
        "run_id": receipt.get("run_id"),
        "initial_sha": initial_sha,
        "final_sha": final_sha,
        "git_dirty_before": dirty_before,
        "git_dirty_after": dirty_after,
        "tested_sha": tested_sha,
        "tested_sha_reason": tested_sha_reason,
        "status": receipt.get("status"),
        "exit_code": receipt.get("exit_code"),
        "hypothesis_profile": profile,
        "commands": commands,
        "phases": [
            {
                "name": step.get("name"),
                "duration_s": step.get("duration_s"),
                "exit": step.get("exit"),
                "runner": step.get("runner"),
            }
            for step in steps
            if isinstance(step, Mapping)
        ],
        "coverage": receipt.get("pytest_aggregate", {}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ref", help="managed run id, resolved below .cache/verify/runs")
    parser.add_argument("--json", action="store_true", help="required for machine-readable export")
    args = parser.parse_args(argv)
    if not args.json:
        parser.error("--json is required")
    try:
        payload = normalized_export(repo_root(), args.run_ref)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
