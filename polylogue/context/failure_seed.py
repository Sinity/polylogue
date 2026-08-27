"""Compile a bounded debugging seed from verification evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VERIFY_ROOT = Path(".cache/verify")
SEED_PATH = VERIFY_ROOT / "context-seed.json"


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def latest_postmortem(root: Path = Path.cwd()) -> tuple[Path, dict[str, Any]] | None:
    """Return the newest failed verify receipt and its payload."""
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for path in (root / VERIFY_ROOT / "runs").glob("*/run.json"):
        payload = _json(path)
        if payload and payload.get("status") == "failed":
            candidates.append((path.stat().st_mtime, path, payload))
    if not candidates:
        return None
    _mtime, path, payload = max(candidates, key=lambda item: item[0])
    return path, payload


def _failure_ids(receipt: Mapping[str, Any], root: Path) -> list[str]:
    result: list[str] = []
    for step in receipt.get("steps", ()):
        if not isinstance(step, Mapping):
            continue
        artifact = step.get("artifact_dir")
        if not isinstance(artifact, str):
            continue
        report = _json(root / artifact / "pytest-report.json")
        for test in (report or {}).get("tests", ()):
            if isinstance(test, Mapping) and test.get("outcome") == "failed" and isinstance(test.get("nodeid"), str):
                result.append(test["nodeid"])
        statistics = _json(root / artifact / "statistics.json")
        for failure in (statistics or {}).get("failures", ()):
            if isinstance(failure, str):
                result.append(failure)
    aggregate = receipt.get("pytest_aggregate")
    if not result and isinstance(aggregate, Mapping):
        result.extend(item for item in aggregate.get("failed_tests", ()) if isinstance(item, str))
    return sorted(set(result))


def _envelope_path(root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_absolute() else root / explicit
    configured = os.environ.get("POLYLOGUE_FAILURE_CONTEXT_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else root / path
    for candidate in (root / VERIFY_ROOT / "failure-context.json", root / ".cache" / "failure-context.json"):
        if candidate.is_file():
            return candidate
    return None


def compile_failure_seed(*, root: Path = Path.cwd(), envelope_path: Path | None = None) -> dict[str, Any]:
    """Join the latest failed verify receipt with a workspace context envelope."""
    postmortem = latest_postmortem(root)
    if postmortem is None:
        raise FileNotFoundError("no failed verification postmortem exists under .cache/verify/runs")
    receipt_path, receipt = postmortem
    path = _envelope_path(root, envelope_path)
    if path is None:
        raise FileNotFoundError("workspace failure-context envelope not found; pass --failure-context PATH")
    envelope = _json(path)
    if envelope is None:
        raise ValueError(f"failure-context envelope is not a JSON object: {path}")
    failures = _failure_ids(receipt, root)
    if not failures and isinstance(envelope.get("failure_id"), str):
        failures = [envelope["failure_id"]]
    files: set[str] = set()
    for key in ("implicated_files", "testmon_dependencies"):
        values = envelope.get(key, ())
        if isinstance(values, list):
            files.update(item for item in values if isinstance(item, str))
    recent = envelope.get("recent_changes")
    if isinstance(recent, Mapping):
        files.update(str(item) for item in recent)
    seed = {
        "purpose": "debug",
        "failure_tests": failures,
        "implicated_files": sorted(files),
        "postmortem_ref": str(receipt_path.relative_to(root)),
        "failure_context_ref": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "next_command": f"devtools test {failures[0]}" if failures else "devtools test <failing-node>",
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "postmortem": {
            "run_id": receipt.get("run_id"),
            "diagnosis": receipt.get("diagnosis"),
            "status": receipt.get("status"),
        },
        "failure_context": envelope,
    }


def write_failure_seed(*, root: Path = Path.cwd(), envelope_path: Path | None = None) -> Path:
    payload = compile_failure_seed(root=root, envelope_path=envelope_path)
    destination = root / SEED_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


__all__ = ["SEED_PATH", "compile_failure_seed", "latest_postmortem", "write_failure_seed"]
