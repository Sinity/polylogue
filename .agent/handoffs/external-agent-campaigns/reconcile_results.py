#!/usr/bin/env python3
"""Materialize and verify campaign result indexes from immutable receipts.

``results/index.json`` is a rebuildable query projection.  The authoritative
outcome is written once in either an attempt ``result.json`` or a package
revision ``receipt.json``.  This command deliberately never writes receipts.

Usage::

    reconcile_results.py <wave-root> --check
    reconcile_results.py <wave-root> --write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReconciliationError(ValueError):
    """A projection cannot be trusted or rebuilt without operator judgment."""


@dataclass(frozen=True)
class Receipt:
    path: Path
    workload: str
    job: str
    attempt: str | None
    revision: str | None
    state: str
    payload: dict[str, Any]

    @property
    def key(self) -> tuple[str, str, str]:
        if self.attempt is not None:
            return (self.job, "attempt", self.attempt)
        assert self.revision is not None
        return (self.job, "revision", self.revision)


@dataclass(frozen=True)
class WorkloadPlan:
    index_path: Path
    rendered: str
    errors: tuple[str, ...]
    unrebuildable: bool


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ReconciliationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconciliationError(f"expected an object in {path}")
    return value


def require_text(payload: dict[str, Any], field: str, path: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ReconciliationError(f"{path}: missing non-empty {field}")
    return value


def discover_receipts(workload_root: Path, campaign_id: str, workload: str) -> list[Receipt]:
    results = workload_root / "results"
    receipts: list[Receipt] = []
    for path in sorted(results.glob("*/*/result.json")):
        payload = load_json(path)
        job, attempt = path.parts[-3:-1]
        if not attempt.startswith("a"):
            raise ReconciliationError(f"{path}: result.json must live below an aNN attempt directory")
        _validate_common(payload, path, campaign_id, workload, job)
        if require_text(payload, "attempt_id", path) != attempt:
            raise ReconciliationError(f"{path}: attempt_id does not match its directory")
        revision = payload.get("package_revision")
        if revision is not None and (not isinstance(revision, str) or not revision.startswith("r")):
            raise ReconciliationError(f"{path}: package_revision must be rNN or null")
        receipts.append(Receipt(path, workload, job, attempt, revision, require_text(payload, "state", path), payload))
    for path in sorted(results.glob("*/*/receipt.json")):
        payload = load_json(path)
        job, revision = path.parts[-3:-1]
        if not revision.startswith("r"):
            raise ReconciliationError(f"{path}: receipt.json must live below an rNN revision directory")
        _validate_common(payload, path, campaign_id, workload, job)
        if require_text(payload, "package_revision", path) != revision:
            raise ReconciliationError(f"{path}: package_revision does not match its directory")
        if "attempt_id" in payload:
            raise ReconciliationError(f"{path}: revision receipt must not carry attempt_id; use result.json")
        receipts.append(Receipt(path, workload, job, None, revision, require_text(payload, "state", path), payload))
    keys = [receipt.key for receipt in receipts]
    if len(keys) != len(set(keys)):
        duplicates = sorted(key for key in set(keys) if keys.count(key) > 1)
        raise ReconciliationError(f"{workload_root}: duplicate immutable receipt identities: {duplicates}")
    return receipts


def _validate_common(payload: dict[str, Any], path: Path, campaign_id: str, workload: str, job: str) -> None:
    if require_text(payload, "campaign_id", path) != campaign_id:
        raise ReconciliationError(f"{path}: campaign_id does not match wave")
    if require_text(payload, "workload_id", path) != workload:
        raise ReconciliationError(f"{path}: workload_id does not match directory")
    if require_text(payload, "job_id", path) != job:
        raise ReconciliationError(f"{path}: job_id does not match directory")


def index_key(entry: dict[str, Any], path: Path) -> tuple[str, str, str]:
    job = entry.get("job")
    if not isinstance(job, str) or not job:
        raise ReconciliationError(f"{path}: index entry has no job")
    attempt = entry.get("attempt_id")
    revision = entry.get("package_revision")
    legacy_revision = entry.get("revision")
    if revision is not None and legacy_revision is not None and revision != legacy_revision:
        raise ReconciliationError(f"{path}: index entry {job} has conflicting revision identifiers")
    if revision is None:
        revision = legacy_revision
    if isinstance(attempt, str) and attempt:
        return (job, "attempt", attempt)
    if isinstance(revision, str) and revision:
        return (job, "revision", revision)
    raise ReconciliationError(f"{path}: index entry {job} needs exactly one of attempt_id or package_revision")


def projection_entry(receipt: Receipt) -> dict[str, Any]:
    payload = receipt.payload
    artifact: dict[str, Any] | None = None
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list) and len(artifacts) == 1 and isinstance(artifacts[0], dict):
        artifact = artifacts[0]
    elif isinstance(payload.get("artifact"), dict):
        artifact = payload["artifact"]
    entry: dict[str, Any] = {"job": receipt.job, "workload": receipt.workload, "state": receipt.state}
    if receipt.attempt is not None:
        entry["attempt_id"] = receipt.attempt
    else:
        entry["package_revision"] = receipt.revision
    if receipt.attempt is not None and receipt.revision is not None:
        entry["package_revision"] = receipt.revision
    if artifact is not None:
        entry.update(
            {
                "artifact": artifact.get("filename"),
                "sha256": artifact.get("sha256"),
                "bytes": artifact.get("size_bytes"),
            }
        )
    return {key: value for key, value in entry.items() if value is not None}


def plan_workload(workload_root: Path) -> WorkloadPlan:
    campaign = load_json(workload_root / "campaign.json")
    campaign_id = require_text(campaign, "campaign_id", workload_root / "campaign.json")
    workload = require_text(campaign, "workload_id", workload_root / "campaign.json")
    index_path = workload_root / "results" / "index.json"
    index = load_json(index_path)
    if index.get("campaign_id") != campaign_id or index.get("workload_id") != workload:
        raise ReconciliationError(f"{index_path}: campaign/workload identity does not match campaign.json")
    attempts = index.get("attempts")
    if not isinstance(attempts, list):
        raise ReconciliationError(f"{index_path}: attempts must be an array")
    receipts = discover_receipts(workload_root, campaign_id, workload)
    receipt_by_key = {receipt.key: receipt for receipt in receipts}
    entry_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    errors: list[str] = []
    unrebuildable = False
    for entry in attempts:
        if not isinstance(entry, dict):
            errors.append(f"{index_path}: index contains a non-object entry")
            unrebuildable = True
            continue
        if entry.get("workload") != workload:
            errors.append(f"{index_path}: index entry workload does not match {workload}")
            continue
        try:
            key = index_key(entry, index_path)
        except ReconciliationError as exc:
            errors.append(str(exc))
            unrebuildable = True
            continue
        if key in entry_by_key:
            errors.append(f"{index_path}: duplicate projection identity {key}")
            unrebuildable = True
        else:
            entry_by_key[key] = entry
    for key, entry in entry_by_key.items():
        receipt = receipt_by_key.get(key)
        if receipt is None:
            errors.append(f"{index_path}: projection {key} has no immutable receipt")
            unrebuildable = True
            continue
        if "status" in entry:
            errors.append(f"{index_path}: projection {key} has forbidden legacy status field")
        if "revision" in entry:
            errors.append(f"{index_path}: projection {key} has forbidden legacy revision field")
        if key[1] == "attempt" and entry.get("package_revision") != receipt.revision:
            errors.append(
                f"{index_path}: projection {key} package_revision {entry.get('package_revision')!r} "
                f"does not equal receipt package_revision {receipt.revision!r}"
            )
        if entry.get("state") != receipt.state:
            errors.append(
                f"{index_path}: projection {key} state {entry.get('state')!r} "
                f"does not equal receipt state {receipt.state!r}"
            )
    for key in receipt_by_key:
        if key not in entry_by_key:
            errors.append(f"{index_path}: immutable receipt {key} has no projection")
    materialized: list[dict[str, Any]] = []
    for receipt in receipts:
        entry = entry_by_key.get(receipt.key)
        if entry is None:
            materialized.append(projection_entry(receipt))
            continue
        projected = dict(entry)
        projected.pop("status", None)
        legacy_revision = projected.pop("revision", None)
        if "package_revision" not in projected and legacy_revision is not None:
            projected["package_revision"] = legacy_revision
        projected["state"] = receipt.state
        materialized.append(projected)
    updated = dict(index)
    updated["attempts"] = materialized
    rendered = json.dumps(updated, indent=2) + "\n"
    if not errors and index_path.read_text() != rendered:
        errors.append(f"{index_path}: projection is not in canonical receipt order/format")
    return WorkloadPlan(index_path, rendered, tuple(errors), unrebuildable)


def atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def workload_roots(wave_root: Path) -> list[Path]:
    roots = sorted(path.parent for path in wave_root.glob("*/campaign.json"))
    if not roots:
        raise ReconciliationError(f"{wave_root}: no workload campaign.json files found")
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wave_root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail on any projection drift (default)")
    mode.add_argument("--write", action="store_true", help="rebuild indexes only; immutable receipts are never changed")
    args = parser.parse_args()
    try:
        plans = [plan_workload(root) for root in workload_roots(args.wave_root)]
    except ReconciliationError as exc:
        plans = []
        problems = [str(exc)]
    else:
        problems = [problem for plan in plans for problem in plan.errors]
        if args.write and not any(plan.unrebuildable for plan in plans):
            for plan in plans:
                atomic_write(plan.index_path, plan.rendered)
            problems = []
    if problems:
        print("receipt/index reconciliation failed:", file=sys.stderr)
        print("\n".join(f"- {problem}" for problem in problems), file=sys.stderr)
        return 2
    print(f"receipt/index reconciliation {'materialized' if args.write else 'verified'} for {args.wave_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
