"""Run the AST-shape ratchet in ``devtools/patterns``.

Enforcing rules may inherit existing file/line matches, but a new match is a
blocking defect.  Stale baseline entries are reported as shrinkable debt.
Pending rules are scanned for visibility and deliberately do not block.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from devtools import repo_root
from devtools.required_gate import evidence_gate_result


@dataclass(frozen=True)
class Rule:
    rule_id: str
    rule_path: Path
    baseline_path: Path
    owner: str
    status: str


def _rules(root: Path) -> tuple[Rule, ...]:
    raw = yaml.safe_load((root / "devtools/patterns/registry.yaml").read_text(encoding="utf-8"))
    entries = raw.get("rules", []) if isinstance(raw, dict) else []
    result: list[Rule] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        result.append(
            Rule(
                rule_id=str(entry["id"]),
                rule_path=root / "devtools/patterns" / str(entry["rule"]),
                baseline_path=root / "devtools/patterns" / str(entry["baseline"]),
                owner=str(entry["owner"]),
                status=str(entry["status"]),
            )
        )
    return tuple(result)


def _baseline(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    locations: set[tuple[str, int]] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        file_name, separator, raw_number = line.rpartition(":")
        if not separator or not file_name or not raw_number.isdigit():
            raise ValueError(f"invalid baseline entry in {path}: {raw_line!r}")
        locations.add((file_name, int(raw_number)))
    return locations


def _scan(root: Path, rule: Rule) -> set[tuple[str, int]]:
    command = [
        "ast-grep",
        "scan",
        "--rule",
        str(rule.rule_path),
        "--json=compact",
        "--globs",
        "*.py",
        "polylogue",
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
        raise RuntimeError(detail)
    payload = json.loads(completed.stdout or "[]")
    if not isinstance(payload, list):
        raise ValueError("ast-grep returned a non-list JSON result")
    matches: set[tuple[str, int]] = set()
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise ValueError("ast-grep returned a malformed match")
        start = item.get("range", {}).get("start", {})
        if not isinstance(start, dict) or not isinstance(start.get("line"), int):
            raise ValueError("ast-grep returned a match without a start line")
        matches.add((item["file"], start["line"]))
    return matches


def _payload(root: Path) -> dict[str, Any]:
    rules = _rules(root)
    details: list[str] = []
    new_matches: list[str] = []
    stale_matches: list[str] = []
    errors: list[str] = []
    missing = 0
    inspected = 0
    executable_available = shutil.which("ast-grep") is not None
    if not executable_available:
        gate = evidence_gate_result(
            gate="patterns",
            executable="ast-grep",
            executable_available=False,
            required_count=sum(rule.status == "enforcing" for rule in rules),
            inspected_count=0,
        )
        return {"blocking": True, "new_matches": [], "stale_matches": [], "required_gate": gate.to_payload()}
    for rule in rules:
        try:
            if rule.status == "enforcing" and not rule.baseline_path.is_file():
                missing += 1
                errors.append(f"{rule.rule_id}: missing baseline {rule.baseline_path.relative_to(root)}")
                continue
            matches = _scan(root, rule)
            baseline = _baseline(rule.baseline_path)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            errors.append(f"{rule.rule_id}: {exc}")
            continue
        new = sorted(matches - baseline)
        stale = sorted(baseline - matches)
        inspected += 1
        if rule.status == "pending":
            if baseline:
                errors.append(f"{rule.rule_id}: pending rule must have an empty baseline")
            details.append(f"{rule.rule_id}: pending ({len(matches)} candidate matches; owner {rule.owner})")
            continue
        new_matches.extend(f"{rule.rule_id} {file_name}:{line} (owner {rule.owner})" for file_name, line in new)
        stale_matches.extend(f"{rule.rule_id} {file_name}:{line}" for file_name, line in stale)
        details.append(f"{rule.rule_id}: enforcing ({len(matches)} matches, {len(stale)} prunable)")
    gate = evidence_gate_result(
        gate="patterns",
        executable="ast-grep",
        executable_available=True,
        required_count=sum(rule.status == "enforcing" for rule in rules),
        inspected_count=inspected,
        missing_count=missing,
        error_count=len(errors),
        semantic_violation_count=len(new_matches),
        details=(*errors, *new_matches, *(f"stale baseline: {item}" for item in stale_matches), *details),
    )
    return {
        "blocking": not gate.ok,
        "new_matches": new_matches,
        "stale_matches": stale_matches,
        "required_gate": gate.to_payload(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = _payload(repo_root())
    except (OSError, yaml.YAMLError, KeyError, RuntimeError, ValueError) as exc:
        payload = {"blocking": True, "error": str(exc)}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        gate = payload.get("required_gate", {})
        for detail in gate.get("details", []):
            print(detail)
        if payload.get("error"):
            print(f"patterns: {payload['error']}")
        print("patterns: " + ("failed" if payload.get("blocking") else "passed"))
    return 1 if payload.get("blocking") else 0


if __name__ == "__main__":
    raise SystemExit(main())
