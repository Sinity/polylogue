"""Run the AST-shape ratchet in ``devtools/patterns``.

Enforcing rules may inherit existing content anchors, but a new match is a
blocking defect. Stale baseline entries are reported as shrinkable debt.
Pending rules are scanned for visibility and deliberately do not block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from devtools import repo_root
from devtools.required_gate import evidence_gate_result

Anchor: TypeAlias = tuple[str, str]


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


def _anchor_text(anchor: Anchor, count: int = 1) -> str:
    file_name, digest = anchor
    suffix = f":{count}" if count != 1 else ""
    return f"{file_name}:{digest}{suffix}"


def _baseline(path: Path) -> Counter[Anchor]:
    if not path.exists():
        return Counter()
    anchors: Counter[Anchor] = Counter()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(":", 2)
        if len(parts) == 2:
            file_name, digest = parts
            count = 1
        elif len(parts) == 3 and parts[2].isdigit():
            file_name, digest, raw_count = parts
            count = int(raw_count)
        else:
            raise ValueError(f"invalid baseline entry in {path}: {raw_line!r}")
        if (
            not file_name
            or len(digest) != hashlib.sha1().digest_size * 2
            or any(character not in "0123456789abcdef" for character in digest)
            or count < 1
        ):
            raise ValueError(f"invalid baseline entry in {path}: {raw_line!r}")
        anchors[(file_name, digest)] += count
    return anchors


def _match_anchor(root: Path, item: dict[str, Any], file_lines: dict[str, list[str]]) -> Anchor:
    file_name = item.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("ast-grep returned a malformed match")
    range_payload = item.get("range")
    if not isinstance(range_payload, dict):
        raise ValueError("ast-grep returned a match without a range")
    start = range_payload.get("start")
    if not isinstance(start, dict) or not isinstance(start.get("line"), int):
        raise ValueError("ast-grep returned a match without a start line")
    line_number = start["line"] + 1
    if line_number < 1:
        raise ValueError("ast-grep returned an invalid start line")
    lines = file_lines.get(file_name)
    if lines is None:
        try:
            lines = (root / file_name).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read matched file {file_name}: {exc}") from exc
        file_lines[file_name] = lines
    if line_number > len(lines):
        raise ValueError(f"ast-grep match line is outside {file_name}: {line_number}")
    normalized_line = lines[line_number - 1].strip()
    digest = hashlib.sha1(normalized_line.encode("utf-8")).hexdigest()
    return file_name, digest


def _scan(root: Path, rule: Rule) -> Counter[Anchor]:
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
    matches: Counter[Anchor] = Counter()
    file_lines: dict[str, list[str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("ast-grep returned a malformed match")
        matches[_match_anchor(root, item, file_lines)] += 1
    return matches


def _payload(root: Path) -> dict[str, Any]:
    rules = _rules(root)
    details: list[str] = []
    new_matches: list[str] = []
    stale_matches: list[str] = []
    errors: list[str] = []
    missing = 0
    inspected = 0
    new_match_count = 0
    executable_available = shutil.which("ast-grep") is not None
    if not executable_available:
        gate = evidence_gate_result(
            gate="patterns",
            executable="ast-grep",
            executable_available=False,
            required_count=sum(rule.status == "enforcing" for rule in rules),
            inspected_count=0,
            details=(
                "ast-grep is required for the patterns gate; install it with "
                "`uv sync --group audit` and ensure the resulting executable is on PATH",
            ),
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
        new = matches - baseline
        stale = baseline - matches
        inspected += 1
        if rule.status == "pending":
            if baseline:
                errors.append(f"{rule.rule_id}: pending rule must have an empty baseline")
            details.append(f"{rule.rule_id}: pending ({sum(matches.values())} candidate matches; owner {rule.owner})")
            continue
        new_match_count += sum(new.values())
        new_matches.extend(
            f"{rule.rule_id} {_anchor_text(anchor, count)} (owner {rule.owner})"
            for anchor, count in sorted(new.items())
        )
        stale_matches.extend(f"{rule.rule_id} {_anchor_text(anchor, count)}" for anchor, count in sorted(stale.items()))
        details.append(f"{rule.rule_id}: enforcing ({sum(matches.values())} matches, {sum(stale.values())} prunable)")
    gate = evidence_gate_result(
        gate="patterns",
        executable="ast-grep",
        executable_available=True,
        required_count=sum(rule.status == "enforcing" for rule in rules),
        inspected_count=inspected,
        missing_count=missing,
        error_count=len(errors),
        semantic_violation_count=new_match_count,
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
