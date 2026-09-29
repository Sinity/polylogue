"""Run the AST-shape ratchet in ``devtools/patterns``.

Enforcing rules may inherit existing content anchors, but a new match is a
blocking defect. Stale baseline entries are reported as shrinkable debt.
Pending rules are scanned for visibility and deliberately do not block.
"""

from __future__ import annotations

import argparse
import ast
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
from devtools.required_gate import AUDIT_GROUP_SYNC_COMMAND, evidence_gate_result

Anchor: TypeAlias = tuple[str, str, str]


@dataclass(frozen=True)
class Rule:
    rule_id: str
    rule_path: Path
    baseline_path: Path
    owner: str
    status: str


def _rules(root: Path) -> tuple[Rule, ...]:
    raw = yaml.safe_load((root / "devtools/patterns/registry.yaml").read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("rules"), list):
        raise ValueError("pattern registry must be a mapping with a rules list")
    entries = raw["rules"]
    result: list[Rule] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"pattern registry rule {index} must be a mapping")
        required = ("id", "rule", "baseline", "owner", "status")
        if any(not isinstance(entry.get(key), str) or not entry[key] for key in required):
            raise ValueError(f"pattern registry rule {index} has missing or invalid fields")
        if entry["id"] in seen:
            raise ValueError(f"duplicate pattern rule id: {entry['id']}")
        if entry["status"] not in {"enforcing", "pending"}:
            raise ValueError(f"invalid pattern rule status for {entry['id']}: {entry['status']}")
        seen.add(entry["id"])
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
    file_name, digest, context = anchor
    suffix = f":{count}" if count != 1 else ""
    return f"{file_name}:{digest}:{context}{suffix}"


def _baseline(path: Path) -> Counter[Anchor]:
    if not path.exists():
        return Counter()
    anchors: Counter[Anchor] = Counter()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(":", 3)
        if len(parts) == 3:
            file_name, digest, context = parts
            count = 1
        elif len(parts) == 4 and parts[3].isdigit():
            file_name, digest, context, raw_count = parts
            count = int(raw_count)
        else:
            raise ValueError(f"invalid baseline entry in {path}: {raw_line!r}")
        if (
            not file_name
            or len(digest) != hashlib.sha1().digest_size * 2
            or any(character not in "0123456789abcdef" for character in digest)
            or len(context) != hashlib.sha1().digest_size * 2
            or any(character not in "0123456789abcdef" for character in context)
            or count < 1
        ):
            raise ValueError(f"invalid baseline entry in {path}: {raw_line!r}")
        anchors[(file_name, digest, context)] += count
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
    source = "\n".join(lines)
    tree = ast.parse(source, filename=file_name)

    def path(node: ast.AST, trail: tuple[str, ...] = ()) -> tuple[str, ...] | None:
        line_start = node.__dict__.get("lineno")
        line_end = node.__dict__.get("end_lineno")
        if not isinstance(line_start, int) or not isinstance(line_end, int):
            return None
        if not (line_start <= line_number <= line_end):
            return None
        best: tuple[str, ...] = trail + (type(node).__name__,)
        for field_name, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for index, child in enumerate(children):
                if isinstance(child, ast.AST):
                    child_path = path(child, trail + (f"{type(node).__name__}.{field_name}[{index}]",))
                    if child_path is not None and len(child_path) > len(best):
                        best = child_path
        return best

    context_text = "/".join(path(tree) or ("Module",))
    context = hashlib.sha1(context_text.encode("utf-8")).hexdigest()
    return file_name, digest, context


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


def _trusted_baseline(root: Path, path: Path) -> Counter[Anchor] | None:
    """Read the parent revisions' exemption set; synthetic roots have none.

    A merge commit has several parents: an exemption any parent already
    carried is trusted, at the largest count any parent carried it. Reading
    only the first parent would count the base branch's own baseline additions,
    brought in by merging it, as growth.
    """
    try:
        repository = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        relative = path.resolve().relative_to(Path(repository).resolve()).as_posix()
        parents = subprocess.run(
            ["git", "-C", repository, "rev-parse", "HEAD^@"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if not parents:
            raise ValueError("HEAD has no parent revision")
        # A parent that predates the baseline file (a base branch merged into
        # the feature that introduced it) contributes nothing; a file no
        # parent carries has no trusted revision at all.
        contents = [
            completed.stdout
            for parent in parents
            if (
                completed := subprocess.run(
                    ["git", "-C", repository, "show", f"{parent}:{relative}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            ).returncode
            == 0
        ]
        if not contents:
            raise ValueError(f"no parent revision carries {relative}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        if not (root / ".git").exists():
            return None
        raise ValueError(f"cannot load trusted parent baseline for {path}") from exc
    trusted: Counter[Anchor] = Counter()
    for content in contents:
        trusted |= _baseline_text(content)
    return trusted


def _baseline_text(content: str) -> Counter[Anchor]:
    """Parse a trusted revision's entries as full ``(file, digest, context)`` anchors.

    A context-free legacy entry keeps an empty context, so it never authorizes
    an entry in any recorded AST context.
    """
    anchors: Counter[Anchor] = Counter()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(":", 3)
        context = ""
        if len(parts) == 2:
            file_name, digest = parts
            count = 1
        elif len(parts) == 3 and parts[2].isdigit():
            file_name, digest, raw_count = parts
            count = int(raw_count)
        elif len(parts) == 3:
            file_name, digest, context = parts
            count = 1
        elif len(parts) == 4 and parts[3].isdigit():
            file_name, digest, context, raw_count = parts
            count = int(raw_count)
        else:
            raise ValueError(f"invalid trusted baseline entry: {raw_line!r}")
        anchors[(file_name, digest, context)] += count
    return anchors


def _payload(root: Path) -> dict[str, Any]:
    try:
        rules = _rules(root)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        gate = evidence_gate_result(
            gate="patterns",
            executable="ast-grep",
            executable_available=True,
            required_count=0,
            inspected_count=0,
            details=(f"malformed pattern registry: {exc}",),
        )
        return {"blocking": True, "new_matches": [], "stale_matches": [], "required_gate": gate.to_payload()}
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
                "ast-grep is required for the patterns gate; nothing was inspected, so this is "
                "an unprovisioned checkout rather than a pattern finding. Install it with "
                f"`{AUDIT_GROUP_SYNC_COMMAND}` and ensure the resulting executable is on PATH",
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
            trusted_baseline = _trusted_baseline(root, rule.baseline_path)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            errors.append(f"{rule.rule_id}: {exc}")
            continue
        new = matches - baseline
        stale = baseline - matches
        if rule.status == "enforcing" and trusted_baseline is not None:
            baseline_growth = baseline - trusted_baseline
            if baseline_growth:
                errors.append(
                    f"{rule.rule_id}: committed baseline grew: "
                    + ", ".join(_anchor_text(anchor, count) for anchor, count in sorted(baseline_growth.items()))
                )
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
