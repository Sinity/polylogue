"""Fail-closed per-diff checks for newly introduced production surfaces."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root
from devtools.production_reachability import (
    _CallGraph,
    _module_name,
    _production_modules,
    _source_signature,
)


class ConsumerReachabilityError(ValueError):
    """A malformed, stale, duplicate, or incomplete authority input."""


@dataclass(frozen=True, slots=True)
class Finding:
    target: str
    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"target": self.target, "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Report:
    base: str
    head: str
    additions: tuple[str, ...]
    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "authority": {"base": self.base, "head": self.head},
            "additions": list(self.additions),
            "findings": [f.to_dict() for f in self.findings],
        }


_SHA = re.compile(r"^[0-9a-f]{40}$")
_WAIVER = re.compile(r"^consumer-reachability-waiver:\s*(\S+)\s+(.+)$", re.I)
_TABLE = re.compile(
    r"\b(?:CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?|ALTER\s+TABLE)\s+[\"'`]?([A-Za-z_][A-Za-z0-9_]*)", re.I
)
_TOOL_DECORATOR = re.compile(r"@(?:[A-Za-z_][\w.]*\.)?(?:tool|command|register_tool)\b")
_FUNCTION = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _authority(root: Path, base: str | None, head: str | None) -> tuple[str, str]:
    actual = _git(root, "rev-parse", "HEAD")
    resolved_head = head or os.environ.get("CONSUMER_REACHABILITY_HEAD") or actual
    resolved_base = base or os.environ.get("CONSUMER_REACHABILITY_BASE")
    if resolved_base is None:
        try:
            resolved_base = _git(root, "merge-base", "HEAD", "origin/master")
        except subprocess.CalledProcessError:
            resolved_base = _git(root, "rev-parse", "HEAD^")
    for label, value in (("base", resolved_base), ("head", resolved_head)):
        if not _SHA.fullmatch(value):
            raise ConsumerReachabilityError(f"malformed authority {label}: expected a 40-character SHA")
        try:
            _git(root, "cat-file", "-e", f"{value}^{{commit}}")
        except subprocess.CalledProcessError as exc:
            raise ConsumerReachabilityError(f"missing authority {label}: {value}") from exc
    if resolved_head != actual:
        raise ConsumerReachabilityError(f"stale authority head: requested {resolved_head}, checkout is {actual}")
    if resolved_base != resolved_head:
        try:
            _git(root, "merge-base", "--is-ancestor", resolved_base, resolved_head)
        except subprocess.CalledProcessError as exc:
            raise ConsumerReachabilityError("authority base is not an ancestor of authority head") from exc
    return resolved_base, resolved_head


def _added_lines(root: Path, base: str, head: str) -> dict[str, list[str]]:
    diff = _git(root, "diff", "--unified=0", "--no-ext-diff", base, head, "--", "polylogue")
    result: dict[str, list[str]] = {}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            result.setdefault(current, [])
        elif current and line.startswith("+") and not line.startswith("+++"):
            result[current].append(line[1:])
    return result


def _waivers(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConsumerReachabilityError(f"cannot read waiver authority: {path}") from exc
    result: dict[str, str] = {}
    for line in lines:
        match = _WAIVER.fullmatch(line.strip())
        if not match:
            continue
        target, reason = match.groups()
        if target in result:
            raise ConsumerReachabilityError(f"duplicate waiver authority for {target}")
        if len(reason.strip()) < 20:
            raise ConsumerReachabilityError(f"waiver reason too short for {target}")
        result[target] = reason.strip()
    return result


def check(root: Path, *, base: str | None = None, head: str | None = None, waiver_body: Path | None = None) -> Report:
    base_sha, head_sha = _authority(root, base, head)
    additions = _added_lines(root, base_sha, head_sha)
    waivers = _waivers(waiver_body)
    production_root = root / "polylogue"
    graph = _CallGraph(_production_modules(root.resolve(), _source_signature(production_root)))
    reachable: set[str] = set()
    for entrypoint in ("polylogue.api", "polylogue.mcp", "polylogue.daemon", "polylogue.hooks"):
        reachable.update(graph.reachable_from(entrypoint))
    findings: list[Finding] = []
    for relative, lines in additions.items():
        path = root / relative
        if path.suffix != ".py" or not path.exists():
            continue
        module = _module_name(path, root)
        if module not in reachable and relative not in waivers:
            findings.append(Finding(relative, "module", "no production entrypoint reaches the added module"))
        for index, line in enumerate(lines):
            if _TOOL_DECORATOR.search(line):
                function_name = next(
                    (match.group(1) for following in lines[index + 1 :] if (match := _FUNCTION.match(following))),
                    None,
                )
                qualified = f"{module}.{function_name}" if function_name else relative
                if qualified not in reachable and relative not in waivers and qualified not in waivers:
                    findings.append(Finding(qualified, "tool", "added tool has no production route consumer"))
            table_match = _TABLE.search(line)
            if not table_match:
                continue
            name = table_match.group(1)
            reader = re.compile(rf"\b(?:FROM|JOIN)\s+[\"'`]?{re.escape(name)}\b", re.I)
            has_reader = any(
                candidate != relative and not candidate.startswith("tests/") and reader.search("\n".join(content))
                for candidate, content in additions.items()
            ) or any(
                candidate != path and reader.search(candidate.read_text(encoding="utf-8"))
                for candidate in production_root.rglob("*.py")
            )
            if not has_reader and name not in waivers:
                findings.append(Finding(name, "table", "added table has no production reader"))
    return Report(base_sha, head_sha, tuple(additions), tuple(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--pr-body", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = check(repo_root(), base=args.base, head=args.head, waiver_body=args.pr_body)
    except (ConsumerReachabilityError, OSError, subprocess.CalledProcessError) as exc:
        payload = {"ok": False, "diagnostic": {"type": type(exc).__name__, "message": str(exc)}}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"consumer-reachability: ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), sort_keys=True))
    elif report.ok:
        print(f"consumer-reachability: ok ({len(report.additions)} changed production file(s))")
    else:
        for finding in report.findings:
            print(f"consumer-reachability: {finding.kind} {finding.target}: {finding.detail}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
