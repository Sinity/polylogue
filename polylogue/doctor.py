from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DoctorIssue:
    provider: str
    path: Path
    message: str
    severity: str = "error"


@dataclass
class DoctorReport:
    checked: Dict[str, int]
    issues: List[DoctorIssue]


def _check_jsonl(path: Path, max_lines: int = 3) -> Optional[str]:
    try:
        with path.open(encoding="utf-8") as handle:
            for _ in range(max_lines):
                line = handle.readline()
                if not line.strip():
                    continue
                json.loads(line)
                break
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"
    except Exception as exc:  # pragma: no cover
        return str(exc)
    return None


def inspect_codex(base_dir: Path, limit: Optional[int]) -> DoctorReport:
    issues: List[DoctorIssue] = []
    checked = 0
    base_dir = base_dir.expanduser()
    if not base_dir.exists():
        issues.append(DoctorIssue("codex", base_dir, "Codex sessions directory missing", "warning"))
        return DoctorReport({"codex": 0}, issues)
    for path in sorted(base_dir.rglob("*.jsonl")):
        if limit is not None and checked >= limit:
            break
        checked += 1
        error = _check_jsonl(path)
        if error:
            issues.append(DoctorIssue("codex", path, error))
    return DoctorReport({"codex": checked}, issues)


def inspect_claude_code(base_dir: Path, limit: Optional[int]) -> DoctorReport:
    issues: List[DoctorIssue] = []
    checked = 0
    base_dir = base_dir.expanduser()
    if not base_dir.exists():
        issues.append(DoctorIssue("claude-code", base_dir, "Claude Code projects directory missing", "warning"))
        return DoctorReport({"claude-code": 0}, issues)
    for path in sorted(base_dir.rglob("*.jsonl")):
        if limit is not None and checked >= limit:
            break
        checked += 1
        error = _check_jsonl(path)
        if error:
            issues.append(DoctorIssue("claude-code", path, error))
    return DoctorReport({"claude-code": checked}, issues)


def run_doctor(
    *,
    codex_dir: Path,
    claude_code_dir: Path,
    limit: Optional[int] = None,
) -> DoctorReport:
    codex_report = inspect_codex(codex_dir, limit)
    claude_report = inspect_claude_code(claude_code_dir, limit)
    issues = codex_report.issues + claude_report.issues
    counts = {**codex_report.checked, **claude_report.checked}
    return DoctorReport(checked=counts, issues=issues)
