"""Audit merged ``fix:`` PRs that landed without a corresponding witness.

``tests/witnesses/`` is the durable failure-case corpus for closed bugs.
This command scans recent merged PRs that look like bug fixes and flags
those that modified production source under ``polylogue/`` without
adding (or modifying) a file under ``tests/witnesses/``.

It runs as a *recommended* gate inside ``devtools verify``: a flagged PR
is reported and the command exits with a non-zero return code, but
``devtools verify`` does not treat the audit as required when ``gh`` is
unavailable or no PRs match (exit 0). Operators or nightly jobs that
want a hard gate use the explicit invocation.

Suppression policy lives in
``docs/plans/witness-coverage-suppressions.yaml``: PR numbers listed
there are excluded from the audit (with a short rationale).

Usage::

    devtools verify-witness-coverage
    devtools verify-witness-coverage --days 30
    devtools verify-witness-coverage --json
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from polylogue.core.json import dumps, loads

DEFAULT_WINDOW_DAYS = 90
SUPPRESSIONS_PATH = Path("docs/plans/witness-coverage-suppressions.yaml")
WITNESSES_PREFIX = "tests/witnesses/"
SOURCE_PREFIX = "polylogue/"
FIX_SUBJECT_PREFIXES = ("fix:", "fix(")
BUG_LABEL = "bug"


@dataclass(frozen=True, slots=True)
class FlaggedPR:
    number: int
    title: str
    merged_at: str
    url: str
    source_files: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "merged_at": self.merged_at,
            "url": self.url,
            "source_files": list(self.source_files),
        }


@dataclass(frozen=True, slots=True)
class AuditResult:
    examined: int
    flagged: tuple[FlaggedPR, ...]
    suppressed: tuple[int, ...]
    window_days: int
    skipped_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_days": self.window_days,
            "examined": self.examined,
            "flagged": [pr.as_dict() for pr in self.flagged],
            "suppressed": list(self.suppressed),
            "skipped_reason": self.skipped_reason,
        }


def load_suppressions(path: Path = SUPPRESSIONS_PATH) -> set[int]:
    """Return the set of PR numbers that opt out of the audit."""
    if not path.exists():
        return set()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return set()
    suppressions = raw.get("suppressions", []) if isinstance(raw, dict) else []
    out: set[int] = set()
    for item in suppressions:
        if isinstance(item, dict) and isinstance(item.get("pr"), int):
            out.add(item["pr"])
        elif isinstance(item, int):
            out.add(item)
    return out


def _is_fix_pr(title: str, labels: Sequence[str]) -> bool:
    lowered = title.lstrip().lower()
    if any(lowered.startswith(prefix) for prefix in FIX_SUBJECT_PREFIXES):
        return True
    return BUG_LABEL in {label.lower() for label in labels}


def _touches_source(files: Iterable[str]) -> tuple[bool, tuple[str, ...]]:
    matches = tuple(f for f in files if f.startswith(SOURCE_PREFIX))
    return bool(matches), matches


def _adds_witness(files: Iterable[str]) -> bool:
    return any(f.startswith(WITNESSES_PREFIX) for f in files)


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _run_gh(args: Sequence[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def fetch_merged_prs(*, since_iso: str, limit: int = 200) -> list[dict[str, Any]]:
    """Fetch merged PRs newer than ``since_iso`` via ``gh pr list``.

    Returns a list of PR dicts each containing at least ``number``,
    ``title``, ``mergedAt``, ``url``, ``labels`` and ``files``.
    """
    search = f"is:pr is:merged merged:>={since_iso[:10]}"
    rc, stdout, stderr = _run_gh(
        [
            "pr",
            "list",
            "--state",
            "merged",
            "--limit",
            str(limit),
            "--search",
            search,
            "--json",
            "number,title,mergedAt,url,labels,files",
        ]
    )
    if rc != 0:
        raise RuntimeError(f"gh pr list failed (rc={rc}): {stderr.strip()}")
    data: object = loads(stdout) if stdout.strip() else []
    if not isinstance(data, list):
        raise RuntimeError("unexpected gh pr list payload (not a list)")
    prs: list[dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, dict):
            prs.append(entry)
    return prs


def _extract_files(pr: dict[str, Any]) -> list[str]:
    files = pr.get("files") or []
    out: list[str] = []
    for entry in files:
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            out.append(entry["path"])
    return out


def _extract_labels(pr: dict[str, Any]) -> list[str]:
    labels = pr.get("labels") or []
    out: list[str] = []
    for entry in labels:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            out.append(entry["name"])
        elif isinstance(entry, str):
            out.append(entry)
    return out


def audit_prs(
    prs: Iterable[dict[str, Any]],
    *,
    suppressions: set[int],
    window_days: int,
) -> AuditResult:
    """Apply the witness-coverage heuristic to ``prs``."""
    flagged: list[FlaggedPR] = []
    examined = 0
    suppressed_hits: list[int] = []
    for pr in prs:
        number = pr.get("number")
        if not isinstance(number, int):
            continue
        title = str(pr.get("title") or "")
        labels = _extract_labels(pr)
        if not _is_fix_pr(title, labels):
            continue
        if number in suppressions:
            suppressed_hits.append(number)
            continue
        examined += 1
        files = _extract_files(pr)
        touches_src, src_matches = _touches_source(files)
        if not touches_src:
            continue
        if _adds_witness(files):
            continue
        flagged.append(
            FlaggedPR(
                number=number,
                title=title,
                merged_at=str(pr.get("mergedAt") or ""),
                url=str(pr.get("url") or ""),
                source_files=src_matches,
            )
        )
    return AuditResult(
        examined=examined,
        flagged=tuple(flagged),
        suppressed=tuple(sorted(suppressed_hits)),
        window_days=window_days,
    )


def _since_iso(window_days: int, *, now: datetime | None = None) -> str:
    now = now or datetime.now(tz=timezone.utc)
    return (now - timedelta(days=window_days)).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit merged fix PRs for missing witnesses.")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"Look-back window in days (default: {DEFAULT_WINDOW_DAYS}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of PRs to request from gh (default: 200).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--soft",
        action="store_true",
        help=(
            "Soft mode: return 0 even when PRs are flagged or gh is missing. "
            "Used when wired into ``devtools verify`` as a recommended gate."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    suppressions = load_suppressions()

    if not _gh_available():
        result = AuditResult(
            examined=0,
            flagged=(),
            suppressed=(),
            window_days=args.days,
            skipped_reason="gh CLI not available",
        )
        _emit(result, use_json=args.json)
        return 0 if args.soft else 1

    since = _since_iso(args.days)
    try:
        prs = fetch_merged_prs(since_iso=since, limit=args.limit)
    except RuntimeError as exc:
        result = AuditResult(
            examined=0,
            flagged=(),
            suppressed=(),
            window_days=args.days,
            skipped_reason=str(exc),
        )
        _emit(result, use_json=args.json)
        return 0 if args.soft else 1

    result = audit_prs(prs, suppressions=suppressions, window_days=args.days)
    _emit(result, use_json=args.json)
    if result.flagged and not args.soft:
        return 1
    return 0


def _emit(result: AuditResult, *, use_json: bool) -> None:
    if use_json:
        print(dumps(result.as_dict()))
        return
    if result.skipped_reason:
        sys.stderr.write(f"verify-witness-coverage: skipped ({result.skipped_reason})\n")
        return
    sys.stderr.write(
        f"verify-witness-coverage: examined {result.examined} fix PRs over the last {result.window_days} days\n"
    )
    if result.suppressed:
        sys.stderr.write(f"  suppressed: {', '.join(f'#{n}' for n in result.suppressed)}\n")
    if not result.flagged:
        sys.stderr.write("  no missing-witness flags\n")
        return
    for pr in result.flagged:
        sys.stderr.write(f"  ⚠ #{pr.number} {pr.title} ({pr.url})\n")
        for path in pr.source_files[:5]:
            sys.stderr.write(f"      {path}\n")
        extra = len(pr.source_files) - 5
        if extra > 0:
            sys.stderr.write(f"      … +{extra} more source files\n")
    sys.stderr.write(
        f"\nAdd a witness under tests/witnesses/ or list the PR in {SUPPRESSIONS_PATH} with a rationale.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
