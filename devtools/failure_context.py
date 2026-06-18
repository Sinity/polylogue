"""Aggregate pytest failure context for agent inner-loop debugging.

Joins testmon dependency data, git history, fixture references, and committed
witnesses into a single JSON envelope keyed by a pytest failure ID
(e.g. ``tests/unit/storage/test_foo.py::test_bar``).

Usage:
  devtools workspace failure-context tests/unit/storage/test_foo.py::test_bar
  devtools workspace failure-context tests/unit/storage/test_foo.py::test_bar --json
  devtools workspace failure-context tests/unit/storage/test_foo.py::test_bar --days 14
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTMON_DB = REPO_ROOT / ".testmondata"
WITNESS_DIR = REPO_ROOT / "tests" / "witnesses"


def _parse_failure_id(failure_id: str) -> tuple[str, str]:
    if "::" not in failure_id:
        raise ValueError(f"failure id must be 'path::test_name', got {failure_id!r}")
    test_file, _, test_name = failure_id.partition("::")
    return test_file, test_name


def _testmon_dependencies(failure_id: str) -> list[str]:
    if not TESTMON_DB.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{TESTMON_DB}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT f.filename
            FROM file_fp f
            JOIN test_execution_file_fp x ON x.fingerprint_id = f.id
            JOIN test_execution t ON t.id = x.test_execution_id
            WHERE t.test_name = ?
            ORDER BY f.filename
            """,
            (failure_id,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [row[0] for row in rows]


def _recent_changes(files: list[str], days: int) -> dict[str, list[dict[str, str]]]:
    if not files:
        return {}
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    out: dict[str, list[dict[str, str]]] = {}
    for path in files:
        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"--since={since}",
                    "--pretty=format:%h%x09%ad%x09%s",
                    "--date=short",
                    "--",
                    path,
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        commits: list[dict[str, str]] = []
        for line in result.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                commits.append({"sha": parts[0], "date": parts[1], "subject": parts[2]})
        if commits:
            out[path] = commits
    return out


_FIXTURE_RE = re.compile(r"def\s+test_\w+\s*\(([^)]*)\)")
_PARAM_SPLIT = re.compile(r"[,\s]+")


def _related_fixtures(test_file: str, test_name: str) -> list[str]:
    path = REPO_ROOT / test_file
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    base_name = test_name.split("[", 1)[0]
    fixtures: set[str] = set()
    pattern = re.compile(rf"def\s+{re.escape(base_name)}\s*\(([^)]*)\)")
    match = pattern.search(text)
    if match:
        for token in _PARAM_SPLIT.split(match.group(1)):
            token = token.strip()
            if not token or token in {"self", "cls"}:
                continue
            token = token.split(":", 1)[0].strip()
            token = token.split("=", 1)[0].strip()
            if token and token != "*" and not token.startswith("*"):
                fixtures.add(token)
    return sorted(fixtures)


def _similar_witnesses(failure_id: str, limit: int = 5) -> list[dict[str, Any]]:
    if not WITNESS_DIR.exists():
        return []
    test_file, test_name = _parse_failure_id(failure_id)
    base_name = test_name.split("[", 1)[0]
    # Tokenize the failure id into substring keywords for matching.
    tokens = {tok for tok in re.split(r"[\W_/.]+", f"{test_file} {base_name}") if len(tok) > 3}
    matches: list[tuple[int, dict[str, Any]]] = []
    for witness_path in sorted(WITNESS_DIR.glob("*.witness.json")):
        try:
            data = json.loads(witness_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        haystack = json.dumps(data).lower()
        score = sum(1 for tok in tokens if tok.lower() in haystack)
        source_test = data.get("provenance", {}).get("source_test", "")
        if source_test and source_test == test_file:
            score += 10
        if score:
            matches.append(
                (
                    score,
                    {
                        "witness_id": data.get("witness_id"),
                        "path": str(witness_path.relative_to(REPO_ROOT)),
                        "source_test": source_test or None,
                        "score": score,
                    },
                )
            )
    matches.sort(key=lambda item: (-item[0], item[1]["witness_id"] or ""))
    return [entry for _score, entry in matches[:limit]]


def build_envelope(failure_id: str, days: int) -> dict[str, Any]:
    test_file, test_name = _parse_failure_id(failure_id)
    dependencies = _testmon_dependencies(failure_id)
    return {
        "failure_id": failure_id,
        "test_file": test_file,
        "test_name": test_name,
        "testmon_dependencies": dependencies,
        "recent_changes": _recent_changes(dependencies, days),
        "related_fixtures": _related_fixtures(test_file, test_name),
        "similar_witnesses": _similar_witnesses(failure_id),
        "metadata": {
            "days_window": days,
            "testmon_db_present": TESTMON_DB.exists(),
            "witness_dir_present": WITNESS_DIR.exists(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join testmon, git, fixtures, and witnesses for a pytest failure ID.")
    parser.add_argument(
        "failure_id",
        type=str,
        help="Pytest failure ID, e.g. tests/unit/storage/test_foo.py::test_bar",
    )
    parser.add_argument("--days", type=int, default=30, help="Git log window in days (default: 30).")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout (default when stdout is not a TTY).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        envelope = build_envelope(args.failure_id, args.days)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(envelope, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
