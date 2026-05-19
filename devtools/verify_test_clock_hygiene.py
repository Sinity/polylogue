"""Lint: timestamp calls in test files must use the ``frozen_clock`` fixture.

Background: ``tests/infra/frozen_clock.py`` provides a deterministic
``FrozenClock`` and the ``@pytest.mark.frozen_clock_modules(...)`` marker
that pins ``time.time``, ``time.monotonic``, and ``datetime.now`` for the
production modules the test exercises. Tests that bypass the fixture and
read the host wall clock directly (``datetime.now``, ``datetime.utcnow``,
``time.time``, ``time.monotonic``) cannot reproduce edge-cases near
threshold windows and create snapshot/timestamp churn that hides real
regressions (#1300).

This lint:

1. Scans every Python file under ``tests/`` (excluding ``tests/infra/``).
2. Reports calls to ``datetime.now`` / ``datetime.utcnow`` / ``time.time`` /
   ``time.monotonic`` outside the allowlist in
   ``docs/plans/test-clock-allowlist.yaml``.
3. Exits non-zero when any non-allowlisted call is found. Tests that
   genuinely need the host clock add their path to the allowlist with a
   one-line rationale.

The lint is intentionally narrow:

- Only the four time-source calls listed above are flagged. Type hints
  (``-> datetime``), arithmetic against an existing ``now`` value, and
  ``datetime.fromisoformat`` / ``datetime.fromtimestamp`` calls are not
  flagged — they don't read the host clock.
- ``tests/infra/`` is exempt because the fixture and its support live
  there.
- The allowlist is keyed by relative path; a file either is or isn't on
  the list. This keeps the lint behaviour transparent: agents can grep
  the YAML to find what's exempted.

The lint runs in ``devtools verify`` (default tier) so any new
``datetime.now()`` usage in a test file fails the local baseline
immediately, before the change reaches review.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard repo dep
    yaml = None  # type: ignore[assignment]

from devtools import repo_root as _get_root

ROOT = _get_root()
TESTS_DIR = ROOT / "tests"
INFRA_DIR = TESTS_DIR / "infra"
ALLOWLIST_PATH = ROOT / "docs" / "plans" / "test-clock-allowlist.yaml"

# Calls that read the host wall / monotonic clock. ``datetime.fromisoformat``,
# ``datetime.fromtimestamp``, and the like construct deterministic values
# from existing inputs and are explicitly not flagged.
_FORBIDDEN: dict[tuple[str, str], str] = {
    ("datetime", "now"): "datetime.now",
    ("datetime", "utcnow"): "datetime.utcnow",
    ("time", "time"): "time.time",
    ("time", "monotonic"): "time.monotonic",
    ("time", "monotonic_ns"): "time.monotonic_ns",
    ("time", "time_ns"): "time.time_ns",
}


def _load_allowlist() -> dict[str, str]:
    """Return ``{relative_path: rationale}`` from the allowlist YAML."""
    if not ALLOWLIST_PATH.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the test clock allowlist")
    data = yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8")) or {}
    files = data.get("files", []) or []
    out: dict[str, str] = {}
    for entry in files:
        if isinstance(entry, str):
            out[entry] = ""
        elif isinstance(entry, dict):
            path = entry.get("path")
            if isinstance(path, str):
                out[path] = str(entry.get("reason", ""))
    return out


def _iter_forbidden_calls(text: str) -> Iterable[tuple[str, int]]:
    """Yield ``(call_label, lineno)`` for forbidden Attribute-access calls.

    Matches:
        datetime.now(...)
        datetime.utcnow()
        time.time()
        time.monotonic()

    Does not match:
        from datetime import datetime; ... datetime.fromisoformat(...)
        ts: datetime = ...
        datetime.now is used as an attribute reference without call (rare)
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not isinstance(func.value, ast.Name):
            continue
        key = (func.value.id, func.attr)
        label = _FORBIDDEN.get(key)
        if label is not None:
            yield (label, node.lineno)


def _scan_tests() -> dict[Path, list[tuple[str, int]]]:
    """Return ``{test_file: [(call_label, lineno), ...]}``."""
    findings: dict[Path, list[tuple[str, int]]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if INFRA_DIR in path.parents or path == INFRA_DIR:
            continue
        # ``conftest.py`` files are infrastructure-adjacent; checking them
        # would mostly produce false positives (registered fixtures touch
        # the host clock for legitimate reasons such as test isolation).
        if path.name == "conftest.py":
            continue
        hits = list(_iter_forbidden_calls(path.read_text(encoding="utf-8")))
        if hits:
            findings[path] = hits
    return findings


def _format_report(
    *,
    findings: dict[Path, list[tuple[str, int]]],
    allowlist: dict[str, str],
    violations: dict[Path, list[tuple[str, int]]],
) -> str:
    lines = [
        f"test files scanned: {sum(1 for _ in TESTS_DIR.rglob('*.py')) - sum(1 for _ in INFRA_DIR.rglob('*.py'))}",
        f"files with host-clock calls: {len(findings)}",
        f"allowlisted: {len(allowlist)}",
        f"violations: {len(violations)}",
    ]
    if violations:
        lines.append("")
        lines.append("Tests reading the host clock outside the allowlist:")
        for path, hits in sorted(violations.items()):
            rel = path.relative_to(ROOT)
            for label, lineno in hits:
                lines.append(
                    f"  {rel}:{lineno} {label}() — use the ``frozen_clock`` fixture or add to docs/plans/test-clock-allowlist.yaml"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    findings = _scan_tests()
    allowlist = _load_allowlist()
    violations: dict[Path, list[tuple[str, int]]] = {
        path: hits for path, hits in findings.items() if str(path.relative_to(ROOT)) not in allowlist
    }

    if args.json:
        payload = {
            "files_with_host_clock_calls": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "hits": [{"call": label, "line": lineno} for label, lineno in hits],
                    "allowlisted": str(path.relative_to(ROOT)) in allowlist,
                }
                for path, hits in sorted(findings.items())
            ],
            "violations": [str(path.relative_to(ROOT)) for path in sorted(violations)],
            "allowlist": [{"path": path, "reason": reason} for path, reason in sorted(allowlist.items())],
            "ok": not violations,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(findings=findings, allowlist=allowlist, violations=violations))

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
