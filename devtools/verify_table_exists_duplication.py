"""Forbid a new duplicate SQLite existence-check helper outside the canonical module.

Background
----------

polylogue-48h found ~25 independently maintained copies of
``_table_exists``/``table_exists``/``_column_exists``/``_index_exists`` (and
their async variants) scattered across ``cli/``, ``daemon/``, ``storage/``,
``sources/``, ``insights/``, and ``operations/`` -- each trivially small and
subtly different (a ``schema=`` kwarg on some, ``type IN (...)`` alternatives
that never actually match anything in ``sqlite_master`` on others). They were
consolidated into ``polylogue.storage.introspection`` (``table_exists``,
``table_exists_async``, ``column_exists``, ``column_exists_async``,
``index_exists``, ``index_exists_async``). This grep-based tripwire keeps the
consolidation from silently regrowing: a module that wants a table/column/
index existence check should import from ``polylogue.storage.introspection``,
not redefine its own.

What this lint checks
----------------------

Every ``polylogue/**/*.py`` file except ``polylogue/storage/introspection.py``
itself is scanned line-by-line for a top-level (column 0) ``def``/``async def``
whose name matches the forbidden shape:

* ``_table_exists`` / ``table_exists`` (+ ``_sync``/``_async`` suffix variants)
* ``_column_exists`` / ``column_exists`` (+ suffix variants)
* ``_index_exists`` / ``index_exists`` (+ suffix variants)

A thin, behaviorally-distinct wrapper that *delegates* to the canonical
module (e.g. one that also swallows a specific ``sqlite3.OperationalError``,
or checks an ATTACHed schema alias that may not exist yet) is not itself
flagged by name matching alone -- this lint only catches the exact duplicate
*names*, on the theory that a genuinely new name (``_attached_table_exists``,
``_named_table_exists_sync``, ``_schema_object_exists``) signals a real design
choice made under review, while reusing one of the exact retired names is the
easy way to silently reintroduce the duplication this bead removed.

Wired into ``devtools verify --quick`` (the static/generated-surface gate,
alongside the other ``lab policy`` checks): archive-independent, sub-second.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

from devtools import repo_root as _get_root

ROOT = _get_root()

# The one place these names are allowed to be defined.
CANONICAL_MODULE = "polylogue/storage/introspection.py"

_FORBIDDEN_BASE_NAMES = ("table_exists", "column_exists", "index_exists")
_SUFFIXES = ("", "_sync", "_async")

_FORBIDDEN_NAMES = frozenset(
    f"{prefix}{base}{suffix}" for prefix in ("", "_") for base in _FORBIDDEN_BASE_NAMES for suffix in _SUFFIXES
)

_DEF_PATTERN = re.compile(r"^(?:async\s+)?def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(frozen=True, slots=True)
class DuplicationViolation:
    path: str
    lineno: int
    name: str


def scan_source_for_duplicate_definitions(source: str, *, path: str) -> list[DuplicationViolation]:
    """Return every forbidden-named top-level def in *source*.

    Exposed standalone so a test can feed a synthetic source-string fixture
    directly, mirroring ``verify_raw_payload_hash_purity.scan_source_for_payload_concatenation``.
    """
    violations: list[DuplicationViolation] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        match = _DEF_PATTERN.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name in _FORBIDDEN_NAMES:
            violations.append(DuplicationViolation(path=path, lineno=lineno, name=name))
    return violations


def _collect_violations() -> list[DuplicationViolation]:
    violations: list[DuplicationViolation] = []
    for full_path in sorted((ROOT / "polylogue").rglob("*.py")):
        rel = full_path.relative_to(ROOT).as_posix()
        if rel == CANONICAL_MODULE:
            continue
        source = full_path.read_text(encoding="utf-8")
        violations.extend(scan_source_for_duplicate_definitions(source, path=rel))
    return violations


def _format_report(violations: list[DuplicationViolation]) -> str:
    if not violations:
        return (
            "Table/column/index existence-check consolidation intact: no module outside "
            f"{CANONICAL_MODULE} redefines table_exists/column_exists/index_exists (polylogue-48h)."
        )
    lines = [f"SQLite existence-check duplication violations: {len(violations)}", ""]
    for violation in violations:
        lines.append(f"  {violation.path}:{violation.lineno}: def {violation.name}(...)")
    lines.append("")
    lines.append(
        "Policy violation (polylogue-48h): table/column/index existence checks are "
        f"centralized in {CANONICAL_MODULE} (table_exists, table_exists_async, column_exists, "
        "column_exists_async, index_exists, index_exists_async). Import from there instead of "
        "redefining one of these names. If you genuinely need different error-handling or "
        "schema-quoting behavior, write a differently-named thin wrapper that delegates to the "
        "canonical function (see polylogue/storage/usage.py's _table_exists_in_schema for the pattern)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    violations = _collect_violations()

    if args.json:
        payload = {
            "violations": [{"path": v.path, "lineno": v.lineno, "name": v.name} for v in violations],
            "canonical_module": CANONICAL_MODULE,
            "ok": not violations,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(violations))

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
