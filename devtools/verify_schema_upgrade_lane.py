"""Reject in-place storage schema upgrade helpers.

Background
----------

Polylogue has no storage schema upgrade chain: the runtime knows one
canonical archive shape; a database opened at any other ``user_version``
is rejected, and the operator rebuilds from source.

What this lint checks
---------------------

1. Scan ``polylogue/storage/sqlite/`` for upgrade-shaped helpers
   (``build_vN_to_vM``, ``_apply_version_upgrade_plan``, ``migrate_v*``,
   etc.). The names match the historical Polylogue conventions called
   out in ``docs/internals.md`` and in the witness archive
   (``.local/witnesses/new/*schema_upgrades*``).

2. Fail if any helper exists. A schema-shape change edits the canonical
   DDL and requires a fresh archive rebuild, not an in-place patch.

The lint is intentionally narrow. It detects helper names associated
with in-place upgrades; it does not try to infer arbitrary SQL patches.

Wired into ``devtools verify --lab`` rather than the fast default path
because the policy boundary is a lab/architectural concern, not a
per-edit gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root as _get_root

ROOT = _get_root()
STORAGE_SQLITE_DIR = ROOT / "polylogue" / "storage" / "sqlite"

# Upgrade-shaped helper name patterns. Matched against ``def <name>``
# at the top level of any module under ``polylogue/storage/sqlite/``.
#
# Patterns are derived from the historical naming used by upgrade
# helpers that have since been removed (preserved in the witness
# archive under ``.local/witnesses/new/*schema_upgrades*``) and from
# the ``build_vN_to_vM`` / ``_apply_version_upgrade_plan`` naming
# called out as the policy-violating shape in ``docs/internals.md``.
_HELPER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^build_v\d+_to_v\d+$"),
    re.compile(r"^_?apply_version_upgrade(_plan)?$"),
    re.compile(r"^_?upgrade_v\d+_to_v\d+$"),
    re.compile(r"^_?migrate_v\d+(_to_v\d+)?$"),
    re.compile(r"^ensure_schema_upgrades_v\d+$"),
)


@dataclass(frozen=True, slots=True)
class HelperHit:
    name: str
    path: Path
    lineno: int


def _is_helper_name(name: str) -> bool:
    return any(pattern.match(name) for pattern in _HELPER_PATTERNS)


def _collect_upgrade_helpers() -> list[HelperHit]:
    """Return every upgrade-shaped helper defined under storage/sqlite/."""
    hits: list[HelperHit] = []
    if not STORAGE_SQLITE_DIR.exists():
        return hits
    for path in sorted(STORAGE_SQLITE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_helper_name(node.name):
                hits.append(HelperHit(name=node.name, path=path, lineno=node.lineno))
    return hits


def _format_report(*, helpers: list[HelperHit]) -> str:
    lines = [
        f"storage upgrade helpers found: {len(helpers)}",
    ]
    if helpers:
        lines.append("")
        lines.append("Discovered upgrade helpers:")
        for hit in helpers:
            rel = hit.path.relative_to(ROOT)
            lines.append(f"  {rel}:{hit.lineno} def {hit.name}")
        lines.append("")
        lines.append("Policy violation: in-place storage schema upgrades are not supported.")
    if not helpers:
        lines.append("")
        lines.append("Storage schema upgrade policy intact.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    helpers = _collect_upgrade_helpers()

    if args.json:
        payload = {
            "upgrade_helpers": [
                {"name": hit.name, "path": str(hit.path.relative_to(ROOT)), "line": hit.lineno} for hit in helpers
            ],
            "ok": not helpers,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(helpers=helpers))

    return 0 if not helpers else 1


if __name__ == "__main__":
    sys.exit(main())
