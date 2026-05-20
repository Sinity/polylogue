"""Verify the in-place schema-upgrade policy boundary (#1302).

Background
----------

Polylogue intentionally has no schema migration chain. The runtime knows
one canonical schema shape; a database opened at any other ``user_version``
is rejected, and the operator re-ingests from source (see
``docs/internals.md`` § "Schema Versioning Model" and
``CONTRIBUTING.md`` § "Schema-Touching Changes"). The architectural
trade is a simpler runtime in exchange for forbidding silent in-place
upgrades.

That policy only holds if any future exception — a reviewed in-place
upgrade helper added under a documented PR exception — is exercised by
a dedicated test lane against a recorded source-version fixture DB.
Without that, a real production archive could be upgraded by an
untested helper, corrupt data silently, and lose history. That is the
exact failure mode fresh-first was designed to make safe.

What this lint checks
---------------------

1. Scan ``polylogue/storage/sqlite/`` for migration-shaped helpers
   (``build_vN_to_vM``, ``_apply_version_upgrade_plan``, ``migrate_v*``,
   etc.). The names match the historical Polylogue conventions called
   out in ``docs/internals.md`` and in the witness archive
   (``.local/witnesses/new/*schema_upgrades*``).

2. If any are found, require a corresponding driving test under
   ``tests/unit/storage/migrations/`` whose source text references each
   helper symbol by name. The test surface is the contract: every
   upgrade helper must appear in at least one migration test that
   constructs a source-version fixture, applies the helper, and asserts
   the post-upgrade shape.

3. If no upgrade helpers exist — the current and intended steady
   state — the lint passes cleanly. The directory remains in the tree
   as a deliberate verification surface (with a README describing the
   policy) so the lane is discoverable before, not after, someone tries
   to add an upgrade path.

The lint is intentionally narrow. It does **not** require the test
surface to construct fixture databases when no helper exists, and it
does **not** attempt to enforce content-level correctness of an upgrade
helper — only that one cannot be merged without a paired driving test.

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
MIGRATIONS_TEST_DIR = ROOT / "tests" / "unit" / "storage" / "migrations"

# Migration-shaped helper name patterns. Matched against ``def <name>``
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
    """Return every migration-shaped helper defined under storage/sqlite/."""
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


def _collect_migration_test_text() -> str:
    """Concatenate the source of every test under tests/unit/storage/migrations/.

    The lint only needs to verify that each helper name appears in a
    test source file under that directory — that ensures the helper has
    a discoverable, named test driver, not that the test itself is
    correct (pytest covers correctness).
    """
    if not MIGRATIONS_TEST_DIR.exists():
        return ""
    chunks: list[str] = []
    for path in sorted(MIGRATIONS_TEST_DIR.rglob("test_*.py")):
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _format_report(
    *,
    helpers: list[HelperHit],
    missing: list[HelperHit],
    test_dir_present: bool,
) -> str:
    lines = [
        f"upgrade helpers found: {len(helpers)}",
        f"migration test surface present: {test_dir_present}",
        f"helpers without a driving test: {len(missing)}",
    ]
    if helpers:
        lines.append("")
        lines.append("Discovered upgrade helpers:")
        for hit in helpers:
            rel = hit.path.relative_to(ROOT)
            lines.append(f"  {rel}:{hit.lineno} def {hit.name}")
    if missing:
        lines.append("")
        lines.append(
            "Policy violation: in-place upgrade helpers exist without a paired "
            "driving test under tests/unit/storage/migrations/. See "
            "docs/internals.md § 'Schema Versioning Model' and #1302."
        )
        for hit in missing:
            rel = hit.path.relative_to(ROOT)
            lines.append(f"  {hit.name} (defined at {rel}:{hit.lineno}) has no test that references it by name")
    elif not helpers:
        lines.append("")
        lines.append("No in-place upgrade helpers detected — fresh-first policy intact.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    helpers = _collect_upgrade_helpers()
    test_text = _collect_migration_test_text()
    test_dir_present = MIGRATIONS_TEST_DIR.exists()

    missing = [hit for hit in helpers if hit.name not in test_text]

    if args.json:
        payload = {
            "upgrade_helpers": [
                {"name": hit.name, "path": str(hit.path.relative_to(ROOT)), "line": hit.lineno} for hit in helpers
            ],
            "missing_driving_tests": [
                {"name": hit.name, "path": str(hit.path.relative_to(ROOT)), "line": hit.lineno} for hit in missing
            ],
            "migrations_test_dir": str(MIGRATIONS_TEST_DIR.relative_to(ROOT)),
            "migrations_test_dir_present": test_dir_present,
            "ok": not missing,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(helpers=helpers, missing=missing, test_dir_present=test_dir_present))

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
