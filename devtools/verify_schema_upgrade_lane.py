"""Verify schema-evolution policy boundaries.

Background
----------

Polylogue has two schema-evolution regimes:

* Durable tiers (``source.db`` and ``user.db``) may use explicit additive SQL
  migrations with a backup gate.
* Derived/rebuildable tiers (``index.db`` and ``embeddings.db``) do not use
  migration chains. They are rebuilt or blue-green replaced from durable source
  evidence, except for explicitly declared, clone-validated SQL plans.

What this lint checks
---------------------

1. Scan derived-tier storage modules for upgrade-shaped helpers
   (``build_vN_to_vM``, ``_apply_version_upgrade_plan``, ``migrate_v*``,
   etc.). The names match the historical Polylogue conventions called
   out in ``docs/internals.md`` and in the witness archive
   (``.local/witnesses/new/*schema_upgrades*``).

2. Fail if any legacy helper exists for a derived tier, or when the current
   index schema version lacks a delta-class declaration. Durable-tier migrations
   must live under ``polylogue/storage/sqlite/migrations/{source,user}/`` as
   numbered SQL resources.

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
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.lifecycle import IndexDeltaDeclarationReport, index_delta_declaration_report

ROOT = _get_root()
STORAGE_SQLITE_DIR = ROOT / "polylogue" / "storage" / "sqlite"
MIGRATIONS_DIR = STORAGE_SQLITE_DIR / "migrations"
ALLOWED_MIGRATION_TIERS = {"source", "user"}

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
    """Return upgrade-shaped helpers outside the durable migration runner."""
    hits: list[HelperHit] = []
    if not STORAGE_SQLITE_DIR.exists():
        return hits
    for path in sorted(STORAGE_SQLITE_DIR.rglob("*.py")):
        rel_parts = path.relative_to(STORAGE_SQLITE_DIR).parts
        if rel_parts[:1] == ("migrations",) or path.name == "migrations.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_helper_name(node.name):
                hits.append(HelperHit(name=node.name, path=path, lineno=node.lineno))
    return hits


def _invalid_migration_paths() -> list[Path]:
    if not MIGRATIONS_DIR.exists():
        return []
    invalid: list[Path] = []
    for path in sorted(MIGRATIONS_DIR.rglob("*")):
        if path.is_dir() or path.name == "__init__.py":
            continue
        rel = path.relative_to(MIGRATIONS_DIR)
        if (
            len(rel.parts) != 2
            or rel.parts[0] not in ALLOWED_MIGRATION_TIERS
            or not re.match(r"^\d{3,}_[a-z0-9_]+\.sql$", rel.parts[1])
        ):
            invalid.append(path)
    return invalid


def _format_report(
    *, helpers: list[HelperHit], invalid_migrations: list[Path], delta_report: IndexDeltaDeclarationReport
) -> str:
    lines = [
        f"derived-tier upgrade helpers found: {len(helpers)}",
        f"invalid durable migration resources found: {len(invalid_migrations)}",
        f"undeclared index schema deltas found: {len(delta_report['missing_versions'])}",
    ]
    if helpers:
        lines.append("")
        lines.append("Discovered upgrade helpers:")
        for hit in helpers:
            rel = hit.path.relative_to(ROOT)
            lines.append(f"  {rel}:{hit.lineno} def {hit.name}")
        lines.append("")
        lines.append("Policy violation: derived tiers must rebuild or blue-green replace, not migrate in place.")
    if invalid_migrations:
        lines.append("")
        lines.append("Invalid migration resources:")
        for path in invalid_migrations:
            lines.append(f"  {path.relative_to(ROOT)}")
    if not bool(delta_report["ok"]):
        lines.append("")
        lines.append("Index fast-forward declaration drift:")
        lines.append(f"  compatibility floor: v{delta_report['compatibility_floor']}")
        lines.append(f"  missing: {list(delta_report['missing_versions'])}")
        lines.append(f"  duplicate: {list(delta_report['duplicate_versions'])}")
        lines.append(f"  invalid: {list(delta_report['invalid_versions'])}")
        lines.append("")
        lines.append("Policy violation: each index schema bump needs a declared delta class.")
    if not helpers and not invalid_migrations and bool(delta_report["ok"]):
        lines.append("")
        lines.append("Schema evolution policy intact.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    helpers = _collect_upgrade_helpers()
    invalid_migrations = _invalid_migration_paths()
    delta_report = index_delta_declaration_report(INDEX_SCHEMA_VERSION)

    if args.json:
        payload = {
            "upgrade_helpers": [
                {"name": hit.name, "path": str(hit.path.relative_to(ROOT)), "line": hit.lineno} for hit in helpers
            ],
            "invalid_migration_resources": [str(path.relative_to(ROOT)) for path in invalid_migrations],
            "index_delta_declarations": delta_report,
            "ok": not helpers and not invalid_migrations and bool(delta_report["ok"]),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(helpers=helpers, invalid_migrations=invalid_migrations, delta_report=delta_report))

    return 0 if not helpers and not invalid_migrations and bool(delta_report["ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
