"""Verify canonical SQLite schema manifests against archive tier files."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, cast

from polylogue.storage.archive_identity import ArchiveLocation, ArchiveTierName
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.schema_manifest import SchemaManifest, canonical_schema_manifest, schema_manifest_diff

_DURABLE_TIERS = (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)
ROOT = Path(__file__).parents[1]
_DDL_RE = re.compile(r"(?m)^\s*(?P<name>[A-Z][A-Z0-9_]*DDL)\s*=\s*(?:f)?\"\"\"")
_VERSION_RE = re.compile(r"(?m)^\s*(?P<name>[A-Z]+_SCHEMA_VERSION)\s*=\s*(?P<value>\d+)\s*$")


def _check_tier(tier: ArchiveTier, path: Path | None) -> dict[str, Any]:
    expected = canonical_schema_manifest(tier)
    result: dict[str, Any] = {"tier": tier.value, "version": expected.version, "ok": True}
    if path is None or not path.exists():
        if path is not None:
            result["ok"] = False
            result["diff"] = {"missing": [str(path)]}
        return result
    readonly_uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(readonly_uri, uri=True) as conn:
        actual = SchemaManifest.from_connection(conn, tier)
    diff = schema_manifest_diff(expected, actual)
    if actual.version != expected.version:
        diff["version"] = {"expected": expected.version, "actual": actual.version}
    result["ok"] = not any(diff.values())
    if not result["ok"]:
        result["diff"] = diff
    return result


def _git_text(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=ROOT).stdout


def _ddl_and_version(text: str, tier: ArchiveTier) -> tuple[str | None, int | None]:
    version_match = _VERSION_RE.search(text)
    version = (
        int(version_match.group("value"))
        if version_match and version_match.group("name") == f"{tier.value.upper()}_SCHEMA_VERSION"
        else None
    )
    ddl_match = next(
        (match for match in _DDL_RE.finditer(text) if match.group("name") == f"{tier.value.upper()}_DDL"), None
    )
    if ddl_match is None:
        return None, version
    end = text.find('"""', ddl_match.end())
    return (text[ddl_match.end() : end] if end >= 0 else None), version


def _durable_ddl_evolution_violations() -> list[str]:
    """Require durable DDL changes to carry a version bump or migration."""
    try:
        base = _git_text("rev-parse", "--verify", "origin/master").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"cannot determine origin/master for durable DDL comparison: {exc}"]
    violations: list[str] = []
    for tier in _DURABLE_TIERS:
        path = f"polylogue/storage/sqlite/archive_tiers/{tier.value}.py"
        try:
            previous = _git_text("show", f"{base}:{path}")
            current = (ROOT / path).read_text(encoding="utf-8")
        except (OSError, subprocess.CalledProcessError) as exc:
            violations.append(f"{tier.value}: cannot compare DDL: {exc}")
            continue
        old_ddl, old_version = _ddl_and_version(previous, tier)
        new_ddl, new_version = _ddl_and_version(current, tier)
        if old_ddl == new_ddl:
            continue
        migration_prefix = f"polylogue/storage/sqlite/migrations/{tier.value}/"
        changed_migrations = _git_text("diff", "--name-only", base, "--", migration_prefix).splitlines()
        has_migration = any(path_name.endswith(".sql") for path_name in changed_migrations)
        if old_version == new_version and not has_migration:
            violations.append(f"{tier.value} DDL changed without a schema-version bump or numbered migration")
    return violations


def _print_evolution_check(violations: list[str], *, as_json: bool) -> int:
    payload = {"kind": "polylogue.durable-schema-evolution", "ok": not violations, "violations": violations}
    if as_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for violation in violations:
            print(f"FAIL: {violation}")
        print("durable-schema-evolution: PASS" if not violations else "durable-schema-evolution: FAIL")
    return 0 if not violations else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify canonical archive SQLite schema manifests.")
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check-evolution", action="store_true")
    args = parser.parse_args(argv)
    if args.check_evolution:
        return _print_evolution_check(_durable_ddl_evolution_violations(), as_json=args.json)
    results = []
    location = ArchiveLocation.resolve(args.archive_root) if args.archive_root is not None else None
    for tier in ArchiveTier:
        path = (
            None
            if location is None
            else location.active_index_path
            if tier is ArchiveTier.INDEX
            else location.configured_tier(cast(ArchiveTierName, tier.value)).configured_path
        )
        results.append(_check_tier(tier, path))
    payload = {"kind": "polylogue.schema-manifest", "ok": all(item["ok"] for item in results), "tiers": results}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for item in results:
            print(f"{item['tier']}: {'PASS' if item['ok'] else 'FAIL'} (v{item['version']})")
        print("schema-manifest: PASS" if payload["ok"] else "schema-manifest: FAIL")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
