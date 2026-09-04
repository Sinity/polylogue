"""Verify canonical SQLite schema manifests against archive tier files."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.schema_manifest import SchemaManifest, canonical_schema_manifest, schema_manifest_diff


def _check_tier(tier: ArchiveTier, path: Path | None) -> dict[str, Any]:
    expected = canonical_schema_manifest(tier)
    result: dict[str, Any] = {"tier": tier.value, "version": expected.version, "ok": True}
    if path is None or not path.exists():
        return result
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as conn:
        actual = SchemaManifest.from_connection(conn, tier)
    diff = schema_manifest_diff(expected, actual)
    if actual.version != expected.version:
        diff["version"] = {"expected": expected.version, "actual": actual.version}
    result["ok"] = not any(diff.values())
    if not result["ok"]:
        result["diff"] = diff
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify canonical archive SQLite schema manifests.")
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = []
    for tier in ArchiveTier:
        path = args.archive_root / f"{tier.value}.db" if args.archive_root is not None else None
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
