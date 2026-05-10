#!/usr/bin/env python3
"""Verify shipped provider schemas contain semantic role annotations.

Checks that every provider with committed schemas under
``polylogue/schemas/providers/`` has at least one element schema annotated
with ``x-polylogue-semantic-role``.

Exit 0 when all schemas have annotations, 1 otherwise.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from devtools import repo_root as _get_root

PROVIDERS_DIR = _get_root() / "polylogue" / "schemas" / "providers"


def check_provider(provider_dir: Path) -> tuple[str, bool, list[str]]:
    """Check one provider directory. Returns (name, ok, failures)."""
    name = provider_dir.name
    versions_dir = provider_dir / "versions"
    if not versions_dir.exists():
        return name, False, ["no versions directory"]

    failures: list[str] = []
    annotation_count = 0
    for version_dir in sorted(versions_dir.iterdir()):
        if not version_dir.is_dir():
            continue
        elements_dir = version_dir / "elements"
        if not elements_dir.exists():
            continue
        for schema_file in sorted(elements_dir.iterdir()):
            if schema_file.suffix not in (".json", ".gz"):
                continue
            try:
                if schema_file.suffix == ".gz":
                    with gzip.open(schema_file, "rt", encoding="utf-8") as fh:
                        data = json.load(fh)
                else:
                    data = json.loads(schema_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, gzip.BadGzipFile) as exc:
                failures.append(f"{version_dir.name}/{schema_file.name}: {exc}")
                continue
            if _has_semantic_annotations(data):
                annotation_count += 1

    if annotation_count == 0:
        failures.append("no elements have x-polylogue-semantic-role annotations")
    return name, len(failures) == 0, failures


def _has_semantic_annotations(data: object) -> bool:
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "x-polylogue-semantic-role":
                return True
            if _has_semantic_annotations(value):
                return True
    elif isinstance(data, list):
        for item in data:
            if _has_semantic_annotations(item):
                return True
    return False


def main() -> int:
    if not PROVIDERS_DIR.exists():
        print(f"providers directory not found: {PROVIDERS_DIR}")
        return 1

    providers = [d for d in sorted(PROVIDERS_DIR.iterdir()) if d.is_dir()]
    if not providers:
        print("no provider directories found")
        return 1

    ok_count = 0
    total_annotations = 0
    all_ok = True
    for provider_dir in providers:
        if provider_dir.name.startswith("_"):
            continue
        name, ok, failures = check_provider(provider_dir)
        if ok:
            ok_count += 1
        else:
            all_ok = False
            for failure in failures:
                print(f"FAIL [{name}]: {failure}")
        # Count annotations
        for version_dir in sorted((provider_dir / "versions").iterdir()):
            if not version_dir.is_dir():
                continue
            elements_dir = version_dir / "elements"
            if not elements_dir.exists():
                continue
            for schema_file in sorted(elements_dir.iterdir()):
                if schema_file.suffix not in (".json", ".gz"):
                    continue
                try:
                    if schema_file.suffix == ".gz":
                        with gzip.open(schema_file, "rt", encoding="utf-8") as fh:
                            data = json.load(fh)
                    else:
                        data = json.loads(schema_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                total_annotations += _count_annotations(data)

    print(f"Verified {len(providers)} providers: {ok_count} OK, {total_annotations} annotations total")
    return 0 if all_ok else 1


def _count_annotations(data: object) -> int:
    count = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "x-polylogue-semantic-role":
                count += 1
            count += _count_annotations(value)
    elif isinstance(data, list):
        for item in data:
            count += _count_annotations(item)
    return count


if __name__ == "__main__":
    raise SystemExit(main())
