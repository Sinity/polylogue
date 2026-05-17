"""Verify docs/plans/test-closure-matrix.yaml stays grounded in the realized tree.

For every row in the closure matrix, this lint checks:

* each ``target_files`` path exists (file or directory),
* each ``representative_tests`` entry exists (file path or the file portion of
  a ``path::Class::test`` nodeid),
* every ``gate: absent`` row carries at least one ``known_gaps`` bullet,
* every ``gate: required`` / ``gate: optional`` row lists at least one
  representative test,
* ``gate`` is one of ``required | optional | absent``,
* ``domain`` values are unique.

Wired into ``devtools verify`` so that closure-matrix drift fails locally
before a PR is opened. The matrix is the executable backstop for
``docs/plans/test-coverage-domains.yaml`` (qualitative, free-text) and for the
per-domain coverage callouts in issue #997.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from devtools import repo_root as _get_root

ROOT = _get_root()
MANIFEST = ROOT / "docs" / "plans" / "test-closure-matrix.yaml"

_VALID_GATES = frozenset({"required", "optional", "absent"})


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping at top level, got {type(data).__name__}")
    return data


def _split_nodeid(entry: str) -> str:
    """Return the file portion of a ``path::Class::test`` nodeid."""
    return entry.split("::", 1)[0]


def _path_exists(rel: str) -> bool:
    """Accepts a file or a directory (trailing / optional)."""
    candidate = ROOT / rel.rstrip("/")
    return candidate.exists()


def _validate_row(row: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    prefix = f"row[{index}]"

    domain = row.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        errors.append(f"{prefix}: 'domain' must be a non-empty string")
        domain = f"<unnamed:{index}>"
    prefix = f"row[{index}] domain={domain!r}"

    gate = row.get("gate")
    if gate not in _VALID_GATES:
        errors.append(f"{prefix}: 'gate' must be one of {sorted(_VALID_GATES)}, got {gate!r}")

    target_files = row.get("target_files") or []
    if not isinstance(target_files, list) or not target_files:
        errors.append(f"{prefix}: 'target_files' must be a non-empty list")
        target_files = []
    for entry in target_files:
        if not isinstance(entry, str):
            errors.append(f"{prefix}: target_files entry not a string: {entry!r}")
            continue
        if not _path_exists(entry):
            errors.append(f"{prefix}: target_files path missing: {entry}")

    representative_tests = row.get("representative_tests") or []
    if not isinstance(representative_tests, list):
        errors.append(f"{prefix}: 'representative_tests' must be a list")
        representative_tests = []
    for entry in representative_tests:
        if not isinstance(entry, str):
            errors.append(f"{prefix}: representative_tests entry not a string: {entry!r}")
            continue
        file_part = _split_nodeid(entry)
        if not _path_exists(file_part):
            errors.append(f"{prefix}: representative_tests path missing: {entry}")

    known_gaps = row.get("known_gaps") or []
    if not isinstance(known_gaps, list):
        errors.append(f"{prefix}: 'known_gaps' must be a list when present")
        known_gaps = []

    if gate == "absent":
        if not known_gaps:
            errors.append(f"{prefix}: gate='absent' rows must list at least one 'known_gaps' bullet")
    elif gate in ("required", "optional") and not representative_tests:
        errors.append(f"{prefix}: gate={gate!r} rows must list at least one representative test")

    return errors


def _validate(matrix: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    rows = matrix.get("rows")
    if not isinstance(rows, list) or not rows:
        return [f"{MANIFEST.name}: 'rows' must be a non-empty list"]

    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row[{index}]: must be a mapping")
            continue
        errors.extend(_validate_row(row, index))
        domain = row.get("domain")
        if isinstance(domain, str):
            if domain in seen:
                errors.append(f"duplicate domain: {domain!r}")
            seen.add(domain)

    deprecated = matrix.get("deprecated_pointer")
    if deprecated is not None:
        if not isinstance(deprecated, dict):
            errors.append("'deprecated_pointer' must be a mapping when present")
        else:
            legacy = deprecated.get("legacy_manifest")
            if isinstance(legacy, str) and not _path_exists(legacy):
                errors.append(f"deprecated_pointer.legacy_manifest missing: {legacy}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=MANIFEST)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        matrix = _load(args.yaml)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        if args.json:
            json.dump({"blocking": True, "errors": [str(exc)]}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"[BLOCK] failed to load {args.yaml}: {exc}")
        return 1

    errors = _validate(matrix)
    blocking = bool(errors)
    row_count = len(matrix.get("rows") or [])

    if args.json:
        json.dump(
            {"blocking": blocking, "errors": errors, "rows": row_count},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if errors:
            print(f"[BLOCK] closure-matrix errors: {len(errors)}")
            for line in errors:
                print(f"    {line}")
        else:
            print(f"closure-matrix: clean ({row_count} domains)")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
