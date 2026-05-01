"""Verify cross-cut tags in the topology projection.

Reads ``docs/plans/topology-target.yaml`` and asserts that ``cross_cut``
tags on each module match what the module's name suggests:

* a module named ``*_runtime.py`` should be tagged ``lifecycle: runtime``;
* a module named ``*_models.py`` should be tagged ``lifecycle: model``;
* a module under ``polylogue/api/sync/`` should be tagged ``api: sync``;
* a module under ``polylogue/api/`` (outside ``sync/``) should be tagged ``api: async``;
* a module named ``*_reads`` should be tagged ``layer: read``;
* a module named ``*_writes`` or ``*_write_*`` should be tagged ``layer: write``.

The lint catches manual edits to the projection that break tag-naming
consistency. It does NOT enforce architectural rules between tagged
modules (e.g. "no sync module imports an async module") — that is
deferred to a future Phase 2 once the cross-cut tag set is wider.

Phase 1 of #432.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from devtools.verify_topology import parse_yaml as parse_topology_yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT / "docs" / "plans" / "topology-target.yaml"

EXPECTED_LIFECYCLE: dict[str, str] = {
    "_runtime.py": "runtime",
    "_models.py": "model",
}

EXPECTED_LAYER: dict[str, str] = {
    "_reads.py": "read",
    "_reads": "read",
    "_read_": "read",
    "_writes.py": "write",
    "_writes": "write",
    "_write_": "write",
}

EXPECTED_API: dict[str, str] = {}  # path-based after #426; see expected_for()
CONVENTION_TAGS = frozenset(("api", "layer", "lifecycle"))
UI_FACADE_API_PATHS = frozenset(
    (
        "polylogue/ui/facade.py",
        "polylogue/ui/facade_console.py",
        "polylogue/ui/facade_prompts.py",
        "polylogue/ui/facade_rendering.py",
    )
)


def expected_for(name: str, path: str = "") -> dict[str, str]:
    """Return the cross_cut tags the module's filename / path suggests."""
    expected: dict[str, str] = {}
    for suffix, value in EXPECTED_LIFECYCLE.items():
        if name.endswith(suffix):
            expected.setdefault("lifecycle", value)
            break
    for marker, value in EXPECTED_LAYER.items():
        if marker.endswith(".py"):
            if name.endswith(marker):
                expected.setdefault("layer", value)
                break
        else:
            if marker in name:
                expected.setdefault("layer", value)
                break
    # Path-based api tagging after #426: polylogue/api/sync/* → sync, polylogue/api/* → async.
    if path.startswith("polylogue/api/sync/"):
        expected.setdefault("api", "sync")
    elif path.startswith("polylogue/api/") or path in UI_FACADE_API_PATHS:
        expected.setdefault("api", "async")
    return expected


def parse_cross_cut(value: str | dict[str, str] | object) -> dict[str, str]:
    """Cross-cut tags are stored as inline mapping; parse_topology_yaml leaves them
    as string literals like ``{ layer: read }``. Decode if needed."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    if not text:
        return {}
    out: dict[str, str] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            continue
        key, _, val = chunk.partition(":")
        out[key.strip()] = val.strip()
    return out


def filename_of(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=PROJECTION)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    rows = parse_topology_yaml(args.yaml.read_text())
    mismatches: list[dict[str, str]] = []
    missing_tags: list[dict[str, str]] = []
    for row in rows:
        path = str(row.get("path", ""))
        if not path.startswith("polylogue/"):
            continue
        name = filename_of(path)
        actual = parse_cross_cut(row.get("cross_cut", ""))
        expected = expected_for(name, path)
        for key, expected_value in expected.items():
            if key not in actual:
                missing_tags.append({"path": path, "key": key, "expected": expected_value})
            elif actual[key] != expected_value:
                mismatches.append(
                    {
                        "path": path,
                        "key": key,
                        "expected": expected_value,
                        "actual": actual[key],
                    }
                )
        for key, actual_value in actual.items():
            if key in CONVENTION_TAGS and key not in expected:
                mismatches.append(
                    {
                        "path": path,
                        "key": key,
                        "expected": "<absent>",
                        "actual": actual_value,
                    }
                )

    blocking = bool(mismatches)

    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "mismatches": mismatches,
                "missing_tags": missing_tags,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if mismatches:
            print(f"[BLOCK] cross-cut tag mismatches: {len(mismatches)}")
            for entry in mismatches[:10]:
                print(f"    {entry['path']}: {entry['key']}={entry['actual']!r} expected {entry['expected']!r}")
            if len(mismatches) > 10:
                print(f"    ... and {len(mismatches) - 10} more")
        if missing_tags:
            print(f"[warn] modules missing expected cross-cut tags: {len(missing_tags)}")
            for entry in missing_tags[:10]:
                print(f"    {entry['path']}: missing {entry['key']}={entry['expected']!r}")
            if len(missing_tags) > 10:
                print(f"    ... and {len(missing_tags) - 10} more")
        if not mismatches and not missing_tags:
            print("cross-cut tags: clean")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
