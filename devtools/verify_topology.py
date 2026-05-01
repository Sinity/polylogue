"""Verify the realized polylogue/ tree against the topology projection.

Reports:

  * orphans         — file in tree, not declared in YAML
  * missing         — file declared in YAML, not in tree
  * conflicts       — same path declared twice
  * kernel-rule     — root file not in the declared kernel set
  * tbd             — projection cells still marked TBD

Exits 0 if everything passes, 1 if any blocking finding, 2 if only warnings.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT / "docs" / "plans" / "topology-target.yaml"

KERNEL_OWNERS = frozenset({"kernel", "lib-root", "storage-root", "stable"})


def parse_yaml(text: str) -> list[dict[str, Any]]:
    """Tiny YAML parser for our flat list-of-records shape.

    Avoids a PyYAML dep so this can run in any environment. Format is fixed
    by `build_projection.py`.
    """
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line == "files:":
            continue
        if line.startswith("  - path: "):
            if current is not None:
                rows.append(current)
            current = {"path": line[len("  - path: ") :].strip(), "cross_cut": {}}
            continue
        if current is None:
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent != 4:
            continue
        if ": " in stripped:
            key, _, value = stripped.partition(": ")
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1].replace('\\"', '"')
            if key == "loc":
                try:
                    current[key] = int(value)
                except ValueError:
                    current[key] = 0
            elif key == "cross_cut":
                # form: { layer: read, lifecycle: model }
                inner = value.strip("{ }")
                tags = {}
                if inner:
                    for pair in inner.split(", "):
                        k, _, v = pair.partition(": ")
                        tags[k.strip()] = v.strip()
                current[key] = tags
            else:
                current[key] = value
    if current is not None:
        rows.append(current)
    return rows


def walk_tree() -> set[str]:
    return {p.relative_to(ROOT).as_posix() for p in ROOT.glob("polylogue/**/*.py") if "__pycache__" not in p.parts}


def is_root_polylogue(path: str) -> bool:
    if not path.startswith("polylogue/"):
        return False
    suffix = path[len("polylogue/") :]
    return "/" not in suffix


def main(argv: Iterable[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=PROJECTION)
    p.add_argument("--strict-tbd", action="store_true", help="Treat TBD entries as blocking.")
    p.add_argument("--json", action="store_true", help="Emit JSON for tooling.")
    args = p.parse_args(list(argv))

    text = args.yaml.read_text()
    rows = parse_yaml(text)
    declared_paths = [row["path"] for row in rows]

    findings: dict[str, list[str]] = {
        "orphans": [],
        "missing": [],
        "conflicts": [],
        "kernel_rule": [],
        "tbd": [],
    }

    seen: dict[str, int] = {}
    for row in rows:
        seen[row["path"]] = seen.get(row["path"], 0) + 1
        if row.get("target") == "TBD":
            findings["tbd"].append(row["path"])

    findings["conflicts"] = sorted(p for p, c in seen.items() if c > 1)

    realized = walk_tree()
    declared = set(declared_paths)

    findings["orphans"] = sorted(realized - declared)
    findings["missing"] = sorted(declared - realized)

    # Kernel rule: any polylogue/<file>.py whose target is itself but owner is NOT kernel
    # is a hint that the rule is loose. Stricter: all root files that are *not* moving
    # should be declared kernel.
    for row in rows:
        path = row["path"]
        if not is_root_polylogue(path):
            continue
        target = row.get("target", "")
        owner = row.get("owner", "")
        if target == path and owner not in KERNEL_OWNERS:
            findings["kernel_rule"].append(f"{path} stays at root but owner={owner!r}")

    blocking = findings["orphans"] or findings["conflicts"] or findings["missing"] or findings["kernel_rule"]
    warnings = findings["tbd"]
    if args.strict_tbd:
        blocking = blocking or warnings

    if args.json:
        import json

        json.dump(
            {
                "blocking": bool(blocking),
                "counts": {k: len(v) for k, v in findings.items()},
                "findings": findings,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        for kind, items in findings.items():
            if not items:
                continue
            tag = "[BLOCK]" if kind in {"orphans", "missing", "conflicts", "kernel_rule"} else "[warn]"
            print(f"{tag} {kind}: {len(items)}")
            for item in items[:10]:
                print(f"    {item}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")
        print()
        print(f"realized={len(realized)} declared={len(declared)} blocking={bool(blocking)}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
