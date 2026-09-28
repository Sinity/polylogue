"""Report which source files feed the derived schema identity.

``derived_schema_identity`` digests the lowering, materializer and
replay-routing fingerprints, and those are AST closures over imported source.
Editing any file in the closure moves the identity, which forces a full
reconvergence and, under a running rebuild, invalidates it. Membership follows
the import graph rather than directory boundaries, so this command exists to be
asked instead of guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report whether a file feeds the derived schema identity, or list the whole closure."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files to classify. With no paths, list every closure member.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from polylogue.sources.origin_specs import derived_identity_source_closure

    root = _repo_root()
    closure = derived_identity_source_closure()

    if not args.paths:
        members = sorted(_relative(member, root) for member in closure)
        if args.json:
            print(json.dumps({"kind": "polylogue.derived-identity-closure", "count": len(members), "members": members}))
        else:
            for member in members:
                print(member)
            print(f"{len(members)} files feed the derived schema identity.")
        return 0

    resolved_closure = {member.resolve() for member in closure}
    results = [
        {"path": str(path), "in_closure": (path if path.is_absolute() else root / path).resolve() in resolved_closure}
        for path in args.paths
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "kind": "polylogue.derived-identity-closure",
                    "closure_size": len(closure),
                    "results": results,
                }
            )
        )
    else:
        for result in results:
            print(f"{'IN ' if result['in_closure'] else 'out'}  {result['path']}")
        if any(result["in_closure"] for result in results):
            print(
                "\nIN means conservative file membership. Semantic edits to a member may move the derived schema identity; "
                "normalized-only edits may not. Confirm the identity before scheduling reconvergence."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
