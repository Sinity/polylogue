"""Enforce the suppression registry expiry dates.

Every suppression in ``docs/plans/suppressions.yaml`` must have an expiry
date. The lint fails when any suppression is past its expiry date, forcing
review and either renewal or removal.

See `#518 <https://github.com/Sinity/polylogue/issues/518>`_.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polylogue.proof.suppressions import load_suppressions, validate_suppressions

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "plans" / "suppressions.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yaml", type=Path, default=REGISTRY)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    suppressions = load_suppressions(registry=args.yaml)
    errors = validate_suppressions(suppressions)
    blocking = bool(errors)

    if args.json:
        json.dump(
            {
                "blocking": blocking,
                "expired": errors,
                "total": len(suppressions),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        if errors:
            for error in errors:
                print(f"[BLOCK] {error}")
        else:
            print(f"verify-suppressions: all {len(suppressions)} suppressions current")
        print()
        print(f"blocking={blocking}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
