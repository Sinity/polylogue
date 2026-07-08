"""Verify every registered insight product is rigor-contracted or exempt.

Background
----------

``polylogue insights audit`` (``polylogue.insights.audit``) reports a
per-product rigor profile — evidence/inference/fallback coverage, stale
materialization versions, confidence distribution — for every insight
Polylogue ships. The runner used to iterate only the declared
``RigorContract`` matrix (``polylogue.insights.rigor``), so a product
registered in ``INSIGHT_REGISTRY`` with no contract silently vanished from
the audit instead of appearing as uncovered (9e5.28).

What this lint checks
----------------------

Every name in ``polylogue.insights.registry.INSIGHT_REGISTRY`` must be
either:

1. Covered by a ``RigorContract`` row (``rigor_contract_names()``), or
2. Listed in ``RIGOR_EXEMPT`` with an inline justification (genuinely
   non-number-bearing products only).

A registered insight in neither set fails this check. This is a pure static
check — no archive access — so it runs fast in ``devtools verify --lab``
alongside the schema-versioning policy check.
"""

from __future__ import annotations

import argparse
import json


def _uncovered_insight_names() -> tuple[str, ...]:
    from polylogue.insights.registry import INSIGHT_REGISTRY
    from polylogue.insights.rigor import RIGOR_EXEMPT, rigor_contract_names

    registered = set(INSIGHT_REGISTRY)
    covered = set(rigor_contract_names()) | set(RIGOR_EXEMPT)
    return tuple(sorted(registered - covered))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    uncovered = _uncovered_insight_names()

    if args.json:
        print(json.dumps({"uncovered_insight_names": list(uncovered), "ok": not uncovered}, indent=2))
    elif uncovered:
        print(f"insight rigor honesty: {len(uncovered)} registered insight(s) uncovered")
        for name in uncovered:
            print(f"  {name}")
        print("")
        print(
            "Policy violation: every registered insight needs a RigorContract row "
            "(polylogue/insights/rigor.py _RIGOR_MATRIX) or a justified RIGOR_EXEMPT entry."
        )
    else:
        print("insight rigor honesty: every registered insight is contracted or exempt.")

    return 0 if not uncovered else 1


if __name__ == "__main__":
    raise SystemExit(main())
