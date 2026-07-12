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


def _missing_numeric_field_coverage() -> tuple[tuple[str, tuple[str, ...]], ...]:
    from polylogue.insights.rigor import missing_numeric_field_coverage

    return missing_numeric_field_coverage()


def _missing_numeric_item_models() -> tuple[str, ...]:
    from polylogue.insights.rigor import missing_numeric_item_models

    return missing_numeric_item_models()


def _invalid_nullable_field_contracts() -> tuple[tuple[str, tuple[str, ...]], ...]:
    from polylogue.insights.rigor import invalid_nullable_field_contracts

    return invalid_nullable_field_contracts()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    uncovered = _uncovered_insight_names()
    missing_fields = _missing_numeric_field_coverage()
    missing_item_models = _missing_numeric_item_models()
    invalid_contracts = _invalid_nullable_field_contracts()

    if args.json:
        print(
            json.dumps(
                {
                    "uncovered_insight_names": list(uncovered),
                    "missing_numeric_field_coverage": [
                        {"insight_name": name, "field_path": list(field_path)} for name, field_path in missing_fields
                    ],
                    "missing_numeric_item_models": list(missing_item_models),
                    "invalid_nullable_field_contracts": [
                        {"insight_name": name, "field_path": list(field_path)} for name, field_path in invalid_contracts
                    ],
                    "ok": not uncovered and not missing_fields and not missing_item_models and not invalid_contracts,
                },
                indent=2,
            )
        )
    elif uncovered or missing_fields or missing_item_models or invalid_contracts:
        if uncovered:
            print(f"insight rigor honesty: {len(uncovered)} registered insight(s) uncovered")
            for name in uncovered:
                print(f"  {name}")
        if missing_fields:
            print(f"insight rigor honesty: {len(missing_fields)} numeric field coverage declaration(s) missing")
            for name, field_path in missing_fields:
                print(f"  {name}.{'.'.join(field_path)}")
        if missing_item_models:
            print(f"insight rigor honesty: {len(missing_item_models)} registered insight item model(s) missing")
            for name in missing_item_models:
                print(f"  {name}")
        if invalid_contracts:
            print(f"insight rigor honesty: {len(invalid_contracts)} nullable field contract(s) invalid")
            for name, field_path in invalid_contracts:
                print(f"  {name}.{'.'.join(field_path)}")
        print("")
        print(
            "Policy violation: every registered insight needs a RigorContract row "
            "(polylogue/insights/rigor.py _RIGOR_MATRIX) or a justified RIGOR_EXEMPT entry, "
            "and every public numeric field needs a RigorFieldContract or explicit field exemption."
        )
    else:
        print("insight rigor honesty: every registered insight and public numeric field is contracted or exempt.")

    return 0 if not uncovered and not missing_fields and not missing_item_models and not invalid_contracts else 1


if __name__ == "__main__":
    raise SystemExit(main())
