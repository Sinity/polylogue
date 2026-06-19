"""Verify that scenario lanes classified as SEMANTIC_OUTPUT carry semantic assertions.

A lane classified as SEMANTIC_OUTPUT without any stdout_contains,
stdout_not_contains, stdout_is_valid_json, or custom assertion is
vacuous — it claims semantic evidence but only checks exit codes.

Exits 0 (clean) when no violations exist, 1 when violations found.
"""

from __future__ import annotations

import sys

from devtools import repo_root as _get_root
from polylogue.scenarios.assertions import AssertionClass, AssertionSpec


def _load_assertion_specs() -> list[tuple[str, AssertionSpec]]:
    """Load assertion specs from the scenario coverage catalog."""
    import yaml

    repo_root = _get_root()
    coverage_path = repo_root / "docs" / "plans" / "scenario-coverage.yaml"
    if not coverage_path.exists():
        print(f"verify lane-assertions: missing {coverage_path}", file=sys.stderr)
        return []

    with open(coverage_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    specs: list[tuple[str, AssertionSpec]] = []
    for entry in doc.get("lanes", []):
        name = entry.get("name", "unnamed")
        assertion_raw = entry.get("assertion") or {}
        specs.append(
            (
                name,
                AssertionSpec(
                    exit_code=assertion_raw.get("exit_code", 0),
                    stdout_contains=tuple(assertion_raw.get("stdout_contains", ())),
                    stdout_not_contains=tuple(assertion_raw.get("stdout_not_contains", ())),
                    stdout_is_valid_json=assertion_raw.get("stdout_is_valid_json", False),
                    stdout_min_lines=assertion_raw.get("stdout_min_lines"),
                    benchmark_warn_pct=assertion_raw.get("benchmark_warn_pct"),
                    benchmark_fail_pct=assertion_raw.get("benchmark_fail_pct"),
                ),
            )
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    specs = _load_assertion_specs()
    violations: list[str] = []

    for name, spec in specs:
        if spec.classification != AssertionClass.SEMANTIC_OUTPUT:
            continue
        warnings = spec.validate()
        if warnings:
            violations.append(f"  {name}: {warnings[0]}")

    if violations:
        print(f"verify lane-assertions: {len(violations)} vacuous SEMANTIC_OUTPUT lane(s):", file=sys.stderr)
        for v in violations:
            print(v, file=sys.stderr)
        return 1

    print(f"verify lane-assertions: ok ({len(specs)} lanes, 0 vacuous)", file=sys.stderr)
    return 0
