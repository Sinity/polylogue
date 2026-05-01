"""Verify internal consistency across all docs/plans/*.yaml manifests.

Ensures manifest files are valid YAML and their cross-references are
consistent. Runs as part of devtools verify.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def load_manifest(path: Path) -> dict[str, object]:
    """Load a YAML manifest and return its parsed contents."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"expected mapping, got {type(data).__name__}")
        return data
    except Exception as exc:
        raise ValueError(f"failed to load {path}: {exc}") from exc


def check_lint_escalation(plans_dir: Path) -> list[str]:
    """Validate lint-escalation.yaml structure."""
    errors: list[str] = []
    path = plans_dir / "lint-escalation.yaml"
    if not path.exists():
        errors.append(f"missing: {path}")
        return errors

    try:
        data = load_manifest(path)
    except ValueError as exc:
        return [str(exc)]

    rules = data.get("rules")
    if not isinstance(rules, list):
        errors.append(f"{path}: 'rules' must be a list")
        return errors

    rule_ids: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            errors.append(f"{path}: rule is not a mapping")
            continue
        rid = rule.get("id")
        if not isinstance(rid, str) or not rid.strip():
            errors.append(f"{path}: rule missing 'id'")
        elif rid in rule_ids:
            errors.append(f"{path}: duplicate rule id {rid!r}")
        else:
            rule_ids.add(rid)

        severity = rule.get("severity")
        if severity not in ("soft", "hard"):
            errors.append(f"{path}: rule {rid!r} missing or invalid severity")
        sunset = rule.get("sunset")
        if not isinstance(sunset, str):
            errors.append(f"{path}: rule {rid!r} missing sunset date")

    return errors


def check_suppressions(plans_dir: Path) -> list[str]:
    """Validate suppressions.yaml structure."""
    errors: list[str] = []
    path = plans_dir / "suppressions.yaml"
    if not path.exists():
        return errors  # optional, not yet created

    try:
        data = load_manifest(path)
    except ValueError as exc:
        return [str(exc)]

    suppressions = data.get("suppressions")
    if not isinstance(suppressions, list):
        errors.append(f"{path}: 'suppressions' must be a list")
        return errors

    for s in suppressions:
        if not isinstance(s, dict):
            errors.append(f"{path}: suppression is not a mapping")
            continue
        sid = s.get("id")
        if not isinstance(sid, str):
            errors.append(f"{path}: suppression missing 'id'")
        expires = s.get("expires_at")
        if not isinstance(expires, str):
            errors.append(f"{path}: suppression {sid!r} missing 'expires_at'")

    return errors


def check_assurance_domains(plans_dir: Path) -> list[str]:
    """Validate assurance-domains.yaml references existing issues/claims."""
    errors: list[str] = []
    path = plans_dir / "assurance-domains.yaml"
    if not path.exists():
        return errors  # optional, may not yet exist

    try:
        data = load_manifest(path)
    except ValueError as exc:
        return [str(exc)]

    domains = data.get("domains")
    if not isinstance(domains, dict):
        errors.append(f"{path}: 'domains' must be a mapping")
        return errors

    for name, domain in domains.items():
        if not isinstance(domain, dict):
            errors.append(f"{path}: domain {name!r} is not a mapping")
            continue
        if "description" not in domain:
            errors.append(f"{path}: domain {name!r} missing description")
        maturity = domain.get("maturity")
        valid_maturity = {"seed", "nascent", "growing", "established", "complete"}
        if maturity not in valid_maturity:
            errors.append(f"{path}: domain {name!r} invalid maturity {maturity!r}")

    return errors


def check_coverage_gaps(plans_dir: Path) -> list[str]:
    """Validate that passive coverage gaps are tracked as actionable records."""
    errors: list[str] = []
    for path in sorted(plans_dir.glob("*coverage*.yaml")):
        try:
            data = load_manifest(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        gaps = data.get("coverage_gaps")
        if gaps is None:
            continue
        if not isinstance(gaps, list):
            errors.append(f"{path}: 'coverage_gaps' must be a list")
            continue
        for index, gap in enumerate(gaps):
            if not isinstance(gap, dict):
                errors.append(f"{path}: coverage_gaps[{index}] must be a mapping")
                continue
            axis_keys = ("domain", "subject", "area", "dimension", "artifact", "platform", "concern")
            if not any(isinstance(gap.get(key), str) and gap.get(key, "").strip() for key in axis_keys):
                errors.append(f"{path}: coverage_gaps[{index}] missing coverage axis")
            if not isinstance(gap.get("gap"), str) or not gap.get("gap", "").strip():
                errors.append(f"{path}: coverage_gaps[{index}] missing gap text")
            if not isinstance(gap.get("owner"), str) or not gap.get("owner", "").strip():
                errors.append(f"{path}: coverage_gaps[{index}] missing owner")
            if not isinstance(gap.get("next_evidence"), str) or not gap.get("next_evidence", "").strip():
                errors.append(f"{path}: coverage_gaps[{index}] missing next_evidence")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run all manifest consistency checks. Returns exit code."""
    project_root = Path(__file__).resolve().parents[1]
    plans_dir = project_root / "docs" / "plans"

    if not plans_dir.is_dir():
        print(f"error: plans directory not found: {plans_dir}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    for check in (check_lint_escalation, check_suppressions, check_assurance_domains, check_coverage_gaps):
        try:
            all_errors.extend(check(plans_dir))
        except Exception as exc:
            all_errors.append(f"{check.__name__}: {exc}")

    if all_errors:
        for err in all_errors:
            print(f"  ✗ {err}", file=sys.stderr)
        plural = "s" if len(all_errors) != 1 else ""
        print(f"\n{len(all_errors)} manifest consistency error{plural}", file=sys.stderr)
        return 1

    print("  ✓ manifest consistency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
