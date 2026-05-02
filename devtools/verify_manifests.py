"""Verify internal consistency across all docs/plans/*.yaml manifests.

Ensures manifest files are valid YAML and their cross-references are
consistent. Runs as part of devtools verify.
"""

from __future__ import annotations

import shlex
import sys
from datetime import date
from pathlib import Path

import yaml

from devtools.command_catalog import COMMANDS

_COVERAGE_AXIS_KEYS = ("domain", "subject", "area", "dimension", "artifact", "platform", "concern")
_COVERAGE_GAP_SEVERITIES = {"info", "minor", "major", "serious"}
_EVIDENCE_COMMANDS = {"pytest", "ruff", "mypy", "nix", "polylogue", "polylogued", "polylogue-mcp"}
_COVERAGE_COMMAND_FIELDS = {"generated_by", "verified_by", "verification_command"}
_COVERAGE_PATH_FIELDS = {"config_location", "location", "path", "strategies_location"}
_COVERAGE_PATH_LIST_FIELDS = {"locations", "tests"}


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
    gap_ids: set[str] = set()
    gap_subject_leaves: set[str] = set()
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
            label = _coverage_gap_label(path, index, gap)
            gap_id = gap.get("id")
            if not isinstance(gap_id, str) or not gap_id.strip():
                errors.append(f"{label} missing id")
            elif gap_id in gap_ids:
                errors.append(f"{label} duplicate id {gap_id!r}")
            else:
                gap_ids.add(gap_id)
                gap_subject_leaf = _coverage_gap_slug(gap_id)
                if gap_subject_leaf in gap_subject_leaves:
                    errors.append(f"{label} duplicate proof subject slug {gap_subject_leaf!r}")
                else:
                    gap_subject_leaves.add(gap_subject_leaf)
            if not any(isinstance(gap.get(key), str) and gap.get(key, "").strip() for key in _COVERAGE_AXIS_KEYS):
                errors.append(f"{path}: coverage_gaps[{index}] missing coverage axis")
            if not isinstance(gap.get("gap"), str) or not gap.get("gap", "").strip():
                errors.append(f"{label} missing gap text")
            if not isinstance(gap.get("owner"), str) or not gap.get("owner", "").strip():
                errors.append(f"{label} missing owner")
            severity = gap.get("severity")
            if severity not in _COVERAGE_GAP_SEVERITIES:
                errors.append(f"{label} missing or invalid severity")
            for field in ("declared_at", "review_after"):
                if not _valid_iso_date(gap.get(field)):
                    errors.append(f"{label} missing or invalid {field}")
            issue = gap.get("issue")
            suppression = gap.get("suppression")
            if not _valid_issue_ref(issue) and not _valid_suppression_ref(suppression):
                errors.append(f"{label} missing issue or suppression")
            next_evidence = gap.get("next_evidence")
            if not isinstance(next_evidence, str) or not next_evidence.strip():
                errors.append(f"{label} missing next_evidence")
            elif not _resolvable_next_evidence(next_evidence):
                errors.append(f"{label} next_evidence does not resolve to a known command")
    return errors


def check_coverage_references(plans_dir: Path) -> list[str]:
    """Validate locally checkable command and path references in coverage manifests."""
    errors: list[str] = []
    repo_root = _repo_root_for_plans(plans_dir)
    for path in sorted(plans_dir.glob("*coverage*.yaml")):
        try:
            data = load_manifest(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for ref_path, key, value in _iter_manifest_fields(data):
            label = f"{path}: {'.'.join(ref_path)}"
            if (
                key in _COVERAGE_COMMAND_FIELDS
                and isinstance(value, str)
                and value.strip()
                and not _resolvable_command(value)
            ):
                errors.append(f"{label} command does not resolve: {value!r}")
            if key in _COVERAGE_PATH_FIELDS and isinstance(value, str) and value.strip():
                candidate = _manifest_path_token(value)
                if candidate and not _manifest_path_exists(repo_root, candidate):
                    errors.append(f"{label} path does not exist: {candidate!r}")
            if key in _COVERAGE_PATH_LIST_FIELDS and isinstance(value, list):
                for item_index, item in enumerate(value):
                    if not isinstance(item, str) or not item.strip():
                        continue
                    candidate = _manifest_path_token(item)
                    if candidate and not _manifest_path_exists(repo_root, candidate):
                        errors.append(f"{label}[{item_index}] path does not exist: {candidate!r}")
    return errors


def _iter_manifest_fields(value: object, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str, object]]:
    fields: list[tuple[tuple[str, ...], str, object]] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = (*prefix, key)
            fields.append((child_path, key, child))
            fields.extend(_iter_manifest_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.extend(_iter_manifest_fields(child, (*prefix, str(index))))
    return fields


def _repo_root_for_plans(plans_dir: Path) -> Path:
    if plans_dir.name == "plans" and plans_dir.parent.name == "docs":
        return plans_dir.parent.parent
    return plans_dir


def _manifest_path_token(value: str) -> str:
    token = value.strip().split(maxsplit=1)[0].strip()
    return "" if token in {"", "null", "dynamic", "unknown"} else token


def _manifest_path_exists(repo_root: Path, token: str) -> bool:
    path = Path(token)
    if path.is_absolute():
        return path.exists()
    return (repo_root / path).exists()


def _coverage_gap_label(path: Path, index: int, gap: dict[object, object]) -> str:
    gap_id = gap.get("id")
    if isinstance(gap_id, str) and gap_id.strip():
        return f"{path}: coverage_gaps[{index}] {gap_id!r}"
    return f"{path}: coverage_gaps[{index}]"


def _valid_iso_date(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _valid_issue_ref(value: object) -> bool:
    if isinstance(value, int):
        return value > 0
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped.startswith("#"):
        stripped = stripped[1:]
    return stripped.isdecimal() and int(stripped) > 0


def _valid_suppression_ref(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _resolvable_next_evidence(value: str) -> bool:
    return _resolvable_command(value)


def _resolvable_command(value: str) -> bool:
    try:
        tokens = shlex.split(value)
    except ValueError:
        return False
    if not tokens:
        return False
    command = tokens[0]
    if command == "devtools":
        return len(tokens) >= 2 and tokens[1] in COMMANDS
    return command in _EVIDENCE_COMMANDS


def _coverage_gap_slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-") or "unnamed"


def main(argv: list[str] | None = None) -> int:
    """Run all manifest consistency checks. Returns exit code."""
    project_root = Path(__file__).resolve().parents[1]
    plans_dir = project_root / "docs" / "plans"

    if not plans_dir.is_dir():
        print(f"error: plans directory not found: {plans_dir}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    for check in (
        check_lint_escalation,
        check_suppressions,
        check_assurance_domains,
        check_coverage_gaps,
        check_coverage_references,
    ):
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
