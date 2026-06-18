"""Verify internal consistency across all docs/plans/*.yaml manifests.

Ensures manifest files are valid YAML, their cross-references are
consistent, and their structure matches the declared Pydantic models.
Runs as part of devtools verify.
"""

from __future__ import annotations

import shlex
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from devtools import repo_root as _get_root
from devtools.authored_scenario_catalog import get_authored_scenario_catalog
from devtools.command_catalog import COMMANDS, command_name_from_tokens
from devtools.verify_ci_workflows import WorkflowInventory, inventory_workflows
from polylogue.verification.manifests.models import validate_manifest

_COVERAGE_AXIS_KEYS = ("domain", "subject", "area", "dimension", "artifact", "platform", "concern")
_COVERAGE_GAP_SEVERITIES = {"info", "minor", "major", "serious"}
_EVIDENCE_COMMANDS = {
    "pytest",
    "ruff",
    "mypy",
    "nix",
    "polylogue",
    "polylogued",
    "polylogue-mcp",
    # Browser-extension build pipeline (browser-extension/scripts/*.mjs).
    "npm",
    "node",
}
_COVERAGE_COMMAND_FIELDS = {"generated_by", "verified_by", "verification_command"}
_COVERAGE_PATH_FIELDS = {"config_location", "location", "path", "strategies_location"}
_COVERAGE_PATH_LIST_FIELDS = {"locations", "paths_to_mutate", "tests"}


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


def check_coverage_status_claims(plans_dir: Path) -> list[str]:
    """Validate internally contradictory coverage status claims."""
    errors: list[str] = []
    for path in sorted(plans_dir.glob("*coverage*.yaml")):
        try:
            data = load_manifest(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for ref_path, payload in _iter_manifest_mappings(data):
            label = f"{path}: {'.'.join(ref_path)}"
            implemented = payload.get("implemented")
            if implemented is True:
                if not _non_empty_list(payload.get("controls")):
                    errors.append(f"{label} implemented=true but controls are missing or empty")
                if not _has_test_coverage_location(payload.get("test_coverage")):
                    errors.append(f"{label} implemented=true but test_coverage.location is missing")
            elif implemented is False:
                if _non_empty_list(payload.get("controls")):
                    errors.append(f"{label} implemented=false but controls are declared")
                if _has_test_coverage_location(payload.get("test_coverage")):
                    errors.append(f"{label} implemented=false but test_coverage.location is declared")
    return errors


def check_campaign_coverage_catalog(plans_dir: Path) -> list[str]:
    """Validate campaign-coverage.yaml against the authored campaign catalog."""
    errors: list[str] = []
    path = plans_dir / "campaign-coverage.yaml"
    if not path.exists():
        return errors
    try:
        data = load_manifest(path)
    except ValueError as exc:
        return [str(exc)]

    catalog = get_authored_scenario_catalog()
    errors.extend(
        _compare_campaign_section(
            path=path,
            section="mutation_campaigns",
            actual=data.get("mutation_campaigns"),
            expected={
                entry.name: {
                    "paths_to_mutate": tuple(entry.paths_to_mutate),
                    "tests": tuple(entry.tests),
                    "status": "active",
                }
                for entry in catalog.mutation_campaigns
            },
        )
    )
    errors.extend(
        _compare_campaign_section(
            path=path,
            section="benchmark_campaigns",
            actual=data.get("benchmark_campaigns"),
            expected={
                entry.name: {
                    "tests": tuple(entry.tests),
                    "status": "active",
                }
                for entry in catalog.benchmark_campaigns
            },
        )
    )
    errors.extend(_check_campaign_test_paths(path, data, plans_dir))
    return errors


def _check_campaign_test_paths(
    path: Path,
    data: dict[str, object],
    plans_dir: Path,
) -> list[str]:
    """Verify every active campaign declares non-empty tests/paths that exist on disk.

    This closes the manifest-only loophole where a row can carry a name and a
    description but route nowhere executable.
    """
    repo_root = _repo_root_for_plans(plans_dir)
    errors: list[str] = []
    for section in ("mutation_campaigns", "benchmark_campaigns"):
        value = data.get(section)
        if not isinstance(value, list):
            continue
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            label_name = f"{name!r}" if isinstance(name, str) and name.strip() else f"index {index}"
            status = item.get("status", "active")
            if status != "active":
                continue
            tests = item.get("tests")
            if not isinstance(tests, list) or not tests:
                errors.append(
                    f"{path}: {section} campaign {label_name} declares no tests; "
                    "active campaigns must point at executable test paths"
                )
            else:
                for test_index, test_path in enumerate(tests):
                    if not isinstance(test_path, str) or not test_path.strip():
                        errors.append(
                            f"{path}: {section} campaign {label_name} tests[{test_index}] must be a non-empty string"
                        )
                        continue
                    candidate = _manifest_path_token(test_path)
                    if candidate and not _manifest_path_exists(repo_root, candidate):
                        errors.append(
                            f"{path}: {section} campaign {label_name} tests[{test_index}] "
                            f"path does not exist: {candidate!r}"
                        )
            if section == "mutation_campaigns":
                paths_to_mutate = item.get("paths_to_mutate")
                if not isinstance(paths_to_mutate, list) or not paths_to_mutate:
                    errors.append(
                        f"{path}: {section} campaign {label_name} declares no paths_to_mutate; "
                        "active mutation campaigns must target executable source paths"
                    )
            errors.extend(_check_campaign_freshness(path, section, label_name, item, repo_root))
    return errors


def _check_campaign_freshness(
    path: Path,
    section: str,
    label_name: str,
    item: dict[object, object],
    repo_root: Path,
) -> list[str]:
    """Enforce ``freshness_days`` / ``artifact_glob`` when declared.

    Absent fields stay silent so rows opt in gradually. When declared, the most
    recent matching artifact must exist, be non-empty, and (if
    ``freshness_days`` is set) be newer than the declared window. Without an
    explicit ``artifact_glob``, benchmark campaigns default to
    ``.local/benchmark-campaigns/*-<name>.json`` — the path written by
    ``devtools bench campaign run``.
    """
    errors: list[str] = []
    freshness_days = item.get("freshness_days")
    artifact_glob = item.get("artifact_glob")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return errors

    fresh_declared = isinstance(freshness_days, int) and freshness_days > 0
    glob_declared = isinstance(artifact_glob, str) and bool(artifact_glob.strip())
    if not fresh_declared and not glob_declared:
        return errors

    if glob_declared:
        assert isinstance(artifact_glob, str)
        glob = artifact_glob.strip()
    elif section == "benchmark_campaigns":
        glob = f".local/benchmark-campaigns/*-{name}.json"
    elif section == "mutation_campaigns":
        # Default mutation-campaign artifact layout (#1304). Matches
        # devtools.mutmut_campaign.default_artifact_paths.
        glob = f".local/mutation-campaigns/{name}/*.json"
    else:
        errors.append(
            f"{path}: {section} campaign {label_name} declares freshness_days without "
            "an artifact_glob; only mutation_campaigns and benchmark_campaigns have a default location"
        )
        return errors

    matches = sorted(repo_root.glob(glob))
    if not matches:
        errors.append(
            f"{path}: {section} campaign {label_name} declares artifact glob {glob!r} but no matching artifacts exist"
        )
        return errors

    newest = max(matches, key=lambda p: p.stat().st_mtime)
    if newest.stat().st_size == 0:
        errors.append(f"{path}: {section} campaign {label_name} newest artifact {newest.name!r} is empty")

    if fresh_declared:
        assert isinstance(freshness_days, int)
        age = datetime.now(UTC) - datetime.fromtimestamp(newest.stat().st_mtime, tz=UTC)
        if age > timedelta(days=freshness_days):
            errors.append(
                f"{path}: {section} campaign {label_name} newest artifact {newest.name!r} is "
                f"{age.days}d old, exceeding freshness_days={freshness_days}"
            )

    return errors


def _compare_campaign_section(
    *,
    path: Path,
    section: str,
    actual: object,
    expected: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    actual_by_name = _manifest_named_entries(path, section, actual, errors)
    if actual_by_name is None:
        return errors

    actual_names = set(actual_by_name)
    expected_names = set(expected)
    for name in sorted(expected_names - actual_names):
        errors.append(f"{path}: {section} missing catalog campaign {name!r}")
    for name in sorted(actual_names - expected_names):
        errors.append(f"{path}: {section} declares unknown campaign {name!r}")

    for name in sorted(actual_names & expected_names):
        payload = actual_by_name[name]
        for field, expected_value in expected[name].items():
            actual_value: object
            if isinstance(expected_value, tuple):
                actual_value = _string_tuple(payload.get(field))
            else:
                actual_value = payload.get(field)
            if actual_value != expected_value:
                errors.append(
                    f"{path}: {section}[{name!r}] {field} does not match authored catalog "
                    f"(expected {expected_value!r}, got {actual_value!r})"
                )
    return errors


def _manifest_named_entries(
    path: Path,
    section: str,
    value: object,
    errors: list[str],
) -> dict[str, dict[object, object]] | None:
    if value is None:
        return {}
    if not isinstance(value, list):
        errors.append(f"{path}: {section!r} must be a list")
        return None
    entries: dict[str, dict[object, object]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{path}: {section}[{index}] must be a mapping")
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{path}: {section}[{index}] missing name")
            continue
        if name in entries:
            errors.append(f"{path}: {section}[{index}] duplicate campaign name {name!r}")
            continue
        entries[name] = item
    return entries


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _iter_manifest_mappings(
    value: object, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], dict[object, object]]]:
    mappings: list[tuple[tuple[str, ...], dict[object, object]]] = []
    if isinstance(value, dict):
        mappings.append((prefix, value))
        for raw_key, child in value.items():
            mappings.extend(_iter_manifest_mappings(child, (*prefix, str(raw_key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            mappings.extend(_iter_manifest_mappings(child, (*prefix, str(index))))
    return mappings


def _non_empty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def _has_test_coverage_location(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    location = value.get("location")
    return isinstance(location, str) and bool(_manifest_path_token(location))


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
        return command_name_from_tokens(tokens[1:]) in COMMANDS
    return command in _EVIDENCE_COMMANDS


def _coverage_gap_slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-") or "unnamed"


def _command_present_in_workflows(command: str, inventory: WorkflowInventory) -> bool:
    """Return True if ``command`` (or its leading token) appears in any ``run:``.

    Substring match is sufficient — manifests declare canonical CLI commands
    such as ``devtools coverage-gate`` or ``uv build --wheel .`` and workflows
    invoke them either directly or via ``uv run`` wrappers. We strip the
    optional ``uv run`` prefix so the substring still resolves.
    """
    if not command or not command.strip():
        return False
    needle = command.strip()
    for run in inventory.all_run_commands:
        if needle in run:
            return True
    # Fall back to the registered command path, e.g. "devtools render all".
    try:
        tokens = shlex.split(needle)
    except ValueError:
        return False
    if len(tokens) >= 2 and tokens[0] == "devtools":
        command_name = command_name_from_tokens(tokens[1:])
        if command_name is None:
            return False
        prefix = f"devtools {command_name}"
        for run in inventory.all_run_commands:
            if prefix in run:
                return True
    elif len(tokens) >= 2:
        bigram = f"{tokens[0]} {tokens[1]}"
        for run in inventory.all_run_commands:
            if bigram in run:
                return True
    return False


def check_distribution_ci_claims(
    plans_dir: Path,
    inventory: WorkflowInventory | None = None,
) -> list[str]:
    """Verify ``ci_build``/``ci_test``/``ci_present`` claims against workflows.

    ``distribution-coverage.yaml`` declares whether each artifact's build,
    test, and presence is wired into CI. These are locally verifiable: the
    declared ``build_command``/``verification_command`` must appear in some
    workflow ``run:`` step, otherwise the manifest is lying about CI state.
    """
    errors: list[str] = []
    path = plans_dir / "distribution-coverage.yaml"
    if not path.exists():
        return errors
    try:
        data = load_manifest(path)
    except ValueError as exc:
        return [str(exc)]

    wf = inventory if inventory is not None else inventory_workflows(plans_dir.parent.parent / ".github" / "workflows")

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        return errors

    for artifact_name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            continue
        label = f"{path}: artifacts.{artifact_name}"

        if artifact.get("ci_build") is True:
            build_command = artifact.get("build_command")
            if not isinstance(build_command, str) or not build_command.strip():
                errors.append(f"{label} ci_build=true but build_command is missing")
            elif not _command_present_in_workflows(build_command, wf):
                errors.append(
                    f"{label} ci_build=true but build_command {build_command!r} "
                    "does not appear in any workflow run step"
                )

        if artifact.get("ci_test") is True:
            verification_command = artifact.get("verification_command")
            if isinstance(verification_command, str) and verification_command.strip():
                if not _command_present_in_workflows(verification_command, wf):
                    errors.append(
                        f"{label} ci_test=true but verification_command "
                        f"{verification_command!r} does not appear in any workflow run step"
                    )
            else:
                # ci_test without a verification_command must at least show the
                # build_command in CI (nix flake check counts as the verification).
                build_command = artifact.get("build_command")
                if (
                    isinstance(build_command, str)
                    and build_command.strip()
                    and not _command_present_in_workflows(build_command, wf)
                ):
                    errors.append(
                        f"{label} ci_test=true but neither verification_command nor "
                        f"build_command {build_command!r} appears in any workflow run step"
                    )

    return errors


def check_test_quality_ci_claims(
    plans_dir: Path,
    inventory: WorkflowInventory | None = None,
) -> list[str]:
    """Verify ``ci_gate: true`` in test-quality-coverage.yaml is real.

    If a dimension claims ``ci_gate: true``, then either the declared
    ``tool`` invocation or the canonical devtools gate command must appear
    in some workflow ``run:`` step. Otherwise the manifest is claiming a
    CI gate that does not exist.
    """
    errors: list[str] = []
    path = plans_dir / "test-quality-coverage.yaml"
    if not path.exists():
        return errors
    try:
        data = load_manifest(path)
    except ValueError as exc:
        return [str(exc)]

    wf = inventory if inventory is not None else inventory_workflows(plans_dir.parent.parent / ".github" / "workflows")

    dimensions = data.get("dimensions")
    if not isinstance(dimensions, dict):
        return errors

    for dim_name, dim in dimensions.items():
        if not isinstance(dim, dict):
            continue
        if dim.get("ci_gate") is not True:
            continue
        label = f"{path}: dimensions.{dim_name}"
        tool = dim.get("tool") if isinstance(dim.get("tool"), str) else ""
        # Probe candidates: the tool string (e.g. "pytest-cov"), and the
        # canonical devtools gate for known tools.
        candidates: list[str] = []
        if tool:
            candidates.append(tool)
        if tool in {"pytest-cov", "coverage"}:
            candidates.append("devtools coverage-gate")
        if not any(_command_present_in_workflows(candidate, wf) for candidate in candidates):
            errors.append(f"{label} ci_gate=true but no workflow run step invokes any of {candidates!r}")
    return errors


def check_test_coverage_domains(plans_dir: Path) -> list[str]:
    """Validate test-coverage-domains.yaml covering_tests paths exist."""
    errors: list[str] = []
    path = plans_dir / "test-coverage-domains.yaml"
    if not path.exists():
        return errors
    try:
        data = load_manifest(path)
    except ValueError as exc:
        return [str(exc)]

    repo_root = plans_dir.parent.parent
    domains = data.get("domains", [])
    if not isinstance(domains, list):
        return errors
    for domain in domains:
        if not isinstance(domain, dict):
            continue
        name = domain.get("domain", "<unknown>")
        covering = domain.get("covering_tests", [])
        if not isinstance(covering, list):
            continue
        for test_path in covering:
            full = repo_root / test_path
            if not full.exists():
                errors.append(
                    f"test-coverage-domains.yaml: domain {name!r}: covering_tests path does not exist: {test_path!r}"
                )
    return errors


def check_pydantic_models(plans_dir: Path) -> list[str]:
    """Validate every YAML manifest against its Pydantic model schema."""
    errors: list[str] = []
    for path in sorted(plans_dir.glob("*.yaml")):
        try:
            data = load_manifest(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        model_errors = validate_manifest(str(path), data)
        errors.extend(model_errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run all manifest consistency checks. Returns exit code."""
    project_root = _get_root()
    plans_dir = project_root / "docs" / "plans"

    if not plans_dir.is_dir():
        print(f"error: plans directory not found: {plans_dir}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    for check in (
        check_pydantic_models,
        check_lint_escalation,
        check_coverage_gaps,
        check_coverage_references,
        check_coverage_status_claims,
        check_campaign_coverage_catalog,
        check_test_coverage_domains,
        check_distribution_ci_claims,
        check_test_quality_ci_claims,
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
