"""Changed-path routing for proof obligations."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.lib.json import JSONDocument
from polylogue.operations import OperationSpec, build_declared_operation_catalog
from polylogue.proof.catalog import VerificationCatalog, build_verification_catalog
from polylogue.proof.models import ProofObligation, SubjectRef

ChangeKind = Literal[
    "parser",
    "schema.annotation",
    "provider.capability",
    "command",
    "generated_surface",
    "architecture",
    "coverage_manifest",
    "schema_roundtrip",
    "proof_catalog",
    "operation.spec",
    "workflow",
    "test",
    "unknown",
]
CheckScope = Literal["inner_loop", "pr_gate", "deployment_gate"]
DiffStatus = Literal["new", "dropped", "now_failing", "now_passing", "stale_evidence", "suppressed"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALL_COMMAND_KINDS = {"cli.command", "cli.json_command"}
_FULL_PR_GATES = (
    ("devtools", "verify", "--quick"),
    ("devtools", "verify"),
)
_DEPLOYMENT_GATES = (
    ("devtools", "build-package"),
    ("nix", "flake", "check"),
)
_GENERATED_SURFACE_PATHS: Mapping[str, tuple[str, ...]] = {
    "AGENTS.md": ("agents",),
    "docs/cli-reference.md": ("cli-reference",),
    "docs/devtools.md": ("devtools-reference",),
    "docs/README.md": ("docs-surface",),
    "README.md": ("docs-surface",),
    "docs/test-quality-workflows.md": ("quality-reference",),
    "docs/verification-catalog.md": ("verification-catalog",),
}
_ARCHITECTURE_PATH_SUBJECTS: Mapping[str, tuple[str, ...]] = {
    "docs/plans/topology-target.yaml": ("architecture.topology.projection",),
    "docs/topology-status.md": ("architecture.topology.projection",),
    "devtools/verify_topology.py": ("architecture.topology.projection",),
    "devtools/build_topology_projection.py": ("architecture.topology.projection",),
    "docs/plans/layering.yaml": ("architecture.layering.import_rules",),
    "devtools/verify_layering.py": ("architecture.layering.import_rules",),
    "docs/plans/file-size-budgets.yaml": ("architecture.file_budget.loc",),
    "devtools/verify_file_budgets.py": ("architecture.file_budget.loc",),
    "devtools/verify_manifests.py": ("architecture.manifest.consistency",),
    "devtools/verify_witness_lifecycle.py": ("architecture.witness.lifecycle",),
}
_PROVIDER_BY_PARSER: Mapping[str, str] = {
    "chatgpt": "chatgpt",
    "claude": "claude-code",
    "codex": "codex",
}
_DIFF_STATUSES: tuple[DiffStatus, ...] = (
    "new",
    "dropped",
    "now_failing",
    "now_passing",
    "stale_evidence",
    "suppressed",
)


@dataclass(frozen=True, slots=True)
class RecommendedCheck:
    """A command recommended by affected-obligation routing."""

    command: tuple[str, ...]
    scope: CheckScope
    reason: str

    @property
    def rendered_command(self) -> str:
        return " ".join(self.command)

    def to_payload(self) -> JSONDocument:
        return {
            "command": list(self.command),
            "scope": self.scope,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ChangeSubject:
    """Semantic classification of one changed path."""

    id: str
    path: str
    kind: ChangeKind
    reason: str
    subject_ids: tuple[str, ...] = ()
    operation_names: tuple[str, ...] = ()
    surface_names: tuple[str, ...] = ()
    checks: tuple[RecommendedCheck, ...] = ()

    def to_payload(self) -> JSONDocument:
        return {
            "id": self.id,
            "path": self.path,
            "kind": self.kind,
            "reason": self.reason,
            "subject_ids": list(self.subject_ids),
            "operation_names": list(self.operation_names),
            "surface_names": list(self.surface_names),
            "checks": [check.to_payload() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class AffectedObligation:
    """A proof obligation selected by one or more changed subjects."""

    obligation_id: str
    claim_id: str
    subject_id: str
    runner_id: str
    reasons: tuple[str, ...]
    change_subject_ids: tuple[str, ...]

    def to_payload(self) -> JSONDocument:
        return {
            "obligation_id": self.obligation_id,
            "claim_id": self.claim_id,
            "subject_id": self.subject_id,
            "runner_id": self.runner_id,
            "reasons": list(self.reasons),
            "change_subject_ids": list(self.change_subject_ids),
        }


@dataclass(frozen=True, slots=True)
class ObligationDiff:
    """Diff buckets for proof-obligation routing across refs."""

    new: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    now_failing: tuple[str, ...] = ()
    now_passing: tuple[str, ...] = ()
    stale_evidence: tuple[str, ...] = ()
    suppressed: tuple[str, ...] = ()

    def bucket(self, status: DiffStatus) -> tuple[str, ...]:
        if status == "new":
            return self.new
        if status == "dropped":
            return self.dropped
        if status == "now_failing":
            return self.now_failing
        if status == "now_passing":
            return self.now_passing
        if status == "stale_evidence":
            return self.stale_evidence
        return self.suppressed

    def to_payload(self) -> JSONDocument:
        return {
            "new": list(self.new),
            "dropped": list(self.dropped),
            "now_failing": list(self.now_failing),
            "now_passing": list(self.now_passing),
            "stale_evidence": list(self.stale_evidence),
            "suppressed": list(self.suppressed),
        }


@dataclass(frozen=True, slots=True)
class AffectedObligationReport:
    """Complete affected-obligation routing report."""

    base_ref: str
    head_ref: str
    changed_paths: tuple[str, ...]
    change_subjects: tuple[ChangeSubject, ...]
    affected_obligations: tuple[AffectedObligation, ...]
    obligation_diff: ObligationDiff
    inner_loop_checks: tuple[RecommendedCheck, ...]
    pr_gates: tuple[RecommendedCheck, ...]
    deployment_gates: tuple[RecommendedCheck, ...]

    def to_payload(self) -> JSONDocument:
        return {
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "changed_paths": list(self.changed_paths),
            "change_subjects": [subject.to_payload() for subject in self.change_subjects],
            "affected_obligations": [obligation.to_payload() for obligation in self.affected_obligations],
            "obligation_diff": self.obligation_diff.to_payload(),
            "inner_loop_checks": [check.to_payload() for check in self.inner_loop_checks],
            "pr_gates": [check.to_payload() for check in self.pr_gates],
            "deployment_gates": [check.to_payload() for check in self.deployment_gates],
        }


def changed_paths_between_refs(base_ref: str, head_ref: str = "HEAD") -> tuple[str, ...]:
    """Return changed repo-relative paths between two git refs."""
    completed = subprocess.run(
        ("git", "diff", "--name-only", f"{base_ref}..{head_ref}"),
        check=True,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return tuple(sorted(path for path in completed.stdout.splitlines() if path.strip()))


def build_affected_obligation_report(
    changed_paths: Iterable[str],
    *,
    base_ref: str = "master",
    head_ref: str = "HEAD",
    catalog: VerificationCatalog | None = None,
    base_obligation_ids: Iterable[str] | None = None,
    head_obligation_ids: Iterable[str] | None = None,
) -> AffectedObligationReport:
    """Classify changed paths and map them to affected proof obligations."""
    head_catalog = catalog or build_verification_catalog()
    paths = tuple(sorted(dict.fromkeys(_normalize_path(path) for path in changed_paths if path.strip())))
    change_subjects = classify_changed_paths(paths, catalog=head_catalog)
    affected = route_affected_obligations(change_subjects, catalog=head_catalog)
    affected_ids = tuple(obligation.obligation_id for obligation in affected)
    base_ids = tuple(base_obligation_ids) if base_obligation_ids is not None else ()
    head_ids = (
        tuple(head_obligation_ids)
        if head_obligation_ids is not None
        else tuple(obligation.id for obligation in head_catalog.obligations)
    )
    diff = diff_obligation_ids(
        base_ids=base_ids,
        head_ids=head_ids,
        affected_ids=affected_ids,
        suppressed=_suppressed_change_subjects(change_subjects),
    )
    checks = _dedupe_checks(check for subject in change_subjects for check in subject.checks)
    return AffectedObligationReport(
        base_ref=base_ref,
        head_ref=head_ref,
        changed_paths=paths,
        change_subjects=change_subjects,
        affected_obligations=affected,
        obligation_diff=diff,
        inner_loop_checks=tuple(check for check in checks if check.scope == "inner_loop"),
        pr_gates=_standard_checks(_FULL_PR_GATES, scope="pr_gate", reason="full PR gate remains explicit"),
        deployment_gates=_standard_checks(
            _DEPLOYMENT_GATES,
            scope="deployment_gate",
            reason="deployment readiness gate; run when packaging or release confidence matters",
        ),
    )


def classify_changed_paths(
    changed_paths: Iterable[str],
    *,
    catalog: VerificationCatalog | None = None,
    operations: tuple[OperationSpec, ...] | None = None,
) -> tuple[ChangeSubject, ...]:
    """Classify changed paths into semantic subjects."""
    proof_catalog = catalog or build_verification_catalog()
    operation_specs = operations or build_declared_operation_catalog().specs
    subjects_by_path = _subjects_by_source_path(proof_catalog.subjects)
    operation_names_by_path = _operations_by_code_path(operation_specs)
    classified: list[ChangeSubject] = []
    for path in sorted(dict.fromkeys(_normalize_path(item) for item in changed_paths)):
        kind = _classify_path(path)
        subject_ids = _subject_ids_for_path(
            path,
            kind=kind,
            subjects_by_path=subjects_by_path,
            subjects=proof_catalog.subjects,
        )
        operation_names = operation_names_by_path.get(path, ())
        surface_names = _surface_names_for_path(path)
        classified.append(
            ChangeSubject(
                id=f"{kind}:{path}",
                path=path,
                kind=kind,
                reason=_reason_for_change(path, kind),
                subject_ids=subject_ids,
                operation_names=operation_names,
                surface_names=surface_names,
                checks=_checks_for_change(kind, path=path),
            )
        )
    return tuple(classified)


def route_affected_obligations(
    change_subjects: Iterable[ChangeSubject],
    *,
    catalog: VerificationCatalog | None = None,
) -> tuple[AffectedObligation, ...]:
    """Route change subjects to proof obligations."""
    proof_catalog = catalog or build_verification_catalog()
    obligations_by_subject: dict[str, list[ProofObligation]] = defaultdict(list)
    for obligation in proof_catalog.obligations:
        obligations_by_subject[obligation.subject.id].append(obligation)

    reasons_by_obligation: dict[str, list[str]] = defaultdict(list)
    change_ids_by_obligation: dict[str, list[str]] = defaultdict(list)
    obligations_by_id = {obligation.id: obligation for obligation in proof_catalog.obligations}
    for change in change_subjects:
        selected_subject_ids = set(change.subject_ids)
        if change.kind == "proof_catalog":
            selected_subject_ids.update(obligation.subject.id for obligation in proof_catalog.obligations)
        for subject_id in selected_subject_ids:
            for obligation in obligations_by_subject.get(subject_id, []):
                reasons_by_obligation[obligation.id].append(_obligation_reason(change, obligation.subject))
                change_ids_by_obligation[obligation.id].append(change.id)

    affected: list[AffectedObligation] = []
    for obligation_id in sorted(reasons_by_obligation):
        obligation = obligations_by_id[obligation_id]
        affected.append(
            AffectedObligation(
                obligation_id=obligation.id,
                claim_id=obligation.claim.id,
                subject_id=obligation.subject.id,
                runner_id=obligation.runner.id,
                reasons=tuple(dict.fromkeys(reasons_by_obligation[obligation_id])),
                change_subject_ids=tuple(dict.fromkeys(change_ids_by_obligation[obligation_id])),
            )
        )
    return tuple(affected)


def diff_obligation_ids(
    *,
    base_ids: Iterable[str],
    head_ids: Iterable[str],
    affected_ids: Iterable[str],
    suppressed: Iterable[str] = (),
) -> ObligationDiff:
    """Bucket obligation IDs into semantic diff categories."""
    base = set(base_ids)
    head = set(head_ids)
    affected = set(affected_ids)
    new = (head - base) & affected if base else set()
    dropped = base - head
    stable_affected = (base & head & affected) if base else affected
    return ObligationDiff(
        new=tuple(sorted(new)),
        dropped=tuple(sorted(dropped)),
        stale_evidence=tuple(sorted(stable_affected - new)),
        suppressed=tuple(sorted(dict.fromkeys(suppressed))),
    )


def obligation_ids_for_ref(ref: str) -> tuple[str, ...]:
    """Render the verification catalog at a git ref and return obligation IDs."""
    if ref in {"", "HEAD", "WORKTREE"}:
        return tuple(obligation.id for obligation in build_verification_catalog().obligations)

    with tempfile.TemporaryDirectory(prefix="polylogue-proof-ref-") as tmp:
        worktree = Path(tmp) / "worktree"
        subprocess.run(
            ("git", "worktree", "add", "--detach", str(worktree), ref),
            check=True,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        try:
            completed = subprocess.run(
                (sys.executable, "-m", "devtools", "render-verification-catalog", "--json"),
                check=True,
                cwd=worktree,
                capture_output=True,
                text=True,
            )
        finally:
            subprocess.run(
                ("git", "worktree", "remove", "--force", str(worktree)),
                check=False,
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
            )

    payload = json.loads(completed.stdout)
    obligations = payload.get("obligations", [])
    if not isinstance(obligations, list):
        return ()
    ids: list[str] = []
    for item in obligations:
        if isinstance(item, dict):
            obligation_id = item.get("id")
            if isinstance(obligation_id, str):
                ids.append(obligation_id)
    return tuple(sorted(ids))


def render_affected_obligations(report: AffectedObligationReport) -> str:
    """Render a human-readable affected-obligation report."""
    lines = [
        "Affected Obligations",
        f"Refs: {report.base_ref}..{report.head_ref}",
        "",
        "Changed Paths:",
    ]
    lines.extend(f"- {path}" for path in report.changed_paths)
    lines.extend(["", "Change Subjects:"])
    for subject in report.change_subjects:
        operations = f"; operations={', '.join(subject.operation_names)}" if subject.operation_names else ""
        surfaces = f"; surfaces={', '.join(subject.surface_names)}" if subject.surface_names else ""
        lines.append(f"- {subject.id}: {subject.reason}{operations}{surfaces}")
    lines.extend(["", "Obligation Diff:"])
    for status in _DIFF_STATUSES:
        values = report.obligation_diff.bucket(status)
        rendered = ", ".join(values[:8]) if values else "—"
        suffix = f" (+{len(values) - 8} more)" if len(values) > 8 else ""
        lines.append(f"- {status}: {rendered}{suffix}")
    lines.extend(["", "Affected Obligations:"])
    if not report.affected_obligations:
        lines.append("- none")
    for obligation in report.affected_obligations[:30]:
        lines.append(f"- {obligation.obligation_id}: {'; '.join(obligation.reasons)}")
    if len(report.affected_obligations) > 30:
        lines.append(f"- ... {len(report.affected_obligations) - 30} more")
    lines.extend(["", "Recommended Inner-Loop Checks:"])
    lines.extend(_render_checks(report.inner_loop_checks))
    lines.extend(["", "PR Gates:"])
    lines.extend(_render_checks(report.pr_gates))
    lines.extend(["", "Deployment Gates:"])
    lines.extend(_render_checks(report.deployment_gates))
    return "\n".join(lines)


def render_affected_obligations_markdown(report: AffectedObligationReport) -> str:
    """Render a PR-comment-friendly affected-obligation report."""
    lines = [
        "## Proof Obligations",
        "",
        f"**Refs:** `{report.base_ref}..{report.head_ref}`",
        "",
        "### Changed Paths",
    ]
    lines.extend(f"- `{path}`" for path in report.changed_paths) if report.changed_paths else lines.append("- none")
    lines.extend(["", "### Obligation Diff"])
    for status in _DIFF_STATUSES:
        values = report.obligation_diff.bucket(status)
        lines.append(f"- `{status}`: {len(values)}")
    lines.extend(["", "### Affected Obligations"])
    if not report.affected_obligations:
        lines.append("- none")
    for obligation in report.affected_obligations[:20]:
        lines.append(f"- `{obligation.obligation_id}` — {'; '.join(obligation.reasons)}")
    if len(report.affected_obligations) > 20:
        lines.append(f"- ... {len(report.affected_obligations) - 20} more")
    lines.extend(["", "### Required Checks", *_render_checks(report.inner_loop_checks), "", "### PR Gates"])
    lines.extend(_render_checks(report.pr_gates))
    return "\n".join(lines)


def _subjects_by_source_path(subjects: tuple[SubjectRef, ...]) -> dict[str, tuple[str, ...]]:
    by_path: dict[str, list[str]] = defaultdict(list)
    for subject in subjects:
        if subject.source_span is None:
            continue
        by_path[_normalize_path(subject.source_span.path)].append(subject.id)
    return {path: tuple(sorted(ids)) for path, ids in by_path.items()}


def _operations_by_code_path(operations: tuple[OperationSpec, ...]) -> dict[str, tuple[str, ...]]:
    by_path: dict[str, list[str]] = defaultdict(list)
    for operation in operations:
        for code_ref in operation.code_refs:
            path = _path_for_code_ref(code_ref)
            if path is not None:
                by_path[path].append(operation.name)
    return {path: tuple(sorted(names)) for path, names in by_path.items()}


def _path_for_code_ref(code_ref: str) -> str | None:
    parts = code_ref.split(".")
    for index in range(len(parts), 0, -1):
        candidate = Path(*parts[:index]).with_suffix(".py")
        if (_REPO_ROOT / candidate).exists():
            return candidate.as_posix()
    return None


def _subject_ids_for_path(
    path: str,
    *,
    kind: ChangeKind,
    subjects_by_path: Mapping[str, tuple[str, ...]],
    subjects: tuple[SubjectRef, ...],
) -> tuple[str, ...]:
    direct = set(subjects_by_path.get(path, ()))
    if kind == "parser":
        provider = _provider_for_parser_path(path)
        direct.update(
            subject.id
            for subject in subjects
            if subject.kind == "provider.capability" and (provider is None or subject.attrs.get("provider") == provider)
        )
    elif kind == "schema.annotation":
        provider = _provider_for_schema_path(path)
        direct.update(
            subject.id
            for subject in subjects
            if subject.kind == "schema.annotation"
            and (
                (subject.source_span is not None and _normalize_path(subject.source_span.path) == path)
                or (provider is not None and subject.attrs.get("provider") == provider)
            )
        )
        direct.add("schema.roundtrip.provider_packages")
    elif kind == "provider.capability":
        direct.update(subject.id for subject in subjects if subject.kind == "provider.capability")
    elif kind == "command":
        direct.update(subject.id for subject in subjects if subject.kind in _ALL_COMMAND_KINDS)
    elif kind == "generated_surface":
        direct.update(
            subject.id
            for subject in subjects
            if subject.kind == "workflow.claim" and subject.attrs.get("claim_family") == "generated-surfaces"
        )
    elif kind == "architecture":
        direct.update(_ARCHITECTURE_PATH_SUBJECTS.get(path, ()))
        if path.startswith("docs/plans/"):
            direct.add("architecture.manifest.consistency")
    elif kind == "coverage_manifest":
        direct.update(
            subject.id
            for subject in subjects
            if subject.kind.startswith("assurance.coverage_")
            and subject.source_span is not None
            and _normalize_path(subject.source_span.path) == path
        )
        direct.add("architecture.manifest.consistency")
    elif kind == "schema_roundtrip":
        direct.add("schema.roundtrip.provider_packages")
    elif kind == "workflow":
        direct.update(subject.id for subject in subjects if subject.kind == "workflow.claim")
    elif kind == "operation.spec":
        direct.update(subject.id for subject in subjects if subject.kind == "operation.spec")
    return tuple(sorted(direct))


def _classify_path(path: str) -> ChangeKind:
    if path.startswith("polylogue/sources/parsers/"):
        return "parser"
    if path.startswith("polylogue/schemas/providers/"):
        return "schema.annotation"
    if path.startswith("docs/plans/") and (
        path.endswith("-coverage.yaml")
        or path.endswith("assurance-domains.yaml")
        or path.endswith("oracle-quality.yaml")
        or path.endswith("evidence-freshness.yaml")
    ):
        return "coverage_manifest"
    if path in _ARCHITECTURE_PATH_SUBJECTS or path.startswith("docs/plans/"):
        return "architecture"
    if path == "devtools/verify_schema_roundtrip.py" or path.startswith("tests/property/"):
        return "schema_roundtrip"
    if path == "polylogue/archive/provider/capabilities.py":
        return "provider.capability"
    if path.startswith("polylogue/cli/") or path == "polylogue/cli/command_inventory.py":
        return "command"
    if (
        path in _GENERATED_SURFACE_PATHS
        or path.startswith("devtools/render_")
        or path in {"devtools/generated_surfaces.py", "devtools/command_catalog.py"}
    ):
        return "generated_surface"
    if path.startswith("polylogue/proof/") or path == "devtools/affected_obligations.py":
        return "proof_catalog"
    if path == "polylogue/operations/specs.py":
        return "operation.spec"
    if path.startswith(".github/") or path in {"CONTRIBUTING.md", "CLAUDE.md", "AGENTS.md"}:
        return "workflow"
    if path.startswith("tests/"):
        return "test"
    return "unknown"


def _surface_names_for_path(path: str) -> tuple[str, ...]:
    if path in _GENERATED_SURFACE_PATHS:
        return _GENERATED_SURFACE_PATHS[path]
    if (
        path == "devtools/generated_surfaces.py"
        or path == "devtools/command_catalog.py"
        or path.startswith("devtools/render_")
    ):
        return tuple(sorted({name for names in _GENERATED_SURFACE_PATHS.values() for name in names}))
    return ()


def _provider_for_parser_path(path: str) -> str | None:
    stem = Path(path).stem
    return _PROVIDER_BY_PARSER.get(stem)


def _provider_for_schema_path(path: str) -> str | None:
    parts = Path(path).parts
    try:
        provider_index = parts.index("providers") + 1
    except ValueError:
        return None
    return parts[provider_index] if provider_index < len(parts) else None


def _checks_for_change(kind: ChangeKind, *, path: str) -> tuple[RecommendedCheck, ...]:
    if kind == "parser":
        return (
            _check(("pytest", "tests/unit/sources"), "parser semantics changed"),
            _check(
                ("devtools", "pipeline-probe", "--provider", _provider_for_parser_path(path) or "unknown"),
                "parser pipeline probe",
            ),
            _check(("devtools", "affected-obligations", "--path", path), "refresh focused obligation routing"),
        )
    if kind == "schema.annotation":
        return (
            _check(
                ("pytest", "tests/unit/core/test_schema_annotation_contracts.py"), "schema annotation contract changed"
            ),
            _check(
                ("pytest", "tests/unit/proof/test_schema_provider_obligations.py"), "schema proof obligations changed"
            ),
            _check(("devtools", "render-verification-catalog", "--check"), "schema subjects feed generated catalog"),
        )
    if kind == "command":
        return (
            _check(("pytest", "tests/unit/cli"), "CLI command surface changed"),
            _check(
                ("pytest", "tests/unit/proof/test_evidence_runners.py"), "CLI proof runners exercise command output"
            ),
            _check(("devtools", "render-cli-reference", "--check"), "CLI reference is generated from command help"),
        )
    if kind == "generated_surface":
        return (_check(("devtools", "render-all", "--check"), "generated surface or renderer changed"),)
    if kind == "architecture":
        checks = [
            _check(("devtools", "verify-manifests"), "structural manifest changed"),
            _check(("devtools", "affected-obligations", "--path", path), "refresh structural obligation routing"),
        ]
        for subject_id in _ARCHITECTURE_PATH_SUBJECTS.get(path, ()):
            if subject_id == "architecture.topology.projection":
                checks.append(_check(("devtools", "verify-topology"), "topology projection changed"))
            elif subject_id == "architecture.layering.import_rules":
                checks.append(_check(("devtools", "verify-layering"), "layering rules changed"))
            elif subject_id == "architecture.file_budget.loc":
                checks.append(_check(("devtools", "verify-file-budgets"), "file budget rules changed"))
            elif subject_id == "architecture.witness.lifecycle":
                checks.append(_check(("devtools", "verify-witness-lifecycle"), "witness lifecycle changed"))
        return tuple(checks)
    if kind == "coverage_manifest":
        return (
            _check(("devtools", "verify-manifests"), "assurance coverage manifest changed"),
            _check(("devtools", "render-verification-catalog", "--check"), "coverage manifests feed proof subjects"),
            _check(("devtools", "proof-pack", "--path", path, "--markdown"), "coverage changes feed proof-pack gaps"),
        )
    if kind == "schema_roundtrip":
        return (
            _check(("devtools", "verify-schema-roundtrip", "--all"), "schema roundtrip control changed"),
            _check(("pytest", "tests/property/test_schema_roundtrip.py"), "schema roundtrip properties changed"),
        )
    if kind == "proof_catalog":
        return (
            _check(("pytest", "tests/unit/proof"), "proof kernel or catalog changed"),
            _check(("pytest", "tests/unit/devtools/test_render_verification_catalog.py"), "catalog renderer changed"),
            _check(("devtools", "render-verification-catalog", "--check"), "verification catalog must stay current"),
        )
    if kind == "operation.spec":
        return (
            _check(("pytest", "tests/unit/operations/test_specs.py"), "operation spec routing metadata changed"),
            _check(("devtools", "artifact-graph", "--json"), "operation specs feed artifact graph views"),
        )
    if kind == "workflow":
        return (
            _check(
                ("pytest", "tests/unit/devtools/test_command_catalog.py"), "workflow/control-plane contract changed"
            ),
            _check(("devtools", "render-all", "--check"), "workflow docs may feed generated surfaces"),
        )
    return ()


def _check(command: tuple[str, ...], reason: str) -> RecommendedCheck:
    return RecommendedCheck(command=command, scope="inner_loop", reason=reason)


def _standard_checks(
    commands: Iterable[tuple[str, ...]], *, scope: CheckScope, reason: str
) -> tuple[RecommendedCheck, ...]:
    return tuple(RecommendedCheck(command=command, scope=scope, reason=reason) for command in commands)


def _dedupe_checks(checks: Iterable[RecommendedCheck]) -> tuple[RecommendedCheck, ...]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[RecommendedCheck] = []
    for check in checks:
        if check.command in seen:
            continue
        seen.add(check.command)
        deduped.append(check)
    return tuple(deduped)


def _suppressed_change_subjects(change_subjects: tuple[ChangeSubject, ...]) -> tuple[str, ...]:
    return tuple(
        subject.id for subject in change_subjects if not subject.subject_ids and subject.kind in {"test", "unknown"}
    )


def _obligation_reason(change: ChangeSubject, subject: SubjectRef) -> str:
    if change.kind == "proof_catalog":
        return f"{change.path} changes the proof catalog compiler or runner vocabulary"
    if subject.source_span is not None and _normalize_path(subject.source_span.path) == change.path:
        return f"{change.path} is the source span for subject {subject.id}"
    return f"{change.kind} change {change.path} selects subject {subject.id}"


def _reason_for_change(path: str, kind: ChangeKind) -> str:
    if kind == "parser":
        return "provider parser semantics can change normalized archive facts"
    if kind == "schema.annotation":
        return "provider schema annotations feed schema proof subjects"
    if kind == "provider.capability":
        return "provider capability metadata feeds provider proof subjects"
    if kind == "command":
        return "CLI command registration or callbacks feed command subjects"
    if kind == "generated_surface":
        return "generated surface source or output changed"
    if kind == "architecture":
        return "architecture topology, layering, budget, or manifest control changed"
    if kind == "coverage_manifest":
        return "assurance coverage manifest changed"
    if kind == "schema_roundtrip":
        return "schema inference-validation roundtrip control changed"
    if kind == "proof_catalog":
        return "proof catalog source changed"
    if kind == "operation.spec":
        return "declared operation metadata feeds routing and artifact graph subjects"
    if kind == "workflow":
        return "durable workflow policy changed"
    if kind == "test":
        return "test-only change does not select proof obligations by itself"
    return "unclassified path; no proof obligation route exists yet"


def _normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/").removeprefix("./")


def _render_checks(checks: tuple[RecommendedCheck, ...]) -> list[str]:
    if not checks:
        return ["- none"]
    return [f"- `{check.rendered_command}` — {check.reason}" for check in checks]


__all__ = [
    "AffectedObligation",
    "AffectedObligationReport",
    "ChangeKind",
    "ChangeSubject",
    "CheckScope",
    "DiffStatus",
    "ObligationDiff",
    "RecommendedCheck",
    "build_affected_obligation_report",
    "changed_paths_between_refs",
    "classify_changed_paths",
    "diff_obligation_ids",
    "obligation_ids_for_ref",
    "render_affected_obligations",
    "render_affected_obligations_markdown",
    "route_affected_obligations",
]
