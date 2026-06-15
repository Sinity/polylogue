"""Shared command catalog for repository developer tools."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

CommandMain = Callable[[list[str] | None], int]
CONTROL_PLANE = "devtools"
VERIFICATION_LAB_COMMAND_NAMES: tuple[str, ...] = (
    "lab-scenario",
    "schema-generate",
    "schema-promote",
    "schema-audit",
    "verify-schema-roundtrip",
)

CATEGORY_ORDER: tuple[str, ...] = (
    "core",
    "generated surfaces",
    "verification",
    "campaigns",
    "maintenance",
)


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    category: str
    description: str
    module: str
    entrypoint: str = "main"
    use_when: str | None = None
    examples: tuple[str, ...] = ()
    featured: bool = False

    @property
    def invocation(self) -> str:
        return control_plane_command(self.name)

    @property
    def argv(self) -> tuple[str, ...]:
        return control_plane_argv(self.name)

    def resolve_main(self) -> CommandMain:
        module = importlib.import_module(self.module)
        entrypoint = getattr(module, self.entrypoint)
        if not callable(entrypoint):
            raise TypeError(f"{self.module}.{self.entrypoint} is not callable")

        def _main(argv: list[str] | None = None) -> int:
            result = entrypoint(argv)
            if not isinstance(result, int):
                raise TypeError(f"{self.module}.{self.entrypoint} returned {type(result).__name__}, expected int")
            return result

        return _main

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["invocation"] = self.invocation
        data["argv"] = list(self.argv)
        return data


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "status",
        "core",
        "Render the devshell status view.",
        "devtools.project_motd",
        use_when="Check repo state, generated-surface drift, and the next default verification steps.",
        examples=("devtools status", "devtools status --json", "devtools status --verify-generated"),
        featured=True,
    ),
    CommandSpec("motd", "core", "Alias for `status`.", "devtools.project_motd"),
    CommandSpec(
        "render-all",
        "generated surfaces",
        "Refresh or verify generated docs and agent files.",
        "devtools.render_all",
        use_when="Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.",
        examples=("devtools render-all", "devtools render-all --check"),
        featured=True,
    ),
    CommandSpec(
        "render-agents",
        "generated surfaces",
        "Render AGENTS.md from CLAUDE.md and its included files.",
        "devtools.render_agents",
    ),
    CommandSpec(
        "render-cli-reference",
        "generated surfaces",
        "Render docs/cli-reference.md from live CLI help.",
        "devtools.render_cli_reference",
    ),
    CommandSpec(
        "render-cli-output-schemas",
        "generated surfaces",
        "Render JSON Schema artifacts for stable CLI output payloads under docs/schemas/cli-output/.",
        "devtools.render_cli_output_schemas",
        use_when=(
            "Refresh or verify published JSON Schemas after changing the surface payload models "
            "that back stable CLI JSON output (#1272)."
        ),
        examples=(
            "devtools render-cli-output-schemas",
            "devtools render-cli-output-schemas --check",
        ),
    ),
    CommandSpec(
        "render-openapi",
        "generated surfaces",
        "Render docs/openapi/search.yaml from the typed SearchEnvelope Pydantic models (#1266).",
        "devtools.render_openapi",
        use_when=(
            "Refresh or verify the published OpenAPI schema for the daemon HTTP search surface "
            "after changing SearchEnvelope or its referenced models."
        ),
        examples=(
            "devtools render-openapi",
            "devtools render-openapi --check",
        ),
    ),
    CommandSpec(
        "render-devtools-reference",
        "generated surfaces",
        "Render the command catalog inside docs/devtools.md.",
        "devtools.render_devtools_reference",
    ),
    CommandSpec(
        "render-docs-surface",
        "generated surfaces",
        "Render docs/README.md and the README documentation table.",
        "devtools.render_docs_surface",
    ),
    CommandSpec(
        "render-quality-reference",
        "generated surfaces",
        "Render docs/test-quality-workflows.md from live validation, mutation, and benchmark registries.",
        "devtools.render_quality_reference",
    ),
    CommandSpec(
        "render-pages",
        "generated surfaces",
        "Build the GitHub Pages documentation site into .cache/site/.",
        "devtools.render_pages",
        use_when="Build or verify the full GitHub Pages documentation site after changing docs, templates, or design docs.",
        examples=("devtools render-pages", "devtools render-pages --check", "devtools render-pages --serve"),
    ),
    CommandSpec(
        "verify",
        "verification",
        "Run the local verification baseline before pushing or creating a PR.",
        "devtools.verify",
        use_when="Run format, lint, mypy, render-all, and test checks locally before pushing.",
        examples=("devtools verify", "devtools verify --quick", "devtools verify --lab"),
        featured=True,
    ),
    CommandSpec(
        "release-readiness",
        "verification",
        "Validate the externally-presentable release gate definition.",
        "devtools.release_readiness",
        use_when=(
            "Check that the release-readiness gate document, required local commands, "
            "and release PR evidence template are still coherent before touching a release PR."
        ),
        examples=("devtools release-readiness", "devtools release-readiness --json"),
    ),
    CommandSpec(
        "test",
        "verification",
        "Run a focused pytest selection through the managed harness.",
        "devtools.run_tests",
        use_when="Run a specific test file, directory, or -k/-m selection in the inner loop without invoking raw pytest.",
        examples=(
            "devtools test tests/unit/pipeline",
            "devtools test -k hybrid",
            "devtools test tests/unit/storage -x",
        ),
        featured=True,
    ),
    CommandSpec(
        "coverage-gate",
        "verification",
        "Run pytest with the repository coverage floor from pyproject.toml.",
        "devtools.coverage_gate",
        use_when="Enforce the committed coverage ratchet locally or in CI without duplicating threshold values.",
        examples=(
            "devtools coverage-gate",
            "devtools coverage-gate --ignore-integration --term-missing",
            "devtools coverage-gate -- --maxfail=1",
        ),
    ),
    CommandSpec("run-validation-lanes", "verification", "Run named validation lanes.", "devtools.run_validation_lanes"),
    CommandSpec(
        "artifact-graph",
        "verification",
        "Render the runtime artifact, operation, and scenario-coverage map.",
        "devtools.artifact_graph",
        use_when="Inspect the authored runtime graph and see which scenarios currently cover declared artifacts and operations.",
        examples=(
            "devtools artifact-graph",
            "devtools artifact-graph --json",
            "devtools artifact-graph --strict",
        ),
    ),
    CommandSpec(
        "daemon-workload-probe",
        "verification",
        "Inspect daemon ingest workload, convergence debt, and hot query plans.",
        "devtools.daemon_workload_probe",
        use_when=(
            "Diagnose live-ingest residual work, read amplification, convergence debt, and planner regressions "
            "against a real archive without mutating daemon state."
        ),
        examples=(
            "devtools daemon-workload-probe",
            "devtools daemon-workload-probe --json",
            "devtools daemon-workload-probe --db /path/to/index.db --limit 10",
        ),
    ),
    CommandSpec(
        "ingest-amplification-probe",
        "verification",
        "Measure deterministic per-tier ingest write amplification on a synthetic fixture (#1851).",
        "devtools.ingest_amplification_probe",
        use_when=(
            "Establish or compare the post-fix baseline for daemon live-ingest write amplification. "
            "Drives the public batch-ingest path over a deterministic synthetic corpus in a temp dir "
            "and attributes bytes written per archive tier (source/index/embeddings/user/ops) "
            "per append batch. Additive measurement only — does not touch production ingest logic."
        ),
        examples=(
            "devtools ingest-amplification-probe",
            "devtools ingest-amplification-probe --json",
            "devtools ingest-amplification-probe --batches 8 --seed 1851",
        ),
    ),
    CommandSpec(
        "self-verify",
        "verification",
        "Capture and compare archive golden-master envelopes for schema rewrites.",
        "devtools.self_verify",
        use_when=(
            "Freeze v22 read-surface behavior before archive work, then compare candidate "
            "archives against the captured envelope baseline."
        ),
        examples=(
            "devtools self-verify capture --out .local/self-verify/v22.json",
            "devtools self-verify compare .local/self-verify/v22.json .local/self-verify/candidate.json --json",
        ),
    ),
    CommandSpec(
        "worktree-gc",
        "maintenance",
        "Safe worktree garbage collection — list and remove merged or abandoned git worktrees.",
        "devtools.worktree_gc",
        use_when=(
            "Clean up agent and feature worktrees that have been merged or whose branches "
            "have been deleted. Dry-run by default; pass --apply to remove safe candidates. "
            "Never removes dirty worktrees or the main worktree."
        ),
        examples=(
            "devtools worktree-gc",
            "devtools worktree-gc --json",
            "devtools worktree-gc --apply",
            "devtools worktree-gc --apply --force",
        ),
    ),
    CommandSpec(
        "archive-space-report",
        "maintenance",
        "Report SQLite archive file/page/object space by table and index.",
        "devtools.archive_space_report",
        use_when=(
            "Capture read-only dbstat evidence before/after schema rebuilds, VACUUM, "
            "blob cleanup, or storage-layout changes."
        ),
        examples=(
            "devtools archive-space-report",
            "devtools archive-space-report --json",
            "devtools archive-space-report --objects --db /path/to/index.db --limit 50",
        ),
    ),
    CommandSpec(
        "scenario-projections",
        "verification",
        "Render the authored scenario-bearing verification projections.",
        "devtools.scenario_projections",
        use_when="Inspect the unified projection inventory that feeds runtime coverage, generated docs, and control-plane maps.",
        examples=(
            "devtools scenario-projections",
            "devtools scenario-projections --source-kind exercise --artifact-target message_fts",
            "devtools scenario-projections --json",
        ),
    ),
    CommandSpec(
        "verify-topology",
        "verification",
        "Verify the realized polylogue tree against the topology projection.",
        "devtools.verify_topology",
        use_when=(
            "Detect orphans, conflicts, kernel-rule violations, or stale TBD cells against "
            "docs/plans/topology-target.yaml after moving files between packages."
        ),
        examples=(
            "devtools verify-topology",
            "devtools verify-topology --json",
            "devtools verify-topology --strict-tbd",
        ),
    ),
    CommandSpec(
        "build-topology-projection",
        "generated surfaces",
        "Generate docs/plans/topology-target.yaml from the current tree using placement rules.",
        "devtools.build_topology_projection",
        use_when=(
            "Refresh the topology projection after editing placement rules in this script "
            "or after a topology refactor lands."
        ),
        examples=("devtools build-topology-projection",),
    ),
    CommandSpec(
        "render-topology-status",
        "generated surfaces",
        "Render docs/topology-status.md from the topology projection and realized tree.",
        "devtools.render_topology_status",
        use_when=(
            "Refresh the topology drift dashboard after a refactor PR lands. "
            "Wired into devtools render-all so drift fails the generated-surface check."
        ),
        examples=("devtools render-topology-status", "devtools render-topology-status --check"),
    ),
    CommandSpec(
        "render-readme-media",
        "generated surfaces",
        "Generate README media assets (architecture diagrams, flowcharts) under docs/media/.",
        "devtools.generate_readme_media",
        use_when="Refresh architecture diagrams, data-flow charts, and provider-detection flowcharts for the README.",
        examples=(
            "devtools render-readme-media",
            "devtools render-readme-media --list",
            "devtools render-readme-media --name data-flow",
            "devtools render-readme-media --check",
        ),
    ),
    CommandSpec(
        "verify-test-coverage-contracts",
        "verification",
        "Verify every production module >150 AST lines has a matching test file or exemption.",
        "devtools.verify_test_coverage_contracts",
        use_when=(
            "Catches new production modules added without a dedicated test file. "
            "Existing modules are grandfathered via docs/plans/test-coverage-exemptions.yaml. "
            "Run as part of devtools verify --lab."
        ),
        examples=(
            "devtools verify-test-coverage-contracts",
            "devtools verify-test-coverage-contracts --json",
            "devtools verify-test-coverage-contracts --threshold 100",
        ),
    ),
    CommandSpec(
        "verify-closure-matrix",
        "verification",
        "Verify docs/plans/test-closure-matrix.yaml stays grounded in the realized tree.",
        "devtools.verify_closure_matrix",
        use_when=(
            "Keep the per-domain test-closure matrix honest — fails when a declared target file or "
            "representative test path is missing, or when a row violates the gate schema."
        ),
        examples=("devtools verify-closure-matrix", "devtools verify-closure-matrix --json"),
    ),
    CommandSpec(
        "verify-slos",
        "verification",
        "Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements.",
        "devtools.verify_slos",
        use_when=(
            "Run as part of devtools verify --lab, or directly to confirm read-surface "
            "(query / reader / facets / context / cost) latencies stay within their declared SLOs. "
            "Exits non-zero when any measured surface exceeds its budget."
        ),
        examples=(
            "devtools verify-slos",
            "devtools verify-slos --json",
            "devtools verify-slos --skip-benchmarks --json",
        ),
    ),
    CommandSpec(
        "verify-manifests",
        "verification",
        "Verify internal consistency across all docs/plans/*.yaml manifest files.",
        "devtools.verify_manifests",
        use_when=(
            "Catch malformed manifests, duplicate rule IDs, missing required fields, "
            "and cross-manifest reference inconsistencies."
        ),
        examples=("devtools verify-manifests",),
    ),
    CommandSpec(
        "verify-doc-commands",
        "verification",
        "Verify README/docs command examples resolve to live polylogue, polylogued, and devtools commands.",
        "devtools.verify_doc_commands",
        use_when=(
            "Catch doc drift away from the daemon-first command surface. "
            "Fails when README.md or any docs/**/*.md references a subcommand "
            "that is not registered, or a stale invocation like "
            "'polylogued run --enable-api' / 'polylogue run --source'."
        ),
        examples=("devtools verify-doc-commands", "devtools verify-doc-commands --json"),
    ),
    CommandSpec(
        "verify-ci-workflows",
        "verification",
        "Verify CI workflow files reference locally-known devtools commands and existing paths.",
        "devtools.verify_ci_workflows",
        use_when=(
            "Catch CI workflow files that reference unregistered devtools commands or "
            "non-existent paths. Checks only locally verifiable facts — not remote CI state."
        ),
        examples=("devtools verify-ci-workflows", "devtools verify-ci-workflows --json"),
    ),
    CommandSpec(
        "verify-test-infra-currency",
        "verification",
        "Verify tests/infra/ helpers reference only tables that exist in the current SCHEMA_VERSION.",
        "devtools.verify_test_infra_currency",
        use_when=(
            "Catch helpers that target renamed or removed tables (#1208). "
            "When SCHEMA_VERSION bumps, helper SQL drifting away from the live "
            "schema is invisible to testmon-selected runs until an unrelated change "
            "invalidates the affected tests."
        ),
        examples=("devtools verify-test-infra-currency", "devtools verify-test-infra-currency --json"),
    ),
    CommandSpec(
        "verify-schema-upgrade-lane",
        "verification",
        "Reject in-place storage schema upgrade helpers (#1302).",
        "devtools.verify_schema_upgrade_lane",
        use_when=(
            "Enforce the policy boundary documented in docs/internals.md § "
            "'Schema Versioning Model'. Polylogue intentionally has no in-place "
            "storage schema upgrade chain; archive-shape changes edit the canonical "
            "DDL and require a fresh rebuild from source."
        ),
        examples=("devtools verify-schema-upgrade-lane", "devtools verify-schema-upgrade-lane --json"),
    ),
    CommandSpec(
        "verify-test-clock-hygiene",
        "verification",
        "Verify test files use the frozen_clock fixture instead of reading the host wall clock (#1300).",
        "devtools.verify_test_clock_hygiene",
        use_when=(
            "Block new direct calls to datetime.now / datetime.utcnow / "
            "time.time / time.monotonic from test files outside the "
            "allowlist in docs/plans/test-clock-allowlist.yaml. Tests that "
            "genuinely need the host clock add their path to the allowlist "
            "with a one-line rationale."
        ),
        examples=("devtools verify-test-clock-hygiene", "devtools verify-test-clock-hygiene --json"),
    ),
    CommandSpec(
        "verify-distribution-surface",
        "verification",
        "Verify wheel/sdist installed artifacts expose only supported runtime entrypoints.",
        "devtools.verify_distribution_surface",
        use_when=(
            "Build wheel and sdist artifacts, rebuild a wheel from an unpacked sdist without .git, "
            "and smoke installed runtime console scripts."
        ),
        examples=("devtools verify-distribution-surface",),
    ),
    CommandSpec(
        "pipeline-probe",
        "verification",
        "Run typed pipeline probes against synthetic, staged, or archive-subset inputs.",
        "devtools.pipeline_probe",
        use_when="Exercise real pipeline stages and optionally capture emitted summaries as regression cases.",
        examples=(
            "devtools pipeline-probe --provider chatgpt --stage parse",
            "devtools pipeline-probe --input-mode archive-subset --capture-regression live-parse-drift",
        ),
    ),
    CommandSpec(
        "query-memory-budget",
        "verification",
        "Measure query-memory envelopes on generated fixtures.",
        "devtools.query_memory_budget",
        use_when="Assert memory budgets around a concrete query or archive-facing command.",
        examples=("devtools query-memory-budget --max-rss-mb 1536 -- polylogue --plain stats",),
    ),
    CommandSpec(
        "lab-scenario",
        "verification",
        "Run verification-lab showcase scenario sets and baseline checks.",
        "devtools.lab_scenario",
        use_when="Run showcase exercise smoke scenarios and committed baseline checks outside the archive CLI.",
        examples=(
            "devtools lab-scenario list",
            "devtools lab-scenario run archive-smoke --tier 0",
            "devtools lab-scenario verify-baselines",
        ),
    ),
    CommandSpec(
        "schema-generate",
        "verification",
        "Generate provider schema packages and optional evidence clusters.",
        "devtools.schema_generate",
        use_when="Refresh provider schema package artifacts from archive observations outside the archive CLI.",
        examples=("devtools schema-generate --provider chatgpt --cluster",),
    ),
    CommandSpec(
        "schema-promote",
        "verification",
        "Promote a schema evidence cluster into a registered package version.",
        "devtools.schema_promote",
        use_when="Turn reviewed schema evidence clusters into committed provider schema packages.",
        examples=("devtools schema-promote --provider chatgpt --cluster chatgpt-message-v2",),
    ),
    CommandSpec(
        "schema-audit",
        "verification",
        "Run committed provider schema package quality checks.",
        "devtools.schema_audit",
        use_when="Check committed schema package quality gates without presenting them as normal archive usage.",
        examples=("devtools schema-audit --provider chatgpt --json",),
    ),
    CommandSpec(
        "verify-schema-roundtrip",
        "verification",
        "Verify committed provider schema packages reload and roundtrip cleanly.",
        "devtools.verify_schema_roundtrip",
        use_when=(
            "Close the schema inference-validation loop: package manifests must roundtrip through typed models, "
            "and every supported element schema must be reachable from the runtime registry."
        ),
        examples=(
            "devtools verify-schema-roundtrip --provider chatgpt",
            "devtools verify-schema-roundtrip --all --json",
        ),
    ),
    CommandSpec(
        "regression-capture",
        "verification",
        "Capture pipeline-probe summaries as durable local regression cases.",
        "devtools.regression_capture",
        use_when="Turn a live or probe failure JSON summary into a replayable local regression artifact.",
        examples=(
            "devtools regression-capture --input probe.json --name parse-drift",
            "devtools pipeline-probe --json | devtools regression-capture --name parse-drift --tag live",
        ),
    ),
    CommandSpec(
        "verify-layering",
        "verification",
        "Check inter-package imports against declared layering rules from docs/plans/layering.yaml.",
        "devtools.verify_layering",
        use_when=(
            "Diagnose architecture drift: which files import across declared "
            "package boundaries. This runs in verify --quick."
        ),
        examples=("devtools verify-layering", "devtools verify-layering --json"),
    ),
    CommandSpec(
        "evidence-dashboard",
        "verification",
        "Render the pytest-first evidence dashboard or a changed-path trace.",
        "devtools.evidence_dashboard",
        use_when=(
            "Inspect pytest health, contract-evidence inventory, coverage, SLO "
            "catalog, static-gate status, and campaign freshness, or "
            "trace which evidence artifacts cover the changed paths in a PR."
        ),
        examples=(
            "devtools evidence-dashboard --json",
            "devtools evidence-dashboard --markdown",
            "devtools evidence-dashboard trace --base origin/master --head HEAD --markdown",
        ),
    ),
    CommandSpec(
        "build-package",
        "maintenance",
        "Build the default Nix package with the out-link under .local/result.",
        "devtools.build_package",
        use_when="Produce the Nix package artifact with its out-link kept under the repo-local output root.",
        examples=("devtools build-package",),
    ),
    CommandSpec(
        "mutmut-campaign",
        "campaigns",
        "Run focused mutation campaigns and maintain their local index.",
        "devtools.mutmut_campaign",
        use_when="Run or inspect focused mutation-testing work without shrinking the committed mutmut scope.",
        examples=("devtools mutmut-campaign list", "devtools mutmut-campaign run filters"),
        featured=True,
    ),
    CommandSpec(
        "benchmark-campaign",
        "campaigns",
        "Run or compare benchmark campaigns.",
        "devtools.benchmark_campaign",
        use_when="Record durable benchmark artifacts or compare a candidate run against a baseline artifact.",
        examples=(
            "devtools benchmark-campaign list",
            "devtools benchmark-campaign run search-filters",
            "devtools benchmark-campaign compare baseline.json candidate.json",
        ),
        featured=True,
    ),
    CommandSpec(
        "run-benchmark-campaigns",
        "campaigns",
        "Run synthetic benchmark campaigns over generated archives.",
        "devtools.run_campaign",
    ),
    CommandSpec(
        "xtask",
        "maintenance",
        "Record and query agent task execution history (.agent/xtask/tasks.jsonl).",
        "devtools.xtask",
        use_when="Log, view recent, or summarize agent task execution history during development sessions.",
        examples=(
            "devtools xtask log --command 'devtools render-all' --duration-ms 3200 --exit-code 0",
            "devtools xtask recent",
            "devtools xtask recent --count 20",
            "devtools xtask stats",
            "devtools xtask stats --json",
        ),
    ),
    CommandSpec(
        "failure-context",
        "maintenance",
        "Join testmon, git history, and fixtures for a pytest failure ID into a JSON envelope.",
        "devtools.failure_context",
        use_when=(
            "Bootstrap an agent inner-loop debugging session for a failing test — surfaces production "
            "files the test depends on, their recent commits, and fixtures the test uses, "
            "all in one structured envelope."
        ),
        examples=(
            "devtools failure-context tests/unit/storage/test_foo.py::test_bar",
            "devtools failure-context tests/unit/storage/test_foo.py::test_bar --days 14",
        ),
    ),
    CommandSpec(
        "verify-lane-assertions",
        "verification",
        "Verify scenario lanes classified as SEMANTIC_OUTPUT carry semantic assertions.",
        "devtools.verify_lane_assertions",
        use_when="Catch vacuous semantic lanes that claim evidence but only check exit codes.",
        examples=("devtools verify-lane-assertions",),
    ),
)

COMMANDS: dict[str, CommandSpec] = {spec.name: spec for spec in COMMAND_SPECS}


def control_plane_command(*args: str) -> str:
    parts = [CONTROL_PLANE, *args]
    return " ".join(part for part in parts if part)


def control_plane_argv(*args: str) -> tuple[str, ...]:
    return tuple(part for part in (CONTROL_PLANE, *args) if part)


def featured_command_specs(commands: Iterable[CommandSpec] = COMMAND_SPECS) -> tuple[CommandSpec, ...]:
    return tuple(spec for spec in commands if spec.featured)


def verification_lab_command_specs(commands: Iterable[CommandSpec] = COMMAND_SPECS) -> tuple[CommandSpec, ...]:
    by_name = {spec.name: spec for spec in commands}
    return tuple(by_name[name] for name in VERIFICATION_LAB_COMMAND_NAMES)


def grouped_command_specs(commands: Iterable[CommandSpec] = COMMAND_SPECS) -> OrderedDict[str, list[CommandSpec]]:
    grouped: OrderedDict[str, list[CommandSpec]] = OrderedDict((category, []) for category in CATEGORY_ORDER)
    for spec in commands:
        grouped.setdefault(spec.category, [])
        grouped[spec.category].append(spec)
    for _category, specs in grouped.items():
        specs.sort(key=lambda item: item.name)
    return OrderedDict((category, specs) for category, specs in grouped.items() if specs)


__all__ = [
    "CATEGORY_ORDER",
    "COMMANDS",
    "COMMAND_SPECS",
    "CONTROL_PLANE",
    "CommandMain",
    "CommandSpec",
    "control_plane_argv",
    "control_plane_command",
    "featured_command_specs",
    "grouped_command_specs",
    "VERIFICATION_LAB_COMMAND_NAMES",
    "verification_lab_command_specs",
]
