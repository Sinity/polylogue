"""Shared command catalog for repository developer tools."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

CommandMain = Callable[[list[str] | None], int]
CONTROL_PLANE = "devtools"
VERIFICATION_LAB_COMMAND_NAMES: tuple[str, ...] = (
    "lab graph",
    "lab lanes",
    "lab policy backlog-hygiene",
    "lab policy demo-packet-registry",
    "lab policy demo-tour-freshness",
    "lab policy docs-drift",
    "lab policy insight-honesty",
    "lab policy schema-versioning",
    "lab policy timestamp-doctrine",
    "lab provider completeness",
    "lab probe bead-pr-reconciliation",
    "lab probe capture-regression",
    "lab probe cost-reconciliation",
    "lab probe pipeline",
    "lab probe turso",
    "lab projections",
    "lab smoke",
    "lab schema audit",
    "lab schema compare",
    "lab schema explain",
    "lab schema generate",
    "lab schema list",
    "lab schema promote",
    "lab schema roundtrip",
    "lab snapshot read-surface",
    "lab test-economics",
    "lab testmon-proof",
)

CATEGORY_ORDER: tuple[str, ...] = (
    "core",
    "generated surfaces",
    "release",
    "verification lab",
    "verification",
    "benchmarking",
    "workspace",
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
    def command_path(self) -> tuple[str, ...]:
        return tuple(part for part in self.name.split(" ") if part)

    @property
    def invocation(self) -> str:
        return control_plane_command(*self.command_path)

    @property
    def argv(self) -> tuple[str, ...]:
        return control_plane_argv(*self.command_path)

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
    CommandSpec(
        "render all",
        "generated surfaces",
        "Refresh or verify generated docs and agent files.",
        "devtools.render_all",
        use_when="Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.",
        examples=("devtools render all", "devtools render all --check"),
        featured=True,
    ),
    CommandSpec(
        "render cli-reference",
        "generated surfaces",
        "Render docs/cli-reference.md from live CLI help.",
        "devtools.render_cli_reference",
    ),
    CommandSpec(
        "render cli-output-schemas",
        "generated surfaces",
        "Render JSON Schema artifacts for stable CLI output payloads under docs/schemas/cli-output/.",
        "devtools.render_cli_output_schemas",
        use_when=(
            "Refresh or verify published JSON Schemas after changing the surface payload models "
            "that back stable CLI JSON output (#1272)."
        ),
        examples=(
            "devtools render cli-output-schemas",
            "devtools render cli-output-schemas --check",
        ),
    ),
    CommandSpec(
        "render openapi",
        "generated surfaces",
        "Render docs/openapi/search.yaml from typed daemon query payload models.",
        "devtools.render_openapi",
        use_when=(
            "Refresh or verify the published OpenAPI schema for daemon HTTP query routes "
            "after changing a route handler or a shared surface payload model."
        ),
        examples=(
            "devtools render openapi",
            "devtools render openapi --check",
        ),
    ),
    CommandSpec(
        "render webui-design-system",
        "generated surfaces",
        "Render WebUI v2 CSS tokens, public badge contracts, and contrast evidence.",
        "devtools.render_webui_design_system",
        use_when=(
            "Refresh or verify browser design-system contracts after changing the Python theme "
            "palette, the public Origin enum, or evidence-state vocabulary."
        ),
        examples=(
            "devtools render webui-design-system",
            "devtools render webui-design-system --check",
        ),
    ),
    CommandSpec(
        "render webui-client",
        "generated surfaces",
        "Render the committed WebUI TypeScript client from docs/openapi/search.yaml.",
        "devtools.render_webui_client",
        use_when=(
            "Refresh or verify WebUI request/response types and continuation iterators after changing "
            "the generated daemon OpenAPI contract."
        ),
        examples=(
            "devtools render webui-client",
            "devtools render webui-client --check",
        ),
    ),
    CommandSpec(
        "render devtools-reference",
        "generated surfaces",
        "Render the command catalog inside docs/devtools.md.",
        "devtools.render_devtools_reference",
    ),
    CommandSpec(
        "render demo-corpus-datasheet",
        "generated surfaces",
        "Render docs/plans/demo-corpus-construct-audit.md from the demo family registry and a measured seed archive.",
        "devtools.render_demo_corpus_datasheet",
        use_when="Refresh or verify the deterministic demo corpus construct datasheet after changing demo families, constructs, or seed semantics.",
        examples=(
            "devtools render demo-corpus-datasheet",
            "devtools render demo-corpus-datasheet --check",
        ),
    ),
    CommandSpec(
        "render docs-surface",
        "generated surfaces",
        "Render docs/README.md and the README documentation table.",
        "devtools.render_docs_surface",
    ),
    CommandSpec(
        "render quality-reference",
        "generated surfaces",
        "Render docs/test-quality-workflows.md from executable lane, mutation, and benchmark registries.",
        "devtools.render_quality_reference",
    ),
    CommandSpec(
        "render query-discovery",
        "generated surfaces",
        "Render parser-gated query discovery examples and result semantics into docs/search.md.",
        "devtools.render_query_discovery",
        use_when=(
            "Refresh or verify query examples after changing the expression grammar, query-unit metadata, "
            "result-semantics vocabulary, completions, or MCP cookbook recipes."
        ),
        examples=("devtools render query-discovery", "devtools render query-discovery --check"),
    ),
    CommandSpec(
        "render mcp-equivalence",
        "generated surfaces",
        "Render docs/generated/mcp-equivalence.json from executable MCP declarations.",
        "devtools.render_mcp_equivalence",
        use_when=(
            "Refresh or verify MCP discovery names, input/output contracts, role gates, operation owners, "
            "Python parity expectations, and disjoint migration ownership after changing the compatibility surface."
        ),
        examples=("devtools render mcp-equivalence", "devtools render mcp-equivalence --check"),
    ),
    CommandSpec(
        "render mcp-tool-index",
        "generated surfaces",
        "Render the generated exhaustive tool-name appendix into docs/mcp-reference.md.",
        "devtools.render_mcp_tool_index",
        use_when=(
            "Keep every registered MCP tool name individually reachable from the docs tree "
            "(tests/infra/mcp.py:EXPECTED_TOOL_NAMES) after adding or removing a tool, so "
            "`devtools verify docs-coverage` stays clean without hand-duplicating the list."
        ),
        examples=("devtools render mcp-tool-index", "devtools render mcp-tool-index --check"),
    ),
    CommandSpec(
        "render product-workflows",
        "generated surfaces",
        "Render docs/product/workflows.md from executable query-action workflow registries.",
        "devtools.render_product_workflows",
        use_when=(
            "Refresh or verify the product query-action workflow contract after changing action contracts, "
            "read-view surfaces, completion behavior, or workflow golden paths (#2305)."
        ),
        examples=(
            "devtools render product-workflows",
            "devtools render product-workflows --check",
        ),
    ),
    CommandSpec(
        "render pages",
        "generated surfaces",
        "Build the GitHub Pages documentation site into .cache/site/.",
        "devtools.render_pages",
        use_when="Build or verify the full GitHub Pages documentation site after changing docs, templates, or design docs.",
        examples=("devtools render pages", "devtools render pages --check", "devtools render pages --serve"),
    ),
    CommandSpec(
        "render visual-tapes",
        "generated surfaces",
        "Write VHS tape files and optionally capture GIFs for the default visual evidence specs.",
        "devtools.render_visual_tapes",
        use_when="Regenerate the first-contact demo screencast media from the committed tape specs.",
        examples=(
            "devtools render visual-tapes",
            "devtools render visual-tapes --capture",
            "devtools render visual-tapes --check",
        ),
    ),
    CommandSpec(
        "verify",
        "verification",
        "Run the local verification baseline before pushing or creating a PR.",
        "devtools.verify",
        use_when="Run format, lint, mypy, render all, and test checks locally before pushing.",
        examples=("devtools verify", "devtools verify --quick", "devtools verify --lab"),
        featured=True,
    ),
    CommandSpec(
        "verify public-claims",
        "verification",
        "Validate public claims, evidence paths, Beads owners, and retired copy.",
        "devtools.public_claims",
        use_when=(
            "Check externally visible claims after changing README, demos, findings, proof artifacts, "
            "or the claims ledger."
        ),
        examples=(
            "devtools verify public-claims",
            "devtools verify public-claims --json",
        ),
    ),
    CommandSpec(
        "release readiness",
        "release",
        "Validate the externally-presentable release gate definition.",
        "devtools.release_readiness",
        use_when=(
            "Check that the release-readiness gate document, required local commands, "
            "and release PR evidence template are still coherent before touching a release PR."
        ),
        examples=(
            "devtools release readiness",
            "devtools release readiness --json",
            "devtools release readiness --release-body-file /tmp/release-pr-body.md",
        ),
    ),
    CommandSpec(
        "lab provider completeness",
        "verification lab",
        "Report provider/importer package completeness by origin and capture mode.",
        "devtools.provider_completeness",
        use_when=(
            "Inspect detector, parser, fixture, schema, docs, ImportExplain, and caveat coverage "
            "before claiming a provider/importer mode is product-ready."
        ),
        examples=(
            "devtools lab provider completeness",
            "devtools lab provider completeness --json",
            "devtools lab provider completeness --origin codex-session --json",
            "devtools lab provider completeness --check",
        ),
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
        "verify coverage",
        "verification",
        "Run pytest with the repository coverage floor from pyproject.toml.",
        "devtools.coverage_gate",
        use_when="Enforce the committed coverage ratchet locally or in CI without duplicating threshold values.",
        examples=(
            "devtools verify coverage",
            "devtools verify coverage --ignore-integration --term-missing",
            "devtools verify coverage -- --maxfail=1",
        ),
    ),
    CommandSpec(
        "lab lanes",
        "verification lab",
        "Run named validation lanes.",
        "devtools.run_validation_lanes",
        use_when="List, dry-run, or execute authored validation lanes from the executable lane registry.",
        examples=(
            "devtools lab lanes --list",
            "devtools lab lanes --lane frontier-local",
            "devtools lab lanes --lane live-archive-smoke --dry-run",
        ),
    ),
    CommandSpec(
        "lab graph",
        "verification lab",
        "Render the runtime artifact, operation, and scenario-coverage map.",
        "devtools.artifact_graph",
        use_when="Inspect the authored runtime graph and see which scenarios currently cover declared artifacts and operations.",
        examples=(
            "devtools lab graph",
            "devtools lab graph --json",
            "devtools lab graph --strict",
        ),
    ),
    CommandSpec(
        "lab test-economics",
        "verification lab",
        "Report per-package coverage/fix-density/test-cost economics (polylogue-9e5.11).",
        "devtools.test_economics_report",
        use_when=(
            "Decide where test-writing effort or test-suite pruning actually pays off, by "
            "cross-referencing coverage percent, historical fix-commit density, testmon "
            "wall-time cost exposure, and testmon selection fan-out per top-level package."
        ),
        examples=(
            "devtools lab test-economics",
            "devtools lab test-economics --json",
            "devtools lab test-economics --write docs/test-economics.md",
        ),
    ),
    CommandSpec(
        "lab testmon-proof",
        "verification lab",
        "Prove real testmon affected selection against a semantic production mutation.",
        "devtools.testmon_mutation_proof",
        use_when=(
            "Validate the affected-test harness itself: a disposable copy of a real Polylogue module "
            "and existing route test is seeded, semantically mutated, edge-severed, restored, and checked "
            "for bounded unrelated-change selection."
        ),
        examples=("devtools lab testmon-proof", "devtools lab testmon-proof --json"),
    ),
    CommandSpec(
        "lab pytest-witness-repetitions",
        "verification lab",
        "Repeat the exact optimize, WAL, and embedding seed-hang witnesses with durable receipts.",
        "devtools.pytest_witness_repetitions",
        use_when=(
            "Establish that the historical periodic optimize, WAL checkpoint, and embedding backlog "
            "lifecycle witnesses survive consecutive isolated and xdist runs. Each attempt uses the ordinary "
            "managed pytest/supervisor path; failures and timeouts are retained rather than retried."
        ),
        examples=(
            "devtools lab pytest-witness-repetitions",
            "devtools lab pytest-witness-repetitions --attempts 2 --xdist-workers 3 --json",
        ),
    ),
    CommandSpec(
        "bench ingest-amplification",
        "benchmarking",
        "Measure deterministic per-tier ingest write amplification on a synthetic fixture (#1851).",
        "devtools.ingest_amplification_probe",
        use_when=(
            "Establish or compare the post-fix baseline for daemon live-ingest write amplification. "
            "Drives the public batch-ingest path over a deterministic synthetic corpus in a temp dir "
            "and attributes bytes written per archive tier (source/index/embeddings/user/ops) "
            "per append batch. Additive measurement only — does not touch production ingest logic."
        ),
        examples=(
            "devtools bench ingest-amplification",
            "devtools bench ingest-amplification --json",
            "devtools bench ingest-amplification --batches 8 --seed 1851",
        ),
    ),
    CommandSpec(
        "bench ingest-throughput",
        "benchmarking",
        "Measure ingest wall-clock throughput on a synthetic fixture.",
        "devtools.ingest_throughput_probe",
        use_when=(
            "Measure ingest wall-clock / throughput, the time-based counterpart to the "
            "bytes-based ingest-amplification probe. Drives the public batch-ingest path over "
            "a deterministic synthetic corpus in a temp dir and times each append batch, "
            "reporting messages/sessions per second and a per-batch-ms distribution. "
            "Wall-clock is host-variable: diagnostic and campaign-comparable, no CI thresholds. "
            "Additive measurement only — does not touch production ingest logic."
        ),
        examples=(
            "devtools bench ingest-throughput",
            "devtools bench ingest-throughput --json",
            "devtools bench ingest-throughput --batches 20 --seed 2391",
        ),
    ),
    CommandSpec(
        "bench coordination-latency",
        "benchmarking",
        "Measure compact coordination status p50/p95 with raw stage samples.",
        "devtools.coordination_latency_probe",
        examples=("devtools bench coordination-latency --samples 21 --out .local/coordination-latency.json",),
    ),
    CommandSpec(
        "lab snapshot read-surface",
        "verification lab",
        "Capture and compare archive read-surface snapshots.",
        "devtools.self_verify",
        use_when=(
            "Freeze archive read-surface behavior before archive work, then compare candidate "
            "archives against the captured envelope baseline."
        ),
        examples=(
            "devtools lab snapshot read-surface capture --out .local/self-verify/baseline.json",
            "devtools lab snapshot read-surface compare .local/self-verify/baseline.json .local/self-verify/candidate.json --json",
        ),
    ),
    CommandSpec(
        "workspace index-fast-forward",
        "workspace",
        "Apply a declared clone-first index.db fast-forward with receipts and rollback.",
        "devtools.index_fast_forward",
        use_when=(
            "Upgrade a quiesced supported derived index without raw replay: reflink an inactive generation, "
            "apply declared canonical schema/FTS deltas, validate structural equivalence on the clone, "
            "then separately activate or roll back. Semantic-reparse deltas deliberately require rebuild/reprocess."
        ),
        examples=(
            "devtools workspace index-fast-forward plan --source /path/to/index.db",
            "devtools workspace index-fast-forward clone-upgrade --source /path/to/index.db --receipt /path/to/receipt.json",
            "devtools workspace index-fast-forward activate --receipt /path/to/receipt.json --restart",
            "devtools workspace index-fast-forward rollback --receipt /path/to/receipt.json --restart",
        ),
    ),
    CommandSpec(
        "workspace archive-schema-fast-forward",
        "workspace",
        "Clone-forward the v35 archive tiers without raw replay.",
        "devtools.archive_schema_fast_forward",
        use_when=(
            "Advance a stopped, verified v35 archive through the declared source/user durable migrations "
            "and derived v36/index plus v2/embeddings clones. The existing verified backup remains valid: "
            "the durable runner verifies active path plus bytes/version rather than inode identity."
        ),
        examples=(
            "devtools workspace archive-schema-fast-forward prepare --archive-root /realm/db/polylogue --staging-root /realm/staging/polylogue-schema-forward --receipt /realm/staging/polylogue-schema-forward/receipt.json --backup-manifest /realm/staging/verified/manifest.json",
            "devtools workspace archive-schema-fast-forward activate --receipt /realm/staging/polylogue-schema-forward/receipt.json --backup-manifest /realm/staging/verified/manifest.json",
        ),
    ),
    CommandSpec(
        "workspace index-v37-fast-forward",
        "workspace",
        "Clone-forward index v36 to v37 by retiring derived caches without raw replay.",
        "devtools.index_v37_fast_forward",
        use_when=(
            "Advance a stopped exact-shape v36 index to v37. The actuator reflink-clones the active generation, "
            "drops only the three retired run-projection caches, proves surviving schema and row-count parity, "
            "then separately atomically activates the proven generation."
        ),
        examples=(
            "devtools workspace index-v37-fast-forward prepare --archive-root /path/to/archive --receipt /path/to/receipt.json",
            "devtools workspace index-v37-fast-forward activate --receipt /path/to/receipt.json",
        ),
    ),
    CommandSpec(
        "workspace worktree-gc",
        "workspace",
        "Safe worktree garbage collection — list and remove merged, squash-equivalent, or abandoned git worktrees.",
        "devtools.worktree_gc",
        use_when=(
            "Clean up agent and feature worktrees that have been merged or whose branches "
            "have been deleted. Also recognizes fully patch-equivalent squash-merged branches via git cherry. "
            "Dry-run by default; pass --apply to remove safe candidates. "
            "Never removes dirty worktrees or the main worktree."
        ),
        examples=(
            "devtools workspace worktree-gc",
            "devtools workspace worktree-gc --json",
            "devtools workspace worktree-gc --apply",
            "devtools workspace worktree-gc --apply --force",
        ),
    ),
    CommandSpec(
        "demo real-slice-screen",
        "workspace",
        "Read-only extraction + privacy screening of a candidate real-archive session slice.",
        "devtools.proof_world_real_slice",
        use_when=(
            "Assembling a candidate real-archive slice for the shared demo proof world "
            "(polylogue-212.11): pulls sessions read-only via the Polylogue API, flattens them "
            "to text, and screens for secret/credential and PII-adjacent patterns before any "
            "operator decides to fold the slice into a shared fixture. Never mutates the source "
            "archive and never writes into polylogue/scenarios/ on its own."
        ),
        examples=(
            "devtools demo real-slice-screen --archive-root /realm/db/polylogue "
            "--session claude-code-session:<id>:<agent> --out .agent/scratch/real-slice",
            "devtools demo real-slice-screen --archive-root /realm/db/polylogue "
            "--refs-file refs.txt --out .agent/scratch/real-slice",
        ),
    ),
    CommandSpec(
        "workspace dev-loop",
        "workspace",
        "Preflight branch-local daemon, web-shell, and browser-capture development loops.",
        "devtools.dev_loop",
        use_when=(
            "Before running a branch-local polylogued/web-shell/browser-capture loop, "
            "check whether the deployed user service is active, which ports are occupied, "
            "and which isolated archive/log paths this checkout should use."
        ),
        examples=(
            "devtools workspace dev-loop",
            "devtools workspace dev-loop --json",
            "devtools workspace dev-loop --prepare --api-port 8876 --browser-capture-port 8875",
            "devtools workspace dev-loop --api-port 8876 --browser-capture-port 8875 --launch-daemon",
            "devtools workspace dev-loop --capture-cli -- polylogue ops status",
            "devtools workspace dev-loop --receiver-smoke --json",
            "devtools workspace dev-loop --extension-smoke --json",
            "devtools workspace dev-loop --browser-smoke --json",
            "devtools workspace dev-loop --browser-provider-smoke --json",
            "devtools workspace dev-loop --browser-plan --json",
            "devtools workspace dev-loop --browser-live-proof --browser-live-profile-dir .local/browser-profiles/<copy> --json",
            "devtools workspace dev-loop --tui-plan --json",
            "devtools workspace dev-loop --inspect-run .cache/dev-loop/<run-id> --json",
        ),
    ),
    CommandSpec(
        "workspace frontier",
        "workspace",
        "Classify ready and in-progress Beads into devloop batches.",
        "devtools.frontier_report",
        use_when=(
            "During Direction, Velocity, or wait-ahead windows, group the Beads frontier by subsystem, "
            "proof cost, live-runtime risk, schema-lane conflict, and subagent suitability before claiming "
            "or dispatching work."
        ),
        examples=(
            "devtools workspace frontier",
            "devtools workspace frontier --json",
            "devtools workspace frontier --limit 80 --out .agent/task-history/frontier-latest.md",
        ),
    ),
    CommandSpec(
        "workspace deployment-smoke",
        "workspace",
        "Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow.",
        "devtools.deployment_smoke",
        use_when=(
            "After a system rebuild or before live UI probing, verify that the systemwide "
            "polylogue/polylogued binaries, loopback daemon routes, browser-capture receiver, "
            "and browser-capture archive materialization match the expected deployed surface."
        ),
        examples=(
            "devtools workspace deployment-smoke",
            "devtools workspace deployment-smoke --json",
            "devtools workspace deployment-smoke --browser --browser-executable /etc/profiles/per-user/sinity/bin/google-chrome",
            "devtools workspace deployment-smoke --daemon-url http://127.0.0.1:8766 --receiver-url http://127.0.0.1:8765",
        ),
    ),
    CommandSpec(
        "workspace scale-regression",
        "workspace",
        "Run the seeded large-archive scale-regression probe.",
        "devtools.scale_regression_probe",
        use_when=(
            "Before closing scale-hardening work, seed a small archive with large-archive-shaped edge cases "
            "and assert chunked insight rebuilds, bounded giant-session profiles, raw-materialization debt "
            "visibility, reset source preservation, and run-ref no-drop invariants."
        ),
        examples=(
            "devtools workspace scale-regression",
            "devtools workspace scale-regression --json",
            "devtools workspace scale-regression --workdir .cache/scale-regression --keep --json",
        ),
    ),
    CommandSpec(
        "workspace raw-authority-scale-proof",
        "workspace",
        "Run bounded raw-authority replay to a two-census fixed point.",
        "devtools.raw_authority_scale_proof",
        use_when=(
            "Generate a receipt-backed raw-authority scenario before a live replay gate. "
            "The command defaults to the July-15 topology: 15,264 direct candidates and "
            "21,398 expanded memberships."
        ),
        examples=(
            "devtools workspace raw-authority-scale-proof --json",
            "devtools workspace raw-authority-scale-proof --components 10163 --raws 15264 --expanded-raws 21398 --pass-limit 15264 --keep --json",
        ),
    ),
    CommandSpec(
        "workspace temporal-read-profile",
        "workspace",
        "Measure read --view temporal phase timings on the active archive.",
        "devtools.temporal_read_profile",
        use_when=(
            "Profile the shared temporal read-view builder before tuning query, projection, or rendering paths. "
            "The command emits phase timings plus the temporal window summary and can write the report as a "
            "dogfood/demo artifact."
        ),
        examples=(
            "devtools workspace temporal-read-profile --query repo:polylogue --limit 1 --json",
            "devtools workspace temporal-read-profile --query 'repo:polylogue devloop' --limit 3 --out .local/temporal-profile.json",
        ),
    ),
    CommandSpec(
        "workspace temporal-devloop",
        "workspace",
        "Compose git and operating-log events into a temporal evidence window.",
        "devtools.devloop_temporal",
        use_when=(
            "Dogfood temporal analysis on the current devloop without inventing a bespoke report shape: "
            "git commits and OPERATING-LOG headings are normalized as event families and projected through "
            "the shared TemporalEvidenceWindow."
        ),
        examples=(
            "devtools workspace temporal-devloop --since 2026-06-30T00:00:00+02:00 --json",
            "devtools workspace temporal-devloop --out .agent/demos/14-devloop-temporal-dogfood/devloop-events.json",
        ),
    ),
    CommandSpec(
        "workspace temporal-archive-aggregates",
        "workspace",
        "Build run-projection aggregate artifacts from the active archive.",
        "devtools.temporal_archive_aggregates",
        use_when=(
            "Refresh private longitudinal run/observed-event/context-snapshot demo artifacts from "
            "the canonical archive through one reusable command instead of copying raw sqlite3 "
            "queries into README files."
        ),
        examples=(
            "devtools workspace temporal-archive-aggregates --json",
            "devtools workspace temporal-archive-aggregates --out-dir .agent/demos/01-real-archive-temporal-devloops",
        ),
    ),
    CommandSpec(
        "workspace lineage-validation",
        "workspace",
        "Validate lineage-count evidence before citing archive counts externally.",
        "devtools.lineage_validation",
        use_when=(
            "Before publishing archive session/message/cardinality numbers, emit exact physical/logical counts, "
            "session-link inheritance rollups, branch-point integrity checks, and sampled composed-read proof "
            "from the active archive instead of relying on scratch SQL or planner-estimated diagnostics."
        ),
        examples=(
            "devtools workspace lineage-validation --json",
            "devtools workspace lineage-validation --sample-prefix-sharing 100 --json",
            "devtools workspace lineage-validation --out-dir .agent/demos/lineage-validation/current",
        ),
    ),
    CommandSpec(
        "workspace affordance-usage",
        "workspace",
        "Analyze agent affordance/tool usage from archive tool-use rows.",
        "devtools.affordance_usage",
        use_when=(
            "Dogfood agent-tool usage analysis without hand-written SQL: summarize tool families, raw tool names, "
            "origin splits, recent adoption windows, structured failure rates, and representative samples."
        ),
        examples=(
            "devtools workspace affordance-usage --days 7 --json",
            "devtools workspace affordance-usage --detail-pattern codebase-memory --detail-pattern search_code --days 30",
            "devtools workspace affordance-usage --out-dir .agent/demos/agent-affordance-usage",
        ),
    ),
    CommandSpec(
        "workspace demo-shelf",
        "workspace",
        "Refresh or verify current demo shelf indexes.",
        "devtools.demo_shelf",
        use_when=(
            "After curating .agent/demos or another explicit current demo shelf, regenerate "
            "MANIFEST.readable.json and SUMMARY_INDEX.json from one deterministic helper. "
            "Declarative read packages own portable readable bundles; the shelf is the best current set, "
            "not an append-only archive."
        ),
        examples=(
            "devtools workspace demo-shelf",
            "devtools workspace demo-shelf --check",
            "devtools workspace demo-shelf --root /realm/project/sinex/.agent/demos --json",
        ),
    ),
    CommandSpec(
        "workspace degraded-archive-proof",
        "workspace",
        "Build a degraded archive self-healing proof artifact.",
        "devtools.degraded_archive_proof",
        use_when=(
            "Prove WAL, FTS freshness, and planner-stat upkeep on a deterministic archive copy by "
            "deliberately degrading rebuildable state and running the same bounded repair/checkpoint/"
            "optimize primitives used by the daemon."
        ),
        examples=(
            "devtools workspace degraded-archive-proof --json",
            "devtools workspace degraded-archive-proof --out-dir .agent/demos/degraded-archive-proof/current --json",
        ),
    ),
    CommandSpec(
        "workspace cli-surface-audit",
        "workspace",
        "Capture a current-curated CLI surface audit demo.",
        "devtools.cli_surface_audit",
        use_when=(
            "Refresh the CLI surface audit shelf through one reusable command that captures representative "
            "help, status, query, read, and facets outputs while keeping large unbounded diagnostics opt-in."
        ),
        examples=(
            "devtools workspace cli-surface-audit",
            "devtools workspace cli-surface-audit --out-dir .agent/demos/cli-surface-audit/current --json",
            "devtools workspace cli-surface-audit --include-unbounded-dialogue",
        ),
    ),
    CommandSpec(
        "workspace claim-vs-evidence",
        "workspace",
        "Build a structured failure follow-up claim-vs-evidence demo.",
        "devtools.claim_vs_evidence",
        use_when=(
            "Produce a fast, bounded report over structured tool failures and the immediately following "
            "assistant turn, using tool_result is_error/exit_code as the evidence anchor instead of "
            "prose-mined outcome claims."
        ),
        examples=(
            "devtools workspace claim-vs-evidence --json",
            "devtools workspace claim-vs-evidence --limit 5000 --out-dir .agent/demos/claim-vs-evidence",
        ),
    ),
    CommandSpec(
        "workspace read-package",
        "workspace",
        "Render a declarative package of Polylogue read artifacts.",
        "devtools.read_package",
        use_when=(
            "Generate local demo/export packages from a JSON/YAML spec that names ordinary "
            "read views, formats, and output paths, instead of hand-rolling repeated "
            "`polylogue read --view ... --to file` shell snippets."
        ),
        examples=(
            "devtools workspace read-package --spec package.json --session-id 019f... --out-dir product-read",
            "devtools workspace read-package --spec package.yaml --session-id 019f... --out-dir product-read --dry-run --json",
        ),
    ),
    CommandSpec(
        "lab projections",
        "verification lab",
        "Render the authored scenario-bearing verification projections.",
        "devtools.scenario_projections",
        use_when="Inspect the unified projection inventory that feeds runtime coverage, generated docs, and control-plane maps.",
        examples=(
            "devtools lab projections",
            "devtools lab projections --source-kind validation-lane --artifact-target session_insight_rows",
            "devtools lab projections --json",
        ),
    ),
    CommandSpec(
        "verify topology",
        "verification",
        "Verify the realized polylogue tree against the topology projection.",
        "devtools.verify_topology",
        use_when=(
            "Detect orphans, conflicts, kernel-rule violations, or stale TBD cells against "
            "docs/plans/topology-target.yaml after moving files between packages."
        ),
        examples=(
            "devtools verify topology",
            "devtools verify topology --json",
            "devtools verify topology --strict-tbd",
        ),
    ),
    CommandSpec(
        "render topology-projection",
        "generated surfaces",
        "Generate docs/plans/topology-target.yaml from the current tree using placement rules.",
        "devtools.build_topology_projection",
        use_when=(
            "Refresh the topology projection after editing placement rules in this script "
            "or after a topology refactor lands."
        ),
        examples=("devtools render topology-projection",),
    ),
    CommandSpec(
        "render topology-status",
        "generated surfaces",
        "Render docs/topology-status.md from the topology projection and realized tree.",
        "devtools.render_topology_status",
        use_when=(
            "Refresh the topology drift dashboard after a refactor PR lands. "
            "Wired into devtools render all so drift fails the generated-surface check."
        ),
        examples=("devtools render topology-status", "devtools render topology-status --check"),
    ),
    CommandSpec(
        "verify closure-matrix",
        "verification",
        "Verify docs/plans/test-closure-matrix.yaml stays grounded in the realized tree.",
        "devtools.verify_closure_matrix",
        use_when=(
            "Keep the per-domain test-closure matrix honest — fails when a declared target file or "
            "representative test path is missing, or when a row violates the gate schema."
        ),
        examples=("devtools verify closure-matrix", "devtools verify closure-matrix --json"),
    ),
    CommandSpec(
        "bench slo",
        "benchmarking",
        "Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements.",
        "devtools.verify_slos",
        use_when=(
            "Run as part of devtools verify --lab, or directly to confirm read-surface "
            "(query / reader / facets / context / cost) latencies stay within their declared SLOs. "
            "Exits non-zero when any measured surface exceeds its budget."
        ),
        examples=(
            "devtools bench slo",
            "devtools bench slo --json",
            "devtools bench slo --skip-benchmarks --json",
        ),
    ),
    CommandSpec(
        "bench help-latency",
        "benchmarking",
        "Check `--help` wall-clock latency against the interactive-tier cold-CLI budget (polylogue-20d.2).",
        "devtools.help_latency_probe",
        use_when=(
            "Catch CLI import-tax regressions continuously. Runs `polylogue <cmd> --help` for a curated "
            "set of root and nested subcommands as fresh subprocesses and compares the minimum wall time "
            "against the 700ms cold-CLI budget from the 20d.14 interactive SLO tier. Fails when any "
            "'required' target exceeds budget; 'informational' targets (currently `ops maintenance`, "
            "known slow pending a lazy-import refactor) are reported but never block."
        ),
        examples=(
            "devtools bench help-latency",
            "devtools bench help-latency --json",
            "devtools bench help-latency --repeats 5 --out .local/help-latency.json",
        ),
    ),
    CommandSpec(
        "verify manifests",
        "verification",
        "Verify internal consistency across all docs/plans/*.yaml manifest files.",
        "devtools.verify_manifests",
        use_when=(
            "Catch malformed manifests, duplicate rule IDs, missing required fields, "
            "and cross-manifest reference inconsistencies."
        ),
        examples=("devtools verify manifests",),
    ),
    CommandSpec(
        "verify doc-commands",
        "verification",
        "Verify README/docs command examples resolve to live polylogue, polylogued, and devtools commands.",
        "devtools.verify_doc_commands",
        use_when=(
            "Catch doc drift away from the daemon-first command surface. "
            "Fails when README.md or any docs/**/*.md references a subcommand "
            "that is not registered, or a stale invocation like "
            "'polylogued run --enable-api' / 'polylogue run --source'."
        ),
        examples=("devtools verify doc-commands", "devtools verify doc-commands --json"),
    ),
    CommandSpec(
        "verify docs-coverage",
        "verification",
        "Verify every public CLI command, MCP tool, config key, and stable daemon route is named in the docs tree.",
        "devtools.verify_docs_coverage",
        use_when=(
            "Catch doc drift in the other direction from doc-commands: a real public surface "
            "(CLI command, MCP tool, config key, stable daemon route) shipped with zero doc-tree "
            "mention. Fails naming the exact missing entry (polylogue-3tl.9). Pre-existing gaps "
            "are tracked in docs/plans/docs-coverage-baseline.yaml as a ratchet, not an allowlist "
            "to extend."
        ),
        examples=("devtools verify docs-coverage", "devtools verify docs-coverage --json"),
    ),
    CommandSpec(
        "verify ci-workflows",
        "verification",
        "Verify CI workflow files reference locally-known devtools commands and existing paths.",
        "devtools.verify_ci_workflows",
        use_when=(
            "Catch CI workflow files that reference unregistered devtools commands or "
            "non-existent paths. Checks only locally verifiable facts — not remote CI state."
        ),
        examples=("devtools verify ci-workflows", "devtools verify ci-workflows --json"),
    ),
    CommandSpec(
        "verify test-infra-currency",
        "verification",
        "Verify tests/infra/ helpers reference only tables that exist in the current SCHEMA_VERSION.",
        "devtools.verify_test_infra_currency",
        use_when=(
            "Catch helpers that target renamed or removed tables (#1208). "
            "When SCHEMA_VERSION bumps, helper SQL drifting away from the live "
            "schema is invisible to testmon-selected runs until an unrelated change "
            "invalidates the affected tests."
        ),
        examples=("devtools verify test-infra-currency", "devtools verify test-infra-currency --json"),
    ),
    CommandSpec(
        "lab policy schema-versioning",
        "verification lab",
        "Verify durable-tier migration and derived-tier rebuild boundaries.",
        "devtools.verify_schema_upgrade_lane",
        use_when=(
            "Enforce the policy boundary documented in docs/internals.md § "
            "'Schema Versioning Model'. Durable tiers use explicit additive "
            "migrations with a backup gate; derived tiers are rebuilt or "
            "blue-green replaced from source evidence."
        ),
        examples=("devtools lab policy schema-versioning", "devtools lab policy schema-versioning --json"),
    ),
    CommandSpec(
        "lab policy backlog-hygiene",
        "verification lab",
        "Verify Beads backlog structure invariants (.beads/issues.jsonl).",
        "devtools.verify_backlog_hygiene",
        use_when=(
            "Enforce the standing backlog-hygiene invariant lint (polylogue-8jg9.1): 15 checks "
            "over the Beads export catching dangling dependency refs, blocks-cycles, missing "
            "horizon/AC/design content on tech-tree beads, P0/P1 beads without acceptance "
            "criteria, unlabeled non-epic beads, epics with no members or description, stale "
            "'adopted' decisions left open, duplicate titles, and bead ids named but never "
            "created -- catches backlog structure drift before it needs an archaeology sweep "
            "to recover, instead of only a manually-invoked script."
        ),
        examples=(
            "devtools lab policy backlog-hygiene",
            "devtools lab policy backlog-hygiene --json",
            "devtools lab policy backlog-hygiene --fresh",
        ),
    ),
    CommandSpec(
        "lab policy demo-packet-registry",
        "verification lab",
        "Verify every registered 212 demo has a conforming Demo Finding Packet.",
        "devtools.verify_demo_packet_registry",
        use_when=(
            "Enforce the 212 portfolio contract (polylogue-212.7): every demo prompt in "
            ".agent/demos/registry.json must have a packet directory carrying PROMPT.md, "
            "finding.yaml (five-part provenance stanza), report.md (fixed section order), "
            "evidence.ndjson, queries.ndjson, checks.json, and run.log. Catches a missing "
            "or malformed packet before it silently drops out of the demo shelf."
        ),
        examples=("devtools lab policy demo-packet-registry", "devtools lab policy demo-packet-registry --json"),
    ),
    CommandSpec(
        "lab policy demo-tour-freshness",
        "verification lab",
        "Verify a freshly-run demo tour matches the committed docs/examples/demo-tour/ evidence artifacts.",
        "devtools.verify_demo_tour_freshness",
        use_when=(
            "Catch drift between what `polylogue demo tour` actually emits at runtime (transcript, "
            "report, per-step command output, recording tape) and the committed copies under "
            "docs/examples/demo-tour/, modulo an explicit wall-clock-duration mask (polylogue-3tl.17). "
            "Runs the real tour (~10s), so it lives in the lab tier rather than --quick."
        ),
        examples=("devtools lab policy demo-tour-freshness",),
    ),
    CommandSpec(
        "lab policy docs-drift",
        "verification lab",
        "Verify checkable factual claims in the Reference-docs table against current source.",
        "devtools.verify_docs_drift",
        use_when=(
            "Catch doc-vs-code drift in the hand-maintained Reference-docs table "
            "(CLAUDE.md): a backtick-quoted file path that no longer exists, a "
            "'<Tier> schema version N' claim ahead of the tier's current constant, or "
            "a watchlisted table name renamed to a different current name (e.g. "
            "`artifact_observations` renamed to `raw_artifacts`) still asserted as "
            "current (polylogue-9e5.13)."
        ),
        examples=("devtools lab policy docs-drift", "devtools lab policy docs-drift --json"),
    ),
    CommandSpec(
        "lab policy timestamp-doctrine",
        "verification lab",
        "Verify durable-tier DDL never stores a timestamp column as TEXT.",
        "devtools.verify_timestamp_doctrine",
        use_when=(
            "Enforce the time doctrine (UTC epoch-ms canon, docs/internals.md) at DDL-review "
            "time (cpf.1): a TEXT timestamp in source.db/user.db re-introduces tz-unknown "
            "ambiguity and lexicographic-vs-temporal sort divergence, and durable tiers need "
            "an explicit additive migration to fix later -- catching it before merge is orders "
            "cheaper than a copy-forward migration after."
        ),
        examples=("devtools lab policy timestamp-doctrine", "devtools lab policy timestamp-doctrine --json"),
    ),
    CommandSpec(
        "lab policy insight-honesty",
        "verification lab",
        "Verify every registered insight product is rigor-contracted or exempt.",
        "devtools.verify_insight_rigor_honesty",
        use_when=(
            "Enforce that polylogue.insights.registry.INSIGHT_REGISTRY and "
            "polylogue.insights.rigor's contract matrix/exemption list never drift apart "
            "(9e5.28) -- a registered product with neither a RigorContract nor a "
            "RIGOR_EXEMPT entry used to silently vanish from `polylogue ops insights audit` "
            "instead of showing as uncovered."
        ),
        examples=("devtools lab policy insight-honesty", "devtools lab policy insight-honesty --json"),
    ),
    CommandSpec(
        "verify test-clock-hygiene",
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
        examples=("devtools verify test-clock-hygiene", "devtools verify test-clock-hygiene --json"),
    ),
    CommandSpec(
        "verify pytest-timeout-overrides",
        "verification",
        "Verify explicit pytest timeout overrides are positive, bounded, and justified.",
        "devtools.verify_pytest_timeout_overrides",
        use_when=(
            "Check AST-parsed @pytest.mark.timeout decorators and managed pytest command literals. "
            "Values above the pyproject default require an exact manifest rationale."
        ),
        examples=("devtools verify pytest-timeout-overrides", "devtools verify pytest-timeout-overrides --json"),
    ),
    CommandSpec(
        "verify degrade-loudly",
        "verification",
        "Verify broad except-handlers in daemon/storage/insights/coordination log or signal on failure.",
        "devtools.verify_degrade_loudly",
        use_when=(
            "Enforce the degrade-loudly doctrine (polylogue-cpf.4): a broad except-handler "
            "(Exception/BaseException/*.Error) in derived-read, status, or probe code that "
            "swallows the exception with no log call and no re-raise is indistinguishable from "
            "'no data' to a reader. New silent sites must add a log call, or add a typed signal "
            "plus a rationale entry in docs/plans/degrade-loudly-allowlist.yaml."
        ),
        examples=("devtools verify degrade-loudly", "devtools verify degrade-loudly --json"),
    ),
    CommandSpec(
        "release verify-distribution",
        "release",
        "Verify wheel/sdist installed artifacts expose only supported runtime entrypoints.",
        "devtools.verify_distribution_surface",
        use_when=(
            "Build wheel and sdist artifacts, rebuild a wheel from an unpacked sdist without .git, "
            "and smoke installed runtime console scripts."
        ),
        examples=("devtools release verify-distribution",),
    ),
    CommandSpec(
        "lab probe bead-pr-reconciliation",
        "verification lab",
        "Surface beads whose referenced PR merged but the bead is still open.",
        "devtools.verify_bead_pr_reconciliation",
        use_when=(
            "After a merge-heavy stretch (a Workflow campaign, a merge train, or just several PRs "
            "landed close together), check for beads left open by a PR that referenced them -- catches "
            "the reconciliation gap where workers/agents are barred from closing beads themselves and no "
            "follow-up pass ever ran (2026-07-14: a 55-bead campaign left every bead open despite ~20 "
            "PRs merging clean). Advisory only -- reports candidates for a human/agent AC check, never "
            "auto-closes and never fails a gate."
        ),
        examples=(
            "devtools lab probe bead-pr-reconciliation",
            "devtools lab probe bead-pr-reconciliation --since 2026-07-01 --json",
            "devtools lab probe bead-pr-reconciliation --limit 50",
        ),
    ),
    CommandSpec(
        "lab probe cost-reconciliation",
        "verification lab",
        "Reconcile Polylogue token accounting against private provider stores.",
        "devtools.cost_reconciliation_probe",
        use_when=(
            "Validate archive token accounting against optional local Codex state_5.sqlite and "
            "Claude stats-cache.json before publishing cost or usage-analysis claims."
        ),
        examples=(
            "devtools lab probe cost-reconciliation --json",
            "devtools lab probe cost-reconciliation --codex-state ~/.codex/state_5.sqlite --json",
            "devtools lab probe cost-reconciliation --claude-stats-cache ~/.config/claude/stats-cache.json --check",
        ),
    ),
    CommandSpec(
        "lab probe pipeline",
        "verification lab",
        "Run typed pipeline probes against synthetic, staged, or archive-subset inputs.",
        "devtools.pipeline_probe",
        use_when="Run real pipeline stages and optionally capture emitted summaries as regression cases.",
        examples=(
            "devtools lab probe pipeline --provider chatgpt --stage parse",
            "devtools lab probe pipeline --input-mode archive-subset --capture-regression live-parse-drift",
        ),
    ),
    CommandSpec(
        "lab probe turso",
        "verification lab",
        "Probe Turso Database compatibility against Polylogue storage assumptions.",
        "devtools.turso_probe",
        use_when=(
            "Collect executable evidence before changing production storage backends: "
            "Python binding availability, generated-column support, FTS compatibility, MVCC, CDC, "
            "vector functions, ATTACH, and WAL pragma behavior."
        ),
        examples=(
            "devtools lab probe turso --json",
            "devtools lab probe turso --check",
            "devtools lab probe turso --tursodb /nix/store/.../bin/tursodb --json",
        ),
    ),
    CommandSpec(
        "bench memory",
        "benchmarking",
        "Measure query-memory envelopes on generated fixtures.",
        "devtools.query_memory_budget",
        use_when="Assert memory budgets around a concrete query or archive-facing command.",
        examples=("devtools bench memory --max-rss-mb 1536 -- polylogue --plain analyze",),
    ),
    CommandSpec(
        "lab smoke",
        "verification lab",
        "Run direct archive and reader smoke sets.",
        "devtools.lab_scenario",
        use_when="Run direct archive and reader smoke sets outside the archive CLI.",
        examples=(
            "devtools lab smoke list",
            "devtools lab smoke run archive-smoke --tier 0",
        ),
    ),
    CommandSpec(
        "lab schema list",
        "verification lab",
        "List committed schema packages, versions, and evidence manifests.",
        "devtools.schema_inspect",
        entrypoint="list_main",
        use_when="Inspect committed provider schema package catalogs without presenting them as normal archive usage.",
        examples=("devtools lab schema list --provider chatgpt --json",),
    ),
    CommandSpec(
        "lab schema compare",
        "verification lab",
        "Compare two committed schema package versions for a provider.",
        "devtools.schema_inspect",
        entrypoint="compare_main",
        use_when="Review schema package drift between committed versions in the lab surface.",
        examples=("devtools lab schema compare --provider chatgpt --from v1 --to v2 --markdown",),
    ),
    CommandSpec(
        "lab schema explain",
        "verification lab",
        "Explain a committed package element schema with evidence and annotations.",
        "devtools.schema_inspect",
        entrypoint="explain_main",
        use_when="Inspect schema package annotations, semantic roles, and review evidence from the lab surface.",
        examples=("devtools lab schema explain --provider chatgpt --version latest --verbose",),
    ),
    CommandSpec(
        "lab schema generate",
        "verification lab",
        "Generate provider schema packages and optional evidence clusters.",
        "devtools.schema_generate",
        use_when="Refresh provider schema package artifacts from archive observations outside the archive CLI.",
        examples=("devtools lab schema generate --provider chatgpt --cluster",),
    ),
    CommandSpec(
        "lab schema promote",
        "verification lab",
        "Promote a schema evidence cluster into a registered package version.",
        "devtools.schema_promote",
        use_when="Turn reviewed schema evidence clusters into committed provider schema packages.",
        examples=("devtools lab schema promote --provider chatgpt --cluster chatgpt-message-v2",),
    ),
    CommandSpec(
        "lab schema audit",
        "verification lab",
        "Run committed provider schema package quality checks.",
        "devtools.schema_audit",
        use_when="Check committed schema package quality gates without presenting them as normal archive usage.",
        examples=("devtools lab schema audit --provider chatgpt --json",),
    ),
    CommandSpec(
        "lab schema roundtrip",
        "verification lab",
        "Verify committed provider schema packages reload and roundtrip cleanly.",
        "devtools.verify_schema_roundtrip",
        use_when=(
            "Close the schema inference-validation loop: package manifests must roundtrip through typed models, "
            "and every supported element schema must be reachable from the runtime registry."
        ),
        examples=(
            "devtools lab schema roundtrip --provider chatgpt",
            "devtools lab schema roundtrip --all --json",
        ),
    ),
    CommandSpec(
        "lab probe capture-regression",
        "verification lab",
        "Capture pipeline-probe summaries as durable local regression cases.",
        "devtools.regression_capture",
        use_when="Turn a live or probe failure JSON summary into a replayable local regression artifact.",
        examples=(
            "devtools lab probe capture-regression --input probe.json --name parse-drift",
            "devtools lab probe pipeline --json | devtools lab probe capture-regression --name parse-drift --tag live",
        ),
    ),
    CommandSpec(
        "verify layering",
        "verification",
        "Check inter-package imports against declared layering rules from docs/plans/layering.yaml.",
        "devtools.verify_layering",
        use_when=(
            "Diagnose architecture drift: which files import across declared "
            "package boundaries. This runs in verify --quick."
        ),
        examples=("devtools verify layering", "devtools verify layering --json"),
    ),
    CommandSpec(
        "verify evidence",
        "verification",
        "Render the pytest-first evidence dashboard or a changed-path trace.",
        "devtools.evidence_dashboard",
        use_when=(
            "Inspect pytest health, contract-evidence inventory, coverage, SLO "
            "catalog, static-gate status, and campaign freshness, or "
            "trace which evidence artifacts cover the changed paths in a PR."
        ),
        examples=(
            "devtools verify evidence --json",
            "devtools verify evidence --markdown",
            "devtools verify evidence trace --base origin/master --head HEAD --markdown",
        ),
    ),
    CommandSpec(
        "release build-package",
        "release",
        "Build the default Nix package with the out-link under .local/result.",
        "devtools.build_package",
        use_when="Produce the Nix package artifact with its out-link kept under the repo-local output root.",
        examples=("devtools release build-package",),
    ),
    CommandSpec(
        "bench mutation",
        "benchmarking",
        "Run focused mutation campaigns and maintain their local index.",
        "devtools.mutmut_campaign",
        use_when="Run or inspect focused mutation-testing work without shrinking the committed mutmut scope.",
        examples=("devtools bench mutation list", "devtools bench mutation run filters"),
        featured=True,
    ),
    CommandSpec(
        "bench campaign",
        "benchmarking",
        "Run or compare benchmark campaigns.",
        "devtools.benchmark_campaign",
        use_when="Record durable benchmark artifacts or compare a candidate run against a baseline artifact.",
        examples=(
            "devtools bench campaign list",
            "devtools bench campaign run search-filters",
            "devtools bench campaign compare baseline.json candidate.json",
        ),
        featured=True,
    ),
    CommandSpec(
        "bench synthetic",
        "benchmarking",
        "Run synthetic benchmark campaigns over generated archives.",
        "devtools.run_campaign",
        use_when="Generate synthetic archives and run long-haul benchmark workloads.",
        examples=(
            "devtools bench synthetic --list",
            "devtools bench synthetic --scale medium --campaign search-filters",
        ),
    ),
    CommandSpec(
        "workspace tasks",
        "workspace",
        "Record and query local agent task execution history.",
        "devtools.task_history",
        use_when="Log, view recent, or summarize agent task execution history during development sessions.",
        examples=(
            "devtools workspace tasks log --command 'devtools render all' --duration-ms 3200 --exit-code 0",
            "devtools workspace tasks recent",
            "devtools workspace tasks recent --count 20",
            "devtools workspace tasks stats",
            "devtools workspace tasks stats --json",
            "devtools workspace tasks stats --resources",
        ),
    ),
    CommandSpec(
        "workspace failure-context",
        "workspace",
        "Join testmon, git history, and fixtures for a pytest failure ID into a JSON envelope.",
        "devtools.failure_context",
        use_when=(
            "Bootstrap an agent inner-loop debugging session for a failing test — surfaces production "
            "files the test depends on, their recent commits, and fixtures the test uses, "
            "all in one structured envelope."
        ),
        examples=(
            "devtools workspace failure-context tests/unit/storage/test_foo.py::test_bar",
            "devtools workspace failure-context tests/unit/storage/test_foo.py::test_bar --days 14",
        ),
    ),
)

COMMANDS: dict[str, CommandSpec] = {spec.name: spec for spec in COMMAND_SPECS}


def command_name_from_tokens(tokens: Iterable[str], commands: Iterable[CommandSpec] = COMMAND_SPECS) -> str | None:
    """Resolve leading argv tokens to a registered command name."""
    token_tuple = tuple(tokens)
    if not token_tuple:
        return None
    by_path = {spec.command_path: spec.name for spec in commands}
    max_len = max((len(path) for path in by_path), default=0)
    for length in range(min(max_len, len(token_tuple)), 0, -1):
        candidate = token_tuple[:length]
        if candidate in by_path:
            return by_path[candidate]
    return None


def _flatten_argv_parts(args: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(part for arg in args for part in arg.split(" ") if part)


def control_plane_command(*args: str) -> str:
    parts = [CONTROL_PLANE, *_flatten_argv_parts(args)]
    return " ".join(part for part in parts if part)


def control_plane_argv(*args: str) -> tuple[str, ...]:
    return tuple(part for part in (CONTROL_PLANE, *_flatten_argv_parts(args)) if part)


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
    "command_name_from_tokens",
    "control_plane_argv",
    "control_plane_command",
    "featured_command_specs",
    "grouped_command_specs",
    "VERIFICATION_LAB_COMMAND_NAMES",
    "verification_lab_command_specs",
]
