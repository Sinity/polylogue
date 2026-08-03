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
    "lab policy bead-graph",
    "lab policy campaign-archive-boundaries",
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
    "lab schema commit",
    "lab schema compare",
    "lab schema explain",
    "lab schema generate",
    "lab schema list",
    "lab schema parser-diff",
    "lab schema promote",
    "lab schema roundtrip",
    "lab seed-receipt-compare",
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
        "render agent-manual",
        "generated surfaces",
        "Render the declaration-generated six-tool agent manual and packaged integration assets.",
        "devtools.render_agent_manual",
        use_when="Refresh or check cold-start guidance after MCP, query, origin, recipe, or delivery changes.",
        examples=("devtools render agent-manual", "devtools render agent-manual --check"),
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
        "render public-claims",
        "generated surfaces",
        "Render README, launch, findings-page, and verified-export claim views from FINDING assertions.",
        "devtools.render_public_claims",
        use_when=(
            "Refresh public claim Markdown/JSON and the generated YAML compatibility view after changing "
            "FINDING seeds, judgments, capability declarations, or 37t.14 verdict receipts."
        ),
        examples=(
            "devtools render public-claims",
            "devtools render public-claims --check",
            "devtools render public-claims --archive-root /path/to/archive --verdicts /path/to/verdicts.json",
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
        "Verify generated public-claim views, preset parity, sanitized refs, coverage markers, and retired copy.",
        "devtools.public_claims",
        use_when=(
            "Check externally visible claims after changing README, demos, findings, FINDING assertions, "
            "capability declarations, or evidence-integrity receipts."
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
        "lab seed-receipt-compare",
        "verification lab",
        "Compare two workload receipts for a clean, like-for-like seed/incident proof (polylogue-b054.1.1.3).",
        "devtools.seed_receipt_compare",
        use_when=(
            "Judge a seed-testmon or focused pytest run's resource receipt against a named baseline "
            "receipt (e.g. a prior incident) — confirms both runs share workload identity and terminated "
            "cleanly, then scores wall-time speedup and peak-PSS ceiling targets, naming a blocker and "
            "linked follow-up for any unmet target instead of silently dropping it."
        ),
        examples=(
            "devtools lab seed-receipt-compare --baseline incident.json --candidate current.json",
            "devtools lab seed-receipt-compare --baseline incident.json --candidate current.json --json",
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
        "workspace backlog-calibration",
        "workspace",
        "Measured lead-time/discovery/staleness distributions over the bead corpus.",
        "devtools.backlog_calibration",
        use_when=(
            "Re-fit the numbers a backlog-execution plan rests on instead of guessing them: "
            "closed-bead lead-time percentiles split by priority/type/epic-membership/dependency "
            "degree, a censoring-honest survival view (closed-only medians are survivorship-"
            "biased), close-reason classification measuring how much of the backlog closes with "
            "no implementation (already-satisfied/obsolete/duplicate), and created-vs-closed "
            "discovery dynamics (does the backlog drain?). Optionally calibrates PR open->merge "
            "latency by size from a gh dump. Answers a different question than "
            "`workspace beads-state-report` (population shape and graph hygiene, point-in-time) "
            "and `workspace bead-cluster` (what to batch next): this is duration *models* over "
            "history, meant to be re-run as the corpus grows so plans stop quoting stale guesses."
        ),
        examples=(
            "devtools workspace backlog-calibration",
            "devtools workspace backlog-calibration --input beads.jsonl --json",
            "devtools workspace backlog-calibration --prs prs.json",
        ),
    ),
    CommandSpec(
        "workspace bead-batch-show",
        "workspace",
        "Batch-show beads: id, status, prio, title, desc head, deps, notes tail.",
        "devtools.bead_batch_show",
        use_when=(
            "Skim several beads at once (e.g. a fanout cluster or a discovered-follow-up batch) "
            "without a separate `bd show` round-trip per id."
        ),
        examples=("devtools workspace bead-batch-show polylogue-kapb polylogue-2yax",),
    ),
    CommandSpec(
        "workspace bead-cluster",
        "workspace",
        "Footprint/overlap/contention clustering of ready Beads (execution frontier).",
        "devtools.bead_cluster",
        use_when=(
            "Before dispatching a batch of ready beads, see which ones share a file/package "
            "footprint (cluster into one branch/PR sweep) vs. which are genuinely disjoint "
            "(safe to run as parallel worktree lanes), and flag contention -- beads that touch "
            "the same durable migration slot or the same generated surface even without an "
            "exact file-path overlap. Answers a different question than "
            "`workspace delivery-gate-status` (gate-progress board): this is footprint/overlap "
            "clustering over the same live bead data, not gate percentage. Footprints are "
            "advisory -- verify on claim."
        ),
        examples=(
            "devtools workspace bead-cluster",
            "devtools workspace bead-cluster --json",
            "devtools workspace bead-cluster --all-open",
            "devtools workspace bead-cluster --max-priority 1",
            "devtools workspace bead-cluster --input ready.json --validate-roster",
        ),
    ),
    CommandSpec(
        "workspace lane-brief",
        "workspace",
        "Generate a dispatch brief for a bead lane with live footprint/prior-art evidence.",
        "devtools.lane_brief",
        use_when=(
            "Before dispatching a batch of bead ids as one lane/branch to a subagent, produce a "
            "markdown brief carrying the full bead records plus LIVE evidence: every file path "
            "mentioned in the bead text is checked against the working tree (exists? line count? "
            "last 3 commits?) so a dispatcher does not hand a subagent stale bead prose, and closed "
            "beads mentioning the same paths are surfaced as prior art. Sections the tool cannot "
            "fill (measured baseline, non-goals) are emitted as explicit placeholders, never "
            "silently dropped."
        ),
        examples=(
            "devtools workspace lane-brief polylogue-abc polylogue-def",
            "devtools workspace lane-brief polylogue-ei94 --out .agent/scratch/lane-ei94.md",
        ),
    ),
    CommandSpec(
        "workspace lane-init",
        "workspace",
        "Provision a fanout lane worktree: branch, isolated venv, guard check, ledger record.",
        "devtools.lane_init",
        use_when=(
            "Before dispatching a lane in a high-concurrency fanout: creates the worktree/branch "
            "if missing, provisions its OWN venv via uv sync (a shared-venv worktree cannot run "
            "devtools/pytest -- checkout guard), proves import polylogue resolves inside the lane, "
            "runs verify-worktree, appends the lane to .cache/fanout/lanes.jsonl (resumable fanout "
            "state, polylogue-in94), and prints the per-lane POLYLOGUE_PYTEST_WORKERS budget for "
            "the planned concurrency."
        ),
        examples=(
            "devtools workspace lane-init /realm/worktrees/lane-cursors --branch feature/sources/cursor-catchup --beads polylogue-2qrx,polylogue-ix5r",
            "devtools workspace lane-init /realm/worktrees/lane-x --branch feature/x --expected-lanes 16 --json",
        ),
    ),
    CommandSpec(
        "workspace verify-worktree",
        "workspace",
        "Verify an agent lane's claimed worktree exists, is isolated, and is on the expected branch.",
        "devtools.verify_worktree",
        use_when=(
            "Immediately after spawning a worktree-isolated agent lane (and before trusting its "
            "reported diff), confirm the claimed working directory is a real LINKED git worktree "
            "-- not the main checkout (the 2026-08-01 isolation-escape incident left ~1700 "
            "uncommitted lines in the coordinator's live tree because no such check existed) -- "
            "and optionally that the expected branch is checked out. Also reports advisory "
            "hazards: uncommitted changes (worktree auto-cleanup destroys them) and a stale "
            ".beads/issues.jsonl whose reimport can time-machine live bead state (polylogue-2ara)."
        ),
        examples=(
            "devtools workspace verify-worktree /realm/project/polylogue/.claude/worktrees/agent-abc",
            "devtools workspace verify-worktree /realm/worktrees/lane-x --expect-branch feature/x/y",
            "devtools workspace verify-worktree /realm/worktrees/lane-x --json --strict",
        ),
    ),
    CommandSpec(
        "workspace merge-gate",
        "workspace",
        "Structural pre-merge safety check: fresh local-verification receipt + no late review comments.",
        "devtools.merge_gate",
        use_when=(
            "Immediately before squash-merging any PR in a merge train, replacing coordinator memory "
            "(grace-period comment polling, remembering to run the broader local test suite CI skips "
            'per-PR) with a check that fails closed. `record <PR> --command "..."` requires the current '
            "checkout to already be the PR's exact head commit with a clean tree (it refuses otherwise), "
            "runs a local verification "
            "command, and persists a receipt flagging commands that look like they skip tests (e.g. "
            "`verify --quick`). `check <PR>` polls review comments across a real grace window (default "
            "3x20s, covering CodeRabbit's 30-60s late-arrival window) and BLOCKs unless a receipt exists "
            "for the CURRENT head sha within a freshness window with exit_code 0, and no review comment's "
            "created_at is newer than the head commit's timestamp unless explicitly `ack`'d for that exact "
            "head sha. Motivated by two 2026-08-01 incidents: PR #3502 merged before CodeRabbit's findings "
            "posted, and PR #3517 nearly merged with a 43-test regression no CI check or review comment "
            "ever flagged -- plus review findings on this tool itself (recording from an unrelated "
            "checkout, a --quick example that would have missed its own motivating regression, a single "
            "comment snapshot instead of a grace-period poll, and no way to triage a false-positive late "
            "comment without an empty commit). `check --post-status` (polylogue-1cbeh) posts the "
            "verdict as a GitHub commit status (`context=merge-gate`, success/failure) on the PR's "
            "current head sha via `gh api repos/{owner}/{repo}/statuses/{sha}` -- this is what lets "
            "branch protection or `gh pr merge --auto` gate on the same verdict instead of a "
            "coordinator manually cycling checkout->test->check->merge one PR at a time."
        ),
        examples=(
            'devtools workspace merge-gate record 3517 --command "devtools verify"',
            "devtools workspace merge-gate check 3517",
            "devtools workspace merge-gate check 3517 --json --max-age-s 7200 --poll-rounds 1",
            "devtools workspace merge-gate check 3517 --post-status",
            'devtools workspace merge-gate ack 3517 123456789 --reason "already fixed upstream, false positive"',
        ),
    ),
    CommandSpec(
        "workspace merge",
        "workspace",
        "Merge boundary wrapper: refuses `gh pr merge` without a fresh merge-gate receipt.",
        "devtools.merge_boundary",
        use_when=(
            "Replace a bare `gh pr merge --squash` with this at the actual merge boundary "
            "(polylogue-ct3r2 / polylogue-t6iga: duplicate filings of the same finding -- "
            "`merge-gate record/check` and the one-full-verify-per-train rule both existed but "
            "fired only if a coordinator remembered to invoke them). `merge <PR>` auto-records a "
            "merge-gate receipt if none is fresh for the current head sha (running `--command`, "
            'default "devtools verify"), runs `merge-gate check` and refuses to merge on any '
            "BLOCK, strips a doubled `(#N) (#N)` squash-subject suffix (the 2026-07-12/13 "
            "incident), then runs the actual `gh pr merge --squash`. `--dry-run` runs every check "
            "without merging. `--with-verify` immediately runs and records the merge-train's "
            "terminal full-suite verify after merging; otherwise it prints a reminder. "
            "`train-status` reports (exit 1) any PRs merged since the last recorded full-suite "
            "verify -- the structural stand-in for 'a merge-train records the full-suite verify "
            "as its terminal ledger step'. `record-full-verify` runs and records that step "
            "directly, e.g. once per merge-train session boundary."
        ),
        examples=(
            "devtools workspace merge 3517",
            'devtools workspace merge 3517 --command "devtools test tests/unit/foo.py"',
            "devtools workspace merge 3517 --dry-run",
            'devtools workspace merge 3517 --with-verify --verify-command "devtools verify --all"',
            "devtools workspace merge train-status",
            'devtools workspace merge record-full-verify --command "devtools verify --all"',
        ),
    ),
    CommandSpec(
        "workspace merge-conductor",
        "workspace",
        "Mechanical-conflict triage for the PR merge train (dry-run by default).",
        "devtools.merge_conductor",
        use_when=(
            "Before or during a merge train, classify each roster PR's conflicting/modified files "
            "into AUTO-RESOLVABLE mechanical classes (.beads/issues.jsonl take-master, regenerable "
            "surfaces) vs ESCALATE classes (schema migrations, hooks config, anything else) and "
            "detect cross-PR contention (two PRs claiming the same migration slot or generated-"
            "surface family). Implements the mechanical slice of the polylogue-ei94 merge-conductor "
            "design. --execute only touches AUTO-RESOLVABLE PRs, in a scratch worktree, and aborts "
            "back to ESCALATE on any deviation; escalate-class PRs are never touched under --execute."
        ),
        examples=(
            "devtools workspace merge-conductor --pr 3301 --pr 3302",
            "devtools workspace merge-conductor --pr 3301 --json",
            "devtools workspace merge-conductor --pr 3301 --execute",
        ),
    ),
    CommandSpec(
        "workspace bead-reimport-guard",
        "workspace",
        "Monotonic, receipted guard/reconcile/export for bd's JSONL synchronization.",
        "devtools.bd_reimport_guard",
        use_when=(
            "`snapshot`/`check-and-repair` are invoked by .beads-hooks/post-checkout and "
            "post-merge (outside the BEGIN/END BEADS INTEGRATION markers) around bd's own "
            "reimport step -- not normally run by hand. bd's reimport is a blind upsert with no "
            "revision comparison, so an ordinary checkout/pull can silently revert a bead "
            "closed-but-uncommitted in a different worktree sharing the same live Dolt DB. "
            "`reconcile <candidate.jsonl>` is the explicit, hand-invokable path for flows the "
            "git hooks never see -- most importantly right after `git reset --hard`, which "
            "rewrites .beads/issues.jsonl without firing any hook, leaving it live-hazardous "
            "until the next `bd` call reimports it. `export <target.jsonl>` does an atomic, "
            "validated snapshot export. All three write a machine-readable SyncReceipt (per-row "
            "new/updated/equal/skipped_downgrade/conflicted outcome) under "
            ".cache/bd-sync-receipts/; ordinary sync never applies a downgrade -- recovering one "
            "on purpose requires reconcile --allow-downgrade plus --actor/--reason."
        ),
        examples=(
            "devtools workspace bead-reimport-guard snapshot post-checkout",
            "devtools workspace bead-reimport-guard check-and-repair post-checkout",
            "devtools workspace bead-reimport-guard reconcile .beads/issues.jsonl",
            "devtools workspace bead-reimport-guard export /tmp/issues-snapshot.jsonl",
        ),
    ),
    CommandSpec(
        "workspace delivery-gate-status",
        "workspace",
        "Per-release-gate progress board over .beads/issues.jsonl (delivery:<release> / lane:<lane> overlay).",
        "devtools.delivery_gate_status",
        use_when=(
            "Check overall delivery-gate progress (R0-normalize through N-horizon) -- percent "
            "complete, ready/blocked/in-progress counts, and the active frontier gate's exit "
            "criterion. --fresh re-exports .beads/issues.jsonl first since bd updates do not "
            "immediately re-export."
        ),
        examples=(
            "devtools workspace delivery-gate-status",
            "devtools workspace delivery-gate-status --fresh",
            "devtools workspace delivery-gate-status --json",
            "devtools workspace delivery-gate-status --gate delivery:A-trust-floor",
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
        "workspace raw-authority-restart-proof",
        "workspace",
        "Prove raw-authority crash recovery and conserved fixed-point convergence.",
        "devtools.raw_authority_restart_proof",
        use_when=(
            "Exercise fresh synthetic production-format archives across before-commit, "
            "after-commit/pre-finalization, and resumed-batch crash boundaries; then audit "
            "durable conservation and two matching quiescent dry-run censuses."
        ),
        examples=(
            "devtools workspace raw-authority-restart-proof --json",
            "devtools workspace raw-authority-restart-proof --workdir .cache/raw-restart-proof --keep --json",
        ),
    ),
    CommandSpec(
        "workspace raw-authority-daemon-health-proof",
        "workspace",
        "Prove daemon status/health HTTP responsiveness during a real raw-authority drain.",
        "devtools.raw_authority_daemon_health_proof",
        use_when=(
            "Prove polylogue-hjpx.2 AC2's daemon-health responsiveness clause: unlike "
            "raw-authority-scale-proof and raw-authority-restart-proof (which call "
            "repair_raw_materialization in-process with no HTTP surface), this starts a real "
            "polylogued subprocess, lets its own periodic convergence loop drain a synthetic "
            "backlog, and concurrently probes /healthz/live, /healthz/ready, and /api/status "
            "on a fixed interval to record p50/p95/p99/max latency and any unresponsive span."
        ),
        examples=(
            "devtools workspace raw-authority-daemon-health-proof --json",
            "devtools workspace raw-authority-daemon-health-proof --components 200 --raws 200 --keep --json",
        ),
    ),
    CommandSpec(
        "workspace raw-live-source-reconciliation",
        "workspace",
        "Classify quarantined raw evidence against its live source file's current bytes.",
        "devtools.raw_live_source_reconciliation_report",
        use_when=(
            "polylogue-u19l: before deciding whether to act on the ~59GB of quarantined "
            "raw_sessions rows, get a read-only classification of how much of it is still "
            "directly re-verifiable against still-present live source files -- no "
            "revision-graph reasoning required. Never mutates authority state or blobs; "
            "acting on 'reconcilable' rows is a separate, explicitly operator-authorized step."
        ),
        examples=(
            "devtools workspace raw-live-source-reconciliation",
            "devtools workspace raw-live-source-reconciliation --json",
            "devtools workspace raw-live-source-reconciliation --limit 500 --sample-limit 20",
        ),
    ),
    CommandSpec(
        "workspace raw-live-source-reconciliation-apply",
        "workspace",
        "Promote quarantined raw evidence proven correct by live-source verification.",
        "devtools.raw_live_source_reconciliation_apply",
        use_when=(
            "polylogue-u19l: act on the classifier's report -- promote ONLY exact_match / "
            "codex_header_strip_match quarantined raw_sessions rows to revision_authority="
            "'byte_proven' (never diverged/source_missing). Default is dry-run; --apply "
            "requires --backup-manifest pointing at a verified source-tier backup "
            "(polylogue backup --output-dir <dir> --verify). Writes an immutable per-row "
            "receipt; never runs blob GC or VACUUM -- that is a separate, later step."
        ),
        examples=(
            "devtools workspace raw-live-source-reconciliation-apply",
            "devtools workspace raw-live-source-reconciliation-apply --json",
            "devtools workspace raw-live-source-reconciliation-apply --apply "
            "--backup-manifest /realm/staging/polylogue-backup/manifest.json",
        ),
    ),
    CommandSpec(
        "workspace attachment-reacquisition",
        "workspace",
        "Classify historically-unfetched attachments for a source-backed backfill.",
        "devtools.attachment_reacquisition_report",
        use_when=(
            "polylogue-pfdf: before backfilling the 79%-unfetched attachments backlog, get a "
            "read-only classification of how many are recoverable by re-parsing their still-"
            "durable raw_sessions payload with today's parser (e.g. attachments whose "
            "extracted_content the parser didn't extract until 60d93b618), how many are "
            "structurally unrecoverable (ChatGPT sandbox_file Code Interpreter output), and how "
            "many remain undetermined (most commonly Drive/OAuth attachments needing a live "
            "authenticated fetch this classifier never performs). Never mutates state."
        ),
        examples=(
            "devtools workspace attachment-reacquisition",
            "devtools workspace attachment-reacquisition --json",
            "devtools workspace attachment-reacquisition --raw-row-limit 20000 --sample-limit 20",
        ),
    ),
    CommandSpec(
        "workspace attachment-reacquisition-apply",
        "workspace",
        "Backfill acquisition for historically-unfetched attachments.",
        "devtools.attachment_reacquisition_apply",
        use_when=(
            "polylogue-pfdf: act on the classifier's report -- publish freshly re-derived bytes "
            "for every 'reacquirable' attachment and mark every 'unrecoverable' attachment "
            "acquisition_status='unavailable' (distinct from 'unfetched'). Default is dry-run; "
            "--apply requires --manifest-path (immutable JSONL receipt) and --backup-manifest "
            "pointing at a verified index-tier backup (polylogue backup --output-dir <dir> "
            "--verify). 'undetermined' attachments are never touched."
        ),
        examples=(
            "devtools workspace attachment-reacquisition-apply",
            "devtools workspace attachment-reacquisition-apply --json",
            "devtools workspace attachment-reacquisition-apply --apply "
            "--manifest-path /realm/staging/attachment-reacquisition-manifest.jsonl "
            "--backup-manifest /realm/staging/polylogue-backup/manifest.json",
        ),
    ),
    CommandSpec(
        "workspace raw-membership-writeback-apply",
        "workspace",
        "Propagate already-decided membership verdicts onto raw_sessions.revision_authority.",
        "devtools.raw_membership_writeback_apply",
        use_when=(
            "polylogue-lb39z (Phase 1, item 2): 15,737 quarantined raw_sessions rows "
            "(42.95GB, measured 2026-08-02) already have a decided raw_session_memberships "
            "verdict (applied/superseded_equivalent/superseded_prefix, membership "
            "revision_authority='byte_proven') that was never written back -- pure fact "
            "transfer, not new judgment. Default is dry-run; --apply requires "
            "--backup-manifest pointing at a verified source-tier backup (polylogue backup "
            "--output-dir <dir> --verify). Writes an immutable per-row receipt; never runs "
            "blob GC or VACUUM -- that is a separate, later step."
        ),
        examples=(
            "devtools workspace raw-membership-writeback-apply",
            "devtools workspace raw-membership-writeback-apply --json",
            "devtools workspace raw-membership-writeback-apply --apply "
            "--backup-manifest /realm/staging/polylogue-backup/manifest.json",
        ),
    ),
    CommandSpec(
        "workspace raw-byte-duplicate-supersession-apply",
        "workspace",
        "Promote quarantined, logical-key-less raws proven byte-identical to an already-indexed raw.",
        "devtools.raw_byte_duplicate_supersession_apply",
        use_when=(
            "polylogue-6753s (corrected finding, 2026-08-03): of 7,200 unindexed logical-source "
            "heads, 4,305 (17.6 of 22.9 GiB) are revision_authority='quarantined' with "
            "logical_source_key IS NULL -- invisible to every reconciliation path, since all of "
            "them key off logical_source_key -- and are byte-identical (same blob_hash) to some "
            "OTHER raw that already has a materialized session in index.db: re-acquisitions/"
            "re-syncs of already-archived content, not missing content. Default is dry-run; "
            "--apply requires --backup-manifest pointing at a verified source-tier backup "
            "(polylogue backup --output-dir <dir> --verify). Writes an immutable per-row receipt "
            "(raw_byte_duplicate_supersession_receipts) recording which indexed raw_id/session_id "
            "each duplicate matches; never touches index.db or the indexed twin's own row; never "
            "runs blob GC or VACUUM -- those are separate, later steps. The genuinely novel "
            "remainder (no indexed byte-identical twin) is left untouched for polylogue-lkrc/"
            "polylogue-hjpx's reconciler."
        ),
        examples=(
            "devtools workspace raw-byte-duplicate-supersession-apply",
            "devtools workspace raw-byte-duplicate-supersession-apply --json",
            "devtools workspace raw-byte-duplicate-supersession-apply --apply "
            "--backup-manifest /realm/staging/polylogue-backup/manifest.json",
        ),
    ),
    CommandSpec(
        "workspace binary-artifact-sweep",
        "workspace",
        "Find raw_sessions rows whose bytes are a non-session binary format (SQLite, etc).",
        "devtools.binary_artifact_sweep_report",
        use_when=(
            "polylogue-hbtj2: the 2026-08-02 authority-dataflow audit found ~450MB of "
            "SQLite databases (Hermes state.db/verification_evidence.db, Codex "
            "state_5.sqlite) miscaptured as session content and marked 'genuinely "
            "diverged' by the byte-diff classifier. This read-only report classifies "
            "every raw_sessions row by magic bytes (core.binary_signatures) and reports "
            "which are unrecognized binary artifacts vs the one existing Hermes "
            "recognized-session-parser shape. Never mutates source.db."
        ),
        examples=(
            "devtools workspace binary-artifact-sweep",
            "devtools workspace binary-artifact-sweep --json",
            "devtools workspace binary-artifact-sweep --limit 5000 --sample-limit 20",
        ),
    ),
    CommandSpec(
        "workspace binary-artifact-reclassify-apply",
        "workspace",
        "Persist raw_artifacts classification for binary-shaped raw rows.",
        "devtools.binary_artifact_reclassify_apply",
        use_when=(
            "polylogue-hbtj2: act on the sweep's report by writing raw_artifacts rows "
            "(materialize_artifact_observations) for the flagged binary content, so it "
            "carries an explicit non-session classification instead of sitting "
            "unclassified. Does not touch revision_authority or delete anything -- "
            "default is dry-run; --apply performs the write."
        ),
        examples=(
            "devtools workspace binary-artifact-reclassify-apply",
            "devtools workspace binary-artifact-reclassify-apply --json",
            "devtools workspace binary-artifact-reclassify-apply --apply",
        ),
    ),
    CommandSpec(
        "workspace tool-result-history-sweep",
        "workspace",
        "Find claude-code-session raw rows that should reclassify as tool-result/file-history sidecars.",
        "devtools.tool_result_history_sweep_report",
        use_when=(
            "polylogue-omsw: tool-results/<name> tool-call-overflow content and "
            "file-history-snapshot-only projects/<proj>/<uuid>.jsonl streams were "
            "historically admitted as independent claude-code-session raw_sessions rows "
            "instead of sidecars. Both classification gaps are closed for fresh "
            "acquisition (archive.artifact_taxonomy); this read-only report finds "
            "already-ingested rows acquired before that fix landed. Never mutates "
            "source.db."
        ),
        examples=(
            "devtools workspace tool-result-history-sweep",
            "devtools workspace tool-result-history-sweep --json",
            "devtools workspace tool-result-history-sweep --limit 5000 --sample-limit 20",
        ),
    ),
    CommandSpec(
        "workspace tool-result-history-reclassify-apply",
        "workspace",
        "Persist raw_artifacts classification for tool-result/file-history-shaped raw rows.",
        "devtools.tool_result_history_reclassify_apply",
        use_when=(
            "polylogue-omsw: act on the sweep's report by writing raw_artifacts rows "
            "(materialize_artifact_observations) for the flagged tool-results/"
            "file-history-snapshot content, so it carries an explicit non-session "
            "classification instead of sitting misclassified as "
            "coordinator_session_stream. Does not touch revision_authority, "
            "index.db, or delete anything -- default is dry-run; --apply performs "
            "the write. Per the 2026-07-22 hook-inflation precedent, contaminated "
            "rows are reclassified, never deleted."
        ),
        examples=(
            "devtools workspace tool-result-history-reclassify-apply",
            "devtools workspace tool-result-history-reclassify-apply --json",
            "devtools workspace tool-result-history-reclassify-apply --apply",
        ),
    ),
    CommandSpec(
        "workspace antigravity-phantom-sweep",
        "workspace",
        "List antigravity-session rows that are brain-metadata phantom fragments.",
        "devtools.antigravity_phantom_sweep_report",
        use_when=(
            "polylogue-eo81 / polylogue-msia: every antigravity-session row (116/116, "
            "measured 2026-07-31) is a 1-message fragment materialized from a "
            "*.md.metadata.json sidecar under ~/.gemini/antigravity/brain/, tagged "
            "degraded:brain-metadata-fragment at ingest time (PR #1856). Real "
            "conversation content lives in conversations/*.pb, acquired separately by "
            "PR #3441. This read-only report lists the phantom candidates and their "
            "raw payload; never mutates index.db or source.db."
        ),
        examples=(
            "devtools workspace antigravity-phantom-sweep",
            "devtools workspace antigravity-phantom-sweep --json",
            "devtools workspace antigravity-phantom-sweep --limit 200 --sample-limit 20",
        ),
    ),
    CommandSpec(
        "workspace antigravity-phantom-purge-apply",
        "workspace",
        "Delete antigravity brain-metadata phantom sessions and reclassify their raw rows.",
        "devtools.antigravity_phantom_purge_apply",
        use_when=(
            "polylogue-eo81 / polylogue-msia: act on the sweep's report by deleting the "
            "flagged sessions (ArchiveStore.delete_sessions -- rebuild-safe, since PR "
            "#3441's ingest-side content classifier already refuses "
            "*.md.metadata.json as session content) and persisting a scoped "
            "raw_artifacts classification for their raw rows "
            "(materialize_artifact_observations_for_raw_ids). Default is dry-run; "
            "--apply performs both writes."
        ),
        examples=(
            "devtools workspace antigravity-phantom-purge-apply",
            "devtools workspace antigravity-phantom-purge-apply --json",
            "devtools workspace antigravity-phantom-purge-apply --apply",
        ),
    ),
    CommandSpec(
        "workspace agent-meta-sidecar-sweep",
        "workspace",
        "Find agent-*.meta.json subagent-sidecar phantom sessions (message_count=0).",
        "devtools.agent_meta_sidecar_sweep_report",
        use_when=(
            "polylogue-ioz7 (direct fix for the polylogue-b508 audit finding): before a "
            "producer bug was fixed 2026-07-28, a per-subagent metadata sidecar file "
            "(subagents/agent-<id>.meta.json) was materialized into the archive as its own "
            "empty claude-code-session row, a duplicate phantom beside the real, correctly-"
            "ingested subagent transcript. This read-only report classifies every zero-"
            "message index session against the bead's own repro predicate "
            "(message_count=0 AND raw_sessions.source_path LIKE '%.meta.json') -- 4,945 "
            "rows matched on the live archive as of 2026-08-02. Never mutates index.db or "
            "source.db."
        ),
        examples=(
            "devtools workspace agent-meta-sidecar-sweep",
            "devtools workspace agent-meta-sidecar-sweep --json",
            "devtools workspace agent-meta-sidecar-sweep --limit 500 --sample-limit 20",
        ),
    ),
    CommandSpec(
        "workspace agent-meta-sidecar-purge-apply",
        "workspace",
        "Purge agent-*.meta.json subagent-sidecar phantom sessions from index.db.",
        "devtools.agent_meta_sidecar_purge_apply",
        use_when=(
            "polylogue-ioz7: act on the sweep's report by deleting the flagged sessions rows "
            "via ArchiveStore.delete_sessions (the tested, incident-hardened bulk-delete "
            "primitive, not a plain per-row DELETE) and writing an immutable receipt per "
            "purged row. raw_sessions rows and blobs in source.db are never touched, and "
            "re-ingest cannot resurrect a purged row (the producer bug is already fixed). "
            "Default is dry-run; --apply requires --backup-manifest pointing at a verified "
            "backup that includes index.db (the default backup profile omits it -- use "
            "'polylogue backup --profile full_evidence --verify')."
        ),
        examples=(
            "devtools workspace agent-meta-sidecar-purge-apply",
            "devtools workspace agent-meta-sidecar-purge-apply --json",
            "devtools workspace agent-meta-sidecar-purge-apply --apply "
            "--backup-manifest /realm/staging/polylogue-backup/manifest.json",
        ),
    ),
    CommandSpec(
        "workspace unknown-export-reclassification",
        "workspace",
        "Re-run the fixed browser-capture provider probe against stored unknown-export rows.",
        "devtools.unknown_export_reclassification_report",
        use_when=(
            "polylogue-mvq8: before acting on browser-capture raw_sessions rows stamped "
            "unknown-export by the pre-fix 1MiB prefix probe, get a read-only classification "
            "of how many are re-detectable under the fixed structural (ijson) provider scan. "
            "Never mutates origin or blob state; rewriting origin / re-parsing / re-indexing "
            "a 'reclassifiable' row is a separate, explicitly operator-authorized step."
        ),
        examples=(
            "devtools workspace unknown-export-reclassification",
            "devtools workspace unknown-export-reclassification --json",
            "devtools workspace unknown-export-reclassification --source-path-like '' --limit 500",
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
        "workspace basic-usage-demo-check",
        "workspace",
        "Re-run the basic-usage demo suite's commands and assert output shape.",
        "devtools.basic_usage_demo_check",
        use_when=(
            "Guard .agent/demos/basic-usage/ against silent regressions: re-executes each of the eight "
            "documented walkthroughs (find, read, search, resume, cost, lineage, MCP, status/health) against "
            "a freshly seeded demo archive and asserts non-empty/expected-shape output, not exact counts."
        ),
        examples=(
            "polylogue demo seed --root /tmp/polylogue-basic-usage-demo --force --with-overlays",
            "devtools workspace basic-usage-demo-check --archive-root /tmp/polylogue-basic-usage-demo",
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
        "verify agent-integration",
        "verification",
        "Verify manual compilation, parser examples, continuation, native delivery, packaging, and live cutover signatures.",
        "devtools.verify_agent_integration",
        use_when="Validate the six-tool manual or native integration; add --require-live after the MCP cutover lands.",
        examples=(
            "devtools verify agent-integration",
            "devtools verify agent-integration --json",
            "devtools verify agent-integration --require-live",
        ),
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
        "lab policy classifier-fingerprints",
        "verification lab",
        "Verify parser/classifier decision-boundary changes are declared as reparse-requiring or acknowledged.",
        "devtools.verify_classifier_fingerprints",
        use_when=(
            "Catch the gap `lab policy schema-versioning` cannot see (polylogue-gucv): a parser/classifier "
            "under polylogue/sources/ or polylogue/archive/artifact_taxonomy/ (looks_like*/classify_artifact* "
            "functions) changes what it accepts for identical input bytes without any INDEX_SCHEMA_VERSION "
            "bump at all, so already-indexed rows go silently stale with no signal a reparse was needed "
            "(PR #3428 shipped exactly this, green, against the version-keyed gate)."
        ),
        examples=(
            "devtools lab policy classifier-fingerprints",
            "devtools lab policy classifier-fingerprints --json",
            "devtools lab policy classifier-fingerprints --ack polylogue/sources/parsers/foo.py:looks_like_x "
            "--reason 'only tightens a shape that never validly matched' --ref polylogue-abcd",
        ),
    ),
    CommandSpec(
        "lab policy raw-payload-hash-purity",
        "verification lab",
        "Verify no raw-capture write path splices a synthesized literal onto captured bytes before hashing.",
        "devtools.verify_raw_payload_hash_purity",
        use_when=(
            "Prevent a regression of polylogue-u19l's confirmed bug: sources/live/batch.py used to prepend a "
            "synthetic session_meta header onto every Codex append capture before hashing/storing it, so the "
            "stored blob was never a literal byte-slice of the live file, permanently defeating live-source "
            "byte-identity verification for ~59GB of raw rows. Statically forbids concatenating a synthesized "
            "literal (bytes/str constant, f-string, json.dumps()/.encode() result) onto captured bytes anywhere "
            "in the raw-capture write-path modules (WRITE_PATH_MODULES)."
        ),
        examples=("devtools lab policy raw-payload-hash-purity", "devtools lab policy raw-payload-hash-purity --json"),
    ),
    CommandSpec(
        "lab policy position-derived-identity",
        "verification lab",
        "Verify no parser mints cross-revision comparison identity from positional/index data.",
        "devtools.verify_position_derived_identity",
        use_when=(
            "Prevent a regression of polylogue-hith/qkuq's already-fixed attachment-id bug (a synthetic id "
            "seeded partly by array index was unstable across export vintages that reorder/insert entries, "
            "manufacturing false divergence in revision-authority membership comparison) and catch new "
            "occurrences of the same shape (polylogue-gysk3 found the identical hazard still live for "
            "provider_message_id, the sole input to message_identity_hash). Statically scans "
            "polylogue/sources/parsers/ for an identity-bearing field constructed from an f-string/format/"
            "concatenation referencing a loop index/position variable, either inline or via a local variable."
        ),
        examples=(
            "devtools lab policy position-derived-identity",
            "devtools lab policy position-derived-identity --json",
            "devtools lab policy position-derived-identity --ack "
            "'polylogue/sources/parsers/foo.py:parse:provider_message_id' "
            "--reason 'tracked in polylogue-xxxx, not fixed inline' --ref polylogue-xxxx",
        ),
    ),
    CommandSpec(
        "lab policy raw-authority-frontier-executability",
        "verification lab",
        "Verify every raw-authority frontier state has a reachable actuator.",
        "devtools.verify_raw_authority_frontier_executability",
        use_when=(
            "polylogue-w32w / polylogue-lb39z (Phase 1, item 4): "
            "RawAuthorityFrontierItem.__post_init__ raises if a CONSTRUCTED item pairs a "
            "dispatched actuator (_APPLY_DISPATCHED_ACTUATORS) with a non-executable state "
            "(_EXECUTABLE_STATES) -- but that only fires when a test or live classification "
            "actually builds one; a new frontier state or re-paired actuator can ship an "
            "unexercised branch that stays silent until it accumulates against real archive "
            "data (the original defect: 4,174 blockers demanding an unreachable actuator, "
            "undetected for weeks). This lint statically enumerates every literal "
            "(state, actuator) construction site in polylogue/storage/raw_reconciler.py "
            "(_item(...) and _StrategyOverride(...) calls) and re-checks the same invariant "
            "at review time, independent of test coverage."
        ),
        examples=(
            "devtools lab policy raw-authority-frontier-executability",
            "devtools lab policy raw-authority-frontier-executability --json",
        ),
    ),
    CommandSpec(
        "lab policy backlog-hygiene",
        "verification lab",
        "Verify Beads backlog structure invariants (.beads/issues.jsonl).",
        "devtools.verify_backlog_hygiene",
        use_when=(
            "Enforce the standing backlog-hygiene invariant lint (polylogue-8jg9.1): 20 checks "
            "over the Beads export catching dangling dependency refs, blocks-cycles, missing "
            "horizon/AC/design content on tech-tree beads, P0/P1 beads without acceptance "
            "criteria, unlabeled non-epic beads, epics with no members or description, stale "
            "'adopted' decisions left open, duplicate titles, bead ids named but never "
            "created, an unclean/corrupt bd JSONL sync receipt (S1, consuming "
            "polylogue-gxjh.1's monotonic sync contract), an active leaf that is itself an "
            "epic (F1), an active leaf with a missing/dangling/mismatched program ref (F2), a "
            "stale in_progress claim with no recent activity (F3, configurable window), and a "
            "frontier_program=active program with no admitted active leaves (F4) -- catches "
            "backlog structure drift before it needs an archaeology sweep to recover, instead "
            "of only a manually-invoked script. Also reports a non-blocking active-set size "
            "diagnostic (soft target/warn bands, never a hard cap or failure)."
        ),
        examples=(
            "devtools lab policy backlog-hygiene",
            "devtools lab policy backlog-hygiene --json",
            "devtools lab policy backlog-hygiene --fresh",
            "devtools lab policy backlog-hygiene --stale-claim-days 10",
        ),
    ),
    CommandSpec(
        "lab policy bead-graph",
        "verification lab",
        "Bead-graph invariant lint over live `bd` state (cycles, wave labels/inversions, missing AC).",
        "devtools.verify_bead_graph",
        use_when=(
            "Run right before shipping a bead-state delta (matches the sinex bead-graph-lint "
            "convention). Checks LIVE `bd dep cycles` / `bd list --all --json` output rather than "
            "the exported .beads/issues.jsonl snapshot, so it catches drift not yet re-exported. "
            "INTENTIONAL DIVERGENCE from sinex: only duplicate `wave:` labels are flagged "
            "(polylogue's `lane:`/`delivery:`/`horizon:` taxonomy is local and not enforced here)."
        ),
        examples=("devtools lab policy bead-graph",),
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
        "lab policy campaign-archive-boundaries",
        "verification lab",
        "Verify devtools synthetic benchmark/scale campaigns route through ArchiveLocation.",
        "devtools.verify_campaign_archive_boundaries",
        use_when=(
            "Catch a regression of the phantom-benchmark.db bug (polylogue-ovme.3): a campaign "
            "reintroducing a 'benchmark.db' sentinel, an ad hoc tier-path sibling derivation, or "
            "an entry point (generate_archive/run_full_campaign/run_campaign._run) that no longer "
            "routes through CampaignArchiveLocation. Scoped to the devtools campaign boundary only "
            "-- the broader storage/diagnostics/daemon/maintenance/transitions boundary audit is "
            "polylogue-ovme.2's migration surface."
        ),
        examples=(
            "devtools lab policy campaign-archive-boundaries",
            "devtools lab policy campaign-archive-boundaries --json",
        ),
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
        "lab schema commit",
        "verification lab",
        "Persist a real full-corpus schema generation into committed provider packages.",
        "devtools.schema_commit",
        use_when=(
            "Actually regenerate and write `polylogue/schemas/providers/<provider>/versions/...` from the live "
            "archive -- 'lab schema generate' only ever previews and never writes committed package files."
        ),
        examples=(
            "devtools lab schema commit --provider chatgpt --full-corpus --dry-run",
            "devtools lab schema commit --provider chatgpt --full-corpus",
        ),
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
        "lab schema parser-diff",
        "verification lab",
        "List observed provider wire keys that no parser references.",
        "devtools.schema_parser_diff",
        use_when=(
            "Scope a parser batch by evidence before a rebuild: ranks every schema key nothing reads by how "
            "many records actually carry it. Output is a triage queue, not a verdict -- parser-side matching "
            "is name-based, so read the parser before acting on a row."
        ),
        examples=(
            "devtools lab schema parser-diff",
            "devtools lab schema parser-diff --provider codex --min-encountered 1000",
            "devtools lab schema parser-diff --json",
        ),
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
        "Render the pytest-first evidence dashboard.",
        "devtools.evidence_dashboard",
        use_when=(
            "Inspect pytest health, contract-evidence inventory, coverage, SLO "
            "catalog, static-gate status, and campaign freshness."
        ),
        examples=(
            "devtools verify evidence --json",
            "devtools verify evidence --markdown",
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
    CommandSpec(
        "workspace mandate-continuity-replay",
        "workspace",
        "Wire t8t continuity scenarios + work-evidence effects + discovery into one mandate artifact.",
        "devtools.mandate_continuity_replay",
        use_when=(
            "Run the polylogue-z9gh.7 terminal mandate gate: replay the polylogue-t8t continuity scenario "
            "catalog over real MCP stdio JSON-RPC, reconcile this repository's own real git+Beads history "
            "through the polylogue-1vpm.6.2 effect adapters, cross-check every query-tool route step against "
            "the polylogue-z9gh.3 query-discovery catalog, and emit one JSON artifact with a mandate "
            "acceptance-criteria matrix. Defaults to a fresh, privacy-safe synthetic archive; pass "
            "--archive-root for an authorized live-scale replay."
        ),
        examples=(
            "devtools workspace mandate-continuity-replay",
            "devtools workspace mandate-continuity-replay --output .cache/mandate-continuity-replay.json",
            "devtools workspace mandate-continuity-replay --archive-root /path/to/authorized/archive --keep-archive",
        ),
    ),
    CommandSpec(
        "workspace beads-state-report",
        "workspace",
        "Self-contained HTML state-of-the-backlog report over the whole bead population.",
        "devtools.beads_state_report",
        use_when=(
            "Answer 'what shape is the backlog in?' across the whole bead population -- open "
            "AND closed -- rather than the ready frontier. Computes status x priority x type, "
            "epic/program trees with fill bars and per-epic trend sparklines, the blocks-graph "
            "topology (ready/blocked/top blockers/cycles/densest cluster/parallel frontier), a "
            "Pulse section reconstructing open/ready/blocked at now vs 7/14 days ago from "
            "timestamps, creation/closure velocity over trailing windows, an age x priority "
            "heatmap, graph-health review queues (dangling refs, id-vs-edge hierarchy "
            "disagreement, open parents whose children all closed, stale in-progress claims, "
            "duplicate titles), subsystem concentration by both area label and keyword, and the "
            "dated hand-verified VERIFICATION(...) / RECONCILIATION marker subsets. Its findings "
            "list is generated by conditional checks over the data, so regeneration cannot leave "
            "stale claims. Answers a different question than `workspace bead-cluster` "
            "(execution-frontier footprint clustering) and `workspace delivery-gate-status` "
            "(per-gate progress): this is population shape and graph hygiene, not the next batch "
            "to dispatch. Use --fresh, since bd mutations do not immediately re-export."
        ),
        examples=(
            "devtools workspace beads-state-report --fresh",
            "devtools workspace beads-state-report --out /tmp/beads-state.html",
            "devtools workspace beads-state-report --json",
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
