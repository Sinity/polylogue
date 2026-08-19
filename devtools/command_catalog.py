"""Shared command catalog for repository developer tools."""

from __future__ import annotations

import importlib
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

CommandMain = Callable[[list[str] | None], int]
CONTROL_PLANE = "devtools"

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
        "why",
        "core",
        "Explain the most recent verification run, or where verification time went.",
        "devtools.why",
        use_when="A verify failed, bootstrapped unexpectedly, or refused to run, and you want the cause without reading receipt JSON by hand.",
        examples=(
            "devtools why",
            "devtools why --history 24",
            "devtools why --run 20260817T213631Z-testmon-2709409-d5c6e72c",
        ),
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
        "render docs-surface",
        "generated surfaces",
        "Render docs/README.md and the README documentation table.",
        "devtools.render_docs_surface",
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
        "verify ci-commands",
        "verification",
        "Validate devtools invocations in structured CI run fields.",
        "devtools.verify_ci_commands",
        use_when="Catch CI scripts that reference a removed or misspelled devtools command.",
        examples=("devtools verify ci-commands", "devtools verify ci-commands --json"),
    ),
    CommandSpec(
        "verify doc-commands",
        "verification",
        "Validate executable documentation examples against live command inventories.",
        "devtools.verify_doc_commands",
        use_when="Catch README and documentation examples that reference an unknown command path or flag.",
        examples=("devtools verify doc-commands", "devtools verify doc-commands --json"),
    ),
    CommandSpec(
        "verify corpus-fidelity",
        "verification",
        "Run the production corpus-fidelity acceptance gate against an archive root.",
        "devtools.corpus_fidelity",
        use_when=(
            "Run after a promoted index rebuild, alongside `polylogue ops maintenance verify-archive`, "
            "to prove source-backed corpus absence, attachment, and revision fidelity from the registry."
        ),
        examples=(
            "devtools verify corpus-fidelity --archive-root /path/to/archive",
            "devtools verify corpus-fidelity --archive-root /path/to/archive --json",
        ),
    ),
    CommandSpec(
        "verify schema-inference-gate",
        "verification",
        "Run the read-only schema-inference prerequisite and persist a PASS/FAIL receipt.",
        "devtools.schema_inference_gate",
        use_when=(
            "Run before schema inference or the 818fy rebuild. Declare every external source root represented in "
            "source.db; the command scans those roots and runs BlobStore's full verifier without mutating the archive."
        ),
        examples=(
            "devtools verify schema-inference-gate --archive-root /path/to/archive "
            "--ground-truth-root codex-session=/path/to/codex --receipt /path/to/schema-inference-gate-receipt.json",
            "devtools verify schema-inference-gate --archive-root /path/to/archive "
            "--ground-truth-root codex-session=/path/to/codex --receipt /path/to/schema-inference-gate-receipt.json --json",
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
        "verify mutation-freshness",
        "verification",
        "Verify executable mutation campaigns meet the selected freshness and kill-rate thresholds.",
        "devtools.verify_mutation_freshness",
        use_when=(
            "Enforce mutation campaign freshness and kill-rate thresholds after a rotating CI campaign "
            "has produced its local artifacts."
        ),
        examples=(
            "devtools verify mutation-freshness --enforce-kill-rate",
            "devtools verify mutation-freshness --strict --default-freshness-days 30",
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
        "Plan and prove a declared index fast-forward against retained raw replay.",
        "devtools.index_fast_forward",
        use_when=(
            "Advance a stopped index generation across a declared clone-safe schema gap. The actuator clones the "
            "active generation, applies lifecycle operations, proves a deterministic retained-raw sample through "
            "the production parser/materializer route, then atomically activates the proven generation."
        ),
        examples=(
            "devtools workspace index-fast-forward prepare --archive-root /path/to/archive --receipt /path/to/receipt.json",
            "devtools workspace index-fast-forward activate --receipt /path/to/receipt.json",
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
        "workspace bead-cluster",
        "workspace",
        "Footprint/overlap/contention clustering of ready Beads (execution frontier).",
        "devtools.bead_cluster",
        use_when=(
            "Before dispatching a batch of ready beads, see which ones share a file/package "
            "footprint (cluster into one branch/PR sweep) vs. which are genuinely disjoint "
            "(safe to run as parallel worktree lanes), and flag contention -- beads that touch "
            "the same durable migration slot or the same generated surface even without an "
            "exact file-path overlap. Footprints are advisory, so verify them when claiming work."
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
        "workspace pr-scope",
        "workspace",
        "Render stable PR scope intent and inspect its mutable merge attestation.",
        "devtools.pr_scope",
        use_when=(
            "Before publishing a non-draft PR, render the machine-readable v2 intent from assigned and mutated "
            "Beads plus typed dispositions. The PR body stays stable across commits; `sync --pr` reports the "
            "current head- and Bead-bound attestation consumed by merge-gate. CircleCI quick-gate and "
            "`workspace merge` run the same validator; the command never interprets acceptance prose."
        ),
        examples=(
            "devtools workspace pr-scope render --input .agent/pr-scope.json > /tmp/pr-scope.md",
            "devtools workspace pr-scope check --pr 3517",
            "devtools workspace pr-scope sync --pr 3517",
            "devtools workspace pr-scope check-ci --pr 3517 --repo Sinity/polylogue --expected-head-sha $(git rev-parse HEAD)",
            "devtools workspace pr-scope check --body-file pr-body.md --head-sha $(git rev-parse HEAD) --base-sha $(git rev-parse origin/master)",
        ),
    ),
    CommandSpec(
        "workspace merge-gate",
        "workspace",
        "Structural pre-merge safety check: fresh local verification + resolved review threads.",
        "devtools.merge_gate",
        use_when=(
            "Immediately before squash-merging any PR in a merge train, replacing coordinator memory "
            "(grace-period comment polling, remembering to run the broader local test suite CI skips "
            'per-PR) with a check that fails closed. `record <PR> --command "..."` requires the current '
            "checkout to already be the PR's exact head commit with a clean tree (it refuses otherwise), "
            "first validates the same versioned PR-scope carrier CircleCI checks, then runs a local verification "
            "command and persists its typed verification scope. `check <PR>` polls structured GitHub "
            "review-thread state across a real grace window (default 3x20s, covering CodeRabbit's "
            "30-60s late-arrival window) and BLOCKs unless a receipt exists for the CURRENT head sha "
            "within a freshness window with exit_code 0, every review thread is resolved, and GitHub's "
            "review decision is not CHANGES_REQUESTED. The receipt binds the carrier digest plus a fresh "
            "head- and Bead-bound scope attestation, so a changed scope or Bead state requires re-recording. "
            "Motivated by two 2026-08-01 incidents: PR #3502 merged before CodeRabbit's findings "
            "posted, and PR #3517 nearly merged with a 43-test regression no CI check or review comment "
            "ever flagged -- plus review findings on this tool itself (recording from an unrelated "
            "checkout, a --quick example that would have missed its own motivating regression, and a single "
            "review snapshot instead of a grace-period poll). `check --post-status` (polylogue-1cbeh) posts the "
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
            "fired only if a coordinator remembered to invoke them). `merge <PR>` validates the non-draft PR's structured scope carrier and auto-records a merge-gate receipt if none is fresh for the current head sha (running `--command`, "
            'default "devtools verify"), runs `merge-gate check` and refuses to merge on any '
            "BLOCK, strips a doubled `(#N) (#N)` squash-subject suffix (the 2026-07-12/13 "
            "incident), then runs the actual `gh pr merge --squash`. `--dry-run` runs every check "
            "without merging. `--with-verify` immediately runs and records the merge-train's "
            "terminal full-suite verify after merging; otherwise it prints a reminder. "
            "`train-status` reports (exit 1) any PRs merged since the last recorded terminal "
            "verify. The terminal step records itself: a plain `devtools verify` that earns "
            "release-baseline authority at origin/master writes the ledger entry."
        ),
        examples=(
            "devtools workspace merge 3517",
            'devtools workspace merge 3517 --command "devtools test tests/unit/foo.py"',
            "devtools workspace merge 3517 --dry-run",
            "devtools workspace merge 3517 --with-verify",
            "devtools workspace merge train-status",
        ),
    ),
    CommandSpec(
        "workspace bead-reimport-guard",
        "workspace",
        "Monotonic, receipted guard/reconcile/export for bd's JSONL synchronization.",
        "devtools.bd_reimport_guard",
        use_when=(
            "Beads hook auto-import is disabled in .beads/config.yaml because linked worktrees "
            "share live Dolt state but retain stale branch-point JSONL snapshots. "
            "`reconcile <candidate.jsonl>` is the explicit, hand-invokable path for intentional "
            "JSONL imports, including recovery after `git reset --hard`. `export <target.jsonl>` "
            "does an atomic, validated snapshot export. Reconciliation writes a machine-readable SyncReceipt (per-row "
            "new/updated/equal/skipped_downgrade/conflicted outcome) under "
            ".cache/bd-sync-receipts/; ordinary sync never applies a downgrade -- recovering one "
            "on purpose requires reconcile --allow-downgrade plus --actor/--reason."
        ),
        examples=(
            "devtools workspace bead-reimport-guard reconcile .beads/issues.jsonl",
            "devtools workspace bead-reimport-guard export /tmp/issues-snapshot.jsonl",
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
        "workspace raw-failure-disposition-apply",
        "workspace",
        "Apply reviewed terminal dispositions to historical raw parse failures.",
        "devtools.raw_failure_disposition_apply",
        use_when=(
            "polylogue-dyica: terminal historical parse failures predate typed lifecycle evidence. "
            "Default is dry-run; --apply requires an exact JSONL disposition manifest and a verified "
            "source-tier backup. It retains raw bytes and parser errors while replacing only the stale "
            "artifact classification and writing immutable per-row receipts."
        ),
        examples=(
            "devtools workspace raw-failure-disposition-apply --manifest-path /realm/staging/raw-failure-disposition.jsonl",
            "devtools workspace raw-failure-disposition-apply --apply --manifest-path /realm/staging/raw-failure-disposition.jsonl "
            "--backup-manifest /realm/staging/polylogue-backup/manifest.json",
        ),
    ),
    CommandSpec(
        "workspace raw-append-chain-backfill-apply",
        "workspace",
        "Promote membershipless append raws proven correct by live-source verification.",
        "devtools.raw_append_chain_backfill_apply",
        use_when=(
            "polylogue-lb39z (Phase 1, item 3): 2,712 raw_sessions rows (measured 2026-08-02) are "
            "revision_kind='append', revision_authority='quarantined', and have no "
            "raw_session_memberships row at all -- a genuine fixed point, because the only mechanism "
            "that ever promotes an append raw (_promote_contiguous_append_evidence) requires its "
            "byte-contiguous predecessor to already be byte_proven. This proves each such row's own "
            "claimed byte range directly against its live source file's current bytes, independent of "
            "any ancestor's authority. Default is dry-run; --apply requires --backup-manifest pointing "
            "at a verified source-tier backup (polylogue backup --output-dir <dir> --verify). Writes an "
            "immutable per-row receipt (raw_append_chain_backfill_receipts); never runs blob GC or "
            "VACUUM -- that is a separate, later step."
        ),
        examples=(
            "devtools workspace raw-append-chain-backfill-apply",
            "devtools workspace raw-append-chain-backfill-apply --json",
            "devtools workspace raw-append-chain-backfill-apply --apply "
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
        "workspace raw-quarantine-group-dedup-apply",
        "workspace",
        "Promote one representative raw per fully-quarantined byte-identical (source_path, blob_hash) group.",
        "devtools.raw_quarantine_group_dedup_apply",
        use_when=(
            "polylogue-zm4w8 (measured live 2026-08-03): 1,777 raw_sessions rows (22.2 GiB) among "
            "the codex-session quarantine backlog are pure redundant duplicates -- same source_path "
            "AND same blob_hash as another raw_sessions row -- where EVERY member of the group is "
            "still quarantined (no indexed twin anywhere), invisible to "
            "raw-byte-duplicate-supersession-apply (which only matches a quarantined raw against an "
            "already-INDEXED twin). Default is dry-run; --apply requires --backup-manifest pointing "
            "at a verified source-tier backup (polylogue backup --output-dir <dir> --verify). "
            "Materializes exactly one representative raw per group through the real ingest pipeline "
            "(ParsingService.parse_from_raw -> write_parsed_session_to_archive -> "
            "refresh_session_insights_bulk) so it becomes a genuine indexed session, then marks the "
            "rest revision_authority='byte_proven' with an immutable per-row receipt "
            "(raw_quarantine_group_dedup_receipts) pointing at the promoted representative's raw_id "
            "and new session_id. Never deletes blobs or runs GC/VACUUM -- those are separate, later "
            "steps."
        ),
        examples=(
            "devtools workspace raw-quarantine-group-dedup-apply",
            "devtools workspace raw-quarantine-group-dedup-apply --json",
            "devtools workspace raw-quarantine-group-dedup-apply --apply "
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
        "workspace raw-authority-artifact-census",
        "workspace",
        "Census quarantined raws into five authority buckets; apply pages raw_artifacts upserts and records durable receipts.",
        "devtools.raw_authority_artifact_census",
        use_when=(
            "Run one full read-only quarantine census through inspect_raw_artifact. The optional --apply is bounded, "
            "requires a verified source-tier backup and --json. Apply logically upserts only raw_artifacts observations, "
            "writes an immutable source-tier receipt, and resumes only with its returned --census-id plus --after-raw-id; it never changes "
            "raw_sessions rows, revision authority, index rows, or blobs."
        ),
        examples=(
            "devtools workspace raw-authority-artifact-census --json --receipt /realm/tmp/work/raw-census.json",
            "devtools workspace raw-authority-artifact-census --apply --backup-manifest /path/to/manifest.json --json",
            "devtools workspace raw-authority-artifact-census --apply --census-id <id> --after-raw-id raw-000500 --backup-manifest /path/to/manifest.json --json",
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
        "workspace unknown-export-reclassification-apply",
        "workspace",
        "Reclassify proven ChatGPT browser-capture raws and write durable receipts.",
        "devtools.unknown_export_reclassification_apply",
        use_when=(
            "polylogue-s8s54: act only on rows whose complete browser-capture envelope proves "
            "session.provider='chatgpt'. Default is dry-run; --apply requires a verified source-tier "
            "backup manifest and remains locked to the measured ChatGPT spool. The source-tier origin/capture_mode update is receipt-backed, while the "
            "generated index session identity remains untouched for the normal reparse route."
        ),
        examples=(
            "devtools workspace unknown-export-reclassification-apply",
            "devtools workspace unknown-export-reclassification-apply --json",
            "devtools workspace unknown-export-reclassification-apply --apply "
            "--backup-manifest /realm/staging/polylogue-backup/manifest.json",
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
            "devtools workspace lineage-validation --out-dir .local/evidence/lineage-validation/current",
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
            "devtools workspace affordance-usage --out-dir .local/evidence/agent-affordance-usage",
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
            "devtools workspace degraded-archive-proof --out-dir .local/evidence/degraded-archive-proof/current --json",
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
        "lab policy bead-graph",
        "verification lab",
        "Validate typed dependency endpoints, uniqueness, parent cardinality, and cycles in the Beads graph.",
        "devtools.verify_bead_graph",
        use_when=(
            "Run before shipping a bead-state delta. With no source option it checks live `bd` state; "
            "`--export .beads/issues.jsonl` validates the branch snapshot without importing it into the "
            "shared database. The gate reads dependency records only and does not make prose, labels, or "
            "campaign-specific edge lists machine authority."
        ),
        examples=(
            "devtools lab policy bead-graph",
            "devtools lab policy bead-graph --export .beads/issues.jsonl --json",
        ),
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
        "bench memory",
        "benchmarking",
        "Measure query-memory envelopes on generated fixtures.",
        "devtools.query_memory_budget",
        use_when="Assert memory budgets around a concrete query or archive-facing command.",
        examples=("devtools bench memory --max-rss-mb 1536 -- polylogue --plain analyze",),
    ),
    CommandSpec(
        "lab run",
        "verification lab",
        "Run a named archive verification scenario.",
        "devtools.lab_scenario",
        entrypoint="run_main",
        use_when="Run a scenario such as rebuild-safety through the direct lab command path.",
        examples=(
            "devtools lab run rebuild-safety",
            "devtools lab run rebuild-safety --report-dir .cache/rebuild-safety-report --json",
        ),
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
        "Run focused mutation campaigns with isolated execution and JSON artifacts.",
        "devtools.mutmut_campaign",
        use_when="Run or inspect focused mutation-testing work without shrinking the committed mutmut scope.",
        examples=("devtools bench mutation list", "devtools bench mutation run filters"),
        featured=True,
    ),
    CommandSpec(
        "bench nightly-compare",
        "benchmarking",
        "Compare nightly pytest-benchmark output with the committed baseline.",
        "devtools.benchmark_compare_nightly",
        use_when="Check the nightly scale benchmark result against its committed regression threshold.",
        examples=(
            "devtools bench nightly-compare tests/benchmarks/baselines/nightly-baseline.json nightly-results.json",
            "devtools bench nightly-compare baseline.json candidate.json --fail-pct 20",
        ),
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
        "workspace continuity-evidence",
        "workspace",
        "Replay continuity scenarios and verify their query routes are discoverable.",
        "devtools.continuity_evidence",
        use_when=(
            "Replay the continuity scenario catalog over MCP stdio JSON-RPC and cross-check "
            "its query routes against discovery. The default seeds the packaged synthetic corpus. "
            "A supplied --archive-root must be paired with the exact --catalog that describes it; "
            "the runner rejects an unrelated live archive rather than applying synthetic oracles."
        ),
        examples=(
            "devtools workspace continuity-evidence",
            "devtools workspace continuity-evidence --output .cache/continuity-evidence.json",
            "devtools workspace continuity-evidence --archive-root /path/to/archive --catalog /path/to/catalog.json",
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
    return tuple(spec for spec in commands if spec.category == "verification lab")


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
    "verification_lab_command_specs",
]
