# Developer Tools

Use `devtools` for routine repository maintenance. Call individual
`devtools/*.py` modules directly only when you are editing these tools.

It exposes both human and JSON discovery/status forms. Use the JSON forms for
scripts and agents.

## Command Ownership Policy

`devtools` is the repository control plane. It owns orchestration around local
repo readiness: generated-surface rendering, baseline verification, validation
lane dispatch, package/build checks, and branch/PR readiness gates.

Domain validation semantics belong in lab, schema, scenario, or insight
modules first. A `devtools` command may expose them only as a thin operator
entrypoint that delegates to the owning executable check implementation.

Routine command placement:

- keep repo state, rendering, packaging, and PR-readiness orchestration in
  `devtools`;
- keep archive/insight workflows in `polylogue` CLI/API surfaces;
- keep evidence/scenario behavior in lab modules with executable command entrypoints;
- prefer validation lanes and `devtools verify --lab` to compose executable
  lab checks rather than duplicating domain checks inside `devtools verify`.

<!-- BEGIN GENERATED: devtools-command-catalog -->
## Command Catalog

Use these discovery commands before scripting or dispatching subcommands:

```bash
devtools --help
devtools --list-commands
devtools --list-commands --json
devtools status
devtools status --json
```

## Executable Lab Checks

These commands are thin wrappers around concrete schema, provider, pipeline, smoke, and lane checks.
They are not a proof ledger or end-user archive workflow.

| Command | Role |
| --- | --- |
| `devtools lab graph` | Inspect the authored runtime graph and see which scenarios currently cover declared artifacts and operations. |
| `devtools lab lanes` | List, dry-run, or execute authored validation lanes from the executable lane registry. |
| `devtools lab policy backlog-hygiene` | Enforce the standing backlog-hygiene invariant lint (polylogue-8jg9.1): 20 checks over the Beads export catching dangling dependency refs, blocks-cycles, missing horizon/AC/design content on tech-tree beads, P0/P1 beads without acceptance criteria, unlabeled non-epic beads, epics with no members or description, stale 'adopted' decisions left open, duplicate titles, bead ids named but never created, an unclean/corrupt bd JSONL sync receipt (S1, consuming polylogue-gxjh.1's monotonic sync contract), an active leaf that is itself an epic (F1), an active leaf with a missing/dangling/mismatched program ref (F2), a stale in_progress claim with no recent activity (F3, configurable window), and a frontier_program=active program with no admitted active leaves (F4) -- catches backlog structure drift before it needs an archaeology sweep to recover, instead of only a manually-invoked script. Also reports a non-blocking active-set size diagnostic (soft target/warn bands, never a hard cap or failure). |
| `devtools lab policy bead-graph` | Run right before shipping a bead-state delta (matches the sinex bead-graph-lint convention). Checks LIVE `bd dep cycles` / `bd list --all --json` output rather than the exported .beads/issues.jsonl snapshot, so it catches drift not yet re-exported. INTENTIONAL DIVERGENCE from sinex: only duplicate `wave:` labels are flagged (polylogue's `lane:`/`delivery:`/`horizon:` taxonomy is local and not enforced here). |
| `devtools lab policy campaign-archive-boundaries` | Catch a regression of the phantom-benchmark.db bug (polylogue-ovme.3): a campaign reintroducing a 'benchmark.db' sentinel, an ad hoc tier-path sibling derivation, or an entry point (generate_archive/run_full_campaign/run_campaign._run) that no longer routes through CampaignArchiveLocation. Scoped to the devtools campaign boundary only -- the broader storage/diagnostics/daemon/maintenance/transitions boundary audit is polylogue-ovme.2's migration surface. |
| `devtools lab policy demo-packet-registry` | Enforce the 212 portfolio contract (polylogue-212.7): every demo prompt in .agent/demos/registry.json must have a packet directory carrying PROMPT.md, finding.yaml (five-part provenance stanza), report.md (fixed section order), evidence.ndjson, queries.ndjson, checks.json, and run.log. Catches a missing or malformed packet before it silently drops out of the demo shelf. |
| `devtools lab policy demo-tour-freshness` | Catch drift between what `polylogue demo tour` actually emits at runtime (transcript, report, per-step command output, recording tape) and the committed copies under docs/examples/demo-tour/, modulo an explicit wall-clock-duration mask (polylogue-3tl.17). Runs the real tour (~10s), so it lives in the lab tier rather than --quick. |
| `devtools lab policy docs-drift` | Catch doc-vs-code drift in the hand-maintained Reference-docs table (CLAUDE.md): a backtick-quoted file path that no longer exists, a '<Tier> schema version N' claim ahead of the tier's current constant, or a watchlisted table name renamed to a different current name (e.g. `artifact_observations` renamed to `raw_artifacts`) still asserted as current (polylogue-9e5.13). |
| `devtools lab policy insight-honesty` | Enforce that polylogue.insights.registry.INSIGHT_REGISTRY and polylogue.insights.rigor's contract matrix/exemption list never drift apart (9e5.28) -- a registered product with neither a RigorContract nor a RIGOR_EXEMPT entry used to silently vanish from `polylogue ops insights audit` instead of showing as uncovered. |
| `devtools lab policy schema-versioning` | Enforce the policy boundary documented in docs/internals.md § 'Schema Versioning Model'. Durable tiers use explicit additive migrations with a backup gate; derived tiers are rebuilt or blue-green replaced from source evidence. |
| `devtools lab policy timestamp-doctrine` | Enforce the time doctrine (UTC epoch-ms canon, docs/internals.md) at DDL-review time (cpf.1): a TEXT timestamp in source.db/user.db re-introduces tz-unknown ambiguity and lexicographic-vs-temporal sort divergence, and durable tiers need an explicit additive migration to fix later -- catching it before merge is orders cheaper than a copy-forward migration after. |
| `devtools lab provider completeness` | Inspect detector, parser, fixture, schema, docs, ImportExplain, and caveat coverage before claiming a provider/importer mode is product-ready. |
| `devtools lab probe bead-pr-reconciliation` | After a merge-heavy stretch (a Workflow campaign, a merge train, or just several PRs landed close together), check for beads left open by a PR that referenced them -- catches the reconciliation gap where workers/agents are barred from closing beads themselves and no follow-up pass ever ran (2026-07-14: a 55-bead campaign left every bead open despite ~20 PRs merging clean). Advisory only -- reports candidates for a human/agent AC check, never auto-closes and never fails a gate. |
| `devtools lab probe capture-regression` | Turn a live or probe failure JSON summary into a replayable local regression artifact. |
| `devtools lab probe cost-reconciliation` | Validate archive token accounting against optional local Codex state_5.sqlite and Claude stats-cache.json before publishing cost or usage-analysis claims. |
| `devtools lab probe pipeline` | Run real pipeline stages and optionally capture emitted summaries as regression cases. |
| `devtools lab probe turso` | Collect executable evidence before changing production storage backends: Python binding availability, generated-column support, FTS compatibility, MVCC, CDC, vector functions, ATTACH, and WAL pragma behavior. |
| `devtools lab projections` | Inspect the unified projection inventory that feeds runtime coverage, generated docs, and control-plane maps. |
| `devtools lab smoke` | Run direct archive and reader smoke sets outside the archive CLI. |
| `devtools lab schema audit` | Check committed schema package quality gates without presenting them as normal archive usage. |
| `devtools lab schema commit` | Actually regenerate and write `polylogue/schemas/providers/<provider>/versions/...` from the live archive -- 'lab schema generate' only ever previews and never writes committed package files. |
| `devtools lab schema compare` | Review schema package drift between committed versions in the lab surface. |
| `devtools lab schema explain` | Inspect schema package annotations, semantic roles, and review evidence from the lab surface. |
| `devtools lab schema generate` | Refresh provider schema package artifacts from archive observations outside the archive CLI. |
| `devtools lab schema list` | Inspect committed provider schema package catalogs without presenting them as normal archive usage. |
| `devtools lab schema parser-diff` | Scope a parser batch by evidence before a rebuild: ranks every schema key nothing reads by how many records actually carry it. Output is a triage queue, not a verdict -- parser-side matching is name-based, so read the parser before acting on a row. |
| `devtools lab schema promote` | Turn reviewed schema evidence clusters into committed provider schema packages. |
| `devtools lab schema roundtrip` | Close the schema inference-validation loop: package manifests must roundtrip through typed models, and every supported element schema must be reachable from the runtime registry. |
| `devtools lab seed-receipt-compare` | Judge a seed-testmon or focused pytest run's resource receipt against a named baseline receipt (e.g. a prior incident) — confirms both runs share workload identity and terminated cleanly, then scores wall-time speedup and peak-PSS ceiling targets, naming a blocker and linked follow-up for any unmet target instead of silently dropping it. |
| `devtools lab snapshot read-surface` | Freeze archive read-surface behavior before archive work, then compare candidate archives against the captured envelope baseline. |
| `devtools lab test-economics` | Decide where test-writing effort or test-suite pruning actually pays off, by cross-referencing coverage percent, historical fix-commit density, testmon wall-time cost exposure, and testmon selection fan-out per top-level package. |
| `devtools lab testmon-blind-spots` | Inspect an existing coverage JSON report against an existing pytest-testmon database. Declaration-only modules are reported separately from executable validator risk; this command does not run pytest or regenerate coverage. |
| `devtools lab testmon-proof` | Validate the affected-test harness itself: a disposable copy of a real Polylogue module and existing route test is seeded, semantically mutated, edge-severed, restored, and checked for bounded unrelated-change selection. |

## Core Loop

These are the commands worth remembering during normal repo work:

- `devtools status`: Check repo state, generated-surface drift, and the next default verification steps.
  Common forms: `devtools status`, `devtools status --json`, `devtools status --verify-generated`.
- `devtools render all`: Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.
  Common forms: `devtools render all`, `devtools render all --check`.
- `devtools verify`: Run format, lint, mypy, render all, and test checks locally before pushing.
  Common forms: `devtools verify`, `devtools verify --quick`, `devtools verify --lab`.
- `devtools test`: Run a specific test file, directory, or -k/-m selection in the inner loop without invoking raw pytest.
  Common forms: `devtools test tests/unit/pipeline`, `devtools test -k hybrid`, `devtools test tests/unit/storage -x`.
- `devtools bench mutation`: Run or inspect focused mutation-testing work without shrinking the committed mutmut scope.
  Common forms: `devtools bench mutation list`, `devtools bench mutation run filters`.
- `devtools bench campaign`: Record durable benchmark artifacts or compare a candidate run against a baseline artifact.
  Common forms: `devtools bench campaign list`, `devtools bench campaign run search-filters`, `devtools bench campaign compare baseline.json candidate.json`.

## Workspace disposition audit

The utf.1 triage retains workspace commands with operator history, reusable output, or focused tests. The stale archive-schema-fast-forward name is removed in favor of the registered index fast-forward command.

| Named entry | Disposition | Evidence | Replacement |
| --- | --- | --- | --- |
| `devtools workspace index-fast-forward` | `retain` | Recent production commits plus focused devtools/storage tests; emits a reusable proof receipt. | Keep the registered command as the index-tier actuator. |
| `workspace archive-schema-fast-forward` | `remove` | No current CommandSpec, implementation module, focused test, or history entry exists for this name. | Use workspace index-fast-forward for declared derived-index fast-forwards. |
| `devtools workspace degraded-archive-proof` | `retain` | Focused tests and deterministic self-healing proof artifacts cover the command. | Keep the registered command for archive repair evidence. |
| `devtools workspace frontier` | `retain` | Operator-facing frontier report with documented workflow use and structured report output. | Keep the registered command for frontier batching and wait-ahead decisions. |
| `devtools workspace temporal-read-profile` | `retain` | Focused tests cover the report and JSON timing output is reusable for read tuning. | Keep the registered command as the temporal read profiling entrypoint. |
| `devtools workspace temporal-devloop` | `retain` | Focused tests cover structured and Markdown event sources; output is a reusable evidence window. | Keep the registered command as the devloop temporal evidence entrypoint. |
| `devtools workspace temporal-archive-aggregates` | `retain` | Focused tests cover aggregate report construction and reusable archive artifacts. | Keep the registered command as the run-projection aggregate entrypoint. |
| `devtools workspace lineage-validation` | `retain` | Focused tests cover lineage evidence and the command emits reusable count and composition proof. | Keep the registered command before publishing archive cardinality claims. |
| `devtools workspace cli-surface-audit` | `retain` | Focused tests cover bounded output and stale-artifact pruning; the audit shelf is reusable. | Keep the registered command as the current CLI surface audit entrypoint. |
| `devtools demo real-slice-screen` | `retain` | Focused privacy-screening tests cover redaction, PII review, and report generation. | Keep the registered command as the read-only real-archive screening entrypoint. |

Catalog bypass audit sites are machine-checked across workflow runs, CI-owned npm scripts, hooks, and devtools process launches. Direct hook adapters require declared sanctioned exceptions with an exact occurrence and cardinality.

| Site | Status | Registered command | Occurrence | Reason |
| --- | --- | --- | --- | --- |
| `.github/workflows/mutation-testing.yml` | `registered` | `devtools verify mutation-freshness` | not applicable | CI invokes the catalog command so inventory and workflow validation see the freshness gate. |
| `.github/workflows/nightly-scale.yml` | `registered` | `devtools bench nightly-compare` | not applicable | CI invokes the catalog command so the nightly comparison is discoverable and checked. |
| `devtools/pre_push_gate.py` | `registered` | `devtools lab policy backlog-hygiene` | not applicable | The intentional Beads-only route remains narrow while using the registered policy command. |
| `docs/test-economics.md` | `sanctioned-bypass` | `devtools lab test-economics` | not applicable | The generated provenance header preserves the module that emitted the document; operators use the catalog command. |
| `.githooks/pre-push` | `sanctioned-bypass` | `hook adapter` | line 21 (1 expected) | The hook adapter must receive Git's staged stdin update stream before dispatching its catalog-aware gate. |
| `.beads-hooks/pre-push` | `sanctioned-bypass` | `hook adapter` | line 21 (1 expected) | The Beads-augmented hook retains the same stdin adapter before its managed Beads section runs. |

### Core

| Command | Description |
| --- | --- |
| `devtools status` | Render the devshell status view. |

### Generated Surfaces

| Command | Description |
| --- | --- |
| `devtools render agent-manual` | Render the declaration-generated six-tool agent manual and packaged integration assets. |
| `devtools render all` | Refresh or verify generated docs and agent files. |
| `devtools render api-operation-parity` | Render the committed semantic-operation parity matrix and Python facade reference. |
| `devtools render cli-output-schemas` | Render JSON Schema artifacts for stable CLI output payloads under docs/schemas/cli-output/. |
| `devtools render cli-reference` | Render docs/cli-reference.md from live CLI help. |
| `devtools render demo-corpus-datasheet` | Render docs/plans/demo-corpus-construct-audit.md from the demo family registry and a measured seed archive. |
| `devtools render devtools-reference` | Render the command catalog inside docs/devtools.md. |
| `devtools render docs-surface` | Render docs/README.md and the README documentation table. |
| `devtools render mcp-equivalence` | Render docs/generated/mcp-equivalence.json from executable MCP declarations. |
| `devtools render mcp-tool-index` | Render the generated exhaustive tool-name appendix into docs/mcp-reference.md. |
| `devtools render openapi` | Render docs/openapi/search.yaml from typed daemon query payload models. |
| `devtools render pages` | Build the GitHub Pages documentation site into .cache/site/. |
| `devtools render product-workflows` | Render docs/product/workflows.md from executable query-action workflow registries. |
| `devtools render public-claims` | Render README, launch, findings-page, and verified-export claim views from FINDING assertions. |
| `devtools render quality-reference` | Render docs/test-quality-workflows.md from executable lane, mutation, and benchmark registries. |
| `devtools render query-discovery` | Render parser-gated query discovery examples and result semantics into docs/search.md. |
| `devtools render visual-tapes` | Write VHS tape files and optionally capture GIFs for the default visual evidence specs. |
| `devtools render webui-client` | Render the committed WebUI TypeScript client from docs/openapi/search.yaml. |
| `devtools render webui-design-system` | Render WebUI v2 CSS tokens, public badge contracts, and contrast evidence. |

### Release

| Command | Description |
| --- | --- |
| `devtools release build-package` | Build the default Nix package with the out-link under .local/result. |
| `devtools release readiness` | Validate the externally-presentable release gate definition. |
| `devtools release verify-distribution` | Verify wheel/sdist installed artifacts expose only supported runtime entrypoints. |

### Lab Checks

| Command | Description |
| --- | --- |
| `devtools lab graph` | Render the runtime artifact, operation, and scenario-coverage map. |
| `devtools lab lanes` | Run named validation lanes. |
| `devtools lab policy backlog-hygiene` | Verify Beads backlog structure invariants (.beads/issues.jsonl). |
| `devtools lab policy bead-graph` | Bead-graph invariant lint over live `bd` state (cycles, wave labels/inversions, missing AC). |
| `devtools lab policy campaign-archive-boundaries` | Verify devtools synthetic benchmark/scale campaigns route through ArchiveLocation. |
| `devtools lab policy classifier-fingerprints` | Verify parser/classifier decision-boundary changes are declared as reparse-requiring or acknowledged. |
| `devtools lab policy demo-packet-registry` | Verify every registered 212 demo has a conforming Demo Finding Packet. |
| `devtools lab policy demo-tour-freshness` | Verify a freshly-run demo tour matches the committed docs/examples/demo-tour/ evidence artifacts. |
| `devtools lab policy docs-drift` | Verify checkable factual claims in the Reference-docs table against current source. |
| `devtools lab policy insight-honesty` | Verify every registered insight product is rigor-contracted or exempt. |
| `devtools lab policy position-derived-identity` | Verify no parser mints cross-revision comparison identity from positional/index data. |
| `devtools lab policy raw-authority-frontier-executability` | Verify every raw-authority frontier state has a reachable actuator. |
| `devtools lab policy raw-payload-hash-purity` | Verify no raw-capture write path splices a synthesized literal onto captured bytes before hashing. |
| `devtools lab policy schema-versioning` | Verify durable-tier migration and derived-tier rebuild boundaries. |
| `devtools lab policy table-exists-duplication` | Verify no module outside storage/introspection.py redefines table_exists/column_exists/index_exists. |
| `devtools lab policy timestamp-doctrine` | Verify durable-tier DDL never stores a timestamp column as TEXT. |
| `devtools lab probe bead-pr-reconciliation` | Surface beads whose referenced PR merged but the bead is still open. |
| `devtools lab probe capture-regression` | Capture pipeline-probe summaries as durable local regression cases. |
| `devtools lab probe cost-reconciliation` | Reconcile Polylogue token accounting against private provider stores. |
| `devtools lab probe pipeline` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools lab probe turso` | Probe Turso Database compatibility against Polylogue storage assumptions. |
| `devtools lab projections` | Render the authored scenario-bearing verification projections. |
| `devtools lab provider completeness` | Report provider/importer package completeness by origin and capture mode. |
| `devtools lab pytest-witness-repetitions` | Repeat the exact optimize, WAL, and embedding seed-hang witnesses with durable receipts. |
| `devtools lab schema audit` | Run committed provider schema package quality checks. |
| `devtools lab schema commit` | Persist a real full-corpus schema generation into committed provider packages. |
| `devtools lab schema compare` | Compare two committed schema package versions for a provider. |
| `devtools lab schema explain` | Explain a committed package element schema with evidence and annotations. |
| `devtools lab schema generate` | Generate provider schema packages and optional evidence clusters. |
| `devtools lab schema list` | List committed schema packages, versions, and evidence manifests. |
| `devtools lab schema parser-diff` | List observed provider wire keys that no parser references. |
| `devtools lab schema promote` | Promote a schema evidence cluster into a registered package version. |
| `devtools lab schema roundtrip` | Verify committed provider schema packages reload and roundtrip cleanly. |
| `devtools lab seed-receipt-compare` | Compare two workload receipts for a clean, like-for-like seed/incident proof (polylogue-b054.1.1.3). |
| `devtools lab smoke` | Run direct archive and reader smoke sets. |
| `devtools lab snapshot read-surface` | Capture and compare archive read-surface snapshots. |
| `devtools lab test-economics` | Report per-package coverage/fix-density/test-cost economics (polylogue-9e5.11). |
| `devtools lab testmon-blind-spots` | Audit coverage-known files that are absent from the testmon fingerprint graph. |
| `devtools lab testmon-proof` | Prove real testmon affected selection against a semantic production mutation. |

### Verification

| Command | Description |
| --- | --- |
| `devtools reindex-canary` | Run the product's representative inactive-generation reindex canary. |
| `devtools test` | Run a focused pytest selection through the managed harness. |
| `devtools verify` | Run the local verification baseline before pushing or creating a PR. |
| `devtools verify agent-integration` | Verify manual compilation, parser examples, continuation, native delivery, packaging, and live cutover signatures. |
| `devtools verify catalog-bypasses` | Reject direct devtools module or script execution outside sanctioned adapters. |
| `devtools verify ci-workflows` | Verify CI workflow files reference locally-known devtools commands and existing paths. |
| `devtools verify closure-matrix` | Verify docs/plans/test-closure-matrix.yaml stays grounded in the realized tree. |
| `devtools verify corpus-fidelity` | Run the production corpus-fidelity acceptance gate against an archive root. |
| `devtools verify coverage` | Run pytest with the repository coverage floor from pyproject.toml. |
| `devtools verify degrade-loudly` | Verify broad except-handlers in daemon/storage/insights/coordination log or signal on failure. |
| `devtools verify doc-commands` | Verify README/docs command examples resolve to live polylogue, polylogued, and devtools commands. |
| `devtools verify docs-coverage` | Verify every public CLI command, MCP tool, config key, and stable daemon route is named in the docs tree. |
| `devtools verify evidence` | Render the pytest-first evidence dashboard. |
| `devtools verify layering` | Check inter-package imports against declared layering rules from docs/plans/layering.yaml. |
| `devtools verify manifests` | Verify internal consistency across all docs/plans/*.yaml manifest files. |
| `devtools verify mutation-freshness` | Verify fresh mutation campaigns meet their declared kill-rate thresholds. |
| `devtools verify public-claims` | Verify generated public-claim views, preset parity, sanitized refs, coverage markers, and retired copy. |
| `devtools verify pytest-timeout-overrides` | Verify explicit pytest timeout overrides are positive, bounded, and justified. |
| `devtools verify schema-inference-gate` | Run the read-only schema-inference prerequisite and persist a PASS/FAIL receipt. |
| `devtools verify test-infra-currency` | Verify tests/infra/ helpers reference only tables that exist in the current SCHEMA_VERSION. |

### Benchmarking

| Command | Description |
| --- | --- |
| `devtools bench campaign` | Run or compare benchmark campaigns. |
| `devtools bench coordination-latency` | Measure compact coordination status p50/p95 with raw stage samples. |
| `devtools bench help-latency` | Check `--help` wall-clock latency against the interactive-tier cold-CLI budget (polylogue-20d.2). |
| `devtools bench ingest-amplification` | Measure deterministic per-tier ingest write amplification on a synthetic fixture (#1851). |
| `devtools bench ingest-throughput` | Measure ingest wall-clock throughput on a synthetic fixture. |
| `devtools bench memory` | Measure query-memory envelopes on generated fixtures. |
| `devtools bench mutation` | Run focused mutation campaigns and maintain their local index. |
| `devtools bench nightly-compare` | Compare nightly pytest-benchmark output with the committed baseline. |
| `devtools bench slo` | Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements. |
| `devtools bench synthetic` | Run synthetic benchmark campaigns over generated archives. |

### Workspace

| Command | Description |
| --- | --- |
| `devtools demo real-slice-screen` | Read-only extraction + privacy screening of a candidate real-archive session slice. |
| `devtools workspace affordance-usage` | Analyze agent affordance/tool usage from archive tool-use rows. |
| `devtools workspace agent-meta-sidecar-purge-apply` | Purge agent-*.meta.json subagent-sidecar phantom sessions from index.db. |
| `devtools workspace agent-meta-sidecar-sweep` | Find agent-*.meta.json subagent-sidecar phantom sessions (message_count=0). |
| `devtools workspace antigravity-phantom-purge-apply` | Delete antigravity brain-metadata phantom sessions and reclassify their raw rows. |
| `devtools workspace antigravity-phantom-sweep` | List antigravity-session rows that are brain-metadata phantom fragments. |
| `devtools workspace attachment-reacquisition` | Classify historically-unfetched attachments for a source-backed backfill. |
| `devtools workspace attachment-reacquisition-apply` | Backfill acquisition for historically-unfetched attachments. |
| `devtools workspace backlog-calibration` | Measured lead-time/discovery/staleness distributions over the bead corpus. |
| `devtools workspace bead-batch-show` | Batch-show beads: id, status, prio, title, desc head, deps, notes tail. |
| `devtools workspace bead-cluster` | Footprint/overlap/contention clustering of ready Beads (execution frontier). |
| `devtools workspace bead-reimport-guard` | Monotonic, receipted guard/reconcile/export for bd's JSONL synchronization. |
| `devtools workspace beads-state-report` | Self-contained HTML state-of-the-backlog report over the whole bead population. |
| `devtools workspace binary-artifact-reclassify-apply` | Persist raw_artifacts classification for binary-shaped raw rows. |
| `devtools workspace binary-artifact-sweep` | Find raw_sessions rows whose bytes are a non-session binary format (SQLite, etc). |
| `devtools workspace chatgpt-lifecycle-anchor-audit` | Census the current quarantined ChatGPT corpus for lifecycle-anchor conflicts. |
| `devtools workspace claim-vs-evidence` | Build a structured failure follow-up claim-vs-evidence demo. |
| `devtools workspace cli-surface-audit` | Capture a current-curated CLI surface audit demo. |
| `devtools workspace degraded-archive-proof` | Build a degraded archive self-healing proof artifact. |
| `devtools workspace delivery-gate-status` | Per-release-gate progress board over .beads/issues.jsonl (delivery:<release> / lane:<lane> overlay). |
| `devtools workspace demo-shelf` | Refresh or verify current demo shelf indexes. |
| `devtools workspace deployment-smoke` | Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow. |
| `devtools workspace dev-loop` | Preflight branch-local daemon, web-shell, and browser-capture development loops. |
| `devtools workspace failure-context` | Join testmon, git history, and fixtures for a pytest failure ID into a JSON envelope. |
| `devtools workspace frontier` | Classify ready and in-progress Beads into devloop batches. |
| `devtools workspace index-fast-forward` | Plan and prove a declared index fast-forward against retained raw replay. |
| `devtools workspace lane-brief` | Generate a dispatch brief for a bead lane with live footprint/prior-art evidence. |
| `devtools workspace lane-init` | Provision a fanout lane worktree: branch, isolated venv, guard check, ledger record. |
| `devtools workspace lineage-validation` | Validate lineage-count evidence before citing archive counts externally. |
| `devtools workspace mandate-continuity-replay` | Wire t8t continuity scenarios + work-evidence effects + discovery into one mandate artifact. |
| `devtools workspace merge` | Merge boundary wrapper: refuses `gh pr merge` without a fresh merge-gate receipt. |
| `devtools workspace merge-conductor` | Mechanical-conflict triage for the PR merge train (dry-run by default). |
| `devtools workspace merge-gate` | Structural pre-merge safety check: fresh local-verification receipt + no late review comments. |
| `devtools workspace raw-append-chain-backfill-apply` | Promote membershipless append raws proven correct by live-source verification. |
| `devtools workspace raw-authority-daemon-health-proof` | Prove daemon status/health HTTP responsiveness during a real raw-authority drain. |
| `devtools workspace raw-authority-restart-proof` | Prove raw-authority crash recovery and conserved fixed-point convergence. |
| `devtools workspace raw-authority-scale-proof` | Run bounded raw-authority replay to a two-census fixed point. |
| `devtools workspace raw-byte-duplicate-supersession-apply` | Promote quarantined, logical-key-less raws proven byte-identical to an already-indexed raw. |
| `devtools workspace raw-failure-disposition-apply` | Apply reviewed terminal dispositions to historical raw parse failures. |
| `devtools workspace raw-live-source-reconciliation` | Classify quarantined raw evidence against its live source file's current bytes. |
| `devtools workspace raw-live-source-reconciliation-apply` | Promote quarantined raw evidence proven correct by live-source verification. |
| `devtools workspace raw-membership-writeback-apply` | Propagate already-decided membership verdicts onto raw_sessions.revision_authority. |
| `devtools workspace raw-quarantine-group-dedup-apply` | Promote one representative raw per fully-quarantined byte-identical (source_path, blob_hash) group. |
| `devtools workspace read-package` | Render a declarative package of Polylogue read artifacts. |
| `devtools workspace scale-regression` | Run the seeded large-archive scale-regression probe. |
| `devtools workspace tasks` | Record and query local agent task execution history. |
| `devtools workspace temporal-archive-aggregates` | Build run-projection aggregate artifacts from the active archive. |
| `devtools workspace temporal-devloop` | Compose git and operating-log events into a temporal evidence window. |
| `devtools workspace temporal-read-profile` | Measure read --view temporal phase timings on the active archive. |
| `devtools workspace tool-result-history-reclassify-apply` | Persist raw_artifacts classification for tool-result/file-history-shaped raw rows. |
| `devtools workspace tool-result-history-sweep` | Find claude-code-session raw rows that should reclassify as tool-result/file-history sidecars. |
| `devtools workspace unknown-export-reclassification` | Re-run the fixed browser-capture provider probe against stored unknown-export rows. |
| `devtools workspace unknown-export-reclassification-apply` | Reclassify proven ChatGPT browser-capture raws and write durable receipts. |
| `devtools workspace verify-worktree` | Verify an agent lane's claimed worktree exists, is isolated, and is on the expected branch. |
| `devtools workspace worktree-gc` | Safe worktree garbage collection — list and remove merged, squash-equivalent, or abandoned git worktrees. |

<!-- END GENERATED: devtools-command-catalog -->

## Validation and Evidence

When changing semantics, validation, or surfaces:

```bash
devtools lab lanes --list
devtools lab lanes --lane frontier-local
devtools lab smoke run archive-smoke --tier 0
devtools lab smoke run reader-visual-smoke
devtools bench memory --max-rss-mb 1536 -- polylogue --plain analyze
```

Campaign outputs live under `.local/`, not in tracked docs trees.

## Local State Layout

- `.cache/`: disposable cache state.
- `.local/`: untracked local outputs such as campaigns, demo artifacts, and reports.
- `.venv/` and `.direnv/`: kept at the repo root because their tooling expects those locations.
- `.local/result`: preferred repo-local out-link for `devtools release build-package`; a top-level `result` symlink is just Nix's default ad-hoc out-link.

Keep new repo-local outputs in `.cache/` or `.local/` instead of adding new
top-level output roots.
