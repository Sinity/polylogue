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

## Beads graph checks

Use `bd ready` to inspect executable work and `workspace bead-cluster` before
parallel dispatch. Validate branch-local dependency records without importing
an aging worktree into the shared Beads database:

```bash
devtools lab policy bead-graph --export .beads/issues.jsonl --json
```

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
| `devtools lab provider completeness` | Inspect detector, parser, fixture, schema, docs, ImportExplain, and caveat coverage before claiming a provider/importer mode is product-ready. |
| `devtools lab graph` | Inspect declared runtime artifacts, operations, paths, and maintenance targets. |
| `devtools lab testmon-proof` | Validate the affected-test harness itself: a disposable copy of a real Polylogue module and existing route test is seeded, semantically mutated, edge-severed, restored, and checked for bounded unrelated-change selection. |
| `devtools lab snapshot read-surface` | Freeze archive read-surface behavior before archive work, then compare candidate archives against the captured envelope baseline. |
| `devtools lab policy schema-versioning` | Enforce the policy boundary documented in docs/internals.md § 'Schema Versioning Model'. Durable tiers use explicit additive migrations with a backup gate; derived tiers are rebuilt or blue-green replaced from source evidence. |
| `devtools lab policy classifier-fingerprints` | Catch the gap `lab policy schema-versioning` cannot see (polylogue-gucv): a parser/classifier under polylogue/sources/ or polylogue/archive/artifact_taxonomy/ (looks_like*/classify_artifact* functions) changes what it accepts for identical input bytes without any INDEX_SCHEMA_VERSION bump at all, so already-indexed rows go silently stale with no signal a reparse was needed (PR #3428 shipped exactly this, green, against the version-keyed gate). |
| `devtools lab policy bead-graph` | Run before shipping a bead-state delta. With no source option it checks live `bd` state; `--export .beads/issues.jsonl` validates the branch snapshot without importing it into the shared database. The gate reads dependency records only and does not make prose, labels, or campaign-specific edge lists machine authority. |
| `devtools lab policy timestamp-doctrine` | Enforce the time doctrine (UTC epoch-ms canon, docs/internals.md) at DDL-review time (cpf.1): a TEXT timestamp in source.db/user.db re-introduces tz-unknown ambiguity and lexicographic-vs-temporal sort divergence, and durable tiers need an explicit additive migration to fix later -- catching it before merge is orders cheaper than a copy-forward migration after. |
| `devtools lab policy insight-honesty` | Enforce that polylogue.insights.registry.INSIGHT_REGISTRY and polylogue.insights.rigor's contract matrix/exemption list never drift apart (9e5.28) -- a registered product with neither a RigorContract nor a RIGOR_EXEMPT entry used to silently vanish from `polylogue ops insights audit` instead of showing as uncovered. |
| `devtools lab probe cost-reconciliation` | Validate archive token accounting against optional local Codex state_5.sqlite and Claude stats-cache.json before publishing cost or usage-analysis claims. |
| `devtools lab probe pipeline` | Run real pipeline stages and optionally capture emitted summaries as regression cases. |
| `devtools lab probe turso` | Collect executable evidence before changing production storage backends: Python binding availability, generated-column support, FTS compatibility, MVCC, CDC, vector functions, ATTACH, and WAL pragma behavior. |
| `devtools lab run` | Run a scenario such as rebuild-safety through the direct lab command path. |
| `devtools lab smoke` | Run direct archive and reader smoke sets outside the archive CLI. |
| `devtools lab schema list` | Inspect committed provider schema package catalogs without presenting them as normal archive usage. |
| `devtools lab schema compare` | Review schema package drift between committed versions in the lab surface. |
| `devtools lab schema explain` | Inspect schema package annotations, semantic roles, and review evidence from the lab surface. |
| `devtools lab schema generate` | Refresh provider schema package artifacts from archive observations outside the archive CLI. |
| `devtools lab schema commit` | Actually regenerate and write `polylogue/schemas/providers/<provider>/versions/...` from the live archive -- 'lab schema generate' only ever previews and never writes committed package files. |
| `devtools lab schema promote` | Turn reviewed schema evidence clusters into committed provider schema packages. |
| `devtools lab schema audit` | Check committed schema package quality gates without presenting them as normal archive usage. |
| `devtools lab schema parser-diff` | Scope a parser batch by evidence before a rebuild: ranks every schema key nothing reads by how many records actually carry it. Output is a triage queue, not a verdict -- parser-side matching is name-based, so read the parser before acting on a row. |
| `devtools lab schema roundtrip` | Close the schema inference-validation loop: package manifests must roundtrip through typed models, and every supported element schema must be reachable from the runtime registry. |
| `devtools lab probe capture-regression` | Turn a live or probe failure JSON summary into a replayable local regression artifact. |

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

### Core

| Command | Description |
| --- | --- |
| `devtools status` | Render the devshell status view. |

### Generated Surfaces

| Command | Description |
| --- | --- |
| `devtools render agent-manual` | Render the declaration-generated six-tool agent manual and packaged integration assets. |
| `devtools render all` | Refresh or verify generated docs and agent files. |
| `devtools render cli-output-schemas` | Render JSON Schema artifacts for stable CLI output payloads under docs/schemas/cli-output/. |
| `devtools render cli-reference` | Render docs/cli-reference.md from live CLI help. |
| `devtools render devtools-reference` | Render the command catalog inside docs/devtools.md. |
| `devtools render docs-surface` | Render docs/README.md and the README documentation table. |
| `devtools render openapi` | Render docs/openapi/search.yaml from typed daemon query payload models. |
| `devtools render pages` | Build the GitHub Pages documentation site into .cache/site/. |
| `devtools render query-discovery` | Render parser-gated query discovery examples and result semantics into docs/search.md. |
| `devtools render visual-tapes` | Write VHS tape files and optionally capture GIFs for the default visual evidence specs. |
| `devtools render webui-client` | Render the committed WebUI TypeScript client from docs/openapi/search.yaml. |
| `devtools render webui-design-system` | Render WebUI v2 CSS tokens, public badge contracts, and contrast evidence. |

### Release

| Command | Description |
| --- | --- |
| `devtools release build-package` | Build the default Nix package with the out-link under .local/result. |
| `devtools release verify-distribution` | Verify wheel/sdist installed artifacts expose only supported runtime entrypoints. |

### Lab Checks

| Command | Description |
| --- | --- |
| `devtools lab graph` | Render the runtime artifact and operation graph. |
| `devtools lab policy bead-graph` | Validate typed dependency endpoints, uniqueness, parent cardinality, and cycles in the Beads graph. |
| `devtools lab policy classifier-fingerprints` | Verify parser/classifier decision-boundary changes are declared as reparse-requiring or acknowledged. |
| `devtools lab policy insight-honesty` | Verify every registered insight product is rigor-contracted or exempt. |
| `devtools lab policy schema-versioning` | Verify durable-tier migration and derived-tier rebuild boundaries. |
| `devtools lab policy timestamp-doctrine` | Verify durable-tier DDL never stores a timestamp column as TEXT. |
| `devtools lab probe capture-regression` | Capture pipeline-probe summaries as durable local regression cases. |
| `devtools lab probe cost-reconciliation` | Reconcile Polylogue token accounting against private provider stores. |
| `devtools lab probe pipeline` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools lab probe turso` | Probe Turso Database compatibility against Polylogue storage assumptions. |
| `devtools lab provider completeness` | Report provider/importer package completeness by origin and capture mode. |
| `devtools lab run` | Run a named archive verification scenario. |
| `devtools lab schema audit` | Run committed provider schema package quality checks. |
| `devtools lab schema commit` | Persist a real full-corpus schema generation into committed provider packages. |
| `devtools lab schema compare` | Compare two committed schema package versions for a provider. |
| `devtools lab schema explain` | Explain a committed package element schema with evidence and annotations. |
| `devtools lab schema generate` | Generate provider schema packages and optional evidence clusters. |
| `devtools lab schema list` | List committed schema packages, versions, and evidence manifests. |
| `devtools lab schema parser-diff` | List observed provider wire keys that no parser references. |
| `devtools lab schema promote` | Promote a schema evidence cluster into a registered package version. |
| `devtools lab schema roundtrip` | Verify committed provider schema packages reload and roundtrip cleanly. |
| `devtools lab smoke` | Run direct archive and reader smoke sets. |
| `devtools lab snapshot read-surface` | Capture and compare archive read-surface snapshots. |
| `devtools lab testmon-proof` | Prove real testmon affected selection against a semantic production mutation. |

### Verification

| Command | Description |
| --- | --- |
| `devtools reindex-canary` | Run the product's representative inactive-generation reindex canary. |
| `devtools test` | Run a focused pytest selection through the managed harness. |
| `devtools verify` | Run the local verification baseline before pushing or creating a PR. |
| `devtools verify agent-integration` | Verify manual compilation, parser examples, continuation, native delivery, packaging, and live cutover signatures. |
| `devtools verify corpus-fidelity` | Run the production corpus-fidelity acceptance gate against an archive root. |
| `devtools verify coverage` | Run pytest with the repository coverage floor from pyproject.toml. |
| `devtools verify layering` | Check inter-package imports against declared layering rules from docs/plans/layering.yaml. |
| `devtools verify mutation-freshness` | Verify executable mutation campaigns meet the selected freshness and kill-rate thresholds. |
| `devtools verify schema-inference-gate` | Run the read-only schema-inference prerequisite and persist a PASS/FAIL receipt. |

### Benchmarking

| Command | Description |
| --- | --- |
| `devtools bench help-latency` | Check `--help` wall-clock latency against the interactive-tier cold-CLI budget (polylogue-20d.2). |
| `devtools bench ingest-amplification` | Measure deterministic per-tier ingest write amplification on a synthetic fixture (#1851). |
| `devtools bench ingest-throughput` | Measure ingest wall-clock throughput on a synthetic fixture. |
| `devtools bench memory` | Measure query-memory envelopes on generated fixtures. |
| `devtools bench mutation` | Run focused mutation campaigns with isolated execution and JSON artifacts. |
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
| `devtools workspace binary-artifact-reclassify-apply` | Persist raw_artifacts classification for binary-shaped raw rows. |
| `devtools workspace binary-artifact-sweep` | Find raw_sessions rows whose bytes are a non-session binary format (SQLite, etc). |
| `devtools workspace degraded-archive-proof` | Build a degraded archive self-healing proof artifact. |
| `devtools workspace deployment-smoke` | Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow. |
| `devtools workspace dev-loop` | Preflight branch-local daemon, web-shell, and browser-capture development loops. |
| `devtools workspace failure-context` | Join testmon, git history, and fixtures for a pytest failure ID into a JSON envelope. |
| `devtools workspace index-fast-forward` | Plan and prove a declared index fast-forward against retained raw replay. |
| `devtools workspace lane-brief` | Generate a dispatch brief for a bead lane with live footprint/prior-art evidence. |
| `devtools workspace lane-init` | Provision a fanout lane worktree: branch, isolated venv, guard check, ledger record. |
| `devtools workspace lineage-validation` | Validate lineage-count evidence before citing archive counts externally. |
| `devtools workspace mandate-continuity-replay` | Replay continuity scenarios and repository effects through production routes. |
| `devtools workspace merge` | Merge boundary wrapper: refuses `gh pr merge` without a fresh merge-gate receipt. |
| `devtools workspace merge-gate` | Structural pre-merge safety check: fresh local verification + resolved review threads. |
| `devtools workspace pr-scope` | Render stable PR scope intent and inspect its mutable merge attestation. |
| `devtools workspace raw-append-chain-backfill-apply` | Promote membershipless append raws proven correct by live-source verification. |
| `devtools workspace raw-authority-artifact-census` | Census quarantined raws into five authority buckets; apply pages raw_artifacts upserts and records durable receipts. |
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
| `devtools workspace temporal-archive-aggregates` | Build run-projection aggregate artifacts from the active archive. |
| `devtools workspace temporal-read-profile` | Measure read --view temporal phase timings on the active archive. |
| `devtools workspace tool-result-history-reclassify-apply` | Persist raw_artifacts classification for tool-result/file-history-shaped raw rows. |
| `devtools workspace tool-result-history-sweep` | Find claude-code-session raw rows that should reclassify as tool-result/file-history sidecars. |
| `devtools workspace unknown-export-reclassification` | Re-run the fixed browser-capture provider probe against stored unknown-export rows. |
| `devtools workspace unknown-export-reclassification-apply` | Reclassify proven ChatGPT browser-capture raws and write durable receipts. |
| `devtools workspace verify-worktree` | Verify an agent lane's claimed worktree exists, is isolated, and is on the expected branch. |
| `devtools workspace worktree-gc` | Safe worktree garbage collection — list and remove merged, squash-equivalent, or abandoned git worktrees. |

<!-- END GENERATED: devtools-command-catalog -->

## Cursor-authority reconciliation

`polylogue ops maintenance cursor-authority-reconcile` is a dry-run-by-default
repair route for exactly one proven cursor-ahead source. It reads the fixed
`/realm/db/polylogue` archive root, requires the daemon to be stopped, and
writes a plan containing path and raw identifiers only as digests. Apply
requires that immutable plan, a freshly verified `full_evidence` backup
manifest with blob rollback evidence, and a new receipt path. The apply route
uses the normal live full-ingest/replay path under one single-use exact path
and frontier authorization. Receipts distinguish a performed ingest from an
observed recovery, leave cursor row counts null when the before/after state did
not prove them, and record typed deferred or failed post-ingest evidence. It
never accepts a global cursor bypass or writes `ingest_cursor` or accepted-head
rows directly.

The dry-run form is:

```text
polylogue ops maintenance cursor-authority-reconcile \
  --source-path-file /private/path-file \
  --output-plan /private/reconciliation-plan.json
```

The apply form is:

```text
polylogue ops maintenance cursor-authority-reconcile --apply \
  --plan /private/reconciliation-plan.json \
  --backup-manifest /private/full-evidence-backup \
  --receipt /private/reconciliation-receipt.json
```

## Validation and Evidence

When changing semantics, validation, or surfaces:

```bash
devtools verify
devtools test tests/unit/path/to/test_file.py
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
