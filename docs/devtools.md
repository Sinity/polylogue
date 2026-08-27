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
- keep evidence/scenario behavior in verification modules with executable command entrypoints;
- prefer validation lanes and the ordinary verifier to compose executable
  checks rather than duplicating domain checks inside `devtools verify`.

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

## Core Loop

These are the commands worth remembering during normal repo work:

- `devtools status`: Check repo state, generated-surface drift, and the next default verification steps.
  Common forms: `devtools status`, `devtools status --json`, `devtools status --verify-generated`.
- `devtools why`: A verify failed, bootstrapped unexpectedly, or refused to run, and you want the cause without reading receipt JSON by hand.
  Common forms: `devtools why`, `devtools why --history 24`, `devtools why --run 20260817T213631Z-testmon-2709409-d5c6e72c`.
- `devtools render all`: Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.
  Common forms: `devtools render all`, `devtools render all --check`.
- `devtools verify`: Run format, lint, mypy, render all, committed-schema privacy, and test checks locally before pushing.
  Common forms: `devtools verify`, `devtools verify --quick`.
- `devtools test`: Run a specific test file, directory, or -k/-m selection in the inner loop, or inspect the latest full-run timing receipts, without invoking raw pytest.
  Common forms: `devtools test tests/unit/pipeline`, `devtools test -k hybrid`, `devtools test tests/unit/storage -x`, `devtools test --outliers 20`.
- `devtools bench mutation`: Run or inspect focused mutation-testing work without shrinking the committed mutmut scope.
  Common forms: `devtools bench mutation list`, `devtools bench mutation run filters`.

### Core

| Command | Description |
| --- | --- |
| `devtools status` | Render the devshell status view. |
| `devtools why` | Explain the most recent verification run, or where verification time went. |

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

### Verification

| Command | Description |
| --- | --- |
| `devtools bench pipeline` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools test` | Run focused pytest selections or inspect full-run timing outliers. |
| `devtools verify` | Run the local verification baseline before pushing or creating a PR, including the required committed-schema privacy registry check. |
| `devtools verify agent-integration` | Verify manual compilation, parser examples, continuation, native delivery, packaging, and live cutover signatures. |
| `devtools verify atlas` | Check atlas citation anchors and verification-commit freshness. |
| `devtools verify ci-commands` | Validate devtools invocations in structured CI run fields. |
| `devtools verify consumer-reachability` | Require newly added modules, tables, and tools to have production consumers. |
| `devtools verify corpus-fidelity` | Run the production corpus-fidelity acceptance gate against an archive root. |
| `devtools verify coverage` | Run pytest with the repository coverage floor from pyproject.toml. |
| `devtools verify definition-closure` | Evaluate representative definition-to-production closure policies as a bounded JSON matrix. |
| `devtools verify doc-commands` | Validate executable documentation examples against live command inventories. |
| `devtools verify insight-honesty` | Verify every registered insight product is rigor-contracted or exempt. |
| `devtools verify layering` | Check inter-package imports against declared layering rules from docs/plans/layering.yaml. |
| `devtools verify mutation-freshness` | Verify executable mutation campaigns meet the selected freshness and kill-rate thresholds. |
| `devtools verify oracle-integrity` | Verify tests certify production-reachable code and never read ambient user paths. |
| `devtools verify patterns` | Enforce AST-shape defect-family rules with shrinking grandfathered baselines. |
| `devtools verify portfolio-frontier` | Validate complete Beads ambition, active-set, and execution-focus views. |
| `devtools verify provider-completeness` | Report provider/importer package completeness by origin and capture mode. |
| `devtools verify read-surface` | Capture and compare archive read-surface snapshots. |
| `devtools verify reindex-packets` | Validate the current reindex execution packets from the external Beads blocks graph. |
| `devtools verify runtime` | Verify the CPython 3.14 free-threaded runtime and required native extensions. |
| `devtools verify runtime-census` | Census production concurrency boundaries and classify every discovered item. |
| `devtools verify scenario` | Run a named archive verification scenario. |
| `devtools verify schema-audit` | Run committed provider schema package quality checks. |
| `devtools verify schema-inference-gate` | Run the read-only schema-inference prerequisite and persist a PASS/FAIL receipt. |
| `devtools verify schema-roundtrip` | Verify committed provider schema packages reload and roundtrip cleanly. |
| `devtools verify schema-versioning` | Verify durable-tier migration and derived-tier rebuild boundaries. |
| `devtools verify semantic-fidelity` | Run the bounded production-route semantic contradiction and construct-flow census. |
| `devtools verify timestamp-doctrine` | Verify durable-tier DDL never stores a timestamp column as TEXT. |
| `devtools verify webui` | Run the declared typed WebUI generation, contract, unit, and build checks. |
| `devtools workspace schema commit` | Persist a real full-corpus schema generation into committed provider packages. |
| `devtools workspace schema compare` | Compare two committed schema package versions for a provider. |
| `devtools workspace schema explain` | Explain a committed package element schema with evidence and annotations. |
| `devtools workspace schema generate` | Generate provider schema packages and optional evidence clusters. |
| `devtools workspace schema list` | List committed schema packages, versions, and evidence manifests. |
| `devtools workspace schema parser-diff` | List observed provider wire keys that no parser references. |
| `devtools workspace schema promote` | Promote a schema evidence cluster into a registered package version. |

### Benchmarking

| Command | Description |
| --- | --- |
| `devtools bench cli-interaction` | Run the complete installed CLI and direct typed-UDS interaction profile. |
| `devtools bench concurrency` | Run the managed bounded-compute scaling profile across representative workloads. |
| `devtools bench daemon-operation` | Run the installed CLI and direct typed-UDS daemon operation profile. |
| `devtools bench ingest-amplification` | Measure deterministic per-tier ingest write amplification on a synthetic fixture (#1851). |
| `devtools bench ingest-throughput` | Measure ingest wall-clock throughput on a synthetic fixture. |
| `devtools bench memory` | Measure query-memory envelopes on generated fixtures. |
| `devtools bench mutation` | Run focused mutation campaigns with isolated execution and JSON artifacts. |
| `devtools bench slo` | Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements. |

### Workspace

| Command | Description |
| --- | --- |
| `devtools workspace continuity-evidence` | Replay continuity scenarios and verify their query routes are discoverable. |
| `devtools workspace deployment-smoke` | Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow. |
| `devtools workspace index-fast-forward` | Plan and prove a declared index fast-forward against retained raw replay. |
| `devtools workspace lineage-validation` | Validate lineage-count evidence before citing archive counts externally. |
| `devtools workspace physical-identity-census` | Census raw evidence hidden by origin/native session identity collapse. |
| `devtools workspace seeded-archive-cache-gc` | Preview or apply age-gated GC for the shared seeded-archive fixture cache. |

<!-- END GENERATED: devtools-command-catalog -->

## Pattern Ratchet

Pattern baselines use `path:sha1` content anchors, where the digest is computed from the matched line's trimmed first line, so inserting or removing lines does not churn the baseline. Duplicate normalized lines are represented with a count suffix such as `path:sha1:2`; matches beyond the baselined multiset are new blocking debt, while anchors no longer matched remain shrink-only stale debt.

## Cursor-authority reconciliation

`polylogue ops maintenance cursor-authority-reconcile` is a dry-run-by-default
repair route for exactly one proven cursor-ahead source. It reads the
configured `POLYLOGUE_ARCHIVE_ROOT` (using its resolved archive root), requires the daemon to be stopped, and
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
devtools verify scenario run archive-smoke --tier 0
devtools verify scenario run reader-visual-smoke
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
