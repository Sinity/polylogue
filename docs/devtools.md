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
| `devtools lab policy schema-versioning` | Enforce the policy boundary documented in docs/internals.md § 'Schema Versioning Model'. Polylogue intentionally has no in-place storage schema upgrade chain; archive-shape changes edit the canonical DDL and require a fresh rebuild from source. |
| `devtools lab provider completeness` | Inspect detector, parser, fixture, schema, docs, ImportExplain, and caveat coverage before claiming a provider/importer mode is product-ready. |
| `devtools lab probe capture-regression` | Turn a live or probe failure JSON summary into a replayable local regression artifact. |
| `devtools lab probe pipeline` | Run real pipeline stages and optionally capture emitted summaries as regression cases. |
| `devtools lab probe turso` | Collect executable evidence before changing production storage backends: Python binding availability, generated-column support, FTS compatibility, MVCC, CDC, vector functions, ATTACH, and WAL pragma behavior. |
| `devtools lab projections` | Inspect the unified projection inventory that feeds runtime coverage, generated docs, and control-plane maps. |
| `devtools lab smoke` | Run direct archive and reader smoke sets outside the archive CLI. |
| `devtools lab schema audit` | Check committed schema package quality gates without presenting them as normal archive usage. |
| `devtools lab schema compare` | Review schema package drift between committed versions in the lab surface. |
| `devtools lab schema explain` | Inspect schema package annotations, semantic roles, and review evidence from the lab surface. |
| `devtools lab schema generate` | Refresh provider schema package artifacts from archive observations outside the archive CLI. |
| `devtools lab schema list` | Inspect committed provider schema package catalogs without presenting them as normal archive usage. |
| `devtools lab schema promote` | Turn reviewed schema evidence clusters into committed provider schema packages. |
| `devtools lab schema roundtrip` | Close the schema inference-validation loop: package manifests must roundtrip through typed models, and every supported element schema must be reachable from the runtime registry. |
| `devtools lab snapshot read-surface` | Freeze archive read-surface behavior before archive work, then compare candidate archives against the captured envelope baseline. |

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

### Core

| Command | Description |
| --- | --- |
| `devtools status` | Render the devshell status view. |

### Generated Surfaces

| Command | Description |
| --- | --- |
| `devtools render agents` | Render AGENTS.md from CLAUDE.md and its included files. |
| `devtools render all` | Refresh or verify generated docs and agent files. |
| `devtools render cli-output-schemas` | Render JSON Schema artifacts for stable CLI output payloads under docs/schemas/cli-output/. |
| `devtools render cli-reference` | Render docs/cli-reference.md from live CLI help. |
| `devtools render devtools-reference` | Render the command catalog inside docs/devtools.md. |
| `devtools render docs-surface` | Render docs/README.md and the README documentation table. |
| `devtools render openapi` | Render docs/openapi/search.yaml from typed daemon query payload models. |
| `devtools render pages` | Build the GitHub Pages documentation site into .cache/site/. |
| `devtools render product-workflows` | Render docs/product/workflows.md from executable query-action workflow registries. |
| `devtools render quality-reference` | Render docs/test-quality-workflows.md from executable lane, mutation, and benchmark registries. |
| `devtools render topology-projection` | Generate docs/plans/topology-target.yaml from the current tree using placement rules. |
| `devtools render topology-status` | Render docs/topology-status.md from the topology projection and realized tree. |
| `devtools render visual-tapes` | Write VHS tape files and optionally capture GIFs for the default visual evidence specs. |

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
| `devtools lab policy schema-versioning` | Reject in-place storage schema upgrade helpers (#1302). |
| `devtools lab probe capture-regression` | Capture pipeline-probe summaries as durable local regression cases. |
| `devtools lab probe pipeline` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools lab probe turso` | Probe Turso Database compatibility against Polylogue storage assumptions. |
| `devtools lab projections` | Render the authored scenario-bearing verification projections. |
| `devtools lab provider completeness` | Report provider/importer package completeness by origin and capture mode. |
| `devtools lab schema audit` | Run committed provider schema package quality checks. |
| `devtools lab schema compare` | Compare two committed schema package versions for a provider. |
| `devtools lab schema explain` | Explain a committed package element schema with evidence and annotations. |
| `devtools lab schema generate` | Generate provider schema packages and optional evidence clusters. |
| `devtools lab schema list` | List committed schema packages, versions, and evidence manifests. |
| `devtools lab schema promote` | Promote a schema evidence cluster into a registered package version. |
| `devtools lab schema roundtrip` | Verify committed provider schema packages reload and roundtrip cleanly. |
| `devtools lab smoke` | Run direct archive and reader smoke sets. |
| `devtools lab snapshot read-surface` | Capture and compare archive read-surface snapshots. |

### Verification

| Command | Description |
| --- | --- |
| `devtools test` | Run a focused pytest selection through the managed harness. |
| `devtools verify` | Run the local verification baseline before pushing or creating a PR. |
| `devtools verify ci-workflows` | Verify CI workflow files reference locally-known devtools commands and existing paths. |
| `devtools verify closure-matrix` | Verify docs/plans/test-closure-matrix.yaml stays grounded in the realized tree. |
| `devtools verify coverage` | Run pytest with the repository coverage floor from pyproject.toml. |
| `devtools verify doc-commands` | Verify README/docs command examples resolve to live polylogue, polylogued, and devtools commands. |
| `devtools verify evidence` | Render the pytest-first evidence dashboard or a changed-path trace. |
| `devtools verify layering` | Check inter-package imports against declared layering rules from docs/plans/layering.yaml. |
| `devtools verify manifests` | Verify internal consistency across all docs/plans/*.yaml manifest files. |
| `devtools verify test-clock-hygiene` | Verify test files use the frozen_clock fixture instead of reading the host wall clock (#1300). |
| `devtools verify test-infra-currency` | Verify tests/infra/ helpers reference only tables that exist in the current SCHEMA_VERSION. |
| `devtools verify topology` | Verify the realized polylogue tree against the topology projection. |

### Benchmarking

| Command | Description |
| --- | --- |
| `devtools bench campaign` | Run or compare benchmark campaigns. |
| `devtools bench ingest-amplification` | Measure deterministic per-tier ingest write amplification on a synthetic fixture (#1851). |
| `devtools bench ingest-throughput` | Measure ingest wall-clock throughput on a synthetic fixture. |
| `devtools bench memory` | Measure query-memory envelopes on generated fixtures. |
| `devtools bench mutation` | Run focused mutation campaigns and maintain their local index. |
| `devtools bench slo` | Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements. |
| `devtools bench synthetic` | Run synthetic benchmark campaigns over generated archives. |

### Workspace

| Command | Description |
| --- | --- |
| `devtools workspace deployment-smoke` | Probe deployed Polylogue binaries, daemon/web routes, and browser-capture archive flow. |
| `devtools workspace dev-loop` | Preflight branch-local daemon, web-shell, and browser-capture development loops. |
| `devtools workspace failure-context` | Join testmon, git history, and fixtures for a pytest failure ID into a JSON envelope. |
| `devtools workspace tasks` | Record and query local agent task execution history. |
| `devtools workspace temporal-devloop` | Compose git and operating-log events into a temporal evidence window. |
| `devtools workspace temporal-read-profile` | Measure read --view temporal phase timings on the active archive. |
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
