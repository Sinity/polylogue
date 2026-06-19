# Developer Tools

Use `devtools` for routine repository maintenance. Call individual
`devtools/*.py` modules directly only when you are editing these tools.

It exposes both human and JSON discovery/status forms. Use the JSON forms for
scripts and agents.

## Command Ownership Policy

`devtools` is the repository control plane. It owns orchestration around local
repo readiness: generated-surface rendering, baseline verification, validation
lane dispatch, package/build checks, and branch/PR readiness gates.

Domain validation semantics belong in the verification-lab, schema, scenario,
or insight modules first. A `devtools` command may expose them only as a thin
operator entrypoint that delegates to the owning lab or insight implementation.

Routine command placement:

- keep repo state, rendering, packaging, and PR-readiness orchestration in
  `devtools`;
- keep archive/insight workflows in `polylogue` CLI/API surfaces;
- keep evidence/scenario behavior behind the verification-lab surface;
- prefer validation lanes and `devtools verify --lab` to compose lab checks
  rather than duplicating domain checks inside `devtools verify`.

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

## Verification Lab Surface

The verification-lab operator surface intentionally lives in `devtools` for now. These commands operate on
repo verification checks and evidence records, not end-user archive workflows.

| Command | Role |
| --- | --- |
| `devtools lab graph` | Inspect the authored runtime graph and see which scenarios currently cover declared artifacts and operations. |
| `devtools lab lanes` | List, dry-run, or execute authored validation lanes from the verification lab registry. |
| `devtools lab policy schema-versioning` | Enforce the policy boundary documented in docs/internals.md § 'Schema Versioning Model'. Polylogue intentionally has no in-place storage schema upgrade chain; archive-shape changes edit the canonical DDL and require a fresh rebuild from source. |
| `devtools lab probe capture-regression` | Turn a live or probe failure JSON summary into a replayable local regression artifact. |
| `devtools lab probe pipeline` | Exercise real pipeline stages and optionally capture emitted summaries as regression cases. |
| `devtools lab projections` | Inspect the unified projection inventory that feeds runtime coverage, generated docs, and control-plane maps. |
| `devtools lab scenario` | Run showcase exercise smoke scenarios and committed baseline checks outside the archive CLI. |
| `devtools lab schema audit` | Check committed schema package quality gates without presenting them as normal archive usage. |
| `devtools lab schema generate` | Refresh provider schema package artifacts from archive observations outside the archive CLI. |
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
| `devtools render quality-reference` | Render docs/test-quality-workflows.md from live validation, mutation, and benchmark registries. |
| `devtools render readme-media` | Generate README media assets (architecture diagrams, flowcharts) under docs/media/. |
| `devtools render topology-projection` | Generate docs/plans/topology-target.yaml from the current tree using placement rules. |
| `devtools render topology-status` | Render docs/topology-status.md from the topology projection and realized tree. |

### Release

| Command | Description |
| --- | --- |
| `devtools release build-package` | Build the default Nix package with the out-link under .local/result. |
| `devtools release readiness` | Validate the externally-presentable release gate definition. |
| `devtools release verify-distribution` | Verify wheel/sdist installed artifacts expose only supported runtime entrypoints. |

### Verification Lab

| Command | Description |
| --- | --- |
| `devtools lab graph` | Render the runtime artifact, operation, and scenario-coverage map. |
| `devtools lab lanes` | Run named validation lanes. |
| `devtools lab policy schema-versioning` | Reject in-place storage schema upgrade helpers (#1302). |
| `devtools lab probe capture-regression` | Capture pipeline-probe summaries as durable local regression cases. |
| `devtools lab probe pipeline` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools lab projections` | Render the authored scenario-bearing verification projections. |
| `devtools lab scenario` | Run verification-lab showcase scenario sets and baseline checks. |
| `devtools lab schema audit` | Run committed provider schema package quality checks. |
| `devtools lab schema generate` | Generate provider schema packages and optional evidence clusters. |
| `devtools lab schema promote` | Promote a schema evidence cluster into a registered package version. |
| `devtools lab schema roundtrip` | Verify committed provider schema packages reload and roundtrip cleanly. |
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
| `devtools verify lane-assertions` | Verify scenario lanes classified as SEMANTIC_OUTPUT carry semantic assertions. |
| `devtools verify layering` | Check inter-package imports against declared layering rules from docs/plans/layering.yaml. |
| `devtools verify manifests` | Verify internal consistency across all docs/plans/*.yaml manifest files. |
| `devtools verify test-clock-hygiene` | Verify test files use the frozen_clock fixture instead of reading the host wall clock (#1300). |
| `devtools verify test-coverage-contracts` | Verify every production module >150 AST lines has a matching test file or exemption. |
| `devtools verify test-infra-currency` | Verify tests/infra/ helpers reference only tables that exist in the current SCHEMA_VERSION. |
| `devtools verify topology` | Verify the realized polylogue tree against the topology projection. |

### Benchmarking

| Command | Description |
| --- | --- |
| `devtools bench campaign` | Run or compare benchmark campaigns. |
| `devtools bench ingest-amplification` | Measure deterministic per-tier ingest write amplification on a synthetic fixture (#1851). |
| `devtools bench memory` | Measure query-memory envelopes on generated fixtures. |
| `devtools bench mutation` | Run focused mutation campaigns and maintain their local index. |
| `devtools bench slo` | Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements. |
| `devtools bench synthetic` | Run synthetic benchmark campaigns over generated archives. |

### Workspace

| Command | Description |
| --- | --- |
| `devtools workspace failure-context` | Join testmon, git history, and fixtures for a pytest failure ID into a JSON envelope. |
| `devtools workspace tasks` | Record and query local agent task execution history. |
| `devtools workspace worktree-gc` | Safe worktree garbage collection — list and remove merged or abandoned git worktrees. |

### Maintenance

| Command | Description |
| --- | --- |
| `devtools provider completeness` | Report provider/importer package completeness by origin and capture mode. |

<!-- END GENERATED: devtools-command-catalog -->

## Validation and Evidence

When changing semantics, validation, or surfaces:

```bash
devtools lab lanes --list
devtools lab lanes --lane frontier-local
devtools lab scenario verify-baselines
devtools bench memory --max-rss-mb 1536 -- polylogue --plain analyze
```

Campaign outputs live under `.local/`, not in tracked docs trees.

## Local State Layout

- `.cache/`: disposable cache state.
- `.local/`: untracked local outputs such as campaigns, showcases, and reports.
- `.venv/` and `.direnv/`: kept at the repo root because their tooling expects those locations.
- `.local/result`: preferred repo-local out-link for `devtools release build-package`; a top-level `result` symlink is just Nix's default ad-hoc out-link.

Keep new repo-local outputs in `.cache/` or `.local/` instead of adding new
top-level output roots.
