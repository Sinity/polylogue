# Developer Tools

Use `devtools` for routine repository maintenance. Call individual
`devtools/*.py` modules directly only when you are editing these tools.

It exposes both human and JSON discovery/status forms. Use the JSON forms for
scripts and agents.

## Command Ownership Policy

`devtools` is the repository control plane. It owns orchestration around local
repo readiness: generated-surface rendering, baseline verification, validation
lane dispatch, package/build checks, and branch/PR readiness gates.

Domain proof semantics belong in the verification-lab, proof, schema, scenario,
or insight modules first. A `devtools` command may expose them only as a thin
operator entrypoint that delegates to the owning lab or insight implementation.

Routine command placement:

- keep repo state, rendering, packaging, and PR-readiness orchestration in
  `devtools`;
- keep archive/insight workflows in `polylogue` CLI/API surfaces;
- keep proof/evidence/scenario behavior behind the verification-lab surface;
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
| `devtools render-verification-catalog` | Refresh or verify the catalog that anchors changed-path verification reports after changing subjects, claims, runners, or catalog rendering. Use --anti-vacuity to flag claims with gaps. |
| `devtools verification-impact` | Find the checks and inner-loop commands affected by local changes before escalating to full PR gates. Use --full for domain-grouped impact analysis. |
| `devtools semantic-axis-evidence` | Produce comparative performance evidence that describes growth shape over semantic axes instead of machine-specific absolute budgets. |
| `devtools lab-corpus` | Seed synthetic corpus files or complete demo workspaces for lab exercises. |
| `devtools lab-scenario` | Run showcase exercise smoke scenarios and committed baseline checks outside the archive CLI. |
| `devtools schema-generate` | Refresh provider schema package artifacts from archive observations outside the archive CLI. |
| `devtools schema-promote` | Turn reviewed schema evidence clusters into committed provider schema packages. |
| `devtools schema-audit` | Check committed schema package quality gates without presenting them as normal archive usage. |
| `devtools verify-schema-roundtrip` | Close the schema inference-validation loop: package manifests must roundtrip through typed models, and every supported element schema must be reachable from the runtime registry. |

## Core Loop

These are the commands worth remembering during normal repo work:

- `devtools status`: Check repo state, generated-surface drift, and the next default verification steps.
  Common forms: `devtools status`, `devtools status --json`, `devtools status --verify-generated`.
- `devtools render-all`: Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.
  Common forms: `devtools render-all`, `devtools render-all --check`.
- `devtools verify`: Run format, lint, mypy, render-all, and test checks locally before pushing.
  Common forms: `devtools verify`, `devtools verify --quick`, `devtools verify --lab`.
- `devtools mutmut-campaign`: Run or inspect focused mutation-testing work without shrinking the committed mutmut scope.
  Common forms: `devtools mutmut-campaign list`, `devtools mutmut-campaign run filters`.
- `devtools benchmark-campaign`: Record durable benchmark artifacts or compare a candidate run against a baseline artifact.
  Common forms: `devtools benchmark-campaign list`, `devtools benchmark-campaign run search-filters`, `devtools benchmark-campaign compare baseline.json candidate.json`.

### Core

| Command | Description |
| --- | --- |
| `devtools motd` | Alias for `status`. |
| `devtools status` | Render the devshell status view. |

### Generated Surfaces

| Command | Description |
| --- | --- |
| `devtools build-topology-projection` | Generate docs/plans/topology-target.yaml from the current tree using placement rules. |
| `devtools render-agents` | Render AGENTS.md from CLAUDE.md and its included files. |
| `devtools render-all` | Refresh or verify generated docs and agent files. |
| `devtools render-cli-reference` | Render docs/cli-reference.md from live CLI help. |
| `devtools render-devtools-reference` | Render the command catalog inside docs/devtools.md. |
| `devtools render-docs-surface` | Render docs/README.md and the README documentation table. |
| `devtools render-pages` | Build the GitHub Pages documentation site into .cache/site/. |
| `devtools render-quality-reference` | Render docs/test-quality-workflows.md from live validation, mutation, and benchmark registries. |
| `devtools render-readme-media` | Generate README media assets (architecture diagrams, flowcharts) under docs/media/. |
| `devtools render-topology-status` | Render docs/topology-status.md from the topology projection and realized tree. |
| `devtools render-verification-catalog` | Render the verification-lab catalog from check registries; optionally emit anti-vacuity report. |

### Verification

| Command | Description |
| --- | --- |
| `devtools artifact-graph` | Render the runtime artifact, operation, and scenario-coverage map. |
| `devtools coverage-gate` | Run pytest with the repository coverage floor from pyproject.toml. |
| `devtools daemon-workload-probe` | Inspect daemon ingest workload, convergence debt, and hot query plans. |
| `devtools evidence-dashboard` | Render the pytest-first evidence dashboard or a changed-path trace. |
| `devtools evidence-report` | Aggregate verification evidence into a structured status report. |
| `devtools lab-corpus` | Generate verification-lab synthetic corpus fixtures and demo archives. |
| `devtools lab-scenario` | Run verification-lab showcase scenario sets and baseline checks. |
| `devtools pipeline-probe` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools query-memory-budget` | Measure query-memory envelopes on generated fixtures. |
| `devtools regression-capture` | Capture pipeline-probe summaries as durable local regression cases. |
| `devtools run-validation-lanes` | Run named validation lanes. |
| `devtools scenario-projections` | Render the authored scenario-bearing verification projections. |
| `devtools schema-audit` | Run committed provider schema package quality checks. |
| `devtools schema-generate` | Generate provider schema packages and optional evidence clusters. |
| `devtools schema-promote` | Promote a schema evidence cluster into a registered package version. |
| `devtools semantic-axis-evidence` | Generate verification-lab performance evidence across synthetic semantic scale tiers. |
| `devtools verification-impact` | Route changed paths or refs to affected verification checks and focused commands; emit the full PR-confidence report with --full. |
| `devtools verify` | Run the local verification baseline before pushing or creating a PR. |
| `devtools verify-ci-workflows` | Verify CI workflow files reference locally-known devtools commands and existing paths. |
| `devtools verify-closure-matrix` | Verify docs/plans/test-closure-matrix.yaml stays grounded in the realized tree. |
| `devtools verify-cluster-cohesion` | Validate proposed clusters from the topology projection using the import graph. |
| `devtools verify-cross-cuts` | Verify cross-cut tags in the topology projection match module-name conventions. |
| `devtools verify-distribution-surface` | Verify wheel/sdist installed artifacts expose only supported runtime entrypoints. |
| `devtools verify-file-budgets` | Enforce per-file LOC budgets declared in docs/plans/file-size-budgets.yaml. |
| `devtools verify-lane-assertions` | Verify scenario lanes classified as SEMANTIC_OUTPUT carry semantic assertions. |
| `devtools verify-layering` | Check inter-package imports against declared layering rules from docs/plans/layering.yaml. |
| `devtools verify-manifests` | Verify internal consistency across all docs/plans/*.yaml manifest files. |
| `devtools verify-migrations` | Verify migration-completeness against docs/plans/migrations.yaml. |
| `devtools verify-schema-roundtrip` | Verify committed provider schema packages reload and roundtrip cleanly. |
| `devtools verify-slos` | Check read-surface latency budgets in docs/plans/slo-catalog.yaml against benchmark measurements. |
| `devtools verify-suppressions` | Enforce suppression registry expiry dates from docs/plans/suppressions.yaml. |
| `devtools verify-test-ownership` | Verify each production module is imported by at least one unit test. |
| `devtools verify-topology` | Verify the realized polylogue tree against the topology projection. |
| `devtools verify-witness-lifecycle` | Verify committed witness lifecycle health — staleness, unexercised, stale xfails. |

### Campaigns

| Command | Description |
| --- | --- |
| `devtools benchmark-campaign` | Run or compare benchmark campaigns. |
| `devtools mutmut-campaign` | Run focused mutation campaigns and maintain their local index. |
| `devtools run-benchmark-campaigns` | Run synthetic benchmark campaigns over generated archives. |

### Maintenance

| Command | Description |
| --- | --- |
| `devtools build-package` | Build the default Nix package with the out-link under .local/result. |
| `devtools witness-discover` | Save a failure-triggering input as a local witness in .local/witnesses/new/. |
| `devtools witness-minimize` | Apply minimization heuristics to a local witness — shrink, redact, set privacy classification. |
| `devtools witness-promote` | Promote a minimized local witness to tests/witnesses/ for durable commit. |
| `devtools xtask` | Record and query agent task execution history (.agent/xtask/tasks.jsonl). |

<!-- END GENERATED: devtools-command-catalog -->

## Validation and Proof

When changing semantics, validation, or surfaces:

```bash
devtools run-validation-lanes --list
devtools run-validation-lanes --lane frontier-local
devtools lab-scenario verify-baselines
devtools query-memory-budget --max-rss-mb 1536 -- polylogue --plain stats
```

Campaign outputs live under `.local/`, not in tracked docs trees.

## Local State Layout

- `.cache/`: disposable cache state.
- `.local/`: untracked local outputs such as campaigns, showcases, and reports.
- `.venv/` and `.direnv/`: kept at the repo root because their tooling expects those locations.
- `.local/result`: preferred repo-local out-link for `devtools build-package`; a top-level `result` symlink is just Nix's default ad-hoc out-link.

Keep new repo-local outputs in `.cache/` or `.local/` instead of adding new
top-level output roots.
