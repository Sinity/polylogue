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
or product modules first. A `devtools` command may expose them only as a thin
operator entrypoint that delegates to the owning lab/product implementation.

Routine command placement:

- keep repo state, rendering, packaging, and PR-readiness orchestration in
  `devtools`;
- keep archive/product workflows in `polylogue` CLI/API surfaces;
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

The proof-lab operator surface intentionally lives in `devtools` for now. These commands operate on
repo proof obligations and evidence records, not end-user archive workflows.

| Command | Role |
| --- | --- |
| `devtools render-verification-catalog` | Refresh or verify the proof-obligation catalog that anchors the verification-lab surface after changing proof subjects, claims, runners, or catalog rendering. |
| `devtools affected-obligations` | Find the proof obligations and inner-loop checks affected by local changes before escalating to full PR gates. |
| `devtools semantic-axis-evidence` | Produce comparative performance evidence that describes growth shape over semantic axes instead of machine-specific absolute budgets. |
| `devtools lab-corpus` | Seed synthetic corpus files or complete demo workspaces for lab exercises. |
| `devtools lab-scenario` | Run showcase exercise smoke scenarios and committed baseline checks outside the product CLI. |
| `devtools schema-generate` | Refresh provider schema package artifacts from archive observations outside the product CLI. |
| `devtools schema-promote` | Turn reviewed schema evidence clusters into committed provider schema packages. |
| `devtools schema-audit` | Check committed schema package quality gates without presenting them as normal archive usage. |

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
| `devtools render-quality-reference` | Render docs/test-quality-workflows.md from live validation, mutation, and benchmark registries. |
| `devtools render-verification-catalog` | Render the verification-lab proof catalog from obligation registries. |

### Verification

| Command | Description |
| --- | --- |
| `devtools affected-obligations` | Route changed paths or refs to affected verification-lab proof obligations and focused checks. |
| `devtools artifact-graph` | Render the runtime artifact, operation, and scenario-coverage map. |
| `devtools coverage-gate` | Run pytest with the repository coverage floor from pyproject.toml. |
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
| `devtools verify` | Run the local verification baseline before pushing or creating a PR. |
| `devtools verify-cluster-cohesion` | Validate proposed clusters from the topology projection using the import graph. |
| `devtools verify-file-budgets` | Enforce per-file LOC budgets declared in docs/plans/file-size-budgets.yaml. |
| `devtools verify-migrations` | Verify migration-completeness against docs/plans/migrations.yaml. |
| `devtools verify-showcase` | Verify committed showcase/demo surfaces. |
| `devtools verify-test-ownership` | Verify each production module is imported by at least one unit test. |
| `devtools verify-topology` | Verify the realized polylogue tree against the topology projection. |

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
| `devtools inject-semantic-annotations` | Annotate baseline provider schemas with semantic-role metadata. |

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
