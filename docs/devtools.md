# Developer Tools

Use `devtools` for routine repository maintenance. Call individual
`devtools/*.py` modules directly only when you are editing these tools.

It exposes both human and JSON discovery/status forms. Use the JSON forms for
scripts and agents.

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
| `devtools lab-scenario` | Run showcase exercise smoke scenarios outside the product CLI. |

## Core Loop

These are the commands worth remembering during normal repo work:

- `devtools status`: Check repo state, generated-surface drift, and the next default verification steps.
  Common forms: `devtools status`, `devtools status --json`, `devtools status --verify-generated`.
- `devtools render-all`: Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.
  Common forms: `devtools render-all`, `devtools render-all --check`.
- `devtools verify`: Run format, lint, mypy, render-all, and test checks locally before pushing.
  Common forms: `devtools verify`, `devtools verify --quick`.
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
| `devtools lab-corpus` | Generate verification-lab synthetic corpus fixtures and demo archives. |
| `devtools lab-scenario` | Run verification-lab showcase scenario sets. |
| `devtools pipeline-probe` | Run typed pipeline probes against synthetic, staged, or archive-subset inputs. |
| `devtools query-memory-budget` | Measure query-memory envelopes on generated fixtures. |
| `devtools regression-capture` | Capture pipeline-probe summaries as durable local regression cases. |
| `devtools run-validation-lanes` | Run named validation lanes. |
| `devtools scenario-projections` | Render the authored scenario-bearing verification projections. |
| `devtools semantic-axis-evidence` | Generate verification-lab performance evidence across synthetic semantic scale tiers. |
| `devtools verify` | Run the local verification baseline before pushing or creating a PR. |
| `devtools verify-showcase` | Verify committed showcase/demo surfaces. |

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
devtools verify-showcase
devtools query-memory-budget --max-rss-mb 1536 -- polylogue --plain stats
```

Campaign outputs live under [`.local/README.md`](../.local/README.md), not in
tracked docs trees.

## Local State Layout

- [`.cache/README.md`](../.cache/README.md): disposable cache state
- [`.local/README.md`](../.local/README.md): untracked local outputs
- `.venv/` and `.direnv/`: kept at the repo root because their tooling expects those locations
- `.local/result`: preferred repo-local out-link for `devtools build-package`; a top-level `result` symlink is just Nix's default ad-hoc out-link

Keep new repo-local outputs in one of those hidden roots.
