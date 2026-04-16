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

## Core Loop

These are the commands worth remembering during normal repo work:

- `devtools status`: Check repo state, generated-surface drift, and the next default verification steps.
  Common forms: `devtools status`, `devtools status --json`, `devtools status --verify-generated`.
- `devtools render-all`: Refresh or verify every generated repo surface together after changing docs, CLI help, or agent memory.
  Common forms: `devtools render-all`, `devtools render-all --check`.
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

### Verification

| Command | Description |
| --- | --- |
| `devtools pipeline-probe` | Run synthetic pipeline probes against generated archives. |
| `devtools query-memory-budget` | Measure query-memory envelopes on generated fixtures. |
| `devtools run-scale-lanes` | Run scale-validation lanes. |
| `devtools run-validation-lanes` | Run named validation lanes. |
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

Use these tools for:

- generated-document refresh
- generated-document drift checks
- showcase verification
- validation lanes
- mutation campaigns
- benchmark campaigns
- pipeline probes and query-memory checks

## Generated Surfaces

Refresh the generated documentation and agent surfaces together:

```bash
devtools render-all
devtools render-all --check
```

That refreshes:

- `AGENTS.md` from `CLAUDE.md`
- `docs/cli-reference.md` from live CLI help
- the generated command catalog in this file
- `docs/test-quality-workflows.md` from the quality registries
- `docs/README.md` and the generated docs table in `README.md`

## Validation and Proof

```bash
devtools run-validation-lanes --list
devtools run-validation-lanes --lane frontier-local
devtools verify-showcase
devtools query-memory-budget --max-rss-mb 1536 -- polylogue --plain stats
```

Use these when changing semantics, validation contracts, or user-facing
surfaces.

## Mutation and Benchmark Campaigns

```bash
devtools mutmut-campaign list
devtools mutmut-campaign run filters
devtools benchmark-campaign list
devtools benchmark-campaign run search-filters
devtools benchmark-campaign compare baseline.json candidate.json
```

Campaign outputs live under [`.local/README.md`](../.local/README.md), not in
tracked docs trees.

Use `devtools build-package` when you want the standard local Nix build output
without spraying a `result` symlink into the repo root.

## Local State Layout

- [`.cache/README.md`](../.cache/README.md): disposable cache state
- [`.local/README.md`](../.local/README.md): untracked local outputs
- `.venv/` and `.direnv/`: kept at the repo root because their tooling expects those locations
- `.local/result`: canonical local Nix build out-link

Keep new repo-local outputs in one of those hidden roots.
