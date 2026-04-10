# Developer Tools

`python -m devtools` is the repo-maintenance entrypoint.

Use `python -m devtools` for routine repo-maintenance work. Call individual
`devtools/*.py` modules directly only when you are editing these tools.

It exposes both human and JSON discovery/status forms. Use the JSON forms for
scripts and agents.

<!-- BEGIN GENERATED: devtools-command-catalog -->
## Command Catalog

Use these discovery commands before scripting or dispatching subcommands:

```bash
python -m devtools --help
python -m devtools --list-commands
python -m devtools --list-commands --json
python -m devtools status
python -m devtools status --json
```

### Core

| Command | Description |
| --- | --- |
| `python -m devtools motd` | Alias for `status`. |
| `python -m devtools status` | Render the devshell status view. |

### Generated Surfaces

| Command | Description |
| --- | --- |
| `python -m devtools render-agents` | Render AGENTS.md from CLAUDE.md and its included files. |
| `python -m devtools render-all` | Refresh or verify generated docs and agent files. |
| `python -m devtools render-cli-reference` | Render docs/cli-reference.md from live CLI help. |
| `python -m devtools render-devtools-reference` | Render the command catalog inside docs/devtools.md. |
| `python -m devtools render-docs-surface` | Render docs/README.md and the README documentation table. |
| `python -m devtools render-quality-reference` | Render docs/test-quality-workflows.md from live validation, mutation, and benchmark registries. |

### Verification

| Command | Description |
| --- | --- |
| `python -m devtools pipeline-probe` | Run synthetic pipeline probes against generated archives. |
| `python -m devtools query-memory-budget` | Measure query-memory envelopes on generated fixtures. |
| `python -m devtools run-scale-lanes` | Run scale-validation lanes. |
| `python -m devtools run-validation-lanes` | Run named validation lanes. |
| `python -m devtools verify-showcase` | Verify committed showcase/demo surfaces. |

### Campaigns

| Command | Description |
| --- | --- |
| `python -m devtools benchmark-campaign` | Run or compare benchmark campaigns. |
| `python -m devtools mutmut-campaign` | Run focused mutation campaigns and maintain their local index. |
| `python -m devtools run-benchmark-campaigns` | Run synthetic benchmark campaigns over generated archives. |

### Maintenance

| Command | Description |
| --- | --- |
| `python -m devtools inject-semantic-annotations` | Annotate baseline provider schemas with semantic-role metadata. |

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
python -m devtools render-all
python -m devtools render-all --check
```

That refreshes:

- `AGENTS.md` from `CLAUDE.md`
- `docs/cli-reference.md` from live CLI help
- the generated command catalog in this file
- `docs/test-quality-workflows.md` from the quality registries
- `docs/README.md` and the generated docs table in `README.md`

## Validation and Proof

```bash
python -m devtools run-validation-lanes --list
python -m devtools run-validation-lanes --lane frontier-local
python -m devtools verify-showcase
python -m devtools query-memory-budget --max-rss-mb 1536 -- python -m polylogue --plain stats
```

Use these when changing semantics, validation contracts, or user-facing
surfaces.

## Mutation and Benchmark Campaigns

```bash
python -m devtools mutmut-campaign list
python -m devtools mutmut-campaign run filters
python -m devtools benchmark-campaign list
python -m devtools benchmark-campaign run search-filters
python -m devtools benchmark-campaign compare baseline.json candidate.json
```

Campaign outputs live under [`.local/README.md`](../.local/README.md), not in
tracked docs trees.

## Local State Layout

- [`.cache/README.md`](../.cache/README.md): disposable cache state
- [`.local/README.md`](../.local/README.md): untracked local outputs
- `.venv/`, `.direnv/`, and `result*`: tool-owned roots that stay at the repo root

Keep new repo-local outputs in one of those hidden roots.
