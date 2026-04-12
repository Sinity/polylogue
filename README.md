# Polylogue

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-4584b6?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="MIT License"></a>
  <a href="https://github.com/sinity/polylogue/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/sinity/polylogue/ci.yml?branch=master&label=ci" alt="CI status"></a>
</p>

Polylogue is a local archive for AI conversation exports. It ingests provider
exports into a SQLite archive with lexical search, optional semantic retrieval,
derived products, and rendered outputs.

## What It Covers

- archive substrate: acquire, parse, normalize, store, query
- derived products: profiles, work events, phases, threads, day and week summaries
- surfaces: CLI, Python API, MCP server, static site, dashboard
- verification and maintenance: schema verification, synthetic corpora, showcase baselines,
  validation lanes, mutation campaigns, benchmark campaigns

Supported providers:

- ChatGPT
- Claude web
- Claude Code
- Codex
- Gemini via Google Drive

## Quickstart

### From Source

```bash
git clone https://github.com/sinity/polylogue
cd polylogue
direnv allow   # or: nix develop
```

### Try It With Synthetic Data

```bash
eval "$(polylogue audit generate --seed --env-only)"

polylogue "error handling" list --limit 5
polylogue products profiles --limit 5
polylogue run site -o ./site-preview
```

The seeded environment exercises the real archive pipeline and keeps demo data
isolated from your normal archive.

### Ingest Real Exports

Polylogue follows the XDG layout by default:

- `~/.local/share/polylogue/polylogue.db` for the archive database
- `~/.local/share/polylogue/inbox/` for dropped exports
- `~/.local/share/polylogue/render/` for rendered outputs
- `~/.local/share/polylogue/site/` for generated publication sites

Typical flow:

```bash
cp ~/Downloads/conversations.json ~/.local/share/polylogue/inbox/
polylogue run all
polylogue -p chatgpt --latest open
```

## CLI Shape

The root command is query-first:

```bash
polylogue "error handling" list --limit 10
polylogue --provider claude-ai --since "last week" stats --by provider
polylogue --latest open
polylogue "urgent" --tag review delete --dry-run
```

The pipeline and archive-maintenance surfaces are explicit verbs:

```bash
polylogue run all
polylogue run materialize
polylogue doctor --repair --preview
polylogue audit --only exercises --tier 0
polylogue mcp
```

See [docs/cli-reference.md](docs/cli-reference.md) for the generated command
reference from live help output.

### Shell Completion

For zsh:

```bash
mkdir -p ~/.zfunc
polylogue completions --shell zsh > ~/.zfunc/_polylogue
fpath=(~/.zfunc $fpath)
autoload -Uz compinit && compinit
```

The completion surface includes command descriptions plus live archive-backed
values for recent conversation IDs, tags, tools, and configured sources.

## Core Surfaces

### Archive and Query

- provider detection from file content, not filename conventions alone
- FTS-backed lexical search with composable filters
- optional vector search for semantic similarity
- local-first storage and rendering

### Durable Products

```bash
polylogue products profiles --limit 10
polylogue products phases --limit 10
polylogue products threads --limit 10
```

These products are materialized read models over the archive, not one-off
reports.

### Publication and MCP

```bash
polylogue run site -o ./site-preview
polylogue mcp
```

- the site surface publishes a browsable HTML archive with search
- the MCP surface exposes archive retrieval to AI assistants

### Library API

```python
from polylogue import Polylogue

async with Polylogue() as archive:
    stats = await archive.stats()
    convs = await (
        archive.filter()
        .provider("claude-ai")
        .contains("error handling")
        .limit(10)
        .list()
    )
```

The Python API is async-first and shares the same archive semantics as the CLI.

## Developer Tools

Repository maintenance work is centralized under `devtools`:

```bash
devtools --help
devtools --list-commands --json
devtools status
devtools render-all
devtools run-validation-lanes --lane frontier-local
```

Use it for generated surfaces, validation lanes, mutation campaigns, benchmark
campaigns, showcase verification, and repository hygiene. See
[docs/devtools.md](docs/devtools.md). JSON forms are available for scripts and
other tooling.

<!-- BEGIN GENERATED: docs-surface -->
## Documentation

For the full docs map, see [docs/README.md](docs/README.md).

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli-reference.md) | Generated command reference from live help output. |
| [Configuration](docs/configuration.md) | XDG paths, environment variables, and runtime configuration. |
| [Library API](docs/library-api.md) | Async archive API, filters, and query patterns. |
| [Developer Tools](docs/devtools.md) | `devtools` guide for generated surfaces, validation, and repo hygiene. |
| [Architecture](docs/architecture.md) | System rings, ownership boundaries, and data flow. |
| [Providers](docs/providers/README.md) | Provider-specific parsing and export-format notes. |

## Contributor Guides

| Document | Description |
|----------|-------------|
| [Contributing](CONTRIBUTING.md) | Branching, issues, PRs, squash-merge history, and repo policy. |
| [Testing](TESTING.md) | Baseline test matrix, protected surfaces, and QA entrypoints. |
| [Agent Guide](CLAUDE.md) | Repository-specific agent memory, workflow rules, and included references. |

<!-- END GENERATED: docs-surface -->

## Development

```bash
pytest -q --ignore=tests/integration
ruff check polylogue tests devtools
devtools render-all --check
devtools build-package
nix flake check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and PR rules,
[TESTING.md](TESTING.md) for the testing guide, and
[docs/internals.md](docs/internals.md) for debugging landmarks and invariants.

## Versioning

`pyproject.toml` records the last tagged release. Development builds are
identified by git metadata, and `polylogue --version` includes the current
commit hash so day-to-day builds stay identifiable without fake release churn.

Routine merges do not bump the package version. Only release-tagging slices
change `version = "X.Y.Z"`.

## License

[MIT](LICENSE)
