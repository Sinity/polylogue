# Polylogue

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-4584b6?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="MIT License"></a>
  <a href="https://github.com/sinity/polylogue/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/sinity/polylogue/ci.yml?branch=master&label=ci" alt="CI status"></a>
</p>

Polylogue is a local archive for AI conversation exports. It ingests provider
exports into a SQLite archive with lexical search, optional semantic retrieval,
derived products, and rendered outputs.

## What It Does

- Ingest exports from ChatGPT, Claude, Claude Code, Codex, and Gemini
- Full-text and semantic search across all conversations
- Derived products: session profiles, work events, threads, summaries
- CLI, Python API, MCP server, static site, dashboard
- Synthetic corpora, QA exercises, mutation testing, benchmarks

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

Demo data is isolated from your normal archive.

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

- provider detection from file content
- full-text search with composable filters
- optional vector search for semantic similarity

### Durable Products

```bash
polylogue products profiles --limit 10
polylogue products phases --limit 10
polylogue products threads --limit 10
```

Products are materialized from the archive and updated incrementally.

### Publication and MCP

```bash
polylogue run site -o ./site-preview
polylogue mcp
```

- static HTML archive with search
- archive queries for AI assistants via MCP

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

The Python API is async-first.

## Developer Tools

Repository maintenance work is centralized under `devtools`:

```bash
devtools --help
devtools --list-commands --json
devtools status
devtools render-all
devtools run-validation-lanes --lane frontier-local
```

See [docs/devtools.md](docs/devtools.md) for the full command catalog.

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
| [Agent Guide](CLAUDE.md) | Agent memory and working rules. |

<!-- END GENERATED: docs-surface -->

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md), [TESTING.md](TESTING.md), and
[docs/internals.md](docs/internals.md).

## License

[MIT](LICENSE)
