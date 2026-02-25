<p align="center">
  <img src="docs/assets/hero-banner.svg" alt="polylogue" width="700">
</p>

<p align="center">
  <strong>Preserve, index, and expose your AI conversation history as a queryable, programmable archive.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-4584b6?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="MIT License"></a>
  <a href="https://github.com/sinity/polylogue/releases"><img src="https://img.shields.io/badge/version-0.1.0-f97316" alt="Version 0.1.0"></a>
</p>

---

<p align="center">
  <img src="demos/output/01-overview.gif" alt="polylogue overview" width="700">
</p>

## What It Does

Polylogue archives AI conversations from **ChatGPT, Claude, Claude Code, Gemini, and Codex** into a unified, searchable local database. Drop your exports in a folder, run one command, and get instant full-text search across every conversation you've ever had.

- **Zero-config**: Drop exports in `~/.local/share/polylogue/inbox/`, run `polylogue run`, done
- **Sub-second search**: FTS5-powered full-text search with smartcase matching
- **Semantic search**: Vector similarity via sqlite-vec embeddings (optional, Voyage AI)
- **Library-first**: Async Python API with composable filter chains — the CLI is a thin wrapper
- **Local-first**: All data stays on your machine. SQLite database, no external services

## Try It Now

No data needed. Generate a synthetic archive and explore:

```bash
eval $(polylogue demo --seed --env-only)

polylogue                              # Archive stats
polylogue "error handling"             # Full-text search
polylogue -p claude --latest           # Latest Claude conversation
polylogue dashboard                    # Interactive TUI
```

## Quick Start

### 1. Install

```bash
# With uv (recommended)
uv tool install polylogue

# With pip
pip install polylogue

# From source (Nix)
git clone https://github.com/sinity/polylogue && cd polylogue
nix develop   # or: uv sync
```

### 2. Export Your Conversations

| Provider | How to Export |
|----------|--------------|
| **ChatGPT** | Settings → Data Controls → Export → Download `conversations.json` |
| **Claude** | Download from claude.ai conversation history |
| **Claude Code** | Auto-discovered from `~/.claude/projects/` |
| **Codex** | Auto-discovered from `~/.codex/sessions/` |
| **Gemini** | Google Drive sync — run `polylogue auth` for OAuth setup |

### 3. Drop in Inbox & Run

```bash
# Copy or symlink your exports
cp ~/Downloads/conversations.json ~/.local/share/polylogue/inbox/
ln -s ~/.claude/projects ~/.local/share/polylogue/inbox/claude-code

# Run the pipeline
polylogue run

# Search
polylogue "error handling"
```

No config file needed. That's it.

## Feature Highlights

<table>
<tr>
<td width="50%">

### Ingest From Anywhere

```bash
polylogue run
```

Auto-detects provider format from file content. Handles JSON, JSONL, ZIP archives (with bomb protection), and Google Drive sync.

<img src="demos/output/02-run.gif" alt="polylogue run" width="100%">

</td>
<td width="50%">

### Search Across Providers

```bash
polylogue "error handling" -p claude
polylogue --has thinking --since "last week"
```

Filter by provider, date, content type, tags — combine any filters freely.

<img src="demos/output/03-search.gif" alt="polylogue search" width="100%">

</td>
</tr>
<tr>
<td width="50%">

### Interactive Dashboard

```bash
polylogue dashboard
```

Textual-based TUI for browsing, searching, and reading conversations interactively.

<img src="demos/output/04-dashboard.gif" alt="polylogue dashboard" width="100%">

</td>
<td width="50%">

### Generate Static Sites

```bash
polylogue site -o ./public
```

Browsable HTML archive with per-provider views, statistics dashboard, and client-side search.

<img src="demos/output/05-site.gif" alt="polylogue site" width="100%">

</td>
</tr>
<tr>
<td width="50%">

### MCP Integration

```json
{
  "mcpServers": {
    "polylogue": {
      "command": "polylogue",
      "args": ["mcp"]
    }
  }
}
```

Search your archive from Claude Desktop or Claude Code via the Model Context Protocol.

</td>
<td width="50%">

### Library API

```python
async with Polylogue() as archive:
    stats = await archive.stats()
    convs = await (archive.filter()
        .provider("claude")
        .contains("error")
        .limit(10)
        .list())
```

Async-first Python API with composable filters, projections, and batch operations.

</td>
</tr>
</table>

## Query Language

Polylogue's CLI treats positional arguments as search terms. No subcommand prefix — just type what you're looking for:

```bash
# Basic search
polylogue "error handling"             # Full-text search (FTS5)
polylogue "Error"                      # Case-sensitive (has uppercase)
polylogue "auth" "token"               # AND: both terms must appear

# Semantic search (requires VOYAGE_API_KEY)
polylogue --similar "how to debug memory leaks"

# Filters (all composable)
polylogue "error" -p claude,chatgpt    # By provider (comma = OR)
polylogue --since "last week"          # Natural language dates
polylogue --until 2025-01-01           # ISO dates
polylogue --has thinking               # Has reasoning traces
polylogue --has tools                  # Has tool use
polylogue -t project:backend           # By tag (supports key:value)
polylogue --title "API design"         # Title contains

# Sort & limit
polylogue --latest                     # Most recent conversation
polylogue --sort tokens --reverse      # Most expensive first
polylogue --sort longest -n 10         # 10 longest conversations
polylogue --sample 5                   # Random sample

# Output
polylogue "error" -f json              # JSON format
polylogue "error" -f csv               # CSV format
polylogue "error" -o browser           # Open in browser
polylogue "error" -o clipboard         # Copy to clipboard
polylogue "error" --fields id,title,date  # Select columns
polylogue "error" --count              # Just the count
polylogue "error" --stats-by provider  # Aggregate by provider

# Content transforms
polylogue -i abc123 --transform strip-tools     # Hide tool calls
polylogue -i abc123 --transform strip-thinking  # Hide reasoning
polylogue -i abc123 -d                          # Dialogue only (user + assistant)

# Metadata modification
polylogue -i abc123 --set title "My Title"
polylogue -i abc123 --set summary "Brief description"
polylogue -i abc123 --add-tag important,project:backend
polylogue "old stuff" --delete --dry-run        # Preview bulk delete
```

## Supported Providers

| Provider | Format | Auto-detected By | ID |
|----------|--------|------------------|----|
| ChatGPT | `conversations.json` | `mapping` field with UUID graph | `chatgpt` |
| Claude (web) | `.jsonl` | `chat_messages` array | `claude` |
| Claude Code | `.json` array | `parentUuid`/`sessionId` markers | `claude-code` |
| Codex | `.jsonl` | Session envelope structure | `codex` |
| Gemini | Google Drive API | `chunkedPrompt.chunks` structure | `gemini` |

ZIP archives are supported (nested ZIPs too, with bomb protection). Provider detection is automatic from file content — no configuration needed.

## Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| Markdown | `-f markdown` | Default. Syntax-highlighted, human-readable |
| JSON | `-f json` | Machine-readable, with all metadata |
| HTML | `-f html` | Styled for browser viewing |
| CSV | `-f csv` | Tabular, for spreadsheets |
| Obsidian | `-f obsidian` | Markdown with YAML frontmatter and `[[wikilinks]]` |
| Org | `-f org` | Emacs org-mode format |
| YAML | `-f yaml` | Structured, human-readable |
| Plaintext | `-f plaintext` | Stripped of all markup |

Output can be sent to stdout (default), `--output browser`, or `--output clipboard`.

## Pipeline

```bash
polylogue run                          # Full pipeline: acquire → parse → render → index
polylogue run --source claude          # Single source
polylogue run --preview                # Preview counts, confirm before writing
polylogue run --stage parse            # Single stage only

# Watch mode — continuous sync
polylogue run --watch                  # Watch sources for changes
polylogue run --watch --notify         # Desktop notifications on new conversations
polylogue run --watch --webhook URL    # Webhook on new conversations
```

**Stages**: `acquire` → `parse` → `render` → `index`

The pipeline is idempotent — re-running imports is always safe. Content hashing (SHA-256 + NFC normalization) ensures unchanged conversations are skipped.

## Subcommands

```bash
polylogue check                        # Health check: DB integrity, index status, stats
polylogue check --repair               # Auto-fix issues (orphaned refs, stale FTS entries)
polylogue check --deep                 # Full SQLite integrity check

polylogue embed                        # Generate vector embeddings for semantic search
polylogue embed --stats                # Show embedding coverage
polylogue embed --model voyage-4-large # Use larger model

polylogue tags                         # List all tags with counts
polylogue tags -p claude --json        # Tags for a provider, as JSON

polylogue site -o ./public             # Build static HTML archive
polylogue site --title "My Archive"    # Custom title
polylogue site --search-provider lunr  # Client-side search engine

polylogue dashboard                    # Interactive Textual TUI

polylogue auth                         # Google OAuth flow (for Gemini/Drive)
polylogue auth --revoke                # Revoke stored credentials

polylogue reset --database             # Delete SQLite database
polylogue reset --all                  # Reset everything

polylogue completions --shell fish     # Generate shell completions
polylogue mcp                          # Start MCP server (stdio)
```

## MCP Integration

Polylogue provides a [Model Context Protocol](https://modelcontextprotocol.io/) server, giving AI assistants direct access to your conversation archive.

**Claude Code** (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "polylogue": {
      "command": "polylogue",
      "args": ["mcp"]
    }
  }
}
```

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`): same format.

| Capability | Available |
|------------|-----------|
| **Tools** | `search`, `list_conversations`, `get_conversation`, `stats` |
| **Resources** | `polylogue://stats`, `polylogue://conversations`, `polylogue://conversation/{id}` |
| **Prompts** | `analyze-errors`, `summarize-week`, `extract-code` |

Resources support query parameters for filtering: `polylogue://conversations?provider=claude&since=2024-01-01&limit=50`

[Full MCP documentation →](docs/mcp-integration.md)

## Library API

Polylogue is library-first — the CLI is a thin wrapper around the Python API.

```python
from polylogue import Polylogue

async with Polylogue() as archive:
    # Archive-wide stats
    stats = await archive.stats()

    # Search with composable filters
    convs = await (archive.filter()
        .provider("claude")
        .contains("error handling")
        .since("2024-06-01")
        .limit(10)
        .list())

    # Retrieve a single conversation
    conv = await archive.get("chatgpt:abc123")

    # Message-level projections
    for msg in conv.project().substantive().min_words(50).iter():
        print(f"{msg.role}: {msg.text[:80]}...")

    # Semantic search (requires embeddings)
    similar = await archive.filter().similar("debugging memory leaks").limit(5).list()

    # Metadata
    await archive.set_metadata("chatgpt:abc123", title="Auth Bug Investigation")
    await archive.add_tags("chatgpt:abc123", ["important", "project:backend"])
```

**Filter chain methods** (all chainable):

| Method | Purpose |
|--------|---------|
| `.contains(text)` / `.exclude_text(text)` | FTS search |
| `.provider(*names)` / `.exclude_provider(*names)` | Filter by provider |
| `.tag(*tags)` / `.exclude_tag(*tags)` | Filter by tag |
| `.has(*types)` | Content type: `thinking`, `tools`, `summary`, `attachments` |
| `.since(date)` / `.until(date)` | Date range (strings or datetime) |
| `.title(pattern)` / `.id(prefix)` | Text matching |
| `.similar(text)` | Semantic similarity (vector search) |
| `.sort(field)` | `date`, `tokens`, `messages`, `words`, `longest`, `random` |
| `.reverse()` / `.limit(n)` / `.sample(n)` | Order and limit |

**Terminal methods** (async): `.list()`, `.first()`, `.count()`, `.delete()`

[Full library API documentation →](docs/library-api.md)

## Demo & Showcase

Polylogue includes a complete demo system for exploring features without real data.

### Seed Mode

Create a full demo environment — synthetic database with realistic conversations from all providers:

```bash
# Interactive — prints env vars and instructions
polylogue demo --seed

# Shell integration — eval sets env vars in current shell
eval $(polylogue demo --seed --env-only)

# Customize
polylogue demo --seed -p chatgpt,claude -n 10
```

The seeded environment runs through the real pipeline (`acquire → parse → render → index`), so the demo exercises the exact same code paths as production.

### Corpus Mode

Write raw provider-format files (JSON, JSONL) to disk for inspection:

```bash
polylogue demo --corpus                       # All providers, 3 each
polylogue demo --corpus -p chatgpt -n 5       # ChatGPT only, 5 files
polylogue demo --corpus -o /tmp/corpus        # Custom output directory
```

Useful for inspecting wire formats, testing parser changes, or generating fixture data.

### Showcase Mode

Exercise the entire CLI surface area (58 exercises across 7 groups) and generate a verification report:

```bash
polylogue demo --showcase                     # Full validation
polylogue demo --showcase --live              # Read-only against real data
polylogue demo --showcase --json              # Machine-readable report
polylogue demo --showcase --verbose           # Print each exercise output
```

The showcase seeds a workspace, runs every query mode, output format, filter combination, and mutation — then produces a summary report, JSON results, and a markdown cookbook of all commands with output.

[Full demo documentation →](docs/demo.md)

## Configuration

**Zero-config by default.** Polylogue follows the [XDG Base Directory](https://specifications.freedesktop.org/basedir-spec/latest/) specification:

| Path | Purpose |
|------|---------|
| `~/.local/share/polylogue/polylogue.db` | SQLite database |
| `~/.local/share/polylogue/inbox/` | Drop exports here |
| `~/.local/share/polylogue/render/` | Rendered output |
| `~/.config/polylogue/` | OAuth credentials |

**Environment variables:**

| Variable | Purpose |
|----------|---------|
| `POLYLOGUE_ARCHIVE_ROOT` | Custom database location |
| `POLYLOGUE_RENDER_ROOT` | Custom render output |
| `VOYAGE_API_KEY` | Voyage AI key for semantic search |
| `POLYLOGUE_FORCE_PLAIN` | Force non-interactive output |
| `POLYLOGUE_LOG` | Log level: `error`, `warn`, `info`, `debug` |

[Full configuration documentation →](docs/configuration.md)

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli-reference.md) | Complete command reference with tips and examples |
| [Library API](docs/library-api.md) | Python API — filter chains, projections, async patterns |
| [Data Model](docs/data-model.md) | Conversation / Message / Attachment schemas |
| [Configuration](docs/configuration.md) | XDG paths, environment variables, observability |
| [Architecture](docs/architecture.md) | System design, layers, data flow, thread safety |
| [MCP Integration](docs/mcp-integration.md) | Model Context Protocol server for Claude Desktop/Code |
| [Demo & Showcase](docs/demo.md) | Demo command, synthetic data, surface-area validation |
| [Providers](docs/providers/) | Provider formats, detection, session integration |
| [Internals](docs/internals.md) | Developer reference — invariants, schemas, debugging |

## Development

```bash
git clone https://github.com/sinity/polylogue && cd polylogue

# Enter dev environment
nix develop              # Nix (recommended)
# or
uv sync                  # uv

# Run tests
pytest -q                # Quick run (4200+ tests)
pytest --cov=polylogue   # With coverage (90% minimum enforced)

# Lint & type check
ruff check polylogue/ tests/
mypy polylogue/
```

See [CLAUDE.md](CLAUDE.md) for development guidelines, [docs/internals.md](docs/internals.md) for implementation details, and [demos/](demos/) for screencast generation.

## License

[MIT](LICENSE)

---

<p align="center">
  <strong>Project Status</strong>: Active development (v0.1.0)<br>
  <a href="https://github.com/sinity/polylogue/issues">Issues</a> · <a href="https://github.com/sinity/polylogue">Source</a>
</p>
