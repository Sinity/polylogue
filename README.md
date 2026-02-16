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
- **Library-first**: Clean Python API (sync + async) with composable filter chains — CLI is a thin wrapper
- **Local-first**: All data stays on your machine, always

## Feature Highlights

<table>
<tr>
<td width="50%">

### Ingest From Anywhere

```bash
polylogue run
```

Auto-detects provider format from file content. Handles JSON, JSONL, ZIP archives, and Google Drive sync.

<img src="demos/output/02-run.gif" alt="polylogue run" width="100%">

</td>
<td width="50%">

### Search Across Providers

```bash
polylogue "error handling" -p claude
polylogue --has thinking --since "last week"
```

Filter by provider, date, content type, tags — combine any filters.

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

### Try It First

```bash
polylogue demo --seed
```

Generate a synthetic archive with realistic conversations from all providers. Explore the UI, search, and API without importing your own data.

</td>
<td width="50%">

### Library API

```python
async with Polylogue() as archive:
    stats = await archive.stats()
    convs = await archive.filter().provider("claude").list()
```

Async-first API with concurrent queries and parallel batch operations.

</td>
</tr>
</table>

## Quick Start

### 1. Install

```bash
# With uv (recommended)
uv tool install polylogue

# With pip
pip install polylogue

# From source
git clone https://github.com/sinity/polylogue && cd polylogue && uv sync
```

### 2. Export Your Conversations

| Provider | How to Export |
|----------|--------------|
| **ChatGPT** | Settings → Data Controls → Export → Download `conversations.json` |
| **Claude** | Download from claude.ai conversation history |
| **Claude Code** | Auto-discovered from `~/.claude/projects/` |
| **Codex** | Auto-discovered from `~/.codex/sessions/` |
| **Gemini** | Via Google Drive sync (`polylogue auth` for OAuth setup) |

### 3. Drop in Inbox & Run

```bash
# Copy or symlink your exports
cp ~/Downloads/conversations.json ~/.local/share/polylogue/inbox/
ln -s ~/.claude/projects ~/.local/share/polylogue/inbox/claude-code

# Run the pipeline
polylogue run

# Search
polylogue "error handling"
polylogue "auth" -p claude --since "last week"
polylogue --latest --output browser
```

No config file needed. That's it.

### Try It First (No Data Needed)

```bash
# Generate a demo environment with synthetic conversations
eval $(polylogue demo --seed --env-only)

# Now explore — search, filter, dashboard all work
polylogue "error handling"
polylogue dashboard
```

## Supported Providers

| Provider | Format | Auto-detected By | Normalized Name |
|----------|--------|------------------|-----------------|
| ChatGPT | `conversations.json` | `mapping` field with message graph | `chatgpt` |
| Claude (web) | `.jsonl` | `chat_messages` array | `claude` |
| Claude Code | `.json` array | `parentUuid`/`sessionId` markers | `claude-code` |
| Codex | `.jsonl` | Session envelope structure | `codex` |
| Gemini | Google Drive API | `chunkedPrompt.chunks` structure | `gemini` |

ZIP archives are supported (nested ZIPs too, with bomb protection). Provider detection is automatic from file content.

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli-reference.md) | Full command reference: query syntax, filters, output formats, modifiers, all modes |
| [Library API](docs/library-api.md) | Python API: filter chains, projections, terminal methods |
| [Data Model](docs/data-model.md) | Conversation/Message/Attachment schemas, semantic classification |
| [Configuration](docs/configuration.md) | XDG paths, environment variables, Drive setup, backup |
| [MCP Integration](docs/mcp-integration.md) | Model Context Protocol server for Claude Desktop/Code |
| [Architecture](docs/architecture.md) | System design, component overview, data flow |
| [Demo & Synthetic Data](docs/demo.md) | Demo command, synthetic corpus, test fixtures |

## MCP Integration

Use polylogue as an MCP server for **Claude Desktop** or **Claude Code** — search your conversation archive from within Claude:

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

Tools: `search`, `list_conversations`, `get_conversation`, `stats`. [Full MCP docs →](docs/mcp-integration.md)

## Development

```bash
git clone https://github.com/sinity/polylogue && cd polylogue
nix develop   # or: uv sync
pytest        # run tests
```

See [CLAUDE.md](CLAUDE.md) for development guidelines and [demos/](demos/) for screencast generation.

## License

MIT License.

---

<p align="center">
  <strong>Project Status</strong>: Active development (v0.1.0)<br>
  <a href="https://github.com/sinity/polylogue/issues">Issues</a> · <a href="https://github.com/sinity/polylogue">Source</a>
</p>
