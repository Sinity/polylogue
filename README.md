# Polylogue

> Preserve, index, and expose your AI conversation history as a queryable, programmable archive.

## Overview

Polylogue is a local-first tool for archiving AI conversations from multiple providers (ChatGPT, Claude, Claude Code, Gemini) into a unified, searchable database.

### Core Principles

| Principle | What It Means |
|-----------|---------------|
| **Preserve** | Conversations don't disappear when providers change. Durable local storage in provider-agnostic format. |
| **Index** | Find any conversation by content, not just title. Sub-second full-text search across all providers. |
| **Expose** | Other tools can build on the archive. Clean library API, CLI, and MCP server. |

### Design Philosophy

- **Zero-config by default**: Just drop exports in `~/.local/share/polylogue/inbox/` and run `polylogue run`
- **Library-first**: The Python API (`polylogue.lib`) is primary; CLI is a thin wrapper
- **Filter-chaining**: Query with composable filters, not memorized subcommands
- **Local-first**: All data stays on your machine. Sync TO local, never upload.

## Installation

```bash
# With uv (recommended)
uv tool install polylogue

# With pip
pip install polylogue

# From source
git clone https://github.com/sinity/polylogue
cd polylogue
uv sync
```

## Quick Start

### 1. Export Your Conversations

- **ChatGPT**: Settings → Data Controls → Export → Download `conversations.json`
- **Claude**: Download from claude.ai conversation history
- **Claude Code**: Exports are in `~/.claude/projects/`

### 2. Drop in Inbox

```bash
# Create inbox (auto-created on first sync if missing)
mkdir -p ~/.local/share/polylogue/inbox

# Copy or symlink your exports
cp ~/Downloads/conversations.json ~/.local/share/polylogue/inbox/chatgpt/
ln -s ~/.claude/projects ~/.local/share/polylogue/inbox/claude-code
```

### 3. Run

```bash
polylogue run
```

### 4. Search

```bash
polylogue "error handling"
polylogue "auth" -p claude --since "last week"
polylogue --latest --output browser
```

That's it. No config file needed.

## CLI Reference

### Invocation Modes

```bash
polylogue [QUERY...] [FILTERS...] [OUTPUT...]    # Query mode (default)
polylogue run [OPTIONS...]                        # Run pipeline (ingest → render → index)
polylogue embed [OPTIONS...]                      # Generate vector embeddings
polylogue tags [OPTIONS...]                       # List tags with counts
polylogue site [OPTIONS...]                       # Build static HTML archive
polylogue sources [OPTIONS...]                    # List configured sources
polylogue dashboard                               # Launch TUI dashboard
polylogue mcp                                     # MCP server mode
polylogue check                                   # Health check
polylogue auth                                    # OAuth flow (Drive)
polylogue reset                                   # Reset database/state
polylogue completions --shell SHELL               # Generate shell completions
```

### Query Mode

Query mode is the default. Running `polylogue` without arguments shows archive statistics.

#### Query Syntax

```bash
polylogue "error"                    # FTS search (smartcase: lowercase=insensitive)
polylogue "error" "python"           # AND: both terms required
polylogue "Error"                    # Case-sensitive (has uppercase)
```

Positional arguments are implicit `--contains` (FTS). Multiple positional args are ANDed.

#### Filters

| Flag | Short | Description |
|------|-------|-------------|
| `--contains TEXT` | `-c` | FTS term (repeatable = AND) |
| `--exclude-text TEXT` | | Exclude FTS term (repeatable) |
| `--provider NAME,...` | `-p` | Include providers (comma = OR) |
| `--exclude-provider NAME,...` | | Exclude providers |
| `--tag TAG,...` | `-t` | Include tags (comma = OR, supports `key:value`) |
| `--exclude-tag TAG,...` | | Exclude tags |
| `--title TEXT` | | Title contains |
| `--has TYPE,...` | | Has: `thinking`, `tools`, `summary`, `attachments` |
| `--since DATE` | | After date (`today`, `yesterday`, `"last week"`, `2025-01-01`) |
| `--until DATE` | | Before date |
| `--id PREFIX` | `-i` | ID prefix match |
| `--limit N` | `-n` | Max results |
| `--latest` | | Most recent (= `--sort date --limit 1`) |
| `--sort FIELD` | | Sort by: `date` (default), `tokens`, `messages`, `words`, `longest`, `random` |
| `--reverse` | | Reverse sort order |
| `--sample N` | | Random sample of N conversations |

**Comma = OR** for structured fields (provider, tag). Repeated flags = OR for same field, AND across fields.

#### Output

| Flag | Short | Description |
|------|-------|-------------|
| `--output DEST,...` | `-o` | Output destinations: `browser`, `clipboard`, `stdout` (default: `stdout`) |
| `--format FMT` | `-f` | Format: `markdown` (default), `json`, `html`, `obsidian`, `org`, `yaml`, `plaintext`, `csv` |
| `--fields FIELD,...` | | Select fields for list/json: `id`, `title`, `provider`, `date`, `messages`, `words`, `tags`, `summary` |
| `--list` | | Force list format (even for single result) |
| `--stats` | | Only statistics, no content |
| `--count` | | Print matched count and exit |
| `--stats-by DIM` | | Aggregate statistics by dimension: `provider`, `month`, `year`, `day` |
| `--open` | | Open result in browser/editor |
| `--transform XFORM` | | Transform output: `strip-tools`, `strip-thinking`, `strip-all` |
| `--stream` | | Stream output (low memory, requires `--latest` or `-i ID`) |
| `--dialogue-only` | `-d` | Show only user/assistant messages |

**Smart defaults**:

- No query → show stats
- Single result → show content
- Multiple results → show list
- `--output browser` → always HTML
- Content to non-stdout → stats printed to stdout

**Multiple outputs**: `--output browser,clipboard` performs both actions. Content is rendered once, sent to multiple destinations.

**Clipboard behavior**:

- Single conversation: Full markdown content copied
- Multiple conversations (with `--list` or when query returns many): Each conversation separated by `---` delimiter
- Format respects `--format` flag (markdown default, or json)

**`--stats` output** (for filtered results):

```
Matched: 12 conversations

Messages: 847 total (234 user, 421 assistant)
Words: 45,231
Thinking: 89 traces
Tool use: 156 calls
Attachments: 23
Date range: 2025-01-18 to 2025-01-24
```

**`--stats-by` output** (replaces individual `--by-*` flags):

```bash
polylogue --stats-by month                        # Activity histogram by month
polylogue -p claude --stats-by provider           # Provider breakdown for Claude
polylogue --stats-by year                         # Year-by-year overview
polylogue --since 2025-01 --stats-by day          # Daily breakdown
```

**`--stream` mode**: Streams messages to stdout one at a time for constant memory usage on large conversations. Supports `--dialogue-only` to filter to user/assistant messages. Output format is controlled via `--format` (plaintext, markdown, or json for JSON Lines).

**`--transform` options**:

| Transform | Effect |
|-----------|--------|
| `strip-tools` | Remove tool call/result messages |
| `strip-thinking` | Remove thinking/reasoning traces |
| `strip-all` | Remove both tools and thinking |

#### Modifiers (Write Operations)

```bash
# Metadata (unified k:v storage)
polylogue -i abc123 --set title "My Custom Title"
polylogue -i abc123 --set summary "Brief description..."
polylogue -i abc123 --set priority high            # Custom metadata key
polylogue -i abc123 --add-tag important,project:foo
polylogue -i abc123 --delete                       # Remove from archive

# Bulk operations with safety
polylogue "urgent" --add-tag review --dry-run      # Preview changes
polylogue -p old --delete --dry-run                # Preview deletions
polylogue -p old --delete --force                  # Skip confirmation
```

**Metadata**: Title, summary, and tags are stored as unified k:v metadata. Custom keys are allowed. Access via `--fields` or filter with `--has`.

**`--delete` safety**: Requires at least one filter flag (`-i`, `-p`, `-t`, `--since`, etc.). Cannot delete entire archive without explicit filter.

**`--dry-run`**: Shows what would be changed without executing. Works with `--add-tag`, `--set`, and `--delete`.

**`--force`**: Skips confirmation prompts for bulk operations (more than 10 conversations).

**List output format**:

```
  ID (24 chars)             DATE        [PROVIDER    ]  TITLE (MSG COUNT)
  claude:a8f2c3d4e5f6...    2025-01-24  [claude-code ]  Debugging OAuth (42 msgs)
  chatgpt:b9d8e7f6a5...     2025-01-23  [chatgpt     ]  Python patterns (18 msgs)
```

**ID prefix matching**: If prefix is ambiguous (matches multiple), error with list of matches. Use longer prefix to disambiguate.

### Run Mode

```bash
polylogue run                             # Run pipeline on all sources
polylogue run --source claude             # Run only for claude source
polylogue run --preview                   # Preview counts, confirm before writing
polylogue run --stage parse               # Run only parse stage
polylogue run --stage all                 # Run all stages (default)
polylogue run --format markdown           # Render as Markdown (default: html)
polylogue run --watch                     # Watch sources and sync continuously
polylogue run --watch --notify            # Desktop notification on new conversations
polylogue run --watch --exec "echo new"   # Execute command on new conversations
polylogue run --watch --webhook URL       # Call webhook on new conversations
```

**Pipeline stages**: `acquire` → `parse` → `render` → `index` → `generate-schemas`. Default runs all stages. Use `--stage` to run specific stages.

**Source scoping**: Use `--source NAME` (repeatable) to process only specific sources. Use `polylogue sources` to list available sources.

**Deduplication**: Conversations are identified by content hash (SHA-256 of normalized content). Re-importing the same conversation is a no-op. Modified conversations (same provider ID, different content) update the existing record.

**Partial failures**: Pipeline continues on individual file failures, reports errors at end. Exit code 0 if any files succeeded, non-zero only if all failed.

### Embed Mode

```bash
polylogue embed                           # Embed all unembedded conversations
polylogue embed -c <id>                   # Embed specific conversation
polylogue embed --model voyage-4-large    # Use larger model
polylogue embed --rebuild                 # Re-embed everything
polylogue embed --stats                   # Show embedding statistics
polylogue embed -n 50                     # Limit to 50 conversations
```

Generates vector embeddings using Voyage AI, stored in sqlite-vec for semantic search. Requires `VOYAGE_API_KEY` environment variable.

### Tags

```bash
polylogue tags                            # List all tags with counts
polylogue tags -p claude                  # Tags for Claude conversations only
polylogue tags --json                     # Machine-readable output
polylogue tags -n 10                      # Top 10 tags
```

### Site

```bash
polylogue site                            # Build static HTML site
polylogue site -o ./public                # Build to custom directory
polylogue site --title "My Archive"       # Custom site title
polylogue site --no-search                # Disable search index
polylogue site --search-provider lunr     # Use lunr.js instead of pagefind
polylogue site --no-dashboard             # Skip dashboard page
```

Generates a browsable static HTML site with index pages, per-provider views, dashboard with statistics, and client-side search.

### Sources

```bash
polylogue sources                         # List configured sources
polylogue sources --json                  # JSON output
```

### Dashboard

```bash
polylogue dashboard                       # Launch TUI dashboard
```

Opens the Textual-based TUI (Mission Control) for interactive browsing.

### Other Modes

```bash
polylogue mcp                             # Start MCP server (stdio)
polylogue check                           # Health check (DB, index, stats)
polylogue check --verbose                 # Show breakdown by provider
polylogue check --repair                  # Fix issues that can be auto-fixed
polylogue check --repair --preview        # Preview what would be repaired
polylogue check --repair --vacuum         # Compact database after repair
polylogue check --deep                    # Run SQLite integrity check
polylogue check --json                    # Machine-readable output
polylogue auth                            # OAuth flow for Google Drive
polylogue auth --refresh                  # Force token refresh
polylogue auth --revoke                   # Revoke stored credentials
polylogue reset --database                # Delete SQLite database
polylogue reset --render                  # Delete rendered outputs
polylogue reset --cache                   # Delete search indexes
polylogue reset --auth                    # Delete OAuth tokens
polylogue reset --all                     # Reset everything
polylogue reset --all --yes               # Non-interactive reset
```

### Global Flags

```bash
polylogue --version                       # Version
polylogue --plain                         # Force non-interactive plain output
polylogue -v                              # Verbose output
polylogue -h / --help                     # Help
```

### Shell Completions

Generate and install completions:

```bash
# Fish
polylogue completions --shell fish > ~/.config/fish/completions/polylogue.fish

# Zsh
polylogue completions --shell zsh > ~/.zfunc/_polylogue

# Bash
polylogue completions --shell bash > /etc/bash_completion.d/polylogue
```

### Technical Details

**FTS (Full-Text Search)**:

- SQLite FTS5 with default tokenizer
- Smartcase: all-lowercase query → case-insensitive; contains uppercase → case-sensitive
- Supports phrase queries with quotes: `"exact phrase"`

**Date parsing**: Uses `dateparser` library. Supports:

- ISO format: `2025-01-15`, `2025-01-15T10:30:00`
- Relative: `today`, `yesterday`, `"last week"`, `"2 days ago"`, `"last month"`
- Natural language: `"January 15"`, `"Jan 2025"`

**Exit codes**:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (invalid args, config error) |
| 2 | No results found (for queries) |
| 3 | Partial failure (some items failed in sync) |

**Terminal output**:

- Colors enabled by default on TTY, respects `NO_COLOR` env var
- Rich-formatted output (tables, panels) when interactive; plain text when piped
- Set `POLYLOGUE_FORCE_PLAIN=1` to force plain output, or use `--plain`

**Logging**: Set `POLYLOGUE_LOG=debug` for verbose logging to stderr. Levels: `error`, `warn`, `info`, `debug`.

### Examples

```bash
# Statistics
polylogue

# Search
polylogue "OAuth bug"
polylogue "error" "python" -p claude,chatgpt

# Filter and output
polylogue -p claude --has thinking --output browser
polylogue --latest --output browser,clipboard

# Sorting and sampling
polylogue --sort tokens --reverse --limit 10     # Longest conversations
polylogue --sort random --limit 5                # Random 5
polylogue --sample 10                            # Random sample of 10

# Field selection
polylogue -p claude --fields id,title,tokens --format json

# Aggregation
polylogue --stats-by month                       # Activity by month
polylogue -p claude --stats-by provider          # Provider breakdown

# Exclusions
polylogue "error" --exclude-text "warning" --exclude-provider gemini
polylogue -t important --exclude-tag archived

# Transforms
polylogue --latest --transform strip-tools       # Clean output without tool calls
polylogue -i abc123 --dialogue-only              # Just the conversation

# Streaming (memory-efficient for large conversations)
polylogue --latest --stream                      # Stream most recent
polylogue -i abc123 --stream -d                  # Stream dialogue only

# Count
polylogue -p claude --count                      # Quick count

# Metadata
polylogue -i abc123 --set title "The OAuth Fix"
polylogue -i abc123 --set summary "Fixed OAuth by..."
polylogue -i abc123 --add-tag project:polylogue,important
polylogue --tag project:polylogue --list

# Bulk operations with safety
polylogue -p claude --since "last month" --add-tag review --dry-run
polylogue -p old --delete --dry-run

# Run pipeline
polylogue run
polylogue run --source claude
polylogue run --preview
polylogue run --watch --notify

# Embeddings
polylogue embed
polylogue embed --stats

# Static site
polylogue site -o ./public

# Maintenance
polylogue check --repair --vacuum
```

## Library API

Polylogue is designed library-first. The CLI wraps the Python API.

### Basic Usage

```python
from polylogue.lib.filters import ConversationFilter
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.backends.sqlite import SQLiteBackend

backend = SQLiteBackend()
repo = ConversationRepository(backend=backend)

# Search
results = ConversationFilter(repo).contains("error").provider("claude").list()
for conv in results:
    print(f"{conv.id}: {conv.display_title}")

# Single conversation
conv = ConversationFilter(repo).id("abc123").first()
```

### Filter Chain API

```python
# Chainable, lazy evaluation
results = (ConversationFilter(repo)
    .contains("error")
    .contains("python")             # AND
    .provider("claude", "chatgpt")  # OR
    .since("2025-01-01")
    .has("thinking")
    .limit(10)
    .list())                        # Terminal: list(), first(), count(), delete()

# Exclusion filters
results = (ConversationFilter(repo)
    .contains("error")
    .no_contains("warning")
    .no_provider("gemini")
    .no_tag("archived")
    .list())

# Lightweight summaries (no message loading)
summaries = (ConversationFilter(repo)
    .provider("claude")
    .since("2025-01-01")
    .list_summaries())              # Returns ConversationSummary (no messages)

# Check if summaries are sufficient
f = ConversationFilter(repo).provider("claude")
if f.can_use_summaries():
    results = f.list_summaries()    # Fast path
else:
    results = f.list()              # Loads full conversations

# Custom predicates
results = (ConversationFilter(repo)
    .where(lambda c: len(c.messages) > 50)
    .list())

# Sorting and sampling
results = (ConversationFilter(repo)
    .sort("tokens")
    .reverse()
    .sample(10)
    .list())

# Conversation structure filters
roots = ConversationFilter(repo).is_root().list()
continuations = ConversationFilter(repo).is_continuation().list()
```

### Available Filter Methods

| Method | Description |
|--------|-------------|
| `.contains(text)` | FTS term (chainable = AND) |
| `.no_contains(text)` | Exclude FTS term |
| `.provider(*names)` | Include providers |
| `.no_provider(*names)` | Exclude providers |
| `.tag(*tags)` | Include tags |
| `.no_tag(*tags)` | Exclude tags |
| `.has(*types)` | Content types: `thinking`, `tools`, `summary`, `attachments` |
| `.title(pattern)` | Title contains pattern |
| `.id(prefix)` | ID prefix match |
| `.since(date)` | After date (str or datetime) |
| `.until(date)` | Before date (str or datetime) |
| `.similar(text)` | Semantic similarity (requires vector index) |
| `.sort(field)` | Sort: `date`, `tokens`, `messages`, `words`, `longest`, `random` |
| `.reverse()` | Reverse sort order |
| `.limit(n)` | Max results |
| `.sample(n)` | Random sample |
| `.where(predicate)` | Custom filter predicate |
| `.is_root()` | Root conversations only |
| `.is_continuation()` | Continuation conversations only |
| `.is_sidechain()` | Sidechain conversations only |
| `.has_branches()` | Conversations with branching messages |
| `.parent(id)` | Children of a given parent |

### Terminal Methods

| Method | Description |
|--------|-------------|
| `.list()` | Execute and return `list[Conversation]` |
| `.list_summaries()` | Execute and return `list[ConversationSummary]` (lightweight, no messages) |
| `.first()` | Execute and return first match or `None` |
| `.count()` | Execute and return count (uses SQL fast path when possible) |
| `.delete()` | Delete matching conversations (returns count deleted) |
| `.pick()` | Interactive picker (TTY) or first match (non-TTY) |
| `.can_use_summaries()` | Check if `list_summaries()` is valid for current filters |

### Pipeline (Run)

```python
from polylogue.pipeline.runner import run_sources
from polylogue.services import get_service_config

config = get_service_config()
result = run_sources(config=config, stage="all")
```

## Data Model

### Conversation

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique ID (`provider:provider_id`) |
| `provider` | str | Source provider (`chatgpt`, `claude`, `claude-code`, `gemini`) |
| `original_title` | str? | Provider's original title |
| `created_at` | datetime? | Creation timestamp |
| `updated_at` | datetime? | Last update timestamp |
| `content_hash` | str | SHA-256 for deduplication |
| `metadata` | dict | User metadata (k:v, see below) |

**Metadata** (unified k:v storage):

| Key | Type | Description |
|-----|------|-------------|
| `title` | str | User-set title (overrides original) |
| `summary` | str | User or LLM-generated summary |
| `tags` | list[str] | Tags (`important`, `project:foo`) |
| (custom) | str | Any user-defined key |

Display title precedence: `metadata.title` > `original_title` > truncated ID.

### Message

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Message ID |
| `role` | str | `user`, `assistant`, `system`, `tool` |
| `text` | str? | Message content |
| `timestamp` | datetime? | Message timestamp |
| `parent_id` | str? | Parent message (for branching) |
| `provider_meta` | dict? | Provider-specific data (content_blocks, cost, duration, etc.) |
| `attachments` | list | File attachments |

### Attachments

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Attachment ID |
| `name` | str? | Filename |
| `mime_type` | str? | MIME type |
| `size_bytes` | int? | File size |
| `path` | str? | Local path (if downloaded) |

Attachments are stored as references. For Drive sources, attachments can be downloaded on demand.

### Branching

Conversations may have branching structure (e.g., ChatGPT "edit and regenerate"). The `parent_id` field links messages to their parent, forming a tree.

**Current behavior**: Messages are flattened to a list in creation order. Branch structure is preserved in `provider_meta` for future use.

### Provider-Specific Metadata

Some providers include additional metadata:

**Claude Code**:

- `cost_usd`: API cost in USD
- `duration_ms`: Response generation time
- `model`: Model used (e.g., `claude-3-opus`)

Access via `message.provider_meta` or convenience properties:

```python
msg.cost_usd      # float or None
msg.duration_ms   # int or None
conv.total_cost_usd    # Sum of all message costs
conv.total_duration_ms # Sum of all durations
```

### Semantic Classification

Messages have classification properties derived from content and metadata:

| Property | Meaning |
|----------|---------|
| `is_user` | From user |
| `is_assistant` | From assistant |
| `is_system` | System prompt |
| `is_tool_use` | Tool call or result |
| `is_thinking` | Reasoning/thinking trace |
| `is_context_dump` | Pasted file content, context |
| `is_noise` | Tool use, context dump, or system |
| `is_substantive` | Real dialogue (not noise, not thinking) |

**Provider-specific detection**:

- **ChatGPT**: Thinking detected via `content_type: "thoughts"` or `"reasoning_recap"` in metadata
- **Claude Code**: Thinking via `content_blocks` with `type: "thinking"`; tool use via `type: "tool_use"` or `"tool_result"`
- **Gemini**: Thinking via `isThought` marker in chunk metadata
- **Claude (web)**: No structured thinking (simple text messages)

### Tags

Tags support `key:value` notation for namespacing:

```
important              # Simple tag
project:polylogue      # Namespaced
status:wip             # Namespaced
```

## File Layout

Polylogue follows XDG Base Directory specification:

```
~/.local/share/polylogue/           # XDG_DATA_HOME/polylogue
├── polylogue.db                    # SQLite database
├── inbox/                          # Drop exports here (or symlink)
│   ├── chatgpt/                    # Organize by provider (optional)
│   │   └── conversations.json
│   └── claude/
│       └── export.jsonl
└── render/                         # Rendered output
    ├── html/
    │   └── claude/
    │       └── abc123.html
    └── md/
        └── claude/
            └── abc123.md

~/.claude/projects/                  # Auto-discovered: Claude Code sessions
~/.codex/sessions/                   # Auto-discovered: Codex sessions

~/.config/polylogue/                # XDG_CONFIG_HOME/polylogue
└── polylogue-credentials.json      # Google OAuth credentials (if using Drive)

~/.local/state/polylogue/           # XDG_STATE_HOME/polylogue
└── token.json                      # OAuth token cache
```

### Inbox Conventions

- Drop provider exports directly in `inbox/` or in subdirectories
- Subdirectory names are for organization only (provider auto-detected from content)
- Symlinks are followed
- Files are processed recursively
- Supported formats: `.json`, `.jsonl`, `.zip`

## Configuration

**No configuration file.** Polylogue is truly zero-config. Paths follow XDG Base Directory specification.

### Environment Overrides

Optional environment variables for vector search and API keys:

| Variable | Alternative | Description |
|----------|-------------|-------------|
| `POLYLOGUE_VOYAGE_API_KEY` | `VOYAGE_API_KEY` | Voyage AI API key for embeddings |
| `POLYLOGUE_FORCE_PLAIN` | | Force non-interactive plain output |
| `POLYLOGUE_LOG` | | Log level: `error`, `warn`, `info`, `debug` |
| `POLYLOGUE_CREDENTIAL_PATH` | | Path to OAuth client JSON |
| `POLYLOGUE_TOKEN_PATH` | | Path to OAuth token |

### Backup and Export

The database is a single SQLite file. To backup:

```bash
cp ~/.local/share/polylogue/polylogue.db ~/backups/polylogue-$(date +%Y%m%d).db
```

To export all conversations as JSON:

```bash
polylogue --format json > conversations.json

# Or with filters
polylogue -p claude --format json > claude-conversations.json
```

The inbox directory contains original exports and can be re-synced to rebuild the database.

### Google Drive Integration

For Gemini conversations via Google Drive:

1. Create OAuth credentials at [Google Cloud Console](https://console.cloud.google.com/)
2. Download to `~/.config/polylogue/polylogue-credentials.json`
3. Run `polylogue auth` to complete OAuth flow

The "Google AI Studio" folder is automatically synced (hardcoded).

## MCP Integration

Polylogue provides an MCP (Model Context Protocol) server for integration with Claude Desktop, Claude Code, and other MCP clients.

**Primary use case**: Claude Code's `/history` command can use polylogue to search past sessions semantically, rather than just grepping JSONL files.

### Starting the Server

```bash
polylogue mcp
```

Runs in stdio mode (standard for MCP). Logs to stderr.

### Claude Code Configuration

Add to `~/.claude/settings.json`:

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

### Claude Desktop Configuration

Add to `~/.config/claude/claude_desktop_config.json`:

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

### Available Tools

| Tool | Description |
|------|-------------|
| `search` | Search conversations by text query with optional provider/date filters |
| `list_conversations` | List recent conversations, optionally filtered |
| `get_conversation` | Get a single conversation by ID (supports prefix matching) |
| `stats` | Archive statistics: totals, provider breakdown, database size |

### Available Resources

| Resource | Description |
|----------|-------------|
| `polylogue://stats` | Archive statistics |
| `polylogue://conversations` | List all conversations (up to 1000) |
| `polylogue://conversation/{id}` | Single conversation content |

### Available Prompts

| Prompt | Description |
|--------|-------------|
| `analyze_errors` | Analyze error patterns and solutions across conversations |
| `summarize_week` | Summarize key insights from the past week |
| `extract_code` | Extract and organize code snippets by language |

## Architecture Notes

### Components

```
polylogue/
├── cli/               # CLI commands (run, check, mcp, auth, reset, embed, tags, site, etc.)
├── sources/           # Source detection, provider parsers, Drive integration
├── pipeline/          # Ingestion → rendering → indexing orchestration
├── storage/           # SQLite backend, async repository, FTS5/sqlite-vec search
├── schemas/           # Unified schema, provider models, schema inference
├── lib/               # Core domain models, filters, projections, hashing
├── rendering/         # Markdown/HTML output renderers
├── ui/                # Terminal UI (Rich-based plain + Textual TUI)
├── site/              # Static site generator
└── mcp/               # Model Context Protocol server
```

### Key Abstractions

- **SearchProvider**: Protocol for search (FTS5 local, sqlite-vec vector)
- **VectorProvider**: Protocol for embedding-based similarity search
- **ConversationFilter**: Fluent filter builder for composable queries
- **ConversationRepository**: Data access layer wrapping SQLite backend

### Schema and Migrations

Database schema version is stored in DB. On startup, polylogue checks version and runs migrations automatically if needed. Migrations are forward-only (no downgrade). Backup before major version upgrades.

### Data Flow

```
Provider Exports → Ingest → Normalize → Store (SQLite) → Index (FTS5)
                                                      ↓
                              CLI/API ← Query ← Filter Chain
                                 ↓
                              Render → Markdown/HTML/JSON
```

See `ARCHITECTURE.md` for detailed documentation.

## Supported Providers

| Provider | Format | Auto-detected By | Normalized Name |
|----------|--------|------------------|-----------------|
| ChatGPT | `conversations.json` | `mapping` field with message graph | `chatgpt` |
| Claude (web) | `.jsonl` | `chat_messages` array | `claude` |
| Claude Code | `.json` array | `parentUuid`/`sessionId` markers | `claude-code` |
| Gemini | Google Drive API | `chunkedPrompt.chunks` structure | `gemini` |

**Provider detection priority**: If file matches multiple patterns, first match wins in order: Claude Code → ChatGPT → Claude → Gemini.

**ZIP archives**: Polylogue extracts and processes `.zip` files recursively. Nested ZIPs are supported. ZIP bomb protection: max 100:1 compression ratio, max 500MB uncompressed.

**Encoding handling**: UTF-8 assumed. Fallback chain: UTF-8 → UTF-8-sig → UTF-16 → UTF-32 → UTF-8 with errors ignored. Null bytes stripped.

## Development

```bash
# Clone and setup
git clone https://github.com/sinity/polylogue
cd polylogue
uv sync

# Run tests
uv run pytest

# Type checking
uv run mypy polylogue/

# Linting
uv run ruff check polylogue/ tests/
```

## Roadmap

Features planned for future implementation:

| Feature | Description |
|---------|-------------|
| `--similar` CLI flag | Semantic similarity ranking via embeddings in query mode |
| LLM annotation | Batch LLM-generated titles, summaries, and tags (`--annotate`) |
| Interactive picker | `--pick` flag with `fzf` integration for query mode |
| Branch navigation | `--branch`, `--all-branches` for tree-structured conversations |
| Fork detection | Auto-detect forked/edited conversations |
| Watch webhooks | `polylogue run --watch --webhook` integration |
| DuckDB backend | Alternative storage backend |
| Tag implication rules | e.g., `project:* → has-project` |
| Progressive summarization | Handle very long conversations for annotation |

## License

MIT License. See `LICENSE` file.

---

**Project Status**: Active development (v0.1.0). Core query and sync functionality is the priority.

**Feedback**: Issues and PRs welcome at [github.com/sinity/polylogue](https://github.com/sinity/polylogue).
