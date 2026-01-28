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

- **Zero-config by default**: Just drop exports in `~/.local/share/polylogue/inbox/` and run `polylogue sync`
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
git clone https://github.com/yourname/polylogue
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

### 3. Sync

```bash
polylogue sync
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
polylogue sync [OPTIONS...]                       # Sync mode
polylogue mcp                                     # MCP server mode
polylogue check                                   # Health check
polylogue auth                                    # OAuth flow (Drive)
polylogue reset                                   # Reset database
```

### Query Mode

Query mode is the default. Running `polylogue` without arguments shows archive statistics:

```
polylogue v0.5.0

Archive: 1,234 conversations (156 MB)
  claude-code:   512 (41%)  │  ████████████
  chatgpt:       398 (32%)  │  █████████
  claude:        247 (20%)  │  ██████
  gemini:         77 (6%)   │  ██

Messages: 45,231 total, 12.4M words
Tags: 23 unique, 156 tagged conversations
Last sync: 2 hours ago

Recent:
  claude-code:a8f2c   today       "Polylogue CLI redesign"
  chatgpt:b3d91       yesterday   "Python async patterns"
  claude:c7e43        2024-01-20  "Debugging OAuth flow"
```

#### Query Syntax

```bash
polylogue "error"                    # FTS search (smartcase: lowercase=insensitive)
polylogue "error" "python"           # AND: both terms required
polylogue "Error"                    # Case-sensitive (has uppercase)
polylogue --regex "err(or|ors)"      # Regex pattern
polylogue --similar "best practices" # Rank by semantic similarity (embeddings)
```

Positional arguments are implicit `--contains` (FTS). Multiple positional args are ANDed.

#### Filters

| Flag | Short | Description |
|------|-------|-------------|
| `--contains TEXT` | `-c` | FTS term (repeatable = AND) |
| `--no-contains TEXT` | `-C` | Exclude FTS term |
| `--regex PATTERN` | | Regex match |
| `--no-regex PATTERN` | | Exclude regex match |
| `--provider NAME,...` | `-p` | Include providers (comma = OR) |
| `--no-provider NAME,...` | `-P` | Exclude providers |
| `--tag TAG,...` | `-t` | Include tags (comma = OR, supports `key:value`) |
| `--no-tag TAG,...` | `-T` | Exclude tags |
| `--title TEXT` | | Title contains |
| `--has TYPE,...` | | Has: `thinking`, `tools`, `summary`, `comment`, `attachments` |
| `--no-has TYPE,...` | | Missing types |
| `--delete` | | Delete matched conversations (requires filter) |
| `--since DATE` | | After date (`today`, `yesterday`, `"last week"`, `2024-01-01`) |
| `--until DATE` | | Before date |
| `--id PREFIX` | `-i` | ID prefix match |
| `--limit N` | `-n` | Max results |
| `--latest` | | Most recent (= `--sort date --limit 1`) |
| `--sort FIELD` | | Sort by: `date` (default), `tokens`, `messages`, `words`, `longest`, `random` |
| `--reverse` | | Reverse sort order |
| `--sample N` | | Random sample of N conversations |
| `--similar TEXT` | | Rank by embedding similarity (mutually exclusive with `--sort`) |

**Negation pattern**: Uppercase short flag = negation (`-p` include, `-P` exclude).

**Comma = OR** for structured fields (provider, tag). Repeated flags = OR for same field, AND across fields.

#### Output

| Flag | Description |
|------|-------------|
| `--output DEST,...` | Output destinations: `browser`, `clipboard`, `stdout` (default: `stdout`) |
| `--format FMT` | Format: `markdown` (default), `json`, `html`, `obsidian`, `org` |
| `--fields FIELD,...` | Select fields for list/json: `id`, `title`, `provider`, `date`, `messages`, `tokens`, `tags`, `summary`, or any metadata key |
| `--list` | Force list format (even for single result) |
| `--stats` | Only statistics, no content (see below) |
| `--pick` | Interactive picker (uses `fzf` if available) |
| `--by-month` | Aggregate output: histogram by month |
| `--by-provider` | Aggregate output: count by provider |
| `--by-tag` | Aggregate output: count by tag |

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

**`--pick` behavior**: Uses `fzf` if available in PATH. Falls back to numbered selection prompt (enter number to select). With `--pick`, after selection the chosen conversation is processed according to other flags.

**`--stats` output** (for filtered results):

```
Query: "error" -p claude --since "last week"
Matched: 12 conversations

Messages: 847 total (234 user, 421 assistant, 192 other)
Words: 45,231 total (12,847 user, 31,204 assistant)
Thinking: 89 traces (2,341 words)
Tool use: 156 calls
Attachments: 23
Date range: 2024-01-18 to 2024-01-24
```

#### Modifiers (Write Operations)

```bash
# Metadata (unified k:v storage)
polylogue -i abc123 --set title "My Custom Title"
polylogue -i abc123 --set summary "Brief description..."
polylogue -i abc123 --set priority high            # Custom metadata key
polylogue -i abc123 --unset priority               # Remove metadata key
polylogue -i abc123 --add-tag important,project:foo
polylogue -i abc123 --rm-tag archived
polylogue -i abc123 --delete                       # Remove from archive

# LLM annotation (batch operation on filtered results)
polylogue -p claude --since "last month" --annotate "Focus on technical decisions. Suggest tags from: project:*, lang:*, topic:*"
```

**Metadata**: Title, summary, and tags are stored as unified k:v metadata. Custom keys are allowed. Access via `--fields` or filter with `--has`.

**`--annotate`**: LLM generates title, summary, and tags for all matched conversations. Shows cost estimate and asks for confirmation. For long conversations, uses progressive summarization. Provide a prompt to guide the LLM's focus and tag vocabulary.

**`--delete` safety**: Requires at least one filter flag (`-i`, `-p`, `-t`, `--since`, etc.). Cannot delete entire archive without explicit filter.

**List output format**:

```
  ID (24 chars)             DATE        [PROVIDER    ]  TITLE (MSG COUNT)
  claude:a8f2c3d4e5f6...    2024-01-24  [claude-code ]  Debugging OAuth (42 msgs)
  chatgpt:b9d8e7f6a5...     2024-01-23  [chatgpt     ]  Python patterns (18 msgs)
```

**ID prefix matching**: Minimum 4 characters. If prefix is ambiguous (matches multiple), error with list of matches. Use longer prefix or `--pick` to disambiguate.

### Sync Mode

```bash
polylogue sync                       # Sync all sources
polylogue sync -p claude             # Sync only from claude inbox subdirs
polylogue sync --watch               # Watch mode (continuous)
polylogue sync --clipboard           # Import from clipboard
polylogue sync --file path.json      # Import specific file
polylogue sync --force-render        # Re-render all existing conversations

# Watch mode enhancements
polylogue sync --watch --notify                    # Desktop notification on new
polylogue sync --watch --exec "echo {id} >> log"   # Run command on new
polylogue sync --watch --webhook https://...       # POST to webhook on new
```

**Watch mode**: Polls inbox every 5 seconds for new/modified files. Logs sync activity to stderr. Ctrl+C to stop. Does not watch Drive (use manual sync for Drive). The `--notify`, `--exec`, and `--webhook` flags trigger on each new conversation.

**Deduplication**: Conversations are identified by content hash (SHA-256 of normalized content). Re-importing the same conversation is a no-op. Modified conversations (same provider ID, different content) update the existing record.

**Partial failures**: Sync continues on individual file failures, reports errors at end. Exit code 0 if any files succeeded, non-zero only if all failed.

**Rendering**: Markdown and HTML are generated during sync for new/modified conversations. To force re-render:

```bash
polylogue sync --force-render        # Re-render all
polylogue sync --force-render -i abc # Re-render specific conversation
```

**Delete/prune**: To remove conversations from the archive:

```bash
polylogue -i abc123 --delete         # Delete specific conversation
polylogue -P gemini --delete         # Delete all Gemini conversations
polylogue --delete                   # ERROR: requires filter (safety)
```

Deletion removes from DB and deletes render files. Original inbox files are NOT deleted.

**Title display**: `user_title` (if set) > `original_title` (from provider) > truncated ID. Set user title with `--title`, clear with `--title ""`.

### Other Modes

```bash
polylogue mcp                        # Start MCP server (stdio)
polylogue check                      # Health check (DB, index, stats)
polylogue check --repair             # Fix issues that can be auto-fixed
polylogue check --vacuum             # Compact database, reclaim space
polylogue auth                       # OAuth flow for Google Drive
polylogue reset                      # Reset database (interactive confirmation)
polylogue reset --confirm            # Non-interactive reset
POLYLOGUE_FORCE=1 polylogue reset    # Automation escape hatch
```

**`polylogue check` output**:

```
Database: OK (156 MB, 1234 conversations)
FTS Index: OK (45231 messages indexed)
Inbox: OK (3 subdirs, 47 files)
Renders: OK (1198 HTML, 1198 MD)
Drive: Not configured

Issues: None
```

Checks performed:

- Database accessible and schema current
- FTS index exists and row count matches
- Inbox directory readable
- Render directory writable
- Drive credentials valid (if configured)
- Orphaned records (messages without conversations)
- Missing renders (conversations not yet rendered)

### Global Flags

```bash
polylogue --version                  # Version
polylogue --completions SHELL        # Generate completions (bash, zsh, fish)
polylogue --help                     # Help
```

### Shell Completions

Generate and install completions:

```bash
# Fish
polylogue --completions fish > ~/.config/fish/completions/polylogue.fish

# Zsh
polylogue --completions zsh > ~/.zfunc/_polylogue

# Bash
polylogue --completions bash > /etc/bash_completion.d/polylogue
```

**Dynamic completions**: Completions query the database for:

- `--provider` / `-p`: Available providers (from indexed conversations)
- `--id` / `-i`: Recent conversation IDs (sorted by recency, shows title hint)
- `--tag` / `-t`: Existing tags
- `--since` / `--until`: Date suggestions (`today`, `yesterday`, `"last week"`, ISO format)

### Technical Details

**FTS (Full-Text Search)**:

- SQLite FTS5 with default tokenizer
- Smartcase: all-lowercase query → case-insensitive; contains uppercase → case-sensitive
- Supports phrase queries with quotes: `"exact phrase"`

**Regex**: Python `re` module syntax. Patterns are matched against message text.

**Date parsing**: Uses `dateparser` library. Supports:

- ISO format: `2024-01-15`, `2024-01-15T10:30:00`
- Relative: `today`, `yesterday`, `"last week"`, `"2 days ago"`, `"last month"`
- Natural language: `"January 15"`, `"Jan 2024"`

**Exit codes**:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (invalid args, config error) |
| 2 | No results found (for queries) |
| 3 | Partial failure (some items failed in sync) |

**Terminal output**:

- Colors enabled by default on TTY, respects `NO_COLOR` env var
- Long output (>50 lines) paged via `$PAGER` (default: `less -R`)
- Use `--no-pager` to disable, or pipe to disable automatically

**Logging**: Set `POLYLOGUE_LOG=debug` for verbose logging to stderr. Levels: `error`, `warn`, `info`, `debug`.

### Examples

```bash
# Statistics
polylogue

# Search
polylogue "OAuth bug"
polylogue "error" "python" -p claude,chatgpt
polylogue --regex "async.*await" --since "2024-01-01"

# Filter and output
polylogue -p claude --has thinking --output browser
polylogue --latest --output browser,clipboard
polylogue "auth" --pick --output browser

# Sorting and sampling
polylogue --sort tokens --reverse --limit 10     # Longest conversations
polylogue --sort random --limit 5                # Random 5
polylogue --sample 10                            # Random sample of 10

# Field selection
polylogue -p claude --fields id,title,tokens --format json

# Aggregation
polylogue --by-month                             # Activity histogram
polylogue -p claude --by-tag                     # Tag distribution for Claude

# Exclusions
polylogue "error" -C "warning" -P gemini
polylogue -t important -T archived

# Metadata
polylogue -i abc123 --set title "The OAuth Fix"
polylogue -i abc123 --set summary "Fixed OAuth by..."
polylogue -i abc123 --add-tag project:polylogue,important
polylogue --tag project:polylogue --list

# LLM annotation
polylogue -p claude --since "last month" --annotate "Technical focus, suggest tags from: project:*, lang:*"

# Sync
polylogue sync
polylogue sync --watch --notify
polylogue sync --clipboard

# Maintenance
polylogue check --vacuum
```

## Library API

Polylogue is designed library-first. The CLI wraps the Python API.

### Basic Usage

```python
from polylogue import Polylogue

poly = Polylogue()  # Uses XDG defaults

# Statistics
stats = poly.stats()
print(f"Total: {stats.conversation_count} conversations")

# Search
results = poly.filter().contains("error").provider("claude").list()
for conv in results:
    print(f"{conv.id}: {conv.title}")

# Single conversation
conv = poly.filter().id("abc123").first()
print(conv.to_markdown())
```

### Filter Chain API

```python
# Chainable, lazy evaluation
results = (poly.filter()
    .contains("error")
    .contains("python")          # AND
    .provider("claude", "chatgpt")  # OR
    .since("2024-01-01")
    .has("thinking")
    .limit(10)
    .list())                     # Terminal: list(), first(), count()

# Negation
results = (poly.filter()
    .contains("error")
    .no_contains("warning")
    .no_provider("gemini")
    .list())

# Semantic ranking
results = (poly.filter()
    .similar("best practices for authentication")
    .limit(5)
    .list())
```

### Conversation Model

```python
conv = poly.filter().latest().first()

# Properties
conv.id                    # "claude:abc123"
conv.title                 # "Debugging OAuth"
conv.provider              # "claude"
conv.created_at            # datetime
conv.updated_at            # datetime
conv.message_count         # 42
conv.word_count            # 3847
conv.tags                  # ["important", "project:foo"]
conv.summary               # "Manual or LLM summary"

# Messages
for msg in conv.messages:
    print(f"[{msg.role}] {msg.text[:100]}...")

# Semantic classification
for msg in conv.messages:
    if msg.is_substantive:     # Real dialogue
        print(msg.text)
    if msg.is_thinking:        # Reasoning trace
        print(f"Thinking: {msg.text[:50]}...")
    if msg.is_tool_use:        # Tool call/result
        pass  # Skip

# Projections
clean = conv.substantive_only()    # Filter to substantive messages
pairs = list(conv.iter_pairs())    # User/assistant dialogue pairs
thinking = list(conv.iter_thinking())  # Thinking traces only

# Output
conv.to_markdown()         # Markdown string
conv.to_json()             # Dict
conv.open()                # Open in browser
conv.copy()                # Copy to clipboard
```

### Sync and Import

```python
# Sync all
poly.sync()

# Sync specific providers
poly.sync(providers=["claude"])

# Watch mode
poly.sync(watch=True)  # Blocking, continuous

# Import
poly.import_file(Path("~/Downloads/conversations.json"))
poly.import_clipboard()
```

### Custom Backends

```python
from polylogue import Polylogue
from polylogue.backends import SQLiteBackend, MemoryBackend

# Default (XDG path)
poly = Polylogue()

# Custom database path
poly = Polylogue(backend=SQLiteBackend("/custom/path/polylogue.db"))

# In-memory (testing)
poly = Polylogue(backend=MemoryBackend())
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

**Planned**: Branch-aware navigation and filtering (e.g., `--branch main`, `--all-branches`).

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

Future: Tag implication rules (e.g., `project:* → has-project`).

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
| `POLYLOGUE_QDRANT_URL` | `QDRANT_URL` | Qdrant server URL for `--similar` vector search |
| `POLYLOGUE_QDRANT_API_KEY` | `QDRANT_API_KEY` | Qdrant authentication token |
| `POLYLOGUE_VOYAGE_API_KEY` | `VOYAGE_API_KEY` | Voyage AI API key for embeddings |

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
| `polylogue_find` | Search conversations |
| `polylogue_get` | Get single conversation |
| `polylogue_list` | List conversations with filters |
| `polylogue_stats` | Archive statistics |

### Available Resources

| Resource | Description |
|----------|-------------|
| `polylogue://stats` | Archive statistics |
| `polylogue://conversations` | List all conversations |
| `polylogue://conversation/{id}` | Single conversation content |

## Architecture Notes

### Components

```
polylogue/
├── lib/               # Core library (models, repository, projections)
├── storage/           # Storage layer (SQLite, FTS5, backends)
├── ingestion/         # Import from providers (ChatGPT, Claude, etc.)
├── pipeline/          # Async ingestion pipeline (runner, services)
├── rendering/         # Output rendering (Markdown, HTML)
├── cli/               # CLI wrapper
├── mcp/               # MCP server
├── server/            # FastAPI backend server
├── ui/                # Terminal UI facade (Rich, Textual)
├── health.py          # System health checks
├── verify.py          # Data quality verification
└── analytics/         # Usage metrics and statistics
```

### Key Abstractions

- **StorageBackend**: Protocol for database operations (SQLite, Memory, future: DuckDB)
- **SearchProvider**: Protocol for search (FTS5, future: embeddings)
- **Polylogue**: Main entry point, wraps storage + search + sync

### Schema and Migrations

Database schema version is stored in DB. On startup, polylogue checks version and runs migrations automatically if needed. Migrations are forward-only (no downgrade). Backup before major version upgrades.

Schema changes are backward compatible within minor versions. A v0.5 database can be read by v0.5.x but may not work with v0.6.

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

**Stdin support**:

```bash
cat export.json | polylogue sync --file -
pbpaste | polylogue sync --clipboard   # macOS
xclip -o | polylogue sync --clipboard  # Linux (auto-detected)
```

## Development

```bash
# Clone and setup
git clone https://github.com/yourname/polylogue
cd polylogue
uv sync

# Run tests
uv run pytest

# Type checking
uv run mypy polylogue/

# Linting
uv run ruff check polylogue/ tests/
```

## Planned Features

Features marked for future implementation:

| Feature | Description | Status |
|---------|-------------|--------|
| `--similar` | Semantic similarity ranking via embeddings | Planned |
| `--annotate` | LLM-generated titles, summaries, and tags (batch) | Planned |
| Progressive summarization | Handle very long conversations | Planned |
| Message-level annotation | Descriptions for individual messages | Considered |
| Branch navigation | `--branch`, `--all-branches` for tree-structured conversations | Planned |
| Fork detection | Auto-detect forked/edited conversations | Considered |
| `--format obsidian` | Obsidian-compatible export | Planned |
| `--format org` | Org-mode export | Planned |
| Watch webhooks | `sync --watch --webhook` | Planned |
| DuckDB backend | Alternative storage backend | Considered |

## License

MIT License. See `LICENSE` file.

---

**Project Status**: Active development. Core query and sync functionality is the priority. Embedding and LLM features are planned for later.

**Feedback**: Issues and PRs welcome at [github.com/yourname/polylogue](https://github.com/yourname/polylogue).
