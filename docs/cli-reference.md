[← Back to README](../README.md)

# CLI Reference

## Invocation Modes

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

## Query Mode

Query mode is the default. Running `polylogue` without arguments shows archive statistics.

### Query Syntax

```bash
polylogue "error"                    # FTS search (smartcase: lowercase=insensitive)
polylogue "error" "python"           # AND: both terms required
polylogue "Error"                    # Case-sensitive (has uppercase)
```

Positional arguments are implicit `--contains` (FTS). Multiple positional args are ANDed.

### Filters

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

### Output

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

### Modifiers (Write Operations)

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

## Run Mode

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

## Embed Mode

```bash
polylogue embed                           # Embed all unembedded conversations
polylogue embed -c <id>                   # Embed specific conversation
polylogue embed --model voyage-4-large    # Use larger model
polylogue embed --rebuild                 # Re-embed everything
polylogue embed --stats                   # Show embedding statistics
polylogue embed -n 50                     # Limit to 50 conversations
```

Generates vector embeddings using Voyage AI, stored in sqlite-vec for semantic search. Requires `VOYAGE_API_KEY` environment variable.

## Tags

```bash
polylogue tags                            # List all tags with counts
polylogue tags -p claude                  # Tags for Claude conversations only
polylogue tags --json                     # Machine-readable output
polylogue tags -n 10                      # Top 10 tags
```

## Site

```bash
polylogue site                            # Build static HTML site
polylogue site -o ./public                # Build to custom directory
polylogue site --title "My Archive"       # Custom site title
polylogue site --no-search                # Disable search index
polylogue site --search-provider lunr     # Use lunr.js instead of pagefind
polylogue site --no-dashboard             # Skip dashboard page
```

Generates a browsable static HTML site with index pages, per-provider views, dashboard with statistics, and client-side search.

## Sources

```bash
polylogue sources                         # List configured sources
polylogue sources --json                  # JSON output
```

## Dashboard

```bash
polylogue dashboard                       # Launch TUI dashboard
```

Opens the Textual-based TUI (Mission Control) for interactive browsing.

## Demo

Generate synthetic conversations for testing and exploration:

```bash
polylogue demo --seed                       # Seed demo DB, print env vars
polylogue demo --seed --env-only            # Shell-friendly (for eval)
polylogue demo --corpus                     # Write raw provider-format files
polylogue demo --corpus -p chatgpt -n 5     # ChatGPT only, 5 conversations
polylogue demo --corpus -o /tmp/corpus      # Custom output directory
```

**Two modes:**

- `--seed`: Creates a full demo environment (database + rendered files) and prints environment variables (`POLYLOGUE_ARCHIVE_ROOT`, etc.) to point your shell at the demo data. Use `eval $(polylogue demo --seed --env-only)` for seamless shell integration.
- `--corpus`: Writes raw provider-format files (JSON, JSONL) to disk for inspection. Useful for understanding wire formats or testing parsers.

**Options:**

| Flag | Short | Description |
|------|-------|-------------|
| `--seed` | | Create full demo environment |
| `--corpus` | | Generate raw fixture files |
| `--provider NAME` | `-p` | Providers to include (repeatable, default: all) |
| `--count N` | `-n` | Conversations per provider (default: 3) |
| `--output-dir PATH` | `-o` | Output directory |
| `--env-only` | | Print export statements only (for eval) |

## Other Modes

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

## Global Flags

```bash
polylogue --version                       # Version
polylogue --plain                         # Force non-interactive plain output
polylogue -v                              # Verbose output
polylogue -h / --help                     # Help
```

## Shell Completions

Generate and install completions:

```bash
# Fish
polylogue completions --shell fish > ~/.config/fish/completions/polylogue.fish

# Zsh
polylogue completions --shell zsh > ~/.zfunc/_polylogue

# Bash
polylogue completions --shell bash > /etc/bash_completion.d/polylogue
```

## Technical Details

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

## Examples

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

---

**See also:** [Library API](library-api.md) · [Data Model](data-model.md) · [Configuration](configuration.md) · [MCP Integration](mcp-integration.md)
