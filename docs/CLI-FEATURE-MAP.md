# Polylogue CLI Feature Map

Complete reference of all CLI commands, features, and workflows.

---

## 🎯 Core Workflows

### Workflow 1: First-Time Setup
```bash
polylogue config init           # Create config file interactively
polylogue sources               # List configured import sources
polylogue run --preview         # Preview what will be imported
polylogue run                   # Import conversations + build index
```

### Workflow 2: Search & Explore
```bash
polylogue search "python async"              # Full-text search
polylogue search "error" --since "last week" # Search recent conversations
polylogue view abc123                        # View specific conversation
polylogue view --provider claude --query "python"  # List filtered conversations
```

### Workflow 3: Analytics & Insights
```bash
polylogue analytics --provider-comparison    # Compare AI providers
polylogue view --projection stats            # Conversation statistics
polylogue verify --verbose                   # Data quality check
```

### Workflow 4: Ongoing Sync
```bash
polylogue run --source last        # Import only newest source
polylogue run --stage ingest       # Import without rendering/indexing
polylogue index                    # Rebuild search index only
```

---

## 📋 Command Reference

### Data Import & Processing

#### `polylogue run` - Main Pipeline
**Purpose**: Import conversations → Render outputs → Build search index

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--preview` | flag | - | Dry run (no writes) |
| `--stage` | choice | all | Run specific stage: ingest, render, index, all |
| `--source` | text | all | Limit to source (repeatable, use 'last' for newest) |
| `--format` | choice | html | Output format: markdown, html |
| `--config` | path | - | Override config file location |

**Examples**:
```bash
polylogue run                                  # Full pipeline (import + render + index)
polylogue run --preview                        # Preview without writing
polylogue run --source chatgpt-exports         # Import specific source only
polylogue run --source last                    # Import only the newest source
polylogue run --stage ingest                   # Import only (skip render/index)
polylogue run --format markdown                # Render as .md instead of .html
```

**Output**: Creates `.md`/`.html` files in render directory, updates SQLite database, builds FTS5 index.

---

#### `polylogue sources` - List Import Sources
**Purpose**: Show configured import sources and their status

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--json` | flag | - | Output as JSON |
| `--config` | path | - | Override config file |

**Output**:
```
Source: chatgpt-exports
  Path: ~/Downloads/conversations.json
  Provider: chatgpt
  Enabled: yes
  Last run: 2024-01-15 10:30:00
  Conversations: 142
```

---

### Search & Discovery

#### `polylogue search` - Full-Text Search
**Purpose**: Search conversations using FTS5 (keyword) or Qdrant (semantic)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `[QUERY]` | argument | - | Search query (FTS5 syntax) |
| `--limit` | integer | 20 | Max results to return |
| `--source` | text | - | Filter by source/provider name |
| `--since` | text | - | Filter by date (ISO or natural language) |
| `--latest` | flag | - | Open most recent render instead |
| `--list` | flag | - | Print all hits (skip interactive picker) |
| `--verbose` | flag | - | Show snippets alongside hits |
| `--json` | flag | - | Output JSON |
| `--json-lines` | flag | - | Output JSON Lines |
| `--csv` | path | - | Write CSV to file |
| `--open` | flag | - | Open result after selection |

**Natural Language Dates**:
- `--since "last week"` - 7 days ago
- `--since "yesterday"` - 1 day ago
- `--since "2 months ago"` - 2 months ago
- `--since "2024-01-15"` - ISO format

**Search Syntax**:
- Simple: `python error` - matches both words
- Phrase: `"async await"` - exact phrase
- Boolean: `python AND async` - both required
- Exclusion: `python NOT django` - exclude django
- Prefix: `python*` - matches python, pythonic, etc.

**Examples**:
```bash
polylogue search "python async"                       # Search for phrase
polylogue search python --since "last week" --limit 10  # Recent Python conversations
polylogue search "error" --source claude --json       # JSON output for scripting
polylogue search --latest                             # Open most recent render
polylogue search "django" --verbose                   # Show snippets
```

**Interactive Mode**: If no flags, presents picker to select conversation and opens it.

---

#### `polylogue view` - Semantic Projections
**Purpose**: View conversations with filtering/transformation (projections)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `[CONVERSATION_ID]` | argument | - | Specific conversation to view |
| `-p, --projection` | choice | clean | Projection to apply (see below) |
| `--limit` | integer | 20 | Max conversations to list |
| `--offset` | integer | 0 | Skip first N conversations |
| `--provider` | text | - | Filter by provider (chatgpt, claude, etc.) |
| `--since` | text | - | Filter by date (ISO or natural language) |
| `--until` | text | - | Filter before date |
| `--query` | text | - | FTS search to filter conversations |
| `--json` | flag | - | Output JSON |
| `--json-lines` | flag | - | Output JSON Lines |
| `--list` | flag | - | List only (no content) |

**Projections** (filters applied to messages):

| Projection | Description | Use Case |
|------------|-------------|----------|
| `full` | All messages (no filtering) | See everything including system/tool messages |
| `dialogue` | User/assistant only (no system/tool) | Focus on conversation flow |
| `clean` | Substantive dialogue (no noise) | **Default** - Skip context dumps, tool spam |
| `pairs` | User-assistant turn pairs | See Q&A structure |
| `user` | User messages only | Review your questions/prompts |
| `assistant` | Assistant messages only | Review AI responses |
| `thinking` | Thinking/reasoning traces only | Claude's <thinking> blocks |
| `stats` | Statistics summary | Message counts, word counts, metadata |

**Examples**:
```bash
# List conversations
polylogue view                                      # Recent 20 conversations
polylogue view --provider claude                    # Claude conversations only
polylogue view --query "python" --limit 50          # Search + filter
polylogue view --since "last month" --json          # JSON output

# View specific conversation
polylogue view abc123                               # Clean projection (default)
polylogue view abc123 -p full                       # All messages
polylogue view abc123 -p thinking                   # Only <thinking> blocks
polylogue view abc123 -p pairs                      # User-assistant pairs
polylogue view abc123 -p stats                      # Statistics only

# Analysis workflows
polylogue view --provider claude -p stats --json    # Stats for all Claude chats
polylogue view --query "error" -p user              # See all your error questions
```

**Output Format**: Plain text (with Rich formatting) or JSON/JSON-Lines for scripting.

---

### Analytics & Insights

#### `polylogue analytics` - Archive Insights
**Purpose**: Analyze conversation patterns and provider comparison

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--provider-comparison` | flag | default | Show provider comparison metrics |
| `--json` | flag | - | Output JSON |

**Provider Comparison Metrics**:
- Conversation counts by provider
- Message counts (total, user, assistant)
- Average messages per conversation
- Average word counts (user vs assistant)
- Tool use frequency and percentage
- Thinking trace frequency and percentage

**Example Output**:
```
Provider Comparison
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Provider   ┃ Conversations ┃ Messages┃ Avg Msgs/Conv┃ Avg User Words┃ Tool Use %   ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ claude-code│ 245           │ 6,821   │ 27.8        │ 125          │ 78.4%        │
│ claude     │ 189           │ 3,456   │ 18.3        │ 98           │ 0.0%         │
│ chatgpt    │ 142           │ 2,134   │ 15.0        │ 87           │ 12.7%        │
└────────────┴───────────────┴─────────┴─────────────┴──────────────┴──────────────┘
Total: 576 conversations, 12,411 messages
```

**Examples**:
```bash
polylogue analytics                              # Default: provider comparison
polylogue analytics --provider-comparison        # Explicit
polylogue analytics --json                       # JSON for scripting/graphing
```

---

### Maintenance & Verification

#### `polylogue verify` - Data Integrity Check
**Purpose**: Detect orphaned references, schema issues, data quality problems

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--json` | flag | - | Output JSON |
| `--verbose` | flag | - | Show breakdown by provider |

**Checks**:
- Orphaned message references (conversation_id not found)
- Orphaned attachment references (attachment_id not found)
- Attachment ref_count mismatches
- Schema version consistency
- Provider metadata integrity

**Example Output**:
```
Data Integrity Check
✓ Schema version: 4
✓ Conversations: 576 (no orphans)
✓ Messages: 12,411 (no orphans)
✓ Attachments: 234 (ref_count matches)
✓ Attachment refs: 456 (all valid)

No issues found.
```

**Examples**:
```bash
polylogue verify                    # Quick check
polylogue verify --verbose          # Provider breakdown
polylogue verify --json             # Machine-readable output
```

---

#### `polylogue index` - Rebuild Search Index
**Purpose**: Rebuild FTS5 full-text search index

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | - | Override config file |

**When to use**:
- After manual database edits
- Search results seem stale
- Index corruption suspected
- Upgrading FTS schema

**Examples**:
```bash
polylogue index                     # Rebuild FTS5 index
polylogue index --config custom.json # Use custom config
```

**Note**: `polylogue run` automatically updates the index incrementally. Full rebuild is rarely needed.

---

### Configuration Management

#### `polylogue config` - Configuration Commands
**Purpose**: Manage configuration file

**Subcommands**:

##### `polylogue config init`
Create new config interactively

**Example**:
```bash
polylogue config init
# Prompts:
# - Archive location
# - Default sources
# - Index settings
# - Render preferences
```

##### `polylogue config show`
Display current configuration

**Options**:
- `--json` - Output JSON

**Examples**:
```bash
polylogue config show               # Pretty-printed YAML
polylogue config show --json        # JSON output
```

##### `polylogue config edit`
Open config in editor

**Example**:
```bash
polylogue config edit               # Opens in $EDITOR
```

##### `polylogue config set`
Set specific config value

**Example**:
```bash
polylogue config set archive_root /new/path
polylogue config set index.provider qdrant
```

---

#### `polylogue state` - State Management
**Purpose**: Manage application state

**Subcommands**:

##### `polylogue state reset`
Reset application state (clears cache, rebuilds index)

**Examples**:
```bash
polylogue state reset               # Full reset
polylogue state reset --confirm     # Skip confirmation
```

---

### Export & Integration

#### `polylogue export` - Export Archive
**Purpose**: Export conversations to portable format

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--out` | path | - | Output path for JSONL export |

**Format**: JSON Lines (.jsonl) - one conversation per line

**Examples**:
```bash
polylogue export --out archive.jsonl              # Export all
polylogue export --out backup-$(date +%F).jsonl   # Timestamped backup
```

**Use cases**:
- Backup archive
- Share with tools/scripts
- Migrate to another system
- Analysis with pandas/duckdb

---

### Web Server

#### `polylogue serve` - API Server
**Purpose**: Start FastAPI server for web UI and REST API

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | text | 127.0.0.1 | Host to bind |
| `--port` | integer | 8000 | Port to bind |

**Endpoints**:
- `GET /` - Web UI (modern.html template)
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation
- `GET /api/search?q={query}` - Search endpoint

**Examples**:
```bash
polylogue serve                            # Start on http://localhost:8000
polylogue serve --host 0.0.0.0 --port 3000 # Bind to all interfaces
```

**Access**:
- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs (Swagger UI)

---

### Shell Integration

#### `polylogue completions` - Shell Completion
**Purpose**: Generate shell completion scripts

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--shell` | choice | yes | Shell type: bash, zsh, fish |

**Examples**:
```bash
# Bash
polylogue completions --shell bash > ~/.bash_completion.d/polylogue
source ~/.bash_completion.d/polylogue

# Zsh
polylogue completions --shell zsh > ~/.zsh/completions/_polylogue
# Add to .zshrc: fpath=(~/.zsh/completions $fpath)

# Fish
polylogue completions --shell fish > ~/.config/fish/completions/polylogue.fish
```

**Features**: Tab completion for commands, options, and arguments.

---

## 🎨 Global Options

Available on all commands:

| Option | Description |
|--------|-------------|
| `--plain` | Force non-interactive plain output (CI/CD mode) |
| `--interactive` | Force interactive output (colors, spinners) |
| `--config PATH` | Override config file location |
| `-v, --verbose` | Show detailed information |
| `--version` | Show version and exit |
| `-h, --help` | Show help message |

**Auto-detection**:
- TTY mode: Interactive by default
- Piped output: Plain mode automatically
- `POLYLOGUE_FORCE_PLAIN=1`: Force plain mode

---

## 🔗 Command Relationships

```
┌─────────────────────────────────────────────────────────┐
│                    polylogue run                        │
│  (Main pipeline: ingest → render → index)               │
└────┬──────────────────────────────────┬─────────────────┘
     │                                   │
     │                                   ↓
     │                          ┌────────────────┐
     │                          │ polylogue index│
     │                          │ (FTS5 rebuild) │
     │                          └────────┬───────┘
     ↓                                   │
┌──────────────────┐                    │
│ polylogue sources│                    │
│ (List imports)   │                    │
└──────────────────┘                    ↓
                              ┌──────────────────────────┐
                              │   polylogue search       │
                              │   (FTS5/Qdrant query)    │
                              └────────┬─────────────────┘
                                       │
                                       ↓
                              ┌──────────────────────────┐
                              │   polylogue view         │
                              │   (Semantic projections) │
                              └──────────────────────────┘

                              ┌──────────────────────────┐
                              │   polylogue analytics    │
                              │   (Insights dashboard)   │
                              └──────────────────────────┘

┌─────────────────┐          ┌──────────────────────────┐
│ polylogue verify│          │   polylogue export       │
│ (Data integrity)│          │   (JSONL backup)         │
└─────────────────┘          └──────────────────────────┘

┌─────────────────┐          ┌──────────────────────────┐
│ polylogue config│          │   polylogue serve        │
│ (Settings)      │          │   (Web UI + API)         │
└─────────────────┘          └──────────────────────────┘
```

---

## 📊 Feature Matrix

| Feature | Commands | Description |
|---------|----------|-------------|
| **Import** | `run`, `sources` | ChatGPT, Claude, Codex, Gemini → SQLite |
| **Search** | `search` | FTS5 (keyword) or Qdrant (semantic) |
| **Browse** | `view` | List conversations with filters |
| **Filter** | `search`, `view` | By provider, date, content |
| **Projections** | `view` | Semantic message filtering (8 types) |
| **Analytics** | `analytics` | Provider comparison, statistics |
| **Rendering** | `run --format` | Markdown or HTML output |
| **Indexing** | `run`, `index` | FTS5 full-text search index |
| **Export** | `export` | JSONL portable format |
| **Web UI** | `serve` | Browser-based interface + REST API |
| **Verification** | `verify` | Data integrity checks |
| **Configuration** | `config` | YAML-based settings |
| **Shell Integration** | `completions` | Tab completion for bash/zsh/fish |

---

## 🚀 Power User Tips

### 1. Natural Language Dates
```bash
polylogue search "python" --since "3 days ago"
polylogue view --since "last month" --until "last week"
```

### 2. JSON Output for Scripting
```bash
# Get conversation IDs from last week
polylogue view --since "last week" --json | jq -r '.[].conversation_id'

# Export search results to CSV
polylogue search "error" --csv errors.csv

# Analytics to JSON for graphing
polylogue analytics --json > stats.json
```

### 3. Projection Chains (via jq)
```bash
# Get all your questions about Python
polylogue view --query "python" -p user --json | jq -r '.messages[].text'

# Extract thinking traces
polylogue view abc123 -p thinking --json
```

### 4. Incremental Updates
```bash
# Daily sync
polylogue run --source last --stage ingest

# Weekly full rebuild
polylogue run
```

### 5. Search Operators
```bash
# Boolean: both terms required
polylogue search "python AND async"

# Exclusion: exclude django
polylogue search "python NOT django"

# Phrase: exact match
polylogue search '"async/await pattern"'

# Prefix: matches variations
polylogue search "python*"  # pythonic, python3, etc.
```

---

## 📁 File Locations

| Resource | Default Location | Override |
|----------|-----------------|----------|
| Config | `~/.config/polylogue/config.json` | `--config PATH` or `POLYLOGUE_CONFIG` |
| Database | `~/.local/state/polylogue/polylogue.db` | Set in config: `archive_root` |
| Renders | `~/.local/state/polylogue/render/` | Set in config: `render_root` |
| Logs | stderr (structured JSON via structlog) | - |

---

## 🎯 Common Use Cases

### Daily Workflow
```bash
# Morning: sync new conversations
polylogue run --source last

# Search for something you discussed
polylogue search "kubernetes deployment"

# Review recent Python conversations
polylogue view --query "python" --since "this week"
```

### Research Workflow
```bash
# Find all discussions about a topic
polylogue search "machine learning" --limit 100 --json > ml-convs.json

# Analyze conversation patterns
polylogue analytics --provider-comparison --json

# Extract specific message types
polylogue view --query "debugging" -p user --json-lines
```

### Maintenance Workflow
```bash
# Weekly: verify integrity
polylogue verify --verbose

# Monthly: full re-import + rebuild
polylogue run --preview  # Preview first
polylogue run            # Execute

# Backup
polylogue export --out backup-$(date +%F).jsonl
```

### Development Workflow
```bash
# Start API server
polylogue serve --host 0.0.0.0

# Test search in browser
open http://localhost:8000

# Query API programmatically
curl http://localhost:8000/api/search?q=python | jq
```

---

## 🔧 Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `POLYLOGUE_CONFIG` | Override config path | `/custom/config.json` |
| `POLYLOGUE_FORCE_PLAIN` | Force plain output | `1`, `true`, `yes` |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant auth | `secret-key` |
| `VOYAGE_API_KEY` | Voyage AI embeddings | `api-key` |

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Config not found" | Run `polylogue config init` |
| "Index not built" | Run `polylogue index` or `polylogue run` |
| Search returns nothing | Check if index exists: `polylogue verify` |
| "Database locked" | Stop other polylogue processes |
| Import fails | Check source path in config |
| Slow search | Rebuild index: `polylogue index` |

---

## 📚 Related Documentation

- [Architecture Roadmap](./architecture-roadmap.md) - Future features
- [CLAUDE.md](../CLAUDE.md) - Developer guide
- [Performance Benchmarks](../tests/benchmarks/) - Performance metrics
- [API Documentation](http://localhost:8000/docs) - REST API (when serving)
