[← Back to Docs](README.md)

# Search & Query Reference

Polylogue uses a query-first CLI grammar: bare tokens are search terms, flags
are filters, and trailing subcommands are verbs.

## Grammar

```
polylogue [<terms>] [filters] [verb]
```

- **terms**: space-separated words matched by full-text search (AND when
  repeated with `--contains`)
- **filters**: `--flag` filters that narrow the conversation set
- **verb**: optional trailing subcommand (`list`, `stats`, `count`, etc.)

## Filters

### Identity and content

| Flag | Description |
|------|-------------|
| `--id`, `-i` | Conversation ID (exact or prefix match) |
| `--contains`, `-c` | FTS term (repeatable = AND) |
| `--exclude-text` | Exclude conversations matching this term |
| `--title` | Title contains substring |
| `--provider`, `-p` | Include providers (comma = OR) |
| `--exclude-provider` | Exclude providers |
| `--repo`, `-r` | Filter by repository name |
| `--referenced-path` | File path contains substring (repeatable = AND) |
| `--cwd-prefix` | Working directory starts with this prefix |

### Content type

| Flag | Description |
|------|-------------|
| `--has-tool-use` | Only conversations with tool calls |
| `--has-thinking` | Only conversations with reasoning/thinking blocks |
| `--has-paste` | Only conversations with pasted content |
| `--typed-only` | Only typed (non-pasted) content |
| `--has`, `--has-type` | Filter by content type: `thinking`, `tools`, `summary`, `attachments` |

### Message stats

| Flag | Description |
|------|-------------|
| `--min-messages` | Minimum message count |
| `--max-messages` | Maximum message count |
| `--min-words` | Minimum word count |
| `--message-type` | Filter by message type |

### Semantic actions

| Flag | Description |
|------|-------------|
| `--action` | Require semantic action: `file_read`, `file_write`, `file_edit`, `shell`, `search`, `web`, `agent`, `subagent`, `git` (repeatable = AND) |
| `--exclude-action` | Exclude semantic action (repeatable = AND) |
| `--action-sequence` | Require ordered action subsequence (comma-separated) |
| `--action-text` | Text match within action evidence (repeatable = AND) |
| `--tool` | Require normalized tool name (repeatable = AND) |
| `--exclude-tool` | Exclude normalized tool name (repeatable = AND) |

### Time and scope

| Flag | Description |
|------|-------------|
| `--since` | Only conversations on or after this date/time |
| `--until` | Only conversations on or before this date/time |
| `--limit`, `-n` | Maximum results |
| `--offset` | Start offset |
| `--latest` | Newest-first sort |
| `--sort` | Sort order |
| `--reverse` | Reverse sort direction |
| `--sample` | Random sample of N conversations |

### Tags

| Flag | Description |
|------|-------------|
| `--tag`, `-t` | Include tags (comma = OR, supports `key:value`) |
| `--exclude-tag` | Exclude tags |

### Retrieval

| Flag | Description |
|------|-------------|
| `--retrieval-lane` | Query lane: `auto`, `dialogue`, `actions`, `hybrid` |
| `--similar` | Semantic similarity query (requires embeddings) |

### Output modifiers

| Flag | Description |
|------|-------------|
| `--no-code-blocks` | Strip code blocks from output |
| `--no-tool-calls` | Strip tool call blocks |
| `--no-tool-outputs` | Strip tool result blocks |
| `--no-file-reads` | Strip file read blocks |
| `--prose-only` | Show only authored prose text |
| `--dialogue-only` | Show only user/assistant messages |
| `--message-role` | Filter by role (`user`, `assistant`, `system`, `tool`) |

## Verbs

Verbs determine the action applied to the matched conversation set.

| Verb | Description |
|------|-------------|
| `list` | List matched conversations with metadata |
| `count` | Print count of matched conversations |
| `stats` | Grouped statistics (`--by provider`, `month`, `year`, `day`, `action`, `tool`, `repo`, `work-kind`) |
| `show` | Display full conversation content |
| `open` | Open conversation in browser/editor |
| `bulk-export` | Export matched conversations to file |
| `messages` | Show individual messages |
| `raw` | Show raw (unparsed) conversation data |
| `select` | Select and print a single field |
| `delete` | Delete matched conversations (requires `--dry-run` confirmation) |

## Retrieval Lanes

| Lane | Description |
|------|-------------|
| `auto` | Chooses FTS5 or hybrid based on query and available indexes |
| `dialogue` | FTS5 over message text (`messages_fts` virtual table, `unicode61` tokenizer) |
| `actions` | FTS5 over action event text (`action_events_fts`) |
| `hybrid` | Reciprocal Rank Fusion combining FTS5 and vector similarity (requires embeddings) |

## FTS5 Syntax

The `messages_fts` virtual table uses SQLite's FTS5 with the `unicode61`
tokenizer. Prefix queries use `*`:

```bash
polylogue "refactor*"
```

Phrase queries use quotes:

```bash
polylogue '"null pointer exception"'
```

Boolean operators combine terms:

```bash
polylogue "refactor AND schema NOT test"
```

Column filters restrict matches:

```bash
polylogue 'text:css {conversation_id claude-code}: refactor'
```

## Output Formats

| Format | Description |
|--------|-------------|
| `markdown` | Default -- formatted markdown with syntax-highlighted code blocks |
| `json` | Full conversation as JSON |
| `jsonl` | One JSON object per line (used by `bulk-export`) |
| `yaml` | YAML representation |
| `plaintext` | Plain text, no formatting |
| `html` | HTML with Pygments syntax highlighting |
| `obsidian` | YAML frontmatter + markdown body |
| `org` | Org-mode format |
| `csv` | Messages as rows |

Set format with `-f` / `--format` on a verb:

```bash
polylogue "sqlite locking" list --format json
polylogue --since yesterday bulk-export --format jsonl
```

## Empty Result Diagnostics

When a query returns no results:

1. Check provider spelling: `polylogue -p claude-code list` (not `claude_code`)
2. Expand the time window: `--since 2024-01` instead of `--since yesterday`
3. Verify the archive has data: `polylogue count` (no filters)
4. Check FTS index health: `polylogued status` shows `fts_readiness`
5. Run `polylogue check` for schema and index integrity
6. If using `--similar`, ensure embeddings are built (check `polylogue stats` for embedding coverage)
