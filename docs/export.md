[← Back to Docs](README.md)

# Export

Polylogue exports sessions in multiple formats, both singly and in bulk. Export
is delivered through the `read` verb of the query-first CLI.

## Single Export

Export one session by ID with a root filter that narrows the archive to a
single match:

```bash
polylogue --id claude-ai:abc123 read
```

Specify format:

```bash
polylogue --id claude-ai:abc123 read --format json
polylogue --id claude-ai:abc123 read --format html
polylogue --id claude-ai:abc123 read --format obsidian
```

## Bulk Export

Export every session matching a filter chain with `read --all`:

```bash
polylogue --origin claude-code-session --since 2026-01 read --all
polylogue --tag important read --all --format markdown
polylogue "refactor" --has-tool-use read --all --format ndjson
```

### Piping

`read --all --format ndjson` emits one JSON line per session, suitable for
piping:

```bash
polylogue -p claude-code read --all --format ndjson | jq '.title'
polylogue --since "last month" read --all --format ndjson | wc -l
```

## Export Formats

| Format | Description |
|--------|-------------|
| `text` | Plain text -- message content only, no formatting |
| `markdown` | Default -- formatted markdown with message roles, timestamps, and code blocks |
| `json` | Full session as structured JSON |
| `ndjson` | One JSON object per line (bulk default) |
| `yaml` | YAML representation |
| `html` | HTML with Pygments syntax highlighting on code blocks |
| `obsidian` | YAML frontmatter + markdown body, compatible with Obsidian vaults |
| `org` | Org-mode format for Emacs users |
| `csv` | Comma-separated rows |

Set format with `--format` / `-f`:

```bash
polylogue --id <id> read --format json
polylogue --since yesterday read --all --format html
```

## Content Blocks

Exports include all content blocks: text, thinking blocks, tool use, tool
results, images, code blocks, and document references. Use output modifiers to
filter:

```bash
polylogue --id <id> read --prose-only        # Only authored prose
polylogue --id <id> read --no-tool-calls     # Strip tool call blocks
polylogue --id <id> read --no-tool-outputs   # Strip tool result blocks
polylogue --id <id> read --no-code-blocks    # Strip code blocks
polylogue --id <id> read --no-file-reads     # Strip file read blocks
polylogue --dialogue-only --id <id> read     # User/assistant messages only
```

## Sharing Considerations

- Exports contain full message content including tool inputs/outputs and
  thinking blocks. Review before sharing.
- Cost estimates are approximate; API-reported token counts are exact where
  available.
- Attachment references are preserved but binary blobs are not exported.
- Use `--prose-only` or content filters to produce cleaner output for
  non-technical audiences.
