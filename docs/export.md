[← Back to Docs](README.md)

# Export

Polylogue exports sessions in multiple formats, both singly and in bulk.

## Single Export

Export one session by ID:

```bash
polylogue export claude-ai:abc123
```

Specify format:

```bash
polylogue export claude-ai:abc123 --format json
polylogue export claude-ai:abc123 --format html
polylogue export claude-ai:abc123 --format obsidian
```

## Bulk Export

Export every session matching a filter chain:

```bash
polylogue --provider claude-code --since 2026-01 bulk-export
polylogue --tag important bulk-export --format markdown
polylogue "refactor" --has-tool-use bulk-export --format jsonl
```

### Piping

`bulk-export --format jsonl` emits one JSON line per session, suitable for
piping:

```bash
polylogue -p claude-code bulk-export --format jsonl | jq '.title'
polylogue --since "last month" bulk-export --format jsonl | wc -l
```

## Export Formats

| Format | Description |
|--------|-------------|
| `markdown` | Default -- formatted markdown with message roles, timestamps, and code blocks |
| `json` | Full session as structured JSON |
| `jsonl` | One JSON object per line (bulk-export default) |
| `yaml` | YAML representation |
| `plaintext` | Plain text -- message content only, no formatting |
| `html` | HTML with Pygments syntax highlighting on code blocks |
| `obsidian` | YAML frontmatter + markdown body, compatible with Obsidian vaults |
| `org` | Org-mode format for Emacs users |

Set format with `--format` / `-f`:

```bash
polylogue export <id> --format json
polylogue --since yesterday bulk-export --format html
```

## Content Blocks

Exports include all content blocks: text, thinking blocks, tool use, tool
results, images, code blocks, and document references. Use output modifiers to
filter:

```bash
polylogue export <id> --prose-only        # Only authored prose
polylogue export <id> --no-tool-calls     # Strip tool call blocks
polylogue export <id> --no-tool-outputs   # Strip tool result blocks
polylogue export <id> --no-code-blocks    # Strip code blocks
polylogue export <id> --no-file-reads     # Strip file read blocks
polylogue export <id> --dialogue-only     # User/assistant messages only
```

## Sharing Considerations

- Exports contain full message content including tool inputs/outputs and
  thinking blocks. Review before sharing.
- Cost estimates are approximate; API-reported token counts are exact where
  available.
- Attachment references are preserved but binary blobs are not exported.
- Use `--prose-only` or content filters to produce cleaner output for
  non-technical audiences.
