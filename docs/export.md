[← Back to Docs](README.md)

# Export

Polylogue does not have a separate export command. Export is query selection
plus a `read` projection plus a renderer. Use `read` for one selected session
and `read --all` when the query result set itself is the export target.

## Single Session

Read one session by ID with a root filter that narrows the archive to a single
match:

```bash
polylogue --id claude-ai:abc123 read
```

Specify format:

```bash
polylogue --id claude-ai:abc123 read --format json
polylogue --id claude-ai:abc123 read --format html
polylogue --id claude-ai:abc123 read --format obsidian
```

## Query-Set Reads

Read every session matching a filter chain with `read --all`:

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
| `ndjson` | One JSON object per line (query-set default) |
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

Reads include all selected content blocks: text, thinking blocks, tool use, tool
results, images, code blocks, and document references. Use query-unit
expressions or explicit read views to narrow what is selected before rendering.

## Sharing Considerations

- Rendered output can contain full message content including tool inputs/outputs and
  thinking blocks. Review before sharing.
- Cost estimates are approximate; API-reported token counts are exact where
  available.
- Attachment references are preserved but binary blobs are not rendered.
- Use query-unit filters or a narrower read view to produce cleaner output for
  non-technical audiences.
