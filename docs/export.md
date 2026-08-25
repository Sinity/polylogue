[← Back to Docs](README.md)

# Export

Polylogue does not have a separate export command. Export is query selection
plus a `read` projection plus a renderer. Use `read` for one selected session
and `read --all` when the query result set itself is the export target.

## Single Session

Read one session by ID with a root filter that narrows the archive to a single
match:

```bash
polylogue --id claude-ai-export:abc123 read
```

Specify format:

```bash
polylogue --id claude-ai-export:abc123 read --format json
polylogue --id claude-ai-export:abc123 read --format yaml
```

## Query-Set Reads

Read every session matching a filter chain with `read --all`:

```bash
polylogue --origin claude-code-session --since 2026-01 find 'origin:claude-code-session' then read --all
polylogue --tag important find 'tag:important' then read --all --format markdown
polylogue --has-tool-use find "refactor" then read --all --format ndjson
```

### Piping

`read --all --format ndjson` emits one JSON line per session, suitable for
piping:

```bash
polylogue --origin claude-code-session find 'origin:claude-code-session' then read --all --format ndjson | jq '.title'
polylogue --since "last month" find 'since:"last month"' then read --all --format ndjson | wc -l
```

## Export Formats

| Format | Description |
|--------|-------------|
| `markdown` | Human-readable query-set output |
| `json` | Query-set envelope as structured JSON |
| `ndjson` | One JSON object per line |
| `yaml` | Query-set envelope as YAML |
| `plaintext` | Unformatted query-set rows |
| `csv` | Comma-separated query-set rows |

These are the formats supported by the standard query-set export path. Default
output varies by view and cardinality, so specify a format in automation. Other
views have their own contracts; use `polylogue read --views --format json` to
inspect the current view-specific formats before relying on one.

Set format with `--format` / `-f`:

```bash
polylogue --id <id> read --format json
polylogue --since yesterday find 'since:yesterday' then read --all --format yaml
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
