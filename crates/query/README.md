# Session queries

`bind` turns query text into a `SessionQuery`. The CLI calls it to check a query before sending it; the daemon calls it again before reading the store. Surfaces only render the result.

Supported forms:

```text
sessions
origin:claude-code kind:main
claude-code:opaque-session-id
sessions where origin:claude-code text:"shared phrase" refact*
chrono | sort by last_active desc | limit 10
```

- Conditions next to each other, or joined by `AND`, must all match.
- Words are letters and digits; `word*` is a prefix; quoted text is a phrase.
- `kind:*` includes subagent sessions. An exact ID also matches subagents.
- `limit` applies after filtering and sorting, and the total match count stays visible. No stage may follow `limit`.
- `find` marks the rest as search text, so `find list` searches for "list" instead of listing everything.

Anything else, such as other fields, `OR`, `NOT`, grouping or other sorts, returns an error with its position. It is never treated as search text.

Results are newest first by the latest known message time, unknown times last, then by descending ID. Text search uses the same order and reports that relevance ranking is not available. Words may match in different text blocks of one message; a phrase must match within one text block. Rows whose author is unknown are left out and counted. No matches over an archive that is not fully parsed is reported as degraded, not as an empty success.

`bind_tools` handles `tool_calls` and `tool_items` queries, which need `session:ID`, accept an optional `text:` and a final `| limit N` (1 to 4096, default 100).

The engine and store run the query; see the top-level README for limits.
