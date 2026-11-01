# Capture coverage

Polylogue collects session material from each tool's fixed directories. It
uses `CLAUDE_CONFIG_DIR` and `CODEX_HOME` when set. Provider exports enter
through an explicit `polylogue import`; account directories are not watched.

## Codex rollouts

Both `sessions/**/rollout-*.jsonl` and
`archived_sessions/**/rollout-*.jsonl` are collected, along with their `.zst`
encodings. The original compressed bytes are stored as a `codex-compressed`
byte input at their physical path. The decoded transcript uses the same
`codex` input as the plain path, without the `.zst` suffix. Parsing and byte
ranges refer to that decoded transcript. Both are stored before parsing and
survive removal of the provider's files and an index rebuild.

When compression preserves the transcript, the collector compares its decoded
hash with the stored transcript. It records the new encoding and confirms the
existing transcript revision. It creates neither another session nor a
rewritten transcript revision. Recompressing at another compression level
also leaves the transcript revision unchanged. Changed decoded content starts
a new transcript revision.

## Side files

The collection decision applies to every file below a listed directory,
including nested files and file extensions not known when Polylogue was built.

| Tool and file kind | Collection and session link |
| --- | --- |
| Claude Code transcripts, todos, tasks, subagent metadata, job state and timeline, file history, tool results, plans, project memory | Collected by their existing declarations. Nested project memory is also collected. |
| Claude Code `paste-cache/` | Original bytes stored; a `paste_file` event gives the cache path. A hash or paste ID alone does not establish a session. |
| Claude Code `history.jsonl` paste entries | `paste_span` events on the explicit `sessionId` keep the marker, paste fields and a byte span when the pasted content occurs exactly once in `display`. Content without a display match and hash-only entries remain distinct. |
| Claude Code `teams/` | Collected; JSON with `leadSessionId` or another explicit session ID links to that session. Inbox files without a session ID remain input events. |
| Claude Code `usage-data/` | Collected; session metadata and facets use their explicit session ID or complete UUID filename. HTML reports remain input events. |
| Claude Code `history-summaries/` | Collected; short filename prefixes do not establish a session link. |
| Claude Code `remote-agents/*.meta.json` | Collected; explicit session IDs establish links. |
| Claude Code other `jobs/` files | Prompt drafts, diffs, HTML and other files are collected. Explicit session IDs establish links; the shared job directory remains in each event's path. |
| Claude Code `shell-snapshots/`, `debug/`, `mcp-logs/` | Collected as execution context. A complete session UUID in a debug filename establishes a link. |
| Codex `attachments/` | Collected; the enclosing directory names the thread. |
| Codex `memories/` | Collected as shared context; a memory filename does not establish a thread. |
| Codex `shell_snapshots/` | Collected; the complete thread ID before the timestamp establishes a link. |
| Codex `dictation-history`, `visualizations/` | Collected when present, including a dictation-history directory or file. Explicit session IDs establish links. |
| Codex `logs_*.sqlite`, `queue_*.sqlite` | Captured through the existing consistent SQLite row dump, including the WAL. Logs and pending prompts become events linked by `thread_id`. |
| Codex `local_images`, local-image records and image `file://` URLs | Absolute referenced files are collected even outside the tool root. Their recorded path links the transcript reference to the file observation. Missing files are capture errors. Relative paths use the recorded session or turn working directory; a missing directory is a capture error. |
| Other files under a source root | Listed as unmatched in the capture health report. No bytes are collected until their kind has a collection declaration. This includes tool installation files, settings, authentication files and newly introduced session formats. |
| Directory symbolic links | Not traversed. Listed as unmatched when no declaration claims them, so cycles cannot hide or stall the root scan. |

Side-file events cite the original input byte range and keep its path and size.
A file that names no session is still stored and counted. Polylogue does not
infer session identity from nearby times, abbreviated IDs or shared prose.

## Health report

`polylogue status` includes `capture_health`, also present in `last_pass`.
Each source root reports the number of files, the number matching declarations,
all unmatched paths, paths rejected by content recognition, and unreadable
paths. A full scan refreshes it. Cooperative SQLite capture turns keep the last
full report visible. A missing root is empty; a root or directory that exists
but cannot be read appears as unreadable. Individual directory-entry errors
are reported against their directory.

The report covers the tool directory, rather than all of home or XDG state.
Polylogue's own spools are reported separately. A changed tool layout therefore
appears in the next report even when no existing collection pattern matches it.

## Importing provider exports

Import a provider ZIP directly to keep its exact original bytes and each
member's relationship to it:

```sh
polylogue import /path/to/chatgpt-export.zip
polylogue import /path/to/claude-export.zip
```

An extracted export directory also works:

```sh
polylogue import /path/to/extracted-export
```

ChatGPT's single `conversations.json` and sharded
`conversations-NNN.json` arrays are recognized, including shards whose first
key is `async_status`. Claude exports use their `uuid` conversation shape.
Nested export directories are supported. ZIPs and directory imports keep
attachment files and sidecars alongside the conversations. Importing the same
bytes again reuses their stored inputs. An index rebuild reads the stored
members even when the original export has been removed.
