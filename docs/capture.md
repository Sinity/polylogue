# Capture

Where Polylogue's inputs come from: the location model and every live origin, hooks, tool databases, manual import of exports, AI Studio through Drive, and capture status. What happens to the bytes once collected is in [archive.md](archive.md); the parsed model is in [sessions.md](sessions.md). The browser extension's capture and backfill are in [extension.md](extension.md). In primary mode capture is unchanged and every collected input also becomes a Sinex material ([primary-mode.md](primary-mode.md)).

## Principles

- **Every relevant origin, at full fidelity.** Every AI tool whose sessions the user has is captured, with every side file that carries conversation content, context or harness-side evidence, even before a parser for it exists, because harnesses prune their own files and a pruned file is lost. A degraded form (a flattened transcript, a page scrape where the provider's data is available) is never the stored record. (ruling 2026-02-15, cc-queued:7ebe20ed-a8bf-4392-98ab-bce1ea619c70@2026-02-15T02:42:31.922Z).
- **Nothing disappears silently.** Every file under a declared root is collected, ignored with a recorded reason, or reported unrecognized. (ruling 2026-07-31, cc:38baa1de-9715-48fa-8175-f2a29d92800e:bc4b84cc-1c23-46df-bd01-dbb4219d5d47).
- **Fixed locations, strict recognition, fail closed.** Polylogue reads data where each tool writes it; there are no path settings and no custom roots. Input that does not match its declared shape is refused, never reinterpreted. (ruling 2026-09-27, cc:62be45f2-6c40-49ce-a88e-bc566948628c:75bf98d6-b1e7-461a-bd53-501ec33de178).
- **Live is the normal case.** A growing file is tailed; a live origin needs no import step.
- **Exports arrive by hand.** Provider exports are brought in with `polylogue import`; nothing is picked up from a download or inbox directory. (ruling 2026-07-06, cc-queued:f7680f9c-32e4-4f2f-9180-b9977e1f4702@2026-07-06T17:12:07.017Z).
- **Databases are captured as rows, not images** ([Tool databases](#tool-databases)).
- **No size limit on valid input.** Payloads, attachments and export members of any size stream through.

## Locations

### Roots, locations and the ignored list

- Each origin declares its **root**, a fixed set of **location** patterns, and an **ignored list** with one reason per entry (secret or account state, program files, cache, derived pile, the tool's lock files).
- The root honours the tool's own home variable where the tool defines one (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `HERMES_HOME`); otherwise it is the tool's fixed path under `$HOME`. Polylogue has no setting of its own for any path.
- Ignoring a tool's credential files is not filtering content: they are account state, not AI work.

### Every file in one class

| Class | Meaning | What happens |
|---|---|---|
| `collected` | matched by a location | stored as bytes (or row dumps), and parsed when the origin has a parser |
| `ignored` | matched by an ignored entry | not stored; counted with its reason |
| `unrecognized` | matched by neither | not stored, not parsed; counted per root, directory pattern and extension, and shown in status |

- **Collecting does not wait for a parser.** An origin has two separate switches: *collect*, on for every declared location, and *parse*. A declared origin without a parser stores its bytes from the first day with parse state `no_parser_yet` ([archive.md](archive.md#parsing)).
- **Fail closed.** A collected file that fails its origin's shape check is `refused`, with the reason, as a parse-state fact; it is never read as another origin's file. A Claude Code transcript found in Drive is not ingested as a Claude Code session, because it would duplicate the real one.
- **A bad record is not a bad file.** A malformed line in a JSONL file is counted unreadable and the complete records after it are still read; a parser error stops that parser's rows at its last good record while the bytes keep being collected.
- Hidden directories are entered only when a location names them. Symlinked roots resolve once. An unreadable directory is a reported state, never an empty result. A `.git` directory inside a root is ignored with its reason.
- **Discovery follows source changes.** Full walks run at startup and when the watcher reports overflow or an error. Ordinary events schedule affected locations; idle time does not trigger a whole-tree walk. Collector schedules and live-file checks continue independently.
- **The coverage table** `(root, class, pattern or reason, files, bytes, newest mtime)` is computed from the discovery walk the daemon already performs (the start-up walk plus file notifications). It is an index fact, rebuilt with the index; there is no separate scan.

### Live origins

Paths are relative to each root. "Parse" says what becomes sessions, documents or events; everything collected is stored as bytes whether or not it is parsed.

**Claude Code** (root `$CLAUDE_CONFIG_DIR`, default `~/.claude`)

| Location | Parse |
|---|---|
| `projects/*/*.jsonl`; `projects/*/*/subagents/**/agent-*.jsonl`; `subagents/workflows/*/journal.jsonl` | transcripts, subagents, workflow journals |
| `*.meta.json`, `tool-results/*`, `memory/*`, `todos/`, `tasks/`, `plans/`, `file-history/`, `jobs/**` | side files of their session ([sessions.md](sessions.md#sidecars-and-linked-materials)); todo and plan items in full |
| `projects/*/*/workflows/*.json`, `workflows/scripts/*.js` | workflow run snapshots as facts; scripts as documents |
| `projects/*/sessions-index.json` | competing title snapshots |
| `projects/*/bridge-pointer.json`, `projects/*/*/remote-agents/*.meta.json` | remote-run events and spawn links |
| `history.jsonl` | prompt events and paste evidence |
| `paste-cache/*.txt` | paste bodies |
| `teams/*.json` | team membership and messages as events |
| `usage-data/*`, `history-summaries/*.md` | documents attached to sessions |
| `shell-snapshots/*.sh` | session context documents |
| `sessions/*.json` | live-state evidence (process to session) |
| `debug/*`, `mcp-logs/**` | bytes only |

Ignored: `.credentials.json`, `.claude.json`, `mcp-needs-auth-cache.json` (secrets or account state); `plugins/`, `skills/`, `agents/`, `hooks/`, `statsig/`, `telemetry/`, `cache/`, `ccusage/`, `stats-cache.json`, `backups/`, `chrome/`, `ide/` (configuration, program files, caches); the user's own projects that happen to sit under the root.

**Codex** (root `$CODEX_HOME`, default `~/.codex`)

| Location | Parse |
|---|---|
| `sessions/**/rollout-*.jsonl` and `rollout-*.jsonl.zst`; `archived_sessions/**` | sessions. A compressed rollout is the same input as its plain form: identity is content and native ID, never the path. When Codex replaces a plain rollout with its `.zst`, the decoded bytes continue the stored input and any lines written since are appended; the compressed file's own bytes are kept too. Every zstd frame in the file is decoded. A truncated or undecodable file is reported and stores nothing until a complete file is there |
| `history.jsonl`, `session_index.jsonl` | prompt events, titles |
| rows of the `state_*`, `goals_*`, `memories_*` and `queue_*` databases | goals, memories and queued prompts as events |
| `memories/**/*.md` (not `memories/.git`) | memory documents |
| `attachments/**`, `shell_snapshots/**`, `ambient-suggestions/**` | attachments, context documents |
| files named by `local_images` in a rollout line | attachment bytes, read when the line is observed; the opened descriptor must be a regular file, and a nonregular source is reported unreadable without waiting for a pipe writer ([sessions.md](sessions.md#attachments)) |

Ignored: `logs_2.sqlite` (diagnostic log); `thread_history_*.sqlite` (a projection of the rollouts); `sqlite/codex-dev.db` (developer database); `worktrees/`, `packages/`, `plugins/`, `vendor_imports/`, `skills/`, `cache/`, `tmp/`, lock directories; `auth.json`; configuration files; backup tarballs left in the tree (importable by hand).

**Gemini CLI** (root `~/.gemini`): `tmp/*/chats/session-*.json` and `.jsonl`; subagent `chats/<id>/*.json`; `tool-outputs/session-*/*` (overflow sidecars, whose full text replaces the truncated inline text); `logs.json` (prompt log); `history/*` checkpoints kept as git refs and objects, so a checkpoint ID in a transcript resolves, with checkpoint commits as events. Ignored: credential files, `skills/`, settings, `*.partial`. The `antigravity*` subtrees belong to Antigravity.

**Antigravity**: everything under `~/.gemini/antigravity/` (`conversations/*.pb`, `brain/**`, `annotations/*.pbtxt`, `implicit/*`, `code_tracker/**`, `knowledge/**`, `user_settings.pb`); `~/.gemini/antigravity-cli/` (`conversations/*.db` and `conversation_summaries.db` as rows, `history.jsonl`, `brain/**` including `.system_generated/logs/*.jsonl`, `annotations/`, `implicit/`, `knowledge/`, `jetski_state.pbtxt`); and `~/.config/Antigravity/User/{globalStorage,workspaceStorage/*}/state.vscdb` as rows. Ignored: the OAuth token, `bin/`, `builtin/`, `cache/`, `updater/`, lock files, `crashes/`, MCP schema caches, `log/`. Its readers, from the plainest format to the hardest:
1. CLI transcripts `brain/<id>/.system_generated/logs/transcript_full.jsonl`; where a trajectory store of the same ID exists, its `step-<idx>` IDs are the message IDs;
2. CLI trajectory stores, decoded with descriptors from the installed language server;
3. summaries and prompts (`conversation_summaries.db`, `history.jsonl`);
4. brain artifacts, as versioned documents of their conversation;
5. IDE `.pb` conversations, through the language server's own RPC on a Unix socket or with a per-run token, never an unauthenticated loopback port; without the server, the session exists with its metadata and an event saying the transcript could not be read;
6. `state.vscdb` rows.

**Hermes** (root `$HERMES_HOME`, default `~/.hermes`, and each `profiles/<name>/` as a full home): `state.db` and `verification_evidence.db` as rows, every column; `sessions/session_*.json`, `sessions/saved/*.json`, `sessions/request_dump_*.json`; `observability/nemo-relay/atif/*.json`, `atof/*.jsonl`, `usage.json`; `pastes/*`, `.hermes_history`, `memories/*`, `checkpoints/*`, `cron/*`; `SOUL.md` and `config.yaml` as context documents. Ignored: `.env`, `auth*`, `hermes-agent/` (the program's checkout), `bin/`, `skills/`, caches, `logs/`, `sandboxes/`, `processes.json`. JSON snapshots and `state.db` sessions merge by native ID: the snapshot is the record and live events enrich it.

**Grok CLI** (root `~/.grok`): every file of each session directory `sessions/<cwd>/<id>/` (`chat_history.jsonl`, `updates.jsonl`, `events.jsonl`, `rewind_points.jsonl`, `usage.json`, `summary.json`, `signals.json`, `tool_definitions.json`, `prompt_context.json`, `resources_state.json`, `plan_mode.json`, `subagents/*/*.json`, `recap_requests/*.json`, `goal/*.md`, `terminal/*.log`); `sessions/*/prompt_history.jsonl`; `memory-v2/**` (Markdown, and SQLite as rows); `memtrace/**`; `worktrees.db` as rows. Rewind points are branch events, subagents are linked sessions, terminal logs are tool-output sidecars. Ignored: auth files, program and plugin directories, caches, docs, lock files.

**`llm` CLI** (`~/.config/io.datasette.llm/logs.db`, as rows): a session per `conversation_id`, with prompt and response messages, model, tokens and attachments.

**`pi`** (`~/.pi/agent/sessions/**/*.jsonl`): collected, with parse state `no_parser_yet` until a parser exists.

**Other producers.** Claude Code web runs are their own origin, `claude-code-web`, once a real transcript file exists. Codex Cloud tasks are origin `codex-cloud` with their own ID namespace (`task_e_…`), from `codex.json` in ChatGPT exports and from the extension.

## Hooks

Hooks are a capture channel: every event each harness emits is recorded whole. Hook-only facts (permission answers, pre-compaction context, exact paste bodies, process identity) exist nowhere else. Which events are declared and how they are installed is in [operations.md](operations.md#installing-into-harnesses); what context hooks bring to a session is in [agents.md](agents.md#brief).

- **Every event is installed and spooled.** Claude Code's full set (including `Notification`, `PermissionRequest`, `PermissionDenied`, `PostToolUseFailure`, `SubagentStart`, `SubagentStop`, `PreCompact`, `PostCompact`, `Elicitation`, `ElicitationResult`, `CwdChanged`, `FileChanged`, `WorktreeCreate`, `Setup`, `SessionEnd`), Codex's full set including `PermissionRequest`, Gemini CLI's (`SessionStart`, `BeforeAgent`, `BeforeTool`, `AfterTool`, `AfterAgent`, `Notification`, `PreCompress`, `SessionEnd`), Hermes's lifecycle events, and Grok CLI's if it offers any. The set is taken from each installed harness version by probe.
- **Recording.** `polylogue hook EVENT --harness NAME` appends the payload, whole and with no size cap, as one line of the per-process spool file under `$XDG_STATE_HOME/polylogue/hooks/`, and syncs it (`fdatasync`) before returning. Recording opens no database and contacts nothing. A harness treats a non-zero hook exit as a refusal (Claude Code blocks the tool call on exit 2), so the hook always exits 0: an unwritable spool, an unreadable configuration file, a deleted working directory or an unknown `--harness` writes nothing to stdout and one line to stderr. A recording entry takes under 5 ms typically and 15 ms at p95 when the state directory is on a disk that is not saturated by bulk writes; a sync behind heavy writeback can take hundreds of milliseconds, so deployments keep the state directory off bulk storage.
- **The spool line** is the payload plus:

  ```json
  {"harness": "claude-code", "event": "PreToolUse", "at_ms": 1791588190000, "session_id": "…",
   "proc": {"harness_pid": 41237, "harness_start": 98213377, "boot_id": "7b1e…", "cwd": "/home/op/morrow"},
   "inherited": {"claude_code_session_id": "…"}}
  ```

  - `proc` is the **process identity**: `harness_pid` is the hook's parent (the harness runs the hook directly), `harness_start` the process start time, `boot_id` the kernel's boot ID. PID reuse and reboots never let a new process inherit an old session.
  - `inherited` records another harness's session variables present in the environment (a Codex run launched from a Claude Code shell). It is launch evidence for delegation, never the child's identity. No other environment variable is read or stored.
  - At session start the line carries `git_identity` (`common_dir`, `root_commits`, `head`, `branch`) and `repository`, read by common's `git-identity` within a 50 ms budget: the roots are every root reachable from `HEAD`, sorted, shallow-clone boundaries left out (the set Polylogue's repository lookup and Sinex's shell-session capture also record), and `repository` is the `origin` URL with any user and password removed. A shallow clone whose history reaches no root, or a missed budget, records the reason in `native_unavailable` instead.
  - Hermes's `conversation_history` is left out of the line, because its transcript is read from `state.db`.
- **A hook event belongs to its session and never creates one.** The daemon reads spool files like any growing file; a hook line keeps its identity across re-storage.
- **Hooks complete transcripts** where the transcript is short of the truth: `PreCompact` gives the pre-compaction context, `PostToolUse` restores a truncated tool result (the order is inline, then sidecar, then hook), `UserPromptSubmit` gives exact paste bodies.

## Tool databases

- **A tool's SQLite database is captured as committed rows, never as a database image.** Each capture keeps rows the tool later deleted (a row seen once stays in the archive) and writes no export side file. Uncommitted write-ahead-log rows are not read.
- Captured: Codex `state`, `goals`, `memories`, `queue`; Hermes `state.db`, `verification_evidence.db` and each profile's `state.db`; Antigravity `conversations/*.db`, `conversation_summaries.db` and each `state.vscdb`; Grok `memory-v2/**/*.sqlite` and `worktrees.db`; `llm` `logs.db`.
- Each row-dump header keeps the table's SQL and, from the same read transaction, its column names and order, declared types, nullability, primary-key order, hidden or generated columns and `WITHOUT ROWID`. The schema tools report this separately from the JSON types seen in rows, and leave SQL and default expressions out ([archive.md](archive.md#schema-tools)).
- A table whose schema no parser knows keeps its raw rows readable and is reported as [format drift](archive.md#format-drift).

## Import

`polylogue import PATH... [--format NAME] [--explain] [--wait]` takes files, directories, ZIPs and tar, tgz and tar.zst archives, including nested containers and multi-part Takeouts (each part is its own container; members join by path). The CLI, MCP and HTTP all take host paths. Every member streams; there is no size limit.

- **Recognition by declared export structure, never by sniffing.** A package is recognized by its structure and its members are parsed by that package's origin; nothing is guessed across origins. A path that matches no declared structure is refused, with the structures tried, and is not stored: the command answers at once, so refusing loses nothing, while a stored unknown file would have no owner and no reader. Members of a recognized package that no parser reads (`chat.html`, `.dat` assets) are stored as package members.
- **Original containers are kept whole**, and their members are also stored decoded, addressed as `container!/member`. Members deduplicate across overlapping exports; containers do not.
- **Importing the same bytes again changes nothing.** A newer export merges by field ([sessions.md](sessions.md#one-conversation-through-several-routes)), so a field only the older export had survives.
- An encrypted or corrupt member refuses its whole container before anything is stored. So does a member whose decoded size is more than its container declared, a container that holds its own bytes (directly or through containers inside it), and a container whose declared decoded size does not fit in the free staging space; all are checked before decoding the bytes they describe. A container repeated at another path in the same import is kept as an original and expanded once.
- In an export, one element the parser cannot read (nested deeper than the 128 levels the JSON reader builds into a value, for example) is counted as unreadable on its own; the other elements still parse, and the export's bytes stay stored. A key an element repeats is counted as an unread field (`path (repeated key; earlier value)`): readers see only its last value.
- The answer lists `imported`, `already_stored` and `refused` (with the structures tried), and per package the counts of sessions, documents and assets. `--explain` classifies every member without storing anything.

| Structure | Recognized by | Origin |
|---|---|---|
| ChatGPT export, every vintage | `conversations.json` or `conversations-NNN.json` shards, with `chat.html`, `user.json`, asset members, `conversation_asset_file_names.json`, `library_files.json`, `codex.json` | `chatgpt`, `codex-cloud` |
| ChatGPT single conversation | the conversation API shape (`mapping`, `current_node`, `conversation_id`) | `chatgpt` |
| claude.ai export | `conversations.json`, `users.json`, `projects/`, `memories.json`, `design_chats/` | `claude-ai`, `claude-design` |
| claude.ai single conversation | the `chat_messages` and `uuid` shape | `claude-ai` |
| Grok account export | `prod-grok-backend.json`, shape-checked while streaming | `grok` |
| Google Takeout | `Takeout/My Activity/Gemini Apps/`; `Drive/Google AI Studio/*`; `Gemini/` (Gems); `NotebookLM/` | `gemini-app`; `ai-studio`, joined to Drive sessions by file ID; Gems as documents; NotebookLM stored without a parser. Other Takeout products are refused members, counted and not stored |
| AI Studio archive | prompt JSON members, extensionless ones included | `ai-studio` |
| OpenRouter chat-room export | its JSON shape | `openrouter` |
| OpenTelemetry GenAI | OTLP-JSON `resourceSpans` | `otel-genai` |
| OpenAI chat array | only with `--format openai-chat`, since the shape has no reliable structure | `openai-chat` |
| aider | `.aider.llm.history` beside `.aider.chat.history.md` | `aider`: the LLM history is the session, the Markdown a companion document |
| Polylogue export bundle | the bundle manifest ([reading.md](reading.md#writing-out)) | the bundle's own records |
| The predecessor's spools | hook spool NDJSON in every old layout, `browser_llm_session` v1 envelopes, `raw_sessions` blobs | the payload's origin |
| The predecessor's archive | its `source.db` with `raw_sessions` and their blob files, read without opening it for writing | each source's origin, with its original source facts and append lineage |
| A live-origin file outside its root | only with `--format NAME`, or when the path keeps the tool's own layout | that origin |

### An archive from before the baseline

`polylogue import --from-archive PATH [--dry-run]` imports an archive written by a Polylogue build before the version 1 baseline ([archive.md](archive.md#evolution)): the folder holding its `archive.db`, `blobs/` and `embeddings.db`. It is a declared older input format, imported once into a fresh archive, not migrated in place; opening such a folder as an archive refuses it and names this command.

- **The older archive is only read.** Its databases are opened read-only and immutable; one with a non-empty write-ahead log is copied with its log into the importing archive's folder and the copy is read. A running Polylogue that holds the older archive's lock refuses the import. Nothing in the older folder is written, renamed or deleted.
- **Stored bytes keep their history.** Every input keeps its ID and format, and every revision, extent and observation keeps its number, chain, class, path, machine, file facts, version and time, so bytes deleted at their source and every superseded revision stay readable. Each blob, inline or a file of zstd frames, is decoded and must hash to its name and length before it is stored; blobs no extent cites are stored too.
- **Authored state.** Notes and settings become assertions with their original IDs, authors, machines and times (`via: import`); a later value of a setting revises the earlier one. A `hidden` tag becomes a `hidden` session assertion, retaining its value as the reason. Deliveries with their events, and kept results, become the requests that wrote them, with their original IDs, times and outcomes. Vectors keep their text hash, key, chunk, values and time; a different vector already stored for the same text, key and chunk is refused, never replaced.
- **Every row lands or is refused.** The answer's `older_archive` gives, per table of the older archive, the rows found and how many were stored, already present, refused or kept, with every refusal and its reason: an input whose bytes do not verify or that an excision changed, a note kind or relation with no counterpart, a delivery no kept request wrote, chat actions, backfill jobs and excisions. Every row other than the stored bytes' is also kept verbatim as one imported input, `archive.db!/rows.jsonl`, so a refused row is still readable. A refused input is set aside whole in `archive.db!/refused-inputs.jsonl`: each of its input, revision, extent and observation rows and the rows and stored bytes of the blobs it cites, with its refusal reason.
- **Import before the first pass.** An input whose location the archive already watches as another input is refused, since one file would get two inputs. A daemon reads its sources in its first pass, so the import runs against a fresh archive whose daemon has not read any source yet; sources are read after the import, and the imported locations are then the same inputs.
- **Importing again changes nothing**: every row is already present.
- `--dry-run` reads and checks everything, including every blob's bytes, and writes nothing; rows already present are not told apart and count as stored, and refusals only the writer can decide (an input held differently, a conflicting assertion or request, a vector different from one held) are only known on import, as the answer says.
- **Counts follow what was stored.** A delivery is counted as stored when it is in the archive after the import, whichever of the requests carrying its record stored it.

Aider's LLM log holds snapshots of the context sent and the response returned, with no native session or message IDs: the first recorded request fixes a content-derived session ID, and equal sent prefixes share message IDs. Its times carry no zone, so they stay unknown with the original strings kept. The paired Markdown is a companion document.

Hand-made Markdown chat dumps are not a structure: a flattened transcript is a degraded form of a conversation that has a real source.

## AI Studio through Drive

AI Studio stores its prompts in Google Drive, which is a known location like a tool's home directory.

- The collector follows Drive's changes feed from a start page token, not folder relisting, and writes every Drive revision of each prompt, oldest first, into the spool.
- A fetched revision has immutable pending bytes until the archive accepts the exact complete body at that prompt's stable input path. The committed revision cursor advances after acceptance; in primary mode, producer preparation remains pending until confirmed application. A retry or restart republishes the same retained bytes. Saving acceptance precedes pending-body cleanup, which leaves the watched prompt present and creates no source-gone observation. The predecessor's download cursor retains its last local body without assuming a revision identity, obtains archive acceptance, then enumerates retained Drive revisions from the beginning under this rule.
- `driveDocument` and `driveImage` attachments are fetched by Drive ID as attachment bytes. Revision lineage and `branchParent` edges are recorded; a Drive mode that cannot give lineage fails closed.
- The collector is on whenever the user has authorized a token; the one-time OAuth consent is the only gate, because a known location is never dark behind a setting. Without a token, status shows "not authorized" with the command to run.
- A prompt deleted in Drive keeps its archived revisions and is marked gone.

## Source clocks

Parsers keep each source time with its quality, its precision and any interval end the source declares. Whole epoch seconds and offset timestamps without fractions have second precision; fractional timestamps, epoch milliseconds and nanoseconds keep their subsecond instants. A missing or unreadable time has no instant and unknown quality. Capture observations and file times keep their own quality and never fill in a missing message time.

Where a format records events rather than a session creation time, the earliest record or activity time is the session's start, and says so. An OpenTelemetry span event keeps its start and end as an interval; its output message uses the call's start and records that basis with both original fields in `otel_span_clock`. Input history without message times stays unknown.

Gemini activity times keep their original text and zone. Explicit offsets are resolved without the host's time zone; ambiguous zone abbreviations stay unknown. AI Studio chunk times belong to the chunks; the Drive listing's file time is file metadata, separate from message times.

## Capture status

`polylogue status` (CLI, MCP, HTTP and web) answers "is everything being captured?" in one place. Its capture section holds:
- **per origin:** state (`live`, `collector off`, `not authorized`, `no parser yet`), newest input time, input and session counts, refusals, and format drift;
- **per root:** the coverage classes with their reasons and counts;
- **hooks:** per harness and event, whether it is wired and whether events arrived recently;
- **extension:** the native-messaging host manifest present and pointing at this binary; per provider, the last capture and its state; running backfills;
- **Drive:** token state, last change token, pending revisions;
- **imports:** packages imported and paths refused by the last import;
- **field accounting:** `unused` paths per origin and reading version ([archive.md](archive.md#parsing)).

Its figures and verdicts use the shared status format of [command-surface](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/command-surface/README.md); the rest of status is in [operations.md](operations.md#status). The health report raises an item for every non-empty `unrecognized` class, every refusal, every unwired hook event, and every origin silent for longer than its own usual gap. Status is computed on request from stored facts; nothing is checked on a schedule.
