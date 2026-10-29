# Capture

Where Polylogue's inputs come from: the location model and every live origin, hooks, tool databases, manual import of exports, AI Studio through Drive, and capture status. What happens to the bytes once collected is in [archive.md](archive.md); the parsed model is in [sessions.md](sessions.md). The browser extension's capture and backfill are in [extension.md](extension.md). In primary mode capture is unchanged and every collected input also becomes a Sinex material ([primary-mode.md](primary-mode.md)).

## Principles

- **Every relevant origin, at full fidelity.** Every AI tool whose sessions the user has is captured, with every side file that carries conversation content, context or harness-side evidence, even before a parser for it exists, because harnesses prune their own files and a pruned file is lost. A degraded form (a flattened transcript, a page scrape where the provider's data is available) is never the stored record. Ruling, 2026-02-15: "better to capture way huge schemas rather than missing something, anything at all." (cc-queued:7ebe20ed-a8bf-4392-98ab-bce1ea619c70@2026-02-15T02:42:31.922Z).
- **Nothing disappears silently.** Every file under a declared root is collected, ignored with a recorded reason, or reported unrecognized. Ruling, 2026-07-31: "we dont want things we wrongfully classify to disappear, being silently skipped over forever. we want to account for things." (cc:38baa1de-9715-48fa-8175-f2a29d92800e:bc4b84cc-1c23-46df-bd01-dbb4219d5d47).
- **Fixed locations, strict recognition, fail closed.** Polylogue reads data where each tool writes it; there are no path settings and no custom roots. Input that does not match its declared shape is refused, never reinterpreted. Ruling, 2026-09-27: "fail early, fail close etc. reject input which is not correct" (cc:62be45f2-6c40-49ce-a88e-bc566948628c:75bf98d6-b1e7-461a-bd53-501ec33de178).
- **Live is the normal case.** A growing file is tailed; a live origin needs no import step.
- **Exports arrive by hand.** Provider exports are brought in with `polylogue import`; nothing is picked up from a download or inbox directory. Ruling, 2026-07-06: "we dont' import from stuff 'sitting in inbox'. this is just ephemeral dirty copy." (cc-queued:f7680f9c-32e4-4f2f-9180-b9977e1f4702@2026-07-06T17:12:07.017Z).
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
- Hidden directories are entered only when a location names them. Symlinked roots resolve once. An unreadable directory is a reported state, never an empty result. A `.git` directory inside a root is ignored with its reason.
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
| `sessions/**/rollout-*.jsonl` and `rollout-*.jsonl.zst`; `archived_sessions/**` | sessions. A compressed rollout is the same input as its plain form: identity is content and native ID, never the path |
| `history.jsonl`, `session_index.jsonl` | prompt events, titles |
| rows of the `state_*`, `goals_*`, `memories_*` and `queue_*` databases | goals, memories and queued prompts as events |
| `memories/**/*.md` (not `memories/.git`) | memory documents |
| `attachments/**`, `shell_snapshots/**`, `ambient-suggestions/**` | attachments, context documents |
| files named by `local_images` in a rollout line | attachment bytes, read when the line is observed ([sessions.md](sessions.md#attachments)) |

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
- **Recording.** `polylogue hook EVENT --harness NAME` appends the payload, whole and with no size cap, as one line of the per-process spool file, and syncs it (`fdatasync`) before returning. Recording opens no database and contacts nothing. An unwritable spool writes one line to stderr and exits 0, so a hook never breaks its harness. A recording entry takes under 5 ms typically and 15 ms at p95.
- **The spool line** is the payload plus:

  ```json
  {"harness": "claude-code", "event": "PreToolUse", "at_ms": 1791588190000, "session_id": "…",
   "proc": {"harness_pid": 41237, "harness_start": 98213377, "boot_id": "7b1e…", "cwd": "/home/op/morrow"},
   "inherited": {"claude_code_session_id": "…"}}
  ```

  - `proc` is the **process identity**: `harness_pid` is the hook's parent (the harness runs the hook directly), `harness_start` the process start time, `boot_id` the kernel's boot ID. PID reuse and reboots never let a new process inherit an old session.
  - `inherited` records another harness's session variables present in the environment (a Codex run launched from a Claude Code shell). It is launch evidence for delegation, never the child's identity. The native terminal adapter variables described below are also recorded when present.
  - Hermes's `conversation_history` is left out of the line, because its transcript is read from `state.db`.
- **Native work context.** At session start, the hook observes Git's common directory, root commits, current revision and branch, and the repository named in `remote.origin.url`, using one shared 50 ms budget. An unavailable or timed-out observation is recorded as unknown with its reason. A source cwd selects where Git observes these facts; it does not identify the repository. When Kitty supplies its native process and window variables, the hook records that adapter key with host, boot and parent-process identity. Structured task and pull-request references supplied by a harness remain in the payload; prompts, goal prose and shell commands do not supply objective identities.
- **A hook event belongs to its session and never creates one.** The daemon reads spool files like any growing file; a hook line keeps its identity across re-storage.
- **Hooks complete transcripts** where the transcript is short of the truth: `PreCompact` gives the pre-compaction context, `PostToolUse` restores a truncated tool result (the order is inline, then sidecar, then hook), `UserPromptSubmit` gives exact paste bodies.

## Tool databases

- **A tool's SQLite database is captured as committed rows, never as a database image.** Each capture keeps rows the tool later deleted (a row seen once stays in the archive) and writes no export side file. Uncommitted write-ahead-log rows are not read.
- Captured: Codex `state`, `goals`, `memories`, `queue`; Hermes `state.db`, `verification_evidence.db` and each profile's `state.db`; Antigravity `conversations/*.db`, `conversation_summaries.db` and each `state.vscdb`; Grok `memory-v2/**/*.sqlite` and `worktrees.db`; `llm` `logs.db`.
- Row-dump headers retain the original SQL and structured native column metadata from the same SQLite read transaction: column names and order, declared type, nullability, primary-key order, hidden/generated status and `WITHOUT ROWID`. Schema tools report this metadata separately from observed JSON types and omit SQL and default expressions. A historical SQL-only header establishes table presence; its unobserved columns remain unknown.
- A table whose schema no parser knows keeps its raw rows readable and is reported as [format drift](archive.md#format-drift).

## Import

`polylogue import PATH... [--format NAME] [--explain] [--wait]` takes files, directories, ZIPs and tar, tgz and tar.zst archives, including nested containers and multi-part Takeouts (each part is its own container; members join by path). The CLI, MCP and HTTP all take host paths. Every member streams; there is no size limit.

- **Recognition by declared export structure, never by sniffing.** A package is recognized by its structure and its members are parsed by that package's origin; nothing is guessed across origins. A path that matches no declared structure is refused, with the structures tried, and is not stored: the command answers at once, so refusing loses nothing, while a stored unknown file would have no owner and no reader. Members of a recognized package that no parser reads (`chat.html`, `.dat` assets) are stored as package members.
- **Original containers are kept whole**, and their members are also stored decoded, addressed as `container!/member`. Members deduplicate across overlapping exports; containers do not.
- **Importing the same bytes again changes nothing.** A newer export merges by field ([sessions.md](sessions.md#one-conversation-through-several-routes)), so a field only the older export had survives.
- An encrypted or corrupt member refuses its whole container before anything is stored.
- Container observations record the container input, encoding kind, member index and exact member-name bytes. Display names are escaped when the original name is not UTF-8. Container membership is independent of explicit import membership.
- Codex plain and zstd rollout copies use the recorded native thread identity. Captured compressed bytes and decoded member bytes remain separate; re-encoding equal decoded content adds an observation without a logical revision. Appends preserve the existing byte chain.
- Native Codex local-image references are read before parsing the observed rollout. Relative names require a recorded absolute working directory. A stored coverage manifest records each reference, its source record range, the recorded directory, and collected, missing, unreadable or unresolved status. Available files retain their exact bytes as content-addressed inputs.
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
| A live-origin file outside its root | only with `--format NAME`, or when the path keeps the tool's own layout | that origin |

Aider's LLM log contains snapshots of the sent context and the returned response. It has no native session or message IDs: the first recorded request fixes its content-derived session identity, and equal sent prefixes share message IDs. Its unzoned log times stay unknown, with the original strings retained. The paired Markdown is a companion document.

Hand-made Markdown chat dumps are not a structure: a flattened transcript is a degraded form of a conversation that has a real source.

## AI Studio through Drive

AI Studio stores its prompts in Google Drive, which is a known location like a tool's home directory.

- The collector follows Drive's changes feed from a start page token, not folder relisting, and writes every Drive revision of each prompt, oldest first, into the spool.
- `driveDocument` and `driveImage` attachments are fetched by Drive ID as attachment bytes. Revision lineage and `branchParent` edges are recorded; a Drive mode that cannot give lineage fails closed.
- The collector is on whenever the user has authorized a token; the one-time OAuth consent is the only gate, because a known location is never dark behind a setting. Without a token, status shows "not authorized" with the command to run.
- A prompt deleted in Drive keeps its archived revisions and is marked gone.

## Capture status

`polylogue status` (CLI, MCP, HTTP and web) answers "is everything being captured?" in one place. Its capture section holds:
- **per origin:** state (`live`, `collector off`, `not authorized`, `no parser yet`), newest input time, input and session counts, refusals, and format drift;
- **per root:** the coverage classes with their reasons and counts;
- **hooks:** per harness and event, whether it is wired and whether events arrived recently;
- **extension:** the native-messaging host manifest present and pointing at this binary; per provider, the last capture and its state; running backfills;
- **Drive:** token state, last change token, pending revisions;
- **imports:** packages imported and paths refused by the last import;
- **field accounting:** `unused` paths per origin and reading version ([archive.md](archive.md#parsing)).

Its figures and verdicts use the shared status format of [command-surface](https://github.com/Sinity/polylogue-sinex-common/blob/master/docs/command-surface.md); the rest of status is in [operations.md](operations.md#status). The health report raises an item for every non-empty `unrecognized` class, every refusal, every unwired hook event, and every origin silent for longer than its own usual gap. Status is computed on request from stored facts; nothing is checked on a schedule.

## Laws

- **classes-partition-files**: a synthetic root with one file per class reports each as `collected`, `ignored` (with its reason) or `unrecognized`, and the classes partition the files.
- **rename-moves-class**: renaming a collected file to an unknown extension moves it to `unrecognized` at the next discovery event.
- **collect-without-parser**: a declared origin with no parser stores its files and reports `no_parser_yet`.
- **refused-stays-refused**: a file failing its origin's shape check is `refused` with a reason and appears under no other origin.
- **known-locations-fully-recognized**: a fixture tree holding every listed location of each live origin has no `unrecognized` file.
- **root-follows-tool-variable**: setting the tool's home variable moves the root; no Polylogue setting does.
- **hook-event-recorded**: a synthetic event of each type lands in the spool and in the index.
- **hook-never-creates-session**: a hook event for an unknown session creates no session.
- **hook-line-durable**: a hook line survives a crash after the hook returns.
- **drive-revisions**: against a fake Drive, three revisions and one `driveImage` give one session with three revisions and the image's bytes; a deleted prompt keeps its revisions and is marked gone.
- **import-every-structure**: a fixture for every row of the structures table imports; an unrelated JSONL file is refused with the structures tried and nothing is stored.
- **reimport-changes-nothing**: importing identical bytes again changes nothing, and a byte-identical copy of a container is `already_stored`.
- **newer-export-keeps-dropped-field**: a newer export that lacks a field keeps the older export's value.
- **corrupt-member-refuses-container**: a corrupt member refuses its container before anything is stored.
- **container-streams**: every member of a container far larger than memory streams with bounded memory.
- **tool-database-keeps-deleted-rows**: a row deleted by the tool after one capture stays in the archive, and no database image or side file is written.

A completed full discovery walk publishes one capture-coverage snapshot per machine. Every declared root is present, including accessible empty roots, absent roots, unavailable roots and roots with unreadable descendants. Entries aggregate file counts, byte counts and newest source mtime by collection class and declared pattern; ignored entries retain their reason. Status reads the last confirmed full-walk snapshot and its observation time. Incremental file events update inputs without replacing that snapshot.

Coverage uses one unrecognized aggregate per root, independent of how many unknown subdirectories exist. Child symbolic links and special filesystem entries are ignored with explicit reasons; discovery does not follow them during either full walks or incremental events.

Python predecessor archives containing `source.db`, `raw_sessions` and their declared blob files are supported external captured-data packages. Import reads their metadata without opening the archive for writing, validates exact source bytes and explicit append lineage, and preserves the original source facts. This is distinct from retired Rust archive storage layouts, whose codecs belong only to the private, explicit storage converter. Neither form authorizes a live archive conversion or reset.
